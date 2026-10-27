"""Controller behavior against simulated clocks and subprocess completion."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from research.comparative_ablation_20261025_v1 import controller, run_one, runtime


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_atomic_retries_transient_windows_replace_without_losing_previous_state(tmp_path, monkeypatch, winerror):
    path = tmp_path / 'PROGRESS.json'
    path.write_text('{"status": "old"}', encoding='utf8')
    replacement = dict(status='new')
    original_replace = Path.replace
    attempts, delays = [], []
    error = PermissionError('simulated Windows file lock')
    error.winerror = winerror

    def replace(temporary, destination):
        assert destination == path
        assert json.loads(path.read_text(encoding='utf8')) == dict(status='old')
        assert json.loads(temporary.read_text(encoding='utf8')) == replacement
        attempts.append(temporary)
        if len(attempts) <= 2:
            raise error
        return original_replace(temporary, destination)

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=delays.append))
    controller.atomic(path, replacement)
    assert len(attempts) == 3
    assert delays == [.1, .2]
    assert json.loads(path.read_text(encoding='utf8')) == replacement
    assert not path.with_suffix('.tmp').exists()


def test_atomic_exhausted_retry_preserves_old_state_and_propagates_error(tmp_path, monkeypatch):
    path = tmp_path / 'PROGRESS.json'
    path.write_text('{"status": "old"}', encoding='utf8')
    attempts, delays = [], []
    error = PermissionError('persistent Windows access denied')
    error.winerror = 5

    def replace(temporary, destination):
        attempts.append(temporary)
        assert path.read_text(encoding='utf8') == '{"status": "old"}'
        raise error

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=delays.append))
    with pytest.raises(PermissionError) as raised:
        controller.atomic(path, dict(status='new'))
    assert raised.value is error
    assert len(attempts) == 6
    assert delays == [.1, .2, .4, .8, 1.6]
    assert path.read_text(encoding='utf8') == '{"status": "old"}'
    assert json.loads(path.with_suffix('.tmp').read_text(encoding='utf8')) == dict(status='new')


@pytest.mark.parametrize('error', [OSError('unrelated I/O error'), PermissionError('non-Windows denial')])
def test_atomic_does_not_retry_unrelated_errors(tmp_path, monkeypatch, error):
    path = tmp_path / 'PROGRESS.json'
    path.write_text('{"status": "old"}', encoding='utf8')
    attempts, delays = [], []

    def replace(temporary, destination):
        attempts.append(temporary)
        raise error

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(controller, 'time', SimpleNamespace(sleep=delays.append))
    with pytest.raises(type(error)) as raised:
        controller.atomic(path, dict(status='new'))
    assert raised.value is error
    assert len(attempts) == 1
    assert delays == []
    assert path.read_text(encoding='utf8') == '{"status": "old"}'


def make_job(method='A1', seed=92601, dataset='jena', *, extended=False):
    k = runtime.KS[dataset]
    return dict(job_id=f'CA_{method}_{dataset}_K{k}_s{seed}' + ('_ext10' if extended else ''),
                method_id=method, dataset=dataset, K=k, seed=seed, extended=extended)


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    report, cache = tmp_path / 'reports', tmp_path / 'cache'
    report.mkdir()
    cache.mkdir()
    monkeypatch.setattr(runtime, 'REPORT', report)
    monkeypatch.setattr(runtime, 'CACHE', cache)
    monkeypatch.setattr(runtime, 'REPO', tmp_path)
    clock = [1000.]
    processes, launched, report_calls, bundle_calls = [], [], [], []
    outcome = lambda job, attempt: (0, 1., .25)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')

    def create_result(job, selected, seconds):
        path = report / 'jobs' / job['job_id'] / 'RESULT.json'
        selection = path.with_name('SELECTION.json')
        save(selection, dict(status='SEALED', job=job, V_used=False, selected_epoch=selected))
        save(path, dict(status='COMPLETE', job=job, elapsed_seconds=seconds, selection_path=str(selection),
                        selection_sha256=runtime.sha(selection),
                        row=dict(dataset=job['dataset'], method_id=job['method_id'], seed=job['seed'],
                                 selected_epoch=selected, MSE=float(92611 - job['seed']))))

    class Process:
        def __init__(self, job, attempt):
            self.job, self.attempt, self.done = job, attempt, False
            self.pid = 10000 + len(processes)

        def poll(self):
            if not self.done:
                self.code, seconds, selected = sim.outcome(self.job, self.attempt)
                clock[0] += seconds
                if self.code == 0:
                    create_result(self.job, selected, seconds)
                self.done = True
            return self.code

    def popen(command, **kwargs):
        job = runtime.read(Path(command[command.index('--job') + 1]))
        attempt = int(command[command.index('--attempt') + 1])
        process = Process(job, attempt)
        process.concurrent_at_start = 1 + sum(not p.done for p in processes)
        processes.append(process)
        launched.append((job, attempt))
        return process

    def render(directory, results, status):
        report_calls.append((results, dict(status)))
        if sim.render_failure:
            sim.render_failure = False
            raise RuntimeError('simulated postprocessing interruption')
        (Path(directory) / 'fixture_report.txt').write_text('simulated report\n', encoding='utf8')

    def bundle(**kwargs):
        bundle_calls.append(kwargs)
        receipt = dict(path=str(report / 'fake_bundle.zip'), zip_crc='PASS')
        save(report / 'BUNDLE_RECEIPT.json', receipt)
        return receipt

    def configure(jobs, extensions=(), *, limit=50000., state=None):
        methods = ('C1', 'C2', 'C3', 'A1', 'A2', 'A3', 'A4', 'A5')
        plan = dict(jobs=list(jobs), extension_jobs=list(extensions), limit_seconds=limit,
                    latest_start=datetime.fromtimestamp(clock[0] + 100000, timezone.utc).isoformat(),
                    budget=dict(profiles={dataset: dict(search_cap3_max_seconds=1., refit_per_epoch_max_seconds=1.)
                                          for dataset in runtime.KS},
                                per_job_expected_seconds={method + '/' + dataset: 1.
                                                          for method in methods for dataset in runtime.KS}))
        save(report / 'RUN_PLAN.json', plan)
        save(report / 'PREREGISTRATION_COMMIT.json', {})
        save(report / 'G1_REUSE_AUDIT.json', dict(runs=[]))
        if state is not None:
            initial = dict(status='RUNNING', run_statuses={}, extension_status={},
                           stop_reason=None, elapsed_seconds=0.)
            initial.update(state)
            save(report / 'PROGRESS.json', initial)
        return plan

    monkeypatch.setattr(controller, 'time', SimpleNamespace(time=lambda: clock[0], sleep=lambda seconds: None))
    monkeypatch.setattr(runtime, 'guard_preserved', lambda: dict(checked=1, changed=[]))
    monkeypatch.setattr(controller.subprocess, 'Popen', popen)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='1000'))
    monkeypatch.setattr(run_one, 'run_job', lambda job: runtime.read(report / 'jobs' / job['job_id'] / 'RESULT.json'))
    package = 'research.comparative_ablation_20261025_v1.'
    monkeypatch.setitem(sys.modules, package + 'analysis',
                        SimpleNamespace(report=render, GENERATED_FILES=('fixture_report.txt',)))
    monkeypatch.setitem(sys.modules, package + 'bundle', SimpleNamespace(build_bundle=bundle))
    sim = SimpleNamespace(report=report, cache=cache, clock=clock, processes=processes, launched=launched,
                          report_calls=report_calls, bundle_calls=bundle_calls, outcome=outcome,
                          configure=configure, save=save, render_failure=False)
    return sim


def test_resume_does_not_restart_family_with_three_failed_jobs(simulation):
    sim = simulation
    jobs = [make_job(seed=seed) for seed in range(92601, 92605)]
    sim.configure(jobs, state=dict(run_statuses={job['job_id']: 'FAILED' for job in jobs[:3]}))
    controller.main()
    assert sim.launched == []
    progress = runtime.read(sim.report / 'PROGRESS.json')
    assert progress['run_statuses'][jobs[3]['job_id']] == 'NOT_RUN_FAMILY_FAILURE'


def test_one_retry_per_job_and_three_failures_stop_family(simulation):
    sim = simulation
    jobs = [make_job(seed=seed) for seed in range(92601, 92605)]
    sim.configure(jobs)
    sim.outcome = lambda job, attempt: (1, 1., .25)
    controller.main()
    assert [(job['seed'], attempt) for job, attempt in sim.launched] == [
        (seed, attempt) for seed in range(92601, 92604) for attempt in (1, 2)]
    statuses = runtime.read(sim.report / 'PROGRESS.json')['run_statuses']
    assert all(statuses[job['job_id']] == 'FAILED' for job in jobs[:3])
    assert statuses[jobs[3]['job_id']] == 'NOT_RUN_FAMILY_FAILURE'


def test_successful_retry_completes_one_logical_job(simulation):
    sim = simulation
    selected = make_job()
    sim.configure([selected])
    sim.outcome = lambda job, attempt: (1 if attempt == 1 else 0, 1., .25)
    controller.main()
    assert [attempt for _, attempt in sim.launched] == [1, 2]
    assert runtime.read(sim.report / 'PROGRESS.json')['run_statuses'][selected['job_id']] == 'COMPLETE'


def test_required_C2_probe_is_one_solo_then_one_pair(simulation):
    sim = simulation
    jobs = [make_job('C2', dataset=d) for d in ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')]
    sim.configure(jobs)
    controller.main()
    assert [p.concurrent_at_start for p in sim.processes] == [1, 1, 2, 1]
    probe = runtime.read(sim.report / 'CONCURRENCY_PROBE.json')
    assert probe['jobs'] == [jobs[1]['job_id'], jobs[2]['job_id']]
    assert probe['subsequent_concurrency'] == 1


def test_fatal_stop_drains_existing_pair_without_starting_retry(simulation, monkeypatch):
    sim = simulation
    jobs = [make_job('C2', dataset=d) for d in ('jena', 'ETTh1', 'ETTh2_safe17400')]
    sim.configure(jobs)
    sim.outcome = lambda job, attempt: (1 if job['dataset'] == 'ETTh1' else 0, 1., .25)
    tripped = [False]

    def guard():
        if not tripped[0] and any(p.job['dataset'] == 'ETTh1' and not p.done for p in sim.processes):
            tripped[0] = True
            raise ValueError('simulated immutable source gate failure')
        return dict(checked=1, changed=[])

    monkeypatch.setattr(runtime, 'guard_preserved', guard)
    controller.main()
    assert tripped[0]
    assert [(job['dataset'], attempt) for job, attempt in sim.launched] == [('jena', 1), ('ETTh1', 1)]
    assert runtime.read(sim.report / 'PROGRESS.json')['status'] == 'STOPPED'
    assert sim.bundle_calls


def test_first_failure_after_budget_is_not_reported_as_two_failures(simulation):
    sim = simulation
    selected = make_job()
    sim.configure([selected], limit=10.)
    sim.outcome = lambda job, attempt: (1, 11., .25)
    controller.main()
    assert len(sim.launched) == 1
    status = runtime.read(sim.report / 'PROGRESS.json')['run_statuses'][selected['job_id']]
    assert status not in ('FAILED', 'COMPLETE')
    assert 'BUDGET' in status


@pytest.mark.parametrize('cap_count', [4, 5])
def test_extension_requires_five_caps_and_waits_until_all_base_jobs_end(simulation, cap_count):
    sim = simulation
    jobs = [make_job(seed=seed) for seed in range(92601, 92611)]
    extensions = [make_job(seed=seed, extended=True) for seed in range(92601, 92611)]
    sim.configure(jobs, extensions)

    def outcome(job, attempt):
        if job['extended']:
            assert all((sim.report / 'jobs' / base['job_id'] / 'RESULT.json').is_file() for base in jobs)
            selected = 10. if job['seed'] < 92606 else 1.
        else:
            selected = 3. if job['seed'] < 92601 + cap_count else .25
        return 0, 1., selected

    sim.outcome = outcome
    controller.main()
    extended = [job for job, _ in sim.launched if job['extended']]
    assert len(extended) == (10 if cap_count == 5 else 0)
    status = runtime.read(sim.report / 'PROGRESS.json')['extension_status']['A1/jena']
    assert status['status'] == ('UNDERTRAINED_AT_10' if cap_count == 5 else 'NOT_TRIGGERED')


def test_first_measured_duration_drops_lower_priority_family(simulation):
    sim = simulation
    jobs = [make_job(seed=s) for s in range(92601, 92605)]
    jobs += [make_job('A5', seed=s) for s in range(92601, 92605)]
    sim.configure(jobs, limit=300.)
    sim.outcome = lambda job, attempt: (0, 120., .25)
    controller.main()
    assert all(job['method_id'] == 'A1' for job, _ in sim.launched)
    statuses = runtime.read(sim.report / 'PROGRESS.json')['run_statuses']
    assert all(statuses[job['job_id']] == 'NOT_RUN_BUDGET' for job in jobs if job['method_id'] == 'A5')


def test_deadline_is_rechecked_after_prelaunch_hashing(simulation, monkeypatch):
    sim = simulation
    selected = make_job()
    sim.configure([selected], limit=10.)

    def guard():
        if (sim.report / 'EXECUTION_CLOCK.json').exists() and not sim.launched:
            sim.clock[0] += 11.
        return dict(checked=1, changed=[])

    monkeypatch.setattr(runtime, 'guard_preserved', guard)
    controller.main()
    assert not sim.launched
    assert runtime.read(sim.report / 'PROGRESS.json')['run_statuses'][selected['job_id']] == 'NOT_RUN_BUDGET'


def test_completed_jobs_can_resume_interrupted_finalization(simulation):
    sim = simulation
    sim.configure([make_job()])
    sim.render_failure = True
    with pytest.raises(RuntimeError, match='postprocessing interruption'):
        controller.main()
    assert (sim.report / 'EXECUTION_COMPLETION.json').is_file()
    controller.main()
    assert len(sim.launched) == 1
    assert len(sim.report_calls) == 2
    assert len(sim.bundle_calls) == 1


def test_stopped_completion_requires_explicit_resolution(simulation):
    sim = simulation
    sim.configure([make_job()])
    sim.save(sim.report / 'EXECUTION_COMPLETION.json', dict(status='STOPPED'))
    with pytest.raises(RuntimeError, match='prior gate stop requires explicit resolution'):
        controller.main()
    assert sim.launched == []
    assert sim.report_calls == []
    assert sim.bundle_calls == []
