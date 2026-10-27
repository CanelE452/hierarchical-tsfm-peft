"""Real filesystem receipts and synthetic processes; never runs model training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from research.comparative_ablation_followup_20261027_v1 import common, controller, job_worker


def make_job(method='A4', seed=92601, dataset='jena', extended=False):
    k = dict(jena=6, ETTh1=4, ETTh2_safe17400=4, appliances=10)[dataset]
    return dict(job_id=f'CA_{method}_{dataset}_K{k}_s{seed}' + ('_ext10' if extended else ''),
                method_id=method, dataset=dataset, K=k, seed=seed, extended=extended)


def access_proof(dataset):
    if dataset == 'jena':
        return dict(new_T_numeric_values_parsed=0, members=[dict(member=name, tail_numeric_values_parsed=0)
            for name in ('x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy')])
    return dict(T_numeric_values_parsed=0)


@pytest.fixture
def sim(tmp_path, monkeypatch):
    paths = dict(REPO=tmp_path, REPORT=tmp_path / 'reports/new', CACHE=tmp_path / 'cache/new',
                 OLD_REPORT=tmp_path / 'reports/old', OLD_CACHE=tmp_path / 'cache/old')
    for name, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(common, name, value)
    monkeypatch.setattr(common, 'PYTHON', Path(sys.executable))
    monkeypatch.setattr(common, 'install_write_guard', lambda: None)
    monkeypatch.setattr(common, 'guard_preserved', lambda: dict(status='PASS'))
    monkeypatch.setattr(common, 'validate_admission', lambda: dict(status='PASS'))
    s = SimpleNamespace(clock=[1000.], jobs=[], launched=[], processes=[], verified=[], reports=[],
                        report_codes={}, report_scientific=False, report_polls=2,
                        outcome=lambda job, attempt: (0, 1., .25))

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')

    def create_result(job, selected=.25, seconds=1.):
        save(common.OLD_REPORT / 'jobs' / job['job_id'] / 'RESULT.json', dict(status='COMPLETE', job=job,
            elapsed_seconds=seconds, row=dict(selected_epoch=selected), task_access=access_proof(job['dataset'])))

    def configure(jobs=None, *, limit=5000., cutoff=100000., extension_seconds=20.):
        s.jobs = list(jobs if jobs is not None else [make_job()])
        monkeypatch.setattr(common, 'LIMIT_SECONDS', limit)
        monkeypatch.setattr(common, 'LAST_START', datetime.fromtimestamp(s.clock[0] + cutoff, timezone.utc).isoformat())
        save(common.REPORT / 'BUDGET.json', dict(per_dataset_a4_seconds={d: 1. for d in common.DATASETS},
            a1_jena_extension_seconds_per_job=extension_seconds))
        excluded = [make_job('A5'), make_job('A3'), make_job('C2', extended=True), make_job('A4', extended=True)]
        save(common.OLD_REPORT / 'RUN_PLAN.json', dict(jobs=[j for j in s.jobs + excluded if not j['extended']],
                                                     extension_jobs=[j for j in s.jobs + excluded if j['extended']]))
        save(common.OLD_REPORT / 'EXECUTION_COMPLETION.json', dict(status='PARTIAL',
            run_statuses={'old_complete': 'COMPLETE'}, extension_status={}))

    class Process:
        def __init__(self, job=None, attempt=None, stage=None):
            self.job, self.attempt, self.stage = job, attempt, stage
            self.pid, self.calls, self.done = 10000 + len(s.processes), 0, False
            self.concurrent_training = sum(p.job is not None and not p.done for p in s.processes)
            self.interim_active_at_start = any(p.stage == 'interim' and not p.done for p in s.processes)
            s.processes.append(self)

        def poll(self):
            if self.done:
                return self.code
            self.calls += 1
            if self.stage is not None:
                if self.calls < s.report_polls:
                    return None
                self.code = s.report_codes.get(self.stage, 0)
                if self.code and s.report_scientific:
                    output = common.REPORT / 'interim_A4' if self.stage == 'interim' else common.REPORT
                    save(output / 'REPORTING_FAILURE.json', dict(classification='SCIENTIFIC_GATE', error='floor mismatch'))
            else:
                self.code, seconds, selected = s.outcome(self.job, self.attempt)
                s.clock[0] += seconds
                if self.code == 0:
                    create_result(self.job, selected, seconds)
                save(common.CACHE / 'worker_receipts' / f'{self.job["job_id"]}_attempt{self.attempt}.json',
                     dict(status='COMPLETE' if self.code == 0 else 'FAILED', exit_code=self.code,
                          wall_seconds=seconds))
            self.done = True
            return self.code

    def popen(args, **kwargs):
        assert '-B' in args and kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        if '--stage' in args or args[-1].endswith('.bundle'):
            stage = args[args.index('--stage') + 1] if '--stage' in args else 'bundle'
            s.reports.append((stage, len(s.launched)))
            return Process(stage=stage)
        assert '--deadline' not in args
        job = common.read(Path(args[args.index('--job') + 1]))
        attempt = int(args[args.index('--attempt') + 1])
        if not (common.REPORT / 'EXECUTION_CLOCK.json').exists():
            save(common.REPORT / 'EXECUTION_CLOCK.json', dict(training_started_unix=s.clock[0],
                limit_seconds=common.LIMIT_SECONDS, latest_start=common.LAST_START))
        process = Process(job=job, attempt=attempt)
        save(common.CACHE / 'worker_admissions' / f'{job["job_id"]}_attempt{attempt}.json',
             dict(job=job, attempt=attempt, admitted_unix=s.clock[0], pid=process.pid))
        s.launched.append((job, attempt))
        return process

    def run(args, **kwargs):
        assert '--verify-only' in args
        job = common.read(Path(args[args.index('--job') + 1]))
        assert (common.OLD_REPORT / 'jobs' / job['job_id'] / 'RESULT.json').is_file()
        s.verified.append(job['job_id'])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common, 'selected_jobs', lambda: list(s.jobs))
    monkeypatch.setattr(controller, 'time', SimpleNamespace(time=lambda: s.clock[0], sleep=lambda delay: s.clock.__setitem__(0, s.clock[0] + delay)))
    monkeypatch.setattr(job_worker, 'time', controller.time)
    monkeypatch.setattr(controller.subprocess, 'Popen', popen)
    monkeypatch.setattr(controller.subprocess, 'run', run)
    monkeypatch.setattr(controller, 'pid_alive', lambda pid: False)
    s.configure, s.save, s.result = configure, save, create_result
    s.configure()
    return s


def test_serial_order_retry_and_three_failures_stop_family(sim):
    sim.configure([make_job(seed=s) for s in range(92601, 92605)])
    sim.outcome = lambda job, attempt: (1, 1., .25)
    assert controller.main([]) == 0
    assert [(j['seed'], a) for j, a in sim.launched] == [(seed, a) for seed in range(92601, 92604) for a in (1, 2)]
    assert all(p.concurrent_training == 0 for p in sim.processes if p.job)
    state = common.read(common.REPORT / 'PROGRESS.json')
    assert state['run_statuses'][sim.jobs[-1]['job_id']] == 'NOT_RUN_FAMILY_FAILURE'


def test_successful_retry_and_scope_are_distinct(sim):
    prior_complete = make_job('C1')
    plan = common.read(common.OLD_REPORT / 'RUN_PLAN.json')
    plan['jobs'].append(prior_complete)
    sim.save(common.OLD_REPORT / 'RUN_PLAN.json', plan)
    old = common.read(common.OLD_REPORT / 'EXECUTION_COMPLETION.json')
    old['run_statuses'][prior_complete['job_id']] = 'COMPLETE'
    sim.save(common.OLD_REPORT / 'EXECUTION_COMPLETION.json', old)
    sim.outcome = lambda job, attempt: (1 if attempt == 1 else 0, 1., .25)
    assert controller.main([]) == 0
    assert [a for _, a in sim.launched] == [1, 2]
    state = common.read(common.REPORT / 'PROGRESS.json')
    assert state['run_statuses'][sim.jobs[0]['job_id']] == 'COMPLETE'
    assert state['prior_run_statuses']['old_complete'] == 'COMPLETE'
    assert prior_complete['job_id'] not in state['run_statuses']
    assert prior_complete['job_id'] in state['prior_complete_job_ids']
    assert state['run_statuses'][make_job('A5')['job_id']] == 'NOT_RUN_SCOPE'
    assert state['extension_status']['C2/jena']['status'] == 'NOT_RUN_SCOPE'


@pytest.mark.parametrize('limit,expected', [(200., False), (203., True)])
def test_all_ten_extension_jobs_must_fit_before_first_admission(sim, limit, expected):
    extensions = [make_job('A1', seed=s, extended=True) for s in range(92601, 92611)]
    sim.configure([make_job(), *extensions], limit=limit)
    assert controller.main([]) == 0
    assert len([j for j, _ in sim.launched if j['extended']]) == (10 if expected else 0)
    receipt = common.read(common.REPORT / 'A1_JENA_EXTENSION_ADMISSION.json')
    assert receipt['n'] == 10 and receipt['required_seconds'] == 200.
    assert receipt['allowed'] == expected
    if not expected:
        state = common.read(common.REPORT / 'PROGRESS.json')
        assert all(state['run_statuses'][job['job_id']] == 'NOT_RUN_BUDGET' for job in extensions)


def test_interim_starts_after_a4_and_does_not_block_extension(sim):
    sim.report_polls = 5
    sim.configure([make_job(), *[make_job('A1', seed=s, extended=True) for s in range(92601, 92611)]])
    controller.main([])
    assert sim.reports[0] == ('interim', 1)
    assert any(p.job and p.job['extended'] and p.interim_active_at_start for p in sim.processes)
    assert sim.reports[-2:] == [('final', 11), ('bundle', 11)]


def test_unstarted_extension_rechecks_budget_after_resume_and_preserves_decisions(sim):
    extensions = [make_job('A1', seed=s, extended=True) for s in range(92601, 92611)]
    sim.configure([make_job(), *extensions], limit=5000.)
    sim.save(common.REPORT / 'EXECUTION_CLOCK.json', dict(training_started_unix=sim.clock[0],
        limit_seconds=common.LIMIT_SECONDS, latest_start=common.LAST_START))
    runner = controller.Controller()
    assert runner.extension_allowed(extensions)
    path = common.REPORT / 'A1_JENA_EXTENSION_ADMISSION.json'
    original = path.read_bytes()
    sim.clock[0] += 4801.
    resumed = controller.Controller()
    assert not resumed.extension_allowed(extensions)
    assert path.read_bytes() == original
    update = common.read(common.REPORT / 'A1_JENA_EXTENSION_ADMISSION_002.json')
    assert update['required_seconds'] == 200. and update['remaining_seconds'] == 199.
    assert update['previous']['sha256'] == common.sha(path)
    assert not resumed.start(extensions[0], 1)
    assert not sim.launched


@pytest.mark.parametrize('evidence', ['worker_admission', 'complete'])
def test_started_extension_keeps_original_clock_and_admission_on_resume(sim, evidence):
    extensions = [make_job('A1', seed=s, extended=True) for s in range(92601, 92611)]
    sim.configure([make_job(), *extensions], limit=5000.)
    clock = common.REPORT / 'EXECUTION_CLOCK.json'
    sim.save(clock, dict(training_started_unix=sim.clock[0], limit_seconds=common.LIMIT_SECONDS,
                        latest_start=common.LAST_START))
    original_clock = clock.read_bytes()
    runner = controller.Controller()
    assert runner.extension_allowed(extensions)
    if evidence == 'worker_admission':
        sim.save(common.CACHE / 'worker_admissions' / f'{extensions[0]["job_id"]}_attempt1.json',
                 dict(job=extensions[0], attempt=1, admitted_unix=sim.clock[0], pid=123))
    else:
        sim.result(extensions[0])
    sim.clock[0] += 4801.
    resumed = controller.Controller()
    assert resumed.extension_allowed(extensions)
    assert clock.read_bytes() == original_clock
    assert not (common.REPORT / 'A1_JENA_EXTENSION_ADMISSION_002.json').exists()
    if evidence == 'complete':
        resumed.run_job(extensions[0])
        assert sim.verified == [extensions[0]['job_id']] and not sim.launched


def test_current_job_finishes_after_budget_without_retry_or_next_job(sim):
    sim.configure([make_job(), make_job(seed=92602)], limit=10.)
    sim.outcome = lambda job, attempt: (1, 11., .25)
    controller.main([])
    assert len(sim.launched) == 1
    state = common.read(common.REPORT / 'PROGRESS.json')
    assert state['run_statuses'][sim.jobs[0]['job_id']] == 'FAILED_ONCE_RETRY_NOT_RUN_BUDGET'
    assert state['run_statuses'][sim.jobs[1]['job_id']] == 'NOT_RUN_BUDGET'
    assert state['elapsed_seconds'] >= 11.


def test_cutoff_is_rechecked_after_expensive_preservation_gate(sim, monkeypatch):
    sim.configure(cutoff=10.)
    calls = [0]
    def guard():
        calls[0] += 1
        if calls[0] == 2:
            sim.clock[0] += 11.
        return dict(status='PASS')
    monkeypatch.setattr(common, 'guard_preserved', guard)
    controller.main([])
    assert not sim.launched
    assert not (common.REPORT / 'EXECUTION_CLOCK.json').exists()


def test_stopped_progress_requires_explicit_operator_resume(sim):
    sim.save(common.REPORT / 'PROGRESS.json', dict(status='STOPPED'))
    with pytest.raises(RuntimeError, match='explicit --resume-stopped'):
        controller.main([])
    assert not sim.launched


def test_existing_complete_uses_registered_verification_and_keeps_clock(sim):
    job = sim.jobs[0]
    sim.result(job)
    controller.main([])
    assert sim.launched == [] and sim.verified == [job['job_id']]
    assert not (common.REPORT / 'EXECUTION_CLOCK.json').exists()


def test_resume_reconstructs_failed_family_count_from_saved_state(sim):
    sim.configure([make_job(seed=s) for s in range(92601, 92605)])
    runner = controller.Controller()
    runner.state['run_statuses'].update({job['job_id']: 'FAILED' for job in sim.jobs[:3]})
    sim.save(common.REPORT / 'PROGRESS.json', runner.state)
    controller.main([])
    assert not sim.launched
    assert common.read(common.REPORT / 'PROGRESS.json')['run_statuses'][sim.jobs[-1]['job_id']] == 'NOT_RUN_FAMILY_FAILURE'


def test_live_orphan_worker_lock_stops_new_controller(sim):
    with controller.CampaignLock(common.CACHE / 'worker.lock'):
        assert controller.main([]) == controller.GATE_EXIT
    assert not sim.launched
    assert common.read(common.REPORT / 'EXECUTION_COMPLETION.json')['status'] == 'STOPPED'


def test_postprocess_failure_resume_never_restarts_training(sim):
    sim.report_codes['final'] = 1
    with pytest.raises(RuntimeError, match='Postprocessing failed'):
        controller.main([])
    assert (common.REPORT / 'EXECUTION_COMPLETION.json').exists()
    sim.report_codes['final'] = 0
    controller.main(['--postprocess-only'])
    assert len(sim.launched) == 1
    assert common.read(common.REPORT / 'POSTPROCESS_STATUS.json')['final']['status'] == 'COMPLETE'


def test_final_scientific_failure_blocks_automatic_postprocess_resume(sim):
    sim.report_codes['final'], sim.report_scientific = 1, True
    with pytest.raises(ValueError, match='Scientific reporting gate'):
        controller.main([])
    assert common.read(common.REPORT / 'EXECUTION_COMPLETION.json')['status'] == 'STOPPED'
    with pytest.raises(RuntimeError, match='explicit --resume-stopped'):
        controller.main(['--postprocess-only'])
    assert len(sim.launched) == 1


def test_scientific_interim_failure_stops_admission_and_requires_resolution(sim):
    sim.configure([make_job(), *[make_job('A1', seed=s, extended=True) for s in range(92601, 92611)]])
    sim.report_codes['interim'], sim.report_scientific = 1, True
    assert controller.main([]) == controller.GATE_EXIT
    assert len(sim.launched) < 11
    assert common.read(common.REPORT / 'EXECUTION_COMPLETION.json')['status'] == 'STOPPED'
    with pytest.raises(RuntimeError, match='explicit --resume-stopped'):
        controller.main(['--postprocess-only'])


def test_exclusive_lock_rejects_second_owner_but_stale_file_is_reusable(tmp_path):
    path = tmp_path / 'controller.lock'
    with controller.CampaignLock(path):
        with pytest.raises(RuntimeError, match='Another live'):
            with controller.CampaignLock(path):
                pass
    assert path.exists()
    with controller.CampaignLock(path):
        pass


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_atomic_retries_windows_replace_and_preserves_old_target(sim, monkeypatch, winerror):
    path = common.REPORT / 'atomic.json'
    sim.save(path, dict(status='old'))
    replace, attempts, delays = Path.replace, [], []
    failure = PermissionError('temporary sharing denial')
    failure.winerror = winerror
    def flaky(temporary, target):
        assert common.read(target)['status'] == 'old'
        attempts.append(temporary)
        if len(attempts) < 3:
            raise failure
        return replace(temporary, target)
    monkeypatch.setattr(Path, 'replace', flaky)
    monkeypatch.setattr(common, 'time', SimpleNamespace(sleep=delays.append))
    common.atomic(path, dict(status='new'))
    assert len(attempts) == 3 and delays == [.1, .2]
    assert common.read(path)['status'] == 'new'


def test_common_writers_reject_original_namespace(sim):
    for writer in (common.write_new, common.atomic):
        with pytest.raises(ValueError, match='outside new namespace'):
            writer(common.OLD_REPORT / 'PROGRESS.json', {})
    assert not (common.OLD_REPORT / 'PROGRESS.json').exists()


def test_wrapper_calls_original_without_deadline_and_starts_own_clock(sim, monkeypatch):
    from research.comparative_ablation_20261025_v1 import run_one
    calls, guarded = [], []
    monkeypatch.setattr(common, 'install_write_guard', lambda: guarded.append(True))
    def exact(job, **kwargs):
        assert guarded and (common.REPORT / 'EXECUTION_CLOCK.json').exists()
        admission = common.read(common.CACHE / 'worker_admissions' / f'{job["job_id"]}_attempt1.json')
        assert admission['job'] == job and admission['admitted_unix'] == sim.clock[0]
        calls.append((job, kwargs))
        sim.result(job)
        return common.read(common.OLD_REPORT / 'jobs' / job['job_id'] / 'RESULT.json')
    monkeypatch.setattr(run_one, 'run_job', exact)
    result = job_worker.run(sim.jobs[0], attempt=1)
    assert result['status'] == 'COMPLETE'
    assert calls == [(sim.jobs[0], {'attempt': 1})]
    assert not (common.OLD_REPORT / 'EXECUTION_CLOCK.json').exists()


def test_wrapper_cutoff_prevents_clock_and_original_call(sim, monkeypatch):
    from research.comparative_ablation_20261025_v1 import run_one
    sim.configure(cutoff=0.)
    monkeypatch.setattr(run_one, 'run_job', lambda *args, **kwargs: pytest.fail('No run may be admitted'))
    result = job_worker.run(sim.jobs[0])
    assert result['exit_code'] == job_worker.BUDGET_EXIT
    assert not (common.REPORT / 'EXECUTION_CLOCK.json').exists()


@pytest.mark.parametrize('message', ['Frozen backbone weights changed', 'Frozen backbone acquired parameter gradients'])
def test_frozen_foundation_assertion_is_immediate_scientific_stop(sim, monkeypatch, message):
    from research.comparative_ablation_20261025_v1 import run_one
    def frozen_failure(*args, **kwargs):
        raise AssertionError(message)
    monkeypatch.setattr(run_one, 'run_job', frozen_failure)
    result = job_worker.run(sim.jobs[0])
    assert result['status'] == 'STOPPED' and result['exit_code'] == job_worker.GATE_EXIT
    assert result['error'] == 'AssertionError: ' + message


def test_scientific_worker_failure_never_retries_or_starts_next_job(sim):
    sim.configure([make_job(), make_job(seed=92602)])
    sim.outcome = lambda job, attempt: (job_worker.GATE_EXIT, 1., .25)
    assert controller.main([]) == controller.GATE_EXIT
    assert [(job['seed'], attempt) for job, attempt in sim.launched] == [(92601, 1)]
    assert common.read(common.REPORT / 'EXECUTION_COMPLETION.json')['status'] == 'STOPPED'


def test_wrapper_verification_has_no_training_admission(sim, monkeypatch):
    from research.comparative_ablation_20261025_v1 import run_one
    sim.result(sim.jobs[0])
    calls = []
    def existing(job, **kwargs):
        calls.append((job, kwargs))
        return dict(status='COMPLETE', task_access=access_proof(job['dataset']))
    monkeypatch.setattr(run_one, 'run_job', existing)
    assert job_worker.run(sim.jobs[0], verify_only=True)['status'] == 'VERIFIED_COMPLETE'
    assert calls == [(sim.jobs[0], {'attempt': 1})]
    assert not (common.REPORT / 'EXECUTION_CLOCK.json').exists()


def test_wrapper_refuses_additional_job_fields(sim):
    with pytest.raises(ValueError, match='six-field'):
        job_worker.validate_spec(dict(sim.jobs[0], deadline=2000.))


def test_jena_actual_bounded_loader_receipt_and_missing_member_proof():
    original = Path(__file__).resolve().parents[2] / 'reports/comparative_ablation_20261025_v1/jobs/CA_A1_jena_K6_s92601/RESULT.json'
    result = common.read(original)
    assert 'T_numeric_values_parsed' not in result['task_access']
    job_worker.assert_protected_T(result, 'jena')
    result['task_access']['members'][0]['tail_numeric_values_parsed'] = 1
    with pytest.raises(ValueError, match='Jena bounded-loader proof'):
        job_worker.assert_protected_T(result, 'jena')


@pytest.mark.parametrize('dataset', ['ETTh1', 'ETTh2_safe17400', 'appliances'])
def test_non_jena_prefix_proof_requires_explicit_zero(dataset):
    job_worker.assert_protected_T(dict(task_access=access_proof(dataset)), dataset)
    with pytest.raises(ValueError, match='proof is missing'):
        job_worker.assert_protected_T(dict(task_access={}), dataset)
