"""Synthetic completion evidence; no model training or real campaign analysis."""
import copy
import json

import pytest

from research.jena_p_branch_20261027_v1 import postprocess, run_one, runtime


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'REPO', tmp_path)
    monkeypatch.setattr(runtime, 'REPORT', tmp_path / 'reports' / runtime.NAME)
    registration = {'seal_sha256': 'registered', 'registered_at_utc': '2026-10-27T10:00:00+00:00', 'files': {}}
    monkeypatch.setattr(run_one, 'validate_registration', lambda job: copy.deepcopy(registration))
    plan = {'jobs': [], 'extension_jobs': []}
    for extended, key in ((False, 'jobs'), (True, 'extension_jobs')):
        for seed in range(92601, 92611):
            for method in run_one.METHOD_FAMILY:
                plan[key].append(dict(job_id=f'JPB_{method}_jena_K6_s{seed}' + ('_ext10' if extended else ''),
                                      method_id=method, dataset='jena', K=6, seed=seed, extended=extended))
    completion = dict(status='TRAINING_COMPLETE', run_statuses={}, attempts={},
                      extension_status={m: {'status': 'NOT_REQUIRED'} for m in run_one.METHOD_FAMILY})

    def complete(job, attempt=1):
        root = runtime.REPORT / 'jobs' / job['job_id']
        public = root / f'attempt_{attempt:03d}'
        private = tmp_path / '.cache' / runtime.NAME / job['job_id']
        private.mkdir(parents=True)
        bundle, prediction = private / 'bundle.pt', private / 'prediction.npz'
        bundle.write_bytes(b'synthetic adapter bytes')
        prediction.write_bytes(b'synthetic archive bytes; arrays tested separately')
        spec = run_one.build_spec(job)
        contract = {key: key + '-hash' for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha')}
        curves = [dict(epoch=epoch, astop_mse=1. if epoch == 1. else 2.) for epoch in [0., *spec['epochs']]]
        search = dict(summary=dict(status='SEGMENT_COMPLETE', selected_epoch=1., selected_astop_mse=1., updates=1),
                      curves=curves)
        refit = dict(status='SEGMENT_COMPLETE', updates=1)
        selection = dict(status='SEALED', job=job, V_used=False, selection_split='A_stop',
                         selected_epoch=1., selected_astop_mse=1., spec=spec, contract=contract, curves=curves)
        save(public / 'SELECTION.json', selection)
        save(public / 'SEARCH_RESULT.json', search)
        save(public / 'REFIT_SUMMARY.json', refit)
        save(public / 'SOURCE_CONTRACT.json', dict(contract, sources={'core.py': 'unchanged'}, spec=spec))
        receipt = dict(job=job, attempt=attempt, registration=registration, status='COMPLETE',
                       training_attempted=True, training_executed=True, T_numeric_values_parsed=0,
                       preserved_before={'checked': 1, 'changed': []}, preserved_after={'checked': 1, 'changed': []})
        save(public / 'RECEIPT.json', receipt)
        save(public / 'START.json', dict(receipt, spec=spec, status='FAILED',
                                        training_attempted=False, training_executed=False))
        save(public / 'OPENED_FILES.json', dict(T_numeric_values_parsed=0, opens=[]))
        save(root / f'WORKER_RECEIPT_attempt{attempt}.json', dict(job_id=job['job_id'], attempt=attempt,
                                                               pid=1234, status='COMPLETE', exit_code=0))
        row = dict(job, reused=False, selected_epoch=1., training_cap=spec['epochs'][-1], cap_selected=False,
                   prediction_path=str(prediction), prediction_sha256=runtime.sha(prediction))
        result = dict(status='COMPLETE', job=job, attempt=attempt, registration=registration, spec=spec,
                      selection_sealed_before_V=True, contract=contract, row=row, rows=[row],
                      bundle_path=str(bundle), bundle_sha256=runtime.sha(bundle),
                      selection_path=str(public / 'SELECTION.json'), selection_sha256=runtime.sha(public / 'SELECTION.json'),
                      search_summary=search['summary'], refit_summary=refit,
                      task_access=dict(new_T_numeric_values_parsed=0,
                                       members=[dict(member=m, tail_numeric_values_parsed=0) for m in
                                                ('x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy')]))
        save(root / 'RESULT.json', result)
        completion['run_statuses'][job['job_id']] = 'COMPLETE'
        return root

    for job in plan['jobs']:
        complete(job)
    return plan, completion, complete


def test_all_base_evidence_count_and_public_hashes(campaign):
    plan, completion, _ = campaign
    results, base_complete, hashes = postprocess.verify_completion_evidence(plan, completion)
    assert base_complete == 20
    assert set(results) == {job['job_id'] for job in plan['jobs']}
    assert len(hashes) == 20 * 9
    assert all(path.startswith('reports/') and '\\' not in path and '.cache/' not in path for path in hashes)
    assert all(runtime.sha(runtime.REPO / path) == digest for path, digest in hashes.items())
    # START is intentionally an initial FAILED placeholder, not the final receipt.


def test_complete_extensions_cannot_hide_missing_base(campaign):
    plan, completion, complete = campaign
    for job in plan['extension_jobs']:
        complete(job)
    for method in run_one.METHOD_FAMILY:
        completion['extension_status'][method]['status'] = 'COMPLETE'
    missing = plan['jobs'][0]['job_id']
    (runtime.REPORT / 'jobs' / missing / 'RESULT.json').unlink()
    with pytest.raises(ValueError, match='Missing completed base job: ' + missing):
        postprocess.verify_completion_evidence(plan, completion)


def test_complete_extension_families_are_verified_too(campaign):
    plan, completion, complete = campaign
    for job in plan['extension_jobs']:
        complete(job)
    for method in run_one.METHOD_FAMILY:
        completion['extension_status'][method]['status'] = 'COMPLETE'
    results, base_complete, hashes = postprocess.verify_completion_evidence(plan, completion)
    assert base_complete == 20 and len(results) == 40 and len(hashes) == 40 * 9


@pytest.mark.parametrize('filename', ['RESULT.json', 'WORKER_RECEIPT_attempt1.json',
                                    'attempt_001/RECEIPT.json', 'attempt_001/SOURCE_CONTRACT.json'])
def test_missing_required_evidence_is_rejected(campaign, filename):
    plan, completion, _ = campaign
    (runtime.REPORT / 'jobs' / plan['jobs'][0]['job_id'] / filename).unlink()
    with pytest.raises(ValueError, match='Missing completed base job|missing public evidence'):
        postprocess.verify_completion_evidence(plan, completion)


@pytest.mark.parametrize('filename,key,value,error', [
    ('WORKER_RECEIPT_attempt1.json', 'exit_code', 1, 'worker receipt'),
    ('attempt_001/RECEIPT.json', 'status', 'FAILED', 'attempt receipt'),
    ('attempt_001/SOURCE_CONTRACT.json', 'data_sha', 'different', 'Source contract'),
    ('attempt_001/REFIT_SUMMARY.json', 'status', 'FAILED', 'refit evidence'),
    ('attempt_001/START.json', 'attempt', 2, 'Start identity'),
])
def test_inconsistent_receipts_are_rejected(campaign, filename, key, value, error):
    plan, completion, _ = campaign
    path = runtime.REPORT / 'jobs' / plan['jobs'][0]['job_id'] / filename
    record = runtime.read(path)
    record[key] = value
    save(path, record)
    with pytest.raises(ValueError, match=error):
        postprocess.verify_completion_evidence(plan, completion)


def test_controller_completion_required(campaign):
    plan, completion, _ = campaign
    completion['run_statuses'][plan['jobs'][0]['job_id']] = 'RUNNING'
    with pytest.raises(ValueError, match='Missing completed base job'):
        postprocess.verify_completion_evidence(plan, completion)


def test_controller_success_attempt_must_match_worker_when_present(campaign):
    plan, completion, _ = campaign
    job_id = plan['jobs'][0]['job_id']
    completion['attempts'][job_id] = [dict(job_id=job_id, attempt=1, exit_code=1, pid=1234)]
    with pytest.raises(ValueError, match='Controller attempt differs'):
        postprocess.verify_completion_evidence(plan, completion)


def test_windows_launcher_pid_may_differ_from_worker_pid(campaign):
    plan, completion, _ = campaign
    job_id = plan['jobs'][0]['job_id']
    completion['attempts'][job_id] = [dict(job_id=job_id, attempt=1, exit_code=0, pid=5678)]
    _, base_complete, _ = postprocess.verify_completion_evidence(plan, completion)
    assert base_complete == 20


def test_partial_extension_successful_second_attempt_is_retained(campaign):
    plan, completion, complete = campaign
    job = plan['extension_jobs'][0]
    complete(job, attempt=2)
    completion['extension_status'][job['method_id']]['status'] = 'INCOMPLETE'
    results, base_complete, hashes = postprocess.verify_completion_evidence(plan, completion)
    assert base_complete == 20 and len(results) == 21
    assert results[job['job_id']]['attempt'] == 2
    assert any(path.endswith('WORKER_RECEIPT_attempt2.json') for path in hashes)


def test_extension_completion_claim_requires_whole_family(campaign):
    plan, completion, _ = campaign
    completion['extension_status']['P_TIME_ONLY']['status'] = 'COMPLETE'
    with pytest.raises(ValueError, match='Extension family completion differs'):
        postprocess.verify_completion_evidence(plan, completion)


def test_recorded_selection_cannot_disagree_with_astop(campaign):
    plan, completion, _ = campaign
    path = runtime.REPORT / 'jobs' / plan['jobs'][0]['job_id'] / 'RESULT.json'
    record = runtime.read(path)
    record['row']['selected_epoch'] = 3.
    save(path, record)
    with pytest.raises(ValueError, match='Result row differs'):
        postprocess.verify_completion_evidence(plan, completion)


def test_public_output_hashes_are_repository_relative(campaign):
    outputs = [runtime.REPORT / name for name in ('RUNS.csv', 'FIG_JENA_P_BRANCHES.png', 'RESULTS_KO.md')]
    for path in outputs:
        path.write_bytes(b'synthetic published artifact')
    hashes = postprocess.public_hashes(outputs)
    assert set(hashes) == {path.relative_to(runtime.REPO).as_posix() for path in outputs}
    assert all(runtime.sha(runtime.REPO / path) == digest for path, digest in hashes.items())
