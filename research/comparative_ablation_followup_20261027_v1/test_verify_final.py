"""Synthetic tests; never run the final checker against the live campaign."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from research.comparative_ablation_followup_20261027_v1 import verify_final as v


def plan_fixture():
    plan = dict(jobs=[], extension_jobs=[])
    for extended in (False, True):
        for method in (('C1', 'C2', 'C3', 'A1', 'A2', 'A3', 'A4', 'A5') if not extended else
                       ('C2', 'C3', 'A1', 'A2', 'A3', 'A4', 'A5')):
            for dataset, k in zip(v.DATASETS, (6, 4, 4, 10)):
                for seed in ([92601] if method == 'C1' else sorted(v.SEEDS)):
                    suffix = '_ext10' if extended else ''
                    job = dict(job_id=f'CA_{method}_{dataset}_K{k}_s{seed}{suffix}',
                               method_id=method, dataset=dataset, K=k, seed=seed, extended=extended)
                    plan['extension_jobs' if extended else 'jobs'].append(job)
    return plan


def completion_fixture(n=50):
    plan = plan_fixture()
    chosen = v.selected_jobs(plan)
    old = {job['job_id'] for job in plan['jobs'] if job['method_id'] in ('C1', 'C2', 'C3', 'A1', 'A2')}
    registered = plan['jobs'] + plan['extension_jobs']
    results = [dict(job=job) for job in chosen[:n]]
    selected_ids = {job['job_id'] for job in chosen}
    status = {job['job_id']: 'NOT_RUN_SCOPE' for job in registered if job['job_id'] not in old | selected_ids}
    status.update({job['job_id']: 'COMPLETE' if i < n else 'NOT_RUN_BUDGET' for i, job in enumerate(chosen)})
    completion = dict(status='COMPLETE' if n == 50 else 'PARTIAL', run_statuses=status,
                      prior_complete_job_ids=sorted(old), stop_reason=None if n == 50 else 'budget')
    return completion, registered, chosen, old, results


@pytest.mark.parametrize('n', [0, 7, 40, 47, 50])
def test_legal_partial_uses_actual_n_and_missing_reasons(n):
    args = completion_fixture(n)
    result = v.check_completion_state(*args)
    assert result['new_complete'] == n
    assert len(result['missing']) == 50-n
    assert result['out_of_scope_n'] == 350
    assert all(row['reason'] == 'NOT_RUN_BUDGET' for row in result['missing'])


def test_nonterminal_and_stopped_have_distinct_states():
    args = list(completion_fixture(7))
    args[0]['status'] = 'RUNNING'
    with pytest.raises(v.Incomplete):
        v.check_completion_state(*args)
    args[0]['status'] = 'STOPPED'
    with pytest.raises(v.RootAction):
        v.check_completion_state(*args)


def test_complete_or_scope_mislabel_is_rejected():
    args = list(completion_fixture(40))
    args[0]['status'] = 'COMPLETE'
    with pytest.raises(ValueError, match='completion status/count'):
        v.check_completion_state(*args)
    args = list(completion_fixture(40))
    out = next(key for key, value in args[0]['run_statuses'].items() if value == 'NOT_RUN_SCOPE')
    args[0]['run_statuses'][out] = 'NOT_RUN_BUDGET'
    with pytest.raises(ValueError, match='mislabeled'):
        v.check_completion_state(*args)


def test_unresolved_selected_status_is_incomplete():
    args = list(completion_fixture(40))
    args[0]['run_statuses'][args[2][40]['job_id']] = 'FAILED_ONCE'
    with pytest.raises(v.Incomplete, match='Unresolved'):
        v.check_completion_state(*args)


def test_fixed_scope_rejects_seed_change():
    plan = plan_fixture()
    next(j for j in plan['jobs'] if j['method_id'] == 'A4')['seed'] = 42
    with pytest.raises(ValueError):
        v.selected_jobs(plan)


def admission_fixture(tmp_path, monkeypatch, tamper=None):
    report = tmp_path / 'reports' / v.NAME
    report.mkdir(parents=True)
    sources = []
    for name in ('common', 'controller', 'job_worker', 'floors'):
        path = tmp_path / f'research/{v.NAME}/{name}.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# uncommitted {name}\n', encoding='utf8')
        sources.append(dict(path=path.relative_to(tmp_path).as_posix(), sha256=v.sha(path)))
    registered_sources = copy.deepcopy(sources)
    if tamper == 'committed_registration':
        registered_sources[0]['sha256'] = '0'*64
    addendum_name = f'reports/{v.NAME}/PREREGISTRATION_ADDENDUM.json'
    markdown_name = f'reports/{v.NAME}/PREREGISTRATION_ADDENDUM.md'
    committed = {addendum_name: json.dumps(dict(new_source_hashes_at_registration=registered_sources)).encode('utf8'),
                 markdown_name: b'# Two-file preregistration commit\n'}
    for name, payload in committed.items():
        (tmp_path / name).write_bytes(payload)
    commit = 'two-file-commit'
    prereg = dict(commit=commit, commit_unix=100, files=[dict(path=name,
        sha256=hashlib.sha256(payload).hexdigest()) for name, payload in committed.items()])
    chosen = v.selected_jobs(plan_fixture())
    clock = dict(training_started_unix=200, first_job_id=chosen[0]['job_id'],
                 limit_seconds=46800, latest_start='2026-10-28T00:00:00+09:00')
    admission = dict(status='PASS', commit=commit, new_sources=copy.deepcopy(sources))
    if tamper == 'admission':
        admission['new_sources'][0]['sha256'] = '0'*64
    elif tamper == 'current_source':
        (tmp_path / sources[0]['path']).write_text('# changed\n', encoding='utf8')
    for name, payload in [('PREREGISTRATION_COMMIT.json', prereg), ('ADMISSION.json', admission),
                          ('EXECUTION_CLOCK.json', clock)]:
        (report / name).write_text(json.dumps(payload), encoding='utf8')
    worker = tmp_path / '.cache' / v.NAME / 'worker_admissions' / 'first.json'
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps(dict(job=chosen[0], attempt=1, admitted_unix=200)), encoding='utf8')
    shown = []

    def fake_git(repo, *args):
        assert repo == tmp_path
        if args == ('show', '-s', '--format=%ct', commit):
            return b'100\n'
        assert args[0] == 'show' and args[1].startswith(commit + ':')
        name = args[1].split(':', 1)[1]
        shown.append(name)
        # Frozen .py files are deliberately absent from the preregistration commit.
        assert name in committed, 'Verifier incorrectly requires a source-file commit: ' + name
        return committed[name]

    monkeypatch.setattr(v, 'git', fake_git)
    monkeypatch.setattr(v.subprocess, 'run', lambda *args, **kwargs: None)
    return chosen, shown, set(committed)


def test_two_file_prereg_commit_binds_uncommitted_sources_through_addendum_hashes(tmp_path, monkeypatch):
    selected, shown, committed_names = admission_fixture(tmp_path, monkeypatch)
    result = v.check_admission(tmp_path, selected)
    assert result['worker_admissions'] == 1 and result['delta_seconds'] == 100
    assert set(shown) == committed_names
    assert not any(name.endswith('.py') for name in shown)


@pytest.mark.parametrize('tamper', ['committed_registration', 'admission', 'current_source'])
def test_registered_admission_current_source_sha_three_way_match_is_required(tmp_path, monkeypatch, tamper):
    selected, _, _ = admission_fixture(tmp_path, monkeypatch, tamper)
    with pytest.raises(ValueError, match='committed addendum SHA'):
        v.check_admission(tmp_path, selected)


@pytest.mark.parametrize('dataset', ['jena', 'ETTh1'])
def test_zero_T_proof_required_and_member_names_are_bound(dataset):
    result = dict(job=dict(dataset=dataset, job_id='test'), task_access={})
    with pytest.raises(ValueError):
        v.no_t(result)
    if dataset == 'jena':
        result['task_access'] = dict(new_T_numeric_values_parsed=0, members=[dict(member=name,
            tail_numeric_values_parsed=0) for name in ('x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy')])
    else:
        result['task_access'] = dict(T_numeric_values_parsed=0)
    v.no_t(result)
    if dataset == 'jena':
        result['task_access']['members'][-1]['member'] = 'extra.npy'
    else:
        result['task_access']['T_numeric_values_parsed'] = 1
    with pytest.raises(ValueError):
        v.no_t(result)


def runs_fixture(extension_n):
    plan = plan_fixture()
    jobs = [job for job in plan['jobs'] if job['method_id'] in ('C1', 'C2', 'C3', 'A1', 'A2', 'A4')]
    jobs += [job for job in plan['extension_jobs'] if job['method_id'] == 'A1' and job['dataset'] == 'jena'][:extension_n]
    records = [dict(job=job, row=dict(job, MSE=1.0)) for job in jobs]
    reuse = []
    for dataset in v.DATASETS:
        for method in ('RAW', 'OURS'):
            for seed in ([92601] if method == 'RAW' else sorted(v.SEEDS)):
                reuse.append(dict(job_id=f'{method}_{dataset}_{seed}', method_id=method,
                                  dataset=dataset, seed=seed, mse_recomputed=1.0))
    run_rows = []
    for source in [r['row'] for r in records] + reuse:
        row = dict(source, MSE=1., status='COMPLETE', extended=source.get('extended', False),
                   cap_selected=source['method_id'] == 'A1' and not source.get('extended', False))
        row['is_primary'] = not row['extended']
        if row['method_id'] == 'A1' and row['dataset'] == 'jena' and extension_n == 10:
            row['is_primary'] = row['extended']
        run_rows.append(row)
    summary = []
    for dataset in v.DATASETS:
        for method in v.METHODS:
            n = 0 if method in ('A3', 'A5') else 1 if method in ('RAW', 'C1') else 10
            extended = method == 'A1' and dataset == 'jena' and extension_n == 10
            summary.append(dict(dataset=dataset, method_id=method, n=n, expected_n=1 if method in ('RAW', 'C1') else 10,
                complete=n > 0, extended=extended, extension_n=extension_n if (method, dataset) == ('A1', 'jena') else 0,
                provisional=method == 'A1' and not extended, undertrained_at_10=False,
                MSE_median=1. if n else ''))
    return run_rows, summary, records, reuse


@pytest.mark.parametrize('extension_n', [0, 9, 10])
def test_primary_extension_requires_all_ten_and_preserves_total_rows(extension_n):
    result = v.validate_runs(*runs_fixture(extension_n))
    assert result['run_rows'] == 248+extension_n
    assert result['A1_jena_primary']['extension_primary'] == (extension_n == 10)
    assert result['A1_jena_primary']['n'] == 10


def test_partial_extension_cannot_be_selected_and_duplicate_rows_fail():
    args = list(runs_fixture(9))
    row = next(r for r in args[0] if r['method_id'] == 'A1' and r['extended'])
    row['is_primary'] = True
    with pytest.raises(ValueError, match='complete-seed rule'):
        v.validate_runs(*args)
    args = list(runs_fixture(10))
    args[0].append(copy.deepcopy(args[0][0]))
    with pytest.raises(ValueError, match='row count/identity'):
        v.validate_runs(*args)


def zip_fixture(tmp_path, extra_name=None, symlink=False):
    report = tmp_path / 'reports' / v.NAME
    report.mkdir(parents=True)
    data_name = extra_name or f'reports/{v.NAME}/RESULTS_KO.md'
    data = b'public summary\n'
    local = tmp_path / data_name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    manifest = report / 'BUNDLE_MANIFEST.json'
    manifest.write_text(json.dumps(dict(files=[dict(path=data_name, bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest())])), encoding='utf8')
    target = report / 'bundle.zip'
    with zipfile.ZipFile(target, 'x') as archive:
        if symlink:
            info = zipfile.ZipInfo(data_name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, data)
        else:
            archive.writestr(data_name, data)
        archive.writestr(f'reports/{v.NAME}/BUNDLE_MANIFEST.json', manifest.read_bytes())
    receipt = report / 'BUNDLE_RECEIPT.json'
    receipt.write_text(json.dumps(dict(sha256=v.sha(target), bytes=target.stat().st_size, entries=2,
                                      manifest_sha256=v.sha(manifest))), encoding='utf8')
    return target, manifest, receipt, tmp_path


def test_zip_independent_crc_entry_sha_and_current_public_bytes(tmp_path):
    args = zip_fixture(tmp_path)
    result = v.check_zip(*args)
    assert result['entries'] == 2 and result['all_entry_sha256'] == 'PASS'
    with pytest.raises(ValueError, match='Required public evidence absent'):
        v.check_zip(*args, required_names=[f'reports/{v.NAME}/RUNS.csv'])
    (tmp_path / f'reports/{v.NAME}/RESULTS_KO.md').write_bytes(b'changed')
    with pytest.raises(ValueError, match='current public evidence'):
        v.check_zip(*args)


def test_zip_rejects_extra_duplicate_and_symlink_members(tmp_path):
    args = zip_fixture(tmp_path / 'extra')
    with zipfile.ZipFile(args[0], 'a') as archive:
        archive.writestr('extra.txt', 'bad')
    with pytest.raises(ValueError, match='entry set'):
        v.check_zip(*args)
    args = zip_fixture(tmp_path / 'symlink', symlink=True)
    with pytest.raises(ValueError, match='symlink'):
        v.check_zip(*args)


def test_valid_crc_cannot_hide_an_entry_sha_change(tmp_path):
    args = zip_fixture(tmp_path)
    data_name = f'reports/{v.NAME}/RESULTS_KO.md'
    with zipfile.ZipFile(args[0], 'w') as archive:
        archive.writestr(data_name, b'altered summary\n')
        archive.writestr(f'reports/{v.NAME}/BUNDLE_MANIFEST.json', args[1].read_bytes())
    with pytest.raises(ValueError, match='entry SHA/size mismatch'):
        v.check_zip(*args)


@pytest.mark.parametrize('path', [f'reports/{v.NAME}/secret.npz', f'reports/{v.NAME}/checkpoints/model.json',
    f'reports/{v.NAME}/data/raw.csv', '.cache/model.json', f'reports/{v.NAME}/inside.zip',
    '../escape.json', '/absolute.json', 'C:/absolute.json', f'reports/{v.NAME}/selected_bundle.pt'])
def test_forbidden_archive_members_and_path_escape(path):
    with pytest.raises(ValueError):
        v.archive_public_path(path)


def test_aggregate_status_does_not_claim_campaign_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(v, 'check_preservation', lambda repo: dict(checked=7393))
    monkeypatch.setattr(v, 'check_git_blobs', lambda repo: dict(checked=2450))
    context = dict(selected=[], completion=dict(status='PARTIAL'), detail=dict(new_complete=40))
    monkeypatch.setattr(v, 'result_context', lambda repo: context)
    monkeypatch.setattr(v, 'check_admission', lambda repo, selected: dict(worker_admissions=40))
    monkeypatch.setattr(v, 'check_reports', lambda repo, context: dict(final_runs=248))
    monkeypatch.setattr(v, 'check_zip', lambda *args: dict(entries=42))
    monkeypatch.setattr(v, 'required_archive_members', lambda *args: set())
    monkeypatch.setattr(v, 'read', lambda path: dict(instruction_sha256='instruction'))
    result = v.verify(tmp_path)
    assert result['status'] == 'PASS'
    assert result['execution_status'] == 'PARTIAL'
    assert result['campaign_completion_claim'] is False
    assert result['scope'] == 'independent machine checks only'
    assert len(result['remaining_root_responsibilities']) == 3
