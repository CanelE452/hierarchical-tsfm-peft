"""Independent stdlib checks of public aggregates, evidence bytes and final ZIP.

This does not import campaign execution/analysis code or parse prediction/data
arrays. PASS means machine checks passed, not that the campaign was delivered.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import stat
import subprocess
import zipfile


NAME = 'comparative_ablation_followup_20261027_v1'
OLD_NAME = 'comparative_ablation_20261025_v1'
BASE = '06f142dbba1ab0370fb69366eb6cd28b721d5505'
SEEDS = set(range(92601, 92611))
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
METHODS = ('RAW', 'C1', 'C2', 'C3', 'OURS', 'A1', 'A2', 'A3', 'A4', 'A5')
TERMINAL_MISSING = {'FAILED', 'NOT_RUN_BUDGET', 'NOT_RUN_FAMILY_FAILURE',
                    'FAILED_ONCE_RETRY_NOT_RUN_BUDGET', 'FAILED_ONCE_CONTROLLER_STOPPED'}
JOB_FIELDS = {'job_id', 'method_id', 'dataset', 'K', 'seed', 'extended'}
RECEIPT_NAME = 'FINAL_INDEPENDENT_VALIDATION.json'
REGISTERED_ARTIFACTS = (
    'METRICS_EXISTING.csv', 'ERROR_DISTRIBUTION_EXISTING.csv', 'ERROR_DISTRIBUTION_EXISTING_SUMMARY.csv',
    'VMSE_DEFINITION.md', 'RUNS.csv', 'SUMMARY.csv', 'COMPARATIVE.csv', 'ABLATION.csv',
    'DECISION.json', 'RESULTS_KO.md', 'SLIDE_NUMBERS.md', 'ERROR_DISTRIBUTION.csv',
    'ERROR_DISTRIBUTION_SUMMARY.csv',
    *[f'{name}.{suffix}' for name in ('FIG_COMPARATIVE', 'FIG_ABLATION', 'FIG_ERROR_QUANTILES')
      for suffix in ('csv', 'png', 'svg')])
UPDATED_ARTIFACTS = ('PREDICTION_CHECK.md', 'PREDICTION_CHECK.json', 'SLIDE_NUMBERS_UPDATED.md',
    'D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv', 'D0_D1_VALIDATION.json',
    'FIG_ABLATION_UPDATED.csv', 'FIG_ABLATION_UPDATED.png', 'FIG_ABLATION_UPDATED.svg',
    'REPORTING_INPUTS.json', 'REPORTING_COMPLETION.json', 'PRESERVATION_REPORTING.json')
FINAL_FLOORS = ('D0_SUBSPACE_FLOOR.csv', 'D0_MEAN_SHIFT_CHANNELS.csv')


class Incomplete(Exception):
    pass


class RootAction(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise Incomplete('Missing evidence: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def rows(path):
    path = Path(path)
    if not path.is_file():
        raise Incomplete('Missing aggregate: ' + str(path))
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def flag(value):
    require(value in (True, False, 'True', 'False'), 'Invalid boolean: ' + repr(value))
    return value is True or value == 'True'


def safe_relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name == path.as_posix() and '\\' not in name and ':' not in name and
            not path.is_absolute() and bool(path.parts) and
            all(part not in ('.', '..') for part in path.parts), 'Unsafe relative path: ' + str(name))
    return path


def selected_jobs(plan):
    registered = plan['jobs'] + plan['extension_jobs']
    ids = [row['job_id'] for row in registered]
    require(len(ids) == len(set(ids)) == 564, 'Expected 564 unique original registered jobs')
    chosen = [job for job in registered if (job['method_id'] == 'A4' and job['extended'] is False) or
              (job['method_id'] == 'A1' and job['dataset'] == 'jena' and job['extended'] is True)]
    require(len(chosen) == 50, 'Expected exactly 50 selected jobs')
    for job in chosen:
        require(set(job) == JOB_FIELDS, 'Selected job fields differ: ' + job['job_id'])
        expected_k = {'jena': 6, 'ETTh1': 4, 'ETTh2_safe17400': 4, 'appliances': 10}[job['dataset']]
        suffix = '_ext10' if job['extended'] else ''
        expected_id = f"CA_{job['method_id']}_{job['dataset']}_K{expected_k}_s{job['seed']}{suffix}"
        require(job['job_id'] == expected_id and job['K'] == expected_k, 'Selected identity/K changed')
    for dataset in DATASETS:
        require({j['seed'] for j in chosen if j['method_id'] == 'A4' and j['dataset'] == dataset} == SEEDS,
                'A4 fixed seed set differs: ' + dataset)
    require({j['seed'] for j in chosen if j['method_id'] == 'A1'} == SEEDS, 'A1 extension seed set differs')
    return sorted(chosen, key=lambda j: (j['method_id'] != 'A4', j['seed'], DATASETS.index(j['dataset'])))


def check_preservation(repo):
    baseline = read(repo / 'reports' / NAME / 'PRESERVATION_BASELINE.json')
    require(baseline['baseline_commit'] == BASE, 'Preservation base commit changed')
    files = baseline['files']
    require(len(files) == len({row['path'] for row in files}) == 7393, 'Expected 7393 unique baseline files')
    require(baseline['committed_old_namespace_files'] == 2450 and baseline['original_PRE_HASHES_files'] == 341,
            'Baseline preservation scope counts changed')
    changed = []
    for row in files:
        path = Path(row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            changed.append(str(path))
    require(not changed, 'Baseline file change: ' + ', '.join(changed))
    return dict(checked=len(files), changed=[], baseline_sha256=sha(repo / 'reports' / NAME / 'PRESERVATION_BASELINE.json'))


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.PIPE)


def check_git_blobs(repo):
    listing = git(repo, 'ls-tree', '-r', '-z', BASE, '--', f'research/{OLD_NAME}/', f'reports/{OLD_NAME}/')
    entries = [entry for entry in listing.split(b'\0') if entry]
    require(len(entries) == 2450, 'Expected 2450 original source/report Git blobs')
    changed = []
    for entry in entries:
        metadata, encoded_path = entry.split(b'\t', 1)
        mode, kind, expected = metadata.split()
        name = encoded_path.decode('utf8')
        require(mode == b'100644' and kind == b'blob', 'Unexpected original Git entry: ' + name)
        path = repo / safe_relative(name)
        if not path.is_file():
            changed.append(name)
            continue
        digest = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected.decode():
            changed.append(name)
    require(not changed, 'Original Git blob bytes changed: ' + ', '.join(changed))
    return dict(base_commit=BASE, checked=len(entries), changed=[])


def no_t(result):
    access = result.get('task_access', {})
    if result['job']['dataset'] == 'jena':
        expected = {'x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy'}
        members = access.get('members', [])
        require(access.get('new_T_numeric_values_parsed') == 0 and len(members) == 6 and
                {m.get('member') for m in members} == expected and
                all(m.get('tail_numeric_values_parsed') == 0 for m in members),
                'Jena zero-T/member proof missing or nonzero: ' + result['job']['job_id'])
    else:
        require(access.get('T_numeric_values_parsed') == 0,
                'Zero-T proof missing or nonzero: ' + result['job']['job_id'])


def check_completion_state(completion, registered, selected, old_ids, new_results):
    state = completion.get('status')
    if state == 'STOPPED':
        raise RootAction('Execution STOPPED: ' + str(completion.get('stop_reason')))
    if state not in ('COMPLETE', 'PARTIAL'):
        raise Incomplete('Execution is not terminal: ' + str(state))
    selected_ids = {job['job_id'] for job in selected}
    complete_ids = {result['job']['job_id'] for result in new_results}
    require(complete_ids <= selected_ids, 'Out-of-scope new result')
    status = completion['run_statuses']
    outside = {job['job_id'] for job in registered} - selected_ids - old_ids
    pending = [job for job in selected_ids if job not in status or status[job] in ('RUNNING', 'FAILED_ONCE')]
    if pending:
        raise Incomplete('Unresolved selected statuses: ' + ', '.join(sorted(pending)))
    require(set(status) == selected_ids | outside, 'Final run_statuses identity set differs')
    require(set(completion['prior_complete_job_ids']) == old_ids, 'Prior complete job IDs changed')
    require(all(status[job] == 'NOT_RUN_SCOPE' for job in outside), 'Outside selected scope is mislabeled')
    missing = []
    for job in selected:
        value = status[job['job_id']]
        if job['job_id'] in complete_ids:
            require(value == 'COMPLETE', 'Result/status mismatch: ' + job['job_id'])
        else:
            require(value in TERMINAL_MISSING, 'Missing result has no legal terminal reason: ' + job['job_id'])
            missing.append(dict(job_id=job['job_id'], method_id=job['method_id'], dataset=job['dataset'], reason=value))
    require((state == 'COMPLETE') == (len(complete_ids) == 50), 'Execution completion status/count differs')
    if state == 'PARTIAL':
        require(bool(completion.get('stop_reason')), 'PARTIAL execution lacks a reason')
    groups = []
    for method, dataset in [('A4', dataset) for dataset in DATASETS] + [('A1', 'jena')]:
        group = [job for job in selected if (job['method_id'], job['dataset']) == (method, dataset)]
        n = sum(job['job_id'] in complete_ids for job in group)
        groups.append(dict(method_id=method, dataset=dataset, n=n, expected_n=10,
                           missing_reasons=dict(Counter(status[j['job_id']] for j in group if j['job_id'] not in complete_ids))))
    return dict(execution_status=state, new_complete=len(complete_ids), selected_n=50,
                missing=missing, groups=groups, out_of_scope_n=len(outside), stop_reason=completion.get('stop_reason'))


def result_context(repo):
    old_report, report = repo / 'reports' / OLD_NAME, repo / 'reports' / NAME
    plan = read(old_report / 'RUN_PLAN.json')
    selected = selected_jobs(plan)
    require(read(report / 'EXECUTION_PLAN.json')['jobs'] == selected, 'Execution plan differs from registered selection/order')
    original = read(old_report / 'EXECUTION_COMPLETION.json')
    old_ids = {key for key, value in original['run_statuses'].items() if value == 'COMPLETE'}
    require(len(old_ids) == 164, 'Expected 164 original complete results')
    paths = sorted((old_report / 'jobs').glob('*/RESULT.json'))
    results = [read(path) for path in paths]
    ids = [result['job']['job_id'] for result in results]
    require(len(ids) == len(set(ids)) and old_ids <= set(ids), 'Duplicate/missing original RESULT identities')
    chosen = {job['job_id']: job for job in selected}
    registered = {job['job_id']: job for job in plan['jobs'] + plan['extension_jobs']}
    require(set(ids) - old_ids <= set(chosen), 'Unapproved new result exists')
    new = []
    for path, result in zip(paths, results):
        job = result['job']
        require(result.get('status') == 'COMPLETE' and path.parent.name == job['job_id'] and
                job == registered.get(job['job_id']), 'Invalid RESULT registration/status: ' + str(path))
        if job['job_id'] not in old_ids:
            require(job == chosen[job['job_id']], 'New job differs from selected six fields')
            require(all(result['row'].get(key) == job[key] for key in JOB_FIELDS), 'RESULT row identity differs')
            no_t(result)
            new.append(result)
    completion = read(report / 'EXECUTION_COMPLETION.json')
    detail = check_completion_state(completion, list(registered.values()), selected, old_ids, new)
    return dict(results=results, new=new, selected=selected, old_ids=old_ids, completion=completion,
                detail=detail, result_paths=paths)


def check_admission(repo, selected):
    report = repo / 'reports' / NAME
    prereg = read(report / 'PREREGISTRATION_COMMIT.json')
    admission = read(report / 'ADMISSION.json')
    clock = read(report / 'EXECUTION_CLOCK.json')
    commit = prereg['commit']
    require(admission.get('status') == 'PASS' and admission['commit'] == commit, 'Admission/prereg commit mismatch')
    actual_time = int(git(repo, 'show', '-s', '--format=%ct', commit).strip())
    require(actual_time == prereg['commit_unix'], 'Recorded prereg commit time differs from Git')
    for ancestor, descendant in ((BASE, commit), (commit, 'HEAD')):
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=repo,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    committed_files = {}
    for row in prereg['files']:
        name = safe_relative(row['path']).as_posix()
        committed_files[name] = git(repo, 'show', f'{commit}:{name}')
        require(hashlib.sha256(committed_files[name]).hexdigest() == row['sha256'] and
                sha(repo / name) == row['sha256'], 'Preregistered bytes changed: ' + name)
    addendum_name = f'reports/{NAME}/PREREGISTRATION_ADDENDUM.json'
    require(addendum_name in committed_files, 'Committed registration JSON is missing from prereg receipt')
    committed_addendum = json.loads(committed_files[addendum_name].decode('utf-8-sig'))
    registered_sources = committed_addendum['new_source_hashes_at_registration']
    registered_hashes = {row['path']: row['sha256'] for row in registered_sources}
    require(len(registered_hashes) == len(registered_sources), 'Duplicate source in committed registration hash map')
    frozen = {f'research/{NAME}/{name}.py' for name in ('common', 'controller', 'job_worker', 'floors')}
    require({row['path'] for row in admission['new_sources']} == frozen and len(admission['new_sources']) == 4,
            'Frozen learning source inventory differs')
    for row in admission['new_sources']:
        name = safe_relative(row['path']).as_posix()
        require(sha(repo / name) == row['sha256'] and
                registered_hashes.get(name) == row['sha256'],
                'Frozen learning source differs from admission/committed addendum SHA: ' + name)
    paths = sorted((repo / '.cache' / NAME / 'worker_admissions').glob('*.json'))
    if not paths:
        raise Incomplete('No worker admission evidence')
    items = [read(path) for path in paths]
    expected = {job['job_id']: job for job in selected}
    require(len({(r['job']['job_id'], r['attempt']) for r in items}) == len(items), 'Duplicate worker admission')
    limit = min(clock['training_started_unix'] + 13 * 3600,
                datetime.fromisoformat('2026-10-28T00:00:00+09:00').timestamp())
    require(clock['limit_seconds'] == 13 * 3600 and clock['latest_start'] == '2026-10-28T00:00:00+09:00',
            'Original execution clock limits changed')
    for item in items:
        require(item['job'] == expected.get(item['job']['job_id']) and item['attempt'] in (1, 2),
                'Out-of-scope worker admission')
        require(actual_time < item['admitted_unix'] < limit, 'Worker admitted before prereg or after budget/cutoff')
    first = min(items, key=lambda r: r['admitted_unix'])
    require(first['admitted_unix'] == clock['training_started_unix'] and
            first['job']['job_id'] == clock['first_job_id'], 'First admission/clock binding differs')
    extension = [item for item in items if item['job']['method_id'] == 'A1']
    extension_check = dict(status='NOT_STARTED')
    if extension:
        extension_start = min(item['admitted_unix'] for item in extension)
        decisions = [report / 'A1_JENA_EXTENSION_ADMISSION.json'] + sorted(report.glob('A1_JENA_EXTENSION_ADMISSION_[0-9][0-9][0-9].json'))
        decisions = [read(path) for path in decisions if path.is_file()]
        before = [row for row in decisions if datetime.fromisoformat(row['checked_utc']).timestamp() <= extension_start]
        require(bool(before), 'A1 extension lacks prior whole-ten budget admission')
        latest = max(before, key=lambda row: datetime.fromisoformat(row['checked_utc']).timestamp())
        require(latest.get('allowed') is True and latest['n'] == 10 and
                latest['required_seconds'] <= latest['remaining_seconds'], 'A1 whole-ten budget admission failed')
        extension_check = dict(status='ADMITTED', n=10, required_seconds=latest['required_seconds'],
                               remaining_seconds=latest['remaining_seconds'])
    return dict(prereg_commit=commit, prereg_commit_unix=actual_time,
                first_admission_unix=first['admitted_unix'], delta_seconds=first['admitted_unix']-actual_time,
                worker_admissions=len(items), forbidden_admissions=0, extension=extension_check)


def validate_runs(run_rows, summary_rows, result_records, reuse_records):
    expected = {r['row']['job_id']: r['row'] for r in result_records}
    require(len(expected) == len(result_records), 'Duplicate expected RESULT rows')
    for row in reuse_records:
        require(row['job_id'] not in expected, 'Reused row duplicates RESULT')
        expected[row['job_id']] = row
    ids = [row['job_id'] for row in run_rows]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected), 'RUNS row count/identity set differs')
    require(sum(row['method_id'] == 'RAW' for row in run_rows) == 4 and
            sum(row['method_id'] == 'OURS' for row in run_rows) == 40, 'RAW/OURS reuse count differs')
    for row in run_rows:
        source = expected[row['job_id']]
        require(row['dataset'] == source['dataset'] and row['method_id'] == source['method_id'] and
                int(row['seed']) == int(source['seed']), 'RUNS identity differs: ' + row['job_id'])
        if 'extended' in source:
            require(flag(row['extended']) == flag(source['extended']), 'RUNS extension flag differs from RESULT')
        if source.get('K') is not None:
            require(int(row['K']) == int(source['K']), 'RUNS K differs from receipt')
        mse = float(row['MSE'])
        expected_mse = source.get('MSE', source.get('mse_recomputed'))
        require(math.isfinite(mse) and abs(mse-float(expected_mse)) <= 1e-9, 'RUNS MSE differs from public receipt')
    summaries = {(row['dataset'], row['method_id']): row for row in summary_rows}
    require(len(summaries) == len(summary_rows) == 40 and
            set(summaries) == {(d, m) for d in DATASETS for m in METHODS}, 'SUMMARY group set differs')
    primary = []
    for dataset in DATASETS:
        for method in METHODS:
            group = [r for r in run_rows if (r['dataset'], r['method_id']) == (dataset, method)]
            base = [r for r in group if not flag(r['extended'])]
            extension = [r for r in group if flag(r['extended'])]
            for part in (base, extension):
                require(len(part) == len({int(r['seed']) for r in part}), 'Duplicate seed in RUNS stage')
            expected_seeds = {92601} if method in ('RAW', 'C1') else SEEDS
            extension_complete = {int(r['seed']) for r in extension} == expected_seeds
            chosen = extension if extension_complete else base
            chosen_ids = {r['job_id'] for r in chosen}
            require(all(flag(r['is_primary']) == (r['job_id'] in chosen_ids) for r in group),
                    'Primary extension/base selection violates complete-seed rule: ' + dataset + '/' + method)
            summary = summaries[dataset, method]
            required = method not in ('RAW', 'C1') and len(base) == 10 and sum(flag(r['cap_selected']) for r in base) >= 5
            require(int(summary['n']) == len(chosen) and int(summary['expected_n']) == len(expected_seeds) and
                    flag(summary['complete']) == ({int(r['seed']) for r in chosen} == expected_seeds) and
                    flag(summary['extended']) == extension_complete and
                    int(summary['extension_n']) == len(extension) and
                    flag(summary['provisional']) == (required and not extension_complete) and
                    flag(summary['undertrained_at_10']) == (extension_complete and sum(flag(r['cap_selected']) for r in chosen) >= 5),
                    'SUMMARY primary/completeness/provisional flags differ: ' + dataset + '/' + method)
            if chosen:
                require(abs(float(summary['MSE_median'])-statistics.median(float(r['MSE']) for r in chosen)) <= 1e-12,
                        'SUMMARY median differs: ' + dataset + '/' + method)
            if method == 'A1' and dataset == 'jena':
                primary.append(dict(dataset=dataset, method_id=method, n=len(chosen), extension_n=len(extension),
                                    extension_primary=extension_complete))
    return dict(run_rows=len(run_rows), result_rows=len(result_records), reused_RAW=4, reused_OURS=40,
                duplicate_rows=0, A1_jena_primary=primary[0])


def check_report_seal(directory, expected_results, stage):
    required = set(REGISTERED_ARTIFACTS) | set(UPDATED_ARTIFACTS)
    required.update('registered_analysis/' + name for name in REGISTERED_ARTIFACTS)
    if stage == 'final':
        required.update(FINAL_FLOORS)
    missing = [name for name in sorted(required) if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    if missing:
        raise Incomplete('Missing final/interim artifacts: ' + ', '.join(str(directory / name) for name in missing))
    receipt = read(directory / 'REPORTING_COMPLETION.json')
    require(receipt.get('status') == 'COMPLETE' and receipt.get('stage') == stage, 'Invalid reporting completion receipt')
    for name, expected in receipt['outputs'].items():
        path = directory / safe_relative(name)
        require(path.is_file() and sha(path) == expected, 'Report artifact SHA changed: ' + str(path))
    require(required - {'REPORTING_COMPLETION.json'} <= set(receipt['outputs']), 'Required artifact missing from report seal')
    digest = hashlib.sha256(json.dumps(expected_results, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    require(receipt['input_result_count'] == len(expected_results) and receipt['input_results_sha256'] == digest,
            'Reporting input snapshot differs')
    return dict(stage=stage, required_artifacts=len(required), sealed_artifacts=len(receipt['outputs']))


def check_reports(repo, context):
    report, old_report = repo / 'reports' / NAME, repo / 'reports' / OLD_NAME
    reuse = read(old_report / 'G1_REUSE_AUDIT.json')['runs']
    require(len(reuse) == 44 and sum(r['method_id'] == 'RAW' for r in reuse) == 4 and
            sum(r['method_id'] == 'OURS' for r in reuse) == 40, 'Original reuse audit counts differ')
    final = check_report_seal(report, context['results'], 'final')
    counts = validate_runs(rows(report / 'RUNS.csv'), rows(report / 'SUMMARY.csv'), context['results'], reuse)
    require(counts['run_rows'] == 208 + len(context['new']), 'Final RUNS must equal old164 + RAW4 + OURS40 + new n')
    a4 = [r for r in context['new'] if r['job']['method_id'] == 'A4']
    interim = dict(status='NOT_APPLICABLE_A4_INCOMPLETE', A4_n=len(a4), expected_n=40)
    if len(a4) == 40:
        inputs = [r for r in context['results'] if r['job']['job_id'] in context['old_ids'] or r['job']['method_id'] == 'A4']
        interim = check_report_seal(report / 'interim_A4', inputs, 'interim')
        interim['runs'] = validate_runs(rows(report / 'interim_A4/RUNS.csv'), rows(report / 'interim_A4/SUMMARY.csv'), inputs, reuse)
        require(not interim['runs']['A1_jena_primary']['extension_primary'], 'Interim wrongly uses A1 extension')
    return dict(final=final, final_runs=counts, interim=interim)


def archive_public_path(name):
    path = safe_relative(name)
    forbidden = {'.cache', '__pycache__', 'data', 'datasets', 'raw_data', 'raw_arrays', 'checkpoint', 'checkpoints'}
    require(not {part.lower() for part in path.parts}.intersection(forbidden), 'Private/data archive path: ' + name)
    require(path.suffix.lower() in {'.json', '.csv', '.md', '.txt', '.png', '.svg', '.py', '.log'},
            'Forbidden/nonpublic archive extension: ' + name)
    require(not path.name.lower().startswith(('selected_bundle', 'prediction_array', 'prefix_physical')),
            'Private array/checkpoint archive member: ' + name)
    permitted_roots = (f'reports/{NAME}/', f'reports/{OLD_NAME}/',
                       f'research/{NAME}/', f'research/{OLD_NAME}/', 'references/')
    require(any(name.startswith(root) for root in permitted_roots), 'Outside public campaign roots: ' + name)


def required_archive_members(repo, context):
    report = repo / 'reports' / NAME
    names = set()
    directories = [report]
    if context is not None and sum(r['job']['method_id'] == 'A4' for r in context['new']) == 40:
        directories.append(report / 'interim_A4')
    for directory in directories:
        receipt = read(directory / 'REPORTING_COMPLETION.json')
        names.add((directory / 'REPORTING_COMPLETION.json').relative_to(repo).as_posix())
        names.update((directory / safe_relative(name)).relative_to(repo).as_posix() for name in receipt['outputs'])
    for name in ('ADMISSION.json', 'EXECUTION_PLAN.json', 'EXECUTION_COMPLETION.json', 'EXECUTION_CLOCK.json',
                 'ENVIRONMENT_VALIDATION.json', 'GATES.json', 'PRESERVATION_BASELINE.json', 'BUDGET.json',
                 'PREREGISTRATION_COMMIT.json', 'PREREGISTRATION_ADDENDUM.json', 'PREREGISTRATION_ADDENDUM.md'):
        names.add(f'reports/{NAME}/{name}')
    for namespace in (NAME, OLD_NAME):
        names.update(path.relative_to(repo).as_posix() for path in (repo / 'research' / namespace).glob('*.py'))
    for path in (report / 'preflight').glob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            names.add(path.relative_to(repo).as_posix())
    if context is not None:
        for job in context['selected']:
            directory = repo / 'reports' / OLD_NAME / 'jobs' / job['job_id']
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.txt', '.log', '.png', '.svg'):
                    names.add(path.relative_to(repo).as_posix())
    names.add('references/comparative_ablation_followup_executor_instructions.md')
    return names


def check_zip(target, manifest_path, bundle_receipt_path, repo, required_names=(), instruction_sha256=None):
    if not target.is_file():
        raise Incomplete('Final ZIP is absent: ' + str(target))
    manifest = read(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_name = f'reports/{NAME}/BUNDLE_MANIFEST.json'
    files = manifest['files']
    expected = {row['path']: row for row in files}
    require(len(files) == len(expected), 'Duplicate manifest entry')
    require(manifest_name not in expected, 'Manifest recursively lists itself')
    expected[manifest_name] = dict(bytes=len(manifest_bytes), sha256=hashlib.sha256(manifest_bytes).hexdigest())
    with zipfile.ZipFile(target) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) and set(names) == set(expected), 'ZIP exact entry set differs')
        require(set(required_names) <= set(names), 'Required public evidence absent from ZIP: ' +
                ', '.join(sorted(set(required_names)-set(names))))
        require(archive.testzip() is None, 'ZIP CRC check failed')
        for info in infos:
            name = info.filename
            archive_public_path(name)
            require(not info.is_dir() and not info.flag_bits & 1 and
                    not stat.S_ISLNK(info.external_attr >> 16), 'Directory/encrypted/symlink archive member')
            require(PurePosixPath(name).name not in ('BUNDLE_RECEIPT.json', RECEIPT_NAME),
                    'Post-ZIP external receipt must not be archived')
            digest = hashlib.sha256()
            count = 0
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    count += len(chunk)
                    digest.update(chunk)
            require(count == expected[name]['bytes'] and digest.hexdigest() == expected[name]['sha256'],
                    'ZIP entry SHA/size mismatch: ' + name)
            if name == 'references/comparative_ablation_followup_executor_instructions.md' and instruction_sha256 is not None:
                require(digest.hexdigest() == instruction_sha256, 'Archived instruction differs from admission')
            if not name.startswith('references/'):
                local = repo / safe_relative(name)
                require(local.is_file() and local.stat().st_size == count and sha(local) == digest.hexdigest(),
                        'ZIP entry differs from current public evidence: ' + name)
    receipt = read(bundle_receipt_path)
    require(receipt['sha256'] == sha(target) and receipt['bytes'] == target.stat().st_size and
            receipt['entries'] == len(expected), 'External bundle receipt differs')
    require(receipt.get('manifest_sha256') == hashlib.sha256(manifest_bytes).hexdigest(), 'External manifest SHA differs')
    return dict(path=str(target), entries=len(expected), zip_crc='PASS', exact_entry_set='PASS',
                all_entry_sha256='PASS', archive_sha256=sha(target), forbidden_members=0)


def verify(repo):
    repo = Path(repo).resolve()
    checks = {}
    context = None

    def check(name, call):
        try:
            detail = call()
            checks[name] = dict(status='PASS', **detail)
            return detail
        except Incomplete as exc:
            checks[name] = dict(status='INCOMPLETE', error=str(exc))
        except RootAction as exc:
            checks[name] = dict(status='ROOT_ACTION_REQUIRED', error=str(exc))
        except Exception as exc:
            checks[name] = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}')

    check('preservation_baseline', lambda: check_preservation(repo))
    check('original_git_blobs', lambda: check_git_blobs(repo))
    try:
        context = result_context(repo)
        checks['job_scope_and_T_proofs'] = dict(status='PASS', **context['detail'], T_numeric_values_parsed=0)
    except Incomplete as exc:
        checks['job_scope_and_T_proofs'] = dict(status='INCOMPLETE', error=str(exc))
    except RootAction as exc:
        checks['job_scope_and_T_proofs'] = dict(status='ROOT_ACTION_REQUIRED', error=str(exc))
    except Exception as exc:
        checks['job_scope_and_T_proofs'] = dict(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    selected = context['selected'] if context else None
    if selected is None:
        try:
            selected = selected_jobs(read(repo / 'reports' / OLD_NAME / 'RUN_PLAN.json'))
        except Exception:
            pass
    if selected is not None:
        check('preregistration_and_admission_timing', lambda: check_admission(repo, selected))
    else:
        checks['preregistration_and_admission_timing'] = dict(status='INCOMPLETE', error='Selected job context unavailable')
    if context is not None:
        check('reports_counts_primary_and_artifacts', lambda: check_reports(repo, context))
    else:
        checks['reports_counts_primary_and_artifacts'] = dict(status='INCOMPLETE', error='Terminal valid result context unavailable')
    report = repo / 'reports' / NAME
    check('independent_zip', lambda: check_zip(report / f'{NAME}_bundle.zip', report / 'BUNDLE_MANIFEST.json',
        report / 'BUNDLE_RECEIPT.json', repo, required_archive_members(repo, context),
        read(report / 'ADMISSION.json')['instruction_sha256']))
    statuses = {row['status'] for row in checks.values()}
    status = ('FAIL' if 'FAIL' in statuses else 'ROOT_ACTION_REQUIRED' if 'ROOT_ACTION_REQUIRED' in statuses else
              'INCOMPLETE' if 'INCOMPLETE' in statuses else 'PASS')
    return dict(status=status, scope='independent machine checks only',
        campaign_completion_claim=False, execution_status=context['completion']['status'] if context else None,
        checked_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
        T_numeric_values_parsed_by_this_verifier=0,
        data_access='Public JSON/CSV aggregates and opaque file bytes for SHA/Git/ZIP; no numerical array loading.',
        remaining_root_responsibilities=['Inspect final figures visually', 'Create authorized final local commit',
                                         'Deliver results to the user'],
        external_receipt_note='Created after ZIP; this independent receipt is intentionally outside the ZIP.',
        verifier_source_sha256=sha(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Print checks without writing a receipt')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    target = repo / 'reports' / NAME / RECEIPT_NAME
    if not args.check_only and target.exists():
        raise FileExistsError('Independent receipt already exists; use --check-only: ' + str(target))
    receipt = verify(repo)
    if not args.check_only:
        with target.open('x', encoding='utf8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if receipt['status'] == 'PASS' else 2 if receipt['status'] == 'FAIL' else 1


if __name__ == '__main__':
    raise SystemExit(main())
