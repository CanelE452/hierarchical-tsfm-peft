"""Post-seal, CPU-only independent audit. Does not import campaign production code.

Run only after terminal execution and production analysis. This supplemental
verifier changes no training, selection, aggregation or publication decision.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

NAME = 'p_branch_appl_ett_20261029_v1'
DATASETS = ('appliances', 'ETTh1', 'ETTh2_safe17400')
METHODS = ('P_TIME_ONLY', 'P_CHANNEL_ONLY')
ORDER = ('A1', *METHODS, 'OURS', 'A2')
SEEDS = tuple(range(92601, 92611))
AXES = dict(appliances=(10, 26, 41, 13814, 15788, 11841),
            ETTh1=(4, 7, 36, 12194, 13936, 10452),
            ETTh2_safe17400=(4, 7, 36, 12180, 13920, 10440))
PARAMETERS = dict(appliances=(37114, 35392, 3802), ETTh1=(34023, 33536, 711),
                  ETTh2_safe17400=(34023, 33536, 711))
REFERENCE_EFFECTS = dict(appliances=dict(A1=.6, A2=-1.6), ETTh1=dict(A1=-.1, A2=-.8),
                         ETTh2_safe17400=dict(A1=-5., A2=.5))
TERMINAL = {'TRAINING_COMPLETE', 'TRAINING_INCOMPLETE', 'SCIENTIFIC_SUSPEND',
            'UNKNOWN_SUSPEND', 'USER_CANCELLED', 'STOPPED'}
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
ARRAY_KEYS = ('pred_normalized', 'target_normalized', 'observed', 'origins',
              'channel_ids', 'mean', 'std')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def truth(value):
    require(value in (True, False, 'True', 'False', 'true', 'false'), 'Invalid boolean')
    return str(value).lower() == 'true'


def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    require(bool(ordered) and all(math.isfinite(x) for x in ordered), 'Empty/nonfinite quantile')
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def expected_jobs():
    return [dict(job_id=f'PBA_{method}_{dataset}_K{AXES[dataset][0]}_s{seed}',
                 method_id=method, dataset=dataset, K=AXES[dataset][0], seed=seed, extended=False)
            for seed in SEEDS for dataset in DATASETS for method in METHODS]


def clean_row(source, reused):
    row = dict(source)
    require(row['dataset'] in DATASETS and row['method_id'] in ORDER, 'Unknown result axes')
    for key in ('seed', 'K', 'trainable_parameters', 'normalization_fit_end'):
        value = float(row[key])
        require(math.isfinite(value) and value.is_integer(), 'Noninteger ' + key)
        row[key] = int(value)
    for key in ('MSE', 'MAE', 'selected_epoch', 'training_cap'):
        row[key] = float(row[key])
        require(math.isfinite(row[key]), 'Nonfinite ' + key)
    require(row['MSE'] >= 0 and row['MAE'] >= 0, 'Negative error metric')
    require(row['seed'] in SEEDS and row['K'] == AXES[row['dataset']][0], 'Seed/K mismatch')
    require(row['normalization_fit_end'] == AXES[row['dataset']][5], 'Scaler fit endpoint mismatch')
    require(not truth(row['extended']) and row['training_cap'] == 3., 'Unexpected extension/cap')
    require(row['selected_epoch'] in (0., .25, 1., 3.), 'Unregistered selected epoch')
    row['extended'], row['reused'] = False, reused
    row['cap_selected'] = row['selected_epoch'] == 3.
    if row['method_id'] in ('OURS', *METHODS):
        index = ('OURS', *METHODS).index(row['method_id'])
        require(row['trainable_parameters'] == PARAMETERS[row['dataset']][index], 'Parameter count mismatch')
    return row


def check_archive(row, cohort=None):
    dataset = row['dataset']
    _, channels, count, start, end, _ = AXES[dataset]
    shape = (count, 48, channels)
    path = Path(row['prediction_path'])
    require(sha(path) == row['prediction_sha256'], 'NPZ SHA mismatch: ' + row['job_id'])
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate NPZ member')
        for key in ARRAY_KEYS:
            require(key + '.npy' in names, 'Missing NPZ metadata: ' + key)
            with archive.open(key + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                actual, fortran, dtype = reader(stream)
            expected = shape if key in ARRAY_KEYS[:3] else (count,) if key == 'origins' else (channels,)
            require(actual == expected and not fortran and not dtype.hasobject, 'Unsafe numeric header: ' + key)
            kind = 'b' if key == 'observed' else 'iu' if key == 'origins' else 'SU' if key == 'channel_ids' else 'f'
            require(dtype.kind in kind, 'NPZ dtype mismatch: ' + key)
    with np.load(path, allow_pickle=False) as archive:
        origins = archive['origins'].copy()
        require(np.array_equal(origins, np.arange(start, end - 48 + 1, 48)), 'V origin grid crosses or differs from protected T boundary')
        require(int(origins[-1]) + 48 <= end, 'Protected T overlap')
        arrays = {key: archive[key].copy() for key in ARRAY_KEYS if key != 'origins'}
    arrays.update(origins=origins, dataset=dataset)
    require(np.isfinite(arrays['mean']).all() and np.isfinite(arrays['std']).all()
            and (arrays['std'] > 0).all(), 'Nonfinite/nonpositive normalization')
    if cohort is not None:
        require(cohort['dataset'] == dataset, 'Cross-dataset cohort')
        for key in ARRAY_KEYS[1:]:
            actual, expected = arrays[key], cohort[key]
            require(np.array_equal(actual, expected, equal_nan=actual.dtype.kind == 'f'), 'Cohort/scaler mismatch: ' + key)
    pred, target, mask = (arrays[key] for key in ARRAY_KEYS[:3])
    require(np.isfinite(pred).all() and np.isfinite(target[mask]).all(), 'Nonfinite scored numeric values')
    channel_mse, channel_mae = [], []
    for channel in range(channels):
        selected = mask[..., channel]
        require(selected.any(), 'All-false scored channel')
        errors = pred[..., channel][selected].astype(np.float64) - target[..., channel][selected].astype(np.float64)
        channel_mse.append(math.fsum(float(x) ** 2 for x in errors) / len(errors))
        channel_mae.append(math.fsum(abs(float(x)) for x in errors) / len(errors))
    mse, mae = math.fsum(channel_mse) / channels, math.fsum(channel_mae) / channels
    require(abs(mse - float(row['MSE'])) <= 1e-9 and abs(mae - float(row['MAE'])) <= 1e-9,
            'Independent FP64 metric mismatch: ' + row['job_id'])
    return arrays, dict(job_id=row['job_id'], dataset=dataset, archive_sha256=row['prediction_sha256'],
                        MSE=mse, MAE=mae, shape=list(shape), T_numeric_values_parsed=0)


def summarize(rows):
    require(len({(r['dataset'], r['method_id'], r['seed']) for r in rows}) == len(rows), 'Duplicate result key')
    summaries, contrasts, paired = [], [], []
    for dataset in DATASETS:
        for method in ORDER:
            group = sorted((r for r in rows if (r['dataset'], r['method_id']) == (dataset, method)), key=lambda r: r['seed'])
            primary = tuple(r['seed'] for r in group) == SEEDS
            cap = sum(r['selected_epoch'] == 3. for r in group)
            for row in group:
                row.update(primary=primary, provisional=primary and cap >= 5)
            q1, median, q3 = (quantile([r['MSE'] for r in group], p) for p in (.25, .5, .75)) if group else (None, None, None)
            parameters = group[0]['trainable_parameters'] if group else PARAMETERS[dataset][('OURS', *METHODS).index(method)]
            summaries.append(dict(dataset=dataset, method_id=method, n=len(group), primary=primary,
                status='COMPLETE' if primary else 'INCOMPLETE', MSE_median=median, MSE_Q1=q1, MSE_Q3=q3,
                MSE_IQR=None if not group else q3-q1, MAE_median=quantile([r['MAE'] for r in group], .5) if group else None,
                parameters=parameters, cap=3., extended=False, cap_selected_count=cap, provisional=primary and cap >= 5,
                extension_status='NOT_RUN_SCOPE' if primary and cap >= 5 else 'NOT_REQUIRED' if primary else 'BASE_INCOMPLETE',
                selected_epochs=json.dumps([r['selected_epoch'] for r in group])))
    lookup = {(r['dataset'], r['method_id']): r for r in summaries}
    for dataset in DATASETS:
        ours = {r['seed']: r for r in rows if r['dataset'] == dataset and r['method_id'] == 'OURS'}
        for method in (*METHODS, 'A1', 'A2'):
            variant, full = lookup[dataset, method], lookup[dataset, 'OURS']
            primary = variant['primary'] and full['primary']
            deltas = []
            for row in sorted((r for r in rows if (r['dataset'], r['method_id']) == (dataset, method)), key=lambda r: r['seed']):
                delta = row['MSE'] - ours[row['seed']]['MSE']
                deltas.append(delta)
                paired.append(dict(dataset=dataset, method_id=method, seed=row['seed'], variant_MSE=row['MSE'],
                    full_P_MSE=ours[row['seed']]['MSE'], delta_MSE=delta, primary=primary))
            gap = variant['MSE_median'] - full['MSE_median'] if primary else None
            spread = max(variant['MSE_IQR'], full['MSE_IQR']) if primary else None
            provisional = variant['provisional'] or full['provisional']
            decision = ('FULL_P_LOWER' if gap > spread else 'VARIANT_LOWER' if gap < -spread else 'UNCLEAR') if primary else 'INCOMPLETE'
            if primary and provisional:
                decision += '_PROVISIONAL'
            contrasts.append(dict(dataset=dataset, method_id=method, primary=primary, delta_median_MSE=gap,
                effect_pct=100*gap/full['MSE_median'] if primary else None, spread=spread,
                descriptive_decision=decision, paired_delta_median=quantile(deltas, .5) if primary else None,
                seeds_full_P_lower=sum(d > 0 for d in deltas), n=variant['n'], full_P_n=full['n'],
                cap_selected_count=variant['cap_selected_count'], full_P_cap_selected_count=full['cap_selected_count'], provisional=provisional))
    return summaries, contrasts, paired


def compare_csv(path, expected, keys, fields):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(set(keys + fields).issubset(reader.fieldnames or []), 'CSV schema missing: ' + Path(path).name)
        actual = list(reader)
    def identity(row):
        return tuple(str(row[key]) for key in keys)
    require(len(actual) == len(expected) and len({identity(row) for row in actual}) == len(actual), 'CSV row count/duplicate mismatch: ' + Path(path).name)
    lookup = {identity(row): row for row in actual}
    require(set(lookup) == {identity(row) for row in expected}, 'CSV result keys mismatch: ' + Path(path).name)
    for row in expected:
        for field in fields:
            value, observed = row[field], lookup[identity(row)][field]
            if value is None:
                valid = observed == ''
            elif isinstance(value, bool):
                valid = truth(observed) == value
            elif isinstance(value, (float, int)):
                valid = math.isfinite(float(observed)) and math.isclose(float(observed), value, rel_tol=1e-12, abs_tol=1e-12)
            elif field == 'selected_epochs':
                valid = json.loads(observed) == json.loads(value)
            else:
                valid = observed == value
            require(valid, f'CSV mismatch {Path(path).name}: {identity(row)} / {field}')


def check_slide(path, contrasts):
    lines = Path(path).read_text(encoding='utf8').splitlines()
    for row in contrasts:
        if row['method_id'] not in METHODS:
            continue
        found = [line.split() for line in lines if line.split()[:2] == [row['method_id'], row['dataset']]]
        require(len(found) == 1, 'Slide row missing or duplicated')
        effect = f"{row['effect_pct']:+.1f}%" if row['primary'] else 'INCOMPLETE'
        require(found[0][2:] == [effect, row['descriptive_decision'], f"{row['cap_selected_count']}/10", str(row['n'])], 'Slide signed effect/cap/status mismatch')


def check_job(report, job, registration, protocol, completion, data_registry, bound):
    root = report / 'jobs' / job['job_id']
    result = read(root / 'RESULT.json')
    require(result['status'] == 'COMPLETE' and result['job'] == job and result['registration'] == registration,
            'Result job/registration mismatch')
    attempt = result['attempt']
    require(type(attempt) is int and attempt > 0, 'Invalid attempt')
    public = root / f'attempt_{attempt:03d}'
    names = ['START.json', 'SOURCE_CONTRACT.json', 'SEARCH_RESULT.json', 'SELECTION.json', 'REFIT_SUMMARY.json', 'RECEIPT.json', 'OPENED_FILES.json']
    evidence = {name: read(public / name) for name in names}
    worker_path = root / f'WORKER_RECEIPT_attempt{attempt}.json'
    worker = read(worker_path)
    require(worker['status'] == 'COMPLETE' and worker['exit_code'] == 0 and worker['job_id'] == job['job_id'] and worker['attempt'] == attempt,
            'Successful worker receipt mismatch')
    require(datetime.fromisoformat(registration['registered_at_utc']).timestamp() <= worker['started_unix'], 'Worker predates registration')
    authorization = result['attempt_authorization']
    if attempt <= 2:
        require(authorization == dict(kind='BASE', attempt=attempt), 'Base authorization mismatch')
    else:
        path = Path(authorization['authorization_path'])
        require(path.parent.resolve() == report.resolve() and sha(path) == authorization['authorization_sha256'], 'Recovery authorization path/SHA mismatch')
        approval = read(path)
        source_hashes = {f'research/{NAME}/{name}': protocol['registered_sources'][f'research/{NAME}/{name}'] for name in ('models.py', 'runtime.py', 'run_one.py', 'controller.py')}
        require(approval['status'] == 'APPROVED_TECHNICAL_RECOVERY' and approval['job_id'] == job['job_id']
                and approval['allowed_attempts'] == [attempt if attempt % 2 else attempt-1, attempt+1 if attempt % 2 else attempt]
                and approval['scientific_source_hashes'] == source_hashes and approval['registration_seal_sha256'] == registration['seal_sha256']
                and datetime.fromisoformat(approval['created_utc']).timestamp() <= worker['started_unix'], 'Recovery approval mismatch')
        bound[path] = sha(path)
    spec = result['spec']
    required_spec = dict(run_id=job['job_id'], dataset=job['dataset'], method_id=job['method_id'], family='B2',
        latent_k=job['K'], seed=job['seed'], init_kind='PCA', revin=False, device='cuda', fold='primary', core_dtype='float32',
        lr=.001, effective_batch=16, microbatch=4, F_row_microbatch=128, checkpoint_inputs=False, epochs=[.25, 1., 3.])
    require(all(spec.get(k) == v for k, v in required_spec.items()), 'Numerical recipe mismatch')
    expected_sources = {f'research/{NAME}/{name}': protocol['registered_sources'][f'research/{NAME}/{name}']
                        for name in ('models.py', 'runtime.py', 'run_one.py', 'controller.py')}
    require(spec['new_scientific_source_hashes'] == expected_sources
            and spec['custom_source_sha256'] == expected_sources[f'research/{NAME}/models.py'], 'Worker source hashes mismatch')
    start, receipt = evidence['START.json'], evidence['RECEIPT.json']
    for item in (start, receipt):
        require(item['job'] == job and item['attempt'] == attempt and item['registration'] == registration, 'Attempt identity mismatch')
    require(start['spec'] == spec and receipt['status'] == 'COMPLETE' and receipt['training_attempted'] is True
            and receipt['training_executed'] is True and receipt['T_numeric_values_parsed'] == 0
            and receipt['preserved_before']['changed'] == receipt['preserved_after']['changed'] == [], 'Attempt preservation/training receipt mismatch')
    require(evidence['OPENED_FILES.json']['T_numeric_values_parsed'] == 0, 'Opened-file T audit mismatch')
    require(result['task_access'] == data_registry[job['dataset']]['access'], 'Bounded pre-T data access mismatch')
    contract, source = result['contract'], evidence['SOURCE_CONTRACT.json']
    require(contract['model_sha'] == F_SHA and source['spec'] == spec
            and all(source[k] == v for k, v in contract.items()), 'Foundation/source contract mismatch')
    selection = evidence['SELECTION.json']
    require(Path(result['selection_path']).resolve() == (public / 'SELECTION.json').resolve()
            and sha(public / 'SELECTION.json') == result['selection_sha256'], 'Selection path/SHA mismatch')
    search = evidence['SEARCH_RESULT.json']
    curves = [(float(r['epoch']), float(r['astop_mse'])) for r in search['curves']]
    require([e for e, _ in curves] == [0., .25, 1., 3.] and all(math.isfinite(v) for _, v in curves), 'Incomplete/nonfinite selection curve')
    selected, loss = min(curves, key=lambda pair: (pair[1], pair[0]))
    require(selection['status'] == 'SEALED' and selection['job'] == job and selection['V_used'] is False
            and selection['selection_split'] == 'A_stop' and selection['spec'] == spec and selection['contract'] == contract
            and selection['curves'] == search['curves'] and selection['selected_epoch'] == selected
            and selection['selected_astop_mse'] == loss and result['selection_sealed_before_V'] is True
            and result['search_summary'] == search['summary'] and search['summary']['status'] == 'SEGMENT_COMPLETE', 'A_stop selection mismatch')
    require(evidence['REFIT_SUMMARY.json']['status'] == 'SEGMENT_COMPLETE'
            and result['refit_summary'] == evidence['REFIT_SUMMARY.json'], 'Fresh refit mismatch')
    row = clean_row(result['row'], False)
    require(all(row[k] == v for k, v in job.items()) and row['selected_epoch'] == selected, 'Selected row mismatch')
    require(sha(result['bundle_path']) == result['bundle_sha256'], 'Selected bundle SHA mismatch')
    recorded = [r for r in completion.get('attempts', {}).get(job['job_id'], []) if r.get('attempt') == attempt]
    require(any(r.get('exit_code') == 0 and r.get('job_id') == job['job_id'] for r in recorded)
            and all(r.get('exit_code') == 0 for r in recorded), 'Controller successful attempt binding mismatch')
    require(Path(result['bundle_path']).resolve().is_relative_to(report.parents[1] / '.cache' / NAME)
            and Path(row['prediction_path']).resolve().is_relative_to(report.parents[1] / '.cache' / NAME), 'New numeric artifact outside campaign')
    for path in [root / 'RESULT.json', worker_path, *(public / name for name in names)]:
        bound[path] = sha(path)
    return row


def check_rejection(repo, report, job, status, bound):
    root = report / 'jobs' / job['job_id']
    path = root / 'RESULT_REJECTION.json'
    if not path.exists():
        require(not list((root / 'rejected').glob('*.json')), 'Orphan rejected evidence')
        return dict(rejected_result=False, rejection_reason='', rejected_result_sha256='')
    value = read(path)
    digest, attempt = value['rejected_result_sha256'], value['attempt']
    require(type(attempt) is int and attempt >= 0 and len(digest) == 64, 'Rejected attempt/digest mismatch')
    destination = root / 'rejected' / f'RESULT_attempt{attempt:03d}_{digest[:12]}.json'
    require(status != 'COMPLETE' and value['status'] == 'REJECTED_SCIENTIFIC_EVIDENCE'
            and value['job'] == job and value['run_status'] == status
            and value['trigger'] == 'RESULT_WRITTEN_THEN_TERMINAL_VALIDATION_REJECTED'
            and str(value['reason']).strip() and value['rejected_result_path'] == destination.relative_to(repo).as_posix()
            and value['original_result_path'] == (root / 'RESULT.json').relative_to(repo).as_posix()
            and sha(destination) == digest, 'Rejected result identity/hash mismatch')
    worker = root / f'WORKER_RECEIPT_attempt{attempt}.json'
    if worker.exists():
        require(value['worker_receipt_path'] == worker.relative_to(repo).as_posix()
                and value['worker_receipt_sha256'] == sha(worker), 'Rejected worker receipt mismatch')
        bound[worker] = sha(worker)
    else:
        require(value.get('worker_receipt_missing') is True, 'Missing rejected worker receipt not disclosed')
    bound[path], bound[destination] = sha(path), digest
    return dict(rejected_result=True, rejection_reason=value['reason'], rejected_result_sha256=digest)


def audit(repo, verify_preservation=False):
    repo = Path(repo).resolve()
    report = repo / 'reports' / NAME
    completion = read(report / 'EXECUTION_COMPLETION.json')
    require(completion.get('status') in TERMINAL and completion.get('active') is None, 'Terminal execution required before reading any NPZ')
    plan = read(report / 'RUN_PLAN.json')
    jobs = expected_jobs()
    require(plan['jobs'] == jobs and not plan.get('extension_jobs') and not plan.get('optional_jobs'), 'Exact sixty-job roster/order mismatch')
    require(set(completion['run_statuses']) == {j['job_id'] for j in jobs}, 'Terminal disposition must cover sixty jobs')
    protocol, seal = read(report / 'PREREGISTRATION.json'), read(report / 'REGISTRATION_SEAL.json')
    require(set(seal['files']) == set(protocol['registration_documents']), 'Sealed document roster mismatch')
    bound = {report / 'EXECUTION_COMPLETION.json': sha(report / 'EXECUTION_COMPLETION.json'), report / 'REGISTRATION_SEAL.json': sha(report / 'REGISTRATION_SEAL.json')}
    for relative, digest in seal['files'].items():
        path = (report / relative).resolve()
        require(path.is_relative_to(report) and sha(path) == digest, 'Registration document changed: ' + relative)
        bound[path] = digest
    for relative, digest in protocol['registered_sources'].items():
        path = (repo / relative).resolve()
        require(path.is_relative_to(repo) and sha(path) == digest, 'Registered source changed: ' + relative)
    registration = dict(seal_sha256=sha(report / 'REGISTRATION_SEAL.json'), registered_at_utc=seal['created_utc'], files=seal['files'])
    reference = read(report / 'REFERENCE_RUNS.json')
    require(sha(reference['source']) == reference['source_sha256'], 'Reference source CSV changed')
    rows = [clean_row(row, True) for row in reference['rows']]
    require(len(rows) == 90 and {(r['dataset'], r['method_id'], r['seed']) for r in rows}
            == {(d, m, s) for d in DATASETS for m in ('OURS', 'A1', 'A2') for s in SEEDS}, 'Ninety reference roster mismatch')
    registry = {r['dataset']: r for r in read(report / 'DATA_REGISTRY.json')['datasets']}
    new, rejection_status = [], {}
    for job in jobs:
        path = report / 'jobs' / job['job_id'] / 'RESULT.json'
        if completion['run_statuses'][job['job_id']] == 'COMPLETE':
            require(not (path.parent / 'RESULT_REJECTION.json').exists(), 'Complete result also rejected')
            new.append(check_job(report, job, registration, protocol, completion, registry, bound))
        else:
            require(not path.exists(), 'Unaccepted RESULT remained in numeric primary location')
        rejection_status[job['job_id']] = check_rejection(repo, report, job, completion['run_statuses'][job['job_id']], bound)
    require((completion['status'] == 'TRAINING_COMPLETE') == (len(new) == 60)
            and completion.get('completed_count', len(new)) == len(new), 'Completion count/status mismatch')
    rows += new
    cohorts, checks = {}, []
    for dataset in DATASETS:
        canonical = next(r for r in rows if r['dataset'] == dataset and r['method_id'] == 'OURS' and r['seed'] == 92603)
        cohorts[dataset], _ = check_archive(canonical)
    for row in rows:
        _, check = check_archive(row, cohorts[row['dataset']])
        checks.append(check)
    summaries, contrasts, paired = summarize(rows)
    for contrast in contrasts:
        if contrast['method_id'] in ('A1', 'A2'):
            require(round(contrast['effect_pct'], 1) == REFERENCE_EFFECTS[contrast['dataset']][contrast['method_id']], 'Known reference signed effect changed')
    keys = ['dataset', 'method_id']
    compare_csv(report / 'SUMMARY.csv', summaries, keys, [k for k in summaries[0] if k not in keys])
    compare_csv(report / 'CONTRASTS.csv', contrasts, keys, [k for k in contrasts[0] if k not in keys])
    compare_csv(report / 'PAIRED_DIFFERENCES.csv', paired, keys + ['seed'], ['variant_MSE', 'full_P_MSE', 'delta_MSE', 'primary'])
    row_fields = ['MSE', 'MAE', 'selected_epoch', 'training_cap', 'cap_selected', 'trainable_parameters', 'K', 'normalization_fit_end', 'reused', 'extended', 'primary', 'provisional', 'prediction_sha256']
    for filename, expected in [('RUNS.csv', [r for r in rows if r['primary']]), ('ALL_NEW_RUNS.csv', new)]:
        compare_csv(report / filename, expected, keys + ['seed'], row_fields if expected else ['MSE', 'MAE', 'primary', 'provisional'])
    statuses = [dict(j, status=completion['run_statuses'][j['job_id']], completed=completion['run_statuses'][j['job_id']] == 'COMPLETE', **rejection_status[j['job_id']]) for j in jobs]
    compare_csv(report / 'JOB_STATUS.csv', statuses, ['job_id'], ['dataset', 'method_id', 'K', 'seed', 'extended', 'status', 'completed', 'rejected_result', 'rejection_reason', 'rejected_result_sha256'])
    check_slide(report / 'SLIDE_NUMBERS.md', contrasts)
    for filename in ('SUMMARY.csv', 'CONTRASTS.csv', 'PAIRED_DIFFERENCES.csv', 'RUNS.csv', 'ALL_NEW_RUNS.csv', 'JOB_STATUS.csv', 'SLIDE_NUMBERS.md'):
        bound[report / filename] = sha(report / filename)
    preserved = None
    if verify_preservation:
        entries = read(report / 'PRESERVATION_BASELINE.json')['files']
        for item in entries:
            require(sha(item['path']) == item['sha256'], 'Opaque preservation SHA mismatch')
        preserved = len(entries)
    return dict(status='PASS', scope='POST_SEAL_INDEPENDENT_CPU_RESULT_AUDIT', created_utc=datetime.now(timezone.utc).isoformat(),
        scientific_conditions_changed=False, production_analysis_functions_imported=False, GPU_execution=False,
        T_numeric_values_parsed=0, completed_count=len(new), planned_count=60, checked_prediction_archives=len(checks),
        primary_new_count=sum(r['primary'] for r in new), archive_checks=checks, summaries=summaries, contrasts=contrasts,
        preservation_opaque_hashes_checked=preserved,
        bound_public_files={path.relative_to(repo).as_posix(): digest for path, digest in bound.items()},
        supplemental_source_sha256=sha(__file__), supplemental_test_sha256=sha(Path(__file__).parent / 'tests/test_independent.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--verify-preservation', action='store_true', help='Optional final opaque hashing; never parses protected data')
    parser.add_argument('--write-report', action='store_true', help='Explicit permission to write terminal INDEPENDENT_RESULT_CHECK.json')
    args = parser.parse_args()
    result = audit(args.repo, args.verify_preservation)
    if args.write_report:
        target = args.repo / 'reports' / NAME / 'INDEPENDENT_RESULT_CHECK.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
        temporary.replace(target)
    print(json.dumps({key: result[key] for key in ('status', 'completed_count', 'planned_count', 'checked_prediction_archives', 'primary_new_count', 'T_numeric_values_parsed')}))


if __name__ == '__main__':
    main()
