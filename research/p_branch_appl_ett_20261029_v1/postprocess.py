"""Verify registered multi-data P branches and report complete or partial evidence."""
from datetime import datetime, timezone
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from . import runtime, run_one

DATASETS = ('appliances', 'ETTh1', 'ETTh2_safe17400')
METHODS = ('P_TIME_ONLY', 'P_CHANNEL_ONLY')
ORDER = ('A1', 'P_TIME_ONLY', 'P_CHANNEL_ONLY', 'OURS', 'A2')
LABELS = dict(A1='No P', P_TIME_ONLY='Time branch only', P_CHANNEL_ONLY='Channel branch only',
              OURS='Full P', A2='A2: large channel replacement')
SEEDS = list(range(92601, 92611))
TERMINAL = {'TRAINING_COMPLETE', 'TRAINING_INCOMPLETE', 'SCIENTIFIC_SUSPEND', 'UNKNOWN_SUSPEND', 'USER_CANCELLED', 'STOPPED'}
V_STARTS = dict(appliances=13814, ETTh1=12194, ETTh2_safe17400=12180)
FIGURE = 'FIG_P_BRANCHES_APPL_ETT'
REFERENCE_EFFECTS = dict(appliances=dict(A1=0.6, A2=-1.6), ETTh1=dict(A1=-0.1, A2=-0.8),
                         ETTh2_safe17400=dict(A1=-5.0, A2=0.5))


def require(value, message):
    if not value:
        raise ValueError(message)


def write_csv(path, rows, fieldnames=None):
    keys = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    require(bool(keys), 'An empty CSV needs an explicit schema')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        handle.write(stream.getvalue())
    temporary.replace(path)


def normalized_row(row):
    out = dict(row)
    require(out.get('dataset') in DATASETS and out.get('method_id') in ORDER, 'Unexpected result axes')
    for name in ('seed', 'K', 'trainable_parameters', 'normalization_fit_end'):
        value = float(out[name])
        require(math.isfinite(value) and value.is_integer(), 'Noninteger result field: ' + name)
        out[name] = int(value)
    for name in ('MSE', 'MAE', 'selected_epoch', 'training_cap'):
        out[name] = float(out[name])
        require(math.isfinite(out[name]), 'Nonfinite result field: ' + name)
    for name in ('extended', 'reused'):
        out[name] = str(out.get(name, False)).lower() == 'true'
    require(out['seed'] in SEEDS and out['K'] == runtime.KS[out['dataset']], 'Result seed/K changed')
    require(out['normalization_fit_end'] == runtime.FIT_ENDS[out['dataset']][1], 'Result normalization range changed')
    require(not out['extended'] and out['training_cap'] == 3., 'Only base cap3 results are registered')
    require(out['selected_epoch'] in (0., .25, 1., 3.), 'Selected epoch outside registered schedule')
    out['cap_selected'] = out['selected_epoch'] == 3.
    expected = runtime.EXPECTED_PARAMETERS.get(out['dataset'], {}).get(out['method_id'])
    require(expected is None or out['trainable_parameters'] == expected, 'Trainable parameter count changed')
    out['display_name'] = LABELS[out['method_id']]
    return out


def array_header(archive, name):
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return reader(stream)


def verify_arrays(row, cohort=None):
    dataset = row['dataset']
    require(dataset in DATASETS, 'Unknown dataset in V archive')
    path = Path(row['prediction_path'])
    require(runtime.sha(path) == row['prediction_sha256'], 'Prediction archive changed: ' + row['job_id'])
    shape = tuple(runtime.V_SHAPES[dataset])
    channels, n = shape[2], shape[0]
    keys = ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')
    with zipfile.ZipFile(path) as archive:
        require(all(key + '.npy' in archive.namelist() for key in keys), 'Missing required normalization or V metadata')
        for key in keys:
            header_shape, fortran, dtype = array_header(archive, key)
            expected = shape if key in ('pred_normalized', 'target_normalized', 'observed') else (n,) if key == 'origins' else (channels,)
            require(header_shape == expected and not fortran and not dtype.hasobject, 'V numeric header shape/type changed: ' + key)
            if key == 'observed':
                require(dtype == np.dtype(bool), 'Observation mask must be boolean')
            elif key == 'origins':
                require(dtype.kind in 'iu', 'Origins must be integer')
            elif key == 'channel_ids':
                require(dtype.kind in 'SU', 'Channel names must be non-object strings')
            else:
                require(dtype.kind == 'f', 'V numeric arrays must be floating point')
    with np.load(path, allow_pickle=False) as archive:
        origins = archive['origins'].copy()
        expected = np.arange(V_STARTS[dataset], runtime.T_STARTS[dataset]-48+1, 48, dtype=np.int64)
        require(np.array_equal(origins, expected), 'V origin grid changed or overlaps protected T')
        require(int(origins.max()) + 48 <= runtime.T_STARTS[dataset], 'V target overlaps protected T')
        arrays = {key: archive[key].copy() for key in keys if key != 'origins'}
    arrays['origins'], arrays['_dataset'] = origins, dataset
    require(np.isfinite(arrays['mean']).all() and np.isfinite(arrays['std']).all()
            and (arrays['std'] > 0).all(), 'Normalization arrays must be finite and positive')
    if cohort is not None:
        require(cohort.get('_dataset') == dataset, 'Cross-dataset cohort comparison is forbidden')
        for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
            same = np.array_equal(arrays[key], cohort[key], equal_nan=arrays[key].dtype.kind in 'fc')
            require(same, 'Reference cohort/scaler differs: ' + key)
    pred, target, mask = arrays['pred_normalized'], arrays['target_normalized'], arrays['observed']
    require(np.isfinite(pred).all() and np.isfinite(target[mask]).all(), 'V predictions/observed targets are nonfinite')
    difference = pred.astype(np.float64) - target.astype(np.float64)
    mse_channels, mae_channels = [], []
    for channel in range(channels):
        error = difference[..., channel][mask[..., channel]]
        require(error.size > 0, 'A scored channel has no observed target')
        mse_channels.append(math.fsum(float(x)*float(x) for x in error)/error.size)
        mae_channels.append(math.fsum(abs(float(x)) for x in error)/error.size)
    mse, mae = math.fsum(mse_channels)/channels, math.fsum(mae_channels)/channels
    require(max(abs(mse-float(row['MSE'])), abs(mae-float(row['MAE']))) <= 1e-9,
            'Independent saved-array FP64 metric mismatch')
    if mask.all():
        require(abs(mse-float(np.mean(difference**2))) < 1e-12, 'Fully observed macro/global aggregation differs')
    return arrays, dict(job_id=row['job_id'], dataset=dataset, sha256=row['prediction_sha256'],
        shape=list(shape), MSE=mse, MAE=mae, normalization_arrays_present=True,
        origin_header_checked_before_numeric_materialization=True, T_numeric_values_parsed=0, status='PASS')


def public_hashes(paths):
    return {path.resolve().relative_to(runtime.REPO.resolve()).as_posix(): runtime.sha(path) for path in paths}


def verify_job_evidence(job, registration, completion):
    job_id = job['job_id']
    root = runtime.REPORT / 'jobs' / job_id
    result_path = root / 'RESULT.json'
    statuses = completion['run_statuses']
    result = runtime.read(result_path)
    spec = run_one.build_spec(job)
    if (statuses.get(job_id) != 'COMPLETE' or result.get('status') != 'COMPLETE'
            or result.get('job') != job or result.get('registration') != registration
            or result.get('spec') != spec or result.get('selection_sealed_before_V') is not True):
        raise ValueError('Completed result differs from sealed job/controller: ' + job_id)
    attempt = result['attempt']
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Result has an unregistered attempt: ' + job_id)
    public = root / f'attempt_{attempt:03d}'
    required = [result_path, root / f'WORKER_RECEIPT_attempt{attempt}.json']
    required += [public / name for name in ('START.json', 'SOURCE_CONTRACT.json', 'SEARCH_RESULT.json',
                                           'SELECTION.json', 'REFIT_SUMMARY.json', 'RECEIPT.json',
                                           'OPENED_FILES.json')]
    if any(not path.is_file() for path in required):
        raise ValueError('Completed job is missing public evidence: ' + job_id)
    start = runtime.read(public / 'START.json')
    receipt = runtime.read(public / 'RECEIPT.json')
    worker = runtime.read(required[1])
    require(isinstance(worker.get('started_unix'), (int, float)) and math.isfinite(worker['started_unix'])
            and worker['started_unix'] > 0, 'Worker start time is missing or invalid: ' + job_id)
    require(datetime.fromisoformat(registration['registered_at_utc']).timestamp() <= worker['started_unix'],
            'Worker started before registration: ' + job_id)
    authorization = run_one.validate_attempt(job, attempt, started_unix=worker['started_unix'])
    require(result.get('attempt_authorization') == authorization, 'Result recovery authorization differs: ' + job_id)
    if authorization['kind'] == 'TECHNICAL_RECOVERY':
        authorization_path = Path(authorization['authorization_path'])
        require(runtime.sha(authorization_path) == authorization['authorization_sha256'], 'Recovery authorization changed')
        required.append(authorization_path)
    if (start.get('job') != job or start.get('attempt') != attempt
            or start.get('registration') != registration or start.get('spec') != spec):
        raise ValueError('Start identity differs from completed job: ' + job_id)
    if (receipt.get('status') != 'COMPLETE' or receipt.get('job') != job
            or receipt.get('attempt') != attempt or receipt.get('registration') != registration
            or receipt.get('training_attempted') is not True or receipt.get('training_executed') is not True
            or receipt.get('T_numeric_values_parsed') != 0
            or any(receipt.get(key, {}).get('changed') != [] for key in ('preserved_before', 'preserved_after'))):
        raise ValueError('Successful attempt receipt is invalid: ' + job_id)
    if (worker.get('job_id') != job_id or worker.get('attempt') != attempt
            or worker.get('status') != 'COMPLETE' or worker.get('exit_code') != 0):
        raise ValueError('Successful worker receipt is invalid: ' + job_id)
    recorded_attempts = [r for r in completion.get('attempts', {}).get(job_id, [])
                         if r.get('attempt') == attempt and 'exit_code' in r]
    if any(r['exit_code'] != 0 or r.get('job_id') != job_id for r in recorded_attempts):
        raise ValueError('Controller attempt differs from successful worker: ' + job_id)
    if runtime.read(public / 'OPENED_FILES.json').get('T_numeric_values_parsed') != 0:
        raise ValueError('Opened-file audit does not confirm protected T exclusion: ' + job_id)
    runtime.assert_protected_T(result)
    source = runtime.read(public / 'SOURCE_CONTRACT.json')
    contract = result['contract']
    if (source.get('spec') != spec or not source.get('sources')
            or any(source.get(key) != contract.get(key) for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha'))):
        raise ValueError('Source contract differs from completed job: ' + job_id)
    selection_path = public / 'SELECTION.json'
    if Path(result['selection_path']).resolve() != selection_path.resolve():
        raise ValueError('Result points outside its successful selection attempt: ' + job_id)
    run_one.check_selection(selection_path, result['selection_sha256'], job)
    selection = runtime.read(selection_path)
    search = runtime.read(public / 'SEARCH_RESULT.json')
    selected, selected_mse = run_one.choose_epoch(search, spec['epochs'])
    if (selection.get('spec') != spec or selection.get('contract') != contract
            or selection.get('selection_split') != 'A_stop' or selection.get('curves') != search['curves']
            or selection.get('selected_epoch') != selected or selection.get('selected_astop_mse') != selected_mse
            or result.get('search_summary') != search['summary']):
        raise ValueError('Selection evidence differs from registered A_stop search: ' + job_id)
    refit = runtime.read(public / 'REFIT_SUMMARY.json')
    if refit.get('status') != 'SEGMENT_COMPLETE' or result.get('refit_summary') != refit:
        raise ValueError('Fresh refit evidence is incomplete: ' + job_id)
    row = result['row']
    if (any(row.get(key) != value for key, value in job.items()) or row.get('reused') is not False
            or row.get('selected_epoch') != selected or row.get('training_cap') != spec['epochs'][-1]
            or row.get('cap_selected') != (selected == spec['epochs'][-1])):
        raise ValueError('Result row differs from job or sealed selection: ' + job_id)
    if runtime.sha(result['bundle_path']) != result['bundle_sha256']:
        raise ValueError('Selected adapter bundle changed: ' + job_id)
    if runtime.sha(row['prediction_path']) != row['prediction_sha256']:
        raise ValueError('Completed prediction archive changed: ' + job_id)
    return result, public_hashes(required)


def validate_plan(plan):
    require(not plan.get('extension_jobs') and not plan.get('optional_jobs'), 'No extension or optional jobs are registered')
    expected = [dict(job_id=f'PBA_{method}_{dataset}_K{runtime.KS[dataset]}_s{seed}', method_id=method,
                     dataset=dataset, K=runtime.KS[dataset], seed=seed, extended=False)
                for seed in SEEDS for dataset in DATASETS for method in METHODS]
    require(plan.get('jobs') == expected, 'RUN_PLAN must contain exactly sixty jobs in the registered order')
    for job in expected:
        run_one.validate_job(job)
    return expected


def verify_rejected_result(job, run_status):
    root = runtime.REPORT/'jobs'/job['job_id']
    receipt_path = root/'RESULT_REJECTION.json'
    receipt = runtime.read(receipt_path)
    require(run_status != 'COMPLETE' and receipt.get('status') == 'REJECTED_SCIENTIFIC_EVIDENCE'
            and receipt.get('job') == job and receipt.get('run_status') == run_status,
            'Rejected result does not match the terminal disposition')
    require(receipt.get('trigger') == 'RESULT_WRITTEN_THEN_TERMINAL_VALIDATION_REJECTED'
            and bool(str(receipt.get('reason', '')).strip()), 'Rejected result needs an explicit terminal validation reason')
    attempt = receipt.get('attempt')
    digest = receipt.get('rejected_result_sha256', '')
    require(type(attempt) is int and attempt >= 0 and len(digest) == 64,
            'Rejected result attempt or digest is invalid')
    original = root/'RESULT.json'
    require(receipt.get('original_result_path') == original.relative_to(runtime.REPO).as_posix()
            and not original.exists(), 'A rejected result must preserve its original path and leave no normal RESULT')
    destination = root/'rejected'/f'RESULT_attempt{attempt:03d}_{digest[:12]}.json'
    require(receipt.get('rejected_result_path') == destination.relative_to(runtime.REPO).as_posix()
            and destination.is_file() and runtime.sha(destination) == digest,
            'Rejected result bytes/path changed')
    try:
        created = datetime.fromisoformat(receipt['created_utc']).timestamp()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Rejected result has no valid timestamp') from exc
    require(math.isfinite(created) and created <= datetime.now(timezone.utc).timestamp(),
            'Rejected result timestamp is invalid or in the future')
    public = [receipt_path, destination]
    worker_path = root/f'WORKER_RECEIPT_attempt{attempt}.json'
    if worker_path.is_file():
        require(receipt.get('worker_receipt_path') == worker_path.relative_to(runtime.REPO).as_posix()
                and receipt.get('worker_receipt_sha256') == runtime.sha(worker_path),
                'Rejected result is not bound to its existing worker receipt')
        public.append(worker_path)
    else:
        require(receipt.get('worker_receipt_missing') is True and 'worker_receipt_path' not in receipt,
                'Missing final worker receipt must be explicit')
    record = dict(job_id=job['job_id'], attempt=attempt, run_status=run_status, reason=receipt['reason'],
                  rejected_result_path=receipt['rejected_result_path'], rejected_result_sha256=digest,
                  rejection_record_path=receipt_path.relative_to(runtime.REPO).as_posix(),
                  excluded_from_numeric_analysis=True)
    return record, public_hashes(public)


def verify_completion_evidence(plan, completion):
    require(completion.get('status') in TERMINAL and completion.get('active') is None,
            'Analysis requires a terminal execution with no active worker')
    jobs = validate_plan(plan)
    statuses = completion['run_statuses']
    require(set(statuses) == {job['job_id'] for job in jobs}, 'Terminal job roster missing or unexpected')
    registration = run_one.validate_registration(jobs[0])
    results, verified = {}, {}
    for job in jobs:
        root = runtime.REPORT / 'jobs' / job['job_id']
        path = root / 'RESULT.json'
        if statuses[job['job_id']] == 'COMPLETE':
            require(not (root/'RESULT_REJECTION.json').exists(), 'A completed job also has rejected-result evidence')
            require(path.is_file(), 'Controller completion has no result: ' + job['job_id'])
        elif (root/'RESULT_REJECTION.json').exists():
            _, hashes = verify_rejected_result(job, statuses[job['job_id']])
            verified.update(hashes)
            continue
        elif not path.exists():
            require(not list((root/'rejected').glob('*.json')), 'Orphan rejected-result bytes have no hash-bound receipt')
            continue
        else:
            raise ValueError('A result exists but the terminal controller does not mark it COMPLETE: ' + job['job_id'])
        result, hashes = verify_job_evidence(job, registration, completion)
        require(result['contract']['model_sha'] == runtime.F_SHA, 'Result foundation identity differs')
        require(Path(result['bundle_path']).resolve().is_relative_to(runtime.CACHE.resolve()), 'New bundle is outside the new campaign cache')
        require(Path(result['row']['prediction_path']).resolve().is_relative_to(runtime.CACHE.resolve()), 'New prediction is outside the new campaign cache')
        normalized_row(result['row'])
        results[job['job_id']] = result
        verified.update(hashes)
    count = len(results)
    require((completion['status'] == 'TRAINING_COMPLETE') == (count == 60), 'Top-level completion status disagrees with actual results')
    if 'completed_count' in completion:
        require(completion['completed_count'] == count, 'Terminal completed_count differs')
    return results, count, verified


def validate_reference_rows(rows):
    normalized = [normalized_row(dict(row, reused=True)) for row in rows]
    require(len(normalized) == 90, 'Exactly ninety reference rows are required')
    for dataset in DATASETS:
        for method in ('OURS', 'A1', 'A2'):
            group = [r for r in normalized if r['dataset'] == dataset and r['method_id'] == method]
            require(sorted(r['seed'] for r in group) == SEEDS, 'Reference dataset/method seed coverage changed')
    return normalized


def analyze_rows(reference_rows, new_rows):
    rows = validate_reference_rows(reference_rows) + [normalized_row(row) for row in new_rows]
    seen = set()
    for row in rows:
        key = row['dataset'], row['method_id'], row['seed']
        require(key not in seen, 'Duplicate dataset/method/seed result')
        seen.add(key)
    summaries = []
    for dataset in DATASETS:
        for method in ORDER:
            group = sorted((r for r in rows if (r['dataset'], r['method_id']) == (dataset, method)), key=lambda r:r['seed'])
            n = len(group)
            require(n <= 10, 'Too many seeds in a result family')
            primary = [r['seed'] for r in group] == SEEDS
            cap_count = sum(r['selected_epoch'] == 3. for r in group)
            provisional = primary and cap_count >= 5
            for row in group:
                row.update(primary=primary, provisional=provisional)
            mse = np.array([r['MSE'] for r in group], dtype=np.float64)
            q1, median, q3 = np.quantile(mse, [.25, .5, .75]) if n else (None, None, None)
            parameters = group[0]['trainable_parameters'] if n else runtime.EXPECTED_PARAMETERS[dataset].get(method)
            summaries.append(dict(dataset=dataset, method_id=method, display_name=LABELS[method], n=n,
                primary=primary, status='COMPLETE' if primary else 'INCOMPLETE',
                MSE_median=None if not n else float(median), MSE_Q1=None if not n else float(q1),
                MSE_Q3=None if not n else float(q3), MSE_IQR=None if not n else float(q3-q1),
                MAE_median=float(np.median([r['MAE'] for r in group])) if n else None,
                parameters=parameters, cap=3., extended=False, cap_selected_count=cap_count,
                extension_status='NOT_RUN_SCOPE' if provisional else 'NOT_REQUIRED' if primary else 'BASE_INCOMPLETE',
                provisional=provisional, selected_epochs=json.dumps([r['selected_epoch'] for r in group])))
    lookup = {(r['dataset'],r['method_id']):r for r in summaries}
    contrasts, paired = [], []
    for dataset in DATASETS:
        ours = {r['seed']:r for r in rows if r['dataset'] == dataset and r['method_id'] == 'OURS'}
        for method in (*METHODS,'A1','A2'):
            s, ref = lookup[dataset,method], lookup[dataset,'OURS']
            complete = s['primary'] and ref['primary']
            differences = []
            for row in sorted((r for r in rows if r['dataset'] == dataset and r['method_id'] == method),key=lambda r:r['seed']):
                delta = row['MSE'] - ours[row['seed']]['MSE']
                differences.append(delta)
                paired.append(dict(dataset=dataset, method_id=method, seed=row['seed'], variant_MSE=row['MSE'],
                    full_P_MSE=ours[row['seed']]['MSE'], delta_MSE=delta, primary=complete))
            gap = s['MSE_median']-ref['MSE_median'] if complete else None
            spread = max(s['MSE_IQR'],ref['MSE_IQR']) if complete else None
            decision = ('FULL_P_LOWER' if gap > spread else 'VARIANT_LOWER' if gap < -spread else 'UNCLEAR') if complete else 'INCOMPLETE'
            provisional = s['provisional'] or ref['provisional']
            if complete and provisional:
                decision += '_PROVISIONAL'
            contrasts.append(dict(dataset=dataset, method_id=method, contrast=f'{LABELS[method]} minus Full P',
                primary=complete, delta_median_MSE=gap,
                effect_pct=100*gap/ref['MSE_median'] if complete else None, spread=spread,
                descriptive_decision=decision, paired_delta_median=float(np.median(differences)) if complete else None,
                seeds_full_P_lower=sum(d > 0 for d in differences), n=s['n'], full_P_n=ref['n'],
                cap_selected_count=s['cap_selected_count'], full_P_cap_selected_count=ref['cap_selected_count'],
                provisional=provisional, inference='Exploratory development-V description; no significance/equivalence claim.'))
    return rows, summaries, contrasts, paired


def check_reference_effects(contrasts):
    for dataset in DATASETS:
        for method in ('A1','A2'):
            group = [r for r in contrasts if (r['dataset'],r['method_id']) == (dataset,method)]
            require(len(group) == 1 and group[0]['primary'], 'Reference contrast is incomplete')
            require(round(group[0]['effect_pct'],1) == REFERENCE_EFFECTS[dataset][method], 'Reference slide effect changed')


def slide_numbers(summaries, contrasts):
    lookup = {(r['dataset'],r['method_id']):r for r in summaries}
    lines = ['# P branch numbers for slide16', '', 'Development-V exploratory results; signed effect vs Full P.', '',
             'Jena unchanged: Time branch only +0.2% (UNCLEAR); Channel branch only +8.9% (FULL_P_LOWER).', '',
             '```text','Method               Dataset             Effect   Decision                    cap3_selected n']
    for method in METHODS:
        for dataset in DATASETS:
            row = next(r for r in contrasts if (r['dataset'],r['method_id']) == (dataset,method))
            effect = f"{row['effect_pct']:+.1f}%" if row['primary'] else 'INCOMPLETE'
            lines.append(f"{method:<20} {dataset:<20} {effect:>10} {row['descriptive_decision']:<27} {row['cap_selected_count']:>2}/10 {row['n']:>2}")
    lines += ['```','','```text','Dataset             Time_only_params Channel_only_params']
    for dataset in DATASETS:
        lines.append(f"{dataset:<20} {lookup[dataset,'P_TIME_ONLY']['parameters']:>16} {lookup[dataset,'P_CHANNEL_ONLY']['parameters']:>19}")
    lines += ['```','','Positive effect means the retained-branch variant has higher MSE. Incomplete families are not ten-seed primary results.',
              'PROVISIONAL propagates from either comparison arm; no cap10 extension was run. Unequal capacity and joint LayerNorm/E/D coupling remain.','']
    return '\n'.join(lines)


def prediction_check(contrasts):
    lines = ['# 사전 예측 확인','','아래 예측은 실행 범위와 판정 기준에 사용하지 않았다.']
    for row in contrasts:
        if row['method_id'] not in METHODS:
            continue
        if not row['primary']:
            outcome = 'INCOMPLETE: 평가 불가'
        elif row['method_id'] == 'P_CHANNEL_ONLY' and row['dataset'] == 'ETTh2_safe17400':
            outcome = '일치' if row['effect_pct'] < 0 and row['provisional'] else '불일치'
        else:
            outcome = '일치' if row['descriptive_decision'].startswith('UNCLEAR') else '불일치'
        lines.append(f"- {row['dataset']} / {row['method_id']}: {outcome}; 실제 {row['descriptive_decision']}.")
    return '\n'.join(lines)+'\n'


def draw_figure(rows, summaries, raw_rows, directory):
    require({r['dataset'] for r in raw_rows} == set(DATASETS) and len(raw_rows) == 3, 'Three bound deterministic Raw rows are required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3,figsize=(15,5.8))
    methods = ('OURS',*METHODS,'A1')
    labels = ('Full P','Time only','Channel only','No P')
    colors = ('#9b6caa','#63a6c8','#94c5b1','#b7bdc5')
    lookup = {(r['dataset'],r['method_id']):r for r in summaries}
    for axis,dataset in zip(axes,DATASETS):
        ticks = []
        for pos,(method,label,color) in enumerate(zip(methods,labels,colors)):
            values = [r['MSE'] for r in sorted(rows,key=lambda x:x['seed']) if (r['dataset'],r['method_id']) == (dataset,method)]
            summary = lookup[dataset,method]
            if values:
                box=axis.boxplot([values],positions=[pos],widths=.48,patch_artist=True,showfliers=False)
                box['boxes'][0].set_facecolor(color)
                axis.scatter(pos+np.linspace(-.13,.13,len(values)),values,s=22,color='#333333',alpha=.75,zorder=3)
            else:
                axis.text(pos,.03,'not completed',rotation=90,ha='center',va='bottom',transform=axis.get_xaxis_transform(),fontsize=8)
            suffix = '\nPROVISIONAL' if summary['provisional'] else ''
            if not summary['primary']:
                suffix += f"\nINCOMPLETE n={summary['n']}"
            ticks.append(label+suffix)
        raw = next(r for r in raw_rows if r['dataset']==dataset)
        axis.axhline(float(raw['MSE']),color='#ad4f43',linestyle='--',linewidth=1,label='Raw (existing)')
        axis.set_xticks(range(4),ticks,fontsize=8)
        axis.set_title(dataset)
        axis.set_ylabel('Normalized V MSE (lower is better)')
        axis.grid(axis='y',alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('P branch ablations: exploratory development V, cap3, no extensions',fontsize=12)
    fig.text(.5,.012,'Every completed seed is shown; n<10 is excluded from primary contrasts. Unequal branch capacities and joint LayerNorm/E/D mixing remain.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for suffix in ('png','svg'):
        fig.savefig(directory/f'{FIGURE}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    completion = runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
    preservation = runtime.guard_preserved(full=True)
    plan = runtime.read(runtime.REPORT/'RUN_PLAN.json')
    results, count, verified = verify_completion_evidence(plan,completion)
    rejections = [verify_rejected_result(job, completion['run_statuses'][job['job_id']])[0]
                  for job in plan['jobs'] if (runtime.REPORT/'jobs'/job['job_id']/'RESULT_REJECTION.json').exists()]
    rejection_by_id = {record['job_id']:record for record in rejections}
    reference = runtime.read(runtime.REPORT/'REFERENCE_RUNS.json')
    references = validate_reference_rows(reference['rows'])
    new_rows = [result['row'] for result in results.values()]
    rows,summaries,contrasts,paired = analyze_rows(references,new_rows)
    check_reference_effects(contrasts)
    cohorts, checks = {}, []
    for dataset in DATASETS:
        canonical = next(r for r in references if r['dataset']==dataset and r['method_id']=='OURS' and r['seed']==92603)
        cohorts[dataset], _ = verify_arrays(canonical)
    for row in rows:
        _,check=verify_arrays(row,cohorts[row['dataset']]);checks.append(check)
    primary_new = [r for r in rows if not r['reused'] and r['primary']]
    all_new = [r for r in rows if not r['reused']]
    primary_rows = [r for r in rows if r['primary']]
    status_rows = [dict(job, status=completion['run_statuses'][job['job_id']], completed=job['job_id'] in results,
                        rejected_result=job['job_id'] in rejection_by_id,
                        rejection_reason=rejection_by_id.get(job['job_id'],{}).get('reason',''),
                        rejected_result_sha256=rejection_by_id.get(job['job_id'],{}).get('rejected_result_sha256',''))
                   for job in plan['jobs']]
    output = runtime.REPORT
    write_csv(output/'RUNS.csv',primary_rows)
    write_csv(output/'ALL_NEW_RUNS.csv',all_new,fieldnames=list(all_new[0]) if all_new else
              ['dataset','method_id','seed','job_id','MSE','MAE','primary','provisional'])
    write_csv(output/'SUMMARY.csv',summaries)
    write_csv(output/'CONTRASTS.csv',contrasts)
    write_csv(output/'PAIRED_DIFFERENCES.csv',paired)
    write_csv(output/'JOB_STATUS.csv',status_rows)
    (output/'SLIDE_NUMBERS.md').write_text(slide_numbers(summaries,contrasts),encoding='utf8')
    (output/'PREDICTION_CHECK.md').write_text(prediction_check(contrasts),encoding='utf8')
    draw_figure(rows,summaries,reference.get('raw_rows',[]),output)
    lines=['# Appliances·ETT P 분기 결과','','개발 V의 탐색 결과이며 실행 범위는 과거 결과를 본 뒤 정했다. 보호 T는 평가하지 않았다.',
           f"기본 학습 {count}/60 완료. 상태 {completion['status']}. 연장은 하지 않았다.",
           f'최종 검증에서 거절된 RESULT {len(rejections)}개는 원본 bytes와 해시·사유를 별도 보존하고 수치 분석에서 제외했다.',
           '정확히10seed가 완성된 dataset×method만 주 비교로 썼다. 불완전 실행과 불리하거나 불명확한 결과도 보존했다.','',slide_numbers(summaries,contrasts),
           '시간/채널 분기의 용량은 다르며 공동 LayerNorm·E/D 결합이 남는다. 분기 제거의 조건부 효과이며 축 자체의 보편적 우월성·정보 보존 메커니즘을 입증하지 않는다.',
           'IQR 기준은 기술적 판정이며 유의성·동등성 검정이 아니다. 기존 비교 방법의 최적화 공백이나 독립 테스트 일반화를 해결한 결과가 아니다.',
           f"실행 시작 {completion.get('started_utc','not recorded')}, 종료 {completion.get('completed_utc','not recorded')}. 중단 사유: {completion.get('stop_reason','없음')}.",
           '새 학습 외 모델 선택·추가seed·cap10연장은 수행하지 않았다. 발표파일·대본은 생성하거나 수정하지 않았다.','']
    (output/'RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf8')
    outputs=['RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv','JOB_STATUS.csv',
             'SLIDE_NUMBERS.md','PREDICTION_CHECK.md','RESULTS_KO.md',f'{FIGURE}.png',f'{FIGURE}.svg']
    inputs=['REGISTRATION_SEAL.json','PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json',
            'REFERENCE_RUNS.json','PRESERVATION_BASELINE.json','EXECUTION_COMPLETION.json']
    verified.update(public_hashes(output/name for name in inputs+outputs))
    value=dict(status='PASS',integrity_status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        completed_count=count,base_complete=count,new_result_count=len(all_new),primary_count=len(primary_new),
        rejected_result_count=len(rejections),rejected_results=rejections,
        primary_rows=len(primary_rows),completed_primary_families=sum(s['primary'] for s in summaries if s['method_id'] in METHODS),
        checked_prediction_archives=len(checks),independent_FP64_checks=checks,preservation=preservation,
        T_numeric_values_parsed=0,summary=summaries,contrasts=contrasts,verified_files=verified,
        inputs=dict(registration_seal_sha256=runtime.sha(output/'REGISTRATION_SEAL.json'),
                    completion_sha256=runtime.sha(output/'EXECUTION_COMPLETION.json')))
    runtime.atomic(output/'ANALYSIS_VALIDATION.json',value)
    print(json.dumps(dict(status='PASS',base_complete=count,primary_new_rows=len(primary_new),primary_families=value['completed_primary_families'])))


if __name__ == '__main__':
    main()
