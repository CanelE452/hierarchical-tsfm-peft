"""Audit registered V-only artifacts and describe complete matched screen cohorts."""
from datetime import datetime, timezone
import csv
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from . import runtime, run_one


METHODS = ('OURS', 'TIME_ONLY', 'A5_LAG0', 'E_FIXED')
DATASETS = ('pems32', 'beijing')
COMPONENTS = (('TIME_ONLY', 'FIG_TIME_ONLY', 'Remove channel MLP; retain joint LayerNorm'),
              ('A5_LAG0', 'FIG_LAG0', 'Keep anchor; use lag0 decoder only'),
              ('E_FIXED', 'FIG_E_FIXED', 'Freeze stage-local PCA E'))
COHORT_KEYS = ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def scaler_fingerprint(mean, std):
    value = dict(mean=fingerprint(np.asarray(mean, np.float32)), std=fingerprint(np.asarray(std, np.float32)))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_csv(path, rows, empty_fields):
    stream = io.StringIO(newline='')
    keys = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(empty_fields)
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader(); writer.writerows(rows)
    temporary = path.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8-sig')
    temporary.replace(path)


def verify_plan(plan):
    jobs = plan['jobs'] + plan['optional_jobs']
    require(len(plan['jobs']) == 16 and len(plan['optional_jobs']) == 8 and len({j['job_id'] for j in jobs}) == 24,
            'Expected 16 base and 8 distinct time-only optional jobs')
    require(plan.get('extension_jobs', []) == [] and plan['group_order'] ==
            ['pems32_base', 'beijing_base', 'beijing_seed3', 'pems32_seed3'], 'Group order/no-extension policy changed')
    for rows, seeds in ((plan['jobs'], [92601, 92602]), (plan['optional_jobs'], [92603])):
        for job in rows:
            run_one.validate_job(job)
        for dataset in DATASETS:
            for method in METHODS:
                require(sorted(j['seed'] for j in rows if (j['dataset'], j['method_id']) == (dataset, method)) == seeds,
                        'Registered method/seed group is incomplete')
    return jobs


def verify_completion(jobs, completion):
    terminal = {'COMPLETE', 'FAILED', 'STOPPED_DEADLINE', 'NOT_RUN_BUDGET', 'BASE_INCOMPLETE'}
    state = completion['status']; statuses = completion['run_statuses']
    ids = {j['job_id'] for j in jobs}
    require(state in ('TRAINING_COMPLETE', 'TRAINING_INCOMPLETE', 'NOT_RUN_BUDGET', 'STOPPED')
            and completion.get('active') is None, 'Postprocess requires terminal execution without an active worker')
    require(not(set(statuses)-ids) and not(set(completion.get('attempts', {}))-ids), 'Unregistered execution status/attempt')
    if state == 'STOPPED':
        require(bool(completion.get('stop_reason')) and set(statuses.values()) <= terminal | {'RUNNING', 'FAILED_ONCE'},
                'Stopped execution has unknown status or no reason')
    else:
        require(set(statuses) == ids and set(statuses.values()) <= terminal, 'Normal completion contains missing/nonterminal jobs')
    base = sum(statuses.get(j['job_id']) == 'COMPLETE' for j in jobs if j['seed'] != 92603)
    optional = sum(statuses.get(j['job_id']) == 'COMPLETE' for j in jobs if j['seed'] == 92603)
    counts = dict(base_complete_count=base, optional_complete_count=optional, completed_count=base+optional, extension_jobs=0)
    for key, value in counts.items():
        if state != 'STOPPED' or key in completion:
            require(type(completion.get(key)) is int and completion[key] == value, 'Completion count differs: '+key)
    if state != 'STOPPED':
        expected = ('TRAINING_COMPLETE' if base+optional == len(jobs) else 'NOT_RUN_BUDGET'
                    if base+optional == 0 and set(statuses.values()) <= {'NOT_RUN_BUDGET', 'BASE_INCOMPLETE'}
                    else 'TRAINING_INCOMPLETE')
        require(state == expected, 'Top completion status differs from actual terminal jobs')
    return counts


def select_cohorts(jobs, completion):
    statuses = completion['run_statuses']
    require(not(set(statuses) - {j['job_id'] for j in jobs}), 'Unregistered execution status')
    selected, coverage = [], []
    for dataset in DATASETS:
        base = [j for j in jobs if j['dataset'] == dataset and j['seed'] in (92601, 92602)]
        extra = [j for j in jobs if j['dataset'] == dataset and j['seed'] == 92603]
        base_done = all(statuses.get(j['job_id']) == 'COMPLETE' for j in base)
        extra_count = sum(statuses.get(j['job_id']) == 'COMPLETE' for j in extra)
        n = 3 if base_done and extra_count == 4 else 2 if base_done else 0
        primary = base + extra if n == 3 else base if n == 2 else []
        selected.extend(j['job_id'] for j in primary)
        coverage.append(dict(dataset=dataset, primary_n=n, primary_seeds=json.dumps([92601, 92602, 92603][:n]) if n else '[]',
            base_completed=sum(statuses.get(j['job_id']) == 'COMPLETE' for j in base), base_planned=8,
            optional_completed=extra_count, optional_planned=4, optional_partial=0 < extra_count < 4,
            primary_jobs=len(primary), status='EXPLORATORY' if n == 3 else 'PILOT' if n == 2 else 'INCOMPLETE',
            historically_selected_development_V=True, held_out_test=False))
        saved = completion.get('dataset_status', {}).get(dataset)
        if saved is not None:
            require(saved['primary_n'] == n and set(saved['primary_job_ids']) == {j['job_id'] for j in primary},
                    'Controller matched-cohort decision differs')
    return set(selected), coverage


def verify_technical_import_failure(job, attempt, registration):
    amendment_path = runtime.REPORT/'TECHNICAL_AMENDMENT_001.json'
    require(amendment_path.is_file() and attempt == 1, 'Scientific failure lacks an explicit technical amendment')
    amendment = runtime.read(amendment_path)
    require(amendment.get('failed_job') == job['job_id'] and amendment.get('training_attempted') is False
            and amendment.get('training_executed') is False and amendment.get('T_numeric_values_parsed') == 0,
            'Amendment does not establish a pretraining failure')
    archived = {}; public_files = [amendment_path]
    archive_root = (runtime.REPORT/'revisions'/'pretraining_v1').resolve()
    for row in amendment['files']:
        source = (runtime.REPO/row['original']).resolve(); copy = (runtime.REPO/row['archive']).resolve()
        require(source.is_relative_to(runtime.REPO.resolve()) and copy.is_relative_to(archive_root)
                and row['original'] not in archived and runtime.sha(copy) == row['sha256'], 'Technical archive path/hash differs')
        archived[row['original']] = row
        if '.cache' not in Path(row['original']).parts and copy.suffix in ('.json', '.md', '.py'):
            public_files.append(copy)
    root = runtime.REPORT/'jobs'/job['job_id']; failed = root/'attempt_001'
    evidence = [root/'WORKER_RECEIPT_attempt1.json', failed/'RECEIPT.json', failed/'OPENED_FILES.json', failed/'HOST_GPU_GATE.json']
    for path in evidence:
        entry = archived.get(path.relative_to(runtime.REPO).as_posix())
        require(entry is not None and runtime.sha(path) == entry['sha256'], 'Original technical failure evidence changed')
    worker, receipt, opened = (runtime.read(p) for p in evidence[:3])
    require(receipt['job'] == job and receipt['attempt'] == 1 and receipt['status'] == 'FAILED'
            and receipt['training_attempted'] is False and receipt['training_executed'] is False
            and receipt['T_numeric_values_parsed'] == 0 and opened['T_numeric_values_parsed'] == 0
            and receipt['preserved_before']['changed'] == [] and not (failed/'START.json').exists(),
            'Archived failure occurred after model/data/training admission')
    require(worker['job_id'] == job['job_id'] and worker['attempt'] == 1 and worker['status'] == 'STOPPED'
            and worker['exit_code'] == 20 and worker['error'] == receipt['error']
            and receipt['error'].startswith('PermissionError: Write outside this campaign')
            and receipt['error'].lower().endswith('nul'), 'Failure is not the archived null-device import error')
    cache = runtime.CACHE/'jobs'/job['job_id']/'attempt_001'
    require(not cache.exists() or not any(p.is_file() for p in cache.rglob('*')), 'Pretraining failure contains model artifacts')
    seal_rel = (runtime.REPORT/'REGISTRATION_SEAL.json').relative_to(runtime.REPO).as_posix()
    seal_entry = archived.get(seal_rel)
    require(seal_entry is not None and seal_entry['sha256'] == amendment['original_seal_sha256']
            == receipt['registration']['seal_sha256'] != registration['seal_sha256'], 'Old failure seal binding differs')
    old_seal = runtime.read(runtime.REPO/seal_entry['archive'])
    require(receipt['registration']['files'] == old_seal['files']
            and receipt['registration']['registered_at_utc'] == old_seal['created_utc'], 'Archived registration identity differs')
    for name, digest in old_seal['files'].items():
        entry = archived.get((runtime.REPORT/name).relative_to(runtime.REPO).as_posix())
        require(entry is not None and entry['sha256'] == digest, 'Archived seal document is missing or inconsistent')
    require(datetime.fromisoformat(receipt['started_utc']) < datetime.fromisoformat(amendment['created_utc'])
            <= datetime.fromisoformat(registration['registered_at_utc']), 'Technical amendment chronology differs')
    return public_files+evidence


def verify_attempts(job, registration, completion, successful_attempt=None):
    root = runtime.REPORT/'jobs'/job['job_id']
    records = completion.get('attempts', {}).get(job['job_id'], [])
    numbers = [row.get('attempt') for row in records]
    require(all(type(n) is int and n in (1, 2) for n in numbers) and len(set(numbers)) == len(numbers),
            'Duplicate/unregistered attempt in execution completion')
    saved = {row['attempt']:row for row in records}; workers = {}
    files, outcomes = [], []
    for path in sorted(root.glob('WORKER_RECEIPT_attempt*.json')):
        worker = runtime.read(path); number = worker['attempt']
        require(type(number) is int and number in (1, 2) and number not in workers
                and path.name == f'WORKER_RECEIPT_attempt{number}.json' and worker['job_id'] == job['job_id'],
                'Mismatched/unregistered worker attempt')
        workers[number] = worker; files.append(path)
        require(number in saved and saved[number].get('exit_code') == worker['exit_code'], 'Controller/worker attempt exit differs')
    used = sorted(set(saved) | set(workers))
    require(used == list(range(1, len(used)+1)), 'Attempt sequence is incomplete')
    if successful_attempt is not None:
        require(used and max(used) == successful_attempt, 'Successful result does not match final recorded attempt')
    for number in used:
        worker = workers.get(number); record = saved[number]
        outcome = dict(attempt=number, status=worker['status'] if worker else record.get('status'), exit_code=record.get('exit_code'))
        if number == successful_attempt:
            require(worker is not None and worker['status'] == 'COMPLETE' and worker['exit_code'] == 0,
                    'Successful attempt lacks complete worker evidence')
        elif worker is not None:
            require(worker['status'] != 'COMPLETE' and worker['exit_code'] != 0, 'Successful attempt repeated or omitted')
            if worker['exit_code'] == 20 and successful_attempt is not None:
                files.extend(verify_technical_import_failure(job, number, registration))
                outcome['failure_class'] = 'ARCHIVED_PRETRAINING_IMPORT_FAILURE'
                outcome['model_performance_result'] = False
            else:
                outcome['failure_class'] = 'SCIENTIFIC_GATE_STOP' if worker['exit_code'] == 20 else 'EXECUTION_FAILURE'
        else:
            require(record.get('status') == 'INTERRUPTED_PROCESS' or
                    completion['run_statuses'].get(job['job_id']) == 'STOPPED_DEADLINE', 'Missing worker receipt lacks interruption evidence')
            outcome['failure_class'] = 'INTERRUPTED_PROCESS'
        outcomes.append(outcome)
    return files, outcomes


def validate_npz_metadata(path, registry):
    lo, hi = map(int, registry['split_ranges']['V_select'])
    L, H, C = (registry[k] for k in ('L', 'H', 'C'))
    require(hi == registry['T_start'] and lo >= 0 and lo < hi, 'Registered V/T boundary invalid')
    expected = np.arange(max(lo, L), hi-H+1, H, dtype=np.int64)
    require(len(expected) > 0, 'Registered V has no complete forecast windows')
    shapes = dict(origins=(len(expected),), channel_ids=(C,), mean=(C,), std=(C,),
                  pred_normalized=(len(expected), H, C), target_normalized=(len(expected), H, C), observed=(len(expected), H, C))
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(shapes) and set(archive.namelist()) == {k+'.npy' for k in shapes},
                'Prediction archive contains unexpected members')
        for name, shape in shapes.items():
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
                actual, _, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                    else np.lib.format.read_array_header_2_0(stream))
            require(tuple(actual) == shape and not dtype.hasobject, 'Prediction header axes/type invalid: '+name)
            if name in ('pred_normalized', 'target_normalized', 'mean', 'std'):
                require(dtype == np.dtype('float32'), 'Registered numeric archive must be FP32')
            elif name == 'observed':
                require(dtype == np.dtype('bool'), 'Observation mask must be boolean')
            elif name == 'origins':
                require(dtype == np.dtype('int64'), 'Origins must be int64 metadata')
    with np.load(path, allow_pickle=False) as archive:
        origins, columns = archive['origins'], archive['channel_ids']
        require(np.array_equal(origins, expected) and int(origins[-1])+H <= registry['T_start'],
                'Origins cross protected T or differ from exact V cohort; numeric arrays not read')
        require(columns.tolist() == registry['columns'], 'Channel identity/order differs; numeric arrays not read')
    return shapes['pred_normalized']


def verify_arrays(row, registry, cohort=None):
    path = Path(row['prediction_path'])
    require(runtime.sha(path) == row['prediction_sha256'], 'Prediction SHA changed')
    shape = validate_npz_metadata(path, registry)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(np.isfinite(arrays['mean']).all() and np.isfinite(arrays['std']).all() and np.all(arrays['std'] > 0),
            'Invalid normalization metadata')
    require(scaler_fingerprint(arrays['mean'], arrays['std']) == registry['normalization_sha256']['A60'],
            'Saved normalization differs from registered fresh A60 scaler')
    if cohort is not None:
        for key in COHORT_KEYS:
            require(np.array_equal(arrays[key], cohort[key], equal_nan=arrays[key].dtype.kind in 'fc'),
                    'Matched cohort/scaler differs: '+key)
    mask = arrays['observed']
    p, y = arrays['pred_normalized'], arrays['target_normalized']
    require(np.isfinite(p[mask]).all() and np.isfinite(y[mask]).all(), 'Nonfinite observed prediction/target')
    squared, absolute, counts = [], [], []
    for channel in range(shape[-1]):
        keep = mask[:, :, channel]
        errors = p[:, :, channel][keep].astype(np.float64)-y[:, :, channel][keep].astype(np.float64)
        require(len(errors) > 0, 'A registered channel has no observed targets')
        counts.append(len(errors))
        squared.append(math.fsum(float(e)*float(e) for e in errors)/len(errors))
        absolute.append(math.fsum(abs(float(e)) for e in errors)/len(errors))
    mse, mae = math.fsum(squared)/len(squared), math.fsum(absolute)/len(absolute)
    require(all(math.isfinite(float(row[k])) for k in ('MSE', 'MAE')) and
            max(abs(mse-row['MSE']), abs(mae-row['MAE'])) <= 1e-9, 'Independent FP64 metrics differ')
    require(row['n_origins'] == shape[0] and row['n_observed'] == sum(counts) and
            row['logical_F_rows'] == shape[0]*row['K'], 'Saved evaluation counts differ')
    return {key: arrays[key] for key in COHORT_KEYS}, dict(job_id=row['job_id'], status='PASS',
        prediction_sha256=row['prediction_sha256'], shape=list(shape), MSE=mse, MAE=mae,
        maximum_metric_error=max(abs(mse-row['MSE']), abs(mae-row['MAE'])), metadata_gate_before_numeric_access=True)


def verify_job(job, registration, completion):
    root = runtime.REPORT/'jobs'/job['job_id']
    result_path = root/'RESULT.json'; result = runtime.read(result_path)
    spec = run_one.build_spec(job)
    require(result.get('status') == 'COMPLETE' and result.get('job') == job and result.get('registration') == registration
            and result.get('spec') == spec and result.get('selection_sealed_before_V') is True, 'Result registration differs')
    attempt = result['attempt']; require(type(attempt) is int and attempt in (1, 2), 'Invalid successful attempt')
    public = root/f'attempt_{attempt:03d}'; cache = runtime.CACHE/'jobs'/job['job_id']/f'attempt_{attempt:03d}'
    names = ('START.json', 'SOURCE_CONTRACT.json', 'SEARCH_RESULT.json', 'SELECTION.json', 'REFIT_SUMMARY.json',
             'RECEIPT.json', 'OPENED_FILES.json', 'HOST_GPU_GATE.json')
    files = [result_path, root/f'WORKER_RECEIPT_attempt{attempt}.json', *(public/name for name in names)]
    require(all(p.is_file() for p in files), 'Successful job has missing execution evidence')
    start, receipt, worker = (runtime.read(p) for p in (public/'START.json', public/'RECEIPT.json', files[1]))
    require(start['job'] == job and start['registration'] == registration and start['spec'] == spec
            and start['attempt'] == attempt, 'Start identity differs')
    require(receipt['status'] == 'COMPLETE' and receipt['job'] == job and receipt['registration'] == registration
            and receipt['attempt'] == attempt and receipt['training_attempted'] is True
            and receipt['training_executed'] is True and receipt['T_numeric_values_parsed'] == 0
            and all(receipt[k]['changed'] == [] for k in ('preserved_before', 'preserved_after')), 'Invalid successful receipt')
    require(worker['status'] == 'COMPLETE' and worker['exit_code'] == 0 and worker['job_id'] == job['job_id']
            and worker['attempt'] == attempt, 'Worker completion differs')
    require(runtime.read(public/'OPENED_FILES.json')['T_numeric_values_parsed'] == 0, 'Protected T open audit failed')
    runtime.assert_protected_T(result)
    source = runtime.read(public/'SOURCE_CONTRACT.json')
    require(source['spec'] == spec and source.get('sources') and
            all(source[k] == result['contract'][k] for k in ('source_sha', 'config_sha', 'data_sha', 'model_sha')),
            'Scientific source/data contract differs')
    require(result['contract']['model_sha'] == runtime.F_SHA, 'Frozen foundation identity differs')
    selection_path = public/'SELECTION.json'
    require(Path(result['selection_path']).resolve() == selection_path.resolve(), 'Selection path escapes successful attempt')
    run_one.check_selection(selection_path, result['selection_sha256'], job)
    search, selection = runtime.read(public/'SEARCH_RESULT.json'), runtime.read(selection_path)
    epoch, score = run_one.choose_epoch(search, spec['epochs'])
    require(selection['spec'] == spec and selection['contract'] == result['contract'] and selection['selection_split'] == 'A_stop'
            and selection['curves'] == search['curves'] and selection['selected_epoch'] == epoch
            and selection['selected_astop_mse'] == score and result['search_summary'] == search['summary'], 'A_stop selection differs')
    refit = runtime.read(public/'REFIT_SUMMARY.json')
    require(refit['status'] == 'SEGMENT_COMPLETE' and refit == result['refit_summary'], 'Fresh refit incomplete')
    row = result['row']
    require(all(row.get(k) == v for k, v in job.items()) and row['reused'] is False and row['extended'] is False
            and row['selected_epoch'] == epoch and row['cap_selected'] == (epoch == job['training_cap'])
            and row['normalization_fit_end'] == runtime.FIT_ENDS[job['dataset']][1], 'Result row differs from fixed job')
    require(row['independent_FP64'] == 'PASS' and row['metric_accumulation_dtype'] == 'float64', 'Worker scoring evidence missing')
    for path_key, sha_key, expected in (('bundle_path', 'bundle_sha256', cache/'selected_bundle.pt'),
                                        ('prediction_path', 'prediction_sha256', cache/'selected_A60_V_select.npz')):
        owner = result if path_key == 'bundle_path' else row
        require(Path(owner[path_key]).resolve() == expected.resolve() and runtime.sha(expected) == owner[sha_key],
                'Immutable artifact path/SHA differs')
    gate = runtime.read(public/'HOST_GPU_GATE.json')
    binding = runtime.verify_host_gpu_binding()
    require(gate.get('host_gpu_binding') == binding and gate.get('registration_seal_sha256') == registration['seal_sha256'],
            'Host identity receipt is not bound to this study')
    helper = importlib.import_module(f'research.{runtime.NAME}.host_gpu_gate')
    require(helper.verify_identity_receipt(gate, runtime.REPO/runtime.HOST_GPU_CONTRACT_REL)['status'] == 'PASS',
            'Saved host identity snapshot failed')
    if job['method_id'] == 'E_FIXED':
        audits = result.get('E_INVARIANCE', [])
        require([a['phase'] for a in audits] == ['search', 'refit'] and
                all(a['status'] == 'PASS' and a['E_requires_grad'] is False and
                    len({a[k] for k in ('initial_E_sha256', 'live_E_sha256', 'selected_E_sha256')}) == 1 for a in audits),
                'Stage-local frozen E audit differs')
        audit_path = public/'E_INVARIANCE.json'; require(runtime.read(audit_path) == audits, 'Public E invariance evidence differs')
        files.append(audit_path)
    return result, files


def aggregate(rows):
    summary, contrasts, paired = [], [], []
    for dataset in DATASETS:
        subset = [r for r in rows if r['dataset'] == dataset]
        if not subset:
            continue
        by = {method: sorted((r for r in subset if r['method_id'] == method), key=lambda r:r['seed']) for method in METHODS}
        seeds = [r['seed'] for r in by['OURS']]; n = len(seeds)
        require(n in (2, 3) and seeds == [92601, 92602, 92603][:n]
                and all([r['seed'] for r in by[m]] == seeds for m in METHODS), 'Only complete matched 2/3-seed cohorts may be aggregated')
        stats = {}
        for method, values in by.items():
            cap_count = sum(r['selected_epoch'] == r['training_cap'] for r in values)
            undertrained = cap_count >= math.ceil(2*n/3)
            stat = dict(dataset=dataset, method_id=method, n=n, seed_ids=json.dumps(seeds),
                cap_selected_count=cap_count, cap=values[0]['training_cap'], undertrained=undertrained,
                pilot=n == 2, provisional=n == 2 or undertrained, exploratory=True,
                selected_epochs=json.dumps([r['selected_epoch'] for r in values]),
                trainable_parameters=json.dumps([r['trainable_parameters'] for r in values]))
            for metric in ('MSE', 'MAE'):
                numbers = [float(r[metric]) for r in values]
                require(all(math.isfinite(v) and v >= 0 for v in numbers), 'Invalid aggregate metric')
                q1, median, q3 = map(float, np.quantile(numbers, [.25, .5, .75]))
                stat.update({metric+'_median':median, metric+'_q1':q1, metric+'_q3':q3,
                             metric+'_IQR':q3-q1, metric+'_min':min(numbers), metric+'_max':max(numbers)})
            stats[method] = stat; summary.append(stat)
        for method in METHODS[1:]:
            full, arm = stats['OURS'], stats[method]
            for metric in ('MSE', 'MAE'):
                deltas, percents = [], []
                for baseline, other in zip(by['OURS'], by[method]):
                    delta = other[metric]-baseline[metric]
                    pct = None if baseline[metric] == 0 else 100*delta/baseline[metric]
                    deltas.append(delta); percents.append(pct)
                    paired.append(dict(dataset=dataset, method_id=method, seed=baseline['seed'], metric=metric,
                        OURS=baseline[metric], arm=other[metric], delta_arm_minus_OURS=delta, delta_percent=pct,
                        primary_n=n, pilot=n == 2, comparison_provisional=full['provisional'] or arm['provisional']))
                delta = arm[metric+'_median']-full[metric+'_median']
                threshold = max(arm[metric+'_IQR'], full[metric+'_IQR'])
                positives, negatives = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
                direction = ('FULL_LOWER' if delta > threshold and positives >= math.ceil(2*n/3) else
                             'ARM_LOWER' if -delta > threshold and negatives >= math.ceil(2*n/3) else 'UNCLEAR')
                provisional = full['provisional'] or arm['provisional']
                contrasts.append(dict(dataset=dataset, method_id=method, metric=metric, n=n,
                    full_median=full[metric+'_median'], arm_median=arm[metric+'_median'],
                    delta_of_medians=delta, delta_of_medians_percent=None if full[metric+'_median'] == 0 else 100*delta/full[metric+'_median'],
                    median_paired_delta=float(np.median(deltas)),
                    median_paired_delta_percent=None if any(v is None for v in percents) else float(np.median(percents)),
                    max_arm_IQR=threshold, full_lower_seeds=positives, arm_lower_seeds=negatives, ties=n-positives-negatives,
                    descriptive_direction=direction, comparison_status='PROVISIONAL' if provisional else direction,
                    pilot=n == 2, provisional=provisional, exploratory=True,
                    full_cap_selected_count=full['cap_selected_count'], arm_cap_selected_count=arm['cap_selected_count'],
                    significance_test=False, equivalence_test=False))
    return summary, contrasts, paired


def make_figures(rows, coverage, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for method, stem, label in COMPONENTS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, dataset in zip(axes, DATASETS):
            group = next(c for c in coverage if c['dataset'] == dataset)
            n = group['primary_n']; ax.set_title(f'{dataset} | n={n} | '+group['status'])
            if n == 0:
                ax.text(.5, .5, 'No complete matched base cohort', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off(); continue
            for seed, color in zip([92601, 92602, 92603][:n], ('#0072B2', '#D55E00', '#009E73')):
                pair = [next(r for r in rows if (r['dataset'], r['method_id'], r['seed']) == (dataset, m, seed)) for m in ('OURS', method)]
                ax.plot([0, 1], [r['MSE'] for r in pair], 'o-', color=color, label=str(seed), alpha=.85)
            ax.set_xticks([0, 1], ['Fresh OURS', method]); ax.set_ylabel('Observed-channel macro MSE (V)')
            ax.grid(axis='y', alpha=.2); ax.legend(title='Matched seed', fontsize=8)
            ax.text(.5, -.19, 'Fixed common cap; cap-selected comparisons may remain provisional',
                    ha='center', va='top', transform=ax.transAxes, fontsize=7)
        fig.suptitle(label+'\nHistorically selected development V; exploratory, not an independent test', fontsize=11)
        fig.savefig(output/(stem+'.png'), dpi=180); plt.close(fig)


def write_report(coverage, contrasts, status_rows, output):
    lines = ['# 교차 데이터 구성요소 탐색 결과', '',
        'PeMS32·Beijing은 과거 개발 V 결과를 참고해 선택했다. 새 미관측 데이터 일반화 시험이 아니다. '
        'K는 PeMS32=8, Beijing=6으로 고정했으므로 K 또는 Kτ 선택 규칙을 검증하지 않는다.', '',
        '각 데이터의 fresh OURS와 같은 seed·전체 4구성이 모두 끝난 base만 비교한다. '
        '선택 seed92603은 네 구성 모두 완료된 경우에만 함께 집계했다. 부분 결과는 ALL_NEW_RUNS/JOB_STATUS에 보존한다.', '']
    for group in coverage:
        lines.append(f"- {group['dataset']}: base {group['base_completed']}/8, optional {group['optional_completed']}/4, "
                     f"비교 n={group['primary_n']} ({group['status']}).")
    lines += ['', '차이(+)는 제거/고정한 구성의 오차가 더 커 fresh OURS가 낮다는 뜻이다. '
        '중앙값 차이와 seed별 차이의 중앙값은 다른 통계이며 표에서 분리했다. '
        '중앙값 차이가 두 IQR의 최댓값을 넘고 같은 방향 seed가 ceil(2n/3) 이상이면 기술적 방향만 표시한다. '
        '유의성·동등성 검정이 아니다.', '']
    for contrast in contrasts:
        if contrast['metric'] != 'MSE': continue
        pct = contrast['delta_of_medians_percent']; delta = '정의 불가(기준 0)' if pct is None else f'{pct:+.2f}%'
        lines.append(f"- {contrast['dataset']} / {contrast['method_id']}: MSE 중앙값 차이 {delta}; "
                     f"기술적 방향 {contrast['descriptive_direction']}; 상태 {contrast['comparison_status']}.")
    lines += ['', '## 해석과 한계', '',
        '- n=2는 전부 PILOT/잠정이며 n=3도 탐색 결과다. 어느 한 구성이라도 cap 선택수가 ceil(2n/3) 이상이면 해당 비교는 잠정이다. '
        '좋은 방향이더라도 충분한 최적화나 확정적 구성요소 효과를 주장하지 않는다.',
        '- PeMS32의 cap6은 모든 구성에 동일하며 추가 연장하지 않았다. Beijing은 모두 cap3이다. 같은 cap이 같은 실제 선택 epoch나 계산량을 뜻하지 않는다.',
        '- TIME_ONLY는 채널 MLP를 제거하지만 공동 LayerNorm은 유지한다. 따라서 채널 상호작용 전체의 제거, 시간축 자체의 보편적 우월성 또는 용량을 완전히 통제한 대조가 아니다.',
        '- A5_LAG0는 직접 anchor와 lag0를 유지한다. 추가 lag 참조의 구성 효과이며 개별 lag의 인과성·최적성이나 모든 관측값의 필요성 시험이 아니다.',
        '- E_FIXED는 search/refit 각각의 PCA 초기 E를 고정한다. E 학습 여부의 대조이며 PCA 초기화, 압축의 최적성, K 규칙 또는 독립 테스트 우월성을 단독으로 입증하지 않는다.',
        '- 다른 데이터의 과거 좋은 결과와 새 결과를 하나의 seed 표본으로 섞지 않는다. 미완료·실패·예산 제외도 JOB_STATUS에 모두 공개한다.', '']
    text = '\n'.join(lines)
    (output/'RESULTS_KO.md').write_text(text, encoding='utf8')
    handoff = '# 실험 근거 전달\n\n발표 파일·대본은 생성하지 않았다. 아래는 작성 담당자가 사용할 수 있는 실험 범위와 제한이다.\n\n'+text
    (output/'EVIDENCE_HANDOFF_KO.md').write_text(handoff, encoding='utf8')


def main():
    plan = runtime.read(runtime.REPORT/'RUN_PLAN.json'); jobs = verify_plan(plan)
    registration = run_one.validate_registration(jobs[0])
    completion = runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
    counts = verify_completion(jobs, completion)
    registry = runtime.read(runtime.REPORT/'DATA_REGISTRY.json')['datasets']
    primary_ids, coverage = select_cohorts(jobs, completion)
    all_rows, status_rows, checks, files, cohorts = [], [], [], [], {}
    for job in jobs:
        status = completion['run_statuses'].get(job['job_id'], 'NOT_STARTED')
        result_path = runtime.REPORT/'jobs'/job['job_id']/'RESULT.json'
        require((status == 'COMPLETE') == result_path.exists(), 'Controller/result existence differs: '+job['job_id'])
        successful_attempt = runtime.read(result_path)['attempt'] if status == 'COMPLETE' else None
        attempt_files, outcomes = verify_attempts(job, registration, completion, successful_attempt)
        files.extend(attempt_files)
        display_status = 'STOPPED_INTERRUPTED' if completion['status'] == 'STOPPED' and status in ('RUNNING','FAILED_ONCE') else status
        status_rows.append(dict(job, status=display_status, recorded_status=status, primary_included=job['job_id'] in primary_ids,
            optional=job['seed'] == 92603, attempt_count=len(outcomes), attempt_outcomes=json.dumps(outcomes),
            stop_reason=completion.get('stop_reason', '') if status != 'COMPLETE' else ''))
        if status != 'COMPLETE':
            continue
        result, evidence = verify_job(job, registration, completion); files.extend(evidence)
        row = dict(result['row']); row['primary_included'] = job['job_id'] in primary_ids; row['optional'] = job['seed'] == 92603
        cohort, check = verify_arrays(row, registry[job['dataset']], cohorts.get(job['dataset']))
        cohorts.setdefault(job['dataset'], cohort); checks.append(check); all_rows.append(row)
    primary = [row for row in all_rows if row['primary_included']]
    summary, contrasts, paired = aggregate(primary)
    for row in summary:
        saved = completion.get('method_status', {}).get(row['method_id']+':'+row['dataset'])
        if saved is not None:
            require(saved['cap_selected_count'] == row['cap_selected_count'] and saved['provisional'] == row['provisional'],
                    'Controller cap/provisional disposition differs')
    runtime.guard_preserved()
    tables = [('RUNS.csv', primary, ('job_id','dataset','method_id','seed','MSE','MAE')),
              ('ALL_NEW_RUNS.csv', all_rows, ('job_id','dataset','method_id','seed','MSE','MAE')),
              ('JOB_STATUS.csv', status_rows, ('job_id','status')), ('COVERAGE.csv', coverage, ('dataset','primary_n','status')),
              ('SUMMARY.csv', summary, ('dataset','method_id','n')), ('CONTRASTS.csv', contrasts, ('dataset','method_id','metric','n')),
              ('PAIRED_DIFFERENCES.csv', paired, ('dataset','method_id','seed','metric'))]
    for name, rows, fields in tables: write_csv(runtime.REPORT/name, rows, fields)
    make_figures(primary, coverage, runtime.REPORT); write_report(coverage, contrasts, status_rows, runtime.REPORT)
    output_names = [t[0] for t in tables]+['RESULTS_KO.md','EVIDENCE_HANDOFF_KO.md']+[stem+'.png' for _,stem,_ in COMPONENTS]
    files += [runtime.REPORT/name for name in output_names]
    files += [runtime.REPORT/name for name in ('RUN_PLAN.json','PREREGISTRATION.json','REGISTRATION_SEAL.json','DATA_REGISTRY.json','EXECUTION_COMPLETION.json')]
    validation = dict(status='STOPPED' if completion['status'] == 'STOPPED' else 'PASS' if primary else 'INCOMPLETE', integrity_status='PASS',
        created_utc=datetime.now(timezone.utc).isoformat(), analysis_scope='Matched exploratory development-V component screen',
        completed_count=len(all_rows), primary_count=len(primary), planned_base=16, planned_optional=8, extension_jobs=0,
        execution_status=completion['status'], execution_counts=counts,
        coverage=coverage, summary=summary, contrasts=contrasts, prediction_checks=checks,
        max_metric_recomputation_error=max((c['maximum_metric_error'] for c in checks), default=0.),
        T_numeric_values_parsed=0, GPU_inference_performed=False, independent_test=False,
        artifact_verification_scope='Registered sources, execution/selection/refit/host receipts, immutable bundle hashes and stage E audit; no checkpoint tensor replay.',
        verified_files={p.relative_to(runtime.REPO).as_posix():runtime.sha(p) for p in sorted(set(files))},
        postprocess_source_sha256=runtime.sha(Path(__file__)))
    runtime.atomic(runtime.REPORT/'ANALYSIS_VALIDATION.json', validation)
    print(json.dumps(dict(status=validation['status'], completed=len(all_rows), primary=len(primary), coverage=coverage), ensure_ascii=False))


if __name__ == '__main__':
    main()
