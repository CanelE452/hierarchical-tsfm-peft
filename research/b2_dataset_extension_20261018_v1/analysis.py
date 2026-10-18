"""Validate completed evidence and render the fixed dataset comparison."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


NAME = 'b2_dataset_extension_20261018_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS = (92601, 92602)
DATASETS = ('uci235_household_hourly', 'uci274_sml2010')
LABELS = {'jena': 'Jena*', 'uci235_household_hourly': 'Household', 'uci274_sml2010': 'SML2010'}
AXES = {'uci235_household_hourly': (7, 2, 72, 72), 'uci274_sml2010': (21, 6, 9, 8)}
OBSERVED_TARGETS = {'uci235_household_hourly': 23828, 'uci274_sml2010': 7791}
FIELDS = ('scope', 'dataset', 'method', 'family', 'run_id', 'seed', 'C', 'K', 'L', 'H',
    'B', 'row_cap', 'F_calls', 'F_streams', 'MSE', 'MAE', 'p50_ms', 'q25_ms', 'q75_ms',
    'p95_ms', 'peak_allocated_MiB', 'peak_reserved_MiB', 'selected_epoch', 'cap',
    'training_status', 'n_origins', 'scored_origins', 'observed_targets', 'source_reference',
    'source_sha256', 'prediction_sha256', 'F_state_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def score(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if prediction.shape != target.shape or target.shape != mask.shape or mask.dtype != np.bool_ or target.ndim != 3:
        raise ValueError('Aligned three-dimensional predictions/targets and bool mask required')
    counts = mask.sum((0, 1), dtype=np.int64)
    if (counts <= 0).any() or not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Every channel needs finite observed targets and predictions')
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    sse, sae = (np.sum(value, (0, 1), dtype=np.float64) for value in (error * error, np.abs(error)))
    return dict(MSE=math.fsum((sse / counts).tolist()) / len(counts),
                MAE=math.fsum((sae / counts).tolist()) / len(counts), observed=int(counts.sum()))


def _close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
        raise ValueError('Independent recomputation differs: ' + label)


def validate_cost_receipt(receipt, arrays):
    dataset, family, seed = (receipt.get(key) for key in ('dataset', 'family', 'seed'))
    if dataset not in AXES or family not in ('RAW', 'B2', 'DIRECT_G') or seed not in SEEDS:
        raise ValueError('Unexpected dataset/model/seed identity')
    channels, k, origins, scored_origins = AXES[dataset]
    if (receipt.get('status') != 'COMPLETE' or receipt.get('batch_size') != 8
            or (receipt.get('C'), receipt.get('K'), receipt.get('L'), receipt.get('H')) != (channels, k, 512, 48)
            or receipt.get('device') != 'cuda' or receipt.get('warmups') != 10
            or receipt.get('timed_repetitions') != 20 or receipt.get('TS2_or_prediction_cache_used') is not False
            or receipt.get('validation_allocator_cache_released_before_warmup') is not True):
        raise ValueError('Incomplete matched B8 FP32 cost contract')
    expected_states = {'F'} if family == 'RAW' else {'F', 'adapter'} if family == 'B2' else {'adapter'}
    if receipt.get('states_before') != receipt.get('states_after') or set(receipt.get('states_before', {})) != expected_states:
        raise ValueError('Loaded tensor states changed or required state proof is absent')
    cap = None if family == 'DIRECT_G' else 8 * k
    width = channels if family == 'RAW' else k if family == 'B2' else 0
    validation = receipt.get('validation', {})
    batches = [8] * (origins // 8) + ([origins % 8] if origins % 8 else [])
    calls = [math.ceil(batch * width / cap) if width else 0 for batch in batches]
    if (receipt.get('row_cap') != cap or validation.get('row_cap') != cap
            or validation.get('status') != 'PASS' or validation.get('full_V_origins') != origins
            or validation.get('batch_sizes') != batches or validation.get('last_batch_size') != batches[-1]
            or validation.get('F_calls') != calls):
        raise ValueError('Whole V coverage, final batch or F calls differ')
    for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
        if array_fingerprint(arrays[key]) != validation.get('fingerprints', {}).get(key):
            raise ValueError('Scored-array fingerprint differs: ' + key)
    current, reference = arrays['pred_normalized'], arrays['reference_pred_normalized']
    if (array_fingerprint(current) != validation.get('output_fingerprint')
            or array_fingerprint(reference) != receipt['original_reference'].get('prediction_fingerprint')
            or not np.allclose(current, reference, atol=1e-5, rtol=1e-4)):
        raise ValueError('Original row128 and matched-cap prediction parity differs')
    mask = arrays['observed']
    if (current.shape != (origins, 48, channels) or arrays['origins'].shape != (origins,)
            or len(np.unique(arrays['origins'])) != origins
            or int(np.any(mask, axis=(1, 2)).sum()) != scored_origins):
        raise ValueError('Actual scored origin coverage differs')
    scores = score(current, arrays['target_normalized'], mask)
    if scores['observed'] != OBSERVED_TARGETS[dataset]:
        raise ValueError('Observed target coverage differs from the fixed data audit')
    canonical = score(reference, arrays['target_normalized'], mask)
    for key in ('MSE', 'MAE'):
        _close(scores[key], validation['metrics'][key], key + ' matched cap')
        _close(canonical[key], receipt['original_reference']['metrics'][key], key + ' original row128')
        _close(scores[key], canonical[key], key + ' parity', atol=1e-5, rtol=1e-4)
    trials = receipt.get('trials', [])
    for phase, repetitions in (('warmup', 10), ('timed', 20)):
        subset = [trial for trial in trials if trial.get('phase') == phase]
        if len(subset) != repetitions or [trial.get('iteration') for trial in subset] != list(range(repetitions)):
            raise ValueError('Exact warmup/timed trial coverage differs')
        for trial in subset:
            if (trial.get('batch_size') != 8 or trial.get('F_calls') != (math.ceil(8 * width / cap) if width else 0)
                    or trial.get('F_streams') != 8 * width or trial.get('row_cap') != cap
                    or not math.isfinite(trial.get('whole_ms', math.nan)) or trial['whole_ms'] <= 0):
                raise ValueError('Timed whole path or actual F counters differ')
    if len(trials) != 30:
        raise ValueError('Unregistered additional timing trials')
    timed = [trial for trial in trials if trial['phase'] == 'timed']
    q25, p50, q75, p95 = np.quantile([trial['whole_ms'] for trial in timed], [.25, .5, .75, .95])
    summary = dict(q25_ms=float(q25), p50_ms=float(p50), q75_ms=float(q75), p95_ms=float(p95),
                   iqr_ms=float(q75 - q25), origins_per_second_at_median=8000 / p50)
    for key in ('peak_allocated_MiB', 'peak_reserved_MiB', 'RSS_MiB'):
        if any(not math.isfinite(float(trial[key])) or float(trial[key]) < 0 for trial in timed):
            raise ValueError('Invalid timing memory statistic')
        summary[key] = max(float(trial[key]) for trial in timed)
    if summary['peak_reserved_MiB'] < summary['peak_allocated_MiB']:
        raise ValueError('Reserved memory is smaller than allocated memory')
    for key, value in summary.items():
        _close(value, receipt['summary'][key], key)
    if receipt['summary'].get('repetitions') != 20:
        raise ValueError('Timing summary repetition count differs')
    return dict(metrics=canonical, summary=summary, n_origins=origins,
                scored_origins=scored_origins, observed_targets=scores['observed'])


def _bound(value, repo, cache, report):
    key, relative = value['reference'].split('://', 1)
    root = {'repo': repo, 'cache': cache, 'report': report}.get(key)
    if root is None:
        raise ValueError('Unsupported experiment artifact binding')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or NAME not in path.parts or sha(path) != value['sha256']:
        raise ValueError('New experiment binding changed or escaped its namespace')
    return path


def validate_training(repo):
    report = repo / 'reports' / NAME
    selection_path = report / 'ALL_SEARCHES_SEALED.json'
    selection = read(selection_path)
    expected = {f'B2_{dataset}_K{AXES[dataset][1]}_OFF_s{seed}' for dataset in DATASETS for seed in SEEDS}
    if (selection.get('status') != 'SEALED' or selection.get('all_searches_complete') is not True
            or selection.get('V_used') is not False or selection.get('cap') != 3
            or selection.get('selection_split') != 'A_stop' or set(selection.get('matched_arms', [])) != expected
            or set(selection.get('selections', {})) != expected or set(selection.get('datasets', {})) != set(DATASETS)):
        raise ValueError('All four fixed searches must be sealed before fresh refit/V')
    evidence = {}
    for dataset in DATASETS:
        cache, public = repo / '.cache' / NAME / 'datasets' / dataset, report / 'datasets' / dataset
        _bound(selection['datasets'][dataset]['identity_binding'], repo, cache, public)
        identity, summary = read(public / 'TRAIN_IDENTITY.json'), read(public / 'TRAIN_SUMMARY.json')
        if (summary.get('status') != 'COMPLETE' or summary.get('new_searches') != 2 or summary.get('fresh_refits') != 2
                or len(summary.get('rows', [])) != 2 or identity.get('epochs') != [.25, 1., 3.]
                or identity.get('cap') != 3 or identity.get('revin') is not False):
            raise ValueError('Required fixed B2 fits/refits are incomplete')
        if _bound(summary['selection_binding'], repo, cache, public) != selection_path:
            raise ValueError('Fresh refit points to a different all-search selection seal')
        for relative, expected_sha in identity['source_hashes'].items():
            source = (repo / relative).resolve()
            if not source.is_relative_to(repo) or source.suffix != '.py' or sha(source) != expected_sha:
                raise ValueError('Training producer source changed: ' + relative)
        for row in summary['rows']:
            job = row['run_id']
            _bound(selection['selections'][job]['checkpoint'], repo, cache, public)
            bundle = _bound(row['bundle_binding'], repo, cache, public)
            _bound(row['prediction_binding'], repo, cache, public)
            search = read(public / 'runs' / job / 'cap3' / 'SEARCH_RESULT.json')
            curves = [(float(value['epoch']), float(value['astop_mse'])) for value in search['curves']
                      if value.get('astop_mse') is not None]
            if ([epoch for epoch, value in curves] != [0., .25, 1., 3.]
                    or min(curves, key=lambda item: (item[1], item[0]))[0] != row['selected_epoch']
                    or row.get('all_searches_sealed_before_V') is not True):
                raise ValueError('A_stop selection or pre-V seal differs')
            last_best = curves[-1][1] < min(value for _, value in curves[:-1])
            if selection['selections'][job]['epoch'] != row['selected_epoch']:
                raise ValueError('Fresh refit epoch differs from the global search decision')
            evidence[job] = dict(selected_epoch=row['selected_epoch'], cap=3,
                training_status='CAP3_STILL_IMPROVING' if last_best else 'SELECTED_EPOCH_ZERO' if row['selected_epoch'] == 0 else 'CAP3_CONVERGENCE_UNRESOLVED',
                checkpoint_sha256=sha(bundle), search_train_origins=search['summary']['train_origins'],
                MSE=row['MSE'], MAE=row['MAE'])
        for seed in SEEDS:
            job = f'DirectMLP_{dataset}_s{seed}'
            fit = read(public / (job + '_FIT.json'))
            if (fit.get('status') != 'COMPLETE_VALID' or fit.get('family') != 'DIRECT_G'
                    or fit.get('epochs') != [.25, 1., 3.] or fit.get('seed') != seed
                    or fit.get('F_instances') != 0):
                raise ValueError('Required matched Direct MLP fits are incomplete')
            artifact = _bound(fit['bundle_binding'], repo, cache, public)
            evidence[job] = dict(selected_epoch=fit['selected_epoch'], cap=3,
                training_status='CAP3_STILL_IMPROVING' if fit.get('last_cp_best') and fit.get('last_scores_still_falling')
                    else 'SELECTED_EPOCH_ZERO' if fit['selected_epoch'] == 0 else 'CAP3_CONVERGENCE_UNRESOLVED',
                checkpoint_sha256=sha(artifact), search_train_origins=fit['search_train_origins'])
    return evidence


def validate_auxiliary(repo):
    report, cache = repo / 'reports' / NAME, repo / '.cache' / NAME
    amendment_path = report / 'MEASUREMENT_AMENDMENT_001.json'
    amendment = read(amendment_path)
    if (amendment.get('status') != 'PRESERVED_AUXILIARY_WORKSPACE_CONFIGURATION'
            or amendment.get('old_config') != ':4096:8' or amendment.get('new_training_fits') != 0
            or amendment.get('original_attempts_deleted') is not False):
        raise ValueError('The workspace-configuration technical repeat is not preserved')
    _bound(dict(reference=amendment['old_source_reference'], sha256=amendment['old_source_sha256']),
           repo, cache, report)
    expected = {(dataset, family, seed) for dataset in DATASETS for family in ('B2', 'DIRECT_G') for seed in SEEDS}
    expected |= {(dataset, 'RAW', 92601) for dataset in DATASETS}
    seen, rows = set(), {}
    for item in amendment['routes']:
        path = _bound(dict(reference=item['archive_report_reference'], sha256=item['receipt_sha256']),
                      repo, cache, report)
        prediction = _bound(dict(reference=item['archive_prediction_reference'], sha256=item['prediction_sha256']),
                            repo, cache, report)
        receipt = read(path)
        role = (receipt['dataset'], receipt['family'], receipt['seed'])
        if (role not in expected or role in seen or receipt['run_id'] != item['run_id']
                or receipt['source_sha256'] != amendment['old_source_sha256']
                or receipt['prediction_file_sha256'] != item['prediction_sha256']):
            raise ValueError('Auxiliary route/source/prediction identity differs')
        seen.add(role)
        with np.load(prediction, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        checked = validate_cost_receipt(receipt, arrays)
        matched_metrics = score(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'])
        for key in ('MSE', 'MAE'):
            _close(matched_metrics[key], item[key], 'auxiliary ' + key)
        for key, value in checked['summary'].items():
            _close(value, item['summary'][key], 'auxiliary ' + key)
        rows[item['run_id']] = dict(metrics=checked['metrics'], summary=checked['summary'])
    if (seen != expected or len(rows) != 10 or amendment.get('auxiliary_cost_routes') != 10
            or amendment.get('auxiliary_warmup_trials') != 100 or amendment.get('auxiliary_timed_trials') != 200):
        raise ValueError('All ten auxiliary routes and their complete timing trials must be accounted for')
    return dict(status='AUXILIARY_WORKSPACECONFIG', cost_routes=10, warmup_trials=100,
                timed_trials=200, training_reruns=0, amendment_sha256=sha(amendment_path), rows=rows)


def collect(repo=REPO):
    repo = Path(repo).resolve()
    training = validate_training(repo)
    report = repo / 'reports' / NAME
    paths = sorted((report / 'datasets').glob('*/costs/*.json'))
    if len(paths) != 10:
        raise ValueError(f'Exactly ten completed route receipts required; found {len(paths)}')
    rows, seen, cohorts, f_states, processes = [], set(), {}, set(), set()
    diagnostics = {item['dataset']: item for item in read(report / 'DATA_DIAGNOSTICS.json')['datasets']}
    hashes = {}
    for path in paths:
        receipt = read(path)
        if 'cublas_workspace_config' not in receipt or receipt['cublas_workspace_config'] is not None:
            raise ValueError('Canonical costs must record the current inherited unset workspace setting')
        dataset, family, seed = (receipt.get(key) for key in ('dataset', 'family', 'seed'))
        role = (dataset, family, seed)
        if role in seen:
            raise ValueError('Duplicate measured route identity')
        seen.add(role)
        expected_job = f'Raw_{dataset}' if family == 'RAW' else f'B2_{dataset}_K{AXES[dataset][1]}_OFF_s{seed}' if family == 'B2' else f'DirectMLP_{dataset}_s{seed}'
        if receipt['run_id'] != expected_job or path.name != expected_job + '_B8.json':
            raise ValueError('Measured file/run identity differs')
        cache = repo / '.cache' / NAME / 'datasets' / dataset
        prediction = _bound(dict(reference=receipt['prediction_reference'], sha256=receipt['prediction_file_sha256']), repo, cache, path.parent.parent)
        with np.load(prediction, allow_pickle=False) as archive:
            if set(archive.files) != {'pred_normalized', 'reference_pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'}:
                raise ValueError('Unregistered prediction archive schema')
            arrays = {key: archive[key].copy() for key in archive.files}
        checked = validate_cost_receipt(receipt, arrays)
        if (not np.array_equal(arrays['origins'], diagnostics[dataset]['origins']['V_select'])
                or arrays['channel_ids'].astype(str).tolist() != diagnostics[dataset]['columns']):
            raise ValueError('Scored origins/channels differ from the registered data diagnostic')
        cohort = {key: array_fingerprint(arrays[key]) for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')}
        if dataset in cohorts and cohorts[dataset] != cohort:
            raise ValueError('Raw/B2/MLP scored different targets, masks, origins or scalers')
        cohorts[dataset] = cohort
        processes.add(receipt.get('pid'))
        if family != 'DIRECT_G':
            f_states.add(receipt['F_state_sha256'])
        factory = 'baselines.py' if family == 'DIRECT_G' else 'train.py'
        if (receipt['source_sha256'] != sha(repo / 'research' / NAME / 'costs.py')
                or receipt['factory_source_sha256'] != sha(repo / 'research' / NAME / factory)):
            raise ValueError('Cost producer source bytes changed')
        fit = training.get(expected_job, dict(selected_epoch='', cap=0, training_status='ZERO_SHOT'))
        if family != 'RAW' and receipt['checkpoint_sha256'] != fit['checkpoint_sha256']:
            raise ValueError('Cost checkpoint differs from the completed training identity')
        if family == 'B2':
            for key in ('MSE', 'MAE'):
                _close(checked['metrics'][key], fit[key], key + ' saved training accuracy')
        summary = checked['summary']
        rows.append(dict(scope='NEW_MATCHED_B8', dataset=dataset, method={'RAW': 'Raw', 'B2': 'Ours B2', 'DIRECT_G': 'Direct MLP'}[family],
            family=family, run_id=expected_job, seed=seed, C=receipt['C'],
            K=receipt['C'] if family == 'RAW' else receipt['K'] if family == 'B2' else 0, L=512, H=48, B=8,
            row_cap=receipt['row_cap'], F_calls=receipt['trials'][-1]['F_calls'], F_streams=receipt['trials'][-1]['F_streams'],
            **checked['metrics'], **{key: summary[key] for key in ('p50_ms', 'q25_ms', 'q75_ms', 'p95_ms', 'peak_allocated_MiB', 'peak_reserved_MiB')},
            selected_epoch=fit['selected_epoch'], cap=fit['cap'], training_status=fit['training_status'],
            n_origins=checked['n_origins'], scored_origins=checked['scored_origins'], observed_targets=checked['observed_targets'],
            source_reference=path.relative_to(repo).as_posix(), source_sha256=sha(path),
            prediction_sha256=receipt['prediction_file_sha256'], F_state_sha256=receipt['F_state_sha256']))
        hashes[path.relative_to(repo).as_posix()] = sha(path)
    expected = {(dataset, family, seed) for dataset in DATASETS for family in ('B2', 'DIRECT_G') for seed in SEEDS}
    expected |= {(dataset, 'RAW', 92601) for dataset in DATASETS}
    if seen != expected or len(f_states) != 1:
        raise ValueError('Registered route coverage or shared frozen F identity differs')
    historical = repo / 'reports' / 'memory_budget_20261018_v1' / 'MEMORY_COST_SUMMARY.csv'
    with historical.open(encoding='utf-8-sig', newline='') as stream:
        history = [row for row in csv.DictReader(stream) if row['dataset'] == 'jena' and row['B'] == '8'
                   and row['stage'] == 'confirm' and row['effective_row_microbatch'] == '48'
                   and (row['family'] in ('RAW', 'LORA') or row['family'] == 'B2' and row['K'] == '6')]
    if len(history) != 4:
        raise ValueError('Exactly four historical Jena context rows required')
    for old in history:
        if old['status'] != 'COMPLETE' or old['parity_status'] != 'PASS' or old['warmups'] != '10' or old['timed_trials'] != '20':
            raise ValueError('Historical Jena context is not a confirmed comparison')
        rows.append(dict(scope='HISTORICAL_JENA_CONTEXT', dataset='jena', method={'RAW': 'Raw', 'B2': 'Ours B2', 'LORA': 'LoRA'}[old['family']],
            family=old['family'], run_id=old['accuracy_run_id'], seed=int(old['seed']),
            **{key: int(old[key]) for key in ('C', 'K', 'L', 'H')}, B=8, row_cap=48,
            F_calls=int(old['F_calls']), F_streams=int(old['F_logical_streams']),
            **{key: float(old[key]) for key in ('MSE', 'MAE', 'peak_allocated_MiB', 'peak_reserved_MiB')},
            p50_ms=float(old['median_batch_ms']), p95_ms=float(old['p95_batch_ms']),
            training_status='HISTORICAL_CONTEXT', source_reference=historical.relative_to(repo).as_posix(),
            source_sha256=sha(historical), prediction_sha256=old['prediction_sha256'], F_state_sha256=old['loaded_F_state_sha256']))
    auxiliary = validate_auxiliary(repo)
    differences = []
    for row in rows:
        if row['scope'] != 'NEW_MATCHED_B8':
            continue
        old = auxiliary['rows'][row['run_id']]
        for key in ('MSE', 'MAE'):
            _close(old['metrics'][key], row[key], 'workspace repeat ' + key, atol=1e-5, rtol=1e-4)
        differences.append(old['summary']['peak_allocated_MiB'] - row['peak_allocated_MiB'])
    return rows, dict(status='PASS', new_routes=10, historical_context_rows=4,
        timed_trials=200, warmup_trials=100, training=training, receipt_sha256=hashes,
        auxiliary=auxiliary, total_cost_routes=20, total_warmup_trials=200, total_timed_trials=400,
        workspace_auxiliary_minus_canonical_allocated_MiB=dict(minimum=min(differences), maximum=max(differences)),
        distinct_recorded_pids=len(processes), process_identity_note='OS PIDs may be reused; distinct PID count is descriptive only',
        cohort_fingerprints=cohorts, full_private_predictions_verified=True)


def aggregate(rows, dataset, family):
    selected = [row for row in rows if row['dataset'] == dataset and row['family'] == family]
    if not selected:
        raise ValueError('Missing aggregate group')
    result = dict(n=len(selected), seeds=[row['seed'] for row in selected])
    for metric in ('MSE', 'MAE', 'p50_ms', 'peak_allocated_MiB', 'peak_reserved_MiB'):
        values = [float(row[metric]) for row in selected]
        result[metric] = dict(mean=math.fsum(values) / len(values), minimum=min(values), maximum=max(values))
    return result


def dominates(a, b):
    metrics = ('MSE', 'p50_ms', 'peak_allocated_MiB')
    return all(a[key] <= b[key] for key in metrics) and any(a[key] < b[key] for key in metrics)


def comparisons(rows):
    result = {}
    for dataset in DATASETS:
        raw, ours, direct = (aggregate(rows, dataset, family) for family in ('RAW', 'B2', 'DIRECT_G'))
        mean = lambda group: {key: value['mean'] for key, value in group.items() if isinstance(value, dict)}
        raw_mean, ours_mean, direct_mean = map(mean, (raw, ours, direct))
        paired = {}
        for seed in SEEDS:
            a = next(row for row in rows if (row['dataset'], row['family'], row['seed']) == (dataset, 'DIRECT_G', seed))
            b = next(row for row in rows if (row['dataset'], row['family'], row['seed']) == (dataset, 'B2', seed))
            paired[str(seed)] = dominates(a, b)
        result[dataset] = dict(raw=raw, ours=ours, direct=direct,
            mse_change_pct_vs_raw=100 * (ours_mean['MSE'] / raw_mean['MSE'] - 1),
            mae_change_pct_vs_raw=100 * (ours_mean['MAE'] / raw_mean['MAE'] - 1),
            speedup_vs_raw=raw_mean['p50_ms'] / ours_mean['p50_ms'],
            direct_dominates_ours_mean=dominates(direct_mean, ours_mean),
            direct_dominates_ours_paired_seeds=paired)
    return result


def _figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': NAME})
    colors = {'RAW': '#7D8792', 'B2': '#0072B2', 'DIRECT_G': '#D55E00', 'LORA': '#8B68A6'}
    names = {'RAW': 'Raw', 'B2': 'Ours B2', 'DIRECT_G': 'Direct MLP', 'LORA': 'LoRA'}
    metrics = [('MSE', 'Normalized MSE ↓'), ('p50_ms', 'Whole B8 latency (ms) ↓'),
               ('peak_allocated_MiB', 'Peak allocated (MiB) ↓')]
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 8.9), constrained_layout=False)
    for i, dataset in enumerate(('jena', *DATASETS)):
        families = ('RAW', 'B2', 'LORA') if dataset == 'jena' else ('RAW', 'B2', 'DIRECT_G')
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            maximum = max(aggregate(rows, dataset, family)[metric]['maximum'] for family in families)
            for y, family in enumerate(families):
                value = aggregate(rows, dataset, family)[metric]
                x, lo, hi = value['mean'], value['minimum'], value['maximum']
                ax.barh(y, x, height=.58, color=colors[family], alpha=.95,
                        xerr=np.asarray([[x-lo], [hi-x]]), error_kw=dict(ecolor='#222222', capsize=3, lw=1))
                label_x = max(x, hi) + .02 * maximum
                ax.text(label_x, y, f'{x:.4f}' if metric == 'MSE' else f'{x:.2f}',
                        va='center', ha='left', fontsize=9)
            ax.set_xlim(0, maximum * 1.36 if maximum else 1)
            ax.set_yticks(range(3), [names[family] for family in families] if j == 0 else [''] * 3)
            ax.invert_yaxis()
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#E3E7EB', linewidth=.7)
            if j == 0:
                title = 'Jena* — historical context' if dataset == 'jena' else f'{LABELS[dataset]} — C={AXES[dataset][0]}, K={AXES[dataset][1]}'
                ax.text(0, 1.13, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')
    fig.suptitle('Matched execution constraints: B = 8, same row cap per F call', y=.98, fontsize=15, fontweight='bold')
    fig.legend(handles=[Patch(color=colors[key], label=names[key]) for key in ('RAW', 'B2', 'DIRECT_G', 'LORA')],
               loc='lower center', bbox_to_anchor=(.5, .055), ncol=4, frameon=False)
    fig.text(.06, .025, 'Bars: seed mean; whiskers: minimum–maximum of two seeds (not confidence intervals). Raw/LoRA: one run.\n'
             '* Jena is an earlier development experiment. Compare MSE within each dataset only. All axes start at zero.',
             fontsize=9, color='#414B55')
    fig.subplots_adjust(left=.12, right=.96, top=.82, bottom=.16, hspace=.68, wspace=.30)
    for j, (_, label) in enumerate(metrics):
        position = axes[0, j].get_position()
        fig.text((position.x0 + position.x1) / 2, .905, label, ha='center', va='center', fontsize=11)
    output = {}
    for suffix in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Date': None} if suffix == 'svg' else {'Software': NAME}
        fig.savefig(buffer, format=suffix, dpi=300, metadata=metadata)
        output['COMPARISON.' + suffix] = buffer.getvalue()
    plt.close(fig)
    return output


def _texts(rows, proof):
    effects = comparisons(rows)
    lines = ['# B2 데이터 확장 실험 결과', '',
        '[확인] 이 고정 조건에서 두 후보 모두 B2의 MSE가 Raw보다 나빴고, '
        'Direct MLP는 두 데이터 모두에서 B2보다 MSE·전체 지연·allocated 메모리가 우수했다.', '',
        '[확인] Household와 SML2010의 고정 비교를 완료했다. 4개 B2 search 전체를 봉인한 뒤 fresh A60 refit/V를 실행했고, '
        '동일 seed의 Direct MLP 4개와 Raw 2개를 포함해 총 10개 경로를 새 프로세스에서 측정했다. '
        '후보를 결과에 따라 교체하지 않았다.', '',
        '[전체 seed별 수치](COMPARISON.csv) · [비교 그림](COMPARISON.png) · [영문 슬라이드 문안](SLIDE_CONTENT_EN.md)', '',
        'MSE·MAE는 각 데이터의 고정 A60 정규화 단위에서 관측 equal-channel macro로 계산했다. '
        'L512/H48은 같은 스텝 수이고 실제 시간 범위는 다르다(H48: Jena 8시간, SML 12시간, Household 48시간). '
        '주파수를 강제로 리샘플하지 않았으며 동일 조건은 데이터별 방법 간 비교를 뜻한다. '
        '서로 다른 데이터의 MSE 크기를 직접 비교하지 않는다. 아래 평균·범위는 두 seed의 기술 통계이며 신뢰구간이 아니다.', '']
    slide = ['# Comparison under execution constraints', '',
        'B = 8. All frozen-F routes share the same per-call row cap: Household 16; SML2010 48. '
        'Raw uses four F calls; B2 uses one. Direct MLP uses no F.', '']
    for dataset in DATASETS:
        value = effects[dataset]
        lines.extend([f'## {LABELS[dataset]}', '', '```text',
            'method       seeds   MSE mean [min, max]          MAE mean    B8 median ms    allocated MiB    reserved MiB'])
        for key, name in (('raw', 'Raw'), ('ours', 'Ours B2'), ('direct', 'Direct MLP')):
            group = value[key]
            lines.append(f'{name:<12} {group["n"]:<7} {group["MSE"]["mean"]:.6f} [{group["MSE"]["minimum"]:.6f}, {group["MSE"]["maximum"]:.6f}]  '
                f'{group["MAE"]["mean"]:.6f}    {group["p50_ms"]["mean"]:.3f}           {group["peak_allocated_MiB"]["mean"]:.3f}          {group["peak_reserved_MiB"]["mean"]:.3f}')
        lines.extend(['```', '', f'B2의 Raw 대비 MSE 변화는 {value["mse_change_pct_vs_raw"]:+.2f}%, '
            f'MAE 변화는 {value["mae_change_pct_vs_raw"]:+.2f}%, 전체 B8 지연 기준 속도비는 {value["speedup_vs_raw"]:.2f}배다. '
            '낮은 오차와 빠른 실행을 별개 축으로 해석한다.', '',
            f'Direct MLP가 B2를 MSE·allocated 피크·지연 세 축에서 지배하는지: '
            f'평균={value["direct_dominates_ours_mean"]}, seed92601={value["direct_dominates_ours_paired_seeds"]["92601"]}, '
            f'seed92602={value["direct_dominates_ours_paired_seeds"]["92602"]}. '
            '세 값 모두 작거나 같고 하나 이상 엄격히 작을 때만 지배로 판정했다. MAE와 reserved는 별도 보고한다.', ''])
        for key, name in (('ours', 'B2'), ('direct', 'Direct MLP')):
            group = value[key]
            lines.extend([f'{name} 두 seed의 B8 median 지연 범위는 {group["p50_ms"]["minimum"]:.3f}–{group["p50_ms"]["maximum"]:.3f} ms, '
                f'allocated 피크 범위는 {group["peak_allocated_MiB"]["minimum"]:.3f}–{group["peak_allocated_MiB"]["maximum"]:.3f} MiB, '
                f'reserved 피크 범위는 {group["peak_reserved_MiB"]["minimum"]:.3f}–{group["peak_reserved_MiB"]["maximum"]:.3f} MiB다.', ''])
        slide.extend([f'## {LABELS[dataset]}', '',
            f'- B2 vs Raw: normalized MSE {value["mse_change_pct_vs_raw"]:+.2f}%; MAE {value["mae_change_pct_vs_raw"]:+.2f}%.',
            f'- Whole B8 speed ratio vs Raw: {value["speedup_vs_raw"]:.2f}×.',
            f'- Direct MLP dominates B2 in MSE, allocated peak and latency: mean {value["direct_dominates_ours_mean"]}; '
            f'seed 92601 {value["direct_dominates_ours_paired_seeds"]["92601"]}; seed 92602 {value["direct_dominates_ours_paired_seeds"]["92602"]}.', ''])
    lines.extend(['## 관측 범위와 학습 상태', '',
        'Household는 V 72개 origin을 모두 평가했다. SML2010은 V 9개 origin 중 한 창의 목표가 전부 결측이어서 '
        '실제 오차 기여 origin은 8개다. 그 창을 결과에 따라 제거하거나 보간하지 않았다. '
        'SML의 3개 motor 채널은 A_fit에서 상수이며 21개 채널의 원래 macro 집계에 그대로 포함했다. '
        'SML의 짧은 기간과 결측 블록은 결과의 적용 범위를 제한한다.', '', '```text',
        'run_id  selected_epoch  cap  search_train_origins  training_status'])
    for job, value in sorted(proof['training'].items()):
        lines.append(f'{job}  {value["selected_epoch"]}  3  {value["search_train_origins"]}  {value["training_status"]}')
    lines.extend(['```', '',
        'TRAIN origin 수는 실제 stride1 학습 receipt의 수이며 데이터 진단용 stride48 목록 길이가 아니다. '
        'cap3 끝이 계속 최선인 arm은 학습 상한에서 개선 중이며 수렴한 것으로 해석하지 않는다. '
        'epoch0 선택도 그대로 보고한다.', '', '## 공정한 해석과 비용의 범위', '',
        '두 후보 모두 과거 다른 방법의 개발에 사용된 데이터이고 V는 반복 개발 구간이다. '
        '새 독립 TEST나 사전학습 비중복을 입증하지 않는다. LoRA와 CPU Ridge는 이번 확장에서 실행하지 않아 '
        '그 대안보다 우수하다는 주장을 하지 않는다.', '',
        '그림의 Jena* 4행은 이전 memory_budget_20261018_v1의 B8·confirm·row48·K6 '
        'Raw/B2 두 seed/merged LoRA 문맥이다. 새 후보의 비용과 별도의 과거 실행이며, Jena의 '
        '기존 데이터·정규화 ancestry 및 과거 TEST 노출 이력도 유지된다.', '',
        '실측은 경로별 10 warmup+20 timed, 첫 8개 고정 V origin을 반복했다. 전체 V·불균등 마지막 batch의 '
        'row128/공통 cap parity와 F calls/streams를 따로 검증했다. 시간은 physical+mask부터 causal fill·A60 scaler·실제 모델·'
        'inverse·CPU 반환까지이며 window slicing·model load·검산·hashing은 제외했다. '
        'GPU 수치는 CUDA allocated/reserved 텐서 피크이며 전체 프로세스 메모리 상한이 아니다. '
        '20회 p95는 서비스 tail latency 보장이 아니다.', '',
        'CSV의 K는 F에 들어가는 stream 폭이다(Raw=C, B2=압축 K, Direct MLP=0). '
        '공통 실행 제한은 별도 row_cap 열이며 Raw 채널을 압축하지 않았다.', '',
        '새 경로 10개, warmup 100회, timed 200회 및 private prediction 파일 SHA·array 지문·FP64 오차·'
        'raw timing quantile·학습/선택/체크포인트 연결을 독립 검산했다. seed별 전체 수치와 각 원 receipt 경로는 CSV에 있다.', ''])
    difference = proof['workspace_auxiliary_minus_canonical_allocated_MiB']
    lines.extend(['초기 비용 10개는 새로 강제한 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 사용했으므로 '
        '[환경 정합 기술 반복 기록](MEASUREMENT_AMENDMENT_001.json)에 `AUXILIARY_WORKSPACECONFIG`로 보존했다. '
        '기존 Jena 코드가 부모 환경을 상속했다는 사실은 확인했지만 당시 실제 환경값은 기록이 없어 확정하지 않는다. '
        '본표는 강제 설정을 제거하고 현재 부모의 unset 값을 기록해 비용만 다시 측정한 10개다. '
        f'이번 두 반복의 allocated 피크 차이(auxiliary−본표)는 {difference["minimum"]:.3f}–{difference["maximum"]:.3f} MiB다. '
        '보조 측정을 실패나 별도 정확도 반복으로 처리하지 않았다. 합계는 비용 경로 측정 20개, warmup 200회, timed 400회이며 '
        '학습 재실행은 0회다. 학습 모델은 B2 4개와 Direct MLP 4개(각 search+fresh refit)로 그대로다.', ''])
    slide.extend(['## Figure and footnote', '',
        'Use COMPARISON.svg for vector export. Bars are seed means; whiskers are two-seed min–max ranges, not confidence intervals. '
        'Compare normalized MSE only within a dataset. Jena* is historical context from a separate earlier measurement.', '',
        'L512/H48 denote equal step counts, not equal durations: H48 is 8 h for Jena, 12 h for SML2010, and 48 h for Household. '
        'No common-frequency resampling was imposed.', '',
        'Development V, two seeds, cap3, fixed first B8 group, 10 warmups + 20 timings, allocated tensor memory. '
        'SML2010: 9 V origins, 8 with observed targets; three constant motor channels remain included. '
        'No new LoRA or Ridge baseline. Direct MLP competition must remain visible. No independent TEST claim.', ''])
    slide.extend(['An initial forced `CUBLAS_WORKSPACE_CONFIG=:4096:8` cost pass is preserved as '
        '`AUXILIARY_WORKSPACECONFIG`. The canonical pass inherits the current unset parent value; '
        'the historical Jena environment value was not recorded. Cost-only repeat: 20 route measurements, '
        '200 warmups and 400 timings in total; no retraining.', ''])
    return '\n'.join(lines), '\n'.join(slide)


def generate(repo=REPO, output_dir=None):
    repo = Path(repo).resolve()
    rows, proof = collect(repo)
    text, slide = _texts(rows, proof)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row['scope'], row['dataset'], row['family'], row['seed'])))
    outputs = dict({'COMPARISON.csv': buffer.getvalue().encode('utf-8'),
        'RESULTS_KO.md': text.encode('utf-8'), 'SLIDE_CONTENT_EN.md': slide.encode('utf-8')}, **_figure(rows))
    destination = Path(output_dir).resolve() if output_dir else repo / 'reports' / NAME
    if not any(destination.is_relative_to(root) for root in (repo / 'reports' / NAME, repo / '.cache' / NAME)):
        raise ValueError('Analysis output must remain inside the new report/cache namespace')
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        target = destination / name
        if target.exists() and target.read_bytes() != value:
            raise FileExistsError('Preserve the existing analysis output: ' + name)
    for name, value in outputs.items():
        target = destination / name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(value)
    return dict(proof, outputs={name: hashlib.sha256(value).hexdigest() for name, value in outputs.items()},
                comparisons=comparisons(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only:
        rows, proof = collect(args.repo)
        print(json.dumps(dict(status=proof['status'], rows=len(rows), new_routes=proof['new_routes'])))
    else:
        result = generate(args.repo, args.output_dir)
        print(json.dumps(dict(status=result['status'], outputs=result['outputs']), sort_keys=True))


if __name__ == '__main__':
    main()
