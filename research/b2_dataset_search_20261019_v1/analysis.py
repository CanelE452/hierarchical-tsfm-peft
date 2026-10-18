"""Independently validate a sealed development wave and render its full K grid."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS = (92601, 92602)
POLICIES = ('DEFAULT128', 'COMMON_CAP')
METRICS = ('MSE', 'MAE', 'p50_ms', 'peak_allocated_MiB', 'peak_reserved_MiB')
ARRAYS = {'pred_normalized', 'reference_pred_normalized', 'target_normalized',
          'observed', 'origins', 'channel_ids', 'mean', 'std'}
LABELS = {'ETTh2_safe17400': 'ETTh2 (safe 17400 cohort)', 'appliances': 'Appliances', 'ETTh1': 'ETTh1',
          'ETTm1':'ETTm1','solar_full137':'Solar (all 137 channels)'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + json.dumps(list(value.shape)).encode() + value.tobytes()).hexdigest()


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _close(actual, expected, label, atol=1e-10, rtol=1e-10):
    _require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
             'Independent recomputation differs: ' + label)


def score(prediction, target, mask):
    prediction, target, mask = map(np.asarray, (prediction, target, mask))
    _require(prediction.shape == target.shape == mask.shape and target.ndim == 3 and mask.dtype == np.bool_,
             'Aligned three-dimensional forecasts, targets and boolean mask required')
    counts = mask.sum((0, 1), dtype=np.int64)
    _require(np.all(counts > 0) and np.isfinite(prediction).all() and np.isfinite(target[mask]).all(),
             'Every channel needs finite forecasts and observed targets')
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    sse = (error * error).sum((0, 1), dtype=np.float64)
    sae = np.abs(error).sum((0, 1), dtype=np.float64)
    return dict(MSE=math.fsum((sse / counts).tolist()) / len(counts),
        MAE=math.fsum((sae / counts).tolist()) / len(counts), observed=int(counts.sum()),
        channel_sse=sse.tolist(), channel_sae=sae.tolist(), channel_counts=counts.tolist())


def _metrics(actual, recorded, label):
    for key in ('MSE', 'MAE'):
        _close(actual[key], recorded[key], label + ' ' + key)
    for key in ('observed', 'channel_counts'):
        if key in recorded:
            _require(actual[key] == recorded[key], label + ' observed coverage differs')
    for key in ('channel_sse', 'channel_sae'):
        if key in recorded:
            _require(np.allclose(actual[key], recorded[key], atol=1e-10, rtol=1e-10), label + ' channel errors differ')


def _work(family, k, channels, policy, batch=8):
    q = math.ceil(channels / 4)
    width = channels if family == 'RAW' else k if family == 'B2' else 0
    requested = (128 if policy == 'DEFAULT128' else 8 * q) if width else None
    streams = batch * width
    effective = min(requested, streams) if streams else None
    rows = [min(effective, streams - start) for start in range(0, streams, effective)] if streams else []
    return dict(q=q, width=width, streams=streams, requested_row_cap=requested,
                effective_row_cap=effective, calls=len(rows), row_sizes=rows)


def verify_cost(receipt, arrays, config):
    family, policy = receipt.get('family'), receipt.get('row_policy')
    channels = len(config['columns'])
    _require(receipt.get('schema') == 'b2_dataset_search_cost_v1' and receipt.get('status') == 'COMPLETE'
        and receipt.get('dataset') == config['dataset'] and family in ('RAW', 'B2', 'DIRECT_G')
        and policy in POLICIES and receipt.get('seed') in SEEDS
        and (receipt.get('C'), receipt.get('L'), receipt.get('H'), receipt.get('batch_size')) == (channels, 512, 48, 8),
        'Registered cost identity/schema differs')
    _require(set(arrays) == ARRAYS, 'Unregistered cost prediction archive schema')
    width = channels if family == 'RAW' else receipt['K'] if family == 'B2' else 0
    work = _work(family, receipt['K'], channels, policy)
    _require(receipt.get('expected_work') == work and receipt.get('model_K') == width
        and receipt.get('q') == work['q'] and receipt.get('actual_F_width') == width
        and receipt.get('requested_row_cap') == work['requested_row_cap']
        and receipt.get('effective_row_cap') == work['effective_row_cap'], 'F row policy/cap/work differs')
    states = {'F'} if family == 'RAW' else {'F', 'adapter'} if family == 'B2' else {'adapter'}
    _require(set(receipt.get('states_before', {})) == states and receipt['states_before'] == receipt.get('states_after'),
             'Frozen tensor state proof differs')
    lo, hi = config['split_ranges']['V_select']
    origins = np.arange(max(lo, 512), hi - 48 + 1, 48, dtype=np.int64)
    n = len(origins)
    _require(n >= 8 and np.array_equal(arrays['origins'], origins) and arrays['origins'].dtype.kind in 'iu'
        and arrays['channel_ids'].astype(str).tolist() == config['columns'], 'Full V origin or channel grid differs')
    _require(arrays['pred_normalized'].shape == arrays['reference_pred_normalized'].shape == arrays['target_normalized'].shape
        == arrays['observed'].shape == (n, 48, channels) and arrays['mean'].shape == arrays['std'].shape == (channels,)
        and arrays['pred_normalized'].dtype == np.float64 and arrays['reference_pred_normalized'].dtype in (np.dtype('float32'),np.dtype('float64'))
        and arrays['target_normalized'].dtype == arrays['mean'].dtype == arrays['std'].dtype == np.float32
        and arrays['origins'].dtype == np.int64 and arrays['channel_ids'].dtype.kind == 'U'
        and np.isfinite(arrays['mean']).all() and np.isfinite(arrays['std']).all() and (arrays['std'] > 0).all(),
        'Full V arrays or scaler shape differs')
    parity, original, summary = (receipt[key] for key in ('parity', 'original_reference', 'summary'))
    batches = [min(8, n - start) for start in range(0, n, 8)]
    original_calls = [math.ceil(batch * width / 128) if width else 0 for batch in batches]
    _require(parity.get('status') == original.get('status') == 'PASS'
        and parity.get('full_V_origins') == original.get('full_V_origins') == n
        and parity.get('last_batch_size') == batches[-1] and original.get('F_calls') == original_calls
        and original.get('row_cap') == (128 if width else None)
        and parity.get('atol') == 1e-5 and parity.get('rtol') == 1e-4, 'Full V/tail original-reference contract differs')
    for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
        _require(array_fingerprint(arrays[key]) == parity['input_fingerprints'].get(key), 'Input array fingerprint differs: ' + key)
    current, reference = arrays['pred_normalized'], arrays['reference_pred_normalized']
    _require(array_fingerprint(current) == parity['output_fingerprints'].get('regenerated_normalized')
        and array_fingerprint(reference) == parity['output_fingerprints'].get('original_normalized')
        == original.get('canonical_prediction_fingerprint')
        and np.allclose(current, reference, atol=1e-5, rtol=1e-4), 'Whole V prediction/chunking parity differs')
    actual, canonical = score(current, arrays['target_normalized'], arrays['observed']), score(reference, arrays['target_normalized'], arrays['observed'])
    _metrics(actual, parity['independent_metrics'], 'Cap forecast')
    _metrics(canonical, parity['original_metrics'], 'Canonical forecast')
    _metrics(canonical, original['metrics'], 'Original row128')
    if 'max_abs' in parity:
        _close(float(np.max(np.abs(current-reference))),parity['max_abs'],'Whole V maximum prediction deviation')
    for key,value in parity.get('metric_deltas',{}).items():
        _close(actual[key]-canonical[key],value,'Parity metric delta '+key)
    for key in ('MSE', 'MAE'):
        _close(canonical[key], summary[key], 'Summary ' + key)
        _close(actual[key], summary['recomputed_' + key], 'Summary cap ' + key)
        _close(actual[key], canonical[key], key + ' cap parity', atol=1e-5, rtol=1e-4)
    trials = receipt.get('trials', [])
    _require(len(trials) == 30 and summary.get('warmups') == 10 and summary.get('timed_trials') == 20
        and summary.get('status') == 'COMPLETE' and summary.get('parity_status') == 'PASS'
        and summary.get('F_calls') == work['calls'] and summary.get('F_logical_streams') == work['streams']
        and summary.get('foundation_instances_created') == (0 if family == 'DIRECT_G' else 1)
        and summary.get('core_dtype') == 'float32' and summary.get('cached_forecast_reads_in_timed_path') == 0,
        'B8/FP32/full trial measurement contract differs')
    for phase, count in (('warmup', 10), ('timed', 20)):
        subset = [trial for trial in trials if trial.get('phase') == phase]
        _require(len(subset) == count and [trial.get('index') for trial in subset] == list(range(count))
            and [trial.get('trial') for trial in subset] == list(range(count)), 'Exact raw trial coverage differs')
        for trial in subset:
            _require(trial.get('F_calls') == work['calls'] and trial.get('F_logical_streams') == work['streams']
                and trial.get('F_physical_streams') == work['streams']
                and math.isfinite(trial.get('whole_seconds', math.nan)) and trial['whole_seconds'] > 0,
                'Whole path raw trial/counters differ')
    timed = [trial for trial in trials if trial['phase'] == 'timed']
    q25, p50, q75, p95 = map(float, np.quantile([1000 * trial['whole_seconds'] for trial in timed], [.25, .5, .75, .95]))
    derived = dict(q25_ms=q25, p50_ms=p50, q75_ms=q75, p95_ms=p95)
    for key, value in dict(median_batch_ms=p50, IQR_batch_ms=q75-q25, p95_batch_ms=p95,
                          median_ms_per_origin=p50/8, origins_per_second=8000/p50).items():
        _close(value, summary[key], key)
    for key in ('peak_allocated_MiB', 'peak_reserved_MiB'):
        values = [float(trial[key]) for trial in timed]
        _require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid timed memory')
        derived[key] = max(values)
        _close(max(values), summary[key], key)
        for suffix, value in (('min_MiB', min(values)), ('max_MiB', max(values))):
            summary_key = 'timed_' + key.replace('MiB', suffix)
            if summary_key in summary:
                _close(value, summary[summary_key], summary_key)
    _require(derived['peak_reserved_MiB'] >= derived['peak_allocated_MiB'], 'Reserved is below allocated')
    return dict(**canonical, **derived, n_origins=n, scored_origins=int(np.any(arrays['observed'], axis=(1, 2)).sum()),
                observed_targets=canonical['observed'])


def aggregate(rows):
    groups = {}
    for row in rows:
        key = tuple(row[key] for key in ('dataset', 'row_policy', 'family', 'K'))
        groups.setdefault(key, []).append(row)
    output = []
    order = {'RAW': 0, 'B2': 1, 'DIRECT_G': 2, 'LORA': 3}
    for key, group in sorted(groups.items(), key=lambda item: (item[0][0], POLICIES.index(item[0][1]), order[item[0][2]], item[0][3])):
        value = dict(zip(('dataset', 'row_policy', 'family', 'K'), key), seeds=len(group),
                     seed_ids=','.join(str(row['seed']) for row in sorted(group, key=lambda row: row['seed'])))
        _require(len({row['seed'] for row in group}) == len(group), 'Duplicate seed in summary group')
        for metric in METRICS:
            numbers = [float(row[metric]) for row in group]
            value.update({metric+'_mean': math.fsum(numbers)/len(numbers), metric+'_min': min(numbers), metric+'_max': max(numbers)})
        for name in ('C', 'row_cap', 'F_calls', 'F_streams', 'n_origins', 'scored_origins', 'observed_targets'):
            if name in group[0]:
                _require(all(row[name] == group[0][name] for row in group), 'Seed cohort/cost width differs')
                value[name] = group[0][name]
        output.append(value)
    return output


def _dominates(left, right):
    keys = ('MSE', 'p50_ms', 'peak_allocated_MiB')
    return all(left[k] <= right[k] for k in keys) and any(left[k] < right[k] for k in keys)


def classify(rows):
    output = {}
    for dataset in sorted({row['dataset'] for row in rows}):
        subset = [row for row in rows if row['dataset'] == dataset and row['row_policy'] == 'DEFAULT128']
        raw = [row for row in subset if row['family'] == 'RAW']
        direct = [row for row in subset if row['family'] == 'DIRECT_G']
        _require(len(raw) == 1 and sorted(row['seed'] for row in direct) == list(SEEDS), 'Raw/Direct seed coverage differs')
        raw = raw[0]
        direct_mean = {key: math.fsum(row[key] for row in direct)/2 for key in METRICS}
        widths = {}
        for k in sorted({row['K'] for row in subset if row['family'] == 'B2'}):
            ours = [row for row in subset if row['family'] == 'B2' and row['K'] == k]
            _require(sorted(row['seed'] for row in ours) == list(SEEDS), 'B2 requires both registered seeds')
            mean = {key: math.fsum(row[key] for row in ours)/2 for key in METRICS}
            gain = 100 * (1 - mean['MSE']/raw['MSE'])
            predicates = dict(both_seed_MSE_below_raw=all(row['MSE'] < raw['MSE'] for row in ours),
                mean_MSE_gain_at_least_1pct=mean['MSE'] <= .99*raw['MSE'],
                both_seed_default_latency_below_raw=all(row['p50_ms'] < raw['p50_ms'] for row in ours),
                both_seed_default_allocated_below_raw=all(row['peak_allocated_MiB'] < raw['peak_allocated_MiB'] for row in ours))
            widths[str(k)] = dict(positive=all(predicates.values()), predicates=predicates, MSE_gain_pct_vs_raw=gain,
                mae_change_pct_vs_raw=100*(mean['MAE']/raw['MAE']-1), speedup_vs_raw=raw['p50_ms']/mean['p50_ms'],
                allocated_change_pct_vs_raw=100*(mean['peak_allocated_MiB']/raw['peak_allocated_MiB']-1),
                direct_dominates_mean=_dominates(direct_mean, mean),
                direct_dominates_paired_seeds={str(row['seed']): _dominates(next(d for d in direct if d['seed'] == row['seed']), row) for row in ours})
        output[dataset] = dict(positive=any(value['positive'] for value in widths.values()), widths=widths)
    return output


def _bound(value, repo, cache, public):
    scheme, relative = value['reference'].split('://', 1)
    root = {'repo': repo, 'cache': cache, 'report': public}.get(scheme)
    _require(root is not None, 'Unsupported artifact binding')
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()) and NAME in path.parts and sha(path) == value['sha256'],
             'Artifact binding escaped or bytes changed: ' + value['reference'])
    return path


def _sources(files, repo):
    _require(isinstance(files, dict) and bool(files), 'Producer source graph required')
    for relative, expected in files.items():
        base = repo if '/' in relative or '\\' in relative else repo / 'research/informs_broad_validation_20261015_v1'
        path = (base / relative).resolve()
        _require(path.is_relative_to(repo) and path.suffix == '.py' and sha(path) == expected,
                 'Producer source changed: ' + relative)


def _statistics(task, end):
    x, mask = task['physical'][:end].astype(np.float64), task['observed'][:end]
    counts = mask.sum(0)
    _require(np.all(counts > 0), 'Unobserved training channel')
    mean = np.where(mask, x, 0).sum(0)/counts
    std = np.sqrt((np.where(mask, x-mean, 0)**2).sum(0)/counts)
    std[std == 0] = 1
    return mean.astype(np.float32), std.astype(np.float32)


def _cohort(task, arrays, receipt):
    end = task['split_ranges']['A_stop'][1]
    mean, std = _statistics(task, end)
    _require(np.array_equal(mean, arrays['mean']) and np.array_equal(std, arrays['std']), 'A60 scaler does not match physical prefix')
    rows = arrays['origins'][:, None]+np.arange(48)[None]
    mask = task['observed'][rows]
    target = ((np.where(mask, task['physical'][rows], mean)-mean)/std).astype(np.float32)
    _require(np.array_equal(mask, arrays['observed']) and np.allclose(target, arrays['target_normalized'], atol=1e-5, rtol=1e-4),
             'Scored targets/mask differ from registered physical V')
    history = arrays['origins'][:, None]+np.arange(-512, 0)[None]
    for key, value in dict(physical_windows=task['physical'][history], physical_observed_windows=task['observed'][history]).items():
        _require(array_fingerprint(value) == receipt['parity']['input_fingerprints'][key], 'Physical history fingerprint differs')


def _selected(curves):
    values = [(float(row['epoch']), float(row['astop_mse'])) for row in curves if row.get('astop_mse') not in (None, '')]
    _require([epoch for epoch, _ in values] == [0., .25, 1., 3.] and all(math.isfinite(value) for _, value in values),
             'Complete fixed A_stop search schedule required')
    return min(values, key=lambda item: (item[1], item[0])), values[-1][1] < min(value for _, value in values[:-1])


def _fit_contract(source, receipt, task, family, k, seed, job):
    spec = source['spec']
    expected = dict(run_id=job, family=family, dataset=task['dataset'], seed=seed, device='cuda',
        epochs=[.25,1.,3.],lr=.001,effective_batch=16,microbatch=4,F_row_microbatch=128,checkpoint_inputs=False)
    if family == 'B2':
        expected.update(latent_k=k,revin=False,init_kind='PCA')
    _require(all(spec.get(key) == value for key,value in expected.items()), 'Registered fixed training recipe differs')
    axes = dict(L=512,H=48,C=len(task['columns']),columns=task['columns'],fold='primary',source_array_fingerprints=task['array_sha256'])
    _require(source['config_sha'] == fingerprint(dict(spec=spec,axes=axes)) and source['data_sha'] == task['data_sha256'],
             'Training task/spec fingerprint differs')
    fit_end, refit_end = task['split_ranges']['A_fit'][1], task['split_ranges']['A_stop'][1]
    train_origins = fit_end-512-48+1
    stop_lo, stop_hi = task['split_ranges']['A_stop']
    stop_origins = len(range(max(stop_lo,512),stop_hi-48+1,48))
    _require(receipt.get('status') == 'COMPLETE_VALID' and receipt.get('optimizer') == 'AdamW'
        and receipt.get('source_sha') == source['source_sha']
        and receipt.get('weight_decay') == 0 and receipt['search_fit_end'] == fit_end
        and receipt['refit_fit_end'] == receipt['normalization_fit_end'] == refit_end
        and receipt['search_train_origins'] == train_origins and receipt['search_stop_origins'] == stop_origins
        and receipt['search_exposures'] == math.ceil(3*train_origins)
        and receipt['refit_exposures'] == math.ceil(receipt['selected_epoch']*(refit_end-512-48+1)),
        'Full stride1 search/fresh-refit exposure or scaler ranges differ')


def _search_checkpoint(path, search, task):
    import torch
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    mean,std = _statistics(task,task['split_ranges']['A_fit'][1])
    expected_scaler = fingerprint(dict(mean=array_fingerprint(mean),std=array_fingerprint(std)))
    extra = checkpoint['extra']
    _require(all(checkpoint['contract'].get(key) == value for key,value in search['contract'].items())
        and checkpoint['contract'].get('scaler_sha') == expected_scaler
        and extra['curves'] == search['curves'] and extra['best_epoch'] == search['selected_epoch']
        and extra['best_mse'] == search['summary']['selected_astop_mse']
        and extra['completed_epochs'] == [0.,.25,1.,3.], 'Sealed A_stop checkpoint/scaler/selection differs')


def _bundle(path, source, receipt, task, epoch):
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    contract = bundle['contract']
    _require(bundle['spec'] == source['spec'] and bundle['selected_search_epoch'] == epoch
        and all(contract.get(key) == source.get(key) for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha')),
        'Selected bundle scientific identity differs')
    state = {name: value.detach().cpu().contiguous().numpy() for name, value in bundle['model'].items()}
    state_sha = fingerprint({name: _array_digest(value) for name, value in state.items()})
    _require(state_sha == bundle['model_state_sha'] == receipt['model_state_sha'], 'Selected model tensor SHA differs')
    norm = bundle['normalization']
    mean, std = _statistics(task, task['split_ranges']['A_stop'][1])
    _require(np.array_equal(norm['mean'], mean) and np.array_equal(norm['std'], std)
        and norm['fit_end'] == receipt['normalization_fit_end'] == task['split_ranges']['A_stop'][1],
        'Fresh A60 bundle normalization differs')
    scaler = fingerprint(dict(mean=array_fingerprint(mean), std=array_fingerprint(std)))
    _require(contract['scaler_sha'] == receipt['refit_scaler_sha'] == scaler, 'Bundle scaler SHA differs')
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.tobytes())
    return bundle, digest.hexdigest()


def validate_training(repo, wave, contexts):
    report = repo/'reports'/NAME
    seal_path = report/'waves'/f'WAVE_{wave:03d}_SEARCHES_SEALED.json'
    seal = read(seal_path)
    plan_path = _bound(seal['wave_plan_binding'], repo, repo/'.cache'/NAME, report)
    plan = read(plan_path)
    expected = [f'B2_{ds}_K{k}_OFF_s{seed}' for ds, item in contexts.items() for k in item['entry']['K_grid'] for seed in SEEDS]
    _require(seal.get('status') == 'SEALED' and seal.get('all_searches_complete') is True
        and seal.get('V_used') is False and seal.get('selection_split') == 'A_stop' and seal.get('cap') == 3
        and seal.get('matched_arms') == expected and set(seal['selections']) == set(expected)
        and set(seal['datasets']) == set(contexts) and plan.get('status') == 'SEALED'
        and plan.get('phase') == 'BEFORE_SEARCH' and plan.get('V_used') is False and plan.get('cap') == 3
        and plan.get('epochs') == [.25, 1., 3.] and plan.get('matched_arms') == expected
        and plan.get('dataset_order') == list(contexts), 'Whole-wave search/selection seal differs')
    _bound(plan['protocol_binding'], repo, repo/'.cache'/NAME, report)
    output, f_states = {}, set()
    for dataset, item in contexts.items():
        cache, public, task = item['cache'], item['public'], item['task']
        identity_path = _bound(seal['datasets'][dataset]['identity_binding'], repo, cache, public)
        identity, summary = read(identity_path), read(public/'TRAIN_SUMMARY.json')
        sources = identity['sources']; _sources(sources, repo)
        registration = plan['datasets'][dataset]
        _require(all(registration.get(key) == identity.get(key) for key in ('dataset', 'jobs', 'K_grid', 'data_sha256', 'input_config_sha256', 'prefix_sha256', 'sources'))
            and identity['data_sha256'] == task['data_sha256'] and identity['input_config_sha256'] == item['entry']['config_sha256']
            and identity['prefix_sha256'] == item['entry']['prefix_sha256']
            and identity.get('TRAIN_stride') == 1 and identity.get('evaluation_stride') == 48
            and identity.get('revin') is False and identity.get('epochs') == [.25, 1., 3.]
            and identity.get('T_access') is False and identity.get('V_independent_test') is False
            and summary.get('status') == 'COMPLETE' and summary.get('new_searches') == len(item['entry']['K_grid'])*2
            and summary.get('fresh_refits') == len(item['entry']['K_grid'])*2
            and summary['rows'] == read(public/'VERIFIED_ROWS.json') == read(cache/'VERIFIED_ROWS.json'), 'Fixed training identity/refit coverage differs')
        _require(_bound(summary['selection_binding'], repo, cache, public) == seal_path
            and _bound(identity['wave_plan_binding'], repo, cache, public) == plan_path, 'Training points to a different wave seal')
        item['sources'] = sources
        rows = {row['run_id']: row for row in summary['rows']}
        _require(set(rows) == set(seal['datasets'][dataset]['matched_arms'])
            == {job for job in expected if job.startswith('B2_'+dataset+'_K')}, 'Verified K/seed coverage differs')
        for job, row in rows.items():
            run = public/'runs'/job/'cap3'
            search, result, fit, source = (read(run/name) for name in ('SEARCH_RESULT.json', 'REFIT_RESULT.json', 'BASE_FIT_RECEIPT.json', 'SOURCE_CONTRACT.json'))
            (epoch, value), last_best = _selected(search['curves'])
            selection = seal['selections'][job]
            checkpoint = _bound(selection['checkpoint'], repo, cache, public)
            _require(_bound(search['checkpoint_binding'], repo, cache, public) == checkpoint, 'Seal points to a different search checkpoint')
            _sources(source['files'], repo)
            _fit_contract(source,fit,task,'B2',row['K'],row['seed'],job)
            _search_checkpoint(checkpoint,search,task)
            _require(fingerprint(source['files']) == source['source_sha'] and source['data_sha'] == task['data_sha256']
                and search['contract'] == {key: source[key] for key in ('source_sha','config_sha','data_sha','model_sha')}
                and search['status'] == result['status'] == row['status'] == 'COMPLETE'
                and row['selected_epoch'] == epoch == selection['epoch'] == fit['selected_epoch']
                and selection['astop_mse'] == value and selection['K'] == row['K'] and selection['seed'] == row['seed']
                and selection['model_sha'] == source['model_sha'] == row['model_sha']
                and row['all_searches_sealed_before_V'] is True and row['V_independent_test'] is False
                and row['fit'] == 'fresh_A60' and row['evaluation'] == 'V_select'
                and row['revin'] is False and row['cap'] == 3, 'Search/refit/strict earlier-tie A_stop decision differs')
            _require(_bound(row['selection_decision_binding'], repo, cache, public) == seal_path
                and _bound(result['selection_decision_binding'], repo, cache, public) == seal_path,
                'Fresh refit selection binding differs')
            artifact = _bound(row['bundle_binding'], repo, cache, public)
            _require(_bound(result['bundle_binding'], repo, cache, public) == artifact
                and sha(artifact) == fit['artifact_sha256'] and fit['fresh_A60'] is True,
                'Selected checkpoint/refit artifact differs')
            prediction = _bound(row['prediction_binding'], repo, cache, public)
            bundle, adapter_sha = _bundle(artifact, source, fit, task, epoch)
            _require(bundle['search_summary'] == search['summary'] and result['fit_receipt'] == fit
                and row['last_checkpoint_strictly_best'] == last_best, 'Refit search summary/status differs')
            with np.load(prediction, allow_pickle=False) as archive:
                saved = {name: archive[name].copy() for name in archive.files}
            _metrics(score(saved['pred_normalized'], saved['target_normalized'], saved['observed']), row, 'Saved B2 accuracy')
            status = 'CAP3_STILL_IMPROVING' if last_best else 'SELECTED_EPOCH_ZERO' if epoch == 0 else 'CAP3_CONVERGENCE_UNRESOLVED'
            _require(row['training_status'] == status, 'Training status differs from A_stop curve')
            candidate = dict(row, base_training_device=bundle['spec'].get('device','UNKNOWN'),
                serialized_adapter_bytes=artifact.stat().st_size, serialized_base_checkpoint_bytes=artifact.stat().st_size)
            output[job] = dict(selected_epoch=epoch, cap=3, training_status=status,
                search_train_origins=fit['search_train_origins'], checkpoint_sha256=sha(artifact),
                search_checkpoint_sha256=sha(checkpoint), adapter_sha256=adapter_sha, candidate=candidate,
                MSE=row['MSE'], MAE=row['MAE'], saved_prediction_fingerprint=array_fingerprint(saved['pred_normalized']))
            f_states.add(source['model_sha'])
        for seed in SEEDS:
            job = f'DirectMLP_{dataset}_s{seed}'
            fit = read(public/(job+'_FIT.json')); run = public/'runs'/job
            source, original_fit = read(run/'SOURCE_CONTRACT.json'), read(run/'BASE_FIT_RECEIPT.json')
            _sources(source['files'], repo)
            _fit_contract(source,fit,task,'DIRECT_G',0,seed,job)
            with (run/'search'/'CURVE.csv').open(encoding='utf-8-sig', newline='') as stream:
                (epoch, value), last_best = _selected(list(csv.DictReader(stream)))
            _require(fit.get('status') == 'COMPLETE_VALID' and fit.get('family') == 'DIRECT_G' and fit.get('seed') == seed
                and fit.get('epochs') == [.25, 1., 3.] and fit.get('F_instances') == 0
                and fit.get('T_access') is False and fit.get('V_independent_test') is False
                and fit['selected_epoch'] == epoch and fit['selected_astop_mse'] == value
                and source['source_sha'] == fingerprint(source['files']) and source['model_sha'] == 'NO_FOUNDATION'
                and source['data_sha'] == task['data_sha256']
                and all(fit[key] == val for key, val in original_fit.items() if key != 'reused_complete_artifact'),
                'Fixed Direct MLP source/selection/refit differs')
            artifact = _bound(fit['bundle_binding'], repo, cache, public)
            _require(sha(artifact) == fit['artifact_sha256'], 'Direct MLP artifact SHA differs')
            _, adapter_sha = _bundle(artifact, source, fit, task, epoch)
            output[job] = dict(selected_epoch=epoch, cap=3, training_status='CAP3_STILL_IMPROVING' if last_best else
                'SELECTED_EPOCH_ZERO' if epoch == 0 else 'CAP3_CONVERGENCE_UNRESOLVED',
                search_train_origins=fit['search_train_origins'], checkpoint_sha256=sha(artifact), adapter_sha256=adapter_sha,
                candidate=dict(fit,K=0))
    _require(len(f_states) == 1, 'Wave does not share one pinned frozen F state')
    return output, next(iter(f_states)), sha(seal_path), sha(plan_path)


def _history(repo):
    path = repo/'reports/memory_budget_20261018_v1/MEMORY_COST_SUMMARY.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        source = [row for row in csv.DictReader(stream) if row['dataset'] == 'jena' and row['B'] == '8'
            and row['stage'] == 'confirm' and (row['family'] in ('RAW','LORA') or row['family'] == 'B2' and int(row['K']) in (3,6,12))]
    output = []
    for policy in POLICIES:
        for row in source:
            width = int(row['C']) if row['family'] in ('RAW','LORA') else int(row['K'])
            cap = min(128 if policy == 'DEFAULT128' else 48, 8*width)
            if int(row['effective_row_microbatch']) != cap:
                continue
            _require(row['status'] == 'COMPLETE' and row['parity_status'] == 'PASS'
                and row['warmups'] == '10' and row['timed_trials'] == '20', 'Historical Jena context is incomplete')
            output.append(dict(scope='HISTORICAL_SEPARATE_EXECUTION',dataset='jena',row_policy=policy,family=row['family'],
                run_id=row['accuracy_run_id'],seed=int(row['seed']),K=width,C=21,row_cap=cap,F_calls=int(row['F_calls']),
                MSE=float(row['MSE']),MAE=float(row['MAE']),p50_ms=float(row['median_batch_ms']),
                peak_allocated_MiB=float(row['peak_allocated_MiB']),peak_reserved_MiB=float(row['peak_reserved_MiB']),
                source_reference=path.relative_to(repo).as_posix(),facet_id=row['facet_id']))
    _require(len(output) == 16 and len({(r['row_policy'],r['run_id']) for r in output}) == 16,
             'Eight historical Jena models in both policies required')
    return output, sha(path)


def candidate_registry(repo, wave, judgments):
    report = repo/'reports'/NAME
    output = []
    amendment = read(report/'PREPARATION_AMENDMENT_001.json') if (report/'PREPARATION_AMENDMENT_001.json').exists() else None
    for number in range(1, wave+1):
        path = report/'waves'/f'WAVE_{number:03d}.json'
        if not path.exists():
            continue
        registered = read(path)
        for entry in registered['datasets']:
            dataset = entry['dataset']
            superseded = number == 1 and amendment is not None
            status = 'SUPERSEDED_BEFORE_TRAINING' if superseded else 'POSITIVE_EXPLORATORY' if judgments.get(dataset,{}).get('positive') else 'NEGATIVE_EXPLORATORY' if dataset in judgments else 'PRESERVED_PRIOR_WAVE'
            output.append(dict(wave=number,dataset=dataset,status=status,K_grid=','.join(map(str,entry['K_grid'])),
                config_sha256=entry['config_sha256'],prefix_sha256=entry['prefix_sha256'],
                detail='Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained' if superseded else
                'All K and both seeds disclosed; development V, not independent TEST'))
        for entry in registered.get('excluded',[]):
            output.append(dict(wave=number,dataset=entry['dataset'],status=entry['status'],K_grid='',
                               detail=json.dumps(entry.get('issues',[]),sort_keys=True,ensure_ascii=False)))
    positive=any(value.get('positive') for value in judgments.values())
    for path in sorted(report.glob('CANDIDATE_PLAN_*.json')):
        plan=read(path)
        planned=plan.get('planned_execution_wave')
        if not isinstance(planned,int) or planned<=wave or plan.get('parent_wave',wave+1)>wave:
            continue
        _require(plan.get('numeric_forecast_scores_used_to_choose_candidates') is False
            and plan.get('unchanged_recipe') is True and plan.get('unchanged_K_rule') is True,
            'Conditional candidates must be registered without forecast-dependent recipe changes')
        for entry in plan['ordered_candidates']:
            dataset=entry['dataset'];metadata_path=repo/'.cache'/NAME/'data'/dataset/'DATA_DIAGNOSTIC.json'
            _require(metadata_path.resolve().is_relative_to((repo/'.cache'/NAME/'data').resolve()),'Conditional dataset path escaped namespace')
            metadata=read(metadata_path) if metadata_path.exists() else None
            preparation=metadata.get('status') if metadata else 'REGISTERED_PREPARATION_PENDING'
            if metadata:
                _require(all(metadata.get(key)==entry[key] for key in ('dataset','C','columns','prefix_rows','interval_seconds')),
                         'Conditional preparation JSON differs from registered candidate metadata')
            ready=preparation=='READY'
            status=('READY_NOT_EXECUTED' if positive else 'READY_CONDITIONAL_NEXT_WAVE') if ready else preparation
            output.append(dict(wave=planned,dataset=dataset,status=status,K_grid=','.join(map(str,entry['K_grid'])),
                conditional=True,measured_in_this_report=False,forecast_results_counted=0,
                preparation_status=preparation,config_sha256=sha(metadata_path) if metadata else '',
                prefix_sha256=metadata.get('prefix_sha256','') if metadata else '',
                source_reference=path.relative_to(repo).as_posix(),plan_sha256=sha(path),
                detail='Conditional prefix preparation only; zero training/forecast experiments counted in this report. '
                    +'GPU activation is allowed only after the complete parent wave has no verified positive. '
                    +'Private numeric prefix was not opened or hashed by analysis.'))
    return output


def _verify_process_coverage(processes, facets):
    _require(len(processes) == facets, 'Every measured facet requires a distinct process identity (PID, process_create_time)')


def collect(repo=REPO, wave=2):
    from research.b2_dataset_search_20261019_v1 import train
    repo = Path(repo).resolve(); report = repo/'reports'/NAME; cache = repo/'.cache'/NAME
    wave_path = report/'waves'/f'WAVE_{wave:03d}.json'; registered = read(wave_path)
    protocol_sha = sha(report/'PROTOCOL.md')
    _require(registered.get('wave') == wave and registered.get('status') == 'REGISTERED_BEFORE_TRAINING'
        and registered.get('protocol_sha256') == protocol_sha and registered.get('V_forecast_scores_used') is False,
        'Registered pre-score wave/protocol binding differs')
    contexts = {}
    for entry in registered['datasets']:
        dataset = entry['dataset']; config_path = cache/'data'/dataset/'DATA_DIAGNOSTIC.json'
        prefix_path = config_path.with_name('PREFIX_PHYSICAL.npz')
        _require(sha(config_path) == entry['config_sha256'], 'Registered dataset config changed')
        config, task = train.load_task(config_path, prefix_path)
        _require(config.get('status') == 'READY' and config.get('candidate_training_admitted') is True
            and config['dataset'] == dataset and config['prefix_sha256'] == entry['prefix_sha256']
            and list(train.k_grid(len(task['columns']))) == entry['K_grid'],
                 'Registered prefix/K grid changed')
        contexts[dataset] = dict(entry=entry,config=config,task=task,config_path=config_path,prefix_path=prefix_path,
            cache=cache/'datasets'/dataset,public=report/'datasets'/dataset)
    training, f_sha, seal_sha, plan_sha = validate_training(repo, wave, contexts)
    rows, hashes, processes, environments, cohort_proofs, per_model_reference = [], {}, set(), set(), {}, {}
    for dataset, item in contexts.items():
        channels = len(item['task']['columns']); q = math.ceil(channels/4)
        arms = [('RAW',q,SEEDS[0],'Raw_'+dataset)]
        arms += [('B2',k,seed,f'B2_{dataset}_K{k}_OFF_s{seed}') for k in item['entry']['K_grid'] for seed in SEEDS]
        arms += [('DIRECT_G',q,seed,f'DirectMLP_{dataset}_s{seed}') for seed in SEEDS]
        expected_files = {job+'_'+policy+'_B8.json' for _,_,_,job in arms for policy in POLICIES}
        _require({p.name for p in (item['public']/'costs').glob('*.json')} == expected_files, 'Exact registered cost facet coverage required')
        for family, k, seed, job in arms:
            for policy in POLICIES:
                path = item['public']/'costs'/(job+'_'+policy+'_B8.json'); receipt = read(path)
                _require(all(receipt.get(key) == val for key,val in dict(dataset=dataset,family=family,K=k,seed=seed,
                    run_id=job,row_policy=policy,config_sha256=item['entry']['config_sha256'],prefix_sha256=item['entry']['prefix_sha256'],
                    data_sha256=item['task']['data_sha256'],protocol_sha256=protocol_sha,expected_f_sha256=f_sha,expected_f_params=47_718_016).items()),
                    'Cost scientific/model/data/request identity differs')
                _sources(receipt['producer_sources'], repo)
                _require(read(item['public']/'cost_validation'/path.name) == receipt['parity'], 'Separate parity receipt changed')
                prediction = _bound(dict(reference=receipt['prediction_reference'],sha256=receipt['prediction_file_sha256']),repo,item['cache'],item['public'])
                with np.load(prediction, allow_pickle=False) as archive:
                    arrays = {name:archive[name].copy() for name in archive.files}
                checked = verify_cost(receipt, arrays, item['config']); _cohort(item['task'], arrays, receipt)
                source_binding = dict(config_sha256=item['entry']['config_sha256'],prefix_sha256=item['entry']['prefix_sha256'],
                    data_sha256=item['task']['data_sha256'],sources=item['sources'])
                _require(receipt['source_bindings'] == source_binding, 'Cost source graph differs from training identity')
                request_path = cache/'cost_requests'/(job+'_'+policy+'.json')
                request = read(request_path); request.update(batch_size=8,device='cuda',factory='train:cost_inputs',protocol_sha256=protocol_sha)
                _require(hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest() == receipt['private_request_sha256'], 'Private request SHA differs')
                fit = training.get(job)
                candidate = fit['candidate'] if fit else dict(run_id=job,family='RAW',dataset=dataset,seed=seed,C=channels,K=channels,L=512,H=48)
                _require(hashlib.sha256(json.dumps(candidate,sort_keys=True,allow_nan=False).encode()).hexdigest() == receipt['candidate_identity_sha256'],
                         'Exact trained candidate identity differs')
                summary = receipt['summary']
                _require(summary['loaded_F_state_sha256'] == (f_sha if family != 'DIRECT_G' else 'NO_FOUNDATION_USED')
                    and summary.get('device') == 'cuda' and summary.get('metric_accumulation_dtype') == 'float64'
                    and summary.get('physical_output_dtype') == 'float64'
                    and summary.get('effective_row_microbatch') == receipt['effective_row_cap']
                    and summary.get('requested_row_microbatch') == receipt['requested_row_cap']
                    and receipt['environment'].get('autocast') is False and receipt['environment'].get('quantization') is False
                    and receipt['environment'].get('compile') is False and 'cublas_workspace_config' in receipt,
                    'Loaded F/FP32/eager cost environment differs')
                _require(summary.get('process_scope') == 'fresh OS process, one facet, one serialized model load'
                    and summary.get('process_pid') == summary.get('pid')
                    and all(trial.get('pid') == summary['pid'] for trial in receipt['trials'])
                    and math.isfinite(float(summary.get('process_create_time',math.nan))),
                    'Fresh worker process/trial identity differs')
                if fit:
                    _require(summary['checkpoint_sha256'] == fit['checkpoint_sha256'] and receipt['states_before']['adapter'] == fit['adapter_sha256'],
                             'Cost loaded adapter differs from selected fresh refit')
                if family == 'B2':
                    for key in ('MSE','MAE'):
                        _close(checked[key], fit[key], 'Cost/saved B2 '+key)
                    _require(array_fingerprint(arrays['reference_pred_normalized']) == fit['saved_prediction_fingerprint'], 'Cost changed original B2 predictions')
                cohort = {key:array_fingerprint(arrays[key]) for key in ('origins','channel_ids','mean','std','target_normalized','observed')}
                _require(dataset not in cohort_proofs or cohort_proofs[dataset] == cohort, 'Methods/policies scored different V cohorts')
                cohort_proofs[dataset] = cohort
                canonical = arrays['reference_pred_normalized']
                if job in per_model_reference:
                    _require(np.allclose(canonical,per_model_reference[job],atol=1e-5,rtol=1e-4), 'Policies changed original row128 reference')
                else:
                    per_model_reference[job] = canonical
                processes.add((summary.get('pid'),summary.get('process_create_time')))
                environments.add(json.dumps(receipt['cublas_workspace_config']))
                rows.append(dict(scope='NEW_DEVELOPMENT_WAVE',wave=wave,dataset=dataset,row_policy=policy,family=family,
                    run_id=job,seed=seed,C=channels,K=receipt['model_K'],L=512,H=48,B=8,
                    interval_seconds=item['config']['interval_seconds'],row_cap=receipt['effective_row_cap'],
                    requested_row_cap=receipt['requested_row_cap'],F_calls=summary['F_calls'],F_streams=summary['F_logical_streams'],
                    **{key:checked[key] for key in (*METRICS,'q25_ms','q75_ms','p95_ms','n_origins','scored_origins','observed_targets')},
                    selected_epoch=fit['selected_epoch'] if fit else '',cap=3 if fit else 0,
                    training_status=fit['training_status'] if fit else 'ZERO_SHOT',
                    source_reference=path.relative_to(repo).as_posix(),source_sha256=sha(path),prediction_sha256=sha(prediction),
                    checkpoint_sha256=summary['checkpoint_sha256'],F_state_sha256=summary['loaded_F_state_sha256']))
                hashes[path.relative_to(repo).as_posix()] = sha(path)
    _verify_process_coverage(processes,len(rows))
    _require(len(environments) == 1, 'Cost facets inherited different workspace settings')
    judgments = classify(rows)
    historical, historical_sha = _history(repo)
    candidates = candidate_registry(repo,wave,judgments)
    amendment_path = report/'PREPARATION_AMENDMENT_001.json'
    amendment = read(amendment_path)
    _require(amendment.get('status') == 'REGISTERED_BEFORE_ANY_NEW_GPU_TRAINING_OR_FORECAST_SCORES'
        and amendment['initial_preparation_overlap']['hourly_records'] == 16
        and amendment['initial_preparation_overlap']['consumed_by_new_training_or_forecast_scoring'] is False
        and amendment['replacement']['dataset'] == 'ETTh2_safe17400' and amendment['replacement']['cohort_rows'] == 17400
        and amendment['replacement']['prefix_rows'] == 13920
        and (wave != 2 or registered.get('amendment_sha256') == sha(amendment_path)),
        'Preserved shared-source preparation amendment differs')
    preservation=amendment['preservation']
    for key in ('old_prefix','old_config','old_registry','old_data_audit'):
        reference=preservation[key]
        old=(cache/reference.removeprefix('cache://')) if reference.startswith('cache://') else report/reference
        _require(old.resolve().is_relative_to(cache if reference.startswith('cache://') else report) and old.is_file(),
                 'Superseded preparation/registry was not preserved')
    _require(preservation.get('overwrite_or_delete') is False,'Preparation preservation contract differs')
    return rows, dict(status='PASS',wave=wave,route_facets=len(rows),warmup_trials=10*len(rows),timed_trials=20*len(rows),
        B2_searches=sum(len(item['entry']['K_grid'])*2 for item in contexts.values()),
        B2_fresh_refits=sum(len(item['entry']['K_grid'])*2 for item in contexts.values()),Direct_MLP_search_refits=2*len(contexts),
        protocol_sha256=protocol_sha,wave_sha256=sha(wave_path),all_searches_sealed_sha256=seal_sha,before_search_plan_sha256=plan_sha,
        cost_receipt_hashes=hashes,cohort_fingerprints=cohort_proofs,pinned_F_sha256=f_sha,
        frozen_F_evidence_scope='Pinned initial F hash and exact producer pre/post tensor-state receipts; analysis does not reload F',
        unique_pid_create_time_pairs=len(processes),unique_pid_create_time_pairs_required=True,unique_PIDs_required=False,
        inherited_cublas_workspace_config=json.loads(next(iter(environments))),
        training={job:{k:v for k,v in value.items() if k not in ('candidate',)} for job,value in training.items()},
        datasets={ds:{key:item['config'].get(key) for key in ('C','columns','interval_seconds','original_rows','cohort_rows','prefix_rows','train_structure')}
                  for ds,item in contexts.items()},
        judgments=judgments,candidates=candidates,
        conditional_candidates=[item for item in candidates if item.get('conditional')],
        conditional_candidate_prefixes_opened_or_hashed=False,
        historical=historical,historical_source_sha256=historical_sha,
        preparation_amendment_sha256=sha(amendment_path),preparation_overlap_materialized_hours=16,
        preparation_overlap_used_for_training_or_forecast=False,independent_TEST_claimed=False,
        superseded_numeric_prefix_opened_or_hashed_by_analysis=False,
        physical_output_hash_recomputed=False,physical_output_hash_scope='Normalized arrays and physical history are independently bound; physical output bytes were not saved')


def render_figures(rows, dataset, metadata, synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subset = [row for row in aggregate(rows) if row['dataset'] == dataset]
    style = dict(font_family='DejaVu Sans')
    with plt.rc_context({'font.family':style['font_family'],'font.size':10,'svg.hashsalt':NAME+'_'+dataset,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True}):
        fig, axes = plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
        for i,policy in enumerate(POLICIES):
            selected = [row for row in subset if row['row_policy'] == policy]
            labels = ['Raw' if row['family']=='RAW' else f'B2, K={row["K"]}' if row['family']=='B2' else 'Direct MLP' for row in selected]
            colors = ['#737373' if row['family']=='RAW' else '#0072B2' if row['family']=='B2' else '#D55E00' for row in selected]
            for j,(metric,title) in enumerate((('MSE','Normalized MSE ↓'),('p50_ms','Whole B8 latency (ms) ↓'),('peak_allocated_MiB','Peak allocated (MiB) ↓'))):
                ax = axes[i,j]
                mean = np.array([row[metric+'_mean'] for row in selected]); lower = np.array([row[metric+'_min'] for row in selected]); upper=np.array([row[metric+'_max'] for row in selected])
                positions=np.arange(len(selected))
                ax.barh(positions,mean,color=colors,height=.63,xerr=np.vstack((mean-lower,upper-mean)),error_kw=dict(ecolor='#222222',capsize=3,lw=1))
                extent=max(float(upper.max()),1e-9)
                ax.set_xlim(0,extent*1.28)
                for y,value,high in zip(positions,mean,upper):
                    ax.text(max(value,high)+.025*extent,y,f'{value:.4f}' if metric=='MSE' else f'{value:.2f}',va='center',fontsize=9)
                ax.set_yticks(positions,labels if j==0 else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
                ax.set_title(title if i==0 else '',fontsize=11,pad=10)
                if j==0:
                    cap=8*math.ceil(metadata['C']/4)
                    ax.set_ylabel('Default row128' if policy=='DEFAULT128' else f'Common row cap {cap}',fontsize=11,labelpad=12)
        title=('SYNTHETIC LAYOUT — ' if synthetic else '')+LABELS.get(dataset,dataset)
        fig.suptitle(title+' · L512 / H48 · batch 8',fontsize=14,fontweight='bold')
        fig.supxlabel('Bars: two-seed mean; whiskers: seed min–max (not confidence intervals). Raw: one deterministic model.\n'
            'Compare MSE within this dataset. Default and common-cap facets use the same selected models.',fontsize=9)
        output={}
        for fmt in ('png','svg'):
            stream=io.BytesIO()
            meta={'Software':NAME} if fmt=='png' else {'Date':None,'Creator':NAME,'Description':'SYNTHETIC LAYOUT ONLY' if synthetic else 'Observed development wave; two-seed range, no confidence interval'}
            fig.savefig(stream,format=fmt,dpi=300,metadata=meta,facecolor='white')
            output[fmt]=stream.getvalue()
        plt.close(fig)
    return output


def _csv(rows):
    stream=io.StringIO(newline='')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _texts(rows, proof):
    judged=proof['judgments']; positives=[ds for ds,result in judged.items() if result['positive']]
    horizons='; '.join(f'{LABELS.get(ds,ds)} H48={48*meta["interval_seconds"]/3600:g}시간' for ds,meta in proof['datasets'].items())
    lines=[f'# 데이터·K 탐색 wave {proof["wave"]}', '',
        ('등록한 positive 조건을 만족한 개발 예시는 '+', '.join(positives)+'다.' if positives else '이번 wave에서는 등록한 positive 조건을 만족한 데이터·K 조합이 없었다.'), '',
        '판정은 DEFAULT128에서 B2 두 seed 모두 Raw보다 MSE가 작고, 평균 MSE가 1% 이상 작으며, 두 seed의 지연과 allocated 피크가 모두 Raw보다 작아야 한다. '
        '이는 Raw 대비 탐색적 판정이다. Direct MLP가 더 좋은 경우도 함께 공개하며 MAE·COMMON_CAP은 별도 축으로 보고한다.', '']
    slide=[f'# Registered K trade-offs — wave {proof["wave"]}', '',
        'All candidates, widths and both seeds are disclosed. Positive is a fixed exploratory rule relative to Raw under DEFAULT128; it is not an independent TEST claim.', '']
    summary=aggregate(rows)
    for dataset in judged:
        lines.extend([f'## {LABELS.get(dataset,dataset)}', '', f'판정: {"POSITIVE_EXPLORATORY" if judged[dataset]["positive"] else "NEGATIVE_EXPLORATORY"}.', ''])
        slide.extend([f'## {LABELS.get(dataset,dataset)}', ''])
        for policy in POLICIES:
            lines.extend([f'### {policy}', '', '```text',
                'Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]'])
            for row in summary:
                if row['dataset']!=dataset or row['row_policy']!=policy:continue
                name='Raw' if row['family']=='RAW' else f'B2 K={row["K"]}' if row['family']=='B2' else 'Direct MLP'
                text='  '.join(f'{row[key+"_mean"]:.6f} [{row[key+"_min"]:.6f},{row[key+"_max"]:.6f}]' for key in METRICS)
                lines.append(f'{name:<13} {row["seeds"]:<5}  {text}')
            lines.extend(['```', ''])
        for k,result in judged[dataset]['widths'].items():
            lines.extend([f'K={k}: positive={result["positive"]}, Raw 대비 평균 MSE 감소 {result["MSE_gain_pct_vs_raw"]:+.2f}%, '
                f'MAE 변화 {result["mae_change_pct_vs_raw"]:+.2f}%, 지연 속도비 {result["speedup_vs_raw"]:.2f}배, '
                f'allocated 변화 {result["allocated_change_pct_vs_raw"]:+.2f}%. '
                f'Direct MLP의 MSE·지연·allocated 동시 지배: 평균={result["direct_dominates_mean"]}, '
                f'짝지은 seed={json.dumps(result["direct_dominates_paired_seeds"],sort_keys=True)}. '
                f'조건별 판정={json.dumps(result["predicates"],sort_keys=True)}.', ''])
            slide.extend([f'- K={k}: MSE gain vs Raw {result["MSE_gain_pct_vs_raw"]:+.2f}%; MAE change {result["mae_change_pct_vs_raw"]:+.2f}%; '
                f'default speed ratio {result["speedup_vs_raw"]:.2f}×; positive={result["positive"]}; '
                f'Direct MLP dominates mean={result["direct_dominates_mean"]}.'])
        own=[r for r in rows if r['dataset']==dataset and r['row_policy']=='DEFAULT128']
        first=own[0]
        lines.extend([f'V origin {first["n_origins"]}개 전체, 관측 목표가 있는 origin {first["scored_origins"]}개, '
            f'관측 목표 {first["observed_targets"]}개다. 원점별 관측 여부를 결과에 따라 제거하거나 목표를 보간하지 않았다.', ''])
        structure=proof['datasets'][dataset].get('train_structure') or {}
        lines.extend(['A_fit 구조 진단: '+json.dumps(structure,ensure_ascii=False,sort_keys=True)+'.', ''])
    lines.extend(['## 학습·검산과 해석 범위', '', '```text', 'run_id  selected_epoch  search_train_origins  training_status'])
    for job,item in sorted(proof['training'].items()):
        lines.append(f'{job}  {item["selected_epoch"]}  {item["search_train_origins"]}  {item["training_status"]}')
    lines.extend(['```', '',
        'TRAIN은 실제 stride1 전체 origin이며 진단용 stride48 목록 길이가 아니다. 모든 B2 검색을 A_stop에서 strict minimum·earlier tie로 고른 뒤 '
        'wave 전체 선택을 봉인하고 fresh A60 refit/V를 실행했다. Direct MLP도 자체 고정 A_stop 선택 후 fresh refit했다. '
        'cap3에서 계속 개선한 arm은 수렴한 것으로 해석하지 않는다.', '',
        f'검산한 비용 facet {proof["route_facets"]}개, warmup {proof["warmup_trials"]}회, timed {proof["timed_trials"]}회다. '
        f'B2 검색 {proof["B2_searches"]}개와 fresh refit {proof["B2_fresh_refits"]}개, Direct MLP search/refit {proof["Direct_MLP_search_refits"]}쌍을 포함한다. '
        'private 전체 V NPZ의 bytes·array 지문·FP64 equal-channel 관측 macro MSE/MAE, raw timing quantile·메모리 최대·F counters, '
        'A60 scaler·physical history/target mask·source/model/checkpoint/선택 봉인을 독립 검산했다. physical 출력 byte 배열은 저장되지 않아 '
        '그 지문의 재계산까지 주장하지 않는다.', '',
        '첫 8개 고정 V origin의 physical+mask→causal fill→고정 scaler→실제 모델→inverse→CPU 반환 전체 경로를 잰다. '
        'window slicing·load·검산·hash는 시간에서 제외한다. 피크는 CUDA allocated/reserved 텐서 기준이며 프로세스 hard cap이 아니다. '
        '20회 p95는 서비스 tail 보장이 아니고 두 seed 범위는 통계 신뢰구간이 아니다. 각 facet의 (PID, 생성 시각) pair가 모두 달라야 한다. '
        'OS가 같은 PID를 다른 생성 시각에 재사용하는 경우는 허용한다.', '',
        'H48/L512는 같은 스텝 수지만 실시간 길이는 다르다. 과거 Jena H48=8시간; '+horizons+'. '
        '강제 공통 주파수 리샘플링 없이 데이터 내부 방법 간 조건을 맞췄다. 서로 다른 데이터의 MSE 숫자를 직접 비교하지 않는다.', '',
        '현재 결과는 순차적으로 후보를 찾는 development V 탐색이다. 이전 다른 방법의 데이터 사용, 관련 원천·주파수 변형의 공유, '
        '기반모델 사전학습 중복 가능성을 배제하지 못한다. 새로운 독립 TEST 성능이나 통계적 유의성을 주장하지 않는다. '
        'LoRA·Ridge는 새 wave에서 실행하지 않았다. K·seed·학습 recipe·채널은 V 결과로 바꾸지 않았다.', '',
        'JENA_HISTORICAL_CONTEXT.csv는 과거 memory_budget_20261018_v1의 별도 실행 B8·confirm 결과다. '
        'K3/6/12·Raw·merged LoRA를 문맥으로 보존하며 새 후보 판정에는 쓰지 않는다. 이전 Jena 정규화/데이터 ancestry와 과거 TEST 노출 한계가 유지된다. '
        '과거 CUBLAS 환경의 실제 값은 기록되지 않았고, 새 비용은 부모 환경을 상속한 실제 값을 receipt에 기록했다.', '',
        '## 준비 정정과 전체 후보 보존', '',
        'PREPARATION_AMENDMENT_001에 따라 최초 ETTh2 prefix13936 중 16시간은 더 이른 같은 물리 원천 ETTm2 보호 경계와 겹칠 가능성이 있었다. '
        '최초 준비에서 그 수치가 materialize된 사실은 보존·공개한다. 해당 wave는 GPU 학습·예측 점수 계산 전에 superseded되었고, '
        '그 16시간의 새 학습/forecast 사용은 0이다. 원본·진단·WAVE001을 삭제하거나 덮어쓰지 않았다.', '',
        '교체 ETTh2_safe17400은 원천17420행의 처음17400좌표를 쓰는 custom development cohort이며 공식 ETT split이 아니다. '
        '기존 superseded NPZ를 자르는 대신 원본에서 허용된 prefix13920행만 다시 준비했다. 7개 물리 채널·hourly 주기는 유지했고 '
        'cutoff 결정에 예측 점수나 학습 recipe 변경을 사용하지 않았다.', '', '```text', 'wave dataset status detail'])
    for item in proof['candidates']:
        lines.append(f'{item["wave"]} {item["dataset"]} {item["status"]} {item["detail"]}')
    lines.extend(['```', '',
        '조건부 후속 후보는 CANDIDATE_PLAN_003 등 등록 계획과 준비 JSON의 상태만 읽어 표시했다. '
        'ETTm1(C7, K1/2/4)·Solar full137(C137, K18/35/70)은 현재 wave에 포함된 성능 실험으로 세지 않는다. '
        '부모 wave 전체 검산 후 positive가 없을 때만 다음 GPU wave를 활성화하며, positive가 있으면 준비된 후보를 READY_NOT_EXECUTED로 보존한다. '
        '분석기는 이 조건부 후보의 private numeric NPZ를 열거나 hash하지 않았다.', '',
        '전체 seed 원자료는 RUNS.csv, 평균·범위는 SUMMARY.csv, 모든 후보 상태는 CANDIDATES.csv에 있다.'])
    slide.extend(['', '## Figure footnote', '',
        'Use each dataset SVG: rows are DEFAULT128 and COMMON_CAP=8*ceil(C/4); columns are normalized MSE, whole B8 latency and allocated peak. '
        'Bars show means, whiskers seed min–max (not confidence intervals); Raw has one deterministic model. MAE and reserved memory remain in the tables. '
        'Direct MLP is displayed even when it dominates B2. Compare MSE within datasets only.', '',
        'Development V, adaptive candidate search, full stride1 TRAIN, two fixed seeds, cap3; no new independent TEST, LoRA or Ridge comparison. '
        'Equal L512/H48 steps imply different durations. Historical Jena is a separate earlier execution, with its existing ancestry/exposure limits.', '',
        'Preparation amendment: initial ETTh2 prefix materialized 16 potentially shared-source protected hours, with zero new training/forecast use. '
        'WAVE001 and its preparation are preserved as superseded. ETTh2_safe17400 is a custom first17400-source-coordinate cohort, '
        'with only13920 prefix rows prepared from the physical source; not the official ETT split.', '',
        'Conditionally prepared ETTm1 and full137-channel Solar are registry metadata only, with zero performance experiments counted here. '
        'Activate their next GPU wave only if the complete parent wave has no verified positive; otherwise preserve them as READY_NOT_EXECUTED.'])
    return '\n'.join(lines).encode('utf-8'), '\n'.join(slide).encode('utf-8')


def generate(repo=REPO, wave=2, output_dir=None):
    repo=Path(repo).resolve(); rows,proof=collect(repo,wave)
    text,slide=_texts(rows,proof); summary=aggregate(rows)
    outputs={'RUNS.csv':_csv(rows),'SUMMARY.csv':_csv(summary),'CANDIDATES.csv':_csv(proof['candidates']),
             'JENA_HISTORICAL_CONTEXT.csv':_csv(proof['historical']),'RESULTS_KO.md':text,'SLIDE_CONTENT_EN.md':slide}
    for dataset,meta in proof['datasets'].items():
        for policy in POLICIES:
            outputs[dataset+'_'+policy+'.csv']=_csv([row for row in summary if row['dataset']==dataset and row['row_policy']==policy])
        for extension,data in render_figures(rows,dataset,meta).items():
            outputs[dataset+'_COMPARISON.'+extension]=data
    import matplotlib
    validation=dict(proof,analysis_source_sha256=sha(__file__),numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        outputs={name:hashlib.sha256(value).hexdigest() for name,value in outputs.items()})
    outputs['VALIDATION.json']=(json.dumps(validation,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    destination=Path(output_dir).resolve() if output_dir else repo/'reports'/NAME/'waves'/f'WAVE_{wave:03d}_results'
    _require(any(destination.is_relative_to(root) for root in (repo/'reports'/NAME,repo/'.cache'/NAME)), 'Output escaped new namespace')
    for name,value in outputs.items():
        path=destination/name
        if path.exists() and path.read_bytes()!=value:
            raise FileExistsError('Preserve existing analysis output before rendering: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        path=destination/name
        if not path.exists():
            with path.open('xb') as stream:stream.write(value)
    return validation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=REPO);parser.add_argument('--wave',type=int,default=2)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    if args.check_only:
        rows,proof=collect(args.repo,args.wave)
        print(json.dumps(dict(status=proof['status'],wave=args.wave,route_facets=len(rows),judgments=proof['judgments']),sort_keys=True))
    else:
        result=generate(args.repo,args.wave,args.output_dir)
        print(json.dumps(dict(status=result['status'],outputs=result['outputs']),sort_keys=True))


if __name__=='__main__':
    main()
