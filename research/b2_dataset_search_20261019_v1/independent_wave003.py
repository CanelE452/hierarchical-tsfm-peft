"""CPU-only independent arithmetic audit of the terminal ETTm1 development wave.

This verifier imports neither producers nor analysis helpers. It does not infer,
train, reload a foundation model, inspect protected T, or hash whole raw sources.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]
SEEDS = (92601, 92602)
POLICIES = ('DEFAULT128', 'COMMON_CAP')
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
METRICS = ('MSE', 'MAE', 'p50_ms', 'peak_allocated_MiB', 'peak_reserved_MiB')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + json.dumps(list(array.shape)).encode() + array.tobytes()).hexdigest()


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
        'Independent value differs: ' + label)


def require_terminal(repo):
    repo = Path(repo).resolve()
    cache, report = repo / '.cache' / NAME, repo / 'reports' / NAME
    path = cache / 'TERMINAL_003.json'
    require(path.is_file(), 'Successful terminal evidence is required before any numeric array access')
    recovery_paths = {attempt: report / f'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_{attempt:03d}.json' for attempt in (2, 3)}
    if not any(candidate.exists() for candidate in recovery_paths.values()):
        original = read(path)
        require(original.get('status') == 'COMPLETE' and type(original.get('wave')) is int and original['wave'] == 3
            and type(original.get('exit_code')) is int and original['exit_code'] == 0,
            'Successful terminal exit0 is required before numeric array access')
    wave_sha = sha(report / 'waves' / 'WAVE_003.json')
    source_sha = sha(repo / 'research' / NAME / 'run.py')
    recovery = None
    recoveries = []
    final_attempt = 3 if recovery_paths[3].exists() else 2 if recovery_paths[2].exists() else 1
    for attempt in range(2, final_attempt + 1):
        recovery_path = recovery_paths[attempt]
        require(recovery_path.is_file(), 'Registered terminal recovery is missing its preceding amendment')
        registration = read(recovery_path)
        next_name = f'TERMINAL_003_ATTEMPT_{attempt:03d}.json'
        require(registration.get('status') == 'REGISTERED_BEFORE_RESUMED_PRODUCER'
            and type(registration.get('wave')) is int and registration['wave'] == 3
            and type(registration.get('attempt')) is int and registration['attempt'] == attempt
            and registration.get('previous_terminal') == {'reference': 'cache://' + path.name, 'sha256': sha(path)}
            and registration.get('next_terminal_reference') == 'cache://' + next_name
            and registration.get('unchanged_producer_sources') is True
            and registration.get('wave_sha256') == wave_sha and registration.get('producer_sha256') == source_sha,
            'Registered terminal recovery schema or binding differs')
        if attempt == 3:
            require(registration.get('preceding_recovery') == {
                'reference': 'report://' + recovery_paths[2].name, 'sha256': sha(recovery_paths[2])},
                'Registered terminal recovery preceding amendment binding differs')
            adapter_path = (repo / 'research' / NAME / 'resume_wave003.py').resolve()
            require(adapter_path.is_relative_to(repo) and adapter_path.is_file()
                and registration.get('execution_adapter') == {
                    'reference': 'repo://research/' + NAME + '/resume_wave003.py', 'sha256': sha(adapter_path)}
                and registration.get('identity_serialization_repair') == 'JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY',
                'Registered terminal recovery adapter path, source or repair scope differs')
        previous = read(path)
        require(previous.get('status') == 'FAILED' and type(previous.get('wave')) is int and previous['wave'] == 3
            and type(previous.get('exit_code')) is int and previous['exit_code'] != 0,
            'Registered terminal recovery requires preserved failed wave-3 evidence')
        require(all(type(previous.get(key)) in (int, float) and math.isfinite(previous[key])
            for key in ('started_unix', 'finished_unix', 'wall_seconds')),
            'Registered terminal recovery prior timing is missing or nonfinite')
        registered = registration.get('registered_unix')
        require(previous['finished_unix'] >= previous['started_unix'] and previous['wall_seconds'] >= 0
            and type(registered) in (int, float) and math.isfinite(registered) and registered > previous['finished_unix'],
            'Registered terminal recovery timing does not follow the failed attempt')
        recoveries.append(dict(binding=dict(reference='repo://' + recovery_path.relative_to(repo).as_posix(), sha256=sha(recovery_path)),
            registration=registration, previous_terminal=dict(previous, reference='cache://' + path.name, sha256=sha(path))))
        path = cache / next_name
        require(path.is_file(), 'Registered successful retry terminal is required before numeric array access')
        resumed = read(path)
        require(type(resumed.get('started_unix')) in (int, float) and math.isfinite(resumed['started_unix'])
            and registered <= resumed['started_unix'] and resumed['started_unix'] >= previous['finished_unix'],
            'Registered terminal recovery was not sealed before retry execution')
    if recoveries:
        recovery = dict(recoveries[-1], preceding_recoveries=recoveries[:-1])
    terminal = read(path)
    require(terminal.get('status') == 'COMPLETE' and type(terminal.get('wave')) is int and terminal['wave'] == 3
        and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0,
        'Successful terminal exit0 is required before numeric array access')
    require(all(type(terminal.get(key)) in (int, float) and math.isfinite(terminal[key])
        for key in ('started_unix', 'finished_unix'))
        and terminal['finished_unix'] >= terminal['started_unix'], 'Successful terminal timing is invalid')
    if recovery is not None:
        require(type(terminal.get('wall_seconds')) in (int, float) and math.isfinite(terminal['wall_seconds'])
            and terminal['wall_seconds'] >= 0
            and recovery['registration']['registered_unix'] <= terminal['started_unix']
            and terminal['started_unix'] >= recovery['previous_terminal']['finished_unix'],
            'Registered terminal recovery was not sealed before retry execution')
    events = [json.loads(line) for line in (cache / 'PROCESS_ATTEMPTS.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    selected = {}
    for stage in ('train', 'baselines', 'costs'):
        records = [record for record in events if record.get('wave') == 3 and record.get('stage') == stage]
        require(records and records[-1].get('status') == 'COMPLETE'
            and any(record.get('status') == 'STARTED' for record in records), 'Producer stage is not terminal: ' + stage)
        require(all(record.get('wave_sha256') == wave_sha and record.get('producer_sha256') == source_sha
            and type(record.get('observed_unix')) in (int, float) and math.isfinite(record['observed_unix'])
            for record in records), 'Producer journal wave/source binding changed')
        selected[stage] = records[-1]
        require(terminal['started_unix'] <= records[-1]['observed_unix'] <= terminal['finished_unix'],
            'Producer completion is outside the successful terminal attempt')
    require(selected['train']['observed_unix'] <= selected['baselines']['observed_unix'] <= selected['costs']['observed_unix'],
        'Producer stages completed in the wrong order')
    return dict(terminal_sha256=sha(path), driver=terminal, stage_completions=selected,
        **({'execution_recovery': recovery} if recovery is not None else {}))


def resolve(binding, repo, cache, public):
    scheme, relative = binding['reference'].split('://', 1)
    root = {'repo': Path(repo), 'cache': Path(cache), 'report': Path(public)}.get(scheme)
    require(root is not None, 'Unknown artifact binding scheme')
    root = root.resolve(); path = (root / relative).resolve()
    require(path.is_relative_to(root) and NAME in path.parts and sha(path) == binding['sha256'],
        'Bound artifact path or bytes changed: ' + binding['reference'])
    return path


def check_sources(files, repo):
    require(isinstance(files, dict) and bool(files), 'Nonempty source graph required')
    repo = Path(repo).resolve()
    for name, digest in files.items():
        root = repo if '/' in name or '\\' in name else repo / 'research/informs_broad_validation_20261015_v1'
        path = (root / name).resolve()
        require(path.is_relative_to(repo) and path.suffix == '.py' and sha(path) == digest,
            'Producer source bytes differ: ' + name)


def metrics(prediction, target, observed):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    require(p.ndim == 3 and p.shape == y.shape == mask.shape and mask.dtype == np.bool_,
        'Aligned three-dimensional predictions, targets and bool mask required')
    require(np.isfinite(p).all() and np.isfinite(y[mask]).all(), 'Nonfinite observed forecast or target')
    mses, maes, counts, sses, saes = [], [], [], [], []
    for channel in range(p.shape[-1]):
        selected = mask[:, :, channel]
        require(selected.any(), 'Every original channel needs observed targets')
        error = p[:, :, channel][selected].astype(np.float64) - y[:, :, channel][selected].astype(np.float64)
        sse = math.fsum(float(value) * float(value) for value in error)
        sae = math.fsum(abs(float(value)) for value in error)
        count = int(selected.sum())
        counts.append(count); sses.append(sse); saes.append(sae)
        mses.append(sse / count); maes.append(sae / count)
    return dict(MSE=math.fsum(mses) / len(mses), MAE=math.fsum(maes) / len(maes),
        observed=sum(counts), channel_counts=counts, channel_sse=sses, channel_sae=saes)


def scaler(physical, observed, end):
    require(0 < end <= len(physical) and observed.dtype == np.bool_ and observed.shape == physical.shape,
        'Observed physical prefix required for normalization')
    means, scales = [], []
    for channel in range(physical.shape[1]):
        values = physical[:end, channel][observed[:end, channel]].astype(np.float64)
        require(len(values) > 0 and np.isfinite(values).all(), 'Unobserved or nonfinite normalization channel')
        mean = math.fsum(map(float, values)) / len(values)
        std = math.sqrt(math.fsum((float(value)-mean)**2 for value in values) / len(values))
        means.append(mean); scales.append(std if std else 1.)
    return np.asarray(means, np.float32), np.asarray(scales, np.float32)


def verify_arrays(arrays, config, physical, observed):
    length, horizon, channels = config['L'], config['H'], len(config['columns'])
    lo, hi = config['split_ranges']['V_select']
    origins = np.asarray(list(range(max(lo, length), hi-horizon+1, horizon)), np.int64)
    require(arrays['origins'].dtype == np.int64 and np.array_equal(arrays['origins'], origins), 'Canonical V origins changed')
    require(arrays['channel_ids'].tolist() == config['columns'], 'Original channel order changed')
    mean, std = scaler(physical, observed, config['split_ranges']['A_stop'][1])
    require(np.array_equal(arrays['mean'], mean) and np.array_equal(arrays['std'], std), 'A60-only scaler changed')
    rows = origins[:, None] + np.arange(horizon)[None]
    mask = observed[rows]
    target = ((np.where(mask, physical[rows], mean)-mean)/std).astype(np.float32)
    require(arrays['observed'].dtype == np.bool_ and np.array_equal(arrays['observed'], mask), 'Original target mask changed')
    require(arrays['target_normalized'].dtype == np.float32 and np.array_equal(arrays['target_normalized'], target),
        'Canonical observed normalized targets changed')
    canonical, regenerated = arrays['reference_pred_normalized'], arrays['pred_normalized']
    require(canonical.shape == regenerated.shape == target.shape == (len(origins), horizon, channels)
        and regenerated.dtype == np.float64 and canonical.dtype in (np.dtype('float32'), np.dtype('float64')),
        'Prediction archive shape or registered precision changed')
    require(np.allclose(canonical, regenerated, atol=1e-5, rtol=1e-4), 'Whole V row-policy prediction parity failed')
    score, rebuilt = metrics(canonical, target, mask), metrics(regenerated, target, mask)
    for key in ('MSE', 'MAE'):
        close(score[key], rebuilt[key], 'Regenerated ' + key, atol=1e-5, rtol=1e-4)
    return dict(score, n_origins=len(origins), scored_origins=int(mask.any(axis=(1, 2)).sum()))


def select_epoch(curves):
    pairs = [(float(row['epoch']), float(row['astop_mse'])) for row in curves if row.get('astop_mse') not in (None, '')]
    require([epoch for epoch, _ in pairs] == [0., .25, 1., 3.]
        and all(math.isfinite(value) for _, value in pairs), 'Complete fixed checkpoint schedule required')
    return min(pairs, key=lambda pair: (pair[1], pair[0]))


def exposure_arithmetic(origins, epoch, *, refit=False):
    endpoints = [float(epoch)] if refit else [.25, 1., 3.]
    previous, updates = 0, 0
    for endpoint in endpoints:
        exposure = math.ceil(endpoint * origins)
        updates += math.ceil((exposure-previous) / 16)
        previous = exposure
    return previous, updates


def _quantile(values, fraction):
    values = sorted(values); index = (len(values)-1)*fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high]-values[low])*(index-low)


def cost_statistics(receipt):
    trials, summary = receipt['trials'], receipt['summary']
    family, width = receipt['family'], receipt['actual_F_width']
    require(width == (receipt['C'] if family == 'RAW' else receipt['K'] if family == 'B2' else 0)
        == receipt['model_K'], 'Registered actual F width differs')
    cap = (128 if receipt['row_policy'] == 'DEFAULT128' else 8*math.ceil(receipt['C']/4)) if width else None
    effective = min(cap, 8*width) if width else None
    sizes = [min(effective, 8*width-start) for start in range(0, 8*width, effective)] if width else []
    work = dict(q=math.ceil(receipt['C']/4), width=width, streams=8*width,
        requested_row_cap=cap, effective_row_cap=effective, calls=len(sizes), row_sizes=sizes)
    require(receipt['expected_work'] == work and receipt['requested_row_cap'] == cap
        and receipt['effective_row_cap'] == effective, 'Actual row-cap work differs')
    require(len(trials) == 30 and summary['warmups'] == 10 and summary['timed_trials'] == 20
        and summary['status'] == 'COMPLETE' and summary['parity_status'] == 'PASS', 'Complete raw timing trial schedule required')
    require(summary['process_pid'] == summary['pid'] and math.isfinite(summary['process_create_time'])
        and summary['process_scope'] == 'fresh OS process, one facet, one serialized model load', 'Fresh process identity differs')
    for phase, count in (('warmup', 10), ('timed', 20)):
        group = [trial for trial in trials if trial['phase'] == phase]
        require([row['index'] for row in group] == list(range(count))
            and [row['trial'] for row in group] == list(range(count)), 'Exact raw trial indices differ')
        for row in group:
            require(row['pid'] == summary['pid'] and math.isfinite(row['whole_seconds']) and row['whole_seconds'] > 0
                and row['F_calls'] == len(sizes) and row['F_logical_streams'] == row['F_physical_streams'] == 8*width,
                'Raw trial process/F work/time differs')
    timed = [row for row in trials if row['phase'] == 'timed']
    values = [1000*row['whole_seconds'] for row in timed]
    q25, p50, q75, p95 = [_quantile(values, fraction) for fraction in (.25, .5, .75, .95)]
    derived = dict(p50_ms=p50, IQR_ms=q75-q25, p95_ms=p95)
    for key, value in dict(median_batch_ms=p50, IQR_batch_ms=q75-q25, p95_batch_ms=p95,
            median_ms_per_origin=p50/8, origins_per_second=8000/p50).items():
        close(summary[key], value, key)
    for key in ('peak_allocated_MiB', 'peak_reserved_MiB'):
        values = [row[key] for row in timed]
        require(all(math.isfinite(value) and value >= 0 for value in values), 'Invalid raw memory trial')
        derived[key] = max(values); close(summary[key], max(values), key)
    require(derived['peak_reserved_MiB'] >= derived['peak_allocated_MiB'], 'Reserved peak below allocated')
    require(summary['F_calls'] == len(sizes) and summary['F_logical_streams'] == 8*width, 'Summary F work differs')
    return dict(derived, process_identity=[summary['pid'], summary['process_create_time']])


def judge(rows):
    raw = [row for row in rows if row['family'] == 'RAW']
    direct = [row for row in rows if row['family'] == 'DIRECT_G']
    require(len(raw) == 1 and sorted(row['seed'] for row in direct) == list(SEEDS), 'Raw/Direct seed coverage differs')
    raw = raw[0]
    keys = ('MSE', 'MAE', 'p50_ms', 'peak_allocated_MiB')
    mean_direct = {key: math.fsum(row[key] for row in direct)/2 for key in keys}
    def dominates(left, right):
        axes = ('MSE', 'p50_ms', 'peak_allocated_MiB')
        return all(left[key] <= right[key] for key in axes) and any(left[key] < right[key] for key in axes)
    widths = {}
    for width in (1, 2, 4):
        ours = [row for row in rows if row['family'] == 'B2' and row['K'] == width]
        require(sorted(row['seed'] for row in ours) == list(SEEDS), 'Every registered width needs both seeds')
        mean = {key: math.fsum(row[key] for row in ours)/2 for key in keys}
        predicates = dict(both_seed_MSE_below_Raw=all(row['MSE'] < raw['MSE'] for row in ours),
            mean_MSE_at_least_1pct_below_Raw=mean['MSE'] <= .99*raw['MSE'],
            both_seed_DEFAULT128_median_latency_below_Raw=all(row['p50_ms'] < raw['p50_ms'] for row in ours),
            both_seed_DEFAULT128_allocated_peak_below_Raw=all(row['peak_allocated_MiB'] < raw['peak_allocated_MiB'] for row in ours))
        widths[str(width)] = dict(positive=all(predicates.values()), predicates=predicates, two_seed_mean=mean,
            seed_values={str(row['seed']): {key: row[key] for key in keys} for row in ours},
            MSE_gain_pct_vs_Raw=100*(1-mean['MSE']/raw['MSE']), MAE_change_pct_vs_Raw=100*(mean['MAE']/raw['MAE']-1),
            speed_ratio_vs_Raw=raw['p50_ms']/mean['p50_ms'],
            allocated_change_pct_vs_Raw=100*(mean['peak_allocated_MiB']/raw['peak_allocated_MiB']-1),
            Direct_dominates_two_seed_mean=dominates(mean_direct, mean),
            Direct_dominates_paired_seeds={str(row['seed']): dominates(next(d for d in direct if d['seed'] == row['seed']), row) for row in ours})
    return dict(positive=any(item['positive'] for item in widths.values()),
        Raw={key: raw[key] for key in keys}, Direct_two_seed_mean=mean_direct, widths=widths)


def _load_cohort(repo, entry):
    folder = repo / '.cache' / NAME / 'data' / 'ETTm1'
    path = folder / 'DATA_DIAGNOSTIC.json'; config = read(path)
    require(sha(path) == entry['config_sha256'] and config['prefix_sha256'] == entry['prefix_sha256']
        and config['status'] == 'READY' and config['prefix_only_approved'] is True
        and config['candidate_training_admitted'] is True, 'Registered ETTm1 cohort changed')
    require((config['dataset'], config['C'], config['L'], config['H'], config['original_rows'],
        config['prefix_rows'], config['interval_seconds']) == ('ETTm1', 7, 512, 48, 69680, 55744, 900),
        'Full original ETTm1 cohort/native frequency differs')
    require(config['source_full_file_hashed'] is False and not any(config['protected_numeric_access'].values()),
        'Protected source preparation scope changed')
    bounds = {key: config['split_ranges'][key] for key in ('A_fit', 'A_stop', 'M_cal', 'V_select', 'T_final')}
    expected = dict(A_fit=[0, 34840], A_stop=[34840, 41808], M_cal=[41808, 48776],
        V_select=[48776, 55744], T_final=[55744, 69680])
    require(bounds == expected, 'Registered chronological ETTm1 boundaries differ')
    prefix = folder / 'PREFIX_PHYSICAL.npz'
    with np.load(prefix, allow_pickle=False) as archive:
        require(set(archive.files) == {'physical', 'observed', 'times', 'columns'}, 'Unexpected prefix archive member')
        headers = {}
        for name in archive.files:
            with archive.zip.open(name + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported bounded prefix header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                headers[name] = reader(stream)
        require(headers['physical'] == ((55744, 7), False, np.dtype('float32'))
            and headers['observed'] == ((55744, 7), False, np.dtype('bool'))
            and headers['times'] == ((55744,), False, np.dtype('int64'))
            and headers['columns'][0] == (7,) and headers['columns'][2].kind == 'U',
            'Numeric prefix header exceeds the approved bounded cohort')
        require(sha(prefix) == entry['prefix_sha256'], 'Approved prefix bytes changed')
        values = {name: archive[name].copy() for name in archive.files}
    require(values['columns'].tolist() == config['columns']
        and np.all(np.diff(values['times']) == 900_000_000_000), 'Original channel/time grid changed')
    require({key: array_identity(value)['sha256'] for key, value in values.items()} == config['prefix_array_hashes'],
        'Preparation array fingerprints changed')
    values['physical'][~values['observed']] = np.nan
    require(np.isfinite(values['physical'][values['observed']]).all(), 'Observed physical prefix is nonfinite')
    hashes = {key: array_digest(value) for key, value in values.items()}
    data_sha = fingerprint(dict(arrays=hashes, split_ranges=bounds, L=512, H=48))
    return config, values, hashes, data_sha


def _fit_arithmetic(fit, curves, search_summary, refit_summary, config):
    epoch, selected_mse = select_epoch(curves)
    search_n = config['split_ranges']['A_fit'][1]-config['L']-config['H']+1
    refit_n = config['split_ranges']['A_stop'][1]-config['L']-config['H']+1
    stop_lo, stop_hi = config['split_ranges']['A_stop']
    stop_n = len(range(max(stop_lo, config['L']), stop_hi-config['H']+1, config['H']))
    search_e, search_u = exposure_arithmetic(search_n, 3)
    refit_e, refit_u = exposure_arithmetic(refit_n, epoch, refit=True)
    expected = dict(selected_epoch=epoch, selected_astop_mse=selected_mse,
        search_fit_end=config['split_ranges']['A_fit'][1], refit_fit_end=config['split_ranges']['A_stop'][1],
        normalization_fit_end=config['split_ranges']['A_stop'][1], search_train_origins=search_n,
        search_stop_origins=stop_n, search_exposures=search_e, search_updates=search_u,
        refit_exposures=refit_e, refit_updates=refit_u)
    require(fit['status'] == 'COMPLETE_VALID' and fit['optimizer'] == 'AdamW' and fit['weight_decay'] == 0
        and all(fit[key] == value for key, value in expected.items()), 'Search/refit selection, recipe or full-origin arithmetic differs')
    for label, summary, count, exposures, updates in (('search', search_summary, search_n, search_e, search_u),
            ('refit', refit_summary, refit_n, refit_e, refit_u)):
        require(summary['status'] == 'SEGMENT_COMPLETE' and summary['train_origins'] == count
            and summary['exposures'] == exposures and summary['updates'] == updates,
            'Stage receipt arithmetic differs: ' + label)
        close(summary['actual_epochs'], exposures/count, label + ' actual epoch')
    return expected


def _training(repo, config, values, hashes, data_sha):
    report, cache = repo / 'reports' / NAME, repo / '.cache' / NAME
    public, private = report / 'datasets' / 'ETTm1', cache / 'datasets' / 'ETTm1'
    seal_path = report / 'waves' / 'WAVE_003_SEARCHES_SEALED.json'; seal = read(seal_path)
    plan_path = resolve(seal['wave_plan_binding'], repo, cache, report); plan = read(plan_path)
    jobs = [f'B2_ETTm1_K{k}_OFF_s{seed}' for k in (1, 2, 4) for seed in SEEDS]
    require(seal['status'] == 'SEALED' and seal['all_searches_complete'] is True and seal['V_used'] is False
        and seal['selection_split'] == 'A_stop' and seal['cap'] == 3 and seal['matched_arms'] == jobs
        and set(seal['selections']) == set(jobs) and set(seal['datasets']) == {'ETTm1'}, 'Shared all-search selection seal differs')
    require(plan['status'] == 'SEALED' and plan['phase'] == 'BEFORE_SEARCH' and plan['V_used'] is False
        and plan['matched_arms'] == jobs and plan['dataset_order'] == ['ETTm1']
        and plan['epochs'] == [.25, 1., 3.] and plan['cap'] == 3, 'Before-search wave plan differs')
    require(resolve(plan['protocol_binding'], repo, cache, report) == report / 'PROTOCOL.md',
        'Before-search plan points to a different protocol')
    identity_path = resolve(seal['datasets']['ETTm1']['identity_binding'], repo, private, public)
    identity = read(identity_path); check_sources(identity['sources'], repo)
    manifest = read(report / 'SOURCE_MANIFEST_TRAIN_003.json')
    require(identity['sources'] == manifest['sources'] and manifest['status'] == 'SEALED_BEFORE_TRAINING'
        and manifest['coordinator_sha256'] == sha(repo / 'research' / NAME / 'run.py')
        and manifest['activated_wave_sha256'] == sha(report / 'waves' / 'WAVE_003.json'), 'Pretraining producer manifest changed')
    for key in ('dataset', 'jobs', 'K_grid', 'data_sha256', 'input_config_sha256', 'prefix_sha256', 'sources'):
        require(identity[key] == plan['datasets']['ETTm1'][key], 'Shared search plan identity differs: ' + key)
    require(identity['data_sha256'] == data_sha
        and identity['input_config_sha256'] == sha(cache / 'data' / 'ETTm1' / 'DATA_DIAGNOSTIC.json')
        and identity['prefix_sha256'] == config['prefix_sha256']
        and identity['TRAIN_stride'] == 1 and identity['evaluation_stride'] == 48
        and identity['revin'] is False and identity['T_access'] is False and identity['V_independent_test'] is False,
        'Training cohort/stride/scope differs')
    summary = read(public / 'TRAIN_SUMMARY.json')
    require(summary['status'] == 'COMPLETE' and summary['new_searches'] == summary['fresh_refits'] == 6
        and summary['rows'] == read(public / 'VERIFIED_ROWS.json') == read(private / 'VERIFIED_ROWS.json'), 'Fresh refit coverage differs')
    require(resolve(summary['selection_binding'], repo, private, public) == seal_path
        and resolve(identity['wave_plan_binding'], repo, private, public) == plan_path, 'Shared selection/plan binding differs')
    verified = {row['run_id']: row for row in summary['rows']}
    require(set(verified) == set(jobs), 'All registered widths/seeds must remain visible')
    output, identities = {}, {}
    for job in jobs + [f'DirectMLP_ETTm1_s{seed}' for seed in SEEDS]:
        is_b2 = job in jobs
        run = public / 'runs' / job / 'cap3' if is_b2 else public / 'runs' / job
        source = read(run / 'SOURCE_CONTRACT.json'); check_sources(source['files'], repo)
        fit = read(run / 'BASE_FIT_RECEIPT.json')
        if is_b2:
            search, result = read(run / 'SEARCH_RESULT.json'), read(run / 'REFIT_RESULT.json')
            curves, search_summary, refit_summary = search['curves'], result['search_summary'], result['refit_summary']
            selected = seal['selections'][job]
            row = verified[job]
            require(search['status'] == result['status'] == row['status'] == 'COMPLETE'
                and result['fit_receipt'] == fit and result['selection_sealed_before_V'] is True
                and row['all_searches_sealed_before_V'] is True and row['V_independent_test'] is False,
                'Fresh B2 completion or pre-V seal differs')
            require(search['summary'] == search_summary and search['selected_epoch'] == selected['epoch'] == row['selected_epoch']
                and selected['astop_mse'] == search_summary['selected_astop_mse'] and selected['model_sha'] == F_SHA,
                'Shared seal disagrees with saved selection')
            require(resolve(search['checkpoint_binding'], repo, private, public)
                == resolve(selected['checkpoint'], repo, private, public), 'Sealed search checkpoint differs')
            require(resolve(result['selection_decision_binding'], repo, private, public) == seal_path
                and resolve(row['selection_decision_binding'], repo, private, public) == seal_path, 'Fresh B2 selection binding differs')
            artifact = resolve(row['bundle_binding'], repo, private, public)
            require(resolve(result['bundle_binding'], repo, private, public) == artifact and sha(artifact) == fit['artifact_sha256']
                and fit['fresh_A60'] is True, 'Fresh B2 bundle binding differs')
            require(search['contract'] == {key: source[key] for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha')},
                'Search source/model/data/config binding differs')
        else:
            with (run / 'search' / 'CURVE.csv').open(newline='', encoding='utf-8-sig') as stream:
                curves = list(csv.DictReader(stream))
            search_summary, refit_summary = read(run / 'search' / 'RECEIPT.json'), read(run / 'refit' / 'RECEIPT.json')
            full_fit = read(public / (job + '_FIT.json'))
            require(all(full_fit[key] == value for key, value in fit.items() if key != 'reused_complete_artifact')
                and full_fit['T_access'] is False and full_fit['V_independent_test'] is False and full_fit['F_instances'] == 0,
                'Matched Direct MLP completion differs')
            artifact = resolve(full_fit['bundle_binding'], repo, private, public)
            require(sha(artifact) == fit['artifact_sha256'], 'Direct MLP selected bundle changed')
        spec = source['spec']; family, seed = ('B2' if is_b2 else 'DIRECT_G'), int(job.rsplit('s', 1)[1])
        expected = dict(run_id=job, dataset='ETTm1', family=family, seed=seed, device='cuda', epochs=[.25, 1., 3.],
            lr=.001, effective_batch=16, microbatch=4, F_row_microbatch=128, checkpoint_inputs=False)
        if is_b2:
            expected.update(latent_k=verified[job]['K'], revin=False, init_kind='PCA')
        require(all(spec.get(key) == value for key, value in expected.items()), 'Fixed matched training recipe differs')
        axes = dict(L=512, H=48, C=7, columns=config['columns'], fold='primary', source_array_fingerprints=hashes)
        require(source['source_sha'] == fingerprint(source['files']) and source['data_sha'] == data_sha
            and source['config_sha'] == fingerprint(dict(spec=spec, axes=axes))
            and source['model_sha'] == (F_SHA if is_b2 else 'NO_FOUNDATION') and fit['source_sha'] == source['source_sha'],
            'Scientific source/model/config/data identity differs')
        arithmetic = _fit_arithmetic(fit, curves, search_summary, refit_summary, config)
        mean, std = scaler(values['physical'], values['observed'], config['split_ranges']['A_stop'][1])
        require(fit['refit_scaler_sha'] == fingerprint(dict(mean=array_identity(mean), std=array_identity(std))),
            'Fresh A60 receipt scaler SHA differs')
        output[job] = dict(family=family, seed=seed, **arithmetic, bundle_sha256=sha(artifact),
            source_contract_sha256=sha(run / 'SOURCE_CONTRACT.json'), refit_receipt_sha256=sha(run / 'BASE_FIT_RECEIPT.json'),
            search_last_checkpoint_strictly_best=float(curves[-1]['astop_mse']) < min(float(row['astop_mse']) for row in curves[:-1]))
        identities[job] = dict(source=source, fit=fit, artifact=artifact, row=verified.get(job))
    return output, identities, identity['sources'], sha(seal_path), sha(plan_path)


def audit(repo=REPO):
    repo = Path(repo).resolve(); terminal = require_terminal(repo)
    report, cache = repo / 'reports' / NAME, repo / '.cache' / NAME
    wave_path = report / 'waves' / 'WAVE_003.json'; wave = read(wave_path)
    require(wave['status'] == 'REGISTERED_BEFORE_TRAINING' and wave['wave'] == 3
        and wave['protocol_sha256'] == sha(report / 'PROTOCOL.md') and wave['V_forecast_scores_used'] is False
        and [(entry['dataset'], entry['K_grid']) for entry in wave['datasets']] == [('ETTm1', [1, 2, 4])],
        'Registered wave/protocol/grid differs')
    activation_path = report / 'ACTIVATION_AMENDMENT_WAVE_003.json'; activation = read(activation_path)
    require(sha(activation_path) == wave['activation_amendment_sha256'] and activation['activated_datasets'] == ['ETTm1']
        and activation['activated_wave'] == 3 and activation['new_authorization_overrides_prior_conditional_stop'] is True
        and activation['V_scores_used_to_choose_wave'] is False and activation['protected_T_access'] is False
        and wave['candidate_plan_sha256'] == sha(report / 'CANDIDATE_PLAN_003.json'), 'Latest pre-score activation binding differs')
    config, values, hashes, data_sha = _load_cohort(repo, wave['datasets'][0])
    training, fitted, sources, seal_sha, plan_sha = _training(repo, config, values, hashes, data_sha)
    private, public = cache / 'datasets' / 'ETTm1', report / 'datasets' / 'ETTm1'
    arms = [('RAW', 7, SEEDS[0], 'Raw_ETTm1')]
    arms += [('B2', width, seed, f'B2_ETTm1_K{width}_OFF_s{seed}') for width in (1, 2, 4) for seed in SEEDS]
    arms += [('DIRECT_G', 0, seed, f'DirectMLP_ETTm1_s{seed}') for seed in SEEDS]
    expected = {job + '_' + policy + '_B8.json' for _, _, _, job in arms for policy in POLICIES}
    require({path.name for path in (public / 'costs').glob('*.json')} == expected, 'Exact 18-facet cost coverage required')
    facets, pids, canonical, receipts = [], set(), {}, {}
    for family, width, seed, job in arms:
        for policy in POLICIES:
            path = public / 'costs' / (job + '_' + policy + '_B8.json'); receipt = read(path)
            require(receipt['schema'] == 'b2_dataset_search_cost_v1' and receipt['status'] == 'COMPLETE'
                and all(receipt[key] == value for key, value in dict(dataset='ETTm1', run_id=job, family=family,
                    seed=seed, row_policy=policy, C=7, L=512, H=48, batch_size=8, model_K=width,
                    config_sha256=wave['datasets'][0]['config_sha256'], prefix_sha256=wave['datasets'][0]['prefix_sha256'],
                    data_sha256=data_sha, protocol_sha256=wave['protocol_sha256'], expected_f_sha256=F_SHA,
                    expected_f_params=47_718_016).items()), 'Cost facet scientific/registered identity differs')
            check_sources(receipt['producer_sources'], repo)
            require(receipt['source_bindings'] == dict(config_sha256=wave['datasets'][0]['config_sha256'],
                prefix_sha256=wave['datasets'][0]['prefix_sha256'], data_sha256=data_sha, sources=sources),
                'Cost loaded data/source graph differs from training')
            request = read(cache / 'cost_requests' / (job + '_' + policy + '.json'))
            request.update(batch_size=8, device='cuda', factory='train:cost_inputs', protocol_sha256=wave['protocol_sha256'])
            require(receipt['private_request_sha256'] == hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                'Exact private cost request changed')
            prediction = resolve(dict(reference=receipt['prediction_reference'], sha256=receipt['prediction_file_sha256']), repo, private, public)
            with np.load(prediction, allow_pickle=False) as archive:
                require(set(archive.files) == {'pred_normalized', 'reference_pred_normalized', 'target_normalized',
                    'observed', 'origins', 'channel_ids', 'mean', 'std'}, 'Prediction archive schema changed')
                arrays = {key: archive[key].copy() for key in archive.files}
            score = verify_arrays(arrays, config, values['physical'], values['observed'])
            timing = cost_statistics(receipt); identity = tuple(timing.pop('process_identity'))
            require(identity not in pids, 'Every cost facet needs a distinct PID/create-time pair'); pids.add(identity)
            require(receipt['states_before'] == receipt['states_after'], 'Frozen pre/post model state changed')
            summary, parity = receipt['summary'], receipt['parity']
            require(read(public / 'cost_validation' / path.name) == parity and parity['status'] == 'PASS'
                and parity['full_V_origins'] == score['n_origins'] and parity['last_batch_size'] == score['n_origins'] % 8
                and parity['atol'] == 1e-5 and parity['rtol'] == 1e-4, 'Separate whole-V parity receipt differs')
            require(summary['loaded_F_state_sha256'] == (F_SHA if family != 'DIRECT_G' else 'NO_FOUNDATION_USED')
                and summary['core_dtype'] == 'float32' and summary['metric_accumulation_dtype'] == 'float64'
                and summary['cached_forecast_reads_in_timed_path'] == 0
                and summary['foundation_instances_created'] == (0 if family == 'DIRECT_G' else 1), 'Loaded F/precision/online measurement differs')
            for key in ('MSE', 'MAE'):
                close(summary[key], score[key], 'Canonical summary ' + key)
                close(parity['original_metrics'][key], score[key], 'Original parity ' + key)
                rebuilt = metrics(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'])
                close(summary['recomputed_' + key], rebuilt[key], 'Regenerated summary ' + key)
                close(parity['independent_metrics'][key], rebuilt[key], 'Regenerated parity ' + key)
            for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
                require(parity['input_fingerprints'][key] == array_identity(arrays[key]), 'Input array fingerprint changed: ' + key)
            histories = arrays['origins'][:, None] + np.arange(-512, 0)[None]
            for name, data in dict(physical_windows=values['physical'][histories],
                    physical_observed_windows=values['observed'][histories]).items():
                require(parity['input_fingerprints'][name] == array_identity(data), 'Physical history fingerprint changed')
            reference_sha = array_identity(arrays['reference_pred_normalized'])
            require(parity['output_fingerprints']['original_normalized'] == reference_sha
                and parity['output_fingerprints']['regenerated_normalized'] == array_identity(arrays['pred_normalized'])
                and receipt['original_reference']['canonical_prediction_fingerprint'] == reference_sha, 'Output array fingerprints changed')
            if job in canonical:
                require(np.allclose(canonical[job], arrays['reference_pred_normalized'], atol=1e-5, rtol=1e-4),
                    'Policies do not preserve the same canonical forecasts')
            else:
                canonical[job] = arrays['reference_pred_normalized']
            if family != 'RAW':
                require(summary['checkpoint_sha256'] == sha(fitted[job]['artifact']), 'Timed adapter differs from selected fresh refit')
            if family == 'B2':
                row = fitted[job]['row']
                for key in ('MSE', 'MAE'):
                    close(row[key], score[key], 'Saved B2 ' + key)
            facets.append(dict(run_id=job, family=family, K=width, seed=seed, row_policy=policy,
                **score, **timing, process_identity=list(identity), receipt_sha256=sha(path),
                prediction_sha256=sha(prediction)))
            receipts[path.relative_to(repo).as_posix()] = sha(path)
    judgment = judge([row for row in facets if row['row_policy'] == 'DEFAULT128'])
    return dict(status='PASS', wave=3, dataset='ETTm1',
        result='POSITIVE_VERIFIED_EXPLORATORY' if judgment['positive'] else 'NEGATIVE_VERIFIED_EXPLORATORY',
        positive_verified=judgment['positive'], judgments=dict(ETTm1=judgment),
        analysis_helpers_imported_or_used=False, GPU_used=False, scope='Independent bounded cohort/arithmetic audit of all18 facets and all8 matched fits; no model inference or tensor reload',
        scoring_method='Observed cells cast to FP64 per original channel; math.fsum squared/absolute errors and equal-channel mean',
        timing_method='All20 raw timed trials; independently sorted linear quantiles, memory maxima; no pooling across seeds',
        validated_facets=len(facets), unique_pid_creation_pairs=len(pids), warmup_trials=180, timed_trials=360,
        B2_searches=6, B2_fresh_refits=6, Direct_MLP_search_refits=2, training=training, facets=facets,
        terminal=terminal, wave_sha256=sha(wave_path), activation_sha256=sha(activation_path),
        all_searches_sealed_sha256=seal_sha, before_search_plan_sha256=plan_sha,
        source_manifest_sha256=sha(report / 'SOURCE_MANIFEST_TRAIN_003.json'), producer_sources=sources,
        cost_receipt_hashes=receipts, data_sha256=data_sha, prefix_array_hashes=hashes,
        interval_seconds=900, H48_hours=12, L512_hours=128,
        ancestry='ETTm1 and prior ETTh1 share transformer-station ancestry; development cases are not independent domains',
        original_ETTh2_preparation_amendment='Historical provenance only; unrelated to fresh ETTm1 preparation beyond inherited cohort history',
        protected_T_numeric_access=False, whole_raw_source_hash_recomputed=False, independent_TEST_claimed=False,
        all_registered_negative_widths_preserved=True, source_sha256=sha(__file__), numpy_version=np.__version__)


def write_report(path, result):
    path = Path(path)
    content = (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError('Preserve existing independent audit before writing different bytes')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    result = audit(args.repo)
    if not args.check_only:
        write_report(args.repo / 'reports' / NAME / 'INDEPENDENT_AUDIT_WAVE_003.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'result', 'positive_verified', 'validated_facets')}, sort_keys=True))


if __name__ == '__main__':
    main()
