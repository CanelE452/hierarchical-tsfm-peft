"""Saved-V-array metrics, preregistered decisions, and publication artifacts."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np


DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
PRIMARY_DATASETS = ('jena', 'ETTh1', 'appliances')
METHODS = ('RAW', 'C1', 'C2', 'C3', 'OURS', 'A1', 'A2', 'A3', 'A4', 'A5')
LABELS = dict(RAW='Raw', C1='AdaPTS-PCA', C2='AdaPTS-LinearAE', C3='AdaPTS-dropoutLAE',
              OURS='Ours', A1='No P', A2='Capacity-matched control', A3='Random-orthogonal E',
              A4='Linear decoder', A5='Decoder without lags')
QUANTILES = (50, 75, 90, 95, 99)
SEEDS = set(range(92601, 92611))
ARRAY_FIELDS = ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids')
GENERATED_FILES = (
    'METRICS_EXISTING.csv', 'ERROR_DISTRIBUTION_EXISTING.csv', 'ERROR_DISTRIBUTION_EXISTING_SUMMARY.csv',
    'VMSE_DEFINITION.md', 'RUNS.csv', 'SUMMARY.csv', 'COMPARATIVE.csv', 'ABLATION.csv',
    'DECISION.json', 'RESULTS_KO.md', 'SLIDE_NUMBERS.md', 'ERROR_DISTRIBUTION.csv',
    'ERROR_DISTRIBUTION_SUMMARY.csv',
    *[f'{name}.{suffix}' for name in ('FIG_COMPARATIVE', 'FIG_ABLATION', 'FIG_ERROR_QUANTILES')
      for suffix in ('csv', 'png', 'svg')],
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, text, *, identical_ok=True):
    path = Path(path)
    if path.exists():
        if identical_ok and path.read_text(encoding='utf8') == text:
            return
        raise FileExistsError('Refusing to overwrite report artifact: ' + str(path))
    with path.open('x', encoding='utf8', newline='') as stream:
        stream.write(text)


def _csv(path, rows, *, identical_ok=True):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    _write(path, stream.getvalue(), identical_ok=identical_ok)


def training_scales(task, fit_end):
    """A60-only observed scaler and observed consecutive-pair MASE scale."""
    physical = np.asarray(task['physical'][:fit_end], np.float64)
    observed = np.asarray(task['observed'][:fit_end], bool)
    if physical.shape != observed.shape or physical.ndim != 2 or len(physical) != fit_end:
        raise ValueError('A60 physical/observed shape or endpoint mismatch')
    counts = observed.sum(0)
    if np.any(counts == 0) or not np.isfinite(physical[observed]).all():
        raise ValueError('A60 scaler requires finite observations in every channel')
    mean = np.where(observed, physical, 0).sum(0) / counts
    std_original = np.sqrt((np.where(observed, physical - mean, 0) ** 2).sum(0) / counts)
    std = std_original.copy()
    std[std == 0] = 1
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    pairs = observed[1:] & observed[:-1]
    pair_count = pairs.sum(0)
    delta = np.abs(physical[1:] - physical[:-1]) / std.astype(np.float64)
    scale = np.divide(np.where(pairs, delta, 0).sum(0), pair_count,
                      out=np.zeros_like(std, dtype=np.float64), where=pair_count > 0)
    return dict(mean=mean, std=std, physical_std=std_original,
                mase_scale=scale, mase_pairs=pair_count, fit_end=int(fit_end))


def validate_arrays(values, reference=None):
    if any(key not in values for key in ARRAY_FIELDS):
        raise ValueError('Prediction archive is missing a required coordinate/array')
    p, y, m = (np.asarray(values[key]) for key in ARRAY_FIELDS[:3])
    origins, channels = np.asarray(values['origins']), np.asarray(values['channel_ids'])
    if (p.ndim != 3 or p.shape != y.shape or p.shape != m.shape or m.dtype != np.bool_
            or origins.shape != (p.shape[0],) or channels.shape != (p.shape[2],)
            or not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all()
            or np.any(m.sum((0, 1)) == 0)):
        raise ValueError('Prediction shape, observed mask, coordinates or finite values differ')
    if reference is not None:
        for key in ('target_normalized', 'observed', 'origins', 'channel_ids'):
            if not np.array_equal(values[key], reference[key]):
                raise ValueError('Raw/family evaluation cohort mismatch: ' + key)


def load_arrays(record, reference=None):
    path = Path(record['prediction_path'])
    if _sha(path) != record['prediction_sha256']:
        raise ValueError('Prediction file SHA changed: ' + str(path))
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in ARRAY_FIELDS}
        for key in ('mean', 'std'):
            if key in archive.files:
                values[key] = archive[key]
    validate_arrays(values, reference)
    return values


def bind_physical_targets(values, task, scales):
    """Bind exact observed V truth before deciding whether a MAPE divisor is zero."""
    validate_arrays(values)
    origins = np.asarray(values['origins'])
    horizon = values['target_normalized'].shape[1]
    start, end = task['split_ranges']['V_select']
    if (not np.issubdtype(origins.dtype, np.integer) or horizon != task['H']
            or np.any(origins < start) or np.any(origins + horizon > end)
            or not np.array_equal(values['channel_ids'], np.asarray(task['columns']))):
        raise ValueError('Physical target binding is outside registered V coordinates')
    indices = origins[:, None] + np.arange(horizon)
    physical = np.asarray(task['physical'], np.float32)[indices]
    observed = np.asarray(task['observed'], bool)[indices]
    if not np.array_equal(observed, values['observed']):
        raise ValueError('Physical target observed mask differs from saved V mask')
    mean, std = np.asarray(scales['mean'], np.float32), np.asarray(scales['std'], np.float32)
    normalized = np.where(observed, (physical - mean) / std, 0).astype(np.float32)
    if not np.array_equal(normalized[observed], values['target_normalized'][observed]):
        raise ValueError('Exact physical V truth does not reproduce saved FP32 normalized targets')
    return dict(values, target_physical=physical)


def _finite_mean(values):
    values = np.asarray(values, np.float64)
    active = np.isfinite(values)
    return float(values[active].mean()) if active.any() else None


def metrics(values, scales, *, expected_mse=None, expected_mae=None):
    validate_arrays(values)
    p, y = (np.asarray(values[key], np.float64) for key in ARRAY_FIELDS[:2])
    mask = values['observed']
    count = mask.sum((0, 1))
    error = np.where(mask, p - y, 0)
    sse, sae = (error ** 2).sum((0, 1)), np.abs(error).sum((0, 1))
    mse, mae = float((sse / count).mean()), float((sae / count).mean())
    if expected_mse is not None and abs(mse - float(expected_mse)) > 1e-9:
        raise ValueError('Saved-array MSE differs from receipt by more than 1e-9')
    if expected_mae is not None and abs(mae - float(expected_mae)) > 1e-9:
        raise ValueError('Saved-array MAE differs from receipt by more than 1e-9')
    vmean = np.where(mask, y, 0).sum((0, 1)) / count
    sst = (np.where(mask, y - vmean, 0) ** 2).sum((0, 1))
    r2_active = sst >= 1e-12
    r2 = _finite_mean(1 - sse[r2_active] / sst[r2_active])
    denom = np.asarray(scales['mase_scale'], np.float64)
    mase_active = (np.asarray(scales['mase_pairs']) > 0) & (denom > 0)
    mase_valid = _finite_mean((sae / count)[mase_active] / denom[mase_active])
    physical_mean, physical_std = np.asarray(scales['mean'], np.float64), np.asarray(scales['std'], np.float64)
    if 'target_physical' in values:
        physical_target = np.asarray(values['target_physical'], np.float64)
        if physical_target.shape != y.shape or not np.isfinite(physical_target[mask]).all():
            raise ValueError('Bound physical target shape or observed values differ')
    elif np.all(physical_mean == 0) and np.all(physical_std == 1):
        physical_target = y
    else:
        raise ValueError('Exact physical V targets are required for nontrivial normalization')
    physical_prediction = p * physical_std + physical_mean
    physical_error = np.where(mask, np.abs(physical_prediction - physical_target), 0)
    nonzero = mask & (physical_target != 0)
    nearzero = mask & (np.abs(physical_target) < .1 * np.asarray(scales['physical_std']))
    far = nonzero & ~nearzero

    def mape(selection):
        n = selection.sum((0, 1))
        terms = np.divide(physical_error, np.abs(physical_target),
                          out=np.zeros_like(physical_error), where=selection)
        per_channel = np.divide(100 * terms.sum((0, 1)), n,
                                out=np.full(len(n), np.nan), where=n > 0)
        return _finite_mean(per_channel), int((n == 0).sum())

    mape_nonzero, empty_nonzero = mape(nonzero)
    mape_far, empty_far = mape(far)
    return dict(MSE=mse, MAE=mae, R2=r2, R2_excluded_channels=int((~r2_active).sum()),
        MASE=mase_valid if mase_active.all() else None, MASE_valid_channel_mean=mase_valid,
        MASE_undefined_channels=int((~mase_active).sum()),
        MASE_no_adjacent_pair_channels=int((np.asarray(scales['mase_pairs']) == 0).sum()),
        MAPE_nonzero_pct=mape_nonzero, MAPE_far_from_zero_pct=mape_far,
        MAPE_nonzero_empty_channels=empty_nonzero, MAPE_far_empty_channels=empty_far,
        nearzero_fraction=float((nearzero.sum((0, 1)) / count).mean()),
        zero_fraction=float(((mask & (physical_target == 0)).sum((0, 1)) / count).mean()),
        observed_points=int(mask.sum()), channels=int(len(count)))


def error_distribution(values, raw):
    validate_arrays(values, raw)
    mask = values['observed']
    absolute = np.abs(values['pred_normalized'].astype(np.float64) - values['target_normalized'].astype(np.float64))[mask]
    raw_error = np.abs(raw['pred_normalized'].astype(np.float64) - raw['target_normalized'].astype(np.float64))[mask]
    n = len(raw_error)
    top_count = max(1, math.ceil(.05 * n))
    selected = np.argsort(-raw_error, kind='stable')[:top_count]
    raw_sse = float(np.square(raw_error).sum())
    raw_top_sse = float(np.square(raw_error[selected]).sum())
    result = {f'abs_error_q{q}': float(value) for q, value in
              zip(QUANTILES, np.percentile(absolute, QUANTILES, method='linear'))}
    result.update(fraction_abs_error_better_than_raw=float((absolute < raw_error).mean()),
        raw_top5_count=top_count, observed_points=n, raw_total_SSE=raw_sse,
        raw_top5_SSE=raw_top_sse, raw_top5_SSE_fraction=raw_top_sse / raw_sse if raw_sse else None,
        family_SSE_on_raw_top5=float(np.square(absolute[selected]).sum()),
        raw_top5_cutoff_abs_error=float(raw_error[selected[-1]]),
        raw_top5_tie_rule='stable flattened observed-point order; ceil(0.05*n) points')
    return result


def parameter_count(method, channels, k):
    c = channels
    ours = 33344 + 65 * c + 8 * c * k
    return dict(RAW=0, C1=0, C2=2*c*k+k+c, C3=2*c*k+k+c, OURS=ours,
        A1=8*c*k, A2=ours-33312+(2*c+1)*((33312-c)//(2*c+1))+c,
        A3=ours, A4=ours-6*c*k+c, A5=ours-6*c*k)[method]


def _prepare_existing(audit_path, task_loader):
    audit = _json(audit_path)
    if audit.get('failures') or audit['count_ours'] != 40 or audit['count_raw'] != 4:
        raise ValueError('Existing-array audit has not passed its numerical checks')
    duration_rows = {}
    source_runs = Path(audit['source_worktree']) / 'reports/k_tau_rule_20261023_v1/RUNS.csv'
    with source_runs.open(encoding='utf8', newline='') as stream:
        duration_rows = {row['job_id']: row for row in csv.DictReader(stream)}
    tasks, scales, raw = {}, {}, {}
    for dataset in DATASETS:
        task = task_loader(dataset)
        endpoint = task['split_ranges']['A_stop'][1]
        tasks[dataset], scales[dataset] = task, training_scales(task, endpoint)
        record = next(row for row in audit['runs'] if row['dataset'] == dataset and row['method_id'] == 'RAW')
        raw[dataset] = load_arrays(record)
        bounds = task['split_ranges']['V_select']
        if (raw[dataset]['pred_normalized'].shape[1:] != (task['H'], len(task['columns']))
                or np.any(raw[dataset]['origins'] < bounds[0])
                or np.any(raw[dataset]['origins'] + task['H'] > bounds[1])):
            raise ValueError('Raw prediction coordinates are outside registered V')
        raw[dataset] = bind_physical_targets(raw[dataset], task, scales[dataset])
    rows, errors = [], []
    for record in audit['runs']:
        d, method = record['dataset'], record['method_id']
        values = bind_physical_targets(load_arrays(record, raw[d]), tasks[d], scales[d])
        row = dict(dataset=d, method_id=method, display_name=LABELS[method], K=record.get('K'),
            seed=int(record['seed']), job_id=record['job_id'], reused=True, extended=False,
            selected_epoch=record.get('selected_epoch'), training_cap=3 if method == 'OURS' else 0,
            cap_selected=method == 'OURS' and record.get('selected_epoch') == 3,
            trainable_parameters=parameter_count(method, len(tasks[d]['columns']), int(record.get('K') or len(tasks[d]['columns']))),
            normalization_fit_end=scales[d]['fit_end'], prediction_path=record['prediction_path'],
            prediction_sha256=record['prediction_sha256'], status='COMPLETE')
        times = duration_rows.get(record['job_id'], {})
        for key in ('search_seconds', 'refit_seconds'):
            row[key] = float(times[key]) if times.get(key) else None
        row.update(metrics(values, scales[d], expected_mse=record['mse_recomputed'], expected_mae=record['mae_recomputed']))
        rows.append(row)
        errors.append(dict(dataset=d, method_id=method, seed=row['seed'], job_id=row['job_id'],
                           extended=False, **error_distribution(values, raw[d])))
    return rows, errors, scales, raw, tasks


def _definition():
    return ('# V MSE definition\n\n'
        'V MSE is the mean across original channels of squared forecast errors over observed validation targets, '
        'measured after normalization using statistics fitted only on the A60 training interval.\n\n'
        '[확인] 근거: research/informs_broad_validation_20261015_v1/evaluation.py:39–61; '
        'data_contract.py:61–76; fit_base.py:46–49. V는 개발 검증 구간이며 보호 테스트 T가 아니다.\n')


def _defaults(existing_audit_path, task_loader):
    from . import runtime
    return (Path(existing_audit_path) if existing_audit_path is not None else runtime.REPORT / 'G1_REUSE_AUDIT.json',
            task_loader or runtime.load_task)


def report_existing(output_dir, *, existing_audit_path=None, task_loader=None):
    audit_path, loader = _defaults(existing_audit_path, task_loader)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, _, _, _ = _prepare_existing(audit_path, loader)
    _csv(output / 'METRICS_EXISTING.csv', rows)
    _csv(output / 'ERROR_DISTRIBUTION_EXISTING.csv', errors)
    _csv(output / 'ERROR_DISTRIBUTION_EXISTING_SUMMARY.csv', summarize_errors(errors))
    _write(output / 'VMSE_DEFINITION.md', _definition())
    return dict(metrics=rows, error_distribution=errors, Q3=describe_quantiles(errors))


def summarize_errors(errors):
    rows = []
    for d in DATASETS:
        for method in METHODS:
            group = [r for r in errors if r['dataset'] == d and r['method_id'] == method and r.get('is_primary', True)]
            if not group:
                continue
            row = dict(dataset=d, method_id=method, n=len(group), aggregation='seed median')
            names = [f'abs_error_q{q}' for q in QUANTILES] + [
                'fraction_abs_error_better_than_raw', 'raw_top5_count', 'observed_points',
                'raw_total_SSE', 'raw_top5_SSE', 'raw_top5_SSE_fraction', 'family_SSE_on_raw_top5']
            for name in names:
                values = [r[name] for r in group if r.get(name) is not None]
                row[name + '_median'] = float(np.median(values)) if values else None
            rows.append(row)
    return rows


def choose_primary(rows):
    chosen = []
    for d in DATASETS:
        for method in METHODS:
            group = [r for r in rows if r['dataset'] == d and r['method_id'] == method]
            extension = [r for r in group if r.get('extended', False)]
            expected = {92601} if method in ('C1', 'RAW') else SEEDS
            extension_complete = {r['seed'] for r in extension} == expected
            base = [r for r in group if not r.get('extended', False)]
            if any(len(part) != len({r['seed'] for r in part}) for part in (base, extension)):
                raise ValueError('Duplicate seed in a method/dataset/stage group')
            selected = extension if extension_complete else base
            seen = [r['seed'] for r in selected]
            if len(seen) != len(set(seen)):
                raise ValueError('Duplicate seed in a primary method/dataset group')
            required = method not in ('RAW', 'C1') and len(base) == 10 and sum(bool(r['cap_selected']) for r in base) >= 5
            complete = set(seen) == expected
            for row in group:
                row['is_primary'] = row in selected
            values = np.asarray([r['MSE'] for r in selected], np.float64)
            q1, median, q3 = np.percentile(values, [25, 50, 75], method='linear') if len(values) else (None, None, None)
            summary = dict(dataset=d, method_id=method, display_name=LABELS[method], n=len(selected),
                expected_n=len(expected), complete=complete, MSE_median=None if median is None else float(median),
                MSE_mean=float(values.mean()) if len(values) else None,
                MSE_Q1=None if q1 is None else float(q1), MSE_Q3=None if q3 is None else float(q3),
                MSE_IQR=None if q1 is None else float(q3-q1), MSE_min=float(values.min()) if len(values) else None,
                MSE_max=float(values.max()) if len(values) else None,
                cap_selected_count=sum(bool(r['cap_selected']) for r in selected),
                extended=extension_complete, extension_n=len(extension), extension_required=required,
                provisional=required and not extension_complete,
                undertrained_at_10=extension_complete and sum(bool(r['cap_selected']) for r in selected) >= 5)
            for metric in ('MAE', 'MASE', 'R2', 'MAPE_nonzero_pct', 'MAPE_far_from_zero_pct', 'nearzero_fraction'):
                valid = [r[metric] for r in selected if r.get(metric) is not None]
                summary[metric + '_median'] = float(np.median(valid)) if valid else None
                summary[metric + '_valid_runs'] = len(valid)
            chosen.append(summary)
    raw_lookup = {r['dataset']: r['MSE_median'] for r in chosen if r['method_id'] == 'RAW'}
    for row in chosen:
        row['Raw_MSE'] = raw_lookup[row['dataset']]
    return chosen


def decisions(rows, summaries):
    from scipy.stats import mannwhitneyu
    lookup = {(r['dataset'], r['method_id']): r for r in summaries}
    samples = {(d, m): [r['MSE'] for r in rows if r['dataset'] == d and r['method_id'] == m and r['is_primary']]
               for d in DATASETS for m in METHODS}
    comparative, ablation = [], []
    for d in DATASETS:
        ours = lookup[d, 'OURS']
        competitors = [lookup[d, method] for method in ('C1', 'C2', 'C3')]
        complete = ours['complete'] and all(c['complete'] for c in competitors)
        item = dict(dataset=d, in_primary_aggregate=d in PRIMARY_DATASETS, complete=complete,
                    best_comparator=None, ours_median=ours['MSE_median'], best_median=None,
                    gap=None, spread=None, decision='INCOMPLETE', provisional=False,
                    best_selected_on_V=True)
        if complete:
            best = min(competitors, key=lambda r: r['MSE_median'])
            gap, spread = best['MSE_median'] - ours['MSE_median'], max(best['MSE_IQR'], ours['MSE_IQR'])
            conclusion = 'WIN' if gap > spread else 'LOSS' if gap < -spread else 'TIE'
            provisional = any(c['provisional'] for c in competitors) or ours['provisional']
            item.update(best_comparator=best['method_id'], best_median=best['MSE_median'], gap=gap,
                        spread=spread, decision=conclusion, provisional=provisional,
                        decision_label=conclusion + ('_PROVISIONAL' if provisional else ''))
        for m in ('C1', 'C2', 'C3'):
            a, b = samples[d, 'OURS'], samples[d, m]
            item['MWU_two_sided_p_' + m] = float(mannwhitneyu(a, b, alternative='two-sided').pvalue) if a and b else None
        comparative.append(item)
        for method in ('A1', 'A2', 'A3', 'A4', 'A5'):
            variant = lookup[d, method]
            complete = ours['complete'] and variant['complete']
            effect = variant['MSE_median'] - ours['MSE_median'] if complete else None
            spread = max(variant['MSE_IQR'], ours['MSE_IQR']) if complete else None
            conclusion = ('CONTRIBUTES' if effect > spread else 'HARMS' if effect < -spread else 'UNCLEAR') if complete else 'INCOMPLETE'
            ablation.append(dict(dataset=d, method_id=method, n=variant['n'], complete=complete,
                effect=effect, spread=spread, decision=conclusion,
                provisional=variant['provisional'] or ours['provisional'],
                in_primary_aggregate=d in PRIMARY_DATASETS))
    primary = [r for r in comparative if r['in_primary_aggregate']]
    if not all(r['complete'] for r in primary):
        overall = 'INCOMPLETE'
    else:
        wins = sum(r['decision'] == 'WIN' for r in primary)
        losses = sum(r['decision'] == 'LOSS' for r in primary)
        overall = 'SUPPORTED' if wins >= 2 and losses == 0 else 'NOT_SUPPORTED' if wins == 0 else 'MIXED'
        if any(r['provisional'] for r in primary):
            overall += '_PROVISIONAL'
    counts = {method: {conclusion: sum(r['method_id'] == method and r['in_primary_aggregate'] and r['decision'] == conclusion
                                      for r in ablation)
                       for conclusion in ('CONTRIBUTES', 'HARMS', 'UNCLEAR', 'INCOMPLETE')}
              for method in ('A1', 'A2', 'A3', 'A4', 'A5')}
    return dict(comparative_overall=overall, primary_datasets=list(PRIMARY_DATASETS),
        comparative=comparative, ablation=ablation, ablation_primary_counts=counts,
        comparator_selection='Minimum C1-C3 median selected on V; favors comparators',
        MWU_role='Descriptive two-sided p values; no inferential decision or multiplicity claim')


def describe_quantiles(errors):
    descriptions = []
    for d in DATASETS:
        raw = [r for r in errors if r['dataset'] == d and r['method_id'] == 'RAW']
        ours = [r for r in errors if r['dataset'] == d and r['method_id'] == 'OURS' and r.get('is_primary', True)]
        if not raw or not ours:
            descriptions.append(dict(dataset=d, status='INCOMPLETE'))
            continue
        values = {q: float(np.median([r[f'abs_error_q{q}'] for r in ours])) for q in QUANTILES}
        comparisons = {f'ours_q{q}_below_raw': values[q] < raw[0][f'abs_error_q{q}'] for q in (90, 95, 99)}
        comparisons['ours_q50_above_raw'] = values[50] > raw[0]['abs_error_q50']
        descriptions.append(dict(dataset=d, n_ours=len(ours), status='DESCRIPTIVE_ONLY',
            sentence_matches=all(comparisons.values()), **comparisons,
            ours_quantiles={str(q): values[q] for q in QUANTILES},
            raw_quantiles={str(q): raw[0][f'abs_error_q{q}'] for q in QUANTILES}))
    return descriptions


def _figures(output, rows, errors):
    from . import runtime
    os.environ['MPLCONFIGDIR'] = str(runtime.CACHE / 'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'comparative_ablation_20261025_v1'
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['is_primary']]
    for name, methods in (('FIG_COMPARATIVE', ('C1', 'C2', 'C3', 'OURS')),
                          ('FIG_ABLATION', ('OURS', 'A1', 'A2', 'A3', 'A4', 'A5'))):
        long_rows = [{k:r[k] for k in ('dataset', 'method_id', 'display_name', 'seed', 'job_id', 'MSE', 'extended')}
                     for r in selected if r['method_id'] in (*methods, 'RAW')]
        _csv(output / (name + '.csv'), long_rows)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
        for ax, d in zip(axes.flat, DATASETS):
            labels = []
            for i, method in enumerate(methods):
                values = [r['MSE'] for r in selected if r['dataset'] == d and r['method_id'] == method]
                labels.append(f'{method}\nn={len(values)}')
                if len(values) > 1:
                    ax.boxplot([values], positions=[i], widths=.5, manage_ticks=False,
                               patch_artist=True, boxprops=dict(facecolor='#c8dced'), showfliers=False)
                if values:
                    ax.scatter(i + np.linspace(-.08, .08, len(values)), values, s=17, color='#214d72', zorder=3)
            raw = next((r['MSE'] for r in selected if r['dataset'] == d and r['method_id'] == 'RAW'), None)
            if raw is not None:
                ax.axhline(raw, linestyle='--', color='#555555', linewidth=1.2, label='Raw')
            ax.set_xticks(range(len(methods)), labels)
            ax.set_title(d)
            ax.set_ylabel('V MSE')
            ax.grid(axis='y', alpha=.2)
            ax.legend(loc='best', fontsize=8)
        _save_figure(fig, output, name)
        plt.close(fig)
    long_rows = [dict(dataset=r['dataset'], method_id=r['method_id'], seed=r['seed'],
                      job_id=r['job_id'], percentile=q, absolute_error=r[f'abs_error_q{q}'])
                 for r in errors if r['method_id'] in ('RAW', 'OURS') and r['is_primary'] for q in QUANTILES]
    _csv(output / 'FIG_ERROR_QUANTILES.csv', long_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, d in zip(axes.flat, DATASETS):
        for method, color in (('RAW', '#777777'), ('OURS', '#214d72')):
            values = [float(np.median([r['absolute_error'] for r in long_rows
                                      if r['dataset'] == d and r['method_id'] == method and r['percentile'] == q]))
                      for q in QUANTILES]
            ax.plot(QUANTILES, values, '-o', label=LABELS[method], color=color)
        ax.set_title(d)
        ax.set_xlabel('Absolute error percentile')
        ax.set_ylabel('Absolute error (normalized units)')
        ax.set_xticks(QUANTILES)
        ax.grid(alpha=.2)
        ax.legend()
    _save_figure(fig, output, 'FIG_ERROR_QUANTILES')
    plt.close(fig)


def _save_figure(figure, output, name):
    for suffix in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Date': None} if suffix == 'svg' else {'Software': 'comparative_ablation_20261025_v1'}
        figure.savefig(buffer, format=suffix, dpi=180, metadata=metadata)
        payload = buffer.getvalue()
        path = output / f'{name}.{suffix}'
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError('Existing figure differs from deterministic replay: ' + str(path))
        else:
            with path.open('xb') as stream:
                stream.write(payload)


def report(output_dir, new_results, status, *, existing_audit_path=None, task_loader=None):
    audit_path, loader = _defaults(existing_audit_path, task_loader)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, scales, raw, tasks = _prepare_existing(audit_path, loader)
    _csv(output / 'METRICS_EXISTING.csv', rows, identical_ok=True)
    _csv(output / 'ERROR_DISTRIBUTION_EXISTING.csv', errors, identical_ok=True)
    _csv(output / 'ERROR_DISTRIBUTION_EXISTING_SUMMARY.csv', summarize_errors(errors), identical_ok=True)
    _write(output / 'VMSE_DEFINITION.md', _definition(), identical_ok=True)
    for result in sorted(new_results, key=lambda r: str(r.get('job', {}).get('job_id', r.get('row', {}).get('job_id', '')))):
        if result.get('status') != 'COMPLETE':
            continue
        record = result['row']
        d, method = record['dataset'], record['method_id']
        if d not in DATASETS or method not in METHODS or method in ('RAW', 'OURS'):
            raise ValueError('New result is outside registered comparison/ablation methods')
        if record['seed'] not in SEEDS or (method == 'C1' and record['seed'] != 92601):
            raise ValueError('Result seed differs from the preregistered seed set')
        from .runtime import KS
        if (record['K'] != KS[d] or record['trainable_parameters'] !=
                parameter_count(method, len(tasks[d]['columns']), KS[d])):
            raise ValueError('Result K or parameter count differs from registered architecture')
        values = load_arrays(record, raw[d])
        for key in ('mean', 'std'):
            if key in values and not np.array_equal(values[key], scales[d][key]):
                raise ValueError('Saved prediction scaler differs from registered A60 scaler')
        if record['normalization_fit_end'] != scales[d]['fit_end']:
            raise ValueError('Result normalization endpoint changed')
        values = bind_physical_targets(values, tasks[d], scales[d])
        row = dict(record, display_name=LABELS[method], status='COMPLETE')
        row.update(metrics(values, scales[d], expected_mse=record['MSE'], expected_mae=record['MAE']))
        diagnostic = result.get('deterministic_diagnostic')
        if diagnostic:
            diag_values = bind_physical_targets(load_arrays(diagnostic, raw[d]), tasks[d], scales[d])
            diag = metrics(diag_values, scales[d],
                           expected_mse=diagnostic['MSE'], expected_mae=diagnostic['MAE'])
            row.update(deterministic_diagnostic_MSE=diag['MSE'], deterministic_diagnostic_MAE=diag['MAE'])
        rows.append(row)
        errors.append(dict(dataset=d, method_id=method, seed=int(row['seed']), job_id=row['job_id'],
                           extended=bool(row['extended']), **error_distribution(values, raw[d])))
    summaries = choose_primary(rows)
    active = {r['job_id']: r['is_primary'] for r in rows}
    for row in errors:
        row['is_primary'] = active[row['job_id']]
    decision = decisions(rows, summaries)
    decision.update(Q3=describe_quantiles(errors), error_summary=summarize_errors(errors), execution_status=status,
                    source_audit=str(audit_path), T_numeric_values_read=0)
    _csv(output / 'RUNS.csv', rows)
    _csv(output / 'SUMMARY.csv', summaries)
    _csv(output / 'COMPARATIVE.csv', decision['comparative'])
    _csv(output / 'ABLATION.csv', decision['ablation'])
    _csv(output / 'ERROR_DISTRIBUTION.csv', errors)
    _csv(output / 'ERROR_DISTRIBUTION_SUMMARY.csv', summarize_errors(errors))
    _write(output / 'DECISION.json', json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + '\n')
    _figures(output, rows, errors)
    _write_documents(output, summaries, decision, status)
    return decision


def _fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def _write_documents(output, summaries, decision, status):
    lookup = {(r['dataset'], r['method_id']): r for r in summaries}
    slide = ['# Slide numbers', '', 'V is development validation; no protected T evaluation.', '']
    text = ['# 비교·절제 실험 결과', '', '[확인] 저장된 V 예측을 FP64로 재계산하고 receipt와 1e-9 이내 일치를 확인했다.',
            '', '종합 비교 판정: ' + decision['comparative_overall'], '', '```text',
            'dataset             method   n    median V MSE    IQR          cap hits']
    for row in summaries:
        text.append(f"{row['dataset']:<20}{row['method_id']:<8}{row['n']:<5}{_fmt(row['MSE_median']):<16}{_fmt(row['MSE_IQR']):<13}{row['cap_selected_count']}")
    text.extend(['```', '', '비교 판정과 절제 판정의 상세는 COMPARATIVE.csv, ABLATION.csv에 있다.', '', '오차 분포 설명:'])
    for item in decision['Q3']:
        text.append(f"- {item['dataset']}: 90·95·99% 오차는 Raw보다 낮고 50% 오차는 높다는 문장 = {item.get('sentence_matches', 'INCOMPLETE')}. 기술 통계 설명이며 별도 판정은 아니다.")
    text.extend(['', '비교 판정:', '', '```text', 'dataset             best  gap         spread      decision'])
    for row in decision['comparative']:
        text.append(f"{row['dataset']:<20}{str(row['best_comparator'] or 'NA'):<6}{_fmt(row['gap']):<12}{_fmt(row['spread']):<12}{row.get('decision_label', row['decision'])}")
    text.extend(['```', '', '절제 집계 (jena·ETTh1·appliances):', '', '```text', 'method  CONTRIBUTES  HARMS  UNCLEAR  INCOMPLETE'])
    for method, counts in decision['ablation_primary_counts'].items():
        text.append(f"{method:<8}{counts['CONTRIBUTES']:<13}{counts['HARMS']:<7}{counts['UNCLEAR']:<9}{counts['INCOMPLETE']}")
    text.append('```')
    provisional = [r for r in decision['ablation'] if r['in_primary_aggregate'] and r['provisional']]
    if provisional:
        labels = ', '.join(f"{r['method_id']}/{r['dataset']}={r['decision']}_PROVISIONAL" for r in provisional)
        text.extend(['', '위 절제 집계에는 다음 잠정(PROVISIONAL) 판정이 포함된다: ' + labels + '.',
                     '해당 판정은 필요한 연장이 완료되지 않은 기본 결과에 근거한다. 집계의 CONTRIBUTES 수에 포함되어도 확정된 구성요소 기여로 해석할 수 없다.'])
    text.extend(['', 'Raw 대 OURS 보조 지표 (seed 중앙값):', '', '```text',
                 f"{'dataset':<20} | {'method':<8} | " + ' | '.join(f'{name:<12}' for name in
                     ('MAE', 'R2', 'MASE', 'MAPE(a)%', 'MAPE(c)%', 'nearzero'))])
    for row in summaries:
        if row['method_id'] in ('RAW', 'OURS'):
            values = ' | '.join(f'{_fmt(row[key+"_median"]):<12}' for key in
                               ('MAE', 'R2', 'MASE', 'MAPE_nonzero_pct', 'MAPE_far_from_zero_pct', 'nearzero_fraction'))
            text.append(f"{row['dataset']:<20} | {row['method_id']:<8} | {values}")
    text.extend(['```', '', '전체 family의 보조 지표는 SUMMARY.csv, 오차 분위수와 상위 오차 기여도는 ERROR_DISTRIBUTION_SUMMARY.csv에 있다.',
                 '', 'Raw 대 OURS 절대오차 분위수 (seed 중앙값):', '', '```text',
                 'dataset             method  q50         q90         q95         q99         better fraction'])
    for row in decision['error_summary']:
        if row['method_id'] in ('RAW', 'OURS'):
            values = ''.join(f'{_fmt(row[f"abs_error_q{q}_median"]):<12}' for q in (50, 90, 95, 99))
            text.append(f"{row['dataset']:<20}{row['method_id']:<8}{values}{_fmt(row['fraction_abs_error_better_than_raw_median'])}")
    text.append('```')
    for d in DATASETS:
        ours, raw = lookup[d, 'OURS']['MSE_median'], lookup[d, 'RAW']['MSE_median']
        comp = next(r for r in decision['comparative'] if r['dataset'] == d)
        reduction = 100*(raw-ours)/raw if raw and ours is not None else None
        best = comp['best_median']
        versus = 100*(best-ours)/best if best and ours is not None else None
        medians = ', '.join(f'{m}={_fmt(lookup[d,m]["MSE_median"])}' for m in METHODS if m != 'RAW')
        slide.extend([f'## {d}', '', medians, '', f'Raw={_fmt(raw)}; Ours reduction vs Raw={_fmt(reduction)}%; '
                      f'Ours reduction vs best comparator={_fmt(versus)}%.', '',
                      f'English decision: {comp.get("decision_label", comp["decision"])} under the preregistered median/IQR rule.', ''])
    text.extend(['', '해석 범위와 한계:',
        '- 최선 C1–C3는 V 중앙값으로 골랐다. 비교 대상에 유리한 선택이다.',
        '- C2·C3는 AdaPTS식 어댑터이며 원 논문 재현이 아니다. 차이는 DEVIATIONS_FROM_ADAPTS.md에 있다.',
        '- Mann–Whitney U 양측 p값은 기술 통계로만 보고하며 판정이나 다중비교 유의성 주장에 쓰지 않는다.',
        '- 10 seed가 채워지지 않은 기본 비교·절제 그룹은 INCOMPLETE로 남긴다. 연장 10 seed가 모두 끝나면 연장을 주 결과로 사용한다. 연장이 미완료이면 기본 결과와 PROVISIONAL 표시를 유지하며 모든 실행은 RUNS.csv에 보존한다.',
        '- MASE의 인접 관측쌍이 없거나 척도가 0인 채널이 있으면 전체 MASE는 NA다. 유효 채널만의 평균과 제외 수를 별도 열로 공개한다.',
        '- MASE 출처: [Hyndman & Koehler (2006), Another look at measures of forecast accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). 이 캠페인은 A60의 lag-1 관측쌍으로 척도를 고정한다.',
        '- MAPE의 예측값은 저장된 FP32 정규화 예측과 FP32 학습 scaler를 FP64로 역변환한다. 정답과 0 여부는 같은 origin·horizon·채널·관측 마스크에 결속한 원단위 physical V 정답을 쓴다. 정답을 같은 FP32 scaler로 정규화하면 저장된 관측 target과 정확히 일치해야 한다. 이 방식은 정규화 왕복 반올림이 실제 0을 미세한 비영 값으로 바꾸는 문제를 막으며 epsilon은 쓰지 않는다. MSE·MAE·R2·MASE와 오차 분위수는 기존 정규화 단위를 유지한다.',
        '- 정확한 0과 0 근처 비율 및 채널별 제외 수를 보고하며 0.1 표준편차 기준은 판정에 쓰지 않는다.',
        '- Raw 상위 5%는 관측 점의 절대오차 내림차순에서 ceil(0.05*n)개를 고른다. 동률은 평탄화된 원래 순서를 유지한다.',
        '', '실행 상태와 예산 기록:', '', '```json', json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), '```', '',
        '그림: FIG_COMPARATIVE.png/.svg, FIG_ABLATION.png/.svg, FIG_ERROR_QUANTILES.png/.svg. 각 그림의 같은 이름 CSV에 긴 형식 데이터를 제공한다.', '',
        _definition().split('\n\n', 1)[1]])
    _write(output / 'RESULTS_KO.md', '\n'.join(text) + '\n')
    _write(output / 'SLIDE_NUMBERS.md', '\n'.join(slide) + '\n')
