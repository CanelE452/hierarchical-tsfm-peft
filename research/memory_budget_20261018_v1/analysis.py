"""Deterministic screening, registered gates, and receipt-only public analysis."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TSFM = frozenset({'B2', 'B0', 'RAW', 'LORA'})
GPU_PRACTICAL = TSFM | {'DIRECT_G'}


def schedule(width, B, referenceK):
    if type(B) is not int or B not in (1, 8):
        raise ValueError('Registered batch must be 1 or 8')
    if type(referenceK) is not int or referenceK < 1:
        raise ValueError('Positive registered reference K required')
    if width is None:
        return [None]
    if type(width) is not int or width < 1:
        raise ValueError('Positive integer F width required')
    return list(dict.fromkeys(min(value, B * width) for value in (128, B * referenceK, 1)))


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError('Missing numeric ' + name)
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError('Invalid numeric ' + name)
    return result


def _batch(row):
    value = row.get('B', row.get('batch_size'))
    if type(value) is not int or value not in (1, 8):
        raise ValueError('Registered integer batch required')
    if row.get('batch_size', value) != value:
        raise ValueError('Batch aliases differ')
    return value


def _original(row):
    value = row.get('is_original', row.get('original', False))
    if type(value) is not bool:
        raise ValueError('Original-path flag must be a strict bool')
    return value


def _chunk(row):
    return row.get('effective_row_microbatch', row.get('row_microbatch'))


def _cost(row):
    return (_number(row['peak_allocated_MiB'], 'allocated memory'),
            _number(row['median_batch_ms'], 'whole-path median', positive=True))


def _dominates(left, right):
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def select_confirmations(screen_rows):
    """Original paths plus within-checkpoint cost frontiers; accuracy is unused."""
    groups = defaultdict(list)
    output = []
    identifiers = set()
    for supplied in screen_rows:
        row = dict(supplied)
        identifier = row['facet_id']
        if identifier in identifiers:
            raise ValueError('duplicate screen facet identity')
        identifiers.add(identifier)
        row.update(confirm_selected=False, confirmation_reason='SCREENED_DOMINATED_COST')
        output.append(row)
        if row.get('parity_status') != 'PASS' or row.get('status') == 'INVALID':
            row['confirmation_reason'] = 'INVALID_PARITY'
            continue
        if row.get('stage') != 'screen':
            raise ValueError('Confirmation selection requires screen rows only')
        if _original(row):
            row.update(confirm_selected=True, confirmation_reason='ORIGINAL_PATH_REQUIRED')
        key = (row['dataset'], row['accuracy_run_id'], _batch(row), row.get('device', 'cuda'))
        groups[key].append(row)
    for rows in groups.values():
        for row in rows:
            if _original(row):
                continue
            if _chunk(row) == 1 and 30 * _number(row['median_batch_ms'], 'screen median', positive=True) / 1000 > 120:
                row['confirmation_reason'] = 'SCREENED_COST_EXCESS'
                continue
            cost = _cost(row)
            if not any(_dominates(_cost(other), cost) for other in rows if other is not row):
                row.update(confirm_selected=True, confirmation_reason='SCREEN_COST_FRONTIER')
    return output


def _confirmed(row):
    return (row.get('parity_status') == 'PASS'
            and row.get('stage') == 'confirm'
            and row.get('status') in ('COMPLETE', 'CONFIRMED'))


def pareto_rows(summary_rows):
    """Strict three-axis fronts, with CPU and GPU inference scopes kept separate."""
    grouped = defaultdict(list)
    identifiers = set()
    for supplied in summary_rows:
        if not _confirmed(supplied):
            continue
        row = dict(supplied)
        identifier = row['facet_id']
        if identifier in identifiers:
            raise ValueError('duplicate confirmed facet identity')
        identifiers.add(identifier)
        device = row.get('device', 'cuda')
        layer = row.get('memory_layer')
        if not layer:
            raise ValueError('Explicit memory layer required')
        if device == 'cpu':
            scopes = ['CPU_CONTEXT']
            memory = _number(row['CPU_RSS_sampled_peak_MiB'], 'CPU sampled RSS')
        elif device == 'cuda':
            scopes = (['TSFM'] if row['family'] in TSFM else []) + (
                ['GPU_PRACTICAL'] if row['family'] in GPU_PRACTICAL else [])
            memory = _number(row['peak_allocated_MiB'], 'allocated memory')
        else:
            raise ValueError('Unregistered inference device')
        metrics = (_number(row['MSE'], 'MSE'), memory,
                   _number(row['median_batch_ms'], 'whole-path median', positive=True))
        for scope in scopes:
            grouped[(scope, row['dataset'], _batch(row), device, layer)].append((row, metrics))
    result = []
    for key, values in sorted(grouped.items()):
        for row, metrics in values:
            dominators = sorted(other['facet_id'] for other, cost in values
                                if _dominates(cost, metrics))
            result.append(dict(row, comparison_scope=key[0], dominated=bool(dominators),
                               frontier=not dominators, dominating_facet_ids=dominators))
    return sorted(result, key=lambda value: (value['comparison_scope'], value['dataset'],
                                            _batch(value), value['facet_id']))


def phase_b_decision(rows, *, remaining_seconds, estimated_training_seconds,
                     estimated_measurement_seconds, reserve_seconds=1800):
    remaining = _number(remaining_seconds, 'remaining seconds')
    training = _number(estimated_training_seconds, 'full four-model training estimate')
    measurement = _number(estimated_measurement_seconds, 'full new-facet measurement estimate')
    reserve = _number(reserve_seconds, 'report and publication reserve')
    front = [row for row in pareto_rows(rows) if row['comparison_scope'] == 'TSFM'
             and row['dataset'] == 'jena' and row['family'] == 'B2' and row['frontier']]
    evidence = []
    for frontier in front:
        cohort = [row for row in rows if _confirmed(row) and row['dataset'] == 'jena'
                  and row['family'] == 'B2'
                  and row['accuracy_run_id'] == frontier['accuracy_run_id']
                  and row.get('checkpoint_sha256') == frontier.get('checkpoint_sha256')
                  and _batch(row) == _batch(frontier)]
        for low in cohort:
            for high in cohort:
                if _chunk(low) == _chunk(high):
                    continue
                maximum = low.get('timed_peak_allocated_max_MiB')
                minimum = high.get('timed_peak_allocated_min_MiB')
                if maximum is None or minimum is None:
                    continue
                if _number(maximum, 'low-memory trial maximum') < _number(minimum, 'high-memory trial minimum'):
                    evidence.append(dict(accuracy_run_id=frontier['accuracy_run_id'], B=_batch(frontier),
                        lower_facet_id=low['facet_id'], higher_facet_id=high['facet_id'],
                        lower_trial_max_MiB=float(maximum), higher_trial_min_MiB=float(minimum)))
    reason = ('NO_JENA_B2_TSFM_FRONTIER' if not front else
              'NO_REPEATED_TRIAL_MEMORY_SEPARATION' if not evidence else
              'INSUFFICIENT_FULL_PHASE_B_BUDGET' if remaining < training + measurement + reserve else
              'REGISTERED_GATE_PASSED')
    return dict(status='PROCEED' if reason == 'REGISTERED_GATE_PASSED' else 'SKIPPED_GATE',
        reason=reason, jena_b2_tsfm_frontier_facet_ids=sorted(row['facet_id'] for row in front),
        memory_separation_evidence=sorted(evidence, key=lambda value: (value['accuracy_run_id'], value['B'], value['lower_facet_id'], value['higher_facet_id'])),
        remaining_seconds=remaining, estimated_full_four_model_training_seconds=training,
        estimated_full_measurement_seconds=measurement, reporting_reserve_seconds=reserve,
        required_remaining_seconds=training + measurement + reserve,
        new_training_settings=4, new_measurement_facets=20,
        independent_test=False, accuracy_metric_used_for_confirmation_selection=False)


def _cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) if isinstance(value, (dict, list, tuple)) else value


def _csv(path, rows, fallback=('facet_id', 'status')):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else list(fallback)
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def _metrics(value):
    fields = ('channel_sse', 'channel_sae', 'channel_counts')
    if not any(key in value for key in fields):
        return dict(MSE=_number(value['MSE'], 'saved MSE'),
                    MAE=_number(value['MAE'], 'saved MAE'), source='SAVED_FP64_SCORE')
    if not all(key in value for key in fields):
        raise ValueError('Incomplete channel sufficient statistics')
    sse, sae, counts = (np.asarray(value[key], dtype=np.float64) for key in fields)
    if (sse.ndim != 1 or not len(sse) or sse.shape != sae.shape or sse.shape != counts.shape
            or not all(np.isfinite(array).all() for array in (sse, sae, counts))
            or any(np.any(array < 0) for array in (sse, sae, counts))
            or not np.array_equal(counts, counts.astype(np.int64))):
        raise ValueError('Invalid channel sufficient statistics')
    active = counts > 0
    if not active.any():
        raise ValueError('No observed target channels')
    if np.any(sse[~active] != 0) or np.any(sae[~active] != 0):
        raise ValueError('Unobserved channels cannot contribute error')
    result = dict(MSE=float(np.mean(sse[active] / counts[active])),
                  MAE=float(np.mean(sae[active] / counts[active])),
                  source='INDEPENDENT_FP64_CHANNEL_SUFFICIENT_STATISTICS')
    for key in ('MSE', 'MAE'):
        if not math.isclose(result[key], float(value[key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Saved score contradicts channel sufficient statistics: ' + key)
    return result


def _summary(receipt):
    row = dict(receipt.get('summary', {}))
    row.setdefault('facet_id', receipt['facet_id'])
    if row['facet_id'] != receipt['facet_id']:
        raise ValueError('Receipt and summary facet identities differ')
    row['status'] = receipt['status']
    row.setdefault('device', 'cuda')
    row.setdefault('memory_layer', 'CUDA_ALLOCATED_TENSORS' if row['device'] == 'cuda' else 'CPU_RSS_SAMPLED')
    process = receipt.get('process', {})
    row['process_pid'] = process.get('pid', row.get('process_pid', row.get('pid')))
    row['process_create_time'] = process.get('create_time', row.get('process_create_time'))
    row['process_wall_seconds'] = process.get('wall_seconds')
    if receipt['status'] != 'COMPLETE':
        row['reason'] = receipt.get('failure') or receipt.get('reason') or 'Incomplete measurement'
        return row
    if receipt.get('parity', {}).get('status') != 'PASS' or row.get('parity_status') != 'PASS':
        raise ValueError('Complete receipt must have full-V parity PASS')
    trials = receipt.get('trials', [])
    warm = [trial for trial in trials if trial['phase'] == 'warmup']
    timed = [trial for trial in trials if trial['phase'] == 'timed']
    if len(warm) != row['warmups'] or len(timed) != row['timed_trials']:
        raise ValueError('Actual repeat counts differ from receipt')
    expected = {'screen': (2, 3), 'confirm': (10, 20)}
    if row['stage'] not in expected or (len(warm), len(timed)) != expected[row['stage']]:
        raise ValueError('Actual repeats differ from registered screen/confirm protocol')
    for trial in trials:
        if trial.get('pid') != row['process_pid']:
            raise ValueError('Trial PID differs from isolated worker PID')
    times = np.asarray([_number(trial['whole_seconds'], 'trial time', positive=True) for trial in timed], np.float64)
    batch = _batch(row)
    q25, median, q75, p95 = np.percentile(times, [25, 50, 75, 95]) * 1000
    row.update(median_batch_ms=float(median), IQR_batch_ms=float(q75 - q25),
        p95_batch_ms=float(p95), median_ms_per_origin=float(median / batch),
        origins_per_second=float(batch * 1000 / median),
        statistic_source='INDEPENDENT_FP64_RAW_TIMED_TRIAL_RECOMPUTATION')
    if row['device'] == 'cuda':
        allocated = [_number(trial['peak_allocated_MiB'], 'trial allocated peak') for trial in timed]
        reserved = [_number(trial['peak_reserved_MiB'], 'trial reserved peak') for trial in timed]
        if any(a > r for a, r in zip(allocated, reserved)):
            raise ValueError('Reserved allocator memory cannot be below allocated tensors')
        row.update(peak_allocated_MiB=max(allocated), peak_reserved_MiB=max(reserved),
            timed_peak_allocated_min_MiB=min(allocated), timed_peak_allocated_max_MiB=max(allocated))
    else:
        if any(trial.get('peak_allocated_MiB') is not None or trial.get('peak_reserved_MiB') is not None for trial in timed):
            raise ValueError('CPU context cannot claim CUDA allocator measurements')
        row.update(peak_allocated_MiB=None, peak_reserved_MiB=None)
    return row


def export_receipts(receipts, report_root, *, make_figures=True):
    """Rebuild aggregate products from portable receipts, without running a model."""
    loaded = []
    for item in receipts:
        if isinstance(item, (str, Path)):
            content = Path(item).read_bytes()
            value = json.loads(content)
            value['analysis_receipt_sha256'] = hashlib.sha256(content).hexdigest()
        else:
            value = dict(item)
            value['analysis_receipt_sha256'] = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        loaded.append(value)
    identifiers = [(value['facet_id'], value.get('summary', {}).get('stage')) for value in loaded]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate receipt facet identity')
    summaries, trials, accuracy, audits, manifest = [], [], [], [], []
    for receipt in sorted(loaded, key=lambda value: (value['facet_id'], value.get('summary', {}).get('stage', ''))):
        row = _summary(receipt)
        row['receipt_sha256'] = receipt['analysis_receipt_sha256']
        summaries.append(row)
        manifest.append(dict(row, reason=row.get('reason', 'ACTUAL_ISOLATED_PROCESS_MEASUREMENT'),
                             status='CONFIRMED' if _confirmed(row) else
                             'SCREENED' if row['status'] == 'COMPLETE' else row['status']))
        identity = {key: row.get(key) for key in ('facet_id', 'accuracy_run_id', 'dataset', 'family', 'seed', 'B', 'batch_size', 'K', 'effective_row_microbatch', 'checkpoint_sha256', 'stage', 'device', 'memory_layer')}
        for trial in receipt.get('trials', []):
            trials.append(dict(identity, **trial))
        parity = receipt.get('parity', {})
        audits.append(dict(facet_id=row['facet_id'], stage=row.get('stage'), accuracy_run_id=row.get('accuracy_run_id'),
                           receipt_sha256=row['receipt_sha256'], **parity))
        original = parity.get('original_metrics', {})
        independent = parity.get('independent_metrics', {})
        if original and independent:
            original = _metrics(original)
            independent = _metrics(independent)
            check = dict(identity, parity_status=parity.get('status'),
                         original_MSE=original['MSE'], original_MAE=original['MAE'],
                         regenerated_MSE=independent['MSE'], regenerated_MAE=independent['MAE'],
                         delta_MSE=float(independent['MSE']) - float(original['MSE']),
                         delta_MAE=float(independent['MAE']) - float(original['MAE']),
                         metric_accumulation_dtype='float64',
                         original_metric_source=original['source'],
                         regenerated_metric_source=independent['source'],
                         full_V_origins=parity.get('full_V_origins'))
            accuracy.append(check)
    selections = {row['facet_id']: row for row in select_confirmations(
        [row for row in summaries if row.get('stage') == 'screen'])}
    for row in manifest:
        selection = selections.get(row['facet_id'])
        row['confirm_selected'] = selection['confirm_selected'] if selection else None
        row['confirmation_reason'] = (selection['confirmation_reason'] if selection
                                      else 'NOT_AVAILABLE_NO_SCREEN_RECEIPT')
        if row.get('stage') == 'screen' and row['status'] == 'SCREENED':
            row['reason'] = row['confirmation_reason']
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    pareto = pareto_rows(summaries)
    products = [('MEMORY_COST_TRIALS.csv', trials), ('MEMORY_COST_SUMMARY.csv', summaries),
        ('ACCURACY_RECHECK.csv', accuracy), ('RUN_MANIFEST.csv', manifest),
        ('PARETO_TABLE.csv', pareto),
        ('RIDGE_CPU_CONTEXT.csv', [row for row in summaries if row['device'] == 'cpu'])]
    for name, rows in products:
        _csv(root / name, rows)
    _json(root / 'PARITY_AUDIT.json', dict(facets=audits, full_V_required=True,
        metric_deltas_recomputed_from_scores=True, independent_test=False,
        process_GPU_sampled_max_is_hard_cap_proof=False))
    if make_figures:
        _figures(summaries, root)
    generated = [name for name, _ in products] + ['PARITY_AUDIT.json']
    if make_figures:
        generated += [name + extension for name in ('FIG1_memory_mse', 'FIG2_memory_latency')
                      for extension in ('.png', '.svg', '.source.csv')]
        if any(_confirmed(row) and row['dataset'] == 'jena' and row['family'] == 'B2'
               and row['K'] in (3, 12) for row in summaries):
            generated += ['FIG3_k_accuracy_memory' + extension for extension in ('.png', '.svg', '.source.csv')]
    return dict(receipt_count=len(loaded), complete_count=sum(value['status'] == 'COMPLETE' for value in summaries),
                confirmed_count=sum(_confirmed(value) for value in summaries),
                invalid_or_incomplete_count=sum(value['status'] != 'COMPLETE' for value in summaries),
                files=sorted(generated))


def _point_label(row):
    chunk = 'r' + str(_chunk(row)) if _chunk(row) is not None else 'no F'
    if row['family'] in ('B2', 'B0'):
        seed = {92601: 's1', 92602: 's2'}[row['seed']]
        return f"{row['family']} {seed} K{row['K']} {chunk}"
    return row['family'] + ' ' + chunk


def _plot_key_rows(rows, yfield):
    precision = 6 if yfield == 'MSE' else 4
    return [[_point_label(row), f"{row['peak_allocated_MiB']:.2f}",
             f"{row[yfield]:.{precision}f}"] for row in rows]


def _key_table(axis, rows, yfield):
    axis.set_axis_off()
    if not rows:
        return
    table = axis.table(cellText=_plot_key_rows(rows, yfield),
        colLabels=['Point key (r = F row limit)', 'Peak MiB', 'MSE' if yfield == 'MSE' else 'ms / origin'],
        cellLoc='left', colWidths=[.57, .20, .23], loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (index, _), cell in table.get_celld().items():
        cell.set_edgecolor('#e4e4e4')
        cell.set_linewidth(.4)
        if index == 0:
            cell.set_facecolor('#f0f0f0')
            cell.get_text().set_weight('bold')


def _canonical_svg_ids(path):
    import xml.etree.ElementTree as ET
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    root = ET.fromstring(content)
    identifiers = [node.attrib['id'] for node in root.iter() if 'id' in node.attrib]
    for index, clip in enumerate(root.iter('{http://www.w3.org/2000/svg}clipPath')):
        original = clip.attrib['id']
        if identifiers.count(original) != 1:
            raise ValueError('SVG clipping identifier must be unique')
        geometry = [(node.tag, sorted((key, value) for key, value in node.attrib.items()
                                     if key != 'id')) for node in clip.iter()]
        digest = hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()[:12]
        canonical = f'clip_{index:03d}_{digest}'
        if canonical in identifiers and canonical != original:
            raise ValueError('Canonical SVG clipping identifier collision')
        content = content.replace('id="' + original + '"', 'id="' + canonical + '"')
        content = content.replace('url(#' + original + ')', 'url(#' + canonical + ')')
        content = content.replace('href="#' + original + '"', 'href="#' + canonical + '"')
    path.write_text(content, encoding='utf-8', newline='\n')


def _figures(rows, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'svg.hashsalt': 'memory_budget_20261018_v1', 'axes.spines.top': False,
        'axes.spines.right': False})
    valid = [row for row in rows if _confirmed(row) and row['device'] == 'cuda']
    colors = {'B2': '#0072B2', 'B0': '#56B4E9', 'RAW': '#777777',
              'LORA': '#E69F00', 'DIRECT_G': '#009E73'}
    markers = {92601: 'o', 92602: 's'}
    seed_key = 'Adapter seeds: s1=92601 (circle), s2=92602 (square). Baselines: diamond, single historical run.'
    max_rows = max([sum(row['dataset'] == dataset and _batch(row) == batch for row in valid)
                    for dataset in ('jena', 'pems32') for batch in (1, 8)] + [0])
    key_height = max(1.2, .18 * (max_rows + 1))
    for name, yfield, ylabel in [('FIG1_memory_mse', 'MSE', 'Development V macro MSE (lower is better)'),
                                 ('FIG2_memory_latency', 'median_ms_per_origin', 'Whole-path latency (ms / origin)')]:
        fig = plt.figure(figsize=(15, 2 * (3.7 + key_height) + 1), layout='constrained')
        grid = fig.add_gridspec(4, 2, height_ratios=[3.7, key_height, 3.7, key_height])
        for i, dataset in enumerate(('jena', 'pems32')):
            for j, batch in enumerate((1, 8)):
                ax = fig.add_subplot(grid[i * 2, j])
                cohort = [row for row in valid if row['dataset'] == dataset and _batch(row) == batch]
                for family in ('RAW', 'LORA', 'B0', 'B2', 'DIRECT_G'):
                    family_rows = [row for row in cohort if row['family'] == family]
                    for row in family_rows:
                        ax.scatter(row['peak_allocated_MiB'], row[yfield], color=colors[family],
                            marker=markers[row['seed']] if family in ('B0', 'B2') else 'D', s=44, alpha=.85,
                            label=family if row is family_rows[0] else None)
                for reference in (150, 180, 200, 220):
                    ax.axvline(reference, color='#dddddd', linewidth=.65, zorder=0)
                if any(row['family'] in TSFM for row in cohort):
                    ax.axvline(47718016 * 4 / 2**20, color='#999999', linestyle='--', linewidth=.8,
                               label='FP32 F weights: 182.03 MiB')
                if cohort:
                    ax.legend(loc='best', fontsize=7)
                else:
                    ax.text(.5, .5, 'No confirmed valid measurements', transform=ax.transAxes, ha='center')
                ax.set_title(f"{'Jena' if dataset == 'jena' else 'PeMS32'} | B={batch}")
                ax.set_xlabel('Peak allocated CUDA tensor memory (MiB / batch)')
                ax.set_ylabel(ylabel)
                ax.grid(axis='y', alpha=.2)
                _key_table(fig.add_subplot(grid[i * 2 + 1, j]), cohort, yfield)
        fig.suptitle('Confirmed whole-path measurements | reused development V\n'
                     'Light-gray reference lines: 150 / 180 / 200 / 220 MiB (illustrative; no process hard-cap certificate)\n' + seed_key, fontsize=11)
        fig.savefig(root / (name + '.png'), dpi=180, metadata={'Software': 'memory_budget_20261018_v1'})
        fig.savefig(root / (name + '.svg'), metadata={'Date': None, 'Creator': 'memory_budget_20261018_v1'})
        _canonical_svg_ids(root / (name + '.svg'))
        plt.close(fig)
        _csv(root / (name + '.source.csv'), valid)
    if any(row['dataset'] == 'jena' and row['family'] == 'B2' and row['K'] in (3, 12) for row in valid):
        selected = [row for row in valid if row['dataset'] == 'jena' and row['family'] == 'B2']
        key_height = max(1.2, .18 * (max(sum(_batch(row) == batch for row in selected) for batch in (1, 8)) + 1))
        fig = plt.figure(figsize=(15, 4.5 + key_height), layout='constrained')
        grid = fig.add_gridspec(2, 2, height_ratios=[3.7, key_height])
        for index, batch in enumerate((1, 8)):
            ax = fig.add_subplot(grid[0, index])
            for k, color in ((3, '#009E73'), (6, '#0072B2'), (12, '#D55E00')):
                subset = [row for row in selected if row['K'] == k and _batch(row) == batch]
                for row in subset:
                    ax.scatter(row['peak_allocated_MiB'], row['MSE'], color=color,
                               marker=markers.get(row.get('seed'), 'o'), label=f'K={k}' if row is subset[0] else None)
            ax.set(title=f'Jena | B={batch}', xlabel='Peak allocated CUDA tensor memory (MiB / batch)',
                   ylabel='Development V macro MSE')
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            _key_table(fig.add_subplot(grid[1, index]), [row for row in selected if _batch(row) == batch], 'MSE')
        fig.suptitle('Separately trained K-specific checkpoints | no instant K switching\n' + seed_key)
        fig.savefig(root / 'FIG3_k_accuracy_memory.png', dpi=180, metadata={'Software': 'memory_budget_20261018_v1'})
        fig.savefig(root / 'FIG3_k_accuracy_memory.svg', metadata={'Date': None, 'Creator': 'memory_budget_20261018_v1'})
        _canonical_svg_ids(root / 'FIG3_k_accuracy_memory.svg')
        plt.close(fig)
        _csv(root / 'FIG3_k_accuracy_memory.source.csv', selected)
