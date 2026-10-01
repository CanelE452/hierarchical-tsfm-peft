"""Aggregate-only v16 report: no model execution, fitting, or prediction rescoring."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from runtime_v16 import HERE, DATASETS, SEEDS, Job, artifact, digest, read_json, require_storage, save_json


ROLES = ('temporal_lora', 'temporal_f0', 'coarse_only', 'direct', 'a', 'f0', 'full_mse', 'native')
PERIODS = ('test_a', 'test_b', 'combined')
LABELS = {'temporal_lora': 'TEMPORAL_LORA', 'temporal_f0': 'TEMPORAL_F0', 'coarse_only': 'COARSE_ONLY',
          'direct': 'DIRECT', 'a': 'A', 'f0': 'F0', 'full_mse': 'FULL_MSE', 'native': 'NATIVE'}
COLORS = {'temporal_lora': '#0072B2', 'a': '#D55E00', 'f0': '#999999',
          'full_mse': '#009E73', 'native': '#CC79A7', 'direct': '#000000'}
MARKERS = {'full': 'o', 'chunk_4k': '^', 'chunk_k': 's', 'direct': 'D'}
FIGURES = ('figure_accuracy_by_period', 'figure_accuracy_cost')
COST_NUMERIC = ('mean_seed_median_ms', 'mean_seed_median_origins_s', 'ms_low', 'ms_high',
                'origins_s_low', 'origins_s_high', 'allocated_bytes', 'allocated_low_bytes', 'allocated_high_bytes',
                'reserved_bytes', 'reserved_low_bytes', 'reserved_high_bytes', 'deployed_parameters',
                'parameter_bytes', 'buffer_bytes', 'deployment_tensor_bytes',
                'this_campaign_adaptation_registered_parameters', 'historical_adaptation_registered_parameters')


def csv_write(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def metric(evaluation, dataset, family, period, name):
    return evaluation['datasets'][dataset]['role_summary'][family]['periods'][period][name]


def accuracy_values(evaluation):
    rows, comparisons = [], []
    for dataset in DATASETS:
        unit = evaluation['datasets'][dataset]
        if set(unit['role_summary']) != set(ROLES) or len(unit['models']) != 15:
            raise ValueError('Every prescribed role and seed must remain in the report')
        entries = [(family, None, 'mean_seed_loss', row['ids'], row['periods'])
                   for family, row in unit['role_summary'].items()]
        entries += [(row['family'], row['seed'], 'single_model', [identifier], row['periods'])
                    for identifier, row in unit['models'].items()]
        for family, seed, aggregation, ids, periods in entries:
            if set(periods) != set(PERIODS):
                raise ValueError('Adverse or missing periods cannot be omitted')
            for period in PERIODS:
                values = periods[period]
                row = {'dataset': dataset, 'family': family, 'seed': seed, 'aggregation': aggregation,
                       'ids': ids, 'period': period,
                       **{name: values[name] for name in ('mse', 'mae', 'signed_mean_error')}}
                for name in ('mse', 'mae'):
                    denominator = metric(evaluation, dataset, 'full_mse', period, name)
                    row[name + '_reference_full_mse_role'] = denominator
                    row[name + '_relative_change_percent'] = 100 * (values[name] - denominator) / denominator if denominator else None
                rows.append(row)
        expected = {'temporal_lora__vs__' + family for family in ROLES if family != 'temporal_lora'}
        if set(unit['comparisons']) != expected:
            raise ValueError('The fixed paired comparison list changed')
        for key, item in unit['comparisons'].items():
            if item['status'] != 'available':
                raise ValueError('A prescribed paired comparison is unavailable')
            for period in PERIODS:
                for name in ('mse', 'mae'):
                    value = item['periods'][period][name]
                    comparisons.append({'dataset': dataset, 'comparison': key, 'period': period, 'metric': name,
                                        **value})
    return rows, comparisons


def config_key(row):
    return tuple(row[key] for key in ('dataset', 'family', 'batch', 'execution', 'chunk_rows'))


def cost_values(device):
    grid_path = HERE / f'cost_{device}_grid.json'
    rows_path = HERE / f'cost_{device}_rows.json'
    if not grid_path.exists():
        return [], {'status': 'PENDING', 'reason': 'The fixed cost grid has not been materialized',
                    'expected_rows': 588 if device == 'cuda' else 48}, {}
    grid = read_json(grid_path)
    expected_count = 588 if device == 'cuda' else 48
    if len(grid['rows']) != expected_count or len({row['key'] for row in grid['rows']}) != expected_count:
        raise ValueError('Cost grid count or uniqueness differs from the contract')
    sources = {'grid': artifact(grid_path), 'binding': artifact(HERE / 'cost_binding.json', grid['binding_sha256'])}
    if rows_path.exists():
        saved = read_json(rows_path)
        raw = saved['rows']
        sources['rows'] = artifact(rows_path)
        reported_status = saved['status']
        if saved['binding_sha256'] != grid['binding_sha256']:
            raise ValueError('Cost rows and grid have different source bindings')
    else:
        paths = sorted((HERE / 'cost_rows' / device).glob('*.json'))
        raw = [read_json(path) for path in paths]
        sources['individual_rows'] = [artifact(path) for path in paths]
        reported_status = 'partial'
    expected = {row['key']: row for row in grid['rows']}
    actual = {}
    for row in raw:
        key = row['key']
        if key in actual or key not in expected or row['status'] != 'complete':
            raise ValueError('Duplicate/unplanned/incomplete raw timing row')
        if row['binding_sha256'] != grid['binding_sha256'] or any(row[name] != value for name, value in expected[key].items()):
            raise ValueError('Raw timing identity differs from the fixed grid')
        actual[key] = row
    if reported_status == 'complete' and set(actual) != set(expected):
        raise ValueError('A complete cost phase is missing fixed rows')
    grouped = defaultdict(list)
    for row in grid['rows']:
        if row['sentinel'] is None:
            grouped[config_key(row)].append(row)
    result = []
    for key, planned in grouped.items():
        dataset, family, batch, execution, chunk = key
        ids = list(dict.fromkeys(row['id'] for row in planned))
        expected_seeds = {None} if family == 'f0' else set(SEEDS)
        if {row['seed'] for row in planned} != expected_seeds or len(planned) != 3 * len(expected_seeds):
            raise ValueError('A timing group must contain every paired seed and three blocks')
        missing = [row['key'] for row in planned if row['key'] not in actual]
        output = {'dataset': dataset, 'family': family, 'device': device, 'batch': batch,
                  'execution': execution, 'chunk_rows': chunk, 'ids': ids,
                  'status': 'PENDING' if missing else 'complete_group',
                  'planned_rows': len(planned), 'completed_rows': len(planned) - len(missing),
                  'missing_row_keys': missing, **dict.fromkeys(COST_NUMERIC)}
        if not missing:
            members = [actual[row['key']] for row in planned]
            by_id = [[row for row in members if row['id'] == identifier] for identifier in ids]
            ms = [row['milliseconds_per_origin']['median'] for row in members]
            speed = [row['origins_per_second']['median'] for row in members]
            if any(not np.isfinite(value) or value <= 0 for value in ms + speed):
                raise ValueError('Timing must be finite and positive')
            output.update(mean_seed_median_ms=float(np.mean([np.median([r['milliseconds_per_origin']['median'] for r in group]) for group in by_id])),
                          mean_seed_median_origins_s=float(np.mean([np.median([r['origins_per_second']['median'] for r in group]) for group in by_id])),
                          ms_low=min(ms), ms_high=max(ms), origins_s_low=min(speed), origins_s_high=max(speed))
            for kind in ('allocated', 'reserved'):
                values = [row['peak_' + kind + '_bytes'] for row in members]
                if all(value is not None for value in values):
                    output[kind + '_bytes'] = float(np.mean([np.median([r['peak_' + kind + '_bytes'] for r in group]) for group in by_id]))
                    output[kind + '_low_bytes'], output[kind + '_high_bytes'] = min(values), max(values)
                elif any(value is not None for value in values):
                    raise ValueError('Mixed missing and present device memory values')
            for name in ('deployed_parameters', 'parameter_bytes', 'buffer_bytes', 'deployment_tensor_bytes',
                         'this_campaign_adaptation_registered_parameters', 'historical_adaptation_registered_parameters'):
                values = [row['model'][name] for row in members]
                if len(set(values)) != 1:
                    raise ValueError('Identical family execution has inconsistent parameter accounting')
                output[name] = values[0]
        result.append(output)
    sentinel = []
    if device == 'cuda':
        for dataset in DATASETS:
            for block in range(3):
                for batch in (1, 4):
                    pair = {row['sentinel']: row for row in actual.values() if row['dataset'] == dataset
                            and row['block'] == block and row['batch'] == batch and row['sentinel'] is not None}
                    item = {'dataset': dataset, 'block': block, 'batch': batch, 'status': 'PENDING'}
                    if set(pair) == {'pre', 'post'}:
                        pre, post = (pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post'))
                        item.update(status='complete_pair', pre_ms=pre, post_ms=post,
                                    relative_change_percent=100 * (post - pre) / pre)
                    sentinel.append(item)
    status = {'status': 'complete' if set(actual) == set(expected) else 'partial',
              'recorded_status': reported_status, 'completed_rows': len(actual), 'expected_rows': len(expected),
              'missing_row_keys': [key for key in expected if key not in actual],
              'complete_groups': sum(row['status'] == 'complete_group' for row in result),
              'planned_groups': len(result), 'sentinel_drift': sentinel,
              'range_meaning': 'Extrema across seeds and block medians; not confidence intervals',
              'partial_policy': 'Every predeclared config retained; numerical summaries require all seeds and all three blocks'}
    return result, status, sources


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    return matplotlib, plt


def save_figure(fig, stem):
    values = {}
    for suffix in ('png', 'svg'):
        path = HERE / (stem + '.' + suffix)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        if suffix == 'svg':
            text = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')
        values[suffix] = artifact(path)
    return values


def accuracy_figure(evaluation):
    mpl, plt = pyplot()
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), layout='constrained')
    relative = [100 * (metric(evaluation, ds, family, period, name) / metric(evaluation, ds, 'full_mse', period, name) - 1)
                for ds in DATASETS for family in ROLES for period in PERIODS for name in ('mse', 'mae')]
    bound = max(10., max(abs(value) for value in relative))
    norm = mpl.colors.SymLogNorm(linthresh=10, vmin=-bound, vmax=bound, base=10)
    cmap = plt.get_cmap('RdBu_r')
    for i, ds in enumerate(DATASETS):
        for j, name in enumerate(('mse', 'mae')):
            ax = axes[i, j]
            numbers = np.array([[metric(evaluation, ds, family, period, name) for period in PERIODS] for family in ROLES])
            reference = np.array([metric(evaluation, ds, 'full_mse', period, name) for period in PERIODS])
            values = 100 * (numbers / reference - 1)
            image = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
            for r in range(len(ROLES)):
                for c in range(3):
                    rgba = cmap(norm(values[r, c]))
                    luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    ax.text(c, r, f'{numbers[r,c]:.4g}\n({values[r,c]:+.1f}%)', ha='center', va='center',
                            fontsize=7.7, color='white' if luminance < .5 else 'black')
            ax.set_xticks(range(3), ('TEST-A', 'TEST-B', 'Pooled'))
            ax.set_yticks(range(len(ROLES)), [LABELS[r] for r in ROLES] if j == 0 else [])
            ax.set_title(f'{chr(65+i*2+j)}  {ds}: {name.upper()}')
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6,
                 label='Relative change vs same-period FULL_MSE (%); symmetric-log color scale')
    fig.suptitle('V16 exposed-development accuracy: value and relative external gap\n'
                 'Mean seed losses; pooled errors/counts; lower is better; no cross-dataset mean', fontsize=12)
    output = save_figure(fig, FIGURES[0])
    plt.close(fig)
    return output


def range_error(value, low, high):
    if value is None or low is None or high is None:
        return None
    return [[max(0., value - low)], [max(0., high - value)]]


def cost_figure(evaluation, rows, gpu_status):
    _, plt = pyplot()
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), layout='constrained', gridspec_kw={'width_ratios': [1, 1, 1.65]})
    for i, dataset in enumerate(DATASETS):
        subset = [row for row in rows if row['dataset'] == dataset and row['device'] == 'cuda']
        complete = [row for row in subset if row['batch'] == 4 and row['status'] == 'complete_group']
        for row in complete:
            family = row['family']
            value = metric(evaluation, dataset, family, 'combined', 'mse')
            common = {'fmt': MARKERS[row['execution']], 'color': COLORS[family], 'capsize': 2,
                      'markersize': 6, 'markeredgecolor': 'white', 'markeredgewidth': .4, 'alpha': .9}
            axes[i, 0].errorbar(row['mean_seed_median_origins_s'], value,
                               xerr=range_error(row['mean_seed_median_origins_s'], row['origins_s_low'], row['origins_s_high']), **common)
            if row['allocated_bytes'] is not None:
                scale = 1024**2
                axes[i, 1].errorbar(row['allocated_bytes'] / scale, value,
                    xerr=range_error(row['allocated_bytes'] / scale, row['allocated_low_bytes'] / scale,
                                     row['allocated_high_bytes'] / scale), **common)
        for j in (0, 1):
            ax = axes[i, j]
            ax.set_title(f'{dataset}: ' + ('B4 throughput' if j == 0 else 'B4 GPU allocation'))
            ax.set_ylabel('Pooled MSE (mean seed losses)')
            ax.set_xlabel('Origins/s (log scale)' if j == 0 else 'Peak allocated memory (MiB)')
            ax.grid(alpha=.2)
            if not complete:
                ax.text(.5, .5, 'PENDING: no complete paired 3-block groups', ha='center', transform=ax.transAxes, fontsize=8)
            if j == 0:
                ax.set_xscale('log')
        b1 = [row for row in rows if row['dataset'] == dataset and row['batch'] == 1]
        ax = axes[i, 2]
        ax.set_axis_off()
        table_rows = []
        for row in b1:
            label = LABELS[row['family']] + '/' + ('CPU' if row['device'] == 'cpu' else row['execution'])
            if row['status'] == 'complete_group':
                latency = f"{row['mean_seed_median_ms']:.3g} [{row['ms_low']:.3g}, {row['ms_high']:.3g}]"
                memory = f"{row['allocated_bytes']/1024**2:.2f}" if row['allocated_bytes'] is not None else 'N/A'
            else:
                latency, memory = 'PENDING', 'PENDING'
            table_rows.append([label, latency, memory])
        if table_rows:
            table = ax.table(cellText=table_rows, colLabels=['B1 path', 'ms/origin [range]', 'Alloc. MiB'],
                             loc='center', cellLoc='left', colWidths=[.43, .4, .17])
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1, 1.5)
            for (r, _), cell in table.get_celld().items():
                cell.set_linewidth(.25)
                if r == 0:
                    cell.set_facecolor('#eeeeee')
        else:
            ax.text(.5, .5, 'B1 timing PENDING', ha='center', transform=ax.transAxes)
        ax.set_title('All B1 executions; CPU listed separately')
        if subset and any(row['status'] != 'complete_group' for row in subset):
            axes[i, 0].text(.02, .98, 'PARTIAL GRID: see missing groups in CSV', transform=axes[i, 0].transAxes,
                            va='top', fontsize=7, color='#a63603')
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='none', label=LABELS[family]) for family, color in COLORS.items()]
    handles += [plt.Line2D([], [], color='#444444', marker=marker, linestyle='none', label=execution) for execution, marker in MARKERS.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=8)
    fig.suptitle('V16 same-session accuracy/cost points; GPU cost status: ' + gpu_status['status'] + '\n'
                 'Bars and B1 brackets span seed/block medians, not CIs. No dominance/frontier claim is inferred.', fontsize=12)
    output = save_figure(fig, FIGURES[1])
    plt.close(fig)
    return output


def build_report(job):
    targets = [HERE / name for name in ('report_values.json', 'accuracy_comparison.csv', 'cost_comparison.csv', 'paired_comparisons.csv')]
    targets += [HERE / (stem + '.' + suffix) for stem in FIGURES for suffix in ('png', 'svg')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve existing/partial reports; explicit repair must retain prior artifacts')
    evaluation = read_json(HERE / 'evaluation01.json')
    if evaluation['status'] != 'complete' or set(evaluation['datasets']) != set(DATASETS):
        raise ValueError('All four prescribed accuracy evaluations must finish before this report')
    require_storage(20 * 1024**2)
    accuracy, comparisons = accuracy_values(evaluation)
    gpu, gpu_status, gpu_sources = cost_values('cuda')
    cpu, cpu_status, cpu_sources = cost_values('cpu')
    costs = gpu + cpu
    for row in costs:
        row['mse'] = metric(evaluation, row['dataset'], row['family'], 'combined', 'mse')
        row['mae'] = metric(evaluation, row['dataset'], row['family'], 'combined', 'mae')
    csv_write(targets[1], accuracy, list(accuracy[0]))
    cost_fields = ['dataset', 'family', 'device', 'batch', 'execution', 'chunk_rows', 'ids', 'status',
                   'planned_rows', 'completed_rows', 'missing_row_keys', *COST_NUMERIC, 'mse', 'mae']
    csv_write(targets[2], costs, cost_fields)
    csv_write(targets[3], comparisons, list(comparisons[0]))
    job.heartbeat('Aggregate CSVs preserved; rendering fixed figures')
    figures = {'accuracy_by_period': accuracy_figure(evaluation),
               'accuracy_cost': cost_figure(evaluation, costs, gpu_status)}
    complete_cost = gpu_status['status'] == cpu_status['status'] == 'complete'
    value = {'schema': 'v16_report_values_1',
             'status': 'complete' if complete_cost else 'partial_cost',
             'scope': 'Four already exposed development datasets; no new model runs, scores, or method selection',
             'evaluation': artifact(HERE / 'evaluation01.json'), 'selection': artifact(HERE / 'selected.json'),
             'seal': artifact(HERE / 'selection_seal.json'),
             'cost_summary': artifact(HERE / 'cost_summary.json') if (HERE / 'cost_summary.json').exists() else None,
             'original_seed_channel_tables': evaluation['tables'],
             'cost_sources': {'gpu': gpu_sources, 'cpu': cpu_sources},
             'source_files': {name: digest(HERE / name) for name in ('report_v16.py', 'evaluate_v16.py',
                 'cost_v16.py', 'model_v16.py', 'runtime_v16.py', 'PLAN.md', 'protocol.json')},
             'tables': {'comparison': artifact(targets[1]), 'paired_differences': artifact(targets[3]),
                        'cost_comparison': artifact(targets[2])}, 'figures': figures,
             'accuracy': accuracy, 'paired_comparisons': comparisons, 'cost': costs,
             'roles': list(ROLES), 'periods': list(PERIODS), 'accuracy_rows': len(accuracy),
             'paired_difference_rows': len(comparisons), 'cost_rows': len(costs),
             'cost_status': {'top_level': 'complete' if complete_cost else 'partial',
                             'gpu': gpu_status['status'], 'cpu': cpu_status['status'],
                             'complete_for_report': complete_cost},
             'cost_coverage': {'gpu': gpu_status, 'cpu': cpu_status},
             'no_new_numeric_claims': {'fit_count': 0, 'model_runs': 0, 'new_forecast_scores': 0,
                                      'bootstrap_recomputed': False, 'frontier_claim': False},
             'notes': ['MSE and MAE remain separate. No averaging of raw scores across datasets.',
                       'Relative change is 100*(candidate-reference)/reference, not percentage points.',
                       'Role means are mean seed losses, never prediction ensembles. F0 has one deterministic instance.',
                       'Paired intervals are copied from the fixed evaluation; they omit parent/method-selection uncertainty.',
                       'Only complete paired three-block cost groups receive summaries; every planned incomplete config remains visible.',
                       'Timing ranges are observed seed/block variability, not statistical confidence intervals.',
                       'Cost is same-session online deployment; no historical timing ratios or forecast caches.',
                       'TEMPORAL_F0 and COARSE_ONLY have accuracy only, not new cost claims.',
                       'New adaptation has 294912 LoRA parameters; the frozen DIRECT parent was previously fitted with 24624.']}
    save_json(targets[0], value)
    job.heartbeat('V16 aggregate report and source-linked figures complete')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', required=True, type=float)
    args = parser.parse_args()
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'v16 aggregate tables and figures', 'fit_count': 0, 'model_runs': 0, 'new_scores': 0}) as job:
        build_report(job)
