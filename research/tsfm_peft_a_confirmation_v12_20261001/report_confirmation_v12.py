"""Render completed confirmation evidence; no data loading, prediction, or fitting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from runtime_confirmation_v12 import HERE, Job, artifact, read_json, save_json


ROLES = ('a', 'level', 'f0', 'full_mse', 'native', 'direct', 'a_main', 'a_last')
DEPLOYED = ROLES[:6]
PERIODS = ('combined', 'test_a', 'test_b')
LABELS = {'a': 'A: latent LoRA + Q*', 'level': 'LEVEL parent', 'f0': 'F0',
          'full_mse': 'Full LoRA (MSE)', 'native': 'Full LoRA (native)',
          'direct': 'Direct NLinear', 'a_main': 'A main only', 'a_last': 'A main + last Q'}
SHORT_LABELS = {'a': 'A', 'level': 'LEVEL', 'f0': 'F0', 'full_mse': 'LoRA\nMSE',
                'native': 'LoRA\nnative', 'direct': 'Direct\nNLinear',
                'a_main': 'A main\nonly', 'a_last': 'A main\n+ last Q'}
COLORS = {'a': '#0072B2', 'level': '#E69F00', 'f0': '#000000',
          'full_mse': '#009E73', 'native': '#CC79A7', 'direct': '#56B4E9',
          'a_main': '#D55E00', 'a_last': '#F0E442'}
MARKERS = {'full': 'o', 'chunk_4k': 's', 'chunk_k': '^', 'direct': 'D'}
MIB = 1024 ** 2


def csv_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, (dict, list)) else value


def write_csv(path, rows):
    if not rows:
        raise ValueError('An expected confirmation table is empty')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: csv_cell(value) for key, value in row.items()} for row in rows)


def checked_source(name):
    path = HERE / name
    receipt = artifact(path)
    value = read_json(path)
    artifact(path, receipt['sha256'])
    return value, receipt


def read_completed_sources():
    sources = {}
    evaluation, sources['evaluation'] = checked_source('confirmation_evaluation01.json')
    cost, sources['cost'] = checked_source('confirmation_cost_summary.json')
    if evaluation['status'] != 'complete' or cost['status'] != 'complete':
        raise ValueError('Both accuracy and the entire prescribed cost grid must be complete')
    if evaluation['evidence_status'] != 'confirmed_from_saved_arrays':
        raise ValueError('Evaluation must be derived from saved actual prediction arrays')
    for key in ('selection_seal', 'prediction_manifest'):
        entry = evaluation[key]
        sources[key] = artifact(entry['path'], entry['sha256'])
    sources['cost_binding'] = artifact(cost['binding']['path'], cost['binding']['sha256'])
    binding = read_json(sources['cost_binding']['path'])
    seal = read_json(sources['selection_seal']['path'])
    for document in (binding, seal):
        for name, sha in document['files'].items():
            artifact(HERE / name, sha)
    datasets = set(evaluation['datasets'])
    if datasets != set(seal['datasets']):
        raise ValueError('Evaluation units differ from the joint selection seal')
    raw = {}
    for section, device, expected in (('gpu', 'cuda', 135), ('cpu', 'cpu', 12)):
        raw[section], sources[section + '_rows'] = checked_source(f'confirmation_cost_{device}_rows.json')
        values = raw[section]
        if values['status'] != 'complete' or cost[section]['status'] != 'complete':
            raise ValueError('An incomplete device phase cannot be presented as a finished comparison')
        if len(values['rows']) != expected * len(datasets) or len(values['rows']) != values['expected_rows']:
            raise ValueError('The prescribed complete cost row count changed')
        if {row['dataset'] for row in values['rows']} != datasets:
            raise ValueError('Cost and evaluation units differ')
        if values['binding_sha256'] != sources['cost_binding']['sha256']:
            raise ValueError('Cost raw rows have a different campaign binding')
        if any(row['status'] != 'complete' or row['binding_sha256'] != values['binding_sha256']
               for row in values['rows']):
            raise ValueError('A cost row is incomplete or belongs to another campaign')
        phase, sources[section + '_summary'] = checked_source(f'confirmation_cost_{device}_summary.json')
        if phase != cost[section]:
            raise ValueError('Combined and device-specific cost summaries differ')
    for dataset, unit in evaluation['datasets'].items():
        if set(unit['role_summary']) != set(ROLES) or len(unit['models']) != 15:
            raise ValueError('Every prescribed model and same-checkpoint Q reference must be reported')
        for family in ROLES:
            ids = unit['role_summary'][family]['ids']
            if len(ids) != (1 if family == 'f0' else 2):
                raise ValueError('Missing model seed in accuracy summary')
            for period in PERIODS:
                for metric in ('mse', 'mae', 'signed_mean_error'):
                    seeds = [unit['models'][identifier]['periods'][period][metric] for identifier in ids]
                    reported = unit['role_summary'][family]['periods'][period][metric]
                    if not np.isfinite(seeds).all() or not np.isclose(np.mean(seeds), reported, rtol=1e-12, atol=1e-12):
                        raise ValueError('Role summary is not the mean of the recorded seed losses')
        for section in ('gpu', 'cpu'):
            for item in cost[section]['instances']:
                if item['dataset'] == dataset and item['id'] not in unit['models']:
                    raise ValueError('Cost has no matching accuracy model')
    return evaluation, cost, raw, sources


def accuracy_tables(evaluation, sources):
    rows, comparisons = [], []
    for dataset, unit in evaluation['datasets'].items():
        for family in ROLES:
            summary = unit['role_summary'][family]
            members = [('mean_seed_loss', None, None, summary['periods'])]
            members += [('seed_loss', identifier, unit['models'][identifier]['seed'],
                         unit['models'][identifier]['periods']) for identifier in summary['ids']]
            for aggregation, identifier, seed, periods in members:
                pointer = f'/datasets/{dataset}/' + (f'models/{identifier}' if identifier else f'role_summary/{family}')
                for period in PERIODS:
                    score = periods[period]
                    row = {'dataset': dataset, 'family': family, 'label': LABELS[family],
                           'aggregation': aggregation, 'id': identifier, 'seed': seed, 'period': period,
                           'mse': score['mse'], 'mae': score['mae'], 'signed_mean_error': score['signed_mean_error'],
                           'origin_count': score['origins'], 'target_counts': score['target_counts'],
                           'fit0_same_checkpoint_reference': family in ('a_main', 'a_last'),
                           'evidence_status': 'confirmed', 'source_sha256': sources['evaluation']['sha256'],
                           'source_pointer': pointer + '/periods/' + period}
                    for reference in ('level', 'f0', 'full_mse', 'native', 'direct'):
                        for metric in ('mse', 'mae'):
                            base = unit['role_summary'][reference]['periods'][period][metric]
                            if identifier and family != 'f0' and reference != 'f0':
                                matches = [model for model in unit['models'].values()
                                           if model['family'] == reference and model['seed'] == seed]
                                if len(matches) != 1:
                                    raise ValueError('Same-seed reference is missing')
                                base = matches[0]['periods'][period][metric]
                            delta = score[metric] - base
                            row[f'{metric}_minus_{reference}'] = delta
                            row[f'{metric}_relative_percent_vs_{reference}'] = 100 * delta / base if base > 0 else None
                    rows.append(row)
        for name, item in unit['comparisons'].items():
            for period in PERIODS:
                for metric in ('mse', 'mae'):
                    comparisons.append({'dataset': dataset, 'comparison': name, 'candidate': item['candidate'],
                        'reference': item['reference'], 'period': period, 'metric': metric,
                        **item['periods'][period][metric], 'evidence_status': 'confirmed',
                        'source_sha256': sources['evaluation']['sha256'],
                        'source_pointer': f'/datasets/{dataset}/comparisons/{name}/periods/{period}/{metric}'})
    return rows, comparisons


def cost_tables(evaluation, cost, raw, sources):
    rows, cpu = [], []
    for section in ('gpu', 'cpu'):
        for index, group in enumerate(cost[section]['groups']):
            dataset, family = group['dataset'], group['family']
            members = [item for item in cost[section]['instances'] if item['dataset'] == dataset
                       and item['family'] == family and item['batch'] == group['batch']
                       and item['execution'] == group['execution'] and item['chunk_rows'] == group['chunk_rows']]
            blocks = [row for row in raw[section]['rows'] if row['dataset'] == dataset
                      and row['family'] == family and row['batch'] == group['batch']
                      and row['execution'] == group['execution'] and row['chunk_rows'] == group['chunk_rows']
                      and row['sentinel'] is None]
            if len(blocks) != 3 * len(members) or {row['id'] for row in blocks} != set(group['ids']):
                raise ValueError('A summarized execution omits seed/block measurements')
            score = evaluation['datasets'][dataset]['role_summary'][family]['periods']['combined']
            row = {**group, 'combined_mse': score['mse'], 'combined_mae': score['mae'],
                   'block_ms_range': [min(r['milliseconds_per_origin']['median'] for r in blocks),
                                      max(r['milliseconds_per_origin']['median'] for r in blocks)],
                   'block_throughput_range': [min(r['origins_per_second']['median'] for r in blocks),
                                              max(r['origins_per_second']['median'] for r in blocks)],
                   'trainable_stage_parameters_by_instance': {item['id']: item['model']['current_stage_registered_parameters'] for item in members},
                   'stage_sum_parameters_by_instance': {item['id']: item['model']['sum_stage_registered_fit_parameters'] for item in members},
                   'deployment_parameters_by_instance': {item['id']: item['model']['deployed_parameters'] for item in members},
                   'deployment_tensor_bytes_by_instance': {item['id']: item['model']['deployment_tensor_bytes'] for item in members},
                   'cost_source_sha256': sources['cost']['sha256'], 'cost_source_pointer': f'/{section}/groups/{index}',
                   'accuracy_source_sha256': sources['evaluation']['sha256'],
                   'accuracy_source_pointer': f'/datasets/{dataset}/role_summary/{family}/periods/combined',
                   'evidence_status': 'confirmed', 'range_interpretation': 'All seed/block point estimates, not a confidence interval'}
            rows.append(row)
            if section == 'cpu':
                cpu.append(row)
    return rows, cpu


def save_figure(fig, stem, source_receipts, caption, plotted_rows, plt):
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    outputs = {}
    for extension in ('png', 'svg'):
        path = folder / (stem + '.' + extension)
        fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white',
                    metadata={'Creator': 'report_confirmation_v12.py'})
        outputs[extension] = artifact(path)
    plt.close(fig)
    return {'outputs': outputs, 'sources': source_receipts, 'caption': caption, 'plotted_values': plotted_rows}


def accuracy_figure(dataset, unit, sources, plt):
    groups = (DEPLOYED, ('a_main', 'a_last', 'a'))
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), layout='constrained')
    plotted = []
    for metric_index, metric in enumerate(('mse', 'mae')):
        for group_index, families in enumerate(groups):
            row = 2 * metric_index + group_index
            for column, period in enumerate(PERIODS):
                ax = axes[row, column]
                values = [unit['role_summary'][family]['periods'][period][metric] for family in families]
                ax.bar(np.arange(len(families)), values, color=[COLORS[family] for family in families],
                       edgecolor='#333333', linewidth=.45, width=.66)
                for i, family in enumerate(families):
                    ids = unit['role_summary'][family]['ids']
                    seed_values = [unit['models'][identifier]['periods'][period][metric] for identifier in ids]
                    jitter = np.linspace(-.12, .12, len(ids)) if len(ids) > 1 else [0.]
                    ax.scatter(i + np.asarray(jitter), seed_values, s=15, c='white', edgecolors='#333333', linewidths=.7, zorder=3)
                    ax.annotate(f'{values[i]:.4f}', (i, max([values[i]] + seed_values)), xytext=(0, 4),
                                textcoords='offset points', ha='center', fontsize=8)
                    plotted.append({'dataset': dataset, 'family': family, 'period': period, 'metric': metric,
                                    'mean_seed_loss': values[i], 'seed_losses': seed_values})
                ax.set_xticks(np.arange(len(families)), [SHORT_LABELS[family] for family in families], fontsize=8)
                ax.set_ylim(0, 1.21 * max(values + [unit['models'][identifier]['periods'][period][metric]
                             for family in families for identifier in unit['role_summary'][family]['ids']]))
                ax.set_axisbelow(True)
                ax.grid(axis='y', alpha=.2)
                ax.spines[['top', 'right']].set_visible(False)
                kind = 'Practical alternatives' if group_index == 0 else 'Same A checkpoint: Q contribution'
                ax.set_title(f"{kind} — {period.replace('_', '-').upper()}", fontsize=10)
                if column == 0:
                    ax.set_ylabel(metric.upper() + ' (lower is better)')
    contract = unit['unit_contract']
    fig.suptitle(f"{dataset}: C={contract['C']}, K={contract['K']}\n"
                 'Bars: mean seed loss; dots: individual seeds; F0 is a single frozen model', fontsize=13)
    caption = ('Observed channel-macro MSE and MAE in TRAIN-standardized coordinates; periods are pooled by channel error sum/count for combined. '
               'Mean seed losses are not an ensemble. Main-only and main-plus-last-Q reuse the identical A checkpoint; '
               'they do not represent independently optimized no-Q models. Panels have separate y-scales so no unfavorable role is omitted.')
    fig.supxlabel('Complete prescribed comparison; separate panel scales. No independently trained no-Q model.', fontsize=10)
    return save_figure(fig, f'confirmation_{dataset}_accuracy_periods',
                       {'evaluation': sources['evaluation']}, caption, plotted, plt)


def cost_figure(dataset, unit, rows, sources, plt):
    from matplotlib.lines import Line2D
    points = [row for row in rows if row['dataset'] == dataset and row['device'] == 'cuda' and row['batch'] == 4]
    if {point['family'] for point in points} != set(DEPLOYED):
        raise ValueError('A required practical alternative is absent from the batch4 figure')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), layout='constrained')
    plotted = []
    for point in points:
        memory = point['mean_seed_median_peak_allocated_bytes']
        if memory is None or memory <= 0:
            raise ValueError('GPU memory is missing; do not display unavailable values as zero')
        y = point['combined_mse']
        x_values = (memory / MIB, point['mean_seed_median_origins_s'])
        ranges = ([v / MIB for v in point['allocated_range_bytes']], point['block_throughput_range'])
        for ax, x, bounds in zip(axes, x_values, ranges):
            ax.plot(bounds, [y, y], color=COLORS[point['family']], linewidth=1, alpha=.55, zorder=2)
            ax.scatter([x], [y], color=COLORS[point['family']], marker=MARKERS[point['execution']],
                       s=75, edgecolors='#333333', linewidths=.6, alpha=.85, zorder=3)
        plotted.append({'family': point['family'], 'execution': point['execution'], 'chunk_rows': point['chunk_rows'],
                        'combined_mse': y, 'peak_allocated_mib': x_values[0], 'throughput_origins_s': x_values[1],
                        'memory_block_range_mib': ranges[0], 'throughput_block_range': ranges[1],
                        'cost_source_pointer': point['cost_source_pointer']})
    for ax in axes:
        ax.set_xscale('log')
        ax.set_ylabel('Combined observed macro MSE (lower is better)')
        ax.set_axisbelow(True)
        ax.grid(alpha=.2, which='both')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Peak allocated GPU memory, MiB (log scale; lower is better)')
    axes[1].set_xlabel('Batch4 origins / second (log scale; higher is better)')
    contract = unit['unit_contract']
    fig.suptitle(f"{dataset}: same-campaign batch4 deployment, C={contract['C']}, K={contract['K']}", fontsize=13)
    colors = [Line2D([0], [0], color=COLORS[family], linewidth=3, label=LABELS[family]) for family in DEPLOYED]
    shapes = [Line2D([0], [0], marker=MARKERS[key], linestyle='None', color='#666666', markersize=7,
                     label={'full': 'Full rows', 'chunk_4k': 'Chunk 4K rows', 'chunk_k': 'Chunk K rows', 'direct': 'Direct'}[key])
              for key in MARKERS]
    axes[0].legend(handles=colors, loc='best', fontsize=8, framealpha=.8)
    axes[1].legend(handles=shapes, loc='best', fontsize=8, framealpha=.8)
    fig.supxlabel('Bars span all seed/block point estimates; not confidence intervals. All approved GPU execution options shown.', fontsize=9)
    caption = ('All prescribed batch4 GPU configurations, including full and microbatched F0 and both LoRA baselines, are shown. '
               'Accuracy uses the corresponding stored full-model evaluation after output parity checks; no new accuracy ensemble. '
               'Cost is mean seed medians after median of three block medians. Horizontal bars retain all seed/block variation. '
               'CPU Direct is reported separately, not assigned fictitious GPU memory. No historical timing ratio or Pareto claim is inferred.')
    return save_figure(fig, f'confirmation_{dataset}_accuracy_cost_batch4',
                       {'evaluation': sources['evaluation'], 'cost': sources['cost'], 'gpu_rows': sources['gpu_rows']},
                       caption, plotted, plt)


def render(job):
    destination = HERE / 'report_confirmation_values.json'
    if destination.exists():
        raise FileExistsError('Preserve the original report; record any correction explicitly')
    evaluation, cost, raw, sources = read_completed_sources()
    source_script = artifact(Path(__file__))
    accuracy, comparisons = accuracy_tables(evaluation, sources)
    cost_rows, cpu_rows = cost_tables(evaluation, cost, raw, sources)
    tables = {'accuracy': ('confirmation_comparison.csv', accuracy),
              'paired_differences': ('confirmation_paired_differences.csv', comparisons),
              'all_cost_configurations': ('confirmation_cost_comparison.csv', cost_rows),
              'cpu_direct': ('confirmation_cpu_direct.csv', cpu_rows)}
    receipts = {}
    for name, (filename, values) in tables.items():
        job.check_limits()
        write_csv(HERE / filename, values)
        receipts[name] = artifact(HERE / filename)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.titleweight': 'normal', 'figure.dpi': 120})
    figures = {}
    for dataset, unit in evaluation['datasets'].items():
        job.heartbeat(dataset + ': render completed confirmation evidence')
        figures[dataset] = {'accuracy': accuracy_figure(dataset, unit, sources, plt),
                            'batch4_cost': cost_figure(dataset, unit, cost_rows, sources, plt)}
    figure_path = HERE / 'confirmation_figure_receipt.json'
    save_json(figure_path, {'status': 'complete', 'source_script': source_script, 'sources': sources,
                           'figures': figures, 'matplotlib_version': matplotlib.__version__})
    values = {'schema': 'v12_confirmation_report_values_1', 'status': 'complete', 'evidence_status': 'confirmed',
              'source_script': source_script, 'sources': sources, 'tables': receipts,
              'figure_receipt': artifact(figure_path), 'accuracy_rows': accuracy,
              'paired_comparisons': comparisons, 'cost_configurations': cost_rows,
              'cpu_direct': cpu_rows, 'cost_instances': {key: cost[key]['instances'] for key in ('gpu', 'cpu')},
              'cost_sentinels': cost['gpu']['sentinels'],
              'cost_raw_counts': {key: len(raw[key]['rows']) for key in ('gpu', 'cpu')},
              'environments': {key: {'environment': raw[key]['environment'], 'actual_device': raw[key]['actual_device'],
                                   'session_ids': raw[key]['session_ids']} for key in ('gpu', 'cpu')},
              'units': {dataset: {'unit_contract': unit['unit_contract'], 'columns': unit['columns'],
                                  'period_origin_counts': unit['period_origin_counts'], 'bootstrap': unit['bootstrap'],
                                  'aggregation': unit['aggregation'], 'A_Q_parity': unit['A_Q_parity']}
                        for dataset, unit in evaluation['datasets'].items()},
              'interpretation': {'accuracy': 'Mean of individual seed losses; not averaged predictions',
                 'relative_percent': '100 * (candidate - reference) / reference; not percentage points',
                 'seed_reference': 'Individual trained seeds use the corresponding reference seed; aggregate rows use mean losses; F0 has one frozen reference',
                 'q_references': 'Same selected A checkpoint; no independently optimized no-Q claim',
                 'timing': 'This campaign only; every prescribed execution retained; no automatic Pareto or stable superiority claim',
                 'cpu': 'Direct CPU is a separate practical deployment path; GPU memory unavailable rather than zero',
                 'scope': 'One building-electricity group where applicable; no claim of cross-domain confirmation',
                 'uncertainty': 'Seed/block ranges are descriptive; bootstrap intervals retain the evaluation source limitations'}}
    save_json(destination, values)
    job.heartbeat('Confirmation report, four tables, and two figures per approved unit complete')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'render completed fresh confirmation evidence', 'fit_count': 0,
                       'model_forward_count': 0}) as job:
        render(job)


if __name__ == '__main__':
    main()
