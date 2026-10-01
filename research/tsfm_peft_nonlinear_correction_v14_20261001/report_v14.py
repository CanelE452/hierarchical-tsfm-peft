"""Small method-centered tables and figures from completed v14 evidence only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from runtime_v14 import HERE, Job, artifact, read_json, save_json
from evaluate_v14 import ROLES, PERIODS

LABELS = {'level_gelu': 'LEVEL + GELU', 'level_linear': 'LEVEL + Linear', 'direct_gelu': 'Direct + GELU',
          'group_res': 'GROUP RES', 'pca_level': 'PCA LEVEL', 'a': 'PCA + LoRA (A)',
          'f0': 'F0', 'full_mse': 'Full LoRA MSE', 'native': 'Full LoRA native', 'direct': 'Direct NLinear'}
COLORS = dict(zip(ROLES, ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#882255',
                         '#000000', '#D55E00', '#56B4E9', '#777777', '#332288')))


def csv_file(name, rows):
    if not rows:
        return None
    path = HERE / name
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)
    return artifact(path)


def figure_files(fig, stem, plt):
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    paths = {}
    for extension in ('png', 'svg'):
        path = folder / (stem + '.' + extension)
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
        paths[extension] = artifact(path)
    plt.close(fig)
    return paths


def render(job):
    destination = HERE / 'report_values.json'
    if destination.exists():
        raise FileExistsError('Preserve the original report; use an explicit correction record')
    evaluation = read_json(HERE / 'evaluation01.json')
    costs = read_json(HERE / 'cost_summary.json')
    choice = read_json(HERE / 'selected.json')
    if evaluation['status'] != 'complete':
        raise ValueError('All four accuracy comparisons must complete')
    sources = {'evaluation': artifact(HERE / 'evaluation01.json'), 'cost': artifact(HERE / 'cost_summary.json'),
               'selection': artifact(HERE / 'selected.json'), 'script': artifact(Path(__file__))}
    rows, paired = [], []
    for dataset, unit in evaluation['datasets'].items():
        for family in ROLES:
            record = unit['role_summary'][family]
            for period in PERIODS:
                score = record['periods'][period]
                rows.append({'dataset': dataset, 'family': family, 'period': period, 'id': None, 'seed': None,
                             'aggregation': 'mean_seed_loss', 'mse': score['mse'], 'mae': score['mae'],
                             'signed_mean_error': score['signed_mean_error'], 'source_sha256': sources['evaluation']['sha256'],
                             'pointer': f'/datasets/{dataset}/role_summary/{family}/periods/{period}'})
                for identifier in record['ids']:
                    member = unit['models'][identifier]
                    score = member['periods'][period]
                    rows.append({'dataset': dataset, 'family': family, 'period': period, 'id': identifier, 'seed': member['seed'],
                                 'aggregation': 'seed_loss', 'mse': score['mse'], 'mae': score['mae'],
                                 'signed_mean_error': score['signed_mean_error'], 'source_sha256': sources['evaluation']['sha256'],
                                 'pointer': f'/datasets/{dataset}/models/{identifier}/periods/{period}'})
        for name, item in unit['comparisons'].items():
            for period in PERIODS:
                for metric in ('mse', 'mae'):
                    paired.append({'dataset': dataset, 'comparison': name, 'period': period, 'metric': metric,
                                   **item['periods'][period][metric], 'source_sha256': sources['evaluation']['sha256']})
    cost_rows, cost_instances = [], {}
    for device in ('gpu', 'cpu'):
        summary = costs.get(device) or {}
        if summary.get('status') == 'complete':
            cost_instances[device] = summary['instances']
            for group in summary['groups']:
                scores = evaluation['datasets'][group['dataset']]['role_summary'][group['family']]['periods']['combined']
                cost_rows.append({**group, 'combined_mse': scores['mse'], 'combined_mae': scores['mae'],
                                  'source_sha256': sources['cost']['sha256']})
    tables = {'accuracy': csv_file('comparison.csv', rows), 'paired': csv_file('paired_differences.csv', paired),
              'cost': csv_file('cost_comparison.csv', cost_rows)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none'})
    datasets = list(evaluation['datasets'])
    fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 3.1 * len(datasets)), layout='constrained', squeeze=False)
    for row_index, dataset in enumerate(datasets):
        unit = evaluation['datasets'][dataset]
        for metric_index, metric in enumerate(('mse', 'mae')):
            ax = axes[row_index, metric_index]
            for index, family in enumerate(ROLES):
                summary = unit['role_summary'][family]
                for offset, period, marker in ((-.16, 'test_a', '<'), (0, 'combined', 'o'), (.16, 'test_b', '>')):
                    value = summary['periods'][period][metric]
                    ax.scatter(index + offset, value, marker=marker, color=COLORS[family], s=30, edgecolors='#333333', linewidths=.3)
                seed_values = [unit['models'][identifier]['periods']['combined'][metric] for identifier in summary['ids']]
                ax.plot([index, index], [min(seed_values), max(seed_values)], color=COLORS[family], linewidth=2)
            ax.set_xticks(range(len(ROLES)), [LABELS[f].replace(' ', '\n', 1) for f in ROLES], fontsize=7)
            ax.set_title(dataset + ': ' + metric.upper())
            ax.set_ylabel('Observed channel-macro ' + metric.upper())
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        job.heartbeat(dataset + ': report all methods and periods')
    fig.suptitle('Frozen-parent correction: all four exposed development datasets\nMean seed losses; not prediction ensembles; no cross-dataset score average')
    fig.supxlabel('Left triangle: period A; circle: pooled combined; right triangle: period B. Line: selected-seed combined range.', fontsize=9)
    figures = {'accuracy': {'outputs': figure_files(fig, 'v14_accuracy_all_periods', plt), 'sources': sources,
        'caption': 'All prescribed references and adverse periods retained. Per-dataset scales; seed ranges are not confidence intervals.'}}
    if costs['status'] == 'complete':
        markers = {'full': 'o', 'chunk_k': '^', 'chunk_4k': 's', 'direct': 'D'}
        cost_datasets = costs['eligible_datasets']
        fig, axes = plt.subplots(len(cost_datasets), 2, figsize=(12, 3.3 * len(cost_datasets)), layout='constrained', squeeze=False)
        for i, dataset in enumerate(cost_datasets):
            points = [r for r in cost_rows if r['device'] == 'cuda' and r['dataset'] == dataset and r['batch'] == 4]
            for point in points:
                y, family = point['combined_mse'], point['family']
                for ax, x in zip(axes[i], (point['mean_seed_median_peak_allocated_bytes'] / 1024 ** 2, point['mean_seed_median_origins_s'])):
                    ax.scatter(x, y, color=COLORS[family], marker=markers[point['execution']], s=45, edgecolors='#333333', linewidths=.4)
            for ax, label in zip(axes[i], ('Peak allocated MiB', 'Origins / second')):
                ax.set_xscale('log')
                ax.set_xlabel(label + ' (log scale)')
                ax.set_ylabel('Combined MSE')
                ax.set_title(dataset)
                ax.grid(alpha=.2)
        handles = [Line2D([0], [0], color=COLORS[f], label=LABELS[f], linewidth=3) for f in ROLES if f != 'group_res']
        fig.legend(handles=handles, loc='outside upper center', ncols=4, fontsize=8)
        fig.supxlabel('Circle: full; square: MSE-LoRA chunk 4K; triangle: MSE-LoRA chunk K; diamond: Direct-based. Three passes/block; no stable Pareto claim.', fontsize=8)
        figures['cost'] = {'outputs': figure_files(fig, 'v14_accuracy_cost_batch4', plt), 'sources': sources,
            'caption': 'Development-selected datasets only, fixed batch4 options. No GROUP cost claim or old/new timing ratios. CPU direct-based methods remain separate. Raw ranges and sentinels govern uncertainty.'}
    value = {'status': 'complete', 'sources': sources, 'tables': tables, 'figures': figures, 'accuracy_rows': rows,
             'paired_comparisons': paired, 'cost_configurations': cost_rows, 'cost_instances': cost_instances,
             'selected_training': {dataset: {family: {key: item[key] for key in
                 ('lr', 'selected_epochs', 'mean_best_val_mse', 'nonstep0_seeds', 'changed_head_seeds', 'parent_role')}
                 for family, item in families.items()} for dataset, families in choice['selected'].items()},
             'lr_candidates': choice['lr_candidates'],
             'cost_status': costs['status'], 'cost_sentinels': (costs.get('gpu') or {}).get('sentinels', []),
             'independent_confirmation': False, 'method': 'Frozen selected PCA LEVEL or direct NLinear parent plus a new matched-size correction; all parent parameters unchanged',
             'adaptation_parameters': {
                 'level_gelu': {'new': 17920, 'parent': 17920, 'staged_total': 35840},
                 'level_linear': {'new': 17920, 'parent': 17920, 'staged_total': 35840},
                 'direct_gelu': {'new': 17920, 'parent': 24624, 'staged_total': 42544},
                 'meaning': 'Fitted parameters across parent and new stages, not total deployment parameters or historical training time'},
             'cost_eligible_datasets': costs.get('eligible_datasets', []),
             'cost_selection_scope': 'Accuracy retained for every dataset; cost measured only under the prespecified development selection rule',
             'claim_limits': 'GELU correction is nonlinear: old shared-linear P/Q commutation or effective rank<=33 claims do not describe the whole new model. Matched parameter count does not imply matched function space or optimization; no novelty or full-prior-method reproduction claim.',
             'interpretation': 'Mean seed losses, relative percent not percentage points; incomplete costs are not a complete frontier'}
    save_json(destination, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s, metadata={'fit_count': 0, 'purpose': 'v14 report'}) as job:
        render(job)


if __name__ == '__main__':
    main()
