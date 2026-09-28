"""Plot completed v5 comparison/cost JSON; never load models or prediction arrays."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from runtime_v5 import HERE, ROOT, Job, digest, save_json

COLORS = {'modified_res': '#0072B2', 'modified_raw': '#D55E00', 'old_res': '#999999',
          'f0': '#333333', 'lora': '#009E73', 'level_only': '#CC79A7'}
LABELS = {'modified_res': 'LEVEL_RES', 'modified_raw': 'Matched LEVEL_RAW',
          'old_res': 'Original RES', 'f0': 'F0', 'lora': 'LoRA', 'level_only': 'LEVEL_ONLY'}
SEED_MARKERS = {92601: 'o', 92602: '^', None: 's'}


def read(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('status') != 'complete':
        raise ValueError(f'Only a completed result may be plotted: {path}')
    return result


def receipt(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'figure.titlesize': 10, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'v5_accuracy_recovery', 'pdf.fonttype': 42,
    })


def save_figure(fig, directory, stem, metadata):
    fig.canvas.draw()
    texts = [item.get_text() for item in fig.findobj(matplotlib.text.Text) if item.get_text()]
    if any('\uac00' <= character <= '\ud7a3' for text in texts for character in text):
        raise ValueError('All figure text must be English')
    for extension in ('png', 'svg'):
        path = directory / f'{stem}.{extension}'
        if path.exists():
            raise FileExistsError('Inspect existing figure output before replacing it')
        options = {'metadata': {'Date': None}} if extension == 'svg' else {}
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.08, **options)
    plt.close(fig)
    metadata.update(stem=stem, figure_text=texts, png_dpi=300,
                    figure_size_inches=list(fig.get_size_inches()))
    metadata['outputs'] = [receipt(directory / f'{stem}.{extension}') for extension in ('png', 'svg')]
    save_json(directory / f'{stem}.source.json', metadata)
    return metadata


def relative_accuracy(reports, sources, directory):
    roles = ['modified_res', 'modified_raw', 'f0', 'lora', 'level_only']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.1), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.29, top=0.75, wspace=0.29)
    fig.suptitle('Accuracy Recovery and Period-Specific Trade-offs', y=0.985)
    handles = [Line2D([], [], marker=marker, color='#333333', linestyle='none', markersize=4,
                      label=label) for marker, label in
               [('o', 'TEST-A / E1'), ('^', 'TEST-B / E2'), ('D', 'Combined / DEV')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.57, 0.91),
               ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.1)
    numeric = []
    for ax, dataset in zip(axes, ['hog', 'bull', 'electricity']):
        report = reports[dataset]
        periods = (['test_a', 'test_b', 'combined'] if dataset == 'hog' else
                   ['e1', 'e2', 'combined'] if dataset == 'bull' else ['dev'])
        offsets = [-0.22, 0, 0.22] if len(periods) == 3 else [0]
        markers = ['o', '^', 'D'] if len(periods) == 3 else ['D']
        plotted = [0.0]
        for row_index, role in enumerate(roles):
            if role not in report['role_scores']:
                raise ValueError(f'The planned LEVEL comparison is missing {dataset}:{role}')
            for offset, marker, period in zip(offsets, markers, periods):
                numerator = report['role_scores'][role][period]['mse']
                denominator = report['role_scores']['old_res'][period]['mse']
                if denominator <= 0:
                    raise ValueError('Relative change requires a positive original RES MSE')
                relative = 100 * (numerator / denominator - 1)
                plotted.append(relative)
                ax.scatter(relative, row_index + offset, color=COLORS[role], marker=marker,
                           edgecolor='black', linewidth=0.35, s=24, zorder=3)
                numeric.append({'dataset': dataset, 'role': role, 'period': period,
                                'mse': numerator, 'old_res_mse': denominator,
                                'relative_mse_change_pct': relative,
                                'mse_pointer': f'role_scores.{role}.{period}.mse',
                                'denominator_pointer': f'role_scores.old_res.{period}.mse',
                                'seed_mse': [{'id': identifier,
                                              'seed': report['models'][identifier]['spec'].get('seed'),
                                              'mse': report['models'][identifier]['scores'][period]['mse']}
                                             for identifier in report['roles'][role]]})
        low, high = min(plotted), max(plotted)
        margin = max(3.0, (high-low)*0.13)
        ax.set_xlim(low-margin, high+margin)
        ax.set_ylim(len(roles)-0.5, -0.6)
        ax.set_yticks(range(len(roles)), [LABELS[role] for role in roles])
        ax.axvline(0, color='#666666', linewidth=0.9, linestyle='--', zorder=1)
        ax.grid(axis='x', color='#e6e6e6', linewidth=0.55)
        title = {'hog': 'Hog (FIXED_ED, K8)', 'bull': 'Bull (FIXED_ED, K4)',
                 'electricity': 'Electricity (CURRENT, K8)'}[dataset]
        ax.set_title(title, pad=9)
        ax.set_xlabel('MSE change vs. original RES\n(%) ↓')
    fig.text(0.56, 0.025,
             'Zero: original RES in the same period. Lower is better.\n'
             'Each panel uses its own horizontal range; datasets are not pooled.',
             ha='center', fontsize=8, linespacing=1.35)
    return save_figure(fig, directory, 'accuracy_recovery', {
        'title': 'Accuracy Recovery and Period-Specific Trade-offs',
        'caption': 'Relative normalized macro MSE change from the original RES within each dataset and period. Each point uses the ratio of stored mean individual-seed losses, not prediction ensembling. Circles and triangles show the two separate periods; diamonds show pooled-within-dataset combined errors/counts, or Electricity DEV. Panels use different horizontal ranges and no dataset-wide score is formed. LEVEL_ONLY is the untrained level-restoration reference. All three datasets are exposed development evidence for v5, including the previously evaluated Hog periods. The plot does not establish independent confirmation or superiority on MAE.',
        'sources': sources, 'numeric_values': numeric,
        'calculation': '100 * (role_scores[role][period].mse / role_scores[old_res][period].mse - 1)',
        'uncertainty': 'Displayed points are descriptive mean-loss comparisons. They are not confidence intervals; seed and conditional temporal uncertainty remain in the source comparison JSON.',
        'prohibited_claim': 'A pooled cross-dataset success score, a fresh holdout, a pure causal mechanism proof, or percent-point improvement.'})


def accuracy_cost(report, cost, sources, directory):
    families = [('f0', 'f0'), ('lora', 'lora'), ('old_res', 'old_res'),
                ('modified_res', 'modified_res'), ('matched_raw', 'modified_raw')]
    rows = [row for row in cost['rows'] if row['batch_origins'] == 4 and row['include_in_primary']]
    if len(rows) != 27:
        raise ValueError('Expected all nine selected models across three batch-4 primary blocks')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.31, top=0.71, wspace=0.25)
    fig.suptitle('Hog Accuracy–Cost Trade-offs After LEVEL Recovery', y=0.985)
    handles = [Line2D([], [], marker='o', color=COLORS[role], linestyle='none', markersize=5,
                      label='Merged LoRA' if role == 'lora' else LABELS[role]) for _, role in families]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.91),
               ncol=3, frameon=False, columnspacing=1.4, handletextpad=0.4)
    numeric, centers = [], []
    all_mse, all_throughput, all_memory = [], [], []
    for family, role in families:
        members = [row for row in rows if row['family'] == family]
        fit_ids = sorted({row['id'] for row in members})
        if set(fit_ids) != set(report['roles'][role]):
            raise ValueError(f'Accuracy and cost model identities differ for {role}')
        role_points = []
        for identifier in fit_ids:
            blocks = sorted([row for row in members if row['id'] == identifier], key=lambda row: row['block'])
            if [row['block'] for row in blocks] != [0, 1, 2]:
                raise ValueError('Do not plot a cost model with incomplete benchmark blocks')
            seed = report['models'][identifier]['spec'].get('seed')
            mse = report['models'][identifier]['scores']['combined']['mse']
            throughput = [row['origins_per_second']['median'] for row in blocks]
            memory = [row['peak_allocated_bytes'] / 1024**2 for row in blocks]
            centers_fit = [median(throughput), median(memory)]
            for axis, observations, center in zip(axes, [throughput, memory], centers_fit):
                axis.plot([min(observations), max(observations)], [mse, mse], color=COLORS[role],
                          linewidth=0.9, alpha=0.75, zorder=2)
                axis.scatter(observations, [mse]*3, color=COLORS[role], marker=SEED_MARKERS[seed],
                             s=11, alpha=0.55, linewidth=0.35, edgecolor='black', zorder=3)
                axis.scatter(center, mse, marker=SEED_MARKERS[seed], s=29, facecolor='white',
                             edgecolor=COLORS[role], linewidth=1.15, zorder=4)
            record = {'id': identifier, 'role': role, 'seed': seed, 'combined_mse': mse,
                      'accuracy_pointer': f'models.{identifier}.scores.combined.mse',
                      'batch4_block_ids': [0, 1, 2], 'batch4_throughput_origins_s': throughput,
                      'batch4_peak_allocated_mib': memory,
                      'median_block_throughput_origins_s': centers_fit[0],
                      'median_block_peak_allocated_mib': centers_fit[1]}
            numeric.append(record)
            role_points.append(record)
            all_mse.append(mse)
            all_throughput.extend(throughput)
            all_memory.extend(memory)
        center = {'role': role, 'mean_seed_mse': mean(p['combined_mse'] for p in role_points),
                  'mean_seed_median_throughput_origins_s': mean(p['median_block_throughput_origins_s'] for p in role_points),
                  'mean_seed_median_peak_allocated_mib': mean(p['median_block_peak_allocated_mib'] for p in role_points)}
        expected_mse = report['role_scores'][role]['combined']['mse']
        if abs(center['mean_seed_mse'] - expected_mse) > 1e-12:
            raise ValueError('Stored role mean and plotted seed mean differ')
        for axis, x in zip(axes, [center['mean_seed_median_throughput_origins_s'], center['mean_seed_median_peak_allocated_mib']]):
            axis.scatter(x, center['mean_seed_mse'], marker='D', color=COLORS[role], s=41,
                         edgecolor='black', linewidth=0.55, zorder=5)
        centers.append(center)
    axes[0].set_xlabel('Batch-4 throughput (origins/s) ↑')
    axes[1].set_xlabel('Peak allocated GPU memory (MiB) ↓')
    axes[0].set_ylabel('Normalized macro MSE ↓')
    axes[0].set_title('Lower-right is better', fontsize=8, pad=7)
    axes[1].set_title('Lower-left is better', fontsize=8, pad=7)
    axes[0].set_xlim(0, max(all_throughput)*1.12)
    axes[1].set_xlim(0, max(all_memory)*1.12)
    spread = max(all_mse)-min(all_mse)
    axes[0].set_ylim(min(all_mse)-max(0.06, spread*.12), max(all_mse)+max(0.06, spread*.15))
    for axis in axes:
        axis.grid(color='#e6e6e6', linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
    drift = cost['drift_checks']
    disjoint = sum(item['iqr_disjoint'] for item in drift['f0_pre_post'])
    reversals = len(drift['pair_order_reversals'])
    fig.text(0.53, 0.055,
             'Hollow: seed 92601 (circle), seed 92602 (triangle), deterministic F0 (square).\n'
             'Diamond: family center; small points / horizontal span: all three block medians / range.\n'
             f'Drift checks: {disjoint} disjoint sentinel IQRs; {reversals} pairwise rank reversals. No stable speed claim.',
             ha='center', fontsize=7.5, linespacing=1.35)
    return save_figure(fig, directory, 'hog_accuracy_cost', {
        'title': 'Hog Accuracy–Cost Trade-offs After LEVEL Recovery',
        'caption': 'Hog combined development MSE and same-session batch-4 deployment cost for the exact same stored model identities. Hollow circles and triangles identify trained seeds; the square is deterministic F0. Small points and horizontal spans retain all three block medians and their observed range; they are not confidence intervals. Diamonds use mean individual-seed MSE and the mean of per-seed median block costs. F0 post sentinels are excluded from the center and retained in the cost JSON and drift summary. Timing covers standardized CPU input, H2D, full forward, and complete CPU forecast return on the same local GPU. Peak allocated GPU memory is distinct from reserved memory, process GPU memory, model size and parameter bytes. Large block drift is retained; no stable speed advantage is inferred from a center. These are exposed development results, not an independent confirmation.',
        'sources': sources, 'numeric_values': numeric, 'family_centers': centers,
        'calculations': {'accuracy': 'Stored combined MSE for each exact run; family center is mean seed loss, not prediction ensemble.',
                         'cost': 'Median of three stored block medians per fit, followed by mean over trained seeds; deterministic F0 counted once.',
                         'memory_units': 'MiB = bytes / 1024**2',
                         'error_bar': 'Observed minimum and maximum of the three block medians, not a confidence interval.'},
        'drift_checks': drift,
        'measurement_contract': {key: cost.get(key) for key in ['device', 'scope', 'precision', 'torch_threads',
                                                              'warmup_calls', 'passes_per_order_block', 'order_blocks',
                                                              'single_uninterrupted_session']},
        'prohibited_claim': 'A stable speedup, equivalent accuracy, overall GPU memory reduction, pretraining-disjoint confirmation, or a timing ratio against prior v4 measurements.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='v5_figures_01')
    parser.add_argument('--cost', default='cost01.json')
    args = parser.parse_args()
    if Path(args.cost).name != args.cost or not args.cost.endswith('.json'):
        raise ValueError('--cost must identify a completed JSON in this v5 folder')
    directory = HERE / 'figures'
    if (directory / 'manifest.json').exists():
        raise FileExistsError('Existing figure manifest must be inspected before a new render')
    paths = {dataset: HERE / f'{dataset}_level_01.json' for dataset in ['hog', 'bull', 'electricity']}
    cost_path = HERE / args.cost
    with Job(args.name, category='cpu_analysis', reserve_seconds=60) as job:
        reports = {dataset: read(path) for dataset, path in paths.items()}
        for dataset, report in reports.items():
            if report['dataset'] != dataset or report['evaluation_scope'] != 'exposed development':
                raise ValueError('Dataset/scope mismatch')
        cost = read(cost_path)
        if cost['method'] != 'level':
            raise ValueError('This fixed summary is for the completed LEVEL development only')
        style()
        directory.mkdir(parents=True, exist_ok=True)
        sources = [receipt(path) for path in paths.values()]
        figures = [relative_accuracy(reports, sources, directory)]
        job.heartbeat('Saved three-dataset accuracy comparison; render same-session Hog cost trade-offs')
        figures.append(accuracy_cost(reports['hog'], cost, [receipt(paths['hog']), receipt(cost_path)], directory))
        manifest = {'status': 'complete', 'name': args.name, 'source_code': receipt(Path(__file__)),
                    'scope': 'Only existing completed JSON arithmetic and rendering; no model/array loads, inference, scoring, fitting or benchmark.',
                    'figures': figures, 'sources': sources + [receipt(cost_path)]}
        save_json(directory / 'manifest.json', manifest)
        print(json.dumps({'manifest': str(directory / 'manifest.json'), 'figures': len(figures)}), flush=True)


if __name__ == '__main__':
    main()
