"""Plot completed v3 decision evidence without loading models or predictions."""
import argparse
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import sys

from runtime import HERE, Job, digest


DATASETS = ('electricity', 'bull')
SEEDS = (92601, 92602)
COLORS = {'electricity': '#0072B2', 'bull': '#D55E00'}
LABELS = {'electricity': 'Electricity', 'bull': 'Bull'}
FAMILY_LABELS = {'f0': 'F0', 'merged_lora': 'LoRA\n(merged)', 'tsfm_res': 'TSFM_RES',
                 'linear_res': 'LINEAR_RES', 'shared_linear': 'SHARED', 'factor_linear': 'FACTOR'}


def read_completed(path, stage=None, rows=None):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('status') != 'complete':
        raise ValueError(f'Only completed results may be plotted: {path}')
    if stage is not None and value.get('stage') != stage:
        raise ValueError(f'Expected {stage} cost stage: {path}')
    if rows is not None and len(value['rows']) != rows:
        raise ValueError(f'Expected {rows} complete cost rows: {path}')
    return value


def accuracy_points(development):
    if development['comparison_role'] != 'selected_tsfm_residual':
        raise ValueError('This figure requires the selected LINEAR_RES versus TSFM_RES comparison')
    points = []
    for dataset, period in (('electricity', 'dev'), ('bull', 'e1'), ('bull', 'e2')):
        row = development['datasets'][dataset]['periods'][period]
        comparison = row['comparison_linear_vs_reference']
        if comparison['status'] != 'available':
            raise ValueError(f'Missing main comparison: {dataset}/{period}')
        linear = {r['seed']: r for r in row['linear_predictions']}
        reference = {r['seed']: r for r in row['reused_predictions']
                     if r['role'] == 'selected_tsfm_residual'}
        if set(linear) != set(SEEDS) or set(reference) != set(SEEDS):
            raise ValueError('Main comparison must contain the two matched trained seeds')
        if any(r['arm'] != 'residual' for r in linear.values()):
            raise ValueError('LINEAR_RAW is not the main backbone-value comparison')
        seed_points = []
        for seed in SEEDS:
            left, right = linear[seed], reference[seed]
            denominator = right['scores']['mse']
            relative = 100 * (left['scores']['mse'] - denominator) / denominator
            seed_points.append({'seed': seed, 'relative_to_reference_pct': relative,
                                'linear_fit': left['fit'], 'reference_fit': right['fit'],
                                'linear_mse': left['scores']['mse'], 'reference_mse': denominator})
        points.append({'dataset': dataset, 'period': period, 'origins': len(row['origins']),
                       'target_shape': row['target_shape'],
                       'mean_relative_pct': comparison['full']['relative_to_reference_pct'],
                       'conditional_block95_pct': comparison['conditional_block95_relative_to_reference_pct'],
                       'full': comparison['full'], 'block_origins': comparison['block_origins'],
                       'draws': comparison['draws'], 'interval_scope': comparison['scope'],
                       'seed_points': seed_points})
    return points


def cost_points(cost, families, scope):
    points = []
    for index, row in enumerate(cost['rows']):
        if (row['family'] in families and row['scope'] == scope and row['batch_origins'] == 1
                and row.get('include_in_primary', True)):
            points.append({'source_row_index': index, **{key: row.get(key) for key in
                           ('id', 'dataset', 'family', 'seed', 'ed_mode', 'latent', 'block',
                            'scope', 'batch_origins', 'sentinel_role', 'order_position')},
                           'median_ms_per_origin': row['milliseconds_per_origin']['median'],
                           'within_block_q25_ms': row['milliseconds_per_origin']['q25'],
                           'within_block_q75_ms': row['milliseconds_per_origin']['q75']})
    centers = []
    for dataset in DATASETS:
        for family in families:
            members = [p for p in points if (p['dataset'], p['family']) == (dataset, family)]
            by_fit = defaultdict(list)
            for point in members:
                if not math.isfinite(point['median_ms_per_origin']) or point['median_ms_per_origin'] <= 0:
                    raise ValueError('Log timing plot requires finite positive recorded medians')
                by_fit[point['id']].append(point)
            expected_fits = 2 if family in ('merged_lora', 'tsfm_res', 'linear_res') else 1
            if len(by_fit) != expected_fits:
                raise ValueError(f'Incomplete cost configurations: {dataset}/{family}')
            for fit, blocks in by_fit.items():
                if len(blocks) != 3 or {p['block'] for p in blocks} != {0, 1, 2}:
                    raise ValueError(f'Expected three recorded order blocks: {fit}')
            summaries = [s for s in cost['summary']['groups'] if
                         (s['dataset'], s['family'], s['scope'], s['batch_origins']) ==
                         (dataset, family, scope, 1)]
            if len(summaries) != 1 or set(summaries[0]['fits']) != set(by_fit):
                raise ValueError(f'Ambiguous recorded group summary: {dataset}/{family}')
            centers.append({'dataset': dataset, 'family': family,
                            'mean_fit_median_ms_per_origin': summaries[0]['mean_fit_median_ms_per_origin'],
                            'fit_ids': summaries[0]['fits']})
    return points, centers


def sentinel_points(gpu):
    points = []
    for index, row in enumerate(gpu['rows']):
        if row['family'] != 'f0' or row['batch_origins'] != 1:
            continue
        if row['scope'] not in ('deployment_cpu_to_cpu', 'gpu_resident'):
            continue
        key = ('milliseconds_per_origin' if row['scope'] == 'deployment_cpu_to_cpu'
               else 'cuda_event_milliseconds_per_origin')
        points.append({'source_row_index': index, **{name: row[name] for name in
                       ('dataset', 'block', 'scope', 'sentinel_role', 'id', 'order_position')},
                       'timing_field': key, 'median_ms_per_origin': row[key]['median'],
                       'within_block_q25_ms': row[key]['q25'], 'within_block_q75_ms': row[key]['q75']})
    expected = {(d, b, s, r) for d in DATASETS for b in range(3)
                for s in ('deployment_cpu_to_cpu', 'gpu_resident') for r in ('pre', 'post')}
    keys = {(r['dataset'], r['block'], r['scope'], r['sentinel_role']) for r in points}
    if len(points) != 24 or keys != expected:
        raise ValueError('F0 sentinel panel requires every pre/post placement in all three blocks')
    return points


def plot_accuracy(ax, points):
    ax.axvline(0, color='#888888', linewidth=0.8, zorder=0)
    for y, point in enumerate(points):
        color = COLORS[point['dataset']]
        lo, hi = point['conditional_block95_pct']
        ax.hlines(y, lo, hi, color=color, linewidth=2)
        ax.vlines([lo, hi], y - 0.07, y + 0.07, color=color, linewidth=1.2)
        ax.scatter(point['mean_relative_pct'], y, marker='D', color=color, s=43, zorder=3)
        for number, seed in enumerate(point['seed_points']):
            ax.scatter(seed['relative_to_reference_pct'], y + (number * 2 - 1) * 0.17,
                       marker='o', s=20, edgecolors=color,
                       facecolors='white' if number == 0 else color, linewidths=0.9, zorder=4)
    ax.set_yticks(range(3), [f'{LABELS[p["dataset"]]} {p["period"].upper()}\n'
                           f'n={p["origins"]} origins' for p in points])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel('100 × (LINEAR_RES MSE − TSFM_RES MSE) / TSFM_RES MSE (%)\n'
                  'Positive: lower error with TSFM_RES', fontsize=8.6)
    ax.set_title('A  Backbone value on exposed development periods\n'
                 '2 fixed seeds; conditional 95% intervals, 7-origin blocks', loc='left', fontsize=10)
    ax.grid(axis='x', color='#DDDDDD', linewidth=0.5, zorder=0)
    ax.margins(x=0.08)


def plot_cost(ax, points, centers, families, title, subtitle):
    for family_number, family in enumerate(families):
        for dataset_number, dataset in enumerate(DATASETS):
            x = family_number + (-0.18 if dataset_number == 0 else 0.18)
            color = COLORS[dataset]
            members = [p for p in points if (p['dataset'], p['family']) == (dataset, family)]
            for point in members:
                seed_offset = -0.025 if point['seed'] == SEEDS[0] else 0.025 if point['seed'] == SEEDS[1] else 0
                block_offset = (point['block'] - 1) * 0.055
                ax.scatter(x + seed_offset + block_offset, point['median_ms_per_origin'], s=23,
                           marker=('o', 's', '^')[point['block']], edgecolors=color,
                           facecolors='white' if point['seed'] == SEEDS[0] else color,
                           linewidths=0.9, alpha=0.85, zorder=3)
            center = next(p for p in centers if (p['dataset'], p['family']) == (dataset, family))
            ax.hlines(center['mean_fit_median_ms_per_origin'], x - 0.12, x + 0.12,
                      color=color, linewidth=2.0, zorder=4)
    ax.set_yscale('log')
    ax.set_xticks(range(len(families)), [FAMILY_LABELS[f] for f in families], fontsize=9)
    ax.set_xlim(-0.55, len(families) - 0.45)
    ax.set_ylabel('Latency (ms/origin; log scale)', fontsize=9)
    ax.set_title(f'{title}\n{subtitle}', loc='left', fontsize=10)
    ax.grid(axis='y', which='major', color='#DDDDDD', linewidth=0.5, zorder=0)
    ax.margins(y=0.16)


def plot_sentinels(ax, points):
    for dataset_number, dataset in enumerate(DATASETS):
        for scope in ('deployment_cpu_to_cpu', 'gpu_resident'):
            for block in range(3):
                members = {p['sentinel_role']: p for p in points if
                           (p['dataset'], p['scope'], p['block']) == (dataset, scope, block)}
                offset = -0.045 if dataset_number == 0 else 0.045
                x = [3 * block + offset, 3 * block + 1 + offset]
                y = [members[r]['median_ms_per_origin'] for r in ('pre', 'post')]
                ax.plot(x, y, color=COLORS[dataset], linewidth=1.3,
                        linestyle='-' if scope == 'deployment_cpu_to_cpu' else '--',
                        marker='o' if scope == 'deployment_cpu_to_cpu' else 's', markersize=4,
                        markerfacecolor=COLORS[dataset] if scope == 'deployment_cpu_to_cpu' else 'white')
    ax.set_xticks([0, 1, 3, 4, 6, 7], ['1 / pre', '1 / post', '2 / pre', '2 / post', '3 / pre', '3 / post'],
                  fontsize=8)
    ax.set_xlabel('Order block / F0 placement', fontsize=9)
    ax.set_ylabel('Latency (ms/origin; log scale)', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('D  F0 drift sentinels; batch = 1; n=3 order blocks\n'
                 '20 passes/point; resident event is not pure kernel time', loc='left', fontsize=10)
    ax.grid(axis='y', which='major', color='#DDDDDD', linewidth=0.5, zorder=0)
    ax.margins(x=0.06, y=0.16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=HERE / 'final_development.json')
    parser.add_argument('--gpu-cost', type=Path, default=HERE / 'cost_gpu_01.json')
    parser.add_argument('--cpu-cost', type=Path, default=HERE / 'cost_cpu_01.json')
    parser.add_argument('--output', type=Path, default=HERE / 'comparison', help='Output prefix, without suffix')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix:
        raise ValueError('--output must be a prefix without a file extension')
    paths = {kind: Path(str(output) + suffix) for kind, suffix in
             (('png', '.png'), ('svg', '.svg'), ('caption', '.caption.json'))}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Preserve existing figures/captions; choose a new output prefix')
    with Job(f'plot_{output.name}', category='cpu_analysis', reserve_seconds=60) as job:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        development = read_completed(args.development)
        gpu = read_completed(args.gpu_cost, stage='gpu', rows=192)
        cpu = read_completed(args.cpu_cost, stage='cpu', rows=48)
        selection_sha = development['selected_linear']['sha256']
        if any(c['selection']['sha256'] != selection_sha for c in (gpu, cpu)):
            raise ValueError('Accuracy and timing must use the same frozen LINEAR selection')
        accuracy = accuracy_points(development)
        gpu_families = ('f0', 'merged_lora', 'tsfm_res', 'linear_res')
        cpu_families = ('linear_res', 'shared_linear', 'factor_linear')
        gpu_points, gpu_centers = cost_points(gpu, gpu_families, 'deployment_cpu_to_cpu')
        cpu_points, cpu_centers = cost_points(cpu, cpu_families, 'cpu_fp32')
        sentinels = sentinel_points(gpu)
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                             'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                             'ytick.labelsize': 8, 'axes.spines.top': False,
                             'axes.spines.right': False, 'svg.fonttype': 'none',
                             'axes.axisbelow': True, 'savefig.facecolor': 'white'})
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.83, hspace=0.78, wspace=0.32)
        fig.suptitle('TSFM backbone value: development accuracy and deployment cost', y=0.975,
                     fontsize=14, fontweight='bold')
        handles = [Line2D([], [], color=COLORS[d], marker='o', linestyle='none', label=LABELS[d]) for d in DATASETS]
        handles += [Line2D([], [], color='#555555', linestyle='-', label='D: deployment wall'),
                    Line2D([], [], color='#555555', linestyle='--', label='D: resident CUDA event')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.942), ncol=4,
                   frameon=False, fontsize=9, columnspacing=2)
        fig.text(0.5, 0.895, 'All evaluation periods were exposed during development; no independent confirmation.',
                 ha='center', fontsize=9, color='#555555')
        plot_accuracy(axes[0, 0], accuracy)
        plot_cost(axes[0, 1], gpu_points, gpu_centers, gpu_families,
                  'B  GPU deployment; CPU input → full CPU output; batch = 1',
                  '24 origins/pass; 3 blocks × 20 passes; 2 seeds (F0: 1 model)')
        plot_cost(axes[1, 0], cpu_points, cpu_centers, cpu_families,
                  'C  CPU deployment; FP32, 4 threads; batch = 1',
                  '3 blocks × 20 passes; LINEAR: 2 seeds; SHARED/FACTOR: 1 fit')
        plot_sentinels(axes[1, 1], sentinels)
        fig.text(0.11, 0.111, 'A: diamonds = relative difference of 2-seed mean MSEs; small circles = matched seed differences. '
                 'Whiskers = stored conditional block 95% interval.', fontsize=8.4)
        fig.text(0.11, 0.085, 'B/C: each point = one 20-pass block median (circle/square/triangle = blocks 1/2/3); '
                 'open/filled = seeds 92601/92602; single fits are filled.', fontsize=8.4)
        fig.text(0.11, 0.059, 'B/C: bars = mean of each fit’s 3-block median; repeated timing points are not independent '
                 'replicates. D: no drift correction or excluded sentinels.', fontsize=8.4)
        caption = {
            'schema_version': 1, 'status': 'complete', 'command': sys.argv,
            'sources': {label: {'path': str(path.resolve()), 'sha256': digest(path)} for label, path in
                        (('development', args.development), ('gpu_cost', args.gpu_cost), ('cpu_cost', args.cpu_cost))},
            'script': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__))},
            'selected_linear': development['selected_linear'],
            'figure': {'inches': [12, 8.5], 'png_dpi': 300, 'palette': COLORS,
                       'font': 'Arial with DejaVu Sans fallback', 'new_fits': 0,
                       'new_predictions': 0, 'new_bootstrap_draws': 0, 'significance_tests': 0},
            'panels': {
                'A': {'plotted_comparisons': accuracy,
                      'formula': '100*(LINEAR_RES MSE - TSFM_RES MSE)/TSFM_RES MSE; positive favors TSFM_RES',
                      'interval': 'Stored 95% non-circular time-block interval; trained seeds fixed, all channels move together. '
                                  'No interval is recomputed; no independent confirmation is claimed.',
                      'seed_delta_formula': 'Same percentage formula applied to each recorded matched-seed pair of MSEs.'},
                'B': {'source': 'gpu_cost', 'selector': {'scope': 'deployment_cpu_to_cpu', 'batch_origins': 1,
                      'include_in_primary': True, 'families': list(gpu_families)},
                      'raw_row_points': gpu_points, 'recorded_group_centers': gpu_centers,
                      'summary_definition': gpu['summary']['definition']},
                'C': {'source': 'cpu_cost', 'selector': {'scope': 'cpu_fp32', 'batch_origins': 1,
                      'include_in_primary': True, 'families': list(cpu_families)},
                      'raw_row_points': cpu_points, 'recorded_group_centers': cpu_centers,
                      'summary_definition': cpu['summary']['definition']},
                'D': {'source': 'gpu_cost', 'selector': {'family': 'f0', 'batch_origins': 1,
                      'sentinel_roles': ['pre', 'post']}, 'raw_row_points': sentinels,
                      'interpretation': 'Deployment synchronized wall versus resident CUDA-event interval; '
                                        'events can include CPU-dispatch idle gaps. Differences are not transfer-only estimates.'}},
            'timing_scope': {'gpu': gpu['timed_scopes'], 'cpu': cpu['timed_scopes'],
                             'device': gpu.get('device'), 'gpu_uncertainty': gpu['uncertainty'],
                             'cpu_uncertainty': cpu['uncertainty']},
            'caption': 'Exposed-development accuracy and batch-one inference latency. A uses existing channel-macro '
                       'MSE and fixed-seed conditional block intervals for Electricity DEV and Bull E1/E2. '
                       'B/C show all selected order-block medians, not confidence intervals from repeated timing. '
                       'F0 post-sentinels appear only in D and are excluded from B group centers. '
                       'The three timing blocks share one local device/session. '
                       'Shapes and fill provide redundant encoding beyond the colorblind-safe dataset palette.'}
        output.parent.mkdir(parents=True, exist_ok=True)
        job.heartbeat('Figure composed from completed JSON; exporting PNG/SVG and auditable caption')
        for kind in ('png', 'svg'):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=kind, dpi=300)
            with paths[kind].open('xb') as handle:
                handle.write(buffer.getvalue())
        plt.close(fig)
        caption['outputs'] = {kind: {'path': str(paths[kind]), 'sha256': digest(paths[kind])} for kind in ('png', 'svg')}
        with paths['caption'].open('x', encoding='utf-8') as handle:
            json.dump(caption, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        job.heartbeat('Figure export complete; no model or prediction arrays loaded')
        print(json.dumps({'outputs': {key: str(path) for key, path in paths.items()}}), flush=True)


if __name__ == '__main__':
    main()
