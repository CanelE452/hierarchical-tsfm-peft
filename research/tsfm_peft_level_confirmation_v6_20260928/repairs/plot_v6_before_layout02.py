"""Render completed v6 JSON records; never load a model or prediction array."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from runtime_v6 import HERE, ROOT, Job, digest, save_json


ROLES = ('level_res', 'old_res', 'level_raw', 'level_only', 'compress',
         'f0', 'lora', 'level_linear_res', 'shared')
GPU_ROLES = ('level_res', 'old_res', 'level_raw', 'level_only',
             'f0', 'lora', 'level_linear_res')
CPU_ROLES = ('level_linear_res', 'shared')
PERIODS = ('combined', 'test_a', 'test_b')
LABELS = {'level_res': 'LEVEL_RES', 'old_res': 'OLD_RES', 'level_raw': 'LEVEL_RAW',
          'level_only': 'LEVEL_ONLY', 'compress': 'COMPRESS', 'f0': 'F0',
          'lora': 'Merged LoRA', 'level_linear_res': 'LEVEL_LINEAR_RES',
          'shared': 'SHARED ridge'}
COLORS = {'level_res': '#0072B2', 'old_res': '#777777', 'level_raw': '#D55E00',
          'level_only': '#CC79A7', 'compress': '#D4A017', 'f0': '#222222',
          'lora': '#009E73', 'level_linear_res': '#7B61A8', 'shared': '#56B4E9'}
SEED_MARKERS = {92601: 'o', 92602: '^', None: 's'}


def read_completed(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('status') != 'complete':
        raise ValueError(f'Only completed measurement JSON may be plotted: {path}')
    return value


def receipt(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def finite(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Do not silently plot a nonfinite metric')
    return value


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'figure.titlesize': 11, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'svg.fonttype': 'none',
        'svg.hashsalt': 'v6_level_confirmation', 'pdf.fonttype': 42,
    })


def save_figure(fig, directory, stem, metadata):
    fig.canvas.draw()
    texts = [item.get_text() for item in fig.findobj(matplotlib.text.Text) if item.get_text()]
    if any('\uac00' <= character <= '\ud7a3' for text in texts for character in text):
        raise ValueError('Publication figures require English text')
    for extension in ('png', 'svg'):
        path = directory / f'{stem}.{extension}'
        if path.exists():
            raise FileExistsError('Preserve and inspect prior figures before a new render')
        options = {'metadata': {'Date': None}} if extension == 'svg' else {}
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.08, **options)
    size = list(fig.get_size_inches())
    plt.close(fig)
    metadata.update(stem=stem, figure_text=texts, png_dpi=300, figure_size_inches=size)
    metadata['outputs'] = [receipt(directory / f'{stem}.{ext}') for ext in ('png', 'svg')]
    save_json(directory / f'{stem}.source.json', metadata)
    return metadata


def method_figure(directory):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.16, top=0.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def box(x, y, width, height, text, color='#f1f3f5'):
        patch = FancyBboxPatch((x, y), width, height,
                              boxstyle='round,pad=0.08', edgecolor='#46515a',
                              linewidth=0.8, facecolor=color)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=8)

    def arrow(start, end, color='#46515a'):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                    mutation_scale=10, linewidth=0.9, color=color))

    box(0.15, 2.55, 1.25, 0.9, 'Past X\n512 x 17')
    box(1.9, 2.55, 1.25, 0.9, 'Fixed E\nTRAIN PCA\n17 -> 5')
    box(3.7, 2.55, 1.45, 0.9, 'Frozen F0\n5 latent series')
    box(5.7, 2.55, 1.25, 0.9, 'Fixed D\n5 -> 17')
    box(8.2, 2.55, 1.55, 0.9, 'Forecast\nb + level + G')
    for start, end in [((1.4, 3), (1.9, 3)), ((3.15, 3), (3.7, 3)),
                       ((5.15, 3), (5.7, 3)), ((6.95, 3), (8.2, 3))]:
        arrow(start, end)
    ax.text(7.55, 3.12, 'b', ha='center')
    box(1.9, 0.85, 1.55, 0.85, 'r = X - D(E(X))\nPast residual')
    box(4.0, 1.18, 1.45, 0.75, 'Subtract last(r)\nDetached level')
    box(5.95, 1.18, 1.55, 0.75, 'Train G only\n512 -> 32 -> 48', '#e5f1f8')
    arrow((0.8, 2.55), (2.35, 1.7))
    arrow((3.45, 1.35), (4.0, 1.55))
    arrow((5.45, 1.55), (5.95, 1.55))
    arrow((7.5, 1.55), (8.75, 2.55))
    ax.plot([2.65, 2.65, 7.9, 7.9], [0.85, 0.65, 0.65, 2.3],
            color='#B05B88', linewidth=0.9)
    arrow((7.9, 2.3), (8.2, 2.7), '#B05B88')
    ax.text(6.1, 0.9, 'Repeat last(r) for 48 steps', color='#8B3F6E', ha='center', fontsize=8)
    ax.text(5, 0.13, 'Matched RAW: same residual level; G receives X - last(X).\n'
            'OLD_RES: b + G(r).  LEVEL_ONLY: b + last(r).  COMPRESS: b.',
            ha='center', va='center', fontsize=8, linespacing=1.6)
    fig.suptitle('Fixed LEVEL Residual Adapter and Matched Controls', y=0.98)
    fig.text(0.5, 0.02, 'Robin: C = 17, K = 5; frozen PCA and Chronos-Bolt-small; '
             '17,920 learned G parameters.\nG is shared, bias-free and linear; '
             'zero output initialization starts at LEVEL_ONLY, not COMPRESS.',
             ha='center', fontsize=8, linespacing=1.5)
    return save_figure(fig, directory, 'method', {
        'title': 'Fixed LEVEL Residual Adapter and Matched Controls',
        'caption': 'The frozen TSFM forecasts five PCA latent series, while the learned shared temporal '
                   'map predicts changes in the observed channel-reconstruction residual. The same '
                   'residual last-level term is restored in LEVEL_RES and LEVEL_RAW; RAW centers its '
                   'original input instead. Only G is learned in these fixed-PCA variants. Arrows show '
                   'information flow, not numerical attribution of forecast accuracy. LEVEL_LINEAR_RES '
                   'substitutes a learned normalized time-linear main predictor; its deployment has no F0.',
        'sources': [receipt(HERE / p) for p in ('PLAN.md', 'protocol.json', 'model_v6.py')],
        'formula': {'level_res': 'b + repeat(last(r), H) + G(r - last(r))',
                    'level_raw': 'b + repeat(last(r), H) + G(X - last(X))',
                    'old_res': 'b + G(r)', 'level_only': 'b + repeat(last(r), H)',
                    'compress': 'b', 'b': 'D(F0(E(X)))', 'r': 'X - D(E(X))'},
        'scope': 'Fixed architecture diagram; no empirical mechanism or novelty proof.'})


def accuracy_figure(report, source, directory):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 7.4), sharex='row', sharey=True)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.14, top=0.865,
                        hspace=0.35, wspace=0.20)
    fig.suptitle('Robin Fixed Comparison: Accuracy by Period', y=0.985)
    handles = [Line2D([], [], marker=mark, color='#333333', linestyle='none',
                      markersize=5, markerfacecolor='white' if mark != 'D' else '#333333', label=label)
               for mark, label in [('D', 'Mean seed loss'), ('o', 'Seed 92601'),
                                   ('^', 'Seed 92602'), ('s', 'Deterministic reference')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.56, 0.948),
               ncol=4, frameon=False, handletextpad=0.4, columnspacing=1.2)
    numeric = []
    for metric_index, metric in enumerate(('mse', 'mae')):
        for period_index, period in enumerate(PERIODS):
            ax = axes[metric_index, period_index]
            for y, role in enumerate(ROLES):
                center = finite(report['role_scores'][role][period][metric])
                seeds = []
                for identifier in report['roles'][role]:
                    model = report['models'][identifier]
                    seed = model['spec'].get('seed')
                    value = finite(model['scores'][period][metric])
                    seeds.append({'id': identifier, 'seed': seed, 'value': value,
                                  'channel_target_count': model['scores'][period]['channel_target_count']})
                    if value is not None:
                        offset = -0.15 if seed == 92601 else 0.15 if seed == 92602 else 0
                        ax.scatter(value, y + offset, marker=SEED_MARKERS[seed],
                                   s=27, facecolor='white', edgecolor=COLORS[role],
                                   linewidth=1.05, zorder=3)
                available = [item['value'] for item in seeds if item['value'] is not None]
                if center is None:
                    ax.text(0.02, y, 'N/A', color=COLORS[role], va='center',
                            transform=ax.get_yaxis_transform())
                else:
                    if len(available) != len(seeds) or abs(mean(available) - center) > 1e-12:
                        raise ValueError('Stored role score must equal mean individual-model losses')
                    ax.scatter(center, y, color=COLORS[role], marker='D', s=28,
                               edgecolor='black', linewidth=0.4, zorder=4)
                numeric.append({'role': role, 'period': period, 'metric': metric,
                                'center': center, 'members': seeds,
                                'pointer': f'role_scores.{role}.{period}.{metric}'})
            ax.set_yticks(range(len(ROLES)), [LABELS[role] for role in ROLES])
            ax.set_ylim(len(ROLES) - 0.4, -0.6)
            ax.set_xlim(left=0)
            ax.set_xlabel(f'TRAIN-standardized macro {metric.upper()} (lower is better)')
            ax.set_title({'combined': 'Combined TEST-A/B', 'test_a': 'TEST-A', 'test_b': 'TEST-B'}[period])
            ax.grid(axis='x', color='#e5e5e5', linewidth=0.55)
            ax.set_axisbelow(True)
    fig.text(0.57, 0.035, 'Combined scores pool channel error sums / observed counts before channel and seed means.\n'
             'Both periods and every planned method are shown. Points are descriptive, not confidence intervals.\n'
             'A mean of seed losses is not a prediction ensemble; the two periods are one site.',
             ha='center', fontsize=8, linespacing=1.5)
    return save_figure(fig, directory, 'robin_accuracy', {
        'title': 'Robin Fixed Comparison: Accuracy by Period',
        'caption': 'TRAIN-standardized channel-macro MSE and MAE on the combined Robin evaluation '
                   'and both fixed periods. Diamonds show the mean of individual model losses; hollow '
                   'circles/triangles identify seeds and squares identify deterministic references. '
                   'Combined losses pool channel error sums and observed target counts across periods '
                   'before channel and seed averaging. Horizontal scales are shared within each metric '
                   'row; no unfavorable method or period is removed. No timing or cross-site average '
                   'is used in this accuracy figure. Conditional block intervals remain in the source JSON.',
        'sources': [source], 'numeric_values': numeric,
        'combined_definition': report['combined_definition'],
        'uncertainty': 'Seed points are not independent confidence intervals; paired time-block intervals are in comparisons[].periods.',
        'exposure': report.get('first_exposure'), 'correction_reason': report.get('correction_reason'),
        'source_comparisons': report['comparisons']})


def matched_cost_points(report, cost, roles, scope):
    rows = [r for r in cost['rows'] if r['include_in_primary'] and r['batch_origins'] == 4]
    expected = {i for role in roles for i in report['roles'][role]}
    if {row['id'] for row in rows} != expected or any(row['scope'] != scope for row in rows):
        raise ValueError('Cost rows must cover exactly the approved methods in the stated deployment scope')
    points = []
    for role in roles:
        for identifier in report['roles'][role]:
            blocks = sorted((r for r in rows if r['id'] == identifier), key=lambda r: r['block'])
            if [row['block'] for row in blocks] != [0, 1, 2]:
                raise ValueError('All three cost blocks are required; no favorable-block selection')
            if any(row['family'] != role for row in blocks):
                raise ValueError('Cost family disagrees with evaluated role')
            model = report['models'][identifier]
            values = [finite(row['origins_per_second']['median']) for row in blocks]
            if any(value is None or value <= 0 for value in values):
                raise ValueError('Throughput must be finite and positive')
            memory = ([finite(row['peak_allocated_bytes']) / 1024**2 for row in blocks]
                      if scope == 'deployment_cpu_to_cpu' else None)
            if memory is not None and any(value <= 0 for value in memory):
                raise ValueError('GPU allocated-memory observations must be positive for log axes')
            points.append({'id': identifier, 'role': role, 'seed': model['spec'].get('seed'),
                           'mse': finite(model['scores']['combined']['mse']),
                           'throughput': values, 'throughput_center': median(values),
                           'allocated_mib': memory, 'allocated_center_mib': median(memory) if memory else None,
                           'source_peak_allocated_bytes': [row.get('peak_allocated_bytes') for row in blocks],
                           'block_ids': [0, 1, 2],
                           'measurement_runs': [row['measurement_run'] for row in blocks],
                           'accuracy_pointer': f'models.{identifier}.scores.combined.mse',
                           'cost_pointer': f'rows[id={identifier},batch_origins=4,include_in_primary=true]'})
    return points


def cost_figure(report, cost, roles, sources, directory, cpu=False):
    scope = 'cpu_fp32' if cpu else 'deployment_cpu_to_cpu'
    points = matched_cost_points(report, cost, roles, scope)
    fig, axes = plt.subplots(1, 1 if cpu else 2, figsize=(6.5, 4.3) if cpu else (9, 5.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.27, top=0.72 if not cpu else 0.77,
                        wspace=0.24)
    title = 'Robin Accuracy and CPU Deployment' if cpu else 'Robin Accuracy and GPU Deployment'
    fig.suptitle(title, y=0.985)
    handles = [Line2D([], [], marker='D', color=COLORS[role], linestyle='none',
                      markersize=5, label=LABELS[role]) for role in roles]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.925),
               ncol=2 if cpu else 3, frameon=False, columnspacing=1.4, handletextpad=0.4)
    centers, unavailable = [], []
    for role in roles:
        members = [p for p in points if p['role'] == role]
        available = [p for p in members if p['mse'] is not None]
        if len(available) != len(members):
            unavailable.append(role)
        for point in available:
            observations = [point['throughput']] if cpu else [point['throughput'], point['allocated_mib']]
            for axis, values in zip(axes, observations):
                axis.plot([min(values), max(values)], [point['mse']] * 2,
                          color=COLORS[role], linewidth=0.9, alpha=0.8)
                axis.scatter(values, [point['mse']] * 3, color=COLORS[role], s=11,
                             marker=SEED_MARKERS[point['seed']], alpha=0.5,
                             edgecolor='black', linewidth=0.3)
                axis.scatter(median(values), point['mse'], facecolor='white',
                             edgecolor=COLORS[role], s=30, linewidth=1,
                             marker=SEED_MARKERS[point['seed']], zorder=4)
        if len(available) == len(members):
            center = {'role': role, 'mse': mean(p['mse'] for p in members),
                      'throughput': mean(p['throughput_center'] for p in members),
                      'allocated_mib': None if cpu else mean(p['allocated_center_mib'] for p in members)}
            expected = finite(report['role_scores'][role]['combined']['mse'])
            if expected is None or abs(center['mse'] - expected) > 1e-12:
                raise ValueError('Accuracy/cost family mean mismatch')
            xvalues = [center['throughput']] if cpu else [center['throughput'], center['allocated_mib']]
            for axis, x in zip(axes, xvalues):
                axis.scatter(x, center['mse'], marker='D', s=47, color=COLORS[role],
                             edgecolor='black', linewidth=0.5, zorder=5)
            centers.append(center)
    axes[0].set_xlabel('Batch-4 throughput (origins/s; log scale)')
    axes[0].set_title('Lower-right is better', fontsize=8)
    axes[0].set_ylabel('Combined TRAIN-standardized macro MSE')
    if not cpu:
        axes[1].set_xlabel('Peak allocated GPU memory (MiB; log scale)')
        axes[1].set_title('Lower-left is better', fontsize=8)
    for axis in axes:
        axis.set_xscale('log')
        axis.set_ylim(bottom=0)
        axis.margins(x=0.16, y=0.12)
        axis.grid(color='#e5e5e5', linewidth=0.55)
        axis.set_axisbelow(True)
    drift = cost.get('drift_checks')
    foot = ('CPU FP32 alternatives only; GPU memory is not a CPU cost measure.' if cpu else
            f"GPU: {cost.get('device', 'recorded device')}. Sentinel IQRs disjoint: "
            f"{sum(item['iqr_disjoint'] for item in drift['f0_pre_post'])}/6; "
            f"pair order reversals: {len(drift['pair_order_reversals'])}.")
    fig.text(0.55, 0.045,
             'Diamond: mean seed loss and mean per-seed median block cost.\n'
             'Hollow: individual seed; small points / spans: three block medians / range.\n' + foot +
             ('\nMSE unavailable (not omitted silently): ' + ', '.join(unavailable) if unavailable else ''),
             ha='center', fontsize=7.5, linespacing=1.45)
    return save_figure(fig, directory, 'robin_cpu_accuracy_cost' if cpu else 'robin_gpu_accuracy_cost', {
        'title': title,
        'caption': 'Robin combined MSE and batch-4 cost for identical saved model identities. '
                   'Per-fit costs use the median of three block medians, then family centers average '
                   'the individual seed values. Diamonds are family centers; hollow seed markers and '
                   'three block points/ranges retain measured variation. Spans are not confidence '
                   'intervals. Logarithmic cost axes display all approved models, including fast small '
                   'linear alternatives, without cropping unfavorable errors. ' +
                   ('This panel contains CPU FP32 deployment only and is a separate practical alternative, '
                    'not a GPU-kernel speed comparison. No GPU memory claim is assigned to CPU models.' if cpu else
                    'GPU deployment starts with standardized CPU input and returns full CPU output. '
                    'Allocated memory is distinct from reserved/process memory and deployed tensor bytes. '
                    'F0 post sentinels are retained in source data but excluded from family centers. '
                    'The figure does not infer stable speed superiority from point centers or compare historical timing.'),
        'sources': sources, 'numeric_values': points, 'family_centers': centers,
        'unavailable_mse_roles': unavailable, 'drift_checks': drift,
        'scope': scope, 'cost_summary': cost['summary'],
        'measurement_contract': {key: cost.get(key) for key in ('stage', 'device', 'scope', 'precision',
                                  'torch_threads', 'order_blocks', 'warmup_calls', 'passes_per_order_block')},
        'calculations': {'mse': 'Stored combined loss, mean of model losses; no ensemble or target rescore.',
                         'cost': 'Per fit median of three block medians, then mean over role members.',
                         'memory': 'MiB = bytes / 1024**2; CPU has no GPU-memory value.'}})


def numeric_claims(report, gpu, cpu, sources, directory):
    claims = []

    def add(label, source, path, value, artifact):
        claims.append({'label': label, 'source': source, 'json_path': path, 'value': value,
                       'artifact': (directory / artifact).relative_to(ROOT).as_posix(),
                       'rendered': json.dumps(value)})

    for role in ROLES:
        for period in PERIODS:
            for metric in ('mse', 'mae'):
                value = report['role_scores'][role][period][metric]
                add(f'{role}/{period}/{metric}/mean', sources[0],
                    ['role_scores', role, period, metric], value, 'robin_accuracy.source.json')
                for identifier in report['roles'][role]:
                    value = report['models'][identifier]['scores'][period][metric]
                    add(f'{identifier}/{period}/{metric}', sources[0],
                        ['models', identifier, 'scores', period, metric], value,
                        'robin_accuracy.source.json')
    for cost, source, artifact in ((gpu, sources[1], 'robin_gpu_accuracy_cost.source.json'),
                                   (cpu, sources[2], 'robin_cpu_accuracy_cost.source.json')):
        for index, row in enumerate(cost['rows']):
            if not row['include_in_primary'] or row['batch_origins'] != 4:
                continue
            add(f"{cost['stage']}/{row['id']}/block{row['block']}/throughput", source,
                ['rows', index, 'origins_per_second', 'median'], row['origins_per_second']['median'], artifact)
            if 'peak_allocated_bytes' in row:
                add(f"{row['id']}/block{row['block']}/allocated_bytes", source,
                    ['rows', index, 'peak_allocated_bytes'], row['peak_allocated_bytes'], artifact)
    return claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--gpu-cost', required=True)
    parser.add_argument('--cpu-cost', required=True)
    parser.add_argument('--directory', default='figures')
    args = parser.parse_args()
    for value in (args.name, args.directory):
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
            raise ValueError('Use a simple unique run/output name')
    paths = [HERE / value for value in (args.evaluation, args.gpu_cost, args.cpu_cost)]
    if any(Path(value).name != value or not value.endswith('.json')
           for value in (args.evaluation, args.gpu_cost, args.cpu_cost)):
        raise ValueError('Inputs must be completed JSON filenames within the v6 folder')
    directory = HERE / args.directory
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Use a fresh output directory; never overwrite a prior render')
    with Job(args.name, category='cpu_analysis', reserve_seconds=90) as job:
        report, gpu, cpu = [read_completed(path) for path in paths]
        if report.get('schema') != 'tsfm_peft_level_confirmation_v6_test_evaluation_v1':
            raise ValueError('Expected the completed fixed Robin evaluation schema')
        if set(report['roles']) != set(ROLES) or len(report['channels']) != 17:
            raise ValueError('Every pre-specified Robin method and channel must be retained')
        if gpu.get('stage') != 'gpu' or cpu.get('stage') != 'cpu':
            raise ValueError('Primary GPU and CPU costs are separate required inputs')
        if any(cost['seal']['sha256'] != report['seal']['sha256'] for cost in (gpu, cpu)):
            raise ValueError('Accuracy and cost refer to different fixed-selection seals')
        style()
        directory.mkdir(parents=True, exist_ok=True)
        sources = [receipt(path) for path in paths]
        figures = [method_figure(directory), accuracy_figure(report, sources[0], directory)]
        job.heartbeat('Rendered method and all-method period MSE/MAE; rendering GPU/CPU trade-offs')
        figures.append(cost_figure(report, gpu, GPU_ROLES, sources[:2], directory))
        figures.append(cost_figure(report, cpu, CPU_ROLES, [sources[0], sources[2]], directory, cpu=True))
        artifacts = [output for figure in figures for output in figure['outputs']]
        artifacts.extend(receipt(directory / f"{figure['stem']}.source.json") for figure in figures)
        all_sources = {item['path']: item for figure in figures for item in figure['sources']}
        manifest = {'status': 'complete', 'name': args.name, 'source_code': receipt(Path(__file__)),
                    'scope': 'Completed JSON arithmetic and rendering only; no models, prediction arrays, target scoring, fitting or benchmarking.',
                    'figures': figures, 'sources': list(all_sources.values()), 'seal': report['seal'],
                    'artifacts': artifacts, 'claims': numeric_claims(report, gpu, cpu, sources, directory)}
        save_json(directory / 'manifest.json', manifest)
        print(json.dumps({'manifest': str(directory / 'manifest.json'), 'figures': len(figures)}), flush=True)


if __name__ == '__main__':
    main()
