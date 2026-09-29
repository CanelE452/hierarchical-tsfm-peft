"""Render source-linked v8 comparison figures; never evaluate a model."""
import argparse
import json
from runtime_v8 import HERE, Job, digest, save_json

LABELS = {'pq_split': 'P/Q independent (v8)', 'raw64': 'RAW64 (v8)',
          'level_res': 'LEVEL (previous)', 'level_raw': 'RAW32 (previous)',
          'direct_nlinear': 'Direct NLinear', 'f0': 'F0', 'lora': 'LoRA'}
COLORS = {'pq_split': '#0072B2', 'raw64': '#009E73', 'level_res': '#56B4E9',
          'level_raw': '#777777', 'direct_nlinear': '#CC79A7', 'f0': '#E69F00', 'lora': '#D55E00'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--robin', default='robin_eval01.json')
    parser.add_argument('--jena', default='jena_eval01.json')
    args = parser.parse_args()
    if (HERE / 'figure_manifest.json').exists():
        raise RuntimeError('Preserve existing rendered evidence')
    with Job(args.job, category='cpu_analysis', reserve_seconds=120):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
        paths = [HERE / args.robin, HERE / args.jena]
        data = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        manifest = {'inputs': {str(p.relative_to(HERE)): digest(p) for p in paths}, 'figures': {}}
        out = HERE / 'figures'
        out.mkdir(exist_ok=True)
        order = list(LABELS)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), layout='constrained')
        for axis, result, title in zip(axes, data, ['Robin: exposed development', 'Jena 2024: exposed development']):
            for y, role in enumerate(order):
                scores = result['role_scores'][role]
                center = scores['combined']['mse']
                ids = scores['combined']['run_ids']
                seeds = [result['models'][i]['scores']['combined']['mse'] for i in ids]
                axis.plot([min(seeds), max(seeds)], [y, y], color=COLORS[role], lw=3)
                axis.plot(center, y, 'o', color=COLORS[role], ms=6)
                axis.plot(scores['test_a']['mse'], y - .15, '|', color=COLORS[role], ms=11)
                axis.plot(scores['test_b']['mse'], y + .15, 'x', color=COLORS[role], ms=5)
                axis.annotate(f'{center:.4f}', (center, y), xytext=(6, 5), textcoords='offset points', fontsize=9)
            axis.set_yticks(range(len(order)), [LABELS[k] for k in order])
            axis.invert_yaxis()
            axis.set_xlabel('Masked channel-macro MSE (TRAIN-standardized)')
            axis.set_title(title, loc='left', fontsize=12)
            axis.grid(axis='x', alpha=.2)
            axis.set_xlim(left=0)
            axis.margins(x=.2, y=.12)
        fig.legend(handles=[Line2D([], [], marker='o', color='black', ls='-', label='Mean seed loss; segment = seed range'),
                            Line2D([], [], marker='|', color='black', ls='', label='Period A'),
                            Line2D([], [], marker='x', color='black', ls='', label='Period B')],
                   loc='outside lower center', ncol=3, frameon=False)
        manifest['figures']['accuracy'] = []
        for ext in ('png', 'pdf', 'svg'):
            path = out / ('accuracy_v8.' + ext)
            fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
            manifest['figures']['accuracy'].append({'path': str(path), 'sha256': digest(path)})
        plt.close(fig)
        manifest['scope'] = 'Seven specified practical roles; other reused controls remain in full evaluation tables. Mean individual-seed losses, not prediction ensemble. Segments are seed ranges, not confidence intervals. No cross-dataset pooled score.'
        save_json(HERE / 'figure_manifest.json', manifest)


if __name__ == '__main__':
    main()
