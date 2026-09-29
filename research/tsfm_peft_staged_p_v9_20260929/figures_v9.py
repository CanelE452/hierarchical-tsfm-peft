"""Render fixed comparisons from completed scores, without inference or fitting."""
import argparse
import json

from runtime_v9 import HERE, Job, digest, save_json

ROLES = [('level_res', 'Parent LEVEL'), ('staged_p', 'Staged P: basic'),
         ('staged_p_alpha', 'Staged P: VAL alpha'), ('staged_raw', 'Staged RAW: basic'),
         ('staged_raw_alpha', 'Staged RAW: VAL alpha'), ('pq_split', 'Joint P/Q v8'),
         ('raw64', 'RAW64 v8'), ('direct_nlinear', 'Direct NLinear'),
         ('f0', 'Uncompressed F0'), ('lora', 'Uncompressed LoRA')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    with Job(args.name, category='cpu_analysis', reserve_seconds=120):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False})
        evaluations = {}
        sources = {}
        for dataset in ('robin', 'jena'):
            path = HERE / f'{dataset}_eval01.json'
            evaluations[dataset] = json.loads(path.read_text(encoding='utf-8'))
            assert evaluations[dataset]['status'] == 'complete'
            sources[dataset] = {'path': str(path), 'sha256': digest(path)}
        target = HERE / 'figures'
        target.mkdir(exist_ok=True)
        outputs = {}

        def export(fig, name):
            for extension in ('svg', 'png'):
                path = target / f'{name}.{extension}'
                if path.exists():
                    raise RuntimeError('Preserve existing figure: ' + str(path))
                fig.savefig(path, dpi=220, bbox_inches='tight')
                if extension == 'svg':
                    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
                outputs[path.name] = {'path': str(path), 'sha256': digest(path)}
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), layout='constrained', sharey=True)
        for ax, (dataset, evaluation) in zip(axes, evaluations.items()):
            for index, (role, label) in enumerate(ROLES):
                value = evaluation['role_scores'][role]['combined']['mse']
                color = '#0072B2' if role.startswith('staged_p') else '#D55E00' if role.startswith('staged_raw') else '#555555'
                ax.scatter(value, index, c=color, marker='D', s=50, zorder=3)
                values = [evaluation['models'][identifier]['scores']['combined']['mse']
                          for identifier in evaluation['roles'][role]]
                if len(values) > 1:
                    ax.scatter(values, [index - .13, index + .13], c=color, marker='|', s=90, zorder=4)
                ax.annotate(f'{value:.4f}', (value, index), xytext=(7, 0),
                            textcoords='offset points', va='center', fontsize=9)
            ax.set_title(dataset.title())
            ax.set_xlabel('Observed-channel macro MSE (lower is better)')
            ax.set_xlim(0, max(evaluation['role_scores'][role]['combined']['mse'] for role, _ in ROLES) * 1.21)
            ax.grid(axis='x', alpha=.2)
        axes[0].set_yticks(np.arange(len(ROLES)), [label for _, label in ROLES])
        axes[0].invert_yaxis()
        fig.suptitle('Fixed staged correction comparisons\nDiamonds: mean seed loss; ticks: individual seeds. Both datasets are exposed development data.', fontsize=12)
        export(fig, 'accuracy_combined')

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
        short_roles = ROLES[:6]
        for column, (dataset, evaluation) in enumerate(evaluations.items()):
            for row, period in enumerate(('test_a', 'test_b')):
                ax = axes[row, column]
                values = [evaluation['role_scores'][role][period]['mse'] for role, _ in short_roles]
                labels = ['Parent', 'P basic', 'P alpha', 'RAW basic', 'RAW alpha', 'Joint v8']
                ax.bar(np.arange(len(values)), values, color=['#777777', '#0072B2', '#56B4E9', '#D55E00', '#E69F00', '#999999'])
                ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha='right')
                ax.set_ylim(0, max(values) * 1.18)
                ax.set_title(f'{dataset.title()} / {period.upper()}')
                ax.set_ylabel('Macro MSE')
                ax.grid(axis='y', alpha=.2)
                for i, value in enumerate(values):
                    ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        fig.suptitle('Both predefined periods, without period or dataset selection\nLoss means over seeds; combined results use channel error sums/counts, not period means.', fontsize=12)
        export(fig, 'accuracy_periods')
        save_json(HERE / f'{args.name}.json', {'status': 'complete', 'sources': sources,
                  'outputs': outputs, 'new_inference': False, 'new_fit': False,
                  'scope': 'Saved point estimates and individual seeds; no independence or confidence claim'})


if __name__ == '__main__':
    main()
