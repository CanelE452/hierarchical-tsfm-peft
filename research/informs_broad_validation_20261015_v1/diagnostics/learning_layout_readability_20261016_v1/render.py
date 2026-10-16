"""Render every recorded learning curve with a separate legend column."""
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUTS = [
    ('V_select', 'snapshot_20261015T222956Z_1792103396179587700',
     '84705e49b52ff6793381b60996ad120972185c6b650fd239f67e49a1b3fc7326'),
    ('T_final', 'snapshot_20261015T223533Z_1792103733249252100',
     '42438b08b86d082e49deb5183fa44ceffa0f538bbc30a596253f13354e584795'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs, outputs = [], []
    destination = Path(__file__).parent / ('snapshot_' + str(time.time_ns()))
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False})
    for split, snapshot, expected in INPUTS:
        folder = ROOT / 'figures' / snapshot
        manifest_path = folder / 'FIGURE_MANIFEST.json'
        assert sha(manifest_path) == expected
        manifest = json.loads(manifest_path.read_bytes())
        entries = {row['file']: row for row in manifest['outputs']}
        tables = []
        for name in ['learning_curves.csv', 'training_diagnostics.csv']:
            path = folder / name
            assert sha(path) == entries[name]['sha256']
            assert path.stat().st_size == entries[name]['bytes']
            with path.open(encoding='utf-8', newline='') as stream:
                table = list(csv.DictReader(stream))
            tables.append(table)
            inputs.append(dict(file=path.relative_to(ROOT).as_posix(),
                               sha256=sha(path), bytes=path.stat().st_size,
                               rows=len(table), split=split))
        curves, diagnostics = tables
        for dataset in sorted({row['dataset'] for row in diagnostics}):
            rows = [row for row in diagnostics if row['dataset'] == dataset]
            curve_rows = [row for row in curves if row['dataset'] == dataset
                          and row['stage'] == 'search' and row.get('astop_mse') not in {'', None}]
            families = sorted({row['family'] for row in rows})
            fig = plt.figure(figsize=(19, 11))
            grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
            axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
                             [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])
            legend_ax = fig.add_subplot(grid[:, 2]); legend_ax.axis('off')
            colors = plt.get_cmap('tab20')
            for family in families:
                labelled = set()
                for run_id in sorted({row['run_id'] for row in curve_rows if row['family'] == family}):
                    group = [row for row in curve_rows if row['run_id'] == run_id]
                    label = family + ' lr=' + group[0]['lr'] + ' maxep=' + group[0]['budget']
                    axes[0, 0].plot([float(row['epoch']) for row in group],
                                    [float(row['astop_mse']) for row in group],
                                    marker='o', markersize=3, linewidth=1, alpha=.65,
                                    color=colors(families.index(family) % 20),
                                    label=label if label not in labelled else None)
                    labelled.add(label)
            axes[0, 0].set(xlabel='A50 search epochs; separate run lines', ylabel='A_stop normalized MSE')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                legend_ax.legend(handles, labels, loc='upper left', fontsize=8,
                                 title='All recorded curves; exact run IDs in source CSV')
            search = [row for row in rows if row['stage'] == 'search']
            axes[0, 1].scatter(np.arange(len(search)), [float(row['selected_epoch'] or 0) for row in search],
                               c=[colors(families.index(row['family']) % 20) for row in search])
            axes[0, 1].set(xlabel='Run index (exact IDs in CSV)', ylabel='Selected A_stop epoch')
            for stage, marker in [('search', 'o'), ('refit', 'x')]:
                group = [row for row in rows if row['stage'] == stage]
                axes[1, 0].scatter([float(row['exposures']) for row in group],
                                   [float(row['updates']) for row in group], marker=marker, label=stage)
                axes[1, 1].scatter([float(row['actual_epochs'] or 0) for row in group],
                                   [float(row['clipping_fraction']) for row in group], marker=marker, label=stage)
            axes[1, 0].set(xlabel='Cumulative origin exposures (search includes inherited)', ylabel='Optimizer updates')
            axes[1, 1].set(xlabel='Actual stage epochs', ylabel='Recorded clipped-step fraction')
            for ax in axes[1]:
                ax.legend(fontsize=9)
            for ax in axes.flat:
                ax.grid(alpha=.2)
            fig.suptitle(dataset + ' / ' + split + ' / recorded training chronology', fontsize=14)
            fig.text(.015, .015, 'Frozen public CSV only; all curves retained. Separate budgets are separate runs.\n'
                     'Selected epoch is not convergence; no new fitting, inference, T selection or generalization test.', fontsize=10)
            fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.92, hspace=.38, wspace=.38)
            stem = 'learning_' + dataset + '_' + split
            for extension in ['png', 'svg']:
                path = destination / (stem + '.' + extension)
                fig.savefig(path, dpi=160, facecolor='white', bbox_inches='tight')
                outputs.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
            plt.close(fig)
    assert all(sha(ROOT / item['file']) == item['sha256'] for item in inputs)
    result = dict(status='PASS_PUBLIC_CSV_RENDER_SOURCE_BYTES_UNCHANGED',
                  source_sha256=sha(Path(__file__)), inputs=inputs, outputs=outputs,
                  original_figures_overwritten=False, all_recorded_curves_retained=True,
                  GPU_models_private_arrays_raw_data_or_training_access=False,
                  numerical_metric_or_case_selection_changed=False, visual_review_pending=True)
    with (destination / 'FIGURE_MANIFEST.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(destination)


if __name__ == '__main__':
    main()
