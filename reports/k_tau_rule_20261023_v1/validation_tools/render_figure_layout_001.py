"""Reproduce one annotation-only layout correction using frozen _figures."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
PRIVATE = Path(__file__).resolve().parent
REPO = PRIVATE.parent.parent
REPORT = REPO / 'reports' / 'k_tau_rule_20261023_v1'
OUTPUT = PRIVATE / 'figure_layout_001'
BACKUP = REPORT / 'attempts' / 'figure_layout_001'
sys.path.insert(0, str(REPO))
from research.k_tau_rule_20261023_v1 import analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (REPORT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class KCurveComplete(Exception):
    pass


def main():
    if OUTPUT.exists() or BACKUP.exists():
        raise FileExistsError('Layout attempt paths already exist; do not overwrite')
    protected = [REPORT / name for name in ('RUNS.csv', 'K_TAU.csv', 'BASELINE_RECALCULATION.csv',
        'RUN_PLAN.json', 'SUMMARY.csv', 'DECISION.json', 'FIG_BOXPLOT.png', 'FIG_BOXPLOT.svg')]
    registration = json.loads((REPORT / 'PREREGISTRATION.json').read_text(encoding='utf-8-sig'))
    protected += [REPO / name for name in registration['registered_sources']]
    before_protected = {str(path.relative_to(REPO)): sha(path) for path in protected}
    names = ('FIG_K_CURVE.png', 'FIG_K_CURVE.svg')
    before = {name: sha(REPORT / name) for name in names}
    BACKUP.mkdir(parents=True)
    for name in names:
        shutil.copyfile(REPORT / name, BACKUP / name)
        assert sha(BACKUP / name) == before[name]
    OUTPUT.mkdir()
    summaries = []
    for row in rows('SUMMARY.csv'):
        summaries.append(dict(row, K_main=int(row['K_main']), Raw=float(row['Raw']),
                              n=int(row['n']), complete=row['complete'] == 'True'))
    runs = rows('RUNS.csv')
    curves, mains = {}, {}
    baseline = rows('BASELINE_RECALCULATION.csv')
    for row in summaries:
        dataset = row['dataset']
        curves[dataset] = {int(item['K']): dict(mean=float(item['MSE_mean']),
            min=float(item['MSE_min']), max=float(item['MSE_max']))
            for item in baseline if item['dataset'] == dataset and item['family'] == 'B2'}
        mains[dataset] = [dict(item, mse=float(item['mse'])) for item in runs
                         if item['dataset'] == dataset and item['role'] == 'main']
        pair = [item['mse'] for item in mains[dataset] if int(item['seed']) in (92601, 92602)]
        assert len(pair) == 2
        curves[dataset][row['K_main']] = dict(mean=float(row['first_two_mean']), min=min(pair), max=max(pair))
    original_text, original_save = Axes.text, Figure.savefig
    changes = []

    def corrected_text(axis, x, y, text, *args, **kwargs):
        if (axis.get_title(loc='left') == 'C  ETTh2_safe17400'
                and text == 'K_tau = 4' and x == .02 and y == .03):
            changes.append(dict(panel='C', dataset='ETTh2_safe17400', text=text,
                                old_axes_position=[x, y], new_axes_position=[x, .10]))
            y = .10
        return original_text(axis, x, y, text, *args, **kwargs)

    def save_requested_then_stop(figure, filename, *args, **kwargs):
        # Stop immediately after the second requested file, before _figures
        # enters its BOXplot iteration. No second figure/statistics are made.
        original_save(figure, filename, *args, **kwargs)
        if Path(filename).name == 'FIG_K_CURVE.svg':
            raise KCurveComplete

    try:
        with patch.object(Axes, 'text', corrected_text), patch.object(Figure, 'savefig', save_requested_then_stop):
            try:
                analysis._figures(OUTPUT, summaries, mains, curves)
            except KCurveComplete:
                pass
    finally:
        plt.close('all')
    assert len(changes) == 1 and all((OUTPUT / name).stat().st_size > 100 for name in names)
    assert not list(OUTPUT.glob('*BOXPLOT*'))
    assert before_protected == {str(path.relative_to(REPO)): sha(path) for path in protected}
    for name in names:
        shutil.copyfile(OUTPUT / name, REPORT / name)
    after = {name: sha(REPORT / name) for name in names}
    receipt = dict(status='COMPLETE_ANNOTATION_LAYOUT_ONLY', created_unix=time.time(),
        change=changes[0], original_files_preserved_at=str(BACKUP),
        private_output=str(OUTPUT), before_sha256=before, after_sha256=after,
        protected_public_tables_boxplots_and_registered_sources_unchanged=True,
        protected_sha256=before_protected, frozen_figure_helper='analysis._figures',
        statistics_changed=False, model_data_PCA_or_training_access=False,
        wrapper_sha256=sha(Path(__file__)))
    with (REPORT / 'FIGURE_LAYOUT_001.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=receipt['status'], change=changes[0], output=str(OUTPUT))))


if __name__ == '__main__':
    main()
