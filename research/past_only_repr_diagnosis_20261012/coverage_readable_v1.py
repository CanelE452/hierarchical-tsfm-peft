import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import Budget, HERE, access_guard, read, sha, write


def main():
    access_guard()
    folder = HERE / 'final_report_v5_20261012'
    manifest = folder / 'MANIFEST_v5.json'
    source = folder / 'raw_metrics_v5.csv'
    target = folder / 'figures' / '08_actual_coverage_readable.png'
    receipt = folder / 'COVERAGE_READABILITY_RECEIPT_v1.json'
    if target.exists() or receipt.exists():
        raise FileExistsError('Fresh readability artifact required')
    parent = read(manifest)
    if sha(source) != parent['outputs']['raw_metrics_csv_sha256']:
        raise ValueError('Verified metrics CSV changed')
    with Budget('visual_closure', False, 'Readable coverage figure from verified CSV') as budget:
        with source.open(encoding='utf-8-sig', newline='') as handle:
            counts = Counter(row['experiment'] for row in csv.DictReader(handle))
        keys = list(counts)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(keys, [counts[key] for key in keys], color='#657e95')
        ax.invert_yaxis()
        ax.bar_label(bars, padding=4)
        ax.set_xlim(0, max(counts.values()) * 1.12)
        ax.set_xlabel('Adopted metric rows (not independent runs or model fits)')
        ax.set_title('Actual verified result coverage')
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        budget.check()
        ledger = str(budget.path)
    write(receipt, {'status': 'COMPLETE_METADATA_VISUAL', 'source': str(source),
        'source_sha256': sha(source), 'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'counts': dict(counts), 'figure': str(target), 'figure_sha256': sha(target),
        'budget_ledger': ledger, 'budget_ledger_sha256': sha(ledger),
        'scope': 'Readability supplement to figure07; no new inference, metrics, or fits',
        'source_script_sha256': sha(Path(__file__))})
    print(str(target))


if __name__ == '__main__':
    main()
