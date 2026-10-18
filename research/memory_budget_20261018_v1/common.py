"""Small campaign identities; importing this module performs no experiment."""
from pathlib import Path
from research.b2_revin_20261018_v1.contracts import (
    atomic_json, binding, fingerprint, read, resolve_binding, sealed_json, sha,
)

NAME = 'memory_budget_20261018_v1'
REPO = Path(__file__).resolve().parents[2]
STARTED_UNIX = 1792308025
OPERATING_SECONDS = 4 * 3600
REPORT_RESERVE_SECONDS = 1800
SCIENTIFIC_DEADLINE = STARTED_UNIX + OPERATING_SECONDS - REPORT_RESERVE_SECONDS


def paths(repo=REPO):
    repo = Path(repo).resolve()
    return dict(repo=repo, cache=repo/'.cache'/NAME, report=repo/'reports'/NAME)


def source_context(asset_root):
    from research.b2_revin_20261018_v1.run import context
    return context(asset_root)


def selected_rows(ctx):
    rows = read(Path(ctx['cache'])/'VERIFIED_ROWS.json')
    result = [row for row in rows if row['final'] is True
              and (row['family'] not in ('B2', 'B0') or row.get('revin') is False)]
    if len(result) != 11 or len({row['run_id'] for row in result}) != 11:
        raise ValueError('Exactly eleven final OFF accuracy identities required')
    for row in result:
        if row.get('independent_FP64') != 'PASS':
            raise ValueError('Unverified parent accuracy')
        resolve_binding(row['prediction_binding'], ctx)
        if row.get('bundle_binding'):
            resolve_binding(row['bundle_binding'], ctx)
    return sorted(result, key=lambda row: (row['dataset'], row['family'], row['seed']))


def source_hashes():
    return {path.relative_to(REPO).as_posix(): sha(path)
            for path in sorted((REPO/'research'/NAME).glob('*.py'))}
