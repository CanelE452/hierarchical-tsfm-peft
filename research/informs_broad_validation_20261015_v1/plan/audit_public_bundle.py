#!/usr/bin/env python3
"""Reaggregate the PROPOSED public error-statistics schema; no model execution.

Required SCORES.csv columns:
run_id,split,cohort,mse,mae,n_observed,n_origins,num_channels,horizon

Required PUBLIC_ERROR_STATS/*.csv[.gz] columns:
run_id,split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count

Statistics must cover every channel and origin, including count=0 rows. Horizon
bands are zero-based half-open and must partition [0,horizon) for every pair.
SSE/SAE must use the same normalized scale as the score being checked. Do not
mix physical-scale statistics into this bundle. This validates arithmetic and
coverage, NOT source data legality, prediction correctness, training or timers.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

SCORE_FIELDS = {'run_id','split','cohort','mse','mae','n_observed','n_origins','num_channels','horizon'}
STAT_FIELDS = {'run_id','split','cohort','origin','channel','horizon_start','horizon_end','sse','sae','count'}


def rows(path: Path, required: set[str]):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path.name}: missing fields {sorted(missing)}')
        for number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f'{path.name}:{number}: malformed CSV row')
            yield number, row


def finite(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{label}: expected finite nonnegative value, got {value!r}')
    return result


def integer(value: str, label: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f'{label}: negative integer')
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, atol: float = 1e-9, rtol: float = 1e-7) -> dict:
    root = root.resolve()
    scores = {}
    for line, row in rows(root / 'SCORES.csv', SCORE_FIELDS):
        key = (row['run_id'], row['split'], row['cohort'])
        if not all(key) or key in scores:
            raise ValueError(f'SCORES.csv:{line}: empty/duplicate score key {key}')
        scores[key] = dict(mse=finite(row['mse'], 'mse'), mae=finite(row['mae'], 'mae'),
                           n_observed=integer(row['n_observed'], 'n_observed'),
                           n_origins=integer(row['n_origins'], 'n_origins'),
                           num_channels=integer(row['num_channels'], 'num_channels'),
                           horizon=integer(row['horizon'], 'horizon'))
        if min(scores[key]['n_origins'], scores[key]['num_channels'], scores[key]['horizon']) < 1:
            raise ValueError(f'{key}: invalid positive shape dimensions')
    if not scores:
        raise ValueError('No numeric scores to verify; use a separate status registry for unrun jobs.')

    # Accumulate by channel before macro averaging. Never average per-origin MSEs.
    channel_values = defaultdict(lambda: [[], [], 0])
    bands = defaultdict(list)
    paths = sorted((root / 'PUBLIC_ERROR_STATS').glob('*.csv')) + sorted((root / 'PUBLIC_ERROR_STATS').glob('*.csv.gz'))
    if not paths:
        raise ValueError('No public error-statistics files found')
    for path in paths:
        for line, row in rows(path, STAT_FIELDS):
            key = (row['run_id'], row['split'], row['cohort'])
            if key not in scores:
                raise ValueError(f'{path.name}:{line}: unknown score key {key}')
            origin, channel = row['origin'], row['channel']
            if not origin or not channel:
                raise ValueError(f'{path.name}:{line}: missing origin/channel')
            start = integer(row['horizon_start'], 'horizon_start')
            stop = integer(row['horizon_end'], 'horizon_end')
            count = integer(row['count'], 'count')
            sse, sae = finite(row['sse'], 'sse'), finite(row['sae'], 'sae')
            if not 0 <= start < stop <= scores[key]['horizon'] or count > stop-start:
                raise ValueError(f'{path.name}:{line}: invalid band/count')
            if count == 0 and (sse != 0.0 or sae != 0.0):
                raise ValueError(f'{path.name}:{line}: nonzero errors with zero observations')
            if count and sae*sae > count*sse + atol + rtol*max(count*sse, sae*sae):
                raise ValueError(f'{path.name}:{line}: SAE/SSE violate Cauchy bound')
            bands[(key, origin, channel)].append((start, stop))
            sums = channel_values[(key, channel)]
            sums[0].append(sse); sums[1].append(sae); sums[2] += count

    pairs_by_key = defaultdict(set)
    for (key, origin, channel), intervals in bands.items():
        cursor = 0
        for start, stop in sorted(intervals):
            if start != cursor:
                raise ValueError(f'{key}/{origin}/{channel}: duplicate, gap or overlapping bands')
            cursor = stop
        if cursor != scores[key]['horizon']:
            raise ValueError(f'{key}/{origin}/{channel}: incomplete horizon')
        pairs_by_key[key].add((origin, channel))

    checked = []
    for key, expected in scores.items():
        pairs = pairs_by_key[key]
        origins = {x[0] for x in pairs}; channels = {x[1] for x in pairs}
        if (len(origins), len(channels), len(pairs)) != (expected['n_origins'], expected['num_channels'], expected['n_origins']*expected['num_channels']):
            raise ValueError(f'{key}: incomplete origin/channel coverage')
        per_mse, per_mae, total = [], [], 0
        for channel in channels:
            sses, saes, count = channel_values[(key, channel)]
            total += count
            if count:
                per_mse.append(math.fsum(sses) / count)
                per_mae.append(math.fsum(saes) / count)
        if not per_mse:
            raise ValueError(f'{key}: no observed values; numeric score is not defined')
        actual = {'mse': math.fsum(per_mse)/len(per_mse),
                  'mae': math.fsum(per_mae)/len(per_mae), 'n_observed': total}
        if total != expected['n_observed']:
            raise ValueError(f'{key}: observed count mismatch')
        for metric in ('mse','mae'):
            if not math.isclose(actual[metric], expected[metric], rel_tol=rtol, abs_tol=atol):
                raise ValueError(f'{key}: {metric} expected={expected[metric]:.17g}, recomputed={actual[metric]:.17g}')
        checked.append({'run_id': key[0], 'split': key[1], 'cohort': key[2], **actual})

    manifest = root / 'PUBLIC_MANIFEST.sha256'
    manifest_status, n_hashes = 'NOT_PROVIDED', 0
    if manifest.is_file():
        seen_paths = set()
        for number, line in enumerate(manifest.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip('*')
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path == manifest or relative in seen_paths:
                raise ValueError(f'manifest:{number}: unsafe, self-referential or duplicate path')
            if len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
                raise ValueError(f'manifest:{number}: invalid SHA256')
            if not path.is_file() or sha256(path) != digest.lower():
                raise ValueError(f'manifest:{number}: missing file/hash mismatch: {relative}')
            seen_paths.add(relative); n_hashes += 1
        manifest_status = 'PASS' if n_hashes else 'EMPTY'

    return {'status': 'PASS', 'scope': 'PUBLIC_STATISTICS_REAGGREGATION_ONLY',
            'checked_score_rows': len(checked), 'statistics_files': len(paths),
            'manifest_status': manifest_status, 'verified_file_hashes': n_hashes,
            'prediction_array_rescoring': False, 'checkpoint_replay': False,
            'training_rerun': False, 'timer_reexecution': False,
            'atol': atol, 'rtol': rtol, 'scores': checked}


def self_test() -> dict:
    """Synthetic arithmetic tests only. Never copy these as scientific results."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp); (root / 'PUBLIC_ERROR_STATS').mkdir()
        sp = root / 'SCORES.csv'; ep = root / 'PUBLIC_ERROR_STATS' / 'sample.csv'
        sp.write_text('run_id,split,cohort,mse,mae,n_observed,n_origins,num_channels,horizon\n'
                      'synthetic,T,fixture,2.5,1.5,4,1,2,2\n', encoding='utf-8')
        ep.write_text('run_id,split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count\n'
                      'synthetic,T,fixture,0,c1,0,2,2,2,2\n'
                      'synthetic,T,fixture,0,c2,0,2,8,4,2\n', encoding='utf-8')
        audit(root)
        sp.write_text(sp.read_text().replace(',2.5,1.5,', ',2.6,1.5,'))
        caught = False
        try:
            audit(root)
        except ValueError:
            caught = True
        if not caught:
            raise AssertionError('tampered score not detected')
        sp.write_text(sp.read_text().replace(',2.6,1.5,', ',2.5,1.5,'))
        ep.write_text(ep.read_text() + 'synthetic,T,fixture,0,c2,0,2,8,4,2\n')
        try:
            audit(root)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate statistics not detected')
    return {'status': 'SELF_TEST_PASS', 'tests': ['valid_macro_average','tampered_score','duplicate_band'],
            'scientific_results_tested': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test and args.root is None:
        parser.error('--root or --self-test is required')
    try:
        result = self_test() if args.self_test else audit(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, csv.Error, AssertionError) as exc:
        print(json.dumps({'status':'FAIL','scope':'PUBLIC_STATISTICS_REAGGREGATION_ONLY','error':str(exc)}, ensure_ascii=False))
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
