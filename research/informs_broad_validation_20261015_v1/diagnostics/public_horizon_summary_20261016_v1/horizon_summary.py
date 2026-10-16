"""Horizon-band summaries from explicitly bound public sufficient statistics only."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time


STAT_FIELDS = ('run_id', 'split', 'cohort', 'origin', 'channel', 'horizon_start', 'horizon_end', 'sse', 'sae', 'count')
IDENTITY_HASHES = ('source_sha', 'config_sha', 'data_sha', 'model_sha', 'scaler_sha')
ATOL, RTOL = 1e-9, 1e-7


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    name = str(relative)
    posix = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or posix.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Expected safe relative public path')
    root = Path(root).resolve(); path = root / posix
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents):
        raise ValueError('Expected relative nonsymlink public path')
    return path


def integer(value, label):
    text = str(value)
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError('Expected nonnegative integer: ' + label)
    return int(text)


def bands(horizon):
    horizon = integer(horizon, 'horizon')
    if not horizon:
        raise ValueError('Positive horizon required')
    edges = (0, (horizon + 3) // 4, (horizon + 1) // 2, horizon)
    return list(zip(('near', 'middle', 'far'), edges[:-1], edges[1:]))


def same_number(actual, expected, label):
    empty = actual is None or actual == '' or actual == 'None'
    if expected is None:
        if not empty:
            raise ValueError('Unexpected numeric value for unobserved ' + label)
    elif empty or not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), abs_tol=ATOL, rel_tol=RTOL):
        raise ValueError('Overall score mismatch: ' + label)


def aggregate_rows(rows, score):
    """Independent FP64 aggregation; each complete origin/channel/band cell remains required."""
    horizon = integer(score['horizon'], 'horizon')
    declared_bands = bands(horizon)
    allowed = {(start, end): label for label, start, end in declared_bands if start < end}
    values = defaultdict(lambda: [[], [], 0])
    seen, origins, channels = set(), set(), set()
    observed_origins = defaultdict(set)
    for row in rows:
        if set(row) != set(STAT_FIELDS):
            raise ValueError('Public statistic CSV schema differs')
        if any(value is None for value in row.values()):
            raise ValueError('Missing public statistic cell')
        if any(str(row[k]) != str(score[k]) for k in ('run_id', 'split', 'cohort')):
            raise ValueError('Statistic identity differs from official score')
        origin = integer(row['origin'], 'origin')
        channel = str(row['channel'])
        if not channel or channel.strip() != channel or any(c in channel for c in '\r\n\t'):
            raise ValueError('Nonempty original channel identifier required')
        start, end = integer(row['horizon_start'], 'start'), integer(row['horizon_end'], 'end')
        if (start, end) not in allowed:
            raise ValueError('Horizon band range differs from ceil(H/4),ceil(H/2),H')
        count = integer(row['count'], 'count')
        sse, sae = float(row['sse']), float(row['sae'])
        if count > end - start or not math.isfinite(sse) or not math.isfinite(sae) or min(sse, sae) < 0:
            raise ValueError('Invalid SSE, SAE, or observed count')
        if count == 0 and (sse != 0 or sae != 0):
            raise ValueError('Unobserved cell must retain exact zero SSE and SAE')
        key = origin, channel, start, end
        if key in seen:
            raise ValueError('Duplicate origin/channel/horizon cell')
        seen.add(key); origins.add(origin); channels.add(channel)
        acc = values[channel, start, end]
        acc[0].append(sse); acc[1].append(sae); acc[2] += count
        if count:
            observed_origins[start, end].add(origin)
    expected_n = integer(score['n_origins'], 'n_origins')
    expected_c = integer(score['num_channels'], 'num_channels')
    if not expected_n or not expected_c or len(origins) != expected_n or len(channels) != expected_c or len(seen) != expected_n * expected_c * len(allowed):
        raise ValueError('Incomplete origin/channel/band coverage')
    totals = {}
    for channel in channels:
        parts = [values[channel, a, b] for a, b in allowed]
        totals[channel] = (math.fsum(x for part in parts for x in part[0]),
                           math.fsum(x for part in parts for x in part[1]), sum(part[2] for part in parts))
    active = [v for v in totals.values() if v[2]]
    overall = dict(mse=math.fsum(v[0] / v[2] for v in active) / len(active) if active else None,
                   mae=math.fsum(v[1] / v[2] for v in active) / len(active) if active else None,
                   n_observed=sum(v[2] for v in totals.values()), n_origins=len(origins), num_channels=len(channels),
                   active_channels=len(active), horizon=horizon)
    for key in ('mse', 'mae'):
        same_number(score[key], overall[key], key)
    for key in ('n_observed', 'n_origins', 'num_channels', 'active_channels', 'horizon'):
        if score.get(key) not in (None, '') and integer(score[key], key) != overall[key]:
            raise ValueError('Overall integer score mismatch: ' + key)
    result = []
    for label, start, end in declared_bands:
        grouped = [values[channel, start, end] for channel in channels] if start < end else []
        valid = [v for v in grouped if v[2]]
        result.append(dict(band=label, horizon_start=start, horizon_end=end,
            mse=math.fsum(math.fsum(v[0]) / v[2] for v in valid) / len(valid) if valid else None,
            mae=math.fsum(math.fsum(v[1]) / v[2] for v in valid) / len(valid) if valid else None,
            n_observed=sum(v[2] for v in grouped), n_origins=len(origins), num_channels=len(channels),
            active_channels=len(valid), observed_origins=len(observed_origins[start, end]),
            horizon=horizon, statistic_cells=len(origins) * len(channels) if start < end else 0))
    return result, overall


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metadata_plan(scores_path, public_root, input_manifest_path):
    """Read small explicit CSV/JSON metadata and stat sizes; never read gzip bytes."""
    public = Path(public_root).resolve()
    scores = Path(scores_path).resolve(); binding_path = Path(input_manifest_path).resolve()
    for path in (scores, binding_path):
        if not path.is_relative_to(public):
            raise ValueError('Scores and input manifest must be within public root')
    bindings = {}
    for row in _json(binding_path)['inputs']:
        if row.get('scope') != 'public':
            continue
        name = row['file']; inside(public, name)
        value = dict(sha256=row['sha256'], bytes=integer(row['bytes'], 'bound bytes'))
        if name in bindings and bindings[name] != value:
            raise ValueError('Contradictory public SHA binding')
        bindings[name] = value
    routes, seen, lock, lock_sha = [], set(), None, None
    with scores.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] not in ('V_select', 'T_final'):
                raise ValueError('Only official V_select/T_final scores are supported')
            name = row['manifest_relative_path']; path = inside(public, name)
            if name in seen:
                raise ValueError('Duplicate score manifest row')
            seen.add(name)
            if name not in bindings or file_sha(path) != bindings[name]['sha256'] or path.stat().st_size != bindings[name]['bytes']:
                raise ValueError('Official manifest SHA/size binding differs')
            manifest = _json(path)
            for key in ('run_id', 'split', 'cohort', *IDENTITY_HASHES):
                if not row.get(key) or row[key] != str(manifest.get(key)):
                    raise ValueError('Score/manifest identity differs: ' + key)
            official = manifest['score']
            for key in ('mse', 'mae'):
                same_number(row[key], official[key], key)
            for key in ('n_observed', 'n_origins', 'num_channels', 'horizon', 'active_channels'):
                if integer(row[key], key) != integer(official[key], key):
                    raise ValueError('Score/manifest integer differs: ' + key)
            shape = manifest['arrays']['pred_normalized']['shape']
            if shape != [int(row['n_origins']), int(row['horizon']), int(row['num_channels'])]:
                raise ValueError('Official shape differs from score dimensions')
            stats_name = manifest['public_stats_relative_path']; stats = inside(public, stats_name)
            if stats.name != 'PUBLIC_ERROR_STATS.csv.gz' or stats_name not in bindings:
                raise ValueError('Official public statistic SHA binding is missing')
            if stats.stat().st_size != bindings[stats_name]['bytes']:
                raise ValueError('Public statistic bound size differs; restore original statistics first')
            if row['split'] == 'T_final':
                lock_path = public / 'SELECTION_LOCK.json'
                if lock is None:
                    lock = _json(lock_path); lock_sha = file_sha(lock_path)
                if not lock.get('locked') or row['run_id'] not in lock.get('allowed_run_ids', []):
                    raise ValueError('T score is outside the actual locked selection')
            routes.append(dict(score=row, manifest_relative_path=name, manifest_sha256=bindings[name]['sha256'],
                               public_stats_relative_path=stats_name, public_stats_sha256=bindings[stats_name]['sha256'],
                               public_stats_bytes=bindings[stats_name]['bytes']))
    if not routes:
        raise ValueError('No official V/T score rows supplied')
    return dict(status='METADATA_ONLY_NO_GZIP_READ', scores_relative_path=scores.relative_to(public).as_posix(),
                scores_sha256=file_sha(scores), input_manifest_relative_path=binding_path.relative_to(public).as_posix(),
                input_manifest_sha256=file_sha(binding_path), routes=routes, gzip_bytes_read=0,
                selection_lock_sha256=lock_sha,
                numeric_analysis_executed=False, source_sha256=file_sha(__file__))


def generate(scores_path, public_root, input_manifest_path):
    public = Path(public_root).resolve()
    plan = metadata_plan(scores_path, public, input_manifest_path)
    output = public / 'diagnostics/public_horizon_summary_20261016_v1' / ('snapshot_' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    results, failures, checked = [], [], []
    metadata_keys = ('run_id', 'split', 'cohort', 'dataset', 'panel', 'fold', 'family', 'latent_k', 'seed',
                     'learning_rate', 'max_epoch', 'selected_epoch', 'training_status', *IDENTITY_HASHES)
    for route in plan['routes']:
        row = route['score']; path = inside(public, route['public_stats_relative_path'])
        try:
            if file_sha(path) != route['public_stats_sha256']:
                raise ValueError('Public statistic SHA differs before aggregation')
            with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != STAT_FIELDS:
                    raise ValueError('Public statistic CSV schema differs')
                groups, overall = aggregate_rows(reader, row)
            if file_sha(path) != route['public_stats_sha256']:
                raise ValueError('Public statistic SHA changed during aggregation')
            metadata = {key: row.get(key, '') for key in metadata_keys}
            metadata.update(recipe=row.get('recipe') or 'CONFIG_SHA:' + row['config_sha'],
                            manifest_relative_path=route['manifest_relative_path'], manifest_sha256=route['manifest_sha256'],
                            public_stats_relative_path=route['public_stats_relative_path'], public_stats_sha256=route['public_stats_sha256'])
            results.extend({**metadata, **band} for band in groups)
            checked.append(dict(manifest_relative_path=route['manifest_relative_path'], overall=overall,
                                public_stats_sha256=route['public_stats_sha256']))
        except (ValueError, OSError, EOFError, OverflowError, csv.Error, UnicodeError) as error:
            failures.append(dict(manifest_relative_path=route['manifest_relative_path'], error_type=type(error).__name__,
                                 reason=str(error) if isinstance(error, ValueError) else 'Cannot read declared public statistic'))
    for route in plan['routes']:
        if file_sha(inside(public, route['manifest_relative_path'])) != route['manifest_sha256']:
            failures.append(dict(manifest_relative_path=route['manifest_relative_path'], reason='Manifest changed during analysis'))
    if file_sha(scores_path) != plan['scores_sha256'] or file_sha(input_manifest_path) != plan['input_manifest_sha256']:
        failures.append(dict(reason='Final score or input-binding table changed during analysis'))
    if plan['selection_lock_sha256'] is not None and file_sha(public / 'SELECTION_LOCK.json') != plan['selection_lock_sha256']:
        failures.append(dict(reason='Selection lock changed during analysis'))
    fields = list(metadata_keys) + ['recipe', 'manifest_relative_path', 'manifest_sha256', 'public_stats_relative_path',
        'public_stats_sha256', 'band', 'horizon_start', 'horizon_end', 'mse', 'mae', 'n_observed', 'n_origins',
        'num_channels', 'active_channels', 'observed_origins', 'horizon', 'statistic_cells']
    with (output / 'HORIZON_SCORES.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    with (output / 'INPUT_SHA.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    result = dict(status='INCOMPLETE_FAILED_INPUTS' if failures else 'PASS_PUBLIC_STATISTICS_ONLY',
        requested_routes=len(plan['routes']), verified_routes=len(checked), output_band_rows=len(results),
        failures=failures, overall_reconciliation=checked, atol=ATOL, rtol=RTOL,
        aggregate='math.fsum of SSE/SAE per channel, divide each channel count, equal mean of observed channels',
        metadata_only=False, raw_arrays_models_GPU_access=False, inference_executed=False,
        target_values_extracted=False, T_selection_or_case_selection=False, CI_or_significance_claimed=False,
        original_statistics_modified=False, global_scores_modified=False,
        output_sha256=file_sha(output / 'HORIZON_SCORES.csv'))
    with (output / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    (output / 'README_KO.md').write_text('# 공개 충분통계의 horizon 요약\n\n'
        '가까움·중간·먼 구간은 ceil(H/4),ceil(H/2),H로 고정했다. 관측이 없는 band는 빈 MSE/MAE와 count0으로 보존했다. '
        '전체 오차는 band 평균이 아니라 채널별 모든 SSE/SAE/count를 다시 합쳐 최종 SCORES와 대사했다. '
        '이 결과는 공개 집계 재검산이며 원예측 독립 재채점·학습·추론·CI·방법 선택이 아니다.\n\n'
        '- [전체 구간 표](HORIZON_SCORES.csv)\n- [입력 결속](INPUT_SHA.json)\n- [검산·실패 범위](RESULT.json)\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--execute-report', action='store_true')
    args = parser.parse_args(argv)
    if args.execute_report:
        output = generate(args.scores, args.public_root, args.input_manifest)
        result = _json(output / 'RESULT.json')
        print(json.dumps(dict(status=result['status'], output=output.relative_to(args.public_root.resolve()).as_posix())))
        return 0 if result['status'] == 'PASS_PUBLIC_STATISTICS_ONLY' else 1
    plan = metadata_plan(args.scores, args.public_root, args.input_manifest)
    print(json.dumps(dict(status=plan['status'], routes=len(plan['routes']), gzip_bytes_read=0,
                         scores_sha256=plan['scores_sha256'], input_manifest_sha256=plan['input_manifest_sha256'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
