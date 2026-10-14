"""Saved-prediction diagnostics only; actual reads require the current CPU gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
DATASETS = ('robin', 'beijing_aotizhongxin')
SEEDS = (92601, 92602)
ROUTES = ('RAW33C', 'E33K', 'E512K')
BIN_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, float('inf'))
BIN_LABELS = ('[0,0.25)', '[0.25,0.5)', '[0.5,1)', '[1,2)', '[2,inf)')
TOLERANCE = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def checked(record):
    path = resolve(record['path'])
    actual = sha(path)
    if actual != str(record['sha256']).lower():
        raise ValueError('DIAGNOSIS_INPUT_SHA_MISMATCH:' + str(path))
    return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}


def json_record(path):
    path = resolve(path)
    record = checked({'path': str(path), 'sha256': sha(path)})
    return json.loads(path.read_text(encoding='utf-8-sig')), record


def audit_binding(audit, producers, records):
    expected = {(d, s) for d in DATASETS for s in SEEDS}
    rows = audit.get('conditions', [])
    if (audit.get('status') != 'PASS' or audit.get('failed_count') != 0
            or len(rows) != 4 or {(r['dataset'], r['seed']) for r in rows} != expected
            or any(r.get('status') != 'PASS' for r in rows)
            or audit.get('source_before_after_SHA_pass') is not True):
        raise ValueError('EXACT_FOUR_CONDITION_SCIENCE_AUDIT_PASS_REQUIRED')
    if len(producers) != 4 or {(p['dataset'], p['seed']) for p in producers} != expected:
        raise ValueError('EXACT_FOUR_PRODUCERS_REQUIRED')
    for producer, record in zip(producers, records):
        if producer.get('status') != 'COMPLETE':
            raise ValueError('COMPLETE_SAVED_PREDICTION_PRODUCER_REQUIRED')
        bound = next(r for r in rows if (r['dataset'], r['seed']) == (producer['dataset'], producer['seed']))
        if bound['producer_sha256'].lower() != record['sha256']:
            raise ValueError('AUDIT_PRODUCER_SHA_BINDING_REQUIRED')
        for flag in ('GPU_execution', 'E_D_F_loaded', 'E_D_F_forward_replayed',
                     'E_D_F_updated', 'protected_TEST_accessed', 'future_oracle_read'):
            if producer.get(flag) is not False:
                raise ValueError('PAST_ONLY_CPU_PRODUCER_FLAG:' + flag)
        if producer.get('source_artifacts_before_after_SHA_pass') is not True:
            raise ValueError('PRODUCER_SOURCE_BEFORE_AFTER_REQUIRED')


def decompose(prediction, reference, target, observed, original_main):
    """A negative contribution improves the fixed condition's macro score.

    Bins and quartiles partition weighted cells; their values are contributions
    to the global condition score, not scores renormalized inside each subgroup.
    """
    import numpy as np
    p, r, y, main = [np.asarray(v, np.float64) for v in (prediction, reference, target, original_main)]
    mask = np.asarray(observed)
    if mask.dtype != np.bool_ or p.ndim != 3 or any(v.shape != p.shape for v in (r, y, main, mask)):
        raise ValueError('SAME_N_H_C_BOOL_MASK_REQUIRED')
    if any(not np.isfinite(v[mask]).all() for v in (p, r, y, main)):
        raise ValueError('FINITE_OBSERVED_PREDICTION_TARGET_REQUIRED')
    pe = np.where(mask, p - y, 0.0)
    re = np.where(mask, r - y, 0.0)
    main_abs = np.abs(np.where(mask, main - y, 0.0))
    counts = mask.sum(axis=(0, 1)).astype(np.int64)
    active = counts > 0
    active_count = int(active.sum())
    weights = np.divide(1.0, active_count * counts, out=np.zeros(len(counts)), where=active) if active_count else np.zeros(len(counts))
    mse = (pe * pe - re * re) * weights
    mae = (np.abs(pe) - np.abs(re)) * weights
    pred_scores = {'mse': float((pe * pe * weights).sum()) if active_count else None,
                   'mae': float((np.abs(pe) * weights).sum()) if active_count else None}
    ref_scores = {'mse': float((re * re * weights).sum()) if active_count else None,
                  'mae': float((np.abs(re) * weights).sum()) if active_count else None}
    delta = {k: pred_scores[k] - ref_scores[k] if active_count else None for k in pred_scores}
    quartile = (4 * np.arange(p.shape[0], dtype=np.int64)) // p.shape[0] if p.shape[0] else np.empty(0, np.int64)
    bins = np.searchsorted(np.asarray(BIN_EDGES), main_abs, side='right') - 1
    channel, block, error_bins = [], [], []
    for c in range(p.shape[2]):
        common = {'channel_index': c, 'global_channel_observed_count': int(counts[c]),
                  'global_active_channels': active_count, 'per_observed_cell_weight': float(weights[c])}
        channel.append({**common, 'observed_count': int(counts[c]),
                        'signed_mse_contribution': float(mse[:, :, c].sum()),
                        'signed_mae_contribution': float(mae[:, :, c].sum())})
        for q in range(4):
            selected = quartile == q
            block.append({**common, 'block_quartile': q, 'window_count': int(selected.sum()),
                          'first_window_index': int(np.flatnonzero(selected)[0]) if selected.any() else None,
                          'last_window_index': int(np.flatnonzero(selected)[-1]) if selected.any() else None,
                          'observed_count': int(mask[selected, :, c].sum()),
                          'signed_mse_contribution': float(mse[selected, :, c].sum()),
                          'signed_mae_contribution': float(mae[selected, :, c].sum())})
        for b, label in enumerate(BIN_LABELS):
            selected = mask[:, :, c] & (bins[:, :, c] == b)
            error_bins.append({**common, 'MAIN_abs_error_bin': label, 'bin_index': b,
                               'observed_count': int(selected.sum()),
                               'signed_mse_contribution': float(mse[:, :, c][selected].sum()),
                               'signed_mae_contribution': float(mae[:, :, c][selected].sum())})
    residuals = {}
    for name, rows in (('channel', channel), ('block', block), ('error_bins', error_bins)):
        for metric in ('mse', 'mae'):
            total = sum(row['signed_' + metric + '_contribution'] for row in rows)
            expected = delta[metric] if delta[metric] is not None else 0.0
            difference = total - expected
            if abs(difference) > TOLERANCE:
                raise ValueError('SIGNED_CONTRIBUTION_CONSERVATION_FAILED:' + name + ':' + metric)
            residuals[name + '_' + metric] = difference
        if sum(row['observed_count'] for row in rows) != int(mask.sum()):
            raise ValueError('PARTITION_OBSERVED_COUNT_CONSERVATION_FAILED:' + name)
    return {'prediction_scores': pred_scores, 'reference_scores': ref_scores, 'delta': delta,
            'observed_count': int(mask.sum()), 'active_channels': active_count,
            'conservation_residuals': residuals, 'channel': channel, 'block': block, 'error_bins': error_bins}


def tune_margins(family):
    """Describe the already selected index; never change selection or tie policy."""
    grid = family['candidate_grid']
    index = family['selected_index']
    selected = grid[index]
    score = selected['tune_actual_Y_mse']
    if selected['candidate_id'] != family['selected_candidate_id'] or selected['status'] != 'COMPLETE' or score is None:
        raise ValueError('ACTUAL_SELECTED_TUNE_RECORD_REQUIRED')
    other = [r['tune_actual_Y_mse'] for i, r in enumerate(grid)
             if i != index and r['status'] == 'COMPLETE' and r.get('tune_actual_Y_mse') is not None]
    zero, mean = grid[0]['tune_actual_Y_mse'], grid[1]['tune_actual_Y_mse']
    rows = []
    for i, row in enumerate(grid):
        value = row.get('tune_actual_Y_mse')
        rows.append({'candidate_index': i, 'candidate_id': row['candidate_id'], 'candidate_status': row['status'],
                     'alpha': row.get('alpha'), 'candidate_tune_mse': value, 'is_selected': i == index,
                     'selected_candidate_id': selected['candidate_id'], 'selected_tune_mse': score,
                     'candidate_minus_selected_mse': value - score if value is not None else None,
                     'selected_improvement_vs_ZERO': zero - score if zero is not None else None,
                     'selected_improvement_vs_MEAN': mean - score if mean is not None else None,
                     'selected_margin_to_best_other': min(other) - score if other else None,
                     'selection_changed': False})
    return rows


def verify_score(actual, declared):
    for key in ('mse', 'mae'):
        a, b = actual[key], declared[key]
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > TOLERANCE:
            raise ValueError('DECLARED_SAVED_SCORE_MISMATCH:' + key)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(args):
    import main_residual_runtime_v1 as runtime
    limit = None
    folder = resolve(args.out)
    initialized = False
    sources, totals = [], []
    result = {'schema': 'main_residual_error_diagnosis_v1', 'status': 'RUNNING', 'GPU_execution': False,
              'E_D_F_loaded': False, 'future_oracle_read': False, 'protected_TEST_accessed': False,
              'new_fit': False, 'selection_changed': False, 'independent_holdout': False,
              'confidence_intervals': None, 'window_partition': 'floor(4*i/N), input window order',
              'MAIN_absolute_error_bin_edges': ['0', '0.25', '0.5', '1', '2', 'inf'],
              'contribution_weights': '1/(active_C * observed_count_channel) fixed per condition/split',
              'totals': totals, 'input_records': sources}
    try:
        # Authorization and the serial CPU lease precede output creation or array reads.
        limit = runtime.ContinuationLimit(args.tag, gpu=False)
        folder.mkdir(parents=True, exist_ok=False); initialized = True
        result.update(authorization_record=limit.authorization_record, run_manifest_record=limit.run_manifest_record)
        limit.checkfull()
        import numpy as np
        import audit_main_residual_v1 as array_io
        producers, records = [], []
        for path in args.results:
            value, record = json_record(path)
            producers.append(value); records.append(record); sources.append(record)
        audit, audit_record = json_record(args.audit)
        audit_binding(audit, producers, records); sources.append(audit_record)
        for producer in producers:
            if (producer['permission_gate']['sha256'] != limit.authorization_record['sha256']
                    or producer['run_manifest']['sha256'] != limit.run_manifest_record['sha256']):
                raise ValueError('SAME_CURRENT_APPROVED_CAMPAIGN_REQUIRED')
        if not audit.get('source_records'):
            raise ValueError('AUDITOR_SOURCE_SHA_RECORD_REQUIRED')
        for record in audit['source_records']:
            sources.append(checked(record))
        for path in (Path(__file__), HERE / 'main_residual_runtime_v1.py', HERE / 'audit_main_residual_v1.py'):
            sources.append(checked({'path': str(path), 'sha256': sha(path)}))
        result['independent_science_audit'] = audit_record
        channel_rows, block_rows, bin_rows, margin_rows = [], [], [], []
        for producer in producers:
            limit.checkfull()
            identity = {'dataset': producer['dataset'], 'seed': producer['seed']}
            for record in producer['source_records'] + producer['source_snapshots']:
                sources.append(checked(record))
                if record.get('snapshot_path') and sha(record['snapshot_path']) != record['sha256']:
                    raise ValueError('PRODUCER_SOURCE_SNAPSHOT_CHANGED')
            splits = ('fit', 'tune', 'val' if producer['dataset'] == 'robin' else 'eval')
            for route in ROUTES:
                margin_rows.extend({**identity, 'route': route, 'route_status': producer['families'][route]['status'], **r}
                                   for r in tune_margins(producer['families'][route]))
            for split in splits:
                limit.checkfull()
                predictions, base = {}, None
                for route in ROUTES:
                    family = producer['families'][route]
                    arrays, record = array_io.load_npz(family['selected_artifacts'][split]); sources.append(record)
                    if base is None:
                        base = arrays
                    else:
                        for key in ('target', 'observed', 'origins', 'mean', 'std', 'original_MAIN_prediction'):
                            array_io.exact_array(arrays[key], base[key], 'DIAGNOSIS_SAME_CONDITION:' + key, equal_nan=True)
                    predictions[route] = arrays['prediction']
                mean_arrays, record = array_io.load_npz(producer['controls']['MEAN_CORRECTION']['artifacts'][split])
                sources.append(record)
                array_io.exact_array(mean_arrays['origins'], base['origins'], 'SHARED_MEAN_ORIGINS')
                predictions['MEAN_CORRECTION'] = mean_arrays['prediction']
                predictions['ORIGINAL_MAIN'] = base['original_MAIN_prediction']
                comparisons = [(r, 'ORIGINAL_MAIN') for r in ('MEAN_CORRECTION',) + ROUTES] + [('E33K', 'MEAN_CORRECTION')]
                for route, reference in comparisons:
                    limit.shared.checkfast()
                    values = decompose(predictions[route], predictions[reference], base['target'], base['observed'], predictions['ORIGINAL_MAIN'])
                    def declared(name):
                        if name == 'ORIGINAL_MAIN': return producer['references'][split][name]['normalized']
                        if name == 'MEAN_CORRECTION': return producer['controls'][name]['splits'][split]['normalized']
                        return producer['families'][name]['splits'][split]['normalized']
                    verify_score(values['prediction_scores'], declared(route)); verify_score(values['reference_scores'], declared(reference))
                    common = {**identity, 'split': split, 'method': route, 'reference': reference,
                              'selection': producer['families'][route]['selected_candidate_id'] if route in ROUTES else 'SHARED_FIT_MEAN',
                              'reference_selection': producer['families'][reference]['selected_candidate_id'] if reference in ROUTES else reference}
                    for field, destination in (('channel', channel_rows), ('block', block_rows), ('error_bins', bin_rows)):
                        destination.extend({**common, **row} for row in values[field])
                    totals.append({**common, **{k: v for k, v in values.items() if k not in ('channel', 'block', 'error_bins')}})
                del predictions, base, mean_arrays
        result['row_counts'] = dict(channel=len(channel_rows), block=len(block_rows), error_bins=len(bin_rows), tune_margins=len(margin_rows))
        for name, rows in (('CHANNEL_CONTRIBUTIONS.csv', channel_rows), ('BLOCK_CONTRIBUTIONS.csv', block_rows),
                           ('ERROR_BINS.csv', bin_rows), ('TUNE_MARGINS.csv', margin_rows)):
            write_csv(folder / name, rows)
        text = korean_report(totals)
        with (folder / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(text)
        limit.checkfull()
        for record in sources:
            limit.shared.checkfast(); checked(record)
        result['source_records_after'] = [checked(record) for record in sources]
        result['source_artifacts_before_after_SHA_pass'] = True
        result['files'] = [checked({'path': str(folder / name), 'sha256': sha(folder / name)}) for name in
                           ('CHANNEL_CONTRIBUTIONS.csv', 'BLOCK_CONTRIBUTIONS.csv', 'ERROR_BINS.csv', 'TUNE_MARGINS.csv', 'REPORT_KO.md')]
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type': type(error).__name__, 'message': str(error)})
    finally:
        receipt = limit.close(result['status'], result.get('failure')) if limit else runtime.LAST_CONSTRUCTOR_TERMINAL
        result['resource_receipt'] = receipt
        if receipt and receipt.get('terminal_status') != 'COMPLETE':
            result.update(status=receipt['terminal_status'], resource_terminal_error=receipt.get('terminal_error'))
        if initialized:
            write_json(folder / 'MANIFEST.json', result)
    print(json.dumps({'status': result['status'], 'out': str(folder), 'row_counts': result.get('row_counts')}, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' else 1


def korean_report(totals):
    lines = ['# MAIN 잔차 보정: 저장된 오차의 기여 분해', '',
             '선택된 예측을 변경하지 않고 같은 실제 Y 관측 마스크로 오차 차이를 분해했다. 음수 기여는 해당 조건의 전체 정규화 채널 macro 오차 개선이다. 각 채널·구간·오차 크기별 값을 다시 정규화하지 않았으므로 각 분해의 합이 전체 점수 차이와 같다.', '',
             '여러 겹치는 창과 두 seed를 독립 표본으로 간주하지 않았다. FIT/TUNE 및 반복 노출된 VAL/EVAL 진단이며, 신뢰구간·유의성·새 모델 선택·일반화 증거를 제공하지 않는다. MSE 개선과 MAE 악화가 함께 있으면 그 교환을 보존해 해석해야 한다.', '',
             '윈도우 블록은 입력 순서의 `floor(4*i/N)`이다. 오차 구간은 모든 비교에서 원래 MAIN의 정규화 절대 오차 `[0,.25,.5,1,2,inf]`로 고정했다. E33K와 MEAN 비교도 같은 구간을 쓴다.', '',
             '- [채널 기여](CHANNEL_CONTRIBUTIONS.csv)', '- [윈도우 블록 기여](BLOCK_CONTRIBUTIONS.csv)',
             '- [MAIN 오차 크기별 기여](ERROR_BINS.csv)', '- [기존 TUNE 선택 여유](TUNE_MARGINS.csv)',
             '- [입력 SHA·검산·자원 기록](MANIFEST.json)', '', '```text',
             'dataset seed split method/reference signed_MSE_delta signed_MAE_delta']
    for row in totals:
        if row['split'] not in ('val', 'eval'): continue
        delta = row['delta']
        mse = format(delta['mse'], '.12g') if delta['mse'] is not None else 'NO_OBSERVED_TARGET'
        mae = format(delta['mae'], '.12g') if delta['mae'] is not None else 'NO_OBSERVED_TARGET'
        lines.append(f"{row['dataset']} {row['seed']} {row['split']} {row['method']}/{row['reference']} {mse} {mae}")
    lines.extend(['```', '', 'ZERO 선택은 기존 MAIN을 보존한 것이므로 새 잔차 학습의 개선으로 해석할 수 없다. E33K와 공유 FIT 평균 보정을 비교하면 과거 K 정보의 추가 이득과 단순 편향 보정 효과를 구분하는 데 도움이 되지만, 이 분해 자체가 인코더·예측기·디코더 중 단일 원인을 확정하지는 않는다.', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true'); mode.add_argument('--execute', action='store_true')
    parser.add_argument('--results', nargs=4)
    parser.add_argument('--audit')
    parser.add_argument('--tag', default='main_residual_error_diagnosis_v1')
    parser.add_argument('--out', default=str(HERE / 'report_main_residual_error_diagnosis_v1'))
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'status': 'SOURCE_ONLY_NOT_EXECUTED', 'methods': list(ROUTES),
                          'comparisons_per_split': 5, 'conditions': 4, 'new_fit': False, 'selection_changed': False}))
        return 0
    if not args.results or not args.audit:
        parser.error('--execute requires --results FOUR and --audit')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
