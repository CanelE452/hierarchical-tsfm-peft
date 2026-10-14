"""Predeclared CPU-only physical-channel budget simulation, not deployment.

Uses existing full-C RAW forecasts and full-C current-state bypass. No model or
GPU inference. The future-dependent oracle is a restricted-menu diagnostic.
"""
from pathlib import Path
import argparse
import csv
import inspect
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'channel_budget'
DATASETS = {'robin': 4, 'beijing_aotizhongxin': 2}
METHODS = ('LAST', 'FIT_FIXED_TOPK', 'PAST_VARIANCE_TOPK', 'FUTURE_ORACLE_TOPK')
PAST_SUPPORT = 128
DECLARATION = {
    'methods_before_scientific_reads': list(METHODS), 'K': DATASETS, 'past_support': PAST_SUPPORT,
    'FIT_FIXED_TOPK': 'FIT channel-average normalized actual-Y MSE gain LAST minus cached full-C RAW; missing channel gain zero',
    'PAST_VARIANCE_TOPK': 'per-query variance of normalized causal-filled past last128 only, ddof0',
    'FUTURE_ORACLE_TOPK': 'actual-future per-query SSE gain weighted by FULL-SPLIT channel observed count and active-channel count',
    'ties': 'ascending physical channel index; exactly K selected even when gains negative',
    'selected_forecast': 'cached full-C RAW on selected K physical channels, current LAST on other channels',
    'full_C_RAW_reference': 'context reference, not a fifth selection method',
    'model_or_method_selection': 'none; preserve all four methods and all three original splits',
    'actual_K_channel_F_queries': False, 'cheap_deployment_or_cost_proof': False,
    'strict_K_only_compression': False, 'future_oracle_deployable': False,
    'universal_forecasting_bound': False, 'independent_holdout': False,
    'historical_full_C_teacher_generation_cost': 'unknown, not assumed zero',
}


def stable_topk(scores, k):
    values = np.asarray(scores, np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or not np.isfinite(values).all() or not 0 <= k <= values.shape[1]:
        raise ValueError('FINITE_N_BY_C_SCORES_AND_VALID_K_REQUIRED')
    return np.argsort(-values, axis=1, kind='stable')[:, :k].astype(np.int64)


def observed_score(prediction, target, observed, std=None):
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    error = np.where(observed, error, 0.)
    if std is not None:
        error *= np.asarray(std, np.float64)
    counts = observed.sum((0, 1))
    active = counts > 0
    mse = np.divide(np.square(error).sum((0, 1)), counts, out=np.zeros(len(counts)), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), counts, out=np.zeros(len(counts)), where=active)
    return {'mse': float(mse[active].mean()) if active.any() else None,
            'mae': float(mae[active].mean()) if active.any() else None,
            'per_channel_mse': [float(v) if a else None for v, a in zip(mse, active)],
            'per_channel_mae': [float(v) if a else None for v, a in zip(mae, active)],
            'observed_counts': counts.tolist(), 'active_channels': int(active.sum())}


def losses_and_gains(values):
    raw, y, observed = (values[name] for name in ('raw_forecast', 'target', 'observed'))
    last = np.broadcast_to(values['raw_last'][:, None], raw.shape)
    last_error = np.where(observed, np.asarray(last, np.float64) - y.astype(np.float64), 0.)
    raw_error = np.where(observed, raw.astype(np.float64) - y.astype(np.float64), 0.)
    last_sse_query = np.square(last_error).sum(axis=1)
    raw_sse_query = np.square(raw_error).sum(axis=1)
    gain_sse_query = last_sse_query - raw_sse_query
    counts = observed.sum((0, 1))
    active = counts > 0
    gain_macro_query = np.divide(gain_sse_query, counts[None],
        out=np.zeros_like(gain_sse_query), where=active[None]) / max(int(active.sum()), 1)
    fit_gain = np.divide(gain_sse_query.sum(axis=0), counts,
                        out=np.zeros(len(counts)), where=active)
    return {'last': last.copy(), 'last_sse_query': last_sse_query,
            'raw_sse_query': raw_sse_query, 'gain_sse_query': gain_sse_query,
            'gain_macro_query': gain_macro_query, 'channel_mse_gain': fit_gain,
            'full_split_observed_count': counts, 'active_channel_count': int(active.sum())}


def past_variance_scores(history):
    if history.ndim != 3 or history.shape[1] < PAST_SUPPORT or not np.isfinite(history).all():
        raise ValueError('FINITE_PAST_AT_LEAST128_REQUIRED')
    return np.var(history[:, -PAST_SUPPORT:].astype(np.float64), axis=1, ddof=0)


def selection_mask(indices, channels):
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.dtype != np.int64 or np.any(indices < 0) or np.any(indices >= channels):
        raise ValueError('VALID_INT64_SELECTED_INDICES_REQUIRED')
    if any(len(np.unique(row)) != len(row) for row in indices):
        raise ValueError('DUPLICATE_SELECTED_CHANNEL')
    mask = np.zeros((len(indices), channels), bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def simulate_split(values, k, fixed_indices, std):
    raw = values['raw_forecast']
    n, _, channels = raw.shape
    evidence = losses_and_gains(values)
    variance = past_variance_scores(values['history'])
    selections = {
        'LAST': np.empty((n, 0), np.int64),
        'FIT_FIXED_TOPK': np.broadcast_to(fixed_indices[None], (n, k)).copy(),
        'PAST_VARIANCE_TOPK': stable_topk(variance, k),
        'FUTURE_ORACLE_TOPK': stable_topk(evidence['gain_macro_query'], k),
    }
    results = {}
    last_score = observed_score(evidence['last'], values['target'], values['observed'])
    for method in METHODS:
        selected = selection_mask(selections[method], channels)
        prediction = np.where(selected[:, None, :], raw, evidence['last']).astype(np.float32)
        score = observed_score(prediction, values['target'], values['observed'])
        gain = float(evidence['gain_macro_query'][selected].sum())
        actual_gain = last_score['mse'] - score['mse']
        if not np.isclose(gain, actual_gain, atol=1e-11, rtol=1e-11):
            raise ValueError('CHANNEL_MACRO_GAIN_DECOMPOSITION_FAILED')
        results[method] = {'indices': selections[method], 'selection_mask': selected,
            'prediction': prediction, 'normalized': score,
            'physical': observed_score(prediction, values['target'], values['observed'], std),
            'macro_MSE_gain_vs_LAST': gain,
            'channel_selected_query_count': selected.sum(0).tolist(),
            'future_dependent': method == 'FUTURE_ORACLE_TOPK'}
    oracle_mse = results['FUTURE_ORACLE_TOPK']['normalized']['mse']
    for method in ('FIT_FIXED_TOPK', 'PAST_VARIANCE_TOPK'):
        if oracle_mse > results[method]['normalized']['mse'] + 1e-11:
            raise ValueError('RESTRICTED_EXACT_K_ORACLE_OPTIMALITY_FAILED')
    return results, evidence, variance


def validate_splits(splits, mean, std, k):
    if tuple(splits) not in (('fit', 'tune', 'val'), ('fit', 'tune', 'eval')):
        raise ValueError('ORIGINAL_THREE_SPLIT_CONTRACT_REQUIRED')
    channels = len(mean)
    if mean.dtype != np.float32 or std.dtype != np.float32 or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('STORED_FP32_STATS_REQUIRED')
    if not 0 < k <= channels:
        raise ValueError('VALID_K_REQUIRED')
    for split, values in splits.items():
        history, raw, y, observed, last = (values[name] for name in ('history', 'raw_forecast', 'target', 'observed', 'raw_last'))
        if (history.ndim != 3 or history.shape[1:] != (512, channels) or raw.shape != y.shape
                or raw.shape != observed.shape or raw.shape != (len(history), 48, channels)
                or last.shape != (len(history), channels) or observed.dtype != bool
                or any(value.dtype != np.float32 for value in (history, raw, y, last))
                or values['origins'].dtype != np.int64):
            raise ValueError('EXACT_INPUT_SHAPE_DTYPE_REQUIRED:' + split)
        if not np.array_equal(last, history[:, -1]) or not np.isfinite(history).all() or not np.isfinite(raw).all() or not np.isfinite(y[observed]).all():
            raise ValueError('FINITE_OBSERVED_Y_PAST_RAW_EXACT_LAST_REQUIRED:' + split)


def assess(splits, mean, std, k):
    validate_splits(splits, mean, std, k)
    fit = losses_and_gains(splits['fit'])
    fixed = stable_topk(fit['channel_mse_gain'], k)[0]
    output = {'fixed_FIT_indices': fixed.tolist(), 'fit_channel_normalized_MSE_gain': fit['channel_mse_gain'].tolist(),
              'fit_channel_observed_counts': fit['full_split_observed_count'].tolist(), 'splits': {}}
    arrays = {}
    for split, values in splits.items():
        methods, evidence, variance = simulate_split(values, k, fixed, std)
        output['splits'][split] = {'methods': {}, 'full_C_RAW_reference': {
            'normalized': observed_score(values['raw_forecast'], values['target'], values['observed']),
            'physical': observed_score(values['raw_forecast'], values['target'], values['observed'], std)},
            'oracle_full_split_denominator_counts': evidence['full_split_observed_count'].tolist(),
            'oracle_active_channel_count': evidence['active_channel_count']}
        arrays[split] = {'history': values['history'], 'history_observed': values['history_observed'],
            'raw_last': values['raw_last'], 'cached_full_C_RAW': values['raw_forecast'],
            'target': values['target'], 'observed': values['observed'], 'origins': values['origins'],
            'mean': mean, 'std': std, 'past_last128_variance': variance,
            'last_sse_per_query_channel': evidence['last_sse_query'],
            'raw_sse_per_query_channel': evidence['raw_sse_query'],
            'future_oracle_gain_SSE': evidence['gain_sse_query'],
            'future_oracle_gain_weighted_macro': evidence['gain_macro_query'],
            'full_split_observed_counts': evidence['full_split_observed_count'],
            'FIT_fixed_indices': fixed, 'FIT_channel_MSE_gain': fit['channel_mse_gain']}
        for method, row in methods.items():
            output['splits'][split]['methods'][method] = {key: value for key, value in row.items() if key not in ('indices', 'selection_mask', 'prediction')}
            arrays[split].update({method + '_indices': row['indices'], method + '_selection_mask': row['selection_mask'], method + '_prediction': row['prediction']})
    return output, arrays


def metric_rows(dataset, result, columns):
    rows = []
    for split, values in result['splits'].items():
        for method, entry in values['methods'].items():
            for scale in ('normalized', 'physical'):
                score = entry[scale]
                rows.append({'dataset': dataset, 'split': split, 'method': method, 'scale': scale,
                    'channel': 'macro', 'channel_index': '', 'mse': score['mse'], 'mae': score['mae'],
                    'observed_count': sum(score['observed_counts']), 'future_dependent': entry['future_dependent'],
                    'macro_MSE_gain_vs_LAST': entry['macro_MSE_gain_vs_LAST'] if scale == 'normalized' else ''})
                for index, (mse, mae, count) in enumerate(zip(score['per_channel_mse'], score['per_channel_mae'], score['observed_counts'])):
                    rows.append({'dataset': dataset, 'split': split, 'method': method, 'scale': scale,
                        'channel': columns[index], 'channel_index': index, 'mse': mse, 'mae': mae,
                        'observed_count': count, 'future_dependent': entry['future_dependent']})
        for scale in ('normalized', 'physical'):
            score = values['full_C_RAW_reference'][scale]
            rows.append({'dataset': dataset, 'split': split, 'method': 'FULL_C_RAW_REFERENCE', 'scale': scale,
                'channel': 'macro', 'channel_index': '', 'mse': score['mse'], 'mae': score['mae'],
                'observed_count': sum(score['observed_counts']), 'future_dependent': False,
                'reference_not_selection_method': True})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(result):
    lines = ['# 물리 채널 K개 배분의 용량 가능성 진단', '',
        '과학 상태: ' + result['status'] + '. CPU posthoc simulation이며 새 배포 방법이나 실제 K채널 TSFM 추론이 아니다.', '',
        '모든 방법은 현재 상태 C개를 우회 보존한다. 선택 채널에 이미 계산된 full-C RAW forecast를 쓰고 나머지는 LAST를 쓴다. '
        'teacher의 과거 생성 비용은 unknown이며, 이번 CPU 영수증을 경량 추론 비용으로 읽지 않는다.', '',
        'FIT_FIXED는 FIT 실제 Y의 채널 평균 gain으로만 고정한다. PAST_VARIANCE는 현재 query의 과거128 normalized causal-filled step만 본다. '
        'FUTURE_ORACLE는 실제 미래 Y와 전체 split 관측 denominator를 사용하므로 배포할 수 없다. '
        '이는 이 RAW/LAST 메뉴에서 정확히 K개를 고르는 정규화 MSE 진단의 최적 선택이며 보편적인 예측 상한/하한이 아니다.', '',
        '동일 normalized channel-macro MSE와 MAE, 원단위·모든 채널·모든 split을 보존한다. '
        'FIT에서 전부 미관측인 채널의 선택 gain은0, tie는 낮은 channel index, 음수 gain이어도 정확히K를 유지한다. '
        '따라서 restricted oracle도 LAST보다 무조건 좋다는 주장을 하지 않는다.', '', '```text',
        '자료                  split    방법                  MSE             MAE']
    for dataset, entry in result.get('datasets', {}).items():
        for split, values in entry['assessment']['splits'].items():
            for method in METHODS:
                score = values['methods'][method]['normalized']
                lines.append(f"{dataset:<21} {split:<8} {method:<22} {score['mse']:.9f}     {score['mae']:.9f}")
            raw = values['full_C_RAW_reference']['normalized']
            lines.append(f"{dataset:<21} {split:<8} {'FULL_C_RAW_REFERENCE':<22} {raw['mse']:.9f}     {raw['mae']:.9f}")
    lines += ['```', '', '## 판단 범위', '',
        '고정/과거 variance가 좋으면 해당 budget 정책의 개발 가능성을 보인다. oracle만 좋으면 과거만으로 선택할 수 있는지 별도 문제가 남는다. '
        'oracle까지 나쁘면 현재 RAW/LAST 메뉴와 정확히K 배분에서 여지가 적은 것이며 다른 representation/forecast 방법의 불가능성은 아니다. '
        '결과가 나쁘다고 K·rank·method를 평가값에 맞춰 바꾸지 않는다.', '',
        'Robin 반복 VAL과 Beijing 이미 노출된 EVAL은 개발 증거다. TEST·새 독립 일반화를 사용하지 않았다. '
        '원 target·mask·origin·FIT mean/std를 그대로 재사용하고 학습/새 F 호출/CP 갱신은 없다.', '']
    return '\n'.join(lines)


def execute(args):
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import followup_runtime as runtime
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'declaration': DECLARATION, 'datasets': {}, 'artifacts': [],
              'new_model_inference': False, 'GPU_executed': False, 'protected_TEST_accessed': False,
              'old_campaign_resumed': False, 'checkpoint_tensor_loaded': False,
              'new_training_updates': 0, 'independent_holdout': False}
    limit = None
    started = time.perf_counter()
    try:
        limit = runtime.FollowupLimit('channel_budget_' + args.tag, gpu=False)
        inputs.install_protected_test_guard()
        probes.write(output / 'PREDECLARED_METHODS.json', DECLARATION)
        bridge_json = probes.read(args.input_bridge)
        source_records = probes.verify_source_records(bridge_json)
        cp_records = [probes.verified_file(record) for record in bridge_json['declared_asset_hashes']['checkpoints_file_hash_only'].values()]
        result.update(original_sources_before=source_records, original_CP_files_before=cp_records)
        source_folder = output / 'sources'
        source_folder.mkdir(exist_ok=False)
        helper_records = []
        for path in (Path(__file__), HERE / 'run_condition_probes.py', HERE / 'prepare_inputs.py', HERE / 'followup_runtime.py'):
            digest = probes.sha(path)
            shutil.copyfile(path, source_folder / path.name)
            helper_records.append({'path': str(path), 'sha256': digest, 'snapshot_path': str(source_folder / path.name)})
        result['helper_sources_before'] = helper_records
        all_metrics = []
        for dataset, k in DATASETS.items():
            limit.check()
            bridge, splits, mean, std, provenance = probes.load_bridge(args.input_bridge, dataset, inputs.verify_phase_barrier)
            teacher_start = time.perf_counter()
            raw_source = probes.check_original_raw(bridge, dataset, splits, mean, std)
            current_teacher_read_wall = time.perf_counter() - teacher_start
            fit_record = next(record for record in bridge['artifacts'] if record['dataset'] == dataset and record['split'] == 'fit')
            with np.load(probes.resolve(fit_record['path']), allow_pickle=False) as opened:
                columns = opened['columns'].tolist()
            assessment, arrays = assess(splits, mean, std, k)
            records = []
            for split, values in arrays.items():
                record = probes.save_npz(output / f'{dataset}_{split}_ALL_POLICIES.npz', values)
                records.append({'split': split, **record})
                limit.check()
            rows = metric_rows(dataset, assessment, columns)
            all_metrics.extend(rows)
            result['datasets'][dataset] = {'K': k, 'C': len(mean), 'columns': columns,
                'input_provenance': provenance, 'RAW_teacher_source': raw_source,
                'current_teacher_hash_load_compare_wall_seconds': current_teacher_read_wall,
                'historical_teacher_generation_compute_seconds': None,
                'historical_teacher_cost_scope': 'unknown earlier full-C forecast generation, excluded from this CPU receipt',
                'assessment': assessment, 'prediction_and_selection_arrays': records}
            result['artifacts'].extend(records)
            print(json.dumps({'event': 'CHANNEL_BUDGET_DATASET_COMPLETE', 'dataset': dataset, 'K': k,
                'fixed_FIT_indices': assessment['fixed_FIT_indices']}), flush=True)
        write_csv(output / 'ALL_METRICS.csv', all_metrics)
        result['metric_rows'] = len(all_metrics)
        result['original_sources_after'] = probes.verify_source_records(bridge_json)
        result['original_CP_files_after'] = [probes.verified_file(record) for record in cp_records]
        for record in helper_records:
            probes.verified_file(record)
        for entry in result['datasets'].values():
            probes.verified_file(entry['RAW_teacher_source'])
            for record in entry['input_provenance']['input_artifacts']:
                probes.verified_file(record)
        if source_records != result['original_sources_after']:
            raise ValueError('ORIGINAL_SOURCE_PRESERVATION_FAILED')
        limit.check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        budget_type = getattr(getattr(limit, 'rt', None), 'BudgetHeld', ())
        result['status'] = 'BUDGET_HELD_PRESERVED' if budget_type and isinstance(error, budget_type) else 'FAILED_PRESERVED'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        result['body_wall_seconds'] = time.perf_counter() - started
        if limit is not None:
            receipt = limit.close(result['status'], result.get('failure'))
            result['shared_resource_receipt'] = receipt
            probes.apply_resource_terminal(result, receipt)
        elif getattr(runtime, 'LAST_CONSTRUCTOR_TERMINAL', None) is not None:
            result['shared_resource_receipt'] = runtime.LAST_CONSTRUCTOR_TERMINAL
            probes.apply_resource_terminal(result, runtime.LAST_CONSTRUCTOR_TERMINAL)
        probes.write(output / 'RESULT.json', result)
        with (output / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(report(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--input-bridge', type=Path)
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'status': 'PREPARATION_ONLY', 'declaration': DECLARATION}, ensure_ascii=False, indent=2))
        return
    if not args.input_bridge or not args.tag or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for character in args.tag):
        parser.error('execute requires original audited bridge and safe fresh lowercase tag')
    result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(CACHE / args.tag), 'metric_rows': result.get('metric_rows')}, ensure_ascii=False))
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
