"""Save actual post-cost deployment arrays and independently audit their parity.

The 7-repeat latency experiment stays sealed. This is a separate single-call
reinference per route/batch, with its own GPU resource receipt and no training.
"""
from pathlib import Path
import argparse
import gc
import time
import traceback
import numpy as np
import torch
import runtime as rt
import run as ru
import train as tr
import benchmark as bench
import model as bm
import data as dt


def independent_preprocess(raw, observed, mean, std):
    raw, observed = np.asarray(raw, np.float32), np.asarray(observed, bool)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if raw.shape != observed.shape or raw.ndim != 3 or raw.shape[-1] != len(mean) or mean.shape != std.shape:
        raise ValueError('INPUT_PREPROCESS_SHAPE')
    if not np.isfinite(raw[observed]).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('INPUT_PREPROCESS_NONFINITE_OR_NONPOSITIVE_SCALE')
    result = np.empty_like(raw)
    for window in range(len(raw)):
        last = mean.copy()
        for row in range(raw.shape[1]):
            for channel in range(raw.shape[2]):
                if observed[window, row, channel]: last[channel] = raw[window, row, channel]
            result[window, row] = (last - mean) / std
    return result


def independent_parity(actual_physical, science_prediction, mean, std, atol=2e-3, rtol=2e-5):
    actual = np.asarray(actual_physical)
    science = np.asarray(science_prediction)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if actual.shape != science.shape or actual.ndim != 3 or actual.shape[1] != 48 or actual.shape[2] != len(mean):
        raise ValueError('DEPLOYMENT_OUTPUT_SHAPE')
    if not np.isfinite(actual).all() or not np.isfinite(science).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('DEPLOYMENT_OUTPUT_NONFINITE_OR_INVALID_SCALE')
    # Match the sealed benchmark's two FP32 inverse-normalization operations.
    reference = np.add(np.multiply(science.astype(np.float32), std), mean)
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowance = atol + rtol * np.abs(reference.astype(np.float64))
    normalized_actual = (actual.astype(np.float64) - mean.astype(np.float64)) / std.astype(np.float64)
    normalized_error = np.abs(normalized_actual - science.astype(np.float64))
    return {'physical_parity_pass': bool(np.all(error <= allowance)), 'physical_atol': atol, 'physical_rtol': rtol,
        'physical_max_abs': float(error.max()), 'physical_mean_abs': float(error.mean()),
        'physical_max_error_to_allowance': float((error / allowance).max()),
        'failed_physical_cells': int((error > allowance).sum()), 'normalized_max_abs': float(normalized_error.max()),
        'normalized_mean_abs': float(normalized_error.mean()), 'normalized_per_channel_max_abs': normalized_error.max((0, 1)).tolist(),
        'total_output_cells': int(actual.size), 'normalized_gap_note': 'Reported independently; the declared acceptance criterion is the original physical-unit tolerance'}


def _read_result(path):
    value = rt.read(path)
    if value['status'] != 'COMPLETE': raise ValueError('SOURCE_RESULT_NOT_COMPLETE: ' + str(path))
    return value, {'path': str(path.resolve().relative_to(rt.ROOT)), 'sha256': rt.sha(path)}


def _expected_routes(dev, fresh, probes, evaluation, lock):
    cases = ru.selected_cases(dev, fresh, lock)
    ru.assert_full_training(dev, fresh)
    ru.assert_selected_coverage(cases, probes['readouts'], probes['controls'], lock)
    routes = []
    aliases = []
    for case in cases:
        record = next(r for r in evaluation['core_evaluation'] if r['dataset'] == case['dataset'] and r['source_label'] == case['label'])
        science = next(r['prediction'] for r in record['entries'] if r['variant'] == 'regular')
        routes.append({'dataset': case['dataset'], 'identity': case['label'], 'route': 'CORE', 'split': record['split'],
            'case': case, 'science_prediction': science, 'model_artifacts': [case['training']['selected_checkpoint']]})
    for control in probes['controls']:
        for method in control['methods']:
            record = next(r for r in evaluation['control_evaluation'] if r['dataset'] == control['dataset'] and r['method'] == method['method'])
            routes.append({'dataset': control['dataset'], 'identity': method['method'], 'route': method['method'], 'split': record['split'],
                'control': method, 'science_prediction': record['prediction'], 'model_artifacts': [] if 'model' not in method else [method['model']]})
    for readout in probes['readouts']:
        case = next(c for c in cases if c['dataset'] == readout['dataset'] and c['label'] == readout['source_label'])
        for method in readout['readouts']:
            if method.get('duplicate_of'):
                aliases.append({'dataset': case['dataset'], 'identity': case['label'] + '_' + method['method'],
                    'route': method['method'], 'canonical_identity': case['label'] + '_' + method['duplicate_of'],
                    'canonical_route': method['duplicate_of'], 'additional_reinference': False})
                continue
            record = next(r for r in evaluation['readout_evaluation'] if r['dataset'] == case['dataset'] and
                r['source_label'] == case['label'] and r['method'] == method['method'])
            routes.append({'dataset': case['dataset'], 'identity': case['label'] + '_' + method['method'],
                'route': method['method'], 'split': record['split'], 'case': case, 'readout': method,
                'science_prediction': record['prediction'], 'model_artifacts': [case['training']['selected_checkpoint'], method['model']]})
    identities = [(r['dataset'], r['identity'], r['route']) for r in routes]
    if len(routes) != 268 or len(set(identities)) != 268: raise ValueError('DEPLOYMENT_ROUTE_COVERAGE_NOT_EXACT_268')
    return routes, aliases


def _cost_mapping(cost, routes):
    expected = {(r['dataset'], r['identity'], r['route'], batch) for r in routes for batch in (1, 8)}
    records = cost['records']
    actual = [(r['dataset'], r['identity'], r['route'], r['batch']) for r in records]
    if len(actual) != 536 or len(set(actual)) != 536 or set(actual) != expected:
        raise ValueError('SEALED_COST_COVERAGE_NOT_EXACT_536')
    result = {}
    for record in records:
        if record['warmup'] != 2 or record['repeats'] != 7 or len(record['seconds']) != 7 or not record['parity_pass']:
            raise ValueError('SEALED_COST_ROW_NOT_VALID_SEVEN_REPEAT')
        result[(record['dataset'], record['identity'], record['route'], record['batch'])] = record
    return result


def _active_metadata(route, model, ridge, backbone):
    fitted_coefficients = 0 if ridge is None else sum(value.numel() for value in ridge.values())
    metadata = {'neural_model_registered_parameters': 0 if model is None else sum(p.numel() for p in model.parameters()),
        'fitted_affine_coefficients': fitted_coefficients, 'trainable_parameters_at_audit': 0,
        'frozen_backbone_registered_parameters': sum(p.numel() for p in backbone.model.parameters()),
        'frozen_backbone_active_at_route': route in ('CORE', 'RAW_BOLT', 'RAW_BOLT_RAW_R'),
        'full_checkpoint_co_resident': model is not None, 'registered_and_active_are_different': True}
    if model is None:
        metadata['active_neural_adaptation_parameters'] = 0
        return metadata
    metadata['full_model_metadata'] = model.metadata()
    if route == 'CORE':
        metadata['active_neural_adaptation_parameters'] = model.metadata()['deployment_active_structural_parameters']
    else:
        prefixes = ['codec.codec.project.', 'codec.codec.input.', 'codec.codec.temporal.', 'codec.codec.output.']
        if route == 'E_ENRICHED_RIDGE' and model.method.startswith('B_'):
            prefixes += ['student_projection.', 'student_fusion.']
            if model.method != 'B_UNIFORM': prefixes += ['student_hash.']
        metadata['active_neural_adaptation_parameters'] = sum(p.numel() for name, p in model.named_parameters() if name.startswith(tuple(prefixes)))
        metadata['decoder_and_F_not_used_by_readout'] = True
    return metadata


def _verify_file(artifact):
    path = rt.ROOT / artifact['path']
    if rt.sha(path) != artifact['sha256']: raise ValueError('DEPLOYMENT_AUDIT_MODEL_OR_SCIENCE_SHA: ' + str(path))


def _load_science(route, pack):
    artifact, split = route['science_prediction'], route['split']
    _verify_file(artifact)
    with np.load(rt.ROOT / artifact['path'], allow_pickle=False) as source:
        expected = source[split + '_prediction'].copy()
        origins = source[split + '_origins'].copy()
        source_mean, source_std = source['mean'].copy(), source['std'].copy()
    if not np.array_equal(origins, pack['origins'][split]): raise ValueError('SCIENCE_VS_ACTUAL_INPUT_ORIGINS_MISMATCH')
    if not np.array_equal(source_mean, pack['mean']) or not np.array_equal(source_std, pack['std']):
        raise ValueError('SCIENCE_VS_ACTUAL_FIT_STATS_MISMATCH')
    if expected.shape != (len(origins), 48, len(pack['mean'])): raise ValueError('SCIENCE_SOURCE_PREDICTION_SHAPE')
    return expected


def main():
    parser = argparse.ArgumentParser()
    for name in ('dev-result', 'fresh-result', 'probe-result', 'evaluation-result', 'cost-result', 'method-lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    folder = rt.HERE / 'deployment_output_audit_v1' / args.tag; folder.mkdir(parents=True, exist_ok=False)
    output = {'tag': args.tag, 'status': 'INCOMPLETE', 'started_unix': time.time(), 'records': [], 'failed_routes': [],
        'expected_canonical_routes': 268, 'expected_batch_outputs': 536, 'new_training_updates': 0,
        'new_model_or_weight_selection': False, 'reinference_calls_per_route_batch': 1,
        'script_sha256': rt.sha(__file__), 'source_results': {},
        'evidence_scope': 'Separate post-cost actual outputs; these arrays are not the original seven timed calls'}
    limit = rt.Limit(args.tag, gpu=True)
    terminal_status, terminal_error = 'COMPLETE', None
    try:
        dev, output['source_results']['dev'] = _read_result(args.dev_result)
        fresh, output['source_results']['fresh'] = _read_result(args.fresh_result)
        probes, output['source_results']['probes'] = _read_result(args.probe_result)
        evaluation, output['source_results']['evaluation'] = _read_result(args.evaluation_result)
        cost, output['source_results']['cost'] = _read_result(args.cost_result)
        lock = ru.check_method_sources(args.method_lock)
        output['method_lock'] = {'path': str(args.method_lock.resolve().relative_to(rt.ROOT)), 'sha256': rt.sha(args.method_lock)}
        output['scientific_source_hashes'] = lock['source_hashes']
        routes, output['aliases'] = _expected_routes(dev, fresh, probes, evaluation, lock)
        timed = _cost_mapping(cost, routes)
        receipts = evaluation['evaluation_open_receipts']
        if {r['dataset'] for r in receipts} != set(ru.FRESH) or len(receipts) != 2:
            raise ValueError('EVALUATION_NOT_ALREADY_OPENED_FOR_BOTH_DATASETS')
        output['original_evaluation_open_receipts'] = receipts
        bm.en.configure()
        forecaster = bm.en.ex.previous.FrozenBolt(); bm.en.replace_normalizer(forecaster)
        output['backbone_before_sha256'] = rt.state_hash(forecaster.model)
        if any(result['backbone_before_sha256'] != output['backbone_before_sha256'] or
               result['backbone_after_sha256'] != output['backbone_before_sha256'] for result in (dev, fresh, cost)):
            raise ValueError('SOURCE_BACKBONE_HASH_NOT_MATCHING_CURRENT_DEPLOYMENT')
        packs = {ds: ru.load_pack(ds, args.method_lock) for ds in ru.LEGACY + ru.FRESH}
        global_artifact = evaluation['global_model_lock']; _verify_file(global_artifact)
        global_path = rt.ROOT / global_artifact['path']
        output['global_model_lock'] = global_artifact
        output['already_opened_EVAL_reopened_only_after_global_gate'] = []
        for ds in ru.FRESH:
            packs[ds] = ru.add_evaluation(packs[ds], dt.open_new_evaluation(ds, global_path, args.method_lock))
            output['already_opened_EVAL_reopened_only_after_global_gate'].append(ds)
        for route in routes:
            model = ridge = None
            pack = packs[route['dataset']]
            try:
                ru.source_check(lock['source_hashes'])
                for artifact in route['model_artifacts']: _verify_file(artifact)
                expected = _load_science(route, pack)
                if 'case' in route: model = tr.load_model(pack, route['case'], rt)
                if 'readout' in route: ridge = bench.load_ridge(route['readout'], rt)
                elif 'control' in route and 'model' in route['control']: ridge = bench.load_ridge(route['control'], rt)
                active = _active_metadata(route['route'], model, ridge, forecaster)
                for batch in (1, 8):
                    raw = pack['raw_history'][route['split']][:batch]
                    observed = pack['raw_history_observed'][route['split']][:batch]
                    origins = pack['origins'][route['split']][:batch]
                    if len(raw) != batch or len(expected) < batch: raise ValueError('INSUFFICIENT_AUDIT_BATCH_WINDOWS')
                    with torch.no_grad():
                        actual = bench.deployment(pack, raw, observed, origins, route['route'], forecaster, model, ridge)
                        torch.cuda.synchronize()
                    independent_input = independent_preprocess(raw, observed, pack['mean'], pack['std'])
                    expected_input = pack['history'][route['split']][:batch]
                    input_gap = float(np.max(np.abs(independent_input.astype(np.float64) - expected_input.astype(np.float64))))
                    if input_gap > 1e-6: raise ValueError('INPUT_PREPROCESS_CONTRACT_MISMATCH: ' + str(input_gap))
                    parity = independent_parity(actual, expected[:batch], pack['mean'], pack['std'])
                    filename = route['dataset'] + '_' + route['identity'] + '_B' + str(batch) + '.npz'
                    actual_output = rt.save_npz(folder / filename, actual_physical=actual, science_prediction=expected[:batch],
                        raw_history=raw, history_observed=observed, origins=origins,
                        mean=pack['mean'], std=pack['std'], source_history_normalized=expected_input,
                        independent_history_normalized=independent_input)
                    record = {'dataset': route['dataset'], 'identity': route['identity'], 'route': route['route'],
                        'split': route['split'], 'batch': batch, 'actual_output': actual_output,
                        'science_prediction': route['science_prediction'], 'model_artifacts': route['model_artifacts'],
                        'source_science_sha256_verified': True, 'source_model_sha256_verified': True,
                        'source_input_origins_exact': True, 'source_mean_std_exact': True,
                        'input_preprocess_max_abs': input_gap, 'independent_CPU_parity': parity,
                        'active_model_metadata': active, 'backbone_sha256': output['backbone_before_sha256'],
                        'new_actual_forward_calls': 1, 'not_added_to_latency_statistics': True,
                        'original_seven_repeat_timing': {'source_result': output['source_results']['cost'],
                            'warmup': timed[(route['dataset'], route['identity'], route['route'], batch)]['warmup'],
                            'repeats': timed[(route['dataset'], route['identity'], route['route'], batch)]['repeats'],
                            'median_seconds': timed[(route['dataset'], route['identity'], route['route'], batch)]['median_seconds']}}
                    output['records'].append(record)
                    rt.write(folder / filename.replace('.npz', '.json'), record)
                    print(rt.json_line({'actual_deployment_output': filename, 'physical_parity': parity['physical_parity_pass'],
                        'normalized_max_abs': parity['normalized_max_abs']}), flush=True)
                    if not parity['physical_parity_pass']:
                        output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'], 'batch': batch,
                            'reason': 'INDEPENDENT_ACTUAL_PHYSICAL_OUTPUT_PARITY_FAILED', 'actual_output': actual_output})
                    limit.check()
            except Exception as exc:
                if isinstance(exc, rt.BudgetHeld) or str(exc).startswith('ACTIVE_SOURCE_CHANGED'): raise
                output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'], 'route': route['route'],
                    'error': str(exc), 'traceback': traceback.format_exc(), 'reason': 'PATH_FAILURE_PRESERVED'})
            finally:
                del model, ridge
                gc.collect()
        output['backbone_after_sha256'] = rt.state_hash(forecaster.model)
        if output['backbone_after_sha256'] != output['backbone_before_sha256']: raise ValueError('BACKBONE_CHANGED_DURING_DEPLOYMENT_AUDIT')
        ru.source_check(lock['source_hashes'])
        expected_keys = {(r['dataset'], r['identity'], r['route'], batch) for r in routes for batch in (1, 8)}
        actual_keys = [(r['dataset'], r['identity'], r['route'], r['batch']) for r in output['records']]
        coverage = len(actual_keys) == 536 and len(set(actual_keys)) == 536 and set(actual_keys) == expected_keys
        output['exact_route_batch_coverage'] = coverage
        output['status'] = 'COMPLETE' if coverage and not output['failed_routes'] else 'FAILED_PRESERVED'
        terminal_status = output['status']
    except Exception as exc:
        terminal_status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        terminal_error = str(exc)
        output.update(status=terminal_status, error=terminal_error, traceback=traceback.format_exc())
    output['actual_saved_batch_outputs'] = len(output['records'])
    output['missing_batch_outputs'] = max(536 - len(output['records']), 0)
    output['finished_unix'] = time.time()
    output['resource_receipt'] = limit.close(terminal_status, terminal_error)
    rt.write(folder / 'RESULT.json', output)
    print(rt.json_line({'deployment_output_audit_terminal': str(folder / 'RESULT.json'), 'status': output['status'],
        'actual_saved_batch_outputs': len(output['records'])}), flush=True)
    if output['status'] != 'COMPLETE': raise SystemExit(2)


if __name__ == '__main__': main()
