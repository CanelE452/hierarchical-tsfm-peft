"""Independent batch-matched outputs for the final canonical deployment routes.

The actual deployment call is separate from the original seven timed calls.
References call the sealed science APIs on independent checkpoint objects;
existing accuracy predictions and checkpoint selection remain unchanged.
"""
import gc
import hashlib
from collections import Counter
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
import torch
import runtime as rt
import batch_science_reference as bs
import train as tr
import probes as pr
import verify_deployment_outputs as original
import run as ru
import model as bm
import benchmark as bench

CUDA_IMPLEMENTATION = 'original_cuda_fp32'
CPU_IMPLEMENTATION = 'cpu_numpy_fp32_ridge_v1'
RIDGE_ROUTES = {'E_RAW_RIDGE', 'E_ENRICHED_RIDGE', 'RAW_SELF33', 'RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R'}


def implementation_mapping(cost, routes):
    expected = {(r['dataset'], r['identity'], r['route'], batch): r for r in routes for batch in (1, 8)}
    producers = cost['timing_producers']; keys = [tuple(row['key']) for row in producers]
    if len(keys) != 536 or len(set(keys)) != 536 or set(keys) != set(expected):
        raise ValueError('COST_IMPLEMENTATION_PRODUCER_COVERAGE')
    rows = {bs.row_key(row): row for row in cost['records']}
    mapping = {}
    for producer in producers:
        key = tuple(producer['key']); route = expected[key]['route']
        wanted = CPU_IMPLEMENTATION if route in RIDGE_ROUTES else CUDA_IMPLEMENTATION
        if route not in RIDGE_ROUTES | {'CORE', 'RAW_BOLT'} or producer['deployment_implementation'] != wanted:
            raise ValueError('COST_IMPLEMENTATION_NOT_RECORDED_UNIFORM_BY_ROUTE')
        if route in RIDGE_ROUTES and rows[key].get('deployment_implementation') != wanted:
            raise ValueError('COST_IMPLEMENTATION_CPU_ROW_PRODUCER_MISMATCH')
        mapping[key] = producer
    if Counter(p['deployment_implementation'] for p in mapping.values()) != {CUDA_IMPLEMENTATION: 296, CPU_IMPLEMENTATION: 240}:
        raise ValueError('COST_IMPLEMENTATION_EXPECTED_296_CUDA_240_CPU')
    return mapping


def sealed_cost_mapping(cost, routes):
    if cost['status'] != 'COMPLETE': raise ValueError('CANONICAL_COST_NOT_COMPLETE')
    expected = {(r['dataset'], r['identity'], r['route'], batch): r['split'] for r in routes for batch in (1, 8)}
    records = cost['records']; keys = [bs.row_key(row) for row in records]
    if len(expected) != 536 or len(keys) != 536 or len(set(keys)) != 536 or set(keys) != set(expected):
        raise ValueError('CANONICAL_COST_COVERAGE_NOT_EXACT_536')
    for row in records:
        times = np.asarray(row['seconds'], np.float64)
        if row['split'] != expected[bs.row_key(row)] or row['warmup'] != 2 or row['repeats'] != 7 or times.shape != (7,) or not np.isfinite(times).all() or np.any(times <= 0):
            raise ValueError('CANONICAL_COST_TIMINGS_NOT_REAL_SEVEN_REPEAT')
        if not row['parity_pass'] or row['physical_parity_atol'] != .002 or row['physical_parity_rtol'] != 2e-5:
            raise ValueError('CANONICAL_COST_PHYSICAL_GATE_CHANGED_OR_FAILED')
        if not np.isclose(row['median_seconds'], np.median(times), rtol=1e-12, atol=0) or not np.isclose(row['median_seconds_per_window'], np.median(times) / row['batch'], rtol=1e-12, atol=0):
            raise ValueError('CANONICAL_COST_MEDIAN_NOT_ACTUAL_SEVEN_REPEAT')
    return {bs.row_key(row): row for row in records}


def load_bulk_prefix(route, pack):
    split = route['split']; path = bs.verify_artifact(route['science_prediction'])
    members = (split + '_prediction', split + '_origins', 'mean', 'std')
    with np.load(path, allow_pickle=False) as source:
        prediction, origins, mean, std = [source[name].copy() for name in members]
    count = len(pack['origins'][split])
    if count < 8 or len(origins) < count or not np.array_equal(origins[:count], pack['origins'][split]):
        raise ValueError('BULK_PREFIX_INPUT_ORIGINS_MISMATCH')
    if not np.array_equal(mean, pack['mean']) or not np.array_equal(std, pack['std']):
        raise ValueError('BULK_PREFIX_FIT_STATISTICS_MISMATCH')
    if prediction.shape != (len(origins), 48, len(mean)) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
        raise ValueError('BULK_PREFIX_PREDICTION_SHAPE_DTYPE_OR_NONFINITE')
    return prediction[:count], {'artifact': route['science_prediction'], 'accessed_members': list(members),
        'full_source_window_count': len(origins), 'loaded_prefix_window_count': count,
        'origins_prefix_exact': True, 'mean_std_exact': True, 'EVAL_target_arrays_opened': False,
        'protected_TEST_opened': False, 'original_bulk_predictions_changed': False}


def model_state(model):
    if model is None: return None
    return {'full_state_sha256': rt.state_hash(model),
        'registered_parameter_sha256': rt.state_hash(dict(model.named_parameters())),
        'buffer_sha256_by_name': {name: rt.state_hash({name: value}) for name, value in model.named_buffers()}}


def coefficient_hash(ridge):
    if ridge is None: return None
    digest = hashlib.sha256()
    for name, value in sorted(ridge.items()):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        array = np.ascontiguousarray(array)
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def independent_science_reference(pack, route, batch, forecaster, limit, device='cuda'):
    split = route['split']
    if batch not in (1, 8) or len(pack['history'][split]) < batch:
        raise ValueError('SCIENCE_REFERENCE_BATCH')
    thin = {'history': {split: pack['history'][split][:batch]},
        'origins': {split: pack['origins'][split][:batch]}}
    model = None
    metadata = {'reference_kind': bs.PROTOCOL['reference_kind'], 'batch': batch,
        'separate_selected_model': 'case' in route, 'EVAL_target_arrays_opened': False,
        'new_model_selection': False, 'not_added_to_latency_statistics': True}
    try:
        if 'case' in route: model = bs.load_selected_model(route['case'], device=device)
        before = model_state(model)
        if route['route'] == 'CORE':
            prediction = tr.predict(model, thin['history'][split], thin['origins'][split],
                forecaster, limit, batch=batch)
            metadata['API'] = 'train.predict(normalized_history[:B],origins[:B],batch=B)'
        elif 'readout' in route:
            prediction = pr.evaluate_readout(thin, split, route['case'], model, route['readout'], rt, limit)
            metadata['API'] = 'probes.evaluate_readout(thin_normalized_past_pack)'
        elif 'control' in route:
            prediction = pr.evaluate_raw_control(thin, split, route['control'], forecaster, rt, limit)
            metadata['API'] = 'probes.evaluate_raw_control(thin_normalized_past_pack)'
        else: raise ValueError('SCIENCE_REFERENCE_ROUTE')
        after = model_state(model)
        metadata.update(model_state_before=before, model_state_after=after)
        changed = []
        if model is not None:
            if before['registered_parameter_sha256'] != after['registered_parameter_sha256']:
                raise ValueError('SCIENCE_REGISTERED_PARAMETERS_CHANGED')
            changed = sorted(name for name in set(before['buffer_sha256_by_name']) | set(after['buffer_sha256_by_name'])
                if before['buffer_sha256_by_name'].get(name) != after['buffer_sha256_by_name'].get(name))
            allowed = {'memory_embedding', 'memory_hash_logits', 'memory_is_frozen'} if 'readout' in route else set()
            if not set(changed) <= allowed: raise ValueError('SCIENCE_UNDECLARED_BUFFER_CHANGED')
        metadata.update(changed_buffer_names=changed, registered_parameters_unchanged=True,
            only_declared_science_memory_buffers_changed=True,
            selected_full_state_unchanged=before == after)
        if prediction.shape != (batch, 48, pack['history'][split].shape[-1]) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
            raise ValueError('SCIENCE_REFERENCE_SHAPE_DTYPE_OR_NONFINITE')
        return prediction, metadata
    finally:
        del model
        gc.collect()


def comparison_bundle(actual, matched, bulk, mean, std):
    matched_check = original.independent_parity(actual, matched, mean, std)
    bulk_check = original.independent_parity(actual, bulk, mean, std)
    normalized = bs.normalized_reinference(matched, bulk)
    return {'actual_to_matched_physical': matched_check,
        'actual_to_original_bulk_physical': bulk_check,
        'matched_science_to_original_bulk_normalized': normalized,
        'critical_parity_gates_pass': bool(matched_check['physical_parity_pass'] and normalized['gate_pass']),
        'original_bulk_physical_failure_relabelled': False,
        'acceptance_note': 'Original physical tolerance against independent same-batch science; prior normalized bound against unchanged bulk; original bulk physical result remains explicit'}


def audit_single_batch(pack, route, batch, matched, bulk, science_metadata,
                       forecaster, model, ridge, deployment):
    split = route['split']
    raw = pack['raw_history'][split][:batch]
    observed = pack['raw_history_observed'][split][:batch]
    origins = pack['origins'][split][:batch]
    history = pack['history'][split][:batch]
    if batch not in (1, 8) or len(raw) != batch or matched.shape != (batch, 48, len(pack['mean'])):
        raise ValueError('OUTPUT_AUDIT_BATCH_OR_REFERENCE_SHAPE')
    normalized_input = original.independent_preprocess(raw, observed, pack['mean'], pack['std'])
    input_gap = float(np.max(np.abs(normalized_input.astype(np.float64) - history.astype(np.float64))))
    before, ridge_before = model_state(model), coefficient_hash(ridge)
    with torch.no_grad():
        actual = deployment(pack, raw, observed, origins, route['route'], forecaster, model, ridge)
        torch.cuda.synchronize()
    after, ridge_after = model_state(model), coefficient_hash(ridge)
    dtype_pass = bool(np.asarray(actual).dtype == np.float32 and matched.dtype == np.float32 and bulk.dtype == np.float32
        and raw.dtype == np.float32 and history.dtype == np.float32 and observed.dtype == np.bool_
        and origins.dtype == np.int64 and pack['mean'].dtype == np.float32 and pack['std'].dtype == np.float32)
    try:
        checks = comparison_bundle(actual, matched, bulk, pack['mean'], pack['std'])
    except ValueError as exc:
        checks = {'critical_parity_gates_pass': False, 'comparison_error': str(exc),
            'actual_to_matched_physical': None, 'original_bulk_physical_failure_relabelled': False}
    parameters_unchanged = model is None or before['registered_parameter_sha256'] == after['registered_parameter_sha256']
    cache_unchanged = model is None or before['buffer_sha256_by_name'] == after['buffer_sha256_by_name']
    checks_pass = bool(checks['critical_parity_gates_pass'] and dtype_pass and input_gap <= 1e-6 and
        parameters_unchanged and cache_unchanged and ridge_before == ridge_after)
    record = {'dataset': route['dataset'], 'identity': route['identity'], 'route': route['route'],
        'split': split, 'batch': batch, 'new_actual_forward_calls': 1,
        'not_added_to_latency_statistics': True, 'EVAL_target_arrays_opened': False,
        'protected_TEST_opened': False, 'new_model_or_weight_selection': False,
        'matched_science_reference': science_metadata, 'comparisons': checks,
        'independent_CPU_parity': checks['actual_to_matched_physical'],
        'FP32_dtype_contract_pass': dtype_pass, 'actual_output_dtype': str(np.asarray(actual).dtype),
        'input_preprocess_max_abs': input_gap, 'independent_input_preprocess_gate_pass': input_gap <= 1e-6,
        'deployment_model_state_before': before, 'deployment_model_state_after': after,
        'deployment_registered_parameters_unchanged': parameters_unchanged,
        'deployment_cache_buffers_unchanged': cache_unchanged,
        'ridge_coefficients_before_sha256': ridge_before, 'ridge_coefficients_after_sha256': ridge_after,
        'ridge_coefficients_unchanged': ridge_before == ridge_after, 'critical_gates_pass': checks_pass}
    arrays = {'actual_physical': np.asarray(actual), 'matched_science_normalized': matched,
        'science_prediction': matched, 'matched_science_physical_CPU_FP32': matched * pack['std'] + pack['mean'],
        'original_bulk_science_normalized': bulk,
        'original_bulk_science_physical_CPU_FP32': bulk * pack['std'] + pack['mean'],
        'raw_history': raw, 'history_observed': observed, 'origins': origins,
        'mean': pack['mean'], 'std': pack['std'], 'source_history_normalized': history,
        'independent_history_normalized': normalized_input}
    return record, arrays


class CappedLimit:
    def __init__(self, limit): self.limit = limit
    def check(self):
        accounting = self.limit.check()
        if accounting['elapsed_seconds'] > 600:
            raise rt.BudgetHeld('DEPLOYMENT_OUTPUT_GPU_PROCESS_WALL_600SECONDS')
        return accounting


def read_complete(path):
    value = rt.read(path)
    if value['status'] != 'COMPLETE': raise ValueError('DEPLOYMENT_SOURCE_NOT_COMPLETE: ' + str(path))
    return value, {'path': path.resolve().relative_to(rt.ROOT).as_posix(), 'sha256': rt.sha(path)}


def unchanged_sources(output):
    ru.source_check(output['source_hashes'])
    references = list(output['source_results'].values()) + [output['method_lock'], output['global_model_lock']]
    references += output['helper_scripts']
    for reference in references:
        try: bs.verify_artifact(reference)
        except ValueError as exc: raise RuntimeError('ACTIVE_SOURCE_CHANGED: ' + reference['path']) from exc


def preflight(args, past_loader):
    lock = ru.check_method_sources(args.method_lock)
    method_ref = {'path': args.method_lock.resolve().relative_to(rt.ROOT).as_posix(), 'sha256': rt.sha(args.method_lock)}
    values, references = {}, {}
    for name in ('dev', 'fresh', 'probe', 'evaluation', 'cost'):
        values[name], references[name] = read_complete(getattr(args, name + '_result'))
        if values[name]['source_hashes'] != lock['source_hashes']:
            raise ValueError('DEPLOYMENT_SOURCES_NOT_ORIGINAL16: ' + name)
    evaluation = values['evaluation']; global_artifact = evaluation['global_model_lock']
    global_lock = rt.read(bs.verify_artifact(global_artifact))
    if not global_lock.get('ALL_SELECTED') or global_lock['method_lock_sha256'] != method_ref['sha256'] or global_lock['source_hashes'] != lock['source_hashes']:
        raise ValueError('DEPLOYMENT_GLOBAL_MODEL_LOCK_NOT_ORIGINAL_ALL_SELECTED')
    receipts = evaluation['evaluation_open_receipts']
    if len(receipts) != 2 or {row['dataset'] for row in receipts} != set(ru.FRESH):
        raise ValueError('DEPLOYMENT_EVAL_NOT_PREVIOUSLY_OPENED_FOR_BOTH_DATASETS')
    for receipt in receipts:
        if not receipt['opened_only_after_all_selected'] or receipt['method_lock_sha256'] != method_ref['sha256'] or receipt['global_model_lock']['sha256'] != global_artifact['sha256']:
            raise ValueError('DEPLOYMENT_PREVIOUS_EVAL_BARRIER_RECEIPT_MISMATCH')
    routes, aliases = bs.build_routes(values['dev'], values['fresh'], values['probe'], evaluation, lock)
    costs = sealed_cost_mapping(values['cost'], routes)
    implementations = implementation_mapping(values['cost'], routes)
    packs, input_evidence = {}, {}
    for dataset in ru.LEGACY + ru.FRESH:
        packs[dataset], input_evidence[dataset] = past_loader.load_past_pack(dataset, values['dev'], values['fresh'], evaluation, args.method_lock, windows=8)
        if 'target' in packs[dataset] or 'observed' in packs[dataset]:
            raise ValueError('DEPLOYMENT_PACK_HAS_FUTURE_TARGET_OR_MASK')
    return values, routes, aliases, costs, implementations, packs, {
        'source_results': references, 'method_lock': method_ref, 'source_hashes': lock['source_hashes'],
        'global_model_lock': global_artifact, 'original_evaluation_open_receipts': receipts,
        'past_input_evidence': input_evidence}


def main():
    parser = argparse.ArgumentParser()
    for name in ('dev-result', 'fresh-result', 'probe-result', 'evaluation-result', 'cost-result', 'method-lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum(): raise ValueError('DEPLOYMENT_EXCLUSIVE_TAG_INVALID')
    folder = rt.HERE / 'deployment_output_audit_batch_v3' / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    output = {'tag': args.tag, 'stage': 'deployment_outputs', 'status': 'INCOMPLETE',
        'started_unix': time.time(), 'records': [], 'failed_routes': [], 'helper_scripts': [],
        'runtime_wall_cap_seconds': 600,
        'runtime_cap_adjustment': rt._artifact(rt.HERE / 'DEPLOYMENT_PROOF_WALL_CAP_ADJUSTMENT_v1.json'),
        'expected_canonical_routes': 268, 'expected_batch_outputs': 536,
        'new_training_updates': 0, 'new_model_or_weight_selection': False,
        'reinference_calls_per_route_batch': 1, 'script_sha256': rt.sha(__file__),
        'EVAL_target_arrays_opened': False, 'protected_TEST_opened': False,
        'evidence_scope': 'Separate post-cost one-call actual outputs; these are not the original seven timed calls',
        'accepted_physical_atol': .002, 'accepted_physical_rtol': 2e-5,
        'matched_to_bulk_normalized_atol': 1e-5, 'matched_to_bulk_normalized_rtol': 2e-6}
    raw_limit = rt.Limit(args.tag, gpu=not args.preflight_only)
    limit = CappedLimit(raw_limit); terminal, error = 'COMPLETE', None
    try:
        import cpu_ridge_deployment as cpu
        import deployment_past_inputs as past
        if cpu.IMPLEMENTATION_ID != CPU_IMPLEMENTATION: raise ValueError('CPU_HEAD_IMPLEMENTATION_NOT_CANONICAL')
        values, routes, aliases, costs, implementations, packs, metadata = preflight(args, past)
        output.update(metadata, aliases=aliases)
        for path in (Path(__file__), Path(bs.__file__), Path(original.__file__), Path(cpu.__file__), Path(past.__file__)):
            destination = folder / path.name
            with destination.open('xb') as handle: handle.write(path.read_bytes())
            output['helper_scripts'].append({'path': path.resolve().relative_to(rt.ROOT).as_posix(), 'sha256': rt.sha(path),
                'snapshot_path': destination.relative_to(rt.ROOT).as_posix(), 'snapshot_sha256': rt.sha(destination)})
        output['scientific_source_hashes'] = output['source_hashes']
        snapshot = rt.source_snapshot(args.tag)
        if snapshot != output['source_hashes']: raise ValueError('DEPLOYMENT_SOURCE_SNAPSHOT_NOT_ORIGINAL16')
        output['scientific_source_snapshot'] = rt._artifact(rt.CACHE / 'source_snapshots' / args.tag / 'MANIFEST.json')
        rt.write(folder / 'PROTOCOL.json', {key: value for key, value in output.items() if key not in ('records', 'failed_routes')})
        unchanged_sources(output)
        if args.preflight_only:
            output.update(status='COMPLETE_PREFLIGHT_ONLY', validated_routes=268, validated_cost_rows=536,
                GPU_model_or_CUDA_loaded=False, actual_saved_batch_outputs=0)
        else:
            bm.en.configure(); forecaster = bm.en.ex.previous.FrozenBolt(); bm.en.replace_normalizer(forecaster)
            output['backbone_before_sha256'] = rt.state_hash(forecaster.model)
            if any(result['backbone_before_sha256'] != output['backbone_before_sha256'] or result['backbone_after_sha256'] != output['backbone_before_sha256'] for result in (values['dev'], values['fresh'], values['cost'])):
                raise ValueError('DEPLOYMENT_SOURCE_BACKBONE_HASH_MISMATCH')
            with cpu.blas_context():
                for route in routes:
                    model = ridge = None; pack = packs[route['dataset']]
                    try:
                        unchanged_sources(output)
                        for artifact in route['model_artifacts']: bs.verify_artifact(artifact)
                        bulk, bulk_metadata = load_bulk_prefix(route, pack)
                        references = {}
                        for batch in (1, 8):
                            try:
                                matched, science_metadata = independent_science_reference(pack, route, batch, forecaster, limit)
                                science_metadata.update(original_bulk_prediction=route['science_prediction'],
                                    model_artifacts=route['model_artifacts'], source_origins_and_mean_std_exact=True,
                                    normalized_vs_original_bulk=bs.normalized_reinference(matched, bulk[:batch]),
                                    BLAS_thread_limit=2)
                                filename = route['dataset'] + '_' + route['identity'] + '_B' + str(batch)
                                science_metadata['artifact'] = rt.save_npz(folder / (filename + '_SCIENCE_REFERENCE.npz'),
                                    matched_science_normalized=matched, original_bulk_science_normalized=bulk[:batch],
                                    origins=pack['origins'][route['split']][:batch], mean=pack['mean'], std=pack['std'])
                                references[batch] = (matched, science_metadata)
                                limit.check()
                            except Exception as exc:
                                if isinstance(exc, rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED', 'BACKBONE_CHANGED')): raise
                                output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'],
                                    'route': route['route'], 'batch': batch, 'stage': 'independent_science_reference',
                                    'error': str(exc), 'traceback': traceback.format_exc(), 'reason': 'PATH_FAILURE_PRESERVED'})
                        if not references: continue
                        if 'case' in route: model = bs.load_selected_model(route['case'])
                        selected_state = model_state(model)
                        preparation = None
                        if route['route'] in RIDGE_ROUTES:
                            ridge = cpu.load_ridge(route.get('readout', route.get('control')), rt)
                            if model is not None: preparation = cpu.prepare_memory(model)
                        prepared_state = model_state(model)
                        if model is not None and selected_state['registered_parameter_sha256'] != prepared_state['registered_parameter_sha256']:
                            raise ValueError('DEPLOYMENT_CACHE_PREPARATION_CHANGED_PARAMETERS')
                        for batch, (matched, science_metadata) in references.items():
                            key = (route['dataset'], route['identity'], route['route'], batch)
                            implementation = implementations[key]['deployment_implementation']
                            deployment = cpu.deployment if implementation == CPU_IMPLEMENTATION else bench.deployment
                            filename = route['dataset'] + '_' + route['identity'] + '_B' + str(batch)
                            try:
                                record, arrays = audit_single_batch(pack, route, batch, matched, bulk[:batch], science_metadata,
                                    forecaster, model, ridge, deployment)
                                metadata_ridge = None if ridge is None else {name: torch.as_tensor(ridge[name]) for name in ('weight', 'bias')}
                                record.update(actual_output=rt.save_npz(folder / (filename + '_ACTUAL.npz'), **arrays),
                                    deployment_implementation=implementation, model_artifacts=route['model_artifacts'],
                                    science_prediction=route['science_prediction'], original_bulk_read_evidence=bulk_metadata,
                                    source_science_sha256_verified=True, source_model_sha256_verified=True,
                                    source_input_origins_exact=True, source_mean_std_exact=True,
                                    active_model_metadata=original._active_metadata(route['route'], model, metadata_ridge, forecaster),
                                    selected_checkpoint_model_state=selected_state, cache_prepared_model_state=prepared_state,
                                    memory_preparation=preparation,
                                    deployment_implementation_metadata=cpu.implementation_metadata(route['route'], model, ridge) if implementation == CPU_IMPLEMENTATION else {'implementation_id': CUDA_IMPLEMENTATION, 'forward_API': 'sealed benchmark.deployment'},
                                    backbone_sha256=output['backbone_before_sha256'],
                                    original_seven_repeat_timing={'source_result': output['source_results']['cost'],
                                        'warmup': costs[key]['warmup'], 'repeats': costs[key]['repeats'],
                                        'median_seconds': costs[key]['median_seconds'], 'timing_producer': implementations[key]})
                                output['records'].append(record); rt.write(folder / (filename + '.json'), record)
                                if not record['critical_gates_pass']:
                                    output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'],
                                        'route': route['route'], 'batch': batch, 'stage': 'actual_deployment_comparison',
                                        'reason': 'CRITICAL_OUTPUT_GATE_FAILED', 'actual_output': record['actual_output']})
                                print(rt.json_line({'actual_deployment_output': filename, 'implementation': implementation,
                                    'critical_gates_pass': record['critical_gates_pass']}), flush=True)
                                limit.check()
                            except Exception as exc:
                                if isinstance(exc, rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED', 'BACKBONE_CHANGED')): raise
                                output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'],
                                    'route': route['route'], 'batch': batch, 'stage': 'actual_deployment',
                                    'error': str(exc), 'traceback': traceback.format_exc(), 'reason': 'PATH_FAILURE_PRESERVED'})
                    except Exception as exc:
                        if isinstance(exc, rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED', 'BACKBONE_CHANGED')): raise
                        output['failed_routes'].append({'dataset': route['dataset'], 'identity': route['identity'],
                            'route': route['route'], 'stage': 'route_assets_or_preparation', 'error': str(exc),
                            'traceback': traceback.format_exc(), 'reason': 'PATH_FAILURE_PRESERVED'})
                    finally:
                        del model, ridge
                        gc.collect()
            output['backbone_after_sha256'] = rt.state_hash(forecaster.model)
            if output['backbone_after_sha256'] != output['backbone_before_sha256']: raise ValueError('BACKBONE_CHANGED_DURING_OUTPUT_AUDIT')
            unchanged_sources(output)
            expected = {(route['dataset'], route['identity'], route['route'], batch) for route in routes for batch in (1, 8)}
            actual = [bs.row_key(record) for record in output['records']]
            coverage = len(actual) == 536 and len(set(actual)) == 536 and set(actual) == expected
            output.update(exact_route_batch_coverage=coverage,
                deployment_implementation_counts=dict(Counter(row['deployment_implementation'] for row in output['records'])),
                all_critical_output_gates_pass=all(row['critical_gates_pass'] for row in output['records']))
            output['status'] = 'COMPLETE' if coverage and output['all_critical_output_gates_pass'] and not output['failed_routes'] else 'FAILED_PRESERVED'
            terminal = output['status']
    except Exception as exc:
        terminal = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc); output.update(status=terminal, error=error, traceback=traceback.format_exc())
    output['actual_saved_batch_outputs'] = len(output['records'])
    output['missing_batch_outputs'] = max(536 - len(output['records']), 0)
    output['finished_unix'] = time.time()
    output['resource_receipt'] = raw_limit.close(terminal, error)
    rt.write(folder / 'RESULT.json', output)
    print(rt.json_line({'deployment_output_audit_terminal': str(folder / 'RESULT.json'), 'status': output['status'],
        'actual_saved_batch_outputs': len(output['records']), 'preflight_only': args.preflight_only}), flush=True)
    if output['status'] not in ('COMPLETE', 'COMPLETE_PREFLIGHT_ONLY'): raise SystemExit(2)


if __name__ == '__main__': main()
