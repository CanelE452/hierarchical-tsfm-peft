"""Balanced whole deployment cost of approved, selected readouts.

Importing this source opens no scientific data or model. The caller must create
the fresh campaign resource guard, verify producer/audit/source bindings and
provide actual physical histories before calling ``measure_dataset``.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research/smallk_failure_followup_20261014'
sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
ROUTES = ('PERSIST', 'MEAN', 'SELF33', 'JOINT33', 'SELF512', 'RAW_BOLT')
ORDER_SEED, REPEATS, WARMUP = 92726, 6, 2
NORM_ATOL, NORM_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def balanced_schedule(seed=ORDER_SEED):
    import numpy as np
    order = np.random.default_rng(seed).permutation(len(ROUTES))
    return [[ROUTES[index] for index in np.roll(order, -block)] for block in range(REPEATS)]


def fill_vectorized(physical, observed, mean):
    import numpy as np
    steps = np.arange(1, physical.shape[1] + 1, dtype=np.int32)[None, :, None]
    last_indices = np.maximum.accumulate(np.where(observed, steps, 0), axis=1)
    initial = np.broadcast_to(mean, (len(physical), 1, len(mean)))
    values = np.concatenate((initial, physical), axis=1)
    return np.take_along_axis(values, last_indices, axis=1).astype(np.float32)


def preprocess(physical, observed, mean, std):
    import numpy as np
    return ((fill_vectorized(physical, observed, mean) - mean) / std).astype(np.float32)


def parity(actual, reference, atol=NORM_ATOL, rtol=NORM_RTOL):
    import numpy as np
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {'pass': False, 'reason': 'shape_or_nonfinite', 'atol': atol, 'rtol': rtol}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {'pass': bool(np.all(np.abs(difference) <= atol + rtol * np.abs(reference))),
            'max_abs': float(np.abs(difference).max()),
            'rmse': float(np.sqrt(np.mean(difference ** 2))), 'atol': atol, 'rtol': rtol}


def validate_physical_inputs(physical, observed, history, mean, std, origins):
    import numpy as np
    if (physical.dtype != np.float32 or observed.dtype != np.bool_
            or history.dtype != np.float32 or physical.shape != observed.shape
            or history.shape != physical.shape or physical.ndim != 3
            or physical.shape[1] != 512 or len(physical) < 8):
        raise ValueError('ACTUAL_PHYSICAL_HISTORY_N512C_FP32_AND_BOOL_MASK_REQUIRED')
    if (mean.dtype != np.float32 or std.dtype != np.float32
            or mean.shape != (physical.shape[-1],) or std.shape != mean.shape
            or origins.dtype != np.int64 or origins.shape != (len(physical),)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()
            or not np.isfinite(physical[observed]).all() or not np.isfinite(history).all()):
        raise ValueError('EXACT_FINITE_FIT_STATS_OR_PHYSICAL_HISTORY_REQUIRED')
    if not np.array_equal(preprocess(physical, observed, mean, std), history):
        raise ValueError('ACTUAL_PHYSICAL_FF_NORMALIZATION_MUST_MATCH_SCIENTIFIC_HISTORY')


class CPUOutputPipeline:
    def __init__(self, predictor, mean, std):
        self.predictor, self.mean, self.std = predictor, mean, std

    def whole(self, physical, observed):
        import numpy as np
        history = preprocess(physical, observed, self.mean, self.std)
        prediction = np.asarray(self.predictor(history))
        if prediction.dtype != np.float32 or prediction.shape != (len(history), 48, history.shape[-1]):
            raise ValueError('NORMALIZED_FP32_N48C_DEPLOYMENT_OUTPUT_REQUIRED')
        if not np.isfinite(prediction).all():
            raise ValueError('NONFINITE_DEPLOYMENT_PREDICTION')
        # CPU inverse and materialized physical return are identical across routes.
        physical_prediction = (prediction * self.std + self.mean).astype(np.float32, copy=False)
        return prediction, physical_prediction.copy()


def save_npz(path, arrays):
    import numpy as np
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': str(Path(path).resolve()), 'sha256': sha(path),
            'bytes': Path(path).stat().st_size,
            'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                            'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                       for key, value in arrays.items()}}


def bolt_runtime_contract(engine, bolt, check):
    """Verify the current executable frozen wrapper without future inputs."""
    import numpy as np
    import torch
    check()
    if bolt.model.training or any(parameter.requires_grad for parameter in bolt.model.parameters()):
        raise ValueError('BOLT_FROZEN_EVAL_REQUIRED')
    normalizer = bolt.model.instance_norm
    if type(normalizer).__name__ != 'InstanceNorm':
        raise ValueError('FRESH_ORIGINAL_INSTANCE_NORM_REQUIRED_BEFORE_PARITY_CHECK')
    quantiles = list(bolt.model.chronos_config.quantiles)
    config = bolt.model.chronos_config
    before = {'class': type(normalizer).__name__, 'eps': float(normalizer.eps),
              'use_arcsinh': bool(normalizer.use_arcsinh)}
    # Constant, nonconstant and missing-containing rows exercise the internal
    # forward normalization that a weights-only hash cannot describe.
    rows = torch.tensor([[1.] * 16, list(np.arange(16, dtype=np.float32) / 16),
                         [float('nan'), 2., 2., 3.] * 4], device='cuda', dtype=torch.float32)
    with torch.no_grad():
        original_value, original_loc_scale = normalizer(rows)
        engine.replace_normalizer(bolt)
        repaired_value, repaired_loc_scale = bolt.model.instance_norm(rows)
    checks = {}
    for name, original, repaired in (
            ('standardized', original_value, repaired_value),
            ('location', original_loc_scale[0], repaired_loc_scale[0]),
            ('scale', original_loc_scale[1], repaired_loc_scale[1])):
        a, b = original.detach().cpu().numpy(), repaired.detach().cpu().numpy()
        checks[name] = {'exact_equal_including_nan': bool(np.array_equal(a, b, equal_nan=True)),
                        'shape': list(a.shape)}
    if not all(value['exact_equal_including_nan'] for value in checks.values()):
        raise ValueError('ORIGINAL_REPAIRED_NORMALIZER_FORWARD_PARITY_FAILED')
    if not quantiles or bolt.median != quantiles.index(.5) or engine.H != 48:
        raise ValueError('EXACT_MEDIAN_Q05_H48_WRAPPER_REQUIRED')
    check()
    after = bolt.model.instance_norm
    return {'architecture': type(bolt.model).__name__, 'evaluation': not bolt.model.training,
            'all_backbone_parameters_frozen': all(not p.requires_grad for p in bolt.model.parameters()),
            'normalizer_before': before,
            'normalizer_after': {'class': type(after).__name__, 'eps': float(after.eps),
                                 'use_arcsinh': bool(after.use_arcsinh)},
            'normalizer_fixture_forward_parity': checks,
            'normalizer_fixture_backward_not_executed_in_cost': True,
            'context_length': int(config.context_length),
            'prediction_length': int(config.prediction_length),
            'input_patch_size': int(config.input_patch_size),
            'input_patch_stride': int(config.input_patch_stride),
            'quantiles': quantiles, 'median_index': int(bolt.median), 'returned_horizon': 48,
            'wrapper_shape': 'FP32 N,L,C -> N*C,L -> model(context) quantile_preds[:,q05,:48] -> N,48,C',
            'pipeline_predict_not_used': True,
            'past_only_fixture_does_not_verify_independent_generalization': True}


def measure_dataset(dataset, physical, observed, history, mean, std, origins,
                    predictors, metadata, scientific_reference, output, check,
                    synchronize, reset_memory, memory_snapshot):
    """Measure 12 cells; guard, audit and model setup belong to the caller.

    ``predictors`` returns CPU FP32 predictions for all six declared routes.
    The Raw Bolt callback therefore includes H2D, actual frozen-F and D2H.
    No target or future observed mask enters any prediction callback.
    """
    import numpy as np
    if set(predictors) != set(ROUTES) or set(scientific_reference) != set(ROUTES):
        raise ValueError('EXACT_SIX_DECLARED_COST_ROUTES_REQUIRED')
    validate_physical_inputs(physical, observed, history, mean, std, origins)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipelines = {route: CPUOutputPipeline(predictors[route], mean, std) for route in ROUTES}
    result = {'dataset': dataset, 'status': 'RUNNING', 'rows': [],
              'order_seed': ORDER_SEED, 'warmup_each_route': WARMUP, 'repeats': REPEATS,
              'new_training_updates': 0, 'coefficient_refit_in_cost': False,
              'protected_TEST_accessed': False, 'independent_holdout': False}
    try:
        for batch in (1, 8):
            check()
            folder = output / ('B' + str(batch))
            folder.mkdir(exist_ok=False)
            values, masks = physical[:batch], observed[:batch]
            references = {}
            for route in ROUTES:
                check()
                normalized, physical_value = pipelines[route].whole(values, masks)
                synchronize()
                source = np.asarray(scientific_reference[route])[:batch]
                source_parity = parity(normalized, source)
                save_npz(folder / (route + '_MATCHED_REFERENCE.npz'), {
                    'normalized': normalized, 'physical': physical_value,
                    'scientific_normalized': source, 'origins': origins[:batch]})
                if not source_parity['pass']:
                    raise ValueError('MATCHED_BATCH_SCIENTIFIC_REFERENCE_PARITY_FAILED:' + route)
                references[route] = {'normalized': normalized.copy(), 'physical': physical_value.copy(),
                                     'scientific_parity': source_parity}
            reference_record = save_npz(folder / 'REFERENCES.npz', {
                **{route + '_normalized': value['normalized'] for route, value in references.items()},
                **{route + '_physical': value['physical'] for route, value in references.items()},
                'origins': origins[:batch]})
            schedule = balanced_schedule()
            for order in schedule[:WARMUP]:
                for route in order:
                    check()
                    pipelines[route].whole(values, masks)
                    synchronize()
            samples = {route: [] for route in ROUTES}
            for block, order in enumerate(schedule):
                for position, route in enumerate(order):
                    check()
                    synchronize()
                    reset_memory()
                    started = time.perf_counter()
                    normalized, physical_value = pipelines[route].whole(values, masks)
                    synchronize()
                    seconds = time.perf_counter() - started
                    memory = memory_snapshot()
                    checks = {'normalized': parity(normalized, references[route]['normalized']),
                              'physical': parity(physical_value, references[route]['physical'], PHYSICAL_ATOL, PHYSICAL_RTOL),
                              'scientific_normalized': parity(normalized, np.asarray(scientific_reference[route])[:batch])}
                    sample_record = save_npz(folder / (route + '_r' + str(block) + '.npz'), {
                        'normalized': normalized, 'physical': physical_value,
                        'origins': origins[:batch]})
                    samples[route].append({'block': block, 'position': position, 'seconds': seconds,
                                           'normalized': normalized.copy(), 'physical': physical_value.copy(),
                                           'parity': checks, 'memory': memory,
                                           'sample_artifact': sample_record})
                    # Every actual timing remains on disk if a later route fails.
                    with (folder / (route + '_r' + str(block) + '.json')).open('x', encoding='utf-8') as handle:
                        json.dump({'route': route, 'batch': batch, 'block': block, 'position': position,
                                   'seconds': seconds, 'parity': checks, 'memory': memory,
                                   'sample_artifact': sample_record}, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    check()
                    if not all(value['pass'] for value in checks.values()):
                        raise ValueError('TIMED_OUTPUT_PARITY_FAILED:' + route)
            raw_seconds = {value['block']: value['seconds'] for value in samples['RAW_BOLT']}
            for route in ROUTES:
                records = samples[route]
                seconds = [value['seconds'] for value in records]
                ratios = [value['seconds'] / raw_seconds[value['block']] for value in records]
                payload = save_npz(folder / (route + '_TIMED.npz'), {
                    'normalized': np.stack([value['normalized'] for value in records]),
                    'physical': np.stack([value['physical'] for value in records]),
                    'block': np.array([value['block'] for value in records], np.int64),
                    'position': np.array([value['position'] for value in records], np.int64),
                    'origins': origins[:batch]})
                result['rows'].append({
                    'dataset': dataset, 'route': route, 'status': 'COMPLETE', 'batch': batch,
                    'warmup_each_route': WARMUP, 'repeats': len(records), 'seconds': seconds,
                    'median_seconds': float(np.median(seconds)),
                    'median_seconds_per_window': float(np.median(seconds)) / batch,
                    'paired_RAW_seconds': [raw_seconds[value['block']] for value in records],
                    'paired_RAW_ratios': ratios, 'paired_RAW_median_ratio': float(np.median(ratios)),
                    'positions': [value['position'] for value in records],
                    'parity': [value['parity'] for value in records],
                    'memory': [value['memory'] for value in records],
                    'selected_metadata': metadata[route], 'timed_outputs': payload,
                    'sample_artifacts': [value['sample_artifact'] for value in records],
                    'reference': reference_record,
                    'whole_includes': 'actual causal CPU fill, FIT normalization, CPU selected readout or H2D/frozen Bolt/D2H, CPU inverse, materialized physical return, final synchronization',
                    'whole_excludes': 'data/model/coefficient load, disk save, references, hashes, guards outside timer',
                    'memory_scope': 'one shared resident Bolt plus all readout coefficient/state objects; per-route reset records total live GPU peak, not isolated incremental allocation',
                    'balanced_scope': 'each route occupies every position once; no independent hardware repetition or thermal/carryover proof',
                    'stage_decomposition': False, 'optimized_native_implementation_claim': False})
            print(json.dumps({'event': 'COST_BATCH_COMPLETE', 'dataset': dataset, 'batch': batch,
                              'completed_rows': len(result['rows'])}), flush=True)
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    return result


def restored_predictors(producer, probes, math):
    """Restore only already selected immutable coefficients; no fit occurs."""
    shared_mean, checked_mean = probes.load_checked_npz(producer['shared_mean_artifact'])
    records = [checked_mean]
    predictors, metadata, objects = {}, {}, {}
    for kind in ('PERSIST', 'MEAN'):
        predictors[kind] = lambda history, kind=kind: math.control_forecast(kind, history[:, -1], shared_mean)
        metadata[kind] = {'selected_kind': kind, 'learned_from': 'FIT' if kind == 'MEAN' else 'none',
                          'shared_control_across_all_readout_pools': True}
    for route in ROUTES[2:5]:
        family = producer['families'][route]
        kind = family['selected_kind']
        if kind not in ('PERSIST', 'MEAN', 'RIDGE'):
            raise ValueError('PREDECLARED_SELECTED_READOUT_KIND_REQUIRED')
        metadata[route] = {'selected_kind': kind, 'selected_index': family['selected_index'],
                           'selected_alpha': family['selected_alpha'],
                           'route_is_selected_nine_pool_not_forced_ridge': True}
        if kind != 'RIDGE':
            predictors[route] = lambda history, kind=kind: math.control_forecast(kind, history[:, -1], shared_mean)
            continue
        state = producer['feature_states'][route]
        state_arrays, checked_state = probes.load_checked_npz(state['artifact'])
        coefficient_arrays, checked_coefficients = probes.load_checked_npz(family['selected_coefficient_artifact'])
        if route == 'JOINT33':
            model = math.joint.restore_model(state, state_arrays, family['selected_model_metadata'], coefficient_arrays)
        else:
            model = math.restore_self_model(state, state_arrays, family['selected_model_metadata'], coefficient_arrays)
        if model.sha256 != family['selected_model_sha256']:
            raise ValueError('EXACT_SELECTED_COEFFICIENT_MODEL_SHA_REQUIRED')
        objects[route] = model
        predictors[route] = lambda history, model=model: model.forecast(history, history[:, -1])
        metadata[route]['selected_model_sha256'] = model.sha256
        records.extend((checked_state, checked_coefficients))
    return predictors, metadata, objects, records


def execute(args, limit):
    import numpy as np
    import torch
    import psutil
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import readout_math_v1 as math
    from threadpoolctl import threadpool_limits
    inputs.install_protected_test_guard()
    output = HERE / 'cost_runs' / args.tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'rows': [], 'planned_rows': 24, 'producer_records': [],
              'actual_GPU_execution': False, 'new_training_updates': 0,
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'coefficient_refit_in_cost': False, 'source_records': [], 'source_snapshots': [],
              'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record}
    bolt = None
    failure = None
    try:
        if not args.no_contention_attested:
            raise ValueError('ROOT_NO_LIVE_TRAINING_OR_OTHER_NEW_CAMPAIGN_JOB_ATTESTATION_REQUIRED')
        full, fast = limit.checkfull, limit.shared.checkfast
        full()
        producer_records = [probes.verified_file({'path': str(probes.resolve(path)), 'sha256': probes.sha(probes.resolve(path))})
                            for path in args.readout_results]
        producers = [probes.read(record['path']) for record in producer_records]
        if len(producers) != 2 or {p.get('dataset') for p in producers} != {'robin', 'beijing_aotizhongxin'}:
            raise ValueError('EXACT_TWO_UNIQUE_READOUT_DATASETS_REQUIRED')
        audit_record = probes.verified_file({'path': str(probes.resolve(args.audit)), 'sha256': probes.sha(probes.resolve(args.audit))})
        audit = probes.read(audit_record['path'])
        if audit.get('status') != 'PASS' or audit.get('totals', {}).get('pool_candidates_checked') != 54:
            raise ValueError('INDEPENDENT_54_POOL_SAVED_ARRAY_AUDIT_PASS_REQUIRED')
        for producer, record in zip(producers, producer_records):
            if (producer.get('status') != 'COMPLETE' or producer.get('GPU_execution') is not False
                    or producer.get('protected_TEST_accessed') is not False or producer.get('independent_holdout') is not False
                    or producer.get('permission_gate', {}).get('sha256') != limit.authorization_record['sha256']
                    or producer.get('run_manifest', {}).get('sha256') != limit.run_manifest_record['sha256']
                    or producer.get('pool_comparisons') != 27):
                raise ValueError('EXACT_CURRENT_APPROVED_COMPLETE_READOUT_PRODUCER_REQUIRED')
            bindings = [r for r in audit.get('producer_records', []) if r.get('dataset') == producer['dataset']]
            if len(bindings) != 1 or bindings[0]['sha256'] != record['sha256']:
                raise ValueError('SCIENTIFIC_AUDIT_EXACT_PRODUCER_SHA_BINDING_REQUIRED')
        result.update(producer_records=producer_records, independent_scientific_audit=audit_record)
        result['source_records'].extend(producer_records + [audit_record])
        snapshots = output / 'sources'
        snapshots.mkdir(exist_ok=False)
        for source in (Path(__file__), HERE/'readout_math_v1.py', HERE/'adaptive_runtime_v1.py',
                       OLD/'matched_readout_math_v1.py', OLD/'stage_separation_math_v1.py',
                       OLD/'prepare_inputs.py', OLD/'run_condition_probes.py'):
            full()
            destination = snapshots / source.name
            with source.open('rb') as original, destination.open('xb') as target:
                shutil.copyfileobj(original, target)
            if probes.sha(source) != probes.sha(destination):
                raise ValueError('COST_EXECUTION_SOURCE_SNAPSHOT_CHANGED')
            result['source_snapshots'].append({'path': str(source), 'sha256': probes.sha(source), 'snapshot_path': str(destination)})
        setup_started = time.perf_counter()
        engine = probes.import_old_engine()
        weight_path = engine.ex.previous.BOLT_DIRECTORY / 'model.safetensors'
        weight_before = probes.verified_file({'path': str(weight_path), 'sha256': probes.sha(weight_path)}, role='actual_local_frozen_Bolt_binary')
        config_path = engine.ex.previous.BOLT_DIRECTORY / 'config.json'
        config_before = probes.verified_file({'path': str(config_path), 'sha256': probes.sha(config_path)}, role='actual_local_Bolt_config')
        result['source_records'].extend((weight_before, config_before))
        full()
        bolt = engine.ex.previous.FrozenBolt()
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}:
            raise ValueError('EXACT_FROZEN_BOLT_DUAL_STATE_HASH_REQUIRED')
        result['runtime_behavior_contract'] = bolt_runtime_contract(engine, bolt, fast)
        result.update(actual_GPU_execution=True, model_binary_before=weight_before,
                      model_config_before=config_before,
                      model_load_and_contract_wall_seconds=time.perf_counter()-setup_started)
        all_objects = {}
        process = psutil.Process()
        def memory_snapshot():
            return {'peak_allocated_vram_bytes': int(torch.cuda.max_memory_allocated()),
                    'peak_reserved_vram_bytes': int(torch.cuda.max_memory_reserved()),
                    'resident_rss_bytes': int(process.memory_info().rss)}
        def raw_predictor(history):
            with torch.no_grad():
                return bolt(engine.tensor(history)).detach().cpu().numpy().copy()
        with torch.no_grad(), threadpool_limits(limits=2):
            for producer in producers:
                full()
                dataset = producer['dataset']
                split = 'val' if dataset == 'robin' else 'eval'
                bridge_record = probes.verified_file(producer['bridge_provenance']['input_bridge'])
                bridge = probes.read(bridge_record['path'])
                phase = bridge['phase_barrier']
                for name in ('audit', 'report', 'manifest'):
                    result['source_records'].append(probes.verified_file(phase[name]))
                inputs.verify_phase_barrier(*(probes.resolve(phase[name]['path']) for name in ('audit', 'report', 'manifest')))
                physical_records = [r for r in bridge['artifacts'] if r.get('dataset') == dataset and r.get('split') == split]
                if len(physical_records) != 1:
                    raise ValueError('EXACT_ONE_REAL_PHYSICAL_HISTORY_BRIDGE_REQUIRED')
                raw, physical_record = probes.load_checked_npz(physical_records[0])
                scientific_raw, scientific_raw_record = probes.load_checked_npz(producer['inputs'][split])
                for key, value in scientific_raw.items():
                    if key not in raw or not np.array_equal(value, raw[key], equal_nan=True):
                        raise ValueError('COST_ORIGINAL_PHYSICAL_AND_SCIENTIFIC_RAW_CONDITION_MISMATCH:' + key)
                predictors, metadata, objects, coefficient_records = restored_predictors(producer, probes, math)
                all_objects[dataset] = objects
                predictors['RAW_BOLT'] = raw_predictor
                metadata['RAW_BOLT'] = {'selected_kind': 'RAW_BOLT', 'median_quantile': .5, 'horizon': 48,
                                         'same_origins_and_FIT_statistics': True, 'backbone_training': False}
                mean_arrays, mean_record = probes.load_checked_npz(producer['shared_mean_artifact'])
                references = {'RAW_BOLT': raw['raw_forecast'],
                              'PERSIST': math.control_forecast('PERSIST', raw['raw_last'], mean_arrays),
                              'MEAN': math.control_forecast('MEAN', raw['raw_last'], mean_arrays)}
                selected_records = []
                for route in ROUTES[2:5]:
                    arrays, selected_record = probes.load_checked_npz(producer['families'][route]['selected_artifacts'][split])
                    for key in ('target', 'observed', 'origins', 'raw_last', 'mean', 'std'):
                        if not np.array_equal(arrays[key], raw[key], equal_nan=True):
                            raise ValueError('SELECTED_COST_CONDITION_MISMATCH:' + route + ':' + key)
                    references[route] = arrays['prediction']
                    selected_records.append(selected_record)
                result['source_records'].extend([bridge_record, physical_record, scientific_raw_record, mean_record] + coefficient_records + selected_records)
                result.setdefault('dataset_input_sources', {})[dataset] = {'physical_bridge': physical_record,
                    'scientific_input': scientific_raw_record, 'development_split': split,
                    'inverse_normalized_proxy_used': False}
                measured = measure_dataset(dataset, raw['history_physical'], raw['history_observed'],
                    raw['history'], raw['mean'], raw['std'], raw['origins'], predictors, metadata, references,
                    output / dataset, fast, torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats, memory_snapshot)
                result['rows'].extend(measured['rows'])
                full()
        result['F_after'] = probes.model_hashes(bolt)
        result['model_binary_after'] = probes.verified_file(weight_before)
        result['model_config_after'] = probes.verified_file(config_before)
        for record in result['source_records'] + result['source_snapshots']:
            probes.verified_file(record)
            if 'snapshot_path' in record and probes.sha(record['snapshot_path']) != record['sha256']:
                raise ValueError('COST_EXECUTION_SOURCE_SNAPSHOT_CHANGED_AFTER')
        if result['F_before'] != result['F_after']:
            raise ValueError('FROZEN_BOLT_STATE_CHANGED')
        result.update(status='COMPLETE', source_before_after_SHA_pass=True, model_binary_before_after_SHA_pass=True,
                      selected_model_objects_loaded=sum(len(values) for values in all_objects.values()),
                      hardware_trial_scope='one actual local process with balanced route positions; not independent repeated hardware runs',
                      primary_cost_scope='all six routes return normalized and physical FP32 CPU arrays; Raw GPU inverse replaced by the same CPU inverse to keep the return contract uniform')
    except Exception as error:
        import adaptive_runtime_v1 as runtime
        failure = {'type': type(error).__name__, 'message': str(error)}
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED', failure=failure)
    finally:
        if bolt is not None and 'F_after' not in result:
            try:
                result['F_after'] = probes.model_hashes(bolt)
            except Exception as error:
                result['F_after_unavailable'] = {'type': type(error).__name__, 'message': str(error)}
        terminal = limit.close(result['status'], failure)
        result['resource_terminal'] = terminal
        result['status'] = terminal['terminal_status']
        with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'cost_result': str(output / 'RESULT.json'), 'status': result['status'],
                          'completed_rows': len(result['rows'])}), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--readout-results', nargs=2, required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--no-contention-attested', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'scientific_execution': False, 'planned_rows': 24, 'routes': ROUTES,
                          'whole_return_contract': 'normalized and physical FP32 CPU'}))
        return 0
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import adaptive_runtime_v1 as runtime
    limit = runtime.ContinuationLimit(args.tag, gpu=True)
    try:
        return execute(args, limit)
    except BaseException as error:
        if not limit.closed:
            limit.close('FAILED_PRESERVED', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
