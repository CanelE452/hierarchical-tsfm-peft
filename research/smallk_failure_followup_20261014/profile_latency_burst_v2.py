"""Paired old-condition RAW/MAIN latency decomposition; no assets on import.

Actual execution is opt-in, one dataset/seed per exclusive output directory.
Scientific inputs and models are loaded only after the completed-Bi bridge gate.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'latency_profiles'
WARMUP, REPEATS = 2, 7
NORM_ATOL, NORM_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5
STAGES = ('fill', 'normalization', 'H2D', 'E', 'F', 'D', 'anchor_inverse', 'D2H')


def fill_loop(physical, observed, mean):
    """The old experiment.preprocess loop before its normalization expression."""
    filled = np.empty_like(physical, dtype=np.float32)
    last = np.broadcast_to(mean, (len(physical), len(mean))).copy()
    for t in range(physical.shape[1]):
        last = np.where(observed[:, t], physical[:, t], last)
        filled[:, t] = last
    return filled


def fill_vectorized(physical, observed, mean):
    steps = np.arange(1, physical.shape[1] + 1, dtype=np.int32)[None, :, None]
    last_indices = np.maximum.accumulate(np.where(observed, steps, 0), axis=1)
    initial = np.broadcast_to(mean, (len(physical), 1, len(mean)))
    values = np.concatenate((initial, physical), axis=1)
    return np.take_along_axis(values, last_indices, axis=1).astype(np.float32)


def normalize(filled, mean, std):
    return ((filled - mean) / std).astype(np.float32)


def vectorized_preprocess(physical, observed, mean, std):
    return normalize(fill_vectorized(physical, observed, mean), mean, std)


def validate_raw_inputs(physical, observed, history, mean, std, origins):
    if (physical.dtype != np.float32 or observed.dtype != np.bool_
            or history.dtype != np.float32 or physical.shape != observed.shape
            or history.shape != physical.shape or physical.ndim != 3
            or physical.shape[1] != 512 or len(physical) < 8):
        raise ValueError('ORIGINAL_RAW_PAST_SHAPE_DTYPE_REQUIRED')
    if (mean.dtype != np.float32 or std.dtype != np.float32
            or mean.shape != (physical.shape[-1],) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0) or origins.dtype != np.int64
            or origins.shape != (len(physical),)):
        raise ValueError('STORED_STATS_OR_ORIGINS_INVALID')
    if not np.isfinite(physical[observed]).all() or not np.isfinite(history).all():
        raise ValueError('OBSERVED_PAST_OR_HISTORY_NONFINITE')
    baseline = normalize(fill_loop(physical, observed, mean), mean, std)
    if not np.array_equal(baseline, history):
        raise ValueError('ORIGINAL_RAW_PAST_DOES_NOT_REPRODUCE_BRIDGE_HISTORY')
    if not np.array_equal(vectorized_preprocess(physical, observed, mean, std), baseline):
        raise ValueError('VECTORIZED_PREPROCESS_EXACT_PARITY_FAILED')


def parity(actual, reference, atol, rtol):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {'pass': False, 'reason': 'shape_or_nonfinite', 'atol': atol, 'rtol': rtol}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {'pass': bool(np.all(np.abs(difference) <= atol + rtol * np.abs(reference))),
            'max_abs': float(np.abs(difference).max()),
            'rmse': float(np.sqrt(np.mean(difference ** 2))), 'atol': atol, 'rtol': rtol}


class TorchBackend:
    def __init__(self, engine):
        import torch
        self.torch, self.engine = torch, engine

    def tensor(self, value):
        return self.engine.tensor(value)

    def to_numpy(self, value):
        return value.detach().cpu().numpy().copy()

    def synchronize(self):
        self.torch.cuda.synchronize()

    def reset_memory(self):
        self.torch.cuda.reset_peak_memory_stats()

    def memory_snapshot(self):
        import psutil
        return {'peak_allocated_vram_bytes': int(self.torch.cuda.max_memory_allocated()),
                'resident_rss_bytes': int(psutil.Process().memory_info().rss),
                'scope': 'Bolt/codec co-resident; not incremental method-exclusive allocation.'}


class DeploymentPipeline:
    def __init__(self, backend, codec, forecaster, mean, std, old_preprocess):
        self.backend, self.codec, self.forecaster = backend, codec, forecaster
        self.mean, self.std, self.old_preprocess = mean, std, old_preprocess

    def whole(self, physical, observed, preprocessing):
        preprocess = self.old_preprocess if preprocessing == 'old_loop' else vectorized_preprocess
        history = preprocess(physical, observed, self.mean, self.std)
        x = self.backend.tensor(history)
        future = self.forecaster(x) if self.codec is None else self.codec.forward_path(x, self.forecaster)
        physical_future = future * self.backend.tensor(self.std) + self.backend.tensor(self.mean)
        return future, self.backend.to_numpy(physical_future)

    def decomposed(self, physical, observed, preprocessing, limit):
        timings = {}

        def stage(name, callback):
            self.backend.synchronize()
            start = time.perf_counter()
            result = callback()
            self.backend.synchronize()
            timings[name] = time.perf_counter() - start
            limit.check()
            return result

        fill = fill_loop if preprocessing == 'old_loop' else fill_vectorized
        filled = stage('fill', lambda: fill(physical, observed, self.mean))
        history = stage('normalization', lambda: normalize(filled, self.mean, self.std))
        x = stage('H2D', lambda: self.backend.tensor(history))
        z = stage('E', lambda: x if self.codec is None else self.codec.encode(x))
        future_z = stage('F', lambda: self.forecaster(z))
        decoded = stage('D', lambda: (future_z, None) if self.codec is None
                        else (self.codec.decode(future_z), self.codec.decode(z[:, -1:])))

        def anchor_inverse():
            output = decoded[0] if self.codec is None else x[:, -1:] + decoded[0] - decoded[1]
            return output, output * self.backend.tensor(self.std) + self.backend.tensor(self.mean)

        normalized, physical_tensor = stage('anchor_inverse', anchor_inverse)
        physical_output = stage('D2H', lambda: self.backend.to_numpy(physical_tensor))
        return normalized, physical_output, timings


def measure_route(pipeline, physical, observed, preprocessing, reference, bulk_reference,
                  output, limit, save_arrays):
    for _ in range(WARMUP):
        pipeline.whole(physical, observed, preprocessing)
        limit.check()
    if hasattr(pipeline.backend, 'reset_memory'):
        pipeline.backend.reset_memory()
    seconds, normalized, physical_values = [], [], []
    for _ in range(REPEATS):
        limit.check()
        pipeline.backend.synchronize()
        start = time.perf_counter()
        future, actual_physical = pipeline.whole(physical, observed, preprocessing)
        pipeline.backend.synchronize()
        seconds.append(time.perf_counter() - start)
        normalized.append(pipeline.backend.to_numpy(future))
        physical_values.append(actual_physical.copy())
    measured = save_arrays(output / 'whole_outputs.npz', {
        'normalized': np.stack(normalized), 'physical': np.stack(physical_values)})
    checks = [{'repeat': index,
               'matched_batch_normalized': parity(value, reference['normalized'], NORM_ATOL, NORM_RTOL),
               'matched_batch_physical': parity(physical_values[index], reference['physical'], PHYSICAL_ATOL, PHYSICAL_RTOL),
               'original_bulk_normalized': parity(value, bulk_reference['normalized'], NORM_ATOL, NORM_RTOL),
               'original_bulk_physical': parity(physical_values[index], bulk_reference['physical'], PHYSICAL_ATOL, PHYSICAL_RTOL)}
              for index, value in enumerate(normalized)]
    whole_gate = all(row['matched_batch_normalized']['pass'] and row['matched_batch_physical']['pass'] for row in checks)
    result = {'status': 'WHOLE_COMPLETE' if whole_gate else 'FAILED_PRESERVED',
              'preprocessing': preprocessing, 'batch': len(physical), 'warmup': WARMUP, 'repeats': REPEATS,
              'whole_seconds': seconds, 'whole_median_seconds': float(np.median(seconds)),
              'whole_median_seconds_per_window': float(np.median(seconds)) / len(physical),
              'whole_outputs': measured, 'whole_parity': checks,
              'normalized_D2H_copy_after_whole_timer': True,
              'whole_includes': 'old causal CPU fill/norm,H2D,E/F/D,anchor,inverse,D2H,final synchronization',
              'whole_excludes': 'disk/model/checkpoint load,reference inference,normalized output copy,output save,hashing'}
    if hasattr(pipeline.backend, 'memory_snapshot'):
        result['whole_memory'] = pipeline.backend.memory_snapshot()
    if not whole_gate:
        return result
    for _ in range(WARMUP):
        pipeline.decomposed(physical, observed, preprocessing, limit)
    if hasattr(pipeline.backend, 'reset_memory'):
        pipeline.backend.reset_memory()
    stages, stage_normalized, stage_physical, stage_checks = [], [], [], []
    for index in range(REPEATS):
        future, actual_physical, row = pipeline.decomposed(physical, observed, preprocessing, limit)
        value = pipeline.backend.to_numpy(future)
        stages.append(row)
        stage_normalized.append(value)
        stage_physical.append(actual_physical.copy())
        stage_checks.append({'repeat': index,
            'matched_batch_normalized': parity(value, reference['normalized'], NORM_ATOL, NORM_RTOL),
            'matched_batch_physical': parity(actual_physical, reference['physical'], PHYSICAL_ATOL, PHYSICAL_RTOL),
            'whole_same_repeat_normalized': parity(value, normalized[index], NORM_ATOL, NORM_RTOL),
            'whole_same_repeat_physical': parity(actual_physical, physical_values[index], PHYSICAL_ATOL, PHYSICAL_RTOL)})
    stage_gate = all(all(record['pass'] for key, record in row.items() if key != 'repeat') for row in stage_checks)
    result.update(status='COMPLETE' if stage_gate else 'FAILED_PRESERVED',
        decomposed_seconds=stages, decomposed_parity=stage_checks,
        stage_median_seconds={name: float(np.median([row[name] for row in stages])) for name in STAGES},
        decomposed_outputs=save_arrays(output / 'decomposed_outputs.npz', {
            'normalized': np.stack(stage_normalized), 'physical': np.stack(stage_physical)}),
        decomposition_note='Each stage has explicit GPU synchronization. Their sum is not uninstrumented whole-path latency.',
        stage_decoder_scope='Both D(z_F) and D(z_last); anchor subtraction and physical inverse are in anchor_inverse.',
        RAW_E_D_identity_note='RAW E and D profile host identity overhead; no encoder or decoder GPU kernels.')
    if hasattr(pipeline.backend, 'memory_snapshot'):
        result['decomposed_memory'] = pipeline.backend.memory_snapshot()
    limit.check()
    return result


def paired_changes(rows):
    result = []
    for method in ('RAW_BOLT', 'MAIN'):
        for batch in (1, 8):
            matched = [row for row in rows if row['method'] == method and row['batch'] == batch and row['status'] == 'COMPLETE']
            if {row['preprocessing'] for row in matched} != {'old_loop', 'vectorized'}:
                continue
            baseline = next(row for row in matched if row['preprocessing'] == 'old_loop')
            candidate = next(row for row in matched if row['preprocessing'] == 'vectorized')
            before, after = baseline['whole_median_seconds'], candidate['whole_median_seconds']
            result.append({'method': method, 'batch': batch, 'baseline_seconds': before,
                           'candidate_seconds': after, 'relative_change_percent': 100 * (after / before - 1),
                           'interpretation': 'Paired single-process latency sample; not repeated independent hardware trials.'})
    return result


def import_sibling(name):
    specification = importlib.util.spec_from_file_location('latency_followup_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def describe():
    return {'status': 'PREPARATION_ONLY', 'scientific_arrays_opened': False, 'model_or_GPU_executed': False,
            'planned_conditions': 'Robin/Beijing,seed92601/92602 separately; RAW_BOLT/MAIN;B1/B8;old_loop/vectorized',
            'required_bridge_fields': ['history_physical', 'history_observed', 'history', 'origins', 'mean', 'std'],
            'resource_scope': 'followup_runtime.FollowupLimit shared remaining Bi caps; no old campaign Limit',
            'local_wall_guard_seconds': 600, 'warmup': WARMUP, 'repeats': REPEATS,
            'parity': {'normalized': [NORM_ATOL, NORM_RTOL], 'matched_batch_physical': [PHYSICAL_ATOL, PHYSICAL_RTOL]},
            'original_bulk_gate': 'Reported separately; never relabelled matching-batch output improvement.'}


def execute(args):
    if not args.execute_after_bi:
        raise ValueError('EXPLICIT_AFTER_BI_EXECUTION_REQUIRED')
    if not args.tag or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.tag):
        raise ValueError('FRESH_SAFE_TAG_REQUIRED')
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    runner = import_sibling('run_condition_probes')
    bridge_module = import_sibling('prepare_inputs')
    limit = None
    start = time.perf_counter()
    result = {'status': 'RUNNING', 'tag': args.tag, 'dataset': args.dataset, 'seed': args.seed,
              'rows': [], 'protected_TEST_accessed': False, 'new_training_updates': 0,
              'measurement_version': 'burst_v2_same_numerics_guard_io_outside_repeated_rows',
              'preserved_original_measurement': 'd3_robin_s92601_v1',
              'local_wall_guard_seconds': 600, 'preparation': describe()}

    class Watchdog:
        fast_only = False

        def check(self):
            if self.fast_only:
                limit.shared.check()
            else:
                limit.check()
            if time.perf_counter() - start >= 600:
                raise RuntimeError('LATENCY_PROFILE_LOCAL_WALL_600S')

    watchdog = Watchdog()
    bolt, codec, bridge, checkpoint = None, None, None, None
    try:
        runtime = import_sibling('followup_runtime')
        limit = runtime.FollowupLimit(args.tag, gpu=True)
        snapshot_folder = output / 'sources'
        snapshot_folder.mkdir(exist_ok=False)
        helper_records = []
        for name in ('profile_latency.py', Path(__file__).name, 'run_condition_probes.py', 'prepare_inputs.py', 'followup_runtime.py'):
            source = HERE / name
            checked = runner.verified_file({'path': str(source), 'sha256': runner.sha(source)})
            destination = snapshot_folder / name
            shutil.copyfile(source, destination)
            checked['snapshot'] = {'path': str(destination), 'sha256': runner.sha(destination)}
            helper_records.append(checked)
        result['helper_sources_before'] = helper_records
        watchdog.check()
        bridge_module.install_protected_test_guard()
        bridge, splits, mean, std, provenance = runner.load_bridge(args.bridge_result, args.dataset, bridge_module.verify_phase_barrier)
        sources_before = runner.verify_source_records(bridge)
        split = 'val' if args.dataset == 'robin' else 'eval'
        record = next(row for row in bridge['artifacts'] if row['dataset'] == args.dataset and row['split'] == split)
        inputs, checked_input = runner.load_checked_npz(record)
        if 'history_physical' not in inputs:
            raise ValueError('BRIDGE_ORIGINAL_RAW_PHYSICAL_PAST_REQUIRED_NOT_DENORMALIZED_HISTORY')
        validate_raw_inputs(inputs['history_physical'], inputs['history_observed'], inputs['history'], mean, std, inputs['origins'])
        engine = runner.import_old_engine()
        import torch
        codec, _, checkpoint = runner.load_main_codec(engine, bridge, args.dataset, args.seed)
        bolt = engine.ex.previous.FrozenBolt()
        engine.replace_normalizer(bolt)
        f_before = runner.model_hashes(bolt)
        if f_before != {'old_algorithm': runner.OLD_F_HASH, 'bi_algorithm': runner.BI_F_HASH}:
            raise ValueError('FROZEN_BOLT_EXPECTED_HASH_MISMATCH')
        codec_before = runner.old_state_hash(codec)
        sys.modules['engine'] = engine
        spec = importlib.util.spec_from_file_location('latency_original_controls', ROOT / 'research/smallk_predictive_alignment_20261013/controls.py')
        controls = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = controls
        spec.loader.exec_module(controls)
        old_assets = bridge_module.PREDICTION_ASSETS[args.dataset]
        bulk = {}
        for method, key in (('RAW_BOLT', 'raw'), ('MAIN', 'main_' + str(args.seed))):
            asset = old_assets[key]
            arrays, checked = runner.load_checked_npz({'path': str(asset.resolved), 'sha256': asset.sha256})
            predictions = runner.compare_reference(arrays, splits, mean, std)
            bulk[method] = {'prediction': predictions[split], 'source': checked}
        result.update(provenance=provenance, raw_input=checked_input, sources_before=sources_before,
                      checkpoint=checkpoint, F_before=f_before, codec_before=codec_before,
                      input_split=split, origins=inputs['origins'][:8].tolist(), original_bulk_sources={key: value['source'] for key, value in bulk.items()},
                      startup_load_cache_reference_outside_latency=True, actual_GPU_executed=True)
        backend = TorchBackend(engine)
        with torch.no_grad():
            for batch in (1, 8):
                physical, observed = inputs['history_physical'][:batch], inputs['history_observed'][:batch]
                old_history = engine.ex.preprocess(physical, observed, mean, std)
                if not np.array_equal(old_history, inputs['history'][:batch]):
                    raise ValueError('ACTUAL_OLD_PREPROCESS_BRIDGE_EXACT_PARITY')
                for method, loaded_codec in (('RAW_BOLT', None), ('MAIN', codec)):
                    expected_normalized = engine.predict(old_history, loaded_codec, bolt, watchdog)
                    expected_physical = controls.deployment({'mean': mean, 'std': std}, physical, observed,
                        'RAW_BOLT' if loaded_codec is None else 'BOLT', {'codec': loaded_codec}, bolt)
                    reference = {'normalized': expected_normalized, 'physical': expected_physical}
                    reference_file = runner.save_npz(output / f'{method}_B{batch}_matched_reference.npz', reference)
                    bulk_reference = {'normalized': bulk[method]['prediction'][:batch],
                                      'physical': bulk[method]['prediction'][:batch] * std + mean}
                    pipeline = DeploymentPipeline(backend, loaded_codec, bolt, mean, std, engine.ex.preprocess)
                    for preprocessing in ('old_loop', 'vectorized'):
                        folder = output / f'{method}_B{batch}_{preprocessing}'
                        folder.mkdir(exist_ok=False)
                        watchdog.check()
                        watchdog.fast_only = True
                        try:
                            row = measure_route(pipeline, physical, observed, preprocessing, reference, bulk_reference,
                                folder, watchdog, runner.save_npz)
                        finally:
                            watchdog.fast_only = False
                            watchdog.check()
                        row['resource_guard_scope'] = ('Full followup storage/CPU-supplement check before and after each '
                            'warm/repeated row; original shared process/campaign/GPU/RSS/VRAM guard at every trial/stage.')
                        row.update(method=method, matched_reference=reference_file)
                        result['rows'].append(row)
                        runner.write(folder / 'RESULT.json', row)
                        if row['status'] != 'COMPLETE':
                            raise ValueError('LATENCY_ROUTE_MATCHED_BATCH_PARITY_FAILED')
                        print(json.dumps({'event': 'LATENCY_ROUTE_COMPLETE', 'method': method, 'batch': batch,
                                          'preprocessing': preprocessing, 'whole_ms': row['whole_median_seconds'] * 1000}), flush=True)
        result.update(F_after=runner.model_hashes(bolt), codec_after=runner.old_state_hash(codec),
                      sources_after=runner.verify_source_records(bridge),
                      checkpoint_after=runner.verified_file(checkpoint['checkpoint']),
                      selected_case_after=runner.verified_file(checkpoint['case']),
                      paired_changes=paired_changes(result['rows']))
        result['helper_sources_after'] = [runner.verified_file(record) for record in helper_records]
        if result['F_before'] != result['F_after'] or result['codec_before'] != result['codec_after'] or sources_before != result['sources_after']:
            raise ValueError('MODEL_SOURCE_PRESERVATION_FAILED')
        watchdog.check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if bolt is not None and 'F_after' not in result:
            try:
                result['F_after'] = runner.model_hashes(bolt)
            except Exception as error:
                result['F_after_unavailable'] = {'type': type(error).__name__, 'message': str(error)}
        if codec is not None and 'codec_after' not in result:
            try:
                result['codec_after'] = runner.old_state_hash(codec)
            except Exception as error:
                result['codec_after_unavailable'] = {'type': type(error).__name__, 'message': str(error)}
        result['partial_coverage'] = {'complete_routes': sum(row['status'] == 'COMPLETE' for row in result['rows']),
                                      'planned_routes': 8, 'actual_repeated_trial_claim': False}
        result['wall_seconds'] = time.perf_counter() - start
        if limit is not None:
            receipt = limit.close(result['status'], result.get('error'))
            result['shared_resource_receipt'] = receipt
            terminal_status = receipt.get('terminal_status', result['status'])
            if terminal_status != 'COMPLETE':
                terminal_error = receipt.get('terminal_error', result.get('error'))
                if terminal_status != result['status'] or terminal_error != result.get('error'):
                    result['before_resource_close'] = {'status': result['status'], 'error': result.get('error')}
                result.update(status=terminal_status, error=terminal_error)
        runner.write(output / 'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--execute-after-bi', action='store_true')
    parser.add_argument('--bridge-result', type=Path)
    parser.add_argument('--dataset', choices=('robin', 'beijing_aotizhongxin'))
    parser.add_argument('--seed', type=int, choices=(92601, 92602))
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.describe or not args.execute_after_bi:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return
    if args.bridge_result is None or args.dataset is None or args.seed is None or args.tag is None:
        parser.error('Actual execution requires --bridge-result --dataset --seed --tag.')
    result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(CACHE / args.tag), 'routes': len(result['rows'])}, ensure_ascii=True))
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
