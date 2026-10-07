"""Seal the fixed S1 schedule and generate exact shared prefix caches."""
import gc
import time

import numpy as np
import torch

from model_readout import Readout, load_backbone
from runtime_readout import (CACHE, DATASETS, HERE, ROOT, SEEDS, array_hash, batches,
                             digest, features_to_cpu, inputs, load_data, read_json,
                             resolve, sample_epoch, save_json)


FEATURE_NAMES = ('h', 'loc', 'scale', 'point')
_verified_cache_files = {}


def require_s1(budget=None):
    authorization = read_json(HERE / 'authorization.json')
    if not authorization.get('s1_execution_authorized', False):
        raise RuntimeError('S1 execution is not authorized')
    if budget is not None and getattr(budget, 'stage', None) != 's1':
        raise RuntimeError('S1 requires the central S1 budget guard')
    calibration = read_json(HERE / 'calibration_result.json')
    if calibration.get('status') != 'PASS' or set(calibration.get('datasets', {})) != set(DATASETS):
        raise RuntimeError('S1 requires the completed three-dataset calibration PASS')
    return authorization


def receipt(path):
    path = resolve(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'bytes': path.stat().st_size}


def verify_receipt(entry):
    path = resolve(entry['path'])
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise RuntimeError('Sealed artifact changed: ' + entry['path'])
    return path


def ensure_disk_space(budget, needed_bytes):
    snapshot = budget.check()
    limit = snapshot['limits'].get('maximum_new_disk_bytes')
    if limit is not None and snapshot['new_disk_bytes'] + needed_bytes > limit:
        raise RuntimeError('STOP_RESOURCE: insufficient approved disk space for atomic cache/checkpoint write')


def _save_array_once(path, array):
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise RuntimeError('Existing fixed origin/schedule array differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.save(handle, array, allow_pickle=False)


def seal_sources():
    path = HERE / 'execution_source_seal.json'
    if path.exists():
        sealed = read_json(path)
        for entry in sealed['artifacts']:
            verify_receipt(entry)
        return sealed
    old = read_json(HERE / 'source_binding.json')['artifacts']
    for entry in old:
        verify_receipt(entry)
    local = ['model_readout.py', 'runtime_readout.py', 'runtime_s0_reference.py', 'budget_guard.py', 'prepare_readout.py', 'train_readout.py',
             'REQUEST_GOAL_PLAN.md', 'PLAN.md', 'data_contract.json', 'source_binding.json',
             's0_verification.json', 'calibration_result.json', 'guard_repair.json',
             's1_budget.json', 'run_readout.py']
    parent_paths = ['research/level_matched_learned_ed_v18_20261006/runtime_v18.py',
                    'research/level_bolt_backbone_scaling_v1_20261006/runtime.py',
                    'research/level_bolt_backbone_scaling_v1_20261006/baselines.py',
                    'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json']
    data_sources = [entry['source'] for entry in read_json(HERE / 'data_contract.json')['source_contracts'].values()]
    for entry in data_sources:
        verify_receipt(entry)
    artifacts = old + data_sources + [receipt(HERE / name) for name in local] + [receipt(ROOT / name) for name in parent_paths]
    sealed = {'schema': 'tsfm_readout_execution_source_seal_v1', 'sealed_utc': time.time(),
              'artifacts': artifacts, 'test_access': False}
    save_json(path, sealed)
    return sealed


def _protocol():
    value = {'schema': 'tsfm_readout_s1_protocol_v1', 'datasets': list(DATASETS),
             'arms': ['OUTPUT_AFFINE', 'NATIVE_HEAD'], 'learning_rates': [1e-3, 1e-4],
             'seeds': list(SEEDS), 'maximum_epochs': 120, 'patience': 6,
             'epoch_origins': 512, 'effective_batch': 4, 'maximum_scientific_fits': 24,
             'maximum_scientific_updates': 368640, 'optimizer': 'AdamW', 'weight_decay': 0.,
             'gradient_clip_norm': 1., 'scheduler': {'name': 'ReduceLROnPlateau', 'factor': .5,
             'patience': 2, 'threshold': 1e-4, 'threshold_mode': 'rel', 'observe_epoch0': False},
             'checkpoint_selection': 'full VAL channel-macro MSE strict minimum including epoch0; earliest exact tie',
             'learning_rate_selection': 'mean of two selected seed VAL losses; exact tie uses first-listed 1e-3',
             'sampler': 'literal phase-balanced SeedSequence([seed,epoch]), without replacement per phase then shuffle',
             'loss_coordinates': 'original inverse normalization then canonical TRAIN-standardized original channels',
             'missing_accumulation': 'entire effective B4 channel counts/valid set; sum micro SSE contributions then one clip/step',
             'dropout': 'all modules eval with the explicit head/output parameters requiring gradients',
             'cache': 'sorted fixed two-seed epochs1..120 TRAIN union plus full VAL; FP32 h/loc/scale/point',
             'test_access': False, 'seed_aggregation': 'mean of seed losses; no forecast ensemble',
             'bootstrap': {'draws': 2000, 'seed': 9262026, 'blocks': {'robin': 7, 'peacock_education': 7, 'jena': 42},
             'salts': {'robin': 0, 'peacock_education': 0, 'jena': 1}, 'strata': ['test_a', 'test_b'],
             'scheme': 'paired channels/seeds/methods, contiguous noncircular moving blocks; ceil draw then truncate'},
             'source_seal': receipt(HERE / 'execution_source_seal.json')}
    path = HERE / 'protocol.json'
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError('Existing S1 protocol differs')
    else:
        save_json(path, value)
    return value


def _seal_origins(budget):
    expected = read_json(HERE / 's0_verification.json')['schedule_union_read_only_replay']
    entries = {}
    for dataset in DATASETS:
        budget.check()
        data = load_data(dataset, include_test=False)
        directory = CACHE / 's1_features' / dataset
        schedule_arrays = {str(seed): np.stack([sample_epoch(data, seed, epoch) for epoch in range(1, 121)])
                           for seed in SEEDS}
        union = np.unique(np.concatenate([value.reshape(-1) for value in schedule_arrays.values()])).astype(np.int64)
        if len(union) != expected[dataset]['count'] or array_hash(union) != expected[dataset]['sorted_origin_bytes_sha256']:
            raise RuntimeError('Fixed TRAIN schedule union differs from S0: ' + dataset)
        val = np.asarray(data['val_origins'], dtype=np.int64)
        if np.intersect1d(union, val).size:
            raise RuntimeError('TRAIN schedule union overlaps VAL origins')
        origins = np.sort(np.concatenate((union, val))).astype(np.int64)
        _save_array_once(directory / 'origins.npy', origins)
        schedules = {}
        for seed, value in schedule_arrays.items():
            path = directory / ('schedule_' + seed + '.npy')
            _save_array_once(path, value)
            schedules[seed] = receipt(path) | {'shape': list(value.shape), 'array_sha256': array_hash(value)}
        entries[dataset] = {'dataset': dataset, 'channels': data['_C'], 'train_union_count': len(union),
                            'val_count': len(val), 'origin_count': len(origins),
                            'train_union_sha256': array_hash(union), 'val_origins_sha256': array_hash(val),
                            'origins': receipt(directory / 'origins.npy') | {'array_sha256': array_hash(origins)},
                            'schedules': schedules, 'data': {'path': data['_path'], 'sha256': digest(data['_path'])},
                            'order': 'sorted unique fixed TRAIN schedule union and full canonical VAL'}
        del data, schedule_arrays
        budget.check()
    path = HERE / 'cache_origins_seal.json'
    sealed = {'schema': 'tsfm_readout_cache_origins_seal_v1', 'datasets': entries,
              'validation_performance_used': False, 'test_access': False,
              'protocol': receipt(HERE / 'protocol.json')}
    if path.exists():
        if read_json(path) != sealed:
            raise RuntimeError('Fixed cache-origin seal differs')
    else:
        save_json(path, sealed)
    return sealed


def _open_arrays(directory, count, channels):
    shapes = {'h': (count, channels, 512), 'loc': (count, channels, 1),
              'scale': (count, channels, 1), 'point': (count, 48, channels)}
    values = {}
    for name, shape in shapes.items():
        path = directory / (name + '.npy')
        if path.exists():
            value = np.load(path, mmap_mode='r+', allow_pickle=False)
            if value.shape != shape or value.dtype != np.float32:
                raise RuntimeError('Existing partial cache shape/dtype differs: ' + name)
        else:
            value = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        values[name] = value
    done_path = directory / 'completed.npy'
    if done_path.exists():
        done = np.load(done_path, mmap_mode='r+', allow_pickle=False)
        if done.shape != (count,) or done.dtype != np.bool_:
            raise RuntimeError('Existing cache completion map differs')
    else:
        done = np.lib.format.open_memmap(done_path, mode='w+', dtype=np.bool_, shape=(count,))
        done[:] = False
        done.flush()
    return values, done


def _reuse_sample(data, report, origins, arrays, done):
    path = resolve(report['cache']['path'])
    sample_sha256 = digest(path)
    index = {int(origin): i for i, origin in enumerate(origins)}
    reused = 0
    with np.load(path, allow_pickle=False) as archive:
        sample = archive['origins'].astype(np.int64, copy=False)
        if array_hash(sample) != report['sample_origin_hash']:
            raise RuntimeError('S0 sample origin seal mismatch')
        if not np.array_equal(sample, np.concatenate((data['train_origins'][:24], data['val_origins']))):
            raise RuntimeError('S0 sample origins differ from canonical TRAIN/VAL')
        sample_features = {key: archive[key] for key in FEATURE_NAMES}
        for key in FEATURE_NAMES:
            if array_hash(sample_features[key]) != report['cache']['array_hashes'][key]:
                raise RuntimeError('S0 feature array seal mismatch: ' + key)
        for position, origin in enumerate(sample):
            destination = index.get(int(origin))
            if destination is None:
                continue
            for key in FEATURE_NAMES:
                value = sample_features[key][position]
                if done[destination] and not np.array_equal(arrays[key][destination], value):
                    raise RuntimeError('Existing cache differs from reused S0 sample: ' + key)
                arrays[key][destination] = value
            done[destination] = True
            reused += 1
    for value in arrays.values():
        value.flush()
    done.flush()
    if digest(path) != sample_sha256:
        raise RuntimeError('Sample cache changed while loading')
    return receipt(path) | {'reused_origin_count': reused, 'exact_array_values_copied': True}


def _prepare_dataset(dataset, origin_entry, calibration, budget):
    started = time.perf_counter()
    directory = CACHE / 's1_features' / dataset
    origins = np.load(verify_receipt(origin_entry['origins']), allow_pickle=False)
    data = load_data(dataset, include_test=False)
    backbone = load_backbone(dataset, device='cuda')
    model = Readout(backbone, data['_C'], 'OUTPUT_AFFINE').eval()
    frozen = model.frozen_hash()
    binding = {'dataset': dataset, 'frozen_parameters_and_all_buffers': frozen,
               'norm_callable': backbone.readout_parent_receipt['instance_norm_forward'],
               'data': origin_entry['data'], 'origins_sha256': array_hash(origins),
               'source_seal_sha256': digest(HERE / 'execution_source_seal.json'),
               'dtype': 'float32', 'quantile_index': 4, 'horizon': 48, 'hidden': 512}
    partial = directory / 'prefix_binding.json'
    if partial.exists():
        if read_json(partial) != binding:
            raise RuntimeError('Partial feature cache prefix binding changed')
    else:
        save_json(partial, binding)
    shapes = {'h': (len(origins), data['_C'], 512), 'loc': (len(origins), data['_C'], 1),
              'scale': (len(origins), data['_C'], 1), 'point': (len(origins), 48, data['_C'])}
    needed = sum(int(np.prod(shape)) * 4 + 256 for key, shape in shapes.items() if not (directory / (key + '.npy')).exists())
    if not (directory / 'completed.npy').exists():
        needed += len(origins) + 256
    ensure_disk_space(budget, needed + 32768)
    arrays, done = _open_arrays(directory, len(origins), origin_entry['channels'])
    budget.check()
    sample_receipt = _reuse_sample(data, calibration['datasets'][dataset], origins, arrays, done)
    generated = 0
    generated_before = int(done.sum())
    generation_start = time.perf_counter()
    try:
        for ids in batches(np.flatnonzero(~done), 4):
            budget.tick('prefix')
            budget.tick('head')
            values = features_to_cpu(model.features(inputs(data, origins[ids]).to('cuda')))
            for key, value in values.items():
                if not np.isfinite(value).all():
                    raise RuntimeError('Nonfinite generated cache feature: ' + key)
                arrays[key][ids] = value
            done[ids] = True
            generated += len(ids)
            if generated % 128 == 0:
                for value in arrays.values():
                    value.flush()
                done.flush()
                print('CACHE ' + dataset + ' ' + str(int(done.sum())) + '/' + str(len(origins)), flush=True)
            budget.check()
    finally:
        for value in arrays.values():
            value.flush()
        done.flush()
    generation_seconds = time.perf_counter() - generation_start
    torch.cuda.synchronize()
    if not bool(done.all()) or model.frozen_hash() != frozen:
        raise RuntimeError('Cache incomplete or frozen prefix weights/buffers changed')
    file_receipts = {key: receipt(directory / (key + '.npy')) | {'shape': list(value.shape), 'dtype': str(value.dtype)}
                     for key, value in arrays.items()}
    result = {'status': 'complete', 'dataset': dataset, 'origin_count': len(origins),
              'origins': origin_entry['origins'], 'features': file_receipts, 'prefix_binding': receipt(partial),
              's0_sample_reuse': sample_receipt, 'origins_existing_at_attempt_start': generated_before,
              'new_origin_forecasts_this_attempt': generated, 'generation_write_seconds_this_attempt': generation_seconds,
              'total_prepare_seconds_this_attempt': time.perf_counter() - started,
              'cache_payload_bytes': sum(value.nbytes for value in arrays.values()),
              'cache_file_bytes': sum(entry['bytes'] for entry in file_receipts.values()),
              'frozen_parameters_and_all_buffers_unchanged': True, 'test_access': False}
    save_json(directory / 'receipt.json', result)
    del model, backbone, data, arrays, done
    gc.collect()
    torch.cuda.empty_cache()
    budget.check()
    return result


def prepare_all(budget):
    require_s1(budget)
    seal_sources()
    _protocol()
    origins_seal = _seal_origins(budget)
    path = HERE / 'cache_manifest.json'
    manifest = read_json(path) if path.exists() else {
        'schema': 'tsfm_readout_s1_cache_manifest_v1', 'datasets': {}, 'test_access': False,
        'origins_seal': receipt(HERE / 'cache_origins_seal.json'),
        'source_seal': receipt(HERE / 'execution_source_seal.json'), 'protocol': receipt(HERE / 'protocol.json')}
    calibration = read_json(HERE / 'calibration_result.json')
    for key in ('origins_seal', 'source_seal', 'protocol'):
        verify_receipt(manifest[key])
    for dataset in DATASETS:
        if dataset in manifest['datasets']:
            load_cache(dataset)
            continue
        manifest['datasets'][dataset] = _prepare_dataset(dataset, origins_seal['datasets'][dataset], calibration, budget)
        save_json(path, manifest)
    manifest['status'] = 'complete'
    save_json(path, manifest)
    return manifest


def load_cache(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    manifest = read_json(HERE / 'cache_manifest.json')
    entry = manifest['datasets'][dataset]
    if entry.get('status') != 'complete':
        raise RuntimeError('Feature cache is not complete: ' + dataset)
    paths = {'origins': entry['origins']} | entry['features']
    for key, value in paths.items():
        path = resolve(value['path'])
        stamp = (value['sha256'], path.stat().st_size, path.stat().st_mtime_ns)
        if _verified_cache_files.get(str(path)) != stamp:
            verify_receipt(value)
            _verified_cache_files[str(path)] = stamp
    origins = np.load(resolve(entry['origins']['path']), mmap_mode='r', allow_pickle=False)
    features = {key: np.load(resolve(value['path']), mmap_mode='r', allow_pickle=False)
                for key, value in entry['features'].items()}
    index = {int(origin): i for i, origin in enumerate(origins)}
    if len(index) != len(origins):
        raise RuntimeError('Cache origins contain duplicate entries')
    return origins, features, index
