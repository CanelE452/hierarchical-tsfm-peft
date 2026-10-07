"""Selected-state technical probes and full-model matched deployment costs.

Nothing runs at import time. Call measure_all only from the central S1 guard.
All selected checkpoint files are read-only; every probe restores a fresh model.
"""
import csv
import gc
import hashlib
import os
import sys
import time
import traceback

import numpy as np
import torch

from model_readout import ARMS, Readout, compute_loss, load_backbone
from prepare_readout import load_cache, receipt, require_s1, verify_receipt
from runtime_readout import (DATASETS, HERE, ROOT, SEEDS, _v18_runtime, array_hash,
                             batches, digest, inputs, load_data, read_json,
                             resolve, save_json, targets)


BLOCKS = 3
WARMUP = 10
MEASURED = 10
TRAINING_UPDATES = 2160
TRAINING_SCOPE = ('CPU context/cache indexing and target/mask creation + H2D + forward + '
                  'point MSE + backward + clip1 + AdamW + synchronized completion; '
                  'budget ledger I/O excluded from individual update timers, included in probe wall')
DEPLOYMENT_SCOPE = ('preconstructed standardized CPU FP32 input + H2D + complete backbone/readout '
                    '+ contiguous CPU [B,48,C] result + CUDA synchronization; '
                    'budget ledger I/O excluded from wrapper timers, included in probe wall')


def _sync():
    torch.cuda.synchronize()


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _frozen_hash(model, allowlist):
    allowed = set(allowlist)
    digest_value = hashlib.sha256()
    parameters = [(name, value) for name, value in model.named_parameters(remove_duplicate=False)
                  if name not in allowed]
    for kind, entries in (('parameter', parameters), ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, value in sorted(entries):
            value = value.detach().cpu().contiguous()
            digest_value.update((kind + ':' + name).encode())
            digest_value.update(str(value.dtype).encode())
            digest_value.update(str(tuple(value.shape)).encode())
            digest_value.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest_value.hexdigest()


def _allowlist(model):
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if not names or any(module.training for module in model.modules()):
        raise RuntimeError('Technical training requires explicit trainables and every module in eval')
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise RuntimeError('Matched cost requires FP32 parameters')
    return names


def _old_rows(dataset):
    small = read_json(ROOT / 'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json')
    controls = read_json(ROOT / 'research/level_chronos2_controls_v1/reuse_manifest.json')
    rows = [dict(row) for row in small['units'][dataset]['rows'] if row['method'] in ('F0', 'LEVEL')]
    rows += [dict(row) for row in controls['units'][dataset]['rows'] if row['method'] == 'MSE_LORA']
    for method in ('LEVEL', 'MSE_LORA'):
        selected = [row for row in rows if row['method'] == method]
        if len(selected) != 2 or {row['seed'] for row in selected} != set(SEEDS):
            raise RuntimeError('Old selected two-seed rows are incomplete: ' + dataset + '/' + method)
    if len([row for row in rows if row['method'] == 'F0']) != 1:
        raise RuntimeError('Expected one F0 deployment instance per dataset')
    return rows


def _new_model(row, channels, budget):
    budget.check()
    saved = torch.load(verify_receipt(row['checkpoint']), map_location='cpu', weights_only=False)
    spec = saved['spec']
    if (spec['dataset'], spec['arm'], spec['seed']) != (row['dataset'], row['method'], row['seed']):
        raise RuntimeError('Selected checkpoint identity differs from joint choice seal')
    if float(spec['learning_rate']) != float(row['learning_rate']):
        raise RuntimeError('Selected checkpoint learning rate differs from joint choice seal')
    model = Readout(load_backbone(row['dataset'], device='cuda'), channels, row['method']).eval()
    model.load_trainable_state(saved['state'])
    if set(_allowlist(model)) != set(model.trainable_names):
        raise RuntimeError('New readout trainables differ from native/output allowlist')
    budget.check()
    return model


def _old_model(row, budget, deploy):
    budget.check()
    runtime = _v18_runtime()
    model = runtime.load_original(row, device='cuda', deploy=deploy)
    loader = sys.modules['matched_v18_original_baselines']
    if deploy:
        wrapper = loader.BaselineForecast(model, row, 'cuda')
        if row['method'] == 'MSE_LORA' and any('.lora_' in name for name, _ in model.named_parameters()):
            raise RuntimeError('MSE-LoRA deployment did not merge its original LoRA adapters')
        budget.check()
        return model, wrapper, None
    model.eval().requires_grad_(False)
    if row['method'] == 'MSE_LORA':
        if model.family not in ('full_lora_mse', 'full_mse') or getattr(model, 'deployed', False):
            raise RuntimeError('Old MSE-LoRA must restore its unmerged point-MSE training family')
        allowed = {name for name, _ in model.named_parameters() if '.lora_' in name}
        if not allowed:
            raise RuntimeError('Original MSE-LoRA adapters were not restored')
        for name, value in model.named_parameters():
            value.requires_grad_(name in allowed)
        if sum(value.numel() for name, value in model.named_parameters() if name in allowed) != 294912:
            raise RuntimeError('Original MSE-LoRA trainable count differs from the bound 294912')
    elif row['method'] == 'LEVEL':
        if model.family not in ('level_res', 'level') or row.get('output_component') == 'main':
            raise RuntimeError('Selected LEVEL row is not the original fitted residual forecast')
        residual_ids = {id(value) for value in model.residual.parameters()}
        for value in model.parameters():
            value.requires_grad_(id(value) in residual_ids)
    else:
        raise ValueError('F0 has no training probe')
    _allowlist(model)
    source = loader.source_module(row)
    if resolve(row['restore_module']).name in ('model_v6.py', 'model_v7.py', 'model_v11.py'):
        loss_function = source.macro_loss
    elif resolve(row['restore_module']).name == 'model_confirmation_v12.py':
        loss_function = sys.modules['runtime_confirmation_v12'].masked_macro_loss
    else:
        raise RuntimeError('Cannot recreate the hash-bound original point-MSE loss callable')
    budget.check()
    return model, None, loss_function


def _training_cache(dataset, data, budget):
    started = time.perf_counter()
    budget.check()
    _, features, index = load_cache(dataset)
    origins = np.asarray(data['train_origins'][:24], dtype=np.int64)
    missing = [int(origin) for origin in origins if int(origin) not in index]
    fallback, fallback_index, fallback_receipt = {}, {}, None
    if missing:
        report = read_json(HERE / ('calibration_' + dataset + '.json'))
        entry = report['cache']
        path = resolve(entry['path'])
        if path.stat().st_size != entry['bytes']:
            raise RuntimeError('Immutable S0 cache file size changed')
        fallback_receipt = receipt(path)
        with np.load(path, allow_pickle=False) as archive:
            sample = archive['origins'].astype(np.int64, copy=False)
            if array_hash(sample) != report['sample_origin_hash']:
                raise RuntimeError('Immutable S0 sample origins changed')
            if not np.array_equal(sample[:24], origins):
                raise RuntimeError('S0 fallback differs from canonical TRAIN first24')
            fallback_index = {int(origin): position for position, origin in enumerate(sample)}
            for key, expected in entry['array_hashes'].items():
                value = archive[key]
                if array_hash(value) != expected:
                    raise RuntimeError('Immutable S0 feature bytes changed: ' + key)
                fallback[key] = np.ascontiguousarray(value[:24])
        if digest(path) != fallback_receipt['sha256']:
            raise RuntimeError('S0 fallback file changed during cache read')
    cpu = {}
    for key in ('h', 'loc', 'scale', 'point'):
        cpu[key] = np.stack([features[key][index[int(origin)]] if int(origin) in index
                             else fallback[key][fallback_index[int(origin)]] for origin in origins])
        if cpu[key].dtype != np.float32:
            raise RuntimeError('Exact immutable CPU features must be FP32')
    del features, fallback
    budget.check()
    return cpu, {'dataset': dataset, 'cache_read_and_verify_seconds': time.perf_counter() - started,
                 'origins_sha256': array_hash(origins), 'origins': 24, 's0_fallback_origins': missing,
                 's0_fallback_receipt': fallback_receipt,
                 'feature_bytes': sum(value.nbytes for value in cpu.values()),
                 'feature_array_hashes': {key: array_hash(value) for key, value in cpu.items()}}


def _guard_calls(row, budget, path='online'):
    if path == 'online':
        budget.tick('prefix')
        budget.tick('head', 3 if row['method'] == 'LEVEL' else 2)
    else:
        budget.tick('head')


def _step(model, optimizer, row, data, cache, update, path, loss_function):
    ids = np.arange((update % 6) * 4, (update % 6) * 4 + 4)
    origins = np.asarray(data['train_origins'][:24], dtype=np.int64)[ids]
    target, observed = targets(data, origins)
    target, observed = target.to('cuda'), observed.to('cuda')
    optimizer.zero_grad(set_to_none=True)
    if path == 'online':
        prediction = model(inputs(data, origins).to('cuda'))
    else:
        keys = ('point',) if row['method'] == 'OUTPUT_AFFINE' else ('h', 'loc', 'scale')
        features = {key: torch.from_numpy(np.ascontiguousarray(cache[key][ids])).to('cuda') for key in keys}
        if any(value.is_inference() for value in features.values()):
            raise RuntimeError('Cached source features must be ordinary tensors')
        prediction = model.from_features(features)
    loss = loss_function(prediction, target, observed)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite technical point-MSE loss')
    loss.backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in parameters):
        raise RuntimeError('Missing/nonfinite original trainable gradients')
    if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
        raise RuntimeError('Frozen parameter received a gradient')
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()


def _training_block(row, data, cache, path, block, budget):
    result = {key: row[key] for key in ('dataset', 'method', 'id', 'seed')}
    result.update(path=path, block=block, status='running', effective_batch=4,
                  warmup_updates=0, measured_updates=0, selected_checkpoint=row['checkpoint'],
                  selected_checkpoint_sha256_before=digest(row['checkpoint']['path']),
                  scope=TRAINING_SCOPE, frozen_prefix_kept_on_gpu=True)
    model, optimizer = None, None
    started = time.perf_counter()
    try:
        if row['source_kind'] == 'readout':
            model, loss_function = _new_model(row, data['_C'], budget), compute_loss
        else:
            model, _, loss_function = _old_model(row, budget, deploy=False)
        allowlist = _allowlist(model)
        frozen = _frozen_hash(model, allowlist)
        versions = {name: value._version for name, value in model.named_parameters() if name not in allowlist}
        initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if name in allowlist}
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-4, weight_decay=0.)
        result.update(trainable_names=list(allowlist), trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
                      original_loss_callable=loss_function.__module__ + '.' + loss_function.__qualname__)
        measured = []
        rss = []
        for update in range(WARMUP + MEASURED):
            _guard_calls(row, budget, path)
            snap = budget.check()
            limit = snap['limits'].get('maximum_technical_optimizer_updates')
            if limit is not None and snap['counters'].get('technical_updates', 0) >= int(limit):
                raise RuntimeError('STOP_RESOURCE: no remaining technical optimizer update budget')
            if update == WARMUP:
                _sync()
                result['adam_state_bytes'] = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                                                 for value in state.values() if torch.is_tensor(value))
                if not result['adam_state_bytes']:
                    raise RuntimeError('Adam moments were not created during warmup')
                result['allocated_bytes_after_adam_warmup'] = torch.cuda.memory_allocated()
                result['reserved_bytes_after_adam_warmup'] = torch.cuda.memory_reserved()
                result['process_rss_bytes_after_adam_warmup'] = snap.get('rss_bytes')
                result['process_lifetime_peak_rss_bytes_after_adam_warmup'] = snap.get('peak_rss_bytes')
                budget.check(force=True)
                torch.cuda.reset_peak_memory_stats()
            _sync()
            before = time.perf_counter()
            _step(model, optimizer, row, data, cache, update, path, loss_function)
            _sync()
            elapsed = time.perf_counter() - before
            result['warmup_updates' if update < WARMUP else 'measured_updates'] += 1
            snap = budget.tick('update')
            if update >= WARMUP:
                measured.append(elapsed)
                if snap.get('rss_bytes') is not None:
                    rss.append(snap['rss_bytes'])
        if _frozen_hash(model, allowlist) != frozen or any(value._version != versions[name] for name, value in model.named_parameters() if name not in allowlist):
            raise RuntimeError('Frozen parameters or all named buffers changed during technical training')
        if not any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters() if name in allowlist):
            raise RuntimeError('Disposable selected trainable state did not change after 20 updates')
        result.update(status='complete', step_seconds=measured, median_step_seconds=float(np.median(measured)),
                      peak_allocated_bytes_after_adam=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes_after_adam=torch.cuda.max_memory_reserved(),
                      peak_sampled_process_rss_bytes=max(rss) if rss else None,
                      frozen_parameters_and_all_buffers_hash=frozen, frozen_unchanged=True)
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result['probe_wall_seconds_including_restore_hash_and_budget_io'] = time.perf_counter() - started
        result['selected_checkpoint_sha256_after'] = digest(row['checkpoint']['path'])
        result['selected_checkpoint_unchanged'] = result['selected_checkpoint_sha256_before'] == result['selected_checkpoint_sha256_after'] == row['checkpoint']['sha256']
        del optimizer, model
        _cleanup()
        save_json(HERE / 'cost_receipts' / ('train_' + row['id'] + '_' + path + '_block' + str(block) + '.json'), result)
    if not result['selected_checkpoint_unchanged']:
        raise RuntimeError('Selected scientific checkpoint changed during technical training')
    return result


class ReadoutForecast:
    def __init__(self, model):
        self.module = model.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, cpu_x):
        if cpu_x.device.type != 'cpu' or cpu_x.dtype != torch.float32:
            raise ValueError('Deployment starts with standardized CPU FP32 input')
        return self.module(cpu_x.to('cuda')).contiguous().cpu()


def _deployment_block(row, data, batch, block, budget, sentinel=None):
    result = {key: row[key] for key in ('dataset', 'method', 'id', 'seed')}
    result.update(batch=batch, block=block, sentinel=sentinel, status='running',
                  origins=24, complete_origin_passes=10, warmup_wrapper_calls=10,
                  scope=DEPLOYMENT_SCOPE, feature_cache_used=False, frozen_prefix_kept_on_gpu=True)
    model, wrapper = None, None
    started = time.perf_counter()
    try:
        budget.check(force=True)
        torch.cuda.reset_peak_memory_stats()
        if row['source_kind'] == 'readout':
            model = _new_model(row, data['_C'], budget)
            wrapper = ReadoutForecast(model)
        else:
            model, wrapper, _ = _old_model(row, budget, deploy=True)
        if any(module.training for module in model.modules()) or any(value.requires_grad for value in model.parameters()):
            raise RuntimeError('Deployment wrapper did not freeze/eval the complete model')
        frozen = _frozen_hash(model, ())
        origins = np.asarray(data['val_origins'][:24], dtype=np.int64)
        cpu_batches = [inputs(data, values) for values in batches(origins, batch)]
        if len(origins) != 24 or any(len(value) > 4 for value in cpu_batches):
            raise RuntimeError('Deployment must use canonical VAL first24 at B1/B4')
        parameters, buffers = list(model.parameters()), list(model.buffers())
        result.update(val_origins_sha256=array_hash(origins),
                      parameter_count=sum(value.numel() for value in parameters),
                      parameter_bytes=sum(value.numel() * value.element_size() for value in parameters),
                      buffer_bytes=sum(value.numel() * value.element_size() for value in buffers),
                      deployment_copy=True, old_deploy_function=row.get('deploy_function'),
                      selected_checkpoint=row.get('checkpoint'))
        for call in range(WARMUP):
            _guard_calls(row, budget)
            output = wrapper(cpu_batches[call % len(cpu_batches)])
            if output.shape != (len(cpu_batches[call % len(cpu_batches)]), 48, data['_C']) or not output.is_contiguous():
                raise RuntimeError('Deployment wrapper output axes/contiguity changed')
            del output
            _sync()
        result['peak_allocated_bytes_including_restore_and_warmup'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes_including_restore_and_warmup'] = torch.cuda.max_memory_reserved()
        budget.check(force=True)
        torch.cuda.reset_peak_memory_stats()
        pass_seconds, rss = [], []
        for repeat in range(MEASURED):
            elapsed = 0.
            for cpu_x in cpu_batches:
                _guard_calls(row, budget)
                _sync()
                before = time.perf_counter()
                output = wrapper(cpu_x)
                _sync()
                elapsed += time.perf_counter() - before
                if output.device.type != 'cpu' or output.dtype != torch.float32 or output.shape != (len(cpu_x), 48, data['_C']) or not output.is_contiguous() or not bool(torch.isfinite(output).all()):
                    raise RuntimeError('Deployment output must be finite contiguous CPU FP32 [B,48,C]')
                del output
                snap = budget.check()
                if snap.get('rss_bytes') is not None:
                    rss.append(snap['rss_bytes'])
            pass_seconds.append(elapsed)
        if _frozen_hash(model, ()) != frozen:
            raise RuntimeError('Full deployment model parameters/all buffers changed')
        median = float(np.median(pass_seconds))
        result.update(status='complete', complete_pass_seconds=pass_seconds,
                      median_complete_24_origin_pass_seconds=median, median_seconds_per_origin=median / 24,
                      origins_per_second=24 / median, latency_aggregation='median of ten complete 24-origin pass sums / 24',
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      resident_allocated_bytes=torch.cuda.memory_allocated(), resident_reserved_bytes=torch.cuda.memory_reserved(),
                      peak_sampled_process_rss_bytes=max(rss) if rss else None,
                      frozen_parameters_and_all_buffers_hash=frozen, frozen_unchanged=True)
        result['peak_allocated_bytes_complete_probe'] = max(result['peak_allocated_bytes_including_restore_and_warmup'], result['peak_allocated_bytes'])
        result['peak_reserved_bytes_complete_probe'] = max(result['peak_reserved_bytes_including_restore_and_warmup'], result['peak_reserved_bytes'])
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result['probe_wall_seconds_including_restore_hash_and_budget_io'] = time.perf_counter() - started
        del wrapper, model
        _cleanup()
        suffix = '_sentinel_' + sentinel if sentinel else ''
        save_json(HERE / 'cost_receipts' / ('deploy_' + row['id'] + '_b' + str(batch) + '_block' + str(block) + suffix + '.json'), result)
    return result


def _summaries(rows, training):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['method'], row['id'], row['path'] if training else row['batch'])
        groups.setdefault(key, []).append(row)
    instances = []
    field = 'median_step_seconds' if training else 'median_seconds_per_origin'
    for (dataset, method, run_id, condition), values in groups.items():
        good = [value[field] for value in values if value['status'] == 'complete']
        instances.append({'dataset': dataset, 'method': method, 'id': run_id,
                          'path' if training else 'batch': condition, 'complete_blocks': len(good),
                          'status': 'complete' if len(good) == BLOCKS else 'incomplete',
                          field: float(np.median(good)) if good else None,
                          'block_min_seconds': min(good) if good else None, 'block_max_seconds': max(good) if good else None,
                          'block_range_is_confidence_interval': False})
    methods = []
    for dataset in DATASETS:
        for method in ('OUTPUT_AFFINE', 'NATIVE_HEAD', 'MSE_LORA', 'LEVEL') if training else ('F0', 'OUTPUT_AFFINE', 'NATIVE_HEAD', 'MSE_LORA', 'LEVEL'):
            for condition in ('online', 'cached') if training else (1, 4):
                values = [value for value in instances if value['dataset'] == dataset and value['method'] == method
                          and value['path' if training else 'batch'] == condition]
                if not values:
                    continue
                expected = 1 if method == 'F0' else 2
                complete = len(values) == expected and all(value['status'] == 'complete' for value in values)
                value = float(np.mean([entry[field] for entry in values])) if complete else None
                methods.append({'dataset': dataset, 'method': method, 'path' if training else 'batch': condition,
                                'selected_seed_instances': len(values), 'status': 'complete' if complete else 'incomplete',
                                'mean_selected_seed_median_seconds': value,
                                'origins_per_second': (1 / value if value else None) if not training else None,
                                'aggregation': 'mean of selected individual-seed three-block medians',
                                'block_range_is_confidence_interval': False})
    return {'instances': instances, 'methods': methods}


def _csv(path, rows, fields):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sentinel(pair, phase, row, data, batch, block, budget):
    try:
        pair[phase] = _deployment_block(row, data, batch, block, budget, sentinel=phase)
    except BaseException:
        pair[phase] = read_json(HERE / 'cost_receipts' / ('deploy_' + row['id'] + '_b' + str(batch)
                                                       + '_block' + str(block) + '_sentinel_' + phase + '.json'))
        pair['status'] = 'failed'
        raise


def _save(result):
    result['training_summary'] = _summaries(result['training_rows'], training=True)
    result['deployment_summary'] = _summaries(result['deployment_rows'], training=False)
    save_json(HERE / 'costrows.json', result)
    _csv(HERE / 'costonline.csv', result['training_rows'], ('dataset', 'method', 'id', 'seed', 'path', 'block', 'status',
         'effective_batch', 'warmup_updates', 'measured_updates', 'median_step_seconds', 'trainable_parameters', 'adam_state_bytes',
         'peak_allocated_bytes_after_adam', 'peak_reserved_bytes_after_adam', 'peak_sampled_process_rss_bytes',
         'frozen_prefix_kept_on_gpu', 'selected_checkpoint_unchanged', 'probe_wall_seconds_including_restore_hash_and_budget_io', 'error'))
    _csv(HERE / 'deploy.csv', result['deployment_rows'], ('dataset', 'method', 'id', 'seed', 'batch', 'block', 'status',
         'median_complete_24_origin_pass_seconds', 'median_seconds_per_origin', 'origins_per_second', 'parameter_count',
         'parameter_bytes', 'buffer_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes', 'resident_allocated_bytes',
         'peak_allocated_bytes_complete_probe', 'peak_reserved_bytes_complete_probe',
         'resident_reserved_bytes', 'peak_sampled_process_rss_bytes', 'feature_cache_used', 'frozen_prefix_kept_on_gpu',
         'probe_wall_seconds_including_restore_hash_and_budget_io', 'error'))


def measure_all(budget):
    require_s1(budget)
    from evaluate_readout import load_selection
    _, selected = load_selection(budget)
    protocol = read_json(HERE / 'cost_protocol.json')
    if (protocol['training']['maximum_technical_updates'], protocol['training']['effective_batch'],
        protocol['training']['blocks'], protocol['training']['warmup_updates_per_block'],
        protocol['training']['measured_updates_per_block']) != (TRAINING_UPDATES, 4, BLOCKS, WARMUP, MEASURED):
        raise RuntimeError('Sealed training cost protocol differs from implemented fixed schedule')
    if (protocol['training']['device'], protocol['training']['learning_rate'], protocol['training']['weight_decay'],
        protocol['training']['clip'], protocol['training']['new_selected_models'], protocol['training']['new_paths']) != ('cuda', 1e-4, 0, 1, 12, ['online', 'cached']):
        raise RuntimeError('Sealed optimizer/path contract differs from implementation')
    if (protocol['deployment']['batches'] != [1, 4] or protocol['deployment']['complete_24_origin_passes'] != MEASURED
            or protocol['deployment']['rotated_blocks'] != BLOCKS or protocol['deployment']['warmup_wrapper_calls'] != WARMUP
            or protocol['deployment']['device'] != 'cuda'
            or protocol['deployment']['methods'] != ['F0', 'OUTPUT_AFFINE', 'NATIVE_HEAD', 'MSE_LORA', 'LEVEL']):
        raise RuntimeError('Sealed deployment cost protocol differs from implemented schedule')
    path = HERE / 'costrows.json'
    if path.exists():
        previous = read_json(path)
        if previous.get('status') == 'complete':
            verify_receipt(previous['protocol'])
            verify_receipt(previous['selection_seal'])
            for entry in previous['checkpoint_preservation']:
                verify_receipt(entry['checkpoint'])
            return previous
        raise RuntimeError('Partial/failed cost measurements are retained; central driver must authorize a bounded retry')
    new_rows = [dict(row, method=row['arm'], learning_rate=row['lr'], source_kind='readout') for row in selected]
    old = {dataset: _old_rows(dataset) for dataset in DATASETS}
    for rows in old.values():
        for row in rows:
            row['source_kind'] = 'baseline'
    checkpoints = {resolve(row['checkpoint']['path']).as_posix(): row['checkpoint']
                   for row in new_rows + [row for rows in old.values() for row in rows] if row.get('checkpoint')}
    preserved = [{'checkpoint': entry, 'before_sha256': digest(entry['path'])} for entry in checkpoints.values()]
    if any(entry['before_sha256'] != entry['checkpoint']['sha256'] for entry in preserved):
        raise RuntimeError('A sealed selected checkpoint changed before cost probes')
    started = time.perf_counter()
    result = {'schema': 'tsfm_readout_cost_rows_v1', 'status': 'running', 'protocol': receipt(HERE / 'cost_protocol.json'),
              'selection_seal': receipt(HERE / 'selection_seal.json'), 'training_rows': [], 'deployment_rows': [],
              'sentinels': [], 'cache_reads': [], 'checkpoint_preservation': preserved,
              'scientific_fits': 0, 'scientific_updates': 0, 'test_access': False,
              'historical_costs_used_in_speed_ratios': False, 'budget_before': budget.snapshot(),
              'timing_guard_io': 'excluded from individual step/wrapper timing; included in elapsed job/probe wall',
              'interpretation': protocol['interpretation']}
    _save(result)
    try:
        for dataset in DATASETS:
            budget.heartbeat('selected-state costs ' + dataset)
            data = load_data(dataset, include_test=False)
            if len(data['train_origins']) < 24 or len(data['val_origins']) < 24:
                raise RuntimeError('Canonical TRAIN/VAL lacks first24 technical origins')
            cache, cache_receipt = _training_cache(dataset, data, budget)
            result['cache_reads'].append(cache_receipt)
            dataset_new = [row for row in new_rows if row['dataset'] == dataset]
            train_instances = dataset_new + [row for row in old[dataset] if row['method'] != 'F0']
            for row in train_instances:
                for training_path in ('online', 'cached') if row['source_kind'] == 'readout' else ('online',):
                    for block in range(BLOCKS):
                        try:
                            measured = _training_block(row, data, cache, training_path, block, budget)
                        except BaseException:
                            measured = read_json(HERE / 'cost_receipts' / ('train_' + row['id'] + '_' + training_path + '_block' + str(block) + '.json'))
                            result['training_rows'].append(measured)
                            _save(result)
                            raise
                        result['training_rows'].append(measured)
                        _save(result)
            del cache
            instances = [row for row in old[dataset] if row['method'] == 'F0'] + dataset_new + [row for row in old[dataset] if row['method'] != 'F0']
            if len(instances) != 9:
                raise RuntimeError('Expected F0 plus four two-seed deployment methods')
            f0 = next(row for row in instances if row['method'] == 'F0')
            for batch in (1, 4):
                for block in range(BLOCKS):
                    pair = {'dataset': dataset, 'batch': batch, 'block': block, 'status': 'running'}
                    result['sentinels'].append(pair)
                    _sentinel(pair, 'before', f0, data, batch, block, budget)
                    rotated = instances[block:] + instances[:block]
                    for position, row in enumerate(rotated):
                        try:
                            measured = _deployment_block(row, data, batch, block, budget)
                        except BaseException:
                            measured = read_json(HERE / 'cost_receipts' / ('deploy_' + row['id'] + '_b' + str(batch) + '_block' + str(block) + '.json'))
                            measured['position_in_rotated_block'] = position
                            result['deployment_rows'].append(measured)
                            pair.update(status='failed', failed_instance_id=row['id'],
                                        after_sentinel_not_run='cost grid stopped at retained technical failure')
                            _save(result)
                            raise
                        measured['position_in_rotated_block'] = position
                        result['deployment_rows'].append(measured)
                        _save(result)
                    _sentinel(pair, 'after', f0, data, batch, block, budget)
                    pair.update(status='complete', after_over_before_latency=pair['after']['median_seconds_per_origin'] / pair['before']['median_seconds_per_origin'],
                                drift_threshold_predeclared=False, drift_interpretation='raw drift retained; no unregistered rejection threshold')
                    _save(result)
            del data
            _cleanup()
        after = budget.snapshot()
        before = result['budget_before']['counters']
        if after['counters'].get('technical_updates', 0) - before.get('technical_updates', 0) != TRAINING_UPDATES:
            raise RuntimeError('Technical update count differs from the exact 1440 new + 720 old schedule')
        if len(result['training_rows']) != 108 or len(result['deployment_rows']) != 162 or len(result['sentinels']) != 18:
            raise RuntimeError('Completed matched cost grid row count changed')
        for counter in ('scientific_fits', 'scientific_updates', 'new_test_predictions'):
            if after['counters'].get(counter, 0) != before.get(counter, 0):
                raise RuntimeError('Cost probes changed a scientific/TEST counter: ' + counter)
        result.update(status='complete', technical_updates=TRAINING_UPDATES, new_technical_updates=1440, old_technical_updates=720,
                      budget_after=after)
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      failure_policy='preserve failed rows and receipts; defer OOM/resource caps to central driver; no process killing')
        raise
    finally:
        result['elapsed_probe_seconds'] = time.perf_counter() - started
        for entry in preserved:
            entry['after_sha256'] = digest(entry['checkpoint']['path'])
            entry['unchanged'] = entry['before_sha256'] == entry['after_sha256'] == entry['checkpoint']['sha256']
        result['selected_checkpoints_unchanged'] = all(entry['unchanged'] for entry in preserved)
        if not result['selected_checkpoints_unchanged']:
            result.update(status='failed', checkpoint_preservation_failure=True)
        _save(result)
    if not result['selected_checkpoints_unchanged']:
        raise RuntimeError('Selected checkpoint bytes changed during cost measurements')
    return result
