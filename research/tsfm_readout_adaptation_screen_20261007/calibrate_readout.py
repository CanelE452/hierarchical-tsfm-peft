"""Bounded, disposable S0 integrity and cost calibration. Never selects a model."""
import copy
import gc
import json
import random
import time
import traceback

import numpy as np
import torch

from model_readout import Readout, compute_loss, load_backbone
from runtime_readout import (Budget, CACHE, HERE, ROOT, array_hash, configure,
                             digest, inputs, load_data, read_json, save_json, targets)


def verify_sources(budget):
    binding = read_json(HERE / 'source_binding.json')
    receipts = []
    for entry in binding['artifacts']:
        actual = digest(entry['path'])
        if actual != entry['sha256']:
            raise AssertionError('Pinned source/model changed: '+entry['path'])
        receipts.append({'path': entry['path'], 'sha256': actual})
        budget.check()
    additional = [
        'research/level_matched_learned_ed_v18_20261006/runtime_v18.py',
        'research/level_bolt_backbone_scaling_v1_20261006/runtime.py',
        'research/level_bolt_backbone_scaling_v1_20261006/baselines.py',
        'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json',
        'research/tsfm_peft_practical_controls_v7_20260928/model_v7.py',
        'research/tsfm_peft_level_confirmation_v6_20260928/runtime_v6.py',
        'research/tsfm_peft_practical_controls_v7_20260928/runtime_v7.py',
        'research/tsfm_peft_a_confirmation_v12_20261001/runtime_v12.py',
        'research/tsfm_peft_a_confirmation_v12_20261001/runtime_confirmation_v12.py',
    ]
    additional += [str((HERE / name).relative_to(ROOT)) for name in
                   ('model_readout.py', 'runtime_readout.py', 'calibrate_readout.py', 'authorization.json')]
    for path in additional:
        receipts.append({'path': path, 'sha256': digest(path)})
        budget.check()
    save_json(HERE / 'implementation_binding.json', {'schema': 'tsfm_readout_implementation_binding_v1',
              'verified_existing_binding': True, 'artifacts': receipts, 'test_data_loaded': False})


def sync():
    torch.cuda.synchronize()


def cpu_features(features):
    return {key: value.detach().cpu().contiguous() for key, value in features.items()}


def feature_slice(features, indices, device='cuda'):
    return {key: value[indices].to(device) for key, value in features.items()}


def forward(model, x, budget):
    budget.tick('prefix')
    budget.tick('head', 2 if model.arm == 'OUTPUT_AFFINE' else 1)
    value = model(x.to('cuda'))
    budget.check()
    return value


def cached_forward(model, features, budget):
    budget.tick('head')
    value = model.from_features(features)
    budget.check()
    return value


def close(a, b, label, atol=1e-5, rtol=1e-4):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=label)
    return {'maximum_absolute_difference': float((a-b).abs().max().cpu()),
            'atol': atol, 'rtol': rtol}


def grads(model):
    values = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError('Missing/nonfinite gradient: ' + name)
            values[name] = parameter.grad.detach().cpu().clone()
    if not any(torch.count_nonzero(value) for value in values.values()):
        raise AssertionError('All trainable gradients are zero')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise AssertionError('Frozen parameter has a gradient')
    return values


def backward(model, data, origins, features, indices, budget, path, micro=4, mask_override=None):
    model.zero_grad(set_to_none=True)
    target, mask = targets(data, origins)
    if mask_override is not None:
        mask = mask_override
    count = mask.sum((0, 1)).to('cuda')
    loss_sum = 0.
    predictions = []
    for start in range(0, len(origins), micro):
        stop = start + micro
        pred = (forward(model, inputs(data, origins[start:stop]), budget) if path == 'online'
                else cached_forward(model, feature_slice(features, indices[start:stop]), budget))
        loss = compute_loss(pred, target[start:stop].to('cuda'), mask[start:stop].to('cuda'), count)
        if not torch.isfinite(loss):
            raise AssertionError('Nonfinite supervised loss')
        loss.backward()
        loss_sum += float(loss.detach().cpu())
        predictions.append(pred.detach().cpu())
    return loss_sum, grads(model), torch.cat(predictions)


def compare_gradients(reference, actual, label):
    if reference.keys() != actual.keys():
        raise AssertionError('Gradient allowlist mismatch')
    detail = {}
    for name in reference:
        detail[name] = close(actual[name], reference[name], label + ':' + name, 1e-6, 1e-4)
    return detail


def batch_ranges(length):
    return [np.arange(start, min(start+4, length)) for start in range(0, length, 4)]


def full_val(model, data, origins, features, offset, budget, path):
    predictions = []
    with torch.no_grad():
        for ids in batch_ranges(len(origins)):
            value = (forward(model, inputs(data, origins[ids]), budget) if path == 'online'
                     else cached_forward(model, feature_slice(features, ids+offset), budget))
            predictions.append(value.cpu())
    prediction = torch.cat(predictions)
    target, mask = targets(data, origins)
    # Evaluation accumulates all origins before the channel-macro division.
    loss = compute_loss(prediction.double(), target.double(), mask)
    return float(loss), prediction


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def rng_draw():
    return [random.random(), float(np.random.random()), float(torch.rand(())), float(torch.rand((), device='cuda'))]


def step(model, optimizer, data, origins, features, ids, budget, path):
    target, mask = targets(data, origins)
    optimizer.zero_grad(set_to_none=True)
    prediction = (forward(model, inputs(data, origins), budget) if path == 'online'
                  else cached_forward(model, feature_slice(features, ids), budget))
    loss = compute_loss(prediction, target.to('cuda'), mask.to('cuda'))
    if not torch.isfinite(loss):
        raise AssertionError('Nonfinite probe loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise AssertionError('Nonfinite/missing probe gradient')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    budget.tick('update')
    optimizer.step()
    budget.check()
    return float(loss.detach().cpu())


def restore_check(model, initial, data, sample, features, budget, dataset):
    model.load_trainable_state(initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-4)
    ids = np.arange(4)
    step(model, optimizer, data, sample[:4], features, ids, budget, 'cached')
    scheduler.step(1.)
    state = {'model': model.trainable_state(), 'optimizer': copy.deepcopy(optimizer.state_dict()),
             'scheduler': scheduler.state_dict(), 'rng': rng_state()}
    path = CACHE / ('restore_' + dataset + '_' + model.arm + '.pt')
    io_start = time.perf_counter()
    torch.save(state, path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    draw = rng_draw()
    step(model, optimizer, data, sample[:4], features, ids, budget, 'cached')
    scheduler.step(2.)
    expected = model.trainable_state()
    expected_scheduler = scheduler.state_dict()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    model.load_trainable_state(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    if draw != rng_draw():
        raise AssertionError('RNG restore failed')
    step(model, optimizer, data, sample[:4], features, ids, budget, 'cached')
    scheduler.step(2.)
    for key, value in model.trainable_state().items():
        close(value, expected[key], 'restored parameter:'+key, 0., 0.)
    if scheduler.state_dict() != expected_scheduler:
        raise AssertionError('Scheduler restore failed')
    for key, value in optimizer.state_dict()['state'].items():
        for name, tensor in value.items():
            close(tensor, expected_optimizer['state'][key][name], 'restored optimizer:'+name, 0., 0.)
    model.load_trainable_state(initial)
    return {'status': 'PASS', 'technical_updates': 3, 'bytes': path.stat().st_size,
            'save_load_and_replay_seconds': time.perf_counter()-io_start}


def run_dataset(dataset, budget):
    print('DATASET ' + dataset, flush=True)
    data = load_data(dataset)
    train = np.asarray(data['train_origins'], dtype=np.int64)
    val = np.asarray(data['val_origins'], dtype=np.int64)
    sample = np.concatenate((train[:24], val))
    backbone = load_backbone(dataset, device='cuda')
    native_initial = copy.deepcopy(backbone.output_patch_embedding.state_dict())
    output = Readout(backbone, data['_C'], 'OUTPUT_AFFINE').eval()
    norm_callable = backbone.instance_norm.forward
    norm_source = getattr(norm_callable, '__func__', norm_callable)
    report = {'dataset': dataset, 'sample_origin_hash': array_hash(sample), 'train_sample': 24,
              'val_origins': len(val), 'channels': data['_C'], 'norm_callable': norm_source.__module__+'.'+norm_source.__qualname__,
              'gates': {}, 'timings': {}, 'cache': {}}
    reference = []
    with torch.no_grad():
        for ids in batch_ranges(len(val)):
            x = inputs(data, val[ids]).to('cuda')
            b, _, c = x.shape
            budget.tick('prefix')
            budget.tick('head')
            value = backbone(context=x.transpose(1, 2).reshape(b*c, 512)).quantile_preds[:, 4, :48]
            reference.append(value.reshape(b, c, 48).transpose(1, 2).cpu())
            budget.check()
    reference = torch.cat(reference)
    features = {}
    prefix_times = []
    sync()
    started = time.perf_counter()
    for ids in batch_ranges(len(sample)):
        before = time.perf_counter()
        budget.tick('prefix')
        budget.tick('head')
        values = cpu_features(output.features(inputs(data, sample[ids]).to('cuda')))
        for key, value in values.items():
            features.setdefault(key, []).append(value)
        sync()
        prefix_times.append((time.perf_counter()-before)/len(ids))
        budget.check()
    features = {key: torch.cat(value) for key, value in features.items()}
    report['timings']['prefix_generation_seconds'] = time.perf_counter()-started
    report['timings']['prefix_seconds_per_origin'] = prefix_times
    cache_path = CACHE / (dataset + '_sample_features.npz')
    started = time.perf_counter()
    np.savez(cache_path, origins=sample, **{key: value.numpy() for key, value in features.items()})
    report['cache']['write_seconds'] = time.perf_counter()-started
    started = time.perf_counter()
    with np.load(cache_path, allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], sample):
            raise AssertionError('Cache origin order differs')
        loaded = {key: torch.from_numpy(archive[key].copy()) for key in features}
    report['cache']['read_seconds'] = time.perf_counter()-started
    for key in features:
        close(loaded[key], features[key], 'cache stored:'+key, 0., 0.)
    features = loaded
    report['cache'].update(path=cache_path.relative_to(ROOT).as_posix(), bytes=cache_path.stat().st_size,
                           array_hashes={key: array_hash(value.numpy()) for key, value in features.items()})
    report['gates']['prefix_native_initial_full_val'] = close(features['point'][24:], reference, 'fullVAL original native F0')
    # Only stored VAL artifacts are read; TEST rows remain opaque metadata.
    parent = read_json(ROOT / 'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json')
    row = next(row for row in parent['units'][dataset]['rows'] if row['method'] == 'F0')
    if row.get('val_prediction'):
        entry = row['val_prediction']
        with np.load(ROOT / entry['path'], allow_pickle=False) as archive:
            if not np.array_equal(archive[entry['origins_key']], val):
                raise AssertionError('Stored parent VAL origin mismatch')
            report['gates']['stored_parent_val_replay'] = close(reference, torch.from_numpy(archive[entry['key']]), 'stored F0 VAL', 1e-6, 0.)
    else:
        report['gates']['stored_parent_val_replay'] = {'status': 'NOT_AVAILABLE', 'substituted': False}
    fixed_x = inputs(data, sample[:4])
    budget.tick('prefix')
    budget.tick('head')
    fresh_features = cpu_features(output.features(fixed_x.to('cuda')))
    report['gates']['fresh_online_features'] = {
        key: close(fresh_features[key], features[key][:4], 'fresh online feature:'+key)
        for key in features}
    with torch.no_grad():
        current = forward(output, fixed_x, budget).cpu()
        singleton = torch.cat([forward(output, fixed_x[i:i+1], budget).cpu() for i in range(4)])
        report['gates']['singleton_batch'] = close(singleton, current, 'singleton/batch')
        permutation = torch.tensor([3, 0, 2, 1])
        shuffled = forward(output, fixed_x[permutation], budget).cpu()
        report['gates']['batch_permutation'] = close(shuffled, current[permutation], 'batch permutation')
        changed = fixed_x.clone()
        changed[:, :, 0] += 7.
        changed_prediction = forward(output, changed, budget).cpu()
        report['gates']['f0_channel_independence'] = close(changed_prediction[:, :, 1:], current[:, :, 1:], 'F0 other channels')
        poisoned = dict(data)
        poisoned['x'] = data['x'].copy()
        o = int(sample[0])
        poisoned['x'][o:o+48] = 999.
        unchanged = forward(output, inputs(poisoned, sample[:1]), budget).cpu()
        report['gates']['future_target_perturbation'] = close(unchanged, singleton[:1], 'future target cannot enter prefix', 0., 0.)
    for arm in ('OUTPUT_AFFINE', 'NATIVE_HEAD'):
        backbone.output_patch_embedding.load_state_dict(native_initial)
        model = Readout(backbone, data['_C'], arm).eval()
        initial = model.trainable_state()
        before_hash = model.frozen_hash()
        allowlist = [(name, list(p.shape), p.numel()) for name, p in model.named_parameters() if p.requires_grad]
        expected_count = data['_C']*(data['_C']+1) if arm == 'OUTPUT_AFFINE' else 2526336
        if sum(row[2] for row in allowlist) != expected_count:
            raise AssertionError('Actual native/output parameter count changed')
        if arm == 'NATIVE_HEAD':
            expected_shapes = {'hidden_layer.weight': [2048, 512], 'hidden_layer.bias': [2048],
                               'output_layer.weight': [576, 2048], 'output_layer.bias': [576],
                               'residual_layer.weight': [576, 512], 'residual_layer.bias': [576]}
            if {name.split('output_patch_embedding.')[1]: shape for name, shape, _ in allowlist} != expected_shapes:
                raise AssertionError('Actual original native output block shapes changed')
        report['gates'][arm] = {'trainable': allowlist, 'trainable_parameters': sum(row[2] for row in allowlist),
                               'dropout_training': [name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout) and module.training]}
        if report['gates'][arm]['dropout_training']:
            raise AssertionError('Dropout active')
        with torch.no_grad():
            _, initial_prediction = full_val(model, data, val, features, 24, budget, 'cached')
        report['gates'][arm]['initial_full_val_F0'] = close(initial_prediction, reference, arm+' initial fullVAL')
        online_loss, online_grad, online_pred = backward(model, data, sample[:4], features, np.arange(4), budget, 'online')
        if any(not bool(torch.count_nonzero(value)) for value in online_grad.values()):
            raise AssertionError('A required trainable tensor has no nonzero gradient')
        cache_loss, cache_grad, cache_pred = backward(model, data, sample[:4], features, np.arange(4), budget, 'cached')
        report['gates'][arm]['online_cached_prediction'] = close(cache_pred, online_pred, arm+' cached prediction')
        if abs(online_loss-cache_loss) > 1e-6+1e-4*abs(online_loss):
            raise AssertionError('Online/cache loss mismatch')
        report['gates'][arm]['online_cached_loss'] = {'online': online_loss, 'cached': cache_loss}
        report['gates'][arm]['online_cached_gradients'] = compare_gradients(online_grad, cache_grad, arm+' cached gradients')
        _, missing = targets(data, sample[:4])
        missing = missing.clone()
        missing[0] = False
        missing[:, :, 0] = False
        missing[1:, :, 1] = False
        missing[1, 0, 1] = True
        full_loss, full_grad, _ = backward(model, data, sample[:4], features, np.arange(4), budget, 'cached', 4, missing)
        for micro in (2, 1):
            micro_loss, micro_grad, _ = backward(model, data, sample[:4], features, np.arange(4), budget, 'cached', micro, missing)
            if abs(micro_loss-full_loss) > 1e-6+1e-4*abs(full_loss):
                raise AssertionError('Accumulated masked loss mismatch')
            report['gates'][arm]['missing_micro_'+str(micro)] = compare_gradients(full_grad, micro_grad, arm+' missing micro'+str(micro))
        report['gates'][arm]['restore'] = restore_check(model, initial, data, sample, features, budget, dataset)
        for path in ('cached', 'online'):
            print(dataset+' '+arm+' '+path+' rates', flush=True)
            model.load_trainable_state(initial)
            model.zero_grad(set_to_none=True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
            measured = []
            torch.cuda.reset_peak_memory_stats()
            for update in range(30):
                ids = np.arange((update % 6)*4, (update % 6)*4+4)
                sync()
                start = time.perf_counter()
                step(model, optimizer, data, sample[ids], features, ids, budget, path)
                sync()
                elapsed = time.perf_counter()-start
                if update >= 10:
                    measured.append(elapsed)
            if all(torch.equal(initial[key], value) for key, value in model.trainable_state().items()):
                raise AssertionError('Disposable optimizer did not update trainable weights')
            adam_bytes = sum(value.numel()*value.element_size() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
            val_times = []
            for repeat in range(4):
                sync()
                start = time.perf_counter()
                full_val(model, data, val, features, 24, budget, path)
                sync()
                if repeat:
                    val_times.append(time.perf_counter()-start)
                budget.check()
            report['timings'][arm+'_'+path] = {'warmup_updates': 10, 'measured_updates': 20,
                'step_seconds': measured, 'median_step_seconds': float(np.median(measured)), 'maximum_step_seconds': max(measured),
                'val_seconds': val_times, 'median_full_val_seconds': float(np.median(val_times)),
                'maximum_full_val_seconds': max(val_times), 'adam_state_bytes': adam_bytes,
                'peak_allocated_bytes_after_adam': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes_after_adam': torch.cuda.max_memory_reserved(),
                'rate_scope': 'CPU data/cache batch indexing + H2D + forward/loss/backward/clip/Adam + synchronized return; includes actual budget ledger I/O',
                'cached_prefix_resident': True, 'deployment_memory_claim': False}
            model.load_trainable_state(initial)
            if model.frozen_hash() != before_hash:
                raise AssertionError('Frozen parameters/all buffers changed')
            del optimizer
        report['gates'][arm]['frozen_parameters_all_buffers'] = {'status': 'PASS', 'before_after_hash': before_hash}
        model.load_trainable_state(initial)
        model.zero_grad(set_to_none=True)
        del model
    backbone.output_patch_embedding.load_state_dict(native_initial)
    report['status'] = 'PASS'
    save_json(HERE / ('calibration_'+dataset+'.json'), report)
    del output, backbone, native_initial, data, features
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    configure()
    CACHE.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'tsfm_readout_calibration_v1', 'scientific_fits': 0, 'test_access': False,
              'selection': False, 'technical_pass_is_scientific_gain': False, 'datasets': {}}
    try:
        with Budget(stage='s0', category='gpu') as budget:
            verify_sources(budget)
            for dataset in ('robin', 'peacock_education', 'jena'):
                result['datasets'][dataset] = run_dataset(dataset, budget)
            result['budget'] = budget.snapshot()
            result['status'] = 'PASS'
    except Exception as error:
        result['status'] = 'STOP_RESOURCE' if 'RESOURCE' in str(error) else 'STOP_DEBUG'
        result['error'] = type(error).__name__+': '+str(error)
        result['traceback'] = traceback.format_exc()
        save_json(HERE / 'calibration_result.json', result)
        print(result['status']+' '+result['error'], flush=True)
        raise
    save_json(HERE / 'calibration_result.json', result)
    print('CALIBRATION PASS '+json.dumps(result['budget']), flush=True)


if __name__ == '__main__':
    main()
