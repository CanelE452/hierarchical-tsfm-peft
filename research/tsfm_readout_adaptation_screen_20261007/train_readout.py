"""Fixed cached S1 fits and joint TRAIN/VAL choices; no TEST reads."""
import gc
import math
import os
import random
import time
import traceback

import numpy as np
import torch

from model_readout import ARMS, Readout, compute_loss, load_backbone
from prepare_readout import ensure_disk_space, load_cache, receipt, require_s1, seal_sources, verify_receipt
from runtime_readout import (CACHE, DATASETS, HERE, ROOT, SEEDS, array_hash, batches,
                             digest, load_data, read_json, resolve, sample_epoch,
                             save_json, targets)


LEARNING_RATES = (1e-3, 1e-4)
MAX_EPOCHS = 120
PATIENCE = 6


def run_spec(dataset, arm, lr, seed):
    tag = {1e-3: '1em3', 1e-4: '1em4'}[lr]
    return {'id': 's1_' + dataset + '_' + arm.lower() + '_lr' + tag + '_seed' + str(seed),
            'dataset': dataset, 'arm': arm, 'learning_rate': lr, 'seed': seed,
            'maximum_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'epoch_origins': 512,
            'effective_batch': 4, 'optimizer': 'AdamW', 'weight_decay': 0., 'gradient_clip': 1.,
            'training_path': 'cached', 'frozen_prefix_kept_on_gpu': True}


def optimizer_for(model, lr):
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, threshold=1e-4, threshold_mode='rel')
    if {id(value) for value in parameters} != {id(value) for group in optimizer.param_groups for value in group['params']}:
        raise RuntimeError('Optimizer parameter allowlist differs')
    return optimizer, scheduler


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def _atomic_torch_save(path, value, budget):
    pending = [value]
    needed = 1024 * 1024
    while pending:
        item = pending.pop()
        if torch.is_tensor(item):
            needed += item.numel() * item.element_size()
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    ensure_disk_space(budget, needed)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)
    budget.check()


def _atomic_prediction(path, prediction, origins, budget):
    ensure_disk_space(budget, prediction.nbytes + origins.nbytes + 4096)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, prediction=prediction, origins=origins)
    os.replace(temporary, path)
    budget.check()


def feature_batch(features, indices, arm, device='cuda'):
    keys = ('point',) if arm == 'OUTPUT_AFFINE' else ('h', 'loc', 'scale')
    return {key: torch.from_numpy(np.ascontiguousarray(features[key][indices])).to(device)
            for key in keys}


def _indices(origin_index, origins):
    return np.asarray([origin_index[int(origin)] for origin in origins], dtype=np.int64)


@torch.no_grad()
def evaluate_cached(model, data, origins, features, origin_index, budget):
    model.eval()
    predictions = []
    sums = np.zeros(data['_C'], dtype=np.float64)
    absolute = np.zeros_like(sums)
    counts = np.zeros(data['_C'], dtype=np.int64)
    for batch_origins in batches(origins, 4):
        budget.tick('head')
        prediction = model.from_features(feature_batch(features, _indices(origin_index, batch_origins), model.arm)).cpu().numpy()
        target, observed = targets(data, batch_origins)
        target, observed = target.numpy(), observed.numpy()
        if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
            raise RuntimeError('Nonfinite forecast/observed target during full VAL')
        error = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.)
        sums += np.square(error).sum(axis=(0, 1))
        absolute += np.abs(error).sum(axis=(0, 1))
        counts += observed.sum(axis=(0, 1))
        predictions.append(prediction)
        budget.check()
    if not bool((counts > 0).all()):
        raise RuntimeError('Every fixed VAL channel requires an observed target')
    result = {'mse': float(np.mean(sums / counts)), 'mae': float(np.mean(absolute / counts)),
              'channel_mse': (sums / counts).tolist(), 'channel_mae': (absolute / counts).tolist(),
              'squared_error_sums': sums.tolist(), 'absolute_error_sums': absolute.tolist(),
              'target_counts': counts.tolist(), 'origins': len(origins), 'loss_coordinates': 'canonical TRAIN-standardized original channels'}
    return result, np.concatenate(predictions)


def _backward_batch(model, optimizer, data, batch_origins, features, origin_index, budget, microbatch, oom_events):
    target, observed = targets(data, batch_origins)
    target, observed = target.to('cuda'), observed.to('cuda')
    channel_count = observed.sum((0, 1))
    indices = _indices(origin_index, batch_origins)
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        try:
            for start in range(0, len(batch_origins), microbatch):
                stop = start + microbatch
                budget.tick('head')
                prediction = model.from_features(feature_batch(features, indices[start:stop], model.arm))
                loss = compute_loss(prediction, target[start:stop], observed[start:stop], channel_count)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                loss_sum += float(loss.detach())
            parameters = [value for value in model.parameters() if value.requires_grad]
            if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in parameters):
                raise RuntimeError('Missing/nonfinite readout parameter gradient')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            return loss_sum, float(grad_norm), microbatch
        except torch.cuda.OutOfMemoryError:
            oom_events.append({'effective_batch': len(batch_origins), 'failed_microbatch': microbatch,
                               'next_microbatch': microbatch // 2 if microbatch > 1 else None,
                               'utc': time.time(), 'extra_optimizer_updates': 0})
            optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch //= 2
            gc.collect()
            torch.cuda.empty_cache()


def _gradients(model):
    return {name: {'finite': bool(torch.isfinite(value.grad).all()),
                   'nonzero_scalars': int(torch.count_nonzero(value.grad)),
                   'max_abs': float(value.grad.abs().max())}
            for name, value in model.named_parameters() if value.requires_grad}


def _binding(spec):
    return {'spec': spec, 'source_seal_sha256': digest(HERE / 'execution_source_seal.json'),
            'protocol_sha256': digest(HERE / 'protocol.json'),
            'cache_manifest_sha256': digest(HERE / 'cache_manifest.json'), 'test_access': False}


def _fit(spec, data, features, origin_index, budget):
    out = HERE / 'runs' / spec['id']
    local = CACHE / 'runs' / spec['id']
    binding = _binding(spec)
    result_path = out / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result.get('status') == 'complete':
            if result['binding'] != binding:
                raise RuntimeError('Completed fit binding changed: ' + spec['id'])
            verify_receipt(result['checkpoint'])
            verify_receipt(result['best_val_prediction'])
            return result
    if (HERE / 'selection_seal.json').exists():
        raise RuntimeError('Cannot fit after the joint choice seal')
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    attempt_path = out / 'attempt.json'
    resume_path = local / 'resume.pt'
    prior = read_json(attempt_path) if attempt_path.exists() else None
    if prior and prior['binding'] != binding:
        raise RuntimeError('Attempt binding changed; preserve and inspect: ' + spec['id'])
    resume_exists = resume_path.exists()
    if prior and not resume_exists and not (prior.get('phase') == 'prefix_restore' and prior.get('actual_updates') == 0):
        raise RuntimeError('Existing incomplete attempt has no exact resume; preserve and inspect: ' + spec['id'])
    if prior and prior.get('optimizer_step_interrupted', False):
        raise RuntimeError('Interrupted optimizer step requires explicit technical inspection')
    if prior is None:
        budget.tick('scientific_fit')
    started = time.perf_counter()
    previous_elapsed = float(prior.get('cumulative_elapsed_s', 0.)) if prior else 0.
    prior_attempts = prior.get('previous_attempts', []) + [{key: value for key, value in prior.items() if key != 'previous_attempts'}] if prior else []
    attempt = {'id': spec['id'], 'binding': binding, 'status': 'running', 'pid': os.getpid(),
               'started_utc': time.time(), 'previous_attempts': prior_attempts, 'phase': 'prefix_restore', 'actual_updates': 0}
    save_json(attempt_path, attempt)
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    random.seed(spec['seed'])
    try:
        backbone = load_backbone(spec['dataset'], device='cuda')
        model = Readout(backbone, data['_C'], spec['arm']).eval()
        initial = model.trainable_state()
        frozen_hash = model.frozen_hash()
        frozen_versions = {name: value._version for name, value in model.named_parameters() if not value.requires_grad}
        optimizer, scheduler = optimizer_for(model, spec['learning_rate'])
    except BaseException as error:
        attempt.update(status='STOP_RESOURCE' if 'RESOURCE' in str(error) else 'STOP_DEBUG',
                       cumulative_elapsed_s=previous_elapsed + time.perf_counter() - started, error=traceback.format_exc())
        save_json(attempt_path, attempt)
        raise
    steps, epoch, next_batch, stale = 0, 0, 0, 0
    best, best_epoch = float('inf'), None
    history, schedules, losses, grad_norms, oom_events = [], [], [], [], []
    microbatch, phase, first_gradients, adam_memory = 4, 'initial', None, None
    optimizer_in_progress = False
    best_choice_path = out / 'best_choice.json'

    def restart_payload():
        return {'schema': 'tsfm_readout_s1_resume_v1', 'binding': binding,
                'state': model.trainable_state(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'steps': steps,
                'epoch': epoch, 'next_batch': next_batch, 'phase': phase,
                'stale': stale, 'best': None if not math.isfinite(best) else best, 'best_epoch': best_epoch,
                'history': history, 'schedules': schedules, 'partial_epoch_losses': losses,
                'partial_epoch_grad_norms': grad_norms, 'microbatch': microbatch, 'oom_events': oom_events,
                'frozen_hash': frozen_hash, 'first_gradients': first_gradients, 'adam_memory': adam_memory,
                'best_choice_sha256': digest(best_choice_path) if best_choice_path.exists() else None}

    def finish_epoch(score, prediction, train_seconds, val_seconds):
        nonlocal best, best_epoch, stale, phase, next_batch, losses, grad_norms
        io_started = time.perf_counter()
        if score['mse'] < best:
            previous_choice = read_json(best_choice_path) if best_choice_path.exists() else None
            slot = 1 - previous_choice['slot'] if previous_choice else 0
            best_path = local / ('best_' + str(slot) + '.pt')
            val_path = local / ('best_val_' + str(slot) + '.npz')
            _atomic_torch_save(best_path, {'schema': 'tsfm_readout_s1_selected_v1', 'spec': spec,
                                         'binding': binding, 'state': model.trainable_state(), 'epoch': epoch,
                                         'steps': steps, 'val_mse': score['mse'], 'frozen_hash': frozen_hash}, budget)
            _atomic_prediction(val_path, prediction, data['val_origins'], budget)
            choice = {'slot': slot, 'epoch': epoch, 'steps': steps, 'mse': score['mse'],
                      'checkpoint': receipt(best_path), 'prediction': receipt(val_path)}
            save_json(best_choice_path, choice)
            best, best_epoch, stale = score['mse'], epoch, 0
        else:
            stale += 1
        lr_used = optimizer.param_groups[0]['lr']
        if epoch >= 1:
            scheduler.step(score['mse'])
        history.append({'epoch': epoch, 'steps': steps,
                        'sampled_train_effective_batch_loss_mean': float(np.mean(losses)) if losses else None,
                        'sampled_train_effective_batch_loss_max': float(np.max(losses)) if losses else None,
                        'gradient_norm_mean_before_clip': float(np.mean(grad_norms)) if grad_norms else None,
                        'gradient_norm_max_before_clip': float(np.max(grad_norms)) if grad_norms else None,
                        'val_mse': score['mse'], 'val_mae': score['mae'], 'best_epoch': best_epoch,
                        'learning_rate_used': lr_used, 'learning_rate_next_epoch': optimizer.param_groups[0]['lr'],
                        'scheduler_observed': epoch >= 1, 'microbatch': microbatch,
                        'cached_train_seconds': train_seconds, 'full_val_seconds': val_seconds,
                        'checkpoint_selection_io_seconds': time.perf_counter() - io_started,
                        'elapsed_s': previous_elapsed + time.perf_counter() - started})
        phase, next_batch, losses, grad_norms = 'post_val_epoch', 0, [], []
        io_started = time.perf_counter()
        _atomic_torch_save(resume_path, restart_payload(), budget)
        history[-1]['resume_checkpoint_io_seconds'] = time.perf_counter() - io_started
        save_json(out / 'curve.json', history)
        print('FIT ' + spec['id'] + ' epoch=' + str(epoch) + ' best=' + str(best_epoch) + ' val=' + str(score['mse']), flush=True)

    try:
        budget.check(force=True)
        torch.cuda.reset_peak_memory_stats()
        if prior and resume_exists:
            saved = torch.load(resume_path, map_location='cpu', weights_only=False)
            if saved['binding'] != binding or saved['frozen_hash'] != frozen_hash:
                raise RuntimeError('Exact resume binding/frozen prefix differs')
            if saved['best_choice_sha256'] != (digest(best_choice_path) if best_choice_path.exists() else None):
                raise RuntimeError('Exact resume choice pointer differs')
            model.load_trainable_state(saved['state'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'])
            steps, epoch, next_batch, stale = saved['steps'], saved['epoch'], saved['next_batch'], saved['stale']
            best, best_epoch = saved['best'] if saved['best'] is not None else float('inf'), saved['best_epoch']
            history, schedules = saved['history'], saved['schedules']
            losses, grad_norms = saved['partial_epoch_losses'], saved['partial_epoch_grad_norms']
            microbatch, phase, oom_events = saved['microbatch'], saved['phase'], saved['oom_events']
            first_gradients, adam_memory = saved['first_gradients'], saved['adam_memory']
            del saved
        else:
            _atomic_torch_save(resume_path, restart_payload(), budget)
        if phase == 'initial':
            before = time.perf_counter()
            score, prediction = evaluate_cached(model, data, data['val_origins'], features, origin_index, budget)
            finish_epoch(score, prediction, 0., time.perf_counter() - before)
        first_epoch = epoch if phase == 'train_in_progress' else epoch + 1
        if stale < PATIENCE:
            for current_epoch in range(first_epoch, MAX_EPOCHS + 1):
                if current_epoch != epoch:
                    epoch, next_batch, losses, grad_norms = current_epoch, 0, [], []
                phase = 'train_in_progress'
                schedule = sample_epoch(data, spec['seed'], epoch)
                schedule_hash = array_hash(schedule)
                sealed_schedule = np.load(CACHE / 's1_features' / spec['dataset'] / ('schedule_' + str(spec['seed']) + '.npy'), mmap_mode='r', allow_pickle=False)[epoch - 1]
                if not np.array_equal(schedule, sealed_schedule):
                    raise RuntimeError('Epoch sample differs from fixed pre-generated schedule')
                if not any(row['epoch'] == epoch for row in schedules):
                    schedules.append({'epoch': epoch, 'sha256': schedule_hash})
                elif next(row for row in schedules if row['epoch'] == epoch)['sha256'] != schedule_hash:
                    raise RuntimeError('Resumed epoch sample hash differs')
                training_started = time.perf_counter()
                model.train()
                for start in range(next_batch * 4, len(schedule), 4):
                    budget.check()
                    loss, grad_norm, microbatch = _backward_batch(model, optimizer, data, schedule[start:start + 4],
                                                                features, origin_index, budget, microbatch, oom_events)
                    if first_gradients is None:
                        if grad_norm <= 0:
                            raise RuntimeError('The initial readout update has zero gradient')
                        first_gradients = _gradients(model)
                    budget.tick('scientific_update')
                    optimizer_in_progress = True
                    optimizer.step()
                    optimizer_in_progress = False
                    steps += 1
                    next_batch = start // 4 + 1
                    losses.append(loss)
                    grad_norms.append(grad_norm)
                    if adam_memory is None:
                        torch.cuda.synchronize()
                        adam_memory = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                                       'optimizer_tensor_bytes': sum(value.numel() * value.element_size()
                                       for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)),
                                       'after_successful_optimizer_update': steps, 'frozen_prefix_kept_on_gpu': True}
                    if steps % 32 == 0:
                        budget.check()
                torch.cuda.synchronize()
                train_seconds = time.perf_counter() - training_started
                before = time.perf_counter()
                score, prediction = evaluate_cached(model, data, data['val_origins'], features, origin_index, budget)
                finish_epoch(score, prediction, train_seconds, time.perf_counter() - before)
                if stale >= PATIENCE:
                    break
        if model.frozen_hash() != frozen_hash or any(value.grad is not None or value._version != frozen_versions[name]
                                                    for name, value in model.named_parameters() if not value.requires_grad):
            raise RuntimeError('Frozen parameters or all registered buffers changed')
        final = model.trainable_state()
        changes = {name: float((value - initial[name]).abs().max()) for name, value in final.items()}
        if steps > 0 and not any(value > 0 for value in changes.values()):
            raise RuntimeError('Readout weights did not change after completed updates')
        choice = read_json(best_choice_path)
        best_path = verify_receipt(choice['checkpoint'])
        verify_receipt(choice['prediction'])
        saved_best = torch.load(best_path, map_location='cpu', weights_only=False)
        phase = 'selected_replay'
        model.load_trainable_state(saved_best['state'])
        replay_start = time.perf_counter()
        replay_score, replay = evaluate_cached(model, data, data['val_origins'], features, origin_index, budget)
        with np.load(resolve(choice['prediction']['path']), allow_pickle=False) as archive:
            if not np.array_equal(archive['origins'], data['val_origins']):
                raise RuntimeError('Selected VAL replay origin mismatch')
            replay_error = float(np.abs(replay - archive['prediction']).max())
        if replay_error > 1e-6 or abs(replay_score['mse'] - best) > 1e-8:
            raise RuntimeError('Selected checkpoint replay differs')
        result = {'status': 'complete', 'id': spec['id'], 'spec': spec, 'binding': binding,
                  'checkpoint': choice['checkpoint'], 'best_val_prediction': choice['prediction'],
                  'best_val_mse': best, 'selected_epoch': best_epoch, 'initial_val_mse': history[0]['val_mse'],
                  'epochs_completed': epoch, 'scientific_updates': steps, 'final_stale': stale,
                  'normality': 'NORMAL_EPOCH0_SELECTED' if best_epoch == 0 else 'NORMAL_VAL_IMPROVEMENT',
                  'termination': 'PATIENCE' if stale >= PATIENCE else 'MAX_EPOCH',
                  'trainable_scalars': sum(value.numel() for value in model.parameters() if value.requires_grad),
                  'parameter_max_updates': changes, 'first_gradients': first_gradients, 'adam_memory': adam_memory,
                  'frozen_parameters_and_all_buffers_hash': frozen_hash, 'frozen_unchanged': True,
                  'training_peak': {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                                    'frozen_prefix_kept_on_gpu': True}, 'oom_events': oom_events,
                  'resume_checkpoint': receipt(resume_path), 'best_choice': receipt(best_choice_path),
                  'schedules': schedules, 'curve': receipt(out / 'curve.json'),
                  'selected_replay': {'maximum_absolute_error': replay_error, 'seconds': time.perf_counter() - replay_start},
                  'elapsed_s': previous_elapsed + time.perf_counter() - started,
                  'cost_scope': 'prefix restore/hash, cache reads, all TRAIN and full VAL, selection/checkpoint I/O and selected VAL replay',
                  'test_access': False, 'seed_aggregation': 'mean seed losses; no forecast ensemble'}
        save_json(result_path, result)
        attempt.update(status='complete', cumulative_elapsed_s=result['elapsed_s'], actual_updates=steps)
        save_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update(status='STOP_RESOURCE' if 'RESOURCE' in str(error) else 'STOP_DEBUG',
                       cumulative_elapsed_s=previous_elapsed + time.perf_counter() - started,
                       actual_updates=steps, epoch=epoch, next_batch=next_batch,
                       optimizer_step_interrupted=optimizer_in_progress, error=traceback.format_exc())
        if phase == 'selected_replay':
            attempt['exact_resume_saved'] = resume_path.exists()
            attempt['resume_preserved_without_selected_state_write'] = True
        elif not optimizer_in_progress:
            try:
                _atomic_torch_save(resume_path, restart_payload(), budget)
                attempt['exact_resume_saved'] = True
            except Exception as resume_error:
                attempt['exact_resume_saved'] = False
                attempt['resume_error'] = str(resume_error)
        save_json(attempt_path, attempt)
        raise
    finally:
        del model, backbone, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def train_all(budget):
    require_s1(budget)
    seal_sources()
    manifest = read_json(HERE / 'cache_manifest.json')
    if manifest.get('status') != 'complete' or set(manifest['datasets']) != set(DATASETS):
        raise RuntimeError('All fixed caches must complete before scientific fits')
    results = []
    for dataset in DATASETS:
        data = load_data(dataset, include_test=False)
        _, features, origin_index = load_cache(dataset)
        for arm in ARMS:
            for lr in LEARNING_RATES:
                for seed in SEEDS:
                    results.append(_fit(run_spec(dataset, arm, lr, seed), data, features, origin_index, budget))
        del data, features, origin_index
        gc.collect()
    save_json(HERE / 'fit_summary.json', {'status': 'complete', 'scientific_fits': len(results),
              'scientific_updates': sum(row['scientific_updates'] for row in results), 'test_access': False,
              'results': [{'id': row['id'], 'selected_epoch': row['selected_epoch'], 'best_val_mse': row['best_val_mse'],
                           'scientific_updates': row['scientific_updates']} for row in results]})
    return results


def select_all():
    require_s1()
    seal_sources()
    started = time.perf_counter()
    seal_path = HERE / 'selection_seal.json'
    if seal_path.exists():
        seal = read_json(seal_path)
        for entry in seal['artifacts']:
            verify_receipt(entry)
        return read_json(HERE / 'selection.json')
    selections, artifacts = {}, []
    for dataset in DATASETS:
        selections[dataset] = {}
        for arm in ARMS:
            candidates = []
            for lr in LEARNING_RATES:
                runs = []
                for seed in SEEDS:
                    spec = run_spec(dataset, arm, lr, seed)
                    path = HERE / 'runs' / spec['id'] / 'result.json'
                    result = read_json(path)
                    if result.get('status') != 'complete' or result['spec'] != spec or result['binding'] != _binding(spec):
                        raise RuntimeError('Selection requires every exact completed fixed fit')
                    artifacts.extend([receipt(path), result['checkpoint'], result['best_val_prediction'], result['curve']])
                    runs.append({'id': result['id'], 'seed': seed, 'learning_rate': lr, 'selected_epoch': result['selected_epoch'],
                                 'best_val_mse': result['best_val_mse'], 'checkpoint': result['checkpoint'],
                                 'best_val_prediction': result['best_val_prediction'], 'result': receipt(path)})
                candidates.append({'learning_rate': lr, 'mean_seed_val_mse': float(np.mean([run['best_val_mse'] for run in runs])),
                                   'runs': runs})
            chosen = min(range(len(candidates)), key=lambda index: candidates[index]['mean_seed_val_mse'])
            selections[dataset][arm] = {'lr': candidates[chosen]['learning_rate'],
                                        'mean_seed_val_mse': candidates[chosen]['mean_seed_val_mse'],
                                        'runs': candidates[chosen]['runs'], 'candidates': candidates,
                                        'tie_rule': 'first-listed 1e-3', 'seed_aggregation': 'loss mean'}
    protocol_names = ('protocol.json', 'execution_source_seal.json', 'cache_origins_seal.json',
                      'cache_manifest.json', 'reuse_manifest.json', 'evaluation_protocol.json', 'cost_protocol.json', 's2_pretest_recommendation.json',
                      'evaluate_readout.py', 'cost_readout.py')
    for name in protocol_names:
        if not (HERE / name).exists():
            raise RuntimeError('Joint choice seal requires pre-TEST protocol: ' + name)
        artifacts.append(receipt(HERE / name))
    selection = {'schema': 'tsfm_readout_s1_selection_v1', 'choices': selections,
                 'all_trainval_selection_complete_before_test': True, 'selected_models': 12,
                 'fits_considered': 24, 'selection_seconds': time.perf_counter() - started,
                 'primary_metric': 'full VAL masked channel-macro MSE', 'seed_prediction_ensemble': False,
                 'all_learning_rate_and_checkpoint_choices_frozen': True}
    save_json(HERE / 'selection.json', selection)
    artifacts.append(receipt(HERE / 'selection.json'))
    for entry in artifacts:
        verify_receipt(entry)
    save_json(seal_path, {'schema': 'tsfm_readout_s1_selection_seal_v1', 'sealed_utc': time.time(),
                         'all_trainval_selection_complete_before_test': True,
                         'global_choice_sealed_before_test': True, 'selected_models': 12,
                         'choices': selections, 'artifacts': artifacts, 'test_predictions_at_sealing': 0})
    return selection
