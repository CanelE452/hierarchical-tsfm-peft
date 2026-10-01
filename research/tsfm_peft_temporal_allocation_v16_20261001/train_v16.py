"""The fixed v16 LoRA matrix and joint TRAIN/VAL selection before evaluation sealing."""
import argparse
import gc
import hashlib
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from model_v16 import build_model, make_checkpoint, restore_model, state_digest, tensor_hashes
from runtime_v16 import (HERE, ROOT, CACHE, DATASETS, SEEDS, Job, artifact,
                         budget_snapshot, configure, digest, environment_receipt, load_data,
                         masked_macro_loss, phase_sample, protocol, read_json, save_json,
                         score_arrays, tensors)

RUNS = HERE / 'runs'
LOCAL_RUNS = CACHE / 'runs'
REPLACEMENTS = HERE / 'replacements'
LRS = (1e-4, 1e-3)
BOUNDARY = 'post-VAL/post-scheduler/complete epoch'


def fit_specs():
    specs = protocol()['fits']
    expected = [(seed, dataset, lr) for seed in SEEDS for dataset in DATASETS for lr in LRS]
    actual = [(row['seed'], row['dataset'], row['lr']) for row in specs]
    if actual != expected or any(row['family'] != 'temporal_lora' for row in specs):
        raise ValueError('Protocol differs from the fixed ordered 16-fit matrix')
    return specs


def source_binding():
    names = ('train_v16.py', 'model_v16.py', 'runtime_v16.py', 'protocol.json',
             'provenance.json', 'data_manifest.json', 'reuse_manifest.json', 'initial_manifest.json')
    paths = [HERE / name for name in names]
    paths += [ROOT / 'src/hier_peft/lora.py',
              ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/model_v11.py',
              ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/runtime_v11.py',
              ROOT / 'research/tsfm_peft_group_basis_v13_20261001/cost_v13.py',
              ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001/model_confirmation_v12.py',
              ROOT / '.venv/Lib/site-packages/chronos/chronos_bolt.py',
              ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8/config.json']
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def assert_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def optimizer_for(model, spec):
    names = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if sum(value.numel() for value in names.values()) != 294912 or not all('lora_' in name for name in names):
        raise RuntimeError('Only the prescribed 294912 LoRA parameters may train')
    optimizer = torch.optim.AdamW(list(names.values()), lr=float(spec['lr']), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, threshold=1e-4, threshold_mode='rel')
    assert {id(p) for p in names.values()} == {id(p) for group in optimizer.param_groups for p in group['params']}
    return optimizer, scheduler


def frozen_state(model):
    names = {name for name, value in model.named_parameters() if not value.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value for name, value in model.state_dict().items() if name in names}


def gradient_record(model):
    return {name: {'numel': value.numel(), 'has_grad': value.grad is not None,
                   'finite': value.grad is not None and bool(torch.isfinite(value.grad).all()),
                   'max_abs': None if value.grad is None else float(value.grad.detach().abs().max()),
                   'nonzero': 0 if value.grad is None else int(torch.count_nonzero(value.grad))}
            for name, value in model.named_parameters() if value.requires_grad}


@torch.no_grad()
def evaluate(model, data, origins, job, label):
    model.eval()
    predictions = []
    for start in range(0, len(origins), 4):
        if start % 64 == 0:
            job.heartbeat({'evaluation': label, 'origins_done': start, 'origins_total': len(origins)})
        x, _, _ = tensors(data, origins[start:start + 4])
        predictions.append(model(x).cpu().numpy())
    prediction = np.concatenate(predictions)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite forecast during TRAIN/VAL evaluation')
    target = np.stack([data['x'][int(origin):int(origin) + 48] for origin in origins])
    mask = np.stack([data['finite'][int(origin):int(origin) + 48] for origin in origins])
    job.heartbeat(label)
    return score_arrays(prediction, target, mask), prediction


def result_for(spec, required=True):
    replacement = REPLACEMENTS / (spec['id'] + '.json')
    path = RUNS / spec['id'] / 'result.json'
    if replacement.exists():
        row = read_json(replacement)
        if row['canonical_id'] != spec['id']:
            raise RuntimeError('Replacement has another canonical fit')
        path = Path(artifact(row['result']['path'], row['result']['sha256'])['path'])
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    result = read_json(path)
    if result['status'] != 'complete' or result['spec'] != spec:
        raise RuntimeError('Saved result is not the canonical completed fit')
    jobs = read_json(HERE / 'ledger.json')['jobs']
    matching = [row for row in jobs if row['category'] == 'neural_fit' and row['label'] == result['id']]
    if len(matching) != 1 or matching[0]['status'] != 'complete' or matching[0]['metadata']['fit_id'] != spec['id']:
        raise RuntimeError('Result lacks a completed matching neural-fit ledger outcome')
    if matching[0]['updates'] != result['actual_updates']:
        raise RuntimeError('Fit update accounting differs from the result')
    artifact(result['checkpoint'], result['checkpoint_sha256'])
    for key in ('best_val_prediction', 'initial_checkpoint', 'last_checkpoint', 'data'):
        artifact(result[key]['path'], result[key]['sha256'])
    if result['initial_source'].get('path'):
        artifact(result['initial_source']['path'], result['initial_source']['sha256'])
    return result


def retry_contract(spec, tag, reason, resume_from):
    if not tag or not re.fullmatch(r'[A-Za-z0-9_-]+', tag) or not reason.strip():
        raise ValueError('A technical retry needs a unique safe tag and concrete reason')
    if any((HERE / name).exists() for name in ('selected.json', 'selection_seal.json', 'test_exposure.json')):
        raise RuntimeError('No retry after joint selection or new TEST exposure')
    jobs = [row for row in read_json(HERE / 'ledger.json')['jobs']
            if row['category'] == 'neural_fit' and row['metadata'].get('fit_id') == spec['id']]
    if not jobs or any(row['status'] != 'failed' for row in jobs):
        raise RuntimeError('Technical retry requires closed failed attempts, never a normal completed negative')
    original = RUNS / jobs[-1]['label'] / 'attempt.json'
    if read_json(original)['spec'] != spec:
        raise RuntimeError('Retry changes the canonical specification')
    attempt_id = spec['id'] + '__retry_' + tag
    if (RUNS / attempt_id).exists() or (LOCAL_RUNS / attempt_id).exists():
        raise RuntimeError('Technical retry tag already exists')
    retry = {'attempt_id': attempt_id, 'canonical_id': spec['id'], 'reason': reason,
             'original_attempt': artifact(original), 'technical_retry': True}
    if resume_from:
        path = Path(resume_from).resolve()
        if path.name != 'last.pt' or path.parent.parent != LOCAL_RUNS.resolve():
            raise ValueError('Resume must use this campaign\'s complete-epoch last.pt')
        source = read_json(RUNS / path.parent.name / 'attempt.json')
        if source['spec'] != spec or source['id'] not in {row['label'] for row in jobs}:
            raise RuntimeError('Resume checkpoint belongs to another fit')
        retry['resume_from'] = artifact(path)
    return retry


def fit(spec, gpu_job, retry=None):
    attempt_id = retry['attempt_id'] if retry else spec['id']
    out, local = RUNS / attempt_id, LOCAL_RUNS / attempt_id
    if out.exists() or local.exists():
        raise RuntimeError('Preserve existing attempt files; use a documented technical retry')
    metadata = {**spec, 'fit_id': spec['id']}
    if retry:
        metadata.update(technical_retry=True, reason=retry['reason'])
    with Job('neural_fit', attempt_id, reserve_s=0, metadata=metadata) as fit_job:
        started = time.perf_counter()
        updates = steps = carried = 0
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        attempt = {'id': attempt_id, 'canonical_id': spec['id'], 'spec': spec, 'status': 'running',
                   'job_id': fit_job.id, 'gpu_job_id': gpu_job.id, 'retry': retry,
                   'source': source_binding(), 'environment': environment_receipt()}
        save_json(out / 'attempt.json', attempt)
        try:
            data = load_data(spec['dataset'], include_test=False)
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            model = build_model(spec['dataset'], spec['seed'], device='cuda', adapt=True)
            initial = model.adapter_state()
            initial_parameters = {name: value.detach().cpu().clone()
                                  for name, value in model.named_parameters() if value.requires_grad}
            initial_hashes = tensor_hashes(initial)
            frozen_hash = state_digest(frozen_state(model))
            frozen_versions = {name: value._version for name, value in model.named_parameters() if not value.requires_grad}
            initial_source, parent = model.initial_receipt, model.parent_receipt
            save_checkpoint(local / 'initial.pt', make_checkpoint(model, spec, epoch=0, steps=0))
            period = data['_phase_period']
            probe = phase_sample(data['train_origins'], spec['seed'], 0, max(96, period), period=period)
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            with torch.no_grad():
                check_x, _, _ = tensors(data, probe[:4])
                enabled = model(check_x)
                with model.backbone.disable_adapter():
                    disabled = model(check_x)
                difference = (enabled - disabled).abs()
                parity = {'passed': bool(torch.all(difference <= 1e-5 + 1e-4 * disabled.abs())),
                          'max_absolute': float(difference.max()), 'atol': 1e-5, 'rtol': 1e-4,
                          'reference': 'identical temporal formula with adapter disabled; not DIRECT'}
                if not parity['passed']:
                    raise RuntimeError('Initial LoRA changes the prescribed TEMPORAL_F0 function')
                del check_x, enabled, disabled, difference
            torch.cuda.reset_peak_memory_stats()
            initial_train, _ = evaluate(model, data, probe, gpu_job, attempt_id + ' initial TRAIN probe')
            optimizer, scheduler = optimizer_for(model, spec)
            history, schedules, gradients = [], [], {}
            best, best_epoch, stale, start_epoch = float('inf'), 0, 0, 0
            best_saved, best_prediction = None, None
            if retry and retry.get('resume_from'):
                resume_path = Path(retry['resume_from']['path'])
                artifact(resume_path, retry['resume_from']['sha256'])
                resumed = torch.load(resume_path, map_location='cpu', weights_only=False)
                if resumed['spec'] != spec or resumed.get('boundary') != BOUNDARY:
                    raise RuntimeError('Resume requires this fit at a complete epoch boundary')
                if int(resumed['epoch']) >= 120 or int(resumed['stale']) >= 6:
                    raise RuntimeError('Completed stopping boundary cannot receive more updates')
                assert_equal(resumed['initial_receipt'], initial_source)
                assert_equal(resumed['parent_receipt'], parent)
                assert_equal(resumed['model_config'], model.model_config())
                if resumed['data_manifest_sha256'] != digest(HERE / 'data_manifest.json'):
                    raise RuntimeError('Resume data contract changed')
                model.restore_adapter(resumed['state'])
                assert state_digest(frozen_state(model)) == frozen_hash
                optimizer.load_state_dict(resumed['optimizer'])
                scheduler.load_state_dict(resumed['scheduler'])
                assert_equal(optimizer.state_dict(), resumed['optimizer'])
                assert_equal(scheduler.state_dict(), resumed['scheduler'])
                history, schedules, gradients = resumed['history'], resumed['schedules'], resumed['gradients']
                best, best_epoch, stale = resumed['best'], resumed['best_epoch'], resumed['stale']
                carried = steps = int(resumed['steps'])
                start_epoch = int(resumed['epoch']) + 1
                best_saved = resumed['selected_checkpoint']
                best_prediction = resumed['selected_prediction']
                if best_saved['spec'] != spec or best_saved['epoch'] != best_epoch:
                    raise RuntimeError('Embedded selected checkpoint differs from the complete epoch boundary')
                if not np.array_equal(resumed['selected_origins'], data['val_origins']):
                    raise RuntimeError('Embedded selected prediction origins changed')
                save_checkpoint(local / 'best.pt', best_saved)
                np.savez_compressed(local / 'best_val.npz', prediction=best_prediction, origins=data['val_origins'])
                with np.load(resume_path.parent / 'origins.npz', allow_pickle=False) as old_origins:
                    assert np.array_equal(probe, old_origins['train_probe'])
                    assert np.array_equal(data['val_origins'], old_origins['val'])
                frozen_versions = {name: value._version for name, value in model.named_parameters() if not value.requires_grad}
                restore_rng(resumed['rng'])
                assert_equal(rng_state(), resumed['rng'])
                save_json(out / 'resume_receipt.json', {'source': retry['resume_from'], 'start_epoch': start_epoch,
                    'carried_steps': carried, 'new_attempt_updates': 0, 'optimizer_scheduler_rng_restored': True,
                    'same_probe_and_val_origins': True, 'selected_restored_from_complete_boundary': True,
                    'max_total_epochs': 120})
                del resumed
            for epoch in range(start_epoch, 121):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512, period=period)
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if updates % 16 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}; batch {start // 4}', updates)
                        optimizer.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = masked_macro_loss(model(x), y, mask)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Nonfinite masked original-resolution loss')
                        loss.backward()
                        if updates < 2:
                            record = gradient_record(model)
                            if not all(row['finite'] for row in record.values()) or not any(row['nonzero'] for row in record.values()):
                                raise RuntimeError('Missing/nonfinite or entirely zero actual LoRA gradients')
                            gradients[str(steps + 1)] = record
                        torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad],
                                                      1, error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                        steps += 1
                        fit_job.updates = updates
                        losses.append(float(loss.detach()))
                    del x, y, mask, loss
                val, prediction = evaluate(model, data, data['val_origins'], gpu_job,
                                           attempt_id + f' VAL epoch {epoch}')
                history.append({'epoch': epoch, 'steps': steps, 'attempt_updates': updates, 'attempt_id': attempt_id,
                                'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    best_saved = make_checkpoint(model, spec, epoch=epoch, steps=steps)
                    best_prediction = prediction.copy()
                    save_checkpoint(local / 'best.pt', best_saved)
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                save_checkpoint(local / 'last.pt', make_checkpoint(model, spec, epoch=epoch, steps=steps,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                    best=best, best_epoch=best_epoch, stale=stale, history=history, schedules=schedules,
                    gradients=gradients, selected_files={name: artifact(local / name) for name in ('best.pt', 'best_val.npz')},
                    selected_checkpoint=best_saved, selected_prediction=best_prediction,
                    selected_origins=data['val_origins'], boundary=BOUNDARY))
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}; best {best_epoch}', updates)
                print(f'{attempt_id} epoch={epoch} best={best_epoch} val_mse={val["mse"]:.8g} elapsed={time.perf_counter()-started:.1f}s', flush=True)
                if epoch and stale >= 6:
                    break
            final_train, _ = evaluate(model, data, probe, gpu_job, attempt_id + ' final TRAIN probe')
            changes = {name: float((value.detach().cpu() - initial_parameters[name]).abs().max())
                       for name, value in model.named_parameters() if value.requires_grad}
            changed = {name: int(torch.count_nonzero(value.detach().cpu() != initial_parameters[name]))
                       for name, value in model.named_parameters() if value.requires_grad}
            if not any(changed.values()):
                raise RuntimeError('No LoRA tensor changed in the scheduled fit')
            assert all(value.grad is None and value._version == frozen_versions[name]
                       for name, value in model.named_parameters() if not value.requires_grad)
            assert state_digest(frozen_state(model)) == frozen_hash, 'Frozen source weights/buffers changed'
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'one online training model/optimizer, initial probe, TRAIN and VAL; before restore diagnostics'}
            trainable_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
            total_count = sum(value.numel() for value in model.parameters())
            last = torch.load(local / 'last.pt', map_location='cpu', weights_only=False)
            assert_equal(model.adapter_state(), last['state'])
            assert_equal(best_saved, last['selected_checkpoint'])
            assert_equal(best_prediction, last['selected_prediction'])
            assert_equal(data['val_origins'], last['selected_origins'])
            assert_equal(optimizer.state_dict(), last['optimizer'])
            assert_equal(scheduler.state_dict(), last['scheduler'])
            fresh_optimizer, fresh_scheduler = optimizer_for(model, spec)
            fresh_optimizer.load_state_dict(last['optimizer'])
            fresh_scheduler.load_state_dict(last['scheduler'])
            assert_equal(fresh_optimizer.state_dict(), optimizer.state_dict())
            assert_equal(fresh_scheduler.state_dict(), scheduler.state_dict())
            before_rng = rng_state()
            restore_rng(last['rng'])
            assert_equal(rng_state(), last['rng'])
            restore_rng(before_rng)
            del model, optimizer, scheduler, fresh_optimizer, fresh_scheduler, last
            gc.collect()
            torch.cuda.empty_cache()
            restored = restore_model(local / 'best.pt', device='cuda')
            replay_score, replay = evaluate(restored, data, data['val_origins'], gpu_job, attempt_id + ' selected replay')
            with np.load(local / 'best_val.npz', allow_pickle=False) as arrays:
                assert np.array_equal(arrays['origins'], data['val_origins'])
                replay_error = float(np.abs(replay - arrays['prediction']).max())
            if replay_error > 1e-6 or abs(replay_score['mse'] - best) > 1e-8:
                raise RuntimeError('Selected checkpoint same-path replay differs')
            assert state_digest(frozen_state(restored)) == frozen_hash
            selected_train, _ = evaluate(restored, data, probe, gpu_job, attempt_id + ' selected TRAIN probe')
            chosen = restored.adapter_state()
            selected_changes = {name: float((chosen[name] - initial[name]).abs().max()) for name in initial}
            if source_binding() != attempt['source']:
                raise RuntimeError('Bound fitting sources changed during the attempt')
            result = {**attempt, 'status': 'complete', 'checkpoint': str(local / 'best.pt'),
                'checkpoint_sha256': digest(local / 'best.pt'), 'best_val_prediction': artifact(local / 'best_val.npz'),
                'initial_checkpoint': artifact(local / 'initial.pt'), 'last_checkpoint': artifact(local / 'last.pt'),
                'initial_source': initial_source, 'initial_state_hashes': initial_hashes, 'parent': parent,
                'initial_f0_parity': parity, 'model_config': restored.model_config(),
                'data': artifact(data['_path']), 'best_val_mse': best, 'selected_epoch': best_epoch,
                'epochs_completed': epoch, 'actual_updates': updates, 'carried_steps': carried, 'cumulative_model_steps': steps,
                'upper_epoch_limit_improving': bool(epoch == 120 and epoch - best_epoch < 6),
                'first_update_gradients': gradients, 'changed_parameter_scalars': changed, 'parameter_max_updates': changes,
                'selected_parameter_max_updates': selected_changes, 'selected_nonstep0': best_epoch > 0,
                'selected_adapter_changed': any(value > 0 for value in selected_changes.values()),
                'frozen_parameters_verified': {'no_gradient': True, 'version_unchanged': True, 'optimizer_exclusion': True,
                                               'state_sha256': frozen_hash, 'restored_state_sha256': frozen_hash},
                'initial_train_probe': initial_train, 'selected_train_probe': selected_train, 'final_train_probe': final_train,
                'initial_val_mse': history[0]['val_mse'], 'train_probe_origins': probe.tolist(),
                'origin_receipt': artifact(local / 'origins.npz'), 'schedules': schedules,
                'trainable_parameters': trainable_count, 'total_parameters': total_count, 'training_peak': peak,
                'restart_roundtrip': {'model_optimizer_scheduler_rng': True, 'extra_updates': 0}, 'replay_error': replay_error,
                'elapsed_s': time.perf_counter() - started}
            save_json(out / 'result.json', result)
            if retry:
                save_json(REPLACEMENTS / (spec['id'] + '.json'), {'canonical_id': spec['id'],
                    'selected_attempt_id': attempt_id, 'result': artifact(out / 'result.json'), 'technical_retry': retry,
                    'selection_basis': 'valid completion after documented technical failure, never performance'})
            attempt.update(status='complete', actual_updates=updates, elapsed_s=result['elapsed_s'])
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('selected checkpoint saved and replayed', updates)
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            fit_job.updates = updates
            attempt.update(status='failed', actual_updates=updates, carried_steps=carried, cumulative_model_steps=steps,
                           elapsed_s=time.perf_counter() - started, error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            raise


def select():
    if (HERE / 'test_exposure.json').exists():
        raise RuntimeError('Cannot select after new TEST exposure')
    specs = fit_specs()
    rows = [result_for(spec) for spec in specs]
    selected, initial, schedules = {}, {}, {}
    for dataset in DATASETS:
        choices = []
        for lr in LRS:
            kept = sorted([row for row in rows if row['spec']['dataset'] == dataset and row['spec']['lr'] == lr],
                          key=lambda row: row['spec']['seed'])
            if [row['spec']['seed'] for row in kept] != list(SEEDS):
                raise RuntimeError('Both paired seeds must finish before LR selection')
            choices.append({'lr': lr, 'mean_best_val_mse': float(np.mean([row['best_val_mse'] for row in kept])),
                            'run_ids': [row['id'] for row in kept]})
        chosen = min(choices, key=lambda row: (row['mean_best_val_mse'], row['lr']))
        kept = sorted([row for row in rows if row['id'] in chosen['run_ids']], key=lambda row: row['spec']['seed'])
        selected[dataset] = {'temporal_lora': {'lr': chosen['lr'], 'seeds': list(SEEDS),
            'run_ids': [row['id'] for row in kept], 'checkpoints': [row['checkpoint'] for row in kept],
            'checkpoint_sha256': [row['checkpoint_sha256'] for row in kept],
            'selected_epochs': [row['selected_epoch'] for row in kept],
            'initial_checkpoints': [row['initial_checkpoint'] for row in kept],
            'best_val_prediction': [row['best_val_prediction'] for row in kept],
            'mean_best_val_mse': chosen['mean_best_val_mse'], 'lr_candidates': choices,
            'selected_nonstep0': [row['selected_nonstep0'] for row in kept],
            'selected_adapter_changed': [row['selected_adapter_changed'] for row in kept]}}
    for row in rows:
        spec = row['spec']
        key = (spec['dataset'], spec['seed'])
        reference = {'tensor_hashes': row['initial_state_hashes'], 'source': row['initial_source'], 'parent': row['parent']}
        if key in initial:
            assert_equal(initial[key], reference)
        initial[key] = reference
        for schedule in row['schedules']:
            key = (spec['dataset'], spec['seed'], schedule['epoch'])
            if key in schedules and schedules[key] != schedule['sha256']:
                raise RuntimeError('Matched LRs used different TRAIN schedules')
            schedules[key] = schedule['sha256']
    output = {'schema': 'v16_selection_1', 'status': 'complete', 'fit_count': len(rows),
              'all_datasets_complete': True, 'selected': selected, 'test_used_for_selection': False,
              'criterion': 'minimum full VAL MSE including epoch0; earliest epoch; common two-seed mean LR; exact tie 1e-4',
              'all_neural_runs': [{key: row[key] for key in ('id', 'canonical_id', 'spec', 'checkpoint',
                                  'checkpoint_sha256', 'best_val_mse', 'selected_epoch', 'actual_updates')} for row in rows],
              'matched_initial_groups': len(initial), 'matched_schedule_groups': len(schedules),
              'protocol': artifact(HERE / 'protocol.json')}
    target = HERE / 'selected.json'
    if target.exists():
        assert_equal(read_json(target), output)
    else:
        save_json(target, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--reserve-s', '--reserve-seconds', dest='reserve_s', type=float, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--all', action='store_true')
    scope.add_argument('--fit')
    parser.add_argument('--retry-tag')
    parser.add_argument('--retry-reason', default='')
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    if any((HERE / name).exists() for name in ('selection_seal.json', 'test_exposure.json')):
        raise RuntimeError('Fitting is closed after selection seal or new TEST exposure')
    specs = fit_specs()
    chosen = specs if args.all else [spec for spec in specs if spec['id'] == args.fit]
    if not chosen:
        parser.error('--fit is outside the prescribed matrix')
    retry = None
    if args.retry_tag:
        if args.all:
            parser.error('Technical retry must name one canonical --fit')
        retry = retry_contract(chosen[0], args.retry_tag, args.retry_reason, args.resume_from)
    elif args.retry_reason or args.resume_from:
        parser.error('Retry metadata requires --retry-tag')
    with Job('gpu', args.label, reserve_s=args.reserve_s,
             metadata={'stage': 'training', 'fit_ids': [spec['id'] for spec in chosen]}) as gpu_job:
        configure()
        for spec in chosen:
            if retry is None and result_for(spec, required=False) is not None:
                gpu_job.heartbeat('reusing verified completed fit ' + spec['id'])
                continue
            fit(spec, gpu_job, retry)
            gpu_job.heartbeat({'completed_fit': spec['id'], 'budget': budget_snapshot()})
        if all(result_for(spec, required=False) is not None for spec in specs):
            select()
            gpu_job.heartbeat('all 16 canonical fits selected; separate CPU evaluation seal pending')
        else:
            gpu_job.heartbeat('completed requested fits; joint selection remains pending')


if __name__ == '__main__':
    main()
