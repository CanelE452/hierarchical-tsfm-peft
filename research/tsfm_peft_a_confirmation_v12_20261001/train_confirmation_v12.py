"""Prospective v12 fitting, with all selection restricted to TRAIN/VAL."""
import argparse
import gc
import hashlib
import random
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from model_confirmation_v12 import (LORA_FAMILIES, build_model, make_checkpoint,
                                    restore_model, state_digest, tensor_hashes)
from runtime_confirmation_v12 import (HERE, CACHE, SEEDS, Job, artifact, budget_snapshot,
                                      configure, digest, environment_receipt, load_data,
                                      masked_macro_loss, phase_sample, protocol, read_json,
                                      save_json, score_arrays, sources_for, tensors)

RUNS = HERE / 'confirmation_runs'
LOCAL_RUNS = CACHE / 'confirmation_runs'
REPLACEMENTS = HERE / 'confirmation_replacements'


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


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
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('Fit has no trainable parameters')
    optimizer = torch.optim.AdamW(parameters, lr=float(spec['lr']), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, threshold=1e-4, threshold_mode='rel')
    assert {id(p) for p in parameters} == {id(p) for group in optimizer.param_groups for p in group['params']}
    return optimizer, scheduler


def frozen_state(model):
    names = {name for name, p in model.named_parameters() if not p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value for name, value in model.state_dict().items() if name in names}


def gradient_record(model):
    return {name: {'numel': p.numel(), 'has_grad': p.grad is not None,
                   'max_abs': None if p.grad is None else float(p.grad.detach().abs().max()),
                   'nonzero': 0 if p.grad is None else int(torch.count_nonzero(p.grad))}
            for name, p in model.named_parameters() if p.requires_grad}


def initial_reference(model, data):
    family, seed, dataset = model.family, model.seed, model.dataset
    kind = 'lora' if family in LORA_FAMILIES else family
    state = model.adapter_state()
    if kind == 'lora':
        state = {name: value for name, value in state.items() if 'lora_' in name}
    elif kind == 'direct':
        state = {name: value for name, value in state.items() if name.startswith('temporal.')}
    path = CACHE / 'confirmation_initial' / dataset / f'{kind}_{seed}.pt'
    record_path = HERE / 'confirmation_initial' / dataset / f'{kind}_{seed}.json'
    if path.exists():
        receipt = read_json(record_path)
        artifact(path, receipt['sha256'])
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert_equal(saved['state'], state)
        assert saved['data_sha256'] == digest(data['_path'])
    else:
        if record_path.exists():
            raise RuntimeError('Initial state receipt exists without its checkpoint')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state': state, 'dataset': dataset, 'seed': seed, 'kind': kind,
                    'data_sha256': digest(data['_path'])}, path)
        receipt = {**artifact(path), 'kind': kind, 'tensor_hashes': tensor_hashes(state),
                   'data_sha256': digest(data['_path'])}
        save_json(record_path, receipt)
    if receipt['tensor_hashes'] != tensor_hashes(state):
        raise RuntimeError('Initial tensor hashes differ across matched fits')
    return receipt


@torch.no_grad()
def evaluate(model, data, origins, gpu_job, label):
    model.eval()
    predictions = []
    for start in range(0, len(origins), 4):
        gpu_job.check_limits()
        x, _, _ = tensors(data, origins[start:start + 4])
        predictions.append(model(x).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.stack([data['x'][int(o):int(o) + 48] for o in origins])
    mask = np.stack([data['finite'][int(o):int(o) + 48] for o in origins])
    gpu_job.heartbeat(label)
    return score_arrays(prediction, target, mask), prediction


def result_for(spec, required=True):
    replacement = REPLACEMENTS / (spec['id'] + '.json')
    path = RUNS / spec['id'] / 'result.json'
    if replacement.exists():
        row = read_json(replacement)
        if row['canonical_id'] != spec['id']:
            raise RuntimeError('Replacement canonical ID differs')
        artifact(row['result']['path'], row['result']['sha256'])
        path = Path(row['result']['path'])
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    result = read_json(path)
    if result['status'] != 'complete' or result['spec'] != spec:
        raise RuntimeError('Result does not match the canonical completed fit')
    artifact(result['checkpoint'], result['checkpoint_sha256'])
    artifact(result['best_val_prediction']['path'], result['best_val_prediction']['sha256'])
    artifact(result['initial_source']['path'], result['initial_source']['sha256'])
    if result.get('parent'):
        artifact(result['parent']['path'], result['parent']['sha256'])
    return result


def retry_contract(spec, tag, reason, resume_from):
    if not tag or not re.fullmatch(r'[A-Za-z0-9_-]+', tag) or not reason.strip():
        raise ValueError('Retry needs a unique safe tag and a concrete technical reason')
    if (HERE / 'selected_confirmation.json').exists() or (HERE / 'confirmation_test_exposure.json').exists():
        raise RuntimeError('No fit retries after final selection or TEST exposure')
    if spec['family'] == 'level' and (HERE / 'parents_selected.json').exists():
        raise RuntimeError('Selected LEVEL parents are fixed')
    if result_for(spec, required=False) is not None:
        raise RuntimeError('A valid completed fit cannot be retried for performance')
    jobs = read_json(HERE / 'ledger.json')['jobs']
    previous = [j for j in jobs if j['category'] == 'neural_fit'
                and j.get('metadata', {}).get('fit_id') == spec['id']]
    if not previous or any(j['status'] != 'failed' for j in previous):
        raise RuntimeError('Every prior canonical attempt must have a closed failed outcome')
    retry = {'canonical_id': spec['id'], 'attempt_id': spec['id'] + '__retry_' + tag,
             'technical_retry': True, 'reason': reason, 'previous_ledger_ids': [j['id'] for j in previous]}
    if resume_from:
        path = Path(resume_from).resolve()
        if path.name != 'last.pt' or path.parent.parent != LOCAL_RUNS.resolve():
            raise ValueError('Resume uses a confirmation run last.pt complete-epoch checkpoint')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['spec'] != spec or saved.get('boundary') != 'post-VAL/post-scheduler/complete epoch':
            raise RuntimeError('Resume specification or epoch boundary differs')
        if saved['epoch'] >= 120 or saved['stale'] >= 6:
            raise RuntimeError('Fit already reached its stop; recover verification without fitting')
        retry['resume_from'] = artifact(path)
    return retry


def fit(spec, gpu_job, retry=None):
    attempt_id = retry['attempt_id'] if retry else spec['id']
    out, local = RUNS / attempt_id, LOCAL_RUNS / attempt_id
    if out.exists() or local.exists():
        raise RuntimeError('Attempt exists; preserve it and use a technical retry label')
    metadata = dict(spec, fit_id=spec['id'])
    if retry:
        metadata.update(technical_retry=True, reason=retry['reason'])
    with Job('neural_fit', attempt_id, reserve_s=0, metadata=metadata) as fit_job:
        started, steps, actual_updates, carried_steps = time.perf_counter(), 0, 0, 0
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        attempt = {'id': attempt_id, 'canonical_id': spec['id'], 'spec': spec,
                   'status': 'running', 'ledger_id': fit_job.id, 'retry': retry,
                   'environment': environment_receipt()}
        save_json(out / 'attempt.json', attempt)
        try:
            data = load_data(spec['dataset'], include_test=False)
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            model = build_model(spec['dataset'], spec['family'], spec['seed'])
            initial = initial_reference(model, data)
            initial_parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            frozen_hash = state_digest(frozen_state(model))
            frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
            torch.save(make_checkpoint(model, spec, initial, epoch=0, steps=0), local / 'initial.pt')
            probe = phase_sample(data['train_origins'], spec['seed'], 0,
                                 max(96, data['_phase_period']), period=data['_phase_period'])
            initial_state_hashes = tensor_hashes(model.adapter_state())
            torch.cuda.reset_peak_memory_stats()
            initial_f0_parity = None
            if spec['family'] in ('full_mse', 'native'):
                with torch.no_grad():
                    check_x, _, _ = tensors(data, probe[:4])
                    with_adapter = model(check_x)
                    with model.backbone.disable_adapter():
                        without_adapter = model(check_x)
                    difference = (with_adapter - without_adapter).abs()
                    initial_f0_parity = {'max_absolute': float(difference.max()),
                                         'passed': bool(torch.all(difference <= 1e-5 + 1e-4 * without_adapter.abs()))}
                    if not initial_f0_parity['passed']:
                        raise RuntimeError('Initial full-channel LoRA differs from F0')
                    del check_x, with_adapter, without_adapter, difference
            initial_train, _ = evaluate(model, data, probe, gpu_job, attempt_id + ' initial TRAIN probe')
            optimizer, scheduler = optimizer_for(model, spec)
            history, schedules, gradients = [], [], {}
            best, best_epoch, stale, start_epoch = float('inf'), 0, 0, 0
            initial_parent_parity = None
            if retry and retry.get('resume_from'):
                artifact(retry['resume_from']['path'], retry['resume_from']['sha256'])
                resume_path = Path(retry['resume_from']['path'])
                resumed = torch.load(resume_path, map_location='cpu', weights_only=False)
                if resumed['initial_reference']['sha256'] != initial['sha256']:
                    raise RuntimeError('Resume initial state changed')
                model.restore_adapter(resumed['state'])
                frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
                optimizer.load_state_dict(resumed['optimizer'])
                scheduler.load_state_dict(resumed['scheduler'])
                restore_rng(resumed['rng'])
                steps = carried_steps = resumed['steps']
                best, best_epoch, stale = resumed['best'], resumed['best_epoch'], resumed['stale']
                start_epoch = resumed['epoch'] + 1
                history, schedules, gradients = resumed['history'], resumed['schedules'], resumed['gradients']
                initial_parent_parity = resumed.get('initial_parent_parity')
                for name in ('best.pt', 'best_val.npz'):
                    source = resume_path.parent / name
                    artifact(source, resumed['selected_files'][name]['sha256'])
                    shutil.copyfile(source, local / name)
                del resumed
            for epoch in range(start_epoch, 121):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512,
                                            period=data['_phase_period'])
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if actual_updates % 16 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}; batch {start // 4}', actual_updates)
                        optimizer.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = model.native_loss(x, y, mask) if spec['family'] == 'native' else masked_macro_loss(model(x), y, mask)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if actual_updates < 2:
                            gradients[str(steps + 1)] = gradient_record(model)
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      1, error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                        actual_updates += 1
                        losses.append(float(loss.detach()))
                val, prediction = evaluate(model, data, data['val_origins'], gpu_job, attempt_id + f' VAL {epoch}')
                if epoch == 0 and spec['family'] == 'a':
                    parent_path = Path(model.parent_receipt['path']).parent / 'best_val.npz'
                    selected_parent = read_json(HERE / 'parents_selected.json')['selected'][spec['dataset']]['level']
                    parent_index = selected_parent['seeds'].index(spec['seed'])
                    artifact(parent_path, selected_parent['best_val_prediction'][parent_index]['sha256'])
                    with np.load(parent_path, allow_pickle=False) as archive:
                        if not np.array_equal(archive['origins'], data['val_origins']):
                            raise RuntimeError('A and parent VAL origins differ')
                        reference = archive['prediction']
                    error = np.abs(prediction.astype(np.float64) - reference.astype(np.float64))
                    initial_parent_parity = {'parent_prediction': artifact(parent_path),
                                             'max_absolute': float(error.max()),
                                             'passed': bool(np.all(error <= 1e-5 + 1e-4 * np.abs(reference)))}
                    if not initial_parent_parity['passed']:
                        raise RuntimeError('Initial A differs from selected LEVEL parent')
                history.append({'epoch': epoch, 'steps': steps, 'attempt_updates': actual_updates,
                                'attempt_id': attempt_id, 'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    torch.save(make_checkpoint(model, spec, initial, epoch=epoch, steps=steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                torch.save(make_checkpoint(model, spec, initial, epoch=epoch, steps=steps,
                                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                           rng=rng_state(), best=best, best_epoch=best_epoch, stale=stale,
                                           history=history, schedules=schedules, gradients=gradients,
                                           initial_parent_parity=initial_parent_parity,
                                           selected_files={name: artifact(local / name) for name in ('best.pt', 'best_val.npz')},
                                           boundary='post-VAL/post-scheduler/complete epoch'), local / 'last.pt')
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}; best {best_epoch}', actual_updates)
                print(f'{attempt_id} epoch={epoch} best={best_epoch} val_mse={val["mse"]:.8g} elapsed={time.perf_counter()-started:.1f}s', flush=True)
                if epoch and stale >= 6:
                    break
            final_train, _ = evaluate(model, data, probe, gpu_job, attempt_id + ' final TRAIN probe')
            changes = {n: float((p.detach().cpu() - initial_parameters[n]).abs().max())
                       for n, p in model.named_parameters() if p.requires_grad}
            changed_counts = {n: int(torch.count_nonzero(p.detach().cpu() != initial_parameters[n]))
                              for n, p in model.named_parameters() if p.requires_grad}
            if not any(value > 0 for value in changes.values()):
                raise RuntimeError('No trainable tensor changed during the scheduled fit')
            assert all(p.grad is None and p._version == frozen_versions[n]
                       for n, p in model.named_parameters() if not p.requires_grad)
            assert state_digest(frozen_state(model)) == frozen_hash, 'Frozen weights/buffers changed'
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'one training model/optimizer; initial, TRAIN and VAL before selected restore'}
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in model.parameters())
            parent_receipt = model.parent_receipt
            last = torch.load(local / 'last.pt', map_location='cpu', weights_only=False)
            assert_equal(model.adapter_state(), last['state'])
            assert_equal(optimizer.state_dict(), last['optimizer'])
            assert_equal(scheduler.state_dict(), last['scheduler'])
            check_optimizer, check_scheduler = optimizer_for(model, spec)
            check_optimizer.load_state_dict(last['optimizer'])
            check_scheduler.load_state_dict(last['scheduler'])
            assert_equal(check_optimizer.state_dict(), optimizer.state_dict())
            assert_equal(check_scheduler.state_dict(), scheduler.state_dict())
            before_rng = rng_state()
            restore_rng(last['rng'])
            assert_equal(rng_state(), last['rng'])
            restore_rng(before_rng)
            del model, optimizer, scheduler, check_optimizer, check_scheduler, last, x, y, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
            restored = restore_model(local / 'best.pt')
            replay_score, replay = evaluate(restored, data, data['val_origins'], gpu_job, attempt_id + ' selected replay')
            with np.load(local / 'best_val.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['origins'], data['val_origins'])
                replay_error = float(np.abs(replay - archive['prediction']).max())
            assert replay_error <= 1e-6 and abs(replay_score['mse'] - best) <= 1e-8
            selected_train, _ = evaluate(restored, data, probe, gpu_job, attempt_id + ' selected TRAIN probe')
            selected_parameters = {n: p.detach().cpu() for n, p in restored.named_parameters() if p.requires_grad}
            result = {**attempt, 'status': 'complete', 'checkpoint': str(local / 'best.pt'),
                      'checkpoint_sha256': digest(local / 'best.pt'), 'best_val_prediction': artifact(local / 'best_val.npz'),
                      'initial_source': initial, 'initial_state_hashes': initial_state_hashes,
                      'initial_checkpoint': artifact(local / 'initial.pt'), 'last_checkpoint': artifact(local / 'last.pt'),
                      'parent': parent_receipt, 'initial_parent_parity': initial_parent_parity,
                      'initial_f0_parity': initial_f0_parity,
                      'best_val_mse': best, 'selected_epoch': best_epoch, 'epochs_completed': epoch,
                      'actual_updates': actual_updates, 'carried_steps': carried_steps, 'cumulative_model_steps': steps,
                      'upper_epoch_limit_improving': bool(epoch == 120 and epoch - best_epoch < 6),
                      'first_update_gradients': gradients, 'parameter_max_updates': changes,
                      'changed_parameter_scalars': changed_counts,
                      'selected_parameter_max_updates': {n: float((p - initial_parameters[n]).abs().max())
                                                         for n, p in selected_parameters.items()},
                      'frozen_parameters_verified': {'no_gradient': True, 'version_unchanged': True,
                                                      'optimizer_exclusion': True, 'state_sha256': frozen_hash},
                      'initial_train_probe': initial_train, 'selected_train_probe': selected_train,
                      'final_train_probe': final_train, 'initial_val_mse': history[0]['val_mse'],
                      'train_probe_origins': probe.tolist(), 'trainable_parameters': trainable_count,
                      'total_parameters': total_count, 'training_peak': peak, 'schedules': schedules,
                      'restart_roundtrip': {'model_optimizer_scheduler_rng': True, 'extra_updates': 0},
                      'replay_error': replay_error, 'data': artifact(data['_path']),
                      'elapsed_s': time.perf_counter() - started}
            save_json(out / 'result.json', result)
            if retry:
                save_json(REPLACEMENTS / (spec['id'] + '.json'),
                          {'canonical_id': spec['id'], 'selected_attempt_id': attempt_id,
                           'result': artifact(out / 'result.json'), 'technical_retry': retry,
                           'selection_basis': 'valid completion after technical failure, never performance'})
            attempt.update(status='complete', actual_updates=actual_updates, elapsed_s=result['elapsed_s'])
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('complete; selected checkpoint restored', actual_updates)
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', actual_updates=actual_updates, carried_steps=carried_steps,
                           cumulative_model_steps=steps, elapsed_s=time.perf_counter() - started,
                           error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('failed; original trace preserved', actual_updates)
            raise


def select(specs, parents=False):
    p = protocol()
    target = HERE / ('parents_selected.json' if parents else 'selected_confirmation.json')
    expected = [s for s in specs if s['family'] == 'level'] if parents else specs
    rows = [result_for(spec) for spec in expected]
    selected = {dataset: {} for dataset in p['units']}
    for dataset in p['units']:
        for family in (('level',) if parents else ('level', 'a', 'full_mse', 'native', 'direct')):
            group = [r for r in rows if r['spec']['dataset'] == dataset and r['spec']['family'] == family]
            learning_rates = [.001] if family == 'level' else [.0001, .001]
            if len(group) != len(SEEDS) * len(learning_rates):
                raise RuntimeError(f'Incomplete selection group {dataset}/{family}')
            means = []
            for lr in learning_rates:
                candidates = [r for r in group if r['spec']['lr'] == lr]
                if sorted(r['spec']['seed'] for r in candidates) != list(SEEDS):
                    raise RuntimeError('Selection is missing a seed')
                means.append({'lr': lr, 'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in candidates]))})
            chosen = min(means, key=lambda row: (row['mean_best_val_mse'], row['lr']))
            kept = sorted([r for r in group if r['spec']['lr'] == chosen['lr']], key=lambda r: r['spec']['seed'])
            selected[dataset][family] = {'lr': chosen['lr'], 'seeds': list(SEEDS),
                                         'run_ids': [r['id'] for r in kept],
                                         'checkpoints': [r['checkpoint'] for r in kept],
                                         'checkpoint_sha256': [r['checkpoint_sha256'] for r in kept],
                                         'selected_epochs': [r['selected_epoch'] for r in kept],
                                         'best_val_prediction': [r['best_val_prediction'] for r in kept],
                                         'parent': [r['parent'] for r in kept],
                                         'mean_best_val_mse': chosen['mean_best_val_mse'], 'lr_selection': means}
    known_initial, known_schedule = {}, {}
    for row in rows:
        spec = row['spec']
        kind = 'lora' if spec['family'] in LORA_FAMILIES else spec['family']
        key = (spec['dataset'], spec['seed'], kind)
        if key in known_initial:
            assert known_initial[key] == row['initial_source']['sha256']
        known_initial[key] = row['initial_source']['sha256']
        for schedule in row['schedules']:
            key = (spec['dataset'], spec['seed'], schedule['epoch'])
            if key in known_schedule:
                assert known_schedule[key] == schedule['sha256']
            known_schedule[key] = schedule['sha256']
    output = {'schema': 'v12_confirmation_selection_1', 'selected': selected,
              'all_neural_runs': [{k: row[k] for k in ('id', 'canonical_id', 'spec', 'checkpoint',
                                                      'checkpoint_sha256', 'best_val_mse', 'selected_epoch')}
                                  for row in rows],
              'test_used_for_selection': False,
              'criterion': 'minimum full VAL MSE including epoch0; earliest epoch; two-seed mean LR; tie 1e-4',
              'matched_initial_groups': len(known_initial), 'matched_schedule_groups': len(known_schedule),
              'protocol': artifact(HERE / 'confirmation_protocol.json')}
    if target.exists():
        if read_json(target) != output:
            raise RuntimeError('Selection already exists with different content')
    else:
        save_json(target, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('parents', 'all'), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reserve-seconds', type=float, required=True)
    parser.add_argument('--retry-fit')
    parser.add_argument('--retry-tag', default='')
    parser.add_argument('--retry-reason', default='')
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    configure()
    p = protocol()
    specs = p['neural_fits']
    if (HERE / 'confirmation_test_exposure.json').exists():
        raise RuntimeError('New confirmation TEST is exposed; no fitting is permitted')
    if args.stage == 'all' and not (HERE / 'parents_selected.json').exists():
        raise RuntimeError('Select all new-unit LEVEL parents before main fits')
    stage_specs = [s for s in specs if (s['family'] == 'level') == (args.stage == 'parents')]
    retry = None
    if args.retry_fit:
        matches = [s for s in stage_specs if s['id'] == args.retry_fit]
        if len(matches) != 1:
            raise ValueError('Retry fit is not in this approved stage')
        stage_specs = matches
        retry = retry_contract(matches[0], args.retry_tag, args.retry_reason, args.resume_from)
    elif args.retry_tag or args.retry_reason or args.resume_from:
        raise ValueError('Retry flags require --retry-fit')
    with Job('gpu', args.label, reserve_s=args.reserve_seconds,
             metadata={'stage': args.stage, 'fit_ids': [s['id'] for s in stage_specs]}) as gpu_job:
        for spec in stage_specs:
            if retry is None and result_for(spec, required=False) is not None:
                gpu_job.heartbeat('reusing verified completed fit ' + spec['id'])
                continue
            fit(spec, gpu_job, retry)
            gpu_job.heartbeat({'completed_fit': spec['id'], 'budget': budget_snapshot()})
        expected = [s for s in specs if s['family'] == 'level'] if args.stage == 'parents' else specs
        if all(result_for(spec, required=False) is not None for spec in expected):
            select(specs, parents=args.stage == 'parents')
        else:
            gpu_job.heartbeat('technical recovery complete; remaining scheduled fits still pending')


if __name__ == '__main__':
    main()
