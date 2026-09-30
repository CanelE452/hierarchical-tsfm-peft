"""Approved v11 neural fitting; all model selection uses TRAIN/VAL only."""
import argparse
import gc
import hashlib
import json
import random
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from model_v11 import (build_model, generate_initial, make_checkpoint, restore_model,
                       state_digest, tensor_hashes)
from runtime_v11 import (CACHE, HERE, Job, digest, load_data, masked_macro_loss,
                         phase_sample, read_json, save_json, score_arrays,
                         source_receipt, sources_for, tensors)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def optimizer_for(model, spec):
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    optimizer = torch.optim.AdamW(params, lr=spec['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, threshold=1e-4, threshold_mode='rel')
    assert {id(p) for p in params} == {id(p) for g in optimizer.param_groups for p in g['params']}
    return optimizer, scheduler


def kind_for(family):
    if family in ('a_p_lora_qfixed', 'full_lora_mse'):
        return 'lora'
    return 'u' if family == 'b_learned_u' else 'direct'


def initial_for(spec, data):
    kind = kind_for(spec['family'])
    path = CACHE / 'initial' / spec['dataset'] / f'{kind}_{spec["seed"]}.pt'
    record = HERE / 'initial' / spec['dataset'] / f'{kind}_{spec["seed"]}.json'
    if path.exists():
        receipt = read_json(record)
        assert digest(path) == receipt['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert tensor_hashes(saved['state']) == receipt['tensor_hashes']
        return saved['state'], receipt
    initial_spec = dict(spec)
    if kind == 'lora':
        initial_spec['family'] = 'full_lora_mse'
    initial = generate_initial(initial_spec, data['basis'],
                               sources=sources_for(spec['dataset'], spec['seed']))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': initial, 'dataset': spec['dataset'], 'seed': spec['seed'],
                'kind': kind, 'data_sha256': digest(data['_path'])}, path)
    receipt = {'path': str(path), 'sha256': digest(path), 'kind': kind,
               'tensor_hashes': tensor_hashes(initial), 'source': source_receipt()}
    save_json(record, receipt)
    return initial, receipt


@torch.no_grad()
def evaluate(model, data, origins, job, label):
    model.eval()
    prediction = []
    for start in range(0, len(origins), 4):
        job.check_limits()
        x, _, _ = tensors(data, origins[start:start + 4])
        prediction.append(model(x).cpu().numpy())
    prediction = np.concatenate(prediction)
    y = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    job.heartbeat(label)
    return score_arrays(prediction, y, mask), prediction


def payload(model, receipt, epoch, steps, **extra):
    return make_checkpoint(model, initial_reference=receipt,
                           extra=dict(epoch=epoch, steps=steps, **extra))


def frozen_state(model):
    names = {n for n, p in model.named_parameters() if not p.requires_grad}
    names.update(n for n, _ in model.named_buffers())
    return {n: value for n, value in model.state_dict().items() if n in names}


def gradient_record(model):
    return {n: {'numel': p.numel(), 'has_grad': p.grad is not None,
                'max_abs': None if p.grad is None else float(p.grad.detach().abs().max()),
                'nonzero': 0 if p.grad is None else int(torch.count_nonzero(p.grad))}
            for n, p in model.named_parameters() if p.requires_grad}


def result_for(spec, required=True):
    replacement_path = HERE / 'replacements' / (spec['id'] + '.json')
    if replacement_path.exists():
        replacement = read_json(replacement_path)
        assert replacement['canonical_id'] == spec['id']
        path = Path(replacement['selected_result']['path'])
        assert digest(path) == replacement['selected_result']['sha256']
    else:
        path = HERE / 'runs' / spec['id'] / 'result.json'
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    item = read_json(path)
    assert item['status'] == 'complete' and item['spec'] == spec
    assert digest(item['checkpoint']) == item['checkpoint_sha256']
    return item


def retry_contract(spec, retry_tag, reason, resume_from):
    if not retry_tag or not re.fullmatch(r'[a-zA-Z0-9_-]+', retry_tag) or not reason.strip():
        raise ValueError('Technical retry requires a safe unique tag and a concrete repair reason')
    if (HERE / 'selected.json').exists() or (HERE / 'test_exposure.json').exists():
        raise RuntimeError('Retry cannot alter an already fixed selection/evaluation')
    if spec['family'] == 'direct_nlinear' and (HERE / 'selected_direct.json').exists():
        raise RuntimeError('Selected S source is fixed; changing it needs an explicit affected-comparison repair')
    original_path = HERE / 'runs' / spec['id'] / 'attempt.json'
    original = read_json(original_path)
    if original['spec'] != spec:
        raise RuntimeError('Retry must preserve the approved canonical fit specification')
    ledger = read_json(HERE / 'ledger.json')['jobs']
    related = [row for row in ledger if row['category'] == 'neural_fit'
               and row['metadata'].get('fit_id') == spec['id']]
    if not related or any(row['status'] == 'running' for row in related):
        raise RuntimeError('Original fit must have a closed ledger outcome before retry')
    invalidation_path = HERE / 'invalidations' / (spec['id'] + '.json')
    invalidation = None
    if original['status'] not in ('failed', 'running', 'complete'):
        raise RuntimeError('Unknown original attempt outcome; inspect before retry')
    if original['status'] == 'complete' or (HERE / 'runs' / spec['id'] / 'result.json').exists() or (HERE / 'replacements' / (spec['id'] + '.json')).exists():
        invalidation = read_json(invalidation_path)
        if invalidation.get('classification') != 'technical' or invalidation.get('canonical_id') != spec['id']:
            raise RuntimeError('A completed/previously replaced fit requires explicit technical invalidation, never score-based retry')
        if not invalidation.get('reason') or not invalidation.get('affected_result_sha256'):
            raise RuntimeError('Technical invalidation must identify the invalid result bytes and cause')
        previous = result_for(spec)
        previous_path = HERE / 'runs' / previous['id'] / 'result.json'
        if digest(previous_path) != invalidation['affected_result_sha256']:
            raise RuntimeError('Invalidation does not identify the currently selected successful attempt')
    attempt_id = spec['id'] + '__retry_' + retry_tag
    if (HERE / 'runs' / attempt_id).exists() or (CACHE / 'runs' / attempt_id).exists():
        raise RuntimeError('Retry tag already exists; preserve the prior attempt')
    retry = {'canonical_id': spec['id'], 'attempt_id': attempt_id, 'technical_retry': True,
             'reason': reason, 'original_attempt': {'path': str(original_path), 'sha256': digest(original_path)},
             'invalidation': None if invalidation is None else {'path': str(invalidation_path), 'sha256': digest(invalidation_path)}}
    if resume_from:
        path = Path(resume_from).resolve()
        if path.parent.parent != (CACHE / 'runs').resolve() or path.name != 'restart.pt':
            raise ValueError('Resume uses an existing v11 run restart.pt, not arbitrary model weights')
        source_attempt = read_json(HERE / 'runs' / path.parent.name / 'attempt.json')
        if source_attempt['spec'] != spec:
            raise RuntimeError('Resume source is another canonical configuration')
        resume = torch.load(path, map_location='cpu', weights_only=False)
        if resume['spec'] != spec or resume.get('boundary') != 'post-VAL/post-scheduler/complete epoch':
            raise RuntimeError('Only the same fit at a complete epoch boundary can resume')
        if int(resume['epoch']) >= 120 or int(resume['stale']) >= 6:
            raise RuntimeError('Training already reached its stopping condition; use replay-only recovery without extra fitting')
        retry['resume_from'] = {'path': str(path), 'sha256': digest(path)}
    return retry


def record_replacement(spec, attempt_id, retry, result_path):
    destination = HERE / 'replacements' / (spec['id'] + '.json')
    previous = read_json(destination) if destination.exists() else None
    receipt = {'canonical_id': spec['id'], 'selected_attempt_id': attempt_id,
               'selected_result': {'path': str(result_path), 'sha256': digest(result_path)},
               'technical_reason': retry['reason'], 'original_attempt': retry['original_attempt'],
               'invalidation': retry['invalidation'], 'previous_replacement': previous,
               'selection_basis': 'valid completion after documented technical failure, not performance'}
    save_json(HERE / 'runs' / attempt_id / 'replacement_receipt.json', receipt)
    save_json(destination, receipt)


def fit(spec, gpu_job, retry=None):
    attempt_id = retry['attempt_id'] if retry else spec['id']
    out, local = HERE / 'runs' / attempt_id, CACHE / 'runs' / attempt_id
    if out.exists() or local.exists():
        raise RuntimeError('Attempt exists: preserve it and use a recorded technical retry ID')
    metadata = dict(spec, fit_id=spec['id'])
    if retry:
        metadata.update(technical_retry=True, reason=retry['reason'], retry_receipt=retry)
    with Job(attempt_id, category='neural_fit', reserve_seconds=0, metadata=metadata) as fit_job:
        started, steps, carried_steps, actual_updates = time.perf_counter(), 0, 0, 0
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        attempt = dict(id=attempt_id, canonical_id=spec['id'], spec=spec, status='running',
                       source=source_receipt(), retry=retry)
        save_json(out / 'attempt.json', attempt)
        try:
            data = load_data(spec['dataset'])
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            initial, receipt = initial_for(spec, data)
            model = build_model(spec, data['basis'], initial=initial, device='cuda',
                                sources=sources_for(spec['dataset'], spec['seed']))
            initial_state = model.adapter_state()
            initial_parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            frozen_hash = state_digest(frozen_state(model))
            frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
            torch.save(payload(model, receipt, 0, 0), local / 'initial.pt')
            probe = phase_sample(data['train_origins'], spec['seed'], 0,
                                 max(96, data['_phase_period']), period=data['_phase_period'])
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            torch.cuda.reset_peak_memory_stats()
            initial_train, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' initial TRAIN probe')
            optimizer, scheduler = optimizer_for(model, spec)
            history, schedules, gradients = [], [], {}
            best, best_epoch, stale = float('inf'), 0, 0
            start_epoch, elapsed_before_resume = 0, 0.
            actual_u_initial = None
            if spec['family'] == 'b_learned_u':
                actual_u_initial = model.U.detach().cpu().clone()
            if retry and retry.get('resume_from'):
                resume_path = Path(retry['resume_from']['path'])
                assert digest(resume_path) == retry['resume_from']['sha256']
                resumed = torch.load(resume_path, map_location='cpu', weights_only=False)
                assert resumed['spec'] == spec
                assert torch.equal(torch.as_tensor(resumed['basis']).float(), torch.as_tensor(data['basis']).float())
                if resumed.get('initial_reference', {}).get('sha256') != receipt['sha256']:
                    raise RuntimeError('Resume initial state differs from the paired original state')
                model.restore_adapter(resumed['state'])
                assert state_digest(frozen_state(model)) == frozen_hash
                frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
                optimizer.load_state_dict(resumed['optimizer'])
                scheduler.load_state_dict(resumed['scheduler'])
                history, schedules = resumed['history'], resumed['schedules']
                gradients = resumed.get('gradients', {})
                best, best_epoch, stale = resumed['best'], resumed['best_epoch'], resumed['stale']
                carried_steps = steps = int(resumed['steps'])
                start_epoch = int(resumed['epoch']) + 1
                elapsed_before_resume = float(history[-1]['elapsed_s'])
                for name in ('best.pt', 'best_val.npz'):
                    source = resume_path.parent / name
                    recorded = resumed.get('selected_files', {}).get(name)
                    if recorded is not None and digest(source) != recorded['sha256']:
                        raise RuntimeError('Resume best checkpoint/prediction changed after the saved epoch boundary')
                    shutil.copyfile(source, local / name)
                old_best = torch.load(local / 'best.pt', map_location='cpu', weights_only=False)
                if old_best['epoch'] != best_epoch or old_best['spec'] != spec:
                    raise RuntimeError('Resume best checkpoint does not match its saved selection state')
                with np.load(resume_path.parent / 'origins.npz', allow_pickle=False) as previous_origins:
                    assert np.array_equal(previous_origins['train_probe'], probe)
                    assert np.array_equal(previous_origins['val'], data['val_origins'])
                restore_rng(resumed['rng'])
                assert_nested_equal(rng_state(), resumed['rng'])
                save_json(out / 'resume_receipt.json', {'source': retry['resume_from'], 'start_epoch': start_epoch,
                    'carried_model_steps': carried_steps, 'new_attempt_updates': 0, 'optimizer_scheduler_rng_restored': True,
                    'same_probe_and_val_origins': True, 'max_total_epochs': 120})
                del old_best, resumed
            for epoch in range(start_epoch, 121):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512,
                                            period=data['_phase_period'])
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if steps % 16 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}, batch {start // 4}', actual_updates)
                        optimizer.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = masked_macro_loss(model(x), y, mask)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if steps < 2 or (retry and actual_updates < 2):
                            key = str(steps + 1) if carried_steps == 0 else 'resumed_step_' + str(steps + 1)
                            gradients[key] = gradient_record(model)
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      1, error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                        actual_updates += 1
                        losses.append(float(loss.detach()))
                val, prediction = evaluate(model, data, data['val_origins'], gpu_job,
                                           spec['id'] + f' VAL epoch {epoch}')
                history.append({'epoch': epoch, 'steps': steps,
                                'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': elapsed_before_resume + time.perf_counter() - started,
                                'attempt_id': attempt_id, 'attempt_updates': actual_updates})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    torch.save(payload(model, receipt, epoch, steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                torch.save(payload(model, receipt, epoch, steps, optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), rng=rng_state(), best=best,
                                   best_epoch=best_epoch, stale=stale, history=history, schedules=schedules,
                                   gradients=gradients,
                                   selected_files={name: {'path': str(local / name), 'sha256': digest(local / name)}
                                                   for name in ('best.pt', 'best_val.npz')},
                                   boundary='post-VAL/post-scheduler/complete epoch'), local / 'restart.pt')
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}; best epoch {best_epoch}', actual_updates)
                print(json.dumps({'id': attempt_id, 'canonical_id': spec['id'], 'epoch': epoch, 'best_epoch': best_epoch,
                                  'val_mse': val['mse'], 'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
                if epoch and stale >= 6:
                    break
            final_train, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' final TRAIN probe')
            changes = {n: float((p.detach().cpu() - initial_parameters[n]).abs().max())
                       for n, p in model.named_parameters() if p.requires_grad}
            counts = {n: int(torch.count_nonzero(p.detach().cpu() != initial_parameters[n]))
                      for n, p in model.named_parameters() if p.requires_grad}
            assert any(x > 0 for x in changes.values()), 'No trainable tensor changed'
            assert all(p.grad is None and p._version == frozen_versions[n]
                       for n, p in model.named_parameters() if not p.requires_grad)
            assert state_digest(frozen_state(model)) == frozen_hash, 'Frozen tensor or buffer changed'
            u_check = None
            if actual_u_initial is not None:
                u = model.U.detach().cpu()
                u_check = {'actual_u_max_update': float((u - actual_u_initial).abs().max()),
                           'gram_max_error': float((u.T @ u - torch.eye(u.shape[1])).abs().max())}
                assert u_check['actual_u_max_update'] > 0 and u_check['gram_max_error'] <= 1e-5
            total_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(),
                    'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'one training model/optimizer; initial/final TRAIN probe and TRAIN/VAL before restore'}
            restart = torch.load(local / 'restart.pt', map_location='cpu', weights_only=False)
            assert_nested_equal(model.adapter_state(), restart['state'])
            assert_nested_equal(optimizer.state_dict(), restart['optimizer'])
            assert_nested_equal(scheduler.state_dict(), restart['scheduler'])
            opt2, sched2 = optimizer_for(model, spec)
            opt2.load_state_dict(restart['optimizer'])
            sched2.load_state_dict(restart['scheduler'])
            assert_nested_equal(opt2.state_dict(), optimizer.state_dict())
            assert_nested_equal(sched2.state_dict(), scheduler.state_dict())
            before_rng = rng_state()
            restore_rng(restart['rng'])
            assert_nested_equal(rng_state(), restart['rng'])
            restore_rng(before_rng)
            del model, optimizer, scheduler, opt2, sched2, restart, x, y, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
            restored = restore_model(local / 'best.pt', device='cuda')
            replay_score, replay = evaluate(restored, data, data['val_origins'], gpu_job, spec['id'] + ' restore replay')
            with np.load(local / 'best_val.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['origins'], data['val_origins'])
                replay_error = float(np.abs(replay - archive['prediction']).max())
            assert replay_error <= 1e-6 and abs(replay_score['mse'] - best) <= 1e-8
            selected_train, _ = evaluate(restored, data, probe, gpu_job, spec['id'] + ' selected TRAIN probe')
            result = dict(attempt, status='complete', checkpoint=str(local / 'best.pt'),
                          checkpoint_sha256=digest(local / 'best.pt'), best_val_mse=best,
                          selected_epoch=best_epoch, epochs_completed=epoch, actual_updates=actual_updates,
                          cumulative_model_steps=steps, carried_model_steps=carried_steps,
                          initial_source=receipt, initial_state_hashes=tensor_hashes(initial_state),
                          initial_checkpoint=str(local / 'initial.pt'),
                          initial_checkpoint_sha256=digest(local / 'initial.pt'),
                          first_update_gradients=gradients, parameter_max_updates=changes,
                          changed_parameter_scalars=counts, actual_u_check=u_check,
                          frozen_parameters_verified={'sha256': frozen_hash, 'no_grad': True,
                                                      'unchanged_parameters_and_buffers': True},
                          initial_train_probe=initial_train, selected_train_probe=selected_train,
                          final_train_probe=final_train, initial_val_mse=history[0]['val_mse'],
                          best_val_prediction={'path': str(local / 'best_val.npz'), 'sha256': digest(local / 'best_val.npz')},
                          restart={'path': str(local / 'restart.pt'), 'sha256': digest(local / 'restart.pt'),
                                   'model_optimizer_scheduler_rng_serialization_checked': True, 'extra_updates': 0},
                          replay_error=replay_error, trainable_parameters=trainable_parameters,
                          total_parameters=total_parameters, training_peak=peak, schedules=schedules,
                          upper_epoch_limit_improving=bool(epoch == 120 and epoch - best_epoch < 6),
                          elapsed_s=time.perf_counter() - started)
            save_json(out / 'result.json', result)
            save_json(out / 'receipt.json', {'id': attempt_id, 'canonical_id': spec['id'], 'result_sha256': digest(out / 'result.json'),
                                           'checkpoint_sha256': result['checkpoint_sha256']})
            attempt.update(status='complete', elapsed_s=result['elapsed_s'], actual_updates=actual_updates,
                           cumulative_model_steps=steps, carried_model_steps=carried_steps)
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('complete: selected VAL checkpoint restored', actual_updates)
            if retry:
                record_replacement(spec, attempt_id, retry, out / 'result.json')
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', actual_updates=actual_updates, cumulative_model_steps=steps,
                           carried_model_steps=carried_steps, elapsed_s=time.perf_counter() - started,
                           error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('failed; original trace preserved', actual_updates)
            raise


def select(specs, direct_only=False):
    result = {'selected': {d: {} for d in ('robin', 'jena', 'hog')},
              'all_neural_runs': [], 'test_used': False, 'test_used_for_selection': False,
              'criterion': 'minimum full VAL per seed including step0; mean seeds LR; exact tie 1e-4'}
    for ds in ('robin', 'jena'):
        records = []
        for seed in (92601, 92602):
            sources = sources_for(ds, seed)
            path = Path(sources['direct_checkpoint'])
            old_result = read_json(Path(str(path).replace('/.cache/', '/research/').replace('\\.cache\\', '\\research\\')).parent / 'result.json')
            records.append(old_result)
        result['selected'][ds]['direct_nlinear'] = {
            'lr': .001, 'run_ids': [r['id'] for r in records],
            'checkpoints': [r['checkpoint'] for r in records],
            'checkpoint_sha256': [r['checkpoint_sha256'] for r in records],
            'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in records])), 'reused': True}
    rows = []
    for spec in specs:
        if direct_only and spec['family'] != 'direct_nlinear':
            continue
        item = result_for(spec)
        rows.append(item)
        result['all_neural_runs'].append({k: item[k] for k in ('id', 'spec', 'checkpoint', 'checkpoint_sha256', 'best_val_mse', 'selected_epoch')})
    for ds, family in sorted({(r['spec']['dataset'], r['spec']['family']) for r in rows}):
        group = [r for r in rows if r['spec']['dataset'] == ds and r['spec']['family'] == family]
        means = [{'lr': lr, 'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in group if r['spec']['lr'] == lr]))}
                 for lr in (1e-4, 1e-3)]
        assert len(group) == 4
        chosen = min(means, key=lambda r: r['mean_best_val_mse'])['lr']
        selected = sorted([r for r in group if r['spec']['lr'] == chosen], key=lambda r: r['spec']['seed'])
        assert [r['spec']['seed'] for r in selected] == [92601, 92602]
        result['selected'][ds][family] = {
            'lr': chosen, 'run_ids': [r['id'] for r in selected],
            'checkpoints': [r['checkpoint'] for r in selected],
            'checkpoint_sha256': [r['checkpoint_sha256'] for r in selected],
            'mean_best_val_mse': min(m['mean_best_val_mse'] for m in means), 'lr_selection': means}
    known_schedules, known_initial = {}, {}
    for row in rows:
        spec = row['spec']
        key = (spec['dataset'], spec['seed'], kind_for(spec['family']))
        if key in known_initial:
            assert known_initial[key] == row['initial_source']['sha256']
        known_initial[key] = row['initial_source']['sha256']
        for schedule in row['schedules']:
            key = (spec['dataset'], spec['seed'], schedule['epoch'])
            if key in known_schedules:
                assert known_schedules[key] == schedule['sha256']
            known_schedules[key] = schedule['sha256']
    result['matched_initial_and_schedule_checks'] = {'initial_groups': len(known_initial), 'dataset_seed_epochs': len(known_schedules)}
    target = HERE / ('selected_direct.json' if direct_only else 'selected.json')
    if target.exists():
        raise RuntimeError('Selection exists: never silently overwrite')
    save_json(target, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('hog_direct', 'main', 'select'), required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=14400)
    parser.add_argument('--id')
    parser.add_argument('--retry-of', help='Canonical approved fit ID; technical failure only')
    parser.add_argument('--retry-tag', help='Unique safe suffix, preserving all earlier attempts')
    parser.add_argument('--reason', default='')
    parser.add_argument('--resume-from', type=Path, help='Same canonical fit restart.pt at a complete epoch boundary')
    args = parser.parse_args()
    configure()
    specs = read_json(HERE / 'protocol.json')['fits']
    if args.retry_of:
        if args.stage == 'select' or args.id:
            parser.error('--retry-of cannot combine with select or --id')
        selected_specs = [spec for spec in specs if spec['id'] == args.retry_of]
        if len(selected_specs) != 1:
            parser.error('--retry-of must identify exactly one approved canonical fit')
        spec = selected_specs[0]
        if (spec['family'] == 'direct_nlinear') != (args.stage == 'hog_direct'):
            parser.error('Retry stage does not match canonical family')
        retry = retry_contract(spec, args.retry_tag, args.reason, args.resume_from)
        with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds,
                 metadata={'purpose': 'documented technical retry', 'canonical_id': spec['id']}) as job:
            fit(spec, job, retry=retry)
        return
    if args.retry_tag or args.resume_from or args.reason:
        parser.error('--retry-tag/--reason/--resume-from require --retry-of')
    if args.stage == 'select':
        with Job(args.job, category='cpu_analysis', reserve_seconds=60):
            select(specs)
        return
    chosen = [s for s in specs if (s['family'] == 'direct_nlinear') == (args.stage == 'hog_direct')]
    if args.id:
        chosen = [s for s in chosen if s['id'] == args.id]
        assert len(chosen) == 1
    with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds) as job:
        for spec in chosen:
            existing = result_for(spec, required=False)
            if existing is not None:
                continue
            fit(spec, job)
    if args.stage == 'hog_direct' and not args.id:
        with Job(args.job + '_selection', category='cpu_analysis', reserve_seconds=60):
            select(specs, direct_only=True)


if __name__ == '__main__':
    main()
