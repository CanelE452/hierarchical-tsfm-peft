"""Approved v7 training and dataset-specific VAL selection; no TEST data access."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from model_v7 import (build_model, initial_edg, initial_direct, macro_loss, restore_model,
                      tensor_hashes, state_digest, NEURAL_FAMILIES, DATASETS)
from runtime_v7 import (CACHE, HERE, Job, digest, environment_receipt, load_data,
                        phase_sample, require_storage, save_json, score_arrays, source_receipt, tensors)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def optimizer_for(model, spec):
    parameters = [p for p in model.parameters() if p.requires_grad]
    group_name = 'temporal' if spec['family'] == 'direct_nlinear' else 'lora' if spec['family'] == 'lora' else 'g'
    groups = [{'params': parameters, 'lr': spec['lr'], 'name': group_name}]
    opt = torch.optim.AdamW(groups, weight_decay=0)
    assert {id(p) for p in parameters} == {id(p) for g in opt.param_groups for p in g['params']}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=2, threshold=1e-4, threshold_mode='rel')
    return opt, scheduler


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
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
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def prepare_initial(name, dataset):
    target = HERE / f'initial_manifest_{dataset}.json'
    if target.exists():
        raise RuntimeError('Initial manifest already exists; preserve it')
    with Job(name, category='cpu_analysis', reserve_seconds=60, metadata={'dataset': dataset}) as job:
        data = load_data(dataset, include_test=False)
        basis = torch.as_tensor(data['basis'], dtype=torch.float32)
        assert tuple(basis.shape) == DATASETS[dataset]
        records = {}
        for kind in (('direct',) if dataset == 'robin' else ('edg', 'direct')):
            records[kind] = {}
            for seed in (92601, 92602):
                path = CACHE / 'initial' / dataset / f'{kind}_seed_{seed}.pt'
                if path.exists():
                    raise RuntimeError('Shared initial checkpoint already exists')
                path.parent.mkdir(parents=True, exist_ok=True)
                state = initial_direct(seed) if kind == 'direct' else initial_edg(basis, seed)
                hashes = tensor_hashes(state)
                torch.save({'state': state, 'basis': basis, 'seed': seed, 'dataset': dataset,
                            'kind': kind, 'epoch': 0, 'steps': 0, 'tensor_hashes': hashes,
                            'data_sha256': digest(data['_path'])}, path)
                records[kind][str(seed)] = {'path': str(path), 'sha256': digest(path), 'tensor_hashes': hashes}
        save_json(target, {'schema_version': 1, 'dataset': dataset, 'records': records, 'data_sha256': digest(data['_path']),
                          'recipe': {'edg': 'CPU fork_rng(seed); E,D,G creation order; overwrite PCA E/D; zero G up',
                                     'direct': 'CPU fork_rng(seed); default nn.Linear(512,48,bias=True); shared across LRs'},
                          'source': source_receipt()})
        job.heartbeat('shared untrained states saved', 0)


def initial_for(dataset, seed, family):
    manifest = json.loads((HERE / f'initial_manifest_{dataset}.json').read_text(encoding='utf-8'))
    kind = 'direct' if family == 'direct_nlinear' else 'edg'
    receipt = manifest['records'][kind][str(seed or 92601)]
    assert digest(receipt['path']) == receipt['sha256']
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    assert saved['dataset'] == dataset and saved['kind'] == kind
    assert saved['epoch'] == saved['steps'] == 0
    assert tensor_hashes(saved['state']) == receipt['tensor_hashes'] == saved['tensor_hashes']
    return saved['state'], receipt


def validate_spec(spec):
    assert spec['dataset'] in DATASETS and spec['seed'] in (92601, 92602)
    channels, latent = DATASETS[spec['dataset']]
    assert spec['channels'] == channels
    assert spec['family'] in NEURAL_FAMILIES
    if spec['family'] == 'direct_nlinear':
        assert spec['arm'] == 'direct' and spec['latent'] == 0 and spec['ed_mode'] == 'none'
        assert spec['lr'] in (1e-3, 1e-4)
        assert spec.get('normalization') == 'train_standardization_only'
    else:
        assert spec['dataset'] == 'jena' and spec['latent'] == latent and spec['ed_mode'] == 'fixed_ed'
        assert spec['arm'] == {'level_res': 'residual', 'old_res': 'residual', 'level_raw': 'raw', 'lora': 'lora'}[spec['family']]
        assert spec['lr'] == (1e-4 if spec['family'] == 'lora' else 1e-3)
        assert spec.get('normalization') == 'context'
    assert spec['epochs'] in (120, 240)
    if spec['epochs'] == 240:
        assert spec.get('resume_from') and spec.get('decision_record')
        decision = json.loads((HERE / spec['decision_record']).read_text(encoding='utf-8'))
        assert spec['id'] in decision['authorized_continuations']
    elif spec.get('resume_from'):
        raise ValueError('Only the pre-TEST authorized 120-to-240 continuation is allowed')
    assert spec['id'] and '..' not in spec['id'] and all(c.isalnum() or c in '_.-' for c in spec['id'])


@torch.no_grad()
def evaluate(model, data, origins, job, label):
    model.eval()
    predictions = []
    for start in range(0, len(origins), 4):
        job.check_limits()
        x, _, _ = tensors(data, origins[start:start + 4])
        predictions.append(model(x).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    job.heartbeat(label)
    return score_arrays(prediction, target, mask), prediction


def payload(model, spec, basis, receipt, epoch, steps):
    return {'spec': dict(spec), 'basis': torch.as_tensor(basis).cpu(), 'state': model.adapter_state(),
            'initial_path': receipt['path'], 'initial_sha256': receipt['sha256'],
            'model_config': model.model_config(), 'epoch': epoch, 'steps': steps}


def gradient_record(model):
    return {name: {'requires_grad': p.requires_grad,
                   'max_abs': None if p.grad is None else float(p.grad.detach().abs().max())}
            for name, p in model.named_parameters() if p.requires_grad or name.startswith(('encoder.', 'decoder.'))}


def result_receipt(result, out):
    return {key: result[key] for key in ('id', 'spec', 'checkpoint', 'checkpoint_sha256',
                                         'best_val_prediction', 'best_val_mse', 'selected_epoch')} | {
        'result_path': str(out / 'result.json'), 'result_sha256': digest(out / 'result.json'),
        'deterministic': result.get('deterministic', False)}


def fit(spec, data, gpu_job):
    validate_spec(spec)
    out, local = HERE / 'runs' / spec['id'], CACHE / 'runs' / spec['id']
    if out.exists() or local.exists():
        raise RuntimeError('Existing attempt: use an explicit new retry/resume id')
    with Job(spec['id'], category='real_fit', reserve_seconds=0, metadata=spec) as fit_job:
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        started = time.perf_counter()
        attempt = {'id': spec['id'], 'spec': spec, 'status': 'running', 'ledger_id': fit_job.id,
                   'source': source_receipt(), 'environment': environment_receipt(), 'data_sha256': digest(data['_path'])}
        save_json(out / 'attempt.json', attempt)
        steps = start_steps = 0
        try:
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            edg, receipt = initial_for(spec['dataset'], spec['seed'], spec['family'])
            if spec['family'] != 'direct_nlinear':
                assert torch.equal(edg['encoder.weight'], torch.as_tensor(data['basis']).T)
                assert torch.equal(edg['decoder.weight'], torch.as_tensor(data['basis']))
            model = build_model(spec, data['basis'], edg, device='cuda')
            initial = model.adapter_state()
            initial_hash = tensor_hashes(initial)
            frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
            frozen_hash = state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
            torch.save(payload(model, spec, data['basis'], receipt, 0, 0), local / 'initial.pt')
            torch.cuda.reset_peak_memory_stats()
            probe = phase_sample(data['train_origins'], spec['seed'], 0,
                                 max(96, data['_phase_period']), period=data['_phase_period'])
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            initial_train, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' initial TRAIN probe')
            opt, scheduler = optimizer_for(model, spec)
            history, schedules, first_gradients = [], [], {}
            best, best_epoch, stale, first_epoch = float('inf'), 0, 0, 0
            if spec.get('resume_from'):
                restart_path = Path(spec['resume_from'])
                restart = torch.load(restart_path, map_location='cpu', weights_only=False)
                for key in ('dataset', 'seed', 'family', 'arm', 'ed_mode', 'latent', 'channels', 'lr'):
                    assert restart['spec'][key] == spec[key]
                assert restart['initial_sha256'] == receipt['sha256']
                assert restart['model_config'] == model.model_config()
                assert restart['epoch'] == 120 and restart['stale'] < 6 and restart['epoch'] - restart['best_epoch'] < 6
                model.restore_adapter(restart['state'])
                opt.load_state_dict(restart['optimizer'])
                scheduler.load_state_dict(restart['scheduler'])
                restore_rng(restart['rng'])
                steps = start_steps = restart['steps']
                best, best_epoch, stale = restart['best'], restart['best_epoch'], restart['stale']
                first_epoch, history, schedules = restart['epoch'] + 1, restart['history'], restart['schedules']
                parent_best = torch.load(restart_path.parent / 'best.pt', map_location='cpu', weights_only=False)
                model_spec = dict(spec)
                parent_best['spec'] = model_spec
                torch.save(parent_best, local / 'best.pt')
                with np.load(restart_path.parent / 'best_val.npz', allow_pickle=False) as archive:
                    np.savez_compressed(local / 'best_val.npz', **{k: archive[k] for k in archive.files})
                frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
            for epoch in range(first_epoch, spec['epochs'] + 1):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512,
                                            period=data['_phase_period'])
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if steps % 10 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}, batch {start // 4}', steps - start_steps)
                        opt.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = model.native_loss(x, y, mask) if spec['family'] == 'lora' else macro_loss(model(x), y, mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if steps - start_steps < 2:
                            first_gradients[str(steps - start_steps + 1)] = gradient_record(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        opt.step()
                        steps += 1
                        losses.append(float(loss.detach()))
                val, prediction = evaluate(model, data, data['val_origins'], gpu_job, spec['id'] + f' VAL epoch {epoch}')
                history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'learning_rates': {g['name']: g['lr'] for g in opt.param_groups},
                                'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    require_storage(16 * 1024**2)
                    torch.save(payload(model, spec, data['basis'], receipt, epoch, steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                restart = payload(model, spec, data['basis'], receipt, epoch, steps)
                restart.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                               stale=stale, best=best, best_epoch=best_epoch, history=history, schedules=schedules,
                               next_batch=0, boundary='post-VAL, post-scheduler, complete epoch')
                torch.save(restart, local / 'restart.pt')
                del restart
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}, best {best_epoch}', steps - start_steps)
                print(json.dumps({'id': spec['id'], 'epoch': epoch, 'best_epoch': best_epoch,
                                  'val_mse': val['mse'], 'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
                if epoch and stale >= 6:
                    break
            final = model.adapter_state()
            changes = {k: float((final[k] - initial[k]).abs().max()) for k in initial}
            changed_counts = {k: int(torch.count_nonzero(final[k] != initial[k])) for k in initial}
            assert any(value > 0 for value in changes.values()), 'No trainable parameter changed'
            if spec['family'] not in ('lora', 'direct_nlinear'):
                for key in ('encoder.weight', 'decoder.weight'):
                    assert (changes[key] == 0) == (spec['ed_mode'] == 'fixed_ed'), f'{key} freeze/update mismatch'
            if spec['family'] in ('level_res', 'old_res', 'level_raw'):
                assert changes['residual.down.weight'] > 0 and changes['residual.up.weight'] > 0
            if spec['family'] == 'direct_nlinear':
                assert changes['temporal.weight'] > 0 and changes['temporal.bias'] > 0
            assert all(p.grad is None and p._version == frozen_versions[n]
                       for n, p in model.named_parameters() if not p.requires_grad)
            assert state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad}) == frozen_hash
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'single training model, optimizer, initial probe and TRAIN/VAL; before restore'}
            total_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            saved_restart = torch.load(local / 'restart.pt', map_location='cpu', weights_only=False)
            assert_nested_equal(model.adapter_state(), saved_restart['state'])
            assert_nested_equal(opt.state_dict(), saved_restart['optimizer'])
            assert_nested_equal(scheduler.state_dict(), saved_restart['scheduler'])
            restored_opt, restored_scheduler = optimizer_for(model, spec)
            restored_opt.load_state_dict(saved_restart['optimizer'])
            restored_scheduler.load_state_dict(saved_restart['scheduler'])
            assert_nested_equal(opt.state_dict(), restored_opt.state_dict())
            assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
            before_rng = rng_state()
            restore_rng(saved_restart['rng'])
            assert_nested_equal(rng_state(), saved_restart['rng'])
            restore_rng(before_rng)
            del model, opt, scheduler, restored_opt, restored_scheduler, saved_restart, x, y, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
            restored = restore_model(local / 'best.pt', device='cuda')
            replay_score, replay = evaluate(restored, data, data['val_origins'], gpu_job, spec['id'] + ' selected replay')
            with np.load(local / 'best_val.npz', allow_pickle=False) as archive:
                assert np.array_equal(archive['origins'], data['val_origins'])
                replay_error = float(np.abs(replay - archive['prediction']).max())
            assert replay_error <= 1e-6 and abs(replay_score['mse'] - best) <= 1e-8
            selected_train, _ = evaluate(restored, data, probe, gpu_job, spec['id'] + ' selected TRAIN probe')
            selected_state = restored.adapter_state()
            result = dict(attempt, status='complete', best_val_mse=best, selected_epoch=best_epoch,
                          checkpoint=str(local / 'best.pt'), checkpoint_sha256=digest(local / 'best.pt'),
                          best_val_prediction={'path': str(local / 'best_val.npz'), 'sha256': digest(local / 'best_val.npz')},
                          initial_source=receipt, initial_parameter_sha256=initial_hash,
                          initial_checkpoint=str(local / 'initial.pt'), initial_checkpoint_sha256=digest(local / 'initial.pt'),
                          first_update_gradients=first_gradients, parameter_max_updates=changes,
                          changed_parameter_scalars=changed_counts,
                          selected_parameter_max_updates={k: float((selected_state[k] - initial[k]).abs().max()) for k in initial},
                          frozen_parameters_verified={'no_gradient': True, 'version_unchanged': True, 'optimizer_exclusion': True, 'frozen_tensor_sha256': frozen_hash},
                          initial_train_probe=initial_train, selected_train_probe=selected_train,
                          initial_val_mse=history[0]['val_mse'], epochs_completed=epoch, total_steps=steps,
                          attempt_updates=steps - start_steps, final_stale=stale,
                          extension_eligible=(epoch == 120 and stale < 6 and epoch - best_epoch < 6),
                          restart_checkpoint=str(local / 'restart.pt'), restart_sha256=digest(local / 'restart.pt'),
                          restart_roundtrip={'model_serialization': True, 'optimizer_serialization': True,
                                             'scheduler_serialization': True, 'optimizer_fresh_instance_restore': True,
                                             'scheduler_fresh_instance_restore': True, 'rng_restore': True, 'additional_updates': 0},
                          replay_error=replay_error, total_parameters=total_parameters, trainable_parameters=trainable_parameters,
                          training_peak=peak, schedules=schedules, elapsed_s=time.perf_counter() - started)
            save_json(out / 'result.json', result)
            save_json(out / 'receipt.json', result_receipt(result, out))
            attempt.update(status='complete', elapsed_s=result['elapsed_s'], actual_updates=steps - start_steps)
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('complete, selected checkpoint replayed', steps - start_steps)
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', elapsed_s=time.perf_counter() - started,
                           actual_updates=steps - start_steps, error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('failed', steps - start_steps)
            raise


def deterministic_baseline(family, data, job):
    if data['_dataset'] != 'jena':
        raise ValueError('Robin references are reused, never rerun here')
    run_id = 'v7_jena_' + family
    out, local = HERE / 'runs' / run_id, CACHE / 'runs' / run_id
    if out.exists() or local.exists():
        raise RuntimeError('Existing deterministic run: preserve it')
    spec = {'id': run_id, 'dataset': 'jena', 'family': family, 'arm': family,
            'ed_mode': 'fixed_ed', 'seed': None, 'latent': 6, 'channels': 21, 'normalization': 'context'}
    initial, receipt = initial_for('jena', None, family)
    model = build_model(spec, data['basis'], initial, device='cuda')
    out.mkdir(parents=True)
    local.mkdir(parents=True)
    val, prediction = evaluate(model, data, data['val_origins'], job, run_id + ' VAL')
    torch.save(payload(model, spec, data['basis'], receipt, 0, 0), local / 'best.pt')
    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
    result = {'id': run_id, 'spec': spec, 'status': 'complete', 'deterministic': True, 'best_val_mse': val['mse'],
              'selected_epoch': 0, 'checkpoint': str(local / 'best.pt'), 'checkpoint_sha256': digest(local / 'best.pt'),
              'best_val_prediction': {'path': str(local / 'best_val.npz'), 'sha256': digest(local / 'best_val.npz')},
              'total_parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': 0,
              'data_sha256': digest(data['_path']), 'source': source_receipt()}
    save_json(out / 'result.json', result)
    save_json(out / 'receipt.json', result_receipt(result, out))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def select(specs, dataset, filename):
    target = HERE / filename
    if target.exists():
        raise RuntimeError('Selection exists; preserve the original')
    receipts, selected, results = [], {}, []
    for spec in specs:
        out = HERE / 'runs' / spec['id']
        result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        assert result['status'] == 'complete' and result['spec'] == spec
        assert digest(result['checkpoint']) == result['checkpoint_sha256']
        assert digest(result['best_val_prediction']['path']) == result['best_val_prediction']['sha256']
        results.append(result)
        receipts.append(result_receipt(result, out))
    expected = {('direct_nlinear', seed, lr) for seed in (92601, 92602) for lr in (1e-3, 1e-4)}
    if dataset == 'jena':
        expected |= {(family, seed, 1e-4 if family == 'lora' else 1e-3)
                     for family in ('level_res', 'old_res', 'level_raw', 'lora') for seed in (92601, 92602)}
    assert len(receipts) == len(expected) and len({r['id'] for r in receipts}) == len(expected)
    assert {(r['spec']['family'], r['spec']['seed'], r['spec']['lr']) for r in receipts} == expected
    matching_checks = []
    for seed in (92601, 92602):
        seeded = [r for r in results if r['spec']['seed'] == seed]
        known_schedule = {}
        for result in seeded:
            for schedule in result['schedules']:
                epoch = schedule['epoch']
                if epoch in known_schedule:
                    assert known_schedule[epoch] == schedule['sha256']
                known_schedule[epoch] = schedule['sha256']
        direct = [r for r in seeded if r['spec']['family'] == 'direct_nlinear']
        assert len(direct) == 2 and direct[0]['initial_parameter_sha256'] == direct[1]['initial_parameter_sha256']
        adapters = [r for r in seeded if r['spec']['family'] in ('level_res', 'old_res', 'level_raw')]
        if adapters:
            assert all(r['initial_parameter_sha256'] == adapters[0]['initial_parameter_sha256'] for r in adapters)
        matching_checks.append({'seed': seed, 'common_epoch_order_hashes_checked': len(known_schedule),
                                'same_direct_initial_values_across_lrs': True,
                                'same_edg_initial_values_across_arms': bool(adapters)})
    deterministic = {}
    for family in (() if dataset == 'robin' else ('f0', 'compress', 'level_only')):
        out = HERE / 'runs' / ('v7_jena_' + family)
        row = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        assert row['status'] == 'complete'
        deterministic[family] = result_receipt(row, out)
    for family in sorted({r['spec']['family'] for r in receipts}):
        rows = sorted((r for r in receipts if r['spec']['family'] == family), key=lambda r: r['spec']['seed'])
        lr_scores = None
        if family == 'direct_nlinear':
            lr_scores = [{'lr': lr, 'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in rows if r['spec']['lr'] == lr]))}
                         for lr in (1e-3, 1e-4)]
            chosen_lr = min(lr_scores, key=lambda row: row['mean_best_val_mse'])['lr']
            rows = [r for r in rows if r['spec']['lr'] == chosen_lr]
        assert [r['spec']['seed'] for r in rows] == [92601, 92602]
        selected[family] = {'ed_mode': rows[0]['spec']['ed_mode'], 'lr': rows[0]['spec']['lr'],
                            'run_ids': [r['id'] for r in rows],
                            'checkpoints': [r['checkpoint'] for r in rows],
                            'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in rows])),
                            'lr_selection': lr_scores}
    save_json(target, {'schema_version': 1, 'dataset': dataset, 'selected': selected, 'all_neural_runs': receipts,
                       'deterministic_models': deterministic,
                       'test_used_for_selection': False,
                       'matching_checks': matching_checks,
                       'criterion': 'per-seed minimum full VAL MSE including step0, earliest tie; DIRECT LR mean two seeds, ties first 1e-3; no TEST access'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specs', default=str(HERE / 'protocol.json'))
    parser.add_argument('--dataset', choices=tuple(DATASETS), required=True)
    parser.add_argument('--id')
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    parser.add_argument('--prepare-initial', action='store_true')
    parser.add_argument('--references', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--selection')
    args = parser.parse_args()
    if sum((args.prepare_initial, args.references, args.select_only, args.id is not None)) != 1:
        parser.error('Choose exactly one action: --id, --prepare-initial, --references, --select-only')
    configure()
    if args.prepare_initial:
        prepare_initial(args.job, args.dataset)
        return
    source = json.loads(Path(args.specs).read_text(encoding='utf-8'))
    raw_specs = source['fits'] if isinstance(source, dict) else source
    default_epochs = source.get('epochs', 120) if isinstance(source, dict) else 120
    specs = [dict(spec, epochs=spec.get('epochs', default_epochs)) for spec in raw_specs if spec['dataset'] == args.dataset]
    for spec in specs:
        validate_spec(spec)
    if args.select_only:
        with Job(args.job, category='cpu_analysis', reserve_seconds=30, metadata={'dataset': args.dataset}):
            select(specs, args.dataset, args.selection or f'selected_{args.dataset}.json')
        return
    with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds, metadata={'dataset': args.dataset}) as job:
        data = load_data(args.dataset, include_test=False)
        if args.references:
            for family in ('f0', 'compress', 'level_only'):
                deterministic_baseline(family, data, job)
        else:
            chosen = [spec for spec in specs if spec['id'] == args.id]
            if len(chosen) != 1:
                raise ValueError('Exactly one approved spec must match --id')
            fit(chosen[0], data, job)


if __name__ == '__main__':
    main()
