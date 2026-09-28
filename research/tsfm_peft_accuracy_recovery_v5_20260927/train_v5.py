"""One reserved v5 fit per invocation; TRAIN fitting and full-VAL selection only."""
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

from model_v5 import (DATASETS, build_model, initial_state, nested_basis, macro_loss,
                      restore_model, state_digest, tensor_hashes)
from runtime_v5 import (CACHE, HERE, ROOT, V2, V4, Job, digest, environment_receipt,
                        load_data, phase_sample, require_storage, save_json,
                        score_arrays, source_receipt, tensors)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def optimizer_for(model, spec):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': spec['lr'], 'name': 'ed_g'}], weight_decay=0)
    assert {id(p) for p in parameters} == {id(p) for group in optimizer.param_groups for p in group['params']}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, threshold=1e-4, threshold_mode='rel')
    return optimizer, scheduler


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


def decision_gate(spec):
    path = HERE / 'decision.json'
    decision = json.loads(path.read_text(encoding='utf-8'))
    allowed = {decision['first_method'], decision['reserve_method']}
    if len(allowed) != 2 or not allowed <= {'level', 'full', 'k'}:
        raise ValueError('Decision must name exactly one first candidate and one different reserve')
    if spec['method'] != decision['active_method'] or spec['method'] not in allowed:
        raise ValueError('Method is not the pre-fit authorized active candidate')
    if spec['dataset'] not in decision['approved_datasets']:
        raise ValueError('Dataset extension has not been released by the recorded decision')
    if spec['lr'] != 1e-3 or spec['epochs'] != 120 or spec.get('resume_from'):
        if spec['id'] not in decision.get('authorized_supplements', []):
            raise ValueError('Learning supplement/restart requires an explicit pre-fit decision record')
    return {'path': str(path), 'sha256': digest(path), 'decision': decision}


def validate_spec(spec):
    channels, base_latent, mode = DATASETS[spec['dataset']]
    latent = min(2 * base_latent, channels) if spec['method'] == 'k' else base_latent
    assert spec['method'] in ('level', 'full', 'k') and spec['variant'] in ('res', 'raw')
    assert (spec['channels'], spec['latent'], spec['ed_mode']) == (channels, latent, mode)
    assert spec['seed'] in (92601, 92602) and spec['lr'] in (1e-3, 1e-4)
    assert spec['epochs'] in (120, 240)
    assert spec['id'] and all(c.isalnum() or c in '_-' for c in spec['id'])
    if spec['epochs'] == 240:
        assert spec.get('resume_from')


def old_initial(dataset, seed, basis):
    if dataset == 'hog':
        manifest = json.loads((V4 / 'initial_manifest.json').read_text(encoding='utf-8'))
        receipt = manifest['records'][str(seed)]
    elif dataset == 'bull':
        run = json.loads((V2 / 'runs' / f'v2_bull_residual_fixed_ed_{seed}' / 'result.json').read_text(encoding='utf-8'))
        receipt = {'path': run['initial_checkpoint'], 'sha256': run['initial_checkpoint_sha256']}
    else:
        return None, {'kind': 'new_explicit_paired_seed_state',
                      'old_v1_initial_equivalence': 'unverified; v1 did not preserve complete initial tensors'}
    path = Path(receipt['path'])
    if digest(path) != receipt['sha256']:
        raise ValueError('Existing untrained initialization hash changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['epoch'] != 0 or saved['steps'] != 0:
        raise ValueError('Cannot warm-start with a learned adapter')
    state = saved['state']
    if not torch.equal(state['encoder.weight'], torch.as_tensor(basis).T) or not torch.equal(state['decoder.weight'], torch.as_tensor(basis)):
        raise ValueError('Historical PCA initialization differs from the reused TRAIN basis')
    if torch.count_nonzero(state['residual.up.weight']):
        raise ValueError('Historical G output layer is not the untrained zero state')
    return state, {'kind': 'verified_existing_untrained_initial', **receipt,
                   'tensor_hashes': tensor_hashes(state)}


def prepare_initial(spec, data):
    key = f"{spec['dataset']}_{spec['method']}_{spec['seed']}"
    metadata_path, path = HERE / 'initial' / (key + '.json'), CACHE / 'initial' / (key + '.pt')
    if metadata_path.exists():
        receipt = json.loads(metadata_path.read_text(encoding='utf-8'))
        if digest(path) != receipt['sha256'] or receipt['data_sha256'] != digest(data['_path']):
            raise ValueError('Existing shared initialization changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if tensor_hashes(saved['state']) != saved['tensor_hashes']:
            raise ValueError('Shared initialization tensors changed')
        return saved['state'], saved['basis'], receipt
    if path.exists():
        raise RuntimeError('Partial shared initialization exists; inspect and recover explicitly')
    started = time.perf_counter()
    basis = np.asarray(data['basis'], dtype=np.float32)
    historical, origin = old_initial(spec['dataset'], spec['seed'], basis)
    old_basis_hash = hashlib.sha256(basis.tobytes()).hexdigest()
    if spec['method'] == 'k':
        basis = nested_basis(data['x'][data['train_start']:data['train_end']], basis, spec['latent'])
    state = initial_state(basis, spec['seed'], spec['method'], historical,
                          base_latent=DATASETS[spec['dataset']][1])
    saved = {'state': state, 'basis': torch.as_tensor(basis), 'epoch': 0, 'steps': 0,
             'tensor_hashes': tensor_hashes(state), 'method': spec['method'], 'dataset': spec['dataset'],
             'seed': spec['seed'], 'old_basis_sha256': old_basis_hash, 'initialization_source': origin}
    require_storage(1024 * 1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    receipt = {'path': str(path), 'sha256': digest(path), 'tensor_hashes': saved['tensor_hashes'],
               'basis_sha256': hashlib.sha256(basis.tobytes()).hexdigest(), 'old_basis_sha256': old_basis_hash,
               'data_sha256': digest(data['_path']), 'source': origin,
               'preparation_elapsed_s': time.perf_counter() - started,
               'pca_statistic_count': int(spec['method'] == 'k'), 'predictive_fit_attempts': 0}
    save_json(metadata_path, receipt)
    return state, saved['basis'], receipt


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
                   'max_abs': None if p.grad is None else float(p.grad.detach().abs().max()),
                   'finite': None if p.grad is None else bool(torch.isfinite(p.grad).all())}
            for name, p in model.named_parameters() if not name.startswith('backbone.')}


def fit(spec, data, gpu_job):
    validate_spec(spec)
    decision = decision_gate(spec)
    out, local = HERE / 'runs' / spec['id'], CACHE / 'runs' / spec['id']
    if out.exists() or local.exists():
        raise RuntimeError('Preserve prior attempt; every retry/restart needs a new id')
    edg, basis, receipt = prepare_initial(spec, data)
    with Job(spec['id'], category='real_fit', reserve_seconds=0, metadata=spec) as fit_job:
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        started = time.perf_counter()
        steps = start_steps = 0
        attempt = {'id': spec['id'], 'spec': spec, 'status': 'running', 'ledger_id': fit_job.id,
                   'source': source_receipt(), 'environment': environment_receipt(),
                   'data_sha256': digest(data['_path']), 'decision_receipt': decision}
        save_json(out / 'attempt.json', attempt)
        try:
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            model = build_model(spec, basis, edg, device='cuda')
            initial = model.adapter_state()
            initial_hashes = tensor_hashes(initial)
            assert initial_hashes == receipt['tensor_hashes']
            frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
            frozen_hash = state_digest(model.backbone.state_dict())
            torch.save(payload(model, spec, basis, receipt, 0, 0), local / 'initial.pt')
            torch.cuda.reset_peak_memory_stats()
            probe = phase_sample(data['train_origins'], spec['seed'], 0, 96)
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            initial_train, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' initial TRAIN probe')
            opt, scheduler = optimizer_for(model, spec)
            history, schedules, first_gradients = [], [], {}
            best, best_epoch, stale, first_epoch = float('inf'), 0, 0, 0
            if spec.get('resume_from'):
                restart_path = Path(spec['resume_from'])
                restart = torch.load(restart_path, map_location='cpu', weights_only=False)
                for key in ('dataset', 'seed', 'method', 'variant', 'ed_mode', 'latent', 'lr'):
                    assert restart['spec'][key] == spec[key]
                assert restart['initial_sha256'] == receipt['sha256']
                if spec['epochs'] == 240:
                    assert restart['epoch'] == 120 and restart['stale'] < 6 and 120 - restart['best_epoch'] < 6
                model.restore_adapter(restart['state'])
                opt.load_state_dict(restart['optimizer'])
                scheduler.load_state_dict(restart['scheduler'])
                restore_rng(restart['rng'])
                steps = start_steps = restart['steps']
                best, best_epoch, stale = restart['best'], restart['best_epoch'], restart['stale']
                first_epoch, history, schedules = restart['epoch'] + 1, restart['history'], restart['schedules']
                parent_best = torch.load(restart_path.parent / 'best.pt', map_location='cpu', weights_only=False)
                parent_best['spec'] = dict(spec)
                torch.save(parent_best, local / 'best.pt')
                with np.load(restart_path.parent / 'best_val.npz', allow_pickle=False) as archive:
                    np.savez_compressed(local / 'best_val.npz', **{k: archive[k] for k in archive.files})
                frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
            for epoch in range(first_epoch, spec['epochs'] + 1):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512)
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if steps % 10 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}, batch {start // 4}', steps-start_steps)
                        opt.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = macro_loss(model(x), y, mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if steps - start_steps < 2:
                            first_gradients[str(steps-start_steps+1)] = gradient_record(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        opt.step()
                        steps += 1
                        losses.append(float(loss.detach()))
                val, prediction = evaluate(model, data, data['val_origins'], gpu_job, spec['id'] + f' VAL epoch {epoch}')
                history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'learning_rates': {group['name']: group['lr'] for group in opt.param_groups},
                                'elapsed_s': time.perf_counter()-started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    require_storage(16 * 1024**2)
                    torch.save(payload(model, spec, basis, receipt, epoch, steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                restart = payload(model, spec, basis, receipt, epoch, steps)
                restart.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                               stale=stale, best=best, best_epoch=best_epoch, history=history, schedules=schedules,
                               next_batch=0, boundary='post-VAL, post-scheduler, complete epoch')
                torch.save(restart, local / 'restart.pt')
                del restart
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}, best {best_epoch}', steps-start_steps)
                print(json.dumps({'id': spec['id'], 'epoch': epoch, 'best_epoch': best_epoch,
                                  'val_mse': val['mse'], 'elapsed_s': round(time.perf_counter()-started, 1)}), flush=True)
                if epoch and stale >= 6:
                    break
            final = model.adapter_state()
            changes = {key: float((final[key]-initial[key]).abs().max()) for key in initial}
            changed_counts = {key: int(torch.count_nonzero(final[key] != initial[key])) for key in initial}
            assert any(changes.values()), 'No trainable parameter changed'
            for key in ('encoder.weight', 'decoder.weight'):
                assert (changes[key] == 0) == (spec['ed_mode'] == 'fixed_ed')
            for key in initial:
                if key.startswith('residual.'):
                    assert changes[key] > 0, f'No actual update: {key}'
            assert all(p.grad is None and p._version == frozen_versions[name]
                       for name, p in model.named_parameters() if not p.requires_grad)
            assert state_digest(model.backbone.state_dict()) == frozen_hash
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'one model, optimizer and initial TRAIN probe/full VAL; before selected restore'}
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
                replay_error = float(np.abs(replay-archive['prediction']).max())
                np.testing.assert_allclose(replay, archive['prediction'], atol=1e-6, rtol=1e-5)
            assert abs(replay_score['mse']-best) <= 1e-8
            selected_train, _ = evaluate(restored, data, probe, gpu_job, spec['id'] + ' selected TRAIN probe')
            selected_state = restored.adapter_state()
            result = dict(attempt, status='complete', best_val_mse=best, selected_epoch=best_epoch,
                          checkpoint=str(local / 'best.pt'), checkpoint_sha256=digest(local / 'best.pt'),
                          best_val_prediction={'path': str(local / 'best_val.npz'), 'sha256': digest(local / 'best_val.npz')},
                          initial_source=receipt, initial_parameter_sha256=initial_hashes,
                          initial_checkpoint=str(local / 'initial.pt'), initial_checkpoint_sha256=digest(local / 'initial.pt'),
                          first_update_gradients=first_gradients, parameter_max_updates=changes,
                          changed_parameter_scalars=changed_counts,
                          selected_parameter_max_updates={key: float((selected_state[key]-initial[key]).abs().max()) for key in initial},
                          frozen_parameters_verified={'no_gradient': True, 'version_unchanged': True,
                                                      'backbone_sha256_unchanged': frozen_hash, 'optimizer_exclusion': True},
                          initial_train_probe=initial_train, selected_train_probe=selected_train,
                          initial_val_mse=history[0]['val_mse'], epochs_completed=epoch, total_steps=steps,
                          attempt_updates=steps-start_steps, final_stale=stale,
                          extension_eligible=(epoch == 120 and stale < 6 and epoch-best_epoch < 6),
                          restart_checkpoint=str(local / 'restart.pt'), restart_sha256=digest(local / 'restart.pt'),
                          restart_roundtrip={'model': True, 'optimizer': True, 'scheduler': True, 'rng': True, 'extra_updates': 0},
                          replay_error=replay_error, total_parameters=total_parameters, trainable_parameters=trainable_parameters,
                          training_peak=peak, schedules=schedules, elapsed_s=time.perf_counter()-started)
            save_json(out / 'result.json', result)
            save_json(out / 'receipt.json', {'id': spec['id'], 'result_sha256': digest(out / 'result.json'),
                                           'checkpoint': result['checkpoint'], 'checkpoint_sha256': result['checkpoint_sha256'],
                                           'curve_sha256': digest(out / 'curve.json'), 'initial': receipt,
                                           'best_val_prediction': result['best_val_prediction']})
            attempt.update(status='complete', elapsed_s=result['elapsed_s'], actual_updates=steps-start_steps)
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('complete; selected checkpoint replayed', steps-start_steps)
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', elapsed_s=time.perf_counter()-started,
                           actual_updates=steps-start_steps, error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('failed', steps-start_steps)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASETS))
    parser.add_argument('--method', required=True, choices=('level', 'full', 'k'))
    parser.add_argument('--variant', required=True, choices=('res', 'raw'))
    parser.add_argument('--seed', type=int, required=True, choices=(92601, 92602))
    parser.add_argument('--name', required=True)
    parser.add_argument('--job-label', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    channels, latent, mode = DATASETS[args.dataset]
    if args.method == 'k':
        latent = min(2 * latent, channels)
    spec = {'id': args.name, 'dataset': args.dataset, 'method': args.method, 'variant': args.variant,
            'seed': args.seed, 'channels': channels, 'latent': latent, 'ed_mode': mode,
            'lr': args.lr, 'epochs': args.epochs}
    if args.resume_from:
        spec['resume_from'] = args.resume_from
    configure()
    validate_spec(spec)
    decision_gate(spec)
    with Job(args.job_label, category='gpu', reserve_seconds=args.reserve_seconds, metadata=spec) as job:
        data = load_data(args.dataset, evaluation=False)
        fit(spec, data, job)


if __name__ == '__main__':
    main()
