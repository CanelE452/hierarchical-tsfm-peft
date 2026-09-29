"""Head-only v9 fitting and finite VAL selection; no TEST access or automatic retries."""
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

from model_v9 import (ALPHAS, DATASETS, NEURAL_FAMILIES, build_model, initial_head,
                      macro_loss, parent_receipt_for, resolve, restore_model,
                      state_digest, tensor_hashes)
from runtime_v9 import (CACHE, HERE, Job, digest, environment_receipt, load_data,
                        phase_sample, require_storage, save_json, score_arrays,
                        source_receipt, tensors)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def optimizer_for(model, spec):
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in parameters} == {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW([{'params': parameters, 'lr': spec['lr'], 'name': 'new_head'}], weight_decay=0)
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


def before_test():
    if any(HERE.glob('test_exposure_*.json')):
        raise RuntimeError('No v9 initialization, fitting or selection after either TEST exposure')


def prepare_initial(name, dataset):
    before_test()
    target = HERE / f'initial_manifest_{dataset}.json'
    if target.exists():
        raise RuntimeError('Initial manifest exists; preserve it')
    with Job(name, category='cpu_analysis', reserve_seconds=60, metadata={'dataset': dataset}) as job:
        data = load_data(dataset, include_test=False)
        basis = torch.as_tensor(data['basis'], dtype=torch.float32)
        assert tuple(basis.shape) == DATASETS[dataset]
        records = {'head': {}}
        for seed in (92601, 92602):
            path = CACHE / 'initial' / dataset / f'head_seed_{seed}.pt'
            if path.exists():
                raise RuntimeError('Shared head initialization already exists')
            path.parent.mkdir(parents=True, exist_ok=True)
            state = initial_head(dataset, seed)
            parent = parent_receipt_for(dataset, seed)
            hashes = tensor_hashes(state)
            torch.save({'state': state, 'basis': basis, 'seed': seed, 'dataset': dataset,
                        'kind': 'head', 'epoch': 0, 'steps': 0, 'tensor_hashes': hashes,
                        'parent_receipt': parent, 'source_initial': parent['head_initial'],
                        'data_sha256': digest(data['_path'])}, path)
            records['head'][str(seed)] = {'path': str(path), 'sha256': digest(path), 'tensor_hashes': hashes,
                                         'source_initial': parent['head_initial']}
        save_json(target, {'schema_version': 1, 'dataset': dataset, 'records': records,
                          'data_sha256': digest(data['_path']), 'source': source_receipt(),
                          'recipe': 'Copy untrained v8 retained head into head.*; identical initial state for P and RAW'})
        job.heartbeat('shared new-head initialization saved', 0)


def initial_for(dataset, seed):
    manifest = json.loads((HERE / f'initial_manifest_{dataset}.json').read_text(encoding='utf-8'))
    receipt = manifest['records']['head'][str(seed)]
    assert digest(receipt['path']) == receipt['sha256']
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    assert saved['dataset'] == dataset and saved['seed'] == seed and saved['kind'] == 'head'
    assert saved['epoch'] == saved['steps'] == 0
    assert tensor_hashes(saved['state']) == receipt['tensor_hashes'] == saved['tensor_hashes']
    assert saved['parent_receipt'] == parent_receipt_for(dataset, seed)
    return saved, receipt


def validate_spec(spec):
    assert spec['dataset'] in DATASETS and spec['seed'] in (92601, 92602)
    assert (spec['channels'], spec['latent']) == DATASETS[spec['dataset']]
    assert spec['family'] in NEURAL_FAMILIES and spec['arm'] == spec['family']
    assert spec['ed_mode'] == 'fixed_ed' and spec['lr'] == 1e-3 and spec['epochs'] == 120
    assert spec.get('normalization') == 'context'
    if spec.get('resume_from'):
        raise ValueError('Recovery requires a separately recorded explicit retry; no automatic continuation')
    assert spec['id'] and '..' not in spec['id'] and all(c.isalnum() or c in '_.-' for c in spec['id'])


def array_parity(actual, reference, atol=1e-5, rtol=1e-4):
    assert actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    violations = delta > atol + rtol * np.abs(reference)
    return {'pass': not bool(violations.any()), 'atol': atol, 'rtol': rtol,
            'max_absolute_difference': float(delta.max()), 'violating_elements': int(violations.sum()),
            'elements': int(delta.size), 'bitwise_equal': bool(np.array_equal(actual, reference))}


@torch.no_grad()
def evaluate(model, data, origins, job, label, integrity=False):
    model.eval()
    predictions = []
    checks = {'correction_q_max_abs': 0., 'correction_q_violating_elements': 0,
              'prediction_q_difference_max_abs': 0., 'prediction_q_violating_elements': 0,
              'full_parent_difference_max_abs': 0., 'elements': 0,
              'correction_q_atol': 1e-5, 'prediction_q_atol': 1e-5, 'prediction_q_rtol': 1e-4}
    for start in range(0, len(origins), 4):
        job.check_limits()
        x, _, _ = tensors(data, origins[start:start + 4])
        if integrity:
            parts = model.parts(x)
            prediction = parts['parent_output'] + parts['scaled_correction']
            correction = parts['correction']
            correction_q = correction - model.parent.decoder(model.parent.encoder(correction))
            parent_q = parts['parent_output'] - model.parent.decoder(model.parent.encoder(parts['parent_output']))
            prediction_q = prediction - model.parent.decoder(model.parent.encoder(prediction))
            delta = (prediction_q - parent_q).abs()
            checks['correction_q_max_abs'] = max(checks['correction_q_max_abs'], float(correction_q.abs().max()))
            checks['correction_q_violating_elements'] += int((correction_q.abs() > 1e-5).sum())
            checks['prediction_q_difference_max_abs'] = max(checks['prediction_q_difference_max_abs'], float(delta.max()))
            checks['prediction_q_violating_elements'] += int((delta > 1e-5 + 1e-4 * parent_q.abs()).sum())
            checks['full_parent_difference_max_abs'] = max(checks['full_parent_difference_max_abs'], float((prediction - parts['parent_output']).abs().max()))
            checks['elements'] += prediction.numel()
            if model.family == 'staged_p' and (checks['correction_q_violating_elements'] or checks['prediction_q_violating_elements']):
                raise RuntimeError('P correction failed the fixed Q-preservation tolerance')
        else:
            prediction = model(x)
        predictions.append(prediction.cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    job.heartbeat(label)
    return score_arrays(prediction, target, mask), prediction, checks if integrity else None


def payload(model, spec, basis, receipt, epoch, steps):
    return {'spec': dict(spec), 'basis': torch.as_tensor(basis).cpu(), 'state': model.adapter_state(),
            'parent_receipt': model.parent_receipt, 'alpha': model.alpha,
            'initial_path': receipt['path'], 'initial_sha256': receipt['sha256'],
            'model_config': model.model_config(), 'epoch': epoch, 'steps': steps}


def frozen_tensors(model):
    return dict(model.parent.named_parameters()) | {'buffer.' + n: b for n, b in model.parent.named_buffers()}


def frozen_check(model, original_hash, versions):
    assert not any(p.requires_grad or p.grad is not None for p in model.parent.parameters())
    assert all(not module.training for module in model.parent.modules())
    current = frozen_tensors(model)
    assert current.keys() == versions.keys()
    assert all(t._version == versions[name] for name, t in current.items())
    assert state_digest(current) == original_hash


def parent_reference(model, data, prediction, score):
    receipt = model.parent_receipt['result']
    path = resolve(receipt['path'])
    assert digest(path) == receipt['sha256']
    result = json.loads(path.read_text(encoding='utf-8'))
    source = result['best_val_prediction']
    saved_path = resolve(source['path'])
    assert digest(saved_path) == source['sha256']
    with np.load(saved_path, allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], data['val_origins'])
        check = array_parity(prediction, saved['prediction'])
        recorded = score_arrays(saved['prediction'], np.stack([data['x'][o:o + 48] for o in data['val_origins']]),
                                np.stack([data['finite'][o:o + 48] for o in data['val_origins']]))
    assert abs(recorded['mse'] - result['best_val_mse']) <= 1e-8
    assert check['pass'], 'Initial output differs from the selected trained parent'
    return {'prediction': source, 'result': receipt, 'parity': check, 'historical_best_val_mse': result['best_val_mse'],
            'initial_val_mse': score['mse'], 'mse_difference': score['mse'] - result['best_val_mse'],
            'parent_epoch': result['selected_epoch'], 'parent_training_elapsed_s': result['elapsed_s']}


def checkpoint_receipt(local, name, epoch, score):
    return {'checkpoint': str(local / f'{name}.pt'), 'checkpoint_sha256': digest(local / f'{name}.pt'),
            'best_val_prediction': {'path': str(local / f'{name}_val.npz'), 'sha256': digest(local / f'{name}_val.npz')},
            'selected_epoch': epoch, 'val_mse': score}


def result_receipt(result, out):
    keys = ('id', 'spec', 'checkpoint', 'checkpoint_sha256', 'best_val_prediction', 'best_val_mse',
            'selected_epoch', 'best_trained', 'alpha_selected', 'parent_receipt')
    return {key: result[key] for key in keys} | {'result_path': str(out / 'result.json'),
            'result_sha256': digest(out / 'result.json'), 'deterministic': False}


def fit(spec, data, gpu_job):
    before_test()
    validate_spec(spec)
    out, local = HERE / 'runs' / spec['id'], CACHE / 'runs' / spec['id']
    if out.exists() or local.exists():
        raise RuntimeError('Existing attempt: preserve it and use an explicit new retry id')
    with Job(spec['id'], category='real_fit', reserve_seconds=0, metadata=spec) as fit_job:
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        started = time.perf_counter()
        attempt = {'id': spec['id'], 'spec': spec, 'status': 'running', 'ledger_id': fit_job.id,
                   'source': source_receipt(), 'environment': environment_receipt(), 'data_sha256': digest(data['_path'])}
        save_json(out / 'attempt.json', attempt)
        steps = 0
        try:
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            saved_initial, receipt = initial_for(spec['dataset'], spec['seed'])
            assert torch.equal(saved_initial['basis'], torch.as_tensor(data['basis']))
            model = build_model(spec, data['basis'], saved_initial['state'], device='cuda')
            initial = model.adapter_state()
            initial_hash = tensor_hashes(initial)
            frozen_versions = {n: t._version for n, t in frozen_tensors(model).items()}
            frozen_hash = state_digest(frozen_tensors(model))
            parent_receipt = model.parent_receipt
            torch.save(payload(model, spec, data['basis'], receipt, 0, 0), local / 'initial.pt')
            torch.cuda.reset_peak_memory_stats()
            probe = phase_sample(data['train_origins'], spec['seed'], 0, max(96, data['_phase_period']), period=data['_phase_period'])
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            initial_train, _, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' initial TRAIN probe')
            opt, scheduler = optimizer_for(model, spec)
            history, schedules, first_gradients = [], [], {}
            best, best_epoch, trained_best, trained_epoch, stale = float('inf'), 0, float('inf'), None, 0
            online_fit_started = time.perf_counter()
            for epoch in range(spec['epochs'] + 1):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512, period=data['_phase_period'])
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    assert all(not module.training for module in model.parent.modules())
                    for start in range(0, len(schedule), 4):
                        if steps % 10 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}, batch {start // 4}', steps)
                        opt.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = macro_loss(model(x), y, mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if steps < 2:
                            first_gradients[str(steps + 1)] = {n: {'requires_grad': p.requires_grad,
                                'max_abs': None if p.grad is None else float(p.grad.detach().abs().max())}
                                for n, p in model.named_parameters() if p.requires_grad}
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., error_if_nonfinite=True)
                        opt.step()
                        steps += 1
                        losses.append(float(loss.detach()))
                val, prediction, preservation = evaluate(model, data, data['val_origins'], gpu_job,
                                                         spec['id'] + f' VAL epoch {epoch}', integrity=epoch == 0)
                if epoch == 0:
                    assert preservation['full_parent_difference_max_abs'] == 0
                    initial_parent_preservation = preservation
                    parent_validation = parent_reference(model, data, prediction, val)
                    parent_prediction = prediction.copy()
                history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'], 'alpha': 1.,
                                'learning_rates': {g['name']: g['lr'] for g in opt.param_groups},
                                'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    require_storage(16 * 1024**2)
                    torch.save(payload(model, spec, data['basis'], receipt, epoch, steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=prediction, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch and val['mse'] < trained_best:
                    trained_best, trained_epoch = val['mse'], epoch
                    torch.save(payload(model, spec, data['basis'], receipt, epoch, steps), local / 'best_trained.pt')
                    np.savez_compressed(local / 'best_trained_val.npz', prediction=prediction, origins=data['val_origins'])
                if epoch:
                    scheduler.step(val['mse'])
                restart = payload(model, spec, data['basis'], receipt, epoch, steps)
                restart.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                               stale=stale, best=best, best_epoch=best_epoch, trained_best=trained_best if epoch else None,
                               trained_epoch=trained_epoch, history=history, schedules=schedules, next_batch=0,
                               boundary='post-VAL, post-scheduler, complete epoch')
                torch.save(restart, local / 'restart.pt')
                del restart
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}, best {best_epoch}, trained {trained_epoch}', steps)
                print(json.dumps({'id': spec['id'], 'epoch': epoch, 'best_epoch': best_epoch,
                                  'best_trained_epoch': trained_epoch, 'val_mse': val['mse'],
                                  'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
                if epoch and stale >= 6:
                    break
            online_fit_elapsed = time.perf_counter() - online_fit_started
            final = model.adapter_state()
            changes = {k: float((final[k] - initial[k]).abs().max()) for k in initial}
            changed_counts = {k: int(torch.count_nonzero(final[k] != initial[k])) for k in initial}
            assert all(value > 0 for value in changes.values()), 'A new head tensor never updated'
            assert steps >= 2 and trained_epoch is not None
            frozen_check(model, frozen_hash, frozen_versions)
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'single online parent+head model and head optimizer; initial probe/TRAIN/VAL before model reload'}
            total_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable_parameters == 17920
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
            replay_score, replay, selected_preservation = evaluate(restored, data, data['val_origins'], gpu_job,
                                                                   spec['id'] + ' basic selected replay', integrity=True)
            with np.load(local / 'best_val.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['origins'], data['val_origins'])
                replay_check = array_parity(replay, saved['prediction'], atol=1e-6, rtol=0)
            assert replay_check['pass'] and abs(replay_score['mse'] - best) <= 1e-8
            selected_train, _, _ = evaluate(restored, data, probe, gpu_job, spec['id'] + ' basic selected TRAIN probe')
            selected_state = restored.adapter_state()
            assert state_digest(frozen_tensors(restored)) == frozen_hash
            del restored
            gc.collect()
            torch.cuda.empty_cache()
            trained = restore_model(local / 'best_trained.pt', device='cuda')
            alpha_started = time.perf_counter()
            alpha_candidates = []
            chosen_alpha, alpha_best = None, float('inf')
            trained_replay = None
            for alpha in ALPHAS:
                trained.set_alpha(alpha)
                score, prediction, preservation = evaluate(trained, data, data['val_origins'], gpu_job,
                                                            spec['id'] + f' best-trained alpha {alpha}', integrity=True)
                alpha_candidates.append({'alpha': alpha, 'val_mse': score['mse'], 'val_mae': score['mae'],
                                         'preservation': preservation, 'checkpoint_epoch': trained_epoch})
                if alpha == 0:
                    assert array_parity(prediction, parent_prediction)['pass']
                if alpha == 1:
                    with np.load(local / 'best_trained_val.npz', allow_pickle=False) as saved:
                        assert np.array_equal(saved['origins'], data['val_origins'])
                        trained_replay = array_parity(prediction, saved['prediction'], atol=1e-6, rtol=0)
                    assert trained_replay['pass'] and abs(score['mse'] - trained_best) <= 1e-8
                if score['mse'] < alpha_best:
                    alpha_best, chosen_alpha = score['mse'], alpha
                    torch.save(payload(trained, spec, data['basis'], receipt, trained_epoch,
                                       history[trained_epoch]['steps']), local / 'alpha_selected.pt')
                    np.savez_compressed(local / 'alpha_selected_val.npz', prediction=prediction, origins=data['val_origins'])
            alpha_elapsed = time.perf_counter() - alpha_started
            assert state_digest(frozen_tensors(trained)) == frozen_hash
            best_trained = checkpoint_receipt(local, 'best_trained', trained_epoch, trained_best)
            alpha_selection = {'schema_version': 1, 'id': spec['id'], 'dataset': spec['dataset'],
                               'family': spec['family'], 'seed': spec['seed'], 'alpha': chosen_alpha,
                               'checkpoint': str(local / 'alpha_selected.pt'), 'checkpoint_sha256': digest(local / 'alpha_selected.pt'),
                               'val_prediction': {'path': str(local / 'alpha_selected_val.npz'), 'sha256': digest(local / 'alpha_selected_val.npz')},
                               'val_mse': alpha_best, 'candidates': alpha_candidates, 'grid': list(ALPHAS),
                               'best_trained_checkpoint': best_trained, 'test_used_for_selection': False,
                               'selection_rule': 'One best-trained alpha1 epoch>=1 checkpoint; minimum whole VAL MSE across fixed alpha grid; exact ties smaller alpha',
                               'alpha0_means': 'trained-parent fallback, not added prediction efficacy'}
            save_json(out / 'alpha_selected.json', alpha_selection)
            alpha_selected = {key: alpha_selection[key] for key in ('alpha', 'checkpoint', 'checkpoint_sha256', 'val_prediction',
                              'val_mse', 'best_trained_checkpoint', 'selection_rule')}
            alpha_selected.update(selection_path=str(out / 'alpha_selected.json'), selection_sha256=digest(out / 'alpha_selected.json'))
            del trained
            gc.collect()
            torch.cuda.empty_cache()
            selected_alpha_model = restore_model(local / 'alpha_selected.pt', device='cuda')
            assert selected_alpha_model.alpha == chosen_alpha
            alpha_score, alpha_prediction, alpha_preservation = evaluate(selected_alpha_model, data, data['val_origins'], gpu_job,
                                                                          spec['id'] + ' selected alpha replay', integrity=True)
            with np.load(local / 'alpha_selected_val.npz', allow_pickle=False) as saved:
                alpha_replay = array_parity(alpha_prediction, saved['prediction'], atol=1e-6, rtol=0)
            assert alpha_replay['pass'] and abs(alpha_score['mse'] - alpha_best) <= 1e-8
            assert state_digest(frozen_tensors(selected_alpha_model)) == frozen_hash
            result = dict(attempt, status='complete', best_val_mse=best, selected_epoch=best_epoch,
                          checkpoint=str(local / 'best.pt'), checkpoint_sha256=digest(local / 'best.pt'),
                          best_val_prediction={'path': str(local / 'best_val.npz'), 'sha256': digest(local / 'best_val.npz')},
                          best_trained=best_trained, alpha_selected=alpha_selected, alpha_candidates=alpha_candidates,
                          auxiliary_val_candidates=len(alpha_candidates), auxiliary_new_fits=0,
                          parent_receipt=parent_receipt, parent_validation=parent_validation,
                          initial_source=receipt, initial_parameter_sha256=initial_hash,
                          initial_checkpoint=str(local / 'initial.pt'), initial_checkpoint_sha256=digest(local / 'initial.pt'),
                          initial_parent_preservation=initial_parent_preservation, selected_preservation=selected_preservation,
                          alpha_selected_preservation=alpha_preservation,
                          first_update_gradients=first_gradients, parameter_max_updates=changes, changed_parameter_scalars=changed_counts,
                          selected_parameter_max_updates={k: float((selected_state[k] - initial[k]).abs().max()) for k in initial},
                          frozen_parameters_verified={'no_gradient': True, 'version_unchanged': True, 'optimizer_exclusion': True,
                              'parent_eval_mode': True, 'parameters_and_buffers': True, 'frozen_tensor_sha256': frozen_hash},
                          initial_train_probe=initial_train, selected_train_probe=selected_train,
                          initial_val_mse=history[0]['val_mse'], epochs_completed=epoch, total_steps=steps,
                          attempt_updates=steps, final_stale=stale,
                          cap_reached_while_improving=(epoch == 120 and stale < 6 and epoch - best_epoch < 6), extension_eligible=False,
                          restart_checkpoint=str(local / 'restart.pt'), restart_sha256=digest(local / 'restart.pt'),
                          restart_roundtrip={'model_serialization': True, 'optimizer_serialization': True,
                              'scheduler_serialization': True, 'optimizer_fresh_instance_restore': True,
                              'scheduler_fresh_instance_restore': True, 'rng_restore': True, 'additional_updates': 0},
                          replay_error=replay_check['max_absolute_difference'], replay=replay_check,
                          best_trained_replay=trained_replay, alpha_selected_replay=alpha_replay,
                          total_parameters=total_parameters, trainable_parameters=trainable_parameters,
                          cumulative_fitted_adapter_parameters=35840, training_peak=peak, schedules=schedules,
                          head_fit_online_elapsed_s=online_fit_elapsed, alpha_selection_elapsed_s=alpha_elapsed,
                          elapsed_s=time.perf_counter() - started,
                          timing_scope='Incremental v9 online parent+new-head fitting/VAL; parent historical fit is excluded, not a total-adaptation speedup')
            save_json(out / 'result.json', result)
            save_json(out / 'receipt.json', result_receipt(result, out))
            attempt.update(status='complete', elapsed_s=result['elapsed_s'], actual_updates=steps)
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('complete: basic, best-trained and selected alpha restored', steps)
            del selected_alpha_model
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', elapsed_s=time.perf_counter() - started, actual_updates=steps, error=traceback.format_exc())
            save_json(out / 'attempt.json', attempt)
            fit_job.heartbeat('failed', steps)
            raise


def select(specs, dataset, filename):
    before_test()
    target = HERE / filename
    if target.exists():
        raise RuntimeError('Selection exists; preserve it')
    expected = {(family, seed) for family in NEURAL_FAMILIES for seed in (92601, 92602)}
    results, receipts, selected, matching = [], [], {}, []
    for spec in specs:
        out = HERE / 'runs' / spec['id']
        result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        assert result['status'] == 'complete' and result['spec'] == spec
        for record in (result, result['best_trained'], result['alpha_selected']):
            assert digest(record['checkpoint']) == record['checkpoint_sha256']
        for record in (result['best_val_prediction'], result['best_trained']['best_val_prediction'], result['alpha_selected']['val_prediction']):
            assert digest(record['path']) == record['sha256']
        assert digest(result['alpha_selected']['selection_path']) == result['alpha_selected']['selection_sha256']
        assert result['trainable_parameters'] == 17920 and result['auxiliary_val_candidates'] == 5
        results.append(result)
        receipts.append(result_receipt(result, out))
    assert len(results) == 4 and {(r['spec']['family'], r['spec']['seed']) for r in results} == expected
    for seed in (92601, 92602):
        rows = [r for r in results if r['spec']['seed'] == seed]
        assert rows[0]['initial_source']['sha256'] == rows[1]['initial_source']['sha256']
        assert rows[0]['parent_receipt'] == rows[1]['parent_receipt']
        assert abs(rows[0]['initial_val_mse'] - rows[1]['initial_val_mse']) <= 1e-8
        orders = {}
        for row in rows:
            for schedule in row['schedules']:
                if schedule['epoch'] in orders:
                    assert orders[schedule['epoch']] == schedule['sha256']
                orders[schedule['epoch']] = schedule['sha256']
        matching.append({'seed': seed, 'same_parent': True, 'shared_initial_source': True,
                         'same_initial_val_score': True, 'common_epoch_order_hashes': len(orders)})
    for family in NEURAL_FAMILIES:
        rows = sorted([r for r in results if r['spec']['family'] == family], key=lambda r: r['spec']['seed'])
        selected[family] = {'ed_mode': 'fixed_ed', 'lr': 1e-3, 'run_ids': [r['id'] for r in rows],
                            'checkpoints': [r['checkpoint'] for r in rows], 'alphas': [1., 1.],
                            'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in rows])), 'policy': 'basic'}
        selected[family + '_alpha'] = {'ed_mode': 'fixed_ed', 'lr': 1e-3, 'run_ids': [r['id'] for r in rows],
                            'checkpoints': [r['alpha_selected']['checkpoint'] for r in rows],
                            'alphas': [r['alpha_selected']['alpha'] for r in rows],
                            'mean_best_val_mse': float(np.mean([r['alpha_selected']['val_mse'] for r in rows])),
                            'policy': 'one best-trained checkpoint plus finite VAL alpha grid'}
    save_json(target, {'schema_version': 1, 'dataset': dataset, 'selected': selected, 'all_neural_runs': receipts,
                       'deterministic_models': {}, 'test_used_for_selection': False, 'matching_checks': matching,
                       'real_fits': 4, 'auxiliary_new_fits': 0, 'auxiliary_val_candidates': 20,
                       'criterion': 'Basic per-seed whole VAL minimum including step0; auxiliary epoch>=1 best-trained plus fixed five-alpha VAL selection; all roles reported'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specs', default=str(HERE / 'protocol.json'))
    parser.add_argument('--dataset', choices=tuple(DATASETS), required=True)
    parser.add_argument('--id')
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    parser.add_argument('--prepare-initial', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--selection')
    args = parser.parse_args()
    if sum((args.prepare_initial, args.select_only, args.id is not None)) != 1:
        parser.error('Choose exactly one action: --id, --prepare-initial, --select-only')
    configure()
    before_test()
    if args.prepare_initial:
        prepare_initial(args.job, args.dataset)
        return
    source = json.loads(Path(args.specs).read_text(encoding='utf-8'))
    raw_specs = source['fits'] if isinstance(source, dict) else source
    specs = [dict(spec, epochs=spec.get('epochs', 120)) for spec in raw_specs if spec['dataset'] == args.dataset]
    for spec in specs:
        validate_spec(spec)
    if args.select_only:
        with Job(args.job, category='cpu_analysis', reserve_seconds=30, metadata={'dataset': args.dataset}):
            select(specs, args.dataset, args.selection or f'selected_{args.dataset}.json')
        return
    with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds, metadata={'dataset': args.dataset}) as job:
        data = load_data(args.dataset, include_test=False)
        chosen = [spec for spec in specs if spec['id'] == args.id]
        if len(chosen) != 1:
            raise ValueError('Exactly one approved spec must match --id')
        fit(chosen[0], data, job)


if __name__ == '__main__':
    main()
