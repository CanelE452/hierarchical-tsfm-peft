import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from model import LinearResidual, macro_loss
from runtime import (CACHE, HERE, ROOT, V2, Job, digest, environment_receipt, load_data,
                     phase_sample, require_storage, save_json, score_arrays, source_receipt, tensors)

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def optimizer_for(model, spec):
    temporal = list(model.temporal.parameters())
    rest = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('temporal.')]
    opt = torch.optim.AdamW([{'params': rest, 'lr': 1e-3, 'name': 'ed_g'},
                             {'params': temporal, 'lr': spec['t_lr'], 'name': 'temporal'}], weight_decay=0)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert expected == {id(p) for group in opt.param_groups for p in group['params']}
    return opt, torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)


def scheduler_step(scheduler, value, ratio):
    scheduler.step(value)
    groups = scheduler.optimizer.param_groups
    groups[1]['lr'] = groups[0]['lr'] * ratio
    scheduler._last_lr = [g['lr'] for g in groups]


def tensor_hashes(state):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k, v in state.items()}


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


@torch.no_grad()
def evaluate(model, data, origins, job, label):
    model.eval()
    preds = []
    for start in range(0, len(origins), 4):
        job.check_limits()
        x, _, _ = tensors(data, origins[start:start + 4])
        preds.append(model(x).cpu().numpy())
    pred = np.concatenate(preds)
    y = np.stack([data['x'][o:o+48] for o in origins])
    mask = np.stack([data['finite'][o:o+48] for o in origins])
    job.heartbeat(label)
    return score_arrays(pred, y, mask), pred


def initial_state_for(spec):
    parent_id = f'v2_{spec["dataset"]}_residual_fixed_ed_{spec["seed"]}'
    path = ROOT / '.cache/tsfm_peft_followup_v2_20260926/runs' / parent_id / 'initial.pt'
    result = json.loads((V2 / 'runs' / parent_id / 'result.json').read_text(encoding='utf-8'))
    assert digest(path) == result['initial_checkpoint_sha256']
    initial = torch.load(path, map_location='cpu', weights_only=False)
    assert initial['epoch'] == 0 and initial['steps'] == 0
    assert tensor_hashes(initial['state']) == result['initial_parameter_sha256']
    return initial['state'], {'path': str(path), 'sha256': digest(path), 'parent_run': parent_id,
                               'electricity_v1_bitwise_initial_verified': False}


def validate_spec(spec):
    assert spec['dataset'] in ('electricity', 'bull') and spec['seed'] in (92601, 92602)
    assert (spec['ed_mode'], spec['latent']) == ({'electricity': ('current', 8), 'bull': ('fixed_ed', 4)}[spec['dataset']])
    assert spec['arm'] in ('residual', 'raw_bypass') and spec['normalization'] in ('context', 'raw')
    assert spec['t_lr'] in (1e-3, 1e-4)
    assert spec['epochs'] in (120, 240)
    if spec['epochs'] == 240 or spec['arm'] != 'residual' or spec['normalization'] != 'context':
        assert (HERE / spec['decision_record']).exists()
    assert all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in spec['id'])


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for u, v in zip(a, b):
            assert_nested_equal(u, v)
    else:
        assert a == b


def fit(spec, data, gpu_job):
    validate_spec(spec)
    out, local = HERE / 'runs' / spec['id'], CACHE / 'runs' / spec['id']
    if out.exists() or local.exists():
        raise RuntimeError('Existing attempt: never overwrite; use explicit new retry id')
    with Job(spec['id'], category='real_fit', reserve_seconds=0, metadata=spec) as fit_job:
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        started = time.perf_counter()
        attempt = dict(spec, status='running', source=source_receipt(), environment=environment_receipt(),
                       data_sha256=digest(data['_path']), ledger_id=fit_job.id)
        save_json(out / 'attempt.json', attempt)
        steps, start_steps = 0, 0
        try:
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            init_edg, init_receipt = initial_state_for(spec)
            model = LinearResidual(spec, init_edg).cuda()
            torch.cuda.reset_peak_memory_stats()
            initial = model.adapter_state()
            initial_hash = tensor_hashes(initial)
            for other in (HERE / 'runs').glob('*/result.json'):
                old = json.loads(other.read_text(encoding='utf-8'))
                if all(old.get(k) == spec.get(k) for k in ('dataset', 'seed')):
                    assert old['initial_parameter_sha256'] == initial_hash, 'paired initialization differs'
            initial_payload = {'state': initial, 'spec': spec, 'model_config': model.model_config(),
                               'epoch': 0, 'steps': 0, 'source_initial': init_receipt}
            torch.save(initial_payload, local / 'initial.pt')
            probe = phase_sample(data['train_origins'], spec['seed'], 0, 96)
            np.savez_compressed(local / 'origins.npz', train_probe=probe, val=data['val_origins'])
            initial_train, _ = evaluate(model, data, probe, gpu_job, spec['id'] + ' initial TRAIN probe')
            x, y, mask = tensors(data, probe[:4])
            main, correction = model.components(x)
            assert correction.abs().max().item() == 0
            assert main.shape == y.shape == mask.shape
            model.zero_grad(set_to_none=True)
            macro_loss(main + correction, y, mask).backward()
            initial_grads = {n: None if p.grad is None else float(p.grad.abs().max()) for n,p in model.named_parameters()}
            assert all(initial_grads[n] is not None and initial_grads[n] > 0 for n in ('temporal.weight', 'temporal.bias'))
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad(set_to_none=True)
            del x, y, mask, main, correction
            opt, scheduler = optimizer_for(model, spec)
            history, best, best_epoch, stale, first_epoch = [], float('inf'), 0, 0, 0
            schedules = []
            if spec.get('resume_from'):
                resume_path = Path(spec['resume_from'])
                restart = torch.load(resume_path, map_location='cpu', weights_only=False)
                for k in ('dataset','seed','ed_mode','latent','arm','normalization','t_lr'):
                    assert restart['spec'][k] == spec[k]
                model.restore_adapter(restart['state'])
                opt.load_state_dict(restart['optimizer'])
                scheduler.load_state_dict(restart['scheduler'])
                restore_rng(restart['rng'])
                steps = start_steps = restart['steps']
                best, best_epoch, stale = restart['best'], restart['best_epoch'], restart['stale']
                first_epoch, history, schedules = restart['epoch'] + 1, restart['history'], restart['schedules']
                parent_best = torch.load(resume_path.parent / 'best.pt', map_location='cpu', weights_only=False)
                torch.save(dict(parent_best, spec=spec), local / 'best.pt')
                with np.load(resume_path.parent / 'best_val.npz') as a:
                    np.savez_compressed(local / 'best_val.npz', **{k:a[k] for k in a.files})
            update_grads = {}
            for epoch in range(first_epoch, spec['epochs'] + 1):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512)
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, len(schedule), 4):
                        if steps % 10 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}, batch {start//4}', steps-start_steps)
                        opt.zero_grad(set_to_none=True)
                        x, y, mask = tensors(data, schedule[start:start + 4])
                        loss = macro_loss(model(x), y, mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if steps - start_steps < 2:
                            update_grads[str(steps-start_steps+1)] = {n: None if p.grad is None else float(p.grad.abs().max()) for n,p in model.named_parameters()}
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        opt.step()
                        steps += 1
                        losses.append(loss.detach().item())
                val, pred = evaluate(model, data, data['val_origins'], gpu_job, spec['id'] + f' VAL epoch {epoch}')
                history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'],
                                'learning_rates': {g['name']:g['lr'] for g in opt.param_groups},
                                'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    require_storage(16 * 1024**2)
                    torch.save({'state': model.adapter_state(), 'spec': spec, 'model_config': model.model_config(),
                                'epoch': epoch, 'steps': steps, 'source_initial': init_receipt}, local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', prediction=pred, origins=data['val_origins'])
                else:
                    stale += 1
                if epoch:
                    scheduler_step(scheduler, val['mse'], spec['t_lr'] / 1e-3)
                torch.save({'state': model.adapter_state(), 'spec': spec, 'model_config': model.model_config(),
                            'epoch': epoch, 'next_batch': 0, 'steps': steps, 'optimizer': opt.state_dict(),
                            'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'stale': stale,
                            'best': best, 'best_epoch': best_epoch, 'history': history, 'schedules': schedules,
                            'boundary': 'post-VAL, post-scheduler, complete epoch; next epoch first batch'}, local / 'restart.pt')
                save_json(out / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}, val {val["mse"]:.7f}', steps-start_steps)
                print(json.dumps({'id':spec['id'], 'epoch':epoch, 'best_epoch':best_epoch, 'val_mse':val['mse'],
                                  'elapsed_s':round(time.perf_counter()-started,1)}), flush=True)
                if epoch and stale >= 6:
                    break
            final = model.adapter_state()
            changes = {k: float((final[k]-initial[k]).abs().max()) for k in initial}
            changed_counts = {k: int(torch.count_nonzero(final[k] != initial[k])) for k in initial}
            for k in ('temporal.weight', 'temporal.bias', 'residual.down.weight', 'residual.up.weight'):
                assert changes[k] > 0, f'{k} did not update'
            for k in ('encoder.weight', 'decoder.weight'):
                assert (changes[k] == 0) == (spec['ed_mode'] == 'fixed_ed'), f'{k} freeze/update mismatch'
            training_peak = {'allocated_bytes': torch.cuda.max_memory_allocated(),
                             'reserved_bytes': torch.cuda.max_memory_reserved(),
                             'scope': 'initial probe plus fit/TRAIN updates/VAL, before simultaneous restoration instances'}
            restart = torch.load(local / 'restart.pt', map_location='cpu', weights_only=False)
            restarted = LinearResidual(spec, init_edg).cuda()
            restarted.restore_adapter(restart['state'])
            ropt, rscheduler = optimizer_for(restarted, spec)
            ropt.load_state_dict(restart['optimizer'])
            rscheduler.load_state_dict(restart['scheduler'])
            assert_nested_equal(opt.state_dict(), ropt.state_dict())
            assert_nested_equal(scheduler.state_dict(), rscheduler.state_dict())
            assert_nested_equal(model.adapter_state(), restarted.adapter_state())
            before_rng = rng_state()
            restore_rng(restart['rng'])
            restored_rng = rng_state()
            assert torch.equal(restored_rng['torch'], restart['rng']['torch'])
            assert all(torch.equal(a,b) for a,b in zip(restored_rng['cuda'], restart['rng']['cuda']))
            restore_rng(before_rng)
            del restarted, ropt, rscheduler, restart
            selected = torch.load(local / 'best.pt', map_location='cpu', weights_only=False)
            restored = LinearResidual(selected['spec'], init_edg).cuda()
            restored.restore_adapter(selected['state'])
            replay_score, replay = evaluate(restored, data, data['val_origins'], gpu_job, spec['id'] + ' selected replay')
            with np.load(local / 'best_val.npz') as archive:
                replay_error = float(np.abs(replay - archive['prediction']).max())
                assert np.array_equal(archive['origins'], data['val_origins'])
            assert replay_error <= 1e-6
            selected_train, _ = evaluate(restored, data, probe, gpu_job, spec['id'] + ' selected TRAIN probe')
            result = dict(attempt, status='complete', initial_source=init_receipt,
                          initial_parameter_sha256=initial_hash, initial_checkpoint=str(local/'initial.pt'),
                          initial_checkpoint_sha256=digest(local/'initial.pt'), initial_gradients=initial_grads,
                          first_update_gradients=update_grads, parameter_max_updates=changes,
                          changed_parameter_scalars=changed_counts,
                          selected_parameter_max_updates={k:float((selected['state'][k]-initial[k]).abs().max()) for k in initial},
                          total_parameters=sum(p.numel() for p in restored.parameters()),
                          trainable_parameters=sum(p.numel() for p in restored.parameters() if p.requires_grad),
                          best_val_mse=best, selected_epoch=best_epoch, epochs_completed=epoch,
                          total_steps=steps, attempt_updates=steps-start_steps, selected_steps=selected['steps'],
                          initial_train_probe=initial_train, selected_train_probe=selected_train,
                          initial_val_mse=history[0]['val_mse'], replay_error=replay_error,
                          restart_roundtrip={'model':True,'optimizer':True,'scheduler':True,'rng':True,'additional_updates':0},
                          checkpoint=str(local/'best.pt'), checkpoint_sha256=digest(local/'best.pt'),
                          restart_checkpoint=str(local/'restart.pt'), restart_sha256=digest(local/'restart.pt'),
                          schedules=schedules, at_improving_boundary=(epoch==spec['epochs'] and best_epoch==epoch),
                          training_peak=training_peak,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          peak_scope='whole fit including simultaneous restoration verification instances',
                          elapsed_s=time.perf_counter()-started)
            save_json(out/'result.json', result)
            attempt.update(status='complete', elapsed_s=result['elapsed_s'], actual_updates=steps-start_steps)
            save_json(out/'attempt.json', attempt)
            fit_job.heartbeat('complete and restored', steps-start_steps)
            del model, restored, opt, scheduler, x, y, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            attempt.update(status='failed', elapsed_s=time.perf_counter()-started, actual_updates=steps-start_steps,
                           error=traceback.format_exc())
            save_json(out/'attempt.json', attempt)
            fit_job.heartbeat('failed', steps-start_steps)
            raise


def select(specs):
    selected = {}
    for dataset in ('electricity','bull'):
        ds = [s for s in specs if s['dataset']==dataset]
        if not ds:
            continue
        groups = {}
        for lr in sorted({s['t_lr'] for s in ds}, reverse=True):
            cohort = [s for s in ds if s['t_lr']==lr]
            assert sorted(s['seed'] for s in cohort)==[92601,92602]
            results = [json.loads((HERE/'runs'/s['id']/'result.json').read_text(encoding='utf-8')) for s in cohort]
            groups[str(lr)] = {'mean_best_val_mse':float(np.mean([r['best_val_mse'] for r in results])),
                               'run_ids':[s['id'] for s in cohort]}
        choice = min(groups, key=lambda k: groups[k]['mean_best_val_mse'])
        selected[dataset] = dict(lr=float(choice), **groups[choice], lr_candidates=groups,
                                 criterion='two individual seed best VAL losses averaged; no DEV/E1/E2 selection')
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--specs', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=2400)
    parser.add_argument('--selection', default='selected_linear.json')
    args=parser.parse_args()
    specs=json.loads(Path(args.specs).read_text(encoding='utf-8'))
    for spec in specs:
        validate_spec(spec)
    pending=[s for s in specs if not (HERE/'runs'/s['id']/'result.json').exists()]
    if pending:
        with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds) as job:
            data={name:load_data(name) for name in {s['dataset'] for s in pending}}
            for spec in pending:
                fit(spec, data[spec['dataset']], job)
    target=HERE/args.selection
    if target.exists():
        raise RuntimeError('Selection already exists; preserve original and use new filename')
    save_json(target, select(specs))


if __name__ == '__main__':
    main()
