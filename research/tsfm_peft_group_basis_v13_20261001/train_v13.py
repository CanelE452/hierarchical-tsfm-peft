"""Accounted frozen-main caching and matched CUDA G-only fitting for v13."""
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

from runtime_v13 import (HERE, CACHE, DATASETS, SEEDS, Job, artifact, budget_snapshot, configure,
                        digest, environment_receipt, load_data, masked_macro_loss, phase_sample,
                        protocol, read_json, save_json, score_arrays, tensors)
from model_v13 import (FAMILIES, build_model, make_checkpoint, restore_model, state_digest, tensor_hashes)

RUNS, LOCAL_RUNS = HERE / 'runs', CACHE / 'runs'


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
    parameters = [value for value in model.parameters() if value.requires_grad]
    assert sum(value.numel() for value in parameters) == 17920
    optimizer = torch.optim.AdamW(parameters, lr=spec['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2,
                                                        threshold=1e-4, threshold_mode='rel')
    return optimizer, scheduler


def cache_binding(dataset):
    return {'dataset': dataset, 'files': {name: digest(HERE / name) for name in
            ('protocol.json', 'data_manifest.json', 'model_v13.py', 'train_v13.py', 'runtime_v13.py')},
            'precision': 'float32', 'device': 'cuda', 'batch_size': 4,
            'scope': 'Frozen F0(XU)U transpose; TRAIN and VAL only; deployment is online'}


def load_cache(dataset):
    record = read_json(HERE / 'main_cache.json')['datasets'][dataset]
    if record['binding'] != cache_binding(dataset):
        raise RuntimeError('Frozen main cache source/data binding changed')
    result = {'record': record}
    for split in ('train', 'val'):
        entry = record[split]
        artifact(entry['path'], entry['sha256'])
        with np.load(entry['path'], allow_pickle=False) as arrays:
            result[split] = {key: arrays[key] for key in arrays.files}
        assert np.isfinite(result[split]['main']).all()
    return result


def cache_main(dataset, job):
    manifest_path = HERE / 'main_cache.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'datasets': {}}
    if dataset in manifest['datasets']:
        load_cache(dataset)
        return
    data = load_data(dataset, include_test=False)
    model = build_model(dataset, 'group_res', SEEDS[0]).eval()
    frozen_before = state_digest(model.backbone.state_dict())
    local = CACHE / 'main_cache' / job.label / dataset
    local.mkdir(parents=True, exist_ok=False)
    record = {'binding': cache_binding(dataset), 'job_id': job.id, 'initial_g': model.initial_receipt}
    with torch.no_grad():
        for split in ('train', 'val'):
            origins = data[split + '_origins']
            outputs = []
            for start in range(0, len(origins), 4):
                job.check_limits()
                x, _, _ = tensors(data, origins[start:start + 4])
                outputs.append(model.main_prediction(x).cpu().numpy())
                if start % 256 == 0:
                    job.heartbeat({'cache': dataset, 'split': split, 'origins_done': start, 'total': len(origins)})
            prediction = np.concatenate(outputs)
            if not np.isfinite(prediction).all():
                raise RuntimeError('Nonfinite frozen main cache')
            path = local / (split + '.npz')
            np.savez_compressed(path, origins=origins, main=prediction)
            record[split] = artifact(path)
            x, _, _ = tensors(data, origins[:4])
            direct = model(x)
            cached = model.forecast_from_main(x, torch.as_tensor(prediction[:4], device='cuda'))
            difference = (direct - cached).abs()
            if not bool(torch.all(difference <= 1e-5 + 1e-4 * direct.abs())):
                raise RuntimeError('Online/cached initial forecast parity failed')
            record[split + '_online_parity'] = {'max_absolute': float(difference.max()), 'passed': True,
                                               'atol': 1e-5, 'rtol': 1e-4}
    assert state_digest(model.backbone.state_dict()) == frozen_before
    record['frozen_backbone_sha256_before_after'] = frozen_before
    record['data'] = artifact(data['_path'])
    manifest['datasets'][dataset] = record
    save_json(manifest_path, manifest)
    del model, x, direct, cached, outputs, prediction, difference
    gc.collect()
    torch.cuda.empty_cache()


def cached_main(cache, origins, split):
    saved = cache[split]
    indices = np.searchsorted(saved['origins'], origins)
    if np.any(indices >= len(saved['origins'])) or not np.array_equal(saved['origins'][indices], origins):
        raise ValueError('Requested origin absent from the immutable main cache')
    return torch.as_tensor(saved['main'][indices], device='cuda', dtype=torch.float32)


@torch.no_grad()
def evaluate(model, data, cache, origins, split, job):
    model.eval()
    predictions = []
    for start in range(0, len(origins), 4):
        job.check_limits()
        group = origins[start:start + 4]
        x, _, _ = tensors(data, group)
        predictions.append(model.forecast_from_main(x, cached_main(cache, group, split)).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return score_arrays(prediction, target, mask), prediction


def result_for(spec, required=True):
    replacement = HERE / 'replacements' / (spec['id'] + '.json')
    path = RUNS / spec['id'] / 'result.json'
    if replacement.exists():
        row = read_json(replacement)
        assert row['canonical_id'] == spec['id']
        path = Path(artifact(row['result']['path'], row['result']['sha256'])['path'])
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    result = read_json(path)
    if result['status'] != 'complete' or result['spec'] != spec:
        raise RuntimeError('Completed fit does not match its canonical specification')
    for name in ('initial_checkpoint', 'last_checkpoint', 'initial_g', 'best_val_prediction'):
        artifact(result[name]['path'], result[name]['sha256'])
    artifact(result['checkpoint'], result['checkpoint_sha256'])
    if result['data_manifest_sha256'] != digest(HERE / 'data_manifest.json'):
        raise RuntimeError('Fit data binding changed')
    return result


def retry_contract(spec, tag, reason, resume_from):
    if not tag or not re.fullmatch(r'[A-Za-z0-9_-]+', tag) or not reason:
        raise ValueError('Recovery requires a unique tag and concrete technical reason')
    if (HERE / 'selected.json').exists() or result_for(spec, required=False) is not None:
        raise RuntimeError('No retry after final selection or a valid completed fit')
    jobs = read_json(HERE / 'ledger.json')['jobs']
    prior = [row for row in jobs if row['category'] == 'neural_fit' and row['metadata']['fit_id'] == spec['id']]
    if not prior or any(row['status'] != 'failed' for row in prior):
        raise RuntimeError('Every prior fit attempt must be a closed failure')
    record = {'attempt_id': spec['id'] + '__retry_' + tag, 'canonical_id': spec['id'],
              'previous_ledger_ids': [row['id'] for row in prior], 'reason': reason, 'technical_retry': True}
    if resume_from:
        path = Path(resume_from).resolve()
        if path.name != 'last.pt' or path.parent.parent != LOCAL_RUNS.resolve():
            raise ValueError('Resume requires a v13 complete-epoch last.pt')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['spec'] != spec or saved['boundary'] != 'post-VAL/post-scheduler/complete epoch':
            raise ValueError('Resume checkpoint has a different specification/boundary')
        if saved['epoch'] >= 120 or saved['stale'] >= 6:
            raise RuntimeError('Completed stop boundary cannot consume more fitting updates')
        record['resume_from'] = artifact(path)
    return record


def fit(spec, gpu_job, retry=None):
    attempt_id = retry['attempt_id'] if retry else spec['id']
    local, public = LOCAL_RUNS / attempt_id, RUNS / attempt_id
    if local.exists() or public.exists():
        raise FileExistsError('Preserve existing attempt; use explicit technical recovery')
    metadata = {**spec, 'fit_id': spec['id'], 'training_device': 'cuda', 'cached_main': True}
    if retry:
        metadata.update(technical_retry=True, reason=retry['reason'])
    with Job('neural_fit', attempt_id, reserve_s=0, metadata=metadata) as fit_job:
        started = time.perf_counter()
        steps = updates = carried = 0
        local.mkdir(parents=True)
        public.mkdir(parents=True)
        attempt = {'id': attempt_id, 'canonical_id': spec['id'], 'spec': spec, 'ledger_id': fit_job.id,
                   'status': 'running', 'retry': retry, 'environment': environment_receipt()}
        save_json(public / 'attempt.json', attempt)
        try:
            data, cache = load_data(spec['dataset']), load_cache(spec['dataset'])
            assert all(np.array_equal(data[split + '_origins'], cache[split]['origins']) for split in ('train', 'val'))
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            random.seed(spec['seed'])
            model = build_model(spec['dataset'], spec['family'], spec['seed'], load_backbone=False)
            assert not hasattr(model, 'backbone')
            initial = model.adapter_state()
            frozen = {name: value for name, value in initial.items() if not name.startswith('residual.')}
            torch.save(make_checkpoint(model, spec, epoch=0, steps=0), local / 'initial.pt')
            probe = phase_sample(data['train_origins'], spec['seed'], 0, max(96, data['_phase_period']), period=data['_phase_period'])
            initial_train, _ = evaluate(model, data, cache, probe, 'train', gpu_job)
            optimizer, scheduler = optimizer_for(model, spec)
            history, schedules, gradients = [], [], {}
            best, best_epoch, stale, start_epoch = float('inf'), 0, 0, 0
            if retry and retry.get('resume_from'):
                artifact(retry['resume_from']['path'], retry['resume_from']['sha256'])
                resumed = torch.load(retry['resume_from']['path'], map_location='cpu', weights_only=False)
                assert resumed['initial_g'] == model.initial_receipt
                assert resumed['cache_record'] == cache['record']
                model.restore_adapter(resumed['state'])
                optimizer.load_state_dict(resumed['optimizer'])
                scheduler.load_state_dict(resumed['scheduler'])
                restore_rng(resumed['rng'])
                steps = carried = resumed['steps']
                best, best_epoch, stale, start_epoch = resumed['best'], resumed['best_epoch'], resumed['stale'], resumed['epoch'] + 1
                history, schedules, gradients = resumed['history'], resumed['schedules'], resumed['gradients']
                for name in ('best.pt', 'best_val.npz'):
                    receipt = resumed['selected_files'][name]
                    shutil.copyfile(artifact(receipt['path'], receipt['sha256'])['path'], local / name)
                del resumed
            frozen_versions = {name: value._version for name, value in model.named_parameters() if not value.requires_grad}
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(start_epoch, 121):
                losses = []
                if epoch:
                    schedule = phase_sample(data['train_origins'], spec['seed'], epoch, 512, period=data['_phase_period'])
                    schedules.append({'epoch': epoch, 'sha256': hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()})
                    model.train()
                    for start in range(0, 512, 4):
                        if updates % 16 == 0:
                            gpu_job.check_limits()
                            fit_job.heartbeat(f'epoch {epoch}; batch {start // 4}', updates)
                        group = schedule[start:start + 4]
                        x, y, mask = tensors(data, group)
                        optimizer.zero_grad(set_to_none=True)
                        loss = masked_macro_loss(model.forecast_from_main(x, cached_main(cache, group, 'train')), y, mask)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if updates < 2:
                            gradients[str(steps + 1)] = {name: {'numel': value.numel(), 'has_grad': value.grad is not None,
                                'max_abs': None if value.grad is None else float(value.grad.abs().max()),
                                'nonzero': 0 if value.grad is None else int(torch.count_nonzero(value.grad))}
                                for name, value in model.named_parameters() if value.requires_grad}
                        torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1, error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                        updates += 1
                        fit_job.updates = updates
                        losses.append(float(loss.detach()))
                val, prediction = evaluate(model, data, cache, data['val_origins'], 'val', gpu_job)
                history.append({'epoch': epoch, 'steps': steps, 'attempt_updates': updates, 'attempt_id': attempt_id,
                                'train_loss': float(np.mean(losses)) if losses else None,
                                'val_mse': val['mse'], 'val_mae': val['mae'], 'lr': optimizer.param_groups[0]['lr'],
                                'elapsed_s': time.perf_counter() - started})
                if val['mse'] < best:
                    best, best_epoch, stale = val['mse'], epoch, 0
                    torch.save(make_checkpoint(model, spec, epoch=epoch, steps=steps), local / 'best.pt')
                    np.savez_compressed(local / 'best_val.npz', origins=data['val_origins'], prediction=prediction)
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val['mse'])
                torch.save(make_checkpoint(model, spec, epoch=epoch, steps=steps, optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rng_state(), best=best, best_epoch=best_epoch, stale=stale,
                    history=history, schedules=schedules, gradients=gradients, cache_record=cache['record'],
                    selected_files={name: artifact(local / name) for name in ('best.pt', 'best_val.npz')},
                    boundary='post-VAL/post-scheduler/complete epoch'), local / 'last.pt')
                save_json(public / 'curve.json', history)
                fit_job.heartbeat(f'epoch {epoch}; best {best_epoch}', updates)
                print(f'{attempt_id} epoch={epoch} best={best_epoch} val={val["mse"]:.9g}', flush=True)
                if epoch and stale >= 6:
                    break
            final_train, _ = evaluate(model, data, cache, probe, 'train', gpu_job)
            final = model.adapter_state()
            assert_equal(frozen, {name: final[name] for name in frozen})
            assert all(value.grad is None and value._version == frozen_versions[name]
                       for name, value in model.named_parameters() if not value.requires_grad)
            changed = {name: int(torch.count_nonzero(final[name] != initial[name])) for name in final if name.startswith('residual.')}
            changes = {name: float((final[name] - initial[name]).abs().max()) for name in changed}
            assert any(changed.values())
            peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved(),
                    'scope': 'CUDA cached-main G fitting only; online F0 cache generation is separately accounted'}
            last = torch.load(local / 'last.pt', map_location='cpu', weights_only=False)
            assert_equal(final, last['state'])
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
            del model, optimizer, scheduler, fresh_optimizer, fresh_scheduler, last, x, y, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
            restored = restore_model(local / 'best.pt', load_backbone=False)
            replay_score, replay = evaluate(restored, data, cache, data['val_origins'], 'val', gpu_job)
            with np.load(local / 'best_val.npz', allow_pickle=False) as arrays:
                assert np.array_equal(arrays['origins'], data['val_origins'])
                replay_error = float(np.abs(replay - arrays['prediction']).max())
            assert replay_error <= 1e-6 and abs(replay_score['mse'] - best) <= 1e-8
            selected_train, _ = evaluate(restored, data, cache, probe, 'train', gpu_job)
            chosen = restored.adapter_state()
            result = {**attempt, 'status': 'complete', 'checkpoint': str(local / 'best.pt'),
                'checkpoint_sha256': digest(local / 'best.pt'), 'initial_checkpoint': artifact(local / 'initial.pt'),
                'last_checkpoint': artifact(local / 'last.pt'), 'best_val_prediction': artifact(local / 'best_val.npz'),
                'initial_g': restored.initial_receipt, 'initial_state_hashes': tensor_hashes(initial),
                'data': artifact(data['_path']), 'data_manifest_sha256': digest(HERE / 'data_manifest.json'),
                'cache_record': cache['record'], 'best_val_mse': best, 'selected_epoch': best_epoch,
                'epochs_completed': epoch, 'actual_updates': updates, 'carried_steps': carried, 'cumulative_model_steps': steps,
                'upper_epoch_limit_improving': bool(epoch == 120 and epoch - best_epoch < 6),
                'first_update_gradients': gradients, 'changed_parameter_scalars': changed, 'parameter_max_updates': changes,
                'selected_parameter_max_updates': {name: float((chosen[name] - initial[name]).abs().max()) for name in changed},
                'frozen_parameters_verified': {'no_gradient': True, 'version_unchanged': True, 'optimizer_exclusion': True,
                    'state_sha256': state_digest(frozen), 'F0': 'Immutable cached main; backbone hash checked during cache generation'},
                'initial_train_probe': initial_train, 'selected_train_probe': selected_train, 'final_train_probe': final_train,
                'initial_val_mse': history[0]['val_mse'], 'train_probe_origins': probe.tolist(), 'schedules': schedules,
                'restart_roundtrip': {'model_optimizer_scheduler_rng': True, 'extra_updates': 0}, 'replay_error': replay_error,
                'trainable_parameters': 17920, 'training_peak': peak, 'elapsed_s': time.perf_counter() - started,
                'training_device': 'cuda', 'deployment': 'online F0, no prediction cache'}
            save_json(public / 'result.json', result)
            if retry:
                save_json(HERE / 'replacements' / (spec['id'] + '.json'), {'canonical_id': spec['id'],
                    'selected_attempt_id': attempt_id, 'result': artifact(public / 'result.json'), 'technical_retry': retry,
                    'selection_basis': 'valid technical completion, never performance'})
            attempt.update(status='complete', actual_updates=updates, elapsed_s=result['elapsed_s'])
            save_json(public / 'attempt.json', attempt)
            fit_job.heartbeat('saved and replayed selected cached-head checkpoint', updates)
            del restored
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            fit_job.updates = updates
            attempt.update(status='failed', actual_updates=updates, carried_steps=carried, cumulative_model_steps=steps,
                           elapsed_s=time.perf_counter() - started, error=traceback.format_exc())
            save_json(public / 'attempt.json', attempt)
            raise


def select_all():
    specs = protocol()['fits']
    rows = [result_for(spec) for spec in specs]
    selected = {dataset: {} for dataset in DATASETS}
    initial, schedules = {}, {}
    for row in rows:
        spec = row['spec']
        key = (spec['dataset'], spec['seed'])
        if key in initial:
            assert initial[key] == row['initial_g']['sha256']
        initial[key] = row['initial_g']['sha256']
        for schedule in row['schedules']:
            key = (spec['dataset'], spec['seed'], schedule['epoch'])
            if key in schedules:
                assert schedules[key] == schedule['sha256']
            schedules[key] = schedule['sha256']
    for dataset in DATASETS:
        for family in FAMILIES:
            kept = sorted([row for row in rows if row['spec']['dataset'] == dataset and row['spec']['family'] == family],
                          key=lambda row: row['spec']['seed'])
            assert [row['spec']['seed'] for row in kept] == list(SEEDS)
            selected[dataset][family] = {'lr': .001, 'seeds': list(SEEDS), 'run_ids': [row['id'] for row in kept],
                'checkpoints': [row['checkpoint'] for row in kept], 'checkpoint_sha256': [row['checkpoint_sha256'] for row in kept],
                'selected_epochs': [row['selected_epoch'] for row in kept],
                'best_val_prediction': [row['best_val_prediction'] for row in kept],
                'mean_best_val_mse': float(np.mean([row['best_val_mse'] for row in kept]))}
    output = {'schema': 'v13_joint_selection_1', 'selected': selected, 'test_used_for_selection': False,
              'all_neural_runs': [{key: row[key] for key in ('id', 'canonical_id', 'spec', 'checkpoint', 'checkpoint_sha256',
                                                          'selected_epoch', 'best_val_mse')} for row in rows],
              'criterion': 'minimum complete VAL MSE including epoch0; earliest exact tie; LR fixed 1e-3',
              'protocol': artifact(HERE / 'protocol.json'), 'data_manifest': artifact(HERE / 'data_manifest.json'),
              'matched_initial_groups': len(initial), 'matched_schedule_groups': len(schedules)}
    path = HERE / 'selected.json'
    if path.exists() and read_json(path) != output:
        raise RuntimeError('Preserve existing final selection')
    if not path.exists():
        save_json(path, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('cache', 'fit'), required=True)
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--fit-id')
    parser.add_argument('--label', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    parser.add_argument('--retry-tag')
    parser.add_argument('--retry-reason')
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    with Job('gpu', args.label, reserve_s=args.reserve_s, metadata={'purpose': 'v13 ' + args.stage}) as job:
        configure()
        if args.stage == 'cache':
            if args.fit_id or args.retry_tag or args.resume_from:
                raise ValueError('Fit recovery arguments do not apply to caching')
            for dataset in ([args.dataset] if args.dataset else DATASETS):
                cache_main(dataset, job)
        else:
            if args.dataset:
                raise ValueError('Use --fit-id for a single fit; --dataset applies only to caching')
            specs = protocol()['fits']
            chosen = [spec for spec in specs if spec['id'] == args.fit_id] if args.fit_id else specs
            if not chosen or args.retry_tag and len(chosen) != 1:
                raise ValueError('Recovery requires exactly one prescribed fit')
            for spec in chosen:
                if result_for(spec, required=False) is not None:
                    if args.retry_tag:
                        raise RuntimeError('No performance retry of a valid completion')
                    continue
                retry = retry_contract(spec, args.retry_tag, args.retry_reason, args.resume_from) if args.retry_tag else None
                if not args.retry_tag and (args.resume_from or args.retry_reason):
                    raise ValueError('Explicit retry tag required for recovery')
                fit(spec, job, retry)
            if all(result_for(spec, required=False) is not None for spec in specs):
                select_all()
        job.heartbeat({'stage_complete': args.stage, 'budget': budget_snapshot()})


if __name__ == '__main__':
    main()
