"""Actual CUDA common-endpoint evaluation; retained CPU records are provenance only."""
import importlib
import os
from pathlib import Path
import time

from . import existing_jobs as existing


def parity_statistics(reference, candidate, atol=1e-5, rtol=1e-4):
    import numpy as np
    if reference.shape != candidate.shape:
        raise ValueError('CPU/CUDA parity axes differ')
    difference = np.abs(reference.astype(np.float64)-candidate.astype(np.float64))
    return dict(status='PASS' if np.allclose(candidate, reference, atol=atol, rtol=rtol) else 'FAILED',
                atol=atol, rtol=rtol, max_absolute_difference=float(difference.max()),
                mean_absolute_difference=float(difference.mean()),
                scope='first one registered origin; normalized metric scale',
                bitwise_equivalence_claimed=False)


def common_endpoint_cuda(job, ctx):
    if job['device'] != 'cuda':
        raise ValueError('CUDA endpoint handler requires an explicit CUDA job')
    specification = existing.endpoint_spec(job)
    modules = existing._modules(ctx)
    np, torch, fit, data, training = (modules[k] for k in ('np', 'torch', 'fit', 'data', 'training'))
    task = existing._jena(ctx, modules)
    seed, family, index, split = job['seed'], job['family'], job['endpoint_index'], job['split']
    checkpoint_path = existing._run_root(ctx, seed, family)/'search'/f'checkpoint_{index:06d}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source_path = existing._run_root(ctx, seed, family, True)/'SOURCE_CONTRACT.json'
    source = existing.read(source_path)
    if checkpoint['updates'] != specification['updates'] or checkpoint['sampler']['exposures'] != specification['exposures']:
        raise ValueError('CUDA endpoint progress differs from original endpoint')
    for name in ('source_sha', 'config_sha', 'data_sha', 'model_sha'):
        if checkpoint['contract'][name] != source[name]:
            raise ValueError('CUDA endpoint original source contract mismatch: '+name)
    if checkpoint['contract']['source_sha'] != fit._source_contract()[0] or checkpoint['contract']['data_sha'] != task['data_sha256']:
        raise ValueError('CUDA endpoint actual scientific source/data differs')
    origins = data.origins(existing.BOUNDS[split], 512, 48, specification['stride'])
    public = Path(ctx['report'])/'scores'/(existing._job_id(job)+'.json')
    pool = None
    started = time.perf_counter()
    try:
        pool, cpu_f = existing._foundation(ctx, modules, 48, 'cpu')
        spec = modules['legacy'].spec_for(seed, 'B2' if family == 'B2_CAP_POINTWISE' else family)
        if family == 'B2_CAP_POINTWISE':
            cap = importlib.import_module('cap_control')
            if source['cap_control_sha256'] != existing.sha(cap.__file__):
                raise ValueError('CAP architecture source differs from retained checkpoint')
            model, norm = cap.create_cap(spec, task, 10482, cpu_f)
        else:
            model, norm = fit.create_model(spec, task, 10482, cpu_f)
        if checkpoint['contract']['scaler_sha'] != fit.scaler_fingerprint(norm['mean'], norm['std']):
            raise ValueError('CUDA endpoint original search scaler differs')
        model.load_state_dict(checkpoint['model'], strict=True)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.eval().requires_grad_(False)
        if split == 'V_select':
            mean, std, _ = data.fit_statistics(task, 12579)
        else:
            mean, std = norm['mean'], norm['std']
        metric_windows = training.FastWindows(task, mean, std, 'cpu')
        cpu_windows = training.FastWindows(task, norm['mean'], norm['std'], 'cpu')
        def metric_prediction(prediction):
            if split != 'V_select':
                return prediction
            # Same FP32 arithmetic and A60 metric scale for both physical devices.
            return (prediction*norm['std']+norm['mean']-mean)/std
        with torch.no_grad():
            first_cpu_x, _, _ = cpu_windows.get(origins[:1])
            cpu_prediction = metric_prediction(fit.predict_model(model, first_cpu_x, cpu_f, 'B2', 6).cpu().numpy())
        pool.close(); pool = None
        pool, gpu_f = existing._foundation(ctx, modules, 48, 'cuda')
        model.to('cuda').eval()
        gpu_windows = training.FastWindows(task, norm['mean'], norm['std'], 'cuda')
        with torch.no_grad():
            first_gpu_x, _, _ = gpu_windows.get(origins[:1])
            gpu_prediction = metric_prediction(fit.predict_model(model, first_gpu_x, gpu_f, 'B2', 6).cpu().numpy())
        parity = parity_statistics(cpu_prediction, gpu_prediction)
        parity.update(origin=int(origins[0]), checkpoint_sha256=existing.sha(checkpoint_path),
                      CPU_prediction_sha256=existing._array_fingerprint(cpu_prediction)['sha256'],
                      CUDA_prediction_sha256=existing._array_fingerprint(gpu_prediction)['sha256'])
        if parity['status'] != 'PASS':
            result = dict(status='FAILED', reason='Fixed CPU-to-CUDA parity tolerance failed',
                          parity=parity, seed=seed, model=family, endpoint_index=index, eval=split,
                          device='cuda', metric_scale=specification['metric_scale'])
            existing.atomic_json(public, result)
            return dict(status='FAILED', reason=result['reason'], artifact_paths=[str(public)], result=result)
        predictions, targets, masks = [], [], []
        with torch.no_grad():
            for lo in range(0, len(origins), 16):
                if time.time() >= existing._deadline(ctx):
                    raise TimeoutError('CUDA common endpoint reached job deadline')
                if ctx.get('lease'):
                    ctx['lease'].check()
                chosen = origins[lo:lo+16]
                x, _, mask = gpu_windows.get(chosen)
                prediction = metric_prediction(fit.predict_model(model, x, gpu_f, 'B2', 6).cpu().numpy())
                _, y, metric_mask = metric_windows.get(chosen)
                if not np.array_equal(mask.cpu().numpy(), metric_mask.numpy()):
                    raise ValueError('CPU/CUDA metric mask differs')
                predictions.append(prediction); targets.append(y.numpy()); masks.append(metric_mask.numpy())
        values = dict(pred_normalized=np.concatenate(predictions), target_normalized=np.concatenate(targets),
                      observed=np.concatenate(masks), origins=origins, channel_ids=np.asarray(task['columns'], str))
        archive = Path(ctx['cache'])/'cuda_endpoints'/(existing._job_id(job)+'.npz')
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.partial.npz')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, archive)
        # Reopen the actual committed archive for the independent stored-array scorer.
        stored = existing._archive(archive, existing.sha(archive))
        score = existing.score_arrays(stored['pred_normalized'], stored['target_normalized'], stored['observed'])
        if split != 'V_select':
            for key in ('origin_sse', 'origin_sae', 'origin_counts'):
                score.pop(key)
        legacy = Path(ctx['repo'])/'reports'/existing.OLD/'same_update'/f's{seed}'/f'{family}_e{index}_{split}.json'
        result = dict(score, status='COMPLETE', dataset='jena', C=21, K=6, L=512, H=48,
                      seed=seed, model=family, params=parameters, device='cuda', dtype='float32',
                      epoch=specification['epoch'], updates=specification['updates'], exposure=specification['exposures'],
                      eval=split, fit='A_fit_search', selection='NOT_SELECTED_COMMON_ENDPOINT',
                      metric_scale=specification['metric_scale'], same_update=True, new_forward=True,
                      independent_stored_FP64=True, subtraction_dtype='float64', accumulation_dtype='float64',
                      parity=parity, checkpoint_sha256=existing.sha(checkpoint_path),
                      source_contract_sha256=existing.sha(source_path),
                      source_code_sha256=existing.sha(__file__), foundation_sha256=gpu_f.initial_hash,
                      scaler_sha256=fit.scaler_fingerprint(mean, std),
                      private_prediction_ref=existing._reference(archive, ctx),
                      private_prediction_archive_sha256=existing.sha(archive),
                      array_fingerprints={k: existing._array_fingerprint(v) for k,v in stored.items()},
                      legacy_CPU_result_sha256=existing.sha(legacy) if legacy.exists() else None,
                      legacy_CPU_result_reused=False, seconds=time.perf_counter()-started)
        existing.atomic_json(public, result)
        return dict(status='COMPLETE', reason='Actual CUDA endpoint with fixed CPU parity and stored-array FP64 score',
                    artifact_paths=[str(public), str(archive), str(checkpoint_path), str(source_path)], result=result)
    finally:
        if pool is not None:
            pool.close()
