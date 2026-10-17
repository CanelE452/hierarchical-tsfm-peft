"""Registered fixed-PE decoder comparison and matched PCA/QR initialization."""
import copy
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .core import atomic_json, digest, read, sha
from . import existing_jobs as existing


def qr_basis(channels, k, init_seed):
    generator = torch.Generator(device='cpu').manual_seed(int(init_seed))
    matrix = torch.randn(channels, k, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix, mode='reduced')
    signs = torch.where(torch.diag(r) < 0, -1., 1.)
    return (q * signs).float().numpy()


def fit_decoder(phi, delta, mask, rms, regularization, horizon_specific=False):
    x, y, mask = np.asarray(phi, np.float64)/rms, np.asarray(delta, np.float64), np.asarray(mask, bool)
    _, h, p = x.shape
    c = y.shape[-1]
    count = mask.sum((0, 1))
    weights = np.zeros((h, p, c) if horizon_specific else (p, c), np.float64)
    for channel in range(c):
        if not count[channel]:
            continue
        for horizon in range(h) if horizon_specific else (None,):
            active = mask[:, :, channel].reshape(-1) if horizon is None else mask[:, horizon, channel]
            f = x.reshape(-1, p)[active] if horizon is None else x[:, horizon][active]
            target = y[:, :, channel].reshape(-1)[active] if horizon is None else y[:, horizon, channel][active]
            penalty = regularization*count[channel]/(h if horizon_specific else 1)
            gram, rhs = f.T @ f + penalty*np.eye(p), f.T @ target
            coefficient = np.linalg.solve(gram, rhs)
            residual = np.linalg.norm(gram @ coefficient-rhs)/max(1., np.linalg.norm(rhs))
            if residual > 1e-8:
                raise ValueError('Masked decoder normal equation residual exceeds 1e-8')
            coefficient[np.asarray(rms) == 0] = 0
            if horizon is None:
                weights[:, channel] = coefficient
            else:
                weights[horizon, :, channel] = coefficient
    return weights


def decoder_predict(phi, rms, weights):
    x = np.asarray(phi, np.float64)/rms
    return np.einsum('nhp,hpc->nhc', x, weights) if weights.ndim == 3 else x @ weights


def array_metrics(prediction, target, mask):
    result = existing.score_arrays(prediction, target, np.asarray(mask, bool))
    result['counts'] = result['channel_counts']
    return result


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez(f, **values)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def foundation(ctx, modules, horizon=48, device='cuda'):
    from gpu_runtime import FrozenBolt
    value = FrozenBolt(Path(ctx['asset'])/modules['legacy'].MODEL_REL, horizon,
                       row_microbatch=128, checkpoint_inputs=False, device=device)
    if value.initial_hash != modules['legacy'].F_SHA:
        raise ValueError('Pinned Bolt model hash changed')
    return value


def check_deadline(ctx):
    if time.time() >= ctx.get('job_deadline', ctx['deadline']):
        raise TimeoutError('Finite job deadline reached')
    if ctx.get('lease'):
        ctx['lease'].check()


def predict_saved(model, norm, task, origins, f, modules, path, ctx):
    windows = modules['training'].FastWindows(task, norm['mean'], norm['std'], str(next(model.parameters()).device))
    prediction, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(origins), 8):
            check_deadline(ctx)
            x, y, m = windows.get(origins[start:start+8])
            p = model(x, f)
            prediction.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(m.cpu().numpy())
    p, y, m = map(np.concatenate, (prediction, targets, masks))
    save_npz(path, pred_normalized=p, target_normalized=y, observed=m,
             origins=origins, channel_ids=np.asarray(task['columns']))
    with np.load(path,allow_pickle=False) as archive:
        stored={key:archive[key] for key in archive.files}
    result=array_metrics(stored['pred_normalized'],stored['target_normalized'],stored['observed'])
    original=array_metrics(p,y,m)
    if max(abs(result[key]-original[key]) for key in ('MSE','MAE'))>1e-12:
        raise ValueError('Saved forecast independent FP64 score mismatch')
    result.update(cohort_metadata(stored,norm,modules),saved_array_independent_FP64='PASS')
    return result


def feature_cache(ctx, modules, task, model, norm, origins, f, checkpoint_sha, label):
    directory = ctx['cache']/'decoder_features'
    directory.mkdir(exist_ok=True)
    key = dict(source_sha256=sha(__file__), weights_sha=f.initial_hash,
               original_scientific_source_sha=modules['fit']._source_contract()[0],
               frozen_runtime_sha256=sha(modules['campaign']/'gpu_runtime.py'),
               instance_norm_source_sha256=getattr(f,'instance_norm_source_sha256',None),
               torch_version=torch.__version__,numpy_version=np.__version__,
               PE_checkpoint_sha256=checkpoint_sha, scaler_sha=modules['fit'].scaler_fingerprint(norm['mean'], norm['std']),
               dtype='float32', device='cuda', backend='torch_eager', L=512, H=48, C=21, K=6,
               origins_sha=modules['data'].array_sha256(np.asarray(origins)), channel_order=task['columns'],
               prefix_data_sha=modules['data'].array_sha256(task['physical']),
               prefix_mask_sha=modules['data'].array_sha256(task['observed']),
               point='config_quantile_0.5', routing='independent_univariate_rows',
               feature='lag-major(0,1,2,4,8,16,32) joined Z,U minus Z[-1-lag]')
    identity = digest(key)
    path = directory/(label+'_'+identity+'.npz')
    manifest = path.with_suffix('.json')
    if path.exists() or manifest.exists():
        if not manifest.exists() or not path.exists() or read(manifest)['key'] != key or read(manifest)['sha256'] != sha(path):
            raise ValueError('Feature cache identity or bytes mismatch')
        with np.load(path, allow_pickle=False) as values:
            return {k: values[k] for k in values.files}, read(manifest)
    windows = modules['training'].FastWindows(task, norm['mean'], norm['std'], 'cuda')
    phis, targets, masks, anchors = [], [], [], []
    started = time.perf_counter()
    model = model.to('cuda').eval()
    with torch.no_grad():
        for start in range(0, len(origins), 16):
            check_deadline(ctx)
            x, y, mask = windows.get(origins[start:start+16])
            z = model.encode(x)
            u = f(z)
            joined = torch.cat((z, u), 1)
            phi = torch.cat([joined[:,512-lag:512-lag+48]-z[:,-1-lag,None]
                             for lag in (0,1,2,4,8,16,32)], -1)
            phis.append(phi.cpu().numpy()); targets.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy()); anchors.append(x[:,-1:].cpu().numpy())
            if start % 512 == 0:
                atomic_json(directory/'progress.json', dict(label=label, done=start+len(x), total=len(origins)))
    values = dict(phi=np.concatenate(phis), target=np.concatenate(targets),
                  mask=np.concatenate(masks), anchor=np.concatenate(anchors))
    save_npz(path, **values)
    receipt = dict(key=key, sha256=sha(path), private_artifacts=[private_binding(path,ctx)], seconds=time.perf_counter()-started,
                   feature_count=42, F_calls='one extraction per checkpoint/stage/origin')
    atomic_json(manifest, receipt)
    f.assert_frozen_unchanged()
    return values, receipt


def decoder_refit(job, ctx):
    modules = existing._modules(ctx)
    task = existing._jena(ctx, modules)
    seed = int(job['seed'])
    f = foundation(ctx, modules)
    selected, norm60, bundle, selected_path = modules['legacy'].make_selected(task, seed, 'B2', f, modules)
    selected_epoch = bundle['selected_search_epoch']
    search_dir = existing._run_root(ctx, seed, 'B2')/'search'
    candidates = []
    for path in sorted(search_dir.glob('checkpoint_*.pt')):
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        if checkpoint['extra']['completed_epochs'][-1] == selected_epoch:
            declaration=read(existing._run_root(ctx,seed,'B2',True)/'SOURCE_CONTRACT.json')
            for name in ('source_sha','config_sha','data_sha','model_sha'):
                if checkpoint['contract'][name]!=declaration[name]:
                    raise ValueError('B03 A_fit checkpoint contract mismatch: '+name)
            normal50=modules['fit'].create_model(modules['legacy'].spec_for(seed),task,10482,f)[1]
            if (checkpoint['contract']['source_sha']!=modules['fit']._source_contract()[0] or
                checkpoint['contract']['data_sha']!=task['data_sha256'] or
                checkpoint['contract']['model_sha']!=f.initial_hash or
                checkpoint['contract']['scaler_sha']!=modules['fit'].scaler_fingerprint(normal50['mean'],normal50['std'])):
                raise ValueError('B03 checkpoint source/data/F/A_fit scaler mismatch')
            existing_score=ctx['repo']/'reports'/existing.OLD/'same_update'/f's{seed}'/f'B2_e{len(checkpoint["extra"]["completed_epochs"])-1}_A_stop.json'
            if sha(path)!=read(existing_score)['checkpoint_sha256']:
                raise ValueError('B03 A_fit checkpoint differs from registered endpoint bytes')
            candidates.append((path, checkpoint))
    search_available = bool(candidates)
    # Downgrade is determined from asset availability before evaluating either decoder.
    lambdas = (.01, .1, 1.) if search_available else (1.,)
    public = ctx['report']/'jobs'/(job['job_id']+'.json')
    rows, cache_receipts, selections, private_artifacts = [], [], {}, []
    stages = [('A60', selected, norm60, selected_path, 12579, 'V_select')]
    if search_available:
        path, checkpoint = candidates[-1]
        model, norm = modules['fit'].create_model(modules['legacy'].spec_for(seed), task, 10482, f)
        model.load_state_dict(checkpoint['model'])
        stages.insert(0, ('A_fit', model, norm, path, 10482, 'A_stop'))
    for stage, model, norm, checkpoint_path, end, eval_split in stages:
        origins = modules['data'].origins((0, end), 512, 48, 1)
        evaluation = modules['data'].origins(modules['legacy'].BOUNDS[eval_split], 512, 48, 48)
        train, train_receipt = feature_cache(ctx, modules, task, model, norm, origins, f, sha(checkpoint_path), f'{seed}_{stage}_fit')
        val, val_receipt = feature_cache(ctx, modules, task, model, norm, evaluation, f, sha(checkpoint_path), f'{seed}_{stage}_{eval_split}')
        cache_receipts.extend([train_receipt, val_receipt])
        rms = np.sqrt(np.mean(np.square(train['phi'].astype(np.float64)), (0, 1)))
        zero = rms == 0
        rms[zero] = 1
        for horizon_specific in (False, True):
            structure = 'horizon' if horizon_specific else 'shared'
            trials = []
            for regularization in lambdas if stage == 'A_fit' else (selections.get(structure, 1.),):
                check_deadline(ctx)
                weights = fit_decoder(train['phi'], train['target']-train['anchor'], train['mask'], rms,
                                      regularization, horizon_specific)
                weights[..., zero, :] = 0
                prediction = val['anchor'] + decoder_predict(val['phi'], rms, weights)
                metrics = array_metrics(prediction, val['target'], val['mask'])
                train_prediction = train['anchor'] + decoder_predict(train['phi'], rms, weights)
                train_metrics = array_metrics(train_prediction, train['target'], train['mask'])
                trials.append((metrics['MSE'], regularization, weights))
                rows.append(dict(method='B2_DECODER_'+structure.upper(), stage=stage,
                                 evaluation=eval_split, lambda_value=regularization, seed=seed,
                                 fit_MSE=train_metrics['MSE'], fit_MAE=train_metrics['MAE'],
                                 trainable_params=int(weights.size), **metrics))
                if stage == 'A60':
                    path = ctx['cache']/'decoder_predictions'/f'{seed}_{structure}.npz'
                    save_npz(path, pred_normalized=prediction, target_normalized=val['target'],
                             observed=val['mask'], origins=evaluation, channel_ids=np.asarray(task['columns']))
                    rows[-1]['prediction_sha256'] = sha(path)
                    with np.load(path,allow_pickle=False) as archive:
                        stored={key:archive[key] for key in archive.files}
                    saved_metrics=array_metrics(stored['pred_normalized'],stored['target_normalized'],stored['observed'])
                    if max(abs(saved_metrics[key]-metrics[key]) for key in ('MSE','MAE'))>1e-12:
                        raise ValueError('Saved decoder forecast FP64 score mismatch')
                    rows[-1].update(cohort_metadata(stored,norm,modules),saved_array_independent_FP64='PASS',
                                     decoder_refit_trainable_params=int(weights.size),
                                     frozen_PE_target_trained_params=34835,
                                     total_target_trained_params=34835+int(weights.size),
                                     additional_parameters_vs_shared=int(weights.size)-882,
                                     extraction_seconds=train_receipt['seconds']+val_receipt['seconds'],
                                     measured_deployment_latency=None)
                    save_npz(path.with_name(path.stem+'_coefficients.npz'), weights=weights, rms=rms)
                    private_artifacts.extend([private_binding(path,ctx),private_binding(path.with_name(path.stem+'_coefficients.npz'),ctx)])
            if stage == 'A_fit':
                selections[structure] = min(trials, key=lambda v: v[0])[1]
        if stage == 'A60':
            original_prediction = val['anchor'] + val['phi'] @ model.D.detach().cpu().numpy().reshape(42,21)
            rows.append(dict(method='B2_ORIGINAL_D', stage=stage, evaluation=eval_split,
                             seed=seed, trainable_params=882,
                             **array_metrics(original_prediction, val['target'], val['mask'])))
            values=dict(pred_normalized=original_prediction,target_normalized=val['target'],observed=val['mask'],
                        origins=evaluation,channel_ids=np.asarray(task['columns']))
            rows[-1].update(cohort_metadata(values,norm,modules),total_target_trained_params=35717,
                             frozen_PE_target_trained_params=34835,decoder_refit_trainable_params=0,
                             original_decoder_coefficients=882,measured_deployment_latency=None)
    f.assert_frozen_unchanged()
    value = dict(status='COMPLETE', family='B03', seed=seed, rows=rows,
                 selection='A_fit PE -> A_stop lambda -> corresponding fresh A60 PE' if search_available else 'PRE_EVAL_DOWNGRADE_FIXED_LAMBDA_1_A60_DIAGNOSTIC',
                 original_selected_epoch=selected_epoch, selected_lambdas=selections,
                 feature_caches=cache_receipts, shared_params=882, horizon_params=42336,
                 private_artifacts=private_artifacts,
                 normalization='fit-stage common RMS; no centering or intercept',
                 loss='channel macro SSE/Nc + lambda/C * shared_norm or lambda/(C*H) * horizon_norm')
    atomic_json(public, value)
    del f
    torch.cuda.empty_cache()
    return dict(status='COMPLETE', reason=None, artifact_paths=[str(public)], result=value)


def profile_training(ctx, modules, task, f, spec, factory=None):
    create = factory or modules['fit'].create_model
    timings, inference_timings = [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model, norm = create(spec, task, 10482, f)
        origins = modules['data'].origins((0,10482), 512,48,1)
        windows = modules['training'].FastWindows(dict(task, physical=task['physical'][:10482],
                      observed=task['observed'][:10482]), norm['mean'], norm['std'], 'cuda')
        counts = windows.target_counts(origins)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        torch.cuda.reset_peak_memory_stats()
        for index in range(6):
            check_deadline(ctx)
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for lo in range(0,16,4):
                x,y,m = windows.get(origins[index*16+lo:index*16+lo+4])
                p = model(x,f)
                loss = modules['training'].macro_loss(p,y,m,counts,len(origins),16)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            torch.cuda.synchronize()
            if index:
                timings.append(time.perf_counter()-start)
        peak = torch.cuda.max_memory_allocated()
        model.eval()
        with torch.no_grad():
            x,_,_=windows.get(origins[:16])
            model(x,f)
            torch.cuda.synchronize()
            for _ in range(5):
                tick=time.perf_counter()
                model(x,f)
                torch.cuda.synchronize()
                inference_timings.append(time.perf_counter()-tick)
        combined_peak=torch.cuda.max_memory_allocated()
        del model, windows, optimizer
    f.assert_frozen_unchanged()
    return dict(warmup_updates=1, measured_updates=5, effective_batch=16,
                seconds=timings, median_seconds=float(np.median(timings)), max_seconds=max(timings),
                forward16_seconds=inference_timings,forward16_max_seconds=max(inference_timings),
                peak_allocated_bytes=combined_peak,training_peak_allocated_bytes=peak,
                disposable_model=True, main_RNG_restored=True)


def select_common_endpoints(checkpoints, origin_count, effective_batch):
    epochs = existing.EPOCHS
    targets = [math.ceil(epoch*origin_count) for epoch in epochs]
    expected_updates = [0]
    for previous, target in zip(targets, targets[1:]):
        expected_updates.append(expected_updates[-1]+math.ceil((target-previous)/effective_batch))
    selected = {}
    for path, checkpoint in checkpoints:
        extra = checkpoint.get('extra', {})
        completed = extra.get('completed_epochs', [])
        curves = extra.get('curves', [])
        if not completed or not curves or completed[-1] not in epochs:
            continue
        index = epochs.index(completed[-1])
        if list(completed) != list(epochs[:index+1]):
            continue
        curve = curves[-1]
        expected = (epochs[index], targets[index], expected_updates[index])
        actual = (completed[-1], checkpoint.get('sampler', {}).get('exposures'), checkpoint.get('updates'))
        progress = (curve.get('epoch'), curve.get('exposures'), curve.get('updates'))
        if actual != expected or progress != expected:
            continue
        if index in selected:
            raise ValueError('Duplicate registered common endpoint: '+str(epochs[index]))
        selected[index] = (path, checkpoint)
    missing = [epoch for index, epoch in enumerate(epochs) if index not in selected]
    if missing:
        raise ValueError('Missing registered common endpoint: '+str(missing))
    return [selected[index] for index in range(len(epochs))]


def initialization(job, ctx):
    modules = existing._modules(ctx)
    task = existing._jena(ctx, modules)
    f = foundation(ctx, modules)
    seed = int(job['seed'])
    original = modules['fit'].create_model
    initial_states = {}
    def factory(spec, task, end, backbone):
        model, norm = original(spec, task, end, backbone)
        p_state = {k:v for k,v in model.state_dict().items() if k.startswith('front.')}
        p_hash = modules['training'].fingerprint({k:modules['data'].array_sha256(v.cpu().numpy()) for k,v in p_state.items()})
        if spec.get('init_kind') == 'QR':
            basis = qr_basis(21,6,10000000+seed)
            with torch.no_grad():
                model.E.copy_(torch.as_tensor(basis, device='cuda'))
                model.D.zero_()
                model.D[0].copy_(torch.linalg.pinv(model.E))
            norm['basis'] = basis
            norm['basis_initialization'] = 'Gaussian_reduced_QR_separate_CPU_generator_fixed_signs'
        initial_states[(spec.get('init_kind','PCA'),end)] = dict(
            P_initial_sha256=p_hash, E_initial=model.E.detach().cpu().numpy().copy(),
            D_initial_sha256=modules['data'].array_sha256(model.D.detach().cpu().numpy()),
            center_sha256=modules['data'].array_sha256(norm['pca_center']))
        return model,norm
    base_spec = dict(run_id='',dataset='jena',family='B2',latent_k=6,seed=seed,fold='primary',
                     epochs=[.25,1.,3.],lr=.001,device='cuda',microbatch=4,effective_batch=16,
                     F_row_microbatch=128,checkpoint_inputs=False, custom_source_sha256=sha(__file__))
    profile = profile_training(ctx,modules,task,f,dict(base_spec,init_kind='PCA'),factory)
    # Both arms include complete search and worst-case 3-epoch A60 refit, plus evaluation/load margin.
    endpoint_eval_cost=2*4*math.ceil(9923/16)*profile['forward16_max_seconds']*1.5
    estimated = 2*(1863+2254)*profile['max_seconds']*1.5+endpoint_eval_cost+120
    profile['conservative_pair_seconds'] = estimated
    profile_path = ctx['report']/'profiles'/(job['job_id']+'.json')
    atomic_json(profile_path,profile)
    remaining = min(ctx.get('job_deadline',ctx['deadline']),ctx['deadline']-4320)-time.time()
    if estimated > remaining:
        return dict(status='SKIPPED_BUDGET',reason='Measured complete matched PCA/QR pair exceeds reserve',
                    artifact_paths=[str(profile_path)],result=profile)
    rows, diagnostics, private_artifacts, sampler_checks, training_curves, endpoint_curves = [], [], [], {}, [], []
    for kind in ('PCA','QR'):
        check_deadline(ctx)
        spec = dict(base_spec,run_id=f'init_{kind}_s{seed}',init_kind=kind)
        modules['fit'].create_model=factory
        original_splits=modules['data'].split_ranges
        modules['data'].split_ranges=registered_jena_splits
        original_segment=modules['training'].train_segment
        def observed_segment(model,windows,origins,sampler,*a,**kw):
            segment=original_segment(model,windows,origins,sampler,*a,**kw)
            training_curves.append(dict(initialization=kind,fit_end=int(max(origins)+task['H']),
                updates=segment['updates'],exposures=segment['exposures'],
                mean_observed_training_batch_loss=segment.get('mean_observed_training_batch_loss'),
                loss_scope='mean stochastic training-batch loss within endpoint segment; not exact endpoint loss'))
            return segment
        modules['training'].train_segment=observed_segment
        try:
            receipt = modules['fit'].fit_base(spec,task,ctx['cache'],ctx['report'],f,
                                            min(ctx['deadline']-4320,ctx.get('job_deadline',ctx['deadline'])))
        finally:
            modules['fit'].create_model=original
            modules['data'].split_ranges=original_splits
            modules['training'].train_segment=original_segment
        if receipt['status'] != 'COMPLETE_VALID':
            return dict(status='PARTIAL_CHECKPOINTED',reason='Matched pair checkpointed at finite deadline',
                        artifact_paths=[str(profile_path)],result=dict(profile=profile,completed_arms=rows))
        path = ctx['cache']/receipt['artifact_cache_relative_path']
        bundle = torch.load(path,weights_only=False,map_location='cpu')
        train_origins=modules['data'].origins((0,10482),512,48,1)
        checkpoints=[(checkpoint_path,torch.load(checkpoint_path,map_location='cpu',weights_only=False))
                     for checkpoint_path in sorted((path.parent/'search').glob('checkpoint_*.pt'))]
        for checkpoint_path,checkpoint in select_common_endpoints(checkpoints,len(train_origins),spec['effective_batch']):
            epoch=checkpoint['extra']['completed_epochs'][-1]
            sampler_hash=digest(checkpoint['sampler'])
            sampler_checks.setdefault(str(epoch),{})[kind]=sampler_hash
            private_artifacts.append(private_binding(checkpoint_path,ctx))
            endpoint_model,endpoint_norm=factory(spec,task,10482,f)
            endpoint_model.load_state_dict(checkpoint['model'])
            endpoint_metrics=training_endpoint(endpoint_model,endpoint_norm,task,train_origins,f,modules,ctx)
            endpoint_curves.append(dict(initialization=kind,seed=seed,epoch=epoch,updates=checkpoint['updates'],
                exposures=checkpoint['sampler']['exposures'],A_fit_MSE=endpoint_metrics['MSE'],
                A_fit_MAE=endpoint_metrics['MAE'],A_stop_MSE=checkpoint['extra']['curves'][-1]['astop_mse'],
                train_eval_origin_stride=1,trainable_params=35717,**endpoint_metrics))
            del endpoint_model
        if kind=='QR':
            for paired in sampler_checks.values():
                if paired.get('PCA')!=paired.get('QR'):
                    raise ValueError('PCA/QR sampler/order/RNG state differs at common endpoint')
        model,norm=factory(spec,task,12579,f)
        model.load_state_dict(bundle['model'])
        origins=modules['data'].origins((14675,16772),512,48,48)
        prediction_path=ctx['cache']/'initialization_predictions'/f'{kind}_{seed}.npz'
        metrics=predict_saved(model,norm,task,origins,f,modules,prediction_path,ctx)
        private_artifacts.extend([private_binding(path,ctx),private_binding(prediction_path,ctx)])
        initial=initial_states[(kind,12579)]
        first=initial['E_initial'].astype(np.float64)
        final=model.E.detach().cpu().numpy().astype(np.float64)
        singular=np.linalg.svd(final,compute_uv=False)
        cosines=np.linalg.svd(np.linalg.qr(first)[0].T@np.linalg.qr(final)[0],compute_uv=False)
        diagnostics.append(dict(initialization=kind,seed=seed,init_seed=10000000+seed if kind=='QR' else None,
                           E_initial_sha256=modules['data'].array_sha256(first),
                           E_final_sha256=modules['data'].array_sha256(final),
                           P_initial_sha256=initial['P_initial_sha256'],center_sha256=initial['center_sha256'],
                           search_P_initial_sha256=initial_states[(kind,10482)]['P_initial_sha256'],
                           initial_orthogonality_max=float(np.abs(first.T@first-np.eye(6)).max()),
                           final_singular_values=singular.tolist(), condition=float(singular[0]/singular[-1]),
                           E_frobenius_change=float(np.linalg.norm(final-first)),
                           subspace_angles_radians=np.arccos(np.clip(cosines,-1,1)).tolist(),
                           selected_epoch=receipt['selected_epoch'], paired_device='cuda'))
        rows.append(dict(method='B2_'+kind,seed=seed,dataset='jena',C=21,K=6,L=512,H=48,
                         device='cuda',split='V_select',fit='fresh_A60',selected_epoch=receipt['selected_epoch'],
                         trainable_params=35717,refit_updates=receipt['refit_updates'],
                         refit_exposures=receipt['refit_exposures'],prediction_sha256=sha(prediction_path),**metrics))
        del model
    if any(diagnostics[0][key]!=diagnostics[1][key] for key in ('P_initial_sha256','search_P_initial_sha256','center_sha256')):
        raise ValueError('PCA/QR matched P or centering initial state differs')
    value=dict(status='COMPLETE',seed=seed,rows=rows,diagnostics=diagnostics,profile=profile,private_artifacts=private_artifacts,
               common_endpoint_sampler_fingerprints=sampler_checks,training_curves=training_curves,
               exact_endpoint_curves=endpoint_curves,
               pairing='Both fresh CUDA; same seed, P, center, stream/order, loss, optimizer and endpoints; only E initialization changes',
               selected_refit_is_not_common_update=True)
    output=ctx['report']/'jobs'/(job['job_id']+'.json')
    atomic_json(output,value)
    f.assert_frozen_unchanged()
    del f
    torch.cuda.empty_cache()
    return dict(status='COMPLETE',reason=None,artifact_paths=[str(output),str(profile_path)],result=value)


def execute_job(job, ctx):
    if job['question_id']=='B03':
        return decoder_refit(job,ctx)
    if job['question_id']=='C02':
        return initialization(job,ctx)
    if job['question_id']=='E02':
        from .costs import measure_jena
        return measure_jena(job,ctx)
    raise ValueError('Unknown new diagnostic job')
def registered_jena_splits(task, fold='primary'):
    expected=dict(A_fit=(0,10482),A_stop=(10482,12579),M_cal=(12579,14675),
                  V_select=(14675,16772),T_final=(16772,20965))
    if fold!='primary' or {k:tuple(v) for k,v in task['split_ranges'].items()}!=expected:
        raise ValueError('Original registered Jena boundaries changed')
    return expected


def private_binding(path, ctx):
    return dict(reference='cache://'+Path(path).relative_to(ctx['cache']).as_posix(),sha256=sha(path))


def cohort_metadata(values, norm, modules):
    arrays={key:existing._array_fingerprint(value) for key,value in values.items()}
    return dict(array_fingerprints=arrays,scaler_sha256=modules['fit'].scaler_fingerprint(norm['mean'],norm['std']),
                target_sha256=arrays['target_normalized']['sha256'],mask_sha256=arrays['observed']['sha256'],
                origins_sha256=arrays['origins']['sha256'],channel_sha256=arrays['channel_ids']['sha256'],
                dtype=str(values['pred_normalized'].dtype),point_statistic='quantile_0.5',
                evaluation_array_shape=list(values['target_normalized'].shape))


def training_endpoint(model,norm,task,origins,f,modules,ctx):
    windows=modules['training'].FastWindows(task,norm['mean'],norm['std'],'cuda')
    sse=np.zeros(21,np.float64);sae=np.zeros(21,np.float64);counts=np.zeros(21,np.int64)
    gpu_sse=torch.zeros(21,dtype=torch.float64,device='cuda')
    model.eval()
    with torch.no_grad():
        for start in range(0,len(origins),16):
            check_deadline(ctx)
            x,y,mask=windows.get(origins[start:start+16])
            prediction=model(x,f)
            difference=prediction.double()-y.double()
            gpu_sse+=torch.where(mask,difference.square(),0).sum((0,1))
            p,t,m=prediction.cpu().numpy(),y.cpu().numpy(),mask.cpu().numpy()
            error=np.where(m,p.astype(np.float64)-t.astype(np.float64),0.)
            sse+=np.square(error).sum((0,1),dtype=np.float64)
            sae+=np.abs(error).sum((0,1),dtype=np.float64)
            counts+=m.sum((0,1),dtype=np.int64)
    active=counts>0
    mse=math.fsum((sse[active]/counts[active]).tolist())/int(active.sum())
    native=float((gpu_sse[active]/torch.as_tensor(counts[active],device='cuda')).mean().cpu())
    if abs(mse-native)>1e-12:
        raise ValueError('C02 exact train endpoint FP64 scorers disagree')
    return dict(MSE=mse,MAE=math.fsum((sae[active]/counts[active]).tolist())/int(active.sum()),
                channel_sse=sse.tolist(),channel_sae=sae.tolist(),channel_counts=counts.tolist(),
                normalized_scale='search_A_fit',FP64_CPU_GPU_aggregate_difference=abs(mse-native))

