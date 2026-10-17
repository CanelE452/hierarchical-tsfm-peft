"""Conditional PeMS32 transfer of all four fixed-PE readouts."""
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch

from . import decoder, decoder_jobs
from .contracts import (atomic_json, binding, digest, read, reusable, resolve_binding, sha,
                        verify_references)
from research.adapter_diagnostics_20261017_v2.new_jobs import save_npz


def select_parent(job, ctx):
    identifier = 'B2_PCA_PEMS32_s92601'
    path = Path(ctx['report'])/'jobs'/(identifier+'.json')
    if path.is_file() and read(path).get('status') == 'COMPLETE':
        registered = next((row for row in ctx.get('plan', {}).get('jobs', []) if row['job_id'] == identifier), None)
        if registered is None or not reusable(Path(ctx['report'])/'receipts'/(identifier+'.json'), registered, ctx):
            raise ValueError('Fresh PCA parent does not have a verified complete sealed receipt')
        parent = read(path)
        epoch = parent['fit_receipt']['selected_epoch']
        endpoints = [row for row in parent['common_endpoint_rows'] if row['epoch'] == epoch and row['evaluation'] == 'A_stop']
        if len(endpoints) != 1:
            raise ValueError('Fresh PCA selected A_fit checkpoint is missing or ambiguous')
        checkpoint, bundle = endpoints[0]['checkpoint_binding'], parent['bundle_binding']
        return dict(kind='FRESH_CUDA_PCA', parent_job_id=identifier, selected_epoch=epoch,
                    A_fit_checkpoint=resolve_binding(checkpoint, ctx), A60_bundle=resolve_binding(bundle, ctx),
                    A_fit_checkpoint_binding=checkpoint, A60_bundle_binding=bundle, parent_receipt=binding(path, ctx))
    if not job.get('fallback_A_fit_checkpoint_binding'):
        raise FileNotFoundError('BLOCKED_ASSET: fallback_A_fit_checkpoint_binding must be pinned before execution')
    from .audit import old_context
    from research.adapter_diagnostics_20261017_v2.core import validate_plan
    old = old_context(ctx)
    old_plan = read(Path(old['report'])/'EXPERIMENT_PLAN.json')
    validate_plan(old_plan, ctx['repo'])
    identifier = 'C01_pems32_cuda_B2'
    registered = next(row for row in old_plan['jobs'] if row['job_id'] == identifier)
    receipt_path = Path(old['report'])/'receipts'/(identifier+'.json')
    receipt = read(receipt_path)
    if receipt['status'] != 'COMPLETE' or receipt['scientific_identity'] != registered['scientific_identity']:
        raise ValueError('Historical CUDA branch is not complete under its sealed identity')
    for relative, expected in receipt['artifact_hashes'].items():
        resolve_binding(dict(reference='report://'+relative, sha256=expected), old)
    handler_path = Path(old['report'])/'jobs'/(identifier+'_handler.json')
    handler = read(handler_path)
    for item in handler['artifact_hashes']:
        resolve_binding(dict(reference=item['path'], sha256=item['sha256']), old)
    if handler['result'] != receipt['result']:
        raise ValueError('Old handler and parent receipt disagree')
    bundle = next(item for item in handler['artifact_hashes'] if item['path'].endswith('/selected_bundle.pt'))
    bundle_path = resolve_binding(dict(reference=bundle['path'], sha256=bundle['sha256']), old)
    checkpoint_path = resolve_binding(job['fallback_A_fit_checkpoint_binding'], ctx)
    return dict(kind='CPU_TO_CUDA_NUMERICAL_BRANCH', parent_job_id=identifier,
                selected_epoch=receipt['result']['selected_epoch'],
                A_fit_checkpoint=checkpoint_path, A60_bundle=bundle_path,
                A_fit_checkpoint_binding=binding(checkpoint_path, ctx), A60_bundle_binding=binding(bundle_path, ctx),
                parent_receipt=binding(receipt_path, ctx), historical_bitwise_equivalence_claimed=False)


def _synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def _counter(foundation, name):
    value = getattr(foundation, name, None)
    return None if value is None else int(value)


def extract_cache(ctx, modules, task, model, norm, origins, foundation, checkpoint_binding,
                  label, check, batch_size=16, prefix=None):
    origins = np.asarray(origins, np.int64)
    l, h, k, c = task['L'], task['H'], model.latent_k, task['physical'].shape[1]
    if len(origins) == 0 or not np.array_equal(np.asarray(task['columns']).shape, (c,)):
        raise ValueError('Empty extraction scope or channel order mismatch')
    source_files = {name: sha(path) for name, path in [('decoder_transfer.py', Path(__file__)), ('decoder.py', Path(decoder.__file__))]}
    for relative in ('gpu_runtime.py', 'training.py', 'networks.py', 'data_contract.py'):
        path = Path(modules['campaign'])/relative
        if path.is_file():
            source_files[relative] = sha(path)
    norm_source = Path(modules['campaign']).parent/'temporal_frontend_campaign_20261015_v1'/'runtime.py'
    if norm_source.is_file():
        source_files['SafeInstanceNorm_runtime.py'] = sha(norm_source)
    key = dict(scientific_sources=source_files, PE_checkpoint=checkpoint_binding,
               scaler_sha=modules['fit'].scaler_fingerprint(norm['mean'], norm['std']),
               weights_sha=foundation.initial_hash, source_data_sha=task['data_sha256'],
               prefix_data_sha=decoder_jobs.array_sha(task['physical']),
               prefix_mask_sha=decoder_jobs.array_sha(task['observed']),
               subset_identity=task.get('subset_identity_payload'),
               instance_norm_source_sha256=getattr(foundation, 'instance_norm_source_sha256', None),
               origins_sha=decoder_jobs.array_sha(origins), channel_order=list(task['columns']),
               torch_version=torch.__version__, numpy_version=np.__version__, L=l, H=h, K=k, C=c,
               dtype='float32', device=str(model.E.device), point='config_quantile_0.5',
               feature='lag-major(0,1,2,4,8,16,32) joined Z,U minus Z[-1-lag]',
               stage=label, backbone_dtype='float32')
    directory = Path(ctx['cache'])/'decoder_transfer_features'
    path = directory/(label+'_'+digest(key)+'.npz')
    manifest_path = path.with_suffix('.json')
    if path.exists() or manifest_path.exists():
        values = decoder_jobs.load_cache(path, key, (len(origins), h, 7*k, c))
        original = read(manifest_path)
        return values, dict(original, artifact=binding(path, ctx), new_F_calls=0, new_F_rows=0,
                            reused=True, current_extraction_seconds=0.)
    windows = modules['training'].FastWindows(task, norm['mean'], norm['std'], str(model.E.device))
    values = {name: [] for name in ('phi', 'target', 'mask', 'anchor')}
    offset = 0
    if prefix is not None:
        prefix_values, prefix_origins = prefix
        offset = len(prefix_origins)
        if not np.array_equal(origins[:offset], prefix_origins):
            raise ValueError('Probe prefix origins do not match full extraction scope')
        for name in values:
            values[name].append(prefix_values[name])
    calls_before, rows_before = _counter(foundation, 'row_calls'), _counter(foundation, 'streams')
    python_calls = 0
    _synchronize(model.E.device)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(offset, len(origins), batch_size):
            check()
            x, target, observed = windows.get(origins[start:start+batch_size])
            z = model.encode(x)
            future = foundation(z)
            python_calls += 1
            if z.dtype != torch.float32 or future.dtype != torch.float32 or tuple(future.shape) != (len(x), h, k):
                raise ValueError('Frozen FP32 foundation extraction shape/dtype mismatch')
            joined = torch.cat((z, future), 1)
            phi = torch.cat([joined[:, l-lag:l-lag+h]-z[:, -1-lag, None] for lag in (0,1,2,4,8,16,32)], -1)
            for name, value in dict(phi=phi, target=target, mask=observed, anchor=x[:, -1:]).items():
                values[name].append(value.detach().cpu().numpy())
    _synchronize(model.E.device)
    seconds = time.perf_counter()-started
    result = {name: np.concatenate(parts) for name, parts in values.items()}
    new_rows = (len(origins)-offset)*k
    if rows_before is not None and _counter(foundation, 'streams')-rows_before != new_rows:
        raise ValueError('Feature extraction did not use exactly one K-row foundation pass per origin')
    foundation.assert_frozen_unchanged()
    save_npz(path, **result)
    value = dict(key=key, sha256=sha(path), extraction_seconds=seconds, current_extraction_seconds=seconds,
                 n_origins=len(origins), new_F_calls=python_calls, new_F_rows=new_rows,
                 F_row_calls=None if calls_before is None else _counter(foundation, 'row_calls')-calls_before,
                 reused=False, probe_prefix_origins_reused=offset)
    atomic_json(manifest_path, value)
    decoder_jobs.load_cache(path, key, (len(origins), h, 7*k, c))
    return result, dict(value, artifact=binding(path, ctx), manifest=binding(manifest_path, ctx))


def admission_estimate(probes, stage_counts, remaining_seconds):
    per_origin = max(row['seconds']/row['n_origins'] for row in probes)
    extraction = 1.5*per_origin*sum(stage_counts.values())
    # Fixed conservative reserve for 40 small readout solves, raw-design QR,
    # serialization and checks; the extraction rate comes from both PE scopes.
    fitting = extraction+180.
    complete = fitting/.8
    available = min(1800., remaining_seconds)
    return dict(estimated_extraction_seconds=extraction, estimated_fitting_seconds=fitting,
                estimated_complete_seconds=complete, verification_reserve_fraction=.2,
                stage_origin_counts=stage_counts,
                admitted=complete <= available and fitting <= max(0., available-360.),
                probe_is_cost_admission_only=True, head_settings=40, arms=list(decoder.ARMS))


def _resources(job, ctx):
    from .audit import load_modules
    modules = ctx.get('modules') or load_modules(ctx)
    if 'task' in ctx:
        task = ctx['task']
    elif 'task_loader' in ctx:
        task = ctx['task_loader'](ctx, modules, 'pems32')
    else:
        task = ctx['load_task']('pems32')
    if (task['L'], task['H'], task['physical'].shape[1], job.get('K', 8), int(job['seed'])) != (512, 12, 32, 8, 92601):
        raise ValueError('Transfer is registered only for PeMS32/L512/H12/K8/seed92601')
    parent = select_parent(job, ctx)
    foundation = ctx.get('foundation')
    if foundation is None:
        foundation = ctx['foundation_loader'](ctx, modules, 12, job.get('device', 'cuda'))
    return modules, task, parent, foundation


def _load_models(ctx, modules, task, parent, foundation):
    from .training_jobs import model_factory
    checkpoint = torch.load(parent['A_fit_checkpoint'], map_location='cpu', weights_only=False)
    bundle = torch.load(parent['A60_bundle'], map_location='cpu', weights_only=False)
    epoch = bundle.get('selected_search_epoch', bundle.get('selected_epoch'))
    if epoch != parent['selected_epoch'] or checkpoint['extra']['completed_epochs'][-1] != epoch:
        raise ValueError('Selected A_fit PE checkpoint and A60 bundle epoch differ')
    bounds = task['split_ranges']
    specification = dict(bundle['spec'], device=str(getattr(foundation, 'device', 'cuda')))
    if (specification['seed'], specification['latent_k'], specification['family'], specification.get('init_kind', 'PCA')) != (92601,8,'B2','PCA'):
        raise ValueError('Parent must be the registered seed92601 B2/PCA/K8 recipe')
    for key in ('source_sha','config_sha','data_sha','model_sha'):
        if checkpoint['contract'][key] != bundle['contract'][key]:
            raise ValueError('A_fit and A60 parent base contracts differ: '+key)
    context = model_factory(dict(ctx, modules=modules)) if parent['kind'] == 'FRESH_CUDA_PCA' else nullcontext(modules['fit'].create_model)
    with context as create:
        fit, fit_norm = create(specification, task, int(bounds['A_fit'][1]), foundation)
        refit, refit_norm = create(specification, task, int(bounds['A_stop'][1]), foundation)
        for model, norm, payload in ((fit, fit_norm, checkpoint), (refit, refit_norm, bundle)):
            contract = payload['contract']
            for key, expected in (('model_sha', foundation.initial_hash), ('data_sha', task['data_sha256']),
                                  ('scaler_sha', modules['fit'].scaler_fingerprint(norm['mean'], norm['std']))):
                if contract[key] != expected:
                    raise ValueError('Parent PE contract differs: '+key)
            if contract['source_sha'] != modules['fit']._source_contract()[0]:
                raise ValueError('Actual parent producer source contract changed')
            model.load_state_dict(payload['model'], strict=True)
            model.eval().requires_grad_(False)
    for key in ('mean', 'std'):
        if not np.array_equal(refit_norm[key], bundle['normalization'][key]):
            raise ValueError('Fresh A60 reconstructed scaler differs from selected bundle')
    return fit, fit_norm, refit, refit_norm


def execute_job(job, ctx):
    started = time.time()
    check = decoder_jobs.budget_checker(dict(ctx, decoder_family_started_unix=started), started)
    verify_check = decoder_jobs.budget_checker(dict(ctx, decoder_family_started_unix=started), started, verification=True)
    modules, task, parent, foundation = _resources(job, ctx)
    fit_model, fit_norm, refit_model, refit_norm = _load_models(ctx, modules, task, parent, foundation)
    bounds = task['split_ranges']
    stages = {
        'A_fit_fit': (fit_model, fit_norm, modules['data'].origins(bounds['A_fit'], 512,12,1), parent['A_fit_checkpoint_binding']),
        'A_fit_A_stop': (fit_model, fit_norm, modules['data'].origins(bounds['A_stop'],512,12,12), parent['A_fit_checkpoint_binding']),
        'A60_fit': (refit_model, refit_norm, modules['data'].origins((0,bounds['A_stop'][1]),512,12,1), parent['A60_bundle_binding']),
        'A60_V_select': (refit_model, refit_norm, modules['data'].origins(bounds['V_select'],512,12,12), parent['A60_bundle_binding'])}
    root = Path(ctx['cache'])/'decoder_transfer'/job['job_id']
    root.mkdir(parents=True, exist_ok=True)
    output = Path(ctx['report'])/'jobs'/(job['job_id']+'.json')
    receipts, probes = [], []
    prefixes = {}
    science = {p.name:sha(p) for p in (Path(__file__), Path(decoder.__file__), Path(decoder_jobs.__file__))}
    for name in ('A_fit_fit', 'A60_fit'):
        model, norm, origins, checkpoint = stages[name]
        probe_origins = origins[:min(16, len(origins))]
        values, receipt = extract_cache(ctx, modules, task, model, norm, probe_origins, foundation, checkpoint,
                                        job['job_id']+'_probe_'+name, check)
        prefixes[name] = values, probe_origins
        receipts.append(receipt)
        probes.append(dict(scope=name, n_origins=len(probe_origins), seconds=max(receipt['extraction_seconds'], 1e-6), artifact=receipt['artifact']))
    remaining = min(ctx['deadline'], ctx.get('job_deadline', started+1800), started+1800)-time.time()
    estimate = admission_estimate(probes, {name:len(row[2]) for name,row in stages.items()}, remaining)
    atomic_json(root/'PROFILE.json', dict(estimate, probes=probes, parent_kind=parent['kind']))
    if not estimate['admitted']:
        value = dict(status='SKIPPED_BUDGET', reason='All four extraction/head/verification stages exceed remaining registered budget',
                     profile=estimate, parent={key:value for key,value in parent.items() if not isinstance(value, Path)},
                     new_F_calls=sum(r['new_F_calls'] for r in receipts), new_F_rows=sum(r['new_F_rows'] for r in receipts),
                     extraction_gpu_seconds=sum(r['current_extraction_seconds'] for r in receipts), private_artifacts=[binding(root/'PROFILE.json', ctx)]+[r['artifact'] for r in receipts])
        atomic_json(output, value)
        return dict(status=value['status'], reason=value['reason'], artifact_paths=[str(output)], result=value)
    def extract(name):
        model, norm, origins, checkpoint = stages[name]
        values, receipt = extract_cache(ctx, modules, task, model, norm, origins, foundation, checkpoint,
                                        job['job_id']+'_'+name, check, prefix=prefixes.get(name))
        receipts.append(receipt)
        return values
    fit, stop, refit = (extract(name) for name in ('A_fit_fit', 'A_fit_A_stop', 'A60_fit'))
    seal_path = root/'SELECTION_SEAL.json'
    def seal(selection):
        deterministic = dict(selection, selection_trials=[{key:value for key,value in trial.items() if key != 'seconds'}
                                                        for trial in selection['selection_trials']])
        body = dict(deterministic, source_hashes=science, scientific_identity=job['scientific_identity'],
                    V_features_loaded=False, parent_A_fit=parent['A_fit_checkpoint_binding'], parent_A60=parent['A60_bundle_binding'])
        body['selection_sha256'] = digest(body)
        if seal_path.exists() and read(seal_path) != body:
            raise ValueError('Existing immutable transfer selection seal differs')
        if not seal_path.exists():
            atomic_json(seal_path, body)
    result = decoder_jobs.run_family(fit, stop, refit, lambda:extract('A60_V_select'), seal, check)
    private = [binding(root/'PROFILE.json', ctx), binding(seal_path, ctx)]+[r['artifact'] for r in receipts]
    validation = result['validation']
    baseline = next(row for row in result['rows'] if row['arm'] == 'D_SHARED')
    retained = sum(p.numel() for p in refit_model.front.parameters())+refit_model.E.numel()
    for row in result['rows']:
        verify_check()
        arm, state = row['arm'], result['states'][row['arm']]
        pred_path, state_path, coefficient_path = root/(arm+'_predictions.npz'), root/(arm+'_state.json'), root/(arm+'_coefficients.npz')
        arrays = dict(pred_normalized=result['predictions'][arm], target_normalized=validation['target'], observed=validation['mask'],
                      origins=stages['A60_V_select'][2], channel_ids=np.asarray(task['columns']))
        save_npz(pred_path, **arrays)
        atomic_json(state_path, state)
        coefficients = dict(coefficients=np.asarray(state['coefficients']), rms=np.asarray(state['rms']), zero_features=np.asarray(state['zero_features']))
        if state['gain'] is not None:
            coefficients['gain'] = np.asarray(state['gain'])
        save_npz(coefficient_path, **coefficients)
        with np.load(pred_path, allow_pickle=False) as archive:
            replay = decoder_jobs.score_arrays(archive['pred_normalized'], archive['target_normalized'], archive['observed'])
        if any(abs(replay[key]-row[key]) > 1e-12 for key in ('MSE', 'MAE')):
            raise ValueError('Transfer saved-array FP64 rescore differs')
        private.extend(binding(path, ctx) for path in (pred_path,state_path,coefficient_path))
        paired = [dict(origin_index=a['origin_index'], delta_MSE=None if a['MSE'] is None else a['MSE']-b['MSE'],
                       delta_MAE=None if a['MAE'] is None else a['MAE']-b['MAE'], channel_counts=a['channel_counts'],
                       channel_delta_sse=(np.asarray(a['channel_sse'])-b['channel_sse']).tolist(),
                       channel_delta_sae=(np.asarray(a['channel_sae'])-b['channel_sae']).tolist())
                  for a,b in zip(row['origin_statistics'],baseline['origin_statistics'])]
        row.update(dataset='pems32', seed=92601, C=32,K=8,L=512,H=12,device=job.get('device','cuda'),
                   units='normalized_A60',point_statistic='quantile_0.5',
                   frozen_PE_target_trained_params=retained,total_target_trained_params=retained+row['retained_decoder_parameters'],
                   prediction_binding=binding(pred_path,ctx),state_binding=binding(state_path,ctx),coefficient_binding=binding(coefficient_path,ctx),
                   prediction_sha256=sha(pred_path),array_fingerprints={key:dict(sha256=decoder_jobs.array_sha(value),shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()},
                   parent_bundle_binding=parent['A60_bundle_binding'],parent_checkpoint_binding=parent['A_fit_checkpoint_binding'],
                   parent_kind=parent['kind'],parent_job_id=parent['parent_job_id'],selected_epoch=parent['selected_epoch'],
                   scaler_sha256=modules['fit'].scaler_fingerprint(refit_norm['mean'],refit_norm['std']),
                   saved_array_independent_FP64='PASS',selection_seal=binding(seal_path,ctx),
                   origin_paired_vs_shared=paired, coefficient_norm=float(np.linalg.norm(state['coefficients'])),
                   fitting_diagnostics=state['diagnostics'],identifiability=state['identifiability'],
                   matching_status='MATCHED_FIXED_PE_FOUR_HEAD_RECIPE',V_exposure='repeated development evaluation')
    if {name:sha(Path(__file__).parent/name) for name in science} != science:
        raise ValueError('Transfer scientific source changed after selection seal')
    foundation.assert_frozen_unchanged()
    value = dict(status='COMPLETE', rows=result['rows'],selection_trials=result['selection_trials'],selections=result['selections'],
                 private_artifacts=private,feature_caches=receipts,parent={key:value for key,value in parent.items() if not isinstance(value,Path)},
                 source_hashes=science,profile=estimate,new_F_calls=sum(r['new_F_calls'] for r in receipts),
                 new_F_rows=sum(r['new_F_rows'] for r in receipts),extraction_gpu_seconds=sum(r['current_extraction_seconds'] for r in receipts),
                 F_weights_unchanged=True,F_weight_hash=foundation.initial_hash,timers=dict(result['timers'],total_wall_seconds=time.time()-started),
                 selection_sealed_before_V=True,all_four_heads_transferred=True,choice_independent_of_Jena_V=True,
                 cost_valid=False,cost_status='UNMEASURED exact FP64 deployable head; parent bundle/state bindings supplied')
    atomic_json(output,value)
    return dict(status='COMPLETE',reason=None,artifact_paths=[str(output)],result=value)
