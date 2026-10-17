"""Scoped factory and registered fresh search/refit jobs; import performs no I/O."""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
from research.adapter_diagnostics_20261017_v2 import new_jobs as previous
from .frontends import LocalFrontend


RECIPE = dict(epochs=[.25, 1., 3.], lr=.001, effective_batch=16, microbatch=4,
              F_row_microbatch=128, checkpoint_inputs=False)
NEW_SCIENCE = ('frontends.py', 'training_jobs.py', 'initialization.py', 'contracts.py')


def _modules(ctx):
    return ctx['modules'] if 'modules' in ctx else existing._modules(ctx)


def scientific_sources(ctx):
    """Actual new producer bytes, also included in fit_base's source/config seal."""
    root = Path(ctx['repo']).resolve()
    here = Path(__file__).resolve().parent
    return {path.relative_to(root).as_posix(): existing.sha(path)
            for name in NEW_SCIENCE if (path := here / name).is_file()}


def _registered_splits(task, fold='primary'):
    if fold != 'primary' or 'split_ranges' not in task:
        raise ValueError('Registered primary split boundaries required')
    bounds = {key: tuple(map(int, value)) for key, value in task['split_ranges'].items()}
    names = ('A_fit', 'A_stop', 'M_cal', 'V_select', 'T_final')
    if set(bounds) != set(names) or bounds['A_fit'][0] != 0:
        raise ValueError('Incomplete registered split boundaries')
    for index, name in enumerate(names):
        lo, hi = bounds[name]
        if lo >= hi or (index and lo != bounds[names[index-1]][1]):
            raise ValueError('Invalid chronological split boundaries')
    return bounds


def _dca_basis(ctx, task, fit_end, k, norm):
    loader = ctx.get('initialization_loader')
    if loader is None:
        from .initialization import load_initialization
        loader = load_initialization
    basis, metadata = loader(ctx, task, fit_end, k, norm)
    metadata = copy.deepcopy(metadata)
    scope = metadata.get('fit_scope', {})
    if not isinstance(scope, dict) or int(scope.get('fit_end', -1)) != int(fit_end):
        raise ValueError('DCA initialization fitting scope mismatch')
    if metadata.get('status') not in ('COMPLETE', 'CONVERGED', 'COMPLETE_VALID'):
        raise ValueError('DCA initialization is not converged')
    if not re.fullmatch(r'[a-fA-F0-9]{64}', metadata.get('cache_sha256', '')):
        raise ValueError('DCA initialization cache SHA is missing')
    data = _modules(ctx)['data']
    fit = _modules(ctx)['fit']
    required = dict(scaler_sha256=fit.scaler_fingerprint(norm['mean'], norm['std']),
                    center_sha256=data.array_sha256(norm['pca_center']),
                    pca_sha256=data.array_sha256(norm['basis']))
    for key, expected in required.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError('DCA initialization binding mismatch: ' + key)
    values = np.asarray(basis)
    if values.dtype != np.float32 or values.shape != (task['physical'].shape[1], k):
        raise ValueError('Saved DCA transform must be actual FP32 [C,K]')
    if not np.isfinite(values).all() or np.linalg.matrix_rank(values.astype(np.float64)) != k:
        raise ValueError('DCA initialization must have finite full rank K')
    if 'E32_sha256' in metadata and metadata['E32_sha256'] != data.array_sha256(values):
        raise ValueError('Saved DCA transform SHA mismatch')
    return values.copy(), metadata


@contextmanager
def model_factory(ctx):
    """One worker's shared factory for fitting, resume, reload and cost routes."""
    modules = _modules(ctx)
    fit, training, data = modules['fit'], modules['training'], modules['data']
    original_create, original_source = fit.create_model, fit._source_contract
    original_body, original_verify = fit._stage_body, training.verify_contract
    original_splits = data.split_ranges
    source_files = scientific_sources(ctx)

    def source_contract():
        _, base_files = original_source()
        actual = scientific_sources(ctx)
        return training.fingerprint(dict(base_files, **actual)), dict(base_files, **actual)

    def create(spec, task, fit_end, foundation):
        initialization_started = time.perf_counter()
        family = spec['family']
        if family not in ('P_SPLIT', 'P_LOCAL_CONTROL', 'B0', 'B2'):
            raise ValueError('Unregistered improvement family: ' + family)
        if scientific_sources(ctx) != source_files:
            raise ValueError('Scientific producer bytes changed inside factory scope')
        internal = dict(spec, family='B0' if family.startswith('P_') else family)
        model, norm = original_create(internal, task, fit_end, foundation)
        if family.startswith('P_'):
            model.front = LocalFrontend(task['physical'].shape[1], split=family == 'P_SPLIT',
                                        seed=int(spec['seed'])).to(device=model.E.device, dtype=model.E.dtype)
        norm['family'] = family
        norm['new_scientific_source_hashes'] = source_files
        kind = spec.get('init_kind', 'PCA')
        binding = dict(kind=kind, fit_scope={'fit_end': int(fit_end)}, cache_sha256=None)
        if kind == 'DCA':
            if family != 'B2':
                raise ValueError('DCA is registered only with existing B2')
            basis, metadata = _dca_basis(ctx, task, fit_end, int(spec['latent_k']), norm)
            norm['pca_basis'] = norm['basis']
            with torch.no_grad():
                model.E.copy_(torch.as_tensor(basis, device=model.E.device, dtype=model.E.dtype))
                model.D.zero_()
                model.D[0].copy_(torch.linalg.pinv(model.E))
            norm['basis'] = model.E.detach().cpu().numpy().copy()
            norm['basis_initialization'] = 'A_fitted_DCA_saved_FP32_PCA_Procrustes'
            binding.update(metadata, kind='DCA')
        elif kind != 'PCA':
            raise ValueError('Unregistered improvement initialization')
        norm['initialization_binding'] = binding
        event = dict(family=family, init_kind=kind, fit_end=int(fit_end),
                     initialization_seconds=time.perf_counter()-initialization_started,
                     P_initial_sha256=training.fingerprint({key:data.array_sha256(value.detach().cpu().numpy())
                                                           for key,value in model.front.state_dict().items()}),
                     E_initial_sha256=data.array_sha256(model.E.detach().cpu().numpy()),
                     D_initial_sha256=data.array_sha256(model.D.detach().cpu().numpy()),
                     center_sha256=data.array_sha256(norm['pca_center']),
                     scaler_sha256=fit.scaler_fingerprint(norm['mean'],norm['std']),
                     initialization_binding=binding)
        if 'initialization_events' in ctx:
            ctx['initialization_events'].append(event)
        return model, norm

    def stage_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir, foundation,
                   deadline, base_contract, model, normalization):
        binding = normalization['initialization_binding']
        extended = dict(base_contract, initialization_fit_scope=binding['fit_scope'],
                        initialization_cache_sha256=binding['cache_sha256'],
                        new_scientific_source_hashes=source_files)
        return original_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir,
                             foundation, deadline, extended, model, normalization)

    def verify(expected, actual):
        original_verify(expected, actual)
        for key in ('initialization_sha', 'initialization_fit_scope',
                    'initialization_cache_sha256', 'new_scientific_source_hashes'):
            if key in expected or key in actual:
                if expected.get(key) != actual.get(key):
                    raise ValueError('Resume/cache contract changed: ' + key)

    fit.create_model, fit._source_contract, fit._stage_body = create, source_contract, stage_body
    training.verify_contract, data.split_ranges = verify, _registered_splits
    try:
        yield create
    finally:
        fit.create_model, fit._source_contract, fit._stage_body = original_create, original_source, original_body
        training.verify_contract, data.split_ranges = original_verify, original_splits


def _spec(job, ctx):
    for key, expected in RECIPE.items():
        if key in job and job[key] != expected:
            raise ValueError('Unregistered recipe override: ' + key)
    if int(job['seed']) not in (92601, 92602):
        raise ValueError('Unregistered training seed')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(job['job_id'])):
        raise ValueError('Unsafe registered job ID')
    sources = scientific_sources(ctx)
    for key, expected in job.get('scientific_source_hashes', {}).items():
        if key in sources and sources[key] != expected:
            raise ValueError('Registered scientific source hash mismatch: ' + key)
    return dict(RECIPE, run_id=job['job_id'], dataset=job['cohort'], family=job['family'],
                init_kind=job.get('init_kind', 'PCA'), latent_k=int(job['K']), seed=int(job['seed']),
                device=job.get('device', 'cuda'), fold='primary', core_dtype='float32',
                registered_intervention={'family': job['family'], 'init_kind': job.get('init_kind', 'PCA')},
                new_scientific_source_hashes=sources, scientific_identity=job.get('scientific_identity', {}))


def _resources(job, ctx):
    modules = _modules(ctx)
    if 'task' in ctx:
        task = ctx['task']
    elif 'task_loader' in ctx:
        task = ctx['task_loader'](ctx, modules, job['cohort'])
    elif job['cohort'] == 'jena':
        task = existing._jena(ctx, modules)
    else:
        raise ValueError('Registered task_loader required for ' + job['cohort'])
    if 'foundation' in ctx:
        foundation = ctx['foundation']
    elif 'foundation_loader' in ctx:
        foundation = ctx['foundation_loader'](ctx, modules, task['H'], job.get('device', 'cuda'))
    else:
        foundation = previous.foundation(ctx, modules, task['H'], job.get('device', 'cuda'))
    return modules, task, foundation


def _check(ctx):
    if time.time() >= min(ctx['deadline'], ctx.get('job_deadline', ctx['deadline'])):
        raise TimeoutError('Registered job deadline reached')
    if ctx.get('lease'):
        ctx['lease'].check()


def _binding(path, ctx):
    return dict(reference='cache://' + Path(path).relative_to(Path(ctx['cache'])).as_posix(),
                sha256=existing.sha(path))


def _counter(foundation):
    return int(getattr(foundation, 'streams', 0))


def _predict_saved(model, norm, task, origins, foundation, modules, path, ctx):
    start = time.perf_counter()
    prefix_end = int(np.max(origins) + task['H']) if len(origins) else 0
    prefix = dict(task, physical=task['physical'][:prefix_end], observed=task['observed'][:prefix_end])
    windows = modules['training'].FastWindows(prefix, norm['mean'], norm['std'], str(model.E.device))
    predictions, targets, masks = [], [], []
    before = _counter(foundation)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(origins), 8):
            _check(ctx)
            x, y, observed = windows.get(origins[lo:lo+8])
            prediction = modules['fit'].predict_model(model, x, foundation, norm['family'])
            if prediction.dtype != torch.float32:
                raise ValueError('Registered core prediction dtype must be FP32')
            predictions.append(prediction.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(observed.cpu().numpy())
    if not predictions:
        raise ValueError('No registered evaluation origins')
    values = dict(pred_normalized=np.concatenate(predictions), target_normalized=np.concatenate(targets),
                  observed=np.concatenate(masks), origins=np.asarray(origins, np.int64),
                  channel_ids=np.asarray(task['columns']), mean=np.asarray(norm['mean']), std=np.asarray(norm['std']))
    metrics = existing.score_arrays(values['pred_normalized'], values['target_normalized'], values['observed'])
    if Path(path).exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(values) or any(not np.array_equal(archive[key],value,equal_nan=True)
                                                       if value.dtype.kind not in ('U','S') else not np.array_equal(archive[key],value)
                                                       for key,value in values.items()):
                raise ValueError('Existing saved prediction array changed: ' + Path(path).name)
    else:
        previous.save_npz(path, **values)
    with np.load(path, allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    independent = existing.score_arrays(stored['pred_normalized'], stored['target_normalized'], stored['observed'])
    if metrics != independent:
        raise ValueError('Saved-array independent FP64 metric mismatch')
    logical = _counter(foundation) - before
    expected_rows = len(origins) * model.latent_k
    if hasattr(foundation, 'streams') and logical != expected_rows:
        raise ValueError('Frontend branch doubled or changed logical F rows')
    fingerprints = {key: existing._array_fingerprint(value) for key, value in stored.items()}
    return dict(metrics, prediction_binding=_binding(path, ctx), independent_FP64='PASS',
                array_fingerprints=fingerprints, logical_F_rows=logical if hasattr(foundation, 'streams') else None,
                n_origins=len(origins), evaluation_stride=task['H'] if len(origins)<2 else int(origins[1]-origins[0]),
                scaler_sha256=modules['fit'].scaler_fingerprint(norm['mean'], norm['std']),
                prediction_dtype=str(stored['pred_normalized'].dtype), metric_accumulation_dtype='float64',
                core_dtype='float32', point_statistic='quantile_0.5',
                units=task.get('units', task.get('physical_units', 'legacy_units_UNRESOLVED')),
                evaluation_seconds=time.perf_counter()-start)


def _validate_loaded_initialization(checkpoint, norm):
    binding = norm['initialization_binding']
    for key, expected in (('initialization_fit_scope', binding['fit_scope']),
                          ('initialization_cache_sha256', binding['cache_sha256'])):
        if checkpoint['contract'].get(key) != expected:
            raise ValueError('Reload initialization contract changed: ' + key)


def execute_job(job, ctx):
    """Complete registered search, fresh A60 refit, checkpoint predictions and V."""
    specification = _spec(job, ctx)
    modules, task, foundation = _resources(job, ctx)
    bounds = _registered_splits(task)
    rows, endpoints, private = [], [], []
    initial = time.perf_counter()
    initialization_events = []
    preparation = 0.
    output = Path(ctx['report']) / 'jobs' / (job['job_id'] + '.json')
    deadline = min(ctx['deadline'], ctx.get('job_deadline', ctx['deadline']))
    try:
        with model_factory(dict(ctx, modules=modules, initialization_events=initialization_events)) as create:
            tick = time.perf_counter()
            receipt = modules['fit'].fit_base(specification, task, ctx['cache'], ctx['report'], foundation, deadline)
            fit_seconds = time.perf_counter()-tick
            if receipt['status'] != 'COMPLETE_VALID':
                result = dict(status='PARTIAL_CHECKPOINTED', recipe=RECIPE, fit_receipt=receipt,
                              run_id=job['job_id'], resumable=True, same_identity_resume=True,
                              source_files=scientific_sources(ctx), initialization_events=initialization_events,
                              seconds=time.perf_counter()-initial)
                existing.atomic_json(output, result)
                return dict(status='PARTIAL_CHECKPOINTED', reason='Search/refit checkpointed at registered deadline',
                            artifact_paths=[str(output)], result=result)
            bundle_path = Path(ctx['cache']) / receipt['artifact_cache_relative_path']
            bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
            private.append(_binding(bundle_path, ctx))
            train_origins = modules['data'].origins(bounds['A_fit'], task['L'], task['H'], 1)
            afit_eval_origins = modules['data'].origins(bounds['A_fit'], task['L'], task['H'], task['H'])
            checkpoints = [(path, torch.load(path, map_location='cpu', weights_only=False))
                           for path in sorted((bundle_path.parent / 'search').glob('checkpoint_*.pt'))]
            selected = previous.select_common_endpoints(checkpoints, len(train_origins), specification['effective_batch'])
            for checkpoint_path, checkpoint in selected:
                _check(ctx)
                tick = time.perf_counter()
                model, norm = create(specification, task, bounds['A_fit'][1], foundation)
                preparation += time.perf_counter()-tick
                _validate_loaded_initialization(checkpoint, norm)
                model.load_state_dict(checkpoint['model'], strict=True)
                epoch = checkpoint['extra']['completed_epochs'][-1]
                private.append(_binding(checkpoint_path, ctx))
                for split in ('A_fit', 'A_stop'):
                    origins = afit_eval_origins if split == 'A_fit' else modules['data'].origins(bounds[split], task['L'], task['H'], task['H'])
                    archive = Path(ctx['cache']) / 'predictions' / job['job_id'] / f'search_e{epoch:g}_{split}.npz'
                    scored = _predict_saved(model, norm, task, origins, foundation, modules, archive, ctx)
                    private.append(scored['prediction_binding'])
                    endpoints.append(dict(scored, run_id=job['job_id'], method=job['family']+'_'+job.get('init_kind','PCA'),
                                          seed=job['seed'], dataset=job['cohort'], epoch=epoch, stage='search',
                                          C=task['physical'].shape[1], K=job['K'], L=task['L'], H=task['H'],
                                          device=specification['device'], source_sha=checkpoint['contract']['source_sha'],
                                          config_sha=checkpoint['contract']['config_sha'], data_sha=checkpoint['contract']['data_sha'],
                                          model_sha=checkpoint['contract']['model_sha'],
                                          evaluation=split, fit='A_fit', updates=checkpoint['updates'],
                                          exposures=checkpoint['sampler']['exposures'],
                                          K_exposures={str(job['K']):checkpoint['sampler']['exposures']},
                                          sampler_sha256=modules['training'].fingerprint(checkpoint['sampler']),
                                          checkpoint_binding=_binding(checkpoint_path, ctx),
                                          selected_astop_mse=checkpoint['extra']['curves'][-1]['astop_mse']))
                del model
            # BASE_FIT_RECEIPT and selected_bundle already exist before any V read.
            selection_seal = existing.sha(Path(ctx['report'])/'runs'/job['job_id']/'BASE_FIT_RECEIPT.json')
            tick = time.perf_counter()
            model, norm = create(specification, task, bounds['A_stop'][1], foundation)
            preparation += time.perf_counter()-tick
            _validate_loaded_initialization(bundle, norm)
            model.load_state_dict(bundle['model'], strict=True)
            origins = modules['data'].origins(bounds['V_select'], task['L'], task['H'], task['H'])
            archive = Path(ctx['cache'])/'predictions'/job['job_id']/'selected_A60_V_select.npz'
            scored = _predict_saved(model, norm, task, origins, foundation, modules, archive, ctx)
            private.append(scored['prediction_binding'])
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rows.append(dict(scored, run_id=job['job_id'], method=job['family']+'_'+job.get('init_kind','PCA'),
                             seed=job['seed'], dataset=job['cohort'], C=task['physical'].shape[1],
                             K=job['K'], L=task['L'], H=task['H'], device=specification['device'],
                             stage='selected_refit', evaluation='V_select', fit='fresh_A60',
                             selected_epoch=receipt['selected_epoch'], updates=receipt['refit_updates'],
                             exposures=receipt['refit_exposures'], K_exposures={str(job['K']):receipt['refit_exposures']},
                             target_trained_params=trainable, P_params=sum(p.numel() for p in model.front.parameters()),
                             E_params=model.E.numel(), D_params=model.D.numel(),
                             frozen_F_params=sum(p.numel() for p in foundation.parameters()) if isinstance(foundation, torch.nn.Module) else None,
                             selection_split='A_stop', V_exposure='previously_reused_development',
                             search_seconds=bundle['search_summary']['seconds'], refit_seconds=bundle['refit_summary']['seconds']))
            result = dict(status='COMPLETE', run_id=job['job_id'], rows=rows, common_endpoint_rows=endpoints,
                          private_artifacts=private, bundle_binding=_binding(bundle_path, ctx),
                          recipe=dict(RECIPE, optimizer='AdamW', weight_decay=0., gradient_clip=1., dtype='float32',
                                      sampler_initialization_rng_separate=True), fit_receipt=receipt,
                          source_files=scientific_sources(ctx), source_sha=bundle['contract']['source_sha'],
                          config_sha=bundle['contract']['config_sha'], data_sha=bundle['contract']['data_sha'],
                          model_sha=bundle['contract']['model_sha'], selection_sealed_before_V=True,
                          selection_receipt_sha256=selection_seal, fit_wall_seconds=fit_seconds,
                          reload_preparation_seconds=preparation, total_seconds=time.perf_counter()-initial,
                          initialization_events=initialization_events,
                          backend_flags=dict(AMP=False, TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                             TF32_cudnn=torch.backends.cudnn.allow_tf32,
                                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                                             cudnn_benchmark=torch.backends.cudnn.benchmark),
                          initialization_binding=norm['initialization_binding'],
                          same_endpoint_and_selected_refit_are_separate=True)
            del model
        existing.atomic_json(output, result)
        return dict(status='COMPLETE', reason=None, artifact_paths=[str(output)], result=result)
    finally:
        if callable(getattr(foundation, 'assert_frozen_unchanged', None)):
            foundation.assert_frozen_unchanged()


def _updates(origins, effective_batch=16):
    targets = [math.ceil(epoch*origins) for epoch in (0., .25, 1., 3.)]
    return sum(math.ceil((target-previous)/effective_batch) for previous,target in zip(targets,targets[1:]))


def profile_pair(jobs, ctx, warmup_updates=1, measured_updates=5):
    """Disposable actual-dimension pair probe; no implicit admission or training."""
    if len(jobs) != 2 or any((j['cohort'], j['seed'], j['K'], j.get('device','cuda')) !=
                            (jobs[0]['cohort'], jobs[0]['seed'], jobs[0]['K'], jobs[0].get('device','cuda')) for j in jobs):
        raise ValueError('Complete paired profile requires same cohort/seed/K/device')
    modules, task, foundation = _resources(jobs[0], ctx)
    device = jobs[0].get('device', 'cuda')
    cuda = str(device).startswith('cuda')
    bounds = _registered_splits(task)
    origins = modules['data'].origins(bounds['A_fit'], task['L'], task['H'], 1)
    afit_eval_origins = modules['data'].origins(bounds['A_fit'], task['L'], task['H'], task['H'])
    refit_origins = modules['data'].origins((0,bounds['A_stop'][1]), task['L'], task['H'], 1)
    astop = modules['data'].origins(bounds['A_stop'], task['L'], task['H'], task['H'])
    val = modules['data'].origins(bounds['V_select'], task['L'], task['H'], task['H'])
    timings, estimated = [], 0.
    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if cuda else []):
            with model_factory(dict(ctx, modules=modules)) as create:
                for job in jobs:
                    _check(ctx)
                    specification = _spec(job,ctx)
                    start = time.perf_counter()
                    model, norm = create(specification, task, bounds['A_fit'][1], foundation)
                    initialization_seconds = time.perf_counter()-start
                    windows = modules['training'].FastWindows(task, norm['mean'],norm['std'],device)
                    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
                    sampler = modules['training'].PermutationStream(origins,job['seed'])
                    counts = windows.target_counts(origins)
                    update_times = []
                    for index in range(warmup_updates+measured_updates):
                        _check(ctx)
                        synchronize()
                        start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        batch = sampler.draw(16)
                        for lo in range(0,16,4):
                            x,y,mask = windows.get(batch[lo:lo+4])
                            loss = modules['training'].macro_loss(model(x,foundation),y,mask,counts,len(origins),16)
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                        synchronize()
                        if index >= warmup_updates:
                            update_times.append(time.perf_counter()-start)
                    model.eval()
                    with torch.no_grad():
                        x,_,_ = windows.get(sampler.draw(8))
                        model(x,foundation)
                        synchronize()
                        start=time.perf_counter()
                        model(x,foundation)
                        synchronize()
                        prediction8_seconds=time.perf_counter()-start
                    train_updates = _updates(len(origins))+math.ceil(3*len(refit_origins)/16)
                    # Search A_stop selection plus saved A_fit/A_stop endpoints and selected V.
                    prediction_batches = 4*math.ceil(len(astop)/4)+4*(math.ceil(len(afit_eval_origins)/8)+math.ceil(len(astop)/8))+math.ceil(len(val)/8)
                    estimate = 1.5*(train_updates*max(update_times)+prediction_batches*prediction8_seconds+6*initialization_seconds)+30
                    estimated += estimate
                    timings.append(dict(family=job['family'],init_kind=job.get('init_kind','PCA'),
                                        seconds=update_times,max_update_seconds=max(update_times),
                                        prediction8_seconds=prediction8_seconds,initialization_seconds=initialization_seconds,
                                        estimated_updates=train_updates,estimated_prediction_batches=prediction_batches,
                                        estimated_complete_arm_seconds=estimate))
                    del optimizer,model,windows
        return dict(arms=[j['family'] for j in jobs],dimensions=dict(C=task['physical'].shape[1],L=task['L'],H=task['H'],K=jobs[0]['K']),
                    measurements=timings,estimated_complete_pair_seconds=estimated,
                    evaluation_stride=task['H'],
                    evaluation_origin_counts=dict(A_fit=len(afit_eval_origins),A_stop=len(astop),V_select=len(val)),
                    includes=['search_CP0_quarter_1_3','fresh_A60_3epoch_worst_case','all_endpoint_predictions',
                              'selected_A60_V','verification_margin'],RNG_restored=True,disposable_models=True,
                    warmup_updates=warmup_updates,measured_updates=measured_updates,
                    estimate_is_admission_probe_not_cost_benchmark=True)
    finally:
        if callable(getattr(foundation,'assert_frozen_unchanged',None)):
            foundation.assert_frozen_unchanged()
