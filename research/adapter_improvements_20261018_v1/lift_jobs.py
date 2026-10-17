"""Conditional paired frozen-B2 output calibration; no foundation construction.

The official refiner is unchanged. Masked macro loss, L512/B2, early endpoint
selection and RMS-penalized output gain are registered campaign adaptations.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import scipy.linalg
import torch

from . import decoder_jobs, lift
from .contracts import atomic_json, binding, digest, read, resolve_binding, sha
from .decoder import horizon_basis


RECIPE = dict(dataset='jena', seed=92601, L=512, H=48, C=21, K=6,
              n_leaders=4, states=20, temperature=1., optimizer='Adam', lr=.001,
              batch_size=128, max_epochs=100, patience=5, schedule='type3',
              loss='masked_macro_MSE_global_Nc', dtype='float32', AMP=False,
              gradient_clip=None, selection='CP0_then_integer_epochs_strict_less',
              base='original_B2_B03_selected_epoch1_frozen_PE_original_D',
              gain_lambdas=[.0001, .001, .01, .1, 1.], gain_gammas=[1., 10., 100.],
              gain_penalty='lambda/C*sum(theta0^2+gamma*theta1^2);theta=fit_RMS*[alpha,beta]',
              complex_parameter_dtype='complex64', metric_accumulation_dtype='float64',
              limit_seconds=3600, verification_fraction=.2,
              method_status='official_pinned_module_Weather_recipe_adaptation_not_paper_reproduction')


def output_gain_grid():
    return [dict(lambda_value=lam, gamma=gamma) for lam in RECIPE['gain_lambdas'] for gamma in RECIPE['gain_gammas']]


def fit_output_gain(base, anchor, target, mask, lambda_value, gamma):
    base, anchor, target = (np.asarray(a, np.float64) for a in (base, anchor, target))
    mask = np.asarray(mask, bool)
    if base.ndim != 3 or target.shape != base.shape or mask.shape != base.shape or anchor.shape != (len(base), 1, base.shape[2]):
        raise ValueError('Output gain requires matching [N,H,C] arrays and [N,1,C] anchors')
    if not np.isfinite(base).all() or not np.isfinite(target[mask]).all() or not np.isfinite(anchor).all():
        raise ValueError('Output gain arrays contain nonfinite observations')
    counts = mask.sum((0, 1))
    if np.any(counts == 0):
        raise ValueError('BLOCKED_DATA_CONTRACT: output gain channel has N_c=0')
    if lambda_value <= 0 or gamma <= 0:
        raise ValueError('Output gain requires positive registered penalties')
    delta, response = base-anchor, target-anchor
    rms = np.sqrt(np.mean(delta*delta, axis=(0, 1)))
    zero = rms == 0
    rms[zero] = 1.
    theta = np.zeros((base.shape[2], 2))
    g = horizon_basis(base.shape[1])
    residual = 0.
    ranks = []
    for channel in range(base.shape[2]):
        if zero[channel]:
            ranks.append(0)
            continue
        z = delta[..., channel]/rms[channel]
        design = np.stack((z, z*g), -1)[mask[..., channel]]
        response_c = response[..., channel][mask[..., channel]]
        penalty = lambda_value*counts[channel]*np.array([1., gamma])
        augmented = np.vstack((design, np.diag(np.sqrt(penalty))))
        y = np.r_[response_c, 0., 0.]
        fitted, _, rank, _ = scipy.linalg.lstsq(augmented, y, lapack_driver='gelsd')
        theta[channel] = fitted
        normal = design.T@design+np.diag(penalty)
        rhs = design.T@response_c
        residual = max(residual, float(np.linalg.norm(normal@fitted-rhs)/max(1., np.linalg.norm(rhs))))
        ranks.append(int(rank))
    state = dict(schema='B2_OUTPUT_GAIN_v1', H=base.shape[1], C=base.shape[2],
                 lambda_value=float(lambda_value), gamma=float(gamma), rms=rms.tolist(), theta=theta.tolist(),
                 deployed_alpha_beta=(theta/rms[:, None]).tolist(), zero_rms_channels=np.flatnonzero(zero).tolist(),
                 penalty_coordinates='RMS-scaled theta = fit_RMS * deployed [alpha,beta]',
                 zero_rms_convention='prediction innovation zero; deployed alpha=1,beta=0 is equivalent on zero features')
    for channel in np.flatnonzero(zero):
        state['deployed_alpha_beta'][channel] = [1., 0.]
    prediction = predict_output_gain(base, anchor, state)
    sse = np.where(mask, (prediction-target)**2, 0.).sum((0, 1))
    data = float(np.mean(sse/counts))
    penalty = float(lambda_value*np.mean(theta[:, 0]**2+gamma*theta[:, 1]**2))
    state['diagnostics'] = dict(data_term=data, penalty=penalty, objective=data+penalty,
                                counts=counts.tolist(), horizon_counts=mask.sum(0).tolist(),
                                normal_equation_relative_residual=residual, augmented_ranks=ranks,
                                normalization='global observed N_c; per-horizon N_hc never substitutes it')
    return state


def predict_output_gain(base, anchor, state):
    delta = np.asarray(base, np.float64)-np.asarray(anchor, np.float64)
    if delta.shape[1:] != (state['H'], state['C']):
        raise ValueError('Output gain prediction axes differ from fitted state')
    theta = np.asarray(state['theta'])
    factor = theta[:, 0][None, None]+horizon_basis(state['H'])[None, :, None]*theta[:, 1][None, None]
    return np.asarray(anchor, np.float64)+delta/np.asarray(state['rms'])*factor


def cached_base_prediction(phi, anchor, original_d):
    if phi.dtype != np.float32 or anchor.dtype != np.float32 or original_d.dtype != torch.float32:
        raise ValueError('Original cached base path requires FP32 phi/anchor/D')
    d = original_d.detach().cpu().reshape(phi.shape[-1], anchor.shape[-1])
    with torch.no_grad():
        return (torch.from_numpy(anchor)+torch.from_numpy(phi)@d).numpy()


class CachedBaseSplit:
    def __init__(self, base, target, mask, anchor, origins, context_getter, name):
        self.base = np.asarray(base, np.float32)
        self.target = np.asarray(target, np.float32)
        self.mask = np.asarray(mask, bool)
        self.anchor = np.asarray(anchor, np.float32)
        self.origins = np.asarray(origins, np.int64)
        self.context_getter = context_getter
        self.name = name
        if (self.base.shape != self.target.shape or self.mask.shape != self.base.shape
                or len(self.origins) != len(self.base) or self.anchor.shape != (len(self.base), 1, self.base.shape[2])):
            raise ValueError('Cached base split axes differ')
        self.counts = self.mask.sum((0, 1))
        if np.any(self.counts == 0):
            raise ValueError('BLOCKED_DATA_CONTRACT: cached base channel has N_c=0')
        if not np.isfinite(self.base).all() or not np.isfinite(self.target[self.mask]).all():
            raise ValueError('Cached base has nonfinite predictions or observed targets')

    def __len__(self):
        return len(self.origins)

    def batch(self, indices, device):
        indices = np.asarray(indices, np.int64)
        x = self.context_getter(indices)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim != 3 or x.shape[0] != len(indices) or x.shape[-1] != self.base.shape[-1]:
            raise ValueError('LIFT context must retain original C channels')
        return (x, torch.as_tensor(self.base[indices], device=device),
                torch.as_tensor(self.target[indices], device=device), torch.as_tensor(self.mask[indices], device=device))

    def fingerprint(self):
        return {name: decoder_jobs.array_sha(getattr(self, name)) for name in ('base', 'target', 'mask', 'anchor', 'origins')}


def masked_macro_loss(prediction, target, mask, counts, n_origins, batch_origins):
    counts = torch.as_tensor(counts, dtype=torch.float32, device=prediction.device)
    if torch.any(counts <= 0):
        raise ValueError('Masked macro loss channel has N_c=0')
    sse = torch.where(mask, (prediction-target).square(), 0.).sum((0, 1))
    return ((sse/counts)*n_origins/batch_origins).mean()


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def predict_refiner(model, split, device, batch_size=128, check=lambda: None):
    model.eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, len(split), batch_size):
            check()
            x, base, _, _ = split.batch(np.arange(lo, min(lo+batch_size, len(split))), device)
            prediction = model(x, base)
            if prediction.shape != base.shape or not torch.isfinite(prediction).all():
                raise FloatingPointError('Invalid official refiner output')
            chunks.append(prediction.cpu().numpy())
    return np.concatenate(chunks)


def _save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    torch.save(value, temporary)
    temporary.replace(path)


def train_refiner(factory, fit, stop, checkpoint_path, identity, *, device, seed,
                  max_epochs=100, patience=5, batch_size=128, check=lambda: None):
    """Full integer epochs; checkpoint sampler, optimizer, RNG and pending evaluation."""
    checkpoint_path = Path(checkpoint_path)
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    contract = dict(identity=identity, fit=fit.fingerprint(), stop=None if stop is None else stop.fingerprint(),
                    seed=seed, max_epochs=max_epochs, patience=patience, batch_size=batch_size, device=str(device))
    state = dict(epoch=0, position=0, order=None, updates=0, exposures=0, pending_evaluation=stop is not None,
                 curves=[], best_mse=None, selected_epoch=0, stale=0, finished=False)
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Refiner resume source/data/scope/recipe identity mismatch')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        state = saved['progress']

    def save(status):
        value = dict(contract=contract, progress=copy.deepcopy(state),
                     model_state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                     optimizer=optimizer.state_dict(), sampler_rng=copy.deepcopy(rng.bit_generator.state),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [], status=status)
        _save_torch(checkpoint_path, value)
        return value

    try:
        while not state['finished']:
            if state['pending_evaluation']:
                prediction = predict_refiner(model, stop, device, batch_size, check)
                metrics = decoder_jobs.score_arrays(prediction, stop.target, stop.mask)
                state['curves'].append(dict(epoch=state['epoch'], updates=state['updates'], exposures=state['exposures'],
                                             MSE=metrics['MSE'], MAE=metrics['MAE']))
                if state['best_mse'] is None or metrics['MSE'] < state['best_mse']:
                    state.update(best_mse=metrics['MSE'], selected_epoch=state['epoch'], stale=0)
                elif state['epoch'] > 0:
                    state['stale'] += 1
                state['pending_evaluation'] = False
                save('RUNNING')
            if state['epoch'] >= max_epochs or (stop is not None and state['stale'] >= patience):
                state['finished'] = True
                break
            if state['order'] is None:
                state['order'] = rng.permutation(len(fit)).tolist()
                state['position'] = 0
            model.train()
            while state['position'] < len(fit):
                check()
                indices = state['order'][state['position']:state['position']+batch_size]
                x, base, target, mask = fit.batch(indices, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, base)
                loss = masked_macro_loss(prediction, target, mask, fit.counts, len(fit), len(indices))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite masked macro refiner loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite official complex refiner gradient')
                optimizer.step()
                state['position'] += len(indices)
                state['exposures'] += len(indices)
                state['updates'] += 1
            state['epoch'] += 1
            state['order'] = None
            state['position'] = 0
            # Same end-of-epoch type3 assignment as official util.tools.
            lr = .001 if state['epoch'] < 3 else .001*.9**(state['epoch']-3)
            for group in optimizer.param_groups:
                group['lr'] = lr
            state['pending_evaluation'] = stop is not None
            save('RUNNING')
        saved = save('COMPLETE')
        return dict(status='COMPLETE', **state, model_state=saved['model_state'],
                    checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path))
    except TimeoutError as exc:
        saved = save('PARTIAL_CHECKPOINTED')
        return dict(status='PARTIAL_CHECKPOINTED', reason=str(exc), **state, model_state=saved['model_state'],
                    checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                    resumable=True, same_identity_resume=True)


def estimate_pair(profile):
    n, nr = int(profile['fit_origins']), int(profile['refit_origins'])
    updates = 100*(math.ceil(n/128)+math.ceil(nr/128))
    fit_seconds = updates*profile['update_seconds']
    search_eval = 101*profile['evaluation_seconds']
    core = fit_seconds+search_eval+profile['gain_grid_seconds']*2+profile['prediction_seconds']*3+profile['prepare_seconds']
    return dict(worst_train_updates=updates, max_search_epochs=100, max_refit_epochs=100,
                training_seconds=fit_seconds, search_evaluation_seconds=search_eval,
                estimated_seconds=core/(1.-RECIPE['verification_fraction']),
                verification_fraction=RECIPE['verification_fraction'],
                assumption='3 timed updates and full A_stop evaluation; 100 search + 100 fresh refit, no early-stop discount')


def admit_pair(estimate, budget):
    available = min(3600., float(budget.get('optional_seconds', 0.)),
                    max(0., float(budget.get('remaining_seconds', 0.))-float(budget.get('reserve_seconds', 0.))))
    if budget.get('device', 'cpu').startswith('cuda'):
        available = min(available, max(0., float(budget.get('remaining_gpu_seconds', 0.))-float(budget.get('gpu_reserve_seconds', 0.))))
    admitted = estimate['estimated_seconds'] <= available
    return dict(admitted=admitted, available_seconds=available, estimate_seconds=estimate['estimated_seconds'],
                paired=True, schedule_shrunk=False, reason=None if admitted else 'Full LIFT100+100 worst-case and paired gain exceed optional60min/remaining/reserves')


def run_pair(fit, stop, refit, load_validation, seal_selection, factory, output, identity, *,
             device, seed, max_epochs=100, patience=5, batch_size=128, check=lambda: None):
    output = Path(output)
    started = time.perf_counter()
    search = train_refiner(factory, fit, stop, output/'search_state.pt', dict(identity, stage='A_fit_search'),
                           device=device, seed=seed, max_epochs=max_epochs, patience=patience, batch_size=batch_size, check=check)
    search_seconds = time.perf_counter()-started
    if search['status'] != 'COMPLETE':
        return dict(status=search['status'], stage='A_fit_search', training=search, new_F_calls=0)
    gain_trials = []
    before = time.perf_counter()
    for setting in output_gain_grid():
        check()
        state = fit_output_gain(fit.base, fit.anchor, fit.target, fit.mask, **setting)
        score = decoder_jobs.score_arrays(predict_output_gain(stop.base, stop.anchor, state), stop.target, stop.mask)
        gain_trials.append(dict(setting, MSE=score['MSE'], MAE=score['MAE'],
                                fit_objective=state['diagnostics']['objective']))
    gain_selected = min(gain_trials, key=lambda trial: (trial['MSE'], trial['lambda_value'], trial['gamma']))
    gain_search_seconds = time.perf_counter()-before
    selection = dict(identity=identity, selected_epoch=search['selected_epoch'], search_curves=search['curves'],
                     selected_search_updates=next(row['updates'] for row in search['curves'] if row['epoch'] == search['selected_epoch']),
                     selected_search_origin_exposures=next(row['exposures'] for row in search['curves'] if row['epoch'] == search['selected_epoch']),
                     gain_trials=gain_trials, gain_selected=gain_selected, V_loaded=False,
                     base_scope='A_fit only for endpoint/configuration selection',
                     selection_split='A_stop', exact_ties='earliest epoch; then ascending lambda,gamma')
    seal_selection(selection)
    before = time.perf_counter()
    fresh = train_refiner(factory, refit, None, output/'refit_state.pt', dict(identity, stage='fresh_A60_refit'),
                          device=device, seed=seed, max_epochs=search['selected_epoch'], patience=patience,
                          batch_size=batch_size, check=check)
    if fresh['status'] != 'COMPLETE':
        return dict(status=fresh['status'], stage='A60_refit', training=fresh, selection=selection, new_F_calls=0)
    gain_state = fit_output_gain(refit.base, refit.anchor, refit.target, refit.mask,
                                 gain_selected['lambda_value'], gain_selected['gamma'])
    refit_seconds = time.perf_counter()-before
    # No V feature array is decompressed before both selections and fresh refits.
    check()
    validation = load_validation()
    model = factory().to(device)
    model.load_state_dict(fresh['model_state'], strict=True)
    predictions = dict(B2_LIFT=predict_refiner(model, validation, device, batch_size, check),
                       B2_OUTPUT_GAIN=predict_output_gain(validation.base, validation.anchor, gain_state))
    rows = []
    for method, prediction in predictions.items():
        rows.append(dict(method=method, **decoder_jobs.score_arrays(prediction, validation.target, validation.mask),
                         selected_epoch=search['selected_epoch'] if method == 'B2_LIFT' else None,
                         refiner_updates=fresh['updates'] if method == 'B2_LIFT' else 0,
                         refiner_origin_exposures=fresh['exposures'] if method == 'B2_LIFT' else 0,
                         prediction_dtype=str(prediction.dtype), core_dtype='float32' if method == 'B2_LIFT' else 'float64',
                         parameter_inventory=lift.parameter_inventory(model) if method == 'B2_LIFT' else
                             dict(parameter_elements=2*validation.base.shape[2], complex_elements=0,
                                  real_degrees_of_freedom=2*validation.base.shape[2], bytes=16*validation.base.shape[2]),
                         lambda_value=None if method == 'B2_LIFT' else gain_selected['lambda_value'],
                         gamma=None if method == 'B2_LIFT' else gain_selected['gamma']))
    return dict(status='COMPLETE', rows=rows, predictions=predictions, validation=validation,
                gain_state=gain_state, selection=selection, search=search, refit=fresh,
                new_F_calls=0, new_F_rows=0, selection_sealed_before_V=True,
                timers=dict(search_seconds=search_seconds, gain_search_seconds=gain_search_seconds,
                            refit_seconds=refit_seconds, total_seconds=time.perf_counter()-started))


def _prepare_inputs(job, ctx, check):
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _run_root
    old, evidence, inventory, modules, task = decoder_jobs._input_contract(job, ctx)
    run_root = _run_root(ctx, int(job['seed']), 'B2')
    paths = dict(A_fit=run_root/'search/checkpoint_000002.pt', A60=run_root/'selected_bundle.pt')
    states = {scope: torch.load(path, map_location='cpu', weights_only=False) for scope, path in paths.items()}
    if float(old['original_selected_epoch']) != 1.:
        raise ValueError('Conditional LIFT base must be the sealed original epoch1 B2')
    decoders = {scope: state['model']['D'] for scope, state in states.items()}
    if any(d.dtype != torch.float32 or tuple(d.shape) != (7, 6, 21) for d in decoders.values()):
        raise ValueError('Original B2 D has unexpected dtype/lag-major axes')
    windows = {scope: modules['training'].FastWindows(task, inventory[scope, 'fit']['norm']['mean'],
                                                     inventory[scope, 'fit']['norm']['std'], 'cpu')
               for scope in ('A_fit', 'A60')}

    def load(scope, split):
        check()
        entry = inventory[scope, split]
        values = decoder_jobs._load_bound(entry, modules, task, check)
        base = cached_base_prediction(values['phi'], values['anchor'], decoders[scope])
        origins = entry['origins']
        def contexts(indices):
            return windows[scope].get(origins[indices])[0]
        return CachedBaseSplit(base, values['target'], values['mask'], values['anchor'], origins, contexts, scope+'/'+split)

    fit, stop, refit = load('A_fit', 'fit'), load('A_fit', 'A_stop'), load('A60', 'fit')
    evidence = dict(original_receipt=evidence, original_selected_epoch=old['original_selected_epoch'],
                    checkpoints={scope: binding(path, ctx) for scope, path in paths.items()},
                    feature_caches=[dict(scope=scope, split=split, artifact=binding(entry['path'], ctx), key=entry['key'])
                                    for (scope, split), entry in inventory.items()],
                    base_prediction_dtype='float32', base_decoder='original learned D, not B03 fitted head',
                    frozen_F_sha256=modules['legacy'].F_SHA,
                    F_integrity='No F instantiated; pinned/hash-verified original feature producers',
                    context_path='full C-channel FastWindows under each original scope scaler',
                    original_cache_extraction_seconds=sum(entry['manifest']['seconds'] for entry in inventory.values()))
    return fit, stop, refit, lambda: load('A60', 'V_select'), evidence, task, modules


def scientific_sources(ctx):
    from . import initialization, decoder
    paths = [Path(__file__), Path(lift.__file__), Path(initialization.__file__),
             Path(decoder.__file__), Path(decoder_jobs.__file__), initialization.MANIFEST]
    root = Path(ctx['repo']).resolve()
    return {path.resolve().relative_to(root).as_posix(): sha(path) for path in paths}


class _GainRoute(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.horizon, self.channels = state['H'], state['C']
        self.register_buffer('rms', torch.tensor(state['rms'], dtype=torch.float64))
        self.register_buffer('theta', torch.tensor(state['theta'], dtype=torch.float64))
        self.register_buffer('basis', torch.tensor(horizon_basis(self.horizon), dtype=torch.float64))

    def forward(self, context, base_prediction):
        if base_prediction.shape[1:] != (self.horizon, self.channels) or context.shape[-1] != self.channels:
            raise ValueError('Output gain route requires original C-channel context/base prediction')
        anchor = context[:, -1:].to(torch.float64)
        factor = self.theta[:, 0][None, None]+self.basis[None, :, None]*self.theta[:, 1][None, None]
        return anchor+(base_prediction.to(torch.float64)-anchor)/self.rms*factor


def load_calibrator(method, artifact, ctx, device='cpu'):
    """Exact saved post-decoder route, before inverse scaling; no F or fitting."""
    path = resolve_binding(artifact, ctx)
    if method == 'B2_OUTPUT_GAIN':
        state = read(path)
        if state.get('scientific_source_hashes') != scientific_sources(ctx):
            raise ValueError('Saved output gain deployment source identity changed')
        return _GainRoute(state).to(device).eval()
    if method != 'B2_LIFT':
        raise ValueError('Unregistered output calibrator method')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle['source_hashes'] != scientific_sources(ctx) or bundle['recipe'] != RECIPE:
        raise ValueError('Saved LIFT deployment source/recipe identity changed')
    model = lift.make_official_refiner(ctx.get('official_source_root', Path(ctx['cache'])/'official'),
                                       length=512, horizon=48, channels=21, n_leaders=4,
                                       states=20, temperature=1., seed=92601)
    model.load_state_dict(bundle['model_state'], strict=True)
    return model.to(device).eval()


def _profile(factory, fit, stop, refit, device, check):
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    indices = np.arange(min(128, len(fit)))
    timings = []
    for index in range(4):
        check()
        _sync(device)
        tick = time.perf_counter()
        x, base, target, mask = fit.batch(indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_macro_loss(model(x, base), target, mask, fit.counts, len(fit), len(indices))
        loss.backward()
        optimizer.step()
        _sync(device)
        if index:
            timings.append(time.perf_counter()-tick)
    tick = time.perf_counter()
    predict_refiner(model, stop, device, 128, check)
    _sync(device)
    evaluation = time.perf_counter()-tick
    tick = time.perf_counter()
    for setting in output_gain_grid():
        check()
        fit_output_gain(fit.base, fit.anchor, fit.target, fit.mask, **setting)
    gain_seconds = time.perf_counter()-tick
    return dict(update_seconds=float(np.mean(timings)), timed_update_seconds=timings,
                warmup_updates=1, timed_updates=3, evaluation_seconds=evaluation,
                gain_grid_seconds=gain_seconds, prediction_seconds=evaluation,
                prepare_seconds=0., fit_origins=len(fit), refit_origins=len(refit),
                batch_size=128, full_A_stop_origins=len(stop), parameter_inventory=lift.parameter_inventory(model),
                logical_new_F_calls=0, GPU_active_profile=str(device).startswith('cuda'))


def execute_job(job, ctx):
    """Profile/admit the complete optional pair, then search/fresh-refit/verify."""
    if int(job.get('seed', -1)) != 92601 or job.get('cohort', 'jena') != 'jena':
        raise ValueError('Conditional LIFT is registered for Jena seed92601 only')
    for name, expected in RECIPE.items():
        if name in job.get('recipe', {}) and job['recipe'][name] != expected:
            raise ValueError('Conditional LIFT recipe changed: '+name)
    sources = scientific_sources(ctx)
    for name, expected in job.get('scientific_source_hashes', {}).items():
        if name in sources and sources[name] != expected:
            raise ValueError('Registered LIFT scientific source hash mismatch: '+name)
    started, cpu_started = time.time(), time.process_time()
    device = job.get('device', 'cpu')
    campaign_reserve = float(ctx.get('lift_budget', {}).get('reserve_seconds', 0.))
    deadline = min(started+3600, ctx.get('deadline', started+3600)-campaign_reserve,
                   ctx.get('job_deadline', started+3600))
    fitting_deadline = deadline-RECIPE['verification_fraction']*max(0., deadline-started)
    phase = dict(verification=False)
    output = Path(ctx['cache'])/'lift'/job['job_id']
    public = Path(ctx['report'])/'jobs'/(job['job_id']+'.json')
    output.mkdir(parents=True, exist_ok=True)

    def check():
        if time.time() >= (deadline if phase['verification'] else fitting_deadline):
            raise TimeoutError('Conditional LIFT paired60min/registered deadline')
        if ctx.get('lease'):
            ctx['lease'].check()

    def finish(result):
        result.update(recipe=RECIPE, scientific_source_hashes=sources,
                      wall_seconds=time.time()-started, process_cpu_seconds=time.process_time()-cpu_started,
                      new_F_calls=0, new_F_rows=0, new_model_loads=0, cost_valid=False,
                      cost_status='UNMEASURED: actual full-path refiner route/reload remains required')
        atomic_json(public, result)
        return dict(status=result['status'], reason=result.get('reason'), rows=result.get('rows', []),
                    artifact_paths=[str(public)], private_artifacts=result.get('private_artifacts', []), result=result)

    if not ctx.get('lift_dependencies_complete', False) and not job.get('profile_only', False):
        return finish(dict(status='NOT_RUN_DEPENDENCY', reason='Mandatory Jena/PeMS matched families and cost/publication reserves not secured'))
    prepare_started = time.perf_counter()
    fit, stop, refit, load_v, evidence, task, modules = _prepare_inputs(job, ctx, check)
    source_root = ctx.get('official_source_root', Path(ctx['cache'])/'official')
    def factory():
        return lift.make_official_refiner(source_root, length=task['L'], horizon=task['H'],
                                          channels=task['physical'].shape[1], n_leaders=4,
                                          states=20, temperature=1., seed=92601)
    profile_path = output/'PROFILE.json'
    identity = dict(scientific_identity=job.get('scientific_identity'), sources=sources,
                    base_evidence_sha256=digest(evidence), recipe=RECIPE)
    if profile_path.exists():
        saved = read(profile_path)
        if saved['identity'] != identity:
            raise ValueError('Existing LIFT profile scientific/base identity changed')
        profile, estimate = saved['profile'], saved['estimate']
    else:
        with modules['blas_controller']:
            profile = _profile(factory, fit, stop, refit, device, check)
        profile['prepare_seconds'] = time.perf_counter()-prepare_started
        estimate = estimate_pair(profile)
        atomic_json(profile_path, dict(identity=identity, profile=profile, estimate=estimate, V_loaded=False))
    budget = dict(ctx.get('lift_budget', {}), device=device)
    budget['optional_seconds'] = min(float(budget.get('optional_seconds', 0.)), max(0., deadline-time.time()))
    budget['remaining_seconds'] = min(float(budget.get('remaining_seconds', 0.)), max(0., ctx.get('deadline', deadline)-time.time()))
    admission = admit_pair(estimate, budget)
    if job.get('profile_only', False) or not admission['admitted']:
        return finish(dict(status='SKIPPED_BUDGET' if not admission['admitted'] else 'NOT_RUN_DEPENDENCY',
                           reason=admission['reason'] or 'Explicit profile-only registered operation',
                           profile=profile, estimate=estimate, admission=admission, rows=[],
                           base_evidence=evidence, private_artifacts=[binding(profile_path, ctx)]))
    selection_path = output/'SELECTION_SEAL.json'
    def seal(selection):
        value = dict(selection, selection_sha256=digest(selection))
        if selection_path.exists() and read(selection_path) != value:
            raise ValueError('Existing immutable LIFT/gain selection seal changed')
        if not selection_path.exists():
            atomic_json(selection_path, value)
    def validation_after_refit():
        phase['verification'] = True
        return load_v()
    with modules['blas_controller']:
        result = run_pair(fit, stop, refit, validation_after_refit, seal, factory, output, identity,
                           device=device, seed=92601, check=check)
    private = [binding(profile_path, ctx)]
    for name in ('search_state.pt', 'refit_state.pt', 'SELECTION_SEAL.json'):
        if (output/name).exists():
            private.append(binding(output/name, ctx))
    if result['status'] != 'COMPLETE':
        training = result.pop('training')
        training.pop('model_state', None)
        order = training.pop('order', None)
        training['sampler_order_sha256'] = None if order is None else decoder_jobs.array_sha(np.asarray(order, np.int64))
        training['checkpoint_path'] = 'search_state.pt' if result['stage'] == 'A_fit_search' else 'refit_state.pt'
        result.update(training=training, base_evidence=evidence, profile=profile, estimate=estimate,
                      admission=admission, private_artifacts=private)
        return finish(result)
    validation = result.pop('validation')
    predictions = result.pop('predictions')
    deployed_path = output/'refiner_bundle.pt'
    deployed = dict(model_state=result['refit']['model_state'], source_hashes=sources, recipe=RECIPE,
                    base_evidence=evidence, selection_sha256=sha(selection_path))
    if deployed_path.exists():
        previous = torch.load(deployed_path, map_location='cpu', weights_only=False)
        if (previous['source_hashes'] != sources or previous['recipe'] != RECIPE
                or previous['selection_sha256'] != deployed['selection_sha256']
                or set(previous['model_state']) != set(deployed['model_state'])
                or any(not torch.equal(previous['model_state'][key], value) for key, value in deployed['model_state'].items())):
            raise ValueError('Existing immutable LIFT deployment bundle changed')
    else:
        _save_torch(deployed_path, deployed)
    private.append(binding(deployed_path, ctx))
    for name in ('search', 'refit'):
        result[name].pop('model_state', None)
        result[name]['checkpoint_path'] = name+'_state.pt'
    gain_path = output/'output_gain_state.json'
    gain_state = result.pop('gain_state')
    atomic_json(gain_path, dict(gain_state, scientific_source_hashes=sources))
    result['gain_fitting_diagnostics'] = gain_state['diagnostics']
    private.append(binding(gain_path, ctx))
    for row in result['rows']:
        check()
        prediction_path = output/(row['method']+'_predictions.npz')
        from research.adapter_diagnostics_20261017_v2.new_jobs import save_npz
        values = dict(pred_normalized=predictions[row['method']], target_normalized=validation.target,
                      observed=validation.mask, origins=validation.origins, channel_ids=np.asarray(task['columns']),
                      base_pred_normalized=validation.base, anchor=validation.anchor)
        if prediction_path.exists():
            with np.load(prediction_path, allow_pickle=False) as saved:
                if set(saved.files) != set(values) or any(not np.array_equal(saved[key], value) for key, value in values.items()):
                    raise ValueError('Existing LIFT prediction arrays changed')
        else:
            save_npz(prediction_path, **values)
        with np.load(prediction_path, allow_pickle=False) as saved:
            replay = decoder_jobs.score_arrays(saved['pred_normalized'], saved['target_normalized'], saved['observed'])
            fingerprints = {key: decoder_jobs.array_sha(saved[key]) for key in saved.files}
        if any(replay[key] != row[key] for key in ('MSE', 'MAE')):
            raise ValueError('Saved output-calibration FP64 rescore differs')
        row.update(seed=92601, dataset='jena', C=21, K=6, L=512, H=48, stage='fresh_A60_refit',
                   evaluation='V_select', selection_split='A_stop', device=device if row['method'] == 'B2_LIFT' else 'cpu',
                   point_statistic='quantile_0.5_plus_deterministic_postcalibration', units='normalized_A60',
                   V_exposure='previously reused development evaluation', matching_status='MATCHED_FROZEN_ORIGINAL_B2_BASE',
                   prediction_binding=binding(prediction_path, ctx), array_fingerprints=fingerprints,
                   independent_FP64='PASS', saved_array_independent_FP64='PASS',
                   selection_seal=binding(selection_path, ctx), new_F_calls=0,
                   refiner_bundle=binding(deployed_path, ctx) if row['method'] == 'B2_LIFT' else binding(gain_path, ctx),
                   base_checkpoint=evidence['checkpoints']['A60'],
                   backbone_dtype='float32', head_dtype='float32' if row['method'] == 'B2_LIFT' else 'float64',
                   full_original_channel_context=True if row['method'] == 'B2_LIFT' else False,
                   training_cost_scope='post-refiner only; original frozen B2 producer costs retained separately')
        row['frozen_base_target_trained_params'] = 35717
        row['refiner_target_trained_real_degrees'] = row['parameter_inventory']['real_degrees_of_freedom']
        row['retained_gain_RMS_buffer_bytes'] = 8*21 if row['method'] == 'B2_OUTPUT_GAIN' else 0
        private.append(binding(prediction_path, ctx))
    if scientific_sources(ctx) != sources:
        raise ValueError('LIFT scientific producer changed after selection seal')
    result.update(base_evidence=evidence, profile=profile, estimate=estimate, admission=admission,
                  private_artifacts=private, matching_control='B2_OUTPUT_GAIN on identical original anchored B2 predictions',
                  preparation_seconds=profile['prepare_seconds'], serialized_refiner_bytes=deployed_path.stat().st_size,
                  resumable_refit_checkpoint_bytes=(output/'refit_state.pt').stat().st_size)
    return finish(result)
