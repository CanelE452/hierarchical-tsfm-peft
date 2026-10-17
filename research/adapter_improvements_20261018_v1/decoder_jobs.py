"""Hash-bound fixed-feature readout jobs; this module never constructs F."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import decoder
from .contracts import atomic_json, binding, digest, read, sha


ORIGINAL_TOLERANCE = 1e-8


def array_sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(list(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def budget_checker(ctx, started, verification=False):
    # The first24min are fitting; the remaining6min are protected for checks.
    family_started = ctx.get('decoder_family_started_unix', started)
    end = min(family_started + (1800 if verification else 1440), ctx.get('deadline', float('inf')),
              ctx.get('job_deadline', float('inf')))
    def check():
        if time.time() >= end:
            raise TimeoutError('D family30min limit including20% verification reserve')
        if ctx.get('lease'):
            ctx['lease'].check()
    return check


def load_cache(path, expected_key, dimensions):
    path = Path(path)
    manifest = read(path.with_suffix('.json'))
    if manifest['key'] != expected_key:
        raise ValueError('Historical feature key binding changed')
    if sha(path) != manifest['sha256']:
        raise ValueError('Historical feature bytes changed')
    n, h, p, c = dimensions
    expected = dict(phi=((n, h, p), np.dtype('float32')), target=((n, h, c), np.dtype('float32')),
                    mask=((n, h, c), np.dtype('bool')), anchor=((n, 1, c), np.dtype('float32')))
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError('Historical feature NPZ schema mismatch')
        result = {key: archive[key] for key in expected}
    for key, (shape, dtype) in expected.items():
        if result[key].shape != shape or result[key].dtype != dtype:
            raise ValueError('Historical feature shape/dtype mismatch: '+key)
        if key != 'mask' and not np.isfinite(result[key]).all():
            raise ValueError('Historical feature nonfinite array: '+key)
    if np.any(result['mask'].sum((0, 1)) == 0):
        raise ValueError('BLOCKED_DATA_CONTRACT: registered cache channel has N_c=0')
    return result


def _ratio(sse, sae, count):
    if np.any(count == 0):
        return dict(MSE=None, MAE=None, status='UNDEFINED')
    return dict(MSE=math.fsum((sse/count).tolist())/len(count),
                MAE=math.fsum((sae/count).tolist())/len(count), status='DEFINED')


def score_arrays(prediction, target, mask):
    p, y, m = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(mask, bool)
    if p.shape != y.shape or m.shape != y.shape or p.ndim != 3 or not np.isfinite(p[m]).all():
        raise ValueError('Prediction/target/mask numerical contract mismatch')
    error = np.where(m, p-y, 0.)
    squared, absolute = error*error, np.abs(error)
    hs, ha, hc = squared.sum(0), absolute.sum(0), m.sum(0, dtype=np.int64)
    cs, ca, cc = hs.sum(0), ha.sum(0), hc.sum(0)
    if np.any(cc == 0):
        raise ValueError('BLOCKED_DATA_CONTRACT: registered evaluation channel has N_c=0')
    value = _ratio(cs, ca, cc)
    bands = []
    for indices in np.array_split(np.arange(y.shape[1]), 3):
        s, a, count = hs[indices].sum(0), ha[indices].sum(0), hc[indices].sum(0)
        bands.append(dict(horizon_start=int(indices[0])+1 if len(indices) else None,
                          horizon_end=int(indices[-1])+1 if len(indices) else None,
                          channel_sse=s.tolist(), channel_sae=a.tolist(), channel_counts=count.tolist(),
                          coverage_channels=int((count > 0).sum()), **_ratio(s, a, count)))
    os, oa, oc = squared.sum(1), absolute.sum(1), m.sum(1, dtype=np.int64)
    origin_metrics = [dict(origin_index=i, channel_sse=s.tolist(), channel_sae=a.tolist(),
                           channel_counts=count.tolist(), **_ratio(s, a, count))
                      for i, (s, a, count) in enumerate(zip(os, oa, oc))]
    value.update(channel_sse=cs.tolist(), channel_sae=ca.tolist(), channel_counts=cc.tolist(),
                 horizon_sse=hs.tolist(), horizon_sae=ha.tolist(), horizon_counts=hc.tolist(),
                 bands=bands, origin_statistics=origin_metrics, observed=int(cc.sum()), n_origins=len(y),
                 evaluation_stride=y.shape[1], target_horizon_overlap=False, metric_accumulation_dtype='float64')
    return value


def _innovation_target(values, legacy=False):
    return values['target']-values['anchor'] if legacy else values['target'].astype(np.float64)-values['anchor'].astype(np.float64)


def _evaluate(values, state):
    prediction = values['anchor'].astype(np.float64) + decoder.predict_head(values['phi'], state)
    return prediction, score_arrays(prediction, values['target'], values['mask'])


def run_family(fit, stop, refit, load_validation, seal_selection, check):
    started = time.perf_counter()
    stats = decoder.prepare_statistics(fit['phi'], _innovation_target(fit), fit['mask'], check=check)
    statistics_seconds = time.perf_counter()-started
    trials = []
    for setting in decoder.registered_grid():
        check()
        before = time.perf_counter()
        state = decoder.fit_statistics(stats, setting['arm'], setting['lambda_value'], setting['gamma'])
        _, scores = _evaluate(stop, state)
        trials.append(dict(setting, MSE=scores['MSE'], MAE=scores['MAE'], fit_objective=state['diagnostics']['objective'],
                           fit_MSE=state['diagnostics']['data_term'], fit_penalty=state['diagnostics']['penalty'],
                           seconds=time.perf_counter()-before,
                           normal_equation_relative_residual=state['diagnostics']['normal_equation_relative_residual'],
                           augmented_svd_relative_difference=state['diagnostics']['augmented_svd_relative_difference']))
    selections = {arm: {key: value for key, value in decoder.select_trial([trial for trial in trials if trial['arm'] == arm]).items()
                        if key != 'seconds'} for arm in decoder.ARMS}
    selection = dict(selection_trials=trials, selections=selections, selection_split='A_stop',
                     selection_scope='A_fit PE/scaler/RMS only; exact FP64 MSE then ascending lambda,gamma',
                     lower_bound_selected={arm: row['lambda_value'] == min(decoder.LAMBDAS) for arm, row in selections.items()})
    seal_selection(selection)
    # The callback is the only place where V feature arrays can become available.
    check()
    validation = load_validation()
    before = time.perf_counter()
    refit_stats = decoder.prepare_statistics(refit['phi'], _innovation_target(refit), refit['mask'], check=check)
    refit_statistics_seconds = time.perf_counter()-before
    states, predictions, rows = {}, {}, []
    for arm, selected in selections.items():
        check()
        before = time.perf_counter()
        state = decoder.fit_statistics(refit_stats, arm, selected['lambda_value'], selected['gamma'])
        prediction, metrics = _evaluate(validation, state)
        coefficient_count = np.asarray(state['coefficients']).size + (np.asarray(state['gain']).size if state['gain'] is not None else 0)
        states[arm], predictions[arm] = state, prediction
        rows.append(dict(method=arm, arm=arm, stage='A60', evaluation='V_select',
                         lambda_value=selected['lambda_value'], gamma=selected['gamma'],
                         selection='LOWER_BOUND_SELECTED' if selected['lambda_value'] == min(decoder.LAMBDAS) else 'INTERIOR_OR_UPPER_GRID',
                         selection_split='A_stop', selection_MSE=selected['MSE'], selection_MAE=selected['MAE'],
                         fit_MSE=state['diagnostics']['data_term'], fit_penalty=state['diagnostics']['penalty'],
                         fit_objective=state['diagnostics']['objective'],
                         retained_decoder_parameters=int(coefficient_count), decoder_refit_trainable_params=int(coefficient_count),
                         current_requires_grad_parameters=0, frozen_PE_target_trained_params=34835,
                         total_target_trained_params=34835+int(coefficient_count),
                         backbone_dtype='float32', core_dtype='float64', head_dtype='float64', prediction_dtype='float64',
                         delta_subtraction_dtype='float64', measured_deployment_latency=None,
                         refit_and_evaluation_seconds=time.perf_counter()-before, **metrics))
    return dict(rows=rows, selection_trials=trials, selections=selections, states=states,
                predictions=predictions, validation=validation, new_F_calls=0,
                timers=dict(A_fit_statistics_seconds=statistics_seconds, A60_statistics_seconds=refit_statistics_seconds,
                            A_stop_search_seconds=sum(t['seconds'] for t in trials)))


def _input_contract(job, ctx):
    from .audit import verify_old_job, load_modules, load_task
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _run_root
    from research.adapter_diagnostics_20261017_v2 import new_jobs
    old, evidence = verify_old_job(f'B03_s{job["seed"]}', ctx)
    modules = load_modules(ctx)
    task = load_task(ctx, 'jena', modules)
    seed = int(job['seed'])
    run_root = _run_root(ctx, seed, 'B2')
    checkpoint_path = run_root/'search'/'checkpoint_000002.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError('BLOCKED_MISSING_A_FIT_CHECKPOINT: registered search checkpoint')
    selected_path = run_root/'selected_bundle.pt'
    torch = modules['torch']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    bundle = torch.load(selected_path, map_location='cpu', weights_only=False)
    if bundle['selected_search_epoch'] != old['original_selected_epoch'] or checkpoint['extra']['completed_epochs'][-1] != old['original_selected_epoch']:
        raise ValueError('Original selected epoch/checkpoint contract mismatch')
    if old['selected_lambdas'] != dict(shared=.01, horizon=.01):
        raise ValueError('Registered old shared/horizon selected lambda differs')
    norm50 = dict(zip(('mean', 'std', 'count'), modules['data'].fit_statistics(task, 10482)))
    norm60 = dict(zip(('mean', 'std', 'count'), modules['data'].fit_statistics(task, 12579)))
    for key in ('mean', 'std'):
        if not np.array_equal(norm60[key], bundle['normalization'][key]):
            raise ValueError('Actual A60 scaler differs from selected bundle')
    inventory = {}
    contracts = dict(A_fit=checkpoint['contract'], A60=bundle['contract'])
    for scope, split, n, boundaries in (('A_fit', 'fit', 9923, (0, 10482)),
                                       ('A_fit', 'A_stop', 43, (10482, 12579)),
                                       ('A60', 'fit', 12020, (0, 12579)),
                                       ('A60', 'V_select', 43, (14675, 16772))):
        paths = list((Path(ctx['previous_cache'])/'decoder_features').glob(f'{seed}_{scope}_{split}_*.npz'))
        if len(paths) != 1:
            raise FileNotFoundError('Expected exactly one original bound feature cache: '+scope+'/'+split)
        path = paths[0]
        manifest = read(path.with_suffix('.json'))
        key = manifest['key']
        matching = [entry for entry in old['feature_caches'] if entry['sha256'] == manifest['sha256'] and entry['key'] == key]
        if len(matching) != 1 or sha(path) != manifest['sha256'] or digest(key) not in path.stem:
            raise ValueError('Original receipt/cache manifest/content identity mismatch')
        origins = modules['data'].origins(boundaries, 512, 48, 1 if split == 'fit' else 48)
        norm = norm50 if scope == 'A_fit' else norm60
        actual = dict(source_sha256=sha(new_jobs.__file__), weights_sha=modules['legacy'].F_SHA,
                      original_scientific_source_sha=modules['fit']._source_contract()[0],
                      frozen_runtime_sha256=sha(modules['campaign']/'gpu_runtime.py'),
                      instance_norm_source_sha256=sha(modules['campaign'].parent/'temporal_frontend_campaign_20261015_v1'/'runtime.py'),
                      PE_checkpoint_sha256=sha(checkpoint_path if scope == 'A_fit' else selected_path),
                      scaler_sha=modules['fit'].scaler_fingerprint(norm['mean'], norm['std']),
                      origins_sha=modules['data'].array_sha256(origins), channel_order=task['columns'],
                      prefix_data_sha=modules['data'].array_sha256(task['physical']),
                      prefix_mask_sha=modules['data'].array_sha256(task['observed']),
                      L=512, H=48, C=21, K=6, dtype='float32', device='cuda', backend='torch_eager',
                      torch_version=torch.__version__, numpy_version=np.__version__,
                      feature='lag-major(0,1,2,4,8,16,32) joined Z,U minus Z[-1-lag]')
        for name, expected in actual.items():
            if key.get(name) != expected:
                raise ValueError('Actual feature contract differs: '+name)
        contract = contracts[scope]
        if (contract['source_sha'] != actual['original_scientific_source_sha'] or contract['model_sha'] != actual['weights_sha'] or
                contract['scaler_sha'] != actual['scaler_sha'] or contract['data_sha'] != task['data_sha256']):
            raise ValueError('PE checkpoint source/data/F/scaler contract differs')
        inventory[(scope, split)] = dict(path=path, key=key, dimensions=(n, 48, 42, 21),
                                         origins=origins, norm=norm, manifest=manifest)
    return old, evidence, inventory, modules, task


def _load_bound(entry, modules, task, check):
    values = load_cache(entry['path'], entry['key'], entry['dimensions'])
    windows = modules['training'].FastWindows(task, entry['norm']['mean'], entry['norm']['std'], 'cpu')
    for start in range(0, len(entry['origins']), 32):
        check()
        x, y, mask = windows.get(entry['origins'][start:start+32])
        end = start+len(x)
        if (not np.array_equal(values['target'][start:end], y.numpy()) or
                not np.array_equal(values['mask'][start:end], mask.numpy()) or
                not np.array_equal(values['anchor'][start:end], x[:, -1:].numpy())):
            raise ValueError('Cached target/mask/anchor differs from bound data/scaler/origins')
    return values


def _reconstruct_old(fit, stop, refit, validation, old, check):
    rows = []
    for stage, training, evaluation, lambdas in (('A_fit', fit, stop, (.01, .1, 1.)),
                                                ('A60', refit, validation, (.01,))):
        if training is None or evaluation is None:
            continue
        stats = decoder.prepare_statistics(training['phi'], _innovation_target(training, legacy=True), training['mask'], check=check)
        for arm, suffix in (('D_SHARED', 'SHARED'), ('D_HORIZON', 'HORIZON')):
            for lam in lambdas:
                check()
                state = decoder.fit_statistics(stats, arm, lam)
                _, actual = _evaluate(evaluation, state)
                expected = next(row for row in old['rows'] if row['method'] == 'B2_DECODER_'+suffix and row['stage'] == stage and row['lambda_value'] == lam)
                difference = {name: abs(actual[name]-expected[name]) for name in ('MSE', 'MAE')}
                if max(difference.values()) > ORIGINAL_TOLERANCE:
                    raise ValueError('Original B03 reconstruction exceeds predeclared1e-8 tolerance')
                rows.append(dict(method=arm, stage=stage, lambda_value=lam, MSE=actual['MSE'], MAE=actual['MAE'],
                                 original_MSE=expected['MSE'], original_MAE=expected['MAE'],
                                 absolute_difference=difference, tolerance=ORIGINAL_TOLERANCE, status='PASS'))
    return rows


def execute_job(job, ctx):
    started, cpu_started = time.time(), time.process_time()
    check = budget_checker(ctx, started)
    verify_check = budget_checker(ctx, started, verification=True)
    public = Path(ctx['report'])/'jobs'/(job['job_id']+'.json')
    output = Path(ctx['cache'])/'decoder'/job['job_id']
    output.mkdir(parents=True, exist_ok=True)
    old, evidence, inventory, modules, task = _input_contract(job, ctx)
    fit = _load_bound(inventory['A_fit', 'fit'], modules, task, check)
    stop = _load_bound(inventory['A_fit', 'A_stop'], modules, task, check)
    refit = _load_bound(inventory['A60', 'fit'], modules, task, check)
    old_reconstruction = _reconstruct_old(fit, stop, None, None, old, check)
    preparation_seconds = time.time()-started
    selection_path = output/'SELECTION_SEAL.json'
    source_paths = [Path(__file__), Path(decoder.__file__)]
    source_hashes = {p.name: sha(p) for p in source_paths}
    def seal(selection):
        value = dict(selection, job_id=job['job_id'], scientific_identity=job.get('scientific_identity'),
                     source_hashes=source_hashes, sealed_at=datetime.now(timezone.utc).isoformat(),
                     V_features_loaded=False, new_F_calls=0)
        value['selection_sha256'] = digest(value)
        if selection_path.exists():
            saved = read(selection_path)
            saved_body = {key: child for key, child in saved.items() if key != 'selection_sha256'}
            if (digest(saved_body) != saved['selection_sha256'] or saved['selections'] != selection['selections'] or
                    saved['source_hashes'] != source_hashes or saved['scientific_identity'] != job.get('scientific_identity')):
                raise ValueError('Existing immutable selection seal differs')
        if not selection_path.exists():
            atomic_json(selection_path, value)
    def load_validation():
        return _load_bound(inventory['A60', 'V_select'], modules, task, check)
    with modules['blas_controller']:
        result = run_family(fit, stop, refit, load_validation, seal, check)
    verification_started = time.time()
    old_reconstruction += _reconstruct_old(None, None, refit, result['validation'], old, verify_check)
    private = [binding(selection_path, ctx)]
    comparison = []
    baseline = result['predictions']['D_SHARED']
    baseline_scores = score_arrays(baseline, result['validation']['target'], result['validation']['mask'])
    for row in result['rows']:
        verify_check()
        arm = row['arm']
        prediction_path, state_path = output/(arm+'_predictions.npz'), output/(arm+'_state.json')
        coefficient_path = output/(arm+'_coefficients.npz')
        state = result['states'][arm]
        values = dict(pred_normalized=result['predictions'][arm], target_normalized=result['validation']['target'],
                      observed=result['validation']['mask'], origins=inventory['A60', 'V_select']['origins'],
                      channel_ids=np.asarray(task['columns']))
        from research.adapter_diagnostics_20261017_v2.new_jobs import save_npz
        save_npz(prediction_path, **values)
        atomic_json(state_path, state)
        coefficient_values = dict(coefficients=np.asarray(state['coefficients']), rms=np.asarray(state['rms']),
                                  zero_features=np.asarray(state['zero_features']), basis=decoder.horizon_basis(state['H']))
        if state['gain'] is not None:
            coefficient_values['gain'] = np.asarray(state['gain'])
        save_npz(coefficient_path, **coefficient_values)
        with np.load(prediction_path, allow_pickle=False) as archive:
            replay = score_arrays(archive['pred_normalized'], archive['target_normalized'], archive['observed'])
            fingerprints = {key: dict(sha256=array_sha(archive[key]), shape=list(archive[key].shape), dtype=str(archive[key].dtype)) for key in archive.files}
        if any(abs(replay[name]-row[name]) > 1e-12 for name in ('MSE', 'MAE')):
            raise ValueError('Saved FP64 prediction independent score mismatch')
        for item in (prediction_path, state_path, coefficient_path):
            private.append(binding(item, ctx))
        row.update(seed=int(job['seed']), dataset='jena', C=21, K=6, L=512, H=48,
                   prediction_binding=binding(prediction_path, ctx), state_binding=binding(state_path, ctx),
                   coefficient_binding=binding(coefficient_path, ctx), array_fingerprints=fingerprints,
                   saved_array_independent_FP64='PASS', prediction_sha256=sha(prediction_path),
                   scaler_sha256=inventory['A60', 'V_select']['key']['scaler_sha'],
                   PE_checkpoint_sha256=inventory['A60', 'V_select']['key']['PE_checkpoint_sha256'],
                   serialized_head_bytes=coefficient_path.stat().st_size,
                   coefficient_norm=float(np.linalg.norm(state['coefficients'])),
                   gain_norm=None if state['gain'] is None else float(np.linalg.norm(state['gain'])),
                   fitting_diagnostics=state['diagnostics'], identifiability=state['identifiability'],
                   selected_epoch=old['original_selected_epoch'], units='normalized_A60', point_statistic='quantile_0.5',
                   V_exposure='repeated development evaluation; no new holdout claim',
                   matching_status='MATCHED_FIXED_PE_FEATURES', new_F_calls=0, device='cpu',
                   selection_seal=binding(selection_path, ctx))
        paired = []
        for candidate, control in zip(row['origin_statistics'], baseline_scores['origin_statistics']):
            paired.append(dict(origin_index=candidate['origin_index'],
                               delta_MSE=None if candidate['MSE'] is None else candidate['MSE']-control['MSE'],
                               delta_MAE=None if candidate['MAE'] is None else candidate['MAE']-control['MAE'],
                               channel_delta_sse=(np.asarray(candidate['channel_sse'])-control['channel_sse']).tolist(),
                               channel_delta_sae=(np.asarray(candidate['channel_sae'])-control['channel_sae']).tolist(),
                               channel_counts=candidate['channel_counts']))
        comparison.append(dict(candidate=arm, comparator='D_SHARED', n_origins=len(paired), seed=int(job['seed']),
                               delta_MSE=row['MSE']-baseline_scores['MSE'], delta_MAE=row['MAE']-baseline_scores['MAE'],
                               origin_paired=paired, evaluation_stride=48, horizon_overlap=False,
                               uncertainty='paired descriptive effects; no IID significance claim'))
    if any(sha(path) != source_hashes[path.name] for path in source_paths):
        raise ValueError('Scientific decoder source changed after selection seal')
    value = dict(status='COMPLETE', family='D_FIXED_PE_40_GRID', seed=int(job['seed']), rows=result['rows'],
                 selection_trials=result['selection_trials'], selections=result['selections'],
                 comparisons=comparison, original_B03_reconstruction=old_reconstruction,
                 original_reuse=evidence, selection_seal=binding(selection_path, ctx),
                 private_artifacts=private,
                 feature_caches=[dict(scope=scope, split=split, key=entry['key'], artifact=binding(entry['path'], ctx))
                                 for (scope, split), entry in inventory.items()],
                 scientific_source_hashes=source_hashes, new_F_calls=0, new_F_rows=0, new_model_loads=0,
                 frozen_F_weight_hash=modules['legacy'].F_SHA,
                 F_integrity='No F object instantiated; consumed byte-verified producer caches bound to pinned F state hash',
                 timers=dict(result['timers'], total_wall_seconds=time.time()-started,
                             preparation_and_original_A_stop_reconstruction_seconds=preparation_seconds,
                             verification_and_serialization_seconds=time.time()-verification_started,
                             process_cpu_seconds=time.process_time()-cpu_started,
                             cache_extraction_seconds=0., original_cache_extraction_seconds=sum(e['manifest']['seconds'] for e in inventory.values())),
                 limit_seconds=1800, fitting_limit_seconds=1440, verification_reserve_seconds=360,
                 original_reconstruction_tolerance=ORIGINAL_TOLERANCE,
                 original_reconstruction_order='Original A_fit grid first; original A60 chosen .01 only after new selection seal',
                 comparison_scope='Each arm selected its own lambda/gamma on A_stop; fixed PE; head-plus-selection-recipe comparison',
                 cost_valid=False, cost_status='UNMEASURED; exact FP64 head install/reload API supplied')
    atomic_json(public, value)
    return dict(status='COMPLETE', reason=None, artifact_paths=[str(public)], result=value)
