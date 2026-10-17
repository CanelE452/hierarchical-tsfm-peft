"""Exact scored-bundle reload and physical-input whole-path cost jobs."""
import csv
import importlib
from contextlib import nullcontext
from pathlib import Path
import platform
import time
import types

import numpy as np
import torch

from .contracts import atomic_json, binding, read, resolve_binding, sha


WARMUPS, TIMED_TRIALS, STAGE_TRIALS = 10, 20, 1


def summarize_trials(trials):
    rows = [row for row in trials if row['phase'] == 'timed']
    if not rows or len({row['batch_size'] for row in rows}) != 1:
        raise ValueError('One batch facet with timed trials is required')
    milliseconds = np.asarray([row['whole_seconds'] for row in rows], np.float64)*1000
    batch = rows[0]['batch_size']
    q25, median, q75, p95 = np.quantile(milliseconds, [.25, .5, .75, .95])
    return dict(median_batch_ms=float(median), IQR_batch_ms=float(q75-q25), p95_batch_ms=float(p95),
                median_ms_per_origin=float(median/batch), IQR_ms_per_origin=float((q75-q25)/batch),
                p95_ms_per_origin=float(p95/batch), timed_trials=len(rows), batch_size=batch,
                p95_scope='20-sample descriptive quantile; no stable tail guarantee',
                stage_sum_equals_whole_claimed=False)


def _windows(task, origins):
    origins = np.asarray(origins)
    if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or not len(origins):
        raise ValueError('Integer origin vector required')
    if (origins < task['L']).any() or (origins > len(task['physical'])-task['H']).any():
        raise ValueError('Cost origin outside physical task')
    return (np.stack([task['physical'][o-task['L']:o] for o in origins]).astype(np.float32),
            np.stack([task['observed'][o-task['L']:o] for o in origins]).astype(bool))


def parity_saved(route, task, saved, batch, *, atol=1e-5, rtol=1e-4, check=None):
    if not np.array_equal(saved['channel_ids'].astype(str), np.asarray(task['columns']).astype(str)):
        raise ValueError('Saved channel order differs from actual task')
    expected = np.asarray(saved['pred_normalized'], np.float64)
    oo = np.asarray(saved['origins'])
    if expected.shape != (len(oo), task['H'], len(task['columns'])) or not np.isfinite(expected).all():
        raise ValueError('Saved prediction axes or finite values invalid')
    bounds = task.get('split_ranges', {}).get('V_select')
    if bounds is not None and (len(np.unique(oo)) != len(oo) or (oo < max(task['L'], bounds[0])).any() or (oo+task['H'] > bounds[1]).any()):
        raise ValueError('Parity origins must be unique registered V origins')
    predictions, counts = [], []
    for first in range(0, len(oo), batch):
        if check is not None:
            check()
        physical, observed = _windows(task, oo[first:first+batch])
        value = route.predict_physical(physical, observed)
        if hasattr(route, 'latent_k'):
            k = len(task['columns']) if route.family in ('RAW','LORA') else route.latent_k
            actual_rows = route.last_receipt.get('F_logical_streams')
            if actual_rows != len(physical)*k:
                raise ValueError('Actual F logical rows differ from B times active K')
        predictions.append((value-route.mean)/route.std)
        counts.append(len(physical))
    actual = np.concatenate(predictions)
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return dict(status='PASS' if passed else 'FAIL', n_origins=len(oo), last_batch_size=counts[-1],
                max_abs=float(np.max(np.abs(actual-expected))), atol=atol, rtol=rtol,
                comparison='physical route inverse-normalized versus entire saved normalized V prediction',
                actual=actual)


def eligible_candidates(rows):
    accepted, skipped = [], []
    for value in rows:
        row = dict(value)
        reason = None
        if row.get('status', 'COMPLETE') not in ('COMPLETE', 'REUSED_VERIFIED'):
            reason = row.get('reason') or 'Accuracy execution is incomplete: '+str(row.get('status'))
        elif row.get('evaluation', 'V_select') != 'V_select':
            reason = 'Only selected V accuracy rows are cost candidates'
        elif any(row.get(key) == 'FAIL' for key in ('independent_FP64', 'saved_array_independent_FP64', 'verification_status')):
            reason = 'Accuracy verification failed'
        elif not row.get('prediction_binding'):
            reason = 'Exact saved prediction binding is absent'
        if reason:
            skipped.append(dict(row, status='NOT_MEASURED', cost_valid=False, reason=reason))
        else:
            accepted.append(row)
    return accepted, skipped


def _public_parity(value):
    return {key: item for key, item in value.items() if key != 'actual'}


def _dtypes(route, candidate):
    foundation = getattr(getattr(route, 'pool', None), 'foundation', None)
    module = getattr(foundation, 'model', foundation)
    values = list(module.parameters()) if isinstance(module, torch.nn.Module) else []
    floating = {str(value.dtype).replace('torch.', '') for value in values if value.is_floating_point()}
    if floating and floating != {'float32'}:
        raise ValueError('Frozen F must retain native FP32 weights')
    model = getattr(route,'model',None)
    backbone_parameters = sum(value.numel() for value in values)
    adapter_values = list(model.parameters()) if isinstance(model,torch.nn.Module) else []
    adapter_parameters = sum(value.numel() for value in adapter_values)
    refiner = getattr(model,'post_decoder_calibrator',None)
    refiner_values = list(refiner.parameters()) if isinstance(refiner,torch.nn.Module) else []
    refiner_buffers = list(refiner.buffers()) if isinstance(refiner,torch.nn.Module) else []
    return dict(backbone_dtype='float32' if floating else 'NO_FOUNDATION_INSTANTIATED',
                head_dtype=candidate.get('head_dtype', candidate.get('core_dtype', 'float32')),
                core_dtype=candidate.get('core_dtype', 'float32'),
                prediction_dtype=candidate.get('prediction_dtype', 'UNKNOWN'),
                physical_output_dtype='float64', metric_accumulation_dtype='float64',
                F_params=backbone_parameters,adapter_params=adapter_parameters,
                adapter_real_degrees_of_freedom=sum(value.numel()*(2 if value.is_complex() else 1) for value in adapter_values),
                refiner_parameter_elements=sum(value.numel() for value in refiner_values),
                refiner_complex_elements=sum(value.numel() for value in refiner_values if value.is_complex()),
                refiner_real_degrees_of_freedom=sum(value.numel()*(2 if value.is_complex() else 1) for value in refiner_values),
                refiner_parameter_bytes=sum(value.numel()*value.element_size() for value in refiner_values),
                refiner_buffer_bytes=sum(value.numel()*value.element_size() for value in refiner_buffers),
                refiner_parameter_dtypes=sorted({str(value.dtype).replace('torch.','') for value in refiner_values}),
                refiner_target_trained_real_degrees=candidate.get('refiner_target_trained_real_degrees',0),
                retained_base_target_trained_params=candidate.get('frozen_base_target_trained_params',candidate.get('frozen_PE_target_trained_params')),
                base_decoder_dtype=str(model.D.dtype).replace('torch.','') if hasattr(model,'D') else 'NOT_PRESENT',
                total_resident_params=backbone_parameters+adapter_parameters,
                serialized_adapter_bytes=candidate.get('serialized_adapter_bytes',candidate.get('serialized_head_bytes','UNKNOWN')),
                serialized_base_checkpoint_bytes=candidate.get('serialized_base_checkpoint_bytes','UNKNOWN'),
                serialized_refiner_bytes=candidate.get('serialized_refiner_bytes','NOT_APPLICABLE'),
                serialized_adapter_scope='sum of separate base checkpoint and calibrator bytes' if candidate.get('refiner_bundle') else 'actual adapter bundle file; D also lists head artifact bytes separately')


def install_calibrator(route, calibrator):
    """Apply a saved post-decoder calibrator to normalized X and anchored B2 Y."""
    if not isinstance(route.model,torch.nn.Module) or hasattr(route.model,'post_decoder_calibrator'):
        raise ValueError('An uncalibrated base model is required')
    route.model.add_module('post_decoder_calibrator',calibrator.eval().requires_grad_(False))
    original = route._predict_normalized
    def predict(self,x,physical,observed):
        base_prediction = original(x,physical,observed)
        return self._step('refiner',lambda:self.model.post_decoder_calibrator(x,base_prediction))
    route._predict_normalized = types.MethodType(predict,route)
    return route


def measure_candidate(candidate, task, saved, loader, *, batch, device, lease=None, check=None,
                      measure=None, atol=1e-5, rtol=1e-4):
    """Load twice; the second exact serialized load is the timed deployment route."""
    from research.adapter_diagnostics_20261017_v2.existing_jobs import RSSSampler
    import psutil
    if batch not in (1, 8) or (device != 'cpu' and not str(device).startswith('cuda')):
        raise ValueError('Registered cost facets are CPU/CUDA, batch 1/8')
    row = {key: candidate.get(key) for key in ('run_id', 'job_id', 'method', 'arm', 'seed', 'dataset', 'C', 'K', 'L', 'H')}
    row.update(accuracy_run_id=candidate.get('run_id'), accuracy_job_id=candidate.get('job_id'),
               device=device, batch_size=batch, cost_valid=False, status='FAILED_PARITY',
               precision_comparison='METHOD_LEVEL', warmups=WARMUPS, timed_trials=TIMED_TRIALS,
               metric_accumulation_dtype='float64', units='ms', cpu_blas_threads=2,
               memory_scope='cost phase; training excluded; process RSS sampled at 5ms',
               load_included_in_warm=False, stage_probe_is_intrusive_separate_scope=True,
               process_scope='one cohort/device worker, sequential routes; not fresh process per method',
               CUDA_reserved_scope='allocator may retain earlier route allocations in this worker',
               whole_path='physical+mask -> causal fill -> normalize -> P/E/F/D/refiner -> inverse scale -> CPU',
               accuracy_prediction_binding=candidate.get('prediction_binding'),
               accuracy_bundle_binding=candidate.get('bundle_binding'),
               accuracy_base_checkpoint_binding=candidate.get('base_checkpoint',candidate.get('parent_bundle_binding')),
               accuracy_refiner_binding=candidate.get('refiner_bundle'),
               accuracy_parent_kind=candidate.get('parent_kind'),
               base_training_device=candidate.get('base_training_device'),base_inference_device=device,
               full_original_channel_context=candidate.get('full_original_channel_context'),
               accuracy_source_kind=candidate.get('accuracy_source_kind','NEW_CAMPAIGN_SELECTED_SCORED_ROW'),
               accuracy_source_evidence=candidate.get('accuracy_source_evidence',[]),
               accuracy_result_binding=candidate.get('accuracy_result_binding'),
               accuracy_receipt_binding=candidate.get('accuracy_receipt_binding'),
               accuracy_source_sha256=candidate.get('source_sha'),accuracy_config_sha256=candidate.get('config_sha'),
               accuracy_foundation_sha256=candidate.get('model_sha'),
               data_sha256=candidate.get('data_sha256',candidate.get('data_sha',task.get('data_sha256'))),
               scaler_sha256=candidate.get('scaler_sha256'), origins_sha256=candidate.get('origins_sha256'),
               point_statistic=candidate.get('point_statistic','quantile_0.5'),
               accuracy_units=candidate.get('units','normalized_A60'),
               cached_forecast_reads_in_timed_path=0,
               serialized_head_bytes=candidate.get('serialized_head_bytes','NOT_APPLICABLE'),
               source_kind='NEW_ACTUAL_MEASUREMENT', training_seconds=candidate.get('training_seconds', 'UNKNOWN'))
    routes, cold, trials = [], [], []
    sampler = RSSSampler(lambda: psutil.Process().memory_info().rss)
    try:
        reference = None
        for index in range(2):
            if check is not None:
                check()
            started = time.perf_counter()
            route = loader()
            routes.append(route)
            row.update(_dtypes(route, candidate))
            physical, observed = _windows(task, saved['origins'][:batch])
            if len(physical) != batch:
                raise ValueError('Cost batch must contain B distinct saved V origins')
            route.predict_physical(physical, observed)
            cold.append(dict(load_plus_first_physical_seconds=time.perf_counter()-started,
                             scope='new route and foundation load plus first physical prediction; OS page cache uncontrolled'))
            parity = parity_saved(route, task, saved, batch, atol=atol, rtol=rtol, check=check)
            row['saved_V_parity' if index == 0 else 'serialized_reload_saved_V_parity'] = _public_parity(parity)
            if parity['status'] != 'PASS':
                row['reason'] = 'Full saved V physical-route parity failed'
                return dict(row=row, trials=[])
            if reference is None:
                reference = parity['actual']
                route.close()
                routes.remove(route)
                del route
            else:
                passed = bool(np.allclose(reference, parity['actual'], atol=atol, rtol=rtol))
                row['serialized_reload_parity'] = dict(status='PASS' if passed else 'FAIL', atol=atol, rtol=rtol,
                    max_abs=float(np.max(np.abs(reference-parity['actual']))), n_origins=len(reference))
                if not passed:
                    row['reason'] = 'Serialized route reload parity failed'
                    return dict(row=row, trials=[])
        if measure is None:
            measure = importlib.import_module('measure_costs').measure_balanced
        sampler.start()
        try:
            trials = measure([route], physical, observed, warmups=WARMUPS, timed_trials=TIMED_TRIALS,
                             stage_trials=STAGE_TRIALS, lease=lease)
            row['memory'] = route.memory_state()
        finally:
            sampler.stop()
        row.update(summarize_trials(trials))
        row.update(cost_valid=True, status='COMPLETE', reason=None, cold_observations=cold,
                   cold_time_scope='two observed loads, no repeated disk-cold distribution',
                   sampled_process_RSS_peak_MiB=sampler.peak/1024**2,
                   F_rows_per_timed_batch=sorted({item['F_logical_streams'] for item in trials if item['phase']=='timed'}))
        return dict(row=row, trials=trials)
    finally:
        for route in routes:
            route.close()


def _load_prediction(candidate, ctx):
    path = resolve_binding(candidate['prediction_binding'], ctx)
    with np.load(path, allow_pickle=False) as archive:
        required = {'pred_normalized', 'origins', 'channel_ids'}
        if not required.issubset(archive.files):
            raise ValueError('Bound scored prediction archive schema differs')
        return {key: archive[key] for key in required}


def historical_raw_candidate(job, ctx, modules):
    """Reuse the exact old Raw accuracy archive, then time real F on this hardware."""
    if job['cohort'] != 'jena':
        return None
    from research.adapter_diagnostics_20261017_v2.costs import prediction_manifest
    run_id = 'primary_jena_E1_RAW_BOLT_SMALL_k0_s92601'
    relative = Path('research/informs_broad_validation_20261015_v1/predictions')/run_id/'V_select/PREDICTION_MANIFEST.json'
    manifest, evidence = prediction_manifest(ctx,relative)
    archive = Path(modules['old_cache'])/'predictions'/run_id/'V_select.npz'
    if sha(archive) != manifest['archive_sha256'] or manifest['run_id'] != run_id or manifest['split'] != 'V_select':
        raise ValueError('Historical Raw accuracy archive identity mismatch')
    return dict(run_id=run_id,job_id='HISTORICAL_RAW_V',method='RAW',dataset='jena',seed=92601,
        C=21,K=21,L=512,H=48,status='REUSED_VERIFIED',evaluation='V_select',
        prediction_binding=binding(archive,ctx),independent_FP64='PENDING_COST_FULL_V_PARITY',
        accuracy_source_kind='REUSED_HISTORICAL_SCORED_RAW',accuracy_source_evidence=evidence,
        scaler_sha256=manifest['scaler_sha'],data_sha=manifest['data_sha'],
        array_fingerprints=manifest['arrays'],core_dtype='float32',head_dtype='float32',prediction_dtype='float32',
        raw_spec=dict(family='RAW',run_id=run_id,foundation_sha=modules['legacy'].F_SHA,canonical_fit_end=12579,
                      row_microbatch='full'),units='normalized_A60',point_statistic='quantile_0.5')


def collect_candidates(job, ctx):
    rows = []
    for path in sorted((Path(ctx['report'])/'jobs').glob('*.json')):
        value = read(path)
        if not isinstance(value, dict) or value.get('family') == 'WHOLE_PATH_COST':
            continue
        receipt_path = Path(ctx['report'])/'receipts'/(path.stem+'.json')
        receipt = read(receipt_path) if receipt_path.is_file() else {}
        relative = path.relative_to(Path(ctx['report'])).as_posix()
        bound = (receipt.get('status') in ('COMPLETE','REUSED_VERIFIED') and receipt.get('result') == value
                 and receipt.get('artifact_hashes',{}).get(relative) == sha(path))
        for item in value.get('rows', []):
            if item.get('dataset') != job['cohort'] or int(item.get('seed', -1)) != int(job.get('seed') or 92601):
                continue
            if item.get('evaluation') not in (None, 'V_select'):
                continue
            row = dict(item, status=value.get('status', 'UNKNOWN') if bound else 'UNBOUND_ACCURACY_RECEIPT',
                       job_id=value.get('run_id', path.stem))
            row.setdefault('run_id', path.stem+':'+str(item.get('arm', item.get('method'))))
            for key in ('source_sha','config_sha','data_sha','model_sha'):
                if key in value:
                    row.setdefault(key,value[key])
            row['accuracy_result_binding'] = binding(path,ctx)
            if receipt:
                row['accuracy_receipt_binding'] = binding(receipt_path,ctx)
            if 'origins' in row.get('array_fingerprints',{}):
                origin_hash = row['array_fingerprints']['origins']
                row['origins_sha256'] = origin_hash['sha256'] if isinstance(origin_hash,dict) else origin_hash
            if value.get('bundle_binding'):
                row['bundle_binding'] = value['bundle_binding']
            if row.get('refiner_bundle'):
                row.pop('bundle_binding',None)
            if row.get('state_binding') and row['status'] in ('COMPLETE','REUSED_VERIFIED'):
                if row.get('parent_bundle_binding'):
                    row['bundle_binding'] = row['parent_bundle_binding']
                    parent = None
                elif row['dataset'] == 'jena':
                    from research.adapter_diagnostics_20261017_v2.existing_jobs import _run_root
                    parent = _run_root(ctx,int(row['seed']),'B2')/'selected_bundle.pt'
                else:
                    row.update(status='MISSING_EXACT_PARENT',reason='Conditional decoder lacks its exact parent_bundle_binding')
                    parent = None
                if parent is not None:
                    if row.get('PE_checkpoint_sha256') and sha(parent) != row['PE_checkpoint_sha256']:
                        raise ValueError('Decoder cost parent differs from its scored PE checkpoint')
                    row['bundle_binding'] = binding(parent,ctx)
            if row.get('bundle_binding') or (row.get('refiner_bundle') and row.get('base_checkpoint')) or row.get('status') != 'COMPLETE':
                rows.append(row)
    for registered in ctx.get('plan',{}).get('jobs',[]):
        if registered.get('kind') not in ('lift_jobs','decoder_transfer') or registered.get('cohort','jena') != job['cohort']:
            continue
        if int(registered.get('seed') or 92601) != int(job.get('seed') or 92601):
            continue
        identifier = registered['job_id']
        if any(row['job_id'] == identifier for row in rows):
            continue
        result_path = Path(ctx['report'])/'jobs'/(identifier+'.json')
        state = read(result_path) if result_path.exists() else {}
        methods = ('B2_LIFT','B2_OUTPUT_GAIN') if registered['kind']=='lift_jobs' else ('D_SHARED','D_HORIZON','D_LINEAR','D_GAIN')
        for method in methods:
            rows.append(dict(run_id=identifier+':'+method,job_id=identifier,method=method,dataset=job['cohort'],
                             seed=registered.get('seed',92601),status=state.get('status','NOT_RUN_DEPENDENCY'),
                             reason=state.get('reason','Registered conditional method has no completed scored prediction row')))
    return eligible_candidates(rows)


def _route_loader(candidate, job, ctx, modules, task):
    raw = candidate.get('raw_spec')
    calibrator_binding = candidate.get('refiner_bundle')
    base_binding = candidate.get('base_checkpoint') if calibrator_binding else candidate.get('bundle_binding')
    bundle_path = None if raw else resolve_binding(base_binding, ctx)
    bundle = None if raw else torch.load(bundle_path, map_location='cpu', weights_only=False)
    if bundle is not None:
        for key in ('source_sha','config_sha','data_sha','model_sha'):
            if key in candidate and candidate[key] != bundle['contract'][key]:
                raise ValueError('Cost bundle differs from scored accuracy contract: '+key)
    if bundle_path is not None:
        candidate['base_training_device'] = bundle['spec'].get('device','UNKNOWN')
        candidate['serialized_base_checkpoint_bytes'] = bundle_path.stat().st_size
        candidate['serialized_adapter_bytes'] = bundle_path.stat().st_size
        if calibrator_binding:
            if bundle['spec']['family'] != 'B2' or candidate.get('state_binding'):
                raise ValueError('Output calibration requires original anchored B2 decoder')
            candidate['serialized_refiner_bytes'] = resolve_binding(calibrator_binding,ctx).stat().st_size
            candidate['serialized_adapter_bytes'] += candidate['serialized_refiner_bytes']
    cache_root = Path(ctx['cache']) if raw else next(Path(ctx[name]).resolve() for name in ('cache','previous_cache','asset') if bundle_path.is_relative_to(Path(ctx[name]).resolve()))
    spec = dict(raw) if raw else dict(bundle['spec'], run_id=candidate['run_id'], artifact_cache_relative_path=bundle_path.relative_to(cache_root).as_posix(),
                artifact_sha256=sha(bundle_path), device=job['device'])
    deployment = modules['deployment']
    def load():
        from .audit import load_foundation
        from .training_jobs import model_factory
        factory = ctx.get('foundation_loader', load_foundation)
        pool = deployment.FoundationPool(None, job['device'], factory=lambda p,h,m,d: factory(ctx,modules,h,d))
        route = None
        try:
            use_factory = bundle is not None and (bundle['spec']['family'] in ('P_SPLIT','P_LOCAL_CONTROL') or bundle['spec'].get('init_kind') == 'DCA')
            scope = model_factory(dict(ctx, modules=modules)) if use_factory else nullcontext()
            with scope:
                route = deployment.Route(spec, task, cache_root, device=job['device'], foundation_pool=pool)
            route.owns_pool = True
            actual_scaler = modules['fit'].scaler_fingerprint(route.mean,route.std)
            if candidate.get('scaler_sha256') and candidate['scaler_sha256'] != actual_scaler:
                route.close()
                raise ValueError('Cost deployment scaler differs from scored accuracy scaler')
            if candidate.get('state_binding'):
                from .decoder import install_head
                state = read(resolve_binding(candidate['state_binding'], ctx))
                with np.load(resolve_binding(candidate['coefficient_binding'], ctx), allow_pickle=False) as archive:
                    for name in ('coefficients','rms','zero_features'):
                        if not np.array_equal(np.asarray(state[name]),archive[name]):
                            raise ValueError('Serialized decoder JSON/NPZ disagree: '+name)
                install_head(route.model, state)
            if calibrator_binding:
                from .lift_jobs import load_calibrator
                calibrator = load_calibrator(candidate['method'],calibrator_binding,ctx,job['device'])
                install_calibrator(route,calibrator)
            return route
        except BaseException:
            if route is not None:
                route.close()
            else:
                pool.close()
            raise
    return load


def _write_trials(path, trials):
    keys = sorted(set().union(*(row.keys() for row in trials))) if trials else ['run_id','phase','trial','batch_size','whole_seconds']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trials)


def execute_job(job, ctx):
    """One exclusive cohort/device cost job; failures remain per-route evidence."""
    from .audit import load_modules, load_task
    modules = load_modules(ctx)
    torch.set_num_threads(2)
    task = load_task(ctx, job['cohort'], modules)
    measure_module = importlib.import_module('measure_costs')
    lease = ctx.get('lease')
    if str(job['device']).startswith('cuda') and lease is None:
        lease = measure_module.worker_lease_check(ctx['cache'], job['device'])
    def check():
        if time.time() >= min(ctx.get('job_deadline', float('inf')), ctx.get('deadline', float('inf'))):
            raise TimeoutError('Cost job registered deadline reached')
        if lease is not None:
            lease.check()
    candidates, skipped = collect_candidates(job, ctx)
    try:
        raw = historical_raw_candidate(job,ctx,modules)
        if raw is not None:
            candidates.insert(0,raw)
    except (FileNotFoundError,ValueError,KeyError) as error:
        skipped.append(dict(run_id='HISTORICAL_RAW_V',method='RAW',dataset=job['cohort'],status='NOT_MEASURED',cost_valid=False,
                            reason='Exact historical Raw scored archive unavailable: '+type(error).__name__))
    rows, trials, private = list(skipped), [], []
    for candidate in candidates:
        for batch in job.get('batches', [1,8]):
            check()
            try:
                saved = _load_prediction(candidate, ctx)
                loader = _route_loader(candidate, job, ctx, modules, task)
                measured = measure_candidate(candidate, task, saved, loader, batch=batch, device=job['device'],
                    lease=lease, check=check, measure=measure_module.measure_balanced)
                rows.append(measured['row'])
                trials.extend(dict(item, accuracy_run_id=candidate['run_id'], cost_job_id=job['job_id']) for item in measured['trials'])
                private.extend(candidate[key] for key in ('prediction_binding','bundle_binding','state_binding','coefficient_binding','base_checkpoint','refiner_bundle','parent_bundle_binding') if key in candidate)
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception as error:
                rows.append(dict(run_id=candidate['run_id'], method=candidate.get('method'), device=job['device'], batch_size=batch,
                                 status='FAILED_COST_ROUTE', cost_valid=False, reason=type(error).__name__,
                                 accuracy_prediction_binding=candidate['prediction_binding']))
    output = Path(ctx['report'])/'jobs'/(job['job_id']+'.json')
    trial_path = output.with_suffix('.trials.csv')
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    _write_trials(trial_path, trials)
    result = dict(status='COMPLETE', family='WHOLE_PATH_COST', rows=rows, private_artifacts=private,
                  cohort=job['cohort'], device=job['device'], hardware=dict(platform=platform.platform(), processor=platform.processor(),
                  GPU=torch.cuda.get_device_name() if str(job['device']).startswith('cuda') else None),
                  source_hashes={str(Path(__file__).relative_to(Path(ctx['repo']))).replace('\\','/'):sha(__file__)},
                  protocol=dict(warmups=WARMUPS,timed_trials=TIMED_TRIALS,stage_trials=STAGE_TRIALS),
                  cost_valid_routes=sum(bool(row.get('cost_valid')) for row in rows),
                  historical_costs='Contextual source measurements retained separately by reporting; never new hardware measurements')
    atomic_json(output, result)
    return dict(status='COMPLETE', reason=None, artifact_paths=[str(output),str(trial_path)],result=result)
