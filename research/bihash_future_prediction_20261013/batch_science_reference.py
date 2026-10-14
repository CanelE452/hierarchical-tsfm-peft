"""Independent batch-matched references from the sealed science APIs.

Only normalized past inputs and existing prediction/stat/origin members are
loaded. Representation readouts get a separate model because the science API
recomputes memory buffers; the deployment model is never used as its reference.
"""
import gc
import numpy as np
import torch
import runtime as rt
import model as bm
import train as tr
import probes as pr
import run as ru


PROTOCOL = {
    'reference_kind': 'original_science_API_batch_matched_independent_model',
    'physical_atol': .002, 'physical_rtol': 2e-5,
    'normalized_reinference_atol': 1e-5, 'normalized_reinference_rtol': 2e-6,
    'timing_scope': 'Original benchmark.deployment with warmup2/repeats7/sync; reference inference and model load excluded',
    'readout_side_effect_isolation': 'Science and deployment selected-checkpoint models are separate; science model is freed before timing',
    'bulk_science_accuracy_or_selection_changed': False,
    'backend_limit': 'Original NumPy ridge/means are retained; CPU science versus CUDA deployment can still fail parity',
    'original_bulk_failure_relabelled': False}


def row_key(record):
    return record['dataset'], record['identity'], record['route'], record['batch']


def verify_artifact(artifact):
    path = rt.ROOT / artifact['path']
    if not path.is_file() or rt.sha(path) != artifact['sha256']:
        raise ValueError('COST_REFERENCE_ARTIFACT_SHA: ' + str(path))
    return path


def build_routes(dev, fresh, probes, evaluation, lock):
    cases = ru.selected_cases(dev, fresh, lock)
    ru.assert_full_training(dev, fresh)
    ru.assert_selected_coverage(cases, probes['readouts'], probes['controls'], lock)
    routes, aliases = [], []
    for ds in ru.LEGACY + ru.FRESH:
        for case in (c for c in cases if c['dataset'] == ds):
            matches = [r for r in evaluation['core_evaluation'] if r['dataset'] == ds and r['source_label'] == case['label']]
            if len(matches) != 1: raise ValueError('COST_CORE_SCIENCE_IDENTITY_NOT_UNIQUE')
            record = matches[0]; regular = [r['prediction'] for r in record['entries'] if r['variant'] == 'regular']
            if len(regular) != 1: raise ValueError('COST_REGULAR_SCIENCE_NOT_UNIQUE')
            routes.append({'dataset': ds, 'identity': case['label'], 'route': 'CORE', 'split': record['split'],
                'case': case, 'science_prediction': regular[0], 'model_artifacts': [case['training']['selected_checkpoint']]})
        for group in (c for c in probes['controls'] if c['dataset'] == ds):
            for control in group['methods']:
                matches = [r for r in evaluation['control_evaluation'] if r['dataset'] == ds and r['method'] == control['method']]
                if len(matches) != 1: raise ValueError('COST_CONTROL_SCIENCE_IDENTITY_NOT_UNIQUE')
                record = matches[0]
                routes.append({'dataset': ds, 'identity': control['method'], 'route': control['method'], 'split': record['split'],
                    'control': control, 'science_prediction': record['prediction'],
                    'model_artifacts': [] if 'model' not in control else [control['model']]})
        for group in (r for r in probes['readouts'] if r['dataset'] == ds):
            case = next(c for c in cases if c['dataset'] == ds and c['label'] == group['source_label'])
            for readout in group['readouts']:
                if readout.get('duplicate_of'):
                    aliases.append({'dataset': ds, 'source_label': case['label'], 'route': readout['method'], 'canonical_route': readout['duplicate_of']})
                    continue
                matches = [r for r in evaluation['readout_evaluation'] if r['dataset'] == ds and
                    r['source_label'] == case['label'] and r['method'] == readout['method']]
                if len(matches) != 1: raise ValueError('COST_READOUT_SCIENCE_IDENTITY_NOT_UNIQUE')
                record = matches[0]
                routes.append({'dataset': ds, 'identity': case['label'] + '_' + readout['method'], 'route': readout['method'],
                    'split': record['split'], 'case': case, 'readout': readout, 'science_prediction': record['prediction'],
                    'model_artifacts': [case['training']['selected_checkpoint'], readout['model']]})
    keys = [(r['dataset'], r['identity'], r['route']) for r in routes]
    if len(keys) != 268 or len(set(keys)) != 268: raise ValueError('COST_CANONICAL_ROUTE_COVERAGE_NOT_EXACT_268')
    if any(r['split'] != ('val' if r['dataset'] in ru.LEGACY else 'eval') for r in routes):
        raise ValueError('COST_ROUTE_SPLIT_CHANGED')
    return routes, aliases


def load_past_only(dataset, fresh, evaluation, method_lock_ref):
    if dataset not in ru.FRESH: raise ValueError('COST_RECOVERY_REQUIRES_NEW_DATASET_PAST_PAYLOAD')
    contract = fresh['contracts'][dataset]
    if contract['method_lock']['sha256'] != method_lock_ref['sha256']: raise ValueError('COST_PAST_CONTRACT_METHOD_LOCK_MISMATCH')
    receipts = [r for r in evaluation['evaluation_open_receipts'] if r['dataset'] == dataset]
    if len(receipts) != 1 or not receipts[0]['opened_only_after_all_selected']:
        raise ValueError('COST_EVAL_NOT_ALREADY_OPENED_AFTER_GLOBAL_GATE')
    if receipts[0]['method_lock_sha256'] != method_lock_ref['sha256'] or receipts[0]['global_model_lock']['sha256'] != evaluation['global_model_lock']['sha256']:
        raise ValueError('COST_EVAL_OPEN_RECEIPT_LOCK_MISMATCH')
    artifact = contract['payloads']['eval_past_input']; path = verify_artifact(artifact)
    members = ('history', 'physical_history', 'history_observed', 'origins', 'mean', 'std')
    with np.load(path, allow_pickle=False) as z: values = {key: z[key].copy() for key in members}
    history, raw, mask = values['history'], values['physical_history'], values['history_observed']
    c = contract['C']; origins = values['origins']
    if history.shape != (len(origins),512,c) or raw.shape != history.shape or mask.shape != history.shape:
        raise ValueError('COST_PAST_ONLY_PAYLOAD_SHAPE')
    if values['mean'].shape != (c,) or values['std'].shape != (c,) or not np.isfinite(history).all() or not np.isfinite(values['mean']).all() or not np.isfinite(values['std']).all() or np.any(values['std'] <= 0):
        raise ValueError('COST_PAST_ONLY_PAYLOAD_STATS')
    if not np.array_equal(bm.en.ex.preprocess(raw,mask,values['mean'],values['std']),history):
        raise ValueError('COST_PAST_ONLY_PREPROCESS_NOT_EXACT')
    for field in ('mean','std'):
        if ru.dt.array_hash(values[field]) != contract['statistics'][field+'_sha256']:
            raise ValueError('COST_PAST_ONLY_STATISTICS_SHA')
    pack = {'dataset': dataset, 'C': c, 'K': contract['K'], 'mean': values['mean'], 'std': values['std'],
        'history': {'eval': history}, 'origins': {'eval': origins}, 'raw_history': {'eval': raw},
        'raw_history_observed': {'eval': mask}, 'columns': contract['columns']}
    return pack, {'artifact': artifact, 'accessed_members': list(members), 'EVAL_target_arrays_opened': False,
        'contract_statistics': contract['statistics'], 'existing_EVAL_open_receipt': receipts[0]}


def load_bulk_science(route, pack):
    split = route['split']; path = verify_artifact(route['science_prediction'])
    members = (split+'_prediction', split+'_origins', 'mean', 'std')
    with np.load(path, allow_pickle=False) as z: expected, origins, mean, std = [z[key].copy() for key in members]
    if not np.array_equal(origins,pack['origins'][split]) or not np.array_equal(mean,pack['mean']) or not np.array_equal(std,pack['std']):
        raise ValueError('COST_SCIENCE_PAST_ORIGINS_OR_STATS_NOT_EXACT')
    if expected.shape != (len(origins),48,len(mean)) or not np.isfinite(expected).all(): raise ValueError('COST_BULK_SCIENCE_SHAPE_OR_NONFINITE')
    return expected


def load_selected_model(case, device='cuda'):
    stored = torch.load(verify_artifact(case['training']['selected_checkpoint']),map_location='cpu',weights_only=True)
    state = stored['model']; basis = state['codec.fixed_basis'].numpy().copy()
    model = bm.FutureBiHash(np.zeros(basis.shape[0],np.float32),basis,case['seed'],case['method'],
        state['bank_delta'].numpy(),state['bank_observed'].numpy(),state['bank_origins'].numpy()).to(device)
    model.load_state_dict(state); model.eval().requires_grad_(False)
    if rt.state_hash(model) != case['training']['selected_hash']: raise ValueError('COST_SELECTED_MODEL_STATE_HASH')
    return model


def science_reference(pack, route, batch, forecaster, limit, device='cuda'):
    split = route['split']
    if batch not in (1,8) or len(pack['history'][split]) < batch: raise ValueError('COST_SCIENCE_REFERENCE_BATCH')
    thin = {'history': {split: pack['history'][split][:batch]}, 'origins': {split: pack['origins'][split][:batch]}}
    model = None
    meta = {'reference_kind': PROTOCOL['reference_kind'], 'batch': batch, 'selected_model_before_sha256': None,
        'selected_model_after_sha256': None, 'separate_selected_model': 'case' in route,
        'EVAL_target_arrays_opened': False, 'not_added_to_latency_statistics': True}
    try:
        if 'case' in route:
            model = load_selected_model(route['case'],device=device)
            meta['selected_model_before_sha256'] = rt.state_hash(model)
            parameters_before = rt.state_hash(dict(model.named_parameters()))
        if route['route'] == 'CORE':
            prediction = tr.predict(model,thin['history'][split],thin['origins'][split],forecaster,limit,batch=batch)
            meta['API'] = 'train.predict(normalized_history[:B],origins[:B],batch=B)'
        elif 'readout' in route:
            prediction = pr.evaluate_readout(thin,split,route['case'],model,route['readout'],rt,limit)
            meta['API'] = 'probes.evaluate_readout(thin_normalized_past_pack)'
            meta['original_API_recomputes_memory_buffers'] = True
        elif 'control' in route:
            prediction = pr.evaluate_raw_control(thin,split,route['control'],forecaster,rt,limit)
            meta['API'] = 'probes.evaluate_raw_control(thin_normalized_past_pack)'
        else: raise ValueError('COST_SCIENCE_REFERENCE_ROUTE')
        if model is not None:
            meta['selected_model_after_sha256'] = rt.state_hash(model)
            meta['registered_parameters_unchanged'] = parameters_before == rt.state_hash(dict(model.named_parameters()))
            meta['selected_full_state_unchanged'] = meta['selected_model_before_sha256'] == meta['selected_model_after_sha256']
            if not meta['registered_parameters_unchanged']: raise ValueError('COST_SCIENCE_REFERENCE_CHANGED_PARAMETERS')
            if route['route'] == 'CORE' and not meta['selected_full_state_unchanged']: raise ValueError('COST_CORE_SCIENCE_REFERENCE_CHANGED_STATE')
        if prediction.shape != (batch,48,pack['history'][split].shape[-1]) or not np.isfinite(prediction).all():
            raise ValueError('COST_MATCHED_SCIENCE_SHAPE_OR_NONFINITE')
        return prediction, meta
    finally:
        del model
        gc.collect()


def normalized_reinference(actual, bulk):
    if actual.shape != bulk.shape or not np.isfinite(actual).all() or not np.isfinite(bulk).all(): raise ValueError('COST_NORMALIZED_REINFERENCE_SHAPE_OR_NONFINITE')
    gap = np.abs(actual.astype(np.float64)-bulk.astype(np.float64)); allowed = 1e-5+2e-6*np.abs(bulk.astype(np.float64))
    return {'gate_pass': bool(np.all(gap <= allowed)), 'atol': 1e-5, 'rtol': 2e-6,
        'max_abs_gap': float(gap.max()), 'failed_cells': int(np.count_nonzero(gap > allowed)),
        'target_derived_MSE_bound_measured': False, 'bulk_science_metrics_changed': False}


def physical_parity(actual, reference_normalized, mean, std):
    if actual.shape != reference_normalized.shape or actual.ndim != 3 or actual.shape[1] != 48:
        raise ValueError('COST_PHYSICAL_PARITY_SHAPE')
    if not all(np.isfinite(a).all() for a in (actual,reference_normalized,mean,std)) or np.any(std <= 0):
        raise ValueError('COST_PHYSICAL_PARITY_NONFINITE_OR_INVALID_SCALE')
    reference = reference_normalized.astype(np.float32)*std.astype(np.float32)+mean.astype(np.float32)
    error = np.abs(actual-reference); allowed = .002+2e-5*np.abs(reference)
    failed = error > allowed
    cells = [{'window_index':int(b),'horizon_1based':int(h)+1,'channel_index':int(c),
        'actual_physical':float(actual[b,h,c]),'reference_physical':float(reference[b,h,c]),
        'mean':float(mean[c]),'std':float(std[c]),'absolute_gap':float(error[b,h,c]),'allowed_gap':float(allowed[b,h,c])}
        for b,h,c in np.argwhere(failed)]
    return {'parity_pass': bool(not failed.any()), 'physical_parity_atol': .002, 'physical_parity_rtol': 2e-5,
        'physical_max_abs_vs_science': float(error.max()),
        'normalized_max_abs_vs_science': float(np.abs((actual-mean)/std-reference_normalized).max()),
        'failed_cell_count': int(failed.sum()), 'failed_cells': cells}
