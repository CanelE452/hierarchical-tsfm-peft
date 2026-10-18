"""Independent FP64 replay of saved forecasts and matched experiment evidence."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import audit, contracts


def _independent_metrics(values):
    prediction, target, observed = (np.asarray(values[name]) for name in
        ('pred_normalized', 'target_normalized', 'observed'))
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape != observed.shape:
        raise ValueError('Aligned origin/horizon/channel arrays are required')
    if observed.dtype != np.bool_:
        raise ValueError('Explicit boolean observed mask required')
    counts = observed.sum((0, 1), dtype=np.int64)
    if not len(counts) or np.any(counts == 0):
        raise ValueError('Every registered channel requires an observed target; N0 is undefined')
    if not np.isfinite(prediction[observed]).all() or not np.isfinite(target[observed]).all():
        raise ValueError('Nonfinite observed prediction or target')
    errors = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    squared, absolute = errors * errors, np.abs(errors)
    sse, sae = squared.sum((0, 1), dtype=np.float64), absolute.sum((0, 1), dtype=np.float64)
    return dict(MSE=math.fsum((sse / counts).tolist()) / len(counts),
        MAE=math.fsum((sae / counts).tolist()) / len(counts), observed=int(counts.sum()),
        n_targets=int(counts.sum()), channel_sse=sse.tolist(), channel_sae=sae.tolist(),
        channel_counts=counts.tolist(), horizon_sse=squared.sum(0, dtype=np.float64).tolist(),
        horizon_sae=absolute.sum(0, dtype=np.float64).tolist(), horizon_counts=observed.sum(0, dtype=np.int64).tolist(),
        origin_sse=squared.sum(1, dtype=np.float64).tolist(), origin_sae=absolute.sum(1, dtype=np.float64).tolist(),
        origin_counts=observed.sum(1, dtype=np.int64).tolist())


def _check_receipt(metrics, receipt):
    aliases = dict(MSE=('MSE', 'mse'), MAE=('MAE', 'mae'), observed=('observed', 'n_targets', 'n_observed'))
    for name, keys in aliases.items():
        for key in keys:
            if key in receipt and (receipt[key] is None or not math.isclose(float(metrics[name]), float(receipt[key]),
                                                                           rel_tol=0., abs_tol=1e-12)):
                raise ValueError('Stored ' + name + ' differs from independent saved-array value')
    for name in ('channel_sse', 'channel_sae', 'channel_counts', 'horizon_sse', 'horizon_sae',
                 'horizon_counts', 'origin_sse', 'origin_sae', 'origin_counts'):
        if name in receipt and not np.allclose(np.asarray(metrics[name]), np.asarray(receipt[name]), rtol=0., atol=1e-12):
            raise ValueError('Stored sufficient statistic differs: ' + name)


def _array_fingerprint(values):
    values = np.ascontiguousarray(values)
    return dict(shape=list(values.shape), dtype=str(values.dtype), sha256=hashlib.sha256(values.tobytes()).hexdigest())


def _direction(value):
    return 'IMPROVED' if value < 0 else ('WORSE' if value > 0 else 'TIED')


def _compare(candidate, control, left, right, kind):
    names = ('dataset', 'seed', 'cap', 'C', 'K', 'L', 'H', 'data_sha', 'model_sha',
             'scaler_sha256', 'core_dtype', 'point_statistic', 'units', 'training_recipe')
    for key in names:
        if candidate.get(key) != control.get(key):
            raise ValueError('Matched comparison identity changed: ' + key)
    if kind == 'REVIN':
        if (candidate.get('family') != 'B2' or control.get('family') != 'B2'
                or candidate.get('revin') is not True or control.get('revin') is not False
                or candidate.get('initial_core_sha256') != control.get('initial_core_sha256')):
            raise ValueError('Matched comparison base initialization or intervention changed')
    elif kind == 'COMPOSITION':
        if (candidate.get('family') != 'B2' or control.get('family') != 'B0'
                or candidate.get('revin') != control.get('revin')
                or candidate.get('initial_ED_center_sha256') != control.get('initial_ED_center_sha256')):
            raise ValueError('Matched comparison composition initialization or normalization changed')
    else:
        raise ValueError('Unregistered matched comparison kind')
    for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'):
        if left[key].dtype != right[key].dtype or not np.array_equal(left[key], right[key]):
            raise ValueError('Matched comparison saved cohort differs: ' + key)
    identifier = f'{kind}_{candidate["dataset"]}_s{candidate["seed"]}_cap{candidate["cap"]}'
    mse_delta, mae_delta = candidate['MSE'] - control['MSE'], candidate['MAE'] - control['MAE']
    result = dict(comparison_id=identifier, kind=kind, dataset=candidate['dataset'], seed=candidate['seed'],
        cap=candidate['cap'], revin=candidate['revin'], candidate_run_id=candidate['run_id'],
        control_run_id=control['run_id'], candidate_MSE=candidate['MSE'], control_MSE=control['MSE'],
        candidate_MAE=candidate['MAE'], control_MAE=control['MAE'], MSE_delta=mse_delta, MAE_delta=mae_delta,
        MSE_percent_delta=100.*mse_delta/control['MSE'] if control['MSE'] else None,
        MAE_percent_delta=100.*mae_delta/control['MAE'] if control['MAE'] else None,
        MSE_direction=_direction(mse_delta), MAE_direction=_direction(mae_delta), match_status='MATCHED',
        n_origins=len(left['origins']), n_targets=candidate['n_targets'], selected_epoch_candidate=candidate.get('selected_epoch'),
        selected_epoch_control=control.get('selected_epoch'), C=candidate['C'], K=candidate['K'], L=candidate['L'], H=candidate['H'],
        limitation='B2 versus B0 changes temporal P capacity; this does not isolate equal-capacity architecture quality' if kind == 'COMPOSITION'
                   else 'Only outer original-channel RevIN changes; repeated development V, two seeds, capped training')
    sse = np.asarray(candidate['origin_sse']) - np.asarray(control['origin_sse'])
    sae = np.asarray(candidate['origin_sae']) - np.asarray(control['origin_sae'])
    counts = np.asarray(candidate['origin_counts'], np.int64)
    if not np.array_equal(counts, control['origin_counts']):
        raise ValueError('Matched comparison origin counts changed')
    reconstructed_mse = math.fsum((sse.sum(0) / counts.sum(0)).tolist()) / candidate['C']
    reconstructed_mae = math.fsum((sae.sum(0) / counts.sum(0)).tolist()) / candidate['C']
    if not math.isclose(reconstructed_mse, mse_delta, rel_tol=0., abs_tol=1e-12) or not math.isclose(reconstructed_mae, mae_delta, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Paired-origin sufficient statistics fail to reconstruct macro metric delta')
    origins = [dict(comparison_id=identifier, kind=kind, dataset=candidate['dataset'], seed=candidate['seed'],
        cap=candidate['cap'], origin=int(origin), channel_id=str(channel), SSE_diff=float(sse[i, c]),
        SAE_diff=float(sae[i, c]), count=int(counts[i, c]))
        for i, origin in enumerate(left['origins']) for c, channel in enumerate(left['channel_ids'])]
    return result, origins


def _check_decision(decision, searches):
    cap, cohort = decision.get('cap'), decision.get('cohort')
    ids = decision.get('matched_arms', [])
    if (decision.get('status') != 'SEALED' or decision.get('selection_split') != 'A_stop'
            or decision.get('V_used') is not False or cap not in (3, 6)
            or len(ids) != 4 or len(set(ids)) != 4):
        raise ValueError('Whole four-arm A_stop-only decision required')
    group = [searches[(name, cap)] for name in ids]
    identities = {(item['spec']['seed'], item['spec']['revin']) for item in group}
    if identities != {(seed, enabled) for seed in contracts.SEEDS for enabled in (False, True)}:
        raise ValueError('Whole two-seed OFF/ON decision required')
    minima, continuing = {False: [], True: []}, []
    for name, item in zip(ids, group):
        spec = item['spec']
        if spec['dataset'] != cohort or spec['family'] != 'B2' or item['status'] != 'COMPLETE':
            raise ValueError('Matched decision cohort or completion mismatch')
        scores = [(float(row['epoch']), float(row['astop_mse'])) for row in item['curves']]
        if scores[-1][0] != cap or any(not math.isfinite(value) for _, value in scores):
            raise ValueError('Matched decision cap incomplete or nonfinite')
        epoch, minimum = min(scores, key=lambda value: (value[1], value[0]))
        if item['selected_epoch'] != epoch:
            raise ValueError('A_stop selection does not use strict minimum and earliest tie')
        minima[spec['revin']].append(minimum)
        continuing.append(scores[-1][1] < min(value for _, value in scores[:-1]))
        saved = decision.get('selections', {}).get(name, {})
        if saved.get('epoch') != epoch or not math.isclose(float(saved.get('astop_mse', math.inf)), minimum, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Matched decision selected checkpoint differs from A_stop curve')
    means = {key: math.fsum(values) / 2 for key, values in minima.items()}
    selected_norm = means[True] < means[False]
    if decision.get('common_revin') is not selected_norm:
        raise ValueError('Common normalization differs from two-seed A_stop-only selection')
    if decision.get('astop_two_seed_means') != {'OFF': means[False], 'ON': means[True]}:
        raise ValueError('Matched decision two-seed A_stop means differ')
    if decision.get('still_improving_at_cap') is not any(continuing) or decision.get('extend_all_to6') is not (any(continuing) and cap == 3):
        raise ValueError('Matched decision extension must include the whole dataset group')
    return dict(cohort=cohort, cap=cap, common_revin=selected_norm, continuing=any(continuing),
                matched_arms=ids, selection='strict A_stop minimum; earlier tie; no V input')


def _recipe(spec):
    return dict(contracts.RECIPE, **{key: spec[key] for key in contracts.RECIPE},
                epochs=spec['epochs'], optimizer='AdamW', weight_decay=0., gradient_clip=1.,
                core_dtype=spec.get('core_dtype', 'float32'), device=spec['device'],
                seed=spec['seed'], init_kind=spec.get('init_kind', 'PCA'))


def _canonical_values(row, task, ctx):
    path = contracts.resolve_binding(row['prediction_binding'], ctx)
    values = audit._arrays(path)
    saved_names = sorted(values)
    required = ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids')
    if any(name not in values for name in required):
        raise ValueError('Saved prediction archive lacks required arrays')
    modules = ctx['modules']
    expected_origins = modules['data'].origins(task['split_ranges']['V_select'], task['L'], task['H'], task['H'])
    if values['origins'].dtype != np.int64 or not np.array_equal(values['origins'], expected_origins):
        raise ValueError('Saved V origins or stride differ from registered task')
    if not np.array_equal(values['channel_ids'], np.asarray(task['columns'])):
        raise ValueError('Saved channel IDs or order differ from registered task')
    if values['pred_normalized'].dtype != np.float32 or values['target_normalized'].dtype != np.float32:
        raise ValueError('Saved registered core prediction/target must be actual FP32')
    if values['pred_normalized'].shape != (len(expected_origins), task['H'], len(task['columns'])):
        raise ValueError('Saved forecast geometry differs from registered task')
    values = audit._canonical_arrays(values, task, modules)
    scaler = modules['fit'].scaler_fingerprint(values['mean'], values['std'])
    if row.get('scaler_sha256', scaler) != scaler:
        raise ValueError('Row scaler fingerprint differs from actual fixed A60 scaler')
    missing = [name for name in ('mean', 'std') if name not in saved_names]
    provenance = 'SAVED_AND_FIXED_A60_VERIFIED' if not missing else 'FIXED_A60_RECONSTRUCTED_NOT_SAVED'
    return values, dict(prediction_binding=row['prediction_binding'], saved_array_names=saved_names,
        reconstructed_scaler_arrays=missing, scaler_provenance=provenance, scaler_sha256=scaler,
        normalization_fit_end=task['split_ranges']['A_stop'][1],
        arrays={name: _array_fingerprint(array) for name, array in values.items()},
        exact_actual_target_mask_origin_channel=True, metric_accumulation_dtype='float64')


def _check_source(files, source_sha, ctx, training_source_sha=None):
    base = Path(ctx['modules']['fit'].__file__).resolve().parent
    for name, expected in files.items():
        path = base / name if '/' not in name else Path(ctx['repo']) / name
        if contracts.sha(path) != expected:
            raise ValueError('Bound scientific producer source changed: ' + name)
    payload = files if training_source_sha is None else dict(deployment_sources=files, training_source_sha=training_source_sha)
    if contracts.fingerprint(payload) != source_sha:
        raise ValueError('Bound scientific producer aggregate SHA differs')


def _check_search(item, task, ctx):
    descriptor = item.get('parent_checkpoint')
    binding = item.get('checkpoint_binding')
    path = contracts.resolve_binding(binding, ctx) if binding else Path(descriptor['path'])
    if descriptor and contracts.sha(path) != descriptor['sha256']:
        raise ValueError('Terminal search checkpoint SHA differs')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    spec, cap = item['spec'], item['cap']
    epochs = list(contracts.INITIAL_EPOCHS if cap == 3 else contracts.EXTENDED_EPOCHS)
    if spec['epochs'] != epochs or spec['family'] not in ('B2', 'B0') or spec['seed'] not in contracts.SEEDS:
        raise ValueError('Search schedule, family or seed is unregistered')
    if item['curves'] != checkpoint['extra']['curves']:
        raise ValueError('Search curve differs from bound terminal checkpoint')
    if checkpoint['extra']['completed_epochs'] != [0.] + epochs:
        raise ValueError('Search did not finish every registered checkpoint')
    minimum = min(item['curves'], key=lambda row: (row['astop_mse'], row['epoch']))
    if item['selected_epoch'] != minimum['epoch'] or checkpoint['extra']['best_epoch'] != minimum['epoch']:
        raise ValueError('Selected A_stop epoch differs from bound minimum')
    legal = ctx['modules']['data'].origins(task['split_ranges']['A_fit'], task['L'], task['H'], 1)
    targets = [0] + [math.ceil(epoch * len(legal)) for epoch in epochs]
    updates = sum(math.ceil((right-left)/16) for left, right in zip(targets, targets[1:]))
    if checkpoint['updates'] != updates or checkpoint['sampler']['exposures'] != cap * len(legal):
        raise ValueError('Search optimizer updates or exposures differ from registered schedule')
    if checkpoint['sampler']['origins_sha'] != hashlib.sha256(legal.tobytes()).hexdigest():
        raise ValueError('Bound sampler legal origins differ')
    for key in ('optimizer', 'torch_rng', 'cuda_rng'):
        if key not in checkpoint:
            raise ValueError('Bound continuation lacks complete training state: ' + key)
    if item.get('source_files'):
        _check_source(item['source_files'], checkpoint['contract']['source_sha'], ctx)
    elif descriptor:
        _check_source(descriptor['source_files'], checkpoint['contract']['source_sha'], ctx)
    if checkpoint['contract']['data_sha'] != task['data_sha256']:
        raise ValueError('Search data contract differs from actual task')
    return checkpoint


def _initial_state(row, inventory, ctx):
    name, cap = row['job_id'], row['cap']
    if row['family'] == 'B2' and row['revin'] is False:
        path = Path(inventory['controls'][name]['parent_checkpoint']['path']).parent / 'checkpoint_000000.pt'
    else:
        stage = 3 if row['family'] == 'B2' else cap
        path = Path(ctx['cache']) / 'runs' / name / f'cap{stage}' / 'search' / 'checkpoint_000000.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    if row['revin']:
        for key, expected in (('revin.affine_weight', 1.), ('revin.affine_bias', 0.)):
            if key not in state or not torch.equal(state[key], torch.full_like(state[key], expected)):
                raise ValueError('RevIN affine initialization changed')
        state = {key.removeprefix('base.'): value for key, value in state.items() if key.startswith('base.')}
    hashes = {key: _array_fingerprint(value.numpy()) for key, value in state.items()}
    return dict(initial_core_sha256=contracts.fingerprint(hashes),
        initial_ED_center_sha256=contracts.fingerprint({key: hashes[key] for key in ('E', 'D', 'center')}),
        initial_checkpoint_binding=contracts.binding(path, ctx))


def _legacy_manifest(row, ctx):
    root = Path(ctx['repo']) / 'research/informs_broad_validation_20261015_v1'
    names = {
        ('jena', 'RAW'): root / 'publication_derivatives/gzip_extra_aux_20261016_v1/files/item_1569.json.gz',
        ('jena', 'RAW_RIDGE'): root / 'publication_derivatives/gzip_extra_aux_20261016_v1/files/item_1573.json.gz',
        ('pems32', 'RAW'): root / ('priority_pems_subset_20261016_v1/predictions/'
            'postT_cpu_pems08_E1_SUBSET_RAW_k0_s92601_C32_priority_v1/V_select/PREDICTION_MANIFEST.json'),
    }
    path = names.get((row['dataset'], row['family']))
    if path is None or not path.is_file():
        return None, None
    manifest = json.loads(gzip.decompress(path.read_bytes())) if path.suffix == '.gz' else contracts.read(path)
    return manifest, contracts.binding(path, ctx)


def _pems_manifest_source(manifest, ctx):
    base = Path(ctx['repo']) / 'research/informs_broad_validation_20261015_v1'
    folder = base / 'priority_pems_subset_20261016_v1'
    plan_path, runner = folder / 'PLAN.json', folder / 'subset_minimum_runner.py'
    plan = contracts.read(plan_path)
    for name, expected in plan['dependencies'].items():
        if contracts.sha(base / name) != expected:
            raise ValueError('PeMS historical prediction dependency source differs: ' + name)
    source_sha = contracts.fingerprint(dict(runner=contracts.sha(runner), **plan['dependencies']))
    condition = next(item for item in plan['conditions'] if item['run_id'] == manifest['run_id'])
    if source_sha != manifest['source_sha'] or contracts.fingerprint(condition['spec']) != manifest['config_sha']:
        raise ValueError('PeMS historical prediction source/config aggregate differs')
    path = folder / 'runs' / manifest['run_id'] / 'SOURCE_CONTRACT.json'
    fitted = contracts.read(path)
    _check_source(fitted['files'], fitted['source_sha'], ctx)
    if fitted['data_sha'] != manifest['data_sha'] or fitted['model_sha'] != plan['foundation_tensor_sha256']:
        raise ValueError('PeMS historical prediction data/F snapshot differs')
    return dict(model_sha=fitted['model_sha'], F_snapshot_sha256=fitted['model_sha'],
                fit_source_sha256=fitted['source_sha'], fit_config_sha256=fitted['config_sha'],
                F_source_binding=contracts.binding(path, ctx)), dict(plan=contracts.binding(plan_path, ctx),
                runner=contracts.binding(runner, ctx), fit_source_contract=contracts.binding(path, ctx),
                prediction_source_schema='fingerprint(runner plus historical dependencies)')


def _bundle_metadata(row, task, search, ctx):
    if 'bundle_binding' not in row:
        return {}, {}
    path = contracts.resolve_binding(row['bundle_binding'], ctx)
    if path.suffix == '.npz':
        return {}, dict(bundle_binding=row['bundle_binding'], bundle_type='Ridge_saved_NPZ_parameters')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle['spec']['family'] != row['family']:
        raise ValueError('Historical or new bundle method family differs from accuracy identity')
    contract = bundle['contract']
    for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha'):
        if key in row and row[key] != contract[key]:
            raise ValueError('Row versus saved bundle binding differs: ' + key)
    if contract['data_sha'] != task['data_sha256']:
        raise ValueError('Saved model bundle data contract differs from registered task')
    mean, std, _ = ctx['modules']['data'].fit_statistics(task, task['split_ranges']['A_stop'][1])
    for key, expected in (('mean', mean), ('std', std)):
        if not np.array_equal(bundle['normalization'][key], expected):
            raise ValueError('Saved bundle normalization differs from actual fixed A60 scaler: ' + key)
    if contract['scaler_sha'] != ctx['modules']['fit'].scaler_fingerprint(mean, std):
        raise ValueError('Saved bundle scaler contract differs from actual fixed A60 scaler')
    model_hash = contracts.fingerprint({key: ctx['modules']['data'].array_sha256(value.numpy())
                                      for key, value in bundle['model'].items()})
    if model_hash != bundle['model_state_sha']:
        raise ValueError('Selected bundle tensor state SHA differs')
    metadata = dict(contract, selected_epoch=bundle['selected_search_epoch'],
        search_updates=bundle['search_summary']['updates'], refit_updates=bundle['refit_summary']['updates'],
        search_seconds=bundle['search_summary']['seconds'], refit_seconds=bundle['refit_summary']['seconds'],
        search_exposures=bundle['search_summary']['exposures'], refit_exposures=bundle['refit_summary']['exposures'],
        target_trained_params=bundle['refit_summary']['trainable_parameters'],
        model_state_sha256=model_hash, F_snapshot_sha256=contract['model_sha'],
        normalization_fit_end=bundle['normalization']['fit_end'])
    if row['family'] in ('B2', 'B0'):
        spec = bundle['spec']
        if (spec['family'] != row['family'] or spec['seed'] != row['seed'] or spec['dataset'] != row['dataset']
                or spec.get('revin', False) is not row['revin'] or spec['latent_k'] != row['K']):
            raise ValueError('Saved bundle registered method identity differs')
        if row['source_kind'] == 'NEW_TRAINED' and spec['run_id'] != row['run_id']:
            raise ValueError('New saved bundle run ID differs from accuracy identity')
        if bundle['search_summary'] != search['summary'] or bundle['selected_search_epoch'] != search['selected_epoch']:
            raise ValueError('Saved refit bundle selection differs from A_stop search')
        if bundle['normalization']['fit_end'] != task['split_ranges']['A_stop'][1]:
            raise ValueError('Selected bundle was not freshly fitted with the A60 normalization scope')
        metadata['training_recipe'] = _recipe(spec)
        prefix = 'base.' if row['revin'] else ''
        state = bundle['model']
        metadata.update(P_params=sum(value.numel() for key, value in state.items() if key.startswith(prefix+'front.')),
                        E_params=state[prefix+'E'].numel(), D_params=state[prefix+'D'].numel(),
                        RevIN_params=sum(value.numel() for key, value in state.items() if key.startswith('revin.')))
    return metadata, dict(bundle_binding=row['bundle_binding'], bundle_type='selected_fresh_A60_torch_bundle',
                          model_state_sha256=model_hash, source_contract=contract)


def _records(ctx, inventory):
    records = {(row['run_id'], 3): dict(row) for row in inventory['control_rows']}
    searches = {(name, 3): item for name, item in inventory['controls'].items()}
    provenance = {}
    for path in sorted((Path(ctx['report']) / 'runs').glob('*/cap*/SEARCH_RESULT.json')):
        item = contracts.read(path)
        name = path.parent.parent.name
        if item['status'] != 'COMPLETE':
            raise ValueError('Incomplete search cannot be published as complete')
        searches[(name, item['cap'])] = item
        provenance[(name, item['cap'])] = dict(search_result=contracts.binding(path, ctx))
    for path in sorted((Path(ctx['report']) / 'runs').glob('*/cap*/REFIT_RESULT.json')):
        result = contracts.read(path)
        if result['status'] != 'COMPLETE' or result['selection_sealed_before_V'] is not True or len(result['rows']) != 1:
            raise ValueError('Complete selected-refit and saved V result required')
        row = dict(result['rows'][0], bundle_binding=result['bundle_binding'], source_kind='NEW_TRAINED', status='COMPLETE',
                   training_status=result['fit_receipt']['training_status'])
        key = (row['run_id'], row['cap'])
        if key in records:
            if any(records[key].get(name) != row.get(name) for name in ('prediction_binding', 'bundle_binding', 'MSE', 'MAE')):
                raise ValueError('Duplicate accuracy identity differs')
        else:
            records[key] = row
        receipt = contracts.read(contracts.resolve_binding(result['selection_receipt_binding'], ctx))
        if receipt != result['fit_receipt']:
            raise ValueError('Refit receipt does not match its exact saved result')
        bundle_path = contracts.resolve_binding(row['bundle_binding'], ctx)
        if receipt['artifact_sha256'] != contracts.sha(bundle_path):
            raise ValueError('Fresh refit receipt artifact SHA differs')
        provenance.setdefault(key, {}).update(refit_result=contracts.binding(path, ctx),
            selection_receipt=result['selection_receipt_binding'], selection_decision=result['selection_decision_binding'])
    for desc in inventory['baselines']:
        row = dict(desc['row'], source_kind='HISTORICAL_VERIFIED', status='HISTORICAL_VERIFIED', training_status='HISTORICAL_SCOPE')
        records[(row['run_id'], 0)] = row
    return records, searches, provenance


def _load_rows(ctx, inventory, records, searches, final_caps, tasks, provenance):
    rows, arrays, proofs = [], {}, []
    for (name, cap), original in sorted(records.items()):
        row = dict(original)
        task = tasks[row['dataset']]
        values, proof = _canonical_values(row, task, ctx)
        metrics = _independent_metrics(values)
        _check_receipt(metrics, row)
        if row.get('data_sha', task['data_sha256']) != task['data_sha256']:
            raise ValueError('Saved row data fingerprint differs from actual task')
        search = searches.get((name, cap))
        metadata, bundle_proof = _bundle_metadata(row, task, search, ctx)
        row.update(metadata)
        row.update(metrics, job_id=name, run_id=name if cap == 0 else name+f'_cap{cap}', cap=cap,
            final=cap == 0 or final_caps[row['dataset']] == cap, units='global_A60_normalized',
            data_sha=task['data_sha256'], scaler_sha256=proof['scaler_sha256'], scaler_provenance=proof['scaler_provenance'],
            prediction_dtype=str(values['pred_normalized'].dtype), target_dtype=str(values['target_normalized'].dtype),
            mask_dtype=str(values['observed'].dtype), metric_accumulation_dtype='float64', independent_FP64='PASS',
            n_origins=len(values['origins']), evaluation_stride=task['H'], evaluation='V_select',
            core_dtype='float64' if row['family'] == 'RAW_RIDGE' else 'float32',
            point_statistic='quantile_0.5', V_independent_test=False,
            V_exposure='previously_reused_development')
        if cap:
            row.update(_initial_state(row, inventory, ctx))
            parent_seconds = float(searches[(name, 3)]['summary']['seconds']) if cap == 6 and row['family'] == 'B2' else 0.
            row['new_stage_search_seconds'] = row['search_seconds']
            row['inherited_parent3_search_seconds'] = parent_seconds
            row['search_seconds'] += parent_seconds
            scores = [(float(point['epoch']), float(point['astop_mse'])) for point in search['curves']]
            row['last_checkpoint_strictly_best'] = scores[-1][0] == cap and scores[-1][1] < min(value for _, value in scores[:-1])
            row['training_status'] = ('TRAINING_CAP_STILL_IMPROVING'
                if row['selected_epoch'] == cap and row['last_checkpoint_strictly_best']
                else 'SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED')
        else:
            manifest, manifest_binding = _legacy_manifest(row, ctx)
            if manifest:
                if manifest['archive_sha256'] != row['prediction_binding']['sha256'] or manifest['data_sha'] != task['data_sha256']:
                    raise ValueError('Historical prediction manifest archive or data SHA differs')
                if manifest['scaler_sha'] != proof['scaler_sha256']:
                    raise ValueError('Historical prediction manifest scaler differs')
                for key, fingerprint in manifest['arrays'].items():
                    if fingerprint != _array_fingerprint(values[key]):
                        raise ValueError('Historical saved-array fingerprint differs: ' + key)
                if 'metadata' in manifest:
                    _check_source(manifest['metadata']['source_files'], manifest['source_sha'], ctx,
                                  training_source_sha=manifest['metadata']['training_source_sha'])
                else:
                    source_metadata, source_proof = _pems_manifest_source(manifest, ctx)
                    row.update(source_metadata)
                    proof['historical_source_proof'] = source_proof
                _check_receipt(metrics, manifest['score'])
                if row['family'] == 'RAW_RIDGE' and manifest['metadata']['artifact_sha256'] != row['bundle_binding']['sha256']:
                    raise ValueError('Historical Ridge prediction and parameter artifact SHA differ')
                row.update(source_sha=manifest['source_sha'], config_sha=manifest['config_sha'],
                    prediction_model_label_sha256=manifest['model_sha'],
                    prediction_model_label_kind='legacy_path_or_artifact_fingerprint; not actual F tensor-state SHA',
                    prediction_manifest_binding=manifest_binding)
                proof['prediction_manifest'] = manifest_binding
                proof['prediction_compute_provenance'] = 'BOUND_HISTORICAL_MANIFEST'
            else:
                proof['prediction_compute_provenance'] = 'BOUND_BUNDLE_AND_SAVED_ARRAYS' if bundle_proof else 'UNRESOLVED_HISTORICAL_SOURCE'
        row.setdefault('model_sha', 'UNRESOLVED_HISTORICAL_F_STATE')
        row.setdefault('source_sha', 'UNRESOLVED_HISTORICAL_SOURCE')
        row.setdefault('config_sha', 'UNRESOLVED_HISTORICAL_CONFIG')
        proof.update(run_id=row['run_id'], job_id=name, dataset=row['dataset'], cap=cap, metrics=metrics,
                     bundle=bundle_proof, artifacts=provenance.get((name, cap), {}), source_sha=row['source_sha'],
                     config_sha=row['config_sha'], data_sha=row['data_sha'], model_sha=row['model_sha'])
        rows.append(row)
        arrays[row['run_id']] = values
        proofs.append(proof)
    return rows, arrays, proofs


def _csv(path, rows):
    if not rows:
        raise ValueError('Required public aggregate table has no rows: ' + Path(path).name)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, separators=(',', ':')) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        handle.write(output.getvalue())


def verify(ctx):
    inventory_path = Path(ctx['cache']) / 'LEGACY_INVENTORY.json'
    inventory = contracts.read(inventory_path)
    expected_controls = {contracts.job_id(cohort, False, seed) for cohort in contracts.COHORTS for seed in contracts.SEEDS}
    if set(inventory['controls']) != expected_controls or {row['run_id'] for row in inventory['control_rows']} != expected_controls:
        raise ValueError('Exactly four registered legacy OFF controls are required')
    if {(item['row']['dataset'], item['row']['family']) for item in inventory['baselines']} != {
            ('jena', 'RAW'), ('jena', 'RAW_RIDGE'), ('jena', 'LORA'), ('pems32', 'RAW'), ('pems32', 'DIRECT_G')}:
        raise ValueError('Only the five registered Raw and strong-baseline references are allowed')
    decision_path = Path(ctx['report']) / 'DECISIONS.json'
    document = contracts.read(decision_path)
    if document.get('status') != 'COMPLETE' or document.get('V_used_for_selection') is not False:
        raise ValueError('Complete A_stop-only decision document required')
    records, searches, provenance = _records(ctx, inventory)
    tasks = {name: ctx['task_loader'](ctx, ctx['modules'], name) for name in contracts.COHORTS}
    for task in tasks.values():
        if len(task['physical']) > task['split_ranges']['V_select'][1]:
            raise ValueError('Verification task materialized excluded T rows')
    checkpoints = {key: _check_search(item, tasks[item['spec']['dataset']], ctx) for key, item in searches.items()}
    checked_decisions = [_check_decision(item, searches) for item in document['decisions']]
    final_caps = {name: max(item['cap'] for item in document['decisions'] if item['cohort'] == name) for name in contracts.COHORTS}
    for cohort in contracts.COHORTS:
        caps = sorted(item['cap'] for item in document['decisions'] if item['cohort'] == cohort)
        initial = next(item for item in document['decisions'] if item['cohort'] == cohort and item['cap'] == 3)
        if caps != ([3, 6] if initial['extend_all_to6'] else [3]):
            raise ValueError('Dataset extension did not preserve all four arms at the registered caps')
    for decision in document['decisions']:
        bound_path = Path(ctx['report']) / 'decisions' / f'{decision["cohort"]}_cap{decision["cap"]}.json'
        if contracts.read(bound_path) != decision:
            raise ValueError('Global decision differs from its sealed per-cohort decision')
        for name in decision['matched_arms']:
            expected = decision['selections'][name]['checkpoint']
            contracts.resolve_binding(expected, ctx)
            item = searches[(name, decision['cap'])]
            actual = item.get('checkpoint_binding') or contracts.binding(item['parent_checkpoint']['path'], ctx)
            if expected != actual:
                raise ValueError('Decision selected search terminal checkpoint binding differs')
            if (name, decision['cap']) not in records:
                raise ValueError('Matched group lacks a fresh-A60 saved V arm')
            value = provenance.get((name, decision['cap']), {}).get('selection_decision')
            if value and value != contracts.binding(bound_path, ctx):
                raise ValueError('V receipt is bound to a different group decision')
    rows, arrays, proofs = _load_rows(ctx, inventory, records, searches, final_caps, tasks, provenance)
    comparisons, origin_differences = [], []
    for decision in document['decisions']:
        for seed in contracts.SEEDS:
            candidate = next(row for row in rows if row['dataset'] == decision['cohort'] and row['seed'] == seed
                and row['family'] == 'B2' and row['revin'] is True and row['cap'] == decision['cap'])
            control = next(row for row in rows if row['dataset'] == decision['cohort'] and row['seed'] == seed
                and row['family'] == 'B2' and row['revin'] is False and row['cap'] == decision['cap'])
            comparison, differences = _compare(candidate, control, arrays[candidate['run_id']], arrays[control['run_id']], 'REVIN')
            comparisons.append(comparison); origin_differences.extend(differences)
    final_jena = next(item for item in document['decisions'] if item['cohort'] == 'jena' and item['cap'] == final_caps['jena'])
    composition = [row for row in rows if row['family'] == 'B0']
    if len(composition) != 2 or {(row['seed'], row['revin'], row['cap']) for row in composition} != {
            (seed, final_jena['common_revin'], final_caps['jena']) for seed in contracts.SEEDS}:
        raise ValueError('Final composition requires exactly the two matched Jena B0 seeds')
    for control in composition:
        candidate = next(row for row in rows if row['family'] == 'B2' and row['dataset'] == 'jena'
            and row['seed'] == control['seed'] and row['revin'] == control['revin'] and row['cap'] == control['cap'])
        comparison, differences = _compare(candidate, control, arrays[candidate['run_id']], arrays[control['run_id']], 'COMPOSITION')
        comparisons.append(comparison); origin_differences.extend(differences)
    history = [dict(run_id=name+f'_cap{cap}', job_id=name, dataset=item['spec']['dataset'], family=item['spec']['family'],
        seed=item['spec']['seed'], revin=item['spec']['revin'], cap=cap, stage='search', epoch=point['epoch'],
        astop_mse=point['astop_mse'], updates=point['updates'], exposures=point['exposures'],
        selected_epoch=item['selected_epoch'], inherited_checkpoint=cap == 6 and point['epoch'] <= 3 and item['spec']['family'] == 'B2')
        for (name, cap), item in sorted(searches.items()) for point in item['curves']]
    stats = [dict(run_id=row['run_id'], dataset=row['dataset'], cap=row['cap'], channel_id=str(channel), horizon=h+1,
        SSE=row['horizon_sse'][h][c], SAE=row['horizon_sae'][h][c], count=row['horizon_counts'][h][c])
        for row in rows for h in range(row['H']) for c, channel in enumerate(arrays[row['run_id']]['channel_ids'])]
    matrix = [dict(run_id=row['run_id'], job_id=row['job_id'], dataset=row['dataset'], family=row['family'], seed=row['seed'],
        revin=row['revin'], cap=row['cap'], final=row['final'], status=row['status'], source_kind=row['source_kind'],
        training_status=row['training_status'], new_training=row['source_kind'] == 'NEW_TRAINED',
        scope='matched minimum' if row['family'] in ('B2', 'B0') else 'historical accuracy reference') for row in rows]
    plan = contracts.read(Path(ctx['report']) / 'CAMPAIGN_CONTRACT.json')
    for item in plan['jobs']:
        if item['family'] == 'B0' and item['revin'] != final_jena['common_revin']:
            matrix.append(dict(run_id=item['job_id'], job_id=item['job_id'], dataset='jena', family='B0', seed=item['seed'],
                revin=item['revin'], cap=final_caps['jena'], final=False, status='NOT_APPLICABLE', source_kind='CONDITIONAL_NOT_SELECTED',
                training_status='NOT_RUN', new_training=False, scope='only A_stop-selected common normalization gets its two B0 seeds'))
    numeric_columns = ('channel_sse', 'channel_sae', 'channel_counts', 'horizon_sse', 'horizon_sae',
                       'horizon_counts', 'origin_sse', 'origin_sae', 'origin_counts')
    public_rows = [{key: value for key, value in row.items() if key not in numeric_columns} for row in rows]
    tables = dict(RUNS=public_rows, COMPARISONS=comparisons, TRAINING_HISTORY=history,
                  ERROR_SUFFICIENT_STATS=stats, PAIR_ORIGIN_DIFFS=origin_differences, EXPERIMENT_MATRIX=matrix)
    for name, values in tables.items():
        _csv(Path(ctx['report']) / (name+'.csv'), values)
    contracts.atomic_json(Path(ctx['cache']) / 'VERIFIED_ROWS.json', rows)
    proof = dict(status='PASS', inputs=dict(inventory=contracts.binding(inventory_path, ctx), decisions=contracts.binding(decision_path, ctx)),
        source_sha256=contracts.sha(__file__), rows=proofs, decisions=checked_decisions,
        independent_metric='FP64 SSE/SAE per original channel, divided by its observed count, equal-channel macro average',
        arrays_in_public_artifacts=False, weights_in_public_artifacts=False, new_F_calls=0, new_training_updates=0,
        T_access=False, M_cal_selection=False, V_independent_test=False,
        A_stop_curve_verification='Bound terminal checkpoint curves, sampler/updates/schedule and selection; no new F execution',
        limits=['Two seeds', 'Repeated development V', 'Pretraining overlap UNKNOWN', 'Legacy T exposure retained',
                'Cap6 last-checkpoint improvement means cap or undertraining; no convergence claim'])
    contracts.atomic_json(Path(ctx['report']) / 'PREDICTION_PROOF.json', proof)
    validation = dict(status='PASS', checks=dict(saved_array_FP64=True, receipt_unrounded_values=True,
        actual_data_target_mask_origin_channel=True, fixed_A60_scaler=True, four_arm_Astop_only_decisions=True,
        matched_ReVIN_only=True, matched_composition=True, sufficient_statistics=True, source_bundle_SHA=True,
        T_access=False, V_independent_test=False), counts={name:len(values) for name, values in tables.items()},
        final_caps=final_caps, common_revin={'jena':final_jena['common_revin']},
        prediction_proof=contracts.binding(Path(ctx['report'])/'PREDICTION_PROOF.json',ctx),
        verified_tables={name:contracts.binding(Path(ctx['report'])/(name+'.csv'),ctx) for name in tables})
    contracts.atomic_json(Path(ctx['report']) / 'VALIDATION_REPORT.json', validation)
    return validation
