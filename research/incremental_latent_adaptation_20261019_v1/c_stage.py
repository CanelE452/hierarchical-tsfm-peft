"""M_select-only two-block selection and fresh full-M_cal refitting.

This module has no data, G, foundation or V loaders. The orchestrator admits a
complete C comparison group before calling it with already audited M_cal data.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.incremental_latent_adaptation_20261019_v1 import engine
from research.incremental_latent_adaptation_20261019_v1.blocks import LatentBlocks, METHODS, frozen_state_hash
from research.informs_broad_validation_20261015_v1.training import cpu_state
from research.raw_context_multiscale_20261018_v1.calibration import metric_statistics


def select_prefix(rows):
    if not rows or any(not np.isfinite(row['M_select_mse']) for row in rows):
        raise ValueError('Finite scores for every registered prefix are required')
    return min(rows, key=lambda row: (row['M_select_mse'], row['K']))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _foundation_counts(foundation):
    return {name: int(getattr(foundation, name, 0))
            for name in ('streams', 'physical_streams', 'calls')}


def _difference(before, after):
    return {key: after[key]-before[key] for key in before}


def _restore_checkpoint(model, path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    return saved


def _cache_stage1(model, data, foundation, directory, device, deadline):
    path = Path(directory)/'stage1_cache.npz'
    receipt_path = Path(directory)/'STAGE1_CACHE.json'
    origin_sha = engine._array_sha(data['origins'])
    state_sha = engine._state_sha(model)
    started = time.perf_counter()
    before = _foundation_counts(foundation)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive['model_sha'].item()) != state_sha
                    or str(archive['origins_sha'].item()) != origin_sha['sha256']):
                raise ValueError('Frozen stage1 cache contract changed')
            correction = archive['correction'].copy()
        reused = True
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if receipt['sha256'] != _sha(path):
            raise ValueError('Preserve original frozen stage1 cache generation receipt')
    else:
        pieces = []
        model.eval()
        with torch.no_grad():
            for lo in range(0, len(data['x']), engine.RECIPE['microbatch']):
                engine._check_deadline(deadline)
                x = torch.as_tensor(data['x'][lo:lo+engine.RECIPE['microbatch']], device=device)
                pieces.append(model.correction_for_prefix(x, model.block_sizes[0], foundation).cpu().numpy())
        correction = np.concatenate(pieces)
        temporary = path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, correction=correction, model_sha=state_sha,
                                origins_sha=origin_sha['sha256'])
        temporary.replace(path)
        reused = False
    if correction.shape != data['g'].shape or not np.isfinite(correction).all():
        raise ValueError('Finite aligned frozen stage1 predictions are required')
    if reused:
        receipt = dict(receipt, reused=True, reload_seconds=time.perf_counter()-started)
    else:
        receipt = dict(path=str(path), sha256=_sha(path), model_sha256=state_sha,
                       origins_identity=origin_sha, correction_identity=engine._array_sha(correction),
                       seconds=time.perf_counter()-started, reused=False,
                       foundation_counts=_difference(before, _foundation_counts(foundation)),
                       backward_rows=0, cache_scope='immutable selected stage1 only; never current stage2 F features')
        engine._atomic_json(receipt_path, receipt)
    return correction, receipt


def _prefix_scores(model, data, select, foundation, device, deadline):
    model.correction_only = False
    rows = []
    for k in (0, model.block_sizes[0], model.k):
        model.prefix_k = k
        prediction = engine.predict(model, data, select, foundation, device=device, deadline=deadline)
        stats = metric_statistics(prediction, data['y'][select], data['mask'][select])
        rows.append(dict(K=k, **{'M_select_'+key: stats[key]
                                for key in ('mse', 'mae', 'n_observed', 'active_channels')}))
    model.prefix_k = model.k
    return rows


def _accounting(parts, blocks_per_part):
    updates = [0, 0]
    exposures = [0, 0]
    for result, indices in zip(parts, blocks_per_part):
        for index in indices:
            updates[index] += result['updates']
            exposures[index] += result['exposures']
    return dict(optimizer_updates=sum(part['updates'] for part in parts),
                updates_per_block=updates, target_exposures_per_block=exposures,
                training_F_forward_rows=sum(part['training_F_forward_rows'] for part in parts),
                training_F_backward_rows=sum(part['training_F_backward_rows'] for part in parts),
                training_F_physical_rows=sum(part['training_F_physical_rows'] for part in parts),
                training_seconds=sum(part['training_seconds'] for part in parts),
                evaluation_seconds=sum(part['evaluation_seconds'] for part in parts),
                checkpoint_seconds=sum(part['checkpoint_seconds'] for part in parts))


def run(oldctx, data, dataset, seed, method, cap, foundation, directory, deadline=None, contract=None):
    if method not in METHODS or cap not in (3, 6):
        raise ValueError('Registered C methods and common completed B cap3/cap6 required')
    expected_k = {'jena': 12, 'pems32': 8}
    if dataset not in expected_k or data['s_model'].kmax != expected_k[dataset]:
        raise ValueError('Exact registered A60 S and dataset K required')
    if 'phi' in data:
        raise ValueError('C cannot reuse a changing latent-path feature cache')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = oldctx.get('device', 'cpu')
    fit = np.asarray(data['split']['fit_indices'], np.int64)
    select = np.asarray(data['split']['select_indices'], np.int64)
    endpoints = (0., .25, 1., 3.) if cap == 3 else (0., .25, 1., 3., 4., 5., 6.)
    c_contract = dict(contract or {}, C_method=method, dataset=dataset, seed=int(seed), cap=int(cap),
                      C_shared_front='original A60 S P and center fixed',
                      C_block_sizes=[expected_k[dataset]//2]*2,
                      C_selection_order='full-path checkpoint M_select minimum then K minimum at that checkpoint; exact K tie smaller',
                      blocks_source_sha256=_sha(Path(__file__).with_name('blocks.py')),
                      models_source_sha256=_sha(Path(__file__).with_name('models.py')),
                      c_stage_source_sha256=_sha(Path(__file__)))
    result = dict(schema='incremental_C_result_v1', status='PARTIAL_CHECKPOINTED', dataset=dataset,
                  seed=int(seed), method=method, cap=int(cap), endpoints=list(endpoints),
                  searches=[], refits=[], stage1_caches=[], frozen_stage1_checks={},
                  prefix_selection_rows=[], selected_prefix=None, state_path=None,
                  new_V_predictions_or_scores=0, contract=c_contract)
    started = time.perf_counter()
    before_counts = _foundation_counts(foundation)

    def finish():
        result['seconds'] = time.perf_counter()-started
        result['foundation_counts_all_work'] = _difference(before_counts, _foundation_counts(foundation))
        positions = [[0], [1]] if method == 'SEQ_BLOCKS' else [[0, 1]]
        result['search_accounting'] = _accounting(result['searches'], positions)
        result['refit_accounting'] = _accounting(result['refits'], positions)
        engine._atomic_json(directory/'RESULT.json', result)
        return result

    try:
        parent = data['s_model']
        model = LatentBlocks(parent, method).to(device)
        first = engine.run_search(model, data, fit, select, seed, foundation, directory/'search1',
                                  endpoints, deadline, contract=c_contract, device=device)
        result['searches'].append(first)
        if first['status'] != 'COMPLETE':
            return finish()
        _restore_checkpoint(model, first['selected_checkpoint'])
        selected_epochs = [first['selected_epoch']]
        if method == 'SEQ_BLOCKS':
            model.set_stage(1, correction_only=True)
            first_frozen = frozen_state_hash(model)
            first_checkpoint_sha = _sha(first['selected_checkpoint'])
            correction, receipt = _cache_stage1(model, data, foundation, directory/'search1', device, deadline)
            result['stage1_caches'].append(dict(phase='search', **receipt))
            stage2_data = dict(data, g=(data['g']+correction).astype(np.float32))
            second = engine.run_search(model, stage2_data, fit, select, seed, foundation,
                                       directory/'search2', endpoints, deadline,
                                       contract=dict(c_contract, frozen_stage1_sha=first_frozen,
                                                     stage1_checkpoint_sha=first_checkpoint_sha), device=device)
            result['searches'].append(second)
            if first_frozen != frozen_state_hash(model) or first_checkpoint_sha != _sha(first['selected_checkpoint']):
                raise ValueError('Stage2 changed frozen stage1 parameter/buffer/optimizer checkpoint')
            result['frozen_stage1_checks'] = dict(parameter_buffer_hash_unchanged=True,
                optimizer_checkpoint_hash_unchanged=True, frozen_state_sha256=first_frozen,
                checkpoint_sha256=first_checkpoint_sha, previous_optimizer_reused=False,
                active_block2_parameters_disjoint=True)
            if second['status'] != 'COMPLETE':
                return finish()
            _restore_checkpoint(model, second['selected_checkpoint'])
            selected_epochs.append(second['selected_epoch'])
            if first_frozen != frozen_state_hash(model):
                raise ValueError('Selected stage2 checkpoint changed stage1')
        result['selected_epoch'] = selected_epochs if method == 'SEQ_BLOCKS' else selected_epochs[0]
        result['prefix_selection_rows'] = _prefix_scores(model, data, select, foundation, device, deadline)
        result['selected_prefix'] = int(select_prefix(result['prefix_selection_rows'])['K'])
        selection = dict(dataset=dataset, seed=int(seed), method=method, selected_epochs=selected_epochs,
                         selected_prefix=result['selected_prefix'], prefix_rows=result['prefix_selection_rows'],
                         selection_split='M_select', checkpoint_rule=c_contract['C_selection_order'],
                         V_used=False, contract=c_contract)
        seal = directory/'SELECTION_SEAL.json'
        if seal.exists():
            if json.loads(seal.read_text(encoding='utf8')) != selection:
                raise ValueError('Preserve previously sealed C selection')
        else:
            engine._atomic_json(seal, selection)
        result['selection_seal_sha256'] = _sha(seal)
        fresh = LatentBlocks(parent, method).to(device)
        refit1 = engine.run_refit(fresh, data, seed, foundation, directory/'refit1', selected_epochs[0],
                                  deadline, contract=c_contract, device=device)
        result['refits'].append(refit1)
        if refit1['status'] != 'COMPLETE':
            return finish()
        if method == 'SEQ_BLOCKS':
            fresh.set_stage(1, correction_only=True)
            fresh_frozen = frozen_state_hash(fresh)
            checkpoint_sha = _sha(refit1['checkpoint_path'])
            correction, receipt = _cache_stage1(fresh, data, foundation, directory/'refit1', device, deadline)
            result['stage1_caches'].append(dict(phase='fresh_full_Mcal_refit', **receipt))
            refit_data = dict(data, g=(data['g']+correction).astype(np.float32))
            refit2 = engine.run_refit(fresh, refit_data, seed, foundation, directory/'refit2', selected_epochs[1],
                                      deadline, contract=dict(c_contract, frozen_stage1_sha=fresh_frozen,
                                                              stage1_checkpoint_sha=checkpoint_sha), device=device)
            result['refits'].append(refit2)
            if fresh_frozen != frozen_state_hash(fresh) or checkpoint_sha != _sha(refit1['checkpoint_path']):
                raise ValueError('Fresh stage2 refit changed freshly fitted frozen stage1')
            result['frozen_stage1_checks'].update(refit_parameter_buffer_hash_unchanged=True,
                refit_optimizer_checkpoint_hash_unchanged=True, refit_frozen_state_sha256=fresh_frozen)
            if refit2['status'] != 'COMPLETE':
                return finish()
        fresh.correction_only = False
        fresh.prefix_k = result['selected_prefix']
        path = directory/'model.pt'
        payload = dict(schema='incremental_C_final_model_v1', model=cpu_state(fresh), method=method,
                       dataset=dataset, seed=int(seed), block_sizes=list(fresh.block_sizes),
                       selected_prefix=result['selected_prefix'], selected_epochs=selected_epochs,
                       selection_seal_sha256=result['selection_seal_sha256'], contract=c_contract,
                       initialization='original A60 S P/center/E; independent zero residual D blocks')
        engine._atomic_torch(path, payload)
        result.update(status='COMPLETE', state_path=str(path), model_sha256=engine._state_sha(fresh),
                      model_file_sha256=_sha(path), fresh_full_Mcal_refit=True,
                      selected_M_select_mse=select_prefix(result['prefix_selection_rows'])['M_select_mse'])
        return finish()
    except engine._BudgetReached:
        return finish()
