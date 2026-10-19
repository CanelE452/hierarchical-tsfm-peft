"""NEW M_cal-only data/parent contract; no V target or foundation reads."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from research.raw_context_multiscale_20261018_v1 import common as old
from research.raw_context_multiscale_20261018_v1 import data as old_data
from research.raw_context_multiscale_20261018_v1.calibration import (
    calibration_split, fit_alpha_diagnostic, metric_statistics,
)


NAME = 'incremental_latent_adaptation_20261019_v1'
PARENT_REPORT = 'reports/raw_context_multiscale_20261018_v1'
EXPECTED = {'jena': dict(C=21, K=12, L=512, H=48, end=14675),
            'pems32': dict(C=32, K=8, L=512, H=12, end=12499)}


def _source_task(ctx, dataset):
    """Decode only [0,M_cal_end), keeping later ZIP numeric payload opaque."""
    expected = old.read(ctx['repo']/'reports/b2_revin_20261018_v1/DATA_CONTRACT.json')['tasks'][dataset]
    campaign = ctx['modules']['campaign']
    primitive = old_data._module(campaign/old_data.PRIMITIVE, 'incremental_primitive')
    end = EXPECTED[dataset]['end']
    if dataset == 'jena':
        source = ctx['asset']/primitive.SOURCE_REL
        if (source.stat().st_size != primitive.SOURCE_BYTES
                or old.sha(ctx['asset']/primitive.CONTRACT_REL) != primitive.CONTRACT_SHA):
            raise ValueError('Jena source size/ancestry changed')
        metadata = old.read(campaign/'DATA_CONTRACT.json')['assets']['jena']['metadata']
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        values, members = {}, []
        with zipfile.ZipFile(source) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError('Duplicate Jena source member')
            for key in ('x', 'finite', 'times', 'mean', 'std', 'columns'):
                values[key], evidence = primitive.read_member(
                    archive, key+'.npy', end if key in ('x', 'finite', 'times') else None)
                members.append(evidence)
        physical = (values['x']*values['std']+values['mean']).astype(np.float32)
        mask, times, columns = values['finite'], values['times'], values['columns'].tolist()
        interval, period = 600, 144
    else:
        subset = old_data._module(campaign/old_data.SUBSET, 'incremental_subset')
        pems = old_data._module(campaign/old_data.PEMS, 'incremental_pems')
        plan = old.read(campaign/'priority_pems_subset_20261016_v1/PLAN.json')
        subset.require_sealed(plan)
        for relative, digest in plan['dependencies'].items():
            if old.sha(campaign/relative) != digest:
                raise ValueError('PeMS subset dependency changed: '+relative)
        metadata = copy.deepcopy(plan['parent_task_contract']['metadata'])
        source = ctx['modules']['old_cache']/metadata['source_reference']
        if source.stat().st_size != metadata['source_bytes']:
            raise ValueError('PeMS source size changed')
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        with zipfile.ZipFile(source) as archive:
            if archive.namelist() != ['data.npy']:
                raise ValueError('PeMS source members changed')
            info = archive.getinfo('data.npy')
            with archive.open(info) as stream:
                raw, evidence = pems.prefix_payload(stream, info.file_size, primitive, rows=end)
        selection = plan['column_selection']
        if selection['selected_ids'] != expected['columns']:
            raise ValueError('Registered PeMS metadata-only channel selection changed')
        physical = raw[:, selection['selected_indices'], 0].astype(np.float32)
        mask, times, columns = np.isfinite(physical), np.arange(end, dtype=np.int64)*300*1000000000, selection['selected_ids']
        members, interval, period = [evidence], 300, 288
        metadata.update(column_selection=copy.deepcopy(selection), actual_channels=32,
                        subset_full_T_identity_not_computed=True)
    if before != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError('Source changed during bounded decoding')
    if mask.dtype != np.bool_ or physical.shape != (end, EXPECTED[dataset]['C']):
        raise ValueError('Bounded prefix shape/mask changed')
    if list(columns) != expected['columns'] or not np.isfinite(physical[mask]).all():
        raise ValueError('Bounded prefix channels/observed values changed')
    if not np.all(np.diff(times) == interval*1000000000):
        raise ValueError('Bounded time grid changed')
    physical[~mask] = np.nan
    task = dict(name=dataset, dataset=dataset, physical=physical, observed=mask,
                times=times, columns=list(columns), L=expected['L'], H=expected['H'],
                K=EXPECTED[dataset]['K'], interval_seconds=interval, period=period,
                data_sha256=expected['data_sha256'], metadata=metadata,
                split_ranges={k: tuple(v) for k, v in expected['split_ranges'].items()},
                materialized_rows=end, original_total_rows=expected['N'])
    receipt = dict(dataset=dataset, decoded_bounds=[0, end], numeric_V_rows_decoded=0,
                   numeric_T_rows_decoded=0, full_raw_source_hash_recomputed=False,
                   inherited_full_data_identity=expected['data_sha256'],
                   inherited_identity_recomputed=False, members=members,
                   prefix_arrays={key: old.array_identity(task[key]) for key in ('physical','observed','times')},
                   physical_IO_scope='Bounded numeric prefix; ZIP buffering/Fortran opaque discards may read later bytes without numeric conversion')
    return task, receipt


def _parent(ctx, task, dataset, role, seed):
    prior = old.read(ctx['repo']/PARENT_REPORT/'DATA_AND_PARENT_BINDINGS.json')
    parents = prior['parents']
    parent = next(p for p in parents if p['dataset']==dataset and p['role']==role and p.get('seed')==seed)
    relative = parent['mirror']['reference'].removeprefix('cache://')
    path = ctx['cache']/relative
    if old.sha(path) != parent['mirror']['sha256']:
        raise ValueError('Exact A60 parent artifact changed')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    norm, spec = bundle['normalization'], bundle['spec']
    fit_end = task['split_ranges']['M_cal'][0]
    if (norm['fit_end'] != fit_end or norm['train_end'] != fit_end
            or bundle['contract']['data_sha'] != task['data_sha256']
            or bundle['contract']['scaler_sha'] != parent['scaler_sha256']):
        raise ValueError('Parent fitting scope, cohort or scaler changed')
    mean, std, counts = ctx['modules']['data'].fit_statistics(task, fit_end)
    if any(not np.array_equal(a,b) for a,b in ((mean,norm['mean']),(std,norm['std']), (counts,norm['observed_counts']))):
        raise ValueError('Parent A60 statistics differ from bounded physical source')
    state_sha = ctx['modules']['training'].fingerprint({
        k: ctx['modules']['data'].array_sha256(v.numpy()) for k,v in bundle['model'].items()})
    if state_sha != parent['model_state_sha256'] or state_sha != bundle['model_state_sha']:
        raise ValueError('Parent model tensors changed')
    expected_stop = ctx['modules']['data'].origins(task['split_ranges']['A_stop'], task['L'], task['H'], task['H'])
    if bundle['search_summary']['stop_origins'] != len(expected_stop):
        raise ValueError('Parent selection cohort differs from A_stop')
    if bundle['selected_search_epoch'] != parent['selected_epoch']:
        raise ValueError('Parent selected epoch changed')
    if parent['M_cal_targets_used_for_parent_fitting_or_selection']:
        raise ValueError('M_cal-exposed parent is ineligible')
    networks = ctx['modules']['fit'].networks
    with torch.random.fork_rng(devices=[]):
        if role == 'S':
            if spec['family'] != 'B2' or spec.get('revin', False):
                raise ValueError('Registered S is B2 RevIN OFF only')
            if bundle['contract']['model_sha'] != prior['F_tensor_sha256']:
                raise ValueError('S foundation tensor identity changed')
            if spec['latent_k'] != task['K'] or parent['training_cap'] != (3 if dataset=='jena' else 6):
                raise ValueError('Registered S K/cap differs')
            model = networks.ForecastAdapter('B2', task['L'], task['H'], len(task['columns']),
                                             task['K'], norm['pca_center'], norm['basis'])
        else:
            if spec['family'] != 'DIRECT_G' or seed != 92601:
                raise ValueError('Exactly one registered DirectG per dataset')
            model = networks.DirectG(task['L'], task['H'], hidden=bundle['model']['input.weight'].shape[0])
    model.load_state_dict(bundle['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, np.asarray(norm['mean']), np.asarray(norm['std']), copy.deepcopy(parent)


def _cache(ctx, task, dataset, role, seed):
    path = ctx['cache']/'predictions'/f'CAL_PARENT_{dataset}_{role}_{seed}'/'M_cal.npz'
    binding_path = path.with_suffix('.binding.json')
    binding = old.read(binding_path)
    previous = old.read(ctx['repo']/PARENT_REPORT/'COMPLEMENTARITY_FITS.json')['alpha_fits']
    if role == 'S':
        item = next(p for p in previous if p['dataset']==dataset and p['seed']==seed)
        original = item['parent_b_prediction_binding']
    else:
        item = next(p for p in previous if p['dataset']==dataset and p['base_role']==role)
        original = item['parent_g_prediction_binding']
    expected_reference = 'cache://'+path.relative_to(ctx['cache']).as_posix()
    if (original['reference'] != expected_reference or binding['sha256'] != original['sha256']
            or old.sha(path) != original['sha256']):
        raise ValueError('M_cal parent cache bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in ('pred_normalized','target_normalized','observed','origins','channel_ids')}
        metadata = json.loads(str(archive['metadata_json'].item()))
    if metadata != binding['metadata'] or metadata['split'] != 'M_cal' or metadata['T_access']:
        raise ValueError('M_cal parent cache metadata changed')
    if metadata['data_sha256'] != task['data_sha256'] or metadata['parent_fit_end'] != task['split_ranges']['M_cal'][0]:
        raise ValueError('M_cal parent cache cohort/scope differs')
    if {k: old.array_identity(v) for k,v in arrays.items()} != binding['arrays']:
        raise ValueError('M_cal cache array identities changed')
    if not np.array_equal(arrays['origins'], calibration_split(task)['origins']):
        raise ValueError('M_cal cache origin cohort changed')
    if arrays['observed'].dtype != np.bool_ or arrays['channel_ids'].tolist() != task['columns']:
        raise ValueError('M_cal cache observation/channel contract changed')
    receipt = dict(reference=path.relative_to(ctx['repo']).as_posix(), sha256=binding['sha256'],
                   binding_reference=binding_path.relative_to(ctx['repo']).as_posix(),
                   binding_sha256=old.sha(binding_path), arrays=binding['arrays'], metadata=metadata)
    return arrays, receipt


def load(ctx, dataset, seed):
    """Return CPU A60 parents and aligned full-M_cal arrays, never V values."""
    if dataset not in EXPECTED or seed not in (92601,92602):
        raise ValueError('Only registered dataset/seed pairs admitted')
    memo = ctx.setdefault('_incremental_data', {})
    if dataset not in memo:
        task, access = _source_task(ctx, dataset)
        g_model, mean, std, parent_g = _parent(ctx,task,dataset,'G',92601)
        g, gb = _cache(ctx,task,dataset,'G',92601)
        raw, rb = _cache(ctx,task,dataset,'Raw',None)
        origins = calibration_split(task)['origins']
        data = ctx['modules']['data']
        x = np.stack([data.history_window(task,int(o),mean,std)[0] for o in origins])
        targets = [data.target_window(task,int(o),mean,std) for o in origins]
        y = np.stack([v[0] for v in targets]); mask = np.stack([v[1] for v in targets])
        for cache in (g,raw):
            if not np.array_equal(cache['observed'],mask) or not np.array_equal(cache['target_normalized'][mask],y[mask]):
                raise ValueError('M_cal parent target/mask differs from bounded source')
        memo[dataset] = dict(task=task, g_model=g_model, mean=mean, std=std, parent_g=parent_g,
                             origins=origins,x=x,g=g['pred_normalized'],y=y,mask=mask,raw=raw['pred_normalized'],
                             cache_bindings={'G':gb,'Raw':rb},data_access=access,split=calibration_split(task))
    result = dict(memo[dataset])
    task = result['task']
    s_model, mean, std, parent_s = _parent(ctx,task,dataset,'S',seed)
    if not np.array_equal(mean,result['mean']) or not np.array_equal(std,result['std']):
        raise ValueError('S/G A60 scalers differ')
    s, sb = _cache(ctx,task,dataset,'S',seed)
    if (not np.array_equal(s['observed'],result['mask'])
            or not np.array_equal(s['target_normalized'][result['mask']],result['y'][result['mask']])):
        raise ValueError('S/G target or observation set differs')
    for receipt,parent in ((result['cache_bindings']['G'],result['parent_g']),(sb,parent_s)):
        if (receipt['metadata']['scaler_sha256'] != parent['scaler_sha256']
                or receipt['metadata']['route_spec'] != parent['route_spec']):
            raise ValueError('Prediction cache bound to a different parent/scaler')
    result.update(s_model=s_model,s=s['pred_normalized'],parent_s=parent_s,
                  cache_bindings=dict(result['cache_bindings'],S=sb))
    return result


def audit(ctx):
    """Independent CPU contract audit and original alpha reconstruction."""
    prior_fits = old.read(ctx['repo']/PARENT_REPORT/'COMPLEMENTARITY_FITS.json')['alpha_fits']
    datasets, parents, caches, alpha = {}, [], {}, []
    for dataset in EXPECTED:
        for seed in (92601,92602):
            value = load(ctx,dataset,seed)
            task, split = value['task'], value['split']
            diagnostic = fit_alpha_diagnostic(value['g'],value['s'],value['y'],value['mask'],
                                               split['fit_indices'],split['select_indices'])
            previous = next(p['diagnostic'] for p in prior_fits
                            if p['dataset']==dataset and p['seed']==seed and p['base_role']=='G')
            for group in ('fit_alpha','refit_alpha'):
                for key in ('alpha','numerator','denominator','objective_g','objective_b','objective_alpha'):
                    if not np.isclose(diagnostic[group][key],previous[group][key],atol=1e-13,rtol=1e-12):
                        raise ValueError('Original G+S coefficient reconstruction mismatch')
            for row, previous_row in zip(diagnostic['rows'],previous['rows']):
                for key in ('M_fit_mse','M_fit_mae','M_select_mse','M_select_mae'):
                    if not np.isclose(row[key],previous_row[key],atol=1e-13,rtol=1e-12):
                        raise ValueError('Original G+S selected score reconstruction mismatch')
            alpha.append(dict(dataset=dataset,seed=seed,diagnostic=diagnostic,
                              previous_public_reference=PARENT_REPORT+'/COMPLEMENTARITY_FITS.json',
                              full_precision_reconstruction='PASS',new_V_values_read=False))
            parents.append(value['parent_s'])
            if seed == 92601:
                parents.append(value['parent_g'])
                datasets[dataset] = dict(C=len(task['columns']),K=task['K'],L=task['L'],H=task['H'],
                    interval_seconds=task['interval_seconds'],columns=task['columns'],split_ranges=task['split_ranges'],
                    Mcal_origins=old.array_identity(split['origins']),Mfit_indices=old.array_identity(split['fit_indices']),
                    Mselect_indices=old.array_identity(split['select_indices']),purged_indices=old.array_identity(split['discarded_indices']),
                    n_Mcal=len(split['origins']),n_Mfit=len(split['fit_indices']),n_Mselect=len(split['select_indices']),
                    n_purged=len(split['discarded_indices']),mcut=split['mcut'],data_access=value['data_access'],
                    normalized_history_identity=old.array_identity(value['x']),
                    observed_target_counts=value['mask'].sum((0,1),dtype=np.int64).tolist(),
                    A60_fit_end=task['split_ranges']['M_cal'][0],A_stop_new_correction_selector=False,
                    G_unique_fitted_parent_count=1,G_seed=92601,G_reused_across_correction_seeds=True)
            for role, receipt in value['cache_bindings'].items():
                key = f'{dataset}_{role}_{seed if role=="S" else 92601 if role=="G" else "None"}'
                caches[key] = receipt
    return dict(schema='NEW_INCREMENTAL_PARENT_AND_DATA_CONTRACT_v1',status='PASS',datasets=datasets,
                parents=parents,parent_cache_count=len(caches),parent_caches=caches,alpha_reconstruction=alpha,
                F_tensor_sha256=old.read(ctx['repo']/PARENT_REPORT/'DATA_AND_PARENT_BINDINGS.json')['F_tensor_sha256'],
                input_contract='Historical window only; fixed A60 observed mean/std; within-window causal fill initialized by A60 mean',
                target_contract='Original bool mask, fixed A60 normalized Y; no residual restandardization',
                selection_contract='M_fit training; target-purged M_select selection; fresh full-Mcal refit after seal; V only after C decisions',
                numeric_V_values_read=0,numeric_T_values_read=0,full_raw_source_hash_recomputed=False,
                V_status='Repeated development evaluation; no independent test claim',
                unresolved=['Legacy Jena inverse full-TRAIN normalization ancestry/FP32 rounding',
                            'Historical protected-data exposure and foundation pretraining overlap not ruled out'],
                audited_sources={str(p.relative_to(ctx['repo'])).replace('\\','/'):old.sha(p) for p in (
                    Path(__file__),ctx['repo']/PARENT_REPORT/'DATA_AND_PARENT_BINDINGS.json',
                    ctx['repo']/PARENT_REPORT/'PARENT_IDENTITIES.csv',ctx['repo']/PARENT_REPORT/'COMPLEMENTARITY_FITS.json',
                    ctx['repo']/'research/raw_context_multiscale_20261018_v1/calibration.py',
                    ctx['repo']/'research/raw_context_multiscale_20261018_v1/data.py',
                    ctx['repo']/'research/informs_broad_validation_20261015_v1/data_contract.py')})
