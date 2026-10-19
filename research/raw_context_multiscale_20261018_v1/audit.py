"""Explicit, bounded data preparation and the exact completed parent models."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from . import common


S_HASHES = dict(
    jena={92601:'d4f0159f715a6d33f5573b653cbf9b2d238f5639d42b9503a9cbc30c25c76fba',
          92602:'4f8e1c87744d2f8a593af4730133ce3f2efd8ab1662cadf7678c410b2e50db7a'},
    pems32={92601:'6a82891560a63ebbba83b575a9f2f2cc80e5a6c0ef5df636e3686aaf7ac37d50',
            92602:'b12dd460ff3e052811b6e667e4bd070a5219d94dd436160c6850692d6a249b2f'})
G_HASHES = dict(jena='3fe6ad6803f4eeab6cb1a39e94eb32fed2edb91708db383fce18e9c676673ad9',
                pems32='8bcc85b1a7d99de25a31ee32e096f1bf6ed4e4076c35a3c41507136ef1c09d8f')


def old_s_root(ctx, dataset, seed):
    if dataset == 'jena':
        return ctx['repo']/'.cache/memory_budget_20261018_v1'/f'runs/B2_jena_K12_OFF_s{seed}/cap3'
    return ctx['repo']/'.cache/b2_revin_20261018_v1'/f'runs/B2_pems32_REVIN_OFF_s{seed}/cap6'


def _mirror(ctx, source, name, expected):
    if common.sha(source) != expected:
        raise ValueError('The exact registered parent changed: '+name)
    target = ctx['cache']/'parents'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    if common.sha(target) != expected:
        raise ValueError('The parent mirror differs from its immutable source')
    return target


def _bundle(ctx, task, source, name, expected):
    path = _mirror(ctx, source, name, expected)
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    norm, spec = bundle['normalization'], bundle['spec']
    end = task['split_ranges']['M_cal'][0]
    if norm['fit_end'] != end or norm.get('train_end', end) != end:
        raise ValueError('A parent must finish fitting before M_cal')
    if bundle['contract']['data_sha'] != task['data_sha256']:
        raise ValueError('Parent cohort differs from the registered data')
    mean, std, _ = ctx['modules']['data'].fit_statistics(task, end)
    if not np.array_equal(mean, norm['mean']) or not np.array_equal(std, norm['std']):
        raise ValueError('Parent uses a different A60 scaler')
    state_sha = ctx['modules']['training'].fingerprint({
        key: ctx['modules']['data'].array_sha256(value.numpy()) for key, value in bundle['model'].items()})
    if state_sha != bundle['model_state_sha']:
        raise ValueError('Serialized parent tensor identity changed')
    if spec['family'] not in ('B2', 'DIRECT_G', 'LORA'):
        raise ValueError('Calibration cannot reuse a previous calibration head')
    search = bundle['search_summary']
    if search['stop_origins'] != len(ctx['modules']['data'].origins(
            task['split_ranges']['A_stop'], task['L'], task['H'], task['H'])):
        raise ValueError('Parent checkpoint selection was not the registered A_stop cohort')
    route = dict(run_id=spec['run_id'], family=spec['family'],
                 artifact_cache_relative_path=path.relative_to(ctx['cache']).as_posix(),
                 artifact_sha256=expected, canonical_fit_end=end,
                 latent_k=int(spec.get('latent_k', len(task['columns']))))
    record = dict(dataset=task['dataset'], run_id=spec['run_id'], seed=int(spec['seed']),
                  family=spec['family'], parent_fit_end=end, normalization_fit_end=end,
                  parent_selection='A_stop', M_cal_targets_used_for_parent_fitting_or_selection=False,
                  source=common.binding(source, ctx), mirror=common.binding(path, ctx),
                  model_state_sha256=state_sha, scaler_sha256=bundle['contract']['scaler_sha'],
                  selected_epoch=bundle['selected_search_epoch'], route_spec=route)
    return record


def inspect_parents(ctx, tasks):
    records = []
    broad = ctx['asset']/'.cache/informs_broad_validation_20261015_v1'
    for dataset, task in tasks.items():
        for seed in common.SEEDS:
            record = _bundle(ctx, task, old_s_root(ctx,dataset,seed)/'selected_bundle.pt',
                             f'S_{dataset}_{seed}.pt', S_HASHES[dataset][seed])
            if record['route_spec']['latent_k'] != common.EXPECTED[dataset]['K']:
                raise ValueError('A different K cannot substitute for this parent')
            record.update(role='S', normalization='OFF', training_cap=3 if dataset=='jena' else 6)
            record['search_result'] = common.binding(old_s_root(ctx,dataset,seed)/'SEARCH_RESULT.json',ctx)
            records.append(record)
        relative = ('runs/primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3/selected_bundle.pt' if dataset=='jena'
                    else 'priority_pems_subset_20261016_v1/runs/postT_cpu_pems08_E1_SUBSET_DIRECT_G_k0_s92601_C32_priority_v1/selected_bundle.pt')
        record = _bundle(ctx,task,broad/relative,f'G_{dataset}_92601.pt',G_HASHES[dataset])
        record.update(role='G', actual_G_seed_count=1, reused_across_S_seeds=True)
        records.append(record)
        records.append(dict(dataset=dataset,role='Raw',family='RAW',seed=None,
                            parent_fit_end=task['split_ranges']['A_stop'][1],
                            M_cal_targets_used_for_parent_fitting_or_selection=False,
                            route_spec=dict(family='RAW',run_id='Raw_'+dataset,canonical_fit_end=task['split_ranges']['A_stop'][1])))
    jena = tasks['jena']
    lora = _bundle(ctx,jena,broad/'priority_valid_family_20261016_v1/runs/postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1/selected_bundle.pt',
                   'LoRA_jena_92601.pt','756ccf0e49a752948590cad156d89d30ef4f91adb1452d09cce9ffa8ec18c9ae')
    lora.update(role='LoRA', deployment='merged once after load/parity, before timing')
    records.append(lora)
    ridge = _mirror(ctx,broad/'ridge/primary_jena_E1_RAW_RIDGE_k0_s92601/artifact.npz',
                    'Ridge_jena_92601.npz','b7aea950cdaa4f93e49175910ebe6073975fbea3d6f399af5c1707bbce60c5e6')
    records.append(dict(dataset='jena',role='Ridge',family='RAW_RIDGE',seed=92601,
                        mirror=common.binding(ridge,ctx),parent_fit_end=12579,device='cpu',
                        route_spec=dict(family='RAW_RIDGE',run_id='Ridge_jena',
                            artifact_cache_relative_path=ridge.relative_to(ctx['cache']).as_posix(),
                            artifact_sha256=common.sha(ridge),canonical_fit_end=12579)))
    return records


def prepare(ctx):
    started = time.time()
    tasks = {name: common.load_task(ctx,name) for name in common.EXPECTED}
    parents = inspect_parents(ctx,tasks)
    value = dict(status='PASS',data=ctx['data_access'],parents=parents,
                 new_T_numeric_values_parsed=0,full_raw_source_hash_recomputed=False,
                 legacy_Jena_full_identity='INHERITED_NOT_RECOMPUTED',
                 prior_T_exposure_and_pretraining_overlap='Not ruled out; V remains development evaluation',
                 F_tensor_sha256=ctx['modules']['legacy'].F_SHA)
    common.sealed_json(ctx['report']/'DATA_AND_PARENT_BINDINGS.json',value)
    common.event(ctx,'audit','COMPLETE',wall_seconds=time.time()-started,parent_count=len(parents))
    return value, tasks
