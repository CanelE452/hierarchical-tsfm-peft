"""Exact legacy controls and input contracts, without fitting or F execution."""
from pathlib import Path
import numpy as np
import torch

from research.adapter_improvements_20261018_v1 import audit as old_audit
from . import contracts, revin

EXPECTED = {
 ('jena',92601): ('067a060b29a46b57de42cda664957bdf780233b6607eb3522d24a4633a036da2','ccd14e0ab659a05c60e2eb5eea76ab00ffef26f7aa911258b8512af972b6d63d','57b07a33ebc0a293eaad16a1b0bda0d2a6e2da82a2c4143e5ef4b0ea74cf1c36'),
 ('jena',92602): ('a40064d4be52a940e885536a3334ec56f3f286d1422506afb4f4c72fb3c07d8c','2cab692e3e9f760f2faf67e72f9c0fd37fdab349f7172f9b89b8c042019133d8','34cdaf29ff0438cc01f1274d1c7ed466252954a6f3ec319b595f3b0000e9a157'),
 ('pems32',92601): ('93eed90ab31c6c4bbe9006b95a2b10a074ba7da308fcc1869525d3f6c123943b','d08c720dc1e95013ad145be141b3204c001c7039cf6e2bfe3c941dfb63293243','fe8ac0e40abda6c631f701f83008e64c20ab3e5a2f7422bfcc45df3f497c403a'),
 ('pems32',92602): ('fc155c3e9f1a876cffbeb2a6a5d6481b92901f778480288c8d6ce16e28dd765b','7bdd7407b8efed5c2e5de080f0df331db167989326901ccafae860ec662e4513','ac0eadb2f6d809897d1a02a428b86af51261cfa530e03e2b1ae027645a59fd1c'),
}


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def _canonical_arrays(values, task, modules):
    mean,std,_=modules['data'].fit_statistics(task,task['split_ranges']['A_stop'][1])
    for key,expected in (('mean',mean),('std',std)):
        if key in values and not np.array_equal(values[key],expected):
            raise ValueError('Saved normalization differs from fixed A60 scaler')
        values[key]=expected
    windows=modules['training'].FastWindows(task,mean,std,'cpu')
    _,target,mask=windows.get(values['origins'])
    if not np.array_equal(target.numpy(),values['target_normalized']) or not np.array_equal(mask.numpy(),values['observed']):
        raise ValueError('Saved targets/masks differ from actual fixed data/A60 scaler')
    return values


def legacy_paths(ctx, cohort, seed):
    name = 'adapter_diagnostics_20261017_v2' if cohort == 'jena' else 'adapter_improvements_20261018_v1'
    run = f'init_PCA_s{seed}' if cohort == 'jena' else f'B2_PCA_PEMS32_s{seed}'
    cache = Path(ctx['repo'])/'.cache'/name
    public = Path(ctx['repo'])/'reports'/name/'runs'/run
    prediction = cache/'initialization_predictions'/f'PCA_{seed}.npz' if cohort == 'jena' else cache/'predictions'/run/'selected_A60_V_select.npz'
    return cache/'runs'/run/'search'/'checkpoint_000003.pt', cache/'runs'/run/'selected_bundle.pt', prediction, public


def legacy_control(ctx, cohort, seed, task):
    modules = ctx['modules']
    cp, bundle_path, prediction_path, public = legacy_paths(ctx,cohort,seed)
    for path, expected in zip((cp,bundle_path,prediction_path),EXPECTED[(cohort,seed)]):
        if contracts.sha(path) != expected:
            raise ValueError('Historical private artifact bytes changed: '+path.name)
    checkpoint = torch.load(cp,map_location='cpu',weights_only=False)
    bundle = torch.load(bundle_path,map_location='cpu',weights_only=False)
    source = contracts.read(public/'SOURCE_CONTRACT.json')
    receipt = contracts.read(public/'BASE_FIT_RECEIPT.json')
    spec = bundle['spec']
    expected_spec = dict(contracts.RECIPE, family='B2', init_kind='PCA', latent_k=6 if cohort=='jena' else 8,
                         seed=seed, device='cuda', dataset=cohort, epochs=list(contracts.INITIAL_EPOCHS))
    if any(spec.get(key)!=value for key,value in expected_spec.items()):
        raise ValueError('Legacy recipe mismatch')
    if receipt['status']!='COMPLETE_VALID' or receipt['artifact_sha256']!=contracts.sha(bundle_path):
        raise ValueError('Legacy selected bundle not bound to complete receipt')
    for name, expected in source['files'].items():
        path = Path(modules['fit'].__file__).parent/name if '/' not in name else Path(ctx['repo'])/name
        if contracts.sha(path)!=expected:
            raise ValueError('Legacy producer source changed: '+name)
    if modules['training'].fingerprint(source['files'])!=checkpoint['contract']['source_sha']:
        raise ValueError('Legacy source aggregate differs')
    if bundle['contract']['data_sha'] != task['data_sha256'] or checkpoint['contract']['data_sha'] != task['data_sha256']:
        raise ValueError('Legacy data contract mismatch')
    initialization=[]
    with revin.model_factory(ctx) as create:
        for fit_end, original_state, original_norm in (
            (task['split_ranges']['A_fit'][1],torch.load(cp.parent/'checkpoint_000000.pt',map_location='cpu',weights_only=False)['model'],None),
            (task['split_ranges']['A_stop'][1],None,bundle['normalization'])):
            off, no = create(dict(spec,device='cpu',revin=False),task,fit_end,lambda x:x)
            on, nn = create(dict(spec,device='cpu',revin=True),task,fit_end,lambda x:x)
            if set(off.state_dict())!=set(on.base.state_dict()) or any(not torch.equal(value,on.base.state_dict()[key]) for key,value in off.state_dict().items()):
                raise ValueError('ON/OFF base initialization differs')
            if original_state is not None and (set(off.state_dict())!=set(original_state) or any(not torch.equal(value,original_state[key]) for key,value in off.state_dict().items())):
                raise ValueError('OFF initialization differs from legacy CP0')
            for key in ('mean','std','basis','pca_center'):
                if not np.array_equal(no[key],nn[key]):
                    raise ValueError('ON/OFF global PCA/scaler differs')
                if original_norm is not None and not np.array_equal(no[key],original_norm[key]):
                    raise ValueError('Legacy A60 initialization/scaler differs')
            initialization.append(dict(fit_end=fit_end,base_state_sha256=modules['training'].fingerprint({key:modules['data'].array_sha256(value.numpy()) for key,value in off.state_dict().items()}),
                scaler_sha256=modules['fit'].scaler_fingerprint(no['mean'],no['std']),P_E_D_same=True,RevIN_extra_parameters=2*len(task['columns'])))
    values=_canonical_arrays(_arrays(prediction_path),task,modules)
    from research.adapter_diagnostics_20261017_v2.existing_jobs import score_arrays
    metrics=score_arrays(values['pred_normalized'],values['target_normalized'],values['observed'])
    spec_new=dict(spec,revin=False)
    result=dict(status='COMPLETE',spec=spec_new,curves=checkpoint['extra']['curves'],selected_epoch=bundle['selected_search_epoch'],
                summary=bundle['search_summary'],cap=3,source_kind='REUSED_VERIFIED',
                parent_checkpoint=dict(path=str(cp),sha256=contracts.sha(cp),contract=checkpoint['contract'],source_files=source['files'],spec=source['spec']))
    row=dict(metrics,run_id=contracts.job_id(cohort,False,seed),method='B2_REVIN_OFF',family='B2',revin=False,seed=seed,dataset=cohort,
             C=len(task['columns']),K=spec['latent_k'],L=task['L'],H=task['H'],cap=3,selected_epoch=bundle['selected_search_epoch'],
             device='cuda',source_kind='REUSED_VERIFIED',status='REUSED_VERIFIED',evaluation='V_select',fit='fresh_A60',
             prediction_binding=contracts.binding(prediction_path,ctx),bundle_binding=contracts.binding(bundle_path,ctx),
             source_sha=bundle['contract']['source_sha'],config_sha=bundle['contract']['config_sha'],data_sha=task['data_sha256'],model_sha=bundle['contract']['model_sha'],
             scaler_sha256=modules['fit'].scaler_fingerprint(values['mean'],values['std']),
             search_seconds=bundle['search_summary']['seconds'],refit_seconds=bundle['refit_summary']['seconds'],
             search_updates=receipt['search_updates'],refit_updates=receipt['refit_updates'],units='global_A60_normalized',
             core_dtype='float32',prediction_dtype=str(values['pred_normalized'].dtype),point_statistic='quantile_0.5',
             independent_FP64='PASS',new_training_updates=0)
    proof=dict(cohort=cohort,seed=seed,status='EXACT_OFF_REUSE_VERIFIED',initialization=initialization,
               checkpoint=contracts.binding(cp,ctx),bundle=contracts.binding(bundle_path,ctx),prediction=contracts.binding(prediction_path,ctx),
               source_contract=contracts.binding(public/'SOURCE_CONTRACT.json',ctx),receipt=contracts.binding(public/'BASE_FIT_RECEIPT.json',ctx),
               same_base_sources=True,same_checkpoint_schedule=True,complete_optimizer_sampler_CPU_CUDA_RNG=True,
               statistics_scope='fixed A_fit/A60 global PCA; outer window normalization is sole intervention')
    return result,row,proof


def baseline_paths(ctx):
    broad=Path(ctx['asset'])/'.cache/informs_broad_validation_20261015_v1'
    diag=Path(ctx['repo'])/'.cache/adapter_diagnostics_20261017_v2'
    lora='postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1'
    direct='postT_cpu_pems08_E1_SUBSET_DIRECT_G_k0_s92601_C32_priority_v1'
    return [
      dict(dataset='jena',method='Raw',family='RAW',prediction=broad/'predictions/primary_jena_E1_RAW_BOLT_SMALL_k0_s92601/V_select.npz',cache_root=broad),
      dict(dataset='jena',method='Ridge',family='RAW_RIDGE',prediction=broad/'predictions/primary_jena_E1_RAW_RIDGE_k0_s92601/V_select.npz',bundle=broad/'ridge/primary_jena_E1_RAW_RIDGE_k0_s92601/artifact.npz',cache_root=broad),
      dict(dataset='jena',method='LoRA_GPU_replica',family='LORA',prediction=diag/'lora_cost/E04_JENA_LORA_PARENT_V2_CUDA_V43.npz',bundle=broad/'priority_valid_family_20261016_v1/runs'/lora/'selected_bundle.pt',cache_root=broad/'priority_valid_family_20261016_v1'),
      dict(dataset='pems32',method='Raw',family='RAW',prediction=broad/'priority_pems_subset_20261016_v1/predictions/postT_cpu_pems08_E1_SUBSET_RAW_k0_s92601_C32_priority_v1/V_select.npz',cache_root=broad),
      dict(dataset='pems32',method='DirectG',family='DIRECT_G',prediction=broad/'priority_pems_subset_20261016_v1/predictions'/direct/'V_select.npz',bundle=broad/'priority_pems_subset_20261016_v1/runs'/direct/'selected_bundle.pt',cache_root=broad/'priority_pems_subset_20261016_v1'),
    ]


def run_audit(ctx):
    tasks={name:old_audit.load_task(ctx,name,ctx['modules']) for name in contracts.COHORTS}
    controls,rows,proofs={},[],[]
    for cohort in contracts.COHORTS:
        for seed in contracts.SEEDS:
            search,row,proof=legacy_control(ctx,cohort,seed,tasks[cohort])
            controls[row['run_id']]=search
            rows.append(row); proofs.append(proof)
    references=[]
    for desc in baseline_paths(ctx):
        values=_canonical_arrays(_arrays(desc['prediction']),tasks[desc['dataset']],ctx['modules'])
        from research.adapter_diagnostics_20261017_v2.existing_jobs import score_arrays
        metrics=score_arrays(values['pred_normalized'],values['target_normalized'],values['observed'])
        reference=_canonical_arrays(_arrays(contracts.resolve_binding(next(row for row in rows if row['dataset']==desc['dataset'])['prediction_binding'],ctx)),tasks[desc['dataset']],ctx['modules'])
        for key in ('target_normalized','observed','origins','channel_ids','mean','std'):
            if values[key].dtype!=reference[key].dtype or not np.array_equal(values[key],reference[key]):
                raise ValueError('Baseline cohort/scaler mismatch: '+desc['method']+'/'+key)
        task=tasks[desc['dataset']]
        row=dict(metrics,run_id='BASELINE_'+desc['dataset']+'_'+desc['method'],method=desc['method'],family=desc['family'],dataset=desc['dataset'],
                 seed=92601,revin=False,cap=0,C=len(task['columns']),K=len(task['columns']) if desc['family'] in ('RAW','LORA') else 0,
                 L=task['L'],H=task['H'],device='cuda' if desc['family']=='LORA' else 'cpu',source_kind='HISTORICAL_VERIFIED',
                 prediction_binding=contracts.binding(desc['prediction'],ctx),evaluation='V_select',independent_FP64='PASS',
                 units='global_A60_normalized',point_statistic='quantile_0.5',prediction_dtype=str(values['pred_normalized'].dtype),core_dtype='float32',
                 scaler_sha256=ctx['modules']['fit'].scaler_fingerprint(values['mean'],values['std']))
        if 'bundle' in desc: row['bundle_binding']=contracts.binding(desc['bundle'],ctx)
        references.append(dict(desc, row=row))
    inventory=dict(controls=controls,control_rows=rows,baselines=[{key:str(value) if isinstance(value,Path) else value for key,value in desc.items()} for desc in references])
    contracts.atomic_json(Path(ctx['cache'])/'LEGACY_INVENTORY.json',inventory)
    public=dict(status='PASS',controls=proofs,baseline_rows=[desc['row'] for desc in references],
                cohort_exact=True,new_F_calls=0,new_training_updates=0,new_T_rows=0,
                B0_reuse='No matched CUDA/common-normalization B0; conditional Jena two seed fresh fits required')
    contracts.sealed_json(Path(ctx['report'])/'REUSE_AUDIT.json',public)
    tasks_public={name:dict(C=len(task['columns']),K=6 if name=='jena' else 8,L=task['L'],H=task['H'],N=task['split_ranges']['T_final'][1],
                      columns=task['columns'],split_ranges=task['split_ranges'],materialized_prefix_rows=len(task['physical']),data_sha256=task['data_sha256'],
                      array_sha256=task.get('array_sha256',{})) for name,task in tasks.items()}
    contracts.sealed_json(Path(ctx['report'])/'DATA_CONTRACT.json',dict(tasks=tasks_public,T_access=False,M_cal_selection=False))
    return inventory,tasks
