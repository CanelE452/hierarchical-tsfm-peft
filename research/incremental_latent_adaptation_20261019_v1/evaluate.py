"""One final accuracy pass, only after all selection decisions are sealed."""
from pathlib import Path
import os
import time

import numpy as np
import torch

from . import common, costs, data_contract, deployment, engine, models, probes


def _parent_prediction(ctx, task, data, method, seed, origins):
    route = deployment.Route(dict(method=method, dataset=task['dataset'], seed=seed),
                             data, ctx=ctx, task=task, device='cuda')
    pieces = []
    try:
        for lo in range(0, len(origins), 4):
            physical, mask = costs.windows(task, origins[lo:lo+4])
            p = route.predict_physical(physical, mask)
            pieces.append(((p-data['mean'])/data['std']).astype(np.float32))
        return np.concatenate(pieces)
    finally:
        route.close()


def run(ctx, dataset):
    deployment.assert_v_sealed()
    manifest_path = common.CACHE/f'FINAL_MANIFEST_{dataset}.json'
    if manifest_path.exists():
        for row in common.read(manifest_path):
            if common.sha(row['prediction_path']) != row['prediction_sha256']:
                raise ValueError('Final sealed prediction bytes changed')
        return common.read(manifest_path)
    task = common.inherited.load_task(ctx, dataset)
    origins = costs.fixed_origins(task)
    entries = []
    base = data_contract.load(ctx, dataset, 92601)
    d = ctx['modules']['data']
    x = np.stack([d.history_window(task, int(o), base['mean'], base['std'])[0] for o in origins])
    targets = [d.target_window(task, int(o), base['mean'], base['std']) for o in origins]
    y, mask = np.stack([v[0] for v in targets]), np.stack([v[1] for v in targets])
    g = _parent_prediction(ctx, task, base, 'G', 92601, origins)
    raw = _parent_prediction(ctx, task, base, 'RAW', 92601, origins)

    def save(method, seed, prediction, data, *, artifact=None, alpha=None, search=None, refit=None, device='cuda', prefix=None):
        run_id = f'{dataset}_{method}_'+('shared' if seed is None else f's{seed}')
        if prefix is not None:
            run_id += f'_K{prefix}'
        directory = common.CACHE/'final'/run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'V.npz'
        model_sha = common.sha(artifact) if artifact else data['parent_g']['mirror']['sha256'] if method=='G' else None
        metadata = dict(campaign=common.NAME, run_id=run_id, dataset=dataset, method=method, seed=seed,
            split='V_select', data_sha256=task['data_sha256'],
            scaler_sha256=data['parent_g']['scaler_sha256'], parent_G_sha256=data['parent_g']['mirror']['sha256'],
            parent_S_sha256=data['parent_s']['mirror']['sha256'] if seed is not None else None,
            model_artifact_sha256=model_sha, C_decision_sha256=common.sha(common.REPORT/'C_DECISIONS.json'),
            F_tensor_sha256=common.read(common.REPORT/'PARENT_AND_DATA_CONTRACT.json')['F_tensor_sha256']
                if method in ('RAW','S','ALPHA_fitted','ALPHA_0','ALPHA_05','ALPHA_1','RESIDUAL_F','READOUT_ONLY','SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX') else None,
            alpha=alpha, prefix=prefix, selection_used_V=False, T_access=False,
            evaluation_role='Repeated development V; no new test', deployment_inverse_scaler_count=1,
            accuracy_path='fixed normalized G plus actual correction; parent predictions may be cached for accuracy only')
        destination=path
        if not path.exists():
            attempt=1
            while (directory/f'V.attempt{attempt:03}.npz').exists():
                attempt+=1
            destination=directory/f'V.attempt{attempt:03}.npz'
        receipt = common.inherited.save_prediction(destination, prediction=prediction,target=y,observed=mask,
            origins=origins,task=task,metadata=metadata)
        if destination!=path:
            os.replace(destination,path)
        common.atomic(directory/'V.binding.json',receipt)
        row = dict(run_id=run_id,dataset=dataset,method=method,seed=seed,prefix=prefix,
            prediction_path=str(path), prediction_sha256=receipt['prediction_sha256'],
            arrays=receipt['arrays'],score=receipt['score'],metadata=metadata,
            split='V_select', **{k:receipt['score'][k] for k in ('mse','mae','n_observed','n_origins')},
            device=device, alpha=alpha, selected_epoch=None if search is None else search['selected_epoch'],
            parent_G_sha256=metadata['parent_G_sha256'],parent_S_sha256=metadata['parent_S_sha256'],
            model_artifact_sha256=model_sha,scaler_sha256=metadata['scaler_sha256'])
        if method in (*common.METHODS,'SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
            row.update(initialization='fresh time MLP zero output' if method=='RESIDUAL_RAW_MLP' else 'original A60 S P/E/center; residual D zero',
                S_parent_F_training_information_reused=method!='RESIDUAL_RAW_MLP',
                F_in_correction_training=method in ('RESIDUAL_F','READOUT_ONLY','SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'),
                G_training_scope='A60; single fixed seed92601 parent shared across correction seeds',
                correction_training_scope='M_fit; selected on target-purged M_select; fresh full Mcal refit')
        if search:
            row.update(M_fit_mse=next(v['M_fit_mse'] for v in search['curves'] if v['epoch']==search['selected_epoch']),
                M_select_mse=search['selected_M_select_mse'],search_updates=search['updates'],
                search_exposures=search['exposures'], search_training_seconds=search['training_seconds'],
                search_evaluation_seconds=search['evaluation_seconds'],
                search_F_forward_rows=search['training_F_forward_rows'],search_F_backward_rows=search['training_F_backward_rows'],
                cap=search['curves'][-1]['epoch'])
        if refit:
            row.update(refit_updates=refit['updates'],refit_exposures=refit['exposures'],
                refit_training_seconds=refit['training_seconds'],refit_evaluation_seconds=refit['evaluation_seconds'],
                full_Mcal_mse=refit['curves'][-1]['M_cal_mse'],
                refit_F_forward_rows=refit['training_F_forward_rows'],refit_F_backward_rows=refit['training_F_backward_rows'])
        entries.append(row)
        common.event('FINAL_ACCURACY',run_id=run_id,mse=row['mse'],mae=row['mae'])

    save('G',None,g,base)
    save('RAW',None,raw,base)
    for kind in ('ZERO','MEAN','OWN','ALL'):
        if kind=='ZERO':
            continue
        artifact=common.CACHE/'probes'/dataset/(kind+'.npz')
        with np.load(artifact,allow_pickle=False) as source:
            probe={k:source[k].copy() for k in source.files}
        correction=probe['mean'] if kind=='MEAN' else probes.ridge_predict(dict(probe,kind=kind),probes.summary(x))
        save(kind,None,g+correction,base,artifact=artifact,device='cpu')
    for seed in common.SEEDS:
        data=data_contract.load(ctx,dataset,seed)
        s=_parent_prediction(ctx,task,data,'S',seed,origins)
        save('S',seed,s,data,artifact=common.inherited.resolve_binding(data['parent_s']['mirror'],ctx))
        alpha=common.read(common.CACHE/'probes'/dataset/f'alpha_s{seed}.json')
        for kind, coefficient in (('ALPHA_0',0.),('ALPHA_05',.5),('ALPHA_1',1.),('ALPHA_fitted',alpha['refit_alpha']['alpha'])):
            save(kind,seed,g+coefficient*(s-g),data,alpha=coefficient)
        for method in common.METHODS:
            directory=common.CACHE/'runs'/dataset/f'{method}_s{seed}'
            search,refit=(common.read(directory/phase/'RESULT.json') for phase in ('search','refit'))
            if refit['status']!='COMPLETE':
                raise ValueError('Final V requires complete fresh refit')
            model=models.make(data['s_model'],method,seed).to('cuda')
            state_path=Path(refit['state_path'])
            model.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=False)['model'])
            foundation=None
            pool=None
            if method in ('RESIDUAL_F','READOUT_ONLY'):
                from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
                pool,foundation=_foundation(ctx,ctx['modules'],task['H'],'cuda')
                foundation.row_microbatch,foundation.checkpoint_inputs=128,False
            pred=engine.predict(model,dict(x=x,g=g,y=y,mask=mask,origins=origins),
                np.arange(len(origins)),foundation,device='cuda',deadline=common.deadline(False))
            if foundation:
                foundation.assert_frozen_unchanged()
                pool.close()
            save(method,seed,pred,data,artifact=state_path,search=search,refit=refit,
                 prefix=task['K'] if method!='RESIDUAL_RAW_MLP' else None)
            entries[-1].update(models.parameter_counts(model), model_state_sha256=refit['final_state_sha256'])
            del model
            torch.cuda.empty_cache()
    decision=common.read(common.REPORT/'C_DECISIONS.json')['datasets'][dataset]
    if decision['status']=='COMPLETE_C_COMPARISON_GROUP':
        from . import blocks
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
        pool,foundation=_foundation(ctx,ctx['modules'],task['H'],'cuda')
        foundation.row_microbatch,foundation.checkpoint_inputs=128,False
        parity_rows=[]
        for row in decision['models']:
            seed,method=row['seed'],row['method']
            data=data_contract.load(ctx,dataset,seed)
            result=common.read(common.CACHE/'C'/dataset/f'{method}_s{seed}'/'RESULT.json')
            state_path=Path(result['state_path'])
            model=blocks.restore(data['s_model'],state_path,device='cuda')
            for prefix in (0,task['K']//2,task['K']):
                model.prefix_k=prefix
                active=dict(x=x,g=g,y=y,mask=mask,origins=origins)
                before=foundation.physical_calls
                prediction=engine.predict(model,active,np.arange(len(origins)),foundation,
                    device='cuda',deadline=common.deadline(False))
                if prefix==0 and foundation.physical_calls!=before:
                    raise AssertionError('Prefix0 invoked F')
                separate=[]
                with torch.no_grad():
                    for lo in range(0,len(x),4):
                        tx=torch.as_tensor(x[lo:lo+4],device='cuda')
                        corr=model.correction_for_prefix(tx,prefix,foundation,block_calls=True)
                        separate.append((torch.as_tensor(g[lo:lo+4],device='cuda')+corr).cpu().numpy())
                separate=np.concatenate(separate)
                np.testing.assert_allclose(prediction,separate,atol=1e-5,rtol=1e-4)
                parity_rows.append(dict(method=method,seed=seed,prefix=prefix,status='PASS',
                    n_origins=len(origins),max_absolute_normalized_difference=float(np.max(np.abs(prediction-separate))),
                    scope='full fixed V, concat one-F route versus separate per-block F calls',
                    zero_prefix_F_calls=0 if prefix==0 else None))
                save(method,seed,prediction,data,artifact=state_path,prefix=prefix)
                entries[-1].update(artifact_selected_prefix=result['selected_prefix'],
                    prefix_is_selected=prefix==result['selected_prefix'],selected_prefix=result['selected_prefix'], selected_epoch=result['selected_epoch'],
                    model_state_sha256=result['model_sha256'],
                    trainable_parameters=8*task['K']*len(task['columns']),
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    search_updates=result['search_accounting']['optimizer_updates'],
                    refit_updates=result['refit_accounting']['optimizer_updates'],
                    search_exposures=result['search_accounting']['target_exposures_per_block'],
                    refit_exposures=result['refit_accounting']['target_exposures_per_block'],
                    search_training_seconds=result['search_accounting']['training_seconds'],
                    refit_training_seconds=result['refit_accounting']['training_seconds'],
                    M_select_mse=next(v['M_select_mse'] for v in result['prefix_selection_rows'] if v['K']==prefix),
                    cap=result['cap'], fixed_P=True,
                    search_F_forward_rows=result['search_accounting']['training_F_forward_rows'],
                    search_F_backward_rows=result['search_accounting']['training_F_backward_rows'])
            del model
            torch.cuda.empty_cache()
        foundation.assert_frozen_unchanged()
        pool.close()
        common.atomic(common.REPORT/f'C_FULL_V_CALL_PARITY_{dataset}.json',dict(status='PASS',rows=parity_rows))
    common.seal(manifest_path,entries)
    return entries
