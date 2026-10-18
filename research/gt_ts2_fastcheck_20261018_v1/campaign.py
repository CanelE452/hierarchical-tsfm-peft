"""Registered TRAIN smoke, frozen lookbacks, and eight fixed adaptation arms."""
from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import common as c
from . import data as d
from . import execution as e
from .analysis import make_schedule
from .training import train_arm


CORE_NAMES=('common.py','data.py','model.py','parents.py','feature_cache.py','analysis.py','training.py','execution.py','campaign.py')


def core_identity():
    return {name:c.file_sha(c.CODE/name) for name in CORE_NAMES}


def _public(value):
    if isinstance(value,Path):
        if not value.resolve().is_relative_to(c.CACHE.resolve()):
            raise ValueError('Training artifact escaped the private campaign cache')
        return 'cache://'+value.relative_to(c.CACHE).as_posix()
    if isinstance(value,dict):return {key:_public(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_public(item) for item in value]
    return value


def _inputs(ctx,spec,route):
    data=ctx['data']
    x,y,m=d.windows(data,data['train_origins'],spec['L'])
    diagnostic_x,diagnostic_y,diagnostic_m=d.windows(data,data['diagnostic_origins'],spec['L'])
    _,input_mask=d.physical_windows(data,data['diagnostic_origins'],spec['L'])
    diagnostic=dict(x=diagnostic_x,targets=diagnostic_y,mask=diagnostic_m,input_mask=input_mask)
    features=None
    if 'feature_record' in route:
        features=e._features(route['feature_record'],'train')
        indices=np.searchsorted(data['train_origins'],data['diagnostic_origins'])
        if not np.array_equal(data['train_origins'][indices],data['diagnostic_origins']):
            raise ValueError('TRAIN diagnostic origins are not in the same legal pool')
        diagnostic['features']=features[indices]
    return x,y,m,features,diagnostic


def gradient_through_future(route,x,features=None):
    """Probe only the F future, so decoder history cannot hide a detached F path."""
    model=route['gt_route'];model.zero_grad(set_to_none=True)
    xt=torch.as_tensor(x[:1],device='cuda')
    latent=model.gt.encode(xt)
    if features is not None:
        ft=torch.as_tensor(features[:1],device='cuda')
        standardized=(ft-model.feature_mean)/model.feature_std
        latent=latent+model.W(standardized).reshape_as(latent)
    latent.retain_grad()
    future=model.foundation.forecast(latent,12)
    future.square().mean().backward()
    groups={'E':list(model.gt.encoder.parameters()),'product':list(model.gt.product.parameters())}
    if model.W is not None:groups['W']=list(model.W.parameters())
    result={name:float(sum(parameter.grad.abs().sum() for parameter in values if parameter.grad is not None))
            for name,values in groups.items()}
    result['F_input']=float(latent.grad.abs().sum()) if latent.grad is not None else 0.
    if any(not np.isfinite(value) or value<=0 for value in result.values()):
        raise ValueError('Actual forecast-future gradient is disconnected/nonfinite: '+str(result))
    if not model.foundation.verify_unchanged()['parameter_gradients_absent']:
        raise ValueError('Frozen F accumulated parameter gradients')
    model.zero_grad(set_to_none=True)
    return dict(status='PASS',gradient_l1=result,objective='actual F future squared mean, excluding native D history')


def smoke():
    e.cuda_setup();ctx=e.context();f=e.foundation(ctx);records=[]
    sources=core_identity()
    for spec in [s for s in e.candidate_specs() if s['phase']=='B' and s['seed']==92601]:
        route=e.build_route(spec,ctx,f,online=False)
        x,y,m,features,diagnostic=_inputs(ctx,spec,route)
        probe=gradient_through_future(route,x,features)
        result=train_arm(route['gt_route'],x,y,m,make_schedule(92601,len(x),20),
            feature_cache=features,diagnostics=diagnostic,output_dir=c.CACHE/'smoke'/spec['run_id'],
            run_id='SMOKE_'+spec['run_id'],seed=92601,microbatch=1,
            callback=lambda update,details:c.progress('TRAIN_smoke',run_id=spec['run_id'],update=update,total=20))
        record=dict(spec=spec,discarded=True,gradient_probe=probe,
            mean_update_seconds=float(np.mean([row['update_wall_seconds'] for row in result['curves']])),
            training=_public(result))
        records.append(record)
        c.seal_json(c.REPORT/'smoke'/(spec['run_id']+'.json'),record)
        del route,x,y,m,features,diagnostic,result;gc.collect();torch.cuda.empty_cache()
    budget=c.current_budget()
    estimate=1.25*max(row['mean_update_seconds'] for row in records)
    decision=c.choose_updates(estimate,budget['GPU_used_seconds'],budget['elapsed_seconds'],180.)
    decision.update(sealed_unix=time.time(),status_scope='all four arms and both seeds uniformly',
        TRAIN_smoke_records=[{'run_id':row['spec']['run_id'],'mean_update_seconds':row['mean_update_seconds']} for row in records],
        budget_at_decision=budget,core_source_sha256=sources,
        smoke_models_discarded=True,final_checkpoint='fixed final update; no development VAL selection',
        prior_new_Phase_A_scores=0,F_state_invariant=f.verify_unchanged())
    if sources!=core_identity():raise ValueError('Scientific core changed while TRAIN profiling ran')
    c.seal_json(c.REPORT/'UPDATE_DECISION.json',decision)
    return decision


def phase_a():
    e.cuda_setup();ctx=e.context();f=e.foundation(ctx)
    if not (c.REPORT/'PARENT_AND_GRADIENT_VALIDATION.json').exists():
        raise ValueError('Historical parents and F gradient must be verified first')
    if not (c.REPORT/'UPDATE_DECISION.json').exists():
        raise ValueError('Candidate/update timing rule must be sealed before new Phase A scores')
    completed=[]
    for spec in [s for s in e.candidate_specs() if s['phase']=='A']:
        path=c.REPORT/'receipts'/(spec['run_id']+'.json')
        if path.exists():
            old=c.read_json(path)
            artifact=c.CACHE/'predictions'/(spec['run_id']+'.npz')
            if old['status']!='COMPLETE' or c.file_sha(artifact)!=old['artifact_sha256']:
                raise ValueError('Existing frozen result cannot be silently replaced')
            completed.append(spec['run_id']);continue
        c.progress('Phase_A',run_id=spec['run_id'],current=len(completed),total=13)
        x,y,m=d.windows(ctx['data'],ctx['data']['val_origins'],spec['L'])
        route=e.build_route(spec,ctx,f)
        before=f.calls
        started=time.perf_counter()
        prediction=e.predict(route,f,x,batch=4 if spec['mode'] in ('ts2','ensemble') else 8)
        elapsed=time.perf_counter()-started
        parameters=sum(parameter.numel() for parameter in f.model.parameters())
        adapter_parameters=0
        if 'gt_route' in route:adapter_parameters+=sum(parameter.numel() for parameter in route['gt_route'].parameters())
        if 'ts2' in route:parameters+=sum(parameter.numel() for parameter in route['ts2'].net.parameters())
        if 'decoder' in route:adapter_parameters+=sum(parameter.numel() for parameter in route['decoder'].parameters())
        e.prediction_artifact(spec,ctx,x,prediction,y,m,
            inference_wall_seconds=elapsed,F_forward_calls=f.calls-before,
            requested_F_context=spec['L']//4 if spec['mode']=='gt' else spec['L'],
            actual_F_context=f.actual_context_length,F_streams_per_origin=21 if spec['mode']=='raw' else 42 if spec['mode']=='gt' else 320 if spec['mode']=='ts2' else [42,320],
            resident_parameters=parameters+adapter_parameters,trainable_parameters=0,
            model_status='FROZEN_EXISTING_PARENT',new_pretraining_updates=0)
        completed.append(spec['run_id'])
        del route,x,y,m,prediction;gc.collect();torch.cuda.empty_cache()
    if not all(f.verify_unchanged().values()):raise ValueError('F invariant failed in frozen comparison')
    return dict(status='PASS',completed=completed,new_training_updates=0)


def phase_b():
    e.cuda_setup();ctx=e.context();f=e.foundation(ctx)
    decision=c.read_json(c.REPORT/'UPDATE_DECISION.json')
    updates=decision['updates']
    if updates not in (100,300):
        return dict(status='NOT_RUN_BUDGET',reason=decision)
    sources=core_identity()
    if sources!=decision['core_source_sha256']:
        raise ValueError('Core source changed after the TRAIN timing/update decision')
    completed=[];schedules={};sampling={}
    for seed in e.SEEDS:
        schedule=make_schedule(seed,len(ctx['data']['train_origins']),updates)
        schedules[seed]=schedule
        matrix=np.full((updates,8),-1,np.int64)
        for i,indices in enumerate(schedule):matrix[i,:len(indices)]=indices
        sampling[str(seed)]={'matrix_sha256':c.array_sha(matrix),'counts':[len(indices) for indices in schedule],
            'seen_windows':sum(map(len,schedule)),'unique_pool_origins':len(ctx['data']['train_origins']),
            'pool_origin_sha256':c.array_sha(ctx['data']['train_origins'])}
        path=c.CACHE/f'schedule_s{seed}.npz'
        if not path.exists():c.seal_npz(path,indices=matrix,origins=ctx['data']['train_origins'])
    if not (c.REPORT/'SAMPLING_SCHEDULES.json').exists():
        c.seal_json(c.REPORT/'SAMPLING_SCHEDULES.json',sampling)
    for base in [s for s in e.candidate_specs() if s['phase']=='B']:
        spec={**base,'updates':updates}
        receipt_path=c.REPORT/'training'/(spec['run_id']+'.json')
        if receipt_path.exists():
            old=c.read_json(receipt_path)
            if old['status']!='COMPLETE':raise ValueError('Prior unsuccessful adaptation requires an explicit new attempt')
            completed.append(spec['run_id']);continue
        c.progress('Phase_B',run_id=spec['run_id'],current=len(completed),total=8,update=0,updates=updates)
        route=e.build_route(spec,ctx,f,online=False)
        x,y,m,feature_cache,diagnostic=_inputs(ctx,spec,route)
        val_x,val_y,val_m=d.windows(ctx['data'],ctx['data']['val_origins'],spec['L'])
        val_features=e._features(route['feature_record'],'val') if 'feature_record' in route else None
        cp0=e.predict(route,f,val_x,val_features,batch=8)
        reference_id=f'A_GT_L{spec["L"]}_s{spec["seed"]}'
        with np.load(c.CACHE/'predictions'/(reference_id+'.npz'),allow_pickle=False) as reference:
            cp0_parity=e.parity(cp0,reference['prediction'])
        cp0_spec={**spec,'run_id':'CP0_'+spec['run_id'],'phase':'CP0','updates':0}
        e.prediction_artifact(cp0_spec,ctx,val_x,cp0,val_y,val_m,CP0_parent_parity=cp0_parity,
            trainable_parameters=sum(parameter.numel() for parameter in route['gt_route'].parameters()),
            GT_initial_state_sha256=c.state_sha(route['gt_route'].gt))
        gradient=gradient_through_future(route,x,feature_cache)
        result=train_arm(route['gt_route'],x,y,m,schedules[spec['seed']],feature_cache=feature_cache,
            diagnostics=diagnostic,output_dir=c.CACHE/'training'/spec['run_id'],run_id=spec['run_id'],
            seed=spec['seed'],microbatch=1,
            callback=lambda update,details:c.progress('Phase_B',run_id=spec['run_id'],current=len(completed),total=8,update=update,updates=updates))
        for group in ('encoder','decoder','product')+ (('W',) if spec['mode'] in ('pre','rand') else ()):
            if result['final_state_delta'][group]['l2']<=0 or result['group_gradient_evidence'][group]['updates_with_nonzero_gradient']<=0:
                raise ValueError('Required trainable group did not participate: '+group)
        final=e.predict(route,f,val_x,val_features,batch=8)
        branch={}
        if spec['mode'] in ('pre','rand'):
            with torch.no_grad():
                xt=torch.as_tensor(val_x[:8],device='cuda')
                ft=torch.as_tensor(val_features[:8],device='cuda')
                with_branch=route['gt_route'].forecast(xt,ft).cpu().numpy()
                removed=route['gt_route'].forecast(xt,None).cpu().numpy()
            branch=dict(fixed_VAL_origin_indices=list(range(8)),max_abs=float(np.abs(with_branch-removed).max()),
                        l2=float(np.linalg.norm(with_branch.astype(np.float64)-removed.astype(np.float64))),
                        interpretation='post-training branch use diagnostic, not a retrained control')
        checkpoint=result['checkpoint_paths'][str(updates)]
        public_result=dict(status='COMPLETE',spec=spec,**_public(result),checkpoint_sha256=c.file_sha(checkpoint),
                           CP0_parent_parity=cp0_parity,gradient_probe=gradient,branch_removal=branch,
                           core_source_sha256=sources,TS2_cache_used_only_for_fixed_features=True,
                           F_output_cache_used=False,new_TS2_pretraining_updates=0,new_GT_reconstruction_pretraining_updates=0)
        c.seal_json(receipt_path,public_result)
        e.prediction_artifact(spec,ctx,val_x,final,val_y,val_m,
            checkpoint_sha256=c.file_sha(checkpoint),trainable_parameters=result['trainable_parameters'],
            training_receipt_sha256=c.file_sha(receipt_path),model_status='FIXED_BUDGET_FORECAST_ADAPTED',
            GT_initial_parent_reference=reference_id,branch_removal=branch)
        if sources!=core_identity():raise ValueError('Scientific core changed during registered adaptation')
        completed.append(spec['run_id'])
        del route,x,y,m,feature_cache,diagnostic,val_x,val_y,val_m,val_features,cp0,final,result
        gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',completed=completed,fixed_updates=updates,source_sha256=sources,F_invariant=f.verify_unchanged())


def cost():
    # This coordinator never initializes CUDA. Each measured route has a new OS process.
    completed=[]
    for spec in e.candidate_specs():
        for batch in (1,8):
            identity=f'{spec["run_id"]}_B{batch}'
            path=c.REPORT/'cost_receipts'/(identity+'.json')
            if path.exists():
                if c.read_json(path)['status'] not in ('PASS','COMPLETE'):
                    raise ValueError('Prior cost failure requires a new attempt, not an overwrite')
                completed.append(identity);continue
            c.progress('cost',run_id=spec['run_id'],batch=batch,current=len(completed),total=42)
            log=c.CACHE/('cost_'+identity+'.log')
            with log.open('xb') as stream:
                process=subprocess.run([sys.executable,'-m',f'research.{c.CODE.name}.cost',
                    '--run-id',spec['run_id'],'--batch',str(batch)],cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError('Independent cost worker failed; preserved '+log.name)
            completed.append(identity)
    return dict(status='PASS',independent_cost_processes=len(completed))
