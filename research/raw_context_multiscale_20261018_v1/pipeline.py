"""Frozen-parent M_cal fitting, sealed V forecasts and fresh-process costs."""
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import calibration,common,deployment


def _origins(task,split):
    lo,hi = task['split_ranges'][split]
    return np.arange(max(lo,task['L']),hi-task['H']+1,1 if split=='M_cal' else task['H'],dtype=np.int64)


def _forecast(ctx,task,spec,run_id,split,*,pool=None,device='cuda'):
    output = ctx['cache']/'predictions'/run_id/(split+'.npz')
    binding_path = output.with_suffix('.binding.json')
    if binding_path.exists():
        bound = common.read(binding_path)
        if not output.exists() or common.sha(output)!=bound['sha256']:
            raise ValueError('Frozen-parent prediction bytes changed')
    route = deployment.make_route(ctx,task,spec,device=device,foundation_pool=pool)
    try:
        origins = _origins(task,split)
        rows = origins[:,None]+np.arange(task['H'])[None]
        mask = task['observed'][rows]
        target = np.where(mask,(task['physical'][rows]-route.mean)/route.std,0).astype(np.float32)
        metadata = dict(run_id=run_id,dataset=task['dataset'],split=split,
            data_sha256=task['data_sha256'],parent_fit_end=task['split_ranges']['A_stop'][1],
            route_spec=spec,scaler_sha256=ctx['modules']['fit'].scaler_fingerprint(route.mean,route.std),
            forecast_source_sha256=common.sha(__file__),T_access=False)
        previous_prediction = None
        if output.exists():
            with np.load(output,allow_pickle=False) as saved:
                if (json.loads(str(saved['metadata_json'].item()))!=metadata
                        or not np.array_equal(saved['origins'],origins)
                        or not np.array_equal(saved['observed'],mask)
                        or not np.array_equal(saved['target_normalized'],target)):
                    raise ValueError('Fixed-parent prediction cache identity changed')
                prediction = saved['pred_normalized'].copy()
                previous_prediction = prediction
        if not output.exists() or not binding_path.exists():
            predictions = []
            with torch.no_grad():
                for start in range(0,len(origins),8):
                    if time.time()>=ctx['deadline']:
                        raise TimeoutError('Registered budget reached during frozen inference')
                    selected = origins[start:start+8]
                    histories = selected[:,None]+np.arange(-task['L'],0)[None]
                    physical_prediction = route.predict_physical(task['physical'][histories],task['observed'][histories])
                    predictions.append(((physical_prediction-route.mean)/route.std).astype(np.float32))
            regenerated = np.concatenate(predictions)
            if previous_prediction is not None:
                if not np.allclose(previous_prediction,regenerated,atol=common.ATOL,rtol=common.RTOL):
                    raise ValueError('Unsealed prediction cache fails fixed-model regeneration parity')
                prediction = previous_prediction
                common.event(ctx,'unsealed_cache_recovery','PASS',run_id=run_id,split=split,
                             prediction_sha256=common.sha(output),max_abs=float(np.abs(prediction-regenerated).max()))
            else:
                prediction = regenerated
        evidence = common.save_prediction(output,prediction=prediction,target=target,observed=mask,
                                           origins=origins,task=task,metadata=metadata)
        evidence['prediction_binding'] = common.binding(output,ctx)
        common.sealed_json(binding_path,dict(sha256=common.sha(output),arrays=evidence['arrays'],metadata=metadata))
        return dict(evidence=evidence,prediction=prediction,target=target,mask=mask,origins=origins,
                    mean=route.mean.copy(),std=route.std.copy())
    finally:
        route.close()


def _past(ctx,task,mean,std,origins):
    windows = ctx['modules']['training'].FastWindows(task,mean,std,'cpu')
    return windows.get(origins)[0].numpy()


def calibrate(ctx,tasks,inventory):
    index = ctx['cache']/'CALIBRATION_MODELS.json'
    if index.exists():
        saved = common.read(index)
        fit_path = ctx['report']/'COMPLEMENTARITY_FITS.json'
        if not fit_path.exists():
            raise ValueError('The calibration completion index requires its sealed public fitting provenance')
        fits = common.read(fit_path)
        if (fits['source_sha256']!=common.sha(calibration.__file__)
                or fits['head_fits']!=[item for item in saved if item['mode']!='alpha']):
            raise ValueError('The completed calibration source or fitting provenance changed')
        alpha_fits = {item['run_id']:item for item in fits['alpha_fits']}
        for item in saved:
            if 'head_binding' in item:
                common.resolve_binding(item['head_binding'],ctx)
            for key in ('parent_g_prediction_binding','parent_b_prediction_binding'):
                if item.get(key) is not None:
                    common.resolve_binding(item[key],ctx)
            if item['mode']=='alpha':
                fitted = alpha_fits[item['run_id'].rsplit('_',1)[0]]
                if fitted['diagnostic']!=item['diagnostic']:
                    raise ValueError('The completed analytic-alpha diagnostic changed')
        return saved
    from .run import gpu_session
    started = time.time()
    records,fit_rows = [],[]
    for dataset,task in tasks.items():
        parents = [p for p in inventory['parents'] if p['dataset']==dataset and p['role'] in ('S','G','Raw')]
        parent_predictions = {}
        model_path = ctx['asset']/ctx['modules']['legacy'].MODEL_REL
        pool = deployment.legacy.FoundationPool(model_path,'cuda')
        with gpu_session(ctx,'fixed_parent_M_cal_cache_'+dataset):
            try:
                for parent in parents:
                    name = f"CAL_PARENT_{dataset}_{parent['role']}_{parent.get('seed')}"
                    print('MCAL_CACHE '+name,flush=True)
                    cache_started = time.perf_counter()
                    values = _forecast(ctx,task,parent['route_spec'],name,'M_cal',pool=pool)
                    common.event(ctx,'parent_cache','COMPLETE',run_id=name,
                                 wall_seconds=time.perf_counter()-cache_started,
                                 prediction_binding=values['evidence']['prediction_binding'])
                    parent_predictions[(parent['role'],parent.get('seed'))] = values
            finally:
                pool.close()
                del pool
                gc.collect()
                torch.cuda.empty_cache()
        g_parent = next(p for p in parents if p['role']=='G')
        g_values = parent_predictions[('G',g_parent['seed'])]
        past = _past(ctx,task,g_values['mean'],g_values['std'],g_values['origins'])
        split = calibration.calibration_split(task)
        for seed in common.SEEDS:
            b_parent = next(p for p in parents if p['role']=='S' and p['seed']==seed)
            b_values = parent_predictions[('S',seed)]
            for base_role in ('Raw','G'):
                parent = next(p for p in parents if p['role']==base_role)
                values = parent_predictions[(base_role,parent.get('seed'))]
                for field in ('target','mask','origins','mean','std'):
                    if not np.array_equal(values[field],b_values[field]):
                        raise ValueError('Calibration parents differ in exact cohort/scaler/mask')
                diagnostic = calibration.fit_alpha_diagnostic(values['prediction'],b_values['prediction'],
                    values['target'],values['mask'],split['fit_indices'],split['select_indices'])
                prefix = f'Alpha_{dataset}_{base_role}_S_s{seed}'
                fit_rows.append(dict(run_id=prefix,dataset=dataset,seed=seed,base_role=base_role,
                                     parent_g=parent,parent_b=b_parent,diagnostic=diagnostic,
                                     parent_g_prediction_binding=values['evidence']['prediction_binding'],
                                     parent_b_prediction_binding=b_values['evidence']['prediction_binding']))
                for label,alpha in (('alpha0',0.),('alpha05',.5),('alpha1',1.),
                                    ('fitted',diagnostic['refit_alpha']['alpha'])):
                    run_id = prefix+'_'+label
                    route = dict(family='COMBINED',run_id=run_id,operation='alpha',alpha=float(alpha),
                                 g_route=parent['route_spec'],b_route=b_parent['route_spec'])
                    records.append(dict(run_id=run_id,dataset=dataset,seed=seed,mode='alpha',
                        base_role=base_role,alpha_label=label,alpha=float(alpha),route_spec=route,
                        parent_g=parent,parent_b=b_parent,diagnostic=diagnostic,selection_split='M_fit_analytic',
                        parent_g_prediction_binding=values['evidence']['prediction_binding'],
                        parent_b_prediction_binding=b_values['evidence']['prediction_binding'],
                        refit_split='M_cal',V_used=False))
            searches = {}
            for control in (False,True):
                variant = 'past_control' if control else 'conditional'
                run_id = f'Head_{dataset}_{variant}_s{seed}'
                args = (past,g_values['prediction'],None if control else b_values['prediction'],
                        g_values['target'],g_values['mask'],g_values['origins'],task)
                target = ctx['cache']/'calibration_heads'/(run_id+'.pt')
                target.parent.mkdir(parents=True,exist_ok=True)
                contract = dict(dataset=dataset,seed=seed,control=control,
                    data_sha256=task['data_sha256'],
                    scaler=dict(mean=common.array_identity(g_values['mean']),std=common.array_identity(g_values['std'])),
                    calibration_source_sha256=common.sha(calibration.__file__),
                    past=common.array_identity(past),origins=common.array_identity(g_values['origins']),
                    g=g_values['evidence']['prediction_binding'],
                    b=None if control else b_values['evidence']['prediction_binding'])
                if target.exists():
                    saved = torch.load(target,map_location='cpu',weights_only=False)
                    if saved.get('contract')!=contract:
                        raise ValueError('The existing head fitting contract changed')
                    with torch.random.fork_rng(devices=[]):
                        head = calibration.ConditionalHead(task['L'],task['H'],hidden=32)
                    head.load_state_dict(saved['state'],strict=True)
                    if calibration.module_state_hash(head)!=saved['receipt']['final_state_sha256']:
                        raise ValueError('The existing head state differs from its fresh-refit receipt')
                    search,receipt = saved['search'],saved['receipt']
                    head_search_seconds = saved.get('head_search_seconds')
                    head_refit_seconds = saved.get('head_refit_seconds')
                else:
                    search_checkpoints = []
                    def preserve_checkpoint(state,row):
                        search_checkpoints.append(dict(state=state,curve=row))
                    head_started = time.perf_counter()
                    search = calibration.head_search(*args,seed=seed,control=control,
                        checkpoint_sink=preserve_checkpoint,deadline=ctx['deadline'])
                    head_search_seconds = time.perf_counter()-head_started
                    head_started = time.perf_counter()
                    state,receipt = calibration.head_refit(*args,seed=seed,
                        selected_epoch=search['selected_epoch'],control=control,deadline=ctx['deadline'])
                    head_refit_seconds = time.perf_counter()-head_started
                    with target.open('xb') as stream:
                        torch.save(dict(state=state,head_config=dict(length=task['L'],horizon=task['H'],hidden=32),
                                        search=search,receipt=receipt,contract=contract,
                                        search_checkpoints=search_checkpoints,
                                        head_search_seconds=head_search_seconds,head_refit_seconds=head_refit_seconds),stream)
                    common.event(ctx,'head_search','COMPLETE',run_id=run_id,wall_seconds=head_search_seconds)
                    common.event(ctx,'head_refit','COMPLETE',run_id=run_id,wall_seconds=head_refit_seconds)
                route = dict(family='COMBINED',run_id=run_id,operation=variant,
                    g_route=g_parent['route_spec'],b_route=b_parent['route_spec'],
                    head_state_path=target.relative_to(ctx['cache']).as_posix(),head_state_sha256=common.sha(target))
                record = dict(run_id=run_id,dataset=dataset,seed=seed,mode=variant,route_spec=route,
                    head_binding=common.binding(target,ctx),parent_g=g_parent,parent_b=None if control else b_parent,
                    parent_g_prediction_binding=g_values['evidence']['prediction_binding'],
                    parent_b_prediction_binding=None if control else b_values['evidence']['prediction_binding'],
                    head_search_seconds=head_search_seconds,head_refit_seconds=head_refit_seconds,
                    search=search,refit=receipt,selection_split='M_select',V_used=False)
                records.append(record)
                searches[variant] = search
            if (searches['conditional']['initialization_sha256']!=searches['past_control']['initialization_sha256']
                    or searches['conditional']['sampler_sha256']!=searches['past_control']['sampler_sha256']):
                raise ValueError('Conditional and past-control must have the same initial state and sample schedule')
    fit_path = ctx['report']/'COMPLEMENTARITY_FITS.json'
    payload = dict(status='SEALED_BEFORE_V',
        alpha_fits=fit_rows,head_fits=[r for r in records if r['mode']!='alpha'],V_used=False,
        head_search_count=sum(r['mode']!='alpha' for r in records),
        head_refit_count=sum(r['mode']!='alpha' for r in records),wall_seconds=time.time()-started,
        source_sha256=common.sha(calibration.__file__))
    if fit_path.exists():
        payload['wall_seconds'] = common.read(fit_path)['wall_seconds']
    common.sealed_json(fit_path,payload)
    # This index is the completion marker, written after all public provenance.
    common.sealed_json(index,records)
    common.event(ctx,'calibration','COMPLETE',wall_seconds=time.time()-started,models=len(records))
    return records


def _costs(ctx,arms):
    from . import costs
    results = []
    directory = ctx['cache']/'cost_requests'
    directory.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        variants = [('native',arm['route_spec'])]
        if arm['mode']=='A':
            variants.append(('two_call',dict(arm['route_spec'],schedule='two_call')))
        for schedule,spec in variants:
            for batch in (1,8):
                cost_id = arm['run_id']+'_'+schedule+'_B'+str(batch)
                directory_out = ctx['report']/'costs'
                original_output = directory_out/(cost_id+'.json')
                request = dict(asset_root=str(ctx['asset']),dataset=arm['dataset'],route_spec=spec,
                    accuracy_run_id=arm['run_id'],cost_id=cost_id,batch_size=batch,
                    device='cpu' if arm['mode']=='Ridge' else 'cuda',
                    prediction_path=str(common.resolve_binding(arm['prediction_binding'],ctx)),
                    prediction_sha256=arm['prediction_binding']['sha256'])
                path = directory/(cost_id+'.json')
                common.sealed_json(path,request)
                existing = [original_output,*sorted(directory_out.glob(cost_id+'.attempt*.json'))]
                output = None
                for candidate in existing:
                    if not candidate.exists():
                        continue
                    value = common.read(candidate)
                    if value.get('status')=='COMPLETE':
                        costs.validate_receipt(value,request,ctx,path)
                        output = candidate
                        break
                    if value.get('status')!='INVALID':
                        raise ValueError('Unrecognized prior cost attempt: '+str(candidate))
                if output is None:
                    if time.time()>=ctx['deadline']:
                        raise TimeoutError('Registered budget reached before the next cost facet')
                    output = original_output
                    attempt = 1
                    while output.exists() or output.with_suffix('.trials.csv').exists():
                        output = directory_out/(cost_id+f'.attempt{attempt:03}.json')
                        attempt += 1
                    if output!=original_output:
                        common.event(ctx,'cost_retry','STARTED',cost_id=cost_id,
                                     output=output.relative_to(ctx['report']).as_posix(),
                                     prior_attempts_preserved=True,fresh_process=True)
                    print('COST '+cost_id,flush=True)
                    subprocess.run([sys.executable,'-u','-m','research.'+common.NAME+'.costs',
                                    '--request',str(path),'--output',str(output),
                                    '--deadline-unix',str(ctx['deadline'])],cwd=ctx['repo'],check=True)
                value = common.read(output)
                costs.validate_receipt(value,request,ctx,path)
                results.append(dict(cost_id=cost_id,run_id=arm['run_id'],schedule=schedule,
                                    batch=batch,output=common.binding(output,ctx)))
    common.sealed_json(ctx['report']/'COST_INDEX.json',dict(status='COMPLETE',facets=results))
    return results


def complete(ctx,tasks,inventory,finals):
    from .run import gpu_session
    seal = common.read(ctx['report']/'MODELS_SEALED_BEFORE_V.json')
    if seal.get('status')!='SEALED' or seal.get('V_used') is not False:
        raise ValueError('All model and transfer decisions must be sealed before new V')
    arms = []
    for item in finals:
        arms.append({key:copy.deepcopy(value) for key,value in item.items() if key not in ('search','refit')})
        arms[-1]['search_summary'] = item['search']['summary']
        arms[-1]['curves'] = item['search']['curves']
        arms[-1]['search_training_segments'] = item['search'].get('training_segments',[])
        if 'refit' in item:
            arms[-1]['refit_summary'] = item['refit']['refit_summary']
            arms[-1]['refit_training_segments'] = item['refit'].get('training_segments',[])
    for parent in inventory['parents']:
        if parent['role']=='S':
            continue
        arms.append(dict(dataset=parent['dataset'],mode=parent['role'],seed=parent.get('seed'),
                         run_id=parent['role']+'_'+parent['dataset'],route_spec=parent['route_spec'],
                         reused=True,parent=parent))
    arms.extend(common.read(ctx['cache']/'CALIBRATION_MODELS.json'))
    for dataset,task in tasks.items():
        with gpu_session(ctx,'final_V_'+dataset):
            for arm in (a for a in arms if a['dataset']==dataset):
                device = 'cpu' if arm['mode']=='Ridge' else 'cuda'
                print('V_FORECAST '+arm['run_id'],flush=True)
                values = _forecast(ctx,task,arm['route_spec'],arm['run_id'],'V_select',device=device)
                arm.update(prediction_binding=values['evidence']['prediction_binding'],
                           score=values['evidence']['score'],array_identities=values['evidence']['arrays'])
    common.sealed_json(ctx['cache']/'SCORED_ARMS.json',arms)
    common.sealed_json(ctx['report']/'SCORES.json',dict(status='COMPLETE',arms=arms,V_independent_test=False))
    with gpu_session(ctx,'fresh_process_whole_path_costs'):
        _costs(ctx,arms)
    from . import report
    report.verify(ctx)
    report.generate(ctx)
    common.event(ctx,'experiment','COMPLETE')
