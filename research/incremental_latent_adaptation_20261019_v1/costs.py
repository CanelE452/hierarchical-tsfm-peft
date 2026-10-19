"""Fresh-process whole-path inference cost worker, after B/C sealing only."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

for _variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'

import numpy as np
import psutil
import torch

from research.memory_budget_20261018_v1.worker import ProcessSampler,_memory,_sync
from research.raw_context_multiscale_20261018_v1 import common as inherited
from . import blocks,common,data_contract
from .deployment import Route,assert_v_sealed,c_prefix_metadata,METHODS


PROTOCOL = dict(warmups=10,timed_trials=20,batches=[1,8],F_row_microbatch=128,
    whole_path='physical + original past bool mask -> causal fill -> fixed A60 scaler -> G/actual correction -> inverse scaler -> CPU numpy',
    timing_origins='first eight fixed V origins; B1 cycles those origins and B8 repeats the ordered bundle',
    cached_forecasts_in_timed_path=0,cached_features_in_timed_path=0,CPU_threads=2,
    parity_atol=1e-5,parity_rtol=1e-4)


def validate_request(request):
    if request.get('dataset') not in common.DATASETS or request.get('method') not in METHODS:
        raise ValueError('Registered dataset/method required')
    if type(request.get('seed')) is not int or request['seed'] not in common.SEEDS:
        raise ValueError('Registered correction/S seed required')
    if type(request.get('batch_size')) is not int or request['batch_size'] not in (1,8):
        raise ValueError('Registered B1/B8 required')
    if request.get('device','cuda') not in ('cpu','cuda','cuda:0'):
        raise ValueError('Only registered CPU/first CUDA device supported')
    if request['method'] in ('OWN','ALL','MEAN') and request.get('device','cuda')!='cpu':
        raise ValueError('Ridge/mean probe costs are a separate CPU scope')
    return dict(request)


def fixed_origins(task):
    start,stop = task['split_ranges']['V_select']
    if stop>len(task['physical']):
        raise ValueError('Full V prefix is not materialized')
    return np.arange(max(start,task['L']),stop-task['H']+1,task['H'],dtype=np.int64)


def timing_origin_bundles(origins,batch_size):
    values = np.asarray(origins)
    if values.ndim!=1 or values.dtype.kind not in 'iu' or not len(values) or batch_size not in (1,8):
        raise ValueError('Ordered nonempty integer origin cohort and B1/B8 required')
    selected = values[np.arange(8)%len(values)]
    return [selected[i:i+1] for i in range(8)] if batch_size==1 else [selected]


def windows(task,origins):
    return (np.stack([task['physical'][int(o)-task['L']:int(o)] for o in origins]).astype(np.float32),
            np.stack([task['observed'][int(o)-task['L']:int(o)] for o in origins]))


def latency_statistics(trials,batch_size):
    values = np.array([r['whole_seconds']*1000 for r in trials if r['phase']=='timed'],np.float64)
    if len(values)!=20 or not np.isfinite(values).all() or np.min(values)<=0:
        raise ValueError('Exactly twenty positive measured whole-path trials required')
    q25,median,q75 = np.quantile(values,[.25,.5,.75])
    return dict(median_batch_ms=float(median),IQR_batch_ms=float(q75-q25),
                median_origin_ms=float(median/batch_size),IQR_origin_ms=float((q75-q25)/batch_size),
                median_ms_per_batch=float(median),median_ms_per_origin=float(median/batch_size),
                B8_origin_is_single_request_latency=False)


def _deadline(request):
    if time.time()>=float(request.get('deadline_unix',1e12)):
        raise TimeoutError('Registered inference cost deadline reached')


def _reference(request):
    if not request.get('prediction_reference_path'):
        return None
    path = Path(request['prediction_reference_path'])
    if not request.get('prediction_reference_sha256') or common.sha(path)!=request['prediction_reference_sha256']:
        raise ValueError('Exact saved V prediction SHA required')
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in ('pred_normalized','target_normalized','observed','origins','channel_ids')}


def full_v_parity(route,task,origins,reference,request):
    """Verify all fixed V origins outside timing; never select/retrain here."""
    assert_v_sealed()
    predictions=[]
    for lo in range(0,len(origins),request['batch_size']):
        _deadline(request)
        physical,mask = windows(task,origins[lo:lo+request['batch_size']])
        prediction=route.predict_physical(physical,mask)
        predictions.append(((prediction-route.mean)/route.std).astype(np.float32))
    prediction=np.concatenate(predictions)
    indices=origins[:,None]+np.arange(task['H'])[None]
    observed=task['observed'][indices]
    target=np.where(observed,(task['physical'][indices]-route.mean)/route.std,0.).astype(np.float32)
    parity=dict(status='NO_REFERENCE',n_origins=len(origins),new_selection=False)
    if reference is not None:
        for key,expected in (('origins',origins),('observed',observed),('channel_ids',np.asarray(task['columns']))):
            if not np.array_equal(reference[key],expected):
                raise ValueError('Whole-V parity cohort differs: '+key)
        if reference['observed'].dtype!=np.bool_ or not np.array_equal(reference['target_normalized'][observed],target[observed]):
            raise ValueError('Whole-V parity observed targets differ')
        if not np.allclose(prediction,reference['pred_normalized'],atol=1e-5,rtol=1e-4):
            raise ValueError('Whole physical deployment differs from final accuracy forecasts')
        parity.update(status='PASS',max_absolute_normalized_difference=float(np.max(np.abs(prediction-reference['pred_normalized']))),
                      atol=1e-5,rtol=1e-4)
    return dict(**parity,metric=common.score(prediction,target,observed),
                prediction_identity=common.identity(prediction),origins_identity=common.identity(origins))


def measured_trials(route,bundles,*,request,sampler=None):
    rows=[]
    for phase,count in (('warmup',10),('timed',20)):
        if sampler is not None:
            sampler.set_phase(phase)
        for trial in range(count):
            _deadline(request)
            physical,observed = bundles[trial%len(bundles)]
            _sync(route.device)
            if phase=='timed' and route.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(route.device)
            started=time.perf_counter()
            output=route.predict_physical(physical,observed)
            _sync(route.device)
            elapsed=time.perf_counter()-started
            receipt=dict(route.last_receipt)
            if receipt['cached_forecast_or_features_reads']!=0:
                raise ValueError('Cached prediction/features in timed deployment path')
            row=dict(phase=phase,trial=trial,whole_seconds=elapsed,batch_size=len(physical),
                     ms_batch=elapsed*1000,ms_origin=elapsed*1000/len(physical),
                     F_logical_rows=receipt['F_logical_rows'],F_physical_rows=receipt['F_physical_rows'],
                     F_calls=receipt['F_calls'],F_instances_created=receipt['F_instances_created'],
                     F_invocations=receipt['F_invocations'],selected_prefix=receipt['selected_prefix'],
                     executed_prefix=receipt.get('executed_prefix',receipt['selected_prefix']),
                     artifact_selected_prefix=receipt.get('artifact_selected_prefix'),
                     prefix_is_selected=receipt.get('prefix_is_selected'),
                     G_only_without_F_resident=receipt.get('G_only_without_F_resident'),
                     cached_forecast_or_features_reads=0,
                     output_identity=common.identity(output),**_memory(route.device))
            if phase=='warmup':
                row['peak_allocated_MiB']=row['peak_reserved_MiB']=None
            if row['F_instances_created']>1:
                raise ValueError('More than one resident F instance created')
            rows.append(row)
    return rows


def _load_parent_models(ctx,task,request):
    """Reuse the audited exact parent loader without materializing M_cal windows."""
    method,dataset,seed = request['method'],request['dataset'],request['seed']
    alpha=float(request.get('alpha',.5))
    need_g=method not in ('S','RAW') and not (method=='ALPHA_fitted' and alpha==1)
    need_s=method in ('S',*common.METHODS) or (method=='ALPHA_fitted' and alpha!=0)
    if method in blocks.METHODS:
        need_s=c_prefix_metadata(request,task)['executed_prefix']>0
    parents=dict(task=task)
    norms=[]
    if need_g:
        model,mean,std,parent=data_contract._parent(ctx,task,dataset,'G',92601)
        parents.update(g_model=model,parent_g=parent)
        norms.append((mean,std))
    if need_s:
        model,mean,std,parent=data_contract._parent(ctx,task,dataset,'S',seed)
        parents.update(s_model=model,parent_s=parent)
        norms.append((mean,std))
    if not norms:
        mean,std,_=ctx['modules']['data'].fit_statistics(task,task['split_ranges']['M_cal'][0])
        norms.append((mean,std))
    for mean,std in norms[1:]:
        if not np.array_equal(mean,norms[0][0]) or not np.array_equal(std,norms[0][1]):
            raise ValueError('Deployment parent canonical scalers differ')
    parents.update(mean=norms[0][0],std=norms[0][1])
    return parents


def worker(request,asset_root,*,request_path=None):
    request=validate_request(request)
    seal=assert_v_sealed()
    _deadline(request)
    wall_started=time.perf_counter()
    device=request.get('device','cuda')
    torch.set_num_threads(2)
    ctx=inherited.context(asset_root,repo=common.REPO,device=device)
    data_started=time.perf_counter()
    parent_task,data_access=data_contract._source_task(ctx,request['dataset'])
    # This first V numeric materialization is guarded before calling the inherited loader.
    task=inherited.load_task(ctx,request['dataset'])
    data_seconds=time.perf_counter()-data_started
    parent_started=time.perf_counter()
    parents=_load_parent_models(ctx,parent_task,request)
    parent_load_seconds=time.perf_counter()-parent_started
    route=Route(request,parents,ctx=ctx,task=task,device=device)
    route.model_load_seconds+=parent_load_seconds
    del parents
    del parent_task
    ctx.pop('_incremental_data',None)
    gc.collect()
    route.warm_load()
    _sync(device)
    cold_memory=_memory(device)
    cold_rss=psutil.Process().memory_info().rss/1024**2
    origins=fixed_origins(task)
    window_started=time.perf_counter()
    bundles=[windows(task,oo) for oo in timing_origin_bundles(origins,request['batch_size'])]
    slicing_seconds=time.perf_counter()-window_started
    reference=_reference(request)
    # Accuracy parity is separate from cost measurement and never changes decisions.
    parity=full_v_parity(route,task,origins,reference,request)
    del reference
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    sampler=ProcessSampler(device)
    sampler.start()
    try:
        trials=measured_trials(route,bundles,request=request,sampler=sampler)
    finally:
        memory=sampler.stop()
    result=dict(status='COMPLETE',dataset=request['dataset'],method=request['method'],seed=request['seed'],
        batch_size=request['batch_size'],device=device,request=request,
        request_sha256=common.sha(request_path) if request_path else None,
        protocol=PROTOCOL,C_decisions_binding=seal,
        pid=os.getpid(),process_create_time=psutil.Process().create_time(),
        device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        torch_version=torch.__version__,dtype='FP32_model_FP64_physical_inverse',
        F_tensor_sha256=ctx['modules']['legacy'].F_SHA if route.needs_foundation else None,
        producer_sources={p.relative_to(common.REPO).as_posix():common.sha(p) for p in (
            Path(__file__),Path(__file__).with_name('deployment.py'),Path(__file__).with_name('models.py'),
            Path(__file__).with_name('blocks.py'),Path(__file__).with_name('data_contract.py'))},
        executed_prefix=route.k if request['method'] in blocks.METHODS else request.get('prefix'),
        artifact_selected_prefix=route.artifact_selected_prefix,
        prefix_is_selected=route.c_prefix_info.get('prefix_is_selected'),
        model_cold_load_seconds=route.model_load_seconds,parent_CPU_load_seconds=parent_load_seconds,
        cold_load_scope='exact parent construction/artifact read plus correction construction/state read, device transfer and one F load; excludes framework imports/data decode',
        data_preparation_seconds=data_seconds,Mcal_parent_source_access=data_access,
        window_slicing_seconds=slicing_seconds,window_slicing_inside_timed_path=False,
        cold_memory=cold_memory,cold_CPU_RSS_MiB=cold_rss,
        peak_reset_scope='after warmup and before each timed trial; summary maximum across timed trials',
        full_V_parity=parity,trials=trials,memory=memory,
        whole_worker_seconds=time.perf_counter()-wall_started,**route.resident_state(),
        **latency_statistics(trials,request['batch_size']))
    route.close()
    result['whole_worker_seconds']=time.perf_counter()-wall_started
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--asset-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous cost attempt; choose a new output path')
    result=worker(common.read(args.request),args.asset_root,request_path=args.request)
    common.atomic(output,result)
    print(json.dumps({key:result[key] for key in ('status','dataset','method','seed','batch_size','device','median_batch_ms')},allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
