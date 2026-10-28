"""Fresh-process, matched B8 whole-path costs for the new dataset campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from research.memory_budget_20261018_v1.worker import _fingerprint as fingerprint
from research.memory_budget_20261018_v1.worker import independent_metrics


NAME = 'b2_dataset_extension_20261018_v1'
ATOL, RTOL = 1e-5, 1e-4
FAMILIES = {'RAW', 'B2', 'DIRECT_G'}


def validate_request(value):
    value = dict(value)
    if value.get('family') not in FAMILIES or value.get('batch_size', 8) != 8:
        raise ValueError('Registered families require exactly physical batch 8')
    if type(value.get('K')) is not int or value['K'] < 1:
        raise ValueError('K must be a positive integer')
    for key in ('dataset', 'run_id'):
        if not isinstance(value.get(key), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', value[key]):
            raise ValueError('Unsafe request identity: ' + key)
    if value.get('seed') not in (92601, 92602):
        raise ValueError('Registered seed must be 92601 or 92602')
    if value.get('factory') not in ('train:cost_inputs', 'baselines:cost_inputs'):
        raise ValueError('Factory must belong to this campaign')
    if value.get('device', 'cuda') != 'cuda':
        raise ValueError('Matched inference costs require the registered CUDA device')
    value.update(batch_size=8, device='cuda')
    return value


def expected_work(request, channels, batch):
    width = channels if request['family'] == 'RAW' else request['K'] if request['family'] == 'B2' else 0
    cap = 8 * request['K'] if width else None
    return dict(row_cap=cap, width=width, streams=batch * width,
                calls=math.ceil(batch * width / cap) if width else 0)


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def reset_peaks(device):
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()


def release_validation_cache(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def memory(device):
    cuda = str(device).startswith('cuda')
    return dict(allocated_MiB=torch.cuda.memory_allocated()/1024**2 if cuda else 0.,
        reserved_MiB=torch.cuda.memory_reserved()/1024**2 if cuda else 0.,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if cuda else 0.,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2 if cuda else 0.,
        RSS_MiB=psutil.Process(os.getpid()).memory_info().rss/1024**2)


def _cap(route, cap):
    if hasattr(route, 'spec'):
        route.spec.update(row_microbatch=cap, F_row_microbatch=cap)
    foundation = getattr(getattr(route, 'pool', None), 'foundation', None)
    if foundation is not None:
        foundation.row_microbatch = cap


def _calls(route, request, channels, batch, cap):
    work = expected_work(request, channels, batch)
    work['row_cap'] = cap
    work['calls'] = math.ceil(work['streams']/cap) if work['streams'] else 0
    receipt = route.last_receipt
    if (receipt.get('F_calls') != work['calls']
            or receipt.get('F_logical_streams') != work['streams']
            or receipt.get('F_physical_streams') != work['streams']):
        raise ValueError('Actual F call/stream counters differ from the matched row cap')
    return work


def _windows(task, origins):
    return (np.stack([task['physical'][int(o)-512:int(o)] for o in origins]).astype(np.float32),
            np.stack([task['observed'][int(o)-512:int(o)] for o in origins]).astype(bool))


def _cohort(route, inputs, request):
    task, saved = inputs['task'], inputs['saved']
    channels = len(task['columns'])
    if task['L'] != 512 or task['H'] != 48 or request['K'] != math.ceil(channels/4):
        raise ValueError('Registered L512/H48/K=ceil(C/4) contract differs')
    physical, observed = np.asarray(task['physical']), np.asarray(task['observed'])
    if physical.ndim != 2 or physical.shape[1] != channels or observed.shape != physical.shape or observed.dtype != np.bool_:
        raise ValueError('Physical data and boolean observation mask differ')
    origins = np.asarray(saved['origins'])
    lo, hi = task['split_ranges']['V_select']
    if (origins.ndim != 1 or origins.dtype.kind not in 'iu' or len(origins) < 8
            or len(np.unique(origins)) != len(origins) or np.any(origins < max(lo,512))
            or np.any(origins+48 > min(hi,len(physical)))):
        raise ValueError('Full V origins must be unique legal forecasts with a full timing batch')
    if not np.array_equal(np.asarray(saved['channel_ids']).astype(str), np.asarray(task['columns']).astype(str)):
        raise ValueError('Scored channel order differs')
    for key in ('mean', 'std'):
        value = np.asarray(getattr(route,key))
        if value.shape != (channels,) or not np.isfinite(value).all() or not np.array_equal(value,saved[key]):
            raise ValueError('Fixed A60 scaler differs from saved accuracy')
    if np.any(np.asarray(route.std) <= 0):
        raise ValueError('Fixed A60 scaler must have positive channel standard deviations')
    target, mask = np.asarray(saved['target_normalized']), np.asarray(saved['observed'])
    if target.shape != (len(origins),48,channels) or mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError('Scored target/mask shape differs')
    real_mask = np.stack([observed[int(o):int(o)+48] for o in origins])
    real_target = np.stack([physical[int(o):int(o)+48] for o in origins])
    normalized = (real_target-np.asarray(route.mean))/np.asarray(route.std)
    if not np.array_equal(real_mask,mask) or not np.allclose(normalized[mask],target[mask],atol=ATOL,rtol=RTOL):
        raise ValueError('Scored targets/masks differ from physical V and fixed scaler')
    return origins, channels


def full_validation(route, inputs, request, row_cap=None, allow_new_reference=False):
    origins, channels = _cohort(route,inputs,request)
    cap = row_cap if row_cap is not None else expected_work(request,channels,8)['row_cap']
    _cap(route,cap)
    predictions, batch_sizes, calls = [], [], []
    input_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    for start in range(0,len(origins),8):
        physical, observed = _windows(inputs['task'],origins[start:start+8])
        input_hash.update(physical.tobytes()); mask_hash.update(observed.tobytes())
        value = route.predict_physical(physical,observed)
        if value.shape != (len(physical),48,channels) or not np.isfinite(value).all():
            raise ValueError('Physical forecast is nonfinite or has the wrong shape')
        work = _calls(route,request,channels,len(physical),cap)
        predictions.append((value-np.asarray(route.mean,np.float64))/np.asarray(route.std,np.float64))
        batch_sizes.append(len(physical)); calls.append(work['calls'])
    actual = np.concatenate(predictions)
    expected = inputs['saved'].get('pred_normalized')
    if expected is None and not (allow_new_reference and request['family'] in ('RAW','DIRECT_G')):
        raise ValueError('Only RAW/DirectG may generate a new accuracy reference')
    if expected is not None and (np.asarray(expected).shape != actual.shape or not np.allclose(actual,expected,atol=ATOL,rtol=RTOL)):
        raise ValueError('Full V original-row128 prediction parity failed')
    metrics = independent_metrics(dict(inputs['saved'],pred_normalized=actual))
    fingerprints = {key:fingerprint(inputs['saved'][key]) for key in
                    ('origins','channel_ids','mean','std','target_normalized','observed')}
    fingerprints.update(physical_windows=dict(shape=[len(origins),512,channels],dtype='float32',sha256=input_hash.hexdigest()),
        history_observed=dict(shape=[len(origins),512,channels],dtype='bool',sha256=mask_hash.hexdigest()))
    return actual, dict(status='PASS',full_V_origins=len(origins),batch_sizes=batch_sizes,
        last_batch_size=batch_sizes[-1],F_calls=calls,row_cap=cap,atol=ATOL,rtol=RTOL,
        max_abs=float(np.max(np.abs(actual-np.asarray(expected)))) if expected is not None else 0.,
        metrics=metrics,fingerprints=fingerprints,output_fingerprint=fingerprint(actual))


def _states(route, inputs, request, device):
    foundation = getattr(getattr(route,'pool',None),'foundation',None)
    if request['family'] in ('RAW','B2'):
        if foundation is None or foundation.initial_hash != inputs['expected_f_sha']:
            raise ValueError('F state identity differs from the registered foundation')
        if getattr(foundation,'checkpoint_inputs',False):
            raise ValueError('F inference checkpointing must be disabled')
    elif foundation is not None:
        raise ValueError('Direct-only route must not instantiate F')
    modules = {name:value for name,value in [('adapter',getattr(route,'model',None)),
        ('F',getattr(foundation,'model',None))] if isinstance(value,torch.nn.Module)}
    result = {}
    for name,model in modules.items():
        if model.training or any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('Inference model must be FP32 eval')
        if name=='F' and any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('F parameters must remain frozen without gradients')
        if name=='F' and inputs.get('expected_f_params') is not None and sum(p.numel() for p in model.parameters())!=inputs['expected_f_params']:
            raise ValueError('Loaded F parameter count differs')
        if str(device).startswith('cuda') and any(p.device.type!='cuda' for p in model.parameters()):
            raise ValueError('Loaded inference parameters must be on CUDA')
        digest = hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def measure_inputs(request, inputs, device='cuda'):
    request = validate_request(request)
    started = time.perf_counter()
    route = inputs['route_loader']()
    try:
        if hasattr(route,'activate'): route.activate()
        synchronize(device)
        load_seconds = time.perf_counter()-started
        loaded = memory(device)
        before = _states(route,inputs,request,device)
        _cohort(route,inputs,request)
        saved = dict(inputs['saved'])
        inputs = dict(inputs,saved=saved)
        new_reference=saved.get('pred_normalized') is None
        original = dict(row_cap=128 if request['family']!='DIRECT_G' else None,
                        generated_new_accuracy_reference=new_reference,
                        generated_new_RAW_reference=new_reference and request['family']=='RAW')
        if saved.get('pred_normalized') is None:
            saved['pred_normalized'], original['validation'] = full_validation(route,inputs,request,
                row_cap=original['row_cap'],allow_new_reference=True)
        original['prediction_fingerprint'] = fingerprint(saved['pred_normalized'])
        original['metrics']=independent_metrics(saved)
        prediction, validation = full_validation(route,inputs,request)
        for key in ('MSE','MAE'):
            if not math.isclose(validation['metrics'][key],original['metrics'][key],abs_tol=ATOL,rel_tol=RTOL):
                raise ValueError('Full V original-reference metric parity failed')
        physical, observed = _windows(inputs['task'],saved['origins'][:8])
        trials = []
        release_validation_cache(device)
        reset_peaks(device)
        for phase, repetitions in (('warmup',10),('timed',20)):
            for iteration in range(repetitions):
                synchronize(device)
                if phase=='timed': reset_peaks(device)
                start = time.perf_counter()
                output = route.predict_physical(physical,observed)
                synchronize(device)
                elapsed = time.perf_counter()-start
                peaks = memory(device)
                if output.shape != (8,48,len(inputs['task']['columns'])) or not np.isfinite(output).all() or elapsed<=0:
                    raise ValueError('Timed full-path prediction or elapsed time is invalid')
                work = _calls(route,request,len(inputs['task']['columns']),8,validation['row_cap'])
                normalized = (output-np.asarray(route.mean,np.float64))/np.asarray(route.std,np.float64)
                if not np.allclose(normalized,prediction[:8],atol=ATOL,rtol=RTOL):
                    raise ValueError('Timed online forecast differs from validated whole V')
                trials.append(dict(phase=phase,iteration=iteration,batch_size=8,whole_ms=elapsed*1000,
                    F_calls=work['calls'],F_streams=work['streams'],row_cap=work['row_cap'],**peaks))
        after = _states(route,inputs,request,device)
        if before != after: raise ValueError('Loaded inference tensor state changed')
        timed = [r for r in trials if r['phase']=='timed']
        q25,p50,q75,p95 = np.quantile([r['whole_ms'] for r in timed],[.25,.5,.75,.95])
        summary = dict(p50_ms=float(p50),q25_ms=float(q25),q75_ms=float(q75),iqr_ms=float(q75-q25),p95_ms=float(p95),
            repetitions=20,origins_per_second_at_median=8000/float(p50),
            **{key:max(r[key] for r in timed) for key in ('peak_allocated_MiB','peak_reserved_MiB','RSS_MiB')})
        receipt = dict(status='COMPLETE',dataset=request['dataset'],family=request['family'],run_id=request['run_id'],
            seed=request['seed'],K=request['K'],C=len(inputs['task']['columns']),L=512,H=48,batch_size=8,
            device=str(device),pid=os.getpid(),precision='FP32; TF32 off; AMP off',row_cap=validation['row_cap'],
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            warmups=10,timed_repetitions=20,TS2_or_prediction_cache_used=False,model_load_seconds=load_seconds,
            loaded_baseline=loaded,original_reference=original,validation=validation,summary=summary,trials=trials,
            states_before=before,states_after=after,wall_seconds=time.perf_counter()-started,
            F_state_sha256=inputs['expected_f_sha'] if request['family']!='DIRECT_G' else 'NO_FOUNDATION_USED',
            checkpoint_sha256=getattr(route,'artifact_sha256',None),
            validation_allocator_cache_released_before_warmup=True,
            timing_scope='physical history+bool mask -> local causal fill -> fixed A60 scale -> E/P/F/D -> inverse -> CPU return',
            excluded=['window slicing','model loading','full-V parity','hashing','metric scoring'],
            memory_scope='per-trial CUDA allocated/reserved including loaded resident weights; sampled process RSS',
            quantile_scope='20 repeated B8 calls on the first fixed V group; not service-tail guarantees')
        arrays = dict(saved,pred_normalized=prediction,reference_pred_normalized=saved['pred_normalized'])
        return receipt, arrays
    finally:
        if hasattr(route,'close'): route.close()


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)


def _output_roots(report_dir, cache_dir):
    report,cache = Path(report_dir).resolve(),Path(cache_dir).resolve()
    if (NAME not in report.parts or NAME not in cache.parts or report==cache
            or report.is_relative_to(cache) or cache.is_relative_to(report)):
        raise ValueError('Use separate report/cache slots inside this new campaign namespace')
    return report,cache


def execute_request(request, report_dir, cache_dir):
    request=validate_request(request)
    report,cache=_output_roots(report_dir,cache_dir)
    identity=request['run_id']+'_B8'
    receipt_path=report/'costs'/(identity+'.json')
    validation_path=report/'cost_validation'/(identity+'.json')
    prediction_path=cache/'costs'/(identity+'_predictions.npz')
    if any(path.exists() for path in (receipt_path,validation_path,prediction_path)):
        raise FileExistsError('Preserve the existing cost attempt and choose fresh slots')
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        factory_name,function=request['factory'].split(':')
        factory_module=importlib.import_module('research.'+NAME+'.'+factory_name)
        factory=getattr(factory_module,function)
        inputs=factory(request,report,cache)
        receipt,arrays=measure_inputs(request,inputs)
        prediction_path.parent.mkdir(parents=True,exist_ok=True)
        with prediction_path.open('xb') as stream: np.savez_compressed(stream,**arrays)
        receipt.update(prediction_reference='cache://costs/'+prediction_path.name,
            prediction_file_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            factory_source_sha256=hashlib.sha256(Path(factory_module.__file__).read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        _write_json(validation_path,receipt['validation'])
        _write_json(receipt_path,receipt)
        return receipt
    except Exception as error:
        failure=dict(status='FAILED',run_id=request['run_id'],dataset=request['dataset'],family=request['family'],
            error_type=type(error).__name__,detail='See preserved private worker log; no validation PASS claimed')
        if not validation_path.exists(): _write_json(validation_path,failure)
        if not receipt_path.exists(): _write_json(receipt_path,failure)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--report-dir',required=True,type=Path)
    parser.add_argument('--cache-dir',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    request=validate_request(json.loads(args.request.read_text(encoding='utf-8-sig')))
    if args.worker:
        receipt=execute_request(request,args.report_dir,args.cache_dir)
        print(json.dumps(dict(status=receipt['status'],run_id=receipt['run_id'],summary=receipt['summary'])),flush=True)
        return 0
    report,cache=_output_roots(args.report_dir,args.cache_dir)
    log=cache/'cost_logs'/(request['run_id']+'_'+str(time.time_ns())+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with log.open('xb') as stream:
        process=subprocess.run([sys.executable,'-m','research.'+NAME+'.costs','--worker',
            '--request',str(args.request.resolve()),'--report-dir',str(report),'--cache-dir',str(cache)],
            cwd=Path(__file__).resolve().parents[2],env=env,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps(dict(exit_code=process.returncode,private_log_reference='cache://cost_logs/'+log.name)),flush=True)
    return process.returncode


if __name__=='__main__':
    raise SystemExit(main())
