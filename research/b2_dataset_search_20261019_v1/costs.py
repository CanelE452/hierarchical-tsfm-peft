"""Immutable B8 row128/common-cap facets using the existing measurement core."""
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
import torch

from research.memory_budget_20261018_v1 import worker as core
from research.b2_dataset_extension_20261018_v1.costs import _cap, _states


NAME = 'b2_dataset_search_20261019_v1'
ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 1e-5, 1e-4


def sha(path):
    return core._sha(path)


def k_grid(channels):
    q = math.ceil(channels / 4)
    return sorted({max(1, math.ceil(q / 2)), q, min(channels, 2 * q)})


def validate_request(request):
    value = dict(request)
    for key in ('dataset', 'run_id'):
        if not isinstance(value.get(key), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', value[key]):
            raise ValueError('Unsafe cost identity: ' + key)
    if value.get('family') not in ('RAW', 'B2', 'DIRECT_G'):
        raise ValueError('Unregistered model family')
    if value.get('row_policy') not in ('DEFAULT128', 'COMMON_CAP'):
        raise ValueError('Unregistered row policy')
    if type(value.get('K')) is not int or value['K'] < 1 or value.get('seed') not in (92601, 92602):
        raise ValueError('Fixed positive K and registered seed required')
    if value.get('factory', 'train:cost_inputs') != 'train:cost_inputs':
        raise ValueError('The new campaign factory is required')
    if value.get('batch_size', 8) != 8 or value.get('device', 'cuda') != 'cuda':
        raise ValueError('Both facets require the same CUDA B8 contract')
    for key in ('config_path', 'prefix_path', 'model_dir'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError('Missing private input path: ' + key)
    value.update(batch_size=8, device='cuda', factory='train:cost_inputs')
    return value


def policy_work(request, channels, batch):
    q = math.ceil(channels / 4)
    width = channels if request['family'] == 'RAW' else request['K'] if request['family'] == 'B2' else 0
    requested = (128 if request['row_policy'] == 'DEFAULT128' else 8 * q) if width else None
    streams = batch * width
    effective = min(requested, streams) if streams else None
    rows = [min(effective, streams-start) for start in range(0, streams, effective)] if streams else []
    return dict(q=q, width=width, streams=streams, requested_row_cap=requested,
                effective_row_cap=effective, calls=len(rows), row_sizes=rows)


def validate_cohort(route, inputs, request):
    task, saved = inputs['task'], inputs['saved']
    channels = len(task['columns'])
    q = math.ceil(channels/4)
    if (task['L'], task['H']) != (512, 48) or task.get('dataset') != request['dataset']:
        raise ValueError('Registered dataset/L512/H48 differs')
    if request['K'] not in k_grid(channels) or (request['family'] != 'B2' and request['K'] != q):
        raise ValueError('K is outside the fixed grid or baseline comparison q')
    config = inputs.get('source_ctx', {}).get('config', {})
    for key in ('K_grid', 'k_grid'):
        if key in config and list(config[key]) != k_grid(channels):
            raise ValueError('Configuration width grid differs from the protocol')
    lo, hi = task['split_ranges']['V_select']
    origins = np.asarray(saved['origins'])
    expected = np.arange(max(lo, 512), hi-48+1, 48, dtype=np.int64)
    if (len(expected) < 8 or origins.dtype.kind not in 'iu' or origins.shape != expected.shape
            or not np.array_equal(origins, expected) or hi > len(task['physical'])):
        raise ValueError('Full registered V origin grid/order differs')
    physical, observed = np.asarray(task['physical']), np.asarray(task['observed'])
    if (physical.ndim != 2 or physical.shape[1] != channels or observed.shape != physical.shape
            or observed.dtype != np.bool_):
        raise ValueError('Physical input/observation schema differs')
    if not np.array_equal(np.asarray(saved['channel_ids']).astype(str), np.asarray(task['columns']).astype(str)):
        raise ValueError('Saved channel order differs')
    for key in ('mean', 'std'):
        value = np.asarray(getattr(route, key))
        if value.shape != (channels,) or not np.isfinite(value).all() or not np.array_equal(value, saved[key]):
            raise ValueError('Fixed A60 scaler differs')
    if np.any(route.std <= 0):
        raise ValueError('Nonpositive fixed scaler')
    target, mask = np.asarray(saved['target_normalized']), np.asarray(saved['observed'])
    if target.shape != (len(origins), 48, channels) or mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError('Future target/mask schema differs')
    rows = origins[:, None] + np.arange(48)[None]
    true_mask = observed[rows]
    target_physical = physical[rows]
    normalized = (target_physical-route.mean)/route.std
    if not np.array_equal(true_mask, mask) or not np.allclose(normalized[mask], target[mask], atol=ATOL, rtol=RTOL):
        raise ValueError('Target values or observed mask differ from the physical cohort')
    return origins, channels


def _original_reference(route, inputs, request):
    origins, channels = validate_cohort(route, inputs, request)
    width = channels if request['family'] == 'RAW' else request['K'] if request['family'] == 'B2' else 0
    cap = 128 if width else None
    _cap(route, cap)
    parts, counts = [], []
    for start in range(0, len(origins), 8):
        physical, observed = core._windows(inputs['task'], origins[start:start+8])
        prediction = route.predict_physical(physical, observed)
        if prediction.shape != (len(physical),48,channels) or not np.isfinite(prediction).all():
            raise ValueError('Original row128 physical forecast is invalid')
        counts.append(core._check_calls(route,len(physical),width,cap)['F_calls'])
        parts.append((prediction-route.mean)/route.std)
    actual = np.concatenate(parts)
    expected = inputs['saved'].get('pred_normalized')
    if expected is None and request['family'] == 'B2':
        raise ValueError('B2 requires its exact original trained prediction binding')
    if expected is not None and (np.asarray(expected).shape != actual.shape
            or not np.allclose(actual,expected,atol=ATOL,rtol=RTOL)):
        raise ValueError('Original row128 trained prediction parity failed')
    saved = dict(inputs['saved'],pred_normalized=actual if expected is None else expected)
    metrics = core.independent_metrics(saved)
    candidate = inputs.get('candidate', {})
    for key in ('MSE','MAE'):
        if key in candidate and not math.isclose(float(candidate[key]),metrics[key],rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Bound accuracy metric differs from the independent FP64 rescore')
    return saved, dict(status='PASS',row_cap=cap,generated_new=expected is None,
        full_V_origins=len(origins),F_calls=counts,metrics=metrics,
        actual_prediction_fingerprint=core._fingerprint(actual),
        canonical_prediction_fingerprint=core._fingerprint(saved['pred_normalized']),
        max_abs=float(np.max(np.abs(actual-expected))) if expected is not None else 0.,atol=ATOL,rtol=RTOL)


class _ObservedRoute:
    def __init__(self, route, inputs, request, device, chunks):
        self.route,self.inputs,self.request,self.device = route,inputs,request,device
        self.chunks=chunks
        self.predictions=[]
        self.after=None
        self.closed=False

    def __getattr__(self, name):
        return getattr(self.route,name)

    def predict_physical(self, physical, observed):
        prediction=self.route.predict_physical(physical,observed)
        if len(self.predictions)<self.chunks:
            self.predictions.append(((prediction-self.route.mean)/self.route.std).copy())
        return prediction

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.after=_states(self.route,self.inputs,self.request,self.device)
        finally:self.route.close()


def measure_inputs(request, inputs, *, device='cuda', sampler_factory=core.ProcessSampler):
    request=validate_request(request)
    load_started=time.perf_counter()
    route=inputs.get('route_loader',inputs.get('loader'))()
    probe=None
    try:
        if hasattr(route,'activate'):route.activate()
        core._sync(device)
        load_seconds=time.perf_counter()-load_started
        loaded_before_reference=core._memory(device)
        before=_states(route,inputs,request,device)
        saved,original=_original_reference(route,inputs,request)
        origins,channels=validate_cohort(route,dict(inputs,saved=saved),request)
        work=policy_work(request,channels,8)
        candidate=inputs.get('candidate',{})
        model_k=candidate.get('K',work['width'])
        public_candidate={key:candidate[key] for key in ('bundle_binding','prediction_binding','prediction_dtype') if key in candidate}
        public_candidate.update(run_id=request['run_id'],family=request['family'],dataset=request['dataset'],seed=request['seed'],
            C=channels,K=model_k,L=512,H=48,**{key:original['metrics'][key] for key in ('MSE','MAE')})
        public_candidate.setdefault('prediction_binding',dict(reference='array://generated_row128',
            sha256=core._fingerprint(saved['pred_normalized'])['sha256']))
        probe=_ObservedRoute(route,inputs,request,device,math.ceil(len(origins)/8))
        active=dict(inputs,candidate=public_candidate,saved=saved,loader=lambda:probe,
                    registered_rows=(128,8*work['q']))
        facet=request['run_id']+'_'+request['row_policy']+'_B8'
        core_request=dict(asset_root='NEW_NAMESPACE_SUPPLIED_INPUTS_ONLY',accuracy_run_id=request['run_id'],
            batch_size=8,row_microbatch=work['requested_row_cap'],stage='confirm',facet_id=facet,
            protocol_sha256=request.get('protocol_sha256','0'*64))
        measured=core.evaluate_facet(core_request,active,device=device,sampler_factory=sampler_factory)
        if before!=probe.after:raise ValueError('Loaded model/F tensor states changed across inference')
        predictions=np.concatenate(probe.predictions)
        if core._fingerprint(predictions)!=measured['parity']['output_fingerprints']['regenerated_normalized']:
            raise ValueError('Captured whole V prediction differs from independent measurement parity')
        measured['summary']['model_load_seconds']=load_seconds
        measured.update(schema='b2_dataset_search_cost_v1',dataset=request['dataset'],run_id=request['run_id'],
            family=request['family'],seed=request['seed'],K=request['K'],model_K=model_k,q=work['q'],C=channels,
            L=512,H=48,batch_size=8,row_policy=request['row_policy'],actual_F_width=work['width'],
            requested_row_cap=work['requested_row_cap'],effective_row_cap=work['effective_row_cap'],
            expected_work=work,original_reference=original,states_before=before,states_after=probe.after,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            loaded_before_reference=loaded_before_reference,
            baseline_scope='Measurement-core baseline after row128 full-V verification and empty_cache; first cold loaded baseline is separate',
            state_check='FP32 eval; frozen F; exact pre/post tensor-state equality',
            policy_scope='COMMON_CAP is 8*q, fixed across every K and Raw within this dataset')
        arrays=dict(saved,pred_normalized=predictions,reference_pred_normalized=saved['pred_normalized'])
        return measured,arrays
    finally:
        if probe is None:route.close()
        elif not probe.closed:probe.close()


def output_paths(request, report_dir, cache_dir):
    report,cache=Path(report_dir).resolve(),Path(cache_dir).resolve()
    if (NAME not in report.parts or NAME not in cache.parts or report==cache
            or report.is_relative_to(cache) or cache.is_relative_to(report)):
        raise ValueError('Separate output roots inside the new namespace are required')
    identity=request['run_id']+'_'+request['row_policy']+'_B8'
    return dict(receipt=report/'costs'/(identity+'.json'),validation=report/'cost_validation'/(identity+'.json'),
                prediction=cache/'costs'/(identity+'_predictions.npz'))


def verify_sources(sources, repo=ROOT):
    repo=Path(repo).resolve()
    for relative,expected in sources.items():
        path=(repo/relative).resolve()
        if not path.is_relative_to(repo) or sha(path)!=expected:
            raise ValueError('Producer source changed: '+relative)


def producer_sources(factory_module, inputs):
    paths=[Path(__file__),Path(factory_module.__file__),Path(core.__file__),
        ROOT/'research/b2_dataset_extension_20261018_v1/costs.py']
    pins={path.resolve().relative_to(ROOT).as_posix():sha(path) for path in paths}
    pins.update(inputs.get('source_ctx',{}).get('scientific_source_hashes',{}))
    verify_sources(pins)
    return pins


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)


def _identity(request,sources,inputs=None):
    inputs=inputs or {}
    return dict(**{key:request[key] for key in ('dataset','run_id','family','seed','K','row_policy')},
        config_sha256=sha(request['config_path']),prefix_sha256=sha(request['prefix_path']),
        candidate_identity_sha256=hashlib.sha256(json.dumps(inputs.get('candidate',{}),sort_keys=True,allow_nan=False).encode()).hexdigest(),
        data_sha256=inputs.get('task',{}).get('data_sha256'),
        expected_f_sha256=inputs.get('expected_f_sha'),expected_f_params=inputs.get('expected_f_params'),
        protocol_sha256=request.get('protocol_sha256'),
        private_request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
        producer_sources=sources,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def _reuse(paths,identity):
    if not paths['receipt'].exists():
        if paths['validation'].exists() or paths['prediction'].exists():
            raise FileExistsError('Preserve the partial cost attempt')
        return None
    value=json.loads(paths['receipt'].read_text(encoding='utf-8'))
    if value.get('status')!='COMPLETE' or any(value.get(key)!=expected for key,expected in identity.items()):
        raise ValueError('Existing cost contract/source/environment differs; preserve this attempt')
    validation=json.loads(paths['validation'].read_text(encoding='utf-8'))
    if (validation!=value['parity'] or validation.get('status')!='PASS'
            or sha(paths['prediction'])!=value['prediction_file_sha256']):
        raise ValueError('Existing cost prediction/validation binding changed')
    return value


def execute_request(request, report_dir, cache_dir):
    request=validate_request(request)
    paths=output_paths(request,report_dir,cache_dir)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    identity={key:request[key] for key in ('dataset','run_id','family','seed','K','row_policy')}
    try:
        protocol=sha(ROOT/'reports'/NAME/'PROTOCOL.md')
        if request.get('protocol_sha256',protocol)!=protocol:
            raise ValueError('Cost protocol binding differs')
        request['protocol_sha256']=protocol
        factory=importlib.import_module('research.'+NAME+'.train')
        inputs=factory.cost_inputs(request,Path(report_dir).resolve(),Path(cache_dir).resolve())
        sources=producer_sources(factory,inputs)
        identity=_identity(request,sources,inputs)
        existing=_reuse(paths,identity)
        if existing is not None:return existing
        result,arrays=measure_inputs(request,inputs)
        verify_sources(sources)
        paths['prediction'].parent.mkdir(parents=True,exist_ok=True)
        with paths['prediction'].open('xb') as stream:np.savez_compressed(stream,**arrays)
        result.update(identity,prediction_file_sha256=sha(paths['prediction']),
            prediction_reference='cache://costs/'+paths['prediction'].name)
        _json(paths['validation'],result['parity'])
        _json(paths['receipt'],result)
        return result
    except Exception as error:
        failure=dict(status='FAILED',**identity,error_type=type(error).__name__,
            detail='Inspect the private worker log; preserve this attempt')
        if not paths['validation'].exists():_json(paths['validation'],failure)
        if not paths['receipt'].exists():_json(paths['receipt'],failure)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--report-dir',required=True,type=Path)
    parser.add_argument('--cache-dir',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if NAME not in args.request.resolve().parts:
        raise ValueError('The private cost request must be inside the new namespace')
    request=validate_request(json.loads(args.request.read_text(encoding='utf-8-sig')))
    if args.worker:
        result=execute_request(request,args.report_dir,args.cache_dir)
        print(json.dumps(dict(status=result['status'],run_id=result['run_id'],row_policy=result['row_policy'])),flush=True)
        return 0
    output_paths(request,args.report_dir,args.cache_dir)
    log=Path(args.cache_dir)/'cost_logs'/(request['run_id']+'_'+request['row_policy']+'_'+str(time.time_ns())+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with log.open('xb') as stream:
        process=subprocess.run([sys.executable,'-m','research.'+NAME+'.costs','--worker',
            '--request',str(args.request.resolve()),'--report-dir',str(Path(args.report_dir).resolve()),
            '--cache-dir',str(Path(args.cache_dir).resolve())],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps(dict(exit_code=process.returncode,log_reference='cache://cost_logs/'+log.name)),flush=True)
    return process.returncode


if __name__=='__main__':
    raise SystemExit(main())
