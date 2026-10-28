"""One immutable accuracy model and one memory facet per OS process."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback
import uuid

import numpy as np
import psutil
import torch


ATOL, RTOL = 1e-5, 1e-4
F_PARAMETER_COUNT = 47_718_016
REPO = Path(__file__).resolve().parents[2]
STAGES = {'screen':(2,3), 'confirm':(10,20)}
FAMILIES = {'B2','B0','RAW','LORA','DIRECT_G','RAW_RIDGE'}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        return
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def validate_request(value):
    required = {'asset_root','accuracy_run_id','batch_size','row_microbatch',
                'stage','protocol_sha256','facet_id'}
    if not isinstance(value,dict) or not required.issubset(value):
        raise ValueError('Incomplete memory facet request')
    if type(value['batch_size']) is not int or value['batch_size'] not in (1,8):
        raise ValueError('Registered prediction batches are integer 1 or 8')
    chunk = value['row_microbatch']
    if chunk is not None and (type(chunk) is not int or chunk<1):
        raise ValueError('F row microbatch must be a positive integer or null')
    if value['stage'] not in STAGES:
        raise ValueError('Registered stages are screen and confirm')
    for key in ('accuracy_run_id','facet_id'):
        if not isinstance(value[key],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}',value[key]):
            raise ValueError('Unsafe '+key)
    if not isinstance(value['protocol_sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',value['protocol_sha256']):
        raise ValueError('A sealed protocol SHA256 is required')
    return dict(value)


def select_accuracy(rows, run_id):
    selected = [dict(row) for row in rows if row.get('run_id')==run_id]
    if len(selected)!=1:
        raise ValueError('Exact accuracy run must occur once')
    row = selected[0]
    if (row.get('final') is not True or row.get('revin') is not False
            or row.get('family') not in FAMILIES or row.get('evaluation')!='V_select'
            or row.get('independent_FP64')!='PASS'
            or row.get('status') not in ('COMPLETE','REUSED_VERIFIED','HISTORICAL_VERIFIED')
            or not row.get('prediction_binding')
            or (row['family']!='RAW' and not row.get('bundle_binding'))):
        raise ValueError('Memory facets require an exact verified final OFF V accuracy row')
    if row['family'] in ('B0','LORA','RAW_RIDGE') and row.get('dataset')!='jena':
        raise ValueError('Unregistered dataset/family pair')
    if row['family']=='DIRECT_G' and row.get('dataset')!='pems32':
        raise ValueError('DirectG is registered only for PeMS32')
    return row


def validate_parent_binding(candidate, request):
    identity=hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if request.get('expected_candidate_sha256') not in (None,identity):
        raise ValueError('Canonical parent accuracy row differs from the preregistered identity')
    if request.get('parent_prediction_sha256') not in (None,candidate['prediction_binding']['sha256']):
        raise ValueError('Parent accuracy prediction SHA differs from preregistration')
    expected=request.get('parent_checkpoint_sha256')
    bundle=candidate.get('bundle_binding')
    if bundle and expected not in (None,bundle['sha256']):
        raise ValueError('Parent checkpoint SHA differs from preregistration')
    if not bundle and expected not in (None,'NO_SERIALIZED_PARENT'):
        raise ValueError('Raw has no serialized parent checkpoint; F-state identity is checked separately')
    return identity


def independent_metrics(values):
    pred,target,mask = (np.asarray(values[key]) for key in
                        ('pred_normalized','target_normalized','observed'))
    if pred.ndim!=3 or pred.shape!=target.shape or pred.shape!=mask.shape or mask.dtype!=np.bool_:
        raise ValueError('Aligned predictions/targets and boolean observation mask required')
    counts = mask.sum((0,1),dtype=np.int64)
    if not len(counts) or (counts==0).any():
        raise ValueError('Every scored channel needs observed targets')
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Nonfinite prediction or observed target')
    error = np.where(mask,pred.astype(np.float64)-target.astype(np.float64),0.)
    sse = (error*error).sum((0,1),dtype=np.float64)
    sae = np.abs(error).sum((0,1),dtype=np.float64)
    return dict(MSE=math.fsum((sse/counts).tolist())/len(counts),
                MAE=math.fsum((sae/counts).tolist())/len(counts),observed=int(counts.sum()),
                channel_sse=sse.tolist(),channel_sae=sae.tolist(),channel_counts=counts.tolist(),
                accumulation_dtype='float64',aggregation='equal-channel observed macro')


def _ridge_loader(candidate, task, source_ctx):
    from research.b2_revin_20261018_v1 import contracts
    path = contracts.resolve_binding(candidate['bundle_binding'],source_ctx)
    cache_root = path.parent
    spec = dict(family='RAW_RIDGE',run_id=candidate['run_id'],
                artifact_cache_relative_path=path.name,artifact_sha256=candidate['bundle_binding']['sha256'],
                canonical_fit_end=task['split_ranges']['A_stop'][1])
    def load():
        contracts.resolve_binding(candidate['bundle_binding'],source_ctx)
        return source_ctx['modules']['deployment'].Route(spec,task,cache_root,device='cpu')
    return load


def prepare_inputs(request):
    from research.b2_revin_20261018_v1 import audit, contracts, costs, run
    source_ctx = run.context(request['asset_root'])
    inventory = contracts.read(Path(source_ctx['cache'])/'LEGACY_INVENTORY.json')
    source_bindings={}
    if request.get('candidate_file'):
        rows_path,source_ctx,source_bindings=_phase_b_source(request,source_ctx)
    else:
        rows_path = Path(source_ctx['cache'])/'VERIFIED_ROWS.json'
    candidate = select_accuracy(contracts.read(rows_path),request['accuracy_run_id'])
    parent_identity=validate_parent_binding(candidate,request)
    task = source_ctx['task_loader'](source_ctx,source_ctx['modules'],candidate['dataset'])
    saved = audit._canonical_arrays(costs._saved(candidate,source_ctx),task,source_ctx['modules'])
    loader = (_ridge_loader(candidate,task,source_ctx) if candidate['family']=='RAW_RIDGE'
              else costs.route_loader(candidate,task,source_ctx,inventory))
    for key in ('C','L','H'):
        actual = len(task['columns']) if key=='C' else task[key]
        if int(candidate[key])!=actual:
            raise ValueError('Accuracy task axis differs: '+key)
    if candidate.get('data_sha256',candidate.get('data_sha'))!=task['data_sha256']:
        raise ValueError('Accuracy task data SHA differs')
    return dict(candidate=candidate,task=task,saved=saved,loader=loader,source_ctx=source_ctx,
                expected_f_sha=source_ctx['modules']['legacy'].F_SHA,
                expected_f_params=F_PARAMETER_COUNT,
                registered_rows=(128,request['batch_size']*(6 if candidate['dataset']=='jena' else 8),1),
                source_bindings=dict(source_bindings,parent_accuracy_row_sha256=parent_identity,
                    verified_rows_sha256=contracts.sha(rows_path),
                    scientific_sources=source_ctx['scientific_source_hashes'],
                    runtime_source_sha256=_runtime_sources(source_ctx)))


def _runtime_sources(ctx):
    repo=Path(ctx['repo']).resolve()
    paths={Path(value.__file__).resolve() for value in ctx['modules'].values()
           if getattr(value,'__file__',None)}
    paths.update(repo/relative for relative in (
        'research/memory_budget_20261018_v1/worker.py',
        'research/b2_revin_20261018_v1/costs.py',
        'research/b2_revin_20261018_v1/audit.py',
        'research/adapter_improvements_20261018_v1/audit.py',
        'research/adapter_diagnostics_20261017_v2/existing_jobs.py',
        'research/informs_broad_validation_20261015_v1/gpu_runtime.py',
        'research/informs_broad_validation_20261015_v1/networks.py',
        'research/informs_broad_validation_20261015_v1/resources.py',
        'research/temporal_frontend_campaign_20261015_v1/runtime.py'))
    paths.update(repo/relative for relative in ctx['scientific_source_hashes'])
    return {path.relative_to(repo).as_posix():_sha(path) for path in sorted(paths)
            if path.is_relative_to(repo) and path.is_file()}


def _phase_b_source(request, parent_ctx):
    from . import common
    paths=common.paths(parent_ctx['repo'])
    rows_path=Path(request['candidate_file']).resolve()
    if rows_path!=(paths['cache']/'B_PHASE_ROWS.json').resolve():
        raise ValueError('Phase B candidate file must be the registered new cache artifact')
    if (not isinstance(request.get('candidate_file_sha256'),str)
            or not re.fullmatch(r'[a-f0-9]{64}',request['candidate_file_sha256'])
            or request['candidate_file_sha256']!=common.sha(rows_path)):
        raise ValueError('Phase B candidate file SHA changed')
    identity_path=paths['report']/'B_TRAINING_IDENTITY.json'
    identity=common.read(identity_path)
    ctx=dict(parent_ctx,**paths,scientific_source_hashes=identity['scientific_source_hashes'])
    gate_path=common.resolve_binding(identity['gate_binding'],ctx)
    if gate_path!=(paths['report']/'PHASE_B_DECISION.json').resolve():
        raise ValueError('Phase B training gate path differs')
    gate=common.read(gate_path)
    if (gate.get('status')!='PROCEED' or gate.get('eligible') is not True
            or gate.get('phase_a_all_confirmed') is not True
            or gate.get('decision_identity')!=common.fingerprint({key:value for key,value in gate.items() if key!='decision_identity'})):
        raise ValueError('Phase B inference requires a sealed successful Phase A gate')
    protocol_path=common.resolve_binding(gate['protocol_binding'],ctx)
    if (protocol_path!=(paths['report']/'PROTOCOL.md').resolve()
            or common.sha(protocol_path)!=request['protocol_sha256']
            or identity['protocol_binding']!=gate['protocol_binding']):
        raise ValueError('Phase B inference protocol differs from training gate')
    selection_path=paths['report']/'B_SELECTIONS_SEALED.json'
    selection=common.read(selection_path)
    expected={f'B2_jena_K{k}_OFF_s{seed}' for k in (3,12) for seed in (92601,92602)}
    if (selection.get('status')!='SEALED' or selection.get('V_used') is not False
            or selection.get('selection_split')!='A_stop' or selection.get('cap')!=3
            or selection.get('all_four_searches_complete') is not True
            or set(selection.get('matched_arms',[]))!=expected
            or selection.get('phase_a_gate_binding')!=identity['gate_binding']):
        raise ValueError('Phase B all-search selection seal is incomplete')
    summary=common.read(paths['report']/'B_TRAINING_SUMMARY.json')
    if summary.get('status')!='COMPLETE' or summary.get('trained_models')!=4:
        raise ValueError('All four Phase B models must finish before inference')
    if (common.resolve_binding(summary['source_identity_binding'],ctx)!=identity_path.resolve()
            or common.resolve_binding(summary['selection_binding'],ctx)!=selection_path.resolve()):
        raise ValueError('Phase B training summary producer/selection binding differs')
    rows=common.read(rows_path)
    if len(rows)!=4 or {row.get('job_id') for row in rows}!=expected:
        raise ValueError('Phase B inference requires exactly four registered trained models')
    for row in rows:
        if (row.get('source_context')!='MEMORY_BUDGET_PHASE_B'
                or row.get('campaign_namespace')!=common.NAME or row.get('K') not in (3,12)
                or row.get('family')!='B2' or row.get('revin') is not False or row.get('cap')!=3):
            raise ValueError('Phase B candidate provenance or model recipe differs')
        if common.resolve_binding(row['selection_decision_binding'],ctx)!=selection_path.resolve():
            raise ValueError('Phase B candidate selection binding differs')
        common.resolve_binding(row['selection_receipt_binding'],ctx)
        matches=[item for item in summary.get('rows',[]) if item.get('run_id')==row.get('run_id')]
        if len(matches)!=1 or any(matches[0].get(key)!=row.get(key) for key in
                ('job_id','K','seed','bundle_binding','prediction_binding','selection_receipt_binding')):
            raise ValueError('Phase B cost parent differs from the completed training summary')
    return rows_path,ctx,dict(phase_b_gate_binding=identity['gate_binding'],
        phase_b_training_identity_binding=common.binding(identity_path,ctx),
        phase_b_selection_binding=common.binding(selection_path,ctx))


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def _memory(device):
    if not str(device).startswith('cuda'):
        return dict(allocated_MiB=None,reserved_MiB=None,peak_allocated_MiB=None,peak_reserved_MiB=None)
    return dict(allocated_MiB=torch.cuda.memory_allocated(device)/1024**2,
                reserved_MiB=torch.cuda.memory_reserved(device)/1024**2,
                peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2,
                peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/1024**2)


def _driver_fallback():
    try:
        options = dict(capture_output=True,text=True,timeout=10,check=False)
        if os.name=='nt':
            options['creationflags']=subprocess.CREATE_NO_WINDOW
        result = subprocess.run(['nvidia-smi','--query-gpu=driver_version,name',
                                 '--format=csv,noheader'],**options)
        return result.stdout.strip() if result.returncode==0 else 'UNAVAILABLE'
    except (OSError,subprocess.SubprocessError):
        return 'UNAVAILABLE'


class ProcessSampler:
    """Sample this PID only. Sampling never certifies an instantaneous hard cap."""
    def __init__(self, device, interval_seconds=.05):
        self.device,self.interval = str(device),interval_seconds
        self.process = psutil.Process()
        self.pid = self.process.pid
        self.phase = 'warmup'
        self.samples = []
        self.latest_gpu_mib = None
        self.process_gpu_status = 'NOT_APPLICABLE_CPU'
        self.process_gpu_reason = 'CPU facet has no GPU memory coordinate'
        self.driver = 'NOT_APPLICABLE_CPU'
        self._read_gpu,self._nvml = None,None
        self._stop = threading.Event()
        self._thread = None
        if self.device.startswith('cuda'):
            self._configure_nvml()

    def _configure_nvml(self):
        self.process_gpu_status = 'UNAVAILABLE'
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            driver = pynvml.nvmlSystemGetDriverVersion()
            self.driver = driver.decode() if isinstance(driver,bytes) else str(driver)
            # These experiments use one physical GPU; refuse a hidden remapping.
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible not in (None,'','0'):
                raise RuntimeError('CUDA_VISIBLE_DEVICES remapping has no registered NVML device binding')
            index = int(self.device.split(':')[1]) if ':' in self.device else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            getters = []
            for name in ('nvmlDeviceGetComputeRunningProcesses','nvmlDeviceGetGraphicsRunningProcesses'):
                getter = getattr(pynvml,name,None)
                if getter is not None:
                    try:
                        getter(handle)
                        getters.append(getter)
                    except pynvml.NVMLError:
                        pass
            if not getters:
                raise RuntimeError('Driver does not expose per-PID GPU memory')
            def read():
                values=[]
                for getter in getters:
                    for process in getter(handle):
                        if int(process.pid)==self.pid:
                            memory=getattr(process,'usedGpuMemory',None)
                            if memory is not None and 0<=int(memory)<2**60:
                                values.append(int(memory)/1024**2)
                if not values:
                    raise RuntimeError('This PID GPU bytes unavailable (including WDDM unavailable-memory fields)')
                return max(values)
            self.latest_gpu_mib = read()
            self._read_gpu=read
            self.process_gpu_status='AVAILABLE'
            self.process_gpu_reason=None
        except (ImportError,RuntimeError,ValueError,OSError) as error:
            self.process_gpu_reason=type(error).__name__+': '+str(error)
        except Exception as error:
            self.process_gpu_reason=type(error).__name__+': driver per-PID query unavailable'
        if self.driver=='NOT_APPLICABLE_CPU':
            self.driver=_driver_fallback()

    def _sample(self):
        rss=self.process.memory_info().rss/1024**2
        gpu=None
        if self._read_gpu is not None:
            try:
                gpu=self._read_gpu()
                self.latest_gpu_mib=gpu
            except Exception as error:
                self.process_gpu_status='UNAVAILABLE'
                self.process_gpu_reason=type(error).__name__+': per-PID sampling interrupted'
                self._read_gpu=None
        self.samples.append(dict(phase=self.phase,elapsed_seconds=time.perf_counter()-self.started,
                                 CPU_RSS_MiB=rss,process_GPU_MiB=gpu))

    def start(self):
        self.started=time.perf_counter()
        self._sample()
        def observe():
            while not self._stop.wait(self.interval):
                self._sample()
        self._thread=threading.Thread(target=observe,name='memory-facet-pid-sampler',daemon=True)
        self._thread.start()

    def set_phase(self,phase):
        self.phase=phase

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        gpu=[sample['process_GPU_MiB'] for sample in self.samples if sample['process_GPU_MiB'] is not None]
        result=dict(CPU_RSS_sampled_peak_MiB=max(row['CPU_RSS_MiB'] for row in self.samples),
                    process_GPU_sampled_peak_MiB=max(gpu) if gpu else None,
                    process_GPU_sampling_status=self.process_gpu_status,
                    process_GPU_sampling_reason=self.process_gpu_reason,
                    sampling_interval_seconds=self.interval,sample_count=len(self.samples),
                    samples=self.samples,sampled_peaks_are_not_instantaneous_guarantees=True,
                    scope='warmups and timed trials plus inter-trial checks/hashing; model load and full V parity excluded')
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
        return result


def _windows(task, origins):
    return (np.stack([task['physical'][int(o)-task['L']:int(o)] for o in origins]).astype(np.float32),
            np.stack([task['observed'][int(o)-task['L']:int(o)] for o in origins]).astype(bool))


def _effective_chunk(request, candidate, registered_rows=None):
    family=candidate['family']
    if family in ('DIRECT_G','RAW_RIDGE'):
        if request['row_microbatch'] is not None:
            raise ValueError('Direct methods require null F row microbatch')
        return None,0
    if request['row_microbatch'] is None:
        raise ValueError('F methods require a row microbatch')
    width=int(candidate['C'] if family in ('RAW','LORA') else candidate['K'])
    rows=request['batch_size']*width
    chunk=min(request['row_microbatch'],rows)
    if registered_rows is not None and chunk not in {min(int(value),rows) for value in registered_rows}:
        raise ValueError('F row microbatch is outside the preregistered common schedule')
    return chunk,width


def _validate_route(route, inputs, device):
    candidate,task,saved=inputs['candidate'],inputs['task'],inputs['saved']
    if route.family!=candidate['family']:
        raise ValueError('Loaded model family differs from accuracy')
    if str(route.device)!=str(device):
        raise ValueError('Loaded model device differs from facet')
    if not np.array_equal(route.mean,saved['mean']) or not np.array_equal(route.std,saved['std']):
        raise ValueError('Loaded scaler differs from scored accuracy')
    if not np.array_equal(saved['channel_ids'].astype(str),np.asarray(task['columns']).astype(str)):
        raise ValueError('Saved channel order differs from task')
    bundle=candidate.get('bundle_binding')
    if bundle and getattr(route,'artifact_sha256',None)!=bundle['sha256']:
        raise ValueError('Loaded checkpoint SHA differs from scored accuracy')
    direct=candidate['family'] in ('DIRECT_G','RAW_RIDGE')
    expected_instances=0 if direct else 1
    if route.pool.instances_created!=expected_instances:
        raise ValueError('Facet must load exactly one foundation, or zero for direct methods')
    foundation=getattr(route.pool,'foundation',None)
    if direct and foundation is not None:
        raise ValueError('Direct-only facet instantiated a foundation')
    if not direct:
        if foundation.initial_hash!=inputs['expected_f_sha']:
            raise ValueError('Loaded foundation state identity differs')
        if foundation.checkpoint_inputs is not False:
            raise ValueError('Inference F input checkpointing must remain disabled')
        parameters=list(foundation.model.parameters())
        if sum(value.numel() for value in parameters)!=inputs['expected_f_params']:
            raise ValueError('Loaded foundation parameter count differs')
        if {value.dtype for value in parameters if value.is_floating_point()}!={torch.float32}:
            raise ValueError('Frozen foundation must retain FP32 weights')
        if any(value.device.type!=torch.device(device).type for value in parameters):
            raise ValueError('Loaded foundation tensor device differs')
        if any(value.requires_grad for value in parameters):
            raise ValueError('Frozen foundation parameter gradients were enabled')
        if candidate['family'] in ('B2','B0') and route.latent_k!=int(candidate['K']):
            raise ValueError('Loaded model latent row count differs')
        if foundation.model.training:
            raise ValueError('Frozen foundation is not eval')
    model=getattr(route,'model',None)
    if model is not None:
        if model.training:
            raise ValueError('Loaded inference model is not eval')
        dtypes={value.dtype for value in model.parameters() if value.is_floating_point()}
        if dtypes and dtypes!={torch.float32}:
            raise ValueError('Adapter/direct model must retain FP32 weights')
    if candidate.get('prediction_dtype') not in (None,str(np.asarray(saved['pred_normalized']).dtype)):
        raise ValueError('Saved prediction precision differs from accuracy metadata')
    if candidate['family']=='RAW_RIDGE':
        if np.asarray(route.ridge_artifact['coefficients']).dtype!=np.float64:
            raise ValueError('Ridge native FP64 core changed')


def _check_calls(route, actual_batch, width, chunk):
    streams=actual_batch*width
    calls=math.ceil(streams/chunk) if chunk is not None else 0
    receipt=route.last_receipt
    if (receipt.get('F_logical_streams')!=streams or receipt.get('F_physical_streams')!=streams
            or receipt.get('F_calls')!=calls):
        raise ValueError('Actual F call/stream counts differ from the registered row schedule')
    return dict(F_logical_streams=streams,F_physical_streams=streams,F_calls=calls)


def _parity(route, inputs, request, width, chunk, check):
    saved,task=inputs['saved'],inputs['task']
    expected=np.asarray(saved['pred_normalized'])
    origins=np.asarray(saved['origins'])
    if (origins.ndim!=1 or not np.issubdtype(origins.dtype,np.integer)
            or len(np.unique(origins))!=len(origins) or len(origins)<request['batch_size']):
        raise ValueError('Unique full V origins and one complete timing batch required')
    lo,hi=task['split_ranges']['V_select']
    if (origins<max(lo,task['L'])).any() or (origins+task['H']>hi).any():
        raise ValueError('Parity origins leave registered development V')
    if expected.shape!=(len(origins),task['H'],len(task['columns'])):
        raise ValueError('Saved prediction shape differs from full task')
    source_metrics=independent_metrics(saved)
    for key in ('MSE','MAE'):
        if not math.isclose(source_metrics[key],float(inputs['candidate'][key]),rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Original accuracy differs from independent FP64 rescore')
    predictions,physical_predictions,counts=[],[],[]
    input_hash,mask_hash=hashlib.sha256(),hashlib.sha256()
    for start in range(0,len(origins),request['batch_size']):
        check()
        physical,observed=_windows(task,origins[start:start+request['batch_size']])
        input_hash.update(np.ascontiguousarray(physical).tobytes())
        mask_hash.update(np.ascontiguousarray(observed).tobytes())
        value=route.predict_physical(physical,observed)
        _check_calls(route,len(physical),width,chunk)
        if value.shape!=(len(physical),task['H'],len(task['columns'])) or not np.isfinite(value).all():
            raise ValueError('Full V route output shape or finite values invalid')
        physical_predictions.append(value)
        predictions.append((value-route.mean)/route.std)
        counts.append(len(physical))
    actual=np.concatenate(predictions)
    if not np.allclose(actual,expected,atol=ATOL,rtol=RTOL):
        raise ValueError('Full V chunk prediction parity failed; max_abs='+str(float(np.max(np.abs(actual-expected)))))
    metrics=independent_metrics(dict(saved,pred_normalized=actual))
    for key in ('MSE','MAE'):
        if not math.isclose(metrics[key],source_metrics[key],abs_tol=ATOL,rel_tol=RTOL):
            raise ValueError('Chunk FP64 metric parity failed: '+key)
    fingerprints={key:_fingerprint(saved[key]) for key in
                  ('origins','channel_ids','mean','std','target_normalized','observed')}
    fingerprints.update(physical_windows=dict(shape=[len(origins),task['L'],len(task['columns'])],
                         dtype='float32',sha256=input_hash.hexdigest()),
                        physical_observed_windows=dict(shape=[len(origins),task['L'],len(task['columns'])],
                         dtype='bool',sha256=mask_hash.hexdigest()))
    return dict(status='PASS',full_V_origins=len(origins),last_batch_size=counts[-1],
                atol=ATOL,rtol=RTOL,max_abs=float(np.max(np.abs(actual-expected))),
                input_fingerprints=fingerprints,
                output_fingerprints=dict(original_normalized=_fingerprint(expected),
                    regenerated_normalized=_fingerprint(actual),
                    regenerated_physical=_fingerprint(np.concatenate(physical_predictions))),
                independent_metrics=metrics,original_metrics=source_metrics,
                metric_deltas={key:metrics[key]-source_metrics[key] for key in ('MSE','MAE')},
                scope='entire saved development V, original order, real physical-input inference')


def evaluate_facet(request, inputs, *, device, sampler_factory=ProcessSampler, lease=None):
    request=validate_request(request)
    candidate=inputs['candidate']
    chunk,width=_effective_chunk(request,candidate,inputs.get('registered_rows'))
    warmups,repeats=STAGES[request['stage']]
    check=lease.check if lease is not None else lambda:None
    check()
    load_started=time.perf_counter()
    route=inputs['loader']()
    sampler=None
    try:
        _sync(device)
        load_seconds=time.perf_counter()-load_started
        _validate_route(route,inputs,device)
        if chunk is not None:
            route.spec['row_microbatch']=route.spec['F_row_microbatch']=chunk
        window_started=time.perf_counter()
        physical,observed=_windows(inputs['task'],inputs['saved']['origins'][:request['batch_size']])
        window_seconds=time.perf_counter()-window_started
        lora_parity={'status':'NOT_APPLICABLE'}
        lora_seconds=0.
        if candidate['family']=='LORA':
            check()
            lora_started=time.perf_counter()
            foundation=route.pool.foundation
            previous_chunk=foundation.row_microbatch
            foundation.row_microbatch=chunk
            try:
                lora_parity=route.check_lora_parity(physical,observed,atol=ATOL,rtol=RTOL)
            finally:
                foundation.row_microbatch=previous_chunk
            lora_parity['F_row_microbatch_used']=chunk
            _sync(device)
            lora_seconds=time.perf_counter()-lora_started
            if lora_parity.get('status')!='PASS':
                raise ValueError('LoRA merge parity failed')
        _sync(device)
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
            _sync(device)
        baseline=_memory(device)
        baseline_rss=psutil.Process().memory_info().rss/1024**2
        parity_started=time.perf_counter()
        parity=_parity(route,inputs,request,width,chunk,check)
        parity_seconds=time.perf_counter()-parity_started
        parity['lora_merge_parity']=lora_parity
        sampler=sampler_factory(device,interval_seconds=.05)
        sampler.start()
        trials=[]
        try:
            for phase,count in (('warmup',warmups),('timed',repeats)):
                sampler.set_phase(phase)
                for index in range(count):
                    check()
                    _sync(device)
                    if phase=='timed' and str(device).startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats(device)
                    started=time.perf_counter()
                    prediction=route.predict_physical(physical,observed)
                    _sync(device)
                    elapsed=time.perf_counter()-started
                    calls=_check_calls(route,len(physical),width,chunk)
                    if prediction.shape!=(request['batch_size'],inputs['task']['H'],len(inputs['task']['columns'])) or not np.isfinite(prediction).all():
                        raise ValueError('Timing output shape or finite values invalid')
                    memory=_memory(device)
                    trials.append(dict(phase=phase,trial=index,index=index,whole_seconds=elapsed,
                        route_internal_whole_seconds=route.last_receipt.get('whole_seconds'),
                        prediction_sha256=_fingerprint(prediction)['sha256'],pid=os.getpid(),
                        CPU_RSS_MiB=psutil.Process().memory_info().rss/1024**2,
                        process_GPU_MiB=sampler.latest_gpu_mib,
                        peak_allocated_MiB=memory['peak_allocated_MiB'] if phase=='timed' else None,
                        peak_reserved_MiB=memory['peak_reserved_MiB'] if phase=='timed' else None,**calls))
        finally:
            samples=sampler.stop()
        timed=[row for row in trials if row['phase']=='timed']
        milliseconds=np.asarray([row['whole_seconds'] for row in timed],np.float64)*1000
        q25,median,q75,p95=np.quantile(milliseconds,[.25,.5,.75,.95])
        allocated=[row['peak_allocated_MiB'] for row in timed if row['peak_allocated_MiB'] is not None]
        reserved=[row['peak_reserved_MiB'] for row in timed if row['peak_reserved_MiB'] is not None]
        resident=route.memory_state()
        for relative,expected in inputs.get('source_bindings',{}).get('runtime_source_sha256',{}).items():
            if _sha(Path(inputs['source_ctx']['repo'])/relative)!=expected:
                raise ValueError('Runtime producer source changed during the facet: '+relative)
        source_sha=getattr(getattr(route.pool,'foundation',None),'initial_hash','NO_FOUNDATION_USED')
        original=chunk is None or chunk==min(128,request['batch_size']*width)
        summary=dict(facet_id=request['facet_id'],stage=request['stage'],status='COMPLETE',
            accuracy_run_id=candidate['run_id'],dataset=candidate['dataset'],family=candidate['family'],
            seed=int(candidate['seed']),B=request['batch_size'],batch_size=request['batch_size'],
            C=int(candidate['C']),K=int(candidate['K']),L=int(candidate['L']),H=int(candidate['H']),
            device=str(device),row_microbatch=chunk,effective_row_microbatch=chunk,
            requested_row_microbatch=request['row_microbatch'],original=original,is_original=original,
            checkpoint_sha256=(candidate.get('bundle_binding') or {}).get('sha256',source_sha),
            prediction_sha256=candidate['prediction_binding']['sha256'],loaded_F_state_sha256=source_sha,
            parity_status='PASS',MSE=parity['original_metrics']['MSE'],MAE=parity['original_metrics']['MAE'],
            recomputed_MSE=parity['independent_metrics']['MSE'],recomputed_MAE=parity['independent_metrics']['MAE'],
            warmups=warmups,timed_trials=repeats,median_batch_ms=float(median),IQR_batch_ms=float(q75-q25),
            p95_batch_ms=float(p95),median_ms_per_origin=float(median/request['batch_size']),
            origins_per_second=float(request['batch_size']*1000/median),
            baseline_allocated_MiB=baseline['allocated_MiB'],baseline_reserved_MiB=baseline['reserved_MiB'],
            peak_allocated_MiB=max(allocated) if allocated else None,
            peak_reserved_MiB=max(reserved) if reserved else None,
            timed_peak_allocated_min_MiB=min(allocated) if allocated else None,
            timed_peak_allocated_max_MiB=max(allocated) if allocated else None,
            timed_peak_reserved_min_MiB=min(reserved) if reserved else None,
            timed_peak_reserved_max_MiB=max(reserved) if reserved else None,
            model_loaded_CPU_RSS_MiB=baseline_rss,
            resident_weights_gpu_MiB=resident.get('resident_weights_gpu_MiB'),
            resident_weights_cpu_MiB=resident.get('resident_weights_cpu_MiB'),
            active_adapter_MiB=resident.get('active_adapter_MiB'),
            F_logical_streams=request['batch_size']*width,F_calls=timed[0]['F_calls'],
            foundation_instances_created=route.pool.instances_created,
            model_load_seconds=load_seconds,foundation_load_seconds=getattr(route.pool,'load_seconds',None),
            timing_window_extraction_seconds=window_seconds,full_V_parity_seconds=parity_seconds,
            lora_merge_parity_seconds=lora_seconds,
            empty_cache_once_before_loaded_baseline=str(device).startswith('cuda'),
            process_pid=os.getpid(),process_create_time=psutil.Process().create_time(),pid=os.getpid(),
            process_GPU_sampled_peak_MiB=samples['process_GPU_sampled_peak_MiB'],
            process_GPU_sampling_status=samples['process_GPU_sampling_status'],
            process_GPU_sampling_reason=samples['process_GPU_sampling_reason'],
            CPU_RSS_sampled_peak_MiB=samples['CPU_RSS_sampled_peak_MiB'],
            memory_layer='pytorch_allocated_tensor' if str(device).startswith('cuda') else 'cpu_RSS',
            core_dtype='float64' if candidate['family']=='RAW_RIDGE' else 'float32',
            backbone_dtype='float32' if width else 'NO_FOUNDATION_USED',
            physical_output_dtype='float64',metric_accumulation_dtype='float64',
            p95_scope=str(repeats)+'-sample descriptive quantile; no stable tail guarantee',
            process_scope='fresh OS process, one facet, one serialized model load',
            baseline_scope='deployed model ready after one-time LoRA merge parity if applicable; all CUDA facets synchronize/empty_cache once/synchronize before loaded baseline; before full V parity and warmups',
            peak_reset_scope='after warmup and before every timed trial; reported peak is maximum across timed trials',
            timing_scope='physical window and bool mask through fill/scaler/model/inverse/CPU output; window extraction and model load excluded',
            cached_forecast_reads_in_timed_path=0,stage_probe_trials=0,
            protocol_sha256=request['protocol_sha256'])
        return dict(status='COMPLETE',summary=summary,trials=trials,parity=parity,
                    accuracy=dict(candidate),sampling=samples,
                    source_bindings=inputs.get('source_bindings',{}),
                    environment=dict(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                        driver=sampler.driver,device=str(device),sampling_interval_seconds=.05,
                        allocator_backend=torch.cuda.get_allocator_backend() if str(device).startswith('cuda') else 'NOT_APPLICABLE',
                        CUDA_synchronization='before and after each timed whole path',
                        autocast=False,quantization=False,compile=False))
    finally:
        route.close()


def _safe_failure(error, request):
    content=''.join(traceback.format_exception(type(error),error,error.__traceback__))
    for path,label in ((str(REPO),'<repo>'),(str(request.get('asset_root','')),'<asset>'),
                       (str(request.get('resource_cache','')),'<resource-cache>')):
        if path:
            content=content.replace(path,label)
    # Handshake environment is never included. Limit external exception payloads.
    return dict(exception_type=type(error).__name__,message=content.splitlines()[-1] if content else type(error).__name__,
                traceback=content[-16000:])


def execute_request(request, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Preserve the existing facet attempt; use another output path')
    started=time.time()
    identity=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    receipt=dict(schema_version=1,status='RUNNING',facet_id=request.get('facet_id'),
                 protocol_sha256=request.get('protocol_sha256'),request_identity_sha256=identity,
                 process=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),started_unix=started),
                 worker_source_sha256=_sha(__file__),failure=None)
    _json(output,receipt,exclusive=True)
    try:
        request=validate_request(request)
        protocol=REPO/'reports/memory_budget_20261018_v1/PROTOCOL.md'
        if _sha(protocol)!=request['protocol_sha256']:
            raise ValueError('Memory protocol document changed after preregistration')
        inputs=prepare_inputs(request)
        device='cpu' if inputs['candidate']['family']=='RAW_RIDGE' else 'cuda'
        lease=None
        if device=='cuda':
            if not request.get('resource_cache'):
                raise ValueError('CUDA facet requires the new campaign parent resource lease')
            from resources import validate_gpu_child
            lease=validate_gpu_child(request['resource_cache'])
        with lease if lease is not None else nullcontext():
            if lease is not None:
                inputs['source_ctx']['lease']=lease
            receipt.update(evaluate_facet(request,inputs,device=device,lease=lease))
    except BaseException as error:
        receipt.update(status='INVALID',failure=_safe_failure(error,request))
        receipt.setdefault('summary',{})
        receipt['summary'].update(status='INVALID',facet_id=request.get('facet_id'),
                                  stage=request.get('stage'),accuracy_run_id=request.get('accuracy_run_id'),
                                  parity_status='FAIL_OR_NOT_COMPLETED')
    receipt['process'].update(finished_unix=time.time(),wall_seconds=time.time()-started)
    _json(output,receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--candidate-file',type=Path)
    args=parser.parse_args(argv)
    request=json.loads(args.request.read_text(encoding='utf-8-sig'))
    if args.candidate_file is not None:
        if request.get('candidate_file') not in (None,str(args.candidate_file)):
            parser.error('--candidate-file differs from request candidate_file')
        request['candidate_file']=str(args.candidate_file)
    result=execute_request(request,args.output)
    print(result['status']+' '+str(result.get('facet_id')),flush=True)
    if result['status']!='COMPLETE':
        print(result['failure']['message'],flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
