import importlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def worker():
    return importlib.import_module('research.memory_budget_20261018_v1.worker')


def request(**changes):
    value = dict(asset_root='unused-assets', accuracy_run_id='fixture', batch_size=8,
                 row_microbatch=3, stage='screen', protocol_sha256='a'*64,
                 facet_id='fixture_B8_R3_screen', resource_cache='unused-resource')
    value.update(changes)
    return value


@pytest.mark.parametrize('changes', [dict(batch_size=True), dict(batch_size=2),
    dict(row_microbatch=0), dict(row_microbatch=True), dict(stage='fast'),
    dict(facet_id='../escape'), dict(protocol_sha256='not-a-sha')])
def test_request_rejects_unregistered_or_ambiguous_facets(changes):
    with pytest.raises(ValueError):
        worker().validate_request(request(**changes))


def test_only_final_off_exact_accuracy_identity_is_selected():
    row = dict(run_id='fixture', family='B2', dataset='jena', final=True, revin=False,
               evaluation='V_select', independent_FP64='PASS', status='COMPLETE',
               prediction_binding={'sha256':'p'}, bundle_binding={'sha256':'b'})
    assert worker().select_accuracy([row], 'fixture') == row
    for key, value in [('final',False), ('revin',True), ('independent_FP64','FAIL'),
                       ('evaluation','T_final')]:
        with pytest.raises(ValueError):
            worker().select_accuracy([dict(row, **{key:value})], 'fixture')
    with pytest.raises(ValueError):
        worker().select_accuracy([row,row], 'fixture')


class ToyRoute:
    def __init__(self, family='B2', broken_calls=False, dtype=torch.float32):
        self.family, self.latent_k = family, 2 if family=='B2' else 3
        self.device = 'cpu'
        self.spec = {'row_microbatch':128, 'F_row_microbatch':128}
        self.mean, self.std = np.zeros(3,np.float32), np.ones(3,np.float32)
        self.model = torch.nn.Linear(3,3,dtype=dtype).eval()
        self.foundation_model = torch.nn.Linear(1,1,bias=False,dtype=dtype).eval().requires_grad_(False)
        self.artifact_sha256='b'*64
        self.pool = SimpleNamespace(instances_created=1,
            foundation=SimpleNamespace(model=self.foundation_model,initial_hash='f'*64,
                                       checkpoint_inputs=False,row_microbatch=128))
        self.last_receipt = {}
        self.calls, self.events, self.closed = [], [], False
        self.broken_calls = broken_calls

    def activate(self):
        self.events.append('activate')

    def check_lora_parity(self, physical, observed, atol, rtol):
        self.events.append('lora_parity_merge')
        return dict(status='PASS',max_abs=0.,atol=atol,rtol=rtol,
                    actual_direct_F_chunk=self.pool.foundation.row_microbatch)

    def predict_physical(self, physical, observed):
        self.events.append('predict')
        self.calls.append(len(physical))
        width = 3 if self.family=='LORA' else self.latent_k
        streams = len(physical)*width
        count = math.ceil(streams/self.spec['row_microbatch'])
        self.last_receipt = dict(F_logical_streams=streams,F_physical_streams=streams,
                                 F_calls=count+int(self.broken_calls),whole_seconds=.01)
        filled = np.where(observed,physical,0.)
        return np.repeat(filled[:,-1:],2,axis=1).astype(np.float64)

    def memory_state(self):
        return dict(resident_weights_gpu_MiB=0.,resident_weights_cpu_MiB=.01,
                    active_adapter_MiB=0.,foundation_instances_created=1)

    def close(self):
        self.closed = True


class ToySampler:
    def __init__(self, device, interval_seconds=.05):
        self.device, self.phase = device, 'warmup'
        self.driver = 'fixture-driver'
        self.process_gpu_status = 'NOT_APPLICABLE_CPU'
        self.process_gpu_reason = 'CPU facet'
        self.latest_gpu_mib = None

    def start(self): pass
    def set_phase(self,phase): self.phase=phase
    def stop(self):
        return dict(CPU_RSS_sampled_peak_MiB=20.,process_GPU_sampled_peak_MiB=None,
                    process_GPU_sampling_status=self.process_gpu_status,
                    process_GPU_sampling_reason=self.process_gpu_reason,
                    sampling_interval_seconds=.05,samples=[])


def toy_inputs(family='B2', **route_options):
    w = worker()
    physical = np.arange(20*3,dtype=np.float32).reshape(20,3)/10
    observed = np.ones_like(physical,dtype=bool)
    origins = np.arange(5,16,dtype=np.int64)
    task = dict(physical=physical,observed=observed,columns=['a','b','c'],L=5,H=2,
                data_sha256='d'*64,split_ranges={'V_select':(5,17)})
    prediction = np.stack([np.repeat(physical[o-1:o],2,axis=0) for o in origins])
    target = prediction + np.float32(.125)
    saved = dict(origins=origins,channel_ids=np.asarray(task['columns']),
                 pred_normalized=prediction.astype(np.float32),target_normalized=target,
                 observed=np.ones_like(target,dtype=bool),mean=np.zeros(3,np.float32),
                 std=np.ones(3,np.float32))
    metrics = w.independent_metrics(saved)
    candidate = dict(run_id='fixture',family=family,dataset='jena',seed=92601,C=3,
                     K=2 if family=='B2' else 3,L=5,H=2,MSE=metrics['MSE'],MAE=metrics['MAE'],
                     bundle_binding={'sha256':'b'*64},prediction_binding={'sha256':'p'*64},
                     model_sha='f'*64,core_dtype='float32',revin=False,final=True)
    route = ToyRoute(family=family,**route_options)
    loads=[]
    def loader():
        loads.append(1)
        return route
    return dict(candidate=candidate,task=task,saved=saved,loader=loader,
                expected_f_sha='f'*64,expected_f_params=1),route,loads


def test_single_load_full_v_ragged_tail_and_screen_repetitions(monkeypatch):
    w=worker()
    inputs,route,loads=toy_inputs()
    result=w.evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert len(loads)==1 and route.closed
    assert result['status']=='COMPLETE'
    assert result['parity']['status']=='PASS'
    assert result['parity']['full_V_origins']==11
    assert result['parity']['last_batch_size']==3
    assert route.calls[:2]==[8,3]
    assert len(route.calls)==2+2+3
    assert route.spec['row_microbatch']==route.spec['F_row_microbatch']==3
    assert len([r for r in result['trials'] if r['phase']=='warmup'])==2
    assert len([r for r in result['trials'] if r['phase']=='timed'])==3
    assert {r['F_calls'] for r in result['trials']}=={6}
    assert result['summary']['process_GPU_sampled_peak_MiB'] is None
    assert result['summary']['baseline_reserved_MiB'] is None
    assert result['summary']['MSE']==inputs['candidate']['MSE']
    assert result['summary']['is_original'] is False


def test_confirm_has_ten_warmups_twenty_trials_and_original_clamp():
    inputs,route,loads=toy_inputs()
    result=worker().evaluate_facet(request(stage='confirm',row_microbatch=128),inputs,
                                  device='cpu',sampler_factory=ToySampler)
    assert len(loads)==1
    assert route.spec['row_microbatch']==16
    assert result['summary']['is_original'] is True
    assert result['summary']['effective_row_microbatch']==16
    assert len(result['trials'])==30
    assert result['summary']['timed_trials']==20


def test_lora_merge_parity_precedes_every_prediction_and_occurs_once():
    inputs,route,loads=toy_inputs('LORA')
    result=worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert result['status']=='COMPLETE'
    assert route.events[0]=='lora_parity_merge'
    assert route.events.count('lora_parity_merge')==1
    assert result['parity']['lora_merge_parity']['actual_direct_F_chunk']==3
    assert route.pool.foundation.row_microbatch==128
    assert len(loads)==1


def test_wrong_f_call_count_invalidates_facet_and_closes_route():
    inputs,route,loads=toy_inputs(broken_calls=True)
    with pytest.raises(ValueError,match='F call'):
        worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert len(loads)==1 and route.closed


def test_changed_precision_or_checkpoint_identity_is_rejected():
    inputs,route,_=toy_inputs(dtype=torch.float16)
    with pytest.raises(ValueError,match='FP32'):
        worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert route.closed
    inputs,route,_=toy_inputs()
    route.artifact_sha256='wrong-checkpoint'
    with pytest.raises(ValueError,match='checkpoint'):
        worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert route.closed
    inputs,route,_=toy_inputs()
    inputs['expected_f_sha']='x'*64
    with pytest.raises(ValueError,match='foundation'):
        worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert route.closed


def test_prediction_parity_failure_cannot_produce_confirmed_metrics():
    inputs,route,_=toy_inputs()
    inputs['saved']['pred_normalized']=inputs['saved']['pred_normalized']+1
    metrics=worker().independent_metrics(inputs['saved'])
    inputs['candidate'].update(MSE=metrics['MSE'],MAE=metrics['MAE'])
    with pytest.raises(ValueError,match='parity'):
        worker().evaluate_facet(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert route.closed


def test_masked_metric_is_channel_macro_not_flat_micro():
    observed=np.array([[[True,True],[True,False]]])
    values=dict(pred_normalized=np.array([[[1.,4.],[3.,999.]]]),
                target_normalized=np.zeros((1,2,2)),observed=observed)
    result=worker().independent_metrics(values)
    assert result['MSE']==10.5
    assert result['MAE']==3.
    assert result['observed']==3


def test_real_row_forecaster_order_with_nonuniform_tail_and_missing_input():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('memory_budget_test_runtime',root/'research/informs_broad_validation_20261015_v1/gpu_runtime.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values=torch.arange(8*5*3,dtype=torch.float32).reshape(8,5,3)
    full=module.RowForecaster(lambda rows:rows[:,-2:],2,128,False)
    split=module.RowForecaster(lambda rows:rows[:,-2:],2,5,False)
    assert torch.equal(full(values),split(values))
    assert split.physical_calls==5 and split.streams==24
    assert torch.isfinite(split(torch.zeros_like(values))).all()


def test_missing_channel_causal_fill_is_finite_and_preserves_observed_values():
    from research.informs_broad_validation_20261015_v1.data_contract import causal_forward_fill
    physical=np.array([[np.nan,2.,np.nan],[3.,np.nan,np.nan],[4.,5.,np.nan]],np.float32)
    mask=np.isfinite(physical)
    filled=causal_forward_fill(physical,mask,np.array([10.,20.,30.],np.float32))
    assert np.array_equal(filled,np.array([[10.,2.,30.],[3.,2.,30.],[4.,5.,30.]],np.float32))
    assert np.isfinite(filled).all()
    assert np.array_equal(filled[mask],physical[mask])


def test_gpu_allocated_and_reserved_are_separate_tensor_allocator_metrics(monkeypatch):
    w=worker()
    for name,value in [('memory_allocated',7),('memory_reserved',9),
                       ('max_memory_allocated',12),('max_memory_reserved',16)]:
        monkeypatch.setattr(w.torch.cuda,name,lambda device,value=value:value*1024**2)
    result=w._memory('cuda')
    assert result==dict(allocated_MiB=7.,reserved_MiB=9.,peak_allocated_MiB=12.,peak_reserved_MiB=16.)
    assert 'process_GPU_MiB' not in result


@pytest.mark.parametrize('unavailable',[False,True])
def test_nvml_sampler_uses_current_pid_and_preserves_wddm_unavailability(monkeypatch,unavailable):
    import sys
    w=worker()
    current=2**64-1 if unavailable else 189*1024**2
    closed=[]
    fake=SimpleNamespace(nvmlInit=lambda:None,nvmlSystemGetDriverVersion=lambda:b'595.79',
        nvmlDeviceGetHandleByIndex=lambda index:index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle:[SimpleNamespace(pid=os.getpid(),usedGpuMemory=current),
                                                            SimpleNamespace(pid=os.getpid()+100,usedGpuMemory=999*1024**2)],
        nvmlShutdown=lambda:closed.append(True),NVMLError=RuntimeError)
    monkeypatch.setitem(sys.modules,'pynvml',fake)
    sampler=w.ProcessSampler('cuda')
    sampler.start()
    result=sampler.stop()
    assert sampler.driver=='595.79' and closed==[True]
    assert result['sampling_interval_seconds']==.05
    assert result['sampled_peaks_are_not_instantaneous_guarantees'] is True
    assert result['process_GPU_sampled_peak_MiB']==(None if unavailable else 189.)
    assert result['process_GPU_sampling_status']==('UNAVAILABLE' if unavailable else 'AVAILABLE')


class ToyDirectRoute(ToyRoute):
    def __init__(self,family):
        super().__init__(family=family)
        self.latent_k=0
        self.pool.instances_created=0
        self.pool.foundation=None
        if family=='RAW_RIDGE':
            self.model=None
            self.ridge_artifact={'coefficients':np.zeros((3,33,2),np.float64)}

    def predict_physical(self,physical,observed):
        self.calls.append(len(physical))
        self.last_receipt=dict(F_logical_streams=0,F_physical_streams=0,F_calls=0)
        return np.repeat(physical[:,-1:],2,axis=1).astype(np.float64)


@pytest.mark.parametrize('family',['RAW_RIDGE','DIRECT_G'])
def test_direct_facets_never_instantiate_foundation_and_have_no_chunk(family):
    inputs,_,_=toy_inputs()
    route=ToyDirectRoute(family)
    inputs['loader']=lambda:route
    inputs['candidate'].update(family=family,K=0,dataset='jena' if family=='RAW_RIDGE' else 'pems32')
    result=worker().evaluate_facet(request(row_microbatch=None),inputs,device='cpu',sampler_factory=ToySampler)
    assert route.closed
    assert result['summary']['F_calls']==0 and result['summary']['F_logical_streams']==0
    assert result['summary']['foundation_instances_created']==0
    assert result['summary']['row_microbatch'] is None and result['summary']['is_original'] is True
    assert result['summary']['core_dtype']==('float64' if family=='RAW_RIDGE' else 'float32')


def test_raw_checkpoint_identity_uses_actual_loaded_f_not_missing_bundle():
    inputs,route,_=toy_inputs('RAW')
    inputs['candidate']['bundle_binding']=None
    result=worker().evaluate_facet(request(row_microbatch=128,batch_size=1),inputs,
                                  device='cpu',sampler_factory=ToySampler)
    assert result['summary']['checkpoint_sha256']=='f'*64
    assert result['parity']['last_batch_size']==1


def test_cuda_direct_facet_requires_parent_lease_before_model_load(tmp_path,monkeypatch):
    w=worker()
    monkeypatch.setattr(w,'REPO',tmp_path)
    protocol=tmp_path/'reports/memory_budget_20261018_v1/PROTOCOL.md'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('sealed-protocol',encoding='utf-8')
    inputs,route,loads=toy_inputs()
    inputs['candidate']['family']='DIRECT_G'
    monkeypatch.setattr(w,'prepare_inputs',lambda request:inputs)
    value=request(protocol_sha256=w._sha(protocol))
    value.pop('resource_cache')
    result=w.execute_request(value,tmp_path/'receipt.json')
    assert result['status']=='INVALID' and not loads
    assert 'resource lease' in result['failure']['message']
    assert result['process']['pid']==os.getpid()
    assert result['process']['create_time']>0
    assert json.loads((tmp_path/'receipt.json').read_text())['status']=='INVALID'


def test_receipt_is_write_once_for_existing_attempt(tmp_path,monkeypatch):
    w=worker()
    path=tmp_path/'receipt.json'
    path.write_text('{"status":"INVALID","proof":"preserve"}',encoding='utf-8')
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        w.execute_request(request(),path)
    assert path.read_bytes()==before


def test_exact_parent_row_prediction_checkpoint_bindings_reject_drift_before_loading():
    w=worker()
    inputs,_,_=toy_inputs()
    candidate=inputs['candidate']
    digest=w.validate_parent_binding(candidate,{})
    value=request(expected_candidate_sha256=digest,parent_checkpoint_sha256='b'*64,
                  parent_prediction_sha256='p'*64)
    assert w.validate_parent_binding(candidate,value)==digest
    for key in ('expected_candidate_sha256','parent_checkpoint_sha256','parent_prediction_sha256'):
        with pytest.raises(ValueError):
            w.validate_parent_binding(candidate,dict(value,**{key:'wrong'}))
    raw=dict(candidate,family='RAW',bundle_binding=None)
    w.validate_parent_binding(raw,dict(parent_checkpoint_sha256='NO_SERIALIZED_PARENT'))
    with pytest.raises(ValueError,match='Raw'):
        w.validate_parent_binding(raw,dict(parent_checkpoint_sha256='old-path-hash'))


def test_cuda_cache_cleared_once_before_parity_and_peak_reset_each_timed_trial(monkeypatch):
    w=worker()
    inputs,route,_=toy_inputs()
    route.device='cuda'
    events=[]
    monkeypatch.setattr(w,'_validate_route',lambda *args:None)
    monkeypatch.setattr(w,'_sync',lambda device:events.append('sync'))
    monkeypatch.setattr(w,'_memory',lambda device:dict(allocated_MiB=7.,reserved_MiB=9.,
                                                      peak_allocated_MiB=12.,peak_reserved_MiB=16.))
    monkeypatch.setattr(w.torch.cuda,'empty_cache',lambda:events.append(('empty',len(route.calls))))
    monkeypatch.setattr(w.torch.cuda,'reset_peak_memory_stats',lambda device:events.append(('reset',len(route.calls))))
    monkeypatch.setattr(w.torch.cuda,'get_allocator_backend',lambda:'native')
    result=w.evaluate_facet(request(),inputs,device='cuda',sampler_factory=ToySampler)
    assert [event for event in events if isinstance(event,tuple) and event[0]=='empty']==[('empty',0)]
    resets=[event[1] for event in events if isinstance(event,tuple) and event[0]=='reset']
    assert resets==[4,5,6]  # two full-V batches, two warmups, then three timed trials
    assert result['summary']['empty_cache_once_before_loaded_baseline'] is True
    assert result['summary']['peak_allocated_MiB']==12.
    assert result['summary']['peak_reserved_MiB']==16.
