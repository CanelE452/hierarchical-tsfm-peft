import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    try:
        return importlib.import_module('research.b2_dataset_extension_20261018_v1.costs')
    except ModuleNotFoundError:
        pytest.fail('The independent new-dataset whole-path cost worker is missing')


def request(family='RAW'):
    return dict(dataset='household', family=family, run_id=family+'_household_s92601',
                seed=92601, K=2, factory='train:cost_inputs', device='cuda')


def fixture_inputs(family='RAW', count=11):
    physical=np.arange(640*7,dtype=np.float32).reshape(640,7)/100
    origins=np.arange(512,512+count,dtype=np.int64)
    mean=np.arange(7,dtype=np.float32)
    std=np.full(7,2.,np.float32)
    target=np.stack([physical[o:o+48] for o in origins])
    normalized=(target-mean)/std
    foundation=SimpleNamespace(row_microbatch=128,initial_hash='fixed',checkpoint_inputs=False,
        model=torch.nn.Linear(1,1).eval().requires_grad_(False))

    class Route:
        def __init__(self):
            self.mean,self.std=mean,std
            self.family=family
            self.spec={}
            self.pool=SimpleNamespace(foundation=foundation)
            self.last_receipt={}
            self.batches=[]

        def predict_physical(self,past,observed):
            assert past.shape==observed.shape and past.shape[1:]==(512,7)
            self.batches.append(len(past))
            width=7 if family=='RAW' else 2 if family=='B2' else 0
            cap=foundation.row_microbatch
            self.last_receipt=dict(F_logical_streams=len(past)*width,
                F_physical_streams=len(past)*width,F_calls=math.ceil(len(past)*width/cap) if width else 0)
            return np.repeat(past[:,-1:, :].astype(np.float64),48,axis=1)

    route=Route()
    expected=np.stack([np.repeat(physical[o-1:o],48,axis=0) for o in origins]).astype(np.float64)
    saved=dict(pred_normalized=(expected-mean)/std,target_normalized=normalized,
        observed=np.ones_like(normalized,bool),origins=origins,mean=mean,std=std,
        channel_ids=np.asarray([str(i) for i in range(7)]))
    task=dict(physical=physical,observed=np.ones_like(physical,bool),columns=saved['channel_ids'],
        L=512,H=48,split_ranges={'V_select':(512,640)})
    return dict(route_loader=lambda:route,task=task,saved=saved,expected_f_sha='fixed'),route


def test_matched_cap_uses_eight_K_for_raw_and_B2_with_actual_tail():
    c=module()
    assert c.expected_work(request(),7,8)==dict(row_cap=16,width=7,streams=56,calls=4)
    assert c.expected_work(request('B2'),7,8)==dict(row_cap=16,width=2,streams=16,calls=1)
    assert c.expected_work(request(),7,3)==dict(row_cap=16,width=7,streams=21,calls=2)
    assert c.expected_work(request('DIRECT_G'),7,8)==dict(row_cap=None,width=0,streams=0,calls=0)


def test_full_V_reexecutes_physical_path_and_keeps_unequal_tail():
    c=module();inputs,route=fixture_inputs()
    actual,proof=c.full_validation(route,inputs,request())
    assert route.batches==[8,3]
    assert proof['status']=='PASS' and proof['batch_sizes']==[8,3]
    assert proof['F_calls']==[4,2] and proof['full_V_origins']==11
    np.testing.assert_allclose(actual,inputs['saved']['pred_normalized'],atol=1e-5,rtol=1e-4)
    assert proof['metrics']['channel_counts']==[11*48]*7
    assert len(proof['fingerprints']['origins']['sha256'])==64


def test_full_V_rejects_changed_prediction_or_scaler_before_timing():
    c=module();inputs,route=fixture_inputs()
    inputs['saved']['pred_normalized'][0,0,0]+=1
    with pytest.raises(ValueError,match='parity'):
        c.full_validation(route,inputs,request())
    inputs,route=fixture_inputs()
    inputs['saved']['std']=np.ones(7,np.float32)
    with pytest.raises(ValueError,match='scaler'):
        c.full_validation(route,inputs,request())


def test_actual_call_counter_must_match_row_cap():
    c=module();inputs,route=fixture_inputs()
    original=route.predict_physical
    def bad(past,observed):
        value=original(past,observed)
        route.last_receipt['F_calls']=99
        return value
    route.predict_physical=bad
    with pytest.raises(ValueError,match='call'):
        c.full_validation(route,inputs,request())


def test_measurement_runs_ten_warmups_twenty_timings_and_resets_peaks(monkeypatch):
    c=module();inputs,route=fixture_inputs('B2')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    clock=SimpleNamespace(value=0.,resets=0,releases=0)
    def tick():
        clock.value+=.01
        return clock.value
    monkeypatch.setattr(c.time,'perf_counter',tick)
    monkeypatch.setattr(c,'synchronize',lambda device:None)
    monkeypatch.setattr(c,'memory',lambda device:dict(allocated_MiB=5.,reserved_MiB=9.,peak_allocated_MiB=8.,peak_reserved_MiB=12.,RSS_MiB=20.))
    monkeypatch.setattr(c,'reset_peaks',lambda device:setattr(clock,'resets',clock.resets+1))
    monkeypatch.setattr(c,'release_validation_cache',lambda device:setattr(clock,'releases',clock.releases+1))
    result,arrays=c.measure_inputs(request('B2'),inputs,device='cpu')
    assert result['status']=='COMPLETE' and result['validation']['batch_sizes']==[8,3]
    assert len(result['trials'])==30 and clock.resets==21
    assert clock.releases==1
    assert sum(t['phase']=='timed' for t in result['trials'])==20
    assert result['summary']['repetitions']==20
    assert result['summary']['peak_allocated_MiB']==8.
    assert result['cublas_workspace_config']==':16:8'
    assert all(t['batch_size']==8 and t['F_calls']==1 for t in result['trials'])
    np.testing.assert_array_equal(arrays['origins'],inputs['saved']['origins'])


def test_raw_creates_new_128_reference_then_validates_matched_cap(monkeypatch):
    c=module();inputs,route=fixture_inputs()
    inputs['saved'].pop('pred_normalized')
    monkeypatch.setattr(c,'synchronize',lambda device:None)
    monkeypatch.setattr(c,'memory',lambda device:dict(allocated_MiB=0.,reserved_MiB=0.,peak_allocated_MiB=0.,peak_reserved_MiB=0.,RSS_MiB=1.))
    monkeypatch.setattr(c,'reset_peaks',lambda device:None)
    result,arrays=c.measure_inputs(request(),inputs,device='cpu')
    assert result['original_reference']['generated_new_RAW_reference'] is True
    assert result['original_reference']['row_cap']==128
    assert result['original_reference']['metrics']['MSE']==pytest.approx(result['validation']['metrics']['MSE'])
    assert result['validation']['row_cap']==16
    assert arrays['pred_normalized'].shape==(11,48,7)
    assert np.isfinite(arrays['pred_normalized']).all()


def test_direct_reference_is_new_and_contains_no_foundation(monkeypatch):
    c=module();inputs,route=fixture_inputs('DIRECT_G')
    route.pool.foundation=None
    inputs['saved']['pred_normalized']=None
    monkeypatch.setattr(c,'synchronize',lambda device:None)
    monkeypatch.setattr(c,'memory',lambda device:dict(allocated_MiB=0.,reserved_MiB=0.,peak_allocated_MiB=0.,peak_reserved_MiB=0.,RSS_MiB=1.))
    monkeypatch.setattr(c,'reset_peaks',lambda device:None)
    result,arrays=c.measure_inputs(request('DIRECT_G'),inputs,device='cpu')
    assert result['original_reference']['generated_new_accuracy_reference'] is True
    assert result['original_reference']['row_cap'] is None
    assert result['validation']['F_calls']==[0,0]
    assert result['F_state_sha256']=='NO_FOUNDATION_USED'
    assert arrays['pred_normalized'].shape==(11,48,7)


def test_failure_receipt_is_new_scoped_and_does_not_leak_private_error(tmp_path,monkeypatch):
    c=module()
    root=tmp_path/c.NAME
    report,cache=root/'reports',root/'cache'
    def failing(*args):raise ValueError('secret C:/private/data.npz')
    monkeypatch.setattr(c.importlib,'import_module',lambda name:SimpleNamespace(cost_inputs=failing))
    with pytest.raises(ValueError,match='secret'):
        c.execute_request(request(),report,cache)
    import json
    receipt=report/'costs'/(request()['run_id']+'_B8.json')
    value=json.loads(receipt.read_text())
    assert value['status']=='FAILED' and value['error_type']=='ValueError'
    assert 'private/data' not in receipt.read_text()
    before=receipt.read_bytes()
    with pytest.raises(FileExistsError):c.execute_request(request(),report,cache)
    assert receipt.read_bytes()==before


def test_old_namespace_output_is_rejected(tmp_path):
    c=module()
    with pytest.raises(ValueError,match='namespace'):
        c._output_roots(tmp_path/'old_reports',tmp_path/'old_cache')


@pytest.mark.parametrize('workspace',[None,':16:8'])
def test_fresh_worker_inherits_workspace_without_overriding_unset_or_existing_value(tmp_path,monkeypatch,workspace):
    import json
    import os
    c=module()
    if workspace is None:monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    else:monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',workspace)
    report,cache=tmp_path/c.NAME/'report',tmp_path/c.NAME/'cache'
    request_path=tmp_path/'private_request.json'
    request_path.write_text(json.dumps(request()))
    calls=[]
    def spawn(command,**kwargs):
        calls.append((command,kwargs['env']))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess,'run',spawn)
    monkeypatch.setattr(c,'execute_request',lambda *args:pytest.fail('Coordinator must launch a fresh OS worker'))
    assert c.main(['--request',str(request_path),'--report-dir',str(report),'--cache-dir',str(cache)])==0
    assert len(calls)==1 and '--worker' in calls[0][0]
    child_env=calls[0][1]
    actual_workspace=child_env.get('CUBLAS_WORKSPACE_CONFIG')
    actual_present='CUBLAS_WORKSPACE_CONFIG' in child_env
    assert actual_workspace==workspace
    assert actual_present==(workspace is not None)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==workspace


@pytest.mark.parametrize('change',['batch','K','id','factory','family'])
def test_invalid_request_rejected(change):
    c=module();value=request()
    value[{'batch':'batch_size','K':'K','id':'run_id','factory':'factory','family':'family'}[change]]= {
        'batch':1,'K':0,'id':'../old','factory':'old.worker:main','family':'LORA'}[change]
    with pytest.raises(ValueError):c.validate_request(value)
