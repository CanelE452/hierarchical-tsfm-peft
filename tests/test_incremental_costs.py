import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.incremental_latent_adaptation_20261019_v1 import common,costs,deployment,models
from research.incremental_latent_adaptation_20261019_v1 import blocks
from research.informs_broad_validation_20261015_v1.gpu_runtime import RowForecaster
from research.informs_broad_validation_20261015_v1.networks import DirectG,ForecastAdapter


class FakePool:
    def __init__(self,horizon):
        self.horizon=horizon
        self.foundation=None
        self.instances_created=0
        self.get_calls=0

    def get(self,horizon):
        assert horizon==self.horizon
        self.get_calls+=1
        if self.foundation is None:
            self.foundation=RowForecaster(lambda rows:rows[:,-1:].expand(-1,horizon)+.125,horizon,128,False)
            self.foundation.model=torch.nn.Linear(1,1,bias=False)
            self.foundation.model.chronos_config=SimpleNamespace(context_length=2048,prediction_length=64)
            self.instances_created+=1
        return self.foundation

    def counters(self):
        return {k:int(getattr(self.foundation,k,0)) for k in ('streams','physical_streams','physical_calls')}

    def close(self):
        self.foundation=None


@pytest.fixture
def parents():
    task=dict(L=512,H=4,K=2,columns=['a','b','c'],split_ranges={'V_select':(512,600)})
    center=np.zeros(3,np.float32)
    basis=np.eye(3,dtype=np.float32)[:,:2]
    torch.manual_seed(22)
    return dict(task=task,s_model=ForecastAdapter('B2',512,4,3,2,center,basis),
                g_model=DirectG(512,4),mean=np.array([2,-3,1],np.float32),std=np.array([2,4,.5],np.float32))


def physical_inputs(parents,batch=2):
    normalized=np.arange(512,dtype=np.float32)[None,:,None]/512
    physical=np.broadcast_to(normalized,(batch,512,3))*parents['std']+parents['mean']
    return physical.copy(),np.ones_like(physical,dtype=bool)


def artifact(tmp_path,parent,method):
    model=models.make(parent,method,92601)
    path=tmp_path/(method+'.pt')
    torch.save({'model':model.state_dict()},path)
    return dict(state_path=str(path),state_sha256=common.sha(path),model_state_sha256=deployment.state_hash(model))


def test_g_uses_causal_past_fill_and_no_foundation(parents):
    pool=FakePool(4)
    route=deployment.Route(dict(method='G',seed=92601),parents,device='cpu',foundation_pool=pool)
    physical,observed=physical_inputs(parents)
    observed[:,-1]=False
    physical[:,-1]=np.nan
    expected=np.broadcast_to(physical[:,-2,None],(2,4,3))
    result=route.predict_physical(physical,observed)
    np.testing.assert_allclose(result,expected,atol=1e-6)
    assert pool.get_calls==0 and route.last_receipt['F_calls']==0
    assert route.last_receipt['cached_forecast_or_features_reads']==0


@pytest.mark.parametrize('method',['RESIDUAL_Q','RESIDUAL_RAW_MLP'])
def test_no_f_methods_do_not_load_foundation(parents,tmp_path,method):
    pool=FakePool(4)
    spec=dict(method=method,seed=92601,**artifact(tmp_path,parents['s_model'],method))
    route=deployment.Route(spec,parents,device='cpu',foundation_pool=pool)
    physical,observed=physical_inputs(parents)
    route.warm_load()
    result=route.predict_physical(physical,observed)
    np.testing.assert_allclose(result,np.broadcast_to(physical[:,-1,None],(2,4,3)),atol=1e-6)
    assert pool.get_calls==0 and route.last_receipt['F_physical_rows']==0


def test_readout_timed_deployment_still_executes_original_f(parents,tmp_path):
    pool=FakePool(4)
    spec=dict(method='READOUT_ONLY',seed=92601,**artifact(tmp_path,parents['s_model'],'READOUT_ONLY'))
    route=deployment.Route(spec,parents,device='cpu',foundation_pool=pool)
    physical,observed=physical_inputs(parents)
    route.warm_load()
    for _ in range(2):
        route.predict_physical(physical,observed)
        assert route.last_receipt['F_logical_rows']==4
        assert route.last_receipt['F_physical_rows']==4
        assert route.last_receipt['F_calls']==1
        assert route.last_receipt['F_invocations'][0]['actual_head_horizon']==64
    assert pool.instances_created==1
    assert route.resident_state()['F_instances_created']==1


def test_raw_executes_original_channels_and_single_inverse(parents):
    pool=FakePool(4)
    route=deployment.Route(dict(method='RAW',seed=92601),parents,device='cpu',foundation_pool=pool)
    physical,observed=physical_inputs(parents)
    result=route.predict_physical(physical,observed)
    expected=np.broadcast_to(physical[:,-1,None]+.125*parents['std'],(2,4,3))
    np.testing.assert_allclose(result,expected,atol=1e-6)
    assert route.last_receipt['F_logical_rows']==6
    assert route.last_receipt['selected_prefix']==3


def test_alpha_executes_both_parents_in_the_same_route(parents):
    pool=FakePool(4)
    route=deployment.Route(dict(method='ALPHA_fitted',seed=92601,alpha=.4),parents,device='cpu',foundation_pool=pool)
    physical,observed=physical_inputs(parents)
    x=torch.from_numpy((physical-parents['mean'])/parents['std'])
    with torch.no_grad():
        g=parents['g_model'](x)
        s=parents['s_model'](x,lambda z:z[:,-1:].expand(-1,4,-1)+.125)
        expected=((g+.4*(s-g)).double()*torch.from_numpy(parents['std']).double()+torch.from_numpy(parents['mean']).double()).numpy()
    np.testing.assert_allclose(route.predict_physical(physical,observed),expected,atol=1e-6)
    assert route.last_receipt['F_logical_rows']==4
    assert route.g_model is not None and route.s_model is not None


def test_mean_probe_is_a_real_cpu_residual_path(parents,tmp_path):
    path=tmp_path/'MEAN.npz'
    correction=np.full((4,3),.25,np.float64)
    np.savez_compressed(path,mean=correction)
    route=deployment.Route(dict(method='MEAN',seed=92601,state_path=str(path),state_sha256=common.sha(path)),parents,device='cpu')
    physical,observed=physical_inputs(parents)
    expected=np.broadcast_to(physical[:,-1,None]+.25*parents['std'],(2,4,3))
    np.testing.assert_allclose(route.predict_physical(physical,observed),expected,atol=1e-6)
    assert route.last_receipt['F_calls']==0


def test_changed_artifact_is_rejected(parents,tmp_path):
    spec=dict(method='RESIDUAL_Q',seed=92601,**artifact(tmp_path,parents['s_model'],'RESIDUAL_Q'))
    with open(spec['state_path'],'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='file SHA256'):
        deployment.Route(spec,parents,device='cpu')


def test_prefix_zero_does_not_load_correction_or_foundation(parents):
    pool=FakePool(4)
    route=deployment.Route(dict(method='RESIDUAL_F',seed=92601,prefix=0),parents,device='cpu',foundation_pool=pool)
    route.warm_load()
    route.predict_physical(*physical_inputs(parents))
    assert route.model is None
    assert pool.get_calls==0
    assert route.resident_state()['F_instances_created']==0


def test_v_access_gate_requires_completed_seal(tmp_path):
    with pytest.raises(ValueError,match='must be sealed'):
        deployment.assert_v_sealed(tmp_path)
    path=tmp_path/'C_DECISIONS.json'
    path.write_text(json.dumps({'status':'PARTIAL'}),encoding='utf8')
    with pytest.raises(ValueError,match='must be sealed'):
        deployment.assert_v_sealed(tmp_path)
    path.write_text(json.dumps({'status':'SEALED_BEFORE_V'}),encoding='utf8')
    assert deployment.assert_v_sealed(tmp_path)['status']=='SEALED_BEFORE_V'


def test_bundles_and_true_batch_per_origin_units():
    origins=np.arange(100,132,4,dtype=np.int64)
    assert len(costs.timing_origin_bundles(origins,1))==8
    np.testing.assert_array_equal(costs.timing_origin_bundles(origins,8)[0],origins)
    trials=[dict(phase='warmup',whole_seconds=99.) for _ in range(10)]
    trials += [dict(phase='timed',whole_seconds=.008) for _ in range(20)]
    stats=costs.latency_statistics(trials,8)
    assert stats['median_batch_ms']==8
    assert stats['median_origin_ms']==1
    assert stats['IQR_batch_ms']==0
    assert not stats['B8_origin_is_single_request_latency']
    with pytest.raises(ValueError,match='twenty'):
        costs.latency_statistics(trials[:-1],8)


def test_full_v_parity_checks_all_origins_and_exact_target_mask(parents,tmp_path,monkeypatch):
    monkeypatch.setattr(common,'REPORT',tmp_path)
    (tmp_path/'C_DECISIONS.json').write_text(json.dumps({'status':'SEALED_BEFORE_V'}),encoding='utf8')
    task=dict(parents['task'],physical=np.broadcast_to(parents['mean'],(600,3)).copy(),
              observed=np.ones((600,3),dtype=bool))
    route=deployment.Route(dict(method='G',seed=92601),parents,task=task,device='cpu')
    origins=costs.fixed_origins(task)
    reference=dict(pred_normalized=np.zeros((22,4,3),np.float32),target_normalized=np.zeros((22,4,3),np.float32),
                   observed=np.ones((22,4,3),bool),origins=origins,channel_ids=np.asarray(task['columns']))
    result=costs.full_v_parity(route,task,origins,reference,{'batch_size':8})
    assert result['status']=='PASS' and result['n_origins']==22
    assert result['metric']['mse']==0
    reference['observed'][0,0,0]=False
    with pytest.raises(ValueError,match='cohort differs'):
        costs.full_v_parity(route,task,origins,reference,{'batch_size':8})


def test_measurement_runs_ten_warmups_twenty_actual_predictions(parents):
    route=deployment.Route(dict(method='G',seed=92601),parents,device='cpu')
    physical,observed=physical_inputs(parents,batch=1)
    trials=costs.measured_trials(route,[(physical,observed)],request={})
    assert len(trials)==30
    assert sum(row['phase']=='warmup' for row in trials)==10
    assert sum(row['phase']=='timed' for row in trials)==20
    assert all(row['F_calls']==0 for row in trials)
    assert all(row['peak_allocated_MiB'] is None for row in trials)
    assert not torch.cuda.is_initialized()


def test_cuda_peak_scope_resets_only_before_twenty_timed_trials(monkeypatch):
    calls=[]
    monkeypatch.setattr(costs,'_sync',lambda device:None)
    monkeypatch.setattr(costs,'_memory',lambda device:dict(allocated_MiB=1.,reserved_MiB=2.,peak_allocated_MiB=3.,peak_reserved_MiB=4.))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device:calls.append(device))
    class Route:
        device='cuda'
        last_receipt={}
        def predict_physical(self,x,m):
            self.last_receipt=dict(cached_forecast_or_features_reads=0,F_logical_rows=0,F_physical_rows=0,F_calls=0,
                                   F_instances_created=0,F_invocations=[],selected_prefix=0)
            return np.zeros((len(x),1,1))
    trials=costs.measured_trials(Route(),[(np.zeros((1,2,1)),np.ones((1,2,1),bool))],request={})
    assert len(calls)==20
    assert all(r['peak_allocated_MiB'] is None for r in trials if r['phase']=='warmup')
    assert all(r['peak_allocated_MiB']==3. for r in trials if r['phase']=='timed')


def c_artifact(tmp_path,parent,method,selected_prefix=1):
    model=blocks.LatentBlocks(parent,method)
    with torch.no_grad():
        model.D_blocks[0].fill_(.01)
        model.D_blocks[1].fill_(.02)
    path=tmp_path/(method+'.pt')
    torch.save(dict(schema='incremental_C_final_model_v1',model=model.state_dict(),
                    method=method,selected_prefix=selected_prefix),path)
    seal=tmp_path/'SELECTION_SEAL.json'
    seal.write_text(json.dumps({'selected_prefix':selected_prefix}),encoding='utf8')
    return dict(state_path=str(path),state_sha256=common.sha(path),
                model_state_sha256=deployment.state_hash(model),selection_seal_sha256=common.sha(seal))


@pytest.mark.parametrize('method',blocks.METHODS)
def test_C_zero_prefix_is_G_only_without_S_or_model_artifact(parents,method):
    only_g={key:value for key,value in parents.items() if key!='s_model'}
    pool=FakePool(4)
    route=deployment.Route(dict(method=method,seed=92601,prefix=0,artifact_selected_prefix=1),
                           only_g,device='cpu',foundation_pool=pool)
    route.warm_load()
    physical,observed=physical_inputs(parents)
    expected=np.broadcast_to(physical[:,-1,None],(2,4,3))
    np.testing.assert_allclose(route.predict_physical(physical,observed),expected,atol=1e-6)
    assert route.model is None and route.s_model is None
    assert pool.get_calls==0 and route.last_receipt['F_calls']==0
    assert route.last_receipt['artifact_selected_prefix']==1
    assert route.last_receipt['executed_prefix']==0
    assert route.last_receipt['G_only_without_F_resident']


@pytest.mark.parametrize('method',blocks.METHODS)
def test_C_uses_sealed_prefix_by_default_and_preserves_it_under_override(parents,tmp_path,method):
    artifact=c_artifact(tmp_path,parents['s_model'],method)
    physical,observed=physical_inputs(parents)
    normalized=torch.as_tensor((physical-parents['mean'])/parents['std'])
    rows=[]
    resident=[]
    for override,expected_k in (({},1),({'prefix':2},2)):
        pool=FakePool(4)
        spec=dict(method=method,seed=92601,**artifact,**override)
        route=deployment.Route(spec,parents,device='cpu',foundation_pool=pool)
        restored=blocks.restore(parents['s_model'],artifact['state_path'])
        restored.prefix_k=expected_k
        with torch.no_grad():
            prediction=parents['g_model'](normalized)+restored(
                normalized,lambda z:z[:,-1:].expand(-1,4,-1)+.125)
            expected=(prediction.double()*torch.from_numpy(parents['std']).double()+torch.from_numpy(parents['mean']).double()).numpy()
        np.testing.assert_allclose(route.predict_physical(physical,observed),expected,atol=1e-6)
        assert pool.instances_created==1
        assert route.last_receipt['F_calls']==1
        assert route.last_receipt['F_logical_rows']==2*expected_k
        assert route.last_receipt['artifact_selected_prefix']==1
        assert route.last_receipt['executed_prefix']==expected_k
        assert route.last_receipt['prefix_is_selected']==(expected_k==1)
        rows.append(route.last_receipt['F_logical_rows'])
        resident.append(route.resident_state()['resident_parameters'])
    assert rows==[2,4]
    assert resident[0]==resident[1]


def test_C_parent_loader_skips_S_only_for_zero_prefix(parents,monkeypatch):
    loaded=[]
    def parent(ctx,task,dataset,role,seed):
        loaded.append(role)
        return parents['g_model' if role=='G' else 's_model'],parents['mean'],parents['std'],{'role':role}
    monkeypatch.setattr(costs.data_contract,'_parent',parent)
    base=dict(dataset='jena',seed=92601,method='JOINT_PREFIX')
    costs._load_parent_models({},parents['task'],dict(base,prefix=0,artifact_selected_prefix=1))
    assert loaded==['G']
    loaded.clear()
    costs._load_parent_models({},parents['task'],dict(base,prefix=1,artifact_selected_prefix=1))
    assert loaded==['G','S']


def test_C_default_zero_uses_only_light_selection_metadata(parents,tmp_path,monkeypatch):
    spec=dict(method='SEQ_BLOCKS',seed=92601,
              **c_artifact(tmp_path,parents['s_model'],'SEQ_BLOCKS',selected_prefix=0))
    def forbidden(*args,**kwargs):
        raise AssertionError('Zero prefix must not deserialize correction weights')
    monkeypatch.setattr(torch,'load',forbidden)
    only_g={key:value for key,value in parents.items() if key!='s_model'}
    pool=FakePool(4)
    route=deployment.Route(spec,only_g,device='cpu',foundation_pool=pool)
    route.warm_load()
    route.predict_physical(*physical_inputs(parents))
    assert route.k==0 and route.model is None
    assert route.last_receipt['artifact_selected_prefix']==0
    assert route.last_receipt['prefix_is_selected']
    assert pool.get_calls==0


def test_C_zero_with_already_resident_F_is_labelled_separately(parents):
    pool=FakePool(4)
    pool.get(4)
    route=deployment.Route(dict(method='JOINT_FULL',seed=92601,prefix=0),
                           parents,device='cpu',foundation_pool=pool)
    route.predict_physical(*physical_inputs(parents))
    assert route.last_receipt['F_calls']==0
    assert not route.last_receipt['G_only_without_F_resident']
    assert route.resident_state()['F_instances_created']==1


def test_C_rejects_unknown_prefix_and_changed_selection_metadata(parents,tmp_path):
    spec=dict(method='JOINT_FULL',seed=92601,**c_artifact(tmp_path,parents['s_model'],'JOINT_FULL'))
    with pytest.raises(ValueError,match='prefix'):
        deployment.Route(dict(spec,prefix=3),parents,device='cpu')
    (tmp_path/'SELECTION_SEAL.json').write_text(json.dumps({'selected_prefix':0}),encoding='utf8')
    with pytest.raises(ValueError,match='selection'):
        deployment.Route(spec,parents,device='cpu')
