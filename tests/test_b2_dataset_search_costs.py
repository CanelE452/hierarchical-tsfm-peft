import hashlib
import importlib
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_memory_budget_worker import ToySampler


def costs():
    try:return importlib.import_module('research.b2_dataset_search_20261019_v1.costs')
    except ModuleNotFoundError:pytest.fail('New arbitrary-width two-policy cost worker is missing')


def request(family='B2',K=12,policy='COMMON_CAP'):
    return dict(dataset='fixture',family=family,run_id=family+'_fixture_K'+str(K),seed=92601,
        K=K,row_policy=policy,factory='train:cost_inputs',config_path='private/config.json',
        prefix_path='private/prefix.npz',model_dir='private/model')


def fixture(family='B2',K=12):
    c=21
    physical=np.arange(1040*c,dtype=np.float32).reshape(1040,c)/100
    observed=np.ones_like(physical,bool)
    origins=np.arange(512,1040-48+1,48,dtype=np.int64)
    target=np.stack([physical[o:o+48] for o in origins])
    prediction=np.stack([np.repeat(physical[o-1:o],48,axis=0) for o in origins]).astype(np.float64)
    saved=dict(origins=origins,pred_normalized=prediction,target_normalized=target,
        observed=np.ones_like(target,bool),mean=np.zeros(c,np.float32),std=np.ones(c,np.float32),
        channel_ids=np.asarray([str(i) for i in range(c)]))
    task=dict(physical=physical,observed=observed,columns=saved['channel_ids'].tolist(),L=512,H=48,
        dataset='fixture',data_sha256='d'*64,split_ranges={'V_select':(512,1040)})
    class Route:
        def __init__(self):
            self.family=family;self.latent_k=K if family=='B2' else c
            self.device='cpu';self.spec={};self.mean=saved['mean'];self.std=saved['std']
            self.model=torch.nn.Linear(c,c).eval()
            f=torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False)
            self.pool=SimpleNamespace(instances_created=0 if family=='DIRECT_G' else 1,
                foundation=None if family=='DIRECT_G' else SimpleNamespace(model=f,initial_hash='f'*64,
                    checkpoint_inputs=False,row_microbatch=128))
            self.artifact_sha256='b'*64 if family!='RAW' else None
            self.last_receipt={};self.batches=[];self.closed=False
        def predict_physical(self,x,mask):
            self.batches.append(len(x));width=0 if family=='DIRECT_G' else c if family=='RAW' else K
            cap=self.spec.get('row_microbatch',128)
            self.last_receipt=dict(F_logical_streams=len(x)*width,F_physical_streams=len(x)*width,
                F_calls=math.ceil(len(x)*width/cap) if width else 0)
            return np.repeat(x[:,-1:].astype(np.float64),48,axis=1)
        def memory_state(self):
            return dict(resident_weights_gpu_MiB=0.,resident_weights_cpu_MiB=.1,active_adapter_MiB=.1)
        def close(self):self.closed=True;self.pool.foundation=None
    route=Route();loads=[]
    def loader():loads.append(1);return route
    candidate=dict(run_id=request(family,K)['run_id'],family=family,dataset='fixture',seed=92601,C=c,
        K=c if family=='RAW' else 0 if family=='DIRECT_G' else K,
        L=512,H=48,bundle_binding={'sha256':'b'*64} if family!='RAW' else None,
        prediction_binding={'reference':'cache://original.npz','sha256':'p'*64})
    return dict(task=task,saved=saved,candidate=candidate,route_loader=loader,loader=loader,
        expected_f_sha='f'*64,expected_f_params=1),route,loads


@pytest.mark.parametrize('K',[3,6,12])
def test_common_row_cap_is_fixed_across_model_K(K):
    c=costs();work=c.policy_work(request(K=K),21,8)
    assert work['requested_row_cap']==48 and work['streams']==8*K
    assert work['calls']==math.ceil(8*K/48)
    assert c.policy_work(request('RAW',6),21,8)['calls']==4


def test_default128_and_ragged_tail_keep_actual_work():
    c=costs()
    assert c.policy_work(request(policy='DEFAULT128'),21,8)==dict(q=6,width=12,streams=96,
        requested_row_cap=128,effective_row_cap=96,calls=1,row_sizes=[96])
    assert c.policy_work(request(),21,3)['row_sizes']==[36]
    assert c.policy_work(request('RAW',6),21,3)['row_sizes']==[48,15]
    assert c.policy_work(request('B2',162),321,8)['requested_row_cap']==648
    assert c.policy_work(request('B2',432),862,8)['row_sizes']==[1728,1728]
    assert c.policy_work(request('RAW',216),862,3)['row_sizes']==[1728,858]


def test_real_core_measurement_preserves_unequal_tail_states_and_thirty_trials():
    c=costs();inputs,route,loads=fixture()
    result,arrays=c.measure_inputs(request(),inputs,device='cpu',sampler_factory=ToySampler)
    assert result['status']=='COMPLETE' and result['row_policy']=='COMMON_CAP'
    assert loads==[1] and route.closed
    assert result['parity']['full_V_origins']==11 and result['parity']['last_batch_size']==3
    assert result['states_before']==result['states_after']
    assert len(result['trials'])==30 and sum(r['phase']=='timed' for r in result['trials'])==20
    assert {r['F_calls'] for r in result['trials']}=={2}
    assert result['original_reference']['row_cap']==128
    assert result['actual_F_width']==12 and result['q']==6
    np.testing.assert_allclose(arrays['pred_normalized'],inputs['saved']['pred_normalized'])


@pytest.mark.parametrize('family',['RAW','DIRECT_G'])
def test_new_reference_is_generated_without_refitting(family):
    c=costs();inputs,route,loads=fixture(family,6)
    inputs['saved']['pred_normalized']=None
    result,arrays=c.measure_inputs(request(family,6,'DEFAULT128'),inputs,device='cpu',sampler_factory=ToySampler)
    assert loads==[1] and result['original_reference']['generated_new'] is True
    assert np.isfinite(arrays['pred_normalized']).all()
    assert result['actual_F_width']==(21 if family=='RAW' else 0)
    assert result['K']==6 and result['model_K']==(21 if family=='RAW' else 0)
    assert result['original_reference']['row_cap']==(128 if family=='RAW' else None)


@pytest.mark.parametrize('bad',['nongrid','wrongRawK','missing_origin','reordered_origin','wrong_target'])
def test_cohort_rejects_unregistered_width_or_changed_full_grid(bad):
    c=costs();inputs,route,_=fixture();r=request()
    if bad=='nongrid':r['K']=5
    elif bad=='wrongRawK':r=request('RAW',12)
    elif bad=='missing_origin':
        inputs['saved']={k:v[:-1] if k in ('origins','pred_normalized','target_normalized','observed') else v
                         for k,v in inputs['saved'].items()}
    elif bad=='reordered_origin':inputs['saved']['origins']=inputs['saved']['origins'][::-1]
    else:inputs['saved']['target_normalized'][0,0,0]+=1
    with pytest.raises(ValueError):c.validate_cohort(route,inputs,r)


def test_facet_paths_include_policy_and_refuse_old_namespace(tmp_path):
    c=costs();root=tmp_path/c.NAME
    a=c.output_paths(request(),root/'report',root/'cache')
    b=c.output_paths(request(policy='DEFAULT128'),root/'report',root/'cache')
    assert a['receipt']!=b['receipt'] and a['receipt'].name==request()['run_id']+'_COMMON_CAP_B8.json'
    with pytest.raises(ValueError):c.output_paths(request(),tmp_path/'old/report',tmp_path/'old/cache')


def test_source_binding_mismatch_is_rejected(tmp_path):
    c=costs();source=tmp_path/'producer.py';source.write_text('original')
    pins={'producer.py':hashlib.sha256(source.read_bytes()).hexdigest()}
    c.verify_sources(pins,tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'):c.verify_sources(pins,tmp_path)


def test_resume_identity_binds_checkpoint_data_and_original_prediction(tmp_path):
    c=costs();r=request()
    config,prefix=tmp_path/'config.json',tmp_path/'prefix.npz'
    config.write_text('{}');prefix.write_bytes(b'approved-prefix')
    r.update(config_path=str(config),prefix_path=str(prefix))
    initial=c._identity(r,{})
    assert 'candidate_identity_sha256' in initial and 'data_sha256' in initial
    inputs=dict(candidate={'bundle_binding':{'sha256':'old'},'prediction_binding':{'sha256':'p'}},
                task={'data_sha256':'d'},expected_f_sha='f')
    before=c._identity(r,{},inputs)
    inputs['candidate']['bundle_binding']['sha256']='changed'
    after=c._identity(r,{},inputs)
    assert before['candidate_identity_sha256']!=after['candidate_identity_sha256']


def test_completed_resume_rejects_changed_npz_and_workspace(tmp_path):
    c=costs();root=tmp_path/c.NAME
    paths=c.output_paths(request(),root/'report',root/'cache')
    validation={'status':'PASS'};identity={'run_id':'fixture','cublas_workspace_config':None}
    paths['prediction'].parent.mkdir(parents=True);paths['prediction'].write_bytes(b'prediction')
    c._json(paths['validation'],validation)
    receipt=dict(status='COMPLETE',**identity,parity=validation,prediction_file_sha256=c.sha(paths['prediction']))
    c._json(paths['receipt'],receipt)
    assert c._reuse(paths,identity)==receipt
    with pytest.raises(ValueError,match='environment'):
        c._reuse(paths,dict(identity,cublas_workspace_config=':4096:8'))
    paths['prediction'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding'):c._reuse(paths,identity)


def test_factory_contract_failure_is_preserved_without_gpu_or_path_leak(tmp_path,monkeypatch):
    c=costs();root=tmp_path/c.NAME
    def bad(*args):raise ValueError('Private C:/secret/input.npz')
    monkeypatch.setattr(c.importlib,'import_module',lambda name:SimpleNamespace(cost_inputs=bad))
    with pytest.raises(ValueError,match='Private'):
        c.execute_request(request(),root/'report',root/'cache')
    paths=c.output_paths(request(),root/'report',root/'cache')
    assert json.loads(paths['receipt'].read_text())['status']=='FAILED'
    assert 'secret/input' not in paths['receipt'].read_text()


@pytest.mark.parametrize('workspace',[None,':16:8'])
def test_cli_spawns_fresh_process_with_parent_workspace_unchanged(tmp_path,monkeypatch,workspace):
    c=costs()
    if workspace is None:monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    else:monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',workspace)
    root=tmp_path/c.NAME;root.mkdir()
    p=root/'request.json';p.write_text(json.dumps(request()))
    calls=[]
    def spawn(command,**kwargs):calls.append((command,kwargs['env']));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess,'run',spawn)
    assert c.main(['--request',str(p),'--report-dir',str(root/'report'),'--cache-dir',str(root/'cache')])==0
    assert '--worker' in calls[0][0]
    actual=calls[0][1].get('CUBLAS_WORKSPACE_CONFIG')
    assert actual==workspace and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==workspace


def test_cli_rejects_request_outside_new_namespace(tmp_path,monkeypatch):
    c=costs();p=tmp_path/'request.json';p.write_text(json.dumps(request()))
    root=tmp_path/c.NAME
    def no_process(*args,**kwargs):pytest.fail('Invalid request must not launch a worker')
    monkeypatch.setattr(c.subprocess,'run',no_process)
    with pytest.raises(ValueError,match='namespace'):
        c.main(['--request',str(p),'--report-dir',str(root/'report'),'--cache-dir',str(root/'cache')])
