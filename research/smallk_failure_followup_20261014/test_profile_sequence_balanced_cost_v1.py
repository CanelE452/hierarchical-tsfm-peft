"""Meaningful synthetic tests prepared ONLY; not run before human extension."""
from copy import deepcopy
import inspect
from pathlib import Path
import numpy as np
import pytest
from research.smallk_failure_followup_20261014 import profile_sequence_balanced_cost_v1 as cost
from research.smallk_failure_followup_20261014 import profile_latency as base


def fixture():
    values=[]
    for d in cost.DATASETS:
        for s in cost.SEEDS:
            for h in cost.HEADS:
                cp=[{'new_update':u,'tune_actual_Y_mse':1-u/1000,'checkpoint':{'path':str(u)}} for u in (0,100,200,400,600)]
                values.append({'dataset':d,'seed':s,'head':h,'objective':'Y','status':'COMPLETE','checkpoints':cp,
                    'selected_index':4,'selected_new_update':600,'selected_checkpoint':cp[4]['checkpoint'],
                    'learning_executed':{'optimizer_updates':600,'full_learning_proof_pass':True,'E_updated':True,'D_updated':True,'Head_updated':True,'frozen_F_input_gradient_nonzero_diagnostic_count':5},
                    'F_frozen_pass':True,'F_has_no_parameter_gradients':True,'initial_fixed_basis_unchanged':True,
                    'initial_all_split_original_MAIN_parity':{k:{'pass':True} for k in ('fit','tune','val' if d=='robin' else 'eval')},
                    'original_MAIN_checkpoint':{'checkpoint':{'sha256':d+str(s)}},'initial_state_hashes':{'state_sha_old_algorithm':'same'},'training_order_sha256':str(s)})
    return values


def test_every_route_occupies_every_position_once_and_schedule_reproducible():
    schedule=cost.balanced_schedule()
    assert schedule==cost.balanced_schedule() and len(schedule)==7
    for block in schedule:assert set(block)==set(cost.ROUTES) and len(block)==7
    for route in cost.ROUTES:assert sorted(block.index(route) for block in schedule)==list(range(7))


def test_exact8_actual600_head_learning_same_initialization_and_selection_gates():
    assert len(cost.validate_producers(fixture()))==8
    with pytest.raises(ValueError,match='EXACT8'):cost.validate_producers(fixture()[:-1])
    values=fixture();values[0]['learning_executed']['Head_updated']=False
    with pytest.raises(ValueError,match='HEAD_F_PROOF'):cost.validate_producers(values)
    values=fixture();values[0]['checkpoints'][0]['tune_actual_Y_mse']=0.
    with pytest.raises(ValueError,match='TUNE_SELECTION'):cost.validate_producers(values)
    values=fixture();values[1]['training_order_sha256']='other'
    with pytest.raises(ValueError,match='SAME_INITIAL'):cost.validate_producers(values)


def test_audit_literal_PASS_and_exact8_producer_SHA_binding():
    values=fixture();records=[{'sha256':str(i)} for i in range(8)]
    rows=[{'dataset':p['dataset'],'seed':p['seed'],'head':p['head'],'producer_sha256':r['sha256']} for p,r in zip(values,records)]
    cost.audit_binding({'status':'PASS','failed_count':0,'conditions':rows},values,records)
    rows[0]['producer_sha256']='other'
    with pytest.raises(ValueError,match='SHA_BINDING'):cost.audit_binding({'status':'PASS','failed_count':0,'conditions':rows},values,records)


class Backend:
    def synchronize(self):pass
    def reset_memory(self):pass
    def to_numpy(self,value):return value.copy()
    def memory_snapshot(self):return {'synthetic':True}


class Pipeline:
    def __init__(self,backend):self.backend=backend;self.calls=0
    def whole(self,physical,observed,kind):
        assert kind=='vectorized';self.calls+=1
        future=np.repeat(physical[:,-1:],48,axis=1)
        return future,future*2+1


def save(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
    return {'path':str(path)}


def test_balanced_all7_outputs_paired_RAW_blocks_and_original_bulk_flags(tmp_path):
    backend=Backend();pipelines={n:Pipeline(backend) for n in cost.ROUTES}
    physical=np.zeros((1,512,2),np.float32);mask=np.ones_like(physical,bool)
    norm,actual=next(iter(pipelines.values())).whole(physical,mask,'vectorized')
    refs={n:{'normalized':norm,'physical':actual} for n in cost.ROUTES}
    bulk={n:{'normalized':norm+1,'physical':actual+2} for n in cost.ROUTES}
    rows=cost.measure_group(pipelines,physical,mask,refs,bulk,tmp_path,lambda:None,lambda:None,save,base)
    assert len(rows)==7 and all(r['status']=='COMPLETE' and r['repeats']==7 for r in rows)
    assert all(sorted(r['position'])==list(range(7)) for r in rows)
    assert all(len(r['paired_RAW_ratios'])==7 for r in rows)
    assert not any(p['original_bulk_normalized']['pass'] for r in rows for p in r['parity'])
    for route in cost.ROUTES:
        with np.load(tmp_path/route/'TIMED_OUTPUTS.npz',allow_pickle=False) as saved:assert saved['normalized'].shape==(7,1,48,2)


def test_budget_exception_retains_partial_samples_and_does_not_fabricate7(tmp_path):
    backend=Backend();pipelines={n:Pipeline(backend) for n in cost.ROUTES}
    physical=np.zeros((1,512,2),np.float32);mask=np.ones_like(physical,bool)
    norm,actual=next(iter(pipelines.values())).whole(physical,mask,'vectorized');refs={n:{'normalized':norm,'physical':actual} for n in cost.ROUTES}
    calls=[]
    def fast():
        calls.append(1)
        if len(calls)==18:raise RuntimeError('synthetic budget held')
    with pytest.raises(RuntimeError,match='budget held'):cost.measure_group(pipelines,physical,mask,refs,refs,tmp_path,fast,lambda:None,save,base)
    assert (tmp_path/'ROWS.json').is_file()
    import json
    rows=json.loads((tmp_path/'ROWS.json').read_text())['rows']
    assert rows and all(r['status']!='COMPLETE' and r['repeats']<7 for r in rows)


def test_authorization_before_assets_and_frozen_numeric_pipeline_used():
    source=inspect.getsource(cost)
    assert source.index('runtime.ContinuationLimit(')<source.index('        import torch')
    assert 'AUTHORIZATION.is_file()' in source and 'base.DeploymentPipeline(' in source
    assert 'def fill_vectorized' not in source and 'decomposed(' not in source
    assert 'restore_selected_state(' in source and 'worker.forward_trace' in source
    assert 'local_cap_seconds' in source and 'LOCAL_SECONDS=180' in source


class ArrayBackend(Backend):
    def tensor(self,value):return np.asarray(value,np.float32)


class ArrayCodec:
    def encode(self,x):return x[:,:,:1]*np.float32(.5)
    def decode(self,z):return np.concatenate((z,z*np.float32(2)),axis=2)
    def forward_path(self,x,f):
        z=self.encode(x);future=f(z)
        return x[:,-1:]+self.decode(future)-self.decode(z[:,-1:])


def array_forecaster(z):return z[:,-1:]+np.arange(48,dtype=np.float32)[None,:,None]*np.float32(.01)


def array_head(z,future):
    difference=future-z[:,-1:]
    return np.concatenate((difference*np.float32(.1),difference*np.float32(.2)),axis=2)


def array_trace(codec,f,head,x):
    z=codec.encode(x);future=f(z);residual=head(z,future)
    return x[:,-1:]+codec.decode(future)-codec.decode(z[:,-1:])+residual,z,future,residual


def stage_fixture(codec):
    physical=np.arange(1024,dtype=np.float32).reshape(1,512,2)
    observed=np.ones_like(physical,bool)
    pipeline=base.DeploymentPipeline(ArrayBackend(),codec,array_forecaster,np.array([1,2],np.float32),np.array([1,2],np.float32),None)
    return pipeline,physical,observed


def test_separate_stages_match_exact_whole_math_for_RAW_MAIN_and_new_head():
    for codec in (None,ArrayCodec(),cost.HeadRoute(ArrayCodec(),array_head,array_trace)):
        pipeline,physical,observed=stage_fixture(codec)
        expected,expected_physical=pipeline.whole(physical,observed,'vectorized');timings={}
        actual,actual_physical=cost.synchronized_stage_trace(pipeline,physical,observed,lambda:None,base,timings)
        assert set(timings)=={'fill','normalization','H2D','E','F','oldD','head','anchor_inverse','D2H'}
        assert np.array_equal(actual,expected) and np.array_equal(actual_physical,expected_physical)


def test_stage7_outputs_gate_and_sync_sum_scope_are_preserved(tmp_path):
    pipeline,physical,observed=stage_fixture(cost.HeadRoute(ArrayCodec(),array_head,array_trace))
    normalized,actual=pipeline.whole(physical,observed,'vectorized');reference={'normalized':normalized,'physical':actual}
    whole={'normalized':np.stack([normalized]*7),'physical':np.stack([actual]*7)};rows=[]
    row=cost.measure_stage_route(pipeline,physical,observed,reference,whole,tmp_path/'stages',lambda:None,lambda:None,save,base,rows)
    assert row['status']=='COMPLETE' and row['warmup_completed']==2 and row['repeats']==7
    assert row['stage_sum_is_whole_latency'] is False and len(row['parity'])==7 and rows==[row]


def test_stage_budget_midtrial_records_attempt_and_NOT_MEASURED_without_7_claim(tmp_path):
    pipeline,physical,observed=stage_fixture(ArrayCodec())
    normalized,actual=pipeline.whole(physical,observed,'vectorized');reference={'normalized':normalized,'physical':actual}
    whole={'normalized':np.stack([normalized]*7),'physical':np.stack([actual]*7)};rows=[];calls=[]
    def fast():
        calls.append(1)
        if len(calls)==50:raise RuntimeError('stage synthetic budget held')
    with pytest.raises(RuntimeError,match='budget held'):
        cost.measure_stage_route(pipeline,physical,observed,reference,whole,tmp_path/'stages',fast,lambda:None,save,base,rows)
    assert rows[0]['status']=='PARTIAL_OR_NOT_MEASURED_PRESERVED' and rows[0]['repeats']==0
    assert rows[0]['stage_outputs'] is None and rows[0]['attempted_stages'][0]['timings']
