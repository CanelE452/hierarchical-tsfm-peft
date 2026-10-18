import importlib
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import psutil
import torch
from torch import nn

from research.b2_revin_20261018_v1 import contracts


def cost_module():
    try:
        return importlib.import_module('research.b2_revin_20261018_v1.costs')
    except ModuleNotFoundError:
        pytest.fail('The registered final-model cost worker is missing')


def final_rows():
    rows = []
    for cohort in ('jena', 'pems32'):
        for enabled in (False, True):
            for seed in (92601, 92602):
                rows.append(dict(dataset=cohort, family='B2', method='B2_REVIN_ON' if enabled else 'B2_REVIN_OFF',
                    run_id=contracts.job_id(cohort, enabled, seed)+'_cap6', job_id=contracts.job_id(cohort, enabled, seed),
                    cap=6, revin=enabled, seed=seed, final=True, status='COMPLETE', evaluation='V_select',
                    independent_FP64='PASS', prediction_binding={'reference':'cache://prediction.npz','sha256':'a'*64},
                    bundle_binding={'reference':'cache://bundle.pt','sha256':'b'*64}))
    for seed in (92601, 92602):
        rows.append(dict(dataset='jena', family='B0', method='B0_REVIN_ON',
            run_id=contracts.job_id('jena', True, seed, 'B0')+'_cap6', job_id=contracts.job_id('jena', True, seed, 'B0'),
            cap=6, revin=True, seed=seed, final=True, status='COMPLETE', evaluation='V_select',
            independent_FP64='PASS', prediction_binding={'reference':'cache://prediction.npz','sha256':'a'*64},
            bundle_binding={'reference':'cache://bundle.pt','sha256':'b'*64}))
    for cohort, method, family in [('jena','Raw','RAW'),('jena','LoRA_GPU_replica','LORA'),
                                  ('pems32','Raw','RAW'),('pems32','DirectG','DIRECT_G'),('jena','Ridge','RAW_RIDGE')]:
        rows.append(dict(dataset=cohort, method=method, family=family, run_id='BASELINE_'+cohort+'_'+method,
            cap=0, final=True, status='COMPLETE', seed=92601, revin=False, evaluation='V_select',
            independent_FP64='PASS', prediction_binding={'reference':'cache://prediction.npz','sha256':'a'*64},
            bundle_binding=None if family=='RAW' else {'reference':'cache://bundle.pt','sha256':'b'*64}))
    return rows


def decisions():
    return dict(status='COMPLETE', decisions=[dict(cohort=cohort, cap=cap, common_revin=cohort=='jena')
                 for cohort in ('jena','pems32') for cap in (3,6)])


def test_only_final_registered_fourteen_models_receive_gpu_facets():
    costs = cost_module()
    rows = final_rows()
    old = dict(rows[0], run_id=rows[0]['job_id']+'_cap3', cap=3, final=False)
    selected = costs.select_final_rows(rows+[old], decisions())
    assert len(selected) == 14
    assert all(row['family'] != 'RAW_RIDGE' for row in selected)
    assert all(row['cap'] in (0,6) for row in selected)
    assert len({row['run_id'] for row in selected}) == 14


def test_wrong_cap_or_incomplete_group_is_rejected_before_measurement():
    costs = cost_module()
    rows = final_rows()
    with pytest.raises(ValueError, match='cap'):
        costs.select_final_rows([dict(rows[0],cap=3)]+rows[1:], decisions())
    with pytest.raises(ValueError, match='coverage'):
        costs.select_final_rows(rows[1:], decisions())
    with pytest.raises(ValueError, match='bool'):
        costs.select_final_rows([dict(rows[0],final='true')]+rows[1:], decisions())


class FakePool:
    def __init__(self, model_path, device, factory):
        self.model_path, self.device, self.factory = model_path, device, factory
        self.foundation = None
        self.instances_created = 0
        self.closed = False

    def close(self):
        self.closed = True


class FakeLoadedRoute:
    def __init__(self, spec, task, cache_root, **kwargs):
        self.spec, self.task, self.cache_root = spec, task, cache_root
        self.pool = kwargs['foundation_pool']
        self.family = spec['family']
        self.mean, self.std = np.zeros(3,np.float32),np.ones(3,np.float32)
        self.owns_pool = False
        self.closed = False

    def close(self):
        self.closed = True
        if self.owns_pool:
            self.pool.close()


def fixture_context(tmp_path):
    ctx = {key:tmp_path/key for key in ('repo','asset','report','cache')}
    for path in ctx.values():
        path.mkdir()
    ctx['modules'] = dict(deployment=SimpleNamespace(FoundationPool=FakePool,Route=FakeLoadedRoute),
        fit=SimpleNamespace(scaler_fingerprint=lambda mean,std:'fixture-scaler'),
        legacy=SimpleNamespace(MODEL_REL='models/pinned',F_SHA='f'*64))
    ctx['foundation_loader'] = lambda *args: None
    return ctx


@pytest.mark.parametrize('family,enabled,expected', [('B2',None,'legacy'),('B2',False,'revin'),
    ('B2',True,'revin'),('B0',True,'revin'),('LORA',None,'legacy'),('DIRECT_G',None,'legacy'),('RAW',None,'legacy')])
def test_loader_dispatch_preserves_exact_bundle_and_accuracy_identity(tmp_path, monkeypatch, family, enabled, expected):
    costs = cost_module()
    ctx = fixture_context(tmp_path)
    row = dict(run_id='unique_accuracy_cap6',job_id='base_job',dataset='jena',method=family,family=family,
               scaler_sha256='fixture-scaler',prediction_binding={'reference':'cache://prediction.npz','sha256':'a'*64})
    if family != 'RAW':
        spec = dict(family=family, run_id='base_job', device='cpu',latent_k=2)
        if enabled is not None:
            spec['revin'] = enabled
        bundle = dict(spec=spec,contract=dict(source_sha='s',config_sha='c',data_sha='d',model_sha='m'))
        path = ctx['cache']/'bundle.pt'
        torch.save(bundle,path)
        row['bundle_binding'] = contracts.binding(path,ctx)
    called = []

    def load_revin(spec,task,cache_root,**kwargs):
        called.append('revin')
        kwargs.pop('ctx')
        return FakeLoadedRoute(spec,task,cache_root,**kwargs)

    monkeypatch.setattr(costs,'load_route',load_revin)
    loader = costs.route_loader(row,dict(H=5,split_ranges={'A_stop':(75,90)}),ctx,dict(baselines=[]))
    first,second = loader(),loader()
    assert (called == ['revin','revin']) == (expected=='revin')
    assert first.pool is not second.pool
    assert first.owns_pool and second.owns_pool
    assert first.spec['run_id']=='unique_accuracy_cap6'
    assert first.spec['row_microbatch']==128
    if family != 'RAW':
        assert first.spec['artifact_sha256']==row['bundle_binding']['sha256']
        assert first.spec['artifact_cache_relative_path']=='bundle.pt'
    else:
        assert first.spec['foundation_sha']=='f'*64
    first.close();second.close()
    assert first.pool.closed and second.pool.closed


class PhysicalToyRoute:
    def __init__(self):
        self.device='cpu'
        self.family='B2'
        self.latent_k=2
        self.mean=np.zeros(3,np.float32)
        self.std=np.ones(3,np.float32)
        self.model=nn.Linear(3,3)
        self.pool=SimpleNamespace(foundation=SimpleNamespace(model=nn.Linear(3,2)))
        self.closed=False
        self.last_receipt={}
        self.calls=0

    def predict_physical(self,physical,observed):
        self.calls+=1
        self.last_receipt={'F_logical_streams':len(physical)*2}
        return np.repeat(physical[:,-1:],2,axis=1).astype(np.float64)

    def memory_state(self):
        return dict(foundation_instances_created=1)

    def close(self):
        self.closed=True


def toy_measure(routes,physical,observed,*,warmups,timed_trials,stage_trials,lease):
    assert (warmups,timed_trials,stage_trials)==(10,20,1)
    rows=[]
    for phase,count in (('warmup',warmups),('timed',timed_trials),('stage_probe',stage_trials)):
        for index in range(count):
            routes[0].predict_physical(physical,observed)
            rows.append(dict(phase=phase,batch_size=len(physical),whole_seconds=.001*(index+1),
                             F_logical_streams=len(physical)*2))
    return rows


def test_actual_parity_measurement_normalizes_latency_per_origin_and_closes_both_loads():
    costs=cost_module()
    physical=np.arange(40*3,dtype=np.float32).reshape(40,3)
    task=dict(physical=physical,observed=np.ones_like(physical,dtype=bool),L=8,H=2,
              columns=['a','b','c'],split_ranges={'V_select':(16,40)})
    origins=np.arange(16,36,2,dtype=np.int64)
    predictions=np.stack([np.repeat(physical[origin-1:origin],2,axis=0) for origin in origins])
    saved=dict(origins=origins,pred_normalized=predictions,channel_ids=np.array(task['columns']))
    created=[]

    def loader():
        route=PhysicalToyRoute();created.append(route);return route

    candidate=dict(run_id='toy_cap6',job_id='toy',method='B2_REVIN_ON',family='B2',dataset='jena',
                   seed=92601,revin=True,cap=6,final=True,C=3,K=2,L=8,H=2,
                   source_kind='HISTORICAL_VERIFIED',prediction_manifest_binding={'reference':'repo://manifest.json','sha256':'b'*64},
                   prediction_binding={'reference':'cache://toy.npz','sha256':'a'*64})
    result=costs.measure_one(candidate,task,saved,loader,batch=8,ctx={},device='cpu',measure=toy_measure)
    row=result['row']
    assert row['cost_valid'] is True
    assert row['accuracy_run_id']=='toy_cap6' and row['cap']==6
    assert row['median_batch_ms']==pytest.approx(10.5)
    assert row['median_ms_per_origin']==pytest.approx(10.5/8)
    assert row['F_params']==8 and row['adapter_params']==12
    assert row['cached_forecast_reads_in_timed_path']==0
    assert row['saved_V_parity']['status']=='PASS' and row['serialized_reload_saved_V_parity']['status']=='PASS'
    assert row['serialized_reload_parity']['status']=='PASS'
    assert row['F_rows_per_timed_batch']==[16]
    assert row['accuracy_source_kind']=='HISTORICAL_VERIFIED'
    assert row['accuracy_source_evidence']==[candidate['prediction_manifest_binding']]
    assert len(created)==2 and all(route.closed for route in created)
    assert created[1].calls==34


def test_completed_receipt_cannot_be_reused_under_changed_accuracy_or_trials(tmp_path):
    costs=cost_module()
    ctx=fixture_context(tmp_path)
    trial=ctx['report']/'trial.csv';trial.write_text('phase,batch_size\ntimed,8\n')
    identity=dict(accuracy_run_id='candidate_cap6',cap=6,batch_size=8,prediction_sha='a'*64)
    row=dict(status='COMPLETE',cost_valid=True,accuracy_run_id='candidate_cap6',cap=6,batch_size=8,device='cuda')
    receipt=dict(status='COMPLETE',identity=identity,row=row,trials_binding=contracts.binding(trial,ctx))
    assert costs.validate_complete_receipt(receipt,identity,ctx)==row
    with pytest.raises(ValueError,match='identity'):
        costs.validate_complete_receipt(receipt,dict(identity,cap=3),ctx)
    trial.write_text('changed')
    with pytest.raises(ValueError,match='SHA'):
        costs.validate_complete_receipt(receipt,identity,ctx)


def test_historical_ridge_context_requires_exact_saved_archive_and_receipt(tmp_path):
    costs=cost_module()
    ctx=fixture_context(tmp_path)
    prediction=ctx['cache']/'ridge.npz';prediction.write_bytes(b'verified Ridge prediction fixture')
    bundle=ctx['cache']/'ridge-artifact.npz';bundle.write_bytes(b'verified Ridge artifact fixture')
    candidate=dict(run_id='BASELINE_jena_Ridge',final=True,dataset='jena',family='RAW_RIDGE',seed=92601,
                   prediction_binding=contracts.binding(prediction,ctx),bundle_binding=contracts.binding(bundle,ctx))
    relative='reports/adapter_diagnostics_20261017_v2/receipts/E01_jena_matched.json'
    source=ctx['repo']/relative;source.parent.mkdir(parents=True);source.write_text('{}')
    path=ctx['repo']/'reports/adapter_diagnostics_20261017_v2/COST_FULL_PRECISION.csv'
    rows=[dict(job_id='E01_jena_matched',method='Ridge',device='cpu',B=batch,
               accuracy_archive_sha256=contracts.sha(prediction),status='COMPLETE',parity_state='PASS',
               source_forward_parity='PASS',source_file=relative,source_sha256=contracts.sha(source),
               median_ms_per_origin=.4/batch,median_ms_per_batch=.4,IQR_ms_per_origin=.02/batch,
               scope='whole physical inference',warmups=10,repetitions=20,trainable_params=33264)
          for batch in (1,8)]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=costs._ridge_context([candidate],ctx)
    assert {row['batch_size'] for row in result}=={1,8}
    assert all(row['source_kind']=='HISTORICAL_CPU_CONTEXT' and row['device']=='cpu' for row in result)
    assert all(row['p95_ms_per_origin'] is None for row in result)
    assert all(row['accuracy_run_id']=='BASELINE_jena_Ridge' and row['cap']==0 for row in result)
    prediction.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        costs._ridge_context([candidate],ctx)


def test_loader_rejects_accuracy_contract_change_before_a_route_exists(tmp_path):
    costs=cost_module()
    ctx=fixture_context(tmp_path)
    path=ctx['cache']/'bundle.pt'
    torch.save(dict(spec=dict(family='B2',run_id='original',device='cpu'),
                    contract=dict(source_sha='s',config_sha='c',data_sha='d',model_sha='m')),path)
    row=dict(run_id='candidate_cap6',family='B2',dataset='jena',source_sha='another-source',
             bundle_binding=contracts.binding(path,ctx))
    with pytest.raises(ValueError,match='scored accuracy contract'):
        costs.route_loader(row,dict(H=5,split_ranges={'A_stop':(75,90)}),ctx,dict(baselines=[]))


def terminal_process():
    child=subprocess.Popen([sys.executable,'-B','-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE)
    try:
        identity=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time())
    finally:
        child.stdin.close();child.wait(timeout=10)
    return identity


def running_receipt_fixture(tmp_path,worker,**extra):
    ctx=fixture_context(tmp_path)
    prediction=ctx['cache']/'prediction.npz';prediction.write_bytes(b'bound prediction')
    bundle=ctx['cache']/'bundle.pt';bundle.write_bytes(b'bound model')
    identity=dict(accuracy_run_id='fixture_cap6',cap=6,batch_size=8,
        prediction_binding=contracts.binding(prediction,ctx),bundle_binding=contracts.binding(bundle,ctx),
        source_hashes={'frozen_cost_producer':'f'*64},protocol=dict(device='cuda',warmups=10,timed_trials=20))
    path=ctx['report']/'costs/fixture_cap6_B8.json'
    contracts.atomic_json(path,dict(status='RUNNING',identity=identity,attempt=1,
        worker_pid=worker['pid'],worker_create_time=worker['create_time'],authorized_children=[],**extra))
    return ctx,path,identity


def test_dead_running_cost_registers_identical_retry_and_preserves_old_bytes(tmp_path):
    costs=cost_module()
    assert hasattr(costs,'resume_cost_attempt'), 'Stale RUNNING recovery is missing'
    ctx,path,identity=running_receipt_fixture(tmp_path,terminal_process())
    previous=path.read_bytes()
    orphan=path.with_suffix('.trials.csv');orphan.write_bytes(b'incomplete timing attempt')
    retry=costs.resume_cost_attempt(path,identity,ctx)
    assert retry['status']=='RETRY_REGISTERED' and retry['identity']==identity and retry['attempt']==2
    archived=contracts.resolve_binding(retry['previous_running_binding'],ctx)
    assert archived.read_bytes()==previous
    registration=contracts.read(contracts.resolve_binding(retry['retry_registration_binding'],ctx))
    assert registration['previous_worker_evidence']['terminal'] is True
    assert registration['previous_worker_evidence']['pid']==contracts.read(archived)['worker_pid']
    assert registration['restart_scope']=='entire same facet; same inputs/source/protocol'
    assert registration['orphan_trial_bindings']==[contracts.binding(orphan,ctx)]
    assert orphan.read_bytes()==b'incomplete timing attempt'
    assert costs._trial_path(path,2)!=orphan
    after=path.read_bytes()
    assert costs.resume_cost_attempt(path,identity,ctx)==retry
    assert path.read_bytes()==after


def test_live_running_cost_is_refused_without_receipt_mutation(tmp_path):
    costs=cost_module()
    assert hasattr(costs,'resume_cost_attempt'), 'Stale RUNNING recovery is missing'
    process=psutil.Process()
    ctx,path,identity=running_receipt_fixture(tmp_path,dict(pid=process.pid,create_time=process.create_time()))
    previous=path.read_bytes()
    with pytest.raises(RuntimeError,match='alive'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous
    assert not (path.parent/'attempts').exists()


def test_live_authorized_child_blocks_dead_worker_recovery(tmp_path):
    costs=cost_module()
    assert hasattr(costs,'resume_cost_attempt'), 'Stale RUNNING recovery is missing'
    ctx,path,identity=running_receipt_fixture(tmp_path,terminal_process())
    old=contracts.read(path)
    process=psutil.Process()
    old['authorized_children']=[dict(pid=process.pid,create_time=process.create_time())]
    contracts.atomic_json(path,old)
    previous=path.read_bytes()
    with pytest.raises(RuntimeError,match='alive'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous


@pytest.mark.parametrize('status',['FAILED','COMPLETE'])
def test_terminal_cost_receipt_is_not_replaced_by_retry_registration(tmp_path,status):
    costs=cost_module()
    assert hasattr(costs,'resume_cost_attempt'), 'Stale RUNNING recovery is missing'
    ctx,path,identity=running_receipt_fixture(tmp_path,terminal_process())
    value=contracts.read(path);value['status']=status;contracts.atomic_json(path,value)
    previous=path.read_bytes()
    with pytest.raises(ValueError,match='terminal'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous


def test_active_lease_authorized_child_also_blocks_recovery(tmp_path):
    costs=cost_module()
    worker=terminal_process()
    ctx,path,identity=running_receipt_fixture(tmp_path,worker)
    process=psutil.Process()
    contracts.atomic_json(ctx['cache']/'ACTIVE_GPU_LEASE.json',dict(**worker,
        authorized_children=[dict(pid=process.pid,create_time=process.create_time())]))
    previous=path.read_bytes()
    with pytest.raises(RuntimeError,match='alive'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous


def test_unverifiable_worker_liveness_does_not_register_a_retry(tmp_path,monkeypatch):
    costs=cost_module()
    ctx,path,identity=running_receipt_fixture(tmp_path,terminal_process())
    previous=path.read_bytes()

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(costs.psutil,'Process',denied)
    with pytest.raises(RuntimeError,match='cannot be established'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous
    assert not (path.parent/'attempts').exists()


def test_stale_running_with_changed_input_bytes_does_not_register_retry(tmp_path):
    costs=cost_module()
    ctx,path,identity=running_receipt_fixture(tmp_path,terminal_process())
    previous=path.read_bytes()
    contracts.resolve_binding(identity['prediction_binding'],ctx).write_bytes(b'changed input')
    with pytest.raises(ValueError,match='SHA'):
        costs.resume_cost_attempt(path,identity,ctx)
    assert path.read_bytes()==previous
    assert not (path.parent/'attempts').exists()
