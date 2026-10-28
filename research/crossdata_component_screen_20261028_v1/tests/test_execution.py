"""Bounded CPU-only execution and host registration checks."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research.crossdata_component_screen_20261028_v1 import runtime,run_one,host_gpu_gate


def job(dataset='pems32',method='OURS',seed=92601):
    return dict(job_id=f'XC_{method}_{dataset}_K{runtime.KS[dataset]}_s{seed}',
                method_id=method,dataset=dataset,K=runtime.KS[dataset],seed=seed,
                training_cap=runtime.DATASETS[dataset]['cap'])


@pytest.mark.parametrize('dataset',['pems32','beijing'])
@pytest.mark.parametrize('method',list(run_one.METHOD_FAMILY))
def test_exact_dataset_horizon_and_fixed_schedules(dataset,method):
    for seed in (92601,92602,92603):
        j=job(dataset,method,seed);run_one.validate_job(j)
        assert run_one.schedule(j)==([.25,1.,3.,6.] if dataset=='pems32' else [.25,1.,3.])
        spec=run_one.build_spec(j)
        assert spec['E_trainable']==(method!='E_FIXED') and spec['family']=='B2'
        assert spec['effective_batch']==16 and spec['microbatch']==4


@pytest.mark.parametrize('change',[{'seed':92604},{'training_cap':3},{'K':6},{'dataset':'jena'},
                                  {'method_id':'P_CHANNEL_ONLY'},{'job_id':'old'},{'extended':True}])
def test_unregistered_axes_are_rejected(change):
    with pytest.raises(ValueError):run_one.validate_job(dict(job(),**change))


def test_selection_is_complete_A_stop_and_earliest_exact_tie():
    search=dict(summary=dict(status='SEGMENT_COMPLETE',selected_epoch=.25,selected_astop_mse=1.),
                curves=[dict(epoch=e,astop_mse=v) for e,v in [(0.,2.),(.25,1.),(1.,1.),(3.,3.),(6.,4.)]])
    assert run_one.choose_epoch(search,[.25,1.,3.,6.])==(.25,1.)
    search['curves'].pop()
    with pytest.raises(ValueError):run_one.choose_epoch(search,[.25,1.,3.,6.])


@pytest.mark.parametrize('mutation',[None,'A_stop','prefix','tail','registry_scaler'])
def test_dataset_prefix_boundaries_and_registry_fail_closed(monkeypatch,mutation):
    task=dict(dataset='pems32',L=512,H=12,physical=np.zeros((14284,32)),observed=np.ones((14284,32),bool),
              split_ranges=dict(A_fit=[0,8928],A_stop=[8928,10713],V_select=[12499,14284],T_final=[14284,17856]))
    access=dict(new_T_numeric_values_parsed=0,members=[dict(tail_numeric_values_parsed=0)])
    description=dict(normalization_sha256={'A_fit':'fit','A60':'refit'})
    monkeypatch.setattr(runtime,'describe_task',lambda task:description)
    expected=copy.deepcopy(description)
    if mutation=='A_stop':task['split_ranges']['A_stop'][1]+=1
    elif mutation=='prefix':task['physical']=task['physical'][:-1]
    elif mutation=='tail':access['members'][0]['tail_numeric_values_parsed']=1
    elif mutation=='registry_scaler':expected['normalization_sha256']['A60']='changed'
    if mutation:
        with pytest.raises(ValueError):runtime.validate_task(task,access,expected)
    else:assert runtime.validate_task(task,access,expected)==description


def setup_lease(tmp_path,monkeypatch):
    report,cache,upstream=tmp_path/'report',tmp_path/'cache',tmp_path/'upstream'
    e_report=tmp_path/'reports/ppt_e_fixed_20261028_v1'
    for p in (report,cache,upstream,e_report):p.mkdir(parents=True)
    for key,value in [('REPORT',report),('CACHE',cache),('UPSTREAM_REPORT',upstream),('REPO',tmp_path)]:
        monkeypatch.setattr(runtime,key,value)
    (report/'REGISTRATION_SEAL.json').write_text('{}')
    seal=runtime.sha(report/'REGISTRATION_SEAL.json')
    (report/'ROOT_LAUNCH_GATE.json').write_text(json.dumps(dict(status='PASS',allow_gpu=True,registration_seal_sha256=seal)))
    for p in (upstream,e_report):(p/'EXECUTION_COMPLETION.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE')))
    (cache/'GPU_LEASE.json').write_text(json.dumps(dict(status='ACTIVE',token='fixed',pid=100,created_utc='owner',registration_seal_sha256=seal)))
    line=f'python -m research.{runtime.NAME}.run_one --lease-token fixed'
    records={100:dict(pid=100,ppid=50,created_utc='owner',command_line=f'python -m research.{runtime.NAME}.controller'),
             200:dict(pid=200,ppid=100,created_utc='wrapper',command_line=line),
             300:dict(pid=300,ppid=200,created_utc='worker',command_line=line),
             50:dict(pid=50,ppid=1,created_utc='launcher',command_line=f'python -m research.{runtime.NAME}.controller')}
    monkeypatch.setattr(runtime,'process_snapshot',lambda:records)
    monkeypatch.setattr(runtime.os,'getpid',lambda:300)
    return records


@pytest.mark.parametrize('mutation',[None,'token','pid_reused','parent','main_live','E_live','cost_live'])
def test_root_lease_windows_wrapper_and_existing_campaign_exclusion(tmp_path,monkeypatch,mutation):
    records=setup_lease(tmp_path,monkeypatch);token='fixed'
    if mutation=='token':token='wrong'
    elif mutation=='pid_reused':records[100]['created_utc']='replacement'
    elif mutation=='parent':records[200]['command_line']='python unrelated.py'
    elif mutation in ('main_live','E_live','cost_live'):
        module={'main_live':'ppt_justification_20261028_v1.run_one','E_live':'ppt_e_fixed_20261028_v1.run_one','cost_live':'ppt_justification_20261028_v1.cost_profile'}[mutation]
        records[400]=dict(pid=400,command_line='python -m research.'+module)
    if mutation:
        with pytest.raises((ValueError,RuntimeError)):runtime.require_active_lease(token)
    else:assert [x['pid'] for x in runtime.require_active_lease(token)['verified_ancestry']]==[300,200,100,50]


@pytest.mark.parametrize('other',[None,'cost','E','unknown_GPU'])
def test_new_host_module_is_allowed_but_other_campaigns_and_unknown_contexts_rejected(other):
    policy=dict(device=dict(gpu_uuid='synthetic'),desktop_processes=[],kernel_engine_provider=None)
    records={1:dict(pid=1,created=1.,command=['python','-m',f'research.{runtime.NAME}.run_one'])}
    allowed={1:1.};contexts=[]
    if other in ('cost','E'):
        module='research.ppt_justification_20261028_v1.cost_profile' if other=='cost' else 'research.ppt_e_fixed_20261028_v1.run_one'
        records[2]=dict(pid=2,created=2.,command=['python','-m',module]);allowed[2]=2.
    elif other=='unknown_GPU':contexts=[dict(pid=99,gpu_uuid='synthetic')]
    if other:
        with pytest.raises(RuntimeError):host_gpu_gate.validate_identity(policy,records,contexts,allowed)
    else:assert host_gpu_gate.validate_identity(policy,records,contexts,allowed)['allowed_processes']=={'1':1.}


def test_host_receipt_precedes_any_model_loading_and_attempts_are_immutable(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'report');monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(run_one,'validate_registration',lambda job:dict(seal='synthetic'))
    monkeypatch.setattr(runtime,'guard_preserved',lambda:dict(changed=[]))
    monkeypatch.setattr(runtime,'record_opens',lambda:set())
    j=job();public=runtime.REPORT/'jobs'/j['job_id']/'attempt_001';gate=dict(status='PASS',mode='identity_only')
    def stop():
        assert runtime.read(public/'HOST_GPU_GATE.json')==gate
        raise RuntimeError('synthetic stop before model loading')
    monkeypatch.setattr(runtime,'load_modules',stop)
    with pytest.raises(RuntimeError,match='synthetic stop'):run_one.run_job(j,host_gate=gate)
    with pytest.raises(FileExistsError):run_one.run_job(j,host_gate=gate)


def test_worker_shared_reservations_live_through_process_exit(tmp_path,monkeypatch):
    path=tmp_path/'job.json';path.write_text(json.dumps(job()))
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'report');monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(runtime,'SHARED_CACHE',tmp_path/'shared')
    acquired=[];released=[]
    class Lock:
        def __init__(self,path):self.path=path
        def __enter__(self):acquired.append(self.path);return self
        def __exit__(self,*args):released.append(self.path)
    monkeypatch.setattr(runtime,'CampaignLock',Lock)
    for name in ('install_write_guard','require_active_lease','require_idle_gpu','guard_preserved','atomic','write_new','assert_protected_T'):
        monkeypatch.setattr(runtime,name,lambda *args:None)
    monkeypatch.setattr(run_one,'run_job',lambda *args,**kwargs:dict(status='COMPLETE'))
    count=len(run_one.WORKER_RESERVATIONS)
    try:
        assert run_one.main(['--job',str(path),'--lease-token','synthetic'])==0
        assert acquired==[runtime.SHARED_CACHE/'worker.lock',runtime.CACHE/'worker.lock'] and released==[]
    finally:del run_one.WORKER_RESERVATIONS[count:]


def test_write_boundary_allows_only_campaign_outputs_and_exact_shared_locks(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'report');monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(runtime,'SHARED_CACHE',tmp_path/'old_cache')
    for path in (runtime.REPORT/'x.json',runtime.CACHE/'tmp/x.bin',runtime.SHARED_CACHE/'controller.lock',runtime.SHARED_CACHE/'worker.lock'):
        assert runtime.allowed_write_path(path)
    for path in (tmp_path/'research/models.py',runtime.SHARED_CACHE/'SOURCE.json',tmp_path/'other/result.json'):
        assert not runtime.allowed_write_path(path)


def test_null_device_exception_is_exact_and_open_only(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'read',lambda path:dict(files=[]))
    hooks=[]
    monkeypatch.setattr(runtime.sys,'addaudithook',hooks.append)
    runtime.install_write_guard();audit=hooks[0]
    allowed=[runtime.os.devnull]
    denied=[str(tmp_path/'outside.bin'),str(tmp_path/'nul'),str(tmp_path/'link_to_null')]
    if runtime.os.name=='nt':
        allowed += ['NUL',r'\\.\NUL','//./nul']
        denied += [r'.\nul',r'C:\outside\NUL',r'\\.\CON',r'\\.\COM1',r'\\.\NUL\child',r'\\?\C:\outside\NUL']
    for path in allowed:
        audit('open',(path,'r+',runtime.os.O_RDWR))
        with pytest.raises(PermissionError):audit('os.remove',(path,-1))
    for path in denied:
        assert not runtime.is_native_null_device(path)
        with pytest.raises(OSError):audit('open',(path,'w',runtime.os.O_WRONLY))
