"""Synthetic CPU-only recovery and lifecycle evidence; never launches a real worker."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.p_branch_appl_ett_20261029_v1 import controller as c, supervisor as s, runtime, run_one


def jobs():
    return [dict(job_id=f'PBA_{m}_{d}_K{runtime.KS[d]}_s{seed}',method_id=m,dataset=d,K=runtime.KS[d],
                 seed=seed,extended=False)
            for seed in range(92601,92611) for d in runtime.DATASETS for m in run_one.METHOD_FAMILY]


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


@pytest.fixture
def manager(monkeypatch,tmp_path):
    report,cache=tmp_path/'reports',tmp_path/'cache'
    report.mkdir();cache.mkdir()
    monkeypatch.setattr(runtime,'REPORT',report)
    monkeypatch.setattr(runtime,'CACHE',cache)
    monkeypatch.setattr(c,'SHARED_CACHE',tmp_path/'shared')
    monkeypatch.setattr(run_one,'validate_registration',lambda job:dict(status='TEST_REGISTRATION'))
    monkeypatch.setattr(runtime,'guard_preserved',lambda **kwargs:dict(changed=[]))
    put(report/'RUN_PLAN.json',dict(jobs=jobs(),extension_jobs=[]))
    put(report/'PREREGISTRATION.json',dict(execution_policy=dict(admission_deadline=None,retries=1,
        resource_wait_seconds=60,heartbeat_seconds=1,max_controller_restarts=3)))
    return c.Controller()


def test_fixed_sixty_order_and_null_cutoff(manager):
    assert len(manager.jobs)==60 and manager.extensions==[] and manager.deadline is None
    assert [(j['dataset'],j['method_id']) for j in manager.jobs[:6]]==[
        (d,m) for d in runtime.DATASETS for m in run_one.METHOD_FAMILY]
    assert c.admission_allowed(None,9999999,60,9999999999)
    assert not c.admission_allowed(100,50,60,0)


def test_changed_order_rejected(manager):
    manager.jobs[0],manager.jobs[1]=manager.jobs[1],manager.jobs[0]
    with pytest.raises(c.ScientificSuspend):manager.validate_plan()


def test_extension_rejected(manager):
    manager.extensions=[dict(manager.jobs[0],extended=True)]
    with pytest.raises(c.ScientificSuspend):manager.validate_plan()


@pytest.mark.parametrize('record,code,timeout,expected',[
    ({'failure_kind':'TECHNICAL'},20,False,'TECHNICAL'),
    ({'failure_kind':'SCIENTIFIC'},1,False,'SCIENTIFIC'),
    ({'failure_kind':'UNKNOWN'},20,False,'UNKNOWN'),
    ({'status':'STOPPED'},20,False,'UNKNOWN'),
    ({'status':'USER_CANCELLED'},130,False,'USER_CANCELLED'),
    (None,20,False,'UNKNOWN'),(None,-9,False,'TECHNICAL'),(None,1,True,'TECHNICAL')])
def test_failure_is_evidence_not_just_exitcode(record,code,timeout,expected):
    assert c.receipt_failure(record,code,timeout)[0]==expected


def test_pid_reuse_is_not_the_original_process():
    record=dict(pid=10,created=12,command=[])
    assert c.same_identity(record,10,12)
    assert not c.same_identity(record,10,11)
    assert not c.same_identity(record,11,12)


def test_exact_worker_command(tmp_path):
    path=tmp_path/'job.json'
    record=dict(command=['python','-m',f'research.{runtime.NAME}.run_one','--job',str(path),'--attempt','2'])
    assert c.matches_worker(record,path,2)
    assert not c.matches_worker(record,path,1)
    assert not c.matches_worker(record,tmp_path/'other.json',2)
    record['command'][2]='arbitrary.compute'
    assert not c.matches_worker(record,path,2)


def test_desktop_contexts_do_not_block_training(monkeypatch):
    monkeypatch.setattr(c.psutil,'process_iter',lambda *a,**k:[])
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,
        stdout='100,C:/Chrome/chrome.exe\n101,C:/RustDesk/rustdesk.exe\n102,C:/Windows/dwm.exe\n'))
    assert c.foreign_workloads()==[]


def test_foreign_gpu_compute_blocks(monkeypatch):
    monkeypatch.setattr(c.psutil,'process_iter',lambda *a,**k:[])
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='100,C:/env/python.exe\n'))
    assert c.foreign_workloads()[0]['kind']=='FOREIGN_GPU_COMPUTE'
    assert c.foreign_workloads({100})==[]


def test_own_venv_controller_wrapper_is_not_a_foreign_experiment(monkeypatch):
    wrapper=dict(pid=99,created=1.,command=['python','-m',f'research.{runtime.NAME}.controller'],
                 cwd=str(runtime.REPO),name='python.exe')
    monkeypatch.setattr(c.psutil,'Process',lambda *a:SimpleNamespace(parents=lambda:[SimpleNamespace(pid=99)]))
    monkeypatch.setattr(c.psutil,'process_iter',lambda *a,**k:[SimpleNamespace(pid=99)])
    monkeypatch.setattr(c,'process_record',lambda pid:wrapper)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=''))
    assert c.foreign_workloads()==[]


def test_resource_wait_does_not_consume_attempt(manager,monkeypatch):
    calls=[];states=iter([[{'kind':'FOREIGN_GPU_COMPUTE'}],[]])
    monkeypatch.setattr(c,'foreign_workloads',lambda *a,**k:next(states))
    monkeypatch.setattr(c.time,'sleep',lambda seconds:calls.append(seconds))
    assert manager.resource_wait(manager.jobs[0])
    assert calls==[60] and manager.state['attempts']=={}
    assert not list(runtime.CACHE.rglob('*attempt*'))
    assert len(manager.state['wait_events'])==1


def test_cancelled_wait_cannot_restart(manager):
    put(runtime.CACHE/'USER_STOP_REQUEST.json',dict(request='stop'))
    with pytest.raises(c.UserCancelled):manager.resource_wait(manager.jobs[0])


def test_stale_lock_file_is_acquirable_but_live_os_lock_waits(manager,monkeypatch):
    path=runtime.CACHE/'worker.lock';path.write_text('stale',encoding='utf8')
    monkeypatch.setattr(c,'foreign_workloads',lambda:[])
    assert manager.resource_wait(manager.jobs[0])
    lock=runtime.CampaignLock(path);lock.__enter__()
    releases=[]
    def release(seconds):
        releases.append(seconds);lock.__exit__(None,None,None)
    monkeypatch.setattr(c.time,'sleep',release)
    assert manager.resource_wait(manager.jobs[0])
    assert releases==[60] and manager.state['attempts']=={}


def test_attempts_are_durable_and_root_wave_is_explicit(manager,monkeypatch):
    job=manager.jobs[0];root=runtime.REPORT/'jobs'/job['job_id']
    for attempt in (1,2):
        put(root/f'WORKER_RECEIPT_attempt{attempt}.json',dict(status='FAILED',failure_kind='TECHNICAL',exit_code=1))
    assert manager.next_attempt(job)==3
    monkeypatch.setattr(run_one,'allowed_attempts',lambda j:[1,2,3,4])
    assert manager.next_attempt(job)==3
    put(root/'WORKER_RECEIPT_attempt3.json',dict(status='FAILED',failure_kind='TECHNICAL',exit_code=1))
    assert manager.next_attempt(job)==4


def test_scientific_and_unknown_failures_cannot_be_automatically_retried(manager):
    job=manager.jobs[0];path=runtime.REPORT/'jobs'/job['job_id']/'WORKER_RECEIPT_attempt1.json'
    put(path,dict(status='SCIENTIFIC_SUSPEND',failure_kind='SCIENTIFIC',exit_code=20))
    with pytest.raises(c.ScientificSuspend):manager.next_attempt(job)
    put(path,dict(status='SCIENTIFIC_SUSPEND',failure_kind='UNKNOWN',exit_code=20))
    with pytest.raises(c.UnknownSuspend):manager.next_attempt(job)


def test_complete_job_is_reused_without_spawning(manager,monkeypatch):
    job=manager.jobs[0];put(runtime.REPORT/'jobs'/job['job_id']/'RESULT.json',{})
    monkeypatch.setattr(c,'verify_result',lambda j:{'status':'COMPLETE'})
    monkeypatch.setattr(manager,'spawn_attempt',lambda *a:pytest.fail('completed training repeated'))
    assert manager.run(job)
    assert manager.state['run_statuses'][job['job_id']]=='COMPLETE'


def test_complete_artifact_without_final_worker_receipt_cannot_be_reused(manager):
    job=manager.jobs[0]
    result=dict(status='COMPLETE',job=job,registration=run_one.validate_registration(job),
                spec=run_one.build_spec(job),selection_sealed_before_V=True,attempt=1)
    put(runtime.REPORT/'jobs'/job['job_id']/'RESULT.json',result)
    with pytest.raises(c.UnknownSuspend,match='final worker receipt'):
        c.verify_result(job)


def test_two_technical_attempts_do_not_abort_following_jobs(manager,monkeypatch):
    attempts=[]
    monkeypatch.setattr(manager,'next_attempt',lambda j:1)
    monkeypatch.setattr(manager,'spawn_attempt',lambda j,a:(SimpleNamespace(poll=lambda:1),dict(attempt=a)))
    monkeypatch.setattr(manager,'wait_active',lambda *a:False)
    monkeypatch.setattr(manager,'finish_attempt',lambda j,a,*rest:attempts.append((j['job_id'],a['attempt'])) or False)
    assert not manager.run(manager.jobs[0]);assert not manager.run(manager.jobs[1])
    assert [a for _,a in attempts]==[1,2,1,2]
    assert manager.state['run_statuses'][manager.jobs[1]['job_id']]=='FAILED_TERMINAL'


def test_all_registered_jobs_are_visited_despite_terminal_job_failures(manager,monkeypatch):
    visited=[];smoke=[]
    monkeypatch.setattr(manager,'recover_worker',lambda:None)
    def fail(job):
        visited.append(job['job_id']);manager.state['run_statuses'][job['job_id']]='FAILED_TERMINAL';return False
    monkeypatch.setattr(manager,'run',fail)
    monkeypatch.setattr(manager,'first_seed_smoke',lambda:smoke.append(True))
    monkeypatch.setattr(manager,'postprocess',lambda:None)
    assert manager.execute()==1
    assert visited==[j['job_id'] for j in manager.jobs] and smoke==[True]
    assert manager.state['status']=='TRAINING_INCOMPLETE' and manager.state['completed_count']==0


def test_recovered_worker_keeps_original_watchdog_start(manager,monkeypatch):
    job=manager.jobs[0];descriptor=manager.descriptor(job)
    active=dict(job_id=job['job_id'],dataset=job['dataset'],attempt=1,started_unix=10.,descriptor=str(descriptor))
    manager.state['active']=active
    put(runtime.CACHE/'LIVE_WORKER.json',dict(status='RUNNING',pid=20,process_creation_time=5.,
        job_id=job['job_id'],job=job,attempt=1,started_unix=20.))
    record=dict(pid=20,created=5.,command=['python','-m',f'research.{runtime.NAME}.run_one','--job',str(descriptor),'--attempt','1'])
    monkeypatch.setattr(c,'process_record',lambda pid:record)
    seen=[]
    monkeypatch.setattr(manager,'wait_active',lambda a:seen.append(dict(a)) or False)
    monkeypatch.setattr(manager,'finish_attempt',lambda *a,**k:True)
    manager.recover_worker()
    assert seen[0]['started_unix']==10. and manager.state['active'] is None


def test_reused_pid_live_marker_is_ignored(manager,monkeypatch):
    job=manager.jobs[0];descriptor=manager.descriptor(job)
    put(runtime.CACHE/'LIVE_WORKER.json',dict(status='RUNNING',pid=20,process_creation_time=5.,
        job_id=job['job_id'],job=job,attempt=1,started_unix=20.))
    record=dict(pid=20,created=6.,command=['python','-m',f'research.{runtime.NAME}.run_one','--job',str(descriptor),'--attempt','1'])
    monkeypatch.setattr(c,'process_record',lambda pid:record)
    monkeypatch.setattr(manager,'wait_active',lambda *a:pytest.fail('waited for reused PID'))
    manager.recover_worker()
    assert manager.state['active'] is None


def test_registered_watchdogs_are_fixed(manager):
    assert manager.hard_walls==c.HARD_WALLS
    manager.state['estimates']['appliances']=9999
    assert manager.hard_walls['appliances']==2700


def test_running_attempt_ignores_campaign_admission_deadline(manager,monkeypatch):
    manager.deadline=0
    active=dict(job_id=manager.jobs[0]['job_id'],dataset='appliances',attempt=1,
                started_unix=c.time.time(),descriptor='fixture.json')
    workers=iter([[dict(pid=90,created=1.)],[]])
    monkeypatch.setattr(manager,'active_workers',lambda value:next(workers))
    monkeypatch.setattr(c.time,'sleep',lambda seconds:None)
    assert manager.wait_active(active) is False


def test_fixed_wall_timeout_terminates_only_the_recorded_job(manager,monkeypatch):
    active=dict(job_id=manager.jobs[0]['job_id'],dataset='appliances',attempt=1,
                started_unix=c.time.time()-2701,descriptor='fixture.json')
    monkeypatch.setattr(manager,'active_workers',lambda value:[dict(pid=90,created=1.)])
    seen=[]
    monkeypatch.setattr(c,'terminate_owned',lambda *args:seen.append(args) or [{'pid':90}])
    assert manager.wait_active(active) is True
    assert seen==[('fixture.json',1,active['started_unix'])]


def first_seed_data(manager,monkeypatch):
    result_map={};references=[]
    for dataset in runtime.DATASETS:
        shape=runtime.V_SHAPES[dataset]
        origins=np.arange(runtime.V_RANGES[dataset][0],runtime.V_RANGES[dataset][1]-48+1,48,dtype=np.int64)
        values=dict(pred_normalized=np.zeros(shape,np.float32),target_normalized=np.zeros(shape,np.float32),
            observed=np.ones(shape,bool),origins=origins,channel_ids=np.array([f'c{i}' for i in range(shape[2])]),
            mean=np.zeros(shape[2],np.float32),std=np.ones(shape[2],np.float32))
        path=runtime.CACHE/f'{dataset}_reference.npz';np.savez_compressed(path,**values)
        references.append(dict(dataset=dataset,method_id='OURS',seed=92603,prediction_path=str(path),prediction_sha256=runtime.sha(path)))
        for job in [j for j in manager.jobs[:6] if j['dataset']==dataset]:
            p=runtime.CACHE/(job['job_id']+'.npz');new=dict(values,pred_normalized=np.full(shape,.25,np.float32))
            np.savez_compressed(p,**new)
            row=dict(prediction_path=str(p),prediction_sha256=runtime.sha(p),MSE=.0625,MAE=.25,
                     trainable_parameters=runtime.EXPECTED_PARAMETERS[dataset][job['method_id']])
            result_map[job['job_id']]=dict(row=row)
            put(runtime.REPORT/'jobs'/job['job_id']/'RESULT.json',{'fixture':True})
    put(runtime.REPORT/'REFERENCE_RUNS.json',dict(rows=references))
    monkeypatch.setattr(c,'verify_result',lambda j:result_map[j['job_id']])
    return result_map


def test_first_seed_smoke_is_dataset_specific_fp64_and_params(manager,monkeypatch):
    first_seed_data(manager,monkeypatch);manager.first_seed_smoke()
    check=runtime.read(runtime.REPORT/'FIRST_SEED_SMOKE_VALIDATION.json')
    assert check['status']=='PASS' and check['completed']==6
    assert all(row['MSE']==.0625 and row['T_numeric_values_parsed']==0 for row in check['jobs'])


def test_first_seed_smoke_rejects_wrong_scaler(manager,monkeypatch):
    results=first_seed_data(manager,monkeypatch)
    row=results[manager.jobs[0]['job_id']]['row'];p=Path(row['prediction_path'])
    with np.load(p,allow_pickle=False) as archive:values={k:archive[k] for k in archive.files}
    values['std']=np.full(values['std'].shape,2.,np.float32);np.savez_compressed(p,**values);row['prediction_sha256']=runtime.sha(p)
    with pytest.raises(c.ScientificSuspend):manager.first_seed_smoke()


def test_first_seed_smoke_rejects_nonfinite_prediction(manager,monkeypatch):
    results=first_seed_data(manager,monkeypatch)
    row=results[manager.jobs[0]['job_id']]['row'];p=Path(row['prediction_path'])
    with np.load(p,allow_pickle=False) as archive:values={k:archive[k] for k in archive.files}
    values['pred_normalized'].flat[0]=np.nan;np.savez_compressed(p,**values);row['prediction_sha256']=runtime.sha(p)
    with pytest.raises(c.ScientificSuspend,match='nonfinite'):
        manager.first_seed_smoke()


@pytest.mark.parametrize('status,code,expected',[
    ('TRAINING_COMPLETE',0,'TERMINAL'),('TRAINING_INCOMPLETE',1,'TERMINAL'),
    ('SCIENTIFIC_SUSPEND',20,'TERMINAL'),('UNKNOWN_SUSPEND',20,'TERMINAL'),
    ('USER_CANCELLED',130,'TERMINAL'),('RUNNING',20,'UNKNOWN_SUSPEND'),
    ('CONTROLLER_CRASHED',70,'RESTART_TECHNICAL'),('RUNNING',None,'RESTART_TECHNICAL')])
def test_supervisor_restart_decision_never_bypasses_scientific_or_user_stop(status,code,expected):
    assert s.restart_decision(dict(status=status),code)==expected


def test_supervisor_durable_restart_limit_does_not_reset(manager):
    put(runtime.REPORT/'REGISTRATION_SEAL.json',{'test':True})
    put(runtime.REPORT/'SUPERVISOR_STATUS.json',dict(status='RESTART_LIMIT_REACHED',restart_count=3,launch_count=4,launches=[]))
    supervisor=s.Supervisor()
    assert supervisor.state['restart_count']==3 and supervisor.run()==70


def test_supervisor_adopts_actual_child_when_venv_wrapper_disappears(manager,monkeypatch):
    put(runtime.REPORT/'REGISTRATION_SEAL.json',{'test':True})
    supervisor=s.Supervisor()
    wrapper=dict(pid=10,created=1.,command=['python','-m',f'research.{runtime.NAME}.controller'],cwd=str(runtime.REPO))
    child=dict(pid=11,created=2.,command=wrapper['command'],cwd=str(runtime.REPO))
    snapshots=iter([[child],[]])
    monkeypatch.setattr(s,'controller_candidates',lambda *args:next(snapshots))
    monkeypatch.setattr(s,'matching_controller',lambda *args:None)
    monkeypatch.setattr(s.time,'sleep',lambda seconds:None)
    assert supervisor.wait_existing(wrapper) is None
    assert supervisor.state['active_controller']=={'pid':11,'created':2.}


def test_supervisor_durable_launch_is_recorded_even_when_controller_dies_immediately(manager,monkeypatch):
    put(runtime.REPORT/'REGISTRATION_SEAL.json',{'test':True})
    supervisor=s.Supervisor();supervisor.maximum=0
    monkeypatch.setattr(supervisor,'existing_controller',lambda:None)
    monkeypatch.setattr(s,'matching_controller',lambda *args:None)
    monkeypatch.setattr(s,'controller_candidates',lambda *args:[])
    monkeypatch.setattr(c,'process_record',lambda pid:None)
    monkeypatch.setattr(s.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(pid=44,wait=lambda **kw:70))
    assert supervisor.run()==70
    assert supervisor.state['launch_count']==1 and len(supervisor.state['launches'])==1
    assert supervisor.state['status']=='RESTART_LIMIT_REACHED'


def test_controller_constructor_does_not_bypass_previous_scientific_suspend(manager):
    put(runtime.REPORT/'PROGRESS.json',dict(status='SCIENTIFIC_SUSPEND',run_statuses={},attempts={},active=None))
    with pytest.raises(c.ScientificSuspend):c.Controller(resume_stopped=True)


@pytest.mark.parametrize('error,status',[(c.ScientificSuspend,'SCIENTIFIC_SUSPEND'),
    (c.UnknownSuspend,'UNKNOWN_SUSPEND'),(c.UserCancelled,'USER_CANCELLED')])
@pytest.mark.parametrize('count',[0,1])
def test_actual_main_terminal_path_produces_analysis_accepted_zero_or_partial(manager,monkeypatch,error,status,count):
    from research.p_branch_appl_ett_20261029_v1 import postprocess
    from research.p_branch_appl_ett_20261029_v1.tests.test_analysis import complete_evidence
    monkeypatch.setattr(runtime,'REPO',runtime.REPORT.parent)
    if count:
        _,registration=complete_evidence(manager.jobs[0],monkeypatch)
        monkeypatch.setattr(run_one,'validate_registration',lambda job:registration)
    active_job=manager.jobs[1]
    manager.state['active']=dict(job_id=active_job['job_id'],dataset=active_job['dataset'],attempt=1,
        descriptor=str(manager.descriptor(active_job)),started_unix=1.)
    stopped=[]
    monkeypatch.setattr(c,'terminate_owned',lambda *args:stopped.append(args) or [])
    monkeypatch.setattr(manager,'active_workers',lambda active:[])
    monkeypatch.setattr(runtime,'install_write_guard',lambda:None)
    monkeypatch.setattr(c,'prevent_sleep',lambda active:None)
    monkeypatch.setattr(c,'Controller',lambda resume=False:manager)
    def fail():raise error('Synthetic terminal evidence')
    monkeypatch.setattr(manager,'execute',fail)
    observed=[]
    def validate():
        completion=runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
        observed.append(postprocess.verify_completion_evidence(manager.plan,completion)[1])
    monkeypatch.setattr(manager,'postprocess',validate)
    assert c.main([])==(2 if status=='USER_CANCELLED' else 20)
    completion=runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
    assert completion['active'] is None and len(completion['run_statuses'])==60
    assert completion['completed_count']==count and observed==[count] and len(stopped)==1
    assert completion['run_statuses'][active_job['job_id']]=='NOT_RUN_'+status


def test_terminal_invalid_result_is_preserved_as_hash_bound_rejected_evidence(manager,monkeypatch,tmp_path):
    monkeypatch.setattr(runtime,'REPO',tmp_path)
    job=manager.jobs[0];path=runtime.REPORT/'jobs'/job['job_id']/'RESULT.json'
    put(path,dict(attempt=1,synthetic_invalid=True));digest=runtime.sha(path)
    def reject(job):raise c.UnknownSuspend('Synthetic missing final receipt')
    monkeypatch.setattr(c,'verify_result',reject)
    manager.terminalize('UNKNOWN_SUSPEND','Synthetic failed final validation')
    rejection=runtime.read(path.parent/'RESULT_REJECTION.json')
    assert not path.exists() and runtime.sha(tmp_path/rejection['rejected_result_path'])==digest
    assert rejection['job']==job and rejection['worker_receipt_missing'] is True
    assert manager.state['run_statuses'][job['job_id']]=='UNKNOWN_SUSPEND'
    assert rejection['trigger']=='RESULT_WRITTEN_THEN_TERMINAL_VALIDATION_REJECTED'
