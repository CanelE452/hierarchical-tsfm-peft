"""Synthetic CPU-only cohort, boundary, status, and inference tests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.crossdata_component_screen_20261028_v1 import postprocess as post, runtime


def jobs():
    result = []
    for seeds in ((92601, 92602), (92603,)):
        for dataset in post.DATASETS:
            for seed in seeds:
                for method in post.METHODS:
                    k=runtime.KS[dataset]
                    result.append(dict(job_id=f'XC_{method}_{dataset}_K{k}_s{seed}', method_id=method,
                        dataset=dataset, K=k, seed=seed, training_cap=runtime.DATASETS[dataset]['cap']))
    return result


def plan():
    all_jobs=jobs()
    return dict(jobs=all_jobs[:16], optional_jobs=all_jobs[16:], extension_jobs=[],
        group_order=['pems32_base','beijing_base','beijing_seed3','pems32_seed3'])


def complete_rows(n=3):
    return [dict(job, MSE=1. if job['method_id']=='OURS' else 2., MAE=1. if job['method_id']=='OURS' else 1.4,
                 selected_epoch=1., trainable_parameters=100) for job in jobs()
            if job['seed'] in [92601,92602,92603][:n]]


def test_exact_registered_job_grid_and_order():
    assert len(post.verify_plan(plan()))==24
    bad=plan(); bad['optional_jobs'][0]=bad['optional_jobs'][1]
    with pytest.raises(ValueError): post.verify_plan(bad)
    bad=plan(); bad['group_order'].reverse()
    with pytest.raises(ValueError): post.verify_plan(bad)


@pytest.mark.parametrize('extra', [0,1,2,3,4])
def test_optional_seed_enters_only_when_all_four_arms_complete(extra):
    all_jobs=jobs(); statuses={j['job_id']:'COMPLETE' for j in all_jobs if j['seed']!=92603}
    optional=[j for j in all_jobs if j['dataset']=='beijing' and j['seed']==92603]
    for j in optional[:extra]:statuses[j['job_id']]='COMPLETE'
    included,coverage=post.select_cohorts(all_jobs,dict(run_statuses=statuses))
    bj=next(x for x in coverage if x['dataset']=='beijing')
    assert bj['primary_n']==(3 if extra==4 else 2)
    assert bj['optional_partial']==(0<extra<4)
    assert sum(j['job_id'] in included for j in optional)==(4 if extra==4 else 0)


def test_partial_base_cannot_be_rescued_by_optional_or_pairwise_selection():
    all_jobs=jobs(); statuses={j['job_id']:'COMPLETE' for j in all_jobs}
    missing=next(j for j in all_jobs if j['dataset']=='beijing' and j['method_id']=='E_FIXED' and j['seed']==92602)
    statuses[missing['job_id']]='FAILED'
    included,coverage=post.select_cohorts(all_jobs,dict(run_statuses=statuses))
    assert next(c for c in coverage if c['dataset']=='beijing')['primary_n']==0
    assert all(not j['job_id'] in included for j in all_jobs if j['dataset']=='beijing')


def test_controller_primary_claim_must_match_actual_complete_cohort():
    all_jobs=jobs(); completion=dict(run_statuses={},dataset_status={'pems32':dict(primary_n=2,primary_job_ids=[])})
    with pytest.raises(ValueError,match='Controller matched'):post.select_cohorts(all_jobs,completion)


def test_all_n2_are_pilot_provisional_even_if_descriptive_direction_is_clear():
    summary,contrasts,_=post.aggregate(complete_rows(2))
    assert all(x['pilot'] and x['provisional'] for x in summary)
    assert all(x['descriptive_direction']=='FULL_LOWER' and x['comparison_status']=='PROVISIONAL' for x in contrasts)


@pytest.mark.parametrize('method', ['OURS','TIME_ONLY'])
def test_either_arm_cap_selection_makes_only_affected_comparisons_provisional(method):
    rows=complete_rows()
    for row in rows:
        if row['dataset']=='pems32' and row['method_id']==method and row['seed'] in (92601,92602):row['selected_epoch']=6.
    _,contrasts,_=post.aggregate(rows)
    selected=[x for x in contrasts if x['dataset']=='pems32' and x['method_id']=='TIME_ONLY']
    assert all(x['provisional'] and x['comparison_status']=='PROVISIONAL' for x in selected)
    assert all(not x['provisional'] for x in contrasts if x['dataset']=='beijing')


def test_strict_IQR_rule_and_matching_paired_direction():
    rows=complete_rows()
    for row in rows:
        if row['dataset']=='pems32':
            idx=row['seed']-92601
            row['MSE']=([0.,1.,2.] if row['method_id']=='OURS' else [1.,2.,3.])[idx]
    _,contrasts,_=post.aggregate(rows)
    c=next(x for x in contrasts if x['dataset']=='pems32' and x['method_id']=='TIME_ONLY' and x['metric']=='MSE')
    assert c['delta_of_medians']==c['max_arm_IQR']==1.
    assert c['descriptive_direction']=='UNCLEAR'


def test_median_difference_and_median_paired_difference_are_not_conflated():
    rows=complete_rows()
    for row in rows:
        idx=row['seed']-92601
        row['MSE']=([1.,10.,100.] if row['method_id']=='OURS' else [2.,12.,90.])[idx]
    _,contrasts,_=post.aggregate(rows)
    c=next(x for x in contrasts if x['metric']=='MSE')
    assert c['delta_of_medians']==2.
    assert c['median_paired_delta']==1.


def test_unmatched_or_duplicate_seed_rows_refused():
    rows=complete_rows(); rows.pop()
    with pytest.raises(ValueError,match='complete matched'):post.aggregate(rows)
    rows=complete_rows(); rows.append(dict(rows[0]))
    with pytest.raises(ValueError,match='complete matched'):post.aggregate(rows)


def array_fixture(tmp_path):
    registry=dict(L=4,H=2,C=2,T_start=12,columns=['a','b'],split_ranges={'V_select':[6,12]})
    values=dict(pred_normalized=np.ones((3,2,2),np.float32),target_normalized=np.zeros((3,2,2),np.float32),
        observed=np.ones((3,2,2),bool),origins=np.array([6,8,10],np.int64),channel_ids=np.array(['a','b']),
        mean=np.zeros(2,np.float32),std=np.ones(2,np.float32))
    registry['normalization_sha256']={'A60':post.scaler_fingerprint(values['mean'],values['std'])}
    path=tmp_path/'V.npz';np.savez_compressed(path,**values)
    row=dict(job_id='synthetic',dataset='synthetic',K=2,prediction_path=str(path),prediction_sha256=runtime.sha(path),
             MSE=1.,MAE=1.,n_origins=3,n_observed=12,logical_F_rows=6)
    return row,registry,values


def test_fp64_observed_channel_macro_is_independent_and_equal_channel_weight(tmp_path):
    row,registry,values=array_fixture(tmp_path)
    values['observed'][1:,:,0]=False; values['pred_normalized'][:,:,1]=3.
    np.savez_compressed(row['prediction_path'],**values)
    row.update(prediction_sha256=runtime.sha(row['prediction_path']),MSE=5.,MAE=2.,n_observed=8)
    cohort,check=post.verify_arrays(row,registry)
    assert check['MSE']==5. and check['MAE']==2. and check['metadata_gate_before_numeric_access']
    assert set(cohort)==set(post.COHORT_KEYS)


def test_T_crossing_is_rejected_before_numeric_archive_members_are_accessed(tmp_path,monkeypatch):
    row,registry,values=array_fixture(tmp_path); values['origins'][-1]=11
    np.savez_compressed(row['prediction_path'],**values); row['prediction_sha256']=runtime.sha(row['prediction_path'])
    real=post.np.load; accessed=[]
    class Guard:
        def __enter__(self):self.archive=real(row['prediction_path'],allow_pickle=False);return self
        def __exit__(self,*args):self.archive.close()
        def __getitem__(self,key):accessed.append(key);return self.archive[key]
    monkeypatch.setattr(post.np,'load',lambda *a,**k:Guard())
    with pytest.raises(ValueError,match='numeric arrays not read'):post.verify_arrays(row,registry)
    assert accessed==['origins','channel_ids']


@pytest.mark.parametrize('fault',['mask_dtype','shape','extra_member','channels','scaler','nonfinite','reported_metric','reported_count'])
def test_invalid_prediction_evidence_is_rejected(tmp_path,fault):
    row,registry,values=array_fixture(tmp_path)
    if fault=='mask_dtype':values['observed']=values['observed'].astype(np.float32)
    elif fault=='shape':values['pred_normalized']=np.ones((4,2,2),np.float32)
    elif fault=='extra_member':values['unregistered']=np.zeros(1)
    elif fault=='channels':values['channel_ids']=np.array(['b','a'])
    elif fault=='scaler':values['mean'][0]=1.
    elif fault=='nonfinite':values['pred_normalized'][0,0,0]=np.nan
    elif fault=='reported_metric':row['MSE']=.9
    else:row['logical_F_rows']=7
    np.savez_compressed(row['prediction_path'],**values);row['prediction_sha256']=runtime.sha(row['prediction_path'])
    with pytest.raises(ValueError):post.verify_arrays(row,registry)


def test_reference_cohort_changed_targets_are_rejected(tmp_path):
    row,registry,values=array_fixture(tmp_path)
    cohort,_=post.verify_arrays(row,registry);cohort['target_normalized'][0,0,0]=1.
    with pytest.raises(ValueError,match='Matched cohort'):post.verify_arrays(row,registry,cohort)


def test_figures_and_handoff_keep_unavailable_dataset_and_pilot_visible(tmp_path):
    rows=[r for r in complete_rows(2) if r['dataset']=='pems32']
    state={j['job_id']:'COMPLETE' for j in jobs() if j['dataset']=='pems32' and j['seed']!=92603}
    _,coverage=post.select_cohorts(jobs(),dict(run_statuses=state))
    _,contrasts,_=post.aggregate(rows)
    post.make_figures(rows,coverage,tmp_path);post.write_report(coverage,contrasts,[],tmp_path)
    assert len(list(tmp_path.glob('FIG_*.png')))==3
    text=(tmp_path/'EVIDENCE_HANDOFF_KO.md').read_text('utf8')
    assert 'PILOT' in text and 'INCOMPLETE' in text and '공동 LayerNorm' in text and '미관측' in text
    assert not list(tmp_path.glob('*.pptx')) and not list(tmp_path.glob('*.pdf'))


def completion_fixture(completed=24):
    all_jobs=jobs()
    statuses={j['job_id']:('COMPLETE' if i<completed else 'NOT_RUN_BUDGET') for i,j in enumerate(all_jobs)}
    return dict(status='TRAINING_COMPLETE' if completed==24 else 'TRAINING_INCOMPLETE' if completed else 'NOT_RUN_BUDGET',
        run_statuses=statuses,active=None,base_complete_count=min(16,completed),
        optional_complete_count=max(0,completed-16),completed_count=completed,extension_jobs=0)


@pytest.mark.parametrize('completed',[0,8,16,20,24])
def test_terminal_counts_match_actual_complete_jobs(completed):
    assert post.verify_completion(jobs(),completion_fixture(completed))['completed_count']==completed


@pytest.mark.parametrize('fault',['running','failed_once','missing','count','top_complete','top_incomplete','active','unknown','extension'])
def test_stale_or_inconsistent_completion_refused(fault):
    state=completion_fixture(16); job_id=jobs()[-1]['job_id']
    if fault=='running':state['run_statuses'][job_id]='RUNNING'
    elif fault=='failed_once':state['run_statuses'][job_id]='FAILED_ONCE'
    elif fault=='missing':state['run_statuses'].pop(job_id)
    elif fault=='count':state['completed_count']=17
    elif fault=='top_complete':state['status']='TRAINING_COMPLETE'
    elif fault=='top_incomplete':state=completion_fixture();state['status']='TRAINING_INCOMPLETE'
    elif fault=='active':state['active']={'pid':42}
    elif fault=='unknown':state['run_statuses']['invented']='COMPLETE'
    else:state['extension_jobs']=1
    with pytest.raises(ValueError):post.verify_completion(jobs(),state)


def test_explicit_stopped_diagnostic_state_is_not_normal_training_completion():
    state=dict(status='STOPPED',active=None,stop_reason='pretraining import failure',
               run_statuses={jobs()[0]['job_id']:'RUNNING'})
    assert post.verify_completion(jobs(),state)['completed_count']==0
    state['status']='TRAINING_COMPLETE'
    with pytest.raises(ValueError):post.verify_completion(jobs(),state)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def technical_failure_fixture(tmp_path,monkeypatch):
    repo=tmp_path;report=repo/'reports'/runtime.NAME;cache=repo/'.cache'/runtime.NAME
    monkeypatch.setattr(runtime,'REPO',repo);monkeypatch.setattr(runtime,'REPORT',report);monkeypatch.setattr(runtime,'CACHE',cache)
    job=jobs()[0];root=report/'jobs'/job['job_id'];failed=root/'attempt_001'
    old_document=report/'RUN_PLAN.json';write_json(old_document,plan())
    old_seal=dict(created_utc='2026-10-28T04:40:00+00:00',files={'RUN_PLAN.json':runtime.sha(old_document)})
    seal=report/'REGISTRATION_SEAL.json';write_json(seal,old_seal)
    old_registration=dict(seal_sha256=runtime.sha(seal),registered_at_utc=old_seal['created_utc'],files=old_seal['files'])
    registration=dict(seal_sha256='f'*64,registered_at_utc='2026-10-28T04:45:00+00:00',files={})
    error='PermissionError: Write outside this campaign and exact shared lock files: \\nul'
    receipt=dict(job=job,attempt=1,status='FAILED',registration=old_registration,started_utc='2026-10-28T04:41:00+00:00',
        training_attempted=False,training_executed=False,T_numeric_values_parsed=0,preserved_before={'changed':[]},error=error)
    worker=dict(job_id=job['job_id'],attempt=1,status='STOPPED',exit_code=20,error=error)
    paths=[old_document,seal,root/'WORKER_RECEIPT_attempt1.json',failed/'RECEIPT.json',failed/'OPENED_FILES.json',failed/'HOST_GPU_GATE.json']
    for p,v in zip(paths[2:],[worker,receipt,dict(T_numeric_values_parsed=0),dict(status='PASS')]):write_json(p,v)
    raw=cache/'private.log';raw.parent.mkdir(parents=True);raw.write_bytes(b'private diagnostic bytes');paths.append(raw)
    rows=[]
    for path in paths:
        original=path.relative_to(repo);copy=report/'revisions'/'pretraining_v1'/original
        copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
        rows.append(dict(original=original.as_posix(),archive=copy.relative_to(repo).as_posix(),sha256=runtime.sha(copy)))
    amendment=dict(created_utc='2026-10-28T04:43:00+00:00',failed_job=job['job_id'],
        training_attempted=False,training_executed=False,T_numeric_values_parsed=0,
        original_seal_sha256=old_registration['seal_sha256'],files=rows)
    write_json(report/'TECHNICAL_AMENDMENT_001.json',amendment)
    return job,registration,amendment,receipt


def test_archived_pretraining_failure_is_allowed_without_public_rawlogs(tmp_path,monkeypatch):
    job,registration,_,_=technical_failure_fixture(tmp_path,monkeypatch)
    evidence=post.verify_technical_import_failure(job,1,registration)
    assert evidence and all('.cache' not in path.parts and path.suffix!='.log' for path in evidence)


@pytest.mark.parametrize('fault',['archive_hash','receipt_hash','training','start','model_artifact','old_seal','amendment_job','chronology','missing_seal_document','different_error'])
def test_old_seal_exception_requires_exact_archived_pretraining_evidence(tmp_path,monkeypatch,fault):
    job,registration,amendment,receipt=technical_failure_fixture(tmp_path,monkeypatch)
    failed=runtime.REPORT/'jobs'/job['job_id']/'attempt_001'
    if fault=='archive_hash':(runtime.REPO/amendment['files'][0]['archive']).write_bytes(b'changed')
    elif fault=='receipt_hash':(failed/'RECEIPT.json').write_bytes(b'changed')
    elif fault=='training':amendment['training_executed']=True
    elif fault=='start':write_json(failed/'START.json',{})
    elif fault=='model_artifact':path=runtime.CACHE/'jobs'/job['job_id']/'attempt_001'/'model.pt';path.parent.mkdir(parents=True);path.write_bytes(b'checkpoint')
    elif fault=='old_seal':registration['seal_sha256']=amendment['original_seal_sha256']
    elif fault=='amendment_job':amendment['failed_job']='other'
    elif fault=='chronology':registration['registered_at_utc']='2026-10-28T04:42:00+00:00'
    elif fault=='missing_seal_document':amendment['files']=amendment['files'][1:]
    else:
        receipt['error']='ValueError: scientific gate'
        path=failed/'RECEIPT.json';write_json(path,receipt)
        entry=next(x for x in amendment['files'] if x['original']==path.relative_to(runtime.REPO).as_posix())
        (runtime.REPO/entry['archive']).write_bytes(path.read_bytes());entry['sha256']=runtime.sha(path)
    write_json(runtime.REPORT/'TECHNICAL_AMENDMENT_001.json',amendment)
    with pytest.raises(ValueError):post.verify_technical_import_failure(job,1,registration)


def test_successful_retry_binds_failed_import_and_new_attempt(tmp_path,monkeypatch):
    job,registration,_,_=technical_failure_fixture(tmp_path,monkeypatch)
    root=runtime.REPORT/'jobs'/job['job_id']
    write_json(root/'WORKER_RECEIPT_attempt2.json',dict(job_id=job['job_id'],attempt=2,status='COMPLETE',exit_code=0))
    completion=dict(run_statuses={job['job_id']:'COMPLETE'},attempts={job['job_id']:[dict(attempt=1,exit_code=20),dict(attempt=2,exit_code=0)]})
    _,outcomes=post.verify_attempts(job,registration,completion,2)
    assert outcomes[0]['failure_class']=='ARCHIVED_PRETRAINING_IMPORT_FAILURE'
    assert outcomes[0]['model_performance_result'] is False
    assert outcomes[1]['status']=='COMPLETE'


@pytest.mark.parametrize('fault',['exit_mismatch','duplicate','extra','missing','success_repeated','success_omitted'])
def test_attempt_history_cannot_hide_or_repeat_success(tmp_path,monkeypatch,fault):
    job,registration,_,_=technical_failure_fixture(tmp_path,monkeypatch)
    root=runtime.REPORT/'jobs'/job['job_id']
    worker=dict(job_id=job['job_id'],attempt=2,status='COMPLETE',exit_code=0)
    write_json(root/'WORKER_RECEIPT_attempt2.json',worker)
    rows=[dict(attempt=1,exit_code=20),dict(attempt=2,exit_code=0)]
    if fault=='exit_mismatch':rows[1]['exit_code']=1
    elif fault=='duplicate':rows.append(dict(rows[0]))
    elif fault=='extra':rows.append(dict(attempt=3,exit_code=0))
    elif fault=='missing':rows.pop(0)
    elif fault=='success_repeated':write_json(root/'WORKER_RECEIPT_attempt1.json',dict(worker,attempt=1));rows[0]['exit_code']=0
    completion=dict(run_statuses={job['job_id']:'COMPLETE'},attempts={job['job_id']:rows})
    with pytest.raises(ValueError):post.verify_attempts(job,registration,completion,None if fault=='success_omitted' else 2)
