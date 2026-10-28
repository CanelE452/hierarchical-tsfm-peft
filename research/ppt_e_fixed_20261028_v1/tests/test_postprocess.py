"""Synthetic CPU-only analysis tests; no campaign results or protected T read."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.ppt_e_fixed_20261028_v1 import postprocess as pp


def row(method, seed, value=1., *, extended=False, selected=1.):
    return dict(job_id=f'{method}_{seed}'+('_ext10' if extended else ''),
                method_id=method,dataset='jena',seed=seed,K=6,MSE=value,MAE=value**.5,
                selected_epoch=selected,training_cap=10. if extended else 3.,extended=extended,
                trainable_parameters=35591 if method=='E_FIXED' else 35717,
                normalization_fit_end=12579,reused=method=='OURS',provisional=False)


def groups(full=(1,2,3,4,5),fixed=(5,6,7,8,9)):
    return [row(method,seed,value) for method,values in [('OURS',full),('E_FIXED',fixed)]
            for seed,value in zip(pp.SEEDS,values)]


def save_prediction(tmp_path, **changes):
    target=np.zeros((2,2,2),np.float32)
    pred=np.ones_like(target);pred[:,:,1]=4
    observed=np.ones_like(target,dtype=bool);observed[0,:,1]=False;observed[1,0,1]=False
    arrays=dict(pred_normalized=pred,target_normalized=target,observed=observed,
                origins=np.array([10,12]),channel_ids=np.array([0,1]),
                mean=np.zeros(2,np.float32),std=np.ones(2,np.float32))
    arrays.update(changes)
    path=tmp_path/'synthetic.npz';np.savez(path,**{k:v for k,v in arrays.items() if v is not None})
    r=dict(job_id='synthetic',dataset='jena',prediction_path=str(path),prediction_sha256=pp.runtime.sha(path),MSE=8.5,MAE=2.5)
    return arrays,r,dict(shape=[2,2,2],T_start=14)


def test_fp64_channel_macro_is_not_pooled_cell_error(tmp_path):
    arrays,r,contract=save_prediction(tmp_path)
    returned,check=pp.verify_arrays(r,None,contract)
    assert check['MSE']==8.5 and check['MAE']==2.5
    assert np.mean((arrays['pred_normalized'][arrays['observed']])**2)==4
    assert np.array_equal(returned['mean'],np.zeros(2))


@pytest.mark.parametrize('mutation', ['bytes','metric','nan_metric','shape','T','scaler','cohort','mask','origins','channel_ids','empty_channel'])
def test_prediction_validation_rejects_invalid_evidence(tmp_path,mutation):
    arrays,r,contract=save_prediction(tmp_path);cohort=copy.deepcopy(arrays)
    if mutation=='bytes': r['prediction_sha256']='0'*64
    elif mutation=='metric': r['MSE']=4.
    elif mutation=='nan_metric': r['MSE']=float('nan')
    elif mutation=='shape': contract['shape']=[2,1,2]
    elif mutation=='T': contract['T_start']=13
    elif mutation=='scaler': arrays['std'][0]=0
    elif mutation=='cohort': cohort['target_normalized'][0,0,0]=1
    elif mutation=='mask': arrays['observed']=arrays['observed'].astype(int);arrays['observed'][0,0,0]=2
    elif mutation=='origins': arrays['origins'][1]=10
    elif mutation=='channel_ids': arrays['channel_ids'][1]=0
    elif mutation=='empty_channel': arrays['observed'][:,:,1]=False
    if mutation in ('scaler','mask','origins','channel_ids','empty_channel'):
        np.savez(r['prediction_path'],**arrays);r['prediction_sha256']=pp.runtime.sha(r['prediction_path'])
    with pytest.raises(ValueError): pp.verify_arrays(r,cohort,contract)


def test_only_registered_legacy_archives_can_omit_both_scalers(tmp_path):
    arrays,r,contract=save_prediction(tmp_path,mean=None,std=None)
    with pytest.raises(ValueError,match='missing normalization'): pp.verify_arrays(r,None,contract)
    r['job_id']='HISTORICAL_jena_K6_s92601'
    assert pp.verify_arrays(r,None,contract)[1]['legacy_exception']==r['job_id']
    arrays,r,contract=save_prediction(tmp_path,std=None)
    r['job_id']='HISTORICAL_jena_K6_s92602'
    with pytest.raises(ValueError,match='complete mean/std'): pp.verify_arrays(r,None,contract)


def test_T_boundary_rejected_before_any_numeric_prediction_or_target_read(tmp_path,monkeypatch):
    _,r,contract=save_prediction(tmp_path)
    class GuardedArchive:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __getitem__(self,key):
            if key=='origins': return np.array([10,13])
            if key=='channel_ids': return np.array([0,1])
            raise AssertionError('Protected numeric arrays were accessed before the boundary gate')
    monkeypatch.setattr(pp.np,'load',lambda *args,**kwargs:GuardedArchive())
    with pytest.raises(ValueError,match='numeric arrays were not read'): pp.verify_arrays(r,None,contract)


def test_aggregate_exact_descriptive_rule_and_paired_direction():
    summaries,contrasts,paired=pp.aggregate(groups())
    c=contrasts[0]
    assert c['delta_median_MSE']==4 and c['paired_delta_median']==4
    assert c['descriptive_decision']=='REFERENCE_LOWER' and c['seeds_reference_lower']==5
    assert c['effect_pct']==pytest.approx(100*4/3)
    assert [p['seed'] for p in paired]==list(pp.SEEDS)
    assert {s['parameters'] for s in summaries}=={35717,35591}


def test_paired_median_is_not_difference_of_group_medians():
    _,contrasts,_=pp.aggregate(groups((1,2,3,4,100),(10,2,3,4,5)))
    assert contrasts[0]['delta_median_MSE']==1
    assert contrasts[0]['paired_delta_median']==0


@pytest.mark.parametrize('fixed,decision', [((3,4,5,6,7),'UNCLEAR'),((1,2,3,4,5),'UNCLEAR'),((.1,.2,.3,.4,.5),'VARIANT_LOWER')])
def test_iqr_rule_strict_boundary_and_direction(fixed,decision):
    assert pp.aggregate(groups(fixed=fixed))[1][0]['descriptive_decision']==decision


def test_provisional_suffix_is_not_significance():
    rows=groups();rows[-1]['provisional']=True
    assert pp.aggregate(rows)[1][0]['descriptive_decision']=='REFERENCE_LOWER_PROVISIONAL'


@pytest.mark.parametrize('mutation',['missing','duplicate','mixed_cap','parameters','nonfinite','missing_reference'])
def test_aggregate_rejects_incomplete_or_mixed_family(mutation):
    rows=groups()
    if mutation=='missing': rows.pop()
    elif mutation=='duplicate': rows[-1]['seed']=92604
    elif mutation=='mixed_cap': rows[-1]['training_cap']=10.
    elif mutation=='parameters': rows[-1]['trainable_parameters']=35717
    elif mutation=='nonfinite': rows[-1]['MSE']=float('nan')
    elif mutation=='missing_reference': rows=rows[5:]
    with pytest.raises(ValueError): pp.aggregate(rows)


def study(base_epochs=(1,1,1,1,1), ext_epochs=(), status='TRAINING_COMPLETE', extension='NOT_REQUIRED'):
    plan=dict(jobs=pp.prepare.jobs(),extension_jobs=pp.prepare.jobs(True),optional_jobs=[])
    results={}
    for jobs,epochs in ((plan['jobs'],base_epochs),(plan['extension_jobs'],ext_epochs)):
        for job,epoch in zip(jobs,epochs):
            r=row('E_FIXED',job['seed'],1.,extended=job['extended'],selected=epoch);r.update(job)
            results[job['job_id']]=dict(row=r)
    state=dict(status=extension,cap_selected_count=sum(e==3 for e in base_epochs))
    if len(ext_epochs)==5:
        state.update(cap10_selected_count=sum(e==10 for e in ext_epochs),undertrained_at_10=sum(e==10 for e in ext_epochs)>=3)
    completion=dict(status=status,extension_status={'E_FIXED:jena':state})
    return plan,completion,results


def test_primary_complete_base_no_extension():
    primary,all_new,coverage=pp.select_primary(*study())
    assert len(primary)==len(all_new)==5
    assert coverage['scientific_complete'] and not coverage['provisional']
    assert coverage['primary_kind']=='BASE' and coverage['extension_status']=='NOT_REQUIRED'


@pytest.mark.parametrize('count',[0,2,4])
def test_required_partial_extension_keeps_complete_base_provisional(count):
    primary,all_new,coverage=pp.select_primary(*study((3,3,3,1,1),(6,)*count,extension='INCOMPLETE'))
    assert len(primary)==5 and len(all_new)==5+count
    assert all(not r['extended'] for r in primary)
    assert coverage['provisional'] and not coverage['scientific_complete']


@pytest.mark.parametrize('epochs,provisional',[((6,6,6,3,1),False),((10,10,10,6,3),True)])
def test_complete_extension_replaces_every_base_seed(epochs,provisional):
    primary,all_new,coverage=pp.select_primary(*study((3,3,3,1,1),epochs,extension='COMPLETE'))
    assert len(primary)==5 and len(all_new)==10 and all(r['extended'] for r in primary)
    assert coverage['primary_kind']=='EXTENDED' and coverage['provisional']==provisional
    assert sum(r['primary'] for r in all_new)==5


@pytest.mark.parametrize('mutation',['live','wrong_trigger','wrong_count','untriggered','false_complete','cap10','extension_first'])
def test_primary_rejects_inconsistent_terminal_evidence(mutation):
    args=study();plan,completion,results=args;state=completion['extension_status']['E_FIXED:jena']
    if mutation=='live': state['status']='ADMITTED'
    elif mutation=='wrong_trigger': state['status']='NOT_RUN_BUDGET'
    elif mutation=='wrong_count': state['cap_selected_count']=1
    elif mutation=='untriggered': args=study((1,)*5,(6,),extension='NOT_REQUIRED')
    elif mutation=='false_complete': args=study((3,)*5,(6,),extension='COMPLETE')
    elif mutation=='cap10':
        args=study((3,)*5,(10,)*5,extension='COMPLETE');args[1]['extension_status']['E_FIXED:jena']['undertrained_at_10']=False
    elif mutation=='extension_first': args=study((3,)*4,(6,),status='TRAINING_INCOMPLETE',extension='BASE_INCOMPLETE')
    with pytest.raises(ValueError): pp.select_primary(*args)


def test_no_budget_is_no_estimate_and_no_zero_score():
    primary,all_new,coverage=pp.select_primary(*study((),(),status='NOT_RUN_BUDGET',extension='NOT_REACHED'))
    assert primary==all_new==[] and coverage['primary_n']==0
    assert not coverage['primary_complete'] and coverage['extension_required'] is None


def test_resume_budget_exclusion_keeps_full_base_provisional_with_explicit_admission():
    args=study((3,)*5,(6,)*2,status='NOT_RUN_BUDGET',extension='ADMITTED')
    with pytest.raises(ValueError,match='Resume budget'): pp.select_primary(*args)
    args[1]['campaign_admission']=dict(status='NOT_RUN_BUDGET',remaining_base=0,reserved_extensions=3)
    primary,_,coverage=pp.select_primary(*args)
    assert len(primary)==5 and coverage['provisional'] and coverage['extension_status']=='NOT_RUN_BUDGET'


@pytest.mark.parametrize('status',['RUNNING','STOPPED','UNKNOWN'])
def test_nonterminal_completion_is_rejected_before_files_are_read(status):
    with pytest.raises(ValueError,match='terminal training'): pp.verify_completion({},dict(status=status))


def test_completion_accounts_for_every_job_and_rejects_live_status():
    plan=study()[0];completion=dict(status='NOT_RUN_BUDGET',run_statuses={})
    with pytest.raises(ValueError,match='every registered'): pp.verify_completion(plan,completion)
    completion['run_statuses']={j['job_id']:'NOT_RUN_BUDGET' for j in plan['jobs']+plan['extension_jobs']}
    completion['run_statuses'][plan['jobs'][0]['job_id']]='RUNNING'
    with pytest.raises(ValueError,match='nonterminal'): pp.verify_completion(plan,completion)


def synthetic_bundle(tmp_path,monkeypatch,selected=1.):
    monkeypatch.setattr(pp.runtime,'REPO',tmp_path)
    monkeypatch.setattr(pp.runtime,'REPORT',tmp_path/'reports')
    monkeypatch.setattr(pp.runtime,'CACHE',tmp_path/'cache')
    job=pp.prepare.jobs()[0];public=pp.runtime.REPORT/'jobs'/job['job_id']/'attempt_001'
    private=pp.runtime.CACHE/'jobs'/job['job_id']/'attempt_001'
    public.mkdir(parents=True);private.mkdir(parents=True)
    spec=dict(epochs=[.25,1.,3.]);contract={'synthetic':'bound'}
    stages=[];records=[]
    for phase,end,value in [('search',10482,1.),('refit',12579,2.)]:
        E=torch.full((21,6),value);digest=pp.float32_sha(E.numpy())
        count=3 if phase=='search' else int(selected>0)
        updates=3 if phase=='search' else int(selected>0)
        record=dict(phase=phase,status='PASS',fit_end=end,E_requires_grad=False,E_gradient_absent=True,
                    frozen_parameters=126,trainable_parameters=35591,updates=updates,
                    optimizer_parameter_group_checks=count,initial_E_sha256=digest,live_E_sha256=digest,selected_E_sha256=digest)
        epochs=[0.,.25,1.,3.] if phase=='search' else [0.]+([selected] if selected>0 else [])
        summary=dict(updates=updates,contract=dict(fit_end=end),E_INVARIANCE=record,completed_epochs=epochs)
        folder=private/phase;folder.mkdir()
        for index in range(1+count):
            torch.save(dict(model={'E':E,'D':torch.ones(1)},updates=index,
                extra={'best_state':{'E':E.clone(),'D':torch.ones(1)},'completed_epochs':epochs[:index+1],
                       'curves':[dict(epoch=e,updates=i) for i,e in enumerate(epochs[:index+1])]},
                contract=summary['contract']),folder/f'checkpoint_{index:06d}.pt')
        records.append(record);stages.append(summary)
    (public/'E_INVARIANCE.json').write_text(json.dumps(records),encoding='utf8')
    bundle=dict(model={'E':torch.full((21,6),2.),'D':torch.ones(1)},normalization=dict(basis=np.full((21,6),2.,np.float32),
                fit_end=12579,mean=np.zeros(21,np.float32),std=np.ones(21,np.float32)),spec=spec,contract=contract,
                selected_search_epoch=selected,selection_sha256='selection',search_summary=stages[0],refit_summary=stages[1])
    path=private/'selected_bundle.pt';torch.save(bundle,path)
    result=dict(job=job,row={'selected_epoch':selected},attempt=1,E_INVARIANCE=records,
                search_summary=stages[0],refit_summary=stages[1],spec=spec,contract=contract,
                selection_sha256='selection',bundle_path=str(path),bundle_sha256=pp.runtime.sha(path))
    return result,bundle,public,private


@pytest.mark.parametrize('selected',[0.,1.])
def test_actual_tensors_verify_stage_local_basis_and_zero_update_refit(tmp_path,monkeypatch,selected):
    result,_,_,_=synthetic_bundle(tmp_path,monkeypatch,selected)
    check,files=pp.verify_E_artifacts(result)
    assert check['status']=='PASS' and check['search_and_refit_basis_may_differ']
    assert check['stage_E_checks'][0]['E_sha256']!=check['stage_E_checks'][1]['E_sha256']
    assert check['stage_E_checks'][1]['optimizer_parameter_group_checks']==int(selected>0)
    assert len(files)==1


@pytest.mark.parametrize('mutation',['selected_E','basis','live_checkpoint','selected_checkpoint','checkpoint_count','optimizer_audit','E_grad','wrong_refit','copied_checkpoint','changed_D'])
def test_real_saved_tensor_and_audit_mismatch_is_rejected(tmp_path,monkeypatch,mutation):
    result,bundle,public,private=synthetic_bundle(tmp_path,monkeypatch)
    if mutation in ('selected_E','basis','wrong_refit','changed_D'):
        if mutation=='selected_E': bundle['model']['E'][0,0]=9
        elif mutation=='basis': bundle['normalization']['basis'][0,0]=9
        elif mutation=='wrong_refit': bundle['normalization']['fit_end']=10482
        else: bundle['model']['D'][0]=9
        torch.save(bundle,result['bundle_path'])
    elif mutation in ('live_checkpoint','selected_checkpoint'):
        path=private/'search/checkpoint_000001.pt';value=torch.load(path,weights_only=False)
        if mutation=='live_checkpoint': value['model']['E'][0,0]=9
        else: value['extra']['best_state']['E'][0,0]=9
        torch.save(value,path)
    elif mutation=='checkpoint_count': (private/'search/checkpoint_000001.pt').unlink()
    elif mutation=='copied_checkpoint': (private/'search/checkpoint_000001.pt').write_bytes((private/'search/checkpoint_000000.pt').read_bytes())
    else:
        record=result['E_INVARIANCE'][0]
        record['optimizer_parameter_group_checks' if mutation=='optimizer_audit' else 'E_gradient_absent']=0 if mutation=='optimizer_audit' else False
        (public/'E_INVARIANCE.json').write_text(json.dumps(result['E_INVARIANCE']),encoding='utf8')
    with pytest.raises(ValueError): pp.verify_E_artifacts(result)


def test_budget_report_and_figure_never_invent_zero_score(tmp_path):
    _,_,coverage=pp.select_primary(*study((),(),status='NOT_RUN_BUDGET',extension='NOT_REACHED'))
    pp.write_csv(tmp_path/'SUMMARY.csv',[],('MSE_median','n'))
    pp.make_figure([],coverage,[],[],tmp_path)
    pp.write_report(coverage,[],[],'NOT_RUN_BUDGET',tmp_path)
    assert (tmp_path/'SUMMARY.csv').read_text(encoding='utf-8-sig').strip()=='MSE_median,n'
    text=(tmp_path/'RESULTS_KO.md').read_text(encoding='utf8')
    assert 'NOT_RUN_BUDGET' in text and '미해결' in text and 'MSE 중앙값' not in text
    for suffix in ('png','pdf','svg'): assert (tmp_path/f'FIG_E_FIXED.{suffix}').stat().st_size>100


def test_five_seed_figure_and_report_render_synthetic_data(tmp_path):
    rows=groups();summaries,contrasts,_=pp.aggregate(rows)
    _,_,coverage=pp.select_primary(*study())
    pp.make_figure(rows,coverage,summaries,contrasts,tmp_path)
    pp.write_report(coverage,summaries,contrasts,'PASS',tmp_path)
    text=(tmp_path/'RESULTS_KO.md').read_text(encoding='utf8')
    assert 'REFERENCE_LOWER' in text and '짝 차이의 중앙값' in text and '독립 테스트' in text


def attempt_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(pp.runtime,'REPO',tmp_path);monkeypatch.setattr(pp.runtime,'REPORT',tmp_path/'reports')
    job=pp.prepare.jobs()[0];root=pp.runtime.REPORT/'jobs'/job['job_id'];public=root/'attempt_001';public.mkdir(parents=True)
    result=dict(attempt=2,registration={'seal':'synthetic'})
    for attempt in (1,2):
        folder=root/f'attempt_{attempt:03d}';folder.mkdir(exist_ok=True)
        worker=dict(job_id=job['job_id'],attempt=attempt,status='FAILED' if attempt==1 else 'COMPLETE',exit_code=1 if attempt==1 else 0)
        (root/f'WORKER_RECEIPT_attempt{attempt}.json').write_text(json.dumps(worker),encoding='utf8')
        (folder/'RECEIPT.json').write_text(json.dumps(dict(job=job,attempt=attempt,registration=result['registration'],T_numeric_values_parsed=0)),encoding='utf8')
    return job,result,{},root


def test_all_retry_receipts_are_bound_and_checked(tmp_path,monkeypatch):
    job,result,completion,_=attempt_fixture(tmp_path,monkeypatch)
    rows,files=pp.verify_attempts(job,result,completion)
    assert [r['exit_code'] for r in rows]==[1,0] and len(files)==4


@pytest.mark.parametrize('mutation',['repeated_success','repeated_scientific_stop','prior_T','false_missing_result','post_success_attempt'])
def test_attempt_audit_rejects_forbidden_retries_and_hidden_failure(tmp_path,monkeypatch,mutation):
    job,result,completion,root=attempt_fixture(tmp_path,monkeypatch)
    if mutation in ('repeated_success','repeated_scientific_stop'):
        path=root/'WORKER_RECEIPT_attempt1.json';worker=json.loads(path.read_text());worker['exit_code']=0 if mutation=='repeated_success' else 20
        path.write_text(json.dumps(worker))
    elif mutation=='prior_T':
        path=root/'attempt_001/RECEIPT.json';receipt=json.loads(path.read_text());receipt['T_numeric_values_parsed']=1;path.write_text(json.dumps(receipt))
    elif mutation=='post_success_attempt': (root/'attempt_003').mkdir()
    with pytest.raises(ValueError): pp.verify_attempts(job,result,completion,successful=mutation!='false_missing_result')


@pytest.mark.parametrize('mutation',[None,'seal','mode','binding','unknown_pid','snapshot'])
def test_host_receipt_requires_bound_policy_and_real_saved_snapshot_verification(tmp_path,monkeypatch,mutation):
    binding={'mode':'identity_only','sha':'synthetic'}
    receipt=dict(status='PASS',mode='identity_only',host_gpu_binding=binding,desktop_concurrent_load=True,
                 registration_seal_sha256='sealed',unexpected_gpu_pids=[],identity_snapshot_sha256='snapshot',
                 policy_sha256='policy',observed_gpu_process_count=2,allowed_desktop_context_count=2)
    if mutation=='seal': receipt['registration_seal_sha256']='old'
    elif mutation=='mode': receipt['mode']='benchmark'
    elif mutation=='binding': receipt['host_gpu_binding']={'different':True}
    elif mutation=='unknown_pid': receipt['unexpected_gpu_pids']=[100]
    path=tmp_path/'HOST_GPU_GATE.json';path.write_text(json.dumps(receipt),encoding='utf8')
    monkeypatch.setattr(pp.runtime,'verify_host_gpu_binding',lambda *args:binding)
    calls=[]
    def verify(value,contract):
        calls.append(value['identity_snapshot_sha256'])
        if mutation=='snapshot': raise ValueError('saved private snapshot changed')
        return {'status':'PASS'}
    helper=SimpleNamespace(__file__=str(pp.runtime.REPO/pp.runtime.HOST_GPU_HELPER_REL),verify_identity_receipt=verify)
    monkeypatch.setitem(sys.modules,f'research.{pp.runtime.UPSTREAM_NAME}.host_gpu_gate',helper)
    if mutation:
        with pytest.raises(ValueError): pp.verify_host_gate(path,{'seal_sha256':'sealed'})
    else:
        checked=pp.verify_host_gate(path,{'seal_sha256':'sealed'})
        assert calls==['snapshot'] and checked['private_snapshot_verified']
