import copy

import numpy as np
import pytest


def metric_rows():
    rows=[]
    for policy in ('DEFAULT128','COMMON_CAP'):
        rows.append(dict(dataset='fixture',row_policy=policy,family='RAW',K=7,seed=92601,
            MSE=1.,MAE=.5,p50_ms=10.,peak_allocated_MiB=250.,peak_reserved_MiB=260.))
        for seed,mse in ((92601,.989),(92602,.991)):
            rows.append(dict(dataset='fixture',row_policy=policy,family='B2',K=2,seed=seed,
                MSE=mse,MAE=.6,p50_ms=8. if policy=='DEFAULT128' else 12.,
                peak_allocated_MiB=240. if policy=='DEFAULT128' else 251.,peak_reserved_MiB=260.))
            rows.append(dict(dataset='fixture',row_policy=policy,family='DIRECT_G',K=0,seed=seed,
                MSE=.8,MAE=.4,p50_ms=1.,peak_allocated_MiB=10.,peak_reserved_MiB=20.))
    return rows


def test_fixed_positive_rule_uses_both_default_seeds_and_does_not_hide_mae():
    from research.b2_dataset_search_20261019_v1 import analysis
    result=analysis.classify(metric_rows())['fixture']
    assert result['positive'] is True
    assert result['widths']['2']['positive'] is True
    assert result['widths']['2']['mae_change_pct_vs_raw'] > 0
    assert result['widths']['2']['direct_dominates_mean'] is True


@pytest.mark.parametrize('change', ('seed_error','mean_gain','latency_equal','memory_equal'))
def test_each_registered_positive_predicate_is_required(change):
    from research.b2_dataset_search_20261019_v1 import analysis
    rows=metric_rows()
    selected=[r for r in rows if r['row_policy']=='DEFAULT128' and r['family']=='B2']
    if change=='seed_error':selected[1]['MSE']=1.
    elif change=='mean_gain':
        for row in selected:row['MSE']=.995
    elif change=='latency_equal':selected[1]['p50_ms']=10.
    else:selected[1]['peak_allocated_MiB']=250.
    assert analysis.classify(rows)['fixture']['positive'] is False


def fixture_receipt(policy='COMMON_CAP'):
    from research.b2_dataset_search_20261019_v1 import analysis as a
    origins=np.arange(2800,2800+9*48,48,dtype=np.int64)
    arrays=dict(pred_normalized=np.full((9,48,7),.25,np.float64),
        reference_pred_normalized=np.full((9,48,7),.25,np.float32),
        target_normalized=np.zeros((9,48,7),np.float32),observed=np.ones((9,48,7),bool),
        origins=origins,channel_ids=np.array([f'c{i}' for i in range(7)]),
        mean=np.zeros(7,np.float32),std=np.ones(7,np.float32))
    metrics=a.score(arrays['pred_normalized'],arrays['target_normalized'],arrays['observed'])
    cap=16 if policy=='COMMON_CAP' else 32
    call_count=2 if policy=='COMMON_CAP' else 1
    trials=[dict(phase=phase,trial=i,index=i,whole_seconds=(10+i)/1000,
        F_calls=call_count,F_logical_streams=32,F_physical_streams=32,
        peak_allocated_MiB=240. if phase=='timed' else None,
        peak_reserved_MiB=260. if phase=='timed' else None)
        for phase,n in [('warmup',10),('timed',20)] for i in range(n)]
    q25,p50,q75,p95=np.quantile(np.arange(10,30),[.25,.5,.75,.95])
    fingerprints={k:a.array_fingerprint(arrays[k]) for k in ('origins','channel_ids','mean','std','target_normalized','observed')}
    fingerprints.update(physical_windows=dict(shape=[9,512,7],dtype='float32',sha256='a'*64),
        physical_observed_windows=dict(shape=[9,512,7],dtype='bool',sha256='b'*64))
    config=dict(dataset='fixture',C=7,L=512,H=48,columns=arrays['channel_ids'].tolist(),
        interval_seconds=3600,split_ranges=dict(V_select=[2800,3232]))
    receipt=dict(status='COMPLETE',schema='b2_dataset_search_cost_v1',dataset='fixture',family='B2',
        run_id='B2_fixture_K4_OFF_s92601',seed=92601,K=4,model_K=4,q=2,C=7,L=512,H=48,
        batch_size=8,row_policy=policy,actual_F_width=4,requested_row_cap=16 if policy=='COMMON_CAP' else 128,
        effective_row_cap=cap,states_before={'F':'f'*64,'adapter':'a'*64},states_after={'F':'f'*64,'adapter':'a'*64},
        expected_work=dict(q=2,width=4,streams=32,requested_row_cap=16 if policy=='COMMON_CAP' else 128,
            effective_row_cap=cap,calls=call_count,row_sizes=[16,16] if policy=='COMMON_CAP' else [32]),
        original_reference=dict(status='PASS',full_V_origins=9,F_calls=[1,1],row_cap=128,metrics=metrics,
            canonical_prediction_fingerprint=a.array_fingerprint(arrays['reference_pred_normalized'])),
        parity=dict(status='PASS',full_V_origins=9,last_batch_size=1,atol=1e-5,rtol=1e-4,
            input_fingerprints=fingerprints,output_fingerprints=dict(
                original_normalized=a.array_fingerprint(arrays['reference_pred_normalized']),
                regenerated_normalized=a.array_fingerprint(arrays['pred_normalized'])),
            original_metrics=metrics,independent_metrics=metrics),
        summary=dict(status='COMPLETE',parity_status='PASS',MSE=metrics['MSE'],MAE=metrics['MAE'],
            recomputed_MSE=metrics['MSE'],recomputed_MAE=metrics['MAE'],warmups=10,timed_trials=20,
            median_batch_ms=p50,IQR_batch_ms=q75-q25,p95_batch_ms=p95,median_ms_per_origin=p50/8,
            origins_per_second=8000/p50,peak_allocated_MiB=240.,peak_reserved_MiB=260.,
            F_calls=call_count,F_logical_streams=32,foundation_instances_created=1,
            core_dtype='float32',cached_forecast_reads_in_timed_path=0),trials=trials)
    return receipt,arrays,config


def test_receipt_recomputes_full_v_fp64_and_raw_trial_statistics():
    from research.b2_dataset_search_20261019_v1 import analysis
    receipt,arrays,config=fixture_receipt()
    value=analysis.verify_cost(receipt,arrays,config)
    assert value['MSE']==.0625
    assert value['MAE']==.25
    assert value['n_origins']==9 and value['scored_origins']==9
    assert value['p50_ms']==19.5


@pytest.mark.parametrize('change', ('trial','quantile','origin','mask','state','cap','reference'))
def test_receipt_tampering_is_rejected(change):
    from research.b2_dataset_search_20261019_v1 import analysis
    receipt,arrays,config=fixture_receipt()
    if change=='trial':receipt['trials'][0]['F_calls']=1
    elif change=='quantile':receipt['summary']['median_batch_ms']+=1
    elif change=='origin':arrays['origins'][0]+=1
    elif change=='mask':arrays['observed'][0,0,0]=False
    elif change=='state':receipt['states_after']['adapter']='changed'
    elif change=='cap':receipt['effective_row_cap']=32
    else:arrays['reference_pred_normalized'][0,0,0]=1.
    with pytest.raises(ValueError):analysis.verify_cost(receipt,arrays,config)


def test_figures_have_deterministic_metadata_and_synthetic_marker():
    from research.b2_dataset_search_20261019_v1 import analysis
    rows=metric_rows()
    metadata=dict(dataset='fixture',C=7,interval_seconds=3600)
    first=analysis.render_figures(rows,'fixture',metadata,synthetic=True)
    second=analysis.render_figures(rows,'fixture',metadata,synthetic=True)
    assert first==second
    assert b'SYNTHETIC' in first['svg']
    assert b'<dc:date>' not in first['svg']


def test_aggregation_keeps_both_policies_and_actual_seed_ranges():
    from research.b2_dataset_search_20261019_v1 import analysis
    summary=analysis.aggregate(metric_rows())
    ours=[r for r in summary if r['family']=='B2' and r['row_policy']=='DEFAULT128'][0]
    assert ours['MSE_mean']==.99
    assert ours['MSE_min']==.989 and ours['MSE_max']==.991
    assert ours['seeds']==2
    assert len(summary)==6


def test_selection_keeps_earlier_exact_ties_and_requires_whole_schedule():
    from research.b2_dataset_search_20261019_v1 import analysis as a
    curves=[dict(epoch=e,astop_mse=v) for e,v in ((0,2),(.25,1),(1,1),(3,1.1))]
    assert a._selected(curves)==((.25,1.),False)
    with pytest.raises(ValueError):a._selected(curves[:-1])


def training_fixture():
    from research.b2_dataset_search_20261019_v1 import analysis as a
    task=dict(dataset='fixture',columns=['a','b'],L=512,H=48,
        physical=np.arange(4000,dtype=np.float32).reshape(2000,2),observed=np.ones((2000,2),bool),
        split_ranges=dict(A_fit=(0,1000),A_stop=(1000,1200),M_cal=(1200,1400),V_select=(1400,1600),T_final=(1600,2000)),
        array_sha256={},data_sha256='data')
    spec=dict(run_id='job',family='B2',dataset='fixture',seed=92601,device='cuda',epochs=[.25,1.,3.],
        lr=.001,effective_batch=16,microbatch=4,F_row_microbatch=128,checkpoint_inputs=False,latent_k=1,
        revin=False,init_kind='PCA')
    axes=dict(L=512,H=48,C=2,columns=['a','b'],fold='primary',source_array_fingerprints={})
    source=dict(spec=spec,source_sha='source',data_sha='data',model_sha='F',config_sha=a.fingerprint(dict(spec=spec,axes=axes)))
    fit=dict(status='COMPLETE_VALID',optimizer='AdamW',source_sha='source',weight_decay=0,search_fit_end=1000,refit_fit_end=1200,
        normalization_fit_end=1200,search_train_origins=441,search_stop_origins=4,search_exposures=1323,
        selected_epoch=1.,refit_exposures=641)
    return source,fit,task


@pytest.mark.parametrize('change', ('TRAIN_stride','refit_scaler','cap','seed','K'))
def test_fixed_recipe_full_train_origins_and_fresh_refit_ranges_are_enforced(change):
    from research.b2_dataset_search_20261019_v1 import analysis as a
    source,fit,task=training_fixture()
    a._fit_contract(source,fit,task,'B2',1,92601,'job')
    if change=='TRAIN_stride':fit['search_train_origins']-=1
    elif change=='refit_scaler':fit['normalization_fit_end']=1000
    elif change=='cap':source['spec']['epochs']=[.25,1.]
    elif change=='seed':source['spec']['seed']=92602
    else:source['spec']['latent_k']=2
    with pytest.raises(ValueError):a._fit_contract(source,fit,task,'B2',1,92601,'job')


def test_checkpoint_and_model_bundle_are_independently_cpu_verified(tmp_path):
    import torch
    from research.b2_dataset_search_20261019_v1 import analysis as a
    source,fit,task=training_fixture()
    mean,std=a._statistics(task,1200)
    fit['refit_scaler_sha']=a.fingerprint(dict(mean=a.array_fingerprint(mean),std=a.array_fingerprint(std)))
    state=dict(weight=torch.arange(4,dtype=torch.float32).reshape(2,2))
    fit['model_state_sha']=a.fingerprint({key:a._array_digest(value.numpy()) for key,value in state.items()})
    bundle=dict(model=state,spec=source['spec'],contract=dict(source_sha='source',data_sha='data',model_sha='F',
        config_sha=source['config_sha'],scaler_sha=fit['refit_scaler_sha']),selected_search_epoch=1.,
        normalization=dict(mean=mean,std=std,fit_end=1200),model_state_sha=fit['model_state_sha'])
    path=tmp_path/'SYNTHETIC_bundle.pt';torch.save(bundle,path)
    _,state_sha=a._bundle(path,source,fit,task,1.)
    assert len(state_sha)==64
    bundle['model']['weight'][0,0]=99;torch.save(bundle,path)
    with pytest.raises(ValueError):a._bundle(path,source,fit,task,1.)
    curves=[dict(epoch=e,astop_mse=v) for e,v in ((0,2),(.25,1.5),(1,1),(3,1.1))]
    search=dict(contract={k:source[k] for k in ('source_sha','data_sha','model_sha','config_sha')},curves=curves,
        selected_epoch=1.,summary=dict(selected_astop_mse=1.))
    mean,std=a._statistics(task,1000)
    checkpoint=dict(contract=dict(search['contract'],scaler_sha=a.fingerprint(dict(mean=a.array_fingerprint(mean),std=a.array_fingerprint(std)))),
        extra=dict(curves=curves,best_epoch=1.,best_mse=1.,completed_epochs=[0.,.25,1.,3.]))
    torch.save(checkpoint,path);a._search_checkpoint(path,search,task)
    checkpoint['extra']['best_epoch']=3.;torch.save(checkpoint,path)
    with pytest.raises(ValueError):a._search_checkpoint(path,search,task)


def test_sources_and_artifact_bindings_reject_changed_bytes_and_escape(tmp_path):
    import json
    from research.b2_dataset_search_20261019_v1 import analysis as a
    repo=tmp_path/'repo';folder=repo/'research'/a.NAME;folder.mkdir(parents=True)
    source=folder/'source.py';source.write_text('pass\n')
    files={source.relative_to(repo).as_posix():a.sha(source)}
    a._sources(files,repo)
    source.write_text('changed\n')
    with pytest.raises(ValueError):a._sources(files,repo)
    public=repo/'reports'/a.NAME;public.mkdir(parents=True)
    receipt=public/'SYNTHETIC.json';receipt.write_text(json.dumps(dict(status='test')))
    bound=dict(reference='report://SYNTHETIC.json',sha256=a.sha(receipt))
    assert a._bound(bound,repo,repo/'.cache'/a.NAME,public)==receipt
    with pytest.raises(ValueError):a._bound(dict(bound,reference='report://../../outside.json'),repo,repo,public)


def test_physical_prefix_scaler_target_masks_and_history_match_scored_cohort():
    from research.b2_dataset_search_20261019_v1 import analysis as a
    receipt,arrays,config=fixture_receipt()
    task=dict(physical=np.zeros((3232,7),np.float32),observed=np.ones((3232,7),bool),
        split_ranges=dict(A_stop=(2000,2400)))
    history=arrays['origins'][:,None]+np.arange(-512,0)[None]
    receipt['parity']['input_fingerprints'].update(physical_windows=a.array_fingerprint(task['physical'][history]),
        physical_observed_windows=a.array_fingerprint(task['observed'][history]))
    a._cohort(task,arrays,receipt)
    task['observed'][2800,0]=False
    with pytest.raises(ValueError):a._cohort(task,arrays,receipt)


def test_superseded_and_excluded_candidates_stay_in_registry(tmp_path):
    import json
    from research.b2_dataset_search_20261019_v1 import analysis as a
    report=tmp_path/'reports'/a.NAME;(report/'waves').mkdir(parents=True)
    entry=dict(dataset='ETTh2',K_grid=[1,2,4],config_sha256='c',prefix_sha256='p')
    (report/'waves'/'WAVE_001.json').write_text(json.dumps(dict(datasets=[entry],excluded=[dict(dataset='excluded',status='NOT_READY',issues=['missing'])])))
    (report/'PREPARATION_AMENDMENT_001.json').write_text('{}')
    entry=dict(entry,dataset='safe')
    (report/'waves'/'WAVE_002.json').write_text(json.dumps(dict(datasets=[entry])))
    rows=a.candidate_registry(tmp_path,2,dict(safe=dict(positive=False)))
    assert [r['status'] for r in rows]==['SUPERSEDED_BEFORE_TRAINING','NOT_READY','NEGATIVE_EXPLORATORY']


def test_conditional_ready_candidate_is_metadata_only_and_not_a_performance_experiment(tmp_path,monkeypatch):
    import json
    from research.b2_dataset_search_20261019_v1 import analysis as a
    report=tmp_path/'reports'/a.NAME;report.mkdir(parents=True)
    entry=dict(dataset='ETTm1',C=7,columns=[str(i) for i in range(7)],prefix_rows=55744,
        interval_seconds=900,K_grid=[1,2,4])
    plan=dict(planned_execution_wave=3,parent_wave=2,numeric_forecast_scores_used_to_choose_candidates=False,
        unchanged_recipe=True,unchanged_K_rule=True,ordered_candidates=[entry])
    (report/'CANDIDATE_PLAN_003.json').write_text(json.dumps(plan))
    folder=tmp_path/'.cache'/a.NAME/'data'/'ETTm1';folder.mkdir(parents=True)
    (folder/'DATA_DIAGNOSTIC.json').write_text(json.dumps(dict(entry,status='READY',prefix_sha256='metadata_only')))
    (folder/'PREFIX_PHYSICAL.npz').write_bytes(b'NOT_A_NUMERIC_ARCHIVE_DO_NOT_OPEN')
    def forbidden(*args,**kwargs):raise AssertionError('Conditional numeric prefix must not be loaded')
    monkeypatch.setattr(a.np,'load',forbidden)
    row=a.candidate_registry(tmp_path,2,dict(current=dict(positive=True)))[0]
    assert row['status']=='READY_NOT_EXECUTED' and row['forecast_results_counted']==0
    assert row['measured_in_this_report'] is False
    row=a.candidate_registry(tmp_path,2,dict(current=dict(positive=False)))[0]
    assert row['status']=='READY_CONDITIONAL_NEXT_WAVE'


def test_render_hashes_match_in_an_independent_python_process():
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import sys
    from research.b2_dataset_search_20261019_v1 import analysis as a
    rows=metric_rows();metadata=dict(dataset='fixture',C=7,interval_seconds=3600)
    expected={k:hashlib.sha256(v).hexdigest() for k,v in a.render_figures(rows,'fixture',metadata,synthetic=True).items()}
    code='import hashlib,json,sys;from research.b2_dataset_search_20261019_v1 import analysis as a;v=json.load(sys.stdin);print(json.dumps({k:hashlib.sha256(b).hexdigest() for k,b in a.render_figures(v["rows"],"fixture",v["meta"],synthetic=True).items()},sort_keys=True))'
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(a.__file__).resolve().parents[2],
        input=json.dumps(dict(rows=rows,meta=metadata)),capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==expected


def test_distinct_facets_cannot_share_pid_and_process_creation_time():
    from research.b2_dataset_search_20261019_v1 import analysis as a
    processes={(1234,1792338000.),(1234,1792338000.)}
    with pytest.raises(ValueError,match='process identity'):
        a._verify_process_coverage(processes,2)


def test_pid_reuse_with_distinct_process_creation_times_is_allowed():
    from research.b2_dataset_search_20261019_v1 import analysis as a
    processes={(1234,1792338000.),(1234,1792338001.)}
    a._verify_process_coverage(processes,2)
