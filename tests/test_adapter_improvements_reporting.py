import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.reporting')
    except ModuleNotFoundError:
        pytest.fail('Public sufficient-statistic reporter is not implemented')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def public_fixture(tmp_path):
    report=tmp_path/'public'
    old=tmp_path/'old'
    old.mkdir()
    report.mkdir()
    jobs=[]
    for family in ('P_SPLIT','P_LOCAL_CONTROL'):
        job=dict(job_id=family+'_s92601',kind='training',cohort='jena',seed=92601,family=family,K=6,device='cuda')
        jobs.append(job)
        scale=1. if family=='P_LOCAL_CONTROL' else 1.2
        counts=np.tile(np.array([3,6]),(48,1))
        sse=counts*np.array([1.,4.])*scale
        sae=counts*np.array([.5,1.])*scale
        values=dict(run_id=job['job_id'],method=family+'_PCA',dataset='jena',seed=92601,
                    stage='selected_refit',evaluation='V_select',fit='fresh_A60',C=2,K=6,L=512,H=48,
                    device='cuda',core_dtype='float32',prediction_dtype='float32',metric_accumulation_dtype='float64',
                    point_statistic='quantile_0.5',scaler_sha256='s'*64,source_sha='source',config_sha=family,
                    data_sha='data',model_sha='foundation',n_origins=6,evaluation_stride=48,
                    independent_FP64='PASS',MSE=2.5*scale,MAE=.75*scale,selected_epoch=1.,
                    channel_sse=sse.sum(0).tolist(),channel_sae=sae.sum(0).tolist(),channel_counts=counts.sum(0).tolist(),
                    horizon_sse=sse.tolist(),horizon_sae=sae.tolist(),horizon_counts=counts.tolist(),
                    array_fingerprints={key:dict(sha256=key,shape=[6,48,2],dtype='float32') for key in
                                        ('target_normalized','observed','origins','channel_ids')},
                    prediction_binding=dict(reference='cache://'+family+'.npz',sha256='p'*64),
                    target_trained_params=2160)
        result=dict(status='COMPLETE',rows=[values],recipe=dict(epochs=[.25,1.,3.],lr=.001,
                     microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False),
                    backend_flags=dict(AMP=False,TF32_matmul=False,TF32_cudnn=False))
        dump(report/'jobs'/(job['job_id']+'.json'),result)
        dump(report/'receipts'/(job['job_id']+'.json'),dict(status='COMPLETE',job_id=job['job_id']))
    jobs.append(dict(job_id='D_JENA_s92602',kind='decoder',cohort='jena',seed=92602,K=6,device='cpu'))
    dump(report/'receipts/D_JENA_s92602.json',dict(status='SKIPPED_BUDGET',reason='pair reserve'))
    dump(report/'EXPERIMENT_PLAN.json',dict(plan_sha256='plan',jobs=jobs,source_hashes={'science.py':'x'},
         registered_targets=dict(accuracy='UNREGISTERED',speedup='UNREGISTERED')))
    return dict(repo=Path(__file__).resolve().parents[1],report=report,public_old_root=old)


def test_masked_bands_reconstruct_by_counts_and_keep_undefined_channels():
    m=impl()
    rows=[dict(horizon=1,channel_id=0,SSE=4.,SAE=2.,count=2),
          dict(horizon=1,channel_id=1,SSE=9.,SAE=3.,count=1),
          dict(horizon=2,channel_id=0,SSE=9.,SAE=3.,count=3),
          dict(horizon=2,channel_id=1,SSE=0.,SAE=0.,count=0)]
    value=m.aggregate_statistics(rows, channels=[0,1])
    assert value['MSE']==pytest.approx((13/5+9)/2)
    assert value['MAE']==pytest.approx((5/5+3)/2)
    undefined=m.aggregate_statistics(rows[2:],channels=[0,1])
    assert undefined['MSE'] is None and undefined['coverage_channels']==1


def test_match_requires_full_signature_and_rejects_different_dtype_or_mask():
    m=impl()
    base=dict(cohort='jena',seed=92601,C=21,K=6,L=512,H=48,stage='selected_refit',evaluation='V_select',
              device='cuda',core_dtype='float32',point_statistic='quantile_0.5',data_sha='d',model_sha='f',
              scaler_sha256='s',target_sha256='y',mask_sha256='m',origins_sha256='o',channel_sha256='c',
              recipe_signature='r',backend_signature='b',source_signature='science',record_origin='NEW')
    assert m.match_rows(base,dict(base),'P')['match_status']=='MATCHED'
    for key,value in [('core_dtype','float64'),('mask_sha256','other')]:
        assert m.match_rows(base,dict(base,**{key:value}),'P')['match_status']=='CONTEXTUAL'
    incomplete=dict(base);incomplete.pop('model_sha')
    assert m.match_rows(base,incomplete,'P')['match_status']=='INCOMPLETE'


def test_report_preserves_negative_effect_counts_and_actual_origin_bindings(tmp_path,monkeypatch):
    m=impl();ctx=public_fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Public reporting must never open private arrays'))
    state=m.build_report(ctx)
    assert state['new_executed_scientific_jobs']==2
    assert state['reused_scientific_jobs']==0
    assert state['contextual_rows']==0
    assert state['execution_status']=='PARTIAL'
    assert state['publication_status']=='SEPARATE_REMOTE_PROOF'
    rows=csv_rows(ctx['report']/'COMPARISONS.csv')
    selected=[row for row in rows if row['question']=='P' and row['stage']=='selected_refit']
    assert len(selected)==1 and selected[0]['match_status']=='MATCHED'
    assert float(selected[0]['delta_MSE'])==pytest.approx(.5)
    assert float(selected[0]['delta_MAE'])==pytest.approx(.15)
    assert float(selected[0]['delta_MSE_pct'])==pytest.approx(20)
    runs=csv_rows(ctx['report']/'RUNS.csv')
    assert any(row['status']=='SKIPPED_BUDGET' for row in runs)
    assert all(row['source_sha256']!='UNKNOWN' for row in runs if row['record_origin']=='NEW')
    assert '전체 우수' not in (ctx['report']/'RESULT_ANALYSIS_KO.md').read_text(encoding='utf-8')


def test_contextual_legacy_rows_are_not_new_experiments_and_ridge_dtype_is_explicit(tmp_path):
    m=impl();ctx=public_fixture(tmp_path)
    with (ctx['public_old_root']/'RUNS_FULL_PRECISION.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','cohort','method','dtype','MSE','MAE','source_family','input_info_scope'])
        writer.writeheader()
        writer.writerow(dict(run_id='oldridge',cohort='jena',method='Ridge',dtype='float32',MSE=.5,MAE=.4))
        writer.writerow(dict(run_id='oldg',cohort='jena',method='Direct G',source_family='DIRECT_G',
                             input_info_scope='Own-channel; shared innovation plus anchor',MSE=.7,MAE=.5))
    state=m.build_report(ctx)
    assert state['contextual_rows']==2 and state['new_executed_scientific_jobs']==2
    rows=csv_rows(ctx['report']/'RUNS.csv')
    ridge=next(row for row in rows if row['run_id']=='oldridge')
    assert ridge['core_dtype']=='float64' and ridge['prediction_dtype']=='float32'
    assert ridge['record_origin']=='CONTEXTUAL'
    direct=next(row for row in rows if row['run_id']=='oldg')
    assert direct['method']=='G-only' and 'shared innovation' in direct['input_info_scope']


def test_public_regeneration_is_source_only_and_reproduces_tables_and_figure_data(tmp_path):
    m=impl();ctx=public_fixture(tmp_path)
    m.build_report(ctx)
    destination=tmp_path/'regenerated'
    m.regenerate(ctx['report'],destination)
    for name in ('RUNS.csv','COMPARISONS.csv','COST.csv','CHANNEL_HORIZON_STATS.csv','SAME_UPDATE.csv','PAIRED_ORIGIN_EFFECTS.csv'):
        assert (ctx['report']/name).read_bytes()==(destination/name).read_bytes()
    assert (ctx['report']/'RESULT_ANALYSIS_KO.md').read_bytes()==(destination/'RESULT_ANALYSIS_KO.md').read_bytes()
    index=json.loads((destination/'FIGURE_INDEX.json').read_text(encoding='utf-8'))
    assert {entry['figure_id'] for entry in index['figures']}=={'F1','F2','F3','F4','F5'}
    for entry in index['figures']:
        assert (destination/entry['png']).read_bytes().startswith(b'\x89PNG')
        assert '<svg' in (destination/entry['svg']).read_text(encoding='utf-8')
        assert (destination/entry['source_csv']).read_bytes()==(ctx['report']/entry['source_csv']).read_bytes()


def test_empty_report_has_headers_explicit_unknowns_and_unavailable_figures(tmp_path):
    m=impl();root=tmp_path/'empty';root.mkdir();old=tmp_path/'old';old.mkdir()
    state=m.build_report(dict(repo=Path(__file__).resolve().parents[1],report=root,public_old_root=old))
    assert state['execution_status']=='PARTIAL'
    assert (root/'COST.csv').read_text(encoding='utf-8').startswith('cost_id,')
    index=json.loads((root/'FIGURE_INDEX.json').read_text(encoding='utf-8'))
    assert all(row['status']=='UNAVAILABLE' for row in index['figures'])


def test_supplemental_costs_are_public_rows_without_extra_scientific_experiments(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    cost_job=dict(job_id='COST_JENA_CUDA_s92602',kind='costs',cohort='jena',seed=92602,device='cuda')
    dump(root/'SUPPLEMENTAL_PLANS/COST_JENA_SEED2_v1.json',dict(plan_sha256='supplement',jobs=[cost_job]))
    dump(root/'jobs/COST_JENA_CUDA_s92602.json',dict(status='COMPLETE',family='WHOLE_PATH_COST',rows=[
        dict(run_id='scored_s92602',method='D_LINEAR',dataset='jena',seed=92602,device='cuda',batch_size=1,
             median_ms_per_origin=12.3,cost_valid=True,status='COMPLETE',accuracy_run_id='scored_s92602')]))
    dump(root/'receipts/COST_JENA_CUDA_s92602.json',dict(status='COMPLETE'))
    tables,state=m.collect_public(ctx)
    assert len(tables['COST.csv'])==1 and tables['COST.csv'][0]['seed']==92602
    assert state['new_executed_scientific_jobs']==2
    assert state['supplemental_registered_jobs']==1 and state['original_registered_jobs']==3


def test_canonical_archive_proof_overrides_incompatible_producer_hash_conventions(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report'];proofs=[]
    for index,family in enumerate(('P_SPLIT','P_LOCAL_CONTROL')):
        path=root/'jobs'/(family+'_s92601.json');value=json.loads(path.read_text())
        row=value['rows'][0]
        row['array_fingerprints']['target_normalized']['sha256']='producer_convention_'+str(index)
        dump(path,value)
        proofs.append(dict(source_job_id=family+'_s92601',prediction_reference=row['prediction_binding']['reference'],
             prediction_sha256=row['prediction_binding']['sha256'],validation_status='PASS',metric_scale_ifproven='normalized',
             canonical_cohort_signature={key:dict(shape=[6,48,2],dtype='float32',sha256_raw_C_bytes=key)
                                         for key in ('target','mask','origins','channel_ids')}))
    dump(root/'VALIDATION_REPORT.json',dict(status='PASS',verified_prediction_archives=proofs))
    tables,_=m.collect_public(ctx)
    assert tables['COMPARISONS.csv'][0]['match_status']=='MATCHED'
    assert all(row['signature_proof']=='PASS' for row in tables['RUNS.csv'] if row['record_origin']=='NEW')


def authoritative_cost_fixture():
    def binding(name):return dict(reference='cache://'+name,sha256=name*64)
    run=dict(row_id='new',run_id='new',job_id='training',cohort='jena',method='P_SPLIT_PCA',evaluation='V_select',stage='selected_refit',
             MSE=1.,MAE=.5,status='COMPLETE',record_origin='NEW',source_file='jobs/new.json',source_sha256='x',
             n_origins=6,prediction_binding=binding('p'),checkpoint_binding=binding('b'),independent_FP64='PASS')
    cost=dict(cost_id='cost0',accuracy_run_id='new',accuracy_job_id='training',method='P_SPLIT_PCA',cohort='jena',device='cuda',batch_size=8,
              record_origin='NEW',job_id='cost',median_ms_per_origin=2.,cost_valid=True,status='COMPLETE',
              source_file='jobs/cost.json',source_sha256='c',
              accuracy_prediction_binding=binding('p'),accuracy_bundle_binding=binding('b'),
              accuracy_result_binding=dict(reference='report://jobs/new.json',sha256='x'),
              **{key:dict(status='PASS',n_origins=6) for key in ('saved_V_parity','serialized_reload_saved_V_parity','serialized_reload_parity')})
    raw_run=dict(run,row_id='raw',run_id='raw',job_id='HISTORICAL_RAW_V',method='Raw',record_origin='CONTEXTUAL',
                 source_file='old.csv',source_sha256='old',prediction_binding=binding('r'),checkpoint_binding=None,
                 signature_proof='PASS',MSE=2.,MAE=1.)
    raw=dict(cost,cost_id='rawcost',accuracy_run_id='raw',accuracy_job_id='HISTORICAL_RAW_V',method='Raw',
             median_ms_per_origin=6.,accuracy_prediction_binding=binding('r'),accuracy_bundle_binding=None,
             accuracy_result_binding=None,accuracy_source_kind='REUSED_HISTORICAL_SCORED_RAW')
    return run,cost,raw_run,raw


def test_presentation_cost_speedup_requires_same_measured_job_and_batch():
    m=impl();run,cost,raw_run,raw=authoritative_cost_fixture()
    rows=m.presentation_accuracy_rows({'RUNS.csv':[run,raw_run],'COST.csv':[cost,raw]})
    assert rows[0]['speedup_vs_Raw']==3.
    raw['batch_size']=1
    rows=m.presentation_accuracy_rows({'RUNS.csv':[run,raw_run],'COST.csv':[cost,raw]})
    assert rows[0]['speedup_vs_Raw'] is None


@pytest.mark.parametrize('mutation',['false_valid','string_false','failed_status','failed_parity','partial_V','wrong_prediction','wrong_bundle'])
def test_invalid_or_wrong_archive_costs_remain_unranked_in_figures_and_ppt(mutation):
    import copy
    m=impl();run,cost,raw_run,raw=authoritative_cost_fixture();cost=copy.deepcopy(cost)
    if mutation=='false_valid':cost['cost_valid']=False
    elif mutation=='string_false':cost['cost_valid']='False'
    elif mutation=='failed_status':cost['status']='FAILED_PARITY'
    elif mutation=='failed_parity':cost['saved_V_parity']['status']='FAIL'
    elif mutation=='partial_V':cost['serialized_reload_saved_V_parity']['n_origins']=1
    elif mutation=='wrong_prediction':cost['accuracy_prediction_binding']['sha256']='changed'
    elif mutation=='wrong_bundle':cost['accuracy_bundle_binding']['sha256']='changed'
    tables={'RUNS.csv':[run,raw_run],'COST.csv':[cost,raw],'COMPARISONS.csv':[],'CHANNEL_HORIZON_STATS.csv':[],'JOB_STATUS.csv':[]}
    sources=m.figure_sources(tables)
    assert all(row.get('cost_id')!='cost0' for row in sources['F1'])
    assert all(row.get('method')!='P_SPLIT_PCA' for row in sources['F5'] if row.get('kind')=='warm_latency')
    selected=next(row for row in m.presentation_accuracy_rows(tables) if row['method']=='P_SPLIT_PCA')
    assert selected['median_ms_per_origin'] is None and selected['speedup_vs_Raw'] is None
    assert len(tables['COST.csv'])==2


def test_decoder_cost_requires_exact_parent_and_saved_head_bindings():
    import copy
    m=impl();run,cost,_,_=authoritative_cost_fixture()
    state=dict(reference='cache://head.json',sha256='h'*64)
    coefficients=dict(reference='cache://head.npz',sha256='c'*64)
    run=dict(run,method='D_LINEAR',checkpoint_binding=None,PE_checkpoint_sha256='b'*64,
             state_binding=state,coefficient_binding=coefficients)
    cost=dict(cost,method='D_LINEAR',_source_private_artifacts=[state,coefficients])
    assert m.authoritative_cost_join(cost,run)['status']=='PASS'
    missing=copy.deepcopy(cost);missing['_source_private_artifacts'].pop()
    assert m.authoritative_cost_join(missing,run)['status']=='FAIL'
    wrong=copy.deepcopy(cost);wrong['accuracy_bundle_binding']['sha256']='otherPE'
    assert m.authoritative_cost_join(wrong,run)['status']=='FAIL'


def test_measured_memory_and_combined_decoder_timers_keep_actual_scope():
    m=impl()
    value=m._cost(dict(method='D_LINEAR',memory=dict(peak_allocated_MiB=2.,peak_reserved_MiB=3.),
         sampled_process_RSS_peak_MiB=4.),{},'jobs/cost.json','source',0,'NEW')
    assert value['peak_allocated_bytes']==2*1024**2
    assert value['peak_reserved_bytes']==3*1024**2
    assert value['process_RSS_peak_bytes']==4*1024**2
    row=dict(row_id='d',run_id='d',job_id='decoder',method='D_LINEAR',cohort='jena',seed=92601,
             device='cpu',stage='selected_refit',evaluation='V_select',record_origin='NEW',
             refit_and_evaluation_seconds=2.,result_timers=dict(A_stop_search_seconds=3.),source_file='jobs/d.json',source_sha256='d')
    sources=m.figure_sources({'RUNS.csv':[row],'COST.csv':[],'COMPARISONS.csv':[],'CHANNEL_HORIZON_STATS.csv':[],'JOB_STATUS.csv':[]})
    fitting=[r for r in sources['F5'] if r.get('kind')=='training_cost']
    assert any(r['metric']=='refit_and_evaluation_seconds' and r['seconds']==2 for r in fitting)
    assert any(r['metric']=='A_stop_search_seconds' and r['seconds']==3 for r in fitting)


def test_verified_historical_raw_manifest_proof_enables_canonical_accuracy_delta(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    with (ctx['public_old_root']/'RUNS_FULL_PRECISION.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','cohort','method','dtype','MSE','MAE','metric_scale','point_statistic','scaler_sha256'])
        writer.writeheader();writer.writerow(dict(run_id='old_raw',cohort='jena',method='Raw',dtype='float32',MSE=5.,MAE=1.5,
             metric_scale='canonical_A60',point_statistic='quantile_0.5',scaler_sha256='legacy_scaler'))
    signature={key:dict(shape=[6,48,2],dtype='float32',sha256_raw_C_bytes=key) for key in ('target','mask','origins','channel_ids')}
    proofs=[]
    for family in ('P_SPLIT','P_LOCAL_CONTROL'):
        row=json.loads((root/'jobs'/(family+'_s92601.json')).read_text())['rows'][0]
        proofs.append(dict(source_job_id=family+'_s92601',source_run_id=row['run_id'],method=row['method'],
            prediction_reference=row['prediction_binding']['reference'],prediction_sha256=row['prediction_binding']['sha256'],
            validation_status='PASS',metric_scale_ifproven='normalized',scaler_sha256_ifproven='canonical_scaler',
            canonical_cohort_signature=signature,statistics=dict(n_origins=6)))
    proofs.append(dict(source_job_id='HISTORICAL_RAW_V',source_run_id='old_raw',method='RAW',
        prediction_reference='previous_cache://raw.npz',prediction_sha256='r'*64,validation_status='PASS',
        metric_scale_ifproven='normalized',scaler_sha256_ifproven='canonical_scaler',canonical_cohort_signature=signature,
        statistics=dict(MSE=5.,MAE=1.5,n_origins=6)))
    dump(root/'VALIDATION_REPORT.json',dict(status='PASS',verified_prediction_archives=proofs))
    tables,_=m.collect_public(ctx)
    rows=m.presentation_accuracy_rows(tables)
    selected=next(row for row in rows if row['method']=='P_SPLIT_PCA')
    assert selected['delta_MSE_pct_vs_Raw']==pytest.approx(-40.)
    raw=next(row for row in tables['RUNS.csv'] if row['method']=='Raw')
    assert raw['record_origin']=='CONTEXTUAL' and raw['job_id']=='HISTORICAL_RAW_V'
    assert raw['prediction_binding']['sha256']=='r'*64


def test_passing_bound_costs_survive_public_csv_replay(tmp_path):
    m=impl();run,cost,raw_run,raw=authoritative_cost_fixture()
    m._write_csv(tmp_path/'runs.csv',[run,raw_run],m.RUN_FIELDS)
    m._write_csv(tmp_path/'cost.csv',[cost,raw],m.COST_FIELDS)
    tables={'RUNS.csv':csv_rows(tmp_path/'runs.csv'),'COST.csv':csv_rows(tmp_path/'cost.csv')}
    row=next(row for row in m.presentation_accuracy_rows(tables) if row['method']=='P_SPLIT_PCA')
    assert row['speedup_vs_Raw']==3. and row['median_ms_per_origin']==2.


def test_compact_statistics_keep_all_legacy_endpoints_and_join_short_run_ids(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);old=ctx['public_old_root']
    long_id='legacy_selected_'+'x'*400
    endpoint='A03_B0_s92601_e0_A_fit'
    with (old/'RUNS_FULL_PRECISION.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','cohort','method','MSE','MAE'])
        writer.writeheader();writer.writerow(dict(run_id=long_id,cohort='jena',method='B0',MSE=1.,MAE=.5))
    with (old/'SAME_UPDATE_RESULTS.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','cohort','method','split','epoch','MSE','MAE'])
        writer.writeheader();writer.writerow(dict(run_id=endpoint,cohort='jena',method='B0',split='A_fit',epoch=0.,MSE=.3,MAE=.2))
    legacy_stats=[dict(run_id=long_id,horizon=1,channel_id=0,SSE='0.12345678901234567',SAE='0.76543210987654321',count=3),
                  dict(run_id=endpoint,horizon=2,channel_id=1,SSE='4.123456789012345',SAE='2.234567890123456',count=6)]
    with (old/'ERROR_BY_HORIZON.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','horizon','channel_id','SSE','SAE','count'])
        writer.writeheader();writer.writerows(legacy_stats)
    tables,state=m.collect_public(ctx)
    expected={'row_id','horizon','channel_id','SSE','SAE','count'}
    stats=tables['CHANNEL_HORIZON_STATS.csv']
    assert all(set(row)==expected for row in stats)
    assert len(stats)==194
    lookup={row['row_id']:row for row in tables['RUNS.csv']}
    assert len(lookup)==len(tables['RUNS.csv'])
    assert all(len(row_id)==6 and row_id.startswith('r') for row_id in lookup)
    selected=next(row for row in lookup.values() if row['run_id']==long_id)
    endpoint_row=next(row for row in lookup.values() if row['run_id']==endpoint)
    assert selected['producer_row_id'].startswith(long_id)
    preserved=next(row for row in stats if row['row_id']==selected['row_id'])
    assert preserved['SSE']=='0.12345678901234567' and preserved['SAE']=='0.76543210987654321'
    assert any(row['row_id']==endpoint_row['row_id'] for row in tables['SAME_UPDATE.csv'])
    assert endpoint_row['record_origin']=='CONTEXTUAL'
    assert state['contextual_rows']==1 and state['contextual_endpoint_rows']==1
    assert all(row['candidate_row_id'] in lookup and row.get('control_row_id') in lookup for row in tables['COMPARISONS.csv'])
    m._write_csv(tmp_path/'stats.csv',stats,m.STAT_FIELDS)
    assert (tmp_path/'stats.csv').read_text(encoding='utf-8').splitlines()[0]=='row_id,horizon,channel_id,SSE,SAE,count'


def test_compact_row_ids_reject_duplicate_producer_ids():
    m=impl()
    tables={'RUNS.csv':[dict(row_id='duplicate'),dict(row_id='duplicate')],
            'SAME_UPDATE.csv':[],'CHANNEL_HORIZON_STATS.csv':[]}
    with pytest.raises(ValueError,match='Duplicate producer row ID'):
        m.compact_row_ids(tables)


@pytest.mark.parametrize('producer_status',['COMPLETE_PROTOCOL','COMPLETE'])
def test_historical_context_cost_stays_visible_without_claiming_accuracy_binding_or_speedup(producer_status):
    m=impl()
    contextual=dict(cost_id='oldG',method='G-only',cohort='jena',seed=92601,device='cpu',batch_size=1,
         record_origin='CONTEXTUAL',status='CONTEXTUAL_HISTORICAL',producer_status=producer_status,parity_state='PASS',
         median_ms_per_origin=.1356,core_dtype='float32',cost_valid='HISTORICAL_PROTOCOL_ONLY',
         source_file='old/COST.csv',source_sha256='old',scope='Historical CPU BLAS2 whole physical path')
    tables={'RUNS.csv':[],'COST.csv':[contextual],'COMPARISONS.csv':[],'CHANNEL_HORIZON_STATS.csv':[],'JOB_STATUS.csv':[]}
    rows=m.presentation_accuracy_rows(tables)
    assert len(rows)==1 and rows[0]['median_ms_per_origin']==.1356
    assert rows[0]['MSE'] is None and rows[0]['MAE'] is None
    assert rows[0]['speedup_vs_Raw'] is None
    assert rows[0]['cost_comparison_status']=='NONAUTHORITATIVE_HISTORICAL_CONTEXT'
    assert rows[0]['cost_record_origin']=='CONTEXTUAL'


def test_initialization_attempt_costs_are_shared_once_and_preserve_overlapping_scopes(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    plan=json.loads((root/'EXPERIMENT_PLAN.json').read_text())
    job=dict(job_id='DCA_INIT_JENA_A_fit',kind='initialization',cohort='jena',scope='A_fit',K=6,device='cpu')
    plan['jobs'].append(job);dump(root/'EXPERIMENT_PLAN.json',plan)
    dump(root/'jobs/DCA_INIT_JENA_A_fit.json',dict(status='NOT_CONVERGED',fit_scope=dict(name='A_fit',fit_end=80),
         diagnostics=dict(preparation_wall_seconds=3.062)))
    dump(root/'receipts/DCA_INIT_JENA_A_fit.json',dict(status='NOT_CONVERGED',consumed_resource=dict(wall_seconds=6.11)))
    tables,_=m.collect_public(ctx)
    entry=next(row for row in tables['JOB_STATUS.csv'] if row['job_id']==job['job_id'])
    assert entry['initialization_preparation_seconds']==3.062 and entry['parent_job_wall_seconds']==6.11
    sources=m.figure_sources(tables)
    entries=[row for row in sources['F5'] if row.get('job_id')==job['job_id']]
    assert len(entries)==2 and {row['seconds'] for row in entries}=={3.062,6.11}
    assert all(row['seed']=='SHARED_ONCE' and row['K']==6 and row['fit_scope']=='A_fit' for row in entries)
    assert all('overlap' in row['timing_scope'] and row['attempt_status']=='NOT_CONVERGED' for row in entries)


def test_public_paired_origins_use_fixed_channel_macro_and_keep_undefined_origin(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    for family in ('P_SPLIT','P_LOCAL_CONTROL'):
        path=root/'jobs'/(family+'_s92601.json');value=json.loads(path.read_text());row=value['rows'][0]
        row['n_origins']=2
        row['origin_counts']=[[1,9],[0,3]]
        row['origin_sse']=[[1,18],[0,6]] if family=='P_SPLIT' else [[4,9],[0,3]]
        row['origin_sae']=[[.5,9],[0,3]] if family=='P_SPLIT' else [[2,4.5],[0,1.5]]
        dump(path,value)
    tables,_=m.collect_public(ctx)
    effects=tables['PAIRED_ORIGIN_EFFECTS.csv']
    assert len(effects)==2
    first,second=effects
    assert first['origin_index']==0 and first['C']==2 and first['n_origins']==2
    assert first['candidate_MSE']==pytest.approx((1/1+18/9)/2)
    assert first['control_MSE']==pytest.approx((4/1+9/9)/2)
    assert first['delta_MSE']==pytest.approx(-1.)
    assert first['delta_MAE']==pytest.approx(-.5)
    assert first['status']=='DEFINED' and first['evaluation_stride']==48 and first['horizon_overlap'] is False
    assert second['status']=='UNDEFINED' and second['delta_MSE'] is None and second['delta_MAE'] is None
    assert second['candidate_coverage_channels']==1 and second['control_coverage_channels']==1
    assert first['candidate_source_sha256'] and first['control_source_sha256']
    assert first['candidate_source_collection']=='rows' and first['candidate_source_row_index']==0


def test_decoder_public_origin_dictionary_path_matches_array_path_and_rejects_bad_index_alignment():
    m=impl()
    dictionary=dict(origin_statistics=[dict(origin_index=0,channel_sse=[1,18],channel_sae=[.5,9],channel_counts=[1,9]),
                                      dict(origin_index=1,channel_sse=[0,6],channel_sae=[0,3],channel_counts=[0,3])])
    arrays=dict(origin_sse=[[1,18],[0,6]],origin_sae=[[.5,9],[0,3]],origin_counts=[[1,9],[0,3]])
    assert m.origin_statistics(dictionary,2)==m.origin_statistics(arrays,2)
    dictionary['origin_statistics'][1]['origin_index']=0
    with pytest.raises(ValueError,match='origin index alignment'):
        m.origin_statistics(dictionary,2)


def historical_g_controls():
    repo=Path(__file__).resolve().parents[1]
    values=csv_rows(repo/'reports/adapter_diagnostics_20261017_v2/RUNS_FULL_PRECISION.csv')
    return repo,[row for row in values if row.get('source_family')=='DIRECT_G' and row['cohort'] in ('jena','pems32')]


def test_pinned_public_g_costs_keep_four_context_facets_p90_and_exact_route():
    m=impl();repo,controls=historical_g_controls()
    rows,bindings=m.historical_g_costs(repo,controls)
    assert len(rows)==4
    assert {(row['cohort'],row['batch_size']) for row in rows}=={('jena',1),('jena',8),('pems32',1),('pems32',8)}
    for row in rows:
        assert row['method']=='G-only' and row['comparison_scope']=='HISTORICAL_CPU2_CONTEXT'
        assert row['record_origin']=='CONTEXTUAL' and row['cost_valid'] is not True
        assert row['core_dtype']=='float32' and row['physical_output_dtype']=='float64'
        assert row['p90_ms_per_origin']>0 and row.get('p95_ms_per_origin') in (None,'UNKNOWN')
        assert row['host_identity']=='UNKNOWN' and row['new_measurement'] is False
        assert row['source_file'] in bindings and row['trials_source_file'] in bindings
        assert row['trials_rows']==50 and row['F_calls']==0
    jena=next(row for row in rows if row['cohort']=='jena' and row['batch_size']==8)
    assert jena['median_ms_per_origin']==pytest.approx(.0003303999546915293*1000/8)
    assert 'TRANSITIVE' in jena['accuracy_binding_scope']
    tables={'RUNS.csv':[],'COST.csv':rows,'COMPARISONS.csv':[],'CHANNEL_HORIZON_STATS.csv':[],'JOB_STATUS.csv':[]}
    assert len(m.figure_sources(tables)['F1'])==4
    assert len(m.presentation_accuracy_rows(tables))==4
    assert all(row['speedup_vs_Raw'] is None for row in m.presentation_accuracy_rows(tables))


@pytest.mark.parametrize('mutation',['checkpoint','scaler','receipt_sha','trials_sha'])
def test_historical_g_import_rejects_source_or_route_mutation(mutation,monkeypatch):
    m=impl();repo,controls=historical_g_controls()
    controls=[dict(row) for row in controls]
    if mutation=='checkpoint':controls[0]['checkpoint_sha256']='wrong'
    elif mutation=='scaler':controls[0]['scaler_sha256']='wrong'
    else:
        original=m._sha
        def changed(path):
            text=Path(path).as_posix()
            if ('DIRECT_G_B1.json' in text and mutation=='receipt_sha') or ('DIRECT_G_B1.trials.csv.gz' in text and mutation=='trials_sha'):
                return 'changed'
            return original(path)
        monkeypatch.setattr(m,'_sha',changed)
    with pytest.raises(ValueError,match='Historical G'):
        m.historical_g_costs(repo,controls)


def finding_tables():
    m=impl()
    base=dict(cohort='jena',seed=92601,C=21,K=6,L=512,H=48,stage='selected_refit',evaluation='V_select',
              device='cpu',core_dtype='float64',point_statistic='quantile_0.5',model_sha='f',
              scaler_sha256='s',target_sha256='y',mask_sha256='m',origins_sha256='o',channel_sha256='c',
              recipe_signature='r',backend_signature='b',source_signature='science',record_origin='NEW',
              source_file='jobs/test.json',source_sha256='source',status='COMPLETE')
    runs=[]
    for seed,linear,gain,split in [(92601,(1.2,.4),(.9,.45),(1.2,.4)),(92602,(.8,.6),(.85,.55),(.8,.6))]:
        for method,metrics in [('D_SHARED',(1.,.5)),('D_LINEAR',linear),('D_GAIN',gain),
                               ('P_LOCAL_CONTROL_PCA',(1.,.5)),('P_SPLIT_PCA',split)]:
            identity=method+'_'+str(seed)
            runs.append(dict(base,row_id=identity,run_id=identity,method=method,seed=seed,MSE=metrics[0],MAE=metrics[1]))
    for cohort,scale in [('jena',9876.),('pems32',1.)]:
        for method,metrics in [('B2_PCA',(.8,.35)),('B2_DCA',(.7*scale,.3*scale))]:
            identity=cohort+method
            runs.append(dict(base,row_id=identity,run_id=identity,method=method,cohort=cohort,MSE=metrics[0],MAE=metrics[1]))
    comparisons=m._comparisons(runs)
    comparisons.append(dict(comparisons[0],stage='search',evaluation='A_fit',epoch=.25,
                            delta_MSE=9999.,delta_MAE=8888.))
    return {'RUNS.csv':runs,'COMPARISONS.csv':comparisons,'COST.csv':[],
            'JOB_STATUS.csv':[dict(cohort='jena',kind='initialization',fit_scope='A_fit',status='NOT_CONVERGED')]}


def test_generated_findings_use_selected_matched_pairs_and_preserve_mixed_seed_signs():
    m=impl();tables=finding_tables()
    paragraphs=m.selected_findings(tables)
    assert len(paragraphs)==4
    p,e,d,cost=paragraphs
    assert 'jena seed 92601' in p and 'jena seed 92602' in p
    assert 'MSE +0.2 (악화)' in p and 'MAE -0.1 (개선)' in p
    assert 'MSE -0.2 (개선)' in p and 'MAE +0.1 (악화)' in p
    assert 'pems32 seed 92601' in e and 'MSE -0.1 (개선)' in e
    assert 'jena' in e and 'NOT_CONVERGED' in e and 'UNKNOWN' in e
    assert 'D_LINEAR−D_SHARED' in d and 'D_LINEAR−D_GAIN' in d
    assert 'MSE +0.3 (악화)' in d and 'MAE -0.05 (개선)' in d
    assert 'MSE -0.05 (개선)' in d and 'MAE +0.05 (악화)' in d
    assert 'UNKNOWN' in cost
    assert all('9999' not in paragraph and '8888' not in paragraph and '6912' not in paragraph for paragraph in paragraphs)


def test_direct_linear_gain_finding_requires_actual_matched_selected_run_signatures():
    m=impl();tables=finding_tables()
    gain=next(row for row in tables['RUNS.csv'] if row['method']=='D_GAIN' and row['seed']==92601)
    gain['mask_sha256']='different'
    paragraph=m.selected_findings(tables)[2]
    seed1=paragraph.split('jena seed 92601:')[1].split('jena seed 92602:')[0]
    assert 'D_LINEAR−D_GAIN UNKNOWN' in seed1
    assert 'MSE +0.3' not in seed1


def test_generated_cost_finding_uses_exact_authority_and_same_job_batch_only():
    m=impl();run,cost,raw_run,raw=authoritative_cost_fixture()
    comparison=dict(match_status='MATCHED',question='P',stage='selected_refit',evaluation='V_select',
        cohort='jena',seed=92601,candidate='P_SPLIT_PCA',control='P_LOCAL_CONTROL_PCA',
        candidate_row_id=run['row_id'],control_row_id='control',delta_MSE=.2,delta_MAE=-.1)
    tables={'RUNS.csv':[dict(run,seed=92601),raw_run],'COST.csv':[cost,raw],
            'COMPARISONS.csv':[comparison],'JOB_STATUS.csv':[]}
    paragraph=m.selected_findings(tables)[3]
    assert 'cuda B8' in paragraph and '3x' in paragraph and 'jobs/cost.json' in paragraph
    raw['batch_size']=1
    assert 'UNKNOWN' in m.selected_findings(tables)[3]
    raw['batch_size']=8;cost['accuracy_bundle_binding']=dict(reference='cache://b',sha256='wrong')
    assert 'UNKNOWN' in m.selected_findings(tables)[3]


def test_analysis_places_selected_findings_before_separate_complete_endpoint_details(tmp_path):
    m=impl();ctx=public_fixture(tmp_path)
    tables,state=m.collect_public(ctx)
    common=dict(tables['COMPARISONS.csv'][0],stage='search',evaluation='A_stop',epoch=.25,
                matching_label='MATCHED_COMMON_ENDPOINT',delta_MSE=9999.)
    tables['COMPARISONS.csv'].append(common)
    m.write_documents(tables,state,tmp_path)
    text=(tmp_path/'RESULT_ANALYSIS_KO.md').read_text(encoding='utf-8')
    assert text.index('선택 결과 핵심 해석')<text.index('선택-refit V 상세')<text.index('공통 endpoint 상세')
    assert '9999' not in text.split('선택-refit V 상세')[0]
    assert '9999' in text.split('공통 endpoint 상세')[1]


def test_clustered_accuracy_legend_titles_and_cost_xlabel_stay_inside_canvas(tmp_path,monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    m=impl();plt=m._plot_style();captured=[]
    original_close=plt.close
    def capture(figure):
        if hasattr(figure,'axes'):captured.append(figure)
        original_close(figure)
    monkeypatch.setattr(plt,'close',capture)
    scope='HISTORICAL_CPU2_CONTEXT | CPU threads 2 | host UNKNOWN | measured physical whole path with the original protocol retained'
    points=[dict(cohort='jena',device='cpu',batch_size=8,record_origin='CONTEXTUAL',hardware_protocol_scope=scope,
                 model_scope='primary_Bolt_or_direct',method=method,seed=seed,median_ms_per_origin=1.+index*1e-8,
                 MSE=.2+index*1e-10,MAE=.1+index*1e-10,source_file='public.csv',source_sha256='source')
            for index,(method,seed) in enumerate((method,seed) for method in
                ('B0','B2','D_SHARED','D_HORIZON','D_LINEAR','D_GAIN','P_SPLIT_PCA','P_LOCAL_CONTROL_PCA')
                for seed in (92601,92602))]
    costs=[dict(kind='training_cost',cohort='jena',method='D_LINEAR',seed=92601,metric='refit_and_evaluation_seconds',value=12.,
                source_file='jobs/d.json',source_sha256='source')]
    m.render_figures({'F1':points,'F5':costs},tmp_path,dpi=80)
    accuracy,cost=captured
    data_axes=[ax for ax in accuracy.axes if ax.get_ylabel() in ('MSE (normalized)','MAE (normalized)')]
    assert len(data_axes)==2
    expected_labels={str(row['method'])+' accuracy s'+str(row['seed']) for row in points}
    legends=[ax.get_legend() for ax in accuracy.axes if ax.get_legend() is not None]
    assert len(legends)==2
    for legend in legends:
        assert expected_labels<={' '.join(text.get_text().split()) for text in legend.get_texts()}
    for ax,metric in zip(data_axes,('MSE','MAE')):
        assert len(ax.collections)==len(points) and not ax.texts
        assert [tuple(collection.get_offsets()[0]) for collection in ax.collections]==[
            (row['median_ms_per_origin'],row[metric]) for row in points]
        assert '\n' in ax.get_title()
        assert ' '.join(ax.get_title().split())==' '.join((m._facet(points[0])+' | primary_Bolt_or_direct').split())
    assert '\n' in cost.axes[1].get_xlabel()
    assert 'jena | D_LINEAR | s92601' in cost.axes[1].get_yticklabels()[0].get_text()
    painted=[];original_draw=Text.draw
    def record_draw(text,renderer):
        if text.get_visible() and text.get_text().strip():painted.append(text)
        return original_draw(text,renderer)
    monkeypatch.setattr(Text,'draw',record_draw)
    for figure in captured:
        painted.clear();FigureCanvasAgg(figure).draw();renderer=figure.canvas.get_renderer();canvas=figure.bbox
        for text in painted:
            bounds=text.get_window_extent(renderer)
            assert bounds.x0>=canvas.x0-.5 and bounds.x1<=canvas.x1+.5, text.get_text()
            assert bounds.y0>=canvas.y0-.5 and bounds.y1<=canvas.y1+.5, text.get_text()
    for ax in data_axes:
        assert not ax.title.get_window_extent().overlaps(accuracy._suptitle.get_window_extent())


def test_repeated_raw_measurements_keep_accuracy_seed_and_measurement_identity():
    m=impl();run,cost,raw_run,raw=authoritative_cost_fixture()
    raw_run['seed']=92601
    supplement=dict(raw,cost_id='COST_JENA_CUDA_s92602:0',job_id='COST_JENA_CUDA_s92602',seed=92602)
    tables={'RUNS.csv':[run,raw_run],'COST.csv':[raw,supplement],'COMPARISONS.csv':[],
            'CHANNEL_HORIZON_STATS.csv':[],'JOB_STATUS.csv':[]}
    sources=m.figure_sources(tables)
    for key in ('F1','F5'):
        rows=sources[key]
        assert len(rows)==2 and all(row['seed']==92601 for row in rows)
        assert {row['measurement_job_id'] for row in rows}=={'cost','COST_JENA_CUDA_s92602'}
        assert {row['measurement_seed'] for row in rows}=={'UNKNOWN',92602}
        labels=m._cost_labels(rows)
        assert len(set(labels))==2 and all('Raw accuracy s92601' in label for label in labels)
        assert any('COST_JENA_CUDA_s92602:0' in label for label in labels)
        assert all('accuracy s92602' not in label for label in labels)


def test_cost_rows_reserve_actual_multiline_glyph_height_without_overlap(tmp_path,monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    m=impl();plt=m._plot_style();captured=[];original_close=plt.close
    def capture(figure):
        if hasattr(figure,'axes'):captured.append(figure)
        original_close(figure)
    monkeypatch.setattr(plt,'close',capture)
    rows=[dict(kind='training_cost',cohort='jena',method='D_SHARED',seed=92601,
               metric='refit_seconds',value=index+1.,source_file='jobs/d.json',source_sha256='source') for index in range(28)]
    for cohort in ('jena','pems32'):
        rows += [dict(kind='training_cost',cohort=cohort,method='DCA_INIT A_fit shared once per dataset/K/scope; preparation and parent wall overlap',
                    seed='SHARED_ONCE',K=6,fit_scope='A_fit',metric='initialization_preparation_seconds',value=3.062,
                    source_file='jobs/dca.json',source_sha256='source'),
                 dict(kind='training_cost',cohort=cohort,method='JOB_FAMILY D_JENA_s92601_REGISTERED_FIXED_PE_FOUR_HEAD_RECIPE',seed=92601,
                    metric='frozen_cache_extraction_and_separately_accounted_refit_and_evaluation_seconds',value=12.,
                    source_file='jobs/d.json',source_sha256='source')]
    m.render_figures({'F5':rows},tmp_path,dpi=80)
    figure=captured[0];FigureCanvasAgg(figure).draw();renderer=figure.canvas.get_renderer();axis=figure.axes[1]
    labels=axis.get_yticklabels();bounds=[label.get_window_extent(renderer) for label in labels]
    assert len(axis.patches)==len(rows) and [bar.get_width() for bar in axis.patches]==[row['value'] for row in rows]
    assert any(label.get_text().count('\n')>=2 and 'DCA_INIT' in label.get_text() for label in labels)
    assert any(label.get_text().count('\n')>=2 and 'JOB_FAMILY' in label.get_text() for label in labels)
    for previous,current in zip(bounds,bounds[1:]):
        assert previous.y0>current.y1+1., 'Adjacent rendered cost row glyphs overlap'
    for bound in bounds:
        assert bound.x0>=figure.bbox.x0 and bound.x1<=figure.bbox.x1
        assert bound.y0>=figure.bbox.y0 and bound.y1<=figure.bbox.y1


@pytest.mark.parametrize('existing_csv',[False,True])
def test_multik_receipt_bound_producer_bytes_survive_build_and_public_replay(tmp_path,existing_csv):
    import hashlib
    import io
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    plan=json.loads((root/'EXPERIMENT_PLAN.json').read_text())
    plan['jobs'].append(dict(job_id='MULTIK_AUDIT',kind='multik_audit',cohort='existing'))
    dump(root/'EXPERIMENT_PLAN.json',plan)
    audit_rows=[dict(parent='parent "one", original',stage='search',K=6,completion=None,
                     actual_F_rows=12,checkpoint_bytes=None),
                dict(parent='parent two',stage='refit',K=3,completion='UNKNOWN',switch_time=.12345678901234567)]
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=sorted(set().union(*(row.keys() for row in audit_rows))))
    writer.writeheader();writer.writerows(audit_rows)
    producer_bytes=buffer.getvalue().encode('utf-8')
    expected_sha=hashlib.sha256(producer_bytes).hexdigest()
    dump(root/'jobs/MULTIK_AUDIT.json',dict(status='COMPLETE',rows=audit_rows,prefix_rule='registered lag-major prefix'))
    dump(root/'receipts/MULTIK_AUDIT.json',dict(status='COMPLETE',artifact_hashes={'MULTIK_AUDIT.csv':expected_sha}))
    producer_job_bytes=(root/'jobs/MULTIK_AUDIT.json').read_bytes()
    receipt_bytes=(root/'receipts/MULTIK_AUDIT.json').read_bytes()
    if existing_csv:(root/'MULTIK_AUDIT.csv').write_bytes(producer_bytes)
    m.build_report(ctx)
    assert (root/'MULTIK_AUDIT.csv').read_bytes()==producer_bytes
    assert hashlib.sha256((root/'MULTIK_AUDIT.csv').read_bytes()).hexdigest()==expected_sha
    destination=tmp_path/'replay'
    m.regenerate(root,destination)
    assert (destination/'MULTIK_AUDIT.csv').read_bytes()==producer_bytes
    assert (root/'jobs/MULTIK_AUDIT.json').read_bytes()==producer_job_bytes
    assert (root/'receipts/MULTIK_AUDIT.json').read_bytes()==receipt_bytes


def test_multik_receipt_mismatch_rejects_write_before_changing_bound_csv(tmp_path):
    m=impl();path=tmp_path/'MULTIK_AUDIT.csv';original=b'previous immutable producer bytes\r\n'
    path.write_bytes(original)
    with pytest.raises(ValueError,match='original receipt binding'):
        m._write_multik_csv(path,[dict(parent='different',K=6)],expected_sha='0'*64)
    assert path.read_bytes()==original


def test_large_redundant_paired_payload_stays_in_producer_and_paired_table_not_runs(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);root=ctx['report']
    plan=json.loads((root/'EXPERIMENT_PLAN.json').read_text())
    job=dict(job_id='D_PEMS32',kind='decoder_transfer',cohort='pems32',seed=92601,K=8,device='cuda')
    plan['jobs'].append(job);dump(root/'EXPERIMENT_PLAN.json',plan)
    original=json.loads((root/'jobs/P_SPLIT_s92601.json').read_text())
    channels=32;origins=148;horizon=12;rows=[]
    for method,scale in [('D_SHARED',1.),('D_HORIZON',1.2),('D_LINEAR',2.1234567890123456),('D_GAIN',.9)]:
        ss=[horizon*.1*(channel+1)*scale for channel in range(channels)]
        aa=[value*.5 for value in ss]
        origin_rows=[dict(origin_index=index,channel_sse=ss,channel_sae=aa,channel_counts=[horizon]*channels)
                     for index in range(origins)]
        paired=[dict(origin_index=index,delta_MSE=1.65*(scale-1),delta_MAE=.825*(scale-1),
                     channel_counts=[horizon]*channels,
                     channel_delta_sse=[horizon*.1*(channel+1)*(scale-1) for channel in range(channels)],
                     channel_delta_sae=[horizon*.05*(channel+1)*(scale-1) for channel in range(channels)])
                for index in range(origins)]
        rows.append(dict(original['rows'][0],run_id=method,method=method,dataset='pems32',seed=92601,C=channels,
            K=8,H=horizon,n_origins=origins,evaluation_stride=horizon,core_dtype='float64',
            MSE=1.65*scale,MAE=.825*scale,
            horizon_sse=[[origins*.1*(channel+1)*scale for channel in range(channels)] for _ in range(horizon)],
            horizon_sae=[[origins*.05*(channel+1)*scale for channel in range(channels)] for _ in range(horizon)],
            horizon_counts=[[origins]*channels for _ in range(horizon)],
            origin_statistics=origin_rows,origin_paired_vs_shared=paired))
    assert len(json.dumps(rows[2]['origin_paired_vs_shared']))>csv.field_size_limit()
    dump(root/'jobs/D_PEMS32.json',dict(original,rows=rows))
    dump(root/'receipts/D_PEMS32.json',dict(status='COMPLETE'))
    producer_bytes=(root/'jobs/D_PEMS32.json').read_bytes()
    old_limit=csv.field_size_limit()
    m.build_report(ctx)
    saved=csv_rows(root/'RUNS.csv')
    assert saved and all('origin_paired_vs_shared' not in row for row in saved)
    assert csv.field_size_limit()==old_limit
    assert (root/'jobs/D_PEMS32.json').read_bytes()==producer_bytes
    assert json.loads(producer_bytes)['rows'][2]['origin_paired_vs_shared']==rows[2]['origin_paired_vs_shared']
    paired=csv_rows(root/'PAIRED_ORIGIN_EFFECTS.csv')
    decoder=[row for row in paired if row['candidate'].startswith('D_')]
    assert len(decoder)==3*origins and all(row['status']=='DEFINED' for row in decoder)
    linear=next(row for row in decoder if row['candidate']=='D_LINEAR')
    assert float(linear['delta_MSE'])==pytest.approx(1.65*(2.1234567890123456-1))


def test_228_ppt_rows_nullable_historical_metadata_replay_exact_csv_and_korean_table(tmp_path,monkeypatch):
    m=impl();ctx=public_fixture(tmp_path);tables,state=m.collect_public(ctx);source=ctx['report']
    base=next(row for row in tables['RUNS.csv'] if row.get('stage')=='selected_refit')
    tables['RUNS.csv']=[dict(base,row_id='r'+str(index),run_id='selected_'+str(index),method='B0',seed=92601)
                        for index in range(182)]
    tables['COMPARISONS.csv']=[];tables['CHANNEL_HORIZON_STATS.csv']=[];tables['SAME_UPDATE.csv']=[]
    costs=[]
    for index in range(46):
        cost=dict(cost_id='legacy:'+str(index),method='G-only' if index>=42 else 'B0',cohort='jena',seed=None if index<18 else 92601,C=21,
            device='cpu',batch_size=8,record_origin='CONTEXTUAL',status='CONTEXTUAL_HISTORICAL',
            producer_status='COMPLETE_PROTOCOL',parity_state='PASS',cost_valid='HISTORICAL_PROTOCOL_ONLY',
            median_ms_per_origin=.1356,accuracy_MSE=.7,accuracy_MAE=.5,
            source_file='old/COST.csv',source_sha256='old',core_dtype='float32')
        if index<42:cost['process_RSS_peak_bytes']=1024*1024
        else:cost.update(K=0,accuracy_binding_scope='EXACT_HISTORICAL_PUBLIC_MANIFEST',accuracy_source_file='old/manifest.json',
            accuracy_source_sha256='manifest',comparison_scope='HISTORICAL_CPU2_CONTEXT',
            hardware_protocol_scope='CPU threads 2 | host UNKNOWN',host_identity='UNKNOWN',p90_ms_per_origin=.14,
            trials_source_file='old/TRIALS.csv.gz',trials_source_sha256='trials')
        costs.append(cost)
    tables['COST.csv']=costs
    for name,fields in m.TABLES.items():m._write_csv(source/name,tables[name],fields)
    m._json(source/'REPORT_STATE.json',state)
    m.write_documents(tables,state,source)
    monkeypatch.setattr(m,'render_figures',lambda *args,**kwargs:[])
    replay=tmp_path/'replay';m.regenerate(source,replay)
    rows=csv_rows(source/'PPT_T1_ACCURACY_COST.csv')
    assert len(rows)==228
    for name in ('PPT_T1_ACCURACY_COST.csv','PPT_TABLES_KO.md'):
        assert (source/name).read_bytes()==(replay/name).read_bytes()
    contextual=rows[-46:]
    assert all(row['seed']=='UNKNOWN' for row in contextual[:18])
    assert 'None' not in (source/'PPT_TABLES_KO.md').read_text(encoding='utf-8')
    assert all(row['K']=='UNKNOWN' and row['accuracy_binding_scope']=='UNKNOWN' for row in contextual[:42])
    assert all(row['process_RSS_peak_bytes']=='UNKNOWN' for row in contextual[42:])
    assert all(row['p90_ms_per_origin']=='0.14' and row['K']=='0' for row in contextual[42:])


def test_claim_table_distinguishes_partial_e_coverage_from_p_execution_and_repro_evidence(tmp_path):
    m=impl();ctx=public_fixture(tmp_path);_,state=m.collect_public(ctx);tables=finding_tables()
    pca=next(row for row in tables['RUNS.csv'] if row['method']=='B2_PCA' and row['cohort']=='pems32')
    dca=next(row for row in tables['RUNS.csv'] if row['method']=='B2_DCA' and row['cohort']=='pems32')
    tables['RUNS.csv'] += [dict(pca,row_id='pems_pca_seed2',run_id='pems_pca_seed2',seed=92602),
                          dict(dca,row_id='pems_dca_seed2',run_id='pems_dca_seed2',seed=92602,MSE=.9,MAE=.2)]
    tables['COMPARISONS.csv']=m._comparisons(tables['RUNS.csv']);tables['CHANNEL_HORIZON_STATS.csv']=[]
    m.write_documents(tables,state,tmp_path)
    claims={row['claim']:row for row in csv_rows(tmp_path/'PPT_T4_CLAIMS.csv')}
    assert claims['E 초기화 기여']['status']=='부분'
    e=claims['E 초기화 기여']['evidence']
    assert 'pems32 2 seed' in e and '일관된 감소 없음' in e
    assert 'jena' in e and 'NOT_CONVERGED' in e and 'UNKNOWN' in e
    assert claims['P 분해 기여']['status']=='대응 실행 완료·우위 미입증'
    assert claims['재현성']['status']=='별도 증거 확인'
    assert all(name in claims['재현성']['evidence'] for name in
               ('TEST_VALIDATION.json','VALIDATION_REPORT.json','PUBLIC_REPLAY_VERIFICATION.json','VISUAL_INSPECTION.json'))
