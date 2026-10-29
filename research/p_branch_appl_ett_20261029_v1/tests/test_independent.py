"""Synthetic, CPU-only independent-audit regressions; no real campaign arrays."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from research.p_branch_appl_ett_20261029_v1 import independent_result_check as audit


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_archive(tmp_path, **changes):
    dataset = 'ETTh1'
    _, channels, count, start, end, _ = audit.AXES[dataset]
    shape = (count, 48, channels)
    arrays = dict(pred_normalized=np.full(shape, 2., np.float32), target_normalized=np.zeros(shape, np.float32),
        observed=np.ones(shape, bool), origins=np.arange(start, end-48+1, 48, dtype=np.int64),
        channel_ids=np.array([f'c{x}' for x in range(channels)]), mean=np.zeros(channels, np.float32), std=np.ones(channels, np.float32))
    arrays.update(changes)
    path = tmp_path / 'synthetic.npz'
    np.savez_compressed(path, **arrays)
    return dict(dataset=dataset, job_id='synthetic', prediction_path=str(path), prediction_sha256=audit.sha(path), MSE=4., MAE=2.)


def result_rows(include_new=True):
    rows = []
    for dataset in audit.DATASETS:
        for method in audit.ORDER:
            if not include_new and method in audit.METHODS:
                continue
            for index, seed in enumerate(audit.SEEDS):
                rows.append(dict(dataset=dataset, method_id=method, seed=seed,
                    MSE=1.+index*.01+(.2 if method=='P_TIME_ONLY' else -.2 if method=='P_CHANNEL_ONLY' else 0),
                    MAE=.5+index*.01, selected_epoch=.25, trainable_parameters=17))
    return rows


def test_quantile_linear_interpolation_and_not_paired_median_of_medians():
    assert audit.quantile([0, 1, 9, 10], .25) == .75
    assert audit.quantile([0, 1, 9, 10], .5) == 5.
    assert audit.quantile([0, 1, 9, 10], .75) == 9.25


def test_archive_recomputes_fp64_and_detects_wrong_metric(tmp_path):
    row = fixture_archive(tmp_path)
    _, check = audit.check_archive(row)
    assert (check['MSE'], check['MAE']) == (4., 2.)
    with pytest.raises(ValueError, match='FP64'):
        audit.check_archive(dict(row, MSE=4.001))


def test_score_is_channel_macro_not_observation_weighted(tmp_path):
    shape=(36,48,7)
    mask=np.zeros(shape,bool);mask[0,0,:]=True;mask[:,:,0]=True
    pred=np.ones(shape,np.float32);pred[:,:,0]=3.
    row=fixture_archive(tmp_path, observed=mask, pred_normalized=pred)
    row.update(MSE=15/7, MAE=9/7)
    _,check=audit.check_archive(row)
    assert check['MSE']==15/7


@pytest.mark.parametrize('changes',[
    {'pred_normalized':np.zeros((37,48,7),np.float32)},
    {'mean':np.zeros((8,),np.float32)},
    {'observed':np.ones((36,48,7),np.uint8)},
    {'target_normalized':np.empty((36,48,7),object)},
])
def test_bad_headers_rejected_before_any_numpy_numeric_load(tmp_path,monkeypatch,changes):
    row=fixture_archive(tmp_path,**changes)
    monkeypatch.setattr(audit.np,'load',lambda *a,**k:pytest.fail('Numeric loading preceded header rejection'))
    with pytest.raises(ValueError,match='header|dtype'):
        audit.check_archive(row)


def test_origin_boundary_rejected_before_prediction_materialization(tmp_path,monkeypatch):
    row=fixture_archive(tmp_path,origins=np.arange(12194+48,13936-48+1+48,48,dtype=np.int64))
    original=np.load
    class OriginsOnly:
        def __enter__(self):
            self.inner=original(row['prediction_path'],allow_pickle=False);return self
        def __exit__(self,*args):self.inner.close()
        def __getitem__(self,key):
            assert key=='origins','Protected boundary must be checked before floating arrays'
            return self.inner[key]
    monkeypatch.setattr(audit.np,'load',lambda *a,**k:OriginsOnly())
    with pytest.raises(ValueError,match='protected T'):
        audit.check_archive(row)


def test_missing_normalization_is_not_excused(tmp_path):
    row=fixture_archive(tmp_path)
    with np.load(row['prediction_path'],allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files if k!='mean'}
    np.savez_compressed(row['prediction_path'],**arrays);row['prediction_sha256']=audit.sha(row['prediction_path'])
    with pytest.raises(ValueError,match='metadata'):
        audit.check_archive(row)


def test_cohort_and_scaler_exact_match_and_crossdata_rejected(tmp_path):
    row=fixture_archive(tmp_path);cohort,_=audit.check_archive(row)
    changed=dict(cohort,mean=np.full(7,.00001,np.float32))
    with pytest.raises(ValueError,match='scaler'):
        audit.check_archive(row,changed)
    with pytest.raises(ValueError,match='Cross-dataset'):
        audit.check_archive(row,dict(cohort,dataset='appliances'))


@pytest.mark.parametrize('changes',[
    {'observed':np.zeros((36,48,7),bool)},
    {'pred_normalized':np.full((36,48,7),np.nan,np.float32)},
    {'std':np.zeros(7,np.float32)},
])
def test_invalid_score_inputs_fail(tmp_path,changes):
    with pytest.raises(ValueError,match='All-false|Nonfinite|normalization'):
        audit.check_archive(fixture_archive(tmp_path,**changes))


@pytest.mark.parametrize('full_cap,variant_cap,provisional',[(4,4,False),(4,5,True),(5,4,True)])
def test_cap_threshold_propagates_from_either_party(full_cap,variant_cap,provisional):
    rows=result_rows()
    for row in rows:
        count=full_cap if row['method_id']=='OURS' else variant_cap
        if row['seed']-92601<count:row['selected_epoch']=3.
    _,contrasts,_=audit.summarize(rows)
    contrast=next(r for r in contrasts if r['dataset']=='ETTh1' and r['method_id']=='P_CHANNEL_ONLY')
    assert contrast['provisional'] is provisional
    assert contrast['effect_pct']<0
    assert contrast['descriptive_decision']==('VARIANT_LOWER_PROVISIONAL' if provisional else 'VARIANT_LOWER')


@pytest.mark.parametrize('n',[0,1,9,10])
def test_partial_family_never_becomes_primary_or_zero_effect(n):
    rows=[r for r in result_rows() if r['method_id']!='P_CHANNEL_ONLY' or r['seed'] in audit.SEEDS[:n]]
    summaries,contrasts,paired=audit.summarize(rows)
    contrast=next(r for r in contrasts if r['dataset']=='ETTh1' and r['method_id']=='P_CHANNEL_ONLY')
    assert contrast['primary'] is (n==10)
    assert (contrast['effect_pct'] is None) is (n<10)
    assert len([r for r in paired if r['dataset']=='ETTh1' and r['method_id']=='P_CHANNEL_ONLY'])==n
    assert all(not r['primary'] for r in rows if r['method_id']=='P_CHANNEL_ONLY') if n<10 else True


def test_duplicate_result_and_finite_metric_mismatch_rejected(tmp_path):
    rows=result_rows()
    with pytest.raises(ValueError,match='Duplicate'):
        audit.summarize(rows+[rows[0]])
    path=tmp_path/'CONTRASTS.csv';expected=[dict(dataset='ETTh1',method_id='P_CHANNEL_ONLY',effect_pct=-.7,primary=True)]
    write_csv(path,[dict(expected[0],effect_pct=.7)])
    with pytest.raises(ValueError,match='CSV mismatch'):
        audit.compare_csv(path,expected,['dataset','method_id'],['effect_pct','primary'])
    write_csv(path,[dict(expected[0],primary=False)])
    with pytest.raises(ValueError,match='CSV mismatch'):
        audit.compare_csv(path,expected,['dataset','method_id'],['effect_pct','primary'])


def test_empty_partial_csv_requires_blanks_and_keeps_signed_slide(tmp_path):
    path=tmp_path/'CONTRASTS.csv';expected=[dict(dataset='ETTh1',method_id='P_CHANNEL_ONLY',effect_pct=None)]
    write_csv(path,expected)
    audit.compare_csv(path,expected,['dataset','method_id'],['effect_pct'])
    write_csv(path,[dict(expected[0],effect_pct=0)])
    with pytest.raises(ValueError,match='CSV mismatch'):
        audit.compare_csv(path,expected,['dataset','method_id'],['effect_pct'])
    rows=result_rows();_,contrasts,_=audit.summarize(rows)
    slide=tmp_path/'SLIDE_NUMBERS.md'
    lines=[f"{r['method_id']} {r['dataset']} {r['effect_pct']:+.1f}% {r['descriptive_decision']} {r['cap_selected_count']}/10 {r['n']}" for r in contrasts if r['method_id'] in audit.METHODS]
    slide.write_text('\n'.join(lines),encoding='utf8');audit.check_slide(slide,contrasts)
    slide.write_text('\n'.join(lines).replace('-19.1%','+19.1%'),encoding='utf8')
    with pytest.raises(ValueError,match='signed'):
        audit.check_slide(slide,contrasts)


def test_live_execution_rejected_before_any_npz_or_other_input(tmp_path,monkeypatch):
    report=tmp_path/'reports'/audit.NAME;report.mkdir(parents=True)
    (report/'EXECUTION_COMPLETION.json').write_text(json.dumps(dict(status='RUNNING',active={})),encoding='utf8')
    monkeypatch.setattr(audit.np,'load',lambda *a,**k:pytest.fail('Live NPZ must not be opened'))
    with pytest.raises(ValueError,match='Terminal'):
        audit.audit(tmp_path)


def complete_chain(tmp_path):
    report=tmp_path/'reports'/audit.NAME;job=audit.expected_jobs()[0]
    root=report/'jobs'/job['job_id'];public=root/'attempt_001';public.mkdir(parents=True)
    cache=tmp_path/'.cache'/audit.NAME;cache.mkdir(parents=True)
    def save(path,value):
        path.write_text(json.dumps(value),encoding='utf8');return audit.sha(path)
    sources={f'research/{audit.NAME}/{name}':'test-'+name for name in ('models.py','runtime.py','run_one.py','controller.py')}
    protocol=dict(registered_sources=sources)
    registration=dict(seal_sha256='seal',registered_at_utc='2026-10-29T00:00:00+00:00',files={})
    spec=dict(run_id=job['job_id'],dataset=job['dataset'],method_id=job['method_id'],family='B2',latent_k=10,
        seed=92601,init_kind='PCA',revin=False,device='cuda',fold='primary',core_dtype='float32',lr=.001,
        effective_batch=16,microbatch=4,F_row_microbatch=128,checkpoint_inputs=False,epochs=[.25,1.,3.],
        new_scientific_source_hashes=sources,custom_source_sha256=sources[f'research/{audit.NAME}/models.py'])
    contract=dict(source_sha='source',config_sha='config',data_sha='data',model_sha=audit.F_SHA)
    curves=[dict(epoch=e,astop_mse=v) for e,v in [(0.,4.),(.25,1.),(1.,1.),(3.,3.)]]
    search=dict(summary=dict(status='SEGMENT_COMPLETE'),curves=curves)
    refit=dict(status='SEGMENT_COMPLETE')
    selection=dict(status='SEALED',job=job,V_used=False,selection_split='A_stop',spec=spec,contract=contract,
        curves=curves,selected_epoch=.25,selected_astop_mse=1.)
    selected=public/'SELECTION.json';selected_sha=save(selected,selection)
    bundle=cache/'bundle.pt';bundle.write_bytes(b'opaque synthetic checkpoint; never deserialized')
    row=dict(job,MSE=1.,MAE=.5,selected_epoch=.25,training_cap=3,trainable_parameters=35392,
        normalization_fit_end=11841,reused=False,prediction_path=str(cache/'v.npz'),prediction_sha256='opaque')
    result=dict(status='COMPLETE',job=job,registration=registration,attempt=1,attempt_authorization=dict(kind='BASE',attempt=1),
        spec=spec,contract=contract,task_access={'synthetic':'access'},selection_path=str(selected),selection_sha256=selected_sha,
        selection_sealed_before_V=True,search_summary=search['summary'],refit_summary=refit,row=row,
        bundle_path=str(bundle),bundle_sha256=audit.sha(bundle))
    save(root/'RESULT.json',result)
    save(root/'WORKER_RECEIPT_attempt1.json',dict(status='COMPLETE',exit_code=0,job_id=job['job_id'],attempt=1,started_unix=1800000000.))
    save(public/'START.json',dict(job=job,attempt=1,registration=registration,spec=spec))
    save(public/'SOURCE_CONTRACT.json',dict(contract,spec=spec,sources={'old_core.py':'source'}))
    save(public/'SEARCH_RESULT.json',search);save(public/'REFIT_SUMMARY.json',refit)
    save(public/'RECEIPT.json',dict(job=job,attempt=1,registration=registration,status='COMPLETE',training_attempted=True,
        training_executed=True,T_numeric_values_parsed=0,preserved_before={'changed':[]},preserved_after={'changed':[]}))
    save(public/'OPENED_FILES.json',dict(T_numeric_values_parsed=0))
    completion=dict(attempts={job['job_id']:[dict(job_id=job['job_id'],attempt=1,exit_code=0)]})
    registry={job['dataset']:dict(access=result['task_access'])}
    return report,job,registration,protocol,completion,registry,{}


def test_independent_chain_validates_tie_break_and_hashes_without_loading_bundle(tmp_path):
    args=complete_chain(tmp_path)
    row=audit.check_job(*args)
    assert row['selected_epoch']==.25
    path=args[0]/'jobs'/args[1]['job_id']/'attempt_001'/'SELECTION.json'
    value=audit.read(path);value['selected_epoch']=1.;path.write_text(json.dumps(value),encoding='utf8')
    with pytest.raises(ValueError,match='SHA'):
        audit.check_job(*args)


@pytest.mark.parametrize('target,changes,match',[
    ('RECEIPT.json',{'T_numeric_values_parsed':1},'receipt'),
    ('WORKER_RECEIPT_attempt1.json',{'exit_code':20},'worker receipt'),
    ('START.json',{'attempt':2},'identity'),
])
def test_chain_rejects_tainted_receipt_identity(tmp_path,target,changes,match):
    args=complete_chain(tmp_path);root=args[0]/'jobs'/args[1]['job_id']
    path=root/target if target.startswith('WORKER') else root/'attempt_001'/target
    value=audit.read(path);value.update(changes);path.write_text(json.dumps(value),encoding='utf8')
    with pytest.raises(ValueError,match=match):
        audit.check_job(*args)
