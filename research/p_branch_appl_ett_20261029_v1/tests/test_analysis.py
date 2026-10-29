"""CPU-only synthetic tests: no campaign predictions or protected-T values are read."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from research.p_branch_appl_ett_20261029_v1 import package, postprocess as analysis, run_one, runtime


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
    return runtime.sha(path)


def jobs():
    return [dict(job_id=f'PBA_{method}_{dataset}_K{runtime.KS[dataset]}_s{seed}',
                 method_id=method, dataset=dataset, K=runtime.KS[dataset], seed=seed, extended=False)
            for seed in analysis.SEEDS for dataset in analysis.DATASETS for method in analysis.METHODS]


def row(dataset, method, seed, *, mse=1., selected=.25):
    return dict(dataset=dataset, method_id=method, seed=seed, K=runtime.KS[dataset],
                job_id=f'fixture_{method}_{dataset}_{seed}', reused=method in ('OURS', 'A1', 'A2'),
                extended=False, selected_epoch=selected, training_cap=3., cap_selected=selected == 3.,
                trainable_parameters=runtime.EXPECTED_PARAMETERS[dataset].get(method, 123),
                normalization_fit_end=runtime.FIT_ENDS[dataset][1], MSE=mse, MAE=1.)


def references():
    out=[]
    for dataset in analysis.DATASETS:
        for method in ('OURS', 'A1', 'A2'):
            mse=1. if method == 'OURS' else 1.+analysis.REFERENCE_EFFECTS[dataset][method]/100.
            for seed in analysis.SEEDS:
                out.append(row(dataset, method, seed, mse=mse,
                               selected=3. if dataset == 'ETTh2_safe17400' and method == 'A1' else .25))
    return out


def new_family(dataset='appliances', method='P_TIME_ONLY', n=10, cap=0, mse=1.1):
    return [row(dataset, method, seed, mse=mse, selected=3. if i < cap else .25)
            for i, seed in enumerate(analysis.SEEDS[:n])]


def lookup(values, dataset, method):
    return next(value for value in values if (value['dataset'], value['method_id']) == (dataset, method))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'REPO', tmp_path)
    monkeypatch.setattr(runtime, 'REPORT', tmp_path/package.REPORT)
    monkeypatch.setattr(runtime, 'CACHE', tmp_path/'.cache'/runtime.NAME)
    runtime.REPORT.mkdir(parents=True)
    runtime.CACHE.mkdir(parents=True)
    return tmp_path


def test_plan_exact_order_and_empty_extensions():
    plan=dict(jobs=jobs(), extension_jobs=[], optional_jobs=[])
    assert len(analysis.validate_plan(plan)) == 60
    bad=dict(plan, jobs=plan['jobs'][::-1])
    with pytest.raises(ValueError, match='order'): analysis.validate_plan(bad)
    with pytest.raises(ValueError, match='extension'): analysis.validate_plan(dict(plan, extension_jobs=[plan['jobs'][0]]))


@pytest.mark.parametrize('n', [0, 1, 9, 10])
def test_partial_families_stay_out_of_primary(n):
    rows, summary, contrasts, paired=analysis.analyze_rows(references(), new_family(n=n))
    group=lookup(summary, 'appliances', 'P_TIME_ONLY')
    contrast=lookup(contrasts, 'appliances', 'P_TIME_ONLY')
    assert len(rows) == 90+n and len(summary) == 15 and len(contrasts) == 12
    assert group['n'] == n and group['primary'] is (n == 10)
    assert contrast['primary'] is (n == 10)
    assert (contrast['effect_pct'] is not None) is (n == 10)
    assert all(r['primary'] is (n == 10) for r in paired if r['method_id'] == 'P_TIME_ONLY')
    text=analysis.slide_numbers(summary, contrasts)
    assert 'Jena unchanged: Time branch only +0.2%' in text and 'Channel branch only +8.9%' in text
    assert text.count('INCOMPLETE') >= (0 if n == 10 else 1)
    analysis.check_reference_effects(contrasts)


def test_mixed_family_cohorts_and_duplicate_seed():
    new=new_family(n=10)+new_family(method='P_CHANNEL_ONLY', n=9)
    rows, summary, contrasts, _=analysis.analyze_rows(references(), new)
    assert lookup(summary, 'appliances', 'P_TIME_ONLY')['primary'] is True
    assert lookup(summary, 'appliances', 'P_CHANNEL_ONLY')['primary'] is False
    assert sum(r['primary'] and not r['reused'] for r in rows) == 10
    assert lookup(contrasts, 'appliances', 'P_CHANNEL_ONLY')['effect_pct'] is None
    with pytest.raises(ValueError, match='Duplicate'): analysis.analyze_rows(references(), new+[new[0]])
    broken=references();broken[0]['seed']=92602
    with pytest.raises(ValueError, match='coverage'): analysis.analyze_rows(broken, [])


@pytest.mark.parametrize('cap, expected', [(4, False), (5, True)])
def test_cap_threshold_and_reused_no_p_propagation(cap, expected):
    _, summary, contrasts, _=analysis.analyze_rows(references(), new_family(cap=cap))
    assert lookup(summary, 'appliances', 'P_TIME_ONLY')['provisional'] is expected
    assert lookup(contrasts, 'appliances', 'P_TIME_ONLY')['provisional'] is expected
    no_p=lookup(contrasts, 'ETTh2_safe17400', 'A1')
    assert no_p['provisional'] is True and no_p['cap_selected_count'] == 10
    assert no_p['descriptive_decision'] == 'VARIANT_LOWER_PROVISIONAL'


def test_iqr_rule_and_selection_schema():
    new=new_family(mse=1.1)
    _, _, contrasts, _=analysis.analyze_rows(references(), new)
    result=lookup(contrasts, 'appliances', 'P_TIME_ONLY')
    assert result['descriptive_decision'] == 'FULL_P_LOWER'
    assert result['effect_pct'] == pytest.approx(10.)
    assert result['paired_delta_median'] == pytest.approx(.1)
    for field, value in [('K', 99), ('seed',92601.9), ('normalization_fit_end', 1), ('selected_epoch', 10.), ('extended', True)]:
        broken=dict(new[0], **{field:value})
        with pytest.raises(ValueError): analysis.normalized_row(broken)


def archive_row(tmp_path, dataset='ETTh1', transform=None):
    shape=runtime.V_SHAPES[dataset];channels=shape[2]
    values=dict(pred_normalized=np.ones(shape, np.float32), target_normalized=np.zeros(shape, np.float32),
                observed=np.ones(shape, bool), origins=np.arange(analysis.V_STARTS[dataset], runtime.T_STARTS[dataset]-48+1, 48),
                channel_ids=np.asarray([f'channel{i}' for i in range(channels)]),
                mean=np.zeros(channels, np.float32), std=np.ones(channels, np.float32))
    if transform: transform(values)
    path=tmp_path/(dataset+'.npz')
    np.savez_compressed(path, **values)
    return dict(row(dataset, 'P_TIME_ONLY', 92601), prediction_path=str(path), prediction_sha256=runtime.sha(path))


@pytest.mark.parametrize('dataset', analysis.DATASETS)
def test_saved_archive_dataset_shapes_fp64_and_norm(tmp_path, dataset):
    item=archive_row(tmp_path, dataset)
    canonical, check=analysis.verify_arrays(item)
    assert check['status'] == 'PASS' and check['MSE'] == check['MAE'] == 1.
    assert check['shape'] == list(runtime.V_SHAPES[dataset])
    assert check['T_numeric_values_parsed'] == 0
    analysis.verify_arrays(item, canonical)
    with pytest.raises(ValueError, match='Cross-dataset'):
        analysis.verify_arrays(item, dict(canonical, _dataset='other'))


def test_archive_header_rejection_precedes_numeric_read(tmp_path, monkeypatch):
    item=archive_row(tmp_path, transform=lambda v: v.update(pred_normalized=np.ones((37,48,7), np.float32)))
    def forbidden(*args, **kwargs): raise AssertionError('Numeric read happened before shape rejection')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='header'): analysis.verify_arrays(item)


@pytest.mark.parametrize('change, message', [
    (lambda v: v['origins'].__setitem__(-1, 20000), 'origin'),
    (lambda v: v.pop('mean'), 'Missing'),
    (lambda v: v['std'].__setitem__(0, 0.), 'positive'),
    (lambda v: v.update(observed=v['observed'].astype(np.float32)), 'boolean'),
    (lambda v: v['pred_normalized'].__setitem__((0,0,0), np.nan), 'nonfinite'),
])
def test_invalid_archive_rejected(tmp_path, change, message):
    item=archive_row(tmp_path, transform=change)
    with pytest.raises(ValueError, match=message): analysis.verify_arrays(item)


def test_hash_scaler_mask_metric_mismatch(tmp_path):
    item=archive_row(tmp_path)
    canonical, _=analysis.verify_arrays(item)
    with pytest.raises(ValueError, match='changed'): analysis.verify_arrays(dict(item, prediction_sha256='bad'))
    with pytest.raises(ValueError, match='mismatch'): analysis.verify_arrays(dict(item, MSE=2.))
    for key in ('mean', 'std', 'target_normalized', 'observed', 'channel_ids', 'origins'):
        other=dict(canonical);other[key]=canonical[key].copy()
        value=other[key]
        if value.dtype.kind in 'SU': value.flat[0]='changed'
        elif value.dtype == bool: value.flat[0]=False
        else: value.flat[0]+=1
        with pytest.raises(ValueError, match='cohort/scaler'): analysis.verify_arrays(item, other)


def test_partial_observed_channel_macro_metric(tmp_path):
    def change(v):
        v['observed'][1:, :, 0]=False
        v['pred_normalized'][..., 0]=2.
    item=archive_row(tmp_path, transform=change)
    item.update(MSE=10./7., MAE=8./7.)
    _, check=analysis.verify_arrays(item)
    assert check['MSE'] == pytest.approx(10./7.)


def complete_evidence(job, monkeypatch, attempt=1):
    monkeypatch.setattr(run_one, 'source_hashes', lambda: {'fixture.py':'abc'})
    monkeypatch.setattr(runtime, 'assert_protected_T', lambda result: None)
    registration=dict(seal_sha256='fixture', registered_at_utc='2026-10-29T00:00:00+00:00', files={})
    spec=run_one.build_spec(job)
    root=runtime.REPORT/'jobs'/job['job_id'];public=root/f'attempt_{attempt:03d}'
    contract=dict(source_sha='source', config_sha='config', data_sha='data', model_sha=runtime.F_SHA)
    curves=[dict(epoch=e, astop_mse=m) for e, m in [(0.,4.),(.25,1.),(1.,2.),(3.,3.)]]
    search=dict(summary=dict(status='SEGMENT_COMPLETE', selected_epoch=.25, selected_astop_mse=1.), curves=curves)
    selection=dict(status='SEALED', job=job, V_used=False, selection_split='A_stop', selected_epoch=.25,
                   selected_astop_mse=1., spec=spec, contract=contract, curves=curves)
    refit=dict(status='SEGMENT_COMPLETE', updates=4)
    bundle=runtime.CACHE/(job['job_id']+'.pt');bundle.write_bytes(b'synthetic opaque bundle')
    pred=runtime.CACHE/(job['job_id']+'.npz');pred.write_bytes(b'synthetic opaque archive')
    selected=public/'SELECTION.json';selection_sha=save(selected, selection)
    result_row=dict(row(job['dataset'], job['method_id'], job['seed']), **job, reused=False,
                    prediction_path=str(pred), prediction_sha256=runtime.sha(pred))
    authorization=dict(kind='BASE', attempt=attempt)
    if attempt > 2:
        auth_path=runtime.REPORT/'RECOVERY_AUTHORIZATION_001.json'
        save(auth_path, dict(status='APPROVED_TECHNICAL_RECOVERY', job_id=job['job_id'], allowed_attempts=[3,4],
                            registration_seal_sha256=runtime.sha(runtime.REPORT/'REGISTRATION_SEAL.json'),
                            scientific_source_hashes=run_one.source_hashes(), created_utc='2026-10-29T00:00:00+00:00'))
        authorization=run_one.validate_attempt(job, attempt, started_unix=1800000000.)
    result=dict(status='COMPLETE', job=job, registration=registration, spec=spec,
                selection_sealed_before_V=True, attempt=attempt, attempt_authorization=authorization, contract=contract,
                selection_path=str(selected), selection_sha256=selection_sha, search_summary=search['summary'], refit_summary=refit,
                row=result_row, bundle_path=str(bundle), bundle_sha256=runtime.sha(bundle))
    save(root/'RESULT.json', result)
    save(root/f'WORKER_RECEIPT_attempt{attempt}.json', dict(status='COMPLETE', exit_code=0, job_id=job['job_id'], attempt=attempt,
                                                        started_unix=1800000000.))
    save(public/'START.json', dict(job=job, attempt=attempt, registration=registration, spec=spec))
    save(public/'SOURCE_CONTRACT.json', dict(contract, sources={'fixture.py':'abc'}, spec=spec))
    save(public/'SEARCH_RESULT.json', search)
    save(public/'REFIT_SUMMARY.json', refit)
    save(public/'RECEIPT.json', dict(status='COMPLETE', job=job, attempt=attempt, registration=registration,
                                   training_attempted=True, training_executed=True, T_numeric_values_parsed=0,
                                   preserved_before={'changed':[]}, preserved_after={'changed':[]}))
    save(public/'OPENED_FILES.json', dict(T_numeric_values_parsed=0))
    return result, registration


def test_completion_zero_one_and_missing_result(isolated, monkeypatch):
    plan=dict(jobs=jobs(), extension_jobs=[])
    completion=dict(status='TRAINING_INCOMPLETE', active=None, completed_count=0,
                    run_statuses={j['job_id']:'READY' for j in jobs()})
    monkeypatch.setattr(run_one, 'validate_registration', lambda job: {})
    assert analysis.verify_completion_evidence(plan, completion)[1] == 0
    completion['run_statuses'][jobs()[0]['job_id']]='COMPLETE';completion['completed_count']=1
    with pytest.raises(ValueError, match='no result'): analysis.verify_completion_evidence(plan, completion)
    result, registration=complete_evidence(jobs()[0], monkeypatch)
    monkeypatch.setattr(run_one, 'validate_registration', lambda job: registration)
    results, count, verified=analysis.verify_completion_evidence(plan, completion)
    assert count == 1 and len(results) == 1 and len(verified) == 9
    selection=Path(result['selection_path']);value=runtime.read(selection);value['curves'][0]['astop_mse']=99.
    save(selection, value)
    with pytest.raises(ValueError, match='seal'): analysis.verify_completion_evidence(plan, completion)


def test_recovery_attempt_evidence_binds_prior_authorization(isolated, monkeypatch):
    save(runtime.REPORT/'REGISTRATION_SEAL.json', dict(files={}, created_utc='2026-10-29T00:00:00+00:00'))
    job=jobs()[0];result, registration=complete_evidence(job, monkeypatch, attempt=3)
    completion=dict(run_statuses={job['job_id']:'COMPLETE'})
    _, verified=analysis.verify_job_evidence(job, registration, completion)
    assert any(name.endswith('RECOVERY_AUTHORIZATION_001.json') for name in verified)
    auth=runtime.REPORT/'RECOVERY_AUTHORIZATION_001.json';value=runtime.read(auth)
    value['created_utc']='2030-01-01T00:00:00+00:00';save(auth, value)
    with pytest.raises(ValueError, match='before attempt'): analysis.verify_job_evidence(job, registration, completion)


def rejected_evidence(job, monkeypatch, status='SCIENTIFIC_SUSPEND'):
    _, registration=complete_evidence(job, monkeypatch)
    root=runtime.REPORT/'jobs'/job['job_id'];original=root/'RESULT.json'
    digest=runtime.sha(original);destination=root/'rejected'/f'RESULT_attempt001_{digest[:12]}.json'
    destination.parent.mkdir();original.rename(destination)
    worker=root/'WORKER_RECEIPT_attempt1.json'
    save(worker,dict(job_id=job['job_id'],attempt=1,status=status,exit_code=20))
    value=dict(status='REJECTED_SCIENTIFIC_EVIDENCE',job=job,run_status=status,attempt=1,
        trigger='RESULT_WRITTEN_THEN_TERMINAL_VALIDATION_REJECTED',
        original_result_path=original.relative_to(runtime.REPO).as_posix(),
        rejected_result_path=destination.relative_to(runtime.REPO).as_posix(),rejected_result_sha256=digest,
        reason='Synthetic final worker receipt failed after result creation',created_utc='2026-10-29T00:00:00+00:00',
        worker_receipt_path=worker.relative_to(runtime.REPO).as_posix(),worker_receipt_sha256=runtime.sha(worker))
    save(root/'RESULT_REJECTION.json',value)
    return value,registration


@pytest.mark.parametrize('status',['SCIENTIFIC_SUSPEND','UNKNOWN_SUSPEND','USER_CANCELLED'])
def test_hash_bound_rejection_excludes_invalid_result_without_scoring(isolated,monkeypatch,status):
    job=jobs()[0];receipt,registration=rejected_evidence(job,monkeypatch,status)
    monkeypatch.setattr(run_one,'validate_registration',lambda job:registration)
    completion=dict(status=status,active=None,completed_count=0,
                    run_statuses={j['job_id']:status if j==job else 'NOT_RUN_'+status for j in jobs()})
    results,count,verified=analysis.verify_completion_evidence(dict(jobs=jobs(),extension_jobs=[]),completion)
    assert results == {} and count == 0 and len(verified) == 3
    assert receipt['rejected_result_path'] in verified
    record,_=analysis.verify_rejected_result(job,status)
    assert record['excluded_from_numeric_analysis'] is True
    assert record['reason'] == receipt['reason']


@pytest.mark.parametrize('field,value',[
    ('rejected_result_sha256','a'*64),('trigger','SKIP_ANY_RESULT'),
    ('original_result_path','reports/elsewhere/RESULT.json'),('run_status','COMPLETE'),
    ('reason',''),('worker_receipt_sha256','bad'),('created_utc','2030-01-01T00:00:00+00:00')])
def test_rejection_requires_exact_preserved_evidence(isolated,monkeypatch,field,value):
    job=jobs()[0];receipt,_=rejected_evidence(job,monkeypatch)
    receipt[field]=value
    save(runtime.REPORT/'jobs'/job['job_id']/'RESULT_REJECTION.json',receipt)
    with pytest.raises(ValueError):analysis.verify_rejected_result(job,'SCIENTIFIC_SUSPEND')


@pytest.mark.parametrize('n', [0, 1, 9, 10, 60])
def test_reporting_zero_partial_full_paths(isolated, monkeypatch, n):
    plan=dict(jobs=jobs(), extension_jobs=[])
    selected=jobs()[:n]
    completion=dict(status='TRAINING_COMPLETE' if n == 60 else 'TRAINING_INCOMPLETE', active=None,
                    completed_count=n, run_statuses={j['job_id']:'COMPLETE' if j in selected else 'READY' for j in jobs()})
    for name, value in [('EXECUTION_COMPLETION.json',completion),('RUN_PLAN.json',plan),
        ('REFERENCE_RUNS.json',dict(rows=references(), raw_rows=[dict(dataset=d,MSE=1.5) for d in analysis.DATASETS]))]:
        save(runtime.REPORT/name, value)
    for name in ['REGISTRATION_SEAL.json','PREREGISTRATION.json','PRESERVATION_BASELINE.json']:save(runtime.REPORT/name,{})
    (runtime.REPORT/'PREREGISTRATION.md').write_text('synthetic fixture only',encoding='utf8')
    results={j['job_id']:dict(row=dict(row(j['dataset'],j['method_id'],j['seed']),**j,reused=False)) for j in selected}
    monkeypatch.setattr(analysis,'verify_completion_evidence',lambda p,c:(results,n,{}))
    monkeypatch.setattr(runtime,'guard_preserved',lambda **kwargs:dict(changed=[]))
    monkeypatch.setattr(analysis,'verify_arrays',lambda r,cohort=None:({'_dataset':r['dataset']},dict(status='PASS',job_id=r['job_id'],T_numeric_values_parsed=0)))
    analysis.main()
    checked=runtime.read(runtime.REPORT/'ANALYSIS_VALIDATION.json')
    assert checked['completed_count'] == n and checked['checked_prediction_archives'] == 90+n
    assert len(list(csv.DictReader((runtime.REPORT/'JOB_STATUS.csv').open(encoding='utf-8-sig')))) == 60
    assert len(list(csv.DictReader((runtime.REPORT/'ALL_NEW_RUNS.csv').open(encoding='utf-8-sig')))) == n
    assert (runtime.REPORT/'FIG_P_BRANCHES_APPL_ETT.png').stat().st_size > 1000
    assert not list(runtime.REPORT.glob('*.pdf')) and not list(runtime.REPORT.glob('*.pptx'))


def package_fixture(repo, n):
    report=repo/package.REPORT;source=repo/package.SOURCE;source.mkdir(parents=True)
    own=source/'package.py';own.write_bytes(Path(package.__file__).read_bytes())
    plan=dict(jobs=jobs(), extension_jobs=[])
    selected=jobs()[:n];statuses={j['job_id']:'COMPLETE' if j in selected else 'FAILED_TERMINAL' for j in jobs()}
    primary=sum(10 for d in runtime.DATASETS for m in analysis.METHODS
                if all(statuses[j['job_id']]=='COMPLETE' for j in jobs() if (j['dataset'],j['method_id']) == (d,m)))
    completion=dict(status='TRAINING_COMPLETE' if n == 60 else 'TRAINING_INCOMPLETE',active=None,run_statuses=statuses)
    for name in package.REQUIRED:
        (report/name).write_text('synthetic public fixture',encoding='utf8')
    save(report/'RUN_PLAN.json',plan)
    completed_sha=save(report/'EXECUTION_COMPLETION.json',completion)
    analysis_sha=save(report/'ANALYSIS_VALIDATION.json',dict(status='PASS',integrity_status='PASS',T_numeric_values_parsed=0,
                completed_count=n,base_complete=n,primary_count=primary,
                verified_files={package.REPORT+'EXECUTION_COMPLETION.json':completed_sha}))
    save(report/'PREREGISTRATION.json',dict(registered_sources={package.SOURCE+'package.py':runtime.sha(own)}))
    save(report/'REGISTRATION_SEAL.json',dict(files={'PREREGISTRATION.json':runtime.sha(report/'PREREGISTRATION.json')}))
    gate=dict(status='PASS',scope='EXPERIMENT_RESULTS_ONLY',analysis_sha256=analysis_sha,completion_sha256=completed_sha,
              allow_partial=True,partial_reason='Synthetic incomplete technical-failure fixture',
              reviewed_files={package.REPORT+name:runtime.sha(report/name) for name in package.REVIEWED},
              figures_visual_review={package.FIGURE:dict(status='PASS',sha256=runtime.sha(report/package.FIGURE))})
    save(report/'ROOT_RESULTS_GATE.json',gate)
    return gate


@pytest.mark.parametrize('n',[0,1,9,10,60])
def test_results_only_package_zero_partial_complete_exact_members(isolated,n):
    package_fixture(isolated,n)
    for name in ['should_not_publish.npz','weights.pt','deck.pptx','slides.pdf','private_host.json','secret.json','credentials.json','draft_summary.md']:
        (runtime.REPORT/name).write_bytes(b'synthetic private fixture')
    cache=runtime.REPORT/'.cache';cache.mkdir();(cache/'private.json').write_text('{}',encoding='utf8')
    receipt=package.package_results(isolated)
    assert receipt['completed_count'] == n and receipt['partial'] is (n < 60)
    assert receipt['production_readback']['exact_members'] == 'PASS'
    zipped=runtime.REPORT/package.ZIP_NAME
    first_hash=runtime.sha(zipped)
    assert package.package_results(isolated) == receipt and runtime.sha(zipped) == first_hash
    with zipfile.ZipFile(zipped) as archive:
        assert not any(name.endswith(('.pptx','.pdf','.npz','.pt')) or '.cache' in name or any(x in name for x in ('private_host','secret.json','credentials.json','draft_summary')) for name in archive.namelist())
        manifest=archive.read(package.REPORT+package.MANIFEST)
    assert package.verify_zip(zipped,manifest)['all_member_sha256_and_sizes'] == 'PASS'
    with zipfile.ZipFile(zipped,'a') as archive:archive.writestr('unexpected.json','{}')
    with pytest.raises(ValueError,match='exact-member'):package.verify_zip(zipped,manifest)


def test_root_gate_partial_and_changed_bound_evidence_rejected(isolated):
    gate=package_fixture(isolated,0)
    gate['allow_partial']=False;save(runtime.REPORT/'ROOT_RESULTS_GATE.json',gate)
    with pytest.raises(ValueError,match='Partial/zero'):package.prepare_results(isolated)
    gate['allow_partial']=True;save(runtime.REPORT/'ROOT_RESULTS_GATE.json',gate)
    package.prepare_results(isolated)
    (runtime.REPORT/'SUMMARY.csv').write_text('changed',encoding='utf8')
    with pytest.raises(ValueError,match='changed'):package.prepare_results(isolated)


@pytest.mark.parametrize('status',['SCIENTIFIC_SUSPEND','UNKNOWN_SUSPEND','USER_CANCELLED'])
def test_suspended_zero_run_reporting_and_packaging(isolated, monkeypatch, status):
    gate=package_fixture(isolated,0)
    completion=runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json');completion['status']=status
    digest=save(runtime.REPORT/'EXECUTION_COMPLETION.json',completion)
    value=runtime.read(runtime.REPORT/'ANALYSIS_VALIDATION.json')
    value['verified_files'][package.REPORT+'EXECUTION_COMPLETION.json']=digest
    gate['completion_sha256']=digest
    gate['analysis_sha256']=save(runtime.REPORT/'ANALYSIS_VALIDATION.json',value)
    for name in ['EXECUTION_COMPLETION.json','ANALYSIS_VALIDATION.json']:
        gate['reviewed_files'][package.REPORT+name]=runtime.sha(runtime.REPORT/name)
    save(runtime.REPORT/'ROOT_RESULTS_GATE.json',gate)
    monkeypatch.setattr(run_one,'validate_registration',lambda job:{})
    assert analysis.verify_completion_evidence(dict(jobs=jobs(),extension_jobs=[]),completion)[1] == 0
    receipt=package.package_results(isolated)
    assert receipt['completion_status'] == status and receipt['partial'] is True


def test_zip_creation_never_overwrites_reviewed_old_bundle(isolated):
    gate=package_fixture(isolated,0);package.package_results(isolated)
    target=runtime.REPORT/package.ZIP_NAME;before=target.read_bytes()
    name=package.REPORT+'SUMMARY.csv';path=isolated/name;path.write_text('new reviewed summary',encoding='utf8')
    gate['reviewed_files'][name]=runtime.sha(path);save(runtime.REPORT/'ROOT_RESULTS_GATE.json',gate)
    with pytest.raises(ValueError,match='payload differs|metadata differs'):package.package_results(isolated)
    assert target.read_bytes() == before
