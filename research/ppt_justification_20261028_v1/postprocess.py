"""Validate saved V evidence and describe the registered five-seed contrasts."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from . import runtime, run_one


SEEDS = tuple(range(92601, 92606))
MANDATORY = (('NO_F_LINEAR', 'jena'), ('NO_F_LINEAR', 'appliances'), ('NO_ANCHOR', 'ETTh1'), ('NO_F_LINEAR_NO_P', 'jena'))
OPTIONAL = (('A5_LAG0', 'ETTh1'), ('NO_ANCHOR', 'appliances'))
ALL_FAMILIES = MANDATORY + OPTIONAL
CONTRAST_SPECS = (
    ('F_WITH_P_JENA','NO_F_LINEAR','OURS','jena'),
    ('F_WITH_P_APPLIANCES','NO_F_LINEAR','OURS','appliances'),
    ('ANCHOR_ETTH1','NO_ANCHOR','OURS','ETTh1'),
    ('ANCHOR_APPLIANCES','NO_ANCHOR','OURS','appliances'),
    ('LAG_ETTH1','A5_LAG0','OURS','ETTh1'),
    ('F_WITHOUT_P_JENA','NO_F_LINEAR_NO_P','A1','jena'),
    ('P_WITHOUT_F_JENA','NO_F_LINEAR_NO_P','NO_F_LINEAR','jena'),
    ('P_WITH_F_JENA','A1','OURS','jena'),
)
LABELS = {'OURS': 'Full P', 'A1': 'No P; frozen F', 'NO_F_LINEAR': 'P retained; F replaced by linear head',
          'NO_F_LINEAR_NO_P': 'No P; F replaced by linear head',
          'NO_ANCHOR': 'Last-observation path removed', 'A5_LAG0': 'Lag 0 only; anchor retained'}
LEGACY_WITHOUT_SCALERS = {'HISTORICAL_jena_K6_s92601', 'HISTORICAL_jena_K6_s92602'}
HISTORICAL_INPUTS = {
    'research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/RESULTS_v2.csv':
        '0cfd809a0345ac5a35a1135553786eb945cbe114833700108e132a7bb6266393',
    'reports/adapter_diagnostics_20261017_v2/QUESTIONS_AND_DECISIONS_KO.md':
        '68b6f9d6725f14e9efc20f0fa3cb612fa80efdebd35a1ab0380bb05d42b8c72b',
    'reports/adapter_diagnostics_20261017_v2/METHOD_CONTRACT.md':
        'b8ff9af34bbec5183729d7cb6649992e73ab582c3b4a9647857676fff136ce16',
}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing an unlabelled empty scientific table: '+str(path))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader(); writer.writerows(rows)
    temporary = path.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8-sig')
    temporary.replace(path)


def normalized_row(row):
    out = dict(row)
    for name in ('seed', 'K', 'trainable_parameters', 'normalization_fit_end'):
        out[name] = int(float(out[name]))
    for name in ('MSE', 'MAE', 'selected_epoch', 'training_cap'):
        out[name] = float(out[name])
    for name in ('extended', 'reused', 'provisional'):
        out[name] = str(out.get(name, False)).lower() == 'true'
    out['display_name'] = LABELS[out['method_id']]
    return out


def verify_arrays(row, cohort, contract):
    path = Path(row['prediction_path'])
    if runtime.sha(path) != row['prediction_sha256']:
        raise ValueError('Prediction archive changed: '+row['job_id'])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids')}
        for key in ('mean', 'std'):
            if key in archive.files:
                arrays[key] = archive[key]
            elif row['job_id'] not in LEGACY_WITHOUT_SCALERS:
                raise ValueError('Unexpected missing normalization metadata: '+row['job_id'])
    p, y, mask = arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'].astype(bool)
    if list(p.shape) != list(contract['shape']) or y.shape != p.shape or mask.shape != p.shape:
        raise ValueError('V shape differs from the registered reference cohort')
    if int(arrays['origins'].max()) + p.shape[1] > int(contract['T_start']):
        raise ValueError('V prediction would cross protected T')
    if cohort is not None:
        for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
            if key not in arrays or key not in cohort:
                continue
            if not np.array_equal(arrays[key], cohort[key], equal_nan=arrays[key].dtype.kind in 'fc'):
                raise ValueError('Reference cohort/scaler differs: '+key)
    difference = p.astype(np.float64) - y.astype(np.float64)
    if not np.isfinite(difference[mask]).all():
        raise ValueError('Nonfinite prediction or target')
    counts = mask.sum(axis=(0,1))
    if np.any(counts == 0):
        raise ValueError('A registered channel has no observed targets')
    observed_difference = np.where(mask, difference, 0.0)
    mse = float(np.mean((observed_difference**2).sum(axis=(0,1))/counts))
    mae = float(np.mean(np.abs(observed_difference).sum(axis=(0,1))/counts))
    if max(abs(mse-row['MSE']), abs(mae-row['MAE'])) > 1e-9:
        raise ValueError('Independent saved-array metric mismatch')
    return arrays, {'job_id':row['job_id'], 'dataset':row['dataset'], 'status':'PASS',
                   'sha256':row['prediction_sha256'], 'shape':list(p.shape), 'MSE':mse, 'MAE':mae,
                   'normalization_arrays_present':'mean' in arrays and 'std' in arrays,
                   'legacy_exception':row['job_id'] if row['job_id'] in LEGACY_WITHOUT_SCALERS else None}


def aggregate(rows):
    """Five-seed groups only; never silently blend the historical ten-seed table."""
    groups = {}
    for row in rows:
        key = (row['method_id'], row['dataset'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (method, dataset), group in sorted(groups.items(), key=lambda x:(x[0][1],x[0][0])):
        group = sorted(group, key=lambda r:r['seed'])
        if tuple(r['seed'] for r in group) != SEEDS:
            raise ValueError('Exactly the five registered seeds are required: '+str((method,dataset)))
        mse = np.array([r['MSE'] for r in group], dtype=np.float64)
        mae = np.array([r['MAE'] for r in group], dtype=np.float64)
        if not np.isfinite(mse).all() or not np.isfinite(mae).all():
            raise ValueError('Nonfinite aggregate inputs')
        q1, median, q3 = np.quantile(mse, [.25,.5,.75])
        parameters = {r['trainable_parameters'] for r in group}
        if len(parameters) != 1:
            raise ValueError('Parameter count changed within a registered family')
        if len({(r['training_cap'],r['extended']) for r in group}) != 1:
            raise ValueError('Mixed base/extension schedules within one family')
        devices = sorted({str(r.get('device','not recorded')) for r in group})
        timings = {name:[float(r[name]) for r in group if r.get(name) not in (None,'','not recorded')]
                   for name in ('search_seconds','refit_seconds','evaluation_seconds','total_wall_seconds')}
        summaries.append(dict(method_id=method, dataset=dataset, display_name=LABELS[method], n=5,
            seed_ids=json.dumps(list(SEEDS)), MSE_median=float(median), MSE_Q1=float(q1), MSE_Q3=float(q3),
            MSE_IQR=float(q3-q1), MAE_median=float(np.median(mae)), parameters=next(iter(parameters)),
            additional_parameters=24624 if method.startswith('NO_F_LINEAR') else 0,
            cap=group[0]['training_cap'], extended=group[0]['extended'],
            selected_epochs=json.dumps([r['selected_epoch'] for r in group]),
            provisional=any(r.get('provisional',False) for r in group), device='; '.join(devices),
            **{name+'_median':float(np.median(values)) if len(values)==5 else None for name,values in timings.items()}))
    lookup = {(s['method_id'],s['dataset']):s for s in summaries}
    contrasts, paired = [], []
    for comparison_id, method, reference_method, dataset in CONTRAST_SPECS:
        if (method,dataset) not in groups:
            continue
        if (reference_method,dataset) not in groups:
            raise ValueError('Variant has no matching registered reference')
        variants = {r['seed']:r for r in groups[(method,dataset)]}
        refs = {r['seed']:r for r in groups[(reference_method,dataset)]}
        s, ref = lookup[(method,dataset)], lookup[(reference_method,dataset)]
        gap = s['MSE_median']-ref['MSE_median']; spread=max(s['MSE_IQR'],ref['MSE_IQR'])
        decision = 'REFERENCE_LOWER' if gap>spread else 'VARIANT_LOWER' if gap < -spread else 'UNCLEAR'
        if s['provisional'] or ref['provisional']:
            decision += '_PROVISIONAL'
        deltas=[]; mae_deltas=[]
        for seed in SEEDS:
            d = variants[seed]['MSE']-refs[seed]['MSE']; deltas.append(d)
            m = variants[seed]['MAE']-refs[seed]['MAE']; mae_deltas.append(m)
            paired.append(dict(comparison_id=comparison_id,dataset=dataset,method_id=method,reference_method_id=reference_method,
                seed=seed,variant_MSE=variants[seed]['MSE'],reference_MSE=refs[seed]['MSE'],delta_MSE=d,
                relative_delta_pct=100*d/refs[seed]['MSE'],variant_MAE=variants[seed]['MAE'],reference_MAE=refs[seed]['MAE'],delta_MAE=m))
        contrasts.append(dict(comparison_id=comparison_id,dataset=dataset,method_id=method,reference_method_id=reference_method,n=5,seed_ids=json.dumps(list(SEEDS)),
            delta_median_MSE=gap, effect_pct=100*gap/ref['MSE_median'], spread=spread,
            descriptive_decision=decision, paired_delta_median=float(np.median(deltas)),
            paired_MAE_delta_median=float(np.median(mae_deltas)), seeds_reference_lower=sum(d>0 for d in deltas),
            seeds_variant_lower=sum(d<0 for d in deltas), seeds_equal=sum(d==0 for d in deltas),
            inference='Exploratory development V; descriptive median/max-IQR rule, not significance or equivalence.'))
    return summaries, contrasts, paired


def public_hashes(paths):
    return {path.resolve().relative_to(runtime.REPO.resolve()).as_posix(): runtime.sha(path)
            for path in paths}


def verify_job_evidence(job, registration, completion):
    job_id = job['job_id']
    root = runtime.REPORT / 'jobs' / job_id
    result_path = root / 'RESULT.json'
    statuses = completion['run_statuses']
    result = runtime.read(result_path)
    spec = run_one.build_spec(job)
    if (statuses.get(job_id) != 'COMPLETE' or result.get('status') != 'COMPLETE'
            or result.get('job') != job or result.get('registration') != registration
            or result.get('spec') != spec or result.get('selection_sealed_before_V') is not True):
        raise ValueError('Completed result differs from sealed job/controller: ' + job_id)
    attempt = result['attempt']
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError('Result has an unregistered attempt: ' + job_id)
    public = root / f'attempt_{attempt:03d}'
    required = [result_path, root / f'WORKER_RECEIPT_attempt{attempt}.json']
    required += [public / name for name in ('START.json', 'SOURCE_CONTRACT.json', 'SEARCH_RESULT.json',
                                           'SELECTION.json', 'REFIT_SUMMARY.json', 'RECEIPT.json',
                                           'OPENED_FILES.json')]
    if any(not path.is_file() for path in required):
        raise ValueError('Completed job is missing public evidence: ' + job_id)
    start = runtime.read(public / 'START.json')
    receipt = runtime.read(public / 'RECEIPT.json')
    worker = runtime.read(required[1])
    if (start.get('job') != job or start.get('attempt') != attempt
            or start.get('registration') != registration or start.get('spec') != spec):
        raise ValueError('Start identity differs from completed job: ' + job_id)
    if (receipt.get('status') != 'COMPLETE' or receipt.get('job') != job
            or receipt.get('attempt') != attempt or receipt.get('registration') != registration
            or receipt.get('training_attempted') is not True or receipt.get('training_executed') is not True
            or receipt.get('T_numeric_values_parsed') != 0
            or any(receipt.get(key, {}).get('changed') != [] for key in ('preserved_before', 'preserved_after'))):
        raise ValueError('Successful attempt receipt is invalid: ' + job_id)
    if (worker.get('job_id') != job_id or worker.get('attempt') != attempt
            or worker.get('status') != 'COMPLETE' or worker.get('exit_code') != 0):
        raise ValueError('Successful worker receipt is invalid: ' + job_id)
    recorded_attempts = [r for r in completion.get('attempts', {}).get(job_id, [])
                         if r.get('attempt') == attempt and 'exit_code' in r]
    if any(r['exit_code'] != 0 or r.get('job_id') != job_id for r in recorded_attempts):
        raise ValueError('Controller attempt differs from successful worker: ' + job_id)
    if runtime.read(public / 'OPENED_FILES.json').get('T_numeric_values_parsed') != 0:
        raise ValueError('Opened-file audit does not confirm protected T exclusion: ' + job_id)
    runtime.assert_protected_T(result)
    source = runtime.read(public / 'SOURCE_CONTRACT.json')
    contract = result['contract']
    if (source.get('spec') != spec or not source.get('sources')
            or any(source.get(key) != contract.get(key) for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha'))):
        raise ValueError('Source contract differs from completed job: ' + job_id)
    selection_path = public / 'SELECTION.json'
    if Path(result['selection_path']).resolve() != selection_path.resolve():
        raise ValueError('Result points outside its successful selection attempt: ' + job_id)
    run_one.check_selection(selection_path, result['selection_sha256'], job)
    selection = runtime.read(selection_path)
    search = runtime.read(public / 'SEARCH_RESULT.json')
    selected, selected_mse = run_one.choose_epoch(search, spec['epochs'])
    if (selection.get('spec') != spec or selection.get('contract') != contract
            or selection.get('selection_split') != 'A_stop' or selection.get('curves') != search['curves']
            or selection.get('selected_epoch') != selected or selection.get('selected_astop_mse') != selected_mse
            or result.get('search_summary') != search['summary']):
        raise ValueError('Selection evidence differs from registered A_stop search: ' + job_id)
    refit = runtime.read(public / 'REFIT_SUMMARY.json')
    if refit.get('status') != 'SEGMENT_COMPLETE' or result.get('refit_summary') != refit:
        raise ValueError('Fresh refit evidence is incomplete: ' + job_id)
    row = result['row']
    if (any(row.get(key) != value for key, value in job.items()) or row.get('reused') is not False
            or row.get('selected_epoch') != selected or row.get('training_cap') != spec['epochs'][-1]
            or row.get('cap_selected') != (selected == spec['epochs'][-1])):
        raise ValueError('Result row differs from job or sealed selection: ' + job_id)
    if runtime.sha(result['bundle_path']) != result['bundle_sha256']:
        raise ValueError('Selected adapter bundle changed: ' + job_id)
    if runtime.sha(row['prediction_path']) != row['prediction_sha256']:
        raise ValueError('Completed prediction archive changed: ' + job_id)
    return result, public_hashes(required)


def verify_completion_evidence(plan, completion):
    if completion.get('status') != 'TRAINING_COMPLETE':
        raise ValueError('Mandatory training is not complete; final report cannot imply completion')
    all_jobs = plan['jobs']+plan['optional_jobs']+plan['extension_jobs']
    if len({j['job_id'] for j in all_jobs}) != len(all_jobs):
        raise ValueError('Duplicate registered job')
    for key, expected, extended in [('jobs',MANDATORY,False),('optional_jobs',OPTIONAL,False),('extension_jobs',ALL_FAMILIES,True)]:
        jobs = plan[key]
        if len(jobs) != 5*len(expected):
            raise ValueError('Registered family count differs: '+key)
        if {(j['method_id'],j['dataset']) for j in jobs} != set(expected):
            raise ValueError('Registered families differ: '+key)
        for job in jobs:
            run_one.validate_job(job)
            if job['extended'] is not extended:
                raise ValueError('Registration extension flag differs')
        for method,dataset in expected:
            if tuple(sorted(j['seed'] for j in jobs if (j['method_id'],j['dataset'])==(method,dataset))) != SEEDS:
                raise ValueError('Registered seeds differ: '+str((method,dataset)))
    registration = run_one.validate_registration(plan['jobs'][0])
    results, verified_files = {}, {}
    for job in all_jobs:
        path = runtime.REPORT/'jobs'/job['job_id']/'RESULT.json'
        state = completion['run_statuses'].get(job['job_id'])
        if not path.is_file():
            if state == 'COMPLETE' or job in plan['jobs']:
                raise ValueError('Missing completed mandatory/result evidence: '+job['job_id'])
            continue
        result, hashes = verify_job_evidence(job, registration, completion)
        row = result['row']
        is_no_f = job['method_id'].startswith('NO_F_LINEAR')
        if row.get('additional_parameters') != (24624 if is_no_f else 0):
            raise ValueError('Incorrect additional head parameter disclosure')
        if is_no_f and (row.get('foundation_kind') != 'NO_FOUNDATION' or row.get('logical_F_rows') != 0 or
                        result['contract']['model_sha'] != 'NO_FOUNDATION'):
            raise ValueError('No-F evidence still uses the frozen foundation model')
        if row.get('device') != result['spec']['device']:
            raise ValueError('Device disclosure differs from registered spec')
        if not np.isfinite(float(row.get('total_wall_seconds',float('nan')))):
            raise ValueError('Training/total time disclosure missing')
        results[job['job_id']] = result; verified_files.update(hashes)
    mandatory_count=sum(j['job_id'] in results for j in plan['jobs'])
    optional_count=sum(j['job_id'] in results for j in plan['optional_jobs'])
    if completion.get('base_complete_count') != mandatory_count or completion.get('optional_complete_count') != optional_count:
        raise ValueError('Completion counters differ from verified results')
    return results, verified_files


def reference_contracts(references):
    contracts = {}
    for r in references['validation']:
        if not {'dataset','job_id','shape','T_start'} <= set(r):
            raise ValueError('Reference validation lacks a dataset/cohort boundary')
        d = r['dataset']
        c = {'shape':r['shape'],'T_start':r['T_start']}
        if d in contracts and contracts[d] != c:
            raise ValueError('Reference cohort boundary differs within a dataset')
        contracts[d] = c
    if set(contracts) != {'jena','appliances','ETTh1'}:
        raise ValueError('Reference datasets differ from registration')
    return contracts


def select_complete_families(plan, completion, verified_results):
    primary, all_new, coverage = [], [], []
    for method,dataset in ALL_FAMILIES:
        bases = [j for j in plan['jobs']+plan['optional_jobs'] if (j['method_id'],j['dataset'])==(method,dataset)]
        extensions = [j for j in plan['extension_jobs'] if (j['method_id'],j['dataset'])==(method,dataset)]
        base_count = sum(j['job_id'] in verified_results for j in bases)
        extension_count = sum(j['job_id'] in verified_results for j in extensions)
        state = completion['extension_status'].get(f'{method}:{dataset}',{})
        extension_complete = extension_count == 5
        if (state.get('status') == 'COMPLETE') != extension_complete:
            raise ValueError('Extension completion status differs from saved evidence')
        if base_count==5:
            if state.get('status') not in ('NOT_REQUIRED','COMPLETE','NOT_RUN_BUDGET','INCOMPLETE'):
                raise ValueError('Completed analysis requires a terminal extension decision')
            cap3_count=sum(float(verified_results[j['job_id']]['row']['selected_epoch'])==3 for j in bases)
            if state.get('cap_selected_count') != cap3_count:
                raise ValueError('Extension trigger count differs from sealed selection')
            if (state.get('status')=='NOT_REQUIRED') != (cap3_count<3):
                raise ValueError('Extension threshold differs from registration')
            if extension_complete:
                cap10_count=sum(float(verified_results[j['job_id']]['row']['selected_epoch'])==10 for j in extensions)
                if state.get('cap10_selected_count')!=cap10_count or state.get('undertrained_at_10')!=(cap10_count>=3):
                    raise ValueError('Cap10 provisional disclosure differs from sealed selection')
        elif extension_count:
            raise ValueError('An extension exists before its complete base family')
        selected_ids = {j['job_id'] for j in (extensions if extension_complete else bases)} if base_count==5 else set()
        provisional = (base_count==5 and cap3_count>=3 and not extension_complete) or state.get('status')=='BASE_INCOMPLETE' or bool(state.get('undertrained_at_10'))
        for job in bases+extensions:
            if job['job_id'] not in verified_results:
                continue
            row = normalized_row(verified_results[job['job_id']]['row'])
            row['primary'] = job['job_id'] in selected_ids
            row['provisional'] = provisional
            all_new.append(row)
            if row['primary']:
                primary.append(row)
        coverage.append(dict(method_id=method,dataset=dataset,required=(method,dataset) in MANDATORY,
            completed_base=base_count,planned_base=5,completed_extension=extension_count,
            primary_complete=base_count==5,primary_n=5 if base_count==5 else 0,
            extension_status=state.get('status','NOT_REACHED'),provisional=provisional,
            interpretation='Five-seed estimate available' if base_count==5 else 'Not estimated; incomplete or not run'))
    return primary, all_new, coverage


def historical_evidence():
    for name,digest in HISTORICAL_INPUTS.items():
        if runtime.sha(runtime.REPO/name) != digest:
            raise ValueError('Pinned historical evidence changed: '+name)
    result_path = next(iter(HISTORICAL_INPUTS))
    with (runtime.REPO/result_path).open(encoding='utf-8-sig',newline='') as f:
        selected = [r for r in csv.DictReader(f) if (r['dataset'],r['role']) in {
            ('jena','R4_B2'),('jena','R5a'),('jena','R5b'),('pems08','B2_PARENT_CPU_FULL'),('pems08','FIXED_B2_NO_BOLT_HEAD')}]
    allowed_scopes = {'REUSED_COMPLETE_V43','NEW_FULL_CPU_V148_AND_INDEPENDENT_FP64','NEW_FULL_CPU_V148_FP64_CPU6'}
    if len(selected) != 5 or any(r['accuracy_scope'] not in allowed_scopes for r in selected):
        raise ValueError('Historical full-V role evidence differs')
    source_rows = [{k:r[k] for k in ('dataset','role','seed','L','H','C','K','MSE','MAE','accuracy_device','accuracy_scope','V_origins')} for r in selected]
    audit_path = runtime.REPORT/'PPT_CLAIM_EVIDENCE_AUDIT.json'
    audit = runtime.read(audit_path)
    if audit.get('protected_T_numeric_values_read') != 0 or audit.get('new_experiment_completion_claim') is not False:
        raise ValueError('Historical preparation audit changed scope')
    for source in audit['sources'].values():
        if runtime.sha(runtime.REPO/source['path']) != source['sha256']:
            raise ValueError('Historical evidence audit source changed: '+source['path'])
    evidence = {'status':'HISTORICAL_EVIDENCE_VERIFIED','source_hashes':HISTORICAL_INPUTS,
        'preparation_audit_sha256':runtime.sha(audit_path),'historical_claims':audit['claims'],
        'source_rows':source_rows,
        'correction_ko':'F 대조가 전혀 없었다는 이전 설명을 정정한다. Jena CPU 1seed R5a/R5b와 PeMS CPU 1seed 고정 head 대조가 이미 있었다. 새 5seed 비교는 이 근거를 확장하며 기존 결과를 대체하거나 삭제하지 않는다.',
        'limitations_ko':['R5a는 최종 PED 고정 후 head 학습이고 R5b는 새 공동학습이다. 동일한 최종 PED 가중치 비교가 아니다.',
            'PeMS의 고정 head 대조는 현재의 Jena/Appliances 공동학습 no-F 실험과 데이터·예측 길이·채널 수가 다르다.',
            'Jena PCA/QR 2seed에서 QR MSE는 각각 +2.86%, +6.21%였다. E와 D 초기화가 함께 바뀌므로 E만의 인과 효과는 아니다.',
            '고정 PE·전체512×6 history 선형 reader에서 P 표현은 B0보다 두 seed MSE가 +2.28%, +0.71%였다. 이 제한된 선형 접근성 반례를 보존한다.',
            '과거 A5 lag0는 모든 데이터 n=0으로 예산상 미실행이었다. 효과 없음으로 해석할 수 없다.'],
        'new_experiment_status':'PENDING until ANALYSIS_VALIDATION passes'}
    runtime.atomic(runtime.REPORT/'HISTORICAL_EVIDENCE.json',evidence)
    return evidence


def claim_map(contrasts=None, coverage=None):
    evidence = {
        'F_VALUE':dict(question='Does frozen F add value compared with a trainable linear temporal head?',
            historical='Jena R5a/R5b and PeMS fixed-head single-seed contrasts exist; old omission corrected.',
            new_families=['NO_F_LINEAR:jena','NO_F_LINEAR:appliances','NO_F_LINEAR_NO_P:jena'],
            boundary='No-F adds 24,624 trainable parameters. Hardware, optimization, P and historical cap differences remain explicit.'),
        'ANCHOR_VALUE':dict(question='Does the direct last-observation output bypass help under the tested parameterization?',
            design='The last available C-channel context value bypasses K-dimensional compression; it may be forward-filled if raw observations are missing. The learned path predicts changes.',
            new_families=['NO_ANCHOR:ETTh1','NO_ANCHOR:appliances'],
            boundary='Input history and latent last-value differencing remain. This tests the direct output level-preservation bypass, not whether observations should be used at all; all-shift robustness is not established.'),
        'LAG_VALUE':dict(question='Does the multi-lag decoder add value beyond lag0 while retaining the anchor?',
            historical='Earlier A5 had n=0; A4 changed the complete decoder and did not isolate lag features.',
            new_families=['A5_LAG0:ETTh1'],
            boundary='Lag0 retains the last-observation anchor and uses fewer decoder parameters; this is not a pure parameter-matched lag test.'),
        'P_GENERALITY':dict(question='How broadly does the pre-compression P module help?',
            historical='Jena 10seed branches: full vs time-only unclear; full vs channel-only lower. Fixed-PE full-history reader two-seed counterexample retained.',
            new_families=['NO_F_LINEAR_NO_P:jena'],
            boundary='Five-seed P×F context is exploratory, with Full P cap3 and historical No P cap10; no intrinsic-axis superiority or independent-test claim.'),
        'PCA_SCOPE':dict(question='What supports PCA initialization and the K heuristic?',
            historical='PCA/QR two-seed evidence changed E and decoder initialization jointly; K variance is an initial-input heuristic.',
            new_families=[],boundary='No new PCA/QR or K tuning in this campaign.')}
    result = {'status':'NEW_RESULTS_AVAILABLE' if contrasts is not None else 'DRAFT_NEW_RESULTS_PENDING',
              'claims':evidence,'contrasts':contrasts or [],'coverage':coverage or [],
              'all_new_results_scope':'Exploratory development V, same five seeds per completed family; never mixed with 10seed tables.'}
    runtime.atomic(runtime.REPORT/'CLAIM_EVIDENCE_MAP.json',result)
    return result


def interaction_summary(rows):
    methods = ('OURS','A1','NO_F_LINEAR','NO_F_LINEAR_NO_P')
    arms = {m:{r['seed']:r for r in rows if r['dataset']=='jena' and r['method_id']==m} for m in methods}
    if any(tuple(sorted(arm)) != SEEDS for arm in arms.values()):
        raise ValueError('Incomplete five-seed Jena 2x2')
    values=[]
    for seed in SEEDS:
        frozen_effect=arms['A1'][seed]['MSE']-arms['OURS'][seed]['MSE']
        linear_effect=arms['NO_F_LINEAR_NO_P'][seed]['MSE']-arms['NO_F_LINEAR'][seed]['MSE']
        values.append(dict(seed=seed,P_removal_with_frozen_F=frozen_effect,P_removal_with_linear_head=linear_effect,
                           difference_of_differences=linear_effect-frozen_effect))
    return dict(dataset='jena',n=5,status='DESCRIPTIVE_ONLY',rows=values,
                median=float(np.median([r['difference_of_differences'] for r in values])),
                formula='(NO_F_LINEAR_NO_P - NO_F_LINEAR) - (A1 - OURS)',
                limitation='Schedule-conditional exploratory description, not a causal interaction test. Historical No P cap10 differs from Full P cap3; parameter counts differ.')


def make_figures(rows, coverage, contrasts, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'font.family':'DejaVu Sans'})
    def group(method,dataset):
        return sorted([r for r in rows if r['method_id']==method and r['dataset']==dataset],key=lambda r:r['seed'])
    def paired_panel(ax,dataset,methods,labels,title):
        groups=[group(m,dataset) for m in methods]
        if any(len(g)!=5 for g in groups):
            ax.text(.5,.5,'Not estimated\nFive-seed family not completed',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off(); ax.set_title(title); return
        values=np.array([[r['MSE'] for r in g] for g in groups])
        for i in range(5):
            ax.plot(range(len(methods)),values[:,i],color='#999999',alpha=.60,lw=.8,zorder=1)
        colors=['#0072B2','#D55E00','#009E73','#CC79A7']
        for i,g in enumerate(groups):
            ax.scatter(np.full(5,i),values[i],s=30,color=colors[i],zorder=3)
            ax.hlines(np.median(values[i]),i-.17,i+.17,color='#111111',lw=2,zorder=4)
        ticks=[label+f'\ncap {g[0]["training_cap"]:g}'+(' ext' if g[0]['extended'] else '')+
               ('\nPROVISIONAL' if any(r.get('provisional') for r in g) else '') for label,g in zip(labels,groups)]
        ax.set_xticks(range(len(methods)),ticks,fontsize=9); ax.set_ylabel('Normalized V MSE ↓')
        ax.set_title(title); ax.grid(axis='y',alpha=.2)
    def save(fig,name):
        for suffix in ('png','svg','pdf'):
            fig.savefig(runtime.REPORT/f'{name}.{suffix}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={'width_ratios':[1.6,1]})
    paired_panel(axes[0],'jena',['OURS','A1','NO_F_LINEAR','NO_F_LINEAR_NO_P'],
                 ['P + frozen F','No P + frozen F','P + linear','No P + linear'],'Jena: same five seeds, P × forecaster')
    paired_panel(axes[1],'appliances',['OURS','NO_F_LINEAR'],['P + frozen F','P + linear'],'Appliances: F replacement')
    fig.suptitle('Exploratory V; each connecting line is one seed; horizontal marks are medians',fontsize=11)
    fig.text(.5,.015,'Linear replacement adds 24,624 trainable parameters. Historical No P cap10 is not a balanced-budget factorial control.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95)); save(fig,'FIG_F_REPLACEMENT')
    fig,axes=plt.subplots(1,3,figsize=(14.0,4.9))
    for ax,dataset,method,title in zip(axes,['ETTh1','appliances','ETTh1'],['NO_ANCHOR','NO_ANCHOR','A5_LAG0'],
                                    ['ETTh1: direct last-value bypass','Appliances: direct last-value bypass','ETTh1: extra lag references']):
        label='No direct bypass' if method=='NO_ANCHOR' else 'Lag0; anchor kept'
        paired_panel(ax,dataset,['OURS',method],['Full P',label],title)
    fig.suptitle('Exploratory V; five matched seeds per completed panel',fontsize=11)
    fig.text(.5,.015,'No-anchor retains input history and latent differences. Lag0 retains the anchor and changes decoder parameter count.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94)); save(fig,'FIG_ANCHOR_LAG')
    fig,ax=plt.subplots(figsize=(10.7,max(4,0.65*len(contrasts)+1.4)))
    names=[]
    for i,c in enumerate(contrasts):
        points=[p['delta_MSE'] for p in paired if p['comparison_id']==c['comparison_id']]
        ax.scatter(points,np.full(5,i)+np.linspace(-.12,.12,5),s=29,color='#0072B2',zorder=3)
        median=float(np.median(points));ax.plot([median,median],[i-.22,i+.22],color='#D55E00',lw=2.5)
        names.append(c['comparison_id'].replace('_',' ')+'\n'+c['descriptive_decision'])
    ax.axvline(0,color='#555555',lw=1);ax.set_yticks(range(len(names)),names,fontsize=9);ax.invert_yaxis()
    ax.set_xlabel('Same-seed ΔMSE = variant − registered reference (negative favors variant)')
    ax.set_title('Every paired difference shown; orange marker = paired median')
    ax.grid(axis='x',alpha=.2)
    fig.text(.5,.015,'The registered decision uses difference of group medians / max IQR; it need not have the same sign as the paired median.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,1));save(fig,'FIG_PAIRED_CONTRASTS')


def write_report(summaries,contrasts,coverage,interaction,all_new):
    base_count=sum(not r['extended'] for r in all_new);ext_count=sum(r['extended'] for r in all_new)
    lines=['# PPT 주장 근거 보강 실험 결과','',
        '이번 표는 seed 92601–92605의 개발 V 탐색 결과다. 기존 10seed 표와 혼합하지 않는다. 보호 T는 평가하지 않았다.','',
        f'새 기본 실행 {base_count}/30개(필수20, 선택10), 연장 {ext_count}개 완료. 새 결과가 유리한지에 따라 실행 범위나 판정 기준을 바꾸지 않았다.','',
        '```text','Dataset     Method                  n    median MSE      Q1         Q3      params']
    for s in summaries:
        lines.append(f'{s["dataset"]:<11} {s["method_id"]:<22} {s["n"]:>2}  {s["MSE_median"]:.6f}  {s["MSE_Q1"]:.6f}  {s["MSE_Q3"]:.6f}  {s["parameters"]:>7}')
    lines+=['```','','## 쉽게 읽는 비교 결과','']
    questions={'F_WITH_P_JENA':'Jena에서 P를 유지하고 F를 선형 head로 바꿈',
        'F_WITH_P_APPLIANCES':'Appliances에서 P를 유지하고 F를 선형 head로 바꿈',
        'F_WITHOUT_P_JENA':'Jena에서 P가 없는 상태로 F를 선형 head로 바꿈',
        'P_WITHOUT_F_JENA':'Jena 선형 head 모델에서 P를 제거함',
        'P_WITH_F_JENA':'Jena 고정 F 모델에서 P를 제거함',
        'ANCHOR_ETTH1':'ETTh1에서 마지막 context의 직접 출력 경로만 제거함',
        'ANCHOR_APPLIANCES':'Appliances에서 마지막 context의 직접 출력 경로만 제거함',
        'LAG_ETTH1':'ETTh1에서 직접 경로는 유지하고 추가 lag 참조를 제거함'}
    for c in contrasts:
        decision=c['descriptive_decision']
        verdict='기준 구성의 오차가 더 낮았다' if decision.startswith('REFERENCE_LOWER') else '변형 구성의 오차가 더 낮았다' if decision.startswith('VARIANT_LOWER') else '차이가 불명확했다. 같거나 효과가 없다는 뜻은 아니다'
        lines.append(f'- {questions[c["comparison_id"]]}: 중앙값 MSE {c["effect_pct"]:+.2f}%. 등록된 기술통계 규칙에서는 {verdict}.'+(' 학습 상한/연장 상태 때문에 잠정적이다.' if 'PROVISIONAL' in decision else ''))
    lines+=['','## 등록 비교와 같은 seed 차이','']
    for c in contrasts:
        lines.append(f'- {c["comparison_id"]}: {c["method_id"]} − {c["reference_method_id"]}, 중앙값 MSE {c["effect_pct"]:+.2f}%, {c["descriptive_decision"]}. 짝 차이 중앙값 {c["paired_delta_median"]:+.6f}; reference가 낮은 seed {c["seeds_reference_lower"]}/5, variant가 낮은 seed {c["seeds_variant_lower"]}/5.')
    lines+=['','양수는 변형의 오차가 더 크다는 뜻이다. 등록 판정은 두 집단 중앙값 차이를 큰 쪽 IQR과 비교한 기술통계 기준이며 유의성·동등성 검정이 아니다. paired 중앙값도 별도로 공개해 서로 다른 요약을 선택적으로 쓰지 않는다.','',
        '## 2×2의 조건부 설명','',f'Jena seed별 [(linear/no-P − linear/P) − (F/no-P − F/P)] 중앙값: {interaction["median"]:+.6f}.',
        '역사적 Full P cap3과 No P cap10의 일정, 선형 대체의 추가 파라미터가 다르므로 인과 상호작용 검정으로 부르지 않는다. 모든 seed별 값은 INTERACTION.json에 있다.','',
        '## 각각 무엇을 검증했나','',
        '- F 대체는 기존 단일 seed 근거를 확장한다. 선형 head는 512→48이며 추가 학습 파라미터가 24,624개다. F의 중앙값 예측과 선형 head의 결정적 회귀를 비교한다. 결과가 좋더라도 모든 TSFM의 불필요성을 뜻하지 않는다.',
        '- No-anchor는 전처리된 마지막 context 값의 직접 출력 우회만 제거한다. 입력 history와 latent z_last 차분은 남는다. 관측값 전체를 써야 하는지를 시험한 것이 아니다.',
        '- 마지막 context는 원시 마지막 값이 결측이면 forward-fill된 값일 수 있다. 직접 경로의 설계 이유는 현재 수준을 K차원 경로 밖에서 보존하고 learned path가 변화를 모델링하게 하는 데 있다.',
        '- A5 lag0는 anchor를 유지한다. 추가 과거 lag 참조를 제거하는 대조이며 decoder 파라미터 수도 달라진다.',
        '- 역사적 F 1seed, PCA/QR 2seed, 고정 PE reader의 P 반례는 HISTORICAL_EVIDENCE.json과 원 감사 문서에 남겼다. 새 결과로 삭제하지 않는다.','',
        '## 실행 범위와 비용','']
    for c in coverage:
        lines.append(f'- {c["method_id"]}/{c["dataset"]}: 기본 {c["completed_base"]}/5, 연장 {c["completed_extension"]}/5, {c["extension_status"]}; '+('5seed 주 결과 있음.' if c['primary_complete'] else '미완료/미실행으로 점수·효과를 판정하지 않음.'))
    lines+=['','```text','Dataset     Method                  device       search_s  refit_s  total_s  add_params']
    def fmt(v):return 'NA' if v is None else f'{v:.2f}'
    for s in summaries:
        lines.append(f'{s["dataset"]:<11} {s["method_id"]:<22} {s["device"]:<12} {fmt(s["search_seconds_median"]):>8} {fmt(s["refit_seconds_median"]):>8} {fmt(s["total_wall_seconds_median"]):>8} {s["additional_parameters"]:>10}')
    lines+=['```','','시간은 실제 기록의 중앙값이며 NA는 미기록이다. 추론 속도나 모델에 고유한 비용으로 일반화하지 않는다. 구체적 selected epochs/cap/provisional은 SUMMARY.csv에 있다.','',
        '## 아직 충분하지 않은 근거','',
        '1. E 고정 대 학습의 직접 비교는 이번 범위에 없다. PCA/QR은 E와 decoder 초기화를 함께 바꾼다.',
        '2. C2/C3의 학습 상한 문제와 기존 비교의 잠정 상태는 해소되지 않는다.',
        '3. P의 정보 보존 메커니즘은 성능 대조만으로 증명되지 않는다. 기존 선형 reader 반례도 남는다.',
        '4. n=5 파일럿·반복 확인한 V이므로 일반성, 동등성, 독립 테스트 성능을 확정할 수 없다.',
        '5. 선택 실험이 예산상 빠졌다면 그 질문은 미해결이다. 문구 정리는 실증 해결과 다르다.','']
    (runtime.REPORT/'RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf8')
    (runtime.REPORT/'REMAINING_GAPS_KO.md').write_text('# 미해결 근거와 후속 우선순위\n\n'+
        '\n'.join(lines[lines.index('## 아직 충분하지 않은 근거')+2:]),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--historical-only',action='store_true');args=parser.parse_args()
    runtime.REPORT.mkdir(parents=True,exist_ok=True)
    history=historical_evidence()
    if args.historical_only:
        claim_map();print(json.dumps({'status':'HISTORICAL_INPUTS_VERIFIED','new_results':'PENDING'}));return
    completion=runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
    plan=runtime.read(runtime.REPORT/'RUN_PLAN.json')
    protocol=runtime.read(runtime.REPORT/'PREREGISTRATION.json')
    registered={(c['comparison_id'],c['variant_method_id'],c['reference_method_id'],c['dataset']) for c in protocol['analysis']['contrasts']}
    if registered != set(CONTRAST_SPECS):
        raise ValueError('Analysis contrasts differ from sealed registration')
    if not protocol['analysis'].get('interaction'):
        raise ValueError('2x2 descriptive interaction was not registered')
    preservation=runtime.guard_preserved()
    verified_results,verified_files=verify_completion_evidence(plan,completion)
    references=runtime.read(runtime.REPORT/'REFERENCE_RUNS.json')
    if references.get('status')!='VERIFIED' or references.get('T_numeric_values_parsed')!=0:
        raise ValueError('References were not independently verified before registration')
    for r in references.get('no_p_extension_evidence',[]):
        for path_key,sha_key in [('result_path','result_sha256'),('selection_path','selection_sha256')]:
            if runtime.sha(r[path_key]) != r[sha_key]:
                raise ValueError('Historical No-P evidence changed')
    contracts=reference_contracts(references)
    ref_rows=[normalized_row(dict(r,reused=True)) for r in references['rows']]
    if len(ref_rows)!=20 or {(r['method_id'],r['dataset']) for r in ref_rows}!={('OURS','jena'),('OURS','appliances'),('OURS','ETTh1'),('A1','jena')}:
        raise ValueError('Reference rows differ from the matched five-seed design')
    primary,all_new,coverage=select_complete_families(plan,completion,verified_results)
    rows=ref_rows+primary
    checks=[]
    for dataset in ('jena','appliances','ETTh1'):
        canonical=next(r for r in ref_rows if r['dataset']==dataset and r['method_id']=='OURS' and r['seed']==92603)
        cohort,_=verify_arrays(canonical,None,contracts[dataset])
        if 'mean' not in cohort or 'std' not in cohort:
            raise ValueError('Canonical five-seed reference lacks scaler arrays')
        for row in [r for r in rows if r['dataset']==dataset]+[r for r in all_new if not r['primary'] and r['dataset']==dataset]:
            _,check=verify_arrays(row,cohort,contracts[dataset]);checks.append(check)
    summaries,contrasts,paired=aggregate(rows)
    interaction=interaction_summary(rows)
    runtime.atomic(runtime.REPORT/'INTERACTION.json',interaction)
    for name,table in [('RUNS.csv',rows),('ALL_NEW_RUNS.csv',all_new),('SUMMARY.csv',summaries),
                       ('CONTRASTS.csv',contrasts),('PAIRED_DIFFERENCES.csv',paired),('COVERAGE.csv',coverage)]:
        write_csv(runtime.REPORT/name,table)
    claim_map(contrasts,coverage)
    history['new_experiment_status']='REPORTED_SEPARATELY_AS_MATCHED_FIVE_SEED_EXPLORATORY_RESULTS'
    runtime.atomic(runtime.REPORT/'HISTORICAL_EVIDENCE.json',history)
    make_figures(rows,coverage,contrasts,paired)
    write_report(summaries,contrasts,coverage,interaction,all_new)
    output_names=['RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv','COVERAGE.csv',
        'INTERACTION.json','CLAIM_EVIDENCE_MAP.json','HISTORICAL_EVIDENCE.json','RESULTS_KO.md','REMAINING_GAPS_KO.md']
    output_names += [f'{name}.{suffix}' for name in ('FIG_F_REPLACEMENT','FIG_ANCHOR_LAG','FIG_PAIRED_CONTRASTS') for suffix in ('png','svg','pdf')]
    inputs=['REGISTRATION_SEAL.json','PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','PRESERVATION_BASELINE.json',
            'REFERENCE_RUNS.json','EXECUTION_COMPLETION.json','PPT_CLAIM_EVIDENCE_AUDIT.json','PPT_CLAIM_EVIDENCE_AUDIT_KO.md']
    verified_files.update(public_hashes(runtime.REPORT/n for n in output_names+inputs))
    verified_files.update(public_hashes([Path(__file__),Path(__file__).parent/'tests/test_postprocess.py']))
    verified_files.update(HISTORICAL_INPUTS)
    output=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),base_complete=20,
        optional_complete=sum(not r['extended'] and (r['method_id'],r['dataset']) in OPTIONAL for r in all_new),
        new_result_count=len(all_new),primary_rows=len(rows),reference_rows=len(ref_rows),verified_files=verified_files,
        checked_prediction_archives=len(checks),independent_FP64_checks=checks,preservation=preservation,T_numeric_values_parsed=0,
        summary=summaries,contrasts=contrasts,coverage=coverage,interaction=interaction,
        inputs={'registration_seal_sha256':runtime.sha(runtime.REPORT/'REGISTRATION_SEAL.json'),
                'completion_sha256':runtime.sha(runtime.REPORT/'EXECUTION_COMPLETION.json')})
    runtime.atomic(runtime.REPORT/'ANALYSIS_VALIDATION.json',output)
    print(json.dumps({'status':'PASS','base_complete':20,'total_new':len(all_new),'primary_rows':len(rows)}))


if __name__=='__main__':
    main()


