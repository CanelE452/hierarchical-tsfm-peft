"""Independently verify the prospective E-only intervention before describing it."""
from datetime import datetime, timezone
import csv
import io
import json
import hashlib
from pathlib import Path

import numpy as np

from . import runtime, run_one, prepare

SEEDS = tuple(range(92601, 92606))
CONTRAST_SPECS = (('E_LEARNING_JENA', 'E_FIXED', 'OURS', 'jena'),)
LABELS = {'OURS':'Full P; E learned', 'E_FIXED':'Full P; PCA E fixed'}
LEGACY_WITHOUT_SCALERS = {'HISTORICAL_jena_K6_s92601', 'HISTORICAL_jena_K6_s92602'}

def write_csv(path, rows, empty_fields=()):
    if not rows:
        if not empty_fields:
            raise ValueError('An empty table needs explicit column names: '+str(path))
    keys = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(empty_fields)
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
        arrays={key:archive[key] for key in ('origins','channel_ids')}
        n,horizon,channels=contract['shape']
        if (arrays['origins'].shape!=(n,) or arrays['channel_ids'].shape!=(channels,)
                or not np.isfinite(arrays['origins']).all() or len(np.unique(arrays['origins']))!=n
                or len(np.unique(arrays['channel_ids']))!=channels):
            raise ValueError('Cohort identity metadata is invalid')
        if int(arrays['origins'].max())+horizon>int(contract['T_start']):
            raise ValueError('V prediction would cross protected T; numeric arrays were not read')
        arrays.update({key:archive[key] for key in ('pred_normalized','target_normalized','observed')})
        for key in ('mean', 'std'):
            if key in archive.files:
                arrays[key] = archive[key]
            elif row['job_id'] not in LEGACY_WITHOUT_SCALERS:
                raise ValueError('Unexpected missing normalization metadata: '+row['job_id'])
    p, y, mask = arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'].astype(bool)
    if list(p.shape) != list(contract['shape']) or y.shape != p.shape or mask.shape != p.shape:
        raise ValueError('V shape differs from the registered reference cohort')
    if (not np.isin(arrays['observed'],[False,True]).all()
            or not np.isfinite([row['MSE'],row['MAE']]).all()):
        raise ValueError('Cohort identity, observation mask, or metric metadata is invalid')
    if ('mean' in arrays)!=('std' in arrays):
        raise ValueError('Normalization metadata must be a complete mean/std pair')
    if 'mean' in arrays and 'std' in arrays:
        if (arrays['mean'].shape!=(p.shape[-1],) or arrays['std'].shape!=(p.shape[-1],)
                or not np.isfinite(arrays['mean']).all() or not np.isfinite(arrays['std']).all()
                or np.any(arrays['std']<=0)):
            raise ValueError('Normalization metadata is invalid')
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
            additional_parameters=0,
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
                                           'OPENED_FILES.json', 'HOST_GPU_GATE.json')]
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


def float32_sha(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes(order='C')).hexdigest()


def verify_E_artifacts(result):
    """Check real CPU-loaded selected/checkpoint tensors, not only claimed hashes."""
    import torch
    job, row = result['job'], result['row']
    public = runtime.REPORT/'jobs'/job['job_id']/f"attempt_{result['attempt']:03d}"
    records = runtime.read(public/'E_INVARIANCE.json')
    if result.get('E_INVARIANCE') != records or [record.get('phase') for record in records] != ['search','refit']:
        raise ValueError('E audit must contain exactly the same search/refit records')
    checks = []
    for record, summary, endpoint in zip(records, (result['search_summary'],result['refit_summary']), (10482,12579)):
        digest_values = [record.get(key) for key in ('initial_E_sha256','live_E_sha256','selected_E_sha256')]
        expected_segments = len(result['spec']['epochs']) if record['phase']=='search' else int(row['selected_epoch']>0)
        if (record.get('status')!='PASS' or record.get('fit_end')!=endpoint
                or record.get('E_requires_grad') is not False or record.get('E_gradient_absent') is not True
                or record.get('frozen_parameters')!=126 or record.get('trainable_parameters')!=35591
                or record.get('updates')!=summary['updates'] or record.get('optimizer_parameter_group_checks')!=expected_segments
                or summary.get('E_INVARIANCE')!=record or len(set(digest_values))!=1
                or not isinstance(digest_values[0],str) or len(digest_values[0])!=64):
            raise ValueError('Stage-local E/optimizer invariant differs')
    bundle_path=Path(result['bundle_path'])
    if bundle_path.resolve() != (runtime.CACHE/'jobs'/job['job_id']/f"attempt_{result['attempt']:03d}"/'selected_bundle.pt').resolve():
        raise ValueError('Selected bundle is outside the successful E attempt')
    bundle=torch.load(bundle_path,map_location='cpu',weights_only=False)
    E=bundle['model']['E'];basis=torch.as_tensor(bundle['normalization']['basis']).float()
    if (not isinstance(E,torch.Tensor) or E.dtype!=torch.float32 or tuple(E.shape)!=(21,6)
            or not torch.equal(E,basis) or float32_sha(E.numpy())!=records[1]['selected_E_sha256']
            or bundle['spec']!=result['spec'] or bundle['contract']!=result['contract']
            or bundle['normalization']['fit_end']!=12579 or bundle['selected_search_epoch']!=row['selected_epoch']
            or bundle['selection_sha256']!=result['selection_sha256']
            or bundle['search_summary']!=result['search_summary'] or bundle['refit_summary']!=result['refit_summary']):
        raise ValueError('Actual selected E bundle differs from fresh A60 PCA/spec/selection')
    for record,summary in zip(records,(result['search_summary'],result['refit_summary'])):
        checkpoints=sorted((bundle_path.parent/record['phase']).glob('checkpoint_*.pt'))
        epochs=[0.,*result['spec']['epochs']] if record['phase']=='search' else [0.]+([row['selected_epoch']] if row['selected_epoch']>0 else [])
        expected_names=[f'checkpoint_{index:06d}.pt' for index in range(len(epochs))]
        if [path.name for path in checkpoints]!=expected_names or summary.get('completed_epochs')!=epochs:
            raise ValueError('Stage checkpoint count differs from the complete fresh schedule')
        entries=[]
        for index,path in enumerate(checkpoints):
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            current=checkpoint['model']['E'];best=checkpoint['extra']['best_state']['E']
            if (tuple(current.shape)!=(21,6) or current.dtype!=torch.float32
                    or not torch.equal(current,best) or float32_sha(current.numpy())!=record['initial_E_sha256']
                    or checkpoint['contract']!=summary['contract']):
                raise ValueError('A live/selected checkpoint changed its stage-local E or contract')
            curves=checkpoint['extra']['curves']
            if (checkpoint['extra'].get('completed_epochs')!=epochs[:index+1]
                    or [point['epoch'] for point in curves]!=epochs[:index+1]
                    or checkpoint.get('updates')!=curves[-1]['updates']
                    or (index==0 and checkpoint['updates']!=0)
                    or (index>0 and checkpoint['updates']<=previous_updates)):
                raise ValueError('A checkpoint does not record the successive fresh training schedule')
            previous_updates=checkpoint['updates']
            entries.append(dict(path=str(path),sha256=runtime.sha(path)))
        if checkpoint['updates']!=summary['updates']:
            raise ValueError('Last checkpoint update count differs from its stage summary')
        if record['phase']=='refit':
            saved=checkpoint['extra']['best_state']
            if set(saved)!=set(bundle['model']) or any(not torch.equal(value,bundle['model'][name]) for name,value in saved.items()):
                raise ValueError('Selected bundle P/D/E tensors differ from the actual final fresh-refit state')
        checks.append(dict(phase=record['phase'],fit_end=record['fit_end'],E_sha256=record['initial_E_sha256'],
                           checkpoints=entries,optimizer_parameter_group_checks=record['optimizer_parameter_group_checks'],
                           checkpoint_schedule_verified=True,selected_refit_state_verified=record['phase']=='refit'))
    return dict(job_id=job['job_id'],status='PASS',bundle_sha256=result['bundle_sha256'],
                normalization_mean_sha256=float32_sha(bundle['normalization']['mean']),
                normalization_std_sha256=float32_sha(bundle['normalization']['std']),
                stage_E_checks=checks,search_and_refit_basis_may_differ=True), public_hashes([public/'E_INVARIANCE.json'])


def verify_host_gate(path, registration):
    receipt=runtime.read(path)
    binding=runtime.verify_host_gpu_binding()
    if (receipt.get('status')!='PASS' or receipt.get('mode')!='identity_only'
            or receipt.get('host_gpu_binding')!=binding or receipt.get('desktop_concurrent_load') is not True
            or receipt.get('registration_seal_sha256')!=registration['seal_sha256']
            or receipt.get('unexpected_gpu_pids')!=[]):
        raise ValueError('Host GPU admission receipt differs from this study seal and identity-only policy')
    helper=runtime.importlib.import_module(f'research.{runtime.UPSTREAM_NAME}.host_gpu_gate')
    if Path(helper.__file__).resolve()!=(runtime.REPO/runtime.HOST_GPU_HELPER_REL).resolve():
        raise ValueError('Host GPU receipt verifier loaded from another checkout')
    check=helper.verify_identity_receipt(receipt,runtime.REPO/runtime.HOST_GPU_CONTRACT_REL)
    if check.get('status')!='PASS':
        raise ValueError('Saved host GPU identity snapshot failed independent validation')
    return dict(status='PASS',mode='identity_only',desktop_concurrent_load=True,
                receipt_sha256=runtime.sha(path),identity_snapshot_sha256=receipt['identity_snapshot_sha256'],
                policy_sha256=receipt['policy_sha256'],observed_gpu_process_count=receipt['observed_gpu_process_count'],
                allowed_desktop_context_count=receipt['allowed_desktop_context_count'],private_snapshot_verified=True)


def verify_attempts(job, result, completion, successful=True):
    root=runtime.REPORT/'jobs'/job['job_id'];selected=result['attempt']
    if selected not in (1,2):
        raise ValueError('Only two prospective attempts are permitted')
    files=[];rows=[]
    directories=sorted(root.glob('attempt_*'))
    if any(not path.name[8:].isdigit() or not 1<=int(path.name[8:])<=selected for path in directories):
        raise ValueError('Unexpected or post-success repeated attempt')
    for attempt in range(1,selected+1):
        worker_path=root/f'WORKER_RECEIPT_attempt{attempt}.json'
        public=root/f'attempt_{attempt:03d}'
        if worker_path.exists():
            worker=runtime.read(worker_path)
            if worker.get('job_id')!=job['job_id'] or worker.get('attempt')!=attempt:
                raise ValueError('Attempt worker identity differs')
            if (attempt<selected or not successful) and (worker.get('exit_code') in (0,20) or worker.get('status')=='COMPLETE'):
                raise ValueError('Successful/scientific-failure attempt cannot be silently repeated')
            rows.append(dict(attempt=attempt,status=worker['status'],exit_code=worker['exit_code']))
            files.append(worker_path)
        else:
            evidence=completion.get('attempts',{}).get(job['job_id'],[])
            if not any(r.get('attempt')==attempt and r.get('status')=='INTERRUPTED_PROCESS' for r in evidence):
                raise ValueError('Missing attempt receipt without an explicit interrupted-process record')
            rows.append(dict(attempt=attempt,status='INTERRUPTED_PROCESS',exit_code=None))
        host_gate=public/'HOST_GPU_GATE.json'
        if (public/'START.json').exists() and not host_gate.exists():
            raise ValueError('An attempted model loading has no host GPU identity receipt')
        if host_gate.exists():
            rows[-1]['host_gpu_gate']=verify_host_gate(host_gate,result['registration'])
        for path in public.glob('*.json'):
            value=runtime.read(path)
            if path.name in ('START.json','RECEIPT.json'):
                if value.get('job')!=job or value.get('attempt')!=attempt or value.get('registration')!=result['registration']:
                    raise ValueError('Attempt public identity/registration differs')
            if path.name in ('START.json','RECEIPT.json','OPENED_FILES.json') and value.get('T_numeric_values_parsed')!=0:
                raise ValueError('An attempted run did not preserve protected T exclusion')
            files.append(path)
        for folder in ('search','refit'):
            files.extend((public/folder).glob('event_*.json'))
    return rows,public_hashes(files)


def verify_completion(plan, completion):
    if completion.get('status') not in ('TRAINING_COMPLETE','TRAINING_INCOMPLETE','NOT_RUN_BUDGET'):
        raise ValueError('Only a terminal training/budget state can be analyzed; scientific stops require investigation')
    if plan['jobs']!=prepare.jobs() or plan['extension_jobs']!=prepare.jobs(True) or plan.get('optional_jobs')!=[]:
        raise ValueError('Analysis plan differs from the single sealed E-only family')
    jobs=plan['jobs']+plan['extension_jobs'];known={job['job_id'] for job in jobs}
    statuses=completion.get('run_statuses',{})
    if set(statuses)!=known:
        raise ValueError('Completion must account for every registered job exactly')
    terminal={'COMPLETE','FAILED','STOPPED_DEADLINE','NOT_RUN_BUDGET','NOT_REQUIRED','BASE_INCOMPLETE'}
    if any(value not in terminal for value in statuses.values()):
        raise ValueError('A nonterminal or unknown run remains in the completed record')
    registration=run_one.validate_registration(plan['jobs'][0])
    results={};files={};e_checks=[];attempts={}
    for job in jobs:
        path=runtime.REPORT/'jobs'/job['job_id']/'RESULT.json'
        if not path.exists():
            if statuses.get(job['job_id'])=='COMPLETE':
                raise ValueError('A claimed complete run has no RESULT')
            root=path.parent
            attempted=[int(p.name[8:]) for p in root.glob('attempt_*') if p.name[8:].isdigit()]
            attempted += [int(p.stem.removeprefix('WORKER_RECEIPT_attempt')) for p in root.glob('WORKER_RECEIPT_attempt*.json')]
            attempted += [entry['attempt'] for entry in completion.get('attempts',{}).get(job['job_id'],[])]
            if attempted:
                identity=dict(attempt=max(attempted),registration=registration)
                audit,bound=verify_attempts(job,identity,completion,successful=False)
                attempts[job['job_id']]=audit;files.update(bound)
            continue
        result,bound=verify_job_evidence(job,registration,completion)
        row=result['row']
        if (row.get('trainable_parameters')!=35591 or row.get('frozen_E_parameters')!=126
                or row.get('additional_parameters')!=0 or row.get('device')!='cuda'
                or row.get('normalization_fit_end')!=12579 or row.get('logical_F_rows')!=258
                or row.get('foundation_kind')!='FROZEN_BOLT' or result['contract']['model_sha']!=runtime.F_SHA
                or row.get('point_statistic')!='quantile_0.5'
                or not np.isfinite(float(row.get('total_wall_seconds',float('nan'))))):
            raise ValueError('Frozen-E parameter/compute/point/time disclosure differs')
        source=runtime.read(path.parent/f"attempt_{result['attempt']:03d}"/'SOURCE_CONTRACT.json')
        if set(source['sources'])!={'fit_base.py','training.py','networks.py','data_contract.py','gpu_runtime.py','evaluation.py'}:
            raise ValueError('Scientific source contract must bind exactly the six reused core modules')
        for name,digest in source['sources'].items():
            relative='research/informs_broad_validation_20261015_v1/'+name
            if Path(name).name!=name or runtime.sha(runtime.REPO/relative)!=digest:
                raise ValueError('A reused scientific source differs from its execution contract')
            bound[relative]=digest
        digest=hashlib.sha256(json.dumps(source['sources'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=result['contract']['source_sha']:
            raise ValueError('Aggregate scientific source hash differs')
        check,e_files=verify_E_artifacts(result)
        attempt_rows,attempt_files=verify_attempts(job,result,completion)
        results[job['job_id']]=result;e_checks.append(check);attempts[job['job_id']]=attempt_rows
        files.update(bound);files.update(e_files);files.update(attempt_files)
    base=sum(job['job_id'] in results for job in plan['jobs'])
    extended=sum(job['job_id'] in results for job in plan['extension_jobs'])
    if completion.get('base_complete_count')!=base or completion.get('optional_complete_count',0)!=0:
        raise ValueError('Execution counters differ from verified completed runs')
    if completion['status']=='TRAINING_COMPLETE' and base!=5:
        raise ValueError('TRAINING_COMPLETE does not contain all five base runs')
    return results,files,e_checks,attempts


def select_primary(plan, completion, results):
    bases,extensions=plan['jobs'],plan['extension_jobs']
    base_count=sum(job['job_id'] in results for job in bases)
    extension_count=sum(job['job_id'] in results for job in extensions)
    state=completion.get('extension_status',{}).get('E_FIXED:jena',{})
    cap3_count=None;cap10_count=None;provisional=False;required=False
    if base_count==5:
        cap3_count=sum(results[j['job_id']]['row']['selected_epoch']==3 for j in bases)
        required=cap3_count>=3
        if completion['status']=='NOT_RUN_BUDGET':
            admission=completion.get('campaign_admission',{})
            if (not required or extension_count==5 or admission.get('status')!='NOT_RUN_BUDGET'
                    or admission.get('remaining_base')!=0 or admission.get('reserved_extensions')!=5-extension_count):
                raise ValueError('Resume budget exclusion does not explain the unfinished required extension')
            state=dict(status='NOT_RUN_BUDGET',cap_selected_count=cap3_count,
                       reason='Full-study admission after interruption could not reserve the remaining required extension.')
        if state.get('status') not in ('NOT_REQUIRED','COMPLETE','NOT_RUN_BUDGET','INCOMPLETE'):
            raise ValueError('Final analysis requires a terminal extension decision')
        if state.get('cap_selected_count')!=cap3_count:
            raise ValueError('Extension trigger count differs from sealed A_stop choices')
        if (state['status']=='NOT_REQUIRED')!=(not required):
            raise ValueError('Extension trigger does not match the registered3/5 threshold')
        if (state['status']=='COMPLETE')!=(extension_count==5):
            raise ValueError('Extension completion flag differs from five saved results')
        if state['status']=='NOT_REQUIRED' and extension_count:
            raise ValueError('An untriggered extension was run')
        if extension_count==5:
            cap10_count=sum(results[j['job_id']]['row']['selected_epoch']==10 for j in extensions)
            if state.get('cap10_selected_count')!=cap10_count or state.get('undertrained_at_10')!=(cap10_count>=3):
                raise ValueError('Cap10 unresolved disclosure differs from selections')
        provisional=(required and extension_count<5) or (cap10_count is not None and cap10_count>=3)
    elif extension_count:
        raise ValueError('An extension exists before five complete base seeds')
    selected={j['job_id'] for j in (extensions if extension_count==5 else bases)} if base_count==5 else set()
    primary=[];all_new=[]
    for job in bases+extensions:
        if job['job_id'] not in results: continue
        row=normalized_row(results[job['job_id']]['row'])
        row.update(primary=job['job_id'] in selected,provisional=provisional)
        all_new.append(row)
        if row['primary']: primary.append(row)
    coverage=dict(method_id='E_FIXED',dataset='jena',completed_base=base_count,completed_extension=extension_count,
                  planned_base=5,planned_extension=5,primary_complete=base_count==5,primary_n=len(primary),
                  primary_kind='EXTENDED' if extension_count==5 else 'BASE' if base_count==5 else 'NONE',
                  extension_required=required if base_count==5 else None,extension_status=state.get('status','NOT_REACHED'),
                  cap3_selected_count=cap3_count,cap10_selected_count=cap10_count,provisional=provisional,
                  scientific_complete=base_count==5 and not provisional,
                  interpretation='Five-seed estimate available' if base_count==5 else 'No five-seed estimate; incomplete or not run')
    return primary,all_new,coverage


def reference_rows_and_contract():
    reference=runtime.read(runtime.REPORT/'REFERENCE_RUNS.json')
    if (reference.get('status')!='BOUND_TO_VERIFIED_UPSTREAM_REFERENCE' or reference.get('T_numeric_values_parsed')!=0
            or runtime.sha(reference['source'])!=reference['source_sha256']):
        raise ValueError('Reference chain differs from the sealed upstream validation')
    rows=[normalized_row(dict(row,reused=True,primary=True,provisional=False)) for row in reference['rows']]
    if tuple(sorted(row['seed'] for row in rows))!=SEEDS or any(
            row['method_id']!='OURS' or row['dataset']!='jena' or row['K']!=6
            or row['normalization_fit_end']!=12579 or row['trainable_parameters']!=35717
            or row['training_cap']!=3 or row['extended'] for row in rows):
        raise ValueError('Full-P reference is not the matched five-seed K6 family')
    contracts={(tuple(v['shape']),v['T_start']) for v in reference['inherited_validation']}
    if contracts!={((43,48,21),16772)}:
        raise ValueError('The inherited Jena reference cohort differs')
    return sorted(rows,key=lambda row:row['seed']),dict(shape=[43,48,21],T_start=16772)


def make_figure(rows, coverage, summaries, contrasts, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style={'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':12,
           'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9.4,4.4))
        if not coverage['primary_complete']:
            for ax in axes: ax.set_axis_off()
            axes[0].text(.5,.5,'No five-seed E-learning estimate\nExperiment not run or incomplete.',ha='center',va='center',transform=axes[0].transAxes)
            axes[1].text(.5,.5,f"Completed base runs: {coverage['completed_base']}/5\nNo performance value is imputed.",ha='center',va='center',transform=axes[1].transAxes)
            title='Jena E-learning control · no complete estimate'
        else:
            by_method={method:sorted([row for row in rows if row['method_id']==method],key=lambda row:row['seed']) for method in ('OURS','E_FIXED')}
            if any(tuple(row['seed'] for row in group)!=SEEDS for group in by_method.values()):
                raise ValueError('A figure needs five matched seed points in each arm')
            full=np.array([row['MSE'] for row in by_method['OURS']]);fixed=np.array([row['MSE'] for row in by_method['E_FIXED']])
            offsets=np.linspace(-.055,.055,5)
            for index in range(5):
                axes[0].plot([offsets[index],1+offsets[index]],[full[index],fixed[index]],color='#9B9B9B',lw=.85,alpha=.8,zorder=1)
            colors=('#0072B2','#D55E00')
            for x,values,color,marker in zip((0,1),(full,fixed),colors,('o','s')):
                axes[0].scatter(x+offsets,values,color=color,marker=marker,s=38,zorder=3)
                q1,median,q3=np.quantile(values,[.25,.5,.75])
                axes[0].errorbar(x+.15,median,yerr=[[median-q1],[q3-median]],fmt='_',ms=13,color='black',elinewidth=2,capsize=4,zorder=4)
            summary={row['method_id']:row for row in summaries}
            axes[0].set_xticks([0,1],[f"Learned E\ncap {summary['OURS']['cap']:g}",f"Fixed PCA E\ncap {summary['E_FIXED']['cap']:g}"])
            axes[0].set_xlim(-.25,1.35);axes[0].set_ylabel('Normalized development-V MSE')
            axes[0].set_title('A  Matched seeds; median and IQR',loc='left',fontweight='bold')
            delta=fixed-full
            axes[1].axhline(0,color='black',lw=1)
            axes[1].scatter(range(5),delta,color='#0072B2',marker='D',s=44,zorder=3)
            axes[1].set_xticks(range(5),[str(seed) for seed in SEEDS],rotation=30)
            axes[1].set_xlabel('Seed');axes[1].set_ylabel('MSE difference: fixed E − learned E')
            bound=max(float(np.ptp(delta)),float(np.max(np.abs(delta))),1e-4)*.15
            axes[1].set_ylim(min(float(delta.min()),0)-bound,max(float(delta.max()),0)+bound)
            axes[1].set_title('B  Positive values favor learning E',loc='left',fontweight='bold')
            for ax in axes: ax.grid(axis='y',alpha=.18)
            title='Jena · K = 6 · only E learning toggled'
            if coverage['provisional']: title+=' · provisional'
            fig.text(.5,.022,'Five matched seeds; repeatedly used development V. IQR is descriptive, not a confidence interval.',ha='center',fontsize=9)
            fig.text(.5,.067,'Trainable adapter parameters: learned E 35,717; fixed E 35,591. Both retain the same stored architecture.',ha='center',fontsize=9)
        fig.suptitle(title,fontsize=14)
        fig.tight_layout(rect=(0,.11,1,.93))
        for suffix in ('png','pdf','svg'):
            fig.savefig(output_dir/f'FIG_E_FIXED.{suffix}',dpi=400,bbox_inches='tight',facecolor='white')
        plt.close(fig)


def write_report(coverage, summaries, contrasts, status, output_dir):
    lines=['# E 학습 기여: Jena K6 대조','',
           '원 B2의 P·고정 F·7-lag D·마지막 입력값 경로·초기 텐서·학습 규칙을 유지하고, PCA 초기화 E의 학습 여부만 바꾼 탐색 대조다. 기존 Full과 같은 seed92601–92605를 사용한다.','',
           f"실제 완료: 기본 {coverage['completed_base']}/5, 조건부 연장 {coverage['completed_extension']}/5. 주 비교 반복 수: {coverage['primary_n']}.",'']
    if not coverage['primary_complete']:
        lines += ['전체 다섯 seed가 완결되지 않아 E 학습의 성능 효과를 추정하지 않았다. 미실행·누락 결과에 0점이나 가상의 성능값을 넣지 않았다.',
                  f'실행/보고 상태: {status}. E 학습 필요성은 아직 미해결이다.']
    else:
        for summary in sorted(summaries,key=lambda row:row['method_id']!='OURS'):
            name='E를 학습한 Full' if summary['method_id']=='OURS' else 'PCA E를 고정한 대조'
            lines.append(f"- {name}: MSE 중앙값 {summary['MSE_median']:.6f}, IQR [{summary['MSE_Q1']:.6f}, {summary['MSE_Q3']:.6f}], MAE 중앙값 {summary['MAE_median']:.6f}. cap {summary['cap']:g}, 선택 epoch {summary['selected_epochs']}.")
        contrast=contrasts[0];decision=contrast['descriptive_decision'].removesuffix('_PROVISIONAL')
        lines += ['',f"고정 E − 학습 E의 중앙값 차이는 {contrast['delta_median_MSE']:+.6f} ({contrast['effect_pct']:+.2f}%)다. seed별 짝 차이의 중앙값은 {contrast['paired_delta_median']:+.6f}로 별도로 계산했다.",
                  f"학습 E가 더 낮은 seed {contrast['seeds_reference_lower']}/5, 고정 E가 더 낮은 seed {contrast['seeds_variant_lower']}/5, 동일 {contrast['seeds_equal']}/5. 등록한 기술적 판정: {contrast['descriptive_decision']}.",'']
        explanation={'REFERENCE_LOWER':'이번 Jena K6와 학습 규칙에서는 E를 학습하는 쪽을 지지하는 기술적 근거가 나왔다.',
                     'VARIANT_LOWER':'이번 조건에서는 고정 E의 오차가 더 낮아, E를 학습해야 한다는 주장을 지지하지 않았다.',
                     'UNCLEAR':'이번 다섯 seed에서는 E 학습의 이득이 등록 기준상 명확하지 않았다. 동등하거나 효과가 없음을 증명한 것은 아니다.'}
        lines.append(explanation[decision])
        if coverage['provisional']:
            lines.append('다만 필요한 연장이 완결되지 않았거나 cap10 선택이3/5 이상이어서 학습 상한이 미해결이다. 위 비교는 잠정적이며 E 학습 효과가 확정됐다고 쓰지 않는다.')
    lines += ['','E 126개 파라미터를 고정하면 학습 파라미터는35,717개에서35,591개로 줄지만 저장 구조와 전체 어댑터 파라미터 수는 같다. P와 D는 여전히 학습한다.',
              'Search의 PCA는 A_fit에서, fresh refit의 PCA는 A60에서 각각 계산한다. E가 각 단계 안에서 초기값과 같은지를 확인하며 두 단계의 PCA 자체가 서로 같을 필요는 없다.',
              '이 결과는 선택한 Jena·K6·초기화·학습 규칙에서 E 학습의 기여를 묻는다. PCA나 K의 최적성, 모든 모델에서 E 학습의 필연성, 독립 테스트 일반화를 검증하지 않는다. 개발 V를 반복 사용한 n5 탐색이며 seed는 최적화 반복이다.',
              '중앙값 차이를 두 그룹의 최대 IQR과 비교하는 규칙은 기술적 요약이다. 유의성 검정·동등성 검정·신뢰구간으로 해석하지 않는다. 보호 T 수치는 읽지 않았다.',
              '','검증: ANALYSIS_VALIDATION.json, COMPLETION_VALIDATION.json. 원점수와 선택 정보: RUNS.csv, ALL_NEW_RUNS.csv, PAIRED_DIFFERENCES.csv.']
    (output_dir/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    plan=runtime.read(runtime.REPORT/'RUN_PLAN.json')
    protocol=runtime.read(runtime.REPORT/'PREREGISTRATION.json')
    completion=runtime.read(runtime.REPORT/'EXECUTION_COMPLETION.json')
    if protocol['analysis']['contrast']!='E_FIXED minus matched OURS for each seed; positive means learning E helped.':
        raise ValueError('The analysis comparison direction was not registered')
    preservation=runtime.guard_preserved()
    results,verified_files,e_checks,attempts=verify_completion(plan,completion)
    primary,all_new,coverage=select_primary(plan,completion,results)
    ref_rows,contract=reference_rows_and_contract()
    rows=[];array_checks=[]
    if results:
        canonical=next(row for row in ref_rows if row['seed']==92603)
        cohort,_=verify_arrays(canonical,None,contract)
        if 'mean' not in cohort or 'std' not in cohort:
            raise ValueError('The canonical Full reference has no explicit normalization arrays')
        e_map={entry['job_id']:entry for entry in e_checks}
        for row in ref_rows+all_new:
            arrays,check=verify_arrays(row,cohort,contract)
            if row['method_id']=='E_FIXED':
                bundle=e_map[row['job_id']]
                if (float32_sha(arrays['mean'])!=bundle['normalization_mean_sha256']
                        or float32_sha(arrays['std'])!=bundle['normalization_std_sha256']):
                    raise ValueError('Saved prediction scaler differs from its actual selected bundle')
            array_checks.append(check)
        rows=ref_rows+primary if primary else []
    else:
        for row in ref_rows:
            if runtime.sha(row['prediction_path'])!=row['prediction_sha256']:
                raise ValueError('Bound Full reference bytes changed during budget exclusion')
    summaries,contrasts,paired=aggregate(rows) if rows else ([],[],[])
    summaries.sort(key=lambda row:row['method_id']!='OURS')
    for summary in summaries:
        summary.update(total_adapter_parameters=35717,frozen_E_parameters=126 if summary['method_id']=='E_FIXED' else 0)
    status='PASS' if coverage['primary_complete'] else 'NOT_RUN_BUDGET' if completion['status']=='NOT_RUN_BUDGET' else 'INCOMPLETE'
    tables=[('RUNS.csv',rows,('dataset','method_id','seed','MSE','MAE','primary','provisional')),
            ('ALL_NEW_RUNS.csv',all_new,('dataset','method_id','seed','MSE','MAE','extended','primary','provisional')),
            ('SUMMARY.csv',summaries,('method_id','dataset','n','MSE_median','MSE_Q1','MSE_Q3','MSE_IQR','MAE_median','parameters','cap','extended','provisional')),
            ('CONTRASTS.csv',contrasts,('comparison_id','dataset','method_id','reference_method_id','n','effect_pct','descriptive_decision','paired_delta_median')),
            ('PAIRED_DIFFERENCES.csv',paired,('comparison_id','seed','variant_MSE','reference_MSE','delta_MSE','relative_delta_pct'))]
    for name,table,empty in tables: write_csv(runtime.REPORT/name,table,empty)
    make_figure(rows,coverage,summaries,contrasts,runtime.REPORT)
    write_report(coverage,summaries,contrasts,status,runtime.REPORT)
    inputs=['REGISTRATION_SEAL.json','PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','REFERENCE_RUNS.json',
            'PRESERVATION_BASELINE.json','UNIT_TESTS.json','EXECUTION_COMPLETION.json']
    for name in ('LAUNCH.json','ROOT_LAUNCH_GATE.json','ROOT_DESIGN_LOCK_BEFORE_PREVIEW.json','POSTPROCESS_TESTS.json'):
        if (runtime.REPORT/name).exists(): inputs.append(name)
    verified_files.update(public_hashes(runtime.REPORT/name for name in inputs))
    verified_files.update(protocol['registered_sources'])
    binding=runtime.verify_host_gpu_binding(protocol)
    verified_files.update({binding['helper_relative_path']:binding['helper_sha256'],
                           binding['contract_relative_path']:binding['contract_sha256']})
    if protocol.get('amendment'):
        verified_files.update(public_hashes([runtime.REPORT/protocol['amendment'],runtime.REPORT/'revisions/preflight_001/ARCHIVE_RECEIPT.json']))
    verified_files.update(public_hashes([Path(__file__),Path(__file__).parent/'tests/test_postprocess.py']))
    completion_validation=dict(status='PASS',scope='Integrity and completeness audit, not a claim of a favorable scientific result.',
        created_utc=datetime.now(timezone.utc).isoformat(),execution_status=completion['status'],
        base_complete=coverage['completed_base'],extension_complete=coverage['completed_extension'],reference_rows=5,
        primary_complete=coverage['primary_complete'],scientific_complete=coverage['scientific_complete'],coverage=coverage,
        attempt_checks=attempts,E_invariance_checks=e_checks,preservation=preservation,T_numeric_values_parsed=0,
        verified_files=dict(verified_files),inputs=dict(registration_seal_sha256=runtime.sha(runtime.REPORT/'REGISTRATION_SEAL.json'),
                                                     completion_sha256=runtime.sha(runtime.REPORT/'EXECUTION_COMPLETION.json')))
    runtime.atomic(runtime.REPORT/'COMPLETION_VALIDATION.json',completion_validation)
    output_names=[name for name,_,_ in tables]+['RESULTS_KO.md','COMPLETION_VALIDATION.json']+[f'FIG_E_FIXED.{suffix}' for suffix in ('png','pdf','svg')]
    verified_files.update(public_hashes(runtime.REPORT/name for name in output_names))
    analysis=dict(status=status,integrity_status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        base_complete=coverage['completed_base'],extension_complete=coverage['completed_extension'],reference_rows=5,
        primary_rows=len(rows),new_result_count=len(all_new),coverage=coverage,summary=summaries,contrasts=contrasts,
        checked_prediction_archives=len(array_checks),independent_FP64_checks=array_checks,
        private_artifact_checks=e_checks,verified_files=verified_files,T_numeric_values_parsed=0,
        inputs=dict(registration_seal_sha256=runtime.sha(runtime.REPORT/'REGISTRATION_SEAL.json'),
                    completion_sha256=runtime.sha(runtime.REPORT/'EXECUTION_COMPLETION.json'),
                    completion_validation_sha256=runtime.sha(runtime.REPORT/'COMPLETION_VALIDATION.json')),
        interpretation='Exploratory repeated development V, five optimization seeds. No significance/equivalence/independent test claim.',
        metric_aggregation='FP64 mean of per-channel MSE/MAE over observed targets; not pooled-cell averaging.')
    runtime.atomic(runtime.REPORT/'ANALYSIS_VALIDATION.json',analysis)
    print(json.dumps(dict(status=status,base_complete=coverage['completed_base'],extension_complete=coverage['completed_extension'],
                          primary_rows=len(rows),provisional=coverage['provisional'])))


if __name__=='__main__':
    main()
