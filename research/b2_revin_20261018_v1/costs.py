"""Final registered models: exact whole-path GPU cost and historical CPU context."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np
import psutil
import torch

from research.adapter_improvements_20261018_v1 import costs as old_cost
from . import contracts
from .deployment import load_route


PROTOCOL = dict(device='cuda', batches=[1,8], warmups=10, timed_trials=20, stage_trials=1,
                F_row_microbatch=128, atol=1e-5, rtol=1e-4,
                whole_path='physical+mask -> causal fill -> global zscore -> outer RevIN if enabled -> P/E/F/D -> outer inverse -> physical inverse -> CPU',
                denominator='origin', saved_V_parity='entire saved V at each batch facet, two serialized loads',
                cached_forecast_reads_in_timed_path=0)


def select_final_rows(rows, decisions):
    if decisions.get('status') != 'COMPLETE':
        raise ValueError('Final selection decisions are incomplete')
    final_decisions = {}
    for value in decisions['decisions']:
        cohort,cap = value['cohort'],value['cap']
        if cohort not in contracts.COHORTS or type(cap) is not int or cap not in (3,6):
            raise ValueError('Unregistered final cap decision')
        if cohort not in final_decisions or cap > final_decisions[cohort]['cap']:
            final_decisions[cohort] = value
    if set(final_decisions) != set(contracts.COHORTS):
        raise ValueError('Final dataset decision coverage is incomplete')
    chosen = final_decisions['jena']['common_revin']
    if type(chosen) is not bool:
        raise ValueError('Chosen common normalization must be bool')
    expected = {(cohort,'B2',enabled,seed) for cohort in contracts.COHORTS
                for enabled in (False,True) for seed in contracts.SEEDS}
    expected.update({('jena','B0',chosen,seed) for seed in contracts.SEEDS})
    expected.update({('jena','RAW',None,92601),('jena','LORA',None,92601),
                     ('pems32','RAW',None,92601),('pems32','DIRECT_G',None,92601)})
    selected,actual,identifiers = [],set(),set()
    for row in rows:
        if type(row.get('final')) is not bool:
            raise ValueError('Final accuracy selection must be strict bool')
        if not row['final']:
            continue
        family,cohort = row['family'],row['dataset']
        if family == 'RAW_RIDGE':
            continue
        if family in ('B2','B0'):
            if type(row.get('revin')) is not bool or type(row.get('cap')) is not int:
                raise ValueError('B2/B0 normalization and cap identities required')
            if row['cap'] != final_decisions[cohort]['cap']:
                raise ValueError('Final cost accuracy row uses another cap')
            role = (cohort,family,row['revin'],row['seed'])
        else:
            if row['cap'] != 0:
                raise ValueError('Historical baseline cap must be zero')
            role = (cohort,family,None,row['seed'])
        if role not in expected or role in actual or row['run_id'] in identifiers:
            raise ValueError('Final model coverage has an extra or duplicate identity')
        if (row.get('evaluation') != 'V_select' or row.get('independent_FP64') != 'PASS'
                or row.get('status','COMPLETE') not in ('COMPLETE','REUSED_VERIFIED','HISTORICAL_VERIFIED')):
            raise ValueError('Final cost requires complete independently verified V accuracy')
        if not row.get('prediction_binding') or (family != 'RAW' and not row.get('bundle_binding')):
            raise ValueError('Final cost lacks exact prediction/bundle bindings')
        actual.add(role);identifiers.add(row['run_id']);selected.append(dict(row))
    if actual != expected:
        raise ValueError('Final model coverage is incomplete')
    return sorted(selected,key=lambda row:(row['dataset'],row['family'],row.get('revin',False),row['seed']))


def _cache_root(path, candidate, ctx, inventory):
    for item in inventory.get('baselines',[]):
        if item['dataset'] == candidate['dataset'] and item['family'] == candidate['family']:
            root = Path(item['cache_root']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Baseline bundle is outside its bound cache root')
            return root
    roots = [Path(ctx['cache']),Path(ctx['repo'])/'.cache/adapter_diagnostics_20261017_v2',
             Path(ctx['repo'])/'.cache/adapter_improvements_20261018_v1']
    for root in roots:
        if path.is_relative_to(root.resolve()):
            return root.resolve()
    raise ValueError('Cost bundle has no registered campaign cache root')


def route_loader(candidate, task, ctx, inventory):
    modules = ctx['modules']
    deployment = modules['deployment']
    bundle_path = bundle = None
    if candidate['family'] == 'RAW':
        spec = dict(family='RAW',run_id=candidate['run_id'],dataset=candidate['dataset'],
                    foundation_sha=modules['legacy'].F_SHA,canonical_fit_end=task['split_ranges']['A_stop'][1])
        root = Path(ctx['cache'])
        use_revin = False
    else:
        bundle_path = contracts.resolve_binding(candidate['bundle_binding'],ctx)
        bundle = torch.load(bundle_path,map_location='cpu',weights_only=False)
        if bundle['spec']['family'] != candidate['family']:
            raise ValueError('Cost bundle family differs from scored row')
        for key in ('source_sha','config_sha','data_sha','model_sha'):
            recorded = candidate.get(key)
            if recorded not in (None,'','UNKNOWN') and recorded != bundle['contract'][key]:
                raise ValueError('Cost bundle differs from scored accuracy contract: '+key)
            if recorded in (None,'','UNKNOWN'):
                candidate[key] = bundle['contract'][key]
        root = _cache_root(bundle_path,candidate,ctx,inventory)
        spec = dict(bundle['spec'],run_id=candidate['run_id'],
                    artifact_cache_relative_path=bundle_path.relative_to(root).as_posix(),
                    artifact_sha256=contracts.sha(bundle_path))
        use_revin = 'revin' in bundle['spec']
        if use_revin and candidate['family'] not in ('B2','B0'):
            raise ValueError('Only B2/B0 may use the RevIN deployment factory')
        candidate['base_training_device'] = bundle['spec'].get('device','UNKNOWN')
        candidate['serialized_adapter_bytes'] = bundle_path.stat().st_size
        candidate['serialized_base_checkpoint_bytes'] = bundle_path.stat().st_size
    spec.update(device='cuda',row_microbatch=128,F_row_microbatch=128,checkpoint_inputs=False)
    model_path = Path(ctx['asset'])/modules['legacy'].MODEL_REL

    def load():
        if bundle_path is not None:
            contracts.resolve_binding(candidate['bundle_binding'],ctx)
        pool = deployment.FoundationPool(model_path,'cuda',factory=lambda p,h,m,d:ctx['foundation_loader'](ctx,modules,h,d))
        route = None
        try:
            if use_revin:
                route = load_route(spec,task,root,ctx=ctx,device='cuda',foundation_pool=pool)
            else:
                route = deployment.Route(spec,task,root,device='cuda',foundation_pool=pool)
            route.owns_pool = True
            scaler = modules['fit'].scaler_fingerprint(route.mean,route.std)
            if candidate.get('scaler_sha256') not in (None,scaler):
                raise ValueError('Cost physical route scaler differs from scored accuracy')
            frozen = getattr(pool,'foundation',None)
            candidate['loaded_F_state_sha256'] = getattr(frozen,'initial_hash','NO_FOUNDATION_INSTANTIATED')
            return route
        except BaseException:
            if route is not None:
                route.close()
            else:
                pool.close()
            raise
    return load


def _check(ctx):
    if time.time() >= ctx.get('deadline',1e12):
        raise TimeoutError('Registered cost deadline reached')
    if ctx.get('lease') is not None:
        ctx['lease'].check()


def measure_one(candidate, task, saved, loader, *, batch, ctx, device='cuda', measure=None):
    candidate.setdefault('accuracy_source_kind',candidate.get('source_kind','VERIFIED_FINAL_ACCURACY_ROW'))
    evidence = list(candidate.get('accuracy_source_evidence',[]))
    if candidate.get('prediction_manifest_binding') and candidate['prediction_manifest_binding'] not in evidence:
        evidence.append(candidate['prediction_manifest_binding'])
    candidate['accuracy_source_evidence'] = evidence
    result = old_cost.measure_candidate(candidate,task,saved,loader,batch=batch,device=device,
        lease=ctx.get('lease'),check=lambda:_check(ctx),measure=measure,
        atol=PROTOCOL['atol'],rtol=PROTOCOL['rtol'])
    row = result['row']
    row.update(cap=candidate['cap'],revin=candidate.get('revin',False),final=True,
        job_id=candidate.get('job_id',candidate['run_id']),accuracy_run_id=candidate['run_id'],
        accuracy_cap=candidate['cap'],loaded_F_state_sha256=candidate.get('loaded_F_state_sha256','UNKNOWN'),
        accuracy_prediction_model_label_sha256=candidate.get('prediction_model_label_sha256','NOT_APPLICABLE'),
        whole_path=PROTOCOL['whole_path'],cost_protocol=PROTOCOL,
        cost_id=candidate['run_id']+('_CUDA_B' if str(device).startswith('cuda') else '_CPU_B')+str(batch),
        source_kind='NEW_FINAL_ACTUAL_GPU_MEASUREMENT' if str(device).startswith('cuda') else 'NEW_ACTUAL_CPU_MEASUREMENT')
    if str(device).startswith('cuda'):
        row['hardware'] = torch.cuda.get_device_name(device)
    return result


def validate_complete_receipt(receipt, identity, ctx):
    if receipt.get('identity') != identity:
        raise ValueError('Cost receipt identity differs from current final accuracy/protocol')
    row = receipt.get('row',{})
    if (receipt.get('status') != 'COMPLETE' or row.get('status') != 'COMPLETE'
            or row.get('cost_valid') is not True):
        raise ValueError('Terminal incomplete cost receipt cannot be overwritten or silently retried')
    for key in ('accuracy_run_id','cap','batch_size'):
        if row.get(key) != identity[key]:
            raise ValueError('Completed cost row identity differs: '+key)
    if row.get('device') != 'cuda':
        raise ValueError('Final measured cost receipt must be CUDA')
    for key in ('prediction_binding','bundle_binding'):
        if identity.get(key):
            contracts.resolve_binding(identity[key],ctx)
    contracts.resolve_binding(receipt['trials_binding'],ctx)
    return row


def _terminal_process(pid, create_time):
    if type(pid) is not int or pid <= 0 or float(create_time) <= 0:
        raise ValueError('Recorded cost worker process identity is invalid')
    evidence = dict(pid=pid,create_time=float(create_time),checked_utc=contracts.utc())
    try:
        process = psutil.Process(pid)
        observed = process.create_time()
        if abs(observed-float(create_time)) >= .001:
            return dict(evidence,terminal=True,reason='PID now belongs to another process',observed_create_time=observed)
        if process.is_running():
            raise RuntimeError('Recorded cost worker or authorized child is still alive')
        return dict(evidence,terminal=True,reason='Recorded process is no longer running')
    except psutil.NoSuchProcess:
        return dict(evidence,terminal=True,reason='Recorded PID is absent')
    except psutil.AccessDenied as error:
        raise RuntimeError('Recorded cost worker liveness cannot be established') from error


def _trial_path(receipt_path, attempt):
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Positive cost attempt number required')
    path = Path(receipt_path)
    return path.with_suffix('.trials.csv') if attempt == 1 else path.with_name(path.stem+f'.attempt{attempt:03d}.trials.csv')


def _preserve_bytes(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Immutable previous cost attempt bytes changed')
        return
    with path.open('xb') as handle:
        handle.write(content);handle.flush();os.fsync(handle.fileno())


def resume_cost_attempt(path, identity, ctx):
    """Register a full same-facet retry only after proving the old owner terminal."""
    path = Path(path).resolve()
    if not path.is_relative_to((Path(ctx['report'])/'costs').resolve()):
        raise ValueError('Cost attempt receipt is outside the campaign report')
    previous_bytes = path.read_bytes()
    receipt = json.loads(previous_bytes)
    if receipt.get('identity') != identity:
        raise ValueError('Cost retry identity differs from frozen input/source/protocol')
    if receipt.get('status') not in ('RUNNING','RETRY_REGISTERED'):
        raise ValueError('Completed/failed terminal cost receipt cannot be replaced by retry registration')
    for key in ('prediction_binding','bundle_binding','decisions_binding'):
        if identity.get(key):
            contracts.resolve_binding(identity[key],ctx)
    if receipt['status'] == 'RETRY_REGISTERED':
        registration = contracts.read(contracts.resolve_binding(receipt['retry_registration_binding'],ctx))
        archived = contracts.read(contracts.resolve_binding(receipt['previous_running_binding'],ctx))
        if (registration['identity'] != identity or registration['attempt'] != receipt['attempt']
                or archived['identity'] != identity or archived['status'] != 'RUNNING'):
            raise ValueError('Registered retry lineage identity changed')
        _terminal_process(archived['worker_pid'],archived['worker_create_time'])
        for child in registration['authorized_child_evidence']:
            _terminal_process(child['pid'],child['create_time'])
        for value in registration['orphan_trial_bindings']:
            contracts.resolve_binding(value,ctx)
        return receipt
    owner = _terminal_process(receipt['worker_pid'],receipt['worker_create_time'])
    children = list(receipt.get('authorized_children',[]))
    active_path = Path(ctx['cache'])/'ACTIVE_GPU_LEASE.json'
    if active_path.exists():
        active = contracts.read(active_path)
        same_owner = (active['pid']==receipt['worker_pid'] and
                      abs(float(active['create_time'])-float(receipt['worker_create_time']))<.001)
        same_lease = receipt.get('lease_id') is not None and active.get('lease_id')==receipt['lease_id']
        if same_owner or same_lease:
            children.extend(child for child in active.get('authorized_children',[]) if child not in children)
    child_evidence = [_terminal_process(child['pid'],child['create_time']) for child in children]
    previous_attempt = receipt.get('attempt',1)
    trial = _trial_path(path,previous_attempt)
    orphan_trials = [contracts.binding(trial,ctx)] if trial.exists() else []
    digest = hashlib.sha256(previous_bytes).hexdigest()
    attempt = previous_attempt+1
    archive = path.parent/'attempts'/(path.stem+f'_attempt{previous_attempt:03d}_'+digest+'.json')
    _preserve_bytes(archive,previous_bytes)
    archive_binding = contracts.binding(archive,ctx)
    registration_path = path.parent/'attempts'/(path.stem+f'_attempt{attempt:03d}_RETRY_REGISTRATION.json')
    if registration_path.exists():
        registration = contracts.read(registration_path)
        if (registration.get('identity')!=identity or registration.get('attempt')!=attempt
                or registration.get('previous_running_binding')!=archive_binding
                or registration.get('orphan_trial_bindings')!=orphan_trials):
            raise ValueError('Immutable cost retry registration changed')
    else:
        registration = dict(status='REGISTERED_RETRY',identity=identity,attempt=attempt,
            previous_running_binding=archive_binding,previous_worker_evidence=owner,
            authorized_child_evidence=child_evidence,orphan_trial_bindings=orphan_trials,
            registered_utc=contracts.utc(),restart_scope='entire same facet; same inputs/source/protocol',
            previous_measurement_disposition='Interrupted measurement is preserved, not a completed cost result')
        contracts.sealed_json(registration_path,registration)
    retry = dict(status='RETRY_REGISTERED',identity=identity,attempt=attempt,
                 previous_running_binding=archive_binding,retry_registration_binding=contracts.binding(registration_path,ctx))
    if path.read_bytes() != previous_bytes:
        raise RuntimeError('Cost attempt receipt changed during recovery')
    contracts.atomic_json(path,retry)
    return retry


def _write_csv(path, rows, *, exclusive=False):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['status']
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
                             if isinstance(value,(dict,list)) else value for key,value in row.items()})


def _sources(ctx):
    paths = [Path(__file__),Path(old_cost.__file__),Path(ctx['modules']['fit'].__file__).parent/'measure_costs.py']
    return {path.resolve().relative_to(Path(ctx['repo']).resolve()).as_posix():contracts.sha(path) for path in paths}


def _saved(candidate,ctx):
    path = contracts.resolve_binding(candidate['prediction_binding'],ctx)
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def _ridge_context(rows,ctx):
    ridge = [row for row in rows if row['final'] is True and row['dataset']=='jena' and row['family']=='RAW_RIDGE']
    if len(ridge) != 1:
        raise ValueError('One independently verified Jena Ridge reference required')
    candidate = ridge[0]
    contracts.resolve_binding(candidate['prediction_binding'],ctx)
    contracts.resolve_binding(candidate['bundle_binding'],ctx)
    path = Path(ctx['repo'])/'reports/adapter_diagnostics_20261017_v2/COST_FULL_PRECISION.csv'
    with path.open(encoding='utf-8-sig',newline='') as handle:
        old = [row for row in csv.DictReader(handle) if row['job_id']=='E01_jena_matched'
               and row['method']=='Ridge' and row['device']=='cpu']
    if len(old) != 2 or {float(row['B']) for row in old} != {1.,8.}:
        raise ValueError('Exact historical Ridge CPU batch coverage missing')
    result=[]
    for item in old:
        if (item['accuracy_archive_sha256'] != candidate['prediction_binding']['sha256']
                or item['status']!='COMPLETE' or item['parity_state']!='PASS' or item['source_forward_parity']!='PASS'):
            raise ValueError('Historical Ridge accuracy/parity binding differs')
        source = Path(ctx['repo'])/item['source_file']
        if not source.resolve().is_relative_to(Path(ctx['repo']).resolve()) or contracts.sha(source)!=item['source_sha256']:
            raise ValueError('Historical Ridge cost source bytes changed')
        batch=int(float(item['B']))
        result.append(dict(run_id=candidate['run_id'],accuracy_run_id=candidate['run_id'],accuracy_cap=0,cap=0,
            method='Ridge',family='RAW_RIDGE',dataset='jena',seed=candidate['seed'],device='cpu',batch_size=batch,
            median_ms_per_origin=float(item['median_ms_per_origin']),median_batch_ms=float(item['median_ms_per_batch']),
            IQR_ms_per_origin=float(item['IQR_ms_per_origin']),p95_ms_per_origin=None,
            status='REUSED_VERIFIED',cost_valid=True,final=True,cost_id='HISTORICAL_E01_RIDGE_CPU_B'+str(batch),
            source_kind='HISTORICAL_CPU_CONTEXT',whole_path=item['scope'],core_dtype='float64',cpu_blas_threads=2,
            warmups=int(item['warmups']),timed_trials=int(item['repetitions']),F_params=0,adapter_params=int(item['trainable_params']),
            accuracy_prediction_binding=candidate['prediction_binding'],accuracy_bundle_binding=candidate['bundle_binding'],
            original_cost_table_binding=contracts.binding(path,ctx),original_cost_receipt_binding=contracts.binding(source,ctx),
            source_file=item['source_file'],source_sha256=item['source_sha256'],
            full_V_double_reload_parity='NOT_AVAILABLE_HISTORICAL',
            cost_protocol_scope='Historical CPU whole path, exact archive binding; separate from new GPU protocol; p95 unavailable'))
    return result


def execute_costs(ctx):
    from . import run
    verified_path = Path(ctx['cache'])/'VERIFIED_ROWS.json'
    decisions_path = Path(ctx['report'])/'DECISIONS.json'
    rows,decisions = contracts.read(verified_path),contracts.read(decisions_path)
    selected = select_final_rows(rows,decisions)
    inventory = contracts.read(Path(ctx['cache'])/'LEGACY_INVENTORY.json')
    sources = _sources(ctx)
    tasks = {cohort:ctx['task_loader'](ctx,ctx['modules'],cohort) for cohort in contracts.COHORTS}
    costs = _ridge_context(rows,ctx)
    proof_rows=[]
    with run.lease(ctx,'cost'):
        for candidate in selected:
            for batch in PROTOCOL['batches']:
                name = candidate['run_id']+'_B'+str(batch)
                if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
                    raise ValueError('Unsafe accuracy/cost filename')
                path = Path(ctx['report'])/'costs'/(name+'.json')
                identity = dict(accuracy_run_id=candidate['run_id'],cap=candidate['cap'],batch_size=batch,
                    prediction_binding=candidate['prediction_binding'],bundle_binding=candidate.get('bundle_binding'),
                    source_hashes=sources,protocol=PROTOCOL,decisions_binding=contracts.binding(decisions_path,ctx))
                attempt,retry = 1,None
                if path.exists():
                    previous = contracts.read(path)
                    if previous.get('status') == 'COMPLETE':
                        row = validate_complete_receipt(previous,identity,ctx)
                        receipt_binding = contracts.binding(path,ctx)
                        costs.append(dict(row,source_sha256=receipt_binding['sha256'],cost_receipt_binding=receipt_binding))
                        proof_rows.append(receipt_binding)
                        print('COST_REUSED '+name,flush=True)
                        continue
                    retry = resume_cost_attempt(path,identity,ctx)
                    attempt = retry['attempt']
                _check(ctx)
                started=time.time()
                lineage = {key:retry[key] for key in ('previous_running_binding','retry_registration_binding')} if retry else {}
                running = dict(status='RUNNING',identity=identity,attempt=attempt,started_utc=contracts.utc(),
                    worker_pid=psutil.Process().pid,worker_create_time=psutil.Process().create_time(),
                    authorized_children=list(getattr(ctx['lease'],'_authorized_children',{}).values()),
                    lease_id=getattr(ctx['lease'],'lease_id',None),**lineage)
                if retry:
                    contracts.atomic_json(path,running)
                else:
                    contracts.sealed_json(path,running)
                print('COST_START '+name,flush=True)
                stop=threading.Event()

                def heartbeat():
                    while not stop.wait(20):
                        print('COST_HEARTBEAT '+name+' elapsed_seconds='+str(round(time.time()-started,1)),flush=True)

                thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
                try:
                    task=tasks[candidate['dataset']]
                    saved=_saved(candidate,ctx)
                    loader=route_loader(candidate,task,ctx,inventory)
                    result=measure_one(candidate,task,saved,loader,batch=batch,ctx=ctx)
                    row=result['row']
                    if row.get('status')!='COMPLETE' or row.get('cost_valid') is not True:
                        raise ValueError('Final physical-route/reload parity failed')
                    if _sources(ctx)!=sources:
                        raise ValueError('Cost scientific source changed during measurement')
                    trial_path=_trial_path(path,attempt)
                    _write_csv(trial_path,result['trials'],exclusive=True)
                    row.update(source_file=path.relative_to(Path(ctx['repo'])).as_posix(),
                        new_measured=True,accuracy_inputs_binding=contracts.binding(verified_path,ctx))
                    receipt=dict(status='COMPLETE',identity=identity,row=row,trials_binding=contracts.binding(trial_path,ctx),
                                 attempt=attempt,completed_utc=contracts.utc(),consumed_wall_seconds=time.time()-started,**lineage)
                    contracts.atomic_json(path,receipt)
                    receipt_binding = contracts.binding(path,ctx)
                    costs.append(dict(row,source_sha256=receipt_binding['sha256'],cost_receipt_binding=receipt_binding))
                    proof_rows.append(receipt_binding)
                    print('COST_COMPLETE '+name+' median_ms_per_origin='+str(row['median_ms_per_origin']),flush=True)
                except BaseException as error:
                    reason=str(error)
                    for key in ('cache','report','asset','repo'):
                        reason=reason.replace(str(ctx[key]),'{'+key.upper()+'_ROOT}')
                    contracts.atomic_json(path,dict(status='FAILED',identity=identity,error_class=type(error).__name__,
                        attempt=attempt,reason=reason,finished_utc=contracts.utc(),consumed_wall_seconds=time.time()-started,**lineage))
                    raise
                finally:
                    stop.set();thread.join(timeout=1)
    _write_csv(Path(ctx['report'])/'COST.csv',costs)
    gpu_rows = [row for row in costs if row['device']=='cuda']
    cpu_rows = [row for row in costs if row['source_kind']=='HISTORICAL_CPU_CONTEXT']
    if len(gpu_rows)!=28 or len(cpu_rows)!=2 or len({row['cost_id'] for row in costs})!=30:
        raise ValueError('Completed final cost coverage differs from registered 28 GPU and 2 contextual CPU rows')
    validation=dict(status='PASS',new_GPU_rows=len(gpu_rows),historical_CPU_context_rows=len(cpu_rows),
                    expected_whole_path_GPU_rows=28,all_measurements_source_bound=True,
                    accuracy_inputs_binding=contracts.binding(verified_path,ctx),decisions_binding=contracts.binding(decisions_path,ctx),
                    source_hashes=sources,protocol=PROTOCOL,cost_receipts=proof_rows)
    contracts.sealed_json(Path(ctx['report'])/'COST_VALIDATION.json',validation)
    return costs
