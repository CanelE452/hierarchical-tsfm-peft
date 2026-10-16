"""Borrow an existing ROOT audit owner; default CLI is metadata only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PINS_REL='diagnostics/multik_execution_ready_1792086454973961300/INPUT_PINS.json'
PINS_SHA='b520924528606a1f2eb83f6302a247cfd9b1f4e5a179adb2f1ff83b5fc3c2ad0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def inside(root,relative):
    root=Path(root).resolve();relative=Path(relative);path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path==root:raise ValueError('Escaped relative binding')
    return path

def metadata_plan(asset_root,cache_root,public_root):
    """Only JSON/CSV and streaming file hashes; no numpy/torch imports."""
    asset,cache,public=map(lambda x:Path(x).resolve(),(asset_root,cache_root,public_root))
    if cache!=asset/'.cache'/public.name:raise ValueError('Cache differs from approved campaign root')
    path=inside(public,PINS_REL)
    if sha(path)!=PINS_SHA:raise ValueError('Pinned preparation metadata changed')
    pins=read(path)
    roots={'asset':asset,'cache':cache,'public':public,'repo':public.parents[1]}
    for entry in pins['bindings']:
        path=inside(roots[entry['scope']],entry['relative_path'])
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Pinned binding changed: '+entry['relative_path'])
    for entry in pins['plans'].values():
        if sha(inside(public,entry['public_relative_path']))!=entry['sha256']:raise ValueError('Plan changed')
    with (public/'EXPERIMENT_REGISTRY.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    byid={r['experiment_id']:r for r in rows}
    if len(byid)!=len(rows):raise ValueError('Duplicate registered IDs')
    for identity,route in pins['routes'].items():
        if identity not in byid or any(byid[identity].get(k)!=v for k,v in route['registry_contract'].items()):raise ValueError('Pinned route scientific contract changed')
    if len(pins['queue'])!=20 or len({x['scope_id'] for x in pins['queue']})!=20:raise ValueError('Full20 scope queue required')
    return pins

def _ledger(ks,rows):
    result=[]
    for phase,count in [('warmup',10),('timed',20)]:
        for a in ks:
            for b in ks:
                if a==b:continue
                group=[x for x in rows if x.get('phase')==phase and x.get('from_K')==a and x.get('to_K')==b]
                within=sum(bool(x.get('resource_within_limit',True)) for x in group)
                result.append(dict(phase=phase,from_K=a,to_K=b,requested=count,actual_complete=len(group),remaining=count-len(group),completed_within_resource=within,remaining_within_resource=count-within))
    return result

def classify_scope(folder,strategy,code,ks):
    folder=Path(folder);bp=folder/'BATCH_RECEIPT.json';rp=folder/strategy/'RESULT.json'
    batch=read(bp) if bp.is_file() else {};result=read(rp) if rp.is_file() else {}
    rows=result.get('rows',[]);ledger=_ledger(ks,rows)
    resource_stop=bool(result.get('stop_new_scopes') or batch.get('resource_limit_stopped_new_scopes') or result.get('resource_events') or result.get('resource_status') in ['DEADLINE_OVERRUN','RESOURCE_GUARD_TRIGGERED','OWNER_UNAVAILABLE'])
    files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}
    checks=[]
    if len(rows)==180:
        actuals=sorted((folder/strategy/'actual_trials').glob('trial_*.json'))
        checks.append([read(p) for p in actuals]==rows)
        csv_path=folder/strategy/'TRIALS.csv'
        if csv_path.is_file():
            with csv_path.open(encoding='utf-8',newline='') as f:csv_rows=list(csv.DictReader(f))
            checks.append(csv_rows==[{k:'' if v is None else str(v) for k,v in row.items()} for row in rows])
        else:checks.append(False)
        recorded=result.get('transition_ledger',[])
        checks.append(len(recorded)==12 and all(all(row.get(k)==v for k,v in expected.items()) for row,expected in zip(recorded,ledger)))
    else:checks.append(False)
    complete=bool(code==0 and batch.get('status')=='COMPLETE_REQUESTED_STRATEGIES' and result.get('strategy')==strategy and result.get('status')=='COMPLETE_PROTOCOL' and result.get('exact_minimum_protocol_complete') and result.get('numeric_execution_status')=='COMPLETE_REQUESTED_TRIALS' and result.get('resource_status')=='WITHIN_OBSERVED_BOUNDARIES' and result.get('post_measure_validation_status')=='PASS' and not resource_stop and len(rows)==180 and all(checks) and all(x['remaining']==0 and x['remaining_within_resource']==0 for x in ledger))
    return dict(exact_minimum_protocol_complete=complete,actual_process_exit_code=code,actual_result_status=result.get('status','MISSING_RESULT'),actual_batch_status=batch.get('status','MISSING_BATCH_RECEIPT'),numeric_execution_status=result.get('numeric_execution_status','UNKNOWN_NO_RESULT'),resource_status=result.get('resource_status','UNKNOWN_NO_RESULT'),post_measure_validation_status=result.get('post_measure_validation_status','UNKNOWN_NO_RESULT'),actual_numeric_trials=len(rows),actual_resource_valid_trials=sum(bool(x.get('resource_within_limit',True)) for x in rows),resource_stop=resource_stop,transition_ledger=ledger,result_files_sha256=files,durable_numeric_integrity_passed=all(checks),numeric_errors_do_not_stop_independent_scopes=True)

def _validate_bindings(folder,scope,pins):
    result=read(Path(folder)/scope['strategy']/'RESULT.json')
    if result['dataset']!=scope['dataset'] or result['batch']!=scope['batch']:raise ValueError('Measured dataset/batch differs')
    for k,identity in scope['route_ids'].items():
        binding=result['accuracy_bindings'][str(k)];route=pins['routes'][identity]
        for key,expected in [('accuracy_run_id',identity),('accuracy_manifest_sha256',route['manifest_sha256']),('accuracy_prediction_model_sha',route['selected_model_sha']),('data_sha256',route['data_sha']),('scaler_sha',route['scaler_sha']),('foundation_state_sha256',pins['foundation_state_sha256'])]:
            if binding.get(key)!=expected:raise ValueError('Measured canonical accuracy binding differs: '+key)
    return True

def _owner(lease,pins,cache):
    if lease is None or getattr(lease,'purpose',None)!='audit' or not getattr(lease,'_entered',False) or getattr(getattr(lease,'lock',None),'file',None) is None or getattr(lease,'pid',None)!=os.getpid():raise ValueError('Existing active same-PID ROOT audit owner required; no new/nested owner')
    lease.check()
    from resources import gpu_usage_seconds
    closed=gpu_usage_seconds(cache)
    latest=lease.started_unix+max(0.,pins['owner_audit_cumulative_ceiling_seconds']-closed)
    if lease.total_gpu_seconds>pins['global_GPU_ceiling_seconds'] or lease.wall_deadline>pins['hard_wall_deadline_unix'] or lease.deadline>min(latest,pins['hard_wall_deadline_unix'])+.001:raise ValueError('Owner deadline does not preserve campaign/final reserve ceiling')
    return closed

def _storage(pins,cache,public):
    size=sum(p.stat().st_size for root in [cache,public] for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    pending=pins['pending_storage_allowance_bytes']
    if size+pending>pins['combined_storage_limit_bytes'] or min(shutil.disk_usage(cache).free,shutil.disk_usage(public).free)<pending:raise ValueError('Combined cache/public pending storage unavailable')
    return size

def _resume(path,pins,public):
    if path is None:return {},None
    path=Path(path).resolve()
    if not path.is_relative_to(public):raise ValueError('Resume batch receipt must be public-relative')
    prior=read(path)
    if prior['pins_sha256']!=PINS_SHA:raise ValueError('Resume uses different inputs')
    rows=prior['scope_ledger']
    expected=[x['scope_id'] for x in pins['queue']]
    if [x['scope_id'] for x in rows]!=expected:raise ValueError('Resume queue differs')
    accepted={}
    for row in rows:
        if not row.get('exact_minimum_protocol_complete'):continue
        folder=inside(public,row['output_public_relative_path'])
        scope=next(s for s in pins['queue'] if s['scope_id']==row['scope_id'])
        actual=classify_scope(folder,scope['strategy'],row['actual_process_exit_code'],scope['K_grid'])
        if not actual['exact_minimum_protocol_complete'] or actual['result_files_sha256']!=row['result_files_sha256']:raise ValueError('Previously completed scope changed')
        accepted[row['scope_id']]=row
    return accepted,{'public_relative_path':path.relative_to(public).as_posix(),'sha256':sha(path)}

def _run_child(cmd,env,cwd,log):
    with Path(log).open('xb') as f:
        process=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        try:return process.wait()
        except BaseException:
            new(Path(log).with_suffix('.WAIT_INTERRUPTED.json'),{'pid':process.pid,'observed_exit_code':process.poll(),'termination_requested':False,'child_state_must_be_checked_before_any_new_GPU_work':True})
            raise

def run_with_owner(lease,asset_root,cache_root,public_root,*,max_seconds=None,resume_from=None):
    """No owner creation/closing. Local max_seconds gates admissions only."""
    asset,cache,public=map(lambda p:Path(p).resolve(),(asset_root,cache_root,public_root))
    if lease is None or getattr(lease,'purpose',None)!='audit' or getattr(lease,'pid',None)!=os.getpid():raise ValueError('Existing ROOT audit owner required before preflight')
    lease.check();pins=metadata_plan(asset,cache,public)
    sys.path.insert(0,str(public))
    closed_before=_owner(lease,pins,cache);storage_before=_storage(pins,cache,public)
    if max_seconds is not None and (not 0<float(max_seconds)<float('inf')):raise ValueError('Positive finite admission window required')
    previous,resume_binding=_resume(resume_from,pins,public)
    stamp=str(time.time_ns());private=cache/'final_execution'/('multik_runs_'+stamp);private.mkdir(parents=True,exist_ok=False)
    output=public/'costs'/('multik_transition_'+stamp);output.mkdir(parents=True,exist_ok=False)
    started=time.time();admit_until=None if max_seconds is None else started+float(max_seconds)
    python=asset/'.venv/Scripts/python.exe'
    if not python.is_file():raise FileNotFoundError('Original project Python is absent')
    ledger=[];stop=None;current=None;owner_resource_events=[]
    new(output/'FULL_QUEUE_PLAN.json',dict(pins_sha256=PINS_SHA,queue=pins['queue'],status='PLAN_ONLY_BEFORE_CHILD_EXECUTION',requested_scopes=20,requested_trials=3600,owner_deadline_unix=lease.deadline,local_admission_deadline_unix=admit_until,max_seconds_is_admission_only=True,child_environment_secret_values_not_recorded=True))
    try:
        for index,scope in enumerate(pins['queue']):
            identity=scope['scope_id'];row=dict(scope,status='NOT_STARTED',exact_minimum_protocol_complete=False,actual_numeric_trials=0,actual_resource_valid_trials=0,transition_ledger=_ledger(scope['K_grid'],[]))
            ledger.append(row)
            if identity in previous:
                row.update(previous[identity],status='REUSED_VERIFIED_COMPLETE',reused_from=resume_binding,child_start_attempted_in_this_call=False,fresh_child_completed_in_this_call=False)
                continue
            if stop is not None:
                row['status']='NOT_STARTED_'+stop;continue
            try:
                lease.check()
                if admit_until is not None and time.time()>=admit_until:stop='LOCAL_ADMISSION_WINDOW';row['status']='NOT_STARTED_'+stop;continue
                if lease.remaining<15:stop='OWNER_RESOURCE_LIMIT';row['status']='NOT_STARTED_'+stop;continue
                metadata_plan(asset,cache,public);_storage(pins,cache,public)
                lease.check()
                if admit_until is not None and time.time()>=admit_until:stop='LOCAL_ADMISSION_WINDOW';row['status']='NOT_STARTED_'+stop;continue
                if lease.remaining<15:stop='OWNER_RESOURCE_LIMIT';row['status']='NOT_STARTED_'+stop;continue
            except Exception as error:
                stop='PREFLIGHT_OR_RESOURCE_BOUNDARY';row.update(status='NOT_STARTED_'+stop,error_type=type(error).__name__);continue
            child_output=output/identity;log=private/(identity+'.log');current=row
            cmd=[str(python),'-X','utf8','-u','-B',str(public/'measure_multik_transition.py'),'--worker','--plan-json',str(inside(public,scope['plan_public_relative_path'])),'--public-root',str(public),'--cache-root',str(cache),'--model',str(inside(asset,pins['foundation_snapshot_asset_relative_path'])),'--device','cuda','--batch',str(scope['batch']),'--strategy',scope['strategy'],'--warmups','10','--timed-trials','20','--output',str(child_output)]
            env=dict(os.environ,**lease.child_environment(),INFORMS_ASSET_ROOT=str(asset),INFORMS_CACHE_ROOT=str(cache),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            begin=time.time();row.update(status='ADMITTED_CHILD_PENDING',started_unix=begin,output_public_relative_path=child_output.relative_to(public).as_posix(),private_log_cache_relative_path=log.relative_to(cache).as_posix(),child_start_attempted_in_this_call=True)
            new(output/f'SCOPE_ADMISSION_{index:02}.json',dict(row,owner_deadline_unix=lease.deadline,fresh_process=True,requested_trials=180))
            code=_run_child(cmd,env,public,log)
            row.update(actual_process_exit_code=code,child_wall_seconds=time.time()-begin,finished_unix=time.time(),private_log_sha256=sha(log),fresh_child_completed_in_this_call=True)
            try:row.update(classify_scope(child_output,scope['strategy'],code,scope['K_grid']))
            except Exception as error:row.update(status='FAILED_OUTPUT_VALIDATION',output_validation_error_type=type(error).__name__)
            else:
                row['status']='COMPLETE_EXACT_PROTOCOL' if row['exact_minimum_protocol_complete'] else 'INCOMPLETE_OR_FAILED'
                if row['exact_minimum_protocol_complete']:
                    try:row['canonical_bindings_match_preflight']=_validate_bindings(child_output,scope,pins)
                    except Exception as error:row.update(status='FAILED_CANONICAL_METADATA_BINDING',exact_minimum_protocol_complete=False,binding_error_type=type(error).__name__)
            if row.get('resource_stop'):stop='OWNER_RESOURCE_LIMIT'
            try:lease.check()
            except Exception as error:
                stop='OWNER_RESOURCE_LIMIT';row['owner_after_child_error_type']=type(error).__name__
                owner_resource_events.append(dict(boundary='after_child',scope_id=identity,error_type=type(error).__name__,detected_unix=time.time()))
            new(output/f'SCOPE_RESULT_{index:02}.json',row);current=None
    except BaseException as error:
        stop='WAIT_OR_CONTROLLER_EXCEPTION'
        if current is not None:current.update(status='WAIT_INTERRUPTED_CHILD_STATE_UNCONFIRMED',controller_error_type=type(error).__name__,termination_requested=False)
        raise
    finally:
        for scope in pins['queue'][len(ledger):]:ledger.append(dict(scope,status='NOT_STARTED_'+str(stop),exact_minimum_protocol_complete=False,actual_numeric_trials=0,actual_resource_valid_trials=0,transition_ledger=_ledger(scope['K_grid'],[])))
        try:metadata_plan(asset,cache,public);postflight='PASS'
        except Exception as error:postflight='FAILED_'+type(error).__name__
        try:lease.check();owner_final_check='PASS'
        except Exception as error:
            owner_final_check='FAILED_'+type(error).__name__;stop='OWNER_RESOURCE_LIMIT'
            owner_resource_events.append(dict(boundary='after_final_metadata_postflight',error_type=type(error).__name__,detected_unix=time.time()))
        resource_status='OWNER_RESOURCE_LIMIT' if stop=='OWNER_RESOURCE_LIMIT' or owner_resource_events else 'WITHIN_OBSERVED_OWNER_BOUNDARIES'
        complete=postflight=='PASS' and owner_final_check=='PASS' and resource_status=='WITHIN_OBSERVED_OWNER_BOUNDARIES' and all(x.get('exact_minimum_protocol_complete',False) for x in ledger)
        receipt=dict(status='COMPLETE_EXACT20_SCOPES' if complete else 'PARTIAL_OR_FAILED_FULL_QUEUE_PRESERVED',pins_sha256=PINS_SHA,scope_ledger=ledger,resumed_from=resume_binding,requested_scopes=20,complete_scopes=sum(bool(x.get('exact_minimum_protocol_complete')) for x in ledger),actual_numeric_trials=sum(x.get('actual_numeric_trials',0) for x in ledger),actual_resource_valid_trials=sum(x.get('actual_resource_valid_trials',0) for x in ledger),stop_reason=stop,owner_deadline_unix=lease.deadline,local_admission_deadline_unix=admit_until,max_seconds_is_admission_only=True,nonpreemptive_last_child_can_exceed_local_admission_window=True,owner_closed_or_replaced_by_launcher=False,closed_GPU_seconds_before=closed_before,storage_bytes_before=storage_before,started_unix=started,finished_unix=time.time(),public_batch_relative_path=output.relative_to(public).as_posix(),remaining_scope_ids=[x['scope_id'] for x in ledger if not x.get('exact_minimum_protocol_complete')],actual_cost_raw_statistics_complete=complete,training_T_or_registry_mutation_performed=False,old_warmed_costs_combined=False)
        receipt['postflight_input_bindings']=postflight
        receipt['final_owner_check']=owner_final_check
        receipt['resource_status']=resource_status
        receipt['owner_resource_events']=owner_resource_events
        receipt['all_child_protocol_flags_complete']=all(x.get('exact_minimum_protocol_complete',False) for x in ledger)
        receipt['trial_resource_counts_scope']='Per-child observed protocol guards; separate final global owner status gates batch completion'
        receipt['newly_executed_numeric_trials']=sum(x.get('actual_numeric_trials',0) for x in ledger if x.get('fresh_child_completed_in_this_call'))
        receipt['reused_verified_numeric_trials']=sum(x.get('actual_numeric_trials',0) for x in ledger if x.get('status')=='REUSED_VERIFIED_COMPLETE')
        receipt['fresh_children_with_exitcode_this_call']=sum(bool(x.get('fresh_child_completed_in_this_call')) for x in ledger)
        new(output/'BATCH_LAUNCHER_RECEIPT.json',receipt)
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',required=True);parser.add_argument('--cache-root',required=True);parser.add_argument('--public-root',required=True)
    args=parser.parse_args();pins=metadata_plan(args.asset_root,args.cache_root,args.public_root)
    print(json.dumps({'status':'METADATA_PREFLIGHT_ONLY','pins_sha256':PINS_SHA,'scope_count':len(pins['queue']),'route_count':len(pins['routes']),'GPU_children_started':0,'numpy_imported':'numpy' in sys.modules,'torch_imported':'torch' in sys.modules,'run_with_owner_requires_existing_ROOT_audit_lease':True},indent=2))

if __name__=='__main__':main()
