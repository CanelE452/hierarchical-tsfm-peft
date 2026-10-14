"""Preserve the original 404 rows and measure only its 132 missing rows.

The frozen benchmark forward, warmups, timing scope and physical tolerance are
unchanged. Independent references use the original science API with matching
batch size; original bulk comparisons and their failures remain explicit.
"""
from pathlib import Path
import argparse
import gc
import time
import traceback
import numpy as np
import psutil
import torch
import runtime as rt
import run as ru
import model as bm
import benchmark as bench
import batch_science_reference as bs


class CappedLimit:
    def __init__(self, limit): self.limit = limit
    def check(self):
        accounting = self.limit.check()
        if accounting['elapsed_seconds'] > 300: raise rt.BudgetHeld('COST_RECOVERY_GPU_PROCESS_WALL_300SECONDS')
        return accounting


def validate_reused_rows(original, routes, expected_count=404, expected_total=536):
    if original['status'] != 'FAILED_PRESERVED': raise ValueError('COST_ORIGINAL_FAILURE_NOT_PRESERVED')
    expected = {(r['dataset'],r['identity'],r['route'],b):r['split'] for r in routes for b in (1,8)}
    if len(expected) != expected_total: raise ValueError('COST_EXPECTED_ROUTE_BATCH_KEYS_NOT_EXACT')
    records = original['records']; keys = [bs.row_key(r) for r in records]
    if len(records) != expected_count or len(set(keys)) != expected_count or not set(keys).issubset(expected):
        raise ValueError('COST_REUSED_SUBSET_NOT_EXACT_UNIQUE_POSITIVE')
    for record in records:
        seconds = np.asarray(record['seconds'],np.float64)
        if record['split'] != expected[bs.row_key(record)] or record['warmup'] != 2 or record['repeats'] != 7 or seconds.shape != (7,) or not np.isfinite(seconds).all() or np.any(seconds <= 0):
            raise ValueError('COST_ORIGINAL_TIMING_NOT_VALID_SEVEN_REPEAT')
        if record['parity_pass'] is not True or record['physical_parity_atol'] != .002 or record['physical_parity_rtol'] != 2e-5:
            raise ValueError('COST_ORIGINAL_PARITY_NOT_ORIGINAL_PASSED_GATE')
        if not np.isclose(record['median_seconds'],np.median(seconds),rtol=1e-12,atol=0) or not np.isclose(record['median_seconds_per_window'],np.median(seconds)/record['batch'],rtol=1e-12,atol=0):
            raise ValueError('COST_ORIGINAL_MEDIAN_NOT_ACTUAL_SEVEN_TIMINGS')
    return records, set(expected)-set(keys)


def measure_batch(pack,route,batch,matched,bulk,forecaster,model,ridge):
    split = route['split']; raw=pack['raw_history'][split][:batch]; observed=pack['raw_history_observed'][split][:batch]; origins=pack['origins'][split][:batch]
    if len(raw) != batch: raise ValueError('COST_MISSING_BATCH_WINDOWS')
    outputs=[]; seconds=[]
    with torch.no_grad():
        for _ in range(2): bench.deployment(pack,raw,observed,origins,route['route'],forecaster,model,ridge)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(7):
            torch.cuda.synchronize(); started=time.perf_counter()
            actual=bench.deployment(pack,raw,observed,origins,route['route'],forecaster,model,ridge)
            torch.cuda.synchronize(); seconds.append(time.perf_counter()-started); outputs.append(actual)
    matched_checks=[bs.physical_parity(actual,matched,pack['mean'],pack['std']) for actual in outputs]
    bulk_checks=[bs.physical_parity(actual,bulk,pack['mean'],pack['std']) for actual in outputs]
    norm_check=bs.normalized_reinference(matched,bulk)
    last=matched_checks[-1]
    valid_timings=len(seconds)==7 and np.isfinite(seconds).all() and all(value>0 for value in seconds)
    row={'identity':route['identity'],'dataset':route['dataset'],'route':route['route'],'split':split,
        'batch':batch,'warmup':2,'repeats':7,'seconds':seconds,'median_seconds':float(np.median(seconds)),
        'median_seconds_per_window':float(np.median(seconds))/batch,
        'physical_max_abs_vs_science':last['physical_max_abs_vs_science'],
        'normalized_max_abs_vs_science':last['normalized_max_abs_vs_science'],
        'physical_parity_rtol':2e-5,'physical_parity_atol':.002,
        'parity_pass':bool(all(r['parity_pass'] for r in matched_checks) and norm_check['gate_pass'] and valid_timings),
        'seven_timings_positive_finite':bool(valid_timings),
        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'rss_bytes':psutil.Process().memory_info().rss,
        'model_metadata':None if model is None else model.metadata(),
        'includes':'CPU L512 window-local causal FF/FIT normalization/H2D/active route/inverse FIT normalization/D2H/sync',
        'excludes':'Disk and model load, FIT memory-key precomputation, teacher future AE or removed auxiliary heads',
        'memory_note':'Frozen Bolt and full current checkpoint co-resident; total peak is not exclusive incremental route memory',
        'reference_kind':bs.PROTOCOL['reference_kind'], 'matched_science_to_original_bulk_normalized':norm_check,
        'all_seven_matched_physical_comparisons':matched_checks,'all_seven_original_bulk_physical_comparisons':bulk_checks,
        'original_bulk_physical_gate_pass':bool(all(r['parity_pass'] for r in bulk_checks)),
        'original_bulk_physical_failure_relabelled':False,
        'diagnostic_array_storage_outside_timed_scope':True}
    arrays={'actual_physical':outputs[-1],'all_7_physical_outputs':np.stack(outputs),
        'matched_science_normalized':matched,'matched_science_physical_CPU_FP32':matched*pack['std']+pack['mean'],
        'original_bulk_science_normalized':bulk,'original_bulk_science_physical_CPU_FP32':bulk*pack['std']+pack['mean'],
        'raw_history':raw,'history_observed':observed,'origins':origins,'mean':pack['mean'],'std':pack['std'],
        'source_history_normalized':pack['history'][split][:batch]}
    return row,arrays


def read_source(path,expected_status='COMPLETE'):
    value=rt.read(path)
    if value['status'] != expected_status: raise ValueError('COST_RECOVERY_SOURCE_STATUS: '+str(path))
    return value,{'path':str(path.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(path)}


def preflight(args):
    lock=ru.check_method_sources(args.method_lock)
    method_ref={'path':str(args.method_lock.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(args.method_lock)}
    source_results={}; values={}
    for name in ('dev','fresh','probe','evaluation','cost'):
        value,ref=read_source(getattr(args,name+'_result'),'FAILED_PRESERVED' if name=='cost' else 'COMPLETE')
        if value['source_hashes'] != lock['source_hashes']: raise ValueError('COST_RECOVERY_SOURCE16_NOT_IDENTICAL: '+name)
        values[name]=value; source_results[name]=ref
    dev,fresh,probes,evaluation,original=[values[n] for n in ('dev','fresh','probe','evaluation','cost')]
    global_path=bs.verify_artifact(evaluation['global_model_lock']); global_lock=rt.read(global_path)
    if not global_lock.get('ALL_SELECTED') or global_lock['method_lock_sha256'] != method_ref['sha256'] or global_lock['source_hashes'] != lock['source_hashes']:
        raise ValueError('COST_RECOVERY_GLOBAL_SELECTION_GATE_NOT_IDENTICAL')
    routes,aliases=bs.build_routes(dev,fresh,probes,evaluation,lock)
    reused,missing=validate_reused_rows(original,routes)
    if {key[0] for key in missing} != {'uci274_sml2010'} or len(missing) != 132:
        raise ValueError('COST_RECOVERY_SCOPE_IS_ORIGINAL_MISSING_SML132_ONLY')
    backbone=original['backbone_before_sha256']
    for name in ('dev','fresh','probe','evaluation'):
        if values[name]['backbone_before_sha256'] != backbone or values[name]['backbone_after_sha256'] != backbone:
            raise ValueError('COST_RECOVERY_SOURCE_BACKBONE_NOT_IDENTICAL')
    original_receipt=original['resource_receipt']; resource=rt.read(bs.verify_artifact(original_receipt))
    if resource['status'] != 'FAILED_PRESERVED' or not resource['gpu'] or resource['wall_seconds'] <= 0:
        raise ValueError('COST_RECOVERY_ORIGINAL_GPU_RECEIPT_NOT_FAILED_POSITIVE')
    artifacts={}
    selected={(str((rt.ROOT/a['path']).resolve()),a['sha256']) for a in global_lock['selected_models']}
    for route in routes:
        if any((str((rt.ROOT/a['path']).resolve()),a['sha256']) not in selected for a in route['model_artifacts']):
            raise ValueError('COST_ROUTE_MODEL_NOT_IN_GLOBAL_SELECTION_LOCK')
        for artifact in route['model_artifacts']+[route['science_prediction']]:
            key=(artifact['path'],artifact['sha256'])
            if key not in artifacts: bs.verify_artifact(artifact); artifacts[key]=artifact
    pack,past_evidence=bs.load_past_only('uci274_sml2010',fresh,evaluation,method_ref)
    contract={'status':'SEALED_COST_BATCH_RECOVERY_PROTOCOL','protocol':bs.PROTOCOL,
        'expected_rows':536,'reused_original_rows':404,'new_missing_rows':132,'new_GPU_process_wall_cap_seconds':300,
        'missing_keys':[list(key) for key in sorted(missing)],'new_dataset_scope':['uci274_sml2010'],
        'original_failed_cost':source_results['cost'],'original_gpu_resource_receipt':original_receipt,
        'original_failed_attempt':{'status':original['status'],'error':original.get('error'),
            'error_type':original.get('error_type'),'failed_paths':original.get('failed_paths',[]),
            'source_result':source_results['cost'],'resource_receipt':original_receipt,
            'status_or_objects_rewritten':False},
        'method_lock':method_ref,'global_model_lock':evaluation['global_model_lock'],
        'source_results':source_results,'source_hashes':lock['source_hashes'],'past_only_evidence':past_evidence,
        'EVAL_target_arrays_opened':False,'protected_TEST_opened':False,'new_training_updates':0,
        'timing_resource_accounting':'Original failed GPU process receipt counted once, plus only the new recovery GPU receipt; reused medians are not new process charges',
        'registered_science_artifact_count_verified':len(artifacts)}
    return contract,lock,values,routes,aliases,reused,missing,pack


def unchanged_sources(contract):
    ru.source_check(contract['source_hashes'])
    refs=list(contract['source_results'].values())+[contract['method_lock'],contract['global_model_lock'],contract['original_gpu_resource_receipt'],contract['past_only_evidence']['artifact']]
    refs += contract.get('helper_scripts',[])
    for ref in refs:
        try:bs.verify_artifact(ref)
        except ValueError as exc:raise RuntimeError('ACTIVE_SOURCE_CHANGED: '+ref['path']) from exc


def main():
    parser=argparse.ArgumentParser()
    for name in ('dev-result','fresh-result','probe-result','evaluation-result','cost-result','method-lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tag',required=True); parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('COST_RECOVERY_INVALID_EXCLUSIVE_TAG')
    folder=rt.HERE/'cost_batch_recovery_v1'/args.tag; folder.mkdir(parents=True,exist_ok=False)
    helper_refs=[]
    for path in (Path(__file__),Path(bs.__file__)):
        destination=folder/path.name
        with destination.open('xb') as handle:handle.write(path.read_bytes())
        helper_refs.append({'path':str(path.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(path),
            'snapshot_path':str(destination.relative_to(rt.ROOT)),'snapshot_sha256':rt.sha(destination)})
    raw_limit=rt.Limit(args.tag,gpu=not args.preflight_only); limit=CappedLimit(raw_limit)
    output={'status':'INCOMPLETE','stage':'cost_preflight' if args.preflight_only else 'cost','tag':args.tag,
        'started_unix':time.time(),'records':[],'failed_paths':[],'timing_producers':[],'helper_scripts':helper_refs,
        'new_training_updates':0,'EVAL_target_arrays_opened':False,'protected_TEST_opened':False,
        'new_model_selection':False,'original_bulk_science_metrics_changed':False,'expected_rows':536,'canonical_paths':268}
    terminal,error='COMPLETE',None
    try:
        contract,lock,values,routes,aliases,reused,missing,pack=preflight(args)
        output.update(source_hashes=lock['source_hashes'],source_results=contract['source_results'],method_lock=contract['method_lock'],
            global_model_lock=contract['global_model_lock'],aliases=aliases,protocol=bs.PROTOCOL,
            original_failed_cost=contract['original_failed_cost'],original_gpu_resource_receipt=contract['original_gpu_resource_receipt'],
            original_failed_attempt=contract['original_failed_attempt'])
        contract['helper_scripts']=helper_refs; rt.write(folder/'PROTOCOL.json',contract)
        output['protocol_artifact']={'path':str((folder/'PROTOCOL.json').relative_to(rt.ROOT)),'sha256':rt.sha(folder/'PROTOCOL.json')}
        snapshot=rt.source_snapshot(args.tag)
        if snapshot != lock['source_hashes']:raise ValueError('COST_RECOVERY_SOURCE_SNAPSHOT_NOT_ORIGINAL16')
        output['scientific_source_snapshot']={'path':str((rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json').relative_to(rt.ROOT)),
            'sha256':rt.sha(rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json')}
        if args.preflight_only:
            unchanged_sources(contract); output.update(status='COMPLETE_PREFLIGHT_ONLY',validated_reused_rows=404,missing_rows_to_measure=132,
                GPU_model_or_CUDA_loaded=False,missing_rows_measured=0)
        else:
            output['records']=list(reused)
            output['timing_producers']=[{'key':list(bs.row_key(row)),'original_row_index':index,
                'source_result':contract['original_failed_cost'],'resource_receipt':contract['original_gpu_resource_receipt'],
                'reference_kind':'original_stored_bulk_science_passed_unchanged','new_timing_calls':0} for index,row in enumerate(reused)]
            bm.en.configure(); forecaster=bm.en.ex.previous.FrozenBolt(); bm.en.replace_normalizer(forecaster)
            output['backbone_before_sha256']=rt.state_hash(forecaster.model)
            if output['backbone_before_sha256'] != values['cost']['backbone_before_sha256']:raise ValueError('BACKBONE_CHANGED_COST_RECOVERY_START')
            for route in routes:
                batches=[b for b in (1,8) if (route['dataset'],route['identity'],route['route'],b) in missing]
                if not batches:continue
                model=ridge=None
                try:
                    unchanged_sources(contract); bulk=bs.load_bulk_science(route,pack)
                    references={}
                    for batch in batches:
                        try:
                            matched,meta=bs.science_reference(pack,route,batch,forecaster,limit)
                            meta.update(original_bulk_prediction=route['science_prediction'],model_artifacts=route['model_artifacts'],
                                origins_exact=True,mean_std_exact=True,normalized_vs_bulk=bs.normalized_reinference(matched,bulk[:batch]))
                            filename=f'{route["dataset"]}_{route["identity"]}_B{batch}'
                            meta['artifact']=rt.save_npz(folder/(filename+'_SCIENCE_REFERENCE.npz'),
                                matched_science_normalized=matched,original_bulk_science_normalized=bulk[:batch],
                                source_history_normalized=pack['history'][route['split']][:batch],
                                origins=pack['origins'][route['split']][:batch],mean=pack['mean'],std=pack['std'])
                            references[batch]=(matched,meta)
                            limit.check()
                        except Exception as exc:
                            if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                            output['failed_paths'].append({'key':[route['dataset'],route['identity'],route['route'],batch],
                                'stage':'independent_science_reference','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                    if not references:continue
                    if 'case' in route:model=bs.load_selected_model(route['case'])
                    if 'readout' in route:ridge=bench.load_ridge(route['readout'],rt)
                    elif 'control' in route and 'model' in route['control']:ridge=bench.load_ridge(route['control'],rt)
                    model_before=None if model is None else rt.state_hash(model)
                    for batch,(matched,meta) in references.items():
                        filename=f'{route["dataset"]}_{route["identity"]}_B{batch}'
                        try:
                            row,arrays=measure_batch(pack,route,batch,matched,bulk[:batch],forecaster,model,ridge)
                            row.update(matched_science_reference=meta,original_bulk_prediction=route['science_prediction'],
                                model_artifacts=route['model_artifacts'],source_origins_and_mean_std_exact=True,
                                deployment_model_state_before_sha256=model_before,
                                deployment_model_state_after_sha256=None if model is None else rt.state_hash(model))
                            row['deployment_selected_state_unchanged']=row['deployment_model_state_after_sha256']==model_before
                            row['parity_pass']=bool(row['parity_pass'] and row['deployment_selected_state_unchanged'])
                            row['actual_output']=rt.save_npz(folder/(filename+'_ACTUAL.npz'),**arrays)
                            output['records'].append(row);rt.write(folder/(filename+'.json'),row)
                            output['timing_producers'].append({'key':list(bs.row_key(row)),
                                'source_result_path':str((folder/'RESULT.json').relative_to(rt.ROOT)),
                                'original_row_index':None,'new_timing_calls':7,'reference_kind':bs.PROTOCOL['reference_kind']})
                            if not row['parity_pass']:
                                output['failed_paths'].append({'key':list(bs.row_key(row)),'stage':'parity_comparison',
                                    'status':'FAILED_PATH_PRESERVED','actual_output':row['actual_output'],
                                    'matched_science_reference':meta,'normalized_bulk_gate_pass':row['matched_science_to_original_bulk_normalized']['gate_pass'],
                                    'all_seven_matched_physical_comparisons':row['all_seven_matched_physical_comparisons']})
                            print(rt.json_line({'missing_cost_row':filename,'parity_pass':row['parity_pass'],
                                'original_bulk_physical_gate_pass':row['original_bulk_physical_gate_pass']}),flush=True)
                            limit.check()
                        except Exception as exc:
                            if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                            output['failed_paths'].append({'key':[route['dataset'],route['identity'],route['route'],batch],
                                'stage':'new_timing','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                except Exception as exc:
                    if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                    output['failed_paths'].append({'dataset':route['dataset'],'identity':route['identity'],'route':route['route'],
                        'stage':'route_assets_or_deployment_model','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                finally:
                    del model,ridge;gc.collect()
            output['backbone_after_sha256']=rt.state_hash(forecaster.model)
            if output['backbone_after_sha256'] != output['backbone_before_sha256']:raise ValueError('BACKBONE_CHANGED_COST_RECOVERY_END')
            unchanged_sources(contract)
            if output['records'][:404] != values['cost']['records']:raise ValueError('COST_ORIGINAL404_OBJECTS_CHANGED')
            expected={(r['dataset'],r['identity'],r['route'],b) for r in routes for b in (1,8)}
            actual=[bs.row_key(row) for row in output['records']]
            coverage=len(actual)==536 and len(set(actual))==536 and set(actual)==expected
            output.update(exact_route_batch_coverage=coverage,reused_original_objects_unchanged=True,
                reused_original_rows=404,new_rows_measured=len(actual)-404,new_rows_missing=536-len(actual),
                all_rows_appropriate_parity_gate_pass=all(row['parity_pass'] for row in output['records']))
            output['status']='COMPLETE' if coverage and output['all_rows_appropriate_parity_gate_pass'] and not output['failed_paths'] else 'FAILED_PRESERVED'
            terminal=output['status']
    except Exception as exc:
        terminal='BUDGET_HELD_PRESERVED' if isinstance(exc,rt.BudgetHeld) else 'FAILED_PRESERVED';error=str(exc)
        output.update(status=terminal,error=error,traceback=traceback.format_exc())
    output['finished_unix']=time.time();output['resource_receipt']=raw_limit.close(terminal,error)
    if not args.preflight_only:
        for producer in output['timing_producers']:
            if producer['new_timing_calls']:producer['resource_receipt']=output['resource_receipt']
        output['timing_resource_receipts_counted_once']=([output['original_gpu_resource_receipt']] if 'original_gpu_resource_receipt' in output else [])+[output['resource_receipt']]
        output['resource_accounting_note']='Original failed cost GPU process once plus recovery GPU process once; repeated timing_producer links and medians are not additional process charges'
    rt.write(folder/'RESULT.json',output)
    print(rt.json_line({'cost_recovery_terminal':str(folder/'RESULT.json'),'status':output['status'],
        'rows':len(output['records']),'preflight_only':args.preflight_only}),flush=True)
    if output['status'] not in ('COMPLETE','COMPLETE_PREFLIGHT_ONLY'):raise SystemExit(2)


if __name__=='__main__':main()
