"""Measure all 120 Ridge deployment routes on CPU FP32 heads at BLAS2."""
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
import batch_science_reference as bs
import cpu_ridge_deployment as cpu
import deployment_past_inputs as dp


class CappedLimit:
    def __init__(self,limit):self.limit=limit
    def check(self):
        value=self.limit.check()
        if value['elapsed_seconds']>300:raise rt.BudgetHeld('CPU_RIDGE_GPU_PROCESS_WALL_300SECONDS')
        return value


def source_read(path,status='COMPLETE'):
    value=rt.read(path)
    if value['status']!=status:raise ValueError('CPU_RIDGE_SOURCE_STATUS: '+str(path))
    return value,{'path':str(path.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(path)}


def bulk_first8(route,pack):
    path=bs.verify_artifact(route['science_prediction']);split=route['split']
    with np.load(path,allow_pickle=False) as z:
        prediction=z[split+'_prediction'][:8].copy();origins=z[split+'_origins'][:8].copy();mean=z['mean'].copy();std=z['std'].copy()
    if not np.array_equal(origins,pack['origins'][split]) or not np.array_equal(mean,pack['mean']) or not np.array_equal(std,pack['std']):
        raise ValueError('CPU_RIDGE_BULK_PAST_STATS_ORIGINS_NOT_EXACT')
    if prediction.shape!=(8,48,len(mean)) or not np.isfinite(prediction).all():raise ValueError('CPU_RIDGE_BULK_FIRST8_SHAPE')
    return prediction


def preflight(args):
    lock=ru.check_method_sources(args.method_lock);sources={};values={}
    for name in ('dev','fresh','probe','evaluation','cuda_cost'):
        value,ref=source_read(getattr(args,name+'_result'),'FAILED_PRESERVED' if name=='cuda_cost' else 'COMPLETE')
        if value['source_hashes']!=lock['source_hashes']:raise ValueError('CPU_RIDGE_SOURCE16_NOT_IDENTICAL: '+name)
        values[name]=value;sources[name]=ref
    if len(values['cuda_cost']['records'])!=536:raise ValueError('CPU_RIDGE_CUDA536_SOURCE_NOT_COMPLETE_COVERAGE')
    method_ref={'path':str(args.method_lock.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(args.method_lock)}
    for name in ('dev','fresh','probe','evaluation'):
        if values['cuda_cost']['source_results'][name]['sha256']!=sources[name]['sha256']:raise ValueError('CPU_RIDGE_CUDA_SOURCE_RESULTS_CHANGED')
    if values['cuda_cost']['method_lock']['sha256']!=method_ref['sha256']:raise ValueError('CPU_RIDGE_CUDA_METHOD_LOCK_CHANGED')
    global_ref=values['evaluation']['global_model_lock'];global_body=rt.read(bs.verify_artifact(global_ref))
    if not global_body.get('ALL_SELECTED') or global_body['method_lock_sha256']!=method_ref['sha256'] or global_body['source_hashes']!=lock['source_hashes']:
        raise ValueError('CPU_RIDGE_GLOBAL_MODEL_SELECTION_LOCK')
    routes,aliases=bs.build_routes(values['dev'],values['fresh'],values['probe'],values['evaluation'],lock)
    selected=[r for r in routes if r['route'] in cpu.RIDGE_ROUTES]
    if len(selected)!=120 or len({(r['dataset'],r['identity'],r['route']) for r in selected})!=120:
        raise ValueError('CPU_RIDGE_UNIFORM120_ROUTE_COVERAGE')
    allowed={(str((rt.ROOT/a['path']).resolve()),a['sha256']) for a in global_body['selected_models']}
    packs={};past={}
    for dataset in ru.LEGACY+ru.FRESH:
        packs[dataset],past[dataset]=dp.load_past_pack(dataset,values['dev'],values['fresh'],values['evaluation'],args.method_lock,windows=8)
    coefficient_metadata=[];checkpoint_metadata={}
    for route in selected:
        pack=packs[route['dataset']];bulk_first8(route,pack)
        for artifact in route['model_artifacts']:
            if (str((rt.ROOT/artifact['path']).resolve()),artifact['sha256']) not in allowed:raise ValueError('CPU_RIDGE_MODEL_NOT_IN_GLOBAL_LOCK')
            bs.verify_artifact(artifact)
        record=route['readout'] if 'readout' in route else route['control'];ridge=cpu.load_ridge(record,rt)
        coefficient_metadata.append({'dataset':route['dataset'],'identity':route['identity'],'route':route['route'],
            'artifact':record['model'],'weight_shape':list(ridge['weight'].shape),'bias_shape':list(ridge['bias'].shape),
            'weight_dtype':str(ridge['weight'].dtype),'bias_dtype':str(ridge['bias'].dtype),
            'coefficient_array_sha256':ru.dt.array_hash(ridge['weight']),'bias_array_sha256':ru.dt.array_hash(ridge['bias'])})
        if 'case' in route:
            case=route['case'];checkpoint=case['training']['selected_checkpoint'];identity=(checkpoint['path'],checkpoint['sha256'])
            if identity not in checkpoint_metadata:
                saved=torch.load(bs.verify_artifact(checkpoint),map_location='cpu',weights_only=True)['model']
                digest=rt.state_hash(saved)
                if digest!=case['training']['selected_hash']:raise ValueError('CPU_RIDGE_PREFLIGHT_CHECKPOINT_FULLSTATE_HASH')
                cache={name:t for name,t in saved.items() if name.startswith('memory_')}
                checkpoint_metadata[identity]={'dataset':case['dataset'],'source_label':case['label'],'checkpoint':checkpoint,
                    'selected_model_state_sha256':digest,'memory_buffers_sha256':rt.state_hash(cache),
                    'memory_is_frozen':bool(saved['memory_is_frozen']),'new_cache_computed_on_CPU':False}
                del saved
        del ridge
    protocol={'status':'SEALED_UNIFORM_CPU_RIDGE240_PROTOCOL','source_results':sources,'source_hashes':lock['source_hashes'],
        'method_lock':method_ref,'global_model_lock':global_ref,'past_only_evidence':past,
        'expected_CPU_Ridge_routes':120,'expected_CPU_Ridge_rows':240,'canonical_CUDA_rows_to_reuse':296,
        'CPU_implementation_id':cpu.IMPLEMENTATION_ID,'CPU_BLAS_threads':2,'Torch_threads':2,
        'independent_reference_kind':bs.PROTOCOL['reference_kind'],'physical_atol':.002,'physical_rtol':2e-5,
        'samebatch_to_bulk_normalized_atol':1e-5,'samebatch_to_bulk_normalized_rtol':2e-6,
        'old_bulk_CPU_thread_count_assumed_equal_to_two':False,'source_bulk_predictions_and_accuracy_unchanged':True,
        'coefficient_metadata':coefficient_metadata,'checkpoint_cache_metadata':list(checkpoint_metadata.values()),
        'all_120_Ridge_routes_instead_of_only_40_failed_rows':True,'warmup':2,'repeats':7,'GPU_process_wall_cap_seconds':300,
        'EVAL_target_arrays_opened':False,'protected_TEST_opened':False,'new_training_updates':0,
        'timing_includes':'raw causal preprocessing/FIT normalization/active GPU E or F/H2D and feature D2H/original CPU FP32 head/inverse/CUDA sync',
        'timing_excludes':'model and coefficient disk load, verified historical FIT memory cache preparation, science reference inference, artifact writing'}
    return protocol,values,selected,aliases,packs


def unchanged(protocol):
    ru.source_check(protocol['source_hashes'])
    for ref in list(protocol['source_results'].values())+[protocol['method_lock'],protocol['global_model_lock']]+protocol.get('helper_scripts',[]):
        try:bs.verify_artifact(ref)
        except ValueError as exc:raise RuntimeError('ACTIVE_SOURCE_CHANGED: '+ref['path']) from exc


def measure(pack,route,batch,matched,bulk,F,model,ridge):
    split=route['split'];raw=pack['raw_history'][split][:batch];mask=pack['raw_history_observed'][split][:batch];origins=pack['origins'][split][:batch]
    outputs=[];seconds=[]
    with torch.no_grad():
        for _ in range(2):cpu.deployment(pack,raw,mask,origins,route['route'],F,model,ridge)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for _ in range(7):
            torch.cuda.synchronize();started=time.perf_counter()
            actual=cpu.deployment(pack,raw,mask,origins,route['route'],F,model,ridge)
            torch.cuda.synchronize();seconds.append(time.perf_counter()-started);outputs.append(actual)
    matched_checks=[bs.physical_parity(v,matched,pack['mean'],pack['std']) for v in outputs]
    bulk_checks=[bs.physical_parity(v,bulk,pack['mean'],pack['std']) for v in outputs]
    norm_check=bs.normalized_reinference(matched,bulk);valid_timings=len(seconds)==7 and np.isfinite(seconds).all() and all(v>0 for v in seconds)
    last=matched_checks[-1]
    row={'dataset':route['dataset'],'identity':route['identity'],'route':route['route'],'split':split,'batch':batch,
        'warmup':2,'repeats':7,'seconds':seconds,'median_seconds':float(np.median(seconds)),
        'median_seconds_per_window':float(np.median(seconds))/batch,'deployment_implementation':cpu.IMPLEMENTATION_ID,
        'physical_parity_atol':.002,'physical_parity_rtol':2e-5,'physical_max_abs_vs_science':last['physical_max_abs_vs_science'],
        'normalized_max_abs_vs_science':last['normalized_max_abs_vs_science'],
        'parity_pass':bool(valid_timings and norm_check['gate_pass'] and all(p['parity_pass'] for p in matched_checks)),
        'seven_timings_positive_finite':bool(valid_timings),'reference_kind':bs.PROTOCOL['reference_kind'],
        'matched_science_to_original_bulk_normalized':norm_check,'all_seven_matched_physical_comparisons':matched_checks,
        'all_seven_original_bulk_physical_comparisons':bulk_checks,'original_bulk_physical_gate_pass':all(p['parity_pass'] for p in bulk_checks),
        'original_bulk_physical_failure_relabelled':False,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
        'rss_bytes':psutil.Process().memory_info().rss,'model_metadata':None if model is None else model.metadata(),
        'implementation_metadata':cpu.implementation_metadata(route['route'],model,ridge),
        'includes':'raw causal preprocessing/FIT normalization/active GPU E or F/H2D and feature D2H/original CPU FP32 head/inverse/CUDA sync',
        'excludes':'model and coefficient disk load, verified historical FIT memory cache preparation, reference inference, artifact writing',
        'memory_note':'Frozen Bolt/full checkpoint co-resident as applicable; Ridge coefficients preloaded on CPU',
        'old_CUDA_timings_relabelled_as_CPU':False}
    arrays={'actual_physical':outputs[-1],'all_7_physical_outputs':np.stack(outputs),'matched_science_normalized':matched,
        'matched_science_physical_CPU_FP32':matched*pack['std']+pack['mean'],'original_bulk_science_normalized':bulk,
        'original_bulk_science_physical_CPU_FP32':bulk*pack['std']+pack['mean'],'raw_history':raw,'history_observed':mask,
        'origins':origins,'mean':pack['mean'],'std':pack['std'],'source_history_normalized':pack['history'][split][:batch]}
    return row,arrays


def main():
    parser=argparse.ArgumentParser()
    for name in ('dev-result','fresh-result','probe-result','evaluation-result','cuda-cost-result','method-lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('CPU_RIDGE_INVALID_EXCLUSIVE_TAG')
    folder=rt.HERE/'cpu_ridge_cost_v1'/args.tag;folder.mkdir(parents=True,exist_ok=False)
    helpers=[]
    for path in (Path(__file__),Path(cpu.__file__),Path(dp.__file__),Path(bs.__file__)):
        destination=folder/path.name
        with destination.open('xb') as h:h.write(path.read_bytes())
        helpers.append({'path':str(path.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(path),
            'snapshot_path':str(destination.relative_to(rt.ROOT)),'snapshot_sha256':rt.sha(destination)})
    raw_limit=rt.Limit(args.tag,gpu=not args.preflight_only);limit=CappedLimit(raw_limit)
    output={'status':'INCOMPLETE','stage':'cpu_ridge_cost_preflight' if args.preflight_only else 'cpu_ridge_cost','tag':args.tag,
        'started_unix':time.time(),'records':[],'failed_paths':[],'helper_scripts':helpers,'timing_producers':[],
        'expected_rows':240,'canonical_paths':120,'new_training_updates':0,'EVAL_target_arrays_opened':False,
        'protected_TEST_opened':False,'new_model_selection':False,'bulk_science_accuracy_changed':False,
        'deployment_implementation':cpu.IMPLEMENTATION_ID}
    terminal,error='COMPLETE',None
    try:
        torch.set_num_threads(2)
        with cpu.blas_context():
            protocol,values,routes,aliases,packs=preflight(args);protocol['helper_scripts']=helpers
            unchanged(protocol);rt.write(folder/'PROTOCOL.json',protocol)
            output.update(source_hashes=protocol['source_hashes'],source_results=protocol['source_results'],
                method_lock=protocol['method_lock'],global_model_lock=protocol['global_model_lock'],aliases=aliases,
                protocol_artifact={'path':str((folder/'PROTOCOL.json').relative_to(rt.ROOT)),'sha256':rt.sha(folder/'PROTOCOL.json')},
                thread_contract_metadata=cpu.implementation_metadata('UNIFORM_PREFLIGHT'))
            if not output['thread_contract_metadata']['all_BLAS_pools_have_two_threads']:raise ValueError('CPU_RIDGE_BLAS2_NOT_ACTUALLY_APPLIED')
            sources=rt.source_snapshot(args.tag)
            if sources!=protocol['source_hashes']:raise ValueError('CPU_RIDGE_SOURCE_SNAPSHOT_CHANGED')
            output['scientific_source_snapshot']={'path':str((rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json').relative_to(rt.ROOT)),
                'sha256':rt.sha(rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json')}
            if args.preflight_only:
                output.update(status='COMPLETE_PREFLIGHT_ONLY',validated_Ridge_routes=120,validated_expected_rows=240,
                    coefficient_and_checkpoint_cache_SHA_verified=True,GPU_model_or_CUDA_loaded=False)
            else:
                bm.en.configure();F=bm.en.ex.previous.FrozenBolt();bm.en.replace_normalizer(F)
                output['backbone_before_sha256']=rt.state_hash(F.model)
                if output['backbone_before_sha256']!=values['cuda_cost']['backbone_before_sha256']:raise ValueError('BACKBONE_CHANGED_CPU_RIDGE_START')
                for route in routes:
                    model=ridge=None;pack=packs[route['dataset']]
                    try:
                        unchanged(protocol);bulk=bulk_first8(route,pack);references={}
                        for batch in (1,8):
                            try:
                                matched,meta=bs.science_reference(pack,route,batch,F,limit)
                                filename=f'{route["dataset"]}_{route["identity"]}_B{batch}'
                                meta.update(original_bulk_prediction=route['science_prediction'],model_artifacts=route['model_artifacts'],
                                    origins_exact=True,mean_std_exact=True,BLAS_threads=2,
                                    normalized_vs_bulk=bs.normalized_reinference(matched,bulk[:batch]),
                                    thread_contract_metadata=cpu.implementation_metadata(route['route']))
                                meta['artifact']=rt.save_npz(folder/(filename+'_SCIENCE_REFERENCE.npz'),matched_science_normalized=matched,
                                    original_bulk_science_normalized=bulk[:batch],source_history_normalized=pack['history'][route['split']][:batch],
                                    origins=pack['origins'][route['split']][:batch],mean=pack['mean'],std=pack['std'])
                                references[batch]=(matched,meta);limit.check()
                            except Exception as exc:
                                if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                                output['failed_paths'].append({'key':[route['dataset'],route['identity'],route['route'],batch],
                                    'stage':'original_science_reference_BLAS2','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                        if not references:continue
                        if 'case' in route:model=bs.load_selected_model(route['case'])
                        cache=cpu.prepare_memory(model);record=route['readout'] if 'readout' in route else route['control'];ridge=cpu.load_ridge(record,rt)
                        model_before=None if model is None else rt.state_hash(model)
                        for batch,(matched,meta) in references.items():
                            filename=f'{route["dataset"]}_{route["identity"]}_B{batch}'
                            try:
                                row,arrays=measure(pack,route,batch,matched,bulk[:batch],F,model,ridge)
                                row.update(matched_science_reference=meta,original_bulk_prediction=route['science_prediction'],
                                    model_artifacts=route['model_artifacts'],source_origins_and_mean_std_exact=True,
                                    preloaded_memory_cache_preparation=cache,deployment_model_state_before_sha256=model_before,
                                    deployment_model_state_after_sha256=None if model is None else rt.state_hash(model))
                                row['deployment_selected_state_unchanged']=row['deployment_model_state_after_sha256']==model_before
                                row['parity_pass']=bool(row['parity_pass'] and row['deployment_selected_state_unchanged'] and row['implementation_metadata']['all_BLAS_pools_have_two_threads'])
                                row['actual_output']=rt.save_npz(folder/(filename+'_ACTUAL.npz'),**arrays)
                                output['records'].append(row);rt.write(folder/(filename+'.json'),row)
                                output['timing_producers'].append({'key':list(bs.row_key(row)),
                                    'deployment_implementation':cpu.IMPLEMENTATION_ID,'source_result_path':str((folder/'RESULT.json').relative_to(rt.ROOT)),
                                    'source_row_index':len(output['records'])-1,'new_timing_calls':7,'reference_kind':row['reference_kind']})
                                if not row['parity_pass']:output['failed_paths'].append({'key':list(bs.row_key(row)),
                                    'stage':'CPU_head_physical_or_reinference_gate','status':'FAILED_PATH_PRESERVED',
                                    'actual_output':row['actual_output'],'matched_science_reference':meta,
                                    'all_seven_matched_physical_comparisons':row['all_seven_matched_physical_comparisons']})
                                print(rt.json_line({'CPU_Ridge_cost_row':filename,'parity_pass':row['parity_pass'],
                                    'original_bulk_physical_gate_pass':row['original_bulk_physical_gate_pass']}),flush=True)
                                limit.check()
                            except Exception as exc:
                                if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                                output['failed_paths'].append({'key':[route['dataset'],route['identity'],route['route'],batch],
                                    'stage':'uniform_CPU_head_timing','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                        for artifact in route['model_artifacts']+[route['science_prediction']]:bs.verify_artifact(artifact)
                    except Exception as exc:
                        if isinstance(exc,rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED','BACKBONE_CHANGED')):raise
                        output['failed_paths'].append({'dataset':route['dataset'],'identity':route['identity'],'route':route['route'],
                            'stage':'route_assets_or_preloaded_models','error':str(exc),'traceback':traceback.format_exc(),'status':'FAILED_PATH_PRESERVED'})
                    finally:del model,ridge;gc.collect()
                output['backbone_after_sha256']=rt.state_hash(F.model)
                if output['backbone_before_sha256']!=output['backbone_after_sha256']:raise ValueError('BACKBONE_CHANGED_CPU_RIDGE_END')
                unchanged(protocol);expected={(r['dataset'],r['identity'],r['route'],b) for r in routes for b in (1,8)}
                keys=[bs.row_key(r) for r in output['records']];coverage=len(keys)==240 and len(set(keys))==240 and set(keys)==expected
                output.update(exact_uniform_CPU240_coverage=coverage,actual_timed_rows=len(keys),missing_timed_rows=240-len(keys),
                    all_rows_critical_gates_pass=all(r['parity_pass'] for r in output['records']))
                output['status']='COMPLETE' if coverage and output['all_rows_critical_gates_pass'] and not output['failed_paths'] else 'FAILED_PRESERVED'
                terminal=output['status']
    except Exception as exc:
        terminal='BUDGET_HELD_PRESERVED' if isinstance(exc,rt.BudgetHeld) else 'FAILED_PRESERVED';error=str(exc)
        output.update(status=terminal,error=error,traceback=traceback.format_exc())
    output['finished_unix']=time.time();output['resource_receipt']=raw_limit.close(terminal,error)
    for producer in output['timing_producers']:producer['resource_receipt']=output['resource_receipt']
    rt.write(folder/'RESULT.json',output)
    print(rt.json_line({'CPU_Ridge_cost_terminal':str(folder/'RESULT.json'),'status':output['status'],'rows':len(output['records'])}),flush=True)
    if output['status'] not in ('COMPLETE','COMPLETE_PREFLIGHT_ONLY'):raise SystemExit(2)


if __name__=='__main__':main()
