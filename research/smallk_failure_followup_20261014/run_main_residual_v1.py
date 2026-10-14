"""Fresh gated CPU corrections to original MAIN; no E/F/D loads or replay.

Import/describe is source only. Actual execution first constructs the new
human-approved runtime, then verifies audited matched inputs and frozen MAIN.
"""
import argparse
from pathlib import Path
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT/'.cache'/HERE.name/'main_residual_v1'
DATASETS = {'robin':(17,4,'val',30), 'beijing_aotizhongxin':(11,2,'eval',146)}
SEEDS = (92601,92602)
CANONICAL_MATCHED_REPORT_SHA = '1573f4ee628fc5598e41178457937c86137198bb4007b11f3790e2659386c6d2'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT))


def safe_tag(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
        raise ValueError('SAFE_FRESH_TAG_REQUIRED')
    return tag


def artifact(producer, stem):
    matches = [r for r in producer['artifacts'] if Path(r['path']).stem == stem]
    if len(matches) != 1:
        raise ValueError('EXACT_SINGLE_MATCHED_ARTIFACT_REQUIRED:'+stem)
    return matches[0]


def validate_matched_flags(producer, dataset, seed):
    if (producer.get('status') != 'COMPLETE' or producer.get('dataset') != dataset or producer.get('seed') != seed
            or producer.get('GPU_execution') is not False or producer.get('E_D_F_loaded') is not False
            or producer.get('E_D_F_forward_replayed') is not False or producer.get('E_D_F_updated') is not False
            or producer.get('future_oracle_read') is not False or producer.get('independent_holdout') is not False):
        raise ValueError('COMPLETE_EXACT_PAST_ONLY_MATCHED_PRODUCER_REQUIRED')


def verify_canonical_audit(probes, path, matched_record, dataset, seed):
    record = probes.verified_file({'path':str(probes.resolve(path)), 'sha256':CANONICAL_MATCHED_REPORT_SHA})
    manifest = probes.read(record['path'])
    if manifest.get('status') != 'COMPLETE' or manifest.get('audit_gate',{}).get('passed') is not True:
        raise ValueError('COMPLETE_CANONICAL_MATCHED_REPORT_REQUIRED')
    inputs = manifest['input_records']
    if not any(r['role']=='producer' and r['sha256'].lower()==matched_record['sha256'] for r in inputs):
        raise ValueError('MATCHED_PRODUCER_NOT_BOUND_TO_CANONICAL_REPORT')
    candidates = [r for r in inputs if r['role']=='independent_audit']
    if len(candidates)!=1:
        raise ValueError('EXACT_SINGLE_MATCHED_AUDIT_REQUIRED')
    audit_record = probes.verified_file({**candidates[0],'sha256':candidates[0]['sha256'].lower()})
    audit = probes.read(audit_record['path'])
    expected = {(d,s) for d in DATASETS for s in SEEDS}
    conditions = audit.get('conditions',[])
    if (audit.get('status')!='PASS' or audit.get('failed_count')!=0 or len(conditions)!=4
            or {(r['dataset'],r['seed']) for r in conditions}!=expected):
        raise ValueError('EXACT_FOUR_CONDITION_MATCHED_AUDIT_PASS_REQUIRED')
    bound = [r for r in conditions if (r['dataset'],r['seed'])==(dataset,seed)]
    if len(bound)!=1 or bound[0].get('status')!='PASS' or bound[0]['producer_sha256'].lower()!=matched_record['sha256']:
        raise ValueError('MATCHED_AUDIT_EXACT_PRODUCER_SHA_REQUIRED')
    return [record,audit_record]


def solution_arrays(solution):
    return {name:getattr(solution,name) for name in ('coefficients','intercept','observed_counts','channel_counts',
        'normal_equation_denominators','design_ranks','regularized_gram_conditions')}


def load_selected_models(producer, probes):
    """Caller must create its approved runtime before any NPZ loading here.

    Pure model restoration only: no fit, E/F/D import, CUDA or model loader.
    A failed route may have a usable selected ZERO/MEAN control; caller must
    report its original route status rather than inventing full scientific fits.
    """
    import main_forecast_residual_readout_v1 as math
    models = {}
    for route in math.ROUTES:
        row = producer['families'].get(route,{})
        if not row.get('selected_model_metadata'):
            models[route]=None; continue
        feature_arrays,_ = probes.load_checked_npz(row['feature_state']['artifact'])
        model_metadata = row['selected_model_metadata']
        arrays = None
        if model_metadata['candidate_id'] != math.ZERO:
            arrays,_ = probes.load_checked_npz(row['selected_coefficient_artifact'])
        models[route] = math.restore_model(row['feature_state'],feature_arrays,model_metadata,arrays)
        if models[route].sha256 != row['selected_model_sha256']:
            raise ValueError('SELECTED_RESIDUAL_MODEL_SHA:'+route)
    return models


def execute(args):
    import main_residual_runtime_v1 as runtime
    limit = runtime.ContinuationLimit('main_residual_'+safe_tag(args.tag),gpu=False)
    output = CACHE/args.tag
    result = {'status':'RUNNING','dataset':args.dataset,'seed':args.seed,'scientific_execution':True,
        'GPU_execution':False,'E_D_F_loaded':False,'E_D_F_forward_replayed':False,'E_D_F_updated':False,
        'future_oracle_read':False,'protected_TEST_accessed':False,'old_campaign_restart':False,
        'independent_holdout':False,'whole_goal_complete':False,'families':{},'controls':{},'artifacts':[],
        'input_artifact_refs':{},'source_records':[],'source_snapshots':[],'timings':{},
        'maximum_new_ridge_coefficients':21,'maximum_candidate_comparisons':27,
        'shared_mean_fits_per_condition':1,'big_input_arrays_copied':False,
        'selection':'actual-Y TUNE normalized observed-channel macro MSE; earliest zero then mean then seven ridge',
        'permission_gate':limit.authorization_record,'run_manifest':limit.run_manifest_record,
        'tolerance':{'atol':1e-5,'rtol':2e-6}}
    initialized = False
    start = time.perf_counter()
    try:
        output.mkdir(parents=True,exist_ok=False); initialized=True
        import numpy as np
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import run_matched_readout_v1 as old_runner
        import matched_readout_contract_v1 as contract
        import frozen_decoder_refit as scoring
        import main_forecast_residual_readout_v1 as math
        from threadpoolctl import threadpool_limits
        inputs.install_protected_test_guard()
        full,fast = limit.checkfull,limit.shared.checkfast
        def save(name, arrays):
            full(); record=probes.save_npz(output/(name+'.npz'),arrays)
            result['artifacts'].append(record); full(); return record
        def scores(prediction, values):
            if prediction.dtype!=np.float32 or prediction.shape!=values['target'].shape or not np.isfinite(prediction).all():
                raise ValueError('FINITE_FP32_N48C_PREDICTION_REQUIRED')
            return {'normalized':scoring.observed_score(prediction,values['target'],values['observed']),
                    'physical_per_channel_diagnostic':scoring.observed_score(prediction,values['target'],values['observed'],std)}
        full(); t=time.perf_counter()
        matched_path=probes.resolve(args.matched_result)
        matched_record=probes.verified_file({'path':str(matched_path),'sha256':probes.sha(matched_path)})
        matched=probes.read(matched_path)
        validate_matched_flags(matched,args.dataset,args.seed)
        result['source_records'].extend(verify_canonical_audit(probes,args.matched_report_manifest,matched_record,args.dataset,args.seed))
        result['matched_source']=matched_record; result['source_records'].append(matched_record)
        for source in matched['source_snapshots']:
            result['source_records'].append(probes.verified_file(source))
            if probes.sha(source['snapshot_path'])!=source['sha256']:
                raise ValueError('MATCHED_SOURCE_SNAPSHOT_CHANGED')
        bridge_record=matched['bridge_provenance']['input_bridge']
        probes.verified_file(bridge_record)
        bridge,original_splits,mean,std,provenance=probes.load_bridge(bridge_record['path'],args.dataset,inputs.verify_phase_barrier)
        if provenance['input_bridge']['sha256']!=bridge_record['sha256']:
            raise ValueError('MATCHED_EXACT_OLD_INPUT_BRIDGE_REQUIRED')
        result['bridge_provenance']=provenance
        result['source_records'].extend(provenance['input_artifacts']+[provenance['input_bridge']])
        result['source_records'].extend(probes.verify_source_records(bridge))
        result['source_records'].append(probes.check_original_raw(bridge,args.dataset,original_splits,mean,std))
        cp_key=('robin' if args.dataset=='robin' else 'beijing')+'_'+str(args.seed)
        cp=probes.verified_file(bridge['declared_asset_hashes']['checkpoints_file_hash_only'][cp_key])
        if cp['sha256']!=matched['original_CP_file_hash_only']['sha256']:
            raise ValueError('SAME_ORIGINAL_MAIN_CP_FILE_SHA_REQUIRED')
        result['original_CP_file_hash_only']=cp; result['source_records'].append(cp)
        stage_record=probes.verified_file(matched['stage_N2_source'])
        stage=probes.read(stage_record['path'])
        contract.validate_stage_reuse(stage,args.dataset,args.seed)
        old_update,old_hash=probes.EXPECTED_CODECS[args.dataset,args.seed]
        if stage['MAIN_checkpoint']['selected_update']!=old_update or stage['codec_before']['old_algorithm']!=old_hash:
            raise ValueError('LOCKED_OLD_MAIN_UPDATE_STATE_METADATA_REQUIRED')
        result['source_records'].append(stage_record); result['old_codec_dual_hash_reference']=stage['codec_before']
        if stage['MAIN_checkpoint']['checkpoint']['sha256']!=cp['sha256']:
            raise ValueError('SAME_ORIGINAL_STAGE_MAIN_CP_REQUIRED')
        main_record=bridge['declared_asset_hashes']['predictions'][args.dataset]['main_'+str(args.seed)]
        main_arrays,checked_main=probes.load_checked_npz(main_record)
        main_predictions=probes.compare_reference(main_arrays,original_splits,mean,std)
        result['original_MAIN_prediction_source']=main_record; result['source_records'].append(checked_main)
        raw_splits={}
        for split, original in original_splits.items():
            full(); record=artifact(matched,'RAW_INPUT_'+split)
            arrays,checked=probes.load_checked_npz(record)
            contract.validate_raw_split(args.dataset,split,arrays,mean,std)
            for name in ('history','history_observed','target','observed','origins','raw_last','raw_forecast'):
                contract.exact(arrays[name],original[name],'RAW:'+split+':'+name)
            contract.exact(arrays['mean'],mean,'RAW:'+split+':mean'); contract.exact(arrays['std'],std,'RAW:'+split+':std')
            raw_splits[split]={**arrays,'main_forecast':main_predictions[split]}
            result['input_artifact_refs'][split]={'raw':record,'cached_E':None,'original_MAIN':main_record}
            result['source_records'].append(checked)
        cached,cached_records={},{}
        try:
            for split, values in raw_splits.items():
                full(); declared=artifact(matched,'CACHED_E_INPUT_'+split)
                arrays,checked=probes.load_checked_npz(declared)
                contract.validate_split(args.dataset,split,values,arrays,mean,std)
                contract.exact(arrays['original_prediction'],values['main_forecast'],'ORIGINAL_MAIN:'+split)
                cached[split]=arrays; cached_records[split]=declared
                result['input_artifact_refs'][split]['cached_E']=declared; result['source_records'].append(checked)
            d2=old_runner.verify_cached_lineage(probes,stage,args.dataset,args.seed,provenance,cp,cached_records,cached)
            result['source_records'].append(d2); result['original_D2_source']=d2
            result['cached_E_original_MAIN_bound']=True
        except Exception as error:
            if isinstance(error,runtime.BudgetHeld): raise
            result['E_cache_failure']={'type':type(error).__name__,'message':str(error)}; cached={}
        result['timings']['input_provenance_and_hash_validation_wall_seconds']=time.perf_counter()-t
        source_folder=output/'sources'; source_folder.mkdir()
        paths=(Path(__file__),HERE/'main_forecast_residual_readout_v1.py',HERE/'matched_readout_math_v1.py',
            HERE/'stage_separation_math_v1.py',HERE/'matched_readout_contract_v1.py',HERE/'run_matched_readout_v1.py',
            HERE/'run_condition_probes.py',HERE/'prepare_inputs.py',HERE/'frozen_decoder_refit.py',
            Path(runtime.CORE_RUNTIME),Path(runtime.APPROVED_PLAN),Path(runtime.METHOD_LOCK),HERE/'fast_storage_scan_v1.py')
        for path in dict.fromkeys(paths):
            full(); digest=probes.sha(path); destination=source_folder/path.name
            with path.open('rb') as src,destination.open('xb') as dst: shutil.copyfileobj(src,dst)
            if probes.sha(destination)!=digest: raise ValueError('SOURCE_SNAPSHOT_COPY_SHA')
            result['source_snapshots'].append({'path':str(path),'sha256':digest,'snapshot_path':str(destination)}); full()
        fit=raw_splits['fit']
        full(); t=time.perf_counter()
        mean_control=math.fit_intercept_only(fit['main_forecast'],fit['target'],fit['observed'])
        result['controls']['MEAN_CORRECTION']={'status':'COMPLETE','model_sha256':mean_control.sha256,
            'new_FIT_mean_wall_seconds':time.perf_counter()-t,'fit_residual_sha256':mean_control.fit_residual_sha256,
            'fit_observed_sha256':mean_control.fit_observed_sha256,
            'coefficient_artifact':save('SHARED_MEAN_COEFFICIENTS',{'correction':mean_control.correction,'observed_counts':mean_control.observed_counts}),
            'splits':{},'artifacts':{}}
        result['controls']['ZERO_CORRECTION']={'status':'COMPLETE','new_training':False,'splits':{},'artifacts':{}}
        result['references']={}
        for split, values in raw_splits.items():
            full()
            for name,prediction in ((math.ZERO,values['main_forecast'].copy()),(math.MEAN,mean_control.forecast(values['main_forecast']))):
                result['controls'][name]['splits'][split]=scores(prediction,values)
                result['controls'][name]['artifacts'][split]=save(name+'_'+split,{'prediction':prediction,'origins':values['origins']})
            result['references'][split]={'ORIGINAL_MAIN':scores(values['main_forecast'],values),'RAW_BOLT':scores(values['raw_forecast'],values)}

        def fit_route(route,past_by_split):
            row={'status':'RUNNING','route':route,'candidate_grid':[],'splits':{},'selected_artifacts':{}}
            result['families'][route]=row; route_start=time.perf_counter()
            state=None; prepared=None; models=[]; failures=[]
            try:
                full(); t=time.perf_counter()
                state=math.fit_feature_state(route,past_by_split['fit'])
                row['FIT_feature_state_wall_seconds']=time.perf_counter()-t
                row['feature_state']={**state.metadata(),'artifact':save(route+'_FIT_FEATURE_STATE',{'center':state.center,'scale':state.scale})}
                # These objects exist before any SVD; the shared controls already
                # have all-split forecasts, and mean coefficients were fitted once.
                zero=math.zero_correction_model(state,fit['target'].shape[2])
                mean_model=math.mean_correction_model(state,fit['target'].shape[2],mean_control)
                for model in (zero,mean_model):
                    entry={'candidate_id':model.candidate_id,'alpha':None,'status':'COMPLETE',
                        'model_sha256':model.sha256,'model_metadata':model.metadata(),
                        'coefficient_artifact':None if model.candidate_id==math.ZERO else result['controls'][math.MEAN]['coefficient_artifact']}
                    for split in ('fit','tune'):
                        entry[split+'_scores']=result['controls'][model.candidate_id]['splits'][split]
                        entry[split+'_artifact']=result['controls'][model.candidate_id]['artifacts'][split]
                    entry['tune_actual_Y_mse']=entry['tune_scores']['normalized']['mse']
                    row['candidate_grid'].append(entry); models.append(model)
                full(); t=time.perf_counter()
                try:
                    with threadpool_limits(limits=2):
                        prepared=math.prepare_with_state(state,past_by_split['fit'],fit['main_forecast'],fit['target'],fit['observed'],
                            mean_control=mean_control,check=fast)
                    row['ridge_preparation_status']='COMPLETE'; row['preparation']=prepared.metadata()
                except Exception as error:
                    if isinstance(error,runtime.BudgetHeld): raise
                    row['ridge_preparation_status']='FAILED_PRESERVED'
                    row['ridge_preparation_failure']={'type':type(error).__name__,'message':str(error)}
                row['new_FIT_factorization_wall_seconds']=time.perf_counter()-t; full()
                for index,alpha in enumerate(math.ALPHAS):
                    entry={'candidate_id':math.CANDIDATE_IDS[index+2],'alpha':alpha,'status':'RUNNING'}
                    row['candidate_grid'].append(entry); models.append(None)
                    t=time.perf_counter()
                    try:
                        full()
                        if prepared is None: raise RuntimeError('RIDGE_PREPARATION_UNAVAILABLE')
                        with threadpool_limits(limits=2): model=prepared.fit(alpha)
                        entry.update(model_metadata=model.metadata(),model_sha256=model.sha256,
                            coefficient_artifact=save(route+'_alpha_'+str(index)+'_COEFFICIENTS',solution_arrays(model.solution)))
                        for split in ('fit','tune'):
                            fast()
                            with threadpool_limits(limits=2): prediction=model.forecast(past_by_split[split],raw_splits[split]['main_forecast'])
                            entry[split+'_scores']=scores(prediction,raw_splits[split])
                            entry[split+'_artifact']=save(route+'_alpha_'+str(index)+'_'+split,{'prediction':prediction,'origins':raw_splits[split]['origins']})
                        entry.update(status='COMPLETE',tune_actual_Y_mse=entry['tune_scores']['normalized']['mse'])
                        models[index+2]=model
                    except Exception as error:
                        if isinstance(error,runtime.BudgetHeld): raise
                        entry.update(status='FAILED_PRESERVED',tune_actual_Y_mse=None,
                                     failure={'type':type(error).__name__,'message':str(error)})
                        failures.append(entry['candidate_id'])
                    entry['fit_apply_inference_save_wall_seconds']=time.perf_counter()-t; full()
                index=math.select_candidate(row['candidate_grid']); model=models[index]; entry=row['candidate_grid'][index]
                row.update(selected_index=index,selected_candidate_id=model.candidate_id,selected_alpha=model.alpha,
                    selected_model_sha256=model.sha256,selected_model_metadata=model.metadata(),
                    selected_coefficient_artifact=entry['coefficient_artifact'],failing_candidate_ids=failures,
                    selected_new_ridge=model.solution is not None,
                    selected_original_MAIN_unchanged=model.candidate_id==math.ZERO)
                for split, values in raw_splits.items():
                    full()
                    with threadpool_limits(limits=2): prediction=model.forecast(past_by_split[split],values['main_forecast'])
                    row['splits'][split]=scores(prediction,values)
                    row['selected_artifacts'][split]=save(route+'_SELECTED_'+split,{'prediction':prediction,
                        'original_MAIN_prediction':values['main_forecast'],'target':values['target'],'observed':values['observed'],
                        'origins':values['origins'],'mean':mean,'std':std})
                row['status']='COMPLETE' if not failures else 'FAILED_PRESERVED'
            except Exception as error:
                row.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                           failure={'type':type(error).__name__,'message':str(error)})
                if isinstance(error,runtime.BudgetHeld): raise
            finally:
                row['total_family_wall_seconds']=time.perf_counter()-route_start
        fit_route('RAW33C',{s:v['history'] for s,v in raw_splits.items()})
        if cached:
            for route in ('E33K','E512K'): fit_route(route,{s:v['latent_past'] for s,v in cached.items()})
        else:
            for route in ('E33K','E512K'):
                result['families'][route]={'status':'NOT_RUN','reason':result['E_cache_failure'],
                    'shared_ZERO_and_MEAN_still_available':True}
        result['new_ridge_candidates_completed']=sum(e['status']=='COMPLETE' for r in result['families'].values()
            for e in r.get('candidate_grid',[]) if e['candidate_id'].startswith('RIDGE_'))
        result['candidate_comparisons_completed']=sum(e['status']=='COMPLETE' for r in result['families'].values() for e in r.get('candidate_grid',[]))
        for record in result['source_records']:
            full(); probes.verified_file(record)
        for record in result['source_snapshots']:
            if probes.sha(record['path'])!=record['sha256'] or probes.sha(record['snapshot_path'])!=record['sha256']:
                raise ValueError('SOURCE_CHANGED_DURING_ACTUAL_RUN')
        result['source_artifacts_before_after_SHA_pass']=True
        result['F_reference_scope']='old D2 producer dual-hash proof; no independent new F/E/D forward or CP state load'
        result['status']='COMPLETE' if all(result['families'][r]['status']=='COMPLETE' for r in math.ROUTES) else 'FAILED_PRESERVED'
        full()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type':type(error).__name__,'message':str(error)})
    finally:
        result['body_wall_seconds']=time.perf_counter()-start
        receipt=limit.close(result['status'],result.get('failure')); result['resource_receipt']=receipt
        if receipt.get('terminal_status')!='COMPLETE':
            result.update(status=receipt['terminal_status'],resource_terminal_error=receipt.get('terminal_error'))
        if initialized: runtime.write_new(output/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true'); mode.add_argument('--execute',action='store_true')
    parser.add_argument('--dataset',choices=tuple(DATASETS)); parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--matched-result'); parser.add_argument('--matched-report-manifest',default=str(HERE/'report_matched_readout_v2/MANIFEST.json'))
    parser.add_argument('--tag'); args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','actual_assets_opened':False,'routes':['RAW33C','E33K','E512K'],
            'pool':['zero','mean','seven ridge'],'new_ridge_per_condition':21,'comparison_per_condition':27,
            'requires_new_human_approved_runtime':True})); return 0
    if not all((args.dataset,args.seed,args.matched_result,args.matched_report_manifest,args.tag)):
        parser.error('execute needs exact original condition/audited matched result/canonical report/fresh tag')
    result=execute(args); print(json.dumps({'status':result['status'],'dataset':args.dataset,'seed':args.seed,'tag':args.tag}))
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
