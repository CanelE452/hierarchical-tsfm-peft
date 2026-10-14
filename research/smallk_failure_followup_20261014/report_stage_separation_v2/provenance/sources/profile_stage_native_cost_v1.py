"""Source-only default: nine-route balanced hybrid native deployment cost.

Actual execution needs the fresh stage authorization/manifest, complete CPU
producers, a SHA-bound independent audit, and root no-contention attestation.
No scientific files, codec, F or GPU are opened on import or describe.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
CACHE=ROOT/'.cache'/HERE.name/'stage_separation_v1'/'cost'
DATASETS=('robin','beijing_aotizhongxin')
SEEDS=(92601,92602)
FAMILIES=('N1','N2','N3')
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.)
ROUTES=('RAW_BOLT','MAIN_s92601','MAIN_s92602')+tuple(f'{f}_s{s}' for s in SEEDS for f in FAMILIES)
ORDER_SEED=92725
LOCAL_SECONDS=180
REPEATS=9


def balanced_schedule(seed=ORDER_SEED):
    permutation=np.random.default_rng(seed).permutation(len(ROUTES))
    return [[ROUTES[i] for i in np.roll(permutation,-block)] for block in range(REPEATS)]


def validate_producers(producers):
    expected={(d,s) for d in DATASETS for s in SEEDS};found={}
    for p in producers:
        key=(p.get('dataset'),p.get('seed'))
        if key not in expected or key in found:raise ValueError('EXACT4_UNIQUE_STAGE_NATIVE_PRODUCERS_REQUIRED')
        if (p.get('status')!='COMPLETE' or p.get('old_E_D_F_updated') is not False
                or p.get('F_loaded') is not False or p.get('E_forward_replayed') is not False
                or p.get('codec_before')!=p.get('codec_after') or not p.get('codec_before')
                or p.get('codec_terminal_hash')!=p['codec_before']):raise ValueError('COMPLETE_ORIGINAL_FROZEN_CPU_NATIVE_PRODUCER_REQUIRED')
        for family in FAMILIES:
            row=p['families'][family];grid=row.get('alpha_grid',[]);index=row.get('selected_index')
            if (row.get('status')!='COMPLETE' or row.get('diagnostic_only') is not False
                    or tuple(r['alpha'] for r in grid)!=ALPHAS
                    or not all(r['tune_actual_Y_mse'] is not None and np.isfinite(r['tune_actual_Y_mse']) for r in grid)
                    or index!=min(range(7),key=lambda i:grid[i]['tune_actual_Y_mse'])
                    or row['selected_alpha']!=grid[index]['alpha']
                    or row['selected_model_sha256']!=grid[index]['model_sha256']
                    or row['selected_coefficient_artifact']!=grid[index]['coefficient_artifact']):
                raise ValueError('COMPLETE_ACTUAL_Y_TUNE_SELECTED_DEPLOYMENT_FAMILY_REQUIRED:'+family)
        if (p['families']['N3']['selected_model_metadata']['fixed_n1_model_sha256']!=p['families']['N1']['selected_model_sha256']
                or p.get('N3_fixed_N1_contract',{}).get('N1_coefficients_reselected_for_N3') is not False):
            raise ValueError('N3_SAME_SELECTED_N1_REQUIRED')
        found[key]=p
    if set(found)!=expected:raise ValueError('EXACT4_UNIQUE_STAGE_NATIVE_PRODUCERS_REQUIRED')
    return found


def audit_binding(audit,producers,records):
    if not str(audit.get('status','')).startswith('PASS') or audit.get('failed_count')!=0:
        raise ValueError('ACTUAL_STAGE_NATIVE_AUDIT_PASS_REQUIRED')
    rows=audit.get('conditions',[])
    expected={(p['dataset'],p['seed']):r['sha256'] for p,r in zip(producers,records)}
    actual={(r['dataset'],r['seed']):r['producer_sha256'] for r in rows}
    if len(rows)!=4 or actual!=expected:raise ValueError('STAGE_NATIVE_AUDIT_EXACT4_PRODUCER_SHA_BINDING_REQUIRED')


class HybridNativeRoute:
    def __init__(self,codec,models,family,predict_cpu,to_device):
        if family not in FAMILIES:raise ValueError('ORACLE_EXCLUDED_FROM_DEPLOYMENT')
        self.codec,self.models,self.family=codec,models,family
        self.predict_cpu,self.to_device=predict_cpu,to_device

    def forward_path(self,history,unused_forecaster,return_delta=False):
        if return_delta:raise ValueError('NATIVE_COST_DELTA_PATH_UNSUPPORTED')
        past=self.codec.encode(history).detach().cpu().numpy().copy()
        raw_last=history[:,-1].detach().cpu().numpy().copy()
        value=self.predict_cpu(self.family,self.models[self.family],past,raw_last,self.models['decoder'],
                               n1_model=self.models['N1'])
        value=np.asarray(value)
        if value.dtype!=np.float32 or value.shape!=(len(past),48,raw_last.shape[1]) or not np.isfinite(value).all():
            raise ValueError('FINITE_FP32_NATIVE_DEPLOYMENT_PREDICTION_REQUIRED')
        return self.to_device(value)


def measure_group(pipelines,physical,observed,references,bulk,folder,fast,full,save,base):
    schedule=balanced_schedule();samples={name:[] for name in ROUTES};rows=[]
    backend=pipelines[ROUTES[0]].backend
    try:
        for order in schedule[:2]:
            for name in order:fast();pipelines[name].whole(physical,observed,'vectorized');fast()
        backend.synchronize();backend.reset_memory()
        for block,order in enumerate(schedule):
            full()
            for position,name in enumerate(order):
                fast();backend.synchronize();started=time.perf_counter()
                normalized,physical_output=pipelines[name].whole(physical,observed,'vectorized')
                backend.synchronize();seconds=time.perf_counter()-started
                normalized=backend.to_numpy(normalized)
                checks={'matched_batch_normalized':base.parity(normalized,references[name]['normalized'],1e-5,2e-6),
                    'matched_batch_physical':base.parity(physical_output,references[name]['physical'],2e-3,2e-5),
                    'original_bulk_normalized':base.parity(normalized,bulk[name]['normalized'],1e-5,2e-6),
                    'original_bulk_physical':base.parity(physical_output,bulk[name]['physical'],2e-3,2e-5)}
                samples[name].append({'block':block,'position':position,'seconds':seconds,'normalized':normalized,
                                     'physical':physical_output.copy(),'parity':checks})
                fast()
                if not checks['matched_batch_normalized']['pass'] or not checks['matched_batch_physical']['pass']:
                    raise ValueError('BALANCED_MATCHED_BATCH_PARITY_FAILED')
            full()
    finally:
        raw_seconds={r['block']:r['seconds'] for r in samples['RAW_BOLT']}
        for name,values in samples.items():
            if not values:continue
            target=folder/name;target.mkdir(exist_ok=False)
            artifact=save(target/'TIMED_ALL9.npz',{'normalized':np.stack([r['normalized'] for r in values]),
                'physical':np.stack([r['physical'] for r in values]),'block':np.array([r['block'] for r in values],np.int64),
                'position':np.array([r['position'] for r in values],np.int64)})
            seconds=[r['seconds'] for r in values]
            ratios=[r['seconds']/raw_seconds[r['block']] if r['block'] in raw_seconds else None for r in values]
            passed=len(values)==REPEATS and all(r['parity']['matched_batch_normalized']['pass'] and r['parity']['matched_batch_physical']['pass'] for r in values)
            rows.append({'route':name,'status':'COMPLETE' if passed else 'PARTIAL_OR_FAILED_PRESERVED','batch':len(physical),
                'warmup_each_route':2,'repeats':len(values),'seconds':seconds,'block':[r['block'] for r in values],
                'position':[r['position'] for r in values],'paired_RAW_seconds':[raw_seconds.get(r['block']) for r in values],
                'paired_RAW_ratios':ratios,'paired_RAW_median_ratio':float(np.median(ratios)) if all(r is not None for r in ratios) else None,
                'median_seconds':float(np.median(seconds)),'median_seconds_per_window':float(np.median(seconds))/len(physical),
                'timed_outputs':artifact,'parity':[r['parity'] for r in values],'memory':backend.memory_snapshot(),
                'normalized_D2H_copy_outside_timer':True,'stage_decomposition':False,
                'reference_scope':'fresh matching-batch E/GPU plus declared identical scientific CPU math; no independent numerical native implementation',
                'balanced_scope':'each of9 routes at each position once; carryover/clock/thermal causes remain unresolved',
                'includes':'causalCPUfill/FITnorm,H2D, RAW/MAIN GPU F or native E GPU/latent+currentD2H/CPU33K FP64ridge/CPU D/normalizedH2D, GPU inverse/physicalD2H/sync',
                'excludes':'load,reference,normalized-outputcopy,metadata,SHA/save,guards outside timer',
                'optimized_native_implementation_claim':False,'forecast_improvement_claim':False})
        with (folder/'ROWS.json').open('x',encoding='utf-8') as f:
            json.dump({'schedule':schedule,'rows':rows},f,ensure_ascii=False,indent=2,allow_nan=False)
    return rows


def execute(args):
    if not args.no_contention_attested:raise ValueError('ROOT_NO_LIVE_TRAINING_AUDIT_REPORT_JOBS_ATTESTATION_REQUIRED')
    import stage_separation_runtime_v1 as runtime
    output=CACHE/args.tag;output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','rows':[],'planned_rows':36,'warmup_each_route':2,'repeats':REPEATS,
        'order_seed':ORDER_SEED,'local_process_cap_seconds':LOCAL_SECONDS,'stage_decomposition':False,
        'oracles_in_deployment':False,'coefficient_refit_in_cost':False,'new_training_updates':0,
        'independent_holdout':False,'protected_TEST_accessed':False,'whole_goal_complete':False}
    limit=None;bolt=None;sources=[];codecs={};restored={};started=time.perf_counter();birth=None
    result['source_records_before']=sources
    try:
        limit=runtime.ContinuationLimit('stage_native_cost_'+args.tag,gpu=True)
        import psutil
        import torch
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import run_stage_native_ridge_v1 as worker
        import frozen_decoder_refit as fdr
        import profile_latency as base
        import profile_joint_cost_v1 as joint
        inputs.install_protected_test_guard();birth=psutil.Process().create_time()
        def fast():
            if time.time()-birth>=LOCAL_SECONDS:raise runtime.BudgetHeld('STAGE_NATIVE_COST_PROCESS_WALL_180S')
            return limit.shared.checkfast()
        def full():fast();return limit.checkfull()
        records=[probes.verified_file({'path':str(probes.resolve(p)),'sha256':probes.sha(probes.resolve(p))}) for p in args.native_results]
        producers=[probes.read(r['path']) for r in records];found=validate_producers(producers)
        audit_record=probes.verified_file({'path':str(probes.resolve(args.native_audit)),'sha256':probes.sha(probes.resolve(args.native_audit))})
        audit_binding(probes.read(audit_record['path']),producers,records);sources.extend(records+[audit_record])
        result.update(native_producers=records,native_audit=audit_record,authorization_record=limit.authorization_record,
                      run_manifest_record=limit.run_manifest_record)
        folder=output/'sources';folder.mkdir(exist_ok=False)
        for name in (Path(__file__).name,'METHOD_LOCK_STAGE_NATIVE_COST_v1.md','run_stage_native_ridge_v1.py',
                     'stage_separation_math_v1.py','frozen_decoder_refit.py','profile_latency.py','profile_joint_cost_v1.py',
                     'run_condition_probes.py','stage_separation_runtime_v1.py','fast_storage_scan_v1.py','METHOD_LOCK_STAGE_SEPARATION_v1.md'):
            path=HERE/name;digest=probes.sha(path);shutil.copyfile(path,folder/name)
            if probes.sha(folder/name)!=digest:raise ValueError('SOURCE_SNAPSHOT_MISMATCH')
            sources.append({'path':str(path),'sha256':digest,'snapshot_path':str(folder/name)})
        engine=probes.import_old_engine();bolt=engine.ex.previous.FrozenBolt();engine.replace_normalizer(bolt)
        result['F_before']=probes.model_hashes(bolt)
        if result['F_before']!={'old_algorithm':probes.OLD_F_HASH,'bi_algorithm':probes.BI_F_HASH}:raise ValueError('ORIGINAL_F_DUAL_HASH_REQUIRED')
        data={};models={};bulk={};bulk_sources={};metadata={}
        for dataset in DATASETS:
            full();bridge,splits,mean,std,provenance=probes.load_bridge(args.input_bridge,dataset,inputs.verify_phase_barrier)
            sources.extend(probes.verify_source_records(bridge));sources.extend(provenance['input_artifacts']);sources.append(provenance['input_bridge'])
            split='val' if dataset=='robin' else 'eval'
            raw,raw_record=probes.load_checked_npz(next(r for r in bridge['artifacts'] if r['dataset']==dataset and r['split']==split))
            base.validate_raw_inputs(raw['history_physical'],raw['history_observed'],raw['history'],mean,std,raw['origins'])
            raw_source=probes.check_original_raw(bridge,dataset,splits,mean,std);sources.extend((raw_record,raw_source))
            data[dataset]=(raw,mean,std);models[dataset,'RAW_BOLT']=None;bulk[dataset,'RAW_BOLT']=raw['raw_forecast']
            bulk_sources[dataset,'RAW_BOLT']=raw_source;metadata[dataset,'RAW_BOLT']={'method':'RAW_BOLT','seed':None}
            result.setdefault('input_provenance',{})[dataset]={'bridge':provenance,'physical_source':raw_record}
            for seed in SEEDS:
                full();p=found[dataset,seed];codec,state,cp=probes.load_main_codec(engine,bridge,dataset,seed)
                if cp['checkpoint']['sha256']!=p['MAIN_checkpoint']['checkpoint']['sha256'] or cp['case']['sha256']!=p['MAIN_checkpoint']['case']['sha256']:
                    raise ValueError('NATIVE_COST_EXACT_ORIGINAL_CP_REQUIRED')
                hashes={'old_algorithm':probes.old_state_hash(codec),'bi_algorithm':probes.bi_state_hash(codec)}
                if hashes!=p['codec_before']:raise ValueError('NATIVE_COST_EXACT_ORIGINAL_CODEC_REQUIRED')
                codecs[dataset,seed]=codec;loaded=worker.load_selected_models(p,probes);restored[dataset,seed]=loaded
                if any(loaded[f] is None or loaded[f].sha256!=p['families'][f]['selected_model_sha256'] for f in FAMILIES):raise ValueError('EXACT_SELECTED_NATIVE_MODELS_REQUIRED')
                for r in p['source_records']+p['source_snapshots']+[p['feature_state']['artifact'],p['source_decoder_artifact']]:
                    sources.append(probes.verified_file(r))
                    if 'snapshot_path' in r and probes.sha(r['snapshot_path'])!=r['sha256']:raise ValueError('SCIENTIFIC_SOURCE_SNAPSHOT_CHANGED')
                sources.extend((cp['checkpoint'],cp['case']))
                name=f'MAIN_s{seed}';models[dataset,name]=codec
                old,old_record=probes.load_checked_npz(bridge['declared_asset_hashes']['predictions'][dataset]['main_'+str(seed)])
                bulk[dataset,name]=probes.compare_reference(old,splits,mean,std)[split];bulk_sources[dataset,name]=old_record;sources.append(old_record)
                metadata[dataset,name]={'method':'ORIGINAL_MAIN','seed':seed,'original_checkpoint':cp}
                for family in FAMILIES:
                    row=p['families'][family];name=f'{family}_s{seed}'
                    models[dataset,name]=HybridNativeRoute(codec,loaded,family,worker.predict_selected_cpu,engine.tensor)
                    arrays,checked=probes.load_checked_npz(row['selected_artifacts'][split]);joint.verify_bulk_conditions(arrays,raw,mean,std)
                    if not np.array_equal(arrays['raw_last'],raw['raw_last']):raise ValueError('EXACT_SELECTED_CURRENT_C_ANCHOR_REQUIRED')
                    bulk[dataset,name]=arrays['prediction'];bulk_sources[dataset,name]=checked
                    sources.extend((checked,row['selected_coefficient_artifact']))
                    metadata[dataset,name]={'method':family,'seed':seed,'selected_alpha':row['selected_alpha'],
                        'selected_model_sha256':row['selected_model_sha256'],'selected_coefficient_artifact':row['selected_coefficient_artifact'],
                        'fit_zero_observed_output_counts':row['zero_observed_output_counts']}
        def model_hashes():
            return {'codec':{str(k):{'old_algorithm':probes.old_state_hash(v),'bi_algorithm':probes.bi_state_hash(v)} for k,v in codecs.items()},
                    'native':{str(k):{**{f:v[f].sha256 for f in FAMILIES},'original_CPU_D':fdr.decoder_state_sha256(v['decoder'])} for k,v in restored.items()}}
        result.update(model_hashes_before=model_hashes(),preparation_load_verify_wall_seconds=time.perf_counter()-started,
            co_resident_memory_scope='F+all original codecs+CPU selected coefficients; not method-exclusive memory',
            native_implementation='audited scientific CPU recipe with explicit GPU/CPU round trips; not an optimized implementation',
            historical_teacher_precompute_wall_seconds=None)
        sys.modules['engine']=engine
        spec=importlib.util.spec_from_file_location('stage_native_original_controls',ROOT/'research/smallk_predictive_alignment_20261013/controls.py')
        controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls);backend=base.TorchBackend(engine)
        with torch.no_grad():
            for dataset in DATASETS:
                raw,mean,std=data[dataset]
                for batch in (1,8):
                    full();group=output/f'{dataset}_B{batch}';group.mkdir(exist_ok=False)
                    physical,mask=raw['history_physical'][:batch],raw['history_observed'][:batch]
                    pipelines={};references={};reference_records=[];bulk_group={}
                    for name in ROUTES:
                        fast();codec=models[dataset,name]
                        normalized=engine.predict(raw['history'][:batch],codec,bolt,type('FastLimit',(),{'check':staticmethod(fast)})())
                        physical_ref=controls.deployment({'mean':mean,'std':std},physical,mask,'RAW_BOLT' if codec is None else 'BOLT',{'codec':codec},bolt)
                        reference_records.append({'route':name,**probes.save_npz(group/(name+'_REFERENCE.npz'),{'normalized':normalized,'physical':physical_ref})})
                        references[name]={'normalized':normalized,'physical':physical_ref}
                        bulk_group[name]={'normalized':bulk[dataset,name][:batch],'physical':bulk[dataset,name][:batch]*std+mean}
                        pipelines[name]=base.DeploymentPipeline(backend,codec,bolt,mean,std,engine.ex.preprocess)
                    probes.write(group/'REFERENCES.json',reference_records);full()
                    try:measure_group(pipelines,physical,mask,references,bulk_group,group,fast,full,probes.save_npz,base)
                    finally:
                        if (group/'ROWS.json').is_file():
                            rows=probes.read(group/'ROWS.json')['rows']
                            for r in rows:r.update(dataset=dataset,origins=raw['origins'][:batch].tolist(),
                                source_reference=next(v for v in reference_records if v['route']==r['route']),
                                original_bulk_source=bulk_sources[dataset,r['route']],**metadata[dataset,r['route']])
                            result['rows'].extend(rows)
                    full()
        full();result['model_hashes_after']=model_hashes()
        if result['model_hashes_after']!=result['model_hashes_before']:raise ValueError('FROZEN_CODEC_OR_COEFFICIENT_MUTATED')
        for r in sources:
            fast();probes.verified_file(r)
            if 'snapshot_path' in r and probes.sha(r['snapshot_path'])!=r['sha256']:raise ValueError('SOURCE_SNAPSHOT_CHANGED')
        result['source_records_after']=[probes.verified_file(r) for r in sources]
        if len(result['rows'])!=36 or any(r['status']!='COMPLETE' for r in result['rows']):raise ValueError('COMPLETE36_WHOLE_ROWS_REQUIRED')
        full();result['status']='COMPLETE'
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if bolt is not None:
            try:
                if birth is not None and time.time()-birth>=LOCAL_SECONDS:raise runtime.BudgetHeld('LOCAL180_BEFORE_FINAL_F_HASH')
                result['F_after']=probes.model_hashes(bolt)
                result['F_has_no_parameter_gradients']=all(p.grad is None for p in bolt.model.parameters())
                if result.get('F_before')!=result['F_after'] or not result['F_has_no_parameter_gradients']:
                    raise ValueError('F_WEIGHT_OR_GRAD_CHANGED')
            except Exception as error:
                result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                              failure={'type':type(error).__name__,'message':str(error)})
        if birth is not None and time.time()-birth>=LOCAL_SECONDS:
            result.update(status='BUDGET_HELD_PRESERVED',failure={'message':'LOCAL180_BEFORE_RESOURCE_CLOSE'})
        receipt=limit.close(result['status'],result.get('failure')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        if receipt is not None:
            result['shared_resource_receipt']=receipt
            if receipt.get('terminal_status',result['status'])!='COMPLETE':result.update(status=receipt['terminal_status'],failure=receipt.get('terminal_error'))
        if birth is not None and time.time()-birth>=LOCAL_SECONDS:
            result.update(status='BUDGET_HELD_PRESERVED',failure={'message':'LOCAL180_AFTER_RESOURCE_CLOSE'})
        result['actual_complete_rows']=sum(r['status']=='COMPLETE' for r in result['rows'])
        with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('input-bridge','native-audit','tag'):parser.add_argument('--'+name)
    parser.add_argument('--native-results',nargs='+');parser.add_argument('--no-contention-attested',action='store_true');args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','planned_rows':36,'routes':ROUTES,'warmup_each_route':2,
            'repeats':REPEATS,'order_seed':ORDER_SEED,'local_cap_seconds':LOCAL_SECONDS,'oracle_cost_included':False,
            'native_compute':'E GPU then same scientific CPU feature/ridge/D then normalizedH2D; all transfers included'}));return
    if (not all((args.input_bridge,args.native_results,args.native_audit,args.tag,args.no_contention_attested))
            or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag)):
        parser.error('execute requires original bridge/new4 CPU results/PASS-bound audit/fresh tag/no-contention attestation')
    result=execute(args);print(json.dumps({'status':result['status'],'actual_complete_rows':result['actual_complete_rows']}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
