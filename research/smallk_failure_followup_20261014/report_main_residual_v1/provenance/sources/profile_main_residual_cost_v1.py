"""SOURCE default: 24 balanced whole-path costs of frozen MAIN corrections.

Actual work requires ROOT GO, fresh authorization, a producer-bound PASS audit
and no-contention attestation. Imports/describe do not open scientific assets.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT/'.cache'/HERE.name/'main_residual_v1'/'cost'
DATASETS = ('robin', 'beijing_aotizhongxin')
ROUTES = ('RAW_BOLT', 'ORIGINAL_MAIN', 'MEAN_CONTROL', 'RAW33C', 'E33K', 'E512K')
FAMILIES = ('RAW33C', 'E33K', 'E512K')
SEED, ORDER_SEED, REPEATS, WARMUP, LOCAL_SECONDS = 92601, 92726, 6, 2, 240


def balanced_schedule(seed=ORDER_SEED):
    import numpy as np
    order = np.random.default_rng(seed).permutation(len(ROUTES))
    return [[ROUTES[i] for i in np.roll(order, -block)] for block in range(REPEATS)]


def audit_binding(audit, producers, records):
    expected = {(d,s) for d in DATASETS for s in (92601,92602)}
    conditions = audit.get('conditions', [])
    if (audit.get('status') != 'PASS' or audit.get('failed_count') != 0
            or len(conditions) != 4 or {(r['dataset'],r['seed']) for r in conditions} != expected
            or any(r.get('status') != 'PASS' for r in conditions)):
        raise ValueError('EXACT_FOUR_CONDITION_MAIN_RESIDUAL_AUDIT_PASS_REQUIRED')
    for p,r in zip(producers,records):
        bound = [c for c in conditions if (c['dataset'],c['seed']) == (p['dataset'],p['seed'])]
        if p['seed'] != SEED or len(bound) != 1 or bound[0]['status'] != 'PASS' or bound[0]['producer_sha256'].lower() != r['sha256']:
            raise ValueError('MAIN_RESIDUAL_COST_PRODUCER_SHA_BINDING_REQUIRED')


class HybridCorrectionRoute:
    def __init__(self, codec, model, route, to_device, mean_control=None):
        if route not in ROUTES[2:]:
            raise ValueError('DECLARED_MAIN_CORRECTION_ROUTE_REQUIRED')
        self.codec, self.model, self.route = codec, model, route
        self.to_device, self.mean_control = to_device, mean_control

    def forward_path(self, history, forecaster, return_delta=False):
        if return_delta:
            raise ValueError('RESIDUAL_COST_DELTA_PATH_UNSUPPORTED')
        past = self.codec.encode(history)
        # Match original ANCHOR_NONLINEAR_D order; E is computed only once.
        main = history[:, -1:] + self.codec.decode(forecaster(past)) - self.codec.decode(past[:, -1:])
        if self.route != 'MEAN_CONTROL' and self.model.solution is None and self.model.mean_control is None:
            return main
        main_cpu = main.detach().cpu().numpy().copy()
        control = self.mean_control if self.route == 'MEAN_CONTROL' else self.model.mean_control
        if control is not None:
            value = control.forecast(main_cpu)
        else:
            feature_input = history if self.route == 'RAW33C' else past
            value = self.model.forecast(feature_input.detach().cpu().numpy().copy(), main_cpu)
        return self.to_device(value)


def measure_group(pipelines, physical, observed, references, bulk, folder, fast, full, save, base):
    import numpy as np
    schedule = balanced_schedule()
    samples = {route:[] for route in ROUTES}
    backend = pipelines['RAW_BOLT'].backend
    try:
        for order in schedule[:WARMUP]:
            for route in order:
                if pipelines[route] is None: continue
                fast(); pipelines[route].whole(physical, observed, 'vectorized'); fast()
        backend.synchronize(); backend.reset_memory()
        for block,order in enumerate(schedule):
            full()
            for position,route in enumerate(order):
                if pipelines[route] is None: continue
                fast(); backend.synchronize(); started = time.perf_counter()
                normalized, physical_output = pipelines[route].whole(physical, observed, 'vectorized')
                backend.synchronize(); seconds = time.perf_counter()-started
                normalized = backend.to_numpy(normalized)
                parity = {'matched_batch_normalized':base.parity(normalized,references[route]['normalized'],1e-5,2e-6),
                    'matched_batch_physical':base.parity(physical_output,references[route]['physical'],.002,2e-5),
                    'original_bulk_normalized':base.parity(normalized,bulk[route]['normalized'],1e-5,2e-6),
                    'original_bulk_physical':base.parity(physical_output,bulk[route]['physical'],.002,2e-5)}
                samples[route].append({'block':block,'position':position,'seconds':seconds,
                    'normalized':normalized,'physical':physical_output.copy(),'parity':parity})
                fast()
                if not parity['matched_batch_normalized']['pass'] or not parity['matched_batch_physical']['pass']:
                    raise ValueError('MATCHED_BATCH_COST_OUTPUT_PARITY_FAILED:'+route)
            full()
    finally:
        rows = []; raw_seconds = {r['block']:r['seconds'] for r in samples['RAW_BOLT']}
        for route,values in samples.items():
            if not values:
                rows.append({'route':route,'status':'NOT_MEASURED','batch':len(physical),'repeats':0})
                continue
            target = folder/route; target.mkdir(exist_ok=False)
            artifact = save(target/'TIMED_ALL6.npz', {'normalized':np.stack([r['normalized'] for r in values]),
                'physical':np.stack([r['physical'] for r in values]),'block':np.array([r['block'] for r in values],np.int64),
                'position':np.array([r['position'] for r in values],np.int64)})
            seconds = [r['seconds'] for r in values]
            ratios = [r['seconds']/raw_seconds[r['block']] if r['block'] in raw_seconds else None for r in values]
            complete = len(values)==REPEATS and all(r['parity']['matched_batch_normalized']['pass'] and r['parity']['matched_batch_physical']['pass'] for r in values)
            rows.append({'route':route,'status':'COMPLETE' if complete else 'PARTIAL_OR_FAILED_PRESERVED',
                'batch':len(physical),'warmup_each_route':WARMUP,'repeats':len(values),'seconds':seconds,
                'block':[r['block'] for r in values],'position':[r['position'] for r in values],
                'paired_RAW_seconds':[raw_seconds.get(r['block']) for r in values],'paired_RAW_ratios':ratios,
                'paired_RAW_median_ratio':float(np.median(ratios)) if all(v is not None for v in ratios) else None,
                'median_seconds':float(np.median(seconds)),'median_seconds_per_window':float(np.median(seconds))/len(physical),
                'timed_outputs':artifact,'parity':[r['parity'] for r in values],'memory':backend.memory_snapshot(),
                'normalized_D2H_copy_outside_timer':True,'stage_decomposition':False})
        with (folder/'ROWS.json').open('x',encoding='utf-8') as handle:
            json.dump({'schedule':schedule,'rows':rows},handle,ensure_ascii=False,indent=2,allow_nan=False)
    return rows


def execute(args):
    import main_residual_runtime_v1 as runtime
    result = {'status':'RUNNING','rows':[],'planned_rows':24,'cost_seed':SEED,'order_seed':ORDER_SEED,
        'warmup_each_route':WARMUP,'repeats':REPEATS,'local_process_cap_seconds':LOCAL_SECONDS,
        'stage_decomposition':False,'new_training_updates':0,'coefficient_refit_in_cost':False,
        'protected_TEST_accessed':False,'independent_holdout':False,'whole_goal_complete':False}
    output = CACHE/args.tag
    limit = None; bolt = None; birth = None; initialized = False; sources = []; codecs = {}; models = {}; means = {}
    try:
        if not args.no_contention_attested: raise ValueError('ROOT_NO_LIVE_FIT_AUDIT_REPORT_ATTESTATION_REQUIRED')
        # No output, NumPy, asset, Torch or GPU access precedes authorization.
        limit = runtime.ContinuationLimit('main_residual_cost_'+args.tag,gpu=True)
        result['gpu_forward_declaration'] = limit.declare_gpu_forward_scope('frozen_F_cost')
        birth = limit.started_unix
        output.mkdir(parents=True,exist_ok=False); initialized = True
        def fast():
            if time.time()-birth >= LOCAL_SECONDS: raise runtime.BudgetHeld('MAIN_RESIDUAL_COST_LOCAL240')
            return limit.shared.checkfast()
        def full():
            fast(); value = limit.checkfull(); fast(); return value
        import numpy as np
        import torch
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import run_main_residual_v2 as worker
        import main_forecast_residual_readout_v1 as readout
        import profile_latency as base
        from threadpoolctl import threadpool_limits
        inputs.install_protected_test_guard()
        producers = []; producer_records = []
        for path,dataset in ((args.robin_result,'robin'),(args.beijing_result,'beijing_aotizhongxin')):
            full(); path = probes.resolve(path); record = probes.verified_file({'path':str(path),'sha256':probes.sha(path)})
            producer = probes.read(path)
            if (producer.get('dataset')!=dataset or producer.get('seed')!=SEED
                    or producer.get('status') not in ('COMPLETE','FAILED_PRESERVED')
                    or producer.get('E_D_F_loaded') is not False or producer.get('E_D_F_updated') is not False
                    or producer.get('E_D_F_forward_replayed') is not False
                    or producer.get('protected_TEST_accessed') is not False or producer.get('future_oracle_read') is not False
                    or producer.get('source_artifacts_before_after_SHA_pass') is not True
                    or producer['permission_gate']['sha256']!=limit.authorization_record['sha256']
                    or producer['run_manifest']['sha256']!=limit.run_manifest_record['sha256']):
                raise ValueError('EXACT_CURRENT_PHASE_CPU_PRODUCER_REQUIRED')
            producers.append(producer); producer_records.append(record); sources.append(record)
        audit_path = probes.resolve(args.audit)
        if audit_path.name != 'MAIN_RESIDUAL_AUDIT_v1_ACTUAL.json': raise ValueError('DECLARED_MAIN_RESIDUAL_AUDIT_REQUIRED')
        audit_record = probes.verified_file({'path':str(audit_path),'sha256':probes.sha(audit_path)})
        audit = probes.read(audit_path)
        audit_binding(audit,producers,producer_records); sources.append(audit_record)
        audit_sources = audit.get('source_records')
        if not isinstance(audit_sources,list) or not audit_sources:
            raise ValueError('PASSED_INDEPENDENT_AUDIT_SOURCE_RECORD_REQUIRED')
        sources.extend(probes.verified_file(record) for record in audit_sources)
        result.update(producers=producer_records,independent_audit=audit_record,
            authorization_record=limit.authorization_record,run_manifest_record=limit.run_manifest_record,
            source_records_before=sources)
        snapshot = output/'sources'; snapshot.mkdir()
        for name in (Path(__file__).name,'run_main_residual_v2.py','main_forecast_residual_readout_v1.py',
                     'main_residual_runtime_v1.py','profile_latency.py','run_condition_probes.py','prepare_inputs.py',
                     'matched_readout_math_v1.py','stage_separation_math_v1.py','METHOD_LOCK_MAIN_RESIDUAL_v1.md','fast_storage_scan_v1.py'):
            path = HERE/name; digest = probes.sha(path); shutil.copyfile(path,snapshot/name)
            if probes.sha(snapshot/name)!=digest: raise ValueError('COST_SOURCE_SNAPSHOT_MISMATCH')
            sources.append({'path':str(path),'sha256':digest,'snapshot_path':str(snapshot/name)})
        started = time.perf_counter()
        engine = probes.import_old_engine(); bolt = engine.ex.previous.FrozenBolt(); engine.replace_normalizer(bolt)
        if bolt.model.training or any(p.requires_grad for p in bolt.model.parameters()):
            raise ValueError('ORIGINAL_F_FROZEN_EVAL_REQUIRED')
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != {'old_algorithm':probes.OLD_F_HASH,'bi_algorithm':probes.BI_F_HASH}:
            raise ValueError('ORIGINAL_F_DUAL_HASH_REQUIRED')
        data = {}; routes = {}; bulk = {}; bulk_sources = {}; metadata = {}
        for producer in producers:
            full(); dataset = producer['dataset']; bridge_path = producer['bridge_provenance']['input_bridge']
            probes.verified_file(bridge_path)
            bridge,splits,mean,std,provenance = probes.load_bridge(bridge_path['path'],dataset,inputs.verify_phase_barrier)
            split = 'val' if dataset=='robin' else 'eval'
            raw_record = next(r for r in bridge['artifacts'] if r['dataset']==dataset and r['split']==split)
            raw,checked = probes.load_checked_npz(raw_record)
            base.validate_raw_inputs(raw['history_physical'],raw['history_observed'],raw['history'],mean,std,raw['origins'])
            sources.extend(probes.verify_source_records(bridge)); sources.extend(provenance['input_artifacts']+[provenance['input_bridge'],checked])
            raw_source = probes.check_original_raw(bridge,dataset,splits,mean,std); sources.append(raw_source)
            codec,state,cp = probes.load_main_codec(engine,bridge,dataset,SEED)
            if (codec.architecture != 'ANCHOR_NONLINEAR_D' or codec.training
                    or any(p.requires_grad for p in codec.parameters())):
                raise ValueError('EXACT_ANCHOR_NONLINEAR_D_FROZEN_EVAL_REQUIRED')
            if cp['checkpoint']['sha256'] != producer['original_CP_file_hash_only']['sha256']:
                raise ValueError('EXACT_ORIGINAL_MAIN_CP_REQUIRED')
            codecs[dataset] = codec
            if {'old_algorithm':probes.old_state_hash(codec),'bi_algorithm':probes.bi_state_hash(codec)} != producer['old_codec_dual_hash_reference']:
                raise ValueError('EXACT_ORIGINAL_MAIN_CODEC_DUAL_HASH_REQUIRED')
            models[dataset] = {}
            for route in FAMILIES:
                try:
                    models[dataset][route] = worker.load_selected_models(
                        {**producer,'families':{route:producer['families'].get(route,{})}},probes)[route]
                except Exception as error:
                    models[dataset][route] = None
                    result.setdefault('unavailable_selected_routes',{})[dataset+':'+route] = {'type':type(error).__name__,'message':str(error)}
            mean_row = producer['controls']['MEAN_CORRECTION']
            arrays,checked_mean = probes.load_checked_npz(mean_row['coefficient_artifact']); sources.append(checked_mean)
            means[dataset] = readout.InterceptOnlyControl(arrays['correction'],arrays['observed_counts'],mean_row['fit_residual_sha256'],mean_row['fit_observed_sha256'])
            if means[dataset].sha256 != mean_row['model_sha256']: raise ValueError('EXACT_SHARED_MEAN_CONTROL_REQUIRED')
            main_arrays,main_source = probes.load_checked_npz(producer['original_MAIN_prediction_source']); sources.append(main_source)
            original = probes.compare_reference(main_arrays,splits,mean,std)[split]
            data[dataset] = (raw,mean,std)
            routes[dataset] = {'RAW_BOLT':None,'ORIGINAL_MAIN':codec,
                'MEAN_CONTROL':HybridCorrectionRoute(codec,None,'MEAN_CONTROL',engine.tensor,means[dataset])}
            bulk[dataset] = {'RAW_BOLT':raw['raw_forecast'],'ORIGINAL_MAIN':original}
            bulk_sources[dataset] = {'RAW_BOLT':raw_source,'ORIGINAL_MAIN':main_source}
            metadata[dataset] = {name:{'selected_candidate_id':name,'selected_alpha':None,'scientific_route_status':'COMPLETE'} for name in ROUTES[:3]}
            mean_predictions,mean_pred_record = probes.load_checked_npz(mean_row['artifacts'][split]); sources.append(mean_pred_record)
            if not np.array_equal(mean_predictions['origins'],raw['origins']): raise ValueError('MEAN_ORIGIN_MISMATCH')
            bulk[dataset]['MEAN_CONTROL'] = mean_predictions['prediction']; bulk_sources[dataset]['MEAN_CONTROL'] = mean_pred_record
            for route in FAMILIES:
                row = producer['families'].get(route,{}); model = models[dataset][route]
                routes[dataset][route] = HybridCorrectionRoute(codec,model,route,engine.tensor) if model is not None else None
                metadata[dataset][route] = {'selected_candidate_id':row.get('selected_candidate_id'),
                    'selected_alpha':row.get('selected_alpha'),'selected_model_sha256':row.get('selected_model_sha256'),
                    'scientific_route_status':row.get('status'),'selected_zero_skips_feature_and_CPU_transfers':model is not None and model.solution is None and model.mean_control is None}
                if model is None: continue
                sources.append(probes.verified_file(row['feature_state']['artifact']))
                if row.get('selected_coefficient_artifact') is not None:
                    sources.append(probes.verified_file(row['selected_coefficient_artifact']))
                selected,record = probes.load_checked_npz(row['selected_artifacts'][split]); sources.append(record)
                for field,value in (('target',raw['target']),('observed',raw['observed']),('origins',raw['origins']),('mean',mean),('std',std),('original_MAIN_prediction',original)):
                    if not np.array_equal(selected[field],value,equal_nan=True): raise ValueError('SELECTED_COST_CONDITION_MISMATCH:'+field)
                bulk[dataset][route] = selected['prediction']; bulk_sources[dataset][route] = record
            for record in producer['source_records']+producer['source_snapshots']:
                sources.append(probes.verified_file(record))
                if 'snapshot_path' in record and probes.sha(record['snapshot_path'])!=record['sha256']:
                    raise ValueError('PRODUCER_SOURCE_SNAPSHOT_CHANGED')
            sources.extend((cp['checkpoint'],cp['case']))
            result.setdefault('input_provenance',{})[dataset] = {'bridge':provenance,'physical_source':raw_record}
        def hashes():
            return {'codec':{d:{'old_algorithm':probes.old_state_hash(c),'bi_algorithm':probes.bi_state_hash(c)} for d,c in codecs.items()},
                'correction':{d:{r:(m.sha256 if m is not None else None) for r,m in values.items()} for d,values in models.items()},
                'shared_mean':{d:m.sha256 for d,m in means.items()}}
        result.update(model_hashes_before=hashes(),preparation_load_verify_wall_seconds=time.perf_counter()-started,
            historical_E_MAIN_cache_generation_wall_seconds=None,historical_learning_wall_seconds=None,
            co_resident_memory_scope='F+two original codecs+CPU readout/mean coefficients; not method-exclusive allocation',
            reference_scope='fresh matching-batch original GPU MAIN plus same scientific CPU correction recipe; no independent math or timing replay',
            whole_includes='causal CPU fill/FIT normalization,H2D,E/F/D/anchor,selected CPU feature/correction transfers,GPU inverse/physical D2H/sync',
            whole_excludes='load,reference,normalized output copy,SHA/save/guards outside timer',
            balanced_scope='Latin6 controls position; thermal/clock/carryover and general speed superiority unproven')
        sys.modules['engine'] = engine
        spec = importlib.util.spec_from_file_location('main_residual_original_controls',ROOT/'research/smallk_predictive_alignment_20261013/controls.py')
        controls = importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)
        backend = base.TorchBackend(engine)
        with torch.no_grad(), threadpool_limits(limits=2):
            for dataset in DATASETS:
                raw,mean,std = data[dataset]
                for batch in (1,8):
                    full(); group = output/f'{dataset}_B{batch}'; group.mkdir()
                    physical,mask = raw['history_physical'][:batch],raw['history_observed'][:batch]
                    pipelines={}; references={}; reference_records=[]; bulk_group={}
                    for route in ROUTES:
                        codec = routes[dataset][route]
                        if route in FAMILIES and codec is None:
                            pipelines[route]=None; continue
                        fast()
                        normalized = engine.predict(raw['history'][:batch],codec,bolt,type('FastLimit',(),{'check':staticmethod(fast)})())
                        physical_ref = controls.deployment({'mean':mean,'std':std},physical,mask,'RAW_BOLT' if route=='RAW_BOLT' else 'BOLT',{'codec':codec},bolt)
                        record = probes.save_npz(group/(route+'_REFERENCE.npz'),{'normalized':normalized,'physical':physical_ref})
                        reference_records.append({'route':route,**record}); references[route]={'normalized':normalized,'physical':physical_ref}
                        bulk_group[route]={'normalized':bulk[dataset][route][:batch],'physical':bulk[dataset][route][:batch]*std+mean}
                        pipelines[route]=base.DeploymentPipeline(backend,codec,bolt,mean,std,engine.ex.preprocess)
                    probes.write(group/'REFERENCES.json',reference_records); full()
                    try: measure_group(pipelines,physical,mask,references,bulk_group,group,fast,full,probes.save_npz,base)
                    finally:
                        if (group/'ROWS.json').is_file():
                            rows=probes.read(group/'ROWS.json')['rows']
                            for row in rows:
                                row.update(dataset=dataset,seed=SEED,origins=raw['origins'][:batch].tolist(),**metadata[dataset][row['route']])
                                row['source_reference']=next((r for r in reference_records if r['route']==row['route']),None)
                                row['original_bulk_source']=bulk_sources[dataset].get(row['route'])
                            result['rows'].extend(rows)
        full(); result['model_hashes_after']=hashes()
        result['codecs_have_no_parameter_gradients']=all(p.grad is None for codec in codecs.values() for p in codec.parameters())
        if result['model_hashes_before']!=result['model_hashes_after']: raise ValueError('FROZEN_CODEC_OR_CPU_CORRECTION_CHANGED')
        if not result['codecs_have_no_parameter_gradients']: raise ValueError('ORIGINAL_CODEC_GRADIENT_CREATED')
        for record in sources:
            fast(); probes.verified_file(record)
            if 'snapshot_path' in record and probes.sha(record['snapshot_path'])!=record['sha256']: raise ValueError('COST_SOURCE_CHANGED')
        result['source_records']=sources
        result['source_records_after']=[probes.verified_file(record) for record in sources]
        result['source_artifacts_before_after_SHA_pass']=True
        result['status']='COMPLETE' if len(result['rows'])==24 and all(r['status']=='COMPLETE' for r in result['rows']) else 'FAILED_PRESERVED'
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      failure={'type':type(error).__name__,'message':str(error)})
    finally:
        result['source_records']=sources
        if bolt is not None:
            try:
                result['F_after']=probes.model_hashes(bolt)
                result['F_has_no_parameter_gradients']=all(p.grad is None for p in bolt.model.parameters())
                if result.get('F_before')!=result['F_after'] or not result['F_has_no_parameter_gradients']: raise ValueError('F_WEIGHT_OR_GRAD_CHANGED')
            except Exception as error:
                result.update(status='FAILED_PRESERVED',failure={'type':type(error).__name__,'message':str(error)})
        if birth is not None and time.time()-birth>=LOCAL_SECONDS: result.update(status='BUDGET_HELD_PRESERVED',failure={'message':'LOCAL240_BEFORE_CLOSE'})
        receipt=limit.close(result['status'],result.get('failure')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        result['resource_receipt']=receipt
        if receipt and receipt.get('terminal_status',result['status'])!='COMPLETE': result.update(status=receipt['terminal_status'],failure=receipt.get('terminal_error'))
        if birth is not None and time.time()-birth>=LOCAL_SECONDS: result.update(status='BUDGET_HELD_PRESERVED',failure={'message':'LOCAL240_AFTER_CLOSE'})
        result['actual_complete_rows']=sum(r['status']=='COMPLETE' for r in result['rows'])
        if initialized:
            with (output/'RESULT.json').open('x',encoding='utf-8') as handle: json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true'); mode.add_argument('--execute',action='store_true')
    for name in ('robin-result','beijing-result','audit','tag'): parser.add_argument('--'+name)
    parser.add_argument('--no-contention-attested',action='store_true'); args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','routes':ROUTES,'planned_rows':24,'seed':SEED,
            'warmup_each_route':WARMUP,'repeats':REPEATS,'local_cap_seconds':LOCAL_SECONDS,'fresh_runtime_first':True})); return
    if not all((args.robin_result,args.beijing_result,args.audit,args.tag,args.no_contention_attested)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):
        parser.error('execute requires two seed92601 producers/PASS audit/fresh tag/no-contention attestation')
    result=execute(args); print(json.dumps({'status':result['status'],'actual_complete_rows':result['actual_complete_rows']}))
    if result['status']!='COMPLETE': raise SystemExit(1)


if __name__=='__main__': main()
