"""SOURCE ONLY by default: 7-route position-balanced whole deployment costs.

No checkpoint/NPZ/model/GPU reads on import. Requires canonical human resource
authorization, complete new8 producer/audit gates, and explicit root execution.
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
CACHE=ROOT/'.cache'/HERE.name/'sequence_balanced_cost'
AUTHORIZATION=HERE/'continuation_authorization_v1.json'
RUN_MANIFEST=HERE/'continuation_run_manifest_v1.json'
DATASETS=('robin','beijing_aotizhongxin')
SEEDS=(92601,92602)
HEADS=('STATIC_RF1','TEMPORAL_RF3')
ORDER_SEED=92714
LOCAL_SECONDS=180
ROUTES=('RAW_BOLT','MAIN_s92601','MAIN_s92602','STATIC_RF1_s92601','TEMPORAL_RF3_s92601','STATIC_RF1_s92602','TEMPORAL_RF3_s92602')


def balanced_schedule(seed=ORDER_SEED):
    permutation=np.random.default_rng(seed).permutation(len(ROUTES))
    return [[ROUTES[i] for i in np.roll(permutation,-block)] for block in range(7)]


def validate_producers(producers):
    expected={(d,s,h) for d in DATASETS for s in SEEDS for h in HEADS};found={};initial={}
    for p in producers:
        key=(p.get('dataset'),p.get('seed'),p.get('head'));learning=p.get('learning_executed',{})
        if key not in expected or key in found:raise ValueError('NEW_SEQUENCE_EXACT8_UNIQUE_CONDITIONS_REQUIRED')
        if (p.get('status')!='COMPLETE' or p.get('objective')!='Y' or learning.get('optimizer_updates')!=600
                or learning.get('full_learning_proof_pass') is not True or learning.get('E_updated') is not True
                or learning.get('D_updated') is not True or learning.get('Head_updated') is not True
                or learning.get('frozen_F_input_gradient_nonzero_diagnostic_count',0)<=0
                or p.get('F_frozen_pass') is not True or p.get('F_has_no_parameter_gradients') is not True
                or p.get('initial_fixed_basis_unchanged') is not True):raise ValueError('SEQUENCE_COMPLETE600_E_D_HEAD_F_PROOF_REQUIRED')
        rows=p['checkpoints'];index=p['selected_index'];cp=p['selected_checkpoint']
        if ([r['new_update'] for r in rows]!=[0,100,200,400,600] or not all(np.isfinite(r['tune_actual_Y_mse']) for r in rows)
                or index!=min(range(5),key=lambda i:rows[i]['tune_actual_Y_mse']) or cp!=rows[index]['checkpoint']
                or p['selected_new_update']!=rows[index]['new_update']):raise ValueError('SEQUENCE_ACTUAL_Y_TUNE_SELECTION_REQUIRED')
        parity=p['initial_all_split_original_MAIN_parity']
        if set(parity)!={'fit','tune','val' if key[0]=='robin' else 'eval'} or not all(r['pass'] for r in parity.values()):raise ValueError('ZERO_HEAD_INITIAL_ALL_SPLIT_MAIN_PARITY_REQUIRED')
        signature=(p['original_MAIN_checkpoint']['checkpoint']['sha256'],p['initial_state_hashes']['state_sha_old_algorithm'],p['training_order_sha256'])
        if key[:2] in initial and initial[key[:2]]!=signature:raise ValueError('SEQUENCE_SAME_INITIAL_CP_AND_ORDER_REQUIRED')
        initial[key[:2]]=signature;found[key]=p
    if set(found)!=expected:raise ValueError('NEW_SEQUENCE_EXACT8_UNIQUE_CONDITIONS_REQUIRED')
    return found


def audit_binding(audit,producers,records):
    if not str(audit.get('status','')).startswith('PASS') or audit.get('failed_count')!=0:raise ValueError('SEQUENCE_PASSED_ACTUAL_AUDIT_REQUIRED')
    rows=audit.get('conditions',[])
    expected={(p['dataset'],p['seed'],p['head']):r['sha256'] for p,r in zip(producers,records)}
    actual={(r['dataset'],r['seed'],r['head']):r['producer_sha256'] for r in rows}
    if len(rows)!=8 or actual!=expected:raise ValueError('SEQUENCE_AUDIT_PRODUCER_SHA_BINDING_REQUIRED')


class HeadRoute:
    def __init__(self,codec,head,forward_trace):self.codec,self.head,self.forward_trace=codec,head,forward_trace
    def forward_path(self,history,forecaster):return self.forward_trace(self.codec,forecaster,self.head,history)[0]


def measure_group(pipelines,physical,observed,references,bulk,folder,fast,full,save,base):
    schedule=balanced_schedule();samples={name:[] for name in ROUTES};artifact_rows=[]
    backend=pipelines[ROUTES[0]].backend;backend.reset_memory()
    try:
        for warm_round in range(2):
            for name in schedule[warm_round]:fast();pipelines[name].whole(physical,observed,'vectorized')
        backend.synchronize()
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
                if not checks['matched_batch_normalized']['pass'] or not checks['matched_batch_physical']['pass']:raise ValueError('BALANCED_MATCHED_BATCH_PARITY_FAILED')
            full()
    finally:
        raw_seconds={r['block']:r['seconds'] for r in samples['RAW_BOLT']}
        for name in ROUTES:
            values=samples[name]
            if not values:continue
            target=folder/name;target.mkdir(exist_ok=False)
            output=save(target/'TIMED_OUTPUTS.npz',{'normalized':np.stack([r['normalized'] for r in values]),
                'physical':np.stack([r['physical'] for r in values]),'block':np.array([r['block'] for r in values],np.int64),
                'position':np.array([r['position'] for r in values],np.int64)})
            seconds=[r['seconds'] for r in values]
            ratios=[r['seconds']/raw_seconds[r['block']] if r['block'] in raw_seconds else None for r in values]
            passed=len(values)==7 and all(r['parity']['matched_batch_normalized']['pass'] and r['parity']['matched_batch_physical']['pass'] for r in values)
            row={'route':name,'status':'COMPLETE' if passed else 'PARTIAL_OR_FAILED_PRESERVED','batch':len(physical),
                'warmup_each_route':2,'repeats':len(values),'seconds':seconds,'block':[r['block'] for r in values],
                'position':[r['position'] for r in values],'paired_RAW_seconds':[raw_seconds.get(r['block']) for r in values],
                'paired_RAW_ratios':ratios,'paired_RAW_median_ratio':float(np.median(ratios)) if all(r is not None for r in ratios) else None,
                'median_seconds':float(np.median(seconds)),'median_seconds_per_window':float(np.median(seconds))/len(physical),
                'parity':[r['parity'] for r in values],'timed_outputs':output,'memory':backend.memory_snapshot(),
                'normalized_D2H_copy_outside_timer':True,'stage_decomposition':False,
                'reference_scope':'same-batch fresh GPU forward; new-head reference reuses declared worker trace, not an independent head implementation',
                'balanced_scope':'each route occupies every position once; cyclic design does not balance all carryover pairs or establish thermal/clock causes',
                'includes':'same causal CPU fill/FITnorm,H2D,E/F/D/head/C-anchor,inverse,physical D2H,sync',
                'excludes':'load/reference/normalized output copy/metadata/save/guards outside timer'}
            artifact_rows.append(row)
        with (folder/'ROWS.json').open('x',encoding='utf-8') as f:
            json.dump({'schedule':schedule,'rows':artifact_rows,'actual_complete_rows':sum(r['status']=='COMPLETE' for r in artifact_rows)},f,ensure_ascii=False,indent=2,allow_nan=False)
    return artifact_rows


def synchronized_stage_trace(pipeline,physical,observed,fast,base,timings):
    backend=pipeline.backend
    wrapped=pipeline.codec
    codec=wrapped.codec if isinstance(wrapped,HeadRoute) else wrapped
    head=wrapped.head if isinstance(wrapped,HeadRoute) else None
    def stage(name,callback):
        fast();backend.synchronize();started=time.perf_counter();value=callback();backend.synchronize()
        timings[name]=time.perf_counter()-started;fast();return value
    filled=stage('fill',lambda:base.fill_vectorized(physical,observed,pipeline.mean))
    history=stage('normalization',lambda:base.normalize(filled,pipeline.mean,pipeline.std))
    x=stage('H2D',lambda:backend.tensor(history))
    past=stage('E',lambda:x if codec is None else codec.encode(x))
    future=stage('F',lambda:pipeline.forecaster(past))
    decoded=stage('oldD',lambda:(future,None) if codec is None else (codec.decode(future),codec.decode(past[:,-1:])))
    residual=stage('head',lambda:None if head is None else head(past,future))
    def anchor_inverse():
        original=decoded[0] if codec is None else x[:,-1:]+decoded[0]-decoded[1]
        normalized=original if residual is None else original+residual
        return normalized,normalized*backend.tensor(pipeline.std)+backend.tensor(pipeline.mean)
    normalized,physical_tensor=stage('anchor_inverse',anchor_inverse)
    physical_output=stage('D2H',lambda:backend.to_numpy(physical_tensor))
    return normalized,physical_output


def measure_stage_route(pipeline,physical,observed,reference,whole_samples,folder,fast,full,save,base,row_sink):
    timings=[];normalized=[];physical_values=[];parity=[];attempts=[];warmup_completed=0;failure=None
    backend=pipeline.backend;folder.mkdir(exist_ok=False)
    try:
        full()
        for _ in range(2):synchronized_stage_trace(pipeline,physical,observed,fast,base,{});warmup_completed+=1
        backend.reset_memory()
        for repeat in range(7):
            attempt={'repeat':repeat,'timings':{}};attempts.append(attempt)
            future,physical_output=synchronized_stage_trace(pipeline,physical,observed,fast,base,attempt['timings'])
            value=backend.to_numpy(future);timings.append(attempt['timings']);normalized.append(value);physical_values.append(physical_output.copy())
            checks={'matched_batch_normalized':base.parity(value,reference['normalized'],1e-5,2e-6),
                'matched_batch_physical':base.parity(physical_output,reference['physical'],2e-3,2e-5),
                'balanced_whole_same_block_normalized':base.parity(value,whole_samples['normalized'][repeat],1e-5,2e-6),
                'balanced_whole_same_block_physical':base.parity(physical_output,whole_samples['physical'][repeat],2e-3,2e-5)}
            parity.append(checks)
            if not all(v['pass'] for v in checks.values()):raise ValueError('SEPARATELY_SYNCHRONIZED_STAGE_OUTPUT_PARITY_FAILED')
        full()
    except Exception as error:
        failure={'type':type(error).__name__,'message':str(error)};raise
    finally:
        artifact=save(folder/'STAGE_OUTPUTS.npz',{'normalized':np.stack(normalized),'physical':np.stack(physical_values)}) if normalized else None
        complete=len(timings)==7 and failure is None and all(all(c['pass'] for c in row.values()) for row in parity)
        row={'status':'COMPLETE' if complete else 'PARTIAL_OR_NOT_MEASURED_PRESERVED','batch':len(physical),
            'warmup_completed':warmup_completed,'repeats':len(timings),'stage_seconds':timings,'attempted_stages':attempts,
            'stage_median_seconds':{name:float(np.median([r[name] for r in timings])) for name in timings[0]} if timings else {},
            'stage_outputs':artifact,'parity':parity,'failure':failure,'memory':backend.memory_snapshot(),
            'measured_separately_after_all_balanced_whole_trials':True,'stage_sum_is_whole_latency':False,
            'synchronization_overhead_in_each_stage':True,'oldD_scope':'D(future_z) and D(last_z)',
            'RAW_MAIN_head_scope':'host identity plus synchronization overhead when no new head; no head GPU kernel',
            'new_head_scope':'actual residual head response on past prefix and predicted future, plus repeated-last response',
            'causal_or_speedup_claim':False}
        row_sink.append(row)
        with (folder/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(row,f,ensure_ascii=False,indent=2,allow_nan=False)
    return row


def execute(args):
    if not AUTHORIZATION.is_file() or not RUN_MANIFEST.is_file():raise PermissionError('CANONICAL_HUMAN_AUTHORIZATION_AND_RUN_MANIFEST_REQUIRED')
    if not args.no_contention_attested:raise ValueError('ROOT_ALL_TRAINING_AUDIT_REPORT_JOBS_ENDED_ATTESTATION_REQUIRED')
    import continuation_runtime_v2 as runtime
    output=CACHE/args.tag;output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','rows':[],'planned_rows':28,'GPU_process_cap_seconds':LOCAL_SECONDS,'order_seed':ORDER_SEED,
        'new_training_updates':0,'stage_decomposition_requested':True,'planned_stage_rows':28,'stage_rows':[],
        'stage_status':'NOT_MEASURED','stage_reason':'whole trials run first; shared/local180s may leave stages unexecuted',
        'independent_holdout':False,'protected_TEST_accessed':False,'whole_goal_complete':False}
    limit=None;bolt=None;models={};sources=[];started=time.perf_counter()
    try:
        limit=runtime.ContinuationLimit('sequence_cost_'+args.tag,gpu=True)
        import psutil
        import torch
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import run_forecast_sequence_v1 as worker
        import forecast_sequence_decoder as heads
        import profile_latency as base
        import profile_joint_cost_v1 as joint
        inputs.install_protected_test_guard();birth=psutil.Process().create_time()
        def fast():
            if time.time()-birth>=LOCAL_SECONDS:raise limit.rt.BudgetHeld('SEQUENCE_COST_PROCESS_WALL_180S')
            return limit.shared.check()
        def full():fast();return limit.check()
        records=[{'path':str(Path(p).resolve()),'sha256':probes.sha(p)} for p in args.sequence_results]
        producers=[probes.read(r['path']) for r in records];found=validate_producers(producers)
        audit=probes.read(args.sequence_audit);audit_record={'path':str(Path(args.sequence_audit).resolve()),'sha256':probes.sha(args.sequence_audit)}
        audit_binding(audit,producers,records);sources.extend(records+[audit_record])
        result.update(sequence_producers=records,sequence_audit=audit_record,authorization_record=limit.authorization_record,run_manifest_record=limit.run_manifest_record)
        folder=output/'sources';folder.mkdir(exist_ok=False)
        for name in (Path(__file__).name,'METHOD_LOCK_SEQUENCE_COST_v1.md','run_forecast_sequence_v1.py','forecast_sequence_decoder.py','profile_latency.py','profile_joint_cost_v1.py','run_condition_probes.py','continuation_runtime_v1.py','continuation_runtime_v2.py','fast_storage_scan_v1.py'):
            source=HERE/name;digest=probes.sha(source);shutil.copyfile(source,folder/name)
            if probes.sha(folder/name)!=digest:raise ValueError('SEQUENCE_COST_SNAPSHOT_MISMATCH')
            sources.append({'path':str(source),'sha256':digest,'snapshot_path':str(folder/name)})
        data={};bulk={};bulk_sources={};route_metadata={};engine=probes.import_old_engine();bolt=engine.ex.previous.FrozenBolt();engine.replace_normalizer(bolt)
        result['F_before']=probes.model_hashes(bolt)
        if result['F_before']!={'old_algorithm':probes.OLD_F_HASH,'bi_algorithm':probes.BI_F_HASH}:raise ValueError('ORIGINAL_F_DUAL_HASH_REQUIRED')
        for dataset in DATASETS:
            bridge,splits,mean,std,provenance=probes.load_bridge(args.input_bridge,dataset,inputs.verify_phase_barrier)
            sources.extend(probes.verify_source_records(bridge));sources.append(provenance['input_bridge']);sources.extend(provenance['input_artifacts'])
            split='val' if dataset=='robin' else 'eval';rec=next(r for r in bridge['artifacts'] if r['dataset']==dataset and r['split']==split)
            raw,checked=probes.load_checked_npz(rec);base.validate_raw_inputs(raw['history_physical'],raw['history_observed'],raw['history'],mean,std,raw['origins'])
            raw_source=probes.check_original_raw(bridge,dataset,splits,mean,std);sources.append(raw_source);data[dataset]=(raw,mean,std)
            result.setdefault('input_provenance',{})[dataset]={'bridge':provenance,'raw_physical_source':checked}
            bulk[dataset,'RAW_BOLT']=raw['raw_forecast'];models[dataset,'RAW_BOLT']=None
            bulk_sources[dataset,'RAW_BOLT']=raw_source;route_metadata[dataset,'RAW_BOLT']={'method':'RAW_BOLT','seed':None,'selected_new_update':None}
            for seed in SEEDS:
                codec,initial,cp=probes.load_main_codec(engine,bridge,dataset,seed);models[dataset,f'MAIN_s{seed}']=codec
                sources.extend((cp['checkpoint'],cp['case']))
                old_arrays,old_rec=probes.load_checked_npz(bridge['declared_asset_hashes']['predictions'][dataset]['main_'+str(seed)])
                bulk[dataset,f'MAIN_s{seed}']=probes.compare_reference(old_arrays,splits,mean,std)[split];sources.append(old_rec)
                bulk_sources[dataset,f'MAIN_s{seed}']=old_rec
                route_metadata[dataset,f'MAIN_s{seed}']={'method':'ORIGINAL_MAIN','seed':seed,'original_checkpoint':cp,'new_training_updates':0}
                for head_type in HEADS:
                    p=found[dataset,seed,head_type];new_codec,initial,new_cp=probes.load_main_codec(engine,bridge,dataset,seed)
                    head=heads.make_head(dataset,head_type,seed).to('cuda')
                    restored=worker.restore_selected_state(torch,new_codec,head,p['selected_checkpoint'],dataset,seed,head_type,probes)
                    if restored!=p['selected_state_hashes'] or restored['new_update']!=p['selected_new_update']:raise ValueError('SEQUENCE_SELECTED_RESTORED_METADATA_MISMATCH')
                    new_codec.eval().requires_grad_(False);head.eval().requires_grad_(False)
                    if not torch.equal(new_codec.state_dict()['fixed_basis'],initial['fixed_basis']):raise ValueError('SEQUENCE_SELECTED_FIXED_BASIS_CHANGED')
                    route=f'{head_type}_s{seed}';models[dataset,route]=HeadRoute(new_codec,head,worker.forward_trace)
                    arrays,selected_rec=probes.load_checked_npz(p['selected_inference'][split]['artifact']);joint.verify_bulk_conditions(arrays,raw,mean,std)
                    bulk[dataset,route]=arrays['prediction'];sources.extend([selected_rec,p['selected_checkpoint']]+[probes.verified_file(r) for r in p['source_snapshots']]);full()
                    bulk_sources[dataset,route]=selected_rec
                    route_metadata[dataset,route]={'method':head_type,'seed':seed,'selected_new_update':p['selected_new_update'],
                        'selected_is_initial_original_MAIN':p['selected_is_initial_original_MAIN'],'selected_checkpoint':p['selected_checkpoint']}
        def model_hashes():
            return {str(k):{'codec':probes.old_state_hash(v.codec if isinstance(v,HeadRoute) else v),
                'head':probes.old_state_hash(v.head) if isinstance(v,HeadRoute) else None} for k,v in models.items() if v is not None}
        result.update(codec_head_hashes_before=model_hashes(),co_resident_memory_scope='F+all original/new codec/head states resident; not method-exclusive',
            preparation_load_verify_wall_seconds=time.perf_counter()-started,source_records_before=sources)
        sys.modules['engine']=engine;spec=importlib.util.spec_from_file_location('sequence_cost_original_controls',ROOT/'research/smallk_predictive_alignment_20261013/controls.py')
        controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls);backend=base.TorchBackend(engine)
        stage_groups=[]
        with torch.no_grad():
            for dataset in DATASETS:
                raw,mean,std=data[dataset]
                for batch in (1,8):
                    full();group=output/f'{dataset}_B{batch}';group.mkdir(exist_ok=False);physical=raw['history_physical'][:batch];mask=raw['history_observed'][:batch]
                    pipelines={};references={};reference_records=[];bulk_group={}
                    for name in ROUTES:
                        codec=models[dataset,name]
                        normalized=worker.predict_trace(engine,codec.codec,bolt,codec.head,raw['history'][:batch],fast)['prediction'] if isinstance(codec,HeadRoute) else engine.predict(raw['history'][:batch],codec,bolt,type('FastLimit',(),{'check':staticmethod(fast)})())
                        physical_ref=controls.deployment({'mean':mean,'std':std},physical,mask,'RAW_BOLT' if codec is None else 'BOLT',{'codec':codec},bolt)
                        reference_records.append({'route':name,**probes.save_npz(group/(name+'_REFERENCE.npz'),{'normalized':normalized,'physical':physical_ref})})
                        references[name]={'normalized':normalized,'physical':physical_ref};bulk_group[name]={'normalized':bulk[dataset,name][:batch],'physical':bulk[dataset,name][:batch]*std+mean}
                        pipelines[name]=base.DeploymentPipeline(backend,codec,bolt,mean,std,engine.ex.preprocess);fast()
                    probes.write(group/'REFERENCES.json',reference_records);full()
                    try:rows=measure_group(pipelines,physical,mask,references,bulk_group,group,fast,full,probes.save_npz,base)
                    finally:
                        if (group/'ROWS.json').is_file():
                            rows=probes.read(group/'ROWS.json')['rows']
                            for r in rows:
                                r.update(dataset=dataset,origins=raw['origins'][:batch].tolist(),source_reference=next(v for v in reference_records if v['route']==r['route']),
                                    original_bulk_source=bulk_sources[dataset,r['route']],**route_metadata[dataset,r['route']])
                            result['rows'].extend(rows)
                    full()
                    stage_groups.append({'dataset':dataset,'batch':batch,'physical':physical,'mask':mask,'references':references,
                        'pipelines':pipelines,'group':group,'origins':raw['origins'][:batch].tolist()})
            result['stage_status']='RUNNING_SEPARATE_AFTER_WHOLE'
            for entry in stage_groups:
                for name in ROUTES:
                    folder=entry['group']/'synchronized_stages'/name
                    folder.parent.mkdir(exist_ok=True)
                    whole,checked=probes.load_checked_npz(next(r['timed_outputs'] for r in result['rows'] if r['dataset']==entry['dataset'] and r['batch']==entry['batch'] and r['route']==name))
                    before=len(result['stage_rows'])
                    try:
                        measure_stage_route(entry['pipelines'][name],entry['physical'],entry['mask'],entry['references'][name],whole,folder,fast,full,probes.save_npz,base,result['stage_rows'])
                    finally:
                        if len(result['stage_rows'])>before:result['stage_rows'][-1].update(dataset=entry['dataset'],route=name,origins=entry['origins'],whole_output_source=checked)
            result['stage_status']='COMPLETE';result['stage_reason']=None
        for r in sources:
            probes.verified_file(r)
            if 'snapshot_path' in r and probes.sha(r['snapshot_path'])!=r['sha256']:raise ValueError('SEQUENCE_COST_SOURCE_SNAPSHOT_CHANGED')
        result['codec_head_hashes_after']=model_hashes()
        result['source_records_after']=[probes.verified_file(r) for r in sources]
        if (result['codec_head_hashes_after']!=result['codec_head_hashes_before'] or len(result['rows'])!=28 or any(r['status']!='COMPLETE' for r in result['rows'])
                or len(result['stage_rows'])!=28 or any(r['status']!='COMPLETE' for r in result['stage_rows'])):raise ValueError('SEQUENCE_COST_PRESERVATION_OR28_WHOLE_STAGE_COVERAGE_FAILED')
        full();result['status']='COMPLETE'
    except Exception as error:
        budget=getattr(getattr(limit,'rt',None),'BudgetHeld',())
        result.update(status='BUDGET_HELD_PRESERVED' if budget and isinstance(error,budget) else 'FAILED_PRESERVED',failure={'type':type(error).__name__,'message':str(error)})
        if result['stage_status']!='COMPLETE':result.update(stage_status='PARTIAL_OR_NOT_MEASURED',stage_reason=str(error))
        raise
    finally:
        if bolt is not None:
            try:
                result['F_after']=probes.model_hashes(bolt);result['F_has_no_parameter_gradients']=all(p.grad is None for p in bolt.model.parameters())
                if result.get('F_before')!=result['F_after'] or not result['F_has_no_parameter_gradients']:result.update(status='FAILED_PRESERVED',failure={'message':'SEQUENCE_F_WEIGHT_OR_GRAD_CHANGED'})
            except Exception as error:result.update(status='FAILED_PRESERVED',F_after_unavailable={'type':type(error).__name__,'message':str(error)})
        receipt=limit.close(result['status'],result.get('failure')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        if receipt is not None:
            result['shared_resource_receipt']=receipt
            if receipt.get('terminal_status',result['status'])!='COMPLETE':result.update(status=receipt.get('terminal_status',result['status']),failure=receipt.get('terminal_error',result.get('failure')))
        result['actual_complete_rows']=sum(r['status']=='COMPLETE' for r in result['rows'])
        result['actual_complete_stage_rows']=sum(r['status']=='COMPLETE' for r in result['stage_rows'])
        with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('input-bridge','sequence-audit','tag'):parser.add_argument('--'+name)
    parser.add_argument('--sequence-results',nargs='+');parser.add_argument('--no-contention-attested',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','actual_GPU_executed':False,'planned_rows':28,'timed_blocks':7,'routes':list(ROUTES),'order_seed':ORDER_SEED,'local_cap_seconds':180,'authorization_required':True}));return
    if not all((args.input_bridge,args.sequence_results,args.sequence_audit,args.tag,args.no_contention_attested)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):parser.error('execute requires original bridge/new8/PASS bound audit/fresh tag/no-contention attestation')
    result=execute(args);print(json.dumps({'status':result['status'],'actual_complete_rows':result['actual_complete_rows']}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
