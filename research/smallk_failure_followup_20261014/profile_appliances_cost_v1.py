"""Preparation-only APP selected6 x B1/B8 + shared RAW x B1/B8 whole cost.

Reuse frozen D3 DeploymentPipeline and joint measure_whole without modification.
Actual --execute requires root GO after every training/audit/report job ends.
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
CACHE=ROOT/'.cache'/HERE.name/'appliances_cost'
SEEDS=(92601,92602)
OBJECTIVES=('MSE','MAE','HUBER_DELTA1')
LOCAL_SECONDS=300


def validate_producers(producers):
    expected={(s,o) for s in SEEDS for o in OBJECTIVES};found={};initial={}
    for p in producers:
        key=(p.get('seed'),p.get('objective'));proof=p.get('learning_executed',{})
        if p.get('dataset')!='appliances' or key not in expected or key in found:raise ValueError('APP_EXACT6_UNIQUE_CONDITIONS_REQUIRED')
        if (p.get('status')!='COMPLETE' or proof.get('optimizer_updates')!=600 or proof.get('full_learning_proof_pass') is not True
                or proof.get('E_updated') is not True or proof.get('D_updated') is not True
                or proof.get('frozen_F_input_gradient_nonzero_diagnostic_count',0)<=0
                or p.get('F_frozen_pass') is not True or p.get('F_has_no_parameter_gradients') is not True
                or p.get('initial_fixed_basis_unchanged') is not True):raise ValueError('APP_COMPLETE600_E_D_F_PROOF_REQUIRED')
        rows=p['checkpoints'];index=p['selected_index'];cp=p['selected_checkpoint']
        if ([r['new_update'] for r in rows]!=[0,100,200,400,600] or not all(np.isfinite(r['tune_actual_Y_mse']) for r in rows)
                or index!=min(range(5),key=lambda i:rows[i]['tune_actual_Y_mse']) or rows[index]['checkpoint']!=cp
                or p['selected_new_update']!=rows[index]['new_update'] or p['selected_codec_hash']!=cp['state_sha_old_algorithm']
                or p['selected_is_initial_original_MAIN']!=(p['selected_new_update']==0)):raise ValueError('APP_ACTUAL_Y_TUNE_SELECTED_CP_REQUIRED')
        original=p['original_MAIN_checkpoint']
        if original.get('q_aux_used_in_new_training') is not False or original.get('q_aux_used_in_new_deployment') is not False:
            raise ValueError('APP_ORIGINAL_Q_PRESERVED_AND_UNUSED_REQUIRED')
        signature=(original['checkpoint']['sha256'],p['initial_codec_hash'],p['training_order_sha256'])
        if key[0] in initial and initial[key[0]]!=signature:raise ValueError('APP_SAME_INITIAL_CP_AND_BATCH_ORDER_REQUIRED')
        initial[key[0]]=signature;found[key]=p
    if set(found)!=expected:raise ValueError('APP_EXACT6_UNIQUE_CONDITIONS_REQUIRED')
    return found


def audit_binding(audit,producers,records):
    if not str(audit.get('status','')).startswith('PASS') or audit.get('failed_count')!=0:raise ValueError('APP_PASSED_AUDIT_REQUIRED')
    rows=audit.get('conditions',[])
    expected={(p['dataset'],p['seed'],p['objective']):r['sha256'] for p,r in zip(producers,records)}
    actual={(r['dataset'],r['seed'],r['objective']):r['producer_sha256'] for r in rows}
    if len(rows)!=6 or actual!=expected:raise ValueError('APP_AUDIT_PRODUCER_SHA_BINDING_REQUIRED')


def execute(args):
    import psutil
    import torch
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import run_appliances_losses_v1 as app
    import profile_latency as base
    import profile_joint_cost_v1 as joint
    import followup_runtime as runtime
    if not args.no_contention_attested:raise ValueError('ROOT_NO_TRAINING_AUDIT_REPORT_JOBS_ATTESTATION_REQUIRED')
    output=CACHE/args.tag;output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','rows':[],'planned_rows':14,'new_training_updates':0,'RAW_PLUS_R_cost_measured':False,
            'protected_TEST_accessed':False,'independent_holdout':False,'local_process_cap_seconds':LOCAL_SECONDS,
            'whole_goal_complete':False,'stage_decomposition':False,'q_aux_used_in_new_deployment':False}
    limit=None;bolt=None;codecs={};birth=psutil.Process().create_time();started=time.perf_counter()
    def fast():
        if time.time()-birth>=LOCAL_SECONDS:raise limit.rt.BudgetHeld('APP_COST_PROCESS_WALL_300S')
        return limit.shared.check()
    def full():fast();return limit.check()
    try:
        limit=runtime.FollowupLimit('app_cost_'+args.tag,gpu=True);inputs.install_protected_test_guard()
        records=[{'path':str(Path(p).resolve()),'sha256':probes.sha(p)} for p in args.app_results]
        producers=[probes.read(r['path']) for r in records];found=validate_producers(producers)
        audit=probes.read(args.app_audit);audit_record={'path':str(Path(args.app_audit).resolve()),'sha256':probes.sha(args.app_audit)}
        audit_binding(audit,producers,records);result.update(APP_producers=records,APP_audit=audit_record)
        bundle,bridge,splits,mean,std,provenance,original_sources=app.load_appliances_bundle(Path(args.input_bridge),probes)
        raw=splits['eval'];base.validate_raw_inputs(raw['history_physical'],raw['history_observed'],raw['history'],mean,std,raw['origins'])
        source_records=provenance['source_assets']+[provenance['input_bridge']]
        source_records.extend({'path':r['path'],'sha256':r['sha256']} for r in bridge['artifacts'])
        folder=output/'sources';folder.mkdir(exist_ok=False)
        for name in (Path(__file__).name,'profile_joint_cost_v1.py','profile_latency.py','run_appliances_losses_v1.py','prepare_appliances_inputs.py','run_condition_probes.py','followup_runtime.py'):
            source=HERE/name;digest=probes.sha(source);shutil.copyfile(source,folder/name)
            if probes.sha(folder/name)!=digest:raise ValueError('APP_COST_SOURCE_SNAPSHOT_MISMATCH')
            source_records.append({'path':str(source),'sha256':digest,'snapshot_path':str(folder/name)})
        full();load_started=time.perf_counter();engine=probes.import_old_engine()
        bolt=engine.ex.previous.FrozenBolt();engine.replace_normalizer(bolt);result['F_before']=probes.model_hashes(bolt)
        if result['F_before']!={'old_algorithm':probes.OLD_F_HASH,'bi_algorithm':probes.BI_F_HASH}:raise ValueError('APP_F_DUAL_HASH_REQUIRED')
        bulk={(None,None):raw['raw_forecast']};bulk_records={};original_cps=[]
        for key,p in found.items():
            codec,initial,cp=app.restore_appliances_codec(engine,bundle,key[0],probes);original_cps.append(cp['checkpoint'])
            checked=probes.verified_file(p['selected_checkpoint']);payload=torch.load(checked['path'],map_location='cpu',weights_only=True)
            joint.verify_checkpoint_metadata(payload,p)
            if (probes.old_state_hash(payload['codec'])!=p['selected_codec_hash'] or probes.bi_state_hash(payload['codec'])!=p['selected_checkpoint']['state_sha_Bi_algorithm']
                    or not torch.equal(payload['codec']['fixed_basis'],initial['fixed_basis'])):raise ValueError('APP_SELECTED_STATE_HASH_OR_BASIS_MISMATCH')
            codec.load_state_dict(payload['codec'],strict=True);codec.eval().requires_grad_(False);codecs[key]=codec
            arrays,rec=probes.load_checked_npz(p['selected_inference']['eval']['artifact']);joint.verify_bulk_conditions(arrays,raw,mean,std)
            bulk[key]=arrays['prediction'];bulk_records[key]=rec;source_records.append(checked)
            source_records.extend(probes.verified_file(r) for r in p['source_snapshots']);full()
        result.update(input_provenance=provenance,selected_bulk_sources={str(k):v for k,v in bulk_records.items()},
            original_CP_before=original_cps,original_sources_before=original_sources,
            codec_hashes_before={str(k):probes.old_state_hash(v) for k,v in codecs.items()},
            F_and_all6_codec_load_verify_wall_seconds=time.perf_counter()-load_started,
            co_resident_memory_scope='FrozenBolt plus all6 selected codecs resident; not method-exclusive VRAM',
            historical_full_C_teacher_precompute_cost='unknown earlier generation; excluded',
            original_RAW_PLUS_R_cost='NOT_MEASURED; existing stronger accuracy reference only')
        sys.modules['engine']=engine;spec=importlib.util.spec_from_file_location('APP_cost_original_controls',ROOT/'research/smallk_predictive_alignment_20261013/controls.py')
        controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls);backend=base.TorchBackend(engine)
        result['preparation_before_first_reference_wall_seconds']=time.perf_counter()-started
        with torch.no_grad():
            for key in [(None,None)]+[(s,o) for s in SEEDS for o in OBJECTIVES]:
                codec=codecs.get(key)
                for batch in (1,8):
                    full();reference_started=time.perf_counter();physical=raw['history_physical'][:batch];mask=raw['history_observed'][:batch]
                    normalized=engine.predict(raw['history'][:batch],codec,bolt,type('FastLimit',(),{'check':staticmethod(fast)})())
                    physical_ref=controls.deployment({'mean':mean,'std':std},physical,mask,'RAW_BOLT' if codec is None else 'BOLT',{'codec':codec},bolt)
                    name=f'app_s{key[0]}_{key[1]}_B{batch}';row_folder=output/name;row_folder.mkdir(exist_ok=False)
                    reference=probes.save_npz(row_folder/'MATCHED_REFERENCE.npz',{'normalized':normalized,'physical':physical_ref});full()
                    reference_seconds=time.perf_counter()-reference_started
                    pipeline=base.DeploymentPipeline(backend,codec,bolt,mean,std,engine.ex.preprocess)
                    row=joint.measure_whole(pipeline,physical,mask,{'normalized':normalized,'physical':physical_ref},
                        {'normalized':bulk[key][:batch],'physical':bulk[key][:batch]*std+mean},fast,row_folder,probes.save_npz,base);full()
                    p=found.get(key)
                    row.update(dataset='appliances',seed=key[0],objective=key[1],method='RAW_BOLT' if codec is None else 'APP_SELECTED',
                        selected_new_update=None if p is None else p['selected_new_update'],selected_is_original_MAIN=None if p is None else p['selected_is_initial_original_MAIN'],
                        matched_reference=reference,matched_reference_preparation_wall_seconds=reference_seconds,origins=raw['origins'][:batch].tolist(),
                        original_bulk_source=probes.verified_file(bundle['raw_references']['RAW_BOLT']) if codec is None else bulk_records[key])
                    result['rows'].append(row);probes.write(row_folder/'RESULT.json',row)
                    if row['status']!='COMPLETE':raise ValueError('APP_MATCHED_BATCH_PARITY_FAILED')
                    print(json.dumps({'event':'APP_COST_ROW_COMPLETE','condition':name,'median_ms':row['median_seconds']*1000}),flush=True)
        for r in records+[audit_record]+source_records+original_cps:
            probes.verified_file(r)
            if 'snapshot_path' in r and probes.sha(r['snapshot_path'])!=r['sha256']:raise ValueError('APP_COST_SNAPSHOT_CHANGED')
        for r in bulk_records.values():probes.verified_file(r)
        result['codec_hashes_after']={str(k):probes.old_state_hash(v) for k,v in codecs.items()}
        if result['codec_hashes_after']!=result['codec_hashes_before'] or len(result['rows'])!=14:raise ValueError('APP_COST_CODEC_OR_COVERAGE14_FAILED')
        full();result['status']='COMPLETE'
    except Exception as error:
        budget=getattr(getattr(limit,'rt',None),'BudgetHeld',())
        result.update(status='BUDGET_HELD_PRESERVED' if budget and isinstance(error,budget) else 'FAILED_PRESERVED',failure={'type':type(error).__name__,'message':str(error)});raise
    finally:
        if bolt is not None:
            try:
                result['F_after']=probes.model_hashes(bolt);result['F_has_no_parameter_gradients']=all(p.grad is None for p in bolt.model.parameters())
                if result.get('F_before')!=result['F_after'] or not result['F_has_no_parameter_gradients']:result.update(status='FAILED_PRESERVED',failure={'message':'APP_F_WEIGHT_OR_GRAD_CHANGED'})
            except Exception as error:result.update(status='FAILED_PRESERVED',F_after_unavailable={'type':type(error).__name__,'message':str(error)})
        receipt=limit.close(result['status'],result.get('failure')) if limit else getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        if receipt is not None:result['shared_resource_receipt']=receipt;probes.apply_resource_terminal(result,receipt)
        result['actual_complete_rows']=sum(r['status']=='COMPLETE' for r in result['rows']);probes.write(output/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('input-bridge','app-audit','tag'):parser.add_argument('--'+name)
    parser.add_argument('--app-results',nargs='+');parser.add_argument('--no-contention-attested',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps({'status':'PREPARATION_ONLY','actual_GPU_executed':False,'planned_rows':14,'local_cap_seconds':300,'RAW_PLUS_R_cost_measured':False}));return
    if not all((args.input_bridge,args.app_results,args.app_audit,args.tag,args.no_contention_attested)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):parser.error('execute requires actual bridge/APP6/PASS audit/fresh safe tag/no-contention attestation')
    result=execute(args);print(json.dumps({'status':result['status'],'actual_complete_rows':result['actual_complete_rows']}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
