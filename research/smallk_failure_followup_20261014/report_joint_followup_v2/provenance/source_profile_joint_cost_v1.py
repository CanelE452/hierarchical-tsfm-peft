"""Preparation-only unless explicitly executed after joint fits/audits settle.

28 whole deployment rows: selected joint12 x B1/B8 plus shared RAW2 x B1/B8.
The existing D3 numerical pipeline/vectorized causal fill is reused unchanged.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'joint_cost'
DATASETS = ('robin', 'beijing_aotizhongxin')
SEEDS = (92601, 92602)
OBJECTIVES = ('Y', 'RAW', 'MIXED')
LOCAL_SECONDS = 300


def validate_joint_producers(producers):
    expected = {(dataset, seed, objective) for dataset in DATASETS for seed in SEEDS for objective in OBJECTIVES}
    found, initialization = {}, {}
    for producer in producers:
        key = (producer.get('dataset'), producer.get('seed'), producer.get('objective'))
        if key not in expected or key in found:
            raise ValueError('JOINT_EXACT12_UNIQUE_CONDITIONS_REQUIRED')
        learning = producer.get('learning_executed', {})
        if (producer.get('status') != 'COMPLETE' or learning.get('optimizer_updates') != 600
                or learning.get('full_learning_proof_pass') is not True
                or learning.get('E_updated') is not True or learning.get('D_updated') is not True
                or learning.get('frozen_F_input_gradient_nonzero_diagnostic_count', 0) <= 0
                or producer.get('F_frozen_pass') is not True or producer.get('F_has_no_parameter_gradients') is not True
                or producer.get('initial_fixed_basis_unchanged') is not True):
            raise ValueError('JOINT_COMPLETE_600_LEARNING_FROZEN_F_PROOF_REQUIRED')
        cp = producer['selected_checkpoint']
        selected_index = producer['selected_index']
        rows = producer['checkpoints']
        if ([row['new_update'] for row in rows] != [0, 100, 200, 400, 600]
                or not all(np.isfinite(row['tune_actual_Y_mse']) for row in rows)
                or selected_index != min(range(len(rows)), key=lambda i: rows[i]['tune_actual_Y_mse'])
                or rows[selected_index]['checkpoint'] != cp
                or producer['selected_new_update'] != rows[selected_index]['new_update']
                or producer['selected_codec_hash'] != cp['state_sha_old_algorithm']
                or producer['selected_is_initial_original_MAIN'] != (producer['selected_new_update'] == 0)):
            raise ValueError('ACTUAL_Y_TUNE_SELECTED_CP_CONTRACT_MISMATCH')
        init = producer['same_initialization_across_objectives_key']
        if (init['dataset'], init['seed']) != key[:2]:
            raise ValueError('JOINT_INITIALIZATION_KEY_METADATA_MISMATCH')
        shared = (init['original_MAIN_CP_sha256'], init['initial_codec_hash'], producer['training_order_sha256'])
        group = key[:2]
        if group in initialization and initialization[group] != shared:
            raise ValueError('JOINT_SAME_INITIALIZATION_PROOF_MISMATCH')
        initialization[group] = shared
        found[key] = producer
    if set(found) != expected:
        raise ValueError('JOINT_EXACT12_UNIQUE_CONDITIONS_REQUIRED')
    return found


def verify_checkpoint_metadata(saved, producer):
    if (saved.get('architecture') != 'ANCHOR_NONLINEAR_D' or saved.get('new_joint_comparison') is not True
            or saved.get('seed') != producer['seed'] or saved.get('objective') != producer['objective']
            or saved.get('new_update') != producer['selected_new_update']):
        raise ValueError('SELECTED_NEW_JOINT_CHECKPOINT_METADATA_MISMATCH')


def verify_audit_binding(audit, producers, records):
    rows = audit.get('conditions', [])
    expected = {(p['dataset'], p['seed'], p['objective']): r['sha256'] for p, r in zip(producers, records)}
    actual = {(r['dataset'], r['seed'], r['objective']): r['producer_sha256'] for r in rows}
    if len(rows) != 12 or actual != expected:
        raise ValueError('JOINT_AUDIT_PRODUCER_SHA_BINDING_MISMATCH')


def verify_bulk_conditions(arrays, values, mean, std):
    for key, exact in (('target', values['target']), ('observed', values['observed']),
                       ('origins', values['origins']), ('mean', mean), ('std', std)):
        if arrays[key].dtype != exact.dtype or not np.array_equal(arrays[key], exact, equal_nan=True):
            raise ValueError('SELECTED_BULK_ORIGINAL_CONDITION_MISMATCH:' + key)
    if arrays['prediction'].dtype != np.float32 or arrays['prediction'].shape != values['target'].shape or not np.isfinite(arrays['prediction']).all():
        raise ValueError('FINITE_FP32_SELECTED_BULK_PREDICTION_REQUIRED')


def measure_whole(pipeline, physical, observed, reference, bulk, fast_check, output, save, base):
    for _ in range(2):
        pipeline.whole(physical, observed, 'vectorized')
        fast_check()
    pipeline.backend.synchronize()
    pipeline.backend.reset_memory()
    seconds, normalized, physical_outputs = [], [], []
    for _ in range(7):
        fast_check()
        pipeline.backend.synchronize()
        started = time.perf_counter()
        forecast, actual_physical = pipeline.whole(physical, observed, 'vectorized')
        pipeline.backend.synchronize()
        seconds.append(time.perf_counter() - started)
        normalized.append(pipeline.backend.to_numpy(forecast))
        physical_outputs.append(actual_physical.copy())
    output_record = save(output / 'TIMED_ALL7.npz', {'normalized': np.stack(normalized), 'physical': np.stack(physical_outputs)})
    rows = [{'repeat': i,
        'matched_batch_normalized': base.parity(normalized[i], reference['normalized'], 1e-5, 2e-6),
        'matched_batch_physical': base.parity(physical_outputs[i], reference['physical'], 2e-3, 2e-5),
        'original_bulk_normalized': base.parity(normalized[i], bulk['normalized'], 1e-5, 2e-6),
        'original_bulk_physical': base.parity(physical_outputs[i], bulk['physical'], 2e-3, 2e-5)} for i in range(7)]
    passed = all(row['matched_batch_normalized']['pass'] and row['matched_batch_physical']['pass'] for row in rows)
    return {'status': 'COMPLETE' if passed else 'FAILED_PRESERVED', 'batch': len(physical), 'warmup': 2, 'repeats': 7,
        'seconds': seconds, 'median_seconds': float(np.median(seconds)), 'median_seconds_per_window': float(np.median(seconds))/len(physical),
        'timed_outputs': output_record, 'parity': rows, 'memory': pipeline.backend.memory_snapshot(),
        'preprocessing': 'same D3 vectorized causal CPU fill', 'normalized_D2H_copy_outside_timer': True,
        'includes': 'CPU raw causal fill/FITnorm,H2D,E/F/D/anchor,inverse,D2H,sync',
        'excludes': 'model/CP/disk load,cache/reference preparation,normalized output copy,SHA/save,fullresource guards',
        'stage_decomposition': False, 'forecasting_improvement_claim': False}


def execute(args):
    import psutil
    import torch
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import profile_latency as base
    import followup_runtime as runtime
    if not args.no_contention_attested:
        raise ValueError('ROOT_NO_LIVE_FITS_AUDITS_REPORT_JOBS_ATTESTATION_REQUIRED')
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'rows': [], 'planned_rows': 28, 'new_training_updates': 0,
        'protected_TEST_accessed': False, 'stage_decomposition': False, 'independent_holdout': False,
        'no_contention_attestation': True, 'local_process_cap_seconds': LOCAL_SECONDS}
    limit, bolt, codecs = None, None, {}
    process_birth = psutil.Process().create_time()
    preparation_started = time.perf_counter()
    def fast_check():
        if time.time()-process_birth >= LOCAL_SECONDS:
            raise limit.rt.BudgetHeld('JOINT_COST_PROCESS_WALL_300S')
        return limit.shared.check()
    def full_check():
        fast_check()
        return limit.check()
    try:
        limit = runtime.FollowupLimit('joint_cost_' + args.tag, gpu=True)
        inputs.install_protected_test_guard()
        audit, audit_record = probes.read(args.joint_audit), {'path': str(Path(args.joint_audit).resolve()), 'sha256': probes.sha(args.joint_audit)}
        if not str(audit.get('status', '')).startswith('PASS') or audit.get('failed_count') != 0:
            raise ValueError('PASSED_JOINT_INDEPENDENT_AUDIT_REQUIRED')
        producer_records = [probes.verified_file({'path': str(Path(path).resolve()), 'sha256': probes.sha(path)}) for path in args.joint_results]
        producers = [probes.read(record['path']) for record in producer_records]
        found = validate_joint_producers(producers)
        verify_audit_binding(audit, producers, producer_records)
        result.update(joint_producers=producer_records, joint_independent_audit=audit_record)
        snapshots = []
        folder = output / 'sources'; folder.mkdir(exist_ok=False)
        for name in (Path(__file__).name, 'METHOD_LOCK_JOINT_COST_v1.md', 'profile_latency.py', 'run_condition_probes.py', 'prepare_inputs.py', 'followup_runtime.py'):
            source = HERE/name; digest = probes.sha(source)
            shutil.copyfile(source, folder/name)
            if probes.sha(folder/name) != digest:raise ValueError('HELPER_SNAPSHOT_SHA_MISMATCH')
            snapshots.append({'path': str(source), 'sha256': digest, 'snapshot_path': str(folder/name), 'snapshot_sha256': digest})
        datasets, old_sources, original_cps, joint_sources, bulk = {}, [], [], [], {}
        for dataset in DATASETS:
            bridge, splits, mean, std, provenance = probes.load_bridge(args.input_bridge, dataset, inputs.verify_phase_barrier)
            old_sources.extend(probes.verify_source_records(bridge))
            split = 'val' if dataset == 'robin' else 'eval'
            rec = next(r for r in bridge['artifacts'] if r['dataset']==dataset and r['split']==split)
            raw, raw_record = probes.load_checked_npz(rec)
            base.validate_raw_inputs(raw['history_physical'], raw['history_observed'], raw['history'], mean, std, raw['origins'])
            datasets[dataset] = (bridge, splits, mean, std, split, raw)
            bulk[dataset, None, None] = {'normalized': splits[split]['raw_forecast'], 'source': probes.check_original_raw(bridge, dataset, splits, mean, std)}
            result.setdefault('input_provenance', {})[dataset] = {'bridge': provenance, 'raw_physical_input': raw_record}
            full_check()
        load_started = time.perf_counter()
        engine = probes.import_old_engine()
        bolt = engine.ex.previous.FrozenBolt(); engine.replace_normalizer(bolt)
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}:
            raise ValueError('ORIGINAL_F_DUAL_HASH_MISMATCH')
        for key, producer in found.items():
            dataset, seed, objective = key; bridge, splits, mean, std, split, raw = datasets[dataset]
            codec, initial_state, original_cp = probes.load_main_codec(engine, bridge, dataset, seed)
            original_cps.extend((original_cp['checkpoint'], original_cp['case']))
            cp = probes.verified_file(producer['selected_checkpoint'])
            saved = torch.load(cp['path'], map_location='cpu', weights_only=True); verify_checkpoint_metadata(saved, producer)
            if (probes.old_state_hash(saved['codec']) != producer['selected_codec_hash']
                    or probes.bi_state_hash(saved['codec']) != producer['selected_checkpoint']['state_sha_Bi_algorithm']
                    or not torch.equal(saved['codec']['fixed_basis'], initial_state['fixed_basis'])):
                raise ValueError('SELECTED_CODEC_HASH_OR_FIXED_BASIS_MISMATCH')
            codec.load_state_dict(saved['codec'], strict=True); codec.eval().requires_grad_(False)
            codecs[key] = codec
            joint_sources.extend(probes.verified_file(r) for r in producer['source_snapshots'])
            artifact = producer['selected_inference'][split]['artifact']
            arrays, checked = probes.load_checked_npz(artifact); verify_bulk_conditions(arrays, splits[split], mean, std)
            bulk[key] = {'normalized': arrays['prediction'], 'source': checked, 'selected_checkpoint': cp}
            full_check()
        result.update(original_sources_before=old_sources, original_CP_before=original_cps, joint_sources_before=joint_sources,
            helper_sources_before=snapshots, codec_hashes_before={str(key):probes.old_state_hash(c) for key,c in codecs.items()},
            F_and_all12_original_structure_selected_CP_load_verify_wall_seconds=time.perf_counter()-load_started,
            co_resident_memory_scope='All12 selected codecs and FrozenBolt remain resident for every RAW/joint row; not method-exclusive allocation',
            historical_full_C_teacher_precompute_cost='unknown earlier cached generation; excluded from this actual deployment receipt')
        sys.modules['engine'] = engine
        spec=importlib.util.spec_from_file_location('joint_cost_original_controls', ROOT/'research/smallk_predictive_alignment_20261013/controls.py')
        controls=importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)
        backend=base.TorchBackend(engine)
        result['preparation_before_first_reference_wall_seconds']=time.perf_counter()-preparation_started
        with torch.no_grad():
            for dataset in DATASETS:
                bridge,splits,mean,std,split,raw=datasets[dataset]
                conditions=[(dataset,None,None)]+[(dataset,seed,obj) for seed in SEEDS for obj in OBJECTIVES]
                for key in conditions:
                    codec=codecs.get(key)
                    for batch in (1,8):
                        full_check()
                        reference_started=time.perf_counter()
                        physical,mask=raw['history_physical'][:batch],raw['history_observed'][:batch]
                        normalized=engine.predict(raw['history'][:batch],codec,bolt,type('FastLimit',(),{'check':staticmethod(fast_check)})())
                        physical_ref=controls.deployment({'mean':mean,'std':std},physical,mask,'RAW_BOLT' if codec is None else 'BOLT',{'codec':codec},bolt)
                        name=f'{dataset}_s{key[1]}_{key[2]}_B{batch}'; row_folder=output/name; row_folder.mkdir(exist_ok=False)
                        reference=probes.save_npz(row_folder/'MATCHED_REFERENCE.npz',{'normalized':normalized,'physical':physical_ref})
                        full_check()
                        reference_preparation_seconds=time.perf_counter()-reference_started
                        pipeline=base.DeploymentPipeline(backend,codec,bolt,mean,std,engine.ex.preprocess)
                        row=measure_whole(pipeline,physical,mask,{'normalized':normalized,'physical':physical_ref},
                            {'normalized':bulk[key]['normalized'][:batch],'physical':bulk[key]['normalized'][:batch]*std+mean},fast_check,row_folder,probes.save_npz,base)
                        full_check()
                        producer=found.get(key)
                        row.update(dataset=dataset,seed=key[1],objective=key[2],method='RAW_BOLT' if codec is None else 'JOINT_SELECTED',
                            matched_reference_preparation_save_verify_wall_seconds=reference_preparation_seconds,
                            selected_new_update=None if producer is None else producer['selected_new_update'],
                            selected_is_original_MAIN=None if producer is None else producer['selected_is_initial_original_MAIN'],
                            matched_reference=reference,original_bulk_source=bulk[key]['source'],origins=raw['origins'][:batch].tolist())
                        result['rows'].append(row); probes.write(row_folder/'RESULT.json',row)
                        if row['status']!='COMPLETE':raise ValueError('JOINT_COST_MATCHED_BATCH_PARITY_FAILED')
                        print(json.dumps({'event':'JOINT_COST_ROW_COMPLETE','condition':name,'median_ms':row['median_seconds']*1000}),flush=True)
        for record in producer_records+[audit_record]+old_sources+original_cps+joint_sources+snapshots:
            probes.verified_file(record)
            if 'snapshot_path' in record and probes.sha(record['snapshot_path'])!=record['sha256']:raise ValueError('HELPER_SNAPSHOT_CHANGED')
        for value in bulk.values():
            probes.verified_file(value['source'])
            if 'selected_checkpoint' in value:
                probes.verified_file(value['selected_checkpoint'])
        for entry in result['input_provenance'].values():
            probes.verified_file(entry['raw_physical_input'])
            probes.verified_file(entry['bridge']['input_bridge'])
            for record in entry['bridge']['input_artifacts']:
                probes.verified_file(record)
        result['codec_hashes_after']={str(key):probes.old_state_hash(c) for key,c in codecs.items()}
        if result['codec_hashes_before']!=result['codec_hashes_after'] or len(result['rows'])!=28:
            raise ValueError('CODEC_PRESERVATION_OR_COST28_COVERAGE_FAILED')
        full_check(); result['status']='COMPLETE'
    except Exception as error:
        budget_type = getattr(getattr(limit, 'rt', None), 'BudgetHeld', ())
        result.update(status='BUDGET_HELD_PRESERVED' if budget_type and isinstance(error, budget_type) else 'FAILED_PRESERVED',
                      failure={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if bolt is not None:
            try:
                result['F_after']=probes.model_hashes(bolt)
                result['F_has_no_parameter_gradients']=all(p.grad is None for p in bolt.model.parameters())
                if not result['F_has_no_parameter_gradients']:result.update(status='FAILED_PRESERVED',failure={'message':'F_PARAMETER_GRADIENT_CREATED'})
                if result.get('F_before')!=result['F_after']:result.update(status='FAILED_PRESERVED',failure={'message':'F_BACKBONE_HASH_CHANGED'})
            except Exception as error:
                result.update(status='FAILED_PRESERVED',F_after_unavailable={'type':type(error).__name__,'message':str(error)})
        if limit is not None:
            result['shared_resource_receipt']=limit.close(result['status'],result.get('failure'));probes.apply_resource_terminal(result,result['shared_resource_receipt'])
        elif getattr(runtime, 'LAST_CONSTRUCTOR_TERMINAL', None) is not None:
            result['shared_resource_receipt']=runtime.LAST_CONSTRUCTOR_TERMINAL
            probes.apply_resource_terminal(result,runtime.LAST_CONSTRUCTOR_TERMINAL)
        result['actual_complete_rows']=sum(row['status']=='COMPLETE' for row in result['rows'])
        probes.write(output/'RESULT.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--describe',action='store_true');m.add_argument('--execute',action='store_true')
    p.add_argument('--input-bridge');p.add_argument('--joint-results',nargs='+');p.add_argument('--joint-audit');p.add_argument('--tag');p.add_argument('--no-contention-attested',action='store_true')
    args=p.parse_args()
    if args.describe:
        print(json.dumps({'status':'PREPARATION_ONLY','actual_GPU_executed':False,'planned_rows':28,'local_cap_seconds':300,'requires_root_GO':True}));return
    if not all((args.input_bridge,args.joint_results,args.joint_audit,args.tag,args.no_contention_attested)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):p.error('execute requires bridge,joint12results,PASSaudit,freshtag,no-contention attestation')
    result=execute(args);print(json.dumps({'status':result['status'],'actual_complete_rows':result['actual_complete_rows']}))
    if result['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
