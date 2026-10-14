"""Four actual failed families: dynamic CPU-head prototype and stage differences.

This is diagnostic inference, not latency measurement. The prototype invokes
the original science APIs, including their coefficient load and memory refresh.
"""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_info
import runtime as rt
import model as bm
import probes as pr
import benchmark as bench
import run as ru
import batch_science_reference as bs

CASES=(('RAW_JOINT_RIDGE','RAW_JOINT_RIDGE'),('RAW_BOLT_RAW_R','RAW_BOLT_RAW_R'),
    ('E_RAW_RIDGE','NO_ALIGN_w0_s92601_E_RAW_RIDGE'),
    ('E_ENRICHED_RIDGE','B_COSINE_w0_s92601_E_ENRICHED_RIDGE'))


def cpu_head_deployment(pack,raw,observed,origins,route,forecaster,model=None,limit=None):
    normalized=bench.preprocess(pack,raw,observed)
    split=route['split'];thin={'history':{split:normalized},'origins':{split:origins}}
    stages={'normalized_input':normalized.copy()}
    block_original,representation_original,ridge_original=pr.block_features,pr.representation_features,pr.ridge_predict
    def blocks(x):
        result=block_original(x);stages['block_features_CPU']=result.copy();return result
    def representation(*args,**kwargs):
        result=representation_original(*args,**kwargs);stages['representation_features_GPU_to_CPU']=result.copy();return result
    def ridge(features,coefficients):
        result=ridge_original(features,coefficients)
        stages.update(features_CPU=np.asarray(features,np.float32).copy(),weight=coefficients['weight'].copy(),
            bias=coefficients['bias'].copy(),delta_CPU=result.copy())
        return result
    class CapturedForecaster:
        def __call__(self,x):
            result=forecaster(x);stages['base_F_GPU_to_CPU']=result.detach().cpu().numpy().copy();return result
    with ExitStack() as context:
        context.enter_context(patch.object(pr,'block_features',blocks))
        context.enter_context(patch.object(pr,'representation_features',representation))
        context.enter_context(patch.object(pr,'ridge_predict',ridge))
        if route['route'] in ('E_RAW_RIDGE','E_ENRICHED_RIDGE'):
            prediction=pr.evaluate_readout(thin,split,route['case'],model,route['readout'],rt,limit)
        elif route['route'] in ('RAW_JOINT_RIDGE','RAW_BOLT_RAW_R'):
            prediction=pr.evaluate_raw_control(thin,split,route['control'],CapturedForecaster(),rt,limit)
        else:raise ValueError('RIDGE_PROTOTYPE_FAMILY_NOT_APPROVED')
    stages['base_normalized']=stages.get('base_F_GPU_to_CPU',normalized[:,-1:]).copy()
    physical=prediction.astype(np.float32)*pack['std'].astype(np.float32)+pack['mean'].astype(np.float32)
    return prediction,physical,stages


def stage_decomposition(historical_physical,cpu_normalized,gpu_normalized,mean,std):
    cpu_physical=cpu_normalized*std+mean;gpu_physical=gpu_normalized*std+mean
    exact_cpu=cpu_normalized.astype(np.float64)*std.astype(np.float64)+mean.astype(np.float64)
    exact_gpu=gpu_normalized.astype(np.float64)*std.astype(np.float64)+mean.astype(np.float64)
    scaled_delta=(gpu_normalized.astype(np.float64)-cpu_normalized.astype(np.float64))*std.astype(np.float64)
    cpu_round=cpu_physical.astype(np.float64)-exact_cpu;gpu_round=gpu_physical.astype(np.float64)-exact_gpu
    historical_residual=historical_physical.astype(np.float64)-gpu_physical.astype(np.float64)
    explanation=scaled_delta+gpu_round-cpu_round+historical_residual
    observed=historical_physical.astype(np.float64)-cpu_physical.astype(np.float64)
    arrays={'CPU_stage_normalized':cpu_normalized,'GPU_stage_normalized':gpu_normalized,
        'CPU_stage_physical':cpu_physical,'GPU_stage_physical_CPU_FP32_inverse':gpu_physical,
        'normalized_stage_delta_times_std_FP64':scaled_delta,'CPU_inverse_rounding_vs_exact_FP64':cpu_round,
        'GPU_stage_inverse_rounding_vs_exact_FP64':gpu_round,
        'historical_vs_current_GPU_stage_physical':historical_residual,
        'historical_minus_CPU_physical_FP64':observed,'error_decomposition_FP64':explanation}
    summary={'normalized_stage_delta_max_abs':float(np.abs(gpu_normalized.astype(np.float64)-cpu_normalized.astype(np.float64)).max()),
        'normalized_stage_delta_times_std_max_abs':float(np.abs(scaled_delta).max()),
        'historical_minus_CPU_head_physical_max_abs':float(np.abs(observed).max()),
        'historical_minus_current_GPU_stage_physical_max_abs':float(np.abs(historical_residual).max()),
        'decomposition_residual_max_abs':float(np.abs(explanation-observed).max()),
        'historical_kernel_cause_confirmed':False}
    return summary,arrays


def cache_arrays(model):
    return {name:value.detach().cpu().numpy().copy() for name,value in model.named_buffers() if name.startswith(('memory_','noise_calls'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--cost-result',type=Path,default=rt.HERE/'cost_batch_recovery_v1/missing132_v1/RESULT.json')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('RIDGE_DIAGNOSTIC_INVALID_EXCLUSIVE_TAG')
    folder=rt.HERE/'ridge_backend_diagnostics_v1'/args.tag;folder.mkdir(parents=True,exist_ok=False)
    with (folder/'SOURCE_AT_RUN.py').open('xb') as h:h.write(Path(__file__).read_bytes())
    limit=rt.Limit(args.tag,gpu=True)
    def check():
        account=limit.check()
        if account['elapsed_seconds']>60:raise rt.BudgetHeld('RIDGE_DIAGNOSTIC_PROCESS_WALL_60SECONDS')
    output={'status':'INCOMPLETE','tag':args.tag,'started_unix':time.time(),'conditions':[],
        'predefined_family_cases':[list(v) for v in CASES],'actual_prototype_calls_per_family':1,
        'new_training_updates':0,'protected_TEST_opened':False,'EVAL_target_arrays_opened':False,
        'current_science_or_tolerance_changed':False,'GPU_process_wall_cap_seconds':60,
        'implementation_id':'prototype_original_science_CPU_FP32_head_dynamic_raw_past',
        'not_latency_measurement':True,'canonical_240_Ridge_rows_executed':False,
        'prototype_includes_original_API_disk_load_and_memory_refresh':True,
        'eventual_timing_requirement':'Preload coefficients/models/verified FIT cache outside timing; include raw preprocessing/GPU E or F/feature D2H/CPU original FP32 head/inverse inside timing',
        'diagnostic_source':{'path':str(Path(__file__).resolve().relative_to(rt.ROOT)),'sha256':rt.sha(__file__),
            'snapshot_path':str((folder/'SOURCE_AT_RUN.py').relative_to(rt.ROOT)),'snapshot_sha256':rt.sha(folder/'SOURCE_AT_RUN.py')}}
    terminal,error='COMPLETE',None
    try:
        cost=rt.read(args.cost_result)
        if cost['status']!='FAILED_PRESERVED' or len(cost['records'])!=536:raise ValueError('RIDGE_DIAGNOSTIC_SOURCE_MUST_BE_PRESERVED_536_FAILED')
        output['source_cost']={'path':str(args.cost_result.resolve().relative_to(rt.ROOT)),'sha256':rt.sha(args.cost_result)}
        contract=rt.read(bs.verify_artifact(cost['protocol_artifact']));ru.source_check(contract['source_hashes'])
        output['scientific_source_hashes']=contract['source_hashes'];output['method_lock']=contract['method_lock']
        fresh=rt.read(bs.verify_artifact(contract['source_results']['fresh']));probes=rt.read(bs.verify_artifact(contract['source_results']['probe']))
        snapshot=rt.source_snapshot(args.tag)
        if snapshot!=contract['source_hashes']:raise ValueError('RIDGE_DIAGNOSTIC_SOURCE_SNAPSHOT_CHANGED')
        output['scientific_source_snapshot']={'path':str((rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json').relative_to(rt.ROOT)),
            'sha256':rt.sha(rt.CACHE/'source_snapshots'/args.tag/'MANIFEST.json')}
        bm.en.configure();forecaster=bm.en.ex.previous.FrozenBolt();bm.en.replace_normalizer(forecaster)
        output['backbone_before_sha256']=rt.state_hash(forecaster.model)
        if output['backbone_before_sha256']!=cost['backbone_before_sha256']:raise ValueError('RIDGE_DIAGNOSTIC_BACKBONE_NOT_EXACT')
        output['matrix_configuration']={'torch_dtype':'float32','NumPy_head_dtype':'float32',
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'torch_float32_matmul_precision':torch.get_float32_matmul_precision(),'torch_threads':torch.get_num_threads(),
            'threadpools':threadpool_info(),'NumPy_version':np.__version__,'Torch_version':torch.__version__}
        for family,identity in CASES:
            model=None
            try:
                row=next(v for v in cost['records'][404:] if v['route']==family and v['identity']==identity and v['batch']==8)
                if row['parity_pass']:raise ValueError('RIDGE_PREDECLARED_CASE_NOT_ACTUALLY_FAILED')
                actual_path=bs.verify_artifact(row['actual_output']);reference_path=bs.verify_artifact(row['matched_science_reference']['artifact'])
                with np.load(actual_path,allow_pickle=False) as z:
                    raw,observed,history,origins,mean,std,old7=[z[k].copy() for k in
                        ('raw_history','history_observed','source_history_normalized','origins','mean','std','all_7_physical_outputs')]
                with np.load(reference_path,allow_pickle=False) as z:
                    expected,ref_history,ref_origins,ref_mean,ref_std=[z[k].copy() for k in ('matched_science_normalized','source_history_normalized','origins','mean','std')]
                if not all(np.array_equal(a,b) for a,b in ((history,ref_history),(origins,ref_origins),(mean,ref_mean),(std,ref_std))):raise ValueError('RIDGE_STORED_PAST_REFERENCE_NOT_EXACT')
                pack={'dataset':row['dataset'],'mean':mean,'std':std};route={'route':family,'split':'eval'}
                cp_before=parameters_before=state_before=None;before_cache={}
                if family.startswith('E_'):
                    label=identity[:-(len(family)+1)];case=next(c for c in fresh['cases'] if c['dataset']==row['dataset'] and c['label']==label)
                    readout_group=next(g for g in probes['readouts'] if g['dataset']==row['dataset'] and g['source_label']==label)
                    readout=next(r for r in readout_group['readouts'] if r['method']==family)
                    route.update(case=case,readout=readout);model=bs.load_selected_model(case)
                    cp_before=case['training']['selected_checkpoint'];state_before=rt.state_hash(model)
                    parameters_before=rt.state_hash(dict(model.named_parameters()));before_cache=cache_arrays(model)
                else:
                    group=next(g for g in probes['controls'] if g['dataset']==row['dataset'])
                    route['control']=next(r for r in group['methods'] if r['method']==family)
                coefficient=route['readout']['model'] if family.startswith('E_') else route['control']['model']
                bs.verify_artifact(coefficient)
                expected_assets={(a['path'].replace('\\','/'),a['sha256']) for a in row['model_artifacts']}
                if (coefficient['path'].replace('\\','/'),coefficient['sha256']) not in expected_assets:raise ValueError('RIDGE_COEFFICIENT_NOT_SAME_AS_FAILED_ROUTE')
                if cp_before and (cp_before['path'].replace('\\','/'),cp_before['sha256']) not in expected_assets:raise ValueError('RIDGE_CP_NOT_SAME_AS_FAILED_ROUTE')
                normalized,physical,stages=cpu_head_deployment(pack,raw,observed,origins,route,forecaster,model,limit)
                torch.cuda.synchronize()
                if not np.array_equal(stages['normalized_input'],history):raise ValueError('RIDGE_DYNAMIC_RAW_PREPROCESS_NOT_EXACT')
                features=stages['features_CPU'];w=stages['weight'];bias=stages['bias'];base=stages['base_normalized']
                with torch.no_grad():
                    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
                    GPU_same_features_delta=(tensor(features)@tensor(w)+tensor(bias)).reshape(normalized.shape)
                    GPU_same_features_norm=(tensor(base)+GPU_same_features_delta).cpu().numpy()
                    native_features=bench._blocks(tensor(history)).flatten(1).cpu().numpy() if not family.startswith('E_') else features.copy()
                    GPU_native_features_delta=(tensor(native_features)@tensor(w)+tensor(bias)).reshape(normalized.shape)
                    GPU_native_features_norm=(tensor(base)+GPU_native_features_delta).cpu().numpy()
                    native_GPU_inverse=(tensor(GPU_native_features_norm)*tensor(std)+tensor(mean)).cpu().numpy()
                decomposition,derived=stage_decomposition(old7[-1],normalized,GPU_native_features_norm,mean,std)
                state_after=None if model is None else rt.state_hash(model)
                parameters_unchanged=model is None or parameters_before==rt.state_hash(dict(model.named_parameters()))
                if not parameters_unchanged:raise ValueError('RIDGE_PROTOTYPE_CHANGED_PARAMETERS')
                after_cache={} if model is None else cache_arrays(model)
                cache_changes={name:float(np.abs(after_cache[name].astype(np.float64)-value.astype(np.float64)).max())
                    for name,value in before_cache.items() if not np.array_equal(value,after_cache[name])}
                arrays=dict(stages);arrays.update(derived);arrays.update(actual_CPU_head_normalized=normalized,
                    actual_CPU_head_physical=physical,matched_science_normalized_reference_only=expected,
                    old_all_7_CUDA_physical=old7,raw_history=raw,history_observed=observed,origins=origins,mean=mean,std=std,
                    GPU_affine_same_CPU_features_delta=GPU_same_features_delta.cpu().numpy(),GPU_affine_same_CPU_features_normalized=GPU_same_features_norm,
                    GPU_native_features=native_features,GPU_affine_native_features_delta=GPU_native_features_delta.cpu().numpy(),
                    GPU_affine_native_features_normalized=GPU_native_features_norm,GPU_native_physical_inverse=native_GPU_inverse)
                for name,value in before_cache.items():arrays['cache_before_'+name]=value;arrays['cache_after_'+name]=after_cache[name]
                result={'family':family,'identity':identity,'batch':8,'actual_prototype_calls':1,
                    'implementation_id':output['implementation_id'],'source_actual':row['actual_output'],
                    'source_reference':row['matched_science_reference']['artifact'],'source_checkpoint':cp_before,'source_ridge':coefficient,
                    'input_preprocessing_exact':True,'origins_mean_std_reference_exact':True,
                    'CPU_head_vs_matched_science':bs.physical_parity(physical,expected,mean,std),
                    'CPU_head_normalized_vs_matched_science_max_abs':float(np.abs(normalized-expected).max()),
                    'NumPy_vsCUDA_affine_same_features_delta_max_abs':float(np.abs(stages['delta_CPU']-GPU_same_features_delta.cpu().numpy()).max()),
                    'NumPy_vsCUDA_raw_block_features_max_abs':None if family.startswith('E_') else float(np.abs(features-native_features).max()),
                    'same_normalized_GPU_vsCPU_inverse_max_abs':float(np.abs(native_GPU_inverse-(GPU_native_features_norm*std+mean)).max()),
                    'current_GPU_head_vs_all_old_7_CUDA_physical_max_abs':float(np.abs(old7-native_GPU_inverse).max()),
                    'stage_decomposition':decomposition,'selected_full_state_before_sha256':state_before,
                    'selected_full_state_after_sha256':state_after,'registered_parameters_unchanged':parameters_unchanged,
                    'memory_cache_buffer_changes':cache_changes,'matrix_shape':{'features':list(features.shape),'weight':list(w.shape),'bias':list(bias.shape)},
                    'arrays':rt.save_npz(folder/(identity+'_B8_STAGE_ARRAYS.npz'),**arrays)}
                for artifact in row['model_artifacts']+[row['actual_output'],row['matched_science_reference']['artifact']]:bs.verify_artifact(artifact)
                output['conditions'].append(result);rt.write(folder/(identity+'_B8.json'),result)
                check()
            finally:
                del model;gc.collect()
        output['backbone_after_sha256']=rt.state_hash(forecaster.model)
        if output['backbone_after_sha256']!=output['backbone_before_sha256']:raise ValueError('RIDGE_DIAGNOSTIC_BACKBONE_CHANGED')
        ru.source_check(contract['source_hashes']);bs.verify_artifact(output['source_cost'])
        for helper in contract['helper_scripts']:bs.verify_artifact(helper)
        check()
        output.update(status='COMPLETE_FOUR_FAMILY_PROTOTYPE_DIAGNOSTIC',scientific16_unchanged=True,
            original_CUDA_536_and_source_arrays_unchanged=True,
            all_four_original_physical_gate_pass=all(v['CPU_head_vs_matched_science']['parity_pass'] for v in output['conditions']))
    except Exception as exc:
        terminal='BUDGET_HELD_PRESERVED' if isinstance(exc,rt.BudgetHeld) else 'FAILED_PRESERVED';error=str(exc)
        output.update(status=terminal,error=error,traceback=traceback.format_exc())
    output['finished_unix']=time.time();output['resource_receipt']=limit.close(terminal,error)
    rt.write(folder/'RESULT.json',output)
    print(rt.json_line({'ridge_prototype_result':str(folder/'RESULT.json'),'status':output['status'],'conditions':len(output['conditions'])}),flush=True)
    if terminal!='COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
