"""Audited aliases of original baseline costs; no new model/GPU execution."""
import copy
import json
from pathlib import Path

import numpy as np

from . import common
from .publish_tables import cost_requests


BASELINES={'G','RAW','S','ALPHA_fitted'}


def _arrays(path,expected):
    path=Path(path)
    if common.sha(path)!=expected:raise ValueError('Reference forecast bytes changed')
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k].copy() for k in ('pred_normalized','target_normalized','observed','origins','channel_ids')}


def _bound(reference,entry,legacy):
    old=_arrays(common.REPO/reference['old_prediction_reference'],reference['old_prediction_sha256'])
    new=_arrays(entry['prediction_path'],entry['prediction_sha256'])
    for key in ('target_normalized','observed','origins','channel_ids'):
        if not np.array_equal(old[key],new[key]):raise ValueError('Legacy/new cost reference cohort differs: '+key)
    parity=legacy['parity']
    if legacy['status']!='COMPLETE' or parity['status']!='PASS' or parity['saved_reference']!='PASS':
        raise ValueError('Original full-V parity incomplete')
    if legacy['prediction_reference_sha256']!=reference['old_prediction_sha256']:
        raise ValueError('Original cost parity is bound to another prediction reference')
    for name,old_key in (('origins','origins'),('observed','observed'),('target','target_normalized')):
        if parity[name]!=common.identity(old[old_key]):
            raise ValueError('Original full-V cost parity exact cohort/targets differ')
    difference=np.abs(old['pred_normalized'].astype(np.float64)-new['pred_normalized'].astype(np.float64))
    previous=float(parity['max_abs'])
    upper=difference+previous
    allowance=1e-5+1e-4*np.abs(new['pred_normalized'].astype(np.float64))
    if not np.isfinite(previous) or previous<0 or np.any(upper>allowance):
        raise ValueError('Original cost-to-new-reference conservative parity bound exceeds fixed tolerance')
    return dict(status='PASS',n_origins=len(new['origins']),atol=1e-5,rtol=1e-4,
        evidence='Conservative per-cell triangle bound: original actual full-V versus hash-bound old reference max_abs plus exact old/new reference difference',
        actual_old_cost_vs_old_reference_max_abs=previous,old_vs_new_reference_max_abs=float(difference.max()),
        conservative_actual_old_cost_vs_new_reference_max_abs=float(upper.max()),
        max_fraction_of_fixed_tolerance=float(np.max(upper/allowance)),
        target_mask_origins_channels_exact=True,original_actual_cost_prediction_arrays_reopened=False,
        original_parity_retained=True,new_model_inference=False,new_selection=False)


def _normalize(candidate,reference,entry,request,path,legacy):
    old=legacy['row'];parity=_bound(reference,entry,legacy)
    if (old['device']!='cuda' or old['batch_size']!=request['batch_size'] or old['warmups']!=10
            or old['timed_trials']!=20 or old['row_microbatch']!=128
            or old['cached_forecast_reads_in_timed_path']!=0 or not old['fresh_process_cli']):
        raise ValueError('Original execution/device/sampling contract differs')
    if entry['method']=='ALPHA_fitted' and legacy['route_spec'].get('alpha')!=request['alpha']:
        raise ValueError('Original alpha differs from current sealed coefficient')
    for relative,digest in legacy['producer_sources'].items():
        if common.sha(common.REPO/relative)!=digest:
            raise ValueError('Original measured producer source has changed: '+relative)
    trials=[]
    active_k=int(old['K'])
    for row in legacy['trials']:
        trial=dict(row)
        trial.update(ms_batch=row['whole_seconds']*1000,ms_origin=row['whole_seconds']*1000/request['batch_size'],
            F_logical_rows=row['F_logical_streams'],F_physical_rows=row['F_physical_streams'],
            F_instances_created=old['foundation_instances_created'],selected_prefix=active_k,
            cached_forecast_or_features_reads=0,
            output_identity=dict(sha256=row['prediction_sha256'],shape=[request['batch_size'],old['H'],old['C']],dtype='float64'),
            allocated_MiB=None,reserved_MiB=None,
            measurement_origin='REUSED_IDENTICAL_CONTRACT',
            original_cost_id=old['cost_id'],original_prediction_sha256=row['prediction_sha256'])
        trials.append(trial)
    memory=dict(legacy['sampling'])
    result=dict(status='COMPLETE',dataset=entry['dataset'],method=entry['method'],seed=request['seed'],
        batch_size=request['batch_size'],device='cuda',request=request,request_sha256=common.sha(path),
        measurement_origin='REUSED_IDENTICAL_CONTRACT',reused_original_measurement=True,
        current_campaign_GPU_seconds=0,new_GPU_operations=0,new_model_evaluations=0,
        actual_fresh_process_original=bool(old['fresh_process_cli']),new_fresh_process_executed=False,
        original_measurement_pid=old['process_pid'],original_measurement_create_time=old['process_create_time'],
        pid=old['process_pid'],process_create_time=old['process_create_time'],
        source_original_receipt=candidate['receipt_reference'],source_original_receipt_sha256=candidate['receipt_sha256'],
        source_original_raw_trials=candidate['trials_reference'],source_original_raw_trials_sha256=candidate['trials_sha256'],
        source_original_request=copy.deepcopy(legacy['request_binding']),
        source_original_request_contract_sha256=legacy['request_contract_sha256'],
        source_original_route_spec=copy.deepcopy(legacy['route_spec']),
        source_original_producer_sources=copy.deepcopy(legacy['producer_sources']),
        source_original_environment=copy.deepcopy(legacy.get('producer_environment',{})),
        source_original_protocol=copy.deepcopy(old['protocol']),protocol=copy.deepcopy(old['protocol']),
        converter_source_sha256=common.sha(Path(__file__)),
        source_reference_reproduction_sha256=common.sha(common.REPORT/'REFERENCE_REPRODUCTION.json'),
        source_original_full_V_parity=copy.deepcopy(legacy['parity']),full_V_parity=parity,
        C_decisions_binding=dict(reference='reports/'+common.NAME+'/C_DECISIONS.json',
            sha256=common.sha(common.REPORT/'C_DECISIONS.json'),status='SEALED_BEFORE_V'),
        device_name='NVIDIA GeForce RTX 4070',
        device_name_evidence='Original campaign HEAD_COMPONENT_COST_INPUTS environments and current SMOKE_jena GPU string; not a per-receipt UUID measurement',
        dtype='FP32_model_FP64_physical_inverse',torch_version=legacy['producer_environment']['torch'],
        model_cold_load_seconds=old['model_load_seconds'],data_preparation_seconds=None,
        window_slicing_seconds=old['input_extraction_seconds'],window_slicing_inside_timed_path=False,
        whole_worker_seconds=None,cold_CPU_RSS_MiB=old['model_loaded_CPU_RSS_MiB'],
        cold_memory=dict(allocated_MiB=old['baseline_allocated_MiB'],reserved_MiB=old['baseline_reserved_MiB'],
                         peak_allocated_MiB=None,peak_reserved_MiB=None),
        original_load_scope=old['baseline_scope'],original_window_scope='Whole-V windows plus fixed timing bundles extracted before model load',
        peak_reset_scope=old['peak_reset_scope'],trials=trials,memory=memory,
        F_instances_created=old['foundation_instances_created'],F_tensor_sha256=old['loaded_F_state_sha256'],
        resident_parameters=old['resident_parameters'],resident_buffers=old['resident_buffer_elements'],
        resident_weights_GPU_MiB=old['resident_weights_gpu_MiB'],resident_weights_CPU_MiB=old['resident_weights_cpu_MiB'],
        median_batch_ms=old['median_batch_ms'],IQR_batch_ms=old['IQR_batch_ms'],
        median_origin_ms=old['median_ms_per_origin'],IQR_origin_ms=old['IQR_ms_per_origin'],
        median_ms_per_batch=old['median_batch_ms'],median_ms_per_origin=old['median_ms_per_origin'],
        missing_field_reasons=dict(data_preparation_seconds='Original receipt does not separately record source-decode/parent-data preparation time',
            whole_worker_seconds='Original receipt has phase times but no measured whole-worker elapsed; components were not summed',
            allocated_MiB='Original trials record timed peak allocation only, not end-of-trial allocation',
            reserved_MiB='Original trials record timed peak reservation only, not end-of-trial reservation'),
        reuse_scope='Same baseline tensors, alpha, physical preprocessing/inverse, fixed cohort/device/precision/rows and timed protocol. Original host-wrapper/load/RSS scope retained, not claimed measured by the new route.')
    return result


def prepare(entries):
    """Create only validated baseline aliases; unresolved cases remain for fresh runs."""
    proof=common.read(common.REPORT/'REFERENCE_REPRODUCTION.json')
    if proof['status']!='PASS':raise ValueError('Fixed reference reproduction must pass before cost reuse')
    reuse=common.read(common.REPORT/'REUSE_MAP.json')
    old_hardware=common.read(common.REPO/'reports/raw_context_multiscale_20261018_v1/HEAD_COMPONENT_COST_INPUTS.json')
    hardware_names={r['environment']['device'] for r in old_hardware['receipts']}
    new_hardware=common.read(common.REPORT/'SMOKE_jena.json')['GPU']
    if hardware_names!={new_hardware} or new_hardware!='NVIDIA GeForce RTX 4070':
        raise ValueError('Original/current campaign GPU model identities differ')
    comparisons={r['new_run_id']:r for r in proof['comparisons']}
    candidates={(r['new_run_id'],r['batch_size']):r for r in reuse['baseline_cost_candidates']}
    by_id={e['run_id']:e for e in entries}
    mapped={};unresolved=[]
    for cost_id,path,request in cost_requests(entries):
        if request['method'] not in BASELINES:continue
        run_id=request['accuracy_run_id'];entry=by_id[run_id]
        try:
            candidate=candidates[(run_id,request['batch_size'])]
            reference=comparisons[run_id]
            source=common.REPO/candidate['receipt_reference']
            if common.sha(source)!=candidate['receipt_sha256']:
                raise ValueError('Original cost receipt changed')
            if common.sha(common.REPO/candidate['trials_reference'])!=candidate['trials_sha256']:
                raise ValueError('Original raw trials changed')
            original=common.read(source)
            normalized=_normalize(candidate,reference,entry,request,path,original)
            destination=common.CACHE/'cost_results'/(cost_id+'.json')
            if destination.exists():
                if common.read(destination)!=normalized:raise ValueError('Preserve existing different cost result')
            else:common.atomic(destination,normalized)
            mapped[cost_id]=dict(legacy_receipt=original,normalized_result=normalized)
        except (KeyError,ValueError,FileNotFoundError) as error:
            unresolved.append(dict(cost_id=cost_id,status='FRESH_MEASUREMENT_REQUIRED',reason=str(error)))
    common.atomic(common.REPORT/'COST_REUSE_VALIDATION.json',dict(status='PASS' if not unresolved else 'PARTIAL',
        reused_facets=len(mapped),unresolved=unresolved,current_campaign_GPU_seconds=0,
        aliases={key:dict(original_receipt=value['normalized_result']['source_original_receipt'],
                         original_receipt_sha256=value['normalized_result']['source_original_receipt_sha256'],
                         alias_sha256=common.sha(common.CACHE/'cost_results'/(key+'.json')))
                 for key,value in mapped.items()},
        source_sha256=common.sha(Path(__file__))))
    return mapped
