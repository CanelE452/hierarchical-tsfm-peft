"""Public summaries and fresh-process cost requests from sealed final artifacts."""
from collections import defaultdict
import json
import math
from pathlib import Path

from . import common


def finalize_manifest():
    entries=[]
    for dataset in common.DATASETS:
        entries.extend(common.read(common.CACHE/f'FINAL_MANIFEST_{dataset}.json'))
    common.seal(common.CACHE/'FINAL_MANIFEST.json',entries)
    rows=[]
    omit=('prediction_path','prediction_sha256','arrays','score','metadata')
    for entry in entries:
        row={k:v for k,v in entry.items() if k not in omit}
        row['prediction_reference']='cache://final/'+entry['run_id']+'/V.npz'
        row['prediction_sha256']=entry['prediction_sha256']
        row['seed']='NA_shared_parent' if entry['seed'] is None else entry['seed']
        for k,v in list(row.items()):
            if v is None:
                row[k]='NA'
            elif isinstance(v,(list,dict)):
                row[k]=json.dumps(v,separators=(',',':'))
        rows.append(row)
    grouped=defaultdict(list)
    for row in rows:
        if str(row['seed']) in ('92601','92602'):
            grouped[(row['dataset'],row['method'],row.get('prefix','NA'))].append(row)
    for group in grouped.values():
        if len(group)!=2:
            continue
        row=dict(group[0],run_id=group[0]['dataset']+'_'+group[0]['method']+'_mean_K'+str(group[0].get('prefix','NA')),
                 seed='mean',aggregation='two_seed_mean; ranges are not confidence intervals')
        for key in ('mse','mae','M_fit_mse','M_select_mse','full_Mcal_mse','search_training_seconds','refit_training_seconds'):
            if all(isinstance(r.get(key),(float,int)) for r in group):
                row[key]=math.fsum(float(r[key]) for r in group)/2
        for key in ('alpha','selected_epoch','parent_G_sha256','parent_S_sha256','model_artifact_sha256',
                    'model_state_sha256','prediction_reference','prediction_sha256','artifact_selected_prefix',
                    'prefix_is_selected','search_exposures','refit_exposures'):
            row[key]='NA; see by-seed'
        rows.append(row)
    common.csv_write(common.REPORT/'RUNS.csv',rows)
    prefix=[row for row in rows if row['method'] in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX')]
    if prefix:
        common.csv_write(common.REPORT/'PREFIX_RESULTS.csv',prefix)
    parity=[]
    for dataset in common.DATASETS:
        p=common.REPORT/f'C_FULL_V_CALL_PARITY_{dataset}.json'
        if p.exists():
            parity.extend(dict(row,dataset=dataset) for row in common.read(p)['rows'])
    if parity:
        common.atomic(common.REPORT/'C_FULL_V_PARITY.json',dict(status='PASS',runs=parity))
    return entries


def cost_requests(entries):
    supported={'G','RAW','S','ALPHA_fitted',*common.METHODS,'MEAN','OWN','ALL',
               'SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'}
    requests=[]
    for entry in entries:
        method,dataset,seed=entry['method'],entry['dataset'],entry['seed'] or 92601
        if method not in supported:
            continue
        for batch in (1,8):
            cost_id=entry['run_id']+f'_B{batch}'
            request=dict(cost_id=cost_id,accuracy_run_id=entry['run_id'],dataset=dataset,method=method,
                seed=seed,batch_size=batch,device='cpu' if method in ('MEAN','OWN','ALL') else 'cuda',
                prediction_reference_path=entry['prediction_path'],
                prediction_reference_sha256=entry['prediction_sha256'],
                deadline_unix=common.start()['final_deadline_unix'])
            if method in common.METHODS:
                request['state_path']=str(common.CACHE/'runs'/dataset/f'{method}_s{seed}'/'refit'/'last.pt')
            elif method in ('MEAN','OWN','ALL'):
                request['state_path']=str(common.CACHE/'probes'/dataset/(method+'.npz'))
            elif method in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
                request.update(state_path=str(common.CACHE/'C'/dataset/f'{method}_s{seed}'/'model.pt'),
                    prefix=entry['prefix'],artifact_selected_prefix=entry['artifact_selected_prefix'])
            if 'state_path' in request:
                request['state_sha256']=common.sha(request['state_path'])
                if request['state_sha256']!=entry['model_artifact_sha256']:
                    raise ValueError('Deployment artifact differs from final accuracy model')
            if entry.get('model_state_sha256'):
                request['model_state_sha256']=entry['model_state_sha256']
            if method=='ALPHA_fitted':
                request['alpha']=entry['alpha']
            path=common.CACHE/'cost_requests'/(cost_id+'.json')
            common.seal(path,request)
            requests.append((cost_id,path,request))
    return requests


def cost_tables():
    summaries,trials=[],[]
    for path in sorted((common.CACHE/'cost_results').glob('*.json')):
        result=common.read(path)
        request=result['request']
        cost_id=request['cost_id']
        prefix=request.get('prefix')
        if prefix is None and result['method'] in ('RESIDUAL_F','RESIDUAL_Q','READOUT_ONLY'):
            prefixes={r['selected_prefix'] for r in result['trials']}
            if len(prefixes)!=1:
                raise ValueError('Phase B cost trials used inconsistent latent K')
            prefix=prefixes.pop()
        fixed=dict(cost_id=cost_id,accuracy_run_id=request['accuracy_run_id'],dataset=result['dataset'],
            method=result['method'],seed=result['seed'],prefix=prefix,
            artifact_selected_prefix=request.get('artifact_selected_prefix'),
            prefix_is_selected=request.get('prefix')==request.get('artifact_selected_prefix') if 'prefix' in request else None,
            batch_size=result['batch_size'],device=result['device'],
            measurement_origin=result.get('measurement_origin','NEW_FRESH_PROCESS'),
            reused_original_measurement=result.get('reused_original_measurement',False),
            new_fresh_process_executed=result.get('new_fresh_process_executed',True),
            source_original_receipt=result.get('source_original_receipt'),
            source_original_receipt_sha256=result.get('source_original_receipt_sha256'),
            original_measurement_pid=result.get('original_measurement_pid'),
            original_measurement_create_time=result.get('original_measurement_create_time'))
        for row in result['trials']:
            trials.append(dict(fixed,**{k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()}))
        timed=[r for r in result['trials'] if r['phase']=='timed']
        memory={k:max(r[k] for r in timed if r[k] is not None) if any(r[k] is not None for r in timed) else None
                for k in ('peak_allocated_MiB','peak_reserved_MiB')}
        summary=dict(fixed,**{k:result[k] for k in ('median_batch_ms','IQR_batch_ms','median_origin_ms','IQR_origin_ms',
            'device_name','dtype','model_cold_load_seconds','data_preparation_seconds','window_slicing_seconds',
            'whole_worker_seconds','resident_parameters','resident_weights_GPU_MiB','resident_weights_CPU_MiB')},
            **memory,**{k:v for k,v in result['memory'].items() if k!='samples'},parity_status=result['full_V_parity']['status'],
            fresh_process=result.get('actual_fresh_process_original',True),
            current_campaign_GPU_seconds=result.get('current_campaign_GPU_seconds'),
            pid=result['pid'],process_create_time=result['process_create_time'],
            cached_forecast_or_features_reads=0,F_instances_created=result['F_instances_created'],
            F_logical_rows_per_batch=timed[0]['F_logical_rows'],F_physical_rows_per_batch=timed[0]['F_physical_rows'],
            F_calls_per_batch=timed[0]['F_calls'],
            request_sha256=result['request_sha256'],result_sha256=common.sha(path))
        summaries.append(summary)
    common.csv_write(common.REPORT/'COST.csv',summaries)
    common.csv_write(common.REPORT/'COST_TRIALS.csv',trials)
    return summaries
