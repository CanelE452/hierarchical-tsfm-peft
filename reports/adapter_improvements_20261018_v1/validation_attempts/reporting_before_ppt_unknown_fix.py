"""Public JSON/CSV sufficient-statistic reporting; never loads arrays or models."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import textwrap


UNKNOWN = 'UNKNOWN'
NAME = 'adapter_improvements_20261018_v1'
SUCCESS = {'COMPLETE', 'REUSED_VERIFIED'}
SCIENCE_KINDS = {'training', 'decoder', 'decoder_transfer', 'lift', 'lift_jobs'}
RUN_FIELDS = ('row_id run_id job_id plan_sha256 source_sha config_sha data_sha model_sha checkpoint_sha256 '
              'cohort method parent seed stage evaluation epoch selected_epoch C K L H device core_dtype '
              'prediction_dtype metric_accumulation_dtype units metric_scale point_statistic scaler_sha256 '
              'target_sha256 mask_sha256 origins_sha256 channel_sha256 n_origins evaluation_stride '
              'MSE MAE status record_origin selection_split V_exposure updates exposures K_exposures '
              'logical_F_rows P_params E_params D_params refiner_params target_trained_params frozen_F_params '
              'total_resident_params preparation_seconds search_seconds refit_seconds cache_seconds '
              'evaluation_seconds independent_FP64 cost_parity prediction_binding checkpoint_binding '
              'source_file source_sha256 source_collection source_row_index recipe_signature backend_signature '
              'source_signature signature_proof input_info_scope').split()
COST_FIELDS = ('cost_id run_id job_id cohort method seed device batch_size core_dtype prediction_dtype '
               'metric_accumulation_dtype median_ms_per_origin median_batch_ms IQR_ms_per_origin '
               'p95_ms_per_origin cold_load_seconds peak_allocated_bytes peak_reserved_bytes '
               'sampled_process_RSS_peak_MiB target_trained_params total_resident_params '
               'cost_valid status record_origin source_file source_sha256 source_row_index').split()
STAT_FIELDS = ['row_id','horizon','channel_id','SSE','SAE','count']
COMPARISON_FIELDS = ('comparison_id question candidate_row_id control_row_id candidate control cohort seed '
                     'stage evaluation epoch selected_epoch control_selected_epoch match_status matching_label reason '
                     'intervention_allowlist delta_MSE delta_MAE delta_MSE_pct delta_MAE_pct '
                     'candidate_MSE control_MSE candidate_MAE control_MAE n_origins evaluation_stride '
                     'horizon_overlap candidate_source_file control_source_file').split()
PAIRED_FIELDS = ('comparison_id candidate_row_id control_row_id candidate control cohort seed origin_index '
                 'candidate_MSE control_MSE delta_MSE candidate_MAE control_MAE delta_MAE status '
                 'candidate_status control_status candidate_coverage_channels control_coverage_channels '
                 'C n_origins evaluation_stride horizon_overlap candidate_source_file candidate_source_sha256 '
                 'candidate_source_collection candidate_source_row_index control_source_file control_source_sha256 '
                 'control_source_collection control_source_row_index candidate_prediction_reference '
                 'candidate_prediction_sha256 control_prediction_reference control_prediction_sha256').split()
TABLES = {'RUNS.csv':RUN_FIELDS, 'COMPARISONS.csv':COMPARISON_FIELDS,
          'COST.csv':COST_FIELDS, 'CHANNEL_HORIZON_STATS.csv':STAT_FIELDS,
          'SAME_UPDATE.csv':RUN_FIELDS, 'MULTIK_AUDIT.csv':['parent','stage','K','completion'],
          'SELECTION_TRIALS.csv':['job_id','cohort','seed','arm','lambda_value','gamma','MSE','MAE'],
          'PAIRED_ORIGIN_EFFECTS.csv':PAIRED_FIELDS,
          'JOB_STATUS.csv':['job_id','kind','cohort','seed','status','reason','source_file','source_sha256']}
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000']
G_COST_SOURCES = (
    ('jena',1,'postT_V_only_cpu_lag_20261016_v1/costs/reference_1792126234985988500_160a7bd2b850442e9e7bb67fde5439b6/DIRECT_G_B1.json',
     '3c2e95e1dc43a299b26750a1b2fa11c6d74116b691ad1fb0e72c8b3747ca60ba','96fd4b9e0d443e4b89ee8cd582d043b39e2de50b5b4ba742a856dd553818cc25'),
    ('jena',8,'postT_V_only_cpu_lag_20261016_v1/costs/reference_1792126234985988500_160a7bd2b850442e9e7bb67fde5439b6/DIRECT_G_B8.json',
     '33578c3d2f63bfef10add0e15bbe95f6f786010d67f3f11dc2803ed2349c4c7e','5762964235c7c2c58650a78cb83ec1f95d706e1f5cf20731a10f31e56920f9d1'),
    ('pems32',1,'priority_pems_subset_20261016_v1/diagnostics/pems_c32_direct_g_cost_b1_root_v1/COST_RECEIPT.json',
     'd2b18c275aec1f45df06205f16b558c7a6d7e8ce36b98ffadcbbfd82014aafd4','9d7d65486f9e8b3d9df1bb2cc45d32ef12295db765353f7df84d9afaccd1822a'),
    ('pems32',8,'priority_pems_subset_20261016_v1/diagnostics/pems_c32_direct_g_cost_b8_root_v1/COST_RECEIPT.json',
     '2a865b1e644bca8282104bdcef7a8f94b25c3926af4301b72ea36ec44d7531d7','68cc0266a4681c45eb3a69e17361f2367a98f705001f23004964b64ad0c644ef'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else default


def _number(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (ValueError,TypeError):
        return None


def _known(value):
    return value is not None and str(value) not in ('','UNKNOWN','None','nan','NA','NOT_APPLICABLE')


def _csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    extras=sorted(set().union(*(row.keys() for row in rows))-set(fields)) if rows else []
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(fields)+extras)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
                                  if isinstance(value,(dict,list,tuple)) else UNKNOWN if value is None else value)
                             for key,value in {**{k:UNKNOWN for k in fields},**row}.items()})


def _json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _write_multik_csv(path,rows,expected_sha=None):
    fields=sorted(set().union(*(row.keys() for row in rows))) if rows else ['parent','stage','K','completion']
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    payload=stream.getvalue().encode('utf-8')
    if expected_sha and hashlib.sha256(payload).hexdigest()!=expected_sha:
        raise ValueError('MULTIK producer CSV differs from the original receipt binding')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)


def _value(row,*keys,default=UNKNOWN):
    for key in keys:
        if _known(row.get(key)):
            return row[key]
    return default


def _binding_hash(value):
    return value.get('sha256',UNKNOWN) if isinstance(value,dict) else UNKNOWN


def _object(value):
    if isinstance(value,(dict,list)):return value
    if isinstance(value,str) and value.startswith(('{','[')):
        try:return json.loads(value)
        except json.JSONDecodeError:return None
    return None


def _same_binding(left,right):
    left,right=_object(left),_object(right)
    return isinstance(left,dict) and isinstance(right,dict) and all(_known(left.get(key)) and left.get(key)==right.get(key)
                                                                  for key in ('reference','sha256'))


def _method(value):
    return 'Raw' if str(value).upper()=='RAW' else value


def _report_reference(value):
    value=str(value).replace('\\','/')
    marker='reports/'+NAME+'/'
    return 'report://'+value.split(marker,1)[1] if marker in value else value if '://' in value else 'report://'+value


def authoritative_cost_join(cost,run):
    """Check actual saved V, serialized model/head and the whole-path parity gate."""
    def fail(reason):return dict(status='FAIL',reason=reason)
    if cost.get('cost_valid') not in (True,'True','true') or cost.get('status')!='COMPLETE':
        return fail('Cost execution/parity gate is not valid COMPLETE')
    if run.get('status') not in ('COMPLETE','REUSED_VERIFIED','CONTEXTUAL_HISTORICAL'):
        return fail('Accuracy execution is incomplete')
    if any(run.get(key)=='FAIL' for key in ('independent_FP64','signature_proof')):
        return fail('Accuracy verification failed')
    if cost.get('accuracy_run_id')!=run.get('run_id') or _method(cost.get('method'))!=_method(run.get('method')):
        return fail('Accuracy ID/method differs')
    if _known(cost.get('accuracy_job_id')) and cost.get('accuracy_job_id')!=run.get('job_id'):
        return fail('Accuracy producer job differs')
    if not _same_binding(cost.get('accuracy_prediction_binding'),run.get('prediction_binding')):
        return fail('Exact saved prediction reference/SHA differs or is missing')
    n=_number(run.get('n_origins'))
    for key in ('saved_V_parity','serialized_reload_saved_V_parity','serialized_reload_parity'):
        parity=_object(cost.get(key))
        if not isinstance(parity,dict) or parity.get('status')!='PASS' or n is None or _number(parity.get('n_origins'))!=n:
            return fail('Full saved V parity missing: '+key)
    raw=_method(run.get('method'))=='Raw'
    if raw:
        if run.get('signature_proof')!='PASS' or cost.get('accuracy_source_kind')!='REUSED_HISTORICAL_SCORED_RAW':
            return fail('Raw requires an independently verified saved manifest/archive proof')
    else:
        source=dict(reference=_report_reference(run.get('source_file')),sha256=run.get('source_sha256'))
        if not _same_binding(cost.get('accuracy_result_binding'),source):
            return fail('Exact scored public producer result differs')
        parent=_object(cost.get('accuracy_bundle_binding')) or _object(cost.get('accuracy_base_checkpoint_binding'))
        if _object(run.get('checkpoint_binding')):
            if not _same_binding(cost.get('accuracy_bundle_binding'),run['checkpoint_binding']):
                return fail('Exact selected bundle reference/SHA differs')
        elif _known(run.get('PE_checkpoint_sha256')):
            if _binding_hash(_object(parent))!=run['PE_checkpoint_sha256']:
                return fail('Exact decoder parent PE bundle SHA differs')
            artifacts=_object(cost.get('_source_private_artifacts')) or []
            for key in ('state_binding','coefficient_binding'):
                expected=run.get(key)
                if not _object(expected) or not any(_same_binding(expected,item) for item in artifacts):
                    return fail('Exact serialized decoder head is missing: '+key)
        elif not (_object(run.get('base_checkpoint')) or _object(run.get('parent_bundle_binding'))):
            return fail('Expected serialized accuracy model binding is missing')
        for run_key,cost_key in (('base_checkpoint','accuracy_base_checkpoint_binding'),
                                 ('parent_bundle_binding','accuracy_base_checkpoint_binding'),
                                 ('refiner_bundle','accuracy_refiner_binding')):
            if _object(run.get(run_key)) and not _same_binding(run[run_key],cost.get(cost_key)):
                return fail('Exact auxiliary model binding differs: '+run_key)
    return dict(status='PASS',reason='Exact saved V/model bindings and all full-V reload parity gates passed')


def _authoritative_cost_matches(cost,runs):
    return [row for row in runs if authoritative_cost_join(cost,row)['status']=='PASS']


def _historical_context_cost(cost):
    return (cost.get('record_origin')=='CONTEXTUAL' and cost.get('producer_status') in ('COMPLETE_PROTOCOL','COMPLETE')
            and (cost.get('parity_state')=='PASS' or
                 (cost.get('comparison_scope')=='HISTORICAL_CPU2_CONTEXT' and
                  cost.get('historical_binding_validation')=='PASS_RECEIPT_TRIALS_ROUTE')))


def aggregate_statistics(rows, channels):
    """Sum SSE/SAE/count first, then average channel ratios over fixed C."""
    values={str(channel):[0.,0.,0] for channel in channels}
    for row in rows:
        if str(row['channel_id']) not in values:
            raise ValueError('Statistics contain an unregistered channel')
        current=values[str(row['channel_id'])]
        current[0]+=float(row['SSE']);current[1]+=float(row['SAE']);current[2]+=int(float(row['count']))
    coverage=sum(v[2]>0 for v in values.values())
    complete=coverage==len(values) and bool(values)
    return dict(MSE=math.fsum(v[0]/v[2] for v in values.values())/len(values) if complete else None,
                MAE=math.fsum(v[1]/v[2] for v in values.values())/len(values) if complete else None,
                coverage_channels=coverage,registered_channels=len(values),count=sum(v[2] for v in values.values()),
                status='DEFINED' if complete else 'UNDEFINED')


def match_rows(candidate, control, question):
    allow={'P':['frontend decomposition versus raw-half control'],
           'E':['E initialization and corresponding pinv D initialization'],
           'D':['head structure and its registered A_stop-selected lambda/gamma recipe']}.get(question,[])
    required=('cohort','seed','C','K','L','H','stage','evaluation','device','core_dtype','point_statistic',
              'scaler_sha256','target_sha256','mask_sha256','origins_sha256','channel_sha256',
              'model_sha','recipe_signature','backend_signature','source_signature')
    missing=[key for key in required if not _known(candidate.get(key)) or not _known(control.get(key))]
    different=[key for key in required if key not in missing and str(candidate[key])!=str(control[key])]
    if question!='D' and _known(candidate.get('data_sha')) and _known(control.get('data_sha')) and candidate['data_sha']!=control['data_sha']:
        different.append('data_sha')
    if candidate.get('stage')=='search':
        for key in ('epoch','updates','exposures','sampler_sha256'):
            if not _known(candidate.get(key)) or not _known(control.get(key)):
                missing.append(key)
            elif candidate[key]!=control[key]:
                different.append(key)
    status='INCOMPLETE' if missing else 'CONTEXTUAL' if different else 'MATCHED'
    return dict(match_status=status,matching_label=('MATCHED_RECIPE_SELECTED_REFIT' if candidate.get('stage')!='search'
                else 'MATCHED_COMMON_ENDPOINT') if status=='MATCHED' else status,
                reason='Missing: '+','.join(missing) if missing else 'Different: '+','.join(different) if different
                else 'Exact registered common signature; learned checkpoint hashes may differ',intervention_allowlist=allow)


def _normalize(row, job, result, plan, source, source_sha, index, collection, origin, proof=None):
    row=dict(row)
    fingerprints=row.get('array_fingerprints',{})
    hashes={key:fingerprints.get(name,{}).get('sha256',UNKNOWN) for key,name in
            [('target_sha256','target_normalized'),('mask_sha256','observed'),('origins_sha256','origins'),('channel_sha256','channel_ids')]}
    recipe=dict(result.get('recipe',{}))
    recipe={key:recipe.get(key,default) for key,default in dict(epochs=[.25,1.,3.],lr=.001,microbatch=4,
             effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,optimizer='AdamW',weight_decay=0.,gradient_clip=1.).items()}
    common={key:value for key,value in plan.get('source_hashes',{}).items()
            if '/informs_broad_validation_20261015_v1/' in key and key.endswith(('.py','.json'))}
    normalized=dict(row,**{key:_value(row,key,default=hashvalue) for key,hashvalue in hashes.items()})
    normalized.update(row_id=f"{job.get('job_id',row.get('run_id','legacy'))}:{collection}:{index}",
        run_id=_value(row,'run_id',default=job.get('job_id',UNKNOWN)+':'+str(row.get('method','UNKNOWN'))),job_id=job.get('job_id',UNKNOWN),
        plan_sha256=plan.get('plan_sha256',UNKNOWN),cohort=_value(row,'dataset','cohort',default=job.get('cohort',UNKNOWN)),
        seed=_value(row,'seed',default=job.get('seed',UNKNOWN)),method=_value(row,'method',default=job.get('family',UNKNOWN)),
        stage=_value(row,'stage',default='selected_refit'),evaluation=_value(row,'evaluation','split',default='V_select'),
        status=_value(row,'status',default=result.get('status',UNKNOWN)),record_origin=origin,
        source_file=source,source_sha256=source_sha,source_collection=collection,source_row_index=index,
        checkpoint_sha256=_value(row,'checkpoint_sha256',default=_binding_hash(row.get('checkpoint_binding',result.get('bundle_binding')))),
        checkpoint_binding=row.get('checkpoint_binding',result.get('bundle_binding',UNKNOWN)),
        source_sha=_value(row,'source_sha',default=result.get('source_sha',UNKNOWN)),
        config_sha=_value(row,'config_sha',default=result.get('config_sha',UNKNOWN)),
        data_sha=_value(row,'data_sha',default=result.get('data_sha',UNKNOWN)),
        model_sha=_value(row,'model_sha',default=result.get('model_sha',result.get('frozen_F_weight_hash',UNKNOWN))),
        core_dtype=_value(row,'core_dtype','head_dtype','dtype',default='float32' if origin!='CONTEXTUAL' else UNKNOWN),
        prediction_dtype=_value(row,'prediction_dtype','dtype'),metric_accumulation_dtype=_value(row,'metric_accumulation_dtype',default='float64'),
        target_trained_params=_value(row,'target_trained_params','total_target_trained_params','trainable_params'),
        independent_FP64=_value(row,'independent_FP64','saved_array_independent_FP64'),
        recipe_signature=_digest(recipe) if origin!='CONTEXTUAL' else UNKNOWN,
        backend_signature=_digest(result.get('backend_flags',{})) if 'backend_flags' in result else UNKNOWN,
        source_signature=_digest(common) if common else _value(row,'source_sha',default=result.get('source_sha',UNKNOWN)),
        signature_proof='PRODUCER_FINGERPRINTS' if fingerprints else 'MISSING',
        point_statistic=_value(row,'point_statistic',default='quantile_0.5'),
        metric_scale=_value(row,'metric_scale',default='normalized_A_fit' if row.get('stage')=='search' else 'normalized_A60'),
        units=_value(row,'units',default='normalized_original_channels'),
        selection_split=_value(row,'selection_split',default='A_stop'),
        V_exposure=_value(row,'V_exposure',default='Repeated development V; historical T use retained'),
        device=_value(row,'device',default=job.get('device',UNKNOWN)))
    if origin=='CONTEXTUAL':
        normalized['status']='CONTEXTUAL_HISTORICAL'
        if 'ridge' in normalized['method'].lower():
            normalized.update(core_dtype='float64',prediction_dtype=_value(row,'prediction_dtype','dtype'),
                              precision_note='CPU | core FP64, stored '+str(_value(row,'prediction_dtype','dtype')).replace('float','FP'))
        if row.get('source_family')=='DIRECT_G':
            normalized['method']='G-only'
    elif normalized['status'] in ('DEFINED','UNDEFINED'):
        normalized['metric_status']=normalized['status']
        normalized['status']=result.get('status',UNKNOWN)
    if proof:
        signature=proof.get('canonical_cohort_signature',{})
        for key,name in [('target_sha256','target'),('mask_sha256','mask'),('origins_sha256','origins'),('channel_sha256','channel_ids')]:
            if name in signature:
                normalized[key]=_digest(signature[name])
        normalized['signature_proof']=proof.get('validation_status',UNKNOWN)
        normalized['prediction_binding']=dict(reference=proof['prediction_reference'],sha256=proof['prediction_sha256'])
        normalized['n_origins']=proof.get('statistics',{}).get('n_origins',normalized.get('n_origins',UNKNOWN))
        if proof.get('source_job_id')=='HISTORICAL_RAW_V':normalized['job_id']='HISTORICAL_RAW_V'
        for key,field in [('scaler_sha256','scaler_sha256_ifproven'),('metric_scale','metric_scale_ifproven'),('model_sha','F_snapshot_ifproven')]:
            if _known(proof.get(field)):
                normalized[key]=proof[field]
        if isinstance(proof.get('recipe_binding'),dict):
            normalized['verified_recipe_binding']=proof['recipe_binding']
    if job.get('kind') in ('decoder','decoder_transfer'):
        normalized['stage']='selected_refit' if normalized['evaluation']=='V_select' else normalized['stage']
        normalized['recipe_signature']=_digest(dict(kind='fixed_PE_40grid',selection='A_fit->A_stop->fresh_A60',degree=1))
        normalized['backend_signature']=_digest(dict(head='numpy_FP64',F='frozen_FP32_cached'))
        normalized['source_signature']=_digest(result.get('scientific_source_hashes',plan.get('source_hashes',{})))
    for key in ('horizon_sse','horizon_sae','horizon_counts','origin_sse','origin_sae','origin_counts',
                'origin_statistics','origin_paired_vs_shared','bands'):
        normalized.pop(key,None)
    if result.get('timers'):normalized['result_timers']=result['timers']
    return normalized


def _statistics(raw, run):
    sse,sae,counts=raw.get('horizon_sse'),raw.get('horizon_sae'),raw.get('horizon_counts')
    if sse is None or sae is None or counts is None:
        return []
    rows=[]
    for h,(ss,aa,cc) in enumerate(zip(sse,sae,counts),1):
        if len(ss)!=len(aa) or len(ss)!=len(cc):
            raise ValueError('Public channel/horizon sufficient-statistic axes differ')
        for channel,(s,a,c) in enumerate(zip(ss,aa,cc)):
            rows.append({**{key:run.get(key,UNKNOWN) for key in ('row_id','run_id','job_id','cohort','method','seed','stage','evaluation','record_origin','source_file','source_sha256')},
                         'horizon':h,'channel_id':channel,'SSE':s,'SAE':a,'count':c})
    return rows


def _comparisons(runs):
    result=[]
    mapping={'P_SPLIT_PCA':('P_LOCAL_CONTROL_PCA','P'),'B2_DCA':('B2_PCA','E'),
             'D_HORIZON':('D_SHARED','D'),'D_LINEAR':('D_SHARED','D'),'D_GAIN':('D_SHARED','D')}
    candidates=[row for row in runs if row.get('record_origin') in ('NEW','REUSED_VERIFIED') and _number(row.get('MSE')) is not None]
    for row in candidates:
        if row.get('method') not in mapping:
            continue
        comparator,question=mapping[row['method']]
        controls=[other for other in candidates if other.get('method')==comparator and
                  all(str(other.get(key))==str(row.get(key)) for key in ('cohort','seed','stage','evaluation')) and
                  (row.get('stage')!='search' or str(other.get('epoch'))==str(row.get('epoch')))]
        if not controls:
            result.append(dict(comparison_id=row['row_id']+'|MISSING',question=question,candidate_row_id=row['row_id'],
                               candidate=row['method'],control=comparator,cohort=row['cohort'],seed=row['seed'],stage=row['stage'],
                               evaluation=row['evaluation'],match_status='INCOMPLETE',reason='Registered control result unavailable'))
        for control in controls:
            match=match_rows(row,control,question)
            values=dict(comparison_id=row['row_id']+'|'+control['row_id'],question=question,
                        candidate_row_id=row['row_id'],control_row_id=control['row_id'],candidate=row['method'],control=control['method'],
                        **{key:row.get(key,UNKNOWN) for key in ('cohort','seed','stage','evaluation','epoch','selected_epoch','n_origins','evaluation_stride')},
                        control_selected_epoch=control.get('selected_epoch',UNKNOWN),horizon_overlap=False,
                        candidate_source_file=row['source_file'],control_source_file=control['source_file'],**match)
            for metric in ('MSE','MAE'):
                left,right=_number(row.get(metric)),_number(control.get(metric))
                values.update({f'candidate_{metric}':left,f'control_{metric}':right,
                               f'delta_{metric}':left-right if left is not None and right is not None else None,
                               f'delta_{metric}_pct':100*(left-right)/right if left is not None and right not in (None,0) else None})
            result.append(values)
    return result


def _cost(row, job, source, source_sha, index, origin, result=None):
    value=dict(row)
    method=_value(row,'method','source_method')
    if str(method).upper()=='RAW':method='Raw'
    value.update(cost_id=f"{job.get('job_id','legacy')}:{index}",job_id=job.get('job_id',row.get('job_id',UNKNOWN)),
        cohort=_value(row,'dataset','cohort',default=job.get('cohort',UNKNOWN)),method=method,
        batch_size=_value(row,'batch_size','B'),median_batch_ms=_value(row,'median_batch_ms','median_ms_per_batch'),
        core_dtype=_value(row,'core_dtype'),prediction_dtype=_value(row,'prediction_dtype'),
        metric_accumulation_dtype=_value(row,'metric_accumulation_dtype',default='float64'),
        record_origin=origin,source_file=source,source_sha256=source_sha,source_row_index=index,
        target_trained_params=_value(row,'target_trained_params','trainable_params'),
        total_resident_params=_value(row,'total_resident_params','total_resident_model_params'),
        cost_valid=row.get('cost_valid',False) if origin!='CONTEXTUAL' else False)
    value['producer_status']=row.get('status',UNKNOWN)
    if result is not None:value['_source_private_artifacts']=result.get('private_artifacts',[])
    memory=_object(row.get('memory')) or {}
    for source_key,target in (('peak_allocated_MiB','peak_allocated_bytes'),('peak_reserved_MiB','peak_reserved_bytes')):
        measured=_number(memory.get(source_key))
        if measured is not None and not _known(value.get(target)):value[target]=measured*1024**2
    rss=_number(row.get('sampled_process_RSS_peak_MiB'))
    if rss is not None and not _known(value.get('process_RSS_peak_bytes')):value['process_RSS_peak_bytes']=rss*1024**2
    if origin=='CONTEXTUAL':
        value['status']='CONTEXTUAL_HISTORICAL'
        value['cost_valid']='HISTORICAL_PROTOCOL_ONLY'
        if 'ridge' in method.lower():
            value.update(core_dtype='float64',prediction_dtype='float32',precision_note='CPU | core FP64, stored FP32')
        times=row.get('individual_timed_seconds')
        if isinstance(times,str) and times.startswith('['):
            import numpy as np
            samples=np.asarray(json.loads(times),float)
            batch=_number(value['batch_size'])
            if len(samples)==20 and batch:
                value['p95_ms_per_origin']=float(np.quantile(samples,.95)*1000/batch)
                value['p95_derivation']='Derived from 20 stored historical timings; no new measurement'
    return value


def historical_g_costs(repo,legacy):
    """Import four pinned public G timings as old-host context, without model I/O."""
    repo=Path(repo);base=repo/'research/informs_broad_validation_20261015_v1'
    controls={row['cohort']:row for row in legacy if row.get('source_family')=='DIRECT_G' and
              row.get('cohort') in ('jena','pems32') and _number(row.get('C')) in (21,32)}
    output=[];bindings={}
    for cohort,batch,relative,receipt_sha,trials_sha in G_COST_SOURCES:
        if cohort not in controls:continue
        control=controls[cohort];path=base/relative
        if not path.is_file() or _sha(path)!=receipt_sha:
            raise ValueError('Historical G receipt source SHA mismatch: '+relative)
        receipt=_read(path)
        if receipt.get('run_id')!=control['run_id'] or receipt.get('artifact_sha256')!=control.get('checkpoint_sha256'):
            raise ValueError('Historical G run/checkpoint identity differs from DIRECT_G control')
        if receipt.get('device')!='cpu' or receipt.get('batch_size')!=batch or receipt.get('status')!='COMPLETE_PROTOCOL':
            raise ValueError('Historical G cost scope changed')
        if (receipt.get('warmups'),receipt.get('timed_trials'),receipt.get('stage_trials'))!=(10,20,20):
            raise ValueError('Historical G trial protocol changed')
        if cohort=='jena':
            scope=receipt.get('accuracy_bindings',{}).get(control['run_id'],{})
            scaler=scope.get('scaler_sha')
            manifest=base/scope['accuracy_manifest_public_relative_path']
            model_source=base/'measure_costs.py'
            expected=receipt.get('source_sha256',{}).get('measure_costs.py')
            if not expected or _sha(model_source)!=expected:
                raise ValueError('Historical G CPU2 source binding changed')
            bindings[model_source.relative_to(repo).as_posix()]=expected
            accuracy_scope='TRANSITIVE_OLD_DIAGNOSTIC_SCORE_CHECKPOINT_SCALER; legacy prediction manifest absent'
            if manifest.is_file():
                if _sha(manifest)!=scope.get('accuracy_manifest_sha256'):
                    raise ValueError('Historical G Jena prediction manifest SHA changed')
                bindings[manifest.relative_to(repo).as_posix()]=_sha(manifest)
                accuracy_scope='BOUND_HISTORICAL_PUBLIC_MANIFEST'
            trials=path.with_name(receipt['trials_file'])
            thread_evidence='CPU2 source-derived from hash-bound original measure_costs.py; thread field absent in receipt'
        else:
            if receipt.get('CPU_threads')!=2:raise ValueError('Historical G CPU2 receipt changed')
            manifest=base/'priority_pems_subset_20261016_v1/predictions'/control['run_id']/'V_select/PREDICTION_MANIFEST.json'
            if not manifest.is_file() or _sha(manifest)!=receipt.get('manifest_sha256'):
                raise ValueError('Historical G PeMS32 prediction manifest SHA changed')
            values=_read(manifest);scaler=values.get('scaler_sha')
            if values.get('run_id')!=control['run_id'] or values.get('data_sha')!=receipt.get('data_sha256'):
                raise ValueError('Historical G PeMS32 prediction identity changed')
            for field,key in [('target_sha256','target_normalized'),('mask_sha256','observed'),('origins_sha256','origins'),('channel_sha256','channel_ids')]:
                if control.get(field)!=values.get('arrays',{}).get(key,{}).get('sha256'):
                    raise ValueError('Historical G PeMS32 cohort binding changed: '+field)
            bindings[manifest.relative_to(repo).as_posix()]=_sha(manifest)
            accuracy_scope='BOUND_HISTORICAL_PUBLIC_MANIFEST; original CPU6 parity plus full-V saved FP64 rescore'
            trials=path.with_name('COST_TRIALS.csv.gz');thread_evidence='CPU_threads=2 in original receipt'
        if scaler!=control.get('scaler_sha256'):
            raise ValueError('Historical G exact scaler binding differs')
        if not trials.is_file() or _sha(trials)!=trials_sha or receipt.get('trials_sha256')!=trials_sha:
            raise ValueError('Historical G trial source SHA mismatch')
        with gzip.open(trials,'rt',encoding='utf-8',newline='') as stream:
            trial_rows=list(csv.DictReader(stream))
        phase_counts=Counter(row['phase'] for row in trial_rows)
        timed=[row for row in trial_rows if row['phase']=='timed']
        if phase_counts!=Counter(warmup=10,timed=20,stage_probe=20) or any(int(row['batch_size'])!=batch for row in trial_rows):
            raise ValueError('Historical G actual trial count/batch changed')
        if any(int(row.get(key,0))!=0 for row in trial_rows for key in ('F_calls','F_logical_streams','F_physical_streams')):
            raise ValueError('Historical G direct predictor unexpectedly invoked F')
        summary=next(row for row in receipt['summaries'] if row['phase']=='timed')
        median=statistics.median(float(row['whole_seconds']) for row in timed)
        if abs(median-float(summary['median_seconds']))>1e-12:
            raise ValueError('Historical G median differs from bound timed trials')
        source=path.relative_to(repo).as_posix();trial_source=trials.relative_to(repo).as_posix()
        bindings[source]=receipt_sha;bindings[trial_source]=trials_sha
        row=_cost(dict(method='G-only',dataset=cohort,run_id=control['run_id'],accuracy_run_id=control['run_id'],
            source_family='DIRECT_G',seed=control.get('seed'),C=control.get('C'),K=0,L=control.get('L'),H=control.get('H'),
            device='cpu',batch_size=batch,status='COMPLETE_PROTOCOL',core_dtype='float32',prediction_dtype='float32',
            physical_output_dtype='float64',inverse_transform_dtype='float64',memory=receipt.get('memory'),
            median_ms_per_origin=median*1000/batch,median_batch_ms=median*1000,
            p90_ms_per_origin=float(summary['p90_seconds'])*1000/batch,p95_ms_per_origin=None,
            accuracy_MSE=control.get('MSE'),accuracy_MAE=control.get('MAE'),
            accuracy_source_file=control.get('source_file'),accuracy_source_sha256=control.get('source_sha256'),
            scaler_sha256=scaler,checkpoint_sha256=receipt['artifact_sha256'],target_trained_params=control.get('target_trained_params',control.get('trainable_params')),
            comparison_scope='HISTORICAL_CPU2_CONTEXT',historical_binding_validation='PASS_RECEIPT_TRIALS_ROUTE',
            accuracy_binding_scope=accuracy_scope,parity_state='HISTORICAL_ONLY_NOT_NEW_FULL_V_PARITY',
            cpu_blas_threads=2,thread_evidence=thread_evidence,host_identity='UNKNOWN',
            hardware_protocol_scope='HISTORICAL_CPU2_CONTEXT | host UNKNOWN',new_measurement=False,
            warmups=10,repetitions=20,stage_trials=20,trials_rows=len(trial_rows),F_calls=0,
            scope='Historical whole physical-input/mask -> fill/normalize -> shared G -> FP64 inverse -> CPU; no F/cached forecast; cold excluded',
            quantile_scope='p90 copied from bound original summary; p95 UNKNOWN; no percentile relabeling',
            trials_source_file=trial_source,trials_source_sha256=trials_sha),
            dict(job_id='HISTORICAL_G_'+cohort+'_B'+str(batch)),source,receipt_sha,0,'CONTEXTUAL')
        output.append(row)
    return output,bindings


def compact_row_ids(tables):
    """Keep statistics at six columns; source metadata stays once in RUNS."""
    runs=tables['RUNS.csv']
    original=[row['row_id'] for row in runs]
    if len(set(original))!=len(original):
        raise ValueError('Duplicate producer row ID in public run inventory')
    mapping={value:f'r{index:05d}' for index,value in enumerate(original)}
    lookup={row['row_id']:row for row in runs}
    compact=[]
    for row in tables['CHANNEL_HORIZON_STATS.csv']:
        run=lookup.get(row['row_id'])
        if run is None:
            raise ValueError('Statistics row has no registered RUNS metadata')
        for field in ('source_file','source_sha256'):
            if _known(row.get(field)):
                run['statistics_'+field]=row[field]
        compact.append({field:mapping[row['row_id']] if field=='row_id' else row[field] for field in STAT_FIELDS})
    for row in runs:
        row['producer_row_id']=row['row_id']
        row['row_id']=mapping[row['row_id']]
    for row in tables['SAME_UPDATE.csv']:
        # In-memory rows normally share the RUNS object; also support copies.
        row['row_id']=mapping.get(row['row_id'],row['row_id'])
        if row['row_id'] not in mapping.values():
            raise ValueError('SAME_UPDATE row has no registered RUNS metadata')
    tables['CHANNEL_HORIZON_STATS.csv']=compact


def origin_statistics(raw,channels):
    """Read public per-origin channel totals from either registered producer form."""
    if 'origin_statistics' in raw:
        values=raw['origin_statistics']
        if any(row.get('origin_index')!=index for index,row in enumerate(values)):
            raise ValueError('Public origin index alignment differs from saved cohort order')
        result=[dict(origin_index=index,sse=row['channel_sse'],sae=row['channel_sae'],count=row['channel_counts'])
                for index,row in enumerate(values)]
    elif any(key in raw for key in ('origin_sse','origin_sae','origin_counts')):
        if not all(key in raw for key in ('origin_sse','origin_sae','origin_counts')):
            raise ValueError('Public origin statistics are incomplete')
        sse,sae,counts=raw['origin_sse'],raw['origin_sae'],raw['origin_counts']
        if len(sse)!=len(sae) or len(sse)!=len(counts):
            raise ValueError('Public origin statistics row axes differ')
        result=[dict(origin_index=index,sse=ss,sae=aa,count=cc) for index,(ss,aa,cc) in enumerate(zip(sse,sae,counts))]
    else:return None
    for row in result:
        if any(len(row[key])!=channels for key in ('sse','sae','count')):
            raise ValueError('Public origin statistics do not retain every registered channel')
        if any(_number(value) is None or float(value)<0 for key in ('sse','sae','count') for value in row[key]):
            raise ValueError('Public origin statistics contain invalid channel totals')
        if any(float(value)!=int(value) for value in row['count']):
            raise ValueError('Public origin statistics require integer observation counts')
    return result


def paired_origin_effects(tables,raw_rows):
    lookup={row['row_id']:row for row in tables['RUNS.csv']};output=[]
    for comparison in tables['COMPARISONS.csv']:
        if (comparison.get('match_status')!='MATCHED' or comparison.get('stage')!='selected_refit'
                or comparison.get('evaluation')!='V_select'):continue
        candidate=lookup[comparison['candidate_row_id']];control=lookup[comparison['control_row_id']]
        channels=int(float(candidate['C']))
        if channels!=int(float(control['C'])):raise ValueError('Paired origin channel registration differs')
        left=origin_statistics(raw_rows.get(candidate['producer_row_id'],{}),channels)
        right=origin_statistics(raw_rows.get(control['producer_row_id'],{}),channels)
        base={key:comparison.get(key,UNKNOWN) for key in ('comparison_id','candidate_row_id','control_row_id','candidate','control','cohort','seed')}
        base.update(C=channels,n_origins=_value(candidate,'n_origins'),evaluation_stride=candidate.get('evaluation_stride',UNKNOWN),
                    horizon_overlap=comparison.get('horizon_overlap',False))
        for label,run in (('candidate',candidate),('control',control)):
            for key in ('source_file','source_sha256','source_collection','source_row_index'):
                base[label+'_'+key]=run.get(key,UNKNOWN)
            prediction=_object(run.get('prediction_binding')) or {}
            base[label+'_prediction_reference']=prediction.get('reference',UNKNOWN)
            base[label+'_prediction_sha256']=prediction.get('sha256',UNKNOWN)
        if left is None or right is None:
            output.append(dict(base,origin_index=None,status='UNAVAILABLE_PUBLIC_ORIGIN_STATISTICS'))
            continue
        if len(left)!=len(right):raise ValueError('Paired public origin count differs')
        for run in (candidate,control):
            registered=_number(run.get('n_origins'))
            if registered is not None and registered!=len(left):raise ValueError('Paired public origin count differs from RUNS')
        base['n_origins']=len(left)
        for index,(a,b) in enumerate(zip(left,right)):
            if a['count']!=b['count']:raise ValueError('Paired public origin observation counts differ')
            scored=[]
            for value in (a,b):
                scored.append(aggregate_statistics([dict(channel_id=channel,SSE=value['sse'][channel],SAE=value['sae'][channel],count=value['count'][channel])
                                                    for channel in range(channels)],range(channels)))
            p,q=scored;defined=p['status']=='DEFINED' and q['status']=='DEFINED'
            output.append(dict(base,origin_index=index,candidate_MSE=p['MSE'],control_MSE=q['MSE'],
                delta_MSE=p['MSE']-q['MSE'] if defined else None,candidate_MAE=p['MAE'],control_MAE=q['MAE'],
                delta_MAE=p['MAE']-q['MAE'] if defined else None,status='DEFINED' if defined else 'UNDEFINED',
                candidate_status=p['status'],control_status=q['status'],candidate_coverage_channels=p['coverage_channels'],
                control_coverage_channels=q['coverage_channels']))
    return output


def collect_public(ctx):
    root=Path(ctx['report']);repo=Path(ctx['repo'])
    plan=_read(root/'EXPERIMENT_PLAN.json',{})
    jobs={row['job_id']:row for row in plan.get('jobs',[])}
    original_jobs=set(jobs)
    supplemental_plans=[]
    for path in sorted((root/'SUPPLEMENTAL_PLANS').glob('*.json')):
        supplemental=_read(path,{})
        supplemental_plans.append(dict(file='SUPPLEMENTAL_PLANS/'+path.name,sha256=_sha(path),plan_sha256=supplemental.get('plan_sha256')))
        for row in supplemental.get('jobs',[]):
            if row['job_id'] in jobs:
                raise ValueError('Duplicate original/supplemental public job ID')
            jobs[row['job_id']]=dict(row,supplemental_plan_sha256=supplemental.get('plan_sha256'))
    validation=_read(root/'VALIDATION_REPORT.json',{})
    proof={ (row.get('source_job_id'),row.get('prediction_reference'),row.get('prediction_sha256')):row
            for row in validation.get('verified_prediction_archives',[]) }
    tables={name:[] for name in TABLES}
    outcomes=[];sources={};raw_source_rows={}
    def reference(path):
        try:
            return path.relative_to(repo).as_posix()
        except ValueError:
            return 'report://'+path.relative_to(root).as_posix()
    for identifier,job in jobs.items():
        path=root/'jobs'/(identifier+'.json')
        receipt_path=root/'receipts'/(identifier+'.json')
        result=_read(path,{})
        receipt=_read(receipt_path,{})
        if isinstance(result.get('result'),dict) and 'rows' not in result:
            result=result['result']
        status=receipt.get('status',result.get('status','PLANNED'))
        source=path if path.exists() else receipt_path if receipt_path.exists() else root/'EXPERIMENT_PLAN.json'
        source_ref=reference(source)
        source_sha=_sha(source) if source.exists() else UNKNOWN
        sources[source_ref]=source_sha
        outcomes.append(dict(job_id=identifier,kind=job.get('kind'),cohort=job.get('cohort'),seed=job.get('seed'),
                             status=status,reason=receipt.get('reason',result.get('reason')),source_file=source_ref,source_sha256=source_sha,
                             plan_scope='ORIGINAL' if identifier in original_jobs else 'SUPPLEMENTAL',
                             plan_sha256=job.get('supplemental_plan_sha256',plan.get('plan_sha256'))))
        if job.get('kind')=='initialization':
            outcomes[-1].update(K=job.get('K',job.get('k')),fit_scope=result.get('fit_scope',{}).get('name',job.get('scope')),
                initialization_preparation_seconds=result.get('diagnostics',{}).get('preparation_wall_seconds'),
                parent_job_wall_seconds=receipt.get('consumed_resource',{}).get('wall_seconds'),
                resource_receipt_source_file=reference(receipt_path),
                resource_receipt_source_sha256=_sha(receipt_path) if receipt_path.is_file() else UNKNOWN)
        origin='REUSED_VERIFIED' if job.get('kind')=='reuse_pca' and status=='REUSED_VERIFIED' else 'NEW'
        rows=result.get('rows',[])
        if job.get('kind') in ('cost','costs') or result.get('family')=='WHOLE_PATH_COST':
            tables['COST.csv'] += [_cost(row,job,source_ref,source_sha,index,'NEW',result) for index,row in enumerate(rows)]
            rows=[]
        elif job.get('kind')=='multik_audit' or result.get('prefix_rule'):
            tables['MULTIK_AUDIT.csv']+=rows
            rows=[]
        for collection,raw_rows in (('rows',rows),('common_endpoint_rows',result.get('common_endpoint_rows',[]))):
            for index,row in enumerate(raw_rows):
                binding=row.get('prediction_binding',{})
                independent=proof.get((identifier,binding.get('reference'),binding.get('sha256')))
                normalized=_normalize(row,job,result,plan,source_ref,source_sha,index,collection,origin,independent)
                raw_source_rows[normalized['row_id']]=row
                tables['RUNS.csv'].append(normalized)
                tables['CHANNEL_HORIZON_STATS.csv']+=_statistics(row,normalized)
                if collection=='common_endpoint_rows':
                    tables['SAME_UPDATE.csv'].append(normalized)
        for trial in result.get('selection_trials',[]):
            tables['SELECTION_TRIALS.csv'].append(dict(trial,job_id=identifier,cohort=job.get('cohort'),seed=job.get('seed'),
                                                       source_file=source_ref,source_sha256=source_sha))
        if not rows and not result.get('common_endpoint_rows'):
            label=(str(job['family'])+'_'+str(job.get('init_kind','PCA'))) if job.get('kind')=='training' else job.get('family',job.get('kind'))
            tables['RUNS.csv'].append(dict(row_id=identifier+':attempt',run_id=identifier,job_id=identifier,
                cohort=job.get('cohort'),method=label,seed=job.get('seed'),
                stage='job_attempt',status=status,record_origin='ATTEMPT',reason=receipt.get('reason',result.get('reason')),
                plan_sha256=plan.get('plan_sha256'),source_file=source_ref,source_sha256=source_sha,
                device=job.get('device'),K=job.get('K')))
    old=Path(ctx.get('public_old_root',repo/'reports/adapter_diagnostics_20261017_v2'))
    legacy_path=old/'RUNS_FULL_PRECISION.csv'
    legacy=[]
    if legacy_path.is_file():
        source=legacy_path.relative_to(repo).as_posix() if legacy_path.is_relative_to(repo) else 'historical-public://RUNS_FULL_PRECISION.csv'
        digest=_sha(legacy_path);sources[source]=digest
        for index,row in enumerate(_csv(legacy_path)):
            historical=[p for p in proof.values() if p.get('source_run_id')==row['run_id'] and
                        p.get('source_job_id')=='HISTORICAL_RAW_V' and _method(p.get('method'))==_method(row.get('method'))
                        and p.get('validation_status')=='PASS' and all(_number(row.get(metric)) is not None and
                            _number(p.get('statistics',{}).get(metric)) is not None and
                            abs(float(row[metric])-float(p['statistics'][metric]))<=1e-10 for metric in ('MSE','MAE'))]
            normalized=_normalize(row,dict(job_id=row['run_id']),{},plan,source,digest,index,'rows','CONTEXTUAL',historical[0] if len(historical)==1 else None)
            legacy.append(normalized)
            tables['RUNS.csv'].append(normalized)
    legacy_selected_count=len(legacy)
    legacy_endpoint_path=old/'SAME_UPDATE_RESULTS.csv'
    legacy_endpoint_count=0
    if legacy_endpoint_path.is_file():
        source=legacy_endpoint_path.relative_to(repo).as_posix() if legacy_endpoint_path.is_relative_to(repo) else 'historical-public://SAME_UPDATE_RESULTS.csv'
        digest=_sha(legacy_endpoint_path);sources[source]=digest
        for index,row in enumerate(_csv(legacy_endpoint_path)):
            normalized=_normalize(dict(row,stage='search'),dict(job_id=row['run_id']),{},plan,source,digest,index,
                                  'common_endpoint_rows','CONTEXTUAL')
            legacy.append(normalized)
            tables['RUNS.csv'].append(normalized)
            tables['SAME_UPDATE.csv'].append(normalized)
            legacy_endpoint_count+=1
    cost_path=old/'COST_FULL_PRECISION.csv'
    if cost_path.is_file():
        source=cost_path.relative_to(repo).as_posix() if cost_path.is_relative_to(repo) else 'historical-public://COST_FULL_PRECISION.csv'
        digest=_sha(cost_path);sources[source]=digest
        tables['COST.csv'] += [_cost(row,{},source,digest,index,'CONTEXTUAL') for index,row in enumerate(_csv(cost_path))]
    historical_g,g_bindings=historical_g_costs(repo,legacy)
    tables['COST.csv']+=historical_g;sources.update(g_bindings)
    old_stats=old/'ERROR_BY_HORIZON.csv'
    if old_stats.is_file():
        source=old_stats.relative_to(repo).as_posix() if old_stats.is_relative_to(repo) else 'historical-public://ERROR_BY_HORIZON.csv'
        digest=_sha(old_stats);sources[source]=digest
        mapping={row['run_id']:row for row in legacy}
        for row in _csv(old_stats):
            run=mapping.get(row['run_id'])
            if run is None:
                raise ValueError('Historical horizon statistics lack public RUNS/SAME_UPDATE metadata: '+row['run_id'])
            tables['CHANNEL_HORIZON_STATS.csv'].append(dict(row,**{key:run.get(key,UNKNOWN) for key in
                ('row_id','job_id','cohort','method','seed','stage','evaluation')},record_origin='CONTEXTUAL',
                source_file=source,source_sha256=digest))
    if not tables['MULTIK_AUDIT.csv'] and (root/'MULTIK_AUDIT.csv').is_file():
        tables['MULTIK_AUDIT.csv']=_csv(root/'MULTIK_AUDIT.csv')
    tables['JOB_STATUS.csv']=outcomes
    compact_row_ids(tables)
    tables['COMPARISONS.csv']=_comparisons(tables['RUNS.csv'])
    tables['PAIRED_ORIGIN_EFFECTS.csv']=paired_origin_effects(tables,raw_source_rows)
    scientific=[row for row in outcomes if row['kind'] in SCIENCE_KINDS]
    new_complete=sum(row['status']=='COMPLETE' for row in scientific)
    reused=sum(row['kind']=='reuse_pca' and row['status']=='REUSED_VERIFIED' for row in outcomes)
    counts=Counter(row['status'] for row in outcomes)
    state=dict(execution_status='COMPLETE' if outcomes and all(row['status'] in SUCCESS for row in outcomes) else 'PARTIAL',
               validation_status=validation.get('status','NOT_YET_VERIFIED'),research_status='DESCRIPTIVE_DEVELOPMENT_EVIDENCE',
               publication_status=ctx.get('publication_status','SEPARATE_REMOTE_PROOF'),
               new_executed_scientific_jobs=new_complete,reused_scientific_jobs=reused,
               new_scored_arms=sum(row.get('record_origin')=='NEW' and row.get('evaluation')=='V_select' and row.get('stage')!='job_attempt'
                                   for row in tables['RUNS.csv']),contextual_rows=legacy_selected_count,
               contextual_endpoint_rows=legacy_endpoint_count,
               contextual_cost_rows=sum(row.get('record_origin')=='CONTEXTUAL' for row in tables['COST.csv']),
               historical_G_context_cost_rows=len(historical_g),
               historical_context_is_not_new_or_reused_experiment_count=True,job_status_counts=dict(counts),
               original_registered_jobs=len(original_jobs),supplemental_registered_jobs=len(jobs)-len(original_jobs),
               supplemental_plans=supplemental_plans,
               plan_sha256=plan.get('plan_sha256',UNKNOWN),scientific_source_hashes=plan.get('source_hashes',{}),
               reporting_source_sha256=_sha(__file__),accuracy_threshold='UNREGISTERED',speedup_target='UNREGISTERED',
               required_operating_batch='UNREGISTERED',historical_T_exposure='Retained; no new T evaluation',
               V_scope='Repeated development V; not an independent unused holdout',input_sources=sources,
               raw_curve_F6='OMITTED: private context/prediction arrays are excluded from public source-only replay',
               ledger_summary=_read(root/'RESOURCE_LEDGER.json',{}),resume_queue=_read(root/'RESUME_QUEUE.json',{}))
    return tables,state


def _stats_groups(stats):
    groups=defaultdict(list)
    for row in stats:
        groups[row['row_id']].append(row)
    return groups


def figure_sources(tables):
    runs=tables['RUNS.csv'];costs=tables['COST.csv'];comparisons=tables['COMPARISONS.csv']
    run_lookup={row['row_id']:row for row in runs}
    grouped=_stats_groups(tables['CHANNEL_HORIZON_STATS.csv'])
    sources={f'F{index}':[] for index in range(1,6)}
    for cost in costs:
        latency=_number(cost.get('median_ms_per_origin'))
        if latency is None:
            continue
        related=_authoritative_cost_matches(cost,runs)
        match=related[0] if len(related)==1 else None
        historical=_historical_context_cost(cost)
        if match is None and not historical:
            continue
        mse=_number(_value(cost,'accuracy_MSE','accuracy_reference_MSE',default=None))
        mae=_number(_value(cost,'accuracy_MAE','accuracy_reference_MAE',default=None))
        if match:
            mse,mae=_number(match.get('MSE')),_number(match.get('MAE'))
        if mse is not None or mae is not None:
            sources['F1'].append(dict(cost_id=cost['cost_id'],run_id=match['run_id'] if match else cost.get('accuracy_run_id',UNKNOWN),
                cohort=cost['cohort'],method=cost['method'],seed=match.get('seed',UNKNOWN) if match else cost.get('seed',UNKNOWN),
                measurement_seed=cost.get('seed',UNKNOWN),measurement_job_id=cost.get('job_id',UNKNOWN),device=cost.get('device',UNKNOWN),
                batch_size=cost.get('batch_size',UNKNOWN),record_origin=cost['record_origin'],MSE=mse,MAE=mae,
                median_ms_per_origin=latency,p95_ms_per_origin=cost.get('p95_ms_per_origin',UNKNOWN),
                p90_ms_per_origin=cost.get('p90_ms_per_origin',UNKNOWN),comparison_scope=cost.get('comparison_scope',UNKNOWN),
                hardware_protocol_scope=cost.get('hardware_protocol_scope'),host_identity=cost.get('host_identity',UNKNOWN),
                cost_authority='EXACT_SCORED_BINDINGS_FULL_V_PARITY' if match else 'NONAUTHORITATIVE_HISTORICAL_CONTEXT',
                model_scope='other_native' if any(name in str(cost['method']).lower() for name in ('moirai','chronos2','chronos-2')) else 'primary_Bolt_or_direct',
                source_file=cost['source_file'],source_sha256=cost['source_sha256'],
                accuracy_source_file=match['source_file'] if match else _value(cost,'accuracy_source_file',default=cost['source_file']),
                accuracy_source_sha256=match['source_sha256'] if match else _value(cost,'accuracy_source_sha256',default=cost['source_sha256'])))
            sources['F5'].append(dict(kind='warm_latency',method=cost['method'],cohort=cost['cohort'],
                seed=match.get('seed',UNKNOWN) if match else cost.get('seed',UNKNOWN),
                measurement_seed=cost.get('seed',UNKNOWN),measurement_job_id=cost.get('job_id',UNKNOWN),
            device=cost.get('device',UNKNOWN),batch_size=cost.get('batch_size',UNKNOWN),record_origin=cost['record_origin'],
            seconds=latency/1000,metric='median_ms_per_origin',value=latency,
            cost_id=cost.get('cost_id'),cost_authority='EXACT_SCORED_BINDINGS_FULL_V_PARITY' if match else 'NONAUTHORITATIVE_HISTORICAL_CONTEXT',
            core_dtype=cost.get('core_dtype',UNKNOWN),prediction_dtype=cost.get('prediction_dtype',UNKNOWN),
            p90_ms_per_origin=cost.get('p90_ms_per_origin',UNKNOWN),hardware_protocol_scope=cost.get('hardware_protocol_scope'),
            comparison_scope=cost.get('comparison_scope',UNKNOWN),host_identity=cost.get('host_identity',UNKNOWN),
            p95_ms_per_origin=cost.get('p95_ms_per_origin',UNKNOWN),source_file=cost['source_file'],source_sha256=cost['source_sha256']))
    for comparison in comparisons:
        if comparison.get('match_status')!='MATCHED':
            continue
        candidate=run_lookup.get(comparison.get('candidate_row_id'));control=run_lookup.get(comparison.get('control_row_id'))
        if not candidate or not control:
            continue
        candidate_stats=grouped.get(candidate['row_id'],[]);control_stats=grouped.get(control['row_id'],[])
        if not candidate_stats or not control_stats:
            continue
        channels=sorted({str(row['channel_id']) for row in candidate_stats})
        if comparison['question']=='D' and candidate.get('evaluation')=='V_select':
            for horizon in sorted({int(float(row['horizon'])) for row in candidate_stats}):
                left=aggregate_statistics([r for r in candidate_stats if int(float(r['horizon']))==horizon],channels)
                right=aggregate_statistics([r for r in control_stats if int(float(r['horizon']))==horizon],channels)
                sources['F2'].append(dict(cohort=candidate['cohort'],seed=candidate['seed'],method=candidate['method'],horizon=horizon,
                    delta_MSE=left['MSE']-right['MSE'] if left['MSE'] is not None and right['MSE'] is not None else None,
                    delta_MAE=left['MAE']-right['MAE'] if left['MAE'] is not None and right['MAE'] is not None else None,
                    coverage_channels=left['coverage_channels'],registered_channels=len(channels),comparison_id=comparison['comparison_id'],
                    source_file=candidate['source_file'],source_sha256=candidate['source_sha256']))
        if comparison['question']=='P':
            if candidate['stage']=='search':
                sources['F3'].append(dict(kind='common_endpoint',cohort=candidate['cohort'],seed=candidate['seed'],
                    stage=candidate['stage'],epoch=candidate.get('epoch'),evaluation=candidate['evaluation'],
                    delta_MSE=comparison.get('delta_MSE'),delta_MAE=comparison.get('delta_MAE'),
                    comparison_id=comparison['comparison_id'],source_file=candidate['source_file'],source_sha256=candidate['source_sha256']))
                continue
            for channel in channels:
                left=aggregate_statistics([r for r in candidate_stats if str(r['channel_id'])==channel],[channel])
                right=aggregate_statistics([r for r in control_stats if str(r['channel_id'])==channel],[channel])
                sources['F3'].append(dict(kind='channel',cohort=candidate['cohort'],seed=candidate['seed'],channel_id=channel,
                    stage=candidate['stage'],epoch=candidate.get('epoch',UNKNOWN),
                    delta_MSE=left['MSE']-right['MSE'] if left['MSE'] is not None and right['MSE'] is not None else None,
                    delta_MAE=left['MAE']-right['MAE'] if left['MAE'] is not None and right['MAE'] is not None else None,
                    comparison_id=comparison['comparison_id'],source_file=candidate['source_file'],source_sha256=candidate['source_sha256']))
    family_timers=set()
    for row in runs:
        if row.get('record_origin') in ('NEW','REUSED_VERIFIED') and row.get('evaluation')=='A_stop' and row.get('method') in ('B2_PCA','B2_DCA'):
            sources['F4'].append(dict(kind='forecast_curve',cohort=row['cohort'],seed=row['seed'],method=row['method'],
                epoch=row.get('epoch',UNKNOWN),MSE=row.get('MSE'),MAE=row.get('MAE'),selected_epoch=row.get('selected_epoch',UNKNOWN),
                source_file=row['source_file'],source_sha256=row['source_sha256']))
        if row.get('record_origin') in ('NEW','REUSED_VERIFIED') and row.get('evaluation')=='V_select' and row.get('stage')!='job_attempt':
            for key in ('preparation_seconds','search_seconds','refit_seconds','cache_seconds','evaluation_seconds','refit_and_evaluation_seconds'):
                seconds=_number(row.get(key))
                if seconds is not None:
                    sources['F5'].append(dict(kind='training_cost',method=row['method'],cohort=row['cohort'],seed=row['seed'],
                        device=row['device'],batch_size=UNKNOWN,record_origin=row['record_origin'],seconds=seconds,
                        metric=key,value=seconds,source_file=row['source_file'],source_sha256=row['source_sha256']))
            for key,value in (_object(row.get('result_timers')) or {}).items():
                identifier=(row.get('job_id'),key)
                seconds=_number(value)
                if seconds is not None and identifier not in family_timers:
                    family_timers.add(identifier)
                    sources['F5'].append(dict(kind='training_cost',method='JOB_FAMILY '+str(row.get('job_id')),cohort=row['cohort'],seed=row['seed'],
                        device=row['device'],batch_size=UNKNOWN,record_origin=row['record_origin'],seconds=seconds,metric=key,value=seconds,
                        timing_scope='Recorded combined/family timer; scopes can overlap and must not be summed',
                        source_file=row['source_file'],source_sha256=row['source_sha256']))
    for row in tables['JOB_STATUS.csv']:
        if str(row.get('job_id','')).startswith('DCA_INIT'):
            sources['F4'].append(dict(kind='initialization_status',cohort=row.get('cohort'),method='DCA initialization',
                seed=UNKNOWN,epoch=UNKNOWN,MSE=None,MAE=None,status=row['status'],reason=row.get('reason'),
                source_file=row['source_file'],source_sha256=row['source_sha256']))
        if row.get('kind')=='initialization':
            for metric in ('initialization_preparation_seconds','parent_job_wall_seconds'):
                seconds=_number(row.get(metric))
                if seconds is not None:
                    parent=metric=='parent_job_wall_seconds'
                    sources['F5'].append(dict(kind='training_cost',job_id=row['job_id'],method='DCA_INIT '+str(row.get('fit_scope')),
                        cohort=row['cohort'],seed='SHARED_ONCE',K=row.get('K'),fit_scope=row.get('fit_scope'),device='cpu',batch_size=UNKNOWN,
                        record_origin='NEW',attempt_status=row.get('status'),seconds=seconds,metric=metric,value=seconds,
                        timing_scope='Shared once per dataset/K/scope; preparation and parent job wall overlap; do not sum',
                        source_file=row['resource_receipt_source_file'] if parent else row['source_file'],
                        source_sha256=row['resource_receipt_source_sha256'] if parent else row['source_sha256']))
    return sources


def _plot_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8,'svg.hashsalt':NAME,
        'axes.prop_cycle':plt.cycler(color=COLORS),'savefig.facecolor':'white'})
    return plt


def _empty(ax, reason):
    ax.text(.5,.55,'UNAVAILABLE',ha='center',va='center',transform=ax.transAxes,fontweight='bold',fontsize=13)
    ax.text(.5,.38,reason,ha='center',va='center',wrap=True,transform=ax.transAxes,fontsize=9)
    ax.set_xticks([]);ax.set_yticks([])


def _facet(row):
    base=str(row.get('cohort'))+' | '+str(row.get('device'))+' | B'+str(row.get('batch_size'))+' | '+str(row.get('record_origin'))
    return base+' | '+str(row['hardware_protocol_scope']) if _known(row.get('hardware_protocol_scope')) else base


def _cost_labels(rows):
    keys=[(_facet(row),str(row['method']),str(row.get('seed',UNKNOWN))) for row in rows]
    repeated=Counter(keys)
    labels=[]
    for row,key in zip(rows,keys):
        label=str(row['method'])+' accuracy s'+str(_value(row,'seed'))
        if repeated[key]>1:
            label+=' | measurement '+str(_value(row,'cost_id','measurement_job_id','source_file'))
        labels.append(label)
    return labels


def render_figures(sources, output, dpi=300):
    plt=_plot_style();output=Path(output);folder=output/'figures';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    titles={'F1':'Accuracy versus whole-path latency','F2':'Decoder effect by horizon: candidate minus shared',
            'F3':'P_SPLIT minus P_LOCAL_CONTROL: paired channel effects',
            'F4':'PCA / DCA A_stop curves and initialization status','F5':'Whole-path latency and preparation / fitting cost'}
    for identifier,rows in sources.items():
        source_path=folder/(identifier+'_source.csv')
        _write_csv(source_path,rows,['kind','cohort','method','seed','source_file','source_sha256'])
        usable=False;reason='No completed matched public statistics yet'
        if identifier=='F1':
            facets=sorted({_facet(row)+' | '+row.get('model_scope','primary') for row in rows})
            row_labels=_cost_labels(rows)
            identities=sorted(set(row_labels))
            markers=('o','s','^','D','v','P','X','h')
            styles={identity:(COLORS[n%len(COLORS)],markers[n%len(markers)]) for n,identity in enumerate(identities)}
            heights=[]
            for facet in facets or ['']:
                labels=[textwrap.fill(label,width=30) for row,label in zip(rows,row_labels)
                        if _facet(row)+' | '+row.get('model_scope','primary')==facet]
                max_lines=max([label.count('\n')+1 for label in labels]+[2])
                heights += [3.2+.18*textwrap.fill(facet,width=58).count('\n'),
                            max(.5,.18*math.ceil((len(set(labels))+1)/2)*max_lines+.15)]
            fig,axes=plt.subplots(2*max(len(facets),1),2,figsize=(12,sum(heights)),squeeze=False,
                                 gridspec_kw={'height_ratios':heights},layout='constrained')
            for i,facet in enumerate(facets):
                group=[(row,label) for row,label in zip(rows,row_labels) if _facet(row)+' | '+row.get('model_scope','primary')==facet]
                for j,metric in enumerate(('MSE','MAE')):
                    ax=axes[2*i,j];legend_ax=axes[2*i+1,j];points=[]
                    for row,label in group:
                        x,y=_number(row.get('median_ms_per_origin')),_number(row.get(metric))
                        if x is None or y is None or x<=0:
                            continue
                        color,marker=styles[label]
                        usable=True;points.append((x,y));ax.scatter(x,y,s=38,color=color,marker=marker,
                            label=textwrap.fill(label,width=30))
                    frontier=[];best=math.inf
                    for x,y in sorted(points):
                        if y<best:
                            frontier.append((x,y));best=y
                    if frontier:
                        ax.plot([x for x,y in frontier],[y for x,y in frontier],':',color='#555555',lw=1,label='Observed Pareto frontier')
                    ax.set_xscale('log');ax.set_xlabel('Warm whole-path ms/origin (log scale)');ax.set_ylabel(metric+' (normalized)')
                    ax.set_title(textwrap.fill(facet,width=58),fontsize=9,pad=9);ax.grid(alpha=.15)
                    handles,labels=ax.get_legend_handles_labels()
                    unique=dict(zip(labels,handles));legend_ax.set_axis_off()
                    if unique:legend_ax.legend(unique.values(),unique.keys(),loc='center',ncol=2,frameon=False,
                                               fontsize=7.5,columnspacing=1.,handletextpad=.4)
            if not facets:
                for ax in axes[0]:_empty(ax,'No cost with an exact public accuracy binding')
                for ax in axes[1]:ax.set_axis_off()
        elif identifier=='F2':
            cohorts=sorted({row['cohort'] for row in rows});fig,axes=plt.subplots(max(len(cohorts),1),2,figsize=(12,max(3.3*len(cohorts),3.5)),squeeze=False,layout='constrained')
            for i,cohort in enumerate(cohorts):
                groups=defaultdict(list)
                for row in rows:
                    if row['cohort']==cohort:groups[(row['method'],str(row['seed']))].append(row)
                for j,metric in enumerate(('delta_MSE','delta_MAE')):
                    ax=axes[i,j]
                    for n,((method,seed),group) in enumerate(sorted(groups.items())):
                        points=sorted((int(float(row['horizon'])),_number(row.get(metric))) for row in group if _number(row.get(metric)) is not None)
                        if points:
                            usable=True;ax.plot([v[0] for v in points],[v[1] for v in points],label=method+' s'+seed,
                                color=COLORS[n//2%len(COLORS)],linestyle='-' if n%2==0 else '--',lw=1.4)
                    for boundary in ((16.5,32.5) if cohort=='jena' else (4.5,8.5)):ax.axvline(boundary,color='#999999',ls=':',lw=.8)
                    ax.axhline(0,color='#444444',lw=.8);ax.set_xlabel('Horizon');ax.set_ylabel(metric+' (negative improves)')
                    ax.set_title(cohort+' | fixed P/E, each head selected on A_stop');ax.legend(frameon=False,ncol=2)
            if not cohorts:
                for ax in axes.flat:_empty(ax,reason)
        elif identifier=='F3':
            def p_facet(row):
                return str(row['cohort'])+' | s'+str(row['seed'])+' | '+('common endpoints' if row.get('kind')=='common_endpoint' else 'selected A60 V')
            facets=sorted({p_facet(row) for row in rows})
            fig,axes=plt.subplots(max(len(facets),1),2,figsize=(12,max(2.8*len(facets),3.5)),squeeze=False,layout='constrained')
            for i,facet in enumerate(facets):
                group=[row for row in rows if p_facet(row)==facet]
                for j,metric in enumerate(('delta_MSE','delta_MAE')):
                    ax=axes[i,j]
                    if group[0].get('kind')=='common_endpoint':
                        for evaluation in sorted({row.get('evaluation') for row in group}):
                            points=sorted((float(row['epoch']),_number(row.get(metric))) for row in group
                                          if row.get('evaluation')==evaluation and _number(row.get(metric)) is not None)
                            if points:
                                usable=True;ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',label=evaluation)
                        ax.set_xlabel('Common search endpoint (epoch)');ax.legend(frameon=False)
                    else:
                        points=sorted((int(float(row['channel_id'])),_number(row.get(metric))) for row in group if _number(row.get(metric)) is not None)
                        if points:
                            usable=True;ax.vlines([p[0] for p in points],0,[p[1] for p in points],color=COLORS[j],lw=1.5)
                            ax.scatter([p[0] for p in points],[p[1] for p in points],color=COLORS[j],s=18)
                        ax.set_xlabel('Registered channel index')
                    ax.axhline(0,color='#444444',lw=.8);ax.set_ylabel(metric+' (negative improves)');ax.set_title(facet)
            if not facets:
                for ax in axes.flat:_empty(ax,reason)
        elif identifier=='F4':
            curves=[row for row in rows if row.get('kind')=='forecast_curve']
            status=[row for row in rows if row.get('kind')=='initialization_status']
            facets=sorted({str(row['cohort'])+' | s'+str(row['seed']) for row in curves})
            fig,axes=plt.subplots(1,max(len(facets),1),figsize=(max(6*len(facets),8),4),squeeze=False,layout='constrained')
            for i,facet in enumerate(facets):
                ax=axes[0,i];groups=defaultdict(list)
                for row in curves:
                    if str(row['cohort'])+' | s'+str(row['seed'])==facet:groups[row['method']].append(row)
                for method,group in groups.items():
                    points=sorted((_number(row.get('epoch')),_number(row.get('MSE'))) for row in group if _number(row.get('epoch')) is not None and _number(row.get('MSE')) is not None)
                    if points:
                        usable=True;ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',label=method)
                ax.set_xlabel('Search endpoint (epoch)');ax.set_ylabel('A_stop normalized macro MSE');ax.set_title(facet);ax.legend(frameon=False)
                local_status=[row for row in status if facet.startswith(str(row.get('cohort'))+' | ')]
                if local_status:ax.text(.02,.02,'DCA init: '+', '.join(sorted({str(row['status']) for row in local_status})),transform=ax.transAxes,fontsize=8)
            if not facets:
                _empty(axes[0,0],'; '.join(sorted({str(row['status']) for row in status})) or reason)
            if status and not curves:reason='Initialization status is available; no valid forecast curve'
        else:
            latency=[row for row in rows if row.get('kind')=='warm_latency']
            fitting=[row for row in rows if row.get('kind')=='training_cost']
            latency_labels=[textwrap.fill(label+' | '+_facet(row)+(' | core FP64, stored FP32' if 'ridge' in str(row['method']).lower() else ''),width=74)
                            for row,label in zip(latency,_cost_labels(latency))]
            fitting_labels=[textwrap.fill(str(row['cohort'])+' | '+str(row['method'])+' | '+
                ('SHARED_ONCE' if row.get('seed')=='SHARED_ONCE' else 's'+str(_value(row,'seed')))+
                (' | scope '+str(row['fit_scope'])+' K'+str(row.get('K',UNKNOWN)) if _known(row.get('fit_scope')) else '')+
                ' | '+str(row['metric']),width=74) for row in fitting]
            line_units=max([sum(label.count('\n')+1.65 for label in labels) for labels in (latency_labels,fitting_labels)]+[0])
            fig,axes=plt.subplots(1,2,figsize=(15,max(5,.16*line_units)),layout='constrained')
            for j,(group,label) in enumerate(((latency,'Warm whole-path ms/origin'),(fitting,'Recorded preparation / fitting seconds (scopes overlap; do not sum)'))):
                ax=axes[j]
                if group:
                    usable=True
                    labels=latency_labels if j==0 else fitting_labels
                    values=[float(r['value']) for r in group]
                    positions=[];edge=0.
                    for row_label in labels:
                        height=row_label.count('\n')+1.65
                        positions.append(edge+height/2);edge+=height
                    ax.barh(positions,values,color=COLORS[j],alpha=.85);ax.set_yticks(positions,labels,fontsize=6)
                    ax.set_ylim(edge,0);ax.set_xscale('log' if all(v>0 for v in values) else 'linear');ax.set_xlabel(textwrap.fill(label,width=42))
                else:_empty(ax,'UNKNOWN: no measured '+label)
        fig.suptitle(identifier+'  '+titles[identifier],fontsize=14,fontweight='bold')
        png=folder/(identifier+'.png');svg=folder/(identifier+'.svg')
        fig.savefig(png,dpi=dpi,metadata={'Software':NAME+' public sufficient-statistic replay'})
        fig.savefig(svg,metadata={'Date':None,'Creator':NAME+' public sufficient-statistic replay'})
        plt.close(fig)
        entries.append(dict(figure_id=identifier,png=png.relative_to(output).as_posix(),svg=svg.relative_to(output).as_posix(),
            source_csv=source_path.relative_to(output).as_posix(),source_sha256=_sha(source_path),
            status='AVAILABLE' if usable else 'UNAVAILABLE',n_rows=len(rows),reason=None if usable else reason,
            scope='Public sufficient statistics only; no private arrays or foundation calls'))
    _json(output/'FIGURE_INDEX.json',dict(figures=entries,reporting_source_sha256=_sha(__file__),dpi=dpi,
          F6='OMITTED: raw curves require private replay',reproducibility='Data/labels/source hashes; PNG renderer depends on font/library version'))
    return entries


def _format(value,digits=6):
    number=_number(value)
    return f'{number:.{digits}g}' if number is not None else str(value) if _known(value) else UNKNOWN


def _text_table(headers, rows):
    text=[[str(value) for value in row] for row in rows]
    if not text:text=[['UNKNOWN']+['']*(len(headers)-1)]
    widths=[max(len(str(headers[j])),max(len(row[j]) for row in text)) for j in range(len(headers))]
    return '```text\n'+'  '.join(str(value).ljust(widths[j]) for j,value in enumerate(headers))+'\n'+\
        '\n'.join('  '.join(value.ljust(widths[j]) for j,value in enumerate(row)) for row in text)+'\n```\n'


def _band_rows(tables):
    lookup={row['row_id']:row for row in tables['RUNS.csv']}
    stats=_stats_groups(tables['CHANNEL_HORIZON_STATS.csv']);rows=[]
    for comparison in tables['COMPARISONS.csv']:
        if comparison.get('match_status')!='MATCHED' or comparison.get('evaluation')!='V_select':continue
        candidate=lookup.get(comparison.get('candidate_row_id'));control=lookup.get(comparison.get('control_row_id'))
        if not candidate or not control:continue
        left,right=stats.get(candidate['row_id'],[]),stats.get(control['row_id'],[])
        if not left or not right:continue
        channels=sorted({str(row['channel_id']) for row in left})
        bands=((1,16),(17,32),(33,48)) if candidate['cohort']=='jena' else ((1,4),(5,8),(9,12))
        values=[]
        for lo,hi in bands:
            a=aggregate_statistics([r for r in left if lo<=int(float(r['horizon']))<=hi],channels)
            b=aggregate_statistics([r for r in right if lo<=int(float(r['horizon']))<=hi],channels)
            values.append('/'.join(_format(a[key]-b[key]) if a[key] is not None and b[key] is not None else UNKNOWN for key in ('MSE','MAE')))
        rows.append([candidate['cohort'],candidate['seed'],candidate['method']+'−'+control['method'],*values,
                     _format(comparison.get('delta_MSE'))+'/'+_format(comparison.get('delta_MAE')),candidate.get('n_origins',UNKNOWN)])
    return rows


def presentation_accuracy_rows(tables):
    runs=[row for row in tables['RUNS.csv'] if row.get('evaluation')=='V_select' and row.get('stage')!='job_attempt'
          and _number(row.get('MSE')) is not None]
    costs=tables['COST.csv'];rows=[]
    for row in runs:
        associated=[cost for cost in costs if authoritative_cost_join(cost,row)['status']=='PASS']
        for cost in associated or [{}]:
            raw_candidates=[other for other in runs if other.get('method')=='Raw' and
                all(_known(row.get(key)) and row.get(key)==other.get(key) for key in
                    ('cohort','scaler_sha256','target_sha256','mask_sha256','origins_sha256','channel_sha256','point_statistic','metric_scale'))]
            baseline=raw_candidates[0] if raw_candidates else {}
            raw_cost=[other for other in costs if _method(other.get('method'))=='Raw' and other.get('job_id')==cost.get('job_id') and
                      all(_known(cost.get(key)) and str(cost.get(key))==str(other.get(key)) for key in ('cohort','device','batch_size','record_origin'))
                      and len(_authoritative_cost_matches(other,runs))==1]
            latency=_number(cost.get('median_ms_per_origin'))
            raw_latency=_number(raw_cost[0].get('median_ms_per_origin')) if len(raw_cost)==1 else None
            out=dict(cohort=row['cohort'],method=row['method'],seed=row.get('seed'),C=row.get('C'),K=row.get('K'),
                device=cost.get('device',row.get('device')),batch_size=cost.get('batch_size',UNKNOWN),MSE=row.get('MSE'),MAE=row.get('MAE'),
                median_ms_per_origin=latency,p95_ms_per_origin=cost.get('p95_ms_per_origin'),
                speedup_vs_Raw=raw_latency/latency if raw_latency is not None and latency not in (None,0) else None,
                peak_allocated_bytes=cost.get('peak_allocated_bytes'),peak_reserved_bytes=cost.get('peak_reserved_bytes'),
                process_RSS_peak_bytes=cost.get('process_RSS_peak_bytes'),target_trained_params=row.get('target_trained_params'),
                total_resident_params=cost.get('total_resident_params'),core_dtype=row.get('core_dtype'),prediction_dtype=row.get('prediction_dtype'),
                record_origin=row.get('record_origin'),selection=row.get('selection_split',row.get('selection',UNKNOWN)),
                cost_record_origin=cost.get('record_origin',UNKNOWN),measurement_scope=cost.get('source_file',UNKNOWN),
                accuracy_comparison_status='EXACT_EVALUATION_COHORT' if baseline else 'UNKNOWN',
                cost_comparison_status='SAME_MEASUREMENT_JOB_BATCH' if raw_latency is not None else 'UNKNOWN',
                run_id=row['run_id'],source_file=row['source_file'],source_sha256=row['source_sha256'],cost_source_file=cost.get('source_file'))
            for metric in ('MSE','MAE'):
                left,right=_number(row.get(metric)),_number(baseline.get(metric))
                out['delta_'+metric+'_pct_vs_Raw']=100*(left-right)/right if left is not None and right not in (None,0) else None
            rows.append(out)
    for cost in costs:
        if not _historical_context_cost(cost):continue
        rows.append(dict(cohort=cost.get('cohort'),method=cost.get('method'),seed=cost.get('seed'),
            C=cost.get('C'),K=cost.get('K'),device=cost.get('device'),batch_size=cost.get('batch_size'),
            MSE=_number(_value(cost,'accuracy_MSE','accuracy_reference_MSE',default=None)),
            MAE=_number(_value(cost,'accuracy_MAE','accuracy_reference_MAE',default=None)),
            median_ms_per_origin=_number(cost.get('median_ms_per_origin')),p95_ms_per_origin=cost.get('p95_ms_per_origin'),
            p90_ms_per_origin=cost.get('p90_ms_per_origin'),comparison_scope=cost.get('comparison_scope'),
            hardware_protocol_scope=cost.get('hardware_protocol_scope'),host_identity=cost.get('host_identity',UNKNOWN),
            accuracy_binding_scope=cost.get('accuracy_binding_scope'),trials_source_file=cost.get('trials_source_file'),
            trials_source_sha256=cost.get('trials_source_sha256'),
            accuracy_source_file=cost.get('accuracy_source_file'),accuracy_source_sha256=cost.get('accuracy_source_sha256'),
            speedup_vs_Raw=None,delta_MSE_pct_vs_Raw=None,delta_MAE_pct_vs_Raw=None,
            peak_allocated_bytes=cost.get('peak_allocated_bytes'),peak_reserved_bytes=cost.get('peak_reserved_bytes'),
            process_RSS_peak_bytes=cost.get('process_RSS_peak_bytes'),target_trained_params=cost.get('target_trained_params'),
            total_resident_params=cost.get('total_resident_params'),core_dtype=cost.get('core_dtype'),prediction_dtype=cost.get('prediction_dtype'),
            record_origin='CONTEXTUAL',cost_record_origin='CONTEXTUAL',selection='HISTORICAL_COST_PROTOCOL',
            accuracy_comparison_status='SOURCE_RECORDED_HISTORICAL_ONLY',
            cost_comparison_status='NONAUTHORITATIVE_HISTORICAL_CONTEXT',measurement_scope=cost.get('scope',cost.get('source_file')),
            run_id=_value(cost,'accuracy_run_id',default='cost_context:'+str(cost.get('cost_id'))),
            source_file=cost.get('source_file'),source_sha256=cost.get('source_sha256'),cost_source_file=cost.get('source_file')))
    return rows


def selected_findings(tables):
    selected=[row for row in tables['COMPARISONS.csv'] if row.get('match_status')=='MATCHED'
              and row.get('stage')=='selected_refit' and row.get('evaluation')=='V_select']
    selected.sort(key=lambda row:tuple(str(row.get(key,'')) for key in ('cohort','seed','candidate','control')))
    failed_init=[row for row in tables['JOB_STATUS.csv'] if row.get('kind')=='initialization' and row.get('status')=='NOT_CONVERGED']
    failed_cohorts={row.get('cohort') for row in failed_init}
    selected=[row for row in selected if row.get('question')!='E' or row.get('cohort') not in failed_cohorts]
    def effect(row):
        values=[]
        for metric in ('MSE','MAE'):
            number=_number(row.get('delta_'+metric))
            values.append(metric+' '+(f'{number:+.6g} ('+('개선' if number<0 else '악화' if number>0 else '동일')+')'
                                      if number is not None else UNKNOWN))
        return ', '.join(values)
    def comparison(row):
        return f"{row['cohort']} seed {row['seed']}: {row['candidate']}−{row['control']} {effect(row)}"
    p=[comparison(row) for row in selected if row.get('question')=='P']
    paragraphs=['[확인] P 선택-refit V의 분해−대조 효과(음수 개선)는 '+('; '.join(p)+'.' if p else 'UNKNOWN.')]
    e=[comparison(row) for row in selected if row.get('question')=='E']
    for cohort in sorted(failed_cohorts):
        scopes=sorted({str(row.get('fit_scope',UNKNOWN)) for row in failed_init if row.get('cohort')==cohort})
        e.append(f"{cohort} DCA {'/'.join(scopes)} NOT_CONVERGED로 정상 완료 PCA/DCA 선택-refit 효과는 UNKNOWN")
    paragraphs.append('[확인] E 초기화의 미래 예측 효과는 '+('; '.join(e)+'.' if e else 'UNKNOWN.')+
                      ' 미수렴 초기화의 objective를 예측 성능으로 바꾸지 않는다.')
    lookup={row['row_id']:row for row in tables['RUNS.csv']}
    decoder=defaultdict(list)
    for row in selected:
        if row.get('question')=='D':decoder[(str(row['cohort']),str(row['seed']))].append(row)
    d=[]
    for (cohort,seed),rows in sorted(decoder.items()):
        linear=[row for row in rows if row.get('candidate')=='D_LINEAR' and row.get('control')=='D_SHARED']
        gain=[row for row in rows if row.get('candidate')=='D_GAIN' and row.get('control')=='D_SHARED']
        parts=['D_LINEAR−D_SHARED '+(effect(linear[0]) if len(linear)==1 else UNKNOWN)]
        left=lookup.get(linear[0].get('candidate_row_id')) if len(linear)==1 else None
        right=lookup.get(gain[0].get('candidate_row_id')) if len(gain)==1 else None
        if (left and right and all(row.get('stage')=='selected_refit' and row.get('evaluation')=='V_select'
                                  and row.get('status') in SUCCESS for row in (left,right))
                and match_rows(left,right,'D')['match_status']=='MATCHED'):
            values={}
            for metric in ('MSE','MAE'):
                a,b=_number(left.get(metric)),_number(right.get(metric))
                values['delta_'+metric]=a-b if a is not None and b is not None else None
            parts.append('D_LINEAR−D_GAIN '+effect(values))
        else:parts.append('D_LINEAR−D_GAIN UNKNOWN')
        d.append(f"{cohort} seed {seed}: "+'; '.join(parts))
    paragraphs.append('[확인] D의 같은 fixed P/E 선택 recipe에서 '+('; '.join(d)+'.' if d else 'D_LINEAR−D_SHARED 및 D_LINEAR−D_GAIN 효과는 UNKNOWN.')+
                      ' MSE와 MAE 및 두 seed의 방향을 각각 읽으며 이를 단일 승자로 합치지 않는다.')
    allowed={lookup[row[key]]['run_id'] for row in selected for key in ('candidate_row_id','control_row_id')
             if row.get(key) in lookup}
    selected_tables={**tables,'RUNS.csv':[row for row in tables['RUNS.csv']
                     if row.get('stage')=='selected_refit' and row.get('evaluation')=='V_select']}
    measured=[row for row in presentation_accuracy_rows(selected_tables) if row.get('run_id') in allowed
              and row.get('cost_record_origin')!='CONTEXTUAL' and row.get('cost_comparison_status')=='SAME_MEASUREMENT_JOB_BATCH'
              and _number(row.get('speedup_vs_Raw')) is not None] if tables['COST.csv'] else []
    grouped=defaultdict(list)
    for row in measured:
        grouped[tuple(str(row.get(key,UNKNOWN)) for key in ('cohort','measurement_scope','device','batch_size'))].append(row)
    c=[]
    for (cohort,scope,device,batch),rows in sorted(grouped.items()):
        ordered=sorted(rows,key=lambda row:(float(row['speedup_vs_Raw']),str(row.get('method')),str(row.get('seed'))))
        limits=[ordered[0]] if len(ordered)==1 else [ordered[0],ordered[-1]]
        values=[f"{row['method']} seed {row.get('seed',UNKNOWN)} {_format(row['median_ms_per_origin'])} ms/origin, "
                f"Raw speedup {_format(row['speedup_vs_Raw'])}x"+(' (Raw보다 느림)' if float(row['speedup_vs_Raw'])<1 else '') for row in limits]
        c.append(f"{cohort} {device} B{batch} [{scope}]: "+' ~ '.join(values))
    unknown=len(allowed-{row['run_id'] for row in measured})
    paragraphs.append('[확인] 같은 측정 job/device/batch와 정확한 저장 예측·bundle의 full-V parity를 통과한 Raw 대비 비용은 '+
                      ('; '.join(c)+f'. 그 외 선택 run {unknown}개의 같은 조건 Raw 비용 비교는 UNKNOWN.' if c else 'UNKNOWN.')+
                      ' Historical 비용은 별도 contextual 조건이며 이 speedup에 섞지 않는다.')
    return paragraphs


def write_documents(tables,state,output):
    output=Path(output)
    runs=tables['RUNS.csv'];comparisons=tables['COMPARISONS.csv'];costs=tables['COST.csv']
    selected=[row for row in runs if row.get('evaluation')=='V_select' and row.get('stage')=='selected_refit' and _number(row.get('MSE')) is not None]
    matched=[row for row in comparisons if row.get('match_status')=='MATCHED' and row.get('stage')=='selected_refit' and row.get('evaluation')=='V_select']
    common=[row for row in comparisons if row.get('match_status')=='MATCHED' and row.get('stage')=='search']
    effect_rows=[[row['cohort'],row['seed'],row['candidate']+'−'+row['control'],_format(row.get('delta_MSE')),
                  _format(row.get('delta_MAE')),_format(row.get('delta_MSE_pct'))+'%',row.get('matching_label','MATCHED')]
                 for row in matched]
    coverage=Counter(str(row.get('cohort'))+' s'+str(row.get('seed'))+' '+str(row.get('method')) for row in selected if row.get('record_origin')!='CONTEXTUAL')
    nonconverged=[row for row in tables['JOB_STATUS.csv'] if row.get('status')=='NOT_CONVERGED']
    headline=(f"실행 {state['execution_status']} / 독립 검증 {state['validation_status']} / "
              f"연구 {state['research_status']} / 게시 {state['publication_status']}")
    counts=(f"신규 과학 job 완료 {state['new_executed_scientific_jobs']}개, 검증 재사용 job {state['reused_scientific_jobs']}개, "
            f"신규 선택 결과 arm {state['new_scored_arms']}개. 과거 contextual 선택 요약 {state['contextual_rows']}행과 "
            f"과거 endpoint {state.get('contextual_endpoint_rows',0)}행은 별도 보존하며 신규 실행·재사용 job 수에 포함하지 않는다.")
    findings=[]
    for row in matched:
        findings.append(f"- [확인] {row['cohort']} seed {row['seed']}: {row['candidate']}−{row['control']} "
            f"MSE {_format(row.get('delta_MSE'))} ({_format(row.get('delta_MSE_pct'))}%), "
            f"MAE {_format(row.get('delta_MAE'))} ({_format(row.get('delta_MAE_pct'))}%). "
            f"대응 상태 {row.get('matching_label','MATCHED')}. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.")
    if not findings:findings=['- [확인] 현재 완전한 대응 signature를 증명한 새 선택 결과 비교가 없다. 완료된 개별 점수를 대응 효과로 확대하지 않는다.']
    common_findings=[f"- [확인] {row['cohort']} seed {row['seed']} {row.get('evaluation',UNKNOWN)} epoch {_format(row.get('epoch'))}: "
        f"{row['candidate']}−{row['control']} MSE {_format(row.get('delta_MSE'))}, MAE {_format(row.get('delta_MAE'))}. "
        f"대응 상태 {row.get('matching_label','MATCHED_COMMON_ENDPOINT')}." for row in common]
    if not common_findings:common_findings=['- [확인] 완전한 대응 signature를 증명한 공통 endpoint 비교는 UNKNOWN이다.']
    dca_note=(f"[확인] 등록 DCA 초기화 {len(nonconverged)}개 scope가 NOT_CONVERGED다. "
              "등록 optimizer 한도에서 종료했으며 downstream DCA 예측은 정상 완료로 기록하지 않는다. PCA 곡선만 있으면 DCA 미래 점수를 만들지 않는다.") if nonconverged else \
             '[확인] DCA 초기화/예측 실행 상태는 JOB_STATUS.csv와 개별 receipt로 구분한다. 미실행 DCA 예측은 UNKNOWN이다.'
    readme=f'''# P/E/D 개선 실험 — 2026-10-18

{headline}

[확인] {counts}

정확도 허용 열화율·목표 speedup·필수 운영 batch는 **UNREGISTERED**다. 임의 합격선을 만들지 않았다. V는 반복 사용된 개발 평가 구간이다. 과거 T 사용 이력은 유지하며 이번 캠페인에는 새 T 평가가 없다.

처음 볼 파일은 [분석](RESULT_ANALYSIS_KO.md), [정확한 수치](RUNS.csv), [대응 효과](COMPARISONS.csv), [origin별 대응 효과](PAIRED_ORIGIN_EFFECTS.csv), [비용](COST.csv), [발표 표](PPT_TABLES_KO.md)다. 그림 [F1](figures/F1.png)–[F5](figures/F5.png)는 각 source CSV에서 생성된다. 게시 상태는 로컬 build에서 추정하지 않으며 최종 remote 근거는 별도 REMOTE_PROOF.json을 따른다.

{dca_note}

선택 결과와 공통 endpoint 비교는 별도 행이다. 서로 다른 selected epoch/update는 같은 A_stop 선택 recipe의 결과 비교이며 같은 update 효과로 표현하지 않는다. 부정 결과·미실행·실패는 남긴다.

CHANNEL_HORIZON_STATS.csv는 row_id/horizon/channel_id/SSE/SAE/count의 6열이다. 각 통계의 method·cohort·seed·source·hash·상태는 RUNS.csv의 같은 row_id에 한 번만 저장한다. r00000 형식 ID는 해당 보고서 inventory 순서의 결정적 식별자이며 원 producer_row_id도 RUNS에 보존한다. 이 정규화로 모든 기존·신규 full precision 통계를 유지하면서 반복 metadata를 줄였다.

공개 통계만으로 표·그림을 재생성하는 실제 명령은 [재현 안내](REPRODUCE_KO.md)에 있다. 원 예측 배열과 checkpoint 검산은 비공개 자산을 요구한다. [남은 작업](RESUME_QUEUE.json)과 [원장](RESOURCE_LEDGER.json)을 함께 확인한다.
'''
    analysis=f'''# 결과 해석과 적용 범위

{headline}

[확인] {counts}

선택 결과 핵심 해석 — MATCHED selected_refit / V_select만 사용한다.

{(chr(10)+chr(10)).join(selected_findings(tables))}

선택-refit V 상세 — 각자의 A_stop 선택 epoch를 허용한 recipe 대응이다.

{chr(10).join(findings)}

공통 endpoint 상세 — 선택-refit 점수와 별개이며 같은 epoch/update/exposure/sampler 조건이다.

{chr(10).join(common_findings)}

{dca_note}

P 분해 효과의 주 대조는 P_SPLIT−P_LOCAL_CONTROL이다. B2 대비 차이는 입력 구조·용량·연산 방식 전체의 차이다. E 비교는 B2+PCA 대 B2+DCA이며 초기 objective/수치 유효성을 미래 예측 개선으로 대신하지 않는다. D 비교는 같은 고정 P/E에서 각 head가 A_stop으로 선택한 lambda/gamma를 사용한 recipe 비교다.

Jena 밴드는 h1–16/17–32/33–48, PeMS32는 h1–4/5–8/9–12다. 전체 macro MSE/MAE는 채널별 SSE/SAE/count를 먼저 합산한 뒤 고정 채널 집합의 ratio를 평균한다. 밴드 macro 값을 직접 더하지 않는다. 관측이 없는 채널을 조용히 빼지 않으며 해당 밴드는 UNDEFINED로 표시한다.

두 seed를 전체 분포의 안정성이나 보편적 유의성으로 해석하지 않는다. origin별 의존성이 있어 IID bootstrap/p-value를 자동 추가하지 않았다. delta는 candidate−control이며 음수가 오차 개선이다.

PAIRED_ORIGIN_EFFECTS.csv는 MATCHED selected-refit V 비교의 origin_index별 candidate/control MSE·MAE와 delta다. 공개 producer JSON의 채널 SSE/SAE/count를 고정 C채널 macro로 계산하며 어느 채널의 count가 0이면 해당 origin은 UNDEFINED다. n_origins·평가 stride·horizon overlap·양쪽 source binding을 함께 남긴다. 이 origin별 macro를 단순 평균해 전체 macro라고 주장하지 않는다. 전체 지표는 RUNS/채널·horizon 통계의 전역 채널 count 집계를 따른다.

과거 Jena GPU batch1의 B2 지연이 Raw보다 나빴던 조건과 Ridge의 강한 기준선 지위를 보존한다. CPU/GPU, batch1/8, historical/new hardware scope를 한 순위로 합치지 않는다. 압축 P/E/D 파라미터 감소는 동결 F 상주 메모리 감소를 뜻하지 않는다. L=512는 유지되며 K는 채널 축 압축이다.

Ridge 파생 레이블은 CPU core FP64 / stored FP32다. 과거 저장 dtype 표기가 연산 dtype처럼 읽힐 수 있던 부분을 이 파생 보고서에서 구분했다. 과거 봉인 파일과 예측값은 수정하지 않는다.

G-only의 기존 Jena/PeMS32 CPU batch1/8 비용 4행은 exact run/checkpoint/scaler 및 원 receipt/trials SHA를 대사한 HISTORICAL_CPU2_CONTEXT다. 실제 host identity는 UNKNOWN이며 새 하드웨어의 Raw speedup 계산에 사용하지 않는다. 원 p90을 p95로 바꾸지 않았다. Jena의 이전 prediction manifest는 현재 clean tree에 없어 기존 진단 score/checkpoint/scaler를 통한 transitive binding으로 제한한다. PeMS32는 원 공개 prediction manifest를 직접 대사한다. 이 4행은 신규 실행·신규 측정 수에 포함하지 않는다.

현재 seed/data/method 커버리지:

{_text_table(['cohort/seed/method','selected rows'],[[key,value] for key,value in sorted(coverage.items())])}

미완료와 UNKNOWN은 RESUME_QUEUE.json에 남긴다. F6 원 곡선은 비공개 배열을 요구하므로 공개 통계 재생성 범위에서 제외했다.
'''
    contract='''# 방법·비교 계약

[확인] 계획·과학 source 식별자는 EXPERIMENT_PLAN.json에 봉인된다. 보고서 producer hash는 REPORT_STATE.json/FIGURE_INDEX.json에 따로 기록하며 보고서 편집은 과거 과학 cache identity를 바꾸지 않는다.

통계 schema: CHANNEL_HORIZON_STATS의 (row_id,horizon,channel_id)는 RUNS.row_id로 조인한다. SSE/SAE/count는 원 full precision 그대로이며 contextual 여부와 통계 source binding은 RUNS의 record_origin/statistics_source_file/statistics_source_sha256을 따른다. SAME_UPDATE와 COMPARISONS도 같은 짧은 row_id를 사용하고, 원 긴 ID는 RUNS.producer_row_id다. 원 ID 중복이나 조인 누락은 생성 오류이며 통계를 조용히 제외하지 않는다.

PAIRED_ORIGIN_EFFECTS는 공개 origin별 SSE/SAE/count에서 계산한 기술 통계다. canonical matching은 index 정렬을 증명하며 실제 시간 ID 대신 origin_index를 공개한다. 한 origin의 고정 채널 중 N_c=0이면 UNDEFINED이고, IID 독립성이나 p-value를 주장하지 않는다.

고정 경로는 X→P→E→동결 단변량 F→lag-major D→원채널 미래다. P 후보는 kernel9/25, width32, bias 없음, causal replicate padding, 마지막 1×1만 zero-init이며 1,152 params다. P_SPLIT은 trailing mean25의 잔차/추세, 대조는 X/2 두 branch다. CP0 identity 이후 학습된 P의 가역성·완전 정보 보존은 보장하지 않는다.

학습 stride1, 평가 strideH; seed92601/92602; CP0/.25/1/3. A_fit/A_stop으로 endpoint를 선택한 뒤 scaler/PCA/DCA를 fresh A60 fit한다. AdamW lr.001, weight decay0, clip1, effective16/micro4, F row128, FP32, checkpoint_inputsFalse다. V는 선택에 사용하지 않는다.

D_SHARED/D_HORIZON/D_LINEAR/D_GAIN은 같은 fixed P/E와 fitting RMS를 사용한다. linear/gain gamma={1,10,100}, lambda={.0001,.001,.01,.1,1}, degree1이다. 불균등 mask는 전체 채널 count N_c 정규화를 유지한다. head별 설정 선택이 다를 수 있다.

DCA는 공식 고정 source objective, T5/n_init1/PCA64 init, CPU FP64 L-BFGS-B maxiter200/maxfun1000/120초를 사용한다. A_fit/A60 scope를 분리하며 미수렴은 downstream 정상 arm을 허용하지 않는다. 저장 정렬 E32를 설치한 뒤 actual FP32 torch pinv로 D[0]을 만든다. deployment loader는 DCA를 최적화하지 않는다.

MATCHED는 같은 data/split/mask/scaler/origins/channel order/F snapshot/device/core precision/recipe를 증명하고 등록 intervention만 허용한다. learned checkpoint 전체 SHA가 같아야 한다는 뜻은 아니다. independent verifier의 실제 NPZ canonical fingerprint를 우선하며 다른 hash convention은 직접 동치로 취급하지 않는다. 공통 endpoint는 epoch/update/exposure/sampler도 같아야 한다.

선택-refit은 같은 선택 recipe에서 각자 선택된 epoch를 허용하고 MATCHED_RECIPE_SELECTED_REFIT로 표시한다. 공통 endpoint는 MATCHED_COMMON_ENDPOINT다. 누락된 필수 signature는 INCOMPLETE, 실제 차이는 CONTEXTUAL이다.

G-only는 원채널 own-history의 channel-shared L→hidden→H innovation + current anchor 직접 예측기다. F를 호출하지 않는다. B2는 TSMixer 아이디어 일부를 적용한 변형, B4/새 P는 TimeMixer 전체 재현이 아니다. 공식 출처와 차이는 METHOD_REFERENCES.md를 따른다.

비용은 physical+mask→fill→normalize→P/E/F/D/refiner→inverse→CPU의 whole path다. warm10/timed20, 별도 stage probe, CPU BLAS2/GPU B1/B8을 분리한다. p95는 20개 표본의 기술 통계다. cost_valid는 정확한 scored bundle reload와 full-V parity 통과 후에만 성립한다. cold/준비/검색/refit/평가/메모리 범위는 분리하며 미측정은 UNKNOWN이다.
'''
    reproduce=f'''# 재현 안내

과학 실행 계획 SHA: `{state['plan_sha256']}`. 보고서 producer SHA: `{state['reporting_source_sha256']}`.

공개 CSV/JSON에서 표·그림만 별도 경로에 재생성:

```powershell
python -m research.adapter_improvements_20261018_v1.reporting --public-input reports/adapter_improvements_20261018_v1 --output .cache/adapter_improvements_public_replay
```

위 명령은 F/model/private NPZ를 열지 않는다. RUNS/COMPARISONS/COST/CHANNEL_HORIZON_STATS/SAME_UPDATE/PAIRED_ORIGIN_EFFECTS 및 figure source CSV만 읽는다. output을 원 report 폴더와 다르게 지정한다. 정확한 full precision 데이터·레이블을 대사하며 PNG/SVG renderer/font 버전 차이는 별도 범위다. 게시 완료 여부는 이 로컬 build가 판단하지 않으며 실제 remote commit·payload 근거는 별도 REMOTE_PROOF.json을 따른다.

CHANNEL_HORIZON_STATS.csv의 6열 통계는 RUNS.csv의 row_id에 조인한다. metadata를 통계 행마다 반복하지 않으며 producer_row_id와 실제 source hash는 RUNS에 남긴다. 과거 선택 결과 {state['contextual_rows']}행과 과거 공통 endpoint {state.get('contextual_endpoint_rows',0)}행은 contextual inventory이고 신규 실험 수가 아니다.

등록 runner의 실제 인터페이스:

```powershell
python -m research.adapter_improvements_20261018_v1.run --help
python -m research.adapter_improvements_20261018_v1.run --mode verify --repo . --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/EXPERIMENT_PLAN.json
```

이번 캠페인에서 사용한 실행 명령은 아래 두 개다. private 경로만 이식 가능한 자리표시자로 표시했다. 첫 명령은 원래 37개 job, 두 번째는 이미 등록된 seed92602의 CPU/GPU 비용 2개 job을 처리한다. 각 명령이 처리한 계획과 실제 과학 arm 완료 수는 다르며, 미수렴·의존성 미실행은 receipt에 보존한다. 명령별 실제 종료코드·환경·검증 범위는 `EXECUTION_VALIDATION.json`을 따른다.

```powershell
python -m research.adapter_improvements_20261018_v1.run --mode execute --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/EXPERIMENT_PLAN.json
python -m research.adapter_improvements_20261018_v1.run --mode execute --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/SUPPLEMENTAL_PLANS/COST_JENA_SEED2_v1.json
```

Python3.11의 기존 프로젝트 가상환경과 실제 기록된 NumPy/OpenBLAS·PyTorch·SciPy 버전을 사용했다. 기본 Anaconda 환경으로 바꾸면 과거 B03 FP32 검산과 달라질 수 있다. CPU BLAS/OMP thread는 2개이며 기존 동결 F snapshot을 offline으로 읽는다. 공개 replay가 private 학습·저장 예측 검산까지 재현했다는 뜻은 아니다. 실제 공개 재생성 결과는 `PUBLIC_REPLAY_VERIFICATION.json`, 시각 확인은 `VISUAL_INSPECTION.json`, 원격 payload 확인은 `REMOTE_PROOF.json`을 별도로 확인한다.

학습/비공개 검산은 동일 private assets·persistent ledger·봉인 plan을 필요로 한다. --only는 기존 등록 job ID만 선택하며 과학 source를 바꿔 같은 identity를 재사용하지 않는다. 인자 없이 runner는 안내만 출력한다. 미완료 명령·의존성·예상시간은 RESUME_QUEUE.json에 있다.

현재 독립 검증 상태는 {state['validation_status']}다. 구현 테스트 통과, 저장 예측 FP64 재검산, whole-path parity, 공개 그림 재생성, main 게시를 각각 구분한다. 과거 test 통과 개수를 이번 새 실험 검증 수로 합치지 않는다.
'''
    privacy='''# 공개·비공개 경계

공개: 코드, aggregate SSE/SAE/count, full precision 지표, relative artifact bindings/SHA, recipe·receipt·ledger·resume queue, 그림 PNG/SVG와 plot source CSV. 공개 sufficient statistics만으로 정량 표·그림을 재생성할 수 있다.

비공개: 원자료 전체, full prediction/target/mask/context 배열, frozen-feature cache, model/optimizer/RNG/sampler checkpoints, 절대 자산·cache 경로, token/credential. 원 데이터 출처가 공개여도 임의 복제본 전체를 Git에 포함하지 않는다.

독립 예측 검산과 replay는 private binding이 필요하다. public-only 보고서 재생성은 모델·학습·대형 다운로드를 수행하지 않는다. F6 qualitative 원 곡선은 private replay 필요 범위이므로 이번 공개 그림에서 생략했다.
'''
    accuracy_rows=presentation_accuracy_rows(tables)
    _write_csv(output/'PPT_T1_ACCURACY_COST.csv',accuracy_rows,['cohort','method','seed','device','batch_size','MSE','MAE',
               'delta_MSE_pct_vs_Raw','delta_MAE_pct_vs_Raw','median_ms_per_origin','p95_ms_per_origin','speedup_vs_Raw',
               'peak_allocated_bytes','peak_reserved_bytes','process_RSS_peak_bytes','target_trained_params','total_resident_params'])
    _write_csv(output/'PPT_T2_SINGLE_CHANGE.csv',matched,COMPARISON_FIELDS)
    t1=[[row['cohort'],row['method'],str(row.get('C',UNKNOWN))+'→'+str(row.get('K',UNKNOWN)),
         row.get('seed',UNKNOWN),str(row.get('device',UNKNOWN))+'/B'+str(row.get('batch_size',UNKNOWN)),
         _format(row.get('MSE')),_format(row.get('MAE')),_format(row.get('delta_MSE_pct_vs_Raw')),
         _format(row.get('delta_MAE_pct_vs_Raw')),_format(row.get('median_ms_per_origin')),
         _format(row.get('p95_ms_per_origin')),_format(row.get('speedup_vs_Raw')),
         _format(row.get('target_trained_params')),_format(row.get('total_resident_params')),
         str(row.get('record_origin',UNKNOWN))+'/'+str(row.get('cost_record_origin',UNKNOWN))+
         (' | '+str(row['comparison_scope']) if _known(row.get('comparison_scope')) else '')]
        for row in accuracy_rows]
    claims=[['Raw 정확도','동일 cohort Raw와 MSE/MAE','부분/미판정','COMPARISONS.csv; 임계값 UNREGISTERED'],
            ['Ridge/LoRA 대비 가치','정확도+실제 비용+동일 정보 계약','미판정','RUNS.csv / COST.csv'],
            ['device/batch 비용','CPU2/GPU B1/B8 whole path parity','부분','COST.csv; UNKNOWN 유지'],
            ['P 분해 기여','P_SPLIT−P_LOCAL_CONTROL 양 seed','부분' if any(r.get('question')=='P' for r in matched) else '미판정','COMPARISONS.csv'],
            ['E 초기화 기여','같은 B2 PCA/DCA 정상 완료','미충족' if nonconverged else '미판정','초기화 미수렴과 예측 우위 구분'],
            ['D 일반화','fixed PE 네 arm, seed별 horizon','부분' if any(r.get('question')=='D' for r in matched) else '미판정','F2 / CHANNEL_HORIZON_STATS.csv'],
            ['다중 K 공유 가치','실제 K 노출/전환/F rows','부분','MULTIK_AUDIT.csv; 누락 UNKNOWN'],
            ['재현성','tests/독립 score/cost/public replay','부분','VALIDATION_REPORT.json / FIGURE_INDEX.json']]
    band_rows=_band_rows(tables)
    band_fields=['cohort','seed','comparison','short_delta_MSE_MAE','middle_delta_MSE_MAE','long_delta_MSE_MAE','whole_delta_MSE_MAE','n_origins']
    _write_csv(output/'PPT_T3_HORIZON_COVERAGE.csv',[dict(zip(band_fields,row)) for row in band_rows],band_fields)
    _write_csv(output/'PPT_T4_CLAIMS.csv',[dict(zip(['claim','required_comparison','status','evidence'],row)) for row in claims],
               ['claim','required_comparison','status','evidence'])
    ppt=f'''# 발표용 표 — 원 수치와 판단 범위

{headline}

표1: 정확도와 실제 비용. 각 cohort/device/batch/accuracy origin/cost origin을 별도 패널로 읽는다. Δ vs Raw와 speedup은 같은 signature 및 hardware/batch가 증명되는 경우에만 COMPARISONS/COST에서 산출하며 현재 표에 없는 값은 UNKNOWN이다. Historical 비용은 독립 contextual measurement 행으로 보존하며 신규 권위 비교나 Raw speedup으로 쓰지 않는다. 정확도 binding이 없는 비용 행의 MSE/MAE는 UNKNOWN이다. G-only 4개 CPU2 비용은 HISTORICAL_CPU2_CONTEXT이며 host UNKNOWN, Jena transitive binding 한계와 p90/p95 구분을 COST.csv에 남긴다.

{_text_table(['Dataset','Method','C→K','Seed','Device/B','MSE↓','MAE↓','ΔMSE% Raw','ΔMAE% Raw','warm ms/origin↓','p95↓','Raw speedup','target params','resident params','accuracy/cost origin'],t1)}

표2: 단일 변경의 효과. selected refit과 common endpoint는 섞지 않는다.

{_text_table(['Dataset','Seed','Candidate−Control','ΔMSE','ΔMAE','ΔMSE%','대응 범위'],effect_rows)}

표3: horizon과 데이터별 적용 범위. 각 칸은 ΔMSE/ΔMAE, 음수 개선. Jena 1–16/17–32/33–48, PeMS32 1–4/5–8/9–12.

{_text_table(['Dataset','Seed','Candidate−Control','Short','Middle','Long','Whole','n_origins'],band_rows)}

표4: 연구 최소 목표 충족 현황. 정확도/속도 수치 목표가 등록되지 않았으므로 관측 개선을 목표 달성 확정으로 쓰지 않는다.

{_text_table(['주장','필요 비교','상태','근거/남은 조건'],claims)}
'''
    for name,text in [('README_KO.md',readme),('RESULT_ANALYSIS_KO.md',analysis),('METHOD_CONTRACT.md',contract),
                      ('REPRODUCE_KO.md',reproduce),('PUBLIC_PRIVATE_ARTIFACTS.md',privacy),('PPT_TABLES_KO.md',ppt)]:
        (output/name).write_text(text,encoding='utf-8')


def build_report(ctx):
    """Build derivative public files; caller schedules this after worker writes stop."""
    tables,state=collect_public(ctx)
    output=Path(ctx['report']);output.mkdir(parents=True,exist_ok=True)
    state['figure_dpi']=int(ctx.get('figure_dpi',300))
    for name,fields in TABLES.items():
        if name=='MULTIK_AUDIT.csv':
            expected=_read(output/'receipts/MULTIK_AUDIT.json',{}).get('artifact_hashes',{}).get(name)
            _write_multik_csv(output/name,tables[name],expected)
        else:_write_csv(output/name,tables[name],fields)
    _json(output/'REPORT_STATE.json',state)
    write_documents(tables,state,output)
    render_figures(figure_sources(tables),output,state['figure_dpi'])
    return state


def regenerate(public_input, output):
    """Replay only public tables/plot sources into a separate output directory."""
    source=Path(public_input).resolve();destination=Path(output).resolve()
    if source==destination:
        raise ValueError('Public replay output must differ from input')
    destination.mkdir(parents=True,exist_ok=True)
    state=_read(source/'REPORT_STATE.json',{})
    if not state:
        raise ValueError('Public source REPORT_STATE.json is required')
    tables={name:_csv(source/name) for name in TABLES}
    for name in TABLES:
        if (source/name).is_file():shutil.copyfile(source/name,destination/name)
        else:_write_csv(destination/name,[],TABLES[name])
    _json(destination/'REPORT_STATE.json',state)
    sources={f'F{i}':_csv(source/'figures'/f'F{i}_source.csv') for i in range(1,6)}
    write_documents(tables,state,destination)
    render_figures(sources,destination,state.get('figure_dpi',300))
    return dict(status='COMPLETE',output=str(destination),private_arrays_opened=0,foundation_calls=0)


def verify(ctx):
    from .verification import verify as independent_verify
    return independent_verify(ctx)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Regenerate report tables/figures from public sufficient statistics only')
    parser.add_argument('--public-input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(regenerate(args.public_input,args.output),ensure_ascii=False))


if __name__=='__main__':main()
