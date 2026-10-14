"""Independent saved cost-array/hash/arithmetic audit; no E/F/timing replay.

Source-only by default. Actual CPU reads require root GO after cost terminal
and the same fresh stage authorization/manifest used by the cost producer.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[1]
if str(STUDY) not in sys.path:sys.path.insert(0,str(STUDY))
DATASETS=('robin','beijing_aotizhongxin')
SEEDS=(92601,92602)
FAMILIES=('N1','N2','N3')
ROUTES=('RAW_BOLT','MAIN_s92601','MAIN_s92602','N1_s92601','N2_s92601','N3_s92601','N1_s92602','N2_s92602','N3_s92602')
ORDER_SEED=92725
REPEATS=9
F_HASHES={'old_algorithm':'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874',
          'bi_algorithm':'e0e0ab8b3e15a2f965a6c1f6692c49122921253da24cc6cceb457d8a038afd7f'}
DECODER_KEYS=('codec.decode.weight','codec.decode.bias','residual.0.weight','residual.0.bias','residual.2.weight','residual.2.bias')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def resolve(path):
    value=Path(path);value=value if value.is_absolute() else ROOT/value
    if any(part.lower()=='test' for part in value.parts):raise PermissionError('PROTECTED_TEST_PATH')
    return value


def checked(record):
    path=resolve(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('SOURCE_SHA_MISMATCH:'+str(path))
    return path


def json_record(record):
    path=checked(record);value=json.loads(path.read_text(encoding='utf-8-sig'));checked(record);return value


def arrays_record(record):
    path=checked(record)
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    checked(record);return arrays


def array_sha(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()


def state_sha(kind,*values):
    h=hashlib.sha256(kind.encode())
    for v in values:h.update((array_sha(v) if isinstance(v,np.ndarray) else str(v)).encode())
    return h.hexdigest()


def coefficient_hashes(producer,family,feature_arrays,coefficient_arrays):
    feature=producer['feature_state'];row=producer['families'][family];meta=row['selected_model_metadata'];s=meta['solution']
    feature_sha=state_sha('FIT_33K_FEATURE_STATE',feature_arrays['center'],feature_arrays['scale'],feature['latent_channels'],
                          feature['fit_rows'],feature['fit_raw_features_sha256'])
    solution_sha=state_sha(s['kind'],s['alpha'],coefficient_arrays['coefficients'],coefficient_arrays['intercept'],
                           coefficient_arrays['observed_counts'],coefficient_arrays['channel_counts'],s['feature_state_sha256'])
    if family!='N3':model_sha=state_sha('FIXED_NATIVE_MODEL',feature_sha,solution_sha)
    else:model_sha=state_sha(meta['kind'],meta['alpha'],coefficient_arrays['effective_weight'],coefficient_arrays['feature_rms'],
                            coefficient_arrays['hidden_weight'],coefficient_arrays['hidden_bias'],meta['fixed_n1_model_sha256'])
    return feature_sha,solution_sha,model_sha


def tensor_state_hashes(state):
    old=hashlib.sha256();bi=hashlib.sha256()
    for key,value in state.items():old.update(key.encode());old.update(value.detach().cpu().contiguous().numpy().tobytes())
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy();bi.update(key.encode());bi.update(str(a.dtype).encode());bi.update(str(a.shape).encode());bi.update(a.tobytes())
    return {'old_algorithm':old.hexdigest(),'bi_algorithm':bi.hexdigest()}


def close_arrays(left,right,scale):
    a,b=np.asarray(left),np.asarray(right);atol,rtol=(1e-5,2e-6) if scale=='normalized' else (2e-3,2e-5)
    return (a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and bool(np.all(np.abs(a.astype(np.float64)-b.astype(np.float64))<=atol+rtol*np.abs(b.astype(np.float64)))))


def masked_score(prediction,target,observed,std=None):
    p,y,m=np.asarray(prediction,np.float64),np.asarray(target,np.float64),np.asarray(observed)
    if p.shape!=y.shape or m.shape!=y.shape or m.dtype!=np.bool_:raise ValueError('METRIC_TARGET_MASK_SHAPE')
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():raise ValueError('METRIC_OBSERVED_NONFINITE')
    error=np.where(m,p-y,0.)
    if std is not None:error=error*np.asarray(std,np.float64)
    count=m.sum((0,1));active=count>0
    if not active.any():return {'mse':None,'mae':None,'active_channels':0,'observed_count':0,'per_channel_counts':count.tolist()}
    mse=(error**2).sum((0,1))[active]/count[active];mae=np.abs(error).sum((0,1))[active]/count[active]
    return {'mse':float(mse.mean()),'mae':float(mae.mean()),'active_channels':int(active.sum()),
            'observed_count':int(count.sum()),'per_channel_counts':count.tolist()}


def expected_schedule():
    order=np.random.default_rng(ORDER_SEED).permutation(9)
    return [[ROUTES[i] for i in np.roll(order,-block)] for block in range(9)]


def timing_checks(row,raw,check,label):
    seconds=np.asarray(row['seconds'],np.float64)
    check(label+':warm2_repeat9',row['status']=='COMPLETE' and row['warmup_each_route']==2 and row['repeats']==9)
    check(label+':seconds9_positive',seconds.shape==(9,) and np.isfinite(seconds).all() and (seconds>0).all())
    check(label+':blocks9',row['block']==list(range(9)))
    check(label+':all_positions',sorted(row['position'])==list(range(9)))
    check(label+':median',np.isclose(np.median(seconds),row['median_seconds'],atol=1e-14,rtol=1e-12))
    check(label+':perwindow',np.isclose(np.median(seconds)/row['batch'],row['median_seconds_per_window'],atol=1e-14,rtol=1e-12))
    raw_time=dict(zip(raw['block'],raw['seconds']));paired=np.asarray([raw_time[b] for b in row['block']])
    ratios=seconds/paired
    check(label+':paired_raw',np.array_equal(row['paired_RAW_seconds'],paired))
    check(label+':ratios',np.allclose(row['paired_RAW_ratios'],ratios,atol=1e-14,rtol=1e-12))
    check(label+':ratio_median',np.isclose(row['paired_RAW_median_ratio'],np.median(ratios),atol=1e-14,rtol=1e-12))
    route_position=[order.index(row['route']) for order in expected_schedule()]
    check(label+':fixed_order92725',row['position']==route_position)


def audit(args):
    import stage_separation_runtime_v1 as runtime
    output=resolve(args.out);csv_path=output.with_name(output.stem+'_metrics.csv')
    if output.exists() or csv_path.exists():raise FileExistsError('FRESH_OUTPUTS_REQUIRED')
    result={'status':'RUNNING','checks':0,'failed_count':0,'failures':[],'whole_rows_checked':0,'metric_rows':0,
        'GPU_execution':False,'E_forward_replayed':False,'F_forward_replayed':False,'timing_replayed':False,
        'whole_goal_complete':False,'independent_generalization':False,
        'scope':'saved9 outputs/references, first-batch masked metrics, CP/coefficient hashes, Latin schedule and timing arithmetic',
        'GPU_state_scope':'producer before/after hash consistency bound to original CP/coefficient files; no independent live GPU state or F replay',
        'metric_scope':'each repeat uses the same first1/8 exposed windows; repetitions are not independent evaluation samples',
        'physical_scope':'physical-GPU errors use inverse-reconstructed normalized targets; original physical future is not reread'}
    limit=None;metrics=[];file_cache={};array_cache={}
    def check(label,value):
        result['checks']+=1
        if not bool(value):result['failed_count']+=1;result['failures'].append(label)
    def value(record):
        key=(str(resolve(record['path'])),record['sha256'])
        if key not in file_cache:file_cache[key]=json_record(record)
        return file_cache[key]
    def arrays(record):
        key=(str(resolve(record['path'])),record['sha256'])
        if key not in array_cache:array_cache[key]=arrays_record(record)
        return array_cache[key]
    try:
        limit=runtime.ContinuationLimit('stage_native_cost_audit_'+args.tag,gpu=False)
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        cost_path=resolve(args.cost_result);cost_record={'path':str(cost_path),'sha256':sha(cost_path)};cost=value(cost_record)
        result['cost_producer']=cost_record
        check('producer_COMPLETE',cost['status']=='COMPLETE')
        check('F_dualhash_preserved',cost['F_before']==cost['F_after']==F_HASHES)
        check('F_no_parameter_gradients',cost['F_has_no_parameter_gradients'] is True)
        check('models_preserved',cost['model_hashes_before']==cost['model_hashes_after'])
        check('no_oracles_or_refit',cost['oracles_in_deployment'] is False and cost['coefficient_refit_in_cost'] is False)
        check('protocol_92725_9',cost['order_seed']==92725 and cost['repeats']==9 and cost['warmup_each_route']==2)
        check('local180_cap',cost['local_process_cap_seconds']==180)
        check('same_auth',cost['authorization_record']['sha256']==limit.authorization_record['sha256'])
        check('same_manifest',cost['run_manifest_record']['sha256']==limit.run_manifest_record['sha256'])
        receipt=value(cost['shared_resource_receipt'])
        check('GPU_receipt_COMPLETE180',receipt['status']=='COMPLETE' and receipt['gpu'] is True and receipt['wall_seconds']<=180)
        check('receipt_auth_binding',receipt['authorization']['sha256']==cost['authorization_record']['sha256'])
        check('receipt_manifest_binding',receipt['run_manifest']['sha256']==cost['run_manifest_record']['sha256'])
        for r in cost['source_records_before']+cost.get('source_records_after',[]):
            limit.shared.checkfast();checked(r)
            if 'snapshot_path' in r:check('snapshot:'+r['snapshot_path'],sha(resolve(r['snapshot_path']))==r['sha256'])
        native_audit=value(cost['native_audit']);check('native_audit_PASS',str(native_audit['status']).startswith('PASS') and native_audit['failed_count']==0)
        native={};expected_native={(d,s) for d in DATASETS for s in SEEDS}
        import torch
        for record in cost['native_producers']:
            limit.shared.checkfast();p=value(record);key=(p['dataset'],p['seed']);check(str(key)+':unique',key not in native)
            native[key]=(p,record);label=str(key);name=str(key)
            check(label+':CPU_producer_COMPLETE',p['status']=='COMPLETE' and p['old_E_D_F_updated'] is False)
            check(label+':native_audit_sha',any(r['dataset']==key[0] and r['seed']==key[1] and r['producer_sha256']==record['sha256'] for r in native_audit['conditions']))
            cp=p['MAIN_checkpoint']['checkpoint'];cp_state=torch.load(checked(cp),map_location='cpu',weights_only=True)['codec'];checked(cp)
            hashes=tensor_state_hashes(cp_state)
            check(label+':independent_CP_state',hashes==p['codec_before']==cost['model_hashes_before']['codec'][name])
            feature=arrays(p['feature_state']['artifact']);decoder=arrays(p['source_decoder_artifact']);decoder_hash=hashlib.sha256()
            for decoder_key in DECODER_KEYS:
                check(label+':CPU_D_'+decoder_key,np.array_equal(decoder[decoder_key],cp_state[decoder_key].numpy()))
                decoder_hash.update(decoder_key.encode());decoder_hash.update(np.ascontiguousarray(decoder[decoder_key]).tobytes())
            check(label+':original_CPU_D_sha',decoder_hash.hexdigest()==p['source_decoder_state_sha256']==cost['model_hashes_before']['native'][name]['original_CPU_D'])
            for family in FAMILIES:
                row=p['families'][family];coeff=arrays(row['selected_coefficient_artifact'])
                fsha,ssha,msha=coefficient_hashes(p,family,feature,coeff)
                check(label+':'+family+':feature_sha',fsha==p['feature_state']['sha256'])
                check(label+':'+family+':solution_sha',ssha==row['selected_model_metadata']['solution']['solution_sha256'])
                check(label+':'+family+':model_sha',msha==row['selected_model_sha256']==cost['model_hashes_before']['native'][name][family])
        check('exact4_native',len(cost['native_producers'])==4 and set(native)==expected_native)
        audit_keys={(r['dataset'],r['seed']) for r in native_audit['conditions']}
        check('native_audit_exact4',len(native_audit['conditions'])==4 and audit_keys==expected_native)
        expected={(d,b,r) for d in DATASETS for b in (1,8) for r in ROUTES}
        whole={(r['dataset'],r['batch'],r['route']):r for r in cost['rows']}
        check('whole_exact36',len(cost['rows'])==36 and set(whole)==expected and cost['actual_complete_rows']==36)
        for key,row in whole.items():
            limit.shared.checkfast();label=str(key);d,b,route=key;split='val' if d=='robin' else 'eval'
            raw=arrays(cost['input_provenance'][d]['physical_source']);timed=arrays(row['timed_outputs']);ref=arrays(row['source_reference'])
            bulk=arrays(row['original_bulk_source']);bp=bulk['prediction'] if 'prediction' in bulk else bulk[split+'_prediction']
            y,mask,mean,std=raw['target'][:b],raw['observed'][:b],raw['mean'],raw['std']
            for field in ('mean','std'):check(label+':bulk_'+field,np.array_equal(bulk[field],raw[field]))
            for field in ('target','observed','origins'):
                bulk_field=field if 'prediction' in bulk else split+'_'+field
                check(label+':bulk_'+field,bulk[bulk_field].dtype==raw[field].dtype and np.array_equal(bulk[bulk_field],raw[field],equal_nan=True))
            if route=='RAW_BOLT':check(label+':RAW_shared_seed',row['seed'] is None and row['method']=='RAW_BOLT')
            elif route.startswith(('N1_','N2_','N3_')):
                family=route.split('_')[0];p=native[d,row['seed']][0]
                check(label+':selected_alpha',row['selected_alpha']==p['families'][family]['selected_alpha'])
                check(label+':selected_model_binding',row['selected_model_sha256']==p['families'][family]['selected_model_sha256'])
            check(label+':origins',row['origins']==raw['origins'][:b].tolist())
            check(label+':reference_route',row['source_reference']['route']==route)
            check(label+':timed_blocks',np.array_equal(timed['block'],row['block']))
            check(label+':timed_positions',np.array_equal(timed['position'],row['position']))
            check(label+':norm_copy_outside',row['normalized_D2H_copy_outside_timer'] is True)
            timing_checks(row,whole[d,b,'RAW_BOLT'],check,label)
            for scale in ('normalized','physical'):
                check(label+':'+scale+'_shape',timed[scale].shape==(9,b,48,y.shape[2]) and ref[scale].shape==(b,48,y.shape[2]))
                check(label+':'+scale+'_FP32',timed[scale].dtype==np.float32 and ref[scale].dtype==np.float32)
            for repeat in range(9):
                for scale in ('normalized','physical'):
                    passed=close_arrays(timed[scale][repeat],ref[scale],scale)
                    check(label+':'+scale+':repeat'+str(repeat),passed)
                    check(label+':declared_'+scale+':repeat'+str(repeat),row['parity'][repeat]['matched_batch_'+scale]['pass']==passed)
                    original=bp[:b] if scale=='normalized' else bp[:b]*std+mean
                    bulk_pass=close_arrays(timed[scale][repeat],original,scale)
                    check(label+':bulk_flag_'+scale+':repeat'+str(repeat),row['parity'][repeat]['original_bulk_'+scale]['pass']==bulk_pass)
                check(label+':inverse_repeat'+str(repeat),close_arrays(timed['physical'][repeat],timed['normalized'][repeat]*std+mean,'physical'))
                for scale,prediction,target,score_std in (
                    ('normalized',timed['normalized'][repeat],y,None),
                    ('physical_from_normalized_error',timed['normalized'][repeat],y,std),
                    ('physical_GPU_inverse_target',timed['physical'][repeat],y.astype(np.float64)*std+mean,None)):
                    score=masked_score(prediction,target,mask,score_std)
                    metrics.append({'dataset':d,'batch':b,'route':route,'repeat':repeat,'scale':scale,**score,
                        'target_window_scope':'same first batch; not independent repeats','sample_count':b})
            result['whole_rows_checked']+=1
        for d in DATASETS:
            check(d+':RAW_shared_source',whole[d,1,'RAW_BOLT']['original_bulk_source']['sha256']==whole[d,8,'RAW_BOLT']['original_bulk_source']['sha256'])
            for b in (1,8):
                for block in range(9):check(f'{d}:{b}:block{block}:position_unique',sorted(whole[d,b,r]['position'][block] for r in ROUTES)==list(range(9)))
        for path,digest in file_cache:check('source_unchanged:'+path,sha(path)==digest)
        for path,digest in array_cache:check('array_unchanged:'+path,sha(path)==digest)
        limit.checkfull();result['status']='PASS_WITHOUT_E_F_OR_TIMING_REPLAY' if result['failed_count']==0 else 'FAIL_PRESERVED'
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      error={'type':type(error).__name__,'message':str(error)})
    finally:
        if metrics:
            with csv_path.open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader()
                for row in metrics:writer.writerow({**row,'per_channel_counts':json.dumps(row['per_channel_counts'])})
            result['metric_rows']=len(metrics);result['metrics_csv']={'path':str(csv_path),'sha256':sha(csv_path)}
        if limit is not None:
            try:
                terminal='COMPLETE' if result['status'].startswith('PASS') else ('BUDGET_HELD_PRESERVED' if result['status']=='BUDGET_HELD_PRESERVED' else 'FAILED_PRESERVED')
                receipt=limit.close(terminal,result.get('error'));result['shared_resource_receipt']=receipt
                if receipt.get('terminal_status')!='COMPLETE':result['status']=receipt['terminal_status'];result['terminal_error']=receipt.get('terminal_error')
            except Exception as error:result.update(status='FAILED_PRESERVED',resource_close_error={'type':type(error).__name__,'message':str(error)})
        else:result['constructor_terminal']=getattr(runtime,'LAST_CONSTRUCTOR_TERMINAL',None)
        result['audit_source_sha256']=sha(__file__)
        with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('cost-result','out','tag'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.describe:
        print(json.dumps({'status':'SOURCE_ONLY_NOT_EXECUTED','planned_whole_rows':36,'saved_repeats':9,
            'CP_read_scope':'CPU original state hash only','E_F_or_timing_replay':False,'metric_scope':'same first1/8 exposed windows'}));return
    if not all((args.cost_result,args.out,args.tag)) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):parser.error('execute requires actual terminal cost/fresh output/tag')
    result=audit(args);print(json.dumps({k:result[k] for k in ('status','checks','failed_count','whole_rows_checked','metric_rows')}))
    if not result['status'].startswith('PASS'):raise SystemExit(1)


if __name__=='__main__':main()
