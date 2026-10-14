"""Independent guarded report file/link, saved-array score and CSV verification.

No model, solver, encoder/backbone/decoder replay or timing replay. The CPU
resource gate precedes all input/output/scientific array access.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATASETS=('robin','beijing_aotizhongxin')
FINAL={'robin':'val','beijing_aotizhongxin':'eval'}
METHODS=('RAW_BOLT','ORIGINAL_MAIN','MEAN_CORRECTION','RAW33C','E33K','E512K')
ROUTES=METHODS[3:]
IDS=('ZERO_CORRECTION','MEAN_CORRECTION')+tuple('RIDGE_ALPHA_'+str(i) for i in range(7))
ALPHAS=(None,None,0.,1e-4,.001,.01,.1,1.,10.)


def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle: return list(csv.DictReader(handle))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def near(a,b):
    return a is None and b is None or a is not None and b is not None and math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-12)


def expected_score(p,method,split):
    if method in METHODS[:2]: return p['references'][split][method]
    if method=='MEAN_CORRECTION': return p['controls'][method]['splits'][split]
    return p['families'][method]['splits'][split]


def observed_score(prediction,target,observed,std,np):
    error=np.asarray(prediction,np.float64)-np.asarray(target,np.float64)
    if std is not None: error=error*std
    error=np.where(observed,error,0.)
    count=observed.sum(axis=(0,1)); active=count>0
    mse=np.divide((error**2).sum((0,1)),count,out=np.zeros(len(count)),where=active)
    mae=np.divide(np.abs(error).sum((0,1)),count,out=np.zeros(len(count)),where=active)
    return {'mse':float(mse[active].mean()) if active.any() else None,
        'mae':float(mae[active].mean()) if active.any() else None,
        'active_channels':int(active.sum()),'observed_counts':count.astype(int).tolist(),
        'per_channel_mse':[float(v) if a else None for v,a in zip(mse,active)],
        'per_channel_mae':[float(v) if a else None for v,a in zip(mae,active)]}


def execute(args):
    import main_residual_runtime_v1 as runtime
    limit=runtime.ContinuationLimit('main_residual_report_verification_v1',gpu=False)
    report=resolve(args.report); output=resolve(args.out); full=limit.checkfull
    result={'status':'RUNNING','checks':0,'failures':[],'independent_E_F_D_or_solver_replay':False,
        'independent_timing_replay':False,'stored_prediction_score_recomputed':True,
        'authorization':limit.authorization_record,'run_manifest':limit.run_manifest_record}
    def check(passed,label):
        result['checks']+=1
        if not passed: result['failures'].append(label)
        limit.shared.checkfast()
    try:
        full()
        if output.exists(): raise FileExistsError(output)
        manifest=read(report/'MANIFEST.json'); terminal=read(report/'RESULT.json')
        check(manifest['status']=='COMPLETE' and terminal['status']=='COMPLETE' and manifest['audit_gate']['passed'],'TERMINAL_COMPLETE_GATE')
        check(terminal['manifest']['sha256']==runtime.sha(report/'MANIFEST.json'),'TERMINAL_MANIFEST_SHA')
        for i,r in enumerate(manifest['artifact_records']):
            if i%16==0: full()
            path=(report/r['path']).resolve()
            check(path.is_relative_to(report) and path.is_file() and path.stat().st_size==r['bytes'] and runtime.sha(path)==r['sha256'],'ARTIFACT:'+r['path'])
        for r in manifest['input_records']:
            check(runtime.sha(resolve(r['path']))==r['sha256'] and runtime.sha(report/r['copy_relative'])==r['sha256'],'INPUT_COPY:'+r['copy_relative'])
        for document in report.rglob('*.md'):
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',document.read_text(encoding='utf-8-sig')):
                if target.startswith(('https://','http://','#')): continue
                path=(document.parent/unquote(target.strip('<>').split('#',1)[0])).resolve()
                check(path.is_relative_to(ROOT) and path.exists() and '.cache' not in path.relative_to(ROOT).parts,'MD_LINK:'+str(document.relative_to(report))+':'+target)
        pngs=sorted((report/'figures').glob('*.png')); svgs=sorted((report/'figures').glob('*.svg'))
        check(len(pngs)==6 and len(svgs)==6 and {p.stem for p in pngs}=={p.stem for p in svgs},'SIX_PNG_SVG_PAIRS')
        from PIL import Image
        for path in pngs:
            with Image.open(path) as im: check(im.width>=1200 and im.height>=800,'IMAGE_DIMENSIONS:'+path.name)
        for path in svgs: check('<svg' in path.read_text(encoding='utf-8')[:2000],'SVG_HEADER:'+path.name)
        producers={}
        for r in manifest['input_records']:
            if r.get('role')=='producer':
                p=read(report/r['copy_relative']); producers[p['dataset'],p['seed']]=p
        expected={(d,s) for d in DATASETS for s in (92601,92602)}
        check(set(producers)==expected,'EXACT_FOUR_PRODUCERS')
        audit=read(report/'provenance/inputs/INDEPENDENT_AUDIT.json')
        check(audit['status']=='PASS' and audit['failed_count']==0 and audit['candidate_rows_checked']==108 and audit['new_ridge_candidates_checked']==84,'SCIENTIFIC_AUDIT_COUNTS')
        for r in manifest['input_records']:
            if r.get('role')=='producer':
                p=read(report/r['copy_relative']); matches=[c for c in audit['conditions'] if (c['dataset'],c['seed'])==(p['dataset'],p['seed'])]
                check(len(matches)==1 and matches[0]['producer_sha256']==r['sha256'] and matches[0]['status']=='PASS','AUDIT_PRODUCER_BINDING:'+p['dataset']+str(p['seed']))
        metrics=read_csv(report/'METRICS.csv'); candidates=read_csv(report/'CANDIDATE_GRID.csv')
        capacity=read_csv(report/'CAPACITY_SELECTION.csv'); cpu=read_csv(report/'CPU_COST.csv'); whole=read_csv(report/'WHOLE_COST.csv')
        for rows,number,name in ((metrics,144,'METRICS'),(candidates,108,'CANDIDATES'),(capacity,12,'CAPACITY'),(cpu,12,'CPU_COST'),(whole,24,'WHOLE_COST')):
            check(len(rows)==number,'ROW_COUNT:'+name)
        metric_keys=[]
        for row in metrics:
            key=(row['dataset'],int(row['seed']),row['method'],row['split'],row['metric_scale']); metric_keys.append(key)
            source=expected_score(producers[key[:2]],key[2],key[3])[key[4]]
            for field in ('mse','mae'): check(near(float(row[field]) if row[field] else None,source[field]),'JSON_CSV:'+str(key)+field)
            check(int(row['active_channels'])==source['active_channels'],'JSON_CSV_ACTIVE:'+str(key))
            for field in ('observed_counts','per_channel_mse','per_channel_mae'): check(json.loads(row[field])==source[field],'JSON_CSV_CHANNEL:'+str(key)+field)
        check(len(set(metric_keys))==144,'UNIQUE_METRICS')
        import numpy as np
        cache={}
        def load_npz(r):
            full(); path=resolve(r['path']); key=str(path)
            if key in cache: return cache[key]
            if runtime.sha(path)!=r['sha256']: raise ValueError('SCIENTIFIC_NPZ_FILE_SHA:'+key)
            with np.load(path,allow_pickle=False) as opened: arrays={n:opened[n].copy() for n in opened.files}
            declared=r.get('arrays')
            if declared is not None:
                check(set(arrays)==set(declared),'NPZ_KEYS:'+key)
                for name,value in arrays.items():
                    meta=declared[name]; digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    check(list(value.shape)==meta['shape'] and str(value.dtype)==meta['dtype'] and digest==meta['sha256'],'NPZ_RAWBYTE_ARRAY_SHA:'+key+name)
            cache[key]=arrays; return arrays
        def exact(a,b,label):
            check(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True),'EXACT_ARRAY:'+label)
        by_key={(r['dataset'],int(r['seed']),r['method'],r['split'],r['metric_scale']):r for r in metrics}
        for (d,seed),p in producers.items():
            full()
            for split in ('fit','tune',FINAL[d]):
                raw=load_npz(p['input_artifact_refs'][split]['raw'])
                for method in METHODS:
                    if method=='RAW_BOLT': prediction=raw['raw_forecast']
                    else:
                        r=(p['controls']['ZERO_CORRECTION' if method=='ORIGINAL_MAIN' else method]['artifacts'][split]
                           if method in ('ORIGINAL_MAIN','MEAN_CORRECTION') else p['families'][method]['selected_artifacts'][split])
                        arrays=load_npz(r); prediction=arrays['prediction']; exact(arrays['origins'],raw['origins'],d+str(seed)+split+method+'origins')
                        if method in ROUTES:
                            for field in ('target','observed','mean','std'): exact(arrays[field],raw[field],d+str(seed)+split+method+field)
                            zero=load_npz(p['controls']['ZERO_CORRECTION']['artifacts'][split])
                            exact(arrays['original_MAIN_prediction'],zero['prediction'],d+str(seed)+split+method+'MAIN')
                    check(prediction.shape==raw['target'].shape and np.isfinite(prediction[raw['observed']]).all(),'PREDICTION_OBSERVED_FINITE:'+d+str(seed)+split+method)
                    for scale in ('normalized','physical_per_channel_diagnostic'):
                        values=observed_score(prediction,raw['target'],raw['observed'],raw['std'] if scale!='normalized' else None,np)
                        row=by_key[d,seed,method,split,scale]
                        for field in ('mse','mae'): check(near(float(row[field]) if row[field] else None,values[field]),'ARRAY_CSV:'+str((d,seed,method,split,scale))+field)
                        check(int(row['active_channels'])==values['active_channels'],'ARRAY_CSV_ACTIVE:'+d+str(seed)+method+split+scale)
                        check(json.loads(row['observed_counts'])==values['observed_counts'],'ARRAY_CSV_COUNTS:'+d+str(seed)+method+split+scale)
                        for field in ('per_channel_mse','per_channel_mae'):
                            saved=json.loads(row[field]); check(len(saved)==len(values[field]) and all(near(a,b) for a,b in zip(saved,values[field])),'ARRAY_CSV_CHANNEL:'+d+str(seed)+method+split+scale+field)
        candidate_keys=[]
        for row in candidates:
            key=(row['dataset'],int(row['seed']),row['method'],int(row['candidate_index'])); candidate_keys.append(key)
            f=producers[key[:2]]['families'][key[2]]; e=f['candidate_grid'][key[3]]
            check(row['candidate_id']==e['candidate_id']==IDS[key[3]] and (None if row['alpha']=='' else float(row['alpha']))==ALPHAS[key[3]],'CANDIDATE_ID_ALPHA:'+str(key))
            check(row['model_sha256']==e['model_sha256'] and row['status']==e['status'],'CANDIDATE_MODEL_STATUS:'+str(key))
            check((row['selected']=='True')==(key[3]==f['selected_index']),'CANDIDATE_SELECTED:'+str(key))
            for split in ('fit','tune'):
                for field in ('mse','mae'): check(near(row[split+'_normalized_'+field],e[split+'_scores']['normalized'][field]),'CANDIDATE_SCORE:'+str(key)+split+field)
        check(len(set(candidate_keys))==108,'UNIQUE_108_CANDIDATES')
        for (d,seed),p in producers.items():
            for route in ROUTES:
                f=p['families'][route]; g=f['candidate_grid']; ix=min(range(9),key=lambda i:g[i]['tune_actual_Y_mse'])
                check(ix==f['selected_index'] and tuple(e['candidate_id'] for e in g)==IDS,'EARLIEST_ACTUAL_Y_SELECTION:'+d+str(seed)+route)
                row=next(r for r in capacity if (r['dataset'],int(r['seed']),r['method'])==(d,seed,route))
                check(row['selected_candidate_id']==f['selected_candidate_id'] and row['selected_model_sha256']==f['selected_model_sha256'],'CAPACITY_SELECTION:'+d+str(seed)+route)
                tune=expected_score(p,route,'tune')['normalized']['mse']
                for baseline,col in (('ORIGINAL_MAIN','ZERO'),('MEAN_CORRECTION','MEAN')):
                    check(near(row['tune_mse_margin_vs_'+col],expected_score(p,baseline,'tune')['normalized']['mse']-tune),'TUNE_MARGIN:'+d+str(seed)+route+col)
                final=expected_score(p,route,FINAL[d])['normalized']
                for baseline,col in (('ORIGINAL_MAIN','MAIN'),('MEAN_CORRECTION','MEAN'),('RAW_BOLT','RAW')):
                    base=expected_score(p,baseline,FINAL[d])['normalized']
                    for v in ('mse','mae'): check(near(row['final_'+v+'_delta_vs_'+col],final[v]-base[v]),'FINAL_DELTA:'+d+str(seed)+route+col+v)
        for d,methods in manifest['summary'].items():
            for method,values in methods.items():
                rs=[r for r in metrics if r['dataset']==d and r['method']==method and r['split']==FINAL[d] and r['metric_scale']=='normalized']
                check(len(rs)==2,'TWO_SEED_SUMMARY:'+d+method)
                for v in ('mse','mae'): check(near(statistics.mean(float(r[v]) for r in rs),values[v]),'SUMMARY:'+d+method+v)
        cost=read(report/'provenance/inputs/WHOLE_COST.json'); cost_audit=read(report/'provenance/inputs/COST_AUDIT.json')
        check(cost['status']=='COMPLETE' and cost_audit['status']=='PASS' and cost_audit['failed_count']==0,'COST_TERMINAL_AUDIT')
        costrecord=next(r for r in manifest['input_records'] if r.get('role')=='whole_cost')
        bound=cost_audit.get('producer') or cost_audit.get('cost_producer') or cost_audit.get('cost_result')
        boundsha=(bound or {}).get('sha256') or cost_audit.get('producer_sha256') or cost_audit.get('cost_producer_sha256')
        check(boundsha==costrecord['sha256'],'COST_AUDIT_EXACT_RESULT_BINDING')
        cost_keys=[]
        for row in whole:
            method='MEAN_CONTROL' if row['method']=='MEAN_CORRECTION' else row['method']
            key=(row['dataset'],int(row['seed']),int(row['batch']),method); cost_keys.append(key)
            r=next(r for r in cost['rows'] if (r['dataset'],r['seed'],r['batch'],r['route'])==key)
            check(len(r['seconds'])==6 and near(row['median_ms'],1000*statistics.median(r['seconds'])) and near(row['median_ms_per_window'],1000*statistics.median(r['seconds'])/r['batch']),'COST_MEDIAN:'+str(key))
            check(near(row['paired_RAW_median_ratio'],statistics.median(r['paired_RAW_ratios'])),'COST_PAIRED_RATIO:'+str(key))
            for field in ('seconds','paired_RAW_seconds','paired_RAW_ratios','block','position'): check(json.loads(row[field])==r[field],'COST_SAMPLES:'+str(key)+field)
            check(row['selected_candidate_id']==str(r.get('selected_candidate_id') or '') and row['status']==r['status'],'COST_SELECTION_STATUS:'+str(key))
        check(len(set(cost_keys))==24,'UNIQUE_24_COST_CELLS')
        for row in cpu:
            p=producers[row['dataset'],int(row['seed'])]; f=p['families'][row['method']]
            for field in ('FIT_feature_state_wall_seconds','new_FIT_factorization_wall_seconds'):
                check(near(row[field],f[field]),'CPU_JSON_CSV:'+row['dataset']+row['method']+field)
            receipt=read(resolve(p['resource_receipt']['path']))
            check(near(row['producer_process_wall_seconds'],receipt['wall_seconds']),'CPU_RECEIPT:'+row['dataset']+str(p['seed'])+row['method'])
        result.update(status='PASS' if not result['failures'] else 'FAILED_PRESERVED',failed_count=len(result['failures']),
            manifest_sha256=runtime.sha(report/'MANIFEST.json'),metrics_rows=144,candidate_rows=108,prediction_groups_scored=72,
            selected_route_prediction_arrays_scored=36,
            whole_cost_rows=24,normalized_two_seed_summary=manifest['summary'],image_paths=[str(p) for p in pngs])
        full()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error,runtime.BudgetHeld) else 'FAILED_PRESERVED',
            failure={'type':type(error).__name__,'message':str(error)},failed_count=len(result['failures']))
    finally:
        receipt=limit.close('COMPLETE' if result['status']=='PASS' else result['status'],result.get('failure')); result['resource_receipt']=receipt
        if receipt.get('terminal_status')!='COMPLETE': result.update(status=receipt['terminal_status'],resource_terminal_error=receipt.get('terminal_error'))
        runtime.write_new(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--report',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(); result=execute(args)
    print(json.dumps({k:result.get(k) for k in ('status','checks','failed_count','failure')})); return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
