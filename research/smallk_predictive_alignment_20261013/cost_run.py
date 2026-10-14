"""Clean deployment measurement after all core/native training has terminated."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import traceback
import numpy as np
import engine as en
import controls
import newdata
import newdata_repair


def new_past_data(dataset,methodpath):
    if dataset=='beijing_aotizhongxin':
        data=newdata_repair.load_training(dataset,methodpath)
        past=newdata_repair.load_evaluation_inputs(dataset,methodpath)
        data['origins']['eval']=past['origins']
        data['evaluation_history_physical']=past['history_physical']
        data['evaluation_history_observed']=past['history_observed']
        return data
    data=newdata.load_training(dataset,methodpath)
    contract=newdata._contract_from_registry(dataset,methodpath,en.CACHE,include_eval_target=False)
    artifact=contract['payloads']['eval_past_input']
    assert en.ex.sha(Path(artifact['path']))==artifact['sha256']
    with np.load(artifact['path'],allow_pickle=False) as z:
        data['origins']['eval']=z['origins'].astype(np.int64)
        data['evaluation_history_physical']=z['physical_history'].astype(np.float32)
        data['evaluation_history_observed']=z['history_mask'].astype(bool)
    return data


def run(tag,devtag,newtag):
    en.configure(); protected=en.ex.access_guard()
    output=en.HERE/('run_'+tag); output.mkdir(exist_ok=False)
    limit=en.Limit(tag)
    result={'status':'RUNNING','records':[],'new_training_updates':0,'protected_test_guarded':protected,
        'source_hashes':{str(p.relative_to(en.ROOT)):en.ex.sha(p) for p in (Path(__file__),en.HERE/'controls.py',en.HERE/'engine.py',en.HERE/'newdata.py',en.HERE/'newdata_repair.py')},
        'purpose':'Whole deployment costs after all core/native GPU training; teacher/q/prototypes removed'}
    try:
        paths=[en.HERE/('run_'+devtag)/'RESULT.json',en.HERE/('run_'+newtag)/'RESULT.json']
        sources=[en.ex.read(p) for p in paths]
        if any(r['status'] not in ('COMPLETE','PARTIAL_FAILED_PATHS','COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION','PARTIAL_RESOLVED') for r in sources): raise ValueError('TERMINAL_SOURCE_RUNS_REQUIRED')
        result['source_results']=[{'path':str(p.relative_to(en.ROOT)),'sha256':en.ex.sha(p)} for p in paths]
        nvidia=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
        result['nvidia_smi_before_load']={'exit_code':nvidia.returncode,'stdout':nvidia.stdout,'stderr':nvidia.stderr}
        competing=[line for line in nvidia.stdout.splitlines() if 'python' in line.lower() and not line.lstrip().startswith(str(os.getpid())+',')]
        if nvidia.returncode!=0 or competing: raise ValueError('COMPETING_PYTHON_COMPUTE_PROCESS_OR_NVIDIA_FAILURE')
        start=time.perf_counter(); bolt=en.ex.previous.FrozenBolt(); en.replace_normalizer(bolt)
        result['bolt_load_wall_seconds']=time.perf_counter()-start; result['bolt_initial_hash']=bolt.initial_hash
        allcases=[c for r in sources for c in r['cases']]
        cases=[c for c in allcases if c.get('stage')!='RAW']
        datasets={d:en.ex.load_data(d,sizes=(256,64,None)) if d in ('jena','robin') else new_past_data(d,en.HERE/'METHOD_LOCK.json')
            for d in {c['dataset'] for c in cases}}
        for case in cases:
            try:
                loaded=controls.models_for_case(case,allcases)
            except Exception as error:
                failure={'dataset':case['dataset'],'label':case['label'],'stage':'MODEL_LOAD','error':str(error),'traceback':traceback.format_exc()}
                result.setdefault('failed_paths',[]).append(failure)
                en.write(output/f'{case["dataset"]}_{case["label"]}_MODEL_LOAD_FAILED.json',failure)
                continue
            split='val' if case['dataset'] in ('jena','robin') else 'eval'
            for method in case['methods']:
                try:
                    timings=controls.cost(datasets[case['dataset']],case,method,loaded,bolt,limit,split)
                except Exception as error:
                    if 'LIMIT_' in str(error) or 'SOURCE_CHANGED' in str(error): raise
                    failure={'dataset':case['dataset'],'label':case['label'],'method':method['method'],'stage':'COST','error':str(error),'traceback':traceback.format_exc()}
                    result.setdefault('failed_paths',[]).append(failure)
                    en.write(output/f'{case["dataset"]}_{case["label"]}_{method["method"]}_COST_FAILED.json',failure)
                    continue
                result['records'].append({'dataset':case['dataset'],'label':case['label'],'method':method['method'],
                    'prediction':method['prediction'],'latency':timings,'selected_models':case.get('selected_models',case.get('training',{}).get('selected_checkpoint'))})
            print(json.dumps({'cost_completed':case['label'],'dataset':case['dataset'],'method_records':len(result['records'])}),flush=True)
            del loaded
        result['canonical_aliases']=[]
        for case in (c for c in allcases if c.get('stage')=='RAW'):
            canonical=next(c for c in allcases if c['dataset']==case['dataset'] and c['label']=='RAW_CONTROLS')
            original=case['methods'][0]
            matched=next(m for m in canonical['methods'] if m['method']=='RAW_BOLT')
            try:
                assert en.ex.sha(en.ROOT/original['prediction']['path'])==original['prediction']['sha256']
                assert en.ex.sha(en.ROOT/matched['prediction']['path'])==matched['prediction']['sha256']
                with np.load(en.ROOT/original['prediction']['path'],allow_pickle=False) as a,np.load(en.ROOT/matched['prediction']['path'],allow_pickle=False) as b:
                    equality={s:bool(np.array_equal(a[s+'_prediction'],b[s+'_prediction'])) for s in ('fit','tune','val')}
                if not all(equality.values()): raise AssertionError('CANONICAL_RAW_DUPLICATE_PREDICTION_CHANGED')
            except Exception as error:
                failure={'dataset':case['dataset'],'label':case['label'],'method':'RAW_BOLT','stage':'CANONICAL_ALIAS','error':str(error)}
                result.setdefault('failed_paths',[]).append(failure)
                en.write(output/f'{case["dataset"]}_RAW_BOLT_ALIAS_FAILED.json',failure)
                continue
            canonical_record=next((r for r in result['records'] if r['dataset']==case['dataset'] and r['label']=='RAW_CONTROLS' and r['method']=='RAW_BOLT'),None)
            result['canonical_aliases'].append({'dataset':case['dataset'],'label':case['label'],'method':'RAW_BOLT',
                'canonical_label':'RAW_CONTROLS','canonical_method':'RAW_BOLT','actual_all_split_prediction_array_equal':equality,
                'latency':None if canonical_record is None else canonical_record['latency'],'note':'Same exact deployed RAW path and predictions; no duplicate timing run counted'})
        assert en.ex.previous.state_hash(bolt.model)==bolt.initial_hash
        for path,digest in result['source_hashes'].items():
            if en.ex.sha(en.ROOT/path)!=digest: raise ValueError('SOURCE_CHANGED '+path)
        result['bolt_final_hash']=en.ex.previous.state_hash(bolt.model)
        result['status']='PARTIAL_FAILED_PATHS' if result.get('failed_paths') else 'COMPLETE'
        limit.check(); limit.close(result['status']); result['resources']=limit.record
        en.write(output/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'cost_method_records':len(result['records']),'resources':limit.record}),flush=True)
    except BaseException as error:
        result.update(status='FAILED_PRESERVED',error=str(error),traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED',str(error)); result['resources']=limit.record
        en.write(output/'FAILURE.json',result)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); parser.add_argument('--devtag',default='dev_v1'); parser.add_argument('--newtag',default='new_v1')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('SAFE_TAG_REQUIRED')
    run(args.tag,args.devtag,args.newtag)
