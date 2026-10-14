"""Fit two genuinely new datasets, lock all models, open EVAL once, then measure."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import time
import traceback
import numpy as np
import torch
import engine as en
import controls as control
import newdata

NEW_DATASETS=('appliances','beijing_aotizhongxin')


def snapshots(output):
    sourcepaths=[Path(__file__),en.HERE/'controls.py',en.HERE/'newdata.py',en.HERE/'engine.py',en.HERE/'alignment.py',en.HERE/'PLAN_KO.md']
    folder=output/'sources'; folder.mkdir()
    records={}
    for path in sourcepaths:
        digest=en.ex.sha(path)
        with (folder/path.name).open('xb') as f: f.write(path.read_bytes())
        assert en.ex.sha(folder/path.name)==digest
        records[str(path.relative_to(en.ROOT))]=digest
    return records


def training_case(data,history,target,observed,seed,architecture,initbasis,mean,bolt,folder,output,limit,method,weight,auxbasis,result):
    return en.safe_fit_case(result,data,history,target,observed,seed,architecture,initbasis,mean,bolt,folder,output,limit,
        'S3',method,weight,auxbasis)


def fit_teacher(data,history,target,observed,seed,bolt,folder,output,limit):
    channels=target['fit'].shape[-1]
    label=f'TEACHER_FULL_IDENTITY_s{seed}'
    casefolder=folder/label; casefolder.mkdir()
    codec=en.Codec(np.zeros(channels,np.float32),np.eye(channels,dtype=np.float32),seed,'DIRECT_PCA').to('cuda')
    training=en.train(data,history,target,observed,codec,bolt,seed,casefolder,limit)
    pred={s:en.predict(x,codec,bolt,limit) for s,x in history.items()}
    basis,basisrecord=en.basis_from_future(pred['fit'],history['fit'],max(1,channels//4))
    artifact=casefolder/'FIT_TEACHER_BASIS.npz'
    with artifact.open('xb') as f: np.savez_compressed(f,basis=basis,fit_forecast=pred['fit'],fit_last=history['fit'][:,-1])
    case={'dataset':data['dataset'],'stage':'S3_TEACHER','label':label,'seed':seed,'architecture':'DIRECT_PCA','K':channels,'L':512,
        'training':training,'methods':[],'basis_fitting':basisrecord,'basis_artifact':{'path':str(artifact.relative_to(en.ROOT)),'sha256':en.ex.sha(artifact)},
        'selected_teacher_is_untrained_u0':training['selected_update']==0}
    en.add_prediction(case,casefolder,'BOLT',data,pred,target,observed)
    en.write(output/(label+'.json'),case)
    return case,basis


def safe_operation(result,output,label,dataset,operation):
    try: return operation()
    except Exception as error:
        if 'LIMIT_' in str(error) or 'SOURCE_CHANGED' in str(error): raise
        failure={'status':'FAILED_PATH_PRESERVED','dataset':dataset,'label':label,'error':str(error),'traceback':traceback.format_exc()}
        en.write(output/(label+'_FAILED.json'),failure); result.setdefault('failed_paths',[]).append(failure)
        print(json.dumps({'failed_path':label,'dataset':dataset,'error':str(error)}),flush=True)
        return None


def add_controls(data,history,target,observed,cases,methodlock,bolt,folder,output,limit,result):
    raw=safe_operation(result,output,'RAW_CONTROLS',data['dataset'],
        lambda:control.raw_controls(data,history,target,observed,bolt,folder/'RAW_CONTROLS',output,limit))
    if raw: result['cases'].append(raw)
    architecture=methodlock['architecture']
    learnedweight=methodlock['weights']['LEARNED_REGIME']
    baseline_stage='S1' if data['dataset'] in ('jena','robin') else 'S3'
    learned_stage='S2' if data['dataset'] in ('jena','robin') else 'S3'
    for seed in en.SEEDS:
        matches=[c for c in cases if c['dataset']==data['dataset'] and c.get('seed')==seed]
        basic=next((c for c in matches if c['stage']==baseline_stage and c['architecture']==architecture and c.get('aux_method')=='MAIN'),None)
        learned=next((c for c in matches if c['stage']==learned_stage and c.get('aux_method')=='LEARNED_REGIME' and c.get('aux_weight')==learnedweight),None)
        for role,case in (('SELECTED_BASELINE',basic),('LEARNED_REGIME',learned)):
            if case is None:
                result.setdefault('unavailable_controls',[]).append({'dataset':data['dataset'],'seed':seed,'role':role,'reason':'Source case unavailable'})
                continue
            record=safe_operation(result,output,case['label']+'_CONTROLS',data['dataset'],
                lambda ca=case:control.codec_controls(data,history,target,observed,ca,bolt,folder/(ca['label']+'_CONTROLS'),output,limit))
            if record:
                record['comparison_role']=role
                result['cases'].append(record)


def selected_artifacts(cases,dataset):
    artifacts=[]
    for case in cases:
        if case['dataset']!=dataset: continue
        if 'training' in case: artifacts.append(case['training']['selected_checkpoint'])
        if 'selected_models' in case: artifacts.append(case['selected_models'])
        for method in case['methods']:
            if 'native_training' in method: artifacts.append(method['native_training']['selected_checkpoint'])
    unique={a['path']:{'path':a['path'],'sha256':a['sha256']} for a in artifacts}
    for artifact in unique.values(): assert en.ex.sha(en.ROOT/artifact['path'])==artifact['sha256']
    return list(unique.values())


def new_fixture(data,history,target,observed,bolt,limit):
    channels=target['fit'].shape[-1]
    x=en.tensor(history['fit'][:2])
    codec=en.Codec(np.zeros(channels,np.float32),np.eye(channels,dtype=np.float32),en.SEEDS[0],'ANCHOR_PCA').to('cuda')
    state=en.ex.previous.clone_state(codec)
    with torch.no_grad():
        raw=bolt(x); predicted=codec.forward_path(x,bolt)
        parity=float((raw-predicted).abs().max())
        codec.codec.output.weight.fill_(.03125)
        altered=x.clone(); altered[:,256:]+=999
        causal=float((codec.encode(x)[:,:256]-codec.encode(altered)[:,:256]).abs().max())
    codec.load_state_dict(state)
    optimizer=torch.optim.AdamW(codec.parameters(),lr=.001,weight_decay=0.)
    actual=x.clone().requires_grad_(True)
    output=codec.forward_path(actual,bolt)
    loss=en.ex.previous.macro_loss(output,en.tensor(target['fit'][:2]),en.ex.boolean(observed['fit'][:2]))
    loss.backward()
    finite=bool(torch.isfinite(actual.grad).all() and all(torch.isfinite(p.grad).all() for p in codec.parameters() if p.grad is not None))
    optimizer.step()
    step=max(float((v-state[k].to(v.device)).abs().max()) for k,v in codec.state_dict().items())
    codec.load_state_dict(state)
    if parity>2e-6 or causal!=0 or not finite or step<=0: raise ValueError('NEW_DATA_IDENTITY_CAUSAL_GRADIENT_FIXTURE_FAILED')
    assert en.ex.previous.state_hash(bolt.model)==bolt.initial_hash
    limit.check()
    return {'identity_anchor_vs_raw_max_abs':parity,'active_causal_prefix_max_abs':causal,'finite_input_and_parameter_gradients':finite,
        'actual_temp_optimizer_step_max_abs':step,'state_restored':True,'F_hash_unchanged':True,
        'actual_filled_input_missing_counts':int((~np.stack([data['mask'][o-512:o] for o in data['origins']['fit'][:2]])).sum())}


def evaluate_case(data,evaluation,case,source_cases,bolt,folder,output,limit):
    history_eval=evaluation['history']
    origins=dict(data['origins']); origins['eval']=evaluation['origins']
    scored_data={**data,'origins':origins}
    loaded=control.models_for_case(case,source_cases)
    final=copy.deepcopy(case); final['methods']=[]
    final['evaluation_scope']='New chronological non-overlapping EVAL; methods locked on legacy development, per-dataset models selected on FIT/TUNE only'
    pathfolder=folder/'EVAL'/case['label']; pathfolder.mkdir(parents=True)
    for method in case['methods']:
        savedpath=en.ROOT/method['prediction']['path']; assert en.ex.sha(savedpath)==method['prediction']['sha256']
        with np.load(savedpath,allow_pickle=False) as saved:
            pred={s:saved[s+'_prediction'] for s in ('fit','tune')}
            target={s:saved[s+'_target'] for s in ('fit','tune')}
            observed={s:saved[s+'_observed'] for s in ('fit','tune')}
        pred['eval']=control.prediction_method(history_eval,method['method'],loaded,bolt,limit)
        target['eval']=evaluation['target']; observed['eval']=evaluation['observed']
        extra={key:value for key,value in method.items() if key not in ('method','prediction','scores')}
        en.add_prediction(final,pathfolder,method['method'],scored_data,pred,target,observed,extra)
    if 'training' in case:
        final['oracle_eval']=en.oracle(loaded['codec'],{'eval':history_eval},{'eval':evaluation['target']},{'eval':evaluation['observed']})
    en.write(output/('EVAL_'+case['label']+'.json'),final)
    print(json.dumps({'completed_EVAL':case['label'],'dataset':data['dataset'],
        'normalized_mse':final['methods'][0]['scores']['eval']['normalized']['mse']}),flush=True)
    return final


def run(tag,devtag):
    en.configure(); protected=en.ex.access_guard()
    output=en.HERE/('run_'+tag); output.mkdir(exist_ok=False)
    folder=en.CACHE/('run_'+tag); folder.mkdir(parents=True,exist_ok=False)
    limit=en.Limit(tag)
    methodpath=en.HERE/'METHOD_LOCK.json'; lock=en.ex.read(methodpath)
    devpath=en.HERE/('run_'+devtag)/'RESULT.json'; dev=en.ex.read(devpath)
    if dev['status'] not in ('COMPLETE','PARTIAL_FAILED_PATHS'): raise ValueError('TERMINAL_DEVELOPMENT_AND_METHOD_LOCK_REQUIRED')
    result={'status':'RUNNING','tag':tag,'cases':[],'protected_test_guarded':protected,'source_hashes':snapshots(output),
        'method_lock':{'path':str(methodpath.relative_to(en.ROOT)),'sha256':en.ex.sha(methodpath)},
        'development_source':{'path':str(devpath.relative_to(en.ROOT)),'sha256':en.ex.sha(devpath)},'newdata_contracts':{}}
    try:
        start=time.perf_counter(); bolt=en.ex.previous.FrozenBolt(); en.replace_normalizer(bolt)
        result['bolt_load_wall_seconds']=time.perf_counter()-start; result['bolt_initial_hash']=bolt.initial_hash
        # Development controls have no influence on the already sealed METHOD_LOCK.
        for dataset in ('jena','robin'):
            data=en.ex.load_data(dataset,sizes=(256,64,None)); histories,targets,observed,_=en.ex.arrays(data,512)
            datafolder=folder/dataset; datafolder.mkdir(); dataoutput=output/dataset; dataoutput.mkdir()
            add_controls(data,histories,targets,observed,dev['cases'],lock,bolt,datafolder,dataoutput,limit,result)
        training_data={}
        for dataset in NEW_DATASETS:
            datafolder=folder/dataset; datafolder.mkdir(); dataoutput=output/dataset; dataoutput.mkdir()
            data=safe_operation(result,dataoutput,'DATA_PREPARATION',dataset,
                lambda ds=dataset:newdata.load_training(ds,methodpath,allow_download=True))
            if data is None: continue
            result['newdata_contracts'][dataset]=newdata.required_model_selection_fields(dataset,methodpath)
            training_data[dataset]=data
            history,target,observed,_=en.ex.arrays(data,512)
            fixture=safe_operation(result,dataoutput,'NEW_DATA_FIXTURE',dataset,
                lambda:new_fixture(data,history,target,observed,bolt,limit))
            if fixture is None: continue
            result.setdefault('fixtures',{})[dataset]=fixture
            mean,pca,_=en.ex.initialization(data,max(1,data['x'].shape[1]//4))
            newcases=[]
            for seed in en.SEEDS:
                teacher=safe_operation(result,dataoutput,f'TEACHER_FULL_IDENTITY_s{seed}',dataset,
                    lambda se=seed:fit_teacher(data,history,target,observed,se,bolt,datafolder,dataoutput,limit))
                teacherbasis=None
                if teacher:
                    teachercase,teacherbasis=teacher; newcases.append(teachercase); result['cases'].append(teachercase)
                architectures=list(dict.fromkeys(('DIRECT_PCA','ANCHOR_PCA',lock['architecture'])))
                for architecture in architectures:
                    basis=pca if architecture.endswith('PCA') else teacherbasis
                    if basis is None:
                        result.setdefault('unavailable_paths',[]).append({'dataset':dataset,'seed':seed,'architecture':architecture,'reason':'Teacher unavailable'})
                        continue
                    case=training_case(data,history,target,observed,seed,architecture,basis,mean,bolt,datafolder,dataoutput,limit,'MAIN',0.,teacherbasis,result)
                    if case: newcases.append(case); result['cases'].append(case)
                architecture=lock['architecture']
                initialbasis=pca if architecture.endswith('PCA') else teacherbasis
                for method in en.METHODS:
                    if initialbasis is None or (teacherbasis is None and method!='BASE_Q'):
                        result.setdefault('unavailable_paths',[]).append({'dataset':dataset,'seed':seed,'method':method,'reason':'Teacher basis unavailable'})
                        continue
                    case=training_case(data,history,target,observed,seed,architecture,initialbasis,mean,bolt,datafolder,dataoutput,limit,
                        method,lock['weights'][method],teacherbasis,result)
                    if case: newcases.append(case); result['cases'].append(case)
            add_controls(data,history,target,observed,newcases,lock,bolt,datafolder,dataoutput,limit,result)
        # Every new dataset model, native checkpoint and correction selection is completed before any EVAL future payload open.
        all_models=[]
        for dataset in training_data: all_models.extend(selected_artifacts(result['cases'],dataset))
        globalpath=output/'MODEL_SELECTION_LOCK.json'
        globallock={'status':'ALL_NEW_MODELS_SELECTED_BEFORE_ANY_EVAL_TARGET_OPEN','created_unix':time.time(),
            'method_lock_sha256':en.ex.sha(methodpath),'evaluation_source_hashes':result['source_hashes'],'selected_models':all_models,
            'newdata_contracts':result['newdata_contracts'],'cases':[{k:c[k] for k in ('dataset','label')} for c in result['cases'] if c['dataset'] in training_data],
            'failed_paths':result.get('failed_paths',[]),'no_EVAL_target_read_by_training_or_selection':True}
        en.write(globalpath,globallock)
        perlocks={}
        for dataset in training_data:
            modelpath=output/(dataset+'_MODEL_SELECTION_LOCK.json')
            payload={**result['newdata_contracts'][dataset],'selected_models':selected_artifacts(result['cases'],dataset)}
            en.write(modelpath,{'status':'MODEL_SELECTION_LOCKED_BEFORE_EVAL_TARGET_OPEN','created_unix':time.time(),
                'global_model_selection_lock':{'path':str(globalpath.relative_to(en.ROOT)),'sha256':en.ex.sha(globalpath)},
                'evaluation_source_hashes':result['source_hashes'],'newdata_contract':payload})
            perlocks[dataset]=modelpath
        result['global_model_selection_lock']={'path':str(globalpath.relative_to(en.ROOT)),'sha256':en.ex.sha(globalpath)}
        corecases=list(result['cases'])
        evaluated=[]
        for dataset,data in training_data.items():
            for path,digest in result['source_hashes'].items():
                if en.ex.sha(en.ROOT/path)!=digest: raise AssertionError('SOURCE_CHANGED_BEFORE_EVAL_OPEN '+path)
            opened_start=time.time()
            evaluation=safe_operation(result,output,dataset+'_EVAL_OPEN',dataset,
                lambda ds=dataset:newdata.open_evaluation(ds,perlocks[ds],method_lock_path=methodpath))
            if evaluation is None: continue
            opened_end=time.time()
            receipt={'dataset':dataset,'first_and_only_open_started_unix':opened_start,'open_completed_unix':opened_end,
                'global_model_lock_created_unix':globallock['created_unix'],'perdataset_model_lock_path':str(perlocks[dataset].relative_to(en.ROOT)),
                'model_selection_lock_sha256':evaluation['model_selection_lock_sha256'],'eval_target_payload_sha256':evaluation['eval_target_payload_sha256'],
                'origins_sha256':en.ex.ahash(evaluation['origins']),'origin_count':len(evaluation['origins']),
                'actual_target_mask_observed_counts':evaluation['observed'].sum((0,1)).tolist()}
            en.write(output/(dataset+'_EVAL_FIRST_OPEN.json'),receipt)
            result.setdefault('evaluation_open_receipts',{})[dataset]=receipt
            for case in corecases:
                if case['dataset']!=dataset: continue
                final=safe_operation(result,output/dataset,case['label']+'_EVAL',dataset,
                    lambda ca=case:evaluate_case(data,evaluation,ca,corecases,bolt,folder/dataset,output/dataset,limit))
                if final: evaluated.append(final)
        result['cases']=[c for c in corecases if c['dataset'] not in training_data]+evaluated
        result['all_training_finished_unix']=globallock['created_unix']
        assert en.ex.previous.state_hash(bolt.model)==bolt.initial_hash
        for path,digest in result['source_hashes'].items():
            if en.ex.sha(en.ROOT/path)!=digest: raise AssertionError('SOURCE_CHANGED_DURING_NEW_RUN '+path)
        result['bolt_final_hash']=en.ex.previous.state_hash(bolt.model)
        result['status']='PARTIAL_FAILED_PATHS' if any(result.get(key) for key in ('failed_paths','unavailable_paths','unavailable_controls')) else 'COMPLETE'
        limit.check(); limit.close(result['status']); result['resources']=limit.record
        result['training_fits']=sum('training' in c for c in corecases)
        result['native_fits']=sum('native_training' in m for c in corecases for m in c['methods'])
        en.write(output/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'training_fits':result['training_fits'],'native_fits':result['native_fits'],'resources':limit.record}),flush=True)
    except BaseException as error:
        result.update(status='FAILED_PRESERVED',error=str(error),traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED',str(error)); result['resources']=limit.record
        en.write(output/'FAILURE.json',result)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); parser.add_argument('--devtag',default='dev_v1')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('SAFE_TAG_REQUIRED')
    run(args.tag,args.devtag)
