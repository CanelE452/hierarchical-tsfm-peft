"""Re-infer every saved FIT/TUNE-selected checkpoint without any EVAL target access."""
from pathlib import Path
import argparse
import json
import time
import traceback
import numpy as np
import torch
import engine as en
import controls
import newdata
import newdata_repair
from alignment import summary_numpy


class PathGuard:
    def __init__(self,result,output,dataset,label):
        self.result,self.output,self.dataset,self.label=result,output,dataset,label
    def __enter__(self): return self
    def __exit__(self,kind,error,tb):
        if error is None: return False
        if not isinstance(error,Exception) or any(s in str(error) for s in ('LIMIT_','SOURCE_CHANGED','FROZEN_BOLT','device-side assert','illegal memory access')): return False
        failure={'status':'FAILED_AUDIT_PATH_PRESERVED','dataset':self.dataset,'label':self.label,
            'error':str(error),'traceback':''.join(traceback.format_exception(kind,error,tb))}
        self.result.setdefault('failed_paths',[]).append(failure)
        en.write(self.output/f'{self.dataset}_{self.label}_FAILED.json',failure)
        print(json.dumps({'failed_audit_path':self.label,'dataset':self.dataset,'error':str(error)}),flush=True)
        return True


def run(tag,devtag,newtag):
    en.configure(); protected=en.ex.access_guard()
    output=en.HERE/('run_'+tag); output.mkdir(exist_ok=False)
    folder=en.CACHE/('run_'+tag); folder.mkdir(parents=True,exist_ok=False)
    limit=en.Limit(tag)
    paths=[en.HERE/('run_'+devtag)/'RESULT.json',en.HERE/('run_'+newtag)/'RESULT.json']
    source=[en.ex.read(p) for p in paths]
    if any(s['status'] not in ('COMPLETE','PARTIAL_FAILED_PATHS','COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION','PARTIAL_RESOLVED') for s in source): raise ValueError('TERMINAL_SOURCES_REQUIRED')
    result={'status':'RUNNING','records':[],'native_records':[],'alignment_records':[],'new_training_updates':0,'protected_test_guarded':protected,
        'source_results':[{'path':str(p.relative_to(en.ROOT)),'sha256':en.ex.sha(p)} for p in paths],
        'source_hashes':{str(p.relative_to(en.ROOT)):en.ex.sha(p) for p in (Path(__file__),en.HERE/'engine.py',en.HERE/'controls.py',en.HERE/'newdata.py',en.HERE/'newdata_repair.py')}}
    try:
        start=time.perf_counter(); bolt=en.ex.previous.FrozenBolt(); en.replace_normalizer(bolt)
        result['bolt_load_wall_seconds']=time.perf_counter()-start; result['bolt_initial_hash']=bolt.initial_hash
        allcases=[c for r in source for c in r['cases']]
        datasets={d:en.ex.load_data(d,sizes=(256,64,None)) if d in ('jena','robin') else
            (newdata_repair.load_training(d,en.HERE/'METHOD_LOCK.json') if d=='beijing_aotizhongxin' else newdata.load_training(d,en.HERE/'METHOD_LOCK.json'))
            for d in {c['dataset'] for c in allcases}}
        arrays={d:en.ex.arrays(data,512)[:3] for d,data in datasets.items()}
        for case in allcases:
            if 'training' not in case: continue
            data=datasets[case['dataset']]; history,target,mask=arrays[case['dataset']]
            cf=folder/case['dataset']/case['label']; cf.mkdir(parents=True)
            for cp in case['training']['checkpoints']:
                with PathGuard(result,output,case['dataset'],case['label']+'_CORE_u'+str(cp['update'])):
                    assert en.ex.sha(en.ROOT/cp['path'])==cp['sha256']
                    saved=torch.load(en.ROOT/cp['path'],map_location='cpu',weights_only=True)
                    basis=saved['codec']['fixed_basis'].numpy()
                    codec=en.Codec(np.zeros(basis.shape[0],np.float32),basis,case['seed'],case['architecture']).to('cuda')
                    codec.load_state_dict(saved['codec']); codec.eval().requires_grad_(False)
                    p=en.predict(history['tune'],codec,bolt,limit)
                    score=en.metrics(data,p,target['tune'],mask['tune'])
                    path=cf/f'u{cp["update"]}_TUNE.npz'
                    with path.open('xb') as f: np.savez_compressed(f,prediction=p,target=target['tune'],observed=mask['tune'],origins=data['origins']['tune'],mean=data['mean'],std=data['std'])
                    maxerror=max(abs(score['normalized'][key]-cp['forecast']['tune']['normalized'][key]) for key in ('mse','mae'))
                    parity=all(abs(score['normalized'][key]-cp['forecast']['tune']['normalized'][key])<=1e-6+2e-6*abs(cp['forecast']['tune']['normalized'][key]) for key in ('mse','mae'))
                    result['records'].append({'dataset':case['dataset'],'label':case['label'],'update':cp['update'],
                        'checkpoint':{'path':cp['path'],'sha256':cp['sha256']},'prediction':{'path':str(path.relative_to(en.ROOT)),'sha256':en.ex.sha(path)},
                        'scores':score,'normalized_metric_max_abs_vs_record':maxerror,'absolute_tolerance':1e-6,'relative_tolerance':2e-6,'pass':parity})
                    if not parity: raise AssertionError('CHECKPOINT_REINFERENCE_SCORE_CHANGED')
            print(json.dumps({'checkpoint_audit_completed':case['label'],'dataset':case['dataset'],'records':len(result['records'])}),flush=True)
            if 'alignment' in case:
                with PathGuard(result,output,case['dataset'],case['label']+'_ALIGNMENT'):
                    selected=case['training']['selected_checkpoint']
                    alignmentpath=(en.ROOT/selected['path']).parent/'alignment.pt'
                    if not alignmentpath.is_file(): raise FileNotFoundError('SELECTED_ALIGNMENT_STATE_MISSING '+str(alignmentpath))
                    state=torch.load(alignmentpath,map_location='cpu',weights_only=True)
                    selectedstate=torch.load(en.ROOT/selected['path'],map_location='cpu',weights_only=True)
                    assert torch.equal(state['q.weight'],selectedstate['q']['weight']) and torch.equal(state['q.bias'],selectedstate['q']['bias'])
                    codec=controls.load_codec(case)
                    started=time.perf_counter()
                    before=en.predict(history['fit'],codec,bolt,limit,delta=True)
                    q=torch.nn.Linear(before.shape[-1],before.shape[-1])
                    q.load_state_dict(selectedstate['q'])
                    with torch.no_grad(): after=q(torch.from_numpy(before)).numpy()
                    targetlatent=state['target'].numpy(); valid=state['valid'].numpy()
                    means,stds=state['means'].numpy(),state['stds'].numpy()
                    latentstd=state['latent_std'].numpy(); summarytarget=state['summary_target'].numpy()
                    complete=state['summary_valid'].numpy().all(1)
                    diagnostics={}
                    for name,values in (('before_q',before),('after_q',after)):
                        latentmse=(((values-targetlatent)/latentstd)**2*valid[:,:,None]).sum()/(max(1,valid.sum())*values.shape[-1])
                        summary,_=summary_numpy(values,valid)
                        summary_error=(((summary-means)/stds)[complete]-summarytarget[complete]).astype(np.float64)
                        summarymse=float(np.square(summary_error).mean()) if complete.any() else None
                        diagnostics[name+'_latent_future_normalized_mse']=float(latentmse)
                        diagnostics[name+'_complete_summary_normalized_mse']=summarymse
                    artifactpath=cf/'SELECTED_FIT_ALIGNMENT.npz'
                    with artifactpath.open('xb') as f:
                        np.savez_compressed(f,latent_fit_pre_q=before,latent_fit_post_q=after,target_latent=targetlatent,
                            valid=valid,means=means,stds=stds,latent_std=latentstd,summary_target=summarytarget,
                            complete_summary_origins=complete,origins=data['origins']['fit'],q_weight=state['q.weight'].numpy(),q_bias=state['q.bias'].numpy())
                    expected=case['alignment']['FIT_diagnostics']
                    errors={name:None if value is None or expected.get(name) is None else abs(value-expected[name]) for name,value in diagnostics.items()}
                    passed=all(value is None or value<=2e-5+2e-6*abs(expected[name]) for name,value in errors.items())
                    result['alignment_records'].append({'dataset':case['dataset'],'label':case['label'],'diagnostics':diagnostics,
                        'max_abs_vs_original':errors,'pass':passed,'actual_FIT_reinference_wall_seconds':time.perf_counter()-started,
                        'alignment_source':{'path':str(alignmentpath.relative_to(en.ROOT)),'sha256':en.ex.sha(alignmentpath)},
                        'prediction':{'path':str(artifactpath.relative_to(en.ROOT)),'sha256':en.ex.sha(artifactpath)},'selected_checkpoint':selected})
                    if not passed: raise AssertionError('ALIGNMENT_Q_PRE_POST_DIAGNOSTICS_CHANGED')
        for case in allcases:
            for method in case['methods']:
                if 'native_training' not in method: continue
                with PathGuard(result,output,case['dataset'],case['label']+'_NATIVE_MODEL'):
                    loaded=controls.models_for_case(case,allcases)
                    data=datasets[case['dataset']]; history,target,mask=arrays[case['dataset']]
                    nativefolder=folder/case['dataset']/case['label']; nativefolder.mkdir(parents=True,exist_ok=True)
                    for cp in method['native_training']['checkpoints']:
                        with PathGuard(result,output,case['dataset'],case['label']+'_NATIVE_u'+str(cp['update'])):
                            assert en.ex.sha(en.ROOT/cp['path'])==cp['sha256']
                            loaded['native'].load_state_dict(torch.load(en.ROOT/cp['path'],map_location='cpu',weights_only=True))
                            pred=en.predict(history['tune'],loaded['codec'],loaded['native'],limit)
                            score=en.metrics(data,pred,target['tune'],mask['tune'])
                            maximum=max(abs(score['normalized'][key]-cp['tune']['normalized'][key]) for key in ('mse','mae'))
                            parity=all(abs(score['normalized'][key]-cp['tune']['normalized'][key])<=1e-6+2e-6*abs(cp['tune']['normalized'][key]) for key in ('mse','mae'))
                            nativepath=nativefolder/f'native_u{cp["update"]}_TUNE.npz'
                            with nativepath.open('xb') as f: np.savez_compressed(f,prediction=pred,target=target['tune'],observed=mask['tune'],origins=data['origins']['tune'],mean=data['mean'],std=data['std'])
                            result['native_records'].append({'dataset':case['dataset'],'label':case['label'],'update':cp['update'],
                                'checkpoint':{'path':cp['path'],'sha256':cp['sha256']},'normalized_metric_max_abs_vs_record':maximum,'scores':score,
                                'prediction':{'path':str(nativepath.relative_to(en.ROOT)),'sha256':en.ex.sha(nativepath)},
                                'absolute_tolerance':1e-6,'relative_tolerance':2e-6,'pass':parity})
                            if not parity: raise AssertionError('NATIVE_CHECKPOINT_REINFERENCE_CHANGED')
        assert en.ex.previous.state_hash(bolt.model)==bolt.initial_hash
        for path,digest in result['source_hashes'].items():
            if en.ex.sha(en.ROOT/path)!=digest: raise ValueError('SOURCE_CHANGED '+path)
        result['bolt_final_hash']=en.ex.previous.state_hash(bolt.model)
        result['status']='PARTIAL_FAILED_PATHS' if result.get('failed_paths') else 'COMPLETE'
        limit.check(); limit.close(result['status']); result['resources']=limit.record
        en.write(output/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'checkpoint_records':len(result['records']),'native_checkpoint_records':len(result['native_records']),
            'alignment_records':len(result['alignment_records']),'resources':limit.record}),flush=True)
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
