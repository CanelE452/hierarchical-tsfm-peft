"""Bounded from-scratch small-K, future alignment and locked new-data study."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('HF_HUB_OFFLINE','1')
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT/'.cache'/HERE.name
OLD = ROOT/'research/forecast_generalization_study_20261013'
sys.path.insert(0,str(OLD))
import experiment as ex
from diagnose_constant_gradient import replace_normalizer, normalizer_fixture
sys.path.insert(0,str(HERE))
from alignment import Alignment, METHODS, WEIGHTS
import numpy as np
import torch
from torch import nn
import psutil

SEEDS = (92601,92602)
CHECKPOINTS = (0,100,200,400,600)
ARCHITECTURES = ('DIRECT_PCA','ANCHOR_PCA','DIRECT_TEACHER','ANCHOR_TEACHER','ANCHOR_POINTWISE','ANCHOR_NONLINEAR_D')
H,L = 48,512


def write(path,value):
    path = Path(path)
    if not (path.resolve().is_relative_to(HERE) or path.resolve().is_relative_to(CACHE)):
        raise ValueError('WRITE_OUTSIDE_NEW_CAMPAIGN')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(ex.jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def save(path,value):
    path = Path(path)
    if not path.resolve().is_relative_to(CACHE): raise ValueError('CHECKPOINT_OUTSIDE_NEW_CACHE')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: torch.save(value,f)
    return {'path':str(path.relative_to(ROOT)),'sha256':ex.sha(path)}


class Limit:
    """No mutable JSON: stdout while active, one terminal receipt."""
    def __init__(self,tag):
        self.tag,self.start = tag,time.perf_counter()
        self.proc=psutil.Process()
        self.previous = sum(r['wall_seconds'] for p in (HERE/'resources').glob('*.json') if (r:=ex.read(p)).get('gpu_process',False))
        self.last_storage=-60
        self.record={'tag':tag,'pid':os.getpid(),'started_unix':time.time(),'gpu_process':True,
            'wall_seconds':0.,'peak_sampled_rss_bytes':0,'peak_allocated_vram_bytes':0,'previous_gpu_process_wall_seconds':self.previous}
    def check(self):
        self.record['wall_seconds']=time.perf_counter()-self.start
        self.record['cumulative_gpu_process_wall_seconds']=self.previous+self.record['wall_seconds']
        self.record['peak_sampled_rss_bytes']=max(self.record['peak_sampled_rss_bytes'],self.proc.memory_info().rss)
        self.record['peak_allocated_vram_bytes']=max(self.record['peak_allocated_vram_bytes'],torch.cuda.max_memory_allocated())
        if self.record['wall_seconds']-self.last_storage>30:
            self.record['new_bytes']=sum(p.stat().st_size for base in (CACHE,HERE) for p in base.rglob('*') if p.is_file())
            self.last_storage=self.record['wall_seconds']
        if self.record['cumulative_gpu_process_wall_seconds']>=7200: raise RuntimeError('LIMIT_GPU_PROCESS_WALL_7200S')
        if max(self.record['peak_sampled_rss_bytes'],self.record['peak_allocated_vram_bytes'])>8*1024**3: raise RuntimeError('LIMIT_RAM_ALLOCATED_VRAM_8GIB')
        if self.record.get('new_bytes',0)>12*1024**3: raise RuntimeError('LIMIT_STORAGE_12GIB')
        campaign_started = ex.read(HERE/'PREFLIGHT.json').get('started_unix',self.record['started_unix']) if (HERE/'PREFLIGHT.json').exists() else self.record['started_unix']
        if time.time()-campaign_started>28800: raise RuntimeError('LIMIT_CAMPAIGN_WALL_8H')
    def close(self,status,error=None):
        self.record.update(wall_seconds=time.perf_counter()-self.start,status=status,error=error)
        self.record['cumulative_gpu_process_wall_seconds']=self.previous+self.record['wall_seconds']
        write(HERE/'resources'/f'{self.tag}_{os.getpid()}_{time.time_ns()}.json',self.record)


def configure():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available(): raise RuntimeError('APPROVED_CUDA_UNAVAILABLE')


def tensor(x): return ex.tensor(x)


class Codec(nn.Module):
    def __init__(self,mean,basis,seed,architecture):
        super().__init__()
        self.architecture=architecture
        self.anchor=architecture.startswith('ANCHOR')
        self.codec=ex.previous.Codec(mean,basis,seed,nonlinear=architecture!='ANCHOR_POINTWISE')
        self.receptive_field=1 if architecture=='ANCHOR_POINTWISE' else 127
        self.residual=None
        if architecture=='ANCHOR_NONLINEAR_D':
            self.residual=nn.Sequential(nn.Linear(basis.shape[1],32),nn.GELU(),nn.Linear(32,basis.shape[0]))
            with torch.no_grad(): self.residual[-1].weight.zero_(); self.residual[-1].bias.zero_()
        self.register_buffer('fixed_basis',torch.from_numpy(basis.copy()))
    def encode(self,x): return self.codec.encode(x)
    def decode(self,z):
        result=self.codec.decode(z)
        return result if self.residual is None else result+self.residual(z)
    def forward_path(self,x,forecaster,return_delta=False):
        z=self.encode(x)
        future=forecaster(z)
        output=self.decode(future)
        if self.anchor: output=x[:,-1:]+output-self.decode(z[:,-1:])
        return (output,future-z[:,-1:]) if return_delta else output
    def metadata(self):
        registered=sum(p.numel() for p in self.parameters())
        inactive=self.codec.decode.bias.numel() if self.anchor else 0
        if self.anchor and self.residual is not None: inactive+=self.residual[-1].bias.numel()
        return {'registered_parameters':registered,'active_structural_parameters':registered-inactive,
            'structurally_cancelled_decoder_bias_parameters':inactive,'RF':self.receptive_field,
            'raw_current_state_bypass_C':self.codec.project.in_features if self.anchor else 0,
            'strict_K_only_compression':not self.anchor,'nonlinear_decoder_hidden':32 if self.residual is not None else None}


def predict(history,codec,forecaster,limit,delta=False):
    result=[]
    with torch.no_grad():
        for i in range(0,len(history),8):
            x=tensor(history[i:i+8])
            value=forecaster(x) if codec is None else codec.forward_path(x,forecaster,return_delta=delta)
            if delta: value=value[1]
            result.append(value.cpu().numpy())
            limit.check()
    return np.concatenate(result)


def teacher_basis(data,history,seed,bolt,limit,folder):
    start=time.perf_counter()
    source=ex.read(OLD/'run_resolved_v1/RESULT.json')
    c=data['x'].shape[1]
    case=next(ca for ca in source['cases'] if ca['dataset']==data['dataset'] and ca.get('seed')==seed and ca.get('k')==c
        and ca.get('objective')=='PRED' and ca.get('initialization',{}).get('kind')=='IDENTITY')
    checkpoint=next(cp for cp in case['training']['checkpoints'] if cp['update']==case['selected_update'])
    assert ex.sha(ROOT/checkpoint['path'])==checkpoint['sha256']
    teacher=ex.Codec(np.zeros(c,np.float32),np.eye(c,dtype=np.float32),seed).to('cuda')
    teacher.load_state_dict(torch.load(ROOT/checkpoint['path'],map_location='cpu',weights_only=True)['codec'])
    teacher.eval().requires_grad_(False)
    future=ex.predict(history['fit'],teacher,bolt,limit)
    basis,basis_record=basis_from_future(future,history['fit'],max(1,c//4))
    artifact=folder/f'TEACHER_FIT_s{seed}.npz'
    with artifact.open('xb') as f: np.savez_compressed(f,forecast=future,last=history['fit'][:,-1],basis=basis)
    record={'source_manifest':str((OLD/'run_resolved_v1/RESULT.json').relative_to(ROOT)),
        'source_manifest_sha256':ex.sha(OLD/'run_resolved_v1/RESULT.json'),'checkpoint':checkpoint,
        'selected_update':case['selected_update'],'teacher_used_original_TUNE':True,
        'teacher_is_untrained_u0':case['selected_update']==0,'new_teacher_training_updates':0,
        'fit_output_artifact':{'path':str(artifact.relative_to(ROOT)),'sha256':ex.sha(artifact)},
        'extraction_wall_seconds':time.perf_counter()-start,**basis_record}
    return basis,record


def basis_from_future(future,history,k):
    changes=(future-history[:,-1:]).reshape(-1,future.shape[-1]).astype(np.float64)
    centered=changes-changes.mean(0)
    values,vectors=np.linalg.eigh(centered.T@centered/len(centered))
    basis=vectors[:,-k:][:,::-1].copy()
    for j in range(k):
        if basis[np.argmax(np.abs(basis[:,j])),j]<0: basis[:,j]*=-1
    basis=basis.astype(np.float32)
    return basis,{'basis_sha256':ex.ahash(basis),'basis_scope':'FIT teacher forecast minus current state covariance only',
        'teacher_delta_mean':changes.mean(0).tolist(),'teacher_delta_eigenvalues':values.tolist(),
        'teacher_delta_explained_variance_fraction':float(values[-k:].sum()/max(values.sum(),1e-30))}


def metrics(data,prediction,target,observed): return ex.metric(data,prediction,target,observed)


def train(data,history,target,observed,codec,bolt,seed,folder,limit,method='MAIN',weight=0.,alignment=None):
    started=time.perf_counter()
    initial=ex.previous.state_hash(codec)
    codec.train().requires_grad_(True)
    parameters=list(codec.parameters())+(list(alignment.q.parameters()) if alignment is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=0.)
    order=np.random.default_rng(seed).integers(0,len(history['fit']),size=(600,4))
    checkpoints,curve=[],[]
    for update in range(601):
        if update in CHECKPOINTS:
            p={s:predict(history[s],codec,bolt,limit) for s in ('fit','tune')}
            artifact=save(folder/f'u{update}.pt',{'codec':ex.previous.clone_state(codec),'q':None if alignment is None else ex.previous.clone_state(alignment.q),
                'update':update,'seed':seed,'architecture':codec.architecture,'method':method,'weight':weight})
            checkpoint={'update':update,**artifact,'forecast':{s:metrics(data,p[s],target[s],observed[s]) for s in p}}
            checkpoints.append(checkpoint)
            print(json.dumps({'stage_fit':folder.name,'dataset':data['dataset'],'architecture':codec.architecture,'seed':seed,'method':method,
                'weight':weight,'update':update,'tune_norm_mse':checkpoint['forecast']['tune']['normalized']['mse'],
                'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if update==600: break
        indices=order[update]
        optimizer.zero_grad(set_to_none=True)
        output,delta=codec.forward_path(tensor(history['fit'][indices]),bolt,True)
        main=ex.previous.macro_loss(output,tensor(target['fit'][indices]),ex.boolean(observed['fit'][indices]))
        aux=main*0 if alignment is None else alignment.loss(method,delta,torch.as_tensor(indices,device='cuda'))
        loss=main+weight*aux
        grad_diagnostic={}
        if alignment is not None and update in (0,99,199,399,599):
            collected={}
            for name,value in (('main',main),('auxiliary',aux)):
                gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                grad_diagnostic[name+'_unclipped_gradient_norm']=float(torch.sqrt(sum(g.detach().square().sum() for g in gradients if g is not None)))
                codecgrad=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(codec.parameters(),gradients[:len(list(codec.parameters()))])]
                collected[name]=torch.cat(codecgrad)
                grad_diagnostic[name+'_codec_gradient_norm']=float(collected[name].norm())
            product=collected['main'].norm()*collected['auxiliary'].norm()
            grad_diagnostic['main_auxiliary_codec_gradient_cosine']=float(torch.dot(collected['main'],collected['auxiliary'])/product) if product>0 else None
        if not torch.isfinite(loss): raise ValueError('NONFINITE_LOSS')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(gradient): raise ValueError('NONFINITE_GRADIENT')
        optimizer.step()
        curve.append({'update':update+1,'main_loss':float(main.detach()),'aux_loss':float(aux.detach()),'loss':float(loss.detach()),'gradient_norm':float(gradient),**grad_diagnostic})
        if update%10==0: limit.check()
    selected=min(checkpoints,key=lambda p:p['forecast']['tune']['normalized']['mse'])
    stored=torch.load(ROOT/selected['path'],map_location='cpu',weights_only=True)
    codec.load_state_dict(stored['codec'])
    if alignment is not None: alignment.q.load_state_dict(stored['q'])
    codec.eval().requires_grad_(False)
    return {'initial_codec_hash':initial,'order_sha256':ex.ahash(order),'batch_indices':order.tolist(),'seed':seed,
        'updates':600,'batch_size':4,'optimizer':'AdamW','lr':.001,'weight_decay':0.,'clip_gradient_norm':1.,
        'F_calls_training':600,'checkpoints':checkpoints,'selected_update':selected['update'],'selected_checkpoint':selected,
        'selected_codec_hash':ex.previous.state_hash(codec),'curve':curve,'train_wall_seconds':time.perf_counter()-started,
        'selection_rule':'TUNE normalized observed channel-macro MSE, earliest exact tie; no VAL/EVAL selection',
        'q_registered_parameters':0 if alignment is None else sum(p.numel() for p in alignment.q.parameters()),
        'q_active_loss_gradient':alignment is not None and weight>0}


def add_prediction(case,folder,name,data,pred,target,observed,extra=None):
    path=folder/(name+'.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f,**{s+'_prediction':v for s,v in pred.items()},**{s+'_target':target[s] for s in pred},
            **{s+'_observed':observed[s] for s in pred},**{s+'_origins':data['origins'][s] for s in pred},mean=data['mean'],std=data['std'])
    record={'method':name,'prediction':{'path':str(path.relative_to(ROOT)),'sha256':ex.sha(path)},
        'scores':{s:metrics(data,p,target[s],observed[s]) for s,p in pred.items()}}
    if extra: record.update(extra)
    case.setdefault('methods',[]).append(record)
    return record


def oracle(codec,history,target,observed):
    if codec.residual is not None: return {'not_applicable':'nonlinear D has no fixed linear column-space diagnostic'}
    basis=codec.codec.decode.weight.detach().cpu().numpy().astype(np.float64)
    bias=codec.codec.decode.bias.detach().cpu().numpy().astype(np.float64)
    result={}
    for split in target:
        y=target[split].astype(np.float64)
        modes={}
        for mode,base in (('DIRECT_AFFINE_Y',bias[None,None]),('ANCHOR_DELTA_Y_MINUS_LAST',history[split][:,-1:])):
            desired=y-base
            flat=desired.reshape(-1,y.shape[-1]); masks=observed[split].reshape(flat.shape)
            projected=np.zeros_like(flat)
            groups={}
            for i,mask in enumerate(masks): groups.setdefault(mask.tobytes(),[]).append(i)
            for rows in groups.values():
                mask=masks[rows[0]]
                if mask.any():
                    z=np.linalg.lstsq(basis[mask],flat[rows][:,mask].T,rcond=None)[0].T
                    projected[rows]=z@basis.T
            projection=projected.reshape(y.shape)+base
            modes[mode]=ex.score(projection,y,observed[split])
        result[split]={'projection_residual':modes['ANCHOR_DELTA_Y_MINUS_LAST' if codec.anchor else 'DIRECT_AFFINE_Y'],
            'same_D_direct_and_anchor_projection_residuals':modes,'fixed_D_rank':int(np.linalg.matrix_rank(basis)),
            'scope':'Uses actual future targets per-origin/per-horizon; fixed-D projection diagnostic only, not deployment or universal lower bound'}
    return result


def fixture(data,history,target,observed,bolt,limit):
    c=data['x'].shape[1]
    model=Codec(np.zeros(c,np.float32),np.eye(c,dtype=np.float32),SEEDS[0],'ANCHOR_PCA').to('cuda')
    x=tensor(history['fit'][:2])
    with torch.no_grad():
        raw=bolt(x)
        actual=model.forward_path(x,bolt)
        difference=float((actual-raw).abs().max())
        causal=model.encode(x)[:,:256]
        changed=x.clone(); changed[:,256:]+=99
        causal_delta=float((model.encode(changed)[:,:256]-causal).abs().max())
        # Activate temporal branch so zero-initialization cannot hide support bugs.
        model.codec.output.weight.fill_(.03125)
        active=model.encode(x)[:,:256]
        active_delta=float((model.encode(changed)[:,:256]-active).abs().max())
    assert difference<2e-6 and causal_delta==0 and active_delta==0
    mean,basis,_=ex.initialization(data,max(1,c//4))
    resolved=ex.read(OLD/'run_resolved_v1/RESULT.json')
    oldcase=next(ca for ca in resolved['cases'] if ca['dataset']==data['dataset'] and ca.get('seed')==SEEDS[0]
        and ca.get('stage')=='G1' and ca.get('objective')=='PRED' and ca.get('config')=='BASE')
    checkpoint=next(cp for cp in oldcase['training']['checkpoints'] if cp['update']==oldcase['selected_update'])
    assert ex.sha(ROOT/checkpoint['path'])==checkpoint['sha256']
    state=torch.load(ROOT/checkpoint['path'],map_location='cpu',weights_only=True)['codec']
    original=ex.Codec(mean,basis,SEEDS[0]).to('cuda'); original.load_state_dict(state)
    direct=Codec(mean,basis,SEEDS[0],'DIRECT_PCA').to('cuda'); direct.codec.load_state_dict(state)
    with torch.no_grad(): baseline_delta=float((direct.forward_path(x,bolt)-original.decode(bolt(original.encode(x)))).abs().max())
    assert baseline_delta==0
    groups=[]
    for architecture in ARCHITECTURES:
        codec=Codec(mean,basis,SEEDS[0],architecture).to('cuda')
        state=ex.previous.clone_state(codec); initial=ex.previous.state_hash(codec)
        optimizer=torch.optim.AdamW(codec.parameters(),lr=.001,weight_decay=0.)
        prediction=codec.forward_path(x,bolt)
        loss=ex.previous.macro_loss(prediction,tensor(target['fit'][:2]),ex.boolean(observed['fit'][:2]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in codec.parameters() if p.grad is not None)
        optimizer.step()
        delta=max(float((value-state[name].to(value.device)).abs().max()) for name,value in codec.state_dict().items())
        assert delta>0
        codec.load_state_dict(state)
        assert ex.previous.state_hash(codec)==initial
        groups.append({'architecture':architecture,'finite_gradient':True,'actual_temp_optimizer_step_max_abs':delta,
            'state_restored':True,**codec.metadata()})
    assert ex.previous.state_hash(bolt.model)==bolt.initial_hash
    missing=np.zeros((1,L,c),np.float32)
    masks=np.zeros_like(missing,dtype=bool)
    fill=ex.preprocess(missing,masks,data['mean'],data['std'])
    assert np.max(np.abs(fill))==0
    return {'identity_anchor_vs_RAW_BOLT_max_abs':difference,'causal_prefix_max_abs':causal_delta,
        'causal_active_temporal_prefix_max_abs':active_delta,'missing_window_reset_all_missing_normalized_zero':True,
        'existing_quarter_selected_checkpoint_forward_parity_max_abs':baseline_delta,
        'existing_quarter_checkpoint':checkpoint,
        'architecture_gradient_temp_update_fixtures':groups,'backbone_hash_unchanged':True}


def fit_case(data,history,target,observed,seed,architecture,basis,mean,bolt,folder,output,limit,stage,method='MAIN',weight=0.,auxiliary_basis=None):
    label=f'{stage}_{architecture}_{method}_w{weight:g}_s{seed}'
    casefolder=folder/label
    casefolder.mkdir(exist_ok=False)
    codec=Codec(mean,basis,seed,architecture).to('cuda')
    initial_oracle=oracle(codec,history,target,observed)
    align=None
    auxstart=time.perf_counter()
    if stage=='S2' or (stage=='S3' and method in METHODS):
        align=Alignment(target['fit'],observed['fit'],history['fit'],basis if auxiliary_basis is None else auxiliary_basis,data['origins']['fit'],seed).to('cuda')
    auxwall=time.perf_counter()-auxstart
    training=train(data,history,target,observed,codec,bolt,seed,casefolder,limit,method,weight,align)
    case={'dataset':data['dataset'],'stage':stage,'label':label,'seed':seed,'architecture':architecture,'aux_method':method,'aux_weight':weight,
        'K':basis.shape[1],'L':L,'H':H,'basis_sha256':ex.ahash(basis),'mean_sha256':ex.ahash(mean),'training':training,
        'capacity':codec.metadata(),'teacher_auxiliary_fit_wall_seconds':auxwall,'methods':[]}
    p={s:predict(history[s],codec,bolt,limit) for s in history}
    add_prediction(case,casefolder,'BOLT',data,p,target,observed)
    case['oracle']=oracle(codec,history,target,observed)
    case['initial_oracle']=initial_oracle
    if align is not None:
        delta=predict(history['fit'],codec,bolt,limit,delta=True)
        case['alignment']={'metadata':align.metadata,'FIT_diagnostics':align.diagnostics(tensor(delta))}
        save(casefolder/'alignment.pt',ex.previous.clone_state(align))
    write(output/(label+'.json'),case)
    return case


def safe_fit_case(result,*args,**kwargs):
    try:
        return fit_case(*args,**kwargs)
    except Exception as error:
        if 'LIMIT_' in str(error) or 'SOURCE_CHANGED' in str(error): raise
        data,history,target,observed,seed,architecture,basis,mean,bolt,folder,output,limit,stage=args[:13]
        method=kwargs.get('method',args[13] if len(args)>13 else 'MAIN')
        weight=kwargs.get('weight',args[14] if len(args)>14 else 0.)
        label=f'{stage}_{architecture}_{method}_w{weight:g}_s{seed}'
        failure={'status':'FAILED_PATH_PRESERVED','dataset':data['dataset'],'label':label,'error':str(error),
            'traceback':traceback.format_exc(),'resources_at_failure':dict(limit.record)}
        write(output/(label+'_FAILED.json'),failure)
        result.setdefault('failed_paths',[]).append(failure)
        print(json.dumps({'failed_path':label,'dataset':data['dataset'],'error':str(error)}),flush=True)
        return None


def choose(cases,raw,kind):
    names=ARCHITECTURES if kind=='architecture' else METHODS
    candidates=[]
    for name in names:
        weights=[0.] if kind=='architecture' or name=='BASE_Q' else list(WEIGHTS)
        scores=[]
        for weight in weights:
            matched=[c for c in cases if (c['architecture']==name if kind=='architecture' else c['aux_method']==name and c['aux_weight']==weight)]
            if len(matched)!=4:
                scores.append({'weight':weight,'coverage':len(matched),'complete':False,'reason':'Requires 2 datasets x 2 seeds complete coverage'})
                continue
            ratios=[c['methods'][0]['scores']['tune']['normalized']['mse']/raw[c['dataset']] for c in matched]
            scores.append({'weight':weight,'coverage':4,'complete':True,'log_mean_TUNE_RAW_ratio':float(np.log(ratios).mean()),'ratios':ratios,
                           'cases':[c['label']+':'+c['dataset'] for c in matched]})
        complete=[s for s in scores if s['complete']]
        best=min(complete,key=lambda v:v['log_mean_TUNE_RAW_ratio']) if complete else None
        candidates.append({'name':name,'chosen_weight':best['weight'] if best else (0. if name=='BASE_Q' else .03),
            'score':best['log_mean_TUNE_RAW_ratio'] if best else None,'complete_candidate':best is not None,
            'fallback_note':None if best else 'No complete development candidate; predeclared weight, not TUNE-selected', 'weight_candidates':scores})
    complete=[c for c in candidates if c['complete_candidate']]
    if not complete: raise ValueError('NO_COMPLETE_CANDIDATE_FOR_'+kind.upper())
    minimum=min(c['score'] for c in complete)
    selected=next(c for c in complete if c['score']<=minimum+1e-12)
    return {'kind':kind,'selected':selected['name'],'candidates':candidates,
        'rule':'TUNE-only normalized MSE / RAW TUNE MSE log mean over 2 datasets x 2 seeds; 1e-12 tie retains declared method order'}


def source_snapshot(output):
    paths=[Path(__file__),HERE/'alignment.py',HERE/'test_engine.py',HERE/'PLAN_KO.md',OLD/'experiment.py',
        OLD/'diagnose_constant_gradient.py',ex.OLD/'component_experiment.py',ROOT/'research/past_only_repr_diagnosis_20261012/common.py']
    folder=output/'sources'; folder.mkdir()
    records={}
    for path in paths:
        digest=ex.sha(path)
        snapshot=folder/path.name
        with snapshot.open('xb') as f: f.write(path.read_bytes())
        assert ex.sha(snapshot)==digest
        records[str(path.relative_to(ROOT))]=digest
    return records


def run(tag,mode):
    configure()
    protected=ex.access_guard()
    output=HERE/('run_'+tag); output.mkdir(exist_ok=False)
    folder=CACHE/('run_'+tag); folder.mkdir(parents=True,exist_ok=False)
    limit=Limit(tag)
    result={'status':'RUNNING','tag':tag,'mode':mode,'cases':[],'fixtures':{},'protected_test_guarded':protected,
        'primary_metric':'Normalized observed channel-macro MSE','source_hashes':source_snapshot(output),
        'device':torch.cuda.get_device_name(),'dtype':'float32','tf32':False}
    try:
        started=time.perf_counter(); bolt=ex.previous.FrozenBolt(); replace_normalizer(bolt)
        result['bolt_load_wall_seconds']=time.perf_counter()-started
        result['normalizer_fixture']=normalizer_fixture(); result['bolt_initial_hash']=bolt.initial_hash
        if mode in ('preflight','dev'):
            datasets={}; historydict={}; raw={}; means={}; bases={}; teachrecords={}
            for dataset in ('jena','robin'):
                data=ex.load_data(dataset,sizes=(256,64,None)); datasets[dataset]=data
                history,target,observed,_=ex.arrays(data,L); historydict[dataset]=(history,target,observed)
                result['fixtures'][dataset]=fixture(data,history,target,observed,bolt,limit)
                contract={'dataset':dataset,'source':str(Path(data['source']).relative_to(ROOT)),'source_sha256':data['source_hash'],
                    'columns':data['columns'],'mean':data['mean'],'std':data['std'],'origins':data['origins'],
                    'origin_hashes':{s:ex.ahash(o) for s,o in data['origins'].items()},'bounds':{s:data[s+'_bounds'] for s in ('fit','tune','val')},
                    'input_boundary':'Reset each actual 512-window at FIT mean; causal observed forward fill only'}
                write(output/(dataset+'_contract.json'),contract)
                if mode=='preflight': continue
                datafolder=folder/dataset; datafolder.mkdir(); dataoutput=output/dataset; dataoutput.mkdir()
                rawpred={s:predict(history[s],None,bolt,limit) for s in history}
                rawcase={'dataset':dataset,'stage':'RAW','label':'RAW','methods':[]}
                add_prediction(rawcase,datafolder,'RAW_BOLT',data,rawpred,target,observed)
                write(dataoutput/'RAW.json',rawcase); result['cases'].append(rawcase)
                raw[dataset]=rawcase['methods'][0]['scores']['tune']['normalized']['mse']
                mean,pca,_=ex.initialization(data,max(1,data['x'].shape[1]//4)); means[dataset]=mean
                bases[dataset]={}; teachrecords[dataset]={}
                for seed in SEEDS:
                    teacher,rec=teacher_basis(data,history,seed,bolt,limit,datafolder)
                    bases[dataset][seed]={'PCA':pca,'TEACHER':teacher}; teachrecords[dataset][seed]=rec
                    write(dataoutput/f'TEACHER_s{seed}.json',rec)
            if mode=='dev':
                s1=[]
                for dataset,data in datasets.items():
                    history,target,observed=historydict[dataset]
                    for seed in SEEDS:
                        for architecture in ARCHITECTURES:
                            basis=bases[dataset][seed]['PCA' if architecture.endswith('PCA') else 'TEACHER']
                            case=safe_fit_case(result,data,history,target,observed,seed,architecture,basis,means[dataset],bolt,folder/dataset,output/dataset,limit,'S1')
                            if case is not None: s1.append(case); result['cases'].append(case)
                selection=choose(s1,raw,'architecture'); result['S1_selection']=selection
                for dataset in datasets:
                    for seed in SEEDS:
                        matched=[c for c in s1 if c['dataset']==dataset and c['seed']==seed]
                        assert len({c['training']['order_sha256'] for c in matched})<=1
                write(output/'S1_SELECTION.json',selection)
                s2=[]
                architecture=selection['selected']
                for dataset,data in datasets.items():
                    history,target,observed=historydict[dataset]
                    for seed in SEEDS:
                        # Auxiliary coordinates always use fixed teacher basis, even if S1 PCA wins.
                        initbasis=bases[dataset][seed]['PCA' if architecture.endswith('PCA') else 'TEACHER']
                        for method in METHODS:
                            for weight in ([0.] if method=='BASE_Q' else WEIGHTS):
                                case=safe_fit_case(result,data,history,target,observed,seed,architecture,initbasis,means[dataset],bolt,folder/dataset,output/dataset,limit,'S2',method,weight,bases[dataset][seed]['TEACHER'])
                                if case is not None: s2.append(case); result['cases'].append(case)
                        same=[c for c in s2 if c['dataset']==dataset and c['seed']==seed]
                        assert len({c['training']['initial_codec_hash'] for c in same})<=1
                        assert len({c['training']['order_sha256'] for c in same})<=1
                alignmentselection=choose(s2,raw,'alignment'); result['S2_selection']=alignmentselection
                write(output/'S2_SELECTION.json',alignmentselection)
                lock={'status':'METHODS_LOCKED_BEFORE_NEW_RAW_DOWNLOAD','created_unix':time.time(),'source_hashes':result['source_hashes'],
                    'architecture':architecture,'S1_selection':selection,'S2_selection':alignmentselection,'K_rule':'max(1,floor(C/4))',
                    'updates':600,'seeds':SEEDS,'L':L,'H':H,'weights':{c['name']:c['chosen_weight'] for c in alignmentselection['candidates']},
                    'S3_S1_methods':['DIRECT_PCA','ANCHOR_PCA',architecture],'S3_S2_methods':METHODS,
                    'selection_rule':'TUNE normalized MSE only; all checkpoints and native/ridge fixed before opening EVAL targets'}
                write(HERE/'METHOD_LOCK.json',lock); result['method_lock']={'path':str((HERE/'METHOD_LOCK.json').relative_to(ROOT)),'sha256':ex.sha(HERE/'METHOD_LOCK.json')}
        elif mode=='new':
            raise NotImplementedError('New-data API integration follows locked development execution')
        for path,digest in result['source_hashes'].items():
            if ex.sha(ROOT/path)!=digest: raise AssertionError('SOURCE_CHANGED_DURING_EXECUTION '+path)
        assert ex.previous.state_hash(bolt.model)==bolt.initial_hash
        result['bolt_final_hash']=ex.previous.state_hash(bolt.model)
        result['status']='PARTIAL_FAILED_PATHS' if result.get('failed_paths') else 'COMPLETE'
        limit.check(); limit.close(result['status']); result['resources']=limit.record
        result['training_fits']=sum('training' in c for c in result['cases'])
        write(output/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'training_fits':result['training_fits'],'resources':limit.record}),flush=True)
    except BaseException as error:
        result.update(status='FAILED_PRESERVED',error=str(error),traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED',str(error)); result['resources']=limit.record
        write(output/'FAILURE.json',result)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); parser.add_argument('--mode',choices=('preflight','dev','new'),required=True)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('SAFE_TAG_REQUIRED')
    run(args.tag,args.mode)
