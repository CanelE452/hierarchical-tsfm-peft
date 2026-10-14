"""Frozen-codec native and raw ridge controls; deployment costs after all training."""
import time
import numpy as np
import torch
from torch import nn
import psutil
import engine as en


class EffectiveRidge(nn.Module):
    def __init__(self,weight,bias,channels):
        super().__init__()
        self.register_buffer('weight',en.tensor(weight))
        self.register_buffer('bias',en.tensor(bias))
        self.channels=channels
    def forward(self,features):
        return (features@self.weight+self.bias).reshape(len(features),en.H,self.channels)


def load_codec(case):
    artifact=case['training']['selected_checkpoint']
    assert en.ex.sha(en.ROOT/artifact['path'])==artifact['sha256']
    saved=torch.load(en.ROOT/artifact['path'],map_location='cpu',weights_only=True)
    basis=saved['codec']['fixed_basis'].numpy()
    codec=en.Codec(np.zeros(basis.shape[0],np.float32),basis,case['seed'],case['architecture']).to('cuda')
    codec.load_state_dict(saved['codec'])
    codec.eval().requires_grad_(False)
    assert en.ex.previous.state_hash(codec)==case['training']['selected_codec_hash']
    return codec


def fit_ridge(data,history,target,observed,base,limit):
    started=time.perf_counter()
    features={s:en.ex.block_features(x) for s,x in history.items()}
    residual={s:target[s]-base[s] for s in target}
    _,selection,fits=en.ex.ridge_grouped(features['fit'],residual['fit'],observed['fit'],features['tune'],residual['tune'],
        observed['tune'],features['tune'][:0],score_scale=np.ones(target['fit'].shape[-1]),budget=limit)
    old=en.ex.previous.TorchRidge(fits,features['fit'].shape[1],target['fit'].shape[-1]).eval()
    model=EffectiveRidge(old.weight,old.bias,target['fit'].shape[-1]).eval()
    with torch.no_grad(): pred={s:base[s]+model(en.tensor(features[s])).cpu().numpy() for s in history}
    selection.update(train_wall_seconds=time.perf_counter()-started,selection_rule='TUNE normalized observed channel-macro MSE',
        FIT_base_predictions_in_sample=True,raw_input_support=512,registered_parameters=model.weight.numel()+model.bias.numel())
    return model,pred,selection


def fit_native(data,history,target,observed,codec,seed,folder,limit):
    started=time.perf_counter()
    codec.eval().requires_grad_(False)
    with torch.no_grad():
        latent={s:torch.cat([codec.encode(en.tensor(x[i:i+32])) for i in range(0,len(x),32)]) for s,x in history.items() if s in ('fit','tune')}
    model=en.ex.NativeShared(512).to('cuda')
    initial=en.ex.previous.state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    order=np.random.default_rng(seed).integers(0,len(history['fit']),size=(600,4))
    curve,checkpoints=[],[]
    def forward(split,indices=None):
        z=latent[split] if indices is None else latent[split][indices]
        future=model(z); result=codec.decode(future)
        if codec.anchor:
            x=history[split] if indices is None else history[split][indices]
            result=en.tensor(x[:,-1:])+result-codec.decode(z[:,-1:])
        return result
    for update in range(601):
        if update in en.CHECKPOINTS:
            with torch.no_grad(): p=forward('tune').cpu().numpy()
            artifact=en.save(folder/f'native_u{update}.pt',en.ex.previous.clone_state(model))
            checkpoints.append({'update':update,**artifact,'tune':en.metrics(data,p,target['tune'],observed['tune'])})
        if update==600: break
        indices=order[update]
        optimizer.zero_grad(set_to_none=True)
        main=en.ex.previous.macro_loss(forward('fit',indices),en.tensor(target['fit'][indices]),en.ex.boolean(observed['fit'][indices]))
        main.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(main) or not torch.isfinite(norm): raise ValueError('NATIVE_NONFINITE_LOSS_GRADIENT')
        optimizer.step(); curve.append({'update':update+1,'loss':float(main.detach()),'gradient_norm':float(norm)})
        if update%20==0: limit.check()
    selected=min(checkpoints,key=lambda cp:cp['tune']['normalized']['mse'])
    model.load_state_dict(torch.load(en.ROOT/selected['path'],map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False)
    pred={s:en.predict(x,codec,model,limit) for s,x in history.items()}
    return model,pred,{'updates':600,'train_wall_seconds':time.perf_counter()-started,'checkpoints':checkpoints,
        'selected_update':selected['update'],'selected_checkpoint':selected,'initial_model_hash':initial,'order_sha256':en.ex.ahash(order),
        'encoder_decoder_frozen':True,'E_D_optimized_for_native':False,'additional_training':True,
        'parameters':sum(p.numel() for p in model.parameters()),'curve':curve,'selection_rule':'TUNE normalized MSE only'}


def raw_controls(data,history,target,observed,bolt,folder,output,limit):
    folder.mkdir(parents=True,exist_ok=False)
    case={'dataset':data['dataset'],'stage':'CONTROLS','label':'RAW_CONTROLS','methods':[],'seed':None,'K':target['fit'].shape[-1],'L':512}
    raw={s:en.predict(x,None,bolt,limit) for s,x in history.items()}
    en.add_prediction(case,folder,'RAW_BOLT',data,raw,target,observed)
    zeros={s:np.zeros_like(v) for s,v in target.items()}
    only,onlypred,onlyselection=fit_ridge(data,history,target,observed,zeros,limit)
    en.add_prediction(case,folder,'RAW_JOINT_RIDGE',data,onlypred,target,observed,{'correction':onlyselection})
    correction,corrected,correctionselection=fit_ridge(data,history,target,observed,raw,limit)
    en.add_prediction(case,folder,'RAW_BOLT_RAW_R',data,corrected,target,observed,{'correction':correctionselection})
    selfmodel,selfpred,selfselection=en.ex.self_ridge(data,history,target,observed,limit)
    en.add_prediction(case,folder,'RAW_SELF33',data,selfpred,target,observed,{'selection':selfselection})
    artifact=en.save(folder/'models.pt',{'ridge_only':en.ex.previous.clone_state(only),'raw_bolt_ridge':en.ex.previous.clone_state(correction),
        'self_ridge':en.ex.previous.clone_state(selfmodel)})
    case['selected_models']=artifact
    en.write(output/'RAW_CONTROLS.json',case)
    return case


def codec_controls(data,history,target,observed,case,bolt,folder,output,limit):
    folder.mkdir(parents=True,exist_ok=False)
    codec=load_codec(case)
    label=case['label']+'_CONTROLS'
    control={'dataset':data['dataset'],'stage':'CONTROLS','label':label,'architecture':case['architecture'],'seed':case['seed'],
        'K':case['K'],'L':512,'source_case':case['label'],'source_checkpoint':case['training']['selected_checkpoint'],'methods':[]}
    base={s:en.predict(x,codec,bolt,limit) for s,x in history.items()}
    en.add_prediction(control,folder,'BOLT',data,base,target,observed)
    correction,corrected,selection=fit_ridge(data,history,target,observed,base,limit)
    en.add_prediction(control,folder,'BOLT_RAW_R',data,corrected,target,observed,{'correction':selection})
    native,npred,native_record=fit_native(data,history,target,observed,codec,case['seed'],folder,limit)
    en.add_prediction(control,folder,'NATIVE',data,npred,target,observed,{'native_training':native_record})
    native_r,ncorrected,nselection=fit_ridge(data,history,target,observed,npred,limit)
    en.add_prediction(control,folder,'NATIVE_RAW_R',data,ncorrected,target,observed,{'correction':nselection})
    control['selected_models']=en.save(folder/'models.pt',{'native':en.ex.previous.clone_state(native),'bolt_ridge':en.ex.previous.clone_state(correction),
        'native_ridge':en.ex.previous.clone_state(native_r),'codec':en.ex.previous.clone_state(codec)})
    en.write(output/(label+'.json'),control)
    return control


def models_for_case(case,source_cases):
    name=case['label']
    if name=='RAW_CONTROLS':
        assert en.ex.sha(en.ROOT/case['selected_models']['path'])==case['selected_models']['sha256']
        saved=torch.load(en.ROOT/case['selected_models']['path'],map_location='cpu',weights_only=True)
        c=case['K']
        ridge=lambda state: EffectiveRidge(state['weight'],state['bias'],c).eval()
        ss=saved['self_ridge']
        selfmodel=en.ex.SelfRidge(np.zeros(tuple(ss['weights'].shape)),np.zeros(tuple(ss['biases'].shape)))
        selfmodel.load_state_dict(ss)
        return {'codec':None,'ridge_only':ridge(saved['ridge_only']),'raw_bolt_ridge':ridge(saved['raw_bolt_ridge']),'self':selfmodel.eval()}
    if case['stage']=='CONTROLS':
        source=next(c for c in source_cases if c['dataset']==case['dataset'] and c['label']==case['source_case'])
        codec=load_codec(source)
        assert en.ex.sha(en.ROOT/case['selected_models']['path'])==case['selected_models']['sha256']
        saved=torch.load(en.ROOT/case['selected_models']['path'],map_location='cpu',weights_only=True)
        assert all(torch.equal(value.detach().cpu(),saved['codec'][key]) for key,value in codec.state_dict().items())
        native=en.ex.NativeShared(512).to('cuda'); native.load_state_dict(saved['native']); native.eval().requires_grad_(False)
        ridge=lambda s: EffectiveRidge(s['weight'],s['bias'],codec.codec.project.in_features).eval()
        return {'codec':codec,'native':native,'bolt_ridge':ridge(saved['bolt_ridge']),'native_ridge':ridge(saved['native_ridge'])}
    return {'codec':load_codec(case)}


def prediction_method(history,name,loaded,bolt,limit):
    codec=loaded['codec']
    if name in ('RAW_BOLT','RAW_BOLT_RAW_R','BOLT','BOLT_RAW_R'): forecaster=bolt
    elif name in ('NATIVE','NATIVE_RAW_R'): forecaster=loaded['native']
    else: forecaster=None
    if name=='RAW_SELF33':
        with torch.no_grad(): return loaded['self'](en.tensor(history)).cpu().numpy()
    base=np.zeros((len(history),48,history.shape[-1]),np.float32) if forecaster is None else en.predict(history,codec,forecaster,limit)
    ridgekey={'RAW_JOINT_RIDGE':'ridge_only','RAW_BOLT_RAW_R':'raw_bolt_ridge','BOLT_RAW_R':'bolt_ridge','NATIVE_RAW_R':'native_ridge'}.get(name)
    if ridgekey:
        with torch.no_grad(): base+=loaded[ridgekey](en.tensor(en.ex.block_features(history))).cpu().numpy()
    return base


def deployment(data,physical,observed,name,loaded,bolt):
    x=en.ex.preprocess(physical,observed,data['mean'],data['std'])
    tensor=en.tensor(x)
    codec=loaded['codec']
    if name=='RAW_SELF33': future=loaded['self'](tensor)
    elif name=='RAW_JOINT_RIDGE': future=loaded['ridge_only'](en.ex.block_features(tensor))
    else:
        forecaster=loaded['native'] if name.startswith('NATIVE') else bolt
        future=forecaster(tensor) if codec is None else codec.forward_path(tensor,forecaster)
        key={'RAW_BOLT_RAW_R':'raw_bolt_ridge','BOLT_RAW_R':'bolt_ridge','NATIVE_RAW_R':'native_ridge'}.get(name)
        if key: future=future+loaded[key](en.ex.block_features(tensor))
    return (future*en.tensor(data['std'])+en.tensor(data['mean'])).cpu().numpy()


def cost(data,case,method,loaded,bolt,limit,split):
    path=en.ROOT/method['prediction']['path']; assert en.ex.sha(path)==method['prediction']['sha256']
    with np.load(path,allow_pickle=False) as z: expected=z[split+'_prediction']
    records=[]
    for batch in (1,8):
        origins=data['origins'][split][:batch]
        if split=='eval' and 'evaluation_history_physical' in data:
            physical=data['evaluation_history_physical'][:batch]; mask=data['evaluation_history_observed'][:batch]
        else:
            physical=np.stack([data['physical'][o-512:o] for o in origins]); mask=np.stack([data['mask'][o-512:o] for o in origins])
        with torch.no_grad():
            for _ in range(2): deployment(data,physical,mask,method['method'],loaded,bolt)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); timings=[]
            for _ in range(7):
                torch.cuda.synchronize(); started=time.perf_counter()
                actual=deployment(data,physical,mask,method['method'],loaded,bolt)
                torch.cuda.synchronize(); timings.append(time.perf_counter()-started)
        reference=expected[:batch]*data['std']+data['mean']
        passed=bool(np.all(np.abs(actual-reference)<=2e-3+2e-5*np.abs(reference)))
        if not passed: raise ValueError('COST_DEPLOYMENT_PREDICTION_PARITY')
        records.append({'batch':batch,'warmup':2,'repeats':7,'seconds':timings,'median_seconds':float(np.median(timings)),
            'median_seconds_per_window':float(np.median(timings))/batch,'physical_max_abs_vs_science':float(np.max(np.abs(actual-reference))),
            'parity_pass':passed,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'resident_rss_bytes':psutil.Process().memory_info().rss,
            'includes':'CPU512 causalFF/FITnorm/H2D/E/F/D/anchor/optionalrawR/inversenorm/D2H/sync; excludes disk/model load and removed alignment teacher/q/prototypes',
            'co_resident_memory_note':'Bolt/current models resident; not exclusive incremental method memory'})
        limit.check()
    return records
