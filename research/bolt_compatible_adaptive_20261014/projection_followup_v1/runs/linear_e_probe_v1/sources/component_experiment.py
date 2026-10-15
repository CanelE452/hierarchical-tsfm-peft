"""Bounded Stage B: train a causal codec through frozen Bolt, then test corrections."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/forecast_component_study_20261013'
OLD = ROOT / 'research/past_only_repr_diagnosis_20261012'
sys.path.insert(0, str(OLD))
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from common import access_guard, ahash, sha, read, jsonable, load_data, windows, masks, physical_score, ridge_grouped, L, H

DEVICE = 'cuda'
CHECKPOINTS = (0, 100, 200, 400, 600)
SEEDS = (92601, 92602)
BOLT_DIRECTORY = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(jsonable(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Limit:
    def __init__(self, tag):
        self.start = time.perf_counter()
        self.process = psutil.Process()
        self.record = {'pid': os.getpid(), 'tag': tag, 'gpu_process': True,
                       'started_unix': time.time(), 'status': 'RUNNING', 'wall_seconds': 0.,
                       'peak_sampled_rss_bytes': 0, 'peak_allocated_vram_bytes': 0}
        self.path = HERE / 'stage_b_resources' / f'{tag}_{os.getpid()}_{time.time_ns()}.json'
        self.path.parent.mkdir(exist_ok=True)
        self.last_storage = -60.
        self.check()

    def check(self):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.record['peak_sampled_rss_bytes'] = max(self.record['peak_sampled_rss_bytes'], self.process.memory_info().rss)
        self.record['peak_allocated_vram_bytes'] = max(self.record['peak_allocated_vram_bytes'], torch.cuda.max_memory_allocated())
        if self.record['wall_seconds']-self.last_storage >= 30:
            self.record['new_cache_bytes'] = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
            self.record['new_research_bytes'] = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
            self.last_storage = self.record['wall_seconds']
        previous = sum(read(p)['wall_seconds'] for p in self.path.parent.glob('*.json') if p != self.path)
        self.record['cumulative_gpu_process_wall_seconds'] = previous+self.record['wall_seconds']
        self._flush()
        if self.record['cumulative_gpu_process_wall_seconds'] >= 3600:
            raise RuntimeError('STAGE_B_GPU_PROCESS_WALL_LIMIT_3600S')
        if max(self.record['peak_sampled_rss_bytes'], self.record['peak_allocated_vram_bytes']) > 8*1024**3:
            raise RuntimeError('STAGE_B_PROCESS_RAM_OR_ALLOCATED_VRAM_LIMIT_8GIB')
        if self.record['new_cache_bytes']+self.record['new_research_bytes'] > 4*1024**3:
            raise RuntimeError('CAMPAIGN_NEW_STORAGE_LIMIT_4GIB')

    def _flush(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def close(self, status, error=None):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.record['status'] = status
        self.record['error'] = error
        self._flush()


class FrozenBolt:
    def __init__(self):
        from chronos import ChronosBoltPipeline
        self.pipeline = ChronosBoltPipeline.from_pretrained(str(BOLT_DIRECTORY), device_map=DEVICE,
                                                            torch_dtype=torch.float32, local_files_only=True)
        self.model = self.pipeline.model.eval().requires_grad_(False)
        self.median = list(self.model.chronos_config.quantiles).index(.5)
        self.initial_hash = state_hash(self.model)

    def __call__(self, z):
        batch, length, channels = z.shape
        rows = z.transpose(1, 2).reshape(batch*channels, length)
        future = self.model(context=rows).quantile_preds[:, self.median, :H]
        return future.reshape(batch, channels, H).transpose(1, 2)


class Codec(nn.Module):
    def __init__(self, pca_mean, basis, seed, nonlinear=True):
        super().__init__()
        torch.manual_seed(seed)
        channels, k = basis.shape
        self.project = nn.Linear(channels, k)
        self.decode = nn.Linear(k, channels)
        self.nonlinear = nonlinear
        if nonlinear:
            self.input = nn.Conv1d(channels, 16, 1)
            self.temporal = nn.ModuleList(nn.Conv1d(16, 16, 3, dilation=d) for d in (1,2,4,8,16,32))
            self.output = nn.Conv1d(16, k, 1)
        with torch.no_grad():
            self.project.weight.copy_(torch.from_numpy(basis.T.copy()))
            self.project.bias.copy_(torch.from_numpy((-pca_mean@basis).copy()))
            self.decode.weight.copy_(torch.from_numpy(basis.copy()))
            self.decode.bias.copy_(torch.from_numpy(pca_mean.copy()))
            if nonlinear:
                self.output.weight.zero_()
                self.output.bias.zero_()

    def encode(self, x):
        z = self.project(x)
        if self.nonlinear:
            v = F.gelu(self.input(x.transpose(1,2)))
            for layer in self.temporal:
                v = F.gelu(layer(F.pad(v, (2*layer.dilation[0], 0))))
            z = z+self.output(v).transpose(1,2)
        return z


class NativeShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(L, H)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.weight[:, -1] = 1
            self.layer.bias.zero_()

    def forward(self, z):
        return self.layer(z.transpose(1,2)).transpose(1,2)


def macro_loss(pred, target, observed):
    count = observed.sum((0,1))
    error = torch.where(observed, (pred-target).square(), 0.).sum((0,1))
    return (error/count.clamp_min(1))[count > 0].mean()


def block_features(x):
    shape = (len(x), 32, 16, x.shape[-1])
    if isinstance(x, np.ndarray):
        return np.concatenate((x.reshape(shape).mean(2), x[:,-1:]), axis=1).reshape(len(x), -1)
    return torch.cat((x.reshape(shape).mean(2), x[:,-1:]), dim=1).reshape(len(x), -1)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def mask_tensor(x):
    return torch.as_tensor(x, dtype=torch.bool, device=DEVICE)


def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def predict(histories, codec, forecaster, limit, decoder=None, batch=8):
    rows = []
    with torch.no_grad():
        for begin in range(0, len(histories), batch):
            x = tensor(histories[begin:begin+batch])
            z = x if codec is None else codec.encode(x)
            future = forecaster(z)
            if codec is not None:
                future = (codec.decode if decoder is None else decoder)(future)
            rows.append(future.cpu().numpy())
            limit.check()
    return np.concatenate(rows)


def persist(z):
    return z[:,-1:].expand(-1,H,-1)


def pca_initialization(data):
    lo, hi = data['fit_bounds']
    x = data['x'][lo:hi].astype(np.float64)
    mean = x.mean(0)
    covariance = (x-mean).T@(x-mean)/len(x)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    k = x.shape[1]//4
    basis = eigenvectors[:,-k:][:,::-1].copy()
    for j in range(k):
        if basis[np.argmax(np.abs(basis[:,j])),j] < 0:
            basis[:,j] *= -1
    return mean.astype(np.float32), basis.astype(np.float32), {'fit_rows':hi-lo, 'k':k,
        'explained_variance_fraction':float(eigenvalues[-k:].sum()/eigenvalues.sum()),
        'fit_scope':'Unique FIT normalized and causally filled timesteps; no TUNE/VAL'}


def fixtures(bolt, codec, histories, targets, observed, limit):
    ramp = torch.linspace(-2,3,L,device=DEVICE).reshape(1,L,1)
    with torch.no_grad():
        own = bolt(ramp)[:,:,0].cpu()
        official = bolt.pipeline.predict(ramp[:,:,0].cpu(), prediction_length=H)[:,bolt.median]
        parity = float((own-official).abs().max())
    if parity != 0:
        raise AssertionError('OFFICIAL_BOLT_PARITY')
    x = tensor(histories[:2]).requires_grad_(True)
    before = state_hash(bolt.model)
    loss = macro_loss(codec.decode(bolt(codec.encode(x))), tensor(targets[:2]), mask_tensor(observed[:2]))
    loss.backward()
    norms = {'input':float(x.grad.norm()),
             'encoder':float(torch.stack([p.grad.norm() for name,p in codec.named_parameters() if not name.startswith('decode.') and p.grad is not None]).norm()),
             'decoder':float(torch.stack([p.grad.norm() for p in codec.decode.parameters() if p.grad is not None]).norm())}
    if not all(np.isfinite(v) and v>0 for v in norms.values()):
        raise AssertionError('FINITE_NONZERO_INPUT_ENCODER_DECODER_GRADIENT')
    if any(p.grad is not None or p.requires_grad for p in bolt.model.parameters()):
        raise AssertionError('BACKBONE_GRADIENT_OR_TRAINABILITY')
    initial_codec = clone_state(codec)
    optimizer = torch.optim.Adam(codec.parameters(),lr=1e-3)
    optimizer.step()
    encoder_change = max(float((p.detach().cpu()-initial_codec[name]).abs().max()) for name,p in codec.named_parameters() if not name.startswith('decode.'))
    decoder_change = max(float((p.detach().cpu()-initial_codec[name]).abs().max()) for name,p in codec.named_parameters() if name.startswith('decode.'))
    if min(encoder_change,decoder_change)<=0:
        raise AssertionError('ACTUAL_OPTIMIZER_UPDATE_DID_NOT_CHANGE_ENCODER_AND_DECODER')
    codec.load_state_dict(initial_codec)
    codec.zero_grad(set_to_none=True)
    with torch.no_grad():
        saved_output = codec.output.weight.clone()
        codec.output.weight.fill_(.03125)
        a = tensor(histories[:2])
        altered = a.clone()
        altered[:,256:] = altered[:,256:].flip(1)+999
        causal_delta = float((codec.encode(a)[:,:256]-codec.encode(altered)[:,:256]).abs().max())
        cropped_delta = float((codec.encode(a)[:,:256]-codec.encode(a[:,:256])).abs().max())
        positive = a.clone(); positive[:,255] += 1
        positive_delta = float((codec.encode(a)[:,255]-codec.encode(positive)[:,255]).abs().max())
        codec.output.weight.copy_(saved_output)
    if max(causal_delta,cropped_delta)>2e-5 or positive_delta<=0:
        raise AssertionError('CAUSAL_COUNTERFACTUAL')
    after = state_hash(bolt.model)
    if before != after:
        raise AssertionError('BACKBONE_HASH_CHANGED')
    limit.check()
    return {'official_median_max_abs':parity, 'gradient_norms':norms, 'finite_loss':float(loss.detach()),
            'temporary_optimizer_step_encoder_max_abs':encoder_change,'temporary_optimizer_step_decoder_max_abs':decoder_change,
            'temporary_optimizer_state_restored':True,
            'future_perturbation_past_max_abs':causal_delta, 'prefix_full_max_abs':cropped_delta,
            'positive_control_max_abs':positive_delta, 'bolt_hash_before':before,'bolt_hash_after':after,
            'trainable_bolt_parameters':0, 'pass':True}


def train_codec(data, histories, targets, observed, codec, bolt, objective, seed, folder, limit):
    started = time.perf_counter()
    codec.train().requires_grad_(True)
    initial = state_hash(codec)
    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-3)
    order = np.random.default_rng(seed).integers(0,len(histories['fit']), size=(600,4))
    history_observed = masks(data,'fit',False)
    curve, checkpoints = [], []
    best = None
    for update in range(601):
        if update in CHECKPOINTS:
            prediction = predict(histories['tune'],codec,bolt,limit)
            metric = physical_score(data,prediction,'tune')
            path = folder/f'{objective}_u{update}.pt'
            with path.open('xb') as f:
                torch.save({'codec':clone_state(codec), 'update':update,'seed':seed,'objective':objective},f)
            checkpoints.append({'update':update,'tune':metric,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            if best is None or metric['mse']<best[0]:
                best = (metric['mse'],update,clone_state(codec))
            print(json.dumps({'dataset':data['dataset'],'seed':seed,'objective':objective,'update':update,
                              'tune_mse':metric['mse'],'train_wall_seconds':time.perf_counter()-started}),flush=True)
        if update == 600:
            break
        ids = order[update]
        x = tensor(histories['fit'][ids])
        optimizer.zero_grad(set_to_none=True)
        if objective=='REC':
            prediction = codec.decode(codec.encode(x))
            loss = macro_loss(prediction,x,mask_tensor(history_observed[ids]))
        else:
            prediction = codec.decode(bolt(codec.encode(x)))
            loss = macro_loss(prediction,tensor(targets['fit'][ids]),mask_tensor(observed['fit'][ids]))
        if not torch.isfinite(loss):
            raise ValueError('NONFINITE_TRAINING_LOSS')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(codec.parameters(),1.)
        if not torch.isfinite(norm):
            raise ValueError('NONFINITE_CODEC_GRADIENT')
        optimizer.step()
        curve.append({'update':update+1,'loss':float(loss.detach()),'gradient_norm_before_clipping':float(norm)})
        if update%10==0:
            limit.check()
    codec.load_state_dict(best[2])
    codec.eval().requires_grad_(False)
    record = {'objective':objective,'seed':seed,'initial_codec_hash':initial,'selected_update':best[1],
              'selected_codec_hash':state_hash(codec),'checkpoint_selection':'TUNE physical macro MSE after frozen Bolt; earliest tie',
              'updates':600,'batch_size':4,'learning_rate':1e-3,'order_sha256':ahash(order),'batch_indices':order.tolist(),
              'origins_seen':sorted(set(data['origins']['fit'][order].flatten().tolist())),
              'checkpoints':checkpoints,'curve':curve,'train_wall_seconds':time.perf_counter()-started,
              'trainable_parameters':sum(p.numel() for p in codec.parameters())}
    return record


def train_native(data, histories, targets, observed, codec, seed, folder, limit):
    started = time.perf_counter()
    codec.eval().requires_grad_(False)
    with torch.no_grad():
        latent = {s:torch.cat([codec.encode(tensor(x[i:i+32])) for i in range(0,len(x),32)]) for s,x in histories.items() if s!='val'}
    native = NativeShared().to(DEVICE)
    optimizer = torch.optim.Adam(native.parameters(),lr=1e-3)
    order = np.random.default_rng(seed).integers(0,len(histories['fit']),size=(600,4))
    curve, checkpoints = [], []
    best = None
    for update in range(601):
        if update in CHECKPOINTS:
            with torch.no_grad():
                prediction = codec.decode(native(latent['tune'])).cpu().numpy()
            metric = physical_score(data,prediction,'tune')
            checkpoints.append({'update':update,'tune':metric})
            if best is None or metric['mse']<best[0]:
                best=(metric['mse'],update,clone_state(native))
        if update == 600:
            break
        ids=order[update]
        optimizer.zero_grad(set_to_none=True)
        loss=macro_loss(codec.decode(native(latent['fit'][ids])),tensor(targets['fit'][ids]),mask_tensor(observed['fit'][ids]))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(native.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError('NONFINITE_NATIVE_LOSS_OR_GRADIENT')
        optimizer.step()
        curve.append({'update':update+1,'loss':float(loss.detach())})
        if update%20==0:
            limit.check()
    native.load_state_dict(best[2]);native.eval().requires_grad_(False)
    return native, {'selected_update':best[1],'checkpoints':checkpoints,'curve':curve,
                    'updates':600,'train_wall_seconds':time.perf_counter()-started,'order_sha256':ahash(order),
                    'trainable_parameters':sum(p.numel() for p in native.parameters()),'encoder_decoder_frozen':True,
                    'latent_history_length':L,'shared_across_latent_channels':True}


def refit_decoder(data, histories, codec, limit):
    with torch.no_grad():
        z = torch.cat([codec.encode(tensor(histories['fit'][i:i+32])) for i in range(0,len(histories['fit']),32)]).cpu().numpy().reshape(-1,codec.project.out_features)
    x=histories['fit'].reshape(-1,histories['fit'].shape[-1]).astype(np.float64)
    valid=masks(data,'fit',False).reshape(x.shape)
    decoder=nn.Linear(z.shape[1],x.shape[1]).to(DEVICE)
    coefs=[];biases=[]
    for c in range(x.shape[1]):
        zz=z[valid[:,c]].astype(np.float64); yy=x[valid[:,c],c]
        zm=zz.mean(0);ym=yy.mean()
        coef=np.linalg.solve((zz-zm).T@(zz-zm)+len(zz)*1e-4*np.eye(z.shape[1]),(zz-zm).T@(yy-ym))
        coefs.append(coef);biases.append(ym-zm@coef)
    with torch.no_grad():
        decoder.weight.copy_(tensor(np.stack(coefs)));decoder.bias.copy_(tensor(np.array(biases)))
    limit.check()
    return decoder.eval().requires_grad_(False), {'lambda':1e-4,'definition':'Pointwise mean observed SSE + lambda ||W||^2; unpenalized intercept; no feature scaling',
        'fit_history_positions':len(z),'unique_input_windows':len(histories['fit']),'observed_counts':valid.sum(0).tolist(),
        'selection':'Fixed lambda, no TUNE/VAL fitting'}


class TorchRidge(nn.Module):
    def __init__(self, fits, features, channels):
        super().__init__()
        weight=np.zeros((features,H*channels),np.float64);bias=np.zeros(H*channels,np.float64)
        for columns, model in fits:
            coef=model.coef if model.coef is not None else model.train.T@model.dual
            effective=coef/model.std[:,None]
            weight[:,columns]=effective;bias[columns]=model.ym-model.mean@effective
        self.register_buffer('weight',tensor(weight.astype(np.float32)))
        self.register_buffer('bias',tensor(bias.astype(np.float32)))
        self.channels=channels

    def forward(self,x):
        return (x@self.weight+self.bias).reshape(len(x),H,self.channels)


def correction(data, feature, targets, observed, base, limit):
    started=time.perf_counter()
    residual={s:targets[s]-base[s] for s in targets}
    pe,selection,fits=ridge_grouped(feature['fit'],residual['fit'],observed['fit'],feature['tune'],residual['tune'],observed['tune'],feature['val'],score_scale=data['std'],budget=limit)
    model=TorchRidge(fits,feature['fit'].shape[1],targets['fit'].shape[-1]).eval()
    with torch.no_grad():
        pred={s:base[s]+model(tensor(feature[s])).cpu().numpy() for s in targets}
    delta=float(np.max(np.abs((pred['val']-base['val'])-pe)))
    selection.update({'train_wall_seconds':time.perf_counter()-started,'float32_effective_ridge_max_abs_vs_float64':delta,
                      'features':feature['fit'].shape[1],'parameter_count':model.weight.numel()+model.bias.numel(),
                      'past_summary':'32 nonoverlapping length16 means plus latest observation; full512 support'})
    return model,pred,selection


def preprocess(physical, observed, mean, std):
    filled=np.empty_like(physical,dtype=np.float32)
    last=np.broadcast_to(mean,(len(physical),len(mean))).copy()
    for t in range(L):
        last=np.where(observed[:,t],physical[:,t],last)
        filled[:,t]=last
    return (filled-mean)/std


def pipeline(data, physical, observed, codec, forecaster, ridge=None, residual_input=False, decoder=None):
    x=tensor(preprocess(physical,observed,data['mean'],data['std']))
    if forecaster is None:
        future=torch.zeros((len(x),H,x.shape[-1]),device=DEVICE)
    elif codec is None:
        future=forecaster(x)
    else:
        z=codec.encode(x)
        future=(codec.decode if decoder is None else decoder)(forecaster(z))
    if ridge is not None:
        correction_input=x-codec.decode(codec.encode(x)) if residual_input else x
        future=future+ridge(block_features(correction_input))
    return (future*tensor(data['std'])+tensor(data['mean'])).cpu().numpy()


def latency(data, codec, forecaster, limit, ridge=None, residual_input=False, decoder=None):
    records=[]
    for batch in (1,8):
        origins=data['origins']['val'][:batch]
        physical=np.stack([data['physical'][o-L:o] for o in origins])
        observed=np.stack([data['mask'][o-L:o] for o in origins])
        with torch.no_grad():
            for _ in range(2):
                pipeline(data,physical,observed,codec,forecaster,ridge,residual_input,decoder)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings=[]
            for _ in range(7):
                torch.cuda.synchronize();started=time.perf_counter()
                output=pipeline(data,physical,observed,codec,forecaster,ridge,residual_input,decoder)
                torch.cuda.synchronize();timings.append(time.perf_counter()-started)
                if not np.isfinite(output).all():
                    raise ValueError('NONFINITE_LATENCY_OUTPUT')
        records.append({'batch':batch,'warmup':2,'repeats':7,'seconds':timings,'median_seconds':float(np.median(timings)),
                        'median_seconds_per_window':float(np.median(timings))/batch,
                        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
                        'resident_rss_bytes':psutil.Process().memory_info().rss,
                        'includes':'CPU raw observation forward-fill from FIT mean, normalization, host-to-GPU, E/F/D as applicable, ridge features/correction, inverse normalization, GPU-to-host; excludes model loading/disk read',
                        'co_resident_model_note':'Frozen Bolt and current experiment modules remain resident in every timing including R-only; total allocation is not incremental method-exclusive memory',
                        '_pipeline_output':output})
        limit.check()
    return records


def add_result(result, folder, name, data, prediction, cost, extra=None):
    for timing in cost:
        actual=timing.pop('_pipeline_output')
        expected=prediction['val'][:timing['batch']]*data['std']+data['mean']
        difference=float(np.max(np.abs(actual-expected)))
        tolerance=2e-3+2e-5*np.abs(expected)
        passed=bool(np.all(np.abs(actual-expected)<=tolerance))
        timing['pipeline_prediction_parity']={'physical_max_abs':difference,'absolute_tolerance':2e-3,
                                             'relative_tolerance':2e-5,'pass':passed}
        if not passed:
            raise AssertionError(f'TIMED_PIPELINE_PREDICTION_MISMATCH: {name} {difference}')
    path=folder/(name+'.npz')
    with path.open('xb') as f:
        np.savez_compressed(f,**{f'{s}_prediction':prediction[s] for s in prediction},
                            val_target=windows(data,'val',True),val_observed=masks(data,'val'),
                            **{f'{s}_origins':data['origins'][s] for s in prediction},mean=data['mean'],std=data['std'])
    record={'method':name,'scores':{s:physical_score(data,prediction[s],s) for s in prediction},
            'prediction':{'path':str(path.relative_to(ROOT)),'sha256':sha(path)},'latency':cost}
    if extra:
        record.update(extra)
    result['methods'].append(record)
    print(json.dumps({'completed':name,'dataset':data['dataset'],'val_mse':record['scores']['val']['mse'],
                      'val_mae':record['scores']['val']['mae']}),flush=True)


def analyze_codec(data, histories, targets, observed, codec, bolt, label, seed, folder, limit, training):
    result={'dataset':data['dataset'],'label':label,'seed':seed,'k':codec.project.out_features,'training':training,'methods':[]}
    codec.eval().requires_grad_(False)
    base={s:predict(x,codec,bolt,limit) for s,x in histories.items()}
    add_result(result,folder,label+'_BOLT',data,base,latency(data,codec,bolt,limit))
    rawfeature={s:block_features(x) for s,x in histories.items()}
    ridge,corrected,selection=correction(data,rawfeature,targets,observed,base,limit)
    add_result(result,folder,label+'_BOLT_RAW_R',data,corrected,latency(data,codec,bolt,limit,ridge),{'correction':selection})
    with torch.no_grad():
        residualfeature={}
        for s,x in histories.items():
            residual=[]
            for i in range(0,len(x),32):
                t=tensor(x[i:i+32]);residual.append(block_features(t-codec.decode(codec.encode(t))).cpu().numpy())
            residualfeature[s]=np.concatenate(residual)
    rridge,correctedr,rselection=correction(data,residualfeature,targets,observed,base,limit)
    add_result(result,folder,label+'_BOLT_RESIDUAL_R',data,correctedr,latency(data,codec,bolt,limit,rridge,True),{'correction':rselection})
    native,native_record=train_native(data,histories,targets,observed,codec,seed,folder,limit)
    native_prediction={s:predict(x,codec,native,limit) for s,x in histories.items()}
    add_result(result,folder,label+'_NATIVE',data,native_prediction,latency(data,codec,native,limit),{'native_training':native_record})
    native_ridge,native_corrected,native_selection=correction(data,rawfeature,targets,observed,native_prediction,limit)
    add_result(result,folder,label+'_NATIVE_RAW_R',data,native_corrected,latency(data,codec,native,limit,native_ridge),{'correction':native_selection})
    persistence={s:predict(x,codec,persist,limit) for s,x in histories.items()}
    add_result(result,folder,label+'_PERSISTENCE',data,persistence,latency(data,codec,persist,limit))
    decoder,drecord=refit_decoder(data,histories,codec,limit)
    refit={s:predict(x,codec,bolt,limit,decoder=decoder) for s,x in histories.items()}
    add_result(result,folder,label+'_BOLT_COMMON_D',data,refit,latency(data,codec,bolt,limit,decoder=decoder),{'decoder_refit':drecord})
    state_path=folder/(label+'_selected_models.pt')
    with state_path.open('xb') as f:
        torch.save({'codec':clone_state(codec),'native':clone_state(native),'raw_ridge':clone_state(ridge),'native_raw_ridge':clone_state(native_ridge),
                    'residual_ridge':clone_state(rridge),'common_decoder':clone_state(decoder),
                    'dataset':data['dataset'],'seed':seed,'label':label,'k':codec.project.out_features},f)
    result['selected_models']={'path':str(state_path.relative_to(ROOT)),'sha256':sha(state_path)}
    return result


def run(tag, mode, dataset=None, seed=None):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():
        raise RuntimeError('Approved CUDA device unavailable')
    protected=access_guard()
    output=HERE/('stage_b_'+tag);output.mkdir(exist_ok=False)
    folder=CACHE/('stage_b_'+tag);folder.mkdir(parents=True,exist_ok=False)
    limit=Limit(tag)
    summary={'status':'RUNNING','tag':tag,'mode':mode,'cases':[],'protected_test_guarded':protected,
             'device':torch.cuda.get_device_name(),'dtype':'float32','tf32_matmul':False,'tf32_cudnn':False,
             'torch_threads':2,'L':L,'H':H,'FIT':256,'TUNE':64,'VAL':'all existing origins',
             'evaluation_scope':'Previously used VAL; development diagnosis only',
             'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),OLD/'common.py',HERE/'PLAN_KO.md')}}
    try:
        modelstart=time.perf_counter();bolt=FrozenBolt()
        summary['bolt_load_wall_seconds']=time.perf_counter()-modelstart
        summary['bolt_initial_hash']=bolt.initial_hash
        summary['bolt_directory']=str(BOLT_DIRECTORY.relative_to(ROOT))
        summary['bolt_parameters']=sum(p.numel() for p in bolt.model.parameters())
        datasets=(dataset,) if dataset else ('jena','robin')
        for name in datasets:
            data=load_data(name,sizes=(256,64,None))
            histories={s:windows(data,s) for s in data['origins']}
            targets={s:windows(data,s,True) for s in data['origins']}
            observed={s:masks(data,s) for s in data['origins']}
            mean,basis,pca_record=pca_initialization(data)
            contract={'dataset':name,'channels':data['columns'],'source':str(Path(data['source']).relative_to(ROOT)),
                      'source_sha256':data['source_hash'],'mean':data['mean'],'std':data['std'],'normalization':'FIT_ONLY',
                      'fit_bounds':data['fit_bounds'],'tune_bounds':data['tune_bounds'],'val_bounds':data['val_bounds'],
                      'origins':data['origins'],'origin_hashes':{s:ahash(v) for s,v in data['origins'].items()},
                      'pca':pca_record,'pca_mean':mean,'pca_basis':basis,
                      'input_fill':'Forward fill resets at window start to FIT mean; observed values only',
                      'encoder_receptive_field':127,'native_past512':'All512 causal latent outputs; old observations may influence multiple latents'}
            write_new(output/(name+'_contract.json'),contract)
            codec=Codec(mean,basis,SEEDS[0]).to(DEVICE)
            summary.setdefault('fixtures',{})[name]=fixtures(bolt,codec,histories['fit'],targets['fit'],observed['fit'],limit)
            if mode=='preflight':
                continue
            if mode in ('all','baseline'):
                case_folder=folder/name;case_folder.mkdir()
                baseline={'dataset':name,'label':'RAW','methods':[]}
                raw={s:predict(x,None,bolt,limit) for s,x in histories.items()}
                add_result(baseline,case_folder,'RAW_BOLT',data,raw,latency(data,None,bolt,limit))
                feature={s:block_features(x) for s,x in histories.items()}
                ridge,pred,rrec=correction(data,feature,targets,observed,raw,limit)
                add_result(baseline,case_folder,'RAW_BOLT_RAW_R',data,pred,latency(data,None,bolt,limit,ridge),{'correction':rrec})
                zeros={s:np.zeros_like(y) for s,y in targets.items()}
                only,opred,orec=correction(data,feature,targets,observed,zeros,limit)
                add_result(baseline,case_folder,'RAW_R_ONLY',data,opred,latency(data,None,None,limit,only),{'correction':orec})
                state_path=case_folder/'RAW_ridges.pt'
                with state_path.open('xb') as f:
                    torch.save({'raw_bolt_ridge':clone_state(ridge),'ridge_only':clone_state(only)},f)
                baseline['selected_models']={'path':str(state_path.relative_to(ROOT)),'sha256':sha(state_path)}
                write_new(output/(name+'_RAW.json'),baseline);summary['cases'].append(baseline)
                pca=Codec(mean,basis,SEEDS[0],nonlinear=False).to(DEVICE).eval().requires_grad_(False)
                pca_result=analyze_codec(data,histories,targets,observed,pca,bolt,'PCA',SEEDS[0],case_folder,limit,{'objective':'PCA_FIT_ONLY','train_wall_seconds':0,'note':'PCA eigendecomposition included in process wall; not a neural objective-matched control'})
                write_new(output/(name+'_PCA.json'),pca_result);summary['cases'].append(pca_result)
            if mode in ('all','fit'):
                for current_seed in ((seed,) if seed else SEEDS):
                    hashes=[];orders=[]
                    for objective in ('REC','PRED'):
                        case_folder=folder/f'{name}_s{current_seed}_{objective}';case_folder.mkdir()
                        codec=Codec(mean,basis,current_seed).to(DEVICE)
                        training=train_codec(data,histories,targets,observed,codec,bolt,objective,current_seed,case_folder,limit)
                        hashes.append(training['initial_codec_hash']);orders.append(training['order_sha256'])
                        result=analyze_codec(data,histories,targets,observed,codec,bolt,objective,current_seed,case_folder,limit,training)
                        write_new(output/f'{name}_s{current_seed}_{objective}.json',result);summary['cases'].append(result)
                    if hashes[0]!=hashes[1] or orders[0]!=orders[1]:
                        raise AssertionError('REC_PRED_INITIALIZATION_OR_ORDER_NOT_MATCHED')
            if state_hash(bolt.model)!=bolt.initial_hash:
                raise AssertionError('BACKBONE_FINAL_HASH_CHANGED')
        summary['bolt_final_hash']=state_hash(bolt.model)
        summary['status']='COMPLETE'
        limit.check();limit.close('COMPLETE')
        summary['resources']=limit.record
        write_new(output/'RESULT.json',summary)
        print(json.dumps({'status':'COMPLETE','output':str(output),'resources':limit.record}),flush=True)
    except BaseException as error:
        summary['status']='FAILED_PRESERVED';summary['error']=str(error);summary['traceback']=traceback.format_exc()
        limit.close('FAILED_PRESERVED',str(error));summary['resources']=limit.record
        write_new(output/'FAILURE.json',summary)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--mode',choices=('preflight','baseline','fit','all'),default='all')
    parser.add_argument('--dataset',choices=('jena','robin'))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Alphanumeric/underscore tag required')
    run(args.tag,args.mode,args.dataset,args.seed)
