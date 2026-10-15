"""Matched normalization/regularization/context study; writes only this campaign."""
from pathlib import Path
import argparse
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
CACHE = ROOT / '.cache' / HERE.name
OLD = ROOT / 'research/forecast_component_study_20261013'
sys.path.insert(0, str(OLD))
import component_experiment as previous
import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from common import access_guard, ahash, sha, read, jsonable, load_data, score, ridge_grouped, Ridge, LAMBDAS, H

DEVICE = 'cuda'
SEEDS = (92601, 92602)
CHECKPOINTS = (0, 100, 200, 400, 600)
CONFIGS = {'BASE': (1e-3, 0.), 'LOW_LR': (3e-4, 0.), 'DECAY': (1e-3, .01)}


def write_new(path, value):
    path = Path(path)
    if not (path.resolve().is_relative_to(HERE) or path.resolve().is_relative_to(CACHE)):
        raise ValueError('Write outside new campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(jsonable(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


class Limit:
    def __init__(self, tag):
        self.start = time.perf_counter()
        self.process = psutil.Process()
        self.cpu_start = sum(self.process.cpu_times()[:2])
        self.path = HERE / 'resources' / f'{tag}_{os.getpid()}_{time.time_ns()}.json'
        self.path.parent.mkdir(exist_ok=True)
        self.record = {'pid': os.getpid(), 'tag': tag, 'status': 'RUNNING', 'gpu_process': True,
                       'started_unix': time.time(), 'wall_seconds': 0., 'peak_sampled_rss_bytes': 0,
                       'peak_allocated_vram_bytes': 0}
        self.last_storage = -60.
        self.check()

    def check(self):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.record['cpu_seconds'] = sum(self.process.cpu_times()[:2])-self.cpu_start
        self.record['peak_sampled_rss_bytes'] = max(self.record['peak_sampled_rss_bytes'], self.process.memory_info().rss)
        self.record['peak_allocated_vram_bytes'] = max(self.record['peak_allocated_vram_bytes'], torch.cuda.max_memory_allocated())
        if self.record['wall_seconds']-self.last_storage >= 30:
            self.record['new_cache_bytes'] = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
            self.record['new_research_bytes'] = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
            self.last_storage = self.record['wall_seconds']
        previous_wall = sum(read(p)['wall_seconds'] for p in self.path.parent.glob('*.json') if p != self.path)
        self.record['cumulative_gpu_process_wall_seconds'] = previous_wall+self.record['wall_seconds']
        self.flush()
        if self.record['cumulative_gpu_process_wall_seconds'] >= 2700:
            raise RuntimeError('NEW_GPU_PROCESS_WALL_LIMIT_2700S')
        if max(self.record['peak_sampled_rss_bytes'], self.record['peak_allocated_vram_bytes']) > 8*1024**3:
            raise RuntimeError('NEW_RAM_OR_ALLOCATED_VRAM_LIMIT_8GIB')
        if self.record['new_cache_bytes']+self.record['new_research_bytes'] > 4*1024**3:
            raise RuntimeError('NEW_STORAGE_LIMIT_4GIB')

    def flush(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path)

    def close(self, status, error=None):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.record['cpu_seconds'] = sum(self.process.cpu_times()[:2])-self.cpu_start
        self.record.update(status=status, error=error)
        self.flush()


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device=DEVICE)


def boolean(value):
    return torch.as_tensor(value, dtype=torch.bool, device=DEVICE)


def preprocess(physical, observed, mean, std):
    filled = np.empty_like(physical, dtype=np.float32)
    last = np.broadcast_to(mean, (len(physical), len(mean))).copy()
    for t in range(physical.shape[1]):
        last = np.where(observed[:, t], physical[:, t], last)
        filled[:, t] = last
    return ((filled-mean)/std).astype(np.float32)


def arrays(data, length):
    history, target, observed, history_observed = {}, {}, {}, {}
    for split, origins in data['origins'].items():
        physical = np.stack([data['physical'][o-length:o] for o in origins])
        history_observed[split] = np.stack([data['mask'][o-length:o] for o in origins])
        history[split] = preprocess(physical, history_observed[split], data['mean'], data['std'])
        target[split] = np.stack([data['x'][o:o+H] for o in origins]).astype(np.float32)
        observed[split] = np.stack([data['mask'][o:o+H] for o in origins])
    return history, target, observed, history_observed


def metric(data, prediction, target, observed):
    result = score(prediction*data['std']+data['mean'], target*data['std']+data['mean'], observed)
    result['normalized'] = score(prediction, target, observed)
    result['scale_note'] = 'Physical channel macro; heterogeneous units, never pooled across datasets'
    return result


def initialization(data, k, kind='PCA'):
    lo, hi = data['fit_bounds']
    x = data['x'][lo:hi].astype(np.float64)
    mean = x.mean(0)
    covariance = (x-mean).T@(x-mean)/len(x)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, -k:][:, ::-1].copy()
    for j in range(k):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    if kind == 'IDENTITY':
        assert k == x.shape[1]
        mean = np.zeros(x.shape[1])
        basis = np.eye(x.shape[1])
    return mean.astype(np.float32), basis.astype(np.float32), {'kind': kind, 'k': k, 'fit_rows': hi-lo,
        'explained_variance_fraction': float(eigenvalues[-k:].sum()/eigenvalues.sum()),
        'basis_sha256': ahash(basis.astype(np.float32)), 'mean_sha256': ahash(mean.astype(np.float32))}


class Codec(nn.Module):
    def __init__(self, mean, basis, seed, receptive_field=127):
        super().__init__()
        torch.manual_seed(seed)
        channels, k = basis.shape
        self.project = nn.Linear(channels, k)
        self.decode = nn.Linear(k, channels)
        self.input = nn.Conv1d(channels, 16, 1)
        dilations = {127: (1, 2, 4, 8, 16, 32), 31: (1, 1, 1, 2, 4, 6)}[receptive_field]
        self.temporal = nn.ModuleList(nn.Conv1d(16, 16, 3, dilation=d) for d in dilations)
        self.output = nn.Conv1d(16, k, 1)
        self.receptive_field = receptive_field
        with torch.no_grad():
            self.project.weight.copy_(torch.from_numpy(basis.T.copy()))
            self.project.bias.copy_(torch.from_numpy((-mean@basis).copy()))
            self.decode.weight.copy_(torch.from_numpy(basis.copy()))
            self.decode.bias.copy_(torch.from_numpy(mean.copy()))
            self.output.weight.zero_()
            self.output.bias.zero_()

    def encode(self, x):
        z = self.project(x)
        v = F.gelu(self.input(x.transpose(1, 2)))
        for layer in self.temporal:
            v = F.gelu(layer(F.pad(v, (2*layer.dilation[0], 0))))
        return z+self.output(v).transpose(1, 2)


class NativeShared(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.layer = nn.Linear(length, H)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.weight[:, -1] = 1
            self.layer.bias.zero_()

    def forward(self, z):
        return self.layer(z.transpose(1, 2)).transpose(1, 2)


def predict(histories, codec, forecaster, limit, decoder=None):
    result = []
    with torch.no_grad():
        for i in range(0, len(histories), 8):
            x = tensor(histories[i:i+8])
            future = forecaster(x if codec is None else codec.encode(x))
            if codec is not None:
                future = (codec.decode if decoder is None else decoder)(future)
            result.append(future.cpu().numpy())
            limit.check()
    return np.concatenate(result)


def reconstruction(histories, codec, limit):
    result = []
    with torch.no_grad():
        for i in range(0, len(histories), 32):
            result.append(codec.decode(codec.encode(tensor(histories[i:i+32]))).cpu().numpy())
            limit.check()
    return np.concatenate(result)


def block_features(x):
    assert x.shape[1] == 512
    if isinstance(x, np.ndarray):
        return np.concatenate((x.reshape(len(x), 32, 16, x.shape[-1]).mean(2), x[:, -1:]), axis=1).reshape(len(x), -1)
    return torch.cat((x.reshape(len(x), 32, 16, x.shape[-1]).mean(2), x[:, -1:]), dim=1).reshape(len(x), -1)


def train_codec(data, histories, targets, observed, history_mask, codec, bolt, objective, config, seed, folder, limit):
    started = time.perf_counter()
    initial = previous.state_hash(codec)
    learning_rate, decay = CONFIGS[config]
    codec.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=learning_rate, weight_decay=decay)
    order = np.random.default_rng(seed).integers(0, len(histories['fit']), size=(600, 4))
    tune_positions = np.stack([np.arange(o-histories['tune'].shape[1], o) for o in data['origins']['tune']])
    post_fit_mask = history_mask['tune'] & (tune_positions >= data['fit_bounds'][1])[:, :, None]
    if not np.any(post_fit_mask):
        raise ValueError('NO_OBSERVED_POST_FIT_RECONSTRUCTION_POSITIONS')
    checkpoints, curve = [], []
    for update in range(601):
        if update in CHECKPOINTS:
            future = {s: predict(histories[s], codec, bolt, limit) for s in ('fit', 'tune')}
            restored = {s: reconstruction(histories[s], codec, limit) for s in ('fit', 'tune')}
            path = folder/f'{objective}_{config}_u{update}.pt'
            with path.open('xb') as f:
                torch.save({'codec': previous.clone_state(codec), 'update': update, 'seed': seed,
                            'objective': objective, 'config': config}, f)
            checkpoint = {'update': update, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                'forecast': {s: metric(data, future[s], targets[s], observed[s]) for s in future},
                'reconstruction': {s: metric(data, restored[s], histories[s], history_mask[s]) for s in restored},
                'reconstruction_tune_post_fit_positions': metric(data, restored['tune'], histories['tune'], post_fit_mask)}
            checkpoints.append(checkpoint)
            print(json.dumps({'dataset': data['dataset'], 'seed': seed, 'objective': objective, 'config': config,
                'L': histories['fit'].shape[1], 'RF': codec.receptive_field, 'K': codec.project.out_features,
                'update': update, 'tune_normalized_mse': checkpoint['forecast']['tune']['normalized']['mse'],
                'tune_reconstruction_normalized_mse': checkpoint['reconstruction']['tune']['normalized']['mse']}), flush=True)
        if update == 600:
            break
        ids = order[update]
        x = tensor(histories['fit'][ids])
        optimizer.zero_grad(set_to_none=True)
        if objective == 'REC':
            prediction = codec.decode(codec.encode(x))
            loss = previous.macro_loss(prediction, x, boolean(history_mask['fit'][ids]))
        else:
            prediction = codec.decode(bolt(codec.encode(x)))
            loss = previous.macro_loss(prediction, tensor(targets['fit'][ids]), boolean(observed['fit'][ids]))
        if not torch.isfinite(loss):
            raise ValueError('NONFINITE_TRAINING_LOSS')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.)
        if not torch.isfinite(gradient_norm):
            raise ValueError('NONFINITE_TRAINING_GRADIENT')
        optimizer.step()
        curve.append({'update': update+1, 'loss': float(loss.detach()), 'gradient_norm_before_clipping': float(gradient_norm)})
        if update % 10 == 0:
            limit.check()
    selected = {'FORECAST': min(checkpoints, key=lambda c: c['forecast']['tune']['normalized']['mse'])['update']}
    if objective == 'REC':
        selected['RECONSTRUCTION'] = min(checkpoints, key=lambda c: c['reconstruction']['tune']['normalized']['mse'])['update']
    return {'objective': objective, 'config': config, 'seed': seed, 'initial_codec_hash': initial,
        'optimizer': 'AdamW', 'learning_rate': learning_rate, 'weight_decay': decay, 'updates': 600,
        'batch_size': 4, 'batch_indices': order.tolist(), 'order_sha256': ahash(order), 'checkpoints': checkpoints,
        'selected_updates': selected, 'curve': curve, 'train_wall_seconds': time.perf_counter()-started,
        'post_fit_reconstruction_diagnostic': {'cut': data['fit_bounds'][1], 'mask_sha256': ahash(post_fit_mask),
            'observed_counts': post_fit_mask.sum((0, 1)).tolist(),
            'unique_post_fit_timestamps': len(np.unique(tune_positions[tune_positions >= data['fit_bounds'][1]])),
            'note': 'Only observed historical positions at or after FIT cut; repeated timestamps retained; never used to select checkpoint'},
        'trainable_parameters': sum(p.numel() for p in codec.parameters()),
        'selection': 'TUNE normalized observed channel-macro MSE; earliest exact tie; VAL never selects'}


def train_native(data, histories, target, observed, codec, seed, limit):
    started = time.perf_counter()
    codec.eval().requires_grad_(False)
    with torch.no_grad():
        latent = {s: torch.cat([codec.encode(tensor(x[i:i+32])) for i in range(0, len(x), 32)])
                  for s, x in histories.items() if s != 'val'}
    native = NativeShared(histories['fit'].shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0.)
    order = np.random.default_rng(seed).integers(0, len(histories['fit']), size=(600, 4))
    best, checkpoints, curve = None, [], []
    for update in range(601):
        if update in CHECKPOINTS:
            with torch.no_grad():
                p = codec.decode(native(latent['tune'])).cpu().numpy()
            value = metric(data, p, target['tune'], observed['tune'])
            checkpoints.append({'update': update, 'tune': value})
            if best is None or value['normalized']['mse'] < best[0]:
                best = (value['normalized']['mse'], update, previous.clone_state(native))
        if update == 600:
            break
        ids = order[update]
        optimizer.zero_grad(set_to_none=True)
        loss = previous.macro_loss(codec.decode(native(latent['fit'][ids])), tensor(target['fit'][ids]), boolean(observed['fit'][ids]))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(native.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise ValueError('NONFINITE_NATIVE_LOSS_GRADIENT')
        optimizer.step()
        curve.append({'update': update+1, 'loss': float(loss.detach())})
        if update % 20 == 0:
            limit.check()
    native.load_state_dict(best[2])
    native.eval().requires_grad_(False)
    return native, {'selected_update': best[1], 'updates': 600, 'learning_rate': 1e-3, 'weight_decay': 0.,
        'checkpoints': checkpoints, 'curve': curve, 'order_sha256': ahash(order),
        'encoder_decoder_frozen': True, 'shared_across_latent_channels': True,
        'trainable_parameters': sum(p.numel() for p in native.parameters()), 'latent_history_length': histories['fit'].shape[1],
        'train_wall_seconds': time.perf_counter()-started, 'criterion': 'TUNE normalized observed channel-macro MSE'}


def correction(data, features, target, observed, base, limit):
    started = time.perf_counter()
    residual = {s: target[s]-base[s] for s in target}
    _, selection, fits = ridge_grouped(features['fit'], residual['fit'], observed['fit'], features['tune'],
        residual['tune'], observed['tune'], features['val'][:0], score_scale=np.ones(target['fit'].shape[-1]), budget=limit)
    model = previous.TorchRidge(fits, features['fit'].shape[1], target['fit'].shape[-1]).eval()
    with torch.no_grad():
        pred = {s: base[s]+model(tensor(features[s])).cpu().numpy() for s in target}
    check = np.zeros((len(features['val']), H*target['fit'].shape[-1]))
    for columns, fitted in fits:
        check[:, columns] = fitted.predict(features['val'])
    check = check.reshape(target['val'].shape)
    selection.update(train_wall_seconds=time.perf_counter()-started,
        float32_effective_ridge_max_abs_vs_float64=float(np.max(np.abs(pred['val']-base['val']-check))),
        features=features['fit'].shape[1], parameter_count=model.weight.numel()+model.bias.numel(),
        selection='TUNE normalized observed channel-macro MSE', correction_raw_input_support=512,
        note='Residual fitted on in-sample FIT base predictions; no cross-fitting; VAL predicted only after TUNE lambda selection')
    return model, pred, selection


def common_decoder(data, histories, history_observed, codec, limit):
    with torch.no_grad():
        z = torch.cat([codec.encode(tensor(histories['fit'][i:i+32])) for i in range(0, len(histories['fit']), 32)]).cpu().numpy()
    z = z.reshape(-1, codec.project.out_features).astype(np.float64)
    x = histories['fit'].reshape(-1, histories['fit'].shape[-1]).astype(np.float64)
    observed = history_observed['fit'].reshape(x.shape)
    coefs, biases = [], []
    for c in range(x.shape[1]):
        zz, yy = z[observed[:, c]], x[observed[:, c], c]
        zm, ym = zz.mean(0), yy.mean()
        coef = np.linalg.solve((zz-zm).T@(zz-zm)+len(zz)*1e-4*np.eye(z.shape[1]), (zz-zm).T@(yy-ym))
        coefs.append(coef)
        biases.append(ym-zm@coef)
    decoder = nn.Linear(z.shape[1], x.shape[1]).to(DEVICE)
    with torch.no_grad():
        decoder.weight.copy_(tensor(np.stack(coefs)))
        decoder.bias.copy_(tensor(np.array(biases)))
    limit.check()
    return decoder.eval().requires_grad_(False), {'lambda': 1e-4, 'selection': 'Fixed identical rule, FIT historical reconstruction only',
        'fit_history_positions': len(z), 'note': 'Same fitting rule; coefficients differ across encoders'}


class SelfRidge(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        self.register_buffer('weights', tensor(weights.astype(np.float32)))
        self.register_buffer('biases', tensor(biases.astype(np.float32)))

    def forward(self, x):
        return torch.einsum('nlc,lhc->nhc', x[:, -(np.arange(33)+1), :], self.weights)+self.biases


def self_ridge(data, history, target, observed, limit):
    started = time.perf_counter()
    features = {s: x[:, -(np.arange(33)+1), :] for s, x in history.items()}
    channels = target['fit'].shape[-1]
    options = []
    for lam in LAMBDAS:
        weights, biases = np.zeros((33, H, channels)), np.zeros((H, channels))
        for c in range(channels):
            groups = {}
            for h in range(H):
                groups.setdefault(observed['fit'][:, h, c].tobytes(), []).append(h)
            for hs in groups.values():
                valid = observed['fit'][:, hs[0], c]
                r = Ridge(lam).fit(features['fit'][valid, :, c], target['fit'][valid, :, c][:, hs])
                coef = r.coef if r.coef is not None else r.train.T@r.dual
                effective = coef/r.std[:, None]
                weights[:, hs, c] = effective
                biases[hs, c] = r.ym-r.mean@effective
        tune = np.einsum('nlc,lhc->nhc', features['tune'], weights)+biases
        options.append((metric(data, tune, target['tune'], observed['tune']), lam, weights, biases))
        limit.check()
    chosen = min(range(len(options)), key=lambda i: options[i][0]['normalized']['mse'])
    _, lam, weights, biases = options[chosen]
    model = SelfRidge(weights, biases).eval()
    with torch.no_grad():
        prediction = {s: model(tensor(x)).cpu().numpy() for s, x in history.items()}
    return model, prediction, {'selected_lambda': lam, 'selected_index': chosen, 'candidates': [{'lambda': o[1], 'tune': o[0]} for o in options],
        'criterion': 'TUNE normalized observed channel-macro MSE', 'train_wall_seconds': time.perf_counter()-started,
        'raw_feature_support': 33, 'forward_fill_support': 512, 'channel_independent': True}


def deployment(data, physical, observed, codec, forecaster, length, ridge=None, decoder=None, self_model=None):
    x = tensor(preprocess(physical[:, -length:], observed[:, -length:], data['mean'], data['std']))
    if self_model is not None:
        future = self_model(x)
    elif forecaster is None:
        future = torch.zeros((len(x), H, x.shape[-1]), device=DEVICE)
    elif codec is None:
        future = forecaster(x)
    else:
        future = (codec.decode if decoder is None else decoder)(forecaster(codec.encode(x)))
    if ridge is not None:
        raw = tensor(preprocess(physical, observed, data['mean'], data['std']))
        future = future+ridge(block_features(raw))
    return (future*tensor(data['std'])+tensor(data['mean'])).cpu().numpy()


def latency(data, prediction, codec, forecaster, length, limit, ridge=None, decoder=None, self_model=None):
    records = []
    raw_length = 512 if ridge is not None or self_model is not None else length
    for batch in (1, 8):
        oo = data['origins']['val'][:batch]
        physical = np.stack([data['physical'][o-raw_length:o] for o in oo])
        observed = np.stack([data['mask'][o-raw_length:o] for o in oo])
        with torch.no_grad():
            for _ in range(2):
                deployment(data, physical, observed, codec, forecaster, length, ridge, decoder, self_model)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(7):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = deployment(data, physical, observed, codec, forecaster, length, ridge, decoder, self_model)
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-start)
        expected = prediction[:batch]*data['std']+data['mean']
        difference = float(np.max(np.abs(output-expected)))
        passed = bool(np.all(np.abs(output-expected) <= 2e-3+2e-5*np.abs(expected)))
        if not passed:
            raise AssertionError(f'DEPLOYMENT_SCIENCE_PARITY {difference}')
        records.append({'batch': batch, 'warmup': 2, 'repeats': 7, 'seconds': timings,
            'median_seconds': float(np.median(timings)), 'median_seconds_per_window': float(np.median(timings))/batch,
            'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(), 'resident_rss_bytes': psutil.Process().memory_info().rss,
            'pipeline_prediction_parity': {'physical_max_abs': difference, 'absolute_tolerance': 2e-3, 'relative_tolerance': 2e-5, 'pass': passed},
            'encoder_history_length': length, 'raw_correction_support': 512 if ridge is not None else None,
            'includes': 'CPU raw-window forward-fill, normalization, H2D, E/F/D or self ridge, optional 512-support correction features, inverse normalization, D2H and synchronization; disk/model load excluded',
            'co_resident_model_note': 'Bolt/current modules remain resident in every timing. Allocation is not incremental method-exclusive memory.'})
        limit.check()
    return records


def add_method(case, folder, name, data, prediction, target, observed, timings, extra=None):
    path = folder/(name+'.npz')
    with path.open('xb') as f:
        np.savez_compressed(f, **{s+'_prediction': p for s, p in prediction.items()},
            **{s+'_target': target[s] for s in prediction}, **{s+'_observed': observed[s] for s in prediction},
            **{s+'_origins': data['origins'][s] for s in prediction}, mean=data['mean'], std=data['std'])
    record = {'method': name, 'scores': {s: metric(data, p, target[s], observed[s]) for s, p in prediction.items()},
        'prediction': {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}, 'latency': timings}
    if extra:
        record.update(extra)
    case['methods'].append(record)
    print(json.dumps({'completed': name, 'dataset': data['dataset'], 'case': case['label'],
                      'val_normalized_mse': record['scores']['val']['normalized']['mse']}), flush=True)


def analyze(data, history, target, observed, history_observed, raw_history, codec, bolt, seed, folder, limit, case, full=True):
    codec.eval().requires_grad_(False)
    length = history['fit'].shape[1]
    base = {s: predict(x, codec, bolt, limit) for s, x in history.items()}
    add_method(case, folder, 'BOLT', data, base, target, observed, latency(data, base['val'], codec, bolt, length, limit))
    if not full:
        return
    features = {s: block_features(x) for s, x in raw_history.items()}
    raw_r, corrected, selection = correction(data, features, target, observed, base, limit)
    add_method(case, folder, 'BOLT_RAW_R', data, corrected, target, observed,
        latency(data, corrected['val'], codec, bolt, length, limit, ridge=raw_r), {'correction': selection})
    native, nrecord = train_native(data, history, target, observed, codec, seed, limit)
    npred = {s: predict(x, codec, native, limit) for s, x in history.items()}
    add_method(case, folder, 'NATIVE', data, npred, target, observed,
        latency(data, npred['val'], codec, native, length, limit), {'native_training': nrecord})
    native_r, ncorrected, nselection = correction(data, features, target, observed, npred, limit)
    add_method(case, folder, 'NATIVE_RAW_R', data, ncorrected, target, observed,
        latency(data, ncorrected['val'], codec, native, length, limit, ridge=native_r), {'correction': nselection})
    persistence = {s: predict(x, codec, previous.persist, limit) for s, x in history.items()}
    add_method(case, folder, 'PERSISTENCE', data, persistence, target, observed,
        latency(data, persistence['val'], codec, previous.persist, length, limit))
    decoder, drecord = common_decoder(data, history, history_observed, codec, limit)
    common = {s: predict(x, codec, bolt, limit, decoder=decoder) for s, x in history.items()}
    add_method(case, folder, 'BOLT_COMMON_D', data, common, target, observed,
        latency(data, common['val'], codec, bolt, length, limit, decoder=decoder), {'decoder_refit': drecord})
    path = folder/'selected_models.pt'
    with path.open('xb') as f:
        torch.save({'codec': previous.clone_state(codec), 'native': previous.clone_state(native),
            'raw_ridge': previous.clone_state(raw_r), 'native_raw_ridge': previous.clone_state(native_r),
            'common_decoder': previous.clone_state(decoder)}, f)
    case['selected_models'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def fixture(data, history, target, observed, bolt, limit):
    mean, basis, _ = initialization(data, data['x'].shape[1]//4)
    codec = Codec(mean, basis, SEEDS[0]).to(DEVICE)
    standard = previous.Codec(mean, basis, SEEDS[0]).to(DEVICE)
    assert previous.state_hash(codec) == previous.state_hash(standard)
    values = previous.fixtures(bolt, codec, history['fit'], target['fit'], observed['fit'], limit)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    a, b = Codec(mean, basis, SEEDS[0]).to(DEVICE), Codec(mean, basis, SEEDS[0]).to(DEVICE)
    adam = torch.optim.Adam(a.parameters(), lr=1e-3)
    adamw = torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=0.)
    maximum = 0.
    for i in range(5):
        x = tensor(history['fit'][i:i+4])
        for model, optimizer in ((a, adam), (b, adamw)):
            optimizer.zero_grad(set_to_none=True)
            loss = previous.macro_loss(model.decode(model.encode(x)), x, torch.ones_like(x, dtype=torch.bool))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        maximum = max(maximum, max(float((p-q).detach().abs().max()) for p, q in zip(a.parameters(), b.parameters())))
    torch.set_rng_state(rng)
    torch.cuda.set_rng_state(cuda_rng)
    if maximum > 2e-6:
        raise AssertionError('ADAM_ADAMW_ZERO_DECAY_5STEP_PARITY')
    values['adam_vs_adamw_wd0_5step_max_abs'] = maximum
    values['optimizer_fixture_rng_restored'] = True
    identity = Codec(np.zeros(data['x'].shape[1], np.float32), np.eye(data['x'].shape[1], dtype=np.float32), SEEDS[0]).to(DEVICE)
    with torch.no_grad():
        actual = tensor(history['fit'][:2])
        raw = bolt(actual)
        identity_prediction = identity.decode(bolt(identity.encode(actual)))
        identity_delta = float((raw-identity_prediction).abs().max())
    if identity_delta != 0:
        raise AssertionError('INITIAL_IDENTITY_FULL_RANK_RAW_BOLT_PARITY')
    values['full_identity_initial_actual_bolt_max_abs_vs_raw'] = identity_delta
    short = Codec(mean, basis, SEEDS[0], 31).to(DEVICE)
    standard = Codec(mean, basis, SEEDS[0], 127).to(DEVICE)
    same_hash = previous.state_hash(short) == previous.state_hash(standard)
    same_capacity = sum(p.numel() for p in short.parameters()) == sum(p.numel() for p in standard.parameters())
    with torch.no_grad():
        short.output.weight.fill_(.03125)
        actual = tensor(history['fit'][:2])
        altered = actual.clone()
        altered[:, :-31] += 99
        short_delta = float((short.encode(actual)[:, -1]-short.encode(altered)[:, -1]).abs().max())
    if not same_hash or not same_capacity or short_delta > 2e-5:
        raise AssertionError('RF31_MATCHED_INITIALIZATION_CAPACITY_AND_SUPPORT')
    values['rf31_matched_capacity_initialization'] = {'state_hash_equal': same_hash, 'parameters_equal': same_capacity,
        'active_branch_outside_last31_max_abs': short_delta, 'pass': True}
    return values


def g0(data, history, target, observed, bolt, output, folder, limit):
    source = read(OLD/'stage_b_full_v1/RESULT.json')
    reconstruction_source = read(OLD/'reconstruction_audit_v1/RESULT.json')
    cases = []
    for old_case in source['cases']:
        if old_case['dataset'] != data['dataset'] or old_case['label'] not in ('REC', 'PRED'):
            continue
        objective, seed = old_case['label'], old_case['seed']
        candidates = old_case['training']['checkpoints']
        selected = {'FORECAST': min(candidates, key=lambda c: c['tune']['normalized']['mse'])['update']}
        reconstruction_case = next(c for c in reconstruction_source['cases'] if c['dataset']==data['dataset'] and c['objective']==objective and c['seed']==seed)
        if objective == 'REC':
            selected['RECONSTRUCTION'] = min(reconstruction_case['snapshots'], key=lambda c: c['reconstruction']['tune']['normalized']['mse'])['update']
        for criterion, update in selected.items():
            cp = next(c for c in candidates if c['update']==update)
            assert sha(ROOT/cp['path']) == cp['sha256']
            mean, basis, _ = initialization(data, old_case['k'])
            codec = Codec(mean, basis, seed).to(DEVICE)
            saved = torch.load(ROOT/cp['path'], map_location='cpu', weights_only=True)
            codec.load_state_dict(saved['codec'])
            label = f'G0_{objective}_{criterion}_s{seed}'
            case_folder = folder/label
            case_folder.mkdir()
            case = {'stage': 'G0', 'dataset': data['dataset'], 'label': label, 'seed': seed,
                'objective': objective, 'selection_role': criterion, 'k': old_case['k'], 'length': 512, 'receptive_field': 127,
                'reused_checkpoint': {'path': cp['path'], 'sha256': cp['sha256'], 'selected_update': update},
                'reused_tune_candidates': candidates, 'new_training_updates': 0,
                'source_reconstruction_sha256': sha(OLD/'reconstruction_audit_v1/RESULT.json'),
                'source_forecast_sha256': sha(OLD/'stage_b_full_v1/RESULT.json'), 'methods': []}
            analyze(data, history, target, observed, None, history, codec, bolt, seed, case_folder, limit, case, full=False)
            write_new(output/(label+'.json'), case)
            cases.append(case)
    return cases


def fit_case(data, spec, bolt, output, folder, limit, raw_history):
    label = f"{spec['stage']}_{spec['objective']}_{spec['config']}_K{spec['k']}_L{spec['length']}_RF{spec['rf']}_{spec['init']}_s{spec['seed']}"
    case_folder = folder/label
    case_folder.mkdir()
    history, target, observed, history_observed = arrays(data, spec['length'])
    mean, basis, initial = initialization(data, spec['k'], spec['init'])
    codec = Codec(mean, basis, spec['seed'], spec['rf']).to(DEVICE)
    training = train_codec(data, history, target, observed, history_observed, codec, bolt, spec['objective'], spec['config'], spec['seed'], case_folder, limit)
    cases = []
    for role, update in training['selected_updates'].items():
        cp = next(c for c in training['checkpoints'] if c['update']==update)
        codec.load_state_dict(torch.load(ROOT/cp['path'], map_location='cpu', weights_only=True)['codec'])
        selected_folder = case_folder/role
        selected_folder.mkdir()
        case = {'stage': spec['stage'], 'dataset': data['dataset'], 'label': label+'_'+role, 'seed': spec['seed'],
            'objective': spec['objective'], 'config': spec['config'], 'k': spec['k'], 'length': spec['length'],
            'receptive_field': spec['rf'], 'initialization': initial, 'selection_role': role,
            'selected_update': update, 'selected_codec_hash': previous.state_hash(codec), 'training': training, 'methods': [],
            'same_target_origins_full512_contract': True,
            'capacity_note': 'RF31/RF127 use the same 6 convolution layers and parameter counts; dilation locations change. Native parameter count changes with L. K changes E/D/output sizes.',
            'dilations': [layer.dilation[0] for layer in codec.temporal],
            'raw_correction_support': 512, 'component_input_support': spec['length']}
        analyze(data, history, target, observed, history_observed, raw_history, codec, bolt, spec['seed'], selected_folder, limit, case,
                full=spec['objective']=='PRED')
        write_new(output/(case['label']+'.json'), case)
        cases.append(case)
    return cases


def safe_fit_case(data, spec, bolt, output, folder, limit, raw_history, result):
    try:
        return fit_case(data, spec, bolt, output, folder, limit, raw_history)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)) or 'LIMIT_' in str(error) or 'FROZEN_BOLT' in str(error):
            raise
        failure = {'status': 'FAILED_PATH_PRESERVED', 'dataset': data['dataset'], 'spec': spec,
                   'error': str(error), 'traceback': traceback.format_exc(), 'resources': dict(limit.record)}
        failure_name = f"FAILED_{spec['stage']}_{spec['objective']}_{spec['config']}_K{spec['k']}_L{spec['length']}_RF{spec['rf']}_{spec['init']}_s{spec['seed']}.json"
        write_new(output/failure_name, failure)
        result.setdefault('failed_paths', []).append(failure)
        print(json.dumps({'failed_path': failure_name, 'error': str(error)}), flush=True)
        torch.cuda.empty_cache()
        return []


def raw_case(data, history, target, observed, bolt, output, folder, limit):
    case_folder = folder/'RAW'
    case_folder.mkdir()
    case = {'stage': 'G3', 'dataset': data['dataset'], 'label': 'RAW', 'seed': None, 'k': data['x'].shape[1], 'length': 512, 'methods': []}
    base = {s: predict(x, None, bolt, limit) for s, x in history.items()}
    add_method(case, case_folder, 'RAW_BOLT', data, base, target, observed, latency(data, base['val'], None, bolt, 512, limit))
    features = {s: block_features(x) for s, x in history.items()}
    r, pred, rec = correction(data, features, target, observed, base, limit)
    add_method(case, case_folder, 'RAW_BOLT_RAW_R', data, pred, target, observed,
        latency(data, pred['val'], None, bolt, 512, limit, ridge=r), {'correction': rec})
    zeros = {s: np.zeros_like(x) for s, x in target.items()}
    ronly, rpred, rrec = correction(data, features, target, observed, zeros, limit)
    add_method(case, case_folder, 'RAW_R_ONLY', data, rpred, target, observed,
        latency(data, rpred['val'], None, None, 512, limit, ridge=ronly), {'correction': rrec})
    self_model, spred, srec = self_ridge(data, history, target, observed, limit)
    add_method(case, case_folder, 'RAW_SELF_LAG', data, spred, target, observed,
        latency(data, spred['val'], None, None, 512, limit, self_model=self_model), {'selection': srec})
    path = case_folder/'raw_models.pt'
    with path.open('xb') as f:
        torch.save({'raw_bolt_ridge': previous.clone_state(r), 'ridge_only': previous.clone_state(ronly),
                    'self_ridge': previous.clone_state(self_model)}, f)
    case['selected_models'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    write_new(output/'RAW.json', case)
    return case


def run(tag, mode):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not (HERE/'PLAN_KO.md').exists():
        raise RuntimeError('ROOT_PLAN_REQUIRED_BEFORE_EXECUTION')
    if not torch.cuda.is_available():
        raise RuntimeError('Approved CUDA unavailable')
    protected = access_guard()
    output = HERE/('run_'+tag)
    output.mkdir(exist_ok=False)
    folder = CACHE/('run_'+tag)
    folder.mkdir(parents=True, exist_ok=False)
    limit = Limit(tag)
    result = {'status': 'RUNNING', 'tag': tag, 'mode': mode, 'cases': [], 'fixtures': {}, 'contracts': {},
        'protected_test_guarded': protected, 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), HERE/'test_experiment.py', HERE/'PLAN_KO.md', OLD/'component_experiment.py', ROOT/'research/past_only_repr_diagnosis_20261012/common.py')},
        'primary_metric': 'Normalized observed channel-macro MSE', 'checkpoint_choice': 'TUNE only; earliest exact tie',
        'scope': 'Repeatedly used development VAL, not independent generalization', 'H': H, 'FIT': 256, 'TUNE': 64,
        'device': torch.cuda.get_device_name(), 'dtype': 'float32', 'tf32_matmul': False, 'tf32_cudnn': False}
    try:
        source_folder = output/'sources'
        source_folder.mkdir()
        result['source_snapshots'] = []
        for source_path, digest in result['source_hashes'].items():
            snapshot = source_folder/Path(source_path).name
            with snapshot.open('xb') as f:
                f.write((ROOT/source_path).read_bytes())
            assert sha(snapshot) == digest
            result['source_snapshots'].append({'source': source_path, 'snapshot': str(snapshot.relative_to(ROOT)), 'sha256': digest})
        started = time.perf_counter()
        bolt = previous.FrozenBolt()
        result['bolt_load_wall_seconds'] = time.perf_counter()-started
        result['bolt_initial_hash'] = bolt.initial_hash
        for dataset in ('jena', 'robin'):
            data = load_data(dataset, sizes=(256, 64, None))
            raw_history, target, observed, history_observed = arrays(data, 512)
            contract = {'dataset': dataset, 'source': str(Path(data['source']).relative_to(ROOT)), 'source_sha256': data['source_hash'],
                'channels': data['columns'], 'mean': data['mean'], 'std': data['std'], 'origins': data['origins'],
                'origin_hashes': {s: ahash(v) for s, v in data['origins'].items()},
                'bounds': {s: data[s+'_bounds'] for s in ('fit', 'tune', 'val')},
                'input_boundary': 'Causal FF resets at each actual L-window start to FIT mean; no payload from missing values',
                'normalization_fit_scope': 'FIT_ONLY', 'order_seed': SEEDS, 'observed_target_masks': {s: ahash(v) for s, v in observed.items()}}
            write_new(output/(dataset+'_contract.json'), contract)
            result['contracts'][dataset] = contract
            result['fixtures'][dataset] = fixture(data, raw_history, target, observed, bolt, limit)
            if mode == 'preflight':
                continue
            data_output, data_folder = output/dataset, folder/dataset
            data_output.mkdir()
            data_folder.mkdir()
            stage0 = g0(data, raw_history, target, observed, bolt, data_output, data_folder, limit)
            result['cases'].extend(stage0)
            write_new(data_output/'G0_RESULT.json', {'status': 'COMPLETE', 'cases': stage0})
            quarter = data['x'].shape[1]//4
            stage1 = []
            for seed in SEEDS:
                for objective in ('REC', 'PRED'):
                    for config in CONFIGS:
                        spec = {'stage': 'G1', 'objective': objective, 'config': config, 'k': quarter, 'length': 512,
                                'rf': 127, 'init': 'PCA', 'seed': seed}
                        stage1.extend(safe_fit_case(data, spec, bolt, data_output, data_folder, limit, raw_history, result))
            result['cases'].extend(stage1)
            matches = {}
            for seed in SEEDS:
                seed_cases = [c for c in stage1 if c['seed']==seed]
                hashes = {c['training']['initial_codec_hash'] for c in seed_cases}
                orders = {c['training']['order_sha256'] for c in seed_cases}
                if len(hashes) != 1 or len(orders) != 1:
                    raise AssertionError('G1_CONFIG_OBJECTIVE_INITIALIZATION_OR_ORDER_UNMATCHED')
                matches[str(seed)] = {'initial_codec_hash': next(iter(hashes)), 'order_sha256': next(iter(orders)),
                                     'matched_across_REC_PRED_and_three_configs': True}
            stage1_failures = [f for f in result.get('failed_paths', []) if f['dataset']==dataset and f['spec']['stage']=='G1']
            write_new(data_output/'G1_RESULT.json', {'status': 'PARTIAL_FAILED_PATHS' if stage1_failures else 'COMPLETE',
                'cases': stage1, 'matched_contracts': matches, 'failed_paths': stage1_failures})
            tune_config = {}
            for config in CONFIGS:
                values = [next(cp for cp in c['training']['checkpoints'] if cp['update']==c['selected_update'])['forecast']['tune']['normalized']['mse']
                          for c in stage1 if c['objective']=='PRED' and c['config']==config]
                tune_config[config] = float(np.mean(values)) if len(values)==2 else None
            complete_configs = [c for c in CONFIGS if tune_config[c] is not None]
            if tune_config['BASE'] is None:
                chosen, candidate, improvement = 'BASE', None, None
            else:
                minimum = min(tune_config[c] for c in complete_configs)
                candidate = next(c for c in complete_configs if tune_config[c] <= minimum+1e-12)
                improvement = (tune_config['BASE']-tune_config[candidate])/tune_config['BASE']
                chosen = candidate if improvement >= .01 else 'BASE'
            selection = {'dataset': dataset, 'chosen_config': chosen, 'seed_mean_tune_normalized_forecast_mse': tune_config,
                'candidate_minimum': candidate, 'candidate_relative_improvement_vs_BASE': improvement,
                'criterion': 'Mean of seed-specific TUNE-selected PRED TUNE scores; earliest config within1e-12 tie; require1% improvement otherwise BASE; no VAL used',
                'note': 'Additional TUNE reuse creates selection optimism; not external generalization'}
            write_new(data_output/'G2_CONFIG_SELECTION.json', selection)
            result.setdefault('stage2_configuration_selection', {})[dataset] = selection
            stage2 = []
            for seed in SEEDS:
                variations = [(data['x'].shape[1]//2, 512, 127, 'PCA'), (data['x'].shape[1], 512, 127, 'PCA'),
                              (data['x'].shape[1], 512, 127, 'IDENTITY'), (quarter, 128, 127, 'PCA'), (quarter, 512, 31, 'PCA')]
                for k, length, rf, init in variations:
                    spec = {'stage': 'G2', 'objective': 'PRED', 'config': chosen, 'k': k, 'length': length,
                            'rf': rf, 'init': init, 'seed': seed}
                    stage2.extend(safe_fit_case(data, spec, bolt, data_output, data_folder, limit, raw_history, result))
            result['cases'].extend(stage2)
            stage2_failures = [f for f in result.get('failed_paths', []) if f['dataset']==dataset and f['spec']['stage']=='G2']
            write_new(data_output/'G2_RESULT.json', {'status': 'PARTIAL_FAILED_PATHS' if stage2_failures else 'COMPLETE',
                'cases': stage2, 'configuration_selection': selection, 'failed_paths': stage2_failures})
            result['cases'].append(raw_case(data, raw_history, target, observed, bolt, data_output, data_folder, limit))
            if previous.state_hash(bolt.model) != bolt.initial_hash:
                raise AssertionError('FROZEN_BOLT_HASH_CHANGED')
        result['bolt_final_hash'] = previous.state_hash(bolt.model)
        for source_path, digest in result['source_hashes'].items():
            if sha(ROOT/source_path) != digest:
                raise AssertionError('EXECUTION_SOURCE_FILE_CHANGED: '+source_path)
        result['source_hashes_unchanged_at_completion'] = True
        result['status'] = 'PARTIAL_FAILED_PATHS' if result.get('failed_paths') else 'COMPLETE'
        limit.check()
        limit.close(result['status'])
        result['resources'] = limit.record
        result['new_training_runs'] = len({(c['dataset'], c['seed'], c['config'], c['objective'], c['k'], c['length'], c['receptive_field'], c['initialization']['kind'])
                                           for c in result['cases'] if 'training' in c})
        result['method_count'] = sum(len(c['methods']) for c in result['cases'])
        write_new(output/'RESULT.json', result)
        print(json.dumps({'status': result['status'], 'method_count': result['method_count'], 'resources': limit.record}), flush=True)
    except BaseException as error:
        result.update(status='FAILED_PRESERVED', error=str(error), traceback=traceback.format_exc())
        limit.close('FAILED_PRESERVED', str(error))
        result['resources'] = limit.record
        write_new(output/'FAILURE.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--mode', choices=('preflight', 'all'), default='all')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Alphanumeric/underscore tag required')
    run(args.tag, args.mode)
