"""Full deployment wall time, prediction parity and active route metadata."""
import time
import numpy as np
import torch
import psutil
import model as bm
import probes as pr


def preprocess(pack, physical, observed):
    return bm.en.ex.preprocess(physical, observed, pack['mean'], pack['std'])


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device='cuda')


def _blocks(x):
    return torch.cat((x.reshape(len(x), 32, 16, x.shape[-1]).mean(2), x[:, -1:]), dim=1)


def load_ridge(record, rt):
    artifact = record['model']
    path = rt.ROOT / artifact['path']
    if rt.sha(path) != artifact['sha256']: raise ValueError('COST_RIDGE_MODEL_SHA')
    with np.load(path, allow_pickle=False) as z:
        return {key: _tensor(z[key]) for key in ('weight', 'bias')}


def deployment(pack, physical, observed, origins, route, forecaster, model=None, ridge=None):
    normalized = preprocess(pack, physical, observed)
    x = _tensor(normalized)
    if route == 'CORE':
        prediction = model(x, torch.as_tensor(origins, dtype=torch.long, device='cuda'), forecaster)
    elif route in ('E_RAW_RIDGE', 'E_ENRICHED_RIDGE'):
        if route == 'E_RAW_RIDGE': z = model.codec.encode(x)
        else: z, _ = model.encode_with_memory(x, torch.as_tensor(origins, dtype=torch.long, device='cuda'))
        features = torch.cat((z.flatten(1), x[:, -1]), dim=1)
        prediction = x[:, -1:] + (features @ ridge['weight'] + ridge['bias']).reshape(len(x), 48, x.shape[-1])
    elif route == 'RAW_SELF33':
        features = _blocks(x).transpose(1, 2)
        delta = torch.einsum('bct,cth->bhc', features, ridge['weight']) + ridge['bias'].T
        prediction = x[:, -1:] + delta
    elif route == 'RAW_BOLT':
        prediction = forecaster(x)
    elif route in ('RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R'):
        delta = (_blocks(x).flatten(1) @ ridge['weight'] + ridge['bias']).reshape(len(x), 48, x.shape[-1])
        prediction = (forecaster(x) if route == 'RAW_BOLT_RAW_R' else x[:, -1:]) + delta
    else:
        raise ValueError('UNKNOWN_DEPLOYMENT_ROUTE')
    return (prediction * _tensor(pack['std']) + _tensor(pack['mean'])).cpu().numpy()


def cost(pack, route, expected, split, forecaster, limit, model=None, ridge=None, identity=None):
    records = []
    for batch in (1, 8):
        if len(expected) < batch: raise ValueError('INSUFFICIENT_COST_WINDOWS')
        physical = pack['raw_history'][split][:batch]
        observed = pack['raw_history_observed'][split][:batch]
        origins = pack['origins'][split][:batch]
        with torch.no_grad():
            for _ in range(2): deployment(pack, physical, observed, origins, route, forecaster, model, ridge)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(7):
                torch.cuda.synchronize(); started = time.perf_counter()
                actual = deployment(pack, physical, observed, origins, route, forecaster, model, ridge)
                torch.cuda.synchronize(); timings.append(time.perf_counter() - started)
        physical_reference = expected[:batch] * pack['std'] + pack['mean']
        error = np.abs(actual - physical_reference)
        passed = bool(np.all(error <= 2e-3 + 2e-5 * np.abs(physical_reference)))
        normalized_error = np.abs((actual - pack['mean']) / pack['std'] - expected[:batch])
        records.append({'identity': identity, 'dataset': pack['dataset'], 'route': route, 'split': split,
            'batch': batch, 'warmup': 2, 'repeats': 7, 'seconds': timings,
            'median_seconds': float(np.median(timings)), 'median_seconds_per_window': float(np.median(timings)) / batch,
            'physical_max_abs_vs_science': float(error.max()), 'normalized_max_abs_vs_science': float(normalized_error.max()),
            'physical_parity_rtol': 2e-5, 'physical_parity_atol': 2e-3, 'parity_pass': passed,
            'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(), 'rss_bytes': psutil.Process().memory_info().rss,
            'model_metadata': None if model is None else model.metadata(),
            'includes': 'CPU L512 window-local causal FF/FIT normalization/H2D/active route/inverse FIT normalization/D2H/sync',
            'excludes': 'Disk and model load, FIT memory-key precomputation, teacher future AE or removed auxiliary heads',
            'memory_note': 'Frozen Bolt and full current checkpoint co-resident; total peak is not exclusive incremental route memory'})
        if not passed: raise ValueError('DEPLOYMENT_SCIENCE_PARITY_FAILURE: ' + str(records[-1]))
        limit.check()
    return records
