"""CPU FP32 deployment cost of the existing shared/factor ridge checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import psutil
import torch
from torch import nn
from threadpoolctl import threadpool_limits, threadpool_info

from run import HERE, ROOT, V1, digest, environment_receipt, load_data, source_receipt


CACHE1 = ROOT / '.cache/tsfm_peft_development_20260926'


class LinearForecast(nn.Module):
    def __init__(self, weights, arm):
        super().__init__()
        self.arm = arm
        keys = ['weight', 'bias'] if arm == 'shared_linear' else [
            'basis', 'common_weight', 'common_bias', 'residual_weight', 'residual_bias']
        for key in keys:
            self.register_buffer(key, torch.as_tensor(weights[key], dtype=torch.float32).clone())

    def temporal(self, x, weight, bias):
        return (x.transpose(1, 2) @ weight + bias).transpose(1, 2)

    def forward(self, x):
        if self.arm == 'shared_linear':
            return self.temporal(x, self.weight, self.bias)
        z = x @ self.basis
        residual = x - z @ self.basis.T
        return (self.temporal(z, self.common_weight, self.common_bias) @ self.basis.T
                + self.temporal(residual, self.residual_weight, self.residual_bias))


def numpy_reference(x, weights, arm):
    x = np.asarray(x, dtype=np.float64)
    def temporal(v, weight, bias):
        return np.einsum('blc,lh->bhc', v, weight) + bias[None, :, None]
    if arm == 'shared_linear':
        return temporal(x, weights['weight'], weights['bias'])
    basis = np.asarray(weights['basis'], dtype=np.float64)
    z = x @ basis
    residual = x - z @ basis.T
    return (temporal(z, weights['common_weight'], weights['common_bias']) @ basis.T
            + temporal(residual, weights['residual_weight'], weights['residual_bias']))


def verify_sources(report):
    checked = {}
    for name, expected in report['source'].items():
        path = V1 / name
        actual = digest(path)
        if actual == expected:
            checked[name] = {'expected_sha256': expected, 'current_sha256': actual, 'match': True}
        elif name == 'model.py':
            revision = '7b1953825fb2cffda07ea25aa2583417ad89e677'
            blob = subprocess.check_output(['git', 'show', revision + ':research/tsfm_peft_development_20260926/model.py'], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError('historical model source hash mismatch')
            checked[name] = {'expected_sha256': expected, 'current_sha256': actual,
                             'historical_commit': revision, 'historical_match': True,
                             'note': 'full-TRAIN linear code has its own PCA; this historical source entry was updated later for neural masked loss'}
        else:
            raise ValueError(f'baseline source hash mismatch: {name}')
    return checked


def load_weights(dataset, arm):
    if dataset == 'electricity':
        report_path = V1 / 'linear_fulltrain_baselines.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        source_check = verify_sources(report)
        penalty = report['selected'][arm]['ridge_penalty']
        if penalty != 0.001:
            raise ValueError('existing baseline selection changed')
        diagnostic = json.loads((V1 / 'residual_rank_diagnostics.json').read_text(encoding='utf-8'))
        path = ROOT / diagnostic['baseline_weights_file']
        expected = diagnostic['baseline_weights_sha256']
    else:
        report_path = V1 / 'protected_evaluation.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        source_check = {}
        for name, expected_source in report['evaluation_source_hashes'].items():
            actual = digest(V1 / name)
            if actual != expected_source:
                raise ValueError(f'protected source changed: {name}')
            source_check[name] = {'sha256': actual, 'match': True}
        spec = report['baselines'][arm]
        path = CACHE1 / 'protected_evaluation' / spec['weights_cache']
        expected = spec['weights_sha256']
    if digest(path) != expected:
        raise ValueError('cached baseline weight hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    if dataset == 'electricity' and arm == 'shared_linear':
        weights = {'weight': weights['direct_weight'], 'bias': weights['direct_bias']}
    receipt = {'weights_file': path.relative_to(ROOT).as_posix(), 'weights_sha256': expected,
               'report_file': report_path.relative_to(ROOT).as_posix(), 'report_sha256': digest(report_path),
               'source_verification': source_check}
    return weights, receipt


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def cpu_memory():
    info = psutil.Process().memory_info()
    return {key: int(getattr(info, key)) for key in ('rss', 'vms', 'peak_wset') if hasattr(info, key)}


@torch.no_grad()
def infer(model, chunk):
    output = model(chunk.to(device='cpu', dtype=torch.float32)).float().cpu()
    if output.shape != (len(chunk), 48, chunk.shape[-1]) or output.device.type != 'cpu':
        raise ValueError('CPU benchmark must return all task channels and horizon')
    return output


def measure():
    started = time.perf_counter()
    cpu_started = time.process_time()
    torch.set_num_threads(4)
    models, inputs, origins_by_dataset, receipts, parity = {}, {}, {}, {}, {}
    for dataset in ['electricity', 'bull']:
        data = load_data(dataset)
        origins = data['val_origins'][np.linspace(0, len(data['val_origins']) - 1, 24, dtype=int)]
        if len(np.unique(origins)) != 24:
            raise ValueError('24 distinct VAL origins required')
        x = np.stack([data['x'][o - 512:o] for o in origins])
        if x.dtype != np.float32:
            raise ValueError('common standardized input must be FP32')
        inputs[dataset] = torch.from_numpy(x)
        origins_by_dataset[dataset] = origins.tolist()
        for arm in ['shared_linear', 'factor_linear']:
            key = dataset + '_' + arm
            weights, receipt = load_weights(dataset, arm)
            m = LinearForecast(weights, arm).eval()
            reference = numpy_reference(x, weights, arm)
            actual = infer(m, inputs[dataset]).numpy().astype(np.float64)
            absolute = np.abs(actual - reference)
            ok = bool(np.allclose(actual, reference, atol=1e-5, rtol=1e-4))
            parity[key] = {'pass': ok, 'atol': 1e-5, 'rtol': 1e-4,
                           'max_abs_error': float(absolute.max()), 'mean_abs_error': float(absolute.mean()),
                           'reference': 'original affine/factor formula in NumPy float64, same existing weights and 24 VAL inputs; no target scoring'}
            if not ok:
                raise ValueError(f'FP32 linear deployment parity failed: {key}: {parity[key]}')
            models[key], receipts[key] = m, receipt
    keys = list(models)
    rows = []
    for block in range(3):
        order = keys.copy()
        if block == 1:
            order.reverse()
        elif block == 2:
            order = order[2:] + order[:2]
        for ordinal, key in enumerate(order):
            dataset = key.split('_')[0]
            m = models[key]
            gc.collect()
            baseline_memory = cpu_memory()
            coefficients = sum(b.numel() for name, b in m.named_buffers() if name != 'basis')
            for batch_size in [1, 4]:
                chunks = list(inputs[dataset].split(batch_size))
                for warmup in range(10):
                    infer(m, chunks[warmup % len(chunks)])
                durations = []
                for _ in range(20):
                    tick = time.perf_counter()
                    for chunk in chunks:
                        infer(m, chunk)
                    durations.append(time.perf_counter() - tick)
                rows.append({'id': key, 'dataset': dataset, 'arm': m.arm, 'seed': None,
                             'block': block, 'order_position': ordinal, 'batch_origins': batch_size,
                             'seconds_per_24_origins': durations,
                             'milliseconds_per_origin': summary([t * 1000 / 24 for t in durations]),
                             'origins_per_second': summary([24 / t for t in durations]),
                             'fitted_coefficient_count': coefficients,
                             'fixed_basis_elements': m.basis.numel() if hasattr(m, 'basis') else 0,
                             'deployed_tensor_elements': sum(b.numel() for b in m.buffers()),
                             'deployed_tensor_bytes': sum(b.numel() * b.element_size() for b in m.buffers()),
                             'torch_requires_grad_parameters': 0,
                             'baseline_process_cpu_memory_bytes': baseline_memory,
                             'after_process_cpu_memory_bytes': cpu_memory(),
                             'memory_scope': 'process RSS/peak working set include interpreter, loaded data and four tiny linear models; no model-specific CPU allocator peak is claimed'})
    return {'status': 'complete', 'rows': rows, 'parity': parity, 'weight_receipts': receipts,
            'val_origins': origins_by_dataset, 'source': source_receipt(), 'environment': environment_receipt(),
            'torch_threads': torch.get_num_threads(), 'threadpools': threadpool_info(),
            'precision': 'float32', 'warmup_calls': 10, 'passes_per_order_block': 20, 'order_blocks': 3,
            'timed_scope': 'common standardized CPU tensor -> complete CPU FP32 linear output; excludes loading, disk I/O, target scoring and logs',
            'wall_seconds': time.perf_counter() - started, 'cpu_process_seconds': time.process_time() - cpu_started,
            'new_fits': 0, 'new_target_scores': 0,
            'accuracy_reference': 'existing v1 factor/shared scores; deployment conversion checked on 24 VAL inputs, no new performance score',
            'uncertainty': 'three order blocks are repeated timing, not independent data evidence; process memory is not directly comparable with CUDA allocated bytes'}


def synthetic_check():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 512, 5)).astype(np.float32)
    weights = {'basis': rng.normal(size=(5, 2)) / 3,
               'common_weight': rng.normal(size=(512, 48)) / 100,
               'common_bias': rng.normal(size=48) / 100,
               'residual_weight': rng.normal(size=(512, 48)) / 100,
               'residual_bias': rng.normal(size=48) / 100,
               'weight': rng.normal(size=(512, 48)) / 100, 'bias': rng.normal(size=48) / 100}
    for arm in ['factor_linear', 'shared_linear']:
        actual = infer(LinearForecast(weights, arm), torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual, numpy_reference(x, weights, arm), atol=1e-5, rtol=1e-4)
    print('CPU synthetic shared/factor FP32-vs-float64 parity PASS; no real data or GPU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-check', action='store_true')
    parser.add_argument('--name', default='linear_cost')
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        raise ValueError('Invalid measurement name')
    with threadpool_limits(limits=4):
        if args.synthetic_check:
            synthetic_check()
            return
        output = HERE / f'{args.name}.json'
        if output.exists():
            raise FileExistsError('preserve the existing linear cost attempt')
        with output.open('x', encoding='utf-8', newline='\n') as f:
            try:
                report = measure()
            except BaseException:
                json.dump({'status': 'failed', 'error': traceback.format_exc(), 'source': source_receipt()}, f, indent=2)
                f.write('\n')
                raise
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        print(json.dumps({'status': report['status'], 'rows': len(report['rows']), 'wall_seconds': report['wall_seconds'], 'new_fits': 0}))


if __name__ == '__main__':
    main()
