"""Repeated full-output deployment cost with verified LoRA merging."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

from model import make_model, pca_basis
from run import HERE, ROOT, GPUJob, digest, load_data, save_json, source_receipt


def cuda_cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}


def process_gpu_memory():
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout
    return [line for line in output.splitlines() if line.split(',')[0].strip() == str(os.getpid())]


@torch.no_grad()
def load_candidate(spec, data, inputs, job):
    if spec.get('checkpoint'):
        path = ROOT / spec['checkpoint']
        assert digest(path) == spec['checkpoint_sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        config, basis = saved['spec'], saved['basis']
        model = make_model(config['arm'], basis, residual_rank=config.get('residual_rank', 32),
                           ed_mode=config.get('ed_mode', 'current'))
        model.restore_adapter(saved['state'])
    else:
        basis = pca_basis(data['x'][int(data['train_start']):int(data['train_end'])], 8 if spec['dataset'] == 'electricity' else 4)
        model = make_model('f0', basis)
    model.eval()
    info = {'registered_trainable_parameters_before_merge': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters_before_merge': sum(p.numel() for p in model.parameters()),
            'task_channels': inputs.shape[-1], 'backbone_series_per_origin': inputs.shape[-1] if model.arm in ('f0', 'lora') else model.encoder.out_features}
    before = None
    if spec.get('checkpoint'):
        before = torch.cat([model(x.to('cuda')).cpu() for x in inputs.split(4)])
        with np.load(path.with_name('best_val.npz'), allow_pickle=False) as archive:
            assert np.array_equal(archive['origins'], data['val_origins'])
            index = np.linspace(0, len(data['val_origins']) - 1, 24, dtype=int)
            stored = torch.from_numpy(archive['prediction'][index])
        torch.testing.assert_close(before, stored, atol=1e-6, rtol=1e-5)
        info['saved_validation_replay_max_abs'] = float((before - stored).abs().max())
    if model.arm == 'lora':
        assert before is not None
        job.heartbeat(f'{spec["id"]} LoRA pre-merge parity')
        model.backbone = model.backbone.merge_and_unload(safe_merge=True)
        after = []
        for x in inputs.split(4):
            after.append(model(x.to('cuda')).cpu())
        after = torch.cat(after)
        error = float((before - after).abs().max())
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)
        info['merge_parity'] = {'pass': True, 'max_abs_error': error, 'atol': 1e-5, 'rtol': 1e-4,
                                'origins': len(inputs), 'comparison': 'same FP32 VAL inputs before/after safe_merge; original checkpoint preserved'}
    info['total_deployed_parameters'] = sum(p.numel() for p in model.parameters())
    info['deployed_parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    return model, info


@torch.no_grad()
def infer(model, chunk):
    output = model(chunk.to(device='cuda', dtype=torch.float32)).float().cpu()
    if output.shape != (len(chunk), 48, chunk.shape[-1]):
        raise RuntimeError('Benchmark must return the complete task output')
    return output


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def measure(args):
    output = HERE / f'{args.name}.json'
    assert not output.exists(), 'Do not overwrite a completed measurement'
    specs_path = Path(args.specs)
    specs = json.loads(specs_path.read_text(encoding='utf-8'))
    assert len({s['id'] for s in specs}) == len(specs)
    rows, inputs_by_dataset, data_by_dataset, origin_lists = [], {}, {}, {}
    for dataset in sorted({s['dataset'] for s in specs}):
        data = load_data(dataset)
        origins = data['val_origins'][np.linspace(0, len(data['val_origins']) - 1, 24, dtype=int)]
        assert len(np.unique(origins)) == 24
        inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]))
        assert inputs.dtype == torch.float32
        data_by_dataset[dataset], inputs_by_dataset[dataset], origin_lists[dataset] = data, inputs, origins.tolist()
    with GPUJob(args.name) as job:
        for block in range(3):
            order = list(range(len(specs)))
            if block == 1:
                order.reverse()
            elif block == 2:
                half = len(order) // 2
                order = order[half:] + order[:half]
            for ordinal, index in enumerate(order):
                spec = specs[index]
                baseline = cuda_cleanup()
                if baseline['allocated']:
                    raise RuntimeError(f'Previous model still owns CUDA allocations: {baseline}')
                inputs = inputs_by_dataset[spec['dataset']]
                model, info = load_candidate(spec, data_by_dataset[spec['dataset']], inputs, job)
                job.heartbeat(f'cost block{block} {spec["id"]}')
                for batch_size in (1, 4):
                    torch.cuda.synchronize()
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.empty_cache()
                    chunks = list(inputs.split(batch_size))
                    for warmup in range(10):
                        infer(model, chunks[warmup % len(chunks)])
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    durations = []
                    for repeat in range(20):
                        job.check_limits()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for chunk in chunks:
                            infer(model, chunk)
                        torch.cuda.synchronize()
                        durations.append(time.perf_counter() - started)
                    row = {'id': spec['id'], 'dataset': spec['dataset'], 'seed': spec.get('seed'),
                           'roles': spec.get('roles', []),
                           'block': block, 'order_position': ordinal, 'batch_origins': batch_size,
                           'seconds_per_24_origins': durations,
                           'milliseconds_per_origin': summary([t * 1000 / 24 for t in durations]),
                           'origins_per_second': summary([24 / t for t in durations]),
                           'baseline_cuda_bytes': baseline, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                           'resident_allocated_bytes': torch.cuda.memory_allocated(),
                           'process_cpu_rss_bytes': psutil.Process().memory_info().rss,
                           'process_gpu_memory_nvidia_smi': process_gpu_memory(), **info}
                    rows.append(row)
                    save_json(HERE / f'{args.name}_partial.json', {'status': 'running', 'rows': rows})
                    job.heartbeat(f'cost block{block} {spec["id"]} batch{batch_size} complete')
                del model
                released = cuda_cleanup()
                assert released['allocated'] == 0, released
                print(json.dumps({'block': block, 'id': spec['id'], 'complete': True}), flush=True)
    save_json(output, {'status': 'complete', 'specs_sha256': digest(specs_path), 'source': source_receipt(),
                       'rows': rows, 'val_origins': origin_lists, 'torch_threads': torch.get_num_threads(),
                       'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
                       'cuda_version': torch.version.cuda, 'float32_matmul_precision': torch.get_float32_matmul_precision(),
                       'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                       'precision': 'float32', 'warmup_calls': 10, 'passes_per_order_block': 20, 'order_blocks': 3,
                       'memory_scope': 'Before each batch shape, synchronize, clear cuBLAS workspaces and empty unused CUDA cache; warm up that shape, then reset peaks. Timed inference retains normal warmed cuBLAS workspaces. Allocated includes resident model, active tensors and workspaces; reserved includes the warmed allocator pool. RSS is the entire process.',
                       'timed_scope': 'common standardized CPU tensor -> H2D -> model -> entire CPU output; synchronized. Excludes model loading, disk I/O, scoring and ledger writes. No precomputed backbone output.',
                       'uncertainty': 'Repeated timing on one device; blocks/repeats are not independent dataset evidence.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specs', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        raise ValueError('Invalid measurement name')
    paths = [HERE / f'{args.name}{suffix}.json' for suffix in ('', '_partial', '_attempt')]
    if any(p.exists() for p in paths):
        raise RuntimeError('Use a new name; preserve completed or failed cost attempts')
    record = {'status': 'running', 'pid': os.getpid(), 'started_utc': time.time(),
              'specs_sha256': digest(Path(args.specs)), 'command': ['measure_cost.py', '--specs', args.specs, '--name', args.name]}
    save_json(paths[-1], record)
    try:
        measure(args)
        record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['ended_utc'] = time.time()
        save_json(paths[-1], record)


if __name__ == '__main__':
    main()
