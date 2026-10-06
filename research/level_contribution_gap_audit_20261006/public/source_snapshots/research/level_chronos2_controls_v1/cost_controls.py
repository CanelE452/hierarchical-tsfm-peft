"""Same-campaign CPU-input to CPU-output inference costs; no fitting.

The timing, cleanup, and block aggregation follow the v12 cost runner. The
model/data bindings and finite grid belong only to this new control campaign.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from runtime import (HERE, CACHE, DATASETS, Job, artifact, check_seal, configure,
                     digest, environment, read_json, save_json)


PASSES, WARMUP, BLOCKS, N_ORIGINS = 10, 10, 3, 24
UNCOMPRESSED = ('F0', 'MSE_LORA')
ZERO_SHOT = ('F0', 'C2_MV', 'C2_SMALL_MV')


def cleanup(device):
    import torch
    gc.collect()
    if device == 'cuda':
        torch.cuda.synchronize()
        if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_memory():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {name: int(getattr(info, name)) for name in ('rss', 'vms', 'peak_wset')
                if hasattr(info, name)}
    except (ImportError, OSError):
        return None


def telemetry():
    fields = 'index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate'
    result = {}
    for name, query in (('gpu', ['--query-gpu=' + fields, '--format=csv,noheader,nounits']),
                        ('processes', ['--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'])):
        try:
            run = subprocess.run(['nvidia-smi', *query], capture_output=True,
                                 text=True, timeout=15)
            result[name] = {'status': run.returncode, 'rows': run.stdout.strip().splitlines(),
                            'error': run.stderr.strip() or None}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {'status': None, 'rows': None, 'error': repr(exc)}
    return {'utc': time.time(), 'fields': fields, **result,
            'unavailable_values': 'N/A is unavailable, never zero'}


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(dataset):
    import torch
    from baselines import load_data
    data = load_data(dataset, include_test=False)
    all_origins = np.asarray(data['val_origins'], dtype=np.int64)
    if len(all_origins) < N_ORIGINS or np.any(np.diff(all_origins) <= 0):
        raise ValueError('Cost requires the first 24 chronological VAL origins')
    origins = all_origins[:N_ORIGINS]
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]).astype(np.float32))
    if tuple(inputs.shape) != (N_ORIGINS, 512, data['_C']):
        raise ValueError('Bound cost input axes changed')
    return {'inputs': inputs, 'origins': origins, 'C': int(data['_C']), 'K': int(data['_K']),
            'input_sha256': array_hash(inputs.numpy()), 'origin_sha256': array_hash(origins),
            'trainval': artifact(data['_path'])}


def configs(method, unit, device):
    c, k = unit['C'], unit['K']
    if method == 'DIRECT_NLINEAR':
        return [{'batch': b, 'chunk_rows': None, 'execution': 'direct'} for b in (1, 4)]
    if device != 'cuda':
        raise ValueError('Only direct NLinear has a prescribed CPU cost path')
    if method in UNCOMPRESSED:
        return [{'batch': 1, 'chunk_rows': c, 'execution': 'full'},
                {'batch': 1, 'chunk_rows': k, 'execution': 'chunk_k'},
                {'batch': 4, 'chunk_rows': 4 * c, 'execution': 'full'},
                {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'chunk_4k'},
                {'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]
    if method == 'LEVEL':
        return [{'batch': 1, 'chunk_rows': k, 'execution': 'full'},
                {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'full'}]
    if method in ('C2_MV', 'C2_SMALL_MV'):
        return [{'batch': b, 'chunk_rows': None, 'execution': 'full_group'} for b in (1, 4)]
    raise ValueError('Unapproved cost method: ' + method)


def rotated(values, block):
    if block == 1:
        return list(reversed(values))
    if block == 2:
        return list(values[1:]) + list(values[:1])
    return list(values)


def instances_by_dataset():
    from baselines import baseline_rows
    result = {}
    for dataset in DATASETS:
        items = [dict(row) for row in baseline_rows(dataset)
                 if row['method'] in ('LEVEL', 'F0', 'MSE_LORA', 'DIRECT_NLINEAR')]
        expected = {'LEVEL': 2, 'F0': 1, 'MSE_LORA': 2, 'DIRECT_NLINEAR': 2}
        if {name: sum(row['method'] == name for row in items) for name in expected} != expected:
            raise ValueError('The prescribed selected baseline instances are incomplete')
        items += [{'id': dataset + '_' + method, 'dataset': dataset, 'method': method,
                   'seed': None, 'model_key': key}
                  for method, key in (('C2_MV', 'C2'), ('C2_SMALL_MV', 'C2_SMALL'))]
        result[dataset] = items
    return result


def planned_grid(instances, prepared, device):
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(list(DATASETS), block):
            items = instances[dataset]
            if device == 'cpu':
                items = [row for row in items if row['method'] == 'DIRECT_NLINEAR']
            ordered = []
            if device == 'cuda':
                f0 = next(row for row in items if row['method'] == 'F0')
                ordered.append((f0, 'pre'))
            ordered.extend((row, None) for row in rotated(items, block))
            if device == 'cuda':
                ordered.append((f0, 'post'))
            for item, sentinel in ordered:
                grid = configs(item['method'], prepared[dataset], device)
                grid = [entry for entry in grid if entry['execution'] == 'full'] if sentinel else rotated(grid, block)
                for config in grid:
                    row = {name: item[name] for name in ('id', 'dataset', 'method', 'seed')}
                    row.update(device=device, block=block, sentinel=sentinel, **config)
                    row['key'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
                    row['planned_sequence'] = len(rows)
                    rows.append(row)
    expected = 279 if device == 'cuda' else 36
    if len(rows) != expected or len({row['key'] for row in rows}) != expected:
        raise ValueError('The prespecified cost grid changed')
    return rows


def call_counts(rows):
    values = []
    for row in rows:
        batch = row['batch']
        timed = PASSES * (N_ORIGINS // batch)
        values.append({'key': row['key'], 'dataset': row['dataset'], 'method': row['method'],
                       'batch': batch, 'sentinel': row['sentinel'],
                       'warmup_wrapper_calls': WARMUP, 'timed_wrapper_calls': timed,
                       'total_wrapper_calls': timed + WARMUP,
                       'timed_origin_units': PASSES * N_ORIGINS,
                       'warmup_origin_units': WARMUP * batch})
    return {'rows': values, 'measurement_rows': len(rows),
            'sentinel_rows': sum(row['sentinel'] is not None for row in rows),
            'total_wrapper_calls': sum(row['total_wrapper_calls'] for row in values),
            'timed_origin_units': sum(row['timed_origin_units'] for row in values),
            'warmup_origin_units': sum(row['warmup_origin_units'] for row in values),
            'not_included': ['preflight', 'model_load', 'merge/export parity'],
            'interpretation': 'Wrapper calls and origin units are not independent statistical samples or model.forward calls'}


def state_bytes(module):
    parameters = sum(p.numel() * p.element_size() for p in module.parameters())
    buffers = sum(p.numel() * p.element_size() for p in module.buffers())
    return {'deployed_parameters': sum(p.numel() for p in module.parameters()),
            'parameter_bytes': parameters, 'buffer_bytes': buffers,
            'deployment_tensor_bytes': parameters + buffers}


def load_deployment(row, device):
    cleanup(device)
    if row['method'] in ('C2_MV', 'C2_SMALL_MV'):
        from c2_models import C2Forecast
        key = 'C2' if row['method'] == 'C2_MV' else 'C2_SMALL'
        model_info = read_json(HERE / 'model_manifest.json')['models'][key]
        wrapper = C2Forecast(model_info['snapshot'], relation='MV', device=device)
    else:
        from baselines import load_baseline
        wrapper = load_baseline(row, device=device, deploy=True)
    wrapper.module.eval()
    wrapper.module.requires_grad_(False)
    cleanup(device)
    info = {'id': row['id'], 'method': row['method'], 'dataset': row['dataset'],
            'seed': row['seed'], **state_bytes(wrapper.module)}
    if hasattr(wrapper, 'receipt'):
        info['deployment_receipt'] = wrapper.receipt() if callable(wrapper.receipt) else wrapper.receipt
    return wrapper, info


def forward_cpu_output(wrapper, cpu_input):
    import torch
    result = wrapper(cpu_input)
    if not isinstance(result, torch.Tensor):
        result = torch.from_numpy(np.asarray(result))
    result = result.float().cpu().contiguous()
    if tuple(result.shape) != (len(cpu_input), 48, cpu_input.shape[2]):
        raise ValueError('Cost output must contain every original channel and H48')
    return result


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def measure(wrapper, prepared, row, device, job):
    import torch
    groups = list(prepared['inputs'].split(row['batch']))
    if row['method'] not in ('C2_MV', 'C2_SMALL_MV'):
        wrapper.set_chunk(row['chunk_rows'])
    elif row['chunk_rows'] is not None:
        raise ValueError('A native multivariate group cannot be channel-chunked')
    cleanup(device)
    resident = {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved()} if device == 'cuda' else None
    for index in range(WARMUP):
        job.check_limits()
        forward_cpu_output(wrapper, groups[index % len(groups)])
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(PASSES):
        job.heartbeat({'phase': 'cost_pass', 'key': row['key'], 'repeat': repeat})
        if device == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        for group in groups:
            forward_cpu_output(wrapper, group)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        job.heartbeat({'phase': 'cost_pass_complete', 'key': row['key'], 'repeat': repeat})
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'final_allocated_bytes': torch.cuda.memory_allocated()} if device == 'cuda' else {
                  'peak_allocated_bytes': None, 'peak_reserved_bytes': None, 'final_allocated_bytes': None}
    return {'seconds_per_24_origins': times,
            'milliseconds_per_origin': describe([value * 1000 / N_ORIGINS for value in times]),
            'origins_per_second': describe([N_ORIGINS / value for value in times]),
            'resident_after_model_load_and_cleanup': resident,
            'process_cpu_memory': process_memory(), 'warmup_wrapper_calls': WARMUP,
            'timed_wrapper_calls': PASSES * len(groups), 'passes': PASSES, **memory}


def binding(prepared):
    check_seal()
    names = ('input_contract.json', 'model_manifest.json', 'preflight_checks.json',
             'evaluation_seal.json', 'runtime.py', 'baselines.py', 'c2_models.py', 'cost_controls.py')
    return {'files': {name: digest(HERE / name) for name in names},
            'inputs': {dataset: {key: value[key] for key in ('C', 'K', 'origin_sha256', 'input_sha256', 'trainval')}
                       | {'origins': value['origins'].tolist()} for dataset, value in prepared.items()},
            'precision': 'float32', 'tf32': False, 'cpu_threads': 4,
            'passes': PASSES, 'warmup_calls': WARMUP, 'blocks': BLOCKS,
            'timing_scope': 'Standardized CPU input through full online inference to all original-channel H48 contiguous CPU output',
            'excluded_from_latency': ['model_load', 'disk', 'scoring', 'hashing', 'ledger', 'telemetry'],
            'test_used_for_cost_selection': False, 'additional_fitting': 0}


def persist_grid(rows, bound, device):
    path = HERE / 'cost_binding.json'
    if path.exists() and read_json(path) != bound:
        raise ValueError('Cost sources changed; prior rows must remain intact')
    if not path.exists():
        save_json(path, bound)
    value = {'binding_sha256': digest(path), 'device': device,
             'rows': rows, 'counts': call_counts(rows)}
    grid_path = HERE / f'cost_{device}_grid.json'
    if grid_path.exists() and read_json(grid_path) != value:
        raise ValueError('The fixed cost grid changed')
    if not grid_path.exists():
        save_json(grid_path, value)
    return value['binding_sha256']


def summarize(rows, device):
    grouped = defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            grouped[(row['dataset'], row['id'], row['method'], row['seed'], row['batch'],
                     row['execution'], row['chunk_rows'])].append(row)
    instances = []
    for group in grouped.values():
        if len(group) != BLOCKS or {row['block'] for row in group} != set(range(BLOCKS)):
            raise ValueError('Every reported configuration requires all three blocks')
        first = group[0]
        instances.append({name: first[name] for name in ('dataset', 'id', 'method', 'seed', 'batch', 'execution', 'chunk_rows')}
            | {'device': device,
               'median_block_median_ms': float(np.median([row['milliseconds_per_origin']['median'] for row in group])),
               'median_block_median_origins_s': float(np.median([row['origins_per_second']['median'] for row in group])),
               'block_ms': [row['milliseconds_per_origin']['median'] for row in sorted(group, key=lambda r: r['block'])],
               'peak_allocated_bytes': [row['peak_allocated_bytes'] for row in group],
               'peak_reserved_bytes': [row['peak_reserved_bytes'] for row in group], 'model': first['model']})
    grouped = defaultdict(list)
    for row in instances:
        grouped[(row['dataset'], row['method'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
    groups = []
    for key, items in grouped.items():
        if len(items) != (1 if key[1] in ZERO_SHOT else 2):
            raise ValueError('A prescribed model or selected seed is missing from cost')
        allocated = [v for row in items for v in row['peak_allocated_bytes'] if v is not None]
        reserved = [v for row in items for v in row['peak_reserved_bytes'] if v is not None]
        groups.append({'dataset': key[0], 'method': key[1], 'batch': key[2], 'execution': key[3],
            'chunk_rows': key[4], 'device': device, 'ids': [row['id'] for row in items],
            'mean_seed_median_ms': float(np.mean([row['median_block_median_ms'] for row in items])),
            'mean_seed_median_origins_s': float(np.mean([row['median_block_median_origins_s'] for row in items])),
            'mean_seed_median_peak_allocated_bytes': float(np.mean([np.median(row['peak_allocated_bytes']) for row in items])) if allocated else None,
            'mean_seed_median_peak_reserved_bytes': float(np.mean([np.median(row['peak_reserved_bytes']) for row in items])) if reserved else None,
            'allocated_range_bytes': [min(allocated), max(allocated)] if allocated else None,
            'reserved_range_bytes': [min(reserved), max(reserved)] if reserved else None,
            'deployment': [row['model'] for row in items]})
    sentinels = []
    if device == 'cuda':
        for dataset in DATASETS:
            for block in range(BLOCKS):
                for batch in (1, 4):
                    pair = {row['sentinel']: row for row in rows if row['dataset'] == dataset
                            and row['block'] == block and row['batch'] == batch and row['sentinel'] is not None}
                    pre, post = (pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post'))
                    sentinels.append({'dataset': dataset, 'block': block, 'batch': batch,
                                      'pre_ms': pre, 'post_ms': post, 'post_relative_change': post / pre - 1})
    return {'status': 'complete', 'device': device, 'instances': instances, 'groups': groups,
            'sentinels': sentinels, 'session_ids': sorted({row['job_id'] for row in rows}),
            'aggregation': 'Median of three block medians per instance; arithmetic mean of selected seed medians',
            'uncertainty': 'All raw blocks/ranges retained; no independent-sample confidence claim'}


def run_cost(device, job):
    instances = instances_by_dataset()
    prepared = {dataset: prepare(dataset) for dataset in DATASETS}
    grid = planned_grid(instances, prepared, device)
    binding_sha = persist_grid(grid, binding(prepared), device)
    final_path = HERE / f'cost_{device}_rows.json'
    if final_path.exists():
        raise FileExistsError('This cost phase already completed; do not silently remeasure')
    folder = HERE / 'cost_rows' / device
    folder.mkdir(parents=True, exist_ok=True)
    results, wrapper, loaded_key = [], None, None
    try:
        for config in grid:
            job.check_limits()
            path = folder / (config['key'] + '.json')
            if path.exists():
                previous = read_json(path)
                if previous['binding_sha256'] != binding_sha or previous['status'] != 'complete':
                    raise ValueError('A prior row has changed source bindings or is invalid')
                if any(previous.get(key) != value for key, value in config.items()):
                    raise ValueError('An existing cost row differs from its fixed identity')
                results.append(previous)
                continue
            failures = list(folder.glob(config['key'] + '.failure_*.json'))
            if len(failures) >= 2:
                raise RuntimeError('The single technical retry for this cost condition is exhausted')
            item = next(row for row in instances[config['dataset']] if row['id'] == config['id'])
            instance_key = (config['dataset'], config['id'], config['block'], config['sentinel'])
            try:
                if instance_key != loaded_key:
                    if wrapper is not None:
                        del wrapper
                        wrapper = None
                    cleanup(device)
                    job.heartbeat({'phase': 'cost_model_load', 'instance': instance_key,
                                   'completed_rows': len(results), 'expected_rows': len(grid)})
                    wrapper, info = load_deployment(item, device)
                    loaded_key = instance_key
                job.heartbeat({'phase': 'cost_measure', **config,
                               'completed_rows': len(results), 'expected_rows': len(grid)})
                before = telemetry() if device == 'cuda' else None
                values = measure(wrapper, prepared[config['dataset']], config, device, job)
                row = {**config, **values, 'status': 'complete', 'binding_sha256': binding_sha,
                       'job_id': job.id, 'pid': os.getpid(), 'finished_utc': time.time(), 'model': info,
                       'telemetry_before': before, 'telemetry_after': telemetry() if device == 'cuda' else None}
                save_json(path, row)
                results.append(row)
                save_json(HERE / f'cost_{device}_rows_partial.json',
                          {'status': 'partial', 'rows': results, 'completed_rows': len(results),
                           'expected_rows': len(grid), 'binding_sha256': binding_sha,
                           'unfinished_keys': [row['key'] for row in grid[len(results):]]})
            except BaseException as exc:
                save_json(folder / (config['key'] + f'.failure_{job.id}.json'),
                          {'status': 'failed', 'config': config, 'binding_sha256': binding_sha,
                           'job_id': job.id, 'error': repr(exc), 'utc': time.time(), 'completed_rows': len(results)})
                raise
        output = {'status': 'complete', 'rows': results, 'expected_rows': len(grid),
                  'binding_sha256': binding_sha, 'environment': environment(),
                  'counts': call_counts(grid), 'session_ids': sorted({row['job_id'] for row in results}),
                  'scope': 'Same campaign; standardized CPU input to full online H48 CPU output; VAL only; no fitting'}
        save_json(final_path, output)
        save_json(HERE / f'cost_{device}_summary.json', summarize(results, device))
        return output
    finally:
        if wrapper is not None:
            del wrapper
        cleanup(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cuda', 'cpu'), required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    category = 'gpu' if args.device == 'cuda' else 'cpu'
    with Job(category, args.job, reserve_s=0,
             metadata={'purpose': 'same-campaign deployment cost', 'device': args.device,
                       'fit_count': 0, 'planned_rows': 279 if args.device == 'cuda' else 36,
                       'passes': PASSES, 'observer': 'Synchronous outside each timed pass'}) as job:
        job._stop.set()
        job._thread.join(timeout=12)
        if job._thread.is_alive():
            raise RuntimeError('Background observer did not stop before timing')
        configure()
        import torch
        with torch.inference_mode():
            run_cost(args.device, job)


if __name__ == '__main__':
    main()
