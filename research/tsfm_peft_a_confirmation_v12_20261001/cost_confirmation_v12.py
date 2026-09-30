"""Fixed fresh-unit cost grid: 135 GPU and 12 DIRECT CPU rows per unit, no fitting."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from runtime_confirmation_v12 import (HERE, CACHE, Job, artifact, configure, digest,
                                      environment_receipt, load_data, protocol, read_json, save_json)
from evaluate_confirmation_v12 import (build_row_model, evaluation_rows,
                                       load_complete_selection, parity)


PASSES, WARMUP, BLOCKS, N_ORIGINS = 10, 10, 3, 24
UNCOMPRESSED = ('f0', 'full_mse', 'native')


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
        return {name: int(getattr(info, name)) for name in ('rss', 'vms', 'peak_wset') if hasattr(info, name)}
    except (ImportError, OSError):
        return None


def telemetry():
    fields = 'index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate'
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=15)
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=15)
    return {'utc': time.time(), 'fields': fields, 'gpu': gpu.stdout.strip().splitlines(),
            'gpu_status': gpu.returncode, 'gpu_error': gpu.stderr.strip() or None,
            'process_gpu_memory': processes.stdout.strip().splitlines(),
            'process_memory_status': processes.returncode, 'unavailable_values': 'N/A is unavailable, never zero'}


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(dataset):
    import torch
    data = load_data(dataset, include_test=False)
    all_origins = np.asarray(data['val_origins'], dtype=np.int64)
    if len(all_origins) < N_ORIGINS or np.any(np.diff(all_origins) <= 0):
        raise ValueError('Cost requires 24 chronological VAL origins; TEST substitution is forbidden')
    origins = all_origins[:N_ORIGINS]
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]).astype(np.float32))
    return {'inputs': inputs, 'origins': origins,
            'target': np.stack([data['x'][o:o + 48] for o in origins]),
            'mask': np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool),
            'input_sha256': array_hash(inputs.numpy()), 'origin_sha256': array_hash(origins),
            'trainval': artifact(data['_path'])}


def configs(family, unit, device):
    c, k = unit['C'], unit['K']
    if family == 'direct':
        return [{'batch': batch, 'chunk_rows': None, 'execution': 'direct'} for batch in (1, 4)]
    if device != 'cuda':
        raise ValueError('Only DIRECT has a prescribed CPU deployment comparison')
    if family in UNCOMPRESSED:
        return [{'batch': 1, 'chunk_rows': c, 'execution': 'full'},
                {'batch': 1, 'chunk_rows': k, 'execution': 'chunk_k'},
                {'batch': 4, 'chunk_rows': 4 * c, 'execution': 'full'},
                {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'chunk_4k'},
                {'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]
    if family not in ('a', 'level'):
        raise ValueError('Unexpected fresh confirmation cost family')
    return [{'batch': 1, 'chunk_rows': k, 'execution': 'full'},
            {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'full'},
            {'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]


def rotated(values, block):
    if block == 1:
        return list(reversed(values))
    if block == 2:
        return list(values[1:]) + list(values[:1])
    return list(values)


def row_key(row):
    keys = ('dataset', 'id', 'family', 'seed', 'device', 'block', 'batch', 'chunk_rows', 'execution', 'sentinel')
    return hashlib.sha256(json.dumps({k: row[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:24]


def planned_grid(instances, device, p):
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(list(p['units']), block):
            items = instances[dataset]
            if device == 'cpu':
                items = [item for item in items if item['family'] == 'direct']
            ordered = []
            if device == 'cuda':
                f0 = next(item for item in items if item['family'] == 'f0')
                ordered.append((f0, 'pre'))
            ordered.extend((item, None) for item in rotated(items, block))
            if device == 'cuda':
                ordered.append((f0, 'post'))
            for item, sentinel in ordered:
                grid = configs(item['family'], p['units'][dataset], device)
                grid = [entry for entry in grid if entry['execution'] == 'full'] if sentinel else rotated(grid, block)
                for config in grid:
                    row = {'dataset': dataset, 'id': item['id'], 'family': item['family'], 'seed': item['seed'],
                           'device': device, 'block': block, 'sentinel': sentinel, **config}
                    row.update(key=row_key(row), planned_sequence=len(rows))
                    rows.append(row)
    expected = len(p['units']) * (135 if device == 'cuda' else 12)
    if not 1 <= len(p['units']) <= 2 or len(rows) != expected or len({r['key'] for r in rows}) != expected:
        raise ValueError('The approved-unit cost grid changed')
    return rows


def forward_cpu_output(model, cpu_input, device):
    import torch
    value = cpu_input.to(device=device, dtype=torch.float32)
    return model(value).float().cpu().contiguous()


def predict(model, inputs, batch, chunk_rows, device):
    import torch
    model.chunk_rows = chunk_rows
    return torch.cat([forward_cpu_output(model, group, device) for group in inputs.split(batch)], dim=0).numpy()


def macro_scores(prediction, prepared):
    error = np.where(prepared['mask'], prediction.astype(np.float64) - prepared['target'].astype(np.float64), 0.)
    count = prepared['mask'].sum((0, 1))
    if np.any(count == 0):
        raise ValueError('A fixed VAL cost channel has no observation')
    return {'mse': float(np.mean((error * error).sum((0, 1)) / count)),
            'mae': float(np.mean(np.abs(error).sum((0, 1)) / count))}


def state_bytes(model):
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(p.numel() * p.element_size() for p in model.buffers())
    return {'deployed_parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': parameters, 'buffer_bytes': buffers,
            'deployment_tensor_bytes': parameters + buffers}


def load_deployment(row, prepared, unit, device, binding_sha, job):
    from model_confirmation_v12 import deployment_copy
    cleanup(device)
    original = build_row_model(row, device=device).eval()
    family = row['family']
    stage_registered = sum(p.numel() for p in original.parameters() if p.requires_grad)
    info = {'selected_model_checkpoint': row.get('checkpoint'),
            'registered_requires_grad_at_restore': stage_registered,
            'parent_fitted_parameters': 17920 if family == 'a' else 0,
            'current_stage_registered_parameters': stage_registered,
            'sum_stage_registered_fit_parameters': stage_registered + (17920 if family == 'a' else 0),
            'interpretation': 'Stage sums are not total deployment size, function-space dimension, or speed ratios'}
    result_path = HERE / 'confirmation_runs' / row['id'] / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        info['fit_result'] = artifact(result_path)
        info['training_result'] = {key: result[key] for key in
                                  ('actual_updates', 'training_peak', 'trainable_parameters', 'elapsed_s',
                                   'changed_parameter_scalars', 'first_update_gradients') if key in result}
    base = predict(original, prepared['inputs'], 4, None, device)
    deployed = deployment_copy(original).eval()
    del original
    cleanup(device)
    exported = predict(deployed, prepared['inputs'], 4, None, device)
    check = parity(exported, base)
    if not check['pass']:
        raise ValueError('Merge/export output parity failed: ' + row['id'])
    info['merged_export_vs_original'] = check
    info.update(state_bytes(deployed))
    info['backbone_loaded'] = hasattr(deployed, 'backbone')
    folder = CACHE / 'confirmation_cost_parity'
    path = folder / (row['id'] + '.npz')
    metadata_path = folder / (row['id'] + '.json')
    if device == 'cuda':
        if path.exists() or metadata_path.exists():
            if not path.exists() or not metadata_path.exists():
                raise ValueError('Partial parity artifact needs explicit repair')
            metadata = read_json(metadata_path)
            if metadata['binding_sha256'] != binding_sha or digest(path) != metadata['npz_sha256']:
                raise ValueError('Cost parity source binding changed')
            with np.load(path, allow_pickle=False) as arrays:
                if not np.array_equal(arrays['origins'], prepared['origins']):
                    raise ValueError('Saved parity origin order changed')
                replay = parity(base, arrays['prediction'], atol=1e-6, rtol=0)
            if not replay['pass']:
                raise ValueError('Same-path selected-model replay mismatch')
        else:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, prediction=base, origins=prepared['origins'])
            save_json(metadata_path, {'binding_sha256': binding_sha, 'npz_sha256': digest(path),
                                      'id': row['id'], 'device': 'cuda'})
    else:
        metadata = read_json(metadata_path)
        if metadata['binding_sha256'] != binding_sha or digest(path) != metadata['npz_sha256']:
            raise ValueError('CPU DIRECT needs an unchanged same-campaign GPU reference')
        with np.load(path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays['origins'], prepared['origins']):
                raise ValueError('CPU/GPU origin order changed')
            check = parity(base, arrays['prediction'])
        if not check['pass']:
            raise ValueError('DIRECT CPU/GPU parity failed')
        info['cpu_gpu_parity'] = check
    checks = []
    for config in configs(family, unit, device):
        job.check_limits()
        prediction = predict(deployed, prepared['inputs'], config['batch'], config['chunk_rows'], device)
        check = parity(prediction, base)
        checks.append({**config, 'parity': check, 'val_scores': macro_scores(prediction, prepared)})
        if not check['pass']:
            raise ValueError('Full/chunk row-order/numerical parity failed: ' + row['id'])
    info['configuration_parity'] = checks
    info['reference_val_scores'] = macro_scores(base, prepared)
    info['reference_prediction_sha256'] = array_hash(base)
    deployed.chunk_rows = None
    return deployed, info


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'minimum': float(values.min()), 'maximum': float(values.max())}


def measure(model, prepared, config, device, job):
    import torch
    groups = list(prepared['inputs'].split(config['batch']))
    model.chunk_rows = config['chunk_rows']
    cleanup(device)
    resident = {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved()} if device == 'cuda' else None
    for index in range(WARMUP):
        job.check_limits()
        forward_cpu_output(model, groups[index % len(groups)], device)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(PASSES):
        job.heartbeat({'phase': 'confirmation_cost_pass', 'key': config['key'], 'repeat': repeat})
        if device == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        for group in groups:
            forward_cpu_output(model, group, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        job.heartbeat({'phase': 'confirmation_cost_pass_complete', 'key': config['key'], 'repeat': repeat})
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'final_allocated_bytes': torch.cuda.memory_allocated()} if device == 'cuda' else {
                  'peak_allocated_bytes': None, 'peak_reserved_bytes': None, 'final_allocated_bytes': None}
    return {'seconds_per_24_origins': times,
            'milliseconds_per_origin': describe([v * 1000 / N_ORIGINS for v in times]),
            'origins_per_second': describe([N_ORIGINS / v for v in times]),
            'resident_after_model_load_and_cleanup': resident, 'process_cpu_memory': process_memory(),
            'warmup_calls': WARMUP, 'passes': PASSES, **memory}


def binding(prepared):
    names = ['confirmation_protocol.json', 'selected_confirmation.json', 'parents_selected.json',
             'confirmation_data_manifest.json', 'model_confirmation_v12.py', 'runtime_confirmation_v12.py',
             'evaluate_confirmation_v12.py', 'cost_confirmation_v12.py']
    names += [f'data_contract_confirmation_{dataset}.json' for dataset in prepared]
    return {'files': {name: digest(HERE / name) for name in names},
            'inputs': {dataset: {key: value[key] for key in ('origin_sha256', 'input_sha256', 'trainval')}
                       | {'origins': value['origins'].tolist()} for dataset, value in prepared.items()},
            'precision': 'float32', 'tf32': False, 'cpu_threads': 4, 'passes': PASSES,
            'warmup_calls': WARMUP, 'blocks': BLOCKS, 'output': 'All original channels, H48, contiguous CPU',
            'timing_scope': 'Standardized CPU input through full online prediction to CPU output',
            'excluded_from_latency': ['model_load', 'disk', 'scoring', 'hashing', 'ledger', 'telemetry'],
            'test_used_for_cost_selection': False}


def persist_grid(device, rows, bound):
    path = HERE / 'confirmation_cost_binding.json'
    if path.exists():
        if read_json(path) != bound:
            raise ValueError('Cost code/data/selection changed; preserve earlier rows and review scope')
    else:
        save_json(path, bound)
    binding_sha = digest(path)
    grid_path = HERE / f'confirmation_cost_{device}_grid.json'
    value = {'binding_sha256': binding_sha, 'rows': rows, 'expected_rows': len(rows)}
    if grid_path.exists():
        if read_json(grid_path) != value:
            raise ValueError('Fixed cost grid changed')
    else:
        save_json(grid_path, value)
    return binding_sha


def summarize(rows, device, p):
    grouped = defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            grouped[(row['dataset'], row['id'], row['family'], row['seed'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
    instances = []
    for key, group in grouped.items():
        if len(group) != 3 or {row['block'] for row in group} != {0, 1, 2}:
            raise ValueError('All prescribed blocks are required; no favorable-block filtering')
        first = group[0]
        instances.append({key: first[key] for key in ('dataset', 'id', 'family', 'seed', 'batch', 'execution', 'chunk_rows')}
            | {'device': device,
               'median_block_median_ms': float(np.median([r['milliseconds_per_origin']['median'] for r in group])),
               'median_block_median_origins_s': float(np.median([r['origins_per_second']['median'] for r in group])),
               'block_ms': [r['milliseconds_per_origin']['median'] for r in sorted(group, key=lambda r: r['block'])],
               'peak_allocated_bytes': [r['peak_allocated_bytes'] for r in group],
               'peak_reserved_bytes': [r['peak_reserved_bytes'] for r in group], 'model': first['model']})
    grouped = defaultdict(list)
    for row in instances:
        grouped[(row['dataset'], row['family'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
    groups = []
    for key, items in grouped.items():
        if len(items) != (1 if key[1] == 'f0' else 2):
            raise ValueError('A paired model seed is missing from cost')
        allocated = [v for row in items for v in row['peak_allocated_bytes'] if v is not None]
        reserved = [v for row in items for v in row['peak_reserved_bytes'] if v is not None]
        groups.append({'dataset': key[0], 'family': key[1], 'batch': key[2], 'execution': key[3],
            'chunk_rows': key[4], 'device': device, 'ids': [r['id'] for r in items],
            'mean_seed_median_ms': float(np.mean([r['median_block_median_ms'] for r in items])),
            'mean_seed_median_origins_s': float(np.mean([r['median_block_median_origins_s'] for r in items])),
            'mean_seed_median_peak_allocated_bytes': float(np.mean([np.median(r['peak_allocated_bytes']) for r in items])) if allocated else None,
            'mean_seed_median_peak_reserved_bytes': float(np.mean([np.median(r['peak_reserved_bytes']) for r in items])) if reserved else None,
            'allocated_range_bytes': [min(allocated), max(allocated)] if allocated else None,
            'reserved_range_bytes': [min(reserved), max(reserved)] if reserved else None})
    sentinels = []
    if device == 'cuda':
        for dataset in p['units']:
            for block in range(BLOCKS):
                for batch in (1, 4):
                    pair = {r['sentinel']: r for r in rows if r['dataset'] == dataset and r['block'] == block
                            and r['batch'] == batch and r['sentinel'] is not None}
                    pre, post = (pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post'))
                    sentinels.append({'dataset': dataset, 'block': block, 'batch': batch,
                                      'pre_ms': pre, 'post_ms': post, 'post_relative_change': post / pre - 1})
    return {'status': 'complete', 'device': device, 'instances': instances, 'groups': groups,
            'sentinels': sentinels, 'session_ids': sorted({r['job_id'] for r in rows}),
            'aggregation': 'Median of three block medians per instance; then mean of paired seed medians; no iid uncertainty claim'}


def run_cost(device, job):
    import torch
    p, selection = protocol(), load_complete_selection()
    instances = {dataset: evaluation_rows(selection, dataset) for dataset in p['units']}
    prepared = {dataset: prepare(dataset) for dataset in p['units']}
    grid = planned_grid(instances, device, p)
    binding_sha = persist_grid(device, grid, binding(prepared))
    final_path = HERE / f'confirmation_cost_{device}_rows.json'
    if final_path.exists():
        raise FileExistsError('This cost phase already completed; do not silently remeasure')
    folder = HERE / 'confirmation_cost_rows' / device
    folder.mkdir(parents=True, exist_ok=True)
    results, model, loaded_key = [], None, None
    try:
        for config in grid:
            job.check_limits()
            path = folder / (config['key'] + '.json')
            if path.exists():
                previous = read_json(path)
                if previous['binding_sha256'] != binding_sha or previous['status'] != 'complete':
                    raise ValueError('Prior cost row is invalid or bound to another source')
                if any(previous.get(key) != value for key, value in config.items()):
                    raise ValueError('Existing cost row differs from its fixed grid identity')
                results.append(previous)
                continue
            instance = next(r for r in instances[config['dataset']] if r['id'] == config['id'])
            instance_key = (config['dataset'], config['id'], config['block'], config['sentinel'])
            try:
                if instance_key != loaded_key:
                    if model is not None:
                        del model
                        model = None
                    cleanup(device)
                    job.heartbeat({'phase': 'confirmation_cost_load_and_parity', 'instance': instance_key,
                                   'completed_rows': len(results), 'total_rows': len(grid)})
                    model, info = load_deployment(instance, prepared[config['dataset']],
                                                  p['units'][config['dataset']], device, binding_sha, job)
                    loaded_key = instance_key
                job.heartbeat({'phase': 'confirmation_cost_measure', **config,
                               'completed_rows': len(results), 'total_rows': len(grid)})
                before = telemetry() if device == 'cuda' else None
                values = measure(model, prepared[config['dataset']], config, device, job)
                row = {**config, **values, 'status': 'complete', 'binding_sha256': binding_sha,
                       'job_id': job.id, 'pid': os.getpid(), 'finished_utc': time.time(), 'model': info,
                       'telemetry_before': before, 'telemetry_after': telemetry() if device == 'cuda' else None}
                save_json(path, row)
                results.append(row)
                save_json(HERE / f'confirmation_cost_{device}_rows_partial.json',
                          {'status': 'partial', 'rows': results, 'completed_rows': len(results),
                           'expected_rows': len(grid), 'binding_sha256': binding_sha,
                           'unfinished_keys': [r['key'] for r in grid[len(results):]]})
            except BaseException as error:
                save_json(folder / (config['key'] + f'.failure_{job.id}.json'),
                          {'status': 'failed', 'config': config, 'binding_sha256': binding_sha,
                           'job_id': job.id, 'error': repr(error), 'utc': time.time(), 'completed_rows': len(results)})
                raise
        output = {'status': 'complete', 'rows': results, 'expected_rows': len(grid), 'binding_sha256': binding_sha,
                  'environment': environment_receipt(), 'session_ids': sorted({r['job_id'] for r in results}),
                  'actual_device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                  'scope': 'Same fresh confirmation campaign, full online deployment, VAL only, no fitting'}
        save_json(final_path, output)
        save_json(HERE / f'confirmation_cost_{device}_summary.json', summarize(results, device, p))
        if all((HERE / f'confirmation_cost_{kind}_summary.json').exists() for kind in ('cuda', 'cpu')):
            save_json(HERE / 'confirmation_cost_summary.json', {'status': 'complete',
                'gpu': read_json(HERE / 'confirmation_cost_cuda_summary.json'),
                'cpu': read_json(HERE / 'confirmation_cost_cpu_summary.json'),
                'binding': artifact(HERE / 'confirmation_cost_binding.json'),
                'timing_is_not_historical_ratio': True})
        return output
    finally:
        if model is not None:
            del model
        cleanup(device)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gpu', action='store_true')
    mode.add_argument('--cpu', action='store_true')
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    device, category = ('cuda', 'gpu') if args.gpu else ('cpu', 'cpu_analysis')
    expected_rows = len(protocol()['units']) * (135 if args.gpu else 12)
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'fresh confirmation deployment cost', 'device': device, 'fit_count': 0,
                       'planned_rows': expected_rows, 'passes': PASSES,
                       'observer': 'Synchronous outside each timed pass'}) as job:
        job._stop.set()
        job._thread.join(timeout=20)
        if job._thread.is_alive():
            raise RuntimeError('Cost observer did not stop before timing')
        configure()
        import torch
        with torch.inference_mode():
            run_cost(device, job)


if __name__ == '__main__':
    main()
