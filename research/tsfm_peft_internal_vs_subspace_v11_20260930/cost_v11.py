"""Same-campaign v11 deployment cost; fixed 621 GPU / 36 CPU rows.

Every model, parity check and timing run starts inside a reserved runtime Job.
No TEST arrays are read and no parameters are fitted by this module.
"""
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

from runtime_v11 import CACHE, HERE, Job, digest, environment_receipt, load_data, read_json, save_json
from gamma_v11 import DATASETS, SEEDS, configure, load_complete_selection, parity, receipt
from evaluate_v11 import build_row_model, evaluation_rows
from model_v11 import DATASETS as SHAPES, deployment_copy, state_digest

PASSES, WARMUP, BLOCKS, N_ORIGINS = 10, 10, 3, 24
UNCOMPRESSED = ('f0', 'full_lora_mse', 'lora_native')


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
            'process_memory_status': processes.returncode,
            'unavailable_values': 'N/A and missing values are unavailable, never zero'}


def tensor_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(dataset):
    import torch
    data = load_data(dataset, include_test=False)
    all_origins = np.asarray(data['val_origins'], dtype=np.int64)
    if len(all_origins) < N_ORIGINS or np.any(np.diff(all_origins) <= 0):
        raise ValueError('Cost requires at least 24 strictly chronological VAL origins')
    origins = all_origins[:N_ORIGINS]
    inputs = torch.from_numpy(np.stack([data['x'][origin - 512:origin] for origin in origins]).astype(np.float32))
    target = np.stack([data['x'][origin:origin + 48] for origin in origins])
    mask = np.stack([data['finite'][origin:origin + 48] for origin in origins]).astype(bool)
    return {'inputs': inputs, 'origins': origins, 'target': target, 'mask': mask, 'basis': data['basis'],
            'input_sha256': tensor_digest(inputs.numpy()), 'origin_sha256': tensor_digest(origins)}


def configs(family, dataset, device):
    channels, latent = SHAPES[dataset]
    if family == 'direct_nlinear':
        return [{'batch': batch, 'chunk_rows': None, 'execution': 'direct'} for batch in (1, 4)]
    if device != 'cuda':
        raise ValueError('Only DIRECT has the approved CPU deployment comparison')
    if family in UNCOMPRESSED:
        return [{'batch': 1, 'chunk_rows': channels, 'execution': 'full'},
                {'batch': 1, 'chunk_rows': latent, 'execution': 'chunk_k'},
                {'batch': 4, 'chunk_rows': 4 * channels, 'execution': 'full'},
                {'batch': 4, 'chunk_rows': 4 * latent, 'execution': 'chunk_4k'},
                {'batch': 4, 'chunk_rows': latent, 'execution': 'chunk_k'}]
    return [{'batch': 1, 'chunk_rows': latent, 'execution': 'full'},
            {'batch': 4, 'chunk_rows': 4 * latent, 'execution': 'full'},
            {'batch': 4, 'chunk_rows': latent, 'execution': 'chunk_k'}]


def rotated(values, block):
    if block == 1:
        return list(reversed(values))
    if block == 2:
        return list(values[1:]) + list(values[:1])
    return list(values)


def row_key(row):
    selected = {name: row[name] for name in ('dataset', 'id', 'family', 'seed', 'device', 'block',
                                            'batch', 'chunk_rows', 'execution', 'sentinel')}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:24]


def planned_grid(instances, device):
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(list(DATASETS), block):
            items = instances[dataset]
            if device == 'cpu':
                items = [item for item in items if item['family'] == 'direct_nlinear']
            f0 = next((item for item in items if item['family'] == 'f0'), None)
            ordered = []
            if device == 'cuda':
                ordered.append((f0, 'pre'))
            ordered.extend((item, None) for item in rotated(items, block))
            if device == 'cuda':
                ordered.append((f0, 'post'))
            for item, sentinel in ordered:
                grid = configs(item['family'], dataset, device)
                grid = [entry for entry in grid if entry['execution'] == 'full'] if sentinel else rotated(grid, block)
                for config in grid:
                    row = {'dataset': dataset, 'id': item['id'], 'family': item['family'], 'seed': item['seed'],
                           'device': device, 'block': block, 'sentinel': sentinel, **config}
                    row['key'] = row_key(row)
                    row['planned_sequence'] = len(rows)
                    rows.append(row)
    expected = 621 if device == 'cuda' else 36
    if len(rows) != expected or len({row['key'] for row in rows}) != expected:
        raise ValueError(f'Fixed cost grid differs: {len(rows)} instead of {expected}')
    return rows


def forward_cpu_output(model, cpu_input, device):
    import torch
    value = cpu_input.to(device=device, dtype=torch.float32)
    output = model(value)
    return output.float().cpu().contiguous()


def predict(model, inputs, batch, chunk_rows, device):
    import torch
    model.chunk_rows = chunk_rows
    return torch.cat([forward_cpu_output(model, group, device) for group in inputs.split(batch)], dim=0).numpy()


def macro_scores(prediction, prepared):
    error = np.where(prepared['mask'], prediction.astype(np.float64) - prepared['target'].astype(np.float64), 0.0)
    count = prepared['mask'].sum((0, 1))
    if np.any(count == 0):
        raise ValueError('The fixed cost VAL subset has a channel without observations')
    return {'mse': float(np.mean((error * error).sum((0, 1)) / count)),
            'mae': float(np.mean(np.abs(error).sum((0, 1)) / count))}


def state_bytes(model):
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {'deployed_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'parameter_bytes': parameter_bytes, 'buffer_bytes': buffer_bytes,
            'deployment_tensor_bytes': parameter_bytes + buffer_bytes}


def adaptation_parameters(row, original):
    family = row['family']
    channels, latent = SHAPES[row['dataset']]
    current = {'a_p_lora_qfixed': 294912, 'full_lora_mse': 294912, 'b_learned_u': channels * latent,
               'b_learned_gamma': channels * latent, 'direct_nlinear': 24624}.get(family, 0)
    parents = 17920 if family in ('a_p_lora_qfixed', 'base_level') else 24624 if family.startswith('b_') else 0
    gamma = latent if family in ('b_fixed_gamma', 'b_learned_gamma') else 0
    native = 294912 if family == 'lora_native' else 0
    return {'registered_requires_grad_at_restore': sum(parameter.numel() for parameter in original.parameters() if parameter.requires_grad),
            'current_neural_stage_registered_parameters': current,
            'reused_parent_fitted_parameters': parents + native, 'gamma_coefficients': gamma,
            'sum_stage_registered_fit_parameters': current + parents + native + gamma,
            'interpretation': 'Stage sum is not a function-space dimension, total model size, or controlled training-speed comparison'}


def parity_reference_path(row):
    return CACHE / 'cost_parity' / (row['id'] + '.npz')


def load_deployment(row, selection, prepared, device, binding_sha, job):
    import torch
    cleanup(device)
    original = build_row_model(row, selection, prepared['basis'], device=device).eval()
    info = adaptation_parameters(row, original)
    checkpoint = row.get('checkpoint')
    if row['family'] == 'b_learned_gamma':
        selected = selection['selected'][row['dataset']]['b_learned_u']
        index = SEEDS.index(row['seed'])
        checkpoint = {'path': selected['checkpoints'][index], 'sha256': selected['checkpoint_sha256'][index],
                      'run_id': selected['run_ids'][index]}
    info['selected_model_checkpoint'] = checkpoint
    info['actual_updates_and_gradients'] = None
    if checkpoint and checkpoint.get('run_id'):
        result_path = HERE / 'runs' / checkpoint['run_id'] / 'result.json'
        if result_path.exists():
            result = read_json(result_path)
            gradients = result.get('first_update_gradients', {})
            info['actual_updates_and_gradients'] = {
                'result': receipt(result_path), 'actual_optimizer_updates': result.get('actual_updates'),
                'changed_parameter_scalars': sum(result.get('changed_parameter_scalars', {}).values()),
                'gradient_nonzero_scalars_by_recorded_step': {
                    step: sum(item['nonzero'] for item in values.values()) for step, values in gradients.items()},
                'scope': 'Actual recorded early gradients and final-vs-initial parameter changes; not all-step gradient activity'}
    info['source_receipts'] = original.source_receipts
    info['adapter_state_sha256'] = state_digest(original.adapter_state())
    base = predict(original, prepared['inputs'], 4, None, device)
    deployed = deployment_copy(original, prune_zero_gamma=True).eval()
    del original
    cleanup(device)
    check = parity(predict(deployed, prepared['inputs'], 4, None, device), base)
    info['merged_export_pruned_vs_original'] = check
    info.update(state_bytes(deployed))
    info['backbone_loaded'] = hasattr(deployed, 'backbone')
    if row['family'].startswith('b_'):
        info['gamma'] = deployed.gamma.detach().cpu().tolist()
        info['active_k'] = int((deployed.gamma != 0).sum())
        info['exact_zero_indices'] = np.flatnonzero(np.asarray(info['gamma']) == 0).tolist()
    if not check['pass']:
        raise ValueError(f'Merge/export/prune parity failed for {row["id"]}: {check}')
    reference_path = parity_reference_path(row)
    metadata_path = reference_path.with_suffix('.json')
    if device == 'cuda':
        if reference_path.exists() or metadata_path.exists():
            if not reference_path.exists() or not metadata_path.exists():
                raise ValueError('Incomplete saved cost parity reference; preserve and repair explicitly')
            metadata = read_json(metadata_path)
            if metadata['binding_sha256'] != binding_sha or metadata['npz_sha256'] != digest(reference_path):
                raise ValueError('Cost parity reference belongs to changed sources')
            with np.load(reference_path, allow_pickle=False) as arrays:
                replay = parity(base, arrays['prediction'], atol=1e-6, rtol=0)
            if not replay['pass']:
                raise ValueError(f'Identical selected model replay differs: {row["id"]}')
        else:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(reference_path, prediction=base, origins=prepared['origins'])
            save_json(metadata_path, {'binding_sha256': binding_sha, 'npz_sha256': digest(reference_path),
                      'id': row['id'], 'origin_sha256': prepared['origin_sha256'], 'device': 'cuda'})
    else:
        if not reference_path.exists() or not metadata_path.exists():
            raise ValueError('CPU DIRECT cost requires the same-campaign GPU parity reference first')
        metadata = read_json(metadata_path)
        if metadata['binding_sha256'] != binding_sha or metadata['npz_sha256'] != digest(reference_path):
            raise ValueError('GPU parity reference changed before CPU comparison')
        with np.load(reference_path, allow_pickle=False) as arrays:
            cpu_gpu = parity(base, arrays['prediction'])
        info['cpu_gpu_parity'] = cpu_gpu
        if not cpu_gpu['pass']:
            raise ValueError('DIRECT CPU/GPU parity failed at the fixed tolerance')
    checks = []
    for config in configs(row['family'], row['dataset'], device):
        job.check_limits()
        prediction = predict(deployed, prepared['inputs'], config['batch'], config['chunk_rows'], device)
        check = parity(prediction, base)
        record = {**config, 'parity': check, 'val_scores': macro_scores(prediction, prepared)}
        checks.append(record)
        if not check['pass']:
            raise ValueError(f'Full/chunk parity failed for {row["id"]}: {record}')
    info['configuration_parity'] = checks
    info['reference_val_scores'] = macro_scores(base, prepared)
    info['reference_prediction_sha256'] = tensor_digest(base)
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
    resident = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()} if device == 'cuda' else None
    for index in range(WARMUP):
        job.check_limits()
        forward_cpu_output(model, groups[index % len(groups)], device)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(PASSES):
        job.heartbeat({'phase': 'cost_pass', 'key': config['key'], 'repeat': repeat})
        if device == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        for group in groups:
            forward_cpu_output(model, group, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        job.heartbeat({'phase': 'cost_pass_complete', 'key': config['key'], 'repeat': repeat})
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'final_allocated_bytes': torch.cuda.memory_allocated()} if device == 'cuda' else {
                  'peak_allocated_bytes': None, 'peak_reserved_bytes': None, 'final_allocated_bytes': None}
    return {'seconds_per_24_origins': times,
            'milliseconds_per_origin': describe([value * 1000 / N_ORIGINS for value in times]),
            'origins_per_second': describe([N_ORIGINS / value for value in times]),
            'resident_after_model_load_and_cleanup': resident, 'process_cpu_memory': process_memory(),
            'warmup_calls': WARMUP, 'passes': PASSES, **memory}


def binding(selection, gamma, prepared):
    paths = [HERE / name for name in ('PLAN.md', 'protocol.json', 'selected.json', 'gamma_selection.json',
                                      'reuse_manifest.json', 'model_v11.py', 'cost_v11.py', 'runtime_v11.py',
                                      'evaluate_v11.py', 'gamma_v11.py')]
    paths.extend(HERE / f'data_contract_{dataset}.json' for dataset in DATASETS)
    return {'files': {str(path): digest(path) for path in paths},
            'inputs': {dataset: {'origins': item['origins'].tolist(), 'origin_sha256': item['origin_sha256'],
                                'input_sha256': item['input_sha256']} for dataset, item in prepared.items()},
            'precision': 'float32', 'tf32': False, 'cpu_threads': 4, 'passes': PASSES,
            'warmup_calls': WARMUP, 'blocks': BLOCKS, 'output': 'all original channels, H48, CPU',
            'timing_scope': 'standardized CPU input through full online prediction to contiguous CPU output',
            'excluded_from_latency': ['model_load', 'disk', 'scoring', 'hashing', 'ledger', 'telemetry'],
            'selection_complete': True, 'TEST_used_for_cost_selection': False}


def persist_grid(device, rows, binding_value):
    path = HERE / 'cost_binding.json'
    if path.exists():
        if read_json(path) != binding_value:
            raise ValueError('Cost code/data/selection changed; preserve prior rows and explicitly review affected scope')
    else:
        save_json(path, binding_value)
    binding_sha = digest(path)
    grid_path = HERE / f'cost_{device}_grid.json'
    grid = {'binding_sha256': binding_sha, 'rows': rows, 'expected_rows': len(rows)}
    if grid_path.exists():
        if read_json(grid_path) != grid:
            raise ValueError('Fixed cost grid changed')
    else:
        save_json(grid_path, grid)
    return binding_sha


def summarize(rows, device):
    groups = defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            groups[(row['dataset'], row['id'], row['family'], row['seed'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
    per_instance = []
    for key, group in groups.items():
        if len(group) != BLOCKS or {item['block'] for item in group} != set(range(BLOCKS)):
            raise ValueError('Complete summaries require every prescribed block')
        first = group[0]
        per_instance.append({name: first[name] for name in ('dataset', 'id', 'family', 'seed', 'batch', 'execution', 'chunk_rows')} |
                            {'device': device, 'median_block_median_ms': float(np.median([item['milliseconds_per_origin']['median'] for item in group])),
                             'median_block_median_origins_s': float(np.median([item['origins_per_second']['median'] for item in group])),
                             'block_ms': [item['milliseconds_per_origin']['median'] for item in sorted(group, key=lambda item: item['block'])],
                             'peak_allocated_bytes': [item['peak_allocated_bytes'] for item in group],
                             'peak_reserved_bytes': [item['peak_reserved_bytes'] for item in group], 'model': first['model']})
    grouped = defaultdict(list)
    for item in per_instance:
        grouped[(item['dataset'], item['family'], item['batch'], item['execution'], item['chunk_rows'])].append(item)
    centers = []
    for key, items in grouped.items():
        expected = 1 if key[1] == 'f0' else 2
        if len(items) != expected:
            raise ValueError('Summary cannot drop an unfavorable seed')
        allocated = [value for item in items for value in item['peak_allocated_bytes'] if value is not None]
        reserved = [value for item in items for value in item['peak_reserved_bytes'] if value is not None]
        centers.append({'dataset': key[0], 'family': key[1], 'batch': key[2], 'execution': key[3],
                        'chunk_rows': key[4], 'device': device, 'ids': [item['id'] for item in items],
                        'mean_seed_median_ms': float(np.mean([item['median_block_median_ms'] for item in items])),
                        'mean_seed_median_origins_s': float(np.mean([item['median_block_median_origins_s'] for item in items])),
                        'mean_seed_median_peak_allocated_bytes': float(np.mean([np.median(item['peak_allocated_bytes']) for item in items])) if allocated else None,
                        'mean_seed_median_peak_reserved_bytes': float(np.mean([np.median(item['peak_reserved_bytes']) for item in items])) if reserved else None,
                        'allocated_range_bytes': [min(allocated), max(allocated)] if allocated else None,
                        'reserved_range_bytes': [min(reserved), max(reserved)] if reserved else None})
    sentinels = []
    if device == 'cuda':
        for dataset in DATASETS:
            for block in range(BLOCKS):
                for batch in (1, 4):
                    pair = {item['sentinel']: item for item in rows if item['dataset'] == dataset and item['block'] == block
                            and item['batch'] == batch and item['sentinel'] is not None}
                    pre = pair['pre']['milliseconds_per_origin']['median']
                    post = pair['post']['milliseconds_per_origin']['median']
                    sentinels.append({'dataset': dataset, 'block': block, 'batch': batch, 'pre_ms': pre,
                                      'post_ms': post, 'post_relative_change': post / pre - 1})
    return {'status': 'complete', 'device': device, 'instances': per_instance, 'groups': centers,
            'sentinels': sentinels, 'session_ids': sorted({item['job_id'] for item in rows}),
            'aggregation': 'median of three block medians per instance, then arithmetic mean of paired seed medians; no iid uncertainty claim'}


def run_cost(device, job):
    import torch
    selection = load_complete_selection()
    gamma = read_json(HERE / 'gamma_selection.json')
    if gamma['status'] != 'complete' or len(gamma['rows']) != 12:
        raise ValueError('All twelve Gamma fits must precede cost comparison')
    reuse = read_json(HERE / 'reuse_manifest.json')
    instances = {dataset: evaluation_rows(selection, reuse, gamma, dataset) for dataset in DATASETS}
    prepared = {dataset: prepare(dataset) for dataset in DATASETS}
    grid = planned_grid(instances, device)
    bound = binding(selection, gamma, prepared)
    binding_sha = persist_grid(device, grid, bound)
    final_path = HERE / f'cost_{device}_rows.json'
    if final_path.exists():
        raise FileExistsError('This cost phase is already complete; do not remeasure it silently')
    folder = HERE / 'cost_rows' / device
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    model = None
    loaded_key = None
    try:
        for config in grid:
            job.check_limits()
            path = folder / (config['key'] + '.json')
            if path.exists():
                previous = read_json(path)
                if previous['binding_sha256'] != binding_sha or previous['status'] != 'complete':
                    raise ValueError('Partial cost row changed or did not complete')
                if any(previous.get(name) != value for name, value in config.items()):
                    raise ValueError('Completed row does not match its prescribed grid identity')
                results.append(previous)
                continue
            instance = next(item for item in instances[config['dataset']] if item['id'] == config['id'])
            instance_key = (config['dataset'], config['id'], config['block'], config['sentinel'])
            try:
                if loaded_key != instance_key:
                    if model is not None:
                        del model
                        model = None
                    cleanup(device)
                    job.heartbeat({'phase': 'cost_load_and_parity', 'device': device, 'instance': instance_key,
                                   'completed_rows': len(results), 'total_rows': len(grid)})
                    model, model_info = load_deployment(instance, selection, prepared[config['dataset']], device, binding_sha, job)
                    loaded_key = instance_key
                job.heartbeat({'phase': 'cost_measure', **config, 'completed_rows': len(results), 'total_rows': len(grid)})
                before = telemetry() if device == 'cuda' else None
                measured = measure(model, prepared[config['dataset']], config, device, job)
                row = {**config, **measured, 'status': 'complete', 'binding_sha256': binding_sha,
                       'job_id': job.id, 'pid': os.getpid(), 'finished_utc': time.time(), 'model': model_info,
                       'telemetry_before': before, 'telemetry_after': telemetry() if device == 'cuda' else None}
                save_json(path, row)
                results.append(row)
                save_json(HERE / f'cost_{device}_rows_partial.json', {'status': 'partial', 'rows': results,
                          'completed_rows': len(results), 'expected_rows': len(grid), 'binding_sha256': binding_sha,
                          'unfinished_keys': [entry['key'] for entry in grid[len(results):]]})
            except BaseException as error:
                save_json(folder / (config['key'] + f'.failure_{job.id}.json'),
                          {'status': 'failed', 'config': config, 'binding_sha256': binding_sha,
                           'job_id': job.id, 'error': repr(error), 'utc': time.time(), 'completed_rows': len(results)})
                raise
        if len(results) != len(grid):
            raise ValueError('Cost grid is incomplete')
        output = {'status': 'complete', 'rows': results, 'expected_rows': len(grid), 'binding_sha256': binding_sha,
                  'environment': environment_receipt(), 'session_ids': sorted({row['job_id'] for row in results}),
                  'actual_device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                  'scope': 'same v11 campaign, full online deployment, VAL only, no fitting'}
        save_json(final_path, output)
        save_json(HERE / f'cost_{device}_summary.json', summarize(results, device))
        if all((HERE / f'cost_{kind}_summary.json').exists() for kind in ('cuda', 'cpu')):
            save_json(HERE / 'cost_summary.json', {'status': 'complete',
                      'gpu': read_json(HERE / 'cost_cuda_summary.json'), 'cpu': read_json(HERE / 'cost_cpu_summary.json'),
                      'binding': receipt(HERE / 'cost_binding.json'), 'timing_is_not_historical_ratio': True})
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
    parser.add_argument('--reserve-seconds', type=float, default=None)
    args = parser.parse_args()
    device = 'cuda' if args.gpu else 'cpu'
    reserve = args.reserve_seconds if args.reserve_seconds is not None else 7200 if args.gpu else 300
    category = 'gpu' if args.gpu else 'cpu_analysis'
    with Job(args.job, category=category, reserve_seconds=reserve,
             metadata={'phase': 'deployment_cost', 'device': device, 'fit_count': 0,
                       'planned_rows': 621 if args.gpu else 36, 'passes': PASSES,
                       'observer': 'synchronous outside every timed pass'}) as job:
        job._stop.set()
        job._thread.join(timeout=20)
        if job._thread.is_alive():
            raise RuntimeError('Cost observer did not stop before timing')
        job.check_limits()
        import torch
        configure()
        with torch.inference_mode():
            run_cost(device, job)


if __name__ == '__main__':
    main()
