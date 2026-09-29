"""Conditional Robin-only v8 deployment costs; root reserves and launches every run."""
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime_v8 import (HERE, ROOT, V6, Job, digest, load_data, save_json,
                        score_arrays, source_receipt, environment_receipt)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checked_json(path, expected=None):
    path = resolve(path)
    if expected is not None and digest(path) != expected:
        raise ValueError('Source hash changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def decision_guard():
    path = HERE / 'cost_decision.json'
    decision = checked_json(path)
    retained = decision.get('retained_families', [])
    if decision.get('run_cost') is not True or not retained or not set(retained) <= {'pq_split', 'raw64'}:
        raise RuntimeError('Cost requires an explicit retained-value decision for a v8 modification')
    if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise ValueError('Cost decision must record the observed reason')
    for dataset in ('robin', 'jena'):
        receipt = decision['evaluations'][dataset]
        evaluation_path = resolve(receipt['path']).resolve()
        if evaluation_path.parent != HERE.resolve():
            raise ValueError('The condition requires both new v8 accuracy evaluations')
        evaluation = checked_json(evaluation_path, receipt['sha256'])
        if evaluation.get('status') != 'complete' or evaluation.get('dataset') != dataset:
            raise ValueError('Both fixed accuracy comparisons must be complete before costs')
    contract = checked_json(HERE / 'protocol.json')['cost']
    expected = {'dataset': 'robin', 'val_origins': 24, 'batches': [1, 4],
                'warmup_calls': 10, 'passes': 20, 'blocks': 3, 'primary_gpu_rows': 63,
                'sentinel_gpu_rows': 12, 'total_gpu_rows': 75, 'cpu_rows': 0,
                'atol': 1e-5, 'rtol': 1e-4, 'profile_sessions': 0}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Cost grid differs from the approved v8 protocol')
    if set(contract['methods']) != {'pq_split', 'raw64', 'level_res', 'f0', 'lora'}:
        raise ValueError('The cost decision cannot drop comparison methods')
    return decision


def module(path):
    name = '_cost_v8_source_' + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


def candidates():
    contract = checked_json(HERE / 'data_contract_robin.json')
    old = checked_json(contract['source']['v6_selection'], contract['source']['v6_selection_sha256'])
    new = checked_json(HERE / 'selected_robin.json')
    result = []
    for family in ('f0', 'lora', 'level_res', 'pq_split', 'raw64'):
        selection = old if family in ('f0', 'lora', 'level_res') else new
        if family == 'f0':
            records = [selection['deterministic_models']['f0']]
        else:
            ids = selection['selected'][family]['run_ids']
            records = [r for r in selection['all_neural_runs'] if r['id'] in ids]
            if len(ids) != 2 or {r['spec']['seed'] for r in records} != {92601, 92602}:
                raise ValueError('Every neural cost family requires its two selected seeds')
            records.sort(key=lambda r: r['spec']['seed'])
        for record in records:
            if record['spec']['family'] != family or record['spec']['dataset'] != 'robin':
                raise ValueError('Candidate identity differs from the Robin selection')
            checked_json(record['result_path'], record['result_sha256'])
            checkpoint = resolve(record['checkpoint'])
            if digest(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            prediction = record['best_val_prediction']
            if digest(resolve(prediction['path'])) != prediction['sha256']:
                raise ValueError('Stored selected VAL output changed')
            result.append(dict(id=record['id'], family=family, seed=record['spec']['seed'],
                               checkpoint=checkpoint, checkpoint_sha256=record['checkpoint_sha256'],
                               val_prediction=prediction, best_val_mse=record['best_val_mse'],
                               result_path=resolve(record['result_path']),
                               source=V6 / 'model_v6.py' if selection is old else HERE / 'model_v8.py'))
    if len(result) != 9:
        raise ValueError('The approved cost grid requires nine model instances')
    return result


def cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def telemetry():
    fields = 'index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate'
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                         capture_output=True, text=True)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                          capture_output=True, text=True)
    return dict(utc=time.time(), fields=fields, gpu=gpu.stdout.strip().splitlines(),
                process_gpu_memory=apps.stdout.strip().splitlines(),
                gpu_returncode=gpu.returncode, process_returncode=apps.returncode,
                meaning='Unavailable/N/A telemetry is not zero; no device settings are changed')


def cost_environment():
    result = environment_receipt()
    result.update(device=torch.cuda.get_device_name(), device_index=torch.cuda.current_device(),
                  dtype='float32', cpu_threads=torch.get_num_threads(),
                  cpu_interop_threads=torch.get_num_interop_threads(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  inference_mode=torch.is_inference_mode_enabled())
    return result


def configs(family):
    grid = [(1, 'original', None), (4, 'original', None)]
    if family in ('f0', 'lora'):
        grid.append((4, 'generic', 20))
    return grid


def point(model, x, micro):
    batch, length, channels = x.shape
    flat = x.transpose(1, 2).reshape(batch * channels, length)
    output = torch.empty((batch * channels, 48), device=x.device, dtype=x.dtype)
    for start in range(0, len(flat), micro):
        raw = model.backbone(context=flat[start:start + micro])
        output[start:start + micro].copy_(raw.quantile_preds[:, model.median, :48])
        del raw
    return output.reshape(batch, channels, 48).transpose(1, 2)


def forward(model, x, variant='original', micro=None):
    x = x.to(device='cuda', dtype=torch.float32)
    if variant == 'original' and micro is None:
        output = model(x)
    elif variant == 'generic' and micro == 20 and model.family in ('f0', 'lora') and len(x) == 4:
        output = point(model, x, micro)
    else:
        raise ValueError('Only the original paths and uncompressed B4 M20 are approved')
    return output.float().cpu()


def predict(model, inputs, batch, variant='original', micro=None, job=None):
    outputs = []
    for x in inputs.split(batch):
        if job is not None:
            job.check_limits()
        outputs.append(forward(model, x, variant, micro))
    return torch.cat(outputs)


def difference(actual, reference, atol=1e-5, rtol=1e-4):
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite or incompatible parity output')
    delta = (actual - reference).abs()
    violations = delta > atol + rtol * reference.abs()
    return dict(pass_=not bool(violations.any()), atol=atol, rtol=rtol,
                max_absolute_difference=float(delta.max()),
                max_relative_difference=float((delta / reference.abs().clamp_min(1e-12)).max()),
                relative_denominator_floor=1e-12, violating_elements=int(violations.sum()),
                elements=actual.numel(), bitwise_equal=bool(torch.equal(actual, reference)))


def load_candidate(item):
    if digest(item['checkpoint']) != item['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after candidate collection')
    model = module(item['source']).restore_model(item['checkpoint'], device='cuda').eval()
    if model.family != item['family']:
        raise ValueError('Restored model family changed')
    info = dict(trainable_parameters_before_merge=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters_before_merge=sum(p.numel() for p in model.parameters()),
                checkpoint_path=str(item['checkpoint']), checkpoint_sha256=item['checkpoint_sha256'],
                model_source=str(item['source']), model_source_sha256=digest(item['source']))
    expected = {'f0': 0, 'lora': 294912, 'level_res': 17920, 'pq_split': 35840, 'raw64': 35840}
    if info['trainable_parameters_before_merge'] != expected[item['family']]:
        raise ValueError('Unexpected trainable count for the cost candidate')
    return model, info


def merge(model, inputs, job):
    before = predict(model, inputs, 4, job=job)
    model.backbone = model.backbone.merge_and_unload(safe_merge=True)
    after = predict(model, inputs, 4, job=job)
    check = difference(after, before)
    if not check['pass_']:
        raise ValueError('Returned LoRA merged model failed common-VAL parity: ' + str(check))
    return check


def model_bytes(model):
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(p.numel() * p.element_size() for p in model.buffers())
    return dict(deployed_parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes,
                deployment_tensor_bytes=parameter_bytes + buffer_bytes)


def prepare():
    data = load_data('robin', include_test=False)
    origins = np.asarray(data['val_origins'])
    if len(origins) != 30:
        raise ValueError('Robin VAL origin count changed')
    indices = np.linspace(0, 29, 24, dtype=int)
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]))
    return dict(inputs=inputs, origins=origins, selected_inputs=inputs[indices], selected_origins=origins[indices],
                target=np.stack([data['x'][o:o + 48] for o in origins]),
                mask=np.stack([data['finite'][o:o + 48] for o in origins]), data_path=data['_path'])


def binding(items, prepared, decision):
    paths = [HERE / 'cost_v8.py', HERE / 'runtime_v8.py', HERE / 'protocol.json',
             HERE / 'cost_decision.json', HERE / 'selected_robin.json', HERE / 'selected_jena.json',
             HERE / 'data_contract_robin.json', prepared['data_path'], V6 / 'selected.json',
             ROOT / 'src/hier_peft/lora.py']
    paths.extend(resolve(decision['evaluations'][d]['path']) for d in ('robin', 'jena'))
    for item in items:
        paths.extend((item['checkpoint'], item['source'], item['result_path'], resolve(item['val_prediction']['path'])))
    return {str(path): digest(path) for path in paths}


def parity_all(items, prepared, decision, job):
    initial_binding = binding(items, prepared, decision)
    rows = []
    receipts = []
    inputs = prepared['selected_inputs']
    for item in items:
        job.heartbeat('VAL restoration ' + item['id'])
        cleanup()
        model, info = load_candidate(item)
        replay = predict(model, prepared['inputs'], 4, job=job)
        with np.load(resolve(item['val_prediction']['path']), allow_pickle=False) as saved:
            if not np.array_equal(saved['origins'], prepared['origins']):
                raise ValueError('Stored VAL origin order changed')
            reference = torch.from_numpy(saved['prediction'].copy())
        replay_check = difference(replay, reference, atol=1e-6, rtol=0)
        replay_score = score_arrays(replay.numpy(), prepared['target'], prepared['mask'])
        if not replay_check['pass_'] or abs(replay_score['mse'] - item['best_val_mse']) > 1e-8:
            raise ValueError('Selected checkpoint did not reproduce its stored full VAL output')
        receipts.append(dict(id=item['id'], full_val=replay_check, mse=replay_score['mse'],
                             mse_difference=replay_score['mse'] - item['best_val_mse'], model=info))
        del replay, reference
        if item['family'] == 'lora':
            info['merge'] = merge(model, inputs, job)
        reference = predict(model, inputs, 4, job=job)
        for batch, variant, micro in configs(item['family']):
            job.heartbeat(f'parity {item["id"]} B{batch} {variant} M{micro}')
            actual = predict(model, inputs, batch, variant, micro, job)
            check = difference(actual, reference)
            rows.append(dict(id=item['id'], family=item['family'], seed=item['seed'], batch=batch,
                             variant=variant, micro=micro, val=check, model=info))
            save_json(HERE / 'cost_parity_partial.json', dict(rows=rows, restorations=receipts, source=source_receipt()))
            if not check['pass_']:
                raise ValueError('B1/B4/M20 parity failed; no tolerance relaxation')
            del actual
        del model, reference
        cleanup()
    if len(rows) != 21:
        raise ValueError('The parity grid must cover all 21 configurations')
    if binding(items, prepared, decision) != initial_binding:
        raise RuntimeError('Parity source/state/data changed while models were loaded')
    result = dict(status='PASS', rows=rows, restorations=receipts, val_origins=prepared['selected_origins'].tolist(),
                  full_val_origins=prepared['origins'].tolist(), binding=initial_binding,
                  scope='VAL only; restored full VAL and same-model B1/B4/M20 equivalence; no new TEST prediction',
                  source=source_receipt(), environment=cost_environment())
    save_json(HERE / 'cost_parity.json', result)


def describe(values):
    return dict(median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), min=float(min(values)), max=float(max(values)))


def measure(model, inputs, batch, variant, micro, job):
    chunks = list(inputs.split(batch))
    cleanup()
    resident = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
    for index in range(10):
        job.check_limits()
        forward(model, chunks[index % len(chunks)], variant, micro)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(20):
        job.check_limits()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for chunk in chunks:
            forward(model, chunk, variant, micro)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    memory = psutil.Process().memory_info()
    return dict(seconds_per_24_origins=times,
                milliseconds_per_origin=describe([t * 1000 / 24 for t in times]),
                origins_per_second=describe([24 / t for t in times]), loaded_resident=resident,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                final_allocated_bytes=torch.cuda.memory_allocated(),
                process_cpu_memory={k: int(getattr(memory, k)) for k in ('rss', 'vms', 'peak_wset') if hasattr(memory, k)})


def summarize(rows):
    by_fit = defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            by_fit[(row['id'], row['batch'], row['variant'], row['micro'])].append(row)
    fits = []
    for key, values in by_fit.items():
        if len(values) != 3 or {r['block'] for r in values} != {0, 1, 2}:
            raise ValueError('Every cost configuration requires all three blocks')
        first = values[0]
        fits.append(dict(id=key[0], batch=key[1], variant=key[2], micro=key[3], device='cuda',
                         family=first['family'], seed=first['seed'],
                         milliseconds_per_origin=describe([r['milliseconds_per_origin']['median'] for r in values]),
                         origins_per_second=describe([r['origins_per_second']['median'] for r in values]),
                         peak_allocated_bytes=[r['peak_allocated_bytes'] for r in values],
                         peak_reserved_bytes=[r['peak_reserved_bytes'] for r in values], model=first['model']))
    by_family = defaultdict(list)
    for row in fits:
        by_family[(row['family'], row['batch'], row['variant'], row['micro'])].append(row)
    groups = []
    for key, values in by_family.items():
        expected = 1 if key[0] == 'f0' else 2
        if len(values) != expected:
            raise ValueError('A seed is missing from cost aggregation')
        allocated = [v for r in values for v in r['peak_allocated_bytes']]
        reserved = [v for r in values for v in r['peak_reserved_bytes']]
        groups.append(dict(family=key[0], batch=key[1], variant=key[2], micro=key[3], device='cuda',
                           mean_seed_median_ms=float(np.mean([r['milliseconds_per_origin']['median'] for r in values])),
                           mean_seed_median_throughput=float(np.mean([r['origins_per_second']['median'] for r in values])),
                           mean_seed_median_peak_allocated_bytes=float(np.mean([np.median(r['peak_allocated_bytes']) for r in values])),
                           observed_peak_allocated_range_bytes=[int(min(allocated)), int(max(allocated))],
                           observed_peak_reserved_range_bytes=[int(min(reserved)), int(max(reserved))],
                           ids=[r['id'] for r in values]))
    drift = []
    for block in range(3):
        for batch in (1, 4):
            pair = {r['sentinel']: r for r in rows if r['block'] == block and r['batch'] == batch and r['sentinel']}
            if set(pair) != {'pre', 'post'}:
                raise ValueError('Missing F0 drift sentinel')
            pre, post = (pair[k]['milliseconds_per_origin']['median'] for k in ('pre', 'post'))
            drift.append(dict(block=block, batch=batch, pre_ms=pre, post_ms=post, post_relative_change=post / pre - 1))
    return dict(status='complete', fits=fits, groups=groups, sentinels=drift,
                aggregation='Median of three block medians per fit; arithmetic mean of seed centers. Peak center and observed range are distinct; no iid interval claim.',
                scope='Same-session Robin VAL CPU-input to full CPU-output CUDA deployment only; no v7 timing ratios or CPU benchmark')


def cost_grid(items, prepared, decision, initial_binding, job):
    rows = []
    inputs = prepared['selected_inputs']

    def measure_item(item, block, sentinel=None):
        cleanup()
        model, info = load_candidate(item)
        if item['family'] == 'lora':
            info['merge'] = merge(model, inputs, job)
        info.update(model_bytes(model))
        grid = configs(item['family']) if sentinel is None else [(1, 'original', None), (4, 'original', None)]
        if sentinel is None and block == 1:
            grid = grid[::-1]
        elif sentinel is None and block == 2:
            grid = grid[1:] + grid[:1]
        for batch, variant, micro in grid:
            job.heartbeat(f'cost block{block} {item["id"]} B{batch} {variant} M{micro} {sentinel}')
            row = dict(id=item['id'], family=item['family'], seed=item['seed'], block=block, batch=batch,
                       variant=variant, micro=micro, device='cuda', sentinel=sentinel, sequence=len(rows),
                       model=info, telemetry_before=telemetry())
            row.update(measure(model, inputs, batch, variant, micro, job))
            row['telemetry_after'] = telemetry()
            rows.append(row)
            save_json(HERE / 'cost_rows_partial.json', dict(rows=rows, source=source_receipt()))
        del model
        cleanup()

    for block in range(3):
        measure_item(items[0], block, 'pre')
        order = items if block == 0 else items[::-1] if block == 1 else items[2:] + items[:2]
        for item in order:
            measure_item(item, block)
        measure_item(items[0], block, 'post')
    if len(rows) != 75 or sum(r['sentinel'] is None for r in rows) != 63:
        raise ValueError('Expected 63 primary and 12 sentinel GPU rows')
    if binding(items, prepared, decision) != initial_binding:
        raise RuntimeError('Timing source/state/data changed while models were loaded')
    save_json(HERE / 'cost_gpu_rows.json', dict(status='complete', rows=rows, environment=cost_environment(),
                                               source=source_receipt(), decision_sha256=digest(HERE / 'cost_decision.json'),
                                               parity_sha256=digest(HERE / 'cost_parity.json'), binding=initial_binding))
    save_json(HERE / 'cost_summary.json', summarize(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--phase', choices=['parity', 'gpu'], required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    args = parser.parse_args()
    if not 600 <= args.reserve_seconds <= 1200:
        parser.error('Reserve 600–1200 GPU job-seconds within the shared 7200-second cap')
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.name):
        parser.error('Use a unique simple ledger name')
    decision = decision_guard()
    preserved = ['cost_parity.json', 'cost_parity_partial.json'] if args.phase == 'parity' else ['cost_gpu_rows.json', 'cost_rows_partial.json', 'cost_summary.json']
    if any((HERE / name).exists() for name in preserved):
        raise RuntimeError('Preserve the existing completed/partial cost attempt before an explicit repair retry')
    with Job(args.name, category='gpu', reserve_seconds=args.reserve_seconds,
             metadata={'dataset': 'robin', 'fit_count': 0, 'phase': args.phase}) as job:
        global np, torch, psutil
        import numpy as np
        import torch
        import psutil
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            items = candidates()
            prepared = prepare()
            with torch.inference_mode():
                if args.phase == 'parity':
                    parity_all(items, prepared, decision, job)
                else:
                    parity = checked_json(HERE / 'cost_parity.json')
                    initial_binding = binding(items, prepared, decision)
                    if parity.get('status') != 'PASS' or parity['binding'] != initial_binding:
                        raise RuntimeError('Cost requires unchanged parity-tested code, states, selection and data')
                    cost_grid(items, prepared, decision, initial_binding, job)
        except BaseException:
            save_json(HERE / (args.name + '_failure.json'), dict(status='failed', phase=args.phase,
                      error=traceback.format_exc(), source=source_receipt(), environment=environment_receipt()))
            raise


if __name__ == '__main__':
    main()
