"""Measure only the sealed v4 VAL-selected deployment configurations."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime_v4 import (CACHE, HERE, ROOT, Job, environment_receipt,
                        load_data, save_json, source_receipt)
from evaluate_v4 import check_receipt, load_module, load_seal, read_json, receipt, resolve, validate_name


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def cuda_cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}


def process_memory():
    memory = psutil.Process().memory_info()
    return {name: int(getattr(memory, name)) for name in ('rss', 'vms', 'peak_wset') if hasattr(memory, name)}


def telemetry():
    fields = ['index', 'uuid', 'temperature.gpu', 'clocks.current.sm', 'clocks.current.memory',
              'utilization.gpu', 'power.draw', 'pstate']
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
                            capture_output=True, text=True)
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                               capture_output=True, text=True)
    return {'utc': time.time(), 'fields': fields, 'gpu_rows': result.stdout.strip().splitlines(),
            'exit_code': result.returncode, 'stderr': result.stderr.strip(),
            'own_process_gpu_memory': [line for line in processes.stdout.splitlines()
                                       if line.split(',')[0].strip() == str(os.getpid())],
            'process_query_exit': processes.returncode, 'meaning': 'Read-only telemetry; N/A is not zero.'}


def prepare_inputs(sealed):
    data = load_data(include_test=False)
    origins = np.asarray(data['val_origins'], dtype=np.int64)
    if origins.tolist() != sealed['origins']['val'] or len(origins) != 30:
        raise ValueError('VAL origin identity differs from the frozen selection')
    indices = np.linspace(0, len(origins) - 1, 24, dtype=int)
    chosen = origins[indices]
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in chosen]))
    if inputs.shape != (24, 512, 32) or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all():
        raise ValueError('Cost inputs must be finite standardized FP32 [24,512,32]')
    if data['columns'].astype(str).tolist() != sealed['channels']:
        raise ValueError('Cost channel identities/order changed')
    return {'inputs': inputs, 'origins': chosen, 'all_val_origins': origins, 'indices': indices}


def forward_cpu(model, inputs, device):
    return model(inputs.to(device=device, dtype=torch.float32)).float().cpu()


def load_candidate(row, prepared, device, job):
    if row['kind'] == 'shared':
        class Shared(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.register_buffer('weight', torch.as_tensor(weight, dtype=torch.float32).clone())
                self.register_buffer('bias', torch.as_tensor(bias, dtype=torch.float32).clone())

            def forward(self, x):
                return (x.transpose(1, 2) @ self.weight + self.bias).transpose(1, 2)

        with np.load(check_receipt(row['weights']), allow_pickle=False) as archive:
            model = Shared(archive['weight'], archive['bias']).to(device).eval()
    else:
        module = load_module(HERE / 'model_v4.py', '_v4_cost_model')
        model = module.restore_model(resolve(row['checkpoint']), device=device).eval()
    family = row['spec']['family']
    info = {'registered_trainable_parameters_before_merge': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters_before_merge': sum(p.numel() for p in model.parameters()),
            'task_channels': 32, 'backbone_series_per_origin': 0 if family in ('linear_res', 'shared') else
            32 if family in ('f0', 'lora') else 8}
    before = torch.cat([forward_cpu(model, chunk, device) for chunk in prepared['inputs'].split(4)])
    if before.shape != (24, 48, 32) or not torch.isfinite(before).all():
        raise ValueError('Every cost candidate must produce all channels/horizons')
    with np.load(check_receipt(row['best_val_prediction']), allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], prepared['all_val_origins']):
            raise ValueError('Saved VAL prediction origins changed')
        saved = torch.from_numpy(archive['prediction'][prepared['indices']].copy()).float()
    atol, rtol = 1e-5, 1e-4
    absolute_error = (before - saved).abs()
    old_tolerance_violations = int(torch.count_nonzero(absolute_error > 1e-6 + 1e-5 * saved.abs()))
    torch.testing.assert_close(before, saved, atol=atol, rtol=rtol)
    info['validation_replay'] = {'pass': True, 'max_abs_error': float(absolute_error.max()),
                                 'atol': atol, 'rtol': rtol, 'prediction': row['best_val_prediction'],
                                 'elements': before.numel(), 'old_gpu_atol': 1e-6, 'old_gpu_rtol': 1e-5,
                                 'old_gpu_tolerance_violations': old_tolerance_violations,
                                 'repair': receipt(HERE / 'repairs' / 'cost_replay_fix.json')}
    del saved, absolute_error
    if family == 'lora':
        job.heartbeat(f'{row["id"]}: checking returned merged LoRA model')
        model.backbone = model.backbone.merge_and_unload(safe_merge=True)
        after = torch.cat([forward_cpu(model, chunk, device) for chunk in prepared['inputs'].split(4)])
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)
        info['merge_parity'] = {'pass': True, 'max_abs_error': float((after - before).abs().max()),
                                'atol': 1e-5, 'rtol': 1e-4, 'origins': 24}
        del after
    info.update(total_deployed_parameters=sum(p.numel() for p in model.parameters()),
                deployed_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                deployed_buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()))
    info['total_deployed_tensor_bytes'] = info['deployed_parameter_bytes'] + info['deployed_buffer_bytes']
    if family == 'shared':
        info['fitted_affine_coefficients_stored_as_buffers'] = 24624
    del before
    return model, info


def check_limits(job, deadline=None):
    job.check_limits()
    if deadline is not None and time.perf_counter() >= deadline:
        raise RuntimeError('CONDITIONAL_DIAGNOSTIC_600_SECOND_LIMIT')


def measure(model, inputs, batch, device, job, resident=False, deadline=None):
    chunks = list(inputs.split(batch))
    if device == 'cuda':
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    if resident:
        chunks = [chunk.to('cuda') for chunk in chunks]

    def forward(chunk):
        return model(chunk) if resident else forward_cpu(model, chunk, device)

    for warmup in range(10):
        check_limits(job, deadline)
        forward(chunks[warmup % len(chunks)])
    start_event = end_event = None
    if device == 'cuda':
        torch.cuda.synchronize()
        if resident:
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            end_event.record()
            end_event.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations, events = [], []
    for repeat in range(20):
        check_limits(job, deadline)
        if device == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        if resident:
            start_event.record()
        for chunk in chunks:
            forward(chunk)
        if resident:
            end_event.record()
            end_event.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        if resident:
            events.append(start_event.elapsed_time(end_event))
    result = {'seconds_per_24_origins': durations,
              'milliseconds_per_origin': summary([t * 1000 / 24 for t in durations]),
              'origins_per_second': summary([24 / t for t in durations]),
              'process_cpu_memory_bytes': process_memory()}
    if device == 'cuda':
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      resident_allocated_bytes=torch.cuda.memory_allocated())
    if resident:
        result.update(cuda_event_milliseconds_per_24_origins=events,
                      cuda_event_milliseconds_per_origin=summary([t / 24 for t in events]))
    check_limits(job, deadline)
    return result


def row_metadata(row):
    return {'id': row['id'], 'family': row['spec']['family'], 'seed': row['spec'].get('seed'),
            'dataset': 'hog', 'ed_mode': row['spec'].get('ed_mode'), 'latent': row['spec'].get('latent')}


def cost_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['include_in_primary']:
            grouped[(row['id'], row['batch_origins'], row['scope'])].append(row)
    fits = []
    for (identifier, batch, scope), members in sorted(grouped.items()):
        if len(members) != 3 or {r['block'] for r in members} != {0, 1, 2}:
            raise ValueError('Every main cost fit requires three complete order blocks')
        info = {key: members[0][key] for key in ('dataset', 'family', 'seed', 'ed_mode', 'latent')}
        fits.append(dict(info, id=identifier, batch_origins=batch, scope=scope,
                         milliseconds_per_origin=summary([r['milliseconds_per_origin']['median'] for r in members]),
                         origins_per_second=summary([r['origins_per_second']['median'] for r in members]),
                         peak_allocated_bytes=[r.get('peak_allocated_bytes') for r in members],
                         peak_reserved_bytes=[r.get('peak_reserved_bytes') for r in members],
                         total_deployed_tensor_bytes=members[0]['total_deployed_tensor_bytes']))
    groups = defaultdict(list)
    for fit in fits:
        groups[(fit['family'], fit['ed_mode'], fit['batch_origins'], fit['scope'])].append(fit)
    centers = []
    for (family, mode, batch, scope), members in groups.items():
        centers.append({'family': family, 'ed_mode': mode, 'batch_origins': batch, 'scope': scope,
                        'fit_ids': [r['id'] for r in members],
                        'mean_fit_median_ms_per_origin': float(np.mean([r['milliseconds_per_origin']['median'] for r in members])),
                        'mean_fit_median_origins_per_second': float(np.mean([r['origins_per_second']['median'] for r in members]))})
    return {'fits': fits, 'groups': centers,
            'definition': 'Per fit: median of three block medians; group center: arithmetic mean of individual fit medians. '
                          'F0 post excluded. Within-block raw 20-pass IQR remains in rows. No independent-sample confidence interval.'}


def drift_checks(rows):
    sentinels = []
    for block in range(3):
        for batch in (1, 4):
            pair = {r['sentinel_role']: r for r in rows if r.get('sentinel_role') and
                    r['block'] == block and r['batch_origins'] == batch}
            pre, post = [pair[k]['milliseconds_per_origin'] for k in ('pre', 'post')]
            sentinels.append({'block': block, 'batch_origins': batch,
                              'pre_median_ms': pre['median'], 'post_median_ms': post['median'],
                              'post_over_pre': post['median'] / pre['median'],
                              'iqr_disjoint': pre['q75'] < post['q25'] or post['q75'] < pre['q25']})
    reversals = []
    for batch in (1, 4):
        timing = defaultdict(dict)
        for row in rows:
            if row['include_in_primary'] and row['batch_origins'] == batch:
                timing[row['id']][row['block']] = row['milliseconds_per_origin']['median']
        identifiers = sorted(timing)
        for offset, left in enumerate(identifiers):
            for right in identifiers[offset + 1:]:
                differences = [timing[left][b] - timing[right][b] for b in range(3)]
                if min(differences) < 0 < max(differences):
                    reversals.append({'batch_origins': batch, 'left': left, 'right': right,
                                      'block_latency_differences_ms': differences})
    return {'f0_pre_post': sentinels, 'pair_order_reversals': reversals,
            'diagnostic_allowed': any(r['iqr_disjoint'] for r in sentinels) or bool(reversals),
            'interpretation': 'Descriptive drift/rank checks only; no automatic correction, deletion, repetition or causal attribution.'}


def measurement_schedule(identifiers, f0, device):
    middle = [i for i in identifiers if i != f0]
    schedule = []
    for block in range(3):
        ordered = middle if block == 0 else list(reversed(middle)) if block == 1 else middle[len(middle) // 2:] + middle[:len(middle) // 2]
        visits = [(i, None) for i in ordered]
        if device == 'cuda':
            visits = [(f0, 'pre')] + visits + [(f0, 'post')]
        for ordinal, (identifier, sentinel) in enumerate(visits):
            schedule.append((block, ordinal, identifier, sentinel))
    return schedule


def reuse_cost_rows(args, sealed, schedule, models, device):
    if args.reuse_cost is None:
        return [], None
    path = resolve(args.reuse_cost)
    if not path.name.endswith('_partial.json'):
        raise ValueError('Reuse requires the preserved partial JSON from a failed cost attempt')
    original_name = path.name.removesuffix('_partial.json')
    original = read_json(path)
    attempt_path = path.with_name(original_name + '_attempt.json')
    attempt = read_json(attempt_path)
    if original['stage'] != args.stage or attempt['stage'] != args.stage or attempt['status'] != 'failed':
        raise ValueError('Only completed rows of a failed attempt at the same cost stage may be reused')
    if original['seal'] != receipt(HERE / 'evaluation_seal.json'):
        raise ValueError('Cannot reuse measurements from a different model/data/selection seal')
    source = attempt['source']
    for filename in ('model_v4.py', 'train_v4.py', 'evaluate_v4.py', 'runtime_v4.py',
                     'fit_shared_v4.py', 'PLAN.md', 'protocol.json', 'data_contract.json'):
        key = (HERE / filename).relative_to(ROOT).as_posix()
        if source[key] != receipt(HERE / filename)['sha256']:
            raise ValueError(f'Cost measurement dependency changed: {filename}')
    if attempt['environment']['versions'] != environment_receipt()['versions']:
        raise ValueError('Cost measurement package versions changed')
    cost_key = (HERE / 'cost_v4.py').relative_to(ROOT).as_posix()
    repair = read_json(HERE / 'repairs' / 'cost_replay_fix.json')
    if source[cost_key] not in (repair['original_code']['sha256'], repair['repaired_code']['sha256']):
        raise ValueError('The previous cost source is not covered by the recorded repair')
    check_receipt(repair['repaired_code'])
    expected = [(b, identifier, batch, sentinel, ordinal) for b, ordinal, identifier, sentinel in schedule for batch in (1, 4)]
    rows = original['rows']
    keys = [(r['block'], r['id'], r['batch_origins'], r.get('sentinel_role'), r['order_position']) for r in rows]
    if not rows or keys != expected[:len(rows)] or len(rows) >= len(expected):
        raise ValueError('Preserved rows must be a nonempty, unique, unfinished prefix of the sealed cost order')
    for row in rows:
        if any(row[key] != value for key, value in row_metadata(models[row['id']]).items()):
            raise ValueError('Preserved row metadata differs from its sealed model')
        scope = 'deployment_cpu_to_cpu' if device == 'cuda' else 'cpu_fp32'
        if row['scope'] != scope or row['include_in_primary'] != (row.get('sentinel_role') != 'post'):
            raise ValueError('Preserved row deployment/sentinel role changed')
        raw = row['seconds_per_24_origins']
        if len(raw) != 20 or not all(np.isfinite(t) and t > 0 for t in raw):
            raise ValueError('A preserved row has incomplete or invalid timing passes')
        if not row['validation_replay']['pass'] or (row['family'] == 'lora' and not row['merge_parity']['pass']):
            raise ValueError('A preserved row did not pass output parity')
        row.setdefault('measurement_run', original_name)
        row.setdefault('measurement_cost_source_sha256', source[cost_key])
    interrupted_block = rows[-1]['block']
    resume = {'partial': receipt(path), 'attempt': receipt(attempt_path), 'original_source': source,
              'original_measurement_run': original_name, 'reused_rows': len(rows),
              'prior_resume': original.get('resume'), 'repair': receipt(HERE / 'repairs' / 'cost_replay_fix.json'),
              'interrupted_blocks': [interrupted_block], 'single_uninterrupted_session': False,
              'scope': 'Completed rows and their original parity tolerances are retained. '
                       'Failure, VAL parity diagnosis and restart interrupt the original order block; '
                       'its original F0 pre sentinel is retained, not remeasured.'}
    return rows, resume


def measure_primary(args, sealed, prepared, job):
    models = {r['id']: r for r in sealed['models']}
    device = 'cuda' if args.stage == 'gpu' else 'cpu'
    identifiers = sealed['cost']['gpu_model_ids' if device == 'cuda' else 'cpu_model_ids']
    schedule = measurement_schedule(identifiers, sealed['roles']['f0'][0], device)
    rows, resume = reuse_cost_rows(args, sealed, schedule, models, device)
    complete = {(r['block'], r['id'], r['batch_origins'], r.get('sentinel_role')) for r in rows}
    source_hash = receipt(HERE / 'cost_v4.py')['sha256']
    for block, ordinal, identifier, sentinel in schedule:
        batches = [b for b in (1, 4) if (block, identifier, b, sentinel) not in complete]
        if not batches:
            continue
        job.check_limits()
        baseline = cuda_cleanup() if device == 'cuda' else process_memory()
        if device == 'cuda' and baseline['allocated']:
            raise RuntimeError('A previous model/output remains resident on CUDA')
        model, info = load_candidate(models[identifier], prepared, device, job)
        for batch in batches:
            job.heartbeat(f'{args.name}: block{block} {identifier} {sentinel} batch{batch}')
            started = time.time()
            before = telemetry() if device == 'cuda' else None
            measured = measure(model, prepared['inputs'], batch, device, job)
            row = dict(row_metadata(models[identifier]), block=block, order_position=ordinal,
                       batch_origins=batch, sentinel_role=sentinel, include_in_primary=sentinel != 'post',
                       scope='deployment_cpu_to_cpu' if device == 'cuda' else 'cpu_fp32',
                       measurement_run=args.name, measurement_cost_source_sha256=source_hash,
                       measurement_started_utc=started, measurement_completed_utc=time.time(),
                       baseline_memory=baseline, telemetry_before=before,
                       telemetry_after=telemetry() if device == 'cuda' else None, **info, **measured)
            rows.append(row)
            save_json(HERE / f'{args.name}_partial.json', {'status': 'running', 'stage': args.stage,
                      'seal': receipt(HERE / 'evaluation_seal.json'), 'resume': resume, 'rows': rows})
        del model
        if device == 'cuda' and cuda_cleanup()['allocated']:
            raise RuntimeError('Model cleanup left CUDA allocations')
        gc.collect()
    expected = (len(identifiers) + int(device == 'cuda')) * 3 * 2
    if len(rows) != expected:
        raise ValueError('Incomplete precommitted cost design')
    result = {'rows': rows, 'expected_rows': expected, 'summary': cost_summary(rows), 'model_ids': identifiers,
              'resume': resume,
              'scope': 'Standardized FP32 CPU input -> complete CPU [B,48,32] output; synchronized wall for GPU.',
              'memory_scope': 'GPU allocated/reserved peaks after shape-specific warmup. Process RSS/peak working set '
                              'includes runtime/data; deployed tensor bytes include parameters and buffers.'}
    if device == 'cuda':
        result['drift_checks'] = drift_checks(rows)
        result['device'] = torch.cuda.get_device_name()
    return result


def measure_diagnostic(args, sealed, prepared, job, deadline):
    if not args.reason or args.trigger_cost is None:
        raise ValueError('A conditional diagnostic requires an observed reason and completed primary GPU cost')
    trigger = read_json(args.trigger_cost)
    if trigger['status'] != 'complete' or trigger['stage'] != 'gpu' or not trigger['drift_checks']['diagnostic_allowed']:
        raise ValueError('Primary results do not satisfy the precommitted diagnostic trigger')
    if trigger['seal']['sha256'] != receipt(HERE / 'evaluation_seal.json')['sha256']:
        raise ValueError('Diagnostic trigger belongs to another model/data seal')
    models = {r['id']: r for r in sealed['models']}
    identifiers = sealed['roles']['f0'] + [next(i for i in sealed['roles'][f] if models[i]['spec']['seed'] == 92601)
                                          for f in ('tsfm_res', 'linear_res')]
    rows = []
    for identifier in identifiers:
        check_limits(job, deadline)
        if cuda_cleanup()['allocated']:
            raise RuntimeError('Prior CUDA allocations remain before diagnostic')
        model, info = load_candidate(models[identifier], prepared, 'cuda', job)
        for batch in (1, 4):
            measured = measure(model, prepared['inputs'], batch, 'cuda', job, resident=True, deadline=deadline)
            rows.append(dict(row_metadata(models[identifier]), batch_origins=batch, scope='gpu_resident_diagnostic',
                             block=0, include_in_primary=False, measurement_run=args.name,
                             measurement_cost_source_sha256=receipt(HERE / 'cost_v4.py')['sha256'],
                             **info, **measured))
            save_json(HERE / f'{args.name}_partial.json', {'status': 'running', 'stage': args.stage, 'rows': rows})
        del model
        if cuda_cleanup()['allocated']:
            raise RuntimeError('Diagnostic model cleanup left CUDA allocations')
    return {'rows': rows, 'reason': args.reason, 'trigger': receipt(args.trigger_cost),
            'performance_claim': 'One diagnostic session only; excluded from primary deployment comparison.',
            'event_scope': 'Same full forward on GPU-resident inputs; CUDA events measure a stream interval that can '
                           'include CPU-dispatch idle gaps, not a sum of pure kernel times. '
                           'Deployment minus resident time is not a transfer-only estimate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('gpu', 'cpu', 'diagnostic'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reserve-seconds', type=float, required=True)
    parser.add_argument('--reason', default='')
    parser.add_argument('--trigger-cost', type=Path)
    parser.add_argument('--reuse-cost', type=Path,
                        help='Preserved failed-run partial JSON; completed rows are validated and never retimed')
    args = parser.parse_args()
    validate_name(args.name)
    paths = [HERE / f'{args.name}{suffix}.json' for suffix in ('', '_partial', '_attempt')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve all prior cost attempts; use a new run name')
    diagnostic_marker = HERE / 'diagnostic_session.json'
    if args.stage == 'diagnostic' and (diagnostic_marker.exists() or not args.reason or args.trigger_cost is None):
        raise ValueError('Only one conditional diagnostic session is allowed, with recorded trigger and reason')
    if args.stage == 'diagnostic' and args.reserve_seconds > 600:
        raise ValueError('Conditional diagnostic reservation must not exceed 600 seconds')
    if args.stage == 'diagnostic' and args.reuse_cost:
        raise ValueError('The one conditional diagnostic session cannot reuse primary timing rows')
    attempt = {'status': 'running', 'stage': args.stage, 'command': sys.argv, 'started_utc': time.time(),
               'source': source_receipt(), 'environment': environment_receipt()}
    save_json(paths[-1], attempt)
    try:
        category = 'cpu_analysis' if args.stage == 'cpu' else 'gpu'
        with Job(args.name, category=category, reserve_seconds=args.reserve_seconds,
                 metadata={'stage': args.stage, 'conditional_gpu_diagnostic': args.stage == 'diagnostic'}) as job:
            deadline = time.perf_counter() + min(args.reserve_seconds, 600) if args.stage == 'diagnostic' else None
            if args.stage == 'diagnostic':
                with diagnostic_marker.open('x', encoding='utf-8') as handle:
                    json.dump({'name': args.name, 'started_utc': time.time(), 'reason': args.reason,
                               'maximum_seconds': 600, 'trigger': receipt(args.trigger_cost)}, handle, indent=2)
            global np, torch, psutil
            import numpy as np
            import torch
            import psutil
            from threadpoolctl import threadpool_limits, threadpool_info
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            sealed = load_seal()
            with threadpool_limits(limits=4), torch.inference_mode():
                prepared = prepare_inputs(sealed)
                if args.stage == 'diagnostic':
                    result = measure_diagnostic(args, sealed, prepared, job, deadline)
                else:
                    result = measure_primary(args, sealed, prepared, job)
                result.update(schema_version=1, status='complete', stage=args.stage, name=args.name,
                              seal=receipt(HERE / 'evaluation_seal.json'), val_origins=prepared['origins'].tolist(),
                              source=source_receipt(), environment=environment_receipt(),
                              precision='float32', allow_tf32_matmul=False, allow_tf32_cudnn=False,
                              torch_threads=torch.get_num_threads(), threadpools=threadpool_info(),
                              inference_mode=True, warmup_calls=10, passes_per_order_block=20,
                              order_blocks=1 if args.stage == 'diagnostic' else 3,
                              uncertainty='Same local device; per-row measurement runs and interrupted blocks are recorded. '
                                          'Timing passes/blocks are not independent scientific replicates.',
                              no_target_scores=True, new_fits=0, completed_utc=time.time())
                save_json(paths[0], result)
            attempt['status'] = 'complete'
            print(json.dumps({'output': str(paths[0]), 'rows': len(result['rows'])}), flush=True)
    except BaseException:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        attempt['ended_utc'] = time.time()
        save_json(paths[-1], attempt)


if __name__ == '__main__':
    main()
