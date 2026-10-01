"""Same-session online costs, fixed 612 GPU / 48 CPU rows and bounded time."""
import argparse
import time
from functools import lru_cache

import numpy as np
from runtime_v15 import (HERE, ROOT, CACHE, DATASETS, SEEDS, Job, artifact, digest, environment_receipt,
                         import_file, load_data, protocol, read_json, save_json, configure)

NEW = ('raw_free', 'pca_free')
UNCOMPRESSED = ('f0', 'full_mse', 'native')


@lru_cache(maxsize=1)
def helpers():
    from model_v15 import reference_api
    reference_api()
    return import_file(ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001/cost_confirmation_v12.py',
                       'v15_cost_primitives')


def instances(dataset):
    from evaluate_v15 import evaluation_rows, selected
    rows = evaluation_rows(selected(), dataset)
    values = [r for r in rows if r['family'] in (*NEW, 'direct', 'a', *UNCOMPRESSED)]
    if len(values) != 13:
        raise ValueError('Cost requires thirteen model instances per dataset')
    return values


def configs(family, unit, device):
    c, k = unit['C'], unit['K']
    if family == 'direct':
        return [{'batch': b, 'chunk_rows': None, 'execution': 'direct'} for b in (1, 4)]
    values = [{'batch': b, 'chunk_rows': b * (c if family in UNCOMPRESSED else k), 'execution': 'full'} for b in (1, 4)]
    if family in UNCOMPRESSED:
        values += [{'batch': 1, 'chunk_rows': k, 'execution': 'chunk_k'},
                   {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'chunk_4k'},
                   {'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]
    else:
        values += [{'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]
    return values


def grid(device):
    h, p, rows = helpers(), protocol(), []
    for block in range(3):
        for dataset in h.rotated(list(DATASETS), block):
            values = instances(dataset)
            if device == 'cpu':
                values = [r for r in values if r['family'] == 'direct']
            ordered = [(r, None) for r in h.rotated(values, block)]
            if device == 'cuda':
                f0 = next(r for r in values if r['family'] == 'f0')
                ordered = [(f0, 'pre')] + ordered + [(f0, 'post')]
            for model, sentinel in ordered:
                entries = configs(model['family'], p['units'][dataset], device)
                entries = [r for r in entries if r['execution'] == 'full'] if sentinel else h.rotated(entries, block)
                for entry in entries:
                    row = {key: model[key] for key in ('dataset', 'id', 'family', 'seed')}
                    row.update(device=device, block=block, sentinel=sentinel, **entry)
                    row.update(key=h.row_key(row), planned_sequence=len(rows))
                    rows.append(row)
    expected = 612 if device == 'cuda' else 48
    if len(rows) != expected or len({r['key'] for r in rows}) != expected:
        raise ValueError('Prospective cost grid changed')
    return rows


def parity(a, b, atol=1e-5, rtol=1e-4):
    a, b = np.asarray(a), np.asarray(b)
    delta = np.abs(a - b)
    return {'pass': bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                         and np.all(delta <= atol + rtol * np.abs(b))),
            'max_absolute': float(delta.max()),
            'max_relative': float((delta / np.maximum(np.abs(b), np.finfo(np.float32).tiny)).max()),
            'violating_elements': int(np.sum(delta > atol + rtol * np.abs(b))), 'atol': atol, 'rtol': rtol}


def load_original(row, device):
    from model_v15 import build_model, load_reference
    if row['family'] in NEW:
        entry = row['checkpoint']
        artifact(entry['path'], entry['sha256'])
        with np.load(entry['path'], allow_pickle=False) as arrays:
            w = arrays['W']
        return build_model(row['dataset'], row['family'], row['seed'], w, device=device), lambda m: m
    model, deploy, _ = load_reference(row['dataset'], row['family'], row['seed'], device=device)
    return model, deploy


def load_deployment(row, prepared, device, job, binding_sha):
    h = helpers()
    folder = CACHE / 'cost_parity' / device
    folder.mkdir(parents=True, exist_ok=True)
    receipt, archive = folder / (row['id'] + '.json'), folder / (row['id'] + '.npz')
    original, deploy = load_original(row, device)
    original.eval()
    guard(job)
    if receipt.exists():
        prior = read_json(receipt)
        if prior['binding_sha256'] != binding_sha:
            raise ValueError('Cost replay source changed')
        artifact(prior['prediction']['path'], prior['prediction']['sha256'])
        with np.load(prior['prediction']['path'], allow_pickle=False) as a:
            base = a['prediction']
        model = deploy(original).eval()
        del original
        h.cleanup(device)
        check = parity(h.predict(model, prepared['inputs'][:4], 4, None, device), base[:4], 1e-6, 0)
        if not check['pass']:
            raise ValueError('Same-path deployment replay failed')
        info = dict(prior['model'], replay=check, parity_receipt=artifact(receipt))
        return model, info
    base = h.predict(original, prepared['inputs'], 4, None, device)
    info = {'checkpoint': row.get('checkpoint'), 'registered_requires_grad_at_restore': sum(p.numel() for p in original.parameters() if p.requires_grad),
            'adaptation_note': 'Deployment is frozen; closed-form W count differs from requires_grad. Historical parent fits are reused.'}
    info['historical_adaptation_registered_parameters'] = {
        'direct': 24624, 'a': 294912 + 17920, 'f0': 0, 'full_mse': 294912,
        'native': 294912, 'raw_free': 24624, 'pca_free': 24624}[row['family']]
    info['this_campaign_neural_optimizer_updates'] = 0
    info['historical_adaptation_note'] = 'Sums of registered adaptation stages from preserved recipes; not simultaneous trainability or deployment size.'
    model = deploy(original).eval()
    del original
    h.cleanup(device)
    prediction = h.predict(model, prepared['inputs'], 4, None, device)
    check = parity(prediction, base)
    if not check['pass']:
        raise ValueError('Merge/export parity failed')
    info.update(h.state_bytes(model), merge_export_parity=check, configuration_parity=[])
    if row['family'] in NEW:
        c, k, _ = DATASETS[row['dataset']]
        info.update(fitted_decoder_coefficients=c * k, historical_direct_fit_parameters=24624,
                    zero_decoder_fallback=bool(getattr(model, 'zero_decoder', False)))
        from cache_v15 import load_latent_cache, load_direct_cache
        space = 'raw' if row['family'] == 'raw_free' else 'pca'
        s = load_direct_cache(row['dataset'], row['seed'], 'val', prepared['origins'])['prediction']
        f = load_latent_cache(row['dataset'], space, 'val', prepared['origins'])['prediction']
        e = np.asarray(read_json(HERE / 'selector_manifest.json')['datasets'][row['dataset']]['E_' + space], dtype=np.float32)
        with np.load(row['checkpoint']['path'], allow_pickle=False) as a:
            w = a['W'].astype(np.float32)
        cached = s + (f - s @ e) @ w
        info['cached_online_parity'] = parity(base, cached)
        if not info['cached_online_parity']['pass']:
            raise ValueError('Online predictor differs from the cached evaluation function')
    for config in configs(row['family'], protocol()['units'][row['dataset']], device):
        guard(job)
        value = h.predict(model, prepared['inputs'], config['batch'], config['chunk_rows'], device)
        check = parity(value, base)
        if not check['pass']:
            raise ValueError('Full/chunk parity failed')
        info['configuration_parity'].append({**config, 'parity': check, 'val_scores': h.macro_scores(value, prepared)})
    if device == 'cpu':
        gpu_path = CACHE / 'cost_parity/cuda' / (row['id'] + '.json')
        if gpu_path.exists():
            gpu_receipt = read_json(gpu_path)
            artifact(gpu_receipt['prediction']['path'], gpu_receipt['prediction']['sha256'])
            with np.load(gpu_receipt['prediction']['path'], allow_pickle=False) as a:
                check = parity(prediction, a['prediction'])
            if not check['pass']:
                raise ValueError('DIRECT CPU/GPU parity failed')
            info['cpu_gpu_parity'] = check
        else:
            info['cpu_gpu_parity'] = {'status': 'unavailable', 'reason': 'Corresponding GPU parity was not completed within the campaign'}
    np.savez_compressed(archive, prediction=prediction, origins=prepared['origins'])
    save_json(receipt, {'binding_sha256': binding_sha, 'prediction': artifact(archive), 'model': info})
    info['parity_receipt'] = artifact(receipt)
    return model, info


def campaign_elapsed(job):
    jobs = read_json(HERE / 'ledger.json')['jobs']
    spent = sum(r['elapsed_s'] for r in jobs if r['id'] != job.id and r['category'] == job.category
                and r['metadata'].get('v15_cost_campaign'))
    return spent + time.perf_counter() - job.started


def guard(job, margin=5):
    job.check_limits()
    if job.category == 'gpu' and campaign_elapsed(job) >= protocol()['cost']['reserve_seconds'] - margin:
        raise TimeoutError('V15 cost time cap: preserve partial rows')


def measure(model, prepared, config, device, job):
    import torch
    h, settings = helpers(), protocol()['cost']
    chunks = list(prepared['inputs'].split(config['batch']))
    model.chunk_rows = config['chunk_rows']
    h.cleanup(device)
    resident = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()} if device == 'cuda' else None
    for i in range(settings['warmups']):
        guard(job)
        h.forward_cpu_output(model, chunks[i % len(chunks)], device)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(settings['passes']):
        guard(job)
        job.heartbeat({'row': config['key'], 'pass': repeat})
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for chunk in chunks:
            h.forward_cpu_output(model, chunk, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {'seconds_per_24_origins': times, 'milliseconds_per_origin': h.describe(np.asarray(times) * 1000 / 24),
            'origins_per_second': h.describe(24 / np.asarray(times)), 'warmup_calls': settings['warmups'], 'passes': settings['passes'],
            'resident_after_load': resident, 'process_cpu_memory': h.process_memory(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
            'final_allocated_bytes': torch.cuda.memory_allocated() if device == 'cuda' else None}


def run(device, job):
    import torch
    h, prepared = helpers(), {}
    for ds in DATASETS:
        data = load_data(ds)
        origins = data['val_origins'][:24]
        if len(origins) != 24 or np.any(np.diff(origins) <= 0):
            raise ValueError('Cost needs the first twenty-four VAL origins')
        prepared[ds] = {'inputs': torch.from_numpy(np.stack([data['x'][o-512:o] for o in origins]).astype(np.float32)),
            'origins': origins, 'target': np.stack([data['x'][o:o+48] for o in origins]),
            'mask': np.stack([data['finite'][o:o+48] for o in origins]).astype(bool)}
    binding = {'files': {name: digest(HERE / name) for name in ('protocol.json', 'selected.json', 'selector_manifest.json',
                    'reuse_manifest.json', 'model_v15.py', 'runtime_v15.py', 'cost_v15.py')},
        'settings': protocol()['cost'], 'inputs': {ds: {'origins': x['origins'].tolist(), 'sha256': h.array_hash(x['inputs'].numpy())}
                                                   for ds, x in prepared.items()}}
    bound = HERE / 'cost_binding.json'
    if bound.exists() and read_json(bound) != binding:
        raise ValueError('Cost source binding changed')
    save_json(bound, binding)
    bound_sha = digest(bound)
    planned = grid(device)
    save_json(HERE / f'cost_{device}_grid.json', {'binding_sha256': bound_sha, 'rows': planned})
    folder = HERE / 'cost_rows' / device
    folder.mkdir(parents=True, exist_ok=True)
    results, model, identity, outcome = [], None, None, 'partial'
    telemetry = [h.telemetry()] if device == 'cuda' else []
    try:
        for config in planned:
            guard(job, 30)
            path = folder / (config['key'] + '.json')
            if path.exists():
                row = read_json(path)
                if row['binding_sha256'] != bound_sha or any(row[k] != v for k, v in config.items()):
                    raise ValueError('Previously saved row changed')
                results.append(row)
                continue
            key = (config['dataset'], config['id'], config['block'], config['sentinel'])
            if identity != key:
                if model is not None:
                    del model
                    model = None
                h.cleanup(device)
                source = next(r for r in instances(config['dataset']) if r['id'] == config['id'])
                model, info = load_deployment(source, prepared[config['dataset']], device, job, bound_sha)
                identity = key
            row = {**config, **measure(model, prepared[config['dataset']], config, device, job), 'model': info,
                   'binding_sha256': bound_sha, 'job_id': job.id, 'status': 'complete'}
            save_json(path, row)
            results.append(row)
            if len(results) % 12 == 0:
                save_json(HERE / f'cost_{device}_progress.json', {'completed': len(results), 'expected': len(planned), 'last': config})
        outcome = 'complete'
    except TimeoutError as error:
        save_json(HERE / 'cost_guard.json', {'status': 'budget_stopped', 'reason': str(error), 'complete_rows': len(results),
                    'expected_rows': len(planned), 'elapsed_seconds': campaign_elapsed(job)})
        outcome = 'budget_stopped'
    finally:
        if model is not None:
            del model
        h.cleanup(device)
    if device == 'cuda':
        telemetry.append(h.telemetry())
    saved = {'status': outcome, 'rows': results, 'expected_rows': len(planned), 'binding_sha256': bound_sha,
             'environment': environment_receipt(), 'telemetry': telemetry,
             'actual_device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}
    save_json(HERE / f'cost_{device}_rows.json', saved)
    summary = h.summarize(results, device, protocol()) if outcome == 'complete' else {
        'status': outcome, 'complete_rows': len(results), 'expected_rows': len(planned), 'complete_frontier': False}
    save_json(HERE / f'cost_{device}_summary.json', summary)
    save_json(HERE / 'cost_summary.json', {'status': read_json(HERE / 'cost_cuda_summary.json')['status'] if (HERE / 'cost_cuda_summary.json').exists() else 'partial',
        'gpu': read_json(HERE / 'cost_cuda_summary.json') if (HERE / 'cost_cuda_summary.json').exists() else None,
        'cpu': read_json(HERE / 'cost_cpu_summary.json') if (HERE / 'cost_cpu_summary.json').exists() else None,
        'historical_ratios': False, 'binding': artifact(bound), 'accuracy_dependent_dataset_filtering': False})
    print(f'{device}: {outcome}, {len(results)}/{len(planned)} rows', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('gpu', 'cpu'))
    p.add_argument('--job', required=True)
    p.add_argument('--reserve-s', required=True, type=float)
    a = p.parse_args()
    with Job('gpu' if a.action == 'gpu' else 'cpu_analysis', a.job, reserve_s=a.reserve_s,
             metadata={'purpose': 'same-session fixed-grid costs', 'v15_cost_campaign': True}) as job:
        job._stop.set()
        job._thread.join(timeout=20)
        configure()
        import torch
        with torch.no_grad():
            run('cuda' if a.action == 'gpu' else 'cpu', job)
