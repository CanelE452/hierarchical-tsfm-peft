"""Conditional dataset-level cost campaign with a fixed grid and partial outcomes."""
import argparse
import gc
import sys
import time
from functools import lru_cache

import numpy as np

from runtime_v14 import HERE, ROOT, CACHE, Job, artifact, digest, environment_receipt, import_file, load_data, protocol, read_json, save_json
from evaluate_v14 import NEW, V11, V12, V13, evaluation_rows, previous_api, selected


@lru_cache(maxsize=1)
def helpers():
    previous_api()
    return import_file(V12 / 'cost_confirmation_v12.py', 'v14_cost_primitives')


def decision(job):
    value, choice = read_json(HERE / 'evaluation01.json'), selected()
    if value['status'] != 'complete' or set(value['datasets']) != set(protocol()['units']):
        raise ValueError('Cost eligibility requires the entire prescribed accuracy comparison')
    tasks = {}
    for dataset, unit in value['datasets'].items():
        candidate = unit['role_summary']['level_gelu']['periods']['combined']
        against = {}
        for family in ('pca_level', 'level_linear'):
            reference = unit['role_summary'][family]['periods']['combined']
            no_worse = all(candidate[metric] <= reference[metric] for metric in ('mse', 'mae'))
            strict = any(candidate[metric] < reference[metric] for metric in ('mse', 'mae'))
            against[family] = {'reference': {metric: reference[metric] for metric in ('mse', 'mae')},
                               'no_worse_mse_and_mae': no_worse, 'at_least_one_strict': strict}
        tasks[dataset] = {'candidate': {metric: candidate[metric] for metric in ('mse', 'mae')},
                          'against': against, 'trigger': all(row['no_worse_mse_and_mae'] and row['at_least_one_strict'] for row in against.values())}
    eligible = [dataset for dataset, item in tasks.items() if item['trigger']]
    out = {'status': 'eligible' if eligible else 'not_triggered',
           'tasks': tasks, 'eligible_datasets': eligible,
           'excluded_datasets': [dataset for dataset in tasks if dataset not in eligible],
           'meaning': 'Prespecified development cost-selection rule, not significance or a scientific pass threshold; accuracy for all datasets retained',
           'evaluation': artifact(HERE / 'evaluation01.json'), 'selection': artifact(HERE / 'selected.json')}
    path = HERE / 'cost_decision.json'
    if path.exists() and read_json(path) != out:
        raise ValueError('Do not change cost eligibility after evaluation')
    save_json(path, out)
    if out['status'] == 'not_triggered':
        save_json(HERE / 'cost_summary.json', {'status': 'not_triggered', 'decision': artifact(path),
                                             'eligible_datasets': [], 'excluded_datasets': out['excluded_datasets'],
                                             'new_measurements': 0, 'historical_ratios': False})
    job.heartbeat('Dataset-level cost eligibility fixed after all accuracy comparisons')
    return out


def configs(family, unit, device):
    c, k = unit['C'], unit['K']
    if device == 'cpu' or family in ('direct', 'direct_gelu'):
        return [{'batch': b, 'chunk_rows': None, 'execution': 'direct'} for b in (1, 4)]
    full = c if family in ('f0', 'full_mse', 'native') else k
    values = [{'batch': b, 'chunk_rows': b * full, 'execution': 'full'} for b in (1, 4)]
    if family == 'full_mse':
        values += [{'batch': 1, 'chunk_rows': k, 'execution': 'chunk_k'},
                   {'batch': 4, 'chunk_rows': 4 * k, 'execution': 'chunk_4k'},
                   {'batch': 4, 'chunk_rows': k, 'execution': 'chunk_k'}]
    return values


def grid(choice, device):
    h, p, rows = helpers(), protocol(), []
    eligible = read_json(HERE / 'cost_decision.json')['eligible_datasets']
    for block in range(p['cost']['blocks']):
        for dataset in h.rotated(eligible, block):
            instances = [row for row in evaluation_rows(choice, dataset) if row['family'] != 'group_res']
            if device == 'cpu':
                instances = [row for row in instances if row['family'] in ('direct', 'direct_gelu')]
            ordered = [(row, None) for row in h.rotated(instances, block)]
            if device == 'cuda':
                f0 = next(row for row in instances if row['family'] == 'f0')
                ordered = [(f0, 'pre')] + ordered + [(f0, 'post')]
            for row, sentinel in ordered:
                for config in h.rotated(configs(row['family'], p['units'][dataset], device), block):
                    item = {key: row[key] for key in ('dataset', 'id', 'family', 'seed')}
                    item.update(device=device, block=block, sentinel=sentinel, **config)
                    item.update(key=h.row_key(item), planned_sequence=len(rows))
                    rows.append(item)
    expected = (132 if device == 'cuda' else 24) * len(eligible)
    if len(rows) != expected or len({row['key'] for row in rows}) != expected:
        raise ValueError('132 GPU / 24 optional CPU rows per eligible dataset changed')
    return rows


def load_original(row, data, device):
    if row['family'] in NEW:
        from model_v14 import restore_model, deployment_copy
        return restore_model(row['checkpoint']['path'], device=device), deployment_copy
    source = row['source_row']
    if row['source_generation'] == 'v13':
        if str(V13) not in sys.path:
            sys.path.append(str(V13))
        module = import_file(V13 / 'model_v13.py', 'v14_old_group_model')
        item = source.get('checkpoint', row.get('checkpoint'))
        artifact(item['path'], item['sha256'])
        return module.restore_model(item['path'], device=device), module.deployment_copy
    if row['source_generation'] == 'v12':
        previous_api()
        module = import_file(V12 / 'model_confirmation_v12.py', 'v14_old_confirmation_model')
        if row['family'] == 'f0':
            return module.build_model(row['dataset'], 'f0', 92601, device=device), module.deployment_copy
        artifact(source['checkpoint']['path'], source['checkpoint']['sha256'])
        return module.restore_model(source['checkpoint']['path'], device=device), module.deployment_copy
    if row['source_generation'] != 'v11':
        raise ValueError('Unknown authoritative reference restore generation')
    module = import_file(V11 / 'model_v11.py', 'v14_old_model_api')
    if 'checkpoint' in source:
        item = source['checkpoint']
        artifact(item['path'], item['sha256'])
        restore = module.restore_reused_direct if source['family'] == 'direct_nlinear' and row['dataset'] in ('robin', 'jena') else module.restore_model
        return restore(item['path'], device=device), module.deployment_copy
    basis = data['pca_basis']
    spec = {'dataset': row['dataset'], 'seed': row['seed'] or 92601, 'family': source['family']}
    return module.build_model(spec, basis, sources=source['sources'], device=device), module.deployment_copy


def load_deployment(row, data, prepared, device, job, binding_sha256):
    h, api = helpers(), previous_api()
    folder = CACHE / 'cost_parity' / device
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path, prediction_path = folder / (row['id'] + '.json'), folder / (row['id'] + '.npz')
    original, deploy = load_original(row, data, device)
    original.eval()
    guard(job)
    if receipt_path.exists():
        saved = read_json(receipt_path)
        if saved['binding_sha256'] != binding_sha256:
            raise ValueError('Saved cost parity belongs to another campaign')
        artifact(saved['prediction']['path'], saved['prediction']['sha256'])
        with np.load(saved['prediction']['path'], allow_pickle=False) as archive:
            base = archive['prediction']
        model = deploy(original).eval()
        del original
        h.cleanup(device)
        replay = api.parity(h.predict(model, prepared['inputs'][:4], 4, None, device), base[:4], atol=1e-6, rtol=0)
        if not replay['pass']:
            raise ValueError('Restored deployment differs from the verified same-path model')
        info = dict(saved['model'])
        info.update(parity_receipt=artifact(receipt_path), restore_replay=replay,
                    parity_scope='Full VAL/chunk/merge parity once; four-origin restore replay on later loads')
        return model, info
    info = {'checkpoint': row.get('checkpoint'), 'registered_trainable_parameters': sum(p.numel() for p in original.parameters() if p.requires_grad),
            'source_generation': row.get('source_generation', 'v14'), 'source_row': row.get('source_row')}
    if row['family'] in NEW:
        info['new_correction_registered_parameters'] = info['registered_trainable_parameters']
        info['adaptation_scope'] = 'New correction only; parent was fitted previously. Total deployment state includes both; incremental trainable count is not total historical adaptation.'
    base = h.predict(original, prepared['inputs'], 4, None, device)
    model = deploy(original).eval()
    del original
    h.cleanup(device)
    merged = h.predict(model, prepared['inputs'], 4, None, device)
    info['merge_export_parity'] = api.parity(merged, base)
    if not info['merge_export_parity']['pass']:
        raise ValueError('Deployment export changed outputs')
    info.update(h.state_bytes(model))
    info['configuration_parity'] = []
    for config in configs(row['family'], protocol()['units'][row['dataset']], device):
        guard(job)
        prediction = h.predict(model, prepared['inputs'], config['batch'], config['chunk_rows'], device)
        check = api.parity(prediction, base)
        if not check['pass']:
            raise ValueError('Full/chunk output parity failed')
        info['configuration_parity'].append({**config, 'parity': check, 'val_scores': h.macro_scores(prediction, prepared)})
    info['reference_val_scores'] = h.macro_scores(base, prepared)
    if device == 'cpu':
        gpu_path = CACHE / 'cost_parity' / 'cuda' / (row['id'] + '.json')
        if gpu_path.exists():
            gpu = read_json(gpu_path)
            if gpu['binding_sha256'] != binding_sha256:
                raise ValueError('CPU/GPU parity campaign mismatch')
            artifact(gpu['prediction']['path'], gpu['prediction']['sha256'])
            with np.load(gpu['prediction']['path'], allow_pickle=False) as archive:
                check = api.parity(base, archive['prediction'])
            if not check['pass']:
                raise ValueError('Direct CPU/GPU output parity failed')
            info['cpu_gpu_parity'] = check
        else:
            info['cpu_gpu_parity'] = {'status': 'unavailable', 'reason': 'Corresponding GPU configuration not completed'}
    np.savez_compressed(prediction_path, prediction=merged)
    save_json(receipt_path, {'binding_sha256': binding_sha256, 'prediction': artifact(prediction_path), 'model': info,
                             'scope': 'VAL output parity only; predictions never used by timed online forward'})
    info['parity_receipt'] = artifact(receipt_path)
    return model, info


def campaign_elapsed(job):
    jobs = read_json(HERE / 'ledger.json')['jobs']
    spent = sum(row['elapsed_s'] for row in jobs if row['id'] != job.id
                and row['category'] == job.category and row['metadata'].get('v14_cost_campaign'))
    return spent + time.perf_counter() - job.started


def guard(job, margin=5):
    job.check_limits()
    if job.category == 'gpu' and campaign_elapsed(job) >= protocol()['cost']['reserve_seconds'] - margin:
        raise TimeoutError('V14_CORE_COST_CAP: preserve complete rows and unfinished grid')


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
        started = time.perf_counter()
        for chunk in chunks:
            h.forward_cpu_output(model, chunk, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return {'seconds_per_24_origins': times, 'milliseconds_per_origin': h.describe(np.asarray(times) * 1000 / 24),
            'origins_per_second': h.describe(24 / np.asarray(times)), 'warmup_calls': settings['warmups'], 'passes': settings['passes'],
            'resident_after_load': resident, 'process_cpu_memory': h.process_memory(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
            'final_allocated_bytes': torch.cuda.memory_allocated() if device == 'cuda' else None}


def run(device, job):
    import torch
    h, choice = helpers(), selected()
    trigger = read_json(HERE / 'cost_decision.json')
    if trigger['status'] != 'eligible':
        raise ValueError('Cost campaign was not triggered')
    if (HERE / f'cost_{device}_rows.json').exists():
        raise FileExistsError('Cost phase already complete')
    data, prepared = {}, {}
    eligible = trigger['eligible_datasets']
    for dataset in eligible:
        data[dataset] = load_data(dataset, include_test=False)
        ds, origins = data[dataset], data[dataset]['val_origins'][:24]
        if len(origins) != 24 or np.any(np.diff(origins) <= 0):
            raise ValueError('Need the fixed chronological 24 VAL origins')
        prepared[dataset] = {'inputs': torch.from_numpy(np.stack([ds['x'][o - 512:o] for o in origins]).astype(np.float32)),
            'target': np.stack([ds['x'][o:o + 48] for o in origins]), 'mask': np.stack([ds['finite'][o:o + 48] for o in origins]).astype(bool),
            'origins': origins}
    names = ('protocol.json', 'selected.json', 'reuse_manifest.json', 'data_manifest.json', 'cost_decision.json', 'model_v14.py',
             'runtime_v14.py', 'cost_v14.py', 'evaluate_v14.py')
    binding = {'files': {name: digest(HERE / name) for name in names}, 'settings': protocol()['cost'],
               'eligible_datasets': eligible, 'excluded_datasets': trigger['excluded_datasets'],
               'limited_grid': 'Full B1/B4 all listed roles; only full MSE-LoRA also has K/4K row chunks. GROUP costs not measured.',
               'inputs': {dataset: {'origins': value['origins'].tolist(), 'sha256': h.array_hash(value['inputs'].numpy())}
                          for dataset, value in prepared.items()}, 'source_primitives': artifact(V12 / 'cost_confirmation_v12.py')}
    bound_path = HERE / 'cost_binding.json'
    if bound_path.exists() and read_json(bound_path) != binding:
        raise ValueError('Cost campaign binding changed')
    save_json(bound_path, binding)
    bound_sha = digest(bound_path)
    planned = grid(choice, device)
    save_json(HERE / f'cost_{device}_grid.json', {'binding_sha256': bound_sha, 'rows': planned})
    folder = HERE / 'cost_rows' / device
    folder.mkdir(parents=True, exist_ok=True)
    results, model, identity = [], None, None
    outcome = 'partial'
    try:
        for config in planned:
            guard(job, 30)
            path = folder / (config['key'] + '.json')
            if path.exists():
                row = read_json(path)
                if row['binding_sha256'] != bound_sha or any(row[key] != value for key, value in config.items()):
                    raise ValueError('Partial cost row differs from the fixed grid')
                results.append(row)
                continue
            key = (config['dataset'], config['id'], config['block'], config['sentinel'])
            if key != identity:
                if model is not None:
                    del model
                    model = None
                h.cleanup(device)
                instance = next(row for row in evaluation_rows(choice, config['dataset']) if row['id'] == config['id'])
                model, info = load_deployment(instance, data[config['dataset']], prepared[config['dataset']], device, job, bound_sha)
                identity = key
            values = measure(model, prepared[config['dataset']], config, device, job)
            row = {**config, **values, 'model': info, 'binding_sha256': bound_sha, 'job_id': job.id, 'status': 'complete'}
            save_json(path, row)
            results.append(row)
            save_json(HERE / f'cost_{device}_partial.json', {'status': 'partial', 'rows': results, 'expected_rows': len(planned)})
        outcome = 'complete'
    except TimeoutError as error:
        save_json(HERE / 'cost_guard.json', {'status': 'budget_stopped', 'reason': str(error), 'job_id': job.id,
                   'campaign_elapsed_s': campaign_elapsed(job), 'complete_rows': len(results), 'expected_rows': len(planned)})
        outcome = 'budget_stopped'
    finally:
        if model is not None:
            del model
        h.cleanup(device)
    saved = {'status': outcome, 'rows': results, 'expected_rows': len(planned), 'binding_sha256': bound_sha,
             'environment': environment_receipt(), 'actual_device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
             'session_ids': sorted({row['job_id'] for row in results}), 'low_repetition_caution': 'Three passes per block; report drift and raw ranges'}
    save_json(HERE / f'cost_{device}_rows.json', saved)
    p = protocol()
    summary = h.summarize(results, device, {**p, 'units': {dataset: p['units'][dataset] for dataset in eligible}}) if outcome == 'complete' else {
        'status': outcome, 'complete_rows': len(results), 'expected_rows': len(planned), 'complete_frontier': False}
    save_json(HERE / f'cost_{device}_summary.json', summary)
    gpu_path, cpu_path = HERE / 'cost_cuda_summary.json', HERE / 'cost_cpu_summary.json'
    combined = {'status': read_json(gpu_path)['status'] if gpu_path.exists() else 'partial',
                'gpu': read_json(gpu_path) if gpu_path.exists() else None,
                'cpu': read_json(cpu_path) if cpu_path.exists() else {'status': 'not_run_optional'},
                'binding': artifact(bound_path), 'historical_ratios': False, 'decision': artifact(HERE / 'cost_decision.json')}
    combined.update(eligible_datasets=eligible, excluded_datasets=trigger['excluded_datasets'],
                    scope='Development-selected datasets only; no measured v14 cost claim for excluded datasets or GROUP')
    save_json(HERE / 'cost_summary.json', combined)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('decision', 'gpu', 'cpu'))
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    category = 'gpu' if args.action == 'gpu' else 'cpu_analysis'
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'fit_count': 0, 'v14_cost_campaign': args.action != 'decision', 'purpose': 'v14 cost ' + args.action}) as job:
        if args.action == 'decision':
            decision(job)
        else:
            job._stop.set()
            job._thread.join(timeout=20)
            if job._thread.is_alive():
                raise RuntimeError('Timing observer did not stop')
            import torch
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.no_grad():
                run('cuda' if args.action == 'gpu' else 'cpu', job)


if __name__ == '__main__':
    main()
