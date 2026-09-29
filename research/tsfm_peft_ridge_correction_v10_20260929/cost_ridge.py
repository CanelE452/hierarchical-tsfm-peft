"""Conditional online ridge deployment costs; writes only the new v10 ledger/artifacts."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import traceback

from runtime_ridge import HERE, ROOT, V9, Job, digest, load_data, parent_receipt_for, save_json

V6 = ROOT / 'research/tsfm_peft_level_confirmation_v6_20260928'
V7 = ROOT / 'research/tsfm_peft_practical_controls_v7_20260928'
ROLES = ('base_level', 'ridge_p', 'ridge_raw', 'f0', 'lora', 'direct_nlinear')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8'))


def checked(receipt):
    path = resolve(receipt['path'])
    if digest(path) != receipt['sha256']:
        raise ValueError('Artifact changed: ' + str(path))
    return path


def module(path, name):
    path = resolve(path)
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).resolve() != path.resolve():
            raise RuntimeError('Module name is bound to another source: ' + name)
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def helpers():
    module(V9 / 'runtime_v9.py', 'runtime_v9')
    legacy = module(V9 / 'cost_v9.py', '_ridge_cost_v9_helpers')
    legacy.np, legacy.torch, legacy.psutil = np, torch, psutil
    return legacy


def decision_guard(dataset):
    decision = read(HERE / 'cost_decision.json')
    entry = decision['datasets'][dataset]
    if entry['run_cost'] is not True or not entry.get('reason', '').strip():
        raise RuntimeError('Cost requires an affirmative, explained same-dataset decision')
    for name in ('robin', 'jena'):
        path = checked(decision['evaluations'][name])
        value = read(path)
        if value['status'] != 'complete' or value['dataset'] != name:
            raise RuntimeError('Both fixed accuracy evaluations must be complete before cost')
    return decision


def candidates(dataset, legacy):
    selection_path = V6 / 'selected.json' if dataset == 'robin' else V7 / 'selected_jena.json'
    source = V6 / 'model_v6.py' if dataset == 'robin' else V7 / 'model_v7.py'
    items = []
    for family, role in (('level_res', 'base_level'), ('f0', 'f0'), ('lora', 'lora')):
        items.extend(legacy.item_from_record(record, role, source, dataset)
                     for record in legacy.legacy_records(selection_path, family))
    items.extend(legacy.item_from_record(record, 'direct_nlinear', V7 / 'model_v7.py', dataset)
                 for record in legacy.legacy_records(V7 / f'selected_{dataset}.json', 'direct_nlinear'))
    selected = read(HERE / 'selected.json')['datasets'][dataset]
    for role in ('ridge_p', 'ridge_raw'):
        entry = selected[role]
        records = sorted(entry['runs'], key=lambda r: r['spec']['seed'])
        if len(records) != 2 or {r['spec']['seed'] for r in records} != {92601, 92602}:
            raise ValueError('Each ridge arm needs both selected parents')
        for record in records:
            if record['spec']['dataset'] != dataset or record['spec']['family'] != role:
                raise ValueError('Selected ridge specification differs from its role')
            if record['spec']['penalty'] != entry['penalty']:
                raise ValueError('Ridge penalty is not the frozen whole-VAL selection')
            checkpoint = checked({'path': record['checkpoint'], 'sha256': record['checkpoint_sha256']})
            val_prediction = record['best_val_prediction']
            checked(val_prediction)
            items.append({'id': record['id'], 'role': role, 'family': role, 'dataset': dataset,
                          'seed': record['spec']['seed'], 'checkpoint': str(checkpoint),
                          'checkpoint_sha256': record['checkpoint_sha256'],
                          'source': str(HERE / 'model_ridge.py'), 'val_prediction': val_prediction,
                          'val_mse': record['best_val_mse'], 'penalty': entry['penalty']})
    if len(items) != 11 or len({item['id'] for item in items}) != 11:
        raise ValueError('Expected eleven distinct model/seed instances')
    for role in ROLES:
        if sum(item['role'] == role for item in items) != (1 if role == 'f0' else 2):
            raise ValueError('Missing cost role: ' + role)
    for item in (item for item in items if item['role'] == 'base_level'):
        parent = parent_receipt_for(dataset, item['seed'])
        if item['checkpoint_sha256'] != parent['checkpoint']['sha256']:
            raise ValueError('The separately timed baseline is not this ridge model\'s parent')
        checked(parent['model_source'])
    return items


def prepare(dataset):
    data = load_data(dataset, include_test=False)
    origins = np.asarray(data['val_origins'])
    if len(origins) != (30 if dataset == 'robin' else 371):
        raise ValueError('Approved whole-VAL origin count changed')
    indices = np.linspace(0, len(origins) - 1, 24, dtype=int)
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]))
    return {'origins': origins, 'inputs': inputs, 'selected_origins': origins[indices],
            'selected_inputs': inputs[indices], 'data_path': data['_path'],
            'target': np.stack([data['x'][o:o + 48] for o in origins]),
            'mask': np.stack([data['finite'][o:o + 48] for o in origins])}


def binding(items, prepared, decision):
    paths = [HERE / 'cost_ridge.py', HERE / 'runtime_ridge.py', HERE / 'model_ridge.py',
             HERE / 'PLAN.md', HERE / 'protocol.json', HERE / 'selected.json', HERE / 'cost_decision.json',
             V9 / 'cost_v9.py', V9 / 'runtime_v9.py', V9 / 'reuse_manifest.json',
             ROOT / 'src/hier_peft/lora.py', resolve(prepared['data_path']),
             V6 / 'selected.json', V7 / 'selected_robin.json', V7 / 'selected_jena.json']
    paths.extend(checked(decision['evaluations'][dataset]) for dataset in ('robin', 'jena'))
    for item in items:
        paths.extend([resolve(item['checkpoint']), resolve(item['source']), checked(item['val_prediction'])])
        if 'result_path' in item:
            paths.append(resolve(item['result_path']))
    return {str(path): digest(path) for path in dict.fromkeys(paths)}


def restore(item, legacy):
    checkpoint = checked({'path': item['checkpoint'], 'sha256': item['checkpoint_sha256']})
    source = resolve(item['source'])
    model = legacy.module(source).restore_model(checkpoint, device='cuda').eval()
    if model.family != item['family']:
        raise ValueError('Restored family differs from selected cost record')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = {'level_res': 17920, 'f0': 0, 'lora': 294912, 'direct_nlinear': 24624,
                'ridge_p': 0, 'ridge_raw': 0}
    if trainable != expected[item['family']]:
        raise ValueError('Unexpected parameters requiring gradients at restore')
    is_ridge = item['family'] in ('ridge_p', 'ridge_raw')
    if is_ridge and (model.parent.training or any(p.requires_grad for p in model.parent.parameters())):
        raise ValueError('The online ridge parent must stay frozen/eval')
    info = {'checkpoint': str(checkpoint), 'checkpoint_sha256': item['checkpoint_sha256'],
            'source': str(source), 'source_sha256': digest(source),
            'parameters_requiring_gradient_at_restore': trainable,
            'newly_fitted_coefficients_this_round': 24576 if is_ridge else 0,
            'historically_fitted_adapter_parameters': 42496 if is_ridge else trainable,
            'total_parameters_before_merge': sum(p.numel() for p in model.parameters()),
            'online_parent_forward_included': is_ridge,
            'parameter_count_note': 'Ridge uses closed-form fitting; requires_grad=False does not mean zero fitted coefficients.'}
    return model, info


def parity(items, prepared, legacy, job, output):
    rows = []
    for item in items:
        job.check_limits()
        legacy.cleanup()
        model, info = restore(item, legacy)
        replay = legacy.predict(model, prepared['inputs'], 4, job=job)
        with np.load(checked(item['val_prediction']), allow_pickle=False) as saved:
            if not np.array_equal(saved['origins'], prepared['origins']):
                raise ValueError('Stored and live whole-VAL origins differ')
            reference = torch.from_numpy(saved['prediction'].copy())
        check = legacy.difference(replay, reference, atol=1e-5, rtol=1e-4)
        score = legacy.score_arrays(replay.numpy(), prepared['target'], prepared['mask'])
        reference_score = legacy.score_arrays(reference.numpy(), prepared['target'], prepared['mask'])
        score_reconstruction_difference = reference_score['mse'] - item['val_mse']
        if abs(score_reconstruction_difference) > 1e-8:
            raise ValueError('Stored VAL score does not match its selected prediction')
        row = {'id': item['id'], 'restore_full_val': check, 'live_val_mse': score['mse'],
               'stored_val_mse': item['val_mse'], 'live_minus_stored_mse': score['mse'] - item['val_mse'],
               'stored_prediction_score_difference': score_reconstruction_difference}
        if not check['pass_']:
            save_json(output, {'status': 'failed', 'rows': rows + [row]})
            raise ValueError('Online restored model differs from stored VAL prediction; do not increase tolerance')
        del replay, reference
        if item['family'] == 'lora':
            info['merge'] = legacy.merge(model, prepared['selected_inputs'], job)
        baseline = legacy.predict(model, prepared['selected_inputs'], 4, job=job)
        actual = legacy.predict(model, prepared['selected_inputs'], 1, job=job)
        row.update(batch1_vs_batch4=legacy.difference(actual, baseline), model=info)
        rows.append(row)
        save_json(output, {'status': 'running', 'rows': rows})
        if not row['batch1_vs_batch4']['pass_']:
            raise ValueError('Same-function B1/B4 parity failed')
        del actual, baseline, model
        legacy.cleanup()
    save_json(output, {'status': 'PASS', 'rows': rows, 'val_origins': prepared['selected_origins'].tolist(),
                      'full_val_origins': prepared['origins'].tolist(),
                      'tolerance': {'atol': 1e-5, 'rtol': 1e-4},
                      'scope': 'Stored whole-VAL replay, LoRA merge and B1/B4 output agreement; no TEST inference'})


def summarize(rows, legacy, dataset):
    grouped = defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            grouped[(row['id'], row['batch'])].append(row)
    instances = []
    for (identity, batch), values in grouped.items():
        if len(values) != 3 or {row['block'] for row in values} != {0, 1, 2}:
            raise ValueError('Each instance/batch needs three complete timing blocks')
        instances.append({'id': identity, 'role': values[0]['role'], 'seed': values[0]['seed'], 'batch': batch,
                          'median_block_median_ms': float(np.median([row['milliseconds_per_origin']['median'] for row in values])),
                          'median_block_median_throughput': float(np.median([row['origins_per_second']['median'] for row in values])),
                          'block_ms': [row['milliseconds_per_origin']['median'] for row in values],
                          'block_throughput': [row['origins_per_second']['median'] for row in values],
                          'peak_allocated_bytes': [row['peak_allocated_bytes'] for row in values],
                          'peak_reserved_bytes': [row['peak_reserved_bytes'] for row in values],
                          'model': values[0]['model']})
    groups = []
    for role in ROLES:
        for batch in (1, 4):
            values = [row for row in instances if row['role'] == role and row['batch'] == batch]
            if len(values) != (1 if role == 'f0' else 2):
                raise ValueError('Incomplete role/seed aggregation')
            groups.append({'role': role, 'batch': batch, 'device': 'cuda',
                           'instance_ids': [row['id'] for row in values],
                           'mean_seed_median_ms': float(np.mean([row['median_block_median_ms'] for row in values])),
                           'mean_seed_median_throughput': float(np.mean([row['median_block_median_throughput'] for row in values])),
                           'mean_seed_median_peak_allocated_bytes': float(np.mean([np.median(row['peak_allocated_bytes']) for row in values])),
                           'mean_seed_median_peak_reserved_bytes': float(np.mean([np.median(row['peak_reserved_bytes']) for row in values])),
                           'observed_peak_allocated_range_bytes': [min(min(row['peak_allocated_bytes']) for row in values),
                                                                  max(max(row['peak_allocated_bytes']) for row in values)],
                           'observed_peak_reserved_range_bytes': [min(min(row['peak_reserved_bytes']) for row in values),
                                                                 max(max(row['peak_reserved_bytes']) for row in values)]})
    sentinels = []
    for block in range(3):
        for batch in (1, 4):
            pair = {row['sentinel']: row for row in rows if row['sentinel'] and row['block'] == block and row['batch'] == batch}
            if set(pair) != {'pre', 'post'}:
                raise ValueError('Missing F0 drift sentinel')
            pre, post = (pair[key]['milliseconds_per_origin']['median'] for key in ('pre', 'post'))
            sentinels.append({'block': block, 'batch': batch, 'pre_ms': pre, 'post_ms': post,
                              'post_relative_change': post / pre - 1})
    return {'status': 'complete', 'dataset': dataset, 'instances': instances, 'groups': groups,
            'sentinels': sentinels, 'rows': len(rows), 'primary_configurations': 22,
            'aggregation': 'Median of three block medians per model instance, then mean of seed centers; F0 has one deterministic instance.',
            'scope': 'Same-session FP32 GPU online deployment from standardized CPU input to complete CPU output. No previous-session timing ratio; no CPU timing or profile.'}


def benchmark(items, prepared, legacy, job, partial):
    rows = []
    inputs = prepared['selected_inputs']
    f0 = next(item for item in items if item['role'] == 'f0')

    def run(item, block, sentinel=None):
        job.check_limits()
        legacy.cleanup()
        model, info = restore(item, legacy)
        if item['family'] == 'lora':
            info['merge'] = legacy.merge(model, inputs, job)
        info.update(legacy.model_bytes(model))
        batches = (4, 1) if block == 1 and sentinel is None else (1, 4)
        for batch in batches:
            job.heartbeat(f'cost {item["dataset"]} block{block} {item["id"]} B{batch} {sentinel}')
            row = {'id': item['id'], 'role': item['role'], 'family': item['family'], 'seed': item['seed'],
                   'dataset': item['dataset'], 'block': block, 'batch': batch, 'device': 'cuda',
                   'sentinel': sentinel, 'sequence': len(rows), 'model': info,
                   'telemetry_before': legacy.telemetry()}
            row.update(legacy.measure(model, inputs, batch, 'original', None, job))
            row['telemetry_after'] = legacy.telemetry()
            rows.append(row)
            save_json(partial, {'status': 'running', 'rows': rows})
        del model
        legacy.cleanup()

    for block in range(3):
        run(f0, block, 'pre')
        order = items if block == 0 else items[::-1] if block == 1 else items[2:] + items[:2]
        for item in order:
            run(item, block)
        run(f0, block, 'post')
    if len(rows) != 78:
        raise ValueError('Expected 66 primary rows and twelve sentinel rows')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('robin', 'jena'), required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=600)
    args = parser.parse_args()
    if not args.job or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.job):
        parser.error('Use a unique simple ledger label')
    if not 0 < args.reserve_seconds <= 600:
        parser.error('Conditional cost reservation must fit inside the shared 600-second cost cap')
    decision = decision_guard(args.dataset)
    prefix = HERE / f'cost_{args.dataset}_{args.job}'
    paths = {key: Path(str(prefix) + '_' + key + '.json') for key in ('parity', 'partial', 'rows', 'summary', 'failure')}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError('Preserve prior cost attempts; use a new ledger label for a documented retry')
    with Job(args.job, category='gpu', reserve_seconds=args.reserve_seconds,
             metadata={'purpose': 'cost', 'dataset': args.dataset, 'fit_count': 0}) as job:
        global np, torch, psutil
        import numpy as np
        import torch
        import psutil
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        legacy = helpers()
        try:
            items = candidates(args.dataset, legacy)
            prepared = prepare(args.dataset)
            initial_binding = binding(items, prepared, decision)
            with torch.no_grad():
                parity(items, prepared, legacy, job, paths['parity'])
                rows = benchmark(items, prepared, legacy, job, paths['partial'])
            if binding(items, prepared, decision) != initial_binding:
                raise RuntimeError('Bound model/state/data/selection changed while measuring cost')
            save_json(paths['rows'], {'status': 'complete', 'dataset': args.dataset, 'rows': rows,
                      'binding': initial_binding, 'environment': legacy.cost_environment(),
                      'val_origins': prepared['selected_origins'].tolist(),
                      'parity_sha256': digest(paths['parity']),
                      'contract': {'warmup': 10, 'passes_per_block': 20, 'blocks': 3, 'batch': [1, 4],
                                   'origin_count': 24, 'primary_configurations': 22, 'rows_with_sentinels': 78,
                                   'precision': 'FP32', 'TF32': False, 'CPU_threads': 4,
                                   'range': 'CPU standardized input, H2D, online full forward, full CPU output; load/disk/scoring excluded'}})
            summary = summarize(rows, legacy, args.dataset)
            summary.update(rows_receipt={'path': str(paths['rows']), 'sha256': digest(paths['rows'])},
                           parity_receipt={'path': str(paths['parity']), 'sha256': digest(paths['parity'])})
            save_json(paths['summary'], summary)
        except BaseException:
            save_json(paths['failure'], {'status': 'failed', 'dataset': args.dataset,
                      'error': traceback.format_exc(), 'policy': 'Keep all partial rows and the failed job; no tolerance relaxation or favorable-block deletion.'})
            raise


if __name__ == '__main__':
    main()
