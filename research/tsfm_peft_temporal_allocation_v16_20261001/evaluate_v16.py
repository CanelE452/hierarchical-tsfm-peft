"""Joint-sealed temporal allocation forecasts and matched exposed-data scores."""
import argparse
import csv
import gc
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from runtime_v16 import (HERE, ROOT, CACHE, DATASETS, SEEDS, Job, artifact, configure,
                         digest, import_file, load_data, protocol, read_json,
                         resolve, require_storage, save_json, tensors)

NEW = ('temporal_lora',)
CONTROLS = ('temporal_f0', 'coarse_only')
REFERENCES = ('direct', 'a', 'f0', 'full_mse', 'native')
ROLES = NEW + CONTROLS + REFERENCES
PERIODS = ('test_a', 'test_b', 'combined')
V12 = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001'


@lru_cache(maxsize=1)
def previous_api():
    if str(V12) not in sys.path:
        sys.path.append(str(V12))
    return import_file(V12 / 'evaluate_confirmation_v12.py', 'v16_verified_metric_primitives')


def prediction_array(receipt, origins, shape):
    record = artifact(receipt['path'], receipt['sha256'])
    with np.load(resolve(record['path']), allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], origins):
            raise ValueError('Prediction origin order changed')
        value = archive['prediction'].copy()
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Prediction shape or finite-value contract changed')
    return value


def validate_rows(rows):
    if len(rows) != 15 or len({row['id'] for row in rows}) != 15:
        raise ValueError('All fifteen prescribed instances must be retained')
    for family in ROLES:
        expected = [None] if family == 'f0' else list(SEEDS)
        actual = [row['seed'] for row in rows if row['family'] == family]
        if sorted(actual, key=str) != sorted(expected, key=str):
            raise ValueError('Missing fixed family or seed: ' + family)


@lru_cache(maxsize=1)
def selected():
    from train_v16 import fit_specs, result_for
    completed = [result_for(spec) for spec in fit_specs()]
    results = {row['id']: row for row in completed}
    value = read_json(HERE / 'selected.json')
    if len(results) != 16 or set(value['selected']) != set(DATASETS) or value['test_used_for_selection']:
        raise ValueError('All sixteen fits and four selections must finish')
    for ds in DATASETS:
        row = value['selected'][ds]['temporal_lora']
        candidates = []
        for lr in (1e-4, 1e-3):
            members = sorted([r for r in completed if r['spec']['dataset'] == ds and r['spec']['lr'] == lr],
                             key=lambda r: r['spec']['seed'])
            if [r['spec']['seed'] for r in members] != list(SEEDS):
                raise ValueError('A LR lacks its paired seed')
            candidates.append((float(np.mean([r['best_val_mse'] for r in members])), lr,
                               [r['id'] for r in members]))
        best = min(candidates, key=lambda r: (r[0], r[1]))
        if (row['lr'] != best[1] or row['mean_best_val_mse'] != best[0]
                or row['run_ids'] != best[2]):
            raise ValueError('Stored LR differs from full paired VAL choice')
        if row['seeds'] != list(SEEDS) or len(row['checkpoints']) != 2:
            raise ValueError('Both prescribed seeds must be selected')
        for index, seed in enumerate(SEEDS):
            run = results[row['run_ids'][index]]
            if run['spec']['dataset'] != ds or run['spec']['seed'] != seed or run['spec']['lr'] != row['lr']:
                raise ValueError('Selection dataset/seed/common LR mismatch')
            if (row['checkpoints'][index] != run['checkpoint']
                    or row['checkpoint_sha256'][index] != run['checkpoint_sha256']
                    or row['initial_checkpoints'][index] != run['initial_checkpoint']
                    or row['selected_epochs'][index] != run['selected_epoch']):
                raise ValueError('Selected state differs from its completed source fit')
            artifact(row['checkpoints'][index], row['checkpoint_sha256'][index])
    return value


def evaluation_rows(selection, dataset):
    rows = []
    item = selection['selected'][dataset]['temporal_lora']
    for family in NEW + CONTROLS:
        for i, seed in enumerate(SEEDS):
            checkpoint = (item['initial_checkpoints'][i] if family == 'temporal_f0' else
                          {'path': item['checkpoints'][i], 'sha256': item['checkpoint_sha256'][i]})
            rows.append({'id': f'v16_{dataset}_{family}_s{seed}', 'dataset': dataset,
                         'family': family, 'seed': seed, 'run_id': item['run_ids'][i],
                         'checkpoint': checkpoint,
                         'reused': False, 'status': 'planned'})
    for original in read_json(HERE / 'reuse_manifest.json')['datasets'][dataset]['references']:
        row = {name: original[name] for name in ('id', 'family', 'seed', 'checkpoint', 'prediction',
               'source_evaluation', 'source_model_id', 'source_generation', 'source_row') if name in original}
        row.update(dataset=dataset, reused=True, status='planned')
        rows.append(row)
    validate_rows(rows)
    return rows


def seal(job):
    if (HERE / 'selection_seal.json').exists() or (HERE / 'test_exposure.json').exists():
        raise FileExistsError('Preserve selection and first-exposure history')
    selection = selected()
    if any(r['status'] == 'running' and r['category'] in ('neural_fit', 'gpu', 'cpu_analysis')
           for r in read_json(HERE / 'ledger.json')['jobs']):
        raise ValueError('Fitting jobs must terminate before the joint seal')
    files = ('PLAN.md', 'AUTHORIZATION.txt', 'protocol.json', 'provenance.json', 'selected.json',
             'reuse_manifest.json', 'data_manifest.json', 'initial_manifest.json',
             'model_v16.py', 'train_v16.py', 'runtime_v16.py', 'evaluate_v16.py')
    result = {'status': 'sealed', 'created_utc': time.time(),
              'datasets': {ds: {'rows': evaluation_rows(selection, ds)} for ds in DATASETS},
              'files': {name: digest(HERE / name) for name in files},
              'all_fits_selected_before_new_evaluation': True, 'independent_confirmation': False,
              'scope': 'All four datasets are already exposed development'}
    save_json(HERE / 'selection_seal.json', result)
    job.heartbeat('All sixteen fits and four selections jointly sealed')
    return result


def ensure_exposure():
    value = read_json(HERE / 'selection_seal.json')
    if set(value['datasets']) != set(DATASETS):
        raise ValueError('All four dataset selections required')
    for name, sha in value['files'].items():
        artifact(HERE / name, sha)
    path = HERE / 'test_exposure.json'
    seal_sha = digest(HERE / 'selection_seal.json')
    if path.exists():
        if read_json(path)['seal_sha256'] != seal_sha:
            raise ValueError('TEST exposure belongs to another seal')
    else:
        save_json(path, {'seal_sha256': seal_sha, 'first_new_evaluation_utc': time.time(),
                         'datasets': list(DATASETS), 'previously_exposed_development_data': True})
    return value


def predict(job):
    import torch
    from model_v16 import build_model, restore_model
    selection = ensure_exposure()
    destination = HERE / 'prediction_manifest.json'
    if destination.exists():
        raise FileExistsError('Reuse existing immutable forecasts')
    folder = CACHE / 'evaluation_predictions'
    folder.mkdir(parents=True, exist_ok=True)
    output = {'status': 'partial', 'seal': artifact(HERE / 'selection_seal.json'),
              'model_runs': 0, 'new_neural_fits': 0, 'datasets': {}}
    with torch.no_grad():
        for ds, unit in selection['datasets'].items():
            data = load_data(ds, include_test=True)
            origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
            shape = (len(origins), 48, DATASETS[ds][0])
            records = []
            for row in unit['rows']:
                record = dict(row)
                if row['reused']:
                    prediction_array(row['prediction'], origins, shape)
                    record['status'] = 'complete'
                else:
                    path = folder / (row['id'] + '.npz')
                    receipt_path = folder / (row['id'] + '.json')
                    if path.exists() or receipt_path.exists():
                        if not path.exists() or not receipt_path.exists():
                            raise ValueError('Preserve incomplete prediction files for explicit repair')
                        record = read_json(receipt_path)
                        if record['seal_sha256'] != output['seal']['sha256'] or record['id'] != row['id']:
                            raise ValueError('Partial forecast belongs to another sealed model')
                        prediction_array(record['prediction'], origins, shape)
                    else:
                        if row['family'] == 'temporal_f0':
                            artifact(row['checkpoint']['path'], row['checkpoint']['sha256'])
                            model = build_model(ds, row['seed'], device='cuda', adapt=False)
                        else:
                            artifact(row['checkpoint']['path'], row['checkpoint']['sha256'])
                            model = restore_model(row['checkpoint']['path'], device='cuda')
                        model.eval()
                        parts = []
                        for start in range(0, len(origins), 4):
                            job.check_limits()
                            x, _, _ = tensors(data, origins[start:start + 4])
                            y = model.components(x)[0] if row['family'] == 'coarse_only' else model(x)
                            parts.append(y.cpu().numpy())
                            output['model_runs'] += 1
                        prediction = np.concatenate(parts)
                        if prediction.shape != shape or not np.isfinite(prediction).all():
                            raise ValueError('Invalid full-channel prediction')
                        require_storage(prediction.nbytes + origins.nbytes)
                        np.savez_compressed(path, origins=origins, prediction=prediction)
                        record.update(status='complete', prediction=artifact(path),
                                      seal_sha256=output['seal']['sha256'])
                        save_json(receipt_path, record)
                        del model, x, y, parts, prediction
                        gc.collect()
                        torch.cuda.empty_cache()
                records.append(record)
                output['datasets'][ds] = {'models': records, 'shape': list(shape)}
                save_json(HERE / 'prediction_manifest_partial.json', output)
                job.heartbeat(ds + ': predicted/reused ' + row['id'])
            validate_rows(records)
    output.update(status='complete', prescribed_instances=60)
    save_json(destination, output)
    return output


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=None)
def source_evaluation(path, expected):
    artifact(path, expected)
    return read_json(resolve(path))


def comparison(role_terms, indices, block, unit_index, job):
    api = previous_api()
    weights = api.bootstrap_weights(indices, block, unit_index)
    draws = {}
    common_count = role_terms['f0']['count']
    if any(not np.array_equal(item['count'], common_count) for item in role_terms.values()):
        raise ValueError('Paired comparison must use identical original masks')
    for period, weight in weights.items():
        denominator = weight @ common_count
        if np.any(denominator <= 0):
            raise ValueError('A bootstrap draw lacks an observed target channel')
        draws[period] = {}
        for family, terms in role_terms.items():
            job.check_limits()
            draws[period][family] = {
                metric: np.mean((weight @ terms[key]) / denominator, axis=1)
                for metric, key in (('mse', 'squared'), ('mae', 'absolute'))}
    pairs = [('temporal_lora', name) for name in CONTROLS + REFERENCES]
    values = {}
    for left, right in pairs:
        item = {'candidate': left, 'reference': right,
                'direction': 'candidate minus reference; negative favors candidate',
                'periods': {}}
        if left not in role_terms or right not in role_terms:
            raise ValueError('Every prescribed comparison must have both roles')
        item['status'] = 'available'
        for period, index in indices.items():
            a, b = (api.aggregate_terms(role_terms[family], index) for family in (left, right))
            item['periods'][period] = {}
            for metric in ('mse', 'mae'):
                delta = a[metric] - b[metric]
                da, db = draws[period][left][metric], draws[period][right][metric]
                item['periods'][period][metric] = {
                    'candidate': a[metric], 'reference': b[metric], 'absolute_difference': delta,
                    'relative_change_percent': 100 * delta / b[metric] if b[metric] > 0 else None,
                    'conditional_block95_absolute': np.quantile(da - db, [.025, .975]).tolist(),
                    'conditional_block95_relative_percent':
                        np.quantile(100 * (da - db) / db, [.025, .975]).tolist() if np.all(db > 0) else None}
        values[left + '__vs__' + right] = item
    return values


def score(job):
    ensure_exposure()
    api = previous_api()
    destination = HERE / 'evaluation01.json'
    if destination.exists():
        raise FileExistsError('Preserve original scores; corrections require a separate receipt')
    manifest = read_json(HERE / 'prediction_manifest.json')
    if manifest['status'] != 'complete' or set(manifest['datasets']) != set(DATASETS):
        raise ValueError('All prescribed prediction instances must be accounted for')
    artifact(manifest['seal']['path'], manifest['seal']['sha256'])
    output = {'status': 'partial', 'datasets': {}, 'prediction_manifest': artifact(HERE / 'prediction_manifest.json'),
              'selection_seal': artifact(HERE / 'selection_seal.json'), 'independent_confirmation': False,
              'scope': 'Four exposed development datasets; conditional comparisons, no new confirmation',
              'model_runs': 0, 'new_neural_fits': 0, 'optimizer_updates': 0}
    scalar_rows, channel_rows = [], []
    for unit_index, dataset in enumerate(DATASETS):
        data = load_data(dataset, include_test=True)
        a, b = data['test_a_origins'], data['test_b_origins']
        origins = np.concatenate([a, b]).astype(np.int64)
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        indices = {'test_a': np.arange(len(a)), 'test_b': np.arange(len(a), len(origins)),
                   'combined': np.arange(len(origins))}
        terms, models, role_terms, summaries = {}, {}, {}, {}
        rows = manifest['datasets'][dataset]['models']
        validate_rows(rows)
        for row in rows:
            if row['status'] == 'unavailable':
                raise ValueError('No silently unavailable v16 comparison role is permitted')
            prediction = prediction_array(row['prediction'], origins, target.shape)
            term = api.error_terms(prediction, target, mask)
            periods = {period: api.aggregate_terms(term, index) for period, index in indices.items()}
            replay = None
            if row['reused']:
                entry = row['source_evaluation']
                old = source_evaluation(entry['path'], entry['sha256'])['datasets'][dataset]['models'][row['source_model_id']]
                for period in PERIODS:
                    for metric in ('mse', 'mae', 'signed_mean_error'):
                        if not np.isclose(periods[period][metric], old['periods'][period][metric], atol=1e-10, rtol=1e-9):
                            raise ValueError('Reference score replay differs: ' + row['id'])
                replay = {'passed': True, 'atol': 1e-10, 'rtol': 1e-9, 'source': entry}
            models[row['id']] = {**row, 'periods': periods, 'reference_score_replay': replay}
            terms[row['id']] = term
            for period, value in periods.items():
                scalar_rows.append({'dataset': dataset, 'family': row['family'], 'id': row['id'], 'seed': row['seed'],
                    'period': period, 'mse': value['mse'], 'mae': value['mae'], 'signed_mean_error': value['signed_mean_error'],
                    'reused': row['reused']})
                for c, name in enumerate(data['columns']):
                    channel_rows.append({'dataset': dataset, 'family': row['family'], 'id': row['id'], 'seed': row['seed'],
                        'period': period, 'channel': str(name), 'mse': value['channel_mse'][c], 'mae': value['channel_mae'][c],
                        'signed_mean_error': value['channel_bias'][c], 'target_count': value['target_counts'][c]})
        for family in ROLES:
            ids = [identifier for identifier, row in models.items() if row['family'] == family]
            if any(models[identifier]['status'] == 'unavailable' for identifier in ids):
                raise ValueError('All prescribed seed forecasts are required')
            count = terms[ids[0]]['count']
            if any(not np.array_equal(terms[identifier]['count'], count) for identifier in ids):
                raise ValueError('Seed masks differ')
            term = {key: np.mean([terms[identifier][key] for identifier in ids], axis=0)
                    for key in ('squared', 'absolute', 'signed')}
            term['count'] = count
            role_terms[family] = term
            summaries[family] = {'status': 'available', 'ids': ids, 'aggregation': 'Mean seed losses, not ensemble',
                                 'periods': {period: api.aggregate_terms(term, index) for period, index in indices.items()}}
        term_path = CACHE / (dataset + '_origin_terms01.npz')
        if term_path.exists():
            raise FileExistsError('Preserve partial origin terms and explicitly repair interrupted scoring')
        require_storage()
        np.savez_compressed(term_path, origins=origins,
                            **{identifier + '__' + key: value for identifier, item in terms.items() for key, value in item.items()})
        block = int(read_json(HERE / 'data_manifest.json')['datasets'][dataset]['block_origins'])
        output['datasets'][dataset] = {'models': models, 'role_summary': summaries,
            'comparisons': comparison(role_terms, indices, block, unit_index, job),
            'origin_terms': artifact(term_path), 'data': artifact(data['_path']),
            'columns': [str(v) for v in data['columns']],
            'period_origin_counts': {period: len(index) for period, index in indices.items()},
            'bootstrap': {'draws': 2000, 'seed': 9262026, 'block_origins': block,
                          'period_boundaries_crossed': False, 'paired_channels_methods_and_seeds': True,
                          'selection_uncertainty_included': False},
            'aggregation': 'Pool errors/counts by channel across periods, channel macro, then mean seed losses'}
        save_json(HERE / 'evaluation01_partial.json', output)
        job.heartbeat(dataset + ': all prescribed predictions accounted for')
    write_csv(HERE / 'evaluation01_models.csv', scalar_rows)
    write_csv(HERE / 'evaluation01_channels.csv', channel_rows)
    output.update(status='complete', prescribed_instances=60,
                  scored_instances=sum(row['status'] == 'complete' for unit in output['datasets'].values() for row in unit['models'].values()),
                  tables={name: artifact(HERE / f'evaluation01_{name}.csv') for name in ('models', 'channels')})
    save_json(destination, output)
    return output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'predict', 'score'))
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    category = {'seal': 'cpu_check', 'predict': 'gpu', 'score': 'cpu_analysis'}[args.action]
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'v16 ' + args.action, 'fit_count': 0}) as job:
        configure()
        {'seal': seal, 'predict': predict, 'score': score}[args.action](job)


if __name__ == '__main__':
    main()
