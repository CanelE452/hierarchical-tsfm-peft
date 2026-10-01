"""Joint sealed comparison of coordinate/PCA decoders and fixed references."""
import argparse
import csv
import time
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

from runtime_v15 import (HERE, ROOT, CACHE, DATASETS, SEEDS, Job, artifact,
                         digest, import_file, load_data, protocol, read_json,
                         resolve, require_storage, save_json)

NEW = ('raw_free', 'pca_free')
CONTROLS = ('raw_tied', 'raw_interp', 'pca_tied')
REFERENCES = ('direct', 'pca_level', 'a', 'f0', 'full_mse', 'native')
ROLES = NEW + CONTROLS + REFERENCES
PERIODS = ('test_a', 'test_b', 'combined')
LAMBDAS = (0., .001, .1)
V12 = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001'


@lru_cache(maxsize=1)
def previous_api():
    if str(V12) not in sys.path:
        sys.path.append(str(V12))
    return import_file(V12 / 'evaluate_confirmation_v12.py', 'v15_verified_metric_primitives')


def prediction_array(receipt, origins, shape):
    record = artifact(receipt['path'], receipt['sha256'])
    with np.load(resolve(record['path']), allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], origins):
            raise ValueError('Original prediction origin order changed')
        value = archive['prediction'].copy()
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Prediction shape or finite-value contract changed')
    return value


def validate_rows(rows):
    if len(rows) != 21 or len({row['id'] for row in rows}) != 21:
        raise ValueError('All twenty-one prescribed instances must be retained')
    for family in ROLES:
        expected = [None] if family == 'f0' else list(SEEDS)
        actual = [row['seed'] for row in rows if row['family'] == family]
        if sorted(actual, key=str) != sorted(expected, key=str):
            raise ValueError('Missing fixed family or seed: ' + family)


def trainval_cache_binding():
    value = read_json(HERE / 'cache_manifest.json')
    return {dataset: {kind: {split: value['datasets'][dataset][kind][split]
                            for split in ('train', 'val')} for kind in ('raw', 'pca')}
            for dataset in DATASETS}


@lru_cache(maxsize=1)
def selected():
    from fit_v15 import fit_specs, result_for
    specs = fit_specs()
    results = {spec['id']: result_for(spec) for spec in specs}
    value = read_json(HERE / 'selected.json')
    if (len(results) != 48 or value['status'] != 'complete' or value['test_used'] is not False
            or set(value['datasets']) != set(DATASETS)):
        raise ValueError('All forty-eight prescribed coefficient fits must finish before selection is sealed')
    normalized = {'selected': {}, 'source': artifact(HERE / 'selected.json'),
                  'test_used_for_selection': False, 'all_basic_fits': len(results)}
    for dataset in DATASETS:
        if set(value['datasets'][dataset]) != set(NEW):
            raise ValueError('Selected decoder families changed')
        normalized['selected'][dataset] = {}
        for family in NEW:
            original = value['datasets'][dataset][family]
            candidates = original['candidates']
            if len(candidates) != 4 or set(original['seeds']) != {str(seed) for seed in SEEDS}:
                raise ValueError('Every lambda and parent candidate requires both seeds')
            expected = []
            parent_scores = {}
            members = {}
            for penalty in LAMBDAS:
                rows = sorted([row for row in results.values() if row['dataset'] == dataset
                               and row['family'] == family and row['lambda'] == penalty], key=lambda row: row['seed'])
                if [row['seed'] for row in rows] != list(SEEDS):
                    raise ValueError('Missing canonical lambda/seed fit')
                members[penalty] = rows
                for row in rows:
                    parent = row['parent_val_score']['mse']
                    if row['seed'] in parent_scores and parent_scores[row['seed']] != parent:
                        raise ValueError('Frozen direct parent VAL score changed across penalties')
                    parent_scores[row['seed']] = parent
                expected.append({'candidate': 'lambda', 'lambda': penalty,
                    'mean': float(np.mean([row['val_score']['mse'] for row in rows])),
                    'per_seed': {str(row['seed']): row['val_score']['mse'] for row in rows}})
            expected.insert(0, {'candidate': 'parent', 'lambda': None,
                'mean': float(np.mean([parent_scores[seed] for seed in SEEDS])),
                'per_seed': {str(seed): parent_scores[seed] for seed in SEEDS}})
            for index, (saved, actual) in enumerate(zip(candidates, expected)):
                if (saved['candidate'] != actual['candidate'] or saved['lambda'] != actual['lambda']
                        or saved['tie_order'] != index or saved['mean_seed_val_mse'] != actual['mean']
                        or saved['per_seed_val_mse'] != actual['per_seed']):
                    raise ValueError('Stored candidate scores differ from all completed VAL fits')
            best = min(range(4), key=lambda i: (expected[i]['mean'], i))
            chosen = expected[best]
            if (original['selected_candidate'] != chosen['candidate'] or original['selected_lambda'] != chosen['lambda']
                    or original['selected_mean_seed_val_mse'] != chosen['mean']):
                raise ValueError('Selection violates common mean-seed VAL or exact parent-first tie order')
            weights, run_ids = [], []
            for index, seed in enumerate(SEEDS):
                row = original['seeds'][str(seed)]
                receipt = row['weights']
                if receipt is None:
                    raise ValueError('Parent fallback must preserve an explicit zero FP32 decoder artifact')
                artifact(receipt['path'], receipt['sha256'])
                if best == 0:
                    if row['status'] != 'parent_fallback' or row['run_id'] is not None:
                        raise ValueError('Zero decoder is a no-fit parent reference')
                    with np.load(resolve(receipt['path']), allow_pickle=False) as arrays:
                        if np.count_nonzero(arrays['W']) != 0 or arrays['W'].dtype != np.float32:
                            raise ValueError('Fallback must be exact zero FP32 coefficients')
                else:
                    fit = members[chosen['lambda']][index]
                    if (row['run_id'] != fit['fit_id'] or receipt != fit['artifacts']['weights']
                            or row['attempt_label'] != fit['attempt_label']):
                        raise ValueError('Selected exported coefficients differ from the completed actual fit')
                weights.append(receipt)
                run_ids.append(row['run_id'])
            normalized['selected'][dataset][family] = {
                'candidate': 'parent' if best == 0 else chosen['lambda'], 'lambda': chosen['lambda'],
                'seeds': list(SEEDS), 'run_ids': run_ids, 'weights': weights,
                'mean_val_mse': chosen['mean'], 'per_seed_val_mse': chosen['per_seed'],
                'parent_selected': best == 0, 'all_candidates': candidates}
    return normalized


def ensure_exposure():
    value = read_json(HERE / 'selection_seal.json')
    if set(value['datasets']) != set(DATASETS):
        raise ValueError('Selection seal must contain all four exposed datasets')
    for name, sha in value['files'].items():
        artifact(HERE / name, sha)
    if value['trainval_cache'] != trainval_cache_binding():
        raise ValueError('TRAIN/VAL caches changed after selection')
    path = HERE / 'test_exposure.json'
    seal_sha = digest(HERE / 'selection_seal.json')
    if path.exists():
        if read_json(path)['seal_sha256'] != seal_sha:
            raise ValueError('TEST exposure belongs to another selection seal')
    else:
        save_json(path, {'seal_sha256': seal_sha, 'first_new_evaluation_utc': time.time(),
                         'datasets': list(DATASETS), 'previously_exposed_development_data': True})
    return value


def evaluation_rows(selection, dataset):
    rows = []
    for family in NEW:
        item = selection['selected'][dataset][family]
        for i, seed in enumerate(SEEDS):
            rows.append({'id': f'v15_{dataset}_{family}_s{seed}', 'dataset': dataset,
                         'family': family, 'seed': seed, 'candidate': item['candidate'],
                         'run_id': item['run_ids'][i], 'weights': item['weights'][i],
                         'checkpoint': item['weights'][i],
                         'reused': False, 'status': 'planned'})
    selector = read_json(HERE / 'selector_manifest.json')['datasets'][dataset]
    for family in CONTROLS:
        for seed in SEEDS:
            row = {'id': f'v15_{dataset}_{family}_s{seed}', 'dataset': dataset,
                   'family': family, 'seed': seed, 'reused': False, 'status': 'planned'}
            if family == 'raw_interp' and selector['W_interp'] is None:
                if selector['interpolation_rank'] >= selector['K']:
                    raise ValueError('Unavailable interpolation requires recorded rank deficiency')
                row.update(status='unavailable', unavailable_reason='TRAIN PCA pivot interpolation matrix is rank deficient',
                           interpolation_rank=selector['interpolation_rank'], required_rank=selector['K'])
            rows.append(row)
    reuse = read_json(HERE / 'reuse_manifest.json')
    for original in reuse['datasets'][dataset]['references']:
        row = {name: original[name] for name in ('id', 'family', 'seed', 'checkpoint', 'prediction',
               'source_evaluation', 'source_model_id', 'source_generation', 'source_row') if name in original}
        row.update(dataset=dataset, reused=True, status='planned')
        rows.append(row)
    validate_rows(rows)
    return rows


def seal(job):
    destination = HERE / 'selection_seal.json'
    if destination.exists() or (HERE / 'test_exposure.json').exists():
        raise FileExistsError('Preserve selection and first-exposure history')
    selection = selected()
    files = ('PLAN.md', 'AUTHORIZATION.txt', 'protocol.json', 'provenance.json', 'selected.json',
             'reuse_manifest.json', 'data_manifest.json', 'selector_manifest.json', 'direct_cache_refs.json',
             'model_v15.py', 'fit_v15.py', 'runtime_v15.py', 'cache_v15.py', 'evaluate_v15.py')
    ledger = read_json(HERE / 'ledger.json')['jobs']
    if any(row['status'] == 'running' and row['category'] in ('coefficient_fit', 'gpu', 'cpu_analysis') for row in ledger):
        raise ValueError('Cache and fitting jobs must terminate before the joint seal')
    value = {'status': 'sealed', 'created_utc': time.time(),
             'datasets': {dataset: {'rows': evaluation_rows(selection, dataset)} for dataset in DATASETS},
             'files': {name: digest(HERE / name) for name in files},
             'trainval_cache': trainval_cache_binding(),
             'all_fits_selected_before_new_evaluation': True, 'independent_confirmation': False,
             'scope': 'All four datasets are exposed development; no fresh confirmation'}
    save_json(destination, value)
    job.heartbeat('All forty-eight coefficient fits and four dataset selections sealed together')
    return value


def decoder_weights(row, selector):
    family = row['family']
    if family in NEW:
        receipt = row['weights']
        artifact(receipt['path'], receipt['sha256'])
        with np.load(resolve(receipt['path']), allow_pickle=False) as archive:
            weight = archive['W'].copy()
    elif family == 'raw_interp':
        weight = np.asarray(selector['W_interp'], dtype=np.float32)
    else:
        weight = np.asarray(selector['E_raw' if family == 'raw_tied' else 'E_pca'], dtype=np.float32).T.copy()
    if weight.shape != (selector['K'], selector['C']) or weight.dtype != np.float32 or not np.isfinite(weight).all():
        raise ValueError('Exported decoder must be finite FP32 K by C')
    return weight


def predict(job):
    selection = ensure_exposure()
    destination = HERE / 'prediction_manifest.json'
    if destination.exists():
        raise FileExistsError('Prediction manifest already exists; reuse its immutable forecasts')
    cache = read_json(HERE / 'cache_manifest.json')
    selectors = read_json(HERE / 'selector_manifest.json')['datasets']
    folder = CACHE / 'evaluation_predictions'
    folder.mkdir(parents=True, exist_ok=True)
    output = {'status': 'partial', 'seal': artifact(HERE / 'selection_seal.json'),
              'cache_manifest_at_prediction': artifact(HERE / 'cache_manifest.json'),
              'model_runs': 0, 'new_coefficient_fits': 0, 'datasets': {}}
    for dataset, unit in selection['datasets'].items():
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
        shape = (len(origins), 48, data['x'].shape[1])
        selector = selectors[dataset]
        latent = {kind: prediction_array(cache['datasets'][dataset][kind]['test'], origins,
                  (len(origins), 48, selector['K'])) for kind in ('raw', 'pca')}
        direct_rows = {row['seed']: row for row in unit['rows'] if row['family'] == 'direct'}
        direct = {seed: prediction_array(row['prediction'], origins, shape) for seed, row in direct_rows.items()}
        records = []
        for row in unit['rows']:
            job.check_limits()
            record = dict(row)
            if row['status'] == 'unavailable':
                records.append(record)
                continue
            if row['reused']:
                prediction_array(row['prediction'], origins, shape)
                record['status'] = 'complete'
            else:
                path = folder / (row['id'] + '.npz')
                receipt_path = folder / (row['id'] + '.json')
                if path.exists() or receipt_path.exists():
                    if not path.exists() or not receipt_path.exists():
                        raise ValueError('Preserve and explicitly repair incomplete prediction files')
                    prior = read_json(receipt_path)
                    if prior['seal_sha256'] != output['seal']['sha256'] or prior['id'] != row['id']:
                        raise ValueError('Partial prediction does not match this sealed choice')
                    prediction_array(prior['prediction'], origins, shape)
                    record = prior
                else:
                    kind = 'raw' if row['family'].startswith('raw') else 'pca'
                    encoder = np.asarray(selector['E_' + kind], dtype=np.float32)
                    weight = decoder_weights(row, selector)
                    parent = direct[row['seed']]
                    zero = bool(np.count_nonzero(weight) == 0)
                    prediction = parent.copy() if zero else parent + (latent[kind] - parent @ encoder) @ weight
                    if prediction.shape != shape or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                        raise ValueError('Invalid decoder forecast')
                    require_storage(prediction.nbytes + origins.nbytes)
                    np.savez_compressed(path, origins=origins, prediction=prediction)
                    record.update(status='complete', prediction=artifact(path), seal_sha256=output['seal']['sha256'],
                                  latent_prediction=cache['datasets'][dataset][kind]['test'],
                                  direct_prediction=direct_rows[row['seed']]['prediction'],
                                  zero_decoder=zero, direct_parent_alias=direct_rows[row['seed']]['id'] if zero else None,
                                  composition='FP32 S + (cached F0(XE) - S@E)@W; no model execution')
                    save_json(receipt_path, record)
            records.append(record)
            output['datasets'][dataset] = {'models': records, 'shape': list(shape)}
            save_json(HERE / 'prediction_manifest_partial.json', output)
            job.heartbeat(dataset + ': assembled/reused ' + row['id'])
        validate_rows(records)
        output['datasets'][dataset] = {'models': records, 'shape': list(shape)}
    output.update(status='complete', prescribed_instances=84)
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
    pairs = [('raw_free', name) for name in ('pca_free',) + CONTROLS + REFERENCES]
    pairs += [('pca_free', name) for name in ('pca_tied',) + REFERENCES]
    pairs += [('raw_tied', 'direct'), ('raw_interp', 'direct'), ('pca_tied', 'direct')]
    values = {}
    for left, right in pairs:
        item = {'candidate': left, 'reference': right,
                'direction': 'candidate minus reference; negative favors candidate',
                'periods': {}}
        if left not in role_terms or right not in role_terms:
            item.update(status='unavailable', reason='Fixed interpolation is numerically rank deficient')
            values[left + '__vs__' + right] = item
            continue
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
              'model_runs': 0, 'new_coefficient_fits': 0, 'optimizer_updates': 0}
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
                if row['family'] != 'raw_interp':
                    raise ValueError('Only the declared rank-deficient fixed control can be unavailable')
                models[row['id']] = {**row, 'periods': {}, 'reference_score_replay': None}
                continue
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
                if family != 'raw_interp' or not all(models[i]['status'] == 'unavailable' for i in ids):
                    raise ValueError('Unavailable control must be common to both seeds')
                summaries[family] = {'status': 'unavailable', 'ids': ids, 'periods': {},
                                     'reason': models[ids[0]]['unavailable_reason']}
                continue
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
        job.heartbeat(dataset + ': all available predictions and fixed unavailable controls accounted for')
    write_csv(HERE / 'evaluation01_models.csv', scalar_rows)
    write_csv(HERE / 'evaluation01_channels.csv', channel_rows)
    output.update(status='complete', prescribed_instances=84,
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
    category = 'cpu_check' if args.action == 'seal' else 'cpu_analysis'
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'v15 ' + args.action, 'fit_count': 0, 'model_runs': 0}) as job:
        {'seal': seal, 'predict': predict, 'score': score}[args.action](job)


if __name__ == '__main__':
    main()
