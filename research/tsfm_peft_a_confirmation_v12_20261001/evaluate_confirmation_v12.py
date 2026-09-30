"""Joint, preselected v12 confirmation: full-channel metrics and fixed-A Q references."""
import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from runtime_confirmation_v12 import (HERE, CACHE, SEEDS, Job, artifact, configure,
                                      digest, load_data, protocol, read_json, save_json, tensors)


FAMILIES = ('level', 'a', 'full_mse', 'native', 'direct', 'f0')
ROLES = FAMILIES + ('a_main', 'a_last')
PERIODS = ('test_a', 'test_b')
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 9262026
PARITY_ATOL, PARITY_RTOL = 1e-5, 1e-4


def parity(value, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL):
    value, reference = np.asarray(value), np.asarray(reference)
    if value.shape != reference.shape:
        raise ValueError('Parity shape mismatch')
    difference = np.abs(value.astype(np.float64) - reference.astype(np.float64))
    finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
    allowance = atol + rtol * np.abs(reference.astype(np.float64))
    return {'pass': finite and bool(np.all(difference <= allowance)), 'atol': atol, 'rtol': rtol,
            'max_abs': float(difference.max()),
            'max_relative_stabilized_1e_12': float(np.max(difference / np.maximum(np.abs(reference), 1e-12))),
            'outside_tolerance_elements': int(np.count_nonzero(difference > allowance)), 'finite': finite}


def load_complete_selection():
    from train_confirmation_v12 import result_for
    p = protocol()
    selection = read_json(HERE / 'selected_confirmation.json')
    completed = {}
    for spec in p['neural_fits']:
        result = result_for(spec)
        if result['id'] in completed:
            raise ValueError('Two canonical fit configurations resolve to the same attempt')
        completed[result['id']] = result
    if set(selection['selected']) != set(p['units']):
        raise ValueError('All approved confirmation units must be selected together')
    for dataset in p['units']:
        members = selection['selected'][dataset]
        if set(members) != set(FAMILIES) - {'f0'}:
            raise ValueError('Selected families differ from the prescribed confirmation comparison')
        for family in FAMILIES[:-1]:
            item = members[family]
            if len(item['run_ids']) != 2 or len(item['checkpoints']) != 2 or len(item['checkpoint_sha256']) != 2:
                raise ValueError('Each learned family needs both selected seeds')
            if item.get('seeds', list(SEEDS)) != list(SEEDS):
                raise ValueError('Selected seed order changed')
            for index, (path, sha) in enumerate(zip(item['checkpoints'], item['checkpoint_sha256'])):
                artifact(path, sha)
                result = completed[item['run_ids'][index]]
                if (result['spec']['dataset'] != dataset or result['spec']['family'] != family
                        or result['spec']['seed'] != SEEDS[index] or result['spec']['lr'] != item['lr']
                        or result['checkpoint'] != path or result['checkpoint_sha256'] != sha):
                    raise ValueError('Selected checkpoint identity differs from its verified completed fit')
    return selection


def evaluation_rows(selection, dataset):
    rows = []
    for family in FAMILIES:
        if family == 'f0':
            rows.append({'id': f'v12_confirmation_{dataset}_f0', 'dataset': dataset,
                         'family': family, 'seed': None})
            continue
        item = selection['selected'][dataset][family]
        for index, seed in enumerate(SEEDS):
            rows.append({'id': item['run_ids'][index], 'dataset': dataset, 'family': family, 'seed': seed,
                         'checkpoint': artifact(item['checkpoints'][index], item['checkpoint_sha256'][index])})
    if len(rows) != 11 or len({row['id'] for row in rows}) != 11:
        raise ValueError('Each fresh unit requires eleven model instances')
    return rows


def build_row_model(row, device='cuda'):
    from model_confirmation_v12 import build_model, restore_model
    if row['family'] == 'f0':
        return build_model(row['dataset'], 'f0', SEEDS[0], device=device).eval()
    saved = row['checkpoint']
    artifact(saved['path'], saved['sha256'])
    return restore_model(saved['path'], device=device).eval()


def seal_evaluation(job):
    path = HERE / 'confirmation_selection_seal.json'
    if path.exists() or (HERE / 'confirmation_test_exposure.json').exists():
        raise FileExistsError('Preserve the existing joint selection/exposure seal')
    p, selection = protocol(), load_complete_selection()
    decision = read_json(HERE / 'confirmation_no_q_decision.json')
    if set(decision['datasets']) != set(p['units']):
        raise ValueError('All approved fresh-unit TRAIN/VAL-based no-Q decisions must precede TEST')
    for item in decision['datasets'].values():
        if item['no_q_fit'] or item.get('test_error_used_for_decision', False):
            raise ValueError('This fixed same-checkpoint protocol has no independently fitted no-Q comparator')
    files = {name: digest(HERE / name) for name in
             ('selected_confirmation.json', 'confirmation_protocol.json', 'confirmation_data_manifest.json',
              'confirmation_no_q_decision.json', 'parents_selected.json', 'model_confirmation_v12.py',
              'train_confirmation_v12.py', 'evaluate_confirmation_v12.py', 'cost_confirmation_v12.py')}
    datasets = {}
    for dataset, unit in p['units'].items():
        data = load_data(dataset, include_test=False)
        contract_path = HERE / f'data_contract_confirmation_{dataset}.json'
        contract = read_json(contract_path)
        artifact(contract['test']['path'], contract['test']['sha256'])
        datasets[dataset] = {'rows': evaluation_rows(selection, dataset),
                             'data_contract': artifact(contract_path), 'test_arrays': contract['test'],
                             'block_origins': unit['block_origins'], 'unit_contract': unit,
                             'trainval_arrays': artifact(data['_path'])}
    output = {'schema': 'v12_joint_confirmation_selection_v1', 'datasets': datasets, 'files': files,
              'created_utc': time.time(), 'test_used_for_selection': False,
              'model_instances': 11 * len(datasets), 'scored_instances_including_fit0_q_references': 15 * len(datasets),
              'claim_scope': 'Fixed method and selection on the explicitly documented fresh unit; no claim of zero pretraining overlap',
              'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
                            'period_boundaries_crossed': False}}
    save_json(path, output)
    job.heartbeat('All approved confirmation units jointly selected before TEST')
    return output


def ensure_exposure():
    path = HERE / 'confirmation_selection_seal.json'
    seal = read_json(path)
    if set(seal['datasets']) != set(protocol()['units']):
        raise ValueError('Joint seal omits an approved unit')
    for name, sha in seal['files'].items():
        if digest(HERE / name) != sha:
            raise ValueError('Selected contract/source changed after the joint seal: ' + name)
    for entry in seal['datasets'].values():
        artifact(entry['data_contract']['path'], entry['data_contract']['sha256'])
        artifact(entry['test_arrays']['path'], entry['test_arrays']['sha256'])
    destination = HERE / 'confirmation_test_exposure.json'
    if destination.exists():
        if read_json(destination)['seal_sha256'] != digest(path):
            raise ValueError('Prior exposure belongs to another selection')
    else:
        save_json(destination, {'schema': 'v12_joint_confirmation_exposure_v1',
                  'first_access_utc': time.time(), 'seal_sha256': digest(path),
                  'datasets': list(seal['datasets']),
                  'scope': 'First access under this prospective unit contract; source-specific prior exposure limitations remain'})
    return seal


def read_prediction(receipt, origins, shape, key='prediction'):
    artifact(receipt['path'], receipt['sha256'])
    with np.load(receipt['path'], allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], origins):
            raise ValueError('Prediction origin order changed')
        value = archive[key].copy()
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Invalid prediction axes or values')
    return value


def predict_all(job):
    import torch
    seal = ensure_exposure()
    destination = HERE / 'confirmation_prediction_manifest.json'
    if destination.exists():
        raise FileExistsError('Joint prediction is already complete; reuse it')
    folder = CACHE / 'confirmation_predictions01'
    folder.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'v12_confirmation_predictions_v1', 'status': 'predicting',
              'seal': artifact(HERE / 'confirmation_selection_seal.json'), 'datasets': {}}
    for dataset, sealed in seal['datasets'].items():
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data[period + '_origins'] for period in PERIODS]).astype(np.int64)
        shape = (len(origins), 48, data['_C'])
        records = []
        for row in sealed['rows']:
            job.check_limits()
            path = folder / (row['id'] + '.npz')
            receipt_path = folder / (row['id'] + '.receipt.json')
            if path.exists() or receipt_path.exists():
                if not path.exists() or not receipt_path.exists():
                    raise ValueError('Partial prediction needs an explicit repair preserving the original: ' + str(path))
                record = read_json(receipt_path)
                if record['selection_seal_sha256'] != result['seal']['sha256']:
                    raise ValueError('Partial prediction belongs to another selection')
                for key in record['prediction_keys']:
                    read_prediction(record['prediction'], origins, shape, key)
            else:
                model = build_row_model(row)
                outputs = {'prediction': []}
                if row['family'] in ('a', 'level'):
                    outputs['q_prediction'] = []
                if row['family'] == 'a':
                    outputs.update(main_only_prediction=[], main_last_prediction=[])
                with torch.inference_mode():
                    for start in range(0, len(origins), 4):
                        job.check_limits()
                        x, _, _ = tensors(data, origins[start:start + 4], device='cuda')
                        if row['family'] in ('a', 'level'):
                            main, q = model.components(x)
                            prediction = main + q
                            outputs['q_prediction'].append(q.float().cpu().numpy())
                            if row['family'] == 'a':
                                residual = x - model.decoder(model.encoder(x))
                                last = residual[:, -1:, :].expand(-1, 48, -1)
                                outputs['main_only_prediction'].append(main.float().cpu().numpy())
                                outputs['main_last_prediction'].append((main + last).float().cpu().numpy())
                        else:
                            prediction = model(x)
                        outputs['prediction'].append(prediction.float().cpu().numpy())
                arrays = {key: np.concatenate(values).astype(np.float32) for key, values in outputs.items()}
                if any(value.shape != shape or not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError('Invalid fresh confirmation prediction')
                np.savez_compressed(path, origins=origins, **arrays)
                record = {**row, 'prediction': artifact(path), 'prediction_keys': list(arrays),
                          'selection_seal_sha256': result['seal']['sha256'], 'reused': False}
                save_json(receipt_path, record)
                del model, arrays, outputs, x, prediction
                if row['family'] in ('a', 'level'):
                    del main, q
                if row['family'] == 'a':
                    del residual, last
                gc.collect()
                torch.cuda.empty_cache()
            records.append(record)
            result['datasets'][dataset] = {'models': records, 'shape': list(shape),
                                            'test_arrays': artifact(data['_path'])}
            save_json(HERE / 'confirmation_prediction_manifest_partial.json', result)
            job.heartbeat(f'{dataset}: {len(records)}/11 confirmation model predictions')
    result['status'] = 'complete'
    save_json(destination, result)
    return result


def error_terms(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction/target/mask axes differ')
    observed = mask.astype(bool)
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError('Nonfinite observed target or prediction')
    error = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    return {'squared': (error * error).sum(axis=1), 'absolute': np.abs(error).sum(axis=1),
            'signed': error.sum(axis=1), 'count': observed.sum(axis=1)}


def aggregate_terms(terms, index):
    count = terms['count'][index].sum(axis=0)
    if np.any(count == 0):
        raise ValueError('A fixed evaluation channel has no observed target')
    sums = {name: terms[name][index].sum(axis=0) for name in ('squared', 'absolute', 'signed')}
    return {'mse': float(np.mean(sums['squared'] / count)), 'mae': float(np.mean(sums['absolute'] / count)),
            'signed_mean_error': float(np.mean(sums['signed'] / count)),
            'channel_mse': (sums['squared'] / count).tolist(),
            'channel_mae': (sums['absolute'] / count).tolist(), 'channel_bias': (sums['signed'] / count).tolist(),
            'squared_error_sums': sums['squared'].tolist(), 'absolute_error_sums': sums['absolute'].tolist(),
            'signed_error_sums': sums['signed'].tolist(), 'target_counts': count.astype(int).tolist(), 'origins': len(index)}


def space_diagnostic(prediction, target, mask, basis, index):
    complete = mask[index].astype(bool).all(axis=2)
    n, total = int(complete.sum()), int(complete.size)
    if not n:
        return {'status': 'unavailable_no_complete_target_vector', 'complete_rows': 0, 'total_rows': total, 'coverage': 0.}
    error = prediction[index][complete].astype(np.float64) - target[index][complete].astype(np.float64)
    u = basis.astype(np.float64)
    p = (error @ u) @ u.T
    q = error - p
    full, p_energy, q_energy = [np.sum(v * v, axis=1) for v in (error, p, q)]
    return {'status': 'available', 'complete_rows': n, 'total_rows': total, 'coverage': n / total,
            'p_mse': float(p_energy.mean() / u.shape[0]), 'q_mse': float(q_energy.mean() / u.shape[0]),
            'complete_vector_mse': float(full.mean() / u.shape[0]),
            'orthogonal_energy_sum_gap': float(np.max(np.abs(full - p_energy - q_energy))),
            'max_abs_cross_term': float(np.max(np.abs(2 * np.sum(p * q, axis=1)))),
            'gram_max_abs': float(np.max(np.abs(u.T @ u - np.eye(u.shape[1])))),
            'scope': 'Common complete observed target vectors in TRAIN-standardized coordinates',
            'warning': 'Not masked channel-macro MSE; no missing target zero fill; projected MAE is not additive'}


def bootstrap_weights(indices, block, unit_index):
    total = len(indices['combined'])
    weights = {}
    for period_index, period in enumerate(PERIODS):
        local_index = indices[period]
        n = len(local_index)
        if n < block:
            raise ValueError('Fixed period is shorter than the prespecified block')
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, unit_index, period_index]))
        matrix = np.zeros((BOOTSTRAP_DRAWS, total), dtype=np.int32)
        for draw in range(BOOTSTRAP_DRAWS):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            matrix[draw] = np.bincount(local_index[local], minlength=total)
        weights[period] = matrix
    weights['combined'] = weights['test_a'] + weights['test_b']
    return weights


def compare_roles(role_terms, indices, block, unit_index, job):
    weights = bootstrap_weights(indices, block, unit_index)
    draws = {}
    for period, matrix in weights.items():
        denominator = matrix @ role_terms['f0']['count']
        if np.any(denominator == 0):
            raise ValueError('A bootstrap draw has an unobserved fixed channel')
        draws[period] = {}
        for family, terms in role_terms.items():
            job.check_limits()
            draws[period][family] = {metric: np.mean((matrix @ terms[key]) / denominator, axis=1)
                                     for metric, key in (('mse', 'squared'), ('mae', 'absolute'))}
    pairs = [('a', role) for role in ('level', 'a_main', 'a_last', 'f0', 'full_mse', 'native', 'direct')]
    pairs.append(('a_last', 'a_main'))
    results = {}
    for left, right in pairs:
        item = {'candidate': left, 'reference': right, 'periods': {},
                'direction': 'candidate minus reference; negative favors candidate'}
        for period, index in indices.items():
            a, b = aggregate_terms(role_terms[left], index), aggregate_terms(role_terms[right], index)
            metrics = {}
            for metric in ('mse', 'mae'):
                delta = a[metric] - b[metric]
                da, db = draws[period][left][metric], draws[period][right][metric]
                metrics[metric] = {'candidate': a[metric], 'reference': b[metric], 'absolute_difference': delta,
                    'relative_change_percent': 100 * delta / b[metric] if b[metric] > 0 else None,
                    'conditional_block95_absolute': np.quantile(da - db, [.025, .975]).tolist(),
                    'conditional_block95_relative_percent': np.quantile(100 * (da - db) / db, [.025, .975]).tolist() if np.all(db > 0) else None}
            item['periods'][period] = metrics
        results[left + '__vs__' + right] = item
    return {'comparisons': results, 'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
             'block_origins': block, 'period_boundaries_crossed': False,
             'paired_channels_methods_and_seeds': True, 'selection_uncertainty_included': False,
             'scope': 'Conditional on the two selected seeds; parent/model/LR selection uncertainty is not included'}}


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_all(job):
    seal = ensure_exposure()
    manifest = read_json(HERE / 'confirmation_prediction_manifest.json')
    artifact(manifest['seal']['path'], manifest['seal']['sha256'])
    if manifest['status'] != 'complete' or set(manifest['datasets']) != set(seal['datasets']):
        raise ValueError('All approved units must finish all prescribed predictions before joint reporting')
    destination = HERE / 'confirmation_evaluation01.json'
    if destination.exists():
        raise FileExistsError('Preserve the prior confirmation; use a documented correction artifact')
    output = {'schema': 'v12_joint_confirmation_evaluation_v1', 'status': 'scoring', 'datasets': {},
              'selection_seal': artifact(HERE / 'confirmation_selection_seal.json'),
              'prediction_manifest': artifact(HERE / 'confirmation_prediction_manifest.json'),
              'evidence_status': 'confirmed_from_saved_arrays',
              'primary_metric': 'Observed channel-macro MSE in TRAIN-standardized coordinates',
              'scope': 'Prospective fixed-method evaluation under each disclosed unit exposure contract; not proof of zero backbone-pretraining overlap'}
    scalar_rows, channel_rows = [], []
    for unit_index, (dataset, sealed) in enumerate(seal['datasets'].items()):
        data = load_data(dataset, include_test=True)
        period_origins = [np.asarray(data[period + '_origins'], dtype=np.int64) for period in PERIODS]
        origins = np.concatenate(period_origins)
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        indices = {'test_a': np.arange(len(period_origins[0])),
                   'test_b': np.arange(len(period_origins[0]), len(origins)), 'combined': np.arange(len(origins))}
        rows = manifest['datasets'][dataset]['models']
        if [r['id'] for r in rows] != [r['id'] for r in sealed['rows']]:
            raise ValueError('The predicted role set differs from the joint seal')
        terms, models, q_arrays, expanded = {}, {}, {}, []
        for row in rows:
            roles = [(row['family'], row['id'], 'prediction')]
            if row['family'] == 'a':
                roles += [('a_main', row['id'] + '__main_only', 'main_only_prediction'),
                          ('a_last', row['id'] + '__main_last', 'main_last_prediction')]
            if row['family'] in ('a', 'level'):
                q_arrays[(row['family'], row['seed'])] = read_prediction(row['prediction'], origins, target.shape, 'q_prediction')
            for family, identifier, key in roles:
                prediction = read_prediction(row['prediction'], origins, target.shape, key)
                terms[identifier] = error_terms(prediction, target, mask)
                scores = {period: aggregate_terms(terms[identifier], index) for period, index in indices.items()}
                current = {**row, 'id': identifier, 'family': family, 'prediction_key': key,
                           'fit0_same_checkpoint_reference': family in ('a_main', 'a_last')}
                expanded.append(current)
                models[identifier] = {**current, 'periods': scores,
                    'space_common_U0': {period: space_diagnostic(prediction, target, mask, data['basis'], index)
                                        for period, index in indices.items()}}
                for period, score in scores.items():
                    scalar_rows.append({'dataset': dataset, 'family': family, 'id': identifier, 'seed': row['seed'],
                        'period': period, 'mse': score['mse'], 'mae': score['mae'],
                        'signed_mean_error': score['signed_mean_error'], 'evidence_status': 'confirmed'})
                    for channel, column in enumerate(data['columns']):
                        channel_rows.append({'dataset': dataset, 'family': family, 'id': identifier, 'seed': row['seed'],
                            'period': period, 'channel': str(column), 'mse': score['channel_mse'][channel],
                            'mae': score['channel_mae'][channel], 'signed_mean_error': score['channel_bias'][channel],
                            'observed_count': score['target_counts'][channel]})
        role_summary, role_terms = {}, {}
        for family in ROLES:
            members = [r for r in expanded if r['family'] == family]
            if len(members) != (1 if family == 'f0' else 2):
                raise ValueError('Missing an unfavorable role or seed is not allowed')
            count = terms[members[0]['id']]['count']
            if any(not np.array_equal(terms[r['id']]['count'], count) for r in members):
                raise ValueError('Seed target masks differ')
            average = {key: np.mean([terms[r['id']][key] for r in members], axis=0)
                       for key in ('squared', 'absolute', 'signed')}
            average['count'] = count
            role_terms[family] = average
            role_summary[family] = {'ids': [r['id'] for r in members],
                'aggregation': 'Mean seed losses, not prediction ensemble',
                'periods': {period: aggregate_terms(average, index) for period, index in indices.items()}}
        q_checks = {str(seed): parity(q_arrays[('a', seed)], q_arrays[('level', seed)]) for seed in SEEDS}
        if not all(check['pass'] for check in q_checks.values()):
            raise ValueError('A did not preserve its selected LEVEL parent Q predictions')
        term_path = CACHE / f'confirmation_{dataset}_origin_terms01.npz'
        np.savez_compressed(term_path, origins=origins,
                            **{identifier + '__' + key: value for identifier, item in terms.items() for key, value in item.items()})
        output['datasets'][dataset] = {'models': models, 'role_summary': role_summary,
            **compare_roles(role_terms, indices, sealed['block_origins'], unit_index, job),
            'A_Q_parity': q_checks, 'origin_terms': artifact(term_path), 'data_arrays': artifact(data['_path']),
            'columns': [str(c) for c in data['columns']],
            'period_origin_counts': {period: len(index) for period, index in indices.items()},
            'unit_contract': sealed['unit_contract'],
            'aggregation': 'Pool period channel SSE/count; macro across channels; then mean seed losses'}
        save_json(HERE / 'confirmation_evaluation01_partial.json', output)
        job.heartbeat(dataset + ' fixed confirmation scores complete')
        del data, target, mask, terms, models, q_arrays, role_terms
        gc.collect()
    write_csv(HERE / 'confirmation_evaluation01_models.csv', scalar_rows)
    write_csv(HERE / 'confirmation_evaluation01_channels.csv', channel_rows)
    output.update(status='complete', tables={name: artifact(HERE / f'confirmation_evaluation01_{name}.csv')
                                            for name in ('models', 'channels')})
    save_json(destination, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'predict', 'score'))
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    category = 'gpu' if args.action == 'predict' else 'cpu_analysis' if args.action == 'score' else 'cpu_check'
    with Job(category, args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'joint fresh-unit confirmation ' + args.action, 'fit_count': 0}) as job:
        configure()
        {'seal': seal_evaluation, 'predict': predict_all, 'score': score_all}[args.action](job)


if __name__ == '__main__':
    main()
