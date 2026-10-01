"""Saved-forecast P/Q attribution; no fitting, model forward, or GPU work."""
import argparse
import csv
import hashlib
import os
import time
import traceback
from pathlib import Path

for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_variable] = '4'

import numpy as np

from runtime_v14 import HERE, ROOT, SEEDS, Job, artifact, load_data, protocol, read_json, save_json
from evaluate_v14 import previous_api, prediction_array


DATASETS = ('robin', 'jena', 'hog', 'peacock_education')
ORIGINAL_ROLES = ('pca_level', 'a', 'f0', 'full_mse')
PARENTS, DONORS = ('pca_level', 'a'), ('f0', 'full_mse')
PERIODS = ('test_a', 'test_b', 'combined')
OUTPUT = HERE / 'component_swap01.json'
SCORE_ATOL, SCORE_RTOL = 1e-10, 1e-9
ENERGY_ATOL, ENERGY_RTOL = 1e-6, 1e-6
COMPOSITIONS = tuple({'role': f'{swap}__{parent}__{donor}', 'swap': swap,
                      'parent': parent, 'donor': donor}
                     for parent in PARENTS for donor in DONORS for swap in ('swap_p', 'swap_q'))


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def checkpoint_budget(job, reserve_s):
    job.check_limits()
    if time.perf_counter() - job.started >= reserve_s:
        raise TimeoutError('Reserved component-attribution CPU time exhausted; preserve partial output')


def require_sources():
    p = protocol()
    evaluation = read_json(HERE / 'evaluation01.json')
    manifest = read_json(HERE / 'prediction_manifest.json')
    if set(p['units']) != set(DATASETS) or list(SEEDS) != [92601, 92602]:
        raise ValueError('The fixed four-dataset/two-seed contract changed')
    for value in (evaluation, manifest):
        if value['status'] != 'complete' or set(value['datasets']) != set(DATASETS):
            raise ValueError('All original evaluation and prediction records must be complete')
    for receipt in (evaluation['prediction_manifest'], evaluation['selection_seal'], manifest['seal']):
        artifact(receipt['path'], receipt['sha256'])
    names = ('POSTHOC_COMPONENT_PLAN.md', 'protocol.json', 'data_manifest.json', 'reuse_manifest.json',
             'selection_seal.json', 'test_exposure.json', 'prediction_manifest.json', 'evaluation01.json',
             'diagnose_component_swap_v14.py', 'verify_component_swap_v14.py', 'evaluate_v14.py', 'runtime_v14.py')
    sources = {name: artifact(HERE / name) for name in names}
    helper = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001/evaluate_confirmation_v12.py'
    sources['metric_bootstrap_helper'] = artifact(helper)
    return p, evaluation, manifest, sources


def original_rows(manifest, dataset):
    rows = [row for row in manifest['datasets'][dataset]['models'] if row['family'] in ORIGINAL_ROLES]
    for role in ORIGINAL_ROLES:
        seeds = [row['seed'] for row in rows if row['family'] == role]
        expected = [None] if role == 'f0' else list(SEEDS)
        if sorted(seeds, key=str) != sorted(expected, key=str):
            raise ValueError('Missing or duplicate original role/seed: ' + dataset + '/' + role)
    if len(rows) != 7 or len({row['id'] for row in rows}) != 7:
        raise ValueError('Exactly seven original model instances are required')
    return rows


def numeric_close(value, reference, label):
    if not np.allclose(value, reference, atol=SCORE_ATOL, rtol=SCORE_RTOL):
        raise ValueError(label + ': stored original metric differs')


def complete_energy(prediction, target, complete, index, projector):
    selected = complete[index]
    n = int(selected.sum())
    result = {'complete_vectors': n, 'total_vectors': int(selected.size),
              'coverage': n / selected.size if selected.size else None}
    if not n:
        return {**result, 'status': 'unavailable_no_complete_vector', 'full_mse_per_C': None,
                'p_mse_per_C': None, 'q_mse_per_C': None, 'energy_sum_residual': None,
                'max_abs_vector_energy_residual_per_C': None, 'energy_identity_pass': None}
    error = prediction[index][selected] - target[index][selected]
    p = error @ projector
    q = error @ (np.eye(projector.shape[0]) - projector)
    full, pe, qe = (np.mean(v * v, axis=1) for v in (error, p, q))
    passed = bool(np.allclose(full, pe + qe, atol=ENERGY_ATOL, rtol=ENERGY_RTOL))
    if not passed:
        raise ValueError('Complete-vector score-unit P/Q energy identity exceeds fixed tolerance')
    return {**result, 'status': 'available', 'full_mse_per_C': float(full.mean()),
            'p_mse_per_C': float(pe.mean()), 'q_mse_per_C': float(qe.mean()),
            'energy_sum_residual': float(full.mean() - pe.mean() - qe.mean()),
            'max_abs_vector_energy_residual_per_C': float(np.max(np.abs(full - pe - qe))),
            'energy_identity_pass': passed}


def disagreement(left, right, index):
    difference = left[index] - right[index]
    allowance = 1e-5 + 1e-4 * np.abs(right[index])
    return {'mean_squared_difference_per_C': float(np.mean(difference * difference)),
            'mean_absolute_difference': float(np.mean(np.abs(difference))),
            'max_absolute_difference': float(np.max(np.abs(difference))),
            'atol': 1e-5, 'rtol': 1e-4,
            'outside_tolerance_elements': int(np.count_nonzero(np.abs(difference) > allowance)),
            'elements': int(difference.size),
            'within_existing_parity_tolerance': bool(np.all(np.abs(difference) <= allowance))}


def seed_mean_terms(ids, terms):
    count = terms[ids[0]]['count']
    if any(not np.array_equal(terms[identifier]['count'], count) for identifier in ids):
        raise ValueError('Sources/compositions do not share their observed target mask')
    result = {name: np.mean([terms[identifier][name] for identifier in ids], axis=0)
              for name in ('squared', 'absolute', 'signed')}
    result['count'] = count
    return result


def comparisons(role_terms, indices, block, unit_index, job, reserve_s, api):
    weights = api.bootstrap_weights(indices, block, unit_index)
    bootstrap_receipts, draws = {}, {}
    for period, weight in weights.items():
        checkpoint_budget(job, reserve_s)
        bootstrap_receipts[period] = {'shape': list(weight.shape), 'sha256': array_hash(weight),
                                      'origin_draw_count': int(weight[0].sum())}
        weight = weight.astype(np.float64)
        denominator = weight @ role_terms['f0']['count']
        if np.any(denominator <= 0):
            raise ValueError('An unchanged bootstrap draw lacks an observed fixed channel')
        draws[period] = {}
        for role, terms in role_terms.items():
            checkpoint_budget(job, reserve_s)
            joined = np.concatenate([terms['squared'], terms['absolute']], axis=1)
            sampled = weight @ joined
            squared, absolute = np.split(sampled, 2, axis=1)
            draws[period][role] = {'mse': np.mean(squared / denominator, axis=1),
                                    'mae': np.mean(absolute / denominator, axis=1)}
    result = {}
    for specification in COMPOSITIONS:
        left = specification['role']
        for right in (specification['parent'], specification['donor']):
            record = {'candidate': left, 'reference': right, 'periods': {}}
            for period, index in indices.items():
                ls, rs = (api.aggregate_terms(role_terms[role], index) for role in (left, right))
                record['periods'][period] = {}
                for metric in ('mse', 'mae'):
                    ld, rd = draws[period][left][metric], draws[period][right][metric]
                    delta = ls[metric] - rs[metric]
                    record['periods'][period][metric] = {
                        'candidate': ls[metric], 'reference': rs[metric], 'absolute_difference': delta,
                        'relative_change_percent': 100 * delta / rs[metric] if rs[metric] > 0 else None,
                        'conditional_block95_absolute': np.quantile(ld - rd, [.025, .975]).tolist(),
                        'conditional_block95_relative_percent': np.quantile(100 * (ld - rd) / rd, [.025, .975]).tolist()
                        if np.all(rd > 0) else None}
            result[left + '__vs__' + right] = record
    return result, {'draws': 2000, 'seed': 9262026, 'unit_index': unit_index, 'block_origins': block,
                    'period_boundaries_crossed': False, 'same_draws_across_channels_seeds_methods': True,
                    'selection_uncertainty_included': False, 'draw_weight_receipts': bootstrap_receipts,
                    'scope': 'Conditional on these exposed selected models; F0 is one reused source, not two replicates'}


def diagnose_dataset(dataset, unit_index, unit, evaluation, manifest, api, job, reserve_s):
    checkpoint_budget(job, reserve_s)
    data = load_data(dataset, include_test=True)
    origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
    n_a = len(data['test_a_origins'])
    indices = {'test_a': np.arange(n_a), 'test_b': np.arange(n_a, len(origins)), 'combined': np.arange(len(origins))}
    target = np.stack([data['x'][origin:origin + 48] for origin in origins]).astype(np.float64)
    mask = np.stack([data['finite'][origin:origin + 48] for origin in origins]).astype(bool)
    complete = mask.all(axis=-1)
    stored_u = data['pca_basis']
    u = stored_u.astype(np.float64)
    if u.shape != (unit['C'], unit['K']) or not np.isfinite(u).all():
        raise ValueError('Canonical PCA basis has an invalid shape or value')
    gram_error = float(np.max(np.abs(u.T @ u - np.eye(unit['K']))))
    if gram_error > 1e-5:
        raise ValueError('Original PCA Gram error exceeds the fixed tolerance')
    projector, complement = u @ u.T, np.eye(unit['C']) - u @ u.T
    arrays, projections, terms, models, by_seed = {}, {}, {}, {}, {}
    for row in original_rows(manifest, dataset):
        checkpoint_budget(job, reserve_s)
        identifier = row['id']
        old = evaluation['datasets'][dataset]['models'][identifier]
        for name in ('path', 'sha256'):
            if row['prediction'][name] != old['prediction'][name]:
                raise ValueError('Evaluation and manifest prediction binding differ')
        value = prediction_array(row['prediction'], origins, target.shape).astype(np.float64)
        arrays[identifier] = value
        projected, residual = value @ projector, value @ complement
        projections[identifier] = (projected, residual)
        gap = float(np.max(np.abs(projected + residual - value)))
        if not np.allclose(projected + residual, value, atol=1e-10, rtol=1e-10):
            raise ValueError('P/Q forecast recombination failed')
        term = api.error_terms(value, target, mask)
        scores = {period: api.aggregate_terms(term, index) for period, index in indices.items()}
        for period in PERIODS:
            for metric in ('mse', 'mae', 'signed_mean_error', 'target_counts'):
                numeric_close(scores[period][metric], old['periods'][period][metric], identifier + '/' + period + '/' + metric)
        models[identifier] = {'id': identifier, 'kind': 'original', 'role': row['family'], 'seed': row['seed'],
            'source_prediction': row['prediction'], 'periods': scores, 'projection_recombination_max_abs': gap,
            'complete_pq': {period: complete_energy(value, target, complete, index, projector)
                            for period, index in indices.items()},
            'original_score_replay': {'passed': True, 'atol': SCORE_ATOL, 'rtol': SCORE_RTOL}}
        terms[identifier] = term
        by_seed[(row['family'], row['seed'])] = identifier
    for specification in COMPOSITIONS:
        for seed in SEEDS:
            checkpoint_budget(job, reserve_s)
            parent = by_seed[(specification['parent'], seed)]
            donor = by_seed[(specification['donor'], None if specification['donor'] == 'f0' else seed)]
            pp, pq = projections[parent]
            dp, dq = projections[donor]
            value = dp + pq if specification['swap'] == 'swap_p' else pp + dq
            identifier = specification['role'] + '__s' + str(seed)
            term = api.error_terms(value, target, mask)
            models[identifier] = {'id': identifier, 'kind': 'diagnostic_composition', 'role': specification['role'],
                'seed': seed, 'parent_id': parent, 'donor_id': donor, 'swap': specification['swap'],
                'periods': {period: api.aggregate_terms(term, index) for period, index in indices.items()}}
            terms[identifier] = term
    role_terms, role_summary = {}, {}
    for role in ORIGINAL_ROLES + tuple(specification['role'] for specification in COMPOSITIONS):
        ids = [identifier for identifier, record in models.items() if record['role'] == role]
        expected = 1 if role == 'f0' else 2
        if len(ids) != expected:
            raise ValueError('Incorrect original/composition seed count')
        role_terms[role] = seed_mean_terms(ids, terms)
        summary = {'ids': ids, 'source_instances': expected, 'aggregation': 'Mean seed losses; never an ensemble',
                   'periods': {period: api.aggregate_terms(role_terms[role], index) for period, index in indices.items()}}
        if role in ORIGINAL_ROLES:
            summary['complete_pq'] = {}
            for period in PERIODS:
                rows = [models[identifier]['complete_pq'][period] for identifier in ids]
                summary['complete_pq'][period] = {**{name: rows[0][name] for name in ('complete_vectors', 'total_vectors', 'coverage')},
                    **{metric: float(np.mean([row[metric] for row in rows])) if rows[0][metric] is not None else None
                       for metric in ('full_mse_per_C', 'p_mse_per_C', 'q_mse_per_C')}}
        role_summary[role] = summary
    component_checks = {}
    for seed in SEEDS:
        level, a, f0 = (by_seed[(role, None if role == 'f0' else seed)] for role in ('pca_level', 'a', 'f0'))
        component_checks[str(seed)] = {
            'A_Q_vs_LEVEL_Q': {period: disagreement(projections[a][1], projections[level][1], index)
                               for period, index in indices.items()},
            'LEVEL_P_vs_full_F0_P': {period: disagreement(projections[level][0], projections[f0][0], index)
                                    for period, index in indices.items()},
            'source_ids': {'a': a, 'pca_level': level, 'f0': f0}}
    comparison, bootstrap = comparisons(role_terms, indices, unit['block_origins'], unit_index, job, reserve_s, api)
    return {'C': unit['C'], 'K': unit['K'], 'data': artifact(data['_path']),
            'columns': [str(value) for value in data['columns']],
            'basis': {'array_key': 'pca_basis', 'stored_dtype': str(stored_u.dtype), 'stored_sha256': array_hash(stored_u),
                      'projection_dtype': 'float64', 'gram_max_abs_error': gram_error},
            'origins_sha256': array_hash(origins), 'mask_sha256': array_hash(mask),
            'period_origin_counts': {period: len(index) for period, index in indices.items()},
            'complete_target_coverage': {period: {'complete_vectors': int(complete[index].sum()),
                                                 'total_vectors': int(complete[index].size),
                                                 'coverage': float(complete[index].mean())}
                                         for period, index in indices.items()},
            'models': models, 'role_summary': role_summary, 'component_checks': component_checks,
            'comparisons': comparison, 'bootstrap': bootstrap,
            'counts': {'original_instances': 7, 'composed_seed_forecasts': 16, 'role_means': 12, 'comparisons': 16}}


def csv_rows(output):
    scalars, channels, contrasts = [], [], []
    for dataset, unit in output['datasets'].items():
        for collection, records in (('instance', unit['models']), ('mean_seed_loss', unit['role_summary'])):
            for identifier, record in records.items():
                for period, score in record['periods'].items():
                    base = {'dataset': dataset, 'aggregation': collection, 'id': identifier,
                            'role': record.get('role', identifier), 'seed': record.get('seed'), 'period': period}
                    scalars.append({**base, **{name: score[name] for name in ('mse', 'mae', 'signed_mean_error', 'origins')}})
                    for c, column in enumerate(unit['columns']):
                        channels.append({**base, 'channel': column, 'mse': score['channel_mse'][c],
                                         'mae': score['channel_mae'][c], 'signed_mean_error': score['channel_bias'][c],
                                         'target_count': score['target_counts'][c]})
        for name, comparison in unit['comparisons'].items():
            for period, metrics in comparison['periods'].items():
                for metric, row in metrics.items():
                    absolute, relative = row['conditional_block95_absolute'], row['conditional_block95_relative_percent']
                    contrasts.append({'dataset': dataset, 'comparison': name, 'period': period, 'metric': metric,
                                      **{key: row[key] for key in ('candidate', 'reference', 'absolute_difference', 'relative_change_percent')},
                                      'conditional95_low': absolute[0], 'conditional95_high': absolute[1],
                                      'conditional95_relative_low_percent': None if relative is None else relative[0],
                                      'conditional95_relative_high_percent': None if relative is None else relative[1]})
    return {'models': scalars, 'channels': channels, 'comparisons': contrasts}


def write_tables(output):
    result = {}
    for name, rows in csv_rows(output).items():
        path = HERE / f'component_swap01_{name}.csv'
        with path.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result[name] = artifact(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.reserve_s <= 300:
        raise ValueError('Component analysis reserves at most 300 CPU seconds')
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'saved-forecast posthoc component attribution', 'fit_count': 0,
                       'model_forwards': 0, 'gpu_work': False}) as job:
        occupied = [OUTPUT, HERE / 'component_swap01_partial.json', HERE / 'component_swap01_failure.json']
        occupied += [HERE / f'component_swap01_{name}.csv' for name in ('models', 'channels', 'comparisons')]
        if any(path.exists() for path in occupied):
            raise FileExistsError('Preserve prior component analysis; a recorded explicit repair is required')
        output = {'schema': 'v14_posthoc_component_swap_1', 'status': 'partial', 'job_id': job.id,
                  'created_utc': time.time(), 'new_fits': 0, 'optimizer_updates': 0, 'model_forwards': 0,
                  'independent_confirmation': False, 'new_prediction_arrays_saved': False,
                  'method': {'original_roles': list(ORIGINAL_ROLES), 'compositions': list(COMPOSITIONS),
                             'periods': list(PERIODS), 'basis_key': 'pca_basis', 'projection_dtype': 'float64',
                             'formulas': {'swap_p': 'donor @ P + parent @ Q', 'swap_q': 'parent @ P + donor @ Q'},
                             'no_missing_target_projection': True, 'cpu_threads': 4,
                             'original_score_tolerance': {'atol': SCORE_ATOL, 'rtol': SCORE_RTOL},
                             'complete_vector_energy_tolerance': {'atol': ENERGY_ATOL, 'rtol': ENERGY_RTOL},
                             'gram_maxabs_limit': 1e-5},
                  'interpretation': ['All four datasets are exposed development evaluations.',
                      'Fixed saved-forecast compositions are attribution diagnostics, not low-cost proposed models or target oracles.',
                      'Deploying a composition retains donor and parent forecasting costs; no speedup is inferred.',
                      'F0 has one source reused across paired seeds, not two independently fitted models.',
                      'When A and LEVEL retain the same Q, their swap_P forecasts with the same donor are numerical duplicates, not independent evidence.',
                      'Mean seed losses; pool channel error sums/counts across periods before channel averaging.',
                      'P/Q error energies use common complete target vectors and do not replace masked channel-macro metrics.',
                      'LEVEL projected output approximates its compressed main up to recorded FP32 projection leakage.',
                      'LEVEL_P versus F0_P disagreement does not isolate information loss, Transformer nonlinearity or pretraining effects.'],
                  'datasets': {}}
        try:
            p, evaluation, manifest, output['sources'] = require_sources()
            api = previous_api()
            for index, dataset in enumerate(DATASETS):
                output['datasets'][dataset] = diagnose_dataset(dataset, index, p['units'][dataset], evaluation, manifest,
                                                                api, job, args.reserve_s)
                save_json(HERE / 'component_swap01_partial.json', output)
                job.heartbeat(dataset + ': fixed component swaps, source replay and conditional intervals complete')
            checkpoint_budget(job, args.reserve_s)
            output['tables'] = write_tables(output)
            output.update(status='complete', elapsed_s=time.perf_counter() - job.started)
            save_json(OUTPUT, output)
        except Exception:
            save_json(HERE / 'component_swap01_failure.json', {'status': 'failed', 'job_id': job.id,
                      'completed_datasets': list(output['datasets']), 'error': traceback.format_exc(),
                      'partial': 'component_swap01_partial.json', 'model_forwards': 0, 'new_fits': 0})
            raise


if __name__ == '__main__':
    main()
