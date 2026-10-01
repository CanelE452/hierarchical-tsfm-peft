"""Recompute component-swap metrics from bound saved forecasts, without models."""
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

from runtime_v14 import HERE, SEEDS, Job, artifact, load_data, protocol, read_json, save_json
from diagnose_component_swap_v14 import DATASETS, ORIGINAL_ROLES, COMPOSITIONS, PERIODS, csv_rows


OUTPUT = HERE / 'component_swap_checks01.json'


def close(actual, expected, label, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label + ': invalid shape or nonfinite value')
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(label + ': numeric mismatch')


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def budget(job, reserve_s):
    job.check_limits()
    if time.perf_counter() - job.started >= reserve_s:
        raise TimeoutError('Reserved component-check CPU time exhausted')


def load_prediction(receipt, origins, shape):
    artifact(receipt['path'], receipt['sha256'])
    with np.load(receipt['path'], allow_pickle=False) as stored:
        assert np.array_equal(stored['origins'], origins), 'Prediction origin order'
        prediction = stored['prediction'].astype(np.float64)
    assert prediction.shape == shape and np.isfinite(prediction).all()
    return prediction


def terms_for(prediction, target, mask):
    error = np.zeros(target.shape, dtype=np.float64)
    np.subtract(prediction, target, out=error, where=mask)
    assert np.isfinite(error).all()
    return {'squared': np.sum(np.square(error), axis=1), 'absolute': np.sum(np.abs(error), axis=1),
            'signed': np.sum(error, axis=1), 'count': np.sum(mask, axis=1)}


def recompute_score(terms, index):
    count = np.sum(terms['count'][index], axis=0)
    assert np.all(count > 0)
    squared, absolute, signed = (np.sum(terms[key][index], axis=0) for key in ('squared', 'absolute', 'signed'))
    return {'mse': float(np.mean(squared / count)), 'mae': float(np.mean(absolute / count)),
            'signed_mean_error': float(np.mean(signed / count)),
            'channel_mse': (squared / count).tolist(), 'channel_mae': (absolute / count).tolist(),
            'channel_bias': (signed / count).tolist(), 'squared_error_sums': squared.tolist(),
            'absolute_error_sums': absolute.tolist(), 'signed_error_sums': signed.tolist(),
            'target_counts': count.tolist(), 'origins': len(index)}


def check_scores(record, terms, indices, label):
    assert set(record['periods']) == set(PERIODS)
    for period, index in indices.items():
        for name, expected in recompute_score(terms, index).items():
            close(record['periods'][period][name], expected, label + '/' + period + '/' + name)


def check_energy(record, prediction, target, complete, indices, projector):
    for period, index in indices.items():
        saved = record['complete_pq'][period]
        valid = complete[index]
        n, total = int(valid.sum()), int(valid.size)
        assert (saved['complete_vectors'], saved['total_vectors']) == (n, total)
        close(saved['coverage'], n / total, 'Complete coverage')
        if not n:
            assert saved['status'] == 'unavailable_no_complete_vector'
            assert all(saved[key] is None for key in ('full_mse_per_C', 'p_mse_per_C', 'q_mse_per_C',
                                                       'energy_sum_residual', 'max_abs_vector_energy_residual_per_C',
                                                       'energy_identity_pass'))
            continue
        error = prediction[index][valid] - target[index][valid]
        p = error @ projector
        q = error @ (np.eye(projector.shape[0]) - projector)
        full, retained, discarded = (np.sum(v * v, axis=1) / target.shape[-1] for v in (error, p, q))
        assert np.allclose(full, retained + discarded, atol=1e-6, rtol=1e-6)
        expected = {'full_mse_per_C': float(full.mean()), 'p_mse_per_C': float(retained.mean()),
                    'q_mse_per_C': float(discarded.mean()),
                    'energy_sum_residual': float(full.mean() - retained.mean() - discarded.mean()),
                    'max_abs_vector_energy_residual_per_C': float(np.max(np.abs(full - retained - discarded)))}
        assert saved['status'] == 'available' and saved['energy_identity_pass'] is True
        for key, value in expected.items():
            close(saved[key], value, 'Complete energy ' + key)
    n = record['complete_pq']['combined']['complete_vectors']
    if n:
        for name in ('full_mse_per_C', 'p_mse_per_C', 'q_mse_per_C'):
            expected = sum(record['complete_pq'][period][name] * record['complete_pq'][period]['complete_vectors']
                           for period in ('test_a', 'test_b') if record['complete_pq'][period]['complete_vectors']) / n
            close(record['complete_pq']['combined'][name], expected, 'Complete-vector count-weighted pooling')


def check_disagreement(saved, left, right, index):
    delta = left[index] - right[index]
    for name, expected in (('mean_squared_difference_per_C', np.mean(delta * delta)),
                           ('mean_absolute_difference', np.mean(np.abs(delta))),
                           ('max_absolute_difference', np.max(np.abs(delta)))):
        close(saved[name], expected, 'Forecast-component disagreement')
    assert (saved['atol'], saved['rtol']) == (1e-5, 1e-4)
    failures = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(right[index])))
    assert saved['outside_tolerance_elements'] == failures
    assert saved['elements'] == delta.size
    assert saved['within_existing_parity_tolerance'] == (failures == 0)


def verify_dataset(dataset, unit_index, unit, diagnostic, evaluation, manifest, job, reserve_s):
    budget(job, reserve_s)
    data = load_data(dataset, include_test=True)
    artifact(diagnostic['data']['path'], diagnostic['data']['sha256'])
    assert Path(diagnostic['data']['path']).resolve() == Path(data['_path']).resolve()
    origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
    count_a = len(data['test_a_origins'])
    indices = {'test_a': np.arange(count_a), 'test_b': np.arange(count_a, len(origins)), 'combined': np.arange(len(origins))}
    target = np.stack([data['x'][origin:origin + 48] for origin in origins]).astype(np.float64)
    mask = np.stack([data['finite'][origin:origin + 48] for origin in origins]).astype(bool)
    complete = mask.all(axis=-1)
    assert diagnostic['origins_sha256'] == hash_array(origins) and diagnostic['mask_sha256'] == hash_array(mask)
    assert diagnostic['columns'] == [str(value) for value in data['columns']]
    assert (diagnostic['C'], diagnostic['K']) == (unit['C'], unit['K'])
    u = data['pca_basis'].astype(np.float64)
    assert u.shape == (unit['C'], unit['K'])
    basis = diagnostic['basis']
    assert basis['array_key'] == 'pca_basis' and basis['projection_dtype'] == 'float64'
    assert basis['stored_sha256'] == hash_array(data['pca_basis']) and basis['stored_dtype'] == str(data['pca_basis'].dtype)
    gram = float(np.max(np.abs(u.T @ u - np.eye(unit['K']))))
    assert gram <= 1e-5
    close(basis['gram_max_abs_error'], gram, 'PCA Gram error')
    p, q = u @ u.T, np.eye(unit['C']) - u @ u.T
    for period, index in indices.items():
        coverage = diagnostic['complete_target_coverage'][period]
        assert coverage['complete_vectors'] == int(complete[index].sum())
        assert coverage['total_vectors'] == complete[index].size
        close(coverage['coverage'], np.mean(complete[index]), 'Shared complete-vector coverage')
        assert diagnostic['period_origin_counts'][period] == len(index)
    originals = [row for row in manifest['datasets'][dataset]['models'] if row['family'] in ORIGINAL_ROLES]
    assert len(originals) == 7 and len(diagnostic['models']) == 23
    for role in ORIGINAL_ROLES:
        expected = [None] if role == 'f0' else list(SEEDS)
        assert sorted([row['seed'] for row in originals if row['family'] == role], key=str) == sorted(expected, key=str)
    arrays, components, terms, by_seed = {}, {}, {}, {}
    for source in originals:
        budget(job, reserve_s)
        identifier = source['id']
        record = diagnostic['models'][identifier]
        assert (record['kind'], record['role'], record['seed']) == ('original', source['family'], source['seed'])
        assert record['source_prediction'] == source['prediction']
        prediction = load_prediction(source['prediction'], origins, target.shape)
        arrays[identifier] = prediction
        terms[identifier] = terms_for(prediction, target, mask)
        check_scores(record, terms[identifier], indices, identifier)
        check_scores(evaluation['datasets'][dataset]['models'][identifier], terms[identifier], indices, 'Original evaluation replay')
        pp, pq = prediction @ p, prediction @ q
        close(pp + pq, prediction, 'Prediction P/Q recombination', atol=1e-10, rtol=1e-10)
        close(record['projection_recombination_max_abs'], np.max(np.abs(pp + pq - prediction)), 'Saved recombination gap')
        components[identifier] = pp, pq
        check_energy(record, prediction, target, complete, indices, p)
        assert record['original_score_replay'] == {'passed': True, 'atol': 1e-10, 'rtol': 1e-9}
        by_seed[(source['family'], source['seed'])] = identifier
    expected_ids = {row['id'] for row in originals}
    for specification in COMPOSITIONS:
        for seed in SEEDS:
            parent = by_seed[(specification['parent'], seed)]
            donor = by_seed[(specification['donor'], None if specification['donor'] == 'f0' else seed)]
            pp, pq = components[parent]
            dp, dq = components[donor]
            swap_p, swap_q = dp + pq, pp + dq
            close(swap_p + swap_q, arrays[parent] + arrays[donor], 'Dual swaps sum to parent plus donor', atol=1e-10, rtol=1e-10)
            prediction = swap_p if specification['swap'] == 'swap_p' else swap_q
            identifier = specification['role'] + '__s' + str(seed)
            expected_ids.add(identifier)
            record = diagnostic['models'][identifier]
            assert (record['kind'], record['role'], record['seed'], record['swap']) == (
                'diagnostic_composition', specification['role'], seed, specification['swap'])
            assert (record['parent_id'], record['donor_id']) == (parent, donor)
            terms[identifier] = terms_for(prediction, target, mask)
            check_scores(record, terms[identifier], indices, identifier)
    assert set(diagnostic['models']) == expected_ids
    expected_roles = set(ORIGINAL_ROLES) | {row['role'] for row in COMPOSITIONS}
    assert set(diagnostic['role_summary']) == expected_roles
    for role, summary in diagnostic['role_summary'].items():
        ids = [key for key, row in diagnostic['models'].items() if row['role'] == role]
        assert summary['ids'] == ids and summary['source_instances'] == (1 if role == 'f0' else 2)
        count = terms[ids[0]]['count']
        assert all(np.array_equal(terms[key]['count'], count) for key in ids)
        mean = {key: sum(terms[identifier][key] for identifier in ids) / len(ids)
                for key in ('squared', 'absolute', 'signed')}
        mean['count'] = count
        check_scores(summary, mean, indices, 'Mean seed loss ' + role)
        if role in ORIGINAL_ROLES:
            for period in PERIODS:
                energy = summary['complete_pq'][period]
                for name in ('complete_vectors', 'total_vectors', 'coverage'):
                    close(energy[name], diagnostic['complete_target_coverage'][period][name], 'Mean energy coverage')
                for metric in ('full_mse_per_C', 'p_mse_per_C', 'q_mse_per_C'):
                    values = [diagnostic['models'][identifier]['complete_pq'][period][metric] for identifier in ids]
                    if values[0] is None:
                        assert energy[metric] is None
                    else:
                        close(energy[metric], np.mean(values), 'Mean seed energy')
    for seed in SEEDS:
        level, a, f0 = (by_seed[(role, None if role == 'f0' else seed)] for role in ('pca_level', 'a', 'f0'))
        checks = diagnostic['component_checks'][str(seed)]
        assert checks['source_ids'] == {'a': a, 'pca_level': level, 'f0': f0}
        for period, index in indices.items():
            check_disagreement(checks['A_Q_vs_LEVEL_Q'][period], components[a][1], components[level][1], index)
            check_disagreement(checks['LEVEL_P_vs_full_F0_P'][period], components[level][0], components[f0][0], index)
    expected_comparisons = {row['role'] + '__vs__' + reference for row in COMPOSITIONS
                            for reference in (row['parent'], row['donor'])}
    assert set(diagnostic['comparisons']) == expected_comparisons
    for identifier, comparison in diagnostic['comparisons'].items():
        left, right = comparison['candidate'], comparison['reference']
        assert identifier == left + '__vs__' + right and set(comparison['periods']) == set(PERIODS)
        for period, metrics in comparison['periods'].items():
            assert set(metrics) == {'mse', 'mae'}
            for metric, row in metrics.items():
                ls, rs = (diagnostic['role_summary'][role]['periods'][period][metric] for role in (left, right))
                close(row['candidate'], ls, 'Comparison candidate')
                close(row['reference'], rs, 'Comparison reference')
                close(row['absolute_difference'], ls - rs, 'Comparison difference')
                if rs > 0:
                    close(row['relative_change_percent'], 100 * (ls - rs) / rs, 'Relative percent, not percentage points')
                else:
                    assert row['relative_change_percent'] is None
                for name in ('conditional_block95_absolute', 'conditional_block95_relative_percent'):
                    interval = row[name]
                    if interval is not None:
                        assert len(interval) == 2 and np.isfinite(interval).all() and interval[0] <= interval[1]
    bootstrap = diagnostic['bootstrap']
    assert (bootstrap['draws'], bootstrap['seed'], bootstrap['unit_index'], bootstrap['block_origins']) == (
        2000, 9262026, unit_index, 42 if dataset == 'jena' else 7)
    assert bootstrap['period_boundaries_crossed'] is False and bootstrap['selection_uncertainty_included'] is False
    assert bootstrap['same_draws_across_channels_seeds_methods'] is True
    for period in PERIODS:
        receipt = bootstrap['draw_weight_receipts'][period]
        assert receipt['shape'] == [2000, len(origins)] and receipt['origin_draw_count'] == len(indices[period])
        assert len(receipt['sha256']) == 64
    assert diagnostic['counts'] == {'original_instances': 7, 'composed_seed_forecasts': 16, 'role_means': 12, 'comparisons': 16}
    return {'original_sources': 7, 'composed_seed_forecasts': 16, 'mean_roles': 12, 'point_comparisons': 16,
            'all_period_and_channel_metrics_recomputed': True, 'source_score_replay_passed': True,
            'complete_energy_recomputed': True, 'gram_max_abs_error': gram,
            'projection_and_swap_algebra_passed': True, 'bootstrap_recomputed': False}


def verify(job, reserve_s):
    diagnostic = read_json(HERE / 'component_swap01.json')
    evaluation = read_json(HERE / 'evaluation01.json')
    manifest = read_json(HERE / 'prediction_manifest.json')
    p = protocol()
    assert diagnostic['schema'] == 'v14_posthoc_component_swap_1'
    assert diagnostic['status'] == evaluation['status'] == manifest['status'] == 'complete'
    assert set(diagnostic['datasets']) == set(p['units']) == set(DATASETS)
    assert diagnostic['independent_confirmation'] is False
    assert (diagnostic['new_fits'], diagnostic['optimizer_updates'], diagnostic['model_forwards']) == (0, 0, 0)
    assert diagnostic['new_prediction_arrays_saved'] is False
    method = diagnostic['method']
    assert method['original_roles'] == list(ORIGINAL_ROLES) and method['compositions'] == list(COMPOSITIONS)
    assert method['periods'] == list(PERIODS) and method['basis_key'] == 'pca_basis'
    assert method['projection_dtype'] == 'float64' and method['no_missing_target_projection'] is True
    assert method['formulas'] == {'swap_p': 'donor @ P + parent @ Q', 'swap_q': 'parent @ P + donor @ Q'}
    assert method['original_score_tolerance'] == {'atol': 1e-10, 'rtol': 1e-9}
    assert method['complete_vector_energy_tolerance'] == {'atol': 1e-6, 'rtol': 1e-6}
    assert method['gram_maxabs_limit'] == 1e-5
    for name, receipt in diagnostic['sources'].items():
        artifact(receipt['path'], receipt['sha256'])
        if name != 'metric_bootstrap_helper':
            assert Path(receipt['path']).resolve() == (HERE / name).resolve()
    datasets = {}
    for index, dataset in enumerate(DATASETS):
        datasets[dataset] = verify_dataset(dataset, index, p['units'][dataset], diagnostic['datasets'][dataset],
                                           evaluation, manifest, job, reserve_s)
        job.heartbeat(dataset + ': component source, metric, pooling and algebra verification complete')
    expected_tables = csv_rows(diagnostic)
    assert set(diagnostic['tables']) == set(expected_tables)
    for name, expected in expected_tables.items():
        budget(job, reserve_s)
        receipt = diagnostic['tables'][name]
        artifact(receipt['path'], receipt['sha256'])
        with Path(receipt['path']).open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(expected)
        for row, target in zip(rows, expected):
            assert set(row) == set(target)
            for key, value in target.items():
                if value is None:
                    assert row[key] == ''
                elif isinstance(value, (float, int)):
                    close(float(row[key]), value, 'CSV field ' + name + '/' + key)
                else:
                    assert row[key] == str(value)
    return {'datasets': datasets, 'tables_rows': {name: len(rows) for name, rows in expected_tables.items()},
            'original_and_composed_metrics_recomputed': True, 'bootstrap_recomputed': False,
            'bootstrap_scope': 'Source/hash/seed/block/shape/interval sanity checked; resampling was not repeated',
            'independent_reproduction': False, 'new_model_runs': 0, 'new_fits': 0,
            'masked_macro_not_equated_with_complete_energy': True,
            'sources': {'diagnosis': artifact(HERE / 'component_swap01.json'),
                        'verifier': artifact(Path(__file__))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.reserve_s <= 90:
        raise ValueError('Component verification reserves at most 90 CPU seconds')
    with Job('cpu_check', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'saved component-swap verification', 'fit_count': 0, 'model_forwards': 0}) as job:
        if OUTPUT.exists():
            raise FileExistsError('Preserve prior verification; record any explicit repair before rerunning')
        output = {'schema': 'v14_component_swap_checks_1', 'job_id': job.id, 'model_forwards': 0, 'new_fits': 0}
        try:
            output.update(status='passed', checks=verify(job, args.reserve_s))
        except Exception:
            output.update(status='failed', error=traceback.format_exc())
            save_json(OUTPUT, output)
            raise
        output['elapsed_s'] = time.perf_counter() - job.started
        save_json(OUTPUT, output)


if __name__ == '__main__':
    main()
