"""Check saved space-diagnostic algebra and provenance without new predictions."""
import argparse
import hashlib
import traceback

import numpy as np

from runtime_v13 import HERE, SEEDS, Job, artifact, budget_snapshot, digest, load_data, protocol, read_json, save_json


ROLES = ('group_res', 'group_raw', 'pca_level', 'a', 'f0', 'full_mse', 'native', 'direct')
PERIODS = ('test_a', 'test_b', 'combined')
ENERGIES = ('full_energy_mse_per_C', 'p_energy_mse_per_C', 'q_energy_mse_per_C')


def close(actual, expected, name, atol=1e-10, rtol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(name + ': shape or finiteness')
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(name + ': numeric mismatch')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def verify(job):
    diagnostic = read_json(HERE / 'space_diagnosis01.json')
    evaluation = read_json(HERE / 'evaluation01.json')
    manifest = read_json(HERE / 'prediction_manifest.json')
    p = protocol()
    assert diagnostic['schema'] == 'v13_space_diagnosis_1'
    assert diagnostic['status'] == evaluation['status'] == manifest['status'] == 'complete'
    assert set(diagnostic['datasets']) == set(evaluation['datasets']) == set(manifest['datasets']) == set(p['units'])
    method = diagnostic['method']
    assert method['roles'] == list(ROLES) and method['periods'] == list(PERIODS)
    assert method['no_zero_fill'] and method['no_new_prediction_or_fit']
    assert method['projection_basis'] == 'the fixed v13 GROUP U common to every stored method'
    sources = diagnostic['sources']
    for key, name in (('protocol', 'protocol.json'), ('data_manifest', 'data_manifest.json'),
                      ('reuse_manifest', 'reuse_manifest.json'), ('selection_seal', 'selection_seal.json'),
                      ('test_exposure', 'test_exposure.json'), ('prediction_manifest', 'prediction_manifest.json'),
                      ('evaluation01', 'evaluation01.json')):
        receipt = sources[key]
        artifact(receipt['path'], receipt['sha256'])
        assert receipt['sha256'] == digest(HERE / name)
    assert sources['diagnostic_code_sha256'] == digest(HERE / 'diagnose_space_v13.py')
    checked_receipts, datasets, global_residual = set(), {}, 0.
    for dataset, unit in diagnostic['datasets'].items():
        job.check_limits()
        data = load_data(dataset, include_test=True)
        c, k = p['units'][dataset]['C'], p['units'][dataset]['K']
        assert (unit['C'], unit['K']) == (c, k)
        assert unit['columns'] == evaluation['datasets'][dataset]['columns'] == [str(v) for v in data['columns']]
        artifact(unit['data']['path'], unit['data']['sha256'])
        assert unit['data']['sha256'] == digest(data['_path'])
        stored, u = data['basis'], data['basis'].astype(np.float64)
        basis = unit['basis']
        assert u.shape == (c, k) and np.isfinite(u).all()
        assert basis['stored_sha256'] == array_hash(stored) and basis['diagnostic_sha256'] == array_hash(u)
        assert basis['stored_dtype'] == str(stored.dtype) and basis['diagnostic_dtype'] == str(u.dtype)
        assert np.all(u >= 0) and np.all((u > 0).sum(axis=1) == 1)
        groups = [np.flatnonzero(u[:, j] > 0).tolist() for j in range(k)]
        assert basis['groups'] == groups and all(groups)
        gram_error = float(np.max(np.abs(u.T @ u - np.eye(k))))
        assert gram_error <= 1e-5
        close(basis['gram_max_abs_error'], gram_error, 'Gram error', atol=1e-15)
        origins = np.concatenate([data['test_a_origins'], data['test_b_origins']])
        a = len(data['test_a_origins'])
        indices = {'test_a': np.arange(a), 'test_b': np.arange(a, len(origins)), 'combined': np.arange(len(origins))}
        complete = np.stack([data['finite'][o:o + 48].all(axis=-1) for o in origins])
        coverage = unit['complete_vector_coverage']
        for period, index in indices.items():
            counts = coverage[period]
            assert counts['complete_target_vectors'] == int(complete[index].sum())
            assert counts['total_target_vectors'] == len(index) * 48
            close(counts['fraction'], counts['complete_target_vectors'] / counts['total_target_vectors'], 'Coverage fraction')
        for key in ('complete_target_vectors', 'total_target_vectors'):
            assert coverage['combined'][key] == coverage['test_a'][key] + coverage['test_b'][key]
        saved_models = {row['id']: row for row in manifest['datasets'][dataset]['models']}
        assert set(unit['models']) == set(saved_models) == set(evaluation['datasets'][dataset]['models'])
        assert len(unit['models']) == 15
        for role in ROLES:
            seeds = [row['seed'] for row in unit['models'].values() if row['family'] == role]
            assert sorted(seeds, key=str) == sorted([None] if role == 'f0' else list(SEEDS), key=str)
        max_residual = 0.
        for identifier, model in unit['models'].items():
            parent = saved_models[identifier]
            assert (model['id'], model['family'], model['seed'], model['reused']) == (identifier, parent['family'], parent['seed'], parent['reused'])
            receipt = model['prediction']
            assert receipt == parent['prediction'] == evaluation['datasets'][dataset]['models'][identifier]['prediction']
            key = (receipt['path'], receipt['sha256'])
            if key not in checked_receipts:
                artifact(*key)
                checked_receipts.add(key)
            assert set(model['periods']) == set(PERIODS)
            for period, values in model['periods'].items():
                counts = coverage[period]
                for key in ('complete_target_vectors', 'total_target_vectors'):
                    assert values[key] == counts[key]
                close(values['coverage_fraction'], counts['fraction'], 'Model coverage')
                original = values['original_masked_macro']
                assert original['source'] == 'evaluation01.json' and original['model_id'] == identifier and original['period'] == period
                expected = evaluation['datasets'][dataset]['models'][identifier]['periods'][period]
                for key in ('mse', 'mae', 'signed_mean_error'):
                    close(original[key], expected[key], 'Original masked macro')
                if counts['complete_target_vectors'] == 0:
                    assert all(values[key] is None for key in ENERGIES + ('p_plus_q_numerical_residual',))
                    continue
                full, part, residual = (values[key] for key in ENERGIES)
                assert np.isfinite([full, part, residual]).all() and min(full, part, residual) >= 0
                gap = full - part - residual
                close(values['p_plus_q_numerical_residual'], gap, 'P/Q energy algebra', atol=1e-12)
                assert abs(gap) <= k * gram_error * part + 1e-10 * max(1., full)
                for key, energy in (('p_fraction_of_full', part), ('q_fraction_of_full', residual)):
                    if full > 0:
                        close(values[key], energy / full, 'Energy fraction')
                    else:
                        assert values[key] is None
                max_residual = max(max_residual, abs(gap))
            combined_count = coverage['combined']['complete_target_vectors']
            if combined_count:
                for metric in ENERGIES:
                    weighted = sum(model['periods'][period][metric] * coverage[period]['complete_target_vectors']
                                   for period in ('test_a', 'test_b') if coverage[period]['complete_target_vectors']) / combined_count
                    close(model['periods']['combined'][metric], weighted, 'Complete-count weighted period pooling')
        close(unit['max_abs_p_plus_q_numerical_residual'], max_residual, 'Dataset maximum residual', atol=1e-12)
        global_residual = max(global_residual, max_residual)
        datasets[dataset] = {'models': len(unit['models']), 'coverage': coverage, 'gram_max_abs_error': gram_error,
                             'max_abs_energy_residual': max_residual, 'common_group_basis': True}
        job.heartbeat(dataset + ': diagnostic receipts, mask coverage and saved energy algebra checked')
    close(diagnostic['max_abs_p_plus_q_numerical_residual'], global_residual, 'Global maximum residual', atol=1e-12)
    return {'datasets': datasets, 'prediction_receipts_checked': len(checked_receipts),
            'original_masked_macro_matched': True, 'full_prediction_reprojection': False,
            'new_prediction_scoring': False, 'bootstrap_recomputed': False, 'prose_checked': False,
            'sources': {'diagnosis': artifact(HERE / 'space_diagnosis01.json'), 'checker': artifact(HERE / 'verify_space_v13.py')},
            'interpretation': 'Complete-vector energy algebra only; not causality or equality with masked channel-macro MSE'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    destination = HERE / 'space_checks01.json'
    with Job('cpu_check', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'v13 saved space-diagnostic verification', 'fit_count': 0}) as job:
        if destination.exists():
            old = read_json(destination)
            if old['status'] == 'passed':
                raise FileExistsError('Space verification already passed; preserve the existing receipt')
            save_json(HERE / f'space_checks01_prior_job_{old["job_id"]}.json', old)
        output = {'schema': 'v13_space_checks_1', 'job_id': job.id, 'model_runs': 0, 'optimizer_updates': 0,
                  'independent_reproduction': False}
        try:
            output.update(status='passed', checks=verify(job))
        except Exception:
            output.update(status='failed', error=traceback.format_exc())
            save_json(destination, output)
            raise
        save_json(destination, output)
    output['budget_after_check'] = budget_snapshot()
    save_json(destination, output)


if __name__ == '__main__':
    main()
