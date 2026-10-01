"""Saved-forecast temporal P/Q error on completely observed four-position blocks."""
import argparse
import csv
import hashlib
import time
import traceback

import numpy as np

from runtime_v16 import HERE, DATASETS, SEEDS, Job, artifact, load_data, read_json, save_json
from evaluate_v16 import PERIODS, ROLES, prediction_array, validate_rows


OUTPUT = HERE / 'temporal_diagnosis01.json'
TABLE = HERE / 'temporal_diagnosis01.csv'
ENERGY_ATOL, ENERGY_RTOL = 1e-10, 1e-9
PARITY_ATOL, PARITY_RTOL = 1e-5, 1e-4
ENERGIES = ('squared_error', 'p_energy', 'q_energy', 'cross_twice')
MACROS = ('subset_macro_mse', 'p_macro_energy', 'q_macro_energy', 'cross_macro_twice')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def budget(job, reserve_s):
    job.check_limits()
    if time.perf_counter() - job.started >= reserve_s:
        raise TimeoutError('Reserved temporal-diagnosis CPU interval exhausted')


def coverage(mask4, indices):
    mask = mask4[indices]
    complete = mask.all(axis=2)
    counts = complete.sum(axis=(0, 1)).astype(np.int64)
    observed = mask.sum(axis=(0, 1, 2)).astype(np.int64)
    total = int(complete.size)
    return {'complete_channel_blocks': int(counts.sum()), 'possible_channel_blocks': total,
            'complete_block_fraction': float(counts.sum() / total) if total else None,
            'observed_target_values': int(observed.sum()), 'kept_target_values': int(4 * counts.sum()),
            'retained_observed_target_fraction': float(4 * counts.sum() / observed.sum()) if observed.sum() else None,
            'channel_valid_blocks': counts.tolist(), 'channel_observed_target_values': observed.tolist(),
            'valid_channel_indices': np.flatnonzero(counts > 0).tolist(),
            'valid_channel_count': int(np.count_nonzero(counts)), 'total_channel_count': len(counts),
            'all_channels_represented': bool(np.all(counts > 0)),
            'overlapping_forecast_targets_are_not_independent_samples': True}


def energy_sums(prediction4, target4, mask4, indices):
    channels = target4.shape[-1]
    result = {key: np.zeros(channels, dtype=np.float64) for key in ENERGIES}
    complete = mask4[indices].all(axis=2)
    maximum_identity_residual = 0.
    identity_pass = True
    for channel in range(channels):
        valid = complete[:, :, channel]
        if not valid.any():
            continue
        # Choose complete channel-blocks before any target subtraction/projection.
        predicted = prediction4[indices, :, :, channel][valid].astype(np.float64)
        observed = target4[indices, :, :, channel][valid].astype(np.float64)
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError('A complete observed target block is nonfinite')
        error = predicted - observed
        p = np.repeat(error.mean(axis=1, keepdims=True), 4, axis=1)
        q = error - p
        per_block = {'squared_error': (error * error).sum(axis=1),
                     'p_energy': (p * p).sum(axis=1), 'q_energy': (q * q).sum(axis=1),
                     'cross_twice': 2 * (p * q).sum(axis=1)}
        for key in ENERGIES:
            result[key][channel] = per_block[key].sum()
        residual = (per_block['squared_error'] - per_block['p_energy'] - per_block['q_energy']) / 4
        maximum_identity_residual = max(maximum_identity_residual, float(np.abs(residual).max()))
        identity_pass &= bool(np.all(np.abs(residual) <= ENERGY_ATOL + ENERGY_RTOL * per_block['squared_error'] / 4))
    return {**{key: value.tolist() for key, value in result.items()},
            'max_abs_block_mse_energy_identity_residual': maximum_identity_residual,
            'energy_identity_pass': identity_pass}


def summarize(sums, scope, primary):
    counts = np.asarray(scope['channel_valid_blocks'], dtype=np.int64) * 4
    valid = counts > 0
    channels = {}
    output = {'status': 'available' if valid.any() else 'unavailable_no_complete_block',
              'coverage': scope, 'channel_sums': {key: sums[key] for key in ENERGIES},
              'original_masked_macro': {key: primary[key] for key in ('mse', 'mae', 'signed_mean_error')},
              'primary_score_recomputed': False,
              'max_abs_block_mse_energy_identity_residual': sums['max_abs_block_mse_energy_identity_residual'],
              'energy_identity_pass': sums['energy_identity_pass']}
    for key, name in zip(ENERGIES, MACROS):
        values = np.asarray(sums[key], dtype=np.float64)
        channels[name] = [float(values[c] / counts[c]) if counts[c] else None for c in range(len(counts))]
        output[name] = float(np.mean(values[valid] / counts[valid])) if valid.any() else None
    output['channel_values'] = channels
    output['macro_energy_identity_residual'] = (output['subset_macro_mse'] - output['p_macro_energy'] - output['q_macro_energy']) if valid.any() else None
    return output


def combine_sums(left, right):
    return {**{key: (np.asarray(left[key]) + np.asarray(right[key])).tolist() for key in ENERGIES},
            'max_abs_block_mse_energy_identity_residual': max(left['max_abs_block_mse_energy_identity_residual'], right['max_abs_block_mse_energy_identity_residual']),
            'energy_identity_pass': left['energy_identity_pass'] and right['energy_identity_pass']}


def average_sums(members):
    return {**{key: np.mean([item[key] for item in members], axis=0).tolist() for key in ENERGIES},
            'max_abs_block_mse_energy_identity_residual': max(item['max_abs_block_mse_energy_identity_residual'] for item in members),
            'energy_identity_pass': all(item['energy_identity_pass'] for item in members)}


def q_parity(candidate4, reference4, mask4, indices):
    complete = mask4[indices].all(axis=2)
    max_absolute = max_relative = 0.
    compared = violations = 0
    channel_max = []
    for channel in range(mask4.shape[-1]):
        valid = complete[:, :, channel]
        if not valid.any():
            channel_max.append(None)
            continue
        candidate = candidate4[indices, :, :, channel][valid].astype(np.float64)
        cq = candidate - candidate.mean(axis=1, keepdims=True)
        if reference4 is None:
            reference = np.zeros_like(cq)
        else:
            reference = reference4[indices, :, :, channel][valid].astype(np.float64)
            reference = reference - reference.mean(axis=1, keepdims=True)
        difference = np.abs(cq - reference)
        tolerance = PARITY_ATOL + PARITY_RTOL * np.abs(reference)
        violations += int(np.count_nonzero(difference > tolerance))
        compared += int(difference.size)
        maximum = float(difference.max())
        channel_max.append(maximum)
        max_absolute = max(max_absolute, maximum)
        max_relative = max(max_relative, float((difference / np.maximum(np.abs(reference), np.finfo(np.float64).eps)).max()))
    return {'status': 'available' if compared else 'unavailable_no_complete_block',
            'passed': violations == 0 if compared else None, 'compared_values': compared,
            'violating_values': violations, 'max_absolute': max_absolute if compared else None,
            'max_relative_eps_floor': max_relative if compared else None,
            'relative_denominator_floor': float(np.finfo(np.float64).eps),
            'channel_max_absolute': channel_max, 'atol': PARITY_ATOL, 'rtol': PARITY_RTOL,
            'scope': 'Saved forecast Q on the shared completely observed temporal channel-block subset'}


def run(job, reserve_s):
    if OUTPUT.exists() or TABLE.exists():
        raise FileExistsError('Preserve the prior diagnosis and explicitly record any repair')
    evaluation = read_json(HERE / 'evaluation01.json')
    manifest = read_json(HERE / 'prediction_manifest.json')
    for value in (evaluation, manifest):
        if value['status'] != 'complete' or set(value['datasets']) != set(DATASETS):
            raise ValueError('All four fixed evaluations/prediction sets must be complete')
    for receipt in (evaluation['prediction_manifest'], evaluation['selection_seal'], manifest['seal']):
        artifact(receipt['path'], receipt['sha256'])
    names = ('PLAN.md', 'protocol.json', 'provenance.json', 'data_manifest.json', 'selected.json',
             'selection_seal.json', 'test_exposure.json', 'evaluation01.json', 'prediction_manifest.json',
             'diagnose_v16.py', 'evaluate_v16.py', 'model_v16.py', 'runtime_v16.py')
    output = {'schema': 'v16_complete_temporal_block_diagnosis_1', 'status': 'partial', 'job_id': job.id,
              'sources': {name: artifact(HERE / name) for name in names}, 'datasets': {},
              'model_runs': 0, 'new_fits': 0, 'optimizer_updates': 0, 'new_predictions': 0,
              'new_model_selection': False, 'confidence_intervals_added': False,
              'projection': 'Temporal four-position block mean/repetition separately for each channel; not channel PCA',
              'aggregation': 'Sum block energies/counts within each channel; macro over represented channels; mean seed losses, not ensemble',
              'subset_rule': 'Select four fully observed future positions within an origin/channel before target projection; never fill missing targets with zero',
              'limitations': ['Complete-block temporal energy is not the original observed-mask primary macro MSE.',
                             'Subset coverage may vary by channel; absent channels are explicit.',
                             'All four datasets are already exposed development; no independent confirmation.',
                             'Temporal component differences are attribution clues, not unique causes or permission for new fitting.',
                             'F0 has one prediction instance and is not counted as two independent seeds.'],
              'tolerances': {'energy_score_atol': ENERGY_ATOL, 'energy_score_rtol': ENERGY_RTOL,
                             'forecast_q_atol': PARITY_ATOL, 'forecast_q_rtol': PARITY_RTOL}}
    csv_rows = []
    try:
        for dataset in DATASETS:
            budget(job, reserve_s)
            data = load_data(dataset, include_test=True)
            a, b = data['test_a_origins'], data['test_b_origins']
            origins = np.concatenate([a, b]).astype(np.int64)
            indices = {'test_a': np.arange(len(a)), 'test_b': np.arange(len(a), len(origins)),
                       'combined': np.arange(len(origins))}
            c = DATASETS[dataset][0]
            target = np.stack([data['x'][o:o + 48] for o in origins])
            mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
            target4, mask4 = target.reshape(-1, 12, 4, c), mask.reshape(-1, 12, 4, c)
            scopes = {name: coverage(mask4, index) for name, index in indices.items()}
            rows = manifest['datasets'][dataset]['models']
            validate_rows(rows)
            models, arrays, all_sums = {}, {}, {}
            for row in rows:
                budget(job, reserve_s)
                if row['status'] != 'complete':
                    raise ValueError('No method/seed may be omitted from the diagnosis')
                source = evaluation['datasets'][dataset]['models'][row['id']]
                if source['prediction'] != row['prediction'] or source['family'] != row['family'] or source['seed'] != row['seed']:
                    raise ValueError('Prediction identity differs from the evaluated model')
                prediction = prediction_array(row['prediction'], origins, target.shape)
                prediction4 = prediction.reshape(-1, 12, 4, c)
                periods = {name: energy_sums(prediction4, target4, mask4, indices[name]) for name in ('test_a', 'test_b')}
                periods['combined'] = combine_sums(periods['test_a'], periods['test_b'])
                all_sums[row['id']] = periods
                models[row['id']] = {'family': row['family'], 'seed': row['seed'], 'prediction': row['prediction'],
                    'periods': {name: summarize(periods[name], scopes[name], source['periods'][name]) for name in PERIODS}}
                if row['family'] in ('temporal_lora', 'temporal_f0', 'coarse_only', 'direct'):
                    arrays[(row['family'], row['seed'])] = prediction4
            summaries = {}
            for family in ROLES:
                ids = [identifier for identifier, row in models.items() if row['family'] == family]
                primary = evaluation['datasets'][dataset]['role_summary'][family]
                summaries[family] = {'ids': ids, 'aggregation': 'Mean seed losses; F0 is one instance',
                    'periods': {name: summarize(average_sums([all_sums[identifier][name] for identifier in ids]),
                                               scopes[name], primary['periods'][name]) for name in PERIODS}}
            q_checks = {}
            for seed in SEEDS:
                for family in ('temporal_lora', 'temporal_f0', 'coarse_only'):
                    reference = None if family == 'coarse_only' else arrays[('direct', seed)]
                    q_checks[f'{family}_{seed}'] = {'seed': seed, 'candidate': family,
                        'reference': 'zero' if reference is None else 'frozen DIRECT same seed',
                        'periods': {name: q_parity(arrays[(family, seed)], reference, mask4, index) for name, index in indices.items()}}
            differences = {}
            for family in ROLES:
                differences[family + '__vs__full_mse'] = {'candidate': family, 'reference': 'full_mse',
                    'direction': 'Candidate minus reference on the common complete-block subset',
                    'periods': {name: {key: summaries[family]['periods'][name][key] - summaries['full_mse']['periods'][name][key]
                                if summaries[family]['periods'][name][key] is not None else None for key in MACROS}
                                for name in PERIODS}}
            output['datasets'][dataset] = {'data': artifact(data['_path']), 'origin_sha256': array_hash(origins),
                'mask_sha256': array_hash(mask), 'complete_block_mask_sha256': array_hash(mask4.all(axis=2)),
                'columns': [str(name) for name in data['columns']], 'period_origin_counts': {name: len(index) for name, index in indices.items()},
                'coverage': scopes, 'models': models, 'role_summary': summaries,
                'differences_vs_full_mse': differences, 'forecast_q_checks': q_checks}
            for kind, entries in (('single_model', models), ('mean_seed_loss', summaries)):
                for identifier, row in entries.items():
                    family = row['family'] if kind == 'single_model' else identifier
                    for name in PERIODS:
                        value = row['periods'][name]
                        csv_rows.append({'dataset': dataset, 'aggregation': kind, 'id': identifier, 'family': family,
                            'seed': row.get('seed'), 'period': name, **{key: value[key] for key in MACROS},
                            'macro_energy_identity_residual': value['macro_energy_identity_residual'],
                            'valid_channels': scopes[name]['valid_channel_count'], 'all_channels_represented': scopes[name]['all_channels_represented'],
                            'complete_channel_blocks': scopes[name]['complete_channel_blocks'],
                            'complete_block_fraction': scopes[name]['complete_block_fraction'],
                            'retained_observed_target_fraction': scopes[name]['retained_observed_target_fraction'],
                            'original_masked_macro_mse': value['original_masked_macro']['mse']})
            save_json(HERE / 'temporal_diagnosis01_partial.json', output)
            job.heartbeat(dataset + ': temporal complete-block energies; saved predictions only')
            del data, target, mask, target4, mask4, arrays, prediction, prediction4
        with TABLE.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        output.update(status='complete', table=artifact(TABLE), scalar_rows=len(csv_rows), model_instances=60,
            all_energy_identities_passed=all(value['energy_identity_pass'] for unit in output['datasets'].values()
                for model in unit['models'].values() for value in model['periods'].values()),
            all_available_forecast_q_checks_passed=all(value['passed'] for unit in output['datasets'].values()
                for check in unit['forecast_q_checks'].values() for value in check['periods'].values() if value['passed'] is not None))
        save_json(OUTPUT, output)
        return output
    except BaseException:
        save_json(HERE / ('temporal_diagnosis_failure_' + job.id + '.json'),
                  {'status': 'failed', 'job_id': job.id, 'completed_datasets': list(output['datasets']),
                   'error': traceback.format_exc(), 'sources': output['sources'], 'model_runs': 0, 'new_fits': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', required=True, type=float)
    args = parser.parse_args()
    with Job('cpu_analysis', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'temporal complete-block error diagnosis', 'fit_count': 0,
                       'model_runs': 0, 'new_predictions': 0}) as job:
        run(job, args.reserve_s)
