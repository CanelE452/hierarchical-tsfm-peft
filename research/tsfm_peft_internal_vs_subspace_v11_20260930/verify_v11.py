from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from runtime_v11 import HERE, DATASETS, Job, budget_snapshot, digest, load_data, read_json, save_json, verify_provenance
from gamma_v11 import check_receipt, receipt


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-10):
    require(np.allclose(actual, expected, atol=tolerance, rtol=tolerance), message)


def verify_fits(selected):
    protocol = read_json(HERE / 'protocol.json')
    rows = selected['all_neural_runs']
    require(len(rows) == len(protocol['fits']) == 40, 'All forty neural configurations required')
    output = []
    for item in rows:
        result = read_json(HERE / 'runs' / item['id'] / 'result.json')
        require(result['status'] == 'complete', 'Incomplete fit')
        check_receipt({'path': result['checkpoint'], 'sha256': result['checkpoint_sha256']})
        check_receipt(result['restart'])
        require(result['frozen_parameters_verified']['unchanged_parameters_and_buffers'], 'Frozen state changed')
        require(result['frozen_parameters_verified']['no_grad'], 'Frozen parameters have gradients')
        require(result['replay_error'] <= 1e-6, 'Same-path restore replay failed')
        require(result['actual_updates'] > 0, 'Fit did not update')
        require(sum(result['changed_parameter_scalars'].values()) > 0, 'No trainable state changed')
        family = result['spec']['family']
        expected_parameters = (24624 if family == 'direct_nlinear' else
                               result['spec']['channels'] * result['spec']['latent'] if family == 'b_learned_u' else 294912)
        require(result['trainable_parameters'] == expected_parameters, 'Registered training scope differs from protocol')
        if result['spec']['family'] == 'b_learned_u':
            require(result['actual_u_check']['actual_u_max_update'] > 0, 'Constrained U unchanged')
            require(result['actual_u_check']['gram_max_error'] <= 1e-5, 'U lost orthogonality')
        initial = Path(result.get('initial_checkpoint', str(Path(result['checkpoint']).with_name('initial.pt'))))
        require(initial.is_file(), 'Initial checkpoint missing')
        output.append({'id': item['id'], 'checkpoint': receipt(result['checkpoint']),
                       'initial_checkpoint': receipt(initial), 'selected_epoch': result['selected_epoch'],
                       'upper_epoch_limit_improving': result['upper_epoch_limit_improving'],
                       'actual_updates': result['actual_updates'], 'trainable_parameters': result['trainable_parameters']})
    for dataset, families in selected['selected'].items():
        for family, choice in families.items():
            if family == 'direct_nlinear' and dataset != 'hog':
                continue
            candidates = [row for row in rows if row['spec']['dataset'] == dataset and row['spec']['family'] == family]
            means = [(lr, np.mean([row['best_val_mse'] for row in candidates if row['spec']['lr'] == lr]))
                     for lr in (1e-4, 1e-3)]
            chosen = min(means, key=lambda pair: pair[1])
            require(choice['lr'] == chosen[0], 'VAL LR selection differs')
            close(choice['mean_best_val_mse'], chosen[1], 'VAL mean differs')
    return output


def verify_scores(evaluation, job):
    checked = []
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        origins_a, origins_b = data['test_a_origins'], data['test_b_origins']
        origins = np.concatenate((origins_a, origins_b))
        target = np.stack([data['x'][origin:origin + 48] for origin in origins]).astype(np.float64)
        mask = np.stack([data['finite'][origin:origin + 48] for origin in origins]).astype(bool)
        indices = {'test_a': np.arange(len(origins_a)), 'test_b': np.arange(len(origins_a), len(origins)),
                   'combined': np.arange(len(origins))}
        report = evaluation['datasets'][dataset]
        require(len(report['models']) == 19, 'Missing prescribed evaluation instance')
        independently_scored = {}
        for identifier, row in report['models'].items():
            job.check_limits()
            with np.load(check_receipt(row['prediction']), allow_pickle=False) as arrays:
                require(np.array_equal(arrays['origins'], origins), 'Prediction origin order differs')
                prediction = arrays['prediction'].astype(np.float64)
            require(prediction.shape == target.shape and np.isfinite(prediction).all(), 'Prediction shape/finiteness')
            independently_scored[identifier] = {}
            for period, index in indices.items():
                channel_scores = []
                for channel in range(target.shape[-1]):
                    keep = mask[index, :, channel]
                    error = prediction[index, :, channel][keep] - target[index, :, channel][keep]
                    require(len(error) > 0, 'Unobserved evaluation channel')
                    channel_scores.append((np.mean(error ** 2), np.mean(np.abs(error)), np.mean(error)))
                macro = np.mean(channel_scores, axis=0)
                published = row['periods'][period]
                close(macro, [published['mse'], published['mae'], published['signed_mean_error']], 'Direct array macro differs')
                independently_scored[identifier][period] = macro
            checked.append({'dataset': dataset, 'id': identifier, 'prediction_sha256': row['prediction']['sha256']})
        for role, summary in report['role_summary'].items():
            for period in indices:
                expected = np.mean([independently_scored[identifier][period] for identifier in summary['ids']], axis=0)
                actual = summary['periods'][period]
                close(expected, [actual['mse'], actual['mae'], actual['signed_mean_error']], 'Seed-loss mean differs')
        require(all(item['pass'] for item in report['A_Q_projection_parity'].values()), 'A Q projection preservation failed')
        job.heartbeat({'phase': 'saved_prediction_numeric_verification', 'dataset': dataset, 'instances': len(checked)})
    return checked


def verify_cost():
    result = {}
    for device, count in (('cuda', 621), ('cpu', 36)):
        rows = read_json(HERE / f'cost_{device}_rows.json')
        grid = read_json(HERE / f'cost_{device}_grid.json')
        require(rows['status'] == 'complete' and len(rows['rows']) == count, 'Cost campaign incomplete')
        require([row['key'] for row in rows['rows']] == [row['key'] for row in grid['rows']], 'Cost grid/order differs')
        for row in rows['rows']:
            require(row['passes'] == 10 and row['warmup_calls'] == 10, 'Timing repetition contract differs')
            require(len(row['seconds_per_24_origins']) == 10 and min(row['seconds_per_24_origins']) > 0, 'Invalid raw timing')
            require(row['model']['merged_export_pruned_vs_original']['pass'], 'Deployment parity failed')
            require(all(item['parity']['pass'] for item in row['model']['configuration_parity']), 'Chunk parity failed')
            if device == 'cpu':
                require(row['peak_allocated_bytes'] is None and row['peak_reserved_bytes'] is None, 'CPU GPU memory must be N/A')
            else:
                require(0 <= row['peak_allocated_bytes'] <= row['peak_reserved_bytes'], 'Invalid CUDA peaks')
        result[device] = {'rows': count, 'source': receipt(HERE / f'cost_{device}_rows.json'),
                          'sessions': rows['session_ids']}
    return result


def run(job):
    verify_provenance()
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    selected = read_json(HERE / 'selected.json')
    require(not selected['test_used_for_selection'], 'TEST used for neural selection')
    fits = verify_fits(selected)
    gamma = read_json(HERE / 'gamma_selection.json')
    verification = read_json(HERE / 'gamma_verification.json')
    require(len(gamma['rows']) == 12 and gamma['status'] == 'complete' and verification['pass'], 'Gamma incomplete')
    require(all(row['diagnostics']['valid'] for row in gamma['rows']), 'Gamma invalid')
    seal, exposure = read_json(HERE / 'selection_seal.json'), read_json(HERE / 'test_exposure.json')
    require(seal['created_utc'] <= exposure['first_access_utc'], 'Exposure preceded seal')
    require(exposure['seal_sha256'] == digest(HERE / 'selection_seal.json'), 'Selection seal changed')
    evaluation = read_json(HERE / 'evaluation01.json')
    require(evaluation['status'] == 'complete' and not evaluation['independent_confirmation'], 'Evaluation scope invalid')
    numeric = verify_scores(evaluation, job)
    costs = verify_cost()
    ledger = read_json(HERE / 'ledger.json')
    fitted = [row for row in ledger['jobs'] if row['category'] in ('neural_fit', 'coefficient_fit')]
    require(len(fitted) <= 60, 'Fit budget exceeded')
    require(all(row['status'] != 'running' for row in fitted), 'Fitting still active')
    require(all(row['ended_utc'] <= exposure['first_access_utc'] for row in fitted), 'Fitting continued after TEST')
    synthetic = [row for row in ledger['jobs'] if row.get('synthetic')]
    require(len(synthetic) <= 6 and all(row['updates'] <= 10 for row in synthetic), 'Synthetic budget exceeded')
    snapshot = budget_snapshot()
    for category in ('gpu', 'cpu_analysis', 'cpu_check'):
        require(snapshot[category + '_seconds'] <= snapshot['limits'][category + '_seconds'], 'Time budget exceeded')
    require(snapshot['storage_bytes'] <= snapshot['limits']['storage_bytes'], 'Storage budget exceeded')
    return {'status': 'passed', 'fits': fits, 'saved_prediction_checks': numeric, 'costs': costs,
            'budget_snapshot': snapshot, 'gamma_fits': 12, 'independent_reproduction': False,
            'scientific_superiority_asserted': False, 'scope': 'Own saved-array numeric/provenance/contract verification; no new models or fitting',
            'sources': {name: receipt(HERE / name) for name in ('selected.json', 'gamma_selection.json',
                'selection_seal.json', 'evaluation01.json', 'cost_summary.json', 'report_values.json')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'final_checks.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier verification; use an explicit correction filename')
    with Job(args.job, category='cpu_check', reserve_seconds=300,
             metadata={'scope': 'final saved-output verification', 'fits': 0}) as job:
        try:
            save_json(args.output, run(job))
        except BaseException as error:
            save_json(args.output.with_name(args.output.stem + '_failed.json'), {'status': 'failed', 'error': repr(error)})
            raise
    print(args.output)


if __name__ == '__main__':
    main()
