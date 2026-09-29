"""Export small source-linked comparison tables and the conditional cost decision."""
import csv
from runtime_ridge import HERE, Job, read_json, save_json, digest


def export_csv(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    with Job('summary01', 'cpu_analysis', 30):
        evaluations = {'robin': HERE / 'robin_eval01.json', 'jena': HERE / 'jena_eval02.json'}
        summary = {'evaluations': {}, 'datasets': {}, 'seed_loss_mean_not_ensemble': True,
                   'independent_confirmation': False, 'scope': 'exposed development, no channel/period/model reselection'}
        metrics, seeds, channels, comparisons, training = [], [], [], [], []
        for dataset, path in evaluations.items():
            value = read_json(path)
            assert value['status'] == 'complete'
            summary['evaluations'][dataset] = {'path': str(path), 'sha256': digest(path)}
            summary['datasets'][dataset] = {'scores': value['role_scores'], 'comparisons': value['comparisons'],
                                            'space_scores': value['role_space_scores']}
            for role, periods in value['role_scores'].items():
                for period, score in periods.items():
                    metrics.append({'dataset': dataset, 'role': role, 'period': period,
                                    'mse': score['mse'], 'mae': score['mae'], 'mean_signed_error': score['mean_signed_error']})
                    for column, mse, mae, bias in zip(value['channels'], score['channel_mse'], score['channel_mae'], score['channel_mean_signed_error']):
                        channels.append({'dataset': dataset, 'role': role, 'period': period, 'channel': column,
                                         'mse': mse, 'mae': mae, 'mean_signed_error': bias})
            for identifier, row in value['models'].items():
                for period, score in row['scores'].items():
                    seeds.append({'dataset': dataset, 'role': row['role'], 'run': identifier,
                                  'seed': row['spec'].get('seed'), 'period': period,
                                  'mse': score['mse'], 'mae': score['mae']})
            for row in value['comparisons']:
                for period, comparison in row['periods'].items():
                    full = comparison['full']
                    interval = comparison['conditional_block95']
                    comparisons.append({'dataset': dataset, 'left': row['left_role'], 'reference': row['right_role'],
                                        'period': period, 'left_mse': full['left_mse'], 'reference_mse': full['reference_mse'],
                                        'delta_mse': full['delta_mse'], 'relative_mse_pct': full['relative_mse_pct'],
                                        'relative95_low': interval['relative_to_reference_pct'][0] if interval else None,
                                        'relative95_high': interval['relative_to_reference_pct'][1] if interval else None,
                                        'delta_mae': full['delta_mae']})
        cache = read_json(HERE / 'parent_cache.json')
        for path in sorted((HERE / 'runs').glob('*/result.json')):
            result = read_json(path)
            spec = result['spec']
            base = cache['datasets'][spec['dataset']][str(spec['seed'])]
            training.append({**spec, 'parent_train_mse': base['train']['score']['mse'],
                             'ridge_train_mse': result['train_score']['mse'], 'parent_val_mse': base['val']['score']['mse'],
                             'ridge_val_mse': result['val_score']['mse'],
                             'normal_equation_residual': result['solver']['max_relative_normal_equation_residual'],
                             'fit_elapsed_seconds': result['elapsed_s']})
        for name, rows in [('metrics.csv', metrics), ('seed_metrics.csv', seeds), ('channel_metrics.csv', channels),
                           ('comparisons.csv', comparisons), ('training_summary.csv', training)]:
            export_csv(name, rows)
        save_json(HERE / 'comparison.json', summary)
        save_json(HERE / 'cost_decision.json', {'datasets': {
            'robin': {'run_cost': False, 'reason': 'Both selected nonzero corrections worsen whole-VAL and combined development MSE; retain the parent policy. No useful new Robin accuracy result to justify repeated cost.'},
            'jena': {'run_cost': True, 'reason': 'Nonzero P/RAW selected on VAL; P reduces parent and matched-RAW MSE on both periods. Measure practical cost while retaining the inconclusive v8/v9 increment and LoRA/direct-linear counterexamples.'}},
            'evaluations': summary['evaluations'], 'selected_sha256': digest(HERE / 'selected.json'),
            'cost_gpu_cap_total': 600, 'new_method_selection': False})
