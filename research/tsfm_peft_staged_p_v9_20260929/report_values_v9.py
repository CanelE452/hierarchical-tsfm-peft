"""Collect traceable measured values for the decision and short method update."""
import argparse
import csv
import json
from pathlib import Path

from runtime_v9 import HERE, ROOT, Job, budget_snapshot, digest, save_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def receipt(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    target = HERE / 'report_values.json'
    if target.exists():
        raise RuntimeError('Preserve the existing report value artifact')
    with Job(args.name, category='cpu_analysis', reserve_seconds=60):
        protocol = read(HERE / 'protocol.json')
        report = {'status': 'measured_and_derived_from_saved_artifacts', 'sources': {}, 'datasets': {},
                  'relative_change_definition': '(candidate/reference-1)*100 percent, not percentage points',
                  'scope': 'Two exposed development datasets; fixed seed-loss means, not ensembles or independent confirmation'}
        for dataset in ('robin', 'jena'):
            evaluation_path = HERE / f'{dataset}_eval01.json'
            selected_path = HERE / f'selected_{dataset}.json'
            evaluation, selected = read(evaluation_path), read(selected_path)
            assert evaluation['status'] == 'complete' and not selected['test_used_for_selection']
            report['sources'][dataset + '_evaluation'] = receipt(evaluation_path)
            report['sources'][dataset + '_selection'] = receipt(selected_path)
            training = []
            for spec in (s for s in protocol['fits'] if s['dataset'] == dataset):
                result_path = HERE / 'runs' / spec['id'] / 'result.json'
                result = read(result_path)
                assert result['status'] == 'complete'
                parent_path = Path(result['parent_receipt']['result']['path'])
                if not parent_path.is_absolute():
                    parent_path = ROOT / parent_path
                assert digest(parent_path) == result['parent_receipt']['result']['sha256']
                parent_result = read(parent_path)
                training.append({
                    'id': result['id'], 'family': spec['family'], 'seed': spec['seed'],
                    'parent_id': result['parent_receipt']['id'], 'parent_selected_epoch': parent_result['selected_epoch'],
                    'initial_val_mse': result['initial_val_mse'],
                    'basic_epoch': result['selected_epoch'], 'basic_val_mse': result['best_val_mse'],
                    'best_trained_epoch': result['best_trained']['selected_epoch'],
                    'best_trained_val_mse': result['best_trained']['val_mse'],
                    'alpha': result['alpha_selected']['alpha'], 'alpha_val_mse': result['alpha_selected']['val_mse'],
                    'epochs_completed': result['epochs_completed'], 'steps': result['total_steps'],
                    'final_stale': result['final_stale'], 'cap_reached_while_improving': result['cap_reached_while_improving'],
                    'initial_train_probe_mse': result['initial_train_probe']['mse'],
                    'selected_train_probe_mse': result['selected_train_probe']['mse'],
                    'new_trainable_parameters': result['trainable_parameters'],
                    'cumulative_fitted_adapter_parameters': result['cumulative_fitted_adapter_parameters'],
                    'total_parameters': result['total_parameters'],
                    'parameter_tensor_bytes_fp32': result['total_parameters'] * 4,
                    'parent_historical_fit_elapsed_s': parent_result['elapsed_s'],
                    'new_head_online_fit_and_val_elapsed_s': result['head_fit_online_elapsed_s'],
                    'new_alpha_selection_elapsed_s': result['alpha_selection_elapsed_s'],
                    'new_head_attempt_elapsed_s': result['elapsed_s'],
                    'historical_parent_plus_new_attempt_s': parent_result['elapsed_s'] + result['elapsed_s'],
                    'timing_scope': 'Historical procedure elapsed sum per alternative; not controlled same-session learning speed or incremental GPU-job total',
                    'result': receipt(result_path), 'parent_result': receipt(parent_path)})
            roles = {role: {period: {k: value[k] for k in ('mse', 'mae', 'run_ids')}
                            for period, value in periods.items()}
                     for role, periods in evaluation['role_scores'].items()}
            changes = {}
            for role in ('staged_p', 'staged_raw', 'staged_p_alpha', 'staged_raw_alpha'):
                changes[role] = {}
                for reference in ('level_res', 'pq_split', 'raw64', 'staged_raw', 'staged_raw_alpha', 'f0', 'lora', 'direct_nlinear'):
                    left, right = roles[role]['combined']['mse'], roles[reference]['combined']['mse']
                    changes[role][reference] = {'candidate_mse': left, 'reference_mse': right,
                                               'absolute_mse_difference': left - right,
                                               'relative_mse_change_pct': (left / right - 1) * 100}
            report['datasets'][dataset] = {'metric_status': '[확인]', 'roles': roles,
                'training': training, 'combined_changes': changes,
                'comparisons': evaluation['comparisons'], 'space_scores': evaluation['role_space_scores'],
                'q_preservation': evaluation['q_preservation'],
                'q_prediction_parity': evaluation['q_prediction_parity']}
        decision = read(HERE / 'cost_decision.json')
        report['sources']['cost_decision'] = receipt(HERE / 'cost_decision.json')
        report['cost'] = {'decision': decision, 'results': {}}
        for dataset, choice in decision['datasets'].items():
            if choice['run_cost']:
                path = HERE / f'cost_{dataset}_summary.json'
                report['cost']['results'][dataset] = {'receipt': receipt(path), 'summary': read(path)}
        training_path = HERE / 'training_summary.csv'
        if training_path.exists():
            raise FileExistsError('Preserve existing training table')
        columns = ['id', 'parent_id', 'seed', 'parent_selected_epoch', 'basic_epoch', 'best_trained_epoch',
                   'alpha', 'epochs_completed', 'steps', 'initial_val_mse', 'basic_val_mse',
                   'initial_train_probe_mse', 'selected_train_probe_mse', 'new_trainable_parameters',
                   'cumulative_fitted_adapter_parameters', 'total_parameters', 'parent_historical_fit_elapsed_s',
                   'new_head_online_fit_and_val_elapsed_s', 'new_alpha_selection_elapsed_s',
                   'new_head_attempt_elapsed_s', 'historical_parent_plus_new_attempt_s']
        with training_path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['dataset'] + columns)
            writer.writeheader()
            for dataset, value in report['datasets'].items():
                for row in value['training']:
                    writer.writerow({'dataset': dataset, **{key: row[key] for key in columns}})
        report['training_table'] = receipt(training_path)
        report['budget_at_report'] = budget_snapshot()
        report['budget_scope'] = 'Snapshot before final verification/publication; final closed ledger/publication receipt gives final usage'
        save_json(target, report)
        print(json.dumps({'report': str(target), 'datasets': list(report['datasets'])}), flush=True)


if __name__ == '__main__':
    main()
