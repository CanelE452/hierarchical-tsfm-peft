"""Check completed artifacts and accounting; no fitting or new inference."""
import ast
import csv
import json
from pathlib import Path
import subprocess

from runtime_ridge import HERE, ROOT, CACHE, Job, read_json, save_json, digest, budget_snapshot


def receipt_check(item):
    assert digest(item['path']) == item['sha256']


if __name__ == '__main__':
    with Job('final_check01', 'cpu_check', 60):
        preserved = read_json(HERE / 'preservation.json')
        assert all(digest(ROOT / path) == sha for path, sha in preserved['files'].items())
        results = [read_json(p) for p in (HERE / 'runs').glob('*/result.json')]
        expected = {(d, f, s, p) for d in ('robin', 'jena') for f in ('ridge_p', 'ridge_raw')
                    for s in (92601, 92602) for p in (.001, .1)}
        assert {(r['spec']['dataset'], r['spec']['family'], r['spec']['seed'], r['spec']['penalty']) for r in results} == expected
        assert len(results) == 16
        for r in results:
            receipt_check({'path': r['checkpoint'], 'sha256': r['checkpoint_sha256']})
            receipt_check(r['best_val_prediction'])
            assert r['status'] == 'complete' and r['optimizer_updates'] == 0 and not r['test_used']
            assert r['solver']['fitted_coefficients'] == 24576 and r['solver']['horizon_solve_count'] == 48
            assert r['solver']['max_relative_normal_equation_residual'] <= 1e-8
            assert r['fp32_fp64_train_mse_difference'] <= 1e-5
        selected = read_json(HERE / 'selected.json')
        for dataset, families in selected['datasets'].items():
            for family, choice in families.items():
                group = [r for r in results if r['spec']['dataset'] == dataset and r['spec']['family'] == family]
                candidates = [(sum(r['best_val_mse'] for r in group if r['spec']['penalty'] == p) / 2, p) for p in (.001, .1)]
                assert choice['penalty'] == min(candidates)[1]
                assert abs(choice['mean_val_mse'] - min(candidates)[0]) < 1e-12
                assert choice['policy_choice'] == ('parent' if choice['mean_val_mse'] >= choice['parent_val_mse'] else family)
        seal, exposure = read_json(HERE / 'selection_seal.json'), read_json(HERE / 'test_exposure.json')
        assert seal['selected_sha256'] == digest(HERE / 'selected.json')
        assert seal['protocol_sha256'] == digest(HERE / 'protocol.json')
        assert seal['reuse_manifest_sha256'] == digest(HERE / 'reuse_manifest.json')
        assert exposure['seal_sha256'] == digest(HERE / 'selection_seal.json')
        assert selected['created_utc'] <= seal['created_utc'] <= exposure['first_opened_utc']
        assert all(digest(path) == sha for path, sha in seal['complete_result_sha256'].items())
        checks = read_json(HERE / 'online_validation.json')
        assert checks['status'] == 'PASS' and len(checks['rows']) == 8
        assert all(row['violating_elements'] == 0 for row in checks['rows'])
        led = read_json(HERE / 'ledger.json')
        fit_jobs = [r for r in led['jobs'] if r['category'] == 'fit']
        assert len(fit_jobs) == 16 and all(r['status'] == 'complete' for r in fit_jobs)
        assert max(r['ended_utc'] for r in fit_jobs) < seal['created_utc']
        summary = read_json(HERE / 'comparison.json')
        for dataset, item in summary['evaluations'].items():
            receipt_check(item)
            evaluation = read_json(item['path'])
            assert evaluation['status'] == 'complete' and evaluation['dataset'] == dataset
            assert len(evaluation['models']) == 23 and len(evaluation['comparisons']) == 15
            for role, ids in evaluation['roles'].items():
                for period in ('test_a', 'test_b', 'combined'):
                    for metric in ('mse', 'mae'):
                        averaged = sum(evaluation['models'][i]['scores'][period][metric] for i in ids) / len(ids)
                        assert abs(averaged - evaluation['role_scores'][role][period][metric]) < 1e-12
            for model in evaluation['models'].values():
                receipt_check(model['prediction'])
                if model['role'] == 'ridge_p':
                    assert model['q_prediction_parity']['pass']
            for c in evaluation['comparisons']:
                for value in c['periods'].values():
                    f = value['full']
                    assert abs(f['delta_mse'] - (f['left_mse'] - f['reference_mse'])) < 1e-12
                    assert abs(f['relative_mse_pct'] - 100 * f['delta_mse'] / f['reference_mse']) < 1e-10
        with (HERE / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                source = summary['datasets'][row['dataset']]['scores'][row['role']][row['period']]
                assert float(row['mse']) == source['mse'] and float(row['mae']) == source['mae']
        cost_decision = read_json(HERE / 'cost_decision.json')
        cost_records = {}
        for dataset, entry in cost_decision['datasets'].items():
            if entry['run_cost']:
                paths = list(HERE.glob(f'cost_{dataset}_*_summary.json'))
                if not paths:
                    guard = read_json(HERE / 'cost_guard.json')
                    assert guard['status'] == 'stopped_budget' and guard['cost_elapsed_s'] < 600
                    receipt_check(guard['partial'])
                    receipt_check(guard['parity'])
                    partial = read_json(guard['partial']['path'])
                    parity = read_json(guard['parity']['path'])
                    assert len(partial['rows']) == guard['completed_rows'] < 78
                    assert parity['status'] == 'PASS' and len(parity['rows']) == 11
                    cost_records[dataset] = {'status': 'incomplete_budget', 'rows': len(partial['rows']),
                                             'planned_rows': 78, 'speed_ranking': 'unconfirmed',
                                             'guard_sha256': digest(HERE / 'cost_guard.json')}
                    continue
                assert len(paths) == 1
                cost = read_json(paths[0])
                assert cost['status'] == 'complete' and cost['rows'] == 78 and len(cost['groups']) == 12
                receipt_check(cost['rows_receipt'])
                receipt_check(cost['parity_receipt'])
                parity = read_json(cost['parity_receipt']['path'])
                assert parity['status'] == 'PASS' and len(parity['rows']) == 11
                for row in parity['rows']:
                    assert row['restore_full_val']['pass_'] and row['batch1_vs_batch4']['pass_']
                cost_records[dataset] = {'path': str(paths[0]), 'sha256': digest(paths[0])}
        assert any(j['label'] == 'jena_eval01' and j['status'] == 'failed' and 'PermissionError' in j['error'] for j in led['jobs'])
        assert any(j['label'] == 'jena_eval02' and j['status'] == 'complete' for j in led['jobs'])
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        new_files = [p for directory in (HERE, ROOT / 'research/tsfm_peft_followup_diagnosis_20260929')
                     for p in directory.rglob('*') if p.is_file()]
        assert not any(p.suffix.lower() in ('.npz', '.npy', '.pt', '.safetensors', '.zip', '.parquet') for p in new_files)
        budget = budget_snapshot()
        assert budget['fit_attempts'] == 16 and budget['gpu_seconds'] < 1800
        assert budget['cpu_analysis_seconds'] < 1800 and budget['cpu_check_seconds'] < 300
        assert budget['cost_gpu_seconds'] < 600 and budget['storage_bytes'] < 1024**3
        assert budget['active'] == ['final_check01']
        save_json(HERE / 'final_checks.json', {'status': 'PASS_ARTIFACTS_WITH_DOCUMENTED_LIMITATIONS',
                  'preserved_prior_research_files': len(preserved['files']), 'real_fits': 16,
                  'recovery_fits': 0, 'internal_horizon_solves': 16 * 48, 'new_optimizer_updates': 0,
                  'both_selections_precede_evaluation': True, 'selected_online_models_checked': 8,
                  'evaluation_models_per_dataset': 23, 'cost': cost_records,
                  'scope': 'Execution-team artifact/numerical verification, not independent replication or scientific success',
                  'source_sha256': {p.name: digest(p) for p in HERE.glob('*.py')},
                  'comparison_sha256': digest(HERE / 'comparison.json'), 'selection_sha256': digest(HERE / 'selected.json')})
    save_json(HERE / 'budget_final.json', budget_snapshot())
    print(json.dumps(read_json(HERE / 'final_checks.json')), flush=True)
