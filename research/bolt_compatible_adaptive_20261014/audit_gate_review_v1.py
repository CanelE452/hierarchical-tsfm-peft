"""Recheck the gate critique using existing CSV/JSON metadata only."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    output = HERE / 'GATE_REVIEW_METADATA_AUDIT_v1.json'
    if output.exists():
        raise FileExistsError(output)
    table = HERE / 'reports/first_and_dense_v3/tables/COMPARISON.csv'
    with table.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    counts = []
    for method in ('PERSIST', 'RAW_BOLT'):
        for split in ('tune', 'val'):
            row = next(r for r in rows if r['dataset'] == 'robin' and r['method'] == method
                       and r['split'] == split and r['metric_scale'] == 'normalized')
            errors = json.loads(row['per_channel_mse'])
            observed = json.loads(row['observed_counts'])
            active = [i for i, n in enumerate(observed) if n > 0]
            counts.append({'method': method, 'split': split, 'macro_mse': float(row['mse']),
                           'macro_mae': float(row['mae']), 'channels': len(active),
                           'exact_zero_indices': [i for i in active if errors[i] == 0.],
                           'at_most_1e_minus_8_indices': [i for i in active if errors[i] <= 1e-8],
                           'observed_counts': observed})
    expected_zero = (10, 1, 0, 0)
    expected_small = (10, 1, 3, 1)
    assert tuple(len(r['exact_zero_indices']) for r in counts) == expected_zero
    assert tuple(len(r['at_most_1e_minus_8_indices']) for r in counts) == expected_small
    prior = ROOT / 'research/smallk_predictive_alignment_20261013/run_dev_v1/RESULT.json'
    value = json.loads(prior.read_text(encoding='utf-8-sig'))
    pointwise = []
    for case in value['cases']:
        if case.get('architecture') != 'ANCHOR_POINTWISE':
            continue
        training = case['training']
        checkpoint = training['selected_checkpoint']
        path = (ROOT / checkpoint['path']).resolve()
        forecast = checkpoint['forecast']
        pointwise.append({'dataset': case['dataset'], 'seed': case['seed'],
                          'K': case['K'], 'L': case['L'], 'H': case['H'],
                          'updates': training['updates'], 'selected_update': training['selected_update'],
                          'checkpoint_metadata': checkpoint['path'], 'checkpoint_exists': path.is_file(),
                          'checkpoint_content_opened': False,
                          'forecast_score_metadata': forecast})
    assert len(pointwise) == 4 and all(row['checkpoint_exists'] for row in pointwise)
    result = {'status': 'PASS', 'scope': 'Existing CSV/terminal JSON metadata only; no scientific rerun',
              'inputs': [record(table), record(prior)], 'zero_error_definition_check': counts,
              'existing_pointwise_predictive_cases': pointwise,
              'same_pattern_counterexample': {'type': 'logical counterexample, not a data fit',
                  'construction': 'X[t,c]=s[t] for all c; self already observes s; K=1 E selects one channel and D copies it',
                  'implication': 'Extra joint predictive gain is not necessary for redundant-channel compression'},
              'new_training_runs': 0, 'model_or_checkpoint_content_opened': False,
              'prediction_NPZ_opened': False, 'protected_TEST_opened': False,
              'gate_changed_or_budget_reset': False}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'PASS', 'existing_pointwise_cases': len(pointwise),
                      'exact_zero_counts': list(expected_zero), 'at_most_1e_minus_8_counts': list(expected_small),
                      'new_training_runs': 0, 'output': str(output)}))


if __name__ == '__main__':
    main()
