"""One CPU audit of saved evaluation, selection, update counts and repeated cost."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'research/tsfm_peft_development_20260926'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar_scores(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('shape mismatch')
    cmse, cmae, counts = [], [], []
    for c in range(prediction.shape[-1]):
        errors = [float(p) - float(y) for p, y, valid in zip(
            prediction[..., c].flat, target[..., c].flat, mask[..., c].flat) if valid]
        if not errors:
            raise ValueError('no observed target for a channel')
        counts.append(len(errors))
        cmse.append(math.fsum(e * e for e in errors) / len(errors))
        cmae.append(math.fsum(abs(e) for e in errors) / len(errors))
    return {'mse': math.fsum(cmse) / len(cmse), 'mae': math.fsum(cmae) / len(cmae),
            'channel_mse': cmse, 'channel_mae': cmae, 'target_counts': counts,
            'origins': len(prediction)}


def compare_scores(actual, expected):
    errors = {}
    for name in ['mse', 'mae']:
        errors[name] = abs(actual[name] - expected[name])
    for name in ['channel_mse', 'channel_mae']:
        assert len(actual[name]) == len(expected[name])
        errors[name] = max(abs(a - b) for a, b in zip(actual[name], expected[name]))
    assert actual['target_counts'] == expected['target_counts']
    assert actual['origins'] == expected['origins']
    assert max(errors.values()) < 1e-10, errors
    return errors


def load_data(report):
    contract_path = V1 / 'data_contract.json'
    assert digest(contract_path) == report['data_contract_sha256']
    contract = read(contract_path)
    datasets = {}
    for dataset, key in [('electricity', 'electricity_first32'), ('bull', 'bdg2_bull_office')]:
        path = ROOT / contract['datasets'][key]['npz']
        assert digest(path) == report['data_sha256'][dataset]
        with np.load(path, allow_pickle=False) as archive:
            datasets[dataset] = {key: archive[key] for key in archive.files}
    return datasets


def find_result(fit):
    paths = [folder / 'runs' / fit / 'result.json' for folder in [HERE, V1]]
    matches = [path for path in paths if path.exists()]
    assert len(matches) == 1, (fit, matches)
    return matches[0]


def check_evaluation(report, datasets):
    checked, groups, row_keys = [], {}, set()
    for row in report['rows']:
        dataset, period = row['dataset'], row['period']
        key = (dataset, period, row['key'], row['seed'])
        assert key not in row_keys, ('duplicate evaluation row', key)
        row_keys.add(key)
        data = datasets[dataset]
        origins = data['dev_origins' if period == 'dev' else 'eval_' + period + '_origins']
        receipt = row['prediction']
        path = Path(receipt['path'])
        assert digest(path) == receipt['sha256']
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive['prediction']
            assert np.array_equal(archive['origins'], origins)
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['targetmask'][o:o + 48] for o in origins]).astype(bool)
        assert prediction.shape == target.shape and np.isfinite(prediction).all()
        full = scalar_scores(prediction, target, mask)
        full_errors = compare_scores(full, row['scores']['full'])
        half_errors = []
        for part, expected in zip([slice(0, len(origins) // 2), slice(len(origins) // 2, None)], row['scores']['halves']):
            try:
                actual = scalar_scores(prediction[part], target[part], mask[part])
            except ValueError:
                assert expected['status'] == 'unavailable'
                half_errors.append({'unavailable': True})
            else:
                assert expected['status'] == 'available'
                half_errors.append(compare_scores(actual, expected))
        label = (f"{row['arm']}:{row['ed_mode']}:k{row['latent']}"
                 if row['arm'] in ('residual', 'raw_bypass') else row['arm'])
        groups.setdefault((dataset + '/' + period, label), []).append((row, full))
        checked.append({'dataset': dataset, 'period': period, 'fit': row['fit'], 'key': row['key'],
                        'prediction_sha256': receipt['sha256'], 'full_absolute_errors': full_errors,
                        'halves_absolute_errors': half_errors})
    mean_checks = []
    for (period_key, label), entries in groups.items():
        rows, scores = zip(*entries)
        if rows[0]['fit'] is not None:
            assert sorted(row['seed'] for row in rows) == [92601, 92602], (period_key, label)
            assert len({(row['arm'], row['ed_mode'], row['latent']) for row in rows}) == 1
        else:
            assert len(rows) == 1 and rows[0]['seed'] is None
        expected = report['comparisons'][period_key]['mean_scores'][label]
        n = len(scores)
        mse = math.fsum(score['mse'] for score in scores) / n
        mae = math.fsum(score['mae'] for score in scores) / n
        cmse = [math.fsum(score['channel_mse'][c] for score in scores) / n for c in range(len(scores[0]['channel_mse']))]
        cmae = [math.fsum(score['channel_mae'][c] for score in scores) / n for c in range(len(scores[0]['channel_mae']))]
        errors = [abs(mse - expected['seed_loss_mean_mse']), abs(mae - expected['seed_loss_mean_mae'])]
        errors.extend(abs(a - b) for a, b in zip(cmse, expected['mean_channel_mse']))
        errors.extend(abs(a - b) for a, b in zip(cmae, expected['mean_channel_mae']))
        assert max(errors) < 1e-10
        assert {(x['seed'], x['fit']) for x in expected['per_seed']} == {(row['seed'], row['fit']) for row in rows}
        mean_checks.append({'period': period_key, 'key': label, 'seeds': [row['seed'] for row in rows],
                            'mse': mse, 'mae': mae, 'max_absolute_mean_error': max(errors)})
    expected_groups = {(period, key) for period, obj in report['comparisons'].items() for key in obj['mean_scores']}
    assert set(groups) == expected_groups
    return checked, mean_checks


def check_selection(selection, report):
    assert selection['selection'] == report['selection']
    checked = []
    for dataset, selected in selection['selection'].items():
        variants_by_arm = {}
        for arm in ['residual', 'raw_bypass']:
            choice = selected[arm]
            computed = {}
            for variant, ids in choice['variant_fits'].items():
                results = []
                for fit in ids:
                    path = find_result(fit)
                    assert digest(path) == selection['result_hashes'][fit] == report['result_hashes'][fit]
                    result = read(path)
                    assert result['status'] == 'complete' and result['dataset'] == dataset and result['arm'] == arm
                    assert f"{result.get('ed_mode', 'current')}:k{result['latent']}" == variant
                    receipt = selection['validation_receipts'][fit]
                    assert digest(Path(receipt['path'])) == receipt['sha256']
                    results.append(result)
                assert sorted(result['seed'] for result in results) == [92601, 92602]
                computed[variant] = math.fsum(result['best_val_mse'] for result in results) / 2
                assert abs(computed[variant] - choice['variant_scores'][variant]) < 1e-12
            assert set(computed) == set(choice['available_variants'])
            assert choice['variant'] in computed and computed[choice['variant']] == min(computed.values())
            assert choice['fits'] == choice['variant_fits'][choice['variant']]
            assert abs(choice['mean_best_val_mse'] - computed[choice['variant']]) < 1e-12
            variants_by_arm[arm] = set(computed)
            checked.append({'dataset': dataset, 'arm': arm, 'variants': sorted(computed),
                            'selected': choice['variant'], 'selected_mean_VAL': computed[choice['variant']]})
        assert variants_by_arm['residual'] == variants_by_arm['raw_bypass']
        assert selected['selection_opportunity']['status'] == 'matched'
        assert selected['selection_opportunity']['final_fair_selection_comparison'] is True
    return checked


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def timing_summary(values):
    return {'median': statistics.median(values), 'q25': quantile(values, .25),
            'q75': quantile(values, .75), 'min': min(values), 'max': max(values)}


def check_cost(cost, specs, datasets):
    assert cost.get('status', 'complete') == 'complete'
    assert cost['specs_sha256'] == digest(HERE / 'cost_specs.json')
    assert cost['warmup_calls'] == 10 and cost['passes_per_order_block'] == 20 and cost['order_blocks'] == 3
    assert cost['torch_threads'] == 4 and cost['precision'] == 'float32'
    assert len(specs) == 26 and len(cost['rows']) == 156
    by_id = {spec['id']: spec for spec in specs}
    assert len(by_id) == 26
    expected = {(ident, batch, block) for ident in by_id for batch in [1, 4] for block in range(3)}
    found, max_summary_error, merge_ids, replay_ids = set(), 0.0, set(), set()
    for dataset, data in datasets.items():
        origins = data['val_origins'][np.linspace(0, len(data['val_origins']) - 1, 24, dtype=int)]
        assert cost['val_origins'][dataset] == origins.tolist()
    for row in cost['rows']:
        key = (row['id'], row['batch_origins'], row['block'])
        assert key in expected and key not in found
        found.add(key)
        spec = by_id[row['id']]
        assert row['dataset'] == spec['dataset'] and row['seed'] == spec.get('seed')
        assert row['roles'] == spec['roles']
        values = row['seconds_per_24_origins']
        assert len(values) == 20 and all(math.isfinite(x) and x > 0 for x in values)
        for name, derived in [('milliseconds_per_origin', [x * 1000 / 24 for x in values]),
                              ('origins_per_second', [24 / x for x in values])]:
            summary = timing_summary(derived)
            errors = [abs(summary[k] - row[name][k]) for k in summary]
            assert max(errors) < 1e-9
            max_summary_error = max(max_summary_error, *errors)
        assert row['baseline_cuda_bytes'] == {'allocated': 0, 'reserved': 0}
        assert row['peak_reserved_bytes'] >= row['peak_allocated_bytes'] >= row['resident_allocated_bytes']
        assert row['deployed_parameter_bytes'] == 4 * row['total_deployed_parameters']
        assert row['task_channels'] == datasets[row['dataset']]['x'].shape[1]
        if spec.get('checkpoint'):
            assert digest(ROOT / spec['checkpoint']) == spec['checkpoint_sha256']
            error = row['saved_validation_replay_max_abs']
            assert math.isfinite(error) and error >= 0
            replay_ids.add(spec['id'])
            result = read(ROOT / spec['result_file'])
            assert row['registered_trainable_parameters_before_merge'] == result['trainable_parameters']
        if spec['arm'] == 'lora':
            parity = row['merge_parity']
            assert parity['pass'] is True and parity['origins'] == 24
            assert parity['atol'] == 1e-5 and parity['rtol'] == 1e-4
            assert math.isfinite(parity['max_abs_error']) and parity['max_abs_error'] >= 0
            merge_ids.add(spec['id'])
    assert found == expected and len(replay_ids) == 24 and len(merge_ids) == 4
    return {'rows': len(found), 'specs': len(specs), 'timed_passes': len(found) * 20,
            'max_timing_summary_absolute_error': max_summary_error,
            'checkpoint_replay_ids': sorted(replay_ids), 'merged_lora_ids': sorted(merge_ids),
            'replay_scope': 'Completed benchmark executed assert_close against saved VAL and before/after LoRA merge; this CPU audit checks receipts/results, without new model inference.'}


def check_updates(specs):
    output = []
    assert len(specs) == 20
    for spec in specs:
        path = HERE / 'runs' / spec['id'] / 'result.json'
        result = read(path)
        assert result['status'] == 'complete'
        initial_path, selected_path = Path(result['initial_checkpoint']), Path(result['checkpoint'])
        assert digest(initial_path) == result['initial_checkpoint_sha256']
        assert digest(selected_path) == result['checkpoint_sha256']
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        selected = torch.load(selected_path, map_location='cpu', weights_only=False)
        assert set(initial['state']) == set(selected['state'])
        changes, trainable_count = {}, 0
        for name, before in initial['state'].items():
            after = selected['state'][name]
            assert before.shape == after.shape and torch.isfinite(before).all() and torch.isfinite(after).all()
            changed = int(torch.count_nonzero(before != after))
            frozen = spec['ed_mode'] == 'fixed_ed' and name in ['encoder.weight', 'decoder.weight']
            if frozen:
                assert changed == 0
            else:
                trainable_count += before.numel()
            max_update = float((after - before).abs().max())
            assert max_update == result['selected_parameter_max_updates'][name]
            changes[name] = {'elements': before.numel(), 'changed_scalar_elements': changed,
                             'max_absolute_update': max_update, 'intentionally_frozen': frozen}
        assert trainable_count == result['trainable_parameters']
        scalar_count = sum(x['changed_scalar_elements'] for x in changes.values())
        assert scalar_count <= trainable_count
        if result['selected_epoch'] == 0:
            assert scalar_count == 0
        output.append({'id': spec['id'], 'registered_trainable_parameters': trainable_count,
                       'selected_epoch': result['selected_epoch'], 'selected_changed_scalar_elements': scalar_count,
                       'tensors': changes, 'initial_sha256': result['initial_checkpoint_sha256'],
                       'selected_sha256': result['checkpoint_sha256'],
                       'scope': 'Exact scalar differences from saved initial to selected state; not the number changed at any intermediate optimizer step.'})
    return output


def verify():
    start = time.perf_counter()
    cpu_start = time.process_time()
    torch.set_num_threads(4)
    report_path = HERE / 'final_development.json'
    report, selection = read(report_path), read(HERE / 'final_selection.json')
    assert report['status'] == 'complete' and report['selection_sha256'] == digest(HERE / 'final_selection.json')
    attempt = read(HERE / 'final_attempt.json')
    assert attempt['status'] == 'complete' and attempt['report_sha256'] == digest(report_path)
    datasets = load_data(report)
    evaluated, means = check_evaluation(report, datasets)
    selections = check_selection(selection, report)
    cost_path = HERE / 'neural_cost_corrected.json'
    cost_attempt = read(HERE / 'neural_cost_corrected_attempt.json')
    assert cost_attempt['status'] == 'complete'
    cost = read(cost_path)
    costs = check_cost(cost, read(HERE / 'cost_specs.json'), datasets)
    updates = check_updates(read(HERE / 'final_specs.json'))
    legacy = []
    for fit in sorted({row['fit'] for row in report['rows'] if row['fit'] and not row['fit'].startswith('v2_')}):
        result = read(find_result(fit))
        legacy.append({'id': fit, 'registered_trainable_parameters': result['trainable_parameters'],
                       'selected_changed_scalar_elements': None,
                       'reason': 'No saved initial checkpoint for this v1 run; not reconstructed or inferred from per-tensor maximum updates.'})
    errors = [max(row['full_absolute_errors'].values()) for row in evaluated]
    return {'status': 'pass', 'scope': 'One execution-team CPU audit of existing arrays/checkpoints/JSON; no new model inference or fitting; not independent replication.',
            'sources': {p.name: digest(p) for p in [report_path, HERE / 'final_selection.json', cost_path,
                                                   HERE / 'cost_specs.json', HERE / 'final_specs.json', Path(__file__)]},
            'evaluation_rows': evaluated, 'mean_score_checks': means, 'selection_checks': selections,
            'max_absolute_full_metric_error': max(errors), 'cost_checks': costs,
            'new_run_parameter_changes': updates, 'legacy_parameter_change_limitations': legacy,
            'wall_seconds': time.perf_counter() - start, 'cpu_process_seconds': time.process_time() - cpu_start}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='final_artifact_checks')
    parser.add_argument('--prior-failed')
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('--name must contain only letters, digits, underscores or hyphens')
    output = HERE / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve the existing final artifact audit')
    repair = None
    if args.prior_failed:
        prior = Path(args.prior_failed).resolve()
        assert read(prior)['status'] == 'failed'
        repair = {'prior_failed_artifact': str(prior), 'sha256': digest(prior),
                  'classification': 'audit script grouping error; no experimental result changed',
                  'change': 'Group individual fit prediction keys by arm, or by arm:mode:K for residual and raw bypass, matching evaluator mean-score labels.'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        try:
            result = verify()
        except BaseException:
            json.dump({'status': 'failed', 'error': traceback.format_exc(), 'audit_repair': repair}, stream, indent=2)
            stream.write('\n')
            raise
        result['audit_repair'] = repair
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'rows': len(result['evaluation_rows']),
                      'cost_rows': result['cost_checks']['rows'], 'new_fits_checked': len(result['new_run_parameter_changes']),
                      'max_metric_error': result['max_absolute_full_metric_error'], 'seconds': result['wall_seconds']}))
