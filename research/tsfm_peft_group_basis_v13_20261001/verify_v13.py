"""Verify stored v13 predictions, selection, receipts, and costs without model runs."""
import argparse
from collections import defaultdict
import csv
import json
import traceback

import numpy as np

from runtime_v13 import HERE, SEEDS, Job, artifact, budget_snapshot, digest, load_data, protocol, read_json, save_json
from evaluate_v13 import ROLES, PERIODS, NEW


def close(a, b, name, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(name)


def receipts(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            artifact(value['path'], value['sha256'])
        for item in value.values():
            receipts(item)
    elif isinstance(value, list):
        for item in value:
            receipts(item)


def metrics(prediction, target, mask):
    count = mask.sum((0, 1))
    if np.any(count <= 0) or prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise AssertionError('Prediction/mask contract')
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    sums = [v.sum((0, 1)) for v in (error * error, np.abs(error), error)]
    return {'mse': np.mean(sums[0] / count), 'mae': np.mean(sums[1] / count), 'signed_mean_error': np.mean(sums[2] / count),
            'channel_mse': sums[0] / count, 'channel_mae': sums[1] / count, 'channel_bias': sums[2] / count,
            'squared_error_sums': sums[0], 'absolute_error_sums': sums[1], 'signed_error_sums': sums[2], 'target_counts': count}


def fits_and_seal(job):
    from train_v13 import result_for
    p, selection = protocol(), read_json(HERE / 'selected.json')
    seal, exposure = read_json(HERE / 'selection_seal.json'), read_json(HERE / 'test_exposure.json')
    if set(selection['selected']) != set(p['units']) or len(p['fits']) != 16:
        raise AssertionError('All fixed fits must be selected')
    assert selection['test_used_for_selection'] is False
    assert exposure['seal_sha256'] == digest(HERE / 'selection_seal.json')
    assert seal['created_utc'] <= exposure['first_new_evaluation_utc']
    for name, sha in seal['files'].items():
        artifact(HERE / name, sha)
    ledger = read_json(HERE / 'ledger.json')['jobs']
    initial, schedules = {}, {}
    for spec in p['fits']:
        result = result_for(spec)
        receipts(result)
        row = ledger[result['ledger_id']]
        assert row['status'] == 'complete' and row['category'] == 'neural_fit'
        assert row['metadata']['fit_id'] == spec['id'] and row['updates'] == result['actual_updates']
        assert row['ended_utc'] <= seal['created_utc'] and result['actual_updates'] > 0
        assert result['trainable_parameters'] == 17920
        assert result['restart_roundtrip']['model_optimizer_scheduler_rng'] and result['replay_error'] <= 1e-6
        assert all(result['frozen_parameters_verified'][key] for key in ('no_gradient', 'version_unchanged', 'optimizer_exclusion'))
        assert any(v > 0 for v in result['changed_parameter_scalars'].values())
        curve = read_json(HERE / 'runs' / result['id'] / 'curve.json')
        chosen = min(curve, key=lambda point: (point['val_mse'], point['epoch']))
        assert chosen['epoch'] == result['selected_epoch']
        close(chosen['val_mse'], result['best_val_mse'], 'Best checkpoint rule')
        selected = selection['selected'][spec['dataset']][spec['family']]
        i = list(SEEDS).index(spec['seed'])
        assert selected['run_ids'][i] == result['id'] and selected['selected_epochs'][i] == result['selected_epoch']
        assert selected['checkpoints'][i] == result['checkpoint'] and selected['checkpoint_sha256'][i] == result['checkpoint_sha256']
        key = (spec['dataset'], spec['seed'])
        if key in initial:
            assert initial[key] == result['initial_g']['sha256']
        initial[key] = result['initial_g']['sha256']
        for item in result['schedules']:
            key = (spec['dataset'], spec['seed'], item['epoch'])
            if key in schedules:
                assert schedules[key] == item['sha256']
            schedules[key] = item['sha256']
        job.heartbeat('Verified selected fit ' + result['id'])
    assert all(row['ended_utc'] <= seal['created_utc'] for row in ledger if row['category'] == 'neural_fit')
    return {'fits': 16, 'paired_initial_groups': len(initial), 'paired_schedules': len(schedules),
            'all_attempts_before_joint_evaluation': True, 'no_new_model_runs': True}


def saved_evaluation(job):
    value = read_json(HERE / 'evaluation01.json')
    assert value['status'] == 'complete' and value['independent_confirmation'] is False
    receipts(value)
    csv_scores = {}
    for dataset, unit in value['datasets'].items():
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data['test_a_origins'], data['test_b_origins']])
        n = len(data['test_a_origins'])
        periods = {'test_a': np.arange(n), 'test_b': np.arange(n, len(origins)), 'combined': np.arange(len(origins))}
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        assert len(unit['models']) == 15 and set(unit['role_summary']) == set(ROLES)
        by_family = defaultdict(list)
        for identifier, row in unit['models'].items():
            with np.load(row['prediction']['path'], allow_pickle=False) as arrays:
                assert np.array_equal(origins, arrays['origins'])
                prediction = arrays['prediction']
            scores = {}
            for period, index in periods.items():
                scores[period] = metrics(prediction[index], target[index], mask[index])
                for key, expected in scores[period].items():
                    close(row['periods'][period][key], expected, identifier + '/' + period + '/' + key)
                csv_scores[(dataset, identifier, period)] = scores[period]
            for key in ('squared_error_sums', 'absolute_error_sums', 'signed_error_sums', 'target_counts'):
                close(scores['combined'][key], scores['test_a'][key] + scores['test_b'][key], 'Pooled period counts/errors')
            by_family[row['family']].append(scores)
            if row['reused']:
                assert row['reference_score_replay']['passed'] is True
        for family, members in by_family.items():
            assert len(members) == (1 if family == 'f0' else 2)
            for period in PERIODS:
                for metric in ('mse', 'mae', 'signed_mean_error', 'channel_mse', 'channel_mae', 'channel_bias'):
                    close(unit['role_summary'][family]['periods'][period][metric],
                          np.mean([member[period][metric] for member in members], axis=0), 'Mean seed losses')
        for item in unit['comparisons'].values():
            for period in PERIODS:
                for metric in ('mse', 'mae'):
                    a = unit['role_summary'][item['candidate']]['periods'][period][metric]
                    b = unit['role_summary'][item['reference']]['periods'][period][metric]
                    reported = item['periods'][period][metric]
                    close([reported['candidate'], reported['reference'], reported['absolute_difference']], [a, b, a - b], 'Paired difference')
                    close(reported['relative_change_percent'], 100 * (a - b) / b, 'Relative percent')
        job.heartbeat(dataset + ': saved original-array metrics verified')
    with (HERE / 'evaluation01_models.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(csv_scores) == 180
    for row in rows:
        expected = csv_scores[(row['dataset'], row['id'], row['period'])]
        for metric in ('mse', 'mae', 'signed_mean_error'):
            close(float(row[metric]), expected[metric], 'CSV metric')
    with (HERE / 'evaluation01_channels.csv').open(encoding='utf-8', newline='') as stream:
        channels = list(csv.DictReader(stream))
    assert len(channels) == 15 * 3 * sum(len(unit['columns']) for unit in value['datasets'].values())
    for row in channels:
        expected = csv_scores[(row['dataset'], row['id'], row['period'])]
        index = value['datasets'][row['dataset']]['columns'].index(row['channel'])
        for key, target in (('mse', 'channel_mse'), ('mae', 'channel_mae'), ('signed_mean_error', 'channel_bias'), ('target_count', 'target_counts')):
            close(float(row[key]), expected[target][index], 'Channel CSV metric')
    return {'scored_models': 60, 'model_period_rows': 180, 'seed_mean_not_ensemble': True,
            'bootstrap_intervals_recomputed': False, 'new_model_runs': 0}


def costs_and_report(job):
    evaluation, cost = read_json(HERE / 'evaluation01.json'), read_json(HERE / 'cost_summary.json')
    choice, trigger = read_json(HERE / 'selected.json'), read_json(HERE / 'cost_decision.json')
    expected = []
    for dataset, unit in evaluation['datasets'].items():
        score = unit['role_summary']
        expected.append(score['group_res']['periods']['combined']['mse'] < score['pca_level']['periods']['combined']['mse']
                        and any(e > 0 for e in choice['selected'][dataset]['group_res']['selected_epochs']))
    assert (trigger['status'] == 'eligible') == any(expected)
    counts = {}
    if cost['status'] == 'not_triggered':
        assert not any(expected) and cost['new_measurements'] == 0
    else:
        receipts(cost['binding'])
        binding = read_json(cost['binding']['path'])
        for name, sha in binding['files'].items():
            artifact(HERE / name, sha)
        for device, section in (('cuda', 'gpu'), ('cpu', 'cpu')):
            summary = cost.get(section) or {}
            if summary.get('status') == 'not_run_optional':
                continue
            raw = read_json(HERE / f'cost_{device}_rows.json')
            counts[device] = len(raw['rows'])
            grid = read_json(HERE / f'cost_{device}_grid.json')
            assert raw['expected_rows'] == len(grid['rows']) == (480 if device == 'cuda' else 48)
            groups = defaultdict(list)
            for row, planned in zip(raw['rows'], grid['rows']):
                assert all(row[key] == value for key, value in planned.items())
                assert row['status'] == 'complete' and row['binding_sha256'] == cost['binding']['sha256']
                assert row['passes'] == binding['settings']['passes'] and row['warmup_calls'] == 10
                times = np.asarray(row['seconds_per_24_origins'])
                assert len(times) == row['passes'] and np.isfinite(times).all() and np.all(times > 0)
                close(row['milliseconds_per_origin']['median'], np.median(times) * 1000 / 24, 'Median latency')
                close(row['origins_per_second']['median'], np.median(24 / times), 'Median throughput')
                assert row['model']['merge_export_parity']['pass']
                assert all(item['parity']['pass'] for item in row['model']['configuration_parity'])
                receipts(row['model'].get('parity_receipt'))
                if 'restore_replay' in row['model']:
                    assert row['model']['restore_replay']['pass']
                if device == 'cuda':
                    assert 0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                else:
                    assert row['peak_allocated_bytes'] is None and row['peak_reserved_bytes'] is None
                if row['sentinel'] is None:
                    groups[(row['dataset'], row['id'], row['batch'], row['execution'], row['chunk_rows'])].append(row)
            if summary.get('status') == 'complete':
                assert len(raw['rows']) == raw['expected_rows']
                for item in summary['instances']:
                    group = groups.pop((item['dataset'], item['id'], item['batch'], item['execution'], item['chunk_rows']))
                    assert len(group) == 3 and {row['block'] for row in group} == {0, 1, 2}
                    close(item['median_block_median_ms'], np.median([r['milliseconds_per_origin']['median'] for r in group]), 'Block median')
                    close(item['median_block_median_origins_s'], np.median([r['origins_per_second']['median'] for r in group]), 'Block throughput')
                assert not groups
            else:
                assert summary['status'] == 'budget_stopped' and summary['complete_frontier'] is False
    report = read_json(HERE / 'report_values.json')
    assert report['status'] == 'complete' and report['cost_status'] == cost['status']
    receipts(report)
    assert len(report['accuracy_rows']) == 69 * 4
    for row in report['accuracy_rows']:
        ds = evaluation['datasets'][row['dataset']]
        score = ds['models'][row['id']]['periods'][row['period']] if row['id'] else ds['role_summary'][row['family']]['periods'][row['period']]
        for metric in ('mse', 'mae', 'signed_mean_error'):
            close(row[metric], score[metric], 'Report metric')
    return {'cost_status': cost['status'], 'saved_cost_rows': counts, 'historical_timing_ratios': False,
            'report_and_figure_hashes_verified': True, 'complete_frontier_claim': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    checks = []
    with Job('cpu_check', args.job, reserve_s=args.reserve_s, metadata={'purpose': 'v13 saved-artifact verification', 'fit_count': 0}) as job:
        for name, action in (('fits_and_seal', fits_and_seal), ('saved_evaluation', saved_evaluation), ('costs_and_report', costs_and_report)):
            try:
                checks.append({'name': name, 'status': 'passed', 'details': action(job)})
            except Exception:
                checks.append({'name': name, 'status': 'failed', 'error': traceback.format_exc()})
        budget = budget_snapshot()
        assert budget['real_fit_attempts'] <= 20 and budget['technical_reserve_attempts'] <= 4
        assert not [row for row in budget['active_jobs'] if row['id'] != job.id]
        result = {'status': 'passed' if all(row['status'] == 'passed' for row in checks) else 'failed', 'checks': checks,
                  'job_id': job.id, 'model_runs': 0, 'optimizer_updates': 0, 'independent_reproduction': False,
                  'budget_during_check': budget}
        path = HERE / 'final_checks.json'
        if path.exists():
            prior = read_json(path)
            save_json(HERE / f'final_checks_prior_job_{prior["job_id"]}.json', prior)
        save_json(path, result)
    result['budget_after_check'] = budget_snapshot()
    save_json(HERE / 'final_checks.json', result)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
