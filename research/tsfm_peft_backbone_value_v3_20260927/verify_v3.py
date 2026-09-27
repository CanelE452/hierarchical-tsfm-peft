"""CPU verification of saved v3 evidence, without forecasts or parameter updates."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from runtime import (CACHE, HERE, ROOT, LIMITS, Job, digest, elapsed, load_data,
                     phase_sample, save_json, score_arrays, storage_bytes)


SEEDS = [92601, 92602]
EDG = {'encoder.weight', 'decoder.weight', 'residual.down.weight', 'residual.up.weight'}
STATE_KEYS = EDG | {'temporal.weight', 'temporal.bias'}
PERIODS = {'electricity': ['dev'], 'bull': ['e1', 'e2']}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checked_path(path, expected):
    path = resolve(path)
    assert path.is_file(), str(path)
    assert digest(path) == expected, f'hash mismatch: {path}'
    return path


def check_receipt(receipt):
    return checked_path(receipt['path'], receipt['sha256'])


def close(actual, expected, label='number'):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=label)


def verify_scores(actual, recorded):
    for metric in ('mse', 'mae', 'channel_mse'):
        close(actual[metric], recorded[metric], metric)
    assert actual['target_counts'] == recorded['target_counts']
    assert actual['origins'] == recorded['origins']


def arrays_for(data, period):
    key = f'eval_{period}_origins' if period in ('e1', 'e2') else f'{period}_origins'
    origins = data[key]
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return origins, target, mask


def load_prediction(path, origins, expected_hash=None):
    path = checked_path(path, expected_hash) if expected_hash else resolve(path)
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive['origins'], origins), f'origin mismatch: {path}'
        prediction = archive['prediction'].copy()
    assert np.isfinite(prediction).all(), str(path)
    return prediction


def state_hashes(state):
    return {name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


def verify_provenance():
    record = read_json(HERE / 'provenance.json')
    checked_path(HERE / 'PLAN.md', record['plan_sha256'])
    for path, expected in record['preserved_v1_v2'].items():
        checked_path(path, expected)
    return {'plan_sha256': record['plan_sha256'],
            'preserved_v1_v2_files': len(record['preserved_v1_v2']),
            'base_commit': record['base_commit']}


def verify_runs(data_by_dataset, job):
    expected_first = read_json(HERE / 'first8_specs.json')
    assert len(expected_first) == 8 and len({s['id'] for s in expected_first}) == 8
    results = {p.parent.name: read_json(p) for p in sorted((HERE / 'runs').glob('*/result.json'))}
    assert all(s['id'] in results for s in expected_first), 'first eight fits are incomplete'
    for spec in expected_first:
        assert all(results[spec['id']][key] == value for key, value in spec.items())
    paired = {}
    receipts = []
    for identifier, run in results.items():
        job.check_limits()
        assert run['status'] == 'complete' and run['id'] == identifier
        curve = read_json(HERE / 'runs' / identifier / 'curve.json')
        assert [r['epoch'] for r in curve] == list(range(run['epochs_completed'] + 1))
        best = min(curve, key=lambda row: row['val_mse'])
        assert best['epoch'] == run['selected_epoch'], identifier
        close(best['val_mse'], run['best_val_mse'], identifier)
        assert best['steps'] == run['selected_steps']
        assert curve[-1]['steps'] == run['total_steps'] == run['epochs_completed'] * 128
        if run['epochs_completed'] < run['epochs']:
            assert run['epochs_completed'] - best['epoch'] >= 6, identifier
        close(curve[0]['val_mse'], run['initial_val_mse'], identifier)
        for row in curve:
            rates = row['learning_rates']
            close(rates['temporal'], rates['ed_g'] * run['t_lr'] / 1e-3, 'LR ratio')
        parent_path = check_receipt(run['initial_source'])
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        initial_path = checked_path(run['initial_checkpoint'], run['initial_checkpoint_sha256'])
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        assert parent['epoch'] == parent['steps'] == initial['epoch'] == initial['steps'] == 0
        assert set(parent['state']) == EDG and set(initial['state']) == STATE_KEYS
        assert all(torch.equal(initial['state'][key], parent['state'][key]) for key in EDG)
        assert torch.count_nonzero(initial['state']['residual.up.weight']).item() == 0
        hashes = state_hashes(initial['state'])
        assert hashes == run['initial_parameter_sha256']
        pair_key = (run['dataset'], run['seed'])
        if pair_key in paired:
            assert paired[pair_key] == hashes, f'paired initial tensors differ: {identifier}'
        paired[pair_key] = hashes
        selected_path = checked_path(run['checkpoint'], run['checkpoint_sha256'])
        restart_path = checked_path(run['restart_checkpoint'], run['restart_sha256'])
        selected = torch.load(selected_path, map_location='cpu', weights_only=False)
        restart = torch.load(restart_path, map_location='cpu', weights_only=False)
        assert set(selected['state']) == set(restart['state']) == STATE_KEYS
        assert selected['epoch'] == run['selected_epoch'] and selected['steps'] == run['selected_steps']
        assert restart['epoch'] == run['epochs_completed'] and restart['steps'] == run['total_steps']
        assert restart['next_batch'] == 0
        assert all(key in restart for key in ('optimizer', 'scheduler', 'rng', 'stale', 'history', 'schedules'))
        counts = {}
        for key, start in initial['state'].items():
            last, chosen = restart['state'][key], selected['state'][key]
            assert torch.isfinite(last).all() and torch.isfinite(chosen).all()
            counts[key] = int(torch.count_nonzero(last != start))
            assert counts[key] == run['changed_parameter_scalars'][key]
            close(float((last - start).abs().max()), run['parameter_max_updates'][key], key)
            close(float((chosen - start).abs().max()), run['selected_parameter_max_updates'][key], key)
            fixed = run['ed_mode'] == 'fixed_ed' and key.startswith(('encoder.', 'decoder.'))
            assert (counts[key] == 0) == fixed, f'update/freeze mismatch: {identifier}/{key}'
            if fixed:
                assert torch.equal(chosen, start)
        expected_total = 43056 if run['dataset'] == 'electricity' else 42672
        assert sum(value.numel() for value in initial['state'].values()) == run['total_parameters'] == expected_total
        assert run['trainable_parameters'] == (43056 if run['ed_mode'] == 'current' else 42544)
        assert run['initial_gradients']['residual.down.weight'] == 0
        assert run['initial_gradients']['temporal.weight'] > 0 and run['initial_gradients']['temporal.bias'] > 0
        data = data_by_dataset[run['dataset']]
        origins, target, mask = arrays_for(data, 'val')
        prediction = load_prediction(selected_path.with_name('best_val.npz'), origins)
        scores = score_arrays(prediction, target, mask)
        close(scores['mse'], run['best_val_mse'], 'saved VAL MSE')
        close(scores['mae'], best['val_mae'], 'saved VAL MAE')
        for schedule in run['schedules']:
            generated = phase_sample(data['train_origins'], run['seed'], schedule['epoch'], 512)
            assert hashlib.sha256(generated.astype('<i8').tobytes()).hexdigest() == schedule['sha256']
        receipts.append({'id': identifier, 'selected_epoch': best['epoch'], 'epochs_completed': run['epochs_completed'],
                         'selected_steps': selected['steps'], 'attempt_updates': run['attempt_updates'],
                         'saved_val_mse': scores['mse'], 'changed_scalars_at_last_checkpoint': counts,
                         'initial_sha256': digest(initial_path), 'best_sha256': digest(selected_path),
                         'restart_sha256': digest(restart_path)})
        job.heartbeat(f'verified saved run {identifier}', 0)
    return results, {'completed_first_eight': 8, 'completed_runs': len(results), 'runs': receipts,
                     'scope': 'Saved tensors, hashes, recorded schedules and predictions only; no model construction or forward.'}


def verify_selection(path, results):
    selection = read_json(path)
    selection = selection.get('selection', selection)
    checked = {}
    for dataset, entry in selection.items():
        chosen = [results[key] for key in entry['run_ids']]
        assert len(chosen) == 2 and sorted(r['seed'] for r in chosen) == SEEDS
        assert all(r['dataset'] == dataset and r['t_lr'] == entry['lr'] for r in chosen)
        close(np.mean([r['best_val_mse'] for r in chosen]), entry['mean_best_val_mse'], 'selection mean')
        candidates = entry.get('lr_candidates')
        if candidates:
            for candidate in candidates.values():
                cohort = [results[key] for key in candidate['run_ids']]
                assert len(cohort) == 2 and sorted(r['seed'] for r in cohort) == SEEDS
                assert all(r['dataset'] == dataset for r in cohort)
                close(np.mean([r['best_val_mse'] for r in cohort]), candidate['mean_best_val_mse'], 'candidate mean')
            winner = min(candidates, key=lambda key: candidates[key]['mean_best_val_mse'])
            assert float(winner) == entry['lr']
            assert set(candidates[winner]['run_ids']) == set(entry['run_ids'])
        checked[dataset] = {'run_ids': entry['run_ids'], 'lr': entry['lr'],
                            'mean_best_val_mse': entry['mean_best_val_mse'],
                            'selection_opportunities': len(candidates) if candidates else None}
    return {'path': str(path), 'sha256': digest(path), 'selection': checked}


def verify_delta(linear, reference, target, mask, recorded):
    assert len(recorded['temporal_halves']) == len(recorded['paired_seed_deltas']) == 2
    def value(indices):
        left = np.mean([score_arrays(pred[indices], target[indices], mask[indices])['mse'] for pred in linear])
        right = np.mean([score_arrays(pred[indices], target[indices], mask[indices])['mse'] for pred in reference])
        return {'linear_mse': float(left), 'reference_mse': float(right),
                'delta_mse_linear_minus_reference': float(left - right),
                'relative_to_reference_pct': float(100 * (left - right) / right)}
    n = len(target)
    calculated = value(np.arange(n))
    for key, number in calculated.items():
        close(number, recorded['full'][key], key)
    for indices, saved in zip((np.arange(n // 2), np.arange(n // 2, n)), recorded['temporal_halves']):
        for key, number in value(indices).items():
            close(number, saved[key], 'time half ' + key)
    for left, right, saved in zip(linear, reference, recorded['paired_seed_deltas']):
        left_score, right_score = score_arrays(left, target, mask)['mse'], score_arrays(right, target, mask)['mse']
        close(left_score, saved['linear_mse'])
        close(right_score, saved['reference_mse'])
        close(left_score - right_score, saved['delta_mse_linear_minus_reference'])
    assert recorded['block_origins'] == 7 and recorded['draws'] == 2000
    for key in ('conditional_block95_delta_mse', 'conditional_block95_relative_to_reference_pct'):
        assert len(recorded[key]) == 2 and np.isfinite(recorded[key]).all()
        assert recorded[key][0] <= recorded[key][1]
    return calculated


def verify_evaluation(path, data_by_dataset, results, job):
    report = read_json(path)
    assert report['status'] == 'complete'
    check_receipt(report['reuse_manifest'])
    selection_path = check_receipt(report['selected_linear'])
    selected_check = verify_selection(selection_path, results)
    reference_report = None
    if report.get('linear_res_reference'):
        reference_report = read_json(check_receipt(report['linear_res_reference']))
        assert reference_report['status'] == 'complete'
        verify_selection(check_receipt(reference_report['selected_linear']), results)
    verified = []
    for dataset, periods in PERIODS.items():
        assert set(report['datasets'][dataset]['periods']) == set(periods)
        for period in periods:
            saved = report['datasets'][dataset]['periods'][period]
            origins, target, mask = arrays_for(data_by_dataset[dataset], period)
            assert np.array_equal(saved['origins'], origins)
            assert saved['target_shape'] == list(target.shape)
            by_role = defaultdict(list)
            candidate_keys = {row['key'] for row in saved['linear_predictions']}
            for row in saved['reused_predictions'] + saved['linear_predictions']:
                prediction = load_prediction(row['prediction_path'], origins, row['prediction_sha256'])
                scores = score_arrays(prediction, target, mask)
                verify_scores(scores, row['scores'])
                candidate = row['key'] in candidate_keys
                role = 'linear_candidate' if candidate else row['role']
                if candidate:
                    run = results[row['fit']]
                    assert row['fit'] in selected_check['selection'][dataset]['run_ids']
                    checked_path(row['checkpoint'], row['checkpoint_sha256'])
                    assert row['checkpoint_sha256'] == run['checkpoint_sha256']
                    checked_path(row['initial_checkpoint'], row['initial_checkpoint_sha256'])
                    checked_path(row['result_path'], row['result_sha256'])
                by_role[role].append((row, prediction, scores))
                alias = row.get('role')
                if candidate and alias and alias != role:
                    by_role[alias].append((row, prediction, scores))
            for role, rows in by_role.items():
                mean = saved['seed_mean_scores'][role]
                assert mean['count'] == len(rows)
                assert mean['seeds'] == [row['seed'] for row, _, _ in rows]
                for metric in ('mse', 'mae'):
                    close(np.mean([score[metric] for _, _, score in rows]), mean[metric], f'{role} seed mean')
            linear = {row['seed']: pred for row, pred, _ in by_role['linear_candidate']}
            reference = {row['seed']: pred for row, pred, _ in by_role[report['comparison_role']]}
            assert sorted(linear) == sorted(reference) == SEEDS
            delta = verify_delta([linear[s] for s in SEEDS], [reference[s] for s in SEEDS], target, mask,
                                 saved['comparison_linear_vs_reference'])
            period_check = {'dataset': dataset, 'period': period,
                            'prediction_rows': len(saved['reused_predictions']) + len(saved['linear_predictions']),
                            'verified_delta': delta}
            if reference_report is not None:
                reference_rows = saved['linear_res_reference_predictions']
                source_rows = reference_report['datasets'][dataset]['periods'][period]['linear_predictions']
                source_by_seed = {row['seed']: row for row in source_rows}
                assert len(reference_rows) == len(source_rows) == 2
                assert sorted(row['seed'] for row in reference_rows) == sorted(source_by_seed) == SEEDS
                residual_predictions = {}
                candidate_by_seed = {row['seed']: row for row in saved['linear_predictions']}
                for row in reference_rows:
                    source_row = source_by_seed[row['seed']]
                    assert row['arm'] == results[row['fit']]['arm'] == 'residual'
                    assert row['key'] == source_row['key'] and row['fit'] == source_row['fit']
                    assert row['prediction_sha256'] == source_row['prediction_sha256']
                    assert resolve(row['prediction_path']) == resolve(source_row['prediction_path'])
                    prediction = load_prediction(row['prediction_path'], origins, row['prediction_sha256'])
                    scores = score_arrays(prediction, target, mask)
                    verify_scores(scores, row['scores'])
                    verify_scores(scores, source_row['scores'])
                    residual_predictions[row['seed']] = prediction
                    candidate_run = results[candidate_by_seed[row['seed']]['fit']]
                    reference_run = results[row['fit']]
                    assert all(candidate_run[key] == reference_run[key] for key in
                               ('dataset', 'seed', 'ed_mode', 'latent', 'normalization', 't_lr'))
                period_check['verified_delta_vs_linear_res'] = verify_delta(
                    [linear[s] for s in SEEDS], [residual_predictions[s] for s in SEEDS], target, mask,
                    saved['comparison_linear_vs_linear_res'])
                period_check['linear_res_reference_rows'] = len(reference_rows)
                period_check['matched_recipe_verified'] = True
            else:
                assert 'linear_res_reference_predictions' not in saved
                assert 'comparison_linear_vs_linear_res' not in saved
            verified.append(period_check)
            job.heartbeat(f'verified saved evaluation {path.name} {dataset}/{period}', 0)
    return {'path': str(path), 'sha256': digest(path), 'periods': verified,
            'interval_scope': 'Full/seed/half estimates recalculated; saved interval shape, order, block length and draw count checked, bootstrap distribution not rerun.'}


def distribution(values):
    return dict(median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), min=float(min(values)), max=float(max(values)))


def verify_distribution(values, recorded):
    actual = distribution(values)
    assert set(actual) == set(recorded)
    for key, number in actual.items():
        close(number, recorded[key], 'timing ' + key)


def verify_cost(path, expected_stage, results):
    report = read_json(path)
    assert report['status'] == 'complete' and report['stage'] == expected_stage
    rows = report['rows']
    expected_rows = 192 if expected_stage == 'gpu' else 48
    assert len(rows) == expected_rows
    selection_path = check_receipt(report['selection'])
    verify_selection(selection_path, results)
    assert report['warmup_calls'] == 10 and report['passes_per_order_block'] == 20 and report['order_blocks'] == 3
    assert len(report['specs']) == 14
    assert len({s['id'] for s in report['specs']}) == 14
    for dataset in PERIODS:
        cohort = [s for s in report['specs'] if s['dataset'] == dataset]
        assert sum(s['family'] == 'f0' for s in cohort) == 1
        for family in ('merged_lora', 'tsfm_res', 'linear_res'):
            assert sorted(s['seed'] for s in cohort if s['family'] == family) == SEEDS
    primary = defaultdict(list)
    sentinels = defaultdict(dict)
    unique_rows = set()
    for row in rows:
        key = (row['id'], row['block'], row['batch_origins'], row['scope'], row.get('sentinel_role'))
        assert key not in unique_rows, key
        unique_rows.add(key)
        durations = np.asarray(row['seconds_per_24_origins'], dtype=np.float64)
        assert len(durations) == 20 and np.isfinite(durations).all() and (durations > 0).all()
        verify_distribution(durations * 1000 / 24, row['milliseconds_per_origin'])
        verify_distribution(24 / durations, row['origins_per_second'])
        if row['scope'] == 'gpu_resident':
            events = np.asarray(row['cuda_event_milliseconds_per_24_origins'])
            assert len(events) == 20 and np.isfinite(events).all() and (events > 0).all()
            verify_distribution(events / 24, row['cuda_event_milliseconds_per_origin'])
            verify_distribution(24000 / events, row['cuda_event_origins_per_second'])
        if expected_stage == 'gpu':
            assert row['baseline_cuda_bytes']['allocated'] == 0
            assert row['scope'] in ('gpu_resident', 'deployment_cpu_to_cpu')
            assert row['include_in_primary'] == (row.get('sentinel_role') != 'post')
        else:
            assert row['scope'] == 'cpu_fp32'
        if row.get('sentinel_role'):
            assert row['family'] == 'f0'
            assert row['order_position'] == (0 if row['sentinel_role'] == 'pre' else 7)
            sentinels[(row['dataset'], row['block'], row['batch_origins'], row['scope'])][row['sentinel_role']] = row
        if row.get('include_in_primary', True):
            primary[(row['id'], row['batch_origins'], row['scope'])].append(row)
        if row['family'] == 'merged_lora':
            assert row['merge_parity']['pass'] and row['validation_replay']['pass']
        if row['family'] == 'linear_res':
            assert row['validation_replay']['pass'] and row['backbone_series_per_origin'] == 0
            assert row['total_deployed_parameters'] == results[row['id']]['total_parameters']
    assert sum(map(len, primary.values())) == (168 if expected_stage == 'gpu' else 48)
    fit_summary = {(r['id'], r['batch_origins'], r['scope']): r for r in report['summary']['fits']}
    assert set(fit_summary) == set(primary)
    groups = defaultdict(list)
    for key, members in primary.items():
        assert len(members) == 3 and {r['block'] for r in members} == {0, 1, 2}
        summary = fit_summary[key]
        for metric in ('milliseconds_per_origin', 'origins_per_second'):
            verify_distribution([r[metric]['median'] for r in members], summary[metric])
        if key[2] == 'gpu_resident':
            verify_distribution([r['cuda_event_milliseconds_per_origin']['median'] for r in members],
                                summary['cuda_event_milliseconds_per_origin'])
        group_key = tuple(summary[k] for k in ('dataset', 'family', 'ed_mode', 'latent', 'batch_origins', 'scope'))
        groups[group_key].append(summary)
    reported_groups = {tuple(r[k] for k in ('dataset', 'family', 'ed_mode', 'latent', 'batch_origins', 'scope')): r
                       for r in report['summary']['groups']}
    assert set(reported_groups) == set(groups)
    for key, members in groups.items():
        saved = reported_groups[key]
        assert set(saved['fits']) == {r['id'] for r in members}
        close(np.mean([r['milliseconds_per_origin']['median'] for r in members]), saved['mean_fit_median_ms_per_origin'])
        close(np.mean([r['origins_per_second']['median'] for r in members]), saved['mean_fit_median_origins_per_second'])
    if expected_stage == 'gpu':
        assert len(sentinels) == len(report['sentinel_checks']) == 24
        for record in report['sentinel_checks']:
            key = tuple(record[k] for k in ('dataset', 'block', 'batch_origins', 'scope'))
            pair = sentinels[key]
            assert set(pair) == {'pre', 'post'}
            pre, post = [pair[k]['milliseconds_per_origin'] for k in ('pre', 'post')]
            close(pre['median'], record['pre_median_ms'])
            close(post['median'], record['post_median_ms'])
            close(post['median'] / pre['median'], record['post_over_pre'])
            close(abs(post['median'] - pre['median']), record['absolute_change_ms'])
            assert record['pre_post_iqr_disjoint'] == (pre['q75'] < post['q25'] or post['q75'] < pre['q25'])
    return {'path': str(path), 'sha256': digest(path), 'rows': len(rows),
            'primary_rows': sum(map(len, primary.values())), 'fit_summaries': len(fit_summary),
            'configuration_groups': len(groups), 'sentinel_pairs': len(sentinels)}


def verify_ledger(results, own_id):
    ledger = read_json(HERE / 'ledger.json')
    assert ledger['limits'] == LIMITS
    jobs = ledger['jobs']
    assert [r['id'] for r in jobs] == list(range(len(jobs)))
    assert len({r['label'] for r in jobs}) == len(jobs)
    assert all(r['status'] in ('complete', 'failed') or r['id'] == own_id for r in jobs)
    real = [r for r in jobs if r['category'] == 'real_fit']
    synthetic = [r for r in jobs if r.get('synthetic')]
    assert len(real) <= LIMITS['real_fit'] and len(synthetic) <= LIMITS['synthetic_sessions']
    assert all(0 <= r['updates'] <= LIMITS['synthetic_steps'] for r in synthetic)
    for run in results.values():
        entry = jobs[run['ledger_id']]
        assert entry['category'] == 'real_fit' and entry['label'] == run['id'] and entry['status'] == 'complete'
        assert entry['updates'] == run['attempt_updates']
    for entry in real:
        path = HERE / 'runs' / entry['label'] / 'attempt.json'
        attempt = read_json(path)
        assert attempt['status'] == entry['status'] and attempt['actual_updates'] == entry['updates']
    totals = {category: sum(elapsed(r) for r in jobs if r['category'] == category)
              for category in ('gpu', 'cpu_check', 'cpu_analysis')}
    assert totals['gpu'] <= LIMITS['gpu_seconds'] and totals['cpu_check'] <= LIMITS['cpu_check_seconds']
    stored = storage_bytes()
    assert stored <= LIMITS['storage_bytes']
    return {'real_fit_attempts': len(real), 'synthetic_sessions': len(synthetic),
            'synthetic_updates_by_session': {r['label']: r['updates'] for r in synthetic},
            'elapsed_seconds_by_category': totals, 'storage_bytes_at_check': stored,
            'failed_jobs': [r['label'] for r in jobs if r['status'] == 'failed'],
            'scope': 'Current verification job is the sole permitted running ledger row; final receipt is written after it closes.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='final_checks')
    parser.add_argument('--selection', type=Path, default=HERE / 'selected_linear.json')
    parser.add_argument('--evaluation', type=Path, action='append', required=True)
    parser.add_argument('--gpu-cost', type=Path, required=True)
    parser.add_argument('--cpu-cost', type=Path)
    parser.add_argument('--reserve-seconds', type=float, default=120)
    args = parser.parse_args()
    assert args.name and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name)
    output = HERE / f'{args.name}.json'
    if output.exists():
        raise FileExistsError('Preserve original audit; use a new --name for any retry')
    started = time.perf_counter()
    result = {'status': 'running', 'scope': 'Execution-team CPU verification of saved evidence; not independent reproduction.',
              'new_predictions': 0, 'optimizer_updates': 0, 'checks': {}}
    try:
        with Job(args.name, category='cpu_check', reserve_seconds=args.reserve_seconds) as job:
            global np, torch
            import numpy as np
            import torch
            torch.set_num_threads(4)
            result['ledger_id'] = job.id
            result['checks']['preservation'] = verify_provenance()
            data = {dataset: load_data(dataset) for dataset in PERIODS}
            runs, result['checks']['runs'] = verify_runs(data, job)
            result['checks']['selection'] = verify_selection(args.selection, runs)
            result['checks']['evaluations'] = [verify_evaluation(path, data, runs, job) for path in args.evaluation]
            result['checks']['gpu_cost'] = verify_cost(args.gpu_cost, 'gpu', runs)
            if args.cpu_cost:
                result['checks']['cpu_cost'] = verify_cost(args.cpu_cost, 'cpu', runs)
            result['checks']['ledger'] = verify_ledger(runs, job.id)
            job.heartbeat('saved evidence checks passed', 0)
            result['status'] = 'pass'
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
    result['elapsed_s'] = time.perf_counter() - started
    if 'ledger_id' in result:
        result['closed_ledger_job'] = read_json(HERE / 'ledger.json')['jobs'][result['ledger_id']]
    save_json(output, result)
    print(json.dumps({'status': result['status'], 'path': str(output), 'elapsed_s': result['elapsed_s']}))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
