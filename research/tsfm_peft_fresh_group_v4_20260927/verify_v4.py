"""Audit saved v4 evidence on CPU; no model import, forward, fitting or new predictions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import traceback

from runtime_v4 import (HERE, ROOT, Job, LIMITS, budget_snapshot, digest, load_data,
                        phase_sample, save_json, source_receipt, storage_bytes)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path):
    path = resolve(path)
    return {'path': str(path), 'sha256': digest(path)}


def check_receipt(item):
    path = resolve(item['path'])
    assert digest(path) == item['sha256'], f'Content changed: {path}'
    return path


def close(actual, expected, label):
    if expected is None:
        assert actual is None, label
    else:
        assert actual is not None, label
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=label)


def check_budget(own_job=None):
    ledger = read(HERE / 'ledger.json')
    assert ledger['limits'] == LIMITS
    jobs = ledger['jobs']
    assert len({r['id'] for r in jobs}) == len(jobs)
    assert len({r['label'] for r in jobs}) == len(jobs)
    active = [r for r in jobs if r['status'] == 'running' and r['id'] != own_job and r['category'] in ('gpu', 'real_fit')]
    assert not active, f'Other active jobs remain: {[r["label"] for r in active]}'
    assert all(r['status'] in ('complete', 'failed', 'running') for r in jobs)
    snapshot = budget_snapshot()
    assert 19 <= snapshot['real_fit_attempts'] <= LIMITS['real_fit']
    assert snapshot['synthetic_sessions'] <= LIMITS['synthetic_sessions']
    assert all(0 <= r['updates'] <= 10 for r in jobs if r.get('synthetic'))
    for key in ('gpu_seconds', 'cpu_check_seconds', 'storage_bytes'):
        assert snapshot[key] <= LIMITS[key], key
    assert all(r.get('ended_utc', float('inf')) <= read(HERE / 'test_exposure.json')['first_exposed_utc']
               for r in jobs if r['category'] == 'real_fit'), 'Fitting continued after TEST exposure'
    return snapshot


def check_preservation(job):
    provenance = read(HERE / 'provenance.json')
    assert digest(HERE / 'PLAN.md') == provenance['plan_sha256']
    assert digest(HERE / 'protocol.json') == provenance['protocol_sha256']
    preserved = provenance['preserved_files']
    assert len(preserved) == 406
    for index, (path, expected) in enumerate(preserved.items()):
        assert digest(ROOT / path) == expected, f'Preserved v1/v2/v3 file changed: {path}'
        if index % 80 == 0:
            job.check_limits()
    model_source = read(HERE / 'model_source.json')
    assert model_source['revision'] == read(HERE / 'protocol.json')['backbone_revision']
    for item in model_source['files'].values():
        path = check_receipt(item)
        assert path.stat().st_size == item['bytes']
    return {'preserved_files': len(preserved), 'plan': receipt(HERE / 'PLAN.md'),
            'protocol': receipt(HERE / 'protocol.json'), 'backbone_source': receipt(HERE / 'model_source.json')}


def score_terms(prediction, target, observed):
    assert prediction.shape == target.shape == observed.shape
    assert np.isfinite(prediction).all() and np.isfinite(target[observed]).all()
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {'sse': (delta * delta).sum(axis=1), 'sae': np.abs(delta).sum(axis=1),
            'count': observed.sum(axis=1)}


def aggregate(terms, indices):
    count = terms['count'][indices].sum(axis=0)
    sse = terms['sse'][indices].sum(axis=0)
    sae = terms['sae'][indices].sum(axis=0)
    mse = [float(a / n) if n else None for a, n in zip(sse, count)]
    mae = [float(a / n) if n else None for a, n in zip(sae, count)]
    return {'mse': float(np.mean(mse)) if np.all(count > 0) else None,
            'mae': float(np.mean(mae)) if np.all(count > 0) else None,
            'channel_mse': mse, 'channel_mae': mae, 'channel_target_count': count.tolist(),
            'channel_squared_error_sum': sse.tolist(), 'channel_absolute_error_sum': sae.tolist()}


def prediction_terms(entry, data, origins):
    with np.load(check_receipt(entry), allow_pickle=False) as archive:
        assert np.array_equal(archive['origins'], origins), 'Prediction origin order changed'
        prediction = archive['prediction']
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        return score_terms(prediction, target, mask)


def check_training(selection, protocol, data, job):
    rows = selection['all_neural_runs']
    assert len(rows) == 16 and len({r['id'] for r in rows}) == 16
    signature = lambda s: (s['family'], s['ed_mode'], s['seed'])
    assert {signature(r['spec']) for r in rows} == {signature(s) for s in protocol['fits']}
    manifest = read(HERE / 'initial_manifest.json')
    assert manifest['data_sha256'] == digest(data['_path'])
    for item in manifest['records'].values():
        check_receipt(item)
    schedules, counts, checked, temporal_hashes = defaultdict(dict), {}, [], {}
    ledger = read(HERE / 'ledger.json')
    by_id = {r['id']: r for r in ledger['jobs']}
    origins = data['val_origins']
    assert len(origins) == 30 and len(data['train_origins']) == 1945
    for row in rows:
        job.check_limits()
        result = read(check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']}))
        assert result['status'] == 'complete' and result['spec'] == row['spec']
        spec = result['spec']
        ledger_fit = by_id[result['ledger_id']]
        assert ledger_fit['category'] == 'real_fit' and ledger_fit['status'] == 'complete'
        assert ledger_fit['updates'] == result['attempt_updates']
        assert result['data_sha256'] == digest(data['_path'])
        for key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('initial_checkpoint', 'initial_checkpoint_sha256'),
                              ('restart_checkpoint', 'restart_sha256')):
            check_receipt({'path': result[key], 'sha256': result[hash_key]})
        assert row['checkpoint'] == result['checkpoint'] and row['checkpoint_sha256'] == result['checkpoint_sha256']
        assert row['best_val_prediction'] == result['best_val_prediction']
        curves = read(Path(row['result_path']).parent / 'curve.json')
        assert [r['epoch'] for r in curves] == list(range(result['epochs_completed'] + 1))
        best = min(curves, key=lambda r: r['val_mse'])
        assert result['selected_epoch'] == best['epoch']
        close(result['best_val_mse'], best['val_mse'], row['id'] + ' earliest strict best')
        close(row['best_val_mse'], best['val_mse'], row['id'] + ' selection receipt')
        assert result['epochs_completed'] == spec['epochs'] or result['epochs_completed'] - best['epoch'] >= 6
        assert result['total_steps'] == result['epochs_completed'] * 128
        assert all(r['steps'] == r['epoch'] * 128 for r in curves)
        expected_initial = manifest['records'][str(spec['seed'])]['tensor_hashes']
        expected_keys = set() if spec['family'] == 'lora' else {'encoder.weight', 'decoder.weight'}
        if spec['arm'] in ('residual', 'raw_bypass'):
            expected_keys |= {'residual.down.weight', 'residual.up.weight'}
        assert expected_keys <= result['initial_parameter_sha256'].keys()
        for name, expected in expected_initial.items():
            if name in result['initial_parameter_sha256']:
                assert result['initial_parameter_sha256'][name] == expected, row['id'] + ' shared initialization'
        if spec['family'] == 'linear_res':
            assert {'temporal.weight', 'temporal.bias'} <= result['initial_parameter_sha256'].keys()
            temporal = {k: result['initial_parameter_sha256'][k] for k in ('temporal.weight', 'temporal.bias')}
            assert spec['seed'] not in temporal_hashes or temporal_hashes[spec['seed']] == temporal
            temporal_hashes[spec['seed']] = temporal
        changes = result['parameter_max_updates']
        assert any(value > 0 for value in changes.values())
        assert all(np.isfinite(v) and v >= 0 for v in changes.values())
        assert sum(result['changed_parameter_scalars'].values()) > 0
        if spec['family'] != 'lora':
            for name in ('encoder.weight', 'decoder.weight'):
                assert (changes[name] == 0) == (spec['ed_mode'] == 'fixed_ed')
        else:
            assert result['trainable_parameters'] == 294912
        if spec['arm'] in ('residual', 'raw_bypass'):
            assert changes['residual.down.weight'] > 0 and changes['residual.up.weight'] > 0
        if spec['family'] == 'linear_res':
            assert changes['temporal.weight'] > 0 and changes['temporal.bias'] > 0
        assert all(result['frozen_parameters_verified'].values())
        assert result['restart_roundtrip']['additional_updates'] == 0
        assert all(value is True for key, value in result['restart_roundtrip'].items() if key != 'additional_updates')
        assert result['replay_error'] <= 1e-6
        assert set(result['first_update_gradients']) == {'1', '2'}
        for gradients in result['first_update_gradients'].values():
            assert all(v['max_abs'] is None or np.isfinite(v['max_abs']) for v in gradients.values())
        assert [s['epoch'] for s in result['schedules']] == list(range(1, result['epochs_completed'] + 1))
        for schedule in result['schedules']:
            epoch, declared = schedule['epoch'], schedule['sha256']
            actual = phase_sample(data['train_origins'], spec['seed'], epoch, 512).astype('<i8')
            assert len(np.unique(actual)) == 512
            assert hashlib.sha256(actual.tobytes()).hexdigest() == declared
            assert epoch not in schedules[spec['seed']] or schedules[spec['seed']][epoch] == declared
            schedules[spec['seed']][epoch] = declared
        val_score = aggregate(prediction_terms(row['best_val_prediction'], data, origins), np.arange(30))
        close(row['best_val_mse'], val_score['mse'], row['id'] + ' stored VAL prediction')
        counts[row['id']] = result['attempt_updates']
        checked.append(row['id'])
        job.heartbeat('checked training evidence ' + row['id'])
    all_rows = rows + list(selection['deterministic_models'].values())
    for row in selection['deterministic_models'].values():
        result = read(check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']}))
        assert result['deterministic'] and result['spec']['seed'] is None
        check_receipt({'path': row['checkpoint'], 'sha256': row['checkpoint_sha256']})
        close(row['best_val_mse'], aggregate(prediction_terms(row['best_val_prediction'], data, origins), np.arange(30))['mse'],
              row['id'] + ' deterministic VAL')
    for family in ('tsfm_res', 'tsfm_raw', 'linear_res', 'compress', 'lora'):
        modes = {}
        for mode in ('current', 'fixed_ed') if family != 'lora' else ('current',):
            members = [r for r in all_rows if r['spec']['family'] == family and r['spec']['ed_mode'] == mode]
            assert len(members) == (1 if family == 'compress' and mode == 'fixed_ed' else 2)
            modes[mode] = float(np.mean([r['best_val_mse'] for r in members]))
            candidate = selection['candidates'][family][mode]
            assert set(candidate['run_ids']) == {r['id'] for r in members}
            close(candidate['mean_best_val_mse'], modes[mode], family + ' candidate VAL')
        chosen = min(modes, key=lambda m: (modes[m], m != 'fixed_ed'))
        assert selection['selected'][family]['ed_mode'] == chosen
        assert selection['selected'][family]['run_ids'] == selection['candidates'][family][chosen]['run_ids']
        close(selection['selected'][family]['mean_best_val_mse'], modes[chosen], family + ' selected VAL')
    assert selection['res_selected_mode'] == selection['selected']['tsfm_res']['ed_mode']
    return {'effective_neural_cohort': checked, 'actual_updates': counts,
            'shared_initial_and_sample_order': True, 'stored_VAL_scores_recomputed': True,
            'update_and_freeze_scope': 'Recorded actual training checks and hash-bound checkpoints; no model rerun'}


def check_test(evaluation, seal, job):
    assert evaluation['status'] == 'complete' and evaluation['test_reselection'] is False
    check_receipt(evaluation['seal'])
    assert evaluation['seal']['sha256'] == digest(HERE / 'evaluation_seal.json')
    exposure = read(check_receipt(evaluation['first_exposure']))
    assert exposure['seal_sha256'] == evaluation['seal']['sha256']
    assert exposure['first_exposed_utc'] >= seal['sealed_utc']
    assert evaluation['roles'] == seal['roles'] and evaluation['channels'] == seal['channels']
    assert len(evaluation['channels']) == 32
    data = load_data(include_test=True)
    assert data['columns'].astype(str).tolist() == evaluation['channels']
    indices = {'test_a': np.arange(40), 'test_b': np.arange(40, 80), 'combined': np.arange(80)}
    for period in ('test_a', 'test_b'):
        origins = np.asarray(evaluation['origins'][period])
        assert len(origins) == 40 and np.all(np.diff(origins) == 24)
        assert origins.tolist() == seal['origins'][period]
        assert np.array_equal(origins, data[period + '_origins'])
        times = np.asarray(data['times']).astype('datetime64[ns]')
        start, end = [np.datetime64(v, 'ns') for v in seal['splits'][period]]
        assert np.all(times[origins] >= start) and np.all(times[origins + 47] < end)
        assert np.all(times[origins + 47] - times[origins] == np.timedelta64(47, 'h'))
    origins = np.concatenate([evaluation['origins'][p] for p in ('test_a', 'test_b')])
    assert len(np.unique(origins)) == 80
    assert evaluation['origins']['test_a'][-1] + 48 <= evaluation['origins']['test_b'][0]
    models = {r['id']: r for r in seal['models']}
    assert len(models) == 19 and set(models) == set(evaluation['models'])
    scores = {}
    for identifier, row in evaluation['models'].items():
        job.check_limits()
        assert row['spec'] == models[identifier]['spec']
        terms = prediction_terms(row['prediction'], data, origins)
        assert terms['sse'].shape == (80, 32)
        scores[identifier] = {}
        for period, subset in indices.items():
            actual = aggregate(terms, subset)
            declared = row['scores'][period]
            assert declared['missing_channel_indices'] == [i for i, n in enumerate(actual['channel_target_count']) if n == 0]
            for key, value in actual.items():
                if isinstance(value, list) and any(v is None for v in value):
                    for i, v in enumerate(value):
                        close(declared[key][i], v, identifier + '/' + period + '/' + key)
                else:
                    close(declared[key], value, identifier + '/' + period + '/' + key)
            scores[identifier][period] = actual
        for key in ('channel_squared_error_sum', 'channel_absolute_error_sum', 'channel_target_count'):
            close(row['scores']['combined'][key], np.asarray(row['scores']['test_a'][key]) + row['scores']['test_b'][key],
                  identifier + ' pooled A/B ' + key)
    for role, ids in seal['roles'].items():
        assert len(ids) == len(set(ids))
        for period in indices:
            declared = evaluation['role_scores'][role][period]
            assert declared['run_ids'] == ids and declared['count'] == len(ids)
            for metric in ('mse', 'mae'):
                individual = [scores[i][period][metric] for i in ids]
                value = None if any(v is None for v in individual) else float(np.mean(individual))
                close(declared[metric], value, role + '/' + period + ' individual losses, not ensemble')
    comparisons = {r['name']: r for r in evaluation['comparisons']}
    assert len(comparisons) == 9 and set(comparisons) == {r['name'] for r in seal['comparisons']}
    for expected in seal['comparisons']:
        row = comparisons[expected['name']]
        assert all(row[key] == expected[key] for key in ('primary', 'left', 'right', 'matching'))
        for period in indices:
            item = row['periods'][period]
            assert item['draws'] == 2000 and item['strata'] == (['test_a', 'test_b'] if period == 'combined' else [period])
            left = [scores[i][period]['mse'] for i in row['left']]
            right = [scores[i][period]['mse'] for i in row['right']]
            if any(v is None for v in left + right):
                assert item['full'] is None
                continue
            a, b = float(np.mean(left)), float(np.mean(right))
            for key, value in {'left_mse': a, 'reference_mse': b, 'delta_mse': a-b,
                               'relative_to_reference_pct': 100*(a-b)/b if b > 0 else None}.items():
                close(item['full'][key], value, row['name'] + '/' + period + '/' + key)
            for paired in item['paired_seed_deltas']:
                a, b = scores[paired['left_id']][period]['mse'], scores[paired['right_id']][period]['mse']
                close(paired['delta_mse'], a-b, row['name'] + ' paired seed delta')
    return {'models': 19, 'origins': 80, 'period_origins': [40, 40], 'roles': len(seal['roles']),
            'comparisons': len(comparisons), 'all_saved_forecast_scores_recomputed': True,
            'combined_SSE_count_and_seed_loss_mean': True, 'bootstrap_intervals_reexecuted': False}


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(values.min()), 'max': float(values.max())}


def check_cost(path, seal):
    result = read(path)
    assert result['status'] == 'complete' and result['stage'] in ('gpu', 'cpu')
    if result.get('resume'):
        resume = result['resume']
        original = read(check_receipt(resume['partial']))
        assert read(check_receipt(resume['attempt']))['status'] == 'failed'
        repair = read(check_receipt(resume['repair']))
        check_receipt(repair['repaired_code'])
        assert resume['reused_rows'] == len(original['rows'])
        assert resume['single_uninterrupted_session'] is False
        for before, after in zip(original['rows'], result['rows']):
            assert all(after[key] == value for key, value in before.items()), 'A completed cost row was modified'
    check_receipt(result['seal'])
    assert result['seal']['sha256'] == digest(HERE / 'evaluation_seal.json')
    assert result['precision'] == 'float32' and result['torch_threads'] == 4 and result['inference_mode'] is True
    assert result['allow_tf32_matmul'] is False and result['allow_tf32_cudnn'] is False
    assert result['warmup_calls'] == 10 and result['passes_per_order_block'] == 20 and result['order_blocks'] == 3
    assert result['new_fits'] == 0 and result['no_target_scores'] is True
    gpu = result['stage'] == 'gpu'
    ids = seal['cost']['gpu_model_ids' if gpu else 'cpu_model_ids']
    assert result['model_ids'] == ids
    assert len(ids) in (10, 11) if gpu else len(ids) == 3
    rows = result['rows']
    assert len(rows) == result['expected_rows'] == (len(ids) + int(gpu)) * 6
    assert len(result['val_origins']) == 24
    expected_val = np.asarray(seal['origins']['val'])[np.linspace(0, 29, 24, dtype=int)].tolist()
    assert result['val_origins'] == expected_val
    grouped = defaultdict(list)
    f0 = seal['roles']['f0'][0]
    for row in rows:
        assert row['id'] in ids and row['block'] in (0, 1, 2) and row['batch_origins'] in (1, 4)
        assert row['scope'] == ('deployment_cpu_to_cpu' if gpu else 'cpu_fp32')
        times = row['seconds_per_24_origins']
        assert len(times) == 20 and all(np.isfinite(v) and v > 0 for v in times)
        for key, values in (('milliseconds_per_origin', np.asarray(times)*1000/24),
                            ('origins_per_second', 24/np.asarray(times))):
            for stat, expected in numeric_summary(values).items():
                close(row[key][stat], expected, row['id'] + ' raw timing ' + stat)
        assert row['validation_replay']['pass'] is True
        check_receipt(row['validation_replay']['prediction'])
        if row['family'] == 'lora':
            parity = row['merge_parity']
            assert parity['pass'] and parity['origins'] == 24
            assert parity['atol'] == 1e-5 and parity['rtol'] == 1e-4 and np.isfinite(parity['max_abs_error'])
        assert row['total_deployed_tensor_bytes'] == row['deployed_parameter_bytes'] + row['deployed_buffer_bytes']
        if gpu:
            assert row['baseline_memory']['allocated'] == 0
            assert 0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
        assert row['include_in_primary'] == (row['sentinel_role'] != 'post')
        if row['include_in_primary']:
            grouped[(row['id'], row['batch_origins'], row['scope'])].append(row)
    for block in range(3):
        for batch in (1, 4):
            subset = [r for r in rows if r['block'] == block and r['batch_origins'] == batch]
            assert {r['id'] for r in subset} == set(ids)
            middle = [i for i in ids if i != f0]
            ordered = middle if block == 0 else list(reversed(middle)) if block == 1 else middle[len(middle)//2:] + middle[:len(middle)//2]
            expected_order = [f0] + ordered + [f0] if gpu else ordered
            assert [r['id'] for r in sorted(subset, key=lambda r: r['order_position'])] == expected_order
            if gpu:
                sentinels = [r for r in subset if r['id'] == f0]
                assert {r['sentinel_role'] for r in sentinels} == {'pre', 'post'} and len(sentinels) == 2
                assert next(r for r in sentinels if r['sentinel_role'] == 'pre')['order_position'] == 0
                assert next(r for r in sentinels if r['sentinel_role'] == 'post')['order_position'] == len(ids)
    assert len(result['summary']['fits']) == len(ids)*2
    for fit in result['summary']['fits']:
        members = grouped[(fit['id'], fit['batch_origins'], fit['scope'])]
        assert len(members) == 3 and {r['block'] for r in members} == {0, 1, 2}
        for key in ('milliseconds_per_origin', 'origins_per_second'):
            for stat, value in numeric_summary([r[key]['median'] for r in members]).items():
                close(fit[key][stat], value, fit['id'] + ' three block medians')
    for group in result['summary']['groups']:
        members = [r for r in result['summary']['fits'] if r['id'] in group['fit_ids'] and
                   r['batch_origins'] == group['batch_origins'] and r['scope'] == group['scope']]
        assert len(members) == len(group['fit_ids'])
        close(group['mean_fit_median_ms_per_origin'], np.mean([r['milliseconds_per_origin']['median'] for r in members]), 'cost group mean latency')
        close(group['mean_fit_median_origins_per_second'], np.mean([r['origins_per_second']['median'] for r in members]), 'cost group mean throughput')
    return {'source': receipt(path), 'rows': len(rows), 'models': len(ids), 'three_blocks_complete': True,
            'timing_raw_arithmetic_recomputed': True, 'merge_validation': 'recorded 24-VAL forward parity, not rerun',
            'drift_does_not_fail_audit': True}


def audit(args, job):
    report = {'preservation': check_preservation(job)}
    report['budget_during_audit'] = check_budget(job.id)
    protocol = read(HERE / 'protocol.json')
    selection = read(args.selection)
    seal = read(HERE / 'evaluation_seal.json')
    assert seal['selection']['sha256'] == digest(args.selection)
    for key in ('selection', 'shared_selection', 'data_contract', 'protocol', 'plan'):
        check_receipt(seal[key])
    for item in seal['data_receipts'].values():
        check_receipt(item)
    contract = read(check_receipt(seal['data_contract']))
    for stem in ('electricity', 'metadata'):
        assert digest(contract['source'][stem + '_csv']) == contract['source'][stem + '_sha256']
    assert contract['source']['additional_download_bytes'] == 0
    data = load_data(include_test=False)
    assert data['columns'].astype(str).tolist() == protocol['selected_columns'] == seal['channels']
    shared = read(check_receipt(seal['shared_selection']))
    assert shared['test_used_for_selection'] is False
    assert [r['penalty'] for r in shared['fits']] == protocol['ridge_penalties']
    selected_shared = min(enumerate(shared['fits']), key=lambda row: (row[1]['val_mse'], row[0]))[1]
    assert shared['selected_id'] == selected_shared['id']
    for item in shared['fits']:
        check_receipt(item['result_file'])
        check_receipt(item['weights'])
        actual = aggregate(prediction_terms(item['validation_predictions'], data, data['val_origins']), np.arange(30))
        close(item['val_mse'], actual['mse'], item['id'] + ' ridge stored VAL')
    report['training'] = check_training(selection, protocol, data, job)
    report['evaluation'] = check_test(read(args.evaluation), seal, job)
    report['costs'] = [check_cost(path, seal) for path in args.cost]
    assert {read(path)['stage'] for path in args.cost} == {'gpu', 'cpu'}
    forbidden = [str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
                 (p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.safetensors', '.parquet', '.arrow', '.feather', '.bin', '.pkl', '.joblib')
                  or p.name.lower() in ('electricity.csv', 'metadata.csv'))]
    assert not forbidden, f'Raw arrays/weights in publication folder: {forbidden}'
    report['publication_folder'] = {'raw_arrays_or_weights': [], 'storage_bytes': storage_bytes()}
    job.heartbeat('All saved evidence checked; no forecasts or fits produced', 0)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='final_checks')
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--cost', required=True, action='append', type=Path)
    parser.add_argument('--selection', type=Path, default=HERE / 'selected.json')
    parser.add_argument('--reserve-seconds', type=float, default=180)
    args = parser.parse_args()
    assert args.name and all(c.isalnum() or c in '_-' for c in args.name)
    destination = HERE / (args.name + '.json')
    if destination.exists():
        raise RuntimeError('Never overwrite audit evidence; use a new --name')
    result = {'schema_version': 1, 'status': 'running', 'source': source_receipt(), 'new_predictions': 0,
              'optimizer_updates': 0, 'independent_reproduction': False,
              'scope': 'Execution agent checks saved code/hash/numerical evidence; no independent reproduction claim'}
    try:
        with Job(args.name, category='cpu_check', reserve_seconds=args.reserve_seconds) as job:
            global np
            import numpy as np
            result['checks'] = audit(args, job)
        result.update(status='pass', budget=check_budget(), ledger=receipt(HERE / 'ledger.json'), completed_utc=time.time())
    except BaseException:
        result.update(status='failed', traceback=traceback.format_exc(), completed_utc=time.time())
        save_json(destination, result)
        raise
    save_json(destination, result)


if __name__ == '__main__':
    main()
