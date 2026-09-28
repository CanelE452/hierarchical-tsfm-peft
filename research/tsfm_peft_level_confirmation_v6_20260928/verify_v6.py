"""Audit saved v6 evidence on CPU; no model import, forward, fitting or new predictions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import traceback

from runtime_v6 import (HERE, ROOT, Job, LIMITS, budget_snapshot, digest, load_data,
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


def source_version(source, filename):
    key = (HERE / filename).relative_to(ROOT).as_posix()
    expected = source[key]
    if digest(HERE / filename) == expected:
        return {'path': key, 'sha256': expected, 'current': True}
    for path in sorted((HERE / 'repairs').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.txt') and digest(path) == expected:
            return {'path': path.relative_to(ROOT).as_posix(), 'sha256': expected, 'current': False}
    raise AssertionError('Executed source not current or byte-preserved: ' + key)


def check_checkpoint_tensors(result, manifest, data):
    import torch
    torch.set_num_threads(4)
    initial = torch.load(resolve(result['initial_checkpoint']), map_location='cpu', weights_only=False)
    selected = torch.load(resolve(result['checkpoint']), map_location='cpu', weights_only=False)
    restart = torch.load(resolve(result['restart_checkpoint']), map_location='cpu', weights_only=False)
    spec = result['spec']
    shared_receipt = manifest['records'][str(spec['seed'])]
    shared = torch.load(check_receipt(shared_receipt), map_location='cpu', weights_only=False)
    assert initial['epoch'] == initial['steps'] == shared['epoch'] == shared['steps'] == 0
    assert selected['epoch'] == result['selected_epoch']
    assert restart['epoch'] == result['epochs_completed'] and restart['steps'] == result['total_steps']
    assert restart['next_batch'] == 0 and restart['boundary'] == 'post-VAL, post-scheduler, complete epoch'
    assert all(key in restart for key in ('optimizer', 'scheduler', 'rng', 'history', 'schedules'))
    assert restart['schedules'] == result['schedules']
    assert set(initial['state']) == set(selected['state']) == set(restart['state'])
    assert initial['initial_sha256'] == selected['initial_sha256'] == restart['initial_sha256'] == shared_receipt['sha256']
    for saved in (initial, selected, restart):
        assert saved['spec'] == spec and saved['model_config'] == initial['model_config']
        assert torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(shared['basis']))
    trainable = 0
    for name, tensor in initial['state'].items():
        initial_hash = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        assert initial_hash == result['initial_parameter_sha256'][name]
        if name in shared['state']:
            assert torch.equal(tensor, shared['state'][name])
            assert initial_hash == shared_receipt['tensor_hashes'][name]
        actual_change = float((restart['state'][name] - tensor).abs().max())
        selected_change = float((selected['state'][name] - tensor).abs().max())
        close(actual_change, result['parameter_max_updates'][name], 'Actual saved parameter update: ' + name)
        close(selected_change, result['selected_parameter_max_updates'][name], 'Selected saved parameter update: ' + name)
        assert int(torch.count_nonzero(restart['state'][name] != tensor)) == result['changed_parameter_scalars'][name]
        frozen = name in ('encoder.weight', 'decoder.weight')
        if frozen:
            assert actual_change == 0
        else:
            trainable += tensor.numel()
        if name == 'residual.up.weight':
            assert torch.count_nonzero(tensor) == 0
        if spec['family'] == 'lora':
            assert 'lora_' in name, 'A base tensor was saved as a trainable LoRA adapter'
        else:
            assert not name.startswith('backbone.'), 'Frozen backbone duplicated in adapter payload'
    expected_trainable = 294912 if spec['family'] == 'lora' else 42544 if spec['family'] == 'level_linear_res' else 17920
    assert trainable == result['trainable_parameters'] == expected_trainable
    assert len(result['frozen_parameters_verified']['frozen_tensor_sha256']) == 64
    assert result['restart_roundtrip']['additional_updates'] == 0
    assert restart['history'] == read(HERE / 'runs' / result['id'] / 'curve.json')
    assert restart['optimizer']['state'], 'Restart checkpoint has no optimizer moment state'
    with np.load(resolve(result['initial_checkpoint']).parent / 'origins.npz', allow_pickle=False) as archive:
        assert np.array_equal(archive['val'], data['val_origins'])
        assert np.array_equal(archive['train_probe'], phase_sample(data['train_origins'], spec['seed'], 0, 96))
    return {'saved_tensor_updates_checked': True, 'trainable_scalars': trainable}


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
    assert 12 <= snapshot['real_fit_attempts'] <= LIMITS['real_fit']
    assert snapshot['synthetic_sessions'] <= LIMITS['synthetic_sessions']
    assert all(0 <= r['updates'] <= 10 for r in jobs if r.get('synthetic'))
    synthetic_results = [read(path) for path in HERE.glob('synthetic*.json')]
    assert synthetic_results and any(row.get('status') == 'PASS' and row.get('optimizer_updates') == 10
                                     for row in synthetic_results), 'No completed synthetic validity check'
    for key in ('gpu_seconds', 'cpu_check_seconds', 'storage_bytes'):
        assert snapshot[key] <= LIMITS[key], key
    assert all(r.get('ended_utc', float('inf')) <= read(HERE / 'test_exposure.json')['first_exposed_utc']
               for r in jobs if r['category'] in ('real_fit', 'data_prepare')), 'Fitting or data preparation continued after TEST exposure'
    return snapshot


def check_preservation(job):
    provenance = read(HERE / 'provenance.json')
    assert digest(HERE / 'PLAN.md') == provenance['plan_sha256']
    assert digest(HERE / 'protocol.json') == provenance['protocol_sha256']
    assert digest(HERE / 'APPROVAL.txt') == provenance['approval_sha256']
    preserved = provenance['preserved_files']
    assert len(preserved) == 644
    for index, (path, expected) in enumerate(preserved.items()):
        assert digest(ROOT / path) == expected, f'Preserved v1-v5 file changed: {path}'
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
    assert len(rows) == 10 and len({r['id'] for r in rows}) == 10
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
        assert spec['dataset'] == 'robin' and spec['channels'] == 17 and spec['latent'] == 5
        assert spec['ed_mode'] == 'fixed_ed' and spec['seed'] in (92601, 92602)
        assert spec['lr'] == (1e-4 if spec['family'] == 'lora' else 1e-3)
        assert spec['epochs'] in (120, 240)
        if spec['epochs'] == 240:
            assert spec.get('resume_from') and spec.get('decision_record')
            assert spec['id'] in read(HERE / spec['decision_record'])['authorized_continuations']
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
        expected_groups = {'ed_g', 'temporal'} if spec['family'] == 'level_linear_res' else {'lora'} if spec['family'] == 'lora' else {'ed_g'}
        prior_rates = {name: spec.get('t_lr', 1e-3) if name == 'temporal' else spec['lr'] for name in expected_groups}
        for curve in curves:
            assert set(curve['learning_rates']) == expected_groups
            for group, rate in curve['learning_rates'].items():
                assert 0 < rate <= prior_rates[group]
                starting_rate = spec.get('t_lr', 1e-3) if group == 'temporal' else spec['lr']
                ratio = np.log2(starting_rate / rate)
                close(ratio, round(ratio), 'Plateau powers-of-two learning rate')
                prior_rates[group] = rate
        expected_initial = manifest['records'][str(spec['seed'])]['tensor_hashes']
        expected_keys = set() if spec['family'] == 'lora' else {'encoder.weight', 'decoder.weight'}
        if spec['arm'] in ('residual', 'raw_bypass'):
            expected_keys |= {'residual.down.weight', 'residual.up.weight'}
        assert expected_keys <= result['initial_parameter_sha256'].keys()
        for name, expected in expected_initial.items():
            if name in result['initial_parameter_sha256']:
                assert result['initial_parameter_sha256'][name] == expected, row['id'] + ' shared initialization'
        if spec['family'] == 'level_linear_res':
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
        if spec['family'] == 'level_linear_res':
            assert changes['temporal.weight'] > 0 and changes['temporal.bias'] > 0
        assert all(result['frozen_parameters_verified'].values())
        assert result['restart_roundtrip']['additional_updates'] == 0
        assert all(value is True for key, value in result['restart_roundtrip'].items() if key != 'additional_updates')
        assert result['replay_error'] <= 1e-6
        assert set(result['first_update_gradients']) == {'1', '2'}
        for gradients in result['first_update_gradients'].values():
            assert all(v['max_abs'] is None or np.isfinite(v['max_abs']) for v in gradients.values())
            if spec['family'] != 'lora':
                assert all(gradients[name]['requires_grad'] is False and gradients[name]['max_abs'] is None
                           for name in ('encoder.weight', 'decoder.weight'))
                assert all(gradients[name]['requires_grad'] is True and gradients[name]['max_abs'] is not None
                           for name in ('residual.down.weight', 'residual.up.weight'))
        if spec['family'] != 'lora' and not spec.get('resume_from'):
            assert result['first_update_gradients']['1']['residual.down.weight']['max_abs'] == 0
            assert result['first_update_gradients']['2']['residual.down.weight']['max_abs'] > 0
        if spec['family'] == 'level_linear_res':
            assert all(result['first_update_gradients']['1'][name]['max_abs'] > 0
                       for name in ('temporal.weight', 'temporal.bias'))
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
        check_checkpoint_tensors(result, manifest, data)
        for filename in ('model_v6.py', 'train_v6.py'):
            source_version(result['source'], filename)
        counts[row['id']] = result['attempt_updates']
        checked.append(row['id'])
        job.heartbeat('checked training evidence ' + row['id'])
    for row in selection['deterministic_models'].values():
        result = read(check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']}))
        assert result['deterministic'] and result['spec']['seed'] is None
        check_receipt({'path': row['checkpoint'], 'sha256': row['checkpoint_sha256']})
        close(row['best_val_mse'], aggregate(prediction_terms(row['best_val_prediction'], data, origins), np.arange(30))['mse'],
              row['id'] + ' deterministic VAL')
    for family in ('level_res', 'old_res', 'level_raw', 'level_linear_res', 'lora'):
        members = [row for row in rows if row['spec']['family'] == family]
        assert len(members) == 2 and sorted(row['spec']['seed'] for row in members) == [92601, 92602]
        assert all(row['spec']['ed_mode'] == 'fixed_ed' for row in members)
        chosen = selection['selected'][family]
        assert chosen['ed_mode'] == 'fixed_ed' and set(chosen['run_ids']) == {row['id'] for row in members}
        close(chosen['mean_best_val_mse'], float(np.mean([row['best_val_mse'] for row in members])), family + ' seedmean VAL')
    for seed in (92601, 92602):
        level = next(read(row['result_path']) for row in rows if row['spec']['family'] == 'level_res' and row['spec']['seed'] == seed)
        raw = next(read(row['result_path']) for row in rows if row['spec']['family'] == 'level_raw' and row['spec']['seed'] == seed)
        assert level['initial_parameter_sha256'] == raw['initial_parameter_sha256']
        close(level['initial_val_mse'], raw['initial_val_mse'], 'LEVEL/RAW identical initial function')
        assert level['trainable_parameters'] == raw['trainable_parameters'] == 17920
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
    assert len(evaluation['channels']) == 17
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
    assert len(models) == 14 and set(models) == set(evaluation['models'])
    scores, all_terms = {}, {}
    for identifier, row in evaluation['models'].items():
        job.check_limits()
        assert row['spec'] == models[identifier]['spec']
        terms = prediction_terms(row['prediction'], data, origins)
        assert terms['sse'].shape == (80, 17)
        all_terms[identifier] = terms
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
    assert len(comparisons) == 8 and set(comparisons) == {r['name'] for r in seal['comparisons']}
    for expected in seal['comparisons']:
        row = comparisons[expected['name']]
        assert all(row[key] == expected[key] for key in ('primary', 'left', 'right', 'question'))
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
    check_bootstrap(evaluation, all_terms, seal, job)
    return {'models': 14, 'origins': 80, 'period_origins': [40, 40], 'roles': len(seal['roles']),
            'comparisons': len(comparisons), 'all_saved_forecast_scores_recomputed': True,
            'combined_SSE_count_and_seed_loss_mean': True, 'bootstrap_intervals_reexecuted': True}


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(values.min()), 'max': float(values.max())}


def check_bootstrap(evaluation, terms, seal, job):
    models = {row['id']: row for row in seal['models']}
    seeds = (92601, 92602)
    periods = {'test_a': np.arange(40), 'test_b': np.arange(40, 80)}

    def paired(ids):
        if len(ids) == 1:
            return ids * 2
        mapping = {models[identifier]['spec']['seed']: identifier for identifier in ids}
        assert set(mapping) == set(seeds)
        return [mapping[seed] for seed in seeds]

    for comparison in evaluation['comparisons']:
        left_ids, right_ids = paired(comparison['left']), paired(comparison['right'])
        assert comparison['left_seed_ids'] == left_ids and comparison['right_seed_ids'] == right_ids
        counts = terms[left_ids[0]]['count']
        assert all(np.array_equal(terms[i]['count'], counts) for i in left_ids + right_ids)
        left = np.mean([terms[i]['sse'] for i in left_ids], axis=0)
        right = np.mean([terms[i]['sse'] for i in right_ids], axis=0)
        for period in ('test_a', 'test_b', 'combined'):
            strata = list(periods.values()) if period == 'combined' else [periods[period]]
            rng = np.random.default_rng(9262026)
            differences, relatives, missing = [], [], 0
            for draw in range(2000):
                if draw % 250 == 0:
                    job.check_limits()
                chosen = []
                for stratum in strata:
                    starts = rng.integers(0, len(stratum) - 6, int(np.ceil(len(stratum) / 7)))
                    indices = np.concatenate([np.arange(start, start + 7) for start in starts])[:len(stratum)]
                    chosen.append(stratum[indices])
                index = np.concatenate(chosen)
                denominator = counts[index].sum(axis=0)
                if np.any(denominator == 0):
                    missing += 1
                    continue
                a = np.mean(left[index].sum(axis=0) / denominator)
                b = np.mean(right[index].sum(axis=0) / denominator)
                if b <= 0:
                    missing += 1
                    continue
                differences.append(a - b)
                relatives.append(100 * (a - b) / b)
            saved = comparison['periods'][period]
            assert saved['draws'] == 2000 and saved['unavailable_draws'] == missing
            if missing or saved['full'] is None:
                assert saved['conditional_block95'] is None
                assert saved['interval_status'] == 'unavailable_without_dropping_channels'
            else:
                assert saved['interval_status'] == 'available'
                close(saved['conditional_block95']['delta_mse'], np.quantile(differences, [.025, .975]), 'Block delta interval')
                close(saved['conditional_block95']['relative_to_reference_pct'], np.quantile(relatives, [.025, .975]), 'Block relative interval')
        job.heartbeat('checked saved paired blocks ' + comparison['name'])


def check_drift(result):
    rows = result['rows']
    expected = result['drift_checks']
    sentinels = []
    reversals = []
    for block in range(3):
        for batch in (1, 4):
            pair = {row['sentinel_role']: row for row in rows if row.get('sentinel_role') and
                    row['block'] == block and row['batch_origins'] == batch}
            pre, post = [pair[role]['milliseconds_per_origin'] for role in ('pre', 'post')]
            sentinels.append({'block': block, 'batch_origins': batch, 'pre_median_ms': pre['median'],
                              'post_median_ms': post['median'], 'post_over_pre': post['median'] / pre['median'],
                              'iqr_disjoint': pre['q75'] < post['q25'] or post['q75'] < pre['q25']})
    for batch in (1, 4):
        timing = defaultdict(dict)
        for row in rows:
            if row['include_in_primary'] and row['batch_origins'] == batch:
                timing[row['id']][row['block']] = row['milliseconds_per_origin']['median']
        identifiers = sorted(timing)
        for offset, left in enumerate(identifiers):
            for right in identifiers[offset + 1:]:
                differences = [timing[left][block] - timing[right][block] for block in range(3)]
                if min(differences) < 0 < max(differences):
                    reversals.append({'batch_origins': batch, 'left': left, 'right': right,
                                      'block_latency_differences_ms': differences})
    assert sentinels == expected['f0_pre_post'] and reversals == expected['pair_order_reversals']
    assert expected['diagnostic_allowed'] == (any(row['iqr_disjoint'] for row in sentinels) or bool(reversals))


def check_artifact_manifest(path):
    manifest = read(path)
    assert manifest['artifacts'] and manifest['sources'] and manifest['claims'], 'Empty publication evidence manifest'
    artifacts = {str(check_receipt(item).resolve()) for item in manifest['artifacts']}
    sources = {str(check_receipt(item).resolve()) for item in manifest['sources']}
    checked = []
    for claim in manifest['claims']:
        source = check_receipt(claim['source'])
        assert str(source.resolve()) in sources
        value = read(source)
        for component in claim['json_path']:
            value = value[component]
        close(claim['value'], value, claim['label'])
        if 'format_spec' in claim:
            expected_rendered = format(value * claim.get('multiplier', 1), claim['format_spec'])
        else:
            expected_rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
        assert claim['rendered'] == expected_rendered, claim['label'] + ': displayed number differs from source'
        artifact = resolve(claim['artifact'])
        assert str(artifact.resolve()) in artifacts
        assert claim['rendered'] and claim['rendered'] in artifact.read_text(encoding='utf-8'), claim['label']
        checked.append(claim['label'])
    return {'manifest': receipt(path), 'artifacts': len(artifacts), 'registered_claims': checked,
            'scope': 'Every registered numeric/source claim and artifact hash checked; prose meaning and visual layout require separate review.'}


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
    assert len(ids) == 12 if gpu else len(ids) == 3
    rows = result['rows']
    assert len(rows) == result['expected_rows'] == (len(ids) + int(gpu)) * 6
    assert len(result['val_origins']) == 24
    expected_val = np.asarray(seal['origins']['val'])[np.linspace(0, 29, 24, dtype=int)].tolist()
    assert result['val_origins'] == expected_val
    grouped = defaultdict(list)
    f0 = seal['roles']['f0'][0]
    models = {row['id']: row for row in seal['models']}
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
        assert row['family'] == models[row['id']]['spec']['family']
        assert row['task_channels'] == 17
        family = row['family']
        expected_series = 0 if family in ('shared', 'level_linear_res') else 17 if family in ('f0', 'lora') else 5
        assert row['backbone_series_per_origin'] == expected_series
        expected_trainable = 294912 if family == 'lora' else 42544 if family == 'level_linear_res' else 17920 if family in ('level_res', 'level_raw', 'old_res') else 0
        assert row['registered_trainable_parameters_before_merge'] == expected_trainable
        if family == 'shared':
            assert row['fitted_affine_coefficients_stored_as_buffers'] == 24624
            assert row['total_deployed_tensor_bytes'] == 24624 * 4
        if not result.get('resume'):
            assert row['measurement_cost_source_sha256'] == result['source'][(HERE / 'cost_v6.py').relative_to(ROOT).as_posix()]
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
    if gpu:
        check_drift(result)
    source_version(result['source'], 'cost_v6.py')
    return {'source': receipt(path), 'rows': len(rows), 'models': len(ids), 'three_blocks_complete': True,
            'timing_raw_arithmetic_recomputed': True, 'merge_validation': 'recorded 24-VAL forward parity, not rerun',
            'drift_does_not_fail_audit': True}


def check_csv_rows(path, expected_rows, numeric_fields):
    import csv
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert set(reader.fieldnames) == set(expected_rows[0]), 'CSV columns changed: ' + str(path)
    assert len(rows) == len(expected_rows), 'CSV row count changed: ' + str(path)
    for position, (actual, expected) in enumerate(zip(rows, expected_rows)):
        for key, value in expected.items():
            label = str(path) + '/' + str(position) + '/' + key
            if key in numeric_fields:
                parsed = None if actual[key] == '' else float(actual[key])
                close(parsed, value, label)
            else:
                assert actual[key] == ('' if value is None else str(value)), label
    return {'source': receipt(path), 'rows': len(rows), 'all_cells_checked': True}


def check_derived_reports(evaluation, costs, selection, evaluation_path):
    paths = {read(path)['stage']: resolve(path) for path in costs}
    original = {stage: read(path) for stage, path in paths.items()}
    report = {}
    values_path = HERE / 'report_values.json'
    if values_path.exists():
        values = read(values_path)
        assert values['status'] == 'complete'
        available_sources = {str(check_receipt(item).resolve()) for item in values['sources']}
        assert {str(resolve(evaluation_path).resolve()), *(str(path.resolve()) for path in paths.values())} <= available_sources
        check_receipt(values['source_code'])
        assert values['accuracy'] == evaluation['role_scores'] and values['comparisons'] == evaluation['comparisons']
        for stage in ('gpu', 'cpu'):
            measured = original[stage]
            wanted_families = {group['family'] for group in measured['summary']['groups']}
            assert set(values[stage]) == wanted_families
            for group in measured['summary']['groups']:
                family, batch = group['family'], str(group['batch_origins'])
                item = values[stage][family][batch]
                rows = [row for row in measured['rows'] if row['include_in_primary'] and
                        row['family'] == family and str(row['batch_origins']) == batch]
                assert rows
                close(item['throughput'], group['mean_fit_median_origins_per_second'], stage + family + ' throughput')
                close(item['ms_per_origin'], group['mean_fit_median_ms_per_origin'], stage + family + ' latency')
                block_means = [float(np.mean([row['origins_per_second']['median'] for row in rows if row['block'] == block]))
                               for block in range(3)]
                close(item['block_seed_mean_throughput'], block_means, 'Per-block seed-mean throughput')
                for key, source_key in (('deployed_parameters', 'total_deployed_parameters'),
                                         ('deployed_tensor_bytes', 'total_deployed_tensor_bytes'),
                                         ('trainable_before_merge', 'registered_trainable_parameters_before_merge')):
                    assert all(row[source_key] == rows[0][source_key] for row in rows)
                    assert item[key] == rows[0][source_key]
                assert item['fitted_coefficients'] == (24624 if family == 'shared' else rows[0]['registered_trainable_parameters_before_merge'])
                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    if field in rows[0]:
                        peaks = [row[field] for row in rows]
                        assert item[field] == max(peaks) and item[field + '_min'] == min(peaks)
                        close(item[field.replace('_bytes', '_mib')], max(peaks) / 2**20, 'Max-peak bytes to MiB')
                    else:
                        assert field not in item and field.replace('_bytes', '_mib') not in item, 'CPU memory must not be invented as zero'
        level = values['gpu']['level_res']['4']
        expected_references = {'old_res', 'level_raw', 'f0', 'lora', 'level_linear_res'}
        assert set(values['cost_relative_to']) == expected_references
        for family, relative in values['cost_relative_to'].items():
            reference = values['gpu'][family]['4']
            ratio = level['throughput'] / reference['throughput']
            close(relative['batch4_throughput_ratio'], ratio, family + ' throughput ratio')
            close(relative['batch4_throughput_change_pct'], 100 * (ratio - 1), family + ' throughput relative percent')
            close(relative['peak_allocated_change_pct'], 100 * (level['peak_allocated_bytes'] / reference['peak_allocated_bytes'] - 1), family + ' allocated relative percent')
            close(relative['batch1_latency_change_pct'], 100 * (values['gpu']['level_res']['1']['ms_per_origin'] /
                  values['gpu'][family]['1']['ms_per_origin'] - 1), family + ' batch1 latency relative percent')
        assert set(values['channel_directions']) == set(evaluation['roles'])
        channel_means = {family: np.mean([evaluation['models'][identifier]['scores']['combined']['channel_mse']
                                        for identifier in identifiers], axis=0)
                         for family, identifiers in evaluation['roles'].items()}
        for family, item in values['channel_directions'].items():
            differences = channel_means['level_res'] - channel_means[family]
            assert item['channel_order'] == evaluation['channels']
            close(item['delta_mse_by_channel'], differences, family + ' channel deltas')
            assert item['lower'] == int(np.sum(differences < 0))
            assert item['higher'] == int(np.sum(differences > 0))
            assert item['ties'] == int(np.sum(differences == 0))
        training = {row['id']: row for row in values['training']}
        assert len(training) == len(values['training']) == 10
        assert set(training) == {row['id'] for row in selection['all_neural_runs']}
        for row in selection['all_neural_runs']:
            selected_result = read(check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']}))
            assert str(resolve(row['result_path']).resolve()) in available_sources
            for key, value in training[row['id']].items():
                assert value == selected_result[key], 'Reported training value differs from selected run: ' + row['id'] + '/' + key
        report['report_values'] = {'source': receipt(values_path), 'cost_centers_extrema_units_ratios': True,
                                   'accuracy_comparisons_channel_directions_training': True}
    else:
        report['report_values'] = {'status': 'not_present', 'checked': False}
    assembly_path = HERE / 'final_comparison.json'
    csv_path = HERE / 'final_comparison.csv'
    channel_path = HERE / 'final_channel_metrics.csv'
    if any(path.exists() for path in (assembly_path, csv_path, channel_path)):
        assert all(path.exists() for path in (assembly_path, csv_path, channel_path)), 'Incomplete final assembly artifacts'
        assembled = read(assembly_path)
        assert all(assembled[key] == value for key, value in evaluation.items()), 'Assembled evaluation fields changed'
        expected_sources = {'evaluation': resolve(evaluation_path), 'gpu_cost': paths['gpu'], 'cpu_cost': paths['cpu']}
        for key, path in expected_sources.items():
            assert check_receipt(assembled['evidence_sources'][key]).resolve() == path.resolve()
        assert assembled['gpu_cost_summary'] == original['gpu']['summary']
        assert assembled['cpu_cost_summary'] == original['cpu']['summary']
        assert assembled['cost_drift'] == original['gpu']['drift_checks']
        assembly_receipt = read(HERE / 'assembly_receipt.json')
        assert assembly_receipt['status'] == 'complete' and assembly_receipt['new_predictions'] == assembly_receipt['new_fits'] == 0
        check_receipt(assembly_receipt['source'])
        assert {str(check_receipt(item).resolve()) for item in assembly_receipt['outputs']} == {
            str(path.resolve()) for path in (assembly_path, csv_path, channel_path)}
        expected_rows, channel_rows = [], []
        for family, periods in evaluation['role_scores'].items():
            for period, score in periods.items():
                expected_rows.append(dict(family=family, model_id='', seed='', period=period,
                                          mse=score['mse'], mae=score['mae'], aggregation='mean individual model loss'))
        for identifier, model in evaluation['models'].items():
            for period, score in model['scores'].items():
                expected_rows.append(dict(family=model['spec']['family'], model_id=identifier, seed=model['spec'].get('seed'),
                                          period=period, mse=score['mse'], mae=score['mae'], aggregation='individual model'))
                for index, channel in enumerate(evaluation['channels']):
                    channel_rows.append(dict(family=model['spec']['family'], model_id=identifier, seed=model['spec'].get('seed'),
                        period=period, channel=channel, mse=score['channel_mse'][index], mae=score['channel_mae'][index],
                        target_count=score['channel_target_count'][index], squared_error_sum=score['channel_squared_error_sum'][index],
                        absolute_error_sum=score['channel_absolute_error_sum'][index]))
        report['assembly'] = {'json': receipt(assembly_path),
            'metrics_csv': check_csv_rows(csv_path, expected_rows, {'mse', 'mae'}),
            'channel_csv': check_csv_rows(channel_path, channel_rows, {'mse', 'mae', 'target_count', 'squared_error_sum', 'absolute_error_sum'}),
            'evaluation_and_cost_fields_match_original_sources': True}
    else:
        report['assembly'] = {'status': 'not_present', 'checked': False}
    return report


def audit(args, job):
    report = {'preservation': check_preservation(job)}
    report['budget_during_audit'] = check_budget(job.id)
    protocol = read(HERE / 'protocol.json')
    selection = read(args.selection)
    seal = read(HERE / 'evaluation_seal.json')
    assert seal['selection']['sha256'] == digest(args.selection)
    for key in ('selection', 'shared_selection', 'data_contract', 'protocol', 'plan', 'approval'):
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
    evaluation = read(args.evaluation)
    source_version(evaluation['source'], 'evaluate_v6.py')
    source_version(evaluation['source'], 'model_v6.py')
    report['evaluation'] = check_test(evaluation, seal, job)
    report['costs'] = [check_cost(path, seal) for path in args.cost]
    assert {read(path)['stage'] for path in args.cost} == {'gpu', 'cpu'}
    report['derived_reporting'] = check_derived_reports(evaluation, args.cost, selection, args.evaluation)
    forbidden = [str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
                 (p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.safetensors', '.parquet', '.arrow', '.feather', '.bin', '.pkl', '.joblib')
                  or p.name.lower() in ('electricity.csv', 'metadata.csv'))]
    assert not forbidden, f'Raw arrays/weights in publication folder: {forbidden}'
    report['publication_folder'] = {'raw_arrays_or_weights': [], 'storage_bytes': storage_bytes()}
    report['figure_sources'] = check_artifact_manifest(args.figures_manifest) if args.figures_manifest else {'status': 'not_requested'}
    report['manuscript_claim_sources'] = check_artifact_manifest(args.claim_manifest) if args.claim_manifest else {'status': 'not_requested'}
    report['publication_ready'] = args.figures_manifest is not None and args.claim_manifest is not None
    job.heartbeat('All saved evidence checked; no forecasts or fits produced', 0)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='final_checks')
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--cost', required=True, action='append', type=Path)
    parser.add_argument('--selection', type=Path, default=HERE / 'selected.json')
    parser.add_argument('--reserve-seconds', type=float, default=180)
    parser.add_argument('--figures-manifest', type=Path)
    parser.add_argument('--claim-manifest', type=Path)
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
        result.update(status='pass' if result['checks']['publication_ready'] else 'partial_checks_passed', budget=check_budget(), ledger=receipt(HERE / 'ledger.json'), completed_utc=time.time())
    except BaseException:
        result.update(status='failed', traceback=traceback.format_exc(), completed_utc=time.time())
        save_json(destination, result)
        raise
    save_json(destination, result)


if __name__ == '__main__':
    main()
