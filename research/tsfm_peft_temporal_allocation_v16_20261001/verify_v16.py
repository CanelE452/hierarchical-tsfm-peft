"""CPU checks of saved v16 artifacts; no model forward, fitting or benchmark."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from runtime_v16 import (HERE, ROOT, DATASETS, SEEDS, Job, artifact, budget_snapshot,
                         digest, load_data, phase_sample, protocol, read_json, resolve, save_json)

ROLES = ('temporal_lora', 'temporal_f0', 'coarse_only', 'direct', 'a', 'f0', 'full_mse', 'native')
PERIODS = ('test_a', 'test_b', 'combined')
GROUPS = ('contracts', 'predictions', 'cost_report')
_checked_receipts = set()


def close(actual, expected, label, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(label)


def receipts(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            key = (value['path'], value['sha256'])
            if key not in _checked_receipts:
                artifact(*key)
                _checked_receipts.add(key)
        for item in value.values():
            receipts(item)
    elif isinstance(value, list):
        for item in value:
            receipts(item)


def read_prediction(receipt, origins):
    receipts(receipt)
    with np.load(resolve(receipt['path']), allow_pickle=False) as arrays:
        assert np.array_equal(arrays['origins'], origins), 'Original origin order changed'
        prediction = arrays['prediction'].copy()
    assert np.isfinite(prediction).all()
    return prediction


def metrics(prediction, target, mask):
    assert prediction.shape == target.shape == mask.shape
    assert np.isfinite(prediction).all() and np.isfinite(target[mask]).all()
    count = mask.sum((0, 1))
    assert np.all(count > 0)
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    squared, absolute, signed = [value.sum((0, 1)) for value in (error * error, np.abs(error), error)]
    return {'mse': np.mean(squared / count), 'mae': np.mean(absolute / count),
            'signed_mean_error': np.mean(signed / count), 'channel_mse': squared / count,
            'channel_mae': absolute / count, 'channel_bias': signed / count,
            'squared_error_sums': squared, 'absolute_error_sums': absolute,
            'signed_error_sums': signed, 'target_counts': count, 'origins': len(prediction)}


def target_arrays(data, origins):
    return (np.stack([data['x'][int(o):int(o) + 48] for o in origins]),
            np.stack([data['finite'][int(o):int(o) + 48] for o in origins]).astype(bool))


def tensor_hashes(state):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


def equal_state(a, b):
    import torch
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


def curve_policy(curve, spec):
    assert [row['epoch'] for row in curve] == list(range(len(curve)))
    assert 7 <= len(curve) <= 121
    best, chosen_epoch, stale = float('inf'), 0, 0
    scheduler_best, scheduler_bad, lr = float('inf'), 0, spec['lr']
    for row in curve:
        epoch, value = row['epoch'], row['val_mse']
        assert np.isfinite([value, row['val_mae'], row['lr']]).all()
        assert row['steps'] == epoch * 128
        assert (row['train_loss'] is None) == (epoch == 0)
        if epoch:
            assert np.isfinite(row['train_loss'])
        close(row['lr'], lr, 'Scheduler learning rate before its epoch step', atol=1e-15, rtol=1e-12)
        if value < best:
            best, chosen_epoch, stale = value, epoch, 0
        else:
            stale += 1
        if epoch < len(curve) - 1:
            assert stale < 6, 'Training continued past strict patience'
        if epoch:
            if value < scheduler_best * (1 - 1e-4):
                scheduler_best, scheduler_bad = value, 0
            else:
                scheduler_bad += 1
            if scheduler_bad > 2:
                new_lr = lr * .5
                if lr - new_lr > 1e-8:
                    lr = new_lr
                scheduler_bad = 0
    assert stale >= 6 or curve[-1]['epoch'] == 120
    return best, chosen_epoch, stale, lr


def contracts(job):
    import torch
    from train_v16 import fit_specs, result_for
    p, specs = protocol(), fit_specs()
    choice = read_json(HERE / 'selected.json')
    seal, exposure = [read_json(HERE / name) for name in ('selection_seal.json', 'test_exposure.json')]
    ledger = read_json(HERE / 'ledger.json')['jobs']
    assert len(specs) == choice['fit_count'] == 16 and choice['status'] == 'complete'
    assert choice['all_datasets_complete'] and not choice['test_used_for_selection']
    assert set(choice['selected']) == set(seal['datasets']) == set(DATASETS)
    assert seal['status'] == 'sealed' and seal['all_fits_selected_before_new_evaluation']
    assert seal['independent_confirmation'] is False
    assert exposure['seal_sha256'] == digest(HERE / 'selection_seal.json')
    assert seal['created_utc'] <= exposure['first_new_evaluation_utc']
    for name, sha in seal['files'].items():
        receipts({'path': str(HERE / name), 'sha256': sha})
    for name in ('data_manifest.json', 'reuse_manifest.json', 'initial_manifest.json'):
        receipts(read_json(HERE / name))
    results, initials, schedules = [], {}, {}
    for dataset in DATASETS:
        data = load_data(dataset)
        target, mask = target_arrays(data, data['val_origins'])
        for spec in [s for s in specs if s['dataset'] == dataset]:
            result = result_for(spec)
            row = ledger[result['job_id']]
            assert row['label'] == result['id'] and row['category'] == 'neural_fit' and row['status'] == 'complete'
            assert row['metadata']['fit_id'] == spec['id'] and row['metadata']['dataset'] == dataset
            assert row['updates'] == result['actual_updates'] > 0
            parent = ledger[result['gpu_job_id']]
            assert parent['category'] == 'gpu' and parent['status'] in ('complete', 'failed')
            assert row['metadata']['parent_job_id'] == parent['id'] and parent['pid'] == row['pid']
            assert parent['started_utc'] <= row['started_utc'] <= row['ended_utc'] <= parent['ended_utc'] <= seal['created_utc']
            for relative, sha in row['source'].items():
                receipts({'path': row['source_snapshots'][relative], 'sha256': sha})
            for relative, sha in result['source'].items():
                receipts({'path': str(ROOT / relative), 'sha256': sha})
            assert result['data'] == artifact(data['_path'])
            receipts(result['initial_source'])
            receipts(result['parent'])
            curve = read_json(HERE / 'runs' / result['id'] / 'curve.json')
            best, epoch, stale, last_lr = curve_policy(curve, spec)
            assert result['best_val_mse'] == best and result['selected_epoch'] == epoch
            assert result['epochs_completed'] == curve[-1]['epoch']
            assert result['cumulative_model_steps'] == curve[-1]['steps'] == result['actual_updates'] + result['carried_steps']
            assert result['initial_val_mse'] == curve[0]['val_mse']
            assert result['upper_epoch_limit_improving'] == (curve[-1]['epoch'] == 120 and curve[-1]['epoch'] - epoch < 6)
            assert result['selected_nonstep0'] == (epoch > 0)
            initial = torch.load(result['initial_checkpoint']['path'], map_location='cpu', weights_only=False)
            saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            last = torch.load(result['last_checkpoint']['path'], map_location='cpu', weights_only=False)
            for checkpoint in (initial, saved, last):
                assert checkpoint['spec'] == spec and checkpoint['model_config'] == result['model_config']
                assert checkpoint['initial_receipt'] == result['initial_source'] and checkpoint['parent_receipt'] == result['parent']
                assert checkpoint['data_manifest_sha256'] == digest(HERE / 'data_manifest.json')
                state = checkpoint['state']
                assert sum(v.numel() for v in state.values()) == result['trainable_parameters'] == 294912
                assert all('lora_' in name and torch.isfinite(value).all() for name, value in state.items())
            assert initial['epoch'] == initial['steps'] == 0
            hashes = tensor_hashes(initial['state'])
            assert hashes == result['initial_state_hashes'] == result['initial_source']['tensor_hashes']
            assert all(torch.count_nonzero(value) == 0 for name, value in initial['state'].items() if 'lora_B.' in name)
            if result['initial_source']['historical_tensor_reuse']:
                historical = torch.load(result['initial_source']['path'], map_location='cpu', weights_only=False)
                equal_state(historical['state'], initial['state'])
                del historical
            key = (dataset, spec['seed'])
            paired = {'hashes': hashes, 'parent': result['parent'], 'source': result['initial_source']}
            if key in initials:
                equal_state(initials[key], paired)
            initials[key] = paired
            assert saved['epoch'] == epoch and saved['steps'] == epoch * 128
            assert last['epoch'] == result['epochs_completed'] and last['steps'] == result['cumulative_model_steps']
            assert last['boundary'] == 'post-VAL/post-scheduler/complete epoch'
            assert last['best'] == best and last['best_epoch'] == epoch and last['stale'] == stale
            assert last['history'] == curve and last['schedules'] == result['schedules']
            equal_state(last['selected_checkpoint'], saved)
            close(last['optimizer']['param_groups'][0]['lr'], last_lr, 'Final scheduled learning rate')
            assert last['optimizer']['param_groups'][0]['weight_decay'] == 0
            assert all(float(state['step']) == last['steps'] for state in last['optimizer']['state'].values())
            assert set(last['rng']) == {'torch', 'cuda', 'numpy', 'python'}
            assert result['restart_roundtrip'] == {'model_optimizer_scheduler_rng': True, 'extra_updates': 0}
            for name, value in last['state'].items():
                delta = value - initial['state'][name]
                assert result['changed_parameter_scalars'][name] == int(torch.count_nonzero(delta))
                close(result['parameter_max_updates'][name], float(delta.abs().max()), 'Actual last-state parameter change', atol=0, rtol=0)
                close(result['selected_parameter_max_updates'][name], float((saved['state'][name] - initial['state'][name]).abs().max()),
                      'Selected parameter change', atol=0, rtol=0)
            assert any(result['changed_parameter_scalars'].values())
            assert result['selected_adapter_changed'] == any(value > 0 for value in result['selected_parameter_max_updates'].values())
            assert result['initial_f0_parity']['passed'] and result['replay_error'] <= 1e-6
            freeze = result['frozen_parameters_verified']
            assert freeze['no_gradient'] and freeze['version_unchanged'] and freeze['optimizer_exclusion']
            assert freeze['state_sha256'] == freeze['restored_state_sha256']
            assert len(result['first_update_gradients']) >= 2
            for record in result['first_update_gradients'].values():
                assert all(item['finite'] and item['has_grad'] for item in record.values())
                assert any(item['nonzero'] > 0 for item in record.values())
            expected_probe = phase_sample(data['train_origins'], spec['seed'], 0,
                                          max(96, data['_phase_period']), period=data['_phase_period'])
            assert result['train_probe_origins'] == expected_probe.tolist()
            receipts(result['origin_receipt'])
            with np.load(result['origin_receipt']['path'], allow_pickle=False) as archive:
                assert np.array_equal(archive['train_probe'], expected_probe)
                assert np.array_equal(archive['val'], data['val_origins'])
            assert [entry['epoch'] for entry in result['schedules']] == list(range(1, result['epochs_completed'] + 1))
            for entry in result['schedules']:
                schedule = phase_sample(data['train_origins'], spec['seed'], entry['epoch'], 512, period=data['_phase_period'])
                sha = hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest()
                assert entry['sha256'] == sha
                key = (dataset, spec['seed'], entry['epoch'])
                assert key not in schedules or schedules[key] == sha
                schedules[key] = sha
            prediction = read_prediction(result['best_val_prediction'], data['val_origins'])
            close(prediction, last['selected_prediction'], 'Embedded complete-epoch selected predictions', atol=0, rtol=0)
            assert np.array_equal(last['selected_origins'], data['val_origins'])
            score = metrics(prediction, target, mask)
            close(score['mse'], best, 'Independent saved VAL checkpoint score')
            for key in ('initial_train_probe', 'selected_train_probe', 'final_train_probe'):
                assert result[key]['origins'] == len(expected_probe)
                probe_target, probe_mask = target_arrays(data, expected_probe)
                close(result[key]['target_counts'], probe_mask.sum((0, 1)), 'Fixed TRAIN probe mask counts', atol=0, rtol=0)
            results.append(result)
            del initial, saved, last
        job.heartbeat(dataset + ': saved states, strict selection, original schedules and VAL arithmetic checked')
    for dataset in DATASETS:
        group = choice['selected'][dataset]['temporal_lora']
        candidates = []
        for lr in (1e-4, 1e-3):
            members = sorted([r for r in results if r['spec']['dataset'] == dataset and r['spec']['lr'] == lr], key=lambda r: r['spec']['seed'])
            assert [r['spec']['seed'] for r in members] == list(SEEDS)
            candidates.append({'lr': lr, 'mean_best_val_mse': float(np.mean([r['best_val_mse'] for r in members])),
                               'run_ids': [r['id'] for r in members]})
        assert group['lr_candidates'] == candidates
        winner = min(candidates, key=lambda row: (row['mean_best_val_mse'], row['lr']))
        assert all(group[key] == winner[key] for key in ('lr', 'mean_best_val_mse', 'run_ids'))
        assert group['seeds'] == list(SEEDS)
        for index, identifier in enumerate(group['run_ids']):
            result = next(r for r in results if r['id'] == identifier)
            for a, b in (('checkpoints', 'checkpoint'), ('checkpoint_sha256', 'checkpoint_sha256'),
                         ('selected_epochs', 'selected_epoch'), ('initial_checkpoints', 'initial_checkpoint'),
                         ('best_val_prediction', 'best_val_prediction'), ('selected_nonstep0', 'selected_nonstep0'),
                         ('selected_adapter_changed', 'selected_adapter_changed')):
                assert group[a][index] == result[b]
    assert choice['matched_initial_groups'] == len(initials) == 8
    assert choice['matched_schedule_groups'] == len(schedules)
    assert [row['id'] for row in choice['all_neural_runs']] == [result_for(s)['id'] for s in specs]
    attempts = [row for row in ledger if row['category'] == 'neural_fit']
    assert 16 <= len(attempts) <= 20 and sum(bool(r['metadata'].get('technical_retry')) for r in attempts) <= 4
    assert all(row['status'] in ('complete', 'failed') and row['ended_utc'] <= seal['created_utc'] for row in attempts)
    assert len([row for row in attempts if row['status'] == 'complete']) == 16
    for failed in [row for row in attempts if row['status'] == 'failed']:
        successor = [r for r in attempts if r['status'] == 'complete' and r['metadata']['fit_id'] == failed['metadata']['fit_id']]
        assert len(successor) == 1 and successor[0]['metadata']['technical_retry']
        assert failed['ended_utc'] <= successor[0]['started_utc']
    consumed = sum(row['updates'] for row in attempts)
    completed_paths = sum(row['cumulative_model_steps'] for row in results)
    assert consumed >= completed_paths
    return {'canonical_fits': 16, 'actual_attempts': len(attempts), 'consumed_optimizer_updates': consumed,
            'completed_path_updates': completed_paths, 'failed_or_repeated_updates': consumed - completed_paths,
            'all_attempts_before_seal': True,
            'historical_initial_groups': sum(bool(row['source']['historical_tensor_reuse']) for row in initials.values()),
            'checkpoint_arithmetic_checked': True, 'frozen_and_replay_evidence': 'Recorded fit checks and source/state receipts; no model rerun',
            'complete_scope': True, 'independent_reproduction': False}


def predictions(job):
    value = read_json(HERE / 'evaluation01.json')
    manifest = read_json(HERE / 'prediction_manifest.json')
    seal = read_json(HERE / 'selection_seal.json')
    assert value['status'] == manifest['status'] == 'complete'
    assert value['prescribed_instances'] == value['scored_instances'] == manifest['prescribed_instances'] == 60
    assert value['independent_confirmation'] is False and value['model_runs'] == value['new_neural_fits'] == value['optimizer_updates'] == 0
    receipts(value['prediction_manifest'])
    receipts(value['selection_seal'])
    csv_scores, model_count = {}, 0
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        a, b = data['test_a_origins'], data['test_b_origins']
        origins = np.concatenate([a, b])
        assert np.all(np.diff(a) > 0) and np.all(np.diff(b) > 0) and a[-1] < b[0]
        target, mask = target_arrays(data, origins)
        indices = {'test_a': np.arange(len(a)), 'test_b': np.arange(len(a), len(origins)), 'combined': np.arange(len(origins))}
        unit = value['datasets'][dataset]
        assert len(unit['models']) == 15 and set(unit['role_summary']) == set(ROLES)
        assert unit['columns'] == list(map(str, data['columns']))
        assert unit['period_origin_counts'] == {period: len(index) for period, index in indices.items()}
        assert unit['data'] == artifact(data['_path'])
        assert unit['bootstrap'] == {'draws': 2000, 'seed': 9262026, 'block_origins': protocol()['units'][dataset]['block_origins'],
            'period_boundaries_crossed': False, 'paired_channels_methods_and_seeds': True, 'selection_uncertainty_included': False}
        original_rows = {row['id']: row for row in manifest['datasets'][dataset]['models']}
        sealed_rows = {row['id']: row for row in seal['datasets'][dataset]['rows']}
        assert set(unit['models']) == set(original_rows) == set(sealed_rows)
        arrays, by_family, members = {}, defaultdict(list), defaultdict(list)
        receipts(unit['origin_terms'])
        with np.load(resolve(unit['origin_terms']['path']), allow_pickle=False) as stored_terms:
            assert np.array_equal(stored_terms['origins'], origins)
            for identifier, row in unit['models'].items():
                assert row['id'] == identifier and row['dataset'] == dataset and row['status'] == 'complete'
                assert all(row[key] == value for key, value in original_rows[identifier].items())
                assert all(row[key] == value for key, value in sealed_rows[identifier].items() if key != 'status')
                prediction = read_prediction(row['prediction'], origins)
                assert prediction.dtype == np.float32 and prediction.shape == target.shape
                scores = {period: metrics(prediction[index], target[index], mask[index]) for period, index in indices.items()}
                for period, score in scores.items():
                    for key, expected in score.items():
                        close(row['periods'][period][key], expected, identifier + '/' + period + '/' + key)
                    csv_scores[(dataset, identifier, period)] = score
                for key in ('squared_error_sums', 'absolute_error_sums', 'signed_error_sums', 'target_counts'):
                    close(scores['combined'][key], scores['test_a'][key] + scores['test_b'][key], 'Original period pooling')
                error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
                for key, expected in (('squared', (error * error).sum(1)), ('absolute', np.abs(error).sum(1)),
                                      ('signed', error.sum(1)), ('count', mask.sum(1))):
                    close(stored_terms[identifier + '__' + key], expected, 'Saved per-origin terms and masks')
                if row['reused']:
                    assert row['reference_score_replay']['passed']
                    receipts(row['source_evaluation'])
                    source = read_json(resolve(row['source_evaluation']['path']))['datasets'][dataset]['models'][row['source_model_id']]
                    for period in PERIODS:
                        for metric in ('mse', 'mae', 'signed_mean_error'):
                            close(scores[period][metric], source['periods'][period][metric], 'Reused source score')
                else:
                    assert row['seal_sha256'] == digest(HERE / 'selection_seal.json')
                arrays[(row['family'], row['seed'])] = prediction
                by_family[row['family']].append(scores)
                members[row['family']].append(row['seed'])
                model_count += 1
        for seed in SEEDS:
            direct = arrays[('direct', seed)]
            fine = direct - np.repeat(direct.reshape(len(origins), 12, 4, DATASETS[dataset][0]).mean(2), 4, axis=1)
            coarse = arrays[('coarse_only', seed)]
            close(coarse, np.repeat(coarse[:, ::4], 4, axis=1), 'Repeated selected coarse forecast', atol=0, rtol=0)
            close(arrays[('temporal_lora', seed)], coarse + fine, 'Same-selected coarse plus frozen DIRECT fine', atol=1e-5, rtol=1e-4)
            initial_coarse = arrays[('temporal_f0', seed)] - fine
            close(initial_coarse, np.repeat(initial_coarse[:, ::4], 4, axis=1), 'Frozen temporal-F0 block structure', atol=1e-5, rtol=1e-4)
        for family in ROLES:
            expected_seeds = [None] if family == 'f0' else list(SEEDS)
            assert sorted(members[family], key=str) == sorted(expected_seeds, key=str)
            summary = unit['role_summary'][family]
            assert summary['status'] == 'available' and summary['aggregation'] == 'Mean seed losses, not ensemble'
            for period in PERIODS:
                for metric in by_family[family][0][period]:
                    close(summary['periods'][period][metric], np.mean([r[period][metric] for r in by_family[family]], axis=0), 'Mean seed losses')
        expected_pairs = {'temporal_lora__vs__' + family for family in ROLES if family != 'temporal_lora'}
        assert set(unit['comparisons']) == expected_pairs
        for pair in unit['comparisons'].values():
            assert pair['status'] == 'available' and pair['candidate'] == 'temporal_lora'
            for period in PERIODS:
                for metric in ('mse', 'mae'):
                    left = unit['role_summary'][pair['candidate']]['periods'][period][metric]
                    right = unit['role_summary'][pair['reference']]['periods'][period][metric]
                    reported = pair['periods'][period][metric]
                    close([reported['candidate'], reported['reference'], reported['absolute_difference']], [left, right, left - right], 'Comparison point estimates')
                    close(reported['relative_change_percent'], 100 * (left - right) / right, 'Explicit reference denominator')
                    for field in ('conditional_block95_absolute', 'conditional_block95_relative_percent'):
                        ci = np.asarray(reported[field])
                        assert ci.shape == (2,) and np.isfinite(ci).all() and ci[0] <= ci[1]
        job.heartbeat(dataset + ': saved predictions, masks, composition and seed/period arithmetic checked')
    with (HERE / 'evaluation01_models.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(csv_scores) == 180
    for row in rows:
        for metric in ('mse', 'mae', 'signed_mean_error'):
            close(float(row[metric]), csv_scores[(row['dataset'], row['id'], row['period'])][metric], 'Model CSV')
    with (HERE / 'evaluation01_channels.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == sum(DATASETS[dataset][0] for dataset, _, _ in csv_scores)
    for row in rows:
        score = csv_scores[(row['dataset'], row['id'], row['period'])]
        index = value['datasets'][row['dataset']]['columns'].index(row['channel'])
        for key, field in (('mse', 'channel_mse'), ('mae', 'channel_mae'), ('signed_mean_error', 'channel_bias'), ('target_count', 'target_counts')):
            close(float(row[key]), score[field][index], 'Channel CSV')
    return {'instances': model_count, 'model_period_scores': 180, 'source_replay_and_temporal_composition': True,
            'original_masks_and_pooled_counts': True, 'seed_mean_not_ensemble': True,
            'bootstrap_intervals_recomputed': False, 'complete_scope': True, 'independent_reproduction': False}


def rotated(values, block):
    values = list(values)
    return values if block == 0 else values[::-1] if block == 1 else values[1:] + values[:1]


def cost_configs(family, dataset):
    c, k, _ = DATASETS[dataset]
    if family == 'direct':
        return [(1, None, 'direct'), (4, None, 'direct')]
    width = k if family == 'a' else c
    values = [(1, width, 'full'), (4, 4 * width, 'full')]
    return values + ([(4, k, 'chunk_k')] if family == 'a' else
                     [(1, k, 'chunk_k'), (4, 4 * k, 'chunk_4k'), (4, k, 'chunk_k')])


def expected_grid(device):
    seal, grid = read_json(HERE / 'selection_seal.json'), []
    for block in range(3):
        for dataset in rotated(DATASETS, block):
            allowed = ('direct',) if device == 'cpu' else ('temporal_lora', 'direct', 'a', 'f0', 'full_mse', 'native')
            members = [r for r in seal['datasets'][dataset]['rows'] if r['family'] in allowed]
            assert len(members) == (2 if device == 'cpu' else 11)
            ordered = [(r, None) for r in rotated(members, block)]
            if device == 'cuda':
                f0 = next(r for r in members if r['family'] == 'f0')
                ordered = [(f0, 'pre')] + ordered + [(f0, 'post')]
            for model, sentinel in ordered:
                entries = cost_configs(model['family'], dataset)
                entries = [r for r in entries if r[2] == 'full'] if sentinel else rotated(entries, block)
                for batch, chunk, execution in entries:
                    row = {key: model[key] for key in ('dataset', 'id', 'family', 'seed')}
                    row.update(device=device, block=block, sentinel=sentinel, batch=batch, chunk_rows=chunk, execution=execution)
                    row['key'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
                    row['planned_sequence'] = len(grid)
                    grid.append(row)
    assert len(grid) == (588 if device == 'cuda' else 48)
    assert len({r['key'] for r in grid}) == len(grid)
    return grid


def description(values):
    values = np.asarray(values)
    return dict(zip(('median', 'q25', 'q75', 'minimum', 'maximum'),
                    (np.median(values), np.quantile(values, .25), np.quantile(values, .75), np.min(values), np.max(values))))


def parity_receipt(check):
    assert check['pass'] and check['violating_elements'] == 0
    assert check['atol'] == 1e-5 and check['rtol'] == 1e-4
    assert np.isfinite([check['max_absolute'], check['max_relative']]).all()


def cost_groups(rows):
    grouped, instances = defaultdict(list), []
    for row in rows:
        if row['sentinel'] is None:
            grouped[tuple(row[k] for k in ('dataset', 'id', 'family', 'seed', 'batch', 'execution', 'chunk_rows'))].append(row)
    for key, members in grouped.items():
        if len(members) != 3:
            continue
        assert {r['block'] for r in members} == {0, 1, 2}
        instances.append({**{k: members[0][k] for k in ('dataset', 'id', 'family', 'seed', 'batch', 'execution', 'chunk_rows', 'device')},
            'median_block_median_ms': np.median([r['milliseconds_per_origin']['median'] for r in members]),
            'median_block_median_origins_s': np.median([r['origins_per_second']['median'] for r in members]),
            'block_ms': [r['milliseconds_per_origin']['median'] for r in sorted(members, key=lambda r: r['block'])],
            'peak_allocated_bytes': [r['peak_allocated_bytes'] for r in members],
            'peak_reserved_bytes': [r['peak_reserved_bytes'] for r in members], 'model': members[0]['model']})
    return instances


def check_cost_summary(summary, rows, device, complete):
    if not complete:
        assert summary == {'status': 'budget_stopped', 'complete_rows': len(rows),
                           'expected_rows': 588 if device == 'cuda' else 48, 'complete_frontier': False}
        return
    assert summary['status'] == 'complete' and summary['device'] == device
    assert summary['session_ids'] == sorted({r['job_id'] for r in rows})
    expected = cost_groups(rows)
    assert len(summary['instances']) == len(expected)
    for actual, value in zip(summary['instances'], expected):
        for key in value:
            if key in ('median_block_median_ms', 'median_block_median_origins_s', 'block_ms'):
                close(actual[key], value[key], 'Three-block instance arithmetic')
            else:
                assert actual[key] == value[key]
    grouped = defaultdict(list)
    for row in expected:
        grouped[tuple(row[k] for k in ('dataset', 'family', 'batch', 'execution', 'chunk_rows'))].append(row)
    assert len(summary['groups']) == len(grouped)
    for actual, (key, members) in zip(summary['groups'], grouped.items()):
        assert tuple(actual[k] for k in ('dataset', 'family', 'batch', 'execution', 'chunk_rows')) == key
        assert len(members) == (1 if key[1] == 'f0' else 2)
        assert actual['ids'] == [r['id'] for r in members] and actual['device'] == device
        close(actual['mean_seed_median_ms'], np.mean([r['median_block_median_ms'] for r in members]), 'Mean seed latency')
        close(actual['mean_seed_median_origins_s'], np.mean([r['median_block_median_origins_s'] for r in members]), 'Mean seed throughput')
        for kind in ('allocated', 'reserved'):
            values = [v for row in members for v in row['peak_' + kind + '_bytes'] if v is not None]
            if values:
                close(actual['mean_seed_median_peak_' + kind + '_bytes'],
                      np.mean([np.median(row['peak_' + kind + '_bytes']) for row in members]), 'Mean seed peak memory')
                assert actual[kind + '_range_bytes'] == [min(values), max(values)]
            else:
                assert actual['mean_seed_median_peak_' + kind + '_bytes'] is None and actual[kind + '_range_bytes'] is None
    sentinels = []
    if device == 'cuda':
        for dataset in DATASETS:
            for block in range(3):
                for batch in (1, 4):
                    pair = {r['sentinel']: r for r in rows if r['dataset'] == dataset and r['block'] == block
                            and r['batch'] == batch and r['sentinel'] is not None}
                    pre, post = [pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post')]
                    sentinels.append({'dataset': dataset, 'block': block, 'batch': batch,
                                      'pre_ms': pre, 'post_ms': post, 'post_relative_change': post / pre - 1})
    assert len(summary['sentinels']) == len(sentinels)
    for actual, expected in zip(summary['sentinels'], sentinels):
        for key, value in expected.items():
            if isinstance(value, str):
                assert actual[key] == value
            else:
                close(actual[key], value, 'F0 sentinel drift')


def check_csv(receipt, expected):
    receipts(receipt)
    with resolve(receipt['path']).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected)
    for actual, source in zip(rows, expected):
        assert set(actual) == set(source)
        for key, value in source.items():
            if value is None:
                assert actual[key] == ''
            elif isinstance(value, (list, dict)):
                assert json.loads(actual[key]) == value
            elif isinstance(value, (float, int)):
                close(float(actual[key]), value, 'Report CSV ' + key)
            else:
                assert actual[key] == value


def cost_report(job):
    binding = read_json(HERE / 'cost_binding.json')
    bound = digest(HERE / 'cost_binding.json')
    assert binding['settings'] == protocol()['cost']
    for name, sha in binding['files'].items():
        receipts({'path': str(HERE / name), 'sha256': sha})
    prepared = {}
    for dataset in DATASETS:
        data = load_data(dataset)
        origins = data['val_origins'][:24]
        inputs = np.stack([data['x'][o - 512:o] for o in origins]).astype(np.float32)
        assert binding['inputs'][dataset] == {'origins': origins.tolist(), 'sha256': hashlib.sha256(inputs.tobytes()).hexdigest()}
        prepared[dataset] = (origins, *target_arrays(data, origins))
    ledger = read_json(HERE / 'ledger.json')['jobs']
    sealed = {r['id']: r for unit in read_json(HERE / 'selection_seal.json')['datasets'].values() for r in unit['rows']}
    cost_gpu_seconds = sum(r['elapsed_s'] for r in ledger if r['category'] == 'gpu' and r['metadata'].get('v16_cost_campaign'))
    assert cost_gpu_seconds <= protocol()['cost']['reserve_seconds']
    raw, summaries, planned, seen_models = {}, {}, {}, set()
    for device in ('cuda', 'cpu'):
        expected = expected_grid(device)
        plan = read_json(HERE / f'cost_{device}_grid.json')
        assert plan == {'binding_sha256': bound, 'rows': expected}
        saved = read_json(HERE / f'cost_{device}_rows.json')
        rows = saved['rows']
        assert saved['binding_sha256'] == bound and saved['expected_rows'] == len(expected)
        assert saved['status'] in ('complete', 'budget_stopped') and len(rows) <= len(expected)
        complete = saved['status'] == 'complete'
        assert complete == (len(rows) == len(expected))
        if not complete:
            guard = read_json(HERE / 'cost_guard.json')
            assert device == 'cuda' and guard['status'] == 'budget_stopped'
            assert guard['complete_rows'] == len(rows) and guard['expected_rows'] == len(expected)
            assert guard['reason'] == 'V16 cost time cap: preserve partial rows'
            assert guard['elapsed_seconds'] >= protocol()['cost']['reserve_seconds'] - 30
            campaign = [r for r in ledger if r['category'] == 'gpu' and r['metadata'].get('v16_cost_campaign')]
            assert guard['elapsed_seconds'] <= sum(r['elapsed_s'] for r in campaign) + 1
        for row, intended in zip(rows, expected):
            assert all(row[k] == v for k, v in intended.items()), 'Only the exact fixed-grid prefix is eligible'
            assert row['status'] == 'complete' and row['binding_sha256'] == bound
            assert read_json(HERE / 'cost_rows' / device / (row['key'] + '.json')) == row
            record = ledger[row['job_id']]
            assert record['status'] in ('complete', 'failed') and record['category'] == ('gpu' if device == 'cuda' else 'cpu_analysis')
            assert record['metadata']['v16_cost_campaign']
            for relative, sha in record['source'].items():
                receipts({'path': record['source_snapshots'][relative], 'sha256': sha})
            assert row['warmup_calls'] == 10 and row['passes'] == 3
            times = np.asarray(row['seconds_per_24_origins'])
            assert times.shape == (3,) and np.isfinite(times).all() and np.all(times > 0)
            for key, values in (('milliseconds_per_origin', times * 1000 / 24), ('origins_per_second', 24 / times)):
                for name, value in description(values).items():
                    close(row[key][name], value, 'Raw timing arithmetic')
            if device == 'cuda':
                assert 0 < row['final_allocated_bytes'] <= row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                assert row['resident_after_load']['allocated_bytes'] <= row['peak_allocated_bytes']
            else:
                assert row['resident_after_load'] is None and row['peak_allocated_bytes'] is None and row['peak_reserved_bytes'] is None
            info = row['model']
            receipts(info)
            assert info['checkpoint'] == sealed[row['id']].get('checkpoint')
            assert info['deployment_tensor_bytes'] == info['parameter_bytes'] + info['buffer_bytes']
            assert info['historical_adaptation_registered_parameters'] == {
                'direct': 24624, 'a': 312832, 'f0': 0, 'full_mse': 294912, 'native': 294912, 'temporal_lora': 319536}[row['family']]
            assert info['this_campaign_adaptation_registered_parameters'] == (294912 if row['family'] == 'temporal_lora' else 0)
            parity_receipt(info['merge_export_parity'])
            if 'replay' in info:
                assert info['replay']['pass'] and info['replay']['max_absolute'] <= 1e-6
                assert info['replay']['atol'] == 1e-6 and info['replay']['rtol'] == 0
            if row['family'] == 'temporal_lora':
                assert info['new_lora_registered_parameters'] == 294912 and info['historical_direct_fit_parameters'] == 24624
                assert info['temporal_stride'] == 4 and info['inner_context'] == 128 and info['inner_output_crop'] == 12
                assert info['native_output'] == 64 and info['all_original_channels'] and not info['forecast_cache_in_timing']
            identity = (device, row['id'])
            if identity not in seen_models:
                archive = read_json(resolve(info['parity_receipt']['path']))
                assert archive['binding_sha256'] == bound
                origins, target, mask = prepared[row['dataset']]
                prediction = read_prediction(archive['prediction'], origins)
                score = metrics(prediction, target, mask)
                assert [(r['batch'], r['chunk_rows'], r['execution']) for r in info['configuration_parity']] == cost_configs(row['family'], row['dataset'])
                for check in info['configuration_parity']:
                    parity_receipt(check['parity'])
                    close([check['val_scores']['mse'], check['val_scores']['mae']], [score['mse'], score['mae']],
                          'Stored full/chunk VAL scores', atol=1e-5, rtol=1e-4)
                if device == 'cpu':
                    check = info['cpu_gpu_parity']
                    if 'pass' in check:
                        parity_receipt(check)
                        gpu = read_json(resolve(info['parity_receipt']['path']).parent.parent / 'cuda' / (row['id'] + '.json'))
                        close(prediction, read_prediction(gpu['prediction'], origins), 'Saved CPU/GPU output parity', atol=1e-5, rtol=1e-4)
                    else:
                        assert check['status'] == 'unavailable' and raw['cuda']['status'] == 'budget_stopped'
                seen_models.add(identity)
        summary = read_json(HERE / f'cost_{device}_summary.json')
        check_cost_summary(summary, rows, device, complete)
        raw[device], summaries[device], planned[device] = saved, summary, expected
        job.heartbeat(device + ': fixed grid, raw passes, recorded parity and cost aggregation checked')
    combined = read_json(HERE / 'cost_summary.json')
    receipts(combined['binding'])
    complete = all(raw[d]['status'] == 'complete' for d in raw)
    assert combined['gpu'] == summaries['cuda'] and combined['cpu'] == summaries['cpu']
    assert (combined['status'] == 'complete') == complete
    assert not combined['historical_ratios'] and not combined['accuracy_dependent_dataset_filtering']
    report = read_json(HERE / 'report_values.json')
    evaluation = read_json(HERE / 'evaluation01.json')
    receipts(report)
    for name, sha in report['source_files'].items():
        receipts({'path': str(HERE / name), 'sha256': sha})
    assert report['schema'] == 'v16_report_values_1' and report['roles'] == list(ROLES) and report['periods'] == list(PERIODS)
    assert report['status'] == ('complete' if complete else 'partial_cost')
    assert report['cost_status']['complete_for_report'] == complete
    assert report['cost_status']['top_level'] == ('complete' if complete else 'partial')
    assert report['accuracy_rows'] == len(report['accuracy']) == 276
    assert report['paired_difference_rows'] == len(report['paired_comparisons']) == 168
    assert report['cost_rows'] == len(report['cost']) == 108
    accuracy_keys = set()
    for row in report['accuracy']:
        unit = evaluation['datasets'][row['dataset']]
        key = (row['dataset'], row['aggregation'], tuple(row['ids']), row['period'])
        assert key not in accuracy_keys
        accuracy_keys.add(key)
        if row['aggregation'] == 'mean_seed_loss':
            source = unit['role_summary'][row['family']]
            assert row['ids'] == source['ids'] and row['seed'] is None
        else:
            assert row['aggregation'] == 'single_model' and len(row['ids']) == 1
            source = unit['models'][row['ids'][0]]
            assert row['family'] == source['family'] and row['seed'] == source['seed']
        for name in ('mse', 'mae', 'signed_mean_error'):
            close(row[name], source['periods'][row['period']][name], 'Report accuracy')
        for name in ('mse', 'mae'):
            denominator = unit['role_summary']['full_mse']['periods'][row['period']][name]
            close(row[name + '_reference_full_mse_role'], denominator, 'Report external denominator')
            close(row[name + '_relative_change_percent'], 100 * (row[name] - denominator) / denominator, 'Report relative gap')
    paired_keys = set()
    for row in report['paired_comparisons']:
        key = tuple(row[k] for k in ('dataset', 'comparison', 'period', 'metric'))
        assert key not in paired_keys
        paired_keys.add(key)
        source = evaluation['datasets'][key[0]]['comparisons'][key[1]]['periods'][key[2]][key[3]]
        assert all(row[k] == v for k, v in source.items())
    for device, alias in (('cuda', 'gpu'), ('cpu', 'cpu')):
        actual = {r['key']: r for r in raw[device]['rows']}
        grouped = defaultdict(list)
        for row in planned[device]:
            if row['sentinel'] is None:
                grouped[tuple(row[k] for k in ('dataset', 'family', 'batch', 'execution', 'chunk_rows'))].append(row)
        listed = [r for r in report['cost'] if r['device'] == device]
        assert len(listed) == len(grouped)
        coverage = report['cost_coverage'][alias]
        assert coverage['completed_rows'] == len(actual) and coverage['expected_rows'] == len(planned[device])
        assert coverage['missing_row_keys'] == [r['key'] for r in planned[device] if r['key'] not in actual]
        assert coverage['recorded_status'] == raw[device]['status']
        assert coverage['status'] == ('complete' if len(actual) == len(planned[device]) else 'partial')
        assert report['cost_status'][alias] == coverage['status']
        for row, (key, members) in zip(listed, grouped.items()):
            assert tuple(row[k] for k in ('dataset', 'family', 'batch', 'execution', 'chunk_rows')) == key
            missing = [r['key'] for r in members if r['key'] not in actual]
            assert row['missing_row_keys'] == missing and row['planned_rows'] == len(members)
            assert row['completed_rows'] == len(members) - len(missing)
            for name in ('mse', 'mae'):
                close(row[name], evaluation['datasets'][row['dataset']]['role_summary'][row['family']]['periods']['combined'][name], 'Cost/report accuracy link')
            if missing:
                assert row['status'] == 'PENDING'
                assert all(value is None for name, value in row.items() if name.startswith(('mean_seed_', 'allocated_', 'reserved_')))
                continue
            assert row['status'] == 'complete_group'
            values = [actual[r['key']] for r in members]
            ids = list(dict.fromkeys(r['id'] for r in members))
            assert row['ids'] == ids
            by_id = [[r for r in values if r['id'] == identifier] for identifier in ids]
            for field, target_name, low, high in (('milliseconds_per_origin', 'mean_seed_median_ms', 'ms_low', 'ms_high'),
                                                ('origins_per_second', 'mean_seed_median_origins_s', 'origins_s_low', 'origins_s_high')):
                medians = [r[field]['median'] for r in values]
                close(row[target_name], np.mean([np.median([r[field]['median'] for r in group]) for group in by_id]), 'Report paired timing')
                close([row[low], row[high]], [min(medians), max(medians)], 'Report observed ranges')
            for kind in ('allocated', 'reserved'):
                memory = [r['peak_' + kind + '_bytes'] for r in values]
                if device == 'cuda':
                    close(row[kind + '_bytes'], np.mean([np.median([r['peak_' + kind + '_bytes'] for r in group]) for group in by_id]), 'Report memory')
                    close([row[kind + '_low_bytes'], row[kind + '_high_bytes']], [min(memory), max(memory)], 'Report memory range')
                else:
                    assert row[kind + '_bytes'] is None
            for name in ('deployed_parameters', 'parameter_bytes', 'buffer_bytes', 'deployment_tensor_bytes',
                         'this_campaign_adaptation_registered_parameters', 'historical_adaptation_registered_parameters'):
                assert all(r['model'][name] == row[name] for r in values)
        assert coverage['complete_groups'] == sum(r['status'] == 'complete_group' for r in listed)
        assert coverage['planned_groups'] == len(listed)
        assert len(coverage['sentinel_drift']) == (24 if device == 'cuda' else 0)
        for sentinel in coverage['sentinel_drift']:
            pair = {r['sentinel']: r for r in actual.values() if r['sentinel'] is not None and all(r[k] == sentinel[k] for k in ('dataset', 'block', 'batch'))}
            if set(pair) == {'pre', 'post'}:
                assert sentinel['status'] == 'complete_pair'
                pre, post = [pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post')]
                close([sentinel['pre_ms'], sentinel['post_ms'], sentinel['relative_change_percent']], [pre, post, 100 * (post - pre) / pre], 'Report drift')
            else:
                assert sentinel['status'] == 'PENDING'
    check_csv(report['tables']['comparison'], report['accuracy'])
    check_csv(report['tables']['paired_differences'], report['paired_comparisons'])
    check_csv(report['tables']['cost_comparison'], report['cost'])
    assert set(report['figures']) == {'accuracy_by_period', 'accuracy_cost'}
    for figure in report['figures'].values():
        assert set(figure) == {'png', 'svg'}
        assert resolve(figure['png']['path']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert '<svg' in resolve(figure['svg']['path']).read_text(encoding='utf-8')
    assert report['no_new_numeric_claims'] == {'fit_count': 0, 'model_runs': 0, 'new_forecast_scores': 0,
                                               'bootstrap_recomputed': False, 'frontier_claim': False}
    return {'gpu_rows': len(raw['cuda']['rows']), 'cpu_rows': len(raw['cpu']['rows']),
            'missing_gpu_rows': [r for r in planned['cuda'][len(raw['cuda']['rows']):]],
            'raw_passes_and_three_block_seed_aggregation': True, 'parity_scope': 'Saved model and timing-path receipts; no new forward',
            'complete_scope': complete, 'status': 'complete' if complete else 'partial_cost', 'independent_reproduction': False}


def check_budget(current_job=None):
    value = budget_snapshot()
    limits = value['limits']
    assert value['real_fit_attempts'] <= limits['real_fit'] and value['technical_reserve_attempts'] <= limits['technical_reserve']
    assert value['coefficient_fit_attempts'] == 0 and value['synthetic_sessions'] <= limits['synthetic_sessions']
    for category in ('gpu', 'cpu_analysis', 'cpu_check'):
        assert value[category + '_seconds'] <= limits[category + '_seconds']
    assert value['storage_bytes'] <= limits['storage_bytes'] and limits['download_bytes'] == 0
    assert all(row['id'] == current_job and row['pid_alive'] for row in value['active_jobs']), 'Other jobs are still running'
    jobs = read_json(HERE / 'ledger.json')['jobs']
    assert all(row['updates'] <= limits['synthetic_steps'] for row in jobs if row.get('synthetic'))
    return value


def input_receipts(group):
    names = ['verify_v16.py', 'runtime_v16.py', 'protocol.json', 'provenance.json', 'selected.json', 'selection_seal.json', 'test_exposure.json']
    if group == 'contracts':
        names += ['data_manifest.json', 'initial_manifest.json', 'reuse_manifest.json', 'train_v16.py', 'model_v16.py']
        names += [str(path.relative_to(HERE)) for path in sorted((HERE / 'runs').glob('*/result.json'))]
        names += [str(path.relative_to(HERE)) for path in sorted((HERE / 'runs').glob('*/curve.json'))]
    elif group == 'predictions':
        names += ['data_manifest.json', 'evaluate_v16.py', 'prediction_manifest.json', 'evaluation01.json', 'evaluation01_models.csv', 'evaluation01_channels.csv']
    else:
        names += ['cost_v16.py', 'report_v16.py', 'report_values.json', 'evaluation01.json', 'cost_binding.json', 'cost_summary.json']
        names += [f'cost_{device}_{kind}.json' for device in ('cuda', 'cpu') for kind in ('grid', 'rows', 'summary')]
        if (HERE / 'cost_guard.json').exists():
            names.append('cost_guard.json')
    return {name: artifact(HERE / name) for name in names}


def final_summary():
    groups, stale = {}, []
    jobs = read_json(HERE / 'ledger.json')['jobs']
    for group in GROUPS:
        path = HERE / f'verification_{group}.json'
        if not path.exists():
            continue
        row = read_json(path)
        current = row['status'] == 'passed' and jobs[row['job_id']]['status'] == 'complete'
        if current:
            try:
                for item in row['inputs'].values():
                    artifact(item['path'], item['sha256'])
            except (OSError, RuntimeError, ValueError, AssertionError):
                current = False
        if not current:
            stale.append(group)
        groups[group] = {'status': row['status'], 'current': current, 'receipt': artifact(path), 'details': row.get('details')}
    passed = len(groups) == len(GROUPS) and not stale
    value = {'schema': 'v16_saved_artifact_checks_1', 'status': 'passed' if passed else 'incomplete',
             'complete_campaign_verified': bool(passed and all(row['details']['complete_scope'] for row in groups.values())),
             'groups': groups, 'missing_groups': [g for g in GROUPS if g not in groups], 'failed_or_stale_groups': stale,
             'independent_reproduction': False, 'model_runs': 0, 'optimizer_updates': 0,
             'scope': 'CPU self-check of saved source, states, schedules, predictions, arithmetic and timing receipts; not an independent scientific reproduction',
             'budget': budget_snapshot()}
    save_json(HERE / 'final_checks.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=GROUPS)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reserve-s', required=True, type=float)
    args = parser.parse_args()
    output = HERE / f'verification_{args.group}.json'
    if output.exists():
        prior = read_json(output)
        archive = HERE / f'verification_{args.group}_prior_job{prior["job_id"]}.json'
        if archive.exists():
            raise FileExistsError('Preserve previous verification evidence before another attempt')
        archive.write_bytes(output.read_bytes())
    result = {'group': args.group, 'status': 'failed', 'independent_reproduction': False}
    failure = None
    try:
        with Job('cpu_analysis' if args.group == 'predictions' else 'cpu_check', args.label, reserve_s=args.reserve_s,
                 metadata={'purpose': 'saved-artifact self-check', 'group': args.group, 'model_runs': 0, 'optimizer_updates': 0}) as job:
            result['job_id'] = job.id
            result['inputs'] = input_receipts(args.group)
            check_budget(job.id)
            result['details'] = globals()[args.group](job)
            result['budget'] = check_budget(job.id)
            result['status'] = 'passed'
    except Exception as error:
        failure = error
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    if 'job_id' not in result:
        raise failure
    save_json(output, result)
    final = final_summary()
    print(json.dumps({'group': args.group, 'status': result['status'], 'complete_campaign_verified': final['complete_campaign_verified']}), flush=True)
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
