"""CPU verification of saved confirmation artifacts; no model construction or fitting."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from runtime_confirmation_v12 import (HERE, CACHE, ROOT, SEEDS, Job, artifact, budget_snapshot,
                                      digest, load_data, protocol, read_json, save_json)

FAMILIES = ('level', 'a', 'full_mse', 'native', 'direct', 'f0')
ROLES = FAMILIES + ('a_main', 'a_last')
PERIODS = ('test_a', 'test_b', 'combined')


def close(actual, expected, label, atol=1e-10, rtol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(label + ': invalid shape or finite values')
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(label + ': values differ')


def saved_scores(prediction, target, observed):
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise AssertionError('Original prediction, target and observed-mask shapes differ')
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise AssertionError('Nonfinite prediction or observed target')
    count = observed.sum(axis=(0, 1))
    if np.any(count == 0):
        raise AssertionError('A fixed channel has no observed target')
    error = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    squared = np.sum(np.square(error), axis=(0, 1))
    absolute = np.sum(np.abs(error), axis=(0, 1))
    signed = np.sum(error, axis=(0, 1))
    return {'mse': float(np.mean(squared / count)), 'mae': float(np.mean(absolute / count)),
            'signed_mean_error': float(np.mean(signed / count)),
            'channel_mse': squared / count, 'channel_mae': absolute / count,
            'channel_bias': signed / count, 'squared_error_sums': squared,
            'absolute_error_sums': absolute, 'signed_error_sums': signed,
            'target_counts': count}


def prescribed_schedule(origins, seed, epoch, period, n=512):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    result = []
    for phase in range(period):
        group = origins[origins % period == phase]
        count = n // period + int(phase < n % period)
        result.extend(rng.choice(group, count, replace=False))
    return rng.permutation(result).astype('<i8')


class Verification:
    def __init__(self, job):
        self.job = job
        self.checked, self.groups, self.fit_results = {}, [], {}
        self.p = protocol()
        self.selection = None

    def receipt(self, value):
        path = Path(value['path'])
        if not path.is_absolute():
            path = ROOT / path
        key = str(path.resolve())
        if key not in self.checked:
            self.checked[key] = artifact(path)
        if self.checked[key]['sha256'] != value['sha256']:
            raise AssertionError('Receipt hash mismatch: ' + key)
        return path

    def receipts(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.receipt(value)
            for child in value.values():
                self.receipts(child)
        elif isinstance(value, list):
            for child in value:
                self.receipts(child)

    def run(self, name, function):
        try:
            self.job.check_limits()
            details = function()
            self.groups.append({'name': name, 'status': 'passed', 'details': details})
        except FileNotFoundError as error:
            self.groups.append({'name': name, 'status': 'missing', 'error': str(error)})
        except Exception as error:
            self.groups.append({'name': name, 'status': 'failed', 'error': str(error),
                                'traceback': traceback.format_exc()})
        self.job.heartbeat('confirmation final check: ' + name)

    def sources_and_contract(self):
        p = self.p
        assert 1 <= len(p['units']) <= 2
        assert len(p['neural_fits']) == 18 * len(p['units'])
        assert len({row['id'] for row in p['neural_fits']}) == len(p['neural_fits'])
        assert not p['coefficient_fits']
        for dataset, unit in p['units'].items():
            assert unit['K'] == (unit['C'] + 3) // 4
            contract = read_json(HERE / f'data_contract_confirmation_{dataset}.json')
            assert contract['unit'] == unit
            self.receipts(contract)
            group = [s for s in p['neural_fits'] if s['dataset'] == dataset]
            expected = {(family, seed, lr) for family in FAMILIES if family != 'f0'
                        for seed in SEEDS for lr in ([.001] if family == 'level' else [.0001, .001])}
            assert {(s['family'], s['seed'], s['lr']) for s in group} == expected
        no_q = read_json(HERE / 'confirmation_no_q_decision.json')
        assert set(no_q['datasets']) == set(p['units'])
        for row in no_q['datasets'].values():
            assert row['no_q_fit'] is False and row.get('test_error_used_for_decision', False) is False
        return {'units': len(p['units']), 'basic_fits': len(p['neural_fits']),
                'raw_sources_and_data_receipts_verified': True, 'no_q_claim_limited': True}

    def fits_and_selection(self):
        import torch
        from train_confirmation_v12 import result_for

        ledger = read_json(HERE / 'ledger.json')['jobs']
        data = {name: load_data(name, include_test=False) for name in self.p['units']}
        initial_groups, schedules, attempt_accounting = {}, {}, []
        for spec in self.p['neural_fits']:
            result = result_for(spec)
            self.fit_results[spec['id']] = result
            record = ledger[result['ledger_id']]
            assert record['status'] == 'complete' and record['category'] == 'neural_fit'
            assert record['metadata']['fit_id'] == spec['id']
            assert record['label'] == result['id'] and record['updates'] == result['actual_updates']
            assert result['actual_updates'] > 0
            assert result['cumulative_model_steps'] == result['actual_updates'] + result['carried_steps']
            assert result['cumulative_model_steps'] == result['epochs_completed'] * 128
            assert 1 <= result['epochs_completed'] <= 120
            attempts = [row for row in ledger if row['category'] == 'neural_fit'
                        and row['metadata'].get('fit_id') == spec['id']]
            assert sum(row['status'] == 'complete' for row in attempts) == 1
            assert attempts[-1]['id'] == record['id']
            consumed_updates = 0
            for row in attempts:
                attempt = read_json(HERE / 'confirmation_runs' / row['label'] / 'attempt.json')
                assert attempt['spec'] == spec and attempt['ledger_id'] == row['id']
                assert attempt['status'] == row['status'] and row['status'] in ('complete', 'failed')
                assert attempt['actual_updates'] == row['updates'] >= 0
                consumed_updates += attempt['actual_updates']
                if row['status'] == 'failed':
                    assert attempt['error'] and row.get('error')
                for relative, snapshot in row['source_snapshots'].items():
                    if Path(relative).name in ('model_confirmation_v12.py', 'train_confirmation_v12.py',
                                              'runtime_confirmation_v12.py', 'runtime_v12.py', 'confirmation_protocol.json'):
                        self.receipt({'path': snapshot, 'sha256': row['source'][relative]})
            assert consumed_updates >= result['cumulative_model_steps']
            if result['retry']:
                retry = result['retry']
                assert retry['previous_ledger_ids'] == [row['id'] for row in attempts[:-1]]
                assert all(row['status'] == 'failed' for row in attempts[:-1])
                assert retry['technical_retry'] is True and retry['reason'].strip()
                replacement = read_json(HERE / 'confirmation_replacements' / (spec['id'] + '.json'))
                self.receipt(replacement['result'])
                assert replacement['selected_attempt_id'] == result['id']
                assert replacement['technical_retry'] == retry
                if retry.get('resume_from'):
                    self.receipt(retry['resume_from'])
                    resumed = torch.load(retry['resume_from']['path'], map_location='cpu', weights_only=False)
                    assert resumed['spec'] == spec and resumed['steps'] == result['carried_steps']
                    assert resumed['steps'] == resumed['epoch'] * 128
                    assert resumed['boundary'] == 'post-VAL/post-scheduler/complete epoch'
                    assert resumed['initial_reference']['sha256'] == result['initial_source']['sha256']
                    del resumed
                else:
                    assert result['carried_steps'] == 0
            else:
                assert len(attempts) == 1 and result['carried_steps'] == 0
            attempt_accounting.append({'canonical_id': spec['id'], 'selected_attempt_id': result['id'],
                'attempt_count': len(attempts), 'failed_attempts': len(attempts) - 1,
                'actual_updates_consumed_across_attempts': consumed_updates,
                'selected_run_cumulative_model_steps': result['cumulative_model_steps'],
                'discarded_or_repeated_updates': consumed_updates - result['cumulative_model_steps'],
                'successful_attempt_updates': result['actual_updates'], 'carried_steps': result['carried_steps']})
            assert all(result['frozen_parameters_verified'][key] is True for key in
                       ('no_gradient', 'version_unchanged', 'optimizer_exclusion'))
            assert len(result['frozen_parameters_verified']['state_sha256']) == 64
            assert result['restart_roundtrip']['model_optimizer_scheduler_rng'] is True
            assert result['restart_roundtrip']['extra_updates'] == 0 and result['replay_error'] <= 1e-6
            expected_count = {'level': 17920, 'a': 294912, 'full_mse': 294912,
                              'native': 294912, 'direct': 24624}[spec['family']]
            assert result['trainable_parameters'] == expected_count
            assert any(value > 0 for value in result['changed_parameter_scalars'].values())
            for gradient in result['first_update_gradients'].values():
                finite = [v['max_abs'] for v in gradient.values() if v['max_abs'] is not None]
                assert finite and np.isfinite(finite).all()
            assert any(v['nonzero'] > 0 for grad in result['first_update_gradients'].values() for v in grad.values())
            for name in ('initial_source', 'initial_checkpoint', 'last_checkpoint', 'best_val_prediction', 'data'):
                self.receipt(result[name])
            self.receipt({'path': result['checkpoint'], 'sha256': result['checkpoint_sha256']})
            saved_initial = torch.load(result['initial_checkpoint']['path'], map_location='cpu', weights_only=False)
            saved_last = torch.load(result['last_checkpoint']['path'], map_location='cpu', weights_only=False)
            saved_best = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            assert all(saved['spec'] == spec for saved in (saved_initial, saved_last, saved_best))
            assert saved_last['epoch'] == result['epochs_completed'] and saved_last['steps'] == result['cumulative_model_steps']
            assert saved_best['epoch'] == result['selected_epoch']
            assert saved_last['boundary'] == 'post-VAL/post-scheduler/complete epoch'
            assert all(key in saved_last for key in ('optimizer', 'scheduler', 'rng'))
            for name, expected in result['initial_state_hashes'].items():
                array = saved_initial['state'][name].detach().cpu().contiguous().numpy()
                assert hashlib.sha256(array.tobytes()).hexdigest() == expected
            for name, changed in result['changed_parameter_scalars'].items():
                start, last = saved_initial['state'][name], saved_last['state'][name]
                assert int(torch.count_nonzero(start != last)) == changed
                close(float((last - start).abs().max()), result['parameter_max_updates'][name], name)
                close(float((saved_best['state'][name] - start).abs().max()),
                      result['selected_parameter_max_updates'][name], name)
            fixed = set(saved_initial['state']) - set(result['changed_parameter_scalars'])
            assert all(torch.equal(saved_initial['state'][name], saved_last['state'][name]) and
                       torch.equal(saved_initial['state'][name], saved_best['state'][name]) for name in fixed)
            if spec['family'] == 'a':
                assert result['initial_parent_parity']['passed'] is True
                self.receipt(result['initial_parent_parity']['parent_prediction'])
                self.receipt(result['parent'])
                parent = torch.load(result['parent']['path'], map_location='cpu', weights_only=False)
                assert parent['spec']['family'] == 'level' and parent['spec']['seed'] == spec['seed']
                assert parent['spec']['dataset'] == spec['dataset']
                assert all(torch.equal(value, saved_best['state'][name]) for name, value in parent['state'].items())
                del parent
            elif spec['family'] in ('full_mse', 'native'):
                assert result['initial_f0_parity']['passed'] is True
            curve = read_json(HERE / 'confirmation_runs' / result['id'] / 'curve.json')
            assert [r['epoch'] for r in curve] == list(range(result['epochs_completed'] + 1))
            chosen = min(curve, key=lambda row: (row['val_mse'], row['epoch']))
            assert chosen['epoch'] == result['selected_epoch']
            close(chosen['val_mse'], result['best_val_mse'], 'best epoch VAL')
            assert saved_last['stale'] >= 6 or result['epochs_completed'] == 120
            ds = data[spec['dataset']]
            origins = ds['val_origins']
            with np.load(result['best_val_prediction']['path'], allow_pickle=False) as archive:
                assert np.array_equal(archive['origins'], origins)
                prediction = archive['prediction']
            target = np.stack([ds['x'][int(o):int(o) + 48] for o in origins])
            mask = np.stack([ds['finite'][int(o):int(o) + 48] for o in origins]).astype(bool)
            close(saved_scores(prediction, target, mask)['mse'], result['best_val_mse'], 'saved VAL MSE', atol=1e-8)
            assert [r['epoch'] for r in result['schedules']] == list(range(1, result['epochs_completed'] + 1))
            for row in result['schedules']:
                key = (spec['dataset'], spec['seed'], row['epoch'])
                if key not in schedules:
                    schedule = prescribed_schedule(ds['train_origins'], spec['seed'], row['epoch'], ds['_phase_period'])
                    schedules[key] = hashlib.sha256(schedule.tobytes()).hexdigest()
                assert schedules[key] == row['sha256']
            probe = prescribed_schedule(ds['train_origins'], spec['seed'], 0, ds['_phase_period'], max(96, ds['_phase_period']))
            assert np.array_equal(probe, result['train_probe_origins'])
            kind = 'lora' if spec['family'] in ('a', 'full_mse', 'native') else spec['family']
            key = (spec['dataset'], spec['seed'], kind)
            if key in initial_groups:
                assert initial_groups[key] == result['initial_source']['sha256']
            initial_groups[key] = result['initial_source']['sha256']
            del saved_initial, saved_last, saved_best
            self.job.heartbeat('verified fit ' + result['id'])
        self.selection = read_json(HERE / 'selected_confirmation.json')
        parents = read_json(HERE / 'parents_selected.json')
        assert self.selection['test_used_for_selection'] is False and parents['test_used_for_selection'] is False
        assert len(self.selection['all_neural_runs']) == len(self.p['neural_fits'])
        assert {row['canonical_id'] for row in self.selection['all_neural_runs']} == set(self.fit_results)
        for row in self.selection['all_neural_runs']:
            assert all(value == self.fit_results[row['canonical_id']][key] for key, value in row.items())
        assert set(self.selection['selected']) == set(self.p['units'])
        for dataset, selected in self.selection['selected'].items():
            assert set(selected) == set(FAMILIES) - {'f0'}
            assert selected['level'] == parents['selected'][dataset]['level']
            for family, row in selected.items():
                candidates = [r for r in self.fit_results.values() if r['spec']['dataset'] == dataset and r['spec']['family'] == family]
                means = {lr: float(np.mean([r['best_val_mse'] for r in candidates if r['spec']['lr'] == lr]))
                         for lr in ({.001} if family == 'level' else {.0001, .001})}
                chosen_lr = min(means, key=lambda lr: (means[lr], lr))
                assert row['lr'] == chosen_lr and row['seeds'] == list(SEEDS)
                for index, seed in enumerate(SEEDS):
                    r = next(r for r in candidates if r['spec']['seed'] == seed and r['spec']['lr'] == chosen_lr)
                    assert row['run_ids'][index] == r['id'] and row['checkpoints'][index] == r['checkpoint']
                    assert row['checkpoint_sha256'][index] == r['checkpoint_sha256']
                    assert row['selected_epochs'][index] == r['selected_epoch']
                close(row['mean_best_val_mse'], means[chosen_lr], 'LR selection mean')
        return {'completed_canonical_fits': len(self.fit_results), 'matched_initial_groups': len(initial_groups),
                'recreated_epoch_schedules': len(schedules), 'saved_state_updates_verified': True,
                'actual_attempt_accounting': attempt_accounting,
                'actual_updates_consumed': sum(row['actual_updates_consumed_across_attempts'] for row in attempt_accounting),
                'discarded_or_repeated_updates': sum(row['discarded_or_repeated_updates'] for row in attempt_accounting),
                'full_backbone_freeze_evidence': 'Recorded in-fit hash/version/no-gradient checks, not a new model run'}

    def seal_and_exposure(self):
        assert len(self.fit_results) == len(self.p['neural_fits']), 'Fit verification must complete first'
        seal = read_json(HERE / 'confirmation_selection_seal.json')
        exposure = read_json(HERE / 'confirmation_test_exposure.json')
        assert exposure['seal_sha256'] == digest(HERE / 'confirmation_selection_seal.json')
        assert set(seal['datasets']) == set(self.p['units']) == set(exposure['datasets'])
        assert seal['test_used_for_selection'] is False
        assert seal['model_instances'] == 11 * len(self.p['units'])
        assert seal['scored_instances_including_fit0_q_references'] == 15 * len(self.p['units'])
        for name, sha in seal['files'].items():
            self.receipt({'path': str(HERE / name), 'sha256': sha})
        for row in seal['datasets'].values():
            self.receipt(row['data_contract'])
            self.receipt(row['test_arrays'])
        jobs = read_json(HERE / 'ledger.json')['jobs']
        fits = [j for j in jobs if j['category'] in ('neural_fit', 'coefficient_fit')]
        assert all(j['status'] in ('complete', 'failed') and j['ended_utc'] <= seal['created_utc'] for j in fits)
        assert seal['created_utc'] <= exposure['first_access_utc']
        assert all(j['started_utc'] < exposure['first_access_utc'] for j in fits)
        return {'all_fit_attempts_before_joint_seal': True, 'seal_before_first_TEST_access': True,
                'failed_fit_attempts_preserved': sum(j['status'] == 'failed' for j in fits)}

    def saved_prediction_metrics(self):
        assert self.selection is not None, 'Selection verification must complete first'
        seal = read_json(HERE / 'confirmation_selection_seal.json')
        manifest = read_json(HERE / 'confirmation_prediction_manifest.json')
        evaluation = read_json(HERE / 'confirmation_evaluation01.json')
        assert manifest['status'] == evaluation['status'] == 'complete'
        self.receipts(manifest)
        self.receipts(evaluation['tables'])
        self.receipt(evaluation['selection_seal'])
        self.receipt(evaluation['prediction_manifest'])
        assert set(manifest['datasets']) == set(evaluation['datasets']) == set(self.p['units'])
        scores_by_key, channel_by_key = {}, {}
        model_count, role_count = 0, 0
        for dataset in self.p['units']:
            data = load_data(dataset, include_test=True)
            origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
            split = len(data['test_a_origins'])
            indices = {'test_a': np.arange(split), 'test_b': np.arange(split, len(origins)), 'combined': np.arange(len(origins))}
            target = np.stack([data['x'][o:o + 48] for o in origins])
            mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
            rows = manifest['datasets'][dataset]['models']
            assert len(rows) == 11 and [r['id'] for r in rows] == [r['id'] for r in seal['datasets'][dataset]['rows']]
            reported = evaluation['datasets'][dataset]
            self.receipt(reported['origin_terms'])
            self.receipt(reported['data_arrays'])
            with np.load(reported['origin_terms']['path'], allow_pickle=False) as archive:
                assert np.array_equal(archive['origins'], origins)
                saved_terms = {key: archive[key] for key in archive.files if key != 'origins'}
            grouped, q_values, ids = defaultdict(list), {}, set()
            for row in rows:
                assert row['dataset'] == dataset
                if row['family'] == 'f0':
                    assert row['id'] == f'v12_confirmation_{dataset}_f0' and row['seed'] is None
                else:
                    selected = self.selection['selected'][dataset][row['family']]
                    index = list(SEEDS).index(row['seed'])
                    assert row['id'] == selected['run_ids'][index]
                    assert row['checkpoint']['path'] == selected['checkpoints'][index]
                    assert row['checkpoint']['sha256'] == selected['checkpoint_sha256'][index]
                self.receipt(row['prediction'])
                with np.load(row['prediction']['path'], allow_pickle=False) as archive:
                    assert np.array_equal(archive['origins'], origins)
                    arrays = {key: archive[key] for key in row['prediction_keys']}
                if row['family'] in ('a', 'level'):
                    q_values[(row['family'], row['seed'])] = arrays['q_prediction']
                roles = [(row['family'], row['id'], 'prediction')]
                if row['family'] == 'a':
                    close(arrays['prediction'], arrays['main_only_prediction'] + arrays['q_prediction'],
                          'A main + Q reconstruction', atol=1e-5, rtol=1e-4)
                    roles += [('a_main', row['id'] + '__main_only', 'main_only_prediction'),
                              ('a_last', row['id'] + '__main_last', 'main_last_prediction')]
                for family, identifier, key in roles:
                    ids.add(identifier)
                    error = np.where(mask, arrays[key].astype(np.float64) - target.astype(np.float64), 0.)
                    for term, value in (('squared', np.square(error).sum(axis=1)),
                                        ('absolute', np.abs(error).sum(axis=1)),
                                        ('signed', error.sum(axis=1)), ('count', mask.sum(axis=1))):
                        close(saved_terms.pop(identifier + '__' + term), value, identifier + ' saved origin terms')
                    recalculated = {}
                    for period, index in indices.items():
                        score = saved_scores(arrays[key][index], target[index], mask[index])
                        recalculated[period] = score
                        stated = reported['models'][identifier]['periods'][period]
                        for metric, value in score.items():
                            close(stated[metric], value, f'{identifier}/{period}/{metric}')
                        scores_by_key[(dataset, identifier, period)] = score
                        for c, column in enumerate(data['columns']):
                            channel_by_key[(dataset, identifier, period, str(column))] = (
                                score['channel_mse'][c], score['channel_mae'][c],
                                score['channel_bias'][c], score['target_counts'][c])
                        coverage = float(mask[index].all(axis=2).mean())
                        close(reported['models'][identifier]['space_common_U0'][period]['coverage'], coverage, 'common P/Q coverage')
                    for name in ('squared_error_sums', 'absolute_error_sums', 'signed_error_sums', 'target_counts'):
                        close(recalculated['combined'][name], recalculated['test_a'][name] + recalculated['test_b'][name],
                              'Period sums/counts pooled ' + name)
                    grouped[family].append(recalculated)
                model_count += 1
            assert len(ids) == 15 and ids == set(reported['models']) and set(grouped) == set(ROLES)
            assert not saved_terms
            for family, members in grouped.items():
                assert len(members) == (1 if family == 'f0' else 2)
                for period in PERIODS:
                    stated = reported['role_summary'][family]['periods'][period]
                    for metric in ('mse', 'mae', 'signed_mean_error', 'channel_mse', 'channel_mae', 'channel_bias',
                                   'squared_error_sums', 'absolute_error_sums', 'signed_error_sums', 'target_counts'):
                        expected = np.mean([member[period][metric] for member in members], axis=0)
                        close(stated[metric], expected, f'{family}/{period} mean seed losses: {metric}')
            for seed in SEEDS:
                close(q_values[('a', seed)], q_values[('level', seed)], 'frozen parent Q', atol=1e-5, rtol=1e-4)
                assert reported['A_Q_parity'][str(seed)]['pass'] is True
            role_count += len(ids)
            self.job.heartbeat(dataset + ' stored prediction metrics independently re-aggregated')
        with (HERE / 'confirmation_evaluation01_models.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(scores_by_key)
        assert len({(r['dataset'], r['id'], r['period']) for r in rows}) == len(rows)
        for row in rows:
            expected = scores_by_key[(row['dataset'], row['id'], row['period'])]
            for metric in ('mse', 'mae', 'signed_mean_error'):
                close(float(row[metric]), expected[metric], 'model CSV ' + metric)
        with (HERE / 'confirmation_evaluation01_channels.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(channel_by_key)
        assert len({(r['dataset'], r['id'], r['period'], r['channel']) for r in rows}) == len(rows)
        for row in rows:
            expected = channel_by_key[(row['dataset'], row['id'], row['period'], row['channel'])]
            close([float(row[k]) for k in ('mse', 'mae', 'signed_mean_error', 'observed_count')], expected, 'channel CSV')
        return {'prediction_models': model_count, 'scored_roles': role_count,
                'model_period_scores_reaggregated': len(scores_by_key), 'channel_period_scores': len(channel_by_key),
                'period_sums_count_pooling_verified': True, 'seed_loss_mean_not_prediction_ensemble': True,
                'model_loads': 0, 'model_forward_calls': 0, 'optimizer_updates': 0}

    def costs(self):
        assert self.selection is not None, 'Selection verification must complete first'
        binding = read_json(HERE / 'confirmation_cost_binding.json')
        binding_sha = digest(HERE / 'confirmation_cost_binding.json')
        for name, sha in binding['files'].items():
            self.receipt({'path': str(HERE / name), 'sha256': sha})
        assert binding['passes'] == 10 and binding['warmup_calls'] == 10 and binding['blocks'] == 3
        assert binding['precision'] == 'float32' and binding['tf32'] is False and binding['cpu_threads'] == 4
        assert binding['test_used_for_cost_selection'] is False
        assert set(binding['inputs']) == set(self.p['units'])
        parity_references = {}
        for dataset, entry in binding['inputs'].items():
            data = load_data(dataset, include_test=False)
            origins = data['val_origins'][:24].astype(np.int64)
            inputs = np.stack([data['x'][o - 512:o] for o in origins]).astype(np.float32)
            assert np.array_equal(origins, entry['origins'])
            assert hashlib.sha256(origins.tobytes()).hexdigest() == entry['origin_sha256']
            assert hashlib.sha256(inputs.tobytes()).hexdigest() == entry['input_sha256']
            self.receipt(entry['trainval'])
            target = np.stack([data['x'][o:o + 48] for o in origins])
            mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
            identifiers = [identifier for group in self.selection['selected'][dataset].values()
                           for identifier in group['run_ids']] + [f'v12_confirmation_{dataset}_f0']
            for identifier in identifiers:
                reference_path = CACHE / 'confirmation_cost_parity' / (identifier + '.npz')
                metadata = read_json(reference_path.with_suffix('.json'))
                assert metadata['id'] == identifier and metadata['device'] == 'cuda'
                assert metadata['binding_sha256'] == binding_sha
                self.receipt({'path': str(reference_path), 'sha256': metadata['npz_sha256']})
                with np.load(reference_path, allow_pickle=False) as arrays:
                    assert np.array_equal(arrays['origins'], origins)
                    prediction = arrays['prediction']
                parity_references[identifier] = {
                    'sha256': hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
                    'scores': saved_scores(prediction, target, mask)}
        all_counts = {}
        device_summaries = {}
        for device, per_unit in (('cuda', 135), ('cpu', 12)):
            grid = read_json(HERE / f'confirmation_cost_{device}_grid.json')
            saved = read_json(HERE / f'confirmation_cost_{device}_rows.json')
            summary = read_json(HERE / f'confirmation_cost_{device}_summary.json')
            assert saved['status'] == summary['status'] == 'complete'
            assert saved['binding_sha256'] == grid['binding_sha256'] == binding_sha
            expected_count = per_unit * len(self.p['units'])
            assert saved['expected_rows'] == grid['expected_rows'] == len(saved['rows']) == len(grid['rows']) == expected_count
            assert len({r['key'] for r in saved['rows']}) == expected_count
            block_groups = defaultdict(list)
            for sequence, (planned, row) in enumerate(zip(grid['rows'], saved['rows'])):
                assert all(row[key] == value for key, value in planned.items())
                assert row['planned_sequence'] == sequence
                key_fields = ('dataset', 'id', 'family', 'seed', 'device', 'block', 'batch', 'chunk_rows', 'execution', 'sentinel')
                expected_key = hashlib.sha256(json.dumps({key: row[key] for key in key_fields}, sort_keys=True).encode()).hexdigest()[:24]
                assert row['key'] == expected_key
                assert row['status'] == 'complete' and row['binding_sha256'] == binding_sha
                self.receipt(artifact(HERE / 'confirmation_cost_rows' / device / (row['key'] + '.json')))
                assert read_json(HERE / 'confirmation_cost_rows' / device / (row['key'] + '.json')) == row
                assert row['passes'] == row['warmup_calls'] == 10 and row['device'] == device
                times = np.asarray(row['seconds_per_24_origins'], dtype=np.float64)
                assert times.shape == (10,) and np.isfinite(times).all() and np.all(times > 0)
                for key, values in (('milliseconds_per_origin', times * 1000 / 24), ('origins_per_second', 24 / times)):
                    expected = {'median': np.median(values), 'q25': np.quantile(values, .25),
                                'q75': np.quantile(values, .75), 'minimum': values.min(), 'maximum': values.max()}
                    for metric, value in expected.items():
                        close(row[key][metric], value, 'raw pass aggregation ' + key)
                info = row['model']
                family, dataset = row['family'], row['dataset']
                if family == 'f0':
                    assert row['id'] == f'v12_confirmation_{dataset}_f0' and row['seed'] is None
                    assert info['selected_model_checkpoint'] is None
                else:
                    chosen = self.selection['selected'][dataset][family]
                    index = list(SEEDS).index(row['seed'])
                    assert row['id'] == chosen['run_ids'][index]
                    assert info['selected_model_checkpoint']['path'] == chosen['checkpoints'][index]
                    assert info['selected_model_checkpoint']['sha256'] == chosen['checkpoint_sha256'][index]
                    self.receipt(info['fit_result'])
                    fit = self.fit_results[read_json(info['fit_result']['path'])['canonical_id']]
                    for key, value in info['training_result'].items():
                        assert value == fit[key]
                reference = parity_references[row['id']]
                if device == 'cuda':
                    assert len(info['reference_prediction_sha256']) == 64
                    epsilon = 0 if info['reference_prediction_sha256'] == reference['sha256'] else 1e-6
                    allowances = {'mae': epsilon, 'mse': 2 * epsilon * reference['scores']['mae'] + epsilon ** 2}
                    for metric in ('mse', 'mae'):
                        close(info['reference_val_scores'][metric], reference['scores'][metric],
                              'saved cost reference ' + metric, atol=allowances[metric] + 1e-10)
                assert info['merged_export_vs_original']['pass'] is True
                assert info['configuration_parity'] and all(r['parity']['pass'] is True for r in info['configuration_parity'])
                assert any(all(row[key] == config[key] for key in ('batch', 'chunk_rows', 'execution'))
                           for config in info['configuration_parity'])
                for check in [info['merged_export_vs_original']] + [r['parity'] for r in info['configuration_parity']]:
                    assert check['atol'] == 1e-5 and check['rtol'] == 1e-4 and check['finite'] is True
                if device == 'cuda':
                    assert row['peak_allocated_bytes'] > 0 and row['peak_reserved_bytes'] >= row['peak_allocated_bytes']
                    assert row['peak_allocated_bytes'] >= row['final_allocated_bytes']
                else:
                    assert row['family'] == 'direct' and info['cpu_gpu_parity']['pass'] is True
                    assert info['cpu_gpu_parity']['atol'] == 1e-5 and info['cpu_gpu_parity']['rtol'] == 1e-4
                    assert info['cpu_gpu_parity']['finite'] is True
                    assert all(row[key] is None for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'final_allocated_bytes'))
                if row['sentinel'] is None:
                    key = (row['dataset'], row['id'], row['family'], row['seed'], row['batch'], row['execution'], row['chunk_rows'])
                    block_groups[key].append(row)
            assert all(len(rows) == 3 and {r['block'] for r in rows} == {0, 1, 2} for rows in block_groups.values())
            for item in summary['instances']:
                key = tuple(item[k] for k in ('dataset', 'id', 'family', 'seed', 'batch', 'execution', 'chunk_rows'))
                rows = block_groups.pop(key)
                close(item['median_block_median_ms'], np.median([r['milliseconds_per_origin']['median'] for r in rows]), 'block median ms')
                close(item['median_block_median_origins_s'], np.median([r['origins_per_second']['median'] for r in rows]), 'block median throughput')
                close(item['block_ms'], [r['milliseconds_per_origin']['median'] for r in sorted(rows, key=lambda r: r['block'])], 'all block values')
                assert item['peak_allocated_bytes'] == [r['peak_allocated_bytes'] for r in rows]
                assert item['peak_reserved_bytes'] == [r['peak_reserved_bytes'] for r in rows]
            assert not block_groups
            seed_groups = defaultdict(list)
            for item in summary['instances']:
                seed_groups[tuple(item[key] for key in ('dataset', 'family', 'batch', 'execution', 'chunk_rows'))].append(item)
            for group in summary['groups']:
                members = seed_groups.pop(tuple(group[key] for key in ('dataset', 'family', 'batch', 'execution', 'chunk_rows')))
                assert len(members) == (1 if group['family'] == 'f0' else 2)
                assert group['ids'] == [item['id'] for item in members]
                close(group['mean_seed_median_ms'], np.mean([item['median_block_median_ms'] for item in members]), 'mean seed cost')
                close(group['mean_seed_median_origins_s'], np.mean([item['median_block_median_origins_s'] for item in members]), 'mean seed throughput')
                for kind in ('allocated', 'reserved'):
                    values = [value for item in members for value in item[f'peak_{kind}_bytes'] if value is not None]
                    if values:
                        close(group[f'mean_seed_median_peak_{kind}_bytes'],
                              np.mean([np.median(item[f'peak_{kind}_bytes']) for item in members]), 'mean seed peak bytes')
                        assert group[f'{kind}_range_bytes'] == [min(values), max(values)]
                    else:
                        assert group[f'mean_seed_median_peak_{kind}_bytes'] is None and group[f'{kind}_range_bytes'] is None
            assert not seed_groups
            assert len(summary['sentinels']) == (6 * len(self.p['units']) if device == 'cuda' else 0)
            for sentinel in summary['sentinels']:
                pair = {row['sentinel']: row for row in saved['rows'] if row['sentinel'] is not None
                        and all(row[key] == sentinel[key] for key in ('dataset', 'block', 'batch'))}
                pre, post = (pair[name]['milliseconds_per_origin']['median'] for name in ('pre', 'post'))
                close([sentinel['pre_ms'], sentinel['post_ms'], sentinel['post_relative_change']],
                      [pre, post, post / pre - 1], 'F0 sentinel drift')
            for dataset, unit in self.p['units'].items():
                rows = [r for r in saved['rows'] if r['dataset'] == dataset]
                for block in range(3):
                    local = [r for r in rows if r['block'] == block]
                    assert len(local) == (45 if device == 'cuda' else 4)
                    if device == 'cuda':
                        assert {(r['sentinel'], r['batch']) for r in local if r['sentinel']} == {('pre', 1), ('pre', 4), ('post', 1), ('post', 4)}
                    for family in (FAMILIES if device == 'cuda' else ('direct',)):
                        candidates = [r for r in local if r['family'] == family and r['sentinel'] is None]
                        members = {r['id'] for r in candidates}
                        assert len(members) == (1 if family == 'f0' else 2)
                        c, k = unit['C'], unit['K']
                        expected = {(1, None, 'direct'), (4, None, 'direct')} if family == 'direct' else (
                            {(1, c, 'full'), (1, k, 'chunk_k'), (4, 4*c, 'full'), (4, 4*k, 'chunk_4k'), (4, k, 'chunk_k')}
                            if family in ('f0', 'full_mse', 'native') else
                            {(1, k, 'full'), (4, 4*k, 'full'), (4, k, 'chunk_k')})
                        for identifier in members:
                            assert {(r['batch'], r['chunk_rows'], r['execution']) for r in candidates if r['id'] == identifier} == expected
            all_counts[device] = len(saved['rows'])
            device_summaries[device] = summary
        combined = read_json(HERE / 'confirmation_cost_summary.json')
        assert combined['status'] == 'complete' and combined['timing_is_not_historical_ratio'] is True
        assert combined['gpu'] == device_summaries['cuda'] and combined['cpu'] == device_summaries['cpu']
        self.receipt(combined['binding'])
        return {'complete_rows': all_counts, 'all_raw_passes_and_three_blocks_verified': True,
                'merge_chunk_cpu_parity': 'Saved fixed-tolerance checks verified; models not rerun'}

    def budget_and_processes(self):
        budget = budget_snapshot()
        jobs = read_json(HERE / 'ledger.json')['jobs']
        attempts = [j for j in jobs if j['category'] in ('neural_fit', 'coefficient_fit')]
        assert all(j['status'] in ('complete', 'failed') for j in attempts)
        assert sum(j['status'] == 'complete' for j in attempts) == len(self.p['neural_fits'])
        assert budget['real_fit_attempts'] <= len(self.p['neural_fits']) + 4 <= budget['limits']['real_fit']
        assert budget['technical_reserve_attempts'] <= 4 and budget['coefficient_fit_attempts'] == 0
        assert budget['synthetic_sessions'] <= budget['limits']['synthetic_sessions']
        assert all(row['updates'] <= budget['limits']['synthetic_steps'] for row in jobs if row.get('synthetic'))
        for category in ('gpu', 'cpu_analysis', 'cpu_check'):
            assert budget[category + '_seconds'] < budget['limits'][category + '_seconds']
        assert budget['storage_bytes'] < budget['limits']['storage_bytes']
        others = [row for row in budget['active_jobs'] if row['id'] != self.job.id]
        assert not others, 'Other v12 jobs remain running or have unclosed ledger outcomes'
        current = next(row for row in budget['active_jobs'] if row['id'] == self.job.id)
        assert current['pid'] == os.getpid() and current['pid_alive'] is True
        return {'during_check': budget, 'other_active_research_jobs': [],
                'current_check_pid_verified': True, 'completed_job_PIDs_not_treated_as_live_due_to_PID_reuse': True}

    def publication_scope(self):
        relative = HERE.relative_to(ROOT).as_posix()
        output = subprocess.check_output(['git', 'ls-files', '--cached', '-z', '--', relative], cwd=ROOT)
        paths = [p for p in output.decode('utf-8').split('\0') if p]
        prohibited = {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.parquet', '.h5', '.hdf5'}
        wrong = [path for path in paths if Path(path).suffix.lower() in prohibited or
                 any(part in ('.cache', 'raw', 'checkpoints', 'predictions') for part in Path(path).parts)]
        assert not wrong, 'Raw data/weights/array paths are in the current publication index: ' + repr(wrong)
        return {'indexed_v12_files_checked': len(paths), 'raw_checkpoint_prediction_arrays_in_current_index': False,
                'scope': 'Current Git index only; the final later publication staging must be checked separately'}

    def optional_report_receipts(self):
        path = HERE / 'report_confirmation_values.json'
        figure_path = HERE / 'confirmation_figure_receipt.json'
        if not path.exists() and not figure_path.exists():
            return {'present_artifacts_checked': [], 'absent_report_not_invented': True, 'optional_check': True}
        values, figures = read_json(path), read_json(figure_path)
        assert values['status'] == figures['status'] == 'complete'
        self.receipts(values)
        self.receipts(figures)
        evaluation = read_json(HERE / 'confirmation_evaluation01.json')
        cost = read_json(HERE / 'confirmation_cost_summary.json')
        assert values['sources']['evaluation']['sha256'] == digest(HERE / 'confirmation_evaluation01.json')
        assert values['sources']['cost']['sha256'] == digest(HERE / 'confirmation_cost_summary.json')
        assert set(values['units']) == set(figures['figures']) == set(self.p['units'])
        for row in values['accuracy_rows']:
            unit = evaluation['datasets'][row['dataset']]
            scores = unit['models'][row['id']]['periods'] if row['id'] else unit['role_summary'][row['family']]['periods']
            score = scores[row['period']]
            for metric in ('mse', 'mae', 'signed_mean_error', 'target_counts'):
                close(row[metric], score[metric], 'report ' + metric)
            assert row['origin_count'] == score['origins']
            assert row['fit0_same_checkpoint_reference'] == (row['family'] in ('a_main', 'a_last'))
            for family in ('level', 'f0', 'full_mse', 'native', 'direct'):
                reference = unit['role_summary'][family]['periods'][row['period']]
                if row['id'] and row['family'] != 'f0' and family != 'f0':
                    reference = next(member for member in unit['models'].values()
                                     if member['family'] == family and member['seed'] == row['seed'])['periods'][row['period']]
                for metric in ('mse', 'mae'):
                    delta = row[metric] - reference[metric]
                    close(row[f'{metric}_minus_{family}'], delta, 'report difference')
                    percent = row[f'{metric}_relative_percent_vs_{family}']
                    if reference[metric] > 0:
                        close(percent, 100 * delta / reference[metric], 'report relative percent')
                    else:
                        assert percent is None
        assert len(values['accuracy_rows']) == 69 * len(self.p['units'])
        for row in values['cost_configurations']:
            group = next(item for item in cost['gpu' if row['device'] == 'cuda' else 'cpu']['groups']
                         if all(row[key] == item[key] for key in ('dataset', 'family', 'batch', 'execution', 'chunk_rows')))
            assert all(row[key] == value for key, value in group.items())
            score = evaluation['datasets'][row['dataset']]['role_summary'][row['family']]['periods']['combined']
            close([row['combined_mse'], row['combined_mae']], [score['mse'], score['mae']], 'report cost accuracy')
        assert len(values['cost_configurations']) == sum(len(cost[device]['groups']) for device in ('gpu', 'cpu'))
        assert values['cost_instances'] == {device: cost[device]['instances'] for device in ('gpu', 'cpu')}
        assert values['cost_sentinels'] == cost['gpu']['sentinels']
        assert values['cost_raw_counts'] == {'gpu': 135 * len(self.p['units']), 'cpu': 12 * len(self.p['units'])}
        for unit in figures['figures'].values():
            assert set(unit) == {'accuracy', 'batch4_cost'}
            for figure in unit.values():
                assert set(figure['outputs']) == {'png', 'svg'} and figure['caption'] and figure['plotted_values']
        return {'present_artifacts_checked': [path.name, figure_path.name], 'report_values_against_scored_sources': True,
                'figure_outputs_and_source_hashes_verified': True, 'figure_rerendered': False, 'optional_check': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, default=180)
    args = parser.parse_args()
    destination = HERE / 'finalconfirmationchecks.json'
    with Job('cpu_check', args.job, reserve_s=args.reserve_s,
             metadata={'purpose': 'saved confirmation final verification', 'model_forward_calls': 0, 'fit_count': 0}) as job:
        verifier = Verification(job)
        for name, function in (
                ('sources_and_contract', verifier.sources_and_contract),
                ('fits_and_selection', verifier.fits_and_selection),
                ('seal_and_exposure', verifier.seal_and_exposure),
                ('saved_prediction_metrics', verifier.saved_prediction_metrics),
                ('costs', verifier.costs),
                ('budget_and_processes', verifier.budget_and_processes),
                ('publication_scope_at_check', verifier.publication_scope),
                ('optional_report_receipts', verifier.optional_report_receipts)):
            verifier.run(name, function)
        complete = all(group['status'] == 'passed' for group in verifier.groups)
        result = {'status': 'passed' if complete else 'incomplete_or_failed',
                  'overall_confirmation_complete': complete, 'checks': verifier.groups,
                  'verified_unique_receipts': len(verifier.checked), 'job_label': args.job, 'job_id': job.id,
                  'scope': 'Own saved-checkpoint/array/provenance/cost checks; no independent scientific reproduction',
                  'self_check_not_independent_reproduction': True,
                  'new_model_runs': 0, 'optimizer_updates': 0,
                  'sources': {name: artifact(HERE / name) for name in
                              ('confirmation_protocol.json', 'confirmation_binding.json', 'verify_confirmation_v12.py')},
                  'budget_snapshot_during_check': budget_snapshot()}
        if destination.exists():
            prior = read_json(destination)
            previous = HERE / f'finalconfirmationchecks_prior_job_{prior["job_id"]}.json'
            if previous.exists() and read_json(previous) != prior:
                raise RuntimeError('Prior final-check preservation path has different content')
            if not previous.exists():
                save_json(previous, prior)
        save_json(destination, result)
    terminal = read_json(HERE / 'ledger.json')['jobs'][job.id]
    result['check_ledger_terminal'] = {key: terminal[key] for key in ('id', 'label', 'status', 'pid', 'updates', 'elapsed_s', 'ended_utc')}
    result['final_budget_after_check'] = budget_snapshot()
    save_json(destination, result)
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
