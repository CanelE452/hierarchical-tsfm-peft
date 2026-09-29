"""Bounded checks of saved v9 artifacts; no model construction, forward or update."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

from runtime_v9 import (HERE, ROOT, LIMITS, Job, budget_snapshot, digest,
                        load_data, phase_sample, save_json, source_receipt)

DATASETS = ('robin', 'jena')
FAMILIES = ('staged_p', 'staged_raw')
ROLES = ('staged_p', 'staged_raw', 'staged_p_alpha', 'staged_raw_alpha')
HEAD_SHAPES = {'head.down.weight': (32, 512), 'head.up.weight': (48, 32)}
CORE = ('model_v9.py', 'train_v9.py', 'runtime_v9.py', 'protocol.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path):
    path = resolve(path)
    return {'path': str(path), 'sha256': digest(path)}


def checked(item):
    path = resolve(item['path'])
    assert path.is_file(), 'Missing artifact: ' + str(path)
    assert digest(path) == item['sha256'], 'Changed artifact: ' + str(path)
    return path


def file_entry(path, sha256):
    return checked({'path': path, 'sha256': sha256})


def close(actual, expected, label):
    if actual is None or expected is None:
        assert actual is expected, label
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=label)


def tensor_hashes(state):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


def tensor_state_equal(left, right):
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), 'Tensor differs: ' + name


def head_state(state):
    assert set(state) == set(HEAD_SHAPES), 'Checkpoint must save only the new head'
    for name, shape in HEAD_SHAPES.items():
        assert tuple(state[name].shape) == shape and state[name].dtype == torch.float32
        assert torch.isfinite(state[name]).all()
    assert sum(value.numel() for value in state.values()) == 17920


def load_saved(path):
    # Deserialize recorded tensors on CPU only; never import the model loader.
    return torch.load(path, map_location='cpu', weights_only=False)


def saved_prediction(item, origins):
    with np.load(checked(item), allow_pickle=False) as archive:
        assert np.array_equal(archive['origins'], origins), 'Prediction origins/order changed'
        values = archive['prediction'].copy()
    assert values.dtype == np.float32 and np.isfinite(values).all()
    return values


def score(prediction, target, mask):
    assert prediction.shape == target.shape == mask.shape
    assert np.isfinite(prediction).all() and np.isfinite(target[mask]).all()
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    counts = mask.sum((0, 1))
    assert (counts > 0).all(), 'Every declared channel must have observed targets'
    squared, absolute, signed = (error * error).sum((0, 1)), np.abs(error).sum((0, 1)), error.sum((0, 1))
    return {'mse': float(np.mean(squared / counts)), 'mae': float(np.mean(absolute / counts)),
            'channel_mse': (squared / counts).tolist(), 'channel_mae': (absolute / counts).tolist(),
            'channel_mean_signed_error': (signed / counts).tolist(), 'channel_target_count': counts.tolist(),
            'channel_squared_error_sum': squared.tolist(), 'channel_absolute_error_sum': absolute.tolist(),
            'channel_signed_error_sum': signed.tolist()}


def space_score(prediction, target, mask, basis):
    complete = mask.all(axis=2)
    errors = prediction[complete].astype(np.float64) - target[complete].astype(np.float64)
    count, channels = len(errors), target.shape[2]
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    out = {'complete_target_rows': count, 'channel_complete_row_count': [count] * channels}
    for name, matrix, label in [('p', projector, 'p_retained_space'),
                                ('q', np.eye(channels) - projector, 'q_residual_space')]:
        values = errors @ matrix
        out[label + '_mse'] = float(np.mean(values * values)) if count else None
        out[label + '_mae'] = float(np.mean(np.abs(values))) if count else None
        out['channel_' + name + '_mse'] = np.mean(values * values, axis=0).tolist() if count else [None] * channels
        out['channel_' + name + '_mae'] = np.mean(np.abs(values), axis=0).tolist() if count else [None] * channels
        out['channel_' + name + '_mean_signed_error'] = np.mean(values, axis=0).tolist() if count else [None] * channels
    return out


def source_check(sources, filenames=CORE):
    for filename in filenames:
        path = HERE / filename
        key = path.relative_to(ROOT).as_posix()
        assert sources[key] == digest(path), 'Core source changed: ' + filename


def preservation(protocol):
    provenance, preserved = read(HERE / 'provenance.json'), read(HERE / 'preservation.json')
    for name, filename in [('plan', 'PLAN.md'), ('approval', 'APPROVAL.txt'), ('protocol', 'protocol.json')]:
        assert digest(HERE / filename) == provenance[name + '_sha256']
    assert preserved['base_commit'] == provenance['base_commit'] == protocol['base_commit']
    assert len(preserved['tracked_files']) == 1050
    for path, sha256 in preserved['tracked_files'].items():
        file_entry(path, sha256)
    reuse = read(HERE / 'reuse_manifest.json')
    assert reuse['no_new_download'] and reuse['no_backbone_raw_or_array_copy']
    for dataset in DATASETS:
        old = reuse['datasets'][dataset]
        file_entry(old['contract'], old['contract_sha256'])
        file_entry(old['evaluation'], old['evaluation_sha256'])
        assert digest(HERE / f'data_contract_{dataset}.json') == old['contract_sha256']
        for key in ('trainval', 'test'):
            checked(read(HERE / f'data_contract_{dataset}.json')[key])
        for parent in reuse['parents'][dataset].values():
            for key in ('checkpoint', 'model_source', 'result', 'best_val_prediction',
                        'parent_initial', 'head_initial', 'test_prediction'):
                checked(parent[key])
    correction = read(HERE / 'checker_correction01.json')
    assert correction['status'] == 'corrected_and_rechecked_before_any_real_fit'
    failed = read(HERE / correction['failed_result'])
    success = read(HERE / correction['successful_recheck'])
    assert failed['status'] == 'FAIL' and success['status'] == 'PASS'
    checker_key = (HERE / correction['corrected_source']).relative_to(ROOT).as_posix()
    assert failed['source'][checker_key] == digest(HERE / correction['failed_source'])
    assert success['source'][checker_key] == digest(HERE / correction['corrected_source'])
    source_check(failed['source'])
    source_check(success['source'])
    assert success['optimizer_updates'] == 10 and success['real_data'] is False
    assert success['paired_head_initial_tensors_exact'] and success['restart_equal_after_one_update_each']
    for check in success['checks'].values():
        assert check['learned_parent_exact_at_initialization'] and check['all_parent_parameters_and_buffers_unchanged']
        assert check['new_head_actually_updated'] and check['checkpoint_roundtrip']
        assert check['nonunit_alpha_checkpoint_roundtrip'] and check['alpha_zero_returns_parent']
    forbidden = [str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
                 p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.safetensors', '.bin', '.zip')]
    assert not forbidden, 'Private arrays/weights under public folder: ' + repr(forbidden)
    return {'tracked_files_unchanged': 1050, 'base_commit': protocol['base_commit'],
            'data_and_parent_receipts': receipt(HERE / 'reuse_manifest.json'),
            'checker_correction': receipt(HERE / 'checker_correction01.json'),
            'synthetic_success': receipt(HERE / correction['successful_recheck']),
            'no_public_weight_or_prediction_arrays': True}


def checkpoint_checks(result, data, manifest, curve):
    spec, seed = result['spec'], str(result['spec']['seed'])
    rec = manifest['records']['head'][seed]
    assert result['initial_source'] == rec
    initial = load_saved(checked(rec))
    parent = read(HERE / 'reuse_manifest.json')['parents'][spec['dataset']][seed]
    assert initial['parent_receipt'] == result['parent_receipt'] == parent
    assert initial['dataset'] == spec['dataset'] and initial['seed'] == spec['seed']
    assert initial['epoch'] == initial['steps'] == 0 and initial['kind'] == 'head'
    assert initial['data_sha256'] == result['data_sha256']
    head_state(initial['state'])
    assert tensor_hashes(initial['state']) == rec['tensor_hashes'] == initial['tensor_hashes'] == result['initial_parameter_sha256']
    assert torch.count_nonzero(initial['state']['head.up.weight']) == 0
    source_initial = load_saved(checked(rec['source_initial']))
    assert rec['source_initial'] == parent['head_initial']
    assert tensor_hashes(source_initial['state']) == rec['source_initial']['tensor_hashes']
    for name in HEAD_SHAPES:
        assert torch.equal(initial['state'][name], source_initial['state'][name.replace('head.', 'retained.')])
    assert torch.equal(initial['basis'], torch.as_tensor(data['basis']))
    best_trained, alpha = result['best_trained'], result['alpha_selected']
    records = {
        'initial': (result['initial_checkpoint'], result['initial_checkpoint_sha256'], 0, 1.),
        'basic': (result['checkpoint'], result['checkpoint_sha256'], result['selected_epoch'], 1.),
        'trained': (best_trained['checkpoint'], best_trained['checkpoint_sha256'], best_trained['selected_epoch'], 1.),
        'alpha': (alpha['checkpoint'], alpha['checkpoint_sha256'], best_trained['selected_epoch'], alpha['alpha']),
        'restart': (result['restart_checkpoint'], result['restart_sha256'], result['epochs_completed'], 1.),
    }
    states = {}
    for role, (path, sha, epoch, strength) in records.items():
        saved = load_saved(file_entry(path, sha))
        assert saved['spec'] == spec and saved['parent_receipt'] == parent
        assert saved['epoch'] == epoch and saved['steps'] == curve[epoch]['steps']
        assert saved['alpha'] == strength and saved['model_config']['alpha'] == strength
        assert saved['initial_path'] == rec['path'] and saved['initial_sha256'] == rec['sha256']
        assert torch.equal(saved['basis'], initial['basis'])
        head_state(saved['state'])
        cfg = saved['model_config']
        for key in ('dataset', 'family', 'channels', 'latent', 'ed_mode'):
            assert cfg[key] == spec[key]
        assert cfg['context'] == 512 and cfg['horizon'] == 48 and cfg['precision'] == 'float32'
        assert cfg['trainable_parameters'] == 17920 and cfg['cumulative_fitted_adapter_parameters'] == 35840
        assert cfg['head'] == 'shared 512-32-48,bias=False,activation=None'
        assert cfg['input'] == ('PCA reconstruction minus its last value' if spec['family'] == 'staged_p' else 'input minus its last value')
        assert cfg['parent_id'] == parent['id'] and cfg['pretrained_revision'] == read(HERE / 'protocol.json')['backbone_revision']
        assert cfg['parent_checkpoint_sha256'] == parent['checkpoint']['sha256']
        assert cfg['parent_model_source_sha256'] == parent['model_source']['sha256']
        states[role] = saved
    tensor_state_equal(states['initial']['state'], initial['state'])
    tensor_state_equal(states['trained']['state'], states['alpha']['state'])
    if result['selected_epoch'] == 0:
        tensor_state_equal(states['basic']['state'], initial['state'])
    elif result['selected_epoch'] == best_trained['selected_epoch']:
        tensor_state_equal(states['basic']['state'], states['trained']['state'])
    restart = states['restart']
    assert restart['history'] == curve and restart['schedules'] == result['schedules']
    assert restart['next_batch'] == 0 and restart['boundary'] == 'post-VAL, post-scheduler, complete epoch'
    assert restart['stale'] == result['final_stale'] and restart['best_epoch'] == result['selected_epoch']
    assert restart['trained_epoch'] == best_trained['selected_epoch']
    close(restart['best'], result['best_val_mse'], 'restart basic minimum')
    close(restart['trained_best'], best_trained['val_mse'], 'restart trained minimum')
    groups, opt_states = restart['optimizer']['param_groups'], restart['optimizer']['state']
    assert len(groups) == 1 and groups[0]['name'] == 'new_head' and groups[0]['weight_decay'] == 0
    assert len(groups[0]['params']) == len(opt_states) == 2
    assert set(groups[0]['params']) == set(opt_states)
    for state in opt_states.values():
        assert int(state['step']) == result['total_steps']
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    assert restart['scheduler']['last_epoch'] == result['epochs_completed']
    assert restart['scheduler']['_last_lr'] == [groups[0]['lr']]
    assert set(restart['rng']) == {'torch', 'cuda', 'numpy', 'python'}
    assert all(result['restart_roundtrip'][key] for key in result['restart_roundtrip'] if key != 'additional_updates')
    assert result['restart_roundtrip']['additional_updates'] == 0
    for name in HEAD_SHAPES:
        actual = float((states['restart']['state'][name] - initial['state'][name]).abs().max())
        close(actual, result['parameter_max_updates'][name], name + '/last actual update')
        assert actual > 0 and result['changed_parameter_scalars'][name] == int(torch.count_nonzero(states['restart']['state'][name] != initial['state'][name]))
        selected = float((states['basic']['state'][name] - initial['state'][name]).abs().max())
        close(selected, result['selected_parameter_max_updates'][name], name + '/selected actual update')
    for key in ('replay', 'best_trained_replay', 'alpha_selected_replay'):
        check = result[key]
        assert check['pass'] and check['violating_elements'] == 0
        assert check['atol'] == 1e-6 and check['rtol'] == 0 and check['max_absolute_difference'] <= 1e-6
    close(result['replay_error'], result['replay']['max_absolute_difference'], 'basic replay error')
    frozen = result['frozen_parameters_verified']
    assert all(frozen[key] is True for key in ('no_gradient', 'version_unchanged', 'optimizer_exclusion', 'parent_eval_mode', 'parameters_and_buffers'))
    assert len(frozen['frozen_tensor_sha256']) == 64
    assert result['trainable_parameters'] == 17920 and result['cumulative_fitted_adapter_parameters'] == 35840
    assert result['total_parameters'] == 47718016 + 2 * spec['channels'] * spec['latent'] + 35840
    grads = result['first_update_gradients']
    assert set(grads) == {'1', '2'} and all(set(grads[k]) == set(HEAD_SHAPES) for k in grads)
    for step in grads.values():
        assert all(g['requires_grad'] and g['max_abs'] is not None and np.isfinite(g['max_abs']) for g in step.values())
    assert grads['1']['head.down.weight']['max_abs'] == 0
    assert grads['1']['head.up.weight']['max_abs'] > 0 and grads['2']['head.down.weight']['max_abs'] > 0
    assert result['initial_parent_preservation']['full_parent_difference_max_abs'] == 0
    if spec['family'] == 'staged_p':
        checks = [result[k] for k in ('initial_parent_preservation', 'selected_preservation', 'alpha_selected_preservation')]
        checks += [r['preservation'] for r in result['alpha_candidates']]
        assert all(r['correction_q_violating_elements'] == r['prediction_q_violating_elements'] == 0 for r in checks)
    parent_check = result['parent_validation']
    assert parent_check['result'] == parent['result'] and parent_check['prediction'] == parent['best_val_prediction']
    assert parent_check['parity']['pass'] and parent_check['parity']['violating_elements'] == 0
    assert parent_check['parity']['atol'] == 1e-5 and parent_check['parity']['rtol'] == 1e-4
    close(parent_check['initial_val_mse'], result['initial_val_mse'], 'parent initial VAL')
    close(parent_check['historical_best_val_mse'], parent['best_val_mse'], 'parent historical VAL')
    close(parent_check['mse_difference'], result['initial_val_mse'] - parent['best_val_mse'], 'parent VAL difference')
    return {'head_state_files': 5, 'head_parameters': 17920, 'initial_tensors_exact': True,
            'parent_dependency_and_head_tensor_hashes': True, 'basic_best_trained_and_alpha_distinguished': True,
            'restart_state_and_existing_replay_receipts': True,
            'freeze_scope': 'Existing gradient/version/full-parent-hash receipts, pinned dependencies and saved head states; no fresh parent construction'}


def training(dataset, protocol, ledger, job):
    data = load_data(dataset, include_test=False)
    manifest, selected = read(HERE / f'initial_manifest_{dataset}.json'), read(HERE / f'selected_{dataset}.json')
    assert selected['dataset'] == dataset and selected['test_used_for_selection'] is False
    assert set(selected['selected']) == set(ROLES)
    assert selected['real_fits'] == 4 and selected['auxiliary_new_fits'] == 0 and selected['auxiliary_val_candidates'] == 20
    specs = [s for s in protocol['fits'] if s['dataset'] == dataset]
    receipts = {r['id']: r for r in selected['all_neural_runs']}
    assert set(receipts) == {s['id'] for s in specs}
    targets = np.stack([data['x'][o:o + 48] for o in data['val_origins']])
    masks = np.stack([data['finite'][o:o + 48] for o in data['val_origins']]).astype(bool)
    results, output = {}, {}
    for row in ledger['jobs']:
        if row['category'] != 'real_fit' or row['metadata']['dataset'] != dataset:
            continue
        identifier = row['label']
        attempt = read(HERE / 'runs' / identifier / 'attempt.json')
        assert attempt['ledger_id'] == row['id'] and attempt['status'] == row['status']
        assert attempt['actual_updates'] == row['updates']
        source_check(attempt['source'])
        source_check(row['source'])
        if row['status'] == 'failed':
            assert attempt.get('error')
            continue
        assert row['status'] == 'complete'
        result = read(HERE / 'runs' / identifier / 'result.json')
        assert result['status'] == 'complete' and result['ledger_id'] == row['id']
        assert result['attempt_updates'] == result['total_steps'] == row['updates']
        assert result['data_sha256'] == digest(data['_path']) == manifest['data_sha256']
        source_check(result['source'])
        curve = read(HERE / 'runs' / identifier / 'curve.json')
        assert [r['epoch'] for r in curve] == list(range(result['epochs_completed'] + 1))
        assert all(np.isfinite(r['val_mse']) and np.isfinite(r['val_mae']) and r['alpha'] == 1 for r in curve)
        assert all(r['steps'] == r['epoch'] * 128 for r in curve)
        assert all(set(r['learning_rates']) == {'new_head'} and 0 < r['learning_rates']['new_head'] <= .001 for r in curve)
        best, trained = min(curve, key=lambda r: r['val_mse']), min(curve[1:], key=lambda r: r['val_mse'])
        assert best['epoch'] == result['selected_epoch'] and trained['epoch'] == result['best_trained']['selected_epoch']
        close(best['val_mse'], result['best_val_mse'], identifier + '/earliest basic minimum')
        close(trained['val_mse'], result['best_trained']['val_mse'], identifier + '/earliest trained minimum')
        stale, best_value = 0, curve[0]['val_mse']
        for point in curve[1:]:
            if point['val_mse'] < best_value:
                best_value, stale = point['val_mse'], 0
            else:
                stale += 1
            if point['epoch'] < result['epochs_completed']:
                assert stale < 6, 'Training continued past fixed patience'
        assert result['final_stale'] == stale and (stale == 6 or result['epochs_completed'] == 120)
        assert result['cap_reached_while_improving'] == (result['epochs_completed'] == 120 and stale < 6 and 120 - best['epoch'] < 6)
        assert result['extension_eligible'] is False and 1 <= result['epochs_completed'] <= 120
        assert len(result['schedules']) == result['epochs_completed']
        for item in result['schedules']:
            order = phase_sample(data['train_origins'], result['spec']['seed'], item['epoch'], 512, period=data['_phase_period'])
            assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == item['sha256']
        candidates = result['alpha_candidates']
        assert [r['alpha'] for r in candidates] == protocol['alpha_grid']
        assert all(r['checkpoint_epoch'] == trained['epoch'] for r in candidates)
        assert result['auxiliary_val_candidates'] == 5 and result['auxiliary_new_fits'] == 0
        winner = min(candidates, key=lambda r: (r['val_mse'], r['alpha']))
        auxiliary = result['alpha_selected']
        assert winner['alpha'] == auxiliary['alpha']
        close(winner['val_mse'], auxiliary['val_mse'], 'five-alpha selected minimum')
        close(candidates[-1]['val_mse'], result['best_trained']['val_mse'], 'alpha1 trained minimum')
        close(candidates[0]['val_mse'], result['initial_val_mse'], 'alpha0 learned-parent fallback')
        selection = read(file_entry(auxiliary['selection_path'], auxiliary['selection_sha256']))
        assert selection['candidates'] == candidates and selection['grid'] == protocol['alpha_grid']
        assert selection['test_used_for_selection'] is False and selection['best_trained_checkpoint'] == result['best_trained']
        for key in ('alpha', 'checkpoint', 'checkpoint_sha256', 'val_prediction', 'val_mse'):
            assert selection[key] == auxiliary[key]
        for rec, expected in ((result['best_val_prediction'], result['best_val_mse']),
                              (result['best_trained']['best_val_prediction'], trained['val_mse']),
                              (auxiliary['val_prediction'], auxiliary['val_mse'])):
            values = saved_prediction(rec, data['val_origins'])
            close(score(values, targets, masks)['mse'], expected, identifier + '/saved VAL MSE')
        parent_values = saved_prediction(result['parent_validation']['prediction'], data['val_origins'])
        close(score(parent_values, targets, masks)['mse'], result['parent_validation']['historical_best_val_mse'], 'parent saved VAL')
        for probe in ('initial_train_probe', 'selected_train_probe'):
            assert np.isfinite(result[probe]['mse']) and np.isfinite(result[probe]['mae'])
        with np.load(Path(result['checkpoint']).parent / 'origins.npz', allow_pickle=False) as saved:
            expected_probe = phase_sample(data['train_origins'], result['spec']['seed'], 0,
                                          max(96, data['_phase_period']), period=data['_phase_period'])
            assert np.array_equal(saved['train_probe'], expected_probe) and np.array_equal(saved['val'], data['val_origins'])
        output[identifier] = checkpoint_checks(result, data, manifest, curve)
        results[identifier] = result
        job.heartbeat('verified saved training artifacts: ' + identifier)
    for spec in specs:
        result, rec = results[spec['id']], receipts[spec['id']]
        assert result['spec'] == spec
        file_entry(rec['result_path'], rec['result_sha256'])
        for key in ('checkpoint', 'checkpoint_sha256', 'best_val_prediction', 'best_trained', 'alpha_selected', 'parent_receipt'):
            assert rec[key] == result[key]
    for seed in protocol['seeds']:
        pair = [results[s['id']] for s in specs if s['seed'] == seed]
        assert len(pair) == 2 and pair[0]['initial_source'] == pair[1]['initial_source']
        assert pair[0]['parent_receipt'] == pair[1]['parent_receipt']
        close(pair[0]['initial_val_mse'], pair[1]['initial_val_mse'], 'same-seed initial function')
        orders = {}
        for result in pair:
            for item in result['schedules']:
                assert orders.setdefault(item['epoch'], item['sha256']) == item['sha256']
    for family, choice in selected['selected'].items():
        base_family = family.removesuffix('_alpha')
        ids = [s['id'] for s in sorted(specs, key=lambda s: s['seed']) if s['family'] == base_family]
        assert choice['run_ids'] == ids and choice['lr'] == .001 and choice['ed_mode'] == 'fixed_ed'
        entries = [results[i]['alpha_selected'] if family.endswith('_alpha') else results[i] for i in ids]
        assert choice['checkpoints'] == [r['checkpoint'] for r in entries]
        assert choice['alphas'] == [r.get('alpha', 1.) for r in entries]
        close(choice['mean_best_val_mse'], np.mean([r['val_mse'] if family.endswith('_alpha') else r['best_val_mse'] for r in entries]), family + '/mean VAL')
    return {'dataset': dataset, 'completed_required_fits': 4, 'checks': output,
            'same_seed_parent_initial_head_and_origin_orders': True,
            'five_alpha_candidates_no_refit': True, 'selection': receipt(HERE / f'selected_{dataset}.json')}


def parity_stats(left, right, atol=1e-5, rtol=1e-4):
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    violations = delta > atol + rtol * np.abs(right.astype(np.float64))
    return {'atol': atol, 'rtol': rtol, 'max_abs': float(delta.max()),
            'max_rel': float((delta / np.maximum(np.abs(right), 1e-12)).max()),
            'violations': int(violations.sum()), 'elements': int(delta.size), 'pass': not bool(violations.any())}


def evaluation_checks(protocol, job):
    output = {}
    for dataset in DATASETS:
        seal_path, exposure_path = HERE / f'evaluation_seal_{dataset}.json', HERE / f'test_exposure_{dataset}.json'
        seal, exposure = read(seal_path), read(exposure_path)
        assert exposure['seal_sha256'] == digest(seal_path) and seal['test_used_for_selection'] is False
        assert seal['sealed_utc'] <= exposure['first_exposed_utc'] and seal['test_selection_prohibited'] is True
        assert set(seal['all_selection_files_required_before_test']) == set(DATASETS)
        for rec in seal['all_selection_files_required_before_test'].values():
            checked(rec)
        for key in ('selection', 'data_contract', 'reuse_manifest', 'protocol', 'plan', 'approval', 'prior_v8_reuse'):
            checked(seal[key])
        prior = read(checked(seal['prior_v8_reuse']))
        path = HERE / (exposure['evaluation_name'] + '.json')
        result = read(path)
        assert result['status'] == 'complete' and result['dataset'] == dataset and result['test_reselection'] is False
        assert result['roles'] == seal['roles'] and result['new_test_model_predictions'] == 8
        assert {r['id'] for r in seal['models']} == set(result['models'])
        assert len(result['models']) == len(prior['models']) + 8
        checked(result['seal'])
        checked(result['first_exposure'])
        source_check(result['source'], ('evaluate_v9.py',))
        data = load_data(dataset, include_test=True)
        assert result['channels'] == np.asarray(data['columns']).astype(str).tolist()
        periods = {p: np.asarray(data[p + '_origins']) for p in ('test_a', 'test_b')}
        origins = np.concatenate(list(periods.values()))
        indices = {'test_a': np.arange(len(periods['test_a'])),
                   'test_b': np.arange(len(periods['test_a']), len(origins)), 'combined': np.arange(len(origins))}
        targets = np.stack([data['x'][o:o + 48] for o in origins])
        masks = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        for period, period_origins in periods.items():
            assert result['origins'][period] == period_origins.tolist()
            assert len(period_origins) == seal['expected_origins'][period]
            assert np.all(np.diff(period_origins) == seal['evaluation_stride'])
            start, end = map(np.datetime64, seal['splits'][period])
            times = np.asarray(data['times']).astype('datetime64[ns]')
            assert np.all(times[period_origins] >= start) and np.all(times[period_origins + 47] < end)
        checked_scores, spaces, q_arrays = {}, {}, {}
        for identifier, row in result['models'].items():
            values = saved_prediction(row['prediction'], origins)
            checked_scores[identifier], spaces[identifier] = {}, {}
            if identifier in prior['models']:
                assert row['reused_prior_prediction'] is True and row['prediction'] == prior['models'][identifier]['prediction']
            else:
                assert row['reused_prior_prediction'] is False and row['role'] in ROLES
            for period, index in indices.items():
                metrics = score(values[index], targets[index], masks[index])
                for key, value in metrics.items():
                    close(value, row['scores'][period][key], identifier + '/' + period + '/' + key)
                assert row['scores'][period]['missing_channel_indices'] == []
                checked_scores[identifier][period] = metrics
                if 'space_scores' in row:
                    scores = space_score(values[index], targets[index], masks[index], data['basis'])
                    for key, value in scores.items():
                        close(value, row['space_scores'][period][key], identifier + '/' + period + '/' + key)
                    spaces[identifier][period] = scores
            if row['role'] in (*ROLES, 'level_res'):
                q_arrays[identifier] = values
            job.heartbeat('verified saved evaluation: ' + identifier)
        for role, ids in result['roles'].items():
            for period in indices:
                rec = result['role_scores'][role][period]
                assert rec['run_ids'] == ids and rec['count'] == len(ids)
                for metric in ('mse', 'mae'):
                    close(rec[metric], np.mean([checked_scores[i][period][metric] for i in ids]), role + '/mean seed/' + metric)
                if role in result['role_space_scores']:
                    rec = result['role_space_scores'][role][period]
                    for metric in ('p_retained_space_mse', 'q_residual_space_mse', 'p_retained_space_mae', 'q_residual_space_mae'):
                        close(rec[metric], np.mean([spaces[i][period][metric] for i in ids]), role + '/' + metric)
                    assert rec['complete_target_rows'] == int(masks[indices[period]].all(axis=2).sum())
        for role, periods_summary in result['q_preservation'].items():
            for period, row in periods_summary.items():
                current, parent = result['role_space_scores'][role][period], result['role_space_scores']['level_res'][period]
                for metric in ('mse', 'mae'):
                    key = 'q_residual_space_' + metric
                    close(row['role_q_' + metric], current[key], 'Q error role')
                    close(row['parent_q_' + metric], parent[key], 'Q error parent')
                    close(row['delta_q_' + metric], current[key] - parent[key], 'Q error delta')
        projector = data['basis'].astype(np.float64) @ data['basis'].astype(np.float64).T
        q_matrix = np.eye(projector.shape[0]) - projector
        parity = result['q_prediction_parity']
        assert parity['status'] == 'available' and set(parity['roles']) == set(ROLES)
        for role in ROLES:
            assert set(parity['roles'][role]) == set(result['roles'][role])
            for identifier, row in parity['roles'][role].items():
                current, parent = q_arrays[identifier].astype(np.float64), q_arrays[row['parent_id']].astype(np.float64)
                assert result['models'][identifier]['spec']['seed'] == row['seed'] == result['models'][row['parent_id']]['spec']['seed']
                assert row['parent_id'] in result['roles']['level_res']
                current_p = (current.reshape(-1, current.shape[-1]) @ projector).reshape(current.shape)
                current_q = (current.reshape(-1, current.shape[-1]) @ q_matrix).reshape(current.shape)
                parent_p = (parent.reshape(-1, parent.shape[-1]) @ projector).reshape(parent.shape)
                parent_q = (parent.reshape(-1, parent.shape[-1]) @ q_matrix).reshape(parent.shape)
                for key, actual in [('q_prediction_parity', parity_stats(current_q, parent_q)),
                                    ('p_component_delta_against_parent', parity_stats(current_p, parent_p))]:
                    for metric, value in actual.items():
                        close(value, row[key][metric], identifier + '/' + key + '/' + metric)
                close(float(np.abs(current_p + current_q - current).max()), row['projection_reconstruction_max_abs'], 'projection reconstruction')
                if role in ('staged_p', 'staged_p_alpha'):
                    assert row['q_prediction_parity']['pass'] and row['expected_q_preserved'] is True
                if row['alpha'] == 0:
                    for key, value in parity_stats(current, parent).items():
                        close(value, row['alpha0_full_parent_parity'][key], 'alpha0 full parent')
                    assert row['alpha0_full_parent_parity']['pass']
        assert parity['all_expected_q_preservation_pass'] is True
        assert {r['name'] for r in result['comparisons']} == {r['name'] for r in seal['comparisons']}
        for comparison in result['comparisons']:
            for period in indices:
                left = np.mean([checked_scores[i][period]['mse'] for i in comparison['left']])
                right = np.mean([checked_scores[i][period]['mse'] for i in comparison['right']])
                for key, value in {'left_mse': left, 'reference_mse': right, 'delta_mse': left - right,
                                   'relative_to_reference_pct': 100 * (left - right) / right}.items():
                    close(comparison['periods'][period]['full'][key], value, comparison['name'] + '/' + key)
                assert comparison['periods'][period]['block_length_origins'] == protocol['datasets'][dataset]['bootstrap_block_origins']
        output[dataset] = {'evaluation': receipt(path), 'new_predictions': 8,
                           'reused_predictions': len(prior['models']), 'raw_mask_origin_identity': True,
                           'all_saved_MSE_MAE_seed_period_channel_values': True,
                           'complete_row_P_Q_and_full_prediction_Q_parity': True,
                           'complete_target_rows': int(masks.all(axis=2).sum()),
                           'all_target_rows': int(masks.shape[0] * masks.shape[1]),
                           'bootstrap_scope': 'Point effects/block contract checked; 2000 resamples not rerun',
                           'evaluation_scope': 'Previously exposed development data, not independent confirmation'}
    return output


def distribution(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def cost_correction_checks():
    path = HERE / 'cost_checker_correction01.json'
    if not path.is_file():
        assert not list(HERE.glob('cost_*_failure.json')), 'Cost failure needs its preserved correction record'
        return {'present': False}
    correction = read(path)
    for key in ('original', 'updated', 'diagnostic', 'failure', 'preserved_partial'):
        checked(correction[key])
    assert checked(correction['updated']).resolve() == (HERE / 'cost_v9.py').resolve()
    failure, diagnostic = read(checked(correction['failure'])), read(checked(correction['diagnostic']))
    key = (HERE / 'cost_v9.py').relative_to(ROOT).as_posix()
    assert failure['status'] == 'failed' and failure['phase'] == 'parity'
    assert failure['source'][key] == diagnostic['cost_source_sha256'] == correction['original']['sha256']
    source_check(failure['source'])
    source_check(diagnostic['source'])
    assert set(correction['unchanged_core_hashes']) == set(CORE)
    for filename, item in correction['unchanged_core_hashes'].items():
        assert item['unchanged'] is True
        assert item['current_sha256'] == item['diagnostic_source_sha256'] == digest(HERE / filename)
        file_entry(item['path'], item['current_sha256'])
    expected = {'checkpoint_replay_atol': 1e-6, 'checkpoint_replay_rtol': 0,
                'mse_atol': 1e-8, 'batching_atol': 1e-5, 'batching_rtol': 1e-4}
    for name, value in expected.items():
        assert correction['unchanged'][name] == value
    assert all(correction['unchanged'][key] is True for key in
               ('data', 'model', 'training', 'protocol', 'checkpoint_selection', 'alpha_selection'))
    rows = {row['context']: row for row in diagnostic['rows']}
    assert set(rows) == {'no_grad', 'inference_mode'}
    assert rows['no_grad']['difference']['bitwise_equal'] is True
    assert rows['no_grad']['difference']['max_absolute_difference'] == rows['no_grad']['mse_difference'] == 0
    assert rows['inference_mode']['difference']['pass_'] is False
    assert rows['inference_mode']['difference']['violating_elements'] == 10
    return {'present': True, 'record': receipt(path),
            'original_failure_source_partial_and_diagnostic_preserved': True,
            'model_train_runtime_protocol_and_tolerances_unchanged': True,
            'diagnostic_scope': 'One checkpoint: combined construction and forward context; no attribution to a specific kernel',
            'successful_rerun_scope': 'Checked separately against the final current-source parity and timing bindings'}


def cost_checks():
    decision_path = HERE / 'cost_decision.json'
    decision = read(decision_path)
    assert set(decision['datasets']) == set(DATASETS)
    for dataset in DATASETS:
        evaluation = read(checked(decision['evaluations'][dataset]))
        assert evaluation['dataset'] == dataset and evaluation['status'] == 'complete'
    assert decision['run_cost'] == any(item['run_cost'] for item in decision['datasets'].values())
    output, total_rows = {}, 0
    config = lambda row: (row['id'], row['batch'], row['variant'], row['micro'])
    for dataset, choice in decision['datasets'].items():
        assert isinstance(choice['reason'], str) and choice['reason'].strip()
        cost_path, parity_path = HERE / f'cost_{dataset}_gpu_rows.json', HERE / f'cost_{dataset}_parity.json'
        if not choice['run_cost']:
            assert not cost_path.exists(), 'Omitted cost conflicts with completed measurements'
            output[dataset] = {'run_cost': False, 'reason': choice['reason'], 'no_unmeasured_speed_claim': True}
            continue
        parity, costs, summary = read(parity_path), read(cost_path), read(HERE / f'cost_{dataset}_summary.json')
        assert parity['status'] == 'PASS' and costs['status'] == summary['status'] == 'complete'
        assert costs['dataset'] == summary['dataset'] == parity['dataset'] == dataset
        assert costs['decision_sha256'] == digest(decision_path) and costs['parity_sha256'] == digest(parity_path)
        assert costs['binding'] == parity['binding']
        for path, sha in parity['binding'].items():
            file_entry(path, sha)
        aliases = costs['aliases']
        assert aliases == parity['aliases'] == summary['aliases'] and len(aliases) == 17
        expected_roles = {'base_level', 'staged_p', 'staged_p_alpha', 'staged_raw', 'staged_raw_alpha',
                          'joint_pq_v8', 'f0', 'lora', 'direct_nlinear'}
        assert {row['role'] for row in aliases.values()} == expected_roles
        expected = set()
        for alias in aliases.values():
            expected |= {(alias['measurement_id'], batch, 'original', None) for batch in (1, 4)}
            if alias['role'] in ('f0', 'lora'):
                expected.add((alias['measurement_id'], 4, 'generic', 20 if dataset == 'robin' else 24))
        assert {config(row) for row in parity['rows']} == expected
        assert parity['configuration_count'] == costs['configuration_count'] == len(expected) <= 37
        assert len(parity['rows']) == len(expected) and len(parity['restorations']) == 17
        for row in parity['rows']:
            assert row['val']['pass_'] and row['val']['violating_elements'] == 0
            assert row['val']['atol'] == 1e-5 and row['val']['rtol'] == 1e-4
            if row['family'] == 'lora':
                assert row['model']['merge']['pass_']
        for row in parity['restorations']:
            assert row['full_val']['pass_'] and row['full_val']['atol'] == 1e-6 and row['full_val']['rtol'] == 0
            assert abs(row['mse_difference']) <= 1e-8
        rows = costs['rows']
        assert len(rows) == 3 * len(expected) + 12 <= 123
        assert [r['sequence'] for r in rows] == list(range(len(rows)))
        total_rows += len(rows)
        for block in range(3):
            primary = [r for r in rows if r['block'] == block and r['sentinel'] is None]
            sentinels = [r for r in rows if r['block'] == block and r['sentinel'] is not None]
            assert len(primary) == len(expected) and {config(r) for r in primary} == expected
            assert len(sentinels) == 4 and {(r['batch'], r['sentinel']) for r in sentinels} == {(b, k) for b in (1, 4) for k in ('pre', 'post')}
        for row in rows:
            assert row['device'] == 'cuda'
            model = row['model']
            file_entry(model['checkpoint_path'], model['checkpoint_sha256'])
            file_entry(model['model_source'], model['model_source_sha256'])
            assert model['trainable_parameters_at_restore'] == {'f0': 0, 'lora': 294912, 'level_res': 17920,
                'pq_split': 35840, 'direct_nlinear': 24624, 'staged_p': 17920, 'staged_raw': 17920}[row['family']]
            assert model['parameter_bytes'] == 4 * model['deployed_parameters']
            assert model['deployment_tensor_bytes'] == model['parameter_bytes'] + model['buffer_bytes']
            times = np.asarray(row['seconds_per_24_origins'])
            assert len(times) == 20 and np.isfinite(times).all() and (times > 0).all()
            for metric, values in [('milliseconds_per_origin', times * 1000 / 24), ('origins_per_second', 24 / times)]:
                for key, value in distribution(values).items():
                    close(value, row[metric][key], 'cost raw repeats')
            assert 0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
        assert len(summary['fits']) == len(expected)
        lookup = {config(row): row for row in summary['fits']}
        assert set(lookup) == expected
        for row in summary['fits']:
            source = [r for r in rows if r['sentinel'] is None and config(r) == config(row)]
            for metric in ('milliseconds_per_origin', 'origins_per_second'):
                for key, value in distribution([r[metric]['median'] for r in source]).items():
                    close(value, row[metric][key], 'cost block center')
            assert row['peak_allocated_bytes'] == [r['peak_allocated_bytes'] for r in source]
            assert row['peak_reserved_bytes'] == [r['peak_reserved_bytes'] for r in source]
        assert len(summary['groups']) == 20
        for row in summary['groups']:
            members = [a for a in aliases.values() if a['role'] == row['role']]
            assert row['members'] == members and len(members) == (1 if row['role'] == 'f0' else 2)
            source = [lookup[(a['measurement_id'], row['batch'], row['variant'], row['micro'])] for a in members]
            close(row['mean_seed_median_ms'], np.mean([r['milliseconds_per_origin']['median'] for r in source]), 'cost seed latency')
            close(row['mean_seed_median_throughput'], np.mean([r['origins_per_second']['median'] for r in source]), 'cost seed throughput')
            close(row['mean_seed_median_peak_allocated_bytes'], np.mean([np.median(r['peak_allocated_bytes']) for r in source]), 'cost seed memory')
            for metric in ('allocated', 'reserved'):
                values = [v for r in source for v in r['peak_' + metric + '_bytes']]
                assert row['observed_peak_' + metric + '_range_bytes'] == [min(values), max(values)]
        assert len(summary['sentinels']) == 6
        for row in summary['sentinels']:
            pair = {r['sentinel']: r for r in rows if r['sentinel'] and r['block'] == row['block'] and r['batch'] == row['batch']}
            close(row['pre_ms'], pair['pre']['milliseconds_per_origin']['median'], 'pre sentinel')
            close(row['post_ms'], pair['post']['milliseconds_per_origin']['median'], 'post sentinel')
            close(row['post_relative_change'], row['post_ms'] / row['pre_ms'] - 1, 'sentinel drift')
        output[dataset] = {'run_cost': True, 'rows': len(rows), 'configurations': len(expected),
                           'role_aliases': 17, 'summary': receipt(HERE / f'cost_{dataset}_summary.json'),
                           'raw_repeats_blocks_seed_aggregates_and_drift': True}
    assert total_rows <= 246
    return {'decision': receipt(decision_path), 'datasets': output, 'total_rows': total_rows,
            'documented_correction': cost_correction_checks(),
            'scope': 'Same-dataset/session recorded measurements only; no inference about unmeasured costs'}


def analysis_checks(evaluations, path):
    summary = read(path)
    assert summary['status'] == 'complete' and set(summary['inputs']) == set(DATASETS)
    sources, counts = {}, {}
    for dataset, item in summary['inputs'].items():
        assert item == evaluations[dataset]['evaluation']
        sources[dataset] = read(checked(item))
    for name, entry in summary['outputs'].items():
        with checked(entry).open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == entry['columns']
            rows = list(reader)
        assert len(rows) == entry['rows']
        counts[name] = len(rows)
        for row in rows:
            evaluation = sources[row['dataset']]
            if name == 'role_scores_csv':
                expected = evaluation['role_scores'][row['role']][row['period']]
                metrics = ('mse', 'mae')
            elif name == 'model_scores_csv':
                expected = evaluation['models'][row['model_id']]['scores'][row['period']]
                metrics = ('mse', 'mae')
            elif name == 'space_scores_csv':
                expected = evaluation['role_space_scores'][row['role']][row['period']]
                metrics = ('p_retained_space_mse', 'q_residual_space_mse', 'p_retained_space_mae', 'q_residual_space_mae')
                assert int(row['complete_target_rows']) == expected['complete_target_rows']
            elif name == 'comparisons_csv':
                comparison = next(c for c in evaluation['comparisons'] if c['name'] == row['comparison'])
                expected = comparison['periods'][row['period']]['full']
                metrics = ('left_mse', 'reference_mse', 'delta_mse', 'relative_to_reference_pct')
            elif name == 'q_preservation_csv':
                expected = evaluation['q_preservation'][row['role']][row['period']]
                metrics = ('role_q_mse', 'parent_q_mse', 'delta_q_mse', 'role_q_mae', 'parent_q_mae', 'delta_q_mae')
            elif name == 'q_prediction_parity_csv':
                expected = evaluation['q_prediction_parity']['roles'][row['role']][row['model_id']]
                assert row['q_pass'] == str(expected['q_prediction_parity']['pass'])
                close(float(row['q_max_abs']), expected['q_prediction_parity']['max_abs'], 'Q prediction CSV')
                assert int(row['q_violations']) == expected['q_prediction_parity']['violations']
                continue
            else:
                raise AssertionError('Unexpected analysis output: ' + name)
            for metric in metrics:
                close(None if row[metric] == '' else float(row[metric]), expected[metric], name + '/' + metric)
    for dataset, values in summary['dataset_summaries'].items():
        evaluation = sources[dataset]
        assert values['new_test_model_predictions'] == 8
        for role, metrics in values['combined_role_values'].items():
            for metric in ('mse', 'mae'):
                close(metrics['combined_' + metric], evaluation['role_scores'][role]['combined'][metric], 'analysis combined')
        for key, value in values['combined_relative_mse_changes'].items():
            left, right = key.removesuffix('_relative_pct').split('_minus_')
            a, b = evaluation['role_scores'][left]['combined']['mse'], evaluation['role_scores'][right]['combined']['mse']
            close(value, 100 * (a - b) / b, 'analysis relative MSE')
    return {'summary': receipt(path), 'CSV_rows': counts, 'source_linked_scores_and_relative_changes': True}


def document_checks(claims_path, figures_path):
    required = [HERE / name for name in ('PLAN.md', 'STATUS.md', 'TOPIC_DECISION.md', 'METHOD_UPDATE.md')]
    required += [ROOT / 'README.md']
    docs = [receipt(path) for path in required]
    manifest = read(claims_path)
    for item in manifest['artifacts'] + manifest['sources']:
        checked(item)
    registered_artifacts = {checked(item).resolve() for item in manifest['artifacts']}
    registered_sources = {checked(item).resolve() for item in manifest['sources']}
    covered = set()
    assert manifest['claims'], 'No numeric manuscript/report claims registered'
    for claim in manifest['claims']:
        source_path = checked(claim['source'])
        assert source_path.resolve() in registered_sources
        value = read(source_path)
        for key in claim['json_path']:
            value = value[key]
        assert value == claim['value'], 'Numeric source mismatch: ' + claim['label']
        transformed = value * claim.get('multiplier', 1)
        rendered = format(transformed, claim['format_spec']) if 'format_spec' in claim else json.dumps(transformed, ensure_ascii=False)
        assert claim['rendered'] == rendered, 'Numeric rendering mismatch: ' + claim['label']
        artifact = resolve(claim['artifact']).resolve()
        assert artifact in registered_artifacts and rendered in artifact.read_text(encoding='utf-8')
        covered.add(artifact)
    assert {HERE.resolve() / 'TOPIC_DECISION.md', HERE.resolve() / 'METHOD_UPDATE.md', ROOT.resolve() / 'README.md'} <= covered
    figures = read(figures_path)
    assert figures['status'] == 'complete' and figures['new_inference'] is False and figures['new_fit'] is False
    for item in figures['sources'].values():
        checked(item)
    for item in figures['outputs'].values():
        checked(item)
    expected = {name + '.' + extension for name in ('accuracy_combined', 'accuracy_periods') for extension in ('svg', 'png')}
    assert expected <= set(figures['outputs'])
    return {'documents': docs, 'numeric_claims': receipt(claims_path), 'registered_numeric_claims': len(manifest['claims']),
            'figure_manifest': receipt(figures_path), 'figure_files': len(figures['outputs']),
            'scope': 'Registered source/rounded-number/file checks, not exhaustive prose validation or visual review'}


def report_checks():
    path = HERE / 'report_values.json'
    report = read(path)
    assert report['status'] == 'measured_and_derived_from_saved_artifacts'
    for item in report['sources'].values():
        checked(item)
    mapping = {'initial_val_mse': 'initial_val_mse', 'basic_epoch': 'selected_epoch',
               'basic_val_mse': 'best_val_mse', 'epochs_completed': 'epochs_completed',
               'steps': 'total_steps', 'final_stale': 'final_stale',
               'cap_reached_while_improving': 'cap_reached_while_improving',
               'new_trainable_parameters': 'trainable_parameters',
               'cumulative_fitted_adapter_parameters': 'cumulative_fitted_adapter_parameters',
               'total_parameters': 'total_parameters', 'new_head_online_fit_and_val_elapsed_s': 'head_fit_online_elapsed_s',
               'new_alpha_selection_elapsed_s': 'alpha_selection_elapsed_s', 'new_head_attempt_elapsed_s': 'elapsed_s'}
    for dataset, values in report['datasets'].items():
        evaluation = read(checked(report['sources'][dataset + '_evaluation']))
        assert set(values['roles']) == set(evaluation['role_scores'])
        for role, periods in values['roles'].items():
            for period, point in periods.items():
                source = evaluation['role_scores'][role][period]
                assert point['run_ids'] == source['run_ids']
                for metric in ('mse', 'mae'):
                    close(point[metric], source[metric], 'report accuracy')
        for candidate, references in values['combined_changes'].items():
            for reference, row in references.items():
                left = evaluation['role_scores'][candidate]['combined']['mse']
                right = evaluation['role_scores'][reference]['combined']['mse']
                for key, value in {'candidate_mse': left, 'reference_mse': right, 'absolute_mse_difference': left - right,
                                   'relative_mse_change_pct': (left / right - 1) * 100}.items():
                    close(row[key], value, 'report MSE change')
        assert values['comparisons'] == evaluation['comparisons']
        assert values['space_scores'] == evaluation['role_space_scores']
        assert values['q_preservation'] == evaluation['q_preservation']
        assert values['q_prediction_parity'] == evaluation['q_prediction_parity']
        assert len(values['training']) == 4
        for row in values['training']:
            source, parent = read(checked(row['result'])), read(checked(row['parent_result']))
            assert row['id'] == source['id'] and row['seed'] == source['spec']['seed']
            for report_key, result_key in mapping.items():
                close(row[report_key], source[result_key], 'report training/' + report_key)
            for report_key, source_value in {
                'parent_selected_epoch': parent['selected_epoch'], 'parent_historical_fit_elapsed_s': parent['elapsed_s'],
                'best_trained_epoch': source['best_trained']['selected_epoch'], 'best_trained_val_mse': source['best_trained']['val_mse'],
                'alpha': source['alpha_selected']['alpha'], 'alpha_val_mse': source['alpha_selected']['val_mse'],
                'initial_train_probe_mse': source['initial_train_probe']['mse'], 'selected_train_probe_mse': source['selected_train_probe']['mse'],
                'parameter_tensor_bytes_fp32': source['total_parameters'] * 4,
                'historical_parent_plus_new_attempt_s': parent['elapsed_s'] + source['elapsed_s']}.items():
                close(row[report_key], source_value, 'report derived training/' + report_key)
    decision = read(HERE / 'cost_decision.json')
    assert report['cost']['decision'] == decision
    assert set(report['cost']['results']) == {d for d, item in decision['datasets'].items() if item['run_cost']}
    for entry in report['cost']['results'].values():
        assert entry['summary'] == read(checked(entry['receipt']))
    training_path = checked(report['training_table'])
    columns = ['dataset', 'id', 'parent_id', 'seed', 'parent_selected_epoch', 'basic_epoch', 'best_trained_epoch',
               'alpha', 'epochs_completed', 'steps', 'initial_val_mse', 'basic_val_mse',
               'initial_train_probe_mse', 'selected_train_probe_mse', 'new_trainable_parameters',
               'cumulative_fitted_adapter_parameters', 'total_parameters', 'parent_historical_fit_elapsed_s',
               'new_head_online_fit_and_val_elapsed_s', 'new_alpha_selection_elapsed_s',
               'new_head_attempt_elapsed_s', 'historical_parent_plus_new_attempt_s']
    with training_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns
        rows = list(reader)
    expected = {row['id']: (dataset, row) for dataset, values in report['datasets'].items() for row in values['training']}
    assert len(rows) == len(expected) == 8 and len({row['id'] for row in rows}) == 8
    assert {row['id'] for row in rows} == set(expected)
    for row in rows:
        dataset, source = expected[row['id']]
        assert row['dataset'] == dataset
        for key in columns[1:]:
            assert row[key] == str(source[key]), 'Training CSV/report mismatch: ' + row['id'] + '/' + key
    return {'report': receipt(path), 'accuracy_selection_training_and_cost_copies': True,
            'training_table': report['training_table'], 'training_table_rows': 8,
            'training_table_all_fields_match_report': True,
            'relative_MSE_and_parameter_bytes_arithmetic': True,
            'budget_scope': 'Snapshot at report generation is preserved, not treated as the final closed-ledger usage'}


def accounting(protocol, own_id):
    ledger = read(HERE / 'ledger.json')
    assert protocol['limits'] == ledger['limits'] == LIMITS
    assert protocol['base_real_fits'] == 8 and protocol['reserve_real_fits'] == 4
    assert protocol['extension_allowed'] is False and protocol['automatic_extension'] is False
    rows = ledger['jobs']
    assert [r['id'] for r in rows] == list(range(len(rows))) and len({r['label'] for r in rows}) == len(rows)
    for row in rows:
        assert row['reserved_utc'] <= row['started_utc'] and row['reserved_seconds'] >= 0
        assert row['status'] in ('complete', 'failed') or row['id'] == own_id
        assert row['updates'] >= 0
        if row.get('synthetic'):
            assert row['category'] == 'cpu_check' and row['updates'] <= 10
        elif row['category'] not in ('real_fit',):
            assert row['updates'] == 0, 'Optimizer updates missing a fit or synthetic reservation'
    fits = [r for r in rows if r['category'] == 'real_fit']
    assert 8 <= len(fits) <= 12
    synthetic = [r for r in rows if r.get('synthetic')]
    assert len(synthetic) <= 6
    assert any(r['status'] == 'failed' and r['updates'] == 0 for r in synthetic)
    assert any(r['status'] == 'complete' and r['updates'] == 10 for r in synthetic)
    first_fit = min(r['started_utc'] for r in fits)
    assert all(r['ended_utc'] <= first_fit for r in synthetic)
    earliest = min(read(HERE / f'test_exposure_{dataset}.json')['first_exposed_utc'] for dataset in DATASETS)
    assert all(r['ended_utc'] <= earliest for r in fits), 'Fit touched or continued after TEST'
    selection_jobs = []
    for dataset in DATASETS:
        candidates = [r for r in rows if r['category'] == 'cpu_analysis' and r['metadata'].get('dataset') == dataset and 'select' in r['label'].lower()]
        assert candidates, 'Missing dataset selection ledger entry'
        assert all(r['status'] == 'complete' and r['ended_utc'] <= earliest for r in candidates)
        selection_jobs += [r['id'] for r in candidates]
    assert not any(r['category'] == 'data_prepare' and r['started_utc'] >= earliest for r in rows)
    gpu_jobs = sorted([r for r in rows if r['category'] == 'gpu'], key=lambda r: r['started_utc'])
    for left, right in zip(gpu_jobs, gpu_jobs[1:]):
        assert left['ended_utc'] <= right['started_utc'], 'Overlapping GPU jobs'
    state = budget_snapshot()
    for key in ('gpu_seconds', 'cpu_check_seconds', 'storage_bytes'):
        assert state[key] <= LIMITS[key], key + ' exceeded'
    assert state['real_fit_attempts'] == len(fits) and state['synthetic_sessions'] == len(synthetic)
    assert [r['id'] for r in state['active_jobs']] == [own_id]
    return {'snapshot_during_verification': state, 'failed_fit_attempts': sum(r['status'] == 'failed' for r in fits),
            'selection_jobs_before_first_TEST': selection_jobs, 'all_fits_before_first_TEST': True,
            'only_verifier_active': True, 'GPU_accounting': 'GPU parent jobs only; nested fit elapsed not double added',
            'scope': 'Final verifier runtime closes in the ledger after this report; PASS is not scientific success'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', default='final_checks.json')
    parser.add_argument('--analysis', default='analysis01/analysis_summary.json')
    parser.add_argument('--claims', default='numeric_claims.json')
    parser.add_argument('--figures', default='figures01.json')
    args = parser.parse_args()
    if Path(args.output).name != args.output or not args.output.endswith('.json'):
        parser.error('Use a new local JSON output filename')
    if not args.name or any(not (c.isalnum() or c in '_-') for c in args.name):
        parser.error('Use a unique alphanumeric/underscore/hyphen ledger name')
    target = HERE / args.output
    if target.exists():
        raise FileExistsError('Preserve existing verification result: ' + str(target))
    with Job(args.name, category='cpu_check', reserve_seconds=180) as job:
        global np, torch
        import numpy as np
        import torch
        torch.set_num_threads(4)
        result = {'status': 'RUNNING', 'ledger_id': job.id, 'source': source_receipt(), 'checks': {},
                  'scope': 'Execution-team saved-artifact checks; no independent reproduction, model construction, forward, optimizer or new prediction',
                  'scientific_claims_validated_by_PASS': False}
        try:
            protocol, ledger = read(HERE / 'protocol.json'), read(HERE / 'ledger.json')
            result['checks']['preservation'] = preservation(protocol)
            for dataset in DATASETS:
                result['checks']['training_' + dataset] = training(dataset, protocol, ledger, job)
            result['checks']['evaluation'] = evaluation_checks(protocol, job)
            result['checks']['analysis'] = analysis_checks(result['checks']['evaluation'], HERE / args.analysis)
            result['checks']['cost'] = cost_checks()
            result['checks']['report_values'] = report_checks()
            result['checks']['documents'] = document_checks(HERE / args.claims, HERE / args.figures)
            result['checks']['accounting'] = accounting(protocol, job.id)
            job.check_limits()
            result['status'] = 'PASS_WITH_DOCUMENTED_CHECKER_CORRECTIONS'
            result['completed_utc'] = time.time()
            save_json(target, result)
            job.heartbeat('saved-artifact verification complete; no scientific-success assertion')
        except BaseException:
            result['status'] = 'FAIL'
            result['error'] = traceback.format_exc()
            result['failed_utc'] = time.time()
            save_json(target, result)
            raise


if __name__ == '__main__':
    main()
