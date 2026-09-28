"""Bounded CPU audit of saved v7 artifacts; no model construction or new inference."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from runtime_v7 import (HERE, ROOT, V6, Job, LIMITS, budget_snapshot, digest, load_data,
                        phase_sample, save_json, source_receipt)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required completed artifact missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path):
    path = resolve(path)
    return {'path': str(path), 'sha256': digest(path)}


def check_receipt(item):
    path = resolve(item['path'])
    assert path.is_file(), 'Missing file: ' + str(path)
    assert digest(path) == item['sha256'], 'Hash mismatch: ' + str(path)
    return path


def close(actual, expected, label):
    if expected is None:
        assert actual is None, label
    else:
        assert actual is not None, label
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=label)


def file_entry(path, sha):
    return check_receipt({'path': path, 'sha256': sha})


def preservation():
    provenance = read(HERE / 'provenance.json')
    protocol = read(HERE / 'protocol.json')
    for name in ('plan', 'approval', 'protocol'):
        filename = {'plan': 'PLAN.md', 'approval': 'APPROVAL.txt', 'protocol': 'protocol.json'}[name]
        assert digest(HERE / filename) == provenance[name + '_sha256'], filename + ' approval changed'
    folders = ['research/tsfm_peft_development_20260926', 'research/tsfm_peft_followup_v2_20260926',
               'research/tsfm_peft_backbone_value_v3_20260927', 'research/tsfm_peft_fresh_group_v4_20260927',
               'research/tsfm_peft_accuracy_recovery_v5_20260927', 'research/tsfm_peft_level_confirmation_v6_20260928']
    tree = subprocess.run(['git', 'ls-tree', '-r', '--name-only', provenance['base_commit'], '--', 'research'],
                          cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    found = sorted({str(Path(p).parent).replace('\\', '/') for p in tree if p.endswith('/PLAN.md') and
                    any(token in p for token in ('development_20260926', 'followup_v2_', 'backbone_value_v3_',
                                                'fresh_group_v4_', 'accuracy_recovery_v5_', 'level_confirmation_v6_'))})
    folders = sorted(set(folders + found))
    diff = subprocess.run(['git', 'diff', '--name-only', provenance['base_commit'], '--', *folders],
                          cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    assert not diff, 'Preserved v1-v6 tracked files changed: ' + repr(diff)
    old_count = sum(any(p.startswith(folder + '/') for folder in folders) for p in tree)
    assert old_count > 0, 'No baseline research files found'
    model_source = read(V6 / 'model_source.json')
    assert model_source['revision'] == protocol['backbone_revision']
    for entry in model_source['files'].values():
        path = check_receipt(entry)
        assert path.stat().st_size == entry['bytes']
    source_refs = []
    for dataset in ('robin', 'jena'):
        contract = read(HERE / f'data_contract_{dataset}.json')
        for entry in (contract['trainval'], contract['test']):
            source_refs.append(str(check_receipt(entry)))
        if dataset == 'jena':
            assert contract['source']['additional_download_bytes'] == 0
            for key in ('csv', 'zip'):
                path = file_entry(contract['source'][key], contract['source'][key + '_sha256'])
                assert path.stat().st_size == contract['source'][key + '_size_bytes']
        else:
            file_entry(contract['source']['reuse_from_v6'], contract['source']['v6_data_contract_sha256'])
            file_entry(contract['source']['v6_evaluation'], contract['source']['v6_evaluation_sha256'])
    forbidden = [str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
                 p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.safetensors', '.bin', '.zip')]
    assert not forbidden, 'Private arrays/weights under public research folder: ' + repr(forbidden)
    return {'base_commit': provenance['base_commit'], 'preserved_tracked_files': old_count,
            'checked_folders': folders, 'backbone': receipt(V6 / 'model_source.json'),
            'data_array_receipts': source_refs, 'declared_source_receipts_checked': True,
            'no_public_weight_or_prediction_array': True}


def score(prediction, target, mask):
    assert prediction.shape == target.shape == mask.shape
    assert np.isfinite(prediction).all() and np.isfinite(target[mask]).all()
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    count = mask.sum((0, 1))
    squared = (error * error).sum((0, 1))
    absolute = np.abs(error).sum((0, 1))
    signed = error.sum((0, 1))
    valid = count > 0
    channel_mse = [float(a / n) if n else None for a, n in zip(squared, count)]
    channel_mae = [float(a / n) if n else None for a, n in zip(absolute, count)]
    return {'mse': float(np.mean(channel_mse)) if valid.all() else None,
            'mae': float(np.mean(channel_mae)) if valid.all() else None,
            'channel_mse': channel_mse, 'channel_mae': channel_mae,
            'channel_mean_signed_error': [float(a / n) if n else None for a, n in zip(signed, count)],
            'channel_target_count': count.tolist(), 'channel_squared_error_sum': squared.tolist(),
            'channel_absolute_error_sum': absolute.tolist(), 'channel_signed_error_sum': signed.tolist()}


def prediction(entry, origins):
    with np.load(check_receipt(entry), allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], origins), 'Saved prediction origin order changed'
        value = saved['prediction'].copy()
    assert value.dtype == np.float32 and np.isfinite(value).all()
    return value


def core_source(result):
    checked = {}
    for filename in ('model_v7.py', 'runtime_v7.py', 'train_v7.py'):
        key = (HERE / filename).relative_to(ROOT).as_posix()
        expected = result['source'][key]
        blob = (HERE / filename).read_bytes()
        matches = hashlib.sha256(blob).hexdigest() == expected
        scope = 'exact_current_bytes'
        if not matches and filename == 'train_v7.py':
            line = b"                       'test_used_for_selection': False,"
            candidates = [blob.replace(line + ending, b'') for ending in (b'\n', b'\r\n')]
            matches = any(hashlib.sha256(candidate).hexdigest() == expected for candidate in candidates)
            scope = 'documented_selection_metadata_only_line_addition'
        assert matches, 'Executed core source cannot be matched: ' + key
        checked[filename] = scope
    return checked


def checkpoint_check(result, data, manifest):
    import torch
    torch.set_num_threads(4)
    spec = result['spec']
    for key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('initial_checkpoint', 'initial_checkpoint_sha256'),
                          ('restart_checkpoint', 'restart_sha256')):
        file_entry(result[key], result[hash_key])
    initial = torch.load(resolve(result['initial_checkpoint']), map_location='cpu', weights_only=False)
    selected = torch.load(resolve(result['checkpoint']), map_location='cpu', weights_only=False)
    restart = torch.load(resolve(result['restart_checkpoint']), map_location='cpu', weights_only=False)
    kind = 'direct' if spec['family'] == 'direct_nlinear' else 'edg'
    shared_receipt = manifest['records'][kind][str(spec['seed'])]
    shared = torch.load(check_receipt(shared_receipt), map_location='cpu', weights_only=False)
    assert initial['epoch'] == initial['steps'] == shared['epoch'] == shared['steps'] == 0
    assert selected['epoch'] == result['selected_epoch'] and selected['steps'] == selected['epoch'] * 128
    assert restart['epoch'] == result['epochs_completed'] and restart['steps'] == result['total_steps']
    assert restart['next_batch'] == 0 and restart['boundary'] == 'post-VAL, post-scheduler, complete epoch'
    assert restart['schedules'] == result['schedules']
    assert restart['history'] == read(HERE / 'runs' / result['id'] / 'curve.json')
    assert restart['optimizer']['state'] and 'rng' in restart and 'scheduler' in restart
    assert set(initial['state']) == set(selected['state']) == set(restart['state'])
    trainable = 0
    for saved in (initial, selected, restart):
        assert saved['spec'] == spec and saved['model_config'] == initial['model_config']
        assert saved['initial_sha256'] == shared_receipt['sha256']
        assert torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(data['basis']))
    for name, tensor in initial['state'].items():
        hashed = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        assert hashed == result['initial_parameter_sha256'][name]
        if name in shared['state']:
            assert torch.equal(tensor, shared['state'][name]) and hashed == shared_receipt['tensor_hashes'][name]
        actual = float((restart['state'][name] - tensor).abs().max())
        close(actual, result['parameter_max_updates'][name], result['id'] + ': actual update ' + name)
        close(float((selected['state'][name] - tensor).abs().max()),
              result['selected_parameter_max_updates'][name], result['id'] + ': selected update ' + name)
        assert int(torch.count_nonzero(restart['state'][name] != tensor)) == result['changed_parameter_scalars'][name]
        if name in ('encoder.weight', 'decoder.weight'):
            assert actual == 0
        else:
            trainable += tensor.numel()
        if name == 'residual.up.weight':
            assert torch.count_nonzero(tensor) == 0
        if spec['family'] == 'lora':
            assert 'lora_' in name
    expected = 24624 if kind == 'direct' else 294912 if spec['family'] == 'lora' else 17920
    assert result['trainable_parameters'] == trainable == expected
    if kind == 'direct':
        assert set(initial['state']) == {'temporal.weight', 'temporal.bias'}
        assert result['total_parameters'] == 24624 and selected['model_config']['pretrained_revision'] is None
    if spec.get('resume_from'):
        parent = torch.load(resolve(spec['resume_from']), map_location='cpu', weights_only=False)
        assert parent['epoch'] == 120 and parent['stale'] < 6 and 120 - parent['best_epoch'] < 6
        assert result['attempt_updates'] == result['total_steps'] - parent['steps']
    else:
        assert result['attempt_updates'] == result['total_steps']
    assert result['total_steps'] == result['epochs_completed'] * 128
    assert any(value > 0 for value in result['parameter_max_updates'].values())
    first = result['first_update_gradients']
    for update in first.values():
        for gradient in update.values():
            if gradient['requires_grad']:
                assert gradient['max_abs'] is not None and np.isfinite(gradient['max_abs'])
            else:
                assert gradient['max_abs'] is None
    if not spec.get('resume_from') and spec['family'] in ('level_res', 'level_raw', 'old_res'):
        assert first['1']['residual.down.weight']['max_abs'] == 0
        assert first['2']['residual.down.weight']['max_abs'] > 0
        assert first['1']['residual.up.weight']['max_abs'] > 0
    required_updates = ('temporal.weight', 'temporal.bias') if kind == 'direct' else (
        ('residual.down.weight', 'residual.up.weight') if spec['family'] != 'lora' else ())
    assert all(result['parameter_max_updates'][key] > 0 for key in required_updates)
    assert all(result['frozen_parameters_verified'][key] for key in ('no_gradient', 'version_unchanged', 'optimizer_exclusion'))
    assert all(value is True for key, value in result['restart_roundtrip'].items() if key != 'additional_updates')
    assert result['restart_roundtrip']['additional_updates'] == 0
    assert result['replay_error'] <= 1e-6
    with np.load(resolve(result['initial_checkpoint']).parent / 'origins.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['val'], data['val_origins'])
        assert np.array_equal(saved['train_probe'], phase_sample(data['train_origins'], spec['seed'], 0,
                              max(96, data['_phase_period']), period=data['_phase_period']))
    return {'actual_tensor_updates': True, 'checkpoint_and_restart_structure': True,
            'trainable_parameters': trainable, 'frozen_backbone_scope': 'training gradient/version/hash receipts; no new backbone instantiation'}


def training(dataset, protocol, ledger, job):
    data = load_data(dataset, include_test=False)
    manifest = read(HERE / f'initial_manifest_{dataset}.json')
    selection = read(HERE / f'selected_{dataset}.json')
    assert selection['test_used_for_selection'] is False and selection['dataset'] == dataset
    base_specs = [spec for spec in protocol['fits'] if spec['dataset'] == dataset]
    effective = read(HERE / f'effective_specs_{dataset}.json')
    decision = read(HERE / f'extension_decision_{dataset}.json')
    assert decision['test_open'] is False
    assert len(effective) == len(base_specs) == (4 if dataset == 'robin' else 12)
    key = lambda spec: (spec['family'], spec['seed'], spec['lr'])
    assert {key(spec) for spec in effective} == {key(spec) for spec in base_specs}
    selected_receipts = {row['id']: row for row in selection['all_neural_runs']}
    assert set(selected_receipts) == {spec['id'] for spec in effective}
    all_results = {}
    train_jobs = [row for row in ledger['jobs'] if row['category'] == 'real_fit' and row['metadata']['dataset'] == dataset]
    for row in train_jobs:
        identifier = row['label']
        attempt = read(HERE / 'runs' / identifier / 'attempt.json')
        assert attempt['ledger_id'] == row['id'] and attempt['status'] == row['status']
        assert attempt['actual_updates'] == row['updates']
        if row['status'] == 'failed':
            assert attempt.get('error'), 'A failed fit must retain its failure'
            continue
        assert row['status'] == 'complete'
        result = read(HERE / 'runs' / identifier / 'result.json')
        assert result['status'] == 'complete' and result['attempt_updates'] == row['updates']
        assert result['data_sha256'] == digest(data['_path'])
        core = core_source(result)
        curve = read(HERE / 'runs' / identifier / 'curve.json')
        assert curve and curve[0]['epoch'] == 0 and all(np.isfinite(point['val_mse']) for point in curve)
        best = min(curve, key=lambda point: point['val_mse'])
        assert best['epoch'] == result['selected_epoch']
        close(best['val_mse'], result['best_val_mse'], identifier + ' earliest minimum VAL')
        for point in curve:
            assert point['steps'] == point['epoch'] * 128
        for schedule in result['schedules']:
            origins = phase_sample(data['train_origins'], result['spec']['seed'], schedule['epoch'], 512,
                                   period=data['_phase_period'])
            assert hashlib.sha256(origins.astype('<i8').tobytes()).hexdigest() == schedule['sha256']
        val = prediction(result['best_val_prediction'], data['val_origins'])
        targets = np.stack([data['x'][o:o + 48] for o in data['val_origins']])
        masks = np.stack([data['finite'][o:o + 48] for o in data['val_origins']])
        close(score(val, targets, masks)['mse'], result['best_val_mse'], identifier + ' saved VAL prediction')
        tensors = checkpoint_check(result, data, manifest)
        all_results[identifier] = result
        job.heartbeat('verified saved training artifacts: ' + identifier)
    for spec in effective:
        assert spec['id'] in all_results, 'Required effective fit incomplete: ' + spec['id']
        assert all_results[spec['id']]['spec'] == spec
        rec = selected_receipts[spec['id']]
        file_entry(rec['result_path'], rec['result_sha256'])
        assert rec['checkpoint_sha256'] == all_results[spec['id']]['checkpoint_sha256']
    for seed in (92601, 92602):
        seeded = [all_results[spec['id']] for spec in effective if spec['seed'] == seed]
        direct = [r for r in seeded if r['spec']['family'] == 'direct_nlinear']
        assert len(direct) == 2 and direct[0]['initial_parameter_sha256'] == direct[1]['initial_parameter_sha256']
        adapters = [r for r in seeded if r['spec']['family'] in ('level_res', 'old_res', 'level_raw')]
        if adapters:
            assert all(r['initial_parameter_sha256'] == adapters[0]['initial_parameter_sha256'] for r in adapters)
        schedules = {}
        for result in seeded:
            for row in result['schedules']:
                assert schedules.setdefault(row['epoch'], row['sha256']) == row['sha256']
    direct_means = [(lr, float(np.mean([all_results[s['id']]['best_val_mse'] for s in effective
                                       if s['family'] == 'direct_nlinear' and s['lr'] == lr]))) for lr in (1e-3, 1e-4)]
    selected_lr, val_mean = min(direct_means, key=lambda pair: pair[1])
    assert selection['selected']['direct_nlinear']['lr'] == selected_lr
    close(selection['selected']['direct_nlinear']['mean_best_val_mse'], val_mean, dataset + ' chosen LR mean')
    for family, row in selection['selected'].items():
        expected = sorted([s['id'] for s in effective if s['family'] == family and
                           (family != 'direct_nlinear' or s['lr'] == selected_lr)],
                          key=lambda identifier: all_results[identifier]['spec']['seed'])
        assert row['run_ids'] == expected
        close(row['mean_best_val_mse'], np.mean([all_results[i]['best_val_mse'] for i in expected]), family + ' selected mean')
    return {'dataset': dataset, 'completed_fit_artifacts': len(all_results), 'effective_fits': len(effective),
            'selected_lr': selected_lr, 'actual_tensor_updates_and_saved_prediction_scores': True,
            'identical_seed_initialization_and_origin_orders': True,
            'core_source_scope': 'exact model/runtime; train metadata-only one-line variation allowed',
            'selection': receipt(HERE / f'selected_{dataset}.json')}


def evaluation(dataset, job):
    seal_path = HERE / f'evaluation_seal_{dataset}.json'
    seal = read(seal_path)
    exposure = read(HERE / f'test_exposure_{dataset}.json')
    assert exposure['seal_sha256'] == digest(seal_path)
    assert seal['test_used_for_selection'] is False
    assert seal['sealed_utc'] <= exposure['first_exposed_utc']
    for key in ('selection', 'data_contract', 'protocol', 'plan', 'approval'):
        check_receipt(seal[key])
    name = exposure['evaluation_name']
    result_path = HERE / f'{name}.json'
    result = read(result_path)
    assert result['status'] == 'complete' and result['dataset'] == dataset and result['test_reselection'] is False
    check_receipt(result['seal'])
    assert result['roles'] == seal['roles']
    assert {row['id'] for row in seal['models']} == set(result['models'])
    data = load_data(dataset, include_test=True)
    periods = {period: np.asarray(data[period + '_origins']) for period in ('test_a', 'test_b')}
    origins = np.concatenate(list(periods.values()))
    indices = {'test_a': np.arange(len(periods['test_a'])),
               'test_b': np.arange(len(periods['test_a']), len(origins)), 'combined': np.arange(len(origins))}
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
    for period in ('test_a', 'test_b'):
        assert result['origins'][period] == periods[period].tolist()
        assert len(periods[period]) == seal['expected_origins'][period]
        assert np.all(np.diff(periods[period]) == seal['evaluation_stride'])
        start, end = map(np.datetime64, seal['splits'][period])
        times = np.asarray(data['times']).astype('datetime64[ns]')
        assert np.all(times[periods[period]] >= start) and np.all(times[periods[period] + 47] < end)
    checked, arrays = {}, {}
    for identifier, row in result['models'].items():
        array = prediction(row['prediction'], origins)
        arrays[identifier] = array
        checked[identifier] = {}
        for period, index in indices.items():
            value = score(array[index], target[index], mask[index])
            for metric, actual in value.items():
                close(actual, row['scores'][period][metric], identifier + '/' + period + '/' + metric)
            checked[identifier][period] = value
        job.heartbeat('verified stored evaluation: ' + identifier)
    for role, ids in result['roles'].items():
        for period in indices:
            for metric in ('mse', 'mae'):
                close(result['role_scores'][role][period][metric],
                      np.mean([checked[i][period][metric] for i in ids]), role + '/' + period + ' mean seed ' + metric)
    for comparison in result['comparisons']:
        for period in indices:
            left = np.mean([checked[i][period]['mse'] for i in comparison['left']])
            right = np.mean([checked[i][period]['mse'] for i in comparison['right']])
            full = comparison['periods'][period]['full']
            for key, actual in {'left_mse': left, 'reference_mse': right, 'delta_mse': left - right,
                                'relative_to_reference_pct': 100 * (left - right) / right}.items():
                close(full[key], actual, comparison['name'] + '/' + period + '/' + key)
            assert comparison['periods'][period]['block_length_origins'] == (7 if dataset == 'robin' else 42)
    primary = [r for r in result['comparisons'] if r['primary']]
    assert len(primary) == 1 and primary[0]['name'] == dataset + '_level_res_vs_' + ('direct_nlinear' if dataset == 'robin' else 'old_res')
    expected = 16 if dataset == 'robin' else 13
    assert len(result['models']) == expected
    if dataset == 'robin':
        original = read(V6 / 'evaluation01.json')
        assert set(original['models']).issubset(result['models'])
        for identifier, row in original['models'].items():
            assert result['models'][identifier]['prediction'] == row['prediction']
            assert result['models'][identifier]['reused_v6_prediction'] is True
    return {'path': receipt(result_path), 'models': expected, 'saved_predictions_and_all_score_aggregates': True,
            'mean_seed_not_ensemble': True, 'bootstrap_scope': 'point values/period block contract checked; 2000 resamples not rerun'}, (data, result, arrays, target, mask, indices)


def distribution(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def cost_checks(job, robin_context):
    parity = read(HERE / 'microbatch_parity.json')
    assert parity['status'] == 'PASS'
    for path, expected in parity['binding'].items():
        file_entry(path, expected)
    selected = read(HERE / 'selected_robin.json')
    direct = selected['selected']['direct_nlinear']['run_ids']
    families = {'v6_robin_f0': 'f0', **{f'v6_robin_lora_{s}': 'lora' for s in (92601, 92602)},
                **{f'v6_robin_level_res_{s}': 'level_res' for s in (92601, 92602)},
                **{identifier: 'direct_nlinear' for identifier in direct}}
    expected = set()
    for identifier, family in families.items():
        for batch in (1, 4):
            grid = [('original', None)]
            if family == 'level_res':
                grid += [('generic', batch * 5)] + ([('generic', 5)] if batch == 4 else [])
            elif family != 'direct_nlinear':
                grid += [('generic', batch * 17)] + [('generic', m) for m in ([5] if batch == 1 else [20, 5])]
            expected |= {(identifier, batch, variant, micro) for variant, micro in grid}
    assert len(expected) == 35
    parity_keys = set()
    for row in parity['rows']:
        for key in ('val', 'test'):
            if key in row:
                assert row[key]['pass_'] is True and row[key]['violating_elements'] == 0
                assert row[key]['atol'] == 1e-5 and row[key]['rtol'] == 1e-4
        if row.get('kind') != 'cpu_gpu_parity':
            parity_keys.add((row['id'], row['batch'], row['variant'], row['micro']))
            assert row['model']['checkpoint_sha256'] == digest(row['model']['checkpoint_path'])
            if row['family'] == 'lora':
                assert row['model']['merge']['pass_'] is True
    assert parity_keys == expected and len(parity['rows']) == 37
    _, saved_evaluation, saved_arrays, _, _, _ = robin_context
    historical = []
    for row in parity['rows']:
        if row.get('kind') == 'cpu_gpu_parity' or row['variant'] != 'original' or row['batch'] != 4:
            continue
        identifier = row['id']
        raw_hash = hashlib.sha256(saved_arrays[identifier].tobytes()).hexdigest()
        same_bytes = raw_hash == row['prediction_sha256']
        old_scores = saved_evaluation['models'][identifier]['scores']['combined']
        differences = {metric: row['test_' + metric] - old_scores[metric] for metric in ('mse', 'mae')}
        assert all(np.isfinite(value) for value in differences.values())
        if same_bytes:
            for metric in ('mse', 'mae'):
                close(row['test_' + metric], old_scores[metric], 'cost original B4 saved-prediction score')
        historical.append({'id': identifier, 'saved_prediction_raw_sha256': raw_hash,
                           'cost_prediction_raw_sha256': row['prediction_sha256'],
                           'bitwise_same_saved_prediction': same_bytes, 'score_differences': differences,
                           'scope': ('Same saved prediction bytes and scores.' if same_bytes else
                                     'Byte identity not established; aggregate score differences reported. '
                                     'Fresh cost prediction array was not saved, so historical elementwise tolerance is not retested. '
                                     'Merged LoRA can differ from its historical unmerged prediction within the separately tested VAL merge tolerance.')})
    assert len(historical) == 7
    gpu = read(HERE / 'cost_gpu_rows.json')
    cpu = read(HERE / 'cost_cpu_rows.json')
    assert gpu['status'] == cpu['status'] == 'complete' and len(gpu['rows']) == 117 and len(cpu['rows']) == 12
    all_rows = gpu['rows'] + cpu['rows']
    groups = defaultdict(list)
    for device, rows in (('cuda', gpu['rows']), ('cpu', cpu['rows'])):
        for block in (0, 1, 2):
            subset = [r for r in rows if r['block'] == block and r['sentinel'] is None]
            wanted = expected if device == 'cuda' else {(i, b, 'original', None) for i in direct for b in (1, 4)}
            assert {(r['id'], r['batch'], r['variant'], r['micro']) for r in subset} == wanted
            assert len(subset) == len(wanted)
            sentinels = [r for r in rows if r['block'] == block and r['sentinel']]
            assert len(sentinels) == (4 if device == 'cuda' else 0)
            if sentinels:
                assert {(r['sentinel'], r['batch']) for r in sentinels} == {(s, b) for s in ('pre', 'post') for b in (1, 4)}
                assert all(r['id'] == 'v6_robin_f0' and r['variant'] == 'original' for r in sentinels)
        for sequence, row in enumerate(rows):
            assert row['sequence'] == sequence and row['device'] == device
            times = np.asarray(row['seconds_per_24_origins'])
            assert len(times) == 20 and np.isfinite(times).all() and (times > 0).all()
            for field, values in (('milliseconds_per_origin', times * 1000 / 24), ('origins_per_second', 24 / times)):
                for key, actual in distribution(values).items():
                    close(row[field][key], actual, 'cost raw timing ' + field + '/' + key)
            info = row['model']
            file_entry(info['checkpoint_path'], info['checkpoint_sha256'])
            assert info['deployment_tensor_bytes'] == info['parameter_bytes'] + info['buffer_bytes']
            assert info['parameter_bytes'] == 4 * info['deployed_parameters']
            expected_params = {'f0': 0, 'lora': 294912, 'level_res': 17920, 'direct_nlinear': 24624}[row['family']]
            assert info['trainable_parameters_before_merge'] == expected_params
            if row['family'] == 'direct_nlinear':
                assert info['deployed_parameters'] == 24624
            if row['family'] == 'lora':
                assert info['merge']['pass_'] is True and info['merge']['violating_elements'] == 0
            if device == 'cuda':
                assert row['peak_reserved_bytes'] >= row['peak_allocated_bytes'] >= row['loaded_resident']['allocated'] > 0
            if row['sentinel'] is None:
                groups[(row['id'], row['batch'], row['variant'], row['micro'], row['device'])].append(row)
        job.heartbeat('verified ' + device + ' cost rows')
    summary = read(HERE / 'cost_summary.json')
    fits = {(r['id'], r['batch'], r['variant'], r['micro'], r['device']): r for r in summary['fits']}
    assert set(fits) == set(groups)
    for key, rows in groups.items():
        assert len(rows) == 3 and {r['block'] for r in rows} == {0, 1, 2}
        for field in ('milliseconds_per_origin', 'origins_per_second'):
            values = [r[field]['median'] for r in rows]
            for stat, actual in distribution(values).items():
                close(fits[key][field][stat], actual, 'cost three-block summary')
        assert fits[key]['peak_allocated_bytes'] == [r.get('peak_allocated_bytes') for r in rows]
        assert fits[key]['peak_reserved_bytes'] == [r.get('peak_reserved_bytes') for r in rows]
    for group in summary['groups']:
        members = [r for r in fits.values() if all(r[k] == group[k] for k in ('family', 'batch', 'variant', 'micro', 'device'))]
        assert sorted(group['ids']) == sorted(r['id'] for r in members)
        close(group['mean_seed_median_ms'], np.mean([r['milliseconds_per_origin']['median'] for r in members]), 'cost seed-mean latency')
        close(group['mean_seed_median_throughput'], np.mean([r['origins_per_second']['median'] for r in members]), 'cost seed-mean throughput')
        peaks = [p for r in members for p in r['peak_allocated_bytes'] if p is not None]
        assert group['observed_peak_allocated_range_bytes'] == ([min(peaks), max(peaks)] if peaks else None)
        medians = [np.median(r['peak_allocated_bytes']) for r in members if r['peak_allocated_bytes'][0] is not None]
        close(group['mean_seed_median_peak_allocated_bytes'], np.mean(medians) if medians else None, 'cost median memory')
    assert len(summary['sentinels']) == 6
    for item in summary['sentinels']:
        pair = {r['sentinel']: r for r in all_rows if r['sentinel'] and r['block'] == item['block'] and r['batch'] == item['batch']}
        pre, post = (pair[k]['milliseconds_per_origin']['median'] for k in ('pre', 'post'))
        close(item['pre_ms'], pre, 'sentinel pre')
        close(item['post_ms'], post, 'sentinel post')
        close(item['post_relative_change'], post / pre - 1, 'sentinel drift')
    return {'gpu_rows': 117, 'cpu_rows': 12, 'primary_configurations_per_gpu_block': 35,
            'parity_paths': 35, 'direct_cpu_gpu_parity_paths': 2, 'fixed_tolerance_and_hash_binding': True,
            'original_batch4_vs_historical_predictions': historical,
            'raw_distribution_and_three_block_seed_aggregation': True,
            'scope': 'Recorded timing/memory/parity evidence; no new timing or prediction executed'}


def csv_rows(entry):
    path = check_receipt(entry)
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == entry['rows'], 'CSV row receipt: ' + str(path)
    return rows


def diagnostics_check(context, job):
    data, evaluation_result, arrays, target, mask, indices = context
    result_path = HERE / 'robin_diagnostics_corrected_v7.json'
    result = read(result_path)
    assert result['status'] == 'complete'
    assert 'No prediction ensemble' in result['aggregation_contract']
    assert result['original_invalid_scope'] == 'original_diagnostic_invalid_for_mean_seed_conclusions'
    check_receipt(result['evaluation'])
    marker = read(check_receipt(result['correction_marker']))
    assert marker['status'] == result['original_invalid_scope']
    for key in ('original_script', 'original_result', 'corrected_script', 'evaluation'):
        check_receipt(marker[key])
    original = read(check_receipt(marker['original_result']))
    for entry in original['files'].values():
        check_receipt(entry)
    files = {key: check_receipt(entry) for key, entry in result['files'].items()}
    assert len(files) == 9 and not files['complete_row_space_detail_cache'].is_relative_to(HERE)
    columns = np.asarray(data['columns']).astype(str).tolist()
    member_scores = {identifier: score(array, target, mask) for identifier, array in arrays.items()}
    metric_map = {'mse': 'channel_mse', 'mae': 'channel_mae', 'mean_signed_error': 'channel_mean_signed_error'}
    rows = csv_rows(result['files']['role_channel_metrics'])
    roles = result['roles_loaded']
    assert all(ids == evaluation_result['roles'][role] for role, ids in roles.items())
    expected_rows = sum((len(ids) + 1) * len(columns) for ids in roles.values())
    assert len(rows) == expected_rows
    role_values = {}
    for row in rows:
        c = int(row['channel_index'])
        assert columns[c] == row['channel']
        ids = [row['member_id']] if row['scope'] == 'member' else roles[row['role']]
        for metric, key in metric_map.items():
            value = float(np.mean([member_scores[identifier][key][c] for identifier in ids]))
            close(float(row[metric]), value, 'diagnostic member-first channel ' + metric)
        if row['scope'] == 'role_mean_loss':
            role_values[(row['role'], c)] = {metric: float(row[metric]) for metric in metric_map}
    pair_rows = csv_rows(result['files']['pair_channel_excess'])
    for row in pair_rows:
        c = int(row['channel_index'])
        if row['scope'] == 'paired_member':
            left = {metric: member_scores[row['left_member_id']][key][c] for metric, key in metric_map.items()}
            right = {metric: member_scores[row['right_member_id']][key][c] for metric, key in metric_map.items()}
        else:
            assert row['scope'] == 'pair_mean_seed_loss'
            left, right = (role_values[(row[key], c)] for key in ('left_role', 'right_role'))
        for metric in metric_map:
            close(float(row['delta_' + metric]), left[metric] - right[metric], 'diagnostic channel delta')
        close(float(row['left_mse']), left['mse'], 'diagnostic channel left')
        close(float(row['right_mse']), right['mse'], 'diagnostic channel right')
    complete = mask.all(axis=2)
    assert result['coverage']['complete_origin_horizon_rows'] == int(complete.sum())
    close(result['coverage']['coverage_fraction'], float(complete.mean()), 'diagnostic complete coverage')
    for period in ('test_a', 'test_b'):
        sub = complete[indices[period]]
        assert result['coverage']['by_period'][period]['complete_rows'] == int(sub.sum())
        close(result['coverage']['by_period'][period]['coverage_fraction'], float(sub.mean()), 'period coverage')
    basis = np.asarray(data['basis'], dtype=np.float64)
    projector = basis @ basis.T
    space = {}
    for period in ('test_a', 'test_b'):
        ix = indices[period]
        for identifier, array in arrays.items():
            error = (array[ix].astype(np.float64) - target[ix].astype(np.float64))[complete[ix]]
            retained = error @ projector
            remainder = error - retained
            space[(period, identifier)] = np.array([np.mean(retained * retained),
                                                     np.mean(remainder * remainder), np.mean(error * error)])
    space_rows = csv_rows(result['files']['complete_row_space_summary'])
    for row in space_rows:
        period = row['period']
        assert int(row['complete_origin_horizon_rows']) == int(complete[indices[period]].sum())
        ids = roles[row['role']] if row['member_id'] == 'mean_of_member_losses' else [row['member_id']]
        expected = np.mean([space[(period, identifier)] for identifier in ids], axis=0)
        for key, value in zip(('p_retained_space_mse', 'q_residual_space_mse', 'total_complete_row_mse'), expected):
            close(float(row[key]), value, 'complete-row per-member projection ' + key)
    origins = np.concatenate([data['test_a_origins'], data['test_b_origins']])
    context_rows = csv_rows(result['files']['residual_context_rows'])
    assert len(context_rows) == int(mask.any(axis=1).sum())
    features = {}
    x = np.asarray(data['x'], dtype=np.float64)
    for n, origin in enumerate(origins):
        history = x[origin - 512:origin]
        residual = history - history @ projector
        features[n] = {'last': residual[-1], 'recent24_mean': residual[-24:].mean(axis=0),
                       'recent24_std': residual[-24:].std(axis=0), 'recent24_change': residual[-1] - residual[-24]}
    for row in context_rows:
        n, c = int(row['origin_position']), int(row['channel_index'])
        assert int(row['origin']) == int(origins[n]) and row['channel'] == columns[c]
        for key, value in features[n].items():
            close(float(row[key]), value[c], 'past-only residual feature')
        valid = mask[n, :, c]
        errors = [(arrays[i][n, valid, c].astype(np.float64) - target[n, valid, c].astype(np.float64))
                  for i in roles['level_res']]
        close(float(row['level_future_mse']), np.mean([np.mean(error ** 2) for error in errors]), 'context outcome mean-seed MSE')
        close(float(row['level_future_mean_signed_error']), np.mean([np.mean(error) for error in errors]), 'context signed outcome')
    job.heartbeat('verified corrected member-first diagnostics and preserved invalid original')
    return {'corrected_result': receipt(result_path), 'original_invalid_result_preserved': marker['original_result'],
            'all_diagnostic_file_hashes_checked': len(files),
            'member_then_role_channel_losses_and_deltas': True, 'complete_row_coverage_and_space_summary': True,
            'past_residual_features_and_member_first_outcomes': True,
            'scope': 'Channels/deltas, projection summaries, context inputs/outcomes numerically checked; origin/concentration/correlation CSV hashes checked, full statistic recomputation not repeated.'}


def budget_check(own_job, protocol):
    ledger = read(HERE / 'ledger.json')
    assert ledger['limits'] == LIMITS == protocol['limits']
    jobs = ledger['jobs']
    assert len({r['id'] for r in jobs}) == len(jobs) and len({r['label'] for r in jobs}) == len(jobs)
    assert not [r for r in jobs if r['status'] == 'running' and r['id'] != own_job and r['category'] in ('gpu', 'real_fit')]
    fits = [r for r in jobs if r['category'] == 'real_fit']
    assert 16 <= len(fits) <= 20
    assert all(r['status'] in ('complete', 'failed') for r in fits)
    for r in fits:
        assert r['reserved_utc'] is not None and r['reserved_utc'] <= r['started_utc']
        dataset = r['metadata']['dataset']
        exposed = read(HERE / f'test_exposure_{dataset}.json')['first_exposed_utc']
        sealed = read(HERE / f'evaluation_seal_{dataset}.json')['sealed_utc']
        assert r['ended_utc'] <= sealed <= exposed, 'Fitting after that dataset selection seal/TEST exposure'
    for r in jobs:
        if r['category'] == 'data_prepare':
            dataset = r.get('metadata', {}).get('dataset')
            affected = [dataset] if dataset in ('robin', 'jena') else ['robin', 'jena']
            if dataset is None:
                assert r['label'] == 'v7_prepare_data_contracts', 'Unknown unscoped data preparation job'
            for name in affected:
                assert r['ended_utc'] <= read(HERE / f'test_exposure_{name}.json')['first_exposed_utc']
    synthetic = [r for r in jobs if r.get('synthetic')]
    assert len(synthetic) <= 6 and all(0 <= r['updates'] <= 10 for r in synthetic)
    check = read(HERE / 'synthetic_models01.json')
    assert check['status'] == 'PASS' and check['optimizer_updates'] == 10
    deviation = [r for r in jobs if r.get('reserved_utc') is None]
    assert len(deviation) == 1 and deviation[0]['label'] == 'static_launcher_00_failed_before_reservation'
    assert deviation[0]['category'] == 'cpu_check' and deviation[0]['updates'] == 0
    close(deviation[0]['elapsed_s'], .247662, 'charged unreserved static launcher')
    assert 'reservation-order deviation' in deviation[0]['accounting_note']
    snapshot = budget_snapshot()
    for key in ('gpu_seconds', 'cpu_check_seconds', 'storage_bytes'):
        assert snapshot[key] <= LIMITS[key], key + ' cap exceeded'
    return {'snapshot': snapshot, 'base_fit_slots': 16, 'all_started_fit_attempts': len(fits),
            'additional_fit_attempts_including_failures_extensions': len(fits) - 16,
            'reservation_compliance': 'not perfect; explicitly preserved and charged static-launcher deviation',
            'reservation_deviations': deviation}


def supplemental_diagnostics_check(context, job):
    data, evaluation_result, arrays, target, mask, indices = context
    path = HERE / 'robin_q_residual_error_diagnostic_v7.json'
    result = read(path)
    assert result['status'] == 'complete'
    assert 'No prediction ensemble and no target zero-fill' in result['aggregation_contract']
    for key in ('evaluation', 'corrected_diagnostics', 'context_feature_rows'):
        check_receipt(result[key])
    source = (HERE / 'diagnose_residual_error_v7.py').relative_to(ROOT).as_posix()
    file_entry(ROOT / source, result['source'][source])
    corrected = read(HERE / 'robin_diagnostics_corrected_v7.json')
    features = csv_rows(corrected['files']['residual_context_rows'])
    assert check_receipt(result['context_feature_rows']) == check_receipt(corrected['files']['residual_context_rows'])
    features = {(int(row['origin_position']), int(row['channel_index'])): row for row in features}
    members = evaluation_result['roles']['level_res']
    assert len(members) == 2 and set(result['prediction_inputs']['level_prediction_receipts']) == set(members)
    for identifier, entry in result['prediction_inputs']['level_prediction_receipts'].items():
        check_receipt(entry)
        assert entry == evaluation_result['models'][identifier]['prediction']
    complete = mask.all(axis=2)
    origins = np.concatenate([data['test_a_origins'], data['test_b_origins']])
    columns = np.asarray(data['columns']).astype(str).tolist()
    basis = np.asarray(data['basis'], dtype=np.float64)
    q = np.eye(len(columns), dtype=np.float64) - basis @ basis.T
    outcomes = {}
    for n in range(len(origins)):
        admitted = complete[n]
        if not admitted.any():
            continue
        errors = [(arrays[identifier][n, admitted].astype(np.float64) -
                   target[n, admitted].astype(np.float64)) @ q for identifier in members]
        outcomes[n] = (np.mean([error.mean(axis=0) for error in errors], axis=0),
                       np.mean([(error ** 2).mean(axis=0) for error in errors], axis=0))
    rows = csv_rows(result['files']['q_residual_error_rows'])
    keys = set()
    for row in rows:
        n, c = int(row['origin_position']), int(row['channel_index'])
        assert (n, c) not in keys
        keys.add((n, c))
        assert row['scope'] == 'mean_seed_q_outcome' and int(row['member_count']) == 2
        assert row['member_ids'].split(';') == members
        assert int(row['origin']) == int(origins[n]) and row['channel'] == columns[c]
        assert n in indices[row['period']] and int(row['complete_horizon_count']) == int(complete[n].sum())
        for feature in ('last', 'recent24_mean', 'recent24_std', 'recent24_change'):
            close(float(row[feature]), float(features[(n, c)][feature]), 'supplement unchanged residual context feature')
        close(float(row['q_future_mean_signed_error']), outcomes[n][0][c], 'member-first complete-row Q signed error')
        close(float(row['q_future_mse']), outcomes[n][1][c], 'member-first complete-row Q MSE')
    assert keys == {(n, c) for n in outcomes for c in range(len(columns))}
    coverage = result['coverage']
    assert coverage['same_complete_horizon_mask_all_channels_and_level_seeds'] is True
    assert coverage['origin_channel_rows'] == len(origins) * len(columns)
    assert coverage['rows_with_q_outcomes'] == len(rows)
    assert coverage['origin_horizon_rows'] == complete.size and coverage['complete_origin_horizon_rows'] == int(complete.sum())
    close(coverage['coverage_fraction'], float(complete.mean()), 'supplement complete coverage')
    for period in ('test_a', 'test_b'):
        sub, recorded = complete[indices[period]], coverage['by_period'][period]
        assert recorded['origins'] == len(sub) and recorded['origin_horizon_rows'] == sub.size
        assert recorded['complete_origin_horizon_rows'] == int(sub.sum())
        assert recorded['min_complete_horizons_per_origin'] == int(sub.sum(axis=1).min())
        assert recorded['max_complete_horizons_per_origin'] == int(sub.sum(axis=1).max())
        close(recorded['coverage_fraction'], float(sub.mean()), 'supplement period coverage')
    correlations = csv_rows(result['files']['q_residual_error_correlations'])
    expected_correlations = {(period, c, feature, outcome) for period in ('test_a', 'test_b')
                             for c in range(len(columns)) for feature in ('last', 'recent24_mean', 'recent24_std', 'recent24_change')
                             for outcome in ('q_future_mean_signed_error', 'q_future_mse')}
    assert {(r['period'], int(r['channel_index']), r['feature'], r['outcome']) for r in correlations} == expected_correlations
    assert len(correlations) == len(expected_correlations)
    for row in correlations:
        assert int(row['n']) == sum(n in outcomes for n in indices[row['period']])
        assert 'mean seed LEVEL_RES Q' in row['semantics']
    assert set(result['figures']) == {'last_residual_to_q_signed_error', 'recent24std_to_q_mse'}
    for figures in result['figures'].values():
        assert set(figures) == {'png', 'pdf', 'svg'}
        for suffix, entry in figures.items():
            figure = check_receipt(entry)
            assert figure.suffix == '.' + suffix and figure.stat().st_size > 0
            assert figure.is_relative_to(HERE / 'diagnostics_residual_error')
    job.heartbeat('verified supplemental Q residual-error outcomes and scatter receipts')
    return {'artifact': receipt(path), 'numeric_origin_channel_rows': len(rows), 'correlation_rows': len(correlations),
            'figure_files': 6, 'coverage': coverage, 'member_first_complete_row_Q_projection_checked': True,
            'scope': 'Saved prediction Q outcomes and copied past features numerically checked; correlation keys/counts/semantics and figure hashes checked. '
                     'Correlation coefficients and rendered pixels are not independently recomputed.'}


def reports_check(contexts, job):
    report_path = HERE / 'report_values.json'
    report = read(report_path)
    assert report['status'] == 'complete'
    required_sources = {'robin_eval01.json', 'jena_eval01.json', 'cost_summary.json',
                        'microbatch_parity.json', 'robin_diagnostics_corrected_v7.json'}
    assert required_sources.issubset(report['sources'])
    for name, expected in report['sources'].items():
        file_entry(HERE / name, expected)
    expected_accuracy = set()
    for dataset, context in contexts.items():
        evaluation_result = context[1]
        assert report['comparisons'][dataset] == evaluation_result['comparisons']
        for role, periods in evaluation_result['role_scores'].items():
            expected_accuracy.update((dataset, 'mean_individual_seed_loss', role, period, None) for period in periods)
        for identifier, row in evaluation_result['models'].items():
            expected_accuracy.update((dataset, 'individual_model', row['role'], period, identifier) for period in row['scores'])
    actual_accuracy = set()
    for row in report['accuracy']:
        key = (row['dataset'], row['aggregation'], row['role'], row['period'], row.get('id'))
        assert key not in actual_accuracy, 'Duplicate report accuracy row: ' + repr(key)
        actual_accuracy.add(key)
        evaluation_result = contexts[row['dataset']][1]
        if row['aggregation'] == 'mean_individual_seed_loss':
            source = evaluation_result['role_scores'][row['role']][row['period']]
            assert row['model_count'] == source['count']
        else:
            assert row['aggregation'] == 'individual_model'
            model = evaluation_result['models'][row['id']]
            assert row['role'] == model['role'] and row['seed'] == model['spec'].get('seed')
            assert row['model_count'] == 1
            source = model['scores'][row['period']]
        for metric in ('mse', 'mae'):
            close(row[metric], source[metric], 'report source-linked accuracy ' + metric)
    assert actual_accuracy == expected_accuracy
    exported = {}
    for name in ('accuracy', 'training', 'cost'):
        rows = csv_rows(report['files'][name])
        assert len(rows) == len(report[name])
        fields = set().union(*(item.keys() for item in report[name]))
        for row, original in zip(rows, report[name]):
            assert set(row) == fields
            for key in fields:
                value = original.get(key)
                assert row[key] == ('' if value is None else str(value)), 'Report CSV differs: ' + name + '/' + key
        exported[name] = {'rows': len(rows), 'artifact': report['files'][name]}
    diagnostic = read(HERE / 'robin_diagnostics_corrected_v7.json')
    assert report['diagnostic']['coverage'] == diagnostic['coverage']
    space_path = check_receipt(report['diagnostic']['space_source'])
    assert space_path == check_receipt(diagnostic['files']['complete_row_space_summary'])
    spaces = {(r['period'], r['role']): r for r in csv_rows(diagnostic['files']['complete_row_space_summary'])
              if r['member_id'] == 'mean_of_member_losses'}
    assert len(report['diagnostic']['space_deltas']) == 8
    for row in report['diagnostic']['space_deltas']:
        left, right = spaces[(row['period'], 'level_res')], spaces[(row['period'], row['reference'])]
        assert row['complete_rows'] == int(left['complete_origin_horizon_rows']) == int(right['complete_origin_horizon_rows'])
        for output, source in (('delta_retained', 'p_retained_space_mse'),
                               ('delta_discarded', 'q_residual_space_mse'), ('delta_total', 'total_complete_row_mse')):
            close(row[output], float(left[source]) - float(right[source]), 'reported complete-row space delta')
    costs = {(r['family'], r['batch'], r['variant'], r['series_microbatch'], r['device']): r for r in report['cost']}
    relative = report['same_session_cost_changes']
    assert len(relative) == len([r for r in report['cost'] if r['device'] == 'cuda'])
    for row in relative:
        left = costs[(row['family'], row['batch'], row['variant'], row['series_microbatch'], 'cuda')]
        right = costs[('level_res', row['batch'], 'original', None, 'cuda')]
        assert row['reference'] == 'LEVEL original, same-session'
        for output, source in (('ms_relative_pct', 'mean_seed_median_ms_per_origin'),
                               ('throughput_relative_pct', 'mean_seed_median_origins_per_second'),
                               ('allocated_relative_pct', 'typical_peak_allocated_MiB')):
            close(row[output], 100 * (left[source] / right[source] - 1), 'reported same-session relative cost')
    manifest_path = HERE / 'figure_manifest.json'
    manifest = read(manifest_path)
    required_inputs = {'robin_eval01.json', 'jena_eval01.json', 'cost_summary.json', 'cost_gpu_rows.json',
                       'diagnostics_corrected/robin_complete_row_space_summary_corrected.csv'}
    assert required_inputs.issubset({name.replace('\\', '/') for name in manifest['inputs']})
    for name, expected in manifest['inputs'].items():
        file_entry(HERE / name, expected)
    assert set(manifest['figures']) == {'accuracy', 'diagnostics', 'tradeoff'}
    figures = []
    for name, entries in manifest['figures'].items():
        paths = [check_receipt(entry) for entry in entries]
        assert len(paths) == 3 and {path.suffix for path in paths} == {'.png', '.pdf', '.svg'}
        assert all(path.is_relative_to(HERE / 'figures') and path.stat().st_size > 0 for path in paths)
        figures.extend(paths)
    assert len(set(figures)) == 9
    documents = {}
    for name in ('PAPER_DRAFT.md', 'APPENDIX.md', 'CLAIMS_PRIOR_ART.md', 'TOPIC_DECISION.md'):
        path = HERE / name
        assert path.is_file() and path.stat().st_size > 0, 'Required final document missing: ' + name
        documents[name] = receipt(path)
    job.heartbeat('verified report/CSV links and figure/document artifacts')
    return {'report': receipt(report_path), 'source_hashes_checked': len(report['sources']),
            'source_linked_accuracy_rows': len(actual_accuracy), 'csv_exports': exported,
            'complete_row_deltas_and_same_session_cost_ratio_arithmetic': True,
            'figure_manifest': receipt(manifest_path), 'figure_files': len(figures), 'documents': documents,
            'scope': 'Report accuracy compared to evaluation losses; CSV values/counts/hashes and figure inputs/files checked. '
                     'Figure pixels and manuscript numeric prose are not automatically reinterpreted; document existence/hash is not scientific validation.'}


def commands_manifest(ledger):
    entries = []
    for path in sorted(HERE.rglob('*.json')):
        if path.name in ('ledger.json', 'commands.json', 'final_checks.json'):
            continue
        row = read(path)
        if not isinstance(row, dict):
            continue
        command = row.get('command') or row.get('environment', {}).get('command')
        if command:
            entries.append({'artifact': receipt(path), 'command': command,
                            'ledger_id': row.get('ledger_id'), 'status': row.get('status')})
    result = {'scope': 'Commands actually recorded in artifact receipts; ledger-only bookkeeping may lack argv.',
              'entries': entries, 'ledger_jobs': [{k: r.get(k) for k in ('id', 'label', 'category', 'status', 'metadata')} for r in ledger['jobs']]}
    save_json(HERE / 'commands.json', result)
    return {'entries': len(entries), 'artifact': receipt(HERE / 'commands.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', default='final_checks')
    parser.add_argument('--output', default='final_checks.json')
    parser.add_argument('--reserve-seconds', type=float, default=300)
    args = parser.parse_args()
    target = HERE / args.output
    if target.exists():
        raise RuntimeError('Preserve previous verification; use a new output filename')
    with Job(args.job, category='cpu_check', reserve_seconds=args.reserve_seconds) as job:
        global np
        import numpy as np
        result = {'status': 'running', 'started_utc': time.time(), 'source': source_receipt(),
                  'scope': 'Saved evidence CPU audit; no model construction/forward, fitting, new predictions or benchmark.',
                  'independent_replication': False, 'scientific_claim_support_separate': True}
        try:
            protocol = read(HERE / 'protocol.json')
            result['preservation'] = preservation()
            ledger = read(HERE / 'ledger.json')
            result['training'] = [training(dataset, protocol, ledger, job) for dataset in ('robin', 'jena')]
            result['evaluations'] = {}
            contexts = {}
            for dataset in ('robin', 'jena'):
                result['evaluations'][dataset], contexts[dataset] = evaluation(dataset, job)
            result['cost'] = cost_checks(job, contexts['robin'])
            result['diagnostics'] = diagnostics_check(contexts['robin'], job)
            result['supplemental_diagnostics'] = supplemental_diagnostics_check(contexts['robin'], job)
            result['reports'] = reports_check(contexts, job)
            result['budget'] = budget_check(job.id, protocol)
            result['commands'] = commands_manifest(read(HERE / 'ledger.json'))
            result.update(status='PASS_WITH_RECORDED_ACCOUNTING_DEVIATION', completed_utc=time.time())
            save_json(target, result)
            job.heartbeat('bounded saved-artifact checks complete', 0)
        except BaseException:
            result.update(status='FAIL', failed_utc=time.time(), error=traceback.format_exc())
            save_json(target, result)
            raise


if __name__ == '__main__':
    main()
