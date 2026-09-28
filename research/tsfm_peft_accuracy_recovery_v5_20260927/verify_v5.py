"""CPU-only saved-artifact verification; never constructs or executes a model."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np

from runtime_v5 import HERE, ROOT, CACHE, Job, LIMITS, budget_snapshot, digest, phase_sample, save_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-9, rtol=1e-9):
    if actual is None or expected is None:
        require(actual is expected, label + ': null mismatch')
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


HASHES = {}
DATA = {}


def file_hash(path):
    path = resolve(path)
    require(path.is_file(), 'Missing artifact: ' + str(path))
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in HASHES:
        HASHES[key] = digest(path)
    return HASHES[key]


def receipt(record):
    require(file_hash(record['path']) == record['sha256'], 'Receipt mismatch: ' + str(record['path']))
    return resolve(record['path'])


def source_version(source, filename):
    key = (HERE / filename).relative_to(ROOT).as_posix()
    wanted = source[key]
    if file_hash(HERE / filename) == wanted:
        return {'path': key, 'sha256': wanted, 'current': True}
    for path in sorted((HERE / 'repairs').rglob('*.py')):
        if file_hash(path) == wanted:
            return {'path': path.relative_to(ROOT).as_posix(), 'sha256': wanted, 'current': False}
    raise AssertionError('Executed core source not current or archived: ' + filename + ':' + wanted)


def data_for(dataset, evaluation=False):
    key = (dataset, evaluation)
    if key in DATA:
        return DATA[key]
    if dataset == 'hog':
        path = ROOT / '.cache/tsfm_peft_fresh_group_v4_20260927/data' / ('test.npz' if evaluation else 'trainval.npz')
    else:
        name = 'bdg2_bull_office17_l512_h48.npz' if dataset == 'bull' else 'electricity_first32_l512_h48.npz'
        path = ROOT / '.cache/tsfm_peft_development_20260926/data' / name
    with np.load(path, allow_pickle=False) as archive:
        needed = ('x', 'finite', 'targetmask', 'columns', 'train_origins', 'val_origins',
                  'test_a_origins', 'test_b_origins', 'eval_e1_origins', 'eval_e2_origins', 'dev_origins')
        data = {k: archive[k] for k in needed if k in archive.files}
    if 'finite' not in data:
        data['finite'] = data.pop('targetmask')
    data['_path'] = path
    require(data['x'].shape == data['finite'].shape, 'Data/mask axes differ')
    DATA[key] = data
    return data


def score(prediction, data, origins):
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
    require(prediction.shape == target.shape == mask.shape, 'Prediction/target/mask axes differ')
    require(np.isfinite(prediction).all() and np.isfinite(target[mask]).all(), 'Nonfinite observed output')
    delta = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0)
    # Preserve the original horizon-then-origin summation and per-channel denominator.
    squared = (delta * delta).sum(axis=1).sum(axis=0)
    absolute = np.abs(delta).sum(axis=1).sum(axis=0)
    count = mask.sum(axis=1).sum(axis=0)
    require((count > 0).all(), 'A channel has no observed targets')
    return {'mse': float(np.mean(squared / count)), 'mae': float(np.mean(absolute / count)),
            'channel_mse': (squared / count).tolist(), 'channel_mae': (absolute / count).tolist(),
            'channel_squared_error_sum': squared.tolist(), 'channel_absolute_error_sum': absolute.tolist(),
            'channel_target_count': count.tolist()}


def compare_score(actual, saved, label):
    aliases = {'channel_squared_error_sum': 'squared_error_sums',
               'channel_absolute_error_sum': 'absolute_error_sums', 'channel_target_count': 'target_counts'}
    for key, value in actual.items():
        saved_key = key if key in saved else aliases.get(key, key)
        if saved_key in saved:
            close(value, saved[saved_key], label + '/' + key)
    require('mse' in saved and 'mae' in saved, label + ': missing primary metrics')


def load_prediction(record):
    path = receipt(record)
    with np.load(path, allow_pickle=False) as archive:
        return archive['prediction'], archive['origins']


def verify_provenance():
    provenance = read(HERE / 'provenance.json')
    for name, key in (('PLAN.md', 'plan_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('approved_request.txt', 'original_request_sha256')):
        require(file_hash(HERE / name) == provenance[key], 'Approved source changed: ' + name)
    for name, expected in provenance['old_tracked_hashes'].items():
        require(file_hash(ROOT / name) == expected, 'Old tracked file changed: ' + name)
    protocol = read(HERE / 'protocol.json')
    for key, value in protocol['limits'].items():
        require(LIMITS[key] == value, 'Runtime/protocol budget mismatch: ' + key)
    diagnosis = read(HERE / 'v5_hog_diagnosis_01/diagnosis.json')
    archive = read(HERE / 'repairs/diagnose01_source_provenance.json')
    receipt(archive['preserved_executed_source'])
    receipt(archive['diff'])
    require(archive['preserved_executed_source']['sha256'] == diagnosis['script']['sha256'],
            'Diagnosis executed source archive mismatch')
    require(resolve(archive['preserved_executed_source']['path']).stat().st_size == diagnosis['script']['bytes'],
            'Diagnosis executed source byte count mismatch')
    initial_decision = read(HERE / 'decision_initial.json')
    receipt({'path': initial_decision['diagnosis_path'], 'sha256': initial_decision['diagnosis_sha256']})
    return {'approved_files_unchanged': True, 'old_tracked_files_unchanged': len(provenance['old_tracked_hashes']),
            'diagnosis_executed_source_preserved': archive['preserved_executed_source'],
            'source_policy': 'Core model/train versions must match current or exact archived bytes. Later helpers do not invalidate earlier training.'}


def verify_run(path, ledger):
    import torch
    torch.set_num_threads(4)
    result, saved = read(path / 'result.json'), read(path / 'receipt.json')
    spec = result['spec']
    require(result['status'] == 'complete', 'Incomplete result')
    require(result['id'] == saved['id'] == path.name, 'Run identity mismatch')
    require(file_hash(path / 'result.json') == saved['result_sha256'], 'Result hash mismatch')
    require(file_hash(path / 'curve.json') == saved['curve_sha256'], 'Curve hash mismatch')
    attempt = read(path / 'attempt.json')
    require(attempt['status'] == 'complete' and attempt['ledger_id'] == result['ledger_id'], 'Attempt incomplete')
    entry = ledger['jobs'][result['ledger_id']]
    require(entry['label'] == result['id'] and entry['category'] == 'real_fit' and entry['status'] == 'complete', 'Fit ledger mismatch')
    require(entry['updates'] == attempt['actual_updates'] == result['attempt_updates'], 'Actual updates mismatch')
    core = [source_version(result['source'], filename) for filename in ('model_v5.py', 'train_v5.py')]
    for key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('initial_checkpoint', 'initial_checkpoint_sha256'),
                          ('restart_checkpoint', 'restart_sha256')):
        receipt({'path': result[key], 'sha256': result[hash_key]})
    require(saved['checkpoint_sha256'] == result['checkpoint_sha256'], 'Checkpoint receipt disagreement')
    receipt(saved['initial'])
    require(saved['initial'] == result['initial_source'], 'Initial receipt disagreement')
    require(saved['best_val_prediction'] == result['best_val_prediction'], 'VAL receipt disagreement')
    data = data_for(spec['dataset'])
    require(file_hash(data['_path']) == result['data_sha256'] == result['initial_source']['data_sha256'], 'TRAIN/VAL source changed')
    pred, origins = load_prediction(result['best_val_prediction'])
    require(np.array_equal(origins, data['val_origins']), 'Selected VAL origins mismatch')
    val = score(pred, data, origins)
    close(val['mse'], result['best_val_mse'], result['id'] + '/best VAL')
    history = read(path / 'curve.json')
    require([r['epoch'] for r in history] == list(range(result['epochs_completed'] + 1)), 'Missing curve epochs')
    best = min(history, key=lambda row: row['val_mse'])
    require(best['epoch'] == result['selected_epoch'], 'Earliest minimum VAL checkpoint not selected')
    close(best['val_mse'], result['best_val_mse'], 'Selected curve value')
    close(history[0]['val_mse'], result['initial_val_mse'], 'Initial VAL')
    require(result['total_steps'] == 128 * result['epochs_completed'], 'Unexpected epoch update count')
    require(result['attempt_updates'] > 0, 'No real-data updates')
    require(result['epochs_completed'] <= spec['epochs'] <= 240, 'Epoch contract exceeded')
    decision = result['decision_receipt']['decision']
    require(spec['method'] in (decision['first_method'], decision['reserve_method']), 'Unapproved method')
    require(spec['dataset'] in decision['approved_datasets'], 'Unapproved dataset')
    if spec['lr'] != 1e-3 or spec['epochs'] != 120 or spec.get('resume_from'):
        require(result['id'] in decision['authorized_supplements'], 'Supplement missing authorization')
    prior_lr = spec['lr']
    for row in history:
        require(set(row['learning_rates']) == {'ed_g'}, 'Unexpected optimizer groups')
        lr = row['learning_rates']['ed_g']
        require(0 < lr <= prior_lr, 'LR nonpositive or increased')
        close(math.log2(spec['lr'] / lr), round(math.log2(spec['lr'] / lr)), 'Plateau LR ratio')
        prior_lr = lr
    close(history[0]['learning_rates']['ed_g'], spec['lr'], 'Initial LR')
    require([r['epoch'] for r in result['schedules']] == list(range(1, result['epochs_completed'] + 1)), 'Missing TRAIN schedule')
    for row in result['schedules']:
        expected = phase_sample(data['train_origins'], spec['seed'], row['epoch'], 512).astype('<i8')
        require(hashlib.sha256(expected.tobytes()).hexdigest() == row['sha256'], 'TRAIN sample order changed')
    with np.load(resolve(result['initial_checkpoint']).parent / 'origins.npz', allow_pickle=False) as archive:
        require(np.array_equal(archive['val'], data['val_origins']), 'Run VAL origin receipt changed')
        require(np.array_equal(archive['train_probe'], phase_sample(data['train_origins'], spec['seed'], 0, 96)), 'Fixed TRAIN probe changed')
    # Trusted local tensor archives only; no model, optimizer or scheduler is instantiated.
    initial = torch.load(resolve(result['initial_checkpoint']), map_location='cpu', weights_only=False)
    selected = torch.load(resolve(result['checkpoint']), map_location='cpu', weights_only=False)
    restart = torch.load(resolve(result['restart_checkpoint']), map_location='cpu', weights_only=False)
    shared = torch.load(receipt(result['initial_source']), map_location='cpu', weights_only=False)
    require(initial['epoch'] == initial['steps'] == 0, 'Initial checkpoint is trained')
    require(selected['epoch'] == result['selected_epoch'], 'Selected epoch/checkpoint mismatch')
    require(restart['epoch'] == result['epochs_completed'] and restart['steps'] == result['total_steps'], 'Restart position mismatch')
    require(restart['next_batch'] == 0 and restart['boundary'] == 'post-VAL, post-scheduler, complete epoch', 'Restart boundary mismatch')
    require(all(key in restart for key in ('optimizer', 'scheduler', 'rng', 'history', 'schedules')), 'Incomplete restart state')
    require(restart['history'] == history and restart['schedules'] == result['schedules'], 'Restart history mismatch')
    require(selected['initial_sha256'] == initial['initial_sha256'] == result['initial_source']['sha256'], 'Initial source chain mismatch')
    require(set(initial['state']) == set(selected['state']) == set(restart['state']) == set(shared['state']), 'Adapter state key mismatch')
    trainable = 0
    for name, tensor in initial['state'].items():
        require(not name.startswith('backbone.'), 'Frozen backbone duplicated in adapter checkpoint')
        hashed = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        require(hashed == result['initial_parameter_sha256'][name] == result['initial_source']['tensor_hashes'][name], 'Initial tensor hash mismatch')
        require(torch.equal(tensor, shared['state'][name]), 'Shared initial tensor differs')
        change = float((restart['state'][name] - tensor).abs().max())
        selected_change = float((selected['state'][name] - tensor).abs().max())
        close(change, result['parameter_max_updates'][name], 'Actual parameter change')
        close(selected_change, result['selected_parameter_max_updates'][name], 'Selected parameter change')
        require(int(torch.count_nonzero(restart['state'][name] != tensor)) == result['changed_parameter_scalars'][name], 'Updated parameter count')
        frozen = spec['ed_mode'] == 'fixed_ed' and name in ('encoder.weight', 'decoder.weight')
        require(change == 0 if frozen else change > 0, 'Frozen/trainable update mismatch: ' + name)
        if not frozen:
            trainable += tensor.numel()
        if name in ('residual.up.weight', 'residual.linear.weight'):
            require(int(torch.count_nonzero(tensor)) == 0, 'G output initial value is not zero')
        for gradients in result['first_update_gradients'].values():
            gradient = gradients[name]
            require(gradient['requires_grad'] != frozen, 'Wrong requires_grad: ' + name)
            if frozen:
                require(gradient['max_abs'] is None, 'Frozen parameter received gradient')
            else:
                require(gradient['finite'] is True, 'Nonfinite/missing trainable gradient')
        if result['first_update_gradients'] and not frozen:
            require(any(g[name]['max_abs'] > 0 for g in result['first_update_gradients'].values()), 'No early gradient: ' + name)
    require(trainable == result['trainable_parameters'], 'Trainable parameter count mismatch')
    frozen = result['frozen_parameters_verified']
    require(all(frozen[key] is True for key in ('no_gradient', 'version_unchanged', 'optimizer_exclusion')), 'Frozen path checks failed')
    require(re.fullmatch('[a-f0-9]{64}', frozen['backbone_sha256_unchanged']) is not None, 'Missing frozen backbone digest')
    require(all(result['restart_roundtrip'][key] is True for key in ('model', 'optimizer', 'scheduler', 'rng')),
            'Recorded restart roundtrip failed')
    require(result['restart_roundtrip']['extra_updates'] == 0, 'Restart check made extra updates')
    require(math.isfinite(result['replay_error']) and result['replay_error'] >= 0, 'Invalid recorded replay error')
    require(result['training_peak']['reserved_bytes'] >= result['training_peak']['allocated_bytes'] > 0, 'Training memory accounting')
    return result, {'id': result['id'], 'result_sha256': saved['result_sha256'], 'selected_epoch': result['selected_epoch'],
                    'stored_val_mse': val['mse'], 'adapter_tensors_checked': len(initial['state']),
                    'replay_scope': 'Recorded training-time replay checked; no new forward pass', 'core_sources': core}


def verify_pairs(results):
    grouped = defaultdict(lambda: defaultdict(list))
    for result in results.values():
        spec = result['spec']
        grouped[(spec['dataset'], spec['method'], spec['seed'], spec['lr'], spec['epochs'])][spec['variant']].append(result)
    reports = []
    for key, arms in grouped.items():
        if not all(variant in arms for variant in ('res', 'raw')):
            reports.append({'configuration': list(key), 'paired': False, 'scope': 'Unpaired supplement/attempt; cannot alone establish matched-arm value'})
            continue
        for left in arms['res']:
            for right in arms['raw']:
                require(left['initial_parameter_sha256'] == right['initial_parameter_sha256'], 'RES/RAW initial tensors differ')
                require(left['trainable_parameters'] == right['trainable_parameters'], 'RES/RAW trainable capacity differs')
                count = min(len(left['schedules']), len(right['schedules']))
                require(left['schedules'][:count] == right['schedules'][:count], 'RES/RAW common epoch schedules differ')
                if not left['spec'].get('resume_from') and not right['spec'].get('resume_from'):
                    close(left['initial_val_mse'], right['initial_val_mse'], 'RES/RAW initial function VAL')
                reports.append({'left': left['id'], 'right': right['id'], 'paired': True,
                                'same_initial_tensors': True, 'same_trainable_count': True, 'common_epochs': count})
    return reports


def verify_evaluation(path, runs):
    result = read(path)
    dataset = result['dataset']
    data = data_for(dataset, True)
    require(file_hash(data['_path']) == result['data_sha256'], 'Evaluation data hash mismatch')
    require(result['columns'] == data['columns'].tolist(), 'Evaluation channels changed')
    keys = {'hog': {'test_a': 'test_a_origins', 'test_b': 'test_b_origins'},
            'bull': {'e1': 'eval_e1_origins', 'e2': 'eval_e2_origins'}, 'electricity': {'dev': 'dev_origins'}}[dataset]
    periods = {p: data[k] for p, k in keys.items()}
    require(result['period_origins'] == {p: o.tolist() for p, o in periods.items()}, 'Period origins changed')
    all_origins = np.concatenate(list(periods.values()))
    indices, offset = {}, 0
    for period, origins in periods.items():
        indices[period] = np.arange(offset, offset + len(origins))
        offset += len(origins)
    indices['combined'] = np.arange(len(all_origins))
    metrics = {}
    for identifier, model in result['models'].items():
        receipts = model['predictions']
        if 'combined' in receipts:
            prediction, origins = load_prediction(receipts['combined'])
            require(np.array_equal(origins, all_origins), 'Combined prediction origins mismatch')
        else:
            chunks = []
            for period, wanted in periods.items():
                chunk, origins = load_prediction(receipts[period])
                require(np.array_equal(origins, wanted), 'Period prediction origins mismatch')
                chunks.append(chunk)
            prediction = np.concatenate(chunks)
        metrics[identifier] = {}
        for period, ix in indices.items():
            actual = score(prediction[ix], data, all_origins[ix])
            compare_score(actual, model['scores'][period], identifier + '/' + period)
            metrics[identifier][period] = actual
        if model['role'] in ('modified_res', 'modified_raw'):
            require(identifier in runs and model['spec'] == runs[identifier]['spec'], 'Evaluation run mapping mismatch')
    require(set(result['roles']) >= {'modified_res', 'modified_raw', 'old_res', 'old_raw', 'f0', 'lora'}, 'Missing required comparison role')
    for role, identifiers in result['roles'].items():
        require(all(result['models'][i]['role'] == role for i in identifiers), 'Role mapping mismatch')
        if role in ('modified_res', 'modified_raw'):
            require(sorted(result['models'][i]['spec']['seed'] for i in identifiers) == [92601, 92602], 'Missing selected two seeds')
        for period in indices:
            for metric in ('mse', 'mae'):
                close(np.mean([metrics[i][period][metric] for i in identifiers]), result['role_scores'][role][period][metric],
                      role + '/' + period + '/mean individual seed loss')
    for comparison in result['comparisons']:
        for period, values in comparison['periods'].items():
            left = float(np.mean([metrics[i][period]['mse'] for i in comparison['left_seed_ids']]))
            right = float(np.mean([metrics[i][period]['mse'] for i in comparison['right_seed_ids']]))
            wanted = {'left_mse': left, 'reference_mse': right, 'delta_mse': left - right,
                      'relative_to_reference_pct': 100 * (left - right) / right if right > 0 else None}
            for key, value in wanted.items():
                close(values['full'][key], value, comparison['name'] + '/' + period + '/' + key)
            for row in values['paired_seed_deltas']:
                a, b = [metrics[row[key]][period]['mse'] for key in ('left_id', 'right_id')]
                close(row['delta_mse'], a - b, 'Paired seed difference')
                close(row['relative_to_reference_pct'], 100 * (a - b) / b, 'Paired seed relative change')
            require(values['draws'] == 2000, 'Bootstrap draw count changed')
            # Do not generate fresh resamples during verification.
            intervals = values['conditional_block95'] or {}
            for interval in intervals.values():
                if interval is not None:
                    require(len(interval) == 2 and interval[0] <= interval[1], 'Invalid saved interval')
            require(values['strata'] == ([period] if period != 'combined' else list(periods)),
                    'Bootstrap strata cross a period boundary')
    prediction_manifest = read(path.with_name(path.stem + '_predictions.json'))
    for entry in prediction_manifest['entries']:
        require(entry['id'] in result['models'], 'Saved prediction omitted from final evaluation')
        prediction, origins = load_prediction({'path': entry['predictions']['combined'], 'sha256': entry['prediction_sha256']})
        require(np.array_equal(origins, all_origins), 'Prediction manifest origins mismatch')
        if 'result_path' in entry:
            receipt({'path': entry['result_path'], 'sha256': entry['result_sha256']})
            require(entry['checkpoint_sha256'] == runs[entry['id']]['checkpoint_sha256'], 'Prediction checkpoint mismatch')
            compare_score(metrics[entry['id']]['combined'], entry['combined_score'], 'Prediction combined score')
        else:
            require(entry['role'] == 'level_only' and entry['fit_count'] == 0, 'Unexpected untrained reference')
            receipt(entry['reference_source'])
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path), 'dataset': dataset,
            'models_rescored': len(metrics), 'period_origins': {k: len(v) for k, v in periods.items()},
            'scope': 'Saved predictions only; channel-wise masked sums pooled before macro averaging; mean seed losses, no ensemble; intervals not regenerated'}


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': float(min(values)), 'max': float(max(values))}


def compare_summary(saved, values, label):
    for key, value in summary(values).items():
        close(saved[key], value, label + '/' + key)


def verify_cost(path):
    result = read(path)
    rows = result['rows']
    require(len(rows) == result['expected_rows'] == 60, 'Incomplete same-session cost rows')
    require(result['new_fits'] == 0 and result['no_target_scores'] is True, 'Cost stage performed model fitting/scoring')
    require(result['single_uninterrupted_session'] is True and result['resume'] is None, 'Mixed cost sessions')
    require(result['precision'] == 'float32' and result['torch_threads'] == 4, 'Cost precision/threads changed')
    require(result['allow_tf32_matmul'] is False and result['allow_tf32_cudnn'] is False, 'Cost TF32 changed')
    require((result['warmup_calls'], result['passes_per_order_block'], result['order_blocks']) == (10, 20, 3), 'Cost design changed')
    cost_source = source_version(result['source'], 'cost_v5.py')
    contract = read(receipt(result['contract']))
    receipt(contract['val_data'])
    require(len(contract['val_origins']) == 24 and len(contract['model_ids']) == 9, 'Cost input/model count')
    schedule = [(block, ordinal, identifier, sentinel, batch) for block, ordinal, identifier, sentinel in contract['order'] for batch in (1, 4)]
    require(schedule == [(r['block'], r['order_position'], r['id'], r['sentinel_role'], r['batch_origins']) for r in rows], 'Cost order differs from contract')
    grouped = defaultdict(list)
    for row in rows:
        require(row['measurement_run'] == result['name'], 'Historical/new cost rows mixed')
        require(row['measurement_cost_source_sha256'] == cost_source['sha256'], 'Mixed cost source versions')
        require(row['include_in_primary'] == (row['sentinel_role'] != 'post'), 'F0 post sentinel filtering')
        require(row['baseline_memory']['allocated'] == 0, 'Leaked allocation at model boundary')
        require(row['peak_reserved_bytes'] >= row['peak_allocated_bytes'] > 0, 'Cost memory fields invalid')
        require(row['total_deployed_tensor_bytes'] == row['deployed_parameter_bytes'] + row['deployed_buffer_bytes'], 'Deployment byte sum mismatch')
        require(row['deployed_parameter_bytes'] == 4 * row['total_deployed_parameters'], 'FP32 parameter byte mismatch')
        require(row['validation_replay']['pass'] is True, 'Saved VAL replay failed')
        receipt(row['validation_replay']['prediction'])
        for key in ('checkpoint', 'result'):
            if row.get(key):
                receipt(row[key])
        if row['family'] == 'lora':
            require(row['merge_parity']['pass'] is True and row['merge_parity']['returned_merged_backbone_used'] is True, 'LoRA merge validation failed')
        times = np.asarray(row['seconds_per_24_origins'], dtype=float)
        require(len(times) == 20 and np.isfinite(times).all() and (times > 0).all(), 'Raw timing passes invalid')
        compare_summary(row['milliseconds_per_origin'], times * 1000 / 24, 'Raw timing ms')
        compare_summary(row['origins_per_second'], 24 / times, 'Raw timing throughput')
        if row['include_in_primary']:
            grouped[(row['id'], row['batch_origins'], row['scope'])].append(row)
    require(sum(map(len, grouped.values())) == 54, 'Primary cost row count')
    fits = result['summary']['fits']
    require(len(fits) == len(grouped) == 18, 'Cost fit summary count')
    family_groups = defaultdict(list)
    for fit in fits:
        members = grouped[(fit['id'], fit['batch_origins'], fit['scope'])]
        require(len(members) == 3 and {r['block'] for r in members} == {0, 1, 2}, 'Missing cost blocks')
        for key in ('milliseconds_per_origin', 'origins_per_second'):
            compare_summary(fit[key], [r[key]['median'] for r in members], 'Cost block summary')
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            require(fit[key] == [r[key] for r in members], 'Memory blocks discarded')
        family_groups[(fit['family'], fit['ed_mode'], fit['batch_origins'], fit['scope'])].append(fit)
    for group in result['summary']['groups']:
        members = family_groups[(group['family'], group['ed_mode'], group['batch_origins'], group['scope'])]
        require(group['fit_ids'] == [r['id'] for r in members], 'Cost family fit mapping')
        close(group['mean_fit_median_ms_per_origin'], np.mean([r['milliseconds_per_origin']['median'] for r in members]), 'Cost group latency')
        close(group['mean_fit_median_origins_per_second'], np.mean([r['origins_per_second']['median'] for r in members]), 'Cost group throughput')
    for sentinel in result['drift_checks']['f0_pre_post']:
        pair = {r['sentinel_role']: r for r in rows if r['sentinel_role'] and r['block'] == sentinel['block'] and r['batch_origins'] == sentinel['batch_origins']}
        pre, post = pair['pre']['milliseconds_per_origin'], pair['post']['milliseconds_per_origin']
        close(sentinel['post_over_pre'], post['median'] / pre['median'], 'Sentinel drift ratio')
        require(sentinel['iqr_disjoint'] == (pre['q75'] < post['q25'] or post['q75'] < pre['q25']), 'Sentinel drift hidden')
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path), 'rows': len(rows), 'primary_rows': 54,
            'fits': len(fits), 'raw_timing_and_summary_arithmetic': True, 'merge_parity_recorded': True,
            'scope': 'Stored timing arithmetic and integrity only; no new benchmark or independent timing replication'}


def verify_ledger(own_id):
    ledger = read(HERE / 'ledger.json')
    require([r['id'] for r in ledger['jobs']] == list(range(len(ledger['jobs']))), 'Ledger IDs not append-only contiguous')
    require(len({r['label'] for r in ledger['jobs']}) == len(ledger['jobs']), 'Reused ledger label')
    failures = []
    for row in ledger['jobs']:
        if row['id'] == own_id:
            continue
        require(row['status'] != 'running', 'Other active job: ' + row['label'])
        require(row['status'] in ('complete', 'failed'), 'Unknown ledger status')
        require(row['elapsed_s'] >= 0 and row['ended_utc'] >= row['started_utc'], 'Job time invalid')
        if row.get('synthetic'):
            require(row['category'] == 'cpu_check' and 0 <= row['updates'] <= LIMITS['synthetic_steps'], 'Synthetic update cap exceeded')
        if row['status'] == 'failed':
            require(bool(row.get('error')), 'Failed job traceback not preserved')
            record = {'id': row['id'], 'label': row['label'], 'category': row['category'], 'updates': row.get('updates', 0)}
            if row['category'] == 'real_fit':
                path = HERE / 'runs' / row['label'] / 'attempt.json'
                attempt = read(path)
                require(attempt['status'] == 'failed' and bool(attempt.get('error')), 'Failed fit attempt not preserved')
                record['attempt'] = {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)}
            failures.append(record)
    snapshot = budget_snapshot()
    for key, limit in (('real_fit_attempts', 'real_fit'), ('synthetic_sessions', 'synthetic_sessions'),
                       ('gpu_seconds', 'gpu_seconds'), ('cpu_check_seconds', 'cpu_check_seconds'), ('storage_bytes', 'storage_bytes')):
        require(snapshot[key] <= LIMITS[limit], 'Budget exceeded: ' + key)
    diagnosis_seconds = sum(row.get('elapsed_s', 0) for row in ledger['jobs']
                            if row['label'].startswith('v5_hog_diagnosis_'))
    require(diagnosis_seconds <= 1800, 'Bounded initial CPU diagnosis cap exceeded')
    snapshot['active_jobs'] = [r for r in snapshot['active_jobs'] if r['id'] != own_id]
    require(not snapshot['active_jobs'], 'Other active job')
    snapshot['own_check_in_progress'] = own_id
    snapshot['gpu_accounting'] = 'GPU job elapsed time only; nested fit elapsed time is not added twice.'
    return ledger, failures, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='final_checks')
    parser.add_argument('--reserve-seconds', type=float, default=180)
    parser.add_argument('--require-cost', action='store_true')
    args = parser.parse_args()
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', args.name) is not None, 'Invalid output name')
    output = HERE / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve prior verification and use a new name: ' + str(output))
    result = {'status': 'running', 'scope': 'Execution-team saved-artifact verification; not independent reproduction.',
              'script': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': file_hash(__file__)},
              'new_model_executions': 0, 'new_fits': 0, 'new_predictions': 0, 'new_benchmarks': 0,
              'stored_prediction_rescoring': True, 'started_utc': time.time()}
    try:
        with Job(args.name, category='cpu_check', reserve_seconds=args.reserve_seconds,
                 metadata={'saved_artifact_verification': True, 'optimizer_updates': 0}) as job:
            result['provenance'] = verify_provenance()
            ledger, failures, snapshot = verify_ledger(job.id)
            results, checked = {}, []
            for path in sorted((HERE / 'runs').glob('*/result.json')):
                run, report = verify_run(path.parent, ledger)
                results[run['id']] = run
                checked.append(report)
                job.heartbeat('verified run ' + run['id'])
            complete_ledger = {r['label'] for r in ledger['jobs'] if r['category'] == 'real_fit' and r['status'] == 'complete'}
            require(set(results) == complete_ledger, 'Completed fit missing a result or ledger entry')
            result['runs'] = checked
            result['paired_controls'] = verify_pairs(results)
            result['evaluations'], result['costs'] = [], []
            for path in sorted(HERE.glob('*.json')):
                document = read(path)
                if document.get('status') == 'complete' and document.get('evaluation_scope') == 'exposed development' and 'role_scores' in document:
                    result['evaluations'].append(verify_evaluation(path, results))
                    job.heartbeat('verified saved evaluation ' + path.stem)
                elif document.get('status') == 'complete' and 'rows' in document and 'summary' in document and document.get('expected_rows') == 60:
                    result['costs'].append(verify_cost(path))
                    job.heartbeat('verified saved cost ' + path.stem)
            require(bool(result['evaluations']), 'No completed evaluation to verify')
            if args.require_cost:
                require(bool(result['costs']), 'Required final cost measurement missing')
            result['cost_status'] = 'verified' if result['costs'] else 'not performed; no cost claim verified'
            _, result['preserved_failed_jobs'], result['budget'] = verify_ledger(job.id)
            result.update(status='complete', completed_utc=time.time(), artifact_hashes_checked=len(HASHES))
            save_json(output, result)
            job.heartbeat('complete; no model execution, optimizer update, bootstrap or benchmark')
        # Report the actual closed check time without keeping the verifier marked active.
        result['budget'] = budget_snapshot()
        result['budget']['gpu_accounting'] = 'GPU job elapsed time only; nested fit elapsed time excluded.'
        save_json(output, result)
        print(json.dumps({'path': str(output), 'status': result['status'], 'runs': len(result['runs']),
                          'evaluations': len(result['evaluations']), 'costs': len(result['costs'])}), flush=True)
    except BaseException:
        result.update(status='failed', error=traceback.format_exc(), completed_utc=time.time())
        save_json(output, result)
        raise


if __name__ == '__main__':
    main()
