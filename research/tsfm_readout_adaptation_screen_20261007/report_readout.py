"""CPU-only checks and Korean reporting of the completed, sealed S1 campaign.

No model, optimizer, dataset, or prediction-array loading occurs here. NPZ
headers and recorded channel sufficient statistics are checked independently.
The caller may pass its central CPU budget guard to include this work in S1.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASETS = ('robin', 'peacock_education', 'jena')
ARMS = ('OUTPUT_AFFINE', 'NATIVE_HEAD')
SEEDS = (92601, 92602)
PERIODS = ('test_a', 'test_b', 'combined')
PAIRS = (('OUTPUT_AFFINE', 'F0'), ('NATIVE_HEAD', 'F0'),
         ('NATIVE_HEAD', 'OUTPUT_AFFINE'), ('OUTPUT_AFFINE', 'MSE_LORA'),
         ('NATIVE_HEAD', 'MSE_LORA'), ('OUTPUT_AFFINE', 'LEVEL'),
         ('NATIVE_HEAD', 'LEVEL'))


def _read(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return json.load(handle)


def _write_json(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def _write_text(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(value)
    os.replace(temporary, path)


def _require(condition, message):
    if not condition:
        raise RuntimeError('STOP_DEBUG: report verification: ' + message)


def _close(a, b, message):
    _require(math.isfinite(float(a)) and math.isfinite(float(b))
             and math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), message)


def _hash(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Receipts:
    def __init__(self, budget):
        self.budget = budget
        self.checked = {}
        self.large_metadata_only = {}

    def verify(self, entry, allow_large=False):
        path = Path(entry['path'].replace('\\', '/'))
        path = path if path.is_absolute() else ROOT / path
        path = path.resolve()
        _require(path.is_file() and path.stat().st_size == entry['bytes'], 'file size: ' + entry['path'])
        if entry['bytes'] > 32 * 1024 * 1024 and not allow_large:
            self.large_metadata_only[entry['path']] = entry
            return path
        stamp = (entry['sha256'], path.stat().st_size, path.stat().st_mtime_ns)
        if self.checked.get(str(path)) != stamp:
            _require(_hash(path) == entry['sha256'], 'SHA256: ' + entry['path'])
            self.checked[str(path)] = stamp
            if self.budget is not None:
                self.budget.check()
        return path


def _npy_header(handle):
    _require(handle.read(6) == b'\x93NUMPY', 'NPY magic')
    version = tuple(handle.read(2))
    _require(version in ((1, 0), (2, 0), (3, 0)), 'NPY header version')
    width = 2 if version == (1, 0) else 4
    length = struct.unpack('<H' if width == 2 else '<I', handle.read(width))[0]
    _require(length <= 16384, 'NPY header unexpectedly large')
    return ast.literal_eval(handle.read(length).decode('utf-8' if version == (3, 0) else 'latin1'))


def _prediction_headers(path, expected_shape):
    with zipfile.ZipFile(path) as archive:
        _require(set(archive.namelist()) == {'prediction.npy', 'origins.npy'}, 'TEST archive members')
        with archive.open('prediction.npy') as handle:
            prediction = _npy_header(handle)
        with archive.open('origins.npy') as handle:
            origins = _npy_header(handle)
    _require(tuple(prediction['shape']) == expected_shape and prediction['descr'] == '<f4'
             and not prediction['fortran_order'], 'canonical TEST prediction shape/dtype')
    _require(tuple(origins['shape']) == (expected_shape[0],) and origins['descr'] == '<i8', 'TEST origin shape/dtype')
    return {'prediction_shape': list(prediction['shape']), 'prediction_dtype': prediction['descr'],
            'origins_shape': list(origins['shape']), 'array_payload_reloaded': False}


def _score_check(score, channels, message):
    counts = score['target_counts']
    _require(len(counts) == channels and all(isinstance(n, int) and n > 0 for n in counts), message + ' counts')
    for metric, sums_key, channel_key in (('mse', 'squared_error_sums', 'channel_mse'),
                                          ('mae', 'absolute_error_sums', 'channel_mae')):
        _require(math.isfinite(score[metric]) and score[metric] >= 0, message + ' finite loss')
        if sums_key not in score:
            continue
        sums = score[sums_key]
        _require(len(sums) == channels and all(math.isfinite(x) and x >= 0 for x in sums), message + ' error sums')
        ratios = [value / count for value, count in zip(sums, counts)]
        _close(score[metric], statistics.mean(ratios), message + ' macro ' + metric)
        if channel_key in score:
            _require(len(score[channel_key]) == channels, message + ' channel shape')
            for actual, expected in zip(score[channel_key], ratios):
                _close(actual, expected, message + ' channel ' + metric)
    if 'target_count' in score:
        _require(score['target_count'] == sum(counts), message + ' total observed targets')


def _pool_check(periods, channels, message):
    for period, score in periods.items():
        _score_check(score, channels, message + '/' + period)
    a, b, combined = (periods[key] for key in PERIODS)
    _require(combined['origins'] == a['origins'] + b['origins'], message + ' pooled origins')
    _require(combined['target_counts'] == [x + y for x, y in zip(a['target_counts'], b['target_counts'])], message + ' pooled counts')
    for key in ('squared_error_sums', 'absolute_error_sums'):
        if key in combined and key in a and key in b:
            for actual, x, y in zip(combined[key], a[key], b[key]):
                _close(actual, x + y, message + ' pooled ' + key)


def _fits_and_selection(selection, seal, fit_summary, receipts):
    _require(selection['choices'] == seal['choices'], 'joint selection choices')
    _require(seal['all_trainval_selection_complete_before_test'] and seal['global_choice_sealed_before_test']
             and seal['test_predictions_at_sealing'] == 0 and seal['selected_models'] == 12,
             'joint pre-TEST seal and zero TEST counter')
    _require(selection['fits_considered'] == 24 and selection['selected_models'] == 12
             and selection['seed_prediction_ensemble'] is False
             and selection['all_learning_rate_and_checkpoint_choices_frozen'], 'selection contract')
    for entry in seal['artifacts']:
        receipts.verify(entry)
    runs, selected, normality = {}, [], []
    schedule_groups = {}
    for dataset in DATASETS:
        _require(set(selection['choices'][dataset]) == set(ARMS), 'all dataset arms selected')
        for arm in ARMS:
            choice = selection['choices'][dataset][arm]
            _require([row['learning_rate'] for row in choice['candidates']] == [1e-3, 1e-4], 'fixed LR order')
            means = []
            for candidate in choice['candidates']:
                _require([row['seed'] for row in candidate['runs']] == list(SEEDS), 'fixed two seeds')
                losses = []
                for member in candidate['runs']:
                    result = _read(receipts.verify(member['result']))
                    spec = result['spec']
                    _require(result['id'] == member['id'] and result['id'] not in runs
                             and result['status'] == 'complete' and result['test_access'] is False,
                             'complete distinct TRAIN/VAL fit')
                    _require((spec['dataset'], spec['arm'], spec['seed'], spec['learning_rate']) ==
                             (dataset, arm, member['seed'], candidate['learning_rate']), 'fit identity')
                    _require((spec['maximum_epochs'], spec['epoch_origins'], spec['effective_batch'], spec['patience']) ==
                             (120, 512, 4, 6), 'fixed fit opportunities')
                    _require(result['checkpoint'] == member['checkpoint'] and result['best_val_prediction'] == member['best_val_prediction'], 'selected artifact pointers')
                    receipts.verify(result['checkpoint'])
                    receipts.verify(result['best_val_prediction'])
                    curve = _read(receipts.verify(result['curve']))
                    _require([row['epoch'] for row in curve] == list(range(result['epochs_completed'] + 1)), 'complete epoch0..last curve')
                    winner = min(curve, key=lambda row: row['val_mse'])
                    _require(winner['epoch'] == result['selected_epoch'] == member['selected_epoch'], 'strict minimum, earliest exact tie')
                    _close(winner['val_mse'], result['best_val_mse'], 'VAL minimum')
                    _close(result['best_val_mse'], member['best_val_mse'], 'selection VAL loss')
                    _close(curve[0]['val_mse'], result['initial_val_mse'], 'epoch0 VAL')
                    _require(result['scientific_updates'] == result['epochs_completed'] * 128, '128 updates per completed epoch')
                    _require(1 <= result['epochs_completed'] <= 120 and
                             ((result['termination'] == 'PATIENCE' and result['final_stale'] >= 6)
                              or (result['termination'] == 'MAX_EPOCH' and result['epochs_completed'] == 120)),
                             'normal patience versus maximum-epoch termination')
                    running_best, best_epoch = float('inf'), None
                    for row in curve:
                        _require(row['steps'] == row['epoch'] * 128 and row['scheduler_observed'] == (row['epoch'] >= 1), 'scheduler excludes epoch0')
                        _require(math.isfinite(row['val_mse']) and math.isfinite(row['val_mae']), 'finite full VAL curve')
                        if row['val_mse'] < running_best:
                            running_best, best_epoch = row['val_mse'], row['epoch']
                        _require(row['best_epoch'] == best_epoch, 'curve strict-minimum history')
                        if row['epoch'] >= 1:
                            _require(all(math.isfinite(row[key]) for key in ('sampled_train_effective_batch_loss_mean',
                                     'sampled_train_effective_batch_loss_max', 'gradient_norm_mean_before_clip',
                                     'gradient_norm_max_before_clip')), 'finite sampled TRAIN/gradient')
                    _require(result['frozen_unchanged'] and result['frozen_parameters_and_all_buffers_hash']
                             and result['selected_replay']['maximum_absolute_error'] <= 1e-6, 'frozen state and selected VAL replay')
                    _require(result['scientific_updates'] > 0 and any(x > 0 for x in result['parameter_max_updates'].values()), 'actual scientific readout update')
                    _require(result['first_gradients'] and all(row['finite'] for row in result['first_gradients'].values()), 'finite trainable gradients')
                    expected_scalars = 2526336 if arm == 'NATIVE_HEAD' else {'robin': 306, 'peacock_education': 182, 'jena': 462}[dataset]
                    _require(result['trainable_scalars'] == expected_scalars, 'native/output parameter enumeration')
                    schedule = schedule_groups.setdefault((dataset, member['seed']), {})
                    for row in result['schedules']:
                        if row['epoch'] in schedule:
                            _require(schedule[row['epoch']] == row['sha256'], 'same arm/LR sample schedule')
                        schedule[row['epoch']] = row['sha256']
                    runs[result['id']] = result
                    losses.append(result['best_val_mse'])
                    trained = curve[1:]
                    normality.append({'id': result['id'], 'dataset': dataset, 'arm': arm,
                        'seed': member['seed'], 'learning_rate': candidate['learning_rate'],
                        'selected_epoch': result['selected_epoch'], 'epochs_completed': result['epochs_completed'],
                        'termination': result['termination'], 'normality': result['normality'],
                        'oom_events': len(result['oom_events']), 'selected_val_mse': result['best_val_mse'],
                        'initial_val_mse': result['initial_val_mse'], 'last_val_mse': curve[-1]['val_mse'],
                        'first_sampled_train_loss': trained[0]['sampled_train_effective_batch_loss_mean'],
                        'last_sampled_train_loss': trained[-1]['sampled_train_effective_batch_loss_mean'],
                        'max_gradient_norm_before_clip': max(row['gradient_norm_max_before_clip'] for row in trained),
                        'sampled_train_down_last_val_above_selected':
                        trained[-1]['sampled_train_effective_batch_loss_mean'] < trained[0]['sampled_train_effective_batch_loss_mean']
                        and curve[-1]['val_mse'] > result['best_val_mse']})
                mean = statistics.mean(losses)
                _close(mean, candidate['mean_seed_val_mse'], 'two-seed VAL loss mean')
                means.append(mean)
            chosen = min(range(2), key=lambda index: means[index])
            _require(choice['lr'] == choice['candidates'][chosen]['learning_rate']
                     and choice['runs'] == choice['candidates'][chosen]['runs'], 'LR selection; exact tie first-listed 1e-3')
            _close(choice['mean_seed_val_mse'], means[chosen], 'chosen LR mean')
            selected.extend(dict(member, dataset=dataset, arm=arm) for member in choice['runs'])
    _require(len(runs) == 24 and len(selected) == 12 and fit_summary['status'] == 'complete'
             and fit_summary['scientific_fits'] == 24 and fit_summary['test_access'] is False, '24 fits / 12 selected models')
    updates = sum(row['scientific_updates'] for row in runs.values())
    _require(updates == fit_summary['scientific_updates'], 'scientific update sum')
    return runs, selected, normality, updates


def _evaluation(evaluation, manifest, exposure, seal, selection_receipt, selected, contracts, receipts):
    _require(evaluation['status'] == manifest['status'] == 'complete'
             and evaluation['selected_test_models'] == 12, 'complete joint evaluation')
    ids = {row['id'] for row in selected}
    _require(set(manifest['models']) == set(evaluation['models']) == ids, 'exact twelve TEST model instances')
    _require(exposure['selection_seal'] == selection_receipt and exposure['all_trainval_selection_complete_before_test']
             and exposure['started_utc'] >= seal['sealed_utc'], 'seal precedes TEST exposure')
    _require(evaluation['binding'] == manifest['binding'] and manifest['binding']['selection_seal'] == selection_receipt, 'evaluation binding')
    for entry in evaluation['binding'].values():
        receipts.verify(entry)
    receipts.verify(evaluation['prediction_manifest'])
    for entry in evaluation['tables'].values():
        receipts.verify(entry)
    expected_bootstrap = _read(HERE / 'evaluation_protocol.json')['bootstrap']
    _require(expected_bootstrap['draws'] == 2000 and expected_bootstrap['seed'] == 9262026
             and expected_bootstrap['blocks'] == {'robin': 7, 'peacock_education': 7, 'jena': 42}
             and expected_bootstrap['dataset_salts'] == {'robin': 0, 'peacock_education': 0, 'jena': 1},
             'prespecified bootstrap numeric settings')
    _require(all(evaluation['bootstrap'][key] == value for key, value in expected_bootstrap.items()), 'actual bootstrap2000 / RNG / blocks / salts')
    _require(evaluation['bootstrap']['selection_uncertainty_included'] is False
             and evaluation['bootstrap']['period_boundaries_crossed'] is False
             and evaluation['independent_confirmation'] is False, 'conditional temporal evidence scope')
    headers = {}
    for row in selected:
        record, scored = manifest['models'][row['id']], evaluation['models'][row['id']]
        _require(all(scored[key] == value for key, value in record.items()), 'manifest/scoring record')
        _require(record['checkpoint'] == row['checkpoint'] and record['selection_seal'] == selection_receipt
                 and record['selected_epoch'] == row['selected_epoch'] and record['learning_rate'] == row['learning_rate']
                 and record['new_test_model_instances'] == 1 and record['frozen_parameters_and_all_buffers_unchanged'], 'selected TEST identity/current restored frozen state')
        provenance = evaluation['datasets'][row['dataset']]['scoring_provenance']
        _require(record['scoring_provenance'] == provenance and record['origins_sha256'] == provenance['origins_sha256'], 'canonical TEST provenance')
        counts = evaluation['datasets'][row['dataset']]['period_origin_counts']
        shape = (counts['combined'], 48, contracts[row['dataset']]['c'])
        headers[row['id']] = _prediction_headers(receipts.verify(record['prediction']), shape)
        _pool_check(scored['periods'], shape[-1], row['id'])
    for dataset in DATASETS:
        accuracy = evaluation['accuracy'][dataset]
        channels = contracts[dataset]['c']
        for method, periods in accuracy.items():
            _pool_check(periods, channels, dataset + '/' + method)
            for period in PERIODS:
                _require(periods[period]['target_counts'] == accuracy['F0'][period]['target_counts']
                         and periods[period]['origins'] == accuracy['F0'][period]['origins'],
                         'identical canonical targets across methods')
        for arm in ARMS:
            members = [evaluation['models'][row['id']]['periods'] for row in selected
                       if row['dataset'] == dataset and row['arm'] == arm]
            for period in PERIODS:
                score = accuracy[arm][period]
                _require(score['target_counts'] == members[0][period]['target_counts'] == members[1][period]['target_counts'], 'paired seed target counts')
                for metric in ('mse', 'mae'):
                    _close(score[metric], statistics.mean(member[period][metric] for member in members), 'arithmetic mean seed losses, no forecast ensemble')
                for key in ('squared_error_sums', 'absolute_error_sums', 'channel_mse', 'channel_mae'):
                    for actual, a, b in zip(score[key], members[0][period][key], members[1][period][key]):
                        _close(actual, (a + b) / 2, 'mean seed sufficient statistics')
        _require(set(evaluation['comparisons'][dataset]) == {left + '__vs__' + right for left, right in PAIRS}, 'seven prespecified paired comparisons')
        for comparison in evaluation['comparisons'][dataset].values():
            for period, metrics in comparison['periods'].items():
                for metric, values in metrics.items():
                    candidate = accuracy[comparison['candidate']][period][metric]
                    reference = accuracy[comparison['reference']][period][metric]
                    _close(values['candidate'], candidate, 'paired candidate loss')
                    _close(values['reference'], reference, 'paired reference loss')
                    _close(values['absolute_difference'], candidate - reference, 'paired absolute difference')
                    _close(values['relative_change_percent'], 100 * (candidate - reference) / reference, 'relative change exact reference denominator')
                    for key in ('conditional_block95_absolute', 'conditional_block95_relative_percent'):
                        limits = values[key]
                        _require(len(limits) == 2 and all(math.isfinite(x) for x in limits) and limits[0] <= limits[1], 'bootstrap percentile interval')
    return headers


def _costs(cost, receipts):
    _require(cost['status'] == 'complete' and cost['test_access'] is False
             and cost['historical_costs_used_in_speed_ratios'] is False, 'complete same-campaign costs')
    receipts.verify(cost['protocol'])
    receipts.verify(cost['selection_seal'])
    training, deployment, sentinels = cost['training_rows'], cost['deployment_rows'], cost['sentinels']
    _require((len(training), len(deployment), len(sentinels)) == (108, 162, 18), '108 train / 162 deploy / 18 sentinel pairs')
    _require(len({(row['id'], row['path'], row['block']) for row in training}) == 108, 'unique training grid')
    _require(len({(row['id'], row['batch'], row['block']) for row in deployment}) == 162, 'unique deployment grid')
    _require({(row['dataset'], row['batch'], row['block']) for row in sentinels} ==
             {(dataset, batch, block) for dataset in DATASETS for batch in (1, 4) for block in range(3)},
             'exact sentinel dataset/batch/block grid')
    for dataset in DATASETS:
        for method in ('OUTPUT_AFFINE', 'NATIVE_HEAD', 'MSE_LORA', 'LEVEL'):
            for path in ('online', 'cached') if method in ARMS else ('online',):
                values = [row for row in training if (row['dataset'], row['method'], row['path']) == (dataset, method, path)]
                _require({(row['seed'], row['block']) for row in values} ==
                         {(seed, block) for seed in SEEDS for block in range(3)} and len(values) == 6,
                         'exact selected-seed training grid')
        for method in ('F0', 'OUTPUT_AFFINE', 'NATIVE_HEAD', 'MSE_LORA', 'LEVEL'):
            values = [row for row in deployment if row['dataset'] == dataset and row['method'] == method]
            _require(len(values) == (6 if method == 'F0' else 12), 'exact deployment method instance count')
            if method != 'F0':
                _require({(row['seed'], row['batch'], row['block']) for row in values} ==
                         {(seed, batch, block) for seed in SEEDS for batch in (1, 4) for block in range(3)},
                         'exact selected-seed deployment grid')
    for row in training:
        _require(row['status'] == 'complete' and row['effective_batch'] == 4
                 and row['warmup_updates'] == row['measured_updates'] == 10
                 and len(row['step_seconds']) == 10 and row['adam_state_bytes'] > 0
                 and row['frozen_unchanged'] and row['selected_checkpoint_unchanged'], 'full warmup/Adam/step training receipt')
        _require(all(math.isfinite(x) and x > 0 for x in row['step_seconds']), 'finite measured training times')
        _close(row['median_step_seconds'], statistics.median(row['step_seconds']), 'training block median')
        receipts.verify(row['selected_checkpoint'])
        _require(row['selected_checkpoint_sha256_before'] == row['selected_checkpoint_sha256_after'] == row['selected_checkpoint']['sha256'], 'training selected checkpoint preservation')
    updates = sum(row['warmup_updates'] + row['measured_updates'] for row in training)
    _require(updates == cost['technical_updates'] == 2160
             and cost['new_technical_updates'] == 1440 and cost['old_technical_updates'] == 720, 'exact technical updates')
    before, after = cost['budget_before']['counters'], cost['budget_after']['counters']
    _require(after['technical_updates'] - before['technical_updates'] == 2160, 'probe update ledger delta')
    _require(all(after[key] == before[key] for key in ('scientific_fits', 'scientific_updates', 'new_test_predictions')), 'probe does not change scientific/TEST counts')
    for row in deployment + [item[phase] for item in sentinels for phase in ('before', 'after')]:
        _require(row['status'] == 'complete' and row['batch'] in (1, 4) and row['origins'] == 24
                 and row['complete_origin_passes'] == row['warmup_wrapper_calls'] == 10
                 and len(row['complete_pass_seconds']) == 10 and row['feature_cache_used'] is False
                 and row['frozen_unchanged'], 'full-model CPU input/output deployment receipt')
        _require(all(math.isfinite(x) and x > 0 for x in row['complete_pass_seconds']), 'finite deployment times')
        _close(row['median_seconds_per_origin'], statistics.median(row['complete_pass_seconds']) / 24, 'deployment latency per origin')
        _close(row['origins_per_second'], 1 / row['median_seconds_per_origin'], 'deployment throughput')
    for pair in sentinels:
        _require(pair['status'] == 'complete' and pair['before']['method'] == pair['after']['method'] == 'F0', 'before/after F0 sentinel')
        _close(pair['after_over_before_latency'], pair['after']['median_seconds_per_origin'] / pair['before']['median_seconds_per_origin'], 'raw sentinel drift ratio')
    for entry in cost['checkpoint_preservation']:
        receipts.verify(entry['checkpoint'])
        _require(entry['unchanged'] and entry['before_sha256'] == entry['after_sha256'] == entry['checkpoint']['sha256'], 'current preserved selected checkpoint hash')
    _require(cost['selected_checkpoints_unchanged'], 'all probe selected checkpoints preserved')
    for summary_key, rows, condition, value_key in (('training_summary', training, 'path', 'median_step_seconds'),
                                                   ('deployment_summary', deployment, 'batch', 'median_seconds_per_origin')):
        summary = cost[summary_key]
        for instance in summary['instances']:
            blocks = [row[value_key] for row in rows if row['id'] == instance['id'] and row[condition] == instance[condition]]
            _require(instance['status'] == 'complete' and len(blocks) == instance['complete_blocks'] == 3, 'three cost blocks per selected instance')
            _close(instance[value_key], statistics.median(blocks), 'selected-instance block median')
        for method in summary['methods']:
            instances = [row[value_key] for row in summary['instances'] if row['dataset'] == method['dataset']
                         and row['method'] == method['method'] and row[condition] == method[condition]]
            _require(method['status'] == 'complete' and len(instances) == method['selected_seed_instances'] == (1 if method['method'] == 'F0' else 2), 'cost selected seed mean')
            _close(method['mean_selected_seed_median_seconds'], statistics.mean(instances), 'mean seed cost medians')
    return {'technical_updates': updates, 'training_rows': len(training), 'deployment_rows': len(deployment),
            'sentinel_pairs': len(sentinels), 'sentinel_latency_ratio_range':
            [min(row['after_over_before_latency'] for row in sentinels), max(row['after_over_before_latency'] for row in sentinels)]}


def _resource_usage(state, ledger, limits, budget):
    snapshot = budget.check() if budget is not None else state['budget']
    _require(all(row['status'] == 'complete' for row in ledger['jobs'] if row['category'] == 'gpu'), 'completed GPU jobs')
    counters = ledger['counters']
    _require(counters == snapshot['counters'], 'resource snapshot counters')
    peaks = ledger['peaks']
    usage = {'gpu_job_wall_seconds': sum(row['elapsed_wall_seconds'] for row in ledger['jobs'] if row['category'] == 'gpu'),
             'cpu_only_job_wall_seconds': snapshot['cpu_wall_seconds'],
             'cpu_process_seconds': snapshot['cpu_process_seconds'],
             'operation_elapsed_seconds': snapshot['operation_elapsed_seconds'],
             'new_disk_bytes': snapshot['new_disk_bytes'],
             'process_ram_peak_bytes': peaks['rss_bytes'],
             'gpu_allocated_peak_bytes': peaks['gpu_allocated_bytes'],
             'gpu_reserved_peak_bytes': peaks['gpu_reserved_bytes'], **counters}
    mapping = {'gpu_job_wall_seconds': 'maximum_gpu_job_wall_seconds',
               'cpu_only_job_wall_seconds': 'maximum_cpu_only_job_wall_seconds',
               'cpu_process_seconds': 'maximum_cpu_process_time_seconds',
               'operation_elapsed_seconds': 'maximum_elapsed_operation_wall_seconds',
               'new_disk_bytes': 'maximum_new_disk_bytes', 'process_ram_peak_bytes': 'maximum_process_ram_bytes',
               'gpu_allocated_peak_bytes': 'maximum_gpu_allocated_or_reserved_bytes',
               'gpu_reserved_peak_bytes': 'maximum_gpu_allocated_or_reserved_bytes',
               'technical_updates': 'maximum_technical_optimizer_updates', 'prefix_calls': 'maximum_full_prefix_batch_calls',
               'head_calls': 'maximum_head_or_output_only_batch_calls', 'scientific_fits': 'maximum_scientific_fits',
               'scientific_updates': 'maximum_scientific_optimizer_updates', 'new_test_predictions': 'maximum_new_test_predictions'}
    checks = {}
    for key, cap_key in mapping.items():
        _require(usage[key] <= limits[cap_key], 'resource cap: ' + key)
        checks[key] = {'actual': usage[key], 'cap': limits[cap_key], 'fraction_of_cap': usage[key] / limits[cap_key]}
    return usage, checks


def _s2_proposal(evaluation, cost, contracts, selection, normality):
    rows = []
    update_seconds, val_seconds, probe_step_seconds = 0., 0., 0.
    updates_per_fit = 120 * (512 // 4)
    for dataset in DATASETS:
        def maximum(method, field, deployment=False):
            candidates = [row[field] for row in cost['deployment_rows' if deployment else 'training_rows']
                          if row['dataset'] == dataset and row['method'] == method
                          and (row['batch'] == 4 if deployment else row['path'] == 'online')]
            return max(candidates)
        head = maximum('NATIVE_HEAD', 'median_step_seconds')
        lora = maximum('MSE_LORA', 'median_step_seconds')
        head_val = maximum('NATIVE_HEAD', 'median_seconds_per_origin', True)
        lora_val = maximum('MSE_LORA', 'median_seconds_per_origin', True)
        source = _read(ROOT / contracts[dataset]['source']['path'])
        val_origins = len(source['origins']['val']['values'])
        updates = 2 * updates_per_fit * (2 * head + lora)
        validation = 2 * 121 * val_origins * (2 * head_val + lora_val)
        update_seconds += updates
        val_seconds += validation
        probe_step_seconds += 2 * 20 * (2 * head + lora)
        training = [row for row in cost['training_rows'] if row['dataset'] == dataset and row['path'] == 'online']
        peak_head = max(row['peak_reserved_bytes_after_adam'] for row in training if row['method'] == 'NATIVE_HEAD')
        peak_lora = max(row['peak_reserved_bytes_after_adam'] for row in training if row['method'] == 'MSE_LORA')
        head_loss = evaluation['accuracy'][dataset]['NATIVE_HEAD']['combined']['mse']
        lora_loss = evaluation['accuracy'][dataset]['MSE_LORA']['combined']['mse']
        parents = selection['choices'][dataset]['NATIVE_HEAD']['runs']
        parent_ids = {row['id'] for row in parents}
        parent_learning = [row for row in normality if row['id'] in parent_ids]
        initial_val = statistics.mean(row['initial_val_mse'] for row in parent_learning)
        selected_val = selection['choices'][dataset]['NATIVE_HEAD']['mean_seed_val_mse']
        rows.append({'dataset': dataset, 'historical_lora_gap_percent': 100 * (head_loss - lora_loss) / lora_loss,
                     'sealed_initial_head_val_mse': initial_val, 'sealed_selected_head_val_mse': selected_val,
                     'sealed_head_val_reduction_percent': 100 * (initial_val - selected_val) / initial_val,
                     'selected_parent_epochs': [row['selected_epoch'] for row in parents],
                     'validated_learning_state': [row['normality'] for row in parent_learning],
                     'head_online_slowest_block_step_seconds': head, 'lora_online_slowest_block_step_seconds': lora,
                     'head_plus_lora_step_estimate_seconds': head + lora,
                     'head_reserved_peak_bytes_measured': peak_head, 'lora_reserved_peak_bytes_measured': peak_lora,
                     'unmeasured_joint_separate_peak_sum_bytes': peak_head + peak_lora,
                     'maximum_update_projection_seconds': updates, 'maximum_full_val_projection_seconds': validation})
    worthwhile = any(row['sealed_head_val_reduction_percent'] > 0 for row in rows)
    return {'recommendation': 'PROPOSE' if worthwhile else 'DEFER', 'authorized': False, 'executed': False,
            'decision_scope': 'post-TEST exploratory development decision; all three datasets retained; no protected confirmation',
            'post_test_development_decision': True,
            'primary_evidence': 'sealed HEAD VAL changes, independently checked finite gradients/frozen-state learning, and same-campaign online time/Adam-state memory; sufficient optimization remains unproved',
            'secondary_evidence': 'historical TEST LoRA gap is unmatched-parent context only, not the criterion for selecting datasets or measuring internal adaptation gain',
            'reason': 'validated HEAD fitting changes sealed VAL loss at measured costs, while equal extra head training versus internal adaptation remains unresolved'
                      if worthwhile else 'sealed HEAD VAL shows no improvement; optimization and continuation value require review before a new scientific fit',
            'scientific_fits': 12, 'arms': ['HEAD_CONTINUE', 'HEAD_PLUS_LORA'], 'datasets': list(DATASETS),
            'seeds': list(SEEDS), 'learning_rates_per_arm': 1,
            'learning_rate_rule': 'use each dataset selected S1 HEAD learning rate for both arms; no LR search',
            'parent_rule': 'identical selected HEAD checkpoint per dataset/seed; zero-initialized LoRA; same head weights and update schedule',
            'maximum_extra_epochs': 120, 'maximum_updates_per_fit': updates_per_fit,
            'maximum_scientific_updates': 12 * updates_per_fit,
            'two_learning_rates_alternative': {'requires_separate_scope_and_budget': True, 'scientific_fits': 24,
                                             'maximum_scientific_updates': 24 * updates_per_fit},
            'dataset_estimates': rows, 'maximum_update_projection_seconds': update_seconds,
            'maximum_full_val_projection_seconds': val_seconds,
            'raw_fit_plus_val_projection_seconds': update_seconds + val_seconds,
            'illustrative_twofold_fit_val_allowance_seconds': 2 * (update_seconds + val_seconds),
            'estimate_is_approved_cap': False,
            'projection_limitations': 'sum of separately measured HEAD and MSE-LoRA costs is a conservative planning approximation, not a measured joint cost or rigorous bound; restore, checkpoint/guard I/O, TEST, deployment and reporting must be budgeted after the initial joint probe',
            'joint_trainable_scalars': 2526336 + 294912,
            'joint_fp32_weights_grad_adam_payload_bytes': (2526336 + 294912) * 16,
            'separate_initial_probe_proposal': {'maximum_technical_updates': 264,
                'basis': '12 disposable instances x (warmup10 + measured10) plus integrity reserve24',
                'raw_step_projection_seconds': probe_step_seconds,
                'proposed_gpu_job_wall_cap_seconds': max(900, math.ceil(2 * probe_step_seconds + 600)),
                'proposed_process_ram_cap_bytes': 8 * 1024 ** 3,
                'proposed_gpu_allocated_or_reserved_cap_bytes': 8 * 1024 ** 3,
                'scope': 'joint trainables/gradients/zero-update function and actual Adam-state peak; no scientific fit or TEST; separate approval required'},
            'memory_status': 'joint activation/Adam/allocator peak unmeasured; separated measured peak sum is a planning reference only; fail the initial probe rather than silently raise caps',
            'same_update_is_same_time': False, 'automatic_s2_execution': False}


def _report(summary, selection, evaluation, cost, normality, s0, fixture, preservation):
    text = ['# 원채널 출력·native head 적응 S1 결과', '',
        '[확인] S1 계산은 완료되었고 선택·채점·비용·자원 검산은 PASS다. 이 표시는 독립 확증, 내부 적응의 필수성, 새 방법의 신규성을 뜻하지 않는다. '
        '이번 질문은 추가 모듈을 제안하기 전에 표준 출력 보정과 원래 헤드 적응만으로 남은 오차가 얼마나 줄어드는지 확인하는 것이다.', '',
        'OUTPUT_AFFINE은 F0의 원채널 예측에 C→C affine을 적용하며 채널을 섞는다. NATIVE_HEAD는 동결 encode/Transformer decode 표현을 읽는 '
        '원래 비선형 output_patch_embedding 전체를 적응한다. HEAD는 가중치를 공유하는 채널독립 함수다. '
        '두 경로의 차이에는 표현·용량·채널 결합 구조가 함께 들어 있으므로 순수 정보 효과로 해석할 수 없다. '
        'HEAD의 2,526,336개 학습 변수는 기존 MSE-LoRA의 294,912개보다 많고, 직접 point loss에 연결되는 계산상 scalars는 1,173,600개다.', '',
        '[확인] 사전 S0는 모든 canonical VAL에서 초기 OUTPUT/HEAD=F0, online/cache 값·loss·gradient, '
        '결측 effective B4와 micro4/2/1의 분모·gradient, original normalization/inverse, frozen 모든 buffers와 '
        'Adam/scheduler/RNG 복원을 검증했다. 이 정상성 검증은 HEAD를 충분히 최적화했다는 보장이 아니다.', '',
        '[확인] 정확도는 기존 TRAIN 표준화 좌표와 native inverse normalization 뒤의 관측 target/channel-macro 손실이다. '
        '두 seed의 손실을 평균했고 예측 ensemble을 만들지 않았다. TEST-A/B combined는 채널별 error sums와 관측 count를 합친 뒤 ratio와 채널 평균을 계산했다. '
        'MSE/MAE는 낮을수록 좋다. 아래 F0 대비 감소율은 100×(F0−후보)/F0이며 양수는 개선, 음수는 악화다.', '', '```text',
        '자료                F0 MSE     OUTPUT MSE  HEAD MSE    OUTPUT 감소%  HEAD 감소%',
        *[f"{dataset:19} {summary['datasets'][dataset]['F0']['mse']:.6f}   {summary['datasets'][dataset]['OUTPUT_AFFINE']['mse']:.6f}    {summary['datasets'][dataset]['NATIVE_HEAD']['mse']:.6f}    {summary['datasets'][dataset]['OUTPUT_AFFINE']['f0_reduction_percent']:+.3f}        {summary['datasets'][dataset]['NATIVE_HEAD']['f0_reduction_percent']:+.3f}" for dataset in DATASETS],
        '```', '',
        '두 기간과 seed의 변동은 다음과 같다. 대괄호는 같은 LR로 선택된 두 seed 손실의 min/max이며 신뢰구간이 아니다.', '', '```text',
        '자료/경로                             A MSE [seed 범위]              B MSE [seed 범위]              combined MAE',
        *[f"{dataset + '/' + arm:37} {summary['datasets'][dataset][arm]['periods']['test_a']['mse']:.6f} [{summary['datasets'][dataset][arm]['seed_ranges']['test_a'][0]:.6f}, {summary['datasets'][dataset][arm]['seed_ranges']['test_a'][1]:.6f}]  {summary['datasets'][dataset][arm]['periods']['test_b']['mse']:.6f} [{summary['datasets'][dataset][arm]['seed_ranges']['test_b'][0]:.6f}, {summary['datasets'][dataset][arm]['seed_ranges']['test_b'][1]:.6f}]  {summary['datasets'][dataset][arm]['mae']:.6f}" for dataset in DATASETS for arm in ARMS],
        '```', '',
        '[확인] OUTPUT의 combined MAE가 F0보다 높은 자료는 '
        + ', '.join(dataset for dataset in DATASETS if summary['datasets'][dataset]['OUTPUT_AFFINE']['mae'] > summary['datasets'][dataset]['F0']['mae'])
        + '다. MSE 감소를 MAE 개선으로 확대하지 않는다. Jena OUTPUT의 combined MAE는 '
        + f"{summary['datasets']['jena']['OUTPUT_AFFINE']['mae']:.6f}, F0는 {summary['datasets']['jena']['F0']['mae']:.6f}이다.", '',
        '같은 원점·채널·seed를 보존한 paired 차이의 combined MSE 조건부 구간이다. 상대변화는 100×(후보−참조)/참조이며 음수는 후보 개선이다.', '', '```text',
        '자료/비교                                      상대변화%     paired block95 상대변화%',
        *[f"{dataset + '/' + key:47} {values['periods']['combined']['mse']['relative_change_percent']:+.3f}       [{values['periods']['combined']['mse']['conditional_block95_relative_percent'][0]:+.3f}, {values['periods']['combined']['mse']['conditional_block95_relative_percent'][1]:+.3f}]" for dataset in DATASETS for key, values in evaluation['comparisons'][dataset].items()],
        '```', '',
        'bootstrap은 2,000회, seed9262026, Robin/Peacock block7·Jena block42, salt0/0/1이다. A/B를 나눈 noncircular contiguous moving blocks에서 '
        'ceil(n/block)개의 시작점을 추출하고 n까지 잘랐으며 combined는 두 기간의 occurrence weights를 더했다. '
        '구간은 이미 선택된 두 seed와 노출된 개발 target에 조건부인 시간 재표집 불확실성이다. LR/모델선택 불확실성과 독립 도메인 일반화를 포함하지 않으며 0을 포함해도 동등성은 확인되지 않는다.', '',
        '기존 저장 결과는 재적합·재선택 없이 참조했다. MSE-LoRA·LEVEL·NLinear·Mini/Tiny·Chronos-2의 같은 canonical target 정확도는 다음과 같다. '
        '기존 LoRA와의 차이는 같은 HEAD 부모에서 내부 적응을 추가한 효과가 아니다.', '', '```text',
        '자료                참조                         combined MSE  combined MAE',
        *[f"{dataset:19} {method:28} {scores['combined']['mse']:.6f}      {scores['combined']['mae']:.6f}" for dataset in DATASETS for method, scores in evaluation['accuracy'][dataset].items() if method in ('MSE_LORA', 'LEVEL', 'NATIVE_LORA_QUANTILE', 'DIRECT_NLINEAR', 'BOLT_MINI_F0', 'BOLT_TINY_F0', 'C2_MV', 'C2_SMALL_MV', 'C2_UNI', 'C2_SMALL_UNI')],
        '```', '',
        f"[확인] 24 fits의 실제 scientific updates 합은 {summary['counters']['scientific_updates']:,}이고, 12개 선택 모델을 TEST에 평가했다. "
        'epoch0을 포함한 full-VAL strict minimum과 earliest exact tie를 재검산했고 LR은 두 seed VAL 최소손실 평균으로 선택했다. '
        'LR exact tie는 first-listed1e-3이며 scheduler는 epoch1 이후에만 관측했다. 선택 seal은 첫 TEST 접근보다 앞서고 당시 신규 TEST counter는0이었다.', '', '```text',
        '자료/경로                             선택 LR    선택 epoch(seed1/2)  전체 fit epoch0선택/상한도달/OOM',
        *[f"{dataset + '/' + arm:37} {selection['choices'][dataset][arm]['lr']:.0e}      {'/'.join(str(row['selected_epoch']) for row in selection['choices'][dataset][arm]['runs']):18} {sum(row['selected_epoch'] == 0 for row in normality if row['dataset'] == dataset and row['arm'] == arm)}/{sum(row['termination'] == 'MAX_EPOCH' for row in normality if row['dataset'] == dataset and row['arm'] == arm)}/{sum(row['oom_events'] for row in normality if row['dataset'] == dataset and row['arm'] == arm)}" for dataset in DATASETS for arm in ARMS],
        '```', '',
        "[확인] 모든 fit의 TRAIN batch loss/gradient와 full VAL은 유한했고 frozen parameters 및 모든 buffers의 불변 기록과 현재 선택 checkpoint 해시가 일치했다. "
        f"epoch0 선택은 {summary['normality']['epoch0_selected_fits']}개, 최대epoch 종료는 {summary['normality']['max_epoch_fits']}개, 기록된 OOM events는 {summary['normality']['oom_events']}개다. "
        f"sampled TRAIN loss가 처음보다 낮아졌으나 마지막 VAL이 선택 최소보다 높은 곡선은 {summary['normality']['sampled_train_down_last_val_above_selected_fits']}개다. "
        '이 TRAIN 수치는 epoch별 표본 batch 평균이며 full canonical TRAIN 점수와 같지 않다. 정상 epoch0 선택·정상 악화·상한 종료는 기술 실패나 자원중단과 구분한다.', '',
        '[확인] 적응비용과 배포비용을 분리했다. 다음 항목은 서로 중복 합산하지 않았고 모든 LR/seed 적합의 VAL·checkpoint I/O를 포함했다. '
        'cache 준비의 generation/write와 fit 내부 TRAIN/VAL/checkpoint 시간은 구성 설명용이며 해당 부모 elapsed에 다시 더하지 않았다.', '', '```text',
        *[f"{key:57} {value:.3f} s" for key, value in summary['time_accounting'].items() if isinstance(value, (float, int))],
        '```', '',
        '전체 S1 GPU job wall은 인터프리터 시작부터의 포괄 비용이다. 남은 차액에는 fit 바깥 cache reads, canonical 자료·참조 재검증, '
        'bootstrap/채점, source/checkpoint hash와 guard I/O가 들어 있으므로 모든 비용이 포함되며 특정 항목의 단가로 임의 배정하지 않았다. '
        'S0 native calibration과 계측 수정의 시간은 별도 과거 개발비다. 원래 느린 guard를 포함한20-update 측정과 수정 후6개 single-update 진단을 모두 보존했고 '
        'single-update를 steady-state20-update 측정으로 바꾸지 않았다. 이 과거 비용으로 S1 속도비를 만들지 않았다.', '',
        f"[확인] S0 native technical updates는 {s0['counters']['technical_updates']}개다. 별도 CPU 구현 fixture는 toy optimizer40 updates·GPU/모델/자료 로드0·wall {fixture['shell_wall_seconds']:.3f}초였고 실제 CPU process time은 미측정이다. "
        '그 fixture를 native calibration이나 scientific update에 섞지 않았다.', '',
        '[확인] 같은 회차 training probe는 선택 checkpoint의 폐기용 사본에서 B4, warmup10+measured10,3blocks를 측정했다. '
        '108개 행·2,160 technical updates(신규1,440+기존720), Adam moments 생성 뒤 memory와 full optimizer step을 확인했고 선택 파일 pre/post/current hash는 같았다. '
        '아래 시간은 seed별3block median의 평균이며 block range는 CI가 아니다.', '', '```text',
        '자료/경로/path                                  seconds/update',
        *[f"{row['dataset'] + '/' + row['method'] + '/' + row['path']:48} {row['mean_selected_seed_median_seconds']:.6f}" for row in cost['training_summary']['methods']],
        '```', '',
        'Adam 상태와 peak memory는 같은 회차 실제 행에서 경로별 최댓값을 별도로 보존했다. 아래 allocated/reserved는 Adam warmup 이후 peak이며 '
        'weights/gradient/optimizer의 계산상 payload와 다르다.', '', '```text',
        '자료/경로/path                                  trainable   Adam bytes   peak allocated/reserved MiB',
        *[f"{row['dataset'] + '/' + row['method'] + '/' + row['path']:48} {row['trainable_scalars']:9}   {row['adam_state_bytes']:10}   {row['peak_allocated_bytes'] / 1024 ** 2:.2f}/{row['peak_reserved_bytes'] / 1024 ** 2:.2f}" for row in summary['training_memory']],
        '```', '',
        '배포는 cache shortcut 없이 CPU FP32 context→전체 backbone/readout→contiguous CPU[48,C]였다. '
        'B1/B4·VAL 첫24원점·warmup10·24원점 pass10·회전3blocks의162행과18쌍 F0 전후 sentinel을 유지했다.', '', '```text',
        '자료/경로/B                                     seconds/origin  origins/s',
        *[f"{row['dataset'] + '/' + row['method'] + '/B' + str(row['batch']):48} {row['mean_selected_seed_median_seconds']:.6f}        {row['origins_per_second']:.2f}" for row in cost['deployment_summary']['methods']],
        '```', '',
        f"F0 sentinel after/before latency ratio 범위는 {summary['cost_grid']['sentinel_latency_ratio_range'][0]:.3f}–{summary['cost_grid']['sentinel_latency_ratio_range'][1]:.3f}다. "
        '사전 drift rejection threshold는 없었으므로 원값을 보존했다. frozen prefix는 cached 적합과 배포에서 GPU에 남았고 '
        '특징 캐시는 반복 적응의 비용을 바꾸지만 전체 모델 배포의 압축을 뜻하지 않는다. NLinear/Mini/Tiny/Chronos-2의 과거 비용은 이 회차 배포비 우위에 사용하지 않았다.', '',
        '[확인] 다음 사용량이 사전 S1 상한 이내였다. 보고 CPU 작업은 중앙 guard에 합산되며 값의 측정 시각은 verification.json에 남겼다.', '', '```text',
        *[f"{key:37} {row['actual']:>14.3f} / {row['cap']:>14.3f} ({100 * row['fraction_of_cap']:.2f}%)" for key, row in summary['resource_cap_checks'].items()],
        '```', '',
        '가장 강한 반론은 같은 recipe가 native HEAD의 충분한 최적화를 보장하지 않는다는 점이다. '
        'OUTPUT의 채널 혼합과 다른 용량도 성능 차이를 설명할 수 있다. 세 자료의 겹치는 시계열 원점을 독립 도메인 수백 개로 세지 않으며 '
        '이미 노출된 TEST와 서로 다른 부모·탐색 기회의 기존 LoRA만으로 내부 표현 부족을 증명하지 않는다. '
        'Time-PEFT/TRACE의 HeadOnly 및 head+LoRA, AdaPTS decoder/head-first, LogME 전이 지표와 짧은 pilot 탐색이 강한 기존 대안이다. '
        '같은 HEAD 부모 이후 이득과 총 의사결정비용을 줄이는 진단의 차이가 확인되어야 하며 신규성은 미확인이다.', '',
        '원문·공개 코드의 실제 열람 범위와 미확인 차이는 [선행 중복 지도](prior_art_map.md)에 기록했다.', '',
        f"S2 권고는 **{summary['s2']['recommendation']}**다. 아래 historical gap은 100×(HEAD−기존 MSE-LoRA)/기존 MSE-LoRA로 양수이면 기존 LoRA 참조가 낮다. "
        '이 권고를 구체화한 시점은 TEST를 본 뒤이므로 사후 탐색적 개발 결정이다. 주 근거는 봉인된 HEAD VAL 변화·검증된 학습 상태와 이번 회차 실제 시간·Adam 메모리다. '
        'historical TEST gap은 보조 참조이며 자료 선택 기준이나 matched-parent 추가이득의 측정값은 아니다. 세 자료를 모두 유지한다.', '', '```text',
        '자료                초기 HEAD VAL  선택 HEAD VAL  VAL 감소%  선택 epoch(seed1/2)',
        *[f"{row['dataset']:19} {row['sealed_initial_head_val_mse']:.6f}       {row['sealed_selected_head_val_mse']:.6f}       {row['sealed_head_val_reduction_percent']:+.3f}     {'/'.join(map(str, row['selected_parent_epochs']))}" for row in summary['s2']['dataset_estimates']],
        '```', '', '```text',
        '자료                historical LoRA gap%  HEAD online s/update  LoRA online s/update  joint memory별도peak합 GiB',
        *[f"{row['dataset']:19} {row['historical_lora_gap_percent']:+.3f}                 {row['head_online_slowest_block_step_seconds']:.6f}              {row['lora_online_slowest_block_step_seconds']:.6f}               {row['unmeasured_joint_separate_peak_sum_bytes'] / 1024 ** 3:.3f}" for row in summary['s2']['dataset_estimates']],
        '```', '',
        '[추정] 제안 범위는 기존 selected HEAD의 dataset/seed별 동일 checkpoint를 부모로 HEAD_CONTINUE와 HEAD_PLUS_LORA를 '
        '모든3자료·2seed에서 비교하는12 fits다. 각 자료의 selected HEAD LR 하나를 두 경로에 같이 쓰고 새 LR 선택은 하지 않는다. '
        'LoRA 초기수정0·동일 head 초기값·optimizer 재시작 규칙·sample suffix·같은 추가 update checkpoints를 사전 고정한다. '
        '같은 update 수와 같은 시간의 효과는 다르다. 두 LR를 검색하려면24 fits 및 별도 두 배 update 예산으로 승인 범위를 바꿔야 한다.', '',
        "[추정] 추가 최대120epochs×512origins/B4는 fit당15,360updates,12 fits 합184,320updates다. "
        f"이번 회차 online 최느린 block 단가에서 HEAD+LoRA를 HEAD와 LoRA의 합으로 근사하면 update 시간 {summary['s2']['maximum_update_projection_seconds']:.1f}초, "
        f"매epoch fresh full-VAL(초기 포함121회) {summary['s2']['maximum_full_val_projection_seconds']:.1f}초, raw 합 {summary['s2']['raw_fit_plus_val_projection_seconds']:.1f}초다. "
        f"단순2배 여유는 {summary['s2']['illustrative_twofold_fit_val_allowance_seconds']:.1f}초이며 실행 승인 cap이 아니다. "
        'HEAD+LoRA는 prefix가 달라지므로 S1 특징 cache를 사용할 수 없다. 합산 단가는 실제 joint backward 측정이나 엄밀한 상한이 아니고 '
        'restore/checkpoint/guard I/O·TEST·배포·보고 비용을 초기 probe 뒤 따로 확정해야 한다.', '',
        f"[추정] joint trainables는2,821,248개, FP32 weights+grad+Adam m/v raw payload는 {summary['s2']['joint_fp32_weights_grad_adam_payload_bytes']:,}bytes다. "
        '별도 HEAD/LoRA Adam 이후 reserved peak 합은 위 표의 참고값이며 joint activation/allocator peak는 미측정이다. '
        "첫 별도 probe 제안은12개 폐기용 부모×20updates+integrity reserve24=264technical updates, GPU wall cap 후보 "
        f"{summary['s2']['separate_initial_probe_proposal']['proposed_gpu_job_wall_cap_seconds']}초와 process RAM/GPU allocated·reserved 각각8GiB다. "
        'joint actual memory·native head trainable 유지·초기함수 parity를 먼저 측정하여 본실험 견적을 다시 봉인해야 한다. '
        'S2는 별도 승인 전 실행하지 않으며 S3/S4 선택기, 새 PEFT, PCA/G 유지·압축·신규성 주장은 보류한다. 유지하는 것은 matched-parent 질문, '
        '축소하는 것은 현재3자료에서의 관찰을 보편적 결론으로 확대하는 주장이다.', '']
    if preservation is not None:
        text.extend(['[확인] 기존 산출물76개를 현재 해시와 대조해 변경0개를 확인했다. '
                     'PPT/PDF와 무관한 기존 연구 산출물을 포함하는 실제 보호 확인은 [preservation_check.json](preservation_check.json)에 기록되어 있다. '
                     '그 확인 시각·대상·해시를 별도로 참조하며 S1 성능 검증과 섞지 않는다.', ''])
    text.extend(['검산 상세와 기계 판독 요약은 [verification.json](verification.json), 모든 기간·MAE·채널 수치와 source receipt는 '
                 '[evaluation.json](evaluation.json), raw 비용·memory·drift는 [costrows.json](costrows.json)에 있다. '
                 '이 보고는 commit/push나 live remote SHA를 확인하지 않으며 게시 상태는 별도 절차로 보고한다.', ''])
    return '\n'.join(text)


def report_all(budget=None):
    """Verify completed S1 artifacts, write verification.json/report.md, return SUMMARY."""
    started = time.perf_counter()
    verification = {'schema': 'tsfm_readout_s1_verification_v1', 'status': 'RUNNING',
                    'started_utc': time.time(), 'stage1_complete': False,
                    'technical_pass_is_scientific_gain': False, 'independent_confirmation': False,
                    's2_authorized': False, 's2_executed': False, 's3_executed': False, 's4_executed': False,
                    'scope': 'small receipts, checkpoint/file hashes, NPZ headers and recorded channel sufficient-statistic replay; no model or prediction-array loading'}
    try:
        state = _read(HERE / 'campaign_state.json')
        if state['status'] != 'complete':
            verification.update(status='STOP_RESOURCE' if state['status'] == 'STOP_RESOURCE' else 'INCOMPLETE',
                                campaign_state=state['status'], error=state.get('error'), elapsed_seconds=time.perf_counter() - started)
            _write_json(HERE / 'verification.json', verification)
            _write_text(HERE / 'report.md', '# S1 실행 상태\n\n계산 상태: ' + state['status'] + '. '
                        '완료 결과의 기술 PASS 또는 과학적 음성 결과로 해석하지 않는다.\n')
            return {'status': verification['status'], 'stage1_complete': False, 'verification_pass': False}
        _require(not any(state.get(key, False) for key in ('s2_executed', 's3_executed', 's4_executed')), 'authorized stage boundary')
        receipts = Receipts(budget)
        calibration, repair = _read(HERE / 'calibration_result.json'), _read(HERE / 'guard_repair.json')
        s0 = _read(HERE / 's0_budget_ledger.json')
        _require(calibration['status'] == repair['status'] == 'PASS'
                 and set(calibration['datasets']) == set(DATASETS)
                 and all(row['status'] == 'PASS' for row in calibration['datasets'].values()), 'three-dataset native S0 gates')
        _require(s0['counters']['technical_updates'] == 384
                 and all(s0['counters'][key] == 0 for key in ('scientific_fits', 'scientific_updates', 'new_test_predictions')),
                 'S0 native384 separate from scientific/TEST')
        selection, seal = _read(HERE / 'selection.json'), _read(HERE / 'selection_seal.json')
        seal_path = HERE / 'selection_seal.json'
        selection_receipt = {'path': seal_path.relative_to(ROOT).as_posix(), 'sha256': _hash(seal_path), 'bytes': seal_path.stat().st_size}
        fits, selected, normality, updates = _fits_and_selection(selection, seal, _read(HERE / 'fit_summary.json'), receipts)
        contracts = _read(HERE / 'data_contract.json')['source_contracts']
        evaluation = _read(HERE / 'evaluation.json')
        headers = _evaluation(evaluation, _read(HERE / 'prediction_manifest.json'), _read(HERE / 'exposure.json'),
                              seal, selection_receipt, selected, contracts, receipts)
        cost = _read(HERE / 'costrows.json')
        grid = _costs(cost, receipts)
        ledger = _read(HERE / 's1_budget_ledger.json')
        _require(ledger['counters']['scientific_fits'] == 24 and ledger['counters']['scientific_updates'] == updates
                 and ledger['counters']['technical_updates'] == 2160 and ledger['counters']['new_test_predictions'] == 12,
                 'actual counters / scientific result sums')
        usage, cap_checks = _resource_usage(state, ledger, _read(HERE / 's1_budget.json')['limits'], budget)
        cache = _read(HERE / 'cache_manifest.json')
        _require(cache['status'] == 'complete' and cache['test_access'] is False, 'complete TRAIN/VAL feature cache')
        cache_seconds = sum(row['total_prepare_seconds_this_attempt'] for row in cache['datasets'].values())
        generation_seconds = sum(row['generation_write_seconds_this_attempt'] for row in cache['datasets'].values())
        fit_seconds = sum(row['elapsed_s'] for row in fits.values())
        prediction_seconds = sum(row['prediction_seconds'] for row in evaluation['models'].values())
        selection_seconds = selection['selection_seconds']
        probe_seconds = cost['elapsed_probe_seconds']
        accounted = cache_seconds + fit_seconds + prediction_seconds + selection_seconds + probe_seconds
        remaining = usage['gpu_job_wall_seconds'] - accounted
        _require(remaining >= -1e-6, 'nonoverlapping elapsed component accounting')
        time_accounting = {'s1_feature_prepare_including_generation_write_restore_hash_seconds': cache_seconds,
            'all_24_fit_elapsed_including_train_val_checkpoint_replay_seconds': fit_seconds,
            'joint_lr_selection_seconds': selection_seconds, 'new_12_test_prediction_seconds': prediction_seconds,
            'matched_training_and_deployment_probe_elapsed_seconds': probe_seconds,
            'remaining_cache_read_scoring_bootstrap_source_guard_io_overhead_seconds': max(0., remaining),
            'total_s1_gpu_job_wall_seconds': usage['gpu_job_wall_seconds'],
            'generation_write_included_in_feature_prepare_seconds': generation_seconds,
            'fit_train_included_seconds': sum(row['cached_train_seconds'] for result in fits.values() for row in _read(ROOT / result['curve']['path'])),
            'fit_val_included_seconds': sum(row['full_val_seconds'] for result in fits.values() for row in _read(ROOT / result['curve']['path'])),
            'component_rule': 'sum the first six disjoint rows once; generation/train/VAL detail rows are included, not additional'}
        datasets, clear_gains = {}, 0
        for dataset in DATASETS:
            accuracy = evaluation['accuracy'][dataset]
            datasets[dataset] = {method: dict(accuracy[method]['combined']) for method in ('F0', 'MSE_LORA', 'LEVEL')}
            for arm in ARMS:
                members = [evaluation['models'][row['id']]['periods'] for row in selected if row['dataset'] == dataset and row['arm'] == arm]
                datasets[dataset][arm] = dict(accuracy[arm]['combined'], periods=accuracy[arm],
                    f0_reduction_percent=100 * (accuracy['F0']['combined']['mse'] - accuracy[arm]['combined']['mse']) / accuracy['F0']['combined']['mse'],
                    seed_ranges={period: [min(member[period]['mse'] for member in members), max(member[period]['mse'] for member in members)] for period in PERIODS})
            if any(evaluation['comparisons'][dataset][arm + '__vs__F0']['periods']['combined']['mse']['conditional_block95_absolute'][1] < 0 for arm in ARMS):
                clear_gains += 1
        scientific_description = 'STANDARD_READOUT_HELPFUL' if clear_gains == 3 else 'NO_CLEAR_READOUT_GAIN' if clear_gains == 0 else 'MIXED_READOUT_RESULT'
        s2 = _s2_proposal(evaluation, cost, contracts, selection, normality)
        preservation_path = HERE / 'preservation_check.json'
        preservation = _read(preservation_path) if preservation_path.exists() else None
        if preservation is not None:
            _require(preservation['status'] == 'PASS' and preservation['protected_preexisting_files'] == 76
                     and preservation['protected_preexisting_changed'] == 0, 'actual protected preexisting files')
        training_memory = []
        for method in cost['training_summary']['methods']:
            rows = [row for row in cost['training_rows'] if (row['dataset'], row['method'], row['path']) ==
                    (method['dataset'], method['method'], method['path'])]
            training_memory.append({'dataset': method['dataset'], 'method': method['method'], 'path': method['path'],
                'trainable_scalars': max(row['trainable_parameters'] for row in rows),
                'adam_state_bytes': max(row['adam_state_bytes'] for row in rows),
                'peak_allocated_bytes': max(row['peak_allocated_bytes_after_adam'] for row in rows),
                'peak_reserved_bytes': max(row['peak_reserved_bytes_after_adam'] for row in rows),
                'frozen_prefix_kept_on_gpu': all(row['frozen_prefix_kept_on_gpu'] for row in rows)})
        summary = {'status': 'STAGE1_DONE', 'stage1_complete': True, 'verification_pass': True,
            'scientific_description': scientific_description, 'description_is_research_success_criterion': False,
            'counters': ledger['counters'], 'datasets': datasets, 'normality': {
                'epoch0_selected_fits': sum(row['selected_epoch'] == 0 for row in normality),
                'max_epoch_fits': sum(row['termination'] == 'MAX_EPOCH' for row in normality),
                'oom_events': sum(row['oom_events'] for row in normality),
                'sampled_train_down_last_val_above_selected_fits': sum(row['sampled_train_down_last_val_above_selected'] for row in normality)},
            'cost_grid': grid, 'time_accounting': time_accounting, 'resource_usage': usage,
            'resource_cap_checks': cap_checks, 'training_memory': training_memory, 's2': s2,
            'remaining_decision_gaps': ['same HEAD parent continuation versus added internal LoRA under equal extra updates',
                'joint HEAD+LoRA actual time/Adam-state memory before a numeric S2 cap',
                'independent held-out targets and diagnostic decision cost versus standard short pilots']}
        verification.update(status='PASS', stage1_complete=True, campaign_status='complete',
            checks={'complete_24_fits': True, 'strict_epoch0_earliest_checkpoint': True,
                's0_three_dataset_native_gates_and_384_disposable_updates': True,
                'scheduler_epoch1_only': True, 'two_seed_val_lr_selection_first_listed_exact_tie': True,
                'all_choices_sealed_before_test_zero_at_sealing': True,
                'current_selected_checkpoint_hashes': True, 'recorded_frozen_state_in_fit_and_restored_test': True,
                '12_canonical_test_npz_headers': True, 'channel_macro_and_ab_pooled_counts': True,
                'arithmetic_mean_seed_loss_no_forecast_ensemble': True,
                'exact_relative_change_reference_denominator': True, 'actual_bootstrap_protocol_2000': True,
                '2160_technical_108_train_162_deploy_18_sentinel_pairs': True,
                'selected_checkpoint_probe_pre_post_current_hashes_unchanged': True,
                'cost_three_block_medians_then_seed_means': True, 'resource_caps_and_counter_sums': True},
            prediction_headers=headers, fit_normality=normality,
            actual_hash_checked_files=len(receipts.checked),
            large_file_metadata_only=list(receipts.large_metadata_only.values()),
            large_hash_scope='large backbone/source data/feature files were verified by the scientific driver/evaluator; this CPU report verifies their sealed metadata, not their full bytes again',
            bootstrap_check_scope='checks executed evaluator protocol and saved percentile/difference arithmetic; does not repeat 2000 draws or load prediction arrays',
            preservation_check=preservation, summary=summary, completed_utc=time.time(),
            elapsed_seconds=time.perf_counter() - started)
        _write_text(HERE / 'report.md', _report(summary, selection, evaluation, cost, normality,
                    s0, _read(HERE / 'cost_fixture_receipt.json'), preservation))
        _write_json(HERE / 'verification.json', verification)
        if budget is not None:
            budget.check()
        return summary
    except Exception as error:
        verification.update(status='STOP_RESOURCE' if 'STOP_RESOURCE' in str(error) else 'FAIL',
                            error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter() - started)
        _write_json(HERE / 'verification.json', verification)
        raise
