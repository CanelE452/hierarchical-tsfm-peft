"""Independent CPU, FIT-only numerical audit of the fixed-CP train/deploy gap.

Saved CUDA logits/noise are used as data. This script never regenerates CUDA
random numbers, runs Bolt, trains a model, or opens TEST/EVAL payloads.
"""
from pathlib import Path
import argparse
import gc
import json
import time
import traceback
import numpy as np
import psutil
import verify_primary as vp

METHODS = ('B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL', 'B_TANH')
DATASETS = ('jena', 'robin', 'uci235_household_hourly', 'uci274_sml2010')
SEEDS = (92601, 92602, 92603)


def fit_path(record):
    path = vp.checked_path(record['path'])
    if any('TEST' in p.upper() or 'EVAL' in p.upper() for p in path.parts):
        raise PermissionError('FIT_ONLY_GUARD_REJECTS_BEFORE_HASH')
    if vp.file_hash(path) != record['sha256']:
        raise ValueError('FIT_ASSET_SHA_MISMATCH:' + str(path))
    return path


def asset(audit, name, record):
    path = fit_path(record)
    audit.check(name + ':verified_SHA', True)
    if 'bytes' in record:
        audit.check(name + ':bytes', path.stat().st_size == record['bytes'])
    return path


def reconstruct_codes(logits, noise, temperature, mode):
    values = np.asarray(logits, np.float32)
    if mode == 'tanh':
        if np.asarray(noise).size:
            raise ValueError('TANH_HAS_NO_NOISE')
        return np.tanh(values / np.float32(temperature)).astype(np.float64)[None]
    noise = np.asarray(noise, np.float32)
    if mode not in ('gumbel', 'st') or noise.shape[1:] != (*values.shape, 2):
        raise ValueError('TWO_CLASS_NOISE_SHAPE')
    # Preserve the saved F32 class/noise addition; evaluate the two-class
    # probability independently as a stable logistic in F64.
    positive = (values[None] + noise[..., 0]) / np.float32(temperature)
    negative = (-values[None] + noise[..., 1]) / np.float32(temperature)
    difference = positive.astype(np.float64) - negative.astype(np.float64)
    if mode == 'st':
        return np.where(difference >= 0., 1., -1.)
    return np.tanh(difference / 2.)


def expected_clock(seed, initial_counter, mode):
    start = np.full(8, initial_counter, np.int64)
    if mode in ('gumbel', 'st'):
        start += 2 * np.arange(8, dtype=np.int64)
        counters = np.stack((start, start + 2), -1)
        seeds = seed * 1000003 + np.stack((start, start + 1), -1)
    else:
        counters, seeds = np.stack((start, start), -1), np.empty((8, 0), np.int64)
    return counters, seeds


def affine_budget(values, weight, bias, actual):
    x, w, b = [np.asarray(v, np.float64) for v in (values, weight, bias)]
    expected = x @ w.T + b
    # Standard dot-product bound under FP32 multiplies/adds, conservatively
    # counting 2R+3 rounded operations including bias. No empirical tolerance
    # is selected from the measured result.
    u = 2. ** -24
    operations = 2 * w.shape[-1] + 3
    gamma = operations * u / (1 - operations * u)
    allowance = gamma * (np.abs(x) @ np.abs(w).T + np.abs(b)) + np.finfo(np.float32).tiny
    error = np.abs(np.asarray(actual, np.float64) - expected)
    return {'passed': bool(np.all(error <= allowance)), 'max_abs': float(error.max()),
            'max_allowance': float(allowance.max()), 'maximum_error_to_allowance': float(np.max(error / allowance)),
            'assumption': 'FP32 affine arithmetic, no TF32/mixed-precision products; algebraic rounding bound, not CUDA kernel replay.'}


def normalized_score(prediction, target, observed):
    score = vp.macro_score(prediction, target, observed)
    return {k: score[k] for k in ('mse', 'mae', 'per_channel_mse', 'per_channel_mae')} | {
        'per_channel_observed_count': score['observed_counts']}


def recalculate_metrics(a, mode):
    deploy = normalized_score(a['deploy_prediction'], a['target'], a['observed'])
    draws = [normalized_score(p, a['target'], a['observed']) for p in a['train_predictions']]
    mse, mae = np.array([v['mse'] for v in draws]), np.array([v['mae'] for v in draws])
    difference = a['train_predictions'].astype(np.float64) - a['deploy_prediction'][None]
    query_flips = (a['train_query_codes'] >= 0) != (a['deploy_query_codes'][None] >= 0)
    key_flips = (a['train_key_codes'] >= 0) != (a['deploy_key_codes'][None] >= 0)
    eligible_query, used_keys = a['eligible'].any(-1), a['eligible'].any((0, 1))
    result = {'deploy_fit': deploy, 'train_draw_fit': draws, 'train_draw_mean_mse': float(mse.mean()),
        'train_draw_mean_mae': float(mae.mean()), 'train_draw_mse_sd_ddof1_mechanism_only': float(mse.std(ddof=1)),
        'train_minus_deploy_mse': float(mse.mean() - deploy['mse']), 'train_minus_deploy_mae': float(mae.mean() - deploy['mae']),
        'forecast_gap_rms': float(np.sqrt(np.mean(difference * difference))), 'forecast_gap_max_abs': float(np.abs(difference).max()),
        'query_code_sign_flip_fraction_all': float(query_flips.mean()),
        'query_code_sign_flip_fraction_eligible_rows': float(query_flips[:, eligible_query].mean()) if eligible_query.any() else None,
        'key_code_sign_flip_fraction_all': float(key_flips.mean()),
        'key_code_sign_flip_fraction_used_keys': float(key_flips[:, used_keys].mean()) if used_keys.any() else None,
        'eligible_counts_first_last': np.stack((a['eligible'][:, 0].sum(-1), a['eligible'][:, -1].sum(-1)), -1).tolist(),
        'eligible_row_count': int(eligible_query.sum()), 'used_key_count': int(used_keys.sum()),
        'query_logit_abs_quantiles': np.quantile(np.abs(a['query_logits']), [0., .05, .5, .95, 1.]).tolist(),
        'key_logit_abs_quantiles': np.quantile(np.abs(a['cached_key_logits']), [0., .05, .5, .95, 1.]).tolist(),
        'cache_recompute_forecast_max_abs': float(np.abs(a['recomputed_sign_prediction'] - a['deploy_prediction']).max()),
        'cache_recompute_key_logit_max_abs': float(np.abs(a['recomputed_key_logits'] - a['cached_key_logits']).max())}
    if mode in ('gumbel', 'st'):
        probability = lambda x: (lambda z: z / (1. + z))(np.exp(-2. * np.abs(np.asarray(x, np.float64))))
        result['analytic_gumbel_flip_probability'] = {'query_mean': float(probability(a['recomputed_query_logits']).mean()),
            'used_key_mean': float(probability(a['recomputed_key_logits'][used_keys]).mean()) if used_keys.any() else None}
    return result


def compare_metrics(audit, name, actual, claimed):
    if isinstance(actual, dict):
        for key, value in actual.items():
            compare_metrics(audit, name + ':' + key, value, claimed[key])
    elif isinstance(actual, list) and actual and isinstance(actual[0], dict):
        audit.check(name + ':length', len(actual) == len(claimed))
        for index, value in enumerate(actual):
            compare_metrics(audit, name + ':' + str(index), value, claimed[index])
    elif actual is None:
        audit.check(name, claimed is None)
    else:
        audit.numeric(name, actual, claimed)


def reference(audit, dataset, data, row, cache):
    if dataset in cache:
        audit.check(dataset + ':same_physical_container_all_conditions', row['inputs']['physical_container'] == cache[dataset]['container'])
        return cache[dataset]
    path = asset(audit, dataset + ':FIT_physical_container', row['inputs']['physical_container'])
    with np.load(path, allow_pickle=False) as store:
        if dataset in ('jena', 'robin'):
            old_mean, old_std = store['mean'].astype(np.float32), store['std'].astype(np.float32)
            old_std = np.where(old_std > 1e-6, old_std, 1.).astype(np.float32)
            physical = (store['x'].astype(np.float32) * old_std + old_mean).astype(np.float32)
            observed = store['finite'].astype(bool)
        else:
            physical, observed = store['physical'].astype(np.float32), store['mask'].astype(bool)
    mean, std = vp.fit_statistics(physical, observed, data['contract']['bounds']['fit'])
    cache[dataset] = {'physical': physical, 'observed': observed, 'mean': mean, 'std': std, 'container': row['inputs']['physical_container']}
    return cache[dataset]


def audit_condition(row, case, data, cache):
    import torch
    audit = vp.Audit()
    name = row['dataset'] + ':' + row['label']
    row_path = vp.HERE / 'train_deploy_gap_v1' / 'actual_selected_v1' / (row['dataset'] + '_' + row['label'] + '.json')
    audit.check(name + ':original_condition_JSON_matches', json.loads(row_path.read_text(encoding='utf-8-sig')) == row)
    audit.check(name + ':selected_checkpoint_reference_exact', row['selected_checkpoint'] == case['training']['selected_checkpoint'])
    cp_path = asset(audit, name + ':selected_CP', row['selected_checkpoint'])
    checkpoint = torch.load(cp_path, map_location='cpu', weights_only=True)
    state, evidence = checkpoint['model'], row['evidence']
    selected = case['training']['selected_update']
    audit.check(name + ':actual_CP_identity', checkpoint['update'] == selected and checkpoint['method'] == row['method']
        and checkpoint['seed'] == row['seed'] and checkpoint['weight'] == 0.)
    fingerprint = vp.tensor_state_hash(state)
    audit.check(name + ':actual_CP_fingerprint_before_after', fingerprint == case['training']['selected_hash'] == evidence['model_before_sha256'] == evidence['model_after_sha256'])
    audit.check(name + ':actual_selected_temperature', evidence['primary_update'] == selected and evidence['temperature'] == 1 - .75 * min(max(selected, 0), 600) / 600)
    mode = 'tanh' if row['method'] == 'B_TANH' else ('st' if row['method'] == 'B_ST' else 'gumbel')
    audit.check(name + ':mode_matches_method', evidence['training_hash_mode'] == mode)
    audit.check(name + ':only_hash_memory_modes_sensitive', not evidence['mode_sensitive_auxiliary_modules'] and not any('Dropout' in c or 'BatchNorm' in c for c in evidence['module_classes']))
    audit.check(name + ':restored_buffers_global_rng_evidence', evidence['checkpoint_and_buffers_restored'] is True and evidence['global_rng_unchanged'] is True)
    arrays_path = asset(audit, name + ':actual_arrays', row['arrays'])
    with np.load(arrays_path, allow_pickle=False) as store:
        a = {key: store[key].copy() for key in store.files}
    for key, value in a.items():
        audit.check(name + ':array:' + key + ':metadata', list(value.shape) == row['arrays']['arrays'][key]['shape'] and str(value.dtype) == row['arrays']['arrays'][key]['dtype'])
        if np.issubdtype(value.dtype, np.floating):
            audit.check(name + ':array:' + key + ':finite', np.isfinite(value).all())
    source_path = asset(audit, name + ':selected_FIT_member_source', row['inputs']['selected_prediction'])
    audit.check(name + ':selected_prediction_reference_exact', row['inputs']['selected_prediction'] == case['predictions'])
    with np.load(source_path, allow_pickle=False) as source:
        origins = source['fit_origins'].copy()
        selected_forecast = source['fit_prediction'][-8:].copy()
        for field in ('target', 'observed', 'origins'):
            audit.check(name + ':' + field + ':last_eight_FIT_exact', np.array_equal(a[field], source['fit_' + field][-8:]))
        for field in ('mean', 'std'):
            audit.check(name + ':' + field + ':same_selected_source', np.array_equal(a[field], source[field]))
    audit.check(name + ':exact_prespecified_indices', np.array_equal(a['indices'], np.arange(248, 256, dtype=np.int64)))
    audit.check(name + ':all256FIT_origins_SHA', origins.shape == (256,) and vp.array_hash(origins) == data['origin_sha256']['fit'])
    ref = reference(audit, row['dataset'], data, row, cache)
    for field in ('mean', 'std'):
        audit.check(name + ':' + field + ':independent_FIT_F64_statistics', np.array_equal(a[field], ref[field]) and vp.array_hash(a[field]) == data[field + '_sha256'])
    lo, hi = data['contract']['bounds']['fit']
    audit.check(name + ':past_and_target_within_FIT', np.all(a['origins'] - 512 >= lo) and np.all(a['origins'] + 48 <= hi))
    history, target, _, mask = vp.raw_windows(ref['physical'], ref['observed'], a['origins'], a['mean'], a['std'])
    for field, values in (('history', history), ('target', target), ('observed', mask)):
        audit.check(name + ':' + field + ':independent_physical_source_exact', np.array_equal(a[field], values))
    audit.numeric(name + ':deploy_vs_saved_selected_FIT_distinct_forward', a['deploy_prediction'], selected_forecast, atol=1e-5, rtol=2e-6)
    for key, cp_key in (('bank_origins', 'bank_origins'), ('bank_end_exclusive', 'bank_end_exclusive'), ('bank_valid', 'bank_valid'),
                        ('cached_memory_embedding', 'memory_embedding'), ('cached_key_logits', 'memory_hash_logits')):
        audit.check(name + ':' + key + ':actual_CP_buffer_exact', np.array_equal(a[key], state[cp_key].numpy()))
    audit.check(name + ':CP_cache_frozen', bool(state['memory_is_frozen']))
    audit.check(name + ':bank_ends_actual48', np.array_equal(a['bank_end_exclusive'], a['bank_origins'] + 48))
    audit.check(name + ':bank_valid_from_observation', np.array_equal(a['bank_valid'], state['bank_observed'].flatten(1).any(-1).numpy()))
    for key, data_key in (('bank_origins', 'bank_origin_sha256'), ('bank_delta', 'bank_delta_sha256'), ('bank_observed', 'bank_observed_sha256')):
        audit.check(name + ':' + key + ':data_SHA', vp.array_hash(state[key].numpy()) == data[data_key])
    eligible = vp.memory_eligibility(a['origins'], a['bank_origins'], length=512, horizon=48) & a['bank_valid'][None, None]
    audit.check(name + ':rowwise_causal_eligibility_exact', np.array_equal(a['eligible'], eligible))
    initial_counter = int(state['noise_calls'])
    audit.check(name + ':initial_counter_actual_CP', initial_counter == evidence['initial_noise_counter'])
    counters, seeds = expected_clock(row['seed'], initial_counter, mode)
    audit.check(name + ':actual_counter_progression', np.array_equal(a['noise_counters'], counters))
    audit.check(name + ':logical_seed_progression', np.array_equal(a['noise_seeds'], seeds))
    for platform in ('cpu', 'cuda'):
        audit.check(name + ':' + platform + '_global_rng_restored', np.array_equal(a[platform + '_global_rng_before'], a[platform + '_global_rng_after']))
    code_diagnostics = {}
    for side, logits_key, noise_key, code_key, deploy_key in (
        ('query', 'recomputed_query_logits', 'query_gumbel_noise', 'train_query_codes', 'deploy_query_codes'),
        ('key', 'recomputed_key_logits', 'key_gumbel_noise', 'train_key_codes', 'deploy_key_codes')):
        reconstructed = reconstruct_codes(a[logits_key], a[noise_key], evidence['temperature'], mode)
        if mode == 'tanh':
            reconstructed = np.repeat(reconstructed, 8, 0)
        actual = a[code_key]
        audit.numeric(name + ':' + side + ':CPU_saved_logit_noise_code_formula', reconstructed, actual, atol=2e-6, rtol=2e-6)
        cached_logits = a['query_logits' if side == 'query' else 'cached_key_logits']
        cached_sign = np.where(cached_logits >= 0, 1., -1.)
        recomputed_sign = np.where(a[logits_key] >= 0, 1., -1.)
        audit.check(name + ':' + side + ':deploy_sign_from_cached_actual_logits_exact', np.array_equal(a[deploy_key], cached_sign))
        code_diagnostics[side] = {'maximum_CPU_formula_code_difference': float(np.abs(reconstructed - actual).max()),
            'cached_vs_recomputed_sign_bit_mismatches': int((cached_sign != recomputed_sign).sum()),
            'cached_vs_recomputed_logit_maxabs': float(np.abs(cached_logits.astype(np.float64) - a[logits_key].astype(np.float64)).max())}
        if mode in ('gumbel', 'st'):
            noise = a[noise_key]
            eps = np.finfo(np.float32).eps
            lower, upper = -np.log(-np.log(eps)), -np.log(-np.log(1 - eps))
            audit.check(name + ':' + side + ':saved_noise_clamped_uniform_range', noise.shape[0] == 8 and float(noise.min()) >= lower - 2e-6 and float(noise.max()) <= upper + 2e-6)
            code_diagnostics[side]['actual_saved_noise_range'] = [float(noise.min()), float(noise.max())]
        else:
            audit.check(name + ':' + side + ':no_tanh_Gumbel_noise', a[noise_key].size == 0)
    affine = {}
    for variant, embedding, logits in (('cached', 'cached_memory_embedding', 'cached_key_logits'),
                                      ('recomputed', 'recomputed_memory_embedding', 'recomputed_key_logits')):
        affine[variant] = affine_budget(a[embedding], state['future_hash.weight'].numpy(), state['future_hash.bias'].numpy(), a[logits])
        audit.check(name + ':' + variant + ':Wf_FP32_affine_roundoff_bound', affine[variant]['passed'], affine[variant])
    attention = vp.numpy_attention(a['deploy_query_codes'], a['deploy_key_codes'], a['eligible'])
    audit.numeric(name + ':deploy_attention_independent_cosine_softmax', attention, a['deploy_attention'], atol=2e-6, rtol=2e-6)
    audit.check(name + ':disallowed_attention_exact_zero', np.all(a['deploy_attention'][~a['eligible']] == 0))
    if mode == 'tanh':
        audit.check(name + ':eight_noiseless_train_predictions_identical', all(np.array_equal(p, a['train_predictions'][0]) for p in a['train_predictions']))
    metrics = recalculate_metrics(a, mode)
    compare_metrics(audit, name + ':metric', metrics, row['metrics'])
    audit.check(name + ':CP_disk_SHA_unchanged_after_audit', vp.file_hash(cp_path) == row['selected_checkpoint']['sha256'])
    result = audit.close(dataset=row['dataset'], method=row['method'], seed=row['seed'], label=row['label'],
        selected_update=selected, selected_temperature=evidence['temperature'], initial_counter=initial_counter,
        independently_recomputed_metrics=metrics, code_diagnostics=code_diagnostics, Wf_affine_diagnostics=affine,
        distinct_forward_parity={'deploy_vs_saved_selected_FIT_maxabs': float(np.abs(a['deploy_prediction'].astype(np.float64)-selected_forecast.astype(np.float64)).max()),
          'atol':1e-5,'rtol':2e-6,'scope':'Different actual fixed-checkpoint FIT forwards, separate from strict NPZ-to-own-JSON rescoring.'},
        limitations=['CUDA random number generation is not replayed. Code formulas use the actually saved CUDA logits and noise.',
                    'Wp/E/Bolt forward is not rerun on CPU; checkpoint fingerprint and frozen producer source bind the recorded logits.',
                    'Analytic noise probabilities assume ideal independent standard Gumbels; actual uniforms are FP32-clamped.'])
    del a, state, checkpoint
    return result


def run_audit(result_path):
    start = time.perf_counter()
    import torch
    torch.set_num_threads(2)
    process = psutil.Process(); peak_rss = process.memory_info().rss
    path = vp.checked_path(result_path)
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    if result['status'] != 'COMPLETE':
        raise ValueError('COMPLETE_GAP_RESULT_REQUIRED')
    audit = vp.Audit()
    expected = {(ds, seed, method) for ds in DATASETS for seed in SEEDS for method in METHODS}
    coverage = vp.exact_coverage(result['conditions'], expected, ('dataset','seed','method'))
    audit.check('gap:exact48_complete8draws', coverage['passed'] and result['actual_completed_conditions'] == 48 and not result['failed_conditions']
        and result['uncompleted_conditions'] == 0 and result['draws_per_condition'] == 8)
    lock_path = asset(audit, 'gap:METHOD_LOCK', result['method_lock'])
    lock = json.loads(lock_path.read_text(encoding='utf-8-sig'))
    audit.check('gap:source16_matches_METHOD_LOCK', len(result['source_hashes']) == 16 and result['source_hashes'] == lock['source_hashes'])
    snapshot = vp.ROOT / '.cache' / vp.HERE.name / 'source_snapshots' / ('train_deploy_gap_' + path.parent.name)
    manifest = json.loads((snapshot/'MANIFEST.json').read_text(encoding='utf-8-sig'))
    audit.check('gap:source_snapshot_map', manifest['sha256'] == result['source_hashes'])
    for relative, digest in result['source_hashes'].items():
        audit.check('gap:current_source:' + relative, vp.file_hash(relative) == digest)
        audit.check('gap:snapshot_source:' + relative, vp.file_hash(snapshot/relative) == digest)
    helper = vp.HERE/'measure_train_deploy_gap.py'
    saved_helper = vp.ROOT/'.cache'/vp.HERE.name/'train_deploy_gap_v1'/path.parent.name/'PRODUCER.py'
    audit.check('gap:helper_current_and_snapshot_SHA', vp.file_hash(helper) == result['script_sha256'] == vp.file_hash(saved_helper))
    audit.check('gap:F_weights_receipt_unchanged', bool(result['backbone_before_sha256']) and result['backbone_before_sha256'] == result['backbone_after_sha256'])
    preflight = json.loads((vp.HERE/'PREFLIGHT.json').read_text(encoding='utf-8-sig'))
    for file in preflight['bolt_snapshot']['files']:
        asset(audit,'gap:original_F_file:' + file['path'],file)
    source_cases = {}
    for record in result['sources']:
        source_path = asset(audit, 'gap:primary_result:' + record['path'], record)
        source = json.loads(source_path.read_text(encoding='utf-8-sig'))
        audit.check('gap:primary_complete:' + record['path'], source['status'] == 'COMPLETE' and source['source_hashes'] == result['source_hashes'])
        audit.check('gap:primary_F_fingerprint:' + record['path'], source['backbone_before_sha256'] == source['backbone_after_sha256'] == result['backbone_before_sha256'])
        for case in source['cases']:
            if case['method'] in METHODS:
                key=(case['dataset'],case['seed'],case['method'])
                audit.check('gap:primary_unique:' + str(key),key not in source_cases)
                source_cases[key]=(case,source['data'][case['dataset']],record)
        del source
    audit.check('gap:primary_source_cases_exact48', set(source_cases) == expected and len(source_cases) == 48)
    reports, failed, cache = [], [], {}
    for row in result['conditions']:
        key=(row['dataset'],row['seed'],row['method'])
        try:
            case,data,source_record=source_cases[key]
            audit.check('gap:condition_source_ref:' + str(key),row['source_result'] == source_record)
            report=audit_condition(row,case,data,cache)
            reports.append(report)
            if report['failed_count']:
                failed.append({'identity':key,'failed_count':report['failed_count']})
        except Exception as exc:
            failed.append({'identity':key,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        peak_rss=max(peak_rss,process.memory_info().rss)
        gc.collect()
    audit.check('gap:all48_numerically_audited_without_failed_conditions',len(reports)==48 and not failed)
    resource_path=asset(audit,'gap:original_terminal_GPU_receipt',result['resource_receipt'])
    resource=json.loads(resource_path.read_text(encoding='utf-8-sig'))
    audit.check('gap:original_GPU_receipt_complete',resource['status']=='COMPLETE' and resource['gpu'] is True)
    for relative,digest in result['source_hashes'].items():
        audit.check('gap:source_unchanged_after:' + relative,vp.file_hash(relative)==digest)
    audit.check('gap:CPU_no_CUDA_initialized',not torch.cuda.is_initialized())
    return audit.close(coverage=coverage,condition_audits=reports,failed_conditions=failed,
        result_artifact={'path':str(path.relative_to(vp.ROOT)),'sha256':vp.file_hash(path)},
        original_GPU_resource=result['resource_receipt'],original_GPU_process_wall_seconds=resource['wall_seconds'],
        cpu_wall_seconds=time.perf_counter()-start,observed_peak_rss_bytes=peak_rss,
        memory_measurement_scope='Process RSS sampled at startup and after each released condition, not a continuous peak sampler.',
        all_conditions_numeric_pass=len(reports)==48 and not failed,GPU_executed=False,new_training_performed=False,
        protected_TEST_opened=False,fresh_EVAL_target_opened=False,fresh_EVAL_target_SHA_recomputed=False,
        historical_CUDA_RNG_replayed=False,forecast_improvement_verified=False,independent_generalization_verified=False,
        limitations=['Eight draws on eight prespecified, potentially overlapping FIT windows measure mechanism variation only.',
            'Selected-checkpoint train mode uses temperature(selected_update); it is not replay of the last historical update(selected_update-1).',
            'Actual B_TANH training mode is separate from inference-only tanh at fixed update600.',
            'Stored soft/hard codes and forecasts are rescored; CUDA RNG seeds/forecaster kernels are not rerun.',
            'Shared compressed legacy TRAIN/DEVVAL or fresh FIT/TUNE members are decompressed to obtain physical source values; only FIT windows are constructed/scored.',
            'Passing integrity/causality/scoring checks does not establish forecasting improvement or a cause of prior failures.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=vp.checked_path(args.output)
    if not output.is_relative_to(vp.HERE/'validation_v1'):
        raise ValueError('OUTPUT_OUTSIDE_OWNED_VALIDATION')
    start=time.time(); result=run_audit(args.result)
    result.update(audit_started_unix=start,audit_finished_unix=time.time(),cpu_wall_scope='Measured audit body; startup imports, output serialization and writing excluded.')
    source=Path(__file__);digest=vp.file_hash(source)
    saved=output.with_name(output.stem+'_validator_'+digest[:12]+'.py')
    result['validator_source']={'path':str(source.relative_to(vp.ROOT)),'sha256':digest,'immutable_snapshot':str(saved.relative_to(vp.ROOT)),
        'independent_shared_numeric_helper':{'path':str(Path(vp.__file__).relative_to(vp.ROOT)),'sha256':vp.file_hash(vp.__file__)}}
    json.dumps(vp.jsonable(result),ensure_ascii=False,allow_nan=False)
    saved.parent.mkdir(parents=True,exist_ok=True)
    with saved.open('xb') as handle:handle.write(source.read_bytes())
    vp.write_fresh(output,result)
    print(json.dumps({'status':result['status'],'failed_count':result['failed_count'],'conditions':len(result['condition_audits']),
        'failed_conditions':len(result['failed_conditions']),'cpu_wall_seconds':result['cpu_wall_seconds'],'output':str(output),'sha256':vp.file_hash(output)}))
    return 0 if result['failed_count']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
