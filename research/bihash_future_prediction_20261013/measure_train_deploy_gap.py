"""Matched fixed-checkpoint FIT diagnosis of stochastic training versus sign deployment.

This does not train, select checkpoints, open EVAL targets, or reproduce ExoTimer.
The eight draws are mechanism variation on eight prespecified FIT windows.
"""
from pathlib import Path
import argparse
import gc
import time
import traceback

import numpy as np
import torch
from torch import nn

import runtime as rt
import model as bm
import data as dt

METHODS = ('B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL', 'B_TANH')
DATASETS = ('jena', 'robin', 'uci235_household_hourly', 'uci274_sml2010')
SEEDS = (92601, 92602, 92603)
DRAWS = 8
WINDOWS = 8


def sign(value):
    return np.where(np.asarray(value) >= 0, 1., -1.).astype(np.float32)


def analytic_flip_probability(logits):
    """For logits [I,-I], independent standard Gumbels give sign-flip sigmoid(-2|I|)."""
    margin = np.abs(np.asarray(logits, np.float64))
    ratio = np.exp(-2 * margin)
    return ratio / (1 + ratio)


def macro_scores(prediction, target, observed):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    observed = np.asarray(observed, bool)
    if prediction.shape != target.shape or target.shape != observed.shape or prediction.ndim != 3:
        raise ValueError('FIT_METRIC_SHAPE')
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError('FIT_METRIC_NONFINITE')
    count = observed.sum((0, 1))
    valid = count > 0
    if not valid.all():
        raise ValueError('FIT_CHANNEL_WITHOUT_OBSERVATION')
    error = np.where(observed, prediction - target, 0.)
    channel_mse = (error ** 2).sum((0, 1)) / count
    channel_mae = np.abs(error).sum((0, 1)) / count
    return {'mse': float(channel_mse.mean()), 'mae': float(channel_mae.mean()),
            'per_channel_mse': channel_mse.tolist(), 'per_channel_mae': channel_mae.tolist(),
            'per_channel_observed_count': count.tolist()}


def mode_sensitive_modules(model):
    sensitive = []
    dropout_classes = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                       nn.AlphaDropout, nn.FeatureAlphaDropout)
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            sensitive.append({'name': name, 'class': type(module).__name__, 'reason': 'batch normalization'})
        elif isinstance(module, dropout_classes) and module.p > 0:
            sensitive.append({'name': name, 'class': type(module).__name__, 'reason': 'nonzero dropout'})
        elif isinstance(module, nn.RNNBase) and module.dropout > 0:
            sensitive.append({'name': name, 'class': type(module).__name__, 'reason': 'recurrent dropout'})
    return sensitive


def safe_fit_artifact(artifact):
    path = (rt.ROOT / artifact['path']).resolve()
    forbidden = any('TEST' in part.upper() or 'EVAL' in part.upper()
                    for part in path.relative_to(rt.ROOT).parts) if path.is_relative_to(rt.ROOT) else True
    if forbidden:
        raise ValueError('NON_FIT_ARTIFACT_REJECTED_BEFORE_HASH')
    if rt.sha(path) != artifact['sha256']:
        raise ValueError('FIT_ARTIFACT_SHA_MISMATCH: ' + str(path))
    return path


def expected_cases(results):
    expected = {(ds, seed, method) for ds in DATASETS for seed in SEEDS for method in METHODS}
    records = []
    for result, provenance in results:
        if result['status'] != 'COMPLETE':
            raise ValueError('PRIMARY_RESULT_NOT_COMPLETE')
        for case in result['cases']:
            if case['method'] in METHODS:
                if case['weight'] != 0. or case['training']['updates'] != 600:
                    raise ValueError('PRIMARY_CASE_NOT_FIXED_600_TRAINED')
                records.append((case, result['data'][case['dataset']], provenance))
    identities = [(c['dataset'], c['seed'], c['method']) for c, _, _ in records]
    if len(identities) != 48 or len(set(identities)) != 48 or set(identities) != expected:
        raise ValueError('TRAIN_DEPLOY_COVERAGE_NOT_EXACT_48')
    return records


def load_fit_inputs(case, data_record):
    """Use FIT arrays only; shared compressed containers' byte scope is declared."""
    prediction_path = safe_fit_artifact(case['predictions'])
    with np.load(prediction_path, allow_pickle=False) as payload:
        origins = payload['fit_origins'].copy()
        target = payload['fit_target'][-WINDOWS:].copy()
        observed = payload['fit_observed'][-WINDOWS:].copy()
        mean, std = payload['mean'].copy(), payload['std'].copy()
    if origins.shape != (256,) or origins.dtype != np.int64 or np.any(np.diff(origins) <= 0):
        raise ValueError('FIT_ORIGIN_CONTRACT')
    for key, value in (('mean', mean), ('std', std)):
        if dt.array_hash(value) != data_record[key + '_sha256']:
            raise ValueError('FIT_STATISTICS_HASH: ' + key)
    if dt.array_hash(origins) != data_record['origin_sha256']['fit']:
        raise ValueError('FIT_ORIGIN_HASH')
    selected = origins[-WINDOWS:]
    contract = data_record['contract']
    if case['dataset'] in ('jena', 'robin'):
        artifact = next(r for r in data_record['source_records'] if r['role'] == 'legacy_trainval_npz')
        source = safe_fit_artifact(artifact)
        with np.load(source, allow_pickle=False) as payload:
            stored = payload['x'].astype(np.float32)
            mask = payload['finite'].astype(bool)
            source_mean = payload['mean'].astype(np.float32)
            source_std = np.where(payload['std'].astype(np.float32) > 1e-6,
                                  payload['std'].astype(np.float32), 1.).astype(np.float32)
        physical = (stored * source_std + source_mean).astype(np.float32)
        container_note = 'Legacy compressed TRAIN/DEVVAL x and finite members are decompressed; only the prespecified FIT windows are assembled or scored. No VAL/TUNE target windows are constructed.'
    else:
        artifact = contract['payloads']['fit_tune']
        source = safe_fit_artifact(artifact)
        with np.load(source, allow_pickle=False) as payload:
            physical = payload['physical'].astype(np.float32)
            mask = payload['mask'].astype(bool)
        container_note = 'Shared compressed FIT/TUNE physical and mask members are decompressed; only the prespecified FIT windows are assembled or scored. EVAL payloads are neither opened nor hashed.'
    lo, hi = contract['bounds']['fit']
    if np.any(selected < max(512, lo)) or np.any(selected + 48 > hi):
        raise ValueError('SELECTED_WINDOW_OUTSIDE_FIT')
    history, _, _ = dt._history_windows(physical, mask, selected, mean, std)
    recalculated, _, recalculated_observed = dt._target_windows(physical, mask, selected, mean, std)
    if not np.array_equal(target, recalculated) or not np.array_equal(observed, recalculated_observed):
        raise ValueError('FIT_TARGET_SOURCE_PARITY')
    return {'history': history, 'target': target, 'observed': observed, 'origins': selected,
            'indices': np.arange(256 - WINDOWS, 256, dtype=np.int64), 'mean': mean, 'std': std,
            'provenance': {'selected_prediction': case['predictions'], 'physical_container': artifact,
                           'container_scope': container_note, 'target_use': 'FIT scoring only, never a forward argument'}}


def load_fixed_model(case, device):
    checkpoint = case['training']['selected_checkpoint']
    path = safe_fit_artifact(checkpoint)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['update'] != case['training']['selected_update'] or saved['method'] != case['method']:
        raise ValueError('SELECTED_CHECKPOINT_IDENTITY')
    state = saved['model']
    basis = state['codec.fixed_basis'].numpy().copy()
    model = bm.FutureBiHash(np.zeros(basis.shape[0], np.float32), basis, case['seed'], case['method'],
        state['bank_delta'].numpy(), state['bank_observed'].numpy(), state['bank_origins'].numpy()).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    if rt.state_hash(model) != case['training']['selected_hash']:
        raise ValueError('SELECTED_MODEL_STATE_HASH')
    if not bool(model.memory_is_frozen):
        raise ValueError('SELECTED_MEMORY_NOT_FROZEN')
    return model, checkpoint


def _array(value):
    return value.detach().cpu().numpy().copy()


def _gumbel_noise(logits, seed):
    generator = torch.Generator(device=logits.device).manual_seed(int(seed))
    shape = (*logits.shape, 2)
    uniform = torch.rand(shape, device=logits.device, dtype=logits.dtype, generator=generator)
    uniform = uniform.clamp(torch.finfo(uniform.dtype).eps, 1 - torch.finfo(uniform.dtype).eps)
    return -torch.log(-torch.log(uniform))


def measure_fixed_model(model, inputs, forecaster, check):
    sensitive = mode_sensitive_modules(model)
    if sensitive:
        raise ValueError('UNCONTROLLED_TRAIN_MODE_MODULES: ' + str(sensitive))
    state = rt.clone_state(model)
    before = rt.state_hash(model)
    flags = {name: module.training for name, module in model.named_modules()}
    device = next(model.parameters()).device
    history = rt.tensor(inputs['history'], device)
    origins = torch.as_tensor(inputs['origins'], dtype=torch.long, device=device)
    update = int(inputs['selected_update'])
    temperature = model.temperature(update)
    counter = int(state['noise_calls'])
    arrays = {k: inputs[k] for k in ('history', 'target', 'observed', 'origins', 'indices', 'mean', 'std')}
    arrays.update(bank_origins=_array(model.bank_origins), bank_end_exclusive=_array(model.bank_end_exclusive),
                  bank_valid=_array(model.bank_valid), cached_memory_embedding=_array(model.memory_embedding),
                  cached_key_logits=_array(model.memory_hash_logits),
                  cpu_global_rng_before=torch.random.get_rng_state().numpy().copy())
    if device.type == 'cuda':
        arrays['cuda_global_rng_before'] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
    draw_forecasts, query_codes, key_codes = [], [], []
    query_noise, key_noise, counters, seeds = [], [], [], []
    try:
        with torch.no_grad():
            model.eval()
            prediction, deploy = model(history, origins, forecaster, update=update, return_details=True)
            arrays.update(deploy_prediction=_array(prediction), eligible=_array(deploy['eligible']),
                deploy_query_codes=_array(deploy['query_code']), deploy_key_codes=_array(deploy['memory_code']),
                query_logits=_array(model.student_hash(deploy['student'])),
                deploy_attention=_array(deploy['attention']))
            if rt.state_hash(model) != before:
                raise ValueError('DEPLOY_FORWARD_MUTATED_STATE')
            model.memory_is_frozen.fill_(False)
            cache_prediction, cache_details = model(history, origins, forecaster, update=update, return_details=True)
            arrays.update(recomputed_sign_prediction=_array(cache_prediction),
                recomputed_memory_embedding=_array(cache_details['memory']),
                recomputed_key_logits=_array(model.future_hash(cache_details['memory'])),
                recomputed_query_logits=_array(model.student_hash(cache_details['student'])))
            for draw in range(DRAWS):
                model.load_state_dict(state)
                model.train()
                start_counter = counter + (2 * draw if model._mode() in ('gumbel', 'st') else 0)
                model.noise_calls.fill_(start_counter)
                actual, details = model(history, origins, forecaster, update=update, return_details=True)
                end_counter = int(model.noise_calls)
                expected_end = start_counter + (2 if model._mode() in ('gumbel', 'st') else 0)
                if end_counter != expected_end:
                    raise ValueError('NOISE_COUNTER_PROGRESS')
                draw_forecasts.append(_array(actual))
                query_codes.append(_array(details['query_code']))
                key_codes.append(_array(details['memory_code']))
                counters.append([start_counter, end_counter])
                draw_seeds = []
                if model._mode() in ('gumbel', 'st'):
                    draw_seeds = [model.seed * 1000003 + start_counter,
                                  model.seed * 1000003 + start_counter + 1]
                    q_logits = model.student_hash(details['student'])
                    p_logits = model.future_hash(details['memory'])
                    q_noise, p_noise = _gumbel_noise(q_logits, draw_seeds[0]), _gumbel_noise(p_logits, draw_seeds[1])
                    for logits, noise, actual_code in ((q_logits, q_noise, details['query_code']),
                                                      (p_logits, p_noise, details['memory_code'])):
                        probability = torch.softmax((bm.binary_class_logits(logits) + noise) / temperature, -1)
                        replay = 2 * probability[..., 0] - 1
                        if model._mode() == 'st':
                            hard = torch.where(probability[..., 0] >= probability[..., 1], 1., -1.)
                            replay = hard.detach() - replay.detach() + replay
                        if not torch.equal(replay, actual_code):
                            raise ValueError('ACTUAL_GUMBEL_CODE_REPLAY_NOT_EXACT')
                    query_noise.append(_array(q_noise)); key_noise.append(_array(p_noise))
                seeds.append(draw_seeds)
                mutated = rt.clone_state(model)
                mutated['noise_calls'] = state['noise_calls'].clone()
                if rt.state_hash(mutated) != before:
                    raise ValueError('TRAIN_FORWARD_MUTATED_NONCOUNTER_STATE')
                check()
        arrays.update(train_predictions=np.stack(draw_forecasts), train_query_codes=np.stack(query_codes),
            train_key_codes=np.stack(key_codes), noise_counters=np.asarray(counters, np.int64),
            noise_seeds=np.asarray(seeds, np.int64).reshape(DRAWS, -1),
            query_gumbel_noise=np.stack(query_noise) if query_noise else np.empty((0,), np.float32),
            key_gumbel_noise=np.stack(key_noise) if key_noise else np.empty((0,), np.float32))
    finally:
        model.load_state_dict(state)
        for name, module in model.named_modules():
            module.training = flags[name]
    after = rt.state_hash(model)
    arrays['cpu_global_rng_after'] = torch.random.get_rng_state().numpy().copy()
    if device.type == 'cuda':
        arrays['cuda_global_rng_after'] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
    if after != before or not np.array_equal(arrays['cpu_global_rng_before'], arrays['cpu_global_rng_after']):
        raise ValueError('CHECKPOINT_OR_GLOBAL_RNG_NOT_RESTORED')
    if device.type == 'cuda' and not np.array_equal(arrays['cuda_global_rng_before'], arrays['cuda_global_rng_after']):
        raise ValueError('GLOBAL_CUDA_RNG_CHANGED')
    return arrays, {'model_before_sha256': before, 'model_after_sha256': after,
        'primary_update': update, 'temperature': temperature, 'initial_noise_counter': counter,
        'temperature_scope': 'Fixed selected weights, source temperature(selected_update), and continuation counters. This is not replay of a historical optimization step; the last historical step used selected_update-1.',
        'actual_mode': 'Full adaptation model.train(); FrozenBolt remains eval; no optimizer or gradients',
        'mode_sensitive_auxiliary_modules': sensitive,
        'module_classes': sorted(set(type(m).__name__ for m in model.modules())),
        'cache_scope': 'Deploy uses stored frozen memory; actual training recalculates phi/Wf. A separate recomputed-sign prediction isolates cache rounding from code noise.',
        'training_hash_mode': model._mode(), 'checkpoint_and_buffers_restored': True,
        'global_rng_unchanged': True}


def summarize(arrays, mode):
    deploy = macro_scores(arrays['deploy_prediction'], arrays['target'], arrays['observed'])
    draws = [macro_scores(p, arrays['target'], arrays['observed']) for p in arrays['train_predictions']]
    difference = arrays['train_predictions'].astype(np.float64) - arrays['deploy_prediction'][None]
    q_flips = sign(arrays['train_query_codes']) != arrays['deploy_query_codes'][None]
    k_flips = sign(arrays['train_key_codes']) != arrays['deploy_key_codes'][None]
    eligible_query = arrays['eligible'].any(-1)
    used_keys = arrays['eligible'].any((0, 1))
    result = {'deploy_fit': deploy, 'train_draw_fit': draws,
        'train_draw_mean_mse': float(np.mean([d['mse'] for d in draws])),
        'train_draw_mean_mae': float(np.mean([d['mae'] for d in draws])),
        'train_draw_mse_sd_ddof1_mechanism_only': float(np.std([d['mse'] for d in draws], ddof=1)),
        'train_minus_deploy_mse': float(np.mean([d['mse'] for d in draws]) - deploy['mse']),
        'train_minus_deploy_mae': float(np.mean([d['mae'] for d in draws]) - deploy['mae']),
        'forecast_gap_rms': float(np.sqrt(np.mean(difference ** 2))),
        'forecast_gap_max_abs': float(np.abs(difference).max()),
        'query_code_sign_flip_fraction_all': float(q_flips.mean()),
        'query_code_sign_flip_fraction_eligible_rows': float(q_flips[:, eligible_query].mean()) if eligible_query.any() else None,
        'key_code_sign_flip_fraction_all': float(k_flips.mean()),
        'key_code_sign_flip_fraction_used_keys': float(k_flips[:, used_keys].mean()) if used_keys.any() else None,
        'eligible_counts_first_last': np.stack((arrays['eligible'][:, 0].sum(-1), arrays['eligible'][:, -1].sum(-1)), -1).tolist(),
        'eligible_row_count': int(eligible_query.sum()), 'used_key_count': int(used_keys.sum()),
        'query_logit_abs_quantiles': np.quantile(np.abs(arrays['query_logits']), [0., .05, .5, .95, 1.]).tolist(),
        'key_logit_abs_quantiles': np.quantile(np.abs(arrays['cached_key_logits']), [0., .05, .5, .95, 1.]).tolist(),
        'cache_recompute_forecast_max_abs': float(np.abs(arrays['recomputed_sign_prediction'] - arrays['deploy_prediction']).max()),
        'cache_recompute_key_logit_max_abs': float(np.abs(arrays['recomputed_key_logits'] - arrays['cached_key_logits']).max())}
    if mode in ('gumbel', 'st'):
        result['analytic_gumbel_flip_probability'] = {
            'query_mean': float(analytic_flip_probability(arrays['recomputed_query_logits']).mean()),
            'used_key_mean': float(analytic_flip_probability(arrays['recomputed_key_logits'][used_keys]).mean()) if used_keys.any() else None,
            'derivation': 'Two independent standard Gumbels have a logistic difference. The [I,-I] decision is flipped against deterministic sign with probability sigmoid(-2*abs(I)); temperature divides both classes and does not change their argmax. Floating-point uniforms are clamped in the actual implementation.',
            'scope': 'Per-bit conditional noise probability, not forecasting effect or independent validation'}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-result', type=Path, required=True)
    parser.add_argument('--fresh-result', type=Path, required=True)
    parser.add_argument('--method-lock', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gpu-cap-seconds', type=float, default=300.)
    args = parser.parse_args()
    if not args.tag.replace('_', '').replace('-', '').isalnum() or not 0 < args.gpu_cap_seconds <= 300:
        raise ValueError('TAG_OR_GPU_CAP_CONTRACT')
    bm.en.configure()
    dt.access_guard()
    folder = rt.HERE / 'train_deploy_gap_v1' / args.tag
    folder.mkdir(parents=True, exist_ok=False)
    cache = rt.CACHE / 'train_deploy_gap_v1' / args.tag
    cache.mkdir(parents=True, exist_ok=False)
    limit = rt.Limit('train_deploy_gap_' + args.tag, gpu=True)
    result = {'status': 'RUNNING', 'expected_conditions': 48, 'draws_per_condition': DRAWS,
        'prespecified_windows': 'Last eight of the original 256 FIT origins, indices 248..255; no outcome filtering',
        'conditions': [], 'failed_conditions': [], 'sources': [],
        'limitations': ['Same selected checkpoint and FIT sample; this is a mechanism diagnosis, not generalization.',
                        'Eight stochastic draws are not independent trained models or validation samples.',
                        'FIT windows may overlap in time; their count is not an independent sample count.',
                        'Difference in forecast or code does not establish the cause of prior forecasting failures.',
                        'B_TANH is actual selected-update noiseless training tanh, separate from inference-only fixed-.25 tanh ablation.'],
        'method_lock': rt._artifact(args.method_lock), 'script_sha256': rt.sha(__file__)}
    forecaster = None
    status, error = 'COMPLETE', None
    def check():
        limit.check()
        if limit._elapsed() > args.gpu_cap_seconds:
            raise rt.BudgetHeld('TRAIN_DEPLOY_GPU_PROCESS_CAP_' + str(args.gpu_cap_seconds))
    try:
        lock = rt.read(args.method_lock)
        for relative, expected in lock['source_hashes'].items():
            if rt.sha(rt.ROOT / relative) != expected:
                raise ValueError('FROZEN_SOURCE_CHANGED: ' + relative)
        source_hashes = rt.source_snapshot('train_deploy_gap_' + args.tag)
        if source_hashes != lock['source_hashes']:
            raise ValueError('FROZEN_SOURCE_SCOPE_MISMATCH')
        result['source_hashes'] = source_hashes
        with Path(__file__).open('rb') as source, (cache / 'PRODUCER.py').open('xb') as copy:
            copy.write(source.read())
        source_results = []
        for path in (args.dev_result, args.fresh_result):
            provenance = rt._artifact(path)
            result['sources'].append(provenance)
            source_result = rt.read(path)
            if source_result['source_hashes'] != source_hashes:
                raise ValueError('PRIMARY_SOURCE_SCOPE_MISMATCH')
            source_results.append((source_result, provenance))
        cases = expected_cases(source_results)
        forecaster = bm.en.ex.previous.FrozenBolt()
        bm.en.replace_normalizer(forecaster)
        before_backbone = rt.state_hash(forecaster.model)
        result['backbone_before_sha256'] = before_backbone
        if any(value['backbone_before_sha256'] != before_backbone or
               value['backbone_after_sha256'] != before_backbone for value, _ in source_results):
            raise ValueError('PRIMARY_BACKBONE_HASH_MISMATCH')
        check()
        for case, data_record, provenance in cases:
            identity = case['dataset'] + '_' + case['label']
            model = None
            try:
                inputs = load_fit_inputs(case, data_record)
                inputs['selected_update'] = case['training']['selected_update']
                model, checkpoint = load_fixed_model(case, 'cuda')
                arrays, evidence = measure_fixed_model(model, inputs, forecaster, check)
                independent_eligibility = dt.causal_bank_eligibility(arrays['origins'], arrays['bank_origins'], length=512)
                independent_eligibility &= arrays['bank_valid'][None, None]
                if not np.array_equal(arrays['eligible'], independent_eligibility):
                    raise ValueError('INDEPENDENT_ROW_ELIGIBILITY_MISMATCH')
                if rt.sha(rt.ROOT / checkpoint['path']) != checkpoint['sha256']:
                    raise ValueError('CHECKPOINT_FILE_CHANGED')
                artifact = rt.save_npz(cache / (identity + '.npz'), **arrays)
                record = {'dataset': case['dataset'], 'seed': case['seed'], 'method': case['method'],
                    'label': case['label'], 'source_result': provenance, 'selected_checkpoint': checkpoint,
                    'inputs': inputs['provenance'], 'evidence': evidence,
                    'metrics': summarize(arrays, model._mode()), 'arrays': artifact}
                rt.write(folder / (identity + '.json'), record)
                result['conditions'].append(record)
                print(rt.json_line({'train_deploy_condition': identity, 'completed': len(result['conditions']),
                    'deploy_mse': record['metrics']['deploy_fit']['mse'],
                    'train_draw_mean_mse': record['metrics']['train_draw_mean_mse'],
                    'selected_update': evidence['primary_update'], 'temperature': evidence['temperature']}), flush=True)
            except Exception as exc:
                if isinstance(exc, rt.BudgetHeld):
                    raise
                failure = {'identity': identity, 'error_type': type(exc).__name__, 'error': str(exc),
                           'traceback': traceback.format_exc()}
                result['failed_conditions'].append(failure)
                rt.write(folder / (identity + '_FAILED.json'), failure)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            check()
        result['backbone_after_sha256'] = rt.state_hash(forecaster.model)
        if result['backbone_after_sha256'] != before_backbone:
            raise ValueError('FROZEN_BACKBONE_CHANGED')
        for relative, expected in source_hashes.items():
            if rt.sha(rt.ROOT / relative) != expected:
                raise ValueError('FROZEN_SOURCE_CHANGED_AFTER: ' + relative)
        if rt.sha(__file__) != result['script_sha256']:
            raise ValueError('ACTIVE_DIAGNOSTIC_SCRIPT_CHANGED')
        if result['failed_conditions'] or len(result['conditions']) != 48:
            status = 'FAILED_PRESERVED'
    except Exception as exc:
        error = str(exc)
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        result['terminal_error'] = {'error_type': type(exc).__name__, 'error': error,
                                    'traceback': traceback.format_exc()}
        if forecaster is not None:
            result['backbone_after_sha256'] = rt.state_hash(forecaster.model)
    result['status'] = status
    result['actual_completed_conditions'] = len(result['conditions'])
    result['uncompleted_conditions'] = 48 - len(result['conditions'])
    result['resource_receipt'] = limit.close(status, error)
    rt.write(folder / 'RESULT.json', result)
    print(rt.json_line({'train_deploy_terminal': str(folder / 'RESULT.json'), 'status': status,
                        'actual_completed_conditions': len(result['conditions'])}), flush=True)
    return 0 if status == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
