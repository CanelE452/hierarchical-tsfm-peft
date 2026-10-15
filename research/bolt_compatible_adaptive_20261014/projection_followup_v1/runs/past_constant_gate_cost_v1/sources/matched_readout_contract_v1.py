"""Passed-array contracts only; no files, model loaders, solver or execution."""
import numpy as np

DATASETS = {'robin': (17, 4, 'val', 30), 'beijing_aotizhongxin': (11, 2, 'eval', 146)}
SEEDS = (92601, 92602)
ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10.)


def exact(left, right, label):
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
        raise ValueError('EXACT_OLD_INPUT_REQUIRED:' + label)


def validate_raw_split(dataset, split, values, mean, std):
    c, k, final, final_n = DATASETS[dataset]
    n = {'fit': 256, 'tune': 64, final: final_n}[split]
    history = values['history']
    if history.shape != (n, 512, c) or history.dtype != np.float32 or not np.isfinite(history).all():
        raise ValueError('EXACT_FINITE_NORMALIZED_RAW_N512C_REQUIRED')
    exact(values['raw_last'], history[:, -1], split + ':raw_last_vs_past')
    target, observed = values['target'], values['observed']
    if (target.shape != (n, 48, c) or target.dtype != np.float32 or observed.dtype != np.bool_
            or observed.shape != target.shape or not np.isfinite(target[observed]).all()):
        raise ValueError('FP32_OBSERVED_Y_BOOL_MASK_N48C_REQUIRED')
    past_mask = values['history_observed']
    if past_mask.shape != history.shape or past_mask.dtype != np.bool_:
        raise ValueError('EXACT_BOOL_PAST_OBSERVABILITY_N512C_REQUIRED')
    if (mean.shape != (c,) or std.shape != (c,) or mean.dtype != np.float32 or std.dtype != np.float32
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('SAME_FINITE_ORIGINAL_FIT_STATS_REQUIRED')
    if values['origins'].shape != (n,) or values['origins'].dtype != np.int64:
        raise ValueError('EXACT_ORIGINAL_ORIGIN_INT64_N_REQUIRED')
    return {'dataset': dataset, 'split': split, 'windows': n, 'raw_channels': c, 'latent_channels': k,
            'history_length': 512, 'forecast_length': 48, 'raw_input_validated': True,
            'current_anchor_exactly_matches_past': True, 'future_oracle_required': False}


def validate_split(dataset, split, bridge_values, cached_values, mean, std):
    record = validate_raw_split(dataset, split, bridge_values, mean, std)
    latent = cached_values['latent_past']
    if latent.shape != (record['windows'], 512, record['latent_channels']) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError('EXACT_FINITE_CACHED_PAST_E_N512K_REQUIRED')
    for name in ('target', 'observed', 'origins', 'raw_last', 'history_observed', 'raw_forecast'):
        exact(cached_values[name], bridge_values[name], split + ':' + name)
    exact(cached_values['mean'], mean, split + ':mean')
    exact(cached_values['std'], std, split + ':std')
    exact(cached_values['z_last'], latent[:, -1:], split + ':z_last_vs_past')
    return {**record, 'old_cached_latent_and_input_exact': True,
            'target_mask_anchor_stats_exactly_matched': True}


def validate_stage_reuse(stage, dataset, seed):
    if (stage.get('status') != 'COMPLETE' or stage.get('dataset') != dataset or stage.get('seed') != seed
            or seed not in SEEDS or stage.get('old_E_D_F_updated') is not False
            or stage.get('F_loaded') is not False or stage.get('E_forward_replayed') is not False
            or stage.get('codec_before') != stage.get('codec_after')
            or stage.get('codec_before') != stage.get('codec_terminal_hash') or not stage.get('codec_before')):
        raise ValueError('EXACT_COMPLETE_ORIGINAL_STAGE_N2_LINEAGE_REQUIRED')
    row = stage['families']['N2']
    candidates = row.get('alpha_grid', [])
    if (row.get('status') != 'COMPLETE' or row.get('diagnostic_only') is not False
            or tuple(v.get('alpha') for v in candidates) != ALPHAS
            or not all(v.get('tune_actual_Y_mse') is not None and np.isfinite(v['tune_actual_Y_mse']) for v in candidates)):
        raise ValueError('COMPLETE_SEVEN_ALPHA_ACTUAL_Y_N2_REQUIRED')
    selected = min(range(7), key=lambda i: candidates[i]['tune_actual_Y_mse'])
    chosen = candidates[selected]
    if (row.get('selected_index') != selected or row.get('selected_alpha') != chosen['alpha']
            or row.get('selected_model_sha256') != chosen['model_sha256']
            or row.get('selected_coefficient_artifact') != chosen['coefficient_artifact']):
        raise ValueError('ACTUAL_Y_TUNE_EARLIEST_N2_SELECTION_REQUIRED')
    return row
