"""D2 fixed E/F/GELU-hidden decoder refit from a caller-verified array bundle.

No old campaign import, data/CP loader, model loader, optimizer or GPU call.
The CLI describes the prepared API. Root must authorize scientific execution
after audited Bi completion and supply its existing resource limit.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10.)
SPLIT_ROLES = {'fit': 'training', 'tune': 'model_selection',
               'val': 'reused_development', 'eval': 'exposed_evaluation'}
DECODER_KEYS = ('codec.decode.weight', 'codec.decode.bias', 'residual.0.weight',
                'residual.0.bias', 'residual.2.weight', 'residual.2.bias')


@dataclass(frozen=True)
class DecoderState:
    arrays: dict

    @classmethod
    def from_codec_state(cls, state):
        arrays = {}
        for key in DECODER_KEYS:
            if key not in state:
                raise ValueError('EXACT_ORIGINAL_NONLINEAR_DECODER_KEY: ' + key)
            value = state[key]
            if hasattr(value, 'detach'):
                if value.device.type != 'cpu':
                    raise ValueError('CPU_DECODER_STATE_REQUIRED')
                value = value.detach().numpy()
            value = np.asarray(value)
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('ORIGINAL_DECODER_FP32_FINITE_REQUIRED: ' + key)
            arrays[key] = value.copy()
            arrays[key].setflags(write=False)
        weight = arrays['codec.decode.weight']
        if weight.ndim != 2:
            raise ValueError('ORIGINAL_DECODER_WEIGHT_SHAPE')
        c, k = weight.shape
        shapes = ((c, k), (c,), (32, k), (32,), (c, 32), (c,))
        if min(c, k) < 1 or any(arrays[key].shape != shape for key, shape in zip(DECODER_KEYS, shapes)):
            raise ValueError('ORIGINAL_DECODER_HIDDEN32_SHAPE')
        return cls(arrays)

    @property
    def k(self):
        return self.arrays['codec.decode.weight'].shape[1]

    @property
    def channels(self):
        return self.arrays['codec.decode.weight'].shape[0]

    @property
    def outgoing_weight(self):
        return np.concatenate((self.arrays['codec.decode.weight'].T,
                               self.arrays['residual.2.weight'].T), axis=0)


def _cpu_tensor(value):
    import torch
    return torch.from_numpy(np.array(value, copy=True, order='C'))


def _latent_shapes(z_future, z_last, state):
    if z_future.dtype != np.float32 or z_last.dtype != np.float32:
        raise ValueError('ABSOLUTE_LATENT_FP32_REQUIRED')
    if z_future.ndim != 3 or z_last.shape != (len(z_future), 1, state.k) or z_future.shape[-1] != state.k:
        raise ValueError('ABSOLUTE_Z_FUTURE_Z_LAST_SHAPE')
    if not np.isfinite(z_future).all() or not np.isfinite(z_last).all():
        raise ValueError('ABSOLUTE_LATENT_NONFINITE')


def decoder_features(z_future, z_last, state):
    """No targets/masks: [zF-zlast, GELU(AzF+b)-GELU(Azlast+b)]."""
    from torch.nn import functional as F
    _latent_shapes(z_future, z_last, state)
    weight = _cpu_tensor(state.arrays['residual.0.weight'])
    bias = _cpu_tensor(state.arrays['residual.0.bias'])
    hidden_future = F.gelu(F.linear(_cpu_tensor(z_future), weight, bias), approximate='none').numpy()
    hidden_last = F.gelu(F.linear(_cpu_tensor(z_last), weight, bias), approximate='none').numpy()
    features = np.concatenate((z_future - z_last, hidden_future - hidden_last), axis=-1)
    if not np.isfinite(features).all():
        raise ValueError('NONFINITE_FROZEN_DECODER_FEATURES')
    return features


def original_anchored_prediction(z_future, z_last, raw_last, state):
    """Preserve old FP32 operation order, including mathematically canceled biases."""
    from torch.nn import functional as F
    _latent_shapes(z_future, z_last, state)
    tensors = {key: _cpu_tensor(value) for key, value in state.arrays.items()}

    def decode(z):
        linear = F.linear(_cpu_tensor(z), tensors['codec.decode.weight'], tensors['codec.decode.bias'])
        hidden = F.gelu(F.linear(_cpu_tensor(z), tensors['residual.0.weight'],
                                tensors['residual.0.bias']), approximate='none')
        return linear + F.linear(hidden, tensors['residual.2.weight'], tensors['residual.2.bias'])

    return (_cpu_tensor(raw_last)[:, None] + decode(z_future) - decode(z_last)).numpy()


def factored_original_prediction(z_future, z_last, raw_last, state):
    return raw_last[:, None] + decoder_features(z_future, z_last, state) @ state.outgoing_weight


@dataclass(frozen=True)
class OutputHead:
    weight: np.ndarray
    scaled_weight: np.ndarray
    feature_rms: np.ndarray
    observed_counts: np.ndarray
    alpha: float

    @property
    def missing_fit_channels(self):
        return np.flatnonzero(self.observed_counts == 0).tolist()

    def predict(self, features, raw_last):
        return raw_last[:, None] + np.asarray(features, np.float32) @ self.weight

    def decoder_state(self, original):
        arrays = {key: value.copy() for key, value in original.arrays.items()}
        arrays['codec.decode.weight'] = self.weight[:original.k].T.copy()
        arrays['residual.2.weight'] = self.weight[original.k:].T.copy()
        return arrays


def fit_output_head(fit_features, fit_target, fit_last, fit_observed, alpha, *, fallback_weight=None):
    """FIT-only RMS, no centering/intercept; masked per-channel FP64 ridge.

    Each channel minimizes mean(observed residual squared error) +
    alpha * squared scaled-head norm. Zero-observation channels retain original
    outgoing weights when supplied; otherwise their outgoing weights are zero.
    """
    features = np.asarray(fit_features, np.float64)
    target = np.asarray(fit_target, np.float64)
    last = np.asarray(fit_last, np.float64)
    observed = np.asarray(fit_observed)
    if features.ndim != 3 or target.ndim != 3 or target.shape != observed.shape:
        raise ValueError('HEAD_FIT_SHAPE')
    if features.shape[:2] != target.shape[:2] or last.shape != (len(target), target.shape[-1]):
        raise ValueError('HEAD_FIT_LAST_OR_WINDOW_SHAPE')
    if not features.size or observed.dtype != np.bool_ or not np.isfinite(features).all():
        raise ValueError('HEAD_FIT_FINITE_FEATURES_BOOL_MASK_REQUIRED')
    if not np.isfinite(last).all() or not np.isfinite(target[observed]).all() or alpha < 0 or not np.isfinite(alpha):
        raise ValueError('HEAD_FIT_NONFINITE_OR_NEGATIVE_ALPHA')
    flat = features.reshape(-1, features.shape[-1])
    rms = np.sqrt(np.mean(flat ** 2, axis=0))
    rms[rms == 0] = 1.
    scaled = flat / rms
    residual = (target - last[:, None]).reshape(-1, target.shape[-1])
    mask = observed.reshape(residual.shape)
    counts = mask.sum(axis=0)
    weights = np.zeros((scaled.shape[1], target.shape[-1]), np.float64)
    if fallback_weight is not None:
        if fallback_weight.shape != weights.shape or not np.isfinite(fallback_weight).all():
            raise ValueError('HEAD_FALLBACK_SHAPE_OR_NONFINITE')
        weights[:] = np.asarray(fallback_weight, np.float64) * rms[:, None]
    for channel in np.flatnonzero(counts):
        x, y = scaled[mask[:, channel]], residual[mask[:, channel], channel]
        if alpha == 0:
            weights[:, channel] = np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            gram = x.T @ x / len(x) + alpha * np.eye(x.shape[1])
            weights[:, channel] = np.linalg.solve(gram, x.T @ y / len(x))
    effective = (weights / rms[:, None]).astype(np.float32)
    if fallback_weight is not None:
        effective[:, counts == 0] = fallback_weight[:, counts == 0]
    if not np.isfinite(effective).all():
        raise ValueError('HEAD_SOLVE_NONFINITE')
    return OutputHead(effective, weights.astype(np.float32), rms, counts, float(alpha))


def objective_targets(target, raw_forecast):
    return {'Y': target, 'RAW': raw_forecast, 'MIXED': .5 * target + .5 * raw_forecast}


def observed_score(prediction, target, observed, std=None):
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * std
    error = np.where(observed, error, 0.)
    count = observed.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error ** 2).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    return {'mse': float(mse[active].mean()) if active.any() else None,
            'mae': float(mae[active].mean()) if active.any() else None,
            'per_channel_mse': [float(v) if a else None for v, a in zip(mse, active)],
            'per_channel_mae': [float(v) if a else None for v, a in zip(mae, active)],
            'observed_counts': count.tolist(), 'active_channels': int(active.sum())}


def select_alpha(rows):
    if not rows or any(row['tune_mse'] is None or not np.isfinite(row['tune_mse']) for row in rows):
        raise ValueError('TUNE_OBSERVED_FINITE_SCORE_REQUIRED')
    return min(range(len(rows)), key=lambda index: rows[index]['tune_mse'])


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def decoder_state_sha256(state):
    digest = hashlib.sha256()
    for key in DECODER_KEYS:
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state.arrays[key]).tobytes())
    return digest.hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save(output, name, arrays):
    path = output / name
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': name, 'sha256': _sha(path), 'bytes': path.stat().st_size,
            'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                             'sha256': array_sha256(value)} for key, value in arrays.items()}}


def _difference(left, right):
    error = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
            'max_abs': float(np.max(np.abs(error)))}


def _validate(splits, mean, std, state, source_record, synthetic):
    if not {'fit', 'tune'} <= set(splits) or set(splits) - set(SPLIT_ROLES):
        raise ValueError('SPLIT_SCOPE: FIT/TUNE/reused VAL/already-exposed EVAL only')
    if 'val' in splits and 'eval' in splits:
        raise ValueError('SPLIT_SCOPE: one reused evaluation role per dataset')
    if mean.shape != (state.channels,) or std.shape != mean.shape or mean.dtype != np.float32 or std.dtype != np.float32:
        raise ValueError('EXACT_STORED_FP32_STATS_REQUIRED')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('STATS_NONFINITE_OR_NONPOSITIVE')
    for split, values in splits.items():
        required = ('z_future', 'z_last', 'raw_last', 'original_prediction', 'raw_forecast', 'target', 'observed', 'origins')
        if any(key not in values for key in required):
            raise ValueError('ABSOLUTE_LATENT_BUNDLE_REQUIRED: ' + split)
        future, last = values['z_future'], values['z_last']
        _latent_shapes(future, last, state)
        n, h, _ = future.shape
        if not n or not h or values['raw_last'].shape != (n, state.channels):
            raise ValueError('CURRENT_LAST_OR_EMPTY_SHAPE: ' + split)
        for key in ('raw_last', 'original_prediction', 'raw_forecast', 'target'):
            value = values[key]
            if value.dtype != np.float32:
                raise ValueError('EXACT_OLD_FP32_BUNDLE_REQUIRED: ' + split + '/' + key)
            if key != 'raw_last' and value.shape != (n, h, state.channels):
                raise ValueError('ORIGINAL_C_FORECAST_TARGET_SHAPE: ' + split)
        mask, origins = values['observed'], values['origins']
        if mask.shape != (n, h, state.channels) or mask.dtype != np.bool_:
            raise ValueError('EXACT_OLD_OBSERVED_BOOL_MASK_REQUIRED: ' + split)
        if origins.shape != (n,) or origins.dtype != np.int64 or np.any(np.diff(origins) <= 0):
            raise ValueError('EXACT_OLD_ORDERED_INT64_ORIGINS_REQUIRED: ' + split)
        if any(not np.isfinite(values[key]).all() for key in ('raw_last', 'original_prediction', 'raw_forecast')):
            raise ValueError('NONFINITE_PAST_FORECAST_BUNDLE: ' + split)
        if not np.isfinite(values['target'][mask]).all():
            raise ValueError('NONFINITE_OBSERVED_TARGET: ' + split)
    if not splits['fit']['observed'].any() or not splits['tune']['observed'].any():
        raise ValueError('FIT_TUNE_OBSERVED_REQUIRED')
    if synthetic:
        if source_record.get('fixture_only') is not True:
            raise ValueError('SYNTHETIC_FIXTURE_ONLY_MARK_REQUIRED')
        return
    verification = source_record.get('verification', {})
    required = ('source_checkpoint', 'source_snapshots', 'same_original_main_E_F',
                'absolute_latents', 'original_stats_origins_masks', 'split_roles')
    if any(verification.get(key) is not True for key in required):
        raise ValueError('CALLER_SOURCE_VERIFICATION_REQUIRED')
    if source_record.get('bi_complete') is not True:
        raise ValueError('AUDITED_BI_COMPLETION_REQUIRED')
    if source_record.get('architecture') != 'ANCHOR_NONLINEAR_D' or source_record.get('input_length') != 512:
        raise ValueError('ORIGINAL_ANCHOR_ARCHITECTURE_L512_REQUIRED')
    if source_record.get('decoder_state_sha256') != decoder_state_sha256(state):
        raise ValueError('CALLER_DECODER_STATE_SHA256_MISMATCH')
    specifications = {'robin': (17, 4, 'val', 30), 'beijing_aotizhongxin': (11, 2, 'eval', 146)}
    if source_record.get('dataset') not in specifications or source_record.get('seed') not in (92601, 92602):
        raise ValueError('ORIGINAL_ROBIN_BEIJING_SEED_REQUIRED')
    c, k, evaluation, count = specifications[source_record['dataset']]
    if state.channels != c or state.k != k or set(splits) != {'fit', 'tune', evaluation}:
        raise ValueError('ORIGINAL_DATASET_CHANNEL_K_SPLITS_REQUIRED')
    counts = {'fit': 256, 'tune': 64, evaluation: count}
    if any(values['z_future'].shape[:2] != (counts[split], 48) for split, values in splits.items()):
        raise ValueError('ORIGINAL_WINDOW_COUNT_H48_REQUIRED')
    artifacts = source_record.get('source_artifacts', [])
    aliases = {artifact.get('alias') for artifact in artifacts}
    if not {'MAIN_CP', 'MAIN_PREDICTION', 'RAW_PREDICTION', 'ORIGINAL_SOURCE'} <= aliases:
        raise ValueError('SOURCE_ARTIFACT_ALIASES_REQUIRED')
    for artifact in artifacts:
        sha = artifact.get('sha256', '')
        if not artifact.get('path') or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha):
            raise ValueError('SOURCE_ARTIFACT_PATH_SHA256_REQUIRED')
        if artifact.get('verified_sha256') != sha:
            raise ValueError('SOURCE_ARTIFACT_VERIFIED_SHA256_MISMATCH')


class ProcessGuard:
    def __init__(self, shared_limit):
        import psutil
        self.process = psutil.Process()
        self.shared_limit = shared_limit

    def check(self):
        if time.time() - self.process.create_time() >= 300:
            raise RuntimeError('D2_PROCESS_WALL_300S')
        if self.process.memory_info().rss > 8 * 1024 ** 3:
            raise RuntimeError('D2_PROCESS_RSS_8GIB')
        self.shared_limit.check()


def _scores(prediction, values, std):
    return {'normalized': observed_score(prediction, values['target'], values['observed']),
            'physical': observed_score(prediction, values['target'], values['observed'], std),
            'raw_teacher_normalized': observed_score(prediction, values['raw_forecast'], values['observed'])}


def _csv_rows(method, objective, alpha, selected, split, scores):
    rows = []
    for scale, score in scores.items():
        common = {'method': method, 'objective': objective, 'alpha': alpha, 'selected': selected,
                  'split': split, 'split_role': SPLIT_ROLES[split], 'metric_scale': scale}
        rows.append({**common, 'channel': 'macro', 'mse': score['mse'], 'mae': score['mae'],
                     'observed_count': sum(score['observed_counts'])})
        for channel, (mse, mae, count) in enumerate(zip(score['per_channel_mse'], score['per_channel_mae'], score['observed_counts'])):
            rows.append({**common, 'channel': channel, 'mse': mse, 'mae': mae, 'observed_count': count})
    return rows


def _report(result):
    lines = ['# 동결 E/F·hidden 디코더 출력 재적합', '',
             f"상태: {result['status']}. scientific_execution={result['scientific_execution']}.", '',
             '합성 CPU fixture 결과는 실제 자료의 예측 성능 근거가 아니다.' if not result['scientific_execution']
             else '원 MAIN의 E/F와 GELU hidden을 동결한 개발 증거다. 독립 holdout 결과가 아니다.', '',
             '원 anchor x_last+D(z_F)−D(z_last)를 유지한다. absolute z_F/z_last에서 latent·hidden 차이를 계산한다.',
             'FIT feature RMS만 쓰며 평균 중심화·절편을 넣지 않는다. 정규화·원단위 오차를 별도로 기록한다.',
             'Y / RAW 과거 기반 forecast / 0.5 혼합의 세 목적은 동일 FIT observed mask를 사용한다.',
             'RAW distillation은 실제 미래 정답 oracle이 아니다. 모든 alpha는 TUNE 실제 Y의 채널 macro 정규화 MSE로 선택한다.',
             'E/F를 위한 추가 최적화는 없지만 D 출력 가중치를 실제로 새 적합하고 TUNE 선택한다.',
             '보편적 nonlinear D 하한이나 최적 decoder/전체 방법의 최선 성능을 입증하지 않는다.', '',
             '```text', '목적      선택 alpha        TUNE normalized MSE']
    for objective, row in result['refits'].items():
        if 'selected_alpha' in row:
            lines.append(f"{objective:<9} {row['selected_alpha']:<19g} {row['alpha_grid'][row['selected_index']]['tune_mse']:.9g}")
    lines.extend(['```', '', '원 D 계산·factored D·저장 original prediction의 FP32 차이는 RESULT.json에 별도로 남겼다.',
                  '전체 alpha와 선택 출력의 채널별 MSE/MAE는 METRICS.csv, 실제 입력·원/새 출력 가중치는 NPZ에 보존한다.',
                  f"총 helper wall seconds: {result.get('wall_seconds', 0.):.6f}."])
    if result.get('failure'):
        lines.extend(['', '실패한 새 폴더와 이미 기록한 파일은 보존했다.', json.dumps(result['failure'], ensure_ascii=False)])
    return '\n'.join(lines) + '\n'


def run_decoder_refit(output, *, splits, codec_state, mean, std, source_record,
                      shared_limit=None, execution_authorized=False, synthetic=False,
                      parity_atol=2e-5, parity_rtol=2e-6):
    """No source reads. Caller verifies/constructs the exact original bundle first.

    Scientific calls require audited Bi completion attestation, explicit root
    gate, original source/CP/array hashes and the existing shared Limit. The
    caller owns external watchdog and terminal resource receipt. Synthetic calls
    require fixture_only=True and are labeled non-scientific.
    """
    if not synthetic and not execution_authorized:
        raise ValueError('EXECUTION_GATE_OFF: root launches after audited Bi completion')
    if not synthetic and shared_limit is None:
        raise ValueError('SHARED_LIMIT_REQUIRED')
    if min(parity_atol, parity_rtol) < 0 or not np.isfinite([parity_atol, parity_rtol]).all():
        raise ValueError('FINITE_NONNEGATIVE_PARITY_TOLERANCE_REQUIRED')
    state = DecoderState.from_codec_state(codec_state)
    mean, std = np.asarray(mean), np.asarray(std)
    _validate(splits, mean, std, state, source_record, synthetic)
    output = Path(output).resolve()
    if not synthetic and not output.is_relative_to(CACHE):
        raise ValueError('SCIENTIFIC_OUTPUT_OUTSIDE_FOLLOWUP_CACHE')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    check = ProcessGuard(shared_limit).check if not synthetic else (shared_limit.check if shared_limit else lambda: None)
    result = {'status': 'RUNNING', 'scientific_execution': not synthetic,
              'source_record': source_record, 'source_file_sha256': _sha(__file__),
              'source_decoder_state_sha256': decoder_state_sha256(state),
              'storage_scale': 'normalized', 'normalization_origin_split': 'fit',
              'split_roles': {split: SPLIT_ROLES[split] for split in splits},
              'K': state.k, 'C': state.channels, 'frozen': ['original_MAIN_E', 'original_F', 'decoder_hidden_weight_bias'],
              'features': 'concat(z_F-z_last, GELU(A*z_F+b)-GELU(A*z_last+b))',
              'current_state_bypass_C': state.channels, 'strict_K_only_compression': False,
              'fit_feature_scaling': 'FIT-only per-feature RMS; no mean centering; zero RMS replaced by one',
              'intercept': False, 'solver_dtype': 'float64', 'effective_weight_prediction_dtype': 'float32',
              'ridge_objective': 'per-channel mean masked squared error + alpha*scaled_weight_squared_norm',
              'alpha_grid': list(ALPHAS), 'selection_rule': 'TUNE actual-Y normalized observed channel-macro MSE; earliest exact tie',
              'fit_mask_rule': 'same actual FIT observed mask for Y, RAW past-forecast, .5 mixed',
              'teacher_is_future_ground_truth_oracle': False, 'independent_holdout': False,
              'universal_nonlinear_decoder_lower_bound': False, 'new_decoder_optimization': True,
              'parity_tolerance': {'atol': parity_atol, 'rtol': parity_rtol},
              'reference_comparisons': {}, 'refits': {}, 'artifacts': []}
    location = {'phase': 'source', 'split': None, 'objective': None, 'alpha': None}
    metrics_rows, features = [], {}
    try:
        check()
        result['artifacts'].append(_save(output, 'SOURCE_DECODER.npz', state.arrays))
        for split, values in splits.items():
            location.update(phase='inputs_and_parity', split=split)
            check()
            arrays = {**values, 'mean': mean, 'std': std}
            result['artifacts'].append(_save(output, 'INPUT_' + split + '.npz', arrays))
            features[split] = decoder_features(values['z_future'], values['z_last'], state)
            source = original_anchored_prediction(values['z_future'], values['z_last'], values['raw_last'], state)
            factored = values['raw_last'][:, None] + features[split] @ state.outgoing_weight
            comparisons = {'source_D_vs_stored_original': _difference(source, values['original_prediction']),
                           'factored_original_vs_source_D': _difference(factored, source),
                           'factored_original_vs_stored_original': _difference(factored, values['original_prediction']),
                           'source_D_parity_pass': bool(np.allclose(source, values['original_prediction'], atol=parity_atol, rtol=parity_rtol)),
                           'factored_D_parity_pass': bool(np.allclose(factored, source, atol=parity_atol, rtol=parity_rtol)),
                           'scores': {}}
            result['reference_comparisons'][split] = comparisons
            baselines = {'ORIGINAL_D': values['original_prediction'], 'SOURCE_D_FP32': source,
                         'FACTORED_ORIGINAL_D': factored, 'RAW': values['raw_forecast']}
            result['artifacts'].append(_save(output, 'REFERENCE_' + split + '.npz',
                                              {**baselines, 'features': features[split], 'origins': values['origins']}))
            for name, prediction in baselines.items():
                scores = _scores(prediction, values, std)
                comparisons['scores'][name] = scores
                metrics_rows.extend(_csv_rows(name, '', '', True, split, scores))
            if not comparisons['source_D_parity_pass'] or not comparisons['factored_D_parity_pass']:
                raise ValueError('ORIGINAL_DECODER_NORMALIZED_FP32_PARITY: ' + split)
            check()
        targets = objective_targets(splits['fit']['target'], splits['fit']['raw_forecast'])
        for objective, fit_target in targets.items():
            location.update(phase='ridge_fit', split='fit', objective=objective)
            objective_start = time.perf_counter()
            row = {'objective': objective, 'target_description': {
                'Y': 'actual observed future Y', 'RAW': 'RAW past-only forecast, not future-ground-truth oracle',
                'MIXED': '.5*Y+.5*RAW; same-mask equal-weight squared-loss minimizer'}[objective],
                   'alpha_grid': [], 'splits': {}}
            result['refits'][objective] = row
            heads = []
            for alpha in ALPHAS:
                location.update(alpha=alpha)
                check()
                alpha_start = time.perf_counter()
                head = fit_output_head(features['fit'], fit_target, splits['fit']['raw_last'],
                                       splits['fit']['observed'], alpha, fallback_weight=state.outgoing_weight)
                tune_prediction = head.predict(features['tune'], splits['tune']['raw_last'])
                scores = _scores(tune_prediction, splits['tune'], std)
                row['alpha_grid'].append({'alpha': alpha, 'tune_mse': scores['normalized']['mse'], 'tune_scores': scores,
                                          'wall_seconds': time.perf_counter() - alpha_start,
                                          'missing_fit_channels_original_fallback': head.missing_fit_channels})
                heads.append(head)
                result['artifacts'].append(_save(output, f'{objective}_alpha_{alpha:g}_HEAD.npz', {
                    'outgoing_weight': head.weight, 'scaled_weight': head.scaled_weight, 'feature_rms': head.feature_rms,
                    'fit_observed_counts': head.observed_counts, 'alpha': np.asarray(alpha)}))
                check()
            selected_index = select_alpha(row['alpha_grid'])
            selected = heads[selected_index]
            row.update(selected_alpha=selected.alpha, selected_index=selected_index,
                       feature_rms_sha256=array_sha256(selected.feature_rms), selected_weight_sha256=array_sha256(selected.weight),
                       missing_fit_channels_original_fallback=selected.missing_fit_channels)
            for index, candidate in enumerate(row['alpha_grid']):
                metrics_rows.extend(_csv_rows('REFIT_' + objective, objective, candidate['alpha'],
                                             index == selected_index, 'tune', candidate['tune_scores']))
            result['artifacts'].append(_save(output, objective + '_SELECTED_DECODER.npz', selected.decoder_state(state)))
            for split, values in splits.items():
                location.update(phase='selected_prediction', split=split, alpha=selected.alpha)
                check()
                prediction = selected.predict(features[split], values['raw_last'])
                scores = _scores(prediction, values, std)
                row['splits'][split] = scores
                result['artifacts'].append(_save(output, objective + '_SELECTED_' + split + '.npz',
                                                  {'prediction': prediction, 'origins': values['origins']}))
                if split != 'tune':
                    metrics_rows.extend(_csv_rows('REFIT_' + objective, objective, selected.alpha, True, split, scores))
                check()
            row['wall_seconds'] = time.perf_counter() - objective_start
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={**location, 'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        result['wall_seconds'] = time.perf_counter() - started
        with (output / 'METRICS.csv').open('x', encoding='utf-8', newline='') as handle:
            fieldnames = ('method', 'objective', 'alpha', 'selected', 'split', 'split_role', 'metric_scale',
                          'channel', 'mse', 'mae', 'observed_count')
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)
        with (output / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(_report(result))
        with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    return result


def describe():
    return {'status': 'PREPARATION_ONLY', 'actual_data_opened': False, 'actual_gpu_executed': False,
            'entrypoint': 'run_decoder_refit(fresh_output, splits=verified_absolute_latent_bundle, codec_state=exact_MAIN_CP_codec_state, mean=stored_mean, std=stored_std, source_record=verified_lineage, shared_limit=existing_limit, execution_authorized=True)',
            'input_schema': {'z_future': 'normalized float32[N,48,K] absolute original F(E(X))',
                             'z_last': 'float32[N,1,K] absolute original E(X) last', 'raw_last': 'normalized float32[N,C]',
                             'original_prediction/raw_forecast/target': 'normalized float32[N,48,C]',
                             'observed': 'bool[N,48,C], exact actual Y mask', 'origins': 'int64[N]',
                             'mean/std': 'exact stored float32[C]', 'codec_state': list(DECODER_KEYS)},
            'alpha_grid': list(ALPHAS), 'runtime': 'CPU-only FP32 GELU/features/predictions; FP64 masked per-channel ridge solve',
            'remaining_runner_responsibility': 'Root verifies exact MAIN E/F/hidden, source CP/arrays/statistics/origins/masks; reconstructs missing absolute latents only after audited Bi completion; supplies existing resource limit and external watchdog.',
            'outputs': 'SOURCE_DECODER, INPUT/REFERENCE split, every alpha head, selected decoder/prediction NPZ; METRICS.csv; RESULT.json; REPORT_KO.md'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
