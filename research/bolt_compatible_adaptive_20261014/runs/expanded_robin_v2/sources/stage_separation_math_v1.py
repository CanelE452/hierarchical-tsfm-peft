"""Pure NumPy stage-separation math. No loaders, runtime, GPU or execution CLI.

Source prepared only; fixtures and scientific solvers have not been executed.
Native data loss is channel macro over ALL observed origins/horizons. Its
per-output normal equation therefore uses alpha*N_channel, not alpha*n_hc.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10.)
LENGTH = 512
HORIZON = 48


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state_sha(kind, *values):
    digest = hashlib.sha256(kind.encode())
    for value in values:
        digest.update((array_sha(value) if isinstance(value, np.ndarray) else str(value)).encode())
    return digest.hexdigest()


def _readonly(value, dtype=np.float64):
    value = np.array(value, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def fixed_past_features(latent_past):
    value = np.asarray(latent_past)
    if value.ndim != 3 or value.shape[1] != LENGTH or not value.shape[0] or not value.shape[2]:
        raise ValueError('FINITE_NONEMPTY_N_512_K_PAST_REQUIRED')
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError('PAST_LATENT_FLOAT_FINITE_REQUIRED')
    blocks = value.reshape(len(value), 32, 16, value.shape[2]).mean(axis=2)
    return np.concatenate((blocks, value[:, -1:]), axis=1).reshape(len(value), -1)


@dataclass(frozen=True)
class FeatureState:
    center: np.ndarray
    scale: np.ndarray
    latent_channels: int
    fit_rows: int
    fit_raw_features_sha256: str

    @property
    def sha256(self):
        return _state_sha('FIT_33K_FEATURE_STATE', self.center, self.scale,
                          self.latent_channels, self.fit_rows, self.fit_raw_features_sha256)

    def transform(self, latent_past, *, require_fit_artifact=False):
        raw = fixed_past_features(latent_past)
        if raw.shape[1] != len(self.center):
            raise ValueError('FEATURE_K_MISMATCH')
        if require_fit_artifact and (len(raw) != self.fit_rows or array_sha(raw) != self.fit_raw_features_sha256):
            raise ValueError('SAME_FIT_PAST_FEATURE_ARTIFACT_REQUIRED')
        return (raw.astype(np.float64) - self.center) / self.scale


def fit_feature_state(fit_latent_past):
    raw = fixed_past_features(fit_latent_past)
    center = raw.astype(np.float64).mean(axis=0)
    scale = np.sqrt(np.mean((raw.astype(np.float64) - center) ** 2, axis=0))
    scale[scale == 0] = 1.
    return FeatureState(_readonly(center), _readonly(scale), fit_latent_past.shape[2], len(raw), array_sha(raw))


def complete_prefix_latent_mask(future_observed, latent_channels):
    mask = np.asarray(future_observed)
    if (mask.ndim != 3 or mask.dtype != np.bool_ or mask.shape[1] != HORIZON
            or not mask.shape[0] or not mask.shape[2] or latent_channels < 1):
        raise ValueError('BOOL_N_48_C_FUTURE_MASK_AND_POSITIVE_K_REQUIRED')
    complete = np.logical_and.accumulate(mask.all(axis=2), axis=1)
    return np.repeat(complete[:, :, None], latent_channels, axis=2)


@dataclass(frozen=True)
class _MaskGroup:
    columns: np.ndarray
    observed_rows: int
    x_mean: np.ndarray
    y_mean: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    projected_targets: np.ndarray
    alpha0_weight: np.ndarray
    rank: int


@dataclass(frozen=True)
class RidgeSolution:
    kind: str
    alpha: float
    coefficients: np.ndarray
    intercept: np.ndarray
    observed_counts: np.ndarray
    channel_counts: np.ndarray
    normal_equation_denominators: np.ndarray
    design_ranks: np.ndarray
    regularized_gram_conditions: np.ndarray
    feature_state_sha256: str
    with_intercept: bool

    @property
    def sha256(self):
        return _state_sha(self.kind, self.alpha, self.coefficients, self.intercept,
                          self.observed_counts, self.channel_counts, self.feature_state_sha256)

    def predict(self, design):
        x = np.asarray(design, np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients.shape[0] or not np.isfinite(x).all():
            raise ValueError('FINITE_MATCHING_RIDGE_DESIGN_REQUIRED')
        shape = self.intercept.shape
        return (x @ self.coefficients.reshape(x.shape[1], -1) + self.intercept.reshape(-1)).reshape((len(x),) + shape)


@dataclass(frozen=True)
class PreparedRidge:
    kind: str
    feature_count: int
    output_shape: tuple
    counts: np.ndarray
    channel_counts: np.ndarray
    denominators: np.ndarray
    groups: tuple
    fallback_coefficients: np.ndarray
    feature_state_sha256: str
    with_intercept: bool

    def fit(self, alpha):
        if alpha not in ALPHAS:
            raise ValueError('PREDECLARED_SEVEN_ALPHA_REQUIRED')
        outputs = int(np.prod(self.output_shape))
        weights = np.array(self.fallback_coefficients, copy=True)
        biases = np.zeros(outputs, np.float64)
        ranks = np.zeros(outputs, np.int64)
        conditions = np.full(outputs, np.nan, np.float64)
        for group in self.groups:
            cols = group.columns
            s = group.singular_values
            if alpha == 0:
                current = group.alpha0_weight
            else:
                penalty = alpha * self.denominators[cols]
                gain = s[:, None] / (s[:, None] ** 2 + penalty[None])
                current = group.right_vectors.T @ (gain * group.projected_targets)
            weights[:, cols] = current
            if self.with_intercept:
                biases[cols] = group.y_mean - group.x_mean @ current
            ranks[cols] = group.rank
            largest = float(s[0] ** 2) if len(s) else 0.
            smallest = float(s[-1] ** 2) if group.rank == self.feature_count else 0.
            if alpha == 0:
                conditions[cols] = largest / smallest if smallest > 0 else np.inf
            else:
                penalty = alpha * self.denominators[cols]
                conditions[cols] = (largest + penalty) / (smallest + penalty)
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError('RIDGE_COEFFICIENT_NONFINITE')
        return RidgeSolution(self.kind, float(alpha), _readonly(weights.reshape((self.feature_count,) + self.output_shape)),
            _readonly(biases.reshape(self.output_shape)), _readonly(self.counts.reshape(self.output_shape), np.int64),
            self.channel_counts, _readonly(self.denominators.reshape(self.output_shape)),
            _readonly(ranks.reshape(self.output_shape), np.int64), _readonly(conditions.reshape(self.output_shape)),
            self.feature_state_sha256, self.with_intercept)

    def fit_all(self):
        return tuple(self.fit(alpha) for alpha in ALPHAS)


def prepare_masked_ridge(design, target, observed, *, kind, feature_state_sha256,
                         with_intercept=True, fallback_coefficients=None):
    """Cache each identical-mask group's factorization once for all seven alphas.

    For native target[N,H,Q], data=mean_active_q sum_NH(err²)/Nq,
    penalty=alpha*mean_active_q sum_H ||scaled-W_hq||². Therefore lambda=alpha*Nq.
    Decoder target[NH,C] uses the same expression without separate H weights.
    Alpha0 computes minimum-norm centered W by lstsq; intercept is unpenalized.
    """
    x, y, mask = np.asarray(design, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or y.ndim not in (2, 3):
        raise ValueError('NONEMPTY_MATRIX_DESIGN_AND_N_OUTPUT_TARGET_REQUIRED')
    if len(x) != len(y) or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('RIDGE_TARGET_MASK_SHAPE_OR_DTYPE')
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
        raise ValueError('RIDGE_OBSERVED_VALUES_MUST_BE_FINITE')
    shape = y.shape[1:]
    flat_y, flat_mask = y.reshape(len(y), -1), mask.reshape(len(y), -1)
    counts = flat_mask.sum(axis=0)
    channel_counts = mask.sum(axis=tuple(range(mask.ndim - 1)))
    denominators = np.tile(channel_counts, shape[0]) if y.ndim == 3 else channel_counts.copy()
    outputs = flat_y.shape[1]
    fallback = np.zeros((x.shape[1], outputs), np.float64) if fallback_coefficients is None else np.asarray(fallback_coefficients, np.float64)
    if fallback.shape != (x.shape[1], outputs) or not np.isfinite(fallback).all():
        raise ValueError('RIDGE_FALLBACK_COEFFICIENT_SHAPE_OR_FINITE')
    masks = {}
    for col in np.flatnonzero(counts):
        masks.setdefault(flat_mask[:, col].tobytes(), []).append(int(col))
    groups = []
    for columns in masks.values():
        rows = flat_mask[:, columns[0]]
        gx, gy = x[rows], flat_y[np.ix_(rows, columns)]
        x_mean = gx.mean(axis=0) if with_intercept else np.zeros(x.shape[1])
        y_mean = gy.mean(axis=0) if with_intercept else np.zeros(len(columns))
        centered_x, centered_y = gx - x_mean, gy - y_mean
        alpha0, _, rank, _ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
        left, singular, right = np.linalg.svd(centered_x, full_matrices=False)
        groups.append(_MaskGroup(_readonly(columns, np.int64), int(rows.sum()), _readonly(x_mean), _readonly(y_mean),
            _readonly(singular), _readonly(right), _readonly(left.T @ centered_y), _readonly(alpha0), int(rank)))
    return PreparedRidge(kind, x.shape[1], shape, _readonly(counts, np.int64), _readonly(channel_counts, np.int64),
                         _readonly(denominators), tuple(groups), _readonly(fallback), feature_state_sha256, bool(with_intercept))


@dataclass(frozen=True)
class NativeModel:
    feature_state: FeatureState
    solution: RidgeSolution

    @property
    def sha256(self):
        return _state_sha('FIXED_NATIVE_MODEL', self.feature_state.sha256, self.solution.sha256)

    def latent_forecast(self, latent_past):
        if self.solution.kind != 'N1_LATENT':
            raise ValueError('N1_LATENT_MODEL_REQUIRED')
        innovation = self.solution.predict(self.feature_state.transform(latent_past)).astype(np.float32)
        result = np.asarray(latent_past[:, -1:], np.float32) + innovation
        if not np.isfinite(result).all():
            raise ValueError('N1_FP32_FORECAST_NONFINITE')
        return result

    def direct_forecast(self, latent_past, raw_last):
        if self.solution.kind != 'N2_DIRECT_C':
            raise ValueError('N2_DIRECT_C_MODEL_REQUIRED')
        innovation = self.solution.predict(self.feature_state.transform(latent_past)).astype(np.float32)
        last = np.asarray(raw_last, np.float32)
        if last.shape != (len(innovation), innovation.shape[-1]) or not np.isfinite(last).all():
            raise ValueError('N2_FINITE_CURRENT_C_ANCHOR_REQUIRED')
        result = last[:, None] + innovation
        if not np.isfinite(result).all():
            raise ValueError('N2_FP32_FORECAST_NONFINITE')
        return result


@dataclass(frozen=True)
class PreparedNative:
    feature_state: FeatureState
    ridge: PreparedRidge

    def fit(self, alpha):
        return NativeModel(self.feature_state, self.ridge.fit(alpha))

    def fit_all(self):
        return tuple(self.fit(alpha) for alpha in ALPHAS)


def prepare_n1(feature_state, fit_latent_past, fit_oracle_z, fit_future_observed):
    design = feature_state.transform(fit_latent_past, require_fit_artifact=True)
    oracle = np.asarray(fit_oracle_z)
    k = feature_state.latent_channels
    if oracle.shape != (len(design), HORIZON, k):
        raise ValueError('N1_ABSOLUTE_ORACLE_Z_SHAPE')
    target = oracle - fit_latent_past[:, -1:]
    mask = complete_prefix_latent_mask(fit_future_observed, k)
    ridge = prepare_masked_ridge(design, target, mask, kind='N1_LATENT', feature_state_sha256=feature_state.sha256)
    return PreparedNative(feature_state, ridge)


def prepare_n2(feature_state, fit_latent_past, fit_target, fit_raw_last, fit_observed):
    design = feature_state.transform(fit_latent_past, require_fit_artifact=True)
    y, last = np.asarray(fit_target), np.asarray(fit_raw_last)
    if y.ndim != 3 or y.shape[:2] != (len(design), HORIZON) or last.shape != (len(design), y.shape[2]) or not np.isfinite(last).all():
        raise ValueError('N2_Y_OR_CURRENT_C_SHAPE')
    ridge = prepare_masked_ridge(design, y - last[:, None], fit_observed,
                                kind='N2_DIRECT_C', feature_state_sha256=feature_state.sha256)
    return PreparedNative(feature_state, ridge)


def gelu_fp32(value):
    """Mathematical exact GELU (erf); old Torch/device parity remains to be checked."""
    x = np.asarray(value, np.float32)
    erf = np.fromiter((math.erf(float(v) / math.sqrt(2.)) for v in x.ravel()), dtype=np.float64, count=x.size).reshape(x.shape)
    return (.5 * x.astype(np.float64) * (1. + erf)).astype(np.float32)


def decoder_features(z_future, z_last, hidden_weight, hidden_bias):
    future, last = np.asarray(z_future, np.float32), np.asarray(z_last, np.float32)
    weight, bias = np.asarray(hidden_weight, np.float32), np.asarray(hidden_bias, np.float32)
    if future.ndim != 3 or last.shape != (len(future), 1, future.shape[2]) or weight.shape != (32, future.shape[2]) or bias.shape != (32,):
        raise ValueError('ABSOLUTE_Z_AND_FROZEN_HIDDEN32_SHAPES_REQUIRED')
    if not all(np.isfinite(value).all() for value in (future, last, weight, bias)):
        raise ValueError('FINITE_DECODER_INPUT_REQUIRED')
    hidden_future = gelu_fp32(future @ weight.T + bias)
    hidden_last = gelu_fp32(last @ weight.T + bias)
    return np.concatenate((future - last, hidden_future - hidden_last), axis=2)


@dataclass(frozen=True)
class DecoderCalibration:
    kind: str
    alpha: float
    weight: np.ndarray
    feature_rms: np.ndarray
    observed_counts: np.ndarray
    scaled_solution: RidgeSolution
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    fixed_n1_model_sha256: str | None

    @property
    def diagnostic_only(self):
        return self.kind == 'O1_ORACLE_DECODER'

    @property
    def sha256(self):
        return _state_sha(self.kind, self.alpha, self.weight, self.feature_rms,
                          self.hidden_weight, self.hidden_bias, self.fixed_n1_model_sha256)

    def forecast(self, z_future, z_last, raw_last, *, fixed_n1_model_sha256=None):
        """Low-level decoder arithmetic/audit; a hash does not prove latent origin.

        N3 deployment must use forecast_from_selected_n1, which has no future
        latent argument and computes the bound model's prediction internally.
        """
        if self.kind == 'N3_NATIVE_DECODER' and fixed_n1_model_sha256 != self.fixed_n1_model_sha256:
            raise ValueError('N3_FIXED_SELECTED_N1_MODEL_REQUIRED')
        features = decoder_features(z_future, z_last, self.hidden_weight, self.hidden_bias)
        last = np.asarray(raw_last, np.float32)
        if last.shape != (len(features), self.weight.shape[1]) or not np.isfinite(last).all():
            raise ValueError('DECODER_CURRENT_C_ANCHOR_REQUIRED')
        return last[:, None] + features @ self.weight

    def forecast_from_selected_n1(self, model, latent_past, raw_last):
        if self.kind != 'N3_NATIVE_DECODER' or not isinstance(model, NativeModel) or model.solution.kind != 'N1_LATENT':
            raise ValueError('N3_SELECTED_N1_NATIVE_MODEL_OBJECT_REQUIRED')
        if model.sha256 != self.fixed_n1_model_sha256:
            raise ValueError('N3_FIXED_SELECTED_N1_MODEL_REQUIRED')
        predicted_future = model.latent_forecast(latent_past)
        return self.forecast(predicted_future, np.asarray(latent_past)[:, -1:], raw_last,
                             fixed_n1_model_sha256=model.sha256)


@dataclass(frozen=True)
class PreparedDecoder:
    kind: str
    ridge: PreparedRidge
    feature_rms: np.ndarray
    original_outgoing: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    fixed_n1_model_sha256: str | None

    def fit(self, alpha):
        solution = self.ridge.fit(alpha)
        effective = (solution.coefficients / self.feature_rms[:, None]).astype(np.float32)
        effective[:, solution.observed_counts == 0] = self.original_outgoing[:, solution.observed_counts == 0]
        if not np.isfinite(effective).all():
            raise ValueError('DECODER_EFFECTIVE_FP32_WEIGHT_NONFINITE')
        return DecoderCalibration(self.kind, float(alpha), _readonly(effective, np.float32), self.feature_rms,
            solution.observed_counts, solution, self.hidden_weight, self.hidden_bias, self.fixed_n1_model_sha256)

    def fit_all(self):
        return tuple(self.fit(alpha) for alpha in ALPHAS)


def _prepare_decoder(kind, z_future, z_last, fit_target, fit_raw_last, observed,
                     hidden_weight, hidden_bias, original_outgoing, fixed_n1_model_sha256=None):
    features = decoder_features(z_future, z_last, hidden_weight, hidden_bias)
    y, last, mask = np.asarray(fit_target, np.float64), np.asarray(fit_raw_last, np.float64), np.asarray(observed)
    original = np.asarray(original_outgoing, np.float32)
    if y.ndim != 3 or y.shape[:2] != features.shape[:2] or mask.shape != y.shape or last.shape != (len(y), y.shape[2]):
        raise ValueError('DECODER_Y_LAST_MASK_SHAPES')
    if original.shape != (features.shape[2], y.shape[2]) or not np.isfinite(original).all() or not np.isfinite(last).all():
        raise ValueError('ORIGINAL_DECODER_OUTGOING_OR_LAST_INVALID')
    flat = features.reshape(-1, features.shape[2]).astype(np.float64)
    rms = np.sqrt(np.mean(flat ** 2, axis=0))
    rms[rms == 0] = 1.
    feature_sha = _state_sha(kind, features, rms, fixed_n1_model_sha256)
    ridge = prepare_masked_ridge(flat / rms, (y - last[:, None]).reshape(-1, y.shape[2]), mask.reshape(-1, y.shape[2]),
        kind=kind, feature_state_sha256=feature_sha, with_intercept=False,
        fallback_coefficients=original.astype(np.float64) * rms[:, None])
    return PreparedDecoder(kind, ridge, _readonly(rms), _readonly(original, np.float32),
                           _readonly(hidden_weight, np.float32), _readonly(hidden_bias, np.float32), fixed_n1_model_sha256)


def prepare_o1(fit_oracle_z, fit_z_last, fit_target, fit_raw_last, fit_observed,
               hidden_weight, hidden_bias, original_outgoing):
    return _prepare_decoder('O1_ORACLE_DECODER', fit_oracle_z, fit_z_last, fit_target, fit_raw_last,
                            fit_observed, hidden_weight, hidden_bias, original_outgoing)


def prepare_n3(fixed_n1_model, fit_latent_past, fit_target, fit_raw_last, fit_observed,
               hidden_weight, hidden_bias, original_outgoing):
    if fixed_n1_model.solution.kind != 'N1_LATENT':
        raise ValueError('N3_REQUIRES_SELECTED_N1_LATENT_MODEL')
    fixed_n1_model.feature_state.transform(fit_latent_past, require_fit_artifact=True)
    future = fixed_n1_model.latent_forecast(fit_latent_past)
    return _prepare_decoder('N3_NATIVE_DECODER', future, fit_latent_past[:, -1:], fit_target, fit_raw_last,
                            fit_observed, hidden_weight, hidden_bias, original_outgoing, fixed_n1_model.sha256)


def persistence_forecast(raw_last, horizon=HORIZON):
    last = np.asarray(raw_last, np.float32)
    if last.ndim != 2 or not np.isfinite(last).all() or horizon != HORIZON:
        raise ValueError('FINITE_CURRENT_C_AND_LOCKED_H48_REQUIRED')
    return np.repeat(last[:, None], horizon, axis=1)


@dataclass(frozen=True)
class InterceptOnlyModel:
    innovation: np.ndarray
    observed_counts: np.ndarray

    @property
    def sha256(self):
        return _state_sha('FIT_INTERCEPT_ONLY', self.innovation, self.observed_counts)

    def forecast(self, raw_last):
        last = np.asarray(raw_last, np.float32)
        if last.ndim != 2 or last.shape[1] != self.innovation.shape[1] or not np.isfinite(last).all():
            raise ValueError('INTERCEPT_CURRENT_C_ANCHOR_REQUIRED')
        return last[:, None] + self.innovation


def fit_intercept_only(fit_target, fit_raw_last, fit_observed):
    y, last, mask = np.asarray(fit_target, np.float64), np.asarray(fit_raw_last, np.float64), np.asarray(fit_observed)
    if y.ndim != 3 or y.shape[1] != HORIZON or last.shape != (len(y), y.shape[2]) or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('INTERCEPT_FIT_SHAPES_OR_BOOL_MASK')
    if not np.isfinite(last).all() or not np.isfinite(y[mask]).all():
        raise ValueError('INTERCEPT_OBSERVED_TARGET_OR_LAST_NONFINITE')
    counts = mask.sum(axis=0)
    sums = np.where(mask, y - last[:, None], 0.).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return InterceptOnlyModel(_readonly(means, np.float32), _readonly(counts, np.int64))


def masked_macro_mse(prediction, target, observed):
    p, y, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    if p.ndim != 3 or p.shape != y.shape or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('SCORE_N_H_C_AND_BOOL_MASK_REQUIRED')
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('SCORE_OBSERVED_NONFINITE')
    counts = mask.sum(axis=(0, 1))
    active = counts > 0
    if not active.any():
        raise ValueError('NO_OBSERVED_SCORE_CHANNEL')
    squared = np.where(mask, p - y, 0.) ** 2
    return float((squared.sum(axis=(0, 1))[active] / counts[active]).mean())


def select_alpha(rows):
    if tuple(row['alpha'] for row in rows) != ALPHAS:
        raise ValueError('ALL_SEVEN_PREDECLARED_ALPHA_ROWS_REQUIRED')
    scores = [row['tune_actual_Y_mse'] for row in rows]
    if any(value is None or not np.isfinite(value) for value in scores):
        raise ValueError('FINITE_ACTUAL_Y_TUNE_SELECTION_REQUIRED')
    return min(range(len(rows)), key=lambda index: scores[index])
