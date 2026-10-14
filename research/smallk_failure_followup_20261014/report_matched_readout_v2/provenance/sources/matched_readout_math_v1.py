"""Pure array math for matched direct-C readouts; no loaders/runtime/execution.

SOURCE prepared only. RAW33C and E512K use the old N2 loss, FP32 label
subtraction and FP32 innovation/anchor inference order. Factorizations use one
thin SVD per observed-mask group, never a feature_count squared Gram matrix.
"""
from dataclasses import dataclass

import numpy as np
import stage_separation_math_v1 as stage

ALPHAS = stage.ALPHAS
ROUTES = ('RAW33C', 'E512K')
LENGTH, HORIZON = 512, 48


def readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def past_features(route, past):
    value = np.asarray(past)
    if route not in ROUTES:
        raise ValueError('PREDECLARED_RAW33C_OR_E512K_ROUTE_REQUIRED')
    if value.ndim != 3 or value.shape[1] != LENGTH or not value.shape[0] or not value.shape[2]:
        raise ValueError('NONEMPTY_PAST_N512Q_REQUIRED')
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('ORIGINAL_NORMALIZED_PAST_FP32_FINITE_REQUIRED')
    if route == 'RAW33C':
        # numpy FP32 means precede FIT FP64 statistics, as in the old E33K route.
        blocks = value.reshape(len(value), 32, 16, value.shape[2]).mean(axis=2)
        return np.concatenate((blocks, value[:, -1:]), axis=1).reshape(len(value), -1)
    return value.reshape(len(value), -1)


@dataclass(frozen=True)
class FeatureState:
    route: str
    center: np.ndarray
    scale: np.ndarray
    input_channels: int
    fit_rows: int
    fit_raw_features_sha256: str

    @property
    def sha256(self):
        return stage._state_sha('MATCHED_DIRECT_C_FEATURE_STATE', self.route, self.center,
                                self.scale, self.input_channels, self.fit_rows, self.fit_raw_features_sha256)

    def transform(self, past, *, require_fit_artifact=False):
        raw = past_features(self.route, past)
        if past.shape[2] != self.input_channels or raw.shape[1] != len(self.center):
            raise ValueError('MATCHED_ROUTE_FEATURE_DIMENSION_MISMATCH')
        if require_fit_artifact and (len(raw) != self.fit_rows or stage.array_sha(raw) != self.fit_raw_features_sha256):
            raise ValueError('SAME_FIT_PAST_FEATURE_ARTIFACT_REQUIRED')
        return (raw.astype(np.float64) - self.center) / self.scale

    def metadata(self):
        return {'route': self.route, 'sha256': self.sha256, 'input_channels': self.input_channels,
                'feature_count': len(self.center), 'fit_rows': self.fit_rows,
                'fit_raw_features_sha256': self.fit_raw_features_sha256,
                'feature_raw_dtype': 'float32', 'FIT_statistics_dtype': 'float64',
                'FIT_only_statistics': True, 'zero_scale_replaced_by': 1.,
                'inference_requires_future_or_target': False}


def fit_feature_state(route, fit_past):
    raw = past_features(route, fit_past)
    values = raw.astype(np.float64)
    center = values.mean(axis=0)
    scale = np.sqrt(np.mean((values - center) ** 2, axis=0))
    scale[scale == 0] = 1.
    return FeatureState(route, readonly(center), readonly(scale), fit_past.shape[2], len(raw), stage.array_sha(raw))


def prepare_thin_masked_ridge(design, target, observed, *, kind, feature_state_sha256, check=None):
    """Same stage.prepare_masked_ridge objective, solution and seven-alpha apply.

    This construction substitutes one thin SVD for its separate lstsq+SVD.
    Alpha0 uses the centered minimum-norm solution with NumPy rcond=None's
    float64 eps*max(observed_rows,feature_count)*smax cutoff. Positive-alpha
    solutions keep every singular value, as in the frozen original routine.
    No P-by-P Gram or inverse is formed, including when P > observed_rows.
    Optional checks run before/after every mask-group factorization; a solver
    call itself is cooperatively uninterruptible and must have an outer guard.
    """
    guard = check if check is not None else lambda: None
    guard()
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
    masks = {}
    for col in np.flatnonzero(counts):
        masks.setdefault(flat_mask[:, col].tobytes(), []).append(int(col))
    groups = []
    for columns in masks.values():
        guard()
        rows = flat_mask[:, columns[0]]
        gx, gy = x[rows], flat_y[np.ix_(rows, columns)]
        x_mean, y_mean = gx.mean(axis=0), gy.mean(axis=0)
        centered_x, centered_y = gx-x_mean, gy-y_mean
        left, singular, right = np.linalg.svd(centered_x, full_matrices=False)
        threshold = np.finfo(np.float64).eps * max(centered_x.shape) * (singular[0] if len(singular) else 0.)
        active = singular > threshold
        rank = int(active.sum())
        projected = left.T @ centered_y
        inverse = np.divide(1., singular, out=np.zeros_like(singular), where=active)
        minimum_norm = right.T @ (inverse[:, None] * projected)
        groups.append(stage._MaskGroup(readonly(columns, np.int64), int(rows.sum()), readonly(x_mean),
            readonly(y_mean), readonly(singular), readonly(right), readonly(projected), readonly(minimum_norm), rank))
        guard()
    result = stage.PreparedRidge(kind, x.shape[1], shape, readonly(counts, np.int64),
        readonly(channel_counts, np.int64), readonly(denominators), tuple(groups),
        readonly(np.zeros((x.shape[1], flat_y.shape[1]))), feature_state_sha256, True)
    guard()
    return result


@dataclass(frozen=True)
class ReadoutModel:
    feature_state: FeatureState
    solution: stage.RidgeSolution

    @property
    def sha256(self):
        return stage._state_sha('MATCHED_DIRECT_C_MODEL', self.feature_state.sha256, self.solution.sha256)

    def forecast(self, past, raw_last):
        last = np.asarray(raw_last)
        if (last.dtype != np.float32 or last.shape != (len(past), self.solution.intercept.shape[-1])
                or not np.isfinite(last).all()):
            raise ValueError('SAME_FINITE_NORMALIZED_CURRENT_C_FP32_ANCHOR_REQUIRED')
        if self.solution.kind != 'DIRECT_C_' + self.feature_state.route:
            raise ValueError('SAME_DIRECT_C_ROUTE_MODEL_REQUIRED')
        innovation = self.solution.predict(self.feature_state.transform(past)).astype(np.float32)
        prediction = last[:, None] + innovation
        if not np.isfinite(prediction).all():
            raise ValueError('MATCHED_DIRECT_C_FP32_FORECAST_NONFINITE')
        return prediction

    def metadata(self):
        ranks = self.solution.design_ranks
        conditions = self.solution.regularized_gram_conditions
        finite = conditions[np.isfinite(conditions)]
        counts = self.solution.observed_counts
        return {'route': self.feature_state.route, 'model_sha256': self.sha256,
            'solution_sha256': self.solution.sha256, 'kind': self.solution.kind, 'alpha': self.solution.alpha,
            'feature_state_sha256': self.feature_state.sha256, 'with_intercept': True,
            'coefficient_shape': list(self.solution.coefficients.shape),
            'registered_coefficient_count': int(self.solution.coefficients.size),
            'registered_intercept_count': int(self.solution.intercept.size),
            'observed_output_count': int((counts > 0).sum()), 'zero_observed_output_count': int((counts == 0).sum()),
            'trained_coefficient_count': int(len(self.feature_state.center) * (counts > 0).sum()),
            'trained_intercept_count': int((counts > 0).sum()),
            'design_rank_min': int(ranks.min()), 'design_rank_max': int(ranks.max()),
            'gram_condition_finite_min': float(finite.min()) if len(finite) else None,
            'gram_condition_finite_max': float(finite.max()) if len(finite) else None,
            'gram_condition_inf_count': int(np.isinf(conditions).sum()),
            'gram_condition_nan_count': int(np.isnan(conditions).sum()),
            'condition_arrays_with_NaN_or_Inf_require_NPZ': True,
            'label_innovation_order': 'FP32 Y-last first, then solver FP64',
            'prediction_order': 'FP64 transformed_features@coeff+b then innovation FP32 then current-C FP32 anchor',
            'input_dimensions_and_readout_capacity_differ_between_routes': True,
            'independent_generalization_proven': False}


@dataclass(frozen=True)
class PreparedReadout:
    feature_state: FeatureState
    ridge: stage.PreparedRidge

    def fit(self, alpha):
        return ReadoutModel(self.feature_state, self.ridge.fit(alpha))

    def fit_all(self):
        return tuple(self.fit(alpha) for alpha in ALPHAS)

    def metadata(self):
        return {'route': self.feature_state.route, 'feature_state': self.feature_state.metadata(),
            'alpha_grid': list(ALPHAS), 'factorization_groups': len(self.ridge.groups),
            'factorization': 'one centered thin SVD per identical observed-mask group; all alpha share it',
            'P_by_P_Gram_formed': False, 'alpha0': 'centered minimum-norm; eps*max(n,p)*smax cutoff',
            'unpenalized_intercept': True, 'target': 'normalized observed actual-Y innovation',
            'loss': 'mean_active_c sum_NH mask*error^2/Nc + alpha*mean_active_c sum_h||W_scaled_hc||^2',
            'normal_equation_penalty': 'alpha*Nc, not alpha*n_hc',
            'zero_observed_output_fallback': 'innovation0 and bias0; explicitly unlearned',
            'selection': 'actual-Y TUNE normalized observed-channel macro MSE; earliest exact tie'}


def prepare_route(route, fit_past, fit_target, fit_raw_last, fit_observed, *, check=None):
    guard = check if check is not None else lambda: None
    guard()
    state = fit_feature_state(route, fit_past)
    design = state.transform(fit_past, require_fit_artifact=True)
    y, last, mask = np.asarray(fit_target), np.asarray(fit_raw_last), np.asarray(fit_observed)
    if (y.dtype != np.float32 or last.dtype != np.float32 or y.ndim != 3
            or y.shape[:2] != (len(design), HORIZON) or last.shape != (len(design), y.shape[2])
            or mask.dtype != np.bool_ or mask.shape != y.shape or not np.isfinite(last).all()):
        raise ValueError('ORIGINAL_FP32_Y_LAST_BOOL_MASK_N48C_REQUIRED')
    # Match old N2 exactly: subtraction occurs before FP64 solver conversion.
    target_innovation = y - last[:, None]
    ridge = prepare_thin_masked_ridge(design, target_innovation, mask, kind='DIRECT_C_' + route,
                                    feature_state_sha256=state.sha256, check=check)
    guard()
    return PreparedReadout(state, ridge)


def select_alpha(rows):
    return stage.select_alpha(rows)


def restore_model(feature_metadata, feature_arrays, model_metadata, solution_arrays):
    state = FeatureState(feature_metadata['route'], readonly(feature_arrays['center']), readonly(feature_arrays['scale']),
        feature_metadata['input_channels'], feature_metadata['fit_rows'], feature_metadata['fit_raw_features_sha256'])
    if state.sha256 != feature_metadata['sha256']:
        raise ValueError('RESTORED_MATCHED_FEATURE_STATE_SHA_MISMATCH')
    solution = stage.RidgeSolution(model_metadata['kind'], model_metadata['alpha'],
        *(readonly(solution_arrays[name], np.int64 if name in ('observed_counts', 'channel_counts', 'design_ranks') else np.float64)
          for name in ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
                       'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')),
        model_metadata['feature_state_sha256'], model_metadata['with_intercept'])
    model = ReadoutModel(state, solution)
    if solution.sha256 != model_metadata['solution_sha256'] or model.sha256 != model_metadata['model_sha256']:
        raise ValueError('RESTORED_MATCHED_MODEL_SHA_MISMATCH')
    return model
