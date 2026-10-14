"""SOURCE-only pure math: correct a frozen, past-generated MAIN forecast.

No loaders, runtime, model/CP/data reads, E/F/D calls or execution entry point.
Future-shaped MAIN forecasts are past-generated predictions, not future truth.
Array shape cannot establish that provenance: the future gated runner must bind
original CP/forecast/statistic/origin/mask artifacts. Nothing here trains E/F/D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matched_readout_math_v1 as matched
import stage_separation_math_v1 as stage

ROUTES = ('RAW33C', 'E33K', 'E512K')
ALPHAS = matched.ALPHAS
ZERO = 'ZERO_CORRECTION'
MEAN = 'MEAN_CORRECTION'
CANDIDATE_IDS = (ZERO, MEAN) + tuple('RIDGE_ALPHA_' + str(i) for i in range(len(ALPHAS)))
HORIZON = 48


def past_features(route, past):
    if route not in ROUTES:
        raise ValueError('PREDECLARED_PAST_RAW33C_E33K_E512K_REQUIRED')
    return matched.past_features('E512K' if route == 'E512K' else 'RAW33C', past)


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
        return stage._state_sha('MAIN_RESIDUAL_FIT_FEATURE_STATE', self.route, self.center,
            self.scale, self.input_channels, self.fit_rows, self.fit_raw_features_sha256)

    def transform(self, past, *, require_fit_artifact=False):
        raw = past_features(self.route, past)
        if np.asarray(past).shape[2] != self.input_channels or raw.shape[1] != len(self.center):
            raise ValueError('SAME_PAST_E_FEATURE_DIMENSION_REQUIRED')
        if require_fit_artifact and (len(raw) != self.fit_rows or stage.array_sha(raw) != self.fit_raw_features_sha256):
            raise ValueError('SAME_FIT_PAST_E_ARTIFACT_REQUIRED')
        return (raw.astype(np.float64) - self.center) / self.scale

    def metadata(self):
        return {'route': self.route, 'sha256': self.sha256, 'input_channels': self.input_channels,
            'input_semantics': 'original C past' if self.route == 'RAW33C' else 'frozen E past',
            'feature_count': len(self.center), 'fit_rows': self.fit_rows,
            'fit_raw_features_sha256': self.fit_raw_features_sha256,
            'statistics_FIT_only': True, 'raw_dtype': 'float32', 'statistics_dtype': 'float64'}


def fit_feature_state(route, fit_past_E):
    raw = past_features(route, fit_past_E)
    values = raw.astype(np.float64)
    center = values.mean(axis=0)
    scale = np.sqrt(((values-center) ** 2).mean(axis=0)); scale[scale == 0] = 1.
    return FeatureState(route, matched.readonly(center), matched.readonly(scale), np.asarray(fit_past_E).shape[2],
                         len(raw), stage.array_sha(raw))


def _main(main_forecast, rows, channels=None):
    value = np.asarray(main_forecast)
    if (value.dtype != np.float32 or value.ndim != 3 or value.shape[:2] != (rows, HORIZON)
            or not value.shape[2] or (channels is not None and value.shape[2] != channels)
            or not np.isfinite(value).all()):
        raise ValueError('FINITE_PAST_GENERATED_MAIN_FP32_N48C_REQUIRED')
    return value


def fit_residual_target(fit_main_forecast, fit_target, fit_observed):
    y, mask = np.asarray(fit_target), np.asarray(fit_observed)
    main = _main(fit_main_forecast, len(y))
    if y.dtype != np.float32 or y.shape != main.shape or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('SAME_ORIGINAL_FP32_Y_MAIN_BOOL_MASK_REQUIRED')
    if not np.isfinite(y[mask]).all():
        raise ValueError('FINITE_OBSERVED_FIT_Y_REQUIRED')
    # Do not move this subtraction into the FP64 solver: it is the fixed recipe.
    residual = y - main
    if not np.isfinite(residual[mask]).all():
        raise ValueError('OBSERVED_FP32_Y_MINUS_MAIN_OVERFLOW')
    return residual


@dataclass(frozen=True)
class CorrectionModel:
    feature_state: FeatureState
    channels: int
    solution: stage.RidgeSolution | None
    mean_control: InterceptOnlyControl | None = None

    @property
    def candidate_id(self):
        if self.mean_control is not None:
            return MEAN
        return ZERO if self.solution is None else CANDIDATE_IDS[2+ALPHAS.index(self.solution.alpha)]

    @property
    def alpha(self):
        return None if self.solution is None else self.solution.alpha

    @property
    def sha256(self):
        return stage._state_sha('FROZEN_MAIN_PLUS_PAST_E_CORRECTION', self.feature_state.sha256,
            self.channels, self.candidate_id, self.solution.sha256 if self.solution is not None else None,
            self.mean_control.sha256 if self.mean_control is not None else None)

    def forecast(self, past, main_forecast):
        main = _main(main_forecast, len(past), self.channels)
        value = np.asarray(past)
        if (value.dtype != np.float32 or value.ndim != 3 or value.shape != (len(main),512,self.feature_state.input_channels)
                or not np.isfinite(value).all()):
            raise ValueError('SAME_PAST_INPUT_N512Q_FP32_REQUIRED')
        if self.mean_control is not None:
            if self.solution is not None:
                raise ValueError('MEAN_AND_RIDGE_ARE_SEPARATE_CANDIDATES')
            return self.mean_control.forecast(main)
        if self.solution is None:
            # Preserve original MAIN bits, including signed zero; no forced refit.
            return main.copy()
        design = self.feature_state.transform(past)
        if self.solution.kind != 'MAIN_RESIDUAL_' + self.feature_state.route:
            raise ValueError('SAME_MAIN_RESIDUAL_ROUTE_SOLUTION_REQUIRED')
        correction = self.solution.predict(design).astype(np.float32)
        prediction = main + correction
        if not np.isfinite(prediction).all():
            raise ValueError('MAIN_PLUS_CORRECTION_FP32_FORECAST_NONFINITE')
        return prediction

    def metadata(self):
        return {'route': self.feature_state.route, 'candidate_id': self.candidate_id, 'alpha': self.alpha,
            'model_sha256': self.sha256, 'feature_state_sha256': self.feature_state.sha256,
            'kind': self.solution.kind if self.solution is not None else self.candidate_id,
            'solution_sha256': self.solution.sha256 if self.solution is not None else None,
            'mean_control_sha256': self.mean_control.sha256 if self.mean_control is not None else None,
            'mean_FIT_residual_sha256': self.mean_control.fit_residual_sha256 if self.mean_control is not None else None,
            'mean_FIT_observed_sha256': self.mean_control.fit_observed_sha256 if self.mean_control is not None else None,
            'with_intercept': self.solution is not None or self.mean_control is not None, 'channels': self.channels,
            'feature_count': len(self.feature_state.center),
            'registered_coefficients': int(self.solution.coefficients.size) if self.solution is not None else 0,
            'registered_intercepts': int(self.solution.intercept.size) if self.solution is not None else (48*self.channels if self.mean_control is not None else 0),
            'rank_min': int(self.solution.design_ranks.min()) if self.solution is not None else None,
            'rank_max': int(self.solution.design_ranks.max()) if self.solution is not None else None,
            'zero_observed_outputs': int((self.solution.observed_counts == 0).sum()) if self.solution is not None else None,
            'baseline_original_MAIN_always_selectable': True, 'E_F_D_updated': False,
            'strict_K_only_inference_claim': False,
            'correction_inputs': 'original C past' if self.feature_state.route == 'RAW33C' else 'past E only',
            'raw_route_is_information_control': self.feature_state.route == 'RAW33C',
            'base_input': 'past-generated original MAIN future-shaped prediction',
            'actual_future_or_mask_required_at_inference': False,
            'external_MAIN_provenance_required': True,
            'prediction_order': 'FP64 correction dot+b -> correction FP32 -> add original MAIN FP32',
            'nonfinite_condition_arrays_belong_in_NPZ_only': True,
            'independent_generalization_proven': False}


@dataclass(frozen=True)
class PreparedCorrections:
    feature_state: FeatureState
    channels: int
    ridge: stage.PreparedRidge
    mean_control: InterceptOnlyControl

    def zero_model(self):
        return zero_correction_model(self.feature_state, self.channels)

    def fit(self, alpha):
        return CorrectionModel(self.feature_state, self.channels, self.ridge.fit(alpha))

    def mean_model(self):
        return mean_correction_model(self.feature_state, self.channels, self.mean_control)

    def candidates(self):
        return (self.zero_model(), self.mean_model()) + tuple(self.fit(alpha) for alpha in ALPHAS)

    def metadata(self):
        return {'route': self.feature_state.route, 'feature_state': self.feature_state.metadata(),
            'channels': self.channels, 'candidate_ids': list(CANDIDATE_IDS), 'alpha_grid': list(ALPHAS),
            'baseline': 'original MAIN with exact zero correction', 'baseline_tie_priority': 'first',
            'mean_control_second_candidate_and_shared_per_condition': True,
            'mean_control_sha256': self.mean_control.sha256, 'factorization_groups': len(self.ridge.groups),
            'objective': 'mean_active_c sum_NH observed*residual_error^2/Nc + alpha*mean_active_c sum_h||W_scaled_hc||^2',
            'penalty': 'alpha*Nc, not alpha*n_hc', 'intercept_penalized': False,
            'target_operation': 'FP32 actual-Y minus frozen MAIN, then FP64 solver',
            'zero_observed_output': 'correction0 and bias0; original MAIN preserved',
            'alpha0': 'centered thin-SVD minimum norm with eps*max(n,p)*smax cutoff',
            'original_MAIN_not_retrained': True, 'E_F_D_not_loaded_or_updated': True}


def zero_correction_model(feature_state, channels):
    """Construct before any ridge factorization so failed fits retain MAIN."""
    return CorrectionModel(feature_state, channels, None)


def mean_correction_model(feature_state, channels, mean_control):
    if mean_control.correction.shape != (48, channels):
        raise ValueError('SAME_SHARED_MEAN_C_CHANNELS_REQUIRED')
    return CorrectionModel(feature_state, channels, None, mean_control)


def prepare_with_state(state, fit_past, fit_main_forecast, fit_target, fit_observed, *, mean_control=None, check=None):
    guard = check if check is not None else lambda: None
    guard()
    design = state.transform(fit_past, require_fit_artifact=True)
    main = _main(fit_main_forecast, len(design))
    residual = fit_residual_target(main, fit_target, fit_observed)
    if mean_control is None:
        mean_control = _fit_intercept_from_residual(residual, fit_observed)
    if (mean_control.fit_residual_sha256 != stage.array_sha(residual)
            or mean_control.fit_observed_sha256 != stage.array_sha(np.asarray(fit_observed))
            or mean_control.correction.shape != (48, main.shape[2])):
        raise ValueError('SHARED_MEAN_SAME_FIT_RESIDUAL_MASK_REQUIRED')
    ridge = matched.prepare_thin_masked_ridge(design, residual, fit_observed,
        kind='MAIN_RESIDUAL_' + state.route, feature_state_sha256=state.sha256, check=check)
    guard()
    return PreparedCorrections(state, main.shape[2], ridge, mean_control)


def prepare_route(route, fit_past, fit_main_forecast, fit_target, fit_observed, *, mean_control=None, check=None):
    guard = check if check is not None else lambda: None
    guard()
    state = fit_feature_state(route, fit_past)
    return prepare_with_state(state, fit_past, fit_main_forecast, fit_target, fit_observed, mean_control=mean_control, check=check)


@dataclass(frozen=True)
class InterceptOnlyControl:
    correction: np.ndarray
    observed_counts: np.ndarray
    fit_residual_sha256: str
    fit_observed_sha256: str

    @property
    def sha256(self):
        return stage._state_sha('MAIN_RESIDUAL_FIT_INTERCEPT_ONLY', self.correction, self.observed_counts,
                                self.fit_residual_sha256, self.fit_observed_sha256)

    def forecast(self, main_forecast):
        main = _main(main_forecast, len(main_forecast), self.correction.shape[1])
        prediction = main + self.correction
        if not np.isfinite(prediction).all():
            raise ValueError('INTERCEPT_MAIN_PLUS_CORRECTION_FP32_NONFINITE')
        return prediction


def fit_intercept_only(fit_main_forecast, fit_target, fit_observed):
    residual = fit_residual_target(fit_main_forecast, fit_target, fit_observed)
    return _fit_intercept_from_residual(residual, fit_observed)


def _fit_intercept_from_residual(residual, fit_observed):
    mask = np.asarray(fit_observed)
    counts = mask.sum(axis=0)
    total = np.where(mask, residual.astype(np.float64), 0.).sum(axis=0)
    mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0).astype(np.float32)
    return InterceptOnlyControl(matched.readonly(mean, np.float32), matched.readonly(counts, np.int64),
                                 stage.array_sha(residual), stage.array_sha(mask))


def select_candidate(rows):
    """Zero then mean then ridge; preserve failed slots, never select their scores.

    Caller records every fixed candidate. An explicitly FAILED_PRESERVED slot
    can have no score; a finite COMPLETE zero baseline remains selectable. A
    missing/nonfinite score labelled COMPLETE is a contract error, not a tie.
    """
    if tuple(row.get('candidate_id') for row in rows) != CANDIDATE_IDS:
        raise ValueError('EXACT_ZERO_MEAN_THEN_SEVEN_RIDGE_ROWS_REQUIRED')
    if tuple(row.get('alpha') for row in rows) != (None, None) + ALPHAS:
        raise ValueError('ZERO_BASELINE_IS_DISTINCT_FROM_ALPHA0_REQUIRED')
    eligible = []
    for index, row in enumerate(rows):
        status = row.get('status', 'COMPLETE')
        value = row.get('tune_actual_Y_mse')
        if status == 'FAILED_PRESERVED' and index != 0:
            continue
        if status != 'COMPLETE' or value is None or not np.isfinite(value):
            raise ValueError('FINITE_ACTUAL_Y_TUNE_AND_ZERO_BASELINE_REQUIRED')
        eligible.append(index)
    return min(eligible, key=lambda index: rows[index]['tune_actual_Y_mse'])


def restore_model(feature_metadata, feature_arrays, model_metadata, solution_arrays=None):
    state = FeatureState(feature_metadata['route'], matched.readonly(feature_arrays['center']),
        matched.readonly(feature_arrays['scale']), feature_metadata['input_channels'], feature_metadata['fit_rows'],
        feature_metadata['fit_raw_features_sha256'])
    if state.sha256 != feature_metadata['sha256']:
        raise ValueError('RESTORED_MAIN_RESIDUAL_FEATURE_STATE_SHA')
    solution = None
    mean_control = None
    if model_metadata['candidate_id'] == MEAN:
        mean_control = InterceptOnlyControl(matched.readonly(solution_arrays['correction'],np.float32),
            matched.readonly(solution_arrays['observed_counts'],np.int64), model_metadata['mean_FIT_residual_sha256'],
            model_metadata['mean_FIT_observed_sha256'])
        if mean_control.sha256 != model_metadata['mean_control_sha256']:
            raise ValueError('RESTORED_SHARED_MEAN_CONTROL_SHA')
    elif model_metadata['candidate_id'] != ZERO:
        solution = stage.RidgeSolution(model_metadata['kind'], model_metadata['alpha'],
            *(matched.readonly(solution_arrays[name], np.int64 if name in ('observed_counts', 'channel_counts', 'design_ranks') else np.float64)
              for name in ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
                           'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')),
            model_metadata['feature_state_sha256'], model_metadata['with_intercept'])
        if solution.sha256 != model_metadata['solution_sha256']:
            raise ValueError('RESTORED_MAIN_RESIDUAL_SOLUTION_SHA')
    model = CorrectionModel(state, model_metadata['channels'], solution, mean_control)
    if model.sha256 != model_metadata['model_sha256'] or model.candidate_id != model_metadata['candidate_id']:
        raise ValueError('RESTORED_MAIN_RESIDUAL_MODEL_SHA')
    return model
