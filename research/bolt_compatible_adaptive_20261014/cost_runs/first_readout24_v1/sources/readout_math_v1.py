"""Past-only self readouts with globally shared alpha selection; array math only."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

OLD = Path(__file__).resolve().parents[1] / 'smallk_failure_followup_20261014'
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
import matched_readout_math_v1 as joint
import stage_separation_math_v1 as stage

ALPHAS = stage.ALPHAS
SELF_ROUTES = ('SELF33', 'SELF512')
ROUTES = ('SELF33', 'JOINT33', 'SELF512')
SOLUTION_KEYS = ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
                 'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')


def self_features(route, past):
    x = np.asarray(past)
    if route not in SELF_ROUTES or x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != 512:
        raise ValueError('SELF_ROUTE_FINITE_FP32_N512C_REQUIRED')
    if not x.shape[0] or not x.shape[2] or not np.isfinite(x).all():
        raise ValueError('SELF_ROUTE_FINITE_NONEMPTY_PAST_REQUIRED')
    if route == 'SELF512':
        return x.copy()
    return np.concatenate((x.reshape(len(x), 32, 16, x.shape[2]).mean(axis=2), x[:, -1:]), axis=1)


@dataclass(frozen=True)
class SelfFeatureState:
    route: str
    center: np.ndarray
    scale: np.ndarray
    fit_rows: int
    fit_raw_features_sha256: str

    @property
    def sha256(self):
        return stage._state_sha('SELF_FEATURE_STATE', self.route, self.center, self.scale,
                                self.fit_rows, self.fit_raw_features_sha256)

    def transform(self, past, *, require_fit_artifact=False):
        raw = self_features(self.route, past)
        if raw.shape[1:] != self.center.shape:
            raise ValueError('SELF_FEATURE_DIMENSION_MISMATCH')
        if require_fit_artifact and (len(raw) != self.fit_rows or stage.array_sha(raw) != self.fit_raw_features_sha256):
            raise ValueError('EXACT_FIT_FEATURE_ARTIFACT_REQUIRED')
        return (raw.astype(np.float64) - self.center) / self.scale

    def metadata(self):
        return {'route': self.route, 'sha256': self.sha256, 'fit_rows': self.fit_rows,
                'fit_raw_features_sha256': self.fit_raw_features_sha256,
                'input_channels': self.center.shape[1], 'feature_count': self.center.shape[0],
                'feature_state_shape': list(self.center.shape), 'FIT_only_statistics': True,
                'raw_feature_dtype': 'float32', 'statistics_dtype': 'float64'}


def fit_self_features(route, fit_past):
    raw = self_features(route, fit_past)
    x = raw.astype(np.float64)
    center = x.mean(axis=0)
    scale = np.sqrt(np.mean((x - center) ** 2, axis=0))
    scale[scale == 0] = 1.
    return SelfFeatureState(route, joint.readonly(center), joint.readonly(scale), len(x), stage.array_sha(raw))


@dataclass(frozen=True)
class SelfModel:
    feature_state: SelfFeatureState
    alpha: float
    arrays: dict

    @property
    def sha256(self):
        return stage._state_sha('SELF_READOUT_MODEL', self.feature_state.sha256, self.alpha,
                                *(self.arrays[k] for k in SOLUTION_KEYS))

    def forecast(self, past, raw_last):
        x = self.feature_state.transform(past)
        last = np.asarray(raw_last)
        if last.dtype != np.float32 or last.shape != (len(x), x.shape[2]) or not np.isfinite(last).all():
            raise ValueError('SELF_FP32_CURRENT_ANCHOR_REQUIRED')
        innovation = np.einsum('npc,phc->nhc', x, self.arrays['coefficients'], optimize=False)
        innovation += self.arrays['intercept']
        result = last[:, None] + innovation.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('SELF_FINITE_FORECAST_REQUIRED')
        return result

    def metadata(self):
        rank = self.arrays['design_ranks']
        counts = self.arrays['observed_counts']
        return {'route': self.feature_state.route, 'alpha': self.alpha, 'model_sha256': self.sha256,
                'feature_state_sha256': self.feature_state.sha256,
                'coefficient_shape': list(self.arrays['coefficients'].shape),
                'registered_coefficient_count': int(self.arrays['coefficients'].size),
                'registered_intercept_count': int(self.arrays['intercept'].size),
                'observed_output_count': int((counts > 0).sum()),
                'design_rank_min': int(rank.min()), 'design_rank_max': int(rank.max()),
                'with_intercept': True, 'selection': 'one global normalized channel-macro TUNE MSE',
                'label_order': 'FP32 Y-last before FP64 solver',
                'prediction_order': 'FP64 own features/coefficients then innovationFP32 + currentC FP32',
                'independent_generalization_proven': False}


@dataclass(frozen=True)
class PreparedSelf:
    feature_state: SelfFeatureState
    channel_ridges: tuple

    def fit(self, alpha):
        solutions = tuple(value.fit(alpha) for value in self.channel_ridges)
        arrays = {}
        for key in SOLUTION_KEYS:
            arrays[key] = joint.readonly(np.concatenate([getattr(s, key) for s in solutions], axis=-1),
                np.int64 if key in ('observed_counts', 'channel_counts', 'design_ranks') else np.float64)
        return SelfModel(self.feature_state, float(alpha), arrays)

    def metadata(self):
        return {'route': self.feature_state.route, 'feature_state': self.feature_state.metadata(),
                'factorization_groups': sum(len(r.groups) for r in self.channel_ridges),
                'channel_factorization_groups': [len(r.groups) for r in self.channel_ridges],
                'channel_submodels_per_alpha': len(self.channel_ridges),
                'alpha_grid': list(ALPHAS), 'P_by_P_Gram_formed': False,
                'normal_equation_penalty': 'alpha*Nc across observed FIT windows and H',
                'selection': 'same global macro TUNE MSE as JOINT33; earliest exact tie'}


def prepare_self(route, fit_past, fit_target, fit_raw_last, fit_observed, *, check=None):
    state = fit_self_features(route, fit_past)
    x = state.transform(fit_past, require_fit_artifact=True)
    y, last, mask = np.asarray(fit_target), np.asarray(fit_raw_last), np.asarray(fit_observed)
    if (y.dtype != np.float32 or last.dtype != np.float32 or y.shape != (len(x), 48, x.shape[2])
            or last.shape != (len(x), x.shape[2]) or mask.dtype != np.bool_ or mask.shape != y.shape
            or not np.isfinite(last).all()):
        raise ValueError('SELF_ORIGINAL_FP32_Y_LAST_BOOL_MASK_REQUIRED')
    target = y - last[:, None]
    ridges = []
    for c in range(x.shape[2]):
        ridges.append(joint.prepare_thin_masked_ridge(x[:, :, c], target[:, :, c:c+1], mask[:, :, c:c+1],
            kind='DIRECT_C_' + route, feature_state_sha256=state.sha256, check=check))
    return PreparedSelf(state, tuple(ridges))


def restore_self_model(feature_metadata, feature_arrays, model_metadata, coefficient_arrays):
    state = SelfFeatureState(feature_metadata['route'], joint.readonly(feature_arrays['center']),
        joint.readonly(feature_arrays['scale']), feature_metadata['fit_rows'], feature_metadata['fit_raw_features_sha256'])
    if state.sha256 != feature_metadata['sha256']:
        raise ValueError('SELF_RESTORED_FEATURE_SHA_MISMATCH')
    model = SelfModel(state, model_metadata['alpha'], {
        k: joint.readonly(coefficient_arrays[k], np.int64 if k in ('observed_counts', 'channel_counts', 'design_ranks') else np.float64)
        for k in SOLUTION_KEYS})
    if model.sha256 != model_metadata['model_sha256']:
        raise ValueError('SELF_RESTORED_MODEL_SHA_MISMATCH')
    return model


def fit_shared_mean(target, raw_last, observed):
    y, last, mask = np.asarray(target), np.asarray(raw_last), np.asarray(observed)
    if y.dtype != np.float32 or last.dtype != np.float32 or mask.dtype != np.bool_ or mask.shape != y.shape:
        raise ValueError('SHARED_MEAN_FP32_Y_LAST_BOOL_REQUIRED')
    innovation = y - last[:, None]
    counts = mask.sum(axis=0)
    sums = np.where(mask, innovation.astype(np.float64), 0.).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    return {'innovation': means, 'observed_counts': counts.astype(np.int64)}


def control_forecast(kind, raw_last, shared_mean):
    last = np.asarray(raw_last)
    if last.dtype != np.float32 or last.ndim != 2:
        raise ValueError('CONTROL_FP32_NC_LAST_REQUIRED')
    if kind == 'PERSIST':
        return np.repeat(last[:, None], 48, axis=1)
    if kind == 'MEAN':
        return last[:, None] + shared_mean['innovation']
    raise ValueError('PREDECLARED_CONTROL_REQUIRED')


def observed_score(prediction, target, observed, std=None):
    p, y, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    if p.shape != y.shape or p.ndim != 3 or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError('SCORE_NHC_BOOL_REQUIRED')
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('OBSERVED_SCORE_NONFINITE')
    error = np.where(mask, p-y, 0.)
    if std is not None:
        error = error * np.asarray(std, np.float64)[None, None]
    counts = mask.sum(axis=(0, 1))
    active = counts > 0
    if not active.any():
        raise ValueError('SCORE_NO_OBSERVED_CHANNELS')
    cmse = np.divide((error**2).sum(axis=(0, 1)), counts, out=np.zeros(len(counts)), where=active)
    cmae = np.divide(np.abs(error).sum(axis=(0, 1)), counts, out=np.zeros(len(counts)), where=active)
    return {'mse': float(cmse[active].mean()), 'mae': float(cmae[active].mean()),
            'per_channel_mse': cmse.tolist(), 'per_channel_mae': cmae.tolist(),
            'observed_counts': counts.tolist(), 'active_channels': int(active.sum())}


def select_pool(rows):
    if len(rows) != 9 or tuple(r['kind'] for r in rows[:2]) != ('PERSIST', 'MEAN'):
        raise ValueError('PERSIST_MEAN_SEVEN_RIDGE_POOL_REQUIRED')
    if tuple(r.get('alpha') for r in rows[2:]) != ALPHAS:
        raise ValueError('SAME_PREDECLARED_SEVEN_ALPHA_REQUIRED')
    scores = [r['tune_scores']['normalized']['mse'] for r in rows]
    if not np.isfinite(scores).all():
        raise ValueError('FINITE_GLOBAL_MACRO_TUNE_SCORES_REQUIRED')
    return min(range(9), key=lambda i: scores[i])


def common_information_gate(self_prediction, joint_prediction, target, observed):
    base = observed_score(self_prediction, target, observed)
    candidate = observed_score(joint_prediction, target, observed)
    blocks = []
    for indexes in np.array_split(np.arange(len(target)), 4):
        try:
            a = observed_score(self_prediction[indexes], target[indexes], observed[indexes])
            b = observed_score(joint_prediction[indexes], target[indexes], observed[indexes])
            blocks.append({'indexes': indexes.tolist(), 'self_mse': a['mse'], 'joint_mse': b['mse'],
                           'joint_improved': b['mse'] < a['mse']})
        except ValueError as error:
            if str(error) != 'SCORE_NO_OBSERVED_CHANNELS':
                raise
            blocks.append({'indexes': indexes.tolist(), 'joint_improved': False, 'reason': str(error)})
    eligible = base['mse'] > 1e-8 and base['mae'] > 1e-8
    gain = 100*(1-candidate['mse']/base['mse']) if eligible else None
    mae_change = 100*(candidate['mae']/base['mae']-1) if eligible else None
    passed = eligible and gain >= 1. and mae_change <= 1. and sum(b['joint_improved'] for b in blocks) >= 3
    return {'self_tune': base, 'joint_tune': candidate, 'mse_gain_pct': gain,
            'mae_change_pct': mae_change, 'four_chronological_blocks': blocks, 'passed': bool(passed),
            'ratio_eligible': eligible, 'exploratory_resource_gate_only': True,
            'independent_generalization_proven': False}
