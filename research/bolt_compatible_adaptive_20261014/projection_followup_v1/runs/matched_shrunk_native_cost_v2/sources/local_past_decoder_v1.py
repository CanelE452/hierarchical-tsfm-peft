"""Origin-specific affine decoder from observed known past; no scientific IO."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


LENGTH, HORIZON = 512, 48
SINGULAR_RELATIVE_CUT = 1e-4


def _binding(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


@dataclass(frozen=True)
class LocalPastDecoder:
    decoder_fit64: np.ndarray
    decoder32: np.ndarray
    bias_fit64: np.ndarray
    bias32: np.ndarray
    diagnostics: dict

    def metadata(self):
        return {**self.diagnostics, "state_arrays": {name: _binding(getattr(self, name))
            for name in ("decoder_fit64", "decoder32", "bias_fit64", "bias32")}}

    def forecast(self, raw_last, latent_future, latent_last):
        raw, future, last = (np.asarray(value) for value in (raw_last, latent_future, latent_last))
        n, k, c = self.decoder32.shape
        if (raw.dtype != np.float32 or raw.shape != (n, c)
                or future.dtype != np.float32 or future.shape != (n, HORIZON, k)
                or last.dtype != np.float32 or last.shape != (n, k)
                or not all(np.isfinite(value).all() for value in (raw, future, last))):
            raise ValueError("FINITE_FP32_CURRENT_C_FUTURE48K_LASTK_MATCHED_LOCAL_D_REQUIRED")
        delta = future - last[:, None, :]
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = raw[:, None, :] + np.einsum("nhk,nkc->nhc", delta, self.decoder32, optimize=False)
        if prediction.dtype != np.float32 or not np.isfinite(prediction).all():
            raise ValueError("NONFINITE_FP32_LOCAL_DECODER_FORECAST")
        return np.ascontiguousarray(prediction)

    def reconstruct(self, latent_history):
        z = np.asarray(latent_history)
        n, k, _ = self.decoder32.shape
        if z.dtype != np.float32 or z.shape != (n, LENGTH, k) or not np.isfinite(z).all():
            raise ValueError("FINITE_FP32_N512K_MATCHED_LOCAL_RECONSTRUCTION_REQUIRED")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.einsum("nlk,nkc->nlc", z, self.decoder32, optimize=False) + self.bias32[:, None]
        if not np.isfinite(prediction).all():
            raise ValueError("NONFINITE_FP32_LOCAL_PAST_RECONSTRUCTION")
        return np.ascontiguousarray(prediction)

    def to_npz(self):
        return {"decoder_fit_FP64": self.decoder_fit64.copy(), "decoder_deployed_FP32": self.decoder32.copy(),
                "decoder_bias_fit_FP64": self.bias_fit64.copy(), "decoder_bias_FP32": self.bias32.copy(),
                "metadata_json": np.array(json.dumps(self.metadata(), sort_keys=True, allow_nan=False))}


def fit_local_past_decoder(history, history_observed, latent_history):
    x, observed, z = (np.ascontiguousarray(value) for value in (history, history_observed, latent_history))
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != LENGTH
            or not len(x) or not x.shape[2] or not np.isfinite(x).all()):
        raise ValueError("FINITE_NONEMPTY_FP32_N512C_KNOWN_FILLED_PAST_REQUIRED")
    if observed.dtype != np.bool_ or observed.shape != x.shape:
        raise ValueError("EXACT_BOOL_TRUE_OBSERVED_RAW_PAST_MASK_REQUIRED")
    if (z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != x.shape[:2]
            or not 1 <= z.shape[2] <= x.shape[2] or not np.isfinite(z).all()):
        raise ValueError("FINITE_FP32_N512K_CACHED_FIXED_E_PAST_REQUIRED")
    n, _, c = x.shape
    k = z.shape[2]
    decoder = np.zeros((n, k, c), dtype=np.float64)
    bias = np.zeros((n, c), dtype=np.float64)
    counts = observed.sum(axis=1, dtype=np.int64)
    ranks = np.zeros((n, c), dtype=np.int64)
    affine_ranks = np.zeros((n, c), dtype=np.int64)
    singulars = np.zeros((n, c, k), dtype=np.float64)
    conditions = np.zeros((n, c), dtype=np.float64)
    mse = np.zeros((n, c), dtype=np.float64)
    zmeans = np.zeros((n, c, k), dtype=np.float64)
    ymeans = np.zeros((n, c), dtype=np.float64)
    for window in range(n):
        for channel in range(c):
            valid = observed[window, :, channel]
            if not counts[window, channel]:
                bias[window, channel] = x[window, -1, channel]
                continue
            features = z[window, valid].astype(np.float64)
            target = x[window, valid, channel].astype(np.float64)
            zmean, ymean = features.mean(axis=0), float(target.mean())
            centered_z, centered_y = features - zmean, target - ymean
            coefficient, _, rank, singular = np.linalg.lstsq(centered_z, centered_y, rcond=SINGULAR_RELATIVE_CUT)
            intercept = ymean - float(zmean @ coefficient)
            decoder[window, :, channel], bias[window, channel] = coefficient, intercept
            zmeans[window, channel], ymeans[window, channel] = zmean, ymean
            ranks[window, channel], affine_ranks[window, channel] = rank, rank + 1
            singulars[window, channel, :len(singular)] = singular
            if rank:
                conditions[window, channel] = singular[0] / singular[rank - 1]
            error = features @ coefficient + intercept - target
            mse[window, channel] = float(np.mean(error * error))
    if not np.isfinite(decoder).all() or not np.isfinite(bias).all():
        raise ValueError("NONFINITE_FP64_LOCAL_PAST_MINIMUM_NORM_SOLUTION")
    with np.errstate(over="ignore", invalid="ignore"):
        decoder32, bias32 = decoder.astype(np.float32), bias.astype(np.float32)
    if not np.isfinite(decoder32).all() or not np.isfinite(bias32).all():
        raise ValueError("NONFINITE_FP32_LOCAL_DECODER_CAST")
    deploy64 = decoder32.astype(np.float64)
    norms = np.linalg.norm(deploy64, axis=(1, 2))
    operator_norms = [float(np.linalg.svd(matrix, compute_uv=False)[0]) for matrix in deploy64]
    valid_condition = ranks > 0
    metadata = {"route": "LOCAL_OBSERVED_PAST_CENTERED_AFFINE_MINIMUM_NORM",
        "windows": n, "channels": c, "K": k, "length": LENGTH, "horizon": HORIZON,
        "singular_relative_cut": SINGULAR_RELATIVE_CUT,
        "equivalent_centered_covariance_eigen_relative_cut": SINGULAR_RELATIVE_CUT ** 2,
        "cutoff_scope": "Observed-subset centered latent slope matrix only; analytic intercept is unpenalized and is not part of the cutoff",
        "intercept_rule": "Per-window/channel observed mean raw C minus observed mean latent K dot slope",
        "minimum_norm_coordinate_scope": "Truncated minimum-norm LS inside retained centered-latent singular directions; not unrestricted full-design LS or minimum norm of uncentered augmented [Z,1] coefficients",
        "normal_equation_certificate_scope": "Residual orthogonal to retained singular directions only; discarded nonzero directions can leave a nonzero full-design residual gradient",
        "regularization_scope": "No ridge penalty or threshold search; fixed hard relative SVD truncation 1e-4",
        "true_observed_counts": counts.tolist(), "centered_slope_ranks": ranks.tolist(),
        "affine_model_ranks_with_analytic_intercept": affine_ranks.tolist(),
        "rank_deficient_slope_channels": (ranks < k).tolist(),
        "observed_count_below_two": (counts < 2).tolist(),
        "no_observation_current_C_fallback": (counts == 0).tolist(),
        "no_observation_fallback_rule": "D column exactly zero; reconstruction bias=current C; forecast=current C; no global fallback or future optimality claim",
        "singularvalues_observed_centered_Z": singulars.tolist(),
        "retained_condition_number": conditions.tolist(),
        "retained_condition_number_valid": valid_condition.tolist(),
        "observed_latent_means_per_C": zmeans.tolist(), "observed_raw_C_means": ymeans.tolist(),
        "observed_reconstruction_MSE_real_fit": mse.tolist(),
        "observed_reconstruction_MSE_valid": (counts > 0).tolist(),
        "max_retained_relative_condition_number": float(conditions[valid_condition].max()) if valid_condition.any() else None,
        "max_abs_decoder_per_window": np.max(np.abs(deploy64), axis=(1, 2)).tolist(),
        "decoder_Frobenius_norm_per_window": norms.tolist(),
        "decoder_operator_norm_per_window": operator_norms,
        "finite_FP64_solution_and_FP32_deployment": True,
        "gain_scope": "Relative condition cap is not an absolute D norm or future-error bound; local singular scale, raw target scale and forecast delta matter",
        "fit_input_bindings": {"history": _binding(x), "history_observed": _binding(observed), "cached_latent_history": _binding(z)},
        "past_target_scope": "Only true observed C past responses; causal-filled inputs remain in the unchanged cached E representation",
        "new_E_F_or_native_head_training": False, "inference_time_decoder_refitting": True,
        "future_labels_used_in_local_decoder_fit": False, "TUNE_or_DEV_threshold_selection": False,
        "decoder_interface": "K future streams + current C + origin-specific KxC coefficients fitted from all-C known past and observation masks",
        "past_C_condition_metadata_added": True, "same_decoder_side_information_as_static": False,
        "strict_K_only_compression": False, "representation_learning_improvement_claimed": False,
        "static_fixed_D_floor_reused_as_dynamic_lower_bound": False,
        "native_head_frozen_intervention_not_reoptimized": True,
        "forecast_arithmetic": "FP32 delta=future-last; einsum('nhk,nkc->nhc',delta,D32,optimize=False); currentC+innovation; bias cancels",
        "zero_delta_anchor_scope": "Exact FP32 numeric value for zero delta; arithmetic addition need not preserve signed-zero bits",
        "independent_holdout": False, "scientific_file_IO_in_module": False,
        "scientific_execution_on_import": False}
    return LocalPastDecoder(decoder, np.ascontiguousarray(decoder32), bias, np.ascontiguousarray(bias32), metadata)
