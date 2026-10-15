"""FIT-past-only raw-channel greedy selection and observed affine D; no IO/GPU."""
from dataclasses import dataclass
import hashlib

import numpy as np


RCOND = 1e-4
GRAM_RELATIVE_CUTOFF = RCOND * RCOND


def array_record(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def validate_past(history, observed):
    x, mask = np.asarray(history), np.asarray(observed)
    if (x.dtype != np.float32 or x.ndim != 3 or not len(x) or x.shape[1] != 512 or not x.shape[2]
            or not np.isfinite(x).all() or mask.dtype != np.bool_ or mask.shape != x.shape):
        raise ValueError("FINITE_FILLED_FP32_N512C_PAST_AND_TRUE_OBSERVED_BOOL_REQUIRED")
    return x, mask


@dataclass(frozen=True)
class ObservedPastStats:
    counts: np.ndarray
    feature_means: np.ndarray
    response_means: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    response_variance: np.ndarray
    metadata: dict


def precompute_observed_stats(history, observed):
    x, mask = validate_past(history, observed)
    features = x.reshape(-1, x.shape[2]).astype(np.float64)
    mask = mask.reshape(features.shape)
    c = features.shape[1]
    counts = mask.sum(axis=0, dtype=np.int64)
    means, response_means, gram = np.zeros((c, c)), np.zeros(c), np.zeros((c, c, c))
    cross, variance = np.zeros((c, c)), np.zeros(c)
    for channel in range(c):
        if counts[channel] == 0: continue
        actual = features[mask[:, channel]]
        means[channel] = actual.mean(axis=0)
        centered = actual-means[channel]
        response_means[channel] = means[channel, channel]
        response = centered[:, channel]
        gram[channel] = centered.T @ centered / counts[channel]
        gram[channel] = (gram[channel]+gram[channel].T)/2
        cross[channel] = centered.T @ response / counts[channel]
        variance[channel] = np.mean(response*response)
    for value in (counts, means, response_means, gram, cross, variance): value.setflags(write=False)
    return ObservedPastStats(counts, means, response_means, gram, cross, variance, {
        "history": array_record(x), "observed": array_record(np.asarray(observed)), "windows": len(x),
        "flattened_past_rows": len(features), "channels": c,
        "response_specific_observed_mask_for_centered_statistics": True,
        "filled_missing_values_are_features_only_not_response_labels": True,
        "duplicated_window_rows_independent_samples_claimed": False,
        "future_TUNE_DEV_or_F_scores_used": False})


def score_subset(stats, indices):
    c = len(stats.counts)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) not in (int, np.int64, np.int32) or not 0 <= int(index) < c for index in indices)):
        raise ValueError("NONEMPTY_DISTINCT_VALID_ORIGINAL_CHANNEL_INDICES_REQUIRED")
    chosen = np.asarray(indices, dtype=np.int64)
    mse, rank, condition = [None]*c, [None]*c, [None]*c
    clipped = 0
    for channel, count in enumerate(stats.counts):
        if count == 0: continue
        gram = stats.gram[channel][np.ix_(chosen, chosen)]
        q = stats.cross[channel, chosen]
        eigenvalues, directions = np.linalg.eigh(gram)
        scale = max(float(eigenvalues[-1]), 0.)
        roundoff = 128*np.finfo(np.float64).eps*max(1., float(np.linalg.norm(gram, 2)))
        if eigenvalues[0] < -roundoff: raise ValueError("CENTERED_GRAM_NOT_NUMERICALLY_PSD")
        retained = eigenvalues > scale*GRAM_RELATIVE_CUTOFF
        if scale == 0: retained[:] = False
        coefficient = np.zeros(len(chosen))
        if retained.any():
            basis = directions[:, retained]
            coefficient = basis @ ((basis.T @ q)/eigenvalues[retained])
            condition[channel] = float(np.sqrt(scale/eigenvalues[retained][0]))
        error = float(stats.response_variance[channel]-2*coefficient@q+coefficient@gram@coefficient)
        tolerance = 512*np.finfo(np.float64).eps*max(1., abs(float(stats.response_variance[channel])),
            abs(float(coefficient@q)), abs(float(coefficient@gram@coefficient)))
        if error < -tolerance: raise ValueError("NEGATIVE_RECONSTRUCTION_MSE_BEYOND_FP64_ROUNDOFF")
        if error < 0: clipped += 1
        mse[channel], rank[channel] = max(0., error), int(retained.sum())
    active = stats.counts > 0
    if not active.any(): raise ValueError("NO_ACTIVE_OBSERVED_RESPONSE_CHANNELS")
    return {"macro_normalized_MSE": float(np.mean([mse[index] for index in np.flatnonzero(active)])),
        "per_channel_MSE": mse, "true_observed_counts": stats.counts.tolist(), "active_channels": int(active.sum()),
        "retained_centered_design_rank": rank, "retained_centered_design_condition": condition,
        "indices": chosen.tolist(), "singular_relative_cutoff": RCOND, "Gram_eigen_relative_cutoff": GRAM_RELATIVE_CUTOFF,
        "negative_roundoff_MSE_values_clipped_to_zero": clipped,
        "response_channel_weighting": "Observed MSE per C then equal mean over C with count>0; count1 contributes zero centered residual"}


def subset_encoder(channels, indices):
    if (type(channels) is not int or channels <= 0 or not indices or len(set(indices)) != len(indices)
            or any(type(index) not in (int, np.int64, np.int32) or not 0 <= int(index) < channels for index in indices)):
        raise ValueError("VALID_NONEMPTY_ORIGINAL_CHANNEL_SUBSET_REQUIRED")
    encoder = np.zeros((channels, len(indices)), np.float32)
    encoder[np.asarray(indices, np.int64), np.arange(len(indices))] = np.float32(1)
    return np.zeros(channels, np.float32), encoder


def encode_subset(history, indices):
    history = np.asarray(history)
    if history.dtype != np.float32 or history.ndim < 2 or not np.isfinite(history).all():
        raise ValueError("FINITE_FP32_RAW_CHANNEL_ARRAY_REQUIRED")
    subset_encoder(history.shape[-1], indices)
    return np.ascontiguousarray(history[..., list(indices)])


def greedy_raw_subset(history, observed, K):
    stats = precompute_observed_stats(history, observed)
    c = len(stats.counts)
    if type(K) is not int or not 1 <= K <= c: raise ValueError("REGISTERED_INTEGER_K_FROM_ONE_TO_C_REQUIRED")
    order, stages = [], []
    for stage in range(K):
        candidates = []
        for index in range(c):
            if index in order: continue
            indices = sorted([*order, index])
            candidates.append({"candidate_original_index": index, **score_subset(stats, indices)})
        winner = min(candidates, key=lambda row: (row["macro_normalized_MSE"], row["candidate_original_index"]))
        order.append(winner["candidate_original_index"])
        stages.append({"stage": stage+1, "selected_original_index": order[-1], "selected_so_far_selection_order": list(order),
            "selected_so_far_original_order": sorted(order), "selected_macro_normalized_MSE": winner["macro_normalized_MSE"],
            "candidates": candidates})
    indices = sorted(order)
    center, encoder = subset_encoder(c, indices)
    return {"selected_indices": indices, "selection_order": order, "center_FP32": center, "encoder_FP32": encoder,
        "metadata": {"rule": "FIT observed-past response-macro centered truncated-LS greedy channel subset",
            "indexing": "Zero-based original raw C indices; final columns in original ascending order",
            "selection_tie": "Exact FP64 scalar equality then lowest original channel index",
            "optimal_subset_guaranteed": False, "forecast_objective_used": False,
            "FIT_past_statistics": stats.metadata, "K": K, "channels": c, "stages": stages,
            "final_reconstruction_scores": score_subset(stats, indices), "whitening_used": False,
            "future_TUNE_DEV_or_F_scores_used": False, "scientific_IO_or_GPU_used": False,
            "encode_numeric_contract": "One-hot E/center0 mathematically; direct FP32 column selection preserves values and signed-zero bytes",
            "direct_slice_dense_onehot_byte_identity_claimed": False}}


def fit_global_observed_decoder(history, observed, latent_history):
    x, mask = validate_past(history, observed)
    z = np.asarray(latent_history)
    if (z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != x.shape[:2]
            or not 1 <= z.shape[2] <= x.shape[2] or not np.isfinite(z).all()):
        raise ValueError("ACTUAL_FINITE_FP32_N512K_PAST_FOR_GLOBAL_D_REQUIRED")
    features, responses, masks = z.reshape(-1, z.shape[2]).astype(np.float64), x.reshape(-1, x.shape[2]).astype(np.float64), mask.reshape(-1, x.shape[2])
    k, c = features.shape[1], responses.shape[1]
    counts = masks.sum(axis=0, dtype=np.int64)
    decoder, bias = np.zeros((k, c), np.float64), np.zeros(c, np.float64)
    ranks, conditions, singular, mse, fallbacks, learned = [], [], [], [], [], []
    predictor_zero, response_zero, rank_deficient, identified, fit_executed = [], [], [], [], []
    for channel, count in enumerate(counts):
        if count == 0:
            ranks.append(0); conditions.append(None); singular.append([]); mse.append(None)
            fallbacks.append("NO_OBSERVED_RESPONSE_ZERO_SLOPE_ZERO_BIAS_UNLEARNED"); learned.append(False)
            predictor_zero.append(None); response_zero.append(None); rank_deficient.append(None)
            identified.append(False); fit_executed.append(False); continue
        features_c, target_c = features[masks[:, channel]], responses[masks[:, channel], channel]
        mean_z, mean_y = features_c.mean(axis=0), float(target_c.mean())
        centered_z, centered_y = features_c-mean_z, target_c-mean_y
        slope, _, rank, values = np.linalg.lstsq(centered_z, centered_y, rcond=RCOND)
        zero_z, zero_y = bool(np.all(centered_z == 0)), bool(np.all(centered_y == 0))
        fallback = None
        if count == 1: fallback = "ONE_OBSERVED_RESPONSE_ZERO_UNIDENTIFIED_SLOPE_MEAN_INTERCEPT"
        elif zero_z: fallback = "ZERO_CENTERED_PREDICTOR_VARIANCE_ZERO_UNIDENTIFIED_SLOPE_MEAN_INTERCEPT"
        if fallback is not None: slope[:] = 0.
        decoder[:, channel] = slope
        bias[channel] = mean_y-mean_z@slope
        error = features_c@slope+bias[channel]-target_c
        ranks.append(int(rank)); singular.append(values.tolist())
        conditions.append(float(values[0]/values[rank-1]) if rank else None)
        mse.append(float(np.mean(error*error))); fallbacks.append(fallback); learned.append(bool(fallback is None and rank > 0))
        predictor_zero.append(zero_z); response_zero.append(zero_y)
        rank_deficient.append(bool(rank < k)); identified.append(bool(rank == k)); fit_executed.append(True)
    with np.errstate(over="ignore", invalid="ignore"):
        decoder32, bias32 = decoder.astype(np.float32), bias.astype(np.float32)
    if not all(np.isfinite(value).all() for value in (decoder, bias, decoder32, bias32)):
        raise ValueError("NONFINITE_GLOBAL_OBSERVED_AFFINE_DECODER")
    return {"decoder_FP64": decoder, "decoder_FP32": decoder32, "bias_FP64": bias, "bias_FP32": bias32,
        "metadata": {"rule": "Global FIT per-response-observed centered lstsq; unpenalized analytic intercept",
            "singular_relative_cutoff": RCOND, "true_observed_counts": counts.tolist(), "retained_centered_design_rank": ranks,
            "retained_centered_design_condition": conditions, "centered_design_singularvalues": singular,
            "observed_past_reconstruction_MSE_FP64": mse, "fallback_reason": fallbacks, "learned_slope": learned,
            "zero_centered_predictor_variance": predictor_zero, "constant_observed_response": response_zero,
            "no_observed_responses": (counts == 0).tolist(), "one_observed_response": (counts == 1).tolist(),
            "rank_deficient_slope": rank_deficient, "slope_identified_in_retained_full_K_space": identified,
            "centered_lstsq_fit_executed": fit_executed,
            "constant_response_is_normal_fitted_zero_slope_solution": True,
            "no_observation_fallback_uses_future_or_current_values": False,
            "fit_input_bindings": {"history": array_record(x), "observed": array_record(mask), "latent_history": array_record(z)},
            "state_array_bindings": {name: array_record(value) for name, value in (("decoder_FP64", decoder), ("decoder_FP32", decoder32), ("bias_FP64", bias), ("bias_FP32", bias32))},
            "future_TUNE_DEV_or_F_scores_used": False, "decoder_fitted_at_each_evaluation_origin": False,
            "innovation_bias_scope": "Bias fitted only for past reconstruction; deployment currentC+(futureZ-lastZ)@D32 cancels bias",
            "scientific_IO_or_GPU_used": False}}
