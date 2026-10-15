"""FIT-past subset scoring by direct observed centered LS residuals; no IO/GPU."""
from dataclasses import dataclass

import numpy as np

from raw_subset_projection_v1 import (
    RCOND, array_record, validate_past, subset_encoder, encode_subset,
    fit_global_observed_decoder,
)


@dataclass(frozen=True)
class ObservedPastStats:
    counts: np.ndarray
    features: np.ndarray
    observed: np.ndarray
    metadata: dict


def precompute_observed_stats(history, observed):
    x, mask = validate_past(history, observed)
    features = x.reshape(-1, x.shape[2]).astype(np.float64)
    mask = mask.reshape(features.shape).copy()
    counts = mask.sum(axis=0, dtype=np.int64)
    for value in (counts, features, mask): value.setflags(write=False)
    return ObservedPastStats(counts, features, mask, {
        "history": array_record(x), "observed": array_record(np.asarray(observed)),
        "windows": len(x), "flattened_past_rows": len(features), "channels": features.shape[1],
        "response_specific_observed_mask_for_centered_statistics": True,
        "filled_missing_values_are_features_only_not_response_labels": True,
        "duplicated_window_rows_independent_samples_claimed": False,
        "future_TUNE_DEV_or_F_scores_used": False,
        "Gram_cross_variance_expansion_used_for_scores": False,
        "score_numeric_contract": "Direct per-response observed centered np.linalg.lstsq(rcond=1e-4), then mean squared explicit residual"})


def score_subset(stats, indices):
    c = len(stats.counts)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) not in (int, np.int64, np.int32) or not 0 <= int(index) < c for index in indices)):
        raise ValueError("NONEMPTY_DISTINCT_VALID_ORIGINAL_CHANNEL_INDICES_REQUIRED")
    chosen = np.asarray(indices, dtype=np.int64)
    mse, ranks, conditions, singular = [None]*c, [None]*c, [None]*c, [None]*c
    for channel, count in enumerate(stats.counts):
        if count == 0: continue
        rows = stats.features[stats.observed[:, channel]]
        predictors, response = rows[:, chosen], rows[:, channel]
        centered, target = predictors-predictors.mean(axis=0), response-response.mean()
        slope, _, rank, values = np.linalg.lstsq(centered, target, rcond=RCOND)
        residual = centered@slope-target
        error = float(np.mean(residual*residual))
        if not np.isfinite(error) or not np.isfinite(slope).all():
            raise ValueError("NONFINITE_DIRECT_CENTERED_LS_RECONSTRUCTION_SCORE")
        mse[channel], ranks[channel], singular[channel] = error, int(rank), values.tolist()
        if rank: conditions[channel] = float(values[0]/values[rank-1])
    active = stats.counts > 0
    if not active.any(): raise ValueError("NO_ACTIVE_OBSERVED_RESPONSE_CHANNELS")
    return {"macro_normalized_MSE": float(np.mean([mse[index] for index in np.flatnonzero(active)])),
        "per_channel_MSE": mse, "true_observed_counts": stats.counts.tolist(), "active_channels": int(active.sum()),
        "retained_centered_design_rank": ranks, "retained_centered_design_condition": conditions,
        "centered_design_singularvalues": singular, "indices": chosen.tolist(),
        "singular_relative_cutoff": RCOND,
        "Gram_eigen_relative_cutoff_equivalent_in_real_arithmetic_only": RCOND*RCOND,
        "Gram_cross_variance_expansion_used_for_scores": False,
        "negative_roundoff_MSE_values_clipped_to_zero": 0,
        "score_numeric_contract": "Direct per-response observed centered lstsq coefficient and explicit residual squared mean; no negative clipping",
        "response_channel_weighting": "Observed MSE per C then equal mean over C with count>0; count1 contributes zero centered residual"}


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
            "direct_slice_dense_onehot_byte_identity_claimed": False,
            "numeric_repair_scope": "Same FIT-past masked LS objective, rcond and exact tie rule; explicit direct residuals replace cancellation-prone Gram expansion, so finite-computed candidate ordering can differ",
            "global_decoder_fitter_unchanged_v1_function": True}}
