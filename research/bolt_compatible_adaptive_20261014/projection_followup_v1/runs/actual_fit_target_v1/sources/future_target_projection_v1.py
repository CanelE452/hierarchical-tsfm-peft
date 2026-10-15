"""Supervised FIT-target output space with past-only pointwise deployment.

Pure NumPy arithmetic: this module never reads scientific files or launches fits
on import. The caller supplies only FIT arrays and controls their audited lineage.
"""
import hashlib

import numpy as np

from projection_math_v1 import ProjectionState


LENGTH = 512
HORIZON = 48
EIGENVALUE_RELATIVE_CUT = 1e-8
PINV_RCOND = 1e-15


def _binding(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _canonical_columns(matrix):
    value = np.array(matrix, dtype=np.float64, order="C", copy=True)
    for column in range(value.shape[1]):
        if value[np.argmax(np.abs(value[:, column])), column] < 0:
            value[:, column] *= -1.
    return value


def inverse_decoder_from_deployed_encoder(encoder):
    e32 = np.ascontiguousarray(encoder, dtype=np.float32)
    if e32.ndim != 2 or not e32.shape[1] or not np.isfinite(e32).all():
        raise ValueError("FINITE_CK_DEPLOYED_ENCODER_REQUIRED")
    e64 = e32.astype(np.float64)
    singular = np.linalg.svd(e64, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * PINV_RCOND))
    if rank != e32.shape[1]:
        raise ValueError("FULL_COLUMN_RANK_DEPLOYED_ENCODER_REQUIRED")
    decoder32 = np.ascontiguousarray(np.linalg.pinv(e64, rcond=PINV_RCOND), dtype=np.float32)
    if not np.isfinite(decoder32).all():
        raise ValueError("FINITE_FP32_INVERSE_DECODER_REQUIRED")
    return decoder32, {"rule": "pinv(E64 cast FP32 interpreted as FP64,rcond=1e-15) cast FP32",
                       "rank_deployed_encoder": rank, "pinv_rcond": PINV_RCOND,
                       "encoder_FP32": _binding(e32), "decoder_FP32": _binding(decoder32),
                       "future_labels_used_to_construct_inverse_decoder": False}


def fit_target_subspace(history, history_observed, FIT_target, FIT_target_observed,
                        raw_last, origins, k):
    x, past_observed, target, observed, current, origin = (
        np.ascontiguousarray(value) for value in
        (history, history_observed, FIT_target, FIT_target_observed, raw_last, origins))
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != LENGTH or not len(x)
            or not x.shape[2] or not np.isfinite(x).all()):
        raise ValueError("FINITE_NONEMPTY_FP32_N512C_FIT_HISTORY_REQUIRED")
    n, _, channels = x.shape
    if past_observed.dtype != np.bool_ or past_observed.shape != x.shape:
        raise ValueError("EXACT_BOOL_FIT_HISTORY_OBSERVED_REQUIRED")
    if target.dtype != np.float32 or target.shape != (n, HORIZON, channels):
        raise ValueError("EXACT_FP32_N48C_FIT_TARGET_REQUIRED")
    if observed.dtype != np.bool_ or observed.shape != target.shape:
        raise ValueError("EXACT_BOOL_FIT_TARGET_OBSERVED_REQUIRED")
    if (current.dtype != np.float32 or current.shape != (n, channels)
            or not np.isfinite(current).all()
            or not np.array_equal(current, x[:, -1])):
        raise ValueError("EXACT_FP32_CURRENT_C_FROM_FIT_HISTORY_LAST_REQUIRED")
    if origin.dtype != np.int64 or origin.shape != (n,):
        raise ValueError("EXACT_INT64_FIT_ORIGINS_REQUIRED")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= channels:
        raise ValueError("VALID_INTEGER_K_REQUIRED")
    k = int(k)
    complete = np.all(observed, axis=2)
    if not complete.any():
        raise ValueError("COMPLETE_C_FIT_TARGET_POSITIONS_REQUIRED")
    known = target[complete].astype(np.float64)
    if not np.isfinite(known).all():
        raise ValueError("FINITE_COMPLETE_C_FIT_TARGET_REQUIRED")
    anchors = np.broadcast_to(current[:, None], target.shape)[complete].astype(np.float64)
    innovation = known - anchors
    _, target_singular, target_right = np.linalg.svd(innovation, full_matrices=False)
    target_rank = int(np.sum(target_singular > target_singular[0] * PINV_RCOND))
    if target_rank < k:
        raise ValueError(f"TARGET_INNOVATION_RANK_BELOW_K:{target_rank}<{k}")
    directions = _canonical_columns(target_right[:k].T)
    flat = x.astype(np.float64).reshape(-1, channels)
    center = flat.mean(axis=0)
    centered = flat - center
    projected = centered @ directions
    restricted_covariance = projected.T @ projected / len(flat)
    eigenvalues, eigenvectors = np.linalg.eigh(restricted_covariance)
    maximum = float(eigenvalues[-1])
    if maximum <= 0 or not np.isfinite(eigenvalues).all():
        raise ValueError("RESTRICTED_PAST_VARIANCE_ZERO_OR_NONFINITE")
    rank = int(np.sum(eigenvalues > maximum * EIGENVALUE_RELATIVE_CUT))
    if rank < k:
        raise ValueError(f"RESTRICTED_PAST_WHITENING_RANK_BELOW_K:{rank}<{k}")
    order = np.argsort(eigenvalues)[::-1]
    encoder = _canonical_columns(directions @ eigenvectors[:, order] / np.sqrt(eigenvalues[order])[None])
    latent = centered @ encoder
    latent_covariance = latent.T @ latent / len(flat)
    if not np.allclose(latent_covariance, np.eye(k), atol=1e-7, rtol=0):
        raise ValueError("TARGET_SPACE_UNIT_FIT_LATENT_VARIANCE_REQUIRED")
    decoder32, decoder_metadata = inverse_decoder_from_deployed_encoder(encoder)
    top_energy = float(np.sum(target_singular[:k] ** 2))
    total_energy = float(np.sum(target_singular ** 2))
    gap = (float((target_singular[k - 1] - target_singular[k]) / target_singular[0])
           if k < len(target_singular) else None)
    bindings = {name: _binding(value) for name, value in zip(
        ("history", "history_observed", "FIT_target", "FIT_target_observed", "raw_last", "origins"),
        (x, past_observed, target, observed, current, origin))}
    metadata = {
        "route": "FIT_TARGET_SUBSPACE", "learning_class": "SUPERVISED_FIT_TARGET_SUBSPACE",
        "channels": channels, "K": k, "length": LENGTH, "horizon": HORIZON,
        "fit_windows": n, "fit_positions_with_overlap": len(flat),
        "fit_complete_future_positions": int(complete.sum()),
        "fit_total_future_positions": int(complete.size),
        "fit_complete_future_positions_per_horizon": complete.sum(axis=0).tolist(),
        "FIT_history_observed_counts": past_observed.sum(axis=(0, 1)).tolist(),
        "FIT_target_observed_counts": observed.sum(axis=(0, 1)).tolist(),
        "FIT_origins_unique": int(len(np.unique(origin))),
        "fit_input_bindings": bindings, "input_binding_hash_rule": "SHA256 contiguous raw bytes; shape and dtype stored separately",
        "complete_future_mask": _binding(complete), "FIT_target_innovation": _binding(innovation),
        "target_output_directions": directions.tolist(),
        "target_singularvalues": target_singular.tolist(), "target_numerical_rank": target_rank,
        "target_rank_relative_cut": PINV_RCOND,
        "FIT_target_innovation_total_energy": total_energy,
        "FIT_target_top_K_retained_energy": top_energy,
        "FIT_target_top_K_retained_fraction": top_energy / total_energy,
        "target_K_boundary_relative_singular_gap": gap,
        "target_basis_canonical_sign": "Largest absolute coordinate positive; tied singular spaces need not be unique across numerical libraries",
        "target_objective": "Uncentered rank-K SVD of complete-C FIT actual Y minus fixed current-C anchor; no future mean or intercept",
        "whitening_rank": rank, "eigenvalue_relative_cut": EIGENVALUE_RELATIVE_CUT,
        "restricted_past_covariance": restricted_covariance.tolist(),
        "retained_whitening_eigenvalues": eigenvalues[order].tolist(),
        "latent_fit_covariance": latent_covariance.tolist(),
        "center_covariance_scope": "All causal-forward-filled FIT history positions, including filled values and repeated overlapping positions; global past mean",
        "decoder_objective": "No decoder fit; identical deployed-encoder pseudoinverse rule as PCA_ENCODER_PINV",
        "decoder_construction": decoder_metadata,
        "decoder_target_scope": "No reconstruction-target or future-target least-squares decoder fitting",
        "encoder_fit_dtype": "float64", "serialized_state_dtype": "float64",
        "decoder_serialization_scope": "Exact FP32 deployment D values stored as FP64; no higher precision D at deployment",
        "deployment_dtype": "float32",
        "forecast_formula": "raw_current_C + (predicted_future_K - last_observed_K) @ decoder_FP32; bias cancels",
        "deployment_input_scope": "Past-only pointwise E(X); FIT future labels were used solely to train the frozen output space",
        "actual_future_target_used_in_projection_or_decoder_fit": True,
        "TUNE_or_DEV_used_in_projection_or_decoder_fit": False,
        "target_output_space_is_FIT_only": True, "new_graph_or_contrastive_learning": False,
        "independent_holdout": False, "strict_K_only_compression": False,
        "method_checkpoint_branch_selection_used_by_this_fit": False,
        "claim_scope": "Supervised FIT target-space control, not a new-method novelty claim or a deployable use of per-split future oracle directions"}
    return ProjectionState(center, encoder, decoder32.astype(np.float64), center.copy(), metadata)


def pca_encoder_pinv_control(original_state):
    if original_state.fit_metadata.get("route") != "PCA":
        raise ValueError("AUDITED_ORIGINAL_PCA_STATE_REQUIRED")
    encoder = np.array(original_state.encoder, dtype=np.float64, order="C", copy=True)
    center = np.array(original_state.center, dtype=np.float64, order="C", copy=True)
    decoder32, decoder_metadata = inverse_decoder_from_deployed_encoder(encoder)
    metadata = {"route": "PCA_ENCODER_PINV", "learning_class": "UNSUPERVISED_FIT_PAST_PCA",
                "channels": len(center), "K": encoder.shape[1], "length": LENGTH, "horizon": HORIZON,
                "source_original_PCA_state_metadata": original_state.metadata(),
                "encoder_center_and_whitening_unchanged": True,
                "latent_fit_covariance": original_state.fit_metadata["latent_fit_covariance"],
                "decoder_construction": decoder_metadata,
                "decoder_objective": "No decoder fit; same encoder inverse rule as FIT_TARGET_SUBSPACE",
                "actual_future_target_used_in_projection_or_decoder_fit": False,
                "TUNE_or_DEV_used_in_projection_or_decoder_fit": False,
                "forecast_formula": "raw_current_C + (predicted_future_K - last_observed_K) @ decoder_FP32; bias cancels",
                "deployment_input_scope": "Past-only pointwise unchanged original PCA E(X)",
                "native_head_retraining_required_for_optimized_comparison": True,
                "original_LS_trained_native_head_is_an_optimized_control": False,
                "independent_holdout": False, "strict_K_only_compression": False}
    return ProjectionState(center, encoder, decoder32.astype(np.float64), center.copy(), metadata)
