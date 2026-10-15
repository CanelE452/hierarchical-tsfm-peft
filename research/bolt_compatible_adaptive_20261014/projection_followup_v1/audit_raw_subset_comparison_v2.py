"""Independent direct-residual v2 subset/global D and saved CP audit; no F replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import sys

import audit_projection_results_v1 as prior
import audit_local_past_decoder_probe_v1 as local_audit

HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
SPACES = {"PCA": "PCA_PINV", "FIT_TARGET": "FIT_TARGET_PINV", "RAW_SUBSET": None}
SEEDS, CP_UPDATES = (92601, 92602), (0, 100, 200, 400, 600)
RCOND = 1e-4
np = sys.modules.get("numpy")
require, exact = prior.require, local_audit.exact
raw_binding = local_audit.raw_binding


def initialize_numpy():
    global np
    local_audit.initialize_numpy(); np = sys.modules["numpy"]


def validate_past(history, observed):
    require(history.dtype == np.float32 and history.ndim == 3 and len(history) > 0 and history.shape[1] == 512 and history.shape[2] > 0 and np.isfinite(history).all() and observed.dtype == np.bool_ and observed.shape == history.shape, "FINITE_FP32_PAST_512_TRUE_OBSERVED_MASK")


def direct_subset_scores(history, observed, indices):
    validate_past(history, observed)
    c = history.shape[-1]
    require(len(indices) > 0 and len(set(indices)) == len(indices) and all(type(index) is int and 0 <= index < c for index in indices), "DISTINCT_RAW_CHANNEL_INDICES")
    flat, mask = history.reshape(-1, c).astype(np.float64), observed.reshape(-1, c)
    candidate_features = flat[:, indices]
    errors, ranks, conditions, spectra, counts = [], [], [], [], mask.sum(axis=0)
    for channel in range(c):
        active = mask[:, channel]
        if not active.any(): errors.append(None); ranks.append(None); conditions.append(None); spectra.append(None); continue
        z, y = candidate_features[active], flat[active, channel]
        z, y = z - z.mean(axis=0), y - y.mean()
        slope, _, rank, singular = np.linalg.lstsq(z, y, rcond=RCOND)
        errors.append(float(np.mean(np.square(z @ slope - y))))
        ranks.append(int(rank)); conditions.append(float(singular[0] / singular[rank - 1]) if rank else None); spectra.append(singular.tolist())
    active = counts > 0
    require(active.any(), "NO_OBSERVED_PAST_RESPONSE_CHANNELS")
    return {"macro_normalized_MSE": float(np.mean([errors[i] for i in range(c) if active[i]])), "per_channel_MSE": errors, "true_observed_counts": counts.tolist(), "active_channels": int(active.sum()), "retained_centered_design_rank": ranks, "retained_centered_design_condition": conditions, "centered_design_singularvalues": spectra, "indices": list(indices)}


def check_subset_trace(lock, history, observed, k, names=None):
    require(lock["status"] == "LOCKED_FIT_PAST_ONLY" and lock["FIT_source"]["arrays"]["history"] == raw_binding(history) and lock["FIT_source"]["arrays"]["history_observed"] == raw_binding(observed), "SUBSET_LOCK_SAME_ORIGINAL_FIT_PAST_BINDING")
    metadata, c = lock["selection_metadata"], history.shape[-1]
    require(metadata["K"] == k and metadata["channels"] == c and len(metadata["stages"]) == k and metadata["future_TUNE_DEV_or_F_scores_used"] is metadata["forecast_objective_used"] is metadata["optimal_subset_guaranteed"] is metadata["whitening_used"] is metadata["direct_slice_dense_onehot_byte_identity_claimed"] is False, "GREEDY_FIT_PAST_ONLY_NO_FORECAST_OPTIMALITY_OR_WHITENING")
    stats = metadata["FIT_past_statistics"]
    require(stats["history"] == raw_binding(history) and stats["observed"] == raw_binding(observed) and stats["response_specific_observed_mask_for_centered_statistics"] is stats["filled_missing_values_are_features_only_not_response_labels"] is True and stats["duplicated_window_rows_independent_samples_claimed"] is stats["future_TUNE_DEV_or_F_scores_used"] is stats["Gram_cross_variance_expansion_used_for_scores"] is False and metadata["global_decoder_fitter_unchanged_v1_function"] is True, "RESPONSE_SPECIFIC_TRUE_OBSERVATION_DIRECT_RESIDUAL_STATS_SCOPE")
    order, stages, evaluated = [], [], 0
    for stage_index, stage in enumerate(metadata["stages"], 1):
        rows = stage["candidates"]
        remaining = set(range(c)) - set(order)
        require(len(rows) == len(remaining) and {row["candidate_original_index"] for row in rows} == remaining and stage["stage"] == stage_index, "ALL_REMAINING_GREEDY_CANDIDATES_RECORDED")
        independent = {}
        for row in rows:
            index = row["candidate_original_index"]; chosen = sorted([*order, index])
            actual = direct_subset_scores(history, observed, chosen)
            require(row["indices"] == chosen and row["singular_relative_cutoff"] == RCOND and row["Gram_eigen_relative_cutoff_equivalent_in_real_arithmetic_only"] == 1e-8 and row["Gram_cross_variance_expansion_used_for_scores"] is False and row["true_observed_counts"] == actual["true_observed_counts"] and row["active_channels"] == actual["active_channels"] and row["retained_centered_design_rank"] == actual["retained_centered_design_rank"], "DIRECT_LS_COUNTS_RANK_CUTOFF_AND_INDICES")
            require(row["negative_roundoff_MSE_values_clipped_to_zero"] == 0 and all(value is None or (math.isfinite(value) and value >= 0) for value in row["per_channel_MSE"]), "DIRECT_SQUARED_RESIDUALS_WITHOUT_CLIPPING")
            prior.close(row["macro_normalized_MSE"], float(np.mean([value for value in row["per_channel_MSE"] if value is not None])), "recorded_subset_equal_channel_macro", atol=1e-10, rtol=0)
            for field in ("macro_normalized_MSE", "per_channel_MSE", "retained_centered_design_condition", "centered_design_singularvalues"):
                recorded, target = row[field], actual[field]
                if isinstance(target, list):
                    for left, right in zip(recorded, target):
                        if right is None: require(left is None, "INACTIVE_CHANNEL_NONE_SCORE")
                        else: prior.close(left, right, "subset_candidate:" + field, atol=1e-10, rtol=0 if field != "retained_centered_design_condition" else 1e-8)
                else: prior.close(recorded, target, "subset_candidate:" + field, atol=1e-10, rtol=0)
            independent[index] = actual["macro_normalized_MSE"]; evaluated += 1
        winner = min(rows, key=lambda row: (row["macro_normalized_MSE"], row["candidate_original_index"]))
        require(stage["selected_original_index"] == winner["candidate_original_index"] and stage["selected_macro_normalized_MSE"] == winner["macro_normalized_MSE"], "EXACT_RECORDED_FP64_SCORE_THEN_LOWEST_INDEX_WINNER")
        order.append(winner["candidate_original_index"])
        require(stage["selected_so_far_selection_order"] == order and stage["selected_so_far_original_order"] == sorted(order), "GREEDY_ORDER_TRACE")
        sorted_scores = sorted(independent.items(), key=lambda pair: (pair[1], pair[0]))
        margin = sorted_scores[1][1] - sorted_scores[0][1] if len(sorted_scores) > 1 else None
        stages.append({"stage": stage_index, "selected_original_index": order[-1], "direct_LS_best_index": sorted_scores[0][0], "direct_LS_top2_score_margin": margin, "arithmetic_margin_within_two_registered_score_tolerances": margin is not None and margin <= 2e-10, "winner_not_reselected_by_auditor": True})
    require(order == lock["selection_order"] and sorted(order) == lock["selected_indices"], "FINAL_SUBSET_LOCK_AND_ORDER")
    require(metadata["final_reconstruction_scores"] == {key: value for key, value in winner.items() if key != "candidate_original_index"}, "FINAL_SCORES_SAME_LAST_SELECTED_CANDIDATE")
    final = direct_subset_scores(history, observed, sorted(order))
    prior.close(metadata["final_reconstruction_scores"]["macro_normalized_MSE"], final["macro_normalized_MSE"], "subset_final_direct_LS", atol=1e-10, rtol=0)
    if names is not None: require(lock["selected_channel_names"] == [names[index] for index in sorted(order)], "ORIGINAL_PHYSICAL_FIT_CHANNEL_NAMES")
    return {"candidate_scores_checked": evaluated, "stages": stages, "selected_indices": sorted(order), "selection_order": order, "independent_direct_observed_centered_LS": True, "future_forecast_quality_or_global_subset_optimality_proven": False}


def independent_global_decoder(history, observed, latent):
    validate_past(history, observed)
    require(latent.dtype == np.float32 and latent.ndim == 3 and latent.shape[:2] == history.shape[:2] and 1 <= latent.shape[-1] <= history.shape[-1] and np.isfinite(latent).all(), "FINITE_ACTUAL_FP32_GLOBAL_D_FIT_PAST")
    c, k = history.shape[-1], latent.shape[-1]
    x, z, mask = history.reshape(-1, c).astype(np.float64), latent.reshape(-1, k).astype(np.float64), observed.reshape(-1, c)
    decoder, bias = np.zeros((k, c), np.float64), np.zeros(c, np.float64)
    counts, ranks, conditions, spectra, errors, fallbacks = mask.sum(axis=0), [], [], [], [], []
    zeros_z, zeros_y, deficient, identified, executed, learned, certificates = [], [], [], [], [], [], []
    for channel in range(c):
        active = mask[:, channel]
        if not active.any():
            ranks.append(0); conditions.append(None); spectra.append([]); errors.append(None); fallbacks.append("NO_OBSERVED_RESPONSE_ZERO_SLOPE_ZERO_BIAS_UNLEARNED")
            zeros_z.append(None); zeros_y.append(None); deficient.append(None); identified.append(False); executed.append(False); learned.append(False); certificates.append({"observed_count": 0}); continue
        features, response = z[active], x[active, channel]
        mean_z, mean_y = features.mean(axis=0), response.mean()
        centered, target = features - mean_z, response - mean_y
        u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        retained = singular > singular[0] * RCOND if singular.size else np.zeros(0, bool)
        rank = int(retained.sum())
        slope = vh[retained].T @ ((u[:, retained].T @ target) / singular[retained]) if rank else np.zeros(k)
        zero_z, zero_y = bool(np.all(centered == 0)), bool(np.all(target == 0))
        fallback = "ONE_OBSERVED_RESPONSE_ZERO_UNIDENTIFIED_SLOPE_MEAN_INTERCEPT" if len(features) == 1 else "ZERO_CENTERED_PREDICTOR_VARIANCE_ZERO_UNIDENTIFIED_SLOPE_MEAN_INTERCEPT" if zero_z else None
        if fallback is not None: slope[:] = 0.
        direct, _, direct_rank, _ = np.linalg.lstsq(centered, target, rcond=RCOND)
        require(int(direct_rank) == rank, "INDEPENDENT_SVD_DIRECT_LS_RETAINED_RANK")
        prior.close(slope, direct, "independent_global_SVD_and_direct_LS", atol=1e-10, rtol=1e-10)
        intercept = float(mean_y - mean_z @ slope)
        error = features @ slope + intercept - response
        normal = centered.T @ (centered @ slope - target)
        retained_normal = vh[retained] @ normal
        tolerance = 1e-10 * (1 + float(np.linalg.norm(centered)) * float(np.linalg.norm(target)))
        require(not rank or np.max(np.abs(retained_normal)) <= tolerance, "GLOBAL_D_RETAINED_NORMAL_RESIDUAL")
        null_component = (np.eye(k) - vh[retained].T @ vh[retained]) @ slope
        require(np.max(np.abs(null_component)) <= 1e-10 * (1 + float(np.linalg.norm(slope))), "GLOBAL_CENTERED_SLOPE_MINIMUM_NORM_NULLSPACE_CERTIFICATE")
        decoder[:, channel], bias[channel] = slope, intercept
        ranks.append(rank); conditions.append(float(singular[0] / singular[rank - 1]) if rank else None); spectra.append(singular.tolist()); errors.append(float(np.mean(error * error))); fallbacks.append(fallback)
        zeros_z.append(zero_z); zeros_y.append(zero_y); deficient.append(rank < k); identified.append(rank == k); executed.append(True); learned.append(fallback is None and rank > 0)
        certificates.append({"observed_count": int(counts[channel]), "retained_normal_residual_max_abs": float(np.max(np.abs(retained_normal))) if rank else 0., "normal_tolerance": tolerance, "centered_slope_nullspace_max_abs": float(np.max(np.abs(null_component)))})
    return {"decoder_FP64": decoder, "decoder_FP32": decoder.astype(np.float32), "bias_FP64": bias, "bias_FP32": bias.astype(np.float32), "metadata": {"true_observed_counts": counts.tolist(), "retained_centered_design_rank": ranks, "retained_centered_design_condition": conditions, "centered_design_singularvalues": spectra, "observed_past_reconstruction_MSE_FP64": errors, "fallback_reason": fallbacks, "learned_slope": learned, "zero_centered_predictor_variance": zeros_z, "constant_observed_response": zeros_y, "no_observed_responses": (counts == 0).tolist(), "one_observed_response": (counts == 1).tolist(), "rank_deficient_slope": deficient, "slope_identified_in_retained_full_K_space": identified, "centered_lstsq_fit_executed": executed}, "certificates": certificates}


def check_global_state(metadata, state, history, observed, latent):
    require(set(state) == {"decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32"} and metadata["singular_relative_cutoff"] == RCOND and metadata["future_TUNE_DEV_or_F_scores_used"] is metadata["decoder_fitted_at_each_evaluation_origin"] is metadata["no_observation_fallback_uses_future_or_current_values"] is False and metadata["constant_response_is_normal_fitted_zero_slope_solution"] is True, "FIT_FIXED_D_KNOWN_OBSERVED_PAST_ONLY_FALLBACK_SCOPE")
    require(metadata["fit_input_bindings"] == {"history": raw_binding(history), "observed": raw_binding(observed), "latent_history": raw_binding(latent)}, "GLOBAL_D_EXACT_FIT_INPUT_BINDINGS")
    expected = independent_global_decoder(history, observed, latent)
    for name in state:
        require(metadata["state_array_bindings"][name] == raw_binding(state[name]), "GLOBAL_D_STATE_ARRAY_SHA:" + name)
        if name.endswith("FP64"): prior.close(state[name], expected[name], "independent_global_D:" + name, atol=1e-10, rtol=1e-10)
        else: exact(state[name], state[name.replace("FP32", "FP64")].astype(np.float32), "global_D_exact_deployment_cast:" + name)
    numeric = {"retained_centered_design_condition", "centered_design_singularvalues", "observed_past_reconstruction_MSE_FP64"}
    for name, actual in expected["metadata"].items():
        if name not in numeric: require(metadata[name] == actual, "GLOBAL_D_EXACT_COUNT_FLAG_OR_RANK:" + name); continue
        for left, right in zip(metadata[name], actual):
            if right is None: require(left is None, "GLOBAL_D_NO_OBSERVATIONS_NONE_VALUE")
            else: prior.close(left, right, "global_D_certificate:" + name, atol=1e-10, rtol=1e-10)
    return {"observed_response_channels_checked": int(sum(expected["metadata"]["centered_lstsq_fit_executed"])), "certificates": expected["certificates"], "constant_response_identified_differentiated_from_unobserved_or_unidentified": True, "independent_SVD_and_direct_LS_checked": True}


def independent_native(payload, state):
    require(set(state) == {"decoder", "head.weight", "head.bias"}, "HEAD_AND_FROZEN_FIT_DECODER_CHECKPOINT_KEYS")
    z, current = payload["latent_history"], payload["raw_last"]
    d, w, b = state["decoder"], state["head.weight"], state["head.bias"]
    require(z.dtype == current.dtype == d.dtype == w.dtype == b.dtype == np.float32 and z.ndim == 3 and z.shape[1] == 512 and len(z) > 0 and d.shape == (z.shape[-1], current.shape[-1]) and w.shape == (z.shape[-1], 48, 512) and b.shape == (z.shape[-1], 48) and all(np.isfinite(value).all() for value in (z, current, d, w, b)), "FINITE_FP32_FIXED_D_NATIVE_CONTRACT")
    import torch
    torch.set_num_threads(2)
    d, w, b = (torch.from_numpy(np.ascontiguousarray(value)) for value in (d, w, b))
    predictions, futures = [], []
    with torch.no_grad():
        for start in range(0, len(z), 32):
            history = torch.from_numpy(np.ascontiguousarray(z[start:start + 32])); anchor = torch.from_numpy(np.ascontiguousarray(current[start:start + 32]))
            last = history[:, -1:]
            future = last + (torch.einsum("nlk,khl->nhk", history - last, w) + b.transpose(0, 1).unsqueeze(0))
            predictions.append((anchor[:, None] + (future - last) @ d).numpy().copy()); futures.append(future.numpy().copy())
    return tuple(np.ascontiguousarray(np.concatenate(values)) for values in (predictions, futures))


def encode(history, encoder_state, backend):
    indices = encoder_state["selected_indices"].tolist()
    if indices: return np.ascontiguousarray(history[..., indices])
    if backend == "BOLT_NUMPY": return prior.encode(history, encoder_state["center"], encoder_state["encoder"])
    require(backend == "NATIVE_TORCH32", "DECLARED_ENCODER_BACKEND")
    import torch
    torch.set_num_threads(2)
    center, matrix = (torch.from_numpy(np.ascontiguousarray(encoder_state[key].astype(np.float32))) for key in ("center", "encoder"))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(history), 32):
            value = torch.from_numpy(np.ascontiguousarray(history[start:start + 32]))
            chunks.append(((value - center) @ matrix).numpy().copy())
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def fixed_forecast(current, future, last, decoder, backend):
    if backend == "BOLT_NUMPY": return current[:, None] + (future - last[:, None]) @ decoder
    require(backend == "NATIVE_TORCH32", "DECLARED_FIXED_D_FORECAST_BACKEND")
    import torch
    torch.set_num_threads(2)
    d = torch.from_numpy(np.ascontiguousarray(decoder))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(current), 32):
            anchor, f, z = (torch.from_numpy(np.ascontiguousarray(value[start:start + 32])) for value in (current, future, last))
            chunks.append((anchor[:, None] + (f - z[:, None]) @ d).numpy().copy())
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def new_fixed_space(values, decoder):
    mask = prior.oracle_mask(values["observed"])
    if not mask.any(): return {"status": "NO_COMPLETE_C_TARGET_POSITIONS", "RAW_bound_applied": False}
    complete = values["observed"].all(axis=2)
    target, current = values["target"], values["raw_last"]
    d = decoder.astype(np.float64)
    u, singular, vh = np.linalg.svd(d, full_matrices=False)
    retained = singular > singular[0] * 1e-15 if singular.size else np.zeros(0, bool)
    rank = int(retained.sum()); c = d.shape[1]
    projector = vh[retained].T @ vh[retained] if rank else np.zeros((c, c))
    require(np.max(np.abs(projector - projector.T)) <= 1e-10 and np.max(np.abs(projector @ projector - projector)) <= 1e-10 and abs(float(np.trace(projector)) - rank) <= 1e-10, "NEW_FIXED_D_PROJECTOR_GEOMETRY")
    innovation = (target.astype(np.float64) - current.astype(np.float64)[:, None])[complete]
    projected = innovation @ projector
    coords, _, ls_rank, _ = np.linalg.lstsq(d.T, innovation.T, rcond=1e-15)
    require(int(ls_rank) == rank, "NEW_FIXED_D_SVD_LS_DIAGNOSTIC_RANK")
    prior.close(coords.T @ d, projected, "new_fixed_D_independent_projector_LS", atol=1e-10, rtol=1e-10)
    residual = innovation - projected
    normal = residual @ d.T
    tolerance = 1e-10 * (1 + float(np.linalg.norm(innovation)) * float(np.linalg.norm(d)))
    require(np.max(np.abs(normal)) <= tolerance, "NEW_FIXED_D_DIAGNOSTIC_NORMAL_CERTIFICATE")
    kept_energy, lost_energy = float(np.square(projected).sum()), float(np.square(residual).sum())
    prior.close(float(np.square(innovation).sum()), kept_energy + lost_energy, "new_fixed_D_energy_decomposition", atol=1e-10, rtol=1e-10)
    prediction = np.broadcast_to(current.astype(np.float64)[:, None], target.shape).copy()
    prediction[complete] += projected
    score = prior.scores(prediction, target, mask, values["std"])
    minimum = lost_energy / (int(complete.sum()) * c)
    prior.close(score["normalized"]["mse"], minimum, "new_fixed_D_complete_equal_channel_denominator", atol=1e-10, rtol=1e-10)
    return {"status": "COMPUTED_DIAGNOSTIC_ONLY", "minimum_normalized_MSE_real_D32_numeric_rowspace": minimum, "minimum_solution_scores_complete_C": score, "numeric_rank": rank, "singularvalues_D32_as_real64": singular.tolist(), "complete_target_positions": int(complete.sum()), "relative_numeric_rowspace_cutoff": 1e-15, "normal_residual_max_abs": float(np.abs(normal).max()), "normal_tolerance": tolerance, "retained_energy": kept_energy, "discarded_energy": lost_energy, "RAW_scores_same_complete_C": prior.scores(values["raw_forecast"], target, mask, values["std"]), "minimum_only_normalized_MSE": True, "MAE_or_physical_MSE_minimum_claimed": False, "strict_all_FP32_output_lower_bound_claimed": False, "RAW_bound_applied": False, "old_output_space_floor_reused": False}


def diagnostic_trace(prediction, current, future, last, decoder, values, space, label):
    if space["status"] == "NO_COMPLETE_C_TARGET_POSITIONS": return {"status": space["status"]}
    mask = prior.oracle_mask(values["observed"])
    real = current.astype(np.float64)[:, None] + (future.astype(np.float64) - last.astype(np.float64)[:, None]) @ decoder.astype(np.float64)
    actual_score = prior.scores(prediction, values["target"], mask, values["std"])
    real_score = prior.scores(real, values["target"], mask, values["std"])
    actual_mse, real_mse = actual_score["normalized"]["mse"], real_score["normalized"]["mse"]
    rounding = abs(actual_mse - real_mse); floor = space["minimum_normalized_MSE_real_D32_numeric_rowspace"]
    require(real_mse + 1e-10 >= floor and actual_mse + rounding + 1e-10 >= floor, "NEW_FIXED_D_REAL_TRACE_ONLY_BOUND_WITH_MEASURED_FP32_ALLOWANCE:" + label)
    return {"scores_complete_C": actual_score, "real_trace_scores_complete_C": real_score, "measured_FP32_rounding_normalized_MSE_allowance": rounding, "actual_FP32_minus_minimum": actual_mse - floor, "real_trace_minus_minimum": real_mse - floor}


def score_checked(recorded, prediction, values, label):
    output, count = {}, 0
    for name, mask in (("scores_all_observed", values["observed"]), ("scores_complete_mask", prior.oracle_mask(values["observed"]))):
        if not mask.any():
            require(recorded[name]["status"] == "NO_OBSERVED_TARGETS" and recorded[name]["normalized"] is recorded[name]["physical_per_channel_diagnostic"] is None, "NO_COMPLETE_OBSERVATIONS_EXPLICIT_SCORE_STATE")
            output[name] = recorded[name]; continue
        scores, checked = local_audit.score_checked(recorded[name], prediction, values["target"], mask, values["std"], label + ":" + name)
        output[name], count = scores, count + checked
    return output, count


def source_contract(records):
    paths = {Path(item["path"]).name: Path(item["path"]) for item in records}
    require({"raw_subset_projection_v1.py", "raw_subset_projection_v2.py", "raw_subset_native_head_v1.py", "run_raw_subset_comparison_v2.py", "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md"}.issubset(paths), "NEW_SUBSET_SOURCES_AND_PLAN")
    pure = ast.parse(paths["raw_subset_projection_v2.py"].read_text(encoding="utf-8-sig"))
    functions = {node.name: node for node in pure.body if isinstance(node, ast.FunctionDef)}
    old = ast.parse(paths["raw_subset_projection_v1.py"].read_text(encoding="utf-8-sig"))
    old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    require([arg.arg for arg in functions["greedy_raw_subset"].args.args] == ["history", "observed", "K"] and [arg.arg for arg in old_functions["fit_global_observed_decoder"].args.args] == ["history", "observed", "latent_history"], "NO_FUTURE_TUNE_DEV_OR_F_SCORE_FIT_ARGUMENTS")
    score_text = ast.unparse(functions["score_subset"])
    require("np.linalg.lstsq(centered, target, rcond=RCOND)" in score_text and "np.mean(residual * residual)" in score_text and "np.linalg.eigh" not in score_text and "max(0.0, error)" not in score_text, "DIRECT_LS_EXPLICIT_RESIDUAL_REPAIR_NO_CLIPPING_OR_GRAM_SCORE")
    native_text = paths["raw_subset_native_head_v1.py"].read_text(encoding="utf-8-sig"); native_tree = ast.parse(native_text)
    cls = next(node for node in native_tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeFixedDecoder")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "components")
    require([arg.arg for arg in method.args.args] == ["self", "latent_history", "raw_last"] and "self.decoder.detach()" in ast.unparse(method) and "latent_history.detach()" in ast.unparse(method), "FIXED_D_PAST_ONLY_HEAD_FORWARD")
    trainer = next(node for node in native_tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_native_fixed")
    require("torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.0)" in ast.unparse(trainer), "HEAD_ONLY_REGISTERED_OPTIMIZER")
    runner = ast.parse(paths["run_raw_subset_comparison_v2.py"].read_text(encoding="utf-8-sig"))
    execute = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    context = next(node for node in execute.body if isinstance(node, ast.FunctionDef) and node.name == "context")
    source = ast.unparse(context)
    require("bolt(engine.tensor(z_numpy[start:start + 8]))" in source and "range(0, len(z_numpy), 8)" in source and "'full_C_forecast_slice_used_as_new_inference': False" in source, "ACTUAL_NEW_K_SUBSET_F_CALL_NOT_CACHED_FULL_C_SLICE")
    require("'ACTUAL_NEW_SUBSET_F_INFERENCE'" in source and "'original_F_cache'" in source and "'latent_future'" in source, "NEW_SUBSET_F_AND_MIXED_CACHE_LINEAGE_BRANCHES")
    return {"past_only_greedy_and_global_decoder_API": True, "fresh_head_only_optimizer": True, "actual_new_subset_F_call_source_checked": True, "F_calls_independently_replayed": False, "learning_trajectories_independently_replayed": False}


def check_registry(registry, events, locks):
    expected = {(dataset, space, str(seed)) for dataset in DATASETS for space in SPACES for seed in SEEDS}
    require(registry["status"] == "ALL_TWELVE_RESOLVED_BEFORE_DEV" and len(registry["conditions"]) == 12 and set(locks) == expected and {(row["dataset"], row["space"], row["seed"]) for row in registry["conditions"]} == expected, "ALL_TWELVE_CONDITION_REGISTRY_BEFORE_DEV")
    for row in registry["conditions"]: require(row["lock"] == locks[(row["dataset"], row["space"], row["seed"])][0], "REGISTRY_EXACT_SELECTION_FILE_BINDING")
    require([row["sequence"] for row in events] == list(range(len(events))), "CONTIGUOUS_EXECUTION_EVENTS")
    seal = [row for row in events if row["kind"] == "ALL_TWELVE_LOCKS_AND_REGISTRY_SAVED"]
    require(len(seal) == 1, "SINGLE_ALL_TWELVE_RESOLVED_LOCK_GATE")
    for row in events:
        if row["kind"] == "DEV_INPUT_LOAD_AFTER_ALL_TWELVE_LOCKS" or (row["kind"] == "ACTUAL_NEW_SUBSET_F_INFERENCE" and row["split"] in ("val", "eval")):
            require(row["sequence"] > seal[0]["sequence"], "NO_NEW_DEV_INPUT_OR_SUBSET_F_BEFORE_ALL_LOCKS")
    return {"resolved_native_selection_locks_checked": 12, "registry_before_any_new_DEV_checked": True}


def check_native(entry, lock, contexts, decoder, seed, k, c, development, check):
    metadata = entry["metadata"]
    require(entry["status"] == "COMPLETE" and metadata["seed"] == seed and metadata["updates"] == 600 and metadata["batch_size"] == 4 and metadata["L"] == 512 and metadata["H"] == 48 and metadata["K"] == k and metadata["C"] == c and metadata["optimizer"] == "AdamW" and metadata["learning_rate"] == .001 and metadata["weight_decay"] == 0 and metadata["gradient_clip"] == 1 and metadata["canonical_prediction_batch_size"] == 32 and metadata["checkpoint_updates"] == list(CP_UPDATES), "FRESH_STATIC_D_NATIVE_REGISTERED_TRAINING_CONTRACT")
    require(metadata["initial_head_weights_and_bias_zero"] is metadata["checkpoint_callback_prediction_is_exact_scored_array"] is metadata["fixed_global_FIT_decoder"] is True and metadata["seed_changes_initial_weights"] is metadata["per_origin_decoder_fit_or_rho_selection"] is metadata["decoder_backbone_or_encoder_optimized"] is metadata["old_head_warm_start_or_reuse"] is metadata["future_encoding_in_forward"] is metadata["DEV_loaded_by_trainer"] is metadata["signed_zero_bit_identity_claimed"] is False, "ONLY_ZERO_INITIALIZED_TEMPORAL_HEAD_TRAINED")
    require(metadata["fixed_decoder_before_sha256"] == metadata["fixed_decoder_after_sha256"] == prior.torch_state_sha({"decoder": decoder}), "FIXED_FIT_D_BUFFER_SHA_BEFORE_AFTER")
    bindings = {split: {key: raw_binding(values[key]) for key in ("latent_history", "raw_last", "target", "observed", "origins")} for split, values in contexts.items() if split in ("fit", "tune")}
    require(metadata["payload_array_bindings"] == bindings, "EXACT_FIT_TUNE_HEAD_INPUT_AND_LABEL_BINDINGS")
    order = np.random.default_rng(seed).integers(0, len(contexts["fit"]["latent_history"]), size=(600, 4))
    require(hashlib.sha256(order.tobytes()).hexdigest() == metadata["order_sha256"] and len(entry["learning_curve"]) == 600, "SAME_FIT_ONLY_SEEDED_MINIBATCH_ORDER_600_LOGS")
    for update, row in enumerate(entry["learning_curve"], 1): require(row["update"] == update and math.isfinite(row["loss"]) and row["loss"] >= 0 and math.isfinite(row["gradient_norm_before_clip"]) and row["gradient_norm_before_clip"] >= 0, "FINITE_ACTUAL_NATIVE_UPDATE_LOG")
    require([row["update"] for row in entry["checkpoints"]] == list(CP_UPDATES), "FIVE_ACTUAL_NATIVE_CP_STATES")
    states, predictions, cp_scores, details, checked = {}, {}, [], [], 0
    for point in entry["checkpoints"]:
        state = prior.verified_arrays(point["state_artifact"], check)
        require(prior.torch_state_sha(state) == point["state_sha256"], "NATIVE_CHECKPOINT_STATE_SHA")
        exact(state["decoder"], decoder, "same_new_fixed_D_in_every_CP")
        saved = prior.verified_arrays(point["tune_prediction_artifact"], check)
        require(set(saved) == {"prediction", "origins"}, "CHECKPOINT_TUNE_EXACT_ARRAY_KEYS")
        exact(saved["origins"], contexts["tune"]["origins"], "same_original_CP_TUNE_origins")
        prediction, _ = independent_native(contexts["tune"], state)
        exact(saved["prediction"], prediction, "independent_static_D_Torch32_CP_prediction")
        score = prior.scores(prediction, contexts["tune"]["target"], contexts["tune"]["observed"], contexts["tune"]["std"])["normalized"]
        checked += prior.compare_normalized(point["tune"], score, "static_D_CP_TUNE")
        states[point["update"]], predictions[point["update"]] = state, prediction
        cp_scores.append((score["mse"], point["update"])); details.append({"update": point["update"], "normalized_TUNE": score, "independent_Torch32_prediction_byte_exact": True})
        if point["update"] == 0:
            require(np.count_nonzero(state["head.weight"]) == np.count_nonzero(state["head.bias"]) == 0 and metadata["initial_head_sha256"] == prior.torch_state_sha({"weight": state["head.weight"], "bias": state["head.bias"]}), "FRESH_ZERO_HEAD_INITIAL_STATE")
            require(np.array_equal(prediction, np.broadcast_to(contexts["tune"]["raw_last"][:, None], prediction.shape)) and metadata["CP0_currentC_numeric_max_abs_difference"] == 0., "CP0_NUMERIC_CURRENT_C_PERSISTENCE_NOT_SIGNED_ZERO_BIT_CLAIM")
    selected = min(cp_scores)[1]
    chosen = next(point for point in entry["checkpoints"] if point["update"] == selected)
    literal = min(entry["checkpoints"], key=lambda point: (point["tune"]["mse"], point["update"]))
    require(selected == literal["update"] == metadata["selected_update"] == lock["selected_update"] and metadata["selected_tune"] == lock["selected_tune"] == chosen["tune"] and metadata["selected_state_sha256"] == chosen["state_sha256"] and lock["selected_state_artifact"] == chosen["state_artifact"] and lock["selected_TUNE_prediction_artifact"] == chosen["tune_prediction_artifact"] and lock["DEV_used_for_CP_selection"] is False, "RECOMPUTED_EARLIEST_EXACT_TUNE_MSE_CHECKPOINT_AND_LOCK")
    require(set(entry["scores"]) == set(entry["artifacts"]) == {"fit", "tune", development}, "THREE_ACTUAL_SELECTED_NATIVE_SPLITS")
    result, outputs = {}, {}
    for split, values in contexts.items():
        saved = prior.verified_arrays(entry["artifacts"][split], check)
        require(set(saved) == {"prediction", "latent_future", "origins"}, "SELECTED_NATIVE_FORECAST_KEYS")
        prediction, future = independent_native(values, states[selected])
        exact(saved["origins"], values["origins"], "same_selected_NATIVE_origins")
        exact(saved["prediction"], prediction, "independent_selected_native_prediction_bytes")
        exact(saved["latent_future"], future, "independent_selected_native_latent_future_bytes")
        if split == "tune": exact(prediction, predictions[selected], "selected_TUNE_same_exact_CP_scored_array")
        scores, count = score_checked(entry["scores"][split], prediction, values, "selected_native:" + split)
        checked += count; result[split], outputs[split] = scores, saved
    perturb = entry["future_label_perturbation"]
    require(perturb["prediction_byte_exact"] is perturb["target_and_future_mask_changed"] is True and perturb["windows"] == 2 and perturb["scope"] == "Fixed cached past Z,currentC and FIT-global D only; no upstream raw-future replay or E/D refit", "SCOPED_CACHED_HEAD_PERTURBATION_NOT_GLOBAL_LEAK_PROOF")
    original = {key: value[:2].copy() for key, value in contexts[development].items() if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(contexts[development]["latent_history"])}
    modified = {key: value.copy() for key, value in original.items()}; modified["target"][:] = np.float32(1e6); modified["observed"][:] = False
    a, _ = independent_native(original, states[selected]); b, _ = independent_native(modified, states[selected]); exact(a, b, "independent_fixed_cached_head_future_label_change")
    return {"selected_update": selected, "checkpoints": details, "scores": result, "score_field_checks": checked, "native_fits_checked": 1, "training_updates_logged_checked": 600, "checkpoint_states_checked": 5, "checkpoint_TUNE_prediction_arrays_checked": 5, "selected_native_split_prediction_arrays_checked": 3, "cached_future_label_perturbation_checks": 1, "new_learning_trajectory_independently_replayed": False}, outputs


def load_gates(result_path, check):
    path = Path(result_path).resolve(); binding = {"path": str(path), "sha256": prior.file_sha(path)}
    result = prior.verified_json(binding, check)
    if result.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"): return binding, result, {}, {}
    source = prior.verified_json(result["source_producer"], check); passed = prior.verified_json(result["source_audit"], check)
    require(source["status"] == "COMPLETE" and passed["status"] == "PASS" and passed["producer"] == result["source_producer"] and result["source_producer"]["sha256"] == "5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98" and result["source_audit"]["sha256"] == "14399051dc1614f785f4548daa139d92c8769e59fcbee9f38a98126914aedab6", "EXACT_ORIGINAL_FIT_TARGET_COMPLETE_AND_INDEPENDENT_PASS")
    require(prior.file_sha(passed["audit_source"]["path"]) == passed["audit_source"]["sha256"], "ORIGINAL_IMMUTABLE_INDEPENDENT_AUDITOR_SOURCE")
    expected = {"projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2, "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "saved_prediction_arrays_checked": 88, "output_subspace_diagnostic_cells_checked": 12, "fixed_D_real_method_inequalities_checked": 48, "bound_RAW_inequalities_checked": 0}
    require(all(passed["totals"].get(key) == value for key, value in expected.items()), "FULL_PRIOR_INDEPENDENT_AUDIT_COVERAGE")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "existing_head_weights_reused", "future_labels_used_for_subset_or_decoder_fit", "per_origin_decoder_fit", "whole_deployment_cost_measured", "oracle_used_for_selection", "strict_K_only_compression", "individual_exact_human_method_approval_claimed"): require(result[key] is False, "SOURCE_SCOPE_PAST_ONLY_STATIC_D_EXPOSED_DEV:" + key)
    require(result["FIT_TARGET_E_already_FIT_future_supervised"] is True and result["all_twelve_locks_before_any_DEV_input_or_prediction"] is True and result["hard_elapsed_or_cumulative_resource_caps"] is None and result["cost_status"].startswith("NOT_MEASURED;"), "OLD_SUPERVISED_E_AND_ALL_TWELVE_LOCKS_COST_NOT_MEASURED")
    old = prior.verified_json(result["prior_partial"], check)
    diagnostic = prior.verified_json(result["Gram_failure_diagnostic"], check)
    require(result["prior_partial"]["sha256"] == "4240cedea826e052b227ef2fba4e16ff52cdcd6db69281844485eca6c0b0bd6b" and old["status"] == "PARTIAL_COMPLETED" and old["subset_selection_fits"] == old["actual_subset_F_split_inferences"] == 0 and old["native_training_fits"] == 8 and old["training_updates"] == 4800 and old["checkpoint_states_saved"] == 40 and old["saved_native_split_predictions"] == 24 and old["saved_Bolt_split_predictions"] == 12, "PRESERVED_ZERO_SUBSET_F_V1_ARITHMETIC_FAILURE_WITH_COMPLETED_MIXED_EIGHT_HEADS")
    require(diagnostic["partial_result"] == result["prior_partial"] and diagnostic["status"] == "CONFIRMED_FIRST_STAGE_GRAM_CANCELLATION_VS_DIRECT_RESIDUAL" and diagnostic["FIT_future_targets_read"] is diagnostic["TUNE_DEV_read"] is diagnostic["F_head_forward_training"] is diagnostic["protected_TEST_accessed"] is False, "PRESERVED_FIT_ONLY_NEGATIVE_SCORE_DIAGNOSTIC")
    for dataset in DATASETS: require(old["datasets"][dataset]["subset_selection_failure"]["failure"]["message"] == "NEGATIVE_RECONSTRUCTION_MSE_BEYOND_FP64_ROUNDOFF", "V1_GRAM_GUARD_FAILURE_NOT_FORECAST_METHOD_FAILURE")
    require(Path(result["numeric_repair_addendum"]["path"]).suffix.lower() == ".md" and prior.file_sha(result["numeric_repair_addendum"]["path"]) == result["numeric_repair_addendum"]["sha256"], "FRESH_DIRECT_RESIDUAL_NUMERIC_REPAIR_ADDENDUM")
    return binding, result, source, passed


def mixed_identity(old, current, check):
    counts = {"mixed_native_conditions_compared": 0, "checkpoint_states_and_TUNE_arrays_compared": 0, "selected_native_split_arrays_compared": 0, "mixed_Bolt_split_arrays_compared": 0, "global_D_states_compared": 0, "learning_curves_compared": 0}
    def same_arrays(a, b, label):
        left, right = prior.verified_arrays(a, check), prior.verified_arrays(b, check)
        require(set(left) == set(right), "PRIOR_V1_V2_ARRAY_KEY_IDENTITY:" + label)
        for name in left: exact(left[name], right[name], "prior_v1_v2_same_logical_array:" + label + ":" + name)
    for dataset, (_, _, dev, _) in DATASETS.items():
        for space in ("PCA", "FIT_TARGET"):
            before, after = (parent["datasets"][dataset]["spaces"][space] for parent in (old, current))
            require(before["status"] == after["status"] == "COMPLETE", "REPEATED_MIXED_CONDITIONS_BOTH_COMPLETE")
            same_arrays(before["E_state_artifact"], after["E_state_artifact"], "E")
            for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                require(before["backends"][backend]["metadata"] == after["backends"][backend]["metadata"], "REPEATED_GLOBAL_D_METADATA")
                same_arrays(before["backends"][backend]["state_artifact"], after["backends"][backend]["state_artifact"], "D")
                counts["global_D_states_compared"] += 1
            for split in ("fit", "tune", dev):
                same_arrays(before["inputs"][split], after["inputs"][split], "inputs")
                for field in ("scores_all_observed", "scores_complete_mask"): require(before["Bolt"][split][field] == after["Bolt"][split][field], "REPEATED_MIXED_BOLT_SCORE_IDENTITY")
                same_arrays(before["Bolt"][split]["artifact"], after["Bolt"][split]["artifact"], "Bolt")
                counts["mixed_Bolt_split_arrays_compared"] += 1
            for seed in SEEDS:
                a, b = before["native"][str(seed)], after["native"][str(seed)]
                require({key: value for key, value in a["metadata"].items() if key != "train_wall_seconds"} == {key: value for key, value in b["metadata"].items() if key != "train_wall_seconds"} and a["learning_curve"] == b["learning_curve"] and a["scores"] == b["scores"], "REPEATED_MIXED_HEAD_NUMERIC_METADATA_CURVE_SCORE_IDENTITY")
                require(len(a["checkpoints"]) == len(b["checkpoints"]) == 5, "FIVE_PRIOR_REPEATED_MIXED_CP")
                for old_cp, new_cp in zip(a["checkpoints"], b["checkpoints"]):
                    for field in ("update", "tune", "state_sha256"): require(old_cp[field] == new_cp[field], "REPEATED_MIXED_CHECKPOINT_IDENTITY")
                    for member in ("state_artifact", "tune_prediction_artifact"): same_arrays(old_cp[member], new_cp[member], "CP:" + member)
                    counts["checkpoint_states_and_TUNE_arrays_compared"] += 1
                for split in ("fit", "tune", dev): same_arrays(a["artifacts"][split], b["artifacts"][split], "selected_native"); counts["selected_native_split_arrays_compared"] += 1
                counts["learning_curves_compared"] += 1; counts["mixed_native_conditions_compared"] += 1
    expected = {"mixed_native_conditions_compared": 8, "checkpoint_states_and_TUNE_arrays_compared": 40, "selected_native_split_arrays_compared": 24, "mixed_Bolt_split_arrays_compared": 12, "global_D_states_compared": 8, "learning_curves_compared": 8}
    require(counts == expected, "EIGHT_REEXECUTED_MIXED_CONDITIONS_NOT_NEW_INDEPENDENT_CONDITIONS")
    proof = current["mixed_reexecution_identity"]
    require(proof["status"] == "PASS" and proof["mismatches"] == [] and all(proof[key] == value for key, value in counts.items()) and proof["tolerance_used"] is proof["two_runs_independent_conditions_claimed"] is proof["new_heads_warm_started_or_reused"] is False and proof["total_physical_native_fit_attempts_across_v1_v2"] == old["native_training_fits"] + current["native_training_fits"] == 20 and proof["distinct_registered_native_conditions"] == 12, "INDEPENDENT_ACTUAL_LOGICAL_BYTE_DUPLICATE_REEXECUTION_PROOF")
    return {"status": "PASS", **counts, "total_physical_native_fit_attempts_across_v1_v2": 20, "distinct_registered_native_conditions": 12, "new_independent_conditions_from_reexecution": 0, "training_wall_identity_not_claimed": True, "actual_old_and_new_NPZ_logical_bytes_checked": True}


def audit_result(result_path, meter):
    check = meter.check
    binding, result, source, passed = load_gates(result_path, check)
    if not source: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": binding, "producer_status": result.get("status"), "scientific_assets_opened": False, "preserved_failed_paths": result.get("failed_paths")}
    complete = result["status"] == "COMPLETE"
    guarded = [binding, result["source_producer"], result["source_audit"], passed["audit_source"], result["plan"], result["prior_partial"], result["Gram_failure_diagnostic"], result["numeric_repair_addendum"], *result["protected_prior_records"], *result["input_records"], *result["source_snapshots"]]
    for record in guarded:
        check(); require(prior.file_sha(record["path"]) == record["sha256"], "PRESERVED_SOURCE_INPUT_MODEL_CACHE_AND_SELECTION_FILE_SHA")
        if "snapshot_path" in record: require(prior.file_sha(record["snapshot_path"]) == record["sha256"], "EXECUTED_SOURCE_SNAPSHOT_BYTE_IDENTITY")
    source_checks = source_contract(result["source_snapshots"])
    require(result["F_before"] == result["F_after"] == source["F_before"] == source["F_after"] and set(result["F_before"]) == {"old_algorithm", "bi_algorithm"}, "FROZEN_BOLT_DUAL_SIGNATURES")
    for name in ("model_binary", "model_config"):
        before, after, original = result[name + "_before"], result[name + "_after"], source[name + "_before"]
        require(Path(before["path"]).resolve() == Path(after["path"]).resolve() == Path(original["path"]).resolve() and before["sha256"] == after["sha256"] == original["sha256"] == prior.file_sha(before["path"]), "FROZEN_MODEL_DISK_CONFIG_BINDING")
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(result["runtime_behavior_contract"][key] == source["runtime_behavior_contract"][key], "ORIGINAL_BOLT_RUNTIME:" + key)
    registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    locks = {}
    for record in registry["conditions"]:
        key = (record["dataset"], record["space"], record["seed"])
        lock = prior.verified_json(record["lock"], check); guarded.append(record["lock"])
        require(key not in locks and (lock["dataset"], lock["space"], lock["seed"]) == key and lock["status"] in ("LOCKED", "FAILED_PATH_PRESERVED"), "UNIQUE_EXACT_RESOLVED_NATIVE_SELECTION_LOCK")
        locks[key] = (record["lock"], lock)
    registry_proof = check_registry(registry, result["execution_events"], locks)
    receipt = prior.verified_json(result["resource_terminal"], check)
    require(receipt["status"] == result["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_RAW_SUBSET_EXECUTION_RESOURCE_RECEIPT")
    totals = {"subset_selection_fits_checked": 0, "greedy_candidate_scores_checked": 0, "global_decoder_states_checked": 0, "source_encoder_states_checked": 0,
        "source_split_contexts_checked": 0, "native_fits_checked": 0, "training_updates_logged_checked": 0, "checkpoint_states_checked": 0,
        "checkpoint_TUNE_prediction_arrays_checked": 0, "selected_native_split_prediction_arrays_checked": 0, "Bolt_split_prediction_arrays_checked": 0,
        "actual_subset_F_split_provenance_checks": 0, "actual_subset_F_model_batch_calls_registered": 0, "reused_mixed_F_cache_lineages_checked": 0,
        "resolved_native_selection_locks_checked": len(locks), "cached_future_label_perturbation_checks": 0, "new_fixed_D_output_space_cells_computed": 0,
        "new_E_Y_diagnostic_predictions_computed": 0, "new_fixed_D_real_method_inequalities_checked": 0, "RAW_bound_inequalities_checked": 0,
                "current_C_persistence_split_baselines_computed": 0, "score_field_checks": 0, "protected_prior_file_hashes_checked": len(result["protected_prior_records"])}
    report = {"status": "PASS" if complete else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, "source_producer": result["source_producer"], "source_audit": result["source_audit"], "plan": result["plan"], "prior_partial": result["prior_partial"], "Gram_failure_diagnostic": result["Gram_failure_diagnostic"], "numeric_repair_addendum": result["numeric_repair_addendum"], "selection_registry": result["selection_registry"], "registry_proof": registry_proof, "source_contract": source_checks, "datasets": {}, "totals": totals, "producer_failed_paths": result["failed_paths"], "audit_failed_paths": [], "unverified_partial_conditions": [],
        "protected_TEST_accessed": False, "new_backbone_or_encoder_training_or_F_reinference": False, "new_learning_trajectories_independently_replayed": False,
        "new_output_space_diagnostics_computed_by_this_auditor": True, "new_oracle_used_for_selection_or_deployment": False,
        "new_output_minimum_only_normalized_MSE_in_each_new_static_D32_real_rowspace": True, "MAE_or_physical_MSE_minimum_claimed": False,
        "strict_universal_FP32_output_lower_bound_claimed": False, "old_output_space_floor_reused": False, "independent_generalization_claimed": False,
        "whole_deployment_cost_measured": False, "strict_K_only_compression": False}
    for dataset, (c, k, development, dev_count) in DATASETS.items():
        check(); row = result["datasets"][dataset]
        out = {"subset_selection": None, "spaces": {}, "raw_scores": {}, "current_C_persistence_scores": {}}; report["datasets"][dataset] = out
        if row.get("dataset_input_failure") is not None:
            require(not complete, "COMPLETE_PRODUCER_HAS_FAILED_DATASET_INPUT")
            report["unverified_partial_conditions"].append(row["dataset_input_failure"]); continue
        require(row["channels"] == c and row["K"] == k and row["development_split"] == development and row["raw_input_sources"] == source["datasets"][dataset]["inputs"], "ORIGINAL_RAW_SUBSET_DATA_CONTRACT")
        values = {}
        for split, windows in (("fit", 256), ("tune", 64), (development, dev_count)):
            if split not in row["raw_scores"]:
                require(not complete, "COMPLETE_PRODUCER_MISSING_RAW_SPLIT"); continue
            value = prior.verified_arrays(row["raw_input_sources"][split], check)
            prior.check_raw_split(value, channels=c, windows=windows); values[split] = value
            for name in ("mean", "std"): exact(value[name], values["fit"][name], "same_FIT_only_scaler")
            out["raw_scores"][split], count = score_checked(row["raw_scores"][split], value["raw_forecast"], value, "RAW_" + dataset + ":" + split); totals["score_field_checks"] += count
            persistence = np.broadcast_to(value["raw_last"][:, None], value["target"].shape)
            out["current_C_persistence_scores"][split] = {"scores_all_observed": prior.scores(persistence, value["target"], value["observed"], value["std"]), "scores_complete_mask": prior.scores(persistence, value["target"], prior.oracle_mask(value["observed"]), value["std"]) if prior.oracle_mask(value["observed"]).any() else None, "whole_deployment_cost_measured": False, "new_model_or_checkpoint_selection": False}
            totals["current_C_persistence_split_baselines_computed"] += 1
        selected_indices = None
        if "subset_selection_lock" in row:
            selection = prior.verified_json(row["subset_selection_lock"], check); guarded.append(row["subset_selection_lock"])
            require(selection["FIT_source"] == row["raw_input_sources"]["fit"], "RAW_SUBSET_ORIGINAL_FIT_SOURCE")
            bridge_record = row["canonical_physical_bridge_binding"]["canonical_input_bridge"]
            bridge = prior.verified_json(bridge_record, check)
            fit_records = [record for record in bridge["artifacts"] if record["dataset"] == dataset and record["split"] == "fit"]
            require(len(fit_records) == 1, "ONE_EXACT_PHYSICAL_FIT_CHANNEL_NAME_SOURCE")
            physical_record = dict(fit_records[0]); physical_path = Path(physical_record["path"])
            physical_record["path"] = str(physical_path.resolve() if physical_path.is_absolute() else (ROOT / physical_path).resolve())
            physical_fit = prior.verified_arrays(physical_record, check)
            exact(physical_fit["history"], values["fit"]["history"], "channel_names_same_FIT_history")
            exact(physical_fit["history_observed"], values["fit"]["history_observed"], "channel_names_same_observed_FIT_mask")
            out["subset_selection"] = check_subset_trace(selection, values["fit"]["history"], values["fit"]["history_observed"], k, physical_fit["columns"].tolist())
            selected_indices = selection["selected_indices"]
            totals["subset_selection_fits_checked"] += 1; totals["greedy_candidate_scores_checked"] += out["subset_selection"]["candidate_scores_checked"]
        for space, original_name in SPACES.items():
            condition = row["spaces"][space]
            if condition["status"] != "COMPLETE":
                require(not complete, "COMPLETE_PRODUCER_HAS_PARTIAL_SPACE")
                report["unverified_partial_conditions"].append({"dataset": dataset, "space": space, "metadata": condition}); continue
            try:
                e = prior.verified_arrays(condition["E_state_artifact"], check)
                require(set(e) == {"center", "encoder", "selected_indices"} and e["center"].shape == (c,) and e["encoder"].shape == (c, k) and e["selected_indices"].dtype == np.int64, "SIX_FIXED_ENCODER_ARRAY_STATES")
                if original_name is None:
                    require(selected_indices is not None and condition["subset_selection_lock"] == row["subset_selection_lock"], "FIT_SUBSET_LOCK_MATCHES_DEPLOYED_SPACE")
                    exact(e["selected_indices"], np.asarray(selected_indices, np.int64), "ascending_raw_subset_indices")
                    matrix = np.zeros((c, k), np.float32); matrix[selected_indices, np.arange(k)] = 1.
                    exact(e["encoder"], matrix, "raw_subset_onehot_encoder32")
                    exact(e["center"], np.zeros(c, np.float32), "raw_subset_zero_center32")
                    old_d, old_b = np.zeros((k, c), np.float32), np.zeros(c, np.float32)
                else:
                    original_condition = source["datasets"][dataset]["projections"][original_name]
                    require(condition["original_E_state"] == original_condition["state_artifact"] and e["selected_indices"].size == 0, "ORIGINAL_FIXED_MIXED_ENCODER_SOURCE")
                    old = prior.verified_arrays(condition["original_E_state"], check)
                    for name in ("center", "encoder"): exact(e[name], old[name], "mixed_E_original_exact_state")
                    old_d, old_b = old["decoder"].astype(np.float32), old["decoder_bias"].astype(np.float32)
                e_hashes = {"numpy_center": raw_binding(e["center"])["sha256"], "numpy_encoder": raw_binding(e["encoder"])["sha256"], "native_projection_state": prior.torch_state_sha({"center": e["center"].astype(np.float32), "encoder": e["encoder"].astype(np.float32), "decoder": old_d, "decoder_bias": old_b})}
                require(condition["fixed_E_hashes_before"] == condition["fixed_E_hashes_after"] == e_hashes, "INDEPENDENT_NUMPY_AND_NATIVE_FIXED_E_BEFORE_AFTER_HASHES")
                totals["source_encoder_states_checked"] += 1
                contexts, native_contexts = {}, {}
                for split, value in values.items():
                    archived = prior.verified_arrays(condition["inputs"][split], check)
                    require(set(archived) == {"history", "history_observed", "latent_history_BOLT_NUMPY", "latent_history_NATIVE_TORCH32", "target", "observed", "raw_last", "origins", "mean", "std", "raw_forecast"}, "EXACT_PAST_LATENT_SOURCE_AND_SCORING_ARRAYS")
                    for name in value:
                        if name in archived: exact(archived[name], value[name], "new_space_same_original_raw_input:" + name)
                    z_numpy, z_native = (encode(value["history"], e, backend) for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"))
                    exact(archived["latent_history_BOLT_NUMPY"], z_numpy, "new_space_actual_NumPy_past_encoding")
                    exact(archived["latent_history_NATIVE_TORCH32"], z_native, "new_space_actual_Torch32_past_encoding")
                    contexts[split] = {**value, "latent_history_BOLT_NUMPY": z_numpy, "latent_history_NATIVE_TORCH32": z_native}
                    native_contexts[split] = {**value, "latent_history": z_native}
                    totals["source_split_contexts_checked"] += 1
                output = {"encoder_state": condition["E_state_artifact"], "backends": {}, "Bolt": {}, "native": {}, "new_output_space_diagnostics": {}}; out["spaces"][space] = output
                decoders = {}
                require(set(condition["backends"]) == {"BOLT_NUMPY", "NATIVE_TORCH32"}, "BOTH_ACTUAL_NUMERIC_FIT_DECODER_BACKENDS")
                for backend, entry in condition["backends"].items():
                    state = prior.verified_arrays(entry["state_artifact"], check); decoders[backend] = state["decoder_FP32"]
                    output["backends"][backend] = {"state": entry["state_artifact"], "independent_fit": check_global_state(entry["metadata"], state, values["fit"]["history"], values["fit"]["history_observed"], contexts["fit"]["latent_history_" + backend]), "deployed_decoder_FP32_values": state["decoder_FP32"].tolist(), "deployed_decoder_max_abs": float(np.abs(state["decoder_FP32"]).max()), "deployed_decoder_operator_norm": float(np.linalg.norm(state["decoder_FP32"].astype(np.float64), 2)), "deployed_decoder_Frobenius_norm": float(np.linalg.norm(state["decoder_FP32"].astype(np.float64)))}
                    totals["global_decoder_states_checked"] += 1
                bolt_arrays = {}
                for split, context in contexts.items():
                    entry = condition["Bolt"][split]; saved = prior.verified_arrays(entry["artifact"], check)
                    require(set(saved) == {"prediction", "latent_history", "latent_future", "origins"}, "ACTUAL_BOLT_FUTURE_AND_PREDICTION_ARRAY_KEYS")
                    exact(saved["latent_history"], context["latent_history_BOLT_NUMPY"], "actual_Bolt_same_encoded_past")
                    exact(saved["origins"], context["origins"], "actual_Bolt_same_origins")
                    prediction = fixed_forecast(context["raw_last"], saved["latent_future"], saved["latent_history"][:, -1], decoders["BOLT_NUMPY"], "BOLT_NUMPY")
                    exact(saved["prediction"], prediction, "independent_Bolt_cached_future_new_fixed_D_anchor")
                    output["Bolt"][split], count = score_checked(entry, prediction, context, "new_Bolt:" + space + ":" + split); totals["score_field_checks"] += count
                    if original_name is None:
                        require(entry["mixed_F_cache_reused"] is entry["full_C_forecast_slice_used_as_new_inference"] is False and entry["new_K_F_inference_executed"] is True and entry["full_C_slice_diagnostic"]["diagnostic_only"] is True, "SOURCE_INVOKED_NEW_K_F_NOT_FULL_C_SLICE")
                        sliced = context["raw_forecast"][..., selected_indices]
                        diff = saved["latent_future"].astype(np.float64) - sliced.astype(np.float64)
                        require(entry["full_C_slice_diagnostic"]["byte_equal"] == (saved["latent_future"].tobytes() == np.ascontiguousarray(sliced).tobytes()), "NEW_SUBSET_AND_FULL_C_SLICE_BYTE_DIAGNOSTIC")
                        prior.close(entry["full_C_slice_diagnostic"]["max_abs"], float(np.max(np.abs(saved["latent_future"] - sliced))), "new_subset_F_slice_maxabs", atol=1e-10, rtol=0)
                        prior.close(entry["full_C_slice_diagnostic"]["RMSE"], float(np.sqrt(np.mean(diff * diff))), "new_subset_F_slice_RMSE", atol=1e-10, rtol=0)
                        totals["actual_subset_F_split_provenance_checks"] += 1; totals["actual_subset_F_model_batch_calls_registered"] += (len(context["history"]) + 7) // 8
                    else:
                        original_cache = source["datasets"][dataset]["projections"][original_name]["bolt"]["artifacts"][split]
                        require(entry["mixed_F_cache_reused"] is True and entry["original_F_cache"] == original_cache, "SAME_ORIGINAL_MIXED_F_CACHE_RECORD")
                        cache = prior.verified_arrays(original_cache, check)
                        for name in ("latent_history", "latent_future", "origins"): exact(saved[name], cache[name], "original_mixed_F_cache_lineage:" + name)
                        require(entry["original_previous_D_prediction_SHA"] == raw_binding(cache["prediction"])["sha256"], "PREVIOUS_D_FORECAST_RETAINED_AS_SEPARATE_PROVENANCE")
                        for seed in SEEDS:
                            old_native = prior.verified_arrays(source["datasets"][dataset]["projections"][original_name]["native"][str(seed)]["artifacts"][split], check)
                            exact(old_native["latent_history"], context["latent_history_NATIVE_TORCH32"], "two_original_native_seed_past_cache_lineages")
                            exact(old_native["origins"], context["origins"], "two_original_native_seed_cache_origins")
                        totals["reused_mixed_F_cache_lineages_checked"] += 1
                    bolt_arrays[split] = saved; totals["Bolt_split_prediction_arrays_checked"] += 1
                native_arrays = {}
                for seed in SEEDS:
                    lock = locks[(dataset, space, str(seed))][1]
                    require(lock["status"] == "LOCKED", "COMPLETE_SPACE_WITH_LOCKED_NATIVE_HEAD")
                    details, arrays = check_native(condition["native"][str(seed)], lock, native_contexts, decoders["NATIVE_TORCH32"], seed, k, c, development, check)
                    output["native"][str(seed)], native_arrays[str(seed)] = details, arrays
                    for key in ("native_fits_checked", "training_updates_logged_checked", "checkpoint_states_checked", "checkpoint_TUNE_prediction_arrays_checked", "selected_native_split_prediction_arrays_checked", "cached_future_label_perturbation_checks", "score_field_checks"): totals[key] += details[key]
                for split, context in contexts.items():
                    split_diagnostic = {}; output["new_output_space_diagnostics"][split] = split_diagnostic
                    for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                        d = decoders[backend]; diagnostic = new_fixed_space(context, d)
                        diagnostic["decoder_state"] = condition["backends"][backend]["state_artifact"]
                        diagnostic["actual_methods"] = {}; split_diagnostic[backend] = diagnostic
                        if diagnostic["status"] == "NO_COMPLETE_C_TARGET_POSITIONS": continue
                        totals["new_fixed_D_output_space_cells_computed"] += 1
                        complete = context["observed"].all(axis=2, keepdims=True)
                        target_only_complete = np.where(complete, context["target"], context["raw_last"][:, None]).astype(np.float32)
                        future = encode(target_only_complete, e, backend); last = context["latent_history_" + backend][:, -1]
                        ey = fixed_forecast(context["raw_last"], future, last, d, backend)
                        diagnostic["E_Y"] = diagnostic_trace(ey, context["raw_last"], future, last, d, context, diagnostic, "E_Y")
                        diagnostic["E_Y"]["role"] = "FUTURE_KNOWN_COMPLETE_C_ONLY_DIAGNOSTIC_NOT_DEPLOYABLE_OR_SELECTION"
                        totals["new_E_Y_diagnostic_predictions_computed"] += 1; totals["new_fixed_D_real_method_inequalities_checked"] += 1
                        methods = {"BOLT": bolt_arrays[split]} if backend == "BOLT_NUMPY" else {"NATIVE_s" + seed: native_arrays[seed][split] for seed in native_arrays}
                        for name, saved in methods.items():
                            diagnostic["actual_methods"][name] = diagnostic_trace(saved["prediction"], context["raw_last"], saved["latent_future"], last, d, context, diagnostic, name)
                            totals["new_fixed_D_real_method_inequalities_checked"] += 1
            except Exception as error:
                report["audit_failed_paths"].append({"dataset": dataset, "space": space, "failure": {"type": type(error).__name__, "message": str(error)}})
    for record in guarded: check(); require(prior.file_sha(record["path"]) == record["sha256"], "POST_AUDIT_ORIGINAL_FILES_UNCHANGED")
    expected_counts = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83, "global_decoder_states_checked": 12,
        "source_encoder_states_checked": 6, "source_split_contexts_checked": 18, "native_fits_checked": 12, "training_updates_logged_checked": 7200,
        "checkpoint_states_checked": 60, "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36,
        "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6, "actual_subset_F_model_batch_calls_registered": 103,
        "reused_mixed_F_cache_lineages_checked": 12, "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12,
        "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36, "new_fixed_D_real_method_inequalities_checked": 90, "RAW_bound_inequalities_checked": 0, "current_C_persistence_split_baselines_computed": 6}
    if complete and not report["audit_failed_paths"]:
        report["mixed_reexecution_identity"] = mixed_identity(prior.verified_json(result["prior_partial"], check), result, check)
        require(all(totals[key] == number for key, number in expected_counts.items()) and result["actual_subset_F_model_batch_calls"] == 103 and result["completeness"]["actual"] == result["completeness"]["expected"] and not result["failed_paths"], "FULL_NEW_SUBSET_NATIVE_FIT_CACHE_AND_DIAGNOSTIC_COUNTS")
    elif report["audit_failed_paths"]: report["status"] = "FAILED_PARTIAL_AUDIT_PRESERVED_NO_FULL_PASS"
    return report


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_RAW_SUBSET_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_raw_subset_" + output.stem, gpu=False)
    code = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}; failure = None
    try:
        meter.protected_check(); _, value, source, _ = load_gates(result_path, meter.check)
        if not source: report = {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "producer_status": value.get("status"), "scientific_assets_opened": False}
        else:
            initialize_numpy(); sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
            from prepare_inputs import install_protected_test_guard
            install_protected_test_guard()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2): report = audit_result(result_path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == code["sha256"], "RAW_SUBSET_INDEPENDENT_AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = code; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure, "audit_failed_paths": report.get("audit_failed_paths")}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
