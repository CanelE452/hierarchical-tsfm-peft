"""Optional post-C output-space diagnostic under INPUT_PLAN section 10.5.

Primary-fold B2/B4 small-K seeds/checkpoints are chosen by locked B MSE only.
Only when that route's C MSE exceeds canonical Raw on the same C cohort do we
solve for arbitrary future latent values through its fixed origin-reference D.
The C comparison triggers a diagnostic, never model tuning or a deployment rule.

The joint real map adapts ONLY the pure indexing idea of
research/bolt_compatible_adaptive_20261014/projection_followup_v1/
causal_lag_decoder_v1.py:joint_real_map. No legacy loader/controller/oracle is
imported. This solver uses every original observed element and GLOBAL per-channel
C counts, not the old complete-channel-position diagnostic. The SVD cutoff is
fixed at 1e-12 in FP64, with rank/conditioning/residual certificates preserved.

Free future latent values use actual Y for diagnosis only. They are not E(Y),
an inferred forecast, a learned objective or a proposed method. A low diagnostic
error does not show that past P/E contains predictive information or that F can
learn/reach these latents. A large diagnostic error concerns this fixed D and its
retained numeric space, not every temporal decoder or every small-K method.
FP32 replay is a CPU decoder trace, not a claim of GPU bitwise equivalence.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

import data as dio
from models import TemporalFront
import runtime as rt
from analysis_outputs import (_array_digest, _cache_path, _digest, _read,
                              masked_metrics, verify_gates)


RCOND = 1e-12
HORIZON = 48


def joint_real_map(history, current, decoder, lags=rt.LAGS, horizon=HORIZON):
    """One [H*C,H*K] map per fixed D; all origin-specific known terms in offset."""
    z = np.asarray(history, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    decoder = np.asarray(decoder, dtype=np.float64)
    lags = tuple(int(value) for value in lags)
    if z.ndim != 3 or current.ndim != 2 or decoder.ndim != 2 or horizon < 1:
        raise ValueError("Expected finite latent [N,L,K], current [N,C], decoder [lags*K,C]")
    if not lags or tuple(sorted(set(lags))) != lags or min(lags) < 0 or max(lags) >= z.shape[1]:
        raise ValueError("Lags must be ascending distinct nonnegative known-past positions")
    batch, length, rank = z.shape
    channels = decoder.shape[1]
    if current.shape != (batch, channels) or decoder.shape[0] != len(lags) * rank:
        raise ValueError("Current/latent/decoder dimensions differ")
    if not all(np.isfinite(value).all() for value in (z, current, decoder)):
        raise ValueError("Nonfinite fixed decoder inputs")
    design = np.zeros((horizon * channels, horizon * rank), dtype=np.float64)
    offset = np.broadcast_to(current[:, None, :], (batch, horizon, channels)).copy()
    for index, lag in enumerate(lags):
        block = decoder[index * rank : (index + 1) * rank]
        offset -= (z[:, -1 - lag] @ block)[:, None, :]
        for h in range(horizon):
            source = h - lag
            if source >= 0:
                design[h * channels : (h + 1) * channels, source * rank : (source + 1) * rank] += block.T
            else:
                offset[:, h, :] += z[:, length + source, :] @ block
    return design, offset


def weighted_minimum(history, current, decoder, target, observed, lags=rt.LAGS):
    """Independent free-future vectors, coupled horizons; global macro-MSE weights."""
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed)
    if target.ndim != 3 or observed.dtype != np.bool_ or observed.shape != target.shape:
        raise ValueError("Target/mask must be [N,H,C] with original bool observation mask")
    if not np.isfinite(target[observed]).all():
        raise ValueError("Nonfinite true observed target")
    design, offset = joint_real_map(history, current, decoder, lags, target.shape[1])
    if offset.shape != target.shape:
        raise ValueError("Target and decoded shape differ")
    batch, horizon, channels = target.shape
    rank = np.asarray(history).shape[-1]
    counts = observed.sum(axis=(0, 1))
    active = counts > 0
    if not active.any():
        raise ValueError("No observed targets for an output-space diagnostic")
    channel_weight = np.zeros(channels, dtype=np.float64)
    channel_weight[active] = 1.0 / np.sqrt(int(active.sum()) * counts[active])
    output_weight = np.tile(channel_weight, horizon)
    flattened_mask = observed.reshape(batch, -1)
    rhs = target.reshape(batch, -1) - offset.reshape(batch, -1)
    patterns = {}
    for origin in range(batch):
        key = np.packbits(flattened_mask[origin]).tobytes()
        patterns.setdefault(key, []).append(origin)
    optimal = np.zeros((batch, horizon * rank), dtype=np.float64)
    certificates = []
    weighted_residual_sse = 0.0
    for indices in patterns.values():
        selected = flattened_mask[indices[0]]
        if not selected.any():
            certificates.append({"origin_indices": indices, "selected_output_rows": 0,
                                 "numeric_rank": 0, "singular_values": [], "svd_executed": False})
            continue
        weighted_design = design[selected] * output_weight[selected, None]
        weighted_rhs = rhs[np.ix_(indices, selected)] * output_weight[None, selected]
        left, singular, right = np.linalg.svd(weighted_design, full_matrices=False)
        keep = singular > singular[0] * RCOND if len(singular) and singular[0] > 0 else np.zeros(len(singular), dtype=bool)
        retained_rank = int(keep.sum())
        inverse = (right[keep].T / singular[keep]) @ left[:, keep].T if retained_rank else np.zeros((horizon * rank, int(selected.sum())))
        fitted = weighted_rhs @ inverse.T
        if not np.isfinite(fitted).all():
            raise FloatingPointError("Nonfinite FP64 oracle latent solution")
        optimal[indices] = fitted
        residual = fitted @ weighted_design.T - weighted_rhs
        weighted_residual_sse += float(np.square(residual).sum(dtype=np.float64))
        certificates.append({
            "origin_indices": indices, "selected_output_rows": int(selected.sum()), "unknown_coordinates_kept": horizon * rank,
            "numeric_rank": retained_rank, "svd_executed": True, "singular_values": [float(value) for value in singular],
            "retained_condition_number": float(singular[0] / singular[keep][-1]) if retained_rank else None,
            "smallest_retained_singular_value": float(singular[keep][-1]) if retained_rank else None,
            "cutoff_absolute": float(singular[0] * RCOND) if len(singular) else 0.0,
            "weighted_residual_sse": float(np.square(residual).sum(dtype=np.float64)),
            "weighted_residual_max_abs": float(np.max(np.abs(residual))),
            "normal_equation_max_abs": float(np.max(np.abs(residual @ weighted_design))),
            "retained_left_space_residual_max_abs": float(np.max(np.abs(residual @ left[:, keep]))) if retained_rank else 0.0,
        })
    prediction = (optimal @ design.T).reshape(batch, horizon, channels) + offset
    metric = masked_metrics(prediction, target, observed)
    objective_difference = abs(metric["mse"] - weighted_residual_sse)
    objective_tolerance = max(1e-10, metric["mse"] * 1e-9)
    objective_parity = bool(objective_difference <= objective_tolerance)
    zero_prediction = offset
    persistence = np.repeat(np.asarray(history, np.float64)[:, -1:, :], horizon, axis=1)
    persistence_prediction = (persistence.reshape(batch, -1) @ design.T).reshape(target.shape) + offset
    zero_mse = masked_metrics(zero_prediction, target, observed)["mse"]
    persistence_mse = masked_metrics(persistence_prediction, target, observed)["mse"]
    tolerance = max(1e-10, min(zero_mse, persistence_mse) * 1e-9)
    inequality = bool(metric["mse"] <= min(zero_mse, persistence_mse) + tolerance)
    return {
        "optimal_future": optimal.reshape(batch, horizon, rank), "prediction": prediction,
        "design": design, "offset": offset,
        "metadata": {
            "mse": metric["mse"], "mae_not_minimized": metric["mae"], "channel_mse": metric["channel_mse"],
            "n_observed": metric["n_observed"], "global_C_observed_counts": [int(value) for value in counts],
            "active_channels": int(active.sum()), "channel_sqrt_weights": [float(value) for value in channel_weight],
            "objective": "sum over observed origin,horizon,channel of squared normalized error / (active_C * global_C_observed_count[channel])",
            "rcond": RCOND, "mask_patterns": len(patterns), "SVD_count": sum(certificate["svd_executed"] for certificate in certificates),
            "A_shape": list(design.shape), "unknown_order": "horizon-major then latent K", "output_order": "horizon-major then original C",
            "known_past_and_origin_reference_terms_in_offset": True, "no_unknown_deleted_at_missing_output": True,
            "weighted_residual_sse": weighted_residual_sse,
            "weighted_sse_vs_macro_mse_absolute_difference": objective_difference,
            "weighted_sse_vs_macro_mse_tolerance": objective_tolerance,
            "weighted_sse_vs_macro_mse_parity": objective_parity,
            "certificates": certificates, "zero_free_latent_feasible_mse": zero_mse,
            "persistence_free_latent_feasible_mse": persistence_mse,
            "minimum_inequality_vs_feasible_controls": inequality, "inequality_tolerance": tolerance,
            "numeric_status": "COMPLETE_VALID" if inequality and objective_parity else "UNDERTRAINED_OR_UNRESOLVED",
            "minimum_scope": "retained weighted FP64 numeric singular space at fixed rcond1e-12; not an exact untruncated real or all-FP32-output lower bound",
        },
    }


def _cpu_adapter(spec, channels):
    state = torch.load(_cache_path(spec["checkpoint"]), map_location="cpu", weights_only=True)
    front = TemporalFront(spec["kind"], channels=channels, **spec.get("front_kwargs", {}))
    model = rt.Adapter(front, state["center"].numpy(), state["E"].numpy()).cpu().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return model


def _diagnose(spec, score_row, raw_row, locked_statistics):
    data = dio.load_dataset(spec["dataset"])
    fold = dio.make_fold(data, fold=spec["fold"])
    mean, std, counts = dio.fit_statistics(data, fold["bounds"]["A"])
    np.testing.assert_array_equal(mean, np.asarray(locked_statistics["A_mean"], np.float32))
    np.testing.assert_array_equal(std, np.asarray(locked_statistics["A_std"], np.float32))
    np.testing.assert_array_equal(counts, np.asarray(locked_statistics["A_true_observed_counts"], np.int64))
    archive_path = _cache_path(f"predictions/{spec['id']}.npz")
    with np.load(archive_path, allow_pickle=False) as archive:
        actual_prediction, target, observed, origins = (archive[f"C_{key}"] for key in ("pred", "target", "mask", "origins"))
    np.testing.assert_array_equal(origins, fold["origins"]["C"])
    assert _array_digest(origins) == score_row["cohort"] == raw_row["cohort"], "ORACLE_COHORT_MISMATCH"
    true_y, true_mask = zip(*(dio.target_window(data, int(origin), mean, std) for origin in origins))
    np.testing.assert_array_equal(observed, np.stack(true_mask))
    np.testing.assert_allclose(target, np.stack(true_y), atol=1e-6, rtol=2e-6)
    actual_metric = masked_metrics(actual_prediction, target, observed)
    np.testing.assert_allclose([actual_metric["mse"], actual_metric["mae"]], [float(score_row["mse"]), float(score_row["mae"])], atol=1e-10, rtol=1e-9)
    x = np.stack([dio.history_window(data, int(origin), mean, std)[0] for origin in origins])
    model = _cpu_adapter(spec, x.shape[-1])
    latent = []
    with torch.no_grad():
        for start in range(0, len(x), 16):
            latent.append(model.encode(torch.from_numpy(x[start : start + 16])).numpy())
    z = np.concatenate(latent)
    fitted = weighted_minimum(z, x[:, -1, :], model.D.numpy(), target, observed)
    free = fitted["optimal_future"]
    with np.errstate(over="ignore", invalid="ignore"):
        cast = free.astype(np.float32)
    if not np.isfinite(cast).all():
        replay = {"status": "NUMERIC_UNRESOLVED_FP32_CAST_OVERFLOW", "mse": None}
    else:
        with torch.no_grad():
            replay_prediction = model.decode(torch.from_numpy(x), torch.from_numpy(z), torch.from_numpy(cast)).numpy()
        if not np.isfinite(replay_prediction[observed]).all():
            replay = {"status": "NUMERIC_UNRESOLVED_FP32_DECODE_NONFINITE", "mse": None}
        else:
            replay_metric = masked_metrics(replay_prediction, target, observed)
            cast_real = (cast.astype(np.float64).reshape(len(x), -1) @ fitted["design"].T).reshape(target.shape) + fitted["offset"]
            cast_real_metric = masked_metrics(cast_real, target, observed)
            numerical_difference = masked_metrics(replay_prediction, fitted["prediction"], observed)
            replay = {
                "status": "COMPLETE_VALID", "mse": replay_metric["mse"], "mae_not_minimized": replay_metric["mae"],
                "FP32_latent_cast_FP64_real_map_mse": cast_real_metric["mse"],
                "FP32_decode_minus_FP64_real_mse": replay_metric["mse"] - fitted["metadata"]["mse"],
                "FP32_decode_vs_FP64_real_prediction_mse": numerical_difference["mse"],
                "FP32_decode_vs_FP64_real_prediction_mae": numerical_difference["mae"],
                "FP32_decode_vs_FP64_real_max_abs_observed": float(np.abs(replay_prediction.astype(np.float64) - fitted["prediction"])[observed].max()),
                "device": "cpu", "bitwise_GPU_equivalence_claimed": False,
            }
    return {
        "id": spec["id"], "dataset": spec["dataset"], "fold": spec["fold"], "family": spec["kind"], "r": int(spec["r"]),
        "seed": int(spec["seed"]), "selected_update": int(spec["selected_update"]), "training_status": spec.get("training_status"),
        "status": fitted["metadata"]["numeric_status"] if replay["status"] == "COMPLETE_VALID" else "UNDERTRAINED_OR_UNRESOLVED",
        "role": "POSTHOC_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT",
        "trigger": {"C_mse": actual_metric["mse"], "canonical_Raw_C_mse": float(raw_row["mse"]),
                    "C_mse_difference": actual_metric["mse"] - float(raw_row["mse"]), "cohort": score_row["cohort"]},
        "fixed_D_FP64_real_oracle": fitted["metadata"], "FP32_decoder_replay": replay,
        "oracle_vs_actual_forecast_mse_difference": fitted["metadata"]["mse"] - actual_metric["mse"],
        "free_latent_max_abs": float(np.abs(free).max()), "free_latent_l2_norm": float(np.linalg.norm(free)),
        "C_origins": [int(value) for value in origins], "checkpoint_sha256": _digest(_cache_path(spec["checkpoint"])),
        "prediction_cache_sha256": _digest(archive_path), "F_loaded_or_called": False,
        "future_Y_used_only_in_diagnostic_solver": True, "E_of_Y_used": False,
        "interpretation": "fixed selected-D retained numeric output-space diagnosis; neither predictability/learnability from Ppast nor impossibility of other temporal small-K methods follows",
    }


def run():
    start = time.perf_counter()
    selection, complete = verify_gates()
    output = rt.HERE / "ORACLE_DIAGNOSTIC.json"
    if output.exists():
        raise FileExistsError(output)
    with (rt.HERE / "SCORES.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == complete["rows"] and all(row["status"] == "COMPLETE_VALID" for row in rows)
    C = {row["id"]: row for row in rows if row["split"] == "C"}
    statistics = _read(rt.HERE / "FOLD_STATISTICS.json")
    results = []
    datasets = sorted({spec["dataset"] for spec in selection["routes"] if spec["fold"] == "primary"})
    for dataset in datasets:
        raw_id = f"primary_{dataset}_RAW"
        assert raw_id in C, "CANONICAL_RAW_MISSING:" + dataset
        raw = C[raw_id]
        rank = dio.ASSETS[dataset]["small_k"]
        for family in ("B2", "B4"):
            pool = [spec for spec in selection["routes"] if spec["type"] == "ADAPTER" and spec["dataset"] == dataset
                    and spec["fold"] == "primary" and spec["kind"] == family and spec["r"] == rank
                    and not spec.get("front_kwargs") and spec.get("block", "R2") in ("R2", "R5_STRESS")]
            if not pool:
                results.append({"dataset": dataset, "family": family, "status": "NOT_NEEDED", "reason": "NO_LOCKED_CANDIDATE_IN_THIS_FAMILY_SCOPE"})
                continue
            selected = min(pool, key=lambda spec: (spec["selected_B"]["mse"], spec["id"]))
            row = C[selected["id"]]
            if row["cohort"] != raw["cohort"]:
                results.append({"id": selected["id"], "dataset": dataset, "family": family, "status": "BLOCKED", "reason": "CANONICAL_RAW_COHORT_DIFFERS"})
            elif float(row["mse"]) <= float(raw["mse"]):
                results.append({"id": selected["id"], "dataset": dataset, "family": family, "status": "NOT_NEEDED",
                                "reason": "C_MSE_NOT_GREATER_THAN_CANONICAL_RAW", "selected_B_mse": selected["selected_B"]["mse"],
                                "C_mse": float(row["mse"]), "canonical_Raw_C_mse": float(raw["mse"]), "cohort": row["cohort"]})
            else:
                try:
                    result = _diagnose(selected, row, raw, statistics[dataset]["primary"])
                    result["selected_B_mse"] = selected["selected_B"]["mse"]
                    results.append(result)
                except Exception as error:
                    results.append({"id": selected["id"], "dataset": dataset, "family": family, "status": "INVALID",
                                    "error_type": type(error).__name__, "error": str(error),
                                    "interpretation": "this diagnostic path failed; other predeclared diagnostics continue"})
    receipt = {
        "status": "COMPLETE_VALID" if all(result["status"] in ("COMPLETE_VALID", "NOT_NEEDED") for result in results) else "INCOMPLETE_OR_NUMERIC_UNRESOLVED",
        "role": "optional post-C output-space diagnostic, INPUT_PLAN section10.5; no new method, training, retuning or model selection",
        "criterion": "primary fold, dataset/family B2 or B4 small-K candidate minimized locked B MSE, tie broken lexicographically by id; diagnose only if same-cohort C MSE > canonical primary_dataset_RAW C MSE",
        "trigger_is_significance_test": False, "trigger_margin": 0.0, "rcond": RCOND,
        "device": "cpu", "F_model_loads": 0, "F_forward_calls": 0, "wall_seconds": time.perf_counter() - start,
        "results": results, "source_sha256": _digest(Path(__file__)),
        "analysis_gates_source_sha256": _digest(rt.HERE / "analysis_outputs.py"),
        "scores_sha256": _digest(rt.HERE / "SCORES.csv"), "score_complete_sha256": _digest(rt.HERE / "SCORE_COMPLETE.json"),
        "selection_lock_sha256": _digest(rt.HERE / "SELECTION_LOCK.json"), "cost_lock_sha256": _digest(rt.HERE / "COST_LOCK.json"),
        "strict_FP32_output_lower_bound_claimed": False, "strict_untruncated_real_minimum_claimed": False,
        "past_representation_sufficiency_or_learnability_claimed": False, "all_smallK_methods_impossible_claimed": False,
        "historically_unseen": False,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "wall_seconds", "F_model_loads", "F_forward_calls")}))
    return receipt


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run()
