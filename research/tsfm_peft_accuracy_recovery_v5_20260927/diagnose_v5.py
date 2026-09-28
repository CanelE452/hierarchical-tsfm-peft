"""Bounded CPU diagnosis of saved Hog forecasts; no model construction or inference."""
import os

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "4"

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from runtime_v5 import CACHE, HERE, ROOT, Job, digest, save_json


V4 = ROOT / "research/tsfm_peft_fresh_group_v4_20260927"
CONTEXT, HORIZON, CHANNELS = 512, 48, 32
SEEDS = (92601, 92602)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def checked_path(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if digest(path) != record["sha256"]:
        raise RuntimeError(f"Source hash mismatch: {path}")
    return path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return receipt(path)


def summaries(prediction, target, mask, require_every_channel=True):
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction/target/mask shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    count = mask.sum(axis=(0, 1))
    valid = count > 0
    if require_every_channel and not valid.all():
        raise ValueError("Primary metric has a channel without observed targets")
    squared = np.where(mask, error * error, 0.0).sum(axis=(0, 1))
    absolute = np.where(mask, np.abs(error), 0.0).sum(axis=(0, 1))
    signed = np.where(mask, error, 0.0).sum(axis=(0, 1))
    channel_mse = np.divide(squared, count, out=np.full(CHANNELS, np.nan), where=valid)
    channel_mae = np.divide(absolute, count, out=np.full(CHANNELS, np.nan), where=valid)
    channel_bias = np.divide(signed, count, out=np.full(CHANNELS, np.nan), where=valid)
    return {"mse": float(channel_mse[valid].mean()) if valid.any() else None,
            "mae": float(channel_mae[valid].mean()) if valid.any() else None,
            "signed_bias": float(channel_bias[valid].mean()) if valid.any() else None,
            "observed_forecast_target_terms": int(count.sum()),
            "channels_with_observations": int(valid.sum()),
            "channel_mse": [float(v) if ok else None for v, ok in zip(channel_mse, valid)],
            "channel_mae": [float(v) if ok else None for v, ok in zip(channel_mae, valid)],
            "channel_signed_bias": [float(v) if ok else None for v, ok in zip(channel_bias, valid)],
            "channel_target_count": count.tolist(),
            "channel_squared_error_sum": squared.tolist(),
            "channel_absolute_error_sum": absolute.tolist()}


def relative_change(left, right):
    return None if right == 0 else float(100 * (left - right) / right)


def concentration(values, top_counts):
    positive = np.sort(np.maximum(np.asarray(values, dtype=np.float64).ravel(), 0.0))[::-1]
    total = float(positive.sum())
    return {"positive_excess_sum": total, "positive_count": int(np.count_nonzero(positive)),
            "total_count": len(positive), "top_shares": [
                {"top_count": min(int(n), len(positive)),
                 "share_of_positive_excess": float(positive[:n].sum() / total) if total else None}
                for n in top_counts]}


def pearson(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_saved():
    contract = read_json(V4 / "data_contract.json")
    evaluation = read_json(V4 / "v4_test_01.json")
    seal = read_json(checked_path(evaluation["seal"]))
    trainval_path, test_path = checked_path(contract["trainval"]), checked_path(contract["test"])
    with np.load(trainval_path, allow_pickle=False) as archive:
        trainval = {key: archive[key] for key in archive.files}
    with np.load(test_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if list(data["columns"]) != evaluation["channels"] or not np.array_equal(data["basis"], trainval["basis"]):
        raise ValueError("Stored channel order or TRAIN basis mismatch")
    if not np.array_equal(data["x"][:len(trainval["x"])], trainval["x"]):
        raise ValueError("Evaluation prefix differs from TRAIN/VAL archive")
    a, b = data["test_a_origins"], data["test_b_origins"]
    if not np.array_equal(a, evaluation["origins"]["test_a"]) or not np.array_equal(b, evaluation["origins"]["test_b"]):
        raise ValueError("Stored evaluation origins changed")
    origins = np.concatenate([a, b]).astype(np.int64)
    target = np.stack([data["x"][o:o + HORIZON] for o in origins])
    mask = np.stack([data["finite"][o:o + HORIZON] for o in origins]).astype(bool)
    context = np.stack([data["x"][o - CONTEXT:o] for o in origins])
    if target.shape != (80, HORIZON, CHANNELS) or context.shape != (80, CONTEXT, CHANNELS):
        raise ValueError("Hog context/target axes or sample count changed")
    models, receipts = {}, [receipt(trainval_path), receipt(test_path), receipt(V4 / "v4_test_01.json"),
                            receipt(V4 / "data_contract.json"), receipt(V4 / "evaluation_seal.json")]
    for role in ("tsfm_res", "tsfm_raw", "lora", "f0", "compress"):
        for run_id in evaluation["roles"][role]:
            path = checked_path(evaluation["models"][run_id]["prediction"])
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(archive["origins"], origins):
                    raise ValueError(f"Prediction origin mismatch: {run_id}")
                models[run_id] = archive["prediction"].copy()
            if models[run_id].shape != target.shape:
                raise ValueError(f"Prediction axes mismatch: {run_id}")
            receipts.append(receipt(path))
    return data, trainval, evaluation, seal, origins, target, mask, context, models, receipts


def orthogonal_spaces(trainval):
    started = time.perf_counter()
    train = trainval["x"][int(trainval["train_start"]):int(trainval["train_end"])].astype(np.float64)
    centered = train - train.mean(axis=0, keepdims=True)
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / len(train))
    pca16 = vectors[:, -16:][:, ::-1]
    stored_q8 = trainval["basis"].astype(np.float64)
    q8, _ = np.linalg.qr(stored_q8)
    extra = pca16[:, 8:] - q8 @ (q8.T @ pca16[:, 8:])
    q_extra, _ = np.linalg.qr(extra)
    q16 = np.concatenate([q8, q_extra], axis=1)
    if np.max(np.abs(q16.T @ q16 - np.eye(16))) > 1e-10:
        raise ValueError("Diagnostic Q8 and added TRAIN PCA directions are not orthogonal")
    span_difference = float(np.linalg.norm(q16 @ q16.T - pca16 @ pca16.T))
    if span_difference > 1e-4:
        raise ValueError("Stored Q8 span is inconsistent with TRAIN top-16 PCA")
    return q8, q_extra, q16, pca16, {"seconds": time.perf_counter() - started,
        "training_rows": len(train), "method": "TRAIN-only centered covariance eigendecomposition in float64; QR preserves stored Q8 span; added top-16 directions are orthogonalized against Q8.",
        "top16_subspace_difference_frobenius": span_difference,
        "stored_q8_orthogonality_max_error": float(np.max(np.abs(stored_q8.T @ stored_q8 - np.eye(8)))),
        "eigenvalues_descending": eigenvalues[::-1].tolist(),
        "classification": "Data-preparation PCA statistic, not a predictive-model fit."}


def diagnose(args):
    started = time.perf_counter()
    deadline = started + args.reserve_seconds

    def check_time():
        if time.perf_counter() >= deadline:
            raise RuntimeError("V5 diagnosis CPU time reservation exhausted")

    data, trainval, evaluation, seal, origins, target, mask, context, predictions, sources = load_saved()
    periods = {"test_a": np.arange(40), "test_b": np.arange(40, 80), "combined": np.arange(80)}
    period_for_origin = ["test_a"] * 40 + ["test_b"] * 40
    timestamps = data["times"].astype("datetime64[ns]")
    channels = data["columns"].tolist()
    model_metrics, reproduction = {}, []
    for run_id, pred in predictions.items():
        model_metrics[run_id] = {}
        for period, indices in periods.items():
            metric = summaries(pred[indices], target[indices], mask[indices])
            saved = evaluation["models"][run_id]["scores"][period]
            for key in ("mse", "mae", "channel_mse", "channel_mae", "channel_target_count"):
                if not np.allclose(metric[key], saved[key], atol=1e-10, rtol=1e-10):
                    raise ValueError(f"Stored score mismatch: {run_id}/{period}/{key}")
            reproduction.append({"run_id": run_id, "period": period,
                                 "mse_difference": metric["mse"] - saved["mse"],
                                 "mae_difference": metric["mae"] - saved["mae"]})
            model_metrics[run_id][period] = metric
    check_time()
    q8, q_extra, q16, pca16, pca_record = orthogonal_spaces(trainval)
    raw_basis = trainval["basis"].astype(np.float64)
    residual = context.astype(np.float64) - (context @ raw_basis) @ raw_basis.T
    last = residual[:, -1, :]
    recent_mean = residual[:, -24:, :].mean(axis=1)
    recent_std = residual[:, -24:, :].std(axis=1)
    compress_id = evaluation["roles"]["compress"][0]
    level_only = predictions[compress_id].astype(np.float64) + last[:, None, :]
    level_metrics = {period: summaries(level_only[ix], target[ix], mask[ix]) for period, ix in periods.items()}
    check_time()
    channel_rows, origin_rows, timestamp_rows, projection_rows, relation_rows, concentration_rows = [], [], [], [], [], []
    period_rows = []
    for seed in SEEDS:
        res_id, lora_id = f"v4_hog_tsfm_res_fixed_ed_{seed}", f"v4_hog_lora_{seed}"
        res, lora = predictions[res_id].astype(np.float64), predictions[lora_id].astype(np.float64)
        res_error, lora_error = res - target, lora - target
        correction = target.astype(np.float64) - res
        for period, ix in periods.items():
            sr, sl = model_metrics[res_id][period], model_metrics[lora_id][period]
            period_rows.append({"seed": seed, "period": period, "res_mse": sr["mse"],
                "lora_mse": sl["mse"], "delta_mse_res_minus_lora": sr["mse"] - sl["mse"],
                "relative_mse_change_percent": relative_change(sr["mse"], sl["mse"]),
                "res_mae": sr["mae"], "lora_mae": sl["mae"],
                "delta_mae_res_minus_lora": sr["mae"] - sl["mae"],
                "res_signed_bias": sr["signed_bias"], "lora_signed_bias": sl["signed_bias"],
                "observed_forecast_target_terms": sr["observed_forecast_target_terms"]})
            for c, name in enumerate(channels):
                channel_rows.append({"seed": seed, "period": period, "channel": name,
                    "observed_forecast_target_terms": sr["channel_target_count"][c],
                    "res_mse": sr["channel_mse"][c], "lora_mse": sl["channel_mse"][c],
                    "delta_mse_res_minus_lora": sr["channel_mse"][c] - sl["channel_mse"][c],
                    "relative_mse_change_percent": relative_change(sr["channel_mse"][c], sl["channel_mse"][c]),
                    "res_mae": sr["channel_mae"][c], "lora_mae": sl["channel_mae"][c],
                    "delta_mae_res_minus_lora": sr["channel_mae"][c] - sl["channel_mae"][c],
                    "res_signed_bias_pred_minus_target": sr["channel_signed_bias"][c],
                    "lora_signed_bias_pred_minus_target": sl["channel_signed_bias"][c]})
            delta = res_error[ix] ** 2 - lora_error[ix] ** 2
            channel_count = mask[ix].sum(axis=(0, 1))
            weighted = np.where(mask[ix], delta / (CHANNELS * channel_count[None, None, :]), 0.0)
            observed_excess = weighted[mask[ix]]
            concentration_rows.append({"seed": seed, "period": period,
                "channel_excess": concentration(np.array(sr["channel_mse"]) - np.array(sl["channel_mse"]), [1, 3, 5, 8]),
                "forecast_term_excess": concentration(observed_excess, [max(1, int(np.ceil(len(observed_excess) * f))) for f in (0.001, 0.01, 0.05, 0.10)]),
                "forecast_term_top_fractions": [0.001, 0.01, 0.05, 0.10],
                "weighted_net_delta_mse": float(weighted.sum()),
                "fraction_observed_terms_res_squared_error_larger": float((delta[mask[ix]] > 0).mean()),
                "fraction_channels_res_mse_larger": float((np.array(sr["channel_mse"]) > np.array(sl["channel_mse"])).mean())})
            complete = mask[ix].all(axis=2)
            er, el = res_error[ix][complete], lora_error[ix][complete]
            if len(er):
                def energies(error):
                    inside = np.sum((error @ q8) ** 2, axis=1)
                    added = np.sum((error @ q_extra) ** 2, axis=1)
                    outside_error = error - (error @ q16) @ q16.T
                    outside = np.sum(outside_error ** 2, axis=1)
                    total = np.sum(error ** 2, axis=1)
                    if not np.allclose(inside + added + outside, total, atol=1e-8, rtol=1e-10):
                        raise ValueError("Orthogonal error-energy identity failed")
                    return {"q8": float(inside.mean() / CHANNELS),
                            "q16_minus_q8": float(added.mean() / CHANNELS),
                            "outside_q16": float(outside.mean() / CHANNELS),
                            "total": float(total.mean() / CHANNELS)}
                energy_res, energy_lora = energies(er), energies(el)
                delta_energy = {key: energy_res[key] - energy_lora[key] for key in energy_res}
            else:
                energy_res = energy_lora = delta_energy = None
            projection_rows.append({"seed": seed, "period": period,
                "complete_target_vectors": int(complete.sum()), "all_target_vectors": int(complete.size),
                "complete_vector_fraction": float(complete.mean()),
                "complete_vectors_by_origin": complete.sum(axis=1).tolist(),
                "res_energy_per_channel": energy_res, "lora_energy_per_channel": energy_lora,
                "delta_energy_res_minus_lora": delta_energy,
                "full_masked_macro_delta_mse": sr["mse"] - sl["mse"],
                "completecase_and_full_direction_match": bool(np.sign(delta_energy["total"]) == np.sign(sr["mse"] - sl["mse"])) if delta_energy else None})
            counts = mask[ix].sum(axis=1)
            desired_mean = np.divide(np.where(mask[ix], correction[ix], 0.0).sum(axis=1), counts,
                                     out=np.full(counts.shape, np.nan), where=counts > 0)
            for feature_name, feature in [("residual_last", last), ("residual_mean_last24", recent_mean),
                                          ("residual_std_last24", recent_std)]:
                f = feature[ix]
                expanded = np.broadcast_to(f[:, None, :], correction[ix].shape)
                products = np.where(mask[ix], expanded * correction[ix], 0.0)
                dot = float(np.mean(products.sum(axis=(0, 1)) / channel_count))
                valid_pairs = np.isfinite(desired_mean) & np.isfinite(f)
                nonzero = valid_pairs & (f != 0) & (desired_mean != 0)
                relation_rows.append({"seed": seed, "period": period, "feature": feature_name,
                    "origin_channel_pairs": int(valid_pairs.sum()),
                    "pearson_with_mean_desired_correction": pearson(f, desired_mean),
                    "pearson_with_abs_mean_desired_correction": pearson(f, np.abs(desired_mean)),
                    "same_sign_fraction_excluding_zeros": float((np.sign(f[nonzero]) == np.sign(desired_mean[nonzero])).mean()) if nonzero.any() else None,
                    "nonzero_origin_channel_pairs": int(nonzero.sum()),
                    "channel_equal_observed_dot_with_desired_correction": dot,
                    "direction_note": "Desired correction is target minus stored RES; residual_std is nonnegative and its sign agreement is not a directional mechanism test."})
        for n, origin in enumerate(origins):
            sr = summaries(res[n:n + 1], target[n:n + 1], mask[n:n + 1], False)
            sl = summaries(lora[n:n + 1], target[n:n + 1], mask[n:n + 1], False)
            origin_rows.append({"seed": seed, "period": period_for_origin[n], "origin": int(origin),
                "origin_time": str(timestamps[origin]), "channels_with_observations": sr["channels_with_observations"],
                "observed_forecast_target_terms": sr["observed_forecast_target_terms"],
                "res_mse": sr["mse"], "lora_mse": sl["mse"],
                "delta_mse_res_minus_lora": sr["mse"] - sl["mse"] if sr["mse"] is not None else None,
                "relative_mse_change_percent": relative_change(sr["mse"], sl["mse"]) if sr["mse"] is not None else None,
                "res_mae": sr["mae"], "lora_mae": sl["mae"],
                "res_signed_bias": sr["signed_bias"], "lora_signed_bias": sl["signed_bias"]})
        for period in ("test_a", "test_b"):
            ix = periods[period]
            target_indices = origins[ix, None] + np.arange(HORIZON)[None, :]
            for instant in np.unique(target_indices):
                selected = target_indices == instant
                sr = summaries(res[ix][selected][:, None, :], target[ix][selected][:, None, :], mask[ix][selected][:, None, :], False)
                sl = summaries(lora[ix][selected][:, None, :], target[ix][selected][:, None, :], mask[ix][selected][:, None, :], False)
                timestamp_rows.append({"seed": seed, "period": period, "target_index": int(instant),
                    "target_time": str(timestamps[instant]), "overlapping_forecasts": int(selected.sum()),
                    "channels_with_observations": sr["channels_with_observations"],
                    "res_mse": sr["mse"], "lora_mse": sl["mse"],
                    "delta_mse_res_minus_lora": sr["mse"] - sl["mse"] if sr["mse"] is not None else None,
                    "res_mae": sr["mae"], "lora_mae": sl["mae"],
                    "res_signed_bias": sr["signed_bias"], "lora_signed_bias": sl["signed_bias"]})
            check_time()
    combined_projection = [row for row in projection_rows if row["period"] == "combined"]
    projection_available = all(row["delta_energy_res_minus_lora"] is not None for row in combined_projection)
    mean_delta = {key: float(np.mean([row["delta_energy_res_minus_lora"][key] for row in combined_projection]))
                  for key in ("q8", "q16_minus_q8", "outside_q16", "total")} if projection_available else None
    direction_match = projection_available and all(row["completecase_and_full_direction_match"] for row in combined_projection)
    last_alignment = float(np.mean([row["channel_equal_observed_dot_with_desired_correction"] for row in relation_rows
                                   if row["period"] == "combined" and row["feature"] == "residual_last"]))
    level_improves = level_metrics["combined"]["mse"] < model_metrics[compress_id]["combined"]["mse"]
    level_gate = bool(level_improves and last_alignment > 0)
    k_gate = bool(direction_match and mean_delta["q16_minus_q8"] > 0 and
                  max(0.0, mean_delta["q16_minus_q8"]) > max(0.0, mean_delta["outside_q16"])) if mean_delta else False
    first = "LEVEL" if level_gate else "K" if k_gate else "FULL"
    out = HERE / args.name
    cache = CACHE / args.name
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    array_path = cache / "diagnostic_arrays.npz"
    np.savez_compressed(array_path, origins=origins, level_only_prediction=level_only,
                        residual_last=last, residual_recent24_mean=recent_mean,
                        residual_recent24_std=recent_std, q8_diagnostic=q8,
                        q8_to_q16_diagnostic=q_extra, q16_diagnostic=q16, pca16_train=pca16)
    csv_receipts = {"channel": write_csv(out / "channel_differences.csv", channel_rows),
                    "period_seed": write_csv(out / "period_seed_differences.csv", period_rows),
                    "origin": write_csv(out / "origin_differences.csv", origin_rows),
                    "target_time": write_csv(out / "target_time_differences.csv", timestamp_rows),
                    "residual_relation": write_csv(out / "residual_relations.csv", relation_rows)}
    result = {"schema_version": 1, "status": "complete", "name": args.name,
        "dataset": "hog", "stage": "exposed development diagnosis", "new_model_fits": 0,
        "new_model_inferences": 0, "level_only_algebraic_forecasts": True,
        "sources": sources, "script": receipt(Path(__file__)), "csv": csv_receipts,
        "arrays": receipt(array_path), "saved_score_reproduction": reproduction,
        "res_minus_lora_by_seed_period": period_rows,
        "saved_model_metrics": model_metrics, "level_only": {
            "definition": "stored fixed COMPRESS prediction + broadcast final past reconstruction residual",
            "formula": "D(F0(E(X))) + repeat_H((X-D(E(X)))[-1])", "learned_parameters": 0,
            "comparison_scope": "This is the common initial LEVEL function, not the original RES initial function.",
            "metrics": level_metrics},
        "projection": {"pca_statistics": pca_record, "rows": projection_rows,
            "mean_combined_seed_delta": mean_delta,
            "target_missingness": "Only target vectors with all 32 channels observed are projected. Missing targets are never replaced by zero for projection.",
            "metric_scope": "Complete-case Euclidean error energy divided by 32 differs from the primary masked channel-macro MSE; coverage and full-metric direction are reported.",
            "k_limitation": "Adding independent PCA latent series can address added directions but does not directly change the retained Q8 main forecasts when their basis/scaling is preserved. Q8 error remains; negative excess outside a subspace does not mean zero absolute error there."},
        "positive_excess_concentration": concentration_rows,
        "residual_relations": relation_rows,
        "selection": {"first_modification": first, "level_gate": level_gate,
            "level_only_improves_compress_combined_mse": bool(level_improves),
            "mean_seed_last_residual_direction_dot": last_alignment, "k_gate": k_gate,
            "completecase_full_directions_match_for_both_seeds": bool(direction_match),
            "rule": "LEVEL if LEVEL_ONLY combined masked MSE improves COMPRESS and mean-seed channel-equal observed dot(last residual, target-RES)>0; else K if both-seed complete/full directions agree and positive added-Q16-minus-Q8 excess exceeds positive outside-Q16 excess; else FULL.",
            "interpretation": "A bounded development choice, not a causal mechanism finding. FULL fallback tests the remaining low-rank restriction; it is not itself evidence that rank is the cause."},
        "limitations": ["All Hog TEST-A/B values were previously exposed and are development information in v5.",
            "Overlapping 48-hour forecasts are not independent observations; concentration and correlations are descriptive.",
            "Per-origin and per-target-time diagnostics average channels with available targets and report their count; these are not substitutes for the pooled primary metric.",
            "Direction and correlation do not establish causality; recent statistics use only the 512-point past context.",
            "No site, channel, period, checkpoint, or original result is removed based on this diagnosis."],
        "elapsed_cpu_wall_seconds": time.perf_counter() - started}
    check_time()
    save_json(out / "diagnosis.json", result)
    save_json(HERE / "diagnosis_latest.json", {"path": str(out / "diagnosis.json"),
                                               "sha256": digest(out / "diagnosis.json"),
                                               "first_modification": first})
    print(json.dumps({"status": "complete", "diagnosis": str(out / "diagnosis.json"),
                      "first_modification": first, "elapsed_seconds": result["elapsed_cpu_wall_seconds"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v5_hog_diagnosis_01")
    parser.add_argument("--reserve-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.reserve_seconds <= 1800:
        raise ValueError("Diagnosis is bounded by 1800 CPU wall seconds")
    if (HERE / args.name).exists() or (CACHE / args.name).exists():
        raise FileExistsError("Preserve prior diagnosis; a documented repair requires a new name")
    with threadpool_limits(limits=4), Job(args.name, "cpu_analysis", reserve_seconds=args.reserve_seconds):
        diagnose(args)


if __name__ == "__main__":
    main()
