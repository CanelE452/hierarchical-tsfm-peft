"""Saved A full/main and actual-parent Q/last analysis; no TSFM execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from runtime_v12 import HERE, CACHE, ROOT, OLD, Job, read_json, save_json, artifact, import_file

DATASETS = ("robin", "jena", "hog")
SEEDS = (92601, 92602)
PERIODS = ("test_a", "test_b", "combined")
VARIANTS = ("A_FULL", "A_MAIN", "A_MAIN_LAST")
BASELINES = ("base_level", "f0", "lora_native", "full_lora_mse", "direct_nlinear")
ATOL, RTOL = 1e-5, 1e-4
SCORE_ATOL = 1e-10
SCOPE = ("Previously exposed Robin/Jena/Hog development data; same selected v11 A checkpoints. "
         "Q removal/replacement is a fit-zero post-hoc ablation, not independent main-only training. "
         "The actual approved v12 PLAN and its new confirmation units have not been recovered; "
         "this artifact does not execute or substitute for those units.")


def array_receipt(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256_c_order": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def verify_file(item):
    path = Path(item["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = artifact(path)
    if actual["sha256"] != item["sha256"]:
        raise ValueError(f"Pinned artifact changed: {path}")
    return path


def parity(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Parity requires finite arrays with matching shapes")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    violations = delta > ATOL + RTOL * np.abs(right.astype(np.float64))
    return {"atol": ATOL, "rtol": RTOL, "max_abs": float(delta.max(initial=0)),
            "max_rel": float((delta / np.maximum(np.abs(right), 1e-12)).max(initial=0)),
            "violations": int(violations.sum()), "elements": int(delta.size),
            "pass": bool(not violations.any())}


def load_prediction(item, origins, shape, with_main=False):
    path = verify_file(item)
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["origins"], origins):
            raise ValueError(f"Prediction origins differ: {path}")
        result = {"prediction": saved["prediction"].copy()}
        if with_main:
            result["main_only_prediction"] = saved["main_only_prediction"].copy()
    for name, value in result.items():
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Prediction schema/finiteness differs: {path}/{name}")
    return result


def reconstruct_q(parent, data, origins, dataset, seed, job):
    import torch
    import torch.nn.functional as functional

    checkpoint = verify_file(parent["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = saved["spec"]
    is_level = (spec.get("family") == "level_res" or
                (spec.get("method") == "level" and spec.get("variant") == "res"))
    if (spec.get("dataset") != dataset or int(spec.get("seed", -1)) != seed or
            spec.get("ed_mode") != "fixed_ed" or not is_level):
        raise ValueError("Q source is not the paired fixed-ED trained LEVEL parent")
    state = saved["state"]
    names = {"encoder.weight", "decoder.weight", "residual.down.weight", "residual.up.weight"}
    if set(state) != names:
        raise ValueError("Q parent must contain precisely fixed E/D and learned rank-32 G")
    basis = torch.as_tensor(data["basis"], dtype=torch.float32)
    if (not torch.equal(torch.as_tensor(saved["basis"]).float(), basis) or
            not torch.equal(state["encoder.weight"], basis.T) or
            not torch.equal(state["decoder.weight"], basis)):
        raise ValueError("Actual parent E/D/basis does not equal the sealed data PCA")
    for name, value in state.items():
        if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"Parent tensor is not finite CPU float32: {name}")
    if state["residual.down.weight"].shape != (32, 512) or state["residual.up.weight"].shape != (48, 32):
        raise ValueError("Q parent is not the shared bias-free 512-32-48 predictor")
    qs, lasts = [], []
    with torch.no_grad():
        for start in range(0, len(origins), 4):
            job.check_limits()
            x = torch.as_tensor(np.stack([data["x"][o - 512:o] for o in origins[start:start + 4]]))
            z = functional.linear(x, state["encoder.weight"])
            residual = x - functional.linear(z, state["decoder.weight"])
            last = residual[:, -1:, :].detach()
            hidden = functional.linear((residual - last).transpose(1, 2), state["residual.down.weight"])
            correction = functional.linear(hidden, state["residual.up.weight"]).transpose(1, 2)
            qs.append((last.expand(-1, 48, -1) + correction).numpy().copy())
            lasts.append(last.expand(-1, 48, -1).numpy().copy())
    receipt = {"checkpoint": artifact(checkpoint), "parent_id": parent["id"], "spec": spec,
               "basis": array_receipt(data["basis"]),
               "state": {name: array_receipt(value.numpy()) for name, value in state.items()},
               "calculation": "r=x-D(E(x)); q=last(r)+G(r-last(r)); last=last(r) broadcast to horizon 48",
               "device": "cpu", "dtype": "float32", "batch_size": 4,
               "backbone_loads": 0, "backbone_forward_calls": 0, "optimizer_updates": 0}
    return np.concatenate(qs), np.concatenate(lasts), receipt


def pooled_scores(scores):
    count = np.array(scores["test_a"]["target_counts"]) + np.array(scores["test_b"]["target_counts"])
    sums = {name: np.array(scores["test_a"][name]) + np.array(scores["test_b"][name])
            for name in ("squared_error_sums", "absolute_error_sums", "signed_error_sums")}
    return {"mse": float(np.mean(sums["squared_error_sums"] / count)),
            "mae": float(np.mean(sums["absolute_error_sums"] / count)),
            "signed_mean_error": float(np.mean(sums["signed_error_sums"] / count)),
            "target_counts": count.tolist(), **{k: v.tolist() for k, v in sums.items()}}


def verify_scores(actual, expected, label):
    differences = {}
    actual_counts = np.asarray(actual["target_counts"])
    expected_counts = np.asarray(expected["target_counts"])
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError(f"Observed target counts differ: {label}")
    for metric in ("mse", "mae", "signed_mean_error", "channel_mse", "channel_mae", "channel_bias",
                   "squared_error_sums", "absolute_error_sums", "signed_error_sums"):
        if metric in actual and metric in expected:
            difference = np.abs(np.asarray(actual[metric]) - expected[metric])
            checked = difference / actual_counts if metric.endswith("_sums") else difference
            differences[metric] = {"max_abs_raw": float(np.max(difference)),
                                   "max_abs_score_units": float(np.max(checked))}
            if np.any(checked > SCORE_ATOL):
                raise ValueError(f"Saved/direct/pooling score mismatch: {label}/{metric}")
    return {"target_counts_exact": True, "score_atol": SCORE_ATOL,
            "raw_sum_check": "Divide each channel's absolute raw-sum difference by its exact observed count; same score-unit tolerance",
            "float64_epsilon": float(np.finfo(np.float64).eps),
            "channel_target_counts": actual_counts.tolist(), "differences": differences}


def complete_pq(prediction, target, basis, index, complete):
    observed = complete[index]
    coverage = int(observed.sum())
    total = int(observed.size)
    result = {"complete_vectors": coverage, "total_vectors": total,
              "coverage": coverage / total, "subset": array_receipt(observed),
              "scope": "Identical complete observed target vectors for every model; standardized coordinates"}
    if not coverage:
        return {**result, "status": "unavailable_no_complete_target_vector"}
    error = prediction[index][observed].astype(np.float64) - target[index][observed].astype(np.float64)
    u = basis.astype(np.float64)
    p = (error @ u) @ u.T
    q = error - p
    p_energy, q_energy = np.sum(p * p, axis=1), np.sum(q * q, axis=1)
    full_energy = np.sum(error * error, axis=1)
    return {**result, "status": "available", "p_mse": float(np.mean(p * p)),
            "q_mse": float(np.mean(q * q)), "complete_vector_mse": float(np.mean(error * error)),
            "channel_p_mse": np.mean(p * p, axis=0).tolist(),
            "channel_q_mse": np.mean(q * q, axis=0).tolist(),
            "max_abs_cross_term": float(np.max(np.abs(2 * np.sum(p * q, axis=1)))),
            "energy_sum_gap_max_abs": float(np.max(np.abs(full_energy - p_energy - q_energy))),
            "warning": "Complete-vector P/Q MSE is not the original masked channel-macro MSE"}


def add_model(models, identifier, family, seed, prediction, target, mask, basis, indices,
              complete, oldruntime, source, prior=None):
    scores = {period: oldruntime.score_arrays(prediction[index], target[index], mask[index])
              for period, index in indices.items()}
    pooled = pooled_scores(scores)
    pooling_check = verify_scores(pooled, scores["combined"], identifier + "/direct_pooling")
    if prior is not None:
        for period in PERIODS:
            verify_scores(scores[period], prior[period], identifier + "/" + period)
    models[identifier] = {"id": identifier, "family": family, "seed": seed,
                          "source": source, "prediction": array_receipt(prediction), "periods": scores,
                          "combined_direct_period_pooling": {"pass": True, **pooled, "verification": pooling_check},
                          "prior_score_replay": None if prior is None else "passed",
                          "complete_pq": {period: complete_pq(prediction, target, basis, index, complete)
                                          for period, index in indices.items()}}


def contrasts(models):
    result = []
    for seed in SEEDS:
        indexed = {m["family"]: m for m in models.values() if m["seed"] == seed and m["family"] in VARIANTS}
        for period in PERIODS:
            for left, right, label in (("A_FULL", "A_MAIN", "whole_Q_gain_vs_main"),
                                       ("A_MAIN_LAST", "A_MAIN", "Q_last_gain_vs_main"),
                                       ("A_FULL", "A_MAIN_LAST", "learned_G_gain_over_Q_last")):
                ls, rs = indexed[left]["periods"][period], indexed[right]["periods"][period]
                result.append({"seed": seed, "period": period, "contrast": label,
                               "left": left, "reference": right,
                               "gain_sign": "reference error minus left error; positive means left improves",
                               "mse_gain": rs["mse"] - ls["mse"], "mae_gain": rs["mae"] - ls["mae"],
                               "bias_change": ls["signed_mean_error"] - rs["signed_mean_error"],
                               "channel_mse_gain": (np.array(rs["channel_mse"]) - ls["channel_mse"]).tolist(),
                               "channel_mae_gain": (np.array(rs["channel_mae"]) - ls["channel_mae"]).tolist()})
    return result


def summarize(models):
    result = {}
    for family in (*VARIANTS, *BASELINES):
        members = [m for m in models.values() if m["family"] == family]
        if len(members) != (1 if family == "f0" else 2):
            raise ValueError(f"Unexpected seed multiplicity: {family}")
        result[family] = {"ids": [m["id"] for m in members], "seeds": [m["seed"] for m in members],
                          "aggregation": "arithmetic mean of individual seed losses; never a prediction ensemble",
                          "periods": {period: {metric: float(np.mean([m["periods"][period][metric] for m in members]))
                                                for metric in ("mse", "mae", "signed_mean_error")}
                                      for period in PERIODS}}
    return result


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result, path):
    lines = ["# Fixed A Q contribution on existing development data", "", SCOPE, "",
             "[확인] Saved full/main predictions were reused. Only the actual paired LEVEL parent's E/D/G "
             "was evaluated on CPU; no TSFM load/forward, fitting, or benchmark was performed.", "",
             "`A_MAIN_LAST = saved A main + last context residual x - D(E(x))`, broadcast over 48 steps. "
             "The learned-G contrast compares this last-only Q contribution with the parent's learned Q.", "",
             "Scores use observed targets, pool per-channel SSE/count across test_a/test_b, then average "
             "channels. The combined rows below average seed losses, not predictions.", "", "```text",
             "dataset family                 combined_MSE combined_MAE signed_bias"]
    for dataset, data in result["datasets"].items():
        for family in (*VARIANTS, *BASELINES):
            score = data["summary"][family]["periods"]["combined"]
            lines.append(f"{dataset:7s} {family:22s} {score['mse']:12.8f} {score['mae']:12.8f} {score['signed_mean_error']:11.8f}")
    lines.extend(["```", "", "Positive gain below means the left model improves the reference. "
                  "Every seed and both periods remain available in JSON/CSV.", "", "```text",
                  "dataset contrast                         combined_MSE_gain combined_MAE_gain"])
    for dataset, data in result["datasets"].items():
        for label in ("whole_Q_gain_vs_main", "Q_last_gain_vs_main", "learned_G_gain_over_Q_last"):
            rows = [r for r in data["contrasts"] if r["period"] == "combined" and r["contrast"] == label]
            lines.append(f"{dataset:7s} {label:32s} {np.mean([r['mse_gain'] for r in rows]):17.8f} {np.mean([r['mae_gain'] for r in rows]):17.8f}")
    lines.extend(["```", "", "Complete-vector P/Q diagnostics use exactly the same observed-vector subset "
                  "for every model within each dataset/period. Their coverage is recorded; these "
                  "diagnostics are separate from the masked macro score.", "",
                  "CPU reconstructed main+Q must match the saved GPU full prediction at fixed "
                  "atol=1e-5, rtol=1e-4. Maximum discrepancies and all tolerance violations are retained "
                  "in JSON; the saved GPU full prediction remains the scoring reference.", "",
                  "All baselines reuse existing prediction bytes and reproduce their published v11 "
                  "scores. Original v11 files are preserved. This is neither an independent confirmation "
                  "nor evidence that separately training main-only would produce the same model.", "",
                  f"Numerical source: `{result['name']}.json`. Prediction arrays: `{CACHE / result['name']}`.",
                  ""])
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(name, job):
    import torch

    started = time.perf_counter()
    torch.set_num_threads(4)
    oldruntime = import_file(OLD / "runtime_v11.py", "v12_q_old_runtime")
    selected = read_json(OLD / "selected.json")
    reuse = read_json(OLD / "reuse_manifest.json")
    manifest = read_json(OLD / "prediction_manifest.json")
    previous = read_json(OLD / "evaluation01.json")
    if manifest["status"] != "complete" or previous["status"] != "complete":
        raise ValueError("Source predictions/evaluation must be complete")
    verify_file(manifest["seal"])
    folder = CACHE / name
    if folder.exists() or (HERE / f"{name}.json").exists():
        raise FileExistsError("Preserve prior outputs; explicitly choose a fresh attempt name")
    folder.mkdir(parents=True)
    result = {"schema": "v12_existing_A_Q_contribution_v1", "name": name, "status": "running",
              "scope": SCOPE, "independent_confirmation": False, "new_units_status": "actual_plan_not_recovered",
              "new_fits": 0, "optimizer_updates": 0, "backbone_loads": 0, "backbone_forward_calls": 0,
              "source_files": {path.name: artifact(path) for path in
                               (Path(__file__), OLD / "selected.json", OLD / "reuse_manifest.json",
                                OLD / "prediction_manifest.json", OLD / "evaluation01.json",
                                OLD / "model_v11.py", OLD / "runtime_v11.py")},
              "parity_tolerance": {"atol": ATOL, "rtol": RTOL},
              "score_replay_atol": SCORE_ATOL, "datasets": {}}
    scalar_rows, channel_rows, contrast_rows = [], [], []
    for dataset in DATASETS:
        job.check_limits()
        data = oldruntime.load_data(dataset, include_test=True)
        origins = np.concatenate([data[p + "_origins"] for p in PERIODS[:2]]).astype(np.int64)
        count_a = len(data["test_a_origins"])
        indices = {"test_a": np.arange(count_a), "test_b": np.arange(count_a, len(origins)),
                   "combined": np.arange(len(origins))}
        target = np.stack([data["x"][o:o + 48] for o in origins])
        mask = np.stack([data["finite"][o:o + 48] for o in origins]).astype(bool)
        complete = mask.all(axis=2)
        rows = manifest["datasets"][dataset]["models"]
        models, reconstructions = {}, {}
        for seed_index, seed in enumerate(SEEDS):
            row = next(r for r in rows if r["family"] == "a_p_lora_qfixed" and r["seed"] == seed)
            selection = selected["selected"][dataset]["a_p_lora_qfixed"]
            checkpoint = {"path": selection["checkpoints"][seed_index],
                          "sha256": selection["checkpoint_sha256"][seed_index]}
            if selection["run_ids"][seed_index] != row["id"]:
                raise ValueError("A prediction is not the selected paired checkpoint")
            verify_file(checkpoint)
            parent = reuse["parents"][dataset][str(seed)]
            if (row["sources"]["level_checkpoint_sha256"] != parent["checkpoint"]["sha256"] or
                    Path(row["sources"]["level_checkpoint"]).resolve() != Path(parent["checkpoint"]["path"]).resolve()):
                raise ValueError("A parent source binding differs")
            stored = load_prediction(row["prediction"], origins, target.shape, with_main=True)
            full, main = stored["prediction"], stored["main_only_prediction"]
            q, last, parent_receipt = reconstruct_q(parent, data, origins, dataset, seed, job)
            rebuilt = (main + q).astype(np.float32)
            full_check = parity(rebuilt, full)
            reconstruction = {"parent": parent_receipt, "selected_A_checkpoint": checkpoint,
                              "saved_A_prediction": row["prediction"], "origins": array_receipt(origins),
                              "full_cpu_reconstruction_vs_saved_gpu": full_check,
                              "q_cpu_vs_full_minus_main": parity(q, full.astype(np.float64) - main.astype(np.float64)),
                              "numerical_note": "CPU float32 linear algebra versus original CUDA float32; full-main also includes subtraction rounding"}
            reconstructions[str(seed)] = reconstruction
            save_json(HERE / f"{name}_{dataset}_{seed}_parity.json", reconstruction)
            if not full_check["pass"]:
                raise ValueError(f"Actual-parent CPU main+Q fails fixed parity: {dataset}/{seed}")
            main_last = (main + last).astype(np.float32)
            cache_path = folder / f"{dataset}_{seed}.npz"
            np.savez_compressed(cache_path, origins=origins, q_from_actual_parent=q,
                                q_last=last, main_last_prediction=main_last, reconstructed_full=rebuilt)
            reconstruction["derived_arrays"] = artifact(cache_path)
            old_model = previous["datasets"][dataset]["models"][row["id"]]
            for family, prediction, prior in (("A_FULL", full, old_model["periods"]),
                                               ("A_MAIN", main, old_model["main_only_fit0_ablation"]["periods"]),
                                               ("A_MAIN_LAST", main_last, None)):
                source = {"saved_prediction": row["prediction"], "array_key": "prediction" if family == "A_FULL" else "main_only_prediction",
                          "derived_arrays": artifact(cache_path) if family == "A_MAIN_LAST" else None,
                          "selected_A_id": row["id"]}
                add_model(models, f"{row['id']}__{family}", family, seed, prediction, target, mask,
                          data["basis"], indices, complete, oldruntime, source, prior=prior)
            job.heartbeat(f"Q analysis {dataset} seed {seed}: actual-parent parity and ablations complete")
        for row in rows:
            if row["family"] not in BASELINES:
                continue
            prediction = load_prediction(row["prediction"], origins, target.shape)["prediction"]
            old_model = previous["datasets"][dataset]["models"][row["id"]]
            add_model(models, row["id"], row["family"], row["seed"], prediction, target, mask,
                      data["basis"], indices, complete, oldruntime,
                      {"saved_prediction": row["prediction"], "prior_evaluation": artifact(OLD / "evaluation01.json")},
                      prior=old_model["periods"])
        values = contrasts(models)
        result["datasets"][dataset] = {"data_arrays": artifact(data["_path"]), "basis": array_receipt(data["basis"]),
            "origins": array_receipt(origins), "columns": [str(x) for x in data["columns"]],
            "period_origin_counts": {period: len(index) for period, index in indices.items()},
            "target_mask": array_receipt(mask), "reconstruction": reconstructions,
            "models": models, "summary": summarize(models), "contrasts": values}
        for identifier, model in models.items():
            for period, score in model["periods"].items():
                base = {"dataset": dataset, "family": model["family"], "id": identifier,
                        "seed": model["seed"], "period": period}
                pq = model["complete_pq"][period]
                scalar_rows.append({**base, **{k: score[k] for k in ("mse", "mae", "signed_mean_error")},
                                    "origins": score["origins"], "complete_vectors": pq["complete_vectors"],
                                    "complete_vector_coverage": pq["coverage"], "complete_p_mse": pq.get("p_mse"),
                                    "complete_q_mse": pq.get("q_mse"), "complete_vector_mse": pq.get("complete_vector_mse")})
                for channel, column in enumerate(data["columns"]):
                    channel_rows.append({**base, "channel": str(column), "mse": score["channel_mse"][channel],
                                         "mae": score["channel_mae"][channel], "signed_mean_error": score["channel_bias"][channel],
                                         "target_count": score["target_counts"][channel],
                                         "squared_error_sum": score["squared_error_sums"][channel],
                                         "absolute_error_sum": score["absolute_error_sums"][channel],
                                         "signed_error_sum": score["signed_error_sums"][channel]})
        contrast_rows.extend({"dataset": dataset, **{k: v for k, v in value.items() if not isinstance(v, list)}} for value in values)
        save_json(HERE / f"{name}_partial.json", result)
        job.heartbeat(f"Q analysis {dataset}: saved baseline replay and common-subset P/Q complete")
    for suffix, rows in (("models", scalar_rows), ("channels", channel_rows), ("contrasts", contrast_rows)):
        write_csv(HERE / f"{name}_{suffix}.csv", rows)
    result["status"] = "complete"
    result["elapsed_wall_s"] = time.perf_counter() - started
    result["outputs"] = {suffix: artifact(HERE / f"{name}_{suffix}.csv") for suffix in ("models", "channels", "contrasts")}
    save_json(HERE / f"{name}.json", result)
    write_report(result, HERE / "Q_CONTRIBUTION.md")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="q_contribution01")
    args = parser.parse_args()
    if not re.fullmatch(r"q_contribution[0-9]+", args.name):
        raise ValueError("Attempt name must be q_contribution followed by digits")
    with Job("cpu_analysis", args.name, reserve_s=900) as job:
        try:
            result = analyze(args.name, job)
        except Exception:
            failure = {"status": "failed", "name": args.name, "utc_unix": time.time(),
                       "exception": traceback.format_exc(), "source": artifact(Path(__file__)),
                       "scope": SCOPE, "prior_exception_preserved": True}
            path = HERE / f"{args.name}_failure_{time.time_ns()}.json"
            save_json(path, failure)
            raise
        print(json.dumps({"status": result["status"], "result": str(HERE / f"{args.name}.json"),
                          "new_fits": 0, "backbone_forward_calls": 0, "elapsed_wall_s": result["elapsed_wall_s"]}))


if __name__ == "__main__":
    main()
