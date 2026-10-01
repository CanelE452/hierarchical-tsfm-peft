"""Fit the v15 frozen-feature linear coordinate decoders.

This script is intentionally limited to TRAIN coefficient solving and VAL
selection. It never opens TEST archives and refuses to fit after the joint
selection seal or any evaluation exposure exists.
"""
import argparse
import copy
import math

import numpy as np

from runtime_v15 import (CACHE, DATASETS, HERE, ROOT, SEEDS, Job, artifact,
                         digest, load_data, read_json, require_storage,
                         resolve, save_json)


HORIZON = 48
FAMILIES = ("raw_free", "pca_free")
FEATURE_KIND = {"raw_free": "raw", "pca_free": "pca"}
LAMBDAS = (("0", 0.0), ("0p001", 0.001), ("0p1", 0.1))
V14 = ROOT / "research" / "tsfm_peft_nonlinear_correction_v14_20261001"


def as_float(value):
    return None if value is None else float(value)


def as_int(value):
    return None if value is None else int(value)


def scalar_array(value, dtype=np.float64):
    return np.asarray(value, dtype=dtype)


def json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def json_float_list(values):
    return [json_float(value) for value in values]


def fit_specs():
    protocol = read_json(HERE / "protocol.json")
    by_id = {row["id"]: row for row in protocol["fits"]}
    rows = []
    for dataset in DATASETS:
        for family in FAMILIES:
            for seed in SEEDS:
                for token, penalty in LAMBDAS:
                    fit_id = f"{dataset}_{family}_lam{token}_s{seed}"
                    row = copy.deepcopy(by_id[fit_id])
                    if (row["dataset"], row["family"], int(row["seed"]), float(row["lambda"])) != (
                            dataset, family, seed, penalty):
                        raise ValueError("Protocol fit row does not match the deterministic v15 matrix: " + fit_id)
                    row["lambda_token"] = token
                    row["feature_kind"] = FEATURE_KIND[family]
                    rows.append(row)
    return rows


def spec_by_id(fit_id):
    matches = [row for row in fit_specs() if row["id"] == fit_id]
    if len(matches) != 1:
        raise KeyError("Unknown v15 fit id: " + fit_id)
    return matches[0]


def public_run_dir(label):
    return HERE / "coefficient_runs" / label


def local_run_dir(label):
    return CACHE / "coefficient_runs" / label


def require_no_joint_exposure():
    for name in ("selection_seal.json", "test_exposure.json", "prediction_manifest.json",
                 "evaluation01.json"):
        if (HERE / name).exists():
            raise RuntimeError("Refuse v15 coefficient fitting after selection/evaluation exposure: " + name)


def load_selector_entry(dataset):
    manifest = read_json(HERE / "selector_manifest.json")
    entry = manifest["datasets"][dataset]
    c, k, _ = DATASETS[dataset]
    e_raw = np.asarray(entry["E_raw"], dtype=np.float64)
    e_pca = np.asarray(entry["E_pca"], dtype=np.float64)
    if e_raw.shape != (c, k) or e_pca.shape != (c, k):
        raise ValueError("Selector matrix shape differs from the fixed C/K contract: " + dataset)
    indices = np.asarray(entry["indices"], dtype=np.int64)
    if indices.shape != (k,) or len(set(indices.tolist())) != k:
        raise ValueError("Selector pivot index contract failed: " + dataset)
    if np.any(indices < 0) or np.any(indices >= c):
        raise ValueError("Selector pivot index out of channel range: " + dataset)
    result = copy.deepcopy(entry)
    result["E_raw_shape"] = list(e_raw.shape)
    result["E_pca_shape"] = list(e_pca.shape)
    result["_E_raw64"] = e_raw
    result["_E_pca64"] = e_pca
    result["selector_manifest"] = artifact(HERE / "selector_manifest.json")
    return result


def load_cache_manifest():
    return read_json(HERE / "cache_manifest.json")


def feature_cache_entry(cache_manifest, dataset, family):
    kind = FEATURE_KIND[family]
    entry = cache_manifest["datasets"][dataset][kind]
    for split in ("train", "val"):
        artifact(entry[split]["path"], entry[split]["sha256"])
    return entry


def parent_cache_entry(dataset, seed):
    source_path = V14 / "parent_cache.json"
    parents = read_json(source_path)
    entry = parents["datasets"][dataset]["direct"][str(seed)]
    for split in ("train", "val"):
        artifact(entry[split]["path"], entry[split]["sha256"])
    return {"source": artifact(source_path), "role": "direct", "seed": int(seed), **entry}


def load_prediction_npz(receipt, expected_origins, expected_last_dim, label):
    record = artifact(receipt["path"], receipt["sha256"])
    with np.load(resolve(record["path"]), allow_pickle=False) as archive:
        origins = archive["origins"].astype(np.int64)
        prediction = archive["prediction"].astype(np.float32, copy=True)
    if not np.array_equal(origins, expected_origins):
        raise ValueError(label + " origin order differs from the fixed data split")
    if prediction.shape != (len(expected_origins), HORIZON, expected_last_dim):
        raise ValueError(label + " prediction shape differs from fixed contract")
    if not np.isfinite(prediction).all():
        raise ValueError(label + " contains non-finite values")
    return prediction


def stacked_targets(data, origins):
    y = np.stack([data["x"][int(origin):int(origin) + HORIZON] for origin in origins]).astype(np.float64)
    mask = np.stack([data["finite"][int(origin):int(origin) + HORIZON] for origin in origins]).astype(bool)
    if y.shape[:2] != (len(origins), HORIZON) or mask.shape != y.shape:
        raise ValueError("Target/mask shape differs from fixed origin contract")
    return y, mask


def masked_channel_score(prediction, target, mask):
    finite = mask & np.isfinite(prediction) & np.isfinite(target)
    errors = prediction - target
    counts = finite.sum(axis=(0, 1)).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("Every target channel must have at least one observed value")
    squared = np.where(finite, errors * errors, 0.0).sum(axis=(0, 1))
    absolute = np.where(finite, np.abs(errors), 0.0).sum(axis=(0, 1))
    signed = np.where(finite, errors, 0.0).sum(axis=(0, 1))
    channel_mse = squared / counts
    channel_mae = absolute / counts
    channel_bias = signed / counts
    return {
        "mse": float(np.mean(channel_mse)),
        "mae": float(np.mean(channel_mae)),
        "signed_mean_error": float(np.mean(channel_bias)),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist(),
        "channel_bias": channel_bias.tolist(),
        "target_counts": counts.tolist(),
        "complete_target_channels": int(np.sum(counts > 0)),
        "observed_values": int(np.sum(counts)),
        "mask_contract": "original observed-target mask; no target filling",
    }


def fit_coefficients(z_train, parent_train, y_train, mask_train, penalty, job):
    if z_train.shape[:2] != y_train.shape[:2] or parent_train.shape != y_train.shape:
        raise ValueError("TRAIN design/parent/target shape mismatch")
    if not np.isfinite(z_train).all() or not np.isfinite(parent_train).all():
        raise ValueError("TRAIN frozen features or parent predictions contain non-finite values")
    n_origin, horizon, k = z_train.shape
    c = y_train.shape[-1]
    design = z_train.reshape(n_origin * horizon, k)
    residual_target = (y_train - parent_train).reshape(n_origin * horizon, c)
    observed_mask = mask_train.reshape(n_origin * horizon, c)
    train_scale = float(np.mean(np.sum(design * design, axis=1) / max(k, 1)))
    if not math.isfinite(train_scale):
        raise ValueError("Non-finite TRAIN feature scale")

    weights = np.zeros((k, c), dtype=np.float64)
    counts = np.zeros(c, dtype=np.int64)
    ranks = np.zeros(c, dtype=np.int64)
    base_ranks = [None for _ in range(c)]
    condition = np.full(c, np.nan, dtype=np.float64)
    max_singular = np.full(c, np.nan, dtype=np.float64)
    min_singular = np.full(c, np.nan, dtype=np.float64)
    objective = np.zeros(c, dtype=np.float64)
    data_loss = np.zeros(c, dtype=np.float64)
    penalty_loss = np.zeros(c, dtype=np.float64)
    normal_residual_maxabs = np.zeros(c, dtype=np.float64)
    fp64_norm = np.zeros(c, dtype=np.float64)
    policy = "ordinary_lstsq"

    if train_scale == 0.0:
        policy = "zero_feature_scale_declared_w0"

    for channel in range(c):
        job.check_limits()
        observed = observed_mask[:, channel] & np.isfinite(residual_target[:, channel])
        a = design[observed]
        b = residual_target[observed, channel]
        n = int(observed.sum())
        counts[channel] = n
        if n <= 0:
            raise ValueError("No observed TRAIN residual target for channel " + str(channel))
        if train_scale == 0.0:
            residual = -b
            data_loss[channel] = float(np.mean(residual * residual))
            objective[channel] = data_loss[channel]
            continue
        if penalty == 0.0:
            solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
            base_ranks[channel] = int(rank)
        else:
            scale = math.sqrt(float(n))
            reg = math.sqrt(float(penalty) * train_scale)
            augmented_a = np.vstack((a / scale, reg * np.eye(k, dtype=np.float64)))
            augmented_b = np.concatenate((b / scale, np.zeros(k, dtype=np.float64)))
            solution, _, rank, singular = np.linalg.lstsq(augmented_a, augmented_b, rcond=None)
        weights[:, channel] = solution
        ranks[channel] = int(rank)
        if singular.size:
            max_singular[channel] = float(np.max(singular))
            min_singular[channel] = float(np.min(singular))
            condition[channel] = (float(np.max(singular) / np.min(singular))
                                  if np.min(singular) > 0 else np.nan)
        residual = a @ solution - b
        data_loss[channel] = float(np.mean(residual * residual))
        penalty_loss[channel] = float(float(penalty) * train_scale * np.sum(solution * solution))
        objective[channel] = data_loss[channel] + penalty_loss[channel]
        normal = (a.T @ residual) / float(n) + float(penalty) * train_scale * solution
        normal_residual_maxabs[channel] = float(np.max(np.abs(normal))) if normal.size else 0.0
        fp64_norm[channel] = float(np.linalg.norm(solution))

    diagnostics = {
        "policy": policy,
        "lambda": float(penalty),
        "train_feature_scale_s_E": train_scale,
        "target_counts_by_channel": counts.tolist(),
        "target_count_min": int(np.min(counts)),
        "target_count_max": int(np.max(counts)),
        "base_design_rank_by_channel": base_ranks,
        "augmented_rank_by_channel": ranks.tolist(),
        "rank_min": int(np.min(ranks)),
        "rank_max": int(np.max(ranks)),
        "condition_by_channel": json_float_list(condition),
        "singular_value_max_by_channel": json_float_list(max_singular),
        "singular_value_min_by_channel": json_float_list(min_singular),
        "normal_equation_residual_maxabs_by_channel": normal_residual_maxabs.tolist(),
        "normal_equation_residual_maxabs": float(np.max(normal_residual_maxabs)),
        "fp64_weight_norm_by_channel": fp64_norm.tolist(),
        "fp64_weight_norm": float(np.linalg.norm(weights)),
        "objective_by_channel": objective.tolist(),
        "data_loss_by_channel": data_loss.tolist(),
        "penalty_loss_by_channel": penalty_loss.tolist(),
        "objective_macro": float(np.mean(objective)),
        "data_loss_macro": float(np.mean(data_loss)),
        "penalty_loss_macro": float(np.mean(penalty_loss)),
        "fit_arithmetic": "FP64 least squares; positive lambda uses augmented least squares",
    }
    return weights, diagnostics


def predict_with_weights(parent_prediction, features, weights32):
    adjustment = features.astype(np.float32) @ weights32
    return parent_prediction.astype(np.float32) + adjustment


def innovation_features(f0_cache_prediction, parent_prediction, selector_entry, family):
    matrix_key = "_E_raw64" if family == "raw_free" else "_E_pca64"
    e32 = selector_entry[matrix_key].astype(np.float32)
    projected_parent = parent_prediction.astype(np.float32) @ e32
    return (f0_cache_prediction.astype(np.float32) - projected_parent).astype(np.float64)


def result_candidates_for_fit(fit_id):
    root = HERE / "coefficient_runs"
    if not root.exists():
        return []
    rows = []
    for path in root.glob("*/result.json"):
        try:
            row = read_json(path)
        except Exception:
            continue
        if row.get("fit_id") == fit_id:
            rows.append((path, row))
    return rows


def result_for(spec, required=True):
    rows = []
    ledger = read_json(HERE / "ledger.json")
    by_id = {int(row["id"]): row for row in ledger["jobs"]}
    for path, row in result_candidates_for_fit(spec["id"]):
        if row.get("status") != "complete":
            continue
        if row.get("spec") != spec:
            raise ValueError("Stored v15 fit result spec changed: " + str(path))
        job_row = by_id.get(int(row["ledger_job_id"]))
        if not job_row or job_row.get("category") != "coefficient_fit":
            raise ValueError("Stored result is not bound to a coefficient_fit ledger row: " + str(path))
        if job_row.get("metadata", {}).get("fit_id") != spec["id"] or job_row.get("status") != "complete":
            raise ValueError("Stored result ledger binding is incomplete: " + str(path))
        artifact(row["artifacts"]["weights"]["path"], row["artifacts"]["weights"]["sha256"])
        artifact(row["artifacts"]["objective"]["path"], row["artifacts"]["objective"]["sha256"])
        artifact(row["artifacts"]["receipt"]["path"], row["artifacts"]["receipt"]["sha256"])
        row = copy.deepcopy(row)
        row["result_artifact"] = artifact(path)
        rows.append((path, row))
    if len(rows) > 1:
        raise RuntimeError("More than one complete result is bound to one fit id: " + spec["id"])
    if rows:
        return rows[0][1]
    if required:
        raise FileNotFoundError("Missing complete v15 coefficient result: " + spec["id"])
    return None


def fit_one(spec, parent_job, label, technical_retry=False, reason=None):
    require_no_joint_exposure()
    existing = result_for(spec, required=False)
    if existing is not None:
        return {"status": "already_complete", "fit_id": spec["id"], "result": existing["artifacts"]["result"]}

    metadata = {
        "fit_id": spec["id"],
        "dataset": spec["dataset"],
        "family": spec["family"],
        "seed": int(spec["seed"]),
        "lambda": float(spec["lambda"]),
        "lambda_token": spec["lambda_token"],
        "technical_retry": bool(technical_retry),
    }
    if reason:
        metadata["reason"] = reason
    with Job("coefficient_fit", label, reserve_s=0, metadata=metadata) as job:
        require_storage(1024 * 1024)
        public_dir = public_run_dir(label)
        local_dir = local_run_dir(label)
        if public_dir.exists() or local_dir.exists():
            raise FileExistsError("Refuse to overwrite v15 coefficient attempt directory: " + label)
        public_dir.mkdir(parents=True)
        local_dir.mkdir(parents=True)

        dataset, family, seed = spec["dataset"], spec["family"], int(spec["seed"])
        c, k, _ = DATASETS[dataset]
        penalty = float(spec["lambda"])
        selector = load_selector_entry(dataset)
        cache_manifest = load_cache_manifest()
        feature_cache = feature_cache_entry(cache_manifest, dataset, family)
        parent_cache = parent_cache_entry(dataset, seed)
        data = load_data(dataset, include_test=False)
        train_origins = data["train_origins"].astype(np.int64)
        val_origins = data["val_origins"].astype(np.int64)
        y_train, mask_train = stacked_targets(data, train_origins)
        y_val, mask_val = stacked_targets(data, val_origins)
        f0_train = load_prediction_npz(feature_cache["train"], train_origins, k,
                                       f"{dataset}/{family}/TRAIN F0 cache")
        f0_val = load_prediction_npz(feature_cache["val"], val_origins, k,
                                     f"{dataset}/{family}/VAL F0 cache")
        parent_train = load_prediction_npz(parent_cache["train"], train_origins, c,
                                           f"{dataset}/direct/{seed}/TRAIN parent")
        parent_val = load_prediction_npz(parent_cache["val"], val_origins, c,
                                         f"{dataset}/direct/{seed}/VAL parent")
        z_train = innovation_features(f0_train, parent_train, selector, family)
        z_val = innovation_features(f0_val, parent_val, selector, family)
        job.heartbeat("loaded frozen TRAIN/VAL targets, parent, F0 caches, and formed innovation features")

        weights64, diagnostics = fit_coefficients(z_train, parent_train, y_train, mask_train, penalty, job)
        weights32 = weights64.astype(np.float32)
        parent_val_score = masked_channel_score(parent_val.astype(np.float32), y_val, mask_val)
        val_prediction = predict_with_weights(parent_val, z_val, weights32)
        val_score = masked_channel_score(val_prediction, y_val, mask_val)
        fp64_val_prediction = parent_val.astype(np.float64) + z_val @ weights64
        fp32_loss_delta = {
            "mse_fp64": masked_channel_score(fp64_val_prediction, y_val, mask_val)["mse"],
            "mse_fp32_export": val_score["mse"],
            "maxabs_prediction_delta": float(np.max(np.abs(fp64_val_prediction - val_prediction))),
        }

        weights_path = local_dir / "weights.npz"
        np.savez_compressed(
            weights_path,
            W=weights32,
            W64=weights64,
            lambda_value=scalar_array(penalty),
            train_feature_scale_s_E=scalar_array(diagnostics["train_feature_scale_s_E"]),
            train_counts=np.asarray(diagnostics["target_counts_by_channel"], dtype=np.int64),
            objective_by_channel=np.asarray(diagnostics["objective_by_channel"], dtype=np.float64),
            data_loss_by_channel=np.asarray(diagnostics["data_loss_by_channel"], dtype=np.float64),
            penalty_loss_by_channel=np.asarray(diagnostics["penalty_loss_by_channel"], dtype=np.float64),
        )
        objective = {
            "schema": "v15_coefficient_objective_1",
            "fit_id": spec["id"],
            "ledger_job_id": int(job.id),
            "spec": spec,
            "diagnostics": diagnostics,
            "parent_val_score": parent_val_score,
            "val_score": val_score,
            "val_mse_delta_vs_parent": float(val_score["mse"] - parent_val_score["mse"]),
            "fp64_to_fp32_validation": fp32_loss_delta,
            "selection_split": "VAL only; TEST not loaded",
        }
        save_json(public_dir / "objective.json", objective)
        weights_artifact = artifact(weights_path)
        objective_artifact = artifact(public_dir / "objective.json")
        receipt = {
            "schema": "v15_coefficient_receipt_1",
            "fit_id": spec["id"],
            "ledger_job_id": int(job.id),
            "attempt_label": label,
            "spec": spec,
            "artifacts": {"weights": weights_artifact, "objective": objective_artifact},
            "source_files": {name: digest(HERE / name) for name in
                             ("PLAN.md", "protocol.json", "data_manifest.json", "reuse_manifest.json",
                              "selector_manifest.json", "fit_v15.py", "runtime_v15.py")
                             if (HERE / name).exists()},
            "data": artifact(data["_path"]),
            "selector": {
                "manifest": artifact(HERE / "selector_manifest.json"),
                "indices": selector["indices"],
                "columns": selector.get("columns"),
                "E_raw_shape": selector["E_raw_shape"],
                "E_pca_shape": selector["E_pca_shape"],
            },
            "feature_cache": {
                "manifest_path": str(HERE / "cache_manifest.json"),
                "manifest_mutability": "cache_manifest.json may later add TEST cache receipts; this fit binds the TRAIN/VAL receipts below",
                "kind": FEATURE_KIND[family],
                "prediction_semantics": "stored prediction is F0(XE); fitted design is FP32 F0(XE) - FP32 S*(X)E promoted to FP64",
                "train": feature_cache["train"],
                "val": feature_cache["val"],
            },
            "parent_cache": parent_cache,
            "target_windows": {
                "horizon": HORIZON,
                "train_origin_count": int(len(train_origins)),
                "val_origin_count": int(len(val_origins)),
                "train_origin_first": as_int(train_origins[0]) if len(train_origins) else None,
                "train_origin_last": as_int(train_origins[-1]) if len(train_origins) else None,
                "val_origin_first": as_int(val_origins[0]) if len(val_origins) else None,
                "val_origin_last": as_int(val_origins[-1]) if len(val_origins) else None,
            },
            "no_test_data_loaded": True,
            "optimizer_updates": 0,
            "model_runs": 0,
        }
        save_json(public_dir / "receipt.json", receipt)
        result = {
            "schema": "v15_coefficient_result_1",
            "status": "complete",
            "fit_id": spec["id"],
            "attempt_label": label,
            "ledger_job_id": int(job.id),
            "parent_job_id": int(parent_job.id),
            "spec": spec,
            "dataset": dataset,
            "family": family,
            "seed": seed,
            "lambda": penalty,
            "lambda_token": spec["lambda_token"],
            "feature_kind": FEATURE_KIND[family],
            "objective_macro": diagnostics["objective_macro"],
            "train_feature_scale_s_E": diagnostics["train_feature_scale_s_E"],
            "rank_min": diagnostics["rank_min"],
            "rank_max": diagnostics["rank_max"],
            "normal_equation_residual_maxabs": diagnostics["normal_equation_residual_maxabs"],
            "weight_norm_fp64": diagnostics["fp64_weight_norm"],
            "target_count_min": diagnostics["target_count_min"],
            "target_count_max": diagnostics["target_count_max"],
            "parent_val_score": parent_val_score,
            "val_score": val_score,
            "val_mse_delta_vs_parent": float(val_score["mse"] - parent_val_score["mse"]),
            "fp64_to_fp32_validation": fp32_loss_delta,
            "artifacts": {
                "weights": weights_artifact,
                "objective": objective_artifact,
                "receipt": artifact(public_dir / "receipt.json"),
            },
            "selection_eligible": True,
            "test_used": False,
        }
        save_json(public_dir / "result.json", result)
        job.heartbeat("completed coefficient solve and VAL score", updates=0)
        return {"status": "complete", "fit_id": spec["id"], "result": artifact(public_dir / "result.json")}


def fit_all(job, only=None, retry_suffix=None, technical_retry=False, reason=None):
    require_no_joint_exposure()
    rows = []
    specs = fit_specs()
    if only:
        selected = set(only)
        specs = [row for row in specs if row["id"] in selected]
        missing = selected - {row["id"] for row in specs}
        if missing:
            raise KeyError("Unknown requested fit ids: " + ", ".join(sorted(missing)))
    for index, spec in enumerate(specs, 1):
        job.check_limits()
        label = spec["id"] if retry_suffix is None else spec["id"] + "_" + retry_suffix
        rows.append(fit_one(spec, job, label, technical_retry=technical_retry, reason=reason))
        job.heartbeat(f"fit {index}/{len(specs)}: {spec['id']}", updates=0)
    save_json(HERE / "fit_attempt_summary.json", {
        "schema": "v15_fit_attempt_summary_1",
        "status": "complete",
        "parent_job_id": int(job.id),
        "requested_count": len(specs),
        "results": rows,
        "test_used": False,
    })
    return rows


def selection_candidates(dataset, family):
    parent_by_seed = {}
    lambda_by_token = {token: [] for token, _ in LAMBDAS}
    results_by_seed_token = {}
    for seed in SEEDS:
        parent_values = []
        for token, _ in LAMBDAS:
            spec = spec_by_id(f"{dataset}_{family}_lam{token}_s{seed}")
            result = result_for(spec)
            parent_values.append(float(result["parent_val_score"]["mse"]))
            lambda_by_token[token].append(float(result["val_score"]["mse"]))
            results_by_seed_token[(seed, token)] = result
        if max(parent_values) != min(parent_values):
            raise ValueError("Parent VAL score changed across lambda rows: " + dataset + "/" + family)
        parent_by_seed[seed] = parent_values[0]
    rows = [{
        "candidate": "parent",
        "lambda": None,
        "lambda_token": None,
        "tie_order": 0,
        "mean_seed_val_mse": float(np.mean([parent_by_seed[seed] for seed in SEEDS])),
        "per_seed_val_mse": {str(seed): parent_by_seed[seed] for seed in SEEDS},
        "zero_decoder": True,
    }]
    for tie_order, (token, penalty) in enumerate(LAMBDAS, 1):
        rows.append({
            "candidate": "lambda",
            "lambda": float(penalty),
            "lambda_token": token,
            "tie_order": tie_order,
            "mean_seed_val_mse": float(np.mean(lambda_by_token[token])),
            "per_seed_val_mse": {str(seed): float(results_by_seed_token[(seed, token)]["val_score"]["mse"])
                                 for seed in SEEDS},
            "zero_decoder": False,
            "run_ids": {str(seed): results_by_seed_token[(seed, token)]["fit_id"] for seed in SEEDS},
        })
    return rows, results_by_seed_token


def write_zero_weights(dataset, family, seed, selection_job_id):
    c, k, _ = DATASETS[dataset]
    path = CACHE / "selected_zero_weights" / str(selection_job_id) / dataset / family / str(seed) / "weights.npz"
    if path.exists():
        raise FileExistsError("Refuse to overwrite selected zero-weight receipt: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    require_storage(1024 * 1024)
    np.savez_compressed(
        path,
        W=np.zeros((k, c), dtype=np.float32),
        W64=np.zeros((k, c), dtype=np.float64),
        lambda_value=scalar_array(0.0),
        train_feature_scale_s_E=scalar_array(0.0),
        selection_job_id=np.asarray(selection_job_id, dtype=np.int64),
        zero_decoder=np.asarray(True),
    )
    return artifact(path)


def select(job):
    if (HERE / "selected.json").exists():
        raise FileExistsError("selected.json already exists without a new explicit repair")
    selected = {
        "schema": "v15_selected_coordinate_decoders_1",
        "status": "complete",
        "selection_job_id": int(job.id),
        "metric": "VAL observed channel-macro MSE, mean of two seed losses",
        "test_used": False,
        "tie_rule": "exact ties prefer parent, then lambda0, lambda0p001, lambda0p1",
        "datasets": {},
    }
    for dataset in DATASETS:
        selected["datasets"][dataset] = {}
        for family in FAMILIES:
            candidates, by_seed_token = selection_candidates(dataset, family)
            choice = min(candidates, key=lambda row: (row["mean_seed_val_mse"], row["tie_order"]))
            family_row = {
                "family": family,
                "dataset": dataset,
                "feature_kind": FEATURE_KIND[family],
                "selected_candidate": choice["candidate"],
                "selected_lambda": choice["lambda"],
                "selected_lambda_token": choice["lambda_token"],
                "selected_mean_seed_val_mse": choice["mean_seed_val_mse"],
                "parent_mean_seed_val_mse": candidates[0]["mean_seed_val_mse"],
                "zero_decoder": bool(choice["zero_decoder"]),
                "candidates": candidates,
                "seeds": {},
            }
            for seed in SEEDS:
                if choice["candidate"] == "parent":
                    family_row["seeds"][str(seed)] = {
                        "seed": int(seed),
                        "status": "parent_fallback",
                        "zero_decoder": True,
                        "run_id": None,
                        "weights": write_zero_weights(dataset, family, seed, int(job.id)),
                        "parent_val_mse": float(candidates[0]["per_seed_val_mse"][str(seed)]),
                        "feature_cache_required_for_deployment": False,
                        "zero_weight_artifact_semantics": "checkpoint-compatible W=0 receipt; online parent fallback does not require loading F0",
                        "parent_cache": parent_cache_entry(dataset, seed),
                    }
                else:
                    result = by_seed_token[(seed, choice["lambda_token"])]
                    family_row["seeds"][str(seed)] = {
                        "seed": int(seed),
                        "status": "selected_fit",
                        "zero_decoder": False,
                        "run_id": result["fit_id"],
                        "attempt_label": result["attempt_label"],
                        "lambda": result["lambda"],
                        "lambda_token": result["lambda_token"],
                        "weights": result["artifacts"]["weights"],
                        "result": result["result_artifact"],
                        "objective": result["artifacts"]["objective"],
                        "receipt": result["artifacts"]["receipt"],
                        "val_mse": float(result["val_score"]["mse"]),
                        "parent_val_mse": float(result["parent_val_score"]["mse"]),
                        "feature_cache_required_for_deployment": True,
                    }
            selected["datasets"][dataset][family] = family_row
            job.heartbeat("selected " + dataset + "/" + family)
    save_json(HERE / "selected.json", selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "select"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--reserve-s", type=float, required=True)
    parser.add_argument("--only", action="append", default=None,
                        help="Run only this fit id; repeat for multiple ids.")
    parser.add_argument("--retry-suffix", default=None,
                        help="Append to coefficient job labels for a technical retry; metadata.fit_id remains canonical.")
    parser.add_argument("--technical-retry", action="store_true")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args()
    if args.action == "fit":
        metadata = {"purpose": "v15 coefficient fits", "fit_count": len(args.only) if args.only else 48,
                    "model_runs": 0, "test_used": False}
        with Job("cpu_analysis", args.job, reserve_s=args.reserve_s, metadata=metadata) as job:
            fit_all(job, only=args.only, retry_suffix=args.retry_suffix,
                    technical_retry=args.technical_retry, reason=args.reason)
    else:
        if args.only or args.retry_suffix or args.technical_retry or args.reason:
            raise ValueError("Selection does not accept fit retry arguments")
        with Job("cpu_check", args.job, reserve_s=args.reserve_s,
                 metadata={"purpose": "v15 VAL selection for evaluator-owned seal", "fit_count": 0,
                           "model_runs": 0, "test_used": False}) as job:
            select(job)


if __name__ == "__main__":
    main()
