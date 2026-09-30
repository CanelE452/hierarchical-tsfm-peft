from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from runtime_v11 import CACHE, HERE, ROOT, Job, batches, digest, load_data, read_json, save_json, sources_for
from gamma_v11 import (DATASETS, SEEDS, check_receipt, configure, load_complete_selection, model_for_u,
                       parity, receipt, release_gpu, resolve, selected_checkpoint)

PERIODS = ("test_a", "test_b")
ROLES = ("a_p_lora_qfixed", "full_lora_mse", "b_fixed_u", "b_learned_u", "b_fixed_gamma",
         "b_learned_gamma", "base_level", "direct_nlinear", "f0", "lora_native")
ALIASES = {"level_res": "base_level", "level_tsfm_res": "base_level", "level": "base_level",
           "lora": "lora_native", "direct": "direct_nlinear"}
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 9262026


def role_name(role: str) -> str:
    return ALIASES.get(role, role)


def predicted_receipt(base: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    item = base.get("predictions", {}).get(run_id)
    if item is None:
        return None
    if "prediction" in item and isinstance(item["prediction"], dict):
        item = item["prediction"]
    if "path" not in item or "sha256" not in item:
        raise ValueError(f"Incomplete stored prediction receipt: {run_id}")
    check_receipt(item)
    return item


def evaluation_rows(selection: dict[str, Any], reuse: dict[str, Any], gamma: dict[str, Any],
                    dataset: str) -> list[dict[str, Any]]:
    rows = []
    baseline = reuse["baselines"][dataset]
    for family in ("a_p_lora_qfixed", "full_lora_mse", "b_learned_u", "direct_nlinear"):
        for seed in SEEDS:
            checkpoint = selected_checkpoint(selection, dataset, family, seed)
            row = {"id": checkpoint["run_id"], "dataset": dataset, "family": family, "seed": seed,
                   "checkpoint": checkpoint, "sources": sources_for(dataset, seed)}
            stored = predicted_receipt(baseline, row["id"])
            if stored is not None:
                row["stored_prediction"] = stored
            rows.append(row)
    for seed in SEEDS:
        rows.append({"id": f"v11_{dataset}_b_fixed_u_{seed}", "dataset": dataset,
                     "family": "b_fixed_u", "seed": seed, "sources": sources_for(dataset, seed)})
    for item in gamma["rows"]:
        if item["dataset"] != dataset:
            continue
        family = f"b_{item['kind']}_gamma"
        rows.append({"id": item["id"], "dataset": dataset, "family": family,
                     "seed": int(item["seed"]), "sources": sources_for(dataset, int(item["seed"])),
                     "gamma": item["gamma"], "gamma_features": item["features"],
                     "base_u_kind": item["kind"]})
    by_role = {role_name(key): value for key, value in baseline["roles"].items()}
    for family in ("base_level", "f0", "lora_native"):
        identifiers = list(by_role[family])
        expected = 1 if family == "f0" else 2
        if len(identifiers) != expected:
            raise ValueError(f"Incorrect baseline multiplicity: {dataset}/{family}")
        for index, identifier in enumerate(identifiers):
            seed = None if family == "f0" else SEEDS[index]
            stored = predicted_receipt(baseline, identifier)
            if stored is None:
                raise ValueError(f"Missing promised baseline prediction: {identifier}; explicit recovery required")
            rows.append({"id": identifier, "dataset": dataset, "family": family, "seed": seed,
                         "stored_prediction": stored,
                         "sources": sources_for(dataset, seed if seed is not None else SEEDS[0])})
    if len(rows) != 19 or len({row["id"] for row in rows}) != 19:
        raise ValueError(f"Exactly nineteen unique instances required for {dataset}")
    for role in ROLES:
        selected = [row for row in rows if row["family"] == role]
        expected = 1 if role == "f0" else 2
        if len(selected) != expected or (expected == 2 and {row["seed"] for row in selected} != set(SEEDS)):
            raise ValueError(f"Missing planned paired role: {dataset}/{role}")
    return rows


def seal_evaluation(job: Job) -> dict[str, Any]:
    destination = HERE / "selection_seal.json"
    if destination.exists() or (HERE / "test_exposure.json").exists():
        raise FileExistsError("Do not replace an existing selection/exposure seal")
    selection = load_complete_selection()
    gamma = read_json(HERE / "gamma_selection.json")
    verification = read_json(HERE / "gamma_verification.json")
    reuse = read_json(HERE / "reuse_manifest.json")
    if gamma["status"] != "complete" or len(gamma["rows"]) != 12 or not verification["pass"]:
        raise ValueError("All twelve Gamma fits and parity checks must precede TEST")
    check_receipt(gamma["selected"])
    check_receipt(verification["gamma_selection"])
    datasets = {}
    for dataset in DATASETS:
        rows = evaluation_rows(selection, reuse, gamma, dataset)
        contract = read_json(HERE / f"data_contract_{dataset}.json")
        check_receipt(contract["test"])
        datasets[dataset] = {"rows": rows, "data_contract": receipt(HERE / f"data_contract_{dataset}.json"),
                             "test_arrays": contract["test"], "block_length_origins": 42 if dataset == "jena" else 7}
    output = {"schema": "v11_joint_selection_seal_v1", "datasets": datasets,
              "selected_sha256": digest(HERE / "selected.json"),
              "gamma_selection_sha256": digest(HERE / "gamma_selection.json"),
              "reuse_manifest_sha256": digest(HERE / "reuse_manifest.json"),
              "protocol_sha256": digest(HERE / "protocol.json"),
              "gamma_verification": receipt(HERE / "gamma_verification.json"),
              "evaluation_source": receipt(Path(__file__)), "created_utc": time.time(),
              "evaluation_scope": "All three datasets are previously exposed development evaluations",
              "test_used_for_selection": False, "new_independent_confirmation": False,
              "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                            "boundary_rule": "sample blocks separately inside each fixed period"}}
    save_json(destination, output)
    job.heartbeat({"phase": "joint_selection_sealed", "instances": 57})
    return output


def ensure_exposure() -> dict[str, Any]:
    seal = read_json(HERE / "selection_seal.json")
    for name in ("selected", "gamma_selection", "reuse_manifest", "protocol"):
        if seal[name + "_sha256"] != digest(HERE / f"{name}.json"):
            raise ValueError(f"Sealed selection changed: {name}")
    check_receipt(seal["gamma_verification"])
    destination = HERE / "test_exposure.json"
    if destination.exists():
        if read_json(destination)["seal_sha256"] != digest(HERE / "selection_seal.json"):
            raise ValueError("Prior TEST exposure belongs to another selection seal")
    else:
        save_json(destination, {"schema": "v11_development_exposure_v1", "first_access_utc": time.time(),
                  "seal_sha256": digest(HERE / "selection_seal.json"), "datasets": list(DATASETS),
                  "scope": "Repeated development evaluation; this record does not restore independence"})
    return seal


def build_row_model(row: dict[str, Any], selection: dict[str, Any], basis: np.ndarray, device="cuda"):
    from model_v11 import build_model, restore_model, restore_reused_direct
    family = row["family"]
    if family in ("b_fixed_u", "b_fixed_gamma", "b_learned_gamma"):
        kind = "learned" if family == "b_learned_gamma" else "fixed"
        model, _ = model_for_u(selection, row["dataset"], row["seed"], kind, basis, device)
        if "gamma" in row:
            model.set_gamma(row["gamma"])
        return model.eval()
    if "checkpoint" in row:
        item = row["checkpoint"]
        check_receipt(item)
        if family == "direct_nlinear" and row["dataset"] in ("robin", "jena"):
            return restore_reused_direct(item["path"], device=device).eval()
        return restore_model(item["path"], device=device).eval()
    return build_model({"dataset": row["dataset"], "seed": row["seed"] or SEEDS[0], "family": family},
                       basis, device=device, sources=row["sources"]).eval()


def read_prediction(item: dict[str, Any], origins: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    with np.load(check_receipt(item), allow_pickle=False) as stored:
        prediction = stored["prediction"].copy()
        saved_origins = stored["origins"]
        if not np.array_equal(saved_origins, origins):
            raise ValueError("Stored prediction origin order differs from fixed evaluation")
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise ValueError("Stored prediction shape or finiteness invalid")
    return prediction


def predict_all(job: Job) -> dict[str, Any]:
    import torch
    seal = ensure_exposure()
    selection = load_complete_selection()
    destination = HERE / "prediction_manifest.json"
    if destination.exists():
        raise FileExistsError("Predictions already complete; reuse them in score rather than rerun")
    folder = CACHE / "evaluation01_predictions"
    folder.mkdir(parents=True, exist_ok=True)
    output = {"schema": "v11_prediction_manifest_v1", "status": "predicting",
              "seal": receipt(HERE / "selection_seal.json"), "datasets": {}}
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data[f"{period}_origins"] for period in PERIODS]).astype(np.int64)
        shape = (len(origins), 48, data["x"].shape[1])
        rows = []
        for row in seal["datasets"][dataset]["rows"]:
            job.check_limits()
            record = dict(row)
            if "stored_prediction" in row:
                prediction = read_prediction(row["stored_prediction"], origins, shape)
                record.update(prediction=row["stored_prediction"], reused=True)
                del prediction
            else:
                path = folder / f"{row['id']}.npz"
                receipt_path = folder / f"{row['id']}.receipt.json"
                if path.exists() or receipt_path.exists():
                    if not path.exists() or not receipt_path.exists():
                        raise ValueError(f"Partial prediction requires explicit recovery: {path}")
                    previous = read_json(receipt_path)
                    if previous["selection_seal_sha256"] != output["seal"]["sha256"]:
                        raise ValueError("Partial prediction belongs to a different selection")
                    read_prediction(previous["prediction"], origins, shape)
                    record.update(previous)
                else:
                    model = build_row_model(row, selection, data["basis"])
                    outputs, main_outputs = [], []
                    with torch.inference_mode():
                        for _, x, y, mask in batches(data, origins, batch_size=4, device="cuda"):
                            job.check_limits()
                            if row["family"] == "a_p_lora_qfixed":
                                main, q = model.components(x)
                                prediction = main + q
                                main_outputs.append(main.float().cpu().numpy())
                                del main, q
                            else:
                                prediction = model(x)
                            outputs.append(prediction.float().cpu().numpy())
                            del prediction, x, y, mask
                        extra = {}
                        if row["family"].startswith("b_"):
                            extra["U"] = model.U.detach().float().cpu().numpy().copy()
                    prediction = np.concatenate(outputs).astype(np.float32)
                    if prediction.shape != shape or not np.isfinite(prediction).all():
                        raise ValueError(f"Invalid new prediction: {row['id']}")
                    if main_outputs:
                        extra["main_only_prediction"] = np.concatenate(main_outputs).astype(np.float32)
                    np.savez_compressed(path, prediction=prediction, origins=origins, **extra)
                    record.update(prediction=receipt(path), reused=False,
                                  selection_seal_sha256=output["seal"]["sha256"])
                    save_json(receipt_path, record)
                    del model, prediction, outputs, main_outputs, extra
                    release_gpu()
            rows.append(record)
            output["datasets"][dataset] = {"models": rows, "shape": list(shape),
                                           "test_arrays": receipt(data["_path"])}
            save_json(HERE / "prediction_manifest_partial.json", output)
            job.heartbeat({"phase": "test_predictions", "dataset": dataset, "completed": len(rows), "total": 19,
                           "last_id": row["id"], "reused": record.get("reused")})
        del data
    output["status"] = "complete"
    save_json(destination, output)
    return output


def error_terms(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask axes differ")
    mask = mask.astype(bool)
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    delta = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {"squared": np.sum(delta * delta, axis=1), "absolute": np.sum(np.abs(delta), axis=1),
            "signed": delta.sum(axis=1), "count": mask.sum(axis=1)}


def aggregate_terms(terms: dict[str, np.ndarray], index: np.ndarray) -> dict[str, Any]:
    counts = terms["count"][index].sum(axis=0)
    sums = {key: terms[key][index].sum(axis=0) for key in ("squared", "absolute", "signed")}
    if np.any(counts == 0):
        raise ValueError("A fixed evaluation channel has no observed target; do not silently drop it")
    values = {key: value / counts for key, value in sums.items()}
    return {"mse": float(values["squared"].mean()), "mae": float(values["absolute"].mean()),
            "signed_mean_error": float(values["signed"].mean()),
            "channel_mse": values["squared"].tolist(), "channel_mae": values["absolute"].tolist(),
            "channel_bias": values["signed"].tolist(), "target_counts": counts.astype(int).tolist(),
            "squared_error_sums": sums["squared"].tolist(), "absolute_error_sums": sums["absolute"].tolist(),
            "signed_error_sums": sums["signed"].tolist(), "origins": int(len(index))}


def space_diagnostic(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                     U: np.ndarray, index: np.ndarray) -> dict[str, Any]:
    observed = mask[index].astype(bool).all(axis=2)
    coverage = int(observed.sum())
    total = int(observed.size)
    if not coverage:
        return {"status": "unavailable_no_complete_target_vector", "complete_rows": 0,
                "total_rows": total, "coverage": 0.0}
    error = prediction[index][observed].astype(np.float64) - target[index][observed].astype(np.float64)
    basis = U.astype(np.float64)
    retained = (error @ basis) @ basis.T
    residual = error - retained
    total_energy = np.sum(error * error, axis=1)
    p_energy, q_energy = np.sum(retained * retained, axis=1), np.sum(residual * residual, axis=1)
    cross = 2.0 * np.sum(retained * residual, axis=1)
    channels = U.shape[0]
    return {"status": "available", "scope": "complete observed target vectors in standardized coordinates",
            "complete_rows": coverage, "total_rows": total, "coverage": coverage / total,
            "p_mse": float(p_energy.sum() / (coverage * channels)),
            "q_mse": float(q_energy.sum() / (coverage * channels)),
            "complete_vector_mse": float(total_energy.sum() / (coverage * channels)),
            "orthogonal_energy_sum_gap": float(np.max(np.abs(total_energy - p_energy - q_energy))),
            "max_abs_cross_term": float(np.max(np.abs(cross))),
            "gram_max_abs": float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))),
            "channel_p_mse": np.mean(retained * retained, axis=0).tolist(),
            "channel_q_mse": np.mean(residual * residual, axis=0).tolist(),
            "warning": "Not the original masked channel-macro MSE; missing targets were not zero-filled"}


def basis_diagnostic(U: np.ndarray, U0: np.ndarray) -> dict[str, Any]:
    u, u0 = U.astype(np.float64), U0.astype(np.float64)
    singular = np.clip(np.linalg.svd(u0.T @ u, compute_uv=False), 0.0, 1.0)
    return {"principal_angles_degrees": np.rad2deg(np.arccos(singular)).tolist(),
            "projector_frobenius_distance": float(np.linalg.norm(u @ u.T - u0 @ u0.T)),
            "basis_frobenius_distance": float(np.linalg.norm(u - u0)),
            "overlap_matrix": (u0.T @ u).tolist(),
            "interpretation": "Descriptive only; rotation/sign adaptation and subspace changes are not separated causally"}


def bootstrap_weights(period_indices: dict[str, np.ndarray], block: int, dataset: str) -> dict[str, np.ndarray]:
    total = sum(len(period_indices[period]) for period in PERIODS)
    weights = {}
    dataset_salt = DATASETS.index(dataset)
    for period_index, period in enumerate(PERIODS):
        index = period_indices[period]
        n = len(index)
        if n < block:
            raise ValueError("Insufficient period length for the prespecified block interval")
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, dataset_salt, period_index]))
        matrix = np.zeros((BOOTSTRAP_DRAWS, total), dtype=np.int32)
        for draw in range(BOOTSTRAP_DRAWS):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            matrix[draw] = np.bincount(index[local], minlength=total)
        weights[period] = matrix
    weights["combined"] = weights["test_a"] + weights["test_b"]
    return weights


def comparison_pairs() -> list[tuple[str, str]]:
    candidates = ("a_p_lora_qfixed", "b_fixed_u", "b_learned_u", "b_fixed_gamma", "b_learned_gamma")
    external = ("f0", "full_lora_mse", "lora_native", "direct_nlinear")
    pairs = [(left, right) for left in candidates for right in external]
    pairs += [("a_p_lora_qfixed", "base_level"), ("b_learned_u", "b_fixed_u"),
              ("b_fixed_gamma", "b_fixed_u"), ("b_learned_gamma", "b_learned_u"),
              ("b_learned_gamma", "b_fixed_gamma"), ("a_p_lora_qfixed", "b_learned_u"),
              ("a_p_lora_qfixed", "b_learned_gamma")]
    return list(dict.fromkeys(pairs))


def summarize_roles(rows: list[dict[str, Any]], terms: dict[str, dict[str, np.ndarray]],
                    indices: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    summaries, averaged_terms = {}, {}
    for role in ROLES:
        members = [row for row in rows if row["family"] == role]
        expected_count = terms[members[0]["id"]]["count"]
        if any(not np.array_equal(terms[row["id"]]["count"], expected_count) for row in members):
            raise ValueError("Paired seeds must use identical observed targets")
        averaged = {key: np.mean([terms[row["id"]][key] for row in members], axis=0)
                    for key in ("squared", "absolute", "signed")}
        averaged["count"] = expected_count
        averaged_terms[role] = averaged
        summaries[role] = {"ids": [row["id"] for row in members],
                           "aggregation": "mean of seed losses, never prediction ensemble",
                           "periods": {period: aggregate_terms(averaged, index) for period, index in indices.items()}}
    return summaries, averaged_terms


def paired_comparisons(dataset: str, role_terms: dict[str, dict[str, np.ndarray]],
                       indices: dict[str, np.ndarray], job: Job) -> dict[str, Any]:
    block = 42 if dataset == "jena" else 7
    weights = bootstrap_weights(indices, block, dataset)
    draws = {}
    for period, matrix in weights.items():
        denominator = matrix @ role_terms["f0"]["count"]
        if np.any(denominator == 0):
            raise ValueError("Bootstrap draw has an unobserved fixed channel; do not drop it")
        draws[period] = {}
        for role, terms in role_terms.items():
            job.check_limits()
            draws[period][role] = {
                "mse": np.mean((matrix @ terms["squared"]) / denominator, axis=1),
                "mae": np.mean((matrix @ terms["absolute"]) / denominator, axis=1)}
    output = {}
    for left, right in comparison_pairs():
        item = {"candidate": left, "reference": right, "direction": "candidate minus reference; negative favors candidate",
                "periods": {}}
        for period, index in indices.items():
            candidate = aggregate_terms(role_terms[left], index)
            reference = aggregate_terms(role_terms[right], index)
            metrics = {}
            for metric in ("mse", "mae"):
                delta = candidate[metric] - reference[metric]
                a, b = draws[period][left][metric], draws[period][right][metric]
                relative = 100.0 * delta / reference[metric] if reference[metric] > 0 else None
                relative_samples = 100.0 * (a - b) / b if np.all(b > 0) else None
                metrics[metric] = {"candidate": candidate[metric], "reference": reference[metric],
                    "absolute_difference": delta, "relative_change_percent": relative,
                    "conditional_block95_absolute": np.quantile(a - b, [.025, .975]).tolist(),
                    "conditional_block95_relative_percent": None if relative_samples is None else np.quantile(relative_samples, [.025, .975]).tolist()}
            item["periods"][period] = metrics
        output[f"{left}__vs__{right}"] = item
    return {"comparisons": output, "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "block_origins": block, "paired_channels_methods_and_seeds": True,
            "period_boundaries_crossed": False, "selection_uncertainty_included": False,
            "scope": "Conditional on the two selected parent/model seeds; repeated VAL selection is not covered"}}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_all(job: Job) -> dict[str, Any]:
    seal = ensure_exposure()
    manifest = read_json(HERE / "prediction_manifest.json")
    check_receipt(manifest["seal"])
    if manifest["status"] != "complete" or set(manifest["datasets"]) != set(DATASETS):
        raise ValueError("All planned predictions must be complete before joint reporting")
    destination = HERE / "evaluation01.json"
    if destination.exists():
        raise FileExistsError("Preserve prior evaluation; a correction requires an explicit new artifact")
    output = {"schema": "v11_joint_development_evaluation_v1", "status": "scoring",
              "selection_seal": receipt(HERE / "selection_seal.json"),
              "prediction_manifest": receipt(HERE / "prediction_manifest.json"), "datasets": {},
              "evidence_status": "confirmed_from_saved_arrays", "independent_confirmation": False,
              "primary_metric": "observed channel-macro MSE in TRAIN-standardized coordinates"}
    scalar_rows, channel_rows = [], []
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        periods = [np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS]
        origins = np.concatenate(periods)
        target = np.stack([data["x"][origin:origin + 48] for origin in origins])
        mask = np.stack([data["finite"][origin:origin + 48] for origin in origins]).astype(bool)
        indices = {"test_a": np.arange(len(periods[0])),
                   "test_b": np.arange(len(periods[0]), len(origins)), "combined": np.arange(len(origins))}
        rows = manifest["datasets"][dataset]["models"]
        if [row["id"] for row in rows] != [row["id"] for row in seal["datasets"][dataset]["rows"]]:
            raise ValueError("Prediction role set/order differs from the prespecified seal")
        terms, models, q_arrays = {}, {}, {}
        for row in rows:
            prediction = read_prediction(row["prediction"], origins, target.shape)
            identifier = row["id"]
            terms[identifier] = error_terms(prediction, target, mask)
            periods_scores = {label: aggregate_terms(terms[identifier], index) for label, index in indices.items()}
            model_result = {**row, "periods": periods_scores,
                            "space_common_U0": {label: space_diagnostic(prediction, target, mask, data["basis"], index)
                                                for label, index in indices.items()}}
            if row["family"].startswith("b_"):
                with np.load(check_receipt(row["prediction"]), allow_pickle=False) as stored:
                    own_u = stored["U"].copy()
                model_result["space_own_U"] = {label: space_diagnostic(prediction, target, mask, own_u, index)
                                               for label, index in indices.items()}
                model_result["U_diagnostic"] = basis_diagnostic(own_u, data["basis"])
            if row["family"] in ("a_p_lora_qfixed", "base_level"):
                u = data["basis"].astype(np.float64)
                value = prediction.astype(np.float64)
                q_arrays[(row["family"], row["seed"])] = value - (value @ u) @ u.T
            if row["family"] == "a_p_lora_qfixed":
                with np.load(check_receipt(row["prediction"]), allow_pickle=False) as stored:
                    main = stored["main_only_prediction"].copy()
                main_terms = error_terms(main, target, mask)
                model_result["main_only_fit0_ablation"] = {"scope": "Remove Q from the same trained A checkpoint; no separate fitting",
                    "periods": {label: aggregate_terms(main_terms, index) for label, index in indices.items()}}
            models[identifier] = model_result
            for period, score in periods_scores.items():
                scalar_rows.append({"dataset": dataset, "family": row["family"], "id": identifier,
                    "seed": row["seed"], "period": period, "mse": score["mse"], "mae": score["mae"],
                    "signed_mean_error": score["signed_mean_error"], "reused_prediction": row["reused"],
                    "evidence_status": "confirmed"})
                for channel, column in enumerate(data["columns"]):
                    channel_rows.append({"dataset": dataset, "family": row["family"], "id": identifier,
                        "seed": row["seed"], "period": period, "channel": str(column),
                        "mse": score["channel_mse"][channel], "mae": score["channel_mae"][channel],
                        "signed_mean_error": score["channel_bias"][channel], "observed_count": score["target_counts"][channel]})
            del prediction
        summaries, role_terms = summarize_roles(rows, terms, indices)
        comparisons = paired_comparisons(dataset, role_terms, indices, job)
        q_parity = {str(seed): parity(q_arrays[("a_p_lora_qfixed", seed)], q_arrays[("base_level", seed)]) for seed in SEEDS}
        term_path = CACHE / f"evaluation01_{dataset}_origin_terms.npz"
        np.savez_compressed(term_path, origins=origins,
                            **{f"{identifier}__{key}": value for identifier, item in terms.items() for key, value in item.items()})
        output["datasets"][dataset] = {"models": models, "role_summary": summaries, **comparisons,
            "A_Q_projection_parity": q_parity, "origin_terms": receipt(term_path),
            "data_arrays": receipt(data["_path"]), "columns": [str(value) for value in data["columns"]],
            "period_origin_counts": {period: len(index) for period, index in indices.items()},
            "aggregation": "Pool period SSE/count by channel, macro across channels, mean seed losses"}
        save_json(HERE / "evaluation01_partial.json", output)
        job.heartbeat({"phase": "joint_development_scoring", "dataset": dataset, "completed": len(output["datasets"]), "total": 3})
        del data, target, mask, terms, role_terms, q_arrays
        gc.collect()
    write_csv(HERE / "evaluation01_models.csv", scalar_rows)
    write_csv(HERE / "evaluation01_channels.csv", channel_rows)
    output["status"] = "complete"
    output["tables"] = {"models": receipt(HERE / "evaluation01_models.csv"),
                        "channels": receipt(HERE / "evaluation01_channels.csv")}
    save_json(destination, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seal", "predict", "score"))
    parser.add_argument("--reserve-seconds", type=float, required=True)
    parser.add_argument("--name", help="Fresh ledger job label for an explicitly documented retry")
    args = parser.parse_args()
    configure()
    category = "gpu" if args.action == "predict" else ("cpu_analysis" if args.action == "score" else "cpu_check")
    with Job(args.name or f"evaluation_{args.action}", category=category, reserve_seconds=args.reserve_seconds,
             metadata={"action": args.action, "datasets": list(DATASETS), "fits": 0}) as job:
        {"seal": seal_evaluation, "predict": predict_all, "score": score_all}[args.action](job)


if __name__ == "__main__":
    main()
