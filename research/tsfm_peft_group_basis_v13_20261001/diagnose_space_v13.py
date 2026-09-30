"""Post-hoc P/Q error energy diagnosis for v13 stored predictions.

This diagnostic uses the fixed v13 group basis U for every stored method.  It
does not create predictions, fit models, tune mixtures, or impute missing target
vectors.  Only complete target vectors are projected.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from runtime_v13 import HERE, SEEDS, Job, artifact, digest, load_data, protocol, read_json, save_json


SCHEMA = "v13_space_diagnosis_1"
OUTPUT = HERE / "space_diagnosis01.json"
ROLES = ("group_res", "group_raw", "pca_level", "a", "f0", "full_mse", "native", "direct")
PERIODS = ("test_a", "test_b", "combined")


def role_label(role: str) -> str:
    return role.upper()


def require_complete_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p = protocol()
    evaluation = read_json(HERE / "evaluation01.json")
    prediction_manifest = read_json(HERE / "prediction_manifest.json")
    if evaluation.get("status") != "complete":
        raise RuntimeError("evaluation01.json is not complete")
    if prediction_manifest.get("status") != "complete":
        raise RuntimeError("prediction_manifest.json is not complete")
    if set(evaluation["datasets"]) != set(p["units"]):
        raise RuntimeError("evaluation01.json does not cover all fixed datasets")
    if set(prediction_manifest["datasets"]) != set(p["units"]):
        raise RuntimeError("prediction_manifest.json does not cover all fixed datasets")
    return p, evaluation, prediction_manifest


def source_receipts() -> dict[str, Any]:
    return {
        "protocol": artifact(HERE / "protocol.json"),
        "data_manifest": artifact(HERE / "data_manifest.json"),
        "reuse_manifest": artifact(HERE / "reuse_manifest.json"),
        "selection_seal": artifact(HERE / "selection_seal.json"),
        "test_exposure": artifact(HERE / "test_exposure.json"),
        "prediction_manifest": artifact(HERE / "prediction_manifest.json"),
        "evaluation01": artifact(HERE / "evaluation01.json"),
        "diagnostic_code_sha256": digest(HERE / "diagnose_space_v13.py"),
    }


def period_indices(test_a: np.ndarray, test_b: np.ndarray) -> dict[str, np.ndarray]:
    a = np.arange(len(test_a), dtype=np.int64)
    b = np.arange(len(test_a), len(test_a) + len(test_b), dtype=np.int64)
    return {"test_a": a, "test_b": b, "combined": np.concatenate([a, b])}


def complete_coverage(complete: np.ndarray, indices: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    horizon = complete.shape[1]
    for period, index in indices.items():
        total = int(len(index) * horizon)
        count = int(complete[index].sum())
        result[period] = {
            "complete_target_vectors": count,
            "total_target_vectors": total,
            "fraction": count / total if total else None,
            "mask_rule": "finite target mask.all(axis=-1); shared by every method in this dataset/period",
        }
    return result


def validate_group_basis(u: np.ndarray, channels: int, latent: int) -> dict[str, Any]:
    if u.shape != (channels, latent):
        raise ValueError("Group basis shape differs from the protocol")
    gram = u.T @ u
    gram_error = float(np.max(np.abs(gram - np.eye(latent, dtype=gram.dtype))))
    if gram_error > 1e-5:
        raise ValueError("Group basis is not orthonormal enough for diagnosis")
    if (u < 0).any() or not np.all((u > 0).sum(axis=1) == 1):
        raise ValueError("Diagnostic basis is not a disjoint nonnegative group basis")
    groups = []
    for j in range(latent):
        groups.append(np.nonzero(u[:, j] > 0)[0].astype(int).tolist())
    return {"diagnostic_sha256": hashlib_array(u), "diagnostic_dtype": str(u.dtype),
            "gram_max_abs_error": gram_error, "groups": groups}


def hashlib_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    import hashlib

    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def prediction_array(receipt: dict[str, Any], origins: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    artifact(receipt["path"], receipt["sha256"])
    with np.load(receipt["path"], allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError("Prediction origins differ from the stored v13 evaluation order")
        prediction = archive["prediction"].astype(np.float64, copy=True)
    if prediction.shape != shape:
        raise ValueError(f"Prediction shape differs: {prediction.shape} != {shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")
    return prediction


def role_rows(prediction_manifest: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(row) for row in prediction_manifest["datasets"][dataset]["models"] if row["family"] in ROLES]
    expected = {"f0": [None]}
    for role in ROLES:
        if role != "f0":
            expected[role] = list(SEEDS)
    for role, seeds in expected.items():
        actual = sorted((row.get("seed") for row in rows if row["family"] == role), key=str)
        if actual != sorted(seeds, key=str):
            raise RuntimeError(f"{dataset} missing role/seed rows for {role}: {actual}")
    if len(rows) != 15:
        raise RuntimeError(f"{dataset} diagnostic expects 15 rows including group_raw, got {len(rows)}")
    return sorted(rows, key=lambda row: (ROLES.index(row["family"]), str(row.get("seed")), row["id"]))


def original_masked_macro(evaluation: dict[str, Any], dataset: str, model_id: str, period: str) -> dict[str, Any]:
    periods = evaluation["datasets"][dataset]["models"][model_id]["periods"]
    values = periods[period]
    return {
        "source": "evaluation01.json",
        "model_id": model_id,
        "period": period,
        "mse": values["mse"],
        "mae": values["mae"],
        "signed_mean_error": values["signed_mean_error"],
        "metric_note": "original masked channel-macro values; separate from complete-vector P/Q energies",
    }


def energy_terms(
    prediction: np.ndarray,
    target: np.ndarray,
    complete: np.ndarray,
    indices: np.ndarray,
    u: np.ndarray,
) -> dict[str, Any]:
    selector = complete[indices]
    count = int(selector.sum())
    total = int(selector.size)
    channels = int(target.shape[-1])
    if count == 0:
        return {
            "complete_target_vectors": 0,
            "total_target_vectors": total,
            "coverage_fraction": 0.0 if total else None,
            "full_energy_mse_per_C": None,
            "p_energy_mse_per_C": None,
            "q_energy_mse_per_C": None,
            "p_plus_q_numerical_residual": None,
        }
    error = prediction[indices][selector] - target[indices][selector]
    coefficients = error @ u
    projected = coefficients @ u.T
    residual = error - projected
    full_energy = np.sum(error * error, axis=1) / channels
    p_energy = np.sum(coefficients * coefficients, axis=1) / channels
    q_energy = np.sum(residual * residual, axis=1) / channels
    full = float(np.mean(full_energy))
    p = float(np.mean(p_energy))
    q = float(np.mean(q_energy))
    return {
        "complete_target_vectors": count,
        "total_target_vectors": total,
        "coverage_fraction": count / total if total else None,
        "full_energy_mse_per_C": full,
        "p_energy_mse_per_C": p,
        "q_energy_mse_per_C": q,
        "p_fraction_of_full": p / full if full > 0 else None,
        "q_fraction_of_full": q / full if full > 0 else None,
        "p_plus_q_numerical_residual": full - p - q,
        "vector_space": "common fixed v13 group basis U for every method",
        "missing_policy": "complete target vectors only; no zero-fill for projection",
    }


def diagnose_dataset(
    dataset: str,
    unit: dict[str, Any],
    evaluation: dict[str, Any],
    prediction_manifest: dict[str, Any],
    job: Job,
) -> dict[str, Any]:
    data = load_data(dataset, include_test=True)
    columns = [str(value) for value in data["columns"]]
    origins = np.concatenate([data["test_a_origins"], data["test_b_origins"]]).astype(np.int64)
    indices = period_indices(data["test_a_origins"], data["test_b_origins"])
    target = np.stack([data["x"][origin : origin + 48] for origin in origins]).astype(np.float64)
    mask = np.stack([data["finite"][origin : origin + 48] for origin in origins]).astype(bool)
    complete = mask.all(axis=-1)
    shape = target.shape
    u = data["basis"].astype(np.float64)
    basis_receipt = validate_group_basis(u, unit["C"], unit["K"])
    basis_receipt["stored_dtype"] = str(data["basis"].dtype)
    basis_receipt["stored_sha256"] = hashlib_array(data["basis"])
    models: dict[str, Any] = {}
    max_abs_residual = 0.0
    for row in role_rows(prediction_manifest, dataset):
        job.check_limits()
        prediction = prediction_array(row["prediction"], origins, shape)
        periods = {}
        for period, index in indices.items():
            values = energy_terms(prediction, target, complete, index, u)
            residual = values["p_plus_q_numerical_residual"]
            if residual is not None:
                max_abs_residual = max(max_abs_residual, abs(float(residual)))
            values["original_masked_macro"] = original_masked_macro(evaluation, dataset, row["id"], period)
            periods[period] = values
        models[row["id"]] = {
            "id": row["id"],
            "family": row["family"],
            "label": role_label(row["family"]),
            "seed": row.get("seed"),
            "reused": row.get("reused"),
            "prediction": row["prediction"],
            "periods": periods,
        }
    return {
        "C": unit["C"],
        "K": unit["K"],
        "columns": columns,
        "data": artifact(data["_path"]),
        "basis": basis_receipt,
        "complete_vector_coverage": complete_coverage(complete, indices),
        "models": models,
        "max_abs_p_plus_q_numerical_residual": max_abs_residual,
    }


def run(job: Job) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError("Preserve existing space_diagnosis01.json")
    p, evaluation, prediction_manifest = require_complete_sources()
    result = {
        "schema": SCHEMA,
        "status": "partial",
        "created_utc": time.time(),
        "scope": (
            "Post-hoc exposed-data diagnostic only; all four v13 datasets; "
            "not causal proof and not a new predictor."
        ),
        "method": {
            "roles": list(ROLES),
            "periods": list(PERIODS),
            "projection_basis": "the fixed v13 GROUP U common to every stored method",
            "energy": "mean complete-target-vector Euclidean squared error divided by C",
            "subset": "target finite mask.all(axis=-1), identical across methods in each dataset/period",
            "no_zero_fill": True,
            "no_new_prediction_or_fit": True,
            "diagnostic_question": "whether remaining loss plausibly concentrates in group-space main P or orthogonal Q",
        },
        "sources": source_receipts(),
        "datasets": {},
    }
    max_residual = 0.0
    for dataset, unit in p["units"].items():
        result["datasets"][dataset] = diagnose_dataset(dataset, unit, evaluation, prediction_manifest, job)
        max_residual = max(max_residual, result["datasets"][dataset]["max_abs_p_plus_q_numerical_residual"])
        job.heartbeat(dataset + " P/Q space diagnosis complete")
    result["status"] = "complete"
    result["max_abs_p_plus_q_numerical_residual"] = max_residual
    save_json(OUTPUT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--reserve-s", type=float, required=True)
    args = parser.parse_args()
    metadata = {
        "purpose": "v13 post-hoc P/Q complete-vector space diagnosis",
        "fit_count": 0,
        "output": "space_diagnosis01.json",
        "no_new_prediction_or_fit": True,
    }
    with Job("cpu_analysis", args.job, reserve_s=args.reserve_s, metadata=metadata) as job:
        run(job)


if __name__ == "__main__":
    main()
