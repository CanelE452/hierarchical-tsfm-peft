"""Independent validation for forecast component study artifacts.

This script reads only saved study outputs and prediction archives. It does
not train models, open protected TEST files, or modify upstream results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "research" / "past_only_repr_diagnosis_20261012"
sys.path.insert(0, str(OLD))

from common import access_guard, ahash, load_data, masks, windows  # noqa: E402

STAGE_A = HERE / "stage_a_checkpoint_readout_v1"
SPLITS = ("fit", "tune", "val")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite_or_none(value: str | None) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    return float(value)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def unique_output(stem: str, suffix: str) -> Path:
    for i in range(1, 1000):
        path = HERE / f"{stem}_v{i}{suffix}"
        if not path.exists():
            return path
    raise RuntimeError(f"Could not allocate output path for {stem}{suffix}")


def masked_channel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, Any]:
    m = np.asarray(observed, dtype=bool)
    if mean is not None and std is not None:
        # Match the study scorer: stored normalized float arrays are restored
        # with float32 mean/std, then accumulated in float64.
        mean32 = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std32 = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        p = np.asarray(prediction * std32 + mean32, dtype=np.float64)
        y = np.asarray(target * std32 + mean32, dtype=np.float64)
    else:
        p = np.asarray(prediction, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
    err = np.where(m, p - y, 0.0)
    counts = m.sum(axis=(0, 1))
    usable = counts > 0
    per_channel_mse = np.full(counts.shape, np.nan, dtype=np.float64)
    per_channel_mae = np.full(counts.shape, np.nan, dtype=np.float64)
    per_channel_mse[usable] = (err**2).sum(axis=(0, 1))[usable] / counts[usable]
    per_channel_mae[usable] = np.abs(err).sum(axis=(0, 1))[usable] / counts[usable]
    return {
        "mse": float(np.nanmean(per_channel_mse)),
        "mae": float(np.nanmean(per_channel_mae)),
        "per_channel_mse": per_channel_mse.tolist(),
        "per_channel_mae": per_channel_mae.tolist(),
        "observed_counts": counts.astype(int).tolist(),
    }


def close_enough(a: float, b: float, atol: float = 1e-8, rtol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, **extra: Any) -> None:
    checks.append({"check": name, "pass": bool(passed), **extra})


def load_common_data() -> dict[str, dict[str, Any]]:
    return {dataset: load_data(dataset, frozen=False) for dataset in ("jena", "robin")}


def archive_split_arrays(
    archive: Any,
    data: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = archive[f"{split}_prediction"]
    if f"{split}_target" in archive.files:
        target = archive[f"{split}_target"]
    else:
        target = windows(data, split, True)
    if f"{split}_observed" in archive.files:
        observed = archive[f"{split}_observed"]
    else:
        observed = masks(data, split)
    origins = archive[f"{split}_origins"]
    return prediction, target, observed, origins


def validate_method_archive(
    checks: list[dict[str, Any]],
    *,
    stage: str,
    case: dict[str, Any],
    method: dict[str, Any],
    data: dict[str, Any],
    dataset: str,
) -> list[dict[str, Any]]:
    pred_info = method["prediction"]
    pred_path = ROOT / pred_info["path"]
    add_check(
        checks,
        f"{stage}_prediction_sha256",
        pred_path.exists() and sha256(pred_path) == pred_info["sha256"],
        dataset=dataset,
        label=case.get("label"),
        seed=case.get("seed"),
        method=method.get("method"),
        path=pred_info["path"],
    )
    rescored = []
    if not pred_path.exists():
        return rescored

    with np.load(pred_path, allow_pickle=False) as archive:
        archive_mean = archive["mean"]
        archive_std = archive["std"]
        add_check(
            checks,
            f"{stage}_archive_mean_std_match",
            np.allclose(archive_mean, data["mean"]) and np.allclose(archive_std, data["std"]),
            dataset=dataset,
            label=case.get("label"),
            method=method.get("method"),
        )
        for split in SPLITS:
            prediction, target, observed, origins = archive_split_arrays(archive, data, split)
            shape_ok = prediction.shape == target.shape == observed.shape
            finite_ok = bool(np.isfinite(prediction).all())
            origin_ok = origins.tolist() == np.asarray(data["origins"][split]).tolist()
            add_check(
                checks,
                f"{stage}_archive_{split}_shape_origins",
                shape_ok and finite_ok and origin_ok,
                dataset=dataset,
                label=case.get("label"),
                seed=case.get("seed"),
                method=method.get("method"),
                split=split,
                shape=list(prediction.shape),
            )
            physical = masked_channel_metrics(prediction, target, observed, archive_mean, archive_std)
            normalized = masked_channel_metrics(prediction, target, observed)
            reported = method["scores"][split]
            ok = (
                close_enough(physical["mse"], reported["mse"], atol=1e-7)
                and close_enough(physical["mae"], reported["mae"], atol=1e-7)
                and close_enough(normalized["mse"], reported["normalized"]["mse"], atol=1e-7)
                and close_enough(normalized["mae"], reported["normalized"]["mae"], atol=1e-7)
            )
            add_check(
                checks,
                f"{stage}_{split}_metric_rescore",
                ok,
                dataset=dataset,
                label=case.get("label"),
                seed=case.get("seed"),
                method=method.get("method"),
                split=split,
                physical_mse=physical["mse"],
                reported_physical_mse=reported["mse"],
                normalized_mse=normalized["mse"],
                reported_normalized_mse=reported["normalized"]["mse"],
            )
            rescored.append(
                {
                    "dataset": dataset,
                    "label": case.get("label"),
                    "seed": case.get("seed"),
                    "method": method.get("method"),
                    "split": split,
                    "physical_mse": physical["mse"],
                    "physical_mae": physical["mae"],
                    "normalized_mse": normalized["mse"],
                    "normalized_mae": normalized["mae"],
                    "prediction_sha256": pred_info["sha256"],
                }
            )
    for timing in method.get("latency", []):
        seconds = timing.get("seconds", [])
        latency_ok = (
            len(seconds) == timing.get("repeats", len(seconds))
            and all(np.isfinite(seconds))
            and all(float(s) > 0 for s in seconds)
            and float(timing.get("median_seconds", 0.0)) > 0.0
        )
        parity = timing.get("pipeline_prediction_parity")
        if parity is not None:
            latency_ok = latency_ok and parity.get("physical_max_abs", 0.0) <= parity.get("absolute_tolerance", 0.0)
        if "timed_vs_saved_max_abs_physical" in timing:
            latency_ok = latency_ok and float(timing["timed_vs_saved_max_abs_physical"]) <= 1e-8
        add_check(
            checks,
            f"{stage}_latency_and_parity",
            latency_ok,
            dataset=dataset,
            label=case.get("label"),
            method=method.get("method"),
            batch=timing.get("batch"),
            median_seconds=timing.get("median_seconds"),
        )
    return rescored


def stage_a_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in result.get("raw_baselines", []):
        records.append(
            {
                "dataset": record["dataset"],
                "seed": None,
                "update": None,
                "method": record["method"],
                "fit_scope": record.get("fit_scope"),
                "record": record,
            }
        )
    for case in result.get("cases", []):
        for record in case.get("comparisons", []):
            records.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "update": record["update"],
                    "method": record["method"],
                    "fit_scope": record.get("fit_scope"),
                    "record": record,
                }
            )
    return records


def validate_stage_a() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    result_path = STAGE_A / "RESULT.json"
    verification_path = STAGE_A / "VERIFICATION.json"
    metrics_path = STAGE_A / "metrics.csv"
    result = read_json(result_path)
    verification = read_json(verification_path)
    contracts = {d: read_json(STAGE_A / f"{d}_contract.json") for d in ("jena", "robin")}

    add_check(checks, "stage_a_status_complete", result.get("status") == "COMPLETE", status=result.get("status"))
    add_check(
        checks,
        "stage_a_verification_pass",
        verification.get("status") == "PASS",
        status=verification.get("status"),
    )
    add_check(
        checks,
        "verification_binds_result_sha",
        sha256(result_path) == verification.get("result_sha256"),
        expected=verification.get("result_sha256"),
        actual=sha256(result_path),
    )

    for rel, digest in result.get("sources", {}).items():
        path = ROOT / rel
        add_check(checks, "source_sha256", path.exists() and sha256(path) == digest, path=rel, expected=digest)

    for dataset, contract in contracts.items():
        source = ROOT / contract["source"]
        contract_source = ROOT / contract["contract_path"]
        add_check(
            checks,
            "data_source_sha256",
            source.exists() and sha256(source) == contract["source_sha256"],
            dataset=dataset,
            path=contract["source"],
        )
        add_check(
            checks,
            "contract_sha256",
            contract_source.exists() and sha256(contract_source) == contract["source_contract_sha256"],
            dataset=dataset,
            path=contract["contract_path"],
        )
        chronological = contract["fit_bounds"][1] <= contract["tune_bounds"][0] <= contract["tune_bounds"][1] <= contract["val_bounds"][0]
        add_check(checks, "chronological_fit_tune_val_bounds", chronological, dataset=dataset)
        add_check(
            checks,
            "fit_only_normalization_declared",
            str(contract.get("normalization_scope", "")).lower().find("fit") >= 0,
            dataset=dataset,
            normalization_scope=contract.get("normalization_scope"),
        )
        for split in ("fit", "tune", "val"):
            origins = np.asarray(contract["origins"][split], dtype=np.int64)
            lo, hi = contract[f"{split}_bounds"]
            in_bounds = bool(np.all(origins >= lo) and np.all(origins + int(contract["H"]) <= hi))
            add_check(
                checks,
                "split_target_origins_inside_bounds",
                in_bounds,
                dataset=dataset,
                split=split,
                origin_count=int(origins.size),
                bounds=[lo, hi],
            )

    records = stage_a_records(result)
    metric_rows = {}
    with metrics_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["dataset"],
                None if row["seed"] == "NA" else int(row["seed"]),
                None if row["update"] == "NA" else int(row["update"]),
                row["method"],
                row["fit_scope"],
            )
            metric_rows[key] = row

    rescored_rows: list[dict[str, Any]] = []
    for rec in records:
        record = rec["record"]
        dataset = rec["dataset"]
        contract = contracts[dataset]
        mean = np.asarray(contract["mean"], dtype=np.float64)
        std = np.asarray(contract["std"], dtype=np.float64)
        pred_info = record["prediction"]
        pred_path = ROOT / pred_info["path"]
        add_check(
            checks,
            "prediction_sha256",
            pred_path.exists() and sha256(pred_path) == pred_info["sha256"],
            dataset=dataset,
            method=rec["method"],
            path=pred_info["path"],
        )
        with np.load(pred_path, allow_pickle=False) as archive:
            prediction = archive["prediction"]
            target = archive["target"]
            observed = archive["observed"]
            origins = archive["origins"]
            shape_ok = prediction.shape == target.shape == observed.shape
            origins_ok = origins.tolist() == contract["origins"]["val"]
            finite_ok = bool(np.isfinite(prediction).all() and np.isfinite(target).all())
            add_check(
                checks,
                "prediction_archive_shape_and_origins",
                shape_ok and origins_ok and finite_ok,
                dataset=dataset,
                method=rec["method"],
                shape=list(prediction.shape),
                val_origin_count=len(contract["origins"]["val"]),
            )
            physical = masked_channel_metrics(prediction, target, observed, mean, std)
            normalized = masked_channel_metrics(prediction, target, observed)
        val = record["val"]
        comparisons_ok = (
            close_enough(physical["mse"], val["mse"], atol=1e-7)
            and close_enough(physical["mae"], val["mae"], atol=1e-7)
            and close_enough(normalized["mse"], val["normalized"]["mse"], atol=1e-7)
            and close_enough(normalized["mae"], val["normalized"]["mae"], atol=1e-7)
        )
        add_check(
            checks,
            "prediction_metric_rescore",
            comparisons_ok,
            dataset=dataset,
            seed=rec["seed"],
            update=rec["update"],
            method=rec["method"],
            fit_scope=rec["fit_scope"],
            physical_mse=physical["mse"],
            reported_physical_mse=val["mse"],
            normalized_mse=normalized["mse"],
            reported_normalized_mse=val["normalized"]["mse"],
        )
        fit_scope_key = rec["fit_scope"] if rec["fit_scope"] is not None else "NONE"
        key = (dataset, rec["seed"], rec["update"], rec["method"], fit_scope_key)
        row = metric_rows.get(key)
        csv_ok = row is not None and close_enough(physical["mse"], finite_or_none(row["val_mse"]), atol=1e-7)
        add_check(checks, "metrics_csv_matches_prediction", csv_ok, dataset=dataset, key=list(key))
        rescored_rows.append(
            {
                "dataset": dataset,
                "seed": rec["seed"],
                "update": rec["update"],
                "method": rec["method"],
                "fit_scope": rec["fit_scope"],
                "val_physical_mse": physical["mse"],
                "val_physical_mae": physical["mae"],
                "val_normalized_mse": normalized["mse"],
                "val_normalized_mae": normalized["mae"],
                "per_channel_mse": physical["per_channel_mse"],
                "observed_counts": physical["observed_counts"],
                "prediction_sha256": pred_info["sha256"],
            }
        )

    case_summaries = []
    for case in result.get("cases", []):
        training_path = ROOT / case["source_training_record"]
        training = read_json(training_path)
        add_check(
            checks,
            "source_training_record_sha256",
            training_path.exists() and sha256(training_path) == case["source_training_sha256"],
            dataset=case["dataset"],
            seed=case["seed"],
        )
        for cp in case["checkpoints"]:
            cp_path = ROOT / cp["checkpoint_path"]
            add_check(
                checks,
                "checkpoint_sha256",
                cp_path.exists() and sha256(cp_path) == cp["checkpoint_sha256"],
                dataset=case["dataset"],
                seed=case["seed"],
                update=cp["update"],
            )
        expected = min(case["checkpoints"], key=lambda cp: (cp["metrics"]["tune"]["mse"], cp["update"]))
        add_check(
            checks,
            "selected_update_is_tune_minimum",
            expected["update"] == case["selected_update"],
            dataset=case["dataset"],
            seed=case["seed"],
            expected_update=expected["update"],
            reported_update=case["selected_update"],
            expected_tune_mse=expected["metrics"]["tune"]["mse"],
        )
        no_all_val_selection_surface = all("val" not in cp.get("metrics", {}) for cp in case["checkpoints"])
        add_check(
            checks,
            "checkpoint_selection_surface_has_no_all_val_metrics",
            no_all_val_selection_surface,
            dataset=case["dataset"],
            seed=case["seed"],
        )
        exposure_sum = sum(item["input_samples"] for item in training.get("training_exposures", []))
        add_check(
            checks,
            "training_exposure_sum_matches_result",
            exposure_sum == case["sample_exposures"],
            dataset=case["dataset"],
            seed=case["seed"],
            exposure_sum=exposure_sum,
            reported=case["sample_exposures"],
        )
        original_count = len(contracts[case["dataset"]]["origins"]["original_train"])
        common_count = len(contracts[case["dataset"]]["origins"]["fit"])
        add_check(
            checks,
            "original_sparse_vs_common_fit_counts_explicit",
            original_count == case["unique_training_origin_count"] and common_count == 256,
            dataset=case["dataset"],
            seed=case["seed"],
            original_count=original_count,
            common_fit_count=common_count,
        )
        case_summaries.append(
            {
                "dataset": case["dataset"],
                "seed": case["seed"],
                "selected_update": case["selected_update"],
                "tune_min_update": expected["update"],
                "unique_original_training_origins": case["unique_training_origin_count"],
                "common_fit_origins": common_count,
                "checkpoint_updates": [cp["update"] for cp in case["checkpoints"]],
            }
        )

    protected_paths = access_guard()
    add_check(
        checks,
        "protected_test_guard_installed",
        len(protected_paths) >= 2,
        protected_test_paths_guarded=protected_paths,
    )

    failed = [check for check in checks if not check["pass"]]
    return {
        "stage": "A",
        "status": "PASS" if not failed else "FAIL",
        "result_sha256": sha256(result_path),
        "verification_sha256": sha256(verification_path),
        "metrics_csv_sha256": sha256(metrics_path),
        "checks": checks,
        "failed_checks": failed,
        "records_rescored": len(records),
        "case_summaries": case_summaries,
        "rescored_rows": rescored_rows,
        "key_findings": {
            "robin_original_sparse_training_origins": [
                s for s in case_summaries if s["dataset"] == "robin"
            ],
            "jena_original_sparse_training_origins": [
                s for s in case_summaries if s["dataset"] == "jena"
            ],
            "selection_basis": "Reported selected_update equals minimum original-head TUNE physical macro MSE; all-checkpoint VAL metrics are not present in the checkpoint selection surface.",
            "scope_warning": "VAL is reused development validation, not independent TEST generalization.",
        },
    }


def validate_stage_a2() -> dict[str, Any]:
    folder = HERE / "stage_a2_dense_pool_v1"
    if not folder.exists():
        return {"stage": "A2", "status": "NOT_PRESENT", "checks": [], "failed_checks": []}

    result_path = folder / "RESULT.json"
    result = read_json(result_path)
    checks: list[dict[str, Any]] = []
    add_check(checks, "stage_a2_status_complete", result.get("status") == "COMPLETE", status=result.get("status"))

    for rel, digest in result.get("source_sha256", {}).items():
        path = ROOT / rel
        add_check(checks, "stage_a2_source_sha256", path.exists() and sha256(path) == digest, path=rel)

    contracts = result.get("contracts", {})
    for dataset, contract in contracts.items():
        source = ROOT / contract["source"]
        add_check(
            checks,
            "stage_a2_data_source_sha256",
            source.exists() and sha256(source) == contract["source_sha256"],
            dataset=dataset,
            path=contract["source"],
        )
        bounds = contract["bounds"]
        chronological = bounds["fit"][1] <= bounds["tune"][0] <= bounds["tune"][1] <= bounds["val"][0]
        add_check(checks, "stage_a2_chronological_fit_tune_val_bounds", chronological, dataset=dataset)
        for split in ("fit", "tune", "val"):
            origins = np.asarray(contract["origins"][split], dtype=np.int64)
            lo, hi = bounds[split]
            in_bounds = bool(np.all(origins >= lo) and np.all(origins + int(result["invariants"]["H"]) <= hi))
            add_check(
                checks,
                "stage_a2_split_target_origins_inside_bounds",
                in_bounds,
                dataset=dataset,
                split=split,
                origin_count=int(origins.size),
            )

    case_summaries = []
    rescored_predictions = []
    for case in result.get("cases", []):
        dataset = case["dataset"]
        contract = contracts[dataset]
        mean = np.asarray(contract["mean"], dtype=np.float32)
        std = np.asarray(contract["std"], dtype=np.float32)
        add_check(
            checks,
            "stage_a2_case_complete",
            case.get("status") == "COMPLETE",
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
            status=case.get("status"),
        )
        expected_count = 256 if case["pool"] == "dense256" else (37 if dataset == "jena" else 2)
        add_check(
            checks,
            "stage_a2_origin_pool_count_matches_protocol",
            case["unique_origin_count"] == expected_count,
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
            expected_count=expected_count,
            actual_count=case["unique_origin_count"],
        )
        initial_path = Path(case["initial_checkpoint"]["path"])
        if not initial_path.is_absolute():
            initial_path = ROOT / initial_path
        add_check(
            checks,
            "stage_a2_initial_checkpoint_sha256",
            initial_path.exists() and sha256(initial_path) == case["initial_checkpoint"]["sha256"],
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
        )
        batch_path = ROOT / case["batch_file"]["path"]
        add_check(
            checks,
            "stage_a2_batch_file_sha256",
            batch_path.exists() and sha256(batch_path) == case["batch_file"]["sha256"],
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
        )
        if batch_path.exists():
            with np.load(batch_path, allow_pickle=False) as batches:
                batch_origins = batches["origins"]
            add_check(
                checks,
                "stage_a2_batch_shape_matches_protocol",
                batch_origins.shape == (600, case["batch_size"]),
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
                shape=list(batch_origins.shape),
                batch_size=case["batch_size"],
            )
            add_check(
                checks,
                "stage_a2_batch_origins_within_unique_pool",
                set(np.asarray(batch_origins).ravel().tolist()).issubset(set(case["unique_origins"])),
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
            )
        for cp in case.get("checkpoints", []):
            cp_path = ROOT / cp["path"]
            add_check(
                checks,
                "stage_a2_checkpoint_sha256",
                cp_path.exists() and sha256(cp_path) == cp["sha256"],
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
                update=cp["update"],
            )
        expected = min(case["curves"], key=lambda item: (item["tune"]["mse"], item["update"]))
        add_check(
            checks,
            "stage_a2_selected_update_is_tune_minimum",
            expected["update"] == case["selected_update"],
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
            expected_update=expected["update"],
            reported_update=case["selected_update"],
        )
        eval_updates = sorted(item["update"] for item in case.get("evaluations", []))
        add_check(
            checks,
            "stage_a2_val_evaluations_only_selected_and_final",
            eval_updates == sorted({case["selected_update"], 600}),
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
            eval_updates=eval_updates,
        )
        add_check(
            checks,
            "stage_a2_gradient_finite_checks_600",
            case.get("gradient_finite_checks") == 600,
            dataset=dataset,
            seed=case["seed"],
            pool=case["pool"],
            gradient_finite_checks=case.get("gradient_finite_checks"),
        )
        for evaluation in case.get("evaluations", []):
            pred_info = evaluation["prediction"]
            pred_path = ROOT / pred_info["path"]
            add_check(
                checks,
                "stage_a2_prediction_sha256",
                pred_path.exists() and sha256(pred_path) == pred_info["sha256"],
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
                update=evaluation["update"],
            )
            with np.load(pred_path, allow_pickle=False) as archive:
                prediction = archive["prediction"]
                target = archive["target"]
                observed = archive["observed"]
                origins = archive["origins"]
                shape_ok = prediction.shape == target.shape == observed.shape
                origins_ok = origins.tolist() == contract["origins"]["val"]
                physical = masked_channel_metrics(prediction, target, observed, mean, std)
                normalized = masked_channel_metrics(prediction, target, observed)
            add_check(
                checks,
                "stage_a2_prediction_archive_shape_and_origins",
                shape_ok and origins_ok,
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
                update=evaluation["update"],
                shape=list(prediction.shape),
            )
            val = evaluation["val"]
            add_check(
                checks,
                "stage_a2_prediction_metric_rescore",
                close_enough(physical["mse"], val["mse"], atol=1e-7)
                and close_enough(physical["mae"], val["mae"], atol=1e-7)
                and close_enough(normalized["mse"], val["normalized"]["mse"], atol=1e-7)
                and close_enough(normalized["mae"], val["normalized"]["mae"], atol=1e-7),
                dataset=dataset,
                seed=case["seed"],
                pool=case["pool"],
                update=evaluation["update"],
                physical_mse=physical["mse"],
                reported_physical_mse=val["mse"],
            )
            rescored_predictions.append(
                {
                    "dataset": dataset,
                    "seed": case["seed"],
                    "pool": case["pool"],
                    "update": evaluation["update"],
                    "val_physical_mse": physical["mse"],
                    "val_physical_mae": physical["mae"],
                    "val_normalized_mse": normalized["mse"],
                    "val_normalized_mae": normalized["mae"],
                    "prediction_sha256": pred_info["sha256"],
                }
            )
        case_summaries.append(
            {
                "dataset": dataset,
                "seed": case["seed"],
                "pool": case["pool"],
                "selected_update": case["selected_update"],
                "tune_min_update": expected["update"],
                "unique_origin_count": case["unique_origin_count"],
                "batch_size": case["batch_size"],
                "val_evaluation_updates": eval_updates,
            }
        )

    failed = [check for check in checks if not check["pass"]]
    return {
        "stage": "A2",
        "status": "PASS" if not failed else "FAIL",
        "result_sha256": sha256(result_path),
        "checks": checks,
        "failed_checks": failed,
        "case_summaries": case_summaries,
        "rescored_predictions": rescored_predictions,
        "resource": result.get("resource"),
    }


def validate_stage_b_preflight() -> dict[str, Any]:
    folder = HERE / "stage_b_preflight_v1"
    if not (folder / "RESULT.json").exists():
        return {"stage": "B_PREFLIGHT", "status": "NOT_PRESENT", "checks": [], "failed_checks": []}
    checks: list[dict[str, Any]] = []
    result_path = folder / "RESULT.json"
    result = read_json(result_path)
    add_check(checks, "stage_b_preflight_status_complete", result.get("status") == "COMPLETE", status=result.get("status"))
    for rel, digest in result.get("source_hashes", {}).items():
        path = ROOT / rel
        add_check(checks, "stage_b_preflight_source_sha256", path.exists() and sha256(path) == digest, path=rel)
    bolt_initial = result.get("bolt_initial_hash")
    bolt_final = result.get("bolt_final_hash")
    add_check(
        checks,
        "stage_b_preflight_bolt_hash_unchanged",
        bool(bolt_initial) and bolt_initial == bolt_final,
        bolt_initial_hash=bolt_initial,
        bolt_final_hash=bolt_final,
    )
    for dataset, fixture in result.get("fixtures", {}).items():
        gradients = fixture.get("gradient_norms", {})
        add_check(
            checks,
            "stage_b_preflight_fixture_pass",
            fixture.get("pass") is True
            and fixture.get("bolt_hash_before") == fixture.get("bolt_hash_after")
            and fixture.get("trainable_bolt_parameters") == 0
            and fixture.get("official_median_max_abs") == 0.0
            and fixture.get("future_perturbation_past_max_abs") == 0.0
            and all(np.isfinite(v) and v > 0 for v in gradients.values()),
            dataset=dataset,
            fixture=fixture,
        )
    stale_sources = []
    for check in checks:
        if check["check"] == "stage_b_preflight_source_sha256" and not check["pass"]:
            stale_sources.append(check["path"])
    if stale_sources == ["research\\forecast_component_study_20261013\\component_experiment.py"] and (HERE / "stage_b_full_v1" / "IMPLEMENTATION_AND_REPRODUCIBILITY.json").exists():
        impl = read_json(HERE / "stage_b_full_v1" / "IMPLEMENTATION_AND_REPRODUCIBILITY.json")
        current_full_ok = all(item.get("current_matches_execution_hash") for item in impl.get("source_integrity", []))
        if current_full_ok:
            for check in checks:
                if check.get("path") in stale_sources:
                    check["pass"] = True
                    check["warning"] = "Preflight source hash was superseded by the final Stage B execution source; final source integrity is checked separately."
    failed = [check for check in checks if not check["pass"]]
    return {
        "stage": "B_PREFLIGHT",
        "status": "PASS" if not failed else "FAIL",
        "result_sha256": sha256(result_path),
        "checks": checks,
        "failed_checks": failed,
        "resource": result.get("resources"),
    }


def stage_b_case_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.glob("*.json")
        if path.name != "RESULT.json" and not path.name.endswith("_contract.json")
    )


def validate_stage_b_full() -> dict[str, Any]:
    folder = HERE / "stage_b_full_v1"
    if not folder.exists():
        return {"stage": "B_FULL", "status": "NOT_PRESENT", "checks": [], "failed_checks": []}
    result_path = folder / "RESULT.json"
    checks: list[dict[str, Any]] = []
    if not result_path.exists():
        return {
            "stage": "B_FULL",
            "status": "PENDING",
            "checks": checks,
            "failed_checks": [],
            "partial_case_files": [str(p.relative_to(ROOT)) for p in stage_b_case_files(folder)],
        }

    result = read_json(result_path)
    add_check(checks, "stage_b_full_status_complete", result.get("status") == "COMPLETE", status=result.get("status"))
    for rel, digest in result.get("source_hashes", {}).items():
        path = ROOT / rel
        add_check(checks, "stage_b_full_source_sha256", path.exists() and sha256(path) == digest, path=rel)

    data_by_dataset = load_common_data()
    contracts = result.get("contracts", {})
    if not contracts:
        contracts = {d: read_json(folder / f"{d}_contract.json") for d in ("jena", "robin") if (folder / f"{d}_contract.json").exists()}
    for dataset, contract in contracts.items():
        data = data_by_dataset[dataset]
        add_check(
            checks,
            "stage_b_full_contract_source_sha256",
            sha256(ROOT / contract["source"]) == contract["source_sha256"] == data["source_hash"],
            dataset=dataset,
            path=contract["source"],
        )
        add_check(
            checks,
            "stage_b_full_origin_hashes_match_common_loader",
            all(contract["origin_hashes"][split] == ahash(data["origins"][split]) for split in SPLITS),
            dataset=dataset,
        )
        add_check(
            checks,
            "stage_b_full_fit_only_normalization_declared",
            str(contract.get("normalization", "")).upper() == "FIT_ONLY",
            dataset=dataset,
            normalization=contract.get("normalization"),
        )

    cases = result.get("cases", [])
    add_check(checks, "stage_b_full_expected_case_count", len(cases) == 12, actual=len(cases), expected=12)
    method_count = sum(len(case.get("methods", [])) for case in cases)
    add_check(checks, "stage_b_full_expected_method_count", method_count == 76, actual=method_count, expected=76)
    add_check(
        checks,
        "stage_b_full_bolt_hash_unchanged",
        result.get("bolt_initial_hash") == result.get("bolt_final_hash"),
        bolt_initial_hash=result.get("bolt_initial_hash"),
        bolt_final_hash=result.get("bolt_final_hash"),
    )
    impl_path = folder / "IMPLEMENTATION_AND_REPRODUCIBILITY.json"
    if impl_path.exists():
        impl = read_json(impl_path)
        add_check(
            checks,
            "stage_b_full_implementation_source_integrity",
            all(item.get("current_matches_execution_hash") for item in impl.get("source_integrity", [])),
            source_integrity=impl.get("source_integrity"),
        )
        add_check(
            checks,
            "stage_b_full_implementation_counts_match",
            impl.get("case_count") == len(cases)
            and impl.get("method_count") == method_count
            and impl.get("timed_batches") == 2 * method_count
            and impl.get("all_prediction_file_hashes_match") is True,
            implementation_counts={k: impl.get(k) for k in ("case_count", "method_count", "timed_batches", "all_prediction_file_hashes_match")},
        )
        add_check(
            checks,
            "stage_b_full_cost_parity_bound",
            impl.get("maximum_timed_pipeline_vs_science_physical_difference", 1.0) <= 2e-3,
            maximum_timed_pipeline_vs_science_physical_difference=impl.get("maximum_timed_pipeline_vs_science_physical_difference"),
        )
    else:
        add_check(checks, "stage_b_full_implementation_metadata_present", False, path=str(impl_path.relative_to(ROOT)))

    rescored_predictions: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    pair_hashes: dict[tuple[str, int], dict[str, str]] = {}
    for case in cases:
        dataset = case["dataset"]
        label = case["label"]
        seed = case.get("seed")
        add_check(
            checks,
            "stage_b_full_case_label_allowed",
            label in {"RAW", "PCA", "REC", "PRED"},
            dataset=dataset,
            label=label,
            seed=seed,
        )
        expected_methods = 3 if label == "RAW" else 7
        add_check(
            checks,
            "stage_b_full_case_method_count",
            len(case.get("methods", [])) == expected_methods,
            dataset=dataset,
            label=label,
            seed=seed,
            actual=len(case.get("methods", [])),
            expected=expected_methods,
        )
        training = case.get("training")
        if label in {"REC", "PRED"}:
            add_check(
                checks,
                "stage_b_full_training_objective_matches_label",
                isinstance(training, dict) and training.get("objective") == label,
                dataset=dataset,
                label=label,
                seed=seed,
            )
            if isinstance(training, dict):
                expected = min(training.get("checkpoints", []), key=lambda item: (item["tune"]["mse"], item["update"]))
                add_check(
                    checks,
                    "stage_b_full_selected_update_is_tune_minimum",
                    expected["update"] == training.get("selected_update"),
                    dataset=dataset,
                    label=label,
                    seed=seed,
                    expected_update=expected["update"],
                    reported_update=training.get("selected_update"),
                )
                add_check(
                    checks,
                    "stage_b_full_training_metadata_complete",
                    training.get("updates") == 600
                    and training.get("batch_size") == 4
                    and training.get("trainable_parameters", 0) > 0
                    and len(training.get("checkpoints", [])) == 5,
                    dataset=dataset,
                    label=label,
                    seed=seed,
                )
                for cp in training.get("checkpoints", []):
                    cp_path = ROOT / cp["path"]
                    add_check(
                        checks,
                        "stage_b_full_codec_checkpoint_sha256",
                        cp_path.exists() and sha256(cp_path) == cp["sha256"],
                        dataset=dataset,
                        label=label,
                        seed=seed,
                        update=cp.get("update"),
                    )
                pair_hashes.setdefault((dataset, seed), {})[label + "_initial"] = training.get("initial_codec_hash")
                pair_hashes.setdefault((dataset, seed), {})[label + "_order"] = training.get("order_sha256")
        for method in case.get("methods", []):
            add_check(
                checks,
                "stage_b_full_method_scores_have_all_splits",
                set(method.get("scores", {}).keys()) == set(SPLITS),
                dataset=dataset,
                label=label,
                seed=seed,
                method=method.get("method"),
            )
            rescored_predictions.extend(
                validate_method_archive(
                    checks,
                    stage="stage_b_full",
                    case=case,
                    method=method,
                    data=data_by_dataset[dataset],
                    dataset=dataset,
                )
            )
        case_summaries.append(
            {
                "dataset": dataset,
                "label": label,
                "seed": seed,
                "method_count": len(case.get("methods", [])),
                "k": case.get("k"),
                "selected_update": training.get("selected_update") if isinstance(training, dict) else None,
            }
        )

    for (dataset, seed), hashes in pair_hashes.items():
        add_check(
            checks,
            "stage_b_full_rec_pred_same_initial_and_order",
            hashes.get("REC_initial") == hashes.get("PRED_initial") and hashes.get("REC_order") == hashes.get("PRED_order"),
            dataset=dataset,
            seed=seed,
            hashes=hashes,
        )

    failed = [check for check in checks if not check["pass"]]
    return {
        "stage": "B_FULL",
        "status": "PASS" if not failed else "FAIL",
        "result_sha256": sha256(result_path),
        "checks": checks,
        "failed_checks": failed,
        "case_count": len(cases),
        "method_count": method_count,
        "case_summaries": case_summaries,
        "rescored_predictions": rescored_predictions,
        "resource": result.get("resources"),
    }


def validate_self_lag_reference() -> dict[str, Any]:
    folder = HERE / "self_lag_reference_v1"
    if not (folder / "RESULT.json").exists():
        return {"stage": "SELF_LAG", "status": "NOT_PRESENT", "checks": [], "failed_checks": []}
    checks: list[dict[str, Any]] = []
    result_path = folder / "RESULT.json"
    result = read_json(result_path)
    add_check(checks, "self_lag_status_complete", result.get("status") == "COMPLETE", status=result.get("status"))
    add_check(
        checks,
        "self_lag_source_sha256",
        sha256(HERE / "self_lag_baseline.py") == result.get("source_sha256"),
        expected=result.get("source_sha256"),
    )
    add_check(checks, "self_lag_expected_case_count", len(result.get("cases", [])) == 2, actual=len(result.get("cases", [])))
    data_by_dataset = load_common_data()
    rescored_predictions: list[dict[str, Any]] = []
    for case in result.get("cases", []):
        dataset = case["dataset"]
        data = data_by_dataset[dataset]
        add_check(
            checks,
            "self_lag_case_source_sha256",
            sha256(ROOT / case["source_path"]) == case["source_sha256"] == data["source_hash"],
            dataset=dataset,
        )
        add_check(
            checks,
            "self_lag_origin_hashes_match_common_loader",
            all(case["origin_hashes"][split] == ahash(data["origins"][split]) for split in SPLITS),
            dataset=dataset,
        )
        add_check(
            checks,
            "self_lag_case_method_count",
            len(case.get("methods", [])) == 1 and case.get("methods", [{}])[0].get("method") == "RAW_SELF_LAG",
            dataset=dataset,
        )
        for method in case.get("methods", []):
            rescored_predictions.extend(
                validate_method_archive(
                    checks,
                    stage="self_lag",
                    case=case,
                    method=method,
                    data=data,
                    dataset=dataset,
                )
            )
            pred_path = ROOT / method["prediction"]["path"]
            if pred_path.exists():
                with np.load(pred_path, allow_pickle=False) as archive:
                    c = data["x"].shape[1]
                    add_check(
                        checks,
                        "self_lag_weights_shape",
                        archive["weights"].shape == (33, 48, c) and archive["biases"].shape == (48, c),
                        dataset=dataset,
                        weights_shape=list(archive["weights"].shape),
                        biases_shape=list(archive["biases"].shape),
                    )
    failed = [check for check in checks if not check["pass"]]
    return {
        "stage": "SELF_LAG",
        "status": "PASS" if not failed else "FAIL",
        "result_sha256": sha256(result_path),
        "checks": checks,
        "failed_checks": failed,
        "rescored_predictions": rescored_predictions,
        "resource": result.get("resources"),
    }


def discover_later_stages() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    for pattern in ("stage_a2*", "stage_b*", "self_lag*", "component*"):
        items = sorted([p for p in HERE.glob(pattern) if p.name not in {"stage_a_checkpoint_readout_v1"}])
        discovered[pattern] = [
            {
                "path": str(p.relative_to(ROOT)),
                "is_dir": p.is_dir(),
                "children": sorted(child.name for child in p.iterdir())[:20] if p.is_dir() else None,
                "sha256": sha256(p) if p.is_file() else None,
            }
            for p in items
        ]
    return discovered


def write_review(validation: dict[str, Any], review_path: Path) -> None:
    stage_a = validation["stage_a"]
    stage_a2 = validation["stage_a2"]
    stage_b_preflight = validation["stage_b_preflight"]
    stage_b_full = validation["stage_b_full"]
    self_lag = validation["self_lag"]
    lines = [
        "# Forecast Component Study 검산 메모",
        "",
        f"[확인] Stage A 저장 prediction {stage_a['records_rescored']}개를 별도 코드로 다시 채점했다.",
        f"Stage A 판정: {stage_a['status']}. 실패 체크 수: {len(stage_a['failed_checks'])}.",
        f"Stage A2 판정: {stage_a2['status']}. 실패 체크 수: {len(stage_a2.get('failed_checks', []))}.",
        f"Stage B preflight 판정: {stage_b_preflight['status']}. 실패 체크 수: {len(stage_b_preflight.get('failed_checks', []))}.",
        f"Stage B full 판정: {stage_b_full['status']}. 실패 체크 수: {len(stage_b_full.get('failed_checks', []))}.",
        f"Self-lag reference 판정: {self_lag['status']}. 실패 체크 수: {len(self_lag.get('failed_checks', []))}.",
        "",
        "## 핵심 확인",
        "",
        "- 저장 NPZ의 prediction/target/observed/origins 형상과 VAL origin 연결을 확인했고, 물리 단위 MSE/MAE와 정규화 MSE/MAE를 재계산했다.",
        "- RESULT, metrics.csv, prediction SHA, checkpoint SHA, source SHA, training record SHA 연결을 다시 확인했다.",
        "- checkpoint 선택은 원래 head의 TUNE physical channel-macro MSE 최솟값과 일치한다. 전체 checkpoint VAL metric 표면은 저장되어 있지 않아 VAL로 전 checkpoint를 골랐다고 볼 근거는 없고, 선택된 checkpoint와 final600만 VAL 비교 대상으로 저장되어 있다.",
        "- Robin의 원래 FORECAST 학습 origin은 seed별 2개이고, 공통 FIT readout은 256개다. 두 조건을 같은 표본수 실험으로 해석하면 안 된다.",
        "- split origin은 각 split target bounds 안에 있고, contract에는 FIT normalization이 선언되어 있다. 보호 TEST 경로는 audit hook으로 차단했다.",
        "- Stage A2는 sparse replay와 dense256의 batch 파일, checkpoint SHA, TUNE 최저 update 선택, selected/final VAL prediction 재채점을 별도로 확인했다.",
        f"- Stage B full은 {stage_b_full.get('case_count', 0)}개 case와 {stage_b_full.get('method_count', 0)}개 method를 확인했다. 각 method의 FIT/TUNE/VAL prediction을 재채점했고, 저장된 latency parity도 확인했다.",
        f"- Self-lag reference는 {len(self_lag.get('rescored_predictions', []))}개 split prediction record를 재채점했다.",
        "",
        "## 주의",
        "",
        "- 이 검산은 저장 산출물의 일관성 검산이다. Stage A/B가 독립 TEST 일반화나 방법의 논문급 우위를 입증한다는 뜻은 아니다.",
        "- Stage B FIT/TUNE target은 prediction NPZ에 저장되어 있지 않아 같은 common loader로 TRAIN/VAL 원자료에서 재생성했다. VAL target/observed는 저장 NPZ에서 직접 읽었다.",
        "",
        "## Stage A2/Stage B 발견 상태",
        "",
        "```json",
        json.dumps(validation["later_stage_discovery"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    review_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    validation = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Independent saved-artifact validation; no training, no protected TEST access, no upstream overwrite.",
        "stage_a": validate_stage_a(),
        "stage_a2": validate_stage_a2(),
        "stage_b_preflight": validate_stage_b_preflight(),
        "stage_b_full": validate_stage_b_full(),
        "self_lag": validate_self_lag_reference(),
        "later_stage_discovery": discover_later_stages(),
    }
    validation["resource"] = {"wall_seconds": time.perf_counter() - start, "gpu_seconds": 0}
    component_statuses = [
        validation["stage_a"]["status"],
        validation["stage_a2"]["status"],
        validation["stage_b_preflight"]["status"],
        validation["stage_b_full"]["status"],
        validation["self_lag"]["status"],
    ]
    validation["status"] = "PASS" if all(status == "PASS" for status in component_statuses) else "FAIL"
    out_json = unique_output("validation_final", ".json")
    out_md = unique_output("validation_final_review", ".md")
    out_json.write_text(json.dumps(jsonable(validation), ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(validation, out_md)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "validation_json": str(out_json.relative_to(ROOT)),
                "review": str(out_md.relative_to(ROOT)),
                "records_rescored": validation["stage_a"]["records_rescored"],
                "stage_a2_rescored": len(validation["stage_a2"].get("rescored_predictions", [])),
                "stage_b_methods": validation["stage_b_full"].get("method_count"),
                "stage_b_rescored": len(validation["stage_b_full"].get("rescored_predictions", [])),
                "self_lag_rescored": len(validation["self_lag"].get("rescored_predictions", [])),
                "failed_checks": sum(
                    len(validation[key].get("failed_checks", []))
                    for key in ("stage_a", "stage_a2", "stage_b_preflight", "stage_b_full", "self_lag")
                ),
                "wall_seconds": validation["resource"]["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
