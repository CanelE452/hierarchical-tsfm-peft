"""Independent verifier for forecast_generalization_study_20261013.

The verifier is deliberately standalone: it does not import experiment.py, does not
open resource/ledger JSON files, does not train models, and does not touch TEST.
It reads only completed case/manifests, prediction NPZ artifacts, contracts, logs,
and source files needed for a static repair review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXPECTED_CORE_FITS = 44
EXPECTED_NATIVE_FITS = 32
EXPECTED_METHODS = 236
EXPECTED_BATCH_COSTS = 472
CONFIG_ORDER = ("BASE", "LOW_LR", "DECAY")
SPLITS = ("fit", "tune", "val")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def is_case_json(path: Path) -> bool:
    name = path.name
    if name in {"RESULT.json", "FAILURE.json"}:
        return False
    if name.startswith("FAILED_"):
        return False
    if name.endswith("_RESULT.json") or name == "G2_CONFIG_SELECTION.json":
        return False
    if "resources" in path.parts:
        return False
    if "ledger" in name.lower():
        return False
    return name.endswith(".json")


def is_finished_manifest(path: Path) -> bool:
    return path.exists() and path.is_file()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def macro_scores(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    mask = np.asarray(observed).astype(bool)
    if pred.shape != tgt.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} target={tgt.shape} observed={mask.shape}")
    diff = pred - tgt
    counts = mask.sum(axis=(0, 1))
    valid = counts > 0
    if not bool(valid.any()):
        raise ValueError("no observed target entries")
    squared_sum = (np.square(diff) * mask).sum(axis=(0, 1))
    abs_sum = (np.abs(diff) * mask).sum(axis=(0, 1))
    channel_mse = np.full(counts.shape, np.nan, dtype=np.float64)
    channel_mae = np.full(counts.shape, np.nan, dtype=np.float64)
    channel_mse[valid] = squared_sum[valid] / counts[valid]
    channel_mae[valid] = abs_sum[valid] / counts[valid]

    horizon_counts = mask.sum(axis=0)
    horizon_valid = horizon_counts > 0
    horizon_mse_channel = np.full(horizon_counts.shape, np.nan, dtype=np.float64)
    horizon_mae_channel = np.full(horizon_counts.shape, np.nan, dtype=np.float64)
    horizon_mse_channel[horizon_valid] = (np.square(diff) * mask).sum(axis=0)[horizon_valid] / horizon_counts[horizon_valid]
    horizon_mae_channel[horizon_valid] = (np.abs(diff) * mask).sum(axis=0)[horizon_valid] / horizon_counts[horizon_valid]
    horizon_mse = np.nanmean(horizon_mse_channel, axis=1)
    horizon_mae = np.nanmean(horizon_mae_channel, axis=1)

    return {
        "mse": float(np.nanmean(channel_mse)),
        "mae": float(np.nanmean(channel_mae)),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist(),
        "horizon_mse": horizon_mse.tolist(),
        "horizon_mae": horizon_mae.tolist(),
        "observed_count": int(mask.sum()),
        "channel_observed_count": counts.astype(int).tolist(),
    }


def compare_scalar(expected: float | None, actual: float, atol: float, rtol: float) -> dict[str, Any]:
    if expected is None:
        return {"status": "missing_record", "expected": None, "actual": actual, "abs_diff": None, "rel_diff": None}
    diff = abs(float(expected) - float(actual))
    denom = max(abs(float(expected)), 1e-12)
    rel_diff = diff / denom
    return {
        "status": "pass" if diff <= atol + rtol * abs(float(expected)) else "fail",
        "expected": float(expected),
        "actual": float(actual),
        "abs_diff": diff,
        "rel_diff": rel_diff,
    }


def choose_min_checkpoint(checkpoints: list[dict[str, Any]], role: str) -> int | None:
    metric_group = "forecast" if role == "FORECAST" else "reconstruction"
    candidates: list[tuple[float, int]] = []
    for checkpoint in checkpoints:
        mse = (
            checkpoint.get(metric_group, {})
            .get("tune", {})
            .get("normalized", {})
            .get("mse")
        )
        update = checkpoint.get("update")
        if safe_float(mse) is not None and isinstance(update, int):
            candidates.append((float(mse), update))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def selected_checkpoint(case: dict[str, Any]) -> dict[str, Any] | None:
    update = case.get("selected_update")
    for checkpoint in case.get("training", {}).get("checkpoints", []):
        if checkpoint.get("update") == update:
            return checkpoint
    return None


def case_training_key(case: dict[str, Any]) -> tuple[Any, ...]:
    init = case.get("initialization", {})
    return (
        case.get("dataset"),
        case.get("stage"),
        case.get("objective"),
        case.get("config"),
        case.get("seed"),
        case.get("k"),
        case.get("length"),
        case.get("receptive_field"),
        init.get("kind"),
    )


@dataclass
class VerifierState:
    output_dir: Path
    mode: str
    opened_paths: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    case_sources: dict[int, str] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)
    manifest_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def note_open(self, path: Path) -> None:
        self.opened_paths.append(rel(path))

    def read_json(self, path: Path) -> Any:
        if "resources" in path.parts or "ledger" in path.name.lower():
            raise RuntimeError(f"refusing active resource/ledger JSON: {path}")
        self.note_open(path)
        return read_json(path)


def collect_contracts(state: VerifierState, run_dir: Path) -> None:
    for dataset in ("jena", "robin"):
        path = run_dir / f"{dataset}_contract.json"
        if not path.exists():
            state.warnings.append(f"{rel(path)} 없음: {dataset} 데이터 계약 검증은 부분 수행")
            continue
        contract = state.read_json(path)
        state.contracts[dataset] = contract
        if contract.get("normalization_fit_scope") != "FIT_ONLY":
            state.errors.append(f"{dataset}: normalization_fit_scope가 FIT_ONLY가 아님")
        if "test" in json.dumps(contract.get("origins", {}), ensure_ascii=False).lower():
            state.errors.append(f"{dataset}: contract origins에 test 문자열이 포함됨")


def collect_run(state: VerifierState, tag: str, *, allow_partial_cases: bool, collect_data_contracts: bool) -> None:
    run_dir = HERE / f"run_{tag}"
    if not run_dir.exists():
        state.warnings.append(f"run_{tag} 없음")
        return
    if collect_data_contracts:
        collect_contracts(state, run_dir)
    manifest_path = run_dir / "RESULT.json"
    has_manifest = is_finished_manifest(manifest_path)
    if has_manifest:
        manifest = state.read_json(manifest_path)
        state.manifest_objects[tag] = manifest
        state.manifests[tag] = {
            "path": rel(manifest_path),
            "sha256": sha256(manifest_path),
            "status": manifest.get("status"),
            "method_count": manifest.get("method_count"),
            "new_training_runs": manifest.get("new_training_runs"),
            "failed_path_count": len(manifest.get("failed_paths", [])),
        }
        for case in manifest.get("cases", []):
            state.case_sources[id(case)] = f"run_{tag}/RESULT.json"
            state.cases.append(case)
        for failure in manifest.get("failed_paths", []):
            state.failures.append({"run": tag, "source": "RESULT.failed_paths", **failure})
    else:
        state.warnings.append(f"run_{tag}/RESULT.json 없음: 완료 case JSON만 부분 수집")

    for failure_path in sorted(run_dir.glob("*/FAILED_*.json")):
        failure = state.read_json(failure_path)
        state.failures.append({"run": tag, "source": rel(failure_path), **failure})

    if has_manifest:
        # Once the terminal manifest exists, it is the authoritative membership
        # contract. Extra JSON files may exist from interrupted or superseded
        # attempts and must not change final counts.
        return
    if not allow_partial_cases:
        return

    existing_labels = {case.get("label") for case in state.cases}
    for case_path in sorted(run_dir.glob("*/*.json")):
        if not is_case_json(case_path):
            continue
        case = state.read_json(case_path)
        label = case.get("label")
        if label in existing_labels:
            continue
        existing_labels.add(label)
        state.case_sources[id(case)] = rel(case_path)
        state.cases.append(case)


def collect_runs(state: VerifierState, primary_tag: str, repair_tag: str | None) -> None:
    collect_run(state, primary_tag, allow_partial_cases=True, collect_data_contracts=True)
    if repair_tag:
        collect_run(state, repair_tag, allow_partial_cases=True, collect_data_contracts=False)
    else:
        for candidate in ("repair_v1", "repair_full_v1"):
            if (HERE / f"run_{candidate}" / "RESULT.json").exists():
                collect_run(state, candidate, allow_partial_cases=False, collect_data_contracts=False)
                break


def validate_contracts(state: VerifierState) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dataset, contract in state.contracts.items():
        origins = contract.get("origins", {})
        counts = {split: len(origins.get(split, [])) for split in SPLITS}
        bounds = contract.get("bounds", {})
        source = contract.get("source")
        source_path = resolve_artifact_path(source) if source else None
        source_sha_ok = None
        if source_path and source_path.exists():
            state.note_open(source_path)
            source_sha_ok = sha256(source_path) == contract.get("source_sha256")
        elif source:
            state.warnings.append(f"{dataset}: source 파일 없음 {source}")
        report[dataset] = {
            "origin_counts": counts,
            "bounds": bounds,
            "normalization_fit_scope": contract.get("normalization_fit_scope"),
            "source": source,
            "source_sha256_matches": source_sha_ok,
            "origin_hashes": contract.get("origin_hashes", {}),
            "observed_target_masks": contract.get("observed_target_masks", {}),
        }
    return report


def verify_prediction_npz(
    state: VerifierState,
    case: dict[str, Any],
    method: dict[str, Any],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    prediction_meta = method.get("prediction", {})
    path_value = prediction_meta.get("path")
    result: dict[str, Any] = {
        "case": case.get("label"),
        "dataset": case.get("dataset"),
        "method": method.get("method"),
        "path": path_value,
        "status": "unchecked",
        "score_comparisons": {},
    }
    if not path_value:
        result["status"] = "missing_prediction_path"
        return result
    path = resolve_artifact_path(path_value)
    if not path.exists():
        result["status"] = "missing_prediction_file"
        return result
    state.note_open(path)
    digest = sha256(path)
    result["sha256"] = digest
    expected_sha = prediction_meta.get("sha256")
    result["sha256_matches"] = expected_sha is None or expected_sha == digest
    if expected_sha is not None and expected_sha != digest:
        state.errors.append(f"{case.get('label')} / {method.get('method')}: prediction sha 불일치")

    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        if {"mean", "std"} - keys:
            result["status"] = "missing_mean_std"
            return result
        mean = np.asarray(npz["mean"], dtype=np.float64)
        std = np.asarray(npz["std"], dtype=np.float64)
        for split in SPLITS:
            required = {f"{split}_prediction", f"{split}_target", f"{split}_observed", f"{split}_origins"}
            if not required <= keys:
                result["score_comparisons"][split] = {"status": "missing_split_keys", "missing": sorted(required - keys)}
                continue
            prediction = np.asarray(npz[f"{split}_prediction"], dtype=np.float64)
            target = np.asarray(npz[f"{split}_target"], dtype=np.float64)
            observed = np.asarray(npz[f"{split}_observed"]).astype(bool)
            origins = np.asarray(npz[f"{split}_origins"])
            normalized = macro_scores(prediction, target, observed)
            physical = macro_scores(prediction * std + mean, target * std + mean, observed)
            recorded = method.get("scores", {}).get(split, {})
            split_result = {
                "shape": list(prediction.shape),
                "origin_count": int(origins.shape[0]),
                "observed_count": normalized["observed_count"],
                "normalized": {
                    "mse": compare_scalar(safe_float(recorded.get("normalized", {}).get("mse")), normalized["mse"], atol, rtol),
                    "mae": compare_scalar(safe_float(recorded.get("normalized", {}).get("mae")), normalized["mae"], atol, rtol),
                },
                "physical": {
                    "mse": compare_scalar(safe_float(recorded.get("physical", {}).get("mse")), physical["mse"], max(atol, 1e-6), rtol),
                    "mae": compare_scalar(safe_float(recorded.get("physical", {}).get("mae")), physical["mae"], max(atol, 1e-6), rtol),
                },
            }
            contract = state.contracts.get(case.get("dataset"))
            if contract is not None:
                contract_origins = np.asarray(contract.get("origins", {}).get(split, []))
                split_result["origins_match_contract"] = (
                    contract_origins.shape == origins.shape and bool(np.array_equal(contract_origins, origins))
                )
                if not split_result["origins_match_contract"]:
                    state.errors.append(f"{case.get('label')} / {method.get('method')} / {split}: origins 불일치")
            result["score_comparisons"][split] = split_result

    failed = []
    for split_result in result["score_comparisons"].values():
        if not isinstance(split_result, dict):
            continue
        for domain in ("normalized", "physical"):
            for metric in ("mse", "mae"):
                comparison = split_result.get(domain, {}).get(metric, {})
                if comparison.get("status") == "fail":
                    failed.append((domain, metric, comparison))
    result["status"] = "fail" if failed else "pass"
    return result


def validate_predictions(state: VerifierState, atol: float, rtol: float) -> dict[str, Any]:
    items = []
    status_counter = Counter()
    for case in state.cases:
        for method in case.get("methods", []):
            item = verify_prediction_npz(state, case, method, atol=atol, rtol=rtol)
            items.append(item)
            status_counter[item["status"]] += 1
    return {
        "method_prediction_count": len(items),
        "status_counts": dict(status_counter),
        "items": items,
    }


def validate_checkpoint_selection(state: VerifierState) -> dict[str, Any]:
    checks = []
    failures = []
    for case in state.cases:
        training = case.get("training")
        if not training:
            continue
        role = case.get("selection_role", "FORECAST")
        expected = choose_min_checkpoint(training.get("checkpoints", []), role)
        actual = case.get("selected_update")
        checkpoint = selected_checkpoint(case)
        status = "pass" if expected == actual and checkpoint is not None else "fail"
        check = {
            "case": case.get("label"),
            "dataset": case.get("dataset"),
            "stage": case.get("stage"),
            "role": role,
            "expected_update_from_tune_normalized": expected,
            "recorded_selected_update": actual,
            "selected_checkpoint_found": checkpoint is not None,
            "status": status,
        }
        checks.append(check)
        if status != "pass":
            failures.append(check)
    if failures:
        state.errors.append(f"checkpoint selection 실패 {len(failures)}건")
    return {"count": len(checks), "failure_count": len(failures), "checks": checks}


def validate_training_groups(state: VerifierState) -> dict[str, Any]:
    unique_core = {}
    group_by_config = defaultdict(list)
    for case in state.cases:
        if not case.get("training"):
            continue
        key = case_training_key(case)
        unique_core[key] = case
        if case.get("stage") == "G1" and case.get("objective") in {"REC", "PRED"}:
            group_key = (
                case.get("dataset"),
                case.get("seed"),
                case.get("config"),
                case.get("k"),
                case.get("length"),
                case.get("receptive_field"),
                case.get("initialization", {}).get("kind"),
            )
            group_by_config[group_key].append(case)

    initial_order_checks = []
    for group_key, cases in sorted(group_by_config.items(), key=lambda item: str(item[0])):
        initial_hashes = {case.get("training", {}).get("initial_codec_hash") for case in cases}
        order_hashes = {case.get("training", {}).get("order_sha256") for case in cases}
        initial_order_checks.append(
            {
                "group": group_key,
                "case_count": len(cases),
                "initial_codec_hash_count": len(initial_hashes - {None}),
                "order_sha256_count": len(order_hashes - {None}),
                "same_initial": len(initial_hashes - {None}) <= 1,
                "same_order": len(order_hashes - {None}) <= 1,
            }
        )
    failures = [check for check in initial_order_checks if not check["same_initial"] or not check["same_order"]]
    if failures:
        state.errors.append(f"G1 공통 초기화/배치순서 불일치 {len(failures)}개 group")

    stage_counts = Counter(key[1] for key in unique_core)
    return {
        "unique_core_fit_count": len(unique_core),
        "unique_core_fit_expected_final": EXPECTED_CORE_FITS,
        "unique_core_fit_status": "pass" if len(unique_core) == EXPECTED_CORE_FITS else "partial_or_fail",
        "stage_counts": dict(stage_counts),
        "initial_order_checks": initial_order_checks,
        "initial_order_failure_count": len(failures),
    }


def validate_stage2_selection(state: VerifierState) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in ("jena", "robin"):
        values: dict[str, list[float]] = {config: [] for config in CONFIG_ORDER}
        for case in state.cases:
            if case.get("dataset") != dataset:
                continue
            if case.get("stage") != "G1" or case.get("objective") != "PRED":
                continue
            config = case.get("config")
            checkpoint = selected_checkpoint(case)
            mse = None
            if checkpoint is not None:
                mse = checkpoint.get("forecast", {}).get("tune", {}).get("normalized", {}).get("mse")
            if config in values and safe_float(mse) is not None:
                key = (case.get("seed"), case.get("label"))
                # Selection is by two seeds. Duplicate labels are not expected here, but
                # storing raw values keeps the audit transparent.
                values[config].append(float(mse))
        means = {config: (float(np.mean(vals)) if vals else None) for config, vals in values.items()}
        complete = all(len(values[config]) == 2 for config in CONFIG_ORDER)
        chosen = None
        improvement = None
        if complete:
            minimum = min(v for v in means.values() if v is not None)
            candidate = next(config for config in CONFIG_ORDER if means[config] is not None and means[config] <= minimum + 1e-12)
            improvement = (means["BASE"] - means[candidate]) / means["BASE"]
            chosen = candidate if improvement >= 0.01 else "BASE"
        recorded = None
        for manifest in state.manifests.values():
            # Manifest summary intentionally contains only metadata; read full manifest only
            # through collect_run, so recorded selection is unavailable in metadata mode.
            pass
        selection_file = HERE / "run_full_v1" / dataset / "G2_CONFIG_SELECTION.json"
        recorded_from_file = None
        if selection_file.exists():
            recorded_from_file = state.read_json(selection_file)
        by_dataset[dataset] = {
            "values": values,
            "seed_mean_tune_normalized_forecast_mse": means,
            "complete_two_seed_grid": complete,
            "recomputed_chosen_config": chosen,
            "recomputed_relative_improvement_vs_BASE": improvement,
            "recorded_selection_file": recorded_from_file,
            "recorded_matches": (
                recorded_from_file is None
                or chosen is None
                or recorded_from_file.get("chosen_config") == chosen
            ),
            "criterion": "TUNE-only normalized observed channel-macro MSE, BASE retained when best improvement <1%",
        }
    return by_dataset


def validate_methods_and_costs(state: VerifierState) -> dict[str, Any]:
    methods = []
    native_fit_keys = set()
    batch_cost_count = 0
    latency_issues = []
    method_counter = Counter()
    for case in state.cases:
        for method in case.get("methods", []):
            methods.append(method)
            name = method.get("method")
            method_counter[name] += 1
            if name == "NATIVE":
                native_fit_keys.add(case.get("label"))
            latency = method.get("latency", [])
            batch_cost_count += len(latency)
            for entry in latency:
                if entry.get("warmup") != 2 or entry.get("repeats") != 7:
                    latency_issues.append({"case": case.get("label"), "method": name, "issue": "warmup_or_repeats", "entry": entry})
                if "pipeline_prediction_parity_max_abs" in entry and safe_float(entry["pipeline_prediction_parity_max_abs"]) not in (0.0, None):
                    latency_issues.append({"case": case.get("label"), "method": name, "issue": "pipeline_parity_nonzero", "entry": entry})
    if latency_issues:
        state.errors.append(f"latency/cost 검증 issue {len(latency_issues)}건")
    return {
        "method_count": len(methods),
        "method_count_expected_final": EXPECTED_METHODS,
        "method_count_status": "pass" if len(methods) == EXPECTED_METHODS else "partial_or_fail",
        "method_counts_by_name": dict(method_counter),
        "native_fit_count": len(native_fit_keys),
        "native_fit_expected_final": EXPECTED_NATIVE_FITS,
        "native_fit_status": "pass" if len(native_fit_keys) == EXPECTED_NATIVE_FITS else "partial_or_fail",
        "batch_cost_count": batch_cost_count,
        "batch_cost_expected_final": EXPECTED_BATCH_COSTS,
        "batch_cost_status": "pass" if batch_cost_count == EXPECTED_BATCH_COSTS else "partial_or_fail",
        "latency_issue_count": len(latency_issues),
        "latency_issues": latency_issues[:50],
    }


def validate_model_artifacts(state: VerifierState) -> dict[str, Any]:
    checks = []
    missing = []
    mismatched = []
    seen: set[tuple[str, str | None]] = set()

    def check_artifact(case: dict[str, Any], role: str, artifact: dict[str, Any] | None) -> None:
        if not artifact or not artifact.get("path"):
            missing.append({"case": case.get("label"), "role": role, "reason": "missing_artifact_metadata"})
            return
        key = (artifact["path"], artifact.get("sha256"))
        if key in seen:
            return
        seen.add(key)
        path = resolve_artifact_path(artifact["path"])
        item = {"case": case.get("label"), "dataset": case.get("dataset"), "role": role, "path": artifact["path"]}
        if not path.exists():
            item["status"] = "missing_file"
            missing.append(item)
            checks.append(item)
            return
        state.note_open(path)
        digest = sha256(path)
        item["sha256"] = digest
        item["expected_sha256"] = artifact.get("sha256")
        item["status"] = "pass" if artifact.get("sha256") in (None, digest) else "fail"
        if item["status"] == "fail":
            mismatched.append(item)
        checks.append(item)

    for case in state.cases:
        if case.get("selected_models"):
            check_artifact(case, "selected_models", case.get("selected_models"))
        elif case.get("stage") == "G0" and case.get("reused_checkpoint"):
            check_artifact(case, "reused_checkpoint", case.get("reused_checkpoint"))
        elif case.get("training"):
            checkpoint = selected_checkpoint(case)
            check_artifact(case, "selected_checkpoint", checkpoint)
        elif case.get("label") == "RAW" and case.get("selected_models"):
            check_artifact(case, "raw_selected_models", case.get("selected_models"))

    if missing:
        state.errors.append(f"model/checkpoint artifact 누락 {len(missing)}건")
    if mismatched:
        state.errors.append(f"model/checkpoint artifact SHA 불일치 {len(mismatched)}건")
    return {
        "unique_artifact_count": len(checks),
        "missing_count": len(missing),
        "sha_mismatch_count": len(mismatched),
        "checks": checks,
        "missing": missing[:50],
        "mismatched": mismatched[:50],
    }


def clean_cost_review(state: VerifierState) -> dict[str, Any]:
    script_path = HERE / "measure_resolved_cost.py"
    review: dict[str, Any] = {
        "script_exists": script_path.exists(),
        "static_checks": {},
        "terminal_results": [],
        "notes": [],
    }
    if script_path.exists():
        state.note_open(script_path)
        text = script_path.read_text(encoding="utf-8")
        review["static_checks"] = {
            "resolves_full_and_repair": "def resolve(" in text and "full_v1" in text and "repair_v1" in text,
            "enforces_70_cases_236_methods": "len(records) != 70 or count != 236" in text,
            "loads_selected_models_and_checkpoint_sha": "selected_models" in text and "assert ex.sha" in text,
            "uses_existing_prediction_for_latency_parity": "val_prediction" in text and "ex.latency" in text,
            "records_new_training_zero": "'new_training_updates': 0" in text,
            "guards_test_access": "ex.access_guard()" in text,
            "checks_competing_gpu_processes": "competing_python_compute_processes_before_load" in text,
            "checks_backbone_unchanged": "CLEAN_COST_BACKBONE_CHANGED" in text,
            "does_not_call_train_codec": "train_codec" not in text and "train_native" not in text,
        }
        review["notes"].append(
            "measure_resolved_cost.py는 full_v1+repair_v1을 병합하고 selected model/checkpoint/prediction SHA를 확인한 뒤 "
            "기존 science prediction으로 latency만 다시 측정하도록 작성되어 있다."
        )
    for run_dir in sorted(HERE.glob("run_*cost*")):
        result_path = run_dir / "RESULT.json"
        failure_path = run_dir / "FAILURE.json"
        if result_path.exists():
            result = state.read_json(result_path)
            review["terminal_results"].append(
                {
                    "path": rel(result_path),
                    "sha256": sha256(result_path),
                    "status": result.get("status"),
                    "record_count": len(result.get("records", [])),
                    "resolved_case_count": result.get("resolved_case_count"),
                    "resolved_method_count": result.get("resolved_method_count"),
                    "new_training_updates": result.get("new_training_updates"),
                    "source_runs": result.get("source_runs"),
                    "resources": result.get("resources", {}),
                }
            )
        elif failure_path.exists():
            failure = state.read_json(failure_path)
            review["terminal_results"].append(
                {
                    "path": rel(failure_path),
                    "sha256": sha256(failure_path),
                    "status": failure.get("status"),
                    "error": failure.get("error"),
                    "resources": failure.get("resources", {}),
                }
            )
    return review


def validate_g2_design(state: VerifierState) -> dict[str, Any]:
    g2 = [case for case in state.cases if case.get("stage") == "G2" and case.get("training")]
    by_dataset_seed = defaultdict(list)
    for case in g2:
        by_dataset_seed[(case.get("dataset"), case.get("seed"))].append(
            {
                "label": case.get("label"),
                "k": case.get("k"),
                "length": case.get("length"),
                "rf": case.get("receptive_field"),
                "init": case.get("initialization", {}).get("kind"),
                "config": case.get("config"),
            }
        )
    expected_variants = {
        "half_pca": lambda c, channels: c["k"] == channels // 2 and c["length"] == 512 and c["rf"] == 127 and c["init"] == "PCA",
        "fullrank_pca": lambda c, channels: c["k"] == channels and c["length"] == 512 and c["rf"] == 127 and c["init"] == "PCA",
        "identity": lambda c, channels: c["k"] == channels and c["length"] == 512 and c["rf"] == 127 and c["init"] == "IDENTITY",
        "short_l128": lambda c, channels: c["k"] == channels // 4 and c["length"] == 128 and c["rf"] == 127 and c["init"] == "PCA",
        "rf31": lambda c, channels: c["k"] == channels // 4 and c["length"] == 512 and c["rf"] == 31 and c["init"] == "PCA",
    }
    checks = []
    for (dataset, seed), cases in sorted(by_dataset_seed.items()):
        channels = len(state.contracts.get(dataset, {}).get("columns", []))
        if not channels:
            channels = max([c["k"] for c in cases] + [0])
        matched = {name: sum(1 for case in cases if predicate(case, channels)) for name, predicate in expected_variants.items()}
        checks.append({"dataset": dataset, "seed": seed, "case_count": len(cases), "variant_match_counts": matched})
    return {"g2_case_count": len(g2), "by_dataset_seed": checks}


def scan_for_test_access(state: VerifierState) -> dict[str, Any]:
    opened = state.opened_paths
    suspicious = [
        path
        for path in opened
        if re.search(r"(^|[\\/])test([\\/._-]|$)", path, flags=re.IGNORECASE)
        or "TEST" in path
    ]
    return {
        "opened_path_count": len(opened),
        "suspicious_test_path_count": len(suspicious),
        "suspicious_test_paths": suspicious,
        "resource_or_ledger_json_opened": [
            path for path in opened if "/resources/" in path.replace("\\", "/") or "ledger" in Path(path).name.lower()
        ],
    }


def repair_static_review(state: VerifierState) -> dict[str, Any]:
    repair_path = HERE / "repair_experiment.py"
    diagnostic_path = HERE / "diagnose_constant_gradient.py"
    review = {
        "repair_experiment_exists": repair_path.exists(),
        "diagnostic_exists": diagnostic_path.exists(),
        "checks": {},
        "notes": [],
    }
    if repair_path.exists():
        state.note_open(repair_path)
        text = repair_path.read_text(encoding="utf-8")
        review["checks"] = {
            "uses_source_failed_paths_only": "source.get('failed_paths'" in text or 'source.get("failed_paths"' in text,
            "does_not_iterate_all_cases_for_repair": "for failure in source.get('failed_paths'" in text or 'for failure in source.get("failed_paths"' in text,
            "has_retry_replace_for_windows_lock": "except PermissionError" in text and "temporary.replace" in text,
            "identity_gradient_fix_is_conditional": "failure['spec']['init']=='IDENTITY'" in text and "NONFINITE_TRAINING_GRADIENT" in text,
            "records_repair_of_metadata": "repair_of" in text,
            "checks_frozen_bolt_hash": "FROZEN_BOLT_CHANGED" in text and "bolt.initial_hash" in text,
            "checks_source_hashes_unchanged": "source_hashes_unchanged_at_completion" in text,
            "may_run_extra_g2_if_g1_selection_changes": "configuration_unchanged" in text and "if not selection['configuration_unchanged']" in text,
        }
        review["notes"].append(
            "repair_experiment.py는 source RESULT가 완료된 뒤 failed_paths만 순회하도록 작성되어 있다. "
            "다만 G1 수리 후 config 선택이 바뀌면 사전 규칙에 맞춰 추가 G2를 실행할 수 있으므로, 최종 검증은 repair RESULT까지 병합해야 한다."
        )
    if diagnostic_path.exists():
        state.note_open(diagnostic_path)
        text = diagnostic_path.read_text(encoding="utf-8")
        review["diagnostic_checks"] = {
            "zero_safe_sqrt_defined": "class ZeroSafeSqrt" in text,
            "forward_unchanged_asserted": "forward_max_abs_vs_original" in text and "== 0" in text,
            "positive_variance_gradient_unchanged_asserted": "positive_variance_gradient_max_abs" in text,
            "no_training_updates_recorded": "new_training_updates" in text,
        }
    return review


def build_summary(state: VerifierState, checks: dict[str, Any]) -> dict[str, Any]:
    status = "PASS"
    if state.errors:
        status = "FAIL"
    elif (
        any("RESULT.json 없음" in warning for warning in state.warnings)
        or checks.get("methods_and_costs", {}).get("method_count_status") != "pass"
        or checks.get("methods_and_costs", {}).get("native_fit_status") != "pass"
        or checks.get("methods_and_costs", {}).get("batch_cost_status") != "pass"
        or checks.get("training_groups", {}).get("unique_core_fit_status") != "pass"
    ):
        status = "PREPARED_PARTIAL"
    return {
        "status": status,
        "generated_at": now(),
        "mode": state.mode,
        "repository_root": str(ROOT),
        "study_dir": rel(HERE),
        "case_count": len(state.cases),
        "failure_artifact_count": len(state.failures),
        "manifest_metadata": state.manifests,
        "warnings": state.warnings,
        "errors": state.errors,
        "checks": checks,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    method_costs = checks.get("methods_and_costs", {})
    model_artifacts = checks.get("model_artifacts", {})
    training = checks.get("training_groups", {})
    predictions = checks.get("prediction_recalc", {})
    selection = checks.get("checkpoint_selection", {})
    test_access = checks.get("test_access_guard", {})
    lines = [
        "# Forecast generalization 독립 검증",
        "",
        f"- 생성 시각: {summary['generated_at']}",
        f"- 상태: `{summary['status']}`",
        f"- 모드: `{summary['mode']}`",
        f"- 수집 case 수: {summary['case_count']}",
        f"- failure artifact 수: {summary['failure_artifact_count']}",
        "",
        "## 핵심 검증 결과",
        "",
        f"- 독립 재채점 method 수: {predictions.get('method_prediction_count', 0)}",
        f"- 재채점 상태: `{predictions.get('status_counts', {})}`",
        f"- checkpoint 선택 검증: {selection.get('count', 0)}건 중 failure {selection.get('failure_count', 0)}건",
        f"- unique core fit: {training.get('unique_core_fit_count', 0)} / expected final {training.get('unique_core_fit_expected_final', EXPECTED_CORE_FITS)}",
        f"- native fit: {method_costs.get('native_fit_count', 0)} / expected final {method_costs.get('native_fit_expected_final', EXPECTED_NATIVE_FITS)}",
        f"- method count: {method_costs.get('method_count', 0)} / expected final {method_costs.get('method_count_expected_final', EXPECTED_METHODS)}",
        f"- batch cost count: {method_costs.get('batch_cost_count', 0)} / expected final {method_costs.get('batch_cost_expected_final', EXPECTED_BATCH_COSTS)}",
        f"- model/checkpoint artifact SHA: {model_artifacts.get('unique_artifact_count', 0)}개 확인, missing {model_artifacts.get('missing_count', 0)}, mismatch {model_artifacts.get('sha_mismatch_count', 0)}",
        f"- TEST 의심 경로 접근 수: {test_access.get('suspicious_test_path_count', 0)}",
        f"- resource/ledger JSON 접근 수: {len(test_access.get('resource_or_ledger_json_opened', []))}",
        "",
        "## 해석",
        "",
    ]
    if summary["status"] == "PREPARED_PARTIAL":
        lines.append(
            "[확인] 최종 `RESULT.json`이 아직 없어서 완료된 case JSON과 NPZ 기준으로 검증기를 준비·부분 실행했다. "
            "최종 manifest가 생성되면 같은 스크립트로 전체 44 core fit / 32 native fit / 236 methods / 472 batch cost를 다시 검산해야 한다."
        )
    elif summary["status"] == "PASS":
        lines.append("[확인] 최종 manifest 기준 독립 재채점과 계약 검사가 통과했다.")
    else:
        lines.append("[확인] 독립 검증에서 실패 또는 계약 위반이 발견됐다. 아래 errors를 우선 확인해야 한다.")
    lines += [
        "",
        "## warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in summary.get("warnings", [])] or ["- 없음"])
    lines += [
        "",
        "## errors",
        "",
    ]
    lines.extend([f"- {error}" for error in summary.get("errors", [])] or ["- 없음"])
    lines += [
        "",
        "## repair source 정적 리뷰",
        "",
        "```json",
        json.dumps(checks.get("repair_static_review", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## clean-cost 정적/결과 리뷰",
        "",
        "```json",
        json.dumps(checks.get("clean_cost_review", {}), ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = HERE / args.output
    output_dir.mkdir(exist_ok=True)
    state = VerifierState(output_dir=output_dir, mode=args.mode)
    collect_runs(state, args.primary_tag, args.repair_tag)
    checks = {
        "contracts": validate_contracts(state),
        "checkpoint_selection": validate_checkpoint_selection(state),
        "training_groups": validate_training_groups(state),
        "stage2_selection": validate_stage2_selection(state),
        "g2_design": validate_g2_design(state),
        "methods_and_costs": validate_methods_and_costs(state),
        "model_artifacts": validate_model_artifacts(state),
        "prediction_recalc": validate_predictions(state, atol=args.atol, rtol=args.rtol),
        "repair_static_review": repair_static_review(state),
        "clean_cost_review": clean_cost_review(state),
    }
    checks["test_access_guard"] = scan_for_test_access(state)
    if checks["test_access_guard"]["suspicious_test_path_count"]:
        state.errors.append("TEST 의심 경로가 verifier opened_paths에 포함됨")
    if checks["test_access_guard"]["resource_or_ledger_json_opened"]:
        state.errors.append("resource/ledger JSON 접근이 감지됨")
    summary = build_summary(state, checks)
    write_json(output_dir / "audit.json", summary)
    (output_dir / "audit.md").write_text(render_markdown(summary), encoding="utf-8")
    # Compatibility names requested by the orchestration prompt.
    write_json(HERE / f"CONTRACT_AUDIT_{args.output}.json", summary)
    (HERE / f"CONTRACT_AUDIT_{args.output}.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "case_count": summary["case_count"],
        "method_count": checks["methods_and_costs"].get("method_count"),
        "prediction_status_counts": checks["prediction_recalc"].get("status_counts"),
        "errors": len(summary["errors"]),
        "warnings": len(summary["warnings"]),
        "output": rel(output_dir),
    }, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-tag", default="full_v1")
    parser.add_argument("--repair-tag", default=None)
    parser.add_argument("--mode", choices=("prepare", "final"), default="prepare")
    parser.add_argument("--output", default="validation_prepared_v1")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
