"""Stage-separation terminal-JSON report generator.

SOURCE-only contract until ROOT GO:
- --execute constructs the approved runtime before output folders, matplotlib, or JSON reads.
- Reads terminal JSON only. It never opens scientific NPZ arrays, checkpoints, or F/model weights.
- Keeps future-oracle diagnostics (O0/O1) separate from deployable N1/N2/N3 candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "report_stage_separation_v1"
DATASETS = ("robin", "beijing_aotizhongxin")
SEEDS = (92601, 92602)
FINAL_SPLIT = {"robin": "val", "beijing_aotizhongxin": "eval"}
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
DEPLOY = ("N1", "N2", "N3")
ORACLE = ("O0_ORACLE_OLD_D", "O1")
FIGURES = (
    "01_deploy_mse_mae_dots",
    "02_oracle_diagnostic_scores",
    "03_tune_alpha_grid",
    "04_teacher_coverage_counts",
    "05_fit_tune_eval_overfit",
    "06_b1_b8_paired_cost",
)
TERMINAL = {"COMPLETE", "PASS", "FAIL", "FAILED_PRESERVED", "BUDGET_HELD_PRESERVED", "PARTIAL_WITH_FAILURES"}
SNAPSHOT_SOURCES = (
    "run_stage_native_ridge_v1.py",
    "run_stage_oracle_extract_v1.py",
    "profile_stage_native_cost_v1.py",
    "stage_separation_math_v1.py",
    "frozen_decoder_refit.py",
    "stage_separation_oracle_v1.py",
    "continuation_runtime_v1.py",
    "validation_v1/run_stage_native_audit_v4.py",
    "validation_v1/audit_stage_native_core_v4.py",
    "validation_v1/test_stage_native_audit_v4_numerics.py",
    "validation_v1/run_stage_native_audit_v3.py",
    "validation_v1/audit_stage_native_core_v3.py",
    "validation_v1/test_stage_native_audit_v3.py",
    "validation_v1/run_stage_native_audit_v2.py",
    "validation_v1/audit_stage_native_core_v2.py",
    "validation_v1/test_stage_native_audit_v2.py",
    "validation_v1/audit_stage_native_v1.py",
)


class ReportFailure(Exception):
    pass


def sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_json(path: Path | str, role: str) -> tuple[Any, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ReportFailure(f"MISSING_{role.upper()}_JSON:{resolved}")
    return read_json(resolved), {"role": role, "path": str(resolved), "sha256": sha_file(resolved)}


def write_json_new(path: Path | str, value: Any) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"path": str(destination.resolve()), "sha256": sha_file(destination)}


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def write_csv_new(path: Path | str, rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key not in extras:
                extras.append(key)
    header = fields + sorted(extras)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in header})
    return {"path": str(destination.resolve()), "sha256": sha_file(destination), "rows": len(rows)}


def safe_tag(out: Path | str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(out).name or "stage_report_v1").strip("_").lower()
    return (value or "stage_report_v1")[:80]


def score_pair(score: dict[str, Any] | None) -> tuple[Any, Any, Any, Any]:
    if not isinstance(score, dict):
        return None, None, None, None
    normalized = score.get("normalized", score.get("normalized_actual_Y", score))
    physical = score.get("physical_per_channel_diagnostic", score.get("physical_actual_Y", score.get("physical", {})))
    if not isinstance(normalized, dict):
        normalized = {}
    if not isinstance(physical, dict):
        physical = {}
    return normalized.get("mse"), normalized.get("mae"), physical.get("mse"), physical.get("mae")


def role_for(method: str) -> tuple[str, bool, bool]:
    if method in DEPLOY:
        return "deploy_candidate", False, True
    if method in ORACLE:
        return "future_oracle_diagnostic", True, False
    if method in ("RAW_BOLT", "ORIGINAL_MAIN") or method.startswith("OLD_"):
        return "old_reference", False, method == "RAW_BOLT"
    return "control", False, True


def add_score(rows: list[dict[str, Any]], base: dict[str, Any], method: str, split: str,
              score: dict[str, Any] | None, source: str, extra: dict[str, Any] | None = None) -> None:
    mse, mae, physical_mse, physical_mae = score_pair(score)
    role, diagnostic, deployable = role_for(method)
    rows.append({**base, "method": method, "split": split, "role": role, "diagnostic_only": diagnostic,
                 "deployable": deployable, "source": source, "mse": mse, "mae": mae,
                 "physical_mse_diagnostic": physical_mse, "physical_mae_diagnostic": physical_mae,
                 **(extra or {})})


def refit_method_name(objective: str) -> str:
    return "OLD_F_REFIT_D_" + ("RAW_TEACHER" if objective == "RAW" else str(objective))


def empty_tables() -> dict[str, list[dict[str, Any]]]:
    return {name: [] for name in ("scores", "alphagrid", "coverage", "zero_outputs", "cost", "status", "stagedeltas")}


def require_terminal(value: dict[str, Any], role: str) -> str:
    status = value.get("status")
    if status not in TERMINAL:
        raise ReportFailure(f"TERMINAL_{role.upper()}_STATUS_REQUIRED:{status}")
    return str(status)


def collect_native(result: dict[str, Any], record: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    dataset, seed = result.get("dataset"), result.get("seed")
    if dataset not in DATASETS or seed not in SEEDS:
        raise ReportFailure("UNEXPECTED_NATIVE_PRODUCER_DATASET_SEED")
    base = {"dataset": dataset, "seed": seed, "producer_sha256": record["sha256"], "producer_path": record["path"]}
    tables["status"].append({**base, "component": "native_producer", "status": require_terminal(result, "native_producer"),
                             "failure": result.get("failure"), "protected_TEST_accessed": result.get("protected_TEST_accessed"),
                             "F_loaded": result.get("F_loaded"), "F_forward_replayed": result.get("F_forward_replayed"),
                             "E_forward_replayed": result.get("E_forward_replayed"),
                             "independent_holdout": result.get("independent_holdout")})
    for split, group in (result.get("references") or {}).items():
        if isinstance(group, dict):
            if "RAW" in group:
                add_score(tables["scores"], base, "RAW_BOLT", split, group["RAW"], "native.references")
            if "ORIGINAL_MAIN" in group:
                add_score(tables["scores"], base, "ORIGINAL_MAIN", split, group["ORIGINAL_MAIN"], "native.references")
    for method, control in (result.get("controls") or {}).items():
        for split, score in (control.get("splits") or {}).items():
            add_score(tables["scores"], base, method, split, score, "native.controls",
                      {"model_sha256": control.get("model_sha256")})
    same_native = result.get("SAME_MAIN_NATIVE_existing") or {}
    if same_native:
        source_sha = ((same_native.get("lineage") or {}).get("native_prediction") or {}).get("sha256")
        tables["status"].append({**base, "component": "old_same_main_native_reference",
                                 "method": "OLD_SAME_MAIN_NATIVE", "status": same_native.get("status"),
                                 "new_training": same_native.get("new_training", False),
                                 "failure": same_native.get("failure"), "source_sha256": source_sha})
        for split, score in (same_native.get("splits") or {}).items():
            add_score(tables["scores"], base, "OLD_SAME_MAIN_NATIVE", split, score,
                      "native.SAME_MAIN_NATIVE_existing",
                      {"role": "old_reference", "new_training": False, "source_sha256": source_sha})
    refit = result.get("D2_existing_F_decoder_refits") or {}
    if refit:
        refit_source = refit.get("source") or {}
        tables["status"].append({**base, "component": "old_D2_F_refit_reference", "status": refit.get("status"),
                                 "new_training": False, "failure": refit.get("failure"),
                                 "source_path": refit_source.get("path"), "source_sha256": refit_source.get("sha256")})
        for objective, row in (refit.get("refits") or {}).items():
            method = refit_method_name(objective)
            tables["status"].append({**base, "component": "old_D2_F_refit_reference_objective",
                                     "method": method, "status": refit.get("status"), "new_training": False,
                                     "refit_objective": objective, "selected_alpha": row.get("selected_alpha"),
                                     "selected_index": row.get("selected_index"), "source_sha256": refit_source.get("sha256")})
            for split, score in (row.get("splits") or {}).items():
                add_score(tables["scores"], base, method, split, score,
                          "native.D2_existing_F_decoder_refits",
                          {"role": "old_reference", "new_training": False,
                           "refit_objective": objective, "selected_alpha": row.get("selected_alpha"),
                           "source_sha256": refit_source.get("sha256")})
    if isinstance(result.get("O0_ORACLE_OLD_D"), dict):
        for split, score in (result["O0_ORACLE_OLD_D"].get("splits") or {}).items():
            add_score(tables["scores"], base, "O0_ORACLE_OLD_D", split, score, "native.O0_ORACLE_OLD_D")
    for family, row in (result.get("families") or {}).items():
        if family not in {"O1", "N1", "N2", "N3"}:
            continue
        role, diagnostic, deployable = role_for(family)
        tables["status"].append({**base, "component": "family", "method": family, "status": row.get("status"),
                                 "failure": row.get("failure"), "selected_alpha": row.get("selected_alpha"),
                                 "selected_index": row.get("selected_index"), "selected_model_sha256": row.get("selected_model_sha256"),
                                 "diagnostic_only": diagnostic, "deployable": deployable})
        tables["zero_outputs"].append({**base, "method": family, "zero_observed_output_counts": row.get("zero_observed_output_counts"),
                                       "factorization_groups": row.get("factorization_groups"),
                                       "diagnostic_only": diagnostic, "deployable": deployable})
        for index, alpha in enumerate(row.get("alpha_grid") or []):
            fit_mse, fit_mae, _, _ = score_pair(alpha.get("fit_scores"))
            tune_mse, tune_mae, _, _ = score_pair(alpha.get("tune_scores"))
            tables["alphagrid"].append({**base, "method": family, "alpha_index": index, "alpha": alpha.get("alpha"),
                                        "selected": index == row.get("selected_index"), "model_sha256": alpha.get("model_sha256"),
                                        "coefficient_path": (alpha.get("coefficient_artifact") or {}).get("path"),
                                        "coefficient_sha256": (alpha.get("coefficient_artifact") or {}).get("sha256"),
                                        "fit_mse": fit_mse, "fit_mae": fit_mae, "tune_mse": tune_mse, "tune_mae": tune_mae,
                                        "tune_actual_Y_mse": alpha.get("tune_actual_Y_mse"),
                                        "diagnostic_only": diagnostic, "deployable": deployable})
        for split, score in (row.get("splits") or {}).items():
            add_score(tables["scores"], base, family, split, score, "native.family.selected",
                      {"selected_alpha": row.get("selected_alpha"), "selected_model_sha256": row.get("selected_model_sha256")})
    teacher = result.get("teacher_contract") or {}
    for horizon, counts in enumerate(teacher.get("fit_counts_by_horizon_K") or []):
        count = counts[0] if isinstance(counts, list) and counts else counts
        tables["coverage"].append({**base, "coverage_type": "N1_teacher_complete_future_prefix_fit", "horizon": horizon + 1,
                                   "channel": None, "count": count, "raw_counts_json": counts,
                                   "source": "native.teacher_contract.fit_counts_by_horizon_K",
                                   "diagnostic_scope": "window count per 1-based horizon; raw_counts_json preserves per-K repeated output counts"})
    for channel, count in enumerate(teacher.get("actual_Y_fit_channel_counts") or []):
        tables["coverage"].append({**base, "coverage_type": "actual_Y_fit_channel_observed", "horizon": None,
                                   "channel": channel, "count": count, "raw_counts_json": None,
                                   "source": "native.teacher_contract.actual_Y_fit_channel_counts",
                                   "diagnostic_scope": "actual target mask count, not teacher latent count"})
    add_final_deltas(dataset, seed, base, tables)


def add_final_deltas(dataset: str, seed: int, base: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    final = FINAL_SPLIT[dataset]
    subset = [r for r in tables["scores"] if r.get("dataset") == dataset and r.get("seed") == seed and r.get("split") == final]
    by_method = {r["method"]: r for r in subset}
    for method in DEPLOY:
        row = by_method.get(method)
        if not row:
            continue
        for reference in ("ORIGINAL_MAIN", "RAW_BOLT", "CURRENT_C_PERSISTENCE", "FIT_INTERCEPT_ONLY"):
            ref = by_method.get(reference)
            if ref:
                tables["stagedeltas"].append({**base, "method": method, "reference": reference, "split": final,
                                              "mse": row.get("mse"), "reference_mse": ref.get("mse"),
                                              "delta_mse": subtract(row.get("mse"), ref.get("mse")),
                                              "mae": row.get("mae"), "reference_mae": ref.get("mae"),
                                              "delta_mae": subtract(row.get("mae"), ref.get("mae"))})


def subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def collect_oracle(result: dict[str, Any], record: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    dataset, seed = result.get("dataset"), result.get("seed")
    if dataset not in DATASETS or seed not in SEEDS:
        raise ReportFailure("UNEXPECTED_ORACLE_RESULT_DATASET_SEED")
    base = {"dataset": dataset, "seed": seed, "oracle_sha256": record["sha256"], "oracle_path": record["path"]}
    tables["status"].append({**base, "component": "oracle_diagnostic_extractor", "status": require_terminal(result, "oracle_result"),
                             "diagnostic_only": result.get("diagnostic_only"),
                             "future_observations_used": result.get("future_observations_used"),
                             "past_only_forecasting_result": result.get("past_only_forecasting_result"),
                             "F_loaded": result.get("F_loaded"), "F_forward_replayed": result.get("F_forward_replayed"),
                             "failure": result.get("failure")})
    for split, coverage in (result.get("coverage") or {}).items():
        for horizon, count in enumerate(coverage.get("teacher_prefix_counts_by_horizon") or []):
            tables["coverage"].append({**base, "coverage_type": "oracle_extract_complete_future_prefix",
                                       "split": split, "horizon": horizon, "channel": None, "count": count,
                                       "raw_counts_json": count, "source": "oracle.coverage.teacher_prefix_counts_by_horizon",
                                       "diagnostic_scope": "future oracle diagnostic extractor; not deployment"})
        for channel, count in enumerate(coverage.get("original_Y_channel_counts") or []):
            tables["coverage"].append({**base, "coverage_type": "oracle_extract_actual_Y_channel",
                                       "split": split, "horizon": None, "channel": channel, "count": count,
                                       "raw_counts_json": None, "source": "oracle.coverage.original_Y_channel_counts",
                                       "diagnostic_scope": "target mask count from terminal JSON metadata"})


def collect_reused_raw_controls(audit: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    """Collect optional old RAW readout controls from the native audit supplement.

    These are reused historical controls, not new training in this report and not a
    matched-anchor/readout-capacity claim. The audit may omit them or mark them
    unavailable; both states are reported without fabricating scores.
    """
    for item in audit.get("reused_raw_controls") or []:
        dataset = item.get("dataset")
        if dataset not in DATASETS:
            continue
        source = item.get("source") or {}
        base = {"dataset": dataset, "seed": None, "producer_sha256": None,
                "producer_path": None, "raw_control_source_path": source.get("path"),
                "raw_control_source_sha256": source.get("sha256")}
        tables["status"].append({**base, "component": "reused_raw_control", "status": item.get("status"),
                                 "new_training": item.get("new_training"),
                                 "matched_readout_capacity_or_anchor": item.get("matched_readout_capacity_or_anchor"),
                                 "failure": item.get("failure")})
        methods = item.get("methods") or {}
        if isinstance(methods, dict):
            iterable = [(name, value) for name, value in methods.items()]
        else:
            iterable = [(row.get("method"), row) for row in methods if isinstance(row, dict)]
        for method, row in iterable:
            if not method or not isinstance(row, dict):
                continue
            tables["status"].append({**base, "component": "reused_raw_control_method", "method": method,
                                     "status": row.get("status", item.get("status")),
                                     "prediction_path": (row.get("prediction_record") or row.get("prediction") or {}).get("path"),
                                     "prediction_sha256": (row.get("prediction_record") or row.get("prediction") or {}).get("sha256"),
                                     "priortrainingmetadata": row.get("priortrainingmetadata", row.get("prior_training_metadata"))})
            for split, score in (row.get("splits") or {}).items():
                add_score(tables["scores"], base, method, split, score, "native_audit.reused_raw_controls",
                          {"role": "reused_RAW_control", "diagnostic_only": False, "deployable": True,
                           "new_training": row.get("new_training", item.get("new_training", False)),
                           "matched_readout_capacity_or_anchor": row.get("matched_readout_capacity_or_anchor", item.get("matched_readout_capacity_or_anchor", False))})


def recursive_sha_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower().endswith("sha256") and isinstance(nested, str):
                found.add(nested.upper())
            found |= recursive_sha_values(nested)
    elif isinstance(value, list):
        for item in value:
            found |= recursive_sha_values(item)
    return found


def validate_native_audit_binding(audit: dict[str, Any], native_records: list[dict[str, Any]]) -> dict[str, Any]:
    status = str(audit.get("status"))
    if not status.startswith("PASS") or audit.get("failed_count") != 0:
        raise ReportFailure("PASS_ZERO_FAILURE_NATIVE_AUDIT_REQUIRED")
    expected = {(r["dataset"], int(r["seed"]), r["sha256"].upper()) for r in native_records}
    conditions = audit.get("conditions") or []
    actual = {(r.get("dataset"), int(r.get("seed")), str(r.get("producer_sha256", "")).upper()) for r in conditions}
    if len(expected) != 4 or actual != expected:
        raise ReportFailure("EXACT4_NATIVE_AUDIT_PRODUCER_SHA_BINDING_REQUIRED")
    return {"passed": True, "status": status, "failed_count": audit.get("failed_count"),
            "conditions": len(conditions), "candidate_rows_checked": audit.get("candidate_rows_checked"),
            "selected_predictions_checked": audit.get("selected_predictions_checked"), "checks": audit.get("checks"),
            "preserved_validation_episodes": audit.get("preserved_validation_episodes", audit.get("failed_validation_episodes"))}


def validate_cost(cost: dict[str, Any] | None, cost_record: dict[str, Any] | None,
                  audit: dict[str, Any] | None, audit_record: dict[str, Any] | None) -> dict[str, Any]:
    if cost is None:
        return {"status": "NOT_MEASURED", "audit_passed": False, "rows": 0, "reason": "cost JSON not supplied"}
    status = require_terminal(cost, "cost_result")
    rows = cost.get("rows") or []
    out = {"status": status, "audit_passed": False, "rows": len(rows), "complete_rows": sum(r.get("status") == "COMPLETE" for r in rows),
           "planned_rows": cost.get("planned_rows"), "actual_complete_rows": cost.get("actual_complete_rows"),
           "reason": (cost.get("failure") or {}).get("message") or "terminal cost JSON supplied"}
    if status == "COMPLETE" and (len(rows) != 36 or cost.get("actual_complete_rows") != 36):
        raise ReportFailure("COMPLETE_STAGE_COST_REQUIRES_36_ROWS")
    if audit is None:
        out["reason"] = "cost audit JSON not supplied"
        return out
    audit_status = str(audit.get("status"))
    if not audit_status.startswith("PASS") or audit.get("failed_count") != 0:
        raise ReportFailure("PASS_ZERO_FAILURE_COST_AUDIT_REQUIRED")
    cost_sha = (cost_record or {}).get("sha256", "").upper()
    if cost_sha and cost_sha not in recursive_sha_values(audit):
        raise ReportFailure("COST_AUDIT_PRODUCER_SHA_BINDING_REQUIRED")
    out.update(audit_passed=True, audit_status=audit_status, cost_audit_sha256=(audit_record or {}).get("sha256"))
    return out


def collect_cost(cost: dict[str, Any] | None, gate: dict[str, Any]) -> list[dict[str, Any]]:
    if cost is None or gate["status"] != "COMPLETE":
        return [{"status": "NOT_MEASURED" if cost is None else gate["status"], "reason": gate.get("reason"),
                 "route": None, "dataset": None, "batch": None, "method": None, "seed": None,
                 "median_ms": None, "median_ms_per_window": None, "paired_RAW_median_ratio": None,
                 "repeats": 0, "original_bulk_flags_reported_only": True}]
    rows = []
    for row in cost.get("rows") or []:
        seconds = row.get("seconds") or []
        rows.append({"dataset": row.get("dataset"), "batch": row.get("batch"), "route": row.get("route"),
                     "status": row.get("status"), "method": row.get("method"), "seed": row.get("seed"),
                     "selected_alpha": row.get("selected_alpha"),
                     "selected_model_sha256": row.get("selected_model_sha256"),
                     "fit_zero_observed_output_counts": row.get("fit_zero_observed_output_counts"),
                     "repeats": len(seconds),
                     "median_ms": statistics.median(seconds) * 1000.0 if seconds else row.get("median_seconds"),
                     "median_ms_per_window": row.get("median_seconds_per_window") * 1000.0 if row.get("median_seconds_per_window") is not None else None,
                     "paired_RAW_median_ratio": row.get("paired_RAW_median_ratio"),
                     "paired_RAW_ratios": row.get("paired_RAW_ratios"), "blocks": row.get("block"),
                     "positions": row.get("position"),
                     "source_reference_path": (row.get("source_reference") or {}).get("path"),
                     "source_reference_sha256": (row.get("source_reference") or {}).get("sha256"),
                     "original_bulk_source_path": (row.get("original_bulk_source") or {}).get("path"),
                     "original_bulk_source_sha256": (row.get("original_bulk_source") or {}).get("sha256"),
                     "reference_scope": row.get("reference_scope"), "balanced_scope": row.get("balanced_scope"),
                     "includes": row.get("includes"), "excludes": row.get("excludes"),
                     "original_bulk_flags_reported_only": True})
    return rows


def build_tables(native_results: list[dict[str, Any]], native_records: list[dict[str, Any]],
                 oracle_results: list[dict[str, Any]], oracle_records: list[dict[str, Any]],
                 native_audit: dict[str, Any], cost_result: dict[str, Any] | None = None,
                 cost_audit: dict[str, Any] | None = None, cost_record: dict[str, Any] | None = None,
                 cost_audit_record: dict[str, Any] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(native_results) != 4 or len(oracle_results) != 4:
        raise ReportFailure("EXACT4_NATIVE_AND_ORACLE_RESULTS_REQUIRED")
    native_records = [{**r, "dataset": p.get("dataset"), "seed": p.get("seed")} for p, r in zip(native_results, native_records)]
    oracle_records = [{**r, "dataset": p.get("dataset"), "seed": p.get("seed")} for p, r in zip(oracle_results, oracle_records)]
    expected = {(d, s) for d in DATASETS for s in SEEDS}
    if {(p.get("dataset"), p.get("seed")) for p in native_results} != expected:
        raise ReportFailure("EXACT_NATIVE_DATASET_SEED_COVERAGE_REQUIRED")
    if {(p.get("dataset"), p.get("seed")) for p in oracle_results} != expected:
        raise ReportFailure("EXACT_ORACLE_DATASET_SEED_COVERAGE_REQUIRED")
    tables = empty_tables()
    for value, record in zip(native_results, native_records):
        collect_native(value, record, tables)
    for value, record in zip(oracle_results, oracle_records):
        collect_oracle(value, record, tables)
    gates = {"native_audit": validate_native_audit_binding(native_audit, native_records)}
    collect_reused_raw_controls(native_audit, tables)
    gates["cost"] = validate_cost(cost_result, cost_record, cost_audit, cost_audit_record)
    tables["cost"] = collect_cost(cost_result, gates["cost"])
    return tables, gates


def final_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("split") == FINAL_SPLIT.get(r.get("dataset"))]


def mean_metric(rows: list[dict[str, Any]], dataset: str, method: str, metric: str) -> float | None:
    values = [float(r[metric]) for r in rows if r.get("dataset") == dataset and r.get("method") == method and r.get(metric) is not None]
    return statistics.mean(values) if values else None


def fmt(value: Any, digits: int = 9) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def render_markdown(tables: dict[str, list[dict[str, Any]]], gates: dict[str, Any], manifest_name: str = "MANIFEST.json") -> str:
    rows = final_scores(tables["scores"])
    def cell(dataset: str, method: str, metric: str) -> str:
        return fmt(mean_metric(rows, dataset, method, metric))
    def delta_cell(dataset: str, method: str, reference: str, metric: str) -> str:
        left, right = mean_metric(rows, dataset, method, metric), mean_metric(rows, dataset, reference, metric)
        return fmt(left - right) if left is not None and right is not None else "NA"
    lines = [
        "# Stage separation 실제 결과 보고서",
        "",
        "[확인] 이 보고서는 terminal JSON만 읽어 작성되며, 과학 NPZ·checkpoint·F 모델을 열지 않는다. 하류 결정은 `E-F-D` 경로를 배포에 유지할 근거가 있는지와 future oracle 진단을 deploy 후보와 혼동하지 않는 것이다.",
        "",
        f"과학 감사 gate: native audit `{gates['native_audit']['status']}`, failed_count `{gates['native_audit']['failed_count']}`. 비용 gate: `{gates['cost']['status']}`, audit_passed `{gates['cost'].get('audit_passed')}`.",
        "",
        "## 먼저 답해야 할 여섯 질문",
        "",
        "```text",
        "질문                         Robin seed평균                         Beijing seed평균",
        f"1 deploy 후보 MSE             N1 {cell('robin','N1','mse')} / N2 {cell('robin','N2','mse')} / N3 {cell('robin','N3','mse')}    N1 {cell('beijing_aotizhongxin','N1','mse')} / N2 {cell('beijing_aotizhongxin','N2','mse')} / N3 {cell('beijing_aotizhongxin','N3','mse')}",
        f"2 N2−PERSISTENCE ΔMSE         {delta_cell('robin','N2','CURRENT_C_PERSISTENCE','mse')}                         {delta_cell('beijing_aotizhongxin','N2','CURRENT_C_PERSISTENCE','mse')}",
        f"2 N2−FIT_INTERCEPT ΔMSE       {delta_cell('robin','N2','FIT_INTERCEPT_ONLY','mse')}                         {delta_cell('beijing_aotizhongxin','N2','FIT_INTERCEPT_ONLY','mse')}",
        f"2 N2−RAW_BOLT ΔMSE            {delta_cell('robin','N2','RAW_BOLT','mse')}                         {delta_cell('beijing_aotizhongxin','N2','RAW_BOLT','mse')}",
        f"3 O0/O1 future diag MSE        O0 {cell('robin','O0_ORACLE_OLD_D','mse')} / O1 {cell('robin','O1','mse')}       O0 {cell('beijing_aotizhongxin','O0_ORACLE_OLD_D','mse')} / O1 {cell('beijing_aotizhongxin','O1','mse')}",
        f"4 N3−N1 fixed predictor ΔMSE   {delta_cell('robin','N3','N1','mse')}                         {delta_cell('beijing_aotizhongxin','N3','N1','mse')}",
        "5 delayed-info lag ablation     NOT_IDENTIFIED_IN_THIS_REPORT        NOT_IDENTIFIED_IN_THIS_REPORT",
        "6 contrastive/dynamic graph     NOT_ESTABLISHED_BY_THIS_EXPERIMENT   NOT_ESTABLISHED_BY_THIS_EXPERIMENT",
        "```",
        "",
        "개별 seed와 split별 숫자는 [scores.csv](tables/scores.csv)와 [stagedeltas.csv](tables/stagedeltas.csv)에 있다. causal prefix check는 producer/audit의 특정 입력 스코프 검산이며 보편적 누수 부재 증명이 아니다. N2가 PERSISTENCE/FIT_INTERCEPT/RAW 또는 재사용 RAW readout control을 이기면 이 조건의 past 33K readout usefulness를 지지하지만, encoder class 필요성·독립 일반화·contrastive/dynamic graph 필요성을 증명하지 않는다. Chronos/고정 E 교체의 가치는 이 fixed split 비교 범위에 한정한다.",
        "",
        "## 1. 배포 후보 N1/N2/N3와 원 RAW/MAIN 비교",
        "",
        "```text",
        "dataset                  method                 n   final normalized MSE / MAE",
    ]
    for dataset in DATASETS:
        for method in ("RAW_BOLT", "ORIGINAL_MAIN", "CURRENT_C_PERSISTENCE", "FIT_INTERCEPT_ONLY", "N1", "N2", "N3"):
            selected = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]
            mse = statistics.mean([float(r["mse"]) for r in selected if r.get("mse") is not None]) if selected else None
            mae = statistics.mean([float(r["mae"]) for r in selected if r.get("mae") is not None]) if selected else None
            lines.append(f"{dataset:24} {method:22} {len(selected):1d}   {fmt(mse)} / {fmt(mae)}")
        for method in ("OLD_SAME_MAIN_NATIVE", "OLD_F_REFIT_D_Y"):
            selected = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]
            if selected:
                mse = statistics.mean([float(r["mse"]) for r in selected if r.get("mse") is not None])
                mae = statistics.mean([float(r["mae"]) for r in selected if r.get("mae") is not None])
                lines.append(f"{dataset:24} {method:22} {len(selected):1d}   {fmt(mse)} / {fmt(mae)}")
    lines += ["```", "", "세부 split/seed 수치는 [scores.csv](tables/scores.csv)에 보존한다. RAW의 seed 반복은 같은 저장 RAW 예측 reference를 각 seed 비교에 붙인 것이며 독립 RAW 재학습 2회가 아니다.", ""]
    for dataset in DATASETS:
        best = sorted((mean_metric(rows, dataset, method, "mse"), method) for method in DEPLOY if mean_metric(rows, dataset, method, "mse") is not None)
        main = mean_metric(rows, dataset, "ORIGINAL_MAIN", "mse")
        if best and main is not None:
            lines.append(f"- {DISPLAY[dataset]}: final split 평균 MSE 기준 최저 배포 후보는 `{best[0][1]}`이고 원 MAIN 대비 ΔMSE는 `{best[0][0] - main:+.9f}`다.")
        else:
            lines.append(f"- {DISPLAY[dataset]}: 최저 배포 후보를 계산할 충분한 terminal score가 없다.")
    reused = [r for r in rows if r.get("role") == "reused_RAW_control"]
    if reused:
        lines += ["", "old RAW readout reuse control도 native audit supplement에서 들어온 경우만 별도 표시한다. 이는 새 훈련 비용이나 matched-anchor/readout capacity 주장이 아니다.", "", "```text",
                  "dataset                  method                 n   final normalized MSE / MAE"]
        for dataset in DATASETS:
            for method in sorted({r["method"] for r in reused if r.get("dataset") == dataset}):
                selected = [r for r in reused if r.get("dataset") == dataset and r.get("method") == method]
                mse = statistics.mean([float(r["mse"]) for r in selected if r.get("mse") is not None]) if selected else None
                mae = statistics.mean([float(r["mae"]) for r in selected if r.get("mae") is not None]) if selected else None
                lines.append(f"{dataset:24} {method:22} {len(selected):1d}   {fmt(mse)} / {fmt(mae)}")
        lines += ["```"]
    episodes = gates["native_audit"].get("preserved_validation_episodes")
    if episodes:
        lines += ["", f"보존된 이전 validation failure episode가 audit JSON에 기록되어 있다. 여기에는 v2 raw SHA/hash-algorithm mismatch 계열과 v3 Torch GELU/erf arithmetic mismatch 계열 같은 source/audit 수정 이력이 포함될 수 있다. 이 보고서는 해당 episode를 성공으로 바꾸지 않고, 최종 입력 audit gate `{gates['native_audit']['status']}`와 함께 [status.csv](tables/status.csv)에 현재 terminal 상태를 둔다."]
    lines += [
        "",
        "## 2. O0/O1은 actual future를 쓰는 진단이다",
        "",
        "O0_ORACLE_OLD_D와 O1은 actual future oracle latent 또는 actual future prefix를 쓰는 diagnostic branch다. 배포 후보 N1/N2/N3와 같은 입력 조건이 아니므로 같은 축의 원인 주장으로 읽지 않는다.",
        "",
        "## 3. N2 우세가 말하는 범위",
        "",
        "N2는 past latent에서 C-space innovation을 직접 예측한다. 같은 anchor와 FIT 통계 아래에서 N2가 PERSISTENCE/FIT_INTERCEPT/RAW보다 낮은 오류를 보이면 이 조건의 past 33K readout usefulness는 지지한다. 다만 그 결과만으로 encoder class 필요성, 독립 일반화, delayed-information 원인, contrastive/dynamic graph 필요성을 주장하지 않는다. N1 latent objective도 actual-Y loss와 다르고 complete-future-prefix mask 표본과 actual-Y 관측 표본이 다를 수 있다.",
        "",
        "## 4. teacher coverage와 zero-output 위험",
        "",
        "N1 teacher는 complete future prefix mask로 학습된다. FIT teacher count와 actual-Y channel count는 [coverage.csv](tables/coverage.csv), zero-output 수는 [zero_outputs.csv](tables/zero_outputs.csv)에 보존한다. zero-output teacher output은 학습된 출력이 아니므로 성공 사례로 해석하지 않는다.",
        "",
        "## 5. 비용 측정의 범위",
        "",
        f"비용 상태는 `{gates['cost']['status']}`다. cost rows는 [cost.csv](tables/cost.csv)에 둔다. 비용 producer는 B1/B8 whole route paired RAW 비교이며, hybrid transfer와 staged decomposition 부재 때문에 원인 분해나 최적 구현 비용으로 일반화하지 않는다.",
        "",
        "## 6. 제한",
        "",
        "- 반복 개발 Robin VAL30과 이미 노출된 Beijing EVAL146, seed 2개만 포함한다. 독립 holdout 주장이 아니다.",
        "- 원래 MAIN은 E/D/F/Y end-to-end 예측기로 훈련된 것이며 autoencoder가 아니다. stage separation 수치로 F의 인과 실패를 직접 증명하지 않는다.",
        "- PERSISTENCE와 FIT_INTERCEPT는 단순 배포 control로 포함하지만 새 모델 선택 기준을 바꾸지 않는다.",
        "- 본 생성기는 terminal JSON만 읽는다. NPZ 배열과 checkpoint byte-level 검증은 native audit 결과에 의존한다.",
        "",
        "## 산출물",
        "",
        "- 표: [scores.csv](tables/scores.csv), [alphagrid.csv](tables/alphagrid.csv), [coverage.csv](tables/coverage.csv), [zero_outputs.csv](tables/zero_outputs.csv), [cost.csv](tables/cost.csv), [status.csv](tables/status.csv), [stagedeltas.csv](tables/stagedeltas.csv)",
        "![deploy 점수](figures/01_deploy_mse_mae_dots.png)",
        "![oracle 진단](figures/02_oracle_diagnostic_scores.png)",
        "![TUNE alpha](figures/03_tune_alpha_grid.png)",
        "![teacher coverage](figures/04_teacher_coverage_counts.png)",
        "![fit tune eval overfit](figures/05_fit_tune_eval_overfit.png)",
        "![paired cost](figures/06_b1_b8_paired_cost.png)",
        f"- manifest: `{manifest_name}`",
        "",
    ]
    text = "\n".join(lines)
    if ".cache" in text:
        raise ReportFailure("MARKDOWN_MUST_NOT_LINK_IGNORED_CACHE_PATHS")
    return text


def save_figure(fig: Any, folder: Path, name: str, plt: Any) -> list[dict[str, Any]]:
    records = []
    for suffix in ("png", "svg"):
        path = folder / f"{name}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        records.append({"path": str(path.resolve()), "sha256": sha_file(path), "format": suffix})
    plt.close(fig)
    return records


def placeholder(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_figures(tables: dict[str, list[dict[str, Any]]], gates: dict[str, Any], folder: Path, plt: Any, check=None) -> list[dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    colors = {"RAW_BOLT": "#333333", "ORIGINAL_MAIN": "#777777", "CURRENT_C_PERSISTENCE": "#009E73",
              "FIT_INTERCEPT_ONLY": "#CC79A7", "N1": "#0072B2", "N2": "#D55E00", "N3": "#E69F00",
              "O0_ORACLE_OLD_D": "#56B4E9", "O1": "#CC6677"}
    records: list[dict[str, Any]] = []
    rows = final_scores(tables["scores"])
    panels = [
        (FIGURES[0], ("RAW_BOLT", "ORIGINAL_MAIN", "CURRENT_C_PERSISTENCE", "FIT_INTERCEPT_ONLY", "N1", "N2", "N3"), "Deploy candidates and controls"),
        (FIGURES[1], ("O0_ORACLE_OLD_D", "O1", "ORIGINAL_MAIN"), "O0/O1 use future diagnostics"),
    ]
    for name, methods, title in panels:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for di, dataset in enumerate(DATASETS):
            for mi, metric in enumerate(("mse", "mae")):
                ax = axes[di][mi]
                for idx, method in enumerate(methods):
                    values = [float(r[metric]) for r in rows if r.get("dataset") == dataset and r.get("method") == method and r.get(metric) is not None]
                    if values:
                        ax.scatter([idx] * len(values), values, color=colors.get(method, "#444444"), s=30)
                        ax.plot([idx - .18, idx + .18], [statistics.mean(values)] * 2, color="black")
                ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
                ax.set_title(f"{DISPLAY[dataset]} {metric.upper()}")
                ax.set_ylabel("normalized observed channel-macro")
        fig.suptitle(title)
        records.extend(save_figure(fig, folder, name, plt))
        if check:
            check()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for di, dataset in enumerate(DATASETS):
        for fi, family in enumerate(("N1", "N2")):
            ax = axes[di][fi]
            family_rows = [r for r in tables["alphagrid"] if r.get("dataset") == dataset and r.get("method") == family]
            if not family_rows:
                placeholder(ax, "no alpha rows")
                continue
            for seed in SEEDS:
                seed_rows = sorted([r for r in family_rows if r.get("seed") == seed], key=lambda r: r.get("alpha_index") or 0)
                ax.plot([str(r.get("alpha")) for r in seed_rows], [r.get("tune_actual_Y_mse") for r in seed_rows], marker="o", label=str(seed))
            ax.set_title(f"{DISPLAY[dataset]} {family} TUNE MSE")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=8)
    records.extend(save_figure(fig, folder, FIGURES[2], plt))
    if check:
        check()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for di, dataset in enumerate(DATASETS):
        prefix = [r for r in tables["coverage"] if r.get("dataset") == dataset and r.get("coverage_type") == "N1_teacher_complete_future_prefix_fit"]
        actual = [r for r in tables["coverage"] if r.get("dataset") == dataset and r.get("coverage_type") == "actual_Y_fit_channel_observed"]
        ax = axes[di][0]
        if prefix:
            by_h: dict[int, list[float]] = {}
            for row in prefix:
                by_h.setdefault(int(row["horizon"]), []).append(float(row["count"]))
            xs = sorted(by_h)
            ax.plot(xs, [statistics.mean(by_h[x]) for x in xs])
            ax.set_title(f"{DISPLAY[dataset]} teacher prefix count")
        else:
            placeholder(ax, "teacher counts unavailable")
        ax = axes[di][1]
        if actual:
            by_c: dict[int, list[float]] = {}
            for row in actual:
                by_c.setdefault(int(row["channel"]), []).append(float(row["count"]))
            xs = sorted(by_c)
            ax.bar(xs, [statistics.mean(by_c[x]) for x in xs])
            ax.set_title(f"{DISPLAY[dataset]} actual-Y channel count")
        else:
            placeholder(ax, "actual-Y counts unavailable")
    records.extend(save_figure(fig, folder, FIGURES[3], plt))
    if check:
        check()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for di, dataset in enumerate(DATASETS):
        for mi, metric in enumerate(("mse", "mae")):
            ax = axes[di][mi]
            for idx, split in enumerate(("fit", "tune", FINAL_SPLIT[dataset])):
                values = [float(r[metric]) for r in tables["scores"] if r.get("dataset") == dataset and r.get("method") in DEPLOY and r.get("split") == split and r.get(metric) is not None]
                if values:
                    ax.scatter([idx] * len(values), values, color="#0072B2", s=22)
                    ax.plot([idx - .15, idx + .15], [statistics.mean(values)] * 2, color="black")
            ax.set_xticks(range(3), ["fit", "tune", FINAL_SPLIT[dataset]])
            ax.set_title(f"{DISPLAY[dataset]} selected deploy {metric.upper()}")
    records.extend(save_figure(fig, folder, FIGURES[4], plt))
    if check:
        check()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di][bi]
            cost_rows = [r for r in tables["cost"] if r.get("dataset") == dataset and r.get("batch") == batch and r.get("paired_RAW_median_ratio") is not None]
            if not cost_rows:
                placeholder(ax, f"cost {gates['cost']['status']}")
                continue
            ax.bar(range(len(cost_rows)), [float(r["paired_RAW_median_ratio"]) for r in cost_rows], color="#666666")
            ax.axhline(1.0, color="black", linewidth=.8)
            ax.set_xticks(range(len(cost_rows)), [r.get("route") for r in cost_rows], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{DISPLAY[dataset]} B{batch} / paired RAW ratio")
    records.extend(save_figure(fig, folder, FIGURES[5], plt))
    if check:
        check()
    return records


def copy_inputs(records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    folder = output / "provenance" / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, record in enumerate(records, 1):
        source = Path(record["path"])
        destination = folder / f"{index:02d}_{record['role']}_{source.name}"
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append({**record, "copied_path": str(destination.resolve()), "copied_sha256": sha_file(destination)})
    return copied


def copy_sources(output: Path) -> list[dict[str, Any]]:
    folder = output / "provenance" / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve()] + [HERE / rel for rel in SNAPSHOT_SOURCES if (HERE / rel).is_file()]
    out = []
    for path in dict.fromkeys(p.resolve() for p in paths):
        sha = sha_file(path)
        destination = folder / path.name
        counter = 1
        while destination.exists():
            destination = folder / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        with path.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        out.append({"path": str(path), "sha256": sha, "snapshot_path": str(destination.resolve()), "snapshot_sha256": sha_file(destination)})
    return out


def artifact_inventory(output: Path) -> list[dict[str, Any]]:
    artifacts = []
    manifest_path = output / "MANIFEST.json"
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.resolve() == manifest_path.resolve():
            continue
        artifacts.append({
            "path": str(path.relative_to(output)).replace("\\", "/"),
            "sha256": sha_file(path),
            "bytes": path.stat().st_size,
        })
    return artifacts


def build_final_manifest(result: dict[str, Any], manifest_meta: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        **manifest_meta,
        "generator_result_status": result.get("status"),
        "artifacts": artifact_inventory(output),
        "result": {
            "path": "RESULT.json" if (output / "RESULT.json").is_file() else None,
            "sha256": sha_file(output / "RESULT.json") if (output / "RESULT.json").is_file() else None,
        },
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import stage_separation_runtime_v1 as runtime
    limit = runtime.ContinuationLimit("stage_report_" + safe_tag(args.out), gpu=False)
    output = Path(args.out).resolve()
    status = "RUNNING"
    error_payload = None
    initialized = False
    started_perf = time.perf_counter()
    manifest_meta: dict[str, Any] = {
        "terminal_JSON_only": True,
        "scientific_NPZ_or_CP_opened": False,
        "quality_context": {
            "domain": "time-series forecasting / old MAIN stage-separation diagnostics",
            "downstream_decision": "whether past-only N1/N2/N3 deployment candidates justify E-F-D separation",
            "leakage_guard": "O0/O1 future diagnostics are separated from deployable N1/N2/N3; no independent holdout is claimed",
        },
    }
    result: dict[str, Any] = {"status": status, "GPU_execution": False, "scientific_NPZ_or_CP_opened": False,
                              "terminal_JSON_only": True, "whole_goal_complete": False,
                              "authorization": limit.authorization_record, "run_manifest": limit.run_manifest_record}
    try:
        output.mkdir(parents=True, exist_ok=False)
        initialized = True
        (output / "tables").mkdir()
        (output / "figures").mkdir()
        (output / "provenance").mkdir()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        native_values, native_records, oracle_values, oracle_records, input_records = [], [], [], [], []
        for path in args.native_results:
            value, record = checked_json(path, "native_result")
            native_values.append(value)
            native_records.append(record)
            input_records.append(record)
        for path in args.oracle_results:
            value, record = checked_json(path, "oracle_result")
            oracle_values.append(value)
            oracle_records.append(record)
            input_records.append(record)
        native_audit, native_audit_record = checked_json(args.native_audit, "native_audit")
        cost_result, cost_record = checked_json(args.cost_result, "cost_result")
        cost_audit, cost_audit_record = checked_json(args.cost_audit, "cost_audit")
        input_records.extend([native_audit_record, cost_record, cost_audit_record])
        tables, gates = build_tables(native_values, native_records, oracle_values, oracle_records, native_audit, cost_result, cost_audit, cost_record, cost_audit_record)
        limit.checkfull()
        table_records = {
            "scores": write_csv_new(output / "tables" / "scores.csv", tables["scores"], ["dataset", "seed", "method", "split", "role", "diagnostic_only", "deployable", "mse", "mae", "physical_mse_diagnostic", "physical_mae_diagnostic", "selected_alpha", "selected_model_sha256", "producer_sha256", "source"]),
            "alphagrid": write_csv_new(output / "tables" / "alphagrid.csv", tables["alphagrid"], ["dataset", "seed", "method", "alpha_index", "alpha", "selected", "fit_mse", "fit_mae", "tune_mse", "tune_mae", "tune_actual_Y_mse", "model_sha256", "producer_sha256"]),
            "coverage": write_csv_new(output / "tables" / "coverage.csv", tables["coverage"], ["dataset", "seed", "coverage_type", "split", "horizon", "channel", "count", "source", "diagnostic_scope", "producer_sha256", "oracle_sha256"]),
            "zero_outputs": write_csv_new(output / "tables" / "zero_outputs.csv", tables["zero_outputs"], ["dataset", "seed", "method", "zero_observed_output_counts", "factorization_groups", "diagnostic_only", "deployable", "producer_sha256"]),
            "cost": write_csv_new(output / "tables" / "cost.csv", tables["cost"], ["dataset", "batch", "route", "status", "method", "seed", "median_ms", "median_ms_per_window", "paired_RAW_median_ratio", "repeats", "selected_alpha", "selected_model_sha256", "original_bulk_flags_reported_only", "reason"]),
            "status": write_csv_new(output / "tables" / "status.csv", tables["status"], ["dataset", "seed", "component", "method", "status", "failure", "diagnostic_only", "deployable", "producer_sha256", "oracle_sha256"]),
            "stagedeltas": write_csv_new(output / "tables" / "stagedeltas.csv", tables["stagedeltas"], ["dataset", "seed", "method", "reference", "split", "mse", "reference_mse", "delta_mse", "mae", "reference_mae", "delta_mae", "producer_sha256"]),
        }
        limit.checkfull()
        figure_records = generate_figures(tables, gates, output / "figures", plt, check=limit.checkfull)
        limit.checkfull()
        limit.checkfull()
        copied_inputs = copy_inputs(input_records, output)
        limit.checkfull()
        source_snapshots = copy_sources(output)
        manifest_meta.update({"generator": {"path": str(Path(__file__).resolve()), "sha256": sha_file(__file__)},
                              "native_audit_gate": gates["native_audit"], "cost_gate": gates["cost"],
                              "input_records": input_records, "copied_inputs": copied_inputs,
                              "source_snapshots": source_snapshots, "tables": table_records, "figures": figure_records,
                              "figure_pair_count": len(figure_records) // 2})
        limit.checkfull()
        (output / "REPORT_KO.md").write_text(render_markdown(tables, gates, "MANIFEST.json"), encoding="utf-8")
        (output / "README.md").write_text("# Stage separation report bundle\n\nTerminal JSON only. See `REPORT_KO.md`, `tables/`, `figures/`, and `provenance/`.\n", encoding="utf-8")
        limit.checkfull()
        status = "COMPLETE"
        result.update(status=status, output=str(output),
                      report={"path": str((output / "REPORT_KO.md").resolve()), "sha256": sha_file(output / "REPORT_KO.md")},
                      readme={"path": str((output / "README.md").resolve()), "sha256": sha_file(output / "README.md")},
                      tables=table_records, figures=figure_records, figure_pair_count=len(figure_records) // 2, gates=gates)
    except Exception as error:
        status = "BUDGET_HELD_PRESERVED" if type(error).__name__ == "BudgetHeld" else "FAILED_PRESERVED"
        error_payload = {"type": type(error).__name__, "message": str(error)}
        result.update(status=status, output=str(output), error=error_payload)
    finally:
        try:
            receipt = limit.close(status if status != "RUNNING" else "FAILED_PRESERVED", error_payload)
        except Exception as close_error:
            receipt = {"terminal_status": "FAILED_PRESERVED", "terminal_error": {"type": type(close_error).__name__, "message": str(close_error)}}
        result["resource_receipt"] = receipt
        if receipt.get("terminal_status", result["status"]) != "COMPLETE":
            result["status"] = receipt.get("terminal_status", result["status"])
            if receipt.get("terminal_error") is not None:
                result["resource_terminal_error"] = receipt["terminal_error"]
        result["finished_unix"] = time.time()
        result["body_wall_seconds"] = time.perf_counter() - started_perf
        if initialized and not (output / "RESULT.json").exists():
            write_json_new(output / "RESULT.json", result)
        if initialized and not (output / "MANIFEST.json").exists():
            manifest = build_final_manifest(result, manifest_meta, output)
            manifest_record = write_json_new(output / "MANIFEST.json", manifest)
            result["manifest"] = manifest_record
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--native-results", nargs="+")
    parser.add_argument("--oracle-results", nargs="+")
    parser.add_argument("--native-audit")
    parser.add_argument("--cost-result")
    parser.add_argument("--cost-audit")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({"status": "SOURCE_ONLY_NOT_EXECUTED", "terminal_JSON_only": True,
                          "scientific_NPZ_or_CP_opened": False, "figures_png_svg_pairs": 6,
                          "tables": ["scores", "alphagrid", "coverage", "zero_outputs", "cost", "status", "stagedeltas"],
                          "runtime_before_matplotlib_and_IO": True,
                          "scope": "stage separation report: O0/O1 future diagnostics separated from deployable N1/N2/N3"}, ensure_ascii=False))
        return 0
    if not (args.native_results and len(args.native_results) == 4 and args.oracle_results and len(args.oracle_results) == 4 and args.native_audit and args.cost_result and args.cost_audit and args.out):
        parser.error("execute requires 4 native results, 4 oracle results, native audit, cost result, cost audit, and fresh output directory")
    result = generate(args)
    print(json.dumps({"status": result.get("status"), "output": result.get("output"),
                      "figure_pair_count": result.get("figure_pair_count"), "error": result.get("error")}, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

