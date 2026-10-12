"""Build the final v4 diagnosis report from actual saved artifacts only.

This helper is intentionally inert unless called with ``--run``.  The root
agent should run it only after the remaining science jobs are ready to adopt.
It never writes the existing EVIDENCE/REPORT/metrics/figures files; all outputs
go to a fresh v4 subdirectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
from common import Budget, access_guard  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_existing(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        root_path = ROOT / path
        path = root_path if root_path.exists() else HERE / path
    return path


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def fmt(value: Any, digits: int = 6) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}g}"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def rel_from(base_file_or_dir: Path, target: Path | str) -> str:
    """Relative link target from a Markdown file's directory or output dir."""
    base = Path(base_file_or_dir)
    if base.suffix:
        base = base.parent
    return Path(os.path.relpath(Path(target).resolve(), base.resolve())).as_posix()


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "relative_path": rel(path),
        "exists": path.exists(),
        "sha256": sha256(path) if path.exists() and path.is_file() else None,
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "mtime_unix": path.stat().st_mtime if path.exists() else None,
    }


def load_verification_proof(path_value: str, expected_sha: str) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = resolve_existing(path_value)
    actual = sha256(path)
    if expected_sha and actual != expected_sha:
        raise ValueError(f"verification proof sha mismatch: {actual} != {expected_sha}")
    proof = read_json(path)
    proof["_proof_path"] = str(path)
    proof["_proof_sha256"] = actual
    return proof


def file_sha_binding_check(path_value: Any, expected_sha: Any, kind: str) -> dict[str, Any]:
    if not path_value or not expected_sha:
        return {"kind": kind, "path": str(path_value or ""), "status": "UNCHECKED", "reason": "missing path or expected sha"}
    path = resolve_existing(str(path_value))
    if not path.exists() or not path.is_file():
        return {"kind": kind, "path": rel(path), "status": "FAIL", "expected_sha256": expected_sha, "exists": False}
    current = sha256(path)
    return {"kind": kind, "path": rel(path), "status": "PASS" if current == expected_sha else "FAIL", "expected_sha256": expected_sha, "current_sha256": current, "exists": True}


def npz_array_hash_binding_checks(path_value: Any, expected_hashes: dict[str, Any] | None, kind: str) -> list[dict[str, Any]]:
    if not path_value or not expected_hashes:
        return []
    path = resolve_existing(str(path_value))
    checks: list[dict[str, Any]] = []
    if not path.exists():
        return [{"kind": kind, "path": rel(path), "status": "FAIL", "reason": "NPZ missing for array hash binding"}]
    try:
        with np.load(path, allow_pickle=False) as z:
            for key, expected in sorted(expected_hashes.items()):
                if key not in z.files:
                    checks.append({"kind": kind, "path": rel(path), "array": key, "status": "FAIL", "reason": "array missing"})
                else:
                    current = array_hash(z[key])
                    checks.append({"kind": kind, "path": rel(path), "array": key, "status": "PASS" if current == expected else "FAIL", "expected_hash": expected, "current_hash": current})
    except Exception as exc:
        checks.append({"kind": kind, "path": rel(path), "status": "FAIL", "reason": str(exc)})
    return checks


def all_pass(checks: list[dict[str, Any]]) -> bool:
    return bool(checks) and all(c.get("status") == "PASS" for c in checks)


def e34_case_binding_checks(case: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [file_sha_binding_check(case.get("path"), case.get("sha256"), "E3_E4_case_json")]
    for method in case.get("methods", []) or []:
        checks.append(file_sha_binding_check(method.get("prediction_path"), method.get("prediction_sha256_current"), "E3_E4_prediction_npz"))
        rec = method.get("reconstruction_npz_verification") or {}
        if rec.get("path"):
            checks.append(file_sha_binding_check(rec.get("path"), rec.get("sha256"), "E3_reconstruction_npz"))
        latent = method.get("latent_metric_recalculation") or {}
        if latent.get("latent_path"):
            checks.append(file_sha_binding_check(latent.get("latent_path"), latent.get("latent_sha256_current"), "E4_latent_prediction_npz"))
        if latent.get("reference_latent_path"):
            checks.append(file_sha_binding_check(latent.get("reference_latent_path"), latent.get("reference_latent_sha256_current"), "E4_reference_latent_npz"))
    return checks


def e5_condition_binding_checks(cond: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [file_sha_binding_check(run.get("path"), run.get("sha256"), "E5_result_json"),
              file_sha_binding_check(cond.get("child_path"), cond.get("child_sha256_current"), "E5_child_json"),
              file_sha_binding_check(cond.get("prediction_path"), cond.get("prediction_sha256_current"), "E5_prediction_npz")]
    probe = cond.get("reconstruction_probe_npz_verification") or {}
    if probe.get("path"):
        checks.append(file_sha_binding_check(probe.get("path"), probe.get("sha256"), "E5_reconstruction_probe_npz"))
    return checks


def bolt_run_binding_checks(run: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [file_sha_binding_check(run.get("path"), run.get("sha256"), "E5_BOLT_result_json")]
    for cond in run.get("conditions", []) or []:
        checks.append(file_sha_binding_check(cond.get("child_path"), cond.get("child_sha256_current"), "E5_BOLT_child_json"))
        for method in cond.get("methods", []) or []:
            checks.append(file_sha_binding_check(method.get("prediction_path"), method.get("prediction_sha256_current"), "E5_BOLT_prediction_npz"))
    return checks


def e6_run_binding_checks(run: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [file_sha_binding_check(run.get("path"), run.get("sha256"), "E6_result_json")]
    for graph in (run.get("graph_arrays") or {}).values():
        if isinstance(graph, dict):
            checks.append(file_sha_binding_check(graph.get("path"), graph.get("sha256"), "E6_graph_npz"))
            checks.extend(npz_array_hash_binding_checks(graph.get("path"), graph.get("array_hashes"), "E6_graph_npz_array"))
    for baseline in (run.get("matched_baselines") or {}).values():
        arrays = baseline.get("arrays") if isinstance(baseline, dict) else None
        if isinstance(arrays, dict):
            checks.append(file_sha_binding_check(arrays.get("path"), baseline.get("array_sha256_current") or arrays.get("sha256"), "E6_baseline_npz"))
            checks.extend(npz_array_hash_binding_checks(arrays.get("path"), arrays.get("array_hashes"), "E6_baseline_npz_array"))
    return checks


def e6_condition_binding_checks(cond: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *e6_run_binding_checks(run),
        file_sha_binding_check(cond.get("child_path"), cond.get("child_sha256_current"), "E6_child_json"),
        file_sha_binding_check(cond.get("prediction_path"), cond.get("prediction_sha256_current"), "E6_prediction_npz"),
    ]



def cost_run_binding_checks(run: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [file_sha_binding_check(run.get("path"), run.get("sha256"), "COST_CLOSURE_result_json")]
    for key, kind in (("plan_hash_check", "COST_CLOSURE_plan"), ("gate_hash_check", "COST_CLOSURE_gate"), ("budget_ledger_hash_check", "COST_CLOSURE_budget_ledger")):
        obj = run.get(key) if isinstance(run.get(key), dict) else {}
        checks.append(file_sha_binding_check(obj.get("path"), obj.get("expected_sha256"), kind))
    for rec in value_at(run, "source_pin_check", "records") or []:
        checks.append(file_sha_binding_check(rec.get("path"), rec.get("expected_sha256"), "COST_CLOSURE_source_pin"))
    return checks

def verification_allowed_sources(proof: dict[str, Any] | None) -> dict[str, Any]:
    if not proof:
        return {"status": "UNVERIFIED", "allowed": None, "excluded": [], "binding_checks": []}
    allowed: set[str] = set()
    excluded: list[dict[str, Any]] = []
    binding_checks: list[dict[str, Any]] = []
    e34_reconstruction_audit: list[dict[str, Any]] = []
    e34_originwise_companion_count = 0

    def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = [str(c.get("status")) for c in checks]
        return {
            "count": len(checks),
            "pass": sum(1 for s in statuses if s == "PASS"),
            "fail": sum(1 for s in statuses if s == "FAIL"),
            "unchecked": sum(1 for s in statuses if s == "UNCHECKED"),
        }

    def add_path(path_value: Any, status: str, kind: str, checks: list[dict[str, Any]] | None = None) -> None:
        if not path_value:
            excluded.append({"path": "", "kind": kind, "verification_status": status or "UNCHECKED", "reason": "missing path"})
            return
        path = resolve_existing(str(path_value))
        rel_path = rel(path)
        checks = checks or []
        binding_checks.extend(checks)
        binding_ok = all_pass(checks) if checks else True
        if status == "PASS" and binding_ok:
            allowed.add(rel_path)
        else:
            failed = [c for c in checks if c.get("status") != "PASS"][:12]
            excluded.append({
                "path": rel_path,
                "kind": kind,
                "verification_status": status,
                "binding_status": "PASS" if binding_ok else "FAIL_OR_UNCHECKED",
                "binding_summary": summarize_checks(checks),
                "binding_failures_preview": failed,
            })

    for case in value_at(proof, "E3_E4", "cases") or []:
        case_checks = e34_case_binding_checks(case)
        add_path(case.get("path"), case.get("verification_status"), "E3_E4_case", case_checks)
        for method in case.get("methods", []) or []:
            rec = method.get("reconstruction_npz_verification") or {}
            if rec:
                e34_reconstruction_audit.append({
                    "case_path": case.get("path"),
                    "dataset": case.get("dataset"),
                    "dim": case.get("dim"),
                    "seed": case.get("seed"),
                    "method": method.get("method"),
                    "status": rec.get("status"),
                    "path": rec.get("path"),
                    "sha256": rec.get("sha256"),
                    "reported_metric_max_abs_diff": rec.get("reported_metric_max_abs_diff"),
                    "metric_tolerance": rec.get("metric_tolerance"),
                    "calculated": rec.get("calculated"),
                    "post_hoc_slice_metrics": rec.get("post_hoc_slice_metrics"),
                    "slice_note": rec.get("slice_note"),
                })
            if method.get("originwise_scores"):
                e34_originwise_companion_count += 1
    for run in value_at(proof, "E5", "runs") or []:
        run_status = run.get("verification_status")
        for cond in run.get("conditions", []):
            child_status = "PASS" if run_status == "PASS" and cond.get("verification_status") == "PASS" else (run_status or cond.get("verification_status"))
            add_path(cond.get("child_path"), child_status, "E5_condition", e5_condition_binding_checks(cond, run))
    for run in value_at(proof, "E5_BOLT", "runs") or []:
        if run.get("cost_only_not_science") or str(run.get("verification_status")) == "COST_ONLY_PROFILE_NOT_SCIENCE":
            continue
        add_path(run.get("path"), run.get("verification_status"), "E5_BOLT_result", bolt_run_binding_checks(run))
    for run in value_at(proof, "E6", "runs") or []:
        run_status = run.get("verification_status")
        run_checks = e6_run_binding_checks(run)
        add_path(run.get("path"), run_status, "E6_result", run_checks)
        for cond in run.get("conditions", []):
            child_status = "PASS" if run_status == "PASS" and cond.get("verification_status") == "PASS" else (run_status or cond.get("verification_status"))
            add_path(cond.get("child_path"), child_status, "E6_condition", e6_condition_binding_checks(cond, run))
    for run in value_at(proof, "COST_CLOSURE", "runs") or []:
        add_path(run.get("path"), run.get("verification_status"), "COST_CLOSURE_result", cost_run_binding_checks(run))
    binding_summary = summarize_checks(binding_checks)
    binding_status = "PASS" if binding_summary["fail"] == 0 and binding_summary["unchecked"] == 0 else "FAIL_OR_UNCHECKED"
    return {
        "status": proof.get("overall_status"),
        "proof_path": proof.get("_proof_path"),
        "proof_sha256": proof.get("_proof_sha256"),
        "allowed": allowed,
        "excluded": excluded,
        "binding_status": binding_status,
        "binding_summary": binding_summary,
        "binding_checks": binding_checks[:5000],
        "e34_reconstruction_audit": e34_reconstruction_audit,
        "e34_originwise_companion_count": e34_originwise_companion_count,
    }

def filter_rows_by_verification(rows: list[Row], decision: dict[str, Any]) -> list[Row]:
    allowed = decision.get("allowed")
    if allowed is None:
        return rows
    gated = {"E3", "E4", "E5", "E5_BOLT", "E6", "E6_baseline", "E6_baseline_horizon", "E6_horizon", "E6_record", "E6_record_regime"}
    out = []
    for row in rows:
        if row.experiment not in gated:
            out.append(row)
        elif row.source_path in allowed:
            out.append(row)
    return out


def filter_payload_by_verification(e34: dict[str, Any], e5: dict[str, Any], e6: dict[str, Any],
                                   bolt: dict[str, Any], cost: dict[str, Any], decision: dict[str, Any]) -> None:
    allowed = decision.get("allowed")
    if allowed is None:
        return

    excluded = decision.get("excluded") or []
    if excluded:
        by_kind: dict[str, int] = defaultdict(int)
        for item in excluded:
            by_kind[str(item.get("kind", "unknown"))] += 1
        reason = "final verification proof excluded actual items: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) + "; see verification_decision.excluded for status/binding details"
        e34["held_reason"] = "; ".join(x for x in [e34.get("held_reason"), reason] if x)
        e5["held_reason"] = "; ".join(x for x in [e5.get("held_reason"), reason] if x)
        bolt["held_reason"] = "; ".join(x for x in [bolt.get("held_reason"), reason] if x)
        held = list(e6.get("held_reasons") or [])
        held.append(reason)
        e6["held_reasons"] = held
        cost_held = list(cost.get("held_reasons") or [])
        cost_held.append(reason)
        cost["held_reasons"] = cost_held
        decision["exclusion_summary_by_kind"] = dict(sorted(by_kind.items()))

    def is_allowed(path_value: Any) -> bool:
        if not path_value:
            return False
        try:
            return rel(resolve_existing(str(path_value))) in allowed
        except Exception:
            return False

    before = {
        "E3_E4_cases": len(e34.get("cases", [])),
        "E5_runs": len(e5.get("runs", [])),
        "E5_conditions": sum(len(r.get("conditions", [])) for r in e5.get("runs", [])),
        "E6_runs": len(e6.get("runs", [])),
        "E6_conditions": sum(len(r.get("conditions", [])) for r in e6.get("runs", [])),
        "E5_BOLT_runs": len(bolt.get("runs", [])),
        "COST_CLOSURE_runs": len(cost.get("runs", [])),
    }
    e34["cases"] = [case for case in e34.get("cases", []) if case.get("relative_path") in allowed]
    e34["case_count"] = len(e34.get("cases", []))
    for run in e5.get("runs", []):
        run["conditions"] = [cond for cond in run.get("conditions", []) if is_allowed(cond.get("path"))]
    e5["runs"] = [run for run in e5.get("runs", []) if run.get("conditions")]
    e5["run_count"] = len(e5.get("runs", []))
    for run in e6.get("runs", []):
        result_allowed = run.get("relative_path") in allowed
        run["conditions"] = [cond for cond in run.get("conditions", []) if is_allowed(cond.get("path"))]
        if not result_allowed:
            run["matched_baselines"] = {}
    e6["runs"] = [run for run in e6.get("runs", []) if run.get("conditions") or run.get("matched_baselines")]
    e6["run_count"] = len(e6.get("runs", []))
    for run in bolt.get("runs", []):
        run["_adopted_by_verification"] = run.get("relative_path") in allowed
    bolt["runs"] = [run for run in bolt.get("runs", []) if run.get("_adopted_by_verification")]
    bolt["run_count"] = len(bolt.get("runs", []))
    cost["runs"] = [run for run in cost.get("runs", []) if run.get("relative_path") in allowed]
    cost["run_count"] = len(cost.get("runs", []))
    allowed_cost_paths = {run.get("relative_path") for run in cost.get("runs", [])}
    cost["rows"] = [row for row in cost.get("rows", []) if row.get("run_relative_path") in allowed_cost_paths]
    decision["adopted_scope_after_filter"] = {
        "before": before,
        "after": {
            "E3_E4_cases": e34.get("case_count", 0),
            "E5_runs": e5.get("run_count", 0),
            "E5_conditions": sum(len(r.get("conditions", [])) for r in e5.get("runs", [])),
            "E6_runs": e6.get("run_count", 0),
            "E6_conditions": sum(len(r.get("conditions", [])) for r in e6.get("runs", [])),
            "E5_BOLT_runs": bolt.get("run_count", 0),
            "COST_CLOSURE_runs": cost.get("run_count", 0),
        },
    }

def value_at(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


@dataclass
class Row:
    experiment: str
    dataset: str = ""
    group: str = ""
    method: str = ""
    condition: str = ""
    seed: str | int = ""
    split: str = "val"
    mse: float | None = None
    mae: float | None = None
    normalized_mse: float | None = None
    normalized_mae: float | None = None
    status: str = ""
    device: str = ""
    source_path: str = ""
    note: str = ""
    gate_label: str = ""
    raw_mse_sign: str = ""
    horizon: str = ""
    regime: str = ""
    record_id: str | int = ""
    baseline: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_csv(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "dataset": self.dataset,
            "group": self.group,
            "method": self.method,
            "condition": self.condition,
            "seed": self.seed,
            "split": self.split,
            "mse": "" if self.mse is None else self.mse,
            "mae": "" if self.mae is None else self.mae,
            "normalized_mse": "" if self.normalized_mse is None else self.normalized_mse,
            "normalized_mae": "" if self.normalized_mae is None else self.normalized_mae,
            "status": self.status,
            "device": self.device,
            "source_path": self.source_path,
            "note": self.note,
            "gate_label": self.gate_label,
            "raw_mse_sign": self.raw_mse_sign,
            "horizon": self.horizon,
            "regime": self.regime,
            "record_id": self.record_id,
            "baseline": self.baseline,
        }


def metric_from_obj(obj: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    mse = finite_float(obj.get("mse"))
    mae = finite_float(obj.get("mae"))
    norm = obj.get("normalized") if isinstance(obj.get("normalized"), dict) else {}
    nmse = finite_float(norm.get("mse"))
    nmae = finite_float(norm.get("mae"))
    return mse, mae, nmse, nmae


def mean(values: list[float | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(xs)) if xs else None


def restricted_full_observed_raw_score(npz_path: Path | str) -> dict[str, Any]:
    """Diagnostic raw score on timestamps where every raw channel is observed."""
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            pred = z["prediction"] * z["std"] + z["mean"]
            target = z["target"] * z["std"] + z["mean"]
            mask = z["mask"].astype(bool)
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": str(exc)}
    full_observed = np.broadcast_to(mask.all(axis=2, keepdims=True), mask.shape)
    if not full_observed.any():
        return {"status": "UNAVAILABLE", "error": "no fully observed timestamp"}
    err = np.where(full_observed, pred.astype(np.float64) - target.astype(np.float64), 0.0)
    counts = full_observed.sum(axis=(0, 1)).astype(np.float64)
    if np.any(counts <= 0):
        return {"status": "UNAVAILABLE", "error": "empty channel after restriction"}
    mse = ((err * err).sum(axis=(0, 1)) / counts).mean()
    mae = (np.abs(err).sum(axis=(0, 1)) / counts).mean()
    return {
        "status": "AVAILABLE_DIAGNOSTIC_ONLY",
        "mse": float(mse),
        "mae": float(mae),
        "observed_total": int(full_observed.sum()),
        "mask_hash": array_hash(full_observed.astype(bool)),
        "note": "Diagnostic only: raw score restricted to timestamps where all raw channels are observed; not used for training/selection.",
    }


def by_key(rows: list[Row], key: str) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        out[str(getattr(row, key))].append(row)
    return dict(out)


def collect_resources(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    transient = []
    for path in sorted((HERE / "resources").glob("*.json")):
        try:
            data = read_json(path)
        except Exception as exc:
            transient.append({"path": str(path), "read_error": str(exc), "status": "UNREADABLE"})
            continue
        rec = {
            "path": str(path),
            "sha256": sha256(path),
            "label": data.get("label"),
            "status": data.get("status"),
            "wall_seconds": finite_float(data.get("wall_seconds")) or 0.0,
            "gpu": bool(data.get("gpu")),
            "phase": data.get("phase"),
            "pid": data.get("pid"),
            "failure_reason": data.get("failure_reason"),
        }
        if data.get("status") in ("COMPLETE", "FAILED", "BUDGET_HELD", "FAILED_PRESERVED"):
            records.append(rec)
            manifest.append(file_record(path, "resource_ledger_closed"))
        else:
            transient.append(rec)
    return {
        "count": len(records),
        "cpu_wall_seconds": float(sum(r["wall_seconds"] for r in records if not r["gpu"])),
        "gpu_wall_seconds": float(sum(r["wall_seconds"] for r in records if r["gpu"])),
        "transient_resource_snapshot_count": len(transient),
        "transient_resources_not_bound_as_immutable_inputs": transient[-12:],
        "recent": sorted(records, key=lambda r: Path(r["path"]).stat().st_mtime if Path(r["path"]).exists() else 0)[-12:],
        "largest": sorted(records, key=lambda r: r["wall_seconds"], reverse=True)[:12],
    }



def timing_summary(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    return {
        "batch": block.get("batch"),
        "device": block.get("device"),
        "p50_seconds": block.get("p50_seconds"),
        "p95_seconds": block.get("p95_seconds"),
        "seconds_per_window_p50": block.get("seconds_per_window_p50"),
        "physical_inverse_and_CPU_output_included": block.get("physical_inverse_and_CPU_output_included"),
        "model_loading_disk_and_guards_excluded": block.get("model_loading_disk_and_guards_excluded"),
        "baseline_allocated_vram": block.get("baseline_allocated_vram"),
        "peak_allocated_vram": block.get("peak_allocated_vram"),
        "peak_reserved_vram": block.get("peak_reserved_vram"),
        "vram_peak_scope": block.get("vram_peak_scope"),
    }


def collect_cost_closure(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    rows = []
    expected = {"e2_cpu": 16, "e4_gpu": 72, "e5_gpu": 32}
    for path in sorted(RESULTS.glob("COST_CLOSURE*/RESULT.json")):
        manifest.append(file_record(path, "cost_closure_result"))
        data = read_json(path)
        if data.get("experiment") != "COST_CLOSURE":
            continue
        for pkey, role in (("plan_path", "cost_closure_plan"), ("gate_path", "cost_closure_gate"), ("budget_ledger", "cost_closure_budget_ledger")):
            if data.get(pkey):
                p = resolve_existing(str(data[pkey]))
                if p.exists():
                    manifest.append(file_record(p, role))
        run_rows = []
        for idx, row in enumerate(data.get("rows", []) or []):
            parity = row.get("parity") or {}
            causal = row.get("trained_encoder_history_causality") or row.get("causalBflags")
            rec = {
                "run_path": str(path),
                "run_relative_path": rel(path),
                "part": data.get("part"),
                "status": data.get("status"),
                "row_index": idx,
                "experiment": row.get("experiment"),
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "measurement": row.get("measurement"),
                "origin_hash": row.get("origin_hash"),
                "batch": row.get("batch"),
                "cudnn_allow_tf32": row.get("cudnn_allow_tf32"),
                "cuda_matmul_allow_tf32": row.get("cuda_matmul_allow_tf32"),
                "unique_inputs": row.get("unique_inputs"),
                "repeated_inputs": row.get("repeated_inputs"),
                "parity_status": parity.get("status"),
                "parity_normalized_max_abs": parity.get("normalized_max_abs"),
                "parity_physical_max_abs": parity.get("physical_max_abs"),
                "new_performance_score_computed": parity.get("new_performance_score_computed"),
                "end_to_end": timing_summary(row.get("end_to_end")),
                "cached_head_only": timing_summary(row.get("cached_head_only")),
                "causalBflags": causal,
                "original_export_full_origin_replay": row.get("original_export_full_origin_replay"),
                "inference_precision": row.get("inference_precision"),
                "actual_forward_microbatches": row.get("actual_forward_microbatches"),
                "original_run_device": row.get("original_run_device"),
            }
            run_rows.append(rec)
            rows.append(rec)
        run = {
            "path": str(path),
            "relative_path": rel(path),
            "sha256": sha256(path),
            "part": data.get("part"),
            "status": data.get("status"),
            "row_count": len(data.get("rows", []) or []),
            "expected_row_count": expected.get(str(data.get("part"))),
            "science_updates": data.get("science_updates"),
            "scores_computed": data.get("scores_computed"),
            "parity_flags": data.get("parity_flags"),
            "timer_contract": data.get("timer_contract"),
            "cached_head_contract": data.get("cached_head_contract"),
            "runtime_comparison_limit": data.get("runtime_comparison_limit"),
            "reused_BOLT_cost": data.get("reused_BOLT_cost"),
            "reused_Jena_cost": data.get("reused_Jena_cost"),
            "precision_diagnostic_proof": data.get("precision_diagnostic_proof"),
            "measured_conditions_subset": data.get("measured_conditions_subset"),
            "rows": run_rows,
        }
        runs.append(run)
    held = []
    present = {str(run.get("part")) for run in runs}
    for part in ("e2_cpu", "e4_gpu", "e5_gpu"):
        if part not in present:
            held.append(f"{part}: COST_CLOSURE RESULT.json not present at report time")
    return {"run_count": len(runs), "rows": rows, "runs": runs, "held_reasons": held, "cost_only_not_science": True}

def collect_adoption_v2(manifest: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = HERE / "ADOPTION_EVIDENCE_v2.json"
    manifest.append(file_record(path, "independent_adoption_v2"))
    if not path.exists():
        return None
    return read_json(path)


def dataset_columns() -> dict[str, list[str]]:
    path = HERE / "ASSETS_AND_PREFLIGHT.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    out = {}
    for dataset, obj in (data.get("data") or {}).items():
        if isinstance(obj, dict) and isinstance(obj.get("columns"), list):
            out[str(dataset)] = [str(x) for x in obj["columns"]]
    return out


def device_from_result_or_ledger(data: dict[str, Any]) -> str:
    if data.get("compute_device") or data.get("device"):
        return str(data.get("compute_device") or data.get("device"))
    ledger_path = data.get("budget_ledger")
    if ledger_path:
        try:
            ledger = read_json(resolve_existing(str(ledger_path)))
            return "cuda" if ledger.get("gpu") else "cpu"
        except Exception:
            return ""
    return ""


def collect_e1(adoption: dict[str, Any] | None, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    path = HERE / "E1_RESULT.json"
    manifest.append(file_record(path, "E1_result"))
    if adoption and adoption.get("E1"):
        e1 = adoption["E1"]
        return {
            "source": "ADOPTION_EVIDENCE_v2",
            "status": e1.get("status"),
            "rows": e1.get("rows"),
            "forecast_boundary_A_only": e1.get("forecast_boundary_A_only"),
            "non_boundary_B": e1.get("non_boundary_sensitivity_not_leakage_B"),
            "non_boundary_C": e1.get("non_boundary_sensitivity_not_leakage_C"),
            "checkpoint_hash_all_available_match": e1.get("checkpoint_hash_all_available_match"),
            "note": e1.get("adoption_note"),
        }
    if not path.exists():
        return {"source": str(path), "status": "MISSING"}
    data = read_json(path)
    boundary = {"cold_feature": 0.0, "cold_prediction": 0.0, "cold_warm_prediction": 0.0, "test_count": 0}
    b = {"max_abs": 0.0, "changed_exact_count": 0.0, "relative_l2": 0.0}
    c = {"max_abs": 0.0, "changed_exact_count": 0.0, "relative_l2": 0.0}
    for row in data.get("rows", []):
        for item in row.get("forecast_boundary_tests", []):
            boundary["test_count"] += 1
            for key in ("cold_feature", "cold_prediction", "cold_warm_prediction"):
                value = item.get(key)
                if isinstance(value, dict):
                    boundary[key] = max(boundary[key], abs(float(value.get("max_abs", 0.0))))
                elif value is not None:
                    boundary[key] = max(boundary[key], abs(float(value)))
        for src, dst in (("B_within_history_revision", b), ("C_same_left_append_revision", c)):
            obj = row.get(src, {})
            for key in dst:
                dst[key] = max(dst[key], abs(float(obj.get(key, 0.0))))
    boundary["leak_threshold_from_e1_py"] = 1e-4
    boundary["material_future_leak_detected"] = boundary["cold_prediction"] > 1e-4
    return {
        "source": str(path),
        "status": data.get("status"),
        "rows": len(data.get("rows", [])),
        "forecast_boundary_A_only": boundary,
        "non_boundary_B": b,
        "non_boundary_C": c,
        "checkpoint_hash_all_available_match": None,
        "note": "Only forecast_boundary_tests are used for leakage adjudication; B/C are context sensitivity checks.",
    }


def collect_e2(adoption: dict[str, Any] | None, rows: list[Row], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = HERE / "e2" / "SUMMARY_VERIFIED_run02.json"
    manifest.append(file_record(summary_path, "E2_summary"))
    if not summary_path.exists():
        return {"source": str(summary_path), "status": "MISSING"}
    summary = read_json(summary_path)
    real: dict[str, Any] = {}
    synthetic_gate: dict[str, dict[str, int]] = {}
    synthetic_mse: dict[str, dict[str, int]] = {}
    record_count = 0
    method_row_count = 0
    for rec in summary.get("records", []):
        p = Path(rec.get("path", ""))
        manifest.append(file_record(p, "E2_record"))
        if not p.exists():
            continue
        data = read_json(p)
        models = data.get("models", {})
        if not models:
            continue
        record_count += 1
        self_val = value_at(models, "SELF_LAG", "val") or {}
        all_val = value_at(models, "ALL_CHANNEL_LAG", "val") or {}
        all_minus_self = None
        if finite_float(all_val.get("mse")) is not None and finite_float(self_val.get("mse")) is not None:
            all_minus_self = float(all_val["mse"]) - float(self_val["mse"])
        cross_now_val = value_at(models, "SELF_LAG_PLUS_CROSS_NOW", "val") or {}
        cross_now_minus_self = None
        if finite_float(cross_now_val.get("mse")) is not None and finite_float(self_val.get("mse")) is not None:
            cross_now_minus_self = float(cross_now_val["mse"]) - float(self_val["mse"])
        if data.get("dataset") in ("jena", "robin"):
            real[data["dataset"]] = {
                "SELF_LAG": {"mse": self_val.get("mse"), "mae": self_val.get("mae")},
                "SELF_LAG_PLUS_CROSS_NOW": {"mse": cross_now_val.get("mse"), "mae": cross_now_val.get("mae")},
                "ALL_CHANNEL_LAG": {"mse": all_val.get("mse"), "mae": all_val.get("mae")},
                "ALL_minus_SELF": {
                    "mse_diff": finite_float(all_val.get("mse")) - finite_float(self_val.get("mse")),
                    "mae_diff": finite_float(all_val.get("mae")) - finite_float(self_val.get("mae")),
                },
                "CROSS_NOW_minus_SELF": {
                    "mse_diff": cross_now_minus_self,
                    "mae_diff": (finite_float(cross_now_val.get("mae")) - finite_float(self_val.get("mae"))
                                 if finite_float(cross_now_val.get("mae")) is not None and finite_float(self_val.get("mae")) is not None else None),
                },
                "split_samples": data.get("split_samples"),
                "origin_hashes": data.get("origin_hashes"),
                "target_hashes": data.get("target_hashes"),
                "mask_hashes": data.get("mask_hashes"),
                "shared_arrays": data.get("shared_arrays"),
                "normalization_fit_scope": data.get("normalization_fit_scope"),
            }
        if data.get("synthetic"):
            cond = data["synthetic"]["condition"]
            synthetic_gate.setdefault(cond, {"n": 0})
            synthetic_gate[cond]["n"] += 1
            label = str(data.get("LAG_SIGNAL"))
            synthetic_gate[cond][label] = synthetic_gate[cond].get(label, 0) + 1
            synthetic_mse.setdefault(cond, {"n": 0, "all_lag_mse_better": 0})
            synthetic_mse[cond]["n"] += 1
            if finite_float(all_val.get("mse")) is not None and finite_float(self_val.get("mse")) is not None:
                if float(all_val["mse"]) < float(self_val["mse"]):
                    synthetic_mse[cond]["all_lag_mse_better"] += 1
        for method, val in models.items():
            metric = val.get("val", {})
            mse, mae, nmse, nmae = metric_from_obj(metric)
            raw_sign = ""
            if method == "SELF_LAG":
                raw_sign = "reference"
            elif finite_float(mse) is not None and finite_float(self_val.get("mse")) is not None:
                delta = float(mse) - float(self_val["mse"])
                raw_sign = "better_than_self" if delta < 0 else ("worse_than_self" if delta > 0 else "tie_with_self")
            method_row_count += 1
            rows.append(Row("E2", dataset=str(data.get("dataset", "")), method=method,
                            condition=str((data.get("synthetic") or {}).get("condition", "")),
                            seed=str((data.get("synthetic") or {}).get("seed", "")),
                            mse=mse, mae=mae, normalized_mse=nmse, normalized_mae=nmae,
                            status=data.get("status", ""), source_path=rel(p),
                            note="lag ridge diagnostic; TRAIN/VAL only; SELF is the raw sign reference",
                            gate_label=str(data.get("LAG_SIGNAL", "")), raw_mse_sign=raw_sign))
    result = {
        "source": str(summary_path),
        "status": summary.get("status"),
        "record_count": record_count,
        "method_row_count": method_row_count,
        "real_summary": real,
        "synthetic_gate_label_summary": synthetic_gate,
        "synthetic_mse_better_summary": synthetic_mse,
        "criterion_note": "Gate labels and raw MSE sign are distinct. CSV rows include both gate_label and raw_mse_sign.",
    }
    if adoption and adoption.get("E2"):
        result["adoption_v2_crosscheck"] = {
            "status": adoption["E2"].get("summary_status"),
            "records_verified": adoption["E2"].get("records_verified"),
            "max_abs_reported_metric_diff": adoption["E2"].get("max_abs_reported_metric_diff"),
        }
    return result


def collect_e34(rows: list[Row], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for path in sorted((HERE / "e34").glob("*.json")):
        manifest.append(file_record(path, "E3_E4_case_or_summary"))
        try:
            data = read_json(path)
        except Exception:
            continue
        if data.get("experiment_id") != "E3_E4":
            continue
        case = {
            "path": str(path),
            "relative_path": rel(path),
            "status": data.get("status"),
            "dataset": data.get("dataset"),
            "dim": data.get("dim"),
            "seed": data.get("seed"),
            "scope": data.get("scope"),
            "device": data.get("device", "cpu"),
            "fit_bounds": data.get("fit_bounds"),
            "tune_bounds": data.get("tune_bounds"),
            "eval_bounds": data.get("eval_bounds"),
            "origin_hashes": data.get("origin_hashes"),
            "source_hashes": data.get("source_hashes"),
            "checkpoint_record": data.get("checkpoint_record"),
            "safety_gate_sha": data.get("safety_gate_sha"),
            "same_D_initialization": data.get("E3_same_D_initialization"),
            "reference_isolation": data.get("E4_reference_isolation"),
            "remaining_uncertainties": data.get("remaining_uncertainties"),
            "methods": {},
        }
        for section in ("E3", "E4"):
            for item in data.get(section, []):
                metric = item.get("metrics", {})
                mse, mae, nmse, nmae = metric_from_obj(metric)
                method = item.get("method", "")
                case["methods"][method] = {
                    "section": section,
                    "mse": mse,
                    "mae": mae,
                    "normalized_mse": nmse,
                    "normalized_mae": nmae,
                    "per_channel_mse": metric.get("per_channel_mse"),
                    "per_channel_mae": metric.get("per_channel_mae"),
                    "normalized_per_channel_mse": value_at(metric, "normalized", "per_channel_mse"),
                    "normalized_per_channel_mae": value_at(metric, "normalized", "per_channel_mae"),
                    "prediction_path": item.get("prediction_path"),
                    "prediction_hash": item.get("prediction_hash"),
                    "latent_metrics": item.get("latent_metrics"),
                    "note": item.get("input") or item.get("FIT_latent_target_scope") or item.get("not_lower_bound"),
                }
                if section == "E4" and item.get("latent_metrics") and item.get("prediction_path"):
                    case["methods"][method]["restricted_full_observed_raw"] = restricted_full_observed_raw_score(item.get("prediction_path"))
                rows.append(Row(section, dataset=str(data.get("dataset", "")), group=f"d{data.get('dim')}",
                                method=method, seed=data.get("seed", ""), mse=mse, mae=mae,
                                normalized_mse=nmse, normalized_mae=nmae, status=data.get("status", ""),
                                device=str(data.get("device", "cpu")), source_path=rel(path),
                                note="E3/E4 same decoder diagnostic; reference latent is diagnostic, not deployable"))
        cases.append(case)
    expected_min = 8
    held_reason = None
    if len(cases) < expected_min:
        held_reason = (
            f"Only {len(cases)} E3/E4 case files were present at report time; "
            "remaining d320/Robin/GPU cases are not adopted unless their result JSONs exist."
        )
    return {"case_count": len(cases), "expected_minimum_if_full_scope": expected_min, "held_reason": held_reason, "cases": cases}


def collect_e5(rows: list[Row], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    result_paths = sorted(RESULTS.glob("E5*/RESULT.json"))
    runs = []
    for path in result_paths:
        manifest.append(file_record(path, "E5_result"))
        data = read_json(path)
        if data.get("experiment") != "E5":
            continue
        run = {
            "path": str(path),
            "relative_path": rel(path),
            "status": data.get("status"),
            "dataset": data.get("dataset"),
            "plan_sha256": data.get("plan_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "source_pins": data.get("source_pins"),
            "budget_ledger": data.get("budget_ledger"),
            "budget_ledger_sha256": data.get("budget_ledger_sha256"),
            "interpretation": data.get("interpretation"),
            "device": device_from_result_or_ledger(data),
            "conditions": [],
        }
        for cond in data.get("conditions", []):
            metric = cond.get("val", {})
            mse, mae, nmse, nmae = metric_from_obj(metric)
            child_path = Path(cond.get("path", ""))
            manifest.append(file_record(child_path, "E5_condition_result"))
            child_device = run["device"]
            if child_path.exists():
                try:
                    child_data = read_json(child_path)
                    child_device = child_data.get("compute_device") or child_data.get("device") or child_device
                except Exception:
                    pass
            row = {
                "condition": cond.get("condition"),
                "seed": cond.get("seed"),
                "path": cond.get("path"),
                "sha256": cond.get("sha256"),
                "mse": mse,
                "mae": mae,
                "normalized_mse": nmse,
                "normalized_mae": nmae,
                "device": child_device,
            }
            run["conditions"].append(row)
            rows.append(Row("E5", dataset=str(data.get("dataset", "")), condition=str(cond.get("condition", "")),
                            seed=cond.get("seed", ""), method="direct_ridge_on_frozen_representation",
                            mse=mse, mae=mae, normalized_mse=nmse, normalized_mae=nmae,
                            status=data.get("status", ""), device=str(child_device), source_path=rel(child_path),
                            note="balanced objective comparison; fixed budget, not convergence proof"))
        runs.append(run)
    return {"run_count": len(runs), "runs": runs, "held_reason": None if runs else "No E5 result JSON present."}


def collect_e6(rows: list[Row], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    failures = []
    for path in sorted(RESULTS.glob("E6*/RESULT.json")):
        manifest.append(file_record(path, "E6_result"))
        data = read_json(path)
        if data.get("experiment") != "E6":
            continue
        run = {
            "path": str(path),
            "relative_path": rel(path),
            "dataset": data.get("dataset"),
            "status": data.get("status"),
            "objective": data.get("objective"),
            "compute_device": data.get("compute_device"),
            "plan_sha256": data.get("plan_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "source_pins": data.get("source_pins"),
            "scope": data.get("scope"),
            "TEST_access": data.get("TEST_access"),
            "matched_baselines": data.get("matched_baselines"),
            "conditions": [],
            "interpretation": data.get("interpretation"),
        }
        for baseline_name, baseline_obj in (data.get("matched_baselines") or {}).items():
            metric = baseline_obj.get("val", {}) if isinstance(baseline_obj, dict) else {}
            mse, mae, nmse, nmae = metric_from_obj(metric)
            rows.append(Row("E6_baseline", dataset=str(data.get("dataset", "")), condition=str(baseline_name),
                            method=str(baseline_name), mse=mse, mae=mae,
                            normalized_mse=nmse, normalized_mae=nmae,
                            status=data.get("status", ""), device=str(data.get("compute_device", "")),
                            source_path=rel(path), baseline="matched_pooled_raw",
                            note="matched pooled raw baseline from E6; compare carefully because SELF neural still sees all raw channels in history"))
            for horizon, hmetric in (metric.get("horizon_metrics") or {}).items():
                hmse, hmae, hnmse, hnmae = metric_from_obj(hmetric)
                rows.append(Row("E6_baseline_horizon", dataset=str(data.get("dataset", "")), condition=str(baseline_name),
                                method=str(baseline_name), split=str(horizon), horizon=str(horizon),
                                mse=hmse, mae=hmae, normalized_mse=hnmse, normalized_mae=hnmae,
                                status=data.get("status", ""), device=str(data.get("compute_device", "")),
                                source_path=rel(path), baseline="matched_pooled_raw",
                                note="matched raw baseline horizon sidecar"))
        for cond in data.get("conditions", []):
            metric = cond.get("val", {})
            mse, mae, nmse, nmae = metric_from_obj(metric)
            child_path = Path(cond.get("path", ""))
            manifest.append(file_record(child_path, "E6_condition_result"))
            item = {
                "mode": cond.get("mode"),
                "seed": cond.get("seed"),
                "path": cond.get("path"),
                "sha256": cond.get("sha256"),
                "mse": mse,
                "mae": mae,
                "normalized_mse": nmse,
                "normalized_mae": nmae,
            }
            run["conditions"].append(item)
            mode_note = "graph comparison; TRAIN/VAL diagnostic only"
            if cond.get("mode") == "SELF_ONLY":
                mode_note += "; SELF_ONLY neural input still contains all raw channels over history"
            rows.append(Row("E6", dataset=str(data.get("dataset", "")), condition=str(cond.get("mode", "")),
                            seed=cond.get("seed", ""), method="graph_encoder_decoder",
                            mse=mse, mae=mae, normalized_mse=nmse, normalized_mae=nmae,
                            status=data.get("status", ""), device=str(data.get("compute_device", "")),
                            source_path=rel(child_path),
                            note=mode_note))
            for horizon, hmetric in (metric.get("horizon_metrics") or {}).items():
                hmse, hmae, hnmse, hnmae = metric_from_obj(hmetric)
                rows.append(Row("E6_horizon", dataset=str(data.get("dataset", "")), condition=str(cond.get("mode", "")),
                                seed=cond.get("seed", ""), method="graph_encoder_decoder",
                                split=str(horizon), horizon=str(horizon),
                                mse=hmse, mae=hmae, normalized_mse=hnmse, normalized_mae=hnmae,
                                status=data.get("status", ""), device=str(data.get("compute_device", "")),
                                source_path=rel(child_path), note="E6 horizon sidecar"))
            for rec_metric in metric.get("per_record", []) if isinstance(metric.get("per_record"), list) else []:
                record_id = rec_metric.get("record_id", "")
                rmse, rmae, rnmse, rnmae = metric_from_obj(rec_metric)
                rows.append(Row("E6_record", dataset=str(data.get("dataset", "")), condition=str(cond.get("mode", "")),
                                seed=cond.get("seed", ""), method="graph_encoder_decoder", record_id=record_id,
                                regime="all", mse=rmse, mae=rmae, normalized_mse=rnmse, normalized_mae=rnmae,
                                status=data.get("status", ""), device=str(data.get("compute_device", "")),
                                source_path=rel(child_path), note="E6 per-record audit row"))
                for regime in ("before_change", "after_change"):
                    if isinstance(rec_metric.get(regime), dict):
                        gmse, gmae, gnmse, gnmae = metric_from_obj(rec_metric[regime])
                        rows.append(Row("E6_record_regime", dataset=str(data.get("dataset", "")), condition=str(cond.get("mode", "")),
                                        seed=cond.get("seed", ""), method="graph_encoder_decoder", record_id=record_id,
                                        regime=regime, mse=gmse, mae=gmae, normalized_mse=gnmse, normalized_mae=gnmae,
                                        status=data.get("status", ""), device=str(data.get("compute_device", "")),
                                        source_path=rel(child_path), note="E6 per-record regime audit row"))
        runs.append(run)
    for path in sorted(RESULTS.glob("E6*/FAILURE.json")):
        manifest.append(file_record(path, "E6_failure"))
        data = read_json(path)
        failures.append({
            "path": str(path),
            "relative_path": rel(path),
            "dataset": data.get("dataset"),
            "status": data.get("status"),
            "failure_reason": data.get("failure_reason"),
            "plan_sha256": data.get("plan_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "note": "Preserved failed attempt; not adopted as condition comparison.",
        })
    held = []
    expected = ("synthetic_changed", "synthetic_fixed", "jena")
    present = {str(r.get("dataset")) for r in runs}
    for dataset in expected:
        if dataset not in present:
            held.append(f"{dataset}: no COMPLETE E6 RESULT.json present at report time")
    return {"run_count": len(runs), "runs": runs, "failures": failures, "held_reasons": held}


def collect_bolt_helpers(rows: list[Row], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    profiles = []
    plans = []
    patterns = ["E5_BOLT*", "BOLT*", "E_BOLT*", "*bolt*", "*BOLT*"]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(RESULTS.glob(f"{pattern}/RESULT.json"))
    for path in sorted(paths):
        manifest.append(file_record(path, "bolt_helper_result"))
        data = read_json(path)
        plan = {}
        plan_path_value = data.get("plan_path")
        if plan_path_value:
            plan_path = resolve_existing(str(plan_path_value))
            if plan_path.exists():
                manifest.append(file_record(plan_path, "bolt_helper_bound_plan"))
                plan = read_json(plan_path)
        dataset_value = data.get("dataset") or plan.get("dataset") or "jena"
        run = {
            "path": str(path),
            "relative_path": rel(path),
            "status": data.get("status"),
            "experiment": data.get("experiment") or data.get("experiment_id"),
            "dataset": dataset_value,
            "conditions": data.get("conditions") or data.get("results") or data.get("cases") or [],
            "cost_only_not_science": "COST_ONLY" in str(data.get("status", "")),
        }
        if run["cost_only_not_science"]:
            profiles.append(run)
            continue
        for case in run["conditions"]:
            dataset = str(dataset_value or case.get("dataset") or "jena")
            condition = str(case.get("condition") or case.get("objective") or case.get("case") or "")
            seed = case.get("seed", "")
            methods = case.get("methods")
            if isinstance(methods, list):
                for method in methods:
                    if not isinstance(method, dict):
                        continue
                    metric = method.get("val") or method.get("metrics") or {}
                    mse, mae, nmse, nmae = metric_from_obj(metric)
                    rows.append(Row("E5_BOLT", dataset=dataset, condition=condition,
                                    seed=seed, method=str(method.get("method", "")),
                                    mse=mse, mae=mae, normalized_mse=nmse, normalized_mae=nmae,
                                    status=str(data.get("status", "")), device=str(data.get("compute_device") or data.get("device") or "cuda"),
                                    source_path=rel(path),
                                    note="Bolt attachment helper; adopted only if RESULT.json exists, no plan-only science claim"))
            else:
                metric = case.get("val") or case.get("metrics") or {}
                mse, mae, nmse, nmae = metric_from_obj(metric)
                rows.append(Row("E5_BOLT", dataset=dataset, condition=condition,
                                seed=seed, method=str(case.get("method", "")),
                                mse=mse, mae=mae, normalized_mse=nmse, normalized_mae=nmae,
                                status=str(data.get("status", "")), device=str(data.get("compute_device") or data.get("device") or "cuda"),
                                source_path=rel(path),
                                note="Bolt attachment helper; adopted only if RESULT.json exists, no plan-only science claim"))
        runs.append(run)
    for plan in sorted((HERE / "plans").glob("*BOLT*.json")):
        manifest.append(file_record(plan, "bolt_helper_plan"))
        plans.append({"path": str(plan), "relative_path": rel(plan), "sha256": sha256(plan), "bytes": plan.stat().st_size})
    held = None if runs else "No Bolt E/Jena REC-vs-CONTRAST science RESULT.json present yet; plan/profile files alone are not adopted science results."
    return {"run_count": len(runs), "profile_count": len(profiles), "runs": runs, "profiles": profiles, "plans": plans, "held_reason": held}


def collect_e5_vs_e2_jena_baseline(e2: dict[str, Any], e5: dict[str, Any], rows: list[Row],
                                   manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Jena E5 latent readout against raw lag baselines under matched VAL scope."""
    real = e2.get("real_summary") or {}
    jena = real.get("jena")
    if not jena:
        return {"status": "MISSING_E2_JENA"}

    e5_jena_conditions = [r for r in rows if r.experiment == "E5" and r.dataset == "jena" and r.mse is not None]
    if not e5_jena_conditions:
        return {"status": "MISSING_E5_JENA"}

    e5_condition_means = summarize_objectives(rows, "E5", "condition", dataset_filter="jena", group_by_dataset=False)
    comparison = {
        "status": "AVAILABLE",
        "interpretation": (
            "Compares E5 latent33 direct-ridge readout against E2 raw lag baselines on Jena VAL. "
            "Targets, masks and FIT/TUNE/VAL split hashes are checked where child files expose them, "
            "but E5's last-E readout has RF63 support over the last33 embeddings, so the raw support can cover about 95 ticks; "
            "E2 raw33 uses a different feature contract. This is not a pure compression-only causal comparison."
        ),
        "E2_raw33_baselines": {
            "SELF_LAG": jena.get("SELF_LAG"),
            "SELF_LAG_PLUS_CROSS_NOW": jena.get("SELF_LAG_PLUS_CROSS_NOW"),
            "ALL_CHANNEL_LAG": jena.get("ALL_CHANNEL_LAG"),
            "ALL_minus_SELF": jena.get("ALL_minus_SELF"),
            "CROSS_NOW_minus_SELF": jena.get("CROSS_NOW_minus_SELF"),
        },
        "E5_latent33_condition_means": e5_condition_means,
        "scope_check": {
            "E2_split_samples": jena.get("split_samples"),
            "E2_normalization_fit_scope": jena.get("normalization_fit_scope"),
            "E2_origin_hashes": jena.get("origin_hashes"),
            "E2_target_hashes": jena.get("target_hashes"),
            "E2_mask_hashes": jena.get("mask_hashes"),
            "condition_hash_checks": [],
            "all_checked_conditions_share_val_origin_target_mask": None,
        },
        "caution": (
            "SELF_LAG_PLUS_CROSS_NOW is often stronger than ALL_CHANNEL_LAG in current Jena VAL. "
            "Do not conflate compression with raw feature expansion or d320 capacity changes."
        ),
    }
    expected_origins = jena.get("origin_hashes") or {}
    expected_targets = jena.get("target_hashes") or {}
    expected_masks = jena.get("mask_hashes") or {}
    all_hash_ok = True
    for run in e5.get("runs", []):
        if run.get("dataset") != "jena":
            continue
        for cond in run.get("conditions", []):
            child_path = Path(cond.get("path", ""))
            if not child_path.exists():
                continue
            manifest.append(file_record(child_path, "E5_vs_E2_condition_child"))
            child = read_json(child_path)
            row_meta = child.get("row_metadata", {})
            origin_hashes = {}
            for split in ("fit", "tune", "val"):
                origins = value_at(row_meta, split, "origins")
                if origins is not None:
                    origin_hashes[split] = array_hash(np.asarray(origins, dtype=np.int64))
            npz_path = Path(child.get("prediction_path", ""))
            manifest.append(file_record(npz_path, "E5_vs_E2_prediction_npz"))
            val_target_hash = None
            val_mask_hash = None
            val_origin_hash = origin_hashes.get("val")
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=False) as z:
                    val_target_hash = array_hash(z["target"])
                    val_mask_hash = array_hash(z["mask"].astype(bool))
                    val_origin_hash = array_hash(z["origins"].astype(np.int64))
            hash_ok = {
                "fit_origins": origin_hashes.get("fit") == expected_origins.get("fit"),
                "tune_origins": origin_hashes.get("tune") == expected_origins.get("tune"),
                "val_origins": val_origin_hash == expected_origins.get("val"),
                "val_target": val_target_hash == expected_targets.get("val"),
                "val_mask": val_mask_hash == expected_masks.get("val"),
            }
            all_hash_ok = all_hash_ok and all(hash_ok.values())
            comparison["scope_check"]["condition_hash_checks"].append({
                "condition": cond.get("condition"),
                "seed": cond.get("seed"),
                "path": str(child_path),
                "hash_ok": hash_ok,
                "feature_scaler_scope": value_at(child, "direct_ridge", "feature_scaler_scope"),
                "row_counts": {split: len(value_at(row_meta, split, "origins") or []) for split in ("fit", "tune", "val")},
                "lambda": value_at(child, "direct_ridge", "lambda"),
            })
    comparison["scope_check"]["all_checked_conditions_share_val_origin_target_mask"] = all_hash_ok
    return comparison


def summarize_objectives(rows: list[Row], experiment: str, field: str, *,
                         dataset_filter: str | None = None,
                         group_by_dataset: bool = True) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.experiment == experiment:
            if dataset_filter is not None and row.dataset != dataset_filter:
                continue
            value = str(getattr(row, field))
            key = f"{row.dataset}:{value}" if group_by_dataset else value
            groups[key].append(row)
    return {
        key: {
            "n": len(vals),
            "dataset": sorted({str(v.dataset) for v in vals}),
            field: sorted({str(getattr(v, field)) for v in vals}),
            "mean_mse": mean([v.mse for v in vals]),
            "mean_mae": mean([v.mae for v in vals]),
            "mean_normalized_mse": mean([v.normalized_mse for v in vals]),
            "mean_normalized_mae": mean([v.normalized_mae for v in vals]),
            "seeds": sorted({str(v.seed) for v in vals if v.seed != ""}),
        }
        for key, vals in sorted(groups.items())
    }


def summarize_bolt_objectives(rows: list[Row]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.experiment != "E5_BOLT":
            continue
        key = f"{row.dataset}:{row.condition}:{row.method}"
        groups[key].append(row)
    return {
        key: {
            "n": len(vals),
            "dataset": sorted({str(v.dataset) for v in vals}),
            "condition": sorted({str(v.condition) for v in vals}),
            "method": sorted({str(v.method) for v in vals}),
            "mean_mse": mean([v.mse for v in vals]),
            "mean_mae": mean([v.mae for v in vals]),
            "mean_normalized_mse": mean([v.normalized_mse for v in vals]),
            "mean_normalized_mae": mean([v.normalized_mae for v in vals]),
            "seeds": sorted({str(v.seed) for v in vals if v.seed != ""}),
        }
        for key, vals in sorted(groups.items())
    }


def e3_full_prefix_pairs(e34: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = []
    for case in e34.get("cases", []):
        methods = case.get("methods", {})
        full = methods.get("FULL_BOLT_REC_D")
        prefix = methods.get("PREFIX_BOLT_REC_D") or methods.get("BOLT_LATENT")
        if not isinstance(full, dict) or not isinstance(prefix, dict):
            continue
        pair = {
            "dataset": case.get("dataset"),
            "dim": case.get("dim"),
            "seed": case.get("seed"),
            "FULL_mse": full.get("mse"),
            "PREFIX_mse": prefix.get("mse"),
            "PREFIX_minus_FULL_mse": None,
            "FULL_mae": full.get("mae"),
            "PREFIX_mae": prefix.get("mae"),
            "PREFIX_minus_FULL_mae": None,
            "FULL_normalized_mse": full.get("normalized_mse"),
            "PREFIX_normalized_mse": prefix.get("normalized_mse"),
            "PREFIX_minus_FULL_normalized_mse": None,
        }
        for target, lhs, rhs in (
            ("PREFIX_minus_FULL_mse", "PREFIX_mse", "FULL_mse"),
            ("PREFIX_minus_FULL_mae", "PREFIX_mae", "FULL_mae"),
            ("PREFIX_minus_FULL_normalized_mse", "PREFIX_normalized_mse", "FULL_normalized_mse"),
        ):
            a, b = finite_float(pair.get(lhs)), finite_float(pair.get(rhs))
            if a is not None and b is not None:
                pair[target] = a - b
        pairs.append(pair)
    return pairs


def e3_pair_summary_by_dataset_dim(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[f"{pair.get('dataset')}:d{pair.get('dim')}"].append(pair)
    out = {}
    for key, vals in sorted(groups.items()):
        deltas = [finite_float(v.get("PREFIX_minus_FULL_mse")) for v in vals]
        deltas = [d for d in deltas if d is not None]
        ndeltas = [finite_float(v.get("PREFIX_minus_FULL_normalized_mse")) for v in vals]
        ndeltas = [d for d in ndeltas if d is not None]
        out[key] = {
            "n_pairs": len(vals),
            "prefix_better_mse_count": sum(1 for d in deltas if d < 0),
            "prefix_worse_mse_count": sum(1 for d in deltas if d > 0),
            "delta_mse_range": [min(deltas), max(deltas)] if deltas else None,
            "delta_normalized_mse_range": [min(ndeltas), max(ndeltas)] if ndeltas else None,
            "seeds": sorted({str(v.get("seed")) for v in vals}),
        }
    return out


def e4_case_table(e34: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = [
        "LATENT_SELF_RIDGE",
        "LATENT_JOINT_RIDGE",
        "REFERENCE_LATENT_DIAGNOSTIC",
        "BOLT_LATENT",
        "DIRECT_RAW_RIDGE",
        "LATENT_PERSISTENCE",
    ]
    table = []
    for case in e34.get("cases", []):
        methods = case.get("methods", {})
        row: dict[str, Any] = {
            "dataset": case.get("dataset"),
            "dim": case.get("dim"),
            "seed": case.get("seed"),
        }
        for method in wanted:
            item = methods.get(method)
            row[f"{method}_mse"] = item.get("mse") if isinstance(item, dict) else None
            row[f"{method}_mae"] = item.get("mae") if isinstance(item, dict) else None
            row[f"{method}_normalized_mse"] = item.get("normalized_mse") if isinstance(item, dict) else None
            latent = item.get("latent_metrics") if isinstance(item, dict) and isinstance(item.get("latent_metrics"), dict) else {}
            restricted = item.get("restricted_full_observed_raw") if isinstance(item, dict) and isinstance(item.get("restricted_full_observed_raw"), dict) else {}
            row[f"{method}_latent_mse"] = latent.get("mse")
            row[f"{method}_restricted_full_observed_raw_mse"] = restricted.get("mse")
        table.append(row)
    return table


def bolt_advantage_summary(e4_table: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(e4_table),
        "physical_bolt_better_than_self": 0,
        "physical_bolt_better_than_joint": 0,
        "normalized_bolt_better_than_self": 0,
        "normalized_bolt_better_than_joint": 0,
        "physical_comparable_self": 0,
        "physical_comparable_joint": 0,
        "normalized_comparable_self": 0,
        "normalized_comparable_joint": 0,
    }
    for row in e4_table:
        bolt = finite_float(row.get("BOLT_LATENT_mse"))
        bolt_norm = finite_float(row.get("BOLT_LATENT_normalized_mse"))
        self_raw = finite_float(row.get("LATENT_SELF_RIDGE_mse"))
        joint_raw = finite_float(row.get("LATENT_JOINT_RIDGE_mse"))
        self_norm = finite_float(row.get("LATENT_SELF_RIDGE_normalized_mse"))
        joint_norm = finite_float(row.get("LATENT_JOINT_RIDGE_normalized_mse"))
        if bolt is not None and self_raw is not None:
            summary["physical_comparable_self"] += 1
            summary["physical_bolt_better_than_self"] += int(bolt < self_raw)
        if bolt is not None and joint_raw is not None:
            summary["physical_comparable_joint"] += 1
            summary["physical_bolt_better_than_joint"] += int(bolt < joint_raw)
        if bolt_norm is not None and self_norm is not None:
            summary["normalized_comparable_self"] += 1
            summary["normalized_bolt_better_than_self"] += int(bolt_norm < self_norm)
        if bolt_norm is not None and joint_norm is not None:
            summary["normalized_comparable_joint"] += 1
            summary["normalized_bolt_better_than_joint"] += int(bolt_norm < joint_norm)
    return summary


def e4_latent_raw_reversal_summary(e34: dict[str, Any]) -> list[dict[str, Any]]:
    methods = ["BOLT_LATENT", "LATENT_SELF_RIDGE", "LATENT_JOINT_RIDGE", "LATENT_PERSISTENCE"]
    columns_by_dataset = dataset_columns()
    out = []
    for case in e34.get("cases", []):
        rows = []
        for method in methods:
            item = (case.get("methods") or {}).get(method)
            if not isinstance(item, dict):
                continue
            latent = item.get("latent_metrics") if isinstance(item.get("latent_metrics"), dict) else {}
            restricted = item.get("restricted_full_observed_raw") if isinstance(item.get("restricted_full_observed_raw"), dict) else {}
            rows.append({
                "method": method,
                "latent_mse": latent.get("mse"),
                "reported_raw_mse": item.get("mse"),
                "reported_raw_normalized_mse": item.get("normalized_mse"),
                "restricted_full_observed_raw_mse": restricted.get("mse"),
                "restricted_status": restricted.get("status"),
            })
        latent_rank = [r["method"] for r in sorted(rows, key=lambda r: finite_float(r.get("latent_mse")) if finite_float(r.get("latent_mse")) is not None else float("inf"))]
        raw_rank = [r["method"] for r in sorted(rows, key=lambda r: finite_float(r.get("reported_raw_mse")) if finite_float(r.get("reported_raw_mse")) is not None else float("inf"))]
        restricted_available = all(finite_float(r.get("restricted_full_observed_raw_mse")) is not None for r in rows)
        restricted_rank = [r["method"] for r in sorted(rows, key=lambda r: finite_float(r.get("restricted_full_observed_raw_mse")) if finite_float(r.get("restricted_full_observed_raw_mse")) is not None else float("inf"))]
        comparisons = []
        bolt = (case.get("methods") or {}).get("BOLT_LATENT")
        if isinstance(bolt, dict):
            for other_name in ("LATENT_SELF_RIDGE", "LATENT_JOINT_RIDGE"):
                other = (case.get("methods") or {}).get(other_name)
                if not isinstance(other, dict):
                    continue
                physical_delta = None
                normalized_delta = None
                if finite_float(bolt.get("mse")) is not None and finite_float(other.get("mse")) is not None:
                    physical_delta = float(bolt["mse"]) - float(other["mse"])
                if finite_float(bolt.get("normalized_mse")) is not None and finite_float(other.get("normalized_mse")) is not None:
                    normalized_delta = float(bolt["normalized_mse"]) - float(other["normalized_mse"])
                per_channel = []
                bpc = bolt.get("per_channel_mse") or []
                opc = other.get("per_channel_mse") or []
                bnpc = bolt.get("normalized_per_channel_mse") or []
                onpc = other.get("normalized_per_channel_mse") or []
                columns = columns_by_dataset.get(str(case.get("dataset")), [])
                for i in range(min(len(bpc), len(opc))):
                    delta = finite_float(bpc[i]) - finite_float(opc[i]) if finite_float(bpc[i]) is not None and finite_float(opc[i]) is not None else None
                    ndelta = finite_float(bnpc[i]) - finite_float(onpc[i]) if i < len(bnpc) and i < len(onpc) and finite_float(bnpc[i]) is not None and finite_float(onpc[i]) is not None else None
                    per_channel.append({
                        "index": i,
                        "column": columns[i] if i < len(columns) else f"c{i}",
                        "physical_mse_delta_bolt_minus_other": delta,
                        "normalized_mse_delta_bolt_minus_other": ndelta,
                    })
                high_physical = sorted(
                    [x for x in per_channel if finite_float(x.get("physical_mse_delta_bolt_minus_other")) is not None],
                    key=lambda x: abs(float(x["physical_mse_delta_bolt_minus_other"])),
                    reverse=True,
                )[:5]
                comparisons.append({
                    "other": other_name,
                    "physical_mse_delta_bolt_minus_other": physical_delta,
                    "normalized_mse_delta_bolt_minus_other": normalized_delta,
                    "physical_channel_win_count_bolt": sum(1 for x in per_channel if finite_float(x.get("physical_mse_delta_bolt_minus_other")) is not None and float(x["physical_mse_delta_bolt_minus_other"]) < 0),
                    "normalized_channel_win_count_bolt": sum(1 for x in per_channel if finite_float(x.get("normalized_mse_delta_bolt_minus_other")) is not None and float(x["normalized_mse_delta_bolt_minus_other"]) < 0),
                    "channel_count": len(per_channel),
                    "largest_physical_delta_channels": high_physical,
                })
        out.append({
            "dataset": case.get("dataset"),
            "dim": case.get("dim"),
            "seed": case.get("seed"),
            "rows": rows,
            "bolt_pairwise_channel_deltas": comparisons,
            "latent_rank": latent_rank,
            "reported_raw_rank": raw_rank,
            "restricted_full_observed_raw_rank": restricted_rank if restricted_available else None,
            "ranking_reversal_reported_raw": bool(latent_rank and raw_rank and latent_rank[0] != raw_rank[0]),
            "ranking_reversal_restricted_raw": bool(restricted_available and latent_rank and restricted_rank and latent_rank[0] != restricted_rank[0]),
            "interpretation": "Same-D diagnostic only; if reversal remains under restricted full-observed raw mask, F accuracy alone is insufficient, but this still is not D-only causal proof.",
        })
    return out



def e5_robin_seed_comparison(rows: list[Row], e2: dict[str, Any]) -> dict[str, Any]:
    seed_rows: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row.experiment == "E5" and row.dataset == "robin":
            seed_rows[str(row.seed)][str(row.condition)] = {"mse": row.mse, "mae": row.mae, "source_path": row.source_path}
    out = []
    for seed, conds in sorted(seed_rows.items()):
        contrast = conds.get("CAUSAL_CONTRAST")
        rec = conds.get("CAUSAL_REC")
        full = conds.get("FULL_REC")
        forecast = conds.get("CAUSAL_FORECAST")
        out.append({
            "seed": seed,
            "FULL_REC": full,
            "CAUSAL_REC": rec,
            "CAUSAL_CONTRAST": contrast,
            "CAUSAL_FORECAST": forecast,
            "contrast_minus_rec_mse": None if not (contrast and rec) else finite_float(contrast.get("mse")) - finite_float(rec.get("mse")),
            "contrast_minus_rec_mae": None if not (contrast and rec) else finite_float(contrast.get("mae")) - finite_float(rec.get("mae")),
        })
    raw_self = value_at(e2, "real_summary", "robin", "SELF_LAG")
    return {
        "comparison_conditions": "Robin VAL E5 objectives, two seeds, same saved E/D/ridge artifacts; compared against E2 Robin SELF_LAG only as a simple raw baseline with a different feature contract.",
        "per_seed": out,
        "E2_robin_SELF_LAG_raw_baseline": raw_self,
        "interpretation": "Robin contrast improves over REC in adopted E5 rows, but the simple raw SELF_LAG baseline remains lower MSE/MAE; do not generalize Jena REC-vs-contrast behavior to Robin.",
    }


def e6_jena_updated_vs_self(rows: list[Row]) -> dict[str, Any]:
    by_seed: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        if row.experiment == "E6" and row.dataset == "jena" and row.condition in {"SELF_ONLY", "PAST_UPDATED_LAG_GRAPH"}:
            by_seed[str(row.seed)][row.condition] = row
    records = []
    for seed, vals in sorted(by_seed.items()):
        self_row = vals.get("SELF_ONLY")
        updated = vals.get("PAST_UPDATED_LAG_GRAPH")
        if not self_row or not updated:
            continue
        records.append({
            "seed": seed,
            "SELF_ONLY": {"mse": self_row.mse, "mae": self_row.mae, "source_path": self_row.source_path},
            "PAST_UPDATED_LAG_GRAPH": {"mse": updated.mse, "mae": updated.mae, "source_path": updated.source_path},
            "updated_minus_self_mse": None if updated.mse is None or self_row.mse is None else updated.mse - self_row.mse,
            "updated_minus_self_mae": None if updated.mae is None or self_row.mae is None else updated.mae - self_row.mae,
        })
    return {
        "comparison_conditions": "Jena E6 staged GPU run, same dataset/split and saved graph pipeline; SELF_ONLY still sees all raw-channel history, so this is not a no-cross-channel baseline.",
        "per_seed": records,
        "interpretation": "UPDATED beats SELF for a seed only if both delta MSE and delta MAE are negative. Mixed signs across seeds mean no stable dynamic-graph benefit in the current pipeline.",
    }


def collect_optional_scope_notes(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    lstm_path = HERE / "plans" / "LSTM_pair_v1.json"
    if lstm_path.exists():
        manifest.append(file_record(lstm_path, "optional_lstm_plan"))
        data = read_json(lstm_path)
        notes["LSTM"] = {
            "path": str(lstm_path),
            "relative_path": rel(lstm_path),
            "sha256": sha256(lstm_path),
            "status": data.get("status"),
            "execute": data.get("execute"),
            "reason": data.get("reason"),
            "scope": data.get("scope"),
            "conditional_scientific_gate": data.get("conditional_scientific_gate"),
            "interpretation": data.get("interpretation"),
        }
    robin_e2 = HERE / "e2" / "run01_robin.json"
    if robin_e2.exists():
        manifest.append(file_record(robin_e2, "optional_robin_e2_gate"))
        data = read_json(robin_e2)
        notes["E6_Robin"] = {
            "path": str(robin_e2),
            "relative_path": rel(robin_e2),
            "sha256": sha256(robin_e2),
            "lag_signal": data.get("LAG_SIGNAL"),
            "status": "NOT_APPLICABLE_WITH_REASON" if data.get("LAG_SIGNAL") == "NO_LINEAR_32LAG_SIGNAL" else "REVIEW_GATE",
            "reason": "Robin E2 gate did not show a linear 32-lag signal, so Robin E6 graph NN was skipped by gate rather than by NN failure." if data.get("LAG_SIGNAL") == "NO_LINEAR_32LAG_SIGNAL" else "Robin E2 gate requires review.",
        }
    return notes


def answer_six_questions(e1: dict[str, Any], e2: dict[str, Any], e34: dict[str, Any],
                         e5: dict[str, Any], e6: dict[str, Any], bolt: dict[str, Any],
                         rows: list[Row], e5_vs_e2: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    e1_boundary = e1.get("forecast_boundary_A_only") or {}
    real = e2.get("real_summary") or {}
    e5_agg = summarize_objectives(rows, "E5", "condition")
    e6_agg = summarize_objectives(rows, "E6", "condition")
    bolt_agg = summarize_bolt_objectives(rows)
    e3_pairs = e3_full_prefix_pairs(e34)
    e3_pair_summary = e3_pair_summary_by_dataset_dim(e3_pairs)
    e4_table = e4_case_table(e34)
    e4_reversals = e4_latent_raw_reversal_summary(e34)
    bolt_adv = bolt_advantage_summary(e4_table)
    e5_robin_contrast = e5_robin_seed_comparison(rows, e2)
    e6_jena_updated = e6_jena_updated_vs_self(rows)

    total_pairs = len(e3_pairs)
    prefix_better = sum(1 for p in e3_pairs if (finite_float(p.get("PREFIX_minus_FULL_mse")) or 0.0) < 0)
    prefix_worse = sum(1 for p in e3_pairs if (finite_float(p.get("PREFIX_minus_FULL_mse")) or 0.0) > 0)
    norm_better = sum(1 for p in e3_pairs if (finite_float(p.get("PREFIX_minus_FULL_normalized_mse")) or 0.0) < 0)
    norm_worse = sum(1 for p in e3_pairs if (finite_float(p.get("PREFIX_minus_FULL_normalized_mse")) or 0.0) > 0)
    if total_pairs:
        e3_answer = (
            f"현재 실제 FULL-vs-PREFIX paired case {total_pairs}개에서는 PREFIX가 physical MSE를 낮춘 경우 "
            f"{prefix_better}개, 높인 경우 {prefix_worse}개이고, normalized MSE는 낮춘 경우 {norm_better}개, 높인 경우 {norm_worse}개다."
        )
    else:
        e3_answer = "현재 실제 FULL-vs-PREFIX paired case가 없어 개선 여부를 판정하지 않는다."
    if e34.get("held_reason"):
        e3_next_decision = "보류/제외 항목은 verification_decision.excluded와 held_reason을 보고 재개한다."
    else:
        e3_next_decision = "현재 채택 case 범위 안에서만 해석하고, 다른 예산·데이터 조건으로 확장하지 않는다."

    return [
        {
            "question": "실제 미래 정보 누출이 있었는가?",
            "answer": "채택된 E1 forecast-boundary 검사 범위에서는 material future leakage가 보이지 않는다.",
            "raw": e1_boundary,
            "uncertainty": "A forecast-boundary만 누출 판정에 사용한다. B/C within-history/context 변화는 별도 민감도이며, 모든 배포 경로를 증명하지 않는다.",
            "next_decision": "E1은 통과로 두되 성능 개선·일반화 검증으로 말하지 않는다.",
        },
        {
            "question": "시점별 과거 전용 표현이 예측을 개선했는가?",
            "answer": e3_answer,
            "raw": {
                "FULL_vs_PREFIX_pairs": e3_pairs,
                "paired_summary_by_dataset_dim": e3_pair_summary,
                "E5_vs_E2_Jena_baseline": e5_vs_e2,
            },
            "uncertainty": e34.get("held_reason") or "E3/E4 adopted files only; VAL development diagnostic.",
            "next_decision": e3_next_decision,
        },
        {
            "question": "다른 채널의 지연 정보가 실제로 유용했는가?",
            "answer": "혼재다. Jena real VAL에서 cross-now가 ALL-lag보다 강한 구간이 있어 delayed increment 자체는 아직 입증되지 않았다.",
            "raw": {
                "real_summary": real,
                "synthetic_gate_label_summary": e2.get("synthetic_gate_label_summary"),
                "synthetic_mse_better_summary": e2.get("synthetic_mse_better_summary"),
            },
            "uncertainty": "synthetic gate label과 raw MSE sign은 다른 기준이다. 반복 VAL 진단이며 test 일반화가 아니다.",
            "next_decision": "cross-now, self-lag, all-lag를 분리해서 보고하고, delayed graph의 추가 가치는 E6 matched 결과가 PASS일 때만 주장한다.",
        },
        {
            "question": "표현·잠재 미래 예측·디코더 중 어디에 문제가 남았는가?",
            "answer": "E4 case별로 latent MSE 순위와 decoded raw MSE 순위가 달라질 수 있어 F 정확도만으로 설명하면 부족하다.",
            "raw": {
                "e34_reference_latent_is_diagnostic_not_bound": True,
                "E4_case_table": e4_table,
                "latent_raw_ranking_reversal": e4_reversals,
            },
            "uncertainty": (
                "REFERENCE_LATENT_DIAGNOSTIC은 미래 latent label을 쓰는 분리 진단이며 deployable lower bound가 아니다. "
                "same-target decomposition을 하더라도 q/r/2qr 항은 서술적 항등식이고, r을 decoder-only 책임으로 돌리면 안 된다."
                " restricted_full_observed_raw_mse는 mask population을 맞춘 보조 진단이며 primary raw score를 대체하지 않는다."
            ),
            "next_decision": "decoder-only 개선과 latent predictor 개선을 분리해 유지한다.",
        },
        {
            "question": "대조학습과 동적 그래프가 필요한가, 단순 대안으로 충분한가?",
            "answer": "E5/E6는 dataset·seed별 원수치로 판단해야 한다. Robin에서는 contrast가 REC보다 낫지만 raw SELF_LAG baseline보다 높고, Jena E6 UPDATED는 seed별로 SELF 대비 방향이 갈려 동적 그래프 이득이 안정적이지 않다.",
            "raw": {"E5_by_condition": e5_agg, "E6_by_mode": e6_agg, "E5_Robin_contrast_vs_REC_and_rawSELF": e5_robin_contrast, "E6_Jena_UPDATED_vs_SELF": e6_jena_updated, "E6_held_reasons": e6.get("held_reasons")},
            "uncertainty": (
                "E5는 fixed budget small run이고 dataset별로 해석해야 한다. Jena의 REC-vs-contrast 관계를 Robin에 외삽하지 않고, Robin에서 contrast가 REC보다 나아도 raw/simple SELF baseline과 별도로 비교한다. "
                "best non-REC 선택은 compatibility diagnostic이지 scientific benefit gate가 아니다. E6 SELF_ONLY neural encoder도 history의 모든 raw channel을 섞으므로 no-other-channel baseline이 아니다."
            ),
            "next_decision": "현재 pipeline에서는 동적 그래프 family를 기각하지 말고 hold한다. UPDATED가 seed와 MSE/MAE 모두에서 SELF/FIXED/WRONG보다 낮은 반복 증거가 나오면 그때 필요성을 주장한다.",
        },
        {
            "question": "Chronos-Bolt를 유지할 근거가 있는가?",
            "answer": (
                f"채택된 실제 case 기준으로 PREFIX/FULL은 physical {prefix_better}/{total_pairs}, normalized {norm_better}/{total_pairs} 개선이고, "
                f"E4 Bolt latent는 SELF 대비 physical {bolt_adv.get('physical_bolt_better_than_self')}/{bolt_adv.get('physical_comparable_self')}, "
                f"JOINT 대비 physical {bolt_adv.get('physical_bolt_better_than_joint')}/{bolt_adv.get('physical_comparable_joint')} 개선이다. "
                "따라서 유지/폐기는 전체 조건부 원수치로 판단해야 하며, 단일 Jena d16 결과로 폐기하면 안 된다."
            ),
            "raw": {
                "bolt_helper_run_count": bolt.get("run_count"),
                "bolt_helper_by_dataset_condition_method": bolt_agg,
                "E3_FULL_vs_PREFIX_pairs": e3_pairs,
                "E3_PREFIX_counts": {"physical_better": prefix_better, "physical_worse": prefix_worse, "normalized_better": norm_better, "normalized_worse": norm_worse, "n": total_pairs},
                "E4_Bolt_advantage_counts": bolt_adv,
                "E4_case_table": e4_table,
                "E4_latent_raw_and_normalized_rankings": e4_reversals,
            },
            "uncertainty": (bolt.get("held_reason") or "Bolt helper 결과가 있더라도 같은 dataset·condition·method readout 비교로만 해석한다.") + " Physical macro는 heterogeneous units에 민감하고, normalized companion은 primary를 사후 교체하는 근거가 아니다. d320은 raw channel count보다 큰 representation expansion이므로 compression 근거로 말하지 않는다. Bolt/REC 계열은 MSE와 MAE 방향이 섞일 수 있어 두 지표를 함께 제시한다.",
            "next_decision": "Bolt를 폐기하기보다 native ridge/direct alternatives 대비 physical·normalized·per-channel tradeoff를 함께 보여주는 baseline으로 제한한다.",
        },
    ]


def make_code_table(headers: list[str], records: list[list[Any]]) -> str:
    widths = [len(h) for h in headers]
    strings = []
    for row in records:
        srow = [str(x) for x in row]
        strings.append(srow)
        for i, value in enumerate(srow):
            widths[i] = max(widths[i], len(value))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in strings]
    return "```\n" + "\n".join([line, sep, *body]) + "\n```"


def write_csv(out_dir: Path, rows: list[Row]) -> Path:
    path = out_dir / "raw_metrics_v4.csv"
    fields = list(Row("", "").to_csv().keys())
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())
    return path


def write_cost_csv(out_dir: Path, cost: dict[str, Any]) -> Path:
    path = out_dir / "cost_closure_timing_v4.csv"
    fields = [
        "part", "status", "experiment", "dataset", "method", "condition", "seed", "measurement", "row_index",
        "timing_kind", "batch", "row_batch", "cudnn_allow_tf32", "cuda_matmul_allow_tf32", "device", "p50_seconds", "p95_seconds", "seconds_per_window_p50",
        "baseline_allocated_vram", "peak_allocated_vram", "peak_reserved_vram",
        "physical_inverse_and_CPU_output_included", "model_loading_disk_and_guards_excluded",
        "parity_status", "parity_normalized_max_abs", "parity_physical_max_abs",
        "new_performance_score_computed", "origin_hash", "unique_inputs", "repeated_inputs", "run_relative_path",
    ]
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in cost.get("rows", []) or []:
            timing_items = [(kind, row.get(kind) or {}) for kind in ("end_to_end", "cached_head_only")]
            if not any(t for _, t in timing_items):
                timing_items = [("diagnostic_no_timer", {})]
            for timing_kind, timing in timing_items:
                if not timing and timing_kind != "diagnostic_no_timer":
                    continue
                writer.writerow({
                    "part": row.get("part"),
                    "status": row.get("status"),
                    "experiment": row.get("experiment"),
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "condition": row.get("condition"),
                    "seed": row.get("seed"),
                    "measurement": row.get("measurement"),
                    "row_index": row.get("row_index"),
                    "timing_kind": timing_kind,
                    "batch": timing.get("batch"),
                    "row_batch": row.get("batch"),
                    "cudnn_allow_tf32": row.get("cudnn_allow_tf32"),
                    "cuda_matmul_allow_tf32": row.get("cuda_matmul_allow_tf32"),
                    "device": timing.get("device"),
                    "p50_seconds": timing.get("p50_seconds"),
                    "p95_seconds": timing.get("p95_seconds"),
                    "seconds_per_window_p50": timing.get("seconds_per_window_p50"),
                    "baseline_allocated_vram": timing.get("baseline_allocated_vram"),
                    "peak_allocated_vram": timing.get("peak_allocated_vram"),
                    "peak_reserved_vram": timing.get("peak_reserved_vram"),
                    "physical_inverse_and_CPU_output_included": timing.get("physical_inverse_and_CPU_output_included"),
                    "model_loading_disk_and_guards_excluded": timing.get("model_loading_disk_and_guards_excluded"),
                    "parity_status": row.get("parity_status"),
                    "parity_normalized_max_abs": row.get("parity_normalized_max_abs"),
                    "parity_physical_max_abs": row.get("parity_physical_max_abs"),
                    "new_performance_score_computed": row.get("new_performance_score_computed"),
                    "origin_hash": row.get("origin_hash"),
                    "unique_inputs": row.get("unique_inputs"),
                    "repeated_inputs": row.get("repeated_inputs"),
                    "run_relative_path": row.get("run_relative_path"),
                })
    return path


def make_figures(out_dir: Path, rows: list[Row], e1: dict[str, Any], e2: dict[str, Any],
                 e5: dict[str, Any], e6: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=False, exist_ok=False)
    figures: list[dict[str, Any]] = []

    def save(name: str, title: str, source: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append({"path": str(path), "relative_path": rel(path), "sha256": sha256(path), "title": title, "source": source})

    # 1. E1 boundary versus context sensitivity.
    b = e1.get("non_boundary_B") or {}
    c = e1.get("non_boundary_C") or {}
    a = e1.get("forecast_boundary_A_only") or {}
    vals = [finite_float(a.get("cold_prediction")) or 0.0, finite_float(b.get("max_abs")) or 0.0, finite_float(c.get("max_abs")) or 0.0]
    plt.figure(figsize=(6, 3.4))
    plt.bar(["A future suffix", "B history revision", "C context shift"], vals, color=["#4C78A8", "#F58518", "#E45756"])
    plt.yscale("symlog", linthresh=1e-6)
    plt.ylabel("Max absolute change")
    plt.title("E1 leakage boundary versus context sensitivity")
    save("01_e1_boundary_context.png", "E1 leakage boundary versus context sensitivity", "E1_RESULT / ADOPTION_EVIDENCE_v2")

    # 2. E2 real deltas.
    real = e2.get("real_summary") or {}
    if real:
        datasets = sorted(real)
        fig, axes = plt.subplots(len(datasets), 1, figsize=(5.5, max(3.0, 2.2 * len(datasets))), squeeze=False)
        for ax, dataset in zip(axes[:, 0], datasets):
            val = finite_float(value_at(real[dataset], "ALL_minus_SELF", "mse_diff")) or 0.0
            ax.axhline(0, color="black", linewidth=0.8)
            ax.bar(["ALL - SELF"], [val], color="#54A24B" if val < 0 else "#E45756")
            ax.set_title(dataset)
            ax.set_ylabel("Physical MSE delta")
        fig.suptitle("E2 real validation lag contribution by dataset")
        save("02_e2_real_lag_delta.png", "E2 real validation lag contribution by dataset", "E2 run JSONs / ADOPTION_EVIDENCE_v2")

    # 3. E3/E4 methods by seed.
    e3_pairs = []
    case_map: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.experiment == "E3" and r.mse is not None and r.method in ("FULL_BOLT_REC_D", "PREFIX_BOLT_REC_D"):
            key = (str(r.dataset), str(r.group), str(r.seed))
            case_map[key][r.method] = float(r.mse)
    for (dataset, dim, seed), metrics in sorted(case_map.items()):
        if "FULL_BOLT_REC_D" in metrics and "PREFIX_BOLT_REC_D" in metrics:
            e3_pairs.append((dataset, f"{dim}/s{seed}", metrics["PREFIX_BOLT_REC_D"] - metrics["FULL_BOLT_REC_D"]))
    if e3_pairs:
        by_dataset: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for dataset, label, val in e3_pairs:
            by_dataset[dataset].append((label, val))
        datasets = sorted(by_dataset)
        fig, axes = plt.subplots(len(datasets), 1, figsize=(7.5, max(3.2, 2.6 * len(datasets))), squeeze=False)
        for ax, dataset in zip(axes[:, 0], datasets):
            labels = [x[0] for x in by_dataset[dataset]]
            vals = [x[1] for x in by_dataset[dataset]]
            ax.axhline(0, color="black", linewidth=0.8)
            ax.bar(labels, vals, color=["#54A24B" if v < 0 else "#E45756" for v in vals])
            ax.set_title(dataset)
            ax.set_ylabel("PREFIX - FULL physical MSE")
            ax.tick_params(axis="x", rotation=25)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        fig.suptitle("E3 paired prefix effect by dataset")
        save("03_e3_full_prefix_delta.png", "E3 FULL-vs-PREFIX paired deltas by dataset", "E3/E4 case JSONs")

    def dataset_panel_bar(metric_rows: list[Row], experiment: str, field: str, title: str, color: str, filename: str, source: str) -> None:
        datasets = sorted({r.dataset for r in metric_rows if r.dataset})
        if not datasets:
            return
        fig, axes = plt.subplots(len(datasets), 1, figsize=(7.5, max(3.2, 2.6 * len(datasets))), squeeze=False)
        for ax, dataset in zip(axes[:, 0], datasets):
            agg = summarize_objectives(metric_rows, experiment, field, dataset_filter=dataset, group_by_dataset=False)
            labels = list(agg)
            vals = [agg[k]["mean_mse"] for k in labels]
            ax.bar(labels, vals, color=color)
            ax.set_title(f"{dataset}")
            ax.set_ylabel("Physical macro MSE")
            ax.tick_params(axis="x", rotation=20)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        fig.suptitle(title)
        save(filename, title, source)

    # 4. E5 objective comparison.
    e5_rows = [r for r in rows if r.experiment == "E5" and r.mse is not None]
    if e5_rows:
        dataset_panel_bar(e5_rows, "E5", "condition", "E5 objective comparison by dataset", "#72B7B2", "04_e5_objective_mse.png", "E5 RESULT.json")

    # 5. E6 mode comparison if present.
    e6_rows = [r for r in rows if r.experiment == "E6" and r.mse is not None]
    if e6_rows:
        dataset_panel_bar(e6_rows, "E6", "condition", "E6 graph mode comparison by dataset", "#B279A2", "05_e6_graph_mode_mse.png", "E6 RESULT.json")

    # 6. E4 physical versus normalized companion deltas.
    e4_cases: dict[tuple[str, str, str], dict[str, Row]] = defaultdict(dict)
    for row in rows:
        if row.experiment == "E4" and row.method in ("BOLT_LATENT", "LATENT_SELF_RIDGE", "LATENT_JOINT_RIDGE"):
            e4_cases[(row.dataset, row.group, str(row.seed))][row.method] = row
    deltas_by_dataset: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for key, methods in sorted(e4_cases.items()):
        dataset, dim, seed = key
        bolt_row = methods.get("BOLT_LATENT")
        if not bolt_row:
            continue
        for other_name, short in (("LATENT_SELF_RIDGE", "self"), ("LATENT_JOINT_RIDGE", "joint")):
            other = methods.get(other_name)
            if not other or bolt_row.mse is None or other.mse is None:
                continue
            if bolt_row.normalized_mse is None or other.normalized_mse is None:
                continue
            label = f"{dim}/s{seed} vs {short}"
            deltas_by_dataset[dataset].append((label, float(bolt_row.mse) - float(other.mse), float(bolt_row.normalized_mse) - float(other.normalized_mse)))
    if deltas_by_dataset:
        datasets = sorted(deltas_by_dataset)
        fig, axes = plt.subplots(len(datasets), 2, figsize=(10, max(3.8, 2.8 * len(datasets))), squeeze=False)
        for row_idx, dataset in enumerate(datasets):
            values = deltas_by_dataset[dataset]
            labels = [v[0] for v in values]
            physical = [v[1] for v in values]
            normalized = [v[2] for v in values]
            x = np.arange(len(labels))
            for col_idx, (vals, metric_title, ylabel, color, linthresh) in enumerate((
                (physical, "Physical macro MSE delta", "Bolt - alternative physical MSE", "#4C78A8", 1.0),
                (normalized, "Normalized MSE delta", "Bolt - alternative normalized MSE", "#F58518", 0.05),
            )):
                ax = axes[row_idx, col_idx]
                ax.axhline(0, color="black", linewidth=0.8)
                ax.bar(x, vals, color=color)
                ax.set_title(f"{dataset}: {metric_title}")
                ax.set_ylabel(ylabel)
                ax.set_yscale("symlog", linthresh=linthresh)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=25, ha="right")
        fig.suptitle("E4 Bolt deltas by dataset and metric scale")
        save("06_e4_physical_normalized_deltas.png", "E4 Bolt deltas by dataset and metric scale", "E3/E4 case JSONs")

    # 7. Actual metric rows by experiment.
    counts = defaultdict(int)
    for row in rows:
        counts[row.experiment] += 1
    if counts:
        labels = list(sorted(counts))
        vals = [counts[k] for k in labels]
        plt.figure(figsize=(6, 3.2))
        plt.bar(labels, vals, color="#9D755D")
        plt.ylabel("Adopted metric rows")
        plt.title("Actual adopted result coverage")
        save("07_actual_coverage.png", "Actual adopted result coverage", "Actual result JSON scan")

    return figures[:7]


def write_report(out_dir: Path, payload: dict[str, Any], rows: list[Row], figures: list[dict[str, Any]]) -> tuple[Path, Path]:
    answers = payload["six_question_answers"]
    e1 = payload["E1"]
    e2 = payload["E2"]
    e34 = payload["E3_E4"]
    e5 = payload["E5"]
    e5_vs_e2 = payload["E5_vs_E2_Jena_baseline"]
    e6 = payload["E6"]
    bolt = payload["Bolt_helpers"]
    cost = payload.get("Cost_closure") or {}
    optional_notes = payload.get("Optional_scope_notes") or {}
    resources = payload["resources"]

    lines: list[str] = []
    lines.append("# Past-only representation diagnosis final report v4")
    lines.append("")
    lines.append("이 보고서는 현재 저장소에 실제 존재하는 파일만 채택했다. 보호된 TEST 접근, 새 과학 실행, 모델 추론, plan/gate 생성, commit/push는 하지 않는다.")
    lines.append("")
    verification = payload.get("verification_decision") or {}
    lines.append("## 검증 proof 채택 상태")
    lines.append("")
    lines.append(f"- proof status: {verification.get('status')}")
    lines.append(f"- proof path: `{verification.get('proof_path')}`")
    lines.append(f"- proof sha256: `{verification.get('proof_sha256')}`")
    lines.append(f"- excluded non-PASS/unchecked actual items: {len(verification.get('excluded') or [])}")
    lines.append(f"- binding status: {verification.get('binding_status')}")
    lines.append(f"- binding summary: `{json.dumps(verification.get('binding_summary') or {}, ensure_ascii=False, sort_keys=True)}`")
    scope_after = verification.get("adopted_scope_after_filter") or {}
    if scope_after:
        lines.append(f"- adopted scope after verification: `{json.dumps(scope_after.get('after') or {}, ensure_ascii=False, sort_keys=True)}`")
    lines.append("- E3/E4/E5/E5_BOLT/E6 science rows are adopted only when their final verification status is PASS and their current file/array SHA bindings still match proof. Failed or unchecked files remain in manifest/held notes, not in science means or figures.")
    excluded_preview = verification.get("excluded") or []
    if excluded_preview:
        lines.append(make_code_table(["kind", "status", "binding", "path"], [[x.get("kind"), x.get("verification_status"), x.get("binding_status"), x.get("path")] for x in excluded_preview[:20]]))
    lines.append("")

    rec_audit = verification.get("e34_reconstruction_audit") or []
    if rec_audit:
        lines.append("E3 reconstruction NPZ verification and post-hoc slices:")
        records = []
        for item in rec_audit[:40]:
            calc = item.get("calculated") or {}
            norm = calc.get("normalized") or {}
            records.append([item.get("dataset"), f"d{item.get('dim')}", item.get("seed"), item.get("method"), item.get("status"), fmt(calc.get("mse")), fmt(calc.get("mae")), fmt(norm.get("mse")), item.get("path")])
        lines.append(make_code_table(["dataset", "dim", "seed", "method", "status", "rec_mse", "rec_mae", "rec_norm_mse", "path"], records))
        lines.append("- slice metrics in proof are post-hoc quartile diagnostics of the saved reconstruction window, not a prespecified gate.")
        lines.append(f"- originwise paired companion records in proof: {verification.get('e34_originwise_companion_count')}")
        lines.append("")
    lines.append("## 핵심 6문항 답")
    lines.append("")
    for i, ans in enumerate(answers, start=1):
        lines.append(f"### {i}. {ans['question']}")
        lines.append("")
        lines.append(f"- 답: {ans['answer']}")
        lines.append(f"- 불확실성: {ans['uncertainty']}")
        lines.append(f"- 다음 판단: {ans['next_decision']}")
        lines.append("- 원수치:")
        raw = json.dumps(ans["raw"], ensure_ascii=False, indent=2, allow_nan=False)
        lines.append("```json")
        lines.append(raw)
        lines.append("```")
        lines.append("")

    lines.append("## 비교 조건과 검산 범위")
    lines.append("")
    lines.append("- 모든 adopted 수치는 TRAIN/VAL 진단이다. TEST 접근은 채택하지 않는다.")
    lines.append("- primary metric은 dataset 내부 physical channel macro MSE/MAE다. 채널 단위가 섞여 있으므로 dataset 간 pooled claim을 하지 않는다.")
    lines.append("- normalized companion은 사전 보조 지표로 함께 보고하되 primary physical metric을 사후 교체하지 않는다.")
    lines.append("- REFERENCE_LATENT_DIAGNOSTIC은 미래 latent label을 쓰는 분리 진단이며 deployable lower bound가 아니다.")
    lines.append("- E4 same-target decomposition은 `P-y = (P-REF) + (REF-y)` 서술 항등식이다. 교차항 cancellation이 가능하고, `REF-y`를 decoder-only 잔차로 해석하면 안 된다.")
    lines.append("- E3/E4 physical metric은 TRAIN_ALL_FROZEN_DIAGNOSTIC scaler 기준으로 inverse score를 계산한다. E5/E6는 FIT-only scaler를 쓴다.")
    lines.append("- FULL/PREFIX/latent ridge는 context capacity와 readout capacity가 다르므로 같은 조건의 승패로 과장하지 않는다. PREFIX old TS2는 같은 origin에서도 window start/context가 달라질 수 있어 online stability 증명이 아니다.")
    lines.append("- E5 latent33 readout은 last E_t의 RF63이 겹쳐 raw 기준 약 95 tick support를 볼 수 있으므로, E2 raw33 baseline과의 차이를 compression 자체의 인과 효과로 말하지 않는다.")
    lines.append("- E6 SELF_ONLY DenseTinyE는 history의 raw channels를 모두 섞는다. synthetic에서는 latent d16이 raw C=6보다 넓으므로 graph message 효과를 추가 정보 효과로만 해석하면 안 된다.")
    lines.append("")

    lines.append("## 원수치 요약")
    lines.append("")
    real = e2.get("real_summary") or {}
    real_records = []
    for dataset, obj in real.items():
        d = obj.get("ALL_minus_SELF", {})
        real_records.append([dataset, fmt(d.get("mse_diff")), fmt(d.get("mae_diff"))])
    if real_records:
        lines.append("E2 real VAL ALL_CHANNEL_LAG - SELF_LAG:")
        lines.append(make_code_table(["dataset", "delta_mse", "delta_mae"], real_records))
        lines.append("")

    synthetic_gate = e2.get("synthetic_gate_label_summary") or {}
    synthetic_mse = e2.get("synthetic_mse_better_summary") or {}
    if synthetic_gate or synthetic_mse:
        lines.append("E2 synthetic lag diagnostic counts:")
        syn_records = []
        for condition in sorted(set(synthetic_gate) | set(synthetic_mse)):
            gate = synthetic_gate.get(condition, {})
            mse = synthetic_mse.get(condition, {})
            syn_records.append([
                condition,
                gate.get("n", mse.get("n", "")),
                gate.get("True", gate.get("TRUE", gate.get("PASS", ""))),
                gate.get("False", gate.get("FALSE", gate.get("FAIL", ""))),
                mse.get("all_lag_mse_better", ""),
            ])
        lines.append(make_code_table(["condition", "n", "gate_true/pass", "gate_false/fail", "ALL_MSE_better"], syn_records))
        lines.append("- gate label과 raw MSE sign은 다른 기준이므로 서로 대체하지 않는다.")
        lines.append("")

    e34_records = []
    for case in e34.get("cases", []):
        methods = case.get("methods", {})
        e34_records.append([
            case.get("dataset"), f"d{case.get('dim')}", case.get("seed"),
            fmt(value_at(methods, "FULL_BOLT_REC_D", "mse")),
            fmt(value_at(methods, "PREFIX_BOLT_REC_D", "mse")),
            fmt(value_at(methods, "LATENT_SELF_RIDGE", "mse")),
            fmt(value_at(methods, "LATENT_SELF_RIDGE", "normalized_mse")),
            fmt(value_at(methods, "LATENT_SELF_RIDGE", "latent_metrics", "mse")),
            fmt(value_at(methods, "LATENT_JOINT_RIDGE", "mse")),
            fmt(value_at(methods, "LATENT_JOINT_RIDGE", "normalized_mse")),
            fmt(value_at(methods, "LATENT_JOINT_RIDGE", "latent_metrics", "mse")),
            fmt(value_at(methods, "REFERENCE_LATENT_DIAGNOSTIC", "mse")),
            fmt(value_at(methods, "DIRECT_RAW_RIDGE", "mse")),
            fmt(value_at(methods, "BOLT_LATENT", "mse")),
            fmt(value_at(methods, "BOLT_LATENT", "normalized_mse")),
            fmt(value_at(methods, "BOLT_LATENT", "latent_metrics", "mse")),
            fmt(value_at(methods, "BOLT_LATENT", "restricted_full_observed_raw", "mse")),
        ])
    if e34_records:
        lines.append("E3/E4 diagnostic MSE by actual case:")
        lines.append(make_code_table(["dataset", "dim", "seed", "full", "prefix", "self_raw", "self_norm", "self_lat", "joint_raw", "joint_norm", "joint_lat", "reference_raw", "direct_raw", "bolt_raw", "bolt_norm", "bolt_lat", "bolt_raw_fullobs"], e34_records))
        lines.append("- primary physical macro와 normalized companion의 순위가 다를 수 있다. `bolt_raw_fullobs`는 모든 raw channel이 관측된 timestamp로 제한한 보조 진단이다.")
        lines.append("")

    e3_pairs = e3_full_prefix_pairs(e34)
    if e3_pairs:
        lines.append("E3 paired FULL/PREFIX deltas:")
        lines.append(make_code_table(
            ["dataset", "dim", "seed", "prefix-full_mse", "prefix-full_norm_mse"],
            [[p.get("dataset"), f"d{p.get('dim')}", p.get("seed"), fmt(p.get("PREFIX_minus_FULL_mse")), fmt(p.get("PREFIX_minus_FULL_normalized_mse"))] for p in e3_pairs],
        ))
        lines.append("")

    e5_agg = summarize_objectives(rows, "E5", "condition")
    if e5_agg:
        lines.append("E5 objective means within each dataset:")
        lines.append(make_code_table(["dataset:condition", "n", "mean_mse", "mean_mae", "mean_norm_mse"], [[k, v["n"], fmt(v["mean_mse"]), fmt(v["mean_mae"]), fmt(v["mean_normalized_mse"])] for k, v in e5_agg.items()]))
        lines.append("")

    if e5_vs_e2.get("status") == "AVAILABLE":
        raw = e5_vs_e2.get("E2_raw33_baselines", {})
        lines.append("E5 latent33 readout vs E2 Jena raw33 baselines:")
        records = []
        for name, metric in raw.items():
            if isinstance(metric, dict) and "mse" in metric:
                records.append([name, fmt(metric.get("mse")), fmt(metric.get("mae")), "NA"])
        for name, metric in summarize_objectives(rows, "E5", "condition", dataset_filter="jena", group_by_dataset=False).items():
            records.append([f"E5_{name}", fmt(metric.get("mean_mse")), fmt(metric.get("mean_mae")), fmt(metric.get("mean_normalized_mse"))])
        lines.append(make_code_table(["method", "mse", "mae", "norm_mse"], records))
        scope = e5_vs_e2.get("scope_check", {})
        lines.append(f"- same FIT/TUNE/VAL origin-target-mask hash check: {scope.get('all_checked_conditions_share_val_origin_target_mask')}")
        lines.append("- 주의: raw CROSS_NOW/ALL 확장 baseline과 latent readout을 compression 효과 하나로 묶어 말하면 안 된다. E5는 last-E RF63 support와 learned E capacity가 추가로 개입한다.")
        lines.append("")

    e6_agg = summarize_objectives(rows, "E6", "condition")
    if e6_agg:
        lines.append("E6 graph mode means within each dataset:")
        lines.append(make_code_table(["dataset:mode", "n", "mean_mse", "mean_mae", "mean_norm_mse"], [[k, v["n"], fmt(v["mean_mse"]), fmt(v["mean_mae"]), fmt(v["mean_normalized_mse"])] for k, v in e6_agg.items()]))
        lines.append("")

        e6_baseline_records = []
        for run in e6.get("runs", []):
            for baseline_name, baseline in (run.get("matched_baselines") or {}).items():
                metric = baseline.get("val", {}) if isinstance(baseline, dict) else {}
                e6_baseline_records.append([run.get("dataset"), baseline_name, fmt(metric.get("mse")), fmt(metric.get("mae"))])
        if e6_baseline_records:
            lines.append("E6 matched pooled raw baselines:")
            lines.append(make_code_table(["dataset", "baseline", "mse", "mae"], e6_baseline_records))
            lines.append("- SELF_ONLY neural condition은 history의 모든 raw channel을 섞으므로 no-other-channel baseline으로 해석하지 않는다.")
            lines.append("")

    if bolt.get("plans") or bolt.get("runs"):
        lines.append("Bolt helper status:")
        bolt_records = []
        for run in bolt.get("runs", []):
            bolt_records.append([run.get("dataset"), run.get("experiment"), run.get("status"), run.get("relative_path")])
        for plan in bolt.get("plans", []):
            bolt_records.append(["plan-only", "E5_BOLT", "NOT_ADOPTED_RESULT", plan.get("relative_path")])
        lines.append(make_code_table(["dataset", "experiment", "status", "path"], bolt_records))
        lines.append("- plan-only 파일은 과학 결과로 채택하지 않는다.")
        lines.append("")

    if optional_notes:
        lines.append("## 조건부 미실행·게이트 메모")
        lines.append("")
        lstm = optional_notes.get("LSTM")
        if lstm:
            lines.append("LSTM optional probe:")
            lines.append(make_code_table(["status", "execute", "scope", "reason", "path"], [[lstm.get("status"), lstm.get("execute"), lstm.get("scope"), lstm.get("reason"), lstm.get("relative_path")]]))
            lines.append("- 작은 LSTM은 Jena d16 pair에서 joint가 SELF보다 엄격히 낫다는 gate가 통과하지 않아 NOT_APPLICABLE로 남긴다. 이는 LSTM family rejection이 아니라 현재 조건에서의 보류다.")
            lines.append("")
        robin_e6 = optional_notes.get("E6_Robin")
        if robin_e6:
            lines.append("Robin E6 graph gate:")
            lines.append(make_code_table(["status", "lag_signal", "reason", "path"], [[robin_e6.get("status"), robin_e6.get("lag_signal"), robin_e6.get("reason"), robin_e6.get("relative_path")]]))
            lines.append("- Robin E6는 E2 gate에서 NO_LINEAR_32LAG_SIGNAL이면 NN 실패가 아니라 사전 gate skip으로 해석한다.")
            lines.append("")

    lines.append("## 미완료·보류 항목")
    lines.append("")
    for reason in [e34.get("held_reason"), e5.get("held_reason"), bolt.get("held_reason"), *e6.get("held_reasons", []), *cost.get("held_reasons", [])]:
        if reason:
            lines.append(f"- {reason}")
    for failure in e6.get("failures", []):
        lines.append(f"- E6 failed attempt preserved: {failure.get('relative_path')} / reason={failure.get('failure_reason')}")
    lines.append("")

    lines.append("## 비용과 장비")
    lines.append("")
    if cost.get("runs"):
        lines.append("COST_CLOSURE timing receipts, cost-only and excluded from science counts:")
        cost_records = []
        for run in cost.get("runs", []):
            devices = sorted({str((row.get("end_to_end") or {}).get("device")) for row in run.get("rows", []) if (row.get("end_to_end") or {}).get("device")})
            e2e_p50 = [finite_float((row.get("end_to_end") or {}).get("seconds_per_window_p50")) for row in run.get("rows", [])]
            e2e_p50 = [x for x in e2e_p50 if x is not None]
            cached_p50 = [finite_float((row.get("cached_head_only") or {}).get("seconds_per_window_p50")) for row in run.get("rows", [])]
            cached_p50 = [x for x in cached_p50 if x is not None]
            reused = run.get("reused_Jena_cost") or {}
            cost_records.append([
                run.get("part"), run.get("status"), run.get("row_count"), run.get("expected_row_count"),
                run.get("parity_flags"), reused.get("measured_rows_reused", ""), ",".join(devices),
                f"{fmt(min(e2e_p50))}..{fmt(max(e2e_p50))}" if e2e_p50 else "NA",
                f"{fmt(min(cached_p50))}..{fmt(max(cached_p50))}" if cached_p50 else "NA",
                run.get("measured_conditions_subset") or "",
                run.get("relative_path"),
            ])
        lines.append(make_code_table(["part", "status", "rows", "expected", "parity_flags", "reused_jena_rows", "device", "e2e_sec/window_p50_range", "cached_sec/window_p50_range", "subset", "path"], cost_records))
        lines.append("- Timing은 physical inverse와 CPU output을 포함하지만 model loading, disk, guard 비용은 제외한다. CPU/GPU, batch, dataset, timer boundary는 집계하지 않고 분리해 읽는다. `reused_Jena_cost`가 있으면 새 측정이 아니라 hash-bound 재사용 행으로 표시한다.")
        lines.append("- COST_CLOSURE는 science_updates=0, scores_computed=false 조건의 비용 측정이다. parity flag가 남은 row는 비용 파일 provenance와 분리해 보존하며, precision-matched timing으로 주장하지 않는다.")
        lines.append("")
    lines.append(f"- CPU ledger total seconds observed: {fmt(resources.get('cpu_wall_seconds'))}")
    lines.append(f"- GPU ledger total seconds observed: {fmt(resources.get('gpu_wall_seconds'))}")
    lines.append(f"- transient/open resource ledgers excluded from immutable manifest: {resources.get('transient_resource_snapshot_count')}")
    lines.append("- staged E6가 있으면 completed CPU preparation ledger와 later GPU loading/training ledger를 모두 비용으로 해석한다.")
    lines.append("- 각 result의 `compute_device`/`device` 필드는 manifest와 CSV에 보존한다.")
    lines.append("")

    lines.append("## 산출물")
    lines.append("")
    lines.append("- `MANIFEST_v4.json`: 입력 파일, sha, source handles, held/failure 상태")
    lines.append("- `raw_metrics_v4.csv`: 원수치 long-form CSV")
    lines.append("- `cost_closure_timing_v4.csv`: cost-only timing long-form CSV, science count 제외")
    lines.append("- `PROFESSOR_SUMMARY_KO_v4.md`: 교수님 공유용 요약")
    lines.append("- `REPORT_GENERATION_RECEIPT_v4.json`: report Budget 종료 후 외부 closure receipt로 생성된다.")
    for fig in figures[:8]:
        rel_fig = Path(fig['path']).relative_to(out_dir).as_posix()
        title = fig['title']
        lines.append(f"- [{rel_fig}]({rel_fig}): {title}")
        lines.append(f"  ![{title}]({rel_fig})")
    lines.append("")

    report = out_dir / "REPORT_KO_v4.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prof: list[str] = []
    prof.append("# 교수님 공유용 요약 v4")
    prof.append("")
    prof.append("현재 채택 가능한 결론은 다음과 같습니다.")
    prof.append("")
    for ans in answers:
        prof.append(f"- {ans['question']} {ans['answer']}")
    prof.append("")
    scope_bits = [
        f"E3/E4 adopted cases={e34.get('case_count')}",
        f"E5 adopted runs={e5.get('run_count')}",
        f"E6 adopted runs={e6.get('run_count')}",
        f"Bolt adopted runs={bolt.get('run_count')}",
        f"Cost closure adopted runs={cost.get('run_count')}",
    ]
    prof.append("핵심 제한은 TEST 미사용, 반복 VAL 진단, heterogeneous channel units, final proof PASS 항목만 채택한 범위입니다: " + ", ".join(scope_bits) + ". REF latent는 분리 진단일 뿐 deployable bound가 아닙니다.")
    verification = payload.get("verification_decision") or {}
    prof.append(f"검증 proof status={verification.get('status')}, excluded={len(verification.get('excluded') or [])}.")
    prof.append("")
    if e6_agg:
        prof.append("E6 그래프 결과가 포함되어 있으므로 mode별 비교는 본 보고서의 E6 원수치 표를 기준으로 해석하면 됩니다.")
    else:
        prof.append("E6 그래프 결과가 없거나 실패만 있으면 동적 그래프 필요성은 보류해야 합니다.")
    summary = out_dir / "PROFESSOR_SUMMARY_KO_v4.md"
    summary.write_text("\n".join(prof) + "\n", encoding="utf-8")
    return report, summary


def collect_all(verification_proof: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[Row]]:
    manifest: list[dict[str, Any]] = []
    rows: list[Row] = []
    adoption = collect_adoption_v2(manifest)
    e1 = collect_e1(adoption, manifest)
    e2 = collect_e2(adoption, rows, manifest)
    e34 = collect_e34(rows, manifest)
    e5 = collect_e5(rows, manifest)
    e6 = collect_e6(rows, manifest)
    bolt = collect_bolt_helpers(rows, manifest)
    verification_decision = verification_allowed_sources(verification_proof)
    cost = collect_cost_closure(manifest)
    optional_notes = collect_optional_scope_notes(manifest)
    filter_payload_by_verification(e34, e5, e6, bolt, cost, verification_decision)
    rows = filter_rows_by_verification(rows, verification_decision)
    e5_vs_e2 = collect_e5_vs_e2_jena_baseline(e2, e5, rows, manifest)
    resources = collect_resources(manifest)
    answers = answer_six_questions(e1, e2, e34, e5, e6, bolt, rows, e5_vs_e2)
    payload = {
        "generated_unix": time.time(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "repo_root": str(ROOT),
        "scope": "actual files only; no science execution, no TEST access, no existing report/evidence overwrite",
        "E1": e1,
        "E2": e2,
        "E3_E4": e34,
        "E5": e5,
        "E5_vs_E2_Jena_baseline": e5_vs_e2,
        "E6": e6,
        "Bolt_helpers": bolt,
        "Cost_closure": cost,
        "Optional_scope_notes": optional_notes,
        "resources": resources,
        "verification_decision": {k: (sorted(v) if isinstance(v, set) else v) for k, v in verification_decision.items()},
        "six_question_answers": answers,
        "manifest": manifest,
        "output_contract": {
            "fresh_v4_subdir_only": True,
            "figures_max": 7,
            "figure_labels": "English",
            "korean_pipe_tables": False,
            "protected_TEST_access": "not performed by this helper",
        },
    }
    return payload, rows


def run(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    if args.run and (not args.verify or not args.verify_sha):
        raise ValueError("--run requires --verify and --verify-sha so report adoption is bound to final verification proof")

    if args.dry_run or not args.run:
        verification_proof = load_verification_proof(args.verify, args.verify_sha) if args.verify else None
        payload, rows = collect_all(verification_proof)
        elapsed = time.perf_counter() - start
        if elapsed > args.max_seconds:
            raise TimeoutError(f"collection exceeded --max-seconds={args.max_seconds}: {elapsed:.3f}s")
        preview = {
            "mode": "dry_run",
            "elapsed_seconds": elapsed,
            "metric_rows": len(rows),
            "actual_inputs": len(payload["manifest"]),
            "E34_case_count": payload["E3_E4"]["case_count"],
            "E5_run_count": payload["E5"]["run_count"],
            "E6_run_count": payload["E6"]["run_count"],
            "E6_failures": len(payload["E6"]["failures"]),
            "Bolt_helper_run_count": payload["Bolt_helpers"]["run_count"],
            "Cost_closure_run_count": payload["Cost_closure"]["run_count"],
            "verification_status": value_at(payload, "verification_decision", "status"),
            "verification_excluded_count": len(value_at(payload, "verification_decision", "excluded") or []),
            "would_write": [
                "MANIFEST_v4.json",
                "raw_metrics_v4.csv",
                "cost_closure_timing_v4.csv",
                "REPORT_KO_v4.md",
                "PROFESSOR_SUMMARY_KO_v4.md",
                "figures/*.png",
            ],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    with Budget(2, gpu=False, label="FINAL_REPORT_v4 verified report generation") as budget:
        protected = access_guard()
        verification_proof = load_verification_proof(args.verify, args.verify_sha)
        payload, rows = collect_all(verification_proof)
        payload["protected_test_paths_registered"] = protected
        elapsed = time.perf_counter() - start
        if elapsed > args.max_seconds:
            raise TimeoutError(f"collection exceeded --max-seconds={args.max_seconds}: {elapsed:.3f}s")
        out_dir = Path(args.out_dir) if args.out_dir else HERE / f"final_report_v4_{time.strftime('%Y%m%d_%H%M%S')}"
        if not out_dir.is_absolute():
            out_dir = (HERE / out_dir).resolve()
        if not str(out_dir.resolve()).startswith(str(HERE.resolve())):
            raise ValueError("out_dir must stay inside the new campaign directory")
        out_dir.mkdir(parents=False, exist_ok=False)

        csv_path = write_csv(out_dir, rows)
        cost_csv_path = write_cost_csv(out_dir, payload.get("Cost_closure") or {})
        budget.check()
        figures = make_figures(out_dir, rows, payload["E1"], payload["E2"], payload["E5"], payload["E6"])
        budget.check()
        payload["outputs"] = {
            "out_dir": str(out_dir),
            "raw_metrics_csv": str(csv_path),
            "raw_metrics_csv_sha256": sha256(csv_path),
            "cost_closure_timing_csv": str(cost_csv_path),
            "cost_closure_timing_csv_sha256": sha256(cost_csv_path),
            "figures": figures,
        }
        report, summary = write_report(out_dir, payload, rows, figures)
        budget.check()
        payload["outputs"]["report_ko"] = str(report)
        payload["outputs"]["report_ko_sha256"] = sha256(report)
        payload["outputs"]["professor_summary_ko"] = str(summary)
        payload["outputs"]["professor_summary_ko_sha256"] = sha256(summary)
        report_budget_ledger_path = Path(budget.path)
        payload["report_budget_ledger"] = str(report_budget_ledger_path)
        payload["report_budget_ledger_status_at_manifest_write"] = file_record(report_budget_ledger_path, "report_budget_ledger_running_or_open")
        payload["resource_totals_exclude_current_report_generation"] = True
        manifest_path = out_dir / "MANIFEST_v4.json"
        elapsed_after_generation = time.perf_counter() - start
        if elapsed_after_generation > args.max_seconds:
            payload["status"] = "FAILED_PRESERVED_TIMEOUT_AFTER_GENERATION"
            payload["timeout_reason"] = f"report generation exceeded --max-seconds={args.max_seconds}: {elapsed_after_generation:.3f}s"
            write_json(manifest_path, payload)
            raise TimeoutError(payload["timeout_reason"])
        write_json(manifest_path, payload)
    closed_ledger = file_record(report_budget_ledger_path, "report_budget_ledger_closed")
    receipt = {
        "generated_unix": time.time(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "report_budget_ledger_closed": closed_ledger,
        "input_resource_cpu_wall_seconds_excluding_current_report": payload.get("resources", {}).get("cpu_wall_seconds"),
        "input_resource_gpu_wall_seconds_excluding_current_report": payload.get("resources", {}).get("gpu_wall_seconds"),
        "report_generation_wall_seconds": read_json(report_budget_ledger_path).get("wall_seconds") if report_budget_ledger_path.exists() else None,
        "note": "External closure receipt written after Budget context closes; MANIFEST_v4 immutable input totals intentionally exclude this report-generation ledger.",
    }
    receipt_path = out_dir / "REPORT_GENERATION_RECEIPT_v4.json"
    write_json(receipt_path, receipt)
    print(json.dumps({
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "report": str(report),
        "summary": str(summary),
        "csv": str(csv_path),
        "cost_csv": str(cost_csv_path),
        "figures": [f["path"] for f in figures],
        "report_generation_receipt": str(receipt_path),
        "report_generation_receipt_sha256": sha256(receipt_path),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final v4 reports from actual past-only diagnosis artifacts.")
    parser.add_argument("--run", action="store_true", help="Write a fresh v4 output subdirectory.")
    parser.add_argument("--dry-run", action="store_true", help="Collect manifest and print summary without writing outputs.")
    parser.add_argument("--out-dir", default="", help="Fresh output directory inside the campaign folder. Defaults to final_report_v4_YYYYmmdd_HHMMSS when --run is used.")
    parser.add_argument("--verify", default="", help="FINAL_VERIFY_v4.json path produced before report generation.")
    parser.add_argument("--verify-sha", default="", help="Expected sha256 for --verify.")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="Collection/reporting CPU wall guard for this helper.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))

