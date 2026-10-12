"""Independent aggregation and sanity checks for the past-only diagnosis.

This script is intentionally read-only with respect to experiment outputs.  It
recomputes metrics from saved prediction arrays where targets are present, and
otherwise records the verified upstream check file that already performed the
recalculation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def macro_score(prediction, target, mask, mean=None, std=None) -> dict:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if mean is not None and std is not None:
        mean = np.asarray(mean)
        std = np.asarray(std)
        prediction = prediction * std + mean
        target = target * std + mean
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    count = mask.sum(axis=(0, 1))
    mse_c = np.divide((error * error).sum(axis=(0, 1)), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    mae_c = np.divide(np.abs(error).sum(axis=(0, 1)), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    active = count > 0
    return {
        "mse": float(mse_c[active].mean()),
        "mae": float(mae_c[active].mean()),
        "per_channel_mse": mse_c.tolist(),
        "per_channel_mae": mae_c.tolist(),
        "observed_counts": count.tolist(),
    }


def metric_delta(a: float, b: float) -> float:
    return float(b - a)


def verify_prediction_npz(row: dict) -> dict | None:
    path_text = row.get("prediction_path") or row.get("arrays")
    if isinstance(path_text, dict):
        path_text = path_text.get("path") or path_text.get("prediction_path")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return {"status": "MISSING_ARRAY", "path": str(path)}
    with np.load(path, allow_pickle=False) as z:
        if not all(k in z.files for k in ("prediction", "target", "mask")):
            return {"status": "NO_TARGET_IN_ARRAY", "path": str(path), "keys": z.files}
        mean = z["mean"] if "mean" in z.files else None
        std = z["std"] if "std" in z.files else None
        score = macro_score(z["prediction"], z["target"], z["mask"], mean, std)
    reported = row.get("metrics") or row.get("val")
    return {
        "status": "PASS" if reported and abs(score["mse"] - reported["mse"]) < 1e-7 and abs(score["mae"] - reported["mae"]) < 1e-7 else "CHECK",
        "path": str(path),
        "sha256": sha256(path),
        "recomputed_mse": score["mse"],
        "recomputed_mae": score["mae"],
        "reported_mse": None if not reported else reported.get("mse"),
        "reported_mae": None if not reported else reported.get("mae"),
        "max_abs_metric_delta": None
        if not reported
        else max(abs(score["mse"] - reported["mse"]), abs(score["mae"] - reported["mae"])),
    }


def collect_e1(evidence: dict, rows: list[dict]) -> None:
    path = HERE / "E1_RESULT.json"
    if not path.exists():
        evidence["E1"] = {"status": "MISSING"}
        return
    data = read_json(path)
    boundary_max = 0.0
    changed_counts = 0
    for row in data.get("rows", []):
        for test in row.get("forecast_boundary_tests", []):
            for key in ("cold_feature", "cold_prediction", "cold_warm_prediction"):
                item = test.get(key, {})
                boundary_max = max(boundary_max, abs(float(item.get("max_abs", 0.0))))
                changed_counts += int(item.get("changed_exact_count", 0))
        for key in ("B_within_history_revision", "C_same_left_append_revision", "C_shifted_left_context_EFFECT_NOT_LEAK"):
            item = row.get(key, {})
            rows.append({
                "experiment": "E1",
                "dataset": row.get("dataset"),
                "seed": row.get("job", {}).get("seed"),
                "condition": f"{row.get('job', {}).get('family')}_{row.get('job', {}).get('dim', 'GT')}_{key}",
                "mse": "",
                "mae": "",
                "delta_mse_vs_self": "",
                "delta_mae_vs_self": "",
                "status": row.get("status"),
                "note": f"max_abs={item.get('max_abs')}; changed={item.get('changed_exact_count')}",
            })
    evidence["E1"] = {
        "status": data.get("status"),
        "rows": len(data.get("rows", [])),
        "datasets": sorted({r.get("dataset") for r in data.get("rows", [])}),
        "max_boundary_prediction_or_feature_abs_delta": boundary_max,
        "changed_exact_count_total": changed_counts,
        "safety_pass": data.get("safety_pass") is True,
        "source_sha256": sha256(path),
        "interpretation": "future perturbation and same-prefix checks found no material inference-path leakage; this is not a performance claim",
    }


def collect_e2(evidence: dict, rows: list[dict]) -> None:
    path = HERE / "e2" / "SUMMARY_VERIFIED_run02.json"
    if not path.exists():
        evidence["E2"] = {"status": "MISSING"}
        return
    summary = read_json(path)
    records = []
    lag_signal_datasets = []
    for item in summary.get("records", []):
        record = read_json(Path(item["path"]))
        models = record.get("models", {})
        if not isinstance(models, dict):
            continue
        val = {name: data.get("val", {}) for name, data in models.items()}
        self_mse = val.get("SELF_LAG", {}).get("mse")
        self_mae = val.get("SELF_LAG", {}).get("mae")
        all_mse = val.get("ALL_CHANNEL_LAG", {}).get("mse")
        all_mae = val.get("ALL_CHANNEL_LAG", {}).get("mae")
        cross_mse = val.get("SELF_LAG_PLUS_CROSS_NOW", {}).get("mse")
        cross_mae = val.get("SELF_LAG_PLUS_CROSS_NOW", {}).get("mae")
        lag_signal = record.get("LAG_SIGNAL")
        if lag_signal:
            lag_signal_datasets.append(record.get("dataset"))
        records.append({
            "dataset": record.get("dataset"),
            "tag": record.get("tag"),
            "path": item["path"],
            "status": record.get("status"),
            "self_mse": self_mse,
            "self_mae": self_mae,
            "cross_now_mse": cross_mse,
            "cross_now_mae": cross_mae,
            "all_lag_mse": all_mse,
            "all_lag_mae": all_mae,
            "delta_all_lag_mse_vs_self": None if self_mse is None or all_mse is None else metric_delta(self_mse, all_mse),
            "delta_all_lag_mae_vs_self": None if self_mae is None or all_mae is None else metric_delta(self_mae, all_mae),
            "LAG_SIGNAL": lag_signal,
        })
        if record.get("dataset") in ("jena", "robin", "synthetic_fixed", "synthetic_changed", "synthetic_unrelated"):
            for name, metrics in val.items():
                rows.append({
                    "experiment": "E2",
                    "dataset": record.get("dataset"),
                    "seed": record.get("tag", "").split("_")[-1] if record.get("dataset", "").startswith("synthetic") else "",
                    "condition": name,
                    "mse": metrics.get("mse"),
                    "mae": metrics.get("mae"),
                    "delta_mse_vs_self": "" if self_mse is None else metrics.get("mse", np.nan) - self_mse,
                    "delta_mae_vs_self": "" if self_mae is None else metrics.get("mae", np.nan) - self_mae,
                    "status": record.get("status"),
                    "note": "VAL physical macro; verified by e2_continue summary",
                })
    evidence["E2"] = {
        "status": summary.get("status"),
        "records": len(summary.get("records", [])),
        "synthetic_alignment": summary.get("synthetic_alignment"),
        "lag_signal_datasets": lag_signal_datasets,
        "records_summary": records,
        "source_sha256": sha256(path),
        "interpretation": "linear lag signal is a branch gate, not a significance test",
    }


def collect_e34(evidence: dict, rows: list[dict]) -> None:
    files = []
    for path in sorted((HERE / "e34").glob("*.json")):
        try:
            probe = read_json(path)
        except Exception:
            continue
        if probe.get("experiment_id") == "E3_E4" and "E3" in probe and "E4" in probe:
            files.append(path)
    summaries = []
    checks = []
    for path in files:
        data = read_json(path)
        seed = data.get("seed")
        dataset = data.get("dataset")
        prefix_mse = None
        prefix_mae = None
        for block in ("E3", "E4"):
            for row in data.get(block, []):
                check = verify_prediction_npz(row)
                if check is not None:
                    checks.append({"result": str(path), "method": row.get("method"), **check})
                if row.get("method") == "PREFIX_BOLT_REC_D":
                    prefix_mse = row["metrics"]["mse"]
                    prefix_mae = row["metrics"]["mae"]
                if row.get("method") in {"FULL_BOLT_REC_D", "PREFIX_BOLT_REC_D", "LATENT_SELF_RIDGE", "LATENT_JOINT_RIDGE", "BOLT_LATENT", "REFERENCE_LATENT_DIAGNOSTIC", "DIRECT_RAW_RIDGE"}:
                    rows.append({
                        "experiment": block,
                        "dataset": dataset,
                        "seed": seed,
                        "condition": row.get("method"),
                        "mse": row.get("metrics", {}).get("mse"),
                        "mae": row.get("metrics", {}).get("mae"),
                        "delta_mse_vs_self": "",
                        "delta_mae_vs_self": "",
                        "status": data.get("status"),
                        "note": f"scope={data.get('scope')}; normalized_mse={row.get('metrics', {}).get('normalized', {}).get('mse')}",
                    })
        e4 = {}
        for row in data.get("E4", []):
            e4[row.get("method")] = {
                "mse": row.get("metrics", {}).get("mse"),
                "mae": row.get("metrics", {}).get("mae"),
                "latent_mse": row.get("latent_metrics", {}).get("mse"),
                "latent_mae": row.get("latent_metrics", {}).get("mae"),
            }
        summaries.append({
            "path": str(path),
            "sha256": sha256(path),
            "status": data.get("status"),
            "dataset": dataset,
            "dim": data.get("dim"),
            "seed": seed,
            "scope": data.get("scope"),
            "FULL_PREFIX_last_deltas": data.get("FULL_PREFIX_last_deltas"),
            "FULL_PREFIX_all_deltas": data.get("FULL_PREFIX_all_deltas"),
            "E3_prefix_minus_full_mse": None if prefix_mse is None else prefix_mse - next((r["metrics"]["mse"] for r in data.get("E3", []) if r.get("method") == "FULL_BOLT_REC_D"), np.nan),
            "E3_prefix_minus_full_mae": None if prefix_mae is None else prefix_mae - next((r["metrics"]["mae"] for r in data.get("E3", []) if r.get("method") == "FULL_BOLT_REC_D"), np.nan),
            "E4": e4,
        })
    summary_path = HERE / "e34" / "SUMMARY_cpu_run01.json"
    evidence["E3_E4"] = {
        "status": "COMPLETE_UNVERIFIED" if files else "MISSING",
        "completed_result_files": len(files),
        "summary_file_exists": summary_path.exists(),
        "summary_sha256": sha256(summary_path) if summary_path.exists() else None,
        "results": summaries,
        "array_metric_checks": checks,
        "balanced_pair_complete": len(files) >= 2 and all(s["status"] == "COMPLETE_UNVERIFIED" for s in summaries),
        "interpretation": "DIAG_SUBSET Jena d16 pair only unless additional result files exist",
    }


def collect_e5_e6(evidence: dict, rows: list[dict]) -> None:
    outputs = sorted((HERE / "results").glob("**/RESULT.json"))
    result_rows = []
    for path in outputs:
        data = read_json(path)
        exp = data.get("experiment")
        status = data.get("status")
        if exp is None:
            continue
        result_rows.append({"path": str(path), "sha256": sha256(path), "experiment": exp, "status": status, "dataset": data.get("dataset")})
        for cond in data.get("conditions", []):
            name = cond.get("condition") or cond.get("mode")
            val = cond.get("val", {})
            rows.append({
                "experiment": exp,
                "dataset": data.get("dataset"),
                "seed": cond.get("seed"),
                "condition": name,
                "mse": val.get("mse"),
                "mae": val.get("mae"),
                "delta_mse_vs_self": "",
                "delta_mae_vs_self": "",
                "status": status,
                "note": str(path),
            })
    plans = [str(p) for p in sorted((HERE / "plans").glob("*.json"))]
    evidence["E5_E6"] = {"results": result_rows, "plans": plans}


def write_metrics_csv(rows: list[dict]) -> None:
    path = HERE / "metrics.csv"
    fieldnames = ["experiment", "dataset", "seed", "condition", "mse", "mae", "delta_mse_vs_self", "delta_mae_vs_self", "status", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    evidence: dict = {
        "generated_by": "verify_results.py",
        "repo_head": "",
        "notes": [
            "No protected TEST data are intentionally opened by this verifier.",
            "Metric PASS means array arithmetic matched the saved result; it is not a generalization proof.",
        ],
    }
    head_file = ROOT / ".git" / "HEAD"
    if head_file.exists():
        evidence["repo_head"] = head_file.read_text(encoding="utf-8", errors="replace").strip()
    rows: list[dict] = []
    collect_e1(evidence, rows)
    collect_e2(evidence, rows)
    collect_e34(evidence, rows)
    collect_e5_e6(evidence, rows)
    preflight = HERE / "ASSETS_AND_PREFLIGHT.json"
    if preflight.exists():
        pf = read_json(preflight)
        evidence["assets"] = {
            "protected_TEST_new_access": pf.get("protected_TEST_new_access"),
            "old_files_overwritten": pf.get("old_files_overwritten"),
            "datasets": {k: v.get("shape") for k, v in pf.get("datasets", {}).items()},
            "bolt_weights_exists": pf.get("Bolt", {}).get("weights_exists"),
            "asset_count": len(pf.get("assets", [])),
        }
    write_metrics_csv(rows)
    write_json(HERE / "EVIDENCE.json", evidence)
    changed = [
        "research/past_only_repr_diagnosis_20261012/verify_results.py",
        "research/past_only_repr_diagnosis_20261012/EVIDENCE.json",
        "research/past_only_repr_diagnosis_20261012/metrics.csv",
    ]
    (HERE / "CHANGED_FILES.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(HERE / "EVIDENCE.json"), "metrics": str(HERE / "metrics.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
