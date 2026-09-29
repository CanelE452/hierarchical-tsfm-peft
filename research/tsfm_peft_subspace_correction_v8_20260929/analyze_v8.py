from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

from runtime_v8 import HERE, ROOT, Job, digest, environment_receipt, save_json, source_receipt

PERIODS = ("test_a", "test_b", "combined")
PRIMARY_ROLES = ("pq_split", "raw64", "level_res", "level_raw", "direct_nlinear", "f0", "lora")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()


def receipt(path: str | Path) -> dict[str, str]:
    resolved = resolve(path)
    return {"path": str(resolved), "sha256": digest(resolved)}


def check_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["path"])
    actual = digest(path)
    if actual != item["sha256"]:
        raise ValueError(f"Sealed input changed: {path} expected {item['sha256']} got {actual}")
    return path


def validate_name(name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(char not in allowed for char in name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen name")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": rel(path), "sha256": digest(path), "rows": len(rows), "columns": fieldnames}


def require_complete_evaluation(path: Path, dataset: str) -> dict[str, Any]:
    evaluation = read_json(path)
    if evaluation.get("status") != "complete" or evaluation.get("dataset") != dataset:
        raise ValueError(f"{dataset} evaluation is not complete: {path}")
    check_receipt(evaluation["seal"])
    check_receipt(evaluation["first_exposure"])
    return evaluation


def role_rows(evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, evaluation in evaluations.items():
        for role, periods in sorted(evaluation.get("role_scores", {}).items()):
            for period in PERIODS:
                score = periods.get(period)
                if not score:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "role": role,
                        "period": period,
                        "mse": score.get("mse"),
                        "mae": score.get("mae"),
                        "member_count": score.get("count"),
                        "run_ids": ";".join(score.get("run_ids", [])),
                        "scope": score.get("scope"),
                    }
                )
    return rows


def model_rows(evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, evaluation in evaluations.items():
        for identifier, model in sorted(evaluation.get("models", {}).items()):
            spec = model.get("spec", {})
            for period in PERIODS:
                score = model.get("scores", {}).get(period)
                if not score:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model_id": identifier,
                        "role": model.get("role"),
                        "period": period,
                        "seed": spec.get("seed"),
                        "family": spec.get("family"),
                        "kind": model.get("kind"),
                        "reused_v7_prediction": model.get("reused_v7_prediction"),
                        "mse": score.get("mse"),
                        "mae": score.get("mae"),
                        "missing_channel_indices": ";".join(str(i) for i in score.get("missing_channel_indices", [])),
                    }
                )
    return rows


def space_rows(evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, evaluation in evaluations.items():
        for role, periods in sorted(evaluation.get("role_space_scores", {}).items()):
            for period in PERIODS:
                score = periods.get(period)
                if not score:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "role": role,
                        "period": period,
                        "p_retained_space_mse": score.get("p_retained_space_mse"),
                        "q_residual_space_mse": score.get("q_residual_space_mse"),
                        "p_retained_space_mae": score.get("p_retained_space_mae"),
                        "q_residual_space_mae": score.get("q_residual_space_mae"),
                        "complete_target_rows": score.get("complete_target_rows"),
                        "member_count": score.get("count"),
                        "run_ids": ";".join(score.get("run_ids", [])),
                        "scope": score.get("scope"),
                    }
                )
    return rows


def comparison_rows(evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, evaluation in evaluations.items():
        for comparison in evaluation.get("comparisons", []):
            for period, detail in comparison.get("periods", {}).items():
                full = detail.get("full") or {}
                interval = detail.get("conditional_block95")
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": comparison.get("name"),
                        "period": period,
                        "primary": comparison.get("primary"),
                        "left_ids": ";".join(comparison.get("left", [])),
                        "right_ids": ";".join(comparison.get("right", [])),
                        "left_mse": full.get("left_mse"),
                        "reference_mse": full.get("reference_mse"),
                        "delta_mse": full.get("delta_mse"),
                        "relative_to_reference_pct": full.get("relative_to_reference_pct"),
                        "delta_mse_ci_low": None if interval is None else interval["delta_mse"][0],
                        "delta_mse_ci_high": None if interval is None else interval["delta_mse"][1],
                        "relative_pct_ci_low": None if interval is None else interval["relative_to_reference_pct"][0],
                        "relative_pct_ci_high": None if interval is None else interval["relative_to_reference_pct"][1],
                        "interval_status": detail.get("interval_status"),
                        "question": comparison.get("question"),
                    }
                )
    return rows


def value_by_role(evaluation: dict[str, Any], role: str, metric: str = "mse", period: str = "combined") -> float | None:
    score = evaluation.get("role_scores", {}).get(role, {}).get(period, {})
    value = score.get(metric)
    return None if value is None else float(value)


def pct_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right == 0:
        return None
    return 100.0 * (left - right) / right


def dataset_key_summary(dataset: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    role_values = {
        role: {
            "combined_mse": value_by_role(evaluation, role, "mse"),
            "combined_mae": value_by_role(evaluation, role, "mae"),
        }
        for role in PRIMARY_ROLES
        if role in evaluation.get("role_scores", {})
    }
    anchors: dict[str, Any] = {}
    for reference in ("raw64", "level_res", "level_raw", "direct_nlinear", "f0", "lora"):
        anchors[f"pq_split_minus_{reference}_relative_pct"] = pct_delta(
            role_values.get("pq_split", {}).get("combined_mse"),
            role_values.get(reference, {}).get("combined_mse"),
        )
    for reference in ("level_raw", "level_res", "direct_nlinear", "f0", "lora"):
        anchors[f"raw64_minus_{reference}_relative_pct"] = pct_delta(
            role_values.get("raw64", {}).get("combined_mse"),
            role_values.get(reference, {}).get("combined_mse"),
        )
    primary_comparisons = [
        row
        for row in evaluation.get("comparisons", [])
        if row.get("primary") or row.get("name", "").endswith(("pq_split_vs_lora", "pq_split_vs_f0", "pq_split_vs_level_res", "pq_split_vs_level_raw"))
    ]
    return {
        "dataset": dataset,
        "roles_available": sorted(evaluation.get("role_scores", {})),
        "primary_roles_available": [role for role in PRIMARY_ROLES if role in evaluation.get("role_scores", {})],
        "combined_role_values": role_values,
        "combined_relative_mse_changes": anchors,
        "primary_or_anchor_comparisons": [row.get("name") for row in primary_comparisons],
        "new_test_model_predictions": evaluation.get("new_test_model_predictions"),
        "scope": "Exposed development TEST for v8; no independent confirmation claim.",
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = {
        "robin": require_complete_evaluation(args.robin_evaluation, "robin"),
        "jena": require_complete_evaluation(args.jena_evaluation, "jena"),
    }
    out_dir = HERE / args.name
    if out_dir.exists():
        raise FileExistsError("Preserve existing v8 analysis output; use a new name")
    out_dir.mkdir(parents=True)
    roles = write_csv(out_dir / "role_scores.csv", role_rows(evaluations))
    models = write_csv(out_dir / "model_scores.csv", model_rows(evaluations))
    spaces = write_csv(out_dir / "space_scores.csv", space_rows(evaluations))
    comparisons = write_csv(out_dir / "comparisons.csv", comparison_rows(evaluations))
    summary = {
        "schema": "tsfm_peft_subspace_correction_v8_analysis_v1",
        "status": "complete",
        "name": args.name,
        "created_utc": time.time(),
        "command": sys.argv,
        "source": source_receipt(),
        "environment": environment_receipt(),
        "inputs": {dataset: receipt(args.robin_evaluation if dataset == "robin" else args.jena_evaluation) for dataset in evaluations},
        "outputs": {
            "role_scores_csv": roles,
            "model_scores_csv": models,
            "space_scores_csv": spaces,
            "comparisons_csv": comparisons,
        },
        "dataset_summaries": {dataset: dataset_key_summary(dataset, evaluation) for dataset, evaluation in evaluations.items()},
        "interpretation_limits": [
            "Robin and Jena are exposed development evaluations, not independent protection sets.",
            "Role scores average individual seed losses and do not evaluate a seed ensemble.",
            "P/Q scores use complete target rows only and are diagnostic, not a replacement for masked macro MSE.",
        ],
    }
    summary_path = out_dir / "analysis_summary.json"
    save_json(summary_path, summary)
    return {"analysis": rel(summary_path), "sha256": digest(summary_path), "outputs": summary["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed v8 evaluations without loading arrays or models.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--robin-evaluation", type=Path, default=HERE / "robin_eval01.json")
    parser.add_argument("--jena-evaluation", type=Path, default=HERE / "jena_eval01.json")
    parser.add_argument("--reserve-seconds", type=float, required=True)
    args = parser.parse_args()
    validate_name(args.name)
    attempt_path = HERE / f"{args.name}_attempt.json"
    if attempt_path.exists():
        raise FileExistsError("Preserve existing analysis attempt; use a new name")
    attempt = {"status": "running", "started_utc": time.time(), "command": sys.argv, "source": source_receipt(), "environment": environment_receipt()}
    save_json(attempt_path, attempt)
    try:
        with Job(f"v8_analyze_{args.name}", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"name": args.name, "action": "analyze"}):
            result = run_analysis(args)
        attempt.update(status="complete", completed_utc=time.time(), result=result)
        save_json(attempt_path, attempt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        attempt.update(status="failed", failed_utc=time.time(), error=repr(exc))
        save_json(attempt_path, attempt)
        raise


if __name__ == "__main__":
    main()
