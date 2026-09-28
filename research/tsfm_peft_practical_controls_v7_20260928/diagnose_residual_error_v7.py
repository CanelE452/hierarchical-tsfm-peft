from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from runtime_v7 import HERE, ROOT, Job, digest, environment_receipt, load_data, save_json, source_receipt

CONTEXT = 512
HORIZON = 48
PERIODS = ("test_a", "test_b")
SEEDS = (92601, 92602)
FEATURES = ("last", "recent24_mean", "recent24_std", "recent24_change")
Q_OUTCOMES = ("q_future_mean_signed_error", "q_future_mse")
PERIOD_COLORS = {"test_a": "#1f77b4", "test_b": "#ff7f0e"}


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
        raise ValueError(f"Sealed input changed: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"path": rel(path), "sha256": digest(path), "rows": len(rows)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def comp_seed(identifier: str) -> int | None:
    for seed in SEEDS:
        if str(seed) in identifier:
            return seed
    return None


def role_member_ids(evaluation: dict[str, Any], role: str) -> list[str]:
    return [identifier for identifier in evaluation.get("roles", {}).get(role, []) if identifier in evaluation["models"]]


def load_prediction(item: dict[str, Any], origins: np.ndarray) -> np.ndarray:
    path = check_receipt(item["prediction"])
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Prediction origins mismatch for {path}")
        return archive["prediction"].astype(np.float32)


def load_level_predictions(evaluation: dict[str, Any], origins: np.ndarray) -> dict[str, np.ndarray]:
    ids = role_member_ids(evaluation, "level_res")
    if len(ids) != 2:
        raise ValueError(f"Expected exactly two LEVEL_RES members, found {ids}")
    return {identifier: load_prediction(evaluation["models"][identifier], origins) for identifier in ids}


def period_index(origins_by_period: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, str]]:
    origins = np.concatenate([origins_by_period[p] for p in PERIODS])
    period_indices: dict[str, np.ndarray] = {}
    start = 0
    for period in PERIODS:
        end = start + len(origins_by_period[period])
        period_indices[period] = np.arange(start, end, dtype=np.int64)
        start = end
    index_to_period = {int(i): period for period, indices in period_indices.items() for i in indices}
    return origins, period_indices, index_to_period


def context_feature_rows(corrected: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    item = corrected["files"]["residual_context_rows"]
    path = check_receipt(item)
    return read_csv_rows(path), receipt(path)


def context_map(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int], dict[str, str]]:
    mapped: dict[tuple[str, int, int, int], dict[str, str]] = {}
    for row in rows:
        key = (row["period"], int(row["origin_position"]), int(row["origin"]), int(row["channel_index"]))
        mapped[key] = row
    return mapped


def q_outcome_rows(data: dict[str, Any], evaluation: dict[str, Any], context_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    origins, period_indices, index_to_period = period_index(origins_by_period)
    target = np.stack([data["x"][o:o + HORIZON] for o in origins]).astype(np.float32)
    mask = np.stack([data["finite"][o:o + HORIZON] for o in origins]).astype(bool)
    complete = mask.all(axis=2)
    columns = np.asarray(data["columns"]).astype(str).tolist()
    basis = np.asarray(data["basis"], dtype=np.float64)
    projector = basis @ basis.T
    q_projector = np.eye(projector.shape[0], dtype=np.float64) - projector
    predictions = load_level_predictions(evaluation, origins)
    context = context_map(context_rows)
    per_member: dict[str, dict[str, np.ndarray]] = {}
    for identifier, pred in predictions.items():
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape mismatch for {identifier}: {pred.shape} vs {target.shape}")
        signed = np.full((len(origins), len(columns)), np.nan, dtype=np.float64)
        mse = np.full((len(origins), len(columns)), np.nan, dtype=np.float64)
        counts = complete.sum(axis=1).astype(np.int64)
        for n in range(len(origins)):
            valid_h = complete[n]
            if not valid_h.any():
                continue
            error = pred[n, valid_h].astype(np.float64) - target[n, valid_h].astype(np.float64)
            eq = error @ q_projector
            signed[n] = eq.mean(axis=0)
            mse[n] = (eq * eq).mean(axis=0)
        per_member[identifier] = {"signed": signed, "mse": mse, "counts": counts}
    member_ids = list(per_member)
    rows: list[dict[str, Any]] = []
    for n, origin in enumerate(origins):
        period = index_to_period[int(n)]
        complete_count = int(complete[n].sum())
        if complete_count == 0:
            continue
        for c, channel in enumerate(columns):
            key = (period, int(n), int(origin), int(c))
            feature_row = context.get(key)
            if feature_row is None:
                raise KeyError(f"Missing residual context features for {key}")
            signed_values = np.array([per_member[mid]["signed"][n, c] for mid in member_ids], dtype=np.float64)
            mse_values = np.array([per_member[mid]["mse"][n, c] for mid in member_ids], dtype=np.float64)
            if not np.all(np.isfinite(signed_values)) or not np.all(np.isfinite(mse_values)):
                continue
            row: dict[str, Any] = {
                "scope": "mean_seed_q_outcome",
                "period": period,
                "origin_position": int(n),
                "origin": int(origin),
                "channel_index": int(c),
                "channel": channel,
                "complete_horizon_count": complete_count,
                "member_count": len(member_ids),
                "member_ids": ";".join(member_ids),
                "q_future_mean_signed_error": float(signed_values.mean()),
                "q_future_mse": float(mse_values.mean()),
                "q_contract": "prediction-target is projected through Q=I-BB^T on target-complete horizons before seed losses are averaged; no target zero-fill",
            }
            for feature in FEATURES:
                row[feature] = float(feature_row[feature])
            rows.append(row)
    coverage_by_period: dict[str, dict[str, Any]] = {}
    for period, indices in period_indices.items():
        sub = complete[indices]
        counts = sub.sum(axis=1)
        coverage_by_period[period] = {
            "origins": int(len(indices)),
            "origin_horizon_rows": int(sub.size),
            "complete_origin_horizon_rows": int(sub.sum()),
            "coverage_fraction": float(sub.sum() / sub.size),
            "min_complete_horizons_per_origin": int(counts.min()) if len(counts) else None,
            "max_complete_horizons_per_origin": int(counts.max()) if len(counts) else None,
        }
    coverage = {
        "origin_channel_rows": int(len(origins) * len(columns)),
        "rows_with_q_outcomes": int(len(rows)),
        "origin_horizon_rows": int(complete.size),
        "complete_origin_horizon_rows": int(complete.sum()),
        "coverage_fraction": float(complete.sum() / complete.size),
        "same_complete_horizon_mask_all_channels_and_level_seeds": True,
        "by_period": coverage_by_period,
    }
    prediction_receipts = {identifier: evaluation["models"][identifier]["prediction"] for identifier in member_ids}
    return rows, coverage, {"level_prediction_receipts": prediction_receipts}


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    values = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and values[end] == values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        for outcome in Q_OUTCOMES:
            groups.setdefault((row["period"], int(row["channel_index"]), outcome), []).append(row)
    output: list[dict[str, Any]] = []
    for (period, channel_index, outcome), group in sorted(groups.items()):
        channel = group[0]["channel"]
        y = np.array([float(r[outcome]) for r in group], dtype=np.float64)
        for feature in FEATURES:
            x = np.array([float(r[feature]) for r in group], dtype=np.float64)
            output.append({
                "period": period,
                "channel_index": channel_index,
                "channel": channel,
                "feature": feature,
                "outcome": outcome,
                "n": int(len(group)),
                "pearson": pearson(x, y),
                "spearman": pearson(rankdata(x), rankdata(y)),
                "row_key_fields": "period,origin_position,origin,channel_index",
                "semantics": "mean seed LEVEL_RES Q residual-correction error on target-complete horizons; descriptive exposed-TEST diagnostic only",
            })
    return output


def save_scatter_atlas(rows: list[dict[str, Any]], x_feature: str, y_outcome: str, output_stem: Path) -> dict[str, dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted({(int(row["channel_index"]), row["channel"]) for row in rows})
    fig, axes = plt.subplots(5, 4, figsize=(13.5, 12.0), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, (channel_index, channel) in zip(axes_flat, channels):
        subset = [row for row in rows if int(row["channel_index"]) == channel_index]
        for period in PERIODS:
            period_rows = [row for row in subset if row["period"] == period]
            ax.scatter(
                [float(row[x_feature]) for row in period_rows],
                [float(row[y_outcome]) for row in period_rows],
                s=12,
                alpha=0.75,
                color=PERIOD_COLORS[period],
                label=period,
                linewidths=0,
            )
        if "signed" in y_outcome:
            ax.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.7)
        ax.set_title(channel.replace("Robin_office_", ""), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes_flat[len(channels):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"Robin LEVEL_RES Q diagnostic: {x_feature} vs {y_outcome}\n"
        "Mean of seed-level Q errors; target-complete horizons only; exposed TEST diagnostic",
        fontsize=11,
    )
    fig.supxlabel(x_feature, fontsize=10)
    fig.supylabel(y_outcome, fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, str]] = {}
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        fig.savefig(path, dpi=220)
        receipts[suffix] = receipt(path)
    plt.close(fig)
    return receipts


def run(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    evaluation = read_json(args.evaluation)
    if evaluation.get("dataset") != "robin" or evaluation.get("status") != "complete":
        raise ValueError("Complete Robin v7 evaluation is required")
    check_receipt(evaluation["seal"])
    corrected = read_json(args.corrected_diagnostics)
    if corrected.get("status") != "complete" or corrected.get("original_invalid_scope") != "original_diagnostic_invalid_for_mean_seed_conclusions":
        raise ValueError("Corrected mean-seed diagnostic manifest is required")
    context_rows, context_receipt = context_feature_rows(corrected)
    data = load_data("robin", include_test=True)
    out_dir = HERE / "diagnostics_residual_error"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite supplemental diagnostics directory: {out_dir}")
    out_dir.mkdir(parents=True)
    rows, coverage, prediction_inputs = q_outcome_rows(data, evaluation, context_rows)
    job.heartbeat("computed mean-seed Q outcomes")
    files: dict[str, Any] = {}
    q_rows_path = out_dir / "robin_level_res_q_residual_error_rows_v7.csv"
    files["q_residual_error_rows"] = write_csv(q_rows_path, rows)
    corr_path = out_dir / "robin_level_res_q_residual_error_correlations_v7.csv"
    files["q_residual_error_correlations"] = write_csv(corr_path, correlation_rows(rows))
    job.heartbeat("wrote Q residual error CSVs")
    figures = {
        "last_residual_to_q_signed_error": save_scatter_atlas(rows, "last", "q_future_mean_signed_error", out_dir / "robin_last_residual_to_q_signed_error_atlas_v7"),
        "recent24std_to_q_mse": save_scatter_atlas(rows, "recent24_std", "q_future_mse", out_dir / "robin_recent24std_to_q_mse_atlas_v7"),
    }
    result = {
        "schema": "tsfm_peft_practical_controls_v7_robin_q_residual_error_diagnostic_v1",
        "status": "complete",
        "created_utc": time.time(),
        "evaluation": receipt(args.evaluation),
        "corrected_diagnostics": receipt(args.corrected_diagnostics),
        "context_feature_rows": context_receipt,
        "prediction_inputs": prediction_inputs,
        "source": source_receipt(),
        "environment": environment_receipt(),
        "files": files,
        "figures": figures,
        "coverage": coverage,
        "aggregation_contract": "For each LEVEL_RES seed, prediction-target errors are projected through Q=I-BB^T, squared/signed errors are averaged over target-complete horizons per origin/channel, then seed losses are averaged. No prediction ensemble and no target zero-fill are used.",
        "feature_contract": "Past residual context features are reused from corrected diagnostics; no new feature definition or model inference is introduced.",
        "interpretation_scope": "Descriptive exposed-TEST diagnosis for approved PLAN A3 only; not a new selection, routing, or causal claim.",
    }
    result_path = HERE / "robin_q_residual_error_diagnostic_v7.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    save_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplemental Robin LEVEL_RES Q residual-correction error diagnosis.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--corrected-diagnostics", type=Path, default=HERE / "robin_diagnostics_corrected_v7.json")
    parser.add_argument("--reserve-seconds", type=float, default=240)
    args = parser.parse_args()
    with Job("v7_diagnose_robin_q_residual_error", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": "robin", "diagnostic": "q_residual_error_context"}) as job:
        result = run(args, job)
        job.heartbeat("Q residual-error diagnostic and scatter atlases complete")
    print(json.dumps({"status": result["status"], "result": receipt(HERE / "robin_q_residual_error_diagnostic_v7.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
