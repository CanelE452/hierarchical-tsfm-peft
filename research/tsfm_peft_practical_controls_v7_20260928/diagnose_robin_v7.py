
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from runtime_v7 import HERE, ROOT, Job, digest, environment_receipt, load_data, save_json, source_receipt

CONTEXT = 512
HORIZON = 48
PERIODS = ("test_a", "test_b")
DEFAULT_PAIRS = (
    ("level_res", "old_res"),
    ("level_res", "level_raw"),
    ("level_res", "level_only"),
    ("level_res", "f0"),
    ("level_res", "lora"),
    ("level_res", "direct_nlinear"),
)


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


def load_prediction(item: dict[str, Any], origins: np.ndarray) -> np.ndarray:
    path = check_receipt(item["prediction"])
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Prediction origins mismatch for {path}")
        return archive["prediction"].astype(np.float32)


def role_member_ids(evaluation: dict[str, Any], role: str) -> list[str]:
    ids = list(evaluation.get("roles", {}).get(role, []))
    return [identifier for identifier in ids if identifier in evaluation["models"]]


def role_prediction(evaluation: dict[str, Any], role: str, origins: np.ndarray) -> tuple[np.ndarray, list[str]]:
    ids = role_member_ids(evaluation, role)
    if not ids:
        raise KeyError(f"Missing role in evaluation: {role}")
    predictions = [load_prediction(evaluation["models"][identifier], origins) for identifier in ids]
    return np.mean(predictions, axis=0).astype(np.float32), ids


def score_components(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask axes mismatch")
    err = pred.astype(np.float64) - target.astype(np.float64)
    masked = np.where(mask, err, 0.0)
    return {
        "sq_sum_ch": (masked * masked).sum(axis=(0, 1)),
        "abs_sum_ch": np.abs(masked).sum(axis=(0, 1)),
        "signed_sum_ch": masked.sum(axis=(0, 1)),
        "count_ch": mask.sum(axis=(0, 1)),
        "sq_by_origin_ch": (masked * masked).sum(axis=1),
        "abs_by_origin_ch": np.abs(masked).sum(axis=1),
        "signed_by_origin_ch": masked.sum(axis=1),
        "count_by_origin_ch": mask.sum(axis=1),
    }


def channel_summary(predictions: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comps = {role: score_components(pred, target, mask) for role, pred in predictions.items()}
    for role, comp in comps.items():
        for c, name in enumerate(columns):
            count = int(comp["count_ch"][c])
            rows.append({
                "role": role,
                "channel_index": c,
                "channel": name,
                "target_count": count,
                "mse": float(comp["sq_sum_ch"][c] / count) if count else None,
                "mae": float(comp["abs_sum_ch"][c] / count) if count else None,
                "mean_signed_error": float(comp["signed_sum_ch"][c] / count) if count else None,
            })
    for left, right in DEFAULT_PAIRS:
        if left not in comps or right not in comps:
            continue
        for c, name in enumerate(columns):
            count = int(comps[left]["count_ch"][c])
            if count == 0 or int(comps[right]["count_ch"][c]) != count:
                delta = relpct = None
            else:
                left_mse = float(comps[left]["sq_sum_ch"][c] / count)
                ref_mse = float(comps[right]["sq_sum_ch"][c] / count)
                delta = left_mse - ref_mse
                relpct = 100 * delta / ref_mse if ref_mse > 0 else None
            rows.append({
                "role": f"{left}_minus_{right}",
                "channel_index": c,
                "channel": name,
                "target_count": count,
                "mse": delta,
                "mae": None,
                "mean_signed_error": None,
                "relative_mse_pct": relpct,
            })
    return rows


def origin_summary(predictions: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray, periods: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comps = {role: score_components(pred, target, mask) for role, pred in predictions.items()}
    for period, indices in periods.items():
        for local_pos, idx in enumerate(indices):
            for role, comp in comps.items():
                count = comp["count_by_origin_ch"][idx].sum()
                rows.append({
                    "period": period,
                    "origin_index_in_period": int(local_pos),
                    "global_origin_row": int(idx),
                    "role": role,
                    "target_count": int(count),
                    "mse": float(comp["sq_by_origin_ch"][idx].sum() / count) if count else None,
                    "mae": float(comp["abs_by_origin_ch"][idx].sum() / count) if count else None,
                    "mean_signed_error": float(comp["signed_by_origin_ch"][idx].sum() / count) if count else None,
                })
            for left, right in DEFAULT_PAIRS:
                if left not in comps or right not in comps:
                    continue
                count = comps[left]["count_by_origin_ch"][idx].sum()
                if count == 0 or comps[right]["count_by_origin_ch"][idx].sum() != count:
                    continue
                left_mse = float(comps[left]["sq_by_origin_ch"][idx].sum() / count)
                ref_mse = float(comps[right]["sq_by_origin_ch"][idx].sum() / count)
                rows.append({
                    "period": period,
                    "origin_index_in_period": int(local_pos),
                    "global_origin_row": int(idx),
                    "role": f"{left}_minus_{right}",
                    "target_count": int(count),
                    "mse": left_mse - ref_mse,
                    "mae": None,
                    "mean_signed_error": None,
                    "relative_mse_pct": 100 * (left_mse - ref_mse) / ref_mse if ref_mse > 0 else None,
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
    else:
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"path": rel(path), "sha256": digest(path), "rows": len(rows)}


def projection_diagnostics(predictions: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray, basis: np.ndarray, origins: np.ndarray, periods: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = mask.all(axis=2)
    if basis.shape[0] != target.shape[2]:
        raise ValueError("PCA basis/channel shape mismatch")
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    rows: list[dict[str, Any]] = []
    period_by_index = {int(idx): period for period, indices in periods.items() for idx in indices}
    complete_indices = np.argwhere(complete)
    for n, h in complete_indices:
        y = target[n, h].astype(np.float64)
        for role, pred in predictions.items():
            e = pred[n, h].astype(np.float64) - y
            retained = e @ projector
            residual = e - retained
            rows.append({
                "period": period_by_index[int(n)],
                "origin": int(origins[n]),
                "origin_position": int(n),
                "horizon": int(h),
                "role": role,
                "retained_space_mse": float(np.mean(retained * retained)),
                "residual_space_mse": float(np.mean(residual * residual)),
                "total_complete_row_mse": float(np.mean(e * e)),
            })
    coverage = {
        "complete_origin_horizon_rows": int(len(complete_indices)),
        "total_origin_horizon_rows": int(mask.shape[0] * mask.shape[1]),
        "coverage_fraction": float(len(complete_indices) / (mask.shape[0] * mask.shape[1])),
        "space_note": "Computed only on origin-horizon rows with every target channel observed; no missing target is zero-filled.",
    }
    return rows, coverage


def residual_context_rows(data: dict[str, Any], predictions: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray, origins: np.ndarray, periods: dict[str, np.ndarray], columns: list[str]) -> list[dict[str, Any]]:
    if "level_res" not in predictions:
        return []
    basis = np.asarray(data["basis"], dtype=np.float64)
    projector = basis @ basis.T
    x = np.asarray(data["x"], dtype=np.float64)
    level_error = predictions["level_res"].astype(np.float64) - target.astype(np.float64)
    period_by_index = {int(idx): period for period, indices in periods.items() for idx in indices}
    rows: list[dict[str, Any]] = []
    for n, origin in enumerate(origins):
        history = x[origin - CONTEXT:origin]
        residual = history - history @ projector
        recent = residual[-24:]
        delta = residual[-1] - residual[-24]
        for c, name in enumerate(columns):
            valid = mask[n, :, c]
            rows.append({
                "period": period_by_index[int(n)],
                "origin": int(origin),
                "origin_position": int(n),
                "channel_index": c,
                "channel": name,
                "last_reconstruction_residual": float(residual[-1, c]),
                "recent24_residual_mean": float(np.mean(recent[:, c])),
                "recent24_residual_std": float(np.std(recent[:, c])),
                "recent24_residual_delta": float(delta[c]),
                "future_observed_count": int(valid.sum()),
                "level_future_mean_signed_error": float(level_error[n, valid, c].mean()) if valid.any() else None,
                "level_future_mse": float(np.mean(level_error[n, valid, c] ** 2)) if valid.any() else None,
            })
    return rows


def run(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    evaluation = read_json(args.evaluation)
    if evaluation.get("dataset") != "robin" or evaluation.get("status") != "complete":
        raise ValueError("Robin complete v7 evaluation is required")
    check_receipt(evaluation["seal"])
    data = load_data("robin", include_test=True)
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    origins = np.concatenate([origins_by_period[p] for p in PERIODS])
    periods = {"test_a": np.arange(len(origins_by_period["test_a"]), dtype=np.int64), "test_b": np.arange(len(origins_by_period["test_a"]), len(origins), dtype=np.int64)}
    target = np.stack([data["x"][o:o + HORIZON] for o in origins]).astype(np.float32)
    mask = np.stack([data["finite"][o:o + HORIZON] for o in origins]).astype(bool)
    columns = np.asarray(data["columns"]).astype(str).tolist()
    roles_to_load = sorted({role for pair in DEFAULT_PAIRS for role in pair} | {"compress"})
    predictions: dict[str, np.ndarray] = {}
    members: dict[str, list[str]] = {}
    for role in roles_to_load:
        ids = role_member_ids(evaluation, role)
        if not ids:
            continue
        arrays = [load_prediction(evaluation["models"][identifier], origins) for identifier in ids]
        predictions[role] = np.mean(arrays, axis=0).astype(np.float32)
        members[role] = ids
        job.heartbeat(f"loaded {role}")
    if "level_res" not in predictions:
        raise ValueError("level_res role is required for Robin diagnostics")
    output_dir = HERE / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_csv = write_csv(output_dir / "robin_channel_error_diagnostics.csv", channel_summary(predictions, target, mask, columns))
    job.heartbeat("channel diagnostics written")
    origin_csv = write_csv(output_dir / "robin_origin_error_diagnostics.csv", origin_summary(predictions, target, mask, periods))
    job.heartbeat("origin diagnostics written")
    space_rows, coverage = projection_diagnostics(predictions, target, mask, np.asarray(data["basis"], dtype=np.float32), origins, periods)
    space_csv = write_csv(output_dir / "robin_complete_row_space_diagnostics.csv", space_rows)
    job.heartbeat("space diagnostics written")
    context_csv = write_csv(output_dir / "robin_residual_context_diagnostics.csv", residual_context_rows(data, predictions, target, mask, origins, periods, columns))
    result = {
        "schema": "tsfm_peft_practical_controls_v7_robin_diagnostics_v1",
        "status": "complete",
        "created_utc": time.time(),
        "evaluation": receipt(args.evaluation),
        "source": source_receipt(),
        "environment": environment_receipt(),
        "roles_loaded": members,
        "origins": {period: origins_by_period[period].astype(int).tolist() for period in PERIODS},
        "files": {
            "channel_error_diagnostics": channel_csv,
            "origin_error_diagnostics": origin_csv,
            "complete_row_space_diagnostics": space_csv,
            "residual_context_diagnostics": context_csv,
        },
        "coverage": coverage,
        "leakage_scope": "Diagnostics use stored predictions and exposed Robin TEST targets after fixed v7 evaluation. They are not used to retune LEVEL/K/G/loss/channel selection.",
        "missing_policy": "Projection diagnostics use only complete observed target rows; target missing values are never zero-filled.",
    }
    result_path = HERE / "robin_diagnostics_v7.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    save_json(result_path, result)
    return {"result": rel(result_path), "files": result["files"], "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose exposed Robin v7 prediction errors using stored predictions only.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reserve-seconds", type=float, default=300)
    args = parser.parse_args()
    with Job("v7_diagnose_robin", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": "robin"}) as job:
        result = run(args, job)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
