
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from runtime_v7 import CACHE, HERE, ROOT, Job, digest, environment_receipt, load_data, save_json, source_receipt

CONTEXT = 512
HORIZON = 48
PERIODS = ("test_a", "test_b")
SEEDS = (92601, 92602)
PAIRS = (
    ("level_res", "old_res"),
    ("level_res", "level_raw"),
    ("level_res", "level_only"),
    ("level_res", "f0"),
    ("level_res", "lora"),
    ("level_res", "direct_nlinear"),
)
FEATURES = ("last", "recent24_mean", "recent24_std", "recent24_change")
OUTCOMES = ("level_future_mean_signed_error", "level_future_mse")


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


def optional_receipt(path: str | Path) -> dict[str, Any]:
    resolved = resolve(path)
    return {"path": str(resolved), "exists": resolved.exists(), "sha256": digest(resolved) if resolved.exists() else None}


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


def load_prediction(item: dict[str, Any], origins: np.ndarray) -> np.ndarray:
    path = check_receipt(item["prediction"])
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Prediction origins mismatch for {path}")
        return archive["prediction"].astype(np.float32)


def role_member_ids(evaluation: dict[str, Any], role: str) -> list[str]:
    return [identifier for identifier in evaluation.get("roles", {}).get(role, []) if identifier in evaluation["models"]]


def load_role_predictions(evaluation: dict[str, Any], role: str, origins: np.ndarray) -> dict[str, np.ndarray]:
    ids = role_member_ids(evaluation, role)
    if not ids:
        raise KeyError(f"Missing role in evaluation: {role}")
    return {identifier: load_prediction(evaluation["models"][identifier], origins) for identifier in ids}


def components(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask axes mismatch")
    err = pred.astype(np.float64) - target.astype(np.float64)
    masked = np.where(mask, err, 0.0)
    count_ch = mask.sum(axis=(0, 1)).astype(np.float64)
    count_origin_ch = mask.sum(axis=1).astype(np.float64)
    if np.any(count_ch == 0):
        raise ValueError("Missing entire target channel; do not silently drop it")
    return {
        "sq_ch": (masked * masked).sum(axis=(0, 1)),
        "abs_ch": np.abs(masked).sum(axis=(0, 1)),
        "signed_ch": masked.sum(axis=(0, 1)),
        "count_ch": count_ch,
        "sq_origin_ch": (masked * masked).sum(axis=1),
        "abs_origin_ch": np.abs(masked).sum(axis=1),
        "signed_origin_ch": masked.sum(axis=1),
        "count_origin_ch": count_origin_ch,
        "err": masked,
    }


def per_member_components(predictions: dict[str, dict[str, np.ndarray]], target: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    return {role: {identifier: components(pred, target, mask) for identifier, pred in members.items()} for role, members in predictions.items()}


def aggregate_channel_metrics(comps: dict[str, dict[str, dict[str, np.ndarray]]], columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, members in comps.items():
        per_seed_values = []
        for identifier, comp in members.items():
            seed = comp_seed(identifier)
            count = comp["count_ch"]
            mse = comp["sq_ch"] / count
            mae = comp["abs_ch"] / count
            bias = comp["signed_ch"] / count
            per_seed_values.append((mse, mae, bias))
            for c, channel in enumerate(columns):
                rows.append({
                    "scope": "member",
                    "role": role,
                    "member_id": identifier,
                    "seed": seed,
                    "channel_index": c,
                    "channel": channel,
                    "target_count": int(count[c]),
                    "mse": float(mse[c]),
                    "mae": float(mae[c]),
                    "mean_signed_error": float(bias[c]),
                    "aggregation_contract": "per-member error first",
                })
        mean_mse = np.mean([x[0] for x in per_seed_values], axis=0)
        mean_mae = np.mean([x[1] for x in per_seed_values], axis=0)
        mean_bias = np.mean([x[2] for x in per_seed_values], axis=0)
        count = next(iter(members.values()))["count_ch"]
        for c, channel in enumerate(columns):
            rows.append({
                "scope": "role_mean_loss",
                "role": role,
                "member_id": "mean_of_member_losses",
                "seed": None,
                "channel_index": c,
                "channel": channel,
                "target_count": int(count[c]),
                "mse": float(mean_mse[c]),
                "mae": float(mean_mae[c]),
                "mean_signed_error": float(mean_bias[c]),
                "member_count": len(members),
                "aggregation_contract": "mean of per-member losses, deterministic singleton not ensembled",
            })
    return rows


def comp_seed(identifier: str) -> int | None:
    for seed in SEEDS:
        if str(seed) in identifier:
            return seed
    return None


def paired_members(left_members: dict[str, Any], right_members: dict[str, Any]) -> list[tuple[int, str, str]]:
    def by_seed(members: dict[str, Any]) -> dict[int | None, str]:
        return {comp_seed(identifier): identifier for identifier in members}
    left_seed = by_seed(left_members)
    right_seed = by_seed(right_members)
    if len(left_members) == 1 and None in left_seed:
        only = left_seed[None]
        return [(seed, only, right_seed[seed]) for seed in SEEDS if seed in right_seed]
    if len(right_members) == 1 and None in right_seed:
        only = right_seed[None]
        return [(seed, left_seed[seed], only) for seed in SEEDS if seed in left_seed]
    return [(seed, left_seed[seed], right_seed[seed]) for seed in SEEDS if seed in left_seed and seed in right_seed]


def pair_channel_metrics(comps: dict[str, dict[str, dict[str, np.ndarray]]], columns: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    for left, right in PAIRS:
        if left not in comps or right not in comps:
            continue
        pairs = paired_members(comps[left], comps[right])
        if not pairs:
            continue
        seed_delta_mse = []
        seed_delta_mae = []
        seed_delta_bias = []
        seed_left_mse = []
        seed_right_mse = []
        for seed, left_id, right_id in pairs:
            lc, rc = comps[left][left_id], comps[right][right_id]
            if not np.array_equal(lc["count_ch"], rc["count_ch"]):
                raise ValueError(f"Target counts differ for {left_id}/{right_id}")
            count = lc["count_ch"]
            lmse, rmse = lc["sq_ch"] / count, rc["sq_ch"] / count
            lmae, rmae = lc["abs_ch"] / count, rc["abs_ch"] / count
            lbias, rbias = lc["signed_ch"] / count, rc["signed_ch"] / count
            dmse, dmae, dbias = lmse - rmse, lmae - rmae, lbias - rbias
            seed_delta_mse.append(dmse)
            seed_delta_mae.append(dmae)
            seed_delta_bias.append(dbias)
            seed_left_mse.append(lmse)
            seed_right_mse.append(rmse)
            for c, channel in enumerate(columns):
                rows.append({
                    "scope": "paired_member",
                    "pair": f"{left}_minus_{right}",
                    "left_role": left,
                    "right_role": right,
                    "left_member_id": left_id,
                    "right_member_id": right_id,
                    "seed": seed,
                    "channel_index": c,
                    "channel": channel,
                    "target_count": int(count[c]),
                    "left_mse": float(lmse[c]),
                    "right_mse": float(rmse[c]),
                    "delta_mse": float(dmse[c]),
                    "relative_mse_pct": float(100 * dmse[c] / rmse[c]) if rmse[c] > 0 else None,
                    "delta_mae": float(dmae[c]),
                    "delta_mean_signed_error": float(dbias[c]),
                })
        mean_dmse = np.mean(seed_delta_mse, axis=0)
        mean_dmae = np.mean(seed_delta_mae, axis=0)
        mean_dbias = np.mean(seed_delta_bias, axis=0)
        mean_left = np.mean(seed_left_mse, axis=0)
        mean_right = np.mean(seed_right_mse, axis=0)
        count = next(iter(comps[left].values()))["count_ch"]
        for c, channel in enumerate(columns):
            rows.append({
                "scope": "pair_mean_seed_loss",
                "pair": f"{left}_minus_{right}",
                "left_role": left,
                "right_role": right,
                "seed": "mean_of_paired_member_losses",
                "channel_index": c,
                "channel": channel,
                "target_count": int(count[c]),
                "left_mse": float(mean_left[c]),
                "right_mse": float(mean_right[c]),
                "delta_mse": float(mean_dmse[c]),
                "relative_mse_pct": float(100 * mean_dmse[c] / mean_right[c]) if mean_right[c] > 0 else None,
                "delta_mae": float(mean_dmae[c]),
                "delta_mean_signed_error": float(mean_dbias[c]),
                "positive_excess_mse": float(max(mean_dmse[c], 0.0)),
                "negative_excess_mse": float(min(mean_dmse[c], 0.0)),
            })
        concentration_rows.extend(concentration_summary(f"{left}_minus_{right}", "channel", mean_dmse, columns))
    return rows, concentration_rows


def per_origin_score(comp: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = comp["count_origin_ch"].sum(axis=1)
    sq = comp["sq_origin_ch"].sum(axis=1)
    abs_ = comp["abs_origin_ch"].sum(axis=1)
    signed = comp["signed_origin_ch"].sum(axis=1)
    return sq / count, abs_ / count, signed / count


def origin_metrics(comps: dict[str, dict[str, dict[str, np.ndarray]]], origins: np.ndarray, period_indices: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    index_to_period = {int(i): period for period, indices in period_indices.items() for i in indices}
    for role, members in comps.items():
        member_values = []
        for identifier, comp in members.items():
            mse, mae, bias = per_origin_score(comp)
            member_values.append((mse, mae, bias))
            for i, origin in enumerate(origins):
                rows.append({"scope": "member", "role": role, "member_id": identifier, "seed": comp_seed(identifier), "period": index_to_period[i], "origin_position": i, "origin": int(origin), "mse": float(mse[i]), "mae": float(mae[i]), "mean_signed_error": float(bias[i])})
        mean_mse = np.mean([v[0] for v in member_values], axis=0)
        mean_mae = np.mean([v[1] for v in member_values], axis=0)
        mean_bias = np.mean([v[2] for v in member_values], axis=0)
        for i, origin in enumerate(origins):
            rows.append({"scope": "role_mean_loss", "role": role, "member_id": "mean_of_member_losses", "seed": None, "period": index_to_period[i], "origin_position": i, "origin": int(origin), "mse": float(mean_mse[i]), "mae": float(mean_mae[i]), "mean_signed_error": float(mean_bias[i]), "member_count": len(members)})
    for left, right in PAIRS:
        if left not in comps or right not in comps:
            continue
        pairs = paired_members(comps[left], comps[right])
        seed_dmse = []
        seed_dmae = []
        seed_dbias = []
        for seed, left_id, right_id in pairs:
            lmse, lmae, lbias = per_origin_score(comps[left][left_id])
            rmse, rmae, rbias = per_origin_score(comps[right][right_id])
            dmse, dmae, dbias = lmse - rmse, lmae - rmae, lbias - rbias
            seed_dmse.append(dmse)
            seed_dmae.append(dmae)
            seed_dbias.append(dbias)
            for i, origin in enumerate(origins):
                rows.append({"scope": "paired_member_delta", "pair": f"{left}_minus_{right}", "left_role": left, "right_role": right, "seed": seed, "period": index_to_period[i], "origin_position": i, "origin": int(origin), "delta_mse": float(dmse[i]), "delta_mae": float(dmae[i]), "delta_mean_signed_error": float(dbias[i])})
        if seed_dmse:
            mean_dmse = np.mean(seed_dmse, axis=0)
            mean_dmae = np.mean(seed_dmae, axis=0)
            mean_dbias = np.mean(seed_dbias, axis=0)
            for i, origin in enumerate(origins):
                rows.append({"scope": "pair_mean_seed_loss_delta", "pair": f"{left}_minus_{right}", "left_role": left, "right_role": right, "seed": "mean_of_paired_member_losses", "period": index_to_period[i], "origin_position": i, "origin": int(origin), "delta_mse": float(mean_dmse[i]), "delta_mae": float(mean_dmae[i]), "delta_mean_signed_error": float(mean_dbias[i]), "positive_excess_mse": float(max(mean_dmse[i], 0.0)), "negative_excess_mse": float(min(mean_dmse[i], 0.0))})
            labels = [f"{index_to_period[i]}:{int(origins[i])}" for i in range(len(origins))]
            concentration_rows.extend(concentration_summary(f"{left}_minus_{right}", "origin", mean_dmse, labels))
    return rows, concentration_rows


def concentration_summary(pair: str, axis: str, delta: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    pos = np.maximum(delta, 0.0)
    neg = np.minimum(delta, 0.0)
    pos_total = float(pos.sum())
    neg_total = float(neg.sum())
    rows = [{"pair": pair, "axis": axis, "summary": "totals", "positive_excess_mse_sum": pos_total, "negative_excess_mse_sum": neg_total, "net_delta_mse_sum": float(delta.sum()), "item_count": len(delta)}]
    if pos_total > 0:
        order = np.argsort(-pos)
        for k in (1, 3, 5, 10):
            take = order[: min(k, len(order))]
            rows.append({"pair": pair, "axis": axis, "summary": f"top{k}_positive_share", "positive_excess_mse_sum": float(pos[take].sum()), "positive_excess_share": float(pos[take].sum() / pos_total), "items": ";".join(labels[i] for i in take)})
    if neg_total < 0:
        order = np.argsort(neg)
        for k in (1, 3, 5, 10):
            take = order[: min(k, len(order))]
            rows.append({"pair": pair, "axis": axis, "summary": f"top{k}_negative_improvement_share", "negative_excess_mse_sum": float(neg[take].sum()), "negative_excess_share_of_abs": float(abs(neg[take].sum()) / abs(neg_total)), "items": ";".join(labels[i] for i in take)})
    return rows


def summarize_complete_row_space(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, Any], dict[str, Any]] = {}
    for row in rows:
        key = (row["period"], row["role"], row["member_id"], row["seed"])
        item = groups.setdefault(
            key,
            {
                "period": row["period"],
                "role": row["role"],
                "member_id": row["member_id"],
                "seed": row["seed"],
                "complete_origin_horizon_rows": 0,
                "retained_space_mse_sum": 0.0,
                "residual_space_mse_sum": 0.0,
                "total_complete_row_mse_sum": 0.0,
            },
        )
        item["complete_origin_horizon_rows"] += 1
        item["retained_space_mse_sum"] += float(row["retained_space_mse"])
        item["residual_space_mse_sum"] += float(row["residual_space_mse"])
        item["total_complete_row_mse_sum"] += float(row["total_complete_row_mse"])
    output: list[dict[str, Any]] = []
    for item in sorted(groups.values(), key=lambda x: (x["period"], x["role"], str(x["seed"]), x["member_id"])):
        count = item["complete_origin_horizon_rows"]
        output.append({
            "period": item["period"],
            "role": item["role"],
            "member_id": item["member_id"],
            "seed": item["seed"],
            "complete_origin_horizon_rows": count,
            "p_retained_space_mse": float(item["retained_space_mse_sum"] / count),
            "q_residual_space_mse": float(item["residual_space_mse_sum"] / count),
            "total_complete_row_mse": float(item["total_complete_row_mse_sum"] / count),
            "space_contract": "P/Q summary is averaged from per-member complete rows; detailed rows are stored only in the local cache.",
        })
    return output


def complete_row_space(comps: dict[str, dict[str, dict[str, np.ndarray]]], predictions: dict[str, dict[str, np.ndarray]], target: np.ndarray, mask: np.ndarray, basis: np.ndarray, origins: np.ndarray, period_indices: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    complete = mask.all(axis=2)
    positions = np.argwhere(complete)
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    index_to_period = {int(i): period for period, indices in period_indices.items() for i in indices}
    rows: list[dict[str, Any]] = []
    coverage_by_period = {}
    for period, indices in period_indices.items():
        sub = complete[indices]
        coverage_by_period[period] = {"complete_rows": int(sub.sum()), "total_rows": int(sub.size), "coverage_fraction": float(sub.sum() / sub.size)}
    for n, h in positions:
        y = target[n, h].astype(np.float64)
        for role, members in predictions.items():
            vals = []
            for identifier, pred in members.items():
                e = pred[n, h].astype(np.float64) - y
                retained = e @ projector
                residual = e - retained
                vals.append((identifier, comp_seed(identifier), float(np.mean(retained * retained)), float(np.mean(residual * residual)), float(np.mean(e * e))))
            for identifier, seed, rin, rout, total in vals:
                rows.append({"scope": "member", "period": index_to_period[int(n)], "origin_position": int(n), "origin": int(origins[n]), "horizon": int(h), "role": role, "member_id": identifier, "seed": seed, "retained_space_mse": rin, "residual_space_mse": rout, "total_complete_row_mse": total})
            mean_vals = np.mean(np.array([[v[2], v[3], v[4]] for v in vals], dtype=np.float64), axis=0)
            rows.append({"scope": "role_mean_loss", "period": index_to_period[int(n)], "origin_position": int(n), "origin": int(origins[n]), "horizon": int(h), "role": role, "member_id": "mean_of_member_losses", "seed": None, "retained_space_mse": float(mean_vals[0]), "residual_space_mse": float(mean_vals[1]), "total_complete_row_mse": float(mean_vals[2])})
    coverage = {"complete_origin_horizon_rows": int(len(positions)), "total_origin_horizon_rows": int(mask.shape[0] * mask.shape[1]), "coverage_fraction": float(len(positions) / (mask.shape[0] * mask.shape[1])), "by_period": coverage_by_period, "same_complete_rows_all_models": True, "note": "Complete rows are defined once from target mask and reused for every model; no target zero-fill."}
    return rows, summarize_complete_row_space(rows), coverage


def residual_context_table(data: dict[str, Any], comps: dict[str, dict[str, dict[str, np.ndarray]]], target: np.ndarray, mask: np.ndarray, origins: np.ndarray, period_indices: dict[str, np.ndarray], columns: list[str]) -> list[dict[str, Any]]:
    if "level_res" not in comps:
        return []
    basis = np.asarray(data["basis"], dtype=np.float64)
    projector = basis @ basis.T
    x = np.asarray(data["x"], dtype=np.float64)
    level_members = comps["level_res"]
    index_to_period = {int(i): period for period, indices in period_indices.items() for i in indices}
    rows: list[dict[str, Any]] = []
    for n, origin in enumerate(origins):
        residual = x[origin - CONTEXT:origin] - x[origin - CONTEXT:origin] @ projector
        recent = residual[-24:]
        feature_arrays = {
            "last": residual[-1],
            "recent24_mean": recent.mean(axis=0),
            "recent24_std": recent.std(axis=0),
            "recent24_change": residual[-1] - residual[-24],
        }
        for c, channel in enumerate(columns):
            valid = mask[n, :, c]
            if not valid.any():
                continue
            signed_vals = []
            mse_vals = []
            for comp in level_members.values():
                err = comp["err"][n, valid, c]
                signed_vals.append(float(err.mean()))
                mse_vals.append(float(np.mean(err * err)))
            row = {"period": index_to_period[int(n)], "origin_position": int(n), "origin": int(origin), "channel_index": c, "channel": channel, "future_observed_count": int(valid.sum()), "level_future_mean_signed_error": float(np.mean(signed_vals)), "level_future_mse": float(np.mean(mse_vals)), "member_count": len(level_members)}
            row.update({name: float(values[c]) for name, values in feature_arrays.items()})
            rows.append(row)
    return rows


def correlation_rows(context_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in context_rows:
        for outcome in OUTCOMES:
            groups.setdefault((row["period"], int(row["channel_index"]), outcome), []).append(row)
    output: list[dict[str, Any]] = []
    for (period, c, outcome), rows in groups.items():
        channel = rows[0]["channel"]
        y = np.array([float(r[outcome]) for r in rows], dtype=np.float64)
        for feature in FEATURES:
            x = np.array([float(r[feature]) for r in rows], dtype=np.float64)
            output.append({"period": period, "channel_index": c, "channel": channel, "feature": feature, "outcome": outcome, "n": int(len(rows)), "pearson": pearson(x, y), "spearman": pearson(rankdata(x), rankdata(y)), "interpretation_scope": "descriptive exposed-TEST diagnostic; no causal or routing claim"})
    return output


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
    # Average tied ranks.
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


def write_correction_marker(path: Path, evaluation: Path) -> dict[str, Any]:
    marker = {
        "schema": "tsfm_peft_practical_controls_v7_diagnostic_correction_v1",
        "status": "original_diagnostic_invalid_for_mean_seed_conclusions",
        "created_utc": time.time(),
        "original_script": optional_receipt(HERE / "diagnose_robin_v7.py"),
        "original_result": optional_receipt(HERE / "robin_diagnostics_v7.json"),
        "original_diagnostics_dir": {"path": rel(HERE / "diagnostics"), "exists": (HERE / "diagnostics").exists()},
        "corrected_script": receipt(Path(__file__)),
        "evaluation": receipt(evaluation),
        "does_not_invalidate": "robin_eval01 score/evaluation files remain valid; only the original diagnostic aggregation was wrong.",
        "root_cause": "The original diagnostic averaged seed/member predictions before computing squared/absolute errors, producing ensemble-style diagnostics under labels that implied mean seed loss.",
        "fix_summary": [
            "compute squared/absolute/signed errors per member first",
            "aggregate role and pair diagnostics as mean of per-member or paired-member losses",
            "broadcast deterministic singleton references for paired seed deltas",
            "preserve original script/results and write corrected outputs under diagnostics_corrected/",
            "store detailed complete-row P/Q diagnostics under local cache and publish only a compact period/role/member summary",
            "add positive/negative/net excess concentration and descriptive residual-context correlations",
        ],
    }
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    save_json(path, marker)
    return marker


def run(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    evaluation = read_json(args.evaluation)
    if evaluation.get("dataset") != "robin" or evaluation.get("status") != "complete":
        raise ValueError("Complete Robin v7 evaluation is required")
    check_receipt(evaluation["seal"])
    data = load_data("robin", include_test=True)
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    origins = np.concatenate([origins_by_period[p] for p in PERIODS])
    period_indices = {"test_a": np.arange(len(origins_by_period["test_a"]), dtype=np.int64), "test_b": np.arange(len(origins_by_period["test_a"]), len(origins), dtype=np.int64)}
    target = np.stack([data["x"][o:o + HORIZON] for o in origins]).astype(np.float32)
    mask = np.stack([data["finite"][o:o + HORIZON] for o in origins]).astype(bool)
    columns = np.asarray(data["columns"]).astype(str).tolist()
    roles = sorted({role for pair in PAIRS for role in pair} | {"compress"})
    predictions = {role: load_role_predictions(evaluation, role, origins) for role in roles if role_member_ids(evaluation, role)}
    if "level_res" not in predictions:
        raise ValueError("level_res role is required")
    comps = per_member_components(predictions, target, mask)
    out_dir = HERE / "diagnostics_corrected"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corrected diagnostics directory: {out_dir}")
    out_dir.mkdir(parents=True)
    files: dict[str, Any] = {}
    files["role_channel_metrics"] = write_csv(out_dir / "robin_role_channel_metrics_corrected.csv", aggregate_channel_metrics(comps, columns))
    pair_ch, conc_ch = pair_channel_metrics(comps, columns)
    files["pair_channel_excess"] = write_csv(out_dir / "robin_pair_channel_excess_corrected.csv", pair_ch)
    files["channel_excess_concentration"] = write_csv(out_dir / "robin_channel_excess_concentration_corrected.csv", conc_ch)
    job.heartbeat("channel metrics corrected")
    origin_rows, conc_origin = origin_metrics(comps, origins, period_indices)
    files["origin_metrics"] = write_csv(out_dir / "robin_origin_metrics_corrected.csv", origin_rows)
    files["origin_excess_concentration"] = write_csv(out_dir / "robin_origin_excess_concentration_corrected.csv", conc_origin)
    job.heartbeat("origin metrics corrected")
    cache_out_dir = CACHE / "diagnostics_corrected"
    if cache_out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corrected cache diagnostics directory: {cache_out_dir}")
    cache_out_dir.mkdir(parents=True)
    space_rows, space_summary_rows, coverage = complete_row_space(comps, predictions, target, mask, np.asarray(data["basis"], dtype=np.float32), origins, period_indices)
    cache_space_path = cache_out_dir / "robin_complete_row_space_detail_corrected.csv"
    cache_space_meta = write_csv(cache_space_path, space_rows)
    files["complete_row_space_detail_cache"] = {**receipt(cache_space_path), "relative_path": cache_space_meta["path"], "rows": cache_space_meta["rows"]}
    files["complete_row_space_summary"] = write_csv(out_dir / "robin_complete_row_space_summary_corrected.csv", space_summary_rows)
    context_rows = residual_context_table(data, comps, target, mask, origins, period_indices, columns)
    files["residual_context_rows"] = write_csv(out_dir / "robin_residual_context_rows_corrected.csv", context_rows)
    files["residual_context_correlations"] = write_csv(out_dir / "robin_residual_context_correlations_corrected.csv", correlation_rows(context_rows))
    job.heartbeat("space and residual context corrected")
    correction_marker = write_correction_marker(HERE / "diagnostic_correction.json", args.evaluation)
    result = {
        "schema": "tsfm_peft_practical_controls_v7_robin_corrected_diagnostics_v1",
        "status": "complete",
        "created_utc": time.time(),
        "evaluation": receipt(args.evaluation),
        "source": source_receipt(),
        "environment": environment_receipt(),
        "correction_marker": receipt(HERE / "diagnostic_correction.json"),
        "original_invalid_scope": correction_marker["status"],
        "roles_loaded": {role: list(members) for role, members in predictions.items()},
        "origins": {period: origins_by_period[period].astype(int).tolist() for period in PERIODS},
        "files": files,
        "coverage": coverage,
        "aggregation_contract": "Every diagnostic score is computed per member/seed first, then averaged. Deterministic references are broadcast only for pair deltas. No prediction ensemble is used.",
        "leakage_scope": "Uses exposed Robin TEST targets after fixed v7 evaluation for diagnosis only; does not retune LEVEL/K/G/loss/channel selection.",
        "missing_policy": "Complete-row projection uses a single target complete-row mask for all models; target missing values are never zero-filled.",
        "correlation_scope": "Residual-context correlations are descriptive exposed-TEST diagnostics and make no causal or future routing claim.",
    }
    result_path = HERE / "robin_diagnostics_corrected_v7.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    save_json(result_path, result)
    return {"result": rel(result_path), "files": files, "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected Robin diagnostics using mean seed loss, not prediction ensembles.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reserve-seconds", type=float, default=300)
    args = parser.parse_args()
    with Job("v7_diagnose_robin_corrected", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": "robin", "correction": "mean_seed_loss_diagnostics"}) as job:
        result = run(args, job)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
