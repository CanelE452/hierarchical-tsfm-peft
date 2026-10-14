"""Phase v2 report adapter for simple-alternative diagnostics.

This adapter does not overwrite phase_v1.  It verifies the hashed phase_v1
tables, reuses them for summary/channel/horizon MSE/MAE comparisons, and reads
the already exposed Appliances cached arrays only to add RAW-error-fixed MAE
tail/body contribution diagnostics.  It performs no model inference, CUDA work,
TEST reads, or old campaign restart.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOLLOWUP_CACHE = ROOT / ".cache" / "smallk_failure_followup_20261014"
CACHE_ROOT = FOLLOWUP_CACHE / "simple_alternatives"
PHASE_V1 = CACHE_ROOT / "phase_v1"
DEFAULT_OUTPUT = CACHE_ROOT / "phase_v2"
FALSE_TORCH_GUARD_FAILURE = CACHE_ROOT / "phase_v1_failed_torch_guard_20261014T000000"
TAIL_QUANTILES = ("0.9", "0.95", "0.99")
APPLIANCES_SPLITS = ("fit", "tune", "eval")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(jsonable(rows))


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def direction(gap: float | None) -> str:
    if gap is None:
        return "missing"
    if gap < 0:
        return "gain"
    if gap > 0:
        return "loss"
    return "same"


def resolve_v1_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    lower_parts = {part.lower() for part in path.parts}
    if "test" in lower_parts or path.name.lower() in {"test.npz", "jena_test.npz"}:
        raise PermissionError(f"PROTECTED_TEST_PATH_DENIED:{path}")
    return path


def count_csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def verify_v1_inputs() -> dict[str, Any]:
    result_path = PHASE_V1 / "RESULT.json"
    provenance_path = PHASE_V1 / "PROVENANCE.json"
    result = read_json(result_path)
    provenance = read_json(provenance_path)
    if result.get("status") != "COMPLETE":
        raise ValueError("PHASE_V1_NOT_COMPLETE")
    if result.get("resource_terminal_status") != "COMPLETE":
        raise ValueError("PHASE_V1_RESOURCE_NOT_COMPLETE")
    app_prov = provenance.get("appliances_oldalignment", {})
    if app_prov.get("status") != "FOUND_AND_SCORED":
        raise ValueError("PHASE_V1_APPLIANCES_NOT_FOUND_AND_SCORED")
    table_records = {}
    for name, record in result.get("tables", {}).items():
        path = resolve_v1_path(record["path"])
        actual_sha = sha256_path(path)
        actual_rows = count_csv_data_rows(path)
        if actual_sha != record["sha256"]:
            raise ValueError(f"PHASE_V1_TABLE_SHA_MISMATCH:{name}:{actual_sha}!={record['sha256']}")
        if actual_rows != int(record["rows"]):
            raise ValueError(f"PHASE_V1_TABLE_ROW_MISMATCH:{name}:{actual_rows}!={record['rows']}")
        table_records[name] = {
            "path": rel(path),
            "sha256": actual_sha,
            "rows": actual_rows,
        }
    for record_key in ("report", "provenance", "runtime_receipt"):
        record = result.get(record_key)
        if record:
            path = resolve_v1_path(record["path"])
            actual_sha = sha256_path(path)
            if actual_sha != record["sha256"]:
                raise ValueError(f"PHASE_V1_{record_key.upper()}_SHA_MISMATCH:{actual_sha}!={record['sha256']}")
    return {
        "phase_v1_result": {"path": rel(result_path), "sha256": sha256_path(result_path)},
        "phase_v1_provenance": {"path": rel(provenance_path), "sha256": sha256_path(provenance_path)},
        "phase_v1_status": result.get("status"),
        "phase_v1_resource_terminal_status": result.get("resource_terminal_status"),
        "phase_v1_tables": table_records,
        "appliances_status": app_prov.get("status"),
        "appliances_assets_found": len(app_prov.get("located_assets", [])),
        "false_torch_guard_failure_preserved": FALSE_TORCH_GUARD_FAILURE.exists(),
        "false_torch_guard_failure_path": rel(FALSE_TORCH_GUARD_FAILURE) if FALSE_TORCH_GUARD_FAILURE.exists() else None,
    }


def table_path(v1_manifest: dict[str, Any], name: str) -> Path:
    return resolve_v1_path(v1_manifest["phase_v1_tables"][name]["path"])


def rows_by(rows: list[dict[str, str]], **criteria: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == str(value) for key, value in criteria.items())]


def first_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = rows_by(rows, **criteria)
    if len(matches) != 1:
        raise ValueError(f"ROW_NOT_UNIQUE:{criteria}:{len(matches)}")
    return matches[0]


def metric_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = [f(row[metric]) for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"MISSING_METRIC:{metric}")
    return float(np.mean(values, dtype=np.float64))


def comparison_row(
    *,
    family: str,
    dataset: str,
    split: str,
    entity_type: str,
    entity_id: str | int | None,
    comparison: str,
    candidate_label: str,
    baseline_label: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    seed_count: int | None = None,
) -> dict[str, Any]:
    cand_mse = f(candidate["normalized_mse"])
    base_mse = f(baseline["normalized_mse"])
    cand_mae = f(candidate["normalized_mae"])
    base_mae = f(baseline["normalized_mae"])
    cand_bias = f(candidate["normalized_signed_bias"])
    base_bias = f(baseline["normalized_signed_bias"])
    mse_gap = None if cand_mse is None or base_mse is None else cand_mse - base_mse
    mae_gap = None if cand_mae is None or base_mae is None else cand_mae - base_mae
    return {
        "family": family,
        "dataset": dataset,
        "split": split,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "comparison": comparison,
        "candidate": candidate_label,
        "baseline": baseline_label,
        "seed_count": seed_count,
        "observed_count": candidate.get("observed_count"),
        "active_channels": candidate.get("active_channels"),
        "candidate_mse": cand_mse,
        "baseline_mse": base_mse,
        "mse_gap": mse_gap,
        "mse_relative": None if mse_gap is None or base_mse == 0 else mse_gap / base_mse,
        "mse_direction": direction(mse_gap),
        "candidate_mae": cand_mae,
        "baseline_mae": base_mae,
        "mae_gap": mae_gap,
        "mae_relative": None if mae_gap is None or base_mae == 0 else mae_gap / base_mae,
        "mae_direction": direction(mae_gap),
        "candidate_signed_bias": cand_bias,
        "baseline_signed_bias": base_bias,
        "signed_bias_shift": None if cand_bias is None or base_bias is None else cand_bias - base_bias,
    }


def learned_mean_record(rows: list[dict[str, str]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "normalized_mse": metric_mean(rows, "normalized_mse"),
        "normalized_mae": metric_mean(rows, "normalized_mae"),
        "normalized_signed_bias": metric_mean(rows, "normalized_signed_bias"),
        "observed_count": rows[0].get("observed_count"),
        "active_channels": rows[0].get("active_channels"),
    }
    if extra:
        out.update(extra)
    return out


def build_appliances_comparisons(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for split in APPLIANCES_SPLITS:
        raw = first_row(summary_rows, family="appliances_oldalignment", dataset="appliances", split=split, method="RAW")
        raw_r = first_row(summary_rows, family="appliances_oldalignment", dataset="appliances", split=split, method="RAW+R")
        learned = rows_by(summary_rows, family="appliances_oldalignment", dataset="appliances", split=split, method="LEARNED")
        learned_mean = learned_mean_record(learned, {"observed_count": raw["observed_count"], "active_channels": raw["active_channels"]})
        out.append(
            comparison_row(
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                entity_type="split_macro",
                entity_id="all_channels",
                comparison="RAW+R_vs_RAW",
                candidate_label="RAW+R",
                baseline_label="RAW",
                candidate=raw_r,
                baseline=raw,
                seed_count=1,
            )
        )
        out.append(
            comparison_row(
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                entity_type="split_macro",
                entity_id="all_channels",
                comparison="LEARNED_SEED_MEAN_vs_RAW",
                candidate_label="LEARNED_SEED_MEAN",
                baseline_label="RAW",
                candidate=learned_mean,
                baseline=raw,
                seed_count=len(learned),
            )
        )
        out.append(
            comparison_row(
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                entity_type="split_macro",
                entity_id="all_channels",
                comparison="LEARNED_SEED_MEAN_vs_RAW+R",
                candidate_label="LEARNED_SEED_MEAN",
                baseline_label="RAW+R",
                candidate=learned_mean,
                baseline=raw_r,
                seed_count=len(learned),
            )
        )
    return out


def build_entity_gain_loss(rows: list[dict[str, str]], *, entity_field: str, entity_type: str) -> list[dict[str, Any]]:
    out = []
    entity_values = sorted(
        {int(row[entity_field]) for row in rows if row["family"] == "appliances_oldalignment" and row["dataset"] == "appliances"}
    )
    for split in APPLIANCES_SPLITS:
        for entity_value in entity_values:
            raw = first_row(
                rows,
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                method="RAW",
                **{entity_field: entity_value},
            )
            raw_r = first_row(
                rows,
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                method="RAW+R",
                **{entity_field: entity_value},
            )
            learned = rows_by(
                rows,
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                method="LEARNED",
                **{entity_field: entity_value},
            )
            learned_mean = learned_mean_record(
                learned,
                {
                    "observed_count": raw.get("observed_count"),
                    "active_channels": raw.get("active_channels"),
                },
            )
            out.append(
                comparison_row(
                    family="appliances_oldalignment",
                    dataset="appliances",
                    split=split,
                    entity_type=entity_type,
                    entity_id=entity_value,
                    comparison="RAW+R_vs_RAW",
                    candidate_label="RAW+R",
                    baseline_label="RAW",
                    candidate=raw_r,
                    baseline=raw,
                    seed_count=1,
                )
            )
            out.append(
                comparison_row(
                    family="appliances_oldalignment",
                    dataset="appliances",
                    split=split,
                    entity_type=entity_type,
                    entity_id=entity_value,
                    comparison="LEARNED_SEED_MEAN_vs_RAW",
                    candidate_label="LEARNED_SEED_MEAN",
                    baseline_label="RAW",
                    candidate=learned_mean,
                    baseline=raw,
                    seed_count=len(learned),
                )
            )
            out.append(
                comparison_row(
                    family="appliances_oldalignment",
                    dataset="appliances",
                    split=split,
                    entity_type=entity_type,
                    entity_id=entity_value,
                    comparison="LEARNED_SEED_MEAN_vs_RAW+R",
                    candidate_label="LEARNED_SEED_MEAN",
                    baseline_label="RAW+R",
                    candidate=learned_mean,
                    baseline=raw_r,
                    seed_count=len(learned),
                )
            )
    return out


def build_bridge_baseline_focus(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    retained = {"RAW", "LAST", "SEASONAL_LAG_24", "SEASONAL_LAG_48"}
    for row in summary_rows:
        if row["family"] != "bridge_simple_alternatives" or row["method"] not in retained:
            continue
        out.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "method": row["method"],
                "seed": row["seed"],
                "normalized_mse": f(row["normalized_mse"]),
                "normalized_mae": f(row["normalized_mae"]),
                "normalized_signed_bias": f(row["normalized_signed_bias"]),
                "mse_gap_vs_raw": f(row.get("mse_gap_vs_raw")),
                "mae_gap_vs_raw": f(row.get("mae_gap_vs_raw")),
                "observed_count": row["observed_count"],
                "active_channels": row["active_channels"],
            }
        )
    return out


def fixed_raw_tail_mae_detail() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import research.smallk_failure_followup_20261014.run_simple_alternatives as phase_v1_source

    details = []
    source_reads = []
    specs = phase_v1_source.appliances_method_specs()
    raw_spec = next(spec for spec in specs if spec["method"] == "RAW")
    for split in APPLIANCES_SPLITS:
        raw_arrays, raw_source = phase_v1_source.load_appliance_source(raw_spec, split)
        source_reads.append({"split": split, "method": "RAW", "seed": "shared_reference", **raw_source})
        raw_error = raw_arrays[f"{split}_prediction"].astype(np.float64) - raw_arrays[f"{split}_target"].astype(np.float64)
        observed = raw_arrays[f"{split}_observed"].astype(bool)
        counts = observed.sum(axis=(0, 1))
        active_channels = counts > 0
        active_count = int(active_channels.sum())
        thresholds: dict[str, list[float | None]] = {}
        for quantile in TAIL_QUANTILES:
            q = float(quantile)
            thresholds[quantile] = [
                float(np.quantile(np.abs(raw_error[:, :, channel][observed[:, :, channel]]), q))
                if observed[:, :, channel].any()
                else None
                for channel in range(raw_error.shape[-1])
            ]
        for spec in specs:
            if spec["method"] == "RAW":
                continue
            arrays, source = phase_v1_source.load_appliance_source(spec, split)
            source_reads.append({"split": split, "method": spec["method"], "seed": spec["seed"], **source})
            phase_v1_source.validate_appliance_conditions(raw_arrays, arrays, split, f"{spec['method']}:{spec['seed']}")
            method_error = arrays[f"{split}_prediction"].astype(np.float64) - arrays[f"{split}_target"].astype(np.float64)
            for quantile, per_channel_thresholds in thresholds.items():
                tail = np.zeros_like(observed)
                for channel, threshold in enumerate(per_channel_thresholds):
                    if threshold is not None:
                        tail[:, :, channel] = np.abs(raw_error[:, :, channel]) >= threshold
                for region, selected in (("tail", tail), ("body", ~tail)):
                    method_num = np.where(selected & observed, np.abs(method_error), 0.0).sum(axis=(0, 1))
                    raw_num = np.where(selected & observed, np.abs(raw_error), 0.0).sum(axis=(0, 1))
                    for channel in range(raw_error.shape[-1]):
                        denominator = int(counts[channel])
                        scale = denominator * active_count if denominator else 0
                        method_contrib = float(method_num[channel] / scale) if scale else None
                        raw_contrib = float(raw_num[channel] / scale) if scale else None
                        details.append(
                            {
                                "family": "appliances_oldalignment",
                                "dataset": "appliances",
                                "split": split,
                                "method": spec["method"],
                                "seed": spec["seed"],
                                "channel_index": channel,
                                "quantile": quantile,
                                "region": region,
                                "raw_abs_error_cutoff": per_channel_thresholds[channel],
                                "selected_observed_count": int((selected[:, :, channel] & observed[:, :, channel]).sum()),
                                "normalized_mae_contribution": method_contrib,
                                "raw_normalized_mae_contribution": raw_contrib,
                                "gap_contribution_vs_raw": None if method_contrib is None else method_contrib - raw_contrib,
                                "raw_error_fixed_quantile": True,
                                "channelmacro_denominator_uses_full_split": True,
                            }
                        )
    return details, source_reads


def aggregate_tail_detail(rows: list[dict[str, Any]], *, metric: str) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    if metric == "mse":
        candidate_key = "normalized_mse_contribution"
        raw_key = "raw_normalized_mse_contribution"
    elif metric == "mae":
        candidate_key = "normalized_mae_contribution"
        raw_key = "raw_normalized_mae_contribution"
    else:
        raise ValueError(metric)
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["split"], str(row["quantile"]), row["region"], row["method"], str(row["seed"]))
        group = groups.setdefault(
            key,
            {
                "split": row["split"],
                "quantile": str(row["quantile"]),
                "region": row["region"],
                "method": row["method"],
                "seed": str(row["seed"]),
                "candidate_contribution": 0.0,
                "raw_contribution": 0.0,
                "selected_observed_count_sum": 0,
                "channel_count": 0,
                "raw_error_fixed_quantile": True,
                "channelmacro_denominator_uses_full_split": True,
            },
        )
        candidate = f(row.get(candidate_key))
        raw = f(row.get(raw_key))
        if candidate is not None:
            group["candidate_contribution"] += candidate
        if raw is not None:
            group["raw_contribution"] += raw
        group["selected_observed_count_sum"] += int(row["selected_observed_count"])
        group["channel_count"] += 1
        group["raw_error_fixed_quantile"] = group["raw_error_fixed_quantile"] and str(row["raw_error_fixed_quantile"]) == "True"
        group["channelmacro_denominator_uses_full_split"] = (
            group["channelmacro_denominator_uses_full_split"] and str(row["channelmacro_denominator_uses_full_split"]) == "True"
        )
    return groups


def tail_comparison_rows(mse_detail: list[dict[str, Any]], mae_detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    aggregates = {
        "mse": aggregate_tail_detail(mse_detail, metric="mse"),
        "mae": aggregate_tail_detail(mae_detail, metric="mae"),
    }
    for metric, groups in aggregates.items():
        for split in APPLIANCES_SPLITS:
            for quantile in TAIL_QUANTILES:
                for region in ("body", "tail"):
                    raw_r = groups[(split, quantile, region, "RAW+R", "shared_reference")]
                    learned = [groups[(split, quantile, region, "LEARNED", str(seed))] for seed in (92601, 92602)]
                    learned_candidate = float(np.mean([row["candidate_contribution"] for row in learned], dtype=np.float64))
                    learned_raw = float(np.mean([row["raw_contribution"] for row in learned], dtype=np.float64))
                    records = [
                        ("RAW+R_vs_RAW", "RAW+R", "RAW", raw_r["candidate_contribution"], raw_r["raw_contribution"], 1),
                        ("LEARNED_SEED_MEAN_vs_RAW", "LEARNED_SEED_MEAN", "RAW", learned_candidate, learned_raw, len(learned)),
                        (
                            "LEARNED_SEED_MEAN_vs_RAW+R",
                            "LEARNED_SEED_MEAN",
                            "RAW+R",
                            learned_candidate,
                            raw_r["candidate_contribution"],
                            len(learned),
                        ),
                    ]
                    for comparison, candidate_label, baseline_label, candidate_value, baseline_value, seed_count in records:
                        gap = candidate_value - baseline_value
                        out.append(
                            {
                                "family": "appliances_oldalignment",
                                "dataset": "appliances",
                                "split": split,
                                "metric": metric,
                                "quantile": quantile,
                                "region": region,
                                "comparison": comparison,
                                "candidate": candidate_label,
                                "baseline": baseline_label,
                                "seed_count": seed_count,
                                "candidate_contribution": candidate_value,
                                "baseline_contribution": baseline_value,
                                "gap_contribution": gap,
                                "relative_contribution": None if baseline_value == 0 else gap / baseline_value,
                                "direction": direction(gap),
                                "raw_error_fixed_quantile": True,
                                "channelmacro_denominator_uses_full_split": True,
                                "selected_observed_count_sum": raw_r["selected_observed_count_sum"],
                                "channel_count": raw_r["channel_count"],
                            }
                        )
    return out


def gain_loss_summary(rows: list[dict[str, Any]], *, comparison: str, split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["comparison"] == comparison and row["split"] == split]
    counts = {
        "mse_gain": sum(1 for row in selected if row["mse_direction"] == "gain"),
        "mse_loss": sum(1 for row in selected if row["mse_direction"] == "loss"),
        "mae_gain": sum(1 for row in selected if row["mae_direction"] == "gain"),
        "mae_loss": sum(1 for row in selected if row["mae_direction"] == "loss"),
        "total": len(selected),
    }
    top_mae_losses = sorted(selected, key=lambda row: row["mae_gap"], reverse=True)[:5]
    top_mse_gains = sorted(selected, key=lambda row: row["mse_gap"])[:5]
    return {
        "comparison": comparison,
        "split": split,
        "counts": counts,
        "top_mae_losses": [
            {"entity_id": row["entity_id"], "mae_gap": row["mae_gap"], "mse_gap": row["mse_gap"]} for row in top_mae_losses
        ],
        "top_mse_gains": [
            {"entity_id": row["entity_id"], "mse_gap": row["mse_gap"], "mae_gap": row["mae_gap"]} for row in top_mse_gains
        ],
    }


def source_snapshot(output: Path) -> dict[str, Any]:
    folder = output / "sources"
    folder.mkdir()
    copied = []
    for source in (Path(__file__).resolve(), HERE / "run_simple_alternatives.py", HERE / "followup_runtime.py"):
        target = folder / source.name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append(
            {
                "path": rel(source),
                "sha256": sha256_path(source),
                "snapshot_path": rel(target),
                "snapshot_sha256": sha256_path(target),
            }
        )
    manifest = {"files": copied}
    write_json(folder / "MANIFEST.json", manifest)
    return {"path": rel(folder / "MANIFEST.json"), "sha256": sha256_path(folder / "MANIFEST.json"), "files": copied}


def format_pct(value: float) -> str:
    return f"{100.0 * value:+.6f}%"


def report_lines(result: dict[str, Any]) -> list[str]:
    comparisons = result["key_findings"]["appliances_split_comparisons"]
    eval_raw = next(row for row in comparisons if row["split"] == "eval" and row["comparison"] == "LEARNED_SEED_MEAN_vs_RAW")
    eval_raw_r = next(row for row in comparisons if row["split"] == "eval" and row["comparison"] == "LEARNED_SEED_MEAN_vs_RAW+R")
    raw_lines = [
        row for row in comparisons if row["comparison"] in {"LEARNED_SEED_MEAN_vs_RAW", "LEARNED_SEED_MEAN_vs_RAW+R"}
    ]
    tail_eval = [
        row
        for row in result["key_findings"]["tail_eval_focus"]
        if row["comparison"] in {"LEARNED_SEED_MEAN_vs_RAW", "LEARNED_SEED_MEAN_vs_RAW+R"}
    ]
    bridge_focus = result["key_findings"]["bridge_focus"]
    lines = [
        "# Simple alternatives v2 CPU 진단",
        "",
        "[확인] v2는 phase_v1의 해시 고정 CSV를 검증한 뒤 재집계했고, Appliances MAE tail/body만 이미 노출된 캐시 배열에서 CPU로 다시 계산했다.",
        "",
        f"- 최종 상태: `{result['status']}`",
        f"- resource terminal: `{result.get('resource_terminal_status')}`",
        f"- body wall seconds: `{result.get('body_wall_seconds')}`",
        "- 수행 범위: 새 모델 추론 없음, CUDA/GPU 실행 없음, TEST read 없음, old campaign restart 없음.",
        "- torch 상태: 진단 코드는 torch/CUDA/model을 직접 사용하지 않았다. `FollowupLimit`가 frozen Bi runtime을 로드하면서 torch가 module import로 들어올 수 있다.",
        f"- v1 false torch-guard failure 보존: `{result['provenance_details']['v1_inputs']['false_torch_guard_failure_path']}`",
        "",
        "## 핵심 결론",
        "",
        (
            f"- Appliances eval에서 LEARNED seed 평균은 원래 RAW 대비 MSE {format_pct(eval_raw['mse_relative'])}, "
            f"MAE {format_pct(eval_raw['mae_relative'])}다. 즉 요청한 `MSE 감소 + MAE 증가` 충돌은 RAW 기준에서 실제로 재현된다."
        ),
        (
            f"- 같은 LEARNED seed 평균을 RAW+R과 비교하면 MSE {format_pct(eval_raw_r['mse_relative'])}, "
            f"MAE {format_pct(eval_raw_r['mae_relative'])}로 둘 다 나빠진다. RAW+R은 더 강한 저장 대안이다."
        ),
        "- 이 결과는 저장 배열의 수치 분해이며, 왜 그런지에 대한 causal mechanism이나 새 holdout 성능을 주장하지 않는다.",
        "",
        "## Appliances split macro 비교",
        "",
        "```text",
        "split  comparison                    mse_rel       mae_rel       mse_gap        mae_gap",
    ]
    for row in raw_lines:
        lines.append(
            f"{row['split']:<5}  {row['comparison']:<29}  {row['mse_relative']:+.9f}  {row['mae_relative']:+.9f}  "
            f"{row['mse_gap']:+.12f}  {row['mae_gap']:+.12f}"
        )
    lines += [
        "```",
        "",
        "## RAW-error-fixed tail/body contribution",
        "",
        "아래 값은 q=.9/.95/.99 RAW absolute-error 영역을 고정하고, full-split channel denominator를 유지한 contribution gap이다.",
        "",
        "```text",
        "split metric q    region comparison                    gap_contribution  rel_contribution",
    ]
    for row in tail_eval:
        lines.append(
            f"{row['split']:<5} {row['metric']:<6} {row['quantile']:<4} {row['region']:<5} {row['comparison']:<29} "
            f"{row['gap_contribution']:+.12f} {row['relative_contribution']:+.9f}"
        )
    channel_eval_raw = result["key_findings"]["channel_gain_loss_eval_vs_raw"]
    horizon_eval_raw = result["key_findings"]["horizon_gain_loss_eval_vs_raw"]
    lines += [
        "```",
        "",
        "## Channel / horizon gain-loss 요약",
        "",
        f"- eval channel LEARNED vs RAW: {channel_eval_raw['counts']}",
        f"- eval horizon LEARNED vs RAW: {horizon_eval_raw['counts']}",
        "- 전체 channel/horizon별 gain/loss 행은 `appliances_channel_gain_loss.csv`, `appliances_horizon_gain_loss.csv`에 저장했다.",
        "",
        "```text",
        "top eval channel MAE losses vs RAW:",
    ]
    for row in channel_eval_raw["top_mae_losses"]:
        lines.append(f"channel {row['entity_id']}: mae_gap={row['mae_gap']:+.12f}, mse_gap={row['mse_gap']:+.12f}")
    lines.append("")
    lines.append("top eval horizon MAE losses vs RAW:")
    for row in horizon_eval_raw["top_mae_losses"]:
        lines.append(f"horizon {row['entity_id']}: mae_gap={row['mae_gap']:+.12f}, mse_gap={row['mse_gap']:+.12f}")
    lines += [
        "```",
        "",
        "## LAST / seasonal coarse metrics",
        "",
        "```text",
        "dataset             split method           mse             mae             mse_gap_vs_raw   mae_gap_vs_raw",
    ]
    for row in bridge_focus:
        if not (
            (row["dataset"] == "robin" and row["split"] == "val")
            or (row["dataset"] == "beijing_aotizhongxin" and row["split"] == "eval")
        ):
            continue
        lines.append(
            f"{row['dataset']:<19} {row['split']:<5} {row['method']:<16} {row['normalized_mse']:.12f} "
            f"{row['normalized_mae']:.12f} {row['mse_gap_vs_raw'] if row['mse_gap_vs_raw'] is not None else '':>16} "
            f"{row['mae_gap_vs_raw'] if row['mae_gap_vs_raw'] is not None else '':>16}"
        )
    lines += [
        "```",
        "",
        "## 검증 메모",
        "",
        "- phase_v1 `RESULT.json`의 table SHA와 row count를 모두 다시 확인했다.",
        "- phase_v1 보고서의 resource terminal `None` 표기는 렌더 순서 문제였고, phase_v1 `RESULT.json`의 실제 resource terminal은 `COMPLETE`였다.",
        "- v2 보고서는 `FollowupLimit.close()` 후의 실제 terminal receipt를 반영한다.",
        "- Appliances 자산은 모두 발견·검증된 상태였다.",
    ]
    return lines


def write_report(path: Path, result: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines(result)) + "\n")


def execute(output: Path) -> dict[str, Any]:
    from research.smallk_failure_followup_20261014 import followup_runtime as fr

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "RUNNING",
        "schema": "smallk_simple_alternatives_phase_v2_report",
        "gpu_used": False,
        "new_model_inference": False,
        "old_campaign_restart": False,
        "protected_TEST_access": False,
        "output": rel(output),
    }
    limit = None
    body_started = None
    try:
        try:
            limit = fr.FollowupLimit("simple_alternatives_phase_v2_report", False, followup_roots=(HERE, FOLLOWUP_CACHE, output))
        except Exception as error:
            result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)})
            result["resource_terminal"] = fr.LAST_CONSTRUCTOR_TERMINAL or {
                "terminal_status": "FAILED_PRESERVED",
                "terminal_error": result["failure"],
                "constructor_failure_scope": "FollowupLimit constructor failed before caller-owned instance assignment",
            }
            result["resource_terminal_status"] = result["resource_terminal"].get("terminal_status")
            raise
        body_started = time.perf_counter()
        v1_inputs = verify_v1_inputs()
        limit.check()
        summary_rows = read_csv_rows(table_path(v1_inputs, "summary"))
        channel_rows = read_csv_rows(table_path(v1_inputs, "channels"))
        horizon_rows = read_csv_rows(table_path(v1_inputs, "horizons"))
        tail_mse_rows = read_csv_rows(table_path(v1_inputs, "tail_contributions"))
        appliances_comparisons = build_appliances_comparisons(summary_rows)
        channel_gain_loss = build_entity_gain_loss(channel_rows, entity_field="channel_index", entity_type="channel")
        horizon_gain_loss = build_entity_gain_loss(horizon_rows, entity_field="horizon_1based", entity_type="horizon")
        bridge_focus = build_bridge_baseline_focus(summary_rows)
        limit.check()
        tail_mae_rows, tail_mae_sources = fixed_raw_tail_mae_detail()
        tail_regions = tail_comparison_rows(tail_mse_rows, tail_mae_rows)
        write_csv(output / "appliances_split_comparisons.csv", appliances_comparisons)
        write_csv(output / "appliances_channel_gain_loss.csv", channel_gain_loss)
        write_csv(output / "appliances_horizon_gain_loss.csv", horizon_gain_loss)
        write_csv(output / "bridge_baseline_focus.csv", bridge_focus)
        write_csv(output / "tail_mae_contributions_detail.csv", tail_mae_rows)
        write_csv(output / "tail_region_comparisons.csv", tail_regions)
        source_files = source_snapshot(output)
        eval_vs_raw = next(
            row for row in appliances_comparisons if row["split"] == "eval" and row["comparison"] == "LEARNED_SEED_MEAN_vs_RAW"
        )
        eval_vs_raw_r = next(
            row for row in appliances_comparisons if row["split"] == "eval" and row["comparison"] == "LEARNED_SEED_MEAN_vs_RAW+R"
        )
        self_checks = {
            "phase_v1_hashed_tables_verified": True,
            "appliances_assets_found_and_scored": v1_inputs["appliances_status"] == "FOUND_AND_SCORED",
            "eval_learned_vs_raw_mse_down_mae_up": eval_vs_raw["mse_gap"] < 0 and eval_vs_raw["mae_gap"] > 0,
            "eval_learned_vs_raw_plus_r_both_worse": eval_vs_raw_r["mse_gap"] > 0 and eval_vs_raw_r["mae_gap"] > 0,
            "tail_metrics_include_mse_and_mae": {row["metric"] for row in tail_regions} == {"mse", "mae"},
            "channel_and_horizon_full_rows_present": len(channel_gain_loss) > 0 and len(horizon_gain_loss) > 0,
        }
        if not all(self_checks.values()):
            raise ValueError(f"SELF_CHECK_FAILED:{self_checks}")
        provenance = {
            "v1_inputs": v1_inputs,
            "source_files": source_files,
            "tail_mae_array_sources": tail_mae_sources,
            "cost_scope": {
                "actual_compute": "CPU report aggregation over phase_v1 CSV plus MAE tail contribution over cached Appliances arrays",
                "raw_cache_cost_reuse_declared": True,
                "teacher_or_old_cache_cost_reuse_declared": True,
                "cheap_method_claim": False,
            },
            "domain": {
                "type": "time_series_forecasting",
                "decision_context": "diagnose numeric tradeoff between stored RAW, RAW+R, learned alignment, and simple persistence/seasonal baselines",
                "leakage_controls": [
                    "phase_v1 bridge split roles preserved",
                    "no TEST path reads",
                    "no refit of FIT statistics",
                    "fixed lag 24/48 retained without adaptive selection",
                    "RAW-error-fixed tail regions use RAW absolute-error thresholds",
                ],
            },
        }
        write_json(output / "PROVENANCE.json", provenance)
        result.update(
            body_wall_seconds=time.perf_counter() - body_started,
            tables={
                "appliances_split_comparisons": {
                    "path": rel(output / "appliances_split_comparisons.csv"),
                    "sha256": sha256_path(output / "appliances_split_comparisons.csv"),
                    "rows": len(appliances_comparisons),
                },
                "appliances_channel_gain_loss": {
                    "path": rel(output / "appliances_channel_gain_loss.csv"),
                    "sha256": sha256_path(output / "appliances_channel_gain_loss.csv"),
                    "rows": len(channel_gain_loss),
                },
                "appliances_horizon_gain_loss": {
                    "path": rel(output / "appliances_horizon_gain_loss.csv"),
                    "sha256": sha256_path(output / "appliances_horizon_gain_loss.csv"),
                    "rows": len(horizon_gain_loss),
                },
                "bridge_baseline_focus": {
                    "path": rel(output / "bridge_baseline_focus.csv"),
                    "sha256": sha256_path(output / "bridge_baseline_focus.csv"),
                    "rows": len(bridge_focus),
                },
                "tail_mae_contributions_detail": {
                    "path": rel(output / "tail_mae_contributions_detail.csv"),
                    "sha256": sha256_path(output / "tail_mae_contributions_detail.csv"),
                    "rows": len(tail_mae_rows),
                },
                "tail_region_comparisons": {
                    "path": rel(output / "tail_region_comparisons.csv"),
                    "sha256": sha256_path(output / "tail_region_comparisons.csv"),
                    "rows": len(tail_regions),
                },
            },
            provenance={"path": rel(output / "PROVENANCE.json"), "sha256": sha256_path(output / "PROVENANCE.json")},
            provenance_details=provenance,
            key_findings={
                "appliances_split_comparisons": appliances_comparisons,
                "tail_eval_focus": [row for row in tail_regions if row["split"] == "eval"],
                "channel_gain_loss_eval_vs_raw": gain_loss_summary(
                    channel_gain_loss, comparison="LEARNED_SEED_MEAN_vs_RAW", split="eval"
                ),
                "horizon_gain_loss_eval_vs_raw": gain_loss_summary(
                    horizon_gain_loss, comparison="LEARNED_SEED_MEAN_vs_RAW", split="eval"
                ),
                "bridge_focus": bridge_focus,
                "self_checks": self_checks,
            },
            torch_import_scope=(
                "No diagnostic code imports or uses torch/CUDA/model code. The required FollowupLimit imports the frozen Bi runtime, "
                "and that runtime imports torch at module import time."
            ),
        )
        result["status"] = "COMPLETE"
    except Exception as error:
        result["body_wall_seconds"] = None if body_started is None else time.perf_counter() - body_started
        result.update(
            status="FAILED_PRESERVED",
            failure={"type": type(error).__name__, "message": str(error), "traceback_tail": traceback.format_exc().splitlines()[-8:]},
        )
    finally:
        if limit is not None:
            receipt = limit.close(result["status"], None if result["status"] == "COMPLETE" else result.get("failure"))
            result["resource_terminal"] = receipt
            result["resource_terminal_status"] = receipt.get("terminal_status") or receipt.get("status")
            result["resource_terminal_error"] = receipt.get("terminal_error")
            if result["status"] == "COMPLETE" and result["resource_terminal_status"] != "COMPLETE":
                result["status"] = result["resource_terminal_status"]
                result["failure"] = {"type": "FollowupLimitTerminalStatus", "message": str(result["resource_terminal_error"])}
        result["torch_imported"] = "torch" in sys.modules
        result.setdefault(
            "torch_import_scope",
            "No diagnostic code imports or uses torch/CUDA/model code. If present, torch was imported by the required FollowupLimit/Bi runtime.",
        )
        write_json(output / "FOLLOWUP_RUNTIME_RECEIPT.json", result.get("resource_terminal"))
        result["runtime_receipt"] = {
            "path": rel(output / "FOLLOWUP_RUNTIME_RECEIPT.json"),
            "sha256": sha256_path(output / "FOLLOWUP_RUNTIME_RECEIPT.json"),
        }
        write_report(output / "REPORT_KO.md", result)
        result["report"] = {"path": rel(output / "REPORT_KO.md"), "sha256": sha256_path(output / "REPORT_KO.md")}
        report_text = (output / "REPORT_KO.md").read_text(encoding="utf-8")
        post_close_checks = {
            "report_states_actual_resource_complete": result.get("resource_terminal_status") == "COMPLETE"
            and "resource terminal: `COMPLETE`" in report_text,
            "report_answers_raw_conflict": "RAW 대비 MSE" in report_text and "MAE" in report_text,
            "report_does_not_claim_asset_missing": "asset missing" not in report_text.lower() and "자산 누락" not in report_text,
        }
        write_json(output / "SELF_TEST.json", post_close_checks)
        result["self_test"] = {"path": rel(output / "SELF_TEST.json"), "sha256": sha256_path(output / "SELF_TEST.json"), **post_close_checks}
        write_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = execute(args.output)
    print(json.dumps({"status": result["status"], "output": rel(args.output.resolve()), "terminal": result.get("resource_terminal_status")}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
