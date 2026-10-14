"""CPU-only simple-alternative diagnostics over already exposed arrays.

This script reads only audited bridge inputs, cached prediction NPZs, and their
JSON provenance.  It does not import torch, run GPU code, open TEST assets, or
restart any old campaign.
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
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_ROOT = ROOT / ".cache" / "smallk_failure_followup_20261014" / "simple_alternatives"
DEFAULT_OUTPUT = CACHE_ROOT / "phase_v1"
BRIDGE_RESULT = ROOT / ".cache" / "smallk_failure_followup_20261014" / "input_bridge" / "old_conditions_actual_v1" / "RESULT.json"
FOLLOWUP_CACHE = ROOT / ".cache" / "smallk_failure_followup_20261014"
TOL = 0.0
TAIL_QUANTILES = (0.9, 0.95, 0.99)
SEASONAL_LAGS = (24, 48)
DATASETS = {
    "robin": {"splits": ("fit", "tune", "val"), "main_prefix": "main_", "channels": 17},
    "beijing_aotizhongxin": {"splits": ("fit", "tune", "eval"), "main_prefix": "main_", "channels": 11},
}
APPLIANCES_DIR = ROOT / "research" / "smallk_predictive_alignment_20261013" / "run_new_v1" / "appliances"
APPLIANCES_JSONS = {
    ("RAW", "dev", None): APPLIANCES_DIR / "RAW_CONTROLS.json",
    ("RAW", "eval", None): APPLIANCES_DIR / "EVAL_RAW_CONTROLS.json",
    ("LEARNED", "dev", 92601): APPLIANCES_DIR / "S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92601_CONTROLS.json",
    ("LEARNED", "dev", 92602): APPLIANCES_DIR / "S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92602_CONTROLS.json",
    ("LEARNED", "eval", 92601): APPLIANCES_DIR / "EVAL_S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92601_CONTROLS.json",
    ("LEARNED", "eval", 92602): APPLIANCES_DIR / "EVAL_S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92602_CONTROLS.json",
}


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    lower_parts = {part.lower() for part in path.parts}
    if "test" in lower_parts or path.name.lower() in {"test.npz", "jena_test.npz"}:
        raise PermissionError(f"PROTECTED_TEST_PATH_DENIED:{path}")
    return path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_npz(record: dict, *, arrays_contract: dict | None = None, max_uncompressed: int = 512 * 1024**2):
    path = resolve_path(record["path"])
    before = sha256_path(path)
    expected = record.get("sha256")
    if expected and before != expected:
        raise ValueError(f"SOURCE_SHA_MISMATCH:{path}:{before}!={expected}")
    with zipfile.ZipFile(path) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
    if uncompressed > max_uncompressed:
        raise ValueError(f"NPZ_UNCOMPRESSED_LIMIT:{path}:{uncompressed}")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    after = sha256_path(path)
    if after != before:
        raise ValueError(f"SOURCE_CHANGED_WHILE_READING:{path}")
    if arrays_contract is not None:
        if set(arrays) != set(arrays_contract):
            raise ValueError(f"ARRAY_KEYS_MISMATCH:{path}")
        for key, meta in arrays_contract.items():
            value = arrays[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or array_sha256(value) != meta["sha256"]:
                raise ValueError(f"ARRAY_CONTRACT_MISMATCH:{path}:{key}")
    return arrays, {
        "path": rel(path),
        "sha256": before,
        "bytes": path.stat().st_size,
        "uncompressed_bytes": uncompressed,
    }


def exact_array(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"EXACT_SHAPE_DTYPE_MISMATCH:{label}")
    if not np.array_equal(left, right, equal_nan=True):
        raise ValueError(f"EXACT_CONTENT_MISMATCH:{label}")


def channel_score(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    rows = []
    for channel in range(error.shape[-1]):
        mask = observed[:, :, channel]
        values = error[:, :, channel][mask]
        if len(values):
            rows.append(
                {
                    "channel_index": channel,
                    "observed_count": int(mask.sum()),
                    "normalized_mse": float(np.mean(values**2)),
                    "normalized_mae": float(np.mean(np.abs(values))),
                    "normalized_signed_bias": float(np.mean(values)),
                }
            )
        else:
            rows.append(
                {
                    "channel_index": channel,
                    "observed_count": 0,
                    "normalized_mse": None,
                    "normalized_mae": None,
                    "normalized_signed_bias": None,
                }
            )
    active = [row for row in rows if row["observed_count"]]
    return {
        "normalized_mse": float(np.mean([row["normalized_mse"] for row in active])),
        "normalized_mae": float(np.mean([row["normalized_mae"] for row in active])),
        "normalized_signed_bias": float(np.mean([row["normalized_signed_bias"] for row in active])),
        "active_channels": len(active),
        "observed_count": int(sum(row["observed_count"] for row in active)),
        "channels": rows,
        "error": error,
    }


def horizon_scores(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> list[dict]:
    rows = []
    for horizon in range(prediction.shape[1]):
        sub = channel_score(prediction[:, horizon : horizon + 1], target[:, horizon : horizon + 1], observed[:, horizon : horizon + 1])
        rows.append(
            {
                "horizon_1based": horizon + 1,
                "normalized_mse": sub["normalized_mse"],
                "normalized_mae": sub["normalized_mae"],
                "normalized_signed_bias": sub["normalized_signed_bias"],
                "active_channels": sub["active_channels"],
                "observed_count": sub["observed_count"],
            }
        )
    return rows


def fixed_raw_tail_rows(raw_error: np.ndarray, method_error: np.ndarray, observed: np.ndarray, *, dataset: str, split: str, method: str, seed) -> list[dict]:
    rows = []
    counts = observed.sum(axis=(0, 1))
    active_channels = counts > 0
    for quantile in TAIL_QUANTILES:
        thresholds = []
        for channel in range(raw_error.shape[-1]):
            mask = observed[:, :, channel]
            thresholds.append(float(np.quantile(np.abs(raw_error[:, :, channel][mask]), quantile)) if mask.any() else None)
        tail = np.zeros_like(observed)
        for channel, threshold in enumerate(thresholds):
            if threshold is not None:
                tail[:, :, channel] = np.abs(raw_error[:, :, channel]) >= threshold
        for region, selected in (("tail", tail), ("body", ~tail)):
            method_num = np.where(selected & observed, method_error**2, 0.0).sum(axis=(0, 1))
            raw_num = np.where(selected & observed, raw_error**2, 0.0).sum(axis=(0, 1))
            for channel in range(raw_error.shape[-1]):
                denominator = int(counts[channel])
                scale = denominator * int(active_channels.sum()) if denominator else 0
                method_contrib = float(method_num[channel] / scale) if scale else None
                raw_contrib = float(raw_num[channel] / scale) if scale else None
                rows.append(
                    {
                        "family": "appliances_oldalignment",
                        "dataset": dataset,
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "channel_index": channel,
                        "quantile": quantile,
                        "region": region,
                        "raw_abs_error_cutoff": thresholds[channel],
                        "selected_observed_count": int((selected[:, :, channel] & observed[:, :, channel]).sum()),
                        "normalized_mse_contribution": method_contrib,
                        "raw_normalized_mse_contribution": raw_contrib,
                        "gap_contribution_vs_raw": None if method_contrib is None else method_contrib - raw_contrib,
                        "raw_error_fixed_quantile": True,
                        "channelmacro_denominator_uses_full_split": True,
                    }
                )
    return rows


def method_rows(*, family: str, dataset: str, split: str, method: str, seed, prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, raw_score: dict | None = None):
    score = channel_score(prediction, target, observed)
    summary = {
        "family": family,
        "dataset": dataset,
        "split": split,
        "method": method,
        "seed": seed,
        "normalized_mse": score["normalized_mse"],
        "normalized_mae": score["normalized_mae"],
        "normalized_signed_bias": score["normalized_signed_bias"],
        "active_channels": score["active_channels"],
        "observed_count": score["observed_count"],
    }
    if raw_score is not None:
        summary["raw_normalized_mse"] = raw_score["normalized_mse"]
        summary["raw_normalized_mae"] = raw_score["normalized_mae"]
        summary["mse_gap_vs_raw"] = score["normalized_mse"] - raw_score["normalized_mse"]
        summary["mae_gap_vs_raw"] = score["normalized_mae"] - raw_score["normalized_mae"]
    channels = []
    raw_channels = {row["channel_index"]: row for row in raw_score["channels"]} if raw_score else {}
    for row in score["channels"]:
        out = {
            "family": family,
            "dataset": dataset,
            "split": split,
            "method": method,
            "seed": seed,
            **{key: value for key, value in row.items() if key != "channel_index"},
            "channel_index": row["channel_index"],
        }
        if raw_score:
            raw_row = raw_channels[row["channel_index"]]
            out["mse_gap_vs_raw"] = None if row["normalized_mse"] is None else row["normalized_mse"] - raw_row["normalized_mse"]
            out["mae_gap_vs_raw"] = None if row["normalized_mae"] is None else row["normalized_mae"] - raw_row["normalized_mae"]
            out["signed_bias_shift_vs_raw"] = None if row["normalized_signed_bias"] is None else row["normalized_signed_bias"] - raw_row["normalized_signed_bias"]
        channels.append(out)
    horizons = []
    for row in horizon_scores(prediction, target, observed):
        out = {
            "family": family,
            "dataset": dataset,
            "split": split,
            "method": method,
            "seed": seed,
            **row,
        }
        horizons.append(out)
    return summary, channels, horizons, score


def load_bridge():
    bridge = read_json(BRIDGE_RESULT)
    if bridge.get("status") != "COMPLETE":
        raise ValueError("BRIDGE_NOT_COMPLETE")
    return bridge


def bridge_records(bridge: dict, dataset: str) -> dict[str, dict]:
    return {record["split"]: record for record in bridge["artifacts"] if record["dataset"] == dataset}


def load_bridge_split(record: dict):
    return checked_npz(record, arrays_contract=record["arrays"])


def load_candidate_prediction(record: dict, splits: tuple[str, ...], expected):
    arrays, source = checked_npz(record)
    for key in ("mean", "std"):
        exact_array(arrays[key], expected[key], f"candidate:{record['path']}:{key}")
    for split in splits:
        for field in ("target", "observed", "origins"):
            exact_array(arrays[f"{split}_{field}"], expected[split][field], f"candidate:{record['path']}:{split}:{field}")
    return {split: arrays[f"{split}_prediction"] for split in splits}, source


def seasonal_prediction(history: np.ndarray, lag: int, horizon: int) -> np.ndarray:
    if lag > history.shape[1]:
        raise ValueError(f"LAG_EXCEEDS_HISTORY:{lag}>{history.shape[1]}")
    pieces = [history[:, -lag + (step % lag), :] for step in range(horizon)]
    return np.stack(pieces, axis=1).astype(np.float32, copy=False)


def run_bridge_baselines(bridge: dict):
    summary_rows, channel_rows, horizon_rows = [], [], []
    provenance = {"bridge_result": {"path": rel(BRIDGE_RESULT), "sha256": sha256_path(BRIDGE_RESULT)}, "datasets": {}}
    key_findings = {}
    for dataset, spec in DATASETS.items():
        splits = spec["splits"]
        records = bridge_records(bridge, dataset)
        split_arrays = {}
        sources = []
        for split in splits:
            arrays, source = load_bridge_split(records[split])
            split_arrays[split] = arrays
            sources.append(source)
        expected = {"mean": split_arrays[splits[0]]["mean"], "std": split_arrays[splits[0]]["std"]}
        for split in splits:
            expected[split] = {
                "target": split_arrays[split]["target"],
                "observed": split_arrays[split]["observed"],
                "origins": split_arrays[split]["origins"],
            }
            exact_array(split_arrays[split]["history"][:, -1], split_arrays[split]["raw_last"], f"{dataset}:{split}:raw_last_history_last")
        candidates = {"RAW": {"seed": "shared_reference", "predictions": {split: split_arrays[split]["raw_forecast"] for split in splits}, "source": "bridge_raw_forecast"}}
        candidate_sources = []
        for seed in (92601, 92602):
            record = bridge["declared_asset_hashes"]["predictions"][dataset][f"main_{seed}"]
            predictions, source = load_candidate_prediction(record, splits, expected)
            candidates[f"MAIN_{seed}"] = {"seed": seed, "predictions": predictions, "source": source}
            candidate_sources.append(source)
        for split in splits:
            target = split_arrays[split]["target"]
            observed = split_arrays[split]["observed"]
            raw_summary, raw_channels, raw_horizons, raw_score = method_rows(
                family="bridge_simple_alternatives",
                dataset=dataset,
                split=split,
                method="RAW",
                seed="shared_reference",
                prediction=candidates["RAW"]["predictions"][split],
                target=target,
                observed=observed,
            )
            summary_rows.append(raw_summary)
            channel_rows.extend(raw_channels)
            horizon_rows.extend(raw_horizons)
            baseline_predictions = {
                "LAST": split_arrays[split]["raw_last"][:, None, :].repeat(target.shape[1], axis=1).astype(np.float32, copy=False)
            }
            for lag in SEASONAL_LAGS:
                if lag <= split_arrays[split]["history"].shape[1]:
                    baseline_predictions[f"SEASONAL_LAG_{lag}"] = seasonal_prediction(split_arrays[split]["history"], lag, target.shape[1])
            for method, prediction in baseline_predictions.items():
                row, channels, horizons, _score = method_rows(
                    family="bridge_simple_alternatives",
                    dataset=dataset,
                    split=split,
                    method=method,
                    seed="deterministic",
                    prediction=prediction,
                    target=target,
                    observed=observed,
                    raw_score=raw_score,
                )
                summary_rows.append(row)
                channel_rows.extend(channels)
                horizon_rows.extend(horizons)
            for name, item in candidates.items():
                if name == "RAW":
                    continue
                row, channels, horizons, _score = method_rows(
                    family="bridge_simple_alternatives",
                    dataset=dataset,
                    split=split,
                    method="MAIN",
                    seed=item["seed"],
                    prediction=item["predictions"][split],
                    target=target,
                    observed=observed,
                    raw_score=raw_score,
                )
                summary_rows.append(row)
                channel_rows.extend(channels)
                horizon_rows.extend(horizons)
        provenance["datasets"][dataset] = {
            "input_artifacts": sources,
            "main_prediction_sources": candidate_sources,
            "history_length": {split: int(split_arrays[split]["history"].shape[1]) for split in splits},
            "seasonal_lags_predeclared_retained": [lag for lag in SEASONAL_LAGS if lag <= min(split_arrays[split]["history"].shape[1] for split in splits)],
            "seasonal_lag_unit": "step_count_not_hours",
            "split_roles": bridge["datasets"][dataset]["split_roles"],
        }
        key_findings[dataset] = {
            "last_vs_raw": [
                row
                for row in summary_rows
                if row["family"] == "bridge_simple_alternatives" and row["dataset"] == dataset and row["method"] in {"LAST", "RAW"}
            ]
        }
    return summary_rows, channel_rows, horizon_rows, provenance, key_findings


def method_record_from_json(path: Path, method: str) -> dict:
    data = read_json(path)
    matches = [entry for entry in data.get("methods", []) if entry.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"METHOD_RECORD_NOT_UNIQUE:{path}:{method}")
    record = matches[0]["prediction"]
    source = {"json_path": rel(path), "json_sha256": sha256_path(path), "method": method, "record": record}
    return {"record": record, "source": source, "json": data}


def appliances_method_specs():
    specs = []
    for method, json_method, seed in (
        ("RAW", "RAW_BOLT", "shared_reference"),
        ("RAW+R", "RAW_BOLT_RAW_R", "shared_reference"),
    ):
        dev = method_record_from_json(APPLIANCES_JSONS[("RAW", "dev", None)], json_method)
        eval_record = method_record_from_json(APPLIANCES_JSONS[("RAW", "eval", None)], json_method)
        specs.append({"method": method, "seed": seed, "dev": dev, "eval": eval_record})
    for seed in (92601, 92602):
        dev = method_record_from_json(APPLIANCES_JSONS[("LEARNED", "dev", seed)], "BOLT")
        eval_record = method_record_from_json(APPLIANCES_JSONS[("LEARNED", "eval", seed)], "BOLT")
        specs.append({"method": "LEARNED", "seed": seed, "dev": dev, "eval": eval_record})
    return specs


def load_appliance_source(method_spec: dict, split: str):
    branch = "eval" if split == "eval" else "dev"
    arrays, npz_source = checked_npz(method_spec[branch]["record"])
    return arrays, {"npz": npz_source, "json": method_spec[branch]["source"]}


def validate_appliance_conditions(reference: dict, candidate: dict, split: str, label: str):
    for key in ("mean", "std"):
        exact_array(reference[key], candidate[key], f"appliances:{split}:{label}:{key}")
    for field in ("target", "observed", "origins"):
        exact_array(reference[f"{split}_{field}"], candidate[f"{split}_{field}"], f"appliances:{split}:{label}:{field}")


def run_appliances():
    summary_rows, channel_rows, horizon_rows, tail_rows = [], [], [], []
    provenance = {"status": "RUNNING", "located_assets": [], "asset_proof_before_reads": True}
    try:
        specs = appliances_method_specs()
        provenance["located_assets"] = [
            {
                "method": spec["method"],
                "seed": spec["seed"],
                "dev_json": spec["dev"]["source"],
                "eval_json": spec["eval"]["source"],
            }
            for spec in specs
        ]
        raw_spec = next(spec for spec in specs if spec["method"] == "RAW")
        split_sources = {}
        key_raw_plus_r_vs_raw = []
        key_learned_vs_raw_plus_r = []
        for split in ("fit", "tune", "eval"):
            raw_arrays, raw_source = load_appliance_source(raw_spec, split)
            split_sources.setdefault(split, []).append({"method": "RAW", "seed": "shared_reference", **raw_source})
            raw_prediction = raw_arrays[f"{split}_prediction"]
            target = raw_arrays[f"{split}_target"]
            observed = raw_arrays[f"{split}_observed"]
            raw_summary, raw_channels, raw_horizons, raw_score = method_rows(
                family="appliances_oldalignment",
                dataset="appliances",
                split=split,
                method="RAW",
                seed="shared_reference",
                prediction=raw_prediction,
                target=target,
                observed=observed,
            )
            summary_rows.append(raw_summary)
            channel_rows.extend(raw_channels)
            horizon_rows.extend(raw_horizons)
            scores_by_method = {"RAW": raw_summary}
            errors_by_method = {"RAW": raw_score["error"]}
            for spec in specs:
                if spec["method"] == "RAW":
                    continue
                arrays, source = load_appliance_source(spec, split)
                split_sources.setdefault(split, []).append({"method": spec["method"], "seed": spec["seed"], **source})
                validate_appliance_conditions(raw_arrays, arrays, split, f"{spec['method']}:{spec['seed']}")
                prediction = arrays[f"{split}_prediction"]
                row, channels, horizons, score_value = method_rows(
                    family="appliances_oldalignment",
                    dataset="appliances",
                    split=split,
                    method=spec["method"],
                    seed=spec["seed"],
                    prediction=prediction,
                    target=target,
                    observed=observed,
                    raw_score=raw_score,
                )
                summary_rows.append(row)
                channel_rows.extend(channels)
                horizon_rows.extend(horizons)
                tail_rows.extend(
                    fixed_raw_tail_rows(
                        raw_score["error"],
                        score_value["error"],
                        observed,
                        dataset="appliances",
                        split=split,
                        method=spec["method"],
                        seed=spec["seed"],
                    )
                )
                scores_by_method.setdefault(spec["method"], []).append(row)
                errors_by_method[f"{spec['method']}:{spec['seed']}"] = score_value["error"]
            raw_r = [row for row in summary_rows if row["family"] == "appliances_oldalignment" and row["split"] == split and row["method"] == "RAW+R"][0]
            key_raw_plus_r_vs_raw.append(
                {
                    "split": split,
                    "mse_relative_vs_raw": raw_r["normalized_mse"] / raw_summary["normalized_mse"] - 1.0,
                    "mae_relative_vs_raw": raw_r["normalized_mae"] / raw_summary["normalized_mae"] - 1.0,
                    "mse_gap_vs_raw": raw_r["mse_gap_vs_raw"],
                    "mae_gap_vs_raw": raw_r["mae_gap_vs_raw"],
                }
            )
            learned_rows = [row for row in summary_rows if row["family"] == "appliances_oldalignment" and row["split"] == split and row["method"] == "LEARNED"]
            learned_mean = {
                "normalized_mse": float(np.mean([row["normalized_mse"] for row in learned_rows])),
                "normalized_mae": float(np.mean([row["normalized_mae"] for row in learned_rows])),
                "normalized_signed_bias": float(np.mean([row["normalized_signed_bias"] for row in learned_rows])),
                "seed_count": len(learned_rows),
            }
            key_learned_vs_raw_plus_r.append(
                {
                    "split": split,
                    "learned_seed_mean": learned_mean,
                    "mse_relative_vs_raw_plus_r": learned_mean["normalized_mse"] / raw_r["normalized_mse"] - 1.0,
                    "mae_relative_vs_raw_plus_r": learned_mean["normalized_mae"] / raw_r["normalized_mae"] - 1.0,
                    "mse_relative_vs_raw": learned_mean["normalized_mse"] / raw_summary["normalized_mse"] - 1.0,
                    "mae_relative_vs_raw": learned_mean["normalized_mae"] / raw_summary["normalized_mae"] - 1.0,
                }
            )
        provenance.update(
            status="FOUND_AND_SCORED",
            split_sources=split_sources,
            methods=["RAW", "RAW+R", "LEARNED"],
            splits=["fit", "tune", "eval"],
            scope="existing cached old-alignment arrays; no old campaign restart; EVAL is already exposed by existing JSON/NPZ records",
            raw_cache_cost_reuse_declared=True,
            teacher_cache_cost_reuse_declared=True,
        )
        key_findings = {
            "raw_plus_r_vs_raw": key_raw_plus_r_vs_raw,
            "learned_seed_mean_vs_raw_plus_r": key_learned_vs_raw_plus_r,
            "mae_up_mse_down_cases_vs_raw_plus_r": [
                row
                for row in key_learned_vs_raw_plus_r
                if row["mse_relative_vs_raw_plus_r"] < 0 and row["mae_relative_vs_raw_plus_r"] > 0
            ],
        }
    except Exception as error:
        provenance.update(status="UNAVAILABLE", unavailable_reason={"type": type(error).__name__, "message": str(error)})
        key_findings = {"unavailable": provenance["unavailable_reason"]}
    return summary_rows, channel_rows, horizon_rows, tail_rows, provenance, key_findings


def save_sources(output: Path) -> dict:
    folder = output / "sources"
    folder.mkdir()
    copied = []
    for source in (Path(__file__).resolve(), HERE / "followup_runtime.py"):
        target = folder / source.name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append({"path": rel(source), "sha256": sha256_path(source), "snapshot_path": rel(target), "snapshot_sha256": sha256_path(target)})
    manifest = {"files": copied}
    write_json(folder / "MANIFEST.json", manifest)
    return {"path": rel(folder / "MANIFEST.json"), "sha256": sha256_path(folder / "MANIFEST.json"), "files": copied}


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Simple alternatives CPU 진단",
        "",
        "[확인] 이미 노출·검증된 bridge/D0/old-alignment 배열만 CPU에서 읽었다. 새 모델 추론, GPU, TEST split, old campaign restart는 수행하지 않았다.",
        "",
        f"- 상태: `{result['status']}` / resource terminal `{result.get('resource_terminal_status')}`",
        f"- body wall seconds: `{result.get('body_wall_seconds')}`",
        "- 지표: 관측 mask에서 채널별 MSE/MAE/signed-bias를 계산한 뒤 활성 채널을 균등 평균했다.",
        "- LAST baseline은 bridge `raw_last`를 48 horizon에 broadcast했다.",
        "- seasonal lag 24/48은 step count이며, `history[:, -lag + h % lag]` causal index를 사전 고정해 모두 보존했다.",
        "- Appliances RAW-error tail은 RAW absolute error quantile q=.9/.95/.99 threshold를 고정하고 full-split channel denominator로 contribution을 계산했다.",
        "",
        "## 핵심 수치",
        "",
    ]
    for dataset, rows in result.get("key_findings", {}).get("bridge_simple_alternatives", {}).items():
        last = [row for row in rows["last_vs_raw"] if row["method"] == "LAST"]
        lines.append(f"- {dataset} LAST vs RAW:")
        for row in last:
            lines.append(
                f"  - {row['split']}: LAST mse={row['normalized_mse']:.12g}, mae={row['normalized_mae']:.12g}, "
                f"mse_gap_vs_raw={row.get('mse_gap_vs_raw'):.12g}, mae_gap_vs_raw={row.get('mae_gap_vs_raw'):.12g}"
            )
    app = result.get("key_findings", {}).get("appliances_oldalignment", {})
    if app.get("mae_up_mse_down_cases_vs_raw_plus_r"):
        lines += ["", "- Appliances LEARNED seed-mean cases with MSE down and MAE up vs RAW+R:"]
        for row in app["mae_up_mse_down_cases_vs_raw_plus_r"]:
            lines.append(
                f"  - {row['split']}: MSE rel={row['mse_relative_vs_raw_plus_r']:.6g}, "
                f"MAE rel={row['mae_relative_vs_raw_plus_r']:.6g}, seed_count={row['learned_seed_mean']['seed_count']}"
            )
    else:
        lines += ["", "- Appliances에서 LEARNED seed mean이 RAW+R 대비 MSE 감소와 MAE 증가를 동시에 보인 split은 없거나 자산을 찾지 못했다."]
    lines += [
        "",
        "해석 한계: 이 보고서는 저장 배열의 수치 진단이다. cheap method, causal mechanism, 독립 holdout 성능을 주장하지 않는다.",
    ]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def execute(output: Path) -> dict:
    from research.smallk_failure_followup_20261014 import followup_runtime as fr

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "schema": "smallk_simple_alternatives_phase_v1",
        "gpu_used": False,
        "torch_imported": False,
        "new_model_inference": False,
        "old_campaign_restart": False,
        "protected_TEST_access": False,
        "output": rel(output),
    }
    limit = None
    body_started = None
    try:
        try:
            limit = fr.FollowupLimit("simple_alternatives_phase_v1", False, followup_roots=(HERE, FOLLOWUP_CACHE, output))
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
        bridge = load_bridge()
        limit.check()
        bridge_summary, bridge_channels, bridge_horizons, bridge_prov, bridge_findings = run_bridge_baselines(bridge)
        limit.check()
        app_summary, app_channels, app_horizons, app_tails, app_prov, app_findings = run_appliances()
        summary_rows = bridge_summary + app_summary
        channel_rows = bridge_channels + app_channels
        horizon_rows = bridge_horizons + app_horizons
        write_csv(output / "summary.csv", summary_rows)
        write_csv(output / "channels.csv", channel_rows)
        write_csv(output / "horizons.csv", horizon_rows)
        write_csv(output / "tail_contributions.csv", app_tails)
        provenance = {
            "bridge_simple_alternatives": bridge_prov,
            "appliances_oldalignment": app_prov,
            "source_files": save_sources(output),
            "cost_scope": {
                "actual_compute": "CPU scoring over cached arrays",
                "raw_cache_cost_reuse_declared": True,
                "teacher_or_old_cache_cost_reuse_declared": True,
                "cheap_method_claim": False,
            },
        }
        write_json(output / "PROVENANCE.json", provenance)
        result.update(
            body_wall_seconds=time.perf_counter() - body_started,
            tables={
                "summary": {"path": rel(output / "summary.csv"), "sha256": sha256_path(output / "summary.csv"), "rows": len(summary_rows)},
                "channels": {"path": rel(output / "channels.csv"), "sha256": sha256_path(output / "channels.csv"), "rows": len(channel_rows)},
                "horizons": {"path": rel(output / "horizons.csv"), "sha256": sha256_path(output / "horizons.csv"), "rows": len(horizon_rows)},
                "tail_contributions": {"path": rel(output / "tail_contributions.csv"), "sha256": sha256_path(output / "tail_contributions.csv"), "rows": len(app_tails)},
            },
            provenance={"path": rel(output / "PROVENANCE.json"), "sha256": sha256_path(output / "PROVENANCE.json")},
            key_findings={"bridge_simple_alternatives": bridge_findings, "appliances_oldalignment": app_findings},
        )
        result["status"] = "COMPLETE"
        result["torch_import_scope"] = (
            "No diagnostic code imports or uses torch. The required FollowupLimit imports the frozen Bi runtime, "
            "and that runtime imports torch at module import time."
        )
        write_report(output / "REPORT_KO.md", result)
        result["report"] = {"path": rel(output / "REPORT_KO.md"), "sha256": sha256_path(output / "REPORT_KO.md")}
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
            "No diagnostic code imports or uses torch. If present, torch was imported by the required FollowupLimit/Bi runtime.",
        )
        write_json(output / "FOLLOWUP_RUNTIME_RECEIPT.json", result.get("resource_terminal"))
        result["runtime_receipt"] = {"path": rel(output / "FOLLOWUP_RUNTIME_RECEIPT.json"), "sha256": sha256_path(output / "FOLLOWUP_RUNTIME_RECEIPT.json")}
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
