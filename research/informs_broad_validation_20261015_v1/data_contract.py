"""Explicit safe-prefix data contract; no downloads or data opens at import."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("INFORMS_ASSET_ROOT", str(Path(__file__).resolve().parents[2])))
CACHE = Path(os.environ.get("INFORMS_CACHE_ROOT", str(ROOT / ".cache/informs_broad_validation_20261015_v1")))
ELECTRICITY_PROTECTED_CUTOFF = "2014-10-07T00:00:00"
LEGACY = {
    "beijing": (".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/fit_tune_payload.npz", "78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277", 1524625, 0, 21038, 11, 3600, "physical"),
    "jena": (".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz", "29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933", 1561064, 0, 20965, 21, 600, "legacy_normalized"),
    "robin": (".cache/tsfm_peft_level_confirmation_v6_20260928/data/trainval.npz", "04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14", 111474, 528, 2121, 17, 3600, "legacy_normalized"),
}
CONTRACTS = {
    "beijing": (".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/contract.json", "9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f"),
    "jena": ("research/level_matched_learned_ed_v18_20261006/data_contract_jena.json", "678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820"),
    "robin": ("research/level_matched_learned_ed_v18_20261006/data_contract_robin.json", "b7c02be4f69617e1c60dd874ef6337facde1a4165f8cbd30233aefb0601a85b2"),
}


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    value = hashlib.sha256()
    value.update(str(array.dtype).encode())
    value.update(json.dumps(list(array.shape)).encode())
    value.update(array.tobytes())
    return value.hexdigest()


def split_ranges(data_or_rows, fold="primary"):
    rows = len(data_or_rows["physical"]) if isinstance(data_or_rows, dict) else int(data_or_rows)
    if rows < 1 or fold not in {"primary", "secondary"}:
        raise ValueError("Positive row count and registered fold required")
    if fold == "secondary":
        rows = int(rows * 0.8)
    cuts = [0, int(rows * .5), int(rows * .6), int(rows * .7), int(rows * .8), rows]
    return {name: (cuts[i], cuts[i + 1]) for i, name in enumerate(("A_fit", "A_stop", "M_cal", "V_select", "T_final"))}


def origins(bounds, L, H, stride):
    start, stop = map(int, bounds)
    if min(L, H, stride) < 1 or start < 0 or stop < start:
        raise ValueError("Invalid window/origin bounds")
    return np.arange(max(start, L), stop - H + 1, stride, dtype=np.int64)


def fit_statistics(data, fit_end):
    if fit_end < 1 or fit_end > len(data["physical"]):
        raise ValueError("Statistics range outside allowed data")
    x = np.asarray(data["physical"][:fit_end], dtype=np.float64)
    mask = np.asarray(data["observed"][:fit_end], bool)
    count = mask.sum(axis=0)
    if np.any(count == 0) or not np.isfinite(x[mask]).all():
        raise ValueError("A has unobserved channel or observed nonfinite value")
    mean64 = np.where(mask, x, 0).sum(axis=0) / count
    var = np.where(mask, x - mean64, 0) ** 2
    std64 = np.sqrt(var.sum(axis=0) / count)
    std64[std64 == 0] = 1
    return mean64.astype(np.float32), std64.astype(np.float32), count


def causal_forward_fill(physical, observed, initial):
    physical = np.asarray(physical, np.float32)
    observed = np.asarray(observed, bool)
    if physical.ndim != 2 or physical.shape != observed.shape:
        raise ValueError("Physical/mask shape mismatch")
    if not np.isfinite(physical[observed]).all() or not np.isfinite(initial).all():
        raise ValueError("Observed value or fill initial is nonfinite")
    state = np.asarray(initial, np.float32).copy()
    result = np.empty_like(physical)
    for i in range(len(physical)):
        state[observed[i]] = physical[i, observed[i]]
        result[i] = state
    return result


def history_window(data, origin, mean, std, length=None):
    length = int(data.get("L", 512) if length is None else length)
    if origin < length or origin > len(data["physical"]):
        raise ValueError("History outside allowed prefix")
    mask = data["observed"][origin - length:origin].copy()
    filled = causal_forward_fill(data["physical"][origin - length:origin], mask, mean)
    return ((filled - mean) / std).astype(np.float32), mask


def target_window(data, origin, mean, std, horizon=None):
    horizon = int(data.get("H", 48) if horizon is None else horizon)
    if origin < 0 or origin + horizon > len(data["physical"]):
        raise ValueError("Target outside allowed prefix")
    mask = data["observed"][origin:origin + horizon].copy()
    x = np.where(mask, data["physical"][origin:origin + horizon], mean)
    return ((x - mean) / std).astype(np.float32), mask


def initialize_pca(data, fit_end, k, mean=None, std=None):
    if mean is None or std is None:
        mean, std, _ = fit_statistics(data, fit_end)
    if not 1 <= k <= data["physical"].shape[1]:
        raise ValueError("PCA K outside channel count")
    filled = causal_forward_fill(data["physical"][:fit_end], data["observed"][:fit_end], mean)
    normalized = ((filled - mean) / std).astype(np.float64)
    center = normalized.mean(axis=0)
    centered = normalized - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = np.maximum(eigenvalues[order], 0), vectors[:, order]
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.where(vectors[pivots, np.arange(len(pivots))] < 0, -1, 1)
    vectors *= signs
    E = vectors[:, :k].astype(np.float32)
    return {"E": E, "D": E.T.copy(), "pca_center": center.astype(np.float32),
            "mean": np.asarray(mean, np.float32), "std": np.asarray(std, np.float32),
            "eigenvalues": eigenvalues, "explained_variance": float(eigenvalues[:k].sum() / eigenvalues.sum()) if eigenvalues.sum() else 0.,
            "fit_end": int(fit_end), "fit_scope": "A-only causal filled rows; centered covariance"}


def parse_electricity_prefix(stream, cutoff=ELECTRICITY_PROTECTED_CUTOFF, expected_channels=321):
    header, columns, matrix = [], [], []
    in_data = False
    common_start, common_total, allowed_rows = None, None, None
    cutoff64 = np.datetime64(cutoff, "ns")
    for raw_line in stream:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            header.append(line)
            if line.lower() == "@data":
                in_data = True
            continue
        series_id, timestamp, value_text = line.split(":", 2)
        start = np.datetime64(datetime.strptime(timestamp, "%Y-%m-%d %H-%M-%S"), "ns")
        total = value_text.count(",") + 1
        count = max(0, min(total, int(np.ceil(float((cutoff64 - start) / np.timedelta64(1, "h"))))))
        if common_start is None:
            common_start, common_total, allowed_rows = start, total, count
        if start != common_start or total != common_total or count != allowed_rows:
            raise ValueError("Electricity series time ranges differ")
        # split(maxsplit=count) leaves the complete protected tail opaque text.
        prefix_tokens = value_text.split(",", count)[:count]
        values = np.asarray([float(v) if v != "?" else np.nan for v in prefix_tokens], np.float32)
        columns.append(series_id)
        matrix.append(values)
    if len(matrix) != expected_channels or not matrix:
        raise ValueError("Electricity channel count differs from registered whole panel")
    physical = np.stack(matrix, axis=1)
    times = common_start.astype(np.int64) + np.arange(allowed_rows, dtype=np.int64) * 3600 * 1_000_000_000
    if np.any(times >= cutoff64.astype(np.int64)):
        raise AssertionError("Protected electricity timestamp entered parsed prefix")
    metadata = {"columns": columns, "header": header, "start_timestamp": str(common_start),
                "total_source_rows": common_total, "permitted_rows": allowed_rows,
                "protected_cutoff_exclusive": str(cutoff64), "numeric_values_parsed": allowed_rows * len(columns),
                "tail_numeric_values_not_parsed": (common_total - allowed_rows) * len(columns),
                "protected_tail_policy": "Count/split opaque text only; float conversion permitted prefix only",
                "timestamp_timezone": "Source does not specify timezone; protected project UTC boundary applied as naive matching clock", "timestamp_offset_seconds": 1}
    return physical, times, metadata


def _load_legacy(name):
    relative, expected_hash, size, lo, hi, channels, interval, kind = LEGACY[name]
    path = ROOT / relative
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != expected_hash:
        raise ValueError("Exact safe legacy asset absent or changed")
    contract_path, contract_sha = CONTRACTS[name]
    if sha256_file(ROOT / contract_path) != contract_sha:
        raise ValueError("Pinned legacy contract changed")
    contract = json.loads((ROOT / contract_path).read_text(encoding="utf-8-sig"))
    with np.load(path, allow_pickle=False) as archive:
        mask_key = "mask" if kind == "physical" else "finite"
        observed = archive[mask_key][lo:hi].astype(bool, copy=True)
        if kind == "physical":
            physical = archive["physical"][lo:hi].astype(np.float32, copy=True)
        else:
            x = archive["x"][lo:hi].astype(np.float32, copy=True)
            physical = (x * archive["std"].astype(np.float32) + archive["mean"].astype(np.float32)).astype(np.float32)
        times = archive["times"][lo:hi].copy()
        columns = archive["columns"].tolist()
    if kind == "physical":
        if tuple(contract["bounds"]["fit"]) != (lo, hi):
            raise ValueError("Legacy FIT boundary changed")
    else:
        train_start, train_end = contract["bounds"]["train"]
        if (train_start, train_start + int(.8 * (train_end - train_start))) != (lo, hi):
            raise ValueError("Legacy TRAIN/FIT boundary changed")
    expected_columns = contract.get("columns", contract.get("source", {}).get("parse", {}).get("columns"))
    if expected_columns is not None and columns != expected_columns:
        raise ValueError("Legacy channel order changed")
    times = times.astype("datetime64[ns]").astype(np.int64) if times.dtype.kind in "US" else times.astype(np.int64)
    metadata = {"source_reference": relative, "source_sha256": expected_hash, "source_bytes": size,
                "source_contract_sha256": contract_sha, "permitted_source_bounds": [lo, hi],
                "source_kind": kind, "historically_unseen_to_researcher": False,
                "archive_deserialization_scope": "Required NPZ members decompress in full then FIT slicing; old TUNE/VAL physically deserialized but excluded from all returned arrays/computation",
                "ancestry": "Physical FP32 observations" if kind == "physical" else "FP32 inverse of legacy full-TRAIN normalization; rounding and normalization ancestry remain; not lossless raw A-only", "new_to_campaign": False, "pretraining_overlap_known": "unknown"}
    return physical, observed, times, columns, interval, metadata


def load_task(name, *, cache_dir=None):
    if name.startswith("m5_panel"):
        raise FileNotFoundError("BLOCKED_ACCESS_OR_LICENSE: M5 local source absent, official Kaggle route requires authenticated access; no mirror or terms acceptance")
    if name in LEGACY:
        physical, observed, times, columns, interval, metadata = _load_legacy(name)
    else:
        cache = CACHE if cache_dir is None else Path(cache_dir)
        path = cache / "data" / f"{name}_permitted.npz"
        receipt_path = cache / "data" / f"{name}_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if sha256_file(path) != receipt["cache_sha256"]:
            raise ValueError("Prepared data cache fingerprint mismatch")
        with np.load(path, allow_pickle=False) as archive:
            physical, observed = archive["physical"].copy(), archive["observed"].copy()
            times, columns = archive["times"].copy(), archive["columns"].tolist()
        interval, metadata = receipt["interval_seconds"], receipt["metadata"]
    if physical.shape != observed.shape or physical.ndim != 2 or len(times) != len(physical) or len(columns) != physical.shape[1]:
        raise ValueError("Prepared data axes do not match")
    if not np.isfinite(physical[observed]).all() or not np.all(np.diff(times) == interval * 1_000_000_000):
        raise ValueError("Observed values or timestamp grid invalid")
    physical[~observed] = np.nan
    hashes = {"physical": array_sha256(physical), "observed": array_sha256(observed), "times": array_sha256(times),
              "columns": hashlib.sha256(json.dumps(columns, ensure_ascii=False).encode()).hexdigest()}
    data_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"name": name, "dataset": name, "physical": physical, "observed": observed,
            "times": times, "columns": columns, "L": 512, "H": 12 if name == "pems08" else 48,
            "interval_seconds": interval, "period": 288 if name == "pems08" else (144 if name == "jena" else 24),
            "data_sha256": data_sha, "array_sha256": hashes, "metadata": metadata,
            "split_ranges": split_ranges(len(physical)), "secondary_split_ranges": split_ranges(len(physical), "secondary")}
