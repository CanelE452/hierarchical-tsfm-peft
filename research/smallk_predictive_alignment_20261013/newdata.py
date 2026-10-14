"""Locked UCI data contracts for the small-K follow-up study.

This module deliberately separates:

* FIT/TUNE payloads used for training and model selection.
* EVAL past-input payloads used to run frozen selected models.
* EVAL future target/mask payloads opened only with a model-selection lock.

Actual source downloads are opt-in.  Tests can pass local synthetic archives via
``source_path`` without touching the real UCI datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import io
import json
import time
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / HERE.name
L = 512
H = 48
FIT_N = 256
TUNE_N = 64

APPLIANCE_CHANNELS = (
    "Appliances",
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
)

BEIJING_CHANNELS = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    uci_id: int
    title: str
    page_url: str
    download_url: str
    license: str
    timestamp_column: str
    channels: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    interval_seconds: int
    station: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "appliances": DatasetSpec(
        key="appliances",
        uci_id=374,
        title="Appliances Energy Prediction",
        page_url="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        download_url="https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        license="CC BY 4.0",
        timestamp_column="date",
        channels=APPLIANCE_CHANNELS,
        excluded_columns=("date", "rv1", "rv2"),
        interval_seconds=600,
    ),
    "beijing_aotizhongxin": DatasetSpec(
        key="beijing_aotizhongxin",
        uci_id=501,
        title="Beijing Multi-Site Air Quality Data / Aotizhongxin",
        page_url="https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        download_url="https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
        license="CC BY 4.0",
        timestamp_column="timestamp",
        channels=BEIJING_CHANNELS,
        excluded_columns=("No", "year", "month", "day", "hour", "wd", "station"),
        interval_seconds=3600,
        station="Aotizhongxin",
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _select_origins(values: np.ndarray, maximum: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if maximum is None or len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum).astype(int)]


def _method_lock_record(method_lock_path: Path) -> dict[str, Any]:
    lock_path = Path(method_lock_path)
    if not lock_path.exists():
        raise FileNotFoundError(f"METHOD_LOCK missing: {lock_path}")
    lock = read_json(lock_path)
    status = str(lock.get("status", ""))
    if status != "METHODS_LOCKED_BEFORE_NEW_RAW_DOWNLOAD":
        raise ValueError(f"Unexpected METHOD_LOCK status: {status!r}")
    source_hashes = lock.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("METHOD_LOCK source_hashes must be a non-empty mapping")
    _verify_source_hashes(source_hashes, base=ROOT)
    return {
        "path": str(lock_path.resolve()),
        "sha256": sha256_path(lock_path),
        "status": status,
        "created_unix": lock.get("created_unix"),
        "source_hashes": source_hashes,
    }


def _resolve_declared_path(path: str | Path, base: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def _verify_source_hashes(source_hashes: dict[str, str], base: Path = ROOT) -> None:
    for raw_path, expected in source_hashes.items():
        path = _resolve_declared_path(raw_path, base=base)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Declared source file missing: {path}")
        actual = sha256_path(path)
        if actual != expected:
            raise ValueError(f"Declared source hash mismatch for {path}: {actual} != {expected}")


def _download_source(spec: DatasetSpec, cache_root: Path) -> Path:
    raw_dir = cache_root / "raw_dataset" / spec.key / "download"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"uci_{spec.uci_id}.zip"
    if destination.exists():
        return destination
    temporary = destination.with_suffix(".tmp")
    with urllib.request.urlopen(spec.download_url, timeout=60) as response:
        payload = response.read()
    with temporary.open("xb") as handle:
        handle.write(payload)
    temporary.replace(destination)
    return destination


def _resolve_source(spec: DatasetSpec, cache_root: Path, source_path: str | Path | None, allow_download: bool) -> Path:
    if source_path is not None:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not allow_download:
        raise RuntimeError("SOURCE_NOT_MATERIALIZED: pass source_path or set allow_download=True after METHOD_LOCK")
    return _download_source(spec, cache_root)


def _read_archive_members(source_path: Path) -> list[tuple[str, bytes]]:
    source_path = Path(source_path)
    if zipfile.is_zipfile(source_path):
        members = []
        with zipfile.ZipFile(source_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    members.append((name, archive.read(name)))
                elif name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(archive.read(name))) as nested:
                        for inner_name in nested.namelist():
                            if inner_name.lower().endswith(".csv"):
                                members.append((f"{name}!{inner_name}", nested.read(inner_name)))
        if not members:
            raise ValueError(f"No CSV member in {source_path}")
        return members
    if source_path.suffix.lower() == ".csv":
        return [(source_path.name, source_path.read_bytes())]
    raise ValueError(f"Unsupported source format: {source_path}")


def _load_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import failure is environment-specific.
        raise RuntimeError("pandas is required for UCI CSV ingestion") from exc
    return pd


def _parse_appliances(source_path: Path, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    pd = _load_pandas()
    members = _read_archive_members(source_path)
    chosen = None
    for name, payload in members:
        if "energydata_complete" in name.lower():
            chosen = (name, payload)
            break
    if chosen is None:
        chosen = members[0]
    frame = pd.read_csv(io.BytesIO(chosen[1]))
    if spec.timestamp_column not in frame.columns:
        raise ValueError("Appliances timestamp column missing")
    missing = [column for column in spec.channels if column not in frame.columns]
    if missing:
        raise ValueError(f"Appliances expected columns missing: {missing}")
    timestamps = pd.to_datetime(frame[spec.timestamp_column], errors="raise")
    order = np.argsort(timestamps.to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    values = frame[list(spec.channels)].apply(pd.to_numeric, errors="coerce")
    return _frame_to_arrays(values, timestamps, spec, chosen[0])


def _parse_beijing(source_path: Path, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    pd = _load_pandas()
    frames = []
    source_members = []
    for name, payload in _read_archive_members(source_path):
        if spec.station and spec.station.lower() not in name.lower() and len(frames) == 0:
            # Still parse later if no station-named member exists.
            pass
        frame = pd.read_csv(io.BytesIO(payload))
        source_members.append(name)
        frames.append(frame)
    if not frames:
        raise ValueError("No Beijing CSV frames parsed")
    station_frames = []
    for frame in frames:
        if "station" in frame.columns:
            station_frames.append(frame[frame["station"].astype(str) == spec.station])
        elif spec.station and spec.station.lower() in ",".join(map(str, source_members)).lower():
            station_frames.append(frame)
    frame = pd.concat(station_frames if station_frames else frames, ignore_index=True)
    if frame.empty:
        raise ValueError(f"No rows for station {spec.station}")
    missing = [column for column in spec.channels if column not in frame.columns]
    if missing:
        raise ValueError(f"Beijing expected columns missing: {missing}")
    for column in ("year", "month", "day", "hour"):
        if column not in frame.columns:
            raise ValueError(f"Beijing timestamp component missing: {column}")
    timestamps = pd.to_datetime(
        {
            "year": frame["year"],
            "month": frame["month"],
            "day": frame["day"],
            "hour": frame["hour"],
        },
        errors="raise",
    )
    order = np.argsort(timestamps.to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    values = frame[list(spec.channels)].apply(pd.to_numeric, errors="coerce")
    return _frame_to_arrays(values, timestamps, spec, ",".join(source_members))


def _frame_to_arrays(values: Any, timestamps: Any, spec: DatasetSpec, member_name: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    duplicate_timestamps = int(timestamps.duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"Duplicate timestamps are not allowed for this time-series contract: {duplicate_timestamps}")
    diffs = timestamps.diff().dropna().dt.total_seconds().to_numpy(dtype=np.float64)
    median_interval = float(np.median(diffs)) if len(diffs) else float("nan")
    physical = values.to_numpy(dtype=np.float32)
    mask = np.isfinite(physical)
    physical = np.where(mask, physical, np.nan).astype(np.float32)
    time_strings = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    meta = {
        "source_member": member_name,
        "rows": int(len(values)),
        "columns": list(spec.channels),
        "duplicate_timestamps": duplicate_timestamps,
        "median_interval_seconds": median_interval,
        "expected_interval_seconds": spec.interval_seconds,
        "interval_mismatch_count": int(np.sum(np.abs(diffs - spec.interval_seconds) > 1e-6)) if len(diffs) else 0,
        "missing_values_total": int((~mask).sum()),
        "missing_fraction_by_column": {
            column: float((~mask[:, index]).mean()) for index, column in enumerate(spec.channels)
        },
    }
    if len(physical) < L + H + 20:
        raise ValueError(f"Not enough rows for L={L}, H={H}: {len(physical)}")
    return physical, mask, list(spec.channels), time_strings, meta


def _parse_source(dataset: str, source_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    spec = DATASETS[dataset]
    if dataset == "appliances":
        return _parse_appliances(source_path, spec)
    if dataset == "beijing_aotizhongxin":
        return _parse_beijing(source_path, spec)
    raise KeyError(dataset)


def split_bounds(n_rows: int) -> dict[str, tuple[int, int]]:
    fit_end = int(np.floor(0.60 * n_rows))
    tune_end = int(np.floor(0.80 * n_rows))
    bounds = {"fit": (0, fit_end), "tune": (fit_end, tune_end), "eval": (tune_end, n_rows)}
    for name, (lo, hi) in bounds.items():
        if hi - lo < H:
            raise ValueError(f"{name} split too short for H={H}: {(lo, hi)}")
    return bounds


def make_origins(bounds: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    fit_lo, fit_hi = bounds["fit"]
    tune_lo, tune_hi = bounds["tune"]
    eval_lo, eval_hi = bounds["eval"]
    origins = {
        "fit": _select_origins(np.arange(max(L, fit_lo + L), fit_hi - H + 1, dtype=np.int64), FIT_N),
        "tune": _select_origins(np.arange(max(L, tune_lo), tune_hi - H + 1, dtype=np.int64), TUNE_N),
        "eval": np.arange(max(L, eval_lo), eval_hi - H + 1, H, dtype=np.int64),
    }
    for split, values in origins.items():
        lo, hi = bounds[split]
        if len(values) == 0:
            raise ValueError(f"No legal origins for {split}")
        if np.any(values < lo) or np.any(values - L < 0) or np.any(values + H > hi):
            raise ValueError(f"Illegal target/history bounds for {split}")
    if len(origins["eval"]) > 1 and np.any(np.diff(origins["eval"]) < H):
        raise ValueError("EVAL future horizons overlap")
    return origins


def fit_statistics(physical: np.ndarray, mask: np.ndarray, fit_bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = fit_bounds
    fit_physical = physical[lo:hi]
    fit_mask = mask[lo:hi]
    masked = np.where(fit_mask, fit_physical, np.nan)
    mean = np.nanmean(masked, axis=0).astype(np.float32)
    std = np.nanstd(masked, axis=0).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError("At least one channel has no observed FIT value")
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def causal_fill_full(physical: np.ndarray, mask: np.ndarray, mean: np.ndarray) -> np.ndarray:
    filled = np.empty_like(physical, dtype=np.float32)
    last = mean.astype(np.float32).copy()
    for row in range(len(physical)):
        observed = mask[row]
        last[observed] = physical[row, observed]
        filled[row] = last
    return filled


def window_forward_fill_normalize(physical_window: np.ndarray, mask_window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.empty_like(physical_window, dtype=np.float32)
    last = mean.astype(np.float32).copy()
    for row in range(len(physical_window)):
        observed = mask_window[row]
        last[observed] = physical_window[row, observed]
        filled[row] = last
    return ((filled - mean) / std).astype(np.float32)


def target_normalize(physical_target: np.ndarray, mask_target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe = np.where(mask_target, physical_target, mean).astype(np.float32)
    return ((safe - mean) / std).astype(np.float32)


def _build_payloads(physical: np.ndarray, mask: np.ndarray, columns: list[str], times: list[str], dataset: str) -> dict[str, Any]:
    n_rows = len(physical)
    bounds = split_bounds(n_rows)
    origins = make_origins(bounds)
    mean, std = fit_statistics(physical, mask, bounds["fit"])
    filled = causal_fill_full(physical, mask, mean)
    normalized = ((filled - mean) / std).astype(np.float32)
    tune_end = bounds["tune"][1]

    eval_history = np.stack(
        [window_forward_fill_normalize(physical[o - L : o], mask[o - L : o], mean, std) for o in origins["eval"]]
    ).astype(np.float32)
    eval_history_mask = np.stack([mask[o - L : o] for o in origins["eval"]])
    eval_target_physical = np.stack([physical[o : o + H] for o in origins["eval"]]).astype(np.float32)
    eval_target_mask = np.stack([mask[o : o + H] for o in origins["eval"]])
    eval_target = target_normalize(eval_target_physical, eval_target_mask, mean, std)

    return {
        "dataset": dataset,
        "bounds": bounds,
        "origins": origins,
        "mean": mean,
        "std": std,
        "training": {
            "x": normalized[:tune_end],
            "physical": physical[:tune_end],
            "filled": filled[:tune_end],
            "mask": mask[:tune_end],
            "times": np.array(times[:tune_end]),
        },
        "eval_input": {
            "history": eval_history,
            "physical_history": np.stack([physical[o - L : o] for o in origins["eval"]]).astype(np.float32),
            "history_mask": eval_history_mask,
            "origins": origins["eval"],
            "times": np.array(times),
        },
        "eval_target": {
            "target": eval_target,
            "target_physical": eval_target_physical,
            "target_mask": eval_target_mask,
            "origins": origins["eval"],
        },
        "columns": np.array(columns),
    }


def _source_quality(physical: np.ndarray, mask: np.ndarray, columns: list[str], bounds: dict[str, tuple[int, int]]) -> dict[str, Any]:
    split_missing = {}
    for split, (lo, hi) in bounds.items():
        local = mask[lo:hi]
        split_missing[split] = {
            "rows": hi - lo,
            "observed_fraction": float(local.mean()),
            "missing_fraction_by_column": {
                column: float((~local[:, index]).mean()) for index, column in enumerate(columns)
            },
        }
    return {
        "rows": int(len(physical)),
        "channels": int(len(columns)),
        "overall_observed_fraction": float(mask.mean()),
        "split_missingness": split_missing,
        "domain_leakage_checks": {
            "temporal_order_preserved": True,
            "fit_statistics_scope": "FIT rows only",
            "training_payload_contains_eval_rows": False,
            "eval_past_input_contains_future_target": False,
            "eval_target_payload_open_requires_model_selection_lock": True,
        },
    }


def prepare(
    dataset: str,
    method_lock_path: str | Path,
    *,
    source_path: str | Path | None = None,
    allow_download: bool = False,
    cache_root: str | Path = CACHE,
) -> dict[str, Any]:
    """Materialize split payloads after METHOD_LOCK and return the contract.

    ``allow_download`` defaults to False so this function cannot accidentally
    fetch the real UCI archives during pre-lock or synthetic-test work.
    """
    if dataset not in DATASETS:
        raise KeyError(dataset)
    cache_root = Path(cache_root)
    spec = DATASETS[dataset]
    method_lock = _method_lock_record(Path(method_lock_path))
    registry_path = cache_root / "contract" / f"{dataset}_{method_lock['sha256']}.json"
    source = _resolve_source(spec, cache_root, source_path, allow_download)
    source_sha = sha256_path(source)

    if registry_path.exists():
        registry = read_json(registry_path)
        contract_path = Path(registry["contract_path"])
        if sha256_path(contract_path) != registry["contract_sha256"]:
            raise ValueError("Registered contract JSON hash mismatch")
        contract = read_json(contract_path)
        contract["contract_sha256"] = registry["contract_sha256"]
        if source_path is not None and contract["source"]["sha256"] != source_sha:
            raise ValueError("Existing contract source hash differs from provided source_path")
        _verify_contract_files(contract, include_eval_target=False)
        return contract

    physical, mask, columns, times, parse_meta = _parse_source(dataset, source)
    payloads = _build_payloads(physical, mask, columns, times, dataset)
    data_dir = cache_root / "raw_dataset" / dataset / source_sha[:16]
    if data_dir.exists():
        raise FileExistsError(f"Refusing to reuse incomplete unregistered data dir: {data_dir}")
    data_dir.mkdir(parents=True)

    fit_tune_path = data_dir / "fit_tune_payload.npz"
    eval_input_path = data_dir / "eval_past_input_payload.npz"
    eval_target_path = data_dir / "eval_future_target_payload.npz"

    _write_npz_once(
        fit_tune_path,
        x=payloads["training"]["x"],
        physical=payloads["training"]["physical"],
        filled=payloads["training"]["filled"],
        mask=payloads["training"]["mask"],
        mean=payloads["mean"],
        std=payloads["std"],
        origins_fit=payloads["origins"]["fit"],
        origins_tune=payloads["origins"]["tune"],
        columns=payloads["columns"],
        times=payloads["training"]["times"],
    )
    _write_npz_once(
        eval_input_path,
        history=payloads["eval_input"]["history"],
        physical_history=payloads["eval_input"]["physical_history"],
        history_mask=payloads["eval_input"]["history_mask"],
        origins=payloads["eval_input"]["origins"],
        mean=payloads["mean"],
        std=payloads["std"],
        columns=payloads["columns"],
        times=payloads["eval_input"]["times"],
    )
    _write_npz_once(
        eval_target_path,
        target=payloads["eval_target"]["target"],
        target_physical=payloads["eval_target"]["target_physical"],
        target_mask=payloads["eval_target"]["target_mask"],
        origins=payloads["eval_target"]["origins"],
        mean=payloads["mean"],
        std=payloads["std"],
        columns=payloads["columns"],
    )

    hashes = {
        "fit_tune_payload_sha256": sha256_path(fit_tune_path),
        "eval_input_payload_sha256": sha256_path(eval_input_path),
        "eval_target_payload_sha256": sha256_path(eval_target_path),
    }
    contract = {
        "dataset": dataset,
        "created_unix": time.time(),
        "spec": {
            "uci_id": spec.uci_id,
            "title": spec.title,
            "page_url": spec.page_url,
            "download_url": spec.download_url,
            "license": spec.license,
            "station": spec.station,
            "channels": list(spec.channels),
            "excluded_columns": list(spec.excluded_columns),
            "interval_seconds": spec.interval_seconds,
        },
        "source": {
            "path": str(Path(source).resolve()),
            "sha256": source_sha,
            "parse": parse_meta,
            "raw_eval_values_ingested_for_split_store": True,
            "training_or_selection_reads_eval_target": False,
        },
        "method_lock": method_lock,
        "bounds": {name: list(bounds) for name, bounds in payloads["bounds"].items()},
        "origins": {
            name: {
                "count": int(len(values)),
                "sha256": sha256_array(values.astype(np.int64)),
                "first": int(values[0]),
                "last": int(values[-1]),
            }
            for name, values in payloads["origins"].items()
        },
        "normalization": {
            "fit_only_mean_sha256": sha256_array(payloads["mean"]),
            "fit_only_std_sha256": sha256_array(payloads["std"]),
            "std_floor": 1e-6,
            "input_window_forward_fill": "reset_to_fit_mean_at_each_window_start",
            "future_missing_values": "preserve_actual_observation_mask; no forward-filled target fabrication",
        },
        "payloads": {
            "fit_tune": {"path": str(fit_tune_path.resolve()), "sha256": hashes["fit_tune_payload_sha256"]},
            "eval_past_input": {"path": str(eval_input_path.resolve()), "sha256": hashes["eval_input_payload_sha256"]},
            "eval_future_target": {"path": str(eval_target_path.resolve()), "sha256": hashes["eval_target_payload_sha256"]},
        },
        "quality": _source_quality(physical, mask, columns, payloads["bounds"]),
        "contract_note": "FIT/TUNE loader does not read eval_future_target_payload; open_evaluation verifies MODEL_SELECTION_LOCK.",
    }
    contract_path = data_dir / "contract.json"
    _write_json_once(contract_path, contract)
    contract_sha = sha256_path(contract_path)
    registry = {"dataset": dataset, "contract_path": str(contract_path.resolve()), "contract_sha256": contract_sha}
    _write_json_once(registry_path, registry)
    contract["contract_sha256"] = contract_sha
    return contract


def _contract_from_registry(
    dataset: str,
    method_lock_path: str | Path,
    cache_root: str | Path,
    *,
    include_eval_target: bool = False,
) -> dict[str, Any]:
    method_lock = _method_lock_record(Path(method_lock_path))
    registry_path = Path(cache_root) / "contract" / f"{dataset}_{method_lock['sha256']}.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"No materialized contract registry: {registry_path}")
    registry = read_json(registry_path)
    contract_path = Path(registry["contract_path"])
    if sha256_path(contract_path) != registry["contract_sha256"]:
        raise ValueError("Registered contract JSON hash mismatch")
    contract = read_json(contract_path)
    contract["contract_sha256"] = registry["contract_sha256"]
    if contract["method_lock"]["sha256"] != method_lock["sha256"]:
        raise ValueError("Contract METHOD_LOCK hash differs from current METHOD_LOCK")
    _verify_contract_files(contract, include_eval_target=include_eval_target)
    return contract


def _verify_contract_files(contract: dict[str, Any], *, include_eval_target: bool) -> None:
    payloads = dict(contract["payloads"])
    if not include_eval_target:
        target_payload = payloads.pop("eval_future_target")
        target_path = Path(target_payload["path"])
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(target_path)
    for payload in payloads.values():
        path = Path(payload["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_path(path)
        if actual != payload["sha256"]:
            raise ValueError(f"Payload hash mismatch for {path}: {actual} != {payload['sha256']}")


def load_training(
    dataset: str,
    method_lock_path: str | Path,
    *,
    source_path: str | Path | None = None,
    allow_download: bool = False,
    cache_root: str | Path = CACHE,
) -> dict[str, Any]:
    """Return a legacy-compatible FIT/TUNE dictionary without EVAL target values."""
    if source_path is not None or allow_download:
        contract = prepare(
            dataset,
            method_lock_path,
            source_path=source_path,
            allow_download=allow_download,
            cache_root=cache_root,
        )
    else:
        contract = _contract_from_registry(dataset, method_lock_path, cache_root)
    payload_path = Path(contract["payloads"]["fit_tune"]["path"])
    if sha256_path(payload_path) != contract["payloads"]["fit_tune"]["sha256"]:
        raise ValueError("FIT/TUNE payload hash mismatch before load")
    with np.load(payload_path, allow_pickle=False) as z:
        data = {
            "dataset": dataset,
            "x": z["x"].astype(np.float32),
            "physical": z["physical"].astype(np.float32),
            "filled": z["filled"].astype(np.float32),
            "mask": z["mask"].astype(bool),
            "mean": z["mean"].astype(np.float32),
            "std": z["std"].astype(np.float32),
            "origins": {"fit": z["origins_fit"].astype(np.int64), "tune": z["origins_tune"].astype(np.int64)},
            "fit_bounds": tuple(contract["bounds"]["fit"]),
            "tune_bounds": tuple(contract["bounds"]["tune"]),
            "eval_bounds": tuple(contract["bounds"]["eval"]),
            "columns": [str(value) for value in z["columns"].tolist()],
            "times": [str(value) for value in z["times"].tolist()],
            "interval_seconds": DATASETS[dataset].interval_seconds,
            "source": contract["source"]["path"],
            "source_hash": contract["source"]["sha256"],
            "contract_sha256": contract["contract_sha256"],
            "fit_tune_payload_sha256": contract["payloads"]["fit_tune"]["sha256"],
            "normalization_fit_scope": "FIT_ONLY",
            "input_window_forward_fill": "RESET_TO_FIT_MEAN_PER_WINDOW",
            "eval_target_loaded": False,
        }
    if data["x"].shape[0] != data["tune_bounds"][1]:
        raise ValueError("Training payload unexpectedly includes rows beyond TUNE end")
    return data


def required_model_selection_fields(dataset: str, method_lock_path: str | Path, *, cache_root: str | Path = CACHE) -> dict[str, Any]:
    contract = _contract_from_registry(dataset, method_lock_path, cache_root)
    return {
        "dataset": dataset,
        "source_sha256": contract["source"]["sha256"],
        "method_lock_sha256": contract["method_lock"]["sha256"],
        "contract_sha256": contract["contract_sha256"],
        "fit_tune_payload_sha256": contract["payloads"]["fit_tune"]["sha256"],
        "eval_input_payload_sha256": contract["payloads"]["eval_past_input"]["sha256"],
        "source_hashes": contract["method_lock"]["source_hashes"],
    }


def _lock_payload(lock: dict[str, Any]) -> dict[str, Any]:
    return lock.get("newdata_contract", lock)


def _selected_model_artifacts(lock: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = payload.get("selected_models") or payload.get("model_artifacts") or lock.get("selected_models")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("MODEL_SELECTION_LOCK must name selected model artifacts before EVAL target open")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("MODEL_SELECTION_LOCK selected model artifacts must be objects")
        if not artifact.get("path") or not artifact.get("sha256"):
            raise ValueError("MODEL_SELECTION_LOCK selected model artifacts require path and sha256")
    return artifacts


def _verify_model_artifacts(artifacts: list[dict[str, Any]]) -> None:
    for artifact in artifacts:
        path = _resolve_declared_path(artifact["path"], base=ROOT)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Selected model artifact missing: {path}")
        actual = sha256_path(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"Selected model artifact hash mismatch for {path}: {actual} != {artifact['sha256']}")


def open_evaluation(
    dataset: str,
    model_selection_lock_path: str | Path,
    *,
    method_lock_path: str | Path | None = None,
    cache_root: str | Path = CACHE,
) -> dict[str, Any]:
    """Open EVAL past inputs and future targets after model-selection sealing."""
    lock_path = Path(model_selection_lock_path)
    lock = read_json(lock_path)
    status = str(lock.get("status", ""))
    if status != "MODEL_SELECTION_LOCKED_BEFORE_EVAL_TARGET_OPEN":
        raise ValueError(f"Unexpected MODEL_SELECTION_LOCK status: {status!r}")
    payload = _lock_payload(lock)
    if method_lock_path is None:
        method_lock_hash = payload.get("method_lock_sha256")
        if not method_lock_hash:
            raise ValueError("MODEL_SELECTION_LOCK must include method_lock_sha256 or pass method_lock_path")
        registry_candidates = list((Path(cache_root) / "contract").glob(f"{dataset}_{method_lock_hash}.json"))
        if not registry_candidates:
            raise FileNotFoundError(f"No registry for method lock hash {method_lock_hash}")
        registry = read_json(registry_candidates[0])
        contract_path = Path(registry["contract_path"])
        if sha256_path(contract_path) != registry["contract_sha256"]:
            raise ValueError("Registered contract JSON hash mismatch")
        contract = read_json(contract_path)
        contract["contract_sha256"] = registry["contract_sha256"]
    else:
        contract = _contract_from_registry(dataset, method_lock_path, cache_root, include_eval_target=False)
    _verify_source_hashes(contract["method_lock"]["source_hashes"], base=ROOT)
    expected = {
        "dataset": dataset,
        "source_sha256": contract["source"]["sha256"],
        "method_lock_sha256": contract["method_lock"]["sha256"],
        "contract_sha256": contract["contract_sha256"],
        "fit_tune_payload_sha256": contract["payloads"]["fit_tune"]["sha256"],
        "eval_input_payload_sha256": contract["payloads"]["eval_past_input"]["sha256"],
        "source_hashes": contract["method_lock"]["source_hashes"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"MODEL_SELECTION_LOCK {key} mismatch")
    selected_models = _selected_model_artifacts(lock, payload)
    _verify_model_artifacts(selected_models)
    _verify_contract_files(contract, include_eval_target=True)
    input_path = Path(contract["payloads"]["eval_past_input"]["path"])
    target_path = Path(contract["payloads"]["eval_future_target"]["path"])
    with np.load(input_path, allow_pickle=False) as inp, np.load(target_path, allow_pickle=False) as target:
        origins = inp["origins"].astype(np.int64)
        target_origins = target["origins"].astype(np.int64)
        if not np.array_equal(origins, target_origins):
            raise ValueError("EVAL input/target origins mismatch")
        result = {
            "dataset": dataset,
            "history": inp["history"].astype(np.float32),
            "physical_history": inp["physical_history"].astype(np.float32),
            "history_observed": inp["history_mask"].astype(bool),
            "target": target["target"].astype(np.float32),
            "target_physical": target["target_physical"].astype(np.float32),
            "observed": target["target_mask"].astype(bool),
            "origins": origins,
            "mean": inp["mean"].astype(np.float32),
            "std": inp["std"].astype(np.float32),
            "columns": [str(value) for value in inp["columns"].tolist()],
            "contract_sha256": contract["contract_sha256"],
            "eval_input_payload_sha256": contract["payloads"]["eval_past_input"]["sha256"],
            "eval_target_payload_sha256": contract["payloads"]["eval_future_target"]["sha256"],
            "model_selection_lock_sha256": sha256_path(lock_path),
            "selection_lock_status": status,
        }
    return result


def copy_official_download_command(dataset: str) -> str:
    """Return the exact later command; this function does not run it."""
    if dataset not in DATASETS:
        raise KeyError(dataset)
    spec = DATASETS[dataset]
    return (
        "python -c \"from research.smallk_predictive_alignment_20261013 import newdata; "
        f"newdata.prepare('{dataset}', 'research/smallk_predictive_alignment_20261013/METHOD_LOCK.json', "
        "allow_download=True)\""
    )

