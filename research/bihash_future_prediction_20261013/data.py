"""Data contracts for the BiHash future-prediction campaign.

The module keeps the data boundary explicit:

* training loaders expose FIT/TUNE history and target only;
* EVAL past input and EVAL future target are separate payloads;
* EVAL future target is opened only after a joint model-selection lock;
* legacy Jena/Robin loaders refuse protected TEST files.

Fresh parsers do not download anything. They only materialize from local raw
files declared in a method-lock JSON or passed explicitly by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
L = 512
H = 48
FIT_ORIGINS = 256
TUNE_ORIGINS = 64
BANK_ORIGINS = 128
EVAL_STRIDE = 48

HOUSEHOLD_ALIASES = {"uci235", "uci235_household", "uci235_household_hourly", "household", "household_power"}
SML_ALIASES = {"uci274", "uci274_sml2010", "sml2010", "sml"}
CANONICAL_DATASETS = ("uci235_household_hourly", "uci274_sml2010")
HOUSEHOLD_CHANNELS = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)
SML_ALL_COLUMNS = (
    "Date",
    "Time",
    "Indoor_temperature_dinning_room",
    "Indoor_temperature_room",
    "Weather_forecast_temperature",
    "Carbon_dioxide_dinning_room",
    "Carbon_dioxide_room",
    "Relative_humidity_dinning_room",
    "Relative_humidity_room",
    "Lighting_dinning_room",
    "Lighting_room",
    "Rain_last_15min_fraction",
    "Sun_dusk",
    "Wind",
    "Sun_light_west_facade",
    "Sun_light_east_facade",
    "Sun_light_south_facade",
    "Sun_irradiance",
    "Enthalpic_motor_1",
    "Enthalpic_motor_2",
    "Enthalpic_motor_turbo",
    "Outdoor_temperature",
    "Outdoor_relative_humidity",
    "Day_of_week",
)
SML_CHANNELS = SML_ALL_COLUMNS[2:23]

_PROTECTED_GUARD_INSTALLED = False


@dataclass(frozen=True)
class ParsedSource:
    physical: np.ndarray
    observed: np.ndarray
    columns: list[str]
    times: np.ndarray
    source_records: list[dict[str, Any]]
    metadata: dict[str, Any]


def canonical_dataset(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key in HOUSEHOLD_ALIASES:
        return "uci235_household_hourly"
    if key in SML_ALIASES:
        return "uci274_sml2010"
    if key in CANONICAL_DATASETS or key in {"jena", "robin"}:
        return key
    raise ValueError(f"Unsupported dataset: {dataset}")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
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


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: os.PathLike[str] | str, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _legacy_contract_path(dataset: str) -> Path:
    ds = canonical_dataset(dataset)
    if ds not in {"jena", "robin"}:
        raise ValueError("Legacy loader supports only jena/robin")
    return ROOT / f"research/level_matched_learned_ed_v18_20261006/data_contract_{ds}.json"


def _protected_legacy_test_paths() -> set[Path]:
    paths: set[Path] = set()
    for ds in ("jena", "robin"):
        cpath = _legacy_contract_path(ds)
        if not cpath.exists():
            continue
        contract = read_json(cpath)
        test = contract.get("test", {}).get("path")
        if test:
            paths.add((ROOT / test).resolve())
    return paths


def access_guard() -> list[str]:
    """Install a process-local audit hook that refuses legacy protected TEST reads."""
    global _PROTECTED_GUARD_INSTALLED
    protected = _protected_legacy_test_paths()
    if not _PROTECTED_GUARD_INSTALLED:
        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(args[0])).resolve()
                if path in protected or path.name.lower() in {"test.npz", "jena_test.npz"}:
                    raise PermissionError(f"Protected TEST access denied: {path}")
        sys.addaudithook(audit)
        _PROTECTED_GUARD_INSTALLED = True
    return [str(p) for p in sorted(protected)]


def _check_not_test(path: Path) -> Path:
    resolved = path.resolve()
    if resolved in _protected_legacy_test_paths() or resolved.name.lower() in {"test.npz", "jena_test.npz"}:
        raise PermissionError(f"Protected TEST access denied: {resolved}")
    return resolved


def _record(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": relative_or_absolute(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _payload_record(path: Path) -> dict[str, Any]:
    return {"path": relative_or_absolute(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _uniform_origins(values: Iterable[int], maximum: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.int64)
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum).astype(np.int64)]


def _split_bounds(n: int) -> dict[str, tuple[int, int]]:
    fit_end = int(np.floor(0.6 * n))
    tune_end = int(np.floor(0.8 * n))
    return {"fit": (0, fit_end), "tune": (fit_end, tune_end), "eval": (tune_end, n)}


def _legal_origins(
    bounds: tuple[int, int],
    n_rows: int,
    *,
    stride: int = 1,
    future_observed: np.ndarray | None = None,
) -> np.ndarray:
    lo, hi = bounds
    start = max(L, lo)
    stop = min(hi - H, n_rows - H)
    if stop < start:
        return np.empty(0, dtype=np.int64)
    origins = np.arange(start, stop + 1, stride, dtype=np.int64)
    if future_observed is not None:
        keep = [bool(future_observed[o:o + H].any(axis=0).all()) for o in origins]
        origins = origins[np.asarray(keep, dtype=bool)]
    return origins


def _fit_statistics(physical: np.ndarray, observed: np.ndarray, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = bounds
    values = np.where(observed[lo:hi], physical[lo:hi], np.nan).astype(np.float64, copy=False)
    with np.errstate(invalid="ignore"):
        mean64 = np.nanmean(values, axis=0)
        std64 = np.nanstd(values, axis=0)
    if not np.isfinite(mean64).all():
        raise ValueError("FIT statistics undefined: at least one channel has no FIT observations")
    std64 = np.where(std64 > 1e-6, std64, 1.0)
    return mean64.astype(np.float32), std64.astype(np.float32)


def normalize_future_target(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (physical.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.where(observed, z, 0.0).astype(np.float32)


def window_forward_fill_normalize(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    physical = np.asarray(physical, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if physical.shape != observed.shape or physical.ndim != 2:
        raise ValueError("Window physical/mask must be 2D with identical shape")
    last = np.asarray(mean, dtype=np.float32).copy()
    result = np.empty_like(physical, dtype=np.float32)
    for i in range(len(physical)):
        mask = observed[i]
        last[mask] = physical[i, mask]
        result[i] = (last - mean) / std
    return result.astype(np.float32)


def _history_windows(
    physical: np.ndarray,
    observed: np.ndarray,
    origins: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([physical[o - L:o] for o in origins]).astype(np.float32)
    mask = np.stack([observed[o - L:o] for o in origins]).astype(bool)
    history = np.stack([window_forward_fill_normalize(r, m, mean, std) for r, m in zip(raw, mask)]).astype(np.float32)
    return history, raw, mask


def _target_windows(
    physical: np.ndarray,
    observed: np.ndarray,
    origins: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([physical[o:o + H] for o in origins]).astype(np.float32)
    mask = np.stack([observed[o:o + H] for o in origins]).astype(bool)
    target = normalize_future_target(raw, mask, mean, std)
    return target, raw, mask


def _pca_from_history_pool(history_fit: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pool = history_fit.reshape(-1, history_fit.shape[-1]).astype(np.float64)
    center = pool.mean(axis=0)
    centered = pool - center
    cov = centered.T @ centered / max(len(centered), 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    k = int(min(max(1, k), eigvec.shape[1]))
    basis = eigvec[:, :k].copy()
    for j in range(k):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    return center.astype(np.float32), basis.astype(np.float32), {
        "fit_history_pool_shape": list(pool.shape),
        "covariance_dtype": "float64",
        "eig_order": "descending",
        "canonical_sign": "largest_abs_loading_positive",
        "eigenvalues": eigval[:k].astype(float).tolist(),
        "note": "PCA initializes E/D only; trainable E/D means this is not a fixed-compression claim.",
    }


def _default_k(channels: int, lock: dict[str, Any] | None = None, dataset: str | None = None) -> int:
    if lock:
        canonical = canonical_dataset(dataset) if dataset else None
        for key in (dataset, canonical, "default"):
            if key and isinstance(lock.get("K"), dict) and key in lock["K"]:
                return int(lock["K"][key])
        if isinstance(lock.get("K"), int):
            return int(lock["K"])
    return max(1, channels // 4)


def _build_bank(origins_fit: np.ndarray, history_fit: np.ndarray, target_fit: np.ndarray, observed_fit: np.ndarray) -> dict[str, np.ndarray]:
    bank_origins = _uniform_origins(origins_fit, BANK_ORIGINS)
    positions = np.searchsorted(origins_fit, bank_origins)
    if len(positions) and not np.array_equal(origins_fit[positions], bank_origins):
        raise ValueError("Bank origins must be a subset of FIT origins")
    last = history_fit[:, -1].astype(np.float32)
    full_delta = np.where(observed_fit, target_fit - last[:, None, :], 0.0).astype(np.float32)
    return {
        "bank_origins": bank_origins.astype(np.int64),
        "bank_target_end_exclusive": (bank_origins + H).astype(np.int64),
        "bank_delta": full_delta[positions].astype(np.float32),
        "bank_observed": observed_fit[positions].astype(bool),
        "full_fit_delta": full_delta.astype(np.float32),
        "full_fit_observed": observed_fit.astype(bool),
        "fit_history_last": last.astype(np.float32),
    }


def causal_bank_eligibility(query_origins: np.ndarray, bank_origins: np.ndarray, *, length: int = L) -> np.ndarray:
    query_origins = np.asarray(query_origins, dtype=np.int64)
    bank_end = np.asarray(bank_origins, dtype=np.int64) + H
    row_times = query_origins[:, None] - length + np.arange(length, dtype=np.int64)[None, :]
    return bank_end[None, None, :] <= row_times[:, :, None] + 1


def _assemble_tuple(
    dataset: str,
    physical: np.ndarray,
    observed: np.ndarray,
    columns: list[str],
    times: np.ndarray,
    origins: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    source_records: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    k: int,
    basis: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    pca_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    history: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    future_observed: dict[str, np.ndarray] = {}
    history_observed: dict[str, np.ndarray] = {}
    rawhistory: dict[str, np.ndarray] = {}
    rawhistorymask: dict[str, np.ndarray] = {}
    for split, oo in origins.items():
        hx, hraw, hm = _history_windows(physical, observed, oo, mean, std)
        ty, _traw, tm = _target_windows(physical, observed, oo, mean, std)
        history[split], target[split], future_observed[split] = hx, ty, tm
        history_observed[split] = hm
        rawhistory[split], rawhistorymask[split] = hraw, hm
    if basis is None or pca_mean is None:
        pca_mean, basis, pca_metadata = _pca_from_history_pool(history["fit"], k)
    bank = _build_bank(origins["fit"], history["fit"], target["fit"], future_observed["fit"])
    data = {
        "dataset": dataset,
        "C": int(len(columns)),
        "K": int(basis.shape[1]),
        "columns": list(columns),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "origins": {s: np.asarray(o, dtype=np.int64) for s, o in origins.items()},
        "raw_history": rawhistory,
        "raw_history_observed": rawhistorymask,
        "rawhistory": rawhistory,
        "rawhistorymask": rawhistorymask,
        "source_records": source_records,
        "contract": contract,
        "basis": basis.astype(np.float32),
        "pca_mean": pca_mean.astype(np.float32),
        "init_mean": pca_mean.astype(np.float32),
        "codec_mean": pca_mean.astype(np.float32),
        "pca_metadata": pca_metadata or {},
        "bank": bank,
        "times": times,
        "target_fill_policy": "future target uses no forward fill; missing cells are stored as 0 with bool mask",
        "history_fill_policy": "each L512 history window resets causal forward fill to FIT mean",
    }
    return data, history, target, future_observed, history_observed


def load_legacy(dataset: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load exposed legacy TRAIN/DEVVAL only. Protected TEST is never opened."""
    ds = canonical_dataset(dataset)
    if ds not in {"jena", "robin"}:
        raise ValueError("load_legacy supports only jena/robin")
    protected = access_guard()
    contract_path = _legacy_contract_path(ds)
    contract = read_json(contract_path)
    source = _check_not_test(ROOT / contract["trainval"]["path"])
    if sha256_file(source) != contract["trainval"]["sha256"]:
        raise ValueError(f"Legacy TRAIN/DEVVAL SHA mismatch: {source}")
    with np.load(source, allow_pickle=False) as z:
        x = z["x"].astype(np.float32)
        mask = z["finite"].astype(bool)
        source_mean = z["mean"].astype(np.float32)
        source_std = np.where(z["std"].astype(np.float32) > 1e-6, z["std"].astype(np.float32), 1.0).astype(np.float32)
        columns = [str(v) for v in z["columns"].tolist()]
        times = z["times"].copy()
        val_origins = z["val_origins"].astype(np.int64)
    physical = (x * source_std + source_mean).astype(np.float32)
    physical[~mask] = np.nan
    train_bounds = tuple(contract.get("bounds", {}).get("train", contract.get("train_bounds")))
    if len(train_bounds) != 2:
        raise ValueError("Legacy contract missing train bounds")
    fit_end = train_bounds[0] + int(np.floor(0.8 * (train_bounds[1] - train_bounds[0])))
    bounds = {"fit": (train_bounds[0], fit_end), "tune": (fit_end, train_bounds[1]), "val": tuple(contract["bounds"]["val"])}
    fit_legal_all = _legal_origins(bounds["fit"], len(physical))
    tune_legal_all = _legal_origins(bounds["tune"], len(physical))
    fit_legal_observed = _legal_origins(bounds["fit"], len(physical), future_observed=mask)
    tune_legal_observed = _legal_origins(bounds["tune"], len(physical), future_observed=mask)
    origins = {
        "fit": _uniform_origins(fit_legal_observed, FIT_ORIGINS),
        "tune": _uniform_origins(tune_legal_observed, TUNE_ORIGINS),
        "val": val_origins,
    }
    for split, oo in origins.items():
        if len(oo) == 0 or np.any(oo - L < 0) or np.any(oo + H > len(physical)):
            raise ValueError(f"Illegal legacy origins for {ds}:{split}")
    mean, std = _fit_statistics(physical, mask, bounds["fit"])
    k = _default_k(len(columns), None, ds)
    payload_contract = {
        "kind": "legacy_train_devval_only",
        "legacy_contract_path": relative_or_absolute(contract_path),
        "legacy_contract_sha256": sha256_file(contract_path),
        "protected_test_guard": protected,
        "split_note": "VAL is exposed DEVVAL and remains labelled val; protected TEST is not loaded.",
        "bounds": {k_: list(v) for k_, v in bounds.items()},
        "normalization": "campaign FIT-only mean/std recomputed from reconstructed physical values",
        "legacy_npz_statistics_use": "stored NPZ mean/std are used only to invert saved normalized x into approximate FP32 physical values",
        "source_limit": "legacy physical values are reconstructed from stored normalized x and finite mask, not original raw source bytes",
        "basis": "ignores legacy NPZ basis; initializes fresh PCA from FIT256 history pool with K=floor(C/4)",
        "origin_observability_filter": {
            "rule": "FIT/TUNE origins require every channel to have at least one observed future target cell in the H48 window; availability-only, no value/effect filter",
            "fit_legal_before_filter": int(len(fit_legal_all)),
            "fit_legal_after_filter": int(len(fit_legal_observed)),
            "tune_legal_before_filter": int(len(tune_legal_all)),
            "tune_legal_after_filter": int(len(tune_legal_observed)),
        },
    }
    return _assemble_tuple(
        ds,
        physical,
        mask,
        columns,
        times,
        origins,
        mean,
        std,
        [_record(contract_path, "legacy_contract"), _record(source, "legacy_trainval_npz")],
        payload_contract,
        k=k,
    )


def _extract_from_zip(path: Path, wanted: str | None = None, suffix: str | None = None) -> tuple[bytes, str]:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if wanted is not None:
            matches = [n for n in names if Path(n).name == wanted]
        elif suffix is not None:
            matches = [n for n in names if Path(n).name.lower().endswith(suffix.lower())]
        else:
            matches = names
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one archive member in {path}, found {matches}")
        return archive.read(matches[0]), matches[0]


def _parse_household(source: Path) -> ParsedSource:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Raw source missing; no download is attempted: {source}")
    if source.suffix.lower() == ".zip":
        content, member = _extract_from_zip(source, wanted="household_power_consumption.txt")
        import io
        handle: Any = io.BytesIO(content)
        member_note = member
    else:
        handle = source
        member_note = source.name
    frame = pd.read_csv(handle, sep=";", na_values=["?", ""], low_memory=False)
    required = ["Date", "Time", *HOUSEHOLD_CHANNELS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"UCI235 missing columns: {missing}")
    timestamps = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    order = np.argsort(timestamps.to_numpy())
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    frame = frame.iloc[order].reset_index(drop=True)
    if timestamps.duplicated().any():
        raise ValueError("UCI235 timestamp duplicate violates one-minute uniqueness")
    expected = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq="min")
    if len(expected) != len(timestamps) or not np.array_equal(expected.to_numpy(), timestamps.to_numpy()):
        raise ValueError("UCI235 timestamp grid is not a complete one-minute grid")
    values = frame[list(HOUSEHOLD_CHANNELS)].apply(pd.to_numeric, errors="coerce")
    values.index = timestamps
    observed_minute = values.notna()
    hour_key = values.index.floor("h")
    full_hour = pd.Series(1, index=timestamps).groupby(hour_key).count()
    eligible_hours = full_hour[full_hour == 60].index
    hourly_sum = values.groupby(hour_key).sum(min_count=1).loc[eligible_hours]
    hourly_count = observed_minute.groupby(hour_key).sum().loc[eligible_hours]
    hourly_mean = hourly_sum / hourly_count.replace(0, np.nan)
    observed = hourly_count.to_numpy(dtype=np.int64) >= 1
    physical = hourly_mean.to_numpy(dtype=np.float32)
    physical[~observed] = np.nan
    end_of_hour = (eligible_hours + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S").to_numpy(dtype="U19")
    return ParsedSource(
        physical=np.asfortranarray(physical.astype(np.float32)),
        observed=np.asfortranarray(observed.astype(bool)),
        columns=list(HOUSEHOLD_CHANNELS),
        times=end_of_hour,
        source_records=[_record(source, "raw_source")],
        metadata={
            "dataset_id": 235,
            "parser": "semicolon Date/Time + seven numeric channels, then complete-hour observed means",
            "source_member": member_note,
            "raw_minute_rows": int(len(frame)),
            "hourly_rows": int(len(physical)),
            "structural_boundary_hours_excluded": int(full_hour.ne(60).sum()),
            "timestamp_grid": "complete one-minute grid required before hourly aggregation",
            "availability_label": "each hourly row is labelled by end-of-hour timestamp",
            "hour_observed_rule": "channel observed iff at least one minute in the hour is observed; otherwise mask False and value NaN",
            "estimated_rows_before_raw": "official 2,075,259 minute rows imply about 34,587 complete hours before boundary/quality checks",
        },
    )


def _read_sml_file(path: Path, content: bytes | None = None) -> pd.DataFrame:
    import io
    handle: Any = io.BytesIO(content) if content is not None else path
    frame = pd.read_csv(handle, comment="#", header=None, sep=r"[;,\s]+", engine="python")
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != 24:
        raise ValueError(f"SML2010 file must have 24 columns after comments; got {frame.shape[1]} from {path}")
    frame.columns = list(SML_ALL_COLUMNS)
    return frame


def _sml_sources(source: Path) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Raw source missing; no download is attempted: {source}")
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    members: list[str] = []
    if source.is_dir():
        paths = [source / "NEW-DATA-1.T15.txt", source / "NEW-DATA-2.T15.txt"]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            frames.append(_read_sml_file(p))
            records.append(_record(p, "raw_source"))
            members.append(p.name)
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for wanted in ("NEW-DATA-1.T15.txt", "NEW-DATA-2.T15.txt"):
                matches = [n for n in archive.namelist() if Path(n).name == wanted]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {wanted} in {source}, found {matches}")
                frames.append(_read_sml_file(source, archive.read(matches[0])))
                members.append(matches[0])
        records.append(_record(source, "raw_source"))
    else:
        raise ValueError("SML2010 source must be a directory or zip containing NEW-DATA-1/2.T15.txt")
    return frames, records, members


def _parse_sml2010(source: Path) -> ParsedSource:
    frames, records, members = _sml_sources(Path(source))
    frame = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str), utc=True, errors="raise")
    values = frame[list(SML_CHANNELS)].apply(pd.to_numeric, errors="coerce")
    joined = values.copy()
    joined.insert(0, "timestamp", timestamps)
    collapsed: list[pd.Series] = []
    identical_duplicates = 0
    for timestamp, group in joined.groupby("timestamp", sort=True):
        data = group.drop(columns=["timestamp"])
        first = data.iloc[0]
        same = data.apply(lambda row: bool(((row.isna() & first.isna()) | (row == first)).all()), axis=1)
        if not bool(same.all()):
            raise ValueError(f"SML2010 conflicting duplicate timestamp: {timestamp}")
        identical_duplicates += len(group) - 1
        row = first.copy()
        row.name = timestamp
        collapsed.append(row)
    if not collapsed:
        raise ValueError("SML2010 source produced no rows")
    collapsed_frame = pd.DataFrame(collapsed).sort_index()
    full_index = pd.date_range(collapsed_frame.index[0], collapsed_frame.index[-1], freq="15min", tz="UTC")
    reindexed = collapsed_frame.reindex(full_index)
    physical = reindexed.to_numpy(dtype=np.float32)
    observed = np.isfinite(physical)
    physical[~observed] = np.nan
    times = full_index.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(dtype="U20")
    return ParsedSource(
        physical=np.asfortranarray(physical.astype(np.float32)),
        observed=np.asfortranarray(observed.astype(bool)),
        columns=list(SML_CHANNELS),
        times=times,
        source_records=records,
        metadata={
            "dataset_id": 274,
            "parser": "two T15 files, UTC timestamp merge/sort, Date/Time/Day_of_week excluded",
            "source_members": members,
            "raw_rows": int(sum(len(f) for f in frames)),
            "rows_after_duplicate_collapse": int(len(collapsed_frame)),
            "identical_duplicates_collapsed": int(identical_duplicates),
            "full_grid_rows": int(len(reindexed)),
            "gap_rows_inserted_as_nan_mask_false": int(len(reindexed) - len(collapsed_frame)),
            "timestamp_grid": "15-minute UTC full grid after duplicate collapse",
            "interpolation": "none",
            "estimated_rows_before_raw": "official 4,137 rows before duplicate/gap checks",
        },
    )


def _parse_new_source(dataset: str, source: Path) -> ParsedSource:
    ds = canonical_dataset(dataset)
    if ds == "uci235_household_hourly":
        return _parse_household(source)
    if ds == "uci274_sml2010":
        return _parse_sml2010(source)
    raise ValueError(f"Unsupported new dataset: {dataset}")


def _method_lock(method_lock_path: os.PathLike[str] | str) -> tuple[dict[str, Any], Path, str]:
    path = Path(method_lock_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"METHOD_LOCK missing: {path}")
    lock = read_json(path)
    if lock.get("locked") is False or lock.get("method_lock") is False or str(lock.get("status", "")).upper() in {"DRAFT", "UNLOCKED"}:
        raise ValueError("METHOD_LOCK explicitly says it is not locked")
    return lock, path, sha256_file(path)


def _source_from_lock(dataset: str, lock: dict[str, Any], *, source_path: os.PathLike[str] | str | None = None) -> Path:
    if source_path is not None:
        return Path(source_path).resolve()
    ds = canonical_dataset(dataset)
    candidates = []
    for key in ("newdata_sources", "raw_sources", "sources", "datasets"):
        value = lock.get(key)
        if isinstance(value, dict):
            for name in (ds, dataset):
                entry = value.get(name)
                if isinstance(entry, str):
                    candidates.append(entry)
                elif isinstance(entry, dict):
                    for field in ("path", "source_path", "raw_path", "directory", "zip"):
                        if field in entry:
                            candidates.append(entry[field])
                            break
    env_name = "BIHASH_" + ds.upper() + "_SOURCE"
    if os.environ.get(env_name):
        candidates.append(os.environ[env_name])
    if not candidates:
        raise FileNotFoundError(f"No local raw source declared for {ds}; raw download is not performed")
    return Path(candidates[0]).expanduser().resolve()


def _contract_dir(cache_root: os.PathLike[str] | str | None = None) -> Path:
    return Path(cache_root or CACHE).resolve() / "contract"


def _contract_registry_path(dataset: str, method_hash: str, cache_root: os.PathLike[str] | str | None = None) -> Path:
    return _contract_dir(cache_root) / f"{canonical_dataset(dataset)}_{method_hash}.json"


def _load_registered_contract(
    dataset: str,
    method_lock_path: os.PathLike[str] | str,
    cache_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    _lock, _path, method_hash = _method_lock(method_lock_path)
    registry_path = _contract_registry_path(dataset, method_hash, cache_root)
    if not registry_path.exists():
        raise FileNotFoundError(f"Dataset contract registry missing: {registry_path}")
    registry = read_json(registry_path)
    contract_path = Path(registry["contract_path"]).resolve()
    if sha256_file(contract_path) != registry["contract_sha256"]:
        raise ValueError("Registered contract JSON hash mismatch")
    return read_json(contract_path)


def prepare_new(
    dataset: str,
    method_lock_path: os.PathLike[str] | str,
    *,
    source_path: os.PathLike[str] | str | None = None,
    cache_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Materialize a fresh dataset contract from an already local raw source."""
    ds = canonical_dataset(dataset)
    if ds not in CANONICAL_DATASETS:
        raise ValueError("prepare_new supports only UCI235/UCI274 fresh datasets")
    lock, lock_path, method_hash = _method_lock(method_lock_path)
    source = _source_from_lock(ds, lock, source_path=source_path)
    parsed = _parse_new_source(ds, source)
    n, c = parsed.physical.shape
    if c != len(parsed.columns):
        raise ValueError("Parsed channel count mismatch")
    bounds = _split_bounds(n)
    fit_legal_all = _legal_origins(bounds["fit"], n)
    tune_legal_all = _legal_origins(bounds["tune"], n)
    fit_legal = _legal_origins(bounds["fit"], n, future_observed=parsed.observed)
    tune_legal = _legal_origins(bounds["tune"], n, future_observed=parsed.observed)
    eval_legal = _legal_origins(bounds["eval"], n, stride=EVAL_STRIDE, future_observed=parsed.observed)
    if len(fit_legal) == 0 or len(tune_legal) == 0 or len(eval_legal) == 0:
        raise ValueError(f"Insufficient legal origins after L{L}/H{H} split for {ds}")
    origins_fit = _uniform_origins(fit_legal, FIT_ORIGINS)
    origins_tune = _uniform_origins(tune_legal, TUNE_ORIGINS)
    origins_eval = eval_legal.astype(np.int64)
    mean, std = _fit_statistics(parsed.physical, parsed.observed, bounds["fit"])
    history_fit, _hraw, _hm = _history_windows(parsed.physical, parsed.observed, origins_fit, mean, std)
    k = _default_k(c, lock, ds)
    pca_mean, basis, pca_meta = _pca_from_history_pool(history_fit, k)
    base = Path(cache_root or CACHE).resolve() / ds / method_hash
    base.mkdir(parents=True, exist_ok=True)
    fit_tune_end = bounds["tune"][1]
    fit_tune_path = base / "FIT_TUNE_payload.npz"
    eval_past_path = base / "EVAL_past_input_payload.npz"
    eval_target_path = base / "EVAL_future_target_payload.npz"

    np.savez_compressed(
        fit_tune_path,
        physical=parsed.physical[:fit_tune_end].astype(np.float32),
        mask=parsed.observed[:fit_tune_end].astype(bool),
        times=parsed.times[:fit_tune_end],
        columns=np.asarray(parsed.columns, dtype="U64"),
        mean=mean,
        std=std,
        origins_fit=origins_fit,
        origins_tune=origins_tune,
        basis=basis,
        pca_mean=pca_mean,
    )
    eval_history, eval_raw_history, eval_history_mask = _history_windows(parsed.physical, parsed.observed, origins_eval, mean, std)
    np.savez_compressed(
        eval_past_path,
        history=eval_history,
        physical_history=eval_raw_history,
        history_observed=eval_history_mask,
        origins=origins_eval,
        columns=np.asarray(parsed.columns, dtype="U64"),
        mean=mean,
        std=std,
    )
    eval_target, eval_raw_target, eval_target_mask = _target_windows(parsed.physical, parsed.observed, origins_eval, mean, std)
    np.savez_compressed(
        eval_target_path,
        target=eval_target,
        target_physical=eval_raw_target,
        observed=eval_target_mask,
        origins=origins_eval,
        columns=np.asarray(parsed.columns, dtype="U64"),
        mean=mean,
        std=std,
    )
    payloads = {
        "fit_tune": _payload_record(fit_tune_path),
        "eval_past_input": _payload_record(eval_past_path),
        "eval_future_target": _payload_record(eval_target_path),
    }
    contract = {
        "dataset": ds,
        "created_unix": time.time(),
        "L": L,
        "H": H,
        "C": c,
        "K": int(basis.shape[1]),
        "columns": parsed.columns,
        "method_lock": {"path": relative_or_absolute(lock_path), "sha256": method_hash},
        "source_records": parsed.source_records,
        "parser_metadata": parsed.metadata,
        "bounds": {k_: list(v) for k_, v in bounds.items()},
        "row_split": "floor 60/20/20 by post-parser rows",
        "origins": {
            "fit": {"count": int(len(origins_fit)), "sha256": array_hash(origins_fit), "selection": f"uniform {FIT_ORIGINS} from observability-filtered legal origins"},
            "tune": {"count": int(len(origins_tune)), "sha256": array_hash(origins_tune), "selection": f"uniform {TUNE_ORIGINS} from observability-filtered legal origins"},
            "eval": {"count": int(len(origins_eval)), "sha256": array_hash(origins_eval), "selection": f"stride {EVAL_STRIDE} from all legal origins, every channel has at least one observed future target cell"},
        },
        "origin_observability_filter": {
            "rule": "all FIT/TUNE/EVAL origins require every channel to have at least one observed future target cell in the H48 window; availability-only, no value/effect filter",
            "fit_legal_before_filter": int(len(fit_legal_all)),
            "fit_legal_after_filter": int(len(fit_legal)),
            "tune_legal_before_filter": int(len(tune_legal_all)),
            "tune_legal_after_filter": int(len(tune_legal)),
            "eval_stride": EVAL_STRIDE,
            "eval_legal_after_stride_and_filter": int(len(eval_legal)),
        },
        "payloads": payloads,
        "statistics": {
            "scope": "FIT rows only",
            "reduction_dtype": "float64 then stored float32",
            "constant_std_rule": "std <= 1e-6 is stored as 1",
            "mean_sha256": array_hash(mean),
            "std_sha256": array_hash(std),
        },
        "pca_initialization": {
            **pca_meta,
            "pca_mean_sha256": array_hash(pca_mean),
            "basis_sha256": array_hash(basis),
        },
        "history_contract": "causal forward fill is reset to FIT mean at each L512 window start",
        "target_contract": "future target is not forward filled; missing cells are NaN in physical payload and 0 with bool mask in normalized payload",
        "eval_separation": "FIT_TUNE, EVAL past input, and EVAL future target are separate payloads",
    }
    contract_path = base / "contract.json"
    write_json(contract_path, contract, overwrite=True)
    registry_path = _contract_registry_path(ds, method_hash, cache_root)
    registry = {"dataset": ds, "method_lock_sha256": method_hash, "contract_path": str(contract_path), "contract_sha256": sha256_file(contract_path)}
    write_json(registry_path, registry, overwrite=True)
    return contract


def _resolve_payload_path(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_fit_tune_payload(contract: dict[str, Any]) -> tuple[np.lib.npyio.NpzFile, Path]:
    path = _resolve_payload_path(contract["payloads"]["fit_tune"])
    if sha256_file(path) != contract["payloads"]["fit_tune"]["sha256"]:
        raise ValueError("Payload hash mismatch: FIT_TUNE")
    return np.load(path, allow_pickle=False), path


def load_training_new(
    dataset: str,
    method_lock_path: os.PathLike[str] | str,
    *,
    cache_root: os.PathLike[str] | str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load FIT/TUNE only. This function does not open or hash EVAL target bytes."""
    ds = canonical_dataset(dataset)
    contract = _load_registered_contract(ds, method_lock_path, cache_root)
    z, _payload_path = _load_fit_tune_payload(contract)
    with z:
        physical = z["physical"].astype(np.float32)
        observed = z["mask"].astype(bool)
        times = z["times"].copy()
        columns = [str(v) for v in z["columns"].tolist()]
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)
        origins = {"fit": z["origins_fit"].astype(np.int64), "tune": z["origins_tune"].astype(np.int64)}
        basis = z["basis"].astype(np.float32)
        pca_mean = z["pca_mean"].astype(np.float32)
    data, history, target, future_observed, history_observed = _assemble_tuple(
        ds,
        physical,
        observed,
        columns,
        times,
        origins,
        mean,
        std,
        contract["source_records"],
        contract,
        k=int(contract["K"]),
        basis=basis,
        pca_mean=pca_mean,
        pca_metadata=contract.get("pca_initialization", {}),
    )
    data["eval_target_loaded"] = False
    data["eval_input_loaded"] = False
    data["source_hash"] = contract["source_records"][0]["sha256"] if contract.get("source_records") else None
    return data, history, target, future_observed, history_observed


def required_model_selection_fields(
    dataset: str,
    method_lock_path: os.PathLike[str] | str,
    *,
    cache_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Return lock fields without touching the EVAL target file."""
    ds = canonical_dataset(dataset)
    _lock, _path, method_hash = _method_lock(method_lock_path)
    registry_path = _contract_registry_path(ds, method_hash, cache_root)
    registry = read_json(registry_path)
    contract_path = Path(registry["contract_path"]).resolve()
    contract_hash = sha256_file(contract_path)
    if contract_hash != registry["contract_sha256"]:
        raise ValueError("Registered contract JSON hash mismatch")
    contract = read_json(contract_path)
    return {
        "dataset": ds,
        "method_lock_sha256": method_hash,
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "fit_tune_payload_sha256": contract["payloads"]["fit_tune"]["sha256"],
        "eval_input_payload_sha256": contract["payloads"]["eval_past_input"]["sha256"],
        "eval_target_payload_sha256": contract["payloads"]["eval_future_target"]["sha256"],
        "source_hashes": {r["path"]: r["sha256"] for r in contract["source_records"]},
        "data_sha256": contract_hash,
        "required_before_eval_open": "global lock with ALL_SELECTED and selected model artifact hashes",
    }


def _verify_model_lock(dataset: str, global_model_lock_path: os.PathLike[str] | str, fields: dict[str, Any]) -> dict[str, Any]:
    path = Path(global_model_lock_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Global model lock missing: {path}")
    lock = read_json(path)
    all_selected = bool(lock.get("ALL_SELECTED", False) or lock.get("all_selected", False) or str(lock.get("status", "")).upper() == "ALL_SELECTED")
    if not all_selected:
        raise ValueError("Global model lock must declare ALL_SELECTED before EVAL target open")
    if not (lock.get("created_unix") or lock.get("timestamp") or lock.get("created_at")):
        raise ValueError("Global model lock must include a timestamp")
    contracts = lock.get("newdata_contracts") or lock.get("required_contracts") or {}
    missing = [name for name in CANONICAL_DATASETS if name not in contracts]
    if missing:
        raise ValueError(f"Global model lock must include joint fresh data contracts for both datasets; missing {missing}")
    method_hashes = []
    for name in CANONICAL_DATASETS:
        declared = contracts[name]
        method_hash = declared.get("method_lock_sha256")
        if not method_hash:
            raise ValueError(f"Global model lock missing method_lock_sha256 for {name}")
        method_hashes.append(method_hash)
        cpath_value = declared.get("contract_path")
        if not cpath_value:
            raise ValueError(f"Global model lock missing contract_path for {name}")
        cpath = Path(cpath_value)
        cpath = cpath.resolve() if cpath.is_absolute() else (ROOT / cpath).resolve()
        if not cpath.exists():
            raise FileNotFoundError(f"Declared contract path missing for {name}: {cpath}")
        if sha256_file(cpath) != declared.get("contract_sha256"):
            raise ValueError(f"Declared contract_path SHA mismatch for {name}")
        contract = read_json(cpath)
        if contract.get("method_lock", {}).get("sha256") != method_hash:
            raise ValueError(f"Contract method_lock_sha256 mismatch for {name}")
    if len(set(method_hashes)) != 1 or method_hashes[0] != fields["method_lock_sha256"]:
        raise ValueError("Both fresh data contracts must share the current method_lock_sha256")
    declared_current = contracts[canonical_dataset(dataset)]
    for key in ("contract_sha256", "fit_tune_payload_sha256", "eval_input_payload_sha256", "eval_target_payload_sha256"):
        if declared_current.get(key) != fields[key]:
            raise ValueError(f"Global model lock {key} mismatch for {dataset}")

    def declared_artifacts(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if "path" in value:
                found.append(value)
            for child in value.values():
                found.extend(declared_artifacts(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(declared_artifacts(child))
        return found

    artifact_roots = {key: lock.get(key) for key in ("selected_models", "artifacts", "controls", "readouts") if key in lock}
    for artifact in declared_artifacts(artifact_roots):
        expected = artifact.get("sha256") or artifact.get("artifact_sha256")
        if not expected:
            raise ValueError(f"Declared artifact requires non-empty sha256: {artifact.get('path')}")
        apath = Path(artifact["path"])
        apath = apath.resolve() if apath.is_absolute() else (ROOT / apath).resolve()
        if not apath.exists():
            raise FileNotFoundError(f"Declared artifact missing: {apath}")
        if sha256_file(apath) != expected:
            raise ValueError(f"Declared artifact hash mismatch: {apath}")
    return lock


def open_new_evaluation(
    dataset: str,
    global_model_lock_path: os.PathLike[str] | str,
    method_lock_path: os.PathLike[str] | str,
    *,
    cache_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Open EVAL past input and future target after joint model-selection lock."""
    ds = canonical_dataset(dataset)
    fields = required_model_selection_fields(ds, method_lock_path, cache_root=cache_root)
    lock = _verify_model_lock(ds, global_model_lock_path, fields)
    contract = read_json(fields["contract_path"])
    for record in contract["source_records"]:
        spath = Path(record["path"])
        spath = spath.resolve() if spath.is_absolute() else (ROOT / spath).resolve()
        if not spath.exists() or sha256_file(spath) != record["sha256"]:
            raise ValueError(f"Declared source hash mismatch: {spath}")
    eval_past_path = _resolve_payload_path(contract["payloads"]["eval_past_input"])
    eval_target_path = _resolve_payload_path(contract["payloads"]["eval_future_target"])
    if sha256_file(eval_past_path) != contract["payloads"]["eval_past_input"]["sha256"]:
        raise ValueError("Payload hash mismatch: EVAL past input")
    if sha256_file(eval_target_path) != contract["payloads"]["eval_future_target"]["sha256"]:
        raise ValueError("Payload hash mismatch: EVAL future target")
    with np.load(eval_past_path, allow_pickle=False) as p, np.load(eval_target_path, allow_pickle=False) as t:
        if not np.array_equal(p["origins"], t["origins"]):
            raise ValueError("EVAL input/target origins mismatch")
        if not np.array_equal(p["columns"], t["columns"]):
            raise ValueError("EVAL input/target columns mismatch")
        return {
            "dataset": ds,
            "history": p["history"].astype(np.float32),
            "target": t["target"].astype(np.float32),
            "observed": t["observed"].astype(bool),
            "origins": p["origins"].astype(np.int64),
            "physical_history": p["physical_history"].astype(np.float32),
            "history_observed": p["history_observed"].astype(bool),
            "target_physical": t["target_physical"].astype(np.float32),
            "columns": [str(v) for v in p["columns"].tolist()],
            "mean": p["mean"].astype(np.float32),
            "std": p["std"].astype(np.float32),
            "contract": contract,
            "global_model_lock": lock,
            "payload_hashes": {
                "eval_past_input": sha256_file(eval_past_path),
                "eval_future_target": sha256_file(eval_target_path),
            },
        }


