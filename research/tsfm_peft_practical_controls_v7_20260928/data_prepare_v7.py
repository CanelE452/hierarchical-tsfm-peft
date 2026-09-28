
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from runtime_v7 import CACHE, HERE, ROOT, V1, V6, Job, digest, save_json

CONTEXT = 512
HORIZON = 48
JENA_SOURCE = Path("E:/CODING/proj/mltimeseries/data/jena_mpi_roof/mpi_roof_2024.csv")
JENA_ZIP = Path("E:/CODING/proj/mltimeseries/data/jena_mpi_roof/_zip/mpi_roof_2024.zip")
JENA_SOURCE_SHA256 = "65a47db98ad85b0600f2deab1e11055af8fef9c5aedceadbdcd0011748120558"
JENA_ZIP_SHA256 = "3ce424b9d1671ebc26c09921c7d4f9444282acb1e8a5f020d90a156287ba83aa"
JENA_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m²)",
    "PAR (µmol/m²/s)", "max. PAR (µmol/m²/s)", "Tlog (degC)", "CO2 (ppm)",
]
V6_DATA_CONTRACT = V6 / "data_contract.json"
V6_EVALUATION = V6 / "evaluation01.json"
V6_SELECTION = V6 / "selected.json"
V6_SHARED = V6 / "shared_ridge_selection.json"
V1_LINEAR_FULLTRAIN = V1 / "linear_fulltrain.py"
V1_DATA_PREPARE = V1 / "data_prepare.py"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def save_npz_new(path: Path, **arrays: Any) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    return digest(path)


def parse_time(value: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, format="%d.%m.%Y %H:%M:%S", errors="raise"))


def split_bounds(times: pd.Series, start: str, end: str) -> tuple[int, int]:
    values = times.to_numpy(dtype="datetime64[ns]")
    start64 = np.datetime64(pd.Timestamp(start), "ns")
    end64 = np.datetime64(pd.Timestamp(end), "ns")
    return int(np.searchsorted(values, start64, side="left")), int(np.searchsorted(values, end64, side="left"))


def origin_range(start: int, end: int, stride: int) -> np.ndarray:
    first = max(start, CONTEXT)
    last = end - HORIZON
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


def int64_ns(times: pd.Series) -> np.ndarray:
    return pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def assert_protocol() -> dict[str, Any]:
    protocol = read_json(HERE / "protocol.json")
    if int(protocol["context"]) != CONTEXT or int(protocol["horizon"]) != HORIZON:
        raise RuntimeError("v7 context/horizon contract changed")
    expected = protocol["datasets"]["jena"]["expected_origins"]
    if expected != {"train": 25648, "val": 371, "test_a": 365, "test_b": 365}:
        raise RuntimeError(f"Jena expected origin counts changed: {expected}")
    if int(protocol["datasets"]["jena"]["phase_period"]) != 144:
        raise RuntimeError("Jena phase period must remain 144 for 10-minute data")
    if int(protocol["datasets"]["jena"]["evaluation_stride"]) != 24:
        raise RuntimeError("Jena evaluation stride must remain 24 observations")
    return protocol


def read_jena_frame() -> pd.DataFrame:
    if digest(JENA_SOURCE) != JENA_SOURCE_SHA256:
        raise RuntimeError("Jena CSV hash does not match approved local source")
    if digest(JENA_ZIP) != JENA_ZIP_SHA256:
        raise RuntimeError("Jena ZIP hash does not match approved local official archive receipt")
    df = pd.read_csv(JENA_SOURCE, encoding="latin1")
    if list(df.columns) != ["Date Time", *JENA_COLUMNS]:
        raise RuntimeError(f"Unexpected Jena columns: {list(df.columns)}")
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
    if df["Date Time"].duplicated().any():
        raise RuntimeError("Jena timestamps contain duplicates")
    if str(df["Date Time"].iloc[0]) != "2024-01-01 00:10:00":
        raise RuntimeError(f"Unexpected Jena first timestamp: {df['Date Time'].iloc[0]}")
    if str(df["Date Time"].iloc[-1]) != "2025-01-01 00:00:00":
        raise RuntimeError(f"Unexpected Jena last timestamp: {df['Date Time'].iloc[-1]}")
    diffs = df["Date Time"].diff().dropna()
    if not (diffs == pd.Timedelta(minutes=10)).all():
        raise RuntimeError("Jena source is not an exact 10-minute grid")
    return df


def standardize_train_only(raw: np.ndarray, train_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = raw[train_slice]
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Undefined Jena TRAIN mean/std")
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x = (raw - mean[None, :]) / std[None, :]
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    return x, mean, std


def pca_basis_train_only(x: np.ndarray, train_slice: slice, latent: int) -> np.ndarray:
    if V1_LINEAR_FULLTRAIN.exists():
        module = load_module(V1_LINEAR_FULLTRAIN, "v7_v1_linear_fulltrain")
        basis = module.pca_basis(x[train_slice], latent).astype(np.float32)
    else:
        centered = x[train_slice] - x[train_slice].mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        basis = vt[:latent].T.astype(np.float32)
    if basis.shape != (x.shape[1], latent):
        raise RuntimeError(f"Unexpected PCA basis shape: {basis.shape}")
    return basis


def prepare_jena(protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["datasets"]["jena"]
    data_dir = CACHE / "data"
    trainval_path = data_dir / "jena_trainval.npz"
    test_path = data_dir / "jena_test.npz"
    contract_path = HERE / "data_contract_jena.json"
    existing = [str(p) for p in (trainval_path, test_path, contract_path) if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Jena preparation outputs: {existing}")

    df = read_jena_frame()
    times = df["Date Time"]
    raw = df[JENA_COLUMNS].to_numpy(dtype=np.float32)
    sentinel_mask = raw <= -9999.0 + 1e-6
    raw = raw.astype(np.float32, copy=True)
    raw[sentinel_mask] = np.nan
    finite = np.isfinite(raw)

    bounds = {name: split_bounds(times, start, end) for name, (start, end) in cfg["splits"].items()}
    train_start, train_end = bounds["train"]
    train_rows = np.arange(train_start, train_end, dtype=np.int64)
    train_obs = finite[train_rows].mean(axis=0)
    train_nonconstant = np.nanstd(raw[train_rows], axis=0) > 0
    if raw.shape != (52704, 21):
        raise RuntimeError(f"Unexpected Jena shape: {raw.shape}")
    if train_obs.min() < 0.95 or not bool(np.all(train_nonconstant)):
        raise RuntimeError("Approved Jena TRAIN adequacy failed")

    x, mean, std = standardize_train_only(raw, slice(train_start, train_end))
    pca_started = time.perf_counter()
    basis = pca_basis_train_only(x, slice(train_start, train_end), int(cfg["latent"]))
    pca_seconds = time.perf_counter() - pca_started

    origins = {
        "train": origin_range(*bounds["train"], stride=1),
        "val": origin_range(*bounds["val"], stride=int(cfg["evaluation_stride"])),
        "test_a": origin_range(*bounds["test_a"], stride=int(cfg["evaluation_stride"])),
        "test_b": origin_range(*bounds["test_b"], stride=int(cfg["evaluation_stride"])),
    }
    actual = {key: int(len(value)) for key, value in origins.items()}
    if actual != cfg["expected_origins"]:
        raise RuntimeError(f"Unexpected Jena origin counts: {actual}")
    if not all(np.all(value >= CONTEXT) for value in origins.values()):
        raise RuntimeError("Every Jena origin must have approved historical context")

    trainval_end = bounds["val"][1]
    trainval_sha = save_npz_new(
        trainval_path,
        x=x[:trainval_end].astype(np.float32),
        finite=finite[:trainval_end].astype(bool),
        times=int64_ns(times.iloc[:trainval_end]),
        columns=np.array(JENA_COLUMNS, dtype="U"),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        basis=basis.astype(np.float32),
        train_origins=origins["train"].astype(np.int64),
        val_origins=origins["val"].astype(np.int64),
        train_start=np.array(train_start, dtype=np.int64),
        train_end=np.array(train_end, dtype=np.int64),
        _phase_period=np.array(144, dtype=np.int64),
    )
    test_sha = save_npz_new(
        test_path,
        x=x.astype(np.float32),
        finite=finite.astype(bool),
        times=int64_ns(times),
        columns=np.array(JENA_COLUMNS, dtype="U"),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        basis=basis.astype(np.float32),
        test_a_origins=origins["test_a"].astype(np.int64),
        test_b_origins=origins["test_b"].astype(np.int64),
        train_start=np.array(train_start, dtype=np.int64),
        train_end=np.array(train_end, dtype=np.int64),
        _phase_period=np.array(144, dtype=np.int64),
    )
    contract = {
        "schema": "tsfm_peft_practical_controls_v7_data_contract_jena_v1",
        "dataset": "jena",
        "created_by": rel(Path(__file__)),
        "scope": "Approved v7 Jena data preparation. TEST values are cached for later sealed fixed evaluation; no TEST value statistics, scoring, EDA or selection are reported here.",
        "source": {
            "csv": str(JENA_SOURCE),
            "csv_size_bytes": JENA_SOURCE.stat().st_size,
            "csv_sha256": digest(JENA_SOURCE),
            "zip": str(JENA_ZIP),
            "zip_size_bytes": JENA_ZIP.stat().st_size,
            "zip_sha256": digest(JENA_ZIP),
            "official_url_recorded": "https://weather.bgc-jena.mpg.de/mpi_roof_2024.zip",
            "additional_download_bytes": 0,
            "timestamp_policy": "Use source-naive 10-minute timestamps as-is; do not invent UTC/DST conversion.",
            "license_note": "MPI-BGC official weather site records CC BY 4.0 terms; local archive hash reused without a new download.",
            "exposure_note": "Jena/MPI family and this 2024 source hash were previously exposed in local studies; v7 B is not an independent protected confirmation.",
        },
        "dataset_info": {
            "channels": int(cfg["channels"]),
            "latent_rank": int(cfg["latent"]),
            "context_length": CONTEXT,
            "horizon": HORIZON,
            "interval_minutes": int(cfg["interval_minutes"]),
            "phase_period": int(cfg["phase_period"]),
            "columns": JENA_COLUMNS,
        },
        "splits": {name: {"start": pair[0], "end": pair[1], "bounds": list(map(int, bounds[name]))} for name, pair in cfg["splits"].items()},
        "trainval": {
            "path": rel(trainval_path),
            "sha256": trainval_sha,
            "contains_rows": "[2024-01-01 00:10, 2024-09-01 00:00); no values at or after VAL end.",
            "arrays": ["x", "finite", "times", "columns", "mean", "std", "basis", "train_origins", "val_origins", "train_start", "train_end", "_phase_period"],
        },
        "test": {
            "path": rel(test_path),
            "sha256": test_sha,
            "contains_rows": "[2024-01-01 00:10, 2025-01-01 00:10) source rows; TEST values sealed for fixed evaluation only.",
            "arrays": ["x", "finite", "times", "columns", "mean", "std", "basis", "test_a_origins", "test_b_origins", "train_start", "train_end", "_phase_period"],
            "test_value_statistics_reported": False,
        },
        "origins": {
            "counts": actual,
            "lists": {key: value.astype(int).tolist() for key, value in origins.items()},
            "note": "Origin indices are relative to saved arrays. For each origin o, context is x[o-512:o] and target is x[o:o+48]. VAL/TEST context may use previously observed rows before the target split; targets stay inside split.",
        },
        "train_only_statistics": {
            "row_count": int(train_end - train_start),
            "observed_fraction_min": float(train_obs.min()),
            "observed_fraction_max": float(train_obs.max()),
            "sentinel_or_missing_cells_after_policy": int(np.size(raw[train_rows]) - finite[train_rows].sum()),
            "nonconstant_channels": int(train_nonconstant.sum()),
            "pca_rank": int(cfg["latent"]),
            "pca_seconds": float(pca_seconds),
            "statistics_used_test_values": False,
        },
        "implementation_sources": {
            "data_prepare_v7": {"path": rel(Path(__file__)), "sha256": digest(Path(__file__))},
            "runtime_v7": {"path": rel(HERE / "runtime_v7.py"), "sha256": digest(HERE / "runtime_v7.py")},
            "protocol": {"path": rel(HERE / "protocol.json"), "sha256": digest(HERE / "protocol.json")},
            "v1_linear_fulltrain": {"path": rel(V1_LINEAR_FULLTRAIN), "sha256": digest(V1_LINEAR_FULLTRAIN)} if V1_LINEAR_FULLTRAIN.exists() else None,
        },
    }
    save_json(contract_path, contract)
    return {"contract": rel(contract_path), "trainval": rel(trainval_path), "test": rel(test_path), "counts": actual}


def build_robin_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    contract_path = HERE / "data_contract_robin.json"
    if contract_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Robin data contract: {contract_path}")
    v6_contract = read_json(V6_DATA_CONTRACT)
    expected = protocol["datasets"]["robin"]["expected_origins"]
    if v6_contract["origins"]["counts"] != expected:
        raise RuntimeError("v6 Robin origin counts no longer match approved v7 reuse contract")
    for key in ("trainval", "test"):
        path = ROOT / v6_contract[key]["path"]
        if digest(path) != v6_contract[key]["sha256"]:
            raise RuntimeError(f"v6 Robin {key} cache hash changed")
    v6_eval = read_json(V6_EVALUATION)
    if v6_eval.get("status") != "complete":
        raise RuntimeError("v6 Robin evaluation must remain complete for prediction reuse")
    contract = {
        "schema": "tsfm_peft_practical_controls_v7_data_contract_robin_reuse_v1",
        "dataset": "robin",
        "created_by": rel(Path(__file__)),
        "scope": "Approved v7 Robin data contract reuses sealed v6 Robin arrays and TEST predictions for follow-up direct controls; no v6 result is modified.",
        "source": {
            "reuse_from_v6": rel(V6_DATA_CONTRACT),
            "v6_data_contract_sha256": digest(V6_DATA_CONTRACT),
            "v6_evaluation": rel(V6_EVALUATION),
            "v6_evaluation_sha256": digest(V6_EVALUATION),
            "v6_selection": rel(V6_SELECTION),
            "v6_selection_sha256": digest(V6_SELECTION),
            "v6_shared_selection": rel(V6_SHARED),
            "v6_shared_selection_sha256": digest(V6_SHARED),
            "exposure_note": "Robin TEST was exposed in v6 and remains exposed development/follow-up evidence in v7.",
        },
        "dataset_info": {
            "channels": int(protocol["datasets"]["robin"]["channels"]),
            "latent_rank": int(protocol["datasets"]["robin"]["latent"]),
            "context_length": CONTEXT,
            "horizon": HORIZON,
            "interval_minutes": int(protocol["datasets"]["robin"]["interval_minutes"]),
            "phase_period": int(protocol["datasets"]["robin"]["phase_period"]),
            "columns": v6_contract["dataset"]["selected_columns"],
        },
        "splits": v6_contract["splits"],
        "trainval": v6_contract["trainval"],
        "test": v6_contract["test"],
        "origins": v6_contract["origins"],
        "train_only_statistics": v6_contract["train_only_statistics"],
        "implementation_sources": {
            "data_prepare_v7": {"path": rel(Path(__file__)), "sha256": digest(Path(__file__))},
            "runtime_v7": {"path": rel(HERE / "runtime_v7.py"), "sha256": digest(HERE / "runtime_v7.py")},
            "protocol": {"path": rel(HERE / "protocol.json"), "sha256": digest(HERE / "protocol.json")},
        },
    }
    save_json(contract_path, contract)
    return {"contract": rel(contract_path), "trainval": contract["trainval"]["path"], "test": contract["test"]["path"], "counts": contract["origins"]["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare approved v7 data contracts and Jena arrays.")
    parser.add_argument("--dataset", choices=("robin", "jena", "all"), default="all")
    parser.add_argument("--reserve-seconds", type=float, default=120)
    args = parser.parse_args()
    protocol = assert_protocol()
    outputs: dict[str, Any] = {}
    with Job("v7_prepare_data_contracts", category="data_prepare", reserve_seconds=args.reserve_seconds) as job:
        if args.dataset in ("robin", "all"):
            outputs["robin"] = build_robin_contract(protocol)
            job.heartbeat("robin_contract_ready")
        if args.dataset in ("jena", "all"):
            outputs["jena"] = prepare_jena(protocol)
            job.heartbeat("jena_contract_ready")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
