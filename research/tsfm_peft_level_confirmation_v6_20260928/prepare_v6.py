from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from runtime_v6 import CACHE, HERE, ROOT, Job, digest, save_json


CONTEXT = 512
HORIZON = 48
EVAL_STRIDE = 24
SITE_ID = "Robin"
CHANNELS = 17
LATENT = 5

SOURCE_REVISION = "9b97ccbe90096aff42ed4fd6493bf7ae692d7118"
ELECTRICITY_SHA256 = "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"
METADATA_SHA256 = "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"

ELECTRICITY_PATH = (
    Path("E:/CODING/proj/mltimeseries")
    / "data_external/bdg2_coarse_supervision_v1/raw/electricity.csv"
)
METADATA_PATH = (
    Path("E:/CODING/proj/mltimeseries")
    / "data_external/bdg2_coarse_supervision_v1/raw/metadata.csv"
)
V1_DATA_PREPARE = ROOT / "research/tsfm_peft_development_20260926/data_prepare.py"
V1_LINEAR_FULLTRAIN = ROOT / "research/tsfm_peft_development_20260926/linear_fulltrain.py"

SPLITS = {
    "precontext": ("2016-01-01T00:00:00", "2016-01-23T00:00:00"),
    "train": ("2016-01-23T00:00:00", "2016-04-15T00:00:00"),
    "val": ("2016-04-17T00:00:00", "2016-05-18T00:00:00"),
    "test_a": ("2016-05-20T00:00:00", "2016-06-30T00:00:00"),
    "test_b": ("2016-06-30T00:00:00", "2016-08-10T00:00:00"),
}

EXPECTED_SELECTED = [
    "Robin_office_Addie",
    "Robin_office_Adolph",
    "Robin_office_Antonina",
    "Robin_office_Dina",
    "Robin_office_Donald",
    "Robin_office_Erma",
    "Robin_office_Gayle",
    "Robin_office_Lindsay",
    "Robin_office_Maryann",
    "Robin_office_Sammie",
    "Robin_office_Saul",
    "Robin_office_Serena",
    "Robin_office_Shirlene",
    "Robin_office_Soledad",
    "Robin_office_Victor",
    "Robin_office_Wai",
    "Robin_office_Zelma",
]


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_protocol() -> dict[str, Any]:
    return json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def assert_protocol(protocol: dict[str, Any]) -> None:
    if protocol["selected_columns"] != EXPECTED_SELECTED:
        raise RuntimeError("protocol selected_columns do not match the approved Robin 17 IDs")
    if int(protocol["channels"]) != CHANNELS or int(protocol["latent"]) != LATENT:
        raise RuntimeError("protocol channel/latent contract changed")
    if int(protocol["context"]) != CONTEXT or int(protocol["horizon"]) != HORIZON:
        raise RuntimeError("protocol context/horizon contract changed")
    expected = {"train": 1945, "val": 30, "test_a": 40, "test_b": 40}
    if {k: int(v) for k, v in protocol["expected_origins"].items()} != expected:
        raise RuntimeError("protocol expected origin counts changed")


def save_npz_new(path: Path, **arrays: Any) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing cache file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    return digest(path)


def assert_no_existing_outputs(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing v6 preparation outputs: {existing}")


def timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def split_bounds(times: pd.Series, name: str) -> tuple[int, int]:
    start, end = SPLITS[name]
    values = times.to_numpy()
    return (
        int(np.searchsorted(values, np.datetime64(timestamp(start)), side="left")),
        int(np.searchsorted(values, np.datetime64(timestamp(end)), side="left")),
    )


def origin_range(start: int, end: int, stride: int) -> np.ndarray:
    first = max(start, CONTEXT)
    last_exclusive = end - HORIZON + 1
    if last_exclusive <= first:
        return np.array([], dtype=np.int64)
    return np.arange(first, last_exclusive, stride, dtype=np.int64)


def origins_for(times: pd.Series, name: str, stride: int) -> np.ndarray:
    start, end = split_bounds(times, name)
    return origin_range(start, end, stride=stride)


def int64_ns(times: pd.Series) -> np.ndarray:
    return pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def valid_electricity_flag(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    unexpected = sorted(value for value in text.dropna().unique().tolist() if value not in ("", "Yes"))
    if unexpected:
        raise RuntimeError(f"Unexpected metadata electricity values: {unexpected}")
    return series.notna() & text.notna() & (text == "Yes")


def finite_fraction(raw: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.isfinite(raw[rows]).mean(axis=0).astype(np.float64)


def nonconstant(raw: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return (np.nanstd(raw[rows], axis=0) > 0).astype(bool)


def read_robin_metadata() -> tuple[list[str], dict[str, Any]]:
    metadata = pd.read_csv(METADATA_PATH)
    office = metadata[
        (metadata["site_id"].astype(str) == SITE_ID)
        & (metadata["primaryspaceusage"].astype(str) == "Office")
        & valid_electricity_flag(metadata["electricity"])
    ].copy()
    office_ids = sorted(office["building_id"].astype(str).tolist())
    if office_ids != EXPECTED_SELECTED:
        raise RuntimeError(f"Robin Office electricity IDs changed: {office_ids}")
    timezones = sorted(str(value) for value in office["timezone"].dropna().unique())
    if timezones != ["Europe/London"]:
        raise RuntimeError(f"Unexpected Robin timezone metadata: {timezones}")
    return office_ids, {
        "rows": int(len(office_ids)),
        "timezone_values": timezones,
        "selection_basis": "site_id == Robin, primaryspaceusage == Office, electricity non-empty; building_id sorted",
    }


def read_prefix(office_ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(ELECTRICITY_PATH, usecols=["timestamp", *office_ids])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    duplicates = df["timestamp"].duplicated(keep=False)
    if bool(duplicates.any()):
        sample = df.loc[duplicates, "timestamp"].astype(str).head(10).tolist()
        raise RuntimeError(f"Duplicate raw timestamps would be hidden by sorting: {sample}")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    start = timestamp(SPLITS["precontext"][0])
    end = timestamp(SPLITS["test_b"][1])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    if str(df["timestamp"].iloc[0]) != "2016-01-01 00:00:00":
        raise RuntimeError(f"Unexpected first timestamp: {df['timestamp'].iloc[0]}")
    if str(df["timestamp"].iloc[-1]) != "2016-08-09 23:00:00":
        raise RuntimeError(f"Unexpected last timestamp: {df['timestamp'].iloc[-1]}")
    diffs = df["timestamp"].diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise RuntimeError("Robin prefix is not an exact local-naive hourly grid")
    return df


def compute_selection(df: pd.DataFrame, office_ids: list[str]) -> tuple[list[str], dict[str, Any]]:
    raw_all = df[office_ids].to_numpy(dtype=np.float32)
    times = df["timestamp"]
    train_start, train_end = split_bounds(times, "train")
    train_rows = np.arange(train_start, train_end, dtype=np.int64)
    if len(train_rows) != 1992:
        raise RuntimeError(f"Expected 1992 TRAIN rows, got {len(train_rows)}")

    availability = finite_fraction(raw_all, train_rows)
    variability = nonconstant(raw_all, train_rows)
    eligible_mask = (availability >= 0.95) & variability
    eligible = [name for name, keep in zip(office_ids, eligible_mask, strict=True) if bool(keep)]
    selected = sorted(eligible)[:CHANNELS]
    if selected != EXPECTED_SELECTED:
        raise RuntimeError(f"Robin selected ID set changed: {selected}")

    failed = [
        {
            "id": name,
            "train_observed_fraction": float(frac),
            "train_nonconstant": bool(varies),
            "reason": "train_observed_fraction_lt_0.95" if frac < 0.95 else "train_constant",
        }
        for name, frac, varies in zip(office_ids, availability, variability, strict=True)
        if not (frac >= 0.95 and bool(varies))
    ]
    if failed:
        raise RuntimeError(f"Approved Robin channels must all be TRAIN-eligible: {failed}")

    return selected, {
        "site_id": SITE_ID,
        "rule": "Robin was fixed during planning by site ID order, bounded prior exposure search, TRAIN observed fraction >= 0.95, nonconstant, and all 17 eligible Office electricity IDs.",
        "total_office_ids": len(office_ids),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "latent_rank": LATENT,
        "all_office_ids": office_ids,
        "eligible_ids": eligible,
        "selected_ids": selected,
        "failed_ids": failed,
        "train_observed_fraction_by_id": {
            name: float(value) for name, value in zip(office_ids, availability, strict=True)
        },
        "train_nonconstant_by_id": {
            name: bool(value) for name, value in zip(office_ids, variability, strict=True)
        },
        "selected_observed_fraction_range": [float(availability.min()), float(availability.max())],
        "all_selected_nonconstant": True,
        "selection_used_model_performance": False,
    }


def make_arrays(df: pd.DataFrame, selected: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_prepare = load_module(V1_DATA_PREPARE, "v6_v1_data_prepare")
    linear_fulltrain = load_module(V1_LINEAR_FULLTRAIN, "v6_v1_linear_fulltrain")

    times = df["timestamp"]
    raw = df[selected].to_numpy(dtype=np.float32)
    finite = np.isfinite(raw)
    train_start, train_end = split_bounds(times, "train")
    val_start, val_end = split_bounds(times, "val")
    test_a_start, test_a_end = split_bounds(times, "test_a")
    test_b_start, test_b_end = split_bounds(times, "test_b")

    standardize_started = time.perf_counter()
    x, mean, std = data_prepare.standardize(raw, slice(train_start, train_end))
    standardize_seconds = time.perf_counter() - standardize_started
    pca_started = time.perf_counter()
    basis = linear_fulltrain.pca_basis(x[train_start:train_end], LATENT).astype(np.float32)
    pca_seconds = time.perf_counter() - pca_started
    if basis.shape != (CHANNELS, LATENT):
        raise RuntimeError(f"Unexpected PCA basis shape: {basis.shape}")

    train_origins = origins_for(times, "train", stride=1)
    val_origins = origins_for(times, "val", stride=EVAL_STRIDE)
    test_a_origins = origins_for(times, "test_a", stride=EVAL_STRIDE)
    test_b_origins = origins_for(times, "test_b", stride=EVAL_STRIDE)
    expected = (1945, 30, 40, 40)
    actual = (len(train_origins), len(val_origins), len(test_a_origins), len(test_b_origins))
    if actual != expected:
        raise RuntimeError(f"Unexpected origin counts: {actual}")

    trainval_end = val_end
    trainval = {
        "x": x[:trainval_end].astype(np.float32),
        "finite": finite[:trainval_end].astype(bool),
        "times": int64_ns(times.iloc[:trainval_end]),
        "columns": np.array(selected, dtype="U"),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "basis": basis.astype(np.float32),
        "train_origins": train_origins.astype(np.int64),
        "val_origins": val_origins.astype(np.int64),
        "train_start": np.array(train_start, dtype=np.int64),
        "train_end": np.array(train_end, dtype=np.int64),
    }
    test = {
        "x": x.astype(np.float32),
        "finite": finite.astype(bool),
        "times": int64_ns(times),
        "columns": np.array(selected, dtype="U"),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "basis": basis.astype(np.float32),
        "test_a_origins": test_a_origins.astype(np.int64),
        "test_b_origins": test_b_origins.astype(np.int64),
        "train_start": np.array(train_start, dtype=np.int64),
        "train_end": np.array(train_end, dtype=np.int64),
    }
    meta = {
        "bounds": {
            "train": [train_start, train_end],
            "val": [val_start, val_end],
            "test_a": [test_a_start, test_a_end],
            "test_b": [test_b_start, test_b_end],
        },
        "origin_counts": {
            "train": int(len(train_origins)),
            "val": int(len(val_origins)),
            "test_a": int(len(test_a_origins)),
            "test_b": int(len(test_b_origins)),
        },
        "origin_lists": {
            "train": train_origins.tolist(),
            "val": val_origins.tolist(),
            "test_a": test_a_origins.tolist(),
            "test_b": test_b_origins.tolist(),
        },
        "train_only_statistics": {
            "period": list(SPLITS["train"]),
            "row_count": int(train_end - train_start),
            "mean_std_fit_rows": int(train_end - train_start),
            "pca_fit_rows": int(train_end - train_start),
            "pca_rank": LATENT,
            "pca_source": rel(V1_LINEAR_FULLTRAIN),
            "pca_source_sha256": digest(V1_LINEAR_FULLTRAIN),
            "standardize_source": rel(V1_DATA_PREPARE),
            "standardize_source_sha256": digest(V1_DATA_PREPARE),
            "standardize_seconds": float(standardize_seconds),
            "pca_seconds": float(pca_seconds),
            "import_policy": "Explicit path aliases only; reused v1 data_prepare.standardize and linear_fulltrain.pca_basis without importing TSFM/model runtime modules.",
        },
    }
    return {"trainval": trainval, "test": test}, meta


def build_contract(
    office_meta: dict[str, Any],
    eligibility: dict[str, Any],
    array_meta: dict[str, Any],
    trainval_path: Path,
    trainval_sha: str,
    test_path: Path,
    test_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "tsfm_peft_level_confirmation_v6_data_contract_v1",
        "created_by": rel(Path(__file__)),
        "scope": "Approved v6 data preparation only. TEST values are packaged for later fixed evaluation but no TEST value statistics, scoring, EDA, or selection are reported here.",
        "source": {
            "electricity_csv": str(ELECTRICITY_PATH),
            "electricity_size_bytes": ELECTRICITY_PATH.stat().st_size,
            "electricity_sha256": digest(ELECTRICITY_PATH),
            "metadata_csv": str(METADATA_PATH),
            "metadata_size_bytes": METADATA_PATH.stat().st_size,
            "metadata_sha256": digest(METADATA_PATH),
            "expected_electricity_sha256": ELECTRICITY_SHA256,
            "expected_metadata_sha256": METADATA_SHA256,
            "source_revision": SOURCE_REVISION,
            "additional_download_bytes": 0,
            "timestamp_policy": "Use provided local-naive hourly timestamp grid as-is; no UTC conversion or DST reconstruction.",
            "unit_note": "BDG2 electricity meter reading; official paper describes hourly energy in kWh.",
        },
        "dataset": {
            "site_id": SITE_ID,
            "primaryspaceusage": "Office",
            "channels": CHANNELS,
            "latent_rank": LATENT,
            "context_length": CONTEXT,
            "horizon": HORIZON,
            "metadata": office_meta,
            "selected_columns": EXPECTED_SELECTED,
            "eligibility_train": eligibility,
        },
        "splits": {name: {"start": start, "end": end} for name, (start, end) in SPLITS.items()},
        "trainval": {
            "path": rel(trainval_path),
            "sha256": trainval_sha,
            "contains_rows": "[2016-01-01T00:00:00, 2016-05-18T00:00:00); no values at or after VAL end.",
            "arrays": [
                "x",
                "finite",
                "times",
                "columns",
                "mean",
                "std",
                "basis",
                "train_origins",
                "val_origins",
                "train_start",
                "train_end",
            ],
        },
        "test": {
            "path": rel(test_path),
            "sha256": test_sha,
            "contains_rows": "[2016-01-01T00:00:00, 2016-08-10T00:00:00).",
            "arrays": [
                "x",
                "finite",
                "times",
                "columns",
                "mean",
                "std",
                "basis",
                "test_a_origins",
                "test_b_origins",
                "train_start",
                "train_end",
            ],
            "test_value_statistics_reported": False,
        },
        "origins": {
            "counts": array_meta["origin_counts"],
            "lists": array_meta["origin_lists"],
            "note": "Origin indices are relative to the saved arrays. For each origin o, context is x[o-512:o] and target is x[o:o+48].",
        },
        "train_only_statistics": array_meta["train_only_statistics"],
        "implementation_sources": {
            "prepare_v6": {"path": rel(Path(__file__)), "sha256": digest(Path(__file__))},
            "runtime_v6": {"path": rel(HERE / "runtime_v6.py"), "sha256": digest(HERE / "runtime_v6.py")},
            "protocol": {"path": rel(HERE / "protocol.json"), "sha256": digest(HERE / "protocol.json")},
            "v1_data_prepare": {"path": rel(V1_DATA_PREPARE), "sha256": digest(V1_DATA_PREPARE)},
            "v1_linear_fulltrain": {"path": rel(V1_LINEAR_FULLTRAIN), "sha256": digest(V1_LINEAR_FULLTRAIN)},
        },
    }


def main() -> None:
    protocol = read_protocol()
    assert_protocol(protocol)
    if digest(ELECTRICITY_PATH) != ELECTRICITY_SHA256:
        raise RuntimeError("BDG2 electricity.csv hash does not match the approved source")
    if digest(METADATA_PATH) != METADATA_SHA256:
        raise RuntimeError("BDG2 metadata.csv hash does not match the approved source")

    data_dir = CACHE / "data"
    trainval_path = data_dir / "trainval.npz"
    test_path = data_dir / "test.npz"
    contract_path = HERE / "data_contract.json"
    assert_no_existing_outputs([trainval_path, test_path, contract_path])

    with Job(
        "v6_prepare_robin17_data",
        category="data_prepare",
        reserve_seconds=60,
        metadata={"site": SITE_ID, "channels": CHANNELS, "latent_rank": LATENT},
    ) as job:
        office_ids, office_meta = read_robin_metadata()
        job.heartbeat("metadata_loaded")
        df = read_prefix(office_ids)
        job.heartbeat("source_prefix_loaded")
        selected, eligibility = compute_selection(df, office_ids)
        arrays, array_meta = make_arrays(df, selected)
        job.heartbeat("arrays_prepared")

        trainval_sha = save_npz_new(trainval_path, **arrays["trainval"])
        test_sha = save_npz_new(test_path, **arrays["test"])
        contract = build_contract(
            office_meta=office_meta,
            eligibility=eligibility,
            array_meta=array_meta,
            trainval_path=trainval_path,
            trainval_sha=trainval_sha,
            test_path=test_path,
            test_sha=test_sha,
        )
        save_json(contract_path, contract)
        job.heartbeat("complete")

    print(json.dumps({"contract": rel(contract_path), "trainval": rel(trainval_path), "test": rel(test_path)}, indent=2))


if __name__ == "__main__":
    main()
