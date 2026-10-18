"""Prepare the already separated, pre-TEST UCI prefix for the B2 extension."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


SOURCE_CAMPAIGN = "bihash_future_prediction_20261013"
SOURCE_LOCK = "3338c264ce51d82f9104a0af90fc29c97d6511aa8ec0a6385e929d99317fdb29"
L, H = 512, 48
DATASETS = {
    "uci274_sml2010": {
        "rows": 4784, "interval_seconds": 900, "rows_field": "full_grid_rows",
        "clock": "UTC source timestamps; no timezone conversion or resampling",
        "columns": (
            "Indoor_temperature_dinning_room", "Indoor_temperature_room",
            "Weather_forecast_temperature", "Carbon_dioxide_dinning_room", "Carbon_dioxide_room",
            "Relative_humidity_dinning_room", "Relative_humidity_room", "Lighting_dinning_room",
            "Lighting_room", "Rain_last_15min_fraction", "Sun_dusk", "Wind",
            "Sun_light_west_facade", "Sun_light_east_facade", "Sun_light_south_facade",
            "Sun_irradiance", "Enthalpic_motor_1", "Enthalpic_motor_2", "Enthalpic_motor_turbo",
            "Outdoor_temperature", "Outdoor_relative_humidity",
        ),
    },
    "uci235_household_hourly": {
        "rows": 34587, "interval_seconds": 3600, "rows_field": "hourly_rows",
        "clock": "Naive source end-of-hour labels preserved as nominal nanoseconds; timezone unspecified",
        "columns": (
            "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def split_ranges(n: int) -> dict[str, list[int]]:
    return {"A_fit": [0, n // 2], "A_stop": [n // 2, n * 3 // 5],
            "A60": [0, n * 3 // 5], "M_cal": [n * 3 // 5, n * 7 // 10],
            "V_select": [n * 7 // 10, n * 4 // 5], "T_final": [n * 4 // 5, n]}


def legal_origins(bounds: list[int]) -> np.ndarray:
    return np.arange(max(L, bounds[0]), bounds[1] - H + 1, H, dtype=np.int64)


def _headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy") or "/" in name or name[:-4] in result:
                raise ValueError("Invalid prefix header member")
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("Unsupported prefix header version")
            if dtype.hasobject:
                raise ValueError("Object prefix header refused")
            result[name[:-4]] = (shape, dtype)
    return result


def _admit_headers(path: Path, rows: int, channels: int, *, prepared: bool) -> None:
    headers = _headers(path)
    observed_key = "observed" if prepared else "mask"
    required = {"physical", observed_key, "times", "columns"}
    allowed = required if prepared else required | {"mean", "std", "origins_fit", "origins_tune", "basis", "pca_mean"}
    if not required <= headers.keys() or not headers.keys() <= allowed:
        raise ValueError("Unexpected prefix header keys")
    if headers["physical"] != ((rows, channels), np.dtype("float32")):
        raise ValueError("Wrong physical prefix header or forbidden tail")
    if headers[observed_key] != ((rows, channels), np.dtype("bool")):
        raise ValueError("Wrong observation prefix header")
    shape, dtype = headers["times"]
    if shape != (rows,) or (dtype != np.dtype("int64") if prepared else dtype.kind != "U"):
        raise ValueError("Wrong timestamp prefix header")
    shape, dtype = headers["columns"]
    if shape != (channels,) or dtype.kind != "U":
        raise ValueError("Wrong column prefix header")


def _source(asset_root: Path, dataset: str) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    spec = DATASETS[dataset]
    root = Path(asset_root).resolve()
    folder = root / ".cache" / SOURCE_CAMPAIGN / dataset / SOURCE_LOCK
    contract_path = folder / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    n = spec["rows"]
    expected_bounds = {"fit": [0, n * 3 // 5], "tune": [n * 3 // 5, n * 4 // 5], "eval": [n * 4 // 5, n]}
    if (contract.get("dataset") != dataset or contract.get("L") != L or contract.get("H") != H
            or contract.get("C") != len(spec["columns"]) or contract.get("columns") != list(spec["columns"])
            or contract.get("parser_metadata", {}).get(spec["rows_field"]) != n
            or contract.get("bounds") != expected_bounds
            or contract.get("row_split") != "floor 60/20/20 by post-parser rows"
            or contract.get("eval_separation") != "FIT_TUNE, EVAL past input, and EVAL future target are separate payloads"):
        raise ValueError("Source contract is not the registered separated pre-TEST UCI prefix")
    payload_path = folder / "FIT_TUNE_payload.npz"
    declared = Path(contract["payloads"]["fit_tune"]["path"])
    declared = declared if declared.is_absolute() else root / declared
    if declared.resolve() != payload_path.resolve():
        raise ValueError("FIT_TUNE path cannot point outside the admitted prefix")
    _admit_headers(payload_path, n * 4 // 5, len(spec["columns"]), prepared=False)
    record = contract["payloads"]["fit_tune"]
    if payload_path.stat().st_size != record["bytes"] or sha256_file(payload_path) != record["sha256"]:
        raise ValueError("Source prefix fingerprint mismatch")
    return contract, contract_path, payload_path, spec


def train_structure(physical: np.ndarray, observed: np.ndarray, columns: list[str], end: int) -> dict[str, Any]:
    values = physical[:end].astype(np.float64)
    mask = observed[:end]
    counts = mask.sum(axis=0, dtype=np.int64)
    sums = np.where(mask, values, 0).sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(mask, values - mean, 0)
    variance = np.divide((centered * centered).sum(axis=0), counts, out=np.zeros_like(mean), where=counts > 0)
    std = np.sqrt(variance)
    constant = (std <= 1e-6) & (counts > 0)
    normalized = centered / np.where(std > 1e-6, std, 1)
    singular = np.linalg.svd(normalized, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    tolerance = max(normalized.shape) * np.finfo(np.float64).eps * float(singular[0])
    choices = sorted({max(1, int(np.ceil(len(columns) / 4))), max(1, int(np.ceil(len(columns) / 2)))})
    return {
        "scope": "A_fit_only", "rows": end, "columns": len(columns),
        "observed_counts": counts.tolist(), "missing_fraction_per_channel": (1 - counts / end).tolist(),
        "all_channel_missing_rows": int((~mask.any(axis=1)).sum()),
        "unobserved_columns": [c for c, count in zip(columns, counts) if count == 0],
        "constant_columns": [c for c, flag in zip(columns, constant) if flag],
        "numerical_rank": int((singular > tolerance).sum()),
        "participation_effective_rank": float(total * total / np.dot(energy, energy)) if total else 0,
        "standardized_energy_fraction_by_k": {str(k): float(energy[:k].sum() / total) if total else 0 for k in choices},
        "rank_definition": "Observed A_fit channel centering/scaling; missing cells set to A_fit mean solely for this structure diagnostic; no interpolation or forecast scoring",
        "selection_information": "TRAIN structure only; no target error or A_stop/M_cal/V_select/T_final value enters this diagnostic",
    }


def _validate_arrays(physical: np.ndarray, observed: np.ndarray, times: np.ndarray, columns: np.ndarray,
                     rows: int, spec: dict[str, Any]) -> None:
    if physical.shape != (rows, len(spec["columns"])) or observed.shape != physical.shape:
        raise ValueError("Prefix array shape differs from admitted contract")
    if not np.array_equal(observed, np.isfinite(physical)):
        raise ValueError("Physical missingness differs from original observation mask")
    if columns.tolist() != list(spec["columns"]):
        raise ValueError("Source sensor columns or order differ from registered contract")
    if times.shape != (rows,) or not np.all(np.diff(times) == spec["interval_seconds"] * 10**9):
        raise ValueError("Source clock is not the registered regular interval")


def prepare_dataset(asset_root: Path, dataset: str, output_root: Path) -> dict[str, Any]:
    contract, contract_path, payload_path, spec = _source(Path(asset_root), dataset)
    folder = Path(output_root) / dataset
    if folder.exists():
        raise FileExistsError(f"Prepared data already exists: {folder}")
    with np.load(payload_path, allow_pickle=False) as payload:
        physical = payload["physical"].copy()
        observed = payload["mask"].copy()
        original_labels = payload["times"].copy()
        columns = payload["columns"].copy()
    labels = [str(label).removesuffix("Z") for label in original_labels]
    times = np.asarray(labels, dtype="datetime64[ns]").astype(np.int64)
    rows = spec["rows"] * 4 // 5
    _validate_arrays(physical, observed, times, columns, rows, spec)
    ranges = split_ranges(spec["rows"])
    structure = train_structure(physical, observed, columns.tolist(), ranges["A_fit"][1])
    metadata = {
        "schema_version": 1, "status": "READY", "dataset": dataset, "C": len(columns), "L": L, "H": H,
        "interval_seconds": spec["interval_seconds"], "clock": spec["clock"],
        "original_rows": spec["rows"], "prefix_rows": rows, "prefix_only_approved": True,
        "columns": columns.tolist(), "split_ranges": ranges,
        "origins": {name: legal_origins(ranges[name]).tolist() for name in ("A_fit", "A_stop", "A60", "V_select")},
        "origin_rule": "Stride H48; full future target lies within named role; origin at least L512; past history may cross earlier role boundary",
        "train_structure": structure,
        "source_contract_sha256": sha256_file(contract_path),
        "source_payload_sha256": contract["payloads"]["fit_tune"]["sha256"],
        "source_reference": f"asset://.cache/{SOURCE_CAMPAIGN}/{dataset}/{SOURCE_LOCK}/FIT_TUNE_payload.npz",
        "source_contract_reference": f"asset://.cache/{SOURCE_CAMPAIGN}/{dataset}/{SOURCE_LOCK}/contract.json",
        "source_arrays": {"physical": array_sha256(physical), "observed": array_sha256(observed),
                          "original_time_labels": array_sha256(original_labels), "columns": array_sha256(columns)},
        "physical_units": "Original physical FP32 source values preserved directly; original mean/std and normalized values are not consumed",
        "normalization": "Consumer must fit a fresh A_fit scaler during search and fresh A60 scaler during refit; this file has no scaler",
        "missingness": "Original NaN/mask preserved, no target filling; consumer resets history forward fill to its permitted TRAIN mean at each L512 window start",
        "protected_numeric_access": {"raw_source": False, "old_EVAL_input": False, "old_EVAL_target": False, "T_final": False},
        "development_exposure": "Dataset was used by a different earlier method; these are development data, not a new independently untouched domain confirmation",
        "source_sha256": sha256_file(Path(__file__)),
    }
    if structure["unobserved_columns"] or any(not metadata["origins"][name] for name in metadata["origins"]):
        metadata["status"] = "NOT_READY_TRAIN_STRUCTURE"
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "PREFIX_PHYSICAL.npz"
    with output.open("xb") as handle:
        np.savez_compressed(handle, physical=physical, observed=observed, times=times, columns=columns)
    metadata["prefix_sha256"] = sha256_file(output)
    metadata["prefix_bytes"] = output.stat().st_size
    with (folder / "DATA_DIAGNOSTIC.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return metadata


def load_prepared(dataset_dir: Path) -> dict[str, Any]:
    folder = Path(dataset_dir)
    metadata = json.loads((folder / "DATA_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    dataset = metadata["dataset"]
    if dataset not in DATASETS or metadata.get("prefix_only_approved") is not True:
        raise ValueError("Prepared dataset is not an approved prefix")
    spec = DATASETS[dataset]
    rows = spec["rows"] * 4 // 5
    if metadata.get("prefix_rows") != rows or metadata.get("split_ranges") != split_ranges(spec["rows"]):
        raise ValueError("Prepared prefix ranges differ from contract")
    path = folder / "PREFIX_PHYSICAL.npz"
    _admit_headers(path, rows, len(spec["columns"]), prepared=True)
    if sha256_file(path) != metadata["prefix_sha256"]:
        raise ValueError("Prepared prefix fingerprint mismatch")
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in ("physical", "observed", "times", "columns")}
    _validate_arrays(**result, rows=rows, spec=spec)
    result["metadata"] = metadata
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    if args.public_output is not None and args.public_output.exists():
        raise FileExistsError(args.public_output)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = [prepare_dataset(args.asset_root, dataset, args.output_root) for dataset in names]
    report = {"schema_version": 1, "status": "READY" if all(r["status"] == "READY" for r in results) else "NOT_READY",
              "diagnostic_scope": "A_fit TRAIN structure only; no forecast errors or candidate result selection",
              "datasets": results, "source_sha256": sha256_file(Path(__file__))}
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        with args.public_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": report["status"], "datasets": [{"dataset": r["dataset"], "status": r["status"],
                      "prefix_rows": r["prefix_rows"], "C": r["C"], "train_structure": r["train_structure"]} for r in results]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
