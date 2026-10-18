"""Bounded physical-prefix preparation for the registered dataset search."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path

import numpy as np

from research.b2_dataset_extension_20261018_v1 import data as previous


split_ranges = previous.split_ranges
sha256_file = previous.sha256_file
L, H = 512, 48
ETT_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
AIR_COLUMNS = ("CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
               "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH")
APPLIANCE_COLUMNS = ("Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
    "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
    "RH_out", "Windspeed", "Visibility", "Tdewpoint")
DATASETS = {
    "uci360_air_quality": dict(rows=9357, columns=AIR_COLUMNS, interval_seconds=3600, delimiter=";",
        header=("Date", "Time", *AIR_COLUMNS, "", ""), first_timestamp="2004-03-10 18:00:00",
        last_timestamp="2005-04-04 14:00:00", clock="Source local-naive hourly labels; no timezone conversion"),
    "ETTh2": dict(rows=17420, columns=ETT_COLUMNS, interval_seconds=3600, delimiter=",",
        header=("date", *ETT_COLUMNS), first_timestamp="2016-07-01 00:00:00",
        last_timestamp="2018-06-26 19:00:00", clock="Source naive hourly timestamps; timezone unspecified"),
    "appliances": dict(rows=19735, columns=APPLIANCE_COLUMNS, interval_seconds=600,
        clock="Source naive 10-minute timestamps; timezone unspecified"),
    "ETTh1": dict(rows=17420, columns=ETT_COLUMNS, interval_seconds=3600, delimiter=",",
        header=("date", *ETT_COLUMNS), first_timestamp="2016-07-01 00:00:00",
        last_timestamp="2018-06-26 19:00:00", clock="Source naive hourly timestamps; timezone unspecified"),
}
DEFAULT_WAVE = ("uci360_air_quality", "ETTh2", "appliances")


def source_locations(asset_root, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root = Path(asset_root).resolve()
    siblings = root.parent
    if dataset == "appliances":
        base = root / ".cache/smallk_predictive_alignment_20261013/raw_dataset/appliances/2fccf354445d886e"
        return dict(source=base / "fit_tune_payload.npz", metadata=[base / "contract.json"])
    if dataset == "uci360_air_quality":
        base = siblings / "mltimeseries/data_external/uci_air_quality"
        return dict(source=base / "raw/AirQualityUCI.csv", metadata=[base / "source_qc_manifest.json", base / "acquisition_manifest.json"])
    audit_name = ("branch_mixture_lora_v1_20260922/DATA_AUDIT.json" if dataset == "ETTh2"
                  else "rollout_uncertainty_peft_v1_20260921/DATA_AND_SPLIT_AUDIT.json")
    return dict(source=siblings / "covariate-trust-pilot/data/ett_small" / (dataset + ".csv"),
                metadata=[siblings / "tsfm-peft-method-screen/results" / audit_name,
                          siblings / "covariate-trust-pilot/data/external_datasets_manifest.json"])


def _metadata(path, asset_root):
    raw = Path(path).read_bytes()
    relative = Path(path).resolve().relative_to(Path(asset_root).resolve().parent).as_posix()
    return json.loads(raw), dict(reference="asset-parent://" + relative,
                                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _admit_csv(asset_root, dataset, locations, spec):
    first, first_binding = _metadata(locations["metadata"][0], asset_root)
    second, second_binding = _metadata(locations["metadata"][1], asset_root)
    if dataset == "uci360_air_quality":
        source = first["datasets"]["uci_air_quality"]
        if (source.get("rows") != spec["rows"] or source.get("columns") != list(spec["header"][:-2])
                or source.get("first_timestamp") != spec["first_timestamp"]
                or source.get("last_timestamp") != spec["last_timestamp"]
                or source.get("timestamp_parse_errors") != 0 or source.get("duplicate_timestamps") != 0
                or source.get("hour_gap_seconds_distribution") != {"3600": spec["rows"] - 1}
                or source.get("full_hour_grid_missing_between_first_last") != 0 or second.get("dataset_id") != 360):
            raise ValueError("Air Quality source metadata differs from registered N/schema/clock")
        records = [r for r in second["current_raw_files"] if r["path"].replace("\\", "/").endswith("/raw/AirQualityUCI.csv")]
    else:
        source = first["source"] if dataset == "ETTh2" else first["sources"]["ETTh1"]
        schema, clock = source["schema"], source["time_axis"]
        if (schema.get("n_rows") != spec["rows"] or schema.get("header") != list(spec["header"])
                or schema.get("target_columns") != list(spec["columns"])
                or clock.get("first_timestamp") != spec["first_timestamp"] or clock.get("last_timestamp") != spec["last_timestamp"]
                or clock.get("all_hourly_diffs") is not True or clock.get("duplicate_timestamps") != 0):
            raise ValueError("ETT source metadata differs from registered N/schema/clock")
        records = [r for r in second["datasets"]["ett_small"]["files"] if r["name"] == dataset + ".csv"]
        if len(records) != 1 or records[0]["sha256"] != source["raw_sha256"] or records[0]["bytes"] != source["raw_bytes"]:
            raise ValueError("Two independent ETT metadata records disagree")
    if len(records) != 1 or locations["source"].stat().st_size != records[0]["bytes"]:
        raise ValueError("Source size differs from registered metadata")
    return [first_binding, second_binding], records[0]


class _ConsumedLines:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.handle.readline()
        if not value:
            raise StopIteration
        self.digest.update(value)
        self.bytes += len(value)
        return value.decode("utf-8-sig")


def _csv_prefix(path, dataset, spec):
    count = spec["rows"] * 4 // 5
    physical = np.empty((count, len(spec["columns"])), dtype=np.float32)
    times = np.empty(count, dtype=np.int64)
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with opener(path, "rb") as handle:
        lines = _ConsumedLines(handle)
        reader = csv.reader(lines, delimiter=spec["delimiter"])
        if next(reader, None) != list(spec["header"]):
            raise ValueError("Source header differs from registered physical channels")
        consumed = 0
        for index, row in enumerate(islice(reader, count)):
            if len(row) != len(spec["header"]):
                raise ValueError("CSV prefix record width differs from registered header")
            if dataset == "uci360_air_quality":
                stamp = datetime.strptime(row[0].strip() + " " + row[1].strip(), "%d/%m/%Y %H.%M.%S")
                if any(value.strip() for value in row[-2:]):
                    raise ValueError("Unexpected nonempty Air Quality trailing columns")
                tokens = row[2:-2]
            else:
                stamp = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                tokens = row[1:]
            times[index] = np.datetime64(stamp, "ns").astype(np.int64)
            for channel, value in enumerate(tokens):
                token = value.strip().replace(",", ".")
                number = float(token) if token else np.nan
                if not np.isfinite(number) or (dataset == "uci360_air_quality" and number <= -200):
                    number = np.nan
                physical[index, channel] = number
            consumed += 1
        if consumed != count:
            raise ValueError("Source ended before the admitted numeric prefix")
        provenance = dict(source_prefix_consumed_records=consumed, source_prefix_consumed_bytes=lines.bytes,
            source_prefix_consumed_sha256=lines.digest.hexdigest(),
            source_prefix_hash_scope="Only header and exactly floor(.8N) CSV records consumed by the parser; not a full-source hash",
            source_prefix_hash_encoding="Decompressed consumed CSV bytes" if path.suffix.lower() == ".gz" else "Consumed original CSV bytes")
    if times[0] != np.datetime64(spec["first_timestamp"], "ns").astype(np.int64):
        raise ValueError("Prefix first clock differs from source metadata")
    return physical, np.isfinite(physical), times, np.asarray(spec["columns"], dtype="U64"), provenance


def _appliance_prefix(asset_root, locations, spec):
    contract, binding = _metadata(locations["metadata"][0], asset_root)
    n = spec["rows"]
    source = contract.get("source", {}).get("parse", {})
    if (contract.get("dataset") != "appliances" or source.get("rows") != n
            or source.get("columns") != list(spec["columns"])
            or contract.get("spec", {}).get("channels") != list(spec["columns"])
            or contract.get("spec", {}).get("excluded_columns") != ["date", "rv1", "rv2"]
            or contract.get("spec", {}).get("interval_seconds") != spec["interval_seconds"]
            or contract.get("bounds") != {"fit": [0, n * 3 // 5], "tune": [n * 3 // 5, n * 4 // 5], "eval": [n * 4 // 5, n]}
            or contract.get("quality", {}).get("domain_leakage_checks", {}).get("training_payload_contains_eval_rows") is not False):
        raise ValueError("Appliances metadata does not prove the separated physical prefix")
    path = locations["source"]
    if Path(contract["payloads"]["fit_tune"]["path"]).resolve() != path.resolve():
        raise ValueError("Appliances FIT_TUNE path differs from admitted prefix")
    headers = previous._headers(path)
    allowed = {"x", "physical", "filled", "mask", "mean", "std", "origins_fit", "origins_tune", "columns", "times"}
    rows, channels = n * 4 // 5, len(spec["columns"])
    if (set(headers) != allowed or headers["physical"] != ((rows, channels), np.dtype("float32"))
            or headers["mask"] != ((rows, channels), np.dtype("bool"))
            or headers["times"][0] != (rows,) or headers["times"][1].kind != "U"
            or headers["columns"][0] != (channels,) or headers["columns"][1].kind != "U"):
        raise ValueError("Appliances prefix header contains unexpected arrays, types or tail rows")
    digest = sha256_file(path)
    if digest != contract["payloads"]["fit_tune"]["sha256"]:
        raise ValueError("Appliances separated-prefix fingerprint mismatch")
    with np.load(path, allow_pickle=False) as payload:
        physical = np.array(payload["physical"], dtype=np.float32, copy=True, order="C")
        observed = np.array(payload["mask"], dtype=bool, copy=True, order="C")
        labels = payload["times"].copy()
        columns = payload["columns"].copy()
    times = labels.astype("datetime64[ns]").astype(np.int64)
    provenance = dict(source_prepared_prefix_payload_sha256=digest,
        source_prefix_hash_scope="Entire already isolated physical FIT_TUNE payload only; raw ZIP and EVAL payloads were not read or hashed",
        original_normalized_arrays_consumed=False, physical_inverse_transform_required=False)
    return physical, observed, times, columns, provenance, [binding]


def role_observability(physical, observed, columns, ranges):
    cumulative = np.concatenate([np.zeros((1, len(columns)), dtype=np.int64), observed.cumsum(axis=0)], axis=0)
    result = {}
    for role in ("A_fit", "A_stop", "A60", "V_select"):
        lo, hi = ranges[role]
        stride = 1 if role in ("A_fit", "A60") else H
        origins = np.arange(max(L, lo), hi - H + 1, stride, dtype=np.int64)
        window_counts = cumulative[origins + H] - cumulative[origins]
        counts = window_counts.sum(axis=0)
        result[role] = dict(origin_stride=stride, origins=len(origins),
            first_origin=None if not len(origins) else int(origins[0]), last_origin=None if not len(origins) else int(origins[-1]),
            observed_targets_per_channel=counts.tolist(), observed_scalar_targets=int(counts.sum()),
            possible_scalar_targets=int(len(origins) * H * len(columns)),
            unobserved_columns=[c for c, count in zip(columns, counts) if count == 0],
            origins_with_no_observed_target=int((window_counts.sum(axis=1) == 0).sum()),
            row_observed_counts=observed[lo:hi].sum(axis=0).astype(int).tolist())
    return result


def readiness_issues(roles):
    return [dict(role=role, unobserved_columns=roles[role]["unobserved_columns"])
            for role in ("A_fit", "A_stop") if roles[role]["unobserved_columns"]]


def prepare_dataset(asset_root, dataset, output_root):
    locations = source_locations(asset_root, dataset)
    spec = DATASETS[dataset]
    folder = Path(output_root) / dataset
    if folder.exists():
        raise FileExistsError(f"Prepared dataset already exists: {folder}")
    if dataset == "appliances":
        physical, observed, times, columns, provenance, evidence = _appliance_prefix(asset_root, locations, spec)
    else:
        evidence, historical = _admit_csv(asset_root, dataset, locations, spec)
        physical, observed, times, columns, provenance = _csv_prefix(locations["source"], dataset, spec)
        provenance["historical_full_raw_sha256_in_metadata_not_recomputed"] = historical["sha256"]
    rows = spec["rows"] * 4 // 5
    if (physical.shape != (rows, len(spec["columns"])) or observed.shape != physical.shape
            or columns.tolist() != list(spec["columns"]) or times.shape != (rows,)
            or not np.array_equal(observed, np.isfinite(physical))
            or not np.all(np.diff(times) == spec["interval_seconds"] * 10**9)):
        raise ValueError("Prepared prefix physical values, mask, columns or clock differ from contract")
    ranges = split_ranges(spec["rows"])
    structure = previous.train_structure(physical, observed, columns.tolist(), ranges["A_fit"][1])
    roles = role_observability(physical, observed, columns.tolist(), ranges)
    issues = readiness_issues(roles)
    if structure["unobserved_columns"]:
        issues.append(dict(role="A_fit_rows", unobserved_columns=structure["unobserved_columns"]))
    metadata = dict(schema_version=1, status="NOT_READY_UNOBSERVED_CHANNEL" if issues else "READY",
        dataset=dataset, C=len(columns), L=L, H=H, interval_seconds=spec["interval_seconds"],
        period=24 if spec["interval_seconds"] == 3600 else 144,
        original_rows=spec["rows"], prefix_rows=rows, prefix_only_approved=True,
        candidate_training_admitted=not issues, columns=columns.tolist(), clock=spec["clock"], split_ranges=ranges,
        train_structure=structure, role_observability=roles, readiness_issues=issues,
        qualification_scope="A_fit_and_A_stop_observability_only", column_selection="All preregistered physical channels retained; no validation-based channel removal",
        source_metadata=evidence, source_full_file_hashed=False, **provenance,
        source_reference="asset-parent://" + locations["source"].resolve().relative_to(Path(asset_root).resolve().parent).as_posix(),
        prefix_array_hashes={"physical": previous.array_sha256(physical), "observed": previous.array_sha256(observed),
                             "times": previous.array_sha256(times), "columns": previous.array_sha256(columns)},
        missingness="Original observed values and missing mask; missing targets remain NaN; Air Quality values <=-200 are missing",
        normalization="No source scaler used. Fit fresh A_fit/A60 population statistics in the training consumer; reset causal history fill per window",
        protected_numeric_access=dict(T_final=False, old_EVAL=False, full_raw_file=False),
        development_exposure="Previously used public source or family; this adaptive development search is not independent test evidence",
        source_row_count_note=("UCI page describes 9358 instances; the existing local timestamp/schema QC records 9357 usable hourly rows. This prefix uses the local 9357-row contract and does not claim equality to the page count."
                               if dataset == "uci360_air_quality" else "Original row count follows the bound existing source schema metadata"),
        source_sha256=sha256_file(Path(__file__)),
        dependency_source_hashes={"research/b2_dataset_extension_20261018_v1/data.py": sha256_file(Path(previous.__file__))})
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "PREFIX_PHYSICAL.npz"
    with output.open("xb") as handle:
        np.savez_compressed(handle, physical=np.ascontiguousarray(physical), observed=np.ascontiguousarray(observed),
                            times=np.ascontiguousarray(times), columns=np.ascontiguousarray(columns))
    metadata.update(prefix_sha256=sha256_file(output), prefix_bytes=output.stat().st_size)
    with (folder / "DATA_DIAGNOSTIC.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    args = parser.parse_args()
    if args.public_output is not None and args.public_output.exists():
        raise FileExistsError(args.public_output)
    names = DEFAULT_WAVE if args.dataset == "all" else (args.dataset,)
    results = [prepare_dataset(args.asset_root, name, args.output_root) for name in names]
    report = dict(schema_version=1, status="COMPLETE_DIAGNOSTICS", datasets=results,
                  ready_datasets=[r["dataset"] for r in results if r["status"] == "READY"],
                  source_sha256=sha256_file(Path(__file__)))
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        with args.public_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(dict(status=report["status"], ready_datasets=report["ready_datasets"],
        datasets=[{key: r[key] for key in ("dataset", "status", "C", "prefix_rows", "readiness_issues")} for r in results]),
        ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
