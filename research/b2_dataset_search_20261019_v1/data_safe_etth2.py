"""Prepare the amended ETTh2 cohort without consuming its earlier protected tail."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.b2_dataset_search_20261019_v1 import data as base


REPO = Path(__file__).resolve().parents[2]
NAME = "b2_dataset_search_20261019_v1"
DATASET = "ETTh2_safe17400"
SOURCE_ROWS, COHORT_ROWS, PREFIX_ROWS = 17420, 17400, 13920
PROTECTED_CUTOFF = "2018-02-01 00:00:00"


def _amendment(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "reports" / NAME):
        raise ValueError("The registered amendment must stay inside the search report namespace")
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = dict(status="REGISTERED_BEFORE_ANY_NEW_GPU_TRAINING_OR_FORECAST_SCORES",
        original_candidate="ETTh2", source_original_rows=SOURCE_ROWS, original_prepared_prefix_rows=13936,
        per_file_historical_ETTh2_TEST_start=13936,
        same_station_ETTm2_protected_TEST_start_15minute_rows=55680,
        same_station_earliest_hourly_cutoff=PREFIX_ROWS, earliest_protected_cutoff_timestamp=PROTECTED_CUTOFF)
    replacement = dict(dataset=DATASET, source_original_rows=SOURCE_ROWS, cohort_rows=COHORT_ROWS,
        prefix_rows=PREFIX_ROWS, source_tail_outside_cohort=[COHORT_ROWS, SOURCE_ROWS],
        training_recipe_change=False, forecast_result_used_to_choose_cutoff=False)
    replacement.update({("T_final_within_cohort" if key == "T_final" else key): bounds
                        for key, bounds in base.split_ranges(COHORT_ROWS).items()})
    if (any(value.get(key) != target for key, target in expected.items())
            or any(value.get("replacement", {}).get(key) != target for key, target in replacement.items())):
        raise ValueError("Registered amendment differs from the safe ETTh2 source/cohort/protection contract")
    return dict(reference="repo://" + path.relative_to(REPO).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def prepare_dataset(asset_root, dataset, output_root, *, amendment_path):
    if dataset != DATASET:
        raise ValueError(f"Unsupported dataset: {dataset}")
    folder = Path(output_root) / dataset
    if folder.exists():
        raise FileExistsError(f"Prepared dataset already exists: {folder}")
    amendment = _amendment(amendment_path)
    locations = base.source_locations(asset_root, "ETTh2")
    source_spec = base.DATASETS["ETTh2"]
    evidence, historical = base._admit_csv(asset_root, "ETTh2", locations, source_spec)
    cohort_spec = dict(source_spec, rows=COHORT_ROWS)
    physical, observed, times, columns, provenance = base._csv_prefix(locations["source"], "ETTh2", cohort_spec)
    step = source_spec["interval_seconds"] * 10**9
    first = np.datetime64(source_spec["first_timestamp"], "ns").astype(np.int64)
    cutoff = np.datetime64(PROTECTED_CUTOFF, "ns").astype(np.int64)
    if (physical.shape != (PREFIX_ROWS, len(source_spec["columns"])) or observed.shape != physical.shape
            or physical.dtype != np.dtype("float32") or observed.dtype != np.dtype("bool")
            or columns.tolist() != list(source_spec["columns"]) or times.shape != (PREFIX_ROWS,)
            or not np.array_equal(observed, np.isfinite(physical))):
        raise ValueError("Safe ETTh2 physical/mask/channel prefix differs from the contract")
    if (times.dtype != np.dtype("int64") or times[0] != first or not np.all(np.diff(times) == step)
            or times[-1] + step != cutoff or first + PREFIX_ROWS * step != cutoff):
        raise ValueError("Safe ETTh2 clock must end exclusively at the earlier protected cutoff")
    ranges = base.split_ranges(COHORT_ROWS)
    structure = base.previous.train_structure(physical, observed, columns.tolist(), ranges["A_fit"][1])
    roles = base.role_observability(physical, observed, columns.tolist(), ranges)
    issues = base.readiness_issues(roles)
    if structure["unobserved_columns"]:
        issues.append(dict(role="A_fit_rows", unobserved_columns=structure["unobserved_columns"]))
    provenance["source_prefix_hash_scope"] = (
        "Only the header and exactly 13920 original CSV records consumed for floor(.8 * cohort_N17400); "
        "not a full-source hash or the original N17420 prefix")
    metadata = dict(schema_version=1, status="NOT_READY_UNOBSERVED_CHANNEL" if issues else "READY",
        dataset=dataset, source_dataset="ETTh2", C=len(columns), L=base.L, H=base.H,
        interval_seconds=source_spec["interval_seconds"], period=24,
        original_rows=SOURCE_ROWS, source_rows=SOURCE_ROWS, cohort_rows=COHORT_ROWS,
        prefix_rows=PREFIX_ROWS, prefix_only_approved=True, candidate_training_admitted=not issues,
        columns=columns.tolist(), clock=source_spec["clock"], split_ranges=ranges,
        source_original_clock=dict(first_timestamp=source_spec["first_timestamp"],
                                   last_timestamp=source_spec["last_timestamp"]),
        cohort_definition="First 17400 coordinates of the 17420-row source; custom development cohort, not the official ETT split",
        source_tail_outside_cohort=[COHORT_ROWS, SOURCE_ROWS],
        prefix_clock=dict(first_timestamp=source_spec["first_timestamp"],
            last_timestamp="2018-01-31 23:00:00", exclusive_end_timestamp=PROTECTED_CUTOFF),
        protection_boundaries=dict(per_file_historical_TEST_start=13936,
            per_file_historical_TEST_timestamp="2018-02-01 16:00:00",
            same_station_ETTm2_TEST_start_15minute_rows=55680,
            same_station_protected_hourly_start=PREFIX_ROWS,
            earlier_same_station_cutoff_timestamp=PROTECTED_CUTOFF,
            per_file_minus_same_station_rows=16, value_equivalence_claimed=False,
            basis="Conservative shared physical ancestry and clock boundary; no hourly/minute value equality assumed"),
        train_structure=structure, role_observability=roles, readiness_issues=issues,
        qualification_scope="A_fit_and_A_stop_observability_only",
        column_selection="All seven original physical channels retained; no validation-based channel removal",
        source_metadata=evidence, amendment_binding=amendment, source_full_file_hashed=False,
        **provenance,
        historical_full_raw_sha256_in_metadata_not_recomputed=historical["sha256"],
        source_reference="asset-parent://" + locations["source"].resolve().relative_to(Path(asset_root).resolve().parent).as_posix(),
        prefix_array_hashes={name: base.previous.array_sha256(array) for name, array in
            dict(physical=physical, observed=observed, times=times, columns=columns).items()},
        missingness="Original physical values and observed mask; missing targets remain NaN",
        normalization="No source scaler used. Fit fresh A_fit/A60 statistics in the consumer; causal history fill resets per window",
        protected_numeric_access=dict(T_final=False, earlier_same_station_TEST=False,
                                      old_EVAL=False, superseded_ETTh2_prefix=False, full_raw_file=False),
        development_exposure="Previously used public source/family; adaptive development search is not independent test evidence",
        source_row_count_note="Original metadata remains N17420; split fractions apply only to the separately registered N17400 cohort",
        source_sha256=base.sha256_file(Path(__file__)),
        dependency_source_hashes={
            "research/b2_dataset_search_20261019_v1/data.py": base.sha256_file(Path(base.__file__)),
            "research/b2_dataset_extension_20261018_v1/data.py": base.sha256_file(Path(base.previous.__file__))})
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "PREFIX_PHYSICAL.npz"
    with output.open("xb") as handle:
        np.savez_compressed(handle, physical=np.ascontiguousarray(physical), observed=np.ascontiguousarray(observed),
                            times=np.ascontiguousarray(times), columns=np.ascontiguousarray(columns))
    metadata.update(prefix_sha256=base.sha256_file(output), prefix_bytes=output.stat().st_size)
    with (folder / "DATA_DIAGNOSTIC.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--dataset", choices=("all", DATASET), default=DATASET)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    if args.public_output is not None and args.public_output.exists():
        raise FileExistsError(args.public_output)
    result = prepare_dataset(args.asset_root, DATASET, args.output_root, amendment_path=args.amendment)
    report = dict(schema_version=1, status="COMPLETE_DIAGNOSTICS", datasets=[result],
        ready_datasets=[DATASET] if result["status"] == "READY" else [],
        source_sha256=base.sha256_file(Path(__file__)))
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        with args.public_output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(dict(status=report["status"], ready_datasets=report["ready_datasets"],
        datasets=[{key: result[key] for key in ("dataset", "status", "C", "prefix_rows", "readiness_issues")}]),
        ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
