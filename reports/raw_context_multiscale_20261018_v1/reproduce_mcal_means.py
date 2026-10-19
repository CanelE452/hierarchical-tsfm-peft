"""Public-only, stdlib replay of all paired M_cal diagnostic means.

python reproduce_mcal_means.py --source-dir <report> --output-dir <new-empty-dir>
The original forty model rows and their PASS validation remain untouched.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


SOURCE_CSV = "MCAL_FINAL_FIT_DIAGNOSTICS.csv"
SOURCE_VALIDATION = "MCAL_FINAL_FIT_VALIDATION.json"
OUTPUT_CSV = "MCAL_FINAL_FIT_MEANS.csv"
METRICS = ("M_fit_selected_model_mse", "M_fit_selected_model_mae", "M_select_selected_model_mse",
           "M_select_selected_model_mae", "M_cal_final_model_mse", "M_cal_final_model_mae", "V_report_mse", "V_report_mae")
GROUP_FIELDS = ("dataset", "mode", "base_role", "alpha_label")
COHORT_FIELDS = ("origins_identity", "target_identity", "observed_identity", "scaler_sha256",
                 "A60_normalization_fit_end", "n_origins", "n_observed")
SEEDS = (92601, 92602)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key:decode(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def rows_from_bytes(value):
    return [{key:decode(cell) for key, cell in row.items()}
            for row in csv.DictReader(io.StringIO(value.decode("utf-8-sig"), newline=""))]


def summarize(rows, source_csv_sha, validation_sha):
    if len(rows) != 40:
        raise ValueError("Exactly forty original model rows are required")
    if len({row.get("run_id") for row in rows}) != 40 or any(not row.get("run_id") for row in rows):
        raise ValueError("Forty unique real model run_id values are required")
    expected = {(dataset, "alpha", role, label) for dataset in ("jena", "pems32")
                for role in ("Raw", "G") for label in ("alpha0", "alpha05", "alpha1", "fitted")}
    expected |= {(dataset, mode, None, None) for dataset in ("jena", "pems32") for mode in ("conditional", "past_control")}
    groups = {}
    for row in rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    if set(groups) != expected:
        raise ValueError("All sixteen alpha and four head paired groups must remain present")
    result = []
    original_fields = {key for row in rows for key in row}
    for key in sorted(groups, key=lambda group:tuple("" if value is None else str(value) for value in group)):
        pair = groups[key]
        if len(pair) != 2 or {row.get("seed") for row in pair} != set(SEEDS):
            raise ValueError("Each group requires exactly seed92601 and seed92602; missing/duplicate seeds are not replaced")
        pair.sort(key=lambda row:row["seed"])
        shared = {}
        for field in COHORT_FIELDS:
            left, right = pair[0].get(field), pair[1].get(field)
            if left is None or right is None or left != right:
                raise ValueError("Paired cohort/scaler/fit identity differs or is missing: "+field)
            shared[field] = left
        summary = {field:None for field in original_fields if field != "run_id"}
        summary.update(dict(zip(GROUP_FIELDS, key)), row_kind="two_seed_mean", seed="mean", n_models=2,
            summary_id="two_seed_mean__"+"__".join("none" if value is None else str(value) for value in key),
            shared_cohort_identity=shared, by_seed={str(row["seed"]):row for row in pair},
            aggregation_scope="Arithmetic mean of the eight named metrics only; auxiliary values remain null with exact original rows in by_seed",
            shared_identity_scope="Exact equal cohort/scaler/fit/count identities, copied unchanged into shared_cohort_identity; never averaged",
            M_fit_M_select_scope="Copied pre-refit fitting/selection checkpoint or coefficient diagnostics; not reselected or recomputed",
            M_cal_final_scope="In-sample fixed final-model fit error over the original dense M_cal cohort",
            V_report_scope="Copied completed development-V reporting errors; no new forecasts and no independent-test claim",
            source_csv_binding=dict(reference="report://"+SOURCE_CSV, sha256=source_csv_sha),
            source_validation_binding=dict(reference="report://"+SOURCE_VALIDATION, sha256=validation_sha))
        for metric in METRICS:
            values = [row.get(metric) for row in pair]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ValueError("Required finite metric missing or invalid: "+metric)
            summary[metric] = math.fsum(values)/2
        result.append(summary)
    return result


def reproduce(source_dir, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be separate from and outside the original source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; preserve all existing output files")
    validation_path = source/SOURCE_VALIDATION
    validation_bytes = validation_path.read_bytes()
    validation = json.loads(validation_bytes.decode("utf-8-sig"))
    if validation.get("status") != "PASS" or validation.get("models") != 40:
        raise ValueError("Source final-fit validation must PASS for all forty models")
    csv_path = source/SOURCE_CSV
    csv_bytes = csv_path.read_bytes()
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    if validation.get("CSV_binding") != dict(reference="report://"+SOURCE_CSV, sha256=csv_sha):
        raise ValueError("PASS validation must bind the exact source CSV reference and SHA256")
    rows = summarize(rows_from_bytes(csv_bytes), csv_sha, hashlib.sha256(validation_bytes).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    target = output/OUTPUT_CSV
    fields = sorted({key for row in rows for key in row})
    with target.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return dict(status="COMPLETE", rows=len(rows), output_file=OUTPUT_CSV, sha256=sha(target))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(reproduce(args.source_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
