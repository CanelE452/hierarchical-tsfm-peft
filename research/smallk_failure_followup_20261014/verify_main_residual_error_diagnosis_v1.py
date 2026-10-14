"""Final verifier for MAIN residual error-diagnosis CSV/manifest/report outputs.

The verifier is file/aggregate only: it reads diagnosis CSV/JSON/Markdown and the
four producer RESULT JSONs recorded by the diagnosis manifest. It does not open
NPZ/checkpoint/model assets, does not replay solvers, timing, E/F/D, or F.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / "validation_v1" / "MAIN_ERROR_DIAGNOSIS_FINAL_VERIFICATION_v1.json"
DATASETS = ("robin", "beijing_aotizhongxin")
SEEDS = (92601, 92602)
ROUTES = ("RAW33C", "E33K", "E512K")
ALL_METHODS = ("MEAN_CORRECTION", "RAW33C", "E33K", "E512K")
BIN_LABELS = ("[0,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,2)", "[2,inf)")
DATASET_CHANNELS = {"robin": 17, "beijing_aotizhongxin": 11}
FINAL_SPLIT = {"robin": "val", "beijing_aotizhongxin": "eval"}
TOL = 1e-10


class VerificationFailure(AssertionError):
    pass


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def file_record(path: Path | str) -> dict:
    value = resolve(path)
    return {"path": str(value), "sha256": sha(value), "bytes": value.stat().st_size}


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record_sha(record: dict) -> str:
    value = record.get("sha256") or record.get("verified_sha256")
    if not value:
        raise VerificationFailure("RECORD_SHA_REQUIRED")
    return str(value).lower()


def verify_file(record: dict) -> dict:
    path = resolve(record["path"])
    actual = sha(path)
    if actual.lower() != record_sha(record):
        raise VerificationFailure("FILE_SHA_MISMATCH:" + str(path))
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "role": record.get("role", "")}


def parse_float(value):
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def parse_int(value):
    if value in (None, "", "None", "null"):
        return None
    return int(value)


def close(a, b, tol=TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict) -> tuple:
    return (row["dataset"], int(row["seed"]), row["split"], row["method"], row["reference"])


def source_score(producer: dict, split: str, method: str) -> dict:
    if method == "ORIGINAL_MAIN":
        return producer["references"][split]["ORIGINAL_MAIN"]["normalized"]
    if method == "MEAN_CORRECTION":
        return producer["controls"]["MEAN_CORRECTION"]["splits"][split]["normalized"]
    if method in ROUTES:
        return producer["families"][method]["splits"][split]["normalized"]
    raise VerificationFailure("UNKNOWN_METHOD:" + method)


def identify_producers(manifest: dict) -> tuple[dict[tuple, dict], list[dict]]:
    producers = {}
    records_by_key = {}
    for record in manifest.get("input_records", []):
        try:
            checked = verify_file(record)
            value = read_json(checked["path"])
        except Exception:
            continue
        key = (value.get("dataset"), value.get("seed"))
        if (key[0] in DATASETS and key[1] in SEEDS and isinstance(value.get("families"), dict)
                and value.get("maximum_candidate_comparisons") == 27
                and value.get("maximum_new_ridge_coefficients") == 21
                and "MEAN_CORRECTION" in value.get("controls", {})
                and all(route in value.get("families", {}) for route in ROUTES)):
            if key in producers:
                raise VerificationFailure("DUPLICATE_MAIN_RESIDUAL_PRODUCER_RECORD:" + str(key))
            producers[key] = value
            records_by_key[key] = checked
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(producers) != expected:
        raise VerificationFailure("EXACT_FOUR_CANONICAL_PRODUCERS_REQUIRED")
    return producers, [records_by_key[key] for key in sorted(records_by_key)]


def verify_manifest_files(folder: Path, manifest: dict, failures: list[str]) -> list[dict]:
    checked = []
    for record in manifest.get("files", []):
        try:
            checked.append(verify_file(record))
        except Exception as error:
            failures.append("MANIFEST_FILE:" + type(error).__name__ + ":" + str(error))
    for name in ("CHANNEL_CONTRIBUTIONS.csv", "BLOCK_CONTRIBUTIONS.csv", "ERROR_BINS.csv", "TUNE_MARGINS.csv", "REPORT_KO.md"):
        path = folder / name
        if not path.is_file():
            failures.append("REQUIRED_OUTPUT_MISSING:" + name)
        elif not any(Path(record["path"]).name == name for record in checked):
            failures.append("REQUIRED_OUTPUT_NOT_MANIFESTED:" + name)
    for record in manifest.get("source_records_after", []):
        try:
            verify_file(record)
        except Exception as error:
            failures.append("SOURCE_AFTER:" + type(error).__name__ + ":" + str(error))
    if manifest.get("source_artifacts_before_after_SHA_pass") is not True:
        failures.append("SOURCE_BEFORE_AFTER_NOT_PASS")
    return checked


def verify_links(folder: Path, failures: list[str]) -> dict:
    report = folder / "REPORT_KO.md"
    text = report.read_text(encoding="utf-8-sig")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    bad = []
    for target in links:
        if ".cache" in target.replace("\\", "/"):
            bad.append({"target": target, "reason": "cache-link"})
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue
        if target.startswith("/") or target.startswith("\\"):
            bad.append({"target": target, "reason": "absolute-link"})
            continue
        if not (folder / target).exists():
            bad.append({"target": target, "reason": "missing"})
    if bad:
        failures.append("REPORT_LINK_FAILURES")
    return {"links_checked": len(links), "bad_links": bad}


def verify_totals_against_producers(manifest: dict, producers: dict[tuple, dict], failures: list[str]) -> dict[tuple, dict]:
    totals = {}
    for row in manifest.get("totals", []):
        key = (row["dataset"], int(row["seed"]), row["split"], row["method"], row["reference"])
        totals[key] = row
        producer = producers[(row["dataset"], int(row["seed"]))]
        pred = source_score(producer, row["split"], row["method"])
        ref = source_score(producer, row["split"], row["reference"])
        for metric in ("mse", "mae"):
            if not close(row["prediction_scores"][metric], pred[metric]):
                failures.append("TOTAL_PRED_SCORE:" + str(key) + ":" + metric)
            if not close(row["reference_scores"][metric], ref[metric]):
                failures.append("TOTAL_REF_SCORE:" + str(key) + ":" + metric)
            if not close(row["delta"][metric], row["prediction_scores"][metric] - row["reference_scores"][metric]):
                failures.append("TOTAL_DELTA:" + str(key) + ":" + metric)
    if len(totals) != 60:
        failures.append("TOTALS_COUNT_NOT_60")
    expected = set()
    for dataset in DATASETS:
        for seed in SEEDS:
            for split in ("fit", "tune", FINAL_SPLIT[dataset]):
                for method in ALL_METHODS:
                    expected.add((dataset, seed, split, method, "ORIGINAL_MAIN"))
                expected.add((dataset, seed, split, "E33K", "MEAN_CORRECTION"))
    if set(totals) != expected:
        failures.append("TOTALS_KEYSET_MISMATCH")
    return totals


def verify_partition(rows: list[dict], totals: dict[tuple, dict], kind: str, failures: list[str]) -> int:
    by_key = {}
    for row in rows:
        key = group_key(row)
        by_key.setdefault(key, []).append(row)
        dataset = row["dataset"]
        channels = DATASET_CHANNELS[dataset]
        channel = parse_int(row["channel_index"])
        if channel is None or not (0 <= channel < channels):
            failures.append(kind + ":CHANNEL_RANGE:" + str(key))
        active = parse_int(row["global_active_channels"])
        observed_global = parse_int(row["global_channel_observed_count"])
        weight = parse_float(row["per_observed_cell_weight"])
        if observed_global and active:
            if not close(weight, 1.0 / (active * observed_global)):
                failures.append(kind + ":ROW_WEIGHT_NC:" + str(key))
        elif not close(weight, 0.0):
            failures.append(kind + ":ROW_WEIGHT_ZERO:" + str(key))
        if kind == "channel":
            if parse_int(row["observed_count"]) != observed_global:
                failures.append("CHANNEL_OBSERVED_COUNT:" + str(key))
        elif kind == "block":
            q = parse_int(row["block_quartile"])
            if q is None or not (0 <= q <= 3):
                failures.append("BLOCK_QUARTILE_RANGE:" + str(key))
            first = parse_int(row["first_window_index"])
            last = parse_int(row["last_window_index"])
            wc = parse_int(row["window_count"])
            if wc == 0 and (first is not None or last is not None):
                failures.append("BLOCK_EMPTY_INDEX:" + str(key))
            if wc and (first is None or last is None or first > last):
                failures.append("BLOCK_INDEX_ORDER:" + str(key))
        elif kind == "error_bins":
            idx = parse_int(row["bin_index"])
            if idx is None or not (0 <= idx < len(BIN_LABELS)) or row["MAIN_abs_error_bin"] != BIN_LABELS[idx]:
                failures.append("ERROR_BIN_LABEL:" + str(key))
    for key, group in by_key.items():
        total = totals.get(key)
        if total is None:
            failures.append(kind + ":GROUP_NOT_IN_TOTALS:" + str(key))
            continue
        observed = sum(parse_int(row["observed_count"]) or 0 for row in group)
        if observed != int(total["observed_count"]):
            failures.append(kind + ":OBSERVED_SUM:" + str(key))
        for metric in ("mse", "mae"):
            summed = sum(parse_float(row["signed_" + metric + "_contribution"]) or 0.0 for row in group)
            if not close(summed, total["delta"][metric]):
                failures.append(kind + ":DELTA_SUM:" + str(key) + ":" + metric)
    if set(by_key) != set(totals):
        failures.append(kind + ":GROUP_KEYSET_MISMATCH")
    return len(by_key)


def verify_tune_margins(rows: list[dict], producers: dict[tuple, dict], failures: list[str]) -> int:
    by_key = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]), row["route"])
        by_key.setdefault(key, []).append(row)
    expected = {(dataset, seed, route) for dataset in DATASETS for seed in SEEDS for route in ROUTES}
    if set(by_key) != expected or len(rows) != 108:
        failures.append("TUNE_MARGIN_COVERAGE")
    for key, group in by_key.items():
        group = sorted(group, key=lambda row: int(row["candidate_index"]))
        if len(group) != 9:
            failures.append("TUNE_MARGIN_NINE_ROWS:" + str(key))
            continue
        producer = producers[(key[0], key[1])]
        family = producer["families"][key[2]]
        grid = family["candidate_grid"]
        selected = grid[family["selected_index"]]
        selected_score = selected["tune_actual_Y_mse"]
        zero_score = grid[0].get("tune_actual_Y_mse")
        mean_score = grid[1].get("tune_actual_Y_mse")
        other_scores = [candidate.get("tune_actual_Y_mse") for i, candidate in enumerate(grid)
                        if i != family["selected_index"] and candidate.get("status") == "COMPLETE" and candidate.get("tune_actual_Y_mse") is not None]
        best_other = min(other_scores) if other_scores else None
        for row, candidate in zip(group, grid):
            idx = int(row["candidate_index"])
            value = parse_float(row["candidate_tune_mse"])
            if row["candidate_id"] != candidate["candidate_id"] or row["candidate_status"] != candidate["status"]:
                failures.append("TUNE_MARGIN_CANDIDATE_ID_STATUS:" + str(key) + ":" + str(idx))
            row_alpha = parse_float(row["alpha"])
            candidate_alpha = candidate.get("alpha")
            if not close(row_alpha, candidate_alpha):
                failures.append("TUNE_MARGIN_ALPHA:" + str(key) + ":" + str(idx))
            if not close(value, candidate.get("tune_actual_Y_mse")):
                failures.append("TUNE_MARGIN_SCORE:" + str(key) + ":" + str(idx))
            is_selected = str(row["is_selected"]).lower() == "true"
            if is_selected != (idx == family["selected_index"]):
                failures.append("TUNE_MARGIN_SELECTED_FLAG:" + str(key) + ":" + str(idx))
            if row["selected_candidate_id"] != family["selected_candidate_id"]:
                failures.append("TUNE_MARGIN_SELECTED_ID:" + str(key))
            if not close(parse_float(row["selected_tune_mse"]), selected_score):
                failures.append("TUNE_MARGIN_SELECTED_SCORE:" + str(key))
            if not close(parse_float(row["candidate_minus_selected_mse"]), value - selected_score if value is not None else None):
                failures.append("TUNE_MARGIN_CANDIDATE_MINUS_SELECTED:" + str(key) + ":" + str(idx))
            if not close(parse_float(row["selected_improvement_vs_ZERO"]), zero_score - selected_score if zero_score is not None else None):
                failures.append("TUNE_MARGIN_ZERO_IMPROVEMENT:" + str(key) + ":" + str(idx))
            if not close(parse_float(row["selected_improvement_vs_MEAN"]), mean_score - selected_score if mean_score is not None else None):
                failures.append("TUNE_MARGIN_MEAN_IMPROVEMENT:" + str(key) + ":" + str(idx))
            if not close(parse_float(row["selected_margin_to_best_other"]), best_other - selected_score if best_other is not None else None):
                failures.append("TUNE_MARGIN_BEST_OTHER:" + str(key) + ":" + str(idx))
            if str(row["selection_changed"]).lower() != "false":
                failures.append("TUNE_MARGIN_SELECTION_CHANGED:" + str(key))
    return len(rows)


def weighted_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def interpretation(channel_rows: list[dict], block_rows: list[dict], bin_rows: list[dict], totals: dict[tuple, dict]) -> dict:
    dataset = "beijing_aotizhongxin"
    split = "eval"
    method = "E33K"
    reference = "ORIGINAL_MAIN"
    keys = [(dataset, seed, split, method, reference) for seed in SEEDS]
    final_totals = [totals[key] for key in keys if key in totals]
    def rows_for(rows):
        return [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["method"] == method and row["reference"] == reference]
    channel_summary = []
    for c in range(DATASET_CHANNELS[dataset]):
        selected = [row for row in rows_for(channel_rows) if int(row["channel_index"]) == c]
        channel_summary.append({"channel_index": c,
                                "two_seed_mean_signed_mse_contribution": weighted_mean([float(row["signed_mse_contribution"]) for row in selected]),
                                "two_seed_mean_signed_mae_contribution": weighted_mean([float(row["signed_mae_contribution"]) for row in selected])})
    block_summary = []
    for q in range(4):
        selected = [row for row in rows_for(block_rows) if int(row["block_quartile"]) == q]
        block_summary.append({"block_quartile": q,
                              "two_seed_mean_signed_mse_contribution": weighted_mean([float(row["signed_mse_contribution"]) for row in selected]),
                              "two_seed_mean_signed_mae_contribution": weighted_mean([float(row["signed_mae_contribution"]) for row in selected])})
    bin_summary = []
    for i, label in enumerate(BIN_LABELS):
        selected = [row for row in rows_for(bin_rows) if int(row["bin_index"]) == i]
        bin_summary.append({"MAIN_abs_error_bin": label,
                            "two_seed_mean_signed_mse_contribution": weighted_mean([float(row["signed_mse_contribution"]) for row in selected]),
                            "two_seed_mean_signed_mae_contribution": weighted_mean([float(row["signed_mae_contribution"]) for row in selected])})
    return {"scope": "Beijing eval E33K vs ORIGINAL_MAIN; two-seed arithmetic mean of weighted signed contributions; no CI/IID claim",
            "two_seed_mean_delta": {"mse": weighted_mean([row["delta"]["mse"] for row in final_totals]),
                                      "mae": weighted_mean([row["delta"]["mae"] for row in final_totals])},
            "channels": channel_summary,
            "blocks": block_summary,
            "bins": bin_summary}


def execute(args: argparse.Namespace) -> int:
    import main_residual_runtime_v1 as runtime

    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    output = resolve(args.out)
    result = {"schema": "main_residual_error_diagnosis_final_verification_v1", "status": "RUNNING", "tag": args.tag,
              "started_unix": time.time(), "failures": [], "failed_count": 0, "checks": 0,
              "scope": "CSV/manifest/Markdown aggregate verification only; no CP/NPZ/model/solver/timing/F replay",
              "audit_source_before": file_record(Path(__file__)), "source_records": [file_record(Path(__file__))],
              "resource_authorization": limit.authorization_record, "run_manifest": limit.run_manifest_record}
    output_ready = False
    try:
        if output.exists():
            raise FileExistsError(str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output_ready = True
        limit.checkfull()
        diagnosis = resolve(args.diagnosis)
        folder = diagnosis.parent if diagnosis.name == "MANIFEST.json" else diagnosis
        manifest_path = folder / "MANIFEST.json"
        manifest_record = file_record(manifest_path)
        manifest = read_json(manifest_path)
        failures = result["failures"]
        if manifest.get("status") != "COMPLETE":
            failures.append("DIAGNOSIS_MANIFEST_NOT_COMPLETE")
        producers, producer_records = identify_producers(manifest)
        result["conditions"] = []
        for record in producer_records:
            value = read_json(record["path"])
            result["conditions"].append({"dataset": value["dataset"], "seed": int(value["seed"]), "producer_sha256": record["sha256"]})
        verified_files = verify_manifest_files(folder, manifest, failures)
        link_result = verify_links(folder, failures)
        channel_rows = csv_rows(folder / "CHANNEL_CONTRIBUTIONS.csv")
        block_rows = csv_rows(folder / "BLOCK_CONTRIBUTIONS.csv")
        bin_rows = csv_rows(folder / "ERROR_BINS.csv")
        margin_rows = csv_rows(folder / "TUNE_MARGINS.csv")
        totals = verify_totals_against_producers(manifest, producers, failures)
        channel_groups = verify_partition(channel_rows, totals, "channel", failures)
        block_groups = verify_partition(block_rows, totals, "block", failures)
        bin_groups = verify_partition(bin_rows, totals, "error_bins", failures)
        margins_checked = verify_tune_margins(margin_rows, producers, failures)
        expected_counts = {"channel": 840, "block": 3360, "error_bins": 4200, "tune_margins": 108}
        actual_counts = {"channel": len(channel_rows), "block": len(block_rows), "error_bins": len(bin_rows), "tune_margins": len(margin_rows)}
        if manifest.get("row_counts") != actual_counts:
            failures.append("MANIFEST_ROW_COUNTS_MISMATCH")
        if actual_counts != expected_counts:
            failures.append("EXPECTED_ROW_COUNTS_MISMATCH")
        result.update(diagnosis_manifest=manifest_record, files_checked=verified_files, links=link_result,
                      row_counts=actual_counts, partition_groups={"channel": channel_groups, "block": block_groups, "error_bins": bin_groups},
                      tune_margin_rows_checked=margins_checked, totals_checked=len(totals),
                      interpretation=interpretation(channel_rows, block_rows, bin_rows, totals))
        result["checks"] = len(verified_files) + len(channel_rows) + len(block_rows) + len(bin_rows) + len(margin_rows) + len(totals)
        result["failed_count"] = len(failures)
        result["status"] = "PASS" if not failures else "FAILED_PRESERVED"
        limit.checkfull()
    except Exception as error:
        result["status"] = "BUDGET_HELD_PRESERVED" if isinstance(error, runtime.BudgetHeld) else "FAILED_PRESERVED"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_count"] = max(1, len(result.get("failures", [])))
    finally:
        source_after = file_record(Path(__file__))
        result["audit_source_after"] = source_after
        result["source_before_after_SHA_pass"] = source_after["sha256"] == result["audit_source_before"]["sha256"]
        if result["source_before_after_SHA_pass"] is not True and result.get("status") == "PASS":
            result["status"] = "FAILED_PRESERVED"
            result["failed_count"] = max(1, result.get("failed_count", 0))
            result["error"] = {"type": "VerifierSourceChanged", "message": "source SHA changed before write"}
        terminal = "COMPLETE" if result.get("status") == "PASS" else result.get("status", "FAILED_PRESERVED")
        receipt = limit.close(terminal, result.get("error") or (result.get("failures") if result.get("status") != "PASS" else None))
        result["resource_receipt"] = receipt
        if receipt.get("terminal_status") != "COMPLETE" and result.get("status") == "PASS":
            result["status"] = receipt["terminal_status"]
            result["terminal_error"] = receipt.get("terminal_error")
            result["failed_count"] = max(1, result.get("failed_count", 0))
        result["finished_unix"] = time.time()
        if output_ready:
            write_new_json(output, result)
    interp = result.get("interpretation", {}).get("two_seed_mean_delta", {})
    print(json.dumps({"status": result["status"], "failed_count": result.get("failed_count"), "totals_checked": result.get("totals_checked"),
                      "beijing_E33K_vs_MAIN_eval_two_seed_mean_delta": interp, "out": str(output)}, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnosis")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--tag", default="main_error_diagnosis_final_verification_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({"status": "SOURCE_ONLY_NOT_EXECUTED", "schema": "main_residual_error_diagnosis_final_verification_v1",
                          "expected_totals": 60, "expected_rows": {"channel": 840, "block": 3360, "error_bins": 4200, "tune_margins": 108},
                          "asset_scope": "CSV/JSON/Markdown only; no NPZ/CP/model/F replay"}, ensure_ascii=False))
        return 0
    if not args.diagnosis:
        parser.error("--execute requires --diagnosis folder-or-MANIFEST.json")
    if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in args.tag):
        parser.error("tag must be lowercase safe")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
