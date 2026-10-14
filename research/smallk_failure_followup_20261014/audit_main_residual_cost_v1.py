"""Independent saved-array audit for main residual whole-path cost outputs.

Source default: no scientific assets, NumPy, Torch, CUDA, NPZs, or output files are
opened before the fresh main_residual_runtime_v1.ContinuationLimit(gpu=False).
The audit checks saved timed arrays, reference arrays, producer/audit bindings,
Latin schedule and timing arithmetic. It does not replay E/F, model timing, or
producer helper scoring.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / "validation_v1" / "MAIN_RESIDUAL_COST_AUDIT_v1_ACTUAL.json"
DATASETS = ("robin", "beijing_aotizhongxin")
SEED = 92601
ROUTES = ("RAW_BOLT", "ORIGINAL_MAIN", "MEAN_CONTROL", "RAW33C", "E33K", "E512K")
FAMILIES = ("RAW33C", "E33K", "E512K")
ORDER_SEED = 92726
REPEATS = 6
WARMUP = 2
NORMALIZED_ATOL, NORMALIZED_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5
STRICT_ATOL, STRICT_RTOL = 1e-14, 1e-12


class CostAuditFailure(AssertionError):
    pass


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    value = value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    if any(part.lower() == "test" for part in value.parts):
        raise PermissionError("PROTECTED_TEST_PATH")
    return value


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def file_record(path: Path | str) -> dict:
    value = resolve(path)
    return {"path": str(value), "sha256": sha(value), "bytes": value.stat().st_size}


def record_sha(record: dict) -> str:
    value = record.get("sha256") or record.get("verified_sha256")
    if not value:
        raise CostAuditFailure("RECORD_SHA_REQUIRED")
    return str(value).lower()


def verify_file(record: dict) -> dict:
    path = resolve(record["path"])
    actual = sha(path)
    if actual.lower() != record_sha(record):
        raise CostAuditFailure("FILE_SHA_MISMATCH:" + str(path))
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "role": record.get("role", "")}


def array_sha_raw(value) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_npz(record: dict, *, role: str, guard=lambda: None) -> tuple[dict, dict]:
    import numpy as np

    guard()
    checked = verify_file(record)
    with np.load(checked["path"], allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    declared = record.get("arrays")
    if declared is not None:
        if set(arrays) != set(declared):
            raise CostAuditFailure("NPZ_ARRAY_KEY_MISMATCH:" + role)
        for key, value in arrays.items():
            meta = declared[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or array_sha_raw(value) != meta["sha256"]:
                raise CostAuditFailure("NPZ_ARRAY_CONTENT_MISMATCH:" + role + ":" + key)
    guard()
    return arrays, checked


def close_arrays(left, right, scale: str) -> bool:
    import numpy as np

    atol, rtol = (NORMALIZED_ATOL, NORMALIZED_RTOL) if scale == "normalized" else (PHYSICAL_ATOL, PHYSICAL_RTOL)
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    return bool(np.all(np.abs(left.astype(np.float64) - right.astype(np.float64)) <= atol + rtol * np.abs(right.astype(np.float64))))


def masked_score(prediction, target, observed, std=None) -> dict:
    import numpy as np

    pred = np.asarray(prediction, np.float64)
    tgt = np.asarray(target, np.float64)
    mask = np.asarray(observed)
    if pred.shape != tgt.shape or mask.shape != tgt.shape or mask.dtype != np.bool_:
        raise CostAuditFailure("METRIC_TARGET_MASK_SCHEMA")
    if std is not None:
        error = (pred - tgt) * np.asarray(std, np.float64)
    else:
        error = pred - tgt
    error = np.where(mask, error, 0.0)
    count = mask.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error ** 2).sum(axis=(0, 1)), count, out=np.zeros(len(count)), where=active)
    mae = np.divide(np.abs(error).sum(axis=(0, 1)), count, out=np.zeros(len(count)), where=active)
    return {
        "mse": float(mse[active].mean()) if active.any() else None,
        "mae": float(mae[active].mean()) if active.any() else None,
        "active_channels": int(active.sum()),
        "observed_count": int(count.sum()),
        "per_channel_counts": count.astype(int).tolist(),
    }


def expected_schedule(np_module=None) -> list[list[str]]:
    if np_module is None:
        import numpy as np_module

    order = np_module.random.default_rng(ORDER_SEED).permutation(len(ROUTES))
    return [[ROUTES[i] for i in np_module.roll(order, -block)] for block in range(REPEATS)]


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not bool(condition):
        failures.append(label)


def check_timing(row: dict, raw_row: dict, schedule: list[list[str]], failures: list[str], label: str) -> None:
    import numpy as np

    seconds = np.asarray(row.get("seconds"), np.float64)
    blocks = list(row.get("block", []))
    positions = list(row.get("position", []))
    check(row.get("status") == "COMPLETE", label + ":status", failures)
    check(row.get("warmup_each_route") == WARMUP and row.get("repeats") == REPEATS, label + ":warm2_repeat6", failures)
    check(seconds.shape == (REPEATS,) and np.isfinite(seconds).all() and (seconds > 0).all(), label + ":seconds6_positive", failures)
    check(blocks == list(range(REPEATS)), label + ":blocks6", failures)
    check(sorted(positions) == list(range(len(ROUTES))), label + ":all_positions", failures)
    check(np.isclose(np.median(seconds), row.get("median_seconds"), atol=STRICT_ATOL, rtol=STRICT_RTOL), label + ":median", failures)
    check(np.isclose(np.median(seconds) / row.get("batch"), row.get("median_seconds_per_window"), atol=STRICT_ATOL, rtol=STRICT_RTOL), label + ":per_window", failures)
    raw_seconds = dict(zip(raw_row.get("block", []), raw_row.get("seconds", [])))
    paired = np.asarray([raw_seconds.get(block) for block in blocks], np.float64)
    ratios = seconds / paired
    check(np.array_equal(np.asarray(row.get("paired_RAW_seconds"), np.float64), paired), label + ":paired_raw_seconds", failures)
    check(np.allclose(np.asarray(row.get("paired_RAW_ratios"), np.float64), ratios, atol=STRICT_ATOL, rtol=STRICT_RTOL), label + ":paired_raw_ratios", failures)
    check(np.isclose(row.get("paired_RAW_median_ratio"), np.median(ratios), atol=STRICT_ATOL, rtol=STRICT_RTOL), label + ":paired_raw_median", failures)
    check(positions == [order.index(row["route"]) for order in schedule], label + ":order92726", failures)


def bind_scientific_audit(audit: dict, cost: dict, producer_records: list[dict], optional_results: list[Path]) -> list[dict]:
    expected = {(dataset, seed) for dataset in DATASETS for seed in (92601, 92602)}
    conditions = audit.get("conditions", [])
    if audit.get("status") != "PASS" or audit.get("failed_count") != 0 or len(conditions) != 4:
        raise CostAuditFailure("MAIN_RESIDUAL_SCIENCE_AUDIT_PASS_REQUIRED")
    if {(row.get("dataset"), int(row.get("seed"))) for row in conditions} != expected:
        raise CostAuditFailure("MAIN_RESIDUAL_SCIENCE_AUDIT_CONDITION_COVERAGE")
    if any(row.get("status") != "PASS" for row in conditions):
        raise CostAuditFailure("MAIN_RESIDUAL_SCIENCE_AUDIT_ROW_NOT_PASS")
    if audit.get("candidate_rows_checked") != 108 or audit.get("new_ridge_candidates_checked") != 84 or audit.get("selected_predictions_checked") != 36:
        raise CostAuditFailure("MAIN_RESIDUAL_SCIENCE_AUDIT_TOTAL_COUNTS")
    if audit.get("source_before_after_SHA_pass") is not True:
        raise CostAuditFailure("MAIN_RESIDUAL_SCIENCE_AUDIT_SOURCE_CHANGED")
    rows_by_key = {(row["dataset"], int(row["seed"])): row for row in conditions}
    for record in producer_records:
        producer = read_json(verify_file(record)["path"])
        row = rows_by_key[(producer["dataset"], int(producer["seed"]))]
        if int(producer["seed"]) != SEED or row.get("producer_sha256", "").lower() != record_sha(record):
            raise CostAuditFailure("COST_PRODUCER_NOT_BOUND_TO_SCIENCE_AUDIT")
    if optional_results:
        if len(optional_results) != 4:
            raise CostAuditFailure("OPTIONAL_RESULTS_MUST_BE_FOUR")
        optional = {file_record(path)["sha256"] for path in optional_results}
        declared = {str(row.get("producer_sha256", "")).lower() for row in conditions}
        if optional != declared:
            raise CostAuditFailure("OPTIONAL_RESULTS_DO_NOT_MATCH_AUDIT_CONDITIONS")
    return conditions


def bulk_prediction_for_route(bulk_arrays: dict, split: str):
    if "prediction" in bulk_arrays:
        return bulk_arrays["prediction"]
    key = split + "_prediction"
    if key not in bulk_arrays:
        raise CostAuditFailure("BULK_PREDICTION_KEY_MISSING:" + split)
    return bulk_arrays[key]


def verify_sources(records: list[dict], failures: list[str], guard=lambda: None) -> int:
    checked = 0
    for record in records:
        guard()
        try:
            verify_file(record)
            checked += 1
            if record.get("snapshot_path"):
                expected = record.get("snapshot_sha256") or record_sha(record)
                verify_file({"path": record["snapshot_path"], "sha256": expected})
                checked += 1
        except Exception as error:
            failures.append("SOURCE_RECORD:" + type(error).__name__ + ":" + str(error))
    return checked


def audit(args: argparse.Namespace) -> int:
    import main_residual_runtime_v1 as runtime

    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    output = resolve(args.out) if args.out else DEFAULT_OUT
    result = {
        "schema": "smallk_main_residual_cost_independent_audit_v1",
        "status": "RUNNING",
        "tag": args.tag,
        "started_unix": time.time(),
        "failures": [],
        "failed_count": 0,
        "checks": 0,
        "planned_rows": 24,
        "whole_rows_checked": 0,
        "timed_samples_checked": 0,
        "metric_rows": 0,
        "GPU_execution": False,
        "F_forward_replayed": False,
        "timing_replayed": False,
        "producer_score_helper_used": False,
        "scope": "saved TIMED_ALL6/reference/bulk arrays, masked macro metrics, Latin6 schedule and timing arithmetic only",
        "audit_source_before": file_record(Path(__file__)),
        "source_records": [file_record(Path(__file__))],
        "resource_authorization": limit.authorization_record,
        "run_manifest": limit.run_manifest_record,
    }
    output_ready = False
    metrics = []
    file_cache: dict[tuple[str, str], dict] = {}
    array_cache: dict[tuple[str, str], dict] = {}
    try:
        limit.checkfull()
        if output.exists():
            raise FileExistsError(str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output_ready = True
        import numpy as np
        limit.checkfull()

        def cached_json(record: dict) -> dict:
            checked = verify_file(record)
            key = (checked["path"], checked["sha256"])
            if key not in file_cache:
                file_cache[key] = read_json(checked["path"])
            return file_cache[key]

        def cached_arrays(record: dict, role: str) -> dict:
            checked = verify_file(record)
            key = (checked["path"], checked["sha256"])
            if key not in array_cache:
                array_cache[key] = load_npz(record, role=role, guard=limit.shared.checkfast)[0]
            return array_cache[key]

        cost_record = file_record(args.cost)
        cost = cached_json(cost_record)
        result["cost_producer"] = cost_record
        result["cost_producer_sha256"] = cost_record["sha256"]
        producer_records = [verify_file(record) for record in cost.get("producers", [])]
        audit_record = verify_file(cost.get("independent_audit", file_record(args.audit)))
        if audit_record["sha256"] != file_record(args.audit)["sha256"]:
            raise CostAuditFailure("COST_AUDIT_RECORD_NOT_ARGUMENT")
        science_audit = cached_json(audit_record)
        bind_scientific_audit(science_audit, cost, producer_records, [resolve(path) for path in args.results or []])
        result["scientific_audit"] = audit_record
        result["conditions"] = [
            {"dataset": row["dataset"], "seed": int(row["seed"]), "producer_sha256": row["producer_sha256"], "status": row["status"]}
            for row in science_audit["conditions"]
        ]
        result["cost_conditions"] = []
        failures = result["failures"]

        def add_check(label: str, value: bool) -> None:
            result["checks"] += 1
            check(value, label, failures)

        add_check("cost_COMPLETE", cost.get("status") == "COMPLETE")
        add_check("actual_complete_rows_24", cost.get("actual_complete_rows") == 24 and len(cost.get("rows", [])) == 24)
        add_check("protocol_ORDER92726_REPEATS6_WARM2", cost.get("order_seed") == ORDER_SEED and cost.get("repeats") == REPEATS and cost.get("warmup_each_route") == WARMUP)
        add_check("no_training_or_refit_in_cost", cost.get("new_training_updates") == 0 and cost.get("coefficient_refit_in_cost") is False)
        add_check("no_test_or_holdout_claim", cost.get("protected_TEST_accessed") is False and cost.get("independent_holdout") is False)
        add_check("source_artifacts_prepost", cost.get("source_artifacts_before_after_SHA_pass") is True)
        add_check("same_auth", cost.get("authorization_record", {}).get("sha256") == limit.authorization_record["sha256"])
        add_check("same_manifest", cost.get("run_manifest_record", {}).get("sha256") == limit.run_manifest_record["sha256"])
        add_check("F_hash_preserved", cost.get("F_before") == cost.get("F_after"))
        add_check("F_no_grad", cost.get("F_has_no_parameter_gradients") is True)
        add_check("codec_and_correction_hashes_preserved", cost.get("model_hashes_before") == cost.get("model_hashes_after"))
        add_check("codecs_no_grad", cost.get("codecs_have_no_parameter_gradients") is True)
        add_check("cost_resource_receipt_present", isinstance(cost.get("resource_receipt"), dict))
        if cost.get("resource_receipt"):
            receipt = cost["resource_receipt"]
            add_check("cost_gpu_receipt_complete", receipt.get("terminal_status") == "COMPLETE" or receipt.get("status") == "COMPLETE")
        result["source_records_checked"] = verify_sources(cost.get("source_records_before", []) + cost.get("source_records_after", []) + cost.get("source_records", []), failures, limit.shared.checkfast)

        producers_by_key = {}
        for record in producer_records:
            producer = cached_json(record)
            producers_by_key[(producer["dataset"], int(producer["seed"]))] = {"record": record, "producer": producer}
            result["cost_conditions"].append({"dataset": producer["dataset"], "seed": int(producer["seed"]), "producer_sha256": record["sha256"], "status": producer.get("status")})
            add_check("cost_producer_complete:" + producer["dataset"], producer.get("status") == "COMPLETE")
        add_check("cost_exact_two_seed92601_producers", set(producers_by_key) == {(dataset, SEED) for dataset in DATASETS})

        expected_keys = {(dataset, batch, route) for dataset in DATASETS for batch in (1, 8) for route in ROUTES}
        rows_by_key = {(row.get("dataset"), int(row.get("batch")), row.get("route")): row for row in cost.get("rows", [])}
        add_check("whole_exact24", set(rows_by_key) == expected_keys and len(rows_by_key) == 24)
        schedule = expected_schedule(np)
        for dataset in DATASETS:
            split = "val" if dataset == "robin" else "eval"
            physical_source = cost["input_provenance"][dataset]["physical_source"]
            raw = cached_arrays(physical_source, dataset + ":physical_source")
            mean = raw["mean"]
            std = raw["std"]
            producer = producers_by_key.get((dataset, SEED), {}).get("producer", {})
            for batch in (1, 8):
                raw_row = rows_by_key[(dataset, batch, "RAW_BOLT")]
                for route in ROUTES:
                    limit.shared.checkfast()
                    row = rows_by_key[(dataset, batch, route)]
                    label = dataset + ":B" + str(batch) + ":" + route
                    before = len(failures)
                    check_timing(row, raw_row, schedule, failures, label)
                    result["checks"] += 11
                    timed = cached_arrays(row["timed_outputs"], label + ":timed")
                    reference = cached_arrays(row["source_reference"], label + ":reference")
                    bulk = cached_arrays(row["original_bulk_source"], label + ":bulk")
                    target = raw["target"][:batch]
                    observed = raw["observed"][:batch]
                    physical_target = target.astype(np.float64) * std + mean
                    bulk_prediction = bulk_prediction_for_route(bulk, split)[:batch]
                    add_check(label + ":reference_route", row["source_reference"].get("route") == route)
                    add_check(label + ":origins", row.get("origins") == raw["origins"][:batch].tolist())
                    add_check(label + ":timed_block", np.array_equal(timed["block"], np.asarray(row["block"], np.int64)))
                    add_check(label + ":timed_position", np.array_equal(timed["position"], np.asarray(row["position"], np.int64)))
                    for scale in ("normalized", "physical"):
                        add_check(label + ":" + scale + ":shape", timed[scale].shape == (REPEATS, batch, 48, target.shape[2]) and reference[scale].shape == (batch, 48, target.shape[2]))
                        add_check(label + ":" + scale + ":dtype", timed[scale].dtype == np.float32 and reference[scale].dtype == np.float32)
                    if "origins" in bulk:
                        add_check(label + ":bulk_origins", np.array_equal(bulk["origins"][:batch], raw["origins"][:batch]))
                    if route in FAMILIES:
                        family = producer.get("families", {}).get(route, {})
                        add_check(label + ":selected_binding", row.get("selected_model_sha256") == family.get("selected_model_sha256") and row.get("selected_alpha") == family.get("selected_alpha"))
                        add_check(label + ":zero_skip_semantics", bool(row.get("selected_zero_skips_feature_and_CPU_transfers")) == (row.get("selected_candidate_id") == "ZERO_CORRECTION"))
                    for repeat in range(REPEATS):
                        add_check(label + ":matched_normalized:" + str(repeat), close_arrays(timed["normalized"][repeat], reference["normalized"], "normalized"))
                        add_check(label + ":matched_physical:" + str(repeat), close_arrays(timed["physical"][repeat], reference["physical"], "physical"))
                        add_check(label + ":declared_matched_normalized:" + str(repeat), row["parity"][repeat]["matched_batch_normalized"]["pass"] == close_arrays(timed["normalized"][repeat], reference["normalized"], "normalized"))
                        add_check(label + ":declared_matched_physical:" + str(repeat), row["parity"][repeat]["matched_batch_physical"]["pass"] == close_arrays(timed["physical"][repeat], reference["physical"], "physical"))
                        bulk_normalized_pass = close_arrays(timed["normalized"][repeat], bulk_prediction, "normalized")
                        bulk_physical_pass = close_arrays(timed["physical"][repeat], bulk_prediction * std + mean, "physical")
                        add_check(label + ":declared_bulk_normalized:" + str(repeat), row["parity"][repeat]["original_bulk_normalized"]["pass"] == bulk_normalized_pass)
                        add_check(label + ":declared_bulk_physical:" + str(repeat), row["parity"][repeat]["original_bulk_physical"]["pass"] == bulk_physical_pass)
                        add_check(label + ":inverse_physical:" + str(repeat), close_arrays(timed["physical"][repeat], timed["normalized"][repeat] * std + mean, "physical"))
                        for scale, prediction, metric_target, score_std in (
                            ("normalized", timed["normalized"][repeat], target, None),
                            ("physical_from_normalized_error", timed["normalized"][repeat], target, std),
                            ("physical_output", timed["physical"][repeat], physical_target, None),
                        ):
                            metrics.append({
                                "dataset": dataset,
                                "batch": batch,
                                "route": route,
                                "repeat": repeat,
                                "block": int(timed["block"][repeat]),
                                "position": int(timed["position"][repeat]),
                                "scale": scale,
                                **masked_score(prediction, metric_target, observed, score_std),
                                "same_first_batch_scope": True,
                            })
                        result["timed_samples_checked"] += 1
                    result["whole_rows_checked"] += 1
        for dataset in DATASETS:
            for batch in (1, 8):
                for block in range(REPEATS):
                    add_check(dataset + ":B" + str(batch) + ":block" + str(block) + ":positions_unique", sorted(rows_by_key[(dataset, batch, route)]["position"][block] for route in ROUTES) == list(range(len(ROUTES))))
        for (path, digest) in file_cache:
            add_check("json_unchanged:" + path, sha(path) == digest)
        for (path, digest) in array_cache:
            add_check("npz_unchanged:" + path, sha(path) == digest)
        result["failed_count"] = len(failures)
        result["metric_rows"] = len(metrics)
        if result["whole_rows_checked"] != 24:
            failures.append("WHOLE_ROWS_CHECKED_COUNT")
        if result["timed_samples_checked"] != 144:
            failures.append("TIMED_SAMPLES_CHECKED_COUNT")
        if result["metric_rows"] != 432:
            failures.append("METRIC_ROWS_COUNT")
        result["failed_count"] = len(failures)
        result["status"] = "PASS" if not failures else "FAILED_PRESERVED"
        limit.checkfull()
    except Exception as error:
        result["status"] = "BUDGET_HELD_PRESERVED" if isinstance(error, runtime.BudgetHeld) else "FAILED_PRESERVED"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_count"] = max(1, len(result.get("failures", [])))
    finally:
        metrics_csv = output.with_name(output.stem + "_metrics.csv")
        if output_ready and metrics:
            with metrics_csv.open("x", encoding="utf-8-sig", newline="") as handle:
                fieldnames = list(metrics[0])
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics:
                    row = dict(row)
                    row["per_channel_counts"] = json.dumps(row["per_channel_counts"])
                    writer.writerow(row)
            result["metrics_csv"] = file_record(metrics_csv)
        source_after = file_record(Path(__file__))
        result["audit_source_after"] = source_after
        result["source_before_after_SHA_pass"] = source_after["sha256"] == result["audit_source_before"]["sha256"]
        if result["source_before_after_SHA_pass"] is not True and result.get("status") == "PASS":
            result["status"] = "FAILED_PRESERVED"
            result["error"] = {"type": "AuditSourceChangedDuringRun", "message": "cost audit source changed before output write"}
            result["failed_count"] = max(1, result.get("failed_count", 0))
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
    print(json.dumps({"status": result["status"], "failed_count": result.get("failed_count"), "whole_rows_checked": result.get("whole_rows_checked"), "metric_rows": result.get("metric_rows"), "out": str(output)}, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--cost", "--cost-result", dest="cost")
    parser.add_argument("--audit", dest="audit")
    parser.add_argument("--results", nargs="*")
    parser.add_argument("--tag", default="main_residual_cost_audit_v1")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({
            "status": "SOURCE_ONLY_NOT_EXECUTED",
            "schema": "smallk_main_residual_cost_independent_audit_v1",
            "planned_rows": 24,
            "timed_samples": 144,
            "metric_rows": 432,
            "routes": ROUTES,
            "repeats": REPEATS,
            "order_seed": ORDER_SEED,
            "runtime_first": True,
            "E_F_or_timing_replay": False,
        }, ensure_ascii=False))
        return 0
    if not args.cost or not args.audit:
        parser.error("--execute requires --cost RESULT.json and --audit MAIN_RESIDUAL_AUDIT_v1_ACTUAL.json")
    if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in args.tag):
        parser.error("tag must be lowercase safe")
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
