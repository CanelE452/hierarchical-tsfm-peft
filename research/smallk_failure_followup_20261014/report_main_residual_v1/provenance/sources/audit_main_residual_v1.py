"""Independent CPU arithmetic audit for main-residual producers.

The auditor does not replay E/F/D, refit ridge solvers, or claim solver optimality.
It checks stored producer bytes, source lineage, fixed candidate ordering, saved
coefficient/application arithmetic, selected predictions, and score arithmetic.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / "validation_v1" / "MAIN_RESIDUAL_AUDIT_v1_ACTUAL.json"
CANONICAL_MATCHED_REPORT_SHA = "1573f4ee628fc5598e41178457937c86137198bb4007b11f3790e2659386c6d2"
DATASET_SPLITS = {"robin": (17, "val"), "beijing_aotizhongxin": (11, "eval")}
SEEDS = (92601, 92602)
ROUTES = ("RAW33C", "E33K", "E512K")
ALPHAS = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
ZERO = "ZERO_CORRECTION"
MEAN = "MEAN_CORRECTION"
CANDIDATE_IDS = (ZERO, MEAN) + tuple("RIDGE_ALPHA_" + str(i) for i in range(len(ALPHAS)))
ATOL, RTOL = 1e-5, 2e-6
STRICT_SCORE_ATOL = 1e-10


class AuditFailure(AssertionError):
    pass


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_record(path: Path | str) -> dict:
    path = resolve(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_file(record: dict) -> dict:
    path = resolve(record["path"])
    actual = sha(path)
    if actual.lower() != str(record["sha256"]).lower():
        raise AuditFailure("FILE_SHA_MISMATCH:" + str(path))
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def record_sha(record: dict) -> str:
    value = record.get("sha256") or record.get("verified_sha256")
    if not value:
        raise AuditFailure("RECORD_SHA_REQUIRED")
    return str(value).lower()


def same_record_marker(left: dict, right: dict, label: str) -> None:
    if record_sha(left) != record_sha(right) or Path(str(left.get("path", ""))).stem != Path(str(right.get("path", ""))).stem:
        raise AuditFailure("RECORD_MARKER_MISMATCH:" + label)


def array_sha_raw(value) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_sha_state(value) -> str:
    import numpy as np

    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha(kind: str, *values) -> str:
    import numpy as np

    digest = hashlib.sha256(kind.encode())
    for value in values:
        digest.update((array_sha_state(value) if isinstance(value, np.ndarray) else str(value)).encode())
    return digest.hexdigest()


def load_npz(record: dict) -> tuple[dict, dict]:
    import numpy as np

    checked = verify_file(record)
    with np.load(checked["path"], allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    declared = record.get("arrays")
    if declared is not None:
        if set(arrays) != set(declared):
            raise AuditFailure("NPZ_ARRAY_KEY_MISMATCH:" + checked["path"])
        for key, value in arrays.items():
            meta = declared[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or array_sha_raw(value) != meta["sha256"]:
                raise AuditFailure("NPZ_ARRAY_CONTENT_MISMATCH:" + checked["path"] + ":" + key)
    return arrays, checked


def exact_array(left, right, label: str, *, equal_nan=False) -> None:
    import numpy as np

    if (
        left.dtype != right.dtype
        or left.shape != right.shape
        or not np.array_equal(left, right, equal_nan=equal_nan)
        or array_sha_raw(left) != array_sha_raw(right)
    ):
        raise AuditFailure("ARRAY_EXACT_MISMATCH:" + label)


def close_array(left, right, label: str, *, atol=ATOL, rtol=RTOL) -> dict:
    import numpy as np

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise AuditFailure("ARRAY_SHAPE_MISMATCH:" + label)
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    tol = atol + rtol * np.abs(right.astype(np.float64))
    ok = bool(np.all(diff <= tol))
    if not ok:
        raise AuditFailure("ARRAY_NUMERIC_MISMATCH:" + label + ":max=" + str(float(diff.max())))
    return {"max_abs": float(diff.max()) if diff.size else 0.0, "atol": atol, "rtol": rtol}


def observed_score(prediction, target, observed, std=None) -> dict:
    import numpy as np

    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * std
    error = np.where(observed, error, 0.0)
    count = observed.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error**2).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    return {
        "mse": float(mse[active].mean()) if active.any() else None,
        "mae": float(mae[active].mean()) if active.any() else None,
        "per_channel_mse": [float(v) if a else None for v, a in zip(mse, active)],
        "per_channel_mae": [float(v) if a else None for v, a in zip(mae, active)],
        "observed_counts": count.astype(int).tolist(),
        "active_channels": int(active.sum()),
    }


def assert_score(actual: dict, expected: dict, label: str) -> None:
    for key in ("mse", "mae"):
        if actual[key] is None and expected[key] is None:
            continue
        if abs(float(actual[key]) - float(expected[key])) > STRICT_SCORE_ATOL:
            raise AuditFailure("SCORE_MISMATCH:" + label + ":" + key)
    for key in ("per_channel_mse", "per_channel_mae"):
        actual_values = actual.get(key)
        expected_values = expected.get(key)
        if len(actual_values) != len(expected_values):
            raise AuditFailure("SCORE_PER_CHANNEL_LENGTH_MISMATCH:" + label + ":" + key)
        for index, (actual_value, expected_value) in enumerate(zip(actual_values, expected_values)):
            if actual_value is None and expected_value is None:
                continue
            if actual_value is None or expected_value is None or abs(float(actual_value) - float(expected_value)) > STRICT_SCORE_ATOL:
                raise AuditFailure("SCORE_PER_CHANNEL_MISMATCH:" + label + ":" + key + ":" + str(index))
    if actual.get("observed_counts") != expected.get("observed_counts"):
        raise AuditFailure("SCORE_OBSERVED_COUNTS_MISMATCH:" + label)
    if int(actual.get("active_channels")) != int(expected.get("active_channels")):
        raise AuditFailure("SCORE_ACTIVE_CHANNELS_MISMATCH:" + label)


def past_features(route: str, past):
    import numpy as np

    value = np.asarray(past)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[1] != 512:
        raise AuditFailure("PAST_FEATURE_INPUT_CONTRACT:" + route)
    if route == "E512K":
        return value.reshape(len(value), -1)
    if route in ("RAW33C", "E33K"):
        return np.concatenate((value.reshape(len(value), 32, 16, value.shape[2]).mean(axis=2), value[:, -1:]), axis=1).reshape(len(value), -1)
    raise AuditFailure("UNKNOWN_ROUTE:" + route)


def transformed_features(route: str, past, feature_arrays: dict, feature_meta: dict):
    import numpy as np

    raw = past_features(route, past)
    center, scale = feature_arrays["center"], feature_arrays["scale"]
    if center.dtype != np.float64 or scale.dtype != np.float64 or center.shape != scale.shape or raw.shape[1] != center.shape[0]:
        raise AuditFailure("FEATURE_STATE_SHAPE_DTYPE:" + route)
    if feature_meta["route"] != route or feature_meta["feature_count"] != center.shape[0]:
        raise AuditFailure("FEATURE_STATE_METADATA:" + route)
    return (raw.astype(np.float64) - center) / scale


def solution_sha(kind, alpha, arrays, feature_state_sha256):
    return state_sha(kind, alpha, arrays["coefficients"], arrays["intercept"], arrays["observed_counts"], arrays["channel_counts"], feature_state_sha256)


def model_sha(feature_sha, channels, candidate_id, solution_sha_value, mean_sha_value):
    return state_sha("FROZEN_MAIN_PLUS_PAST_E_CORRECTION", feature_sha, channels, candidate_id, solution_sha_value, mean_sha_value)


def apply_model(route: str, model_meta: dict, feature_arrays: dict, coeff_arrays: dict | None, mean_arrays: dict | None, past, main_forecast):
    import numpy as np

    main = np.asarray(main_forecast)
    if main.dtype != np.float32 or main.ndim != 3 or main.shape[1] != 48 or not np.isfinite(main).all():
        raise AuditFailure("MAIN_FORECAST_CONTRACT:" + route)
    candidate_id = model_meta["candidate_id"]
    if candidate_id == ZERO:
        return main.copy()
    if candidate_id == MEAN:
        correction = mean_arrays["correction"]
        if correction.dtype != np.float32 or correction.shape != main.shape[1:]:
            raise AuditFailure("MEAN_CORRECTION_SHAPE_DTYPE")
        return main + correction
    design = transformed_features(route, past, feature_arrays, model_meta)
    coefficients = coeff_arrays["coefficients"]
    intercept = coeff_arrays["intercept"]
    if coefficients.dtype != np.float64 or intercept.dtype != np.float64:
        raise AuditFailure("RIDGE_COEFF_DTYPE:" + route + ":" + candidate_id)
    correction = (design @ coefficients.reshape(design.shape[1], -1) + intercept.reshape(-1)).reshape((len(design),) + intercept.shape).astype(np.float32)
    return main + correction


def artifact_by_stem(producer: dict, stem: str) -> dict:
    matches = [record for record in producer.get("artifacts", []) if Path(record["path"]).stem == stem]
    if len(matches) != 1:
        raise AuditFailure("EXACT_ARTIFACT_STEM_REQUIRED:" + stem)
    return matches[0]


def verify_source_records(records: list[dict]) -> int:
    checked = 0
    for record in records:
        if not record.get("path") or not (record.get("sha256") or record.get("verified_sha256")):
            raise AuditFailure("SOURCE_RECORD_SCHEMA")
        verify_file({"path": record["path"], "sha256": record_sha(record)})
        checked += 1
        if record.get("snapshot_path"):
            verify_file({"path": record["snapshot_path"], "sha256": record.get("snapshot_sha256") or record_sha(record)})
            checked += 1
    return checked


def verify_canonical_matched_chain(producer: dict, producer_record: dict, detail: dict) -> dict:
    matched_record = producer.get("matched_source")
    if not matched_record:
        raise AuditFailure("MATCHED_SOURCE_RECORD_REQUIRED")
    matched_checked = verify_file(matched_record)
    matched = read_json(matched_checked["path"])
    if matched.get("status") != "COMPLETE" or matched.get("dataset") != producer["dataset"] or int(matched.get("seed")) != int(producer["seed"]):
        raise AuditFailure("MATCHED_SOURCE_IDENTITY_STATUS")
    canonical_records = [r for r in producer.get("source_records", []) if str(r.get("sha256", "")).lower() == CANONICAL_MATCHED_REPORT_SHA]
    if len(canonical_records) != 1:
        raise AuditFailure("CANONICAL_MATCHED_REPORT_1573_RECORD_REQUIRED")
    manifest_record = verify_file(canonical_records[0])
    manifest = read_json(manifest_record["path"])
    if manifest.get("status") != "COMPLETE" or manifest.get("audit_gate", {}).get("passed") is not True:
        raise AuditFailure("CANONICAL_MATCHED_REPORT_GATE_NOT_PASS")
    if not any(r.get("role") == "producer" and str(r.get("sha256", "")).lower() == matched_checked["sha256"] for r in manifest.get("input_records", [])):
        raise AuditFailure("CANONICAL_MATCHED_REPORT_NOT_BOUND_TO_MATCHED_PRODUCER")
    audit_records = [r for r in manifest.get("input_records", []) if r.get("role") == "independent_audit"]
    if len(audit_records) != 1:
        raise AuditFailure("CANONICAL_MATCHED_AUDIT_RECORD_REQUIRED")
    audit_record = verify_file({**audit_records[0], "sha256": str(audit_records[0]["sha256"]).lower()})
    audit = read_json(audit_record["path"])
    matches = [r for r in audit.get("conditions", []) if r.get("dataset") == producer["dataset"] and int(r.get("seed")) == int(producer["seed"])]
    if audit.get("status") != "PASS" or audit.get("failed_count") != 0 or len(matches) != 1 or str(matches[0].get("producer_sha256", "")).lower() != matched_checked["sha256"]:
        raise AuditFailure("MATCHED_AUDIT_CONDITION_BINDING_REQUIRED")
    detail["matched_source"] = matched_checked
    detail["canonical_matched_report"] = manifest_record
    detail["canonical_matched_audit"] = audit_record
    return matched


def verify_source_snapshots(producer: dict) -> int:
    count = 0
    for record in producer.get("source_snapshots", []):
        expected = str(record["sha256"]).lower()
        if sha(resolve(record["path"])).lower() != expected or sha(resolve(record["snapshot_path"])).lower() != expected:
            raise AuditFailure("SOURCE_SNAPSHOT_CHANGED:" + str(record.get("path")))
        count += 1
    return count


def verify_reference_arrays(producer: dict, split: str, raw_arrays: dict, selected_arrays: dict, zero_arrays: dict) -> None:
    for name in ("target", "observed", "origins", "mean", "std"):
        expected = raw_arrays[name] if name in raw_arrays else selected_arrays[name]
        exact_array(selected_arrays[name], expected, split + ":selected:" + name, equal_nan=(name == "target"))
    exact_array(zero_arrays["prediction"], selected_arrays["original_MAIN_prediction"], split + ":zero-is-main")
    exact_array(zero_arrays["origins"], raw_arrays["origins"], split + ":zero-origins")


def verify_coeff_contract(route: str, entry: dict, arrays: dict, channels: int, feature_count: int, fit_observed=None) -> None:
    import numpy as np

    expected_shapes = {
        "coefficients": [feature_count, 48, channels],
        "intercept": [48, channels],
        "observed_counts": [48, channels],
        "channel_counts": [channels],
        "normal_equation_denominators": [48, channels],
        "design_ranks": [48, channels],
        "regularized_gram_conditions": [48, channels],
    }
    for name, shape in expected_shapes.items():
        if list(arrays[name].shape) != shape:
            raise AuditFailure("COEFFICIENT_SHAPE:" + route + ":" + entry["candidate_id"] + ":" + name)
    for name in ("coefficients", "intercept", "normal_equation_denominators", "regularized_gram_conditions"):
        if arrays[name].dtype != np.float64:
            raise AuditFailure("COEFFICIENT_FLOAT64_REQUIRED:" + name)
    for name in ("observed_counts", "channel_counts", "design_ranks"):
        if arrays[name].dtype != np.int64:
            raise AuditFailure("COEFFICIENT_INT64_REQUIRED:" + name)
    denom_expected = np.broadcast_to(arrays["channel_counts"].reshape(1, channels), (48, channels)).astype(np.float64)
    finite = np.isfinite(arrays["normal_equation_denominators"])
    if not np.array_equal(arrays["normal_equation_denominators"][finite], denom_expected[finite]):
        raise AuditFailure("DENOMINATOR_MUST_BE_NC_NOT_NHC:" + route + ":" + entry["candidate_id"])
    if fit_observed is not None:
        exact_array(arrays["observed_counts"], fit_observed.sum(axis=0).astype(np.int64), "coeff-observed-counts:" + route + ":" + entry["candidate_id"])
        exact_array(arrays["channel_counts"], fit_observed.sum(axis=(0, 1)).astype(np.int64), "coeff-channel-counts:" + route + ":" + entry["candidate_id"])
    if arrays["design_ranks"].min() < 0 or arrays["design_ranks"].max() > feature_count:
        raise AuditFailure("DESIGN_RANK_BOUNDS:" + route + ":" + entry["candidate_id"])
    zero_outputs = arrays["observed_counts"].reshape(-1) == 0
    if zero_outputs.any():
        flat_coeff = arrays["coefficients"].reshape(feature_count, -1)
        flat_intercept = arrays["intercept"].reshape(-1)
        if not np.array_equal(flat_coeff[:, zero_outputs], np.zeros_like(flat_coeff[:, zero_outputs])) or not np.array_equal(flat_intercept[zero_outputs], np.zeros_like(flat_intercept[zero_outputs])):
            raise AuditFailure("ZERO_OBSERVED_OUTPUT_WEIGHTS_MUST_BE_ZERO:" + route + ":" + entry["candidate_id"])


def select_candidate(rows: list[dict]) -> int:
    import numpy as np

    if tuple(row.get("candidate_id") for row in rows) != CANDIDATE_IDS:
        raise AuditFailure("EXACT_9_CANDIDATE_ORDER_REQUIRED")
    if tuple(row.get("alpha") for row in rows) != (None, None) + ALPHAS:
        raise AuditFailure("ZERO_MEAN_THEN_SEVEN_ALPHA_ORDER_REQUIRED")
    eligible = []
    for index, row in enumerate(rows):
        if row.get("status") == "FAILED_PRESERVED" and index != 0:
            continue
        value = row.get("tune_actual_Y_mse")
        if row.get("status") != "COMPLETE" or value is None or not np.isfinite(value):
            raise AuditFailure("FINITE_COMPLETE_TUNE_MSE_REQUIRED_FOR_SELECTION")
        eligible.append(index)
    return min(eligible, key=lambda idx: rows[idx]["tune_actual_Y_mse"])


def audit_condition(result_path: Path, np, detail: dict, guard=lambda: None) -> dict:
    guard()
    producer_record = file_record(result_path)
    producer = read_json(result_path)
    row = {
        "dataset": producer.get("dataset"),
        "seed": int(producer.get("seed")),
        "producer_path": str(result_path),
        "producer_sha256": producer_record["sha256"],
        "status": "RUNNING",
        "checks": [],
        "failures": [],
        "candidate_rows_checked": 0,
        "new_ridge_candidates_checked": 0,
        "selected_predictions_checked": 0,
    }
    if producer.get("status") != "COMPLETE":
        raise AuditFailure("PRODUCER_NOT_COMPLETE:" + str(result_path))
    if producer.get("maximum_new_ridge_coefficients") != 21 or producer.get("maximum_candidate_comparisons") != 27:
        raise AuditFailure("PRODUCER_BOUND_COUNTS_CHANGED")
    if producer.get("shared_mean_fits_per_condition") != 1:
        raise AuditFailure("SHARED_MEAN_MUST_BE_FIT_ONCE_PER_CONDITION")
    dataset = row["dataset"]
    if dataset not in DATASET_SPLITS or row["seed"] not in SEEDS:
        raise AuditFailure("UNKNOWN_DATASET_SEED")
    _, eval_split = DATASET_SPLITS[dataset]
    splits = ("fit", "tune", eval_split)
    if producer.get("GPU_execution") is not False or producer.get("E_D_F_loaded") is not False or producer.get("E_D_F_forward_replayed") is not False:
        raise AuditFailure("CPU_PRODUCER_MUST_NOT_LOAD_OR_REPLAY_EFD")
    matched = verify_canonical_matched_chain(producer, producer_record, row)
    row["source_records_checked"] = verify_source_records(producer.get("source_records", []))
    row["source_snapshots_checked"] = verify_source_snapshots(producer)
    if producer.get("original_CP_file_hash_only", {}).get("sha256") != matched.get("original_CP_file_hash_only", {}).get("sha256"):
        raise AuditFailure("ORIGINAL_CP_SHA_NOT_MATCHED")
    if record_sha(producer.get("matched_source", {})) != record_sha(row["matched_source"]):
        raise AuditFailure("MATCHED_SOURCE_FIELD_NOT_BOUND")

    raw = {}
    cached = {}
    for split in splits:
        guard()
        raw_record = producer["input_artifact_refs"][split]["raw"]
        same_record_marker(raw_record, artifact_by_stem(matched, "RAW_INPUT_" + split), "matched-raw:" + split)
        same_record_marker(producer["input_artifact_refs"][split]["original_MAIN"], producer["original_MAIN_prediction_source"], "original-main-source:" + split)
        raw[split], _ = load_npz(raw_record)
        cached_record = producer["input_artifact_refs"][split].get("cached_E")
        if cached_record:
            same_record_marker(cached_record, artifact_by_stem(matched, "CACHED_E_INPUT_" + split), "matched-cached-E:" + split)
            cached[split], _ = load_npz(cached_record)
            for name in ("target", "observed", "origins", "raw_last", "history_observed", "raw_forecast", "mean", "std"):
                exact_array(cached[split][name], raw[split][name], "cached-vs-raw:" + split + ":" + name, equal_nan=(name == "target"))
    mean = raw["fit"]["mean"]
    std = raw["fit"]["std"]
    channels = int(raw["fit"]["target"].shape[2])

    mean_arrays, _ = load_npz(producer["controls"][MEAN]["coefficient_artifact"])
    if mean_arrays["correction"].dtype != np.float32 or mean_arrays["observed_counts"].dtype != np.int64:
        raise AuditFailure("MEAN_CONTROL_DTYPE")

    controls = producer["controls"]
    fit_zero_arrays, _ = load_npz(controls[ZERO]["artifacts"]["fit"])
    fit_residual = raw["fit"]["target"] - fit_zero_arrays["prediction"]
    fit_residual_sha = array_sha_state(fit_residual)
    fit_observed_sha = array_sha_state(raw["fit"]["observed"])
    if controls[MEAN]["fit_residual_sha256"] != fit_residual_sha or controls[MEAN]["fit_observed_sha256"] != fit_observed_sha:
        raise AuditFailure("MEAN_CONTROL_FIT_RESIDUAL_MASK_SHA")
    expected_counts = raw["fit"]["observed"].sum(axis=0).astype(np.int64)
    expected_total = np.where(raw["fit"]["observed"], fit_residual.astype(np.float64), 0.0).sum(axis=0)
    expected_mean = np.divide(expected_total, expected_counts, out=np.zeros_like(expected_total), where=expected_counts > 0).astype(np.float32)
    exact_array(mean_arrays["observed_counts"], expected_counts, "mean-control-observed-counts")
    exact_array(mean_arrays["correction"], expected_mean, "mean-control-correction")
    mean_control_sha = state_sha("MAIN_RESIDUAL_FIT_INTERCEPT_ONLY", mean_arrays["correction"], mean_arrays["observed_counts"], fit_residual_sha, fit_observed_sha)
    if controls[MEAN]["model_sha256"] != mean_control_sha:
        raise AuditFailure("MEAN_CONTROL_MODEL_SHA")

    for split in splits:
        guard()
        zero_arrays, _ = load_npz(controls[ZERO]["artifacts"][split])
        mean_pred_arrays, _ = load_npz(controls[MEAN]["artifacts"][split])
        main = zero_arrays["prediction"]
        if split in cached:
            exact_array(main, cached[split]["original_prediction"], split + ":zero-equals-cached-original-main")
        expected_mean = main + mean_arrays["correction"]
        exact_array(mean_pred_arrays["origins"], raw[split]["origins"], split + ":mean-origins")
        close_array(mean_pred_arrays["prediction"], expected_mean, split + ":mean-prediction")
        assert_score(controls[ZERO]["splits"][split]["normalized"], observed_score(main, raw[split]["target"], raw[split]["observed"]), split + ":zero-norm")
        assert_score(controls[MEAN]["splits"][split]["normalized"], observed_score(mean_pred_arrays["prediction"], raw[split]["target"], raw[split]["observed"]), split + ":mean-norm")
        assert_score(controls[ZERO]["splits"][split]["physical_per_channel_diagnostic"], observed_score(main, raw[split]["target"], raw[split]["observed"], std), split + ":zero-phys")
        assert_score(controls[MEAN]["splits"][split]["physical_per_channel_diagnostic"], observed_score(mean_pred_arrays["prediction"], raw[split]["target"], raw[split]["observed"], std), split + ":mean-phys")
        refs = producer.get("references", {}).get(split, {})
        if "RAW_BOLT" in refs:
            assert_score(refs["RAW_BOLT"]["normalized"], observed_score(raw[split]["raw_forecast"], raw[split]["target"], raw[split]["observed"]), split + ":RAW_BOLT-norm")
            assert_score(refs["RAW_BOLT"]["physical_per_channel_diagnostic"], observed_score(raw[split]["raw_forecast"], raw[split]["target"], raw[split]["observed"], std), split + ":RAW_BOLT-phys")
        if "ORIGINAL_MAIN" in refs:
            assert_score(refs["ORIGINAL_MAIN"]["normalized"], observed_score(main, raw[split]["target"], raw[split]["observed"]), split + ":ORIGINAL_MAIN-norm")
            assert_score(refs["ORIGINAL_MAIN"]["physical_per_channel_diagnostic"], observed_score(main, raw[split]["target"], raw[split]["observed"], std), split + ":ORIGINAL_MAIN-phys")

    for route in ROUTES:
        guard()
        family = producer["families"].get(route)
        if not family or family.get("status") not in ("COMPLETE", "FAILED_PRESERVED"):
            raise AuditFailure("ROUTE_STATUS_UNAVAILABLE:" + route)
        feature_arrays, _ = load_npz(family["feature_state"]["artifact"])
        feature_meta = family["feature_state"]
        past_by_split = {split: raw[split]["history"] for split in splits} if route == "RAW33C" else {split: cached[split]["latent_past"] for split in splits}
        if route != "RAW33C" and set(past_by_split) != set(splits):
            raise AuditFailure("CACHED_E_ROUTE_INPUTS_REQUIRED:" + route)
        fit_raw_features = past_features(route, past_by_split["fit"])
        values = fit_raw_features.astype(np.float64)
        expected_center = values.mean(axis=0)
        expected_scale = np.sqrt(((values - expected_center) ** 2).mean(axis=0))
        expected_scale[expected_scale == 0] = 1.0
        exact_array(feature_arrays["center"], expected_center, route + ":feature-center")
        exact_array(feature_arrays["scale"], expected_scale, route + ":feature-scale")
        if feature_meta["fit_raw_features_sha256"] != array_sha_state(fit_raw_features):
            raise AuditFailure("FEATURE_STATE_FIT_RAW_FEATURES_SHA:" + route)
        expected_feature_sha = state_sha("MAIN_RESIDUAL_FIT_FEATURE_STATE", route, feature_arrays["center"], feature_arrays["scale"],
                                           feature_meta["input_channels"], feature_meta["fit_rows"], feature_meta["fit_raw_features_sha256"])
        if expected_feature_sha != feature_meta["sha256"]:
            raise AuditFailure("FEATURE_STATE_SHA:" + route)
        grid = family.get("candidate_grid", [])
        selected_models: list[dict | None] = []
        for index, entry in enumerate(grid):
            guard()
            row["candidate_rows_checked"] += 1
            candidate_id = entry.get("candidate_id")
            if candidate_id != CANDIDATE_IDS[index]:
                raise AuditFailure("CANDIDATE_ORDER:" + route)
            if candidate_id == ZERO:
                model_meta = entry["model_metadata"]
                if model_sha(model_meta["feature_state_sha256"], model_meta["channels"], candidate_id, None, None) != model_meta["model_sha256"]:
                    raise AuditFailure("ZERO_MODEL_SHA:" + route)
                coeff_arrays = None
                mean_for_model = None
            elif candidate_id == MEAN:
                model_meta = entry["model_metadata"]
                coeff_arrays = None
                mean_for_model = mean_arrays
                if model_sha(model_meta["feature_state_sha256"], model_meta["channels"], candidate_id, None, mean_control_sha) != model_meta["model_sha256"]:
                    raise AuditFailure("MEAN_MODEL_SHA:" + route)
            elif entry.get("status") == "COMPLETE":
                row["new_ridge_candidates_checked"] += 1
                coeff_arrays, _ = load_npz(entry["coefficient_artifact"])
                feature_count = int(feature_arrays["center"].shape[0])
                verify_coeff_contract(route, entry, coeff_arrays, channels, feature_count, raw["fit"]["observed"])
                model_meta = entry["model_metadata"]
                sol_sha = solution_sha(model_meta["kind"], model_meta["alpha"], coeff_arrays, model_meta["feature_state_sha256"])
                if sol_sha != model_meta["solution_sha256"]:
                    raise AuditFailure("RIDGE_SOLUTION_SHA:" + route + ":" + candidate_id)
                if model_sha(model_meta["feature_state_sha256"], model_meta["channels"], candidate_id, sol_sha, None) != model_meta["model_sha256"]:
                    raise AuditFailure("RIDGE_MODEL_SHA:" + route + ":" + candidate_id)
                mean_for_model = None
            else:
                selected_models.append(None)
                continue
            if entry.get("model_sha256") != model_meta["model_sha256"]:
                raise AuditFailure("CANDIDATE_TOPLEVEL_MODEL_SHA:" + route + ":" + candidate_id)
            for split in ("fit", "tune"):
                guard()
                artifact = entry[split + "_artifact"]
                pred_arrays, _ = load_npz(artifact)
                expected = apply_model(route, model_meta, feature_arrays, coeff_arrays, mean_for_model, past_by_split[split], controls[ZERO]["artifacts"][split] and load_npz(controls[ZERO]["artifacts"][split])[0]["prediction"])
                if candidate_id == ZERO:
                    exact_array(pred_arrays["prediction"], expected, route + ":" + candidate_id + ":" + split)
                else:
                    close_array(pred_arrays["prediction"], expected, route + ":" + candidate_id + ":" + split)
                exact_array(pred_arrays["origins"], raw[split]["origins"], route + ":" + candidate_id + ":" + split + ":origins")
                score = observed_score(expected, raw[split]["target"], raw[split]["observed"])
                assert_score(entry[split + "_scores"]["normalized"], score, route + ":" + candidate_id + ":" + split)
                assert_score(entry[split + "_scores"]["physical_per_channel_diagnostic"], observed_score(expected, raw[split]["target"], raw[split]["observed"], std), route + ":" + candidate_id + ":" + split + ":physical")
                if split == "tune" and abs(float(entry["tune_actual_Y_mse"]) - float(score["mse"])) > STRICT_SCORE_ATOL:
                    raise AuditFailure("TUNE_MSE_FIELD:" + route + ":" + candidate_id)
            selected_models.append({"meta": model_meta, "coeff": coeff_arrays, "mean": mean_for_model})
        if len(grid) != 9:
            raise AuditFailure("NINE_CANDIDATES_REQUIRED:" + route)
        selected_index = select_candidate(grid)
        if selected_index != family.get("selected_index") or grid[selected_index]["candidate_id"] != family.get("selected_candidate_id"):
            raise AuditFailure("SELECTED_CANDIDATE_MISMATCH:" + route)
        selected = selected_models[selected_index]
        if selected is None:
            raise AuditFailure("SELECTED_MODEL_FAILED_SLOT:" + route)
        if selected["meta"]["model_sha256"] != family.get("selected_model_sha256") or selected["meta"] != family.get("selected_model_metadata"):
            raise AuditFailure("SELECTED_MODEL_METADATA_SHA:" + route)
        if family.get("selected_coefficient_artifact") != grid[selected_index].get("coefficient_artifact"):
            raise AuditFailure("SELECTED_COEFFICIENT_ARTIFACT_BINDING:" + route)
        for split in splits:
            guard()
            selected_arrays, _ = load_npz(family["selected_artifacts"][split])
            main = load_npz(controls[ZERO]["artifacts"][split])[0]["prediction"]
            expected = apply_model(route, selected["meta"], feature_arrays, selected["coeff"], selected["mean"], past_by_split[split], main)
            if selected["meta"]["candidate_id"] == ZERO:
                exact_array(selected_arrays["prediction"], expected, route + ":selected:" + split)
            else:
                close_array(selected_arrays["prediction"], expected, route + ":selected:" + split)
            exact_array(selected_arrays["original_MAIN_prediction"], main, route + ":selected-main:" + split)
            verify_reference_arrays(producer, split, raw[split], selected_arrays, {"prediction": main, "origins": raw[split]["origins"]})
            assert_score(family["splits"][split]["normalized"], observed_score(expected, raw[split]["target"], raw[split]["observed"]), route + ":selected-norm:" + split)
            assert_score(family["splits"][split]["physical_per_channel_diagnostic"], observed_score(expected, raw[split]["target"], raw[split]["observed"], std), route + ":selected-phys:" + split)
            row["selected_predictions_checked"] += 1
    guard()
    row["status"] = "PASS"
    row["checks"].append("producer_bytes_chain_and_arithmetic_pass")
    return row


def execute(args: argparse.Namespace) -> int:
    import main_residual_runtime_v1 as runtime

    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    output = resolve(args.out) if args.out else DEFAULT_OUT
    result = {
        "schema": "smallk_main_residual_independent_audit_v1",
        "status": "RUNNING",
        "tag": args.tag,
        "started_unix": time.time(),
        "conditions": [],
        "failures": [],
        "expected_conditions": 4,
        "audit_scope": "independent CPU array arithmetic only; no E/F/D replay and no solver optimality proof",
        "audit_source": file_record(Path(__file__)),
        "audit_source_before": file_record(Path(__file__)),
        "source_records": [file_record(Path(__file__))],
        "source_before_after_SHA_pass": None,
        "resource_authorization": limit.authorization_record,
        "run_manifest": limit.run_manifest_record,
        "guard_adapter": limit.guard_adapter_record,
    }
    output_ready = False
    try:
        if output.exists():
            raise FileExistsError(str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output_ready = True
        limit.checkfull()
        import numpy as np
        limit.checkfull()
        result_paths = [resolve(value) for value in args.results]
        if len(result_paths) != 4:
            raise AuditFailure("EXACT_FOUR_MAIN_RESIDUAL_RESULTS_REQUIRED")
        seen = set()
        for path in result_paths:
            if str(path) in seen:
                raise AuditFailure("DUPLICATE_RESULT_PATH:" + str(path))
            seen.add(str(path))
        for path in result_paths:
            limit.checkfull()
            condition = {"producer_path": str(path), "status": "RUNNING"}
            try:
                condition = audit_condition(path, np, condition, limit.shared.checkfast)
            except runtime.BudgetHeld:
                raise
            except Exception as error:
                condition["status"] = "FAILED_PRESERVED"
                condition["failure"] = {"type": type(error).__name__, "message": str(error)}
                result["failures"].append(condition["failure"] | {"producer_path": str(path)})
            result["conditions"].append(condition)
            limit.checkfull()
        expected = {(dataset, seed) for dataset in DATASET_SPLITS for seed in SEEDS}
        actual = {(row.get("dataset"), row.get("seed")) for row in result["conditions"] if row.get("dataset") and row.get("seed")}
        if actual != expected:
            result["failures"].append({"type": "ConditionCoverageMismatch", "message": str(sorted(actual))})
        result["candidate_rows_checked"] = int(sum(row.get("candidate_rows_checked", 0) for row in result["conditions"]))
        result["new_ridge_candidates_checked"] = int(sum(row.get("new_ridge_candidates_checked", 0) for row in result["conditions"]))
        result["selected_predictions_checked"] = int(sum(row.get("selected_predictions_checked", 0) for row in result["conditions"]))
        result["comparison_pool_expected"] = 108
        result["new_ridge_expected"] = 84
        result["selected_predictions_expected"] = 36
        result["producer_sha_binding"] = [
            {"dataset": row.get("dataset"), "seed": row.get("seed"), "producer_sha256": row.get("producer_sha256"), "status": row.get("status")}
            for row in result["conditions"]
        ]
        if result["candidate_rows_checked"] != result["comparison_pool_expected"]:
            result["failures"].append({"type": "CandidatePoolCountMismatch", "message": str(result["candidate_rows_checked"])})
        if result["new_ridge_candidates_checked"] != result["new_ridge_expected"]:
            result["failures"].append({"type": "NewRidgeCountMismatch", "message": str(result["new_ridge_candidates_checked"])})
        if result["selected_predictions_checked"] != result["selected_predictions_expected"]:
            result["failures"].append({"type": "SelectedPredictionCountMismatch", "message": str(result["selected_predictions_checked"])})
        failed = [row for row in result["conditions"] if row.get("status") != "PASS"]
        if result["failures"] or failed or len(result["conditions"]) != 4:
            result["status"] = "FAILED_PRESERVED"
        else:
            result["status"] = "PASS"
        result["failed_count"] = len(result["failures"]) + len(failed)
        limit.checkfull()
    except Exception as error:
        result["status"] = "BUDGET_HELD_PRESERVED" if isinstance(error, runtime.BudgetHeld) else "FAILED_PRESERVED"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_count"] = max(1, len(result.get("failures", [])))
    finally:
        source_after = file_record(Path(__file__))
        result["audit_source_after"] = source_after
        result["source_before_after_SHA_pass"] = source_after["sha256"] == result.get("audit_source", {}).get("sha256")
        if result["source_before_after_SHA_pass"] is not True and result.get("status") == "PASS":
            result["status"] = "FAILED_PRESERVED"
            result["failure"] = {"type": "AuditSourceChangedDuringRun", "message": "audit source SHA changed before output write"}
            result["failed_count"] = max(1, result.get("failed_count", 0))
        close_status = "COMPLETE" if result.get("status") == "PASS" else result.get("status", "FAILED_PRESERVED")
        receipt = limit.close(close_status, result.get("failure") or (result.get("failures") if result.get("status") != "PASS" else None))
        result["resource_receipt"] = receipt
        if receipt.get("terminal_status") != "COMPLETE" and result.get("status") == "PASS":
            result["status"] = receipt["terminal_status"]
            result["resource_terminal_error"] = receipt.get("terminal_error")
            result["failed_count"] = max(1, result.get("failed_count", 0))
        result["finished_unix"] = time.time()
        if output_ready:
            write_new_json(output, result)
    print(json.dumps({"status": result["status"], "failed_count": result.get("failed_count"), "out": str(output)}, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--results", nargs="+", help="exact four main-residual RESULT.json files")
    parser.add_argument("--tag", default="main_residual_audit_v1")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({"status": "SOURCE_ONLY_NOT_EXECUTED", "audit_schema": "smallk_main_residual_independent_audit_v1", "expected_conditions": 4, "expected_candidate_pool": 108, "expected_new_ridge": 84}, ensure_ascii=False))
        return 0
    if not args.results:
        parser.error("--execute requires --results FOUR")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
