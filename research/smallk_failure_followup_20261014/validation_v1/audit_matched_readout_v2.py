"""Independent CPU audit for matched-readout producers.

Default import/describe is source-only. Actual RESULT/NPZ reads occur only under
--execute, after the fresh matched-readout runtime is constructed and the
protected TEST guard is installed. This audit verifies saved arrays and
metadata; it does not rerun SVD/lstsq fitting, E/F/D forward passes, GPU work,
or old campaign jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
VALIDATION = HERE / "validation_v1"
DEFAULT_OUT = VALIDATION / "MATCHED_READOUT_AUDIT_v2.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ALPHAS = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
LENGTH = 512
HORIZON = 48
ATOL = 1e-5
RTOL = 2e-6
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
NEW_ROUTES = ("RAW33C", "E512K")
REUSED_ROUTE = "E33K_REUSED"
EXPECTED_CONDITIONS = {(dataset, seed) for dataset in DATASETS for seed in (92601, 92602)}
EXPECTED_TOTALS = {
    "new_candidate_rows_checked": 56,
    "candidate_fit_tune_predictions_checked": 112,
    "selected_predictions_checked": 36,
    "reused_models_checked": 4,
}


class AuditFailure(Exception):
    pass


class Guard:
    def __init__(self, limit: Any):
        self.limit = limit
        self.count = 0

    def fast(self) -> None:
        self.limit.shared.checkfast()

    def full(self) -> None:
        self.limit.checkfull()

    def artifact(self) -> None:
        self.count += 1
        if self.count % 16 == 0:
            self.full()
        else:
            self.fast()


def sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str, guard: Guard | None = None) -> Any:
    if guard is not None:
        guard.artifact()
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if guard is not None:
        guard.artifact()
    return value


def write_new_json(path: Path | str, value: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"path": str(path.resolve()), "sha256": sha_file(path), "bytes": path.stat().st_size}


def audit_source_marker() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": sha_file(path)}


def add_result_failure(result: dict[str, Any], failure: str) -> None:
    result.setdefault("summary_failures", []).append(failure)
    result["failed_count"] = int(result.get("failed_count") or 0) + 1
    if result.get("status") == "PASS":
        result["status"] = "FAIL"


def apply_source_pin(result: dict[str, Any], before: dict[str, Any]) -> None:
    after = audit_source_marker()
    result["audit_source_before"] = before
    result["audit_source_after"] = after
    if before["sha256"] != after["sha256"]:
        add_result_failure(result, "AUDIT_SOURCE_CHANGED_DURING_EXECUTION")


def raw_array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def model_array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def model_state_sha(kind: str, *values: Any) -> str:
    digest = hashlib.sha256(kind.encode())
    for value in values:
        digest.update((model_array_sha(value) if isinstance(value, np.ndarray) else str(value)).encode())
    return digest.hexdigest()


def matched_feature_state_sha(route: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    return model_state_sha(
        "MATCHED_DIRECT_C_FEATURE_STATE",
        route,
        arrays["center"],
        arrays["scale"],
        metadata["input_channels"],
        metadata["fit_rows"],
        metadata["fit_raw_features_sha256"],
    )


def native_feature_state_sha(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    return model_state_sha(
        "FIT_33K_FEATURE_STATE",
        arrays["center"],
        arrays["scale"],
        metadata["latent_channels"],
        metadata["fit_rows"],
        metadata["fit_raw_features_sha256"],
    )


def solution_sha(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    return model_state_sha(
        metadata["kind"],
        metadata["alpha"],
        arrays["coefficients"],
        arrays["intercept"],
        arrays["observed_counts"],
        arrays["channel_counts"],
        metadata["feature_state_sha256"],
    )


def matched_model_sha(feature_sha256: str, solution_sha256: str) -> str:
    return model_state_sha("MATCHED_DIRECT_C_MODEL", feature_sha256, solution_sha256)


def native_model_sha(feature_sha256: str, solution_sha256: str) -> str:
    return model_state_sha("FIXED_NATIVE_MODEL", feature_sha256, solution_sha256)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def record_sha(record: dict[str, Any]) -> str | None:
    value = record.get("sha256") or record.get("verified_sha256")
    return value.lower() if isinstance(value, str) else None


def read_record_json(record: dict[str, Any], *, role: str, guard: Guard) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, dict) or not record.get("path"):
        raise AuditFailure(role + "_RECORD_SCHEMA")
    path = resolve(record["path"])
    guard.artifact()
    actual = sha_file(path)
    expected = record_sha(record)
    if expected is None or actual.lower() != expected:
        raise AuditFailure(role + "_FILE_SHA_MISMATCH")
    return read_json(path, guard), {"path": str(path.resolve()), "sha256": actual, "role": role}


def canonical_record_marker(record: dict[str, Any]) -> tuple[Any, ...]:
    arrays = record.get("arrays") or {}
    array_marker = tuple(
        sorted(
            (
                name,
                tuple(entry.get("shape", [])),
                entry.get("dtype"),
                entry.get("sha256"),
            )
            for name, entry in arrays.items()
        )
    )
    return (str(resolve(record["path"]).resolve()).lower(), record_sha(record), array_marker)


def assert_same_record_marker(left: dict[str, Any], right: dict[str, Any], label: str, failures: list[str]) -> None:
    if canonical_record_marker(left) != canonical_record_marker(right):
        failures.append("RECORD_MARKER_MISMATCH:" + label)


def load_npz_record(record: dict[str, Any], *, role: str, guard: Guard) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    guard.artifact()
    path = resolve(record["path"])
    actual = sha_file(path)
    if actual != record.get("sha256"):
        raise AuditFailure(f"{role}_FILE_SHA_MISMATCH")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    declared = record.get("arrays")
    if declared is not None:
        if set(arrays) != set(declared):
            raise AuditFailure(f"{role}_ARRAY_KEY_MISMATCH")
        for key, value in arrays.items():
            entry = declared[key]
            if (
                list(value.shape) != entry.get("shape")
                or str(value.dtype) != entry.get("dtype")
                or raw_array_sha(value) != entry.get("sha256")
            ):
                raise AuditFailure(f"{role}_ARRAY_CONTENT_MISMATCH:{key}")
    guard.artifact()
    return arrays, {"path": str(path), "sha256": actual, "role": role}


def expected_splits(dataset: str) -> tuple[str, str, str]:
    if dataset not in DATASETS:
        raise AuditFailure("UNKNOWN_DATASET:" + str(dataset))
    return ("fit", "tune", DATASETS[dataset][2])


def expected_counts(dataset: str) -> dict[str, int]:
    final = DATASETS[dataset][2]
    return {"fit": 256, "tune": 64, final: DATASETS[dataset][3]}


def find_artifact_by_stem(producer: dict[str, Any], stem: str) -> dict[str, Any]:
    matches = [record for record in producer.get("artifacts", []) if Path(record["path"]).stem == stem]
    if len(matches) != 1:
        raise AuditFailure("ARTIFACT_STEM_LOOKUP_FAILED:" + stem)
    return matches[0]


def observed_score(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, std: np.ndarray | None = None) -> dict[str, Any]:
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * np.asarray(std, np.float64)
    observed = np.asarray(observed)
    error = np.where(observed, error, 0.0)
    count = observed.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error ** 2).sum((0, 1)), count, out=np.zeros(len(count), np.float64), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), count, out=np.zeros(len(count), np.float64), where=active)
    return {
        "mse": float(mse[active].mean()) if active.any() else None,
        "mae": float(mae[active].mean()) if active.any() else None,
        "per_channel_mse": [float(value) if is_active else None for value, is_active in zip(mse, active)],
        "per_channel_mae": [float(value) if is_active else None for value, is_active in zip(mae, active)],
        "observed_counts": count.tolist(),
        "active_channels": int(active.sum()),
    }


def approx_equal(left: Any, right: Any, *, atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def assert_score_close(recorded: dict[str, Any], computed: dict[str, Any], *, label: str, failures: list[str]) -> None:
    for key in ("mse", "mae"):
        if not approx_equal(recorded.get(key), computed.get(key)):
            failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}")
    for key in ("active_channels", "observed_counts"):
        if recorded.get(key) != computed.get(key):
            failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}")
    for key in ("per_channel_mse", "per_channel_mae"):
        left = recorded.get(key)
        right = computed.get(key)
        if not isinstance(left, list) or len(left) != len(right):
            failures.append(f"SCORE_{key.upper()}_SCHEMA:{label}")
            continue
        for index, (recorded_value, computed_value) in enumerate(zip(left, right)):
            if recorded_value is None or computed_value is None:
                if recorded_value is not None or computed_value is not None:
                    failures.append(f"SCORE_{key.upper()}_NONE_MISMATCH:{label}:{index}")
            elif not approx_equal(recorded_value, computed_value):
                failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}:{index}")


def assert_array_exact(left: np.ndarray, right: np.ndarray, label: str, failures: list[str]) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
        failures.append("ARRAY_EXACT_MISMATCH:" + label)


def past_features(route: str, past: np.ndarray) -> np.ndarray:
    value = np.asarray(past)
    if value.ndim != 3 or value.shape[1] != LENGTH or not value.shape[0] or not value.shape[2]:
        raise AuditFailure("NONEMPTY_PAST_N512Q_REQUIRED:" + route)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise AuditFailure("PAST_MUST_BE_FP32_FINITE:" + route)
    if route in ("RAW33C", "E33K_REUSED"):
        blocks = value.reshape(len(value), 32, 16, value.shape[2]).mean(axis=2)
        return np.concatenate((blocks, value[:, -1:]), axis=1).reshape(len(value), -1)
    if route == "E512K":
        return value.reshape(len(value), -1)
    raise AuditFailure("UNKNOWN_ROUTE:" + route)


def recompute_feature_state(route: str, fit_past: np.ndarray) -> dict[str, np.ndarray | str | int]:
    raw = past_features(route, fit_past)
    values = raw.astype(np.float64)
    center = values.mean(axis=0)
    scale = np.sqrt(np.mean((values - center) ** 2, axis=0))
    scale[scale == 0] = 1.0
    return {
        "center": center,
        "scale": scale,
        "input_channels": int(fit_past.shape[2]),
        "fit_rows": int(len(fit_past)),
        "fit_raw_features_sha256": model_array_sha(raw),
    }


def transform(route: str, feature_arrays: dict[str, np.ndarray], past: np.ndarray) -> np.ndarray:
    raw = past_features(route, past)
    center = feature_arrays["center"]
    scale = feature_arrays["scale"]
    if raw.shape[1] != len(center) or center.shape != scale.shape:
        raise AuditFailure("FEATURE_DIMENSION_MISMATCH:" + route)
    return (raw.astype(np.float64) - center) / scale


def predict_from_coefficients(design: np.ndarray, coefficient_arrays: dict[str, np.ndarray], raw_last: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficient_arrays["coefficients"], np.float64)
    intercept = np.asarray(coefficient_arrays["intercept"], np.float64)
    flat = design @ coefficients.reshape(design.shape[1], -1) + intercept.reshape(-1)
    innovation = flat.reshape((len(design),) + intercept.shape).astype(np.float32)
    return (np.asarray(raw_last, np.float32)[:, None] + innovation).astype(np.float32)


def check_solution_counts(route: str, coefficient_arrays: dict[str, np.ndarray], observed: np.ndarray, metadata: dict[str, Any], failures: list[str]) -> None:
    mask = np.asarray(observed)
    counts = mask.sum(axis=0).astype(np.int64)
    channel_counts = mask.sum(axis=(0, 1)).astype(np.int64)
    denominators = np.tile(channel_counts, mask.shape[1]).reshape(mask.shape[1:]).astype(np.float64)
    for name, expected, expected_dtype in (
        ("observed_counts", counts, np.int64),
        ("channel_counts", channel_counts, np.int64),
        ("normal_equation_denominators", denominators, np.float64),
    ):
        value = coefficient_arrays.get(name)
        if value is None or value.shape != expected.shape or value.dtype != expected_dtype or not np.array_equal(value, expected):
            failures.append("COUNT_ARRAY_MISMATCH:" + route + ":" + name)
    ranks = coefficient_arrays.get("design_ranks")
    if ranks is None or ranks.dtype != np.int64 or ranks.shape != counts.shape:
        failures.append("RANK_ARRAY_SCHEMA:" + route)
        return
    feature_count = coefficient_arrays["coefficients"].shape[0]
    rank_upper = np.minimum(np.maximum(counts - 1, 0), feature_count)
    if np.any(ranks < 0) or np.any(ranks > rank_upper):
        failures.append("RANK_ARRAY_RANGE:" + route)
    if metadata:
        model_meta = metadata.get("model_metadata", metadata)
        if model_meta.get("design_rank_min") != int(ranks.min()) or model_meta.get("design_rank_max") != int(ranks.max()):
            failures.append("RANK_METADATA_MISMATCH:" + route)
        if model_meta.get("observed_output_count") != int((counts > 0).sum()):
            failures.append("OBSERVED_OUTPUT_COUNT_METADATA:" + route)
        if model_meta.get("zero_observed_output_count") != int((counts == 0).sum()):
            failures.append("ZERO_OUTPUT_COUNT_METADATA:" + route)
        if model_meta.get("coefficient_shape") != list(coefficient_arrays["coefficients"].shape):
            failures.append("COEFFICIENT_SHAPE_METADATA:" + route)
        if model_meta.get("registered_coefficient_count") is not None and model_meta.get("registered_coefficient_count") != int(coefficient_arrays["coefficients"].size):
            failures.append("REGISTERED_COEFFICIENT_COUNT_METADATA:" + route)
        if model_meta.get("registered_intercept_count") is not None and model_meta.get("registered_intercept_count") != int(coefficient_arrays["intercept"].size):
            failures.append("REGISTERED_INTERCEPT_COUNT_METADATA:" + route)
        trained_outputs = int((counts > 0).sum())
        if model_meta.get("trained_coefficient_count") is not None and model_meta.get("trained_coefficient_count") != int(feature_count * trained_outputs):
            failures.append("TRAINED_COEFFICIENT_COUNT_METADATA:" + route)
        if model_meta.get("trained_intercept_count") is not None and model_meta.get("trained_intercept_count") != trained_outputs:
            failures.append("TRAINED_INTERCEPT_COUNT_METADATA:" + route)
        conditions = coefficient_arrays.get("regularized_gram_conditions")
        if conditions is not None:
            finite = conditions[np.isfinite(conditions)]
            if model_meta.get("gram_condition_inf_count") is not None and model_meta.get("gram_condition_inf_count") != int(np.isinf(conditions).sum()):
                failures.append("GRAM_CONDITION_INF_METADATA:" + route)
            if model_meta.get("gram_condition_nan_count") is not None and model_meta.get("gram_condition_nan_count") != int(np.isnan(conditions).sum()):
                failures.append("GRAM_CONDITION_NAN_METADATA:" + route)
            if len(finite):
                if model_meta.get("gram_condition_finite_min") is not None and not approx_equal(model_meta["gram_condition_finite_min"], float(finite.min())):
                    failures.append("GRAM_CONDITION_MIN_METADATA:" + route)
                if model_meta.get("gram_condition_finite_max") is not None and not approx_equal(model_meta["gram_condition_finite_max"], float(finite.max())):
                    failures.append("GRAM_CONDITION_MAX_METADATA:" + route)


def check_matched_model_hashes(route: str, model_meta: dict[str, Any], coeffs: dict[str, np.ndarray], feature_sha: str, failures: list[str], label: str) -> None:
    if model_meta.get("feature_state_sha256") != feature_sha:
        failures.append("MATCHED_MODEL_FEATURE_SHA_METADATA:" + label)
    actual_solution = solution_sha(model_meta, coeffs)
    if model_meta.get("solution_sha256") != actual_solution:
        failures.append("MATCHED_SOLUTION_SHA_MISMATCH:" + label)
    actual_model = matched_model_sha(feature_sha, actual_solution)
    if model_meta.get("model_sha256") != actual_model:
        failures.append("MATCHED_MODEL_SHA_MISMATCH:" + label)
    if model_meta.get("kind") != "DIRECT_C_" + route or model_meta.get("with_intercept") is not True:
        failures.append("MATCHED_MODEL_KIND_OR_INTERCEPT:" + label)


def check_native_reuse_hashes(stage: dict[str, Any], feature_arrays: dict[str, np.ndarray], coeffs: dict[str, np.ndarray], failures: list[str]) -> None:
    feature_meta = stage["feature_state"]
    feature_sha = native_feature_state_sha(feature_arrays, feature_meta)
    if feature_meta.get("sha256") != feature_sha:
        failures.append("REUSED_NATIVE_FEATURE_SHA_MISMATCH")
    selected = stage["families"]["N2"]
    solution_meta = selected["selected_model_metadata"]["solution"]
    actual_solution = solution_sha(solution_meta, coeffs)
    if solution_meta.get("solution_sha256") != actual_solution:
        failures.append("REUSED_NATIVE_SOLUTION_SHA_MISMATCH")
    actual_model = native_model_sha(feature_sha, actual_solution)
    if selected.get("selected_model_sha256") != actual_model:
        failures.append("REUSED_NATIVE_MODEL_SHA_MISMATCH")
    if solution_meta.get("feature_state_sha256") != feature_sha or solution_meta.get("kind") != "N2_DIRECT_C":
        failures.append("REUSED_NATIVE_MODEL_METADATA_MISMATCH")


def load_raw_inputs(producer: dict[str, Any], guard: Guard, failures: list[str]) -> dict[str, dict[str, np.ndarray]]:
    dataset = producer.get("dataset")
    c = DATASETS[dataset][0]
    counts = expected_counts(dataset)
    raw_inputs = {}
    for split in expected_splits(dataset):
        arrays, _ = load_npz_record(find_artifact_by_stem(producer, "RAW_INPUT_" + split), role="RAW_INPUT_" + split, guard=guard)
        n = counts[split]
        history = arrays.get("history")
        target = arrays.get("target")
        observed = arrays.get("observed")
        if history is None or history.shape != (n, LENGTH, c) or history.dtype != np.float32 or not np.isfinite(history).all():
            failures.append("RAW_HISTORY_SCHEMA:" + split)
        if target is None or target.shape != (n, HORIZON, c) or target.dtype != np.float32:
            failures.append("RAW_TARGET_SCHEMA:" + split)
        if observed is None or observed.shape != (n, HORIZON, c) or observed.dtype != np.bool_:
            failures.append("RAW_OBSERVED_SCHEMA:" + split)
        if target is not None and observed is not None and not np.isfinite(target[observed]).all():
            failures.append("RAW_OBSERVED_TARGET_NONFINITE:" + split)
        target_innovation = arrays["target"] - arrays["raw_last"][:, None]
        if target_innovation.dtype != np.float32 or not np.isfinite(target_innovation[arrays["observed"]]).all():
            failures.append("RAW_TARGET_INNOVATION_FP32_OBSERVED_FINITE:" + split)
        if arrays["mean"].dtype != np.float32 or arrays["std"].dtype != np.float32:
            failures.append("RAW_STATS_DTYPE:" + split)
        assert_array_exact(arrays["raw_last"], arrays["history"][:, -1], "raw_last:" + split, failures)
        raw_inputs[split] = arrays
    first = raw_inputs[expected_splits(dataset)[0]]
    for split, arrays in raw_inputs.items():
        assert_array_exact(arrays["mean"], first["mean"], "mean:" + split, failures)
        assert_array_exact(arrays["std"], first["std"], "std:" + split, failures)
    return raw_inputs


def load_cached_inputs(producer: dict[str, Any], raw_inputs: dict[str, dict[str, np.ndarray]], guard: Guard, failures: list[str]) -> dict[str, dict[str, np.ndarray]]:
    cached = {}
    if producer.get("families", {}).get("E512K", {}).get("status") != "COMPLETE" and producer.get("families", {}).get(REUSED_ROUTE, {}).get("status") != "COMPLETE":
        return cached
    for split, raw in raw_inputs.items():
        arrays, _ = load_npz_record(find_artifact_by_stem(producer, "CACHED_E_INPUT_" + split), role="CACHED_E_INPUT_" + split, guard=guard)
        cached[split] = arrays
        for name in ("target", "observed", "origins", "raw_last", "history_observed", "raw_forecast", "mean", "std"):
            assert_array_exact(arrays[name], raw[name], "cached_vs_raw:" + split + ":" + name, failures)
        assert_array_exact(arrays["z_last"], arrays["latent_past"][:, -1:], "cached_z_last:" + split, failures)
        if arrays["latent_past"].dtype != np.float32 or not np.isfinite(arrays["latent_past"]).all():
            failures.append("CACHED_LATENT_NONFINITE_OR_DTYPE:" + split)
    lineage = producer.get("cached_E_lineage", {})
    if cached and (
        lineage.get("original_bridge_and_input_artifact_SHA_bound") is not True
        or lineage.get("D2_to_stage_cached_array_metadata_and_bytes_bound") is not True
        or lineage.get("independent_E_forward_replayed") is not False
    ):
        failures.append("CACHED_E_LINEAGE_FLAGS")
    return cached


def check_source_records(records: list[dict[str, Any]], failures: list[str], guard: Guard) -> int:
    checked = 0
    for record in records:
        guard.artifact()
        expected = record.get("sha256") or record.get("verified_sha256")
        paths = []
        if record.get("path"):
            paths.append(("path", record["path"], expected))
        if record.get("snapshot_path"):
            paths.append(("snapshot_path", record["snapshot_path"], record.get("snapshot_sha256") or expected))
        if not paths or not expected:
            failures.append("SOURCE_RECORD_SCHEMA")
            continue
        for label, path, expected_sha in paths:
            resolved = resolve(path)
            if not resolved.exists():
                failures.append("SOURCE_RECORD_MISSING:" + label + ":" + str(resolved))
                continue
            if sha_file(resolved) != expected_sha:
                failures.append("SOURCE_RECORD_SHA_MISMATCH:" + label + ":" + str(resolved))
            checked += 1
    return checked


def check_route_feature_state(
    producer: dict[str, Any],
    route: str,
    past_by_split: dict[str, np.ndarray],
    guard: Guard,
    failures: list[str],
) -> dict[str, np.ndarray]:
    row = producer["families"][route]
    record = row["feature_state"]["artifact"]
    arrays, _ = load_npz_record(record, role=route + "_FIT_FEATURE_STATE", guard=guard)
    recomputed = recompute_feature_state(route, past_by_split["fit"])
    if model_array_sha(past_features(route, past_by_split["fit"])) != row["feature_state"].get("fit_raw_features_sha256"):
        failures.append("FIT_RAW_FEATURE_SHA_METADATA:" + route)
    for name in ("center", "scale"):
        if not np.allclose(arrays[name], recomputed[name], atol=0.0, rtol=0.0):
            failures.append("FIT_FEATURE_STATE_" + name.upper() + "_MISMATCH:" + route)
    if row["feature_state"].get("input_channels") != recomputed["input_channels"] or row["feature_state"].get("fit_rows") != recomputed["fit_rows"]:
        failures.append("FIT_FEATURE_STATE_SHAPE_METADATA:" + route)
    feature_sha = matched_feature_state_sha(route, arrays, row["feature_state"])
    if row["feature_state"].get("sha256") != feature_sha:
        failures.append("FIT_FEATURE_STATE_SHA_MISMATCH:" + route)
    return arrays


def audit_new_route(
    producer: dict[str, Any],
    route: str,
    past_by_split: dict[str, np.ndarray],
    raw_inputs: dict[str, dict[str, np.ndarray]],
    guard: Guard,
    failures: list[str],
) -> dict[str, int]:
    row = producer.get("families", {}).get(route, {})
    counts = {"candidate_rows": 0, "candidate_predictions": 0, "selected_predictions": 0}
    if row.get("status") != "COMPLETE":
        failures.append("NEW_ROUTE_NOT_COMPLETE:" + route)
        return counts
    alpha_rows = row.get("alpha_grid", [])
    if tuple(entry.get("alpha") for entry in alpha_rows) != ALPHAS:
        failures.append("ALPHA_GRID_NOT_EXACT_SEVEN:" + route)
        return counts
    feature_arrays = check_route_feature_state(producer, route, past_by_split, guard, failures)
    feature_sha = matched_feature_state_sha(route, feature_arrays, row["feature_state"])
    selected = min(range(len(alpha_rows)), key=lambda index: alpha_rows[index]["tune_actual_Y_mse"])
    if row.get("selected_index") != selected or row.get("selected_alpha") != alpha_rows[selected]["alpha"]:
        failures.append("SELECTION_EARLIEST_TIE_MISMATCH:" + route)
    if row.get("selected_model_sha256") != alpha_rows[selected].get("model_sha256"):
        failures.append("SELECTION_MODEL_SHA_MISMATCH:" + route)
    if row.get("selected_coefficient_artifact") != alpha_rows[selected].get("coefficient_artifact"):
        failures.append("SELECTION_COEFFICIENT_ARTIFACT_MISMATCH:" + route)
    for index, entry in enumerate(alpha_rows):
        counts["candidate_rows"] += 1
        coeffs, _ = load_npz_record(entry["coefficient_artifact"], role=f"{route}_ALPHA_{index}_COEFFICIENTS", guard=guard)
        check_solution_counts(route, coeffs, raw_inputs["fit"]["observed"], entry["model_metadata"], failures)
        check_matched_model_hashes(route, entry["model_metadata"], coeffs, feature_sha, failures, f"{route}:{index}")
        if entry.get("model_sha256") != entry["model_metadata"].get("model_sha256"):
            failures.append("CANDIDATE_MODEL_SHA_FIELD_MISMATCH:" + route + ":" + str(index))
        for split in ("fit", "tune"):
            artifact, _ = load_npz_record(entry[split + "_artifact"], role=f"{route}_ALPHA_{index}_{split}", guard=guard)
            prediction = predict_from_coefficients(transform(route, feature_arrays, past_by_split[split]), coeffs, raw_inputs[split]["raw_last"])
            if not np.allclose(prediction, artifact["prediction"], atol=ATOL, rtol=RTOL):
                failures.append("CANDIDATE_FORMULA_PREDICTION_MISMATCH:" + route + ":" + str(index) + ":" + split)
            normalized = observed_score(artifact["prediction"], raw_inputs[split]["target"], raw_inputs[split]["observed"])
            physical = observed_score(artifact["prediction"], raw_inputs[split]["target"], raw_inputs[split]["observed"], raw_inputs[split]["std"])
            assert_score_close(entry[split + "_scores"]["normalized"], normalized, label=f"{route}:{index}:{split}", failures=failures)
            assert_score_close(entry[split + "_scores"]["physical_per_channel_diagnostic"], physical, label=f"{route}:{index}:{split}:physical", failures=failures)
            if not approx_equal(entry["tune_actual_Y_mse"], entry["tune_scores"]["normalized"]["mse"], atol=0.0, rtol=0.0):
                failures.append("TUNE_ACTUAL_Y_MSE_FIELD_MISMATCH:" + route + ":" + str(index))
            counts["candidate_predictions"] += 1
    chosen_coeffs, _ = load_npz_record(row["selected_coefficient_artifact"], role=route + "_SELECTED_COEFFICIENTS", guard=guard)
    check_solution_counts(route, chosen_coeffs, raw_inputs["fit"]["observed"], row["selected_model_metadata"], failures)
    check_matched_model_hashes(route, row["selected_model_metadata"], chosen_coeffs, feature_sha, failures, route + ":selected")
    if row.get("selected_model_sha256") != row["selected_model_metadata"].get("model_sha256"):
        failures.append("SELECTED_MODEL_METADATA_SHA_FIELD_MISMATCH:" + route)
    for split, raw in raw_inputs.items():
        artifact, _ = load_npz_record(row["selected_artifacts"][split], role=route + "_SELECTED_" + split, guard=guard)
        prediction = predict_from_coefficients(transform(route, feature_arrays, past_by_split[split]), chosen_coeffs, raw["raw_last"])
        if not np.allclose(prediction, artifact["prediction"], atol=ATOL, rtol=RTOL):
            failures.append("SELECTED_FORMULA_PREDICTION_MISMATCH:" + route + ":" + split)
        for field in ("target", "observed", "origins", "raw_last", "mean", "std"):
            assert_array_exact(artifact[field], raw[field], route + ":selected:" + split + ":" + field, failures)
        normalized = observed_score(artifact["prediction"], artifact["target"], artifact["observed"])
        physical = observed_score(artifact["prediction"], artifact["target"], artifact["observed"], artifact["std"])
        assert_score_close(row["splits"][split]["normalized"], normalized, label=f"{route}:selected:{split}", failures=failures)
        assert_score_close(row["splits"][split]["physical_per_channel_diagnostic"], physical, label=f"{route}:selected:{split}:physical", failures=failures)
        counts["selected_predictions"] += 1
    return counts


def audit_reused_route(
    producer: dict[str, Any],
    cached: dict[str, dict[str, np.ndarray]],
    raw_inputs: dict[str, dict[str, np.ndarray]],
    stage: dict[str, Any] | None,
    guard: Guard,
    failures: list[str],
) -> dict[str, int]:
    row = producer.get("families", {}).get(REUSED_ROUTE, {})
    counts = {"reused_models": 0, "selected_predictions": 0}
    if row.get("status") != "COMPLETE":
        failures.append("REUSED_ROUTE_NOT_COMPLETE")
        return counts
    if not cached:
        failures.append("REUSED_ROUTE_MISSING_CACHED_INPUT")
        return counts
    feature_arrays, _ = load_npz_record(row["feature_source"], role="E33K_REUSED_FEATURE_STATE", guard=guard)
    coeffs, _ = load_npz_record(row["coefficient_source"], role="E33K_REUSED_COEFFICIENTS", guard=guard)
    if stage is None:
        failures.append("REUSED_ROUTE_STAGE_LINEAGE_MISSING")
    else:
        selected = stage.get("families", {}).get("N2", {})
        if row.get("selected_alpha") != selected.get("selected_alpha") or row.get("selected_model_sha256") != selected.get("selected_model_sha256"):
            failures.append("REUSED_SELECTED_MODEL_MARKER_MISMATCH")
        assert_same_record_marker(row["coefficient_source"], selected["selected_coefficient_artifact"], "reused_N2_coefficient", failures)
        assert_same_record_marker(row["feature_source"], stage["feature_state"]["artifact"], "reused_N2_feature_state", failures)
        check_native_reuse_hashes(stage, feature_arrays, coeffs, failures)
    recomputed = recompute_feature_state(REUSED_ROUTE, cached["fit"]["latent_past"])
    for name in ("center", "scale"):
        if not np.allclose(feature_arrays[name], recomputed[name], atol=0.0, rtol=0.0):
            failures.append("REUSED_FEATURE_STATE_" + name.upper() + "_MISMATCH")
    check_solution_counts(REUSED_ROUTE, coeffs, raw_inputs["fit"]["observed"], {}, failures)
    counts["reused_models"] = 1
    for split, raw in raw_inputs.items():
        artifact, _ = load_npz_record(row["selected_artifacts"][split], role="E33K_REUSED_SELECTED_" + split, guard=guard)
        prediction = predict_from_coefficients(transform(REUSED_ROUTE, feature_arrays, cached[split]["latent_past"]), coeffs, raw["raw_last"])
        if not np.allclose(prediction, artifact["prediction"], atol=ATOL, rtol=RTOL):
            failures.append("REUSED_SELECTED_FORMULA_PREDICTION_MISMATCH:" + split)
        for field in ("target", "observed", "origins", "raw_last", "mean", "std"):
            assert_array_exact(artifact[field], raw[field], "reused:selected:" + split + ":" + field, failures)
        normalized = observed_score(artifact["prediction"], artifact["target"], artifact["observed"])
        physical = observed_score(artifact["prediction"], artifact["target"], artifact["observed"], artifact["std"])
        assert_score_close(row["splits"][split]["normalized"], normalized, label="reused:selected:" + split, failures=failures)
        assert_score_close(row["splits"][split]["physical_per_channel_diagnostic"], physical, label="reused:selected:" + split + ":physical", failures=failures)
        parity = row.get("cached_prediction_checks", {}).get(split)
        if not isinstance(parity, dict) or parity.get("pass") is not True:
            failures.append("REUSED_CACHED_PARITY_FLAG:" + split)
        counts["selected_predictions"] += 1
    return counts


def audit_references(
    producer: dict[str, Any],
    raw_inputs: dict[str, dict[str, np.ndarray]],
    cached: dict[str, dict[str, np.ndarray]],
    failures: list[str],
) -> int:
    checked = 0
    for split, raw in raw_inputs.items():
        refs = producer.get("references", {}).get(split, {})
        if "RAW_BOLT" in refs:
            normalized = observed_score(raw["raw_forecast"], raw["target"], raw["observed"])
            physical = observed_score(raw["raw_forecast"], raw["target"], raw["observed"], raw["std"])
            assert_score_close(refs["RAW_BOLT"]["normalized"], normalized, label="RAW_BOLT:" + split, failures=failures)
            assert_score_close(refs["RAW_BOLT"]["physical_per_channel_diagnostic"], physical, label="RAW_BOLT:" + split + ":physical", failures=failures)
            checked += 1
        if "ORIGINAL_MAIN" in refs:
            if split not in cached or "original_prediction" not in cached[split]:
                failures.append("ORIGINAL_MAIN_CACHED_PREDICTION_MISSING:" + split)
                continue
            normalized = observed_score(cached[split]["original_prediction"], raw["target"], raw["observed"])
            physical = observed_score(cached[split]["original_prediction"], raw["target"], raw["observed"], raw["std"])
            assert_score_close(refs["ORIGINAL_MAIN"]["normalized"], normalized, label="ORIGINAL_MAIN:" + split, failures=failures)
            assert_score_close(refs["ORIGINAL_MAIN"]["physical_per_channel_diagnostic"], physical, label="ORIGINAL_MAIN:" + split + ":physical", failures=failures)
            checked += 1
    return checked


def find_source_record_by_sha(producer: dict[str, Any], expected_sha: str | None, label: str, failures: list[str]) -> dict[str, Any] | None:
    if not expected_sha:
        failures.append("SOURCE_SHA_LOOKUP_EMPTY:" + label)
        return None
    matches = [record for record in producer.get("source_records", []) if record_sha(record) == expected_sha.lower()]
    if len(matches) != 1:
        failures.append("SOURCE_SHA_LOOKUP_NOT_EXACT:" + label)
        return None
    return matches[0]


def audit_lineage(producer: dict[str, Any], cached: dict[str, dict[str, np.ndarray]], guard: Guard, failures: list[str]) -> dict[str, Any] | None:
    """Read declared old-stage/D2/report audit records after the runtime guard.

    This binds the new producer to the canonical old N2 reuse gate and D2 cached
    metadata. It remains a shared-gate lineage audit, not an independent E/F
    forward replay.
    """
    stage_record = producer.get("stage_N2_source")
    if not isinstance(stage_record, dict):
        failures.append("STAGE_N2_SOURCE_MISSING")
        return None
    try:
        stage, checked_stage = read_record_json(stage_record, role="STAGE_N2_SOURCE", guard=guard)
    except AuditFailure as error:
        failures.append(str(error))
        return None
    dataset = producer.get("dataset")
    seed = producer.get("seed")
    if stage.get("status") != "COMPLETE" or stage.get("dataset") != dataset or stage.get("seed") != seed:
        failures.append("STAGE_N2_SOURCE_CONDITION_MISMATCH")
    d2_record = producer.get("cached_E_lineage", {}).get("original_D2_source")
    if not isinstance(d2_record, dict):
        failures.append("D2_SOURCE_MISSING")
    else:
        try:
            d2, _ = read_record_json(d2_record, role="D2_SOURCE", guard=guard)
            assert_same_record_marker(d2_record, stage["D2_source"], "stage_D2_source", failures)
            cp = producer.get("original_CP_file_hash_only", {})
            cp_sha = record_sha(cp)
            if not isinstance(cp, dict) or not cp.get("path") or cp_sha is None:
                failures.append("ORIGINAL_CP_FILE_RECORD_MISSING")
            elif sha_file(resolve(cp["path"])).lower() != cp_sha:
                failures.append("ORIGINAL_CP_FILE_SHA_MISMATCH")
            if (
                d2.get("status") != "COMPLETE"
                or d2.get("stage") != "D2"
                or d2.get("dataset") != dataset
                or d2.get("seed") != seed
                or d2.get("MAIN_codec_frozen_pass") is not True
                or d2.get("F_frozen_pass") is not True
                or d2.get("F_before") != d2.get("F_after")
                or d2.get("MAIN_checkpoint", {}).get("checkpoint", {}).get("sha256") != cp_sha
                or stage.get("MAIN_checkpoint", {}).get("checkpoint", {}).get("sha256") != cp_sha
            ):
                failures.append("D2_STAGE_CP_LINEAGE_MISMATCH")
            if d2.get("bridge_provenance", {}).get("input_bridge", {}).get("sha256") != stage.get("bridge_provenance", {}).get("input_bridge", {}).get("sha256"):
                failures.append("D2_STAGE_BRIDGE_SHA_MISMATCH")
            for split, arrays in cached.items():
                stage_input = [r for r in stage.get("artifacts", []) if Path(r["path"]).stem == "INPUT_" + split]
                d2_input = [r for r in d2.get("artifacts", []) if Path(r["path"]).stem == "ABSOLUTE_LATENTS_" + split]
                producer_input = find_artifact_by_stem(producer, "CACHED_E_INPUT_" + split)
                if len(stage_input) != 1 or len(d2_input) != 1:
                    failures.append("CACHED_LINEAGE_INPUT_RECORD_NOT_EXACT:" + split)
                    continue
                for name in ("latent_past", "z_last", "target", "observed", "origins", "raw_last", "history_observed", "mean", "std", "raw_forecast", "original_prediction"):
                    actual = raw_array_sha(arrays[name])
                    for source_name, record in (("stage", stage_input[0]), ("d2", d2_input[0]), ("producer", producer_input)):
                        declared = record.get("arrays", {}).get(name)
                        if not declared or declared.get("shape") != list(arrays[name].shape) or declared.get("dtype") != str(arrays[name].dtype) or declared.get("sha256") != actual:
                            failures.append("CACHED_ARRAY_METADATA_MISMATCH:" + split + ":" + name + ":" + source_name)
        except AuditFailure as error:
            failures.append(str(error))
    bridge_sha = stage.get("bridge_provenance", {}).get("input_bridge", {}).get("sha256")
    if find_source_record_by_sha(producer, bridge_sha, "input_bridge", failures) is None:
        failures.append("PRODUCER_BRIDGE_SOURCE_RECORD_MISSING")
    manifest_records = [record for record in producer.get("source_records", []) if Path(record.get("path", "")).name == "MANIFEST.json"]
    if len(manifest_records) != 1:
        failures.append("CANONICAL_STAGE_REPORT_RECORD_NOT_EXACT")
    else:
        try:
            manifest, _ = read_record_json(manifest_records[0], role="CANONICAL_STAGE_REPORT", guard=guard)
            if manifest.get("generator_result_status") != "COMPLETE" or manifest.get("native_audit_gate", {}).get("passed") is not True:
                failures.append("CANONICAL_STAGE_REPORT_GATE_NOT_PASS")
            records = manifest.get("input_records", [])
            if not any(record.get("role") == "native_result" and record.get("sha256", "").lower() == checked_stage["sha256"].lower() for record in records):
                failures.append("CANONICAL_STAGE_REPORT_NATIVE_RESULT_BINDING")
            audit_records = [record for record in records if record.get("role") == "native_audit"]
            if len(audit_records) != 1:
                failures.append("CANONICAL_NATIVE_AUDIT_RECORD_NOT_EXACT")
            else:
                audit, _ = read_record_json(audit_records[0], role="CANONICAL_NATIVE_AUDIT", guard=guard)
                matches = [row for row in audit.get("conditions", []) if row.get("dataset") == dataset and row.get("seed") == seed]
                if audit.get("status") != "PASS" or audit.get("failed_count") != 0 or len(audit.get("conditions", [])) != 4:
                    failures.append("CANONICAL_NATIVE_AUDIT_NOT_PASS_FOUR")
                if len(matches) != 1 or matches[0].get("producer_sha256") != checked_stage["sha256"]:
                    failures.append("CANONICAL_NATIVE_AUDIT_PRODUCER_BINDING")
        except AuditFailure as error:
            failures.append(str(error))
    return stage


def audit_one(path: Path | str, guard: Guard) -> dict[str, Any]:
    path = resolve(path)
    producer = read_json(path, guard)
    producer_sha = sha_file(path)
    failures: list[str] = []
    dataset = producer.get("dataset")
    seed = producer.get("seed")
    if producer.get("status") != "COMPLETE":
        failures.append("PRODUCER_NOT_COMPLETE")
    if dataset not in DATASETS or seed not in (92601, 92602):
        failures.append("PRODUCER_CONDITION_IDENTITY")
    if producer.get("protected_TEST_accessed") is not False or producer.get("GPU_execution") is not False:
        failures.append("PRODUCER_TEST_OR_GPU_FLAG")
    if (
        producer.get("E_D_F_loaded") is not False
        or producer.get("E_D_F_forward_replayed") is not False
        or producer.get("E_D_F_updated") is not False
        or producer.get("future_oracle_read") is not False
    ):
        failures.append("PRODUCER_E_F_FUTURE_SCOPE_FLAGS")
    if producer.get("old_N2_retrained") is not False or producer.get("old_E33K_and_NEW_routes_have_different_input_capacity") is not True:
        failures.append("PRODUCER_REUSE_CAPACITY_FLAGS")
    raw_inputs = load_raw_inputs(producer, guard, failures)
    cached = load_cached_inputs(producer, raw_inputs, guard, failures)
    source_records_checked = check_source_records(producer.get("source_records", []), failures, guard)
    source_records_checked += check_source_records(producer.get("source_snapshots", []), failures, guard)
    reference_checks = audit_references(producer, raw_inputs, cached, failures)
    stage = audit_lineage(producer, cached, guard, failures)

    totals = {"candidate_rows": 0, "candidate_predictions": 0, "selected_predictions": 0, "reused_models": 0}
    route_inputs = {
        "RAW33C": {split: values["history"] for split, values in raw_inputs.items()},
        "E512K": {split: values["latent_past"] for split, values in cached.items()} if cached else {},
    }
    for route in NEW_ROUTES:
        if route == "E512K" and not route_inputs[route]:
            failures.append("E512K_MISSING_CACHED_INPUT")
            continue
        counts = audit_new_route(producer, route, route_inputs[route], raw_inputs, guard, failures)
        for key, value in counts.items():
            totals[key] += value
    reused = audit_reused_route(producer, cached, raw_inputs, stage, guard, failures)
    totals["reused_models"] += reused["reused_models"]
    totals["selected_predictions"] += reused["selected_predictions"]

    family_status = {family: producer.get("families", {}).get(family, {}).get("status", "MISSING") for family in (*NEW_ROUTES, REUSED_ROUTE)}
    return {
        "dataset": dataset,
        "seed": seed,
        "producer": {"path": str(path.resolve()), "sha256": producer_sha},
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "family_status": family_status,
        "new_candidate_rows_checked": totals["candidate_rows"],
        "candidate_fit_tune_predictions_checked": totals["candidate_predictions"],
        "selected_predictions_checked": totals["selected_predictions"],
        "reused_models_checked": totals["reused_models"],
        "source_records_checked": source_records_checked,
        "reference_scores_checked": reference_checks,
        "scope": {
            "actual_E_or_F_forward_replayed": False,
            "solver_svd_lstsq_replayed": False,
            "audit_equations": "raw/cached input exactness, FIT feature statistics, saved FP64 coefficients, FP32 innovation and current-C anchor, score/selection/capacity metadata",
            "lineage_scope": "shared old-stage/native-audit/D2 metadata gate; independent E/F forward replay remains false",
        },
    }


def producer_paths_from_queue(path: Path | str, guard: Guard) -> list[Path]:
    payload = read_json(path, guard)
    candidates = payload.get("matched_records") or payload.get("conditions") or payload.get("producers") or payload.get("rows")
    if not isinstance(candidates, list):
        raise AuditFailure("QUEUE_MATCHED_RECORDS_LIST_REQUIRED")
    paths = []
    for row in candidates:
        if isinstance(row, str):
            paths.append(resolve(row))
            continue
        if not isinstance(row, dict):
            raise AuditFailure("QUEUE_ROW_SCHEMA")
        record = row.get("matched_result") or row.get("producer_result") or row.get("result") or row
        if isinstance(record, str):
            paths.append(resolve(record))
            continue
        if not isinstance(record, dict) or not record.get("path"):
            raise AuditFailure("QUEUE_PRODUCER_RESULT_PATH_REQUIRED")
        paths.append(resolve(record["path"]))
    return paths


def execute(paths: list[Path], *, out: Path, guard: Guard) -> dict[str, Any]:
    if len(paths) != 4:
        raise AuditFailure("EXACT_FOUR_MATCHED_READOUT_PRODUCERS_REQUIRED")
    resolved = [resolve(path).resolve() for path in paths]
    if len({str(path).lower() for path in resolved}) != 4:
        raise AuditFailure("EXACT_FOUR_UNIQUE_MATCHED_READOUT_PRODUCERS_REQUIRED")
    rows = [audit_one(path, guard) for path in resolved]
    failures = [failure for row in rows for failure in row["failures"]]
    condition_set = {(row["dataset"], row["seed"]) for row in rows}
    if condition_set != EXPECTED_CONDITIONS or len(condition_set) != len(rows):
        failures.append("EXACT_DATASET_SEED_2X2_CONDITIONS_REQUIRED")
    conditions = [
        {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "producer_path": row["producer"]["path"],
            "producer_sha256": row["producer"]["sha256"],
            "status": row["status"],
        }
        for row in rows
    ]
    result = {
        "schema": "smallk_matched_readout_audit_v2",
        "status": "PASS" if not failures else "FAIL",
        "failed_count": len(failures),
        "conditions": conditions,
        "rows": rows,
        "new_candidate_rows_checked": sum(row["new_candidate_rows_checked"] for row in rows),
        "candidate_fit_tune_predictions_checked": sum(row["candidate_fit_tune_predictions_checked"] for row in rows),
        "selected_predictions_checked": sum(row["selected_predictions_checked"] for row in rows),
        "reused_models_checked": sum(row["reused_models_checked"] for row in rows),
        "source_records_checked": sum(row["source_records_checked"] for row in rows),
        "scientific_scope": "independent CPU array audit of matched readout saved outputs; no E/F forward, no GPU, no SVD/lstsq refit",
    }
    for key, expected in EXPECTED_TOTALS.items():
        if result[key] != expected:
            failures.append("EXPECTED_TOTAL_MISMATCH:" + key + ":" + str(result[key]) + "!=" + str(expected))
    result["status"] = "PASS" if not failures else "FAIL"
    result["failed_count"] = len(failures)
    result["summary_failures"] = failures
    return result


def describe() -> dict[str, Any]:
    return {
        "status": "SOURCE_ONLY_NOT_EXECUTED",
        "schema": "smallk_matched_readout_audit_v2",
        "expected_conditions": "exactly 4 unique matched RESULT.json files: robin/beijing_aotizhongxin x seeds 92601/92602",
        "families": [*NEW_ROUTES, REUSED_ROUTE],
        "expected_new_candidate_rows": "4 producers x 2 new routes x 7 alpha = 56",
        "expected_candidate_fit_tune_predictions": "4 producers x 2 new routes x 7 alpha x FIT/TUNE = 112",
        "expected_selected_predictions": "4 producers x (RAW33C/E512K/E33K_REUSED) x 3 splits = 36",
        "expected_reused_models": "4 old N2 E33K selected-model reuses",
        "checks": [
            "RAW_INPUT and CACHED_E_INPUT exact target/mask/origin/current-anchor/stat contracts",
            "FIT-only feature center/scale recomputation and model-scheme raw-feature SHA",
            "saved FP64 coefficients -> FP32 innovation -> FP32 current-C anchor prediction",
            "normalized and physical observed-channel macro/per-channel MSE/MAE",
            "seven-alpha earliest TUNE selection",
            "count/rank/capacity metadata consistency without SVD refit",
            "H x C observed-count and rank arrays with float64 normal-equation denominators",
            "source/snapshot/artifact SHA verification",
            "audit source SHA before/after execution pin",
            "canonical old-stage/native-audit/D2 shared-gate lineage binding without independent E/F forward replay",
            "no TEST/GPU/F/future scope flags",
        ],
        "no_actual_reads_on_import_or_describe": True,
    }


def execute_with_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import matched_readout_runtime_v1 as runtime

    out_path = Path(args.out)
    source_before = audit_source_marker()
    try:
        limit = runtime.ContinuationLimit("matched_audit_" + safe_tag(args.tag), gpu=False)
    except Exception as error:
        result = {
            "schema": "smallk_matched_readout_audit_v2",
            "status": "FAILED_PRESERVED",
            "failed_count": 1,
            "conditions": [],
            "rows": [],
            "failure": {"type": type(error).__name__, "message": str(error)},
            "scientific_scope": "runtime constructor failed before scientific assets were opened",
        }
        apply_source_pin(result, source_before)
        if out_path.exists():
            result["artifact_write"] = {"status": "SKIPPED_PREEXISTING_OUTPUT", "path": str(out_path.resolve())}
        else:
            result["artifact"] = write_new_json(out_path, result)
        return result
    import prepare_inputs as inputs

    guard = Guard(limit)
    result: dict[str, Any] | None = None
    status_for_receipt = "COMPLETE"
    terminal_error = None
    try:
        inputs.install_protected_test_guard()
        guard.full()
        if out_path.exists():
            raise AuditFailure("AUDIT_OUTPUT_MUST_BE_FRESH:" + str(out_path.resolve()))
        paths = [resolve(path) for path in args.producer_result]
        if args.queue:
            paths.extend(producer_paths_from_queue(args.queue, guard))
        result = execute(paths, out=out_path, guard=guard)
        return result
    except Exception as error:
        status_for_receipt = "FAILED_PRESERVED"
        terminal_error = {"type": type(error).__name__, "message": str(error)}
        result = {
            "schema": "smallk_matched_readout_audit_v2",
            "status": "FAILED_PRESERVED",
            "failed_count": 1,
            "conditions": [],
            "rows": [],
            "failure": terminal_error,
            "scientific_scope": "audit aborted before successful PASS/FAIL verdict; preserve source and error",
        }
        return result
    finally:
        receipt = limit.close(status_for_receipt, terminal_error)
        if result is not None:
            apply_source_pin(result, source_before)
            result["resource_receipt"] = receipt
            terminal_status = receipt.get("terminal_status") if isinstance(receipt, dict) else None
            if terminal_status != "COMPLETE":
                result["status"] = terminal_status or "FAILED_PRESERVED"
                result["failed_count"] = max(1, int(result.get("failed_count") or 0))
                result["resource_terminal_status_downgraded"] = True
            if out_path.exists():
                result["artifact_write"] = {"status": "SKIPPED_PREEXISTING_OUTPUT", "path": str(out_path.resolve())}
            else:
                result["artifact"] = write_new_json(out_path, result)


def safe_tag(tag: str) -> str:
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in tag):
        raise ValueError("SAFE_AUDIT_TAG_REQUIRED")
    return tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--producer-result", action="append", default=[])
    parser.add_argument("--queue")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default="audit_matched_readout_v2")
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    result = execute_with_runtime(args)
    print(json.dumps({"status": result["status"], "failed_count": result.get("failed_count"), "conditions": result.get("conditions", [])}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
