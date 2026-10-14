"""Independent saved-array audit for the approved adaptive readout campaign.

No scientific file is read at import time. The numerical verifier receives
already authorized arrays and does not call the producer's math, encoder,
decoder, forecaster, SVD, fitting routine or protected TEST. This checks saved
arithmetic, first-order ridge equations, masking, selection and provenance;
it is not an independent holdout or a replay of historical E/F/D behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

# Actual entry points initialize NumPy only after creating the campaign guard.
# Guarded fixtures import NumPy first, so their import is already authorized.
np = sys.modules.get("numpy")


ALPHAS = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
ROUTES = ("SELF33", "JOINT33", "SELF512")
LENGTH = 512
HORIZON = 48
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def initialize_numpy():
    global np
    import numpy as loaded_numpy
    np = loaded_numpy


class AuditFailure(ValueError):
    pass


def require(condition, label):
    if not condition:
        raise AuditFailure(label)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=True), "EXACT_ARRAY_MISMATCH:" + label)


def verify_locked_origins(origins, locked):
    """Phase0 stores legacy origins as a list and new-data origins as a digest."""
    require(origins.dtype == np.int64 and origins.ndim == 1 and len(origins) > 0, "INT64_ORIGIN_VECTOR_REQUIRED")
    if isinstance(locked, list):
        exact(origins, np.asarray(locked, np.int64), "locked_origins")
        return "EXACT_VECTOR"
    require(isinstance(locked, dict) and set(locked) == {"count", "sha256", "first", "last"},
            "DECLARED_PHASE0_ORIGIN_SCHEMA_REQUIRED")
    require(len(origins) == locked["count"] and int(origins[0]) == locked["first"]
            and int(origins[-1]) == locked["last"]
            and hashlib.sha256(np.ascontiguousarray(origins).tobytes()).hexdigest() == locked["sha256"],
            "PHASE0_ORIGIN_SUMMARY_CONTENT_DIGEST_MISMATCH")
    return "EXACT_RAW_INT64_BYTES_DIGEST"


def close(actual, expected, label, *, rtol=2e-6, atol=1e-5):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True),
            "NUMERIC_MISMATCH:" + label)
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def raw_features(route, history):
    x = np.asarray(history)
    require(route in ROUTES, "UNKNOWN_PREDECLARED_ROUTE")
    require(x.ndim == 3 and x.shape[1] == LENGTH and len(x) > 0
            and x.shape[2] > 0 and x.dtype == np.float32 and np.isfinite(x).all(),
            "FINITE_NONEMPTY_FP32_N512C_HISTORY_REQUIRED")
    if route == "SELF512":
        return x.copy()
    means = np.mean(x.reshape(len(x), 32, 16, x.shape[2]), axis=2, dtype=np.float32)
    features = np.concatenate((means, x[:, -1:]), axis=1)
    return features.reshape(len(x), -1) if route == "JOINT33" else features


def expected_feature_state(route, fit_history):
    values = raw_features(route, fit_history).astype(np.float64)
    center = values.mean(axis=0)
    scale = np.sqrt(np.mean(np.square(values - center), axis=0))
    scale[scale == 0] = 1.0
    return center, scale


def independent_design(route, history, center, scale):
    values = raw_features(route, history).astype(np.float64)
    c, s = np.asarray(center), np.asarray(scale)
    require(c.dtype == np.float64 and s.dtype == np.float64 and c.shape == values.shape[1:]
            and s.shape == c.shape and np.isfinite(c).all() and np.isfinite(s).all()
            and (s > 0).all(), "FINITE_FIT_FP64_FEATURE_STATE_REQUIRED")
    return (values - c) / s


def prediction_from_saved(route, history, raw_last, center, scale, coefficients, intercept):
    design = independent_design(route, history, center, scale)
    last, weight, bias = np.asarray(raw_last), np.asarray(coefficients), np.asarray(intercept)
    channels = history.shape[2]
    require(last.dtype == np.float32 and last.shape == (len(history), channels), "FP32_CURRENT_ANCHOR_REQUIRED")
    exact(last, history[:, -1], "anchor_vs_past")
    require(weight.dtype == np.float64 and bias.dtype == np.float64
            and weight.shape == (design.shape[1], HORIZON, channels)
            and bias.shape == (HORIZON, channels) and np.isfinite(weight).all()
            and np.isfinite(bias).all(), "FINITE_RIDGE_COEFFICIENT_SHAPES_REQUIRED")
    if route == "JOINT33":
        innovation64 = (design @ weight.reshape(design.shape[1], -1)).reshape(len(design), HORIZON, channels)
    else:
        # Each output channel receives only its own channel's past.
        innovation64 = np.empty((len(design), HORIZON, channels), dtype=np.float64)
        for channel in range(channels):
            innovation64[:, :, channel] = design[:, :, channel] @ weight[:, :, channel]
    innovation64 += bias
    return last[:, None] + innovation64.astype(np.float32)


def check_raw_split(values, *, expected_channels=None, expected_origins=None):
    history, target, mask = values["history"], values["target"], values["observed"]
    require(history.dtype == np.float32 and history.ndim == 3 and history.shape[1] == LENGTH
            and np.isfinite(history).all(), "INVALID_PAST_ARRAY")
    channels = history.shape[2]
    if expected_channels is not None:
        require(channels == expected_channels, "DATASET_CHANNEL_COUNT_MISMATCH")
    require(target.dtype == np.float32 and target.shape == (len(history), HORIZON, channels)
            and mask.dtype == np.bool_ and mask.shape == target.shape
            and np.isfinite(target[mask]).all(), "OBSERVED_FP32_TARGET_BOOL_MASK_REQUIRED")
    require(values["history_observed"].dtype == np.bool_
            and values["history_observed"].shape == history.shape, "PAST_OBSERVABILITY_SHAPE")
    exact(values["raw_last"], history[:, -1], "raw_last")
    origins = values["origins"]
    require(origins.dtype == np.int64 and origins.shape == (len(history),)
            and (np.diff(origins) > 0).all(), "STRICTLY_INCREASING_INT64_ORIGINS_REQUIRED")
    if expected_origins is not None:
        verify_locked_origins(origins, expected_origins)
    for name in ("mean", "std"):
        require(values[name].dtype == np.float32 and values[name].shape == (channels,)
                and np.isfinite(values[name]).all(), "RAW_FIT_STATS:" + name)
    require((values["std"] > 0).all(), "POSITIVE_RAW_FIT_STD_REQUIRED")
    return {"windows": len(history), "channels": channels,
            "first_origin": int(origins[0]), "last_origin": int(origins[-1]),
            "observed_counts": mask.sum(axis=(0, 1)).tolist()}


def fit_mean_innovation(target, raw_last, observed):
    require(target.dtype == np.float32 and raw_last.dtype == np.float32, "MEAN_FP32_INPUT_REQUIRED")
    innovation = target - raw_last[:, None]
    count = observed.sum(axis=0)
    total = np.where(observed, innovation.astype(np.float64), 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)
    return mean, count


def scores(prediction, target, observed, std):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    require(p.shape == y.shape and y.ndim == 3 and mask.shape == y.shape
            and mask.dtype == np.bool_ and np.isfinite(p[mask]).all()
            and np.isfinite(y[mask]).all(), "MASKED_SCORE_INPUT_INVALID")
    channels = y.shape[2]
    count = mask.sum(axis=(0, 1))
    active = count > 0
    require(active.any(), "NO_OBSERVED_CHANNEL")
    error = np.where(mask, p.astype(np.float64) - y.astype(np.float64), 0.0)
    per_mse = np.divide(np.square(error).sum(axis=(0, 1)), count,
                        out=np.zeros(channels, np.float64), where=active)
    per_mae = np.divide(np.abs(error).sum(axis=(0, 1)), count,
                        out=np.zeros(channels, np.float64), where=active)
    scaling = np.asarray(std, np.float64)
    require(scaling.shape == (channels,) and np.isfinite(scaling).all() and (scaling > 0).all(), "SCORE_STD_INVALID")
    physical_mse, physical_mae = per_mse * scaling ** 2, per_mae * scaling
    common = {"observed_counts": count.tolist(), "active_channels": int(active.sum())}
    return {"normalized": {"mse": float(per_mse[active].mean()),
                            "mae": float(per_mae[active].mean()),
                            "per_channel_mse": per_mse.tolist(), "per_channel_mae": per_mae.tolist(), **common},
            "physical_per_channel_diagnostic": {"mse": float(physical_mse[active].mean()),
                         "mae": float(physical_mae[active].mean()),
                         "per_channel_mse": physical_mse.tolist(), "per_channel_mae": physical_mae.tolist(), **common}}


def compare_scores(actual, expected, label):
    checks = 0
    for units in ("normalized", "physical_per_channel_diagnostic"):
        for metric in ("mse", "mae", "per_channel_mse", "per_channel_mae"):
            close(actual[units][metric], expected[units][metric], label + ":" + units + ":" + metric,
                  rtol=2e-12, atol=2e-12)
            checks += 1
        require(actual[units]["observed_counts"] == expected[units]["observed_counts"], "SCORE_MASK_COUNT_MISMATCH:" + label)
        require(actual[units]["active_channels"] == expected[units]["active_channels"], "SCORE_ACTIVE_CHANNEL_COUNT_MISMATCH:" + label)
        checks += 2
    return checks


def normal_equation_checks(route, design, target, raw_last, observed, coefficients, intercept, alpha,
                           *, relative_tolerance=2e-7):
    """Check all active (h,c) equations with alpha*Nc, without solving/refitting.

    An alpha0 first-order equation does not establish the producer's minimum-
    norm tie choice. Its provenance, predictions and rank bounds are checked
    separately; no independent SVD is claimed by this audit.
    """
    require(alpha in ALPHAS, "UNAPPROVED_ALPHA")
    innovation = (target - raw_last[:, None]).astype(np.float64)
    channel_count = observed.sum(axis=(0, 1))
    output_count = observed.sum(axis=0)
    max_relative, max_intercept, active_outputs = 0.0, 0.0, 0
    for channel in range(target.shape[2]):
        x = design if route == "JOINT33" else design[:, :, channel]
        for horizon in range(target.shape[1]):
            rows = observed[:, horizon, channel]
            w = coefficients[:, horizon, channel]
            b = intercept[horizon, channel]
            if not rows.any():
                require(np.count_nonzero(w) == 0 and b == 0, "ZERO_OBSERVED_FALLBACK_MUST_BE_UNLEARNED")
                continue
            gx, gy = x[rows], innovation[rows, horizon, channel]
            residual = gx @ w + b - gy
            penalty = alpha * channel_count[channel]
            gradient = gx.T @ residual + penalty * w
            scale = max(1.0, float(np.max(np.abs(gx.T @ gy))), float(penalty * np.max(np.abs(w))))
            relative = float(np.max(np.abs(gradient))) / scale
            intercept_error = abs(float(residual.mean()))
            intercept_scale = max(1.0, float(np.max(np.abs(gy))), abs(float(b)))
            require(relative <= relative_tolerance, "NORMAL_EQUATION_ALPHA_NC_FAILURE")
            require(intercept_error <= relative_tolerance * intercept_scale, "UNPENALIZED_INTERCEPT_FAILURE")
            max_relative = max(max_relative, relative)
            max_intercept = max(max_intercept, intercept_error)
            active_outputs += 1
    return {"active_outputs_checked": active_outputs, "max_scaled_normal_equation_residual": max_relative,
            "max_absolute_intercept_equation_residual": max_intercept,
            "channel_counts": channel_count.tolist(), "observed_counts": output_count.tolist(),
            "penalty_contract": "alpha*N_channel_over_all_origins_and_horizons",
            "independent_solver_replayed": False, "alpha0_minimum_norm_reproved": False}


def select_pool(pool):
    require(len(pool) == 9, "NINE_PREDECLARED_CANDIDATES_REQUIRED")
    require([row["kind"] for row in pool] == ["PERSIST", "MEAN"] + ["RIDGE"] * 7,
            "POOL_ORDER_MISMATCH")
    require(tuple(row["alpha"] for row in pool[2:]) == ALPHAS, "POOL_ALPHA_GRID_MISMATCH")
    values = [float(row["tune_scores"]["normalized"]["mse"]) for row in pool]
    require(np.isfinite(values).all(), "FINITE_GLOBAL_MACRO_SELECTION_REQUIRED")
    selected = min(range(len(values)), key=lambda index: values[index])
    return selected


def chronological_blocks(prediction, target, observed, origins, std):
    require(origins.dtype == np.int64 and origins.ndim == 1 and len(origins) == len(prediction)
            and len(origins) >= 4 and (np.diff(origins) > 0).all(), "SORTED_TUNE_ORIGINS_FOR_FOUR_BLOCKS_REQUIRED")
    result = []
    for block, rows in enumerate(np.array_split(np.arange(len(origins)), 4)):
        value = scores(prediction[rows], target[rows], observed[rows], std)
        result.append({"block": block, "rows": len(rows), "first_origin": int(origins[rows[0]]),
                       "last_origin": int(origins[rows[-1]]), "scores": value})
    return result


def common_information_gate(self_scores, joint_scores, self_blocks, joint_blocks):
    s, j = self_scores["normalized"], joint_scores["normalized"]
    require(len(self_blocks) == len(joint_blocks) == 4, "FOUR_COMMON_CHRONOLOGICAL_BLOCKS_REQUIRED")
    for left, right in zip(self_blocks, joint_blocks):
        require((left["block"], left["rows"], left["first_origin"], left["last_origin"])
                == (right["block"], right["rows"], right["first_origin"], right["last_origin"]),
                "SELF_JOINT_BLOCK_ORIGIN_MISMATCH")
    if s["mse"] <= 1e-8 or s["mae"] <= 1e-8:
        return {"status": "INCONCLUSIVE", "passed": False, "mse_relative_reduction": None,
                "mae_relative_increase": None, "improved_blocks": None,
                "reason": "SELF_BASELINE_AT_OR_BELOW_1E8_THRESHOLD"}
    gain = (s["mse"] - j["mse"]) / s["mse"]
    mae_increase = (j["mae"] - s["mae"]) / s["mae"]
    improved = sum(right["scores"]["normalized"]["mse"] < left["scores"]["normalized"]["mse"]
                   for left, right in zip(self_blocks, joint_blocks))
    return {"status": "PASS" if gain >= .01 and mae_increase <= .01 and improved >= 3 else "FAIL",
            "passed": bool(gain >= .01 and mae_increase <= .01 and improved >= 3),
            "mse_relative_reduction": float(gain), "mae_relative_increase": float(mae_increase),
            "improved_blocks": improved,
            "scope": "exploratory_TUNE_resource_gate_not_significance_or_independent_generalization"}


def state_sha(kind, *values):
    digest = hashlib.sha256(kind.encode())
    for value in values:
        digest.update((array_sha(value) if isinstance(value, np.ndarray) else str(value)).encode())
    return digest.hexdigest()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verified_json(record, check):
    check()
    path = resolve(record["path"])
    expected = record.get("sha256") or record.get("verified_sha256")
    require(expected and file_sha(path) == expected, "JSON_FILE_SHA_MISMATCH:" + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verified_arrays(record, check):
    check()
    path = resolve(record["path"])
    require(file_sha(path) == record["sha256"], "NPZ_FILE_SHA_MISMATCH:" + str(path))
    with np.load(path, allow_pickle=False) as opened:
        arrays = {name: opened[name].copy() for name in opened.files}
    declared = record["arrays"]
    require(set(arrays) == set(declared), "NPZ_ARRAY_KEYS_MISMATCH:" + str(path))
    for name, array in arrays.items():
        metadata = declared[name]
        require(list(array.shape) == metadata["shape"] and str(array.dtype) == metadata["dtype"]
                and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == metadata["sha256"],
                "NPZ_ARRAY_BYTES_MISMATCH:" + str(path) + ":" + name)
    check()
    return arrays


def feature_state_digest(route, metadata, arrays):
    if route == "JOINT33":
        return state_sha("MATCHED_DIRECT_C_FEATURE_STATE", metadata["route"], arrays["center"], arrays["scale"],
                         metadata["input_channels"], metadata["fit_rows"], metadata["fit_raw_features_sha256"])
    return state_sha("SELF_FEATURE_STATE", route, arrays["center"], arrays["scale"],
                     metadata["fit_rows"], metadata["fit_raw_features_sha256"])


SOLUTION_KEYS = ("coefficients", "intercept", "observed_counts", "channel_counts",
                 "normal_equation_denominators", "design_ranks", "regularized_gram_conditions")


def model_digest(route, feature_digest, metadata, arrays):
    if route == "JOINT33":
        solution = state_sha(metadata["kind"], metadata["alpha"], arrays["coefficients"], arrays["intercept"],
                             arrays["observed_counts"], arrays["channel_counts"], feature_digest)
        require(solution == metadata["solution_sha256"], "JOINT_SOLUTION_SHA_MISMATCH")
        return state_sha("MATCHED_DIRECT_C_MODEL", feature_digest, solution)
    return state_sha("SELF_READOUT_MODEL", feature_digest, metadata["alpha"], *(arrays[name] for name in SOLUTION_KEYS))


def audit_result(path, limit):
    """Read only this campaign's completed producer and its declared lineage."""
    check = limit.shared.checkfast
    limit.checkfull()
    path = resolve(path)
    producer_record = {"path": str(path), "sha256": file_sha(path)}
    result = verified_json(producer_record, check)
    dataset = result["dataset"]
    require(dataset in ("robin", "beijing_aotizhongxin") and result["status"] == "COMPLETE", "COMPLETED_APPROVED_PRODUCER_REQUIRED")
    require(result.get("protected_TEST_accessed") is False and result.get("GPU_execution") is False
            and result.get("model_weights_loaded") is False and result.get("old_campaign_restart") is False
            and result.get("independent_holdout") is False, "PRODUCER_SCOPE_DECLARATION_MISMATCH")
    require(result["permission_gate"] == limit.authorization_record
            and result["run_manifest"] == limit.run_manifest_record, "SAME_NEW_CAMPAIGN_REQUIRED")
    final_split = "val" if dataset == "robin" else "eval"
    splits = ("fit", "tune", final_split)
    expected_c = 17 if dataset == "robin" else 11
    expected_rows = {"fit": 256, "tune": 64, final_split: 30 if dataset == "robin" else 146}
    require(set(result["inputs"]) == set(splits) and set(result["families"]) == set(ROUTES), "PREDECLARED_SPLITS_AND_ROUTES_REQUIRED")
    old = verified_json(result["old_producer"], check)
    require(old["dataset"] == dataset and old["seed"] == 92601 and old["status"] == "COMPLETE", "FIXED_OLD_JOINT_PRODUCER_MISMATCH")
    bridge = verified_json(result["bridge_provenance"]["input_bridge"], check)
    phase0_path = HERE / "ENVIRONMENT_ASSET_MANIFEST_v1.json"
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8-sig"))
    locked = [row for row in phase0["dataset_metadata"] if row["dataset"] == ("robin" if dataset == "robin" else "beijing_aotizhongxin")]
    require(len(locked) == 1 and locked[0]["C"] == expected_c, "PHASE0_LOCKED_DATA_CONTRACT_REQUIRED")
    locked = locked[0]
    contract = verified_json({"path": locked["contract_path"], "sha256": locked["contract_SHA256"]}, check)
    require(isinstance(contract, dict), "ORIGINAL_CONTRACT_JSON_REQUIRED")
    source_checked = 0
    for record in result["source_records"]:
        check()
        expected_sha = record.get("sha256") or record.get("verified_sha256")
        require(expected_sha and file_sha(resolve(record["path"])) == expected_sha, "DECLARED_SOURCE_BYTES_CHANGED")
        source_checked += 1
    for record in result["source_snapshots"]:
        check()
        require(file_sha(resolve(record["path"])) == record["sha256"]
                and file_sha(resolve(record["snapshot_path"])) == record["sha256"], "SOURCE_SNAPSHOT_BYTES_CHANGED")
        source_checked += 2
    raw = {}
    raw_checks = {}
    for split in splits:
        arrays = verified_arrays(result["inputs"][split], check)
        require(set(arrays) == {"history", "history_observed", "raw_forecast", "target", "observed", "origins", "raw_last", "mean", "std"},
                "EXACT_RAW_INPUT_KEYS_REQUIRED")
        require(len(arrays["history"]) == expected_rows[split], "EXACT_ORIGINAL_WINDOW_COUNT_REQUIRED")
        raw_checks[split] = check_raw_split(arrays, expected_channels=expected_c, expected_origins=locked["origins"][split])
        lower, upper = locked["bounds"][split]
        require((arrays["origins"] >= lower).all() and (arrays["origins"] + HORIZON <= upper).all(), "FUTURE_TARGET_OUTSIDE_LOCKED_SPLIT")
        if split == "fit":
            require((arrays["origins"] - LENGTH >= lower).all(), "FIT_PAST_OUTSIDE_FIT_PERIOD")
        original_rows = [record for record in old["artifacts"] if Path(record["path"]).stem == "RAW_INPUT_" + split]
        bridge_rows = [record for record in bridge["artifacts"] if record.get("dataset") == dataset and record.get("split") == split]
        require(len(original_rows) == len(bridge_rows) == 1, "ONE_OLD_RAW_AND_BRIDGE_RECORD_REQUIRED")
        for record in (original_rows[0], bridge_rows[0]):
            source = verified_arrays(record, check)
            for name, values in arrays.items():
                exact(values, source[name], "raw_lineage:" + split + ":" + name)
        for name in ("mean", "std"):
            if raw:
                exact(arrays[name], raw["fit"][name], "frozen_raw_FIT_stats:" + split + ":" + name)
        compare_scores(result["references"][split]["RAW_BOLT"],
                       scores(arrays["raw_forecast"], arrays["target"], arrays["observed"], arrays["std"]), "RAW_BOLT:" + split)
        raw[split] = arrays
    shared_mean = verified_arrays(result["shared_mean_artifact"], check)
    mean, mean_count = fit_mean_innovation(raw["fit"]["target"], raw["fit"]["raw_last"], raw["fit"]["observed"])
    exact(shared_mean["innovation"], mean, "shared_FP32_mean")
    exact(shared_mean["observed_counts"], mean_count, "shared_mean_counts")
    controls = {}
    for kind in ("PERSIST", "MEAN"):
        for split in splits:
            controls[kind, split] = (np.repeat(raw[split]["raw_last"][:, None], HORIZON, axis=1) if kind == "PERSIST"
                                     else raw[split]["raw_last"][:, None] + mean)
            compare_scores(result["reference_shared_controls"][kind][split],
                           scores(controls[kind, split], raw[split]["target"], raw[split]["observed"], raw[split]["std"]),
                           "SHARED_CONTROL:" + kind + ":" + split)
    summaries, selected_predictions = {}, {}
    pool_checked, candidate_prediction_checked, selected_prediction_checked = 0, 0, 0
    for route in ROUTES:
        limit.checkfull()
        feature_meta = result["feature_states"][route]
        features = verified_arrays(feature_meta["artifact"], check)
        center, scale = expected_feature_state(route, raw["fit"]["history"])
        exact(features["center"], center, route + ":FIT_feature_center")
        exact(features["scale"], scale, route + ":FIT_feature_population_scale")
        require(feature_meta["fit_rows"] == 256 and feature_meta["input_channels"] == expected_c
                and feature_meta["fit_raw_features_sha256"] == array_sha(raw_features(route, raw["fit"]["history"])),
                "FIT_FEATURE_RAW_ARRAY_DIGEST_MISMATCH")
        feature_digest = feature_state_digest(route, feature_meta, features)
        require(feature_digest == feature_meta["sha256"], "FIT_FEATURE_STATE_DIGEST_MISMATCH")
        design = independent_design(route, raw["fit"]["history"], center, scale)
        family = result["families"][route]
        require(family["status"] == "COMPLETE" and tuple(entry["alpha"] for entry in family["alpha_grid"]) == ALPHAS,
                "COMPLETE_SEVEN_ALPHA_FAMILY_REQUIRED")
        candidate_predictions = {}
        normal_checks = []
        max_prediction_difference = 0.0
        for index, entry in enumerate(family["alpha_grid"]):
            check()
            coefficients = verified_arrays(entry["coefficient_artifact"], check)
            require(set(coefficients) == set(SOLUTION_KEYS), "RIDGE_COEFFICIENT_KEYS_REQUIRED")
            metadata = entry["model_metadata"]
            require(metadata["alpha"] == entry["alpha"] and metadata["feature_state_sha256"] == feature_digest,
                    "RIDGE_MODEL_FEATURE_OR_ALPHA_BINDING_MISMATCH")
            require(model_digest(route, feature_digest, metadata, coefficients) == metadata["model_sha256"]
                    == entry["model_sha256"], "RIDGE_MODEL_DIGEST_MISMATCH")
            counts = raw["fit"]["observed"].sum(axis=0)
            channel_counts = counts.sum(axis=0)
            exact(coefficients["observed_counts"], counts, route + ":output_counts")
            exact(coefficients["channel_counts"], channel_counts, route + ":channel_counts")
            exact(coefficients["normal_equation_denominators"], np.broadcast_to(channel_counts, counts.shape).astype(np.float64), route + ":alpha_Nc")
            ranks = coefficients["design_ranks"]
            require(ranks.dtype == np.int64 and ranks.shape == counts.shape
                    and (ranks >= 0).all() and (ranks <= np.maximum(0, np.minimum(design.shape[1], counts - 1))).all(),
                    "CENTERED_DESIGN_RANK_BOUND_MISMATCH")
            normal_checks.append(normal_equation_checks(route, design, raw["fit"]["target"], raw["fit"]["raw_last"],
                                raw["fit"]["observed"], coefficients["coefficients"], coefficients["intercept"], entry["alpha"]))
            for split in ("fit", "tune"):
                saved = verified_arrays(entry[split + "_artifact"], check)
                exact(saved["origins"], raw[split]["origins"], route + ":candidate_origin")
                expected_prediction = prediction_from_saved(route, raw[split]["history"], raw[split]["raw_last"], center, scale,
                                                            coefficients["coefficients"], coefficients["intercept"])
                max_prediction_difference = max(max_prediction_difference,
                    close(saved["prediction"], expected_prediction, route + ":saved_candidate_prediction"))
                compare_scores(entry[split + "_scores"], scores(saved["prediction"], raw[split]["target"], raw[split]["observed"], raw[split]["std"]),
                               route + ":candidate_scores:" + split)
                candidate_predictions[index, split] = saved["prediction"]
                candidate_prediction_checked += 1
            close(entry["tune_actual_Y_mse"], entry["tune_scores"]["normalized"]["mse"], route + ":primary_selection_alias", rtol=0, atol=0)
        require(len(family["pool"]) == 9, "NINE_POOL_ROWS_REQUIRED")
        for index, entry in enumerate(family["pool"]):
            for split in ("fit", "tune"):
                p = controls[entry["kind"], split] if index < 2 else candidate_predictions[index - 2, split]
                compare_scores(entry[split + "_scores"], scores(p, raw[split]["target"], raw[split]["observed"], raw[split]["std"]),
                               route + ":pool:" + str(index) + ":" + split)
            pool_checked += 1
        selected_index = select_pool(family["pool"])
        chosen = family["pool"][selected_index]
        require(family["selected_index"] == selected_index and family["selected_kind"] == chosen["kind"]
                and family["selected_alpha"] == chosen.get("alpha"), "GLOBAL_MACRO_EARLIEST_POOL_SELECTION_MISMATCH")
        ordered = sorted(row["tune_scores"]["normalized"]["mse"] for row in family["pool"])
        close(family["tune_selection_margin_mse"], ordered[1] - ordered[0], route + ":selection_margin", rtol=1e-12, atol=1e-12)
        for split in splits:
            saved = verified_arrays(family["selected_artifacts"][split], check)
            for key in ("target", "observed", "origins", "raw_last", "mean", "std"):
                exact(saved[key], raw[split][key], route + ":selected_same_contract:" + split + ":" + key)
            if selected_index < 2:
                expected_prediction = controls[chosen["kind"], split]
            elif split in ("fit", "tune"):
                expected_prediction = candidate_predictions[selected_index - 2, split]
            else:
                chosen_coefficients = verified_arrays(chosen["coefficient_artifact"], check)
                expected_prediction = prediction_from_saved(route, raw[split]["history"], raw[split]["raw_last"], center, scale,
                                                            chosen_coefficients["coefficients"], chosen_coefficients["intercept"])
            close(saved["prediction"], expected_prediction, route + ":selected_prediction:" + split)
            compare_scores(family["splits"][split], scores(saved["prediction"], raw[split]["target"], raw[split]["observed"], raw[split]["std"]),
                           route + ":selected_scores:" + split)
            selected_predictions[route, split] = saved["prediction"]
            selected_prediction_checked += 1
        if chosen["kind"] == "RIDGE":
            require(family["selected_model_metadata"] == chosen["model_metadata"]
                    and family["selected_coefficient_artifact"] == chosen["coefficient_artifact"], "SELECTED_MODEL_BINDING_MISMATCH")
        summaries[route] = {"selected_index": selected_index, "selected_kind": chosen["kind"], "selected_alpha": chosen.get("alpha"),
                            "max_saved_prediction_absolute_difference": max_prediction_difference, "normal_equations": normal_checks,
                            "feature_digest": feature_digest, "feature_count": design.shape[1]}
    tune = raw["tune"]
    independent_blocks = {route: chronological_blocks(selected_predictions[route, "tune"], tune["target"], tune["observed"], tune["origins"], tune["std"])
                          for route in ("SELF33", "JOINT33")}
    independent_gate = common_information_gate(result["families"]["SELF33"]["splits"]["tune"], result["families"]["JOINT33"]["splits"]["tune"],
                                             independent_blocks["SELF33"], independent_blocks["JOINT33"])
    lock = verified_json(result["branch_lock"], check)
    require(lock["status"] == "LOCKED_BEFORE_DEV_PAYLOAD_READ" and lock["scientific_inputs_only"] == ["FIT", "TUNE"]
            and lock["DEV_payload_opened_before_lock"] is False
            and lock["Raw_and_SELF512_method_adoption_is_a_separate_gate"] is True, "FIT_TUNE_BRANCH_LOCK_REQUIRED")
    gate = lock["common_information_gate"]
    for stored, route in ((gate["self_tune"], "SELF33"), (gate["joint_tune"], "JOINT33")):
        expected = result["families"][route]["splits"]["tune"]["normalized"]
        for metric in ("mse", "mae", "per_channel_mse", "per_channel_mae"):
            close(stored[metric], expected[metric], "gate_bound_normalized_score:" + route + ":" + metric, rtol=1e-12, atol=1e-12)
        require(stored["observed_counts"] == expected["observed_counts"]
                and stored["active_channels"] == expected["active_channels"], "GATE_SCORE_OBSERVABILITY_BINDING_MISMATCH")
    require(gate["passed"] == independent_gate["passed"] and gate["ratio_eligible"] == (independent_gate["status"] != "INCONCLUSIVE")
            and gate["exploratory_resource_gate_only"] is True and gate["independent_generalization_proven"] is False, "COMMON_INFORMATION_GATE_SCOPE_MISMATCH")
    if independent_gate["status"] != "INCONCLUSIVE":
        close(gate["mse_gain_pct"], independent_gate["mse_relative_reduction"] * 100, "gate_mse_gain_pct", rtol=1e-11, atol=1e-11)
        close(gate["mae_change_pct"], independent_gate["mae_relative_increase"] * 100, "gate_mae_change_pct", rtol=1e-11, atol=1e-11)
    else:
        require(gate["mse_gain_pct"] is None and gate["mae_change_pct"] is None, "GATE_INCONCLUSIVE_RATIOS_MUST_BE_NULL")
    for index, rows in enumerate(np.array_split(np.arange(len(tune["origins"])), 4)):
        recorded = gate["four_chronological_blocks"][index]
        own, joint = independent_blocks["SELF33"][index], independent_blocks["JOINT33"][index]
        require(recorded["indexes"] == rows.tolist(), "GATE_CHRONOLOGICAL_BLOCK_INDEX_MISMATCH")
        close(recorded["self_mse"], own["scores"]["normalized"]["mse"], "block_SELF33_mse", rtol=1e-12, atol=1e-12)
        close(recorded["joint_mse"], joint["scores"]["normalized"]["mse"], "block_JOINT33_mse", rtol=1e-12, atol=1e-12)
        require(recorded["joint_improved"] == (joint["scores"]["normalized"]["mse"] < own["scores"]["normalized"]["mse"]), "BLOCK_IMPROVEMENT_MISMATCH")
    fit_self = result["families"]["SELF33"]["splits"]["fit"]["normalized"]["mse"]
    fit_joint = result["families"]["JOINT33"]["splits"]["fit"]["normalized"]["mse"]
    overfit = fit_joint < fit_self and gate["joint_tune"]["mse"] >= gate["self_tune"]["mse"]
    expected_branch = "SMALLK_PRESERVATION_COMPARISON" if independent_gate["passed"] else "MATCHED_FIT512_ONE_DIAGNOSTIC" if overfit else "STOP_COMMON_REPRESENTATION_WEAK_EVIDENCE"
    require(lock["joint_fit_good_tune_bad"] == bool(overfit) and lock["next_branch"] == expected_branch,
            "FIT_TUNE_ADAPTIVE_BRANCH_MISMATCH")
    for route in ROUTES:
        require(lock["selected_routes"][route] == {name: result["families"][route][name] for name in ("selected_index", "selected_kind", "selected_alpha")},
                "BRANCH_SELECTED_ROUTE_BINDING_MISMATCH")
    require(result["pool_comparisons"] == pool_checked == 27
            and result["new_model_bundles"] + result["reused_model_bundles"] == 21, "ACTUAL_MODEL_AND_POOL_COUNTS_MISMATCH")
    limit.checkfull()
    return {"dataset": dataset, "producer_sha256": producer_record["sha256"], "producer_record": {"dataset": dataset, **producer_record},
            "status": "PASS", "raw_splits_checked": raw_checks, "routes": summaries,
            "source_file_bindings_checked": source_checked,
            "pool_candidates_checked": pool_checked, "candidate_predictions_checked": candidate_prediction_checked,
            "selected_predictions_checked": selected_prediction_checked, "independent_common_gate": independent_gate,
            "independent_chronological_blocks": independent_blocks, "next_branch": expected_branch,
            "new_model_bundles": result["new_model_bundles"], "reused_model_bundles": result["reused_model_bundles"],
            "old_raw_bridge_payloads_byte_equal": True, "Phase0_origin_bounds_exact": True,
            "original_raw_preprocessing_reconstructed": False, "E_F_D_forward_replayed": False,
            "alpha0_minimum_norm_reproved": False, "condition_spectrum_recomputed": False,
            "read_order_proved_by_file_bytes": False,
            "branch_lock_observation": "source-order/declarations and immutable FIT/TUNE lock checked; file bytes alone cannot prove all process reads"}


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-result", action="append", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"scientific_execution": False, "scope": "saved-array independent readout audit"}))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.path[:0] = [str(HERE), str(ROOT / "research/smallk_failure_followup_20261014"), str(ROOT)]
    from adaptive_runtime_v1 import ContinuationLimit
    limit = ContinuationLimit(args.tag, gpu=False)
    output = resolve(args.output)
    report = {"status": "RUNNING", "scientific_execution": True, "conditions": [], "producer_records": [],
              "permission_gate": limit.authorization_record, "run_manifest": limit.run_manifest_record,
              "scope": {"saved_array_math": True, "protected_TEST_accessed": False, "GPU_execution": False,
                        "E_F_D_forward_replayed": False, "independent_holdout": False,
                        "original_raw_preprocessing_reconstructed": False, "alpha0_minimum_norm_reproved": False}}
    error = None
    try:
        require(output.is_relative_to(HERE) and not output.exists(), "NEW_CAMPAIGN_AUDIT_OUTPUT_ONLY")
        initialize_numpy()
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            for path in args.input_result:
                summary = audit_result(path, limit)
                report["conditions"].append(summary)
                report["producer_records"].append(summary["producer_record"])
        require(len({entry["dataset"] for entry in report["conditions"]}) == len(report["conditions"]), "ONE_DETERMINISTIC_CONDITION_PER_DATASET")
        report["totals"] = {key: sum(row[key] for row in report["conditions"]) for key in
                            ("pool_candidates_checked", "candidate_predictions_checked", "selected_predictions_checked",
                             "new_model_bundles", "reused_model_bundles", "source_file_bindings_checked")}
        limit.checkfull()
        report["status"] = "PASS"
    except Exception as failure:
        import traceback
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        report["status"] = "FAIL"
        report["failure"] = error
    finally:
        terminal = limit.close("COMPLETE" if report["status"] == "PASS" else "FAILED_PRESERVED", error)
        report["resource_terminal"] = terminal
        if terminal["terminal_status"] != "COMPLETE":
            report["status"] = "FAIL"
        if output.is_relative_to(HERE) and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"audit": str(output), "status": report["status"], "totals": report.get("totals"), "failure": error}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
