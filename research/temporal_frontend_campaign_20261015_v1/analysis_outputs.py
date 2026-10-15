"""CPU-only post-score descriptions; no fitting, F inference or selection changes.

Run only after ROOT has completed C scoring and explicitly coordinated the CPU
analysis stage. Frozen source/cost/selection receipts are checked before arrays
are opened. Pattern definitions are in patterns.py; groups use 32 evenly spaced
A origins, not the full population of A windows. Future spike labels are strictly
posthoc descriptions and are never inputs to a selector or a deployed model.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import data as dio
from models import TemporalFront
import patterns
import runtime as rt


OUTPUT_NAMES = ("PATTERN_GROUPS.csv", "ERROR_DIAGNOSTICS.csv", "LATENT_DIAGNOSTICS.csv",
                "PATTERN_THRESHOLDS.json", "ANALYSIS_COMPLETE.json")
PATTERN_HISTORY = 512
HORIZON = 48


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for requested output {Path(path).name}")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_path(relative):
    result = (rt.CACHE / relative).resolve()
    if not result.is_relative_to(rt.CACHE.resolve()):
        raise ValueError("Artifact outside this campaign cache")
    return result


def verify_gates():
    """Read receipts/source metadata only; no data or model arrays are opened."""
    complete = _read(rt.HERE / "SCORE_COMPLETE.json")
    cost = _read(rt.HERE / "COST_LOCK.json")
    selection = _read(rt.HERE / "SELECTION_LOCK.json")
    first_c = _read(rt.HERE / "FIRST_C_SCORE.json")
    assert complete["status"] == "COMPLETE_VALID" and complete["F_unchanged"], "SCORING_NOT_COMPLETE_VALID"
    assert cost["status"] == "COMPLETE_VALID", "COST_NOT_COMPLETE_VALID"
    assert selection["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_C", "SELECTION_NOT_LOCKED"
    selection_digest = _digest(rt.HERE / "SELECTION_LOCK.json")
    assert cost["selection_lock"] == selection_digest == first_c["selection_lock"], "SELECTION_RECEIPT_MISMATCH"
    assert first_c["cost_lock"] == _digest(rt.HERE / "COST_LOCK.json"), "COST_RECEIPT_MISMATCH"
    assert cost["general_cost"] == _digest(rt.HERE / "GENERAL_COST_COMPLETE.json"), "GENERAL_COST_RECEIPT_CHANGED"
    assert cost["cost_selection_sha256"] == _digest(rt.HERE / "COST_SELECTION.json"), "COST_SELECTION_CHANGED"
    for name, digest in selection["source_hashes"].items():
        source = (rt.HERE / name).resolve()
        assert source.is_relative_to(rt.HERE.resolve()), "SOURCE_PATH_OUTSIDE_CAMPAIGN"
        assert _digest(source) == digest, "LOCKED_SOURCE_CHANGED:" + name
    for relative, digest in selection["artifact_hashes"].items():
        assert _digest(_cache_path(relative)) == digest, "SELECTED_ARTIFACT_CHANGED:" + relative
    assert _digest(rt.HERE / "FOLD_STATISTICS.json") == selection["statistics_hash"], "FOLD_STATISTICS_CHANGED"
    return selection, complete


def masked_metrics(prediction, target, mask):
    """Original observed targets; macro channels, with empty-group values null."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask shapes differ")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite observed prediction/target")
    counts = mask.sum(axis=(0, 1))
    active = counts > 0
    error = prediction - target
    mse = np.divide(np.where(mask, error * error, 0).sum(axis=(0, 1)), np.maximum(counts, 1))
    mae = np.divide(np.where(mask, np.abs(error), 0).sum(axis=(0, 1)), np.maximum(counts, 1))
    return {
        "mse": float(mse[active].mean()) if active.any() else None,
        "mae": float(mae[active].mean()) if active.any() else None,
        "n_observed": int(counts.sum()), "active_channels": int(active.sum()),
        "channel_counts": [int(value) for value in counts],
        "channel_mse": [float(value) if valid else None for value, valid in zip(mse, active)],
        "channel_mae": [float(value) if valid else None for value, valid in zip(mae, active)],
    }


def _windows(data, origins, mean, std):
    return np.stack([dio.history_window(data, int(origin), mean, std, PATTERN_HISTORY)[0] for origin in origins])


def _even_A_origins(origins):
    origins = np.asarray(origins, dtype=np.int64)
    if not len(origins):
        return origins
    indices = np.unique(np.linspace(0, len(origins) - 1, min(32, len(origins))).astype(np.int64))
    return origins[indices]


def _spike_threshold(data, fold, std):
    lo, hi = fold["bounds"]["A"]
    physical = data["physical"][lo:hi].astype(np.float64)
    mask = data["observed"][lo:hi]
    observed_pairs = mask[1:] & mask[:-1]
    differences = np.abs(physical[1:] - physical[:-1]) / std
    thresholds, counts = [], []
    for channel in range(physical.shape[1]):
        values = differences[:, channel][observed_pairs[:, channel]]
        thresholds.append(float(np.quantile(values, 0.95, method="linear")) if len(values) else None)
        counts.append(int(len(values)))
    return thresholds, counts


def _channel_summary(values, valid):
    counts = valid.sum(axis=0)
    per_channel = np.divide(np.where(valid, values, 0).sum(axis=0), np.maximum(counts, 1))
    active = counts > 0
    return {"macro": float(per_channel[active].mean()) if active.any() else None,
            "channel_values": [float(value) if keep else None for value, keep in zip(per_channel, active)],
            "channel_pairs": [int(value) for value in counts], "n_origin_channel_pairs": int(counts.sum())}


def error_diagnostics(prediction, target, mask, origins, data, mean, std, spike_thresholds):
    """C-only posthoc event/error descriptions with strict original observation masks."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    enough = mask.sum(axis=1) >= 2
    predicted_peak = np.where(mask, prediction, -np.inf).argmax(axis=1)
    target_peak = np.where(mask, target, -np.inf).argmax(axis=1)
    peak_error = np.abs(predicted_peak - target_peak)
    predicted_max = np.where(mask, prediction, -np.inf).max(axis=1)
    predicted_min = np.where(mask, prediction, np.inf).min(axis=1)
    target_max = np.where(mask, target, -np.inf).max(axis=1)
    target_min = np.where(mask, target, np.inf).min(axis=1)
    predicted_range = np.subtract(predicted_max, predicted_min, out=np.zeros_like(predicted_max), where=enough)
    target_range = np.subtract(target_max, target_min, out=np.zeros_like(target_max), where=enough)
    amplitude_error = np.abs(predicted_range - target_range)
    peak = _channel_summary(peak_error, enough)
    amplitude = _channel_summary(amplitude_error, enough)
    previous_observed = np.concatenate((data["observed"][origins - 1, None, :], mask[:, :-1]), axis=1)
    previous_target = np.concatenate((((data["physical"][origins - 1].astype(np.float64) - mean) / std)[:, None, :], target[:, :-1]), axis=1)
    differences = np.abs(target - previous_target)
    threshold = np.asarray([np.nan if value is None else value for value in spike_thresholds])
    classified = mask & previous_observed & np.isfinite(threshold)[None, None, :]
    spike_mask = classified & (differences > threshold[None, None, :])
    spike = masked_metrics(prediction, target, spike_mask)
    non_spike = masked_metrics(prediction, target, classified & ~spike_mask)
    unavailable = masked_metrics(prediction, target, mask & ~classified)
    return {
        "peak_time_absolute_error_steps_macro": peak["macro"],
        "peak_time_absolute_error_seconds_macro": None if peak["macro"] is None else peak["macro"] * data["interval_seconds"],
        "peak_time_channel_values_json": _json(peak["channel_values"]),
        "normalized_range_amplitude_absolute_error_macro": amplitude["macro"],
        "normalized_range_amplitude_channel_values_json": _json(amplitude["channel_values"]),
        "peak_amplitude_valid_origin_channel_pairs": peak["n_origin_channel_pairs"],
        "peak_amplitude_channel_pairs_json": _json(peak["channel_pairs"]),
        "spike_mse": spike["mse"], "spike_mae": spike["mae"], "spike_n_observed": spike["n_observed"],
        "spike_active_channels": spike["active_channels"], "spike_channel_mse_json": _json(spike["channel_mse"]),
        "spike_channel_observed_json": _json(spike["channel_counts"]),
        "non_spike_mse": non_spike["mse"], "non_spike_mae": non_spike["mae"], "non_spike_n_observed": non_spike["n_observed"],
        "adjacent_difference_unavailable_mse": unavailable["mse"], "adjacent_difference_unavailable_n_observed": unavailable["n_observed"],
        "spike_thresholds_normalized_json": _json(spike_thresholds),
        "peak_amplitude_definition": "argmax timing and max-minus-min range on each original observed forecast grid; >=2 observations/pair; ties use earliest observed argmax",
        "spike_definition": "abs actual adjacent difference > A-only channel observed-pair 95th percentile; first forecast compares only an originally observed o-1",
        "event_role": "future-defined posthoc diagnostic; never model input or selector; overlapping target exposures remain dependent",
    }


def _svd_summary(value):
    value = np.asarray(value, dtype=np.float32)
    singular = np.linalg.svd(value, compute_uv=False)
    tolerance = float(singular[0] * max(value.shape) * np.finfo(np.float32).eps) if len(singular) else 0.0
    rank = int((singular > tolerance).sum())
    return {"shape": list(value.shape), "singular_values": [float(number) for number in singular],
            "rank": rank, "rank_tolerance": tolerance,
            "condition_number": float(singular[0] / singular[-1]) if len(singular) and singular[-1] > tolerance else None}


def latent_diagnostics(spec, data, mean, std, B_origins):
    """Rebuild only the small adapter on CPU; never instantiate/call frozen F."""
    checkpoint = _cache_path(spec["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(value.device.type != "cpu" for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("Checkpoint escaped CPU map_location")
    basis = state["E"].detach().numpy()
    center = state["center"].detach().numpy()
    front = TemporalFront(spec["kind"], length=PATTERN_HISTORY, channels=data["physical"].shape[1], **spec.get("front_kwargs", {}))
    model = rt.Adapter(front, center, basis).cpu().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    origins = np.asarray(B_origins[:32], dtype=np.int64)
    x = torch.from_numpy(_windows(data, origins, mean, std))
    with torch.no_grad():
        z = model.encode(x).numpy()
        transformed = model.front(x).numpy()
    flat = z.reshape(-1, z.shape[-1])
    centered = flat - flat.mean(axis=0)
    E = _svd_summary(model.E.detach().numpy())
    D = _svd_summary(model.D.detach().numpy())
    latent = _svd_summary(centered)
    return {
        "id": spec["id"], "dataset": spec["dataset"], "fold": spec["fold"], "kind": spec["kind"], "r": int(spec["r"]),
        "seed": int(spec.get("seed", 0)), "selected_update": int(spec.get("selected_update", 0)),
        "training_status": spec.get("training_status", "NOT_RECORDED"), "B_origins": len(origins), "B_origins_json": _json(origins.tolist()),
        "B_cohort_sha256": _array_digest(origins), "checkpoint_sha256": _digest(checkpoint),
        "latent_variance_per_stream_json": _json([float(value) for value in flat.var(axis=0, ddof=0)]),
        "latent_variance_mean": float(flat.var(axis=0, ddof=0).mean()),
        "transformed_variance_per_channel_json": _json([float(value) for value in transformed.reshape(-1, transformed.shape[-1]).var(axis=0, ddof=0)]),
        "latent_centered_rank": latent["rank"], "latent_rank_tolerance": latent["rank_tolerance"],
        "latent_singular_values_json": _json(latent["singular_values"]),
        "E_rank": E["rank"], "E_rank_tolerance": E["rank_tolerance"], "E_singular_values_json": _json(E["singular_values"]),
        "E_condition_number": E["condition_number"], "D_rank": D["rank"], "D_rank_tolerance": D["rank_tolerance"],
        "D_singular_values_json": _json(D["singular_values"]), "D_condition_number": D["condition_number"],
        "execution_device": "cpu", "F_loaded_or_called": False,
        "interpretation": "selected-CP B-first32 description only; repeated historical rows are not independent; E/D or latent rank is not a temporal-method impossibility bound or performance proof",
    }


def run(output_folder=None):
    start = time.perf_counter()
    selection, score_complete = verify_gates()
    output = (rt.HERE if output_folder is None else Path(output_folder)).resolve()
    if not output.is_relative_to(rt.HERE.resolve()):
        raise ValueError("Output folder must remain inside this campaign")
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        if (output / name).exists():
            raise FileExistsError(output / name)
    with (rt.HERE / "SCORES.csv").open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    assert len(score_rows) == score_complete["rows"], "SCORES_ROW_COUNT_MISMATCH"
    assert all(row["status"] == "COMPLETE_VALID" for row in score_rows), "INVALID_SCORE_ROW"
    specs = {spec["id"]: spec for spec in selection["routes"]}
    statistics = _read(rt.HERE / "FOLD_STATISTICS.json")
    contexts, description_cache, archive_hashes = {}, {}, {}
    group_rows, error_rows, latent_rows, thresholds_record = [], [], [], {}

    def context(dataset, fold_name):
        key = (dataset, fold_name)
        if key not in contexts:
            data = dio.load_dataset(dataset)
            fold = dio.make_fold(data, fold=fold_name)
            mean, std, counts = dio.fit_statistics(data, fold["bounds"]["A"])
            locked = statistics[dataset][fold_name]
            np.testing.assert_array_equal(mean, np.asarray(locked["A_mean"], np.float32))
            np.testing.assert_array_equal(std, np.asarray(locked["A_std"], np.float32))
            np.testing.assert_array_equal(counts, np.asarray(locked["A_true_observed_counts"], np.int64))
            A_origins = _even_A_origins(fold["origins"]["A"])
            A_values = patterns.indicators(_windows(data, A_origins, mean, std), data["period_from_existing_metadata"])
            thresholds = patterns.fit_groups(A_values)
            spike_thresholds, spike_counts = _spike_threshold(data, fold, std)
            contexts[key] = (data, fold, mean, std, thresholds, spike_thresholds)
            thresholds_record[f"{dataset}/{fold_name}"] = {
                "A_origin_count_available": len(fold["origins"]["A"]), "A_origins_used": A_origins.tolist(), "A_sample_count": len(A_origins),
                "A_sampling": "up to 32 uniform indices across all legal A origins, fixed before inspecting forecast errors",
                "threshold_scope_limitation": "median of these <=32 past windows, not full-A window population; this is an exploratory description",
                "pattern_history_length": PATTERN_HISTORY, "thresholds": thresholds,
                "A_finite_indicator_counts": {name: int(np.isfinite(values).sum()) for name, values in A_values.items()},
                "A_indicator_values": {name: [float(value) if np.isfinite(value) else None for value in values] for name, values in A_values.items()},
                "spike_thresholds_normalized": spike_thresholds, "A_observed_consecutive_difference_counts": spike_counts,
                "definitions": patterns.indicator_definitions(data["period_from_existing_metadata"], PATTERN_HISTORY),
                "historically_unseen": False,
            }
        return contexts[key]

    for row in score_rows:
        identifier, split = row["id"], row["split"]
        if split not in ("B", "C"):
            continue
        if identifier in specs:
            spec = specs[identifier]
            prediction_key = f"{split}_pred"
        else:
            base_id, marker, suffix = identifier.rpartition("_matched_RawL")
            assert marker and base_id in specs and suffix.isdigit(), "UNBOUND_SCORE_ID:" + identifier
            spec = specs[base_id]
            prediction_key = f"{split}_matched_RawL{int(suffix)}"
        data, fold, mean, std, thresholds, spike_thresholds = context(row["dataset"], row["fold"])
        archive_path = _cache_path(f"predictions/{spec['id']}.npz")
        if str(archive_path) not in archive_hashes:
            archive_hashes[str(archive_path)] = _digest(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            prediction = archive[prediction_key]
            target, mask, origins = (archive[f"{split}_{name}"] for name in ("target", "mask", "origins"))
        assert mask.dtype == np.bool_ and prediction.shape == target.shape == mask.shape, "CACHE_SHAPE_OR_MASK_INVALID"
        assert target.shape[1:] == (HORIZON, data["physical"].shape[1]), "CACHE_HORIZON_CHANNEL_INVALID"
        assert _array_digest(origins) == row["cohort"], "CACHE_COHORT_MISMATCH"
        expected = fold["origins"][split]
        expected = expected[expected >= int(spec.get("length", 512))]
        np.testing.assert_array_equal(origins, expected)
        true_targets, true_masks = zip(*(dio.target_window(data, int(origin), mean, std) for origin in origins))
        np.testing.assert_array_equal(mask, np.stack(true_masks))
        np.testing.assert_allclose(target, np.stack(true_targets), atol=1e-6, rtol=2e-6)
        overall = masked_metrics(prediction, target, mask)
        assert overall["n_observed"] == int(row["n_observed"]), "OBSERVED_COUNT_MISMATCH"
        np.testing.assert_allclose([overall["mse"], overall["mae"]], [float(row["mse"]), float(row["mae"])], atol=1e-10, rtol=1e-9)
        description_key = (row["dataset"], row["fold"], split, row["cohort"])
        if description_key not in description_cache:
            description_cache[description_key] = patterns.indicators(_windows(data, origins, mean, std), data["period_from_existing_metadata"])
        groups = patterns.assign_groups(description_cache[description_key], thresholds)
        for label, included in groups.items():
            indicator, group = label.rsplit("__", 1)
            metric = masked_metrics(prediction[included], target[included], mask[included])
            group_rows.append({
                "id": identifier, "dataset": row["dataset"], "fold": row["fold"], "split": split,
                "method": row["method"], "kind": row["kind"], "r": int(row["r"]), "seed": int(row["seed"]),
                "cohort": row["cohort"], "model_history_length": int(row["length"]), "indicator_history_length": PATTERN_HISTORY,
                "indicator": indicator, "group": group, "A_median_threshold": thresholds[indicator],
                "n_origins": int(included.sum()), "total_cohort_origins": len(origins), "mse": metric["mse"], "mae": metric["mae"],
                "n_observed": metric["n_observed"], "active_channels": metric["active_channels"],
                "channel_counts_json": _json(metric["channel_counts"]), "channel_mse_json": _json(metric["channel_mse"]),
                "channel_mae_json": _json(metric["channel_mae"]), "boundary_fit_origins": thresholds_record[f"{row['dataset']}/{row['fold']}"]["A_sample_count"],
                "interpretation": "exploratory past-only groups; fixed A32 medians; all low/high/unavailable groups preserved; no selector fit",
            })
        if split == "C":
            diagnostic = error_diagnostics(prediction, target, mask, origins, data, mean, std, spike_thresholds)
            error_rows.append({"id": identifier, "dataset": row["dataset"], "fold": row["fold"], "split": split,
                               "method": row["method"], "kind": row["kind"], "r": int(row["r"]), "seed": int(row["seed"]),
                               "cohort": row["cohort"], "n_origins": len(origins), "n_observed": overall["n_observed"],
                               "mse": overall["mse"], "mae": overall["mae"], **diagnostic})

    for spec in selection["routes"]:
        if spec["type"] != "ADAPTER":
            continue
        data, fold, mean, std, _, _ = context(spec["dataset"], spec["fold"])
        latent_rows.append(latent_diagnostics(spec, data, mean, std, fold["origins"]["B"]))
    _write_csv(output / "PATTERN_GROUPS.csv", group_rows)
    _write_csv(output / "ERROR_DIAGNOSTICS.csv", error_rows)
    _write_csv(output / "LATENT_DIAGNOSTICS.csv", latent_rows)
    _write_json(output / "PATTERN_THRESHOLDS.json", thresholds_record)
    receipt = {
        "status": "COMPLETE_VALID", "device": "cpu", "F_loaded_or_called": False,
        "wall_seconds": time.perf_counter() - start,
        "pattern_group_rows": len(group_rows), "error_diagnostic_rows": len(error_rows), "latent_diagnostic_rows": len(latent_rows),
        "analysis_source_sha256": _digest(Path(__file__)), "patterns_source_sha256": _digest(rt.HERE / "patterns.py"),
        "score_complete_sha256": _digest(rt.HERE / "SCORE_COMPLETE.json"), "scores_sha256": _digest(rt.HERE / "SCORES.csv"),
        "selection_lock_sha256": _digest(rt.HERE / "SELECTION_LOCK.json"), "cost_lock_sha256": _digest(rt.HERE / "COST_LOCK.json"),
        "prediction_archive_sha256": {str(Path(path).relative_to(rt.CACHE.resolve())): digest for path, digest in archive_hashes.items()},
        "output_sha256": {name: _digest(output / name) for name in OUTPUT_NAMES if name != "ANALYSIS_COMPLETE.json"},
        "validation": "locked sources/artifacts/statistics, cost-before-C receipts, every scored cache cohort/observed target and normalized MSE/MAE recomputed",
        "limitations": ["A pattern boundaries use <=32 evenly spaced historical windows", "historically exposed FIT C is not a new independent holdout",
                        "future spike groups are posthoc descriptions", "overlapping forecasts/channels/seeds are not independent samples",
                        "rank/latent variance do not prove usefulness or a temporal-method impossibility bound", "timers/training were not independently rerun by this analysis"],
    }
    _write_json(output / "ANALYSIS_COMPLETE.json", receipt)
    print(_json({key: receipt[key] for key in ("status", "wall_seconds", "pattern_group_rows", "error_diagnostic_rows", "latent_diagnostic_rows")}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run(arguments.output_dir)
