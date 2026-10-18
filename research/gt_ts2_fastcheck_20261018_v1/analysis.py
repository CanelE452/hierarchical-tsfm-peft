"""Independent FP64 forecast statistics and pre-error TRAIN sampling order."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _errors(prediction, target, mask) -> tuple[np.ndarray, np.ndarray]:
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction, target and mask must share [B,H,C] shape")
    if any(size == 0 for size in prediction.shape) or mask.dtype != np.bool_:
        raise ValueError("Forecast arrays must be nonempty and observation mask must be bool")
    if prediction.dtype.kind not in "fiu" or target.dtype.kind not in "fiu":
        raise ValueError("Forecasts and targets must be real numeric arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Forecast or observed target contains NaN/Inf")
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    if not np.isfinite(error).all():
        raise ValueError("FP64 forecast differences are nonfinite")
    return error, mask


def _totals(error: np.ndarray, mask: np.ndarray, axes: tuple[int, ...]):
    count = mask.sum(axis=axes, dtype=np.int64)
    if np.any(count <= 0):
        raise ValueError("Each fixed channel needs observed targets in this scoring unit")
    with np.errstate(over="ignore", invalid="ignore"):
        sse = np.square(error).sum(axis=axes, dtype=np.float64)
        sae = np.abs(error).sum(axis=axes, dtype=np.float64)
    if not np.isfinite(sse).all() or not np.isfinite(sae).all():
        raise ValueError("FP64 error sufficient statistics are nonfinite")
    return sse, sae, count


def _macro(values: np.ndarray, axis=None):
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.mean(axis=axis, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("FP64 channel-macro score is nonfinite")
    return result


def score(prediction, target, mask) -> dict[str, Any]:
    """Aggregate channel sufficient statistics before taking their equal mean."""
    error, mask = _errors(prediction, target, mask)
    sse, sae, count = _totals(error, mask, (0, 1))
    mse, mae = sse / count, sae / count
    return {
        "mse": float(_macro(mse)), "mae": float(_macro(mae)),
        "channel_sse": sse.tolist(), "channel_sae": sae.tolist(), "channel_count": count.tolist(),
        "channel_mse": mse.tolist(), "channel_mae": mae.tolist(), "target_count": int(count.sum()),
    }


def _origins(origins, batch_size: int) -> np.ndarray:
    origins = np.asarray(origins)
    if origins.shape != (batch_size,) or origins.dtype.kind not in "iu":
        raise ValueError("Origins must contain one integer forecast timestamp per prediction")
    return origins


def _origin_rows(error: np.ndarray, mask: np.ndarray, origins: np.ndarray, run_id: str):
    sse, sae, count = _totals(error, mask, (1,))
    mse, mae = _macro(sse / count, axis=1), _macro(sae / count, axis=1)
    return [{
        "run_id": run_id, "origin": int(origin), "mse": float(mse[i]), "mae": float(mae[i]),
        "count": int(count[i].sum()), "channel_count": count[i].tolist(),
    } for i, origin in enumerate(origins)]


def statistics(prediction, target, mask, origins, run_id: str):
    """Return channel/horizon sufficient statistics and descriptive origin scores.

    TOTAL is reconstructed by adding the three disjoint horizon bands. Origin
    scores describe overlapping windows; they are not independent samples and
    their mean need not equal the aggregate channel-macro score.
    """
    error, mask = _errors(prediction, target, mask)
    if error.shape[1] != 48:
        raise ValueError("Registered horizon-band statistics require H=48")
    origins = _origins(origins, len(error))
    bands = (("TOTAL", 0, 48), ("1-16", 0, 16), ("17-32", 16, 32), ("33-48", 32, 48))
    totals = [(name, _totals(error[:, start:end], mask[:, start:end], (0, 1)))
              for name, start, end in bands]
    channel_rows = []
    for channel in range(error.shape[2]):
        for name, (sse, sae, count) in totals:
            channel_rows.append({
                "run_id": run_id, "channel": channel, "horizon_band": name,
                "sse": float(sse[channel]), "sae": float(sae[channel]), "count": int(count[channel]),
                "mse": float(sse[channel] / count[channel]), "mae": float(sae[channel] / count[channel]),
            })
    return channel_rows, _origin_rows(error, mask, origins, run_id)


def _bundle(source) -> dict[str, np.ndarray]:
    fields = ("prediction", "target", "mask", "origins", "mean", "std")
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as archive:
            if any(key not in archive for key in fields):
                raise ValueError("Prediction archive lacks a cohort or scaling field")
            bundle = {key: archive[key] for key in fields}
    elif isinstance(source, Mapping):
        if any(key not in source for key in fields):
            raise ValueError("Prediction bundle lacks a cohort or scaling field")
        bundle = {key: np.asarray(source[key]) for key in fields}
    else:
        raise ValueError("Expected an NPZ path or prediction array mapping")
    error, _ = _errors(bundle["prediction"], bundle["target"], bundle["mask"])
    _origins(bundle["origins"], len(error))
    channels = error.shape[-1]
    if any(bundle[key].shape != (channels,) for key in ("mean", "std")):
        raise ValueError("Scaling vectors must have one value per ordered physical channel")
    if (not np.isfinite(bundle["mean"]).all() or not np.isfinite(bundle["std"]).all()
            or np.any(bundle["std"] <= 0)):
        raise ValueError("Scaling vectors must be finite with positive standard deviations")
    return bundle


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def paired(candidate, reference):
    """Compare only byte-identical target/mask/origin/normalization cohorts.

    Relative changes are percentages of the reference and use original FP64
    aggregate values. Negative changes mean lower error. No significance or
    independent-sample interpretation is attached to the overlapping origins.
    """
    candidate, reference = _bundle(candidate), _bundle(reference)
    if candidate["prediction"].shape != reference["prediction"].shape:
        raise ValueError("Paired prediction shapes differ")
    cohort_hashes = {}
    for key in ("target", "mask", "origins", "mean", "std"):
        cand, ref = candidate[key], reference[key]
        if cand.shape != ref.shape or cand.dtype != ref.dtype or _array_hash(cand) != _array_hash(ref):
            raise ValueError(f"Paired cohort or scaling coordinate differs: {key}")
        cohort_hashes[key] = _array_hash(cand)
    candidate_score = score(candidate["prediction"], candidate["target"], candidate["mask"])
    reference_score = score(reference["prediction"], reference["target"], reference["mask"])
    if reference_score["mse"] <= 0 or reference_score["mae"] <= 0:
        raise ValueError("Relative error changes need positive reference MSE and MAE")
    summary = {
        "cohort_hashes": cohort_hashes, "origins": len(candidate["origins"]),
        "target_count": candidate_score["target_count"], "origins_are_independent_samples": False,
    }
    for metric in ("mse", "mae"):
        cand, ref = candidate_score[metric], reference_score[metric]
        relative_percent = (cand - ref) / ref * 100.
        if not np.isfinite(relative_percent):
            raise ValueError("Relative error percentage is nonfinite")
        summary.update({f"candidate_{metric}": cand, f"reference_{metric}": ref,
                        f"{metric}_delta": cand - ref, f"{metric}_relative_percent": relative_percent})
    candidate_error, mask = _errors(candidate["prediction"], candidate["target"], candidate["mask"])
    reference_error, _ = _errors(reference["prediction"], reference["target"], reference["mask"])
    candidate_rows = _origin_rows(candidate_error, mask, candidate["origins"], "candidate")
    reference_rows = _origin_rows(reference_error, mask, reference["origins"], "reference")
    rows = []
    for cand, ref in zip(candidate_rows, reference_rows):
        rows.append({
            "origin": cand["origin"], "count": cand["count"], "channel_count": cand["channel_count"],
            **{f"candidate_{metric}": cand[metric] for metric in ("mse", "mae")},
            **{f"reference_{metric}": ref[metric] for metric in ("mse", "mae")},
            **{f"{metric}_delta": cand[metric] - ref[metric] for metric in ("mse", "mae")},
        })
    summary["origin_mse_improved_fraction"] = float(np.mean([row["mse_delta"] < 0 for row in rows]))
    summary["origin_mae_improved_fraction"] = float(np.mean([row["mae_delta"] < 0 for row in rows]))
    return summary, rows


def make_schedule(seed: int, pool_size: int, updates: int, batch: int = 8) -> list[np.ndarray]:
    """Shuffle the complete pool per pass, retaining each pass's small tail."""
    for name, value, minimum in (("seed", seed, 0), ("pool_size", pool_size, 1),
                                 ("updates", updates, 0), ("batch", batch, 1)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < updates:
        permutation = rng.permutation(pool_size)
        for start in range(0, pool_size, batch):
            schedule.append(permutation[start:start + batch].copy())
            if len(schedule) == updates:
                break
    return schedule
