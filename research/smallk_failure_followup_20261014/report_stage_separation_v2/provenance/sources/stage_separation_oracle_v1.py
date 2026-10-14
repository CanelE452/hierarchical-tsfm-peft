"""Source-only helpers for stage-separation future-oracle diagnostics.

This module intentionally has no top-level asset, checkpoint, NPZ, model, CUDA,
or runtime construction.  Callers must provide already-loaded arrays and an
already-constructed old codec object.  The oracle path is diagnostic only: it
uses future target values and must stay separate from past-only deployment.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


DEFAULT_PAST_LENGTH = 512
DEFAULT_HORIZON = 48
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 2e-6

FILL_OBSERVED = np.uint8(1)
FILL_CARRY_PAST_LAST = np.uint8(2)
FILL_CARRY_PREVIOUS_FUTURE = np.uint8(3)


class StageSeparationContractError(ValueError):
    """Raised when an in-memory source contract does not match the plan."""


class ProtectedTestPathError(StageSeparationContractError):
    """Raised before a helper would touch a protected TEST path."""


def validate_file_record_contract(record: Mapping[str, Any], *, role: str) -> dict[str, str]:
    """Validate a path/SHA record without opening the file."""

    path = _path_from_record(record)
    assert_no_protected_test_path(record, role=role)
    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise StageSeparationContractError("FILE_RECORD_SHA256_CONTRACT: " + role)
    alphabet = set("0123456789abcdefABCDEF")
    if any(char not in alphabet for char in sha256):
        raise StageSeparationContractError("FILE_RECORD_SHA256_HEX: " + role)
    return {"role": role, "path": str(path), "sha256": sha256.lower()}


@dataclass(frozen=True)
class PastOnlyDeploymentInputs:
    """Past-only deployment payload.

    Shapes:
        history: float32 [N, L, C], normalized old-preprocessed past.
        history_observed: bool [N, L, C], original past observation mask.
        raw_last: float32 [N, C], exactly history[:, -1].
        origins: int64 [N].
        mean/std: float32 [C], fixed FIT normalizer copied from canonical RAW.
    """

    history: np.ndarray
    history_observed: np.ndarray
    raw_last: np.ndarray
    origins: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class FutureOracleContext:
    """Future-oracle encoder context.

    The future part is filled causally from observed future values and the
    filled past last value.  ``encoder_context_observed`` records real
    observations only; carried values are not marked observed.
    """

    encoder_context: np.ndarray
    encoder_context_observed: np.ndarray
    future_filled: np.ndarray
    future_fill_source: np.ndarray
    complete_future_prefix_mask: np.ndarray
    raw_last: np.ndarray
    history: np.ndarray
    history_observed: np.ndarray
    target: np.ndarray
    observed: np.ndarray


def _path_from_record(path_or_record: Any) -> Path:
    if isinstance(path_or_record, Mapping):
        value = path_or_record.get("path")
    else:
        value = path_or_record
    if value is None:
        raise StageSeparationContractError("PATH_RECORD_MISSING_PATH")
    return Path(value)


def assert_no_protected_test_path(path_or_record: Any, *, role: str = "") -> None:
    """Guard protected TEST assets before a caller reads them.

    The guard is intentionally path-based and side-effect free.  It rejects
    exact path components named TEST/test and common protected TEST filenames.
    """

    path = _path_from_record(path_or_record)
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    protected_names = {"test.npz", "test.csv", "test.json", "test.pt", "test.pth"}
    if "test" in lowered_parts or lowered_name in protected_names:
        suffix = f": {role}" if role else ""
        raise ProtectedTestPathError(f"PROTECTED_TEST_ASSET{suffix}: {path}")


def validate_fixed_normalizer(mean: np.ndarray, std: np.ndarray, *, channels: int | None = None) -> None:
    mean = np.asarray(mean)
    std = np.asarray(std)
    if mean.dtype != np.float32 or std.dtype != np.float32:
        raise StageSeparationContractError("FIXED_NORMALIZER_DTYPE")
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise StageSeparationContractError("FIXED_NORMALIZER_SHAPE")
    if channels is not None and mean.shape != (channels,):
        raise StageSeparationContractError("FIXED_NORMALIZER_CHANNELS")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise StageSeparationContractError("FIXED_NORMALIZER_NONFINITE")
    if not np.all(std > 0):
        raise StageSeparationContractError("FIXED_NORMALIZER_NONPOSITIVE_STD")


def _require_array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise StageSeparationContractError("SPLIT_PAYLOAD_MISSING_KEY: " + key)
    value = np.asarray(payload[key])
    return value


def validate_past_only_payload(
    payload: Mapping[str, Any],
    *,
    split_name: str | None = None,
    history_length: int = DEFAULT_PAST_LENGTH,
    channels: int | None = None,
) -> dict[str, Any]:
    """Validate deploy-time past inputs without touching future labels/masks."""

    history = _require_array(payload, "history")
    history_observed = _require_array(payload, "history_observed")
    origins = _require_array(payload, "origins")
    raw_last = _require_array(payload, "raw_last")
    mean = _require_array(payload, "mean")
    std = _require_array(payload, "std")

    if history.dtype != np.float32 or raw_last.dtype != np.float32:
        raise StageSeparationContractError("PAST_ONLY_FLOAT_DTYPE: " + str(split_name))
    if history_observed.dtype != np.bool_:
        raise StageSeparationContractError("PAST_ONLY_MASK_DTYPE: " + str(split_name))
    if history.ndim != 3:
        raise StageSeparationContractError("PAST_ONLY_HISTORY_RANK: " + str(split_name))
    n, l, c = history.shape
    if l != history_length:
        raise StageSeparationContractError("PAST_ONLY_HISTORY_LENGTH: " + str(split_name))
    if channels is not None and c != channels:
        raise StageSeparationContractError("PAST_ONLY_CHANNELS: " + str(split_name))
    if history_observed.shape != history.shape:
        raise StageSeparationContractError("PAST_ONLY_MASK_SHAPE: " + str(split_name))
    if origins.shape != (n,):
        raise StageSeparationContractError("PAST_ONLY_ORIGINS_SHAPE: " + str(split_name))
    if raw_last.shape != (n, c):
        raise StageSeparationContractError("PAST_ONLY_RAW_LAST_SHAPE: " + str(split_name))
    validate_fixed_normalizer(mean, std, channels=c)
    if not np.array_equal(raw_last, history[:, -1]):
        raise StageSeparationContractError("PAST_ONLY_RAW_LAST_NOT_HISTORY_LAST: " + str(split_name))
    if not np.isfinite(history).all() or not np.isfinite(raw_last).all():
        raise StageSeparationContractError("PAST_ONLY_HISTORY_NONFINITE: " + str(split_name))
    return {"windows": int(n), "history_length": int(l), "channels": int(c)}


def validate_split_payload(
    payload: Mapping[str, Any],
    *,
    split_name: str | None = None,
    history_length: int = DEFAULT_PAST_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    channels: int | None = None,
    require_history_physical: bool = False,
) -> dict[str, Any]:
    """Validate an already-loaded bridge split without opening any files."""

    past = validate_past_only_payload(
        payload,
        split_name=split_name,
        history_length=history_length,
        channels=channels,
    )
    history = _require_array(payload, "history")
    history_observed = _require_array(payload, "history_observed")
    target = _require_array(payload, "target")
    observed = _require_array(payload, "observed")

    if target.dtype != np.float32:
        raise StageSeparationContractError("SPLIT_FLOAT_DTYPE: " + str(split_name))
    if observed.dtype != np.bool_:
        raise StageSeparationContractError("SPLIT_MASK_DTYPE: " + str(split_name))
    if target.ndim != 3:
        raise StageSeparationContractError("SPLIT_RANK: " + str(split_name))
    n, _, c = history.shape
    if target.shape != (n, horizon, c):
        raise StageSeparationContractError("SPLIT_SHAPE: " + str(split_name))
    if observed.shape != target.shape:
        raise StageSeparationContractError("SPLIT_MASK_SHAPE: " + str(split_name))
    if not np.isfinite(target[observed]).all():
        raise StageSeparationContractError("SPLIT_OBSERVED_TARGET_NONFINITE: " + str(split_name))

    if require_history_physical:
        history_physical = _require_array(payload, "history_physical")
        physical_past = np.asarray(payload.get("physical_past", history_physical))
        physical_mask = _require_array(payload, "physical_past_mask")
        if history_physical.dtype != np.float32 or physical_past.dtype != np.float32:
            raise StageSeparationContractError("SPLIT_PHYSICAL_DTYPE: " + str(split_name))
        if history_physical.shape != history.shape or physical_past.shape != history.shape:
            raise StageSeparationContractError("SPLIT_PHYSICAL_SHAPE: " + str(split_name))
        if physical_mask.dtype != np.bool_ or physical_mask.shape != history.shape:
            raise StageSeparationContractError("SPLIT_PHYSICAL_MASK_SHAPE: " + str(split_name))
        if not np.array_equal(physical_mask, history_observed):
            raise StageSeparationContractError("SPLIT_PHYSICAL_MASK_NOT_HISTORY_MASK: " + str(split_name))
        if not np.array_equal(history_physical, physical_past, equal_nan=True):
            raise StageSeparationContractError("SPLIT_PHYSICAL_ALIAS_MISMATCH: " + str(split_name))

    return {**past, "horizon": int(horizon)}


def validate_d2_cached_latent_payload(
    payload: Mapping[str, Any],
    *,
    split_name: str | None = None,
    history_length: int = DEFAULT_PAST_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    channels: int | None = None,
    latent_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the source schema produced by run_condition_probes D2 extraction."""

    latent_past = _require_array(payload, "latent_past")
    z_future = _require_array(payload, "z_future")
    z_last = _require_array(payload, "z_last")
    source_gpu_prediction = _require_array(payload, "source_gpu_prediction")
    original_prediction = _require_array(payload, "original_prediction")
    raw_forecast = _require_array(payload, "raw_forecast")
    target = _require_array(payload, "target")
    observed = _require_array(payload, "observed")
    origins = _require_array(payload, "origins")
    raw_last = _require_array(payload, "raw_last")
    history_observed = _require_array(payload, "history_observed")

    if latent_past.dtype != np.float32 or z_future.dtype != np.float32 or z_last.dtype != np.float32:
        raise StageSeparationContractError("D2_LATENT_DTYPE: " + str(split_name))
    if source_gpu_prediction.dtype != np.float32 or original_prediction.dtype != np.float32:
        raise StageSeparationContractError("D2_PREDICTION_DTYPE: " + str(split_name))
    if raw_forecast.dtype != np.float32 or target.dtype != np.float32 or raw_last.dtype != np.float32:
        raise StageSeparationContractError("D2_REFERENCE_DTYPE: " + str(split_name))
    if observed.dtype != np.bool_ or history_observed.dtype != np.bool_:
        raise StageSeparationContractError("D2_MASK_DTYPE: " + str(split_name))
    if latent_past.ndim != 3:
        raise StageSeparationContractError("D2_LATENT_PAST_RANK: " + str(split_name))
    n, l, k = latent_past.shape
    if l != history_length:
        raise StageSeparationContractError("D2_LATENT_PAST_LENGTH: " + str(split_name))
    if latent_dim is not None and k != latent_dim:
        raise StageSeparationContractError("D2_LATENT_DIM: " + str(split_name))
    if z_future.shape != (n, horizon, k) or z_last.shape != (n, 1, k):
        raise StageSeparationContractError("D2_LATENT_FORECAST_SHAPE: " + str(split_name))
    if target.ndim != 3:
        raise StageSeparationContractError("D2_TARGET_RANK: " + str(split_name))
    _, h, c = target.shape
    if h != horizon or (channels is not None and c != channels):
        raise StageSeparationContractError("D2_TARGET_SHAPE: " + str(split_name))
    for name, value in (
        ("source_gpu_prediction", source_gpu_prediction),
        ("original_prediction", original_prediction),
        ("raw_forecast", raw_forecast),
    ):
        if value.shape != (n, horizon, c):
            raise StageSeparationContractError("D2_" + name.upper() + "_SHAPE: " + str(split_name))
    if observed.shape != target.shape or raw_last.shape != (n, c):
        raise StageSeparationContractError("D2_REFERENCE_SHAPE: " + str(split_name))
    if history_observed.shape != (n, history_length, c) or origins.shape != (n,):
        raise StageSeparationContractError("D2_HISTORY_ORIGIN_SHAPE: " + str(split_name))
    if not np.isfinite(latent_past).all() or not np.isfinite(z_future).all() or not np.isfinite(z_last).all():
        raise StageSeparationContractError("D2_LATENT_NONFINITE: " + str(split_name))
    return {"windows": int(n), "history_length": int(l), "horizon": int(h), "channels": int(c), "latent_dim": int(k)}


def prepare_past_only_deployment_inputs(
    payload: Mapping[str, Any],
    *,
    history_length: int = DEFAULT_PAST_LENGTH,
    horizon: int = DEFAULT_HORIZON,
) -> PastOnlyDeploymentInputs:
    """Return only past-side arrays required by the deploy path."""

    _ = horizon
    validate_past_only_payload(payload, history_length=history_length)
    return PastOnlyDeploymentInputs(
        history=np.asarray(payload["history"], dtype=np.float32),
        history_observed=np.asarray(payload["history_observed"], dtype=np.bool_),
        raw_last=np.asarray(payload["raw_last"], dtype=np.float32),
        origins=np.asarray(payload["origins"]),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def causal_fill_future_from_past(
    history: np.ndarray,
    history_observed: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causally fill future values from observed future or filled past last.

    Missing future values are never backfilled from later horizons.  For each
    channel, the carry starts at ``history[:, -1, channel]`` and is updated only
    when that channel is observed at the current future step.
    """

    history = np.asarray(history)
    history_observed = np.asarray(history_observed)
    target = np.asarray(target)
    observed = np.asarray(observed)
    if history.dtype != np.float32 or target.dtype != np.float32:
        raise StageSeparationContractError("ORACLE_FLOAT_DTYPE")
    if history_observed.dtype != np.bool_ or observed.dtype != np.bool_:
        raise StageSeparationContractError("ORACLE_MASK_DTYPE")
    if history.ndim != 3 or target.ndim != 3:
        raise StageSeparationContractError("ORACLE_RANK")
    n, _, c = history.shape
    if history_observed.shape != history.shape or target.shape[0] != n or target.shape[2] != c:
        raise StageSeparationContractError("ORACLE_HISTORY_TARGET_SHAPE")
    if observed.shape != target.shape:
        raise StageSeparationContractError("ORACLE_OBSERVED_SHAPE")
    if not np.isfinite(history).all():
        raise StageSeparationContractError("ORACLE_HISTORY_NONFINITE")
    if not np.isfinite(target[observed]).all():
        raise StageSeparationContractError("ORACLE_OBSERVED_TARGET_NONFINITE")

    horizon = target.shape[1]
    future = np.empty_like(target, dtype=np.float32)
    fill_source = np.empty(target.shape, dtype=np.uint8)
    carry = history[:, -1, :].astype(np.float32, copy=True)
    seen_future = np.zeros((n, c), dtype=np.bool_)
    for step in range(horizon):
        step_observed = observed[:, step, :]
        future[:, step, :] = np.where(step_observed, target[:, step, :], carry)
        fill_source[:, step, :] = np.where(
            step_observed,
            FILL_OBSERVED,
            np.where(seen_future, FILL_CARRY_PREVIOUS_FUTURE, FILL_CARRY_PAST_LAST),
        ).astype(np.uint8, copy=False)
        carry = np.where(step_observed, target[:, step, :], carry).astype(np.float32, copy=False)
        seen_future |= step_observed

    complete_future_prefix_mask = np.logical_and.accumulate(observed.all(axis=2), axis=1)
    return future.astype(np.float32, copy=False), fill_source, complete_future_prefix_mask


def build_oracle_encoder_context(
    history: np.ndarray,
    history_observed: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
) -> FutureOracleContext:
    """Build the 512+48 diagnostic encoder input from in-memory arrays."""

    future, fill_source, complete_prefix = causal_fill_future_from_past(
        history, history_observed, target, observed
    )
    context = np.concatenate([history, future], axis=1).astype(np.float32, copy=False)
    context_observed = np.concatenate([history_observed, observed], axis=1).astype(np.bool_, copy=False)
    return FutureOracleContext(
        encoder_context=context,
        encoder_context_observed=context_observed,
        future_filled=future,
        future_fill_source=fill_source,
        complete_future_prefix_mask=complete_prefix,
        raw_last=np.asarray(history[:, -1], dtype=np.float32),
        history=np.asarray(history, dtype=np.float32),
        history_observed=np.asarray(history_observed, dtype=np.bool_),
        target=np.asarray(target, dtype=np.float32),
        observed=np.asarray(observed, dtype=np.bool_),
    )


def prepare_future_oracle_inputs(payload: Mapping[str, Any]) -> FutureOracleContext:
    """Return future-oracle arrays; callers must keep this diagnostic-only."""

    return build_oracle_encoder_context(
        np.asarray(payload["history"], dtype=np.float32),
        np.asarray(payload["history_observed"], dtype=np.bool_),
        np.asarray(payload["target"], dtype=np.float32),
        np.asarray(payload["observed"], dtype=np.bool_),
    )


def complete_future_teacher_mask(complete_future_prefix_mask: np.ndarray, latent_dim: int) -> np.ndarray:
    """Expand complete-future-prefix coverage to a K-wide teacher mask."""

    mask = np.asarray(complete_future_prefix_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise StageSeparationContractError("TEACHER_PREFIX_MASK_SHAPE")
    if latent_dim <= 0:
        raise StageSeparationContractError("TEACHER_LATENT_DIM")
    return np.repeat(mask[:, :, None], int(latent_dim), axis=2)


def teacher_coverage_by_horizon(complete_future_prefix_mask: np.ndarray) -> list[int]:
    mask = np.asarray(complete_future_prefix_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise StageSeparationContractError("TEACHER_PREFIX_MASK_SHAPE")
    return [int(value) for value in mask.sum(axis=0)]


def _batch_slices(length: int, batch_size: int) -> Iterable[slice]:
    if batch_size <= 0:
        raise StageSeparationContractError("BATCH_SIZE_NONPOSITIVE")
    for begin in range(0, length, batch_size):
        yield slice(begin, min(begin + batch_size, length))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _no_grad_context(no_grad: Callable[[], Any] | None) -> Any:
    return no_grad() if no_grad is not None else nullcontext()


def extract_past_only_deployment_latents(
    codec: Any,
    history: np.ndarray,
    *,
    tensor: Callable[[np.ndarray], Any],
    no_grad: Callable[[], Any] | None,
    check: Callable[[], None],
    synchronize: Callable[[], None] | None = None,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Encode past-only deploy inputs with a caller-supplied codec."""

    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3:
        raise StageSeparationContractError("PAST_ONLY_HISTORY_RANK")
    chunks: list[np.ndarray] = []
    with _no_grad_context(no_grad):
        for item in _batch_slices(len(history), batch_size):
            check()
            encoded = codec.encode(tensor(history[item]))
            chunks.append(_to_numpy(encoded).astype(np.float32, copy=False))
            check()
    if synchronize is not None:
        synchronize()
    latent = np.concatenate(chunks, axis=0) if chunks else np.empty((0, history.shape[1], 0), dtype=np.float32)
    return {
        "latent_past": latent.astype(np.float32, copy=False),
        "z_last": latent[:, -1:].astype(np.float32, copy=False),
    }


def extract_true_future_oracle_latents(
    codec: Any,
    history: np.ndarray,
    history_observed: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    *,
    tensor: Callable[[np.ndarray], Any],
    no_grad: Callable[[], Any] | None,
    check: Callable[[], None],
    synchronize: Callable[[], None] | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Encode the diagnostic 560 context with a caller-supplied codec."""

    context = build_oracle_encoder_context(history, history_observed, target, observed)
    past_length = context.history.shape[1]
    chunks: list[np.ndarray] = []
    with _no_grad_context(no_grad):
        for item in _batch_slices(len(context.encoder_context), batch_size):
            check()
            encoded = codec.encode(tensor(context.encoder_context[item]))
            chunks.append(_to_numpy(encoded).astype(np.float32, copy=False))
            check()
    if synchronize is not None:
        synchronize()
    latent_context = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, context.encoder_context.shape[1], 0), dtype=np.float32)
    )
    return {
        "latent_context": latent_context.astype(np.float32, copy=False),
        "oracle_past_prefix_latent": latent_context[:, :past_length].astype(np.float32, copy=False),
        "oracle_future_latent": latent_context[:, past_length:].astype(np.float32, copy=False),
        "oracle_z_last": latent_context[:, past_length - 1 : past_length].astype(np.float32, copy=False),
        "context": context,
    }


def _select_informative_observed_future_perturb(observed: np.ndarray) -> dict[str, Any] | None:
    """Select one observed future cell for a causal prefix perturbation check."""

    if observed.shape[0] == 0 or observed.shape[1] == 0:
        return None
    observed_any_channel = observed.any(axis=2)
    coordinates = np.argwhere(observed_any_channel)
    if len(coordinates) == 0:
        return None
    latest_step = int(coordinates[:, 1].max())
    sample_index = int(coordinates[coordinates[:, 1] == latest_step, 0].min())
    channels = np.where(observed[sample_index, latest_step])[0].astype(np.int64)
    return {
        "sample_index": sample_index,
        "perturb_step": latest_step,
        "observed_channels": [int(value) for value in channels],
        "selected_step_reason": "latest_future_step_with_observed_channel",
        "selected_first_sample": bool(sample_index == 0),
    }


def extract_normalized_oracle(
    codec: Any,
    history_normalized: np.ndarray,
    target_normalized: np.ndarray,
    mask_bool: np.ndarray,
    *,
    history_observed: np.ndarray | None = None,
    tensor: Callable[[np.ndarray], Any],
    no_grad: Callable[[], Any] | None,
    check: Callable[[], None],
    sync: Callable[[], None] | None = None,
    synchronize: Callable[[], None] | None = None,
    batch_size: int = 8,
    perturb_delta: float = 1.0,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """Driver-facing normalized oracle extraction.

    ``history_normalized`` and ``target_normalized`` are already in the fixed
    FIT-normalized bridge scale.  This helper never re-fits or re-normalizes.
    If ``history_observed`` is omitted, the function records an all-observed
    past mask for the encoder-context mask only; future filling still depends
    solely on ``mask_bool`` and the filled normalized past last value.
    """

    history = np.asarray(history_normalized, dtype=np.float32)
    target = np.asarray(target_normalized, dtype=np.float32)
    observed = np.asarray(mask_bool, dtype=np.bool_)
    if history_observed is None:
        past_mask = np.ones(history.shape, dtype=np.bool_)
        past_mask_source = "omitted_assumed_all_observed_for_context_mask_only"
    else:
        past_mask = np.asarray(history_observed, dtype=np.bool_)
        past_mask_source = "provided"

    past = extract_past_only_deployment_latents(
        codec,
        history,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        synchronize=sync if sync is not None else synchronize,
        batch_size=batch_size,
    )
    diagnostic = extract_true_future_oracle_latents(
        codec,
        history,
        past_mask,
        target,
        observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        synchronize=sync if sync is not None else synchronize,
        batch_size=batch_size,
    )
    context: FutureOracleContext = diagnostic["context"]
    prefix_check = verify_past_prefix_parity(
        past["latent_past"],
        diagnostic["oracle_past_prefix_latent"],
        atol=atol,
        rtol=rtol,
    )

    perturb_selection = _select_informative_observed_future_perturb(observed)
    if perturb_selection is None:
        first_sample_invariance = {
            "status": "NOT_TESTED_NO_OBSERVED_FUTURE",
            "pass": None,
        }
    else:
        first_sample_invariance = verify_future_perturb_prefix_invariance(
            codec,
            history,
            past_mask,
            target,
            observed,
            perturb_step=perturb_selection["perturb_step"],
            perturb_delta=perturb_delta,
            sample_index=perturb_selection["sample_index"],
            observed_channels_only=True,
            tensor=tensor,
            no_grad=no_grad,
            check=check,
            synchronize=sync if sync is not None else synchronize,
            batch_size=batch_size,
            atol=atol,
            rtol=rtol,
        )
        first_sample_invariance.update(perturb_selection)

    return {
        "oracle_z": diagnostic["oracle_future_latent"],
        "latent_context": diagnostic["latent_context"],
        "current_past_latent": past["latent_past"],
        "current_z_last": past["z_last"],
        "oracle_past_prefix_latent": diagnostic["oracle_past_prefix_latent"],
        "oracle_z_last": diagnostic["oracle_z_last"],
        "prefix_check": prefix_check,
        "complete_future_prefix_mask": context.complete_future_prefix_mask,
        "complete_future_prefix_counts": teacher_coverage_by_horizon(context.complete_future_prefix_mask),
        "future_filled": context.future_filled,
        "future_fill_source": context.future_fill_source,
        "first_sample_forward_invariance": first_sample_invariance,
        "normalization_contract": {
            "input_scale": "already_normalized_bridge_or_D2_cache",
            "refit_or_renormalized_here": False,
            "history_observed_source": past_mask_source,
        },
    }


def compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {
            "pass": False,
            "reason": "SHAPE_MISMATCH",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "atol": float(atol),
            "rtol": float(rtol),
        }
    difference = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    rmse = float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0
    return {
        "pass": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        "max_abs": max_abs,
        "rmse": rmse,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def verify_past_prefix_parity(
    past_latent: np.ndarray,
    oracle_past_prefix_latent: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """Verify E(past) equals the past prefix of E(past + causal future)."""

    return compare_arrays(past_latent, oracle_past_prefix_latent, atol=atol, rtol=rtol)


def verify_future_perturb_prefix_invariance(
    codec: Any,
    history: np.ndarray,
    history_observed: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    *,
    perturb_step: int,
    perturb_delta: float,
    sample_index: int | None = None,
    observed_channels_only: bool = False,
    tensor: Callable[[np.ndarray], Any],
    no_grad: Callable[[], Any] | None,
    check: Callable[[], None],
    synchronize: Callable[[], None] | None = None,
    batch_size: int = 8,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """Perturb one future step and check earlier latent prefix invariance.

    A perturbation at future horizon h may affect encoder outputs at context
    positions >= L+h only.  Missing entries at h are ignored by the causal fill.
    ``pass`` is true only when the filled encoder context actually changed and
    the earlier prefix stayed invariant.
    """

    base = build_oracle_encoder_context(history, history_observed, target, observed)
    if perturb_step < 0 or perturb_step >= base.target.shape[1]:
        raise StageSeparationContractError("PERTURB_STEP_OUT_OF_RANGE")
    changed_target = base.target.copy()
    perturb_mask = np.zeros((base.target.shape[0], base.target.shape[2]), dtype=np.bool_)
    if sample_index is None:
        perturb_mask[:, :] = base.observed[:, perturb_step, :] if observed_channels_only else True
    else:
        if sample_index < 0 or sample_index >= base.target.shape[0]:
            raise StageSeparationContractError("PERTURB_SAMPLE_OUT_OF_RANGE")
        perturb_mask[sample_index, :] = (
            base.observed[sample_index, perturb_step, :] if observed_channels_only else True
        )
    changed_target[:, perturb_step, :] = np.where(
        perturb_mask,
        changed_target[:, perturb_step, :] + np.float32(perturb_delta),
        changed_target[:, perturb_step, :],
    )
    changed = build_oracle_encoder_context(base.history, base.history_observed, changed_target, base.observed)

    base_latent = extract_true_future_oracle_latents(
        codec,
        base.history,
        base.history_observed,
        base.target,
        base.observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        synchronize=synchronize,
        batch_size=batch_size,
    )["latent_context"]
    changed_latent = extract_true_future_oracle_latents(
        codec,
        changed.history,
        changed.history_observed,
        changed.target,
        changed.observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        synchronize=synchronize,
        batch_size=batch_size,
    )["latent_context"]
    prefix_length = base.history.shape[1] + perturb_step
    context_prefix = compare_arrays(
        base.encoder_context[:, :prefix_length],
        changed.encoder_context[:, :prefix_length],
        atol=0.0,
        rtol=0.0,
    )
    latent_prefix = compare_arrays(
        base_latent[:, :prefix_length],
        changed_latent[:, :prefix_length],
        atol=atol,
        rtol=rtol,
    )
    filled_context_all = compare_arrays(
        base.encoder_context,
        changed.encoder_context,
        atol=0.0,
        rtol=0.0,
    )
    suffix_difference = compare_arrays(
        base_latent[:, prefix_length:],
        changed_latent[:, prefix_length:],
        atol=atol,
        rtol=rtol,
    )
    context_changed = not filled_context_all["pass"]
    if perturb_mask.any() and context_changed:
        status = "TESTED_CONTEXT_CHANGED_PREFIX_INVARIANCE"
        overall_pass: bool | None = bool(context_prefix["pass"] and latent_prefix["pass"])
    elif not perturb_mask.any():
        status = "NOT_TESTED_NO_OBSERVED_FUTURE"
        overall_pass = None
    else:
        status = "NOT_TESTED_PERTURBATION_DID_NOT_CHANGE_FILLED_CONTEXT"
        overall_pass = None
    return {
        "status": status,
        "pass": overall_pass,
        "perturb_step": int(perturb_step),
        "sample_index": None if sample_index is None else int(sample_index),
        "observed_channels_only": bool(observed_channels_only),
        "perturbed_observed_cells": int(perturb_mask.sum()),
        "prefix_length": int(prefix_length),
        "context_changed": bool(context_changed),
        "context_prefix": context_prefix,
        "latent_prefix": latent_prefix,
        "filled_context_all": filled_context_all,
        "latent_suffix_difference": suffix_difference,
        "scope": "one selected perturbation; suffix change is reported, not required as a universal encoder proof",
    }


def anchor_decode_prediction(raw_last: np.ndarray, decoded_future: np.ndarray, decoded_last: np.ndarray) -> np.ndarray:
    """Reproduce old anchor arithmetic order: (x_last + D(future)) - D(last)."""

    raw_last = np.asarray(raw_last, dtype=np.float32)
    decoded_future = np.asarray(decoded_future, dtype=np.float32)
    decoded_last = np.asarray(decoded_last, dtype=np.float32)
    if raw_last.ndim != 2 or decoded_future.ndim != 3 or decoded_last.ndim != 3:
        raise StageSeparationContractError("ANCHOR_DECODE_RANK")
    if decoded_last.shape[1] != 1 or decoded_future.shape[0] != raw_last.shape[0]:
        raise StageSeparationContractError("ANCHOR_DECODE_SHAPE")
    return ((raw_last[:, None, :] + decoded_future) - decoded_last).astype(np.float32, copy=False)


def source_only_contract() -> dict[str, Any]:
    """Describe the API without touching assets."""

    return {
        "status": "SOURCE_ONLY_NOT_EXECUTED",
        "domain": "old-condition multivariate time-series forecast stage separation",
        "downstream_decision": (
            "diagnose representation, latent predictor, and decoder contributions without "
            "mixing future-oracle diagnostics into past-only deployment ranking"
        ),
        "leakage_controls": [
            "past-only deployment functions do not accept future target or observed masks",
            "future-oracle functions require explicit target and observed inputs",
            "missing future values are causal carry-forward from filled past last or previous observed future",
            "complete-future-prefix mask is explicit and K-wide teacher mask uses that prefix condition",
            "fixed FIT mean/std are validated, never refit here",
            "protected TEST path guard is side-effect free and must run before asset reads",
        ],
        "input_shapes": {
            "history": "float32[N,512,C]",
            "history_observed": "bool[N,512,C]",
            "target": "float32[N,48,C]",
            "observed": "bool[N,48,C]",
            "encoder_context": "float32[N,560,C]",
            "complete_future_prefix_mask": "bool[N,48]",
        },
        "main_functions": [
            "validate_file_record_contract(record, role)",
            "validate_past_only_payload(payload)",
            "validate_split_payload(payload)",
            "validate_d2_cached_latent_payload(payload)",
            "prepare_past_only_deployment_inputs(payload)",
            "build_oracle_encoder_context(history, history_observed, target, observed)",
            "extract_normalized_oracle(codec, history_normalized, target_normalized, mask_bool, history_observed, tensor, no_grad, check, sync)",
            "extract_past_only_deployment_latents(codec, history, tensor, no_grad, check, synchronize)",
            "extract_true_future_oracle_latents(codec, history, history_observed, target, observed, tensor, no_grad, check, synchronize)",
            "verify_past_prefix_parity(past_latent, oracle_past_prefix_latent)",
            "verify_future_perturb_prefix_invariance(codec, history, history_observed, target, observed, perturb_step, perturb_delta, tensor, no_grad, check)",
            "complete_future_teacher_mask(complete_future_prefix_mask, latent_dim)",
        ],
        "source_traced_schemas": {
            "bridge_split_npz": [
                "history",
                "history_observed",
                "target",
                "observed",
                "origins",
                "raw_last",
                "mean",
                "std",
                "optional history_physical/physical_past/physical_past_mask",
            ],
            "d2_cached_latents": [
                "latent_past",
                "z_future",
                "z_last",
                "source_gpu_prediction",
                "original_prediction",
                "raw_forecast",
                "target",
                "observed",
                "origins",
                "raw_last",
                "history_observed",
            ],
        },
        "execution_policy": "No asset/CP/NPZ/model/CUDA/runtime construction on import or describe.",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print source-only contract JSON")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="intentionally unsupported here; actual runners must pass arrays/codecs explicitly",
    )
    args = parser.parse_args(argv)
    if args.execute:
        raise SystemExit("STAGE_SEPARATION_ORACLE_SOURCE_ONLY_NO_DIRECT_EXECUTE")
    print(json.dumps(source_only_contract(), ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
