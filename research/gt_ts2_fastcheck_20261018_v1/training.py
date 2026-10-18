"""Fixed-schedule forecast-only GT adaptation with channel-macro gradients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .common import file_sha, seal_json, state_sha


def _training_arrays(x, targets, masks, features=None):
    x, targets, masks = map(torch.as_tensor, (x, targets, masks))
    if (x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 4 or x.shape[1] % 4
            or x.shape[-1] != 21 or not x.is_floating_point() or not torch.isfinite(x).all()):
        raise ValueError("TRAIN histories must be finite [N,L,21] with complete four-tick blocks")
    if (targets.shape != (len(x), 48, 21) or masks.shape != targets.shape
            or not targets.is_floating_point() or masks.dtype != torch.bool):
        raise ValueError("TRAIN targets/masks must be floating/bool [N,48,21]")
    if not masks.any() or not torch.isfinite(targets[masks]).all():
        raise ValueError("Observed TRAIN targets must be finite and nonempty")
    if features is not None:
        features = torch.as_tensor(features)
        if (features.shape != (len(x), x.shape[1] // 4, 320)
                or not features.is_floating_point() or not torch.isfinite(features).all()):
            raise ValueError("Fixed TS2 cache must be finite pooled [N,L/4,320] features")
    return x, targets, masks, features


def _schedule_arrays(schedule, count, masks):
    result = []
    digest = hashlib.sha256()
    for indices in schedule:
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not 1 <= len(indices) <= 8
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= count).any()):
            raise ValueError("Each fixed effective batch must contain 1..8 valid integer TRAIN indices")
        indices = np.asarray(indices, dtype="<i8").copy()
        index = torch.as_tensor(indices, device=masks.device)
        if not masks.index_select(0, index).any():
            raise ValueError("Every effective batch needs observed forecast targets")
        digest.update(np.asarray([len(indices)], dtype="<i8").tobytes())
        digest.update(indices.tobytes())
        result.append(indices)
    if not result:
        raise ValueError("A nonempty, preselected fixed update schedule is required")
    return result, digest.hexdigest()


def _batch(values, indices, device, *, mask=False):
    if values is None:
        return None
    index = torch.as_tensor(indices, device=values.device)
    return values.index_select(0, index).to(device=device, dtype=torch.bool if mask else torch.float32).detach()


def _gradient_norm(parameters):
    squared = sum(float(parameter.grad.detach().double().square().sum())
                  for parameter in parameters if parameter.grad is not None)
    return squared ** .5


def _parameter_deltas(groups, initial):
    result = {}
    for name, parameters in groups.items():
        differences = [parameter.detach().cpu().double() - before.double()
                       for parameter, before in zip(parameters, initial[name])]
        result[name] = {
            "l2": sum(float(value.square().sum()) for value in differences) ** .5,
            "max_abs": max((float(value.abs().max()) for value in differences), default=0.),
            "changed_tensors": sum(bool(torch.count_nonzero(value)) for value in differences),
            "changed_elements": sum(int(torch.count_nonzero(value)) for value in differences),
        }
    return result


def _score_totals(prediction, target, mask):
    prediction = prediction.detach().cpu().numpy().astype(np.float64)
    target = target.detach().cpu().numpy().astype(np.float64)
    mask = mask.detach().cpu().numpy()
    error = np.where(mask, prediction, 0.) - np.where(mask, target, 0.)
    return (error * error).sum((0, 1)), np.abs(error).sum((0, 1)), mask.sum((0, 1))


def _macro_scores(totals):
    sse, sae, count = totals
    active = count > 0
    if not active.any():
        raise ValueError("Fixed diagnostic has no observed values")
    return float((sse[active] / count[active]).mean()), float((sae[active] / count[active]).mean())


def _diagnose(route, diagnostics, update, device, microbatch):
    mask = diagnostics.get("mask", diagnostics.get("masks"))
    if mask is None:
        raise ValueError("Fixed diagnostics require target masks")
    x, targets, masks, features = _training_arrays(
        diagnostics["x"], diagnostics["targets"], mask, diagnostics.get("features"),
    )
    input_mask = torch.as_tensor(diagnostics.get("input_mask", torch.ones_like(x, dtype=torch.bool)))
    if input_mask.shape != x.shape or input_mask.dtype != torch.bool:
        raise ValueError("Diagnostic input_mask must be boolean [N,L,21]")
    prefix = int(route.gt.reconstruction_zero_startup_steps)
    forecast_totals = [np.zeros(21, np.float64) for _ in range(3)]
    reconstruction_totals = [np.zeros(21, np.float64) for _ in range(3)]
    was_training = route.training
    route.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(x), microbatch):
                indices = np.arange(start, min(start + microbatch, len(x)))
                history = _batch(x, indices, device)
                future = _batch(targets, indices, device)
                observed = _batch(masks, indices, device, mask=True)
                cached = _batch(features, indices, device)
                prediction = route.forecast(history, cached)
                reconstruction = route.gt.decode(route.gt.encode(history))
                reconstruction_mask = _batch(input_mask, indices, device, mask=True)
                for total, value in zip(forecast_totals, _score_totals(prediction, future, observed)):
                    total += value
                for total, value in zip(reconstruction_totals, _score_totals(
                        reconstruction[:, prefix:], history[:, prefix:], reconstruction_mask[:, prefix:])):
                    total += value
    finally:
        route.train(was_training)
    forecast_mse, forecast_mae = _macro_scores(forecast_totals)
    reconstruction_mse, reconstruction_mae = _macro_scores(reconstruction_totals)
    return {
        "update": update, "origin_count": len(x), "forecast_mse": forecast_mse,
        "forecast_mae": forecast_mae, "reconstruction_mse": reconstruction_mse,
        "reconstruction_mae": reconstruction_mae, "reconstruction_excluded_prefix": prefix,
        "forecast_observed_counts": forecast_totals[2].astype(np.int64).tolist(),
        "reconstruction_observed_counts": reconstruction_totals[2].astype(np.int64).tolist(),
        "selection_uses_diagnostic": False,
    }


def train_arm(route, train_x, targets, masks, schedule, *, feature_cache=None,
              diagnostics=None, output_dir, run_id, seed, callback=None, microbatch=1):
    """Use the caller's exact schedule; cache only fixed, pooled TS2 features."""
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or not 1 <= microbatch <= 8:
        raise ValueError("microbatch must be an integer from 1 to 8")
    x, targets, masks, features = _training_arrays(train_x, targets, masks, feature_cache)
    if features is not None and route.W is None:
        raise ValueError("TS2 features require a registered fusion bridge")
    if route.W is not None and features is None:
        raise ValueError("A fusion training arm requires its fixed pooled TS2 feature cache")
    schedule, schedule_sha = _schedule_arrays(schedule, len(x), masks)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preserve the existing attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    route.gt.requires_grad_(True)
    if route.gt.product.graph.requires_grad:
        raise ValueError("The spatial graph must remain fixed")
    graph = route.gt.product.graph.detach().cpu().clone()
    foundation_model = getattr(route.foundation, "model", route.foundation)
    frozen_parameters = []
    foundation_hash = None
    if isinstance(foundation_model, nn.Module):
        foundation_model.eval().requires_grad_(False)
        frozen_parameters = list(foundation_model.parameters())
        if any(parameter.grad is not None for parameter in frozen_parameters):
            raise ValueError("F already has parameter gradients")
        foundation_hash = state_sha(foundation_model)
    groups = {
        "encoder": list(route.gt.encoder.parameters()),
        "decoder": list(route.gt.decoder.parameters()),
        "product": list(route.gt.product.parameters()),
        "W": list(route.W.parameters()) if route.W is not None else [],
    }
    parameters = [parameter for values in groups.values() for parameter in values]
    device = parameters[0].device
    initial = {name: [parameter.detach().cpu().clone() for parameter in values]
               for name, values in groups.items()}
    optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=0.)
    exposure = np.zeros(len(x), dtype=np.int64)
    observed_exposure = 0
    curves, diagnostic_rows, checkpoints = [], [], []
    gradient_evidence = {
        name: {"updates_with_nonzero_gradient": 0, "updates_finite": 0, "maximum_norm": 0.}
        for name in groups
    }
    clip_count = 0
    started = time.perf_counter()

    def save_checkpoint(update):
        path = output_dir / f"checkpoint_{update:06d}.pt"
        checkpoint = {
            "run_id": run_id, "seed": int(seed), "update": update,
            "adapter_state": route.adapter_state(), "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "schedule_sha256": schedule_sha, "fixed_updates": len(schedule),
            "seen_origin_counts": exposure.tolist(), "seen_origin_count": int(exposure.sum()),
            "unique_seen_origins": int(np.count_nonzero(exposure)),
            "observed_target_exposure": observed_exposure, "F_state_sha256": foundation_hash,
            "diagnostics": diagnostic_rows.copy(), "optimizer": "Adam",
            "learning_rate": 1e-4, "weight_decay": 0., "gradient_norm_clip": 1.,
        }
        with path.open("xb") as stream:
            torch.save(checkpoint, stream)
        checkpoints.append({"update": update, "path": path, "sha256": file_sha(path)})

    route.train()
    if diagnostics is not None:
        diagnostic_rows.append(_diagnose(route, diagnostics, 0, device, microbatch))
    save_checkpoint(0)
    if callback is not None:
        callback(0, {"run_id": run_id, "fixed_updates": len(schedule), "checkpoint_path": checkpoints[-1]["path"]})
    with (output_dir / "TRAINING_CURVES.jsonl").open("x", encoding="utf-8") as curve_stream:
        for update, indices in enumerate(schedule, 1):
            update_started = time.perf_counter()
            effective_masks = _batch(masks, indices, device, mask=True)
            counts = effective_masks.sum((0, 1))
            active = counts > 0
            active_count = int(active.sum())
            denominator = counts.clamp_min(1).to(torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for start in range(0, len(indices), microbatch):
                selected = indices[start:start + microbatch]
                history = _batch(x, selected, device)
                target = _batch(targets, selected, device)
                observed = effective_masks[start:start + microbatch]
                cached = _batch(features, selected, device)
                prediction = route.forecast(history, cached)
                error = torch.where(observed, prediction, 0.) - torch.where(observed, target, 0.)
                loss = (error.square().sum((0, 1)) / denominator).sum() / active_count
                if not torch.isfinite(loss):
                    raise ValueError("Forecast adaptation loss became non-finite")
                loss.backward()
                loss_value += float(loss.detach())
            gradient_norms = {name: _gradient_norm(values) for name, values in groups.items()}
            preclip = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            if any(parameter.grad is not None for parameter in frozen_parameters):
                raise RuntimeError("F parameter gradients appeared during adapter training")
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("An adapted parameter became non-finite")
            clip_count += int(preclip > 1.)
            np.add.at(exposure, indices, 1)
            observed_exposure += int(effective_masks.sum())
            row = {
                "run_id": run_id, "seed": int(seed), "update": update,
                "effective_batch": len(indices), "microbatch": microbatch,
                "train_loss": loss_value, "preclip_norm": preclip,
                "clipped": preclip > 1., "clipping_count": clip_count,
                "clipping_fraction": clip_count / update, "finite": True,
                "observed_channel_counts": counts.detach().cpu().tolist(),
                "active_channels": active_count, "seen_origin_count": int(exposure.sum()),
                "unique_seen_origins": int(np.count_nonzero(exposure)),
                "observed_target_exposure": observed_exposure,
                "update_wall_seconds": time.perf_counter() - update_started,
                **{f"gradient_norm_{name}": norm for name, norm in gradient_norms.items()},
            }
            for name, norm in gradient_norms.items():
                gradient_evidence[name]["updates_with_nonzero_gradient"] += int(norm > 0)
                gradient_evidence[name]["updates_finite"] += int(np.isfinite(norm))
                gradient_evidence[name]["maximum_norm"] = max(gradient_evidence[name]["maximum_norm"], norm)
            curves.append(row)
            curve_stream.write(json.dumps(row, allow_nan=False) + "\n")
            if update in (100, 300) and diagnostics is not None:
                diagnostic_rows.append(_diagnose(route, diagnostics, update, device, microbatch))
            if update in (100, 300, len(schedule)):
                save_checkpoint(update)
            if update % 10 == 0 or update == len(schedule):
                curve_stream.flush()
                if callback is not None:
                    callback(update, {**row, "fixed_updates": len(schedule),
                                      "elapsed_seconds": time.perf_counter() - started})
    unchanged = foundation_hash is None or state_sha(foundation_model) == foundation_hash
    gradients_absent = all(parameter.grad is None for parameter in frozen_parameters)
    graph_unchanged = torch.equal(route.gt.product.graph.detach().cpu(), graph)
    if not unchanged or not gradients_absent or not graph_unchanged:
        raise RuntimeError("Frozen foundation/spatial graph invariant failed")
    deltas = _parameter_deltas(groups, initial)
    result = {
        "run_id": run_id, "seed": int(seed), "updates": len(schedule),
        "schedule_sha256": schedule_sha, "seen_origin_counts": exposure.tolist(),
        "seen_origin_count": int(exposure.sum()), "unique_seen_origins": int(np.count_nonzero(exposure)),
        "observed_target_exposure": observed_exposure,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "curves": curves, "diagnostics": diagnostic_rows, "checkpoints": checkpoints,
        "checkpoint_paths": {str(row["update"]): row["path"] for row in checkpoints},
        "parameter_deltas": deltas, "final_state_delta": deltas,
        "group_gradient_evidence": gradient_evidence, "F_state_sha256": foundation_hash,
        "F_state_unchanged": unchanged, "F_unchanged": unchanged,
        "F_parameter_gradients_absent": gradients_absent, "graph_unchanged": graph_unchanged,
        "clipping_count": clip_count, "clipping_fraction": clip_count / len(schedule),
        "wall_seconds": time.perf_counter() - started, "selection_uses_VAL": False,
    }
    seal_json(output_dir / "TRAINING_SUMMARY.json", result, allowed_root=output_dir)
    return result
