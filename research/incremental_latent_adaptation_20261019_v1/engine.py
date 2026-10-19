"""M_cal-only residual fitting with exact checkpoint continuation.

Models return normalized corrections; this engine adds the fixed G forecast.
No data loader, foundation loader, V selector or protected-data access lives here.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from research.informs_broad_validation_20261015_v1.training import (
    PermutationStream, checkpoint_bundle, fingerprint, macro_loss, restore_bundle,
)
from research.raw_context_multiscale_20261018_v1.calibration import metric_statistics


RECIPE = dict(lr=.001, weight_decay=0., gradient_clip=1., effective_batch=16, microbatch=4)


class _BudgetReached(RuntimeError):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.time() >= float(deadline):
        raise _BudgetReached("Registered training deadline reached")


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=digest)


def _state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
    os.replace(temporary, path)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def _prepare(data):
    arrays = {key: np.asarray(data[key]) for key in ("x", "g", "y", "mask", "origins")}
    x, g, y, mask, origins = (arrays[key] for key in ("x", "g", "y", "mask", "origins"))
    if (x.ndim != 3 or g.ndim != 3 or g.shape != y.shape or g.shape != mask.shape
            or len(x) != len(g) or x.shape[-1] != g.shape[-1]):
        raise ValueError("Aligned M_cal x/g/y/mask arrays are required")
    if mask.dtype != np.bool_ or origins.shape != (len(x),) or origins.dtype.kind not in "iu":
        raise ValueError("Original bool mask and one integer origin per window are required")
    if len(origins) == 0 or np.any(np.diff(origins) <= 0):
        raise ValueError("Strictly ordered nonempty origins are required")
    if not np.isfinite(x).all() or not np.isfinite(g).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Finite inputs/G and observed targets are required")
    arrays["y"] = np.where(mask, y, 0.).astype(np.float32)
    if "phi" in data:
        arrays["phi"] = np.asarray(data["phi"])
        if len(arrays["phi"]) != len(x) or not np.isfinite(arrays["phi"]).all():
            raise ValueError("Cached readout features must align with every M_cal origin")
    tensors = {key: torch.as_tensor(value, dtype=torch.bool if key == "mask" else torch.float32)
               for key, value in arrays.items() if key != "origins"}
    return arrays, tensors


def _indices(indices, n):
    values = np.asarray(indices)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
            or values.min() < 0 or values.max() >= n or np.any(np.diff(values) <= 0)):
        raise ValueError("Nonempty ordered unique fitting/selection indices are required")
    return values.astype(np.int64, copy=False)


def _correction(model, tensors, rows, foundation, device):
    if "phi" in tensors:
        return model.from_features(tensors["phi"][rows].to(device))
    return model(tensors["x"][rows].to(device), foundation)


def predict(model, data, indices, foundation=None, *, device=None, deadline=None, batch_size=4):
    """Predict G plus correction without passing targets or masks to the model."""
    arrays, tensors = _prepare(data)
    rows = _indices(indices, len(arrays["x"]))
    device = str(device or next(model.parameters()).device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            _check_deadline(deadline)
            chosen = rows[start:start + batch_size]
            correction = _correction(model, tensors, chosen, foundation, device)
            prediction = tensors["g"][chosen].to(device) + correction
            if tuple(prediction.shape) != tuple(tensors["y"][chosen].shape):
                raise ValueError("Residual output axes differ from original-channel target axes")
            predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions)


def _evaluate(model, arrays, tensors, indices, foundation, device, deadline):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(indices), RECIPE["microbatch"]):
            _check_deadline(deadline)
            rows = indices[start:start + RECIPE["microbatch"]]
            correction = _correction(model, tensors, rows, foundation, device)
            prediction = tensors["g"][rows].to(device) + correction
            if tuple(prediction.shape) != tuple(tensors["y"][rows].shape):
                raise ValueError("Residual output axes differ from original-channel target axes")
            parts.append(prediction.detach().cpu().numpy())
    return metric_statistics(np.concatenate(parts), arrays["y"][indices], arrays["mask"][indices])


def _contract(model, arrays, fit, select, seed, contract, phase):
    data_identity = {key: _array_sha(value) for key, value in arrays.items()}
    research = Path(__file__).resolve().parents[1]
    model_sources = {}
    for index, module in enumerate(model.modules()):
        source = inspect.getsourcefile(type(module))
        if source is not None:
            path = Path(source).resolve()
            if path.is_file() and (index == 0 or path.is_relative_to(research)):
                model_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha=fingerprint(dict(recipe=RECIPE, seed=int(seed), fit=fit.tolist(),
                                    select=None if select is None else select.tolist(), phase=phase)),
        data_sha=fingerprint(data_identity), model_sha=_state_sha(model),
        scaler_sha=(contract or {}).get("scaler_sha", "EXTERNAL_PARENT_CONTRACT"),
        external_contract=contract or {}, data_identity=data_identity, model_sources=model_sources,
        correction_class=type(model).__module__ + "." + type(model).__name__)


def _endpoint_name(epoch):
    return "checkpoint_e" + format(float(epoch), ".8g").replace(".", "p") + ".pt"


def _run(model, data, fit, select, seed, foundation, directory, endpoints, deadline, contract, device, phase):
    arrays, tensors = _prepare(data)
    fit = _indices(fit, len(arrays["x"]))
    select = None if select is None else _indices(select, len(arrays["x"]))
    if select is not None:
        if (np.intersect1d(fit, select).size
                or arrays["origins"][fit].max() + arrays["g"].shape[1] > arrays["origins"][select].min()):
            raise ValueError("M_fit/M_select targets must be chronological and disjoint")
    endpoints = tuple(float(value) for value in endpoints)
    if not endpoints or endpoints[0] != 0 or endpoints != tuple(sorted(set(endpoints))) or min(endpoints) < 0:
        raise ValueError("Ordered distinct endpoints starting at zero are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = str(device or next(model.parameters()).device)
    model.to(device)
    active_contract = _contract(model, arrays, fit, select, seed, contract, phase)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                 lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    sampler = PermutationStream(fit, seed)
    updates, curves, sampled = 0, [], []
    totals = dict(training_seconds=0., evaluation_seconds=0., checkpoint_seconds=0.,
                  training_F_forward_rows=0, training_F_backward_rows=0, training_F_physical_rows=0)
    losses = []
    latest = directory / "last.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["contract"] != active_contract:
            raise ValueError("Complete residual resume contract changed")
        updates = restore_bundle(saved, model, optimizer, sampler, active_contract)
        curves = saved["extra"]["curves"]
        sampled = saved["extra"]["sampled_indices"]
        losses = saved["extra"]["pending_batch_losses"]
        totals.update(saved["extra"]["totals"])
        if any(float(row["epoch"]) not in endpoints for row in curves):
            raise ValueError("Resume endpoints cannot discard previously scored checkpoints")
    else:
        torch.manual_seed(int(seed))
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))

    counts = tensors["mask"][fit].sum((0, 1)).to(device)
    if not torch.any(counts > 0):
        raise ValueError("No observed fitting targets")
    backward_rows = [0]

    def gradient_tracked_foundation(z):
        output = foundation(z)
        if output.requires_grad:
            rows = int(z.shape[0] * z.shape[-1])

            def record(gradient):
                backward_rows[0] += rows
                return gradient

            output.register_hook(record)
        return output

    training_foundation = None if foundation is None else gradient_tracked_foundation

    def save(endpoint=None):
        started = time.perf_counter()
        extra = dict(curves=curves, sampled_indices=sampled, pending_batch_losses=losses, totals=dict(totals),
                     phase=phase, seed=int(seed), exposures=int(sampler.exposures),
                     initial_state_sha256=active_contract["model_sha"])
        bundle = checkpoint_bundle(model, optimizer, sampler, active_contract, updates, extra)
        if endpoint is not None:
            target = directory / _endpoint_name(endpoint)
            if target.exists():
                previous = torch.load(target, map_location="cpu", weights_only=False)
                if previous["contract"] != active_contract or previous["extra"]["curves"] != curves:
                    raise ValueError("Existing scored checkpoint cannot be overwritten with changed results")
            else:
                _atomic_torch(target, bundle)
        _atomic_torch(latest, bundle)
        totals["checkpoint_seconds"] += time.perf_counter() - started

    status = "COMPLETE"
    try:
        for endpoint in endpoints:
            if any(float(row["epoch"]) == endpoint for row in curves):
                continue
            target_exposures = math.ceil(endpoint * len(fit))
            if sampler.exposures > target_exposures:
                raise ValueError("Partial resume advanced beyond an unscored requested endpoint")
            while sampler.exposures < target_exposures:
                _check_deadline(deadline)
                started = time.perf_counter()
                model.train()
                if foundation is not None and hasattr(foundation, "model"):
                    foundation.model.eval()
                before_streams = int(getattr(foundation, "streams", 0))
                before_physical = int(getattr(foundation, "physical_streams", 0))
                before_backward = backward_rows[0]
                previous_sampler = sampler.state_dict()
                previous_rng = torch.get_rng_state()
                previous_cuda_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else []
                rows = sampler.draw(min(RECIPE["effective_batch"], target_exposures - sampler.exposures))
                optimizer.zero_grad(set_to_none=True)
                value = 0.
                try:
                    for start in range(0, len(rows), RECIPE["microbatch"]):
                        chosen = rows[start:start + RECIPE["microbatch"]]
                        if callable(getattr(model, "training_loss", None)):
                            loss = model.training_loss(tensors["x"][chosen].to(device),
                                tensors["g"][chosen].to(device), tensors["y"][chosen].to(device),
                                tensors["mask"][chosen].to(device), training_foundation, counts, len(fit), len(rows))
                        else:
                            correction = _correction(model, tensors, chosen, training_foundation, device)
                            prediction = tensors["g"][chosen].to(device) + correction
                            loss = macro_loss(prediction, tensors["y"][chosen].to(device),
                                              tensors["mask"][chosen].to(device), counts, len(fit), len(rows))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite residual training loss")
                        loss.backward()
                        value += float(loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"])
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite residual gradient norm")
                except BaseException:
                    sampler.load_state_dict(previous_sampler)
                    torch.set_rng_state(previous_rng)
                    if previous_cuda_rng:
                        torch.cuda.set_rng_state_all(previous_cuda_rng)
                    optimizer.zero_grad(set_to_none=True)
                    raise
                optimizer.step()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                updates += 1
                sampled.extend(map(int, rows))
                losses.append(value)
                forward_rows = int(getattr(foundation, "streams", 0)) - before_streams
                totals["training_F_forward_rows"] += forward_rows
                totals["training_F_backward_rows"] += backward_rows[0] - before_backward
                totals["training_F_physical_rows"] += int(getattr(foundation, "physical_streams", 0)) - before_physical
                totals["training_seconds"] += time.perf_counter() - started
                if updates % 10 == 0:
                    save()
                    _atomic_json(directory / "PROGRESS.json", dict(phase=phase, status="TRAINING", updates=updates,
                        exposures=int(sampler.exposures), epoch=sampler.exposures / len(fit),
                        target_epoch=endpoint, training_seconds=totals["training_seconds"]))
            started = time.perf_counter()
            row = dict(epoch=endpoint, endpoint_epochs=endpoint, actual_epochs=sampler.exposures / len(fit),
                       updates=updates, exposures=int(sampler.exposures),
                       M_batch_loss=float(np.mean(losses)) if losses else None,
                       checkpoint_state_sha256=_state_sha(model))
            for name, indices in (("M_fit" if phase == "search" else "M_cal", fit), ("M_select", select)):
                if indices is not None:
                    stats = _evaluate(model, arrays, tensors, indices, foundation, device, deadline)
                    row.update({name + "_" + key: stats[key] for key in ("mse", "mae", "n_observed", "active_channels")})
            totals["evaluation_seconds"] += time.perf_counter() - started
            curves.append(row)
            losses = []
            save(endpoint)
    except _BudgetReached:
        status = "PARTIAL_CHECKPOINTED"
        save()
    except BaseException:
        save()
        raise

    selected = min(curves, key=lambda row: (row["M_select_mse"], row["epoch"])) if select is not None and curves else None
    result = dict(status=status, phase=phase, seed=int(seed), curves=curves,
                  selected_epoch=None if selected is None else selected["epoch"],
                  selected_M_select_mse=None if selected is None else selected["M_select_mse"],
                  selected_checkpoint=None if selected is None else str(directory / _endpoint_name(selected["epoch"])),
                  checkpoint_path=str(latest), state_path=str(latest),
                  checkpoints={format(row["epoch"], ".8g"): str(directory / _endpoint_name(row["epoch"])) for row in curves},
                  updates=updates, exposures=int(sampler.exposures), actual_epochs=sampler.exposures / len(fit),
                  sampled_indices_sha256=hashlib.sha256(np.asarray(sampled, np.int64).tobytes()).hexdigest(),
                  initial_state_sha256=active_contract["model_sha"], final_state_sha256=_state_sha(model),
                  n_fit=len(fit), n_select=0 if select is None else len(select),
                  recipe=RECIPE, contract=active_contract, **totals)
    _atomic_json(directory / "RESULT.json", result)
    return result


def run_search(model, data, fitidx, selectidx, seed, foundation, directory,
               endpoints=(0, .25, 1, 3), deadline=None, *, contract=None, device=None):
    """Search or extend using a freshly reconstructed identical initial model."""
    return _run(model, data, fitidx, selectidx, seed, foundation, directory,
                endpoints, deadline, contract, device, "search")


def run_refit(fresh_model, data, seed, foundation, directory, selected_epoch,
              deadline=None, *, contract=None, device=None):
    """Fit all M_cal from the same original parent initialization after sealing."""
    selected_epoch = float(selected_epoch)
    endpoints = (0,) if selected_epoch == 0 else (0, selected_epoch)
    result = _run(fresh_model, data, np.arange(len(data["x"]), dtype=np.int64), None,
                  seed, foundation, directory, endpoints, deadline, contract, device, "refit")
    result.update(selected_epoch=selected_epoch, fresh_initialization=True,
                  fit_split="M_cal_dense_stride1", initialization_source="same original A60 parent")
    _atomic_json(Path(directory) / "RESULT.json", result)
    return result
