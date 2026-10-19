"""M_cal-only fixed-parent diagnostics and fresh CPU head fitting.

This module has no loaders, file writes, foundation calls or V selection inputs.
The caller owns parent provenance, private state storage and later sealed V use.
"""
import hashlib
import math
import time

import numpy as np
import torch

from research.informs_broad_validation_20261015_v1.evaluation import fit_global_alpha
from research.informs_broad_validation_20261015_v1.networks import ConditionalHead, module_state_hash
from research.informs_broad_validation_20261015_v1.training import PermutationStream, cpu_state, macro_loss


def calibration_split(task):
    """Return dense M_cal origins and target-disjoint 60/40 index subsets."""
    m0, m1 = task["split_ranges"]["M_cal"]
    length, horizon = task["L"], task["H"]
    if any(not isinstance(v, (int, np.integer)) for v in (m0, m1, length, horizon)):
        raise ValueError("Integer M_cal bounds and L/H required")
    if min(length, horizon) < 1 or m0 < 0 or m1 <= m0 or length < horizon:
        raise ValueError("Positive M_cal interval and L>=H required")
    mcut = int(m0 + (3 * (m1 - m0)) // 5)
    origins = np.arange(max(m0, length), m1 - horizon + 1, dtype=np.int64)
    fit = np.flatnonzero(origins + horizon <= mcut)
    select = np.flatnonzero(origins >= mcut)
    discarded = np.flatnonzero((origins + horizon > mcut) & (origins < mcut))
    if not len(fit) or not len(select):
        raise ValueError("M_cal must contain nonempty target-disjoint M_fit and M_select")
    return dict(origins=origins, fit_indices=fit, select_indices=select,
                discarded_indices=discarded, mcut=mcut, bounds=[int(m0), int(m1)])


def validate_parent_scope(metadata, task):
    """Check recorded parent fit/selection endpoints without opening any data."""
    m0 = task["split_ranges"]["M_cal"][0]
    for key in ("parent_fit_end", "parent_selection_end"):
        if key == "parent_selection_end" and key not in metadata:
            continue
        value = metadata.get(key)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not 0 < value <= m0:
            raise ValueError("Parent fitting and configuration selection must end before M_cal")
    if metadata.get("mcal_targets_used", False):
        raise ValueError("Parent fitting and configuration selection must end before M_cal")
    return dict(parent_fit_end=int(metadata["parent_fit_end"]), mcal_start=int(m0))


def _scoring_arrays(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(mask)
    if prediction.ndim != 3 or min(prediction.shape) < 1 or prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Matching positive [origins,horizon,channels] arrays required")
    if mask.dtype != np.bool_:
        raise ValueError("Original bool observation mask required")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Finite prediction and target required on observed targets")
    return prediction, target, mask


def metric_statistics(prediction, target, mask):
    """Independent FP64 channel macro scores and horizon/channel sufficient sums."""
    prediction, target, mask = _scoring_arrays(prediction, target, mask)
    difference = np.zeros(prediction.shape, np.float64)
    np.subtract(prediction, target, out=difference, where=mask)
    horizon_sse = np.sum(difference * difference, axis=0, dtype=np.float64)
    horizon_sae = np.sum(np.abs(difference), axis=0, dtype=np.float64)
    horizon_count = np.sum(mask, axis=0, dtype=np.int64)
    sse, sae, counts = horizon_sse.sum(0), horizon_sae.sum(0), horizon_count.sum(0)
    active = np.flatnonzero(counts)
    return dict(mse=math.fsum(float(sse[c]) / int(counts[c]) for c in active) / len(active) if len(active) else None,
                mae=math.fsum(float(sae[c]) / int(counts[c]) for c in active) / len(active) if len(active) else None,
                n_observed=int(counts.sum()), active_channels=int(len(active)),
                channel_sse=sse.tolist(), channel_sae=sae.tolist(), channel_counts=counts.tolist(),
                horizon_channel_sse=horizon_sse.tolist(), horizon_channel_sae=horizon_sae.tolist(),
                horizon_channel_count=horizon_count.tolist())


def _scores(prediction, target, mask):
    stats = metric_statistics(prediction, target, mask)
    return {key: stats[key] for key in ("mse", "mae", "n_observed", "active_channels")}


def _indices(value, n):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or not len(value):
        raise ValueError("Nonempty integer calibration indices required")
    if len(np.unique(value)) != len(value) or np.any(value < 0) or np.any(value >= n):
        raise ValueError("Unique in-range calibration indices required")
    return value.astype(np.int64, copy=False)


def fit_alpha_diagnostic(g, b, y, mask, fit_indices, select_indices):
    """Fit on M_fit, score fixed/fitted alpha on M_fit/M_select, then fresh M_cal fit.

    The inputs must already be aligned dense M_cal arrays. The split helper's
    integer subsets carry the chronology; no V data can be passed separately.
    """
    g, y, mask = _scoring_arrays(g, y, mask)
    b, _, _ = _scoring_arrays(b, y, mask)
    fit, select = _indices(fit_indices, len(g)), _indices(select_indices, len(g))
    if np.intersect1d(fit, select).size:
        raise ValueError("M_fit and M_select indices must be disjoint")
    if int(fit.max()) + g.shape[1] > int(select.min()):
        raise ValueError("M_fit and M_select targets overlap or violate chronology")
    fitted = fit_global_alpha(g[fit], b[fit], y[fit], mask[fit])
    rows = []
    for label, alpha in (("alpha0", 0.), ("alpha05", .5), ("alpha1", 1.), ("fitted", fitted["alpha"])):
        row = dict(label=label, alpha=float(alpha))
        for name, indices in (("M_fit", fit), ("M_select", select)):
            prediction = g[indices] + alpha * (b[indices] - g[indices])
            row.update({name + "_" + key: value for key, value in _scores(prediction, y[indices], mask[indices]).items()})
        rows.append(row)
    return dict(fit_split="M_fit", checkpoint_split="M_select", fit_alpha=fitted,
                refit_split="M_cal_dense_stride1", refit_alpha=fit_global_alpha(g, b, y, mask), rows=rows)


def _prediction_arrays(past, g, b, control):
    past, g = np.asarray(past, np.float32), np.asarray(g, np.float32)
    if past.ndim != 3 or g.ndim != 3 or min(past.shape) < 1 or min(g.shape) < 1:
        raise ValueError("Positive [origins,length/chorizon,channels] history and parent arrays required")
    if past.shape[0] != g.shape[0] or past.shape[2] != g.shape[2] or past.shape[1] < g.shape[1]:
        raise ValueError("History and parent dimensions must agree with L>=H")
    if not np.isfinite(past).all() or not np.isfinite(g).all():
        raise ValueError("Finite filled past and parent g required")
    if control:
        b = g  # Exact ConditionalHead control ignores b; no B2 array is read.
    else:
        if b is None:
            raise ValueError("Conditional head requires parent b")
        b = np.asarray(b, np.float32)
        if b.shape != g.shape or not np.isfinite(b).all():
            raise ValueError("Finite aligned parent b required")
    return past, g, b


def validate_calibration_arrays(past, g, b, y, mask, origins, task, control=False):
    """Validate arrays against the exact dense M_cal cohort, without task data reads."""
    past, g, b = _prediction_arrays(past, g, b, control)
    _, y, mask = _scoring_arrays(g, y, mask)
    oo = np.asarray(origins)
    split = calibration_split(task)
    if oo.dtype.kind not in "iu" or not np.array_equal(oo, split["origins"]):
        raise ValueError("Head fitting requires the exact registered dense M_cal origins")
    if past.shape[1] != task["L"] or g.shape[1] != task["H"] or len(g) != len(oo):
        raise ValueError("Head arrays differ from task L/H or origin count")
    if "columns" in task and g.shape[2] != len(task["columns"]):
        raise ValueError("Head channel count differs from registered task columns")
    return past, g, b, y, mask, split


def _new_head(length, horizon, seed):
    # Preserve global CPU RNG; the sampler owns an independent NumPy stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        return ConditionalHead(length, horizon, hidden=32).cpu()


def _tensor_inputs(past, g, b, y, mask):
    return tuple(torch.as_tensor(value, device="cpu") for value in
                 (past, g, b, np.where(mask, y, 0).astype(np.float32), mask))


def _schedule(epochs, effective_batch, microbatch, lr):
    values = tuple(float(value) for value in epochs)
    if not values or values[0] != 0 or values != tuple(sorted(set(values))) or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Epoch endpoints must be increasing, finite and start at zero")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (effective_batch, microbatch)) or not 1 <= microbatch <= effective_batch:
        raise ValueError("Require integer 1<=microbatch<=effective_batch")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Positive finite learning rate required")
    return values


def _check_deadline(deadline):
    if deadline is not None and time.time() >= float(deadline):
        raise TimeoutError("Registered wall/GPU budget reached during calibration")


def _train(head, tensors, indices, origins, seed, endpoints, control, effective_batch, microbatch, lr, callback, deadline=None):
    _check_deadline(deadline)
    x, g, b, target, mask = tensors
    counts = mask[indices].sum((0, 1))
    if not torch.any(counts > 0):
        raise ValueError("No observed training targets")
    sampler = PermutationStream(indices, seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
    sample_hash, updates, clipped, losses = hashlib.sha256(), 0, 0, []
    for endpoint in endpoints:
        segment_losses = []
        end_exposures = math.ceil(endpoint * len(indices))
        head.train()
        while sampler.exposures < end_exposures:
            _check_deadline(deadline)
            batch = sampler.draw(min(effective_batch, end_exposures - sampler.exposures))
            sample_hash.update(np.asarray(origins[batch], np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for lo in range(0, len(batch), microbatch):
                _check_deadline(deadline)
                rows = batch[lo:lo + microbatch]
                prediction = head(x[rows], g[rows], b[rows], control=control)
                loss = macro_loss(prediction, target[rows], mask[rows], counts, len(indices), len(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite M_cal head loss")
                loss.backward()
                batch_loss += float(loss.detach())
            _check_deadline(deadline)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite M_cal head gradient")
            clipped += int(norm > 1)
            optimizer.step()
            updates += 1
            segment_losses.append(batch_loss)
        losses.extend(segment_losses)
        callback(dict(endpoint_epochs=endpoint, actual_epochs=sampler.exposures / len(indices),
                      updates=updates, exposures=sampler.exposures,
                      M_batch_loss=float(np.mean(segment_losses)) if segment_losses else None))
        _check_deadline(deadline)
    return dict(updates=updates, exposures=sampler.exposures, actual_epochs=sampler.exposures / len(indices),
                sampler_sha256=sample_hash.hexdigest(),
                clipped_steps=clipped, mean_batch_loss=float(np.mean(losses)) if losses else None)


def _predict(head, past, g, b, control, batch_size, deadline=None):
    head.eval()
    outputs = []
    with torch.no_grad():
        for lo in range(0, len(g), batch_size):
            _check_deadline(deadline)
            tensors = [torch.as_tensor(value[lo:lo + batch_size], device="cpu") for value in (past, g, b)]
            outputs.append(head(*tensors, control=control).numpy())
    _check_deadline(deadline)
    return np.concatenate(outputs, axis=0)


def _recipe(control, effective_batch, microbatch, lr):
    return dict(device="cpu", hidden=32, learning_rate=float(lr), weight_decay=0.,
                effective_batch=effective_batch, microbatch=microbatch, gradient_clip_norm=1.,
                control=bool(control), deployment_parents=["g"] if control else ["g", "b"],
                foundation_calls_during_fit=0)


def head_search(past, g, b, y, mask, origins, task, seed, control=False,
                epochs=(0, .25, 1, 3), effective_batch=16, microbatch=4, lr=.001,
                *, checkpoint_sink=None, deadline=None):
    """Train only M_fit and select strict minimum M_select MSE, earliest exact tie.

    Optional checkpoint_sink(cpu_state_dict, curve_row) observes each scored
    endpoint including CP0. Both arguments are copies; storage belongs to caller.
    deadline is an optional operational Unix timestamp, outside the recipe.
    """
    _check_deadline(deadline)
    endpoints = _schedule(epochs, effective_batch, microbatch, lr)
    past, g, b, y, mask, split = validate_calibration_arrays(past, g, b, y, mask, origins, task, control)
    fit, select = split["fit_indices"], split["select_indices"]
    if not mask[select].any():
        raise ValueError("No observed M_select targets")
    _check_deadline(deadline)
    head = _new_head(task["L"], task["H"], seed)
    initialization = module_state_hash(head)
    curves = []
    def evaluate(row):
        for name, indices in (("M_fit", fit), ("M_select", select)):
            prediction = _predict(head, past[indices], g[indices], b[indices], control, microbatch, deadline)
            row.update({name + "_" + key: value for key, value in _scores(prediction, y[indices], mask[indices]).items()})
        row["checkpoint_state_sha256"] = module_state_hash(head)
        if checkpoint_sink is not None:
            checkpoint_sink(cpu_state(head), dict(row))
        curves.append(row)
    training = _train(head, _tensor_inputs(past, g, b, y, mask), fit, np.asarray(origins), seed, endpoints,
                      control, effective_batch, microbatch, lr, evaluate, deadline)
    selected = min(curves, key=lambda row: (row["M_select_mse"], row["endpoint_epochs"]))
    return dict(seed=int(seed), selected_epoch=selected["endpoint_epochs"], selected_M_select_mse=selected["M_select_mse"],
                initialization_sha256=initialization, curves=curves, **training,
                fit_split="M_fit", checkpoint_split="M_select", mcut=split["mcut"],
                n_fit=len(fit), n_select=len(select), n_discarded=len(split["discarded_indices"]),
                **_recipe(control, effective_batch, microbatch, lr))


def head_refit(past, g, b, y, mask, origins, task, seed, selected_epoch, control=False,
               effective_batch=16, microbatch=4, lr=.001, *, deadline=None):
    """Start from seed initialization and fit all M_cal for the sealed epoch only.

    Returns (CPU state_dict, JSON receipt). No search state is accepted.
    """
    _check_deadline(deadline)
    selected_epoch = float(selected_epoch)
    endpoints = _schedule((0,) if selected_epoch == 0 else (0, selected_epoch), effective_batch, microbatch, lr)
    past, g, b, y, mask, split = validate_calibration_arrays(past, g, b, y, mask, origins, task, control)
    _check_deadline(deadline)
    head = _new_head(task["L"], task["H"], seed)
    initialization = module_state_hash(head)
    indices = np.arange(len(g), dtype=np.int64)
    training = _train(head, _tensor_inputs(past, g, b, y, mask), indices, np.asarray(origins), seed, endpoints,
                      control, effective_batch, microbatch, lr, lambda row: None, deadline)
    receipt = dict(seed=int(seed), selected_epoch=selected_epoch, initialization_sha256=initialization,
                   final_state_sha256=module_state_hash(head), fit_split="M_cal_dense_stride1",
                   n_fit=len(indices), fresh_initialization=True, head_config=dict(length=task["L"], horizon=task["H"], hidden=32),
                   **training, **_recipe(control, effective_batch, microbatch, lr))
    return cpu_state(head), receipt


def predict_head(state, past, g, b=None, control=False, batch_size=4):
    """CPU prediction without targets or scoring; control never reads parent b."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Positive integer prediction batch size required")
    past, g, b = _prediction_arrays(past, g, b, control)
    if any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("Head state must contain CPU tensors")
    head = _new_head(past.shape[1], g.shape[1], 0)
    head.load_state_dict(state, strict=True)
    return _predict(head, past, g, b, control, batch_size)
