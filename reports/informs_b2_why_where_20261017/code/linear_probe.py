"""Fixed-P/E, full-history masked ridge diagnostic; the default performs no I/O.

The output-wise objective is observed SSE/N_j + ||W_j||^2, with an
unpenalized intercept. Features use FIT-only population mean/std; the
intercept is eliminated by the mean over each output's observed FIT rows.
This is a diagnostic reader, not the adapter's D or a new deployed method.
"""
import hashlib
import json
from pathlib import Path
import time


def _check(value, message):
    if not value:
        raise ValueError(message)


def _json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode())
    result.update(json.dumps(list(value.shape)).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def _state_sha(adapter):
    import torch
    result = hashlib.sha256()
    for name, value in adapter.state_dict().items():
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _rss():
    import psutil
    return int(psutil.Process().memory_info().rss)


def _deadline_check(deadline, rss_limit):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Probe deadline reached; partial files are preserved")
    if _rss() > rss_limit:
        raise MemoryError("Probe RSS limit reached; partial files are preserved")


def mask_groups(mask):
    """Share a Gram only for outputs with byte-identical observed row masks."""
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    _check(mask.ndim == 2 and len(mask) >= 2, "Expected [N,D] observed FIT mask")
    groups = {}
    for column in range(mask.shape[1]):
        pattern = np.packbits(mask[:, column], bitorder="little").tobytes()
        groups.setdefault(pattern, []).append(column)
    result = []
    for columns in groups.values():
        count = int(mask[:, columns[0]].sum())
        _check(count >= 2, "Each output needs at least two observed FIT targets")
        result.append((columns, count))
    return result


def memory_estimate(n, p, d, batch, current_rss=0):
    """Conservative resident + mapped + solver workspace estimate, in bytes."""
    gram = 8 * p * p
    coefficients = 8 * p * d
    mapped = 4 * n * p + 8 * n * d + n * d
    blocks = 8 * batch * (3 * p + 3 * d)
    workspace = 128 * 1024**2
    incremental = mapped + 4 * gram + 3 * coefficients + blocks + workspace
    return dict(feature_cache_bytes=4*n*p, target_cache_bytes=8*n*d,
                mask_bytes=n*d, gram_bytes=gram, coefficient_bytes=coefficients,
                solver_gram_copies_reserved=4, coefficient_copies_reserved=3,
                workspace_reserve_bytes=workspace, current_rss_bytes=int(current_rss),
                projected_incremental_rss_bytes=int(incremental),
                projected_rss_bytes=int(current_rss+incremental),
                private_storage_upper_bytes=int(mapped+coefficients+16*p+8*d))


def preflight_probe(task, adapter, fit_origins, eval_origins, batch=32,
                    rss_limit=3*1024**3, max_estimated_flops=10**13):
    """Inspect allowed origin/mask metadata before allocating feature or Gram arrays."""
    import numpy as np
    fit = np.asarray(fit_origins, dtype=np.int64)
    evaluation = np.asarray(eval_origins, dtype=np.int64)
    length, horizon, channels, latent = (int(adapter.length), int(adapter.horizon),
                                        int(adapter.channels), int(adapter.latent_k))
    _check(batch >= 1 and task["L"] == length and task["H"] == horizon,
           "Positive batch and matching task/adapter L,H required")
    _check(task["physical"].shape == task["observed"].shape
           and task["physical"].shape[1] == channels, "Task C/mask mismatch")
    for origins in (fit, evaluation):
        _check(origins.ndim == 1 and len(origins) >= 1
               and np.all(np.diff(origins) > 0)
               and origins[0] >= length and origins[-1]+horizon <= len(task["observed"]),
               "Origins must be increasing, unique, and inside supplied allowed prefix")
    _check(fit[-1]+horizon <= evaluation[0], "FIT target range must precede evaluation")
    p, d = length*latent, horizon*channels
    # Metadata-only read, before any feature/model forward or Gram allocation.
    rows = fit[:, None] + np.arange(horizon)[None, :]
    observed = np.asarray(task["observed"][rows], dtype=bool).reshape(len(fit), d)
    groups = mask_groups(observed)
    estimate = memory_estimate(len(fit), p, d, batch, _rss())
    flops = sum(2*n*p*p + p**3/3 for _, n in groups) + 2*len(fit)*p*d
    estimate.update(schema="FULL_HISTORY_MASKED_RIDGE_PREFLIGHT_v1", fit_origins=len(fit),
                    eval_origins=len(evaluation), features=p, outputs=d,
                    coefficient_count=p*d, observed_mask_groups=len(groups),
                    observed_counts=[count for _, count in groups],
                    estimated_fit_flops=int(flops), rss_limit_bytes=int(rss_limit),
                    max_estimated_flops=int(max_estimated_flops),
                    full_feature_axis=True, dimensionality_reduction=False,
                    model_forward_executed=False, protected_T_access=False)
    estimate["admissible"] = (estimate["projected_rss_bytes"] <= rss_limit
                              and flops <= max_estimated_flops)
    return estimate


def feature_statistics(features, batch=32):
    """Population mean/std by a stable streaming merge, using only FIT features."""
    import numpy as np
    n, p = features.shape
    count, mean, m2 = 0, np.zeros(p), np.zeros(p)
    for start in range(0, n, batch):
        block = np.asarray(features[start:start+batch], dtype=np.float64)
        _check(np.isfinite(block).all(), "Finite extracted FIT features required")
        size, center = len(block), block.mean(axis=0)
        centered = block-center
        delta, total = center-mean, count+size
        m2 += np.sum(centered*centered, axis=0) + delta*delta*(count*size/total)
        mean += delta*(size/total)
        count = total
    scale = np.sqrt(np.maximum(m2/count, 0.0))
    constant = scale == 0
    scale[constant] = 1.0
    return mean, scale, constant


def fit_masked_ridge(features, targets, mask, *, batch=32, deadline=None,
                     rss_limit=3*1024**3, progress=None):
    """Streaming FP64 sufficient statistics; fixed lambda=1; no V selection.

    Cached FP32 full features may be memmaps. Only one mask-group Gram is
    resident at a time. No target with a false mask participates in arithmetic.
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve
    from threadpoolctl import threadpool_info
    _check(all(pool["num_threads"] <= 2 for pool in threadpool_info()
               if pool["user_api"] == "blas"), "Caller must enforce BLAS CPU2")
    x, y, m = features, targets, np.asarray(mask, dtype=bool)
    _check(len(x.shape) == 2 and len(y.shape) == 2 and len(x) == len(y)
           and m.shape == y.shape, "Aligned [N,P], [N,D], [N,D] required")
    n, p, d = len(x), x.shape[1], y.shape[1]
    center, scale, constant = feature_statistics(x, batch)
    groups = mask_groups(m)
    coefficient, intercept = np.zeros((p, d)), np.zeros(d)
    diagnostics = []
    for group_index, (columns, count) in enumerate(groups):
        _deadline_check(deadline, rss_limit)
        gram, rhs = np.zeros((p, p)), np.zeros((p, len(columns)))
        sum_x, sum_y, target_ss = np.zeros(p), np.zeros(len(columns)), np.zeros(len(columns))
        for start in range(0, n, batch):
            valid = m[start:start+batch, columns[0]]
            if not valid.any():
                continue
            a = (np.asarray(x[start:start+batch], dtype=np.float64)[valid]-center)/scale
            a[:, constant] = 0
            b = np.asarray(y[start:start+batch], dtype=np.float64)[np.ix_(valid, columns)]
            _check(np.isfinite(b).all(), "Finite observed targets required")
            gram += a.T @ a
            rhs += a.T @ b
            sum_x += a.sum(axis=0)
            sum_y += b.sum(axis=0)
            target_ss += np.sum(b*b, axis=0)
            _deadline_check(deadline, rss_limit)
        mean_x, mean_y = sum_x/count, sum_y/count
        gram /= count
        gram -= np.outer(mean_x, mean_x)
        gram = (gram+gram.T)*0.5
        rhs /= count
        rhs -= np.outer(mean_x, mean_y)
        gram.flat[::p+1] += 1.0
        factor = cho_factor(gram, lower=True, overwrite_a=False, check_finite=True)
        solved = cho_solve(factor, rhs, check_finite=True)
        residual = gram @ solved-rhs
        denominator = np.linalg.norm(gram)*np.linalg.norm(solved, axis=0)+np.linalg.norm(rhs, axis=0)
        relative = np.linalg.norm(residual, axis=0)/np.maximum(denominator, 1e-30)
        _check(np.isfinite(solved).all() and float(relative.max()) <= 1e-8,
               "Ridge normal-equation residual exceeds 1e-8")
        bias = mean_y-mean_x@solved
        # The feasible intercept-only objective bounds the penalized optimum.
        baseline = np.maximum(target_ss/count-mean_y*mean_y, 0.0)
        objective = baseline-np.sum(rhs*solved, axis=0)
        _check(np.all(objective >= -1e-8*np.maximum(1, baseline))
               and np.all(objective <= baseline+1e-8*np.maximum(1, baseline)),
               "Penalized optimum violates intercept-only feasibility")
        coefficient[:, columns], intercept[columns] = solved, bias
        diagnostics.append(dict(group=group_index, output_columns=columns,
                                observed_count=count, max_relative_residual=float(relative.max()),
                                objective_min=float(objective.min()), objective_max=float(objective.max()),
                                intercept_only_objective_min=float(baseline.min()),
                                intercept_only_objective_max=float(baseline.max())))
        del gram, rhs, factor, solved, residual
        if progress is not None:
            progress(group_index+1, len(groups))
    return dict(coefficient=coefficient, intercept=intercept, feature_center=center,
                feature_scale=scale, constant_features=constant), dict(
                    objective="SSE_observed/N_j + lambda*||W_j||^2", lambda_value=1.0,
                    intercept_penalized=False, feature_scaling="FIT-only population mean/std",
                    masked_centering="output-mask group mean; unpenalized intercept eliminated",
                    solver="FP64 streaming sufficient statistics, scipy Cholesky per identical mask group",
                    feature_flatten_order="row-major L,K; all L*K features retained",
                    constant_features=int(constant.sum()), mask_groups=diagnostics,
                    normal_equation_relative_tolerance=1e-8, V_used_for_fit_or_lambda=False,
                    coefficient_count=p*d, fit_origins=n, features=p, outputs=d)


def ridge_prediction(features, fitted):
    import numpy as np
    a = (np.asarray(features, dtype=np.float64)-fitted["feature_center"])/fitted["feature_scale"]
    a[:, fitted["constant_features"]] = 0
    return a@fitted["coefficient"]+fitted["intercept"]


def _memmap_new(path, dtype, shape):
    import numpy as np
    with Path(path).open("xb") as stream:
        stream.truncate(int(np.prod(shape))*np.dtype(dtype).itemsize)
    return np.memmap(path, mode="r+", dtype=dtype, shape=shape)


def run_probe(task, adapter, normalization, fit_origins, eval_origins,
              private_output, public_output, batch=32, *, run_metadata=None,
              windows_factory=None, rss_limit=3*1024**3, deadline=None,
              max_estimated_flops=10**13):
    """Run one explicitly reserved fit; caller enforces the campaign's six-fit cap.

    private_output/public_output are fresh directories belonging to this run.
    FastWindows supplies original normalized targets and observed masks.
    Only adapter.encode is called; neither adapter.decode nor F is called.
    """
    import numpy as np
    import torch
    if windows_factory is None:
        from training import FastWindows
        windows_factory = FastWindows
    private, public = Path(private_output), Path(public_output)
    _check(not private.exists() and not public.exists(), "Use fresh probe run directories")
    _check(all(p.device.type == "cpu" for p in adapter.parameters()), "CPU-only adapter required")
    _check(torch.get_num_threads() <= 2, "Caller must enforce CPU2 before probe execution")
    for name in ("mean", "std"):
        value = np.asarray(normalization[name])
        _check(value.shape == (adapter.channels,) and np.isfinite(value).all(), "Finite per-channel normalization required")
    _check(np.all(np.asarray(normalization["std"]) > 0), "Positive normalization scale required")
    preflight = preflight_probe(task, adapter, fit_origins, eval_origins, batch,
                                rss_limit, max_estimated_flops)
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    _json_new(private/"PREFLIGHT.json", preflight)
    _check(preflight["admissible"], "Probe memory/compute preflight exceeds registered limit")
    started, initial_sha = time.time(), _state_sha(adapter)
    _deadline_check(deadline, rss_limit)
    adapter.eval()
    adapter.requires_grad_(False)
    _check(not any(p.requires_grad for p in adapter.parameters()), "Fixed P/E required")
    windows = windows_factory(task, normalization["mean"], normalization["std"], "cpu")
    fit_origins, eval_origins = np.asarray(fit_origins, np.int64), np.asarray(eval_origins, np.int64)
    n, p, d = preflight["fit_origins"], preflight["features"], preflight["outputs"]
    features = _memmap_new(private/"FIT_FEATURES_FP32.dat", np.float32, (n, p))
    targets = _memmap_new(private/"FIT_DELTA_FP64.dat", np.float64, (n, d))
    observed = _memmap_new(private/"FIT_OBSERVED_BOOL.dat", np.bool_, (n, d))
    with torch.no_grad():
        for start in range(0, n, batch):
            x, y, mask = windows.get(fit_origins[start:start+batch])
            z = adapter.encode(x, int(adapter.latent_k))
            _check(tuple(z.shape) == (len(x), adapter.length, adapter.latent_k), "Full latent-history shape mismatch")
            _check(torch.isfinite(z).all().item(), "Finite P/E feature extraction required")
            features[start:start+len(x)] = z.cpu().numpy().reshape(len(x), p)
            # Promote before subtraction: no hidden FP32 delta-rounding in the probe target.
            delta = y.cpu().numpy().astype(np.float64)-x[:, -1:, :].cpu().numpy().astype(np.float64)
            local_mask = mask.cpu().numpy().reshape(len(x), d)
            targets[start:start+len(x)] = np.where(local_mask, delta.reshape(len(x), d), 0.0)
            observed[start:start+len(x)] = local_mask
            _deadline_check(deadline, rss_limit)
    features.flush(); targets.flush(); observed.flush()
    fitted, solver = fit_masked_ridge(features, targets, observed, batch=batch,
                                       deadline=deadline, rss_limit=rss_limit)
    with (private/"RIDGE_COEFFICIENTS_FP64.npz").open("xb") as stream:
        np.savez(stream, **fitted, fit_origins=fit_origins, eval_origins=eval_origins)
    horizon, channels = int(adapter.horizon), int(adapter.channels)
    prediction = np.empty((len(eval_origins), horizon, channels), np.float64)
    truth, mask_all = np.empty_like(prediction), np.empty(prediction.shape, dtype=bool)
    with torch.no_grad():
        for start in range(0, len(eval_origins), batch):
            x, y, mask = windows.get(eval_origins[start:start+batch])
            z = adapter.encode(x, int(adapter.latent_k))
            delta = ridge_prediction(z.cpu().numpy().reshape(len(x), p), fitted).reshape(len(x), horizon, channels)
            prediction[start:start+len(x)] = delta+x[:, -1:, :].cpu().numpy().astype(np.float64)
            truth[start:start+len(x)], mask_all[start:start+len(x)] = y.cpu().numpy(), mask.cpu().numpy()
            _deadline_check(deadline, rss_limit)
    _check(np.isfinite(prediction).all(), "Finite probe evaluation predictions required")
    counts = mask_all.sum(axis=(0, 1))
    _check(np.all(counts > 0), "Every evaluated channel requires observed targets")
    error = np.where(mask_all, prediction-truth, 0.0)
    sse, sae = np.sum(error*error, axis=(0, 1)), np.sum(np.abs(error), axis=(0, 1))
    _check(_state_sha(adapter) == initial_sha, "Fixed adapter state changed during probe")
    with (private/"V_PREDICTIONS_FP64.npz").open("xb") as stream:
        np.savez(stream, prediction=prediction, y=truth, mask=mask_all, origins=eval_origins)
    result = dict(schema="FULL_HISTORY_FIXED_PE_RIDGE_PROBE_v1", status="COMPLETE",
                  metadata=run_metadata or {}, evaluation="reused development V; not independent generalization",
                  MSE=float(np.mean(sse/counts)), MAE=float(np.mean(sae/counts)),
                  channel_SSE=sse.tolist(), channel_SAE=sae.tolist(), channel_observed=counts.tolist(),
                  fit_origins=n, eval_origins=len(eval_origins), fit_exposure=n,
                  fit_origins_sha256=_array_sha(fit_origins), eval_origins_sha256=_array_sha(eval_origins),
                  fit_observed_sha256=_array_sha(observed), V_mask_sha256=_array_sha(mask_all),
                  input_features=p, output_features=d, coefficient_count=p*d,
                  parameter_count_with_intercept=p*d+d, solver=solver,
                  preflight=preflight, wall_seconds=time.time()-started,
                  final_rss_bytes=_rss(), adapter_state_sha256=initial_sha,
                  PE_frozen=True, gradient_enabled_during_extraction=False,
                  original_D_called=False, foundation_called=False, new_model_training=False,
                  fitted_probe_count=1, protected_T_access=False,
                  inference_precision="FP32 P/E extraction then FP64 ridge and metrics",
                  claim_limit="Future information accessibility under this fixed reader and lambda; not common-factor identification")
    _json_new(public/"RESULT.json", result)
    return result


if __name__ == "__main__":
    print("SOURCE_ONLY: import run_probe for one explicitly reserved fixed-P/E probe; no input read or model executed")
