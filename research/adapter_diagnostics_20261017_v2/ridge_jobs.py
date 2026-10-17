"""Paired own/cross-channel block Ridge; no model or data IO at import."""
from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time

from . import data_audit
from .core import atomic_json, digest, read, sha


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("B02 finite job deadline reached")


def libraries(repo):
    public = Path(repo) / "research" / data_audit.CAMPAIGN
    data = data_audit.module(public / "data_contract.py", "b02_data")
    previous = sys.modules.get("data_contract")
    sys.modules["data_contract"] = data
    try:
        ridge = data_audit.module(public / "ridge_baseline.py", "b02_ridge")
    finally:
        if previous is None:
            sys.modules.pop("data_contract", None)
        else:
            sys.modules["data_contract"] = previous
    primitive = data_audit.module(public / "priority_b2_validation_20261016_v1" / "safe_jena_prefix.py", "b02_primitive")
    return public, data, ridge, primitive


def load_cohort(cohort, ctx, public, data, primitive):
    import numpy as np
    contracts = read(public / "DATA_CONTRACT.json")["assets"]
    if cohort == "jena":
        task, access = primitive.load_jena_prefix(ctx["asset"], contracts["jena"])
        return task, access
    if cohort != "pems32":
        raise ValueError("Only registered Jena21/PeMS32 B02 cohorts")
    source_cache = Path(ctx["asset"]) / ".cache" / data_audit.CAMPAIGN
    task, access = data_audit._prepared_prefix("pems08", source_cache, contracts["pems08"], primitive)
    task["metadata"] = contracts["pems08"]["metadata"].copy()
    subset_root = public / "priority_pems_subset_20261016_v1"
    plan_path = subset_root / "PLAN.json"
    plan = read(plan_path)
    subset = data_audit.module(subset_root / "subset_minimum_runner.py", "b02_subset")
    if (plan["column_selection"] != subset.select_columns(contracts["pems08"]["columns"])
            or plan["parent_task_contract"] != contracts["pems08"]
            or plan["original_primary_split_ranges"] != contracts["pems08"]["primary_split_ranges"]):
        raise ValueError("Original PeMS32 metadata selection changed")
    task = subset.subset_task(task, plan, dict(np=np, data=data))
    access.update(subset_plan_sha256=sha(plan_path), subset_data_sha256=task["data_sha256"],
                  subset_columns_sha256=data.array_sha256(np.asarray(task["columns"], dtype=str)),
                  selected_indices=plan["column_selection"]["selected_indices"])
    return task, access


def cross_features(prepared, origins):
    import numpy as np
    return np.column_stack([prepared.channel_features(c, origins)[:, :-1]
                            for c in range(prepared.mask.shape[1])] + [np.ones(len(origins))])


def targets(prepared, origins):
    import numpy as np
    rows = [prepared.channel_targets(c, origins) for c in range(prepared.mask.shape[1])]
    return np.stack([r[0] for r in rows], axis=-1), np.stack([r[1] for r in rows], axis=-1)


def fit_cross(design, target, observed, alphas, deadline=float("inf")):
    """Same SSE_c/N_c + alpha L2 convention as the historical own solver."""
    import numpy as np
    n, h, c = target.shape
    p = design.shape[1]
    coefficient = np.zeros((len(alphas), p, h*c), np.float64)
    y, mask = target.reshape(n, h*c), observed.reshape(n, h*c)
    counts = observed.sum(axis=(0, 1))
    regularizer = np.eye(p)
    regularizer[-1, -1] = 0
    groups = {}
    for out in range(h*c):
        total = int(counts[out % c])
        if total and mask[:, out].any():
            key = (np.packbits(mask[:, out]).tobytes(), total)
            groups.setdefault(key, []).append(out)
    fallback, maximum_residual = 0, 0.
    for (_, total), outputs in groups.items():
        check_deadline(deadline)
        usable = mask[:, outputs[0]]
        x, selected_y = design[usable], y[usable][:, outputs]
        gram, rhs = x.T @ x / total, x.T @ selected_y / total
        for index, alpha in enumerate(alphas):
            check_deadline(deadline)
            matrix = gram + float(alpha) * regularizer
            try:
                beta = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
                fallback += 1
            maximum_residual = max(maximum_residual, float(np.max(np.abs(matrix @ beta - rhs))))
            coefficient[index][:, outputs] = beta
    if not np.isfinite(coefficient).all():
        raise ValueError("Nonfinite cross-channel ridge coefficients")
    return coefficient, dict(mask_groups=len(groups), solver_fallbacks=fallback,
                             max_normal_equation_abs_residual=maximum_residual)


def score_delta(predicted_delta, target_delta, observed):
    import numpy as np
    difference = predicted_delta - target_delta
    sse = np.where(observed, difference**2, 0).sum(axis=0, dtype=np.float64)
    sae = np.where(observed, np.abs(difference), 0).sum(axis=0, dtype=np.float64)
    counts = observed.sum(axis=0)
    active = counts.sum(axis=0) > 0
    if not active.any():
        raise ValueError("No observed evaluation target")
    mse = float(np.mean(sse.sum(axis=0)[active] / counts.sum(axis=0)[active]))
    mae = float(np.mean(sae.sum(axis=0)[active] / counts.sum(axis=0)[active]))
    return dict(MSE=mse, MAE=mae, observed_count=int(counts.sum()), active_channels=int(active.sum())), sse, sae, counts


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execute_job(job, ctx):
    import numpy as np
    from threadpoolctl import threadpool_limits
    if job["question_id"] != "B02":
        raise ValueError("Only B02 jobs supported")
    cohort = job.get("cohort", job.get("dataset_cohort"))
    public, data, ridge, primitive = libraries(ctx["repo"])
    task, access = load_cohort(cohort, ctx, public, data, primitive)
    deadline = ctx.get("job_deadline", ctx["deadline"])
    bounds = task["split_ranges"]
    a50, a60 = bounds["A_fit"][1], bounds["A_stop"][1]
    train = data.origins((0, a50), task["L"], task["H"], 1)
    stop = data.origins(bounds["A_stop"], task["L"], task["H"], task["H"])
    refit = data.origins((0, a60), task["L"], task["H"], 1)
    evaluation = data.origins(bounds["V_select"], task["L"], task["H"], task["H"])
    c, h, p = len(task["columns"]), task["H"], 32*len(task["columns"])+1
    # Dominant design, RHS, 5 candidate coefficients and solver workspaces.
    estimated_bytes = 8*(len(refit)*p + 8*p*p + 8*p*h*c + 4*len(refit)*h*c)
    preflight = dict(feature_count_own=33, feature_count_cross=p, estimated_incremental_peak_bytes=estimated_bytes,
                     CPU_threads=2, new_F_calls=0, training_origins=len(train), refit_origins=len(refit))
    if estimated_bytes > 3*1024**3:
        return dict(status="SKIPPED_BUDGET", reason="Cross-channel Gram/design estimated above 3GiB worker limit",
                    artifact_paths=[], result=dict(preflight=preflight))
    private = Path(ctx["cache"]) / "B02" / cohort
    destination = Path(ctx["report"]) / "B02" / cohort
    public_result = destination / "RESULT.json"
    if public_result.exists():
        old = read(public_result)
        outputs = [destination / "RESULT.json", destination / "ERROR_BY_HORIZON_CHANNEL.csv"]
        if (old.get("scientific_identity") == job["scientific_identity"] and
                outputs[1].is_file() and sha(outputs[1]) == old.get("sufficient_statistics_sha256") and
                all(path.is_file() and sha(path) == old["private_artifact_hashes"][key]
                    for key, path in (("coefficients", private / "coefficients.npz"),
                                      ("prediction", private / "prediction.npz")))):
            return dict(status="REUSED_VERIFIED", reason="Same B02 scientific identity and private outputs verified",
                        artifact_paths=[str(path) for path in outputs], result=old)
        raise ValueError("Existing B02 outputs have a different identity or missing private artifacts")
    timings, methods, diagnostics = {}, {}, {}
    started = time.perf_counter()
    with threadpool_limits(limits=2, user_api="blas"):
        check_deadline(deadline)
        mean50, std50, _ = data.fit_statistics(task, a50)
        prepared = ridge.PrefixFeatures(task, mean50, std50, a60)
        x_train, x_stop = cross_features(prepared, train), cross_features(prepared, stop)
        y_train, m_train = targets(prepared, train)
        y_stop, m_stop = targets(prepared, stop)
        stage = time.perf_counter()
        own_candidates, _, own_fallback = ridge.fit_coefficients(prepared, train, ridge.ALPHAS)
        own_scores, _, _ = ridge.stop_scores(prepared, stop, own_candidates)
        selected_own = int(np.argmin(own_scores))
        timings["own_search_seconds"] = time.perf_counter() - stage
        check_deadline(deadline)
        stage = time.perf_counter()
        cross_candidates, search_diagnostic = fit_cross(x_train, y_train, m_train, ridge.ALPHAS, deadline)
        cross_scores = [score_delta((x_stop @ beta).reshape(len(stop), h, c), y_stop, m_stop)[0]["MSE"]
                        for beta in cross_candidates]
        selected_cross = int(np.argmin(cross_scores))
        timings["cross_search_seconds"] = time.perf_counter() - stage
        del prepared, x_train, x_stop, cross_candidates, own_candidates, y_train, m_train
        check_deadline(deadline)
        mean60, std60, _ = data.fit_statistics(task, a60)
        prepared60 = ridge.PrefixFeatures(task, mean60, std60, bounds["V_select"][1])
        stage = time.perf_counter()
        own, _, refit_fallback = ridge.fit_coefficients(prepared60, refit, [ridge.ALPHAS[selected_own]])
        timings["own_refit_seconds"] = time.perf_counter() - stage
        x_refit = cross_features(prepared60, refit)
        y_refit, m_refit = targets(prepared60, refit)
        stage = time.perf_counter()
        cross, refit_diagnostic = fit_cross(x_refit, y_refit, m_refit, [ridge.ALPHAS[selected_cross]], deadline)
        timings["cross_refit_seconds"] = time.perf_counter() - stage
        del x_refit, y_refit, m_refit
        check_deadline(deadline)
        x_v = cross_features(prepared60, evaluation)
        y_v, m_v = targets(prepared60, evaluation)
        cross_delta = (x_v @ cross[0]).reshape(len(evaluation), h, c)
        own_delta = np.stack([prepared60.channel_features(channel, evaluation) @ own[0, channel] for channel in range(c)], axis=-1)
        statistic_rows = []
        for method, predicted, selected, scores in (("OWN_CHANNEL_32BLOCK_RIDGE", own_delta, selected_own, own_scores),
                                                     ("ALL_CHANNEL_32BLOCK_RIDGE", cross_delta, selected_cross, cross_scores)):
            score, sse, sae, counts = score_delta(predicted, y_v, m_v)
            methods[method] = dict(score, selected_alpha=ridge.ALPHAS[selected], A_stop_candidate_MSE=list(map(float, scores)),
                trainable_params=c*h*(33 if method.startswith("OWN") else p), point_statistic="deterministic conditional mean regression",
                C=c, K=None, L=task["L"], H=h, split="V_select", status="COMPLETE",
                input_info_scope="own channel past" if method.startswith("OWN") else "all cohort channels past",
                supervision_scope="A_fit search / A_stop alpha selection / fresh A60 refit", device="cpu", dtype="float64")
            for horizon in range(h):
                for channel in range(c):
                    statistic_rows.append(dict(method=method, horizon=horizon+1, channel_id=task["columns"][channel],
                        SSE=float(sse[horizon, channel]), SAE=float(sae[horizon, channel]), count=int(counts[horizon, channel])))
        diagnostics.update(cross_search=search_diagnostic, cross_refit=refit_diagnostic,
                           own_search_fallbacks=own_fallback, own_refit_fallbacks=refit_fallback)
    save_npz(private / "coefficients.npz", own=own[0], cross=cross[0], mean=mean60, std=std60,
             source_sha=np.asarray(sha(__file__)), scientific_identity=np.asarray(job["scientific_identity"]))
    anchors = np.stack([prepared60.filled[evaluation-1, channel] for channel in range(c)], axis=-1)
    # These real cohorts have no missing observations; assert to avoid a stale window anchor.
    if not prepared60.mask.all():
        anchors = np.stack([data.history_window(task, int(o), mean60, std60)[0][-1] for o in evaluation])
    paired_cells = np.where(m_v, (cross_delta-y_v)**2-(own_delta-y_v)**2, 0).sum(axis=1)
    paired_counts = m_v.sum(axis=1)
    paired_active = paired_counts > 0
    paired_channel_means = paired_cells / np.maximum(paired_counts, 1)
    paired_difference = np.sum(np.where(paired_active, paired_channel_means, 0), axis=1) / paired_active.sum(axis=1)
    save_npz(private / "prediction.npz", own_prediction=own_delta+anchors[:, None], cross_prediction=cross_delta+anchors[:, None],
             target=y_v+anchors[:, None], observed=m_v, origins=evaluation,
             paired_origin_MSE_difference=paired_difference)
    independently_rescored = {}
    with np.load(private / "prediction.npz", allow_pickle=False) as archive:
        for label, method in (("own", "OWN_CHANNEL_32BLOCK_RIDGE"), ("cross", "ALL_CHANNEL_32BLOCK_RIDGE")):
            error = archive[label+"_prediction"].astype(np.float64)-archive["target"].astype(np.float64)
            observed = archive["observed"]
            independent_sse = np.sum(np.where(observed, error*error, 0), axis=(0, 1), dtype=np.float64)
            independent_sae = np.sum(np.where(observed, np.abs(error), 0), axis=(0, 1), dtype=np.float64)
            independent_count = observed.sum(axis=(0, 1))
            active = independent_count > 0
            mse, mae = float(np.mean(independent_sse[active]/independent_count[active])), float(np.mean(independent_sae[active]/independent_count[active]))
            assert abs(mse-methods[method]["MSE"]) < 1e-12 and abs(mae-methods[method]["MAE"]) < 1e-12
            independently_rescored[label] = dict(MSE=mse, MAE=mae)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["method", "horizon", "channel_id", "SSE", "SAE", "count"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(statistic_rows)
    data_audit.atomic(destination / "ERROR_BY_HORIZON_CHANNEL.csv", buffer.getvalue())
    own_score, cross_score = methods["OWN_CHANNEL_32BLOCK_RIDGE"], methods["ALL_CHANNEL_32BLOCK_RIDGE"]
    result = dict(status="COMPLETE", scientific_identity=job["scientific_identity"], dataset_cohort=cohort,
        data_sha256=task["data_sha256"], scaler_sha256=ridge.scaler_fingerprint(mean60, std60),
        channel_order_sha256=data.array_sha256(np.asarray(task["columns"], dtype=str)),
        origins_sha256=data.array_sha256(evaluation), source_sha256=sha(__file__), access=access,
        methods=methods, preflight=preflight, diagnostics=diagnostics, timings=timings,
        independent_saved_FP64_score=independently_rescored,
        total_wall_seconds=time.perf_counter()-started, alpha_grid=list(ridge.ALPHAS),
        loss="mean_c observed SSE_c/N_c + alpha mean_c ||beta_c,nonintercept||^2",
        feature_definition="channel-major concat of32 contiguous block means minus that channel current; one intercept",
        same_A_fit_A_stop_A60_V_origins=True, selection_tie="first registered alpha, ascending order",
        training_origins=len(train), A_stop_origins=len(stop), refit_origins=len(refit), V_origins=len(evaluation),
        delta_MSE_pct=100*(cross_score["MSE"]-own_score["MSE"])/own_score["MSE"],
        delta_MAE_pct=100*(cross_score["MAE"]-own_score["MAE"])/own_score["MAE"],
        private_artifact_hashes=dict(coefficients=sha(private / "coefficients.npz"), prediction=sha(private / "prediction.npz")),
        private_artifacts=[dict(reference="cache://"+path.relative_to(Path(ctx["cache"])).as_posix(), sha256=sha(path))
                           for path in (private / "coefficients.npz", private / "prediction.npz")],
        sufficient_statistics_sha256=sha(destination / "ERROR_BY_HORIZON_CHANNEL.csv"),
        numeric_arrays_published=False, new_T_numeric_values_parsed=0, F_calls=0,
        interpretation="Limited fixed linear reader comparison on repeated V; no universal channel-information absence/presence claim")
    atomic_json(public_result, result)
    return dict(status="COMPLETE", reason=None,
                artifact_paths=[str(public_result), str(destination / "ERROR_BY_HORIZON_CHANNEL.csv")], result=result)


def synthetic_check():
    import numpy as np
    rng = np.random.default_rng(92602)
    x = np.column_stack([rng.normal(size=(40, 7)), np.ones(40)])
    y = rng.normal(size=(40, 3, 2))
    mask = rng.random(y.shape) > .2
    beta, diagnostic = fit_cross(x, y, mask, [.1])
    maximum = 0.
    penalty = np.eye(x.shape[1]); penalty[-1, -1] = 0
    for h in range(3):
        for c in range(2):
            keep, total = mask[:, h, c], int(mask[:, :, c].sum())
            augmented_x = np.concatenate([x[keep]/np.sqrt(total), np.sqrt(.1)*penalty])
            augmented_y = np.concatenate([y[keep, h, c]/np.sqrt(total), np.zeros(x.shape[1])])
            expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
            maximum = max(maximum, float(np.max(np.abs(expected-beta[0, :, h*2+c]))))
    assert maximum < 1e-10
    score, sse, sae, counts = score_delta((x@beta[0]).reshape(40, 3, 2), y, mask)
    assert abs(score["MSE"] - np.mean(sse.sum(0)/counts.sum(0))) < 1e-12
    assert abs(score["MAE"] - np.mean(sae.sum(0)/counts.sum(0))) < 1e-12
    _, data, ridge, _ = libraries(Path(__file__).resolve().parents[2])
    physical = rng.normal(size=(180, 1)).astype(np.float32)
    observed = np.ones_like(physical, bool)
    observed[15:80] = False
    physical[~observed] = np.nan
    toy = dict(physical=physical, observed=observed, L=64, H=4)
    mean, std, _ = data.fit_statistics(toy, 100)
    prepared = ridge.PrefixFeatures(toy, mean, std, 110)
    points = data.origins((0, 100), 64, 4, 1)
    own, _, _ = ridge.fit_coefficients(prepared, points, [.1])
    target, target_mask = targets(prepared, points)
    cross, _ = fit_cross(cross_features(prepared, points), target, target_mask, [.1])
    parity = float(np.max(np.abs(own[0, 0]-cross[0])))
    assert parity < 1e-12
    return dict(status="PASS", independent_lstsq_max_abs_difference=maximum,
                C1_own_cross_coefficient_max_abs_difference=parity, **diagnostic)


if __name__ == "__main__":
    import json
    print(json.dumps(dict(status="SOURCE_ONLY_NOOP", asset_reads=0, F_calls=0)))
