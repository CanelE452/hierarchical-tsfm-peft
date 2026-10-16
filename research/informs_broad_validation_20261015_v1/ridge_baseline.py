"""Own-channel raw-history ridge with chronological selection/refit; CPU only."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

import data_contract

ALPHAS = (.01, .1, 1., 10., 100.)
BLOCKS = 32


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scaler_fingerprint(mean, std):
    entries = {}
    for name, value in (("mean", mean), ("std", std)):
        x = np.ascontiguousarray(value, dtype=np.float32)
        entries[name] = dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())
    return fingerprint(entries)


def block_edges(length):
    if length < BLOCKS:
        raise ValueError("At least 32 history positions required")
    return np.linspace(0, int(length), BLOCKS+1, dtype=np.int64)


def temporal_features(history_normalized):
    x = np.asarray(history_normalized, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("Finite filled [batch,length,channels] history required")
    edges = block_edges(x.shape[1])
    anchor = x[:, -1]
    features = [x[:, edges[i]:edges[i+1]].mean(axis=1)-anchor for i in range(BLOCKS)]
    return np.stack(features+[np.ones_like(anchor)], axis=-1)


def predict_normalized(artifact, x_np):
    x = np.asarray(x_np)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    features = temporal_features(x)
    if coefficients.ndim != 3 or coefficients.shape[:2] != features.shape[1:]:
        raise ValueError("Ridge coefficient channel/feature axes differ")
    delta = np.einsum("bcp,cph->bhc", features, coefficients, optimize=True)
    return delta+x[:, -1:, :].astype(np.float64)


class PrefixFeatures:
    """O(T*C) preprocessing plus per-channel N*33 features, never N*L*C."""
    def __init__(self, task, mean, std, end):
        self.L = int(task["L"])
        self.H = int(task["H"])
        self.edges = block_edges(self.L)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.mask = np.asarray(task["observed"][:end], bool)
        physical = np.asarray(task["physical"][:end], np.float32)
        self.normalized = np.where(self.mask, (physical-self.mean)/self.std, 0).astype(np.float32)
        rows = np.arange(len(self.mask), dtype=np.int64)[:, None]
        previous = np.maximum.accumulate(np.where(self.mask, rows, -1), axis=0)
        channels = np.arange(self.mask.shape[1])
        filled = self.normalized[np.maximum(previous, 0), channels]
        self.filled = np.where(previous >= 0, filled, 0).astype(np.float32)
        self.prefix = np.concatenate((np.zeros((1, len(channels))), np.cumsum(self.filled, axis=0, dtype=np.float64)))
        self.next_observation = np.minimum.accumulate(np.where(self.mask, rows, len(self.mask))[::-1], axis=0)[::-1]

    def channel_features(self, channel, origins):
        origins = np.asarray(origins, dtype=np.int64)
        if np.any(origins < self.L) or np.any(origins > len(self.mask)):
            raise ValueError("Origin outside prepared permitted history")
        start = origins-self.L
        first = self.next_observation[start, channel]
        before = np.where(start > 0, self.filled[np.maximum(start-1, 0), channel], 0).astype(np.float64)
        anchor = np.where(first < origins, self.filled[origins-1, channel], 0).astype(np.float64)
        result = np.empty((len(origins), BLOCKS+1), np.float64)
        for index in range(BLOCKS):
            left, right = start+self.edges[index], start+self.edges[index+1]
            leading = np.clip(first-left, 0, right-left)
            total = self.prefix[right, channel]-self.prefix[left, channel]-before*leading
            result[:, index] = total/(right-left)-anchor
        result[:, -1] = 1.
        return result

    def channel_targets(self, channel, origins):
        origins = np.asarray(origins, np.int64)
        rows = origins[:, None]+np.arange(self.H)[None]
        mask = self.mask[rows, channel]
        anchor = np.where(self.next_observation[origins-self.L, channel] < origins,
                          self.filled[origins-1, channel], 0).astype(np.float64)
        return self.normalized[rows, channel].astype(np.float64)-anchor[:, None], mask


def fit_coefficients(prepared, origins, alphas):
    C, H = prepared.mask.shape[1], prepared.H
    result = np.zeros((len(alphas), C, BLOCKS+1, H), np.float64)
    counts = np.zeros((C, H), np.int64)
    fallback_count = 0
    regularizer = np.eye(BLOCKS+1)
    regularizer[-1, -1] = 0
    for channel in range(C):
        x = prepared.channel_features(channel, origins)
        target, mask = prepared.channel_targets(channel, origins)
        counts[channel] = mask.sum(axis=0)
        total = int(counts[channel].sum())
        if total == 0:
            continue
        rhs = x.T @ np.where(mask, target, 0) / total
        groups = {}
        for horizon in range(H):
            key = np.packbits(mask[:, horizon]).tobytes()
            groups.setdefault(key, []).append(horizon)
        for group in groups.values():
            observed = mask[:, group[0]]
            if not np.any(observed):
                continue
            usable = x if observed.all() else x[observed]
            gram = usable.T @ usable / total
            for a, alpha in enumerate(alphas):
                matrix = gram+float(alpha)*regularizer
                try:
                    coefficients = np.linalg.solve(matrix, rhs[:, group])
                except np.linalg.LinAlgError:
                    coefficients = np.linalg.lstsq(matrix, rhs[:, group], rcond=1e-12)[0]
                    fallback_count += 1
                result[a, channel][:, group] = coefficients
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite closed-form ridge coefficient")
    return result, counts, fallback_count


def stop_scores(prepared, origins, coefficients):
    scores = np.zeros(len(coefficients), np.float64)
    active = 0
    observed_total = 0
    for channel in range(prepared.mask.shape[1]):
        x = prepared.channel_features(channel, origins)
        target, mask = prepared.channel_targets(channel, origins)
        count = int(mask.sum())
        if not count:
            continue
        active += 1
        observed_total += count
        for index, coefficient in enumerate(coefficients):
            error = (x @ coefficient[channel]-target)[mask]
            scores[index] += np.dot(error, error)/count
    if not active:
        raise ValueError("No observed A_stop targets for ridge alpha selection")
    return scores/active, active, observed_total


def load_artifact(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def fit_task(task, cache, public):
    started_unix = time.time()
    started = time.perf_counter()
    cpu_started = psutil.Process().cpu_times()
    cache, public = Path(cache), Path(public)
    run_id = "primary_"+task["name"]+"_E1_RAW_RIDGE_k0_s92601"
    run_dir = public / "runs" / run_id
    receipt_path = run_dir / "RIDGE_FIT_RECEIPT.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve complete ridge fits; validate artifact explicitly before reuse")
    source = {"ridge_baseline.py": data_contract.sha256_file(__file__),
              "data_contract.py": data_contract.sha256_file(data_contract.__file__)}
    source_sha = fingerprint(source)
    bounds = data_contract.split_ranges(task)
    search_end, refit_end = bounds["A_fit"][1], bounds["A_stop"][1]
    mean, std, _ = data_contract.fit_statistics(task, search_end)
    train = data_contract.origins((0, search_end), task["L"], task["H"], 1)
    stops = data_contract.origins(bounds["A_stop"], task["L"], task["H"], task["H"])
    refit_origins = data_contract.origins((0, refit_end), task["L"], task["H"], 1)
    if not len(train) or not len(stops) or not len(refit_origins):
        raise ValueError("Ridge chronological fitting requires nonempty A_fit/A_stop origins")
    config = dict(family="RAW_RIDGE", L=int(task["L"]), H=int(task["H"]), blocks=BLOCKS,
                  features="32 nonoverlap temporal block means minus current + intercept",
                  alphas=list(ALPHAS), penalty="alpha*L2(nonintercept coefficients)",
                  loss="per-channel observed target SSE / channel total observed targets; channel macro mean",
                  selection="minimum H-stride A_stop observed channel macro MSE; tie lower alpha",
                  scaler_search_end=search_end, scaler_refit_end=refit_end, origin_stride=1,
                  target_fill="missing excluded by observed mask", history_fill="window-local reset to training mean",
                  BLAS_threads=2, fit_algorithm="float64 masked closed form", seed=92601)
    contract = dict(data_sha=task["data_sha256"], source_sha=source_sha, config_sha=fingerprint(config),
                    model_sha="NO_FOUNDATION", L=int(task["L"]), H=int(task["H"]))
    with threadpool_limits(limits=2, user_api="blas"):
        search_started = time.perf_counter()
        prepared = PrefixFeatures(task, mean, std, refit_end)
        candidates, search_counts, search_fallbacks = fit_coefficients(prepared, train, ALPHAS)
        scores, active, count = stop_scores(prepared, stops, candidates)
        selected = int(np.argmin(scores))
        search_seconds = time.perf_counter()-search_started
        refit_started = time.perf_counter()
        mean60, std60, _ = data_contract.fit_statistics(task, refit_end)
        refit = PrefixFeatures(task, mean60, std60, refit_end)
        final, refit_counts, refit_fallbacks = fit_coefficients(refit, refit_origins, [ALPHAS[selected]])
        refit_seconds = time.perf_counter()-refit_started
    scaler_sha = scaler_fingerprint(mean60, std60)
    feature_sha = fingerprint(dict(source_sha=source_sha, block_edges=block_edges(task["L"]).tolist(),
                                   reset="window-local Amean zero", anchor="current"))
    predictor_sha = fingerprint(dict(coefficients=data_contract.array_sha256(final[0]), scaler_sha=scaler_sha,
                                     feature_sha=feature_sha, family="RAW_RIDGE"))
    contract.update(scaler_sha=scaler_sha, feature_sha=feature_sha, model_sha=predictor_sha,
                    foundation_sha="NO_FOUNDATION")
    artifact = dict(coefficients=final[0], mean=mean60, std=std60, block_edges=block_edges(task["L"]),
                    alpha=np.asarray(ALPHAS[selected], np.float64), canonical_A60_end=np.asarray(refit_end, np.int64),
                    actual_fit_end=np.asarray(refit_end, np.int64), source_sha=np.asarray(source_sha),
                    data_sha=np.asarray(task["data_sha256"]), feature_sha=np.asarray(feature_sha),
                    scaler_sha=np.asarray(scaler_sha), model_sha=np.asarray(predictor_sha),
                    L=np.asarray(task["L"], np.int64), H=np.asarray(task["H"], np.int64),
                    contract_json=np.asarray(json.dumps(contract, sort_keys=True)))
    artifact_path = cache / "ridge" / run_id / "artifact.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        np.savez_compressed(stream, **artifact)
    cpu_finished = psutil.Process().cpu_times()
    receipt = dict(status="COMPLETE_VALID", run_id=run_id, family="RAW_RIDGE", dataset=task["name"],
                   fold="primary", seed=92601, role="base_parent", GPU_calls=0, F_streams=0,
                   source_files=source, source_sha=source_sha, config=config, config_sha=contract["config_sha"],
                   data_sha=task["data_sha256"], model_sha=contract["model_sha"], foundation_sha="NO_FOUNDATION", scaler_sha=scaler_sha,
                   feature_sha=feature_sha, selected_alpha=ALPHAS[selected], alpha_candidates=list(ALPHAS),
                   A_stop_candidate_MSE=[float(x) for x in scores], A_stop_active_channels=active,
                   A_stop_observed_targets=count, A_stop_origins=len(stops), selection_target_last_exclusive=refit_end,
                   train_origins_search=len(train), train_origins_refit=len(refit_origins),
                   search_observed_targets=int(search_counts.sum()), refit_observed_targets=int(refit_counts.sum()),
                   fit_end=refit_end, search_fit_end=search_end, training_status="CLOSED_FORM_ALL_LEGAL_A_ORIGINS",
                   actual_optimizer_updates=0, effective_epochs="closed_form_all_origins",
                   exposure_comparison="all legal A origins; differs from neural 0.25-epoch screen exposure",
                   actual_parameter_count=int(final[0].size), coefficient_dtype="float64", search_seconds=search_seconds,
                   refit_seconds=refit_seconds, wall_seconds=time.perf_counter()-started,
                   CPU_process_seconds=cpu_finished.user+cpu_finished.system-cpu_started.user-cpu_started.system,
                   numerical_fallbacks=search_fallbacks+refit_fallbacks, started_unix=started_unix,
                   alpha_grid_endpoint_best=selected in (0, len(ALPHAS)-1),
                   alpha_search_limit="registered five candidates only; endpoint best does not establish global optimum",
                   finished_unix=time.time(), T_final_target_access=False, M_cal_target_access=False,
                   V_select_target_access=False, artifact_cache_relative_path=str(artifact_path.relative_to(cache)).replace("\\", "/"),
                   artifact_sha256=data_contract.sha256_file(artifact_path), constant_channels_removed=0,
                   target_score_scope="A_stop only; M/V/T prediction and scoring delegated to locked evaluation")
    _write_new(receipt_path, receipt)
    return receipt, artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--public", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--datasets", nargs="+", choices=["beijing", "jena", "electricity", "pems08"],
                        default=["beijing", "jena", "electricity", "pems08"])
    args = parser.parse_args()
    for name in args.datasets:
        task = data_contract.load_task(name, cache_dir=args.cache)
        receipt, _ = fit_task(task, args.cache, args.public)
        print(json.dumps(dict(run_id=receipt["run_id"], status=receipt["status"],
                              selected_alpha=receipt["selected_alpha"], search_seconds=receipt["search_seconds"],
                              refit_seconds=receipt["refit_seconds"], params=receipt["actual_parameter_count"])), flush=True)
