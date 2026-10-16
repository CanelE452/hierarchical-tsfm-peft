"""Time-out-of-parent-sample calibration using fixed cached forecasts.

This module never loads or calls a foundation model. Its training cost is the
additional head fit only; deployment cost must include both frozen parents.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from data_contract import origins, sha256_file
from evaluation import array_fingerprint, error_statistics, fit_global_alpha
from networks import ConditionalHead, DirectG, module_state_hash, parameter_count
from training import FastWindows, PermutationStream, cpu_state, fingerprint, macro_loss, verify_contract


def scaler_fingerprint(mean, std):
    return fingerprint({"mean": array_fingerprint(np.asarray(mean, np.float32)),
                        "std": array_fingerprint(np.asarray(std, np.float32))})


def macro_metrics(prediction, target, observed):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    mask = np.asarray(observed)
    if prediction.ndim != 3 or prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError("Matching [origin,H,C] arrays and original bool mask required")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Observed prediction/target must be finite")
    counts = mask.sum((0, 1), dtype=np.int64)
    sse, sae = [], []
    for channel in np.flatnonzero(counts):
        delta = prediction[:, :, channel][mask[:, :, channel]] - target[:, :, channel][mask[:, :, channel]]
        sse.append(float(np.dot(delta, delta)) / int(counts[channel]))
        sae.append(float(np.abs(delta).sum(dtype=np.float64)) / int(counts[channel]))
    return {"mse": math.fsum(sse) / len(sse) if sse else None,
            "mae": math.fsum(sae) / len(sae) if sae else None,
            "n_observed": int(counts.sum()), "active_channels": len(sse),
            "n_origins": len(prediction), "horizon": prediction.shape[1], "num_channels": prediction.shape[2]}


def fit_alpha_family(g, b, target, observed):
    fit = fit_global_alpha(g, b, target, observed)
    g, b = np.asarray(g, np.float64), np.asarray(b, np.float64)
    result = {"alpha_fit": fit["alpha"], "fit": fit}
    for label, value in (("alpha0", 0.), ("alpha05", .5), ("alpha1", 1.), ("fitted", fit["alpha"])):
        result[label] = {"alpha": value, **macro_metrics(g + value * (b - g), target, observed)}
    return result


def cache_fingerprint(cache):
    names = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")
    metadata = {key: value for key, value in cache["metadata"].items() if key != "reused_prediction_cache"}
    return fingerprint({"arrays": {name: array_fingerprint(cache[name]) for name in names}, "metadata": metadata})


def expected_contract(spec, task, parent_m, parent_v, g_m, g_v, mean, std):
    parents = {name: cache_fingerprint(value) for name, value in
               (("parent_m", parent_m), ("parent_v", parent_v), ("g_m", g_m), ("g_v", g_v))}
    source_sha = fingerprint({name: sha256_file(Path(__file__).with_name(name)) for name in
                              ("combination.py", "networks.py", "training.py", "evaluation.py", "data_contract.py")})
    head_class = "LegacyResidualHead" if spec["variant"] == "legacy_residual" else "ConditionalHead"
    scientific_spec = {key: value for key, value in spec.items() if key not in {"run_id", "resume_checkpoint"}}
    return {"source_sha": source_sha, "config_sha": fingerprint(scientific_spec), "data_sha": task["data_sha256"],
            "scaler_sha": scaler_fingerprint(mean, std), "parents_sha": fingerprint(parents),
            "model_sha": fingerprint({"class": head_class, "L": task["L"], "H": task["H"], "hidden": 32})}


def validate_completed_receipt(receipt, spec, task, parent_m, parent_v, g_m, g_v, mean, std, cache_root):
    validate_parent_caches(task, parent_m, parent_v, g_m, g_v, mean, std)
    if receipt.get("run_id") != spec["run_id"] or receipt.get("execution_status") != "COMPLETE_VALID":
        raise ValueError("Only the same complete combination run can be reused")
    expected = expected_contract(spec, task, parent_m, parent_v, g_m, g_v, mean, std)
    verify_contract(expected, receipt["contract"])
    if receipt["contract"].get("parents_sha") != expected["parents_sha"]:
        raise ValueError("Parent forecast contract changed")
    if receipt.get("variant") != spec["variant"]:
        raise ValueError("Combination variant changed")
    private = Path(cache_root) / "combinations" / spec["run_id"]
    if json.loads((private / "spec.json").read_text(encoding="utf-8")) != spec:
        raise ValueError("Combination specification changed")
    labels = {"alpha0", "alpha05", "alpha1", "fitted"} if spec["variant"] == "alpha_family" else {"head"}
    if set(receipt["prediction_artifacts"]) != labels:
        raise ValueError("Completed combination prediction artifact family is incomplete")
    for outputs in receipt["prediction_artifacts"].values():
        if set(outputs) != {"M_cal", "V_select"}:
            raise ValueError("Completed combination prediction artifact splits are incomplete")
        for item in outputs.values():
            path = private / item["file"]
            if path.parent.resolve() != private.resolve() or sha256_file(path) != item["sha256"]:
                raise ValueError("Completed combination prediction artifact changed")
    if spec["variant"] != "alpha_family" and sha256_file(private / "selected.pt") != receipt["selected_checkpoint_sha256"]:
        raise ValueError("Completed combination selected model changed")
    return receipt


def _validate_cache(task, cache, split, mean, std):
    for name in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids", "metadata"):
        if name not in cache:
            raise ValueError("Missing parent forecast field: " + name)
    oo = np.asarray(cache["origins"])
    stride = 1 if split == "M_cal" else task["H"]
    expected_origins = origins(task["split_ranges"][split], task["L"], task["H"], stride)
    if not np.array_equal(oo, expected_origins):
        raise ValueError("Parent must cover the exact registered dense-M or stride-H V cohort")
    shape = (len(oo), task["H"], len(task["columns"]))
    if np.asarray(cache["pred_normalized"]).shape != shape or np.asarray(cache["target_normalized"]).shape != shape:
        raise ValueError("Parent original-channel forecast shape mismatch")
    if np.asarray(cache["observed"]).shape != shape or np.asarray(cache["observed"]).dtype != np.bool_:
        raise ValueError("Parent observation mask must be bool and match targets")
    if not np.array_equal(np.asarray(cache["channel_ids"]).astype(str), np.asarray(task["columns"]).astype(str)):
        raise ValueError("Parent channel order mismatch")
    meta = cache["metadata"]
    if meta.get("data_sha256") != task["data_sha256"] or meta.get("scaler_sha") != scaler_fingerprint(mean, std):
        raise ValueError("Parent data/scaler fingerprint mismatch")
    if not meta.get("model_sha") or not meta.get("cohort") or meta.get("split") != split:
        raise ValueError("Parent model/cohort/split metadata missing or inconsistent")
    if not isinstance(meta.get("parent_fit_end"), int) or not 0 < meta["parent_fit_end"] <= task["split_ranges"]["M_cal"][0]:
        raise ValueError("Calibration parents must have frozen before M_cal")
    if not np.isfinite(cache["pred_normalized"]).all():
        raise ValueError("Parent forecast includes nonfinite values")
    for lo in range(0, len(oo), 128):
        rows = oo[lo:lo + 128, None] + np.arange(task["H"])[None]
        observed = task["observed"][rows]
        y = np.where(observed, (task["physical"][rows] - mean) / std, 0).astype(np.float32)
        if not np.array_equal(cache["observed"][lo:lo + 128], observed):
            raise ValueError("Parent target mask differs from actual task")
        if not np.array_equal(np.asarray(cache["target_normalized"])[lo:lo + 128], y):
            raise ValueError("Parent targets differ from actual A-normalized task targets")


def validate_parent_caches(task, parent_m, parent_v, g_m, g_v, mean, std):
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if mean.shape != (len(task["columns"]),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Canonical A60 mean/std shape or values invalid")
    for split, parent, g in (("M_cal", parent_m, g_m), ("V_select", parent_v, g_v)):
        _validate_cache(task, parent, split, mean, std)
        _validate_cache(task, g, split, mean, std)
        if parent["metadata"]["cohort"] != g["metadata"]["cohort"]:
            raise ValueError("Parent and G cohort IDs disagree")
        for name in ("origins", "target_normalized", "observed", "channel_ids"):
            if not np.array_equal(parent[name], g[name]):
                raise ValueError("Parent and G differ on aligned field " + name)
    for first, second in ((parent_m, parent_v), (g_m, g_v)):
        if first["metadata"]["model_sha"] != second["metadata"]["model_sha"]:
            raise ValueError("M and V predictions must use identical frozen parent weights")


def legacy_residual_prediction(b, g, current):
    return np.asarray(b) + (np.asarray(g) - np.asarray(current))


class LegacyResidualHead(DirectG):
    """New M_cal-fitted DirectG innovation added to frozen b; CP0 equals b."""
    def forward(self, x, g, b, control=False):
        return b.detach() + (super().forward(x) - x[:, -1:])


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _torch_save(path, bundle):
    with Path(path).open("xb") as stream:
        torch.save(bundle, stream)


def _private_forecast(path, prediction, parent):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, pred_normalized=np.asarray(prediction, np.float32),
                            target_normalized=parent["target_normalized"], observed=parent["observed"],
                            origins=parent["origins"], channel_ids=parent["channel_ids"])
    return {"file": Path(path).name, "sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
            "arrays": {name: array_fingerprint(value) for name, value in {"pred_normalized": np.asarray(prediction, np.float32),
                       "target_normalized": parent["target_normalized"], "observed": parent["observed"],
                       "origins": parent["origins"], "channel_ids": parent["channel_ids"]}.items()},
            "raw_target_publication": False}


def _public_stats(path, prediction, parent, run_id, split):
    score, rows = error_statistics(np.asarray(prediction, np.float32), parent["target_normalized"], parent["observed"],
                                   parent["origins"], run_id, split, parent["metadata"]["cohort"], parent["channel_ids"])
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count"))
        writer.writeheader()
        writer.writerows(rows)
    return score


def predict_head(head, task, parent, g, mean, std, *, device="cuda", control=None, batch_size=4, deadline=None):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Budget reached before head prediction")
    head.eval()
    control = bool(getattr(head, "default_control", False)) if control is None else bool(control)
    prefix_end = int(parent["origins"][-1]) + task["H"]
    prefix = dict(task, physical=task["physical"][:prefix_end], observed=task["observed"][:prefix_end])
    windows = FastWindows(prefix, np.asarray(mean, np.float32), np.asarray(std, np.float32), device)
    results = []
    with torch.no_grad():
        for lo in range(0, len(parent["origins"]), batch_size):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Budget reached during head prediction")
            x, _, _ = windows.get(parent["origins"][lo:lo + batch_size])
            gp = torch.as_tensor(g["pred_normalized"][lo:lo + batch_size], dtype=torch.float32, device=device)
            bp = torch.as_tensor(parent["pred_normalized"][lo:lo + batch_size], dtype=torch.float32, device=device)
            results.append(head(x, gp, bp, control=control).cpu().numpy())
    return np.concatenate(results)


def load_conditional_checkpoint(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["head_config"]
    cls = LegacyResidualHead if config.get("kind") == "legacy_residual" else ConditionalHead
    head = cls(config["length"], config["horizon"], config["hidden"])
    head.load_state_dict(saved["model"], strict=True)
    head.default_control = bool(saved["control"])
    head.combination_variant = config.get("kind", "conditional")
    return head.to(device)


def fit_combination(spec, task, parent_m, parent_v, g_m, g_v, mean, std, cache_root, public_root, deadline):
    """Fit one registered alpha family, conditional H, past H or old residual.

    Dense M_cal parents are mandatory. V forecasts use exactly H stride. The
    caller owns real-data execution and must include parent pipelines in cost.
    Existing run directories are refused. Exact resumes use a new run ID plus
    spec.resume_checkpoint pointing at the immutable full optimizer checkpoint.
    """
    started = time.time()
    run_id, variant = str(spec["run_id"]), str(spec["variant"])
    if not run_id or Path(run_id).name != run_id or variant not in {"alpha_family", "conditional", "past_control", "legacy_residual"}:
        raise ValueError("Safe run ID and registered combination variant required")
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    validate_parent_caches(task, parent_m, parent_v, g_m, g_v, mean, std)
    cache_before = {name: cache_fingerprint(value) for name, value in (("parent_m", parent_m), ("parent_v", parent_v), ("g_m", g_m), ("g_v", g_v))}
    private = Path(cache_root) / "combinations" / run_id
    public = Path(public_root) / run_id
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    contract = expected_contract(spec, task, parent_m, parent_v, g_m, g_v, mean, std)
    _json(private / "spec.json", spec)
    score_rows, outputs, predictions = [], {}, {}
    receipt = {"run_id": run_id, "dataset": task.get("name", task.get("dataset")), "variant": variant,
               "contract": contract, "parent_cache_fingerprints": cache_before,
               "parent_models": {name: value["metadata"] for name, value in (("parent_m", parent_m), ("parent_v", parent_v), ("g_m", g_m), ("g_v", g_v))},
               "started_unix": started, "fit_split": "M_cal_dense_stride1", "checkpoint_split": "V_select_strideH",
               "scoring_scale": "Original channels normalized with common A_fit_plus_A_stop statistics",
               "foundation_calls_during_H_fit": 0, "deployment_requires_frozen_parent_pipeline_cost": True,
               "deployment_frozen_parent_requirements": ["g"] if variant == "past_control" else (["b"] if variant == "legacy_residual" else ["g", "b"]),
               "private_cache_reference": f"combinations/{run_id}", "updates": 0, "exposures": 0,
               "training_status": "NO_TRAINING", "execution_status": "COMPLETE_VALID"}
    if variant == "alpha_family":
        fitted = fit_alpha_family(g_m["pred_normalized"], parent_m["pred_normalized"], parent_m["target_normalized"], parent_m["observed"])
        receipt["alpha_fit"] = fitted
        receipt["alpha_deployment_requirements"] = {"alpha0": ["g"], "alpha05": ["g", "b"], "alpha1": ["b"], "fitted": ["g", "b"]}
        for label in ("alpha0", "alpha05", "alpha1", "fitted"):
            alpha = fitted[label]["alpha"]
            predictions[label] = {split: (np.asarray(g["pred_normalized"], np.float64) + alpha *
                                          (np.asarray(parent["pred_normalized"], np.float64) - g["pred_normalized"])).astype(np.float32)
                                  for split, parent, g in (("M_cal", parent_m, g_m), ("V_select", parent_v, g_v))}
    else:
        device, seed = str(spec.get("device", "cuda")), int(spec.get("seed", 92601))
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        cls = LegacyResidualHead if variant == "legacy_residual" else ConditionalHead
        head = cls(task["L"], task["H"], 32).to(device)
        initial = cpu_state(head)
        initial_hash = module_state_hash(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(spec.get("learning_rate", .001)), weight_decay=0.)
        sampler = PermutationStream(parent_m["origins"], seed)
        prefix_end = task["split_ranges"]["M_cal"][1]
        prefix = dict(task, physical=task["physical"][:prefix_end], observed=task["observed"][:prefix_end])
        windows = FastWindows(prefix, mean, std, device)
        counts = windows.target_counts(parent_m["origins"])
        index = {int(value): i for i, value in enumerate(parent_m["origins"])}
        effective, micro = int(spec.get("effective_batch", 16)), int(spec.get("microbatch", 4))
        endpoints = [float(value) for value in spec.get("epoch_endpoints", [.25, 1, 3])]
        if min(effective, micro) < 1 or micro > effective or endpoints != sorted(set(endpoints)) or min(endpoints) <= 0:
            raise ValueError("Invalid registered head training schedule")
        control = variant == "past_control"
        curves, best_state, best_mse, best_updates = [], initial, None, 0
        clipped, norms, updates = 0, [], 0
        frequencies = np.zeros(len(parent_m["origins"]), np.int64)
        optimization_seconds = 0.
        interrupted_endpoint, interrupted_losses = None, []
        resumed = spec.get("resume_checkpoint")
        if resumed:
            saved = torch.load(resumed, map_location="cpu", weights_only=True)
            verify_contract(contract, saved["contract"])
            if saved["contract"]["parents_sha"] != contract["parents_sha"] or saved["device"] != device:
                raise ValueError("Resume parent forecasts or device changed")
            head.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            sampler.load_state_dict(saved["sampler"])
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            updates, best_state, best_mse, best_updates = saved["updates"], saved["best_model"], saved["best_V_mse"], saved["best_updates"]
            curves, clipped, norms = list(saved["curves"]), saved["clipped"], list(saved["gradient_norms"])
            frequencies = np.asarray(saved["origin_frequencies"], np.int64)
            optimization_seconds = saved["optimization_seconds"]
            interrupted_endpoint = saved.get("pending_endpoint")
            interrupted_losses = list(saved.get("pending_batch_losses", []))
            if int(frequencies.sum()) != sampler.exposures:
                raise ValueError("Resume origin exposure accounting disagrees with sampler")
            receipt["exact_resume_from_checkpoint_sha256"] = sha256_file(resumed)
            receipt["resumed_from_updates"], receipt["resumed_from_exposures"] = updates, sampler.exposures
        def save_checkpoint(pending_endpoint, pending_losses):
            checkpoint = {"model": cpu_state(head), "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
                          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                          "contract": contract, "updates": updates, "head_config": {"length": task["L"], "horizon": task["H"], "hidden": 32, "kind": variant},
                          "control": control, "best_model": best_state, "best_updates": best_updates, "best_V_mse": best_mse,
                          "device": device, "curves": list(curves), "clipped": clipped, "gradient_norms": list(norms),
                          "origin_frequencies": frequencies.tolist(), "optimization_seconds": optimization_seconds,
                          "pending_endpoint": pending_endpoint, "pending_batch_losses": list(pending_losses)}
            sequence = len(list(private.glob("checkpoint_*.pt")))
            checkpoint_name = f"checkpoint_{sequence:06d}_e{sampler.exposures}_u{updates}.pt"
            _torch_save(private / checkpoint_name, checkpoint)
            receipt["last_full_checkpoint_reference"] = f"combinations/{run_id}/{checkpoint_name}"
            receipt["last_full_checkpoint_sha256"] = sha256_file(private / checkpoint_name)

        # Exposure alone cannot prove that endpoint's V evaluation completed.
        # In particular an expired CP0 or an exact-endpoint pause remains due.
        completed = {float(row["endpoint_epochs"]) for row in curves}
        pending = [epoch for epoch in [0.] + endpoints if epoch not in completed]
        if not pending:
            # A prior segment may have completed training but exhausted its
            # budget while exporting forecasts. Keep a new immutable resume
            # handle before retrying that export without another update.
            save_checkpoint(None, [])
        for endpoint in pending:
            batch_losses = list(interrupted_losses) if endpoint == interrupted_endpoint else []
            target_exposures = int(math.ceil(endpoint * len(parent_m["origins"])))
            segment_started = time.perf_counter()
            while sampler.exposures < target_exposures:
                if deadline is not None and time.time() >= deadline:
                    receipt["execution_status"] = "BUDGET_LIMIT"
                    break
                batch = sampler.draw(min(effective, target_exposures - sampler.exposures))
                np.add.at(frequencies, [index[int(o)] for o in batch], 1)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.
                head.train()
                for lo in range(0, len(batch), micro):
                    oo = batch[lo:lo + micro]
                    x, target, mask = windows.get(oo)
                    rows = [index[int(o)] for o in oo]
                    gp = torch.as_tensor(g_m["pred_normalized"][rows], device=device, dtype=torch.float32)
                    bp = torch.as_tensor(parent_m["pred_normalized"][rows], device=device, dtype=torch.float32)
                    pred = head(x, gp, bp, control=control)
                    loss = macro_loss(pred, target, mask, counts, len(parent_m["origins"]), len(batch))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite M_cal original-future head loss")
                    loss.backward()
                    batch_loss += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite head gradient")
                norms.append(float(norm)); clipped += int(norm > 1)
                optimizer.step(); updates += 1; batch_losses.append(batch_loss)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            optimization_seconds += time.perf_counter() - segment_started
            if receipt["execution_status"] == "BUDGET_LIMIT" or (deadline is not None and time.time() >= deadline):
                receipt["execution_status"] = "BUDGET_LIMIT"
                save_checkpoint(endpoint, batch_losses)
                break
            try:
                vp = predict_head(head, task, parent_v, g_v, mean, std, device=device, control=control, batch_size=micro, deadline=deadline)
            except TimeoutError:
                receipt["execution_status"] = "BUDGET_LIMIT"
                save_checkpoint(endpoint, batch_losses)
                break
            score = macro_metrics(vp, parent_v["target_normalized"], parent_v["observed"])
            if score["mse"] is None:
                raise ValueError("No V observed target for checkpoint selection")
            if best_mse is None or score["mse"] < best_mse:
                best_mse, best_state, best_updates = score["mse"], cpu_state(head), updates
            curve = {"updates": updates, "epoch": sampler.exposures / len(parent_m["origins"]), "exposures": sampler.exposures,
                     "endpoint_epochs": endpoint, "V_mse": score["mse"], "V_mae": score["mae"],
                     "M_batch_loss": float(np.mean(batch_losses)) if batch_losses else None,
                     "clipped_steps_cumulative": clipped, "gradient_norm_max_cumulative": max(norms) if norms else None,
                     "elapsed_seconds": time.time() - started}
            curves.append(curve)
            save_checkpoint(None, [])
        with (public / "curve.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = ("updates", "epoch", "exposures", "endpoint_epochs", "V_mse", "V_mae", "M_batch_loss",
                      "clipped_steps_cumulative", "gradient_norm_max_cumulative", "elapsed_seconds")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(curves)
        last_hash = module_state_hash(head)
        selected_hash, selected_checkpoint_sha = None, None
        if receipt["execution_status"] == "COMPLETE_VALID":
            head.load_state_dict(best_state)
            selected_hash = module_state_hash(head)
            selected = {"model": cpu_state(head), "head_config": {"length": task["L"], "horizon": task["H"], "hidden": 32, "kind": variant},
                        "control": control, "contract": contract, "selected_updates": best_updates, "selected_V_mse": best_mse}
            _torch_save(private / "selected.pt", selected)
            selected_checkpoint_sha = sha256_file(private / "selected.pt")
            try:
                for split, parent, g in (("M_cal", parent_m, g_m), ("V_select", parent_v, g_v)):
                    predictions.setdefault("head", {})[split] = predict_head(head, task, parent, g, mean, std, device=device, control=control, batch_size=micro, deadline=deadline)
            except TimeoutError:
                receipt["execution_status"] = "BUDGET_LIMIT"
                predictions = {}
        if receipt["execution_status"] == "BUDGET_LIMIT":
            receipt["inference_or_evaluation_pending"] = True
        target_diff = np.zeros(prefix_end + 1, np.int64)
        np.add.at(target_diff, parent_m["origins"], frequencies)
        np.add.at(target_diff, parent_m["origins"] + task["H"], -frequencies)
        target_repetitions = target_diff.cumsum()[:-1]
        exposure = {"unique_origins_actually_seen": int((frequencies > 0).sum()),
                    "unique_target_times_actually_seen": int((target_repetitions > 0).sum()),
                    "target_time_max_repetition": int(target_repetitions.max(initial=0)),
                    "observed_targets_with_repetition_by_channel": (target_repetitions[:, None] * prefix["observed"]).sum(0).tolist(),
                    "observed_unique_targets_by_channel": prefix["observed"][target_repetitions > 0].sum(0).tolist(),
                    "origin_frequency_sha256": array_fingerprint(frequencies)}
        receipt.update({"seed": seed, "learning_rate": float(spec.get("learning_rate", .001)), "effective_batch": effective, "microbatch": micro,
                        "registered_epoch_endpoints": endpoints, "actual_epochs": sampler.exposures / len(parent_m["origins"]),
                        "updates": updates, "exposures": sampler.exposures, "unique_M_origins": len(parent_m["origins"]),
                        "selected_updates": best_updates, "selected_V_mse": best_mse, "initial_state_sha": initial_hash,
                        "last_state_sha": last_hash, "selected_state_sha": selected_hash,
                        "head_parameter_count": parameter_count(head), "head_input_width": task["L"] if variant == "legacy_residual" else task["L"] + 2 * task["H"],
                        "control_replacement": "New DirectG on past-current, fitted to Y-b" if variant == "legacy_residual" else
                                               ("Actual recent H current-centered history; all same-capacity parameters live" if control else "Frozen b-g forecast difference"),
                        "initial_prediction_anchor": "Frozen b" if variant == "legacy_residual" else "Frozen g",
                        "legacy_residual_new_fit": variant == "legacy_residual", "fixed_G_used_at_deployment": variant != "legacy_residual",
                        "selected_checkpoint_sha256": selected_checkpoint_sha,
                        "training_status": "UNDERTRAINED_OR_BUDGET_LIMIT" if receipt["execution_status"] == "BUDGET_LIMIT" else
                                           ("UNDERTRAINED_LAST_CHECKPOINT_BEST" if best_updates == updates and updates else "SCREENED_NOT_CONVERGENCE_PROVEN"),
                        "clipped_steps": clipped, "parent_forecast_gradients": False,
                        "exposure": exposure, "cumulative_optimization_seconds_including_prior_resume_segments": optimization_seconds,
                        "selected_parameter_max_abs_change": max(float((best_state[name] - value).abs().max()) for name, value in initial.items()) if best_mse is not None else None,
                        "head_training_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
                        "head_training_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0,
                        "head_fit_peak_scope": "Current process peak including any caller-resident models; not fresh-process deployment memory"})
    for label, by_split in predictions.items():
        outputs[label] = {}
        for split, parent in (("M_cal", parent_m), ("V_select", parent_v)):
            name = f"{label}_{split}.npz" if variant == "alpha_family" else f"{split}.npz"
            outputs[label][split] = _private_forecast(private / name, by_split[split], parent)
            run_label = f"{run_id}__{label}" if variant == "alpha_family" else run_id
            if split == "V_select":
                metrics = _public_stats(public / f"{label}_{split}_error_stats.csv.gz", by_split[split], parent, run_label, split)
            else:
                metrics = {"run_id": run_label, "split": split, "cohort": parent["metadata"]["cohort"],
                           **macro_metrics(by_split[split], parent["target_normalized"], parent["observed"])}
            score_rows.append(metrics)
    with (public / "scores.csv").open("x", encoding="utf-8", newline="") as stream:
        keys = sorted(set().union(*(row.keys() for row in score_rows))) if score_rows else ["run_id", "split", "cohort", "mse", "mae"]
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(score_rows)
    cache_after = {name: cache_fingerprint(value) for name, value in (("parent_m", parent_m), ("parent_v", parent_v), ("g_m", g_m), ("g_v", g_v))}
    if cache_after != cache_before:
        raise AssertionError("Frozen parent forecast arrays or metadata changed")
    receipt.update({"parents_unchanged": True, "prediction_artifacts": outputs, "scores": score_rows,
                    "completed_unix": time.time(), "total_elapsed_seconds": time.time() - started,
                    "T_final_accessed": False, "independent_training_rerun": False})
    _json(public / "receipt.json", receipt)
    return receipt
