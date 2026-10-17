"""Exclusive physical-input Jena E03 costs. Import never loads models or assets."""

import csv
import gc
import importlib
import os
import platform
from pathlib import Path
import threading
import time
import traceback
import uuid

import numpy as np
import psutil
import torch

from .core import atomic_json, read, sha
from . import existing_jobs as existing


METHODS = ("Bolt_Raw", "Bolt_B0", "Bolt_B2", "Ridge", "Moirai_Raw",
           "Moirai_B0", "Moirai_B2", "Chronos2_nativeRaw")
SCOPE = "physical history/mask -> within-window causal fill -> FP32 normalization -> F or P/E/F/D or Ridge -> FP64 inverse normalization -> CPU return"


class ExclusiveCostError(RuntimeError):
    pass


def _quiet(ctx):
    ancestors = {p.pid for p in psutil.Process().parents()} | {os.getpid()}
    other = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in ancestors:
            continue
        command = " ".join(process.info.get("cmdline") or [])
        if "python" in (process.info["name"] or "").lower() and (
                "adapter_diagnostics_20261017_v2" in command or "mechanism_runner.py" in command):
            other.append(process.pid)
    if other:
        raise ExclusiveCostError("exclusive E03 has another campaign worker")
    if time.time() >= ctx.get("job_deadline", ctx["deadline"]):
        raise TimeoutError("E03 reached finite deadline")
    if ctx.get("lease"):
        try:
            ctx["lease"].check()
        except Exception as error:
            raise ExclusiveCostError("exclusive E03 resource lease boundary") from error


def _checkpoint(ctx, rows, trials, private_artifacts, *, complete=False):
    output = Path(ctx["report"]) / "costs/E03_JENA_BACKBONES.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    trial_path = output.with_name("E03_JENA_BACKBONE_TRIALS.csv")
    temporary = trial_path.with_name(trial_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "B", "trial", "ms_batch", "ms_origin"])
            writer.writeheader()
            writer.writerows(trials)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, trial_path)
    finally:
        temporary.unlink(missing_ok=True)
    value = dict(status="COMPLETE" if complete else "PARTIAL_CHECKPOINTED", rows=rows,
        private_artifacts=private_artifacts, measurements_exclusive=True,
        other_campaign_workers_observed=[], scope=SCOPE,
        CPU_GPU_costs_require_separate_plot_facets=True,
        accuracy_note="Missing saved V archives remain MSE/MAE NA. Failed parity remains latency NA.")
    atomic_json(output, value)
    return output, trial_path, value


def _sync():
    torch.cuda.synchronize()


def _reference(ctx, modules, method):
    old = modules["old_cache"]
    if method.startswith("Bolt_"):
        family = method.removeprefix("Bolt_")
        if family == "Raw":
            run = "primary_jena_E1_RAW_BOLT_SMALL_k0_s92601"
            path = old / "predictions" / run / "V_select.npz"
            manifest = modules["campaign"] / "predictions" / run / "V_select/PREDICTION_MANIFEST.json"
        else:
            relative = existing._old_run(92601, family)
            path = old / relative.parent.parent / "predictions" / relative.name / "V_select.npz"
            manifest = modules["campaign"] / relative.parent.parent / "predictions" / relative.name / "V_select/PREDICTION_MANIFEST.json"
        expected = read(manifest)["archive_sha256"] if manifest.is_file() else None
    elif method == "Ridge":
        run = "primary_jena_E1_RAW_RIDGE_k0_s92601"
        path = old / "predictions" / run / "V_select.npz"
        manifest = modules["campaign"] / "predictions" / run / "V_select/PREDICTION_MANIFEST.json"
        expected = read(manifest)["archive_sha256"] if manifest.is_file() else None
    else:
        run = ("chronos2_native_jena" if method == "Chronos2_nativeRaw" else
               "moirai_jena_" + method.removeprefix("Moirai_") + "_s92601" if method == "Moirai_Raw" else
               "moirai_jena_" + method.removeprefix("Moirai_") + "_k6_s92601")
        path = Path(ctx["cache"]) / "backbone_predictions" / (run + ".npz")
        expected = None
        result_file = Path(ctx["report"]) / "jobs" / ("D03_jena.json" if method == "Chronos2_nativeRaw" else "D02_moirai_s92601.json")
        if result_file.exists():
            rows = read(result_file).get("rows", [])
            record = next((row for row in rows if row["method"] == method), None)
            if record is not None:
                expected = record["prediction_sha256"]
    if not path.is_file():
        return None, None
    return existing._archive(path, expected), path


def _load(ctx, modules, task, method):
    """Load one route at a time, preserving distinct per-method load scopes."""
    mean, std, _ = modules["data"].fit_statistics(task, 12579)
    norm, model, foundation, ridge, artifact = dict(mean=mean, std=std), None, None, None, None
    private = []
    if method.startswith("Bolt_"):
        from gpu_runtime import FrozenBolt
        source = Path(ctx["asset"]) / modules["legacy"].MODEL_REL
        foundation = FrozenBolt(source, 48, row_microbatch=128, checkpoint_inputs=False, device="cuda")
        if foundation.initial_hash != modules["legacy"].F_SHA:
            raise ValueError("pinned Bolt weights changed")
        private.extend(p for p in source.glob("*") if p.is_file() and p.suffix in (".json", ".safetensors", ".bin"))
        family = method.removeprefix("Bolt_")
        if family != "Raw":
            model, norm, _, bundle_path = modules["legacy"].make_selected(task, 92601, family, foundation, modules)
            model = model.to("cuda").eval()
            private.append(bundle_path)
    elif method == "Ridge":
        ridge = importlib.import_module("ridge_baseline")
        path = modules["old_cache"] / "ridge/primary_jena_E1_RAW_RIDGE_k0_s92601/artifact.npz"
        artifact = ridge.load_artifact(path)
        norm = dict(mean=artifact["mean"], std=artifact["std"])
        private.append(path)
    else:
        from .backbones import load_backbone
        from .backbone_jobs import prepare_dependencies, private_bindings
        name = "chronos2" if method == "Chronos2_nativeRaw" else "moirai2"
        metadata = read(Path(ctx["report"]) / "BACKBONE_SOURCES.json")["models"][name]
        source = Path(ctx["cache"]) / "backbones/models" / name / metadata["revision"]
        if not (source / "model.safetensors").is_file():
            raise FileNotFoundError("E03 requires previously validated local model")
        if name == "moirai2":
            if not (Path(ctx["cache"]) / "backbones/python_overlay/DEPENDENCY_RECEIPT.json").is_file():
                raise FileNotFoundError("E03 requires previously validated dependency overlay")
            prepare_dependencies(ctx)
        selected_path = None
        if method in ("Moirai_B0", "Moirai_B2"):
            family = method.removeprefix("Moirai_")
            selected_path = Path(ctx["cache"]) / "runs" / f"moirai_jena_{family}_k6_s92601/selected_bundle.pt"
            if not selected_path.is_file():
                raise FileNotFoundError("optional Moirai selected adapter absent")
        smoke = read(Path(ctx["report"]) / "jobs" / ("D01_" + name + ".json"))
        if smoke.get("status") != "PASS":
            raise ValueError("E03 requires successful official backbone smoke")
        bindings = private_bindings(ctx, name)
        foundation = load_backbone(name, device="cuda", model_path=str(source), local_files_only=True)
        if method in ("Moirai_B0", "Moirai_B2"):
            path = selected_path
            bundle = torch.load(path, weights_only=False, map_location="cpu")
            model, norm = modules["fit"].create_model(bundle["spec"], task, 12579, foundation)
            model.load_state_dict(bundle["model"])
            model.eval()
            private.append(path)
    def predict(physical, observed):
        average, scale = norm["mean"], norm["std"]
        filled = physical.copy() if observed.all() else np.stack([
            modules["data"].causal_forward_fill(x, mask, average) for x, mask in zip(physical, observed)])
        normalized = ((filled - average) / scale).astype(np.float32)
        with torch.no_grad():
            if ridge is not None:
                forecast = ridge.predict_normalized(artifact, normalized).astype(np.float64)
            else:
                x = torch.as_tensor(normalized, dtype=torch.float32, device="cuda")
                forecast = (foundation(x) if model is None else model(x, foundation)).double().cpu().numpy()
        return forecast * scale.astype(np.float64) + average.astype(np.float64)
    def official_bolt_parity(physical, observed):
        from chronos import ChronosBoltPipeline
        average, scale = norm["mean"], norm["std"]
        filled = physical.copy() if observed.all() else np.stack([
            modules["data"].causal_forward_fill(x, mask, average) for x, mask in zip(physical, observed)])
        x = torch.as_tensor(((filled-average)/scale).astype(np.float32), device="cuda")
        with torch.no_grad():
            z = x if model is None else model.encode(x)
            wrapped = foundation(z)
            batch, length, channels = z.shape
            rows = z.transpose(1, 2).reshape(batch*channels, length).detach().cpu()
            quantiles = ChronosBoltPipeline(foundation.model).predict(rows, prediction_length=48)
            median = list(foundation.model.chronos_config.quantiles).index(.5)
            official = quantiles[:, median, :48].reshape(batch, channels, 48).transpose(1, 2).to("cuda")
            return bool(torch.allclose(wrapped, official, atol=1e-5, rtol=1e-4))
    return dict(predict=predict, foundation=foundation, model=model, norm=norm,
        core_dtype="float64" if ridge is not None else "float32",
        dtype="FP64 Ridge core; FP32 normalization; FP64 inverse; CPU return" if ridge is not None
              else "FP32 core; FP64 inverse; CPU return",
        trainable_params=int(artifact["coefficients"].size) if ridge is not None else 0 if model is None else sum(p.numel() for p in model.parameters()),
        resident_parameters=int(artifact["coefficients"].size) if ridge is not None else
            sum(p.numel() for p in foundation.model.parameters()) if method.startswith("Bolt_") and model is None else
            sum(p.numel() for p in foundation.parameters()) + (0 if model is None else sum(p.numel() for p in model.parameters()))
            if not method.startswith("Bolt_") else sum(p.numel() for p in foundation.model.parameters()) + sum(p.numel() for p in model.parameters()),
        official_bolt_parity=official_bolt_parity,
        private_paths=private, private_bindings=bindings if method.startswith(("Moirai_", "Chronos2_")) else [])


def execute_job(job, ctx):
    if job["question_id"] != "E03" or job.get("cohort", "jena") != "jena":
        raise ValueError("E03 supports only registered Jena C21")
    if not torch.cuda.is_available():
        return dict(status="BLOCKED_MISSING_ASSET", reason="E03 CUDA comparison device unavailable", artifact_paths=[])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _quiet(ctx)
    modules = existing._modules(ctx)
    task = existing._jena(ctx, modules)
    data = modules["data"]
    origins = data.origins(task["split_ranges"]["V_select"], 512, 48, 48)
    rows, trials, private_artifacts = [], [], []
    for method in METHODS:
        _quiet(ctx)
        route = None
        try:
            started = time.perf_counter()
            route = _load(ctx, modules, task, method)
            _sync()
            cold_load = time.perf_counter() - started
            arrays, archive_path = _reference(ctx, modules, method)
            if archive_path is not None:
                route["private_paths"].append(archive_path)
            private_artifacts.extend(route["private_bindings"])
            private_artifacts.extend(dict(reference=existing._reference(path, ctx), sha256=sha(path)) for path in route["private_paths"])
            for batch in (1, 8):
                selected = origins[:batch]
                physical = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
                observed = np.stack([task["observed"][o-512:o] for o in selected])
                _sync()
                physical_output = route["predict"](physical, observed)
                parity = "OFFICIAL_API_SMOKE_ONLY_NO_SAVED_ACCURACY"
                difference = None
                if arrays is None and method.startswith("Bolt_"):
                    parity = "OFFICIAL_API_ONLY_NO_SAVED_ACCURACY" if route["official_bolt_parity"](physical, observed) else "FAILED"
                elif arrays is None and method == "Ridge":
                    parity = "FIXED_VERIFIED_SOLVER_ARTIFACT_NO_SAVED_ACCURACY"
                if arrays is not None:
                    if not np.array_equal(arrays["origins"], origins) or not np.array_equal(arrays["channel_ids"], np.asarray(task["columns"])):
                        raise ValueError("E03 saved cohort identity changed")
                    normalized = (physical_output - route["norm"]["mean"].astype(np.float64)) / route["norm"]["std"].astype(np.float64)
                    expected = arrays["pred_normalized"][:batch]
                    difference = float(np.max(np.abs(normalized - expected)))
                    parity = "PASS" if np.allclose(normalized, expected, atol=1e-5, rtol=1e-4) else "FAILED"
                if parity == "FAILED":
                    rows.append(dict(method=method, B=batch, status="FAILED_PARITY", parity=parity,
                        max_parity_absolute_difference=difference, median_ms_batch=None, median_ms_origin=None))
                    continue
                samples = [psutil.Process().memory_info().rss]
                stop = threading.Event()
                def sample():
                    while not stop.wait(.005):
                        samples.append(psutil.Process().memory_info().rss)
                watcher = threading.Thread(target=sample, daemon=True)
                torch.cuda.reset_peak_memory_stats()
                watcher.start()
                try:
                    for iteration in range(30):
                        _quiet(ctx)
                        _sync()
                        tick = time.perf_counter()
                        result = route["predict"](physical, observed)
                        _sync()
                        elapsed = time.perf_counter() - tick
                        if result.shape != (batch, 48, 21) or not np.isfinite(result).all():
                            raise ValueError("invalid E03 full path output")
                        if iteration >= 10:
                            trials.append(dict(method=method, B=batch, trial=iteration-10,
                                ms_batch=elapsed*1000, ms_origin=elapsed*1000/batch))
                finally:
                    stop.set()
                    watcher.join()
                samples.append(psutil.Process().memory_info().rss)
                values = np.asarray([t["ms_batch"] for t in trials if t["method"] == method and t["B"] == batch])
                low, high = np.percentile(values, [25, 75])
                accuracy = existing.score_arrays(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"]) if arrays is not None else {}
                rows.append(dict(method=method, B=batch, status="COMPLETE", parity=parity,
                    max_parity_absolute_difference=difference, median_ms_batch=float(np.median(values)),
                    median_ms_origin=float(np.median(values)/batch), IQR_ms_batch=float(high-low),
                    IQR_ms_origin=float((high-low)/batch), warmups=10, repetitions=20,
                    device="cpu" if method == "Ridge" else "cuda", threads=2,
                    hardware=platform.processor() if method == "Ridge" else torch.cuda.get_device_name(),
                    scope=SCOPE, dtype=route["dtype"], core_dtype=route["core_dtype"],
                    physical_input_sha256=data.array_sha256(physical), mask_sha256=data.array_sha256(observed),
                    origins=selected.tolist(), C=21, K=6 if method.endswith(("B0", "B2")) else 21, L=512, H=48,
                    logical_F_rows=0 if method == "Ridge" else batch*(6 if method.endswith(("B0", "B2")) else 21),
                    trainable_params=route["trainable_params"], total_resident_parameters=route["resident_parameters"],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if method != "Ridge" else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if method != "Ridge" else None,
                    process_RSS_peak_sampled_bytes=max(samples), RSS_sampling_ms=5,
                    RSS_scope="per-method resident process, this B warmup+timed block; not activation-only",
                    cold_load_seconds=cold_load, cold_load_scope="this method only, CUDA context may already exist",
                    MSE=accuracy.get("MSE"), MAE=accuracy.get("MAE"),
                    accuracy_source="saved full V archive; this job verifies first B origins, not a fresh full V inference" if arrays is not None else "NA: saved full V archive absent",
                    routing="native_multivariate_per_origin" if method == "Chronos2_nativeRaw" else "independent_univariate" if method != "Ridge" else "own_channel_history",
                    dataset="jena", cohort="jena", split="V_select"))
            if route["foundation"] is not None:
                route["foundation"].assert_frozen_unchanged()
        except (ExclusiveCostError, TimeoutError):
            raise
        except Exception as error:
            # Route failures are isolated; diagnostics with local paths remain private.
            error_path = Path(ctx["cache"]) / "errors" / ("E03_" + method + ".json")
            atomic_json(error_path, dict(error_type=type(error).__name__, traceback=traceback.format_exc()))
            rows[:] = [row for row in rows if row["method"] != method]
            state = "BLOCKED_MISSING_ASSET" if isinstance(error, FileNotFoundError) else "FAILED"
            rows.extend(dict(method=method, B=batch, status=state, median_ms_batch=None,
                median_ms_origin=None, reason="Route " + type(error).__name__ + "; private diagnostic retained")
                for batch in (1, 8))
        finally:
            if route is not None:
                route.clear()
                del route
            gc.collect()
            torch.cuda.empty_cache()
            _checkpoint(ctx, rows, trials, private_artifacts)
    output, trial_path, value = _checkpoint(ctx, rows, trials, private_artifacts, complete=True)
    return dict(status="COMPLETE", reason="Exclusive full physical path; all failed/missing routes retained",
                artifact_paths=[str(output), str(trial_path)], result=value)
