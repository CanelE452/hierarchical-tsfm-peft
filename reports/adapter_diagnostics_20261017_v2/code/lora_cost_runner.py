"""E04 pinned LoRA device-replica replay/costs; runtime requires the campaign GPU lease."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


BROAD = "informs_broad_validation_20261015_v1"
NAMESPACE = "adapter_diagnostics_20261017_v2"
LEGACY_NAMESPACE = "priority_valid_family_20261016_v1"
RUN_ID = "postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1"
ARTIFACT_REL = f"{LEGACY_NAMESPACE}/runs/{RUN_ID}/selected_bundle.pt"
PREDICTION_REL = f"{LEGACY_NAMESPACE}/predictions/{RUN_ID}/V_select.npz"
PUBLIC_RUN_REL = f"research/{BROAD}/{LEGACY_NAMESPACE}/runs/{RUN_ID}"
MANIFEST_REL = f"research/{BROAD}/{LEGACY_NAMESPACE}/predictions/{RUN_ID}/V_select/PREDICTION_MANIFEST.json"
MODEL_REL = ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
F_SHA = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
DATA_SHA = "3928079b90efb066e02a8a3bef76dfefd92452a2ad0b26c29caf14393a161a58"
SCALER_SHA = "5f0c7d6aeddcfe9577e381375d36cadc47d2572469118da73e6ad837031fa2b9"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_routes(job):
    cohort = job.get("cohort", job.get("dataset_cohort", "jena"))
    expected = ["LoRA"] if cohort == "jena" else ["Raw", "B2", "Ridge", "LoRA", "Chronos2_nativeRaw"] if cohort == "pems08" else []
    actual = job.get("routes", expected)
    if not expected or actual != expected:
        raise ValueError("E04 routes differ from the registered cohort comparison")
    return actual


def read_public_json(repo, relative):
    repo = Path(repo).resolve()
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo):
        raise ValueError("Public metadata escaped repository")
    if path.is_file():
        return read(path), {path.relative_to(repo).as_posix(): sha(path)}
    index = repo / "research" / BROAD / "publication_derivatives/gzip_extra_aux_20261016_v1/GZIP_MANIFEST.json"
    matches = [item for item in read(index)["items"] if item["original_file"] == Path(relative).as_posix()]
    if len(matches) != 1:
        raise FileNotFoundError("Pinned public metadata absent")
    item = matches[0]
    packed = (index.parent / item["gzip_file"]).resolve()
    if not packed.is_relative_to(index.parent) or sha(packed) != item["gzip_sha256"]:
        raise ValueError("Public metadata compressed SHA changed")
    raw = gzip.decompress(packed.read_bytes())
    if len(raw) != item["original_bytes"] or hashlib.sha256(raw).hexdigest() != item["original_sha256"]:
        raise ValueError("Public metadata original SHA changed")
    return json.loads(raw), {index.relative_to(repo).as_posix(): sha(index),
                             packed.relative_to(repo).as_posix(): item["gzip_sha256"]}


def build_route_spec(receipt, manifest, source):
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != RUN_ID
            or receipt.get("family") != "LORA" or receipt.get("seed") != 92601
            or receipt.get("artifact_cache_relative_path") != ARTIFACT_REL
            or receipt.get("search_fit_end") != 10482 or receipt.get("refit_fit_end") != 12579
            or receipt.get("refit_scaler_sha") != SCALER_SHA):
        raise ValueError("LoRA fresh selected fit receipt changed")
    if (manifest.get("run_id") != RUN_ID or manifest.get("family") != "LORA"
            or manifest.get("seed") != 92601 or manifest.get("split") != "V_select"
            or manifest.get("data_sha") != DATA_SHA or manifest.get("scaler_sha") != SCALER_SHA
            or manifest.get("artifact_sha256") != receipt["artifact_sha256"]):
        raise ValueError("LoRA saved accuracy data/scaler/artifact identity changed")
    expected_shapes = dict(pred_normalized=[43, 48, 21], target_normalized=[43, 48, 21],
                           observed=[43, 48, 21], origins=[43], channel_ids=[21])
    if any(manifest.get("arrays", {}).get(key, {}).get("shape") != shape
           for key, shape in expected_shapes.items()):
        raise ValueError("LoRA saved V43 axes changed")
    spec = source.get("spec", {})
    if (source.get("data_sha") != DATA_SHA or source.get("model_sha") != F_SHA
            or spec.get("run_id") != RUN_ID or spec.get("family") != "LORA"
            or spec.get("dataset") != "jena" or spec.get("seed") != 92601
            or spec.get("lora_rank") != 8 or spec.get("lora_alpha") != 16
            or spec.get("F_row_microbatch") != 128):
        raise ValueError("LoRA source/foundation/rank contract changed")
    return dict(run_id=RUN_ID, accuracy_run_id=RUN_ID, dataset="jena", family="LORA",
                artifact_cache_relative_path=ARTIFACT_REL, artifact_sha256=receipt["artifact_sha256"],
                expected_prediction_model_sha=manifest["model_sha"], foundation_sha=F_SHA,
                canonical_fit_end=12579, latent_k=21, row_microbatch=128)


def inspect_assets(ctx):
    repo, asset = Path(ctx["repo"]), Path(ctx["asset"])
    old_cache = asset / ".cache" / BROAD
    receipt_path = repo / PUBLIC_RUN_REL / "BASE_FIT_RECEIPT.json"
    manifest_path = repo / MANIFEST_REL
    source_path = repo / PUBLIC_RUN_REL / "SOURCE_CONTRACT.json"
    receipt, manifest, source = read(receipt_path), read(manifest_path), read(source_path)
    spec = build_route_spec(receipt, manifest, source)
    public_inputs = {path.relative_to(repo).as_posix(): sha(path)
                     for path in (receipt_path, manifest_path, source_path)}
    for name, expected in source["files"].items():
        path = repo / "research" / BROAD / name
        if sha(path) != expected:
            raise ValueError("LoRA pinned scientific source changed")
        public_inputs[path.relative_to(repo).as_posix()] = expected
    for relative in (f"research/{BROAD}/deployment.py", f"research/{BROAD}/measure_costs.py",
                     f"research/{NAMESPACE}/existing_jobs.py", f"research/{NAMESPACE}/backbone_costs.py"):
        public_inputs[relative] = sha(repo / relative)
    private = []
    for relative, expected in ((ARTIFACT_REL, receipt["artifact_sha256"]),
                               (PREDICTION_REL, manifest["archive_sha256"])):
        path = old_cache / relative
        if sha(path) != expected:
            raise ValueError("LoRA private artifact SHA changed")
        private.append(dict(reference="asset://.cache/" + BROAD + "/" + relative, sha256=expected))
    model_pin = repo / "research/temporal_frontend_campaign_20261015_v1/RUNTIME_ENVIRONMENT.json"
    public_inputs[model_pin.relative_to(repo).as_posix()] = sha(model_pin)
    model_files = read(model_pin)["model_files"]
    for name in ("config.json", "model.safetensors"):
        path = asset / MODEL_REL / name
        if sha(path) != model_files[name]["sha256"]:
            raise ValueError("LoRA original foundation file SHA changed")
        private.append(dict(reference="asset://" + MODEL_REL + "/" + name, sha256=sha(path)))
    return dict(status="PASS_LOCAL_HASH_BOUND_ASSETS", route_spec=spec, public_inputs=public_inputs,
                private_artifacts=private, original_accuracy=dict(MSE=manifest["score"]["mse"],
                MAE=manifest["score"]["mae"], device=manifest["device"], source=MANIFEST_REL),
                model_loads=0, F_calls=0, new_downloads=0)


def saved_prediction_parity(physical_output, arrays, origins, channels, mean, std):
    import numpy as np
    if not np.array_equal(arrays["origins"], origins):
        raise ValueError("LoRA saved origin identity changed")
    if not np.array_equal(arrays["channel_ids"], channels):
        raise ValueError("LoRA saved channel order changed")
    expected = arrays["pred_normalized"]
    if (expected.shape != physical_output.shape or arrays["target_normalized"].shape != expected.shape
            or arrays["observed"].shape != expected.shape or arrays["observed"].dtype != np.dtype(bool)):
        raise ValueError("LoRA saved target/mask/prediction axes changed")
    normalized = (np.asarray(physical_output, np.float64) - mean.astype(np.float64)) / std.astype(np.float64)
    difference = float(np.max(np.abs(normalized - expected)))
    passed = bool(np.isfinite(normalized).all() and np.allclose(normalized, expected, atol=1e-5, rtol=1e-4))
    return dict(status="PASS" if passed else "FAILED", maximum_absolute_difference=difference,
                atol=1e-5, rtol=1e-4, tolerance_registered_before_measurement=True,
                comparison="CUDA merged device replica versus original saved CPU selected normalized prediction")


def measured_batch_parity(physical_output, replica, selected_origins, channels, mean, std):
    import numpy as np
    indices = []
    for origin in selected_origins:
        matches = np.flatnonzero(replica["origins"] == origin)
        if len(matches) != 1:
            raise ValueError("Measured batch origin missing or duplicated in saved device replica")
        indices.append(int(matches[0]))
    selected = {key: replica[key][indices] for key in
                ("pred_normalized", "target_normalized", "observed", "origins")}
    selected["channel_ids"] = replica["channel_ids"]
    result = saved_prediction_parity(physical_output, selected, selected_origins, channels, mean, std)
    result.update(B=len(selected_origins), origins=selected_origins.tolist(),
                  comparison="Actual timed B output versus matching origins of saved device replica",
                  timing_scope="Comparison performed after stopping the latency timer")
    return result


def _npz_atomic(path, arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_trials(path, trials):
    fields = ["method", "B", "trial", "ms_batch", "ms_origin", "F_logical_streams", "F_physical_streams", "F_calls"]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execute_jena(job, ctx):
    if (job.get("question_id") != "E04" or job.get("device") != "cuda"
            or job.get("cohort", job.get("dataset_cohort", "jena")) != "jena"):
        raise ValueError("E04 supports only registered Jena CUDA LoRA B1/B8")
    if ctx.get("lease") is None:
        raise RuntimeError("E04 requires the active parent campaign GPU lease")
    ctx["lease"].check()
    from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
    from research.adapter_diagnostics_20261017_v2 import backbone_costs as comparable
    from research.adapter_diagnostics_20261017_v2.core import atomic_json
    import numpy as np
    import psutil
    import torch

    comparable._quiet(ctx)
    assets = inspect_assets(ctx)
    modules = existing._modules(ctx)
    data = modules["data"]
    task = existing._jena(ctx, modules)
    origins = data.origins(task["split_ranges"]["V_select"], 512, 48, 48)
    manifest = read(Path(ctx["repo"]) / MANIFEST_REL)
    archive = Path(ctx["asset"]) / ".cache" / BROAD / PREDICTION_REL
    arrays = existing._archive(archive, manifest["archive_sha256"], manifest["arrays"])
    mean, std, _ = data.fit_statistics(task, 12579)
    if modules["fit"].scaler_fingerprint(mean, std) != SCALER_SHA:
        raise ValueError("LoRA actual A60 scaler differs from saved reference")
    if (not np.array_equal(arrays["origins"], origins)
            or not np.array_equal(arrays["channel_ids"], np.asarray(task["columns"]))):
        raise ValueError("LoRA complete V origin/channel identity differs")
    targets = [data.target_window(task, int(origin), mean, std) for origin in origins]
    if (not np.array_equal(arrays["target_normalized"], np.stack([pair[0] for pair in targets]))
            or not np.array_equal(arrays["observed"], np.stack([pair[1] for pair in targets]))):
        raise ValueError("LoRA actual complete V targets/masks differ from pinned reference")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(ctx["report"]) / "costs/E04_JENA_LORA.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    trial_path = output.with_name("E04_JENA_LORA_TRIALS.csv")
    rows, trials = [], []
    private = list(assets["private_artifacts"])
    route = None
    value = dict(status="PARTIAL_CHECKPOINTED", rows=rows, full_v_replay=None,
                 source_method="LoRA", accuracy_run_id=RUN_ID, private_artifacts=private,
                 source_bindings=assets["public_inputs"], scope=comparable.SCOPE,
                 measurements_exclusive=True, training_updates=0, new_downloads=0,
                 original_CPU_accuracy_preserved=True, runner_sha256=sha(__file__))
    try:
        comparable._quiet(ctx)
        cold_started = time.perf_counter()
        route = modules["deployment"].Route(assets["route_spec"], task, modules["old_cache"],
                    Path(ctx["asset"]) / MODEL_REL, "cuda")
        torch.cuda.synchronize()
        cold_load = time.perf_counter() - cold_started
        value["cold_load_seconds"] = cold_load
        value["cold_load_scope"] = "this method route/F construction; excludes merge/parity; CUDA context and OS cache may already exist"
        selected = origins[:8]
        physical = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
        observed = np.stack([task["observed"][o-512:o] for o in selected])
        comparable._quiet(ctx)
        parity_started = time.perf_counter()
        merged = route.check_lora_parity(physical, observed, atol=1e-5, rtol=1e-4)
        if merged["status"] != "PASS":
            raise ValueError("LoRA merged/unmerged parity failed")
        predictions = []
        for start in range(0, len(origins), 8):
            comparable._quiet(ctx)
            selected = origins[start:start+8]
            physical = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
            observed = np.stack([task["observed"][o-512:o] for o in selected])
            physical_prediction = route.predict_physical(physical, observed)
            predictions.append(((physical_prediction - mean.astype(np.float64)) / std.astype(np.float64)).astype(np.float32))
        prediction = np.concatenate(predictions)
        physical_prediction = prediction.astype(np.float64) * std.astype(np.float64) + mean.astype(np.float64)
        parity = saved_prediction_parity(physical_prediction, arrays, origins, np.asarray(task["columns"]), mean, std)
        parity_seconds = time.perf_counter() - parity_started
        replica = dict(arrays, pred_normalized=prediction)
        private_output = Path(ctx["cache"]) / "lora_cost/E04_JENA_LORA_CUDA_V43.npz"
        _npz_atomic(private_output, replica)
        replay_sha = sha(private_output)
        private.append(dict(reference=existing._reference(private_output, ctx), sha256=replay_sha))
        stored = existing._archive(private_output, replay_sha)
        replay = existing.score_arrays(stored["pred_normalized"], stored["target_normalized"], stored["observed"])
        original = existing.score_arrays(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"])
        value["full_v_replay"] = dict(replay, status=parity["status"], device="cuda", device_replica=True,
            run_id=job["job_id"]+"_LoRA_V_replay", cohort="jena", method="LoRA", seed=92601,
            C=21, K=21, L=512, H=48, backend="torch_eager", dtype="float32", point_statistic="quantile_0.5",
            routing="independent_univariate", split="V_select", metric_scale="canonical_A60",
            origin="EXISTING_SELECTED_ROUTE_DEVICE_PARITY_REPLAY", data_sha256=task["data_sha256"],
            source_checkpoint_sha256=assets["route_spec"]["artifact_sha256"],
            prediction_sha256=replay_sha, private_prediction_reference=private_output.relative_to(Path(ctx["cache"])).as_posix(),
            private_prediction_ref=existing._reference(private_output, ctx),
            new_fit=False, training_updates=0, original_accuracy_run_id=RUN_ID, reference_manifest=MANIFEST_REL,
            reference_manifest_sha256=sha(Path(ctx["repo"]) / MANIFEST_REL), source_archive_sha256=replay_sha,
            source_archive_reference=existing._reference(private_output, ctx),
            array_fingerprints={key: existing._array_fingerprint(array) for key, array in replica.items()},
            scaler_sha256=SCALER_SHA, complete_origin_target_mask_channel_identity="EXACT",
            saved_CPU_prediction_parity=parity, merged_unmerged_parity=merged,
            parity_seconds=parity_seconds, parity_scope="merge check + complete V43 physical-input replay; excluded from latency trials",
            original_CPU_MSE=original["MSE"], original_CPU_MAE=original["MAE"],
            CPU_replica_absolute_metric_difference={key: abs(replay[key]-original[key]) for key in ("MSE", "MAE")})
        atomic_json(output, value)
        if parity["status"] != "PASS":
            value["status"] = "FAILED"
            value["reason"] = "Full V43 saved CPU versus CUDA device-replica parity failed; no valid latency"
            atomic_json(output, value)
            return dict(status="FAILED", reason=value["reason"], artifact_paths=[str(output)], result=value)
        fitted_params = sum(p.numel() for p in route.model.parameters() if p.requires_grad)
        foundation_params = sum(p.numel() for p in route.pool.foundation.model.parameters())
        value["full_v_replay"].update(trainable_params=fitted_params, deployed_trainable_params=0,
            selected_epoch=route.bundle.get("selected_epoch"), total_resident_parameters=foundation_params+fitted_params)
        for batch in (1, 8):
            selected = origins[:batch]
            physical = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
            observed = np.stack([task["observed"][o-512:o] for o in selected])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sampler = existing.RSSSampler(lambda: psutil.Process().memory_info().rss)
            measurement_parity = dict(status="PASS", verified_timed_outputs=0, expected_timed_outputs=20,
                maximum_absolute_difference=0., reference_archive_sha256=replay_sha,
                reference_run_id=value["full_v_replay"]["run_id"], B=batch, origins=selected.tolist(),
                atol=1e-5, rtol=1e-4, timing_scope="Each timed output compared after stopping latency timer")
            sampler.start()
            try:
                for iteration in range(30):
                    comparable._quiet(ctx)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    result = route.predict_physical(physical, observed)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    if result.dtype != np.float64 or result.shape != (batch, 48, 21) or not np.isfinite(result).all():
                        raise ValueError("LoRA physical FP64 CPU output changed")
                    receipt = route.last_receipt
                    if receipt["F_logical_streams"] != batch * 21:
                        raise ValueError("LoRA measured logical F rows differ from physical cohort")
                    if iteration >= 10:
                        compared = measured_batch_parity(result, stored, selected, np.asarray(task["columns"]), mean, std)
                        measurement_parity["maximum_absolute_difference"] = max(
                            measurement_parity["maximum_absolute_difference"], compared["maximum_absolute_difference"])
                        if compared["status"] != "PASS":
                            measurement_parity.update(status="FAILED", failed_timed_trial=iteration-10)
                            break
                        measurement_parity["verified_timed_outputs"] += 1
                        trials.append(dict(method="LoRA", B=batch, trial=iteration-10,
                            ms_batch=elapsed*1000, ms_origin=elapsed*1000/batch,
                            **{key: receipt[key] for key in ("F_logical_streams", "F_physical_streams", "F_calls")}))
            finally:
                rss = sampler.stop()
            if measurement_parity["status"] != "PASS":
                trials[:] = [trial for trial in trials if trial["B"] != batch]
                rows.append(dict(method="LoRA", source_method="LoRA", B=batch, status="FAILED_PARITY",
                    parity="FAILED", measured_batch_parity=measurement_parity,
                    accuracy_run_id=value["full_v_replay"]["run_id"], accuracy_archive_sha256=replay_sha,
                    median_ms_batch=None, median_ms_origin=None, IQR_ms_batch=None, IQR_ms_origin=None,
                    reason="Actual measured batch output differs from saved device replica"))
                _write_trials(trial_path, trials)
                atomic_json(output, value)
                continue
            values = np.asarray([trial["ms_batch"] for trial in trials if trial["B"] == batch])
            low, high = np.percentile(values, [25, 75])
            memory = route.memory_state()
            rows.append(dict(method="LoRA", source_method="LoRA", B=batch, status="COMPLETE", parity="PASS",
                measured_batch_parity=measurement_parity,
                accuracy_run_id=value["full_v_replay"]["run_id"], accuracy_source_archive_sha256=replay_sha,
                accuracy_archive_sha256=replay_sha, accuracy_MSE=replay["MSE"], accuracy_MAE=replay["MAE"],
                accuracy_reference_run_id=RUN_ID, accuracy_reference_archive_sha256=manifest["archive_sha256"],
                accuracy_reference_MSE=original["MSE"], accuracy_reference_MAE=original["MAE"],
                checkpoint_sha256=assets["route_spec"]["artifact_sha256"], scaler_sha256=SCALER_SHA,
                max_parity_absolute_difference=measurement_parity["maximum_absolute_difference"],
                full_v_reference_max_parity_absolute_difference=parity["maximum_absolute_difference"],
                median_ms_batch=float(np.median(values)), median_ms_origin=float(np.median(values)/batch),
                IQR_ms_batch=float(high-low), IQR_ms_origin=float((high-low)/batch),
                warmups=10, repetitions=20, device="cuda", threads=2, hardware=torch.cuda.get_device_name(),
                scope=comparable.SCOPE, dtype="FP32 core; FP64 inverse; CPU return", core_dtype="float32",
                inverse_transform_dtype="float64", physical_output_dtype="float64", lora_state="merged",
                physical_input_sha256=data.array_sha256(physical), mask_sha256=data.array_sha256(observed),
                origins=selected.tolist(), C=21, K=21, L=512, H=48, logical_F_rows=batch*21,
                trainable_params=fitted_params, fitted_trainable_params=fitted_params,
                deployed_trainable_params=0, total_resident_parameters=foundation_params+fitted_params,
                GPU_resident_foundation_parameters=foundation_params, CPU_resident_adapter_parameters=fitted_params,
                resident_parameter_scope="GPU merged F + retained CPU fitted adapter; restore/merged snapshots in RSS, not parameter count",
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_RSS_peak_sampled_bytes=rss["process_RSS_sampled_peak_bytes"], RSS_sampling_ms=5,
                RSS_scope="per-method resident process, this B warmup+timed block and post-timer CPU parity checks; excludes full V parity/cold load",
                cold_load_seconds=cold_load, cold_load_scope=value["cold_load_scope"],
                MSE=replay["MSE"], MAE=replay["MAE"],
                accuracy_source="complete CUDA device replica V43 saved replay; original CPU selected reference preserved separately; no new fit",
                device_replica_MSE=replay["MSE"], device_replica_MAE=replay["MAE"],
                routing="independent_univariate", dataset="jena", cohort="jena", split="V_select",
                row_microbatch=128, F_snapshot_revision=Path(MODEL_REL).name,
                resident_weights_gpu_MiB=memory["resident_weights_gpu_MiB"],
                resident_weights_cpu_MiB=memory["resident_weights_cpu_MiB"],
                active_adapter_MiB=memory["active_adapter_MiB"],
                backbone_restore_snapshots_scope="CPU original/merged copies retained for exact teardown; included in process RSS"))
            _write_trials(trial_path, trials)
            atomic_json(output, value)
        value["status"] = "COMPLETE" if len(rows) == 2 and all(row["parity"] == "PASS" for row in rows) else "FAILED"
    finally:
        if route is not None:
            route.close()
            value["backbone_exact_restore_at_close"] = "PASS_EXACT_HASH"
            del route
        gc.collect()
        torch.cuda.empty_cache()
    _write_trials(trial_path, trials)
    atomic_json(output, value)
    return dict(status=value["status"], reason="Exclusive pinned selected LoRA physical path and complete V43 device-replica parity",
                artifact_paths=[str(output), str(trial_path)], result=value)


def inspect_pems_assets(ctx):
    repo, asset, cache = (Path(ctx[key]) for key in ("repo", "asset", "cache"))
    old = asset / ".cache" / BROAD
    pems_data = "d7b31d48ba3f3e15f571ad3f850732e53aab6ec1289cf3f33e40763c295b8697"
    scaler = "94e8c71c42f8d58a1d258fb200d86a24595fe1793203da1543b3e47a60569c60"
    recipes, public, private = {}, {}, []
    def metadata(relative):
        value, bindings = read_public_json(repo, relative)
        public.update(bindings)
        return value
    def bound(path, expected, root="asset"):
        if sha(path) != expected:
            raise ValueError("PeMS cost private asset SHA changed")
        base = asset if root == "asset" else cache
        private.append(dict(reference=root+"://"+path.relative_to(base).as_posix(), sha256=expected))
    for method, run, family in (("Raw", "primary_pems08_E1_RAW_BOLT_SMALL_k0_s92601", "RAW"),
                                ("B2", "primary_pems08_E1_B2_k43_s92601__e1", "B2"),
                                ("Ridge", "primary_pems08_E1_RAW_RIDGE_k0_s92601", "RAW_RIDGE"),
                                ("LoRA", "primary_pems08_E1_LORA_R8_k0_s92601", "LORA")):
        prefix = f"research/{BROAD}"
        route_manifest_rel = f"{prefix}/predictions/{run}/V_select/PREDICTION_MANIFEST.json"
        route_manifest = metadata(route_manifest_rel)
        manifest, manifest_rel = route_manifest, route_manifest_rel
        if method == "B2":
            manifest_rel = f"{prefix}/priority_core_followup_20261016_v1/parent_full_cpu/diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1/V_select/PREDICTION_MANIFEST.json"
            manifest = metadata(manifest_rel)
        if manifest["data_sha"] != pems_data or manifest["scaler_sha"] != scaler or manifest["split"] != "V_select":
            raise ValueError("PeMS cost saved reference data/scaler/split changed")
        if any(manifest["arrays"][key]["shape"] != [148, 12, 170] for key in ("pred_normalized", "target_normalized", "observed")):
            raise ValueError("PeMS cost requires full C170 V148")
        prediction_relative = manifest.get("cache_relative_path", f"predictions/{run}/V_select.npz")
        archive = old / prediction_relative
        bound(archive, manifest["archive_sha256"])
        spec = dict(run_id=run, dataset="pems08", family=family, latent_k=43 if method == "B2" else 170,
                    canonical_fit_end=10713, row_microbatch=128)
        artifact_sha, selected_epoch = None, None
        if method != "Raw":
            receipt_name = "RIDGE_FIT_RECEIPT.json" if method == "Ridge" else "BASE_FIT_RECEIPT.json"
            receipt = metadata(f"{prefix}/runs/{run}/{receipt_name}")
            if receipt["status"] != "COMPLETE_VALID" or receipt["run_id"] != run or receipt["family"] != family:
                raise ValueError("PeMS cost fit receipt identity changed")
            if method != "Ridge" and (receipt["refit_fit_end"] != 10713 or receipt["refit_scaler_sha"] != scaler):
                raise ValueError("PeMS cost selected fresh A60 fit changed")
            artifact_sha = receipt["artifact_sha256"]
            bound(old / receipt["artifact_cache_relative_path"], artifact_sha)
            spec.update(artifact_cache_relative_path=receipt["artifact_cache_relative_path"], artifact_sha256=artifact_sha)
            selected_epoch = receipt.get("selected_epoch")
            if method != "Ridge":
                source = metadata(f"{prefix}/runs/{run}/SOURCE_CONTRACT.json")
                if source["data_sha"] != pems_data or source["model_sha"] != F_SHA:
                    raise ValueError("PeMS cost trained data/foundation changed")
                for filename, expected in source["files"].items():
                    relative = f"{prefix}/{filename}"
                    if sha(repo/relative) != expected:
                        raise ValueError("PeMS cost pinned source changed")
                    public[relative] = expected
                spec["expected_prediction_model_sha"] = route_manifest["model_sha"]
        if method != "Ridge":
            spec["foundation_sha"] = F_SHA
        recipes[method] = dict(route_spec=spec, reference_manifest=manifest,
            reference_manifest_relative=manifest_rel, reference_archive_reference="asset://.cache/"+BROAD+"/"+prediction_relative,
            accuracy_run_id=manifest["run_id"], checkpoint_sha256=artifact_sha, selected_epoch=selected_epoch,
            scaler_sha256=scaler, device="cpu" if method == "Ridge" else "cuda")
    source_rel = f"reports/{NAMESPACE}/BACKBONE_SOURCES.json"
    sources = metadata(source_rel)["models"]["chronos2"]
    native_rel = f"reports/{NAMESPACE}/jobs/D03_pems08.json"
    native = metadata(native_rel)["rows"][0]
    if (native["method"] != "Chronos2_nativeRaw" or native["C"] != 170 or native["H"] != 12
            or native["backbone_revision"] != sources["revision"]):
        raise ValueError("PeMS native Chronos2 accuracy identity changed")
    model = cache / "backbones/models/chronos2" / sources["revision"]
    bound(model / "model.safetensors", sources["weights"]["sha256"], "cache")
    for filename in ("config.json", "README.md"):
        descriptor = next(row for row in sources["sources"] if row["url"].endswith("/"+filename))
        bound(model/filename, descriptor["sha256"], "cache")
    native_archive = cache / native["private_prediction_reference"]
    bound(native_archive, native["prediction_sha256"], "cache")
    recipes["Chronos2_nativeRaw"] = dict(device="cuda", reference_manifest=native,
        reference_manifest_relative=native_rel, reference_archive_reference="cache://"+native["private_prediction_reference"],
        accuracy_run_id="D03_pems08_0", checkpoint_sha256=None, selected_epoch=None,
        scaler_sha256=scaler, model_revision=sources["revision"], model_reference="cache://"+model.relative_to(cache).as_posix())
    pin_rel = "research/temporal_frontend_campaign_20261015_v1/RUNTIME_ENVIRONMENT.json"
    pin = metadata(pin_rel)
    for filename in ("config.json", "model.safetensors"):
        bound(asset/MODEL_REL/filename, pin["model_files"][filename]["sha256"])
    for filename in (f"research/{BROAD}/deployment.py", f"research/{BROAD}/measure_costs.py",
                     f"research/{NAMESPACE}/backbones.py", f"research/{NAMESPACE}/existing_jobs.py",
                     f"research/{NAMESPACE}/backbone_costs.py", f"research/{NAMESPACE}/backbone_jobs.py"):
        public[filename] = sha(repo/filename)
    return dict(status="PASS_LOCAL_HASH_BOUND_ASSETS", routes=recipes, public_inputs=public,
                private_artifacts=private, model_loads=0, F_calls=0, new_downloads=0)


def _execute_pems(job, ctx):
    from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
    from research.adapter_diagnostics_20261017_v2 import backbone_costs as comparable
    from research.adapter_diagnostics_20261017_v2 import backbone_jobs
    from research.adapter_diagnostics_20261017_v2.core import atomic_json
    import numpy as np
    import psutil
    import platform
    import torch
    import traceback

    comparable._quiet(ctx)
    assets = inspect_pems_assets(ctx)
    modules = existing._modules(ctx)
    data = modules["data"]
    task = backbone_jobs._task(ctx, modules, "pems08")
    origins = data.origins(task["split_ranges"]["V_select"], 512, 12, 12)
    mean, std, _ = data.fit_statistics(task, 10713)
    targets = [data.target_window(task, int(origin), mean, std) for origin in origins]
    target, target_mask = np.stack([pair[0] for pair in targets]), np.stack([pair[1] for pair in targets])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(ctx["report"]) / "costs/E04_PEMS08_NATIVE_BASELINES.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    trial_path = output.with_name("E04_PEMS08_NATIVE_BASELINES_TRIALS.csv")
    rows, trials, replay_by_method = [], [], {}
    value = dict(status="PARTIAL_CHECKPOINTED", rows=rows, full_v_replay=replay_by_method,
                 private_artifacts=list(assets["private_artifacts"]), source_bindings=assets["public_inputs"],
                 scope=comparable.SCOPE, cohort="pems08", measurements_exclusive=True,
                 training_updates=0, new_downloads=0, original_accuracy_preserved=True, runner_sha256=sha(__file__))
    for method in registered_routes(job):
        recipe = assets["routes"][method]
        device = recipe["device"]
        route = native = None
        try:
            comparable._quiet(ctx)
            reference = recipe["reference_manifest"]
            archive = existing._resolve(recipe["reference_archive_reference"], ctx)
            arrays = existing._archive(archive, reference.get("archive_sha256", reference.get("prediction_sha256")), reference.get("arrays"))
            if (not np.array_equal(arrays["origins"], origins) or not np.array_equal(arrays["channel_ids"], np.asarray(task["columns"]))
                    or not np.array_equal(arrays["target_normalized"], target) or not np.array_equal(arrays["observed"], target_mask)
                    or modules["fit"].scaler_fingerprint(mean, std) != recipe["scaler_sha256"]):
                raise ValueError("PeMS complete V148 origin/target/mask/channel/scaler identity changed")
            cold_started = time.perf_counter()
            if method == "Chronos2_nativeRaw":
                from research.adapter_diagnostics_20261017_v2.backbones import load_backbone
                native = load_backbone("chronos2", device="cuda", length=512, horizon=12,
                    model_path=str(existing._resolve(recipe["model_reference"], ctx)), local_files_only=True)
                def predict(physical, observed):
                    filled = physical.copy() if observed.all() else np.stack([data.causal_forward_fill(x, mask, mean) for x, mask in zip(physical, observed)])
                    normalized = ((filled-mean)/std).astype(np.float32)
                    with torch.no_grad():
                        result = native(torch.as_tensor(normalized, device="cuda")).double().cpu().numpy()
                    return result*std.astype(np.float64)+mean.astype(np.float64)
            else:
                route = modules["deployment"].Route(recipe["route_spec"], task, modules["old_cache"], Path(ctx["asset"])/MODEL_REL, device)
                predict = route.predict_physical
            if device == "cuda":
                torch.cuda.synchronize()
            cold_load = time.perf_counter()-cold_started
            parity_started = time.perf_counter()
            merged = None
            if method == "LoRA":
                selected = origins[:8]
                x = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
                mask = np.stack([task["observed"][o-512:o] for o in selected])
                merged = route.check_lora_parity(x, mask, atol=1e-5, rtol=1e-4)
            predictions = []
            for start in range(0, len(origins), 4):
                comparable._quiet(ctx)
                selected = origins[start:start+4]
                x = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
                mask = np.stack([task["observed"][o-512:o] for o in selected])
                physical_output = predict(x, mask)
                predictions.append(((physical_output-mean.astype(np.float64))/std.astype(np.float64)).astype(np.float64 if method == "Ridge" else np.float32))
            predictions = np.concatenate(predictions)
            parity = saved_prediction_parity(predictions.astype(np.float64)*std+mean, arrays, origins, np.asarray(task["columns"]), mean, std)
            parity["comparison"] = "Whole physical-path device replica versus original selected saved V148; no new fit"
            replica = dict(arrays, pred_normalized=predictions)
            private_path = Path(ctx["cache"])/"lora_cost"/("E04_PEMS08_"+method+"_V148.npz")
            _npz_atomic(private_path, replica)
            replica_sha = sha(private_path)
            value["private_artifacts"].append(dict(reference=existing._reference(private_path, ctx), sha256=replica_sha))
            stored = existing._archive(private_path, replica_sha)
            score = existing.score_arrays(stored["pred_normalized"], stored["target_normalized"], stored["observed"])
            original = existing.score_arrays(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"])
            replay_by_method[method] = dict(score, status=parity["status"], device=device, device_replica=True,
                run_id=job["job_id"]+"_"+method+"_V_replay", cohort="pems08", method=method, seed=92601,
                C=170, K=43 if method == "B2" else 170, L=512, H=12,
                backend="numpy_blas" if method == "Ridge" else "torch_eager",
                dtype="float64" if method == "Ridge" else "float32",
                point_statistic="deterministic_ridge" if method == "Ridge" else "quantile_0.5",
                routing="own_channel_history" if method == "Ridge" else "native_multivariate_per_origin" if native is not None else "independent_univariate",
                split="V_select", metric_scale="canonical_A60", origin="EXISTING_SELECTED_ROUTE_DEVICE_PARITY_REPLAY",
                data_sha256=task["data_sha256"], source_checkpoint_sha256=recipe["checkpoint_sha256"],
                prediction_sha256=replica_sha, private_prediction_reference=private_path.relative_to(Path(ctx["cache"])).as_posix(),
                private_prediction_ref=existing._reference(private_path, ctx),
                new_fit=False, training_updates=0, original_accuracy_run_id=recipe["accuracy_run_id"],
                reference_manifest=recipe["reference_manifest_relative"], reference_archive_sha256=sha(archive),
                source_archive_reference=existing._reference(private_path, ctx), source_archive_sha256=replica_sha,
                array_fingerprints={key: existing._array_fingerprint(array) for key, array in replica.items()},
                scaler_sha256=recipe["scaler_sha256"], complete_origin_target_mask_channel_identity="EXACT",
                saved_prediction_parity=parity, merged_unmerged_parity=merged,
                parity_seconds=time.perf_counter()-parity_started,
                parity_scope="merge check when applicable + complete physical V148 replay; excluded from timed trials",
                original_MSE=original["MSE"], original_MAE=original["MAE"],
                original_replica_absolute_metric_difference={key: abs(score[key]-original[key]) for key in ("MSE", "MAE")})
            if parity["status"] != "PASS":
                raise ValueError("PeMS complete saved V148 prediction parity failed")
            fitted_params = (int(route.ridge_artifact["coefficients"].size) if method == "Ridge" else
                             0 if method in ("Raw", "Chronos2_nativeRaw") else sum(p.numel() for p in route.model.parameters() if p.requires_grad))
            foundation_params = 0 if method == "Ridge" else sum(p.numel() for p in (native.parameters() if native is not None else route.pool.foundation.model.parameters()))
            replay_by_method[method].update(trainable_params=fitted_params,
                deployed_trainable_params=0 if method == "LoRA" else fitted_params,
                selected_epoch=recipe["selected_epoch"], total_resident_parameters=foundation_params+fitted_params)
            for batch in (1, 8):
                selected = origins[:batch]
                physical = np.stack([task["physical"][o-512:o] for o in selected]).astype(np.float32)
                observed = np.stack([task["observed"][o-512:o] for o in selected])
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                sampler = existing.RSSSampler(lambda: psutil.Process().memory_info().rss)
                measurement_parity = dict(status="PASS", verified_timed_outputs=0, expected_timed_outputs=20,
                    maximum_absolute_difference=0., reference_archive_sha256=replica_sha,
                    reference_run_id=replay_by_method[method]["run_id"], B=batch, origins=selected.tolist(),
                    atol=1e-5, rtol=1e-4, timing_scope="Each timed output compared after stopping latency timer")
                sampler.start()
                try:
                    for iteration in range(30):
                        comparable._quiet(ctx)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        counters = (native.streams, native.physical_streams, native.physical_calls) if native is not None else None
                        tick = time.perf_counter()
                        result = predict(physical, observed)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter()-tick
                        if result.dtype != np.float64 or result.shape != (batch, 12, 170) or not np.isfinite(result).all():
                            raise ValueError("PeMS physical FP64 CPU output changed")
                        counts = (dict(zip(("F_logical_streams", "F_physical_streams", "F_calls"),
                            [native.streams-counters[0], native.physical_streams-counters[1], native.physical_calls-counters[2]]))
                            if native is not None else {key: route.last_receipt[key] for key in ("F_logical_streams", "F_physical_streams", "F_calls")})
                        expected = 0 if method == "Ridge" else batch*(43 if method == "B2" else 170)
                        if counts["F_logical_streams"] != expected:
                            raise ValueError("PeMS measured F rows differ from registered cohort")
                        if iteration >= 10:
                            compared = measured_batch_parity(result, stored, selected, np.asarray(task["columns"]), mean, std)
                            measurement_parity["maximum_absolute_difference"] = max(
                                measurement_parity["maximum_absolute_difference"], compared["maximum_absolute_difference"])
                            if compared["status"] != "PASS":
                                measurement_parity.update(status="FAILED", failed_timed_trial=iteration-10)
                                break
                            measurement_parity["verified_timed_outputs"] += 1
                            trials.append(dict(method=method, B=batch, trial=iteration-10,
                                ms_batch=elapsed*1000, ms_origin=elapsed*1000/batch, **counts))
                finally:
                    rss = sampler.stop()
                if measurement_parity["status"] != "PASS":
                    trials[:] = [trial for trial in trials if trial["method"] != method or trial["B"] != batch]
                    rows.append(dict(method=method, source_method=method, B=batch, status="FAILED_PARITY",
                        parity="FAILED", measured_batch_parity=measurement_parity,
                        accuracy_run_id=replay_by_method[method]["run_id"], accuracy_archive_sha256=replica_sha,
                        median_ms_batch=None, median_ms_origin=None, IQR_ms_batch=None, IQR_ms_origin=None,
                        reason="Actual measured batch output differs from saved device replica", cohort="pems08"))
                    _write_trials(trial_path, trials)
                    atomic_json(output, value)
                    continue
                values = np.asarray([trial["ms_batch"] for trial in trials if trial["method"] == method and trial["B"] == batch])
                low, high = np.percentile(values, [25, 75])
                rows.append(dict(method=method, source_method=method, B=batch, status="COMPLETE", parity="PASS",
                    measured_batch_parity=measurement_parity,
                    accuracy_run_id=replay_by_method[method]["run_id"], accuracy_source_archive_sha256=replica_sha,
                    accuracy_archive_sha256=replica_sha, accuracy_MSE=score["MSE"], accuracy_MAE=score["MAE"],
                    accuracy_reference_run_id=recipe["accuracy_run_id"], accuracy_reference_archive_sha256=sha(archive),
                    accuracy_reference_MSE=original["MSE"], accuracy_reference_MAE=original["MAE"],
                    checkpoint_sha256=recipe["checkpoint_sha256"], scaler_sha256=recipe["scaler_sha256"],
                    median_ms_batch=float(np.median(values)), median_ms_origin=float(np.median(values)/batch),
                    IQR_ms_batch=float(high-low), IQR_ms_origin=float((high-low)/batch), warmups=10, repetitions=20,
                    max_parity_absolute_difference=measurement_parity["maximum_absolute_difference"],
                    full_v_reference_max_parity_absolute_difference=parity["maximum_absolute_difference"], device=device, threads=2,
                    hardware=platform.processor() if device == "cpu" else torch.cuda.get_device_name(), scope=comparable.SCOPE,
                    dtype="FP64 Ridge core; FP32 normalization; FP64 inverse; CPU return" if method == "Ridge" else "FP32 core; FP64 inverse; CPU return",
                    core_dtype="float64" if method == "Ridge" else "float32", inverse_transform_dtype="float64", physical_output_dtype="float64",
                    physical_input_sha256=data.array_sha256(physical), mask_sha256=data.array_sha256(observed), origins=selected.tolist(),
                    C=170, K=43 if method == "B2" else 170, L=512, H=12, logical_F_rows=expected,
                    trainable_params=fitted_params, fitted_trainable_params=fitted_params,
                    deployed_trainable_params=0 if method == "LoRA" else fitted_params,
                    total_resident_parameters=foundation_params+fitted_params,
                    resident_parameter_scope="F + fitted coefficients; LoRA CPU adapter retained, restore snapshots in RSS",
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                    process_RSS_peak_sampled_bytes=rss["process_RSS_sampled_peak_bytes"], RSS_sampling_ms=5,
                    RSS_scope="per-method resident process, this B warmup+timed block and post-timer CPU parity checks; excludes full V replay/cold load",
                    cold_load_seconds=cold_load, cold_load_scope="this method only; excludes merge/parity; CUDA context and OS cache may exist",
                    MSE=score["MSE"], MAE=score["MAE"], device_replica_MSE=score["MSE"], device_replica_MAE=score["MAE"],
                    accuracy_source="complete physical device replica V148 saved replay; original selected reference preserved separately; no new fit",
                    routing="own_channel_history" if method == "Ridge" else "native_multivariate_per_origin" if native is not None else "independent_univariate",
                    row_microbatch=None if native is not None or method == "Ridge" else 128,
                    F_snapshot_revision=recipe.get("model_revision", Path(MODEL_REL).name), selected_epoch=recipe["selected_epoch"],
                    lora_state="merged" if method == "LoRA" else None, dataset="pems08", cohort="pems08", split="V_select"))
                _write_trials(trial_path, trials)
                atomic_json(output, value)
        except (comparable.ExclusiveCostError, TimeoutError):
            raise
        except Exception as error:
            diagnostic = Path(ctx["cache"])/"errors"/("E04_PEMS08_"+method+".json")
            atomic_json(diagnostic, dict(error_type=type(error).__name__, traceback=traceback.format_exc()))
            rows[:] = [row for row in rows if row["method"] != method]
            rows.extend(dict(method=method, source_method=method, B=batch,
                status="BLOCKED_MISSING_ASSET" if isinstance(error, FileNotFoundError) else "FAILED",
                parity="FAILED_OR_UNVERIFIED", median_ms_batch=None, median_ms_origin=None,
                reason="Route "+type(error).__name__+"; private diagnostic retained", cohort="pems08") for batch in (1, 8))
        finally:
            predict = None
            if route is not None:
                route.close()
                del route
            if native is not None:
                native.assert_frozen_unchanged()
                del native
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(output, value)
    value["status"] = "COMPLETE" if len(rows) == 10 and all(row["status"] == "COMPLETE" for row in rows) else "FAILED"
    value["backbone_exact_restore_at_close"] = "PASS_FOR_LOADED_ROUTES"
    _write_trials(trial_path, trials)
    atomic_json(output, value)
    return dict(status=value["status"], reason="Exclusive PeMS170 native and baseline physical paths with complete V148 saved-prediction parity",
                artifact_paths=[str(output), str(trial_path)], result=value)


def execute_job(job, ctx):
    if job.get("question_id") != "E04" or job.get("device") != "cuda":
        raise ValueError("E04 supports only the registered CUDA-owned comparison jobs")
    registered_routes(job)
    if ctx.get("lease") is None:
        raise RuntimeError("E04 requires the active parent campaign GPU lease")
    cohort = job.get("cohort", job.get("dataset_cohort", "jena"))
    if cohort == "pems08":
        return _execute_pems(job, ctx)
    result = _execute_jena(job, ctx)
    result["result"]["full_v_replay"] = {"LoRA": result["result"]["full_v_replay"]}
    from research.adapter_diagnostics_20261017_v2.core import atomic_json
    atomic_json(Path(ctx["report"])/"costs/E04_JENA_LORA.json", result["result"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cohort", choices=("jena", "pems08"), default="jena")
    args = parser.parse_args(argv)
    if not args.inspect:
        print(json.dumps(dict(status="SOURCE_ONLY_NOOP", model_loads=0, F_calls=0)))
        return 0
    if args.asset_root is None:
        parser.error("--inspect requires --asset-root")
    ctx = dict(repo=args.repo, asset=args.asset_root,
               cache=args.cache_root or args.repo/".cache"/NAMESPACE)
    print(json.dumps((inspect_assets if args.cohort == "jena" else inspect_pems_assets)(ctx), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
