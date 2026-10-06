"""Canonical VAL step0 and zero-update autograd/resource gate; never reads TEST."""
import gc
import inspect
import math
import time
import traceback

import numpy as np
import torch

from model_v18 import (GRADIENT_PARITY, PARITY, REPLAY, build_model, state_hash, tensor_hashes)
from train_v18 import (PHASE_PERIODS, backward_effective_batch, evaluate, sample_epoch_origins,
                       validate_protocol)
from runtime_v18 import (CACHE, DATASETS, HERE, LIMITS, ROOT, SEEDS, Job, artifact, array_hash,
                         budget_snapshot, configure, dataset_contract, environment, import_file, inputs,
                         load_data, load_original, original_levels, read_json, save_json, targets)


PARENT_SAMPLERS = {
    "robin": ("research/tsfm_peft_level_confirmation_v6_20260928/runtime_v6.py", "phase_sample"),
    "peacock_education": ("research/tsfm_peft_internal_vs_subspace_v11_20260930/runtime_v11.py", "phase_sample"),
    "jena": ("research/tsfm_peft_practical_controls_v7_20260928/runtime_v7.py", "phase_sample"),
}
FIXED_WARMUP_BACKWARD = 2
FIXED_TIMED_BACKWARD = 3
PROJECTION_FACTOR = 1.25
OPTIMIZER_CPU_ALLOWANCE_S_PER_UPDATE = 0.002
CHECKPOINT_ALLOWANCE_S_PER_EPOCH = 0.02


def source_artifacts():
    names = ("REQUEST.txt", "protocol.json", "parent_reuse_manifest.json", "runtime_v18.py",
             "model_v18.py", "train_v18.py", "preflight_model_v18.py")
    return [artifact(HERE / name) for name in names] + [artifact(HERE / f"data_contract_{d}.json") for d in DATASETS]


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP_DEBUG: " + message)


def parity(actual, reference, tolerance):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32,
            "Prediction shape/dtype differs")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "Nonfinite prediction")
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denominator = np.maximum(np.abs(reference.astype(np.float64)), np.finfo(np.float32).tiny)
    return {"pass": bool(np.all(error <= tolerance["atol"] + tolerance["rtol"] * np.abs(reference))),
            "max_absolute": float(error.max()), "max_relative": float((error / denominator).max()),
            "atol": tolerance["atol"], "rtol": tolerance["rtol"], "shape": list(actual.shape), "dtype": "float32"}


@torch.no_grad()
def original_compress(original, data, origins, job):
    values = []
    original.eval()
    for start in range(0, len(origins), 4):
        job.check_limits()
        x = inputs(data, origins[start:start + 4]).to(next(original.parameters()).device)
        main, _ = original.components(x)
        values.append(main.detach().cpu().contiguous().numpy())
        del x, main
    return np.concatenate(values)


def sampler_checks(dataset, data):
    path, name = PARENT_SAMPLERS[dataset]
    module = import_file(ROOT / path, f"matched_v18_parent_sampler_{dataset}")
    parent = getattr(module, name)
    checks = []
    for seed in SEEDS:
        row = next(row for row in original_levels(dataset) if row["seed"] == seed)
        saved = read_json(row["source_result"]["path"])
        saved_schedule = {record["epoch"]: record["sha256"] for record in saved["schedules"]}
        for epoch in range(1, 121):
            actual = sample_epoch_origins(dataset, seed, epoch)
            arguments = {"n": 512}
            if dataset != "robin":
                arguments["period"] = PHASE_PERIODS[dataset]
            expected = parent(data["train_origins"], seed, epoch, **arguments)
            require(np.array_equal(actual, expected), "Seed/epoch origin order differs from literal parent sampler")
            hashed = array_hash(actual.astype("<i8"))
            if epoch in saved_schedule:
                require(hashed == saved_schedule[epoch], "Recorded parent epoch origin hash differs")
            checks.append({"seed": seed, "epoch": epoch, "sha256": hashed,
                           "recorded_parent_hash_checked": epoch in saved_schedule})
    return {"source": artifact(ROOT / path), "function": name, "phase_period": PHASE_PERIODS[dataset],
            "checks": checks, "all_240_seed_epoch_schedules_literal_parent_identity": True}


def probe_gradients(model, data, origins, job):
    x_cpu = inputs(data, origins)
    target, mask = targets(data, origins)
    records, gradients, oom = [], {}, []
    for microbatch in (4, 2, 1):
        job.check_limits()
        model.zero_grad(set_to_none=True)
        try:
            loss, record = backward_effective_batch(model, x_cpu, target, mask, microbatch)
            gradients[microbatch] = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
            records.append({"microbatch": microbatch, "status": "complete", "loss": loss, "gradients": record,
                            "effective_batch_origins": 4, "optimizer_updates": 0})
        except torch.cuda.OutOfMemoryError as error:
            oom.append({"microbatch": microbatch, "status": "oom", "error": str(error), "optimizer_updates": 0})
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    if not gradients:
        raise RuntimeError("STOP_RESOURCE: frozen-backbone encoder autograd impossible at microbatch1")
    selected = max(gradients)
    reference = gradients[selected]
    require(all(torch.count_nonzero(value) > 0 for value in reference.values()), "Both E/D must receive nonzero gradients")
    differences = []
    reference_loss = next(row["loss"] for row in records if row["microbatch"] == selected)
    for microbatch, current in gradients.items():
        for name, expected in reference.items():
            torch.testing.assert_close(current[name], expected, **GRADIENT_PARITY)
            differences.append({"microbatch": microbatch, "parameter": name,
                                "max_absolute": float((current[name] - expected).abs().max()),
                                "atol": GRADIENT_PARITY["atol"], "rtol": GRADIENT_PARITY["rtol"]})
        current_loss = next(row["loss"] for row in records if row["microbatch"] == microbatch)
        require(abs(current_loss - reference_loss) <= 1e-6 + 1e-4 * abs(reference_loss),
                "Effective4 macro loss differs under observed-count microbatch accumulation")
    elapsed = []
    for index in range(FIXED_WARMUP_BACKWARD + FIXED_TIMED_BACKWARD):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        backward_effective_batch(model, x_cpu, target, mask, selected)
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if index >= FIXED_WARMUP_BACKWARD:
            elapsed.append(duration)
        job.heartbeat({"phase": "zero_update_forward_backward_probe", "measurement": index,
                       "microbatch": selected}, optimizer_updates=0)
    model.zero_grad(set_to_none=True)
    return {"origins": origins.tolist(), "origin_values_sha256": array_hash(origins),
            "input_values_sha256": array_hash(x_cpu.numpy()), "target_values_sha256": array_hash(target),
            "mask_values_sha256": array_hash(mask), "microbatch_records": records, "oom_records": oom,
            "effective_batch_origins": 4, "selected_feasible_microbatch": selected,
            "gradient_parity": differences, "full_effective_count_denominator": True,
            "full_microbatch4_gradient_observed": 4 in gradients,
            "warmup": FIXED_WARMUP_BACKWARD, "timed_repetitions": FIXED_TIMED_BACKWARD,
            "forward_backward_seconds": elapsed, "optimizer_updates": 0, "backbone_gradients_absent": True}


def one_dataset(dataset, job):
    data = load_data(dataset, include_test=False)
    contract = dataset_contract(dataset)
    origins = data["val_origins"]
    require(array_hash(origins) == contract["origins"]["val"]["values_sha256"], "Canonical VAL origin hash differs")
    parents = read_json(HERE / "parent_reuse_manifest.json")["units"][dataset]
    require(data["basis"].dtype == np.float32 and array_hash(data["basis"]) == parents["basis"]["sha256"],
            "Original TRAIN PCA bytes differ")
    schedules = sampler_checks(dataset, data)
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = build_model(dataset, device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    initial = model.adapter_state()
    adapter_initial_hash = tensor_hashes(initial)
    before = state_hash(model)
    backbone_before = state_hash(model.backbone)
    require(torch.equal(initial["encoder.weight"], torch.as_tensor(data["basis"]).T)
            and torch.equal(initial["decoder.weight"], torch.as_tensor(data["basis"])), "E/D PCA orientation differs")
    expected_count = {"robin": 170, "peacock_education": 104, "jena": 252}[dataset]
    receipt = model.receipt(include_state_hash=True)
    require(receipt["trainable_parameter_count"] == expected_count and receipt["all_parameters_fp32"],
            "Actual trainable E/D/FP32 parameter enumeration differs")
    require(receipt["quantiles"][receipt["median_index"]] == 0.5 and receipt["backbone_eval_mode"],
            "Native median/backbone eval contract differs")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    val, prediction = evaluate(model, data, origins, job)
    torch.cuda.synchronize()
    full_val_s = time.perf_counter() - started
    original = load_original(original_levels(dataset)[0], device="cuda", deploy=True)
    require(state_hash(original.backbone) == backbone_before, "Original and matched E/D backbone states differ")
    original_backbone_before = state_hash(original.backbone)
    reference = original_compress(original, data, origins, job)
    step0 = parity(prediction, reference, PARITY)
    require(step0["pass"], "Step0 E/D differs from existing literal fixed PCA COMPRESS b")
    require(original_backbone_before == state_hash(original.backbone), "Original COMPRESS comparison changed backbone")
    del original
    gc.collect()
    torch.cuda.empty_cache()
    local = CACHE / "preflight" / dataset
    local.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(local / "step0_val.npz", prediction=prediction, compress=reference, origins=origins)
    first = origins[:4]
    _, replay = evaluate(model, data, first, job)
    replay_check = parity(replay, prediction[:4], REPLAY)
    require(replay_check["pass"], "Step0 replay differs")
    _, single = evaluate(model, data, first, job, batch_size=1)
    merge_check = parity(single, replay, PARITY)
    require(merge_check["pass"], "Step0 origin batching/order differs")
    probe = probe_gradients(model, data, data["train_origins"][:4], job)
    model.eval()
    after = state_hash(model)
    backbone_after = state_hash(model.backbone)
    require(before == after and backbone_before == backbone_after and tensor_hashes(model.adapter_state()) == adapter_initial_hash,
            "Zero-update preflight changed model/E/D/PCA/backbone state")
    peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "ED step0 validation and forward/backward feasibility; transient original COMPRESS comparison included"}
    receipt = model.receipt(include_state_hash=True)
    result = {"dataset": dataset, "data": artifact(data["_path"]), "contract": artifact(HERE / f"data_contract_{dataset}.json"),
              "literal_PCA": parents["basis"], "adapter_initial_sha256": adapter_initial_hash,
              "model_receipt": receipt, "canonical_VAL_origins": len(origins),
              "VAL_origin_values_sha256": array_hash(origins), "step0_COMPRESS_parity": step0,
              "step0_full_VAL_score": val, "step0_prediction": artifact(local / "step0_val.npz"),
              "replay_parity": replay_check, "single_B4_origin_order_parity": merge_check,
              "sampling": schedules, "autograd_probe": probe,
              "state_preservation": {"before_sha256": before, "after_sha256": after, "unchanged": True,
                                     "backbone_before_sha256": backbone_before, "backbone_after_sha256": backbone_after,
                                     "optimizer_updates": 0}, "memory_peak": peak,
              "measurement": {"model_load_s": load_s, "full_canonical_VAL_s": full_val_s,
                              "VAL_batch_origins": 4, "all_canonical_VAL_origins_used": True}}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def estimate(units, gpu_used_s):
    estimates = {}
    post_training = 0.0
    for dataset, row in units.items():
        step = max(row["autograd_probe"]["forward_backward_seconds"])
        validation = row["measurement"]["full_canonical_VAL_s"]
        load = row["measurement"]["model_load_s"]
        train_epochs = 120 * 128 * (PROJECTION_FACTOR * step + OPTIMIZER_CPU_ALLOWANCE_S_PER_UPDATE)
        all_full_val = 121 * PROJECTION_FACTOR * validation
        checkpoints = 121 * CHECKPOINT_ALLOWANCE_S_PER_EPOCH
        fit_upper = 2 * load + train_epochs + all_full_val + checkpoints + PROJECTION_FACTOR * validation
        contract = dataset_contract(dataset)
        test_origins = contract["origins"]["test_a"]["count"] + contract["origins"]["test_b"]["count"]
        inference_per_origin = validation / row["canonical_VAL_origins"]
        selected_test = 2 * test_origins * inference_per_origin * PROJECTION_FACTOR
        inference_calls = 4 * 3 * sum(10 * b + 10 * 24 for b in (1, 4))
        inference_cost = inference_calls * inference_per_origin * PROJECTION_FACTOR * 4
        training_probe = 4 * 3 * (10 + 10) * step * PROJECTION_FACTOR * 2
        diagnostics = 4 * PROJECTION_FACTOR * validation
        post_unit = 40 * load + selected_test + inference_cost + training_probe + diagnostics
        post_training += post_unit
        estimates[dataset] = {"measured_max_forward_backward_effective4_s": step,
                              "measured_whole_canonical_VAL_s": validation,
                              "canonical_VAL_origins": row["canonical_VAL_origins"],
                              "full_epochs_without_early_stopping": 120, "optimizer_updates_per_epoch": 128,
                              "VAL_passes_including_step0": 121, "fit_upper_s": fit_upper,
                              "four_fits_upper_s": 4 * fit_upper, "selected_TEST_upper_s": selected_test,
                              "inference_cost_upper_s": inference_cost, "training_probe_upper_s": training_probe,
                              "diagnostics_upper_s": diagnostics, "post_training_model_load_allowance": 40,
                              "post_training_upper_s": post_unit}
    projected = gpu_used_s + sum(row["four_fits_upper_s"] for row in estimates.values()) + post_training
    return {"units": estimates, "projection_factor": PROJECTION_FACTOR,
            "optimizer_CPU_allowance_s_per_update": OPTIMIZER_CPU_ALLOWANCE_S_PER_UPDATE,
            "checkpoint_allowance_s_per_epoch": CHECKPOINT_ALLOWANCE_S_PER_EPOCH,
            "measured_preflight_gpu_job_s": gpu_used_s, "post_training_upper_s": post_training,
            "projected_full_campaign_upper_s": projected, "hard_cap_s": LIMITS["gpu_seconds"],
            "assumed_early_stopping": False, "scope_reduced": False,
            "interpretation": "Conservative projection from fixed measured steps, all full VAL passes and post-training scope; hard cumulative watchdog remains authoritative"}


def main():
    configure()
    validate_protocol()
    before_budget = budget_snapshot()
    if before_budget["new_fits"] or before_budget["new_test_predictions"]:
        raise RuntimeError("Preflight must precede every new fit/TEST prediction")
    if (HERE / "preflight.json").exists():
        raise RuntimeError("Existing preflight attempt must be preserved; inspect rather than rerun")
    sources = source_artifacts()
    attempt_path = HERE / "preflight_attempt01.json"
    attempt = {"schema": "matched_learned_ed_v18_preflight_1", "status": "running", "units": {},
               "source_artifacts": sources, "new_fits": 0, "optimizer_updates": 0, "test_accessed": False,
               "environment": environment(), "independent_reproduction": False}
    save_json(attempt_path, attempt)
    try:
        with Job("gpu", "step0_autograd_full_scope_resource_preflight", metadata={"optimizer_updates": 0, "new_fits": 0}) as job:
            for dataset in DATASETS:
                attempt["units"][dataset] = one_dataset(dataset, job)
                save_json(attempt_path, attempt)
                print(f"preflight {dataset} step0/gradient/sampler PASS", flush=True)
            for receipt in sources:
                artifact(receipt["path"], receipt["sha256"])
            initial = {"schema": "matched_learned_ed_v18_literal_initial_models_1",
                       "units": {d: {"literal_PCA": row["literal_PCA"], "adapter_initial_sha256": row["adapter_initial_sha256"],
                                     "backbone_state_sha256": row["model_receipt"]["backbone_state_sha256"],
                                     "model_receipt": row["model_receipt"]} for d, row in attempt["units"].items()},
                       "same_literal_ED_for_both_seeds_and_LRs": True, "new_fits": 0, "optimizer_updates": 0}
            save_json(HERE / "preflight_initial_models.json", initial)
            attempt["initial_models"] = artifact(HERE / "preflight_initial_models.json")
            attempt["estimate"] = estimate(attempt["units"], budget_snapshot()["gpu_seconds"])
            attempt.update(all_checks_passed=True, source_hashes_unchanged=True,
                           status="pass" if attempt["estimate"]["projected_full_campaign_upper_s"] < LIMITS["gpu_seconds"] else "STOP_RESOURCE")
            save_json(attempt_path, attempt)
            if attempt["status"] != "pass":
                raise RuntimeError("STOP_RESOURCE: projected full twelve120epochfits plus TEST/cost exceeds three hours")
    except BaseException as error:
        attempt.update(status="STOP_RESOURCE" if "STOP_RESOURCE" in str(error) else "STOP_DEBUG",
                       error=traceback.format_exc(), all_checks_passed=False)
        save_json(attempt_path, attempt)
        save_json(HERE / "preflight.json", attempt)
        raise
    attempt["terminal_budget"] = budget_snapshot()
    save_json(attempt_path, attempt)
    save_json(HERE / "preflight.json", attempt)
    print(f"preflight PASS full scope upper={attempt['estimate']['projected_full_campaign_upper_s']:.1f}s cap=10800s", flush=True)


if __name__ == "__main__":
    main()
