"""CPU-only, read-only cross-checks of sealed scientific and execution evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from . import analysis as a
from . import common as c
from . import data as d
from . import parents as p


CORE_NAMES = ("common.py", "data.py", "model.py", "parents.py", "feature_cache.py",
              "analysis.py", "training.py", "execution.py", "campaign.py")
ARMS = ("GT_FT_L512", "GT_FT_L1024", "GT_TS_PRE_L1024", "GT_TS_RAND_L1024")
FIELDS = ("prediction", "target", "mask", "origins", "mean", "std")
FINAL_EVIDENCE = ("EXPERIMENT_CONTRACT.json", "START_STATE.json", "UPDATE_DECISION.json",
                  "PHASE_B_ADMISSION.json", "WORKER_PHASE_B_ADMISSION.json", "PARENT_BINDINGS.json",
                  "PARENT_AND_GRADIENT_VALIDATION.json", "SAMPLING_SCHEDULES.json", "SOURCE_AMENDMENTS.json",
                  "FINAL_RESOURCE_SNAPSHOT.json", "TEST_VALIDATION.json", "PUBLIC_REPLAY_VALIDATION.json",
                  "RUNS.csv", "COMPARISONS.csv", "CHANNEL_HORIZON_STATS.csv", "ORIGIN_EFFECTS.csv",
                  "COST.csv", "COST_TRIALS.csv", "TRAINING_CURVES.csv", "JOB_STATUS.csv", "RESOURCE_LEDGER.json",
                  "OBSERVATION_CONTEXT_STATS.csv", "OBSERVATION_DIAGNOSTIC.json")


def expected_runs():
    rows = [dict(run_id=f"A_RAW_L{length}", phase="A", mode="raw", seed=0, L=length, updates=0)
            for length in (512, 1024, 2048)]
    for seed in (92601, 92602):
        rows += [dict(run_id=f"A0_{key}_L512_s{seed}", phase="A0", mode=mode, seed=seed, L=512, updates=0)
                 for key, mode in (("GT", "gt"), ("TS2", "ts2"))]
        rows += [dict(run_id=f"A_GT_L{length}_s{seed}", phase="A", mode="gt", seed=seed, L=length, updates=0)
                 for length in (512, 1024, 2048)]
        rows += [dict(run_id=f"A_{mode.upper()}_L1024_s{seed}", phase="A", mode=mode, seed=seed, L=1024, updates=0)
                 for mode in ("ts2", "ensemble")]
        for arm in ARMS:
            mode = "pre" if "PRE" in arm else "rand" if "RAND" in arm else "gt"
            for prefix, phase, updates in (("B_", "B", 300), ("CP0_B_", "CP0", 0)):
                rows.append(dict(run_id=f"{prefix}{arm}_s{seed}", phase=phase, mode=mode, seed=seed,
                                 L=512 if arm == "GT_FT_L512" else 1024, updates=updates, arm=arm))
    for row in rows:
        if row["phase"] == "A0":
            row["arm"] = "HISTORICAL_CALIBRATED_F_REPRODUCTION"
        elif row["phase"] == "A":
            row["arm"] = {"raw": "RAW_F", "gt": "GT_FROZEN", "ts2": "TS2_NATIVE_D",
                          "ensemble": "OUTPUT_ENSEMBLE_CONTEXT_ONLY"}[row["mode"]]
    return rows


def require_exact_ids(actual, expected, label):
    actual, expected = list(actual), list(expected)
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra or len(set(actual)) != len(actual):
        raise ValueError(f"{label} coverage missing={missing}, extra={extra}, duplicate={len(set(actual)) != len(actual)}")


def readiness():
    expected = expected_runs()
    missing = [name for name in FINAL_EVIDENCE if not (c.REPORT / name).is_file()]
    categories = {"predictions": ("receipts", [row["run_id"] for row in expected]),
                  "training": ("training", [row["run_id"] for row in expected if row["phase"] == "B"]),
                  "features": ("features", [f"{kind}_s{seed}" for seed in (92601, 92602) for kind in ("PRE", "RAND")]),
                  "costs": ("cost_receipts", [f"{row['run_id']}_B{batch}" for row in expected
                                              if row["phase"] in ("A", "B") for batch in (1, 8)])}
    counts = {}
    for label, (directory, identifiers) in categories.items():
        available = [identifier for identifier in identifiers if (c.REPORT / directory / f"{identifier}.json").is_file()]
        counts[label] = {"present": len(available), "required": len(identifiers)}
        missing.extend(f"{directory}/{identifier}.json" for identifier in identifiers if identifier not in available)
    return {"status": "READY" if not missing else "INCOMPLETE", "counts": counts,
            "missing_evidence": missing, "writes_performed": False, "GPU_execution": False}


def independent_score(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if (prediction.ndim != 3 or prediction.shape != target.shape or mask.shape != target.shape
            or mask.dtype != np.bool_ or not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all()):
        raise ValueError("Independent score array shape/finite/mask failure")
    sse, sae, counts = [], [], []
    for channel in range(prediction.shape[-1]):
        observed = mask[..., channel]
        error = prediction[..., channel][observed].astype(np.float64) - target[..., channel][observed].astype(np.float64)
        if not len(error):
            raise ValueError("Independent score needs observations in every fixed channel")
        sse.append(float(np.dot(error, error)))
        sae.append(float(np.abs(error).sum(dtype=np.float64)))
        counts.append(len(error))
    return {"mse": float(np.mean(np.asarray(sse) / counts)), "mae": float(np.mean(np.asarray(sae) / counts)),
            "channel_sse": sse, "channel_sae": sae, "channel_count": counts,
            "channel_mse": (np.asarray(sse) / counts).tolist(), "channel_mae": (np.asarray(sae) / counts).tolist()}


def _close(actual, expected, label, *, rtol=1e-12, atol=1e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError(label + " differs")


def verify_prediction(receipt, bundle, input_history, cohort):
    if input_history.shape != (receipt["origin_count"], receipt["L"], receipt["C"]):
        raise ValueError("Prediction input window shape differs")
    if c.array_sha(input_history) != receipt["input_sha256"]:
        raise ValueError("Prediction input window fingerprint differs")
    if bundle["prediction"].shape != (receipt["origin_count"], receipt["H"], receipt["C"]):
        raise ValueError("Prediction shape differs from registered receipt")
    for key in FIELDS:
        if c.array_sha(bundle[key]) != receipt[key + "_sha256"]:
            raise ValueError("Prediction array fingerprint differs: " + key)
        if key != "prediction" and (bundle[key].dtype != cohort[key].dtype
                                     or not np.array_equal(bundle[key], cohort[key], equal_nan=True)):
            raise ValueError("Prediction cohort differs: " + key)
    measured = independent_score(bundle["prediction"], bundle["target"], bundle["mask"])
    producer = a.score(bundle["prediction"], bundle["target"], bundle["mask"])
    for upper, lower in (("MSE", "mse"), ("MAE", "mae")):
        _close(receipt[upper], measured[lower], "Receipt metric " + upper)
        _close(producer[lower], measured[lower], "Producer/independent metric " + upper)
    for key in ("channel_sse", "channel_sae", "channel_count"):
        _close(producer[key], measured[key], "Producer/independent metric " + key)
    return measured


def verify_frozen_states(before, after, foundation_sha):
    if before != after or before.get("F") != foundation_sha:
        raise ValueError("Frozen F/TS2/adapter tensor states changed or differ from the canonical F")


def expected_F_work(spec, batch, maximum_context=2048):
    mode, length = spec["mode"], spec["L"]
    components = [(length, 21)] if mode == "raw" else [(length // 4, 42)] if mode in ("gt", "pre", "rand") else [(length, 320)]
    if mode == "ensemble":
        components.append((length // 4, 42))
    return {"requested": [length for length, _ in components],
            "actual": [min(length, maximum_context) for length, _ in components],
            "streams": sum(batch * width for _, width in components),
            "calls": sum(math.ceil(batch * width / 128) for _, width in components),
            "components": [{"streams": batch * width,
                            "row_sizes": [min(128, batch * width - start) for start in range(0, batch * width, 128)]}
                           for _, width in components]}


def _inventory(value):
    inventory = value["resident_inventory"]
    totals = {device: {field: 0 for field in ("parameters", "parameter_bytes", "buffer_bytes")}
              for device in ("cpu", "cuda")}
    for devices in inventory["by_module"].values():
        if set(devices) != {"cpu", "cuda"}:
            raise ValueError("Resident inventory must describe CPU and CUDA separately")
        for device, row in devices.items():
            for field in totals[device]:
                if not isinstance(row[field], int) or row[field] < 0:
                    raise ValueError("Resident inventory counter must be a nonnegative integer")
                totals[device][field] += row[field]
    if totals != inventory["by_device"]:
        raise ValueError("Resident module/device inventory does not reaggregate")
    for total, field in (("total_unique_parameters", "parameters"),
                         ("total_unique_parameter_bytes", "parameter_bytes"),
                         ("total_unique_buffer_bytes", "buffer_bytes")):
        if inventory[total] != sum(row[field] for row in totals.values()):
            raise ValueError("Resident inventory system total differs")
    summary = value["summary"]
    if summary["resident_inventory"] != inventory or summary["resident_parameters"] != totals["cuda"]["parameters"]:
        raise ValueError("Summary resident count must use actual unique CUDA parameters")
    for prefix, device in (("CUDA", "cuda"), ("CPU", "cpu")):
        for suffix, field in (("parameters", "parameters"), ("parameter_bytes", "parameter_bytes"), ("buffer_bytes", "buffer_bytes")):
            if summary[f"{prefix}_resident_{suffix}"] != totals[device][field]:
                raise ValueError("Summary resident device counter differs")
    if summary["system_resident_parameters"] != inventory["total_unique_parameters"]:
        raise ValueError("Summary system resident count differs")
    return inventory


def verify_cost_receipt(value, spec, origins, contract, foundation_sha, prediction_receipt=None, prediction_bundle=None):
    if value.get("TS2cache_used") is not False:
        raise ValueError("Cost path must use full online TS2, without a feature cache")
    if value["status"] != "COMPLETE" or value["validation"]["status"] != "PASS":
        raise ValueError("Cost receipt is not complete and validated")
    if any(value["spec"].get(key) != expected for key, expected in spec.items()):
        raise ValueError("Cost model/update specification differs")
    batch = value["batch"]
    if batch not in (1, 8) or value["F_row_microbatch"] != 128:
        raise ValueError("Unregistered cost batch/F row microbatch")
    if value["TS2_present"] != (spec["mode"] in ("pre", "rand", "ts2", "ensemble")):
        raise ValueError("Cost TS2 residence differs from the actual route")
    if value["TS2_pretrained_weights_used"] != (spec["mode"] in ("pre", "ts2", "ensemble")):
        raise ValueError("Cost TS2 pretrained/RAND identity differs")
    verify_frozen_states(value["states_before_sha256"], value["states_after_sha256"], foundation_sha)
    if not all(value["validation"]["parameter_gradients_absent"].values()):
        raise ValueError("Frozen cost models accumulated parameter gradients")
    if value["validation"]["F_unchanged"] is not True or value["validation"]["TS2_unchanged"] is not True:
        raise ValueError("Cost model state invariant failed")
    work = expected_F_work(spec, batch)
    starts = contract["cost"]["fixed_group_indices"][str(batch)]
    trials = value["trials"]
    if len(trials) != 30:
        raise ValueError("Cost needs exactly 10 warmup and 20 timed repetitions")
    for position, row in enumerate(trials):
        stage, iteration = ("warmup", position) if position < 10 else ("timed", position - 10)
        indices = np.arange(starts[iteration % 3], starts[iteration % 3] + batch)
        if (row["stage"] != stage or row["iteration"] != iteration or row["group_index"] != iteration % 3
                or row["groupstart"] != indices[0] or row["batch"] != batch
                or row["origins_sha256"] != c.array_sha(origins[indices])):
            raise ValueError("Cost fixed origin group or repetition order differs")
        for key in ("run_id", "seed", "phase", "L", "updates"):
            if row[key] != spec[key]:
                raise ValueError("Cost trial model/update metadata differs")
        if row["TS2cache_used"] is not False or row["prediction_parity"]["status"] != "PASS":
            raise ValueError("Cost trial online/parity evidence failed")
        if (row["prediction_parity"]["atol"], row["prediction_parity"]["rtol"]) != (1e-5, 1e-4):
            raise ValueError("Cost trial changed the registered forward parity tolerance")
        if row["F_calls"] != work["calls"] or row["F_streams"] != work["streams"]:
            raise ValueError("Actual F physical calls/streams differ from this route")
        if row["requested_context_length"] != work["requested"] or row["actual_context_length"] != work["actual"]:
            raise ValueError("Cost actual/requested F context differs")
        if len(row["F_context_records"]) != len(work["components"]):
            raise ValueError("Cost omitted a real F invocation context")
        for context, expected, requested, actual in zip(row["F_context_records"], work["components"], work["requested"], work["actual"]):
            if context != {**expected, "requested_context_length": requested, "actual_context_length": actual}:
                raise ValueError("Cost F row splitting/context receipt differs")
        if not np.isfinite(row["whole_batch_ms"]) or row["whole_batch_ms"] <= 0:
            raise ValueError("Cost duration must be finite and positive")
        _close(row["origins_per_second"], 1000 * batch / row["whole_batch_ms"], "Cost throughput")
        if (row["peak_allocated_MiB"] < value["loaded_baseline"]["allocated_MiB"]
                or row["peak_reserved_MiB"] < row["peak_allocated_MiB"] or row["RSS_MiB"] <= 0):
            raise ValueError("Cost memory omitted resident models or mixed allocated/reserved")
        expected_scope = "one timed repetition including all resident models" if stage == "timed" else "cumulative warmup phase"
        if row["memory_peak_scope"] != expected_scope:
            raise ValueError("Cost peak reset scope differs")
        if row["pid"] != value["pid"] or row["process_create_time_unix"] != value["process_create_time_unix"]:
            raise ValueError("Cost trial belongs to a different OS process")
        if row["checkpoint_sha256"] != value["checkpoint_sha256"] or row["loaded_source_sha256"] != value["loaded_source_sha256"]:
            raise ValueError("Cost trial source/checkpoint identity differs")
        if prediction_bundle is not None:
            if row["prediction_parity"]["reference_sha256"] != c.array_sha(prediction_bundle["prediction"][indices]):
                raise ValueError("Cost parity reference is not the same fixed origin group")
    timed = trials[10:]
    q25, median, q75, p95 = np.quantile([row["whole_batch_ms"] for row in timed], [.25, .5, .75, .95])
    for key, expected in (("p50_ms", median), ("q25_ms", q25), ("q75_ms", q75),
                          ("iqr_ms", q75 - q25), ("p95_ms", p95), ("origins_per_second_at_median", 1000 * batch / median)):
        _close(value["summary"][key], expected, "Cost summary " + key)
    for key in ("peak_allocated_MiB", "peak_reserved_MiB", "RSS_MiB"):
        _close(value["summary"][key], max(row[key] for row in timed), "Timed memory summary")
        _close(value["warmup_peaks"][key], max(row[key] for row in trials[:10]), "Warmup memory summary")
    if value["summary"]["repetitions"] != 20 or value["summary"]["updates"] != spec["updates"]:
        raise ValueError("Cost summary final update/repetition count differs")
    _inventory(value)
    if prediction_receipt is not None:
        for key, reference_key in (("reference_prediction_sha256", "prediction_sha256"),
                                    ("reference_artifact_sha256", "artifact_sha256")):
            if value[key] != prediction_receipt[reference_key]:
                raise ValueError("Cost reference belongs to a different prediction artifact")
    return {"run_id": spec["run_id"], "batch": batch, "trials": len(trials), "status": "PASS"}


def _private_reference(value):
    if not isinstance(value, str) or not value.startswith("cache://"):
        raise ValueError("Artifact must use the private campaign cache reference")
    relative = Path(value[8:])
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError("Private artifact reference escapes the campaign")
    path = (c.CACHE / relative).resolve()
    if not path.is_relative_to(c.CACHE.resolve()):
        raise ValueError("Private artifact symlink escapes the campaign")
    return path


def _state_dict_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sources(identity, required, expected):
    normalized = {Path(name).name: sha for name, sha in identity.items()}
    for name in required:
        if normalized.get(name) != expected[name]:
            raise ValueError("Used scientific producer source differs: " + name)


def _source_history(core):
    amendments = c.read_json(c.REPORT / "SOURCE_AMENDMENTS.json")
    if amendments["status"] != "EXACT_HISTORICAL_SOURCE_PRESERVED":
        raise ValueError("Historical producer source evidence is incomplete")
    historical = dict(core)
    for amendment in amendments["amendments"]:
        if (amendment["module"] != "execution.py" or amendment["after_sha256"] != core["execution.py"]
                or amendment["parent_reproduction_evidence_preserved"] is not True):
            raise ValueError("Only the exactly preserved A0 execution source amendment is allowed")
        path = (c.REPORT / amendment["historical_file"]).resolve()
        if not path.is_relative_to(c.REPORT.resolve()) or c.file_sha(path) != amendment["before_sha256"]:
            raise ValueError("Historical A0 execution source bytes differ")
        historical["execution.py"] = amendment["before_sha256"]
    if historical["execution.py"] == core["execution.py"]:
        raise ValueError("A0 historical execution version was not bound")
    return historical


def _rng_fingerprint(saved):
    required = ("torch_rng_state", "cuda_rng_state", "numpy_rng_state", "python_rng_state")
    if any(key not in saved for key in required):
        raise ValueError("Checkpoint lacks a complete RNG snapshot")
    generator = torch.Generator(device="cpu")
    generator.set_state(saved["torch_rng_state"])
    local_numpy = np.random.RandomState()
    local_numpy.set_state(saved["numpy_rng_state"])
    local_python = random.Random()
    local_python.setstate(saved["python_rng_state"])
    cuda = saved["cuda_rng_state"]
    if not isinstance(cuda, list) or not cuda or any(value.dtype != torch.uint8 or value.ndim != 1 or not value.numel() for value in cuda):
        raise ValueError("GPU training checkpoint lacks valid captured CUDA RNG bytes")
    numpy_state = saved["numpy_rng_state"]
    return {"torch": c.array_sha(saved["torch_rng_state"].numpy()),
            "cuda": [c.array_sha(value.numpy()) for value in cuda],
            "numpy": hashlib.sha256(numpy_state[1].tobytes() + repr((numpy_state[0], *numpy_state[2:])).encode()).hexdigest(),
            "python": hashlib.sha256(repr(saved["python_rng_state"]).encode()).hexdigest()}


def independent_schedule(seed, count, updates):
    rng, batches = np.random.default_rng(seed), []
    while len(batches) < updates:
        permutation = rng.permutation(count)
        batches.extend(permutation[start:start + 8].astype("<i8") for start in range(0, count, 8))
    batches = batches[:updates]
    digest = hashlib.sha256()
    matrix = np.full((updates, 8), -1, np.int64)
    for position, batch in enumerate(batches):
        digest.update(np.asarray([len(batch)], dtype="<i8").tobytes())
        digest.update(batch.tobytes())
        matrix[position, :len(batch)] = batch
    return batches, matrix, digest.hexdigest()


def _optimizer(saved, parameter_names):
    optimizer = saved["optimizer_state"]
    groups = optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameter_names):
        raise ValueError("Checkpoint optimizer parameter groups differ")
    if (saved["optimizer"] != "Adam" or saved["learning_rate"] != 1e-4 or saved["weight_decay"] != 0.
            or saved["gradient_norm_clip"] != 1. or groups[0]["lr"] != 1e-4 or groups[0]["weight_decay"] != 0.):
        raise ValueError("Registered Adam/learning-rate/clip settings changed")
    if saved["update"] == 0:
        if optimizer["state"]:
            raise ValueError("CP0 optimizer already contains adaptation moments")
        return
    require_exact_ids(optimizer["state"], groups[0]["params"], "optimizer state")
    for identifier, name in zip(groups[0]["params"], parameter_names):
        moment = optimizer["state"][identifier]
        if float(moment["step"]) != saved["update"]:
            raise ValueError("Adam step count differs from the fixed checkpoint update")
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment[key]
            if value.shape != saved["adapter_state"][name].shape or not torch.isfinite(value).all():
                raise ValueError("Adam moment shape/finite check failed")
        if (moment["exp_avg_sq"] < 0).any():
            raise ValueError("Adam second moments must be nonnegative")


def verify_training_receipt(receipt, spec, data, parent_state, feature, foundation_sha, core):
    if receipt["status"] != "COMPLETE" or receipt["spec"] != spec or receipt["updates"] != 300:
        raise ValueError("Training model identity/fixed update contract differs")
    if receipt["core_source_sha256"] != core:
        raise ValueError("Training scientific core differs from the sealed update decision")
    for key in ("F_unchanged", "F_state_unchanged", "F_parameter_gradients_absent", "graph_unchanged"):
        if receipt[key] is not True:
            raise ValueError("Frozen training F/graph invariant failed: " + key)
    if (receipt["F_state_sha256"] != foundation_sha or receipt["selection_uses_VAL"] is not False
            or receipt["F_output_cache_used"] is not False or receipt["new_TS2_pretraining_updates"] != 0
            or receipt["new_GT_reconstruction_pretraining_updates"] != 0):
        raise ValueError("Training used a changed F, output cache or unregistered selection/pretraining")
    batches, matrix, schedule_sha = independent_schedule(spec["seed"], len(data["train_origins"]), 300)
    if receipt["schedule_sha256"] != schedule_sha:
        raise ValueError("Training did not consume the registered complete-pool schedule")
    origins = data["train_origins"]
    target_masks = np.stack([data["mask"][origin:origin + 48] for origin in origins])
    exposure = np.zeros(len(origins), np.int64)
    observed_exposure, clipped = 0, 0
    if len(receipt["curves"]) != 300:
        raise ValueError("Training update curve omits a registered update")
    for update, (indices, row) in enumerate(zip(batches, receipt["curves"]), 1):
        np.add.at(exposure, indices, 1)
        counts = target_masks[indices].sum((0, 1))
        observed_exposure += int(counts.sum())
        if (row["update"] != update or row["effective_batch"] != len(indices) or row["microbatch"] != 1
                or row["observed_channel_counts"] != counts.tolist() or row["active_channels"] != int(np.count_nonzero(counts))
                or row["seen_origin_count"] != int(exposure.sum()) or row["unique_seen_origins"] != int(np.count_nonzero(exposure))
                or row["observed_target_exposure"] != observed_exposure or row["finite"] is not True):
            raise ValueError("Training effective batch/sampling/observed exposure differs")
        if not np.isfinite(row["train_loss"]) or row["train_loss"] < 0 or not np.isfinite(row["preclip_norm"]):
            raise ValueError("Training loss/gradient norm is nonfinite")
        clipped += int(row["preclip_norm"] > 1.)
        if row["clipped"] != (row["preclip_norm"] > 1.) or row["clipping_count"] != clipped:
            raise ValueError("Training clipping evidence differs")
        _close(row["clipping_fraction"], clipped / update, "Clipping fraction")
    if (receipt["seen_origin_counts"] != exposure.tolist() or receipt["seen_origin_count"] != int(exposure.sum())
            or receipt["observed_target_exposure"] != observed_exposure or not np.all(exposure == 12)
            or sum(len(batch) == 5 for batch in batches) != 12):
        raise ValueError("Training final complete-pool/tail exposure differs")
    fusion = spec["mode"] in ("pre", "rand")
    names = [f"gt.{group}.{index}.weight" for group in ("encoder", "decoder") for index in (0, 1)]
    names += ["gt.product." + name for name in ("s00", "s10", "s01", "s11")]
    if fusion:
        names.append("W.weight")
    if receipt["trainable_parameters"] != (13684 if fusion else 244):
        raise ValueError("Training parameter capacity differs")
    checkpoints = receipt["checkpoints"]
    require_exact_ids([row["update"] for row in checkpoints], [0, 100, 300], "checkpoint")
    saved_by_update, rng = {}, None
    for record in checkpoints:
        path = _private_reference(record["path"])
        if c.file_sha(path) != record["sha256"]:
            raise ValueError("Training checkpoint file SHA differs")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        update = record["update"]
        if (saved["run_id"] != spec["run_id"] or saved["seed"] != spec["seed"] or saved["update"] != update
                or saved["fixed_updates"] != 300 or saved["schedule_sha256"] != schedule_sha
                or saved["F_state_sha256"] != foundation_sha):
            raise ValueError("Checkpoint run/update/F/schedule metadata differs")
        state = saved["adapter_state"]
        expected_names = {"gt." + name for name in parent_state}
        if fusion:
            expected_names.update(("W.weight", "feature_mean", "feature_std"))
        if set(state) != expected_names or any(not torch.isfinite(value).all() or value.dtype != torch.float32 for value in state.values()):
            raise ValueError("Checkpoint adapter includes unexpected/F/TS2 tensors or invalid FP32 values")
        if not torch.equal(state["gt.product.graph"], parent_state["product.graph"]):
            raise ValueError("Checkpoint spatial graph changed")
        consumed = np.zeros(len(origins), np.int64)
        for batch in batches[:update]:
            np.add.at(consumed, batch, 1)
        expected_observed = int(sum(target_masks[index].sum() for batch in batches[:update] for index in batch))
        if (saved["seen_origin_counts"] != consumed.tolist() or saved["seen_origin_count"] != int(consumed.sum())
                or saved["unique_seen_origins"] != int(np.count_nonzero(consumed))
                or saved["observed_target_exposure"] != expected_observed):
            raise ValueError("Checkpoint observed target/origin exposure differs")
        fingerprint = _rng_fingerprint(saved)
        if update == 0:
            rng = fingerprint
            if any(not torch.equal(state["gt." + name], value) for name, value in parent_state.items()):
                raise ValueError("CP0 GT initial state differs from its one canonical parent")
            if fusion and torch.count_nonzero(state["W.weight"]):
                raise ValueError("Fusion CP0 bridge was not initialized to zero")
        if fusion:
            if (not np.array_equal(state["feature_mean"].numpy(), feature["mean"])
                    or not np.array_equal(state["feature_std"].numpy(), feature["std"])):
                raise ValueError("Checkpoint bridge statistics differ from the TRAIN-only feature space")
        _optimizer(saved, names)
        saved_by_update[update] = saved
    if receipt["checkpoint_sha256"] != next(row["sha256"] for row in checkpoints if row["update"] == 300):
        raise ValueError("Final trained file differs from the fixed update checkpoint")
    for group, prefixes in (("encoder", ("gt.encoder.",)), ("decoder", ("gt.decoder.",)),
                             ("product", ("gt.product.s",)), ("W", ("W.",))):
        selected = [name for name in names if name.startswith(prefixes)]
        differences = [saved_by_update[300]["adapter_state"][name].double()
                       - saved_by_update[0]["adapter_state"][name].double() for name in selected]
        l2 = math.sqrt(sum(float(value.square().sum()) for value in differences))
        delta = receipt["parameter_deltas"][group]
        _close(delta["l2"], l2, "Parameter group state delta")
        if group != "W" or fusion:
            if l2 <= 0 or delta["changed_elements"] <= 0:
                raise ValueError("Required trained parameter group did not change")
        norms = [row["gradient_norm_" + group] for row in receipt["curves"]]
        if not np.isfinite(norms).all() or np.any(np.asarray(norms) < 0):
            raise ValueError("Training parameter-group gradients are invalid")
        evidence = receipt["group_gradient_evidence"][group]
        if evidence["updates_finite"] != 300 or evidence["updates_with_nonzero_gradient"] != sum(value > 0 for value in norms):
            raise ValueError("Training group gradient evidence differs from its actual curve")
        _close(evidence["maximum_norm"], max(norms), "Maximum group gradient norm")
    probe = receipt["gradient_probe"]
    required = {"E", "product", "F_input"} | ({"W"} if fusion else set())
    if probe["status"] != "PASS" or set(probe["gradient_l1"]) != required or any(not np.isfinite(value) or value <= 0 for value in probe["gradient_l1"].values()):
        raise ValueError("Future-only F input/E/product/W gradient probe failed")
    if [row["update"] for row in receipt["diagnostics"]] != [0, 100, 300]:
        raise ValueError("Fixed TRAIN diagnostics are missing registered checkpoints")
    for row in receipt["diagnostics"]:
        if row["selection_uses_diagnostic"] is not False or row["origin_count"] != 16 or row["reconstruction_excluded_prefix"] != 3:
            raise ValueError("TRAIN diagnostics scope changed")
        if any(not np.isfinite(row[key]) or row[key] < 0 for key in ("forecast_mse", "forecast_mae", "reconstruction_mse", "reconstruction_mae")):
            raise ValueError("TRAIN diagnostics contain invalid metrics")
    if saved_by_update[300]["diagnostics"] != receipt["diagnostics"]:
        raise ValueError("Checkpoint and receipt TRAIN diagnostics differ")
    return {"status": "PASS", "run_id": spec["run_id"], "schedule_sha256": schedule_sha,
            "initial_rng_fingerprint": rng, "checkpoint_state_sha256": _state_dict_sha(saved_by_update[300]["adapter_state"]),
            "initial_state_sha256": _state_dict_sha(saved_by_update[0]["adapter_state"]),
            "seen_windows": int(exposure.sum()), "observed_target_exposure": observed_exposure}


def _features(data, bindings, core):
    index = c.read_json(c.CACHE / "FEATURE_INDEX.json")
    checks = {}
    history = {split: d.windows(data, data[split + "_origins"], 1024)[0] for split in ("train", "val")}
    for seed in (92601, 92602):
        for kind in ("PRE", "RAND"):
            identifier = f"{kind}_s{seed}"
            receipt = c.read_json(c.REPORT / "features" / (identifier + ".json"))
            record = index[str(seed)][kind]
            identity = receipt["identity"]
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if (receipt["status"] != "COMPLETE" or receipt["encoder_fitted"] is not False or receipt["encoder_unchanged"] is not True
                    or receipt["encoder_state_sha256"] != receipt["encoder_state_after_sha256"]
                    or receipt["identity_sha256"] != fingerprint or identity["kind"] != kind or identity["seed"] != seed
                    or identity["length"] != 1024 or identity["data_contract_hash"] != data["contract_hash"]):
                raise ValueError("Frozen feature identity/encoder invariants failed")
            _sources(identity["source_pins"], ("common.py", "data.py", "model.py", "parents.py", "feature_cache.py"), core)
            train = None
            for split in ("train", "val"):
                path = _private_reference(receipt["files"][split]["reference"])
                if Path(record["features_paths"][split]).resolve() != path or c.file_sha(path) != receipt["files"][split]["file_sha256"]:
                    raise ValueError("Feature file reference/SHA differs")
                with np.load(path, allow_pickle=False) as archive:
                    features, origins = archive["features"].copy(), archive["origins"].copy()
                if (features.shape != (len(data[split + "_origins"]), 256, 320) or features.dtype != np.float32
                        or not np.isfinite(features).all() or not np.array_equal(origins, data[split + "_origins"])
                        or c.array_sha(features) != receipt["files"][split]["features_sha256"]
                        or c.array_sha(origins) != receipt["files"][split]["origin_sha256"]
                        or identity["origins_sha256"][split] != c.array_sha(origins)
                        or identity["input_sha256"][split] != c.array_sha(history[split])):
                    raise ValueError("Feature arrays/origins/physical input differ")
                if split == "train":
                    train = features
            mean = train.astype(np.float64).mean((0, 1)).astype(np.float32)
            std64 = train.astype(np.float64).std((0, 1), ddof=0)
            std = np.where(std64 == 0, 1., std64).astype(np.float32)
            for key, value in (("mean", mean), ("std", std)):
                if c.array_sha(value) != receipt["statistics"][key + "_sha256"] or not np.array_equal(value, np.asarray(record[key], np.float32)):
                    raise ValueError("Feature TRAIN-only FP64 population statistics differ")
            if (receipt["statistics"]["sample_count"] != len(train) * 256 or receipt["statistics"]["ddof"] != 0
                    or receipt["statistics"]["exact_constant_dimensions"] != int(np.count_nonzero(std64 == 0))
                    or receipt["parity"]["cache_online_exact"] is not True
                    or receipt["parity"]["cache_online_sha256"] != c.array_sha(train[:4])):
                raise ValueError("Feature TRAIN statistics scope/cache-online proof differs")
            cpu_rng = torch.get_rng_state().clone()
            numpy_rng, python_rng = np.random.get_state(), random.getstate()
            encoder = p.load_ts2(bindings[str(seed)]["TS2"], device="cpu", random=kind == "RAND")
            state_sha, online_sha = c.state_sha(encoder.net), c.state_sha(encoder._net)
            if state_sha != receipt["encoder_state_sha256"] or state_sha != record["state_sha"]:
                raise ValueError("Restored CPU TS2 state differs from actual feature producer")
            representations = p.export_tensor(encoder, torch.from_numpy(history["train"][:4]))
            official = p.export_official(encoder, history["train"][:4])
            np.testing.assert_allclose(representations.numpy(), official, atol=1e-5, rtol=1e-4)
            pooled = representations.reshape(4, 256, 4, 320).mean(2).numpy()
            np.testing.assert_allclose(pooled, train[:4], atol=1e-5, rtol=1e-4)
            current_numpy_rng = np.random.get_state()
            if (not torch.equal(torch.get_rng_state(), cpu_rng) or current_numpy_rng[0] != numpy_rng[0]
                    or not np.array_equal(current_numpy_rng[1], numpy_rng[1]) or current_numpy_rng[2:] != numpy_rng[2:]
                    or random.getstate() != python_rng or c.state_sha(encoder.net) != state_sha
                    or c.state_sha(encoder._net) != online_sha or encoder.net.training or encoder._net.training
                    or any(parameter.requires_grad or parameter.grad is not None for network in (encoder.net, encoder._net) for parameter in network.parameters())):
                raise ValueError("CPU TS2 restore/export changed global RNG, frozen state or modes")
            checks[identifier] = {"status": "PASS", "mean": mean, "std": std, "state_sha256": state_sha,
                                  "online_CPU_state_sha256": online_sha,
                                  "CPU_cache_max_abs": float(np.max(np.abs(pooled - train[:4]))),
                                  "TRAIN_statistic_ticks": len(train) * 256}
            del encoder, features, train, representations, official, pooled
    return checks


def _nested_value(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " public JSON mapping differs")
        for key, value in expected.items():
            _nested_value(actual[key], value, label)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(label + " public JSON sequence differs")
        for value, reference in zip(actual, expected):
            _nested_value(value, reference, label)
    elif isinstance(expected, (float, np.floating)):
        _close(actual, expected, label)
    elif actual != expected:
        raise ValueError(label + " public JSON value differs")


def _csv_value(actual, expected, label):
    if isinstance(expected, (dict, list, tuple)):
        _nested_value(json.loads(actual), expected, label)
    elif isinstance(expected, bool):
        if actual != str(expected):
            raise ValueError(label + " public boolean differs")
    elif isinstance(expected, (float, np.floating)):
        _close(float(actual), expected, label)
    elif isinstance(expected, (int, np.integer)):
        if int(actual) != expected:
            raise ValueError(label + " public integer differs")
    elif actual != ("" if expected is None else str(expected)):
        raise ValueError(label + " public value differs")


def verify_csv(path, expected_rows, keys):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        actual_rows = list(csv.DictReader(stream))
    def identity(row):
        return tuple(str(row.get(key, "")) for key in keys)
    actual = {identity(row): row for row in actual_rows}
    expected = {identity(row): row for row in expected_rows}
    if len(actual) != len(actual_rows) or len(expected) != len(expected_rows) or set(actual) != set(expected):
        raise ValueError(Path(path).name + " public row coverage/identity differs")
    for key, row in expected.items():
        for field, value in row.items():
            if field not in actual[key]:
                raise ValueError(Path(path).name + " public field missing: " + field)
            _csv_value(actual[key][field], value, Path(path).name + ":" + field)
    return len(actual_rows)


def comparison_specs():
    pairs = [("RAW_LOOKBACK", 0, f"A_RAW_L{length}", "A_RAW_L512") for length in (1024, 2048)]
    for seed in (92601, 92602):
        pairs += [("FROZEN_GT_LOOKBACK", seed, f"A_GT_L{length}_s{seed}", f"A_GT_L512_s{seed}") for length in (1024, 2048)]
        pairs += [("FROZEN_GT_VS_RAW", seed, f"A_GT_L{length}_s{seed}", f"A_RAW_L{length}") for length in (512, 1024, 2048)]
        pairs += [("OUTPUT_ENSEMBLE_VS_" + key, seed, f"A_ENSEMBLE_L1024_s{seed}", f"A_{key}_L1024_s{seed}") for key in ("GT", "TS2")]
        pairs += [("ADAPTED_GT_LOOKBACK", seed, f"B_GT_FT_L1024_s{seed}", f"B_GT_FT_L512_s{seed}"),
                  ("PRE_VS_GT", seed, f"B_GT_TS_PRE_L1024_s{seed}", f"B_GT_FT_L1024_s{seed}"),
                  ("PRE_VS_RAND", seed, f"B_GT_TS_PRE_L1024_s{seed}", f"B_GT_TS_RAND_L1024_s{seed}")]
        pairs += [("FINAL_VS_CP0", seed, f"B_{arm}_s{seed}", f"CP0_B_{arm}_s{seed}") for arm in ARMS]
    return pairs


def _origin_statistics(bundle):
    result = []
    for index in range(len(bundle["origins"])):
        measured = independent_score(bundle["prediction"][index:index + 1], bundle["target"][index:index + 1], bundle["mask"][index:index + 1])
        result.append({"origin": int(bundle["origins"][index]), "mse": measured["mse"], "mae": measured["mae"],
                       "count": sum(measured["channel_count"]), "channel_count": measured["channel_count"]})
    return result


def _public_statistics(receipts, bundles, metrics, costs, training):
    channel_rows, origin_rows, comparisons, runs = [], [], [], []
    origins_by_run = {}
    for run_id, bundle in bundles.items():
        origins_by_run[run_id] = _origin_statistics(bundle)
        origin_rows += [{"row_type": "RUN", "run_id": run_id, **row} for row in origins_by_run[run_id]]
        for name, start, end in (("TOTAL", 0, 48), ("1-16", 0, 16), ("17-32", 16, 32), ("33-48", 32, 48)):
            measured = independent_score(bundle["prediction"][:, start:end], bundle["target"][:, start:end], bundle["mask"][:, start:end])
            for channel in range(bundle["prediction"].shape[-1]):
                channel_rows.append({"run_id": run_id, "channel": channel, "horizon_band": name,
                                     "sse": measured["channel_sse"][channel], "sae": measured["channel_sae"][channel],
                                     "count": measured["channel_count"][channel], "mse": measured["channel_mse"][channel],
                                     "mae": measured["channel_mae"][channel]})
        receipt = receipts[run_id]
        row = {key: receipt[key] for key in ("run_id", "phase", "mode", "seed", "L", "updates")}
        row.update(MSE=metrics[run_id]["mse"], MAE=metrics[run_id]["mae"])
        row.update({key: metrics[run_id][key] for key in ("channel_sse", "channel_sae", "channel_count", "channel_mse", "channel_mae")})
        if receipt["phase"] in ("A", "B"):
            inventories = [costs[(run_id, batch)]["resident_inventory"] for batch in (1, 8)]
            if inventories[0] != inventories[1]:
                raise ValueError("Batch1/8 resident model inventories differ")
            row.update(resident_parameters=inventories[0]["by_device"]["cuda"]["parameters"],
                       CPU_resident_parameters=inventories[0]["by_device"]["cpu"]["parameters"],
                       system_resident_parameters=inventories[0]["total_unique_parameters"])
        runs.append(row)
    for name, seed, candidate, reference in comparison_specs():
        cand, ref = bundles[candidate], bundles[reference]
        cohort_hashes = {}
        for key in ("target", "mask", "origins", "mean", "std"):
            if cand[key].dtype != ref[key].dtype or c.array_sha(cand[key]) != c.array_sha(ref[key]):
                raise ValueError("Public paired comparison changed its cohort/scale")
            cohort_hashes[key] = c.array_sha(cand[key])
        identity = {"comparison": name, "seed": seed, "candidate": candidate, "reference": reference}
        row = {**identity, "cohort_hashes": cohort_hashes, "origins": len(cand["origins"]),
               "target_count": int(cand["mask"].sum()), "origins_are_independent_samples": False,
               "status": "PAIRED_CONTRACT_PASS"}
        paired_rows = []
        for candidate_row, reference_row in zip(origins_by_run[candidate], origins_by_run[reference]):
            paired_rows.append({"row_type": "PAIRED", **identity, "origin": candidate_row["origin"],
                                "count": candidate_row["count"], "channel_count": candidate_row["channel_count"],
                                **{f"candidate_{metric}": candidate_row[metric] for metric in ("mse", "mae")},
                                **{f"reference_{metric}": reference_row[metric] for metric in ("mse", "mae")},
                                **{f"{metric}_delta": candidate_row[metric] - reference_row[metric] for metric in ("mse", "mae")}})
        for metric in ("mse", "mae"):
            candidate_value, reference_value = metrics[candidate][metric], metrics[reference][metric]
            if reference_value <= 0:
                raise ValueError("Relative metric has a nonpositive reference")
            row.update({f"candidate_{metric}": candidate_value, f"reference_{metric}": reference_value,
                        f"{metric}_delta": candidate_value - reference_value,
                        f"{metric}_relative_percent": 100 * (candidate_value - reference_value) / reference_value,
                        f"origin_{metric}_improved_fraction": float(np.mean([pair[f"{metric}_delta"] < 0 for pair in paired_rows]))})
        comparisons.append(row)
        origin_rows += paired_rows
    cost_rows, trial_rows, training_rows = [], [], []
    for (run_id, batch), value in costs.items():
        identity = {key: receipts[run_id][key] for key in ("run_id", "phase", "mode", "seed", "L", "arm", "MSE", "MAE")}
        identity["batch"] = batch
        cost_rows.append({**identity, **value["summary"]})
        trial_rows += [{**identity, **row} for row in value["trials"]]
    for run_id, value in training.items():
        for key, row_type in (("curves", "MINIBATCH"), ("diagnostics", "FIXED_TRAIN_DIAGNOSTIC")):
            training_rows += [{"run_id": run_id, "seed": value["seed"], "row_type": row_type,
                               "fixed_updates": value["updates"], **row} for row in value[key]]
    return {"RUNS": verify_csv(c.REPORT / "RUNS.csv", runs, ("run_id",)),
            "CHANNEL_HORIZON_STATS": verify_csv(c.REPORT / "CHANNEL_HORIZON_STATS.csv", channel_rows, ("run_id", "channel", "horizon_band")),
            "ORIGIN_EFFECTS": verify_csv(c.REPORT / "ORIGIN_EFFECTS.csv", origin_rows,
                                         ("row_type", "run_id", "comparison", "seed", "candidate", "reference", "origin")),
            "COMPARISONS": verify_csv(c.REPORT / "COMPARISONS.csv", comparisons, ("comparison", "seed", "candidate", "reference")),
            "COST": verify_csv(c.REPORT / "COST.csv", cost_rows, ("run_id", "batch")),
            "COST_TRIALS": verify_csv(c.REPORT / "COST_TRIALS.csv", trial_rows, ("run_id", "batch", "stage", "iteration")),
            "TRAINING_CURVES": verify_csv(c.REPORT / "TRAINING_CURVES.csv", training_rows, ("run_id", "row_type", "update"))}


def _observation_diagnostic(data, receipts, bundles, metrics):
    proof = c.read_json(c.REPORT / "OBSERVATION_DIAGNOSTIC.json")
    if (proof["status"] != "COMPLETE" or proof["primary_origin_count"] != 371 or proof["primary_cohort_changed"] is not False
            or proof["new_model_fits"] != 0 or proof["CSV_sha256"] != c.file_sha(c.REPORT / "OBSERVATION_CONTEXT_STATS.csv")
            or proof["source_sha256"] != c.file_sha(c.CODE / "observation.py")
            or "post-hoc" not in proof["criterion_registration"].lower()):
        raise ValueError("Supplementary observation diagnostic scope/source differs")
    rows = []
    for name, seed, candidate, reference in comparison_specs():
        if name not in ("RAW_LOOKBACK", "FROZEN_GT_LOOKBACK", "ADAPTED_GT_LOOKBACK", "PRE_VS_GT", "PRE_VS_RAND"):
            continue
        la, lb = receipts[candidate]["L"], receipts[reference]["L"]
        selected = np.asarray([bool(data["mask"][origin - la:origin].all() and data["mask"][origin - lb:origin].all())
                               for origin in data["val_origins"]])
        row = {"comparison": name, "seed": seed, "candidate": candidate, "reference": reference,
               "candidate_L": la, "reference_L": lb, "primary_origin_count": 371,
               "diagnostic_origin_count": int(selected.sum()), "primary_cohort_replaced": False,
               "diagnostic_origins_sha256": c.array_sha(data["val_origins"][selected]),
               "criterion_used_for_selection": False, "registered_before_new_VAL_scores": False,
               "criterion": "every physical history cell observed in BOTH windows; mask-only, no error threshold",
               "status": "COMPLETE" if selected.any() else "NO_FULLY_OBSERVED_HISTORY_CASES",
               "full_candidate_MSE": metrics[candidate]["mse"], "full_reference_MSE": metrics[reference]["mse"],
               "full_MSE_relative_percent": 100 * (metrics[candidate]["mse"] - metrics[reference]["mse"]) / metrics[reference]["mse"]}
        if selected.any():
            scored = {}
            for prefix, run_id in (("candidate", candidate), ("reference", reference)):
                bundle = bundles[run_id]
                scored[prefix] = independent_score(bundle["prediction"][selected], bundle["target"][selected], bundle["mask"][selected])
                for upper, lower in (("MSE", "mse"), ("MAE", "mae")):
                    row[prefix + "_" + upper] = scored[prefix][lower]
            for upper, lower in (("MSE", "mse"), ("MAE", "mae")):
                row[upper + "_relative_percent"] = 100 * (scored["candidate"][lower] - scored["reference"][lower]) / scored["reference"][lower]
        rows.append(row)
    if proof["rows"] != len(rows):
        raise ValueError("Observation diagnostic omitted a registered descriptive comparison")
    return {"status": "PASS", "rows": verify_csv(c.REPORT / "OBSERVATION_CONTEXT_STATS.csv", rows, ("comparison", "seed", "candidate", "reference")),
            "primary_origins": 371, "primary_replaced": False, "criterion_registration": "post-hoc descriptive; no score selection"}


def _log_proof(value):
    path = _private_reference(value["log_reference"])
    if c.file_sha(path) != value["log_sha256"] or value["exit_code"] != 0:
        raise ValueError("Final test/replay actual execution log/exit proof differs")
    return path


def _final_proofs():
    tests = c.read_json(c.REPORT / "TEST_VALIDATION.json")
    replay = c.read_json(c.REPORT / "PUBLIC_REPLAY_VALIDATION.json")
    if tests["status"] != "PASS" or replay["status"] != "PASS":
        raise ValueError("Actual final regression/public replay proof did not pass")
    log = _log_proof(tests)
    _log_proof(replay)
    if tests["passed"] <= 0 or tests["failed"] != 0 or tests["errors"] != 0:
        raise ValueError("Actual regression result has failures/errors or no passed tests")
    exclusions = {"tests/test_model_parity.py", "tests/test_baselines.py"}
    if set(tests["excluded_files"]) != exclusions or not tests.get("exclusion_reason"):
        raise ValueError("Final regression must disclose the two excluded old Labour integration files")
    import re
    passed = re.findall(r"(\d+) passed", log.read_text(encoding="utf-8", errors="replace"))
    if not passed or int(passed[-1]) != tests["passed"]:
        raise ValueError("Final regression pass count differs from its actual stdout log")
    if (tests["source_unchanged"] is not True or not tests["source_sha256"]
            or tests["source_sha256"] != tests["source_sha256_after"]):
        raise ValueError("Regression source changed during its actual execution")
    for name, sha in tests["source_sha256"].items():
        path = (c.ROOT / name).resolve()
        if not path.is_relative_to(c.ROOT.resolve()) or c.file_sha(path) != sha:
            raise ValueError("Regression tested source differs from the current code/tests: " + name)
    required_tested = {(c.CODE / name).relative_to(c.ROOT).as_posix() for name in CORE_NAMES}
    required_tested.update({(c.CODE / "verify.py").relative_to(c.ROOT).as_posix(), "tests/test_gt_ts2_verify.py"})
    if not required_tested.issubset(tests["source_sha256"]):
        raise ValueError("Final regression source proof omits scientific code or verifier tests")
    required = {"RUNS.csv", "COMPARISONS.csv", "CHANNEL_HORIZON_STATS.csv", "COST.csv", "COST_TRIALS.csv"}
    required.update(path.name for pattern in ("FIG*.png", "FIG*.svg") for path in c.REPORT.glob(pattern))
    if len([name for name in required if name.endswith((".png", ".svg"))]) != 6 or not required.issubset(replay["files"]):
        raise ValueError("Public-only replay proof must bind public statistics and all six PNG/SVG images")
    for name, sha in replay["files"].items():
        path = (c.REPORT / name).resolve()
        if not path.is_relative_to(c.REPORT.resolve()) or c.file_sha(path) != sha:
            raise ValueError("Replayed public artifact SHA differs")
    local = replay.get("full_local_replay", {})
    if not replay.get("public_only_scope") or local.get("status") not in ("PASS", "COMPLETE"):
        raise ValueError("Public-only and full local/private replay proofs are both required")
    _log_proof(local)
    if len(local["files"]) != 21:
        raise ValueError("Full local replay must reproduce all 21 derived public artifacts")
    for name, sha in local["files"].items():
        path = (c.REPORT / name).resolve()
        if not path.is_relative_to(c.REPORT.resolve()) or c.file_sha(path) != sha:
            raise ValueError("Full local replay public artifact SHA differs")
    return {"status": "PASS", "regression_passed": tests["passed"], "excluded_old_integration_files": sorted(exclusions),
            "excluded_scope_reason": tests["exclusion_reason"], "replayed_public_files": len(replay["files"])}


def _budget_and_stages(decision, start):
    ledger = c.read_json(c.CACHE / "GPU_LEDGER.json")
    if ledger["status"] != "IDLE":
        raise ValueError("Cannot certify while a GPU lease remains active")
    stages = ledger["stages"]
    summed = 0.
    previous_end = -float("inf")
    for stage in stages:
        if stage["started_unix"] < previous_end or stage["ended_unix"] < stage["started_unix"]:
            raise ValueError("GPU stage wall intervals overlap or run backwards")
        _close(stage["wall_seconds"], stage["ended_unix"] - stage["started_unix"], "Actual GPU stage wall time", atol=1e-6)
        previous_end = stage["ended_unix"]
        summed += stage["wall_seconds"]
    _close(ledger["GPU_seconds"], summed, "Actual committed GPU ledger", atol=1e-6)
    cache_bytes = sum(path.stat().st_size for path in c.CACHE.rglob("*") if path.is_file())
    elapsed = time.time() - start["started_unix"]
    if summed > 5400 or elapsed > 10800 or cache_bytes > 6 * 1024**3:
        raise ValueError("Actual GPU/wall/private-cache campaign budget exceeded")
    snapshot = c.read_json(c.REPORT / "FINAL_RESOURCE_SNAPSHOT.json")
    _close(snapshot["GPU_used_seconds"], summed, "Final resource snapshot GPU usage", atol=1e-6)
    if snapshot["elapsed_seconds"] > elapsed + 1 or snapshot["new_cache_bytes"] > cache_bytes:
        raise ValueError("Final resource snapshot reports impossible future time/cache usage")
    if (decision["updates"] != 300 or decision["status"] != "FIXED_300_UPDATES" or decision["selection_uses_VAL"] is not False
            or decision["prior_new_Phase_A_scores"] != 0 or decision["smoke_models_discarded"] is not True):
        raise ValueError("Uniform 300-update decision was changed or selected after new VAL scores")
    at_decision = decision["budget_at_decision"]
    available = min(4320 - at_decision["GPU_used_seconds"], 8640 - at_decision["elapsed_seconds"])
    estimate = 8 * 300 * decision["seconds_per_update"] + decision["overhead_seconds"]
    _close(decision["training_available_seconds"], available, "Registered 20% training reserve")
    _close(decision["eight_arm_estimate_seconds"], estimate, "Registered full matched-set time estimate")
    if estimate > available:
        raise ValueError("Initial update decision did not preserve the 20% report reserve")
    admissions = {}
    for filename in ("PHASE_B_ADMISSION.json", "WORKER_PHASE_B_ADMISSION.json"):
        value = c.read_json(c.REPORT / filename)
        budget = value["budget"]
        available = min(4320 - budget["GPU_used_seconds"], 8640 - budget["elapsed_seconds"])
        if (value["status"] != "ADMITTED" or value["registered_updates"] != 300 or value["updates_changed"] is not False
                or value["selection_uses_VAL_scores"] is not False or estimate > available
                or value["gate_source_sha256"] != c.file_sha(c.CODE / "admission.py")):
            raise ValueError("Actual parent/worker Phase B admission or reserve proof failed")
        _close(value["current_training_available_seconds"], available, "Admission 20% reserve")
        _close(value["full_eight_arm_estimate_seconds"], estimate, "Admission fixed matched-set estimate")
        if filename.startswith("WORKER") and value["wrapper_source_sha256"] != c.file_sha(c.CODE / "run.py"):
            raise ValueError("Actual worker admission wrapper source differs")
        admissions[filename] = {"status": "PASS", "available_seconds": available, "estimate_seconds": estimate}
    stage_receipts = [c.read_json(path) for path in (c.REPORT / "stage_receipts").glob("*.json")]
    for value in stage_receipts:
        path = _private_reference(value["log_reference"])
        if c.file_sha(path) != value["log_sha256"]:
            raise ValueError("Original failed/completed stage log was overwritten")
        if value["status"] == "COMPLETE" and value["exit_code"] != 0:
            raise ValueError("Completed stage has a failed actual worker exit")
    required = {"validate", "features", "smoke", "phase_a", "phase_b", "cost"}
    if not required.issubset({row["stage"] for row in stage_receipts if row["status"] == "COMPLETE"}):
        raise ValueError("Required actual GPU stage completion evidence is missing")
    return {"status": "PASS", "GPU_seconds": summed, "elapsed_seconds_at_verification": elapsed,
            "private_cache_bytes": cache_bytes, "report_reserve_fraction": .2, "admissions": admissions,
            "preserved_failed_stage_logs": sum(row["status"] == "FAILED" for row in stage_receipts)}


def _original_preservation(asset_root, start):
    manifest_path = c.CACHE / "PROTECTED_FILES.json"
    if c.file_sha(manifest_path) != start["protected_manifest_sha256"]:
        raise ValueError("Original preservation manifest was changed")
    manifest = c.read_json(manifest_path)
    if len(manifest) != start["protected_file_count"]:
        raise ValueError("Original tracked-file preservation coverage differs")
    appended = []
    for name, record in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Original preservation file escapes the repository")
        if relative.suffix.lower() in (".npz", ".npy", ".parquet", ".pkl") and "test" in name.lower():
            raise ValueError("Refusing to open/hash a protected TEST archive from preservation metadata")
        path = c.ROOT / relative
        if name.startswith("_docs/history/"):
            with path.open("rb") as stream:
                original_prefix = stream.read(record["bytes"])
            if hashlib.sha256(original_prefix).hexdigest() != record["sha256"] or path.stat().st_size < record["bytes"]:
                raise ValueError("Original history was changed rather than appended: " + name)
            if path.stat().st_size > record["bytes"]:
                appended.append(name)
        elif path.stat().st_size != record["bytes"] or c.file_sha(path) != record["sha256"]:
            raise ValueError("Original tracked artifact was changed: " + name)
    baseline = c.read_json(c.CACHE / "ORIGINAL_PRESERVATION_BASELINE.json")
    head = subprocess.run(["git", "-C", str(asset_root), "rev-parse", "HEAD"], capture_output=True, check=True).stdout.decode().strip()
    status = subprocess.run(["git", "-C", str(asset_root), "status", "--porcelain=v1", "-z"], capture_output=True, check=True).stdout
    if (head != baseline["HEAD"] or head != start["original_HEAD"]
            or hashlib.sha256(status).hexdigest() != baseline["status_z_sha256"]
            or baseline["status_z_sha256"] != start["original_status_z_sha256"]):
        raise ValueError("Original checkout HEAD/status was changed")
    for filename, record in baseline["presentations"].items():
        path = Path(filename)
        if path.stat().st_size != record["bytes"] or c.file_sha(path) != record["sha256"]:
            raise ValueError("Original presentation was changed")
    return {"status": "PASS", "protected_tracked_files": len(manifest), "history_append_only": appended,
            "original_HEAD": head, "original_status_sha256": hashlib.sha256(status).hexdigest(),
            "preserved_presentation_files": len(baseline["presentations"]), "protected_TEST_archives_opened": 0}


def run_verification(output_path=None):
    ready = readiness()
    if ready["status"] != "READY":
        raise FileNotFoundError("Cannot certify incomplete evidence: " + ", ".join(ready["missing_evidence"]))
    output = Path(output_path) if output_path is not None else c.REPORT / "VALIDATION_REPORT.json"
    if output.exists():
        raise FileExistsError("Preserve the existing validation report; choose a new verification output")
    if torch.cuda.is_initialized():
        raise RuntimeError("Run final verification in a fresh CPU-only process")
    started, started_unix = time.perf_counter(), time.time()
    torch.set_num_threads(4)
    decision = c.read_json(c.REPORT / "UPDATE_DECISION.json")
    core = decision["core_source_sha256"]
    if set(core) != set(CORE_NAMES) or core != {name: c.file_sha(c.CODE / name) for name in CORE_NAMES}:
        raise ValueError("Scientific core differs from the exact nine sealed TRAIN-decision modules")
    historical = _source_history(core)
    from .execution import context
    ctx = context()
    data, bindings = ctx["data"], ctx["parents"]
    contract = c.read_json(c.REPORT / "EXPERIMENT_CONTRACT.json")
    start = c.read_json(c.REPORT / "START_STATE.json")
    if (len(data["train_origins"]) != 197 or len(data["val_origins"]) != 371
            or contract["evaluation"]["origins_sha256"] != c.array_sha(data["val_origins"])
            or contract["data"]["contract_hash"] != data["contract_hash"] or contract["H"] != 48):
        raise ValueError("Sealed TRAIN/VAL primary origin/data contract differs")
    expected = expected_runs()
    require_exact_ids([path.stem for path in (c.REPORT / "receipts").glob("*.json")],
                      [row["run_id"] for row in expected], "prediction")
    histories = {length: d.windows(data, data["val_origins"], length) for length in (512, 1024, 2048)}
    legacy_history = d.windows(data, data["val_origins"], 512, legacy=True)[0]
    cohort = {"target": histories[512][1], "mask": histories[512][2], "origins": data["val_origins"],
              "mean": data["mean"], "std": data["std"]}
    receipts, bundles, metrics = {}, {}, {}
    for spec in expected:
        run_id = spec["run_id"]
        receipt = c.read_json(c.REPORT / "receipts" / (run_id + ".json"))
        if (receipt["status"] != "COMPLETE" or any(receipt.get(key) != value for key, value in spec.items())
                or receipt["data_contract_sha256"] != data["contract_hash"]
                or receipt["parent_data_contract_sha256"] != data["legacy_contract_hash"]):
            raise ValueError("Prediction identity/coordinate contract differs: " + run_id)
        path = _private_reference(receipt["prediction_reference"])
        if path != (c.CACHE / "predictions" / (run_id + ".npz")).resolve() or c.file_sha(path) != receipt["artifact_sha256"]:
            raise ValueError("Prediction canonical artifact reference/SHA differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(FIELDS):
                raise ValueError("Private prediction contains missing/unexpected arrays")
            bundle = {key: archive[key].copy() for key in FIELDS}
        history = legacy_history if spec["phase"] == "A0" else histories[spec["L"]][0]
        metrics[run_id] = verify_prediction(receipt, bundle, history, cohort)
        used = ("common.py", "data.py", "model.py", "parents.py", "analysis.py", "execution.py")
        if spec["phase"] == "A":
            used += ("campaign.py",)
        elif spec["phase"] in ("B", "CP0"):
            used = CORE_NAMES
        _sources(receipt["source_identity"], used, historical if spec["phase"] == "A0" else core)
        if spec["phase"] == "A0":
            parent = bindings[str(spec["seed"])]["GT" if spec["mode"] == "gt" else "TS2"]
            np.testing.assert_allclose(bundle["prediction"], parent["arrays"]["CALIBRATED_F"], atol=1e-5, rtol=1e-4)
            if receipt["reproduction"]["status"] != "PASS" or receipt["reproduction"]["reference_sha256"] != c.array_sha(parent["arrays"]["CALIBRATED_F"]):
                raise ValueError("A0 no longer binds its original historical prediction")
        receipts[run_id], bundles[run_id] = receipt, bundle
    for spec in expected:
        if spec["phase"] == "CP0":
            reference = f"A_GT_L{spec['L']}_s{spec['seed']}"
            np.testing.assert_allclose(bundles[spec["run_id"]]["prediction"], bundles[reference]["prediction"], atol=1e-5, rtol=1e-4)
            if receipts[spec["run_id"]]["CP0_parent_parity"]["status"] != "PASS":
                raise ValueError("Zero-bridge CP0 parent parity evidence failed")
    foundation = c.read_json(c.ROOT / "research/professor_compressor_direct_20261011/FOUNDATION_PROVENANCE.json")
    foundation_sha = foundation["model_state_sha256"]
    validation = c.read_json(c.REPORT / "PARENT_AND_GRADIENT_VALIDATION.json")
    checks = validation["checks"]
    _sources(validation["source_identity"], ("common.py", "data.py", "model.py", "parents.py", "execution.py"), historical)
    if (validation["status"] != "PASS" or checks["F_tensor_state_sha256"] != foundation_sha
            or not all(checks["F_invariant"].values()) or checks["F_context_maximum"] != 2048):
        raise ValueError("Actual historical F/gradient validation failed")
    for fixture in ("ramp", "constant", "real_GT_latent"):
        if checks["F_forward_" + fixture]["status"] != "PASS" or checks["F_gradient_" + fixture]["finite"] is not True:
            raise ValueError("Actual F forward/gradient fixture failed")
        if fixture != "constant" and checks["F_gradient_" + fixture]["l1"] <= 0:
            raise ValueError("Actual nonconstant F input gradient was disconnected")
    for length in (1024, 2048):
        if checks["GT_suffix_" + str(length)]["status"] != "PASS":
            raise ValueError("GT long-window boundary fixture failed")
    feature_checks = _features(data, bindings, core)
    parent_states = {seed: torch.load(bindings[str(seed)]["GT"]["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
                     for seed in (92601, 92602)}
    training, training_checks = {}, {}
    training_specs = [spec for spec in expected if spec["phase"] == "B"]
    require_exact_ids([path.stem for path in (c.REPORT / "training").glob("*.json")],
                      [spec["run_id"] for spec in training_specs], "training")
    for spec in training_specs:
        run_id = spec["run_id"]
        path = c.REPORT / "training" / (run_id + ".json")
        receipt = c.read_json(path)
        if receipts[run_id]["training_receipt_sha256"] != c.file_sha(path) or receipts[run_id]["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
            raise ValueError("Final forecast/training receipt/checkpoint binding differs")
        feature = feature_checks.get(f"{spec['mode'].upper()}_s{spec['seed']}")
        training_checks[run_id] = verify_training_receipt(receipt, spec, data, parent_states[spec["seed"]], feature, foundation_sha, core)
        training[run_id] = receipt
        if spec["mode"] in ("pre", "rand"):
            branch = receipt["branch_removal"]
            if branch["fixed_VAL_origin_indices"] != list(range(8)) or not np.isfinite(branch["l2"]) or branch["l2"] <= 0:
                raise ValueError("Trained TS2 branch removal did not change its actual forecast")
    schedules = c.read_json(c.REPORT / "SAMPLING_SCHEDULES.json")
    for seed in (92601, 92602):
        matched = [training_checks[spec["run_id"]] for spec in training_specs if spec["seed"] == seed]
        if len({row["schedule_sha256"] for row in matched}) != 1 or len({json.dumps(row["initial_rng_fingerprint"], sort_keys=True) for row in matched}) != 1:
            raise ValueError("Same-seed matched arms differ in schedule or parent-construction RNG isolation")
        batches, matrix, _ = independent_schedule(seed, 197, 300)
        with np.load(c.CACHE / f"schedule_s{seed}.npz", allow_pickle=False) as archive:
            if not np.array_equal(matrix, archive["indices"]) or not np.array_equal(archive["origins"], data["train_origins"]):
                raise ValueError("Saved complete-pool schedule differs from the registered seed order")
        summary = schedules[str(seed)]
        if (summary["matrix_sha256"] != c.array_sha(matrix) or summary["counts"] != [len(batch) for batch in batches]
                or summary["seen_windows"] != 2364 or summary["unique_pool_origins"] != 197
                or summary["pool_origin_sha256"] != c.array_sha(data["train_origins"])):
            raise ValueError("Public sampling schedule or small-tail exposure differs")
    costs, cost_checks, process_pairs, f_inventories = {}, [], set(), []
    cost_specs = [spec for spec in expected if spec["phase"] in ("A", "B")]
    required_cost_ids = [f"{spec['run_id']}_B{batch}" for spec in cost_specs for batch in (1, 8)]
    require_exact_ids([path.stem for path in (c.REPORT / "cost_receipts").glob("*.json")], required_cost_ids, "cost")
    for spec in cost_specs:
        for batch in (1, 8):
            run_id = spec["run_id"]
            value = c.read_json(c.REPORT / "cost_receipts" / f"{run_id}_B{batch}.json")
            pair = (value["pid"], value["process_create_time_unix"])
            if pair in process_pairs or pair[1] > value["started_unix"] + 1 or value["ended_unix"] < value["started_unix"]:
                raise ValueError("Cost route/batch was not measured in its own fresh OS process")
            process_pairs.add(pair)
            cost_checks.append(verify_cost_receipt(value, spec, data["val_origins"], contract, foundation_sha,
                                                   receipts[run_id], bundles[run_id]))
            source = value["loaded_source_sha256"]
            _sources(source["new_campaign"], ("common.py", "data.py", "model.py", "parents.py", "execution.py"), core)
            if source["new_campaign"].get((c.CODE / "cost.py").relative_to(c.ROOT).as_posix()) != c.file_sha(c.CODE / "cost.py"):
                raise ValueError("Used cost measurement producer source changed")
            if source["F_installed_model_implementation"] != foundation["installed_source_sha256"]:
                raise ValueError("Measured F implementation differs from the canonical parent")
            states = {"F": foundation_sha}
            if spec["mode"] in ("gt", "pre", "rand", "ensemble"):
                states["GT_adapter"] = (training_checks[run_id]["checkpoint_state_sha256"] if spec["phase"] == "B" else
                                        _state_dict_sha({"gt." + name: tensor for name, tensor in parent_states[spec["seed"]].items()}))
            if spec["mode"] in ("ts2", "pre", "rand", "ensemble"):
                feature = feature_checks[f"{'RAND' if spec['mode'] == 'rand' else 'PRE'}_s{spec['seed']}"]
                states.update(TS2_SWA=feature["state_sha256"], TS2_online_CPU=feature["online_CPU_state_sha256"])
            if spec["mode"] in ("ts2", "ensemble"):
                decoder = torch.load(bindings[str(spec["seed"])]["TS2"]["decoder_checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
                states["TS2_decoder"] = _state_dict_sha(decoder)
            if value["states_before_sha256"] != states or set(value["resident_inventory"]["by_module"]) != set(states):
                raise ValueError("Cost loaded modules do not match the exact selected parent/adapted tensors")
            inventory = value["resident_inventory"]["by_module"]
            f_inventories.append(inventory["F"])
            for name, counts in inventory.items():
                expected_device = "cpu" if name == "TS2_online_CPU" else "cuda"
                if counts["cuda" if expected_device == "cpu" else "cpu"]["parameters"] != 0:
                    raise ValueError("Cost module resides on an unexpected CPU/CUDA device")
                capacity = 638528 if name in ("TS2_SWA", "TS2_online_CPU") else 43797 if name == "TS2_decoder" else 13684 if name == "GT_adapter" and spec["mode"] in ("pre", "rand") else 244 if name == "GT_adapter" else None
                if capacity is not None and counts[expected_device]["parameters"] != capacity:
                    raise ValueError("Cost actual resident model parameter capacity differs")
                if counts[expected_device]["parameter_bytes"] != 4 * counts[expected_device]["parameters"]:
                    raise ValueError("Measured resident parameter bytes are not actual FP32 tensor bytes")
            costs[(run_id, batch)] = value
    if any(value != f_inventories[0] for value in f_inventories) or f_inventories[0]["cuda"]["parameters"] <= 0:
        raise ValueError("One shared canonical F resident parameter count differs across cost routes")
    public_statistics = _public_statistics(receipts, bundles, metrics, costs, training)
    observation = _observation_diagnostic(data, receipts, bundles, metrics)
    final_proofs = _final_proofs()
    budget = _budget_and_stages(decision, start)
    preservation = _original_preservation(ctx["paths"]["asset_root"], start)
    if core != {name: c.file_sha(c.CODE / name) for name in CORE_NAMES} or torch.cuda.is_initialized():
        raise ValueError("Scientific core changed during CPU verification or CUDA was initialized")
    result = {"status": "PASS", "verification_scope": "actual completed development campaign; CPU-only read-only independent cross-check",
              "started_unix": started_unix, "ended_unix": time.time(), "wall_seconds": time.perf_counter() - started,
              "scientific_core_sha256": core, "verifier_source_sha256": c.file_sha(Path(__file__)),
              "prediction_receipts": len(receipts), "training_arms": len(training), "feature_spaces": len(feature_checks),
              "independent_cost_processes": len(costs), "cost_trials": sum(row["trials"] for row in cost_checks),
              "independent_scoring": "FP64 observed channel SSE/SAE/count, then equal-channel mean; checked against producer and public sufficient statistics",
              "public_statistics_rows": public_statistics, "training_checks": training_checks,
              "feature_checks": {key: {name: value for name, value in row.items() if name not in ("mean", "std")} for key, row in feature_checks.items()},
              "cost_checks": cost_checks, "observation_diagnostic": observation, "final_execution_proofs": final_proofs,
              "resource_budget": budget, "original_preservation": preservation,
              "protected_TEST_archives_opened": 0, "CUDA_initialized": False,
              "limits": "same previously exposed development VAL; two seeds; overlapping origins; no independent TEST or confidence interval claim"}
    c.seal_json(output, result, allowed_root=c.REPORT)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("verify",))
    parser.add_argument("--check-only", action="store_true", help="List readiness/missing evidence without writing or loading models")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)
    result = readiness() if args.check_only else run_verification(args.output_path)
    print(c.json_bytes(result).decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
