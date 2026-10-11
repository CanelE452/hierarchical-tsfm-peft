"""Fresh full-VAL reconstruction audit of canonical saved A checkpoints; no fitting/F/TEST."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import torch

from analyze_direct import independent_metric, job_key
from common_direct import (CACHE, HERE, ROOT, Resources, array_digest, digest,
                           read_json, state_digest, write_json)
from data_io import load_data, make_windows
from models_gt import GTConvAE
from ts2_adapter import ReconstructionDecoder, create_ts2, export_offline, freeze_encoder


RTOL, ATOL = 1e-7, 1e-9


def checkpoint_path(value, identifier):
    path = Path(str(value).replace("\\", "/"))
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(CACHE / "runs" / identifier):
        raise ValueError(f"A checkpoint must belong to this exact canonical run: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compare_metrics(measured, reported):
    checks = {}
    for name in ("mse", "mae", "channel_mse", "channel_mae", "horizon_mse", "horizon_mae",
                 "observed_counts", "origin_count", "physical_channel_mae"):
        if name not in reported:
            continue
        actual, expected = measured.get(name), reported[name]
        if isinstance(expected, list):
            if actual is None or len(actual) != len(expected):
                checks[name] = {"pass": False, "reason": "missing or different-length metric"}
                continue
            valid = [(a, b) for a, b in zip(actual, expected) if a is not None and b is not None]
            same_nulls = all((a is None) == (b is None) for a, b in zip(actual, expected))
            difference = max((abs(float(a) - float(b)) for a, b in valid), default=0.)
            okay = same_nulls and all(np.isclose(a, b, rtol=RTOL, atol=ATOL) for a, b in valid)
        else:
            okay = actual is not None and expected is not None and bool(np.isclose(actual, expected, rtol=RTOL, atol=ATOL))
            difference = abs(float(actual) - float(expected)) if actual is not None and expected is not None else None
        checks[name] = {"pass": bool(okay), "max_absolute_difference": difference}
    if "mse" not in checks or "mae" not in checks:
        raise ValueError("Original reconstruction receipt lacks MSE/MAE")
    return checks


def require_original_model_sources(result, family):
    required = [HERE / "data_io.py", HERE / ("models_gt.py" if family == "GT" else "ts2_adapter.py")]
    pins = {Path(str(name).replace("\\", "/")).as_posix(): sha
            for name, sha in result.get("source_pins", {}).items()}
    if family == "TS2":
        required.extend(ROOT / name for name in pins if "/vendor/ts2vec/" in name)
    checked = {}
    for path in required:
        name = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if pins.get(name) != actual:
            raise ValueError(f"Reconstruction source differs from the original run pin: {name}")
        checked[name] = actual
    return checked


def locate_a_checkpoints(result):
    job, training = result["job"], result["training"]
    if job["family"] == "GT":
        path = checkpoint_path(training["checkpoint_path"], job["id"])
        return {"joint_encoder_native_decoder_A": {"path": path, "sha256": digest(path)}}
    encoder, decoder = training["encoder"], training["decoder_reconstruction"]
    paths = {"official_SWA_encoder_A": checkpoint_path(encoder["checkpoint"], job["id"]),
             "reconstruction_native_decoder_A": checkpoint_path(decoder["best_checkpoint"], job["id"])}
    declared = {"official_SWA_encoder_A": encoder["checkpoint_sha256"],
                "reconstruction_native_decoder_A": decoder["best_checkpoint_sha256"]}
    checks = {}
    for name, path in paths.items():
        sha = digest(path)
        if sha != declared[name]:
            raise ValueError(f"Original saved A checkpoint SHA differs: {name}")
        checks[name] = {"path": path, "sha256": sha}
    return checks


@torch.no_grad()
def reconstruct_gt(result, data, past, checkpoints, resources):
    job = result["job"]
    saved = torch.load(checkpoints["joint_encoder_native_decoder_A"]["path"], map_location="cpu", weights_only=False)
    if saved["epoch"] != result["training"]["selected_epoch"]:
        raise ValueError("GT A checkpoint epoch differs from selected reconstruction epoch")
    if not torch.equal(saved["state_dict"]["product.graph"], torch.as_tensor(data["graph"])):
        raise ValueError("GT A checkpoint graph differs from the TRAIN-only checked graph")
    model = GTConvAE(data["graph"], no_spatial=job.get("no_spatial", False))
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.cuda().eval().requires_grad_(False)
    model_hash_before = state_digest(model)
    output = []
    for start in range(0, len(past), 16):
        values = torch.as_tensor(past[start:start + 16], device="cuda", dtype=torch.float32)
        output.append(model(values).cpu().numpy())
        resources.check(job=job["id"], phase="full_VAL_A_reconstruction", completed_origins=min(start + 16, len(past)))
    prediction = np.concatenate(output)
    model_hash_after = state_digest(model)
    if model_hash_after != model_hash_before:
        raise ValueError("GT A parameters/buffers changed during reconstruction inference")
    del model, values, output
    return prediction, {"GT_forward_batch": 16, "A_state_sha256": model_hash_after,
                        "A_state_unchanged": True, "encoder_decoder": "native full A state, no calibrated B"}


@torch.no_grad()
def reconstruct_ts2(result, data, past, checkpoints, resources):
    job, training = result["job"], result["training"]
    config = training["encoder"]["configuration"]
    model = create_ts2(past.shape[-1], job["dim"], job["seed"], device="cuda",
                       batch_size=config["batch_size"], max_train_length=config["max_train_length"])
    model.load(str(checkpoints["official_SWA_encoder_A"]["path"]))
    freeze_encoder(model)
    model._net.cpu()
    encoder_hash_before = state_digest(model.net)
    if encoder_hash_before != training["encoder"]["inference_swa_state_sha256"]:
        raise ValueError("Restored official SWA encoder differs from original inference state")
    decoder = ReconstructionDecoder(job["dim"], past.shape[-1])
    saved = torch.load(checkpoints["reconstruction_native_decoder_A"]["path"], map_location="cpu", weights_only=True)
    if saved["epoch"] != training["decoder_reconstruction"]["best_epoch"]:
        raise ValueError("TS2 A decoder epoch differs from the original reconstruction selection")
    decoder.load_state_dict(saved["state_dict"], strict=True)
    del saved
    decoder.cuda().eval().requires_grad_(False)
    decoder_hash_before = state_digest(decoder)
    if decoder_hash_before != training["decoder_reconstruction"]["selected_state_sha256"]:
        raise ValueError("Restored A decoder differs from original reconstruction state")
    exports = []
    for start in range(0, len(past), 8):
        exports.append(export_offline(model, past[start:start + 8], batch_size=8))
        resources.check(job=job["id"], phase="full_VAL_offline_encoder_A", completed_origins=min(start + 8, len(past)))
    representation = np.concatenate(exports)
    del exports
    output = []
    for start in range(0, len(representation), 32):
        values = torch.from_numpy(representation[start:start + 32]).cuda()
        output.append(decoder(values).cpu().numpy())
        resources.check(job=job["id"], phase="full_VAL_reconstruction_decoder_A", completed_origins=min(start + 32, len(past)))
    prediction = np.concatenate(output)
    encoder_hash_after, decoder_hash_after = state_digest(model.net), state_digest(decoder)
    if encoder_hash_after != encoder_hash_before or decoder_hash_after != decoder_hash_before:
        raise ValueError("TS2 encoder/decoder A state changed during reconstruction inference")
    del model, decoder, values, output, representation
    return prediction, {"TS2_offline_export_batch": 8, "native_decoder_batch": 32,
                        "offline_export": "past-only L512; causal=False; no sliding; no future input",
                        "SWA_encoder_state_sha256": encoder_hash_after,
                        "A_decoder_state_sha256": decoder_hash_after, "A_states_unchanged": True}


def audit_directories(identifier, tag):
    suffix = Path(identifier) / "attempts" / tag if tag else Path(identifier)
    return HERE / "reconstruction_audit" / suffix, CACHE / "reconstruction_audit" / suffix


def validate_existing(receipt, result, path):
    if receipt.get("status") != "PASS_FULL_VAL_RECONSTRUCTION":
        raise FileExistsError("Preserve incomplete/failed audit; retry with a new --audit-tag")
    if receipt["result_sha256"] != digest(path) or receipt["data_contract_hash"] != result["data_contract_hash"]:
        raise ValueError("Existing audit belongs to a different actual result/data contract")
    for item in receipt["A_checkpoints"].values():
        checkpoint = checkpoint_path(item["path"], result["job"]["id"])
        if digest(checkpoint) != item["sha256"]:
            raise ValueError("A checkpoint changed after existing audit")
    npz_path = ROOT / receipt["reconstruction_path"]
    if not npz_path.resolve().is_relative_to(CACHE / "reconstruction_audit") or digest(npz_path) != receipt["reconstruction_sha256"]:
        raise ValueError("Existing full reconstruction archive differs from audit receipt")
    with np.load(npz_path, allow_pickle=False) as archive:
        measured = independent_metric(archive["reconstruction"], archive["targets"], archive["mask"])
        measured["physical_channel_mae"] = [value * float(scale) if value is not None else None
                                              for value, scale in zip(measured["channel_mae"], archive["std"])]
        if array_digest(archive["origins"]) != receipt["array_hashes"]["origins"]:
            raise ValueError("Existing full reconstruction origins differ from receipt")
    checks = compare_metrics(measured, result["reconstruction"])
    if not all(item["pass"] for item in checks.values()):
        raise ValueError("Existing full reconstruction archive fails independent original-metric comparison")
    return {"job": result["job"], "status": receipt["status"], "execution": "reused_verified_full_reconstruction_audit",
            "receipt_path": receipt["receipt_path"], "metric_comparisons": checks}


def audit_job(planned_job, checks, resources, tag):
    identifier = planned_job["id"]
    result_path = HERE / "runs" / identifier / "result.json"
    if not result_path.exists():
        return {"job": planned_job, "status": "CANONICAL_RESULT_MISSING", "reason": "No completed exact-ID result; no checkpoint substitution"}
    result = read_json(result_path)
    if result.get("status") != "complete":
        return {"job": planned_job, "status": "CANONICAL_RESULT_INCOMPLETE", "reason": result.get("status")}
    if result["job"].get("id") != identifier or job_key(result["job"]) != job_key(planned_job):
        raise ValueError("Actual result ID/condition differs from explicit canonical protocol job")
    receipt_dir, arrays_dir = audit_directories(identifier, tag)
    receipt_path = receipt_dir / "receipt.json"
    if receipt_path.exists():
        with resources.stage("full_reconstruction_existing_receipt_CPU_recheck", gpu=False, job=identifier):
            return validate_existing(read_json(receipt_path), result, result_path)
    if receipt_dir.exists() or arrays_dir.exists():
        raise FileExistsError("Preserve partial audit artifacts; use a new --audit-tag")
    receipt_dir.mkdir(parents=True)
    arrays_dir.mkdir(parents=True)
    receipt = {"job": {**planned_job, **result["job"]}, "status": "INCOMPLETE_AUDIT",
               "receipt_path": receipt_path.relative_to(ROOT).as_posix(), "result_path": result_path.relative_to(ROOT).as_posix(),
               "result_sha256": digest(result_path), "created_UTC": datetime.now(timezone.utc).isoformat(),
               "new_optimizer_updates": 0, "new_scientific_fits": 0, "F_forward_calls": 0,
               "F_backward_calls": 0, "protected_TEST_new_reads": 0,
               "metric_tolerance": {"rtol": RTOL, "atol": ATOL}, "resource_stage_start_index": len(resources.value["stages"])}
    write_json(receipt_dir / "initial_audit.json", receipt)
    try:
        with resources.stage("full_reconstruction_TRAIN_VAL_contract_and_A_receipts_CPU", gpu=False, job=identifier):
            source_checks = require_original_model_sources(result, planned_job["family"])
            data = load_data(planned_job["dataset"])
            expected = checks["datasets"][planned_job["dataset"]]
            if data["contract_hash"] != result["data_contract_hash"] or data["contract_hash"] != expected["contract_hash"]:
                raise ValueError("Actual full TRAIN/VAL loader contract differs from original result and DATA_CHECKS")
            past, mask = make_windows(data, data["val_origins"], masked=True)
            origins = data["val_origins"]
            if array_digest(origins) != expected["origin_checks"]["val_origins"]["sha256"]:
                raise ValueError("All-VAL reconstruction origins differ from DATA_CHECKS")
            if len(origins) != expected["origin_checks"]["val_origins"]["count"] or past.shape[1] != 512:
                raise ValueError("Audit must use every original VAL past512 origin")
            checkpoints = locate_a_checkpoints(result)
            receipt.update(data_contract_hash=data["contract_hash"], A_checkpoints=checkpoints,
                           original_model_source_checks=source_checks,
                           array_hashes={"targets": array_digest(past), "mask": array_digest(mask), "origins": array_digest(origins),
                                         "mean": array_digest(data["mean"]), "std": array_digest(data["std"])},
                           shapes={"past": list(past.shape), "mask": list(mask.shape), "origins": list(origins.shape)})
        with resources.stage("fresh_full_VAL_A_encoder_native_decoder_reconstruction_GPU", gpu=True, job=identifier):
            if planned_job["family"] == "GT":
                reconstruction, inference = reconstruct_gt(result, data, past, checkpoints, resources)
            else:
                reconstruction, inference = reconstruct_ts2(result, data, past, checkpoints, resources)
            torch.cuda.synchronize()
            resources.check(job=identifier, phase="full_VAL_A_reconstruction_complete")
        gc.collect()
        torch.cuda.empty_cache()
        with resources.stage("full_reconstruction_independent_metrics_and_new_archive_CPU", gpu=False, job=identifier):
            measured = independent_metric(reconstruction, past, mask)
            measured["physical_channel_mae"] = [value * float(scale) if value is not None else None
                                                  for value, scale in zip(measured["channel_mae"], data["std"])]
            metric_checks = compare_metrics(measured, result["reconstruction"])
            archive_path = arrays_dir / "full_VAL_reconstruction.npz"
            with archive_path.open("xb") as handle:
                np.savez_compressed(handle, reconstruction=reconstruction, targets=past, mask=mask,
                                    origins=origins, mean=data["mean"], std=data["std"])
            receipt.update(status="PASS_FULL_VAL_RECONSTRUCTION" if all(item["pass"] for item in metric_checks.values()) else "FAIL_FULL_VAL_RECONSTRUCTION_METRICS",
                           independent_full_reconstruction=measured, original_reported_reconstruction=result["reconstruction"],
                           metric_comparisons=metric_checks, inference=inference,
                           reconstruction_path=archive_path.relative_to(ROOT).as_posix(), reconstruction_sha256=digest(archive_path),
                           full_VAL_origin_count=len(origins), observed_past_entry_count=int(mask.sum()),
                           numeric_settings={"precision": "FP32", "AMP": False, "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                             "TF32_cudnn": torch.backends.cudnn.allow_tf32, "torch_num_threads": torch.get_num_threads()},
                           source_pins={path.relative_to(ROOT).as_posix(): digest(path) for path in
                                        (Path(__file__), HERE / "analyze_direct.py", HERE / "common_direct.py", HERE / "data_io.py",
                                         HERE / "models_gt.py", HERE / "ts2_adapter.py")})
            resources.check(job=identifier, phase="full_reconstruction_audit_saved")
    except Exception as error:
        receipt.update(status="FAILED_PRESERVED_RECONSTRUCTION_AUDIT", reason=repr(error), traceback=traceback.format_exc(),
                       resume="Preserve this audit; use a fresh --audit-tag after fixing this path or restoring remaining resources")
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        resources.flush()
        receipt["resource_stage_end_index"] = len(resources.value["stages"])
        receipt["resources_cumulative_process_snapshot"] = dict(resources.value)
        receipt["audit_stages"] = resources.value["stages"][receipt["resource_stage_start_index"]:receipt["resource_stage_end_index"]]
        receipt["ended_UTC"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    return {"job": receipt["job"], "status": receipt["status"], "execution": "fresh_A_reconstruction_inference_only",
            "receipt_path": receipt["receipt_path"], "reason": receipt.get("reason"), "metric_comparisons": receipt.get("metric_comparisons")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("GT", "TS2", "all"), default="all")
    parser.add_argument("--dataset", choices=("robin", "jena", "all"), default="all")
    parser.add_argument("--job-id", action="append", help="Optional exact canonical ID; repeat to select more")
    parser.add_argument("--audit-tag", help="Fresh output suffix for retrying an existing partial audit")
    args = parser.parse_args()
    if args.audit_tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.audit_tag):
        parser.error("audit tag must be a simple filename token")
    protocol = read_json(HERE / "protocol.json")
    planned = protocol["jobs"]
    if len(planned) != 16 or len({job["id"] for job in planned}) != 16:
        raise ValueError("Audit requires the explicit, unique canonical 16-job protocol")
    expected = {family: sum(job["family"] == family for job in planned) for family in ("GT", "TS2")}
    if expected != {"GT": 8, "TS2": 8}:
        raise ValueError("Audit requires eight canonical GT and eight canonical TS2 jobs")
    if args.job_id and any(identifier not in {job["id"] for job in planned} for identifier in args.job_id):
        parser.error("Every requested job ID must be explicitly present in the current protocol")
    selected = [job for job in planned if (args.family == "all" or job["family"] == args.family)
                and (args.dataset == "all" or job["dataset"] == args.dataset)
                and (not args.job_id or job["id"] in args.job_id)]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resources = Resources()
    checks = read_json(HERE / "DATA_CHECKS.json")
    runs = []
    for job in selected:
        try:
            outcome = audit_job(job, checks, resources, args.audit_tag)
        except Exception as error:
            outcome = {"job": job, "status": "FAILED_PATH_PRESERVED", "reason": repr(error),
                       "resume": "Existing output preserved; select a fresh --audit-tag for retry"}
        runs.append(outcome)
        print(str({"job": job["id"], "status": outcome["status"], "reason": outcome.get("reason")}), flush=True)
    coverage = []
    for job in planned:
        receipt_dir, _ = audit_directories(job["id"], args.audit_tag)
        path = receipt_dir / "receipt.json"
        if path.exists():
            receipt = read_json(path)
            current_result = HERE / "runs" / job["id"] / "result.json"
            matching = current_result.exists() and receipt.get("result_sha256") == digest(current_result)
            status = receipt.get("status") if matching else "STALE_AUDIT_RESULT_ID_OR_HASH"
            coverage.append({"job": job, "status": status, "receipt_path": path.relative_to(ROOT).as_posix(), "receipt_sha256": digest(path)})
        else:
            coverage.append({"job": job, "status": "AUDIT_NOT_COMPLETE"})
    passes = {family: sum(row["job"]["family"] == family and row["status"] == "PASS_FULL_VAL_RECONSTRUCTION"
                          for row in coverage) for family in expected}
    summary = {"status": "PASS_ALL_16_FULL_VAL_RECONSTRUCTIONS" if passes == expected else "PARTIAL_FULL_RECONSTRUCTION_AUDIT",
               "protocol_sha256": digest(HERE / "protocol.json"), "expected_canonical_jobs": expected,
               "verified_full_reconstructions": passes, "selected_execution_outcomes": runs, "canonical_coverage": coverage,
               "resources": resources.value, "new_scientific_fits": 0, "F_forward_calls": 0, "protected_TEST_new_reads": 0,
               "scope": "Fresh original A E/native-D full past512 VAL reconstruction and independent metrics; no forecast or fitting"}
    summary_path = HERE / "reconstruction_audit" / f"summary_{time.time_ns()}.json"
    write_json(summary_path, summary)
    print(str({"summary": summary_path.relative_to(ROOT).as_posix(), "status": summary["status"], "passes": passes}), flush=True)
    if any(run["status"] != "PASS_FULL_VAL_RECONSTRUCTION" for run in runs):
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
