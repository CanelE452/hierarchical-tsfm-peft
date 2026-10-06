"""Twelve bounded TRAIN/VAL-only matched E/D fits and joint LR selection."""
import argparse
import gc
import hashlib
import math
import random
import time
import traceback

import numpy as np
import torch

from model_v18 import (REPLAY, build_model, compute_loss, make_checkpoint, restore_model,
                       state_hash, tensor_hashes)
from runtime_v18 import (CACHE, DATASETS, HERE, LIMITS, ROOT, SEEDS, Job, artifact, array_hash,
                         begin_fit, budget_snapshot, check_source_seal, configure, dataset_contract,
                         environment, finish_fit, inputs, load_data, read_json, resolve,
                         save_json, score_arrays, targets)


PHASE_PERIODS = {"robin": 24, "peacock_education": 24, "jena": 144}
_TRAIN_ORIGINS = {}


def sample_epoch_origins(dataset, seed, epoch):
    if dataset not in DATASETS or seed not in SEEDS or not 0 <= epoch <= 120:
        raise ValueError("Unapproved dataset/seed/epoch schedule")
    if dataset not in _TRAIN_ORIGINS:
        _TRAIN_ORIGINS[dataset] = load_data(dataset, include_test=False)["train_origins"].copy()
    origins = _TRAIN_ORIGINS[dataset]
    period, n = PHASE_PERIODS[dataset], 512
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected = []
    for phase in range(period):
        group = origins[origins % period == phase]
        count = n // period + int(phase < n % period)
        if len(group) < count:
            raise ValueError("TRAIN phase insufficient for approved sampling without replacement")
        selected.extend(rng.choice(group, count, replace=False))
    return rng.permutation(selected)


def optimizer_for(model, lr):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {name for name, p in model.named_parameters() if p.requires_grad} != {"encoder.weight", "decoder.weight"}:
        raise ValueError("Optimizer must contain only E/D")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, threshold=1e-4, threshold_mode="rel")
    if {id(p) for p in parameters} != {id(p) for group in optimizer.param_groups for p in group["params"]}:
        raise ValueError("Optimizer parameter coverage differs")
    return optimizer, scheduler


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def gradient_record(model):
    return {name: {"present": p.grad is not None,
                   "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                   "max_abs": float(p.grad.detach().abs().max()) if p.grad is not None else None,
                   "nonzero": int(torch.count_nonzero(p.grad)) if p.grad is not None else 0}
            for name, p in model.named_parameters() if p.requires_grad}


def backward_effective_batch(model, x_cpu, target, mask, microbatch):
    if len(x_cpu) != 4 or microbatch not in (4, 2, 1):
        raise ValueError("Matched effective batch is exactly four origins with microbatch4/2/1")
    device = model.encoder.weight.device
    count = torch.as_tensor(mask.sum((0, 1)), device=device)
    total_loss = 0.0
    model.train()
    for start in range(0, 4, microbatch):
        x = x_cpu[start:start + microbatch].to(device)
        y = torch.as_tensor(target[start:start + microbatch], device=device)
        observed = torch.as_tensor(mask[start:start + microbatch], device=device)
        prediction = model(x)
        loss = compute_loss(prediction, y, observed, channel_count=count)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("STOP_DEBUG: nonfinite forecast loss")
        loss.backward()
        total_loss += float(loss.detach())
        del x, y, observed, prediction, loss
    gradients = gradient_record(model)
    if not all(row["present"] and row["finite"] for row in gradients.values()):
        raise RuntimeError("STOP_DEBUG: missing/nonfinite E/D gradient through frozen backbone")
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("STOP_DEBUG: frozen backbone received parameter gradients")
    return total_loss, gradients


@torch.no_grad()
def evaluate(model, data, origins, job=None, batch_size=4):
    model.eval()
    prediction = []
    for start in range(0, len(origins), batch_size):
        if job is not None:
            job.check_limits()
        x = inputs(data, origins[start:start + batch_size]).to(model.encoder.weight.device)
        value = model(x).detach().cpu().contiguous().numpy()
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError("STOP_DEBUG: VAL prediction not finite FP32")
        prediction.append(value)
        del x
    prediction = np.concatenate(prediction)
    target, mask = targets(data, origins)
    return score_arrays(prediction, target, mask), prediction


def run_id(dataset, lr, seed):
    return f"{dataset}_learned_ed_lr{'1e-3' if lr == 1e-3 else '1e-4'}_s{seed}"


def validate_protocol():
    protocol = read_json(HERE / "protocol.json")
    exact = {"max_epochs": 120, "patience": 6, "epoch_origins": 512, "effective_batch_origins": 4,
             "initial_microbatch": 4, "oom_microbatch_sequence": [4, 2, 1], "optimizer": "AdamW",
             "weight_decay": 0, "clip_grad_norm": 1.0, "learning_rates": [1e-3, 1e-4], "seeds": list(SEEDS)}
    if any(protocol.get(key) != value for key, value in exact.items()):
        raise RuntimeError("Training recipe differs from adopted fixed protocol")
    return protocol


def require_preflight():
    record = read_json(HERE / "preflight.json")
    if record.get("status") != "pass" or not record.get("all_checks_passed"):
        raise RuntimeError("STOP_DEBUG: all step0/autograd/preflight checks must pass before fitting")
    if record["optimizer_updates"] != 0 or record["new_fits"] != 0 or record["test_accessed"]:
        raise RuntimeError("Preflight must have no optimizer updates, new fit or TEST access")
    for receipt in record["source_artifacts"]:
        artifact(receipt["path"], receipt["sha256"])
    check_source_seal()
    if record["estimate"]["projected_full_campaign_upper_s"] > LIMITS["gpu_seconds"]:
        raise RuntimeError("STOP_RESOURCE: full twelve-fit upper projection exceeds three hours")
    return record


def check_remaining_projection(preflight):
    budget = budget_snapshot()
    ledger = read_json(HERE / "ledger.json")
    if any(row["status"] != "complete" for row in ledger["fits"]):
        raise RuntimeError("Preserved failed/partial fit requires inspection before any continuation")
    attempted = {row["id"] for row in ledger["fits"]}
    remaining = sum(preflight["estimate"]["units"][dataset]["fit_upper_s"]
                    for dataset in DATASETS for lr in (1e-3, 1e-4) for seed in SEEDS
                    if run_id(dataset, lr, seed) not in attempted)
    remaining += preflight["estimate"]["post_training_upper_s"]
    if budget["gpu_seconds"] + remaining >= LIMITS["gpu_seconds"]:
        raise RuntimeError("STOP_RESOURCE: consumed GPU time plus full remaining scope exceeds cap")


def fit(dataset, lr, seed):
    protocol = validate_protocol()
    if dataset not in DATASETS or lr not in protocol["learning_rates"] or seed not in SEEDS:
        raise ValueError("Unapproved fit specification")
    preflight = require_preflight()
    check_remaining_projection(preflight)
    identifier = run_id(dataset, lr, seed)
    out, local = HERE / "runs" / identifier, CACHE / "runs" / identifier
    if out.exists() or local.exists():
        raise RuntimeError("Existing fit attempt must be preserved; no automatic duplicate/retry")
    spec = {"id": identifier, "dataset": dataset, "method": "MATCHED_LEARNED_ED", "lr": lr,
            "seed": seed, "epochs": 120, "effective_batch_origins": 4}
    with Job("gpu", identifier, metadata=spec) as job:
        out.mkdir(parents=True)
        local.mkdir(parents=True)
        begin_fit(identifier, spec)
        started = time.perf_counter()
        steps, epoch, history, schedules, oom_events, microbatch = 0, 0, [], [], [], 4
        attempt = {"id": identifier, "spec": spec, "status": "running", "job_id": job.id,
                   "protocol": artifact(HERE / "protocol.json"), "preflight": artifact(HERE / "preflight.json"),
                   "source_seal": artifact(HERE / "source_seal.json"), "environment": environment(),
                   "test_accessed": False, "actual_updates": 0}
        save_json(out / "attempt.json", attempt)
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            data = load_data(dataset, include_test=False)
            model = build_model(dataset, device="cuda")
            initial = model.adapter_state()
            initial_hash = tensor_hashes(initial)
            backbone_before = state_hash(model.backbone)
            basis_before = tensor_hashes({"basis": model.basis0})["basis"]
            frozen_versions = {name: p._version for name, p in model.backbone.named_parameters()}
            torch.save(make_checkpoint(model, spec, epoch=0, steps=0, adapter_initial_sha256=initial_hash,
                                       backbone_state_sha256=backbone_before), local / "initial.pt")
            optimizer, scheduler = optimizer_for(model, lr)
            torch.cuda.reset_peak_memory_stats()
            best, best_epoch, stale = math.inf, 0, 0
            first_gradients = {}
            for epoch in range(121):
                losses = []
                if epoch:
                    schedule = sample_epoch_origins(dataset, seed, epoch)
                    schedules.append({"epoch": epoch, "count": 512, "sha256": array_hash(schedule.astype("<i8")),
                                      "phase_period": PHASE_PERIODS[dataset]})
                    for start in range(0, 512, 4):
                        job.check_limits()
                        origins = schedule[start:start + 4]
                        x_cpu, (target, mask) = inputs(data, origins), targets(data, origins)
                        before_rng = rng_state()
                        while True:
                            optimizer.zero_grad(set_to_none=True)
                            try:
                                loss, gradients = backward_effective_batch(model, x_cpu, target, mask, microbatch)
                                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                              1.0, error_if_nonfinite=True)
                                break
                            except torch.cuda.OutOfMemoryError as error:
                                oom_events.append({"epoch": epoch, "effective_batch_index": start // 4,
                                                   "origins_sha256": array_hash(origins), "failed_microbatch": microbatch,
                                                   "optimizer_updates_before_event": steps, "optimizer_step_performed": False,
                                                   "error": str(error)})
                                save_json(out / "oom_events.json", oom_events)
                                optimizer.zero_grad(set_to_none=True)
                                gc.collect()
                                torch.cuda.empty_cache()
                                restore_rng(before_rng)
                                if microbatch == 1:
                                    raise RuntimeError("STOP_RESOURCE: effective batch4 impossible at microbatch1") from error
                                microbatch //= 2
                        if steps < 2:
                            first_gradients[str(steps + 1)] = gradients
                        optimizer.step()
                        steps += 1
                        losses.append(loss)
                        if steps % 16 == 0:
                            job.heartbeat({"epoch": epoch, "batch": start // 4, "microbatch": microbatch}, updates=steps)
                        del x_cpu, target, mask
                val, prediction = evaluate(model, data, data["val_origins"], job)
                history.append({"epoch": epoch, "steps": steps, "train_loss": float(np.mean(losses)) if losses else None,
                                "val_mse": val["mse"], "val_mae": val["mae"], "lr": optimizer.param_groups[0]["lr"],
                                "microbatch": microbatch, "elapsed_s": time.perf_counter() - started})
                if val["mse"] < best:
                    best, best_epoch, stale = val["mse"], epoch, 0
                    torch.save(make_checkpoint(model, spec, epoch=epoch, steps=steps,
                                               adapter_initial_sha256=initial_hash, backbone_state_sha256=backbone_before),
                               local / "best.pt")
                    np.savez_compressed(local / "best_val.npz", prediction=prediction, origins=data["val_origins"])
                else:
                    stale += 1
                if epoch:
                    scheduler.step(val["mse"])
                torch.save(make_checkpoint(model, spec, epoch=epoch, steps=steps,
                                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                                           best=best, best_epoch=best_epoch, stale=stale, history=history, schedules=schedules,
                                           microbatch=microbatch, oom_events=oom_events,
                                           boundary="post-VAL/post-scheduler/complete epoch"), local / "last.pt")
                save_json(out / "curve.json", history)
                job.heartbeat({"epoch": epoch, "best_epoch": best_epoch, "val_mse": val["mse"]}, updates=steps)
                print(f"{identifier} epoch={epoch} best={best_epoch} val_mse={val['mse']:.9g} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if epoch and stale >= 6:
                    break
            final = model.adapter_state()
            final_updates = {name: float((final[name] - initial[name]).abs().max()) for name in initial}
            if not all(value > 0 for value in final_updates.values()):
                raise RuntimeError("STOP_DEBUG: E/D did not both update in the fit")
            backbone_after = state_hash(model.backbone)
            if backbone_before != backbone_after or any(p.grad is not None or p._version != frozen_versions[name]
                                                         for name, p in model.backbone.named_parameters()):
                raise RuntimeError("STOP_DEBUG: frozen backbone weights/buffers/versions changed")
            if tensor_hashes({"basis": model.basis0})["basis"] != basis_before:
                raise RuntimeError("STOP_DEBUG: literal PCA reference buffer changed")
            peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved(),
                    "scope": "One ED model+optimizer, full TRAIN/VAL, before selected checkpoint replay"}
            model.restore_adapter(torch.load(local / "best.pt", map_location="cpu", weights_only=False)["state"])
            selected_state = model.adapter_state()
            selected_updates = {name: float((selected_state[name] - initial[name]).abs().max()) for name in initial}
            if best_epoch > 0 and not all(value > 0 for value in selected_updates.values()):
                raise RuntimeError("STOP_DEBUG: non-step0 selected E/D did not both update")
            replay_score, replay = evaluate(model, data, data["val_origins"], job)
            with np.load(local / "best_val.npz", allow_pickle=False) as archive:
                replay_error = float(np.abs(replay.astype(np.float64) - archive["prediction"]).max())
                if not np.array_equal(archive["origins"], data["val_origins"]):
                    raise RuntimeError("Selected VAL origins changed")
            if replay_error > REPLAY["atol"] or abs(replay_score["mse"] - best) > 1e-8:
                raise RuntimeError("STOP_DEBUG: selected VAL checkpoint replay differs")
            if state_hash(model.backbone) != backbone_before:
                raise RuntimeError("STOP_DEBUG: selected VAL replay changed frozen backbone state")
            result = {**attempt, "status": "complete", "dataset": dataset, "lr": lr, "seed": seed,
                      "checkpoint": artifact(local / "best.pt"), "best_val_prediction": artifact(local / "best_val.npz"),
                      "initial_checkpoint": artifact(local / "initial.pt"), "last_checkpoint": artifact(local / "last.pt"),
                      "best_val_mse": best, "selected_epoch": best_epoch, "initial_val_mse": history[0]["val_mse"],
                      "epochs_completed": epoch, "total_steps": steps, "actual_updates": steps,
                      "effective_batch_origins": 4, "microbatch": microbatch, "oom_events": oom_events,
                      "adapter_initial_sha256": initial_hash, "adapter_selected_sha256": tensor_hashes(selected_state),
                      "final_parameter_max_updates": final_updates, "selected_parameter_max_updates": selected_updates,
                      "selected_non_step0_ED_updated": best_epoch == 0 or all(v > 0 for v in selected_updates.values()),
                      "first_update_gradients": first_gradients, "model_receipt": model.receipt(),
                      "frozen_backbone": {"before_sha256": backbone_before, "after_sha256": backbone_after,
                                          "unchanged": True, "parameter_gradients_absent": True,
                                          "parameter_versions_unchanged": True, "optimizer_exclusion": True,
                                          "eval_mode": not model.backbone.training},
                      "basis_values_sha256": basis_before, "schedules": schedules, "training_peak": peak,
                      "elapsed_s": time.perf_counter() - started, "replay_max_absolute": replay_error,
                      "val_origin_values_sha256": array_hash(data["val_origins"]),
                      "data": artifact(data["_path"]), "test_accessed": False}
            save_json(out / "result.json", result)
            save_json(HERE / "run_results" / f"{identifier}.json", result)
            finish_fit(identifier, {"epochs_completed": epoch, "actual_updates": steps, "selected_epoch": best_epoch,
                                    "result": artifact(out / "result.json"),
                                    "canonical_run_result": artifact(HERE / "run_results" / f"{identifier}.json"),
                                    "elapsed_s": result["elapsed_s"]})
            attempt.update(status="complete", actual_updates=steps, result=artifact(out / "result.json"))
            save_json(out / "attempt.json", attempt)
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
            return result
        except BaseException as error:
            status = "STOP_RESOURCE" if "STOP_RESOURCE" in str(error) else "failed"
            attempt.update(status=status, actual_updates=steps, epochs_completed=epoch,
                           elapsed_s=time.perf_counter() - started, error=traceback.format_exc(),
                           partial_curve=history, schedules=schedules, oom_events=oom_events)
            save_json(out / "attempt.json", attempt)
            save_json(out / "partial.json", attempt)
            finish_fit(identifier, {"actual_updates": steps, "epochs_completed": epoch,
                                    "attempt": artifact(out / "attempt.json"), "elapsed_s": attempt["elapsed_s"]}, status=status)
            raise


def select():
    protocol = validate_protocol()
    require_preflight()
    path = HERE / "selection.json"
    if path.exists():
        raise RuntimeError("Existing joint selection must be preserved")
    results, units = [], {}
    for dataset in DATASETS:
        group = []
        for lr in protocol["learning_rates"]:
            for seed in SEEDS:
                result = read_json(HERE / "runs" / run_id(dataset, lr, seed) / "result.json")
                if result["status"] != "complete" or (result["dataset"], result["lr"], result["seed"]) != (dataset, lr, seed):
                    raise RuntimeError("Joint selection lacks a completed canonical fit")
                for key in ("checkpoint", "best_val_prediction"):
                    artifact(result[key]["path"], result[key]["sha256"])
                group.append(result)
        means = [{"lr": lr, "mean_best_val_mse": float(np.mean([row["best_val_mse"] for row in group if row["lr"] == lr]))}
                 for lr in protocol["learning_rates"]]
        preferred = protocol["lr_exact_tie"][dataset]
        chosen = min(means, key=lambda row: (row["mean_best_val_mse"], row["lr"] != preferred))
        selected = sorted([row for row in group if row["lr"] == chosen["lr"]], key=lambda row: row["seed"])
        units[dataset] = {"lr": chosen["lr"], "seeds": list(SEEDS), "run_ids": [row["id"] for row in selected],
                          "checkpoints": [row["checkpoint"] for row in selected],
                          "best_val_prediction": [row["best_val_prediction"] for row in selected],
                          "selected_epochs": [row["selected_epoch"] for row in selected],
                          "mean_best_val_mse": chosen["mean_best_val_mse"], "lr_selection": means,
                          "exact_tie_preferred_lr": preferred}
        for seed in SEEDS:
            seeded = [row for row in group if row["seed"] == seed]
            for epoch in range(1, min(row["epochs_completed"] for row in seeded) + 1):
                hashes = {next(record["sha256"] for record in row["schedules"] if record["epoch"] == epoch) for row in seeded}
                if len(hashes) != 1:
                    raise RuntimeError("LR fits did not use the identical parent seed/epoch sample order")
            if len({json_hash(row["adapter_initial_sha256"]) for row in seeded}) != 1:
                raise RuntimeError("LR fits did not begin at identical literal PCA E/D")
        results.extend(group)
    output = {"schema": "matched_learned_ed_v18_selection_1", "units": units,
              "all_runs": [{name: row[name] for name in ("id", "dataset", "lr", "seed", "checkpoint", "best_val_prediction",
                                                         "best_val_mse", "selected_epoch", "epochs_completed", "total_steps")}
                           for row in results], "test_used_for_selection": False,
              "criterion": "Minimum full VAL including step0; strict improvement/earliest tie; arithmetic two-seed mean LR",
              "prediction_ensemble": False, "protocol": artifact(HERE / "protocol.json"),
              "preflight": artifact(HERE / "preflight.json"), "source_seal": artifact(HERE / "source_seal.json")}
    save_json(path, output)
    return output


def json_hash(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--lr", type=float, choices=(1e-3, 1e-4))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    configure()
    if args.select:
        with Job("cpu", "joint_VAL_only_selection"):
            select()
    elif args.dataset is not None and args.lr is not None and args.seed is not None:
        fit(args.dataset, args.lr, args.seed)
    else:
        parser.error("Use --dataset --lr --seed for one fit or --select after all twelve fits")


if __name__ == "__main__":
    main()
