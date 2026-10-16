"""Fresh-process memory and cyclic general-session deployment measurements."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import campaign
import data_contract as data
import route_catalog
from resources import BudgetExceeded, GPUReservation, check_storage_budget

HERE, CACHE, ASSET_ROOT = campaign.HERE, campaign.CACHE, campaign.ASSET_ROOT


def model_path(size):
    if size in {None, "small"}:
        return ASSET_ROOT / ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots" / campaign.SMALL_REV
    return CACHE / json.loads((HERE / "MODEL_ASSETS.json").read_text(encoding="utf-8"))["models"][size]["cache_relative_path"]


def execute(specs, size, device, batch, mode, private, public, label, lease):
    try:
        lease.check()
        if lease.remaining < 30:
            raise BudgetExceeded("Cost reservation is too close to its end for another child")
    except (BudgetExceeded, TimeoutError) as error:
        with (public / (label + ".execution.json")).open("x", encoding="utf-8") as handle:
            json.dump(dict(label=label, mode=mode, batch=batch, device=device, model_size=size,
                           returncode=None, child_started=False, failure_class="DEFERRED_RESOURCE",
                           reason=type(error).__name__, completed_unix=time.time()), handle, indent=2)
        raise
    spec_path = private / (label + ".json")
    with spec_path.open("x", encoding="utf-8") as handle:
        json.dump(specs, handle, indent=2)
    output = public / (label + ".json")
    command = [sys.executable, "-X", "utf8", "-u", str(HERE / "measure_costs.py"), "--" + mode,
               "--route-spec", str(spec_path), "--cache-root", str(CACHE), "--model", str(model_path(size)),
               "--device", device, "--batch", str(batch), "--warmups", "10", "--timed-trials", "20",
               "--stage-trials", "20", "--output", str(output)]
    started = time.time()
    print(json.dumps(dict(event="COST_START", label=label, mode=mode, batch=batch)), flush=True)
    env = dict(os.environ, **lease.child_environment(), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    with (private / (label + ".log")).open("xb") as stream:
        code = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT).wait()
    record = dict(label=label, mode=mode, batch=batch, device=device, model_size=size, returncode=code,
                  started_unix=started, completed_unix=time.time(), success_receipt_present=output.is_file(),
                  private_log_sha256=data.sha256_file(private / (label + ".log")))
    budget_stop = identity_failure = False
    if code == 0 and output.is_file():
        record["receipt_relative_path"] = output.relative_to(HERE).as_posix()
    else:
        text = (private / (label + ".log")).read_text(encoding="utf-8", errors="replace")
        budget_stop = "BudgetExceeded:" in text or ("TimeoutError:" in text and any(token in text.lower() for token in ("budget", "deadline", "reservation")))
        identity_failure = any(token in text for token in ("foundation identity", "foundation asset identity", "fingerprint changed", "exact original frozen foundation"))
        record["failure_class"] = ("DEFERRED_RESOURCE" if budget_stop else "FAILED_IDENTITY_CONTRACT" if identity_failure else
                                   "BLOCKED_CUDA_OOM" if "OutOfMemoryError" in text or "CUDA out of memory" in text else "FAILED_INSPECT_PRIVATE_LOG")
    with (public / (label + ".execution.json")).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(dict(event="COST_DONE", label=label, returncode=code)), flush=True)
    if budget_stop:
        raise BudgetExceeded("Cost child stopped at the shared resource boundary; execution receipt/log preserved")
    if identity_failure:
        raise ValueError("Cost child failed exact foundation/artifact identity; inspect its preserved execution receipt")
    return json.loads(output.read_text(encoding="utf-8")) if code == 0 and output.is_file() else None


def condition_plan(routes, skip_raw_variants=False):
    plan = []
    for batch in (1, 8):
        groups = []
        for index, (spec, size) in enumerate(routes):
            device = "cpu" if spec["family"] in {"RAW_RIDGE", "CURRENT", "SEASONAL"} else "cuda"
            variants = ("full", 32, 8, 4) if spec["family"] == "RAW" and not skip_raw_variants else (spec.get("row_microbatch", 128),)
            for row_size in variants:
                plan.append(dict(label=f"fresh_b{batch}_m{index:03d}_rows{row_size}", mode="worker", batch=batch,
                                 route_index=index, row_microbatch=row_size, device=device, model_size=size))
            group = (device, size or "small")
            if group not in groups: groups.append(group)
        for index, (device, size) in enumerate(groups):
            plan.append(dict(label=f"balanced_b{batch}_g{index:02d}", mode="balanced", batch=batch,
                             device=device, model_size=size))
    return plan


def run_conditions(routes, private, public, lease, skip_raw_variants=False, resource_allocation=None):
    """Keep failed conditions and list every fresh/balanced task left at budget stop."""
    plan = condition_plan(routes, skip_raw_variants)
    with (public/"COST_CONDITIONS_PLAN.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(conditions=plan, created_unix=time.time(), source_sha256=data.sha256_file(__file__),
                       resource_allocation=resource_allocation), handle, indent=2)
    candidates, records, remaining, terminal = {}, [], [], None
    for position, condition in enumerate(plan):
        label, batch, mode = condition["label"], condition["batch"], condition["mode"]
        try:
            lease.check()
            check_storage_budget(CACHE, pending_bytes=64 * 1024 ** 2)
            if mode == "worker":
                original, size = routes[condition["route_index"]]
                spec = copy.deepcopy(original)
                spec.update(accuracy_run_id=original["run_id"], row_microbatch=condition["row_microbatch"],
                            run_id=original["run_id"]+"__rows"+str(condition["row_microbatch"]))
            else:
                key = (batch, condition["device"], condition["model_size"])
                per_route = candidates.get(key, {})
                spec = [min(values,key=lambda item:item[0])[1] for values in per_route.values()]
                size = condition["model_size"]
                if not spec:
                    records.append(dict(label=label,status="SKIPPED_NO_SUCCESSFUL_FRESH_ROUTES"))
                    continue
            result = execute(spec,size,condition["device"],batch,mode,private,public,label,lease)
            records.append(dict(label=label,status="MEASURED" if result is not None else "FAILED_CONDITION"))
            if mode == "worker" and result is not None:
                timed = next(summary for summary in result["summaries"] if summary["phase"] == "timed")
                group = (batch, condition["device"], size or "small")
                candidates.setdefault(group,{}).setdefault(condition["route_index"],[]).append((timed["median_seconds"],spec))
        except (BudgetExceeded,TimeoutError) as error:
            remaining, terminal = plan[position:], "DEFERRED_RESOURCE"
            records.append(dict(label=label,status="DEFERRED_RESOURCE",reason=type(error).__name__))
            break
        except Exception as error:
            remaining, terminal = plan[position:], "STOPPED_EXECUTION_ERROR"
            records.append(dict(label=label,status="EXECUTION_FAILED",error_type=type(error).__name__))
            summary = dict(status=terminal,conditions=records,remaining_conditions=remaining,
                           completed_unix=time.time(),actual_full_protocol_completed=False,
                           resource_allocation=resource_allocation)
            with (public/"COST_CAMPAIGN_RECEIPT.json").open("x",encoding="utf-8") as handle:json.dump(summary,handle,indent=2)
            raise
    result = dict(status=terminal or ("PARTIAL_CONDITION_FAILURES" if any(row["status"] != "MEASURED" for row in records) else "COMPLETE_REQUESTED_CONDITIONS"),
                  conditions=records,remaining_conditions=remaining,completed_unix=time.time(),
                  actual_full_protocol_completed=not remaining and all(row["status"]=="MEASURED" for row in records),
                  resource_allocation=resource_allocation)
    with (public/"COST_CAMPAIGN_RECEIPT.json").open("x",encoding="utf-8") as handle:json.dump(result,handle,indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-id", action="append", required=True)
    p.add_argument("--skip-raw-variants", action="store_true")
    p.add_argument("--reserve-final-seconds", type=float, default=3600,
                   help="Leave this much of the shared 43200-second budget for final replay/T; explicit 0 permits filling costs after final evaluation")
    a = p.parse_args()
    if not math.isfinite(a.reserve_final_seconds) or not 0 <= a.reserve_final_seconds < 43200:
        p.error("--reserve-final-seconds must be finite and in [0, 43200)")
    allocation = dict(global_GPU_process_seconds=43200,
                      reserved_for_final_inference_seconds=a.reserve_final_seconds,
                      cost_cumulative_GPU_ceiling_seconds=43200-a.reserve_final_seconds,
                      accounting_scope="cumulative exclusive process-wall in the shared GPU_USAGE ledger; not a fresh per-command budget")
    task = data.load_task(a.dataset)
    routes = [route_catalog.build_route(rid, task, CACHE, HERE) for rid in a.run_id]
    stamp = str(time.time_ns())
    public = HERE / "costs" / (a.dataset + "_" + stamp)
    private = CACHE / "costs" / (a.dataset + "_" + stamp)
    public.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
    with (public / "COST_CANDIDATES_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(dataset=a.dataset, run_ids=a.run_id, route_specs=[r[0] for r in routes],
                       batches=[1, 8], warmups=10, timed_trials=20, stage_trials=20, created_unix=time.time(),
                       raw_balanced_choice_rule="fastest successful fresh Raw row variant at the same batch; record all variants",
                       input_cohort="first registered V stride-H histories; never T outcomes", T_selection=False,
                       resource_allocation=allocation), handle, indent=2)
    try:
        with GPUReservation(CACHE, total_gpu_seconds=allocation["cost_cumulative_GPU_ceiling_seconds"], reserve_seconds=0,
                            wall_deadline=1792050350 + 172800, purpose="audit", kind="full_deployment_cost") as lease:
            run_conditions(routes,private,public,lease,a.skip_raw_variants,resource_allocation=allocation)
    except (BudgetExceeded,TimeoutError) as error:
        receipt = public/"COST_CAMPAIGN_RECEIPT.json"
        if receipt.exists():
            raise
        plan = condition_plan(routes,a.skip_raw_variants)
        if not (public/"COST_CONDITIONS_PLAN.json").exists():
            with (public/"COST_CONDITIONS_PLAN.json").open("x",encoding="utf-8") as handle:
                json.dump(dict(conditions=plan,created_unix=time.time(),source_sha256=data.sha256_file(__file__),
                               resource_allocation=allocation),handle,indent=2)
        with receipt.open("x",encoding="utf-8") as handle:
            json.dump(dict(status="DEFERRED_RESOURCE",reason=type(error).__name__,conditions=[],
                           remaining_conditions=plan,actual_full_protocol_completed=False,completed_unix=time.time(),
                           resource_allocation=allocation),handle,indent=2)


if __name__ == "__main__":
    main()
