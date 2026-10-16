"""Predeclared epoch extension, preserving every initial screening result."""
import argparse
import copy
import gc
import json
import time
import traceback

import torch

import campaign
import data_contract as data
import fit_base
import learning_extension
import ordered_pilot
from postfit import ParentPool
import registry_extension
from resources import GPUReservation, check_storage_budget

HERE, CACHE = campaign.HERE, campaign.CACHE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset")
    p.add_argument("--family", action="append")
    p.add_argument("--parent-id", action="append")
    p.add_argument("--max-epoch", type=int, choices=[1, 3, 10, 30], default=3)
    p.add_argument("--block", default="E1", choices=["E1", "E3", "E4"])
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--include-full-grid", action="store_true")
    a = p.parse_args()
    registry_extension.require_pre_selection(HERE, "epoch extension fitting")
    epochs = [e for e in (.25, 1., 3., 10., 30.) if e <= a.max_epoch]
    candidates = []
    for row in campaign.registry_rows():
        parent_id = row["experiment_id"]
        if row["family"] not in set(campaign.FAMILIES) | {"ORDERED_MULTI_K", "MAXK_PCA_TRAINED"} or row["family"] == "PCA_FIXED_BOLT":
            continue
        if a.parent_id:
            if parent_id not in a.parent_id:
                continue
        elif row["block"] != a.block or "__e" in parent_id or row["fold"] != "primary":
            continue
        if a.dataset and row["dataset"] != a.dataset:
            continue
        if a.family and row["family"] not in a.family:
            continue
        if not a.include_full_grid and row["requirement"] != "MANDATORY_MINIMUM" and not a.parent_id:
            continue
        path = HERE / "runs" / parent_id / "BASE_FIT_RECEIPT.json"
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETE_VALID" or max(receipt["epochs"]) >= a.max_epoch:
            continue
        new_id = parent_id + "__e" + str(a.max_epoch)
        candidates.append((row, receipt, new_id))
    if a.max_jobs:
        candidates = candidates[:a.max_jobs]
    proposals = []
    for row, _, new_id in candidates:
        proposals.append(dict(row, experiment_id=new_id, requirement="REGISTERED_TRAINING_EXTENSION",
                              execution_status="PENDING_QUEUE", training_status="NOT_STARTED", validity_status="NOT_CHECKED",
                              scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED",
                              planned_epoch_endpoints="|".join(map(str, epochs)),
                              reason="Registered A50 search extension and fresh A60 refit; initial run retained",
                              resume_command=f"python deepen.py --parent-id {row['experiment_id']} --max-epoch {a.max_epoch}"))
        if row["family"] in {"ORDERED_MULTI_K", "MAXK_PCA_TRAINED"}:
            c = int(row["num_channels"])
            for k in sorted({(c + 3) // 4, (c + 1) // 2}):
                proposals.append(dict(proposals[-1], experiment_id=new_id + f"__prefix_k{k}", latent_k=k,
                                      reason="Registered prefix inference of extended shared or maximum-only checkpoint"))
    registry_extension.register(HERE, proposals, "Predeclared epoch deepening under original plan; no T selection")
    if not candidates:
        print("No eligible completed screening parent")
        return
    torch.set_num_threads(2)
    pool = None
    task_name = None
    with GPUReservation(CACHE, total_gpu_seconds=43200, reserve_seconds=3600, wall_deadline=1792050350 + 172800,
                        purpose="training", kind="matched_learning_extension") as lease:
        try:
            for row, parent_receipt, new_id in candidates:
                lease.check(); check_storage_budget(CACHE, pending_bytes=512 * 1024 ** 2)
                if task_name != row["dataset"]:
                    if pool is not None:
                        pool.close()
                    task = data.load_task(row["dataset"])
                    pool = ParentPool(task, deadline=lease.deadline - 5)
                    task_name = row["dataset"]
                source_path = HERE / "runs" / row["experiment_id"] / "SOURCE_CONTRACT.json"
                source = json.loads(source_path.read_text(encoding="utf-8"))
                current_source_sha, current_files = fit_base._source_contract()
                reuse_search = source["files"] == current_files
                family = source["spec"]["family"]
                foundation = None if family in fit_base.DIRECT else pool.get_foundation()
                campaign.record_status(new_id, execution_status="RUNNING", reason="Verified epoch extension" if reuse_search else "Fresh restart: previous scientific source differs")
                print(json.dumps(dict(event="DEEPEN_START", parent=row["experiment_id"], run_id=new_id,
                                      epochs=epochs, search_state_reused=reuse_search)), flush=True)
                try:
                    if reuse_search:
                        receipt = learning_extension.extend_fit(row["experiment_id"], new_id, epochs, task, CACHE, HERE,
                                                                foundation, lease.deadline - 5)
                    else:
                        spec = copy.deepcopy(source["spec"])
                        spec.update(run_id=new_id, epochs=epochs, stage_label="FRESH_RESTART_EXTENDED_SCREEN_SOURCE_CHANGED",
                                    historical_parent_run_id=row["experiment_id"])
                        receipt = fit_base.fit_base(spec, task, CACHE, HERE, foundation, lease.deadline - 5)
                        path = HERE / "runs" / new_id / "RESTART_PROVENANCE.json"
                        if not path.exists():
                            campaign.new_json(path, dict(parent_run_id=row["experiment_id"], parent_artifact_sha=parent_receipt["artifact_sha256"],
                                                         original_source_sha=source["source_sha"], current_source_sha=current_source_sha,
                                                         search_weights_optimizer_RNG_reused=False, old_fit_preserved=True,
                                                         reason="Exact source requirement failed; fresh current-source execution, no equivalence assumption"))
                    campaign.record_status(new_id, execution_status=receipt["status"], training_status=receipt["training_status"],
                                           validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Search extension/fresh A60 refit receipt complete; last-best training status kept")
                    if receipt["status"] == "COMPLETE_VALID" and row["family"] in {"ORDERED_MULTI_K", "MAXK_PCA_TRAINED"}:
                        variant = "shared" if row["family"] == "ORDERED_MULTI_K" else "posthoc"
                        for k in ordered_pilot.k_grid(task):
                            alias = new_id if k == len(task["columns"]) else new_id + f"__prefix_k{k}"
                            ordered_pilot.predict_prefix(alias, task, k, CACHE, HERE, foundation, parent_receipt=receipt,
                                                         variant=variant, deadline=lease.deadline - 5)
                            campaign.record_status(alias, execution_status="COMPLETE_VALID", training_status=receipt["training_status"],
                                                   validity_status="PENDING_INDEPENDENT_VALIDATION", reason="Actual prefix inference of extended selected checkpoint; no new prefix-specific fit")
                    print(json.dumps(dict(event="DEEPEN_DONE", run_id=new_id, status=receipt["status"],
                                          selected_epoch=receipt.get("selected_epoch"), training_status=receipt["training_status"],
                                          accounting=receipt.get("extension_accounting"))), flush=True)
                except Exception as error:
                    private = CACHE / "failures"
                    private.mkdir(exist_ok=True)
                    campaign.new_json(private / (new_id + "_" + str(time.time_ns()) + ".json"),
                                      dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
                    campaign.record_status(new_id, execution_status="EXECUTION_FAILED", validity_status="INVALID_OR_NOT_COMPLETED",
                                           reason=type(error).__name__ + "; preserved logs; independent extensions continue")
                    print(json.dumps(dict(event="DEEPEN_FAILED", run_id=new_id, error_type=type(error).__name__)), flush=True)
                    if foundation is not None:
                        foundation.assert_frozen_unchanged()
                gc.collect(); torch.cuda.empty_cache()
        finally:
            if pool is not None:
                pool.close()


if __name__ == "__main__":
    main()
