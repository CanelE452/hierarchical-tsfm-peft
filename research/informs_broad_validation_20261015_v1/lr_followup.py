"""Fixed second-LR study: two fresh Electricity K81 fits, then actual V inference."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import torch

import campaign
import data_contract as data
import fit_base
from postfit import ParentPool
import registry_extension
from resources import BudgetExceeded, GPUReservation, ResourceBusy, check_storage_budget
from training import fingerprint


HERE, CACHE, ASSET_ROOT = campaign.HERE, campaign.CACHE, campaign.ASSET_ROOT
PARENTS = tuple(f"primary_electricity_E1_{family}_k81_s92601" for family in ("B0", "B2"))
EPOCHS = [.25, 1.]
TARGET_LR = .0001
SPEC_DIFFERENCES = {"run_id", "lr", "epochs"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Artifact path must remain inside its declared cache")
    return path


def _new_or_same(path, value):
    path = Path(path)
    if path.is_file():
        if _read(path) != value:
            raise ValueError("Existing LR-study provenance differs; preserve it and use another campaign")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _axes(task, spec):
    return dict(L=int(task["L"]), H=int(task["H"]), C=int(task["physical"].shape[1]),
                columns=list(task["columns"]), fold=spec.get("fold", "primary"),
                source_array_fingerprints=task.get("array_sha256", {}))


def _config_sha(spec, task):
    return fingerprint(dict(spec=spec, axes=_axes(task, spec)))


def build_plan(parent_ids=None, epochs=None):
    """Read-only preflight; no registration, prediction, model load or GPU lease."""
    registry_extension.require_pre_selection(HERE, "fixed second-LR study")
    ids, epochs = list(parent_ids or PARENTS), list(EPOCHS if epochs is None else epochs)
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(PARENTS):
        raise ValueError("Only the two declared Electricity B0/B2 K81 seed92601 parents are supported")
    if epochs != EPOCHS:
        raise ValueError("This bounded study requires epochs [.25, 1]; use registered deepen.py for later epoch3")
    rows = campaign.registry_rows()
    registry = {row["experiment_id"]: row for row in rows}
    if len(registry) != len(rows):
        raise ValueError("Registry contains duplicate IDs")
    task = data.load_task("electricity")
    if task["name"] != "electricity" or task["L"] != 512 or task["H"] != 48 or task["physical"].shape[1] != 321:
        raise ValueError("Actual Electricity task no longer matches L512 H48 C321")
    core_sha, core_files = fit_base._source_contract()
    bounds = data.split_ranges(task, "primary")
    scalers = {}
    for stage, end in (("search", bounds["A_fit"][1]), ("refit", bounds["A_stop"][1])):
        mean, std, _ = data.fit_statistics(task, end)
        scalers[stage] = fit_base.scaler_fingerprint(mean, std)
    foundation_path = ASSET_ROOT / ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots" / campaign.SMALL_REV
    foundation_files = {}
    for name in ("config.json", "model.safetensors"):
        path = foundation_path / name
        if not path.is_file():
            raise FileNotFoundError("Pinned local small foundation checkpoint is absent; no download attempted")
        foundation_files[name] = data.sha256_file(path)
    jobs, proposed = [], []
    for parent_id in ids:
        if parent_id not in registry:
            raise ValueError("Declared LR parent is absent from the actual registry")
        row = registry[parent_id]
        folder = HERE / "runs" / parent_id
        source_path, receipt_path = folder / "SOURCE_CONTRACT.json", folder / "BASE_FIT_RECEIPT.json"
        source, receipt = _read(source_path), _read(receipt_path)
        old = source["spec"]
        if (row["dataset"] != "electricity" or row["family"] not in {"B0", "B2"} or int(row["latent_k"]) != 81
                or int(row["seed"]) != 92601 or row["fold"] != "primary"
                or receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != parent_id
                or receipt.get("family") != row["family"] or float(receipt["selected_epoch"]) != 0.
                or old.get("run_id") != parent_id or old.get("dataset") != "electricity"
                or old.get("family") != row["family"] or old.get("latent_k") != 81
                or old.get("seed") != 92601 or old.get("fold") != "primary"
                or old.get("lr") != .001 or list(old.get("epochs", [])) != [.25]):
            raise ValueError("Parent scientific identity/LR/CP0 trigger differs from the bounded study")
        if source.get("files") != core_files or source.get("source_sha") != core_sha or receipt.get("source_sha") != core_sha:
            raise ValueError("Parent and current frozen six-file scientific source contracts differ")
        if source.get("data_sha") != task["data_sha256"] or source.get("config_sha") != _config_sha(old, task):
            raise ValueError("Actual parent data/config contract differs")
        if (receipt.get("search_fit_end") != bounds["A_fit"][1] or receipt.get("refit_fit_end") != bounds["A_stop"][1]
                or receipt.get("normalization_fit_end") != bounds["A_stop"][1]
                or receipt.get("refit_scaler_sha") != scalers["refit"]):
            raise ValueError("Actual parent A50/A60 split or normalization differs")
        artifact = _inside(CACHE, receipt["artifact_cache_relative_path"])
        if not artifact.is_file() or data.sha256_file(artifact) != receipt["artifact_sha256"]:
            raise ValueError("Actual parent selected checkpoint is absent or changed")
        parent_files = {source_path.relative_to(HERE).as_posix(): data.sha256_file(source_path),
                        receipt_path.relative_to(HERE).as_posix(): data.sha256_file(receipt_path)}
        initializations = {}
        for stage in ("search", "refit"):
            path = folder / stage / "RECEIPT.json"
            summary = _read(path)
            if summary["contract"]["scaler_sha"] != scalers[stage] or not summary["contract"].get("initialization_sha"):
                raise ValueError("Actual parent stage normalization/initialization evidence differs")
            parent_files[path.relative_to(HERE).as_posix()] = data.sha256_file(path)
            initializations[stage] = summary["contract"]["initialization_sha"]
        new_id = parent_id + "__lr0001_e1"
        spec = copy.deepcopy(old)
        spec.update(run_id=new_id, lr=TARGET_LR, epochs=epochs)
        unchanged = {key: value for key, value in old.items() if key not in SPEC_DIFFERENCES}
        if unchanged != {key: value for key, value in spec.items() if key not in SPEC_DIFFERENCES}:
            raise ValueError("LR study modified another model/training hyperparameter")
        provenance = dict(run_id=new_id, parent_run_id=parent_id, fixed_LR_grid=[.001, TARGET_LR],
            changed_spec_keys=sorted(SPEC_DIFFERENCES), unchanged_spec_sha256=fingerprint(unchanged),
            original_spec=old, new_spec=spec, parent_public_files=parent_files,
            parent_artifact_cache_relative_path=receipt["artifact_cache_relative_path"], parent_artifact_sha256=receipt["artifact_sha256"],
            frozen_scientific_source_sha=core_sha, frozen_scientific_files=core_files, data_sha256=task["data_sha256"],
            parent_config_sha256=source["config_sha"], new_config_sha256=_config_sha(spec, task),
            canonical_stage_scaler_sha256=scalers, parent_initialization_sha256=initializations,
            foundation_state_sha256=source["model_sha"], foundation_snapshot=campaign.SMALL_REV, foundation_files_sha256=foundation_files,
            origin_fingerprints=dict(A_fit=data.array_sha256(data.origins(bounds["A_fit"],512,48,1)),
                                     A_stop=data.array_sha256(data.origins(bounds["A_stop"],512,48,48))),
            runner_source_sha256=data.sha256_file(__file__), fresh_A50_search=True, fresh_A60_refit=True,
            parent_weights_optimizer_RNG_reused=False, checkpoint_selection="A_stop normalized channel macro MSE",
            V_used_for_LR_choice=False, M_parents_replaced=False, T_accessed=False,
            comparison_limit="Parent max epoch .25 vs new max epoch1; compare common .25 A_stop curves or matched later epochs for LR attribution")
        if provenance["new_config_sha256"] == provenance["parent_config_sha256"]:
            raise ValueError("LR follow-up must have a new configuration identity")
        jobs.append(dict(parent_id=parent_id, run_id=new_id, spec=spec, provenance=provenance))
        proposed.append(dict(row, experiment_id=new_id, requirement="REGISTERED_FIXED_SECOND_LR_FOLLOWUP",
            execution_status="PENDING_QUEUE", training_status="NOT_STARTED", validity_status="NOT_CHECKED",
            scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED", planned_epoch_endpoints="0.25|1.0",
            reason="MASTER fixed second LR1e-4; fresh A50/A60; A_stop checkpoint choice; V not used for LR selection",
            resume_command=f"python lr_followup.py --parent-id {parent_id} --epochs .25 1"))
    return task, jobs, proposed


def _verify_parent_and_core(job):
    proof = job["provenance"]
    if fit_base._source_contract() != (proof["frozen_scientific_source_sha"], proof["frozen_scientific_files"]):
        raise ValueError("Frozen scientific source changed after LR preflight")
    for relative, sha in proof["parent_public_files"].items():
        if data.sha256_file(_inside(HERE, relative)) != sha:
            raise ValueError("Parent public evidence changed after LR preflight")
    if data.sha256_file(_inside(CACHE, proof["parent_artifact_cache_relative_path"])) != proof["parent_artifact_sha256"]:
        raise ValueError("Parent selected checkpoint changed after LR preflight")


def _verify_fresh_initializations(job):
    proof = job["provenance"]
    source = _read(HERE / "runs" / job["run_id"] / "SOURCE_CONTRACT.json")
    if source["spec"] != job["spec"] or source["config_sha"] != proof["new_config_sha256"]:
        raise ValueError("Realized LR source/config differs from predeclared fresh fit")
    for stage in ("search", "refit"):
        summary = _read(HERE / "runs" / job["run_id"] / stage / "RECEIPT.json")
        if summary["contract"]["initialization_sha"] != proof["parent_initialization_sha256"][stage]:
            raise ValueError("Fresh LR fit changed the matched model initialization")
        if summary["contract"]["scaler_sha"] != proof["canonical_stage_scaler_sha256"][stage]:
            raise ValueError("Fresh LR fit changed the matched normalization")


def run(parent_ids=None, epochs=None):
    task, jobs, proposed = build_plan(parent_ids, epochs)
    batch = HERE / "execution_batches" / ("lr_followup_" + str(time.time_ns()))
    batch.mkdir(parents=True, exist_ok=False)
    _new_or_same(batch / "PLAN.json", dict(run_ids=[job["run_id"] for job in jobs], source_sha256=data.sha256_file(__file__),
        epochs=list(EPOCHS), LR=TARGET_LR, reserve_GPU_seconds=9000, T_accessed=False, V_LR_selection=False))
    results, status = [], "PAIR_FINISHED_REQUIRES_INDEPENDENT_AUDIT"
    try:
        with GPUReservation(CACHE, total_gpu_seconds=43200, reserve_seconds=9000,
                            wall_deadline=1792050350+172800, purpose="training", kind="fixed_second_LR_followup") as lease:
            # The lease is an OS ownership/accounting guard, not a CUDA load.
            # Register under that guard before the first model, fit or forecast.
            registry_extension.require_pre_selection(HERE, "fixed second-LR registration")
            registry_extension.register(HERE, proposed, "MASTER §7.3 fixed second LR; two bounded Electricity K81 fresh fits; before V/T choice")
            for job in jobs:
                _new_or_same(HERE / "runs" / job["run_id"] / "LR_FOLLOWUP_PROVENANCE.json", job["provenance"])
            torch.set_num_threads(2)
            pool = ParentPool(task, deadline=lease.deadline-5)
            try:
                for index, job in enumerate(jobs):
                    receipt = None
                    phase = "pre_fit"
                    try:
                        lease.check(); registry_extension.require_pre_selection(HERE, "fixed LR fresh fitting")
                        _verify_parent_and_core(job)
                        check_storage_budget(CACHE, pending_bytes=256*1024**2)
                        foundation = pool.get_foundation("small")
                        if fit_base._model_sha(foundation) != job["provenance"]["foundation_state_sha256"]:
                            raise ValueError("Actual frozen foundation differs from parent source contract")
                        lease.check()
                        campaign.record_status(job["run_id"], execution_status="RUNNING", reason="Fresh fixed second-LR A50 search; A_stop selection then fresh A60 refit")
                        print(json.dumps(dict(event="LR_FIT_START",run_id=job["run_id"],lr=TARGET_LR,epochs=EPOCHS)),flush=True)
                        phase = "fit"
                        receipt = fit_base.fit_base(job["spec"], task, CACHE, HERE, foundation, lease.deadline-5)
                        if receipt["status"] != "COMPLETE_VALID":
                            campaign.record_status(job["run_id"],execution_status=receipt["status"],training_status=receipt["training_status"],
                                validity_status="INVALID_OR_NOT_COMPLETED",reason="Fixed LR fit incomplete; original parents preserved; same CLI can resume unchanged contract")
                            results.append(dict(run_id=job["run_id"],status=receipt["status"],phase=receipt.get("phase")))
                            continue
                        _verify_fresh_initializations(job)
                        if receipt["refit_scaler_sha"] != job["provenance"]["canonical_stage_scaler_sha256"]["refit"]:
                            raise ValueError("Realized fresh A60 scaler differs")
                        phase = "V_prediction"
                        lease.check()
                        existed = (HERE / "predictions" / job["run_id"] / "V_select/PREDICTION_MANIFEST.json").exists()
                        before = int(foundation.streams)
                        prediction = pool.predict(job["run_id"])
                        reused = bool(prediction["metadata"].get("reused_prediction_cache"))
                        actual_new_F_streams = int(foundation.streams)-before
                        if not existed and (reused or actual_new_F_streams <= 0):
                            raise ValueError("New LR route requires actual V foundation inference; cache-only result refused")
                        if not reused and actual_new_F_streams != prediction["cost_record"]["actual_F_streams"]:
                            raise ValueError("Actual V foundation counters differ from prediction ledger")
                        meta = prediction["metadata"]
                        if meta["run_id"] != job["run_id"] or meta["artifact_sha256"] != receipt["artifact_sha256"] or meta["data_sha256"] != task["data_sha256"]:
                            raise ValueError("Actual V cache belongs to another LR artifact/data")
                        campaign.record_status(job["run_id"],execution_status="PARTIAL_SCREEN_LT3",training_status=receipt["training_status"],
                            validity_status="PENDING_INDEPENDENT_VALIDATION",reason="Fresh LR1e-4 maxepoch1 selected A60 artifact and actual V inference; epoch3/convergence unresolved; M/E2 parents unchanged")
                        outcome = dict(run_id=job["run_id"], status="PARTIAL_SCREEN_LT3", selected_epoch=receipt["selected_epoch"],
                            training_status=receipt["training_status"], search_updates=receipt["search_updates"], refit_updates=receipt["refit_updates"],
                            artifact_sha256=receipt["artifact_sha256"], new_config_sha256=job["provenance"]["new_config_sha256"],
                            V_score=prediction["score"], V_prediction_executed_now=not reused, actual_new_V_F_streams=actual_new_F_streams,
                            fit_artifact_reused=bool(receipt.get("reused_complete_artifact")), V_used_for_LR_choice=False)
                        _new_or_same(batch / (job["run_id"]+".json"),outcome)
                        results.append(outcome)
                        print(json.dumps(dict(event="LR_FIT_AND_V_DONE",**outcome)),flush=True)
                    except (BudgetExceeded, TimeoutError):
                        training_status = receipt.get("training_status", "UNDERTRAINED_OR_BUDGET_LIMIT") if receipt else "UNDERTRAINED_OR_BUDGET_LIMIT"
                        campaign.record_status(job["run_id"],execution_status="DEFERRED_RESOURCE",training_status=training_status,
                            reason="LR follow-up resource deadline at "+phase+"; selected artifact/full-carry state preserved; no new trial after deadline")
                        status="DEFERRED_RESOURCE";results.extend(dict(run_id=job["run_id"],status="DEFERRED_RESOURCE") for job in jobs[index:]);break
                    except Exception as error:
                        private=CACHE/"failures"/(job["run_id"]+"_"+str(time.time_ns())+".json")
                        _new_or_same(private,dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
                        campaign.record_status(job["run_id"],execution_status="EXECUTION_FAILED",validity_status="INVALID_OR_NOT_COMPLETED",reason=type(error).__name__+"; independent paired route continues")
                        results.append(dict(run_id=job["run_id"],status="EXECUTION_FAILED",error_type=type(error).__name__))
                        print(json.dumps(dict(event="LR_FOLLOWUP_FAILED",**results[-1])),flush=True)
            finally:
                pool.close()
    except (BudgetExceeded, ResourceBusy) as error:
        status="DEFERRED_RESOURCE_OR_ACTIVE_OWNER"
        completed={row["run_id"] for row in results}
        results.extend(dict(run_id=job["run_id"],status=status) for job in jobs if job["run_id"] not in completed)
    if status=="PAIR_FINISHED_REQUIRES_INDEPENDENT_AUDIT" and any(row["status"]!="PARTIAL_SCREEN_LT3" for row in results):
        status="PARTIAL_FIT_OR_ROUTE_FAILURES"
    receipt=dict(status=status,results=results,source_sha256=data.sha256_file(__file__),T_accessed=False,
        V_used_for_LR_choice=False,original_parents_replaced=False,M_E2_parents_replaced=False,
        campaign_complete=False,convergence_claimed=False,publication_performed=False)
    _new_or_same(batch/"EXECUTION_RECEIPT.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id",action="append")
    parser.add_argument("--epochs",type=float,nargs="+",default=EPOCHS)
    args=parser.parse_args()
    print(json.dumps(run(args.parent_id,args.epochs),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
