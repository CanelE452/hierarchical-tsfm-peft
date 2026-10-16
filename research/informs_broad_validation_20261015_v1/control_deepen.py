"""Fresh three-epoch DIRECT_G A+M controls; original short runs stay immutable."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import data_contract as data
import registry_extension
from resources import BudgetExceeded, GPUReservation, ResourceBusy, check_storage_budget


HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("INFORMS_CACHE_ROOT", HERE.parents[1] / ".cache" / HERE.name))
DATASETS = ("beijing", "jena", "electricity", "pems08")
EPOCHS = [.25, 1., 3.]
CORE = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fingerprint(value):
    # Same canonical JSON as training.fingerprint, without importing Torch before registration.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    target = (root / relative).resolve()
    if relative.is_absolute() or target == root or not target.is_relative_to(root):
        raise ValueError("Artifact path must remain inside the declared cache")
    return target


def _new_or_same(path, value):
    path = Path(path)
    if path.is_file():
        if _read(path) != value:
            raise ValueError("Existing immutable control follow-up provenance differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _registry():
    with (HERE / "EXPERIMENT_REGISTRY.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["experiment_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate experiment registry IDs")
    return rows


def _record(run_id, **fields):
    rows = _registry()
    selected = [row for row in rows if row["experiment_id"] == run_id]
    if len(selected) != 1:
        raise ValueError("Expected exactly one registered control follow-up")
    selected[0].update({key: str(value) for key, value in fields.items()})
    with (HERE / "EXPERIMENT_REGISTRY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _source_contract():
    files = {name: data.sha256_file(HERE / name) for name in CORE}
    engine = _fingerprint(files)
    files["fit_control.py"] = data.sha256_file(HERE / "fit_control.py")
    return _fingerprint(files), engine, files


def _contract(spec, task, source_sha, engine_sha):
    bounds = data.split_ranges(task, spec["fold"])
    axes = dict(L=int(task["L"]), H=int(task["H"]), C=int(task["physical"].shape[1]),
                columns=list(task["columns"]), fold=spec["fold"],
                normalization_end=bounds["A_stop"][1], train_end=bounds["M_cal"][1],
                evaluation_bounds=list(bounds["V_select"]), data_arrays=task.get("array_sha256", {}))
    return dict(source_sha=source_sha, engine_source_sha=engine_sha,
                config_sha=_fingerprint(dict(spec=spec, axes=axes)), data_sha=task["data_sha256"], model_sha="NO_FOUNDATION")


def _scaler(task):
    import numpy as np
    mean, std, _ = data.fit_statistics(task, data.split_ranges(task, "primary")["A_stop"][1])
    def array_proof(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return _fingerprint(dict(mean=array_proof(mean), std=array_proof(std)))


def _validate_complete(run_id, spec, task, expected, scaler):
    folder = HERE / "runs" / run_id
    source = _read(folder / "CONTROL_SOURCE_CONTRACT.json")
    receipt = _read(folder / "CONTROL_FIT_RECEIPT.json")
    if any(source.get(key) != value for key, value in expected.items()) or source.get("spec") != spec or source.get("files") != _source_contract()[2]:
        raise ValueError("Actual control source/spec/data/model contract differs")
    bounds = data.split_ranges(task, "primary")
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id
            or receipt.get("family") != "DIRECT_G" or receipt.get("role") != "AplusM_control"
            or receipt.get("source_sha") != expected["source_sha"] or receipt.get("epochs") != spec["epochs"]
            or receipt.get("normalization_fit_end") != bounds["A_stop"][1]
            or receipt.get("control_fit_end") != bounds["M_cal"][1] or receipt.get("scaler_sha") != scaler
            or receipt.get("checkpoint_selection_split") != "V_select" or receipt.get("F_logical_streams") != 0
            or receipt.get("fresh_initialization") is not True or not receipt.get("initial_model_state_sha")
            or receipt.get("pretrained_G_or_adapter_parent_reused") is not False
            or receipt.get("T_final_accessed") is not False):
        raise ValueError("Actual completed G control split/scaler/initialization/F0 identity differs")
    artifact = _inside(CACHE, receipt["artifact_cache_relative_path"])
    if not artifact.is_file() or data.sha256_file(artifact) != receipt["artifact_sha256"]:
        raise ValueError("Actual completed G control selected artifact absent or changed")
    private = _read(CACHE / "runs" / run_id / "CONTROL_RUN_CONTRACT.json")
    if private != expected:
        raise ValueError("Private selected control run contract differs")
    return receipt


def build_plan(datasets=None, resume=False):
    """Read-only, Torch-free preflight; missing/invalid parents remain explicit."""
    registry_extension.require_pre_selection(HERE, "fresh G A+M three-epoch study")
    requested = list(datasets or DATASETS)
    if requested == ["all4"]:
        requested = list(DATASETS)
    if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(DATASETS):
        raise ValueError("Use repeated supported --dataset names, or --dataset all4 alone")
    rows = _registry()
    control_sha, core_sha, files = _source_contract()
    jobs, proposed, unavailable = [], [], []
    for ds in requested:
        matches = [row for row in rows if row["dataset"] == ds and row["family"] == "G_REFIT_APLUSM"
                   and row["fold"] == "primary" and int(row["seed"]) == 92601 and "__" not in row["experiment_id"]]
        if len(matches) != 1:
            unavailable.append(dict(dataset=ds, status="MISSING_OR_AMBIGUOUS_PARENT", reason="Expected one original primary G_REFIT_APLUSM seed92601 row"))
            continue
        row, task = matches[0], None
        parent_id = row["experiment_id"]
        try:
            folder = HERE / "runs" / parent_id
            if not (folder / "CONTROL_FIT_RECEIPT.json").is_file():
                unavailable.append(dict(dataset=ds, parent_run_id=parent_id, status="PARENT_NOT_COMPLETE", reason="Actual completed CONTROL_FIT_RECEIPT absent; original work not resumed"))
                continue
            task = data.load_task(ds)
            if (task["name"] != ds or int(row["num_channels"]) != task["physical"].shape[1]
                    or int(row["length"]) != task["L"] or int(row["horizon"]) != task["H"]):
                raise ValueError("Actual task dimensions differ from registered parent")
            source = _read(folder / "CONTROL_SOURCE_CONTRACT.json")
            old = source["spec"]
            if (old.get("run_id") != parent_id or old.get("family") != "DIRECT_G" or old.get("role") != "AplusM_control"
                    or old.get("seed") != 92601 or old.get("fold") != "primary" or old.get("lr") != .001 or old.get("epochs") != [.25]):
                raise ValueError("Only actual completed original G .25-epoch LR.001 parents supported")
            scaler = _scaler(task)
            parent_receipt = _validate_complete(parent_id, old, task, _contract(old, task, control_sha, core_sha), scaler)
            run_id = parent_id + "__e3"
            spec = copy.deepcopy(old); spec.update(run_id=run_id, epochs=list(EPOCHS))
            expected = _contract(spec, task, control_sha, core_sha)
            parent_files = {str(path.relative_to(HERE).as_posix()): data.sha256_file(path)
                            for path in (folder / "CONTROL_SOURCE_CONTRACT.json", folder / "CONTROL_FIT_RECEIPT.json", folder / "control/RECEIPT.json")}
            stage = _read(folder / "control/RECEIPT.json")
            if (not stage["contract"].get("initialization_sha")
                    or stage["contract"].get("scaler_sha") != scaler):
                raise ValueError("Parent stage initialization/scaler proof differs")
            target_public, target_private = HERE / "runs" / run_id, CACHE / "runs" / run_id
            target_complete = (target_public / "CONTROL_FIT_RECEIPT.json").is_file()
            target_started = (target_public / "CONTROL_SOURCE_CONTRACT.json").exists() or target_private.exists()
            if target_complete:
                complete = _validate_complete(run_id, spec, task, expected, scaler)
                if complete["initial_model_state_sha"] != parent_receipt["initial_model_state_sha"]:
                    raise ValueError("Existing complete target initialization differs from original parent")
            elif target_started:
                if not resume:
                    raise ValueError("Partial target exists; preserved. Explicit --resume is required for this new ID only")
                target_source = _read(target_public / "CONTROL_SOURCE_CONTRACT.json")
                if (target_source.get("spec") != spec or target_source.get("files") != files
                        or any(target_source.get(key) != value for key, value in expected.items())
                        or _read(target_private / "CONTROL_RUN_CONTRACT.json") != expected):
                    raise ValueError("Explicit resume target source/config/data/model contract differs")
            provenance = dict(run_id=run_id, parent_run_id=parent_id, original_spec=old, new_spec=spec,
                changed_spec_keys=["epochs", "run_id"], unchanged_spec_sha256=_fingerprint({key:value for key,value in old.items() if key not in {"epochs","run_id"}}),
                parent_public_files=parent_files, parent_artifact_cache_relative_path=parent_receipt["artifact_cache_relative_path"],
                parent_artifact_sha256=parent_receipt["artifact_sha256"], parent_initial_model_state_sha=parent_receipt["initial_model_state_sha"],
                parent_normalization_initialization_sha=stage["contract"]["initialization_sha"],
                source_sha=control_sha, engine_source_sha=core_sha, files=files, data_sha=task["data_sha256"], scaler_sha=scaler,
                expected_run_contract=expected, fresh_from_matched_initialization=True, parent_weights_optimizer_RNG_reused=False,
                checkpoint_selection="V_select normalized channel macro MSE; selection-set evidence only",
                F_logical_streams=0, foundation_constructed=False, T_accessed=False,
                runner_source_sha256=data.sha256_file(__file__),
                comparison_limit="G trains on A+M for 3 epochs; H trains on M for 3 epochs. Exposure, target support and updates are not equal budgets.")
            existing_proof = target_public / "CONTROL_DEEPEN_PROVENANCE.json"
            if existing_proof.exists() and _read(existing_proof) != provenance:
                raise ValueError("Existing target follow-up provenance differs")
            jobs.append(dict(dataset=ds, parent_id=parent_id, run_id=run_id, task=task, spec=spec,
                             provenance=provenance, target_complete=target_complete, target_partial=target_started and not target_complete))
            proposed.append(dict(row, experiment_id=run_id, requirement="REGISTERED_G_APLUSM_EPOCH3_CONTROL",
                execution_status="PENDING_QUEUE", training_status="NOT_STARTED", validity_status="NOT_CHECKED",
                scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED", planned_epoch_endpoints="0.25|1.0|3.0",
                reason="Fresh matched-initialization G A+M 3-epoch control; original .25 retained; V CP selection; F0",
                resume_command=f"python control_deepen.py --dataset {ds} --resume"))
        except (FileNotFoundError, ValueError, KeyError) as error:
            unavailable.append(dict(dataset=ds, parent_run_id=parent_id, status="PREFLIGHT_REFUSED", reason=str(error)))
    return dict(jobs=jobs, proposed=proposed, unavailable=unavailable, datasets=requested, resume_requested=bool(resume))


def _verify_immutable(job):
    proof = job["provenance"]
    if _source_contract() != (proof["source_sha"], proof["engine_source_sha"], proof["files"]):
        raise ValueError("Frozen control/scientific source changed after preflight")
    for relative, sha in proof["parent_public_files"].items():
        if data.sha256_file(_inside(HERE, relative)) != sha:
            raise ValueError("Original parent evidence changed after preflight")
    if data.sha256_file(_inside(CACHE, proof["parent_artifact_cache_relative_path"])) != proof["parent_artifact_sha256"]:
        raise ValueError("Original parent selected checkpoint changed after preflight")


def run(datasets=None, resume=False):
    plan = build_plan(datasets, resume)
    batch = HERE / "execution_batches" / ("control_deepen_" + str(time.time_ns()) + "_" + uuid.uuid4().hex)
    batch.mkdir(parents=True, exist_ok=False)
    _new_or_same(batch / "PLAN.json", dict(run_ids=[job["run_id"] for job in plan["jobs"]],
        unavailable=plan["unavailable"], epochs=EPOCHS, resume_requested=resume,
        runner_source_sha256=data.sha256_file(__file__), reserve_GPU_seconds=9000, T_accessed=False))
    results, status = [], "FINISHED_REQUIRES_INDEPENDENT_AUDIT"
    if plan["jobs"]:
        try:
            with GPUReservation(CACHE, total_gpu_seconds=43200, reserve_seconds=9000,
                                wall_deadline=1792050350+172800, purpose="training", kind="fresh_G_APLUSM_epoch3_control") as lease:
                registry_extension.require_pre_selection(HERE, "G A+M epoch3 registration and fitting")
                registry_extension.register(HERE, plan["proposed"], "Matched fresh G A+M epoch3 controls; preregistered before Torch/model/targets; original .25 retained")
                for job in plan["jobs"]:
                    _new_or_same(HERE / "runs" / job["run_id"] / "CONTROL_DEEPEN_PROVENANCE.json", job["provenance"])
                # Deliberately only after exclusive OS ownership and immutable preregistration.
                import torch
                import fit_control
                import prediction_cache
                torch.set_num_threads(2)
                for index, job in enumerate(plan["jobs"]):
                    phase, receipt = "pre_fit", None
                    try:
                        lease.check(); registry_extension.require_pre_selection(HERE, "fresh G A+M fitting")
                        _verify_immutable(job); check_storage_budget(CACHE, pending_bytes=256*1024**2)
                        _record(job["run_id"], execution_status="RUNNING", reason="Fresh G A+M epoch .25/1/3; A60 normalization; V checkpoint selection; no foundation")
                        phase = "fit"
                        print(json.dumps(dict(event="CONTROL_DEEPEN_START",run_id=job["run_id"],epochs=EPOCHS,F_streams=0)),flush=True)
                        receipt = fit_control.fit_control(job["spec"], job["task"], CACHE, HERE, None, lease.deadline-5)
                        if receipt["status"] != "COMPLETE_VALID":
                            _record(job["run_id"],execution_status=receipt["status"],training_status=receipt["training_status"],
                                validity_status="INVALID_OR_NOT_COMPLETED",reason="New G e3 partial preserved; explicit --resume needed; original .25 untouched")
                            results.append(dict(run_id=job["run_id"],status=receipt["status"],phase=phase)); continue
                        complete = _validate_complete(job["run_id"],job["spec"],job["task"],job["provenance"]["expected_run_contract"],job["provenance"]["scaler_sha"])
                        if complete["initial_model_state_sha"] != job["provenance"]["parent_initial_model_state_sha"]:
                            raise ValueError("Fresh target initialization differs from original G parent")
                        phase = "V_prediction"; lease.check()
                        existed = (HERE / "predictions" / job["run_id"] / "V_select/PREDICTION_MANIFEST.json").exists()
                        prediction = prediction_cache.predict_parent(job["run_id"],job["task"],"V_select",CACHE,HERE,None,
                            receipt=complete,role="AplusM_control",fold="primary",deadline=lease.deadline-5)
                        meta, cost = prediction["metadata"], prediction["cost_record"]
                        reused = bool(meta.get("reused_prediction_cache"))
                        if not existed and reused:
                            raise ValueError("New G route requires actual saved-checkpoint V inference")
                        if (meta["run_id"] != job["run_id"] or meta["artifact_sha256"] != complete["artifact_sha256"]
                                or meta["data_sha256"] != job["task"]["data_sha256"]
                                or cost["actual_F_streams"] != 0 or cost["expected_F_streams"] != 0):
                            raise ValueError("Actual G V prediction artifact/data or F0 ledger differs")
                        outcome = dict(run_id=job["run_id"],status="COMPLETE_VALID",training_status=complete["training_status"],
                            selected_epoch=complete["selected_epoch"],updates=complete["updates"],exposures=complete["exposures"],
                            actual_epochs=complete["actual_epochs"],artifact_sha256=complete["artifact_sha256"],V_score=prediction["score"],
                            V_prediction_executed_now=not reused,fit_artifact_reused=bool(receipt.get("reused_complete_artifact")),
                            actual_V_F_streams=cost["actual_F_streams"],training_F_streams=complete["F_logical_streams"],
                            foundation_constructed=False,convergence_claimed=False,T_accessed=False)
                        _record(job["run_id"],execution_status="COMPLETE_VALID",training_status=complete["training_status"],
                            validity_status="PENDING_INDEPENDENT_VALIDATION",reason="Fresh G A+M epochs .25/1/3 and actual V inference; F0; V selection evidence; convergence unresolved")
                        _new_or_same(batch / (job["run_id"]+".json"),outcome); results.append(outcome)
                        print(json.dumps(dict(event="CONTROL_DEEPEN_DONE",**outcome)),flush=True)
                    except (BudgetExceeded, TimeoutError):
                        status="DEFERRED_RESOURCE"
                        _record(job["run_id"],execution_status=status,training_status="UNDERTRAINED_OR_BUDGET_LIMIT",
                            reason="G e3 deadline at "+phase+"; target preserved; explicit --resume required")
                        results.extend(dict(run_id=pending["run_id"],status=status,phase=phase if pending is job else "not_started") for pending in plan["jobs"][index:]); break
                    except Exception as error:
                        _new_or_same(CACHE / "failures" / (job["run_id"]+"_"+uuid.uuid4().hex+".json"),
                            dict(error_type=type(error).__name__,error=str(error),phase=phase,traceback=traceback.format_exc()))
                        _record(job["run_id"],execution_status="EXECUTION_FAILED",validity_status="INVALID_OR_NOT_COMPLETED",reason=type(error).__name__+"; independent datasets continue")
                        results.append(dict(run_id=job["run_id"],status="EXECUTION_FAILED",phase=phase,error_type=type(error).__name__))
        except (BudgetExceeded, ResourceBusy):
            status="DEFERRED_RESOURCE_OR_ACTIVE_OWNER"
            results.extend(dict(run_id=job["run_id"],status=status) for job in plan["jobs"])
    if not plan["jobs"]:
        status="NO_READY_PARENT"
    elif status=="FINISHED_REQUIRES_INDEPENDENT_AUDIT" and (plan["unavailable"] or any(row["status"]!="COMPLETE_VALID" for row in results)):
        status="PARTIAL_MISSING_OR_FAILED_ROUTES"
    result = dict(status=status,results=results,unavailable=plan["unavailable"],T_accessed=False,
        original_parents_replaced=False,foundation_constructed=False,convergence_claimed=False,campaign_complete=False)
    _new_or_same(batch / "EXECUTION_RECEIPT.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",action="append",choices=(*DATASETS,"all4"))
    parser.add_argument("--resume",action="store_true",help="Explicitly resume only an unchanged partial new __e3 target")
    parser.add_argument("--preflight",action="store_true",help="Read-only; no registration, Torch import, GPU or inference")
    args=parser.parse_args()
    if args.preflight:
        plan=build_plan(args.dataset,args.resume)
        result=dict(ready=[dict(dataset=job["dataset"],parent_id=job["parent_id"],run_id=job["run_id"],
            target_complete=job["target_complete"],target_partial=job["target_partial"],config_sha=job["provenance"]["expected_run_contract"]["config_sha"]) for job in plan["jobs"]],
            unavailable=plan["unavailable"],epochs=EPOCHS,read_only=True)
    else:
        result=run(args.dataset,args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
