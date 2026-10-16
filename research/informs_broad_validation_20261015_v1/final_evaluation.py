"""Explicit stop/selection lock and externally leased replay/final evaluation.

Importing this module never chooses methods, locks selection, loads a backbone,
or accesses evaluation targets. Callers supply the chosen IDs and stop decision.
"""
import csv
import gzip
import json
import os
from pathlib import Path
import time
import traceback

import psutil

import audit_predictions
import data_contract as data
import deployment
import fit_base
import route_catalog
import route_predictions
from resources import BudgetExceeded
from training import fingerprint


SOURCE_FILES = ("final_evaluation.py", "route_predictions.py", "route_catalog.py", "deployment.py",
                "data_contract.py", "evaluation.py", "training.py", "networks.py", "gpu_runtime.py",
                "ridge_baseline.py", "combination.py", "raw_calibration.py", "fit_base.py", "fit_control.py",
                "prediction_cache.py", "ordered_pilot.py", "audit_predictions.py", "audit_campaign.py")
MUTABLE_COLUMNS = {"execution_status", "training_status", "validity_status", "scientific_status",
                   "publication_status", "reason", "resume_command"}
SMALL_REV = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
OUTPUT_DEFINITION = dict(foundation_point="median_quantile_0.5", foundation_quantile=.5,
                         other_methods="deterministic_point_forecast", score_scale="A60_normalized_original_channels",
                         selection_metric="observed_channel_macro_MSE")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _sources():
    return {name: data.sha256_file(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def _rows(public):
    with (Path(public)/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["experiment_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate registry IDs")
    return rows


def _registry_sha(rows):
    return fingerprint(sorted(({k:v for k,v in row.items() if k not in MUTABLE_COLUMNS} for row in rows),
                              key=lambda row: row["experiment_id"]))


def _quiescent(rows, cache):
    if any(row["execution_status"] == "RUNNING" for row in rows):
        raise ValueError("Selection requires no RUNNING registry rows")
    active = Path(cache)/"ACTIVE_GPU_LEASE.json"
    if active.is_file():
        value = _json(active)
        try:
            live = abs(psutil.Process(int(value["pid"])).create_time()-float(value["create_time"])) < .001
        except psutil.NoSuchProcess:
            live = False
        except (psutil.AccessDenied, KeyError, TypeError, ValueError) as error:
            raise ValueError("Cannot verify the recorded GPU owner is quiescent") from error
        if live and value.get("purpose") == "training":
            raise ValueError("Selection requires the live GPU training lease to end")


def _ids(run_ids):
    values = list(map(str, run_ids))
    if not values or len(set(values)) != len(values) or any(Path(value).name != value or value in {".", ".."} for value in values):
        raise ValueError("Explicit unique safe selected run IDs required")
    return values


def _V_manifest(public, run_id):
    candidates = (Path(public)/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json",
                  Path(public)/"ordered_predictions"/run_id/"PREDICTION_MANIFEST.json",
                  Path(public)/"route_predictions"/run_id/"V_select/PREDICTION_MANIFEST.json")
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise ValueError("Selected route has no actual V prediction manifest")
    value = _json(path)
    if value.get("run_id") != run_id or value.get("split") != "V_select":
        raise ValueError("Selected manifest must contain this route's V predictions")
    return path, value


def _audit_binding(public, decision, manifests):
    if decision.get("audit_report_relative_path"):
        path = route_catalog._inside(public, decision["audit_report_relative_path"])
    else:
        paths = sorted((Path(public)/"audits").glob("*/AUDIT_REPORT.json"))
        if not paths:
            raise ValueError("Selection requires an actual independent available-prediction audit")
        path = paths[-1]
    report = _json(path)
    if report.get("errors") != [] or not str(report.get("status", "")).startswith("PASS_AVAILABLE_PREDICTIONS"):
        raise ValueError("Selection requires a successful independent available-prediction audit")
    if report.get("source_sha256") != data.sha256_file(Path(__file__).with_name("audit_campaign.py")):
        raise ValueError("Independent audit source differs from the pinned audit mechanism")
    records_path = path.with_name("PREDICTION_AUDITS.json")
    records = _json(records_path)
    for run_id, (manifest_path, manifest) in manifests.items():
        relative = manifest_path.relative_to(public).as_posix()
        matches = [r for r in records if r.get("run_id") == run_id and r.get("split") == "V_select"
                   and r.get("manifest_public_relative_path") == relative]
        if len(matches) != 1 or matches[0].get("status") != "PASS" or matches[0].get("prediction_rescoring") is not True or matches[0].get("manifest_sha256") != data.sha256_file(manifest_path):
            raise ValueError("Selected V manifest is missing an exact successful independent audit record")
    return dict(report_public_relative_path=path.relative_to(public).as_posix(), report_sha256=data.sha256_file(path),
                records_public_relative_path=records_path.relative_to(public).as_posix(), records_sha256=data.sha256_file(records_path),
                status=report["status"])


def _verify_V_now(manifest_path, manifest, rows, public, cache):
    reference = _json(manifest_path.with_name("REFERENCE_CONTRACT.json"))
    statistics = route_catalog._inside(public,manifest["public_stats_relative_path"])
    opener = gzip.open if statistics.suffix == ".gz" else open
    run_id = manifest["run_id"]
    projection = [dict(run_id=row["experiment_id"], execution_status="COMPLETE_VALID" if row["experiment_id"] == run_id else row["execution_status"],
                       reason=row.get("reason") or "Outside this selected-route V artifact check") for row in rows]
    with opener(statistics,"rt",encoding="utf-8-sig",newline="") as stream:
        return audit_predictions.audit_prediction_bundle(manifest,reference,manifest["score"],csv.DictReader(stream),
                        projection,[row["experiment_id"] for row in rows],[manifest["cost_record"]],cache)


def _verify_public_bindings(spec, public):
    for key, value in spec.items():
        if key.endswith("_public_relative_path"):
            sha_key = key[:-len("_public_relative_path")] + "_sha256"
            if sha_key in spec:
                path = route_catalog._inside(public, value)
                if data.sha256_file(path) != spec[sha_key]:
                    raise ValueError("Portable route public receipt/manifest changed")
        if isinstance(value, dict):
            _verify_public_bindings(value, public)


def _verify_binding(binding, public, cache):
    for item in binding["artifact_bindings"].values():
        route_catalog._bound(cache, item["cache_relative_path"], item["sha256"])
    for prefix in ("V_manifest", "V_reference"):
        path = route_catalog._inside(public, binding[prefix+"_public_relative_path"])
        if data.sha256_file(path) != binding[prefix+"_sha256"]:
            raise ValueError("Locked V manifest/reference changed")
    route_catalog._bound(cache, binding["V_archive_cache_relative_path"], binding["V_archive_sha256"])
    _verify_public_bindings(binding["route_spec"], public)


def _validated_lock(cache, public):
    path = Path(public)/"SELECTION_LOCK.json"
    if not path.is_file():
        raise ValueError("T_final is locked: explicit selection lock absent")
    lock = _json(path)
    if lock.get("locked") is not True or lock.get("source_bindings") != _sources():
        raise ValueError("Locked evaluation sources changed or selection is absent")
    expected = fingerprint({k:v for k,v in lock.items() if k != "selection_sha256"})
    if lock.get("selection_sha256") != expected:
        raise ValueError("Immutable selection lock content changed")
    ids = _ids(lock["allowed_run_ids"])
    rows = _rows(public)
    _quiescent(rows, cache)
    if _registry_sha(rows) != lock["registry_scientific_sha"]:
        raise ValueError("Registry scientific identity changed after selection")
    for prefix in ("report", "records"):
        item = lock["independent_audit"]
        path = route_catalog._inside(public, item[prefix+"_public_relative_path"])
        if data.sha256_file(path) != item[prefix+"_sha256"]:
            raise ValueError("Locked independent audit changed")
    for run_id in ids:
        binding = lock["route_bindings"][run_id]
        if lock["allowed_route_sha"].get(run_id) != fingerprint(binding["route_spec"]):
            raise ValueError("Selection route hash changed")
        _verify_binding(binding, public, cache)
    return lock


def lock_selection(run_ids, decision, public, cache):
    """Seal only caller-selected methods after an explicit stop decision; no T labels."""
    ids, public, cache = _ids(run_ids), Path(public).resolve(), Path(cache).resolve()
    if decision.get("stop_training") is not True or not str(decision.get("reason", "")).strip():
        raise ValueError("Explicit stop_training=True and a nonempty reason are required")
    decision = dict(decision)
    if decision.get("output_definition", OUTPUT_DEFINITION) != OUTPUT_DEFINITION:
        raise ValueError("The registered foundation median-0.5 point definition cannot change at final selection")
    decision["output_definition"] = dict(OUTPUT_DEFINITION)
    rows = _rows(public)
    _quiescent(rows, cache)
    if (public/"SELECTION_LOCK.json").exists():
        previous = _validated_lock(cache, public)
        if previous["allowed_run_ids"] != ids or previous["decision"] != decision:
            raise ValueError("Existing selection cannot be replaced")
        return previous
    registry = {row["experiment_id"]:row for row in rows}
    if any(run_id not in registry for run_id in ids):
        raise ValueError("Selected route is not registered")
    manifests = {run_id:_V_manifest(public, run_id) for run_id in ids}
    audit = _audit_binding(public, decision, manifests)
    bindings, tasks, artifact_shas = {}, {}, {}
    sources = _sources()
    for run_id in ids:
        row = registry[run_id]
        if row["execution_status"] != "COMPLETE_VALID" and not row["execution_status"].startswith("PARTIAL"):
            raise ValueError("Only an actually executed complete/partial route can be selected")
        ds = row["dataset"]
        if ds not in tasks: tasks[ds] = data.load_task(ds)
        task = tasks[ds]
        spec, size = route_catalog.build_route(run_id, task, cache, public)
        path, manifest = manifests[run_id]
        _verify_V_now(path,manifest,rows,public,cache)
        bounds = data.split_ranges(task, spec.get("fold", "primary"))
        mean, std, _ = data.fit_statistics(task, bounds["A_stop"][1])
        if manifest["data_sha"] != task["data_sha256"] or manifest["scaler_sha"] != fit_base.scaler_fingerprint(mean,std):
            raise ValueError("Selected V data/scaler differs from the actual legal A normalization")
        reference = path.with_name("REFERENCE_CONTRACT.json")
        meta = manifest.get("metadata", {})
        binding = dict(route_spec=spec, model_size=size, artifact_bindings=route_catalog.artifact_bindings(spec),
                       V_manifest_public_relative_path=path.relative_to(public).as_posix(), V_manifest_sha256=data.sha256_file(path),
                       V_reference_public_relative_path=reference.relative_to(public).as_posix(), V_reference_sha256=data.sha256_file(reference),
                       V_archive_cache_relative_path=manifest["cache_relative_path"], V_archive_sha256=manifest["archive_sha256"],
                       data_sha=manifest["data_sha"], scaler_sha=manifest["scaler_sha"], V_source_sha=manifest["source_sha"],
                       registry_execution_status=row["execution_status"], training_status=row.get("training_status", ""))
        _verify_binding(binding, public, cache)
        bindings[run_id] = binding
        artifact_shas[run_id] = meta.get("artifact_sha256") or meta.get("parent_artifact_sha256") or spec.get("artifact_sha256") or manifest["model_sha"]
    incomplete = [{key:row.get(key, "") for key in ("experiment_id", "execution_status", "training_status", "reason")}
                  for row in rows if row.get("requirement", "").startswith("MANDATORY") and row["execution_status"] != "COMPLETE_VALID"]
    if _sources() != sources:
        raise ValueError("Evaluation source changed while selection was being prepared")
    lock = dict(version=1, locked=True, locked_unix=time.time(), decision=decision,
                allowed_run_ids=ids, allowed_route_sha={run:fingerprint(value["route_spec"]) for run,value in bindings.items()},
                allowed_artifact_sha=artifact_shas, route_bindings=bindings, source_bindings=sources,
                independent_audit=audit, registry_scientific_sha=_registry_sha(rows),
                mandatory_coverage_complete=not incomplete, mandatory_incomplete=incomplete,
                status="LOCKED_PARTIAL_CAMPAIGN" if incomplete else "LOCKED_SELECTION",
                T_targets_read_during_selection=False, automatic_method_choice=False,
                scope="Explicit chosen routes; saved V audit is not checkpoint replay or convergence proof")
    lock["selection_sha256"] = fingerprint(lock)
    _new(public/"SELECTION_LOCK.json",lock)
    return lock


def _owner(lease, deadline):
    if lease is None or getattr(lease,"purpose",None) != "audit" or not getattr(lease,"_entered",False) or getattr(getattr(lease,"lock",None),"file",None) is None:
        raise ValueError("An externally acquired active audit owner lease is required; no nested reservation")
    lease.check()
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Final evaluation deadline reached")


def _model_path(size, cache, public):
    if size is None: return None
    if size == "small":
        root = Path(os.environ.get("INFORMS_ASSET_ROOT",str(data.ROOT)))
        path = root/".cache/huggingface/models--amazon--chronos-bolt-small/snapshots"/SMALL_REV
    elif size in {"mini","tiny"}:
        entry = _json(Path(public)/"MODEL_ASSETS.json")["models"][size]
        path = route_catalog._inside(cache,entry["cache_relative_path"])
        for value in entry["files"].values():
            route_catalog._bound(cache,value["cache_relative_path"],value["sha256"])
    else:
        raise ValueError("Only pinned official small/mini/tiny foundations are supported")
    if not path.is_dir(): raise FileNotFoundError("Pinned local foundation snapshot is absent; no download is performed")
    return path


def _device(spec):
    return "cpu" if spec["family"] in {"CURRENT","SEASONAL","RAW_RIDGE"} else "cuda"


def _execute(run_ids, cache, public, lease, deadline, final):
    initial_budget_error = None
    try:
        _owner(lease,deadline)
    except (BudgetExceeded,TimeoutError) as error:
        initial_budget_error = error
    cache, public = Path(cache).resolve(), Path(public).resolve()
    lock = _validated_lock(cache,public) if final else None
    ids = _ids(lock["allowed_run_ids"] if final else run_ids)
    rows = {row["experiment_id"]:row for row in _rows(public)}
    batch = ("T_final_" if final else "V_replay_")+str(time.time_ns())
    records, tasks, deferred = [], {}, []
    for index, run_id in enumerate(ids):
        try:
            if initial_budget_error is not None:
                raise initial_budget_error
            _owner(lease,deadline)
            if final: _validated_lock(cache,public)
            row = rows[run_id]
            ds = row["dataset"]
            if ds not in tasks: tasks[ds] = data.load_task(ds)
            task = tasks[ds]
            spec, size = route_catalog.build_route(run_id,task,cache,public)
            if final:
                binding = lock["route_bindings"][run_id]
                if fingerprint(spec) != lock["allowed_route_sha"][run_id] or task["data_sha256"] != binding["data_sha"]:
                    raise ValueError("Actual route/data differs from locked selection")
                bounds = data.split_ranges(task,spec.get("fold","primary"))
                mean,std,_ = data.fit_statistics(task,bounds["A_stop"][1])
                if fit_base.scaler_fingerprint(mean,std) != binding["scaler_sha"]:
                    raise ValueError("Actual A scaler differs from locked selection")
            model_path = _model_path(size,cache,public)
            options = dict(model_path=model_path,device=_device(spec),deadline=deadline,lease=lease)
            if final:
                result = route_predictions.predict_route(run_id,task,"T_final",cache,public,**options)
                manifest = public/"route_predictions"/run_id/"T_final/PREDICTION_MANIFEST.json"
                records.append(dict(run_id=run_id,status="PASS",executed_now=bool(result["executed_now"]),
                                    manifest_public_relative_path=manifest.relative_to(public).as_posix(),
                                    manifest_sha256=data.sha256_file(manifest), actual_F_streams=result["cost_record"]["actual_F_streams"]))
            else:
                result = route_predictions.replay_route(run_id,task,cache,public,**options)
                path = public/"route_replays"/run_id/"REPLAY_RECEIPT.json"
                records.append(dict(run_id=run_id,status=result["status"],executed_now=bool(result["executed_now"]),
                                    receipt_public_relative_path=path.relative_to(public).as_posix(),receipt_sha256=data.sha256_file(path),
                                    actual_F_streams=result["actual_F_streams"]))
        except (BudgetExceeded,TimeoutError) as error:
            deferred = ids[index:]
            records.append(dict(run_id=run_id,status="DEFERRED_RESOURCE",reason=type(error).__name__))
            break
        except Exception as error:
            error_path = cache/"final_evaluation"/batch/(run_id+".json")
            _new(error_path,dict(run_id=run_id,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
            records.append(dict(run_id=run_id,status="EXECUTION_FAILED",error_type=type(error).__name__,
                                private_error_cache_relative_path=error_path.relative_to(cache).as_posix()))
    success = sum(row["status"] == "PASS" for row in records)
    failure = sum(row["status"] not in {"PASS","DEFERRED_RESOURCE"} for row in records)
    result = dict(status="DEFERRED_RESOURCE" if deferred else ("PARTIAL_ROUTE_FAILURES" if failure else "COMPLETE_SELECTED_ROUTES"),
                  phase="T_final" if final else "V_checkpoint_replay",run_ids=ids,records=records,
                  succeeded=success,failed=failure,deferred_run_ids=deferred,
                  executed_now=sum(bool(row.get("executed_now")) for row in records),
                  reused_complete_artifacts=sum(row.get("executed_now") is False for row in records),
                  campaign_completion_claimed=False,registry_mutated=False,created_unix=time.time())
    if final:
        result.update(selection_sha256=lock["selection_sha256"],mandatory_coverage_complete=lock["mandatory_coverage_complete"])
    _new(public/"execution_batches"/(batch+".json"),result)
    return result


def execute_replays(run_ids, cache, public, lease, deadline):
    return _execute(run_ids,cache,public,lease,deadline,False)


def execute_T(cache, public, lease, deadline):
    return _execute(None,cache,public,lease,deadline,True)
