"""Fresh-process head-component costs and public-only deterministic replay."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


NAME = "raw_context_multiscale_head_cost_20261019_v1"
CAMPAIGN = "raw_context_multiscale_20261018_v1"
REPO = Path(__file__).resolve().parents[2]
CSV_NAME = "HEAD_COMPONENT_COST.csv"
TRIALS_NAME = "HEAD_COMPONENT_COST_TRIALS.csv"
VALIDATION_NAME = "HEAD_COMPONENT_COST_VALIDATION.json"
INPUTS_NAME = "HEAD_COMPONENT_COST_INPUTS.json"
SCOPE = "GPU-resident FP32 normalized M_cal x/G/B -> entire ConditionalHead.forward including features -> GPU output plus synchronization"
EXPECTED = {(dataset, mode, seed, batch) for dataset in ("jena", "pems32")
            for mode in ("conditional", "past_control") for seed in (92601, 92602) for batch in (1, 8)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("Preserve differing existing output: "+path.name)
        return
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def context(asset_root, repo=REPO):
    repo = Path(repo).resolve()
    return dict(repo=repo, report=repo/"reports"/CAMPAIGN, cache=repo/".cache"/CAMPAIGN,
                asset=Path(asset_root).resolve())


def binding(path, ctx):
    path = Path(path).resolve()
    for key in ("report", "cache", "repo", "asset"):
        root = Path(ctx[key]).resolve()
        if path.is_relative_to(root):
            return dict(reference=key+"://"+path.relative_to(root).as_posix(), sha256=sha(path))
    raise ValueError("Unregistered artifact root")


def reference_path(reference, ctx):
    key, relative = reference.split("://", 1)
    if key not in ("report", "cache", "repo", "asset"):
        raise ValueError("Unregistered artifact reference")
    root = Path(ctx[key]).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact escapes registered root")
    return path


def resolve(value, ctx):
    path = reference_path(value["reference"], ctx)
    if sha(path) != value["sha256"]:
        raise ValueError("Artifact SHA256 changed: "+value["reference"])
    return path


def _check_sources(ctx, sources):
    for relative, expected in sources.items():
        path = (ctx["repo"]/relative).resolve()
        if not path.is_relative_to(ctx["repo"]) or sha(path) != expected:
            raise ValueError("Frozen source changed: "+relative)


def _journal_path(ctx):
    return ctx["cache"]/"head_component_costs_001"/"EVENTS.jsonl"


def _events(ctx):
    path = _journal_path(ctx)
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()] if path.exists() else []


def owned_wall(events, now):
    started, closed = {}, {}
    for event in events:
        target = started if event["status"] == "STARTED" else closed if event["status"] in ("COMPLETE", "FAILED") else None
        if target is None or event["session_id"] in target:
            raise ValueError("Unknown or duplicate owned-worker event")
        target[event["session_id"]] = event
    if set(closed)-set(started):
        raise ValueError("Closed owned worker lacks its start")
    total = 0.
    for identity, event in started.items():
        if event["time_unix"] > now:
            raise ValueError("Owned-worker start is in the future")
        if identity in closed:
            end = closed[identity]
            elapsed = float(end["wall_seconds"])
            if (elapsed < 0 or not math.isfinite(elapsed) or end["time_unix"] < event["time_unix"]
                    or end["time_unix"] > now or not math.isclose(elapsed, end["time_unix"]-event["time_unix"], rel_tol=0., abs_tol=1e-6)):
                raise ValueError("Invalid owned-worker elapsed wall")
            total += elapsed
        else:
            total += now-event["time_unix"]
    return dict(seconds=total, open_sessions=len(set(started)-set(closed)), closed_sessions=len(closed))


def _completion_references(ctx, scope_path, scope, prerequisites):
    if "replaces_scope_binding" not in scope:
        if (Path(scope_path).name == "HEAD_COMPONENT_COST_SCOPE_RECOVERY_002.json"
                or scope["required_terminal_references"].get("core") != "cache://TERMINAL_CORE_002.json"
                or scope["required_terminal_references"].get("chain") != "cache://TERMINAL_SUPPLEMENT_CHAIN_001.json"
                or scope["required_public_references"].get("primary_validation") != "report://VALIDATION_REPORT.json"):
            raise ValueError("Replacement terminal/PASS paths require explicit registered completion recovery")
        return
    original_path = resolve(scope["replaces_scope_binding"], ctx)
    if (original_path != reference_path("report://HEAD_COMPONENT_COST_SCOPE_001.json", ctx)
            or Path(scope_path).resolve() != reference_path("report://HEAD_COMPONENT_COST_SCOPE_RECOVERY_002.json", ctx)):
        raise ValueError("Recovery must explicitly supersede the exact original head scope")
    original = read(original_path)
    unchanged = ("status", "source_files", "primary_plan_binding", "model_seal_binding",
                 "original_execution_start_binding", "original_deadline_unix", "limits",
                 "GPU_base_ledger_reference", "component_protocol", "scientific_contract_changed", "selection_changes")
    if any(scope[key] != original[key] for key in unchanged):
        raise ValueError("Head recovery cannot change scientific/protocol/source/original budget contracts")
    public = dict(original["required_public_references"])
    public["primary_validation"] = "report://RECOVERY_VALIDATION_REPORT_002.json"
    terminals = dict(core="cache://TERMINAL_COMPLETION_RECOVERY_002.json",
                     chain="cache://TERMINAL_SUPPLEMENT_CHAIN_RECOVERY_002.json")
    if scope["required_public_references"] != public or scope["required_terminal_references"] != terminals:
        raise ValueError("Only the explicit recovered primary PASS and terminal lineage may replace prerequisites")
    from research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 import validate_completion
    completion = validate_completion(ctx)
    if (Path(completion["terminal_path"]).resolve() != reference_path(terminals["core"], ctx)
            or Path(completion["validation_path"]).resolve() != reference_path(public["primary_validation"], ctx)):
        raise ValueError("Completed recovery returned paths outside the registered replacement lineage")
    prerequisites["replaced_head_scope"] = binding(original_path, ctx)


def gates(ctx, scope_path, preflight_path):
    scope, preflight = read(scope_path), read(preflight_path)
    protocol = scope["component_protocol"]
    if (scope.get("status") != "REGISTERED_BEFORE_COMPONENT_MEASUREMENT"
            or scope.get("scientific_contract_changed") is not False or scope.get("selection_changes") is not False
            or len(scope["source_files"]) != 20 or protocol.get("heads") != 8 or protocol.get("facets") != 16
            or protocol.get("batches") != [1, 8] or protocol.get("warmup") != 10 or protocol.get("timed") != 20
            or protocol.get("device") != "cuda" or protocol.get("dtype") != "FP32" or protocol.get("inference_mode") is not True
            or protocol.get("origins") != "first8_dense_Mcal" or protocol.get("split") != "M_cal"
            or protocol.get("atol") != 1e-5 or protocol.get("rtol") != 1e-4 or protocol.get("Fcalls") != 0
            or protocol.get("new_training_updates") != 0
            or any(protocol.get(key) is not False for key in ("new_model_selection", "new_V_forecasts", "T_access"))):
        raise ValueError("Exact registered head component protocol required")
    if (preflight.get("status") != "PASS_SYNTHETIC_AND_FROZEN_CODE" or preflight.get("scope_binding") != binding(scope_path, ctx)
            or preflight.get("actual_evaluation_performed") is not False
            or set(preflight["source_files"]) != {"research/"+NAME+"/"+name for name in ("__init__.py", "run.py")}
            or set(preflight["test_source_files"]) != {"tests/test_raw_context_multiscale_head_cost.py"}):
        raise ValueError("Scope-bound frozen head-cost code preflight required")
    _check_sources(ctx, scope["source_files"])
    _check_sources(ctx, preflight["source_files"])
    _check_sources(ctx, preflight["test_source_files"])
    start_path = resolve(scope["original_execution_start_binding"], ctx)
    start = read(start_path)
    if (start["started_unix"]+start["limits"]["wall_seconds"] != scope["original_deadline_unix"]
            or scope["limits"] != dict(wall_seconds=21600, GPU_reservation_seconds=14400)):
        raise ValueError("Original wall/GPU budget must remain unchanged")
    if time.time() >= scope["original_deadline_unix"]:
        raise TimeoutError("Original core deadline exhausted")
    prerequisites = dict(scope=binding(scope_path, ctx), preflight=binding(preflight_path, ctx),
                         primary_plan=scope["primary_plan_binding"], model_seal=scope["model_seal_binding"])
    resolve(scope["primary_plan_binding"], ctx)
    resolve(scope["model_seal_binding"], ctx)
    _completion_references(ctx, scope_path, scope, prerequisites)
    for name, reference in scope["required_terminal_references"].items():
        path = reference_path(reference, ctx)
        value = read(path)
        expected_status = "SUPPLEMENT_STAGES_COMPLETE_PENDING_ROOT_REVIEW_AND_PUBLICATION" if name == "chain" else "COMPLETE"
        if value.get("status") != expected_status or value.get("exit_code") != 0:
            raise ValueError("Core and supplement chain must be COMPLETE exit0 before private/GPU access")
        prerequisites["terminal_"+name] = binding(path, ctx)
    public = {}
    for name, reference in scope["required_public_references"].items():
        path = reference_path(reference, ctx)
        prerequisites[name] = binding(path, ctx)
        if path.suffix == ".json":
            public[name] = read(path)
    primary, independent, mcal = (public[name] for name in ("primary_validation", "independent_validation", "Mcal_validation"))
    if (primary.get("status") != "PASS" or primary.get("models") != 56 or primary.get("cost_facets") != 120
            or independent.get("status") != "PASS" or independent.get("coverage") !=
                dict(V_models=56, whole_path_cost_facets=120, CP0_models=10, expected_CP0_models=10)
            or mcal.get("status") != "PASS" or mcal.get("models") != 40 or mcal.get("head_models") != 8):
        raise ValueError("Complete primary56/120, independent+CP0 and Mcal40 PASS required")
    if primary["manifest_bindings"]["registered_plan"] != scope["primary_plan_binding"]:
        raise ValueError("Primary audit binds a different original plan")
    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json"):
        resolve(primary["manifest_bindings"][name], ctx)
    if resolve(mcal["CSV_binding"], ctx) != reference_path(scope["required_public_references"]["Mcal_diagnostics"], ctx):
        raise ValueError("Mcal PASS must bind its exact public diagnostics CSV")
    events = ctx["cache"]/"EVENTS.jsonl"
    event_binding = binding(events, ctx)
    if public["final_context"]["resource_events_binding"] != event_binding:
        raise ValueError("Primary EVENTS must match the frozen final report snapshot")
    ledger_path = reference_path(scope["GPU_base_ledger_reference"], ctx)
    ledger = read(ledger_path)
    if (ledger["immutable_core_deadline_unix"] != scope["original_deadline_unix"]
            or ledger["core_execution_started_unix"] != start["started_unix"]
            or ledger.get("unmatched_started_GPU_sessions") != 0):
        raise ValueError("GPU base ledger must preserve original start/deadline with closed sessions")
    base = float(ledger["GPU_reserved_process_wall_upper_bound_seconds"])
    owned = owned_wall(_events(ctx), time.time())
    if not math.isfinite(base) or base < 0 or base+owned["seconds"] >= 14400:
        raise TimeoutError("Original GPU process-wall reservation exhausted")
    prerequisites.update(GPU_base_ledger=binding(ledger_path, ctx), primary_EVENTS=event_binding)
    return dict(scope=scope, preflight=preflight, prerequisite_bindings=prerequisites,
                GPU_base_seconds=base, original_deadline_unix=scope["original_deadline_unix"])


def _percentile(values, fraction):
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def latency_statistics(trials, batch_size):
    if batch_size not in (1, 8):
        raise ValueError("Registered B1/B8 required")
    if (len(trials) != 30 or {(row["phase"], row["trial"]) for row in trials} !=
            {("warmup", index) for index in range(10)} | {("timed", index) for index in range(20)}):
        raise ValueError("Exactly ten warmup and twenty unique timed trials required")
    if any(not math.isfinite(float(row["whole_seconds"])) or float(row["whole_seconds"]) <= 0 for row in trials):
        raise ValueError("Finite positive component trial times required")
    values = sorted(1000*float(row["whole_seconds"]) for row in trials if row["phase"] == "timed")
    median, p95 = _percentile(values, .5), _percentile(values, .95)
    return dict(p50_batch_ms=median, IQR_batch_ms=_percentile(values, .75)-_percentile(values, .25),
                p95_batch_ms=p95, p50_ms_per_origin=median/batch_size, p95_ms_per_origin=p95/batch_size)


def audit_receipts(receipts):
    if len(receipts) != 16:
        raise ValueError("All sixteen head component facets required")
    coverage, processes, inputs, dataset_origins = set(), set(), {}, {}
    rows, trials = [], []
    for receipt in receipts:
        row = receipt["row"]
        key = (row["dataset"], row["mode"], row["seed"], row["batch_size"])
        if key not in EXPECTED or key in coverage or receipt.get("status") != "COMPLETE":
            raise ValueError("Duplicate, missing or unregistered component facet")
        coverage.add(key)
        process = (row["process_pid"], row["process_create_time"])
        if process in processes or process[0] <= 0 or not math.isfinite(float(process[1])):
            raise ValueError("Each facet requires a fresh PID+creation-time identity")
        processes.add(process)
        validate_lineage(receipt)
        if (receipt["parity"].get("status") != "PASS" or receipt["parity"].get("atol") != 1e-5
                or receipt["parity"].get("rtol") != 1e-4 or row.get("F_calls") != 0
                or row.get("forward_scope") != SCOPE or row.get("device") != "cuda"):
            raise ValueError("Fixed reference parity/component-only/F0 contract required")
        for name in ("request_binding", "input_binding", "head_binding"):
            if not receipt.get(name, {}).get("sha256") or not receipt[name].get("reference"):
                raise ValueError("Bound request/input/final-head provenance required")
        head_key = key[:3]
        identity = (row["run_id"], receipt["input_binding"], receipt["head_binding"])
        if head_key in inputs and inputs[head_key] != identity:
            raise ValueError("B1/B8 must use the exact same head state and fixed input bundle")
        inputs[head_key] = identity
        first8 = row["fixed_first8_origins"]
        if len(first8) != 8 or first8 != list(range(first8[0], first8[0]+8)):
            raise ValueError("Exact eight dense Mcal origins required")
        if row["dataset"] in dataset_origins and dataset_origins[row["dataset"]] != first8:
            raise ValueError("Every head and B1/B8 must use identical first8 origins per dataset")
        dataset_origins[row["dataset"]] = first8
        for trial in receipt["trials"]:
            origins = row["fixed_first8_origins"] if row["batch_size"] == 8 else [row["fixed_first8_origins"][trial["trial"] % 8]]
            if trial["origins"] != origins or trial.get("F_calls") != 0:
                raise ValueError("Component trial origin trace or F0 identity differs")
        stats = latency_statistics(receipt["trials"], row["batch_size"])
        for key, value in stats.items():
            if not math.isclose(row[key], value, rel_tol=0., abs_tol=1e-12):
                raise ValueError("Component trial quantile differs: "+key)
        for name in ("peak_allocated_MiB", "peak_reserved_MiB"):
            expected = max(float(trial[name]) for trial in receipt["trials"] if trial["phase"] == "timed")
            if not math.isfinite(expected) or expected < 0 or row[name] != expected:
                raise ValueError("Component allocator peak differs")
        if row.get("process_GPU_sampled_peak_MiB") is not None or row.get("process_GPU_sampling_status") != "NOT_MEASURED":
            raise ValueError("Unmeasured per-PID memory must remain NA, not device-wide memory")
        rows.append(row)
        trials.extend(dict(trial, component_cost_id=row["component_cost_id"], run_id=row["run_id"],
                           dataset=row["dataset"], mode=row["mode"], seed=row["seed"], batch_size=row["batch_size"]) for trial in receipt["trials"])
    if coverage != EXPECTED:
        raise ValueError("Exact 8 heads times B1/B8 coverage required")
    return sorted(rows, key=lambda row:(row["dataset"], row["mode"], row["seed"], row["batch_size"])), sorted(trials,
        key=lambda row:(row["dataset"], row["mode"], row["seed"], row["batch_size"], 0 if row["phase"] == "warmup" else 1, row["trial"]))


def validate_lineage(receipt):
    lineage, launch = receipt["process_lineage"], receipt["launch_proof"]
    row, worker = receipt["row"], lineage["worker"]
    if worker != dict(pid=row["process_pid"], create_time=row["process_create_time"]):
        raise ValueError("Actual benchmark PID/creation must match its process lineage")
    if (launch["collector"] not in lineage["ancestors"] or launch["collector"] == worker
            or launch["collector"] != receipt["request_collector_identity"]
            or (launch["launcher"] != worker and launch["launcher"] not in lineage["ancestors"])):
        raise ValueError("Actual worker must descend from its owned launcher and collector; Windows redirectors are allowed")


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key:json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                         if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def render_public(receipts, resources, output_dir):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Public component output must be new or empty")
    rows, trials = audit_receipts(receipts)
    audit_resources(resources)
    closed = [event for event in resources["owned_events"] if event["status"] == "COMPLETE"]
    for receipt in receipts:
        if sum(event.get("request_binding") == receipt["request_binding"] for event in closed) != 1:
            raise ValueError("Each measured facet must match exactly one completed owned-worker reservation")
    output.mkdir(parents=True, exist_ok=True)
    _csv(output/CSV_NAME, rows)
    _csv(output/TRIALS_NAME, trials)
    validation = dict(status="PASS", component_facets=16, heads=8, trials=480, warmup_trials=160, timed_trials=320,
        scope=SCOPE, whole_path_replacement=False, parent_latency_subtracted=False, add_to_parent_latency_valid=False,
        independent_trial_quantiles="PASS", fixed_reference_parity="PASS", fresh_PID_creation_per_facet="PASS",
        F_calls=0, new_training_updates=0, new_V_forecasts=False, T_access=False, resources=resources,
        source_bindings=resources.get("prerequisite_bindings", {}),
        table_bindings={CSV_NAME:dict(reference="report://"+CSV_NAME, sha256=sha(output/CSV_NAME)),
                        TRIALS_NAME:dict(reference="report://"+TRIALS_NAME, sha256=sha(output/TRIALS_NAME))})
    write_new(output/VALIDATION_NAME, validation)
    write_new(output/INPUTS_NAME, dict(status="COMPLETE_PUBLIC_INPUTS", receipts=receipts, resources=resources,
        tables={CSV_NAME:sha(output/CSV_NAME), TRIALS_NAME:sha(output/TRIALS_NAME)}, validation_sha256=sha(output/VALIDATION_NAME)))
    return validation


def audit_resources(resources):
    snapshot = float(resources["finished_unix"])
    owned = owned_wall(resources["owned_events"], snapshot)
    base = float(resources["GPU_base_seconds"])
    if owned["open_sessions"] or owned != resources["owned_sessions"]:
        raise ValueError("Component resource snapshot must close and count each owned worker exactly once")
    if (not math.isfinite(base) or base < 0
            or not math.isclose(owned["seconds"], resources["component_owned_worker_wall_seconds"], rel_tol=0., abs_tol=1e-9)
            or not math.isclose(base+owned["seconds"], resources["GPU_total_process_wall_upper_bound_seconds"], rel_tol=0., abs_tol=1e-9)):
        raise ValueError("Original GPU base plus owned component wall must match the separate ledger")
    if resources["GPU_limit_seconds"] != 14400 or base+owned["seconds"] > 14400:
        raise ValueError("Original GPU process-wall budget exceeded or changed")
    if snapshot > resources["original_deadline_unix"]:
        raise ValueError("Original wall deadline exceeded before component resource snapshot")
    if resources["primary_EVENTS_before"] != resources["primary_EVENTS_after"]:
        raise ValueError("Frozen primary EVENTS must remain unchanged")
    return owned


def replay(source_dir, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Public replay requires a separate output directory")
    inputs = read(source/INPUTS_NAME)
    if inputs.get("status") != "COMPLETE_PUBLIC_INPUTS":
        raise ValueError("Complete public component inputs required")
    for name, expected in inputs["tables"].items():
        if sha(source/name) != expected:
            raise ValueError("Public component table changed")
    if sha(source/VALIDATION_NAME) != inputs["validation_sha256"]:
        raise ValueError("Public component validation changed")
    return render_public(inputs["receipts"], inputs["resources"], output)


def prepare_inputs(ctx, proof):
    """CPU preparation after all prerequisite gates; never load a foundation."""
    import numpy as np
    import torch
    from research.raw_context_multiscale_20261018_v1 import common, report
    from research.raw_context_multiscale_mcal_diagnostics_20261019_v1 import run as mcal
    active = common.context(ctx["asset"], repo=ctx["repo"], device="cpu")
    tasks = {dataset:common.load_task(active, dataset) for dataset in ("jena", "pems32")}
    models = read(ctx["cache"]/"CALIBRATION_MODELS.json")
    heads = [arm for arm in models if arm["mode"] in ("conditional", "past_control")]
    if len(heads) != 8 or {(arm["dataset"], arm["mode"], arm["seed"], batch) for arm in heads for batch in (1, 8)} != EXPECTED:
        raise ValueError("Exact eight already frozen calibration heads required")
    validation = read(resolve(proof["prerequisite_bindings"]["Mcal_validation"], ctx))
    references = {item["run_id"]:item for item in validation["evidence"]}
    parents, pasts, prepared = {}, {}, []
    for arm in sorted(heads, key=lambda arm:(arm["dataset"], arm["mode"], arm["seed"])):
        if time.time() >= proof["original_deadline_unix"]:
            raise TimeoutError("Original deadline reached during component CPU preparation")
        task = tasks[arm["dataset"]]
        g = mcal.load_parent(active, task, arm["parent_g_prediction_binding"], arm["parent_g"]["route_spec"], parents, proof["original_deadline_unix"])
        control = arm["mode"] == "past_control"
        b = None if control else mcal.load_parent(active, task, arm["parent_b_prediction_binding"], arm["parent_b"]["route_spec"], parents, proof["original_deadline_unix"])
        if b is not None and any(not np.array_equal(g[key], b[key]) for key in ("target_normalized", "observed", "origins", "mean", "std", "channel_ids")):
            raise ValueError("Component parents differ in original cohort/scaler/target/mask")
        if arm["dataset"] not in pasts:
            pasts[arm["dataset"]] = report.canonical_past(task, g["origins"], g["mean"], g["std"])
        past = pasts[arm["dataset"]]
        path = resolve(arm["head_binding"], ctx)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        contract = saved["contract"]
        state_sha = mcal.state_hash(saved["state"])
        if (saved["receipt"] != arm["refit"] or saved["search"] != arm["search"]
                or state_sha != arm["refit"]["final_state_sha256"]
                or saved["head_config"] != dict(length=task["L"], horizon=task["H"], hidden=32)
                or contract["control"] is not control or contract["data_sha256"] != task["data_sha256"]
                or contract["past"] != common.array_identity(past) or contract["origins"] != common.array_identity(g["origins"])
                or contract["scaler"] != dict(mean=common.array_identity(g["mean"]), std=common.array_identity(g["std"]))
                or contract["g"] != arm["parent_g_prediction_binding"]
                or contract["b"] != (None if control else arm["parent_b_prediction_binding"])):
            raise ValueError("Exact final head state, canonical past and scaler/parent contract required")
        reference = references[arm["run_id"]]
        reference_path = resolve(reference["prediction_binding"], ctx)
        with np.load(reference_path, allow_pickle=False) as archive:
            for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids"):
                if common.array_identity(archive[key]) != reference["arrays"][key]:
                    raise ValueError("Independently audited Mcal reference array changed")
            for key in ("target_normalized", "observed", "origins", "channel_ids"):
                if not np.array_equal(archive[key], g[key]):
                    raise ValueError("Component reference differs from canonical Mcal cohort")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata["model_route"] != arm["route_spec"] or metadata["scaler_sha256"] != g["scaler_sha256"]:
                raise ValueError("Mcal reference final state/normalization identity differs")
            reference8 = archive["pred_normalized"][:8].copy()
        arrays = dict(past=past[:8].astype(np.float32), g=g["pred_normalized"][:8].copy(),
                      b=g["pred_normalized"][:8].copy() if control else b["pred_normalized"][:8].copy(),
                      reference=reference8, origins=g["origins"][:8].copy())
        metadata = dict(run_id=arm["run_id"], dataset=arm["dataset"], mode=arm["mode"], seed=arm["seed"],
            state_sha256=state_sha, control=control, head_config=saved["head_config"],
            head_binding=arm["head_binding"], reference_prediction_binding=reference["prediction_binding"],
            parent_g_binding=arm["parent_g_prediction_binding"], parent_b_binding=arm.get("parent_b_prediction_binding"),
            scaler_sha256=g["scaler_sha256"], full_canonical_past_identity=common.array_identity(past),
            input_array_identities={key:common.array_identity(value) for key, value in arrays.items()},
            scope_binding=proof["prerequisite_bindings"]["scope"])
        output = ctx["cache"]/"head_component_costs_001"/"inputs"/(arm["run_id"]+".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            with np.load(output, allow_pickle=False) as archive:
                if any(not np.array_equal(archive[key], value) for key, value in arrays.items()) or json.loads(str(archive["metadata_json"].item())) != metadata:
                    raise ValueError("Preserve differing component input preparation")
        else:
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        prepared.append(dict(metadata=metadata, input_binding=binding(output, ctx), head_binding=arm["head_binding"]))
    return prepared


def _worker_deadline(request):
    if time.time() >= min(request["worker_deadline_unix"], request["original_deadline_unix"]):
        raise TimeoutError("Original wall/GPU reservation reached in owned component worker")


def worker(request_path, output_path):
    request = read(request_path)
    ctx = context(request["asset_root"], request["repo_root"])
    proof = gates(ctx, resolve(request["scope_binding"], ctx), resolve(request["preflight_binding"], ctx))
    if proof["prerequisite_bindings"] != request["prerequisite_bindings"]:
        raise ValueError("Component prerequisites changed after collection admission")
    metadata = request["metadata"]
    if ((metadata["dataset"], metadata["mode"], metadata["seed"], request["batch_size"]) not in EXPECTED
            or metadata["control"] is not (metadata["mode"] == "past_control")
            or metadata["scope_binding"] != request["scope_binding"]
            or metadata["head_binding"] != request["head_binding"]
            or request["original_deadline_unix"] != proof["original_deadline_unix"]
            or request["worker_deadline_unix"] > request["original_deadline_unix"]):
        raise ValueError("Only the admitted frozen head, protocol batch and original deadline may be measured")
    _worker_deadline(request)
    import psutil
    current = psutil.Process(os.getpid())
    lineage = dict(worker=dict(pid=current.pid, create_time=current.create_time()),
                   ancestors=[dict(pid=parent.pid, create_time=parent.create_time()) for parent in current.parents()])
    if request["collector_identity"] not in lineage["ancestors"]:
        raise ValueError("Benchmark process is not descended from the admitted collector")
    write_new(Path(output_path).with_suffix(".started.json"), dict(process_lineage=lineage,
        request_binding=binding(request_path, ctx), collector_identity=request["collector_identity"]))
    import numpy as np
    import torch
    from research.raw_context_multiscale_20261018_v1 import calibration, common
    from research.raw_context_multiscale_mcal_diagnostics_20261019_v1 import run as mcal
    with np.load(resolve(request["input_binding"], ctx), allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in ("past", "g", "b", "reference", "origins")}
        if json.loads(str(archive["metadata_json"].item())) != metadata or any(
                common.array_identity(value) != metadata["input_array_identities"][key] for key, value in arrays.items()):
            raise ValueError("Bound component input arrays changed")
    if any(arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all() for key in ("past", "g", "b", "reference")):
        raise ValueError("Finite resident-input FP32 component tensors required")
    saved = torch.load(resolve(request["head_binding"], ctx), map_location="cpu", weights_only=False)
    if mcal.state_hash(saved["state"]) != metadata["state_sha256"] or saved["head_config"] != metadata["head_config"]:
        raise ValueError("Frozen final component head state changed")
    _worker_deadline(request)
    if not torch.cuda.is_available():
        raise RuntimeError("Registered CUDA device unavailable; do not replace with CPU timing")
    head = calibration.ConditionalHead(**saved["head_config"])
    head.load_state_dict(saved["state"], strict=True)
    head = head.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    x, g = (torch.as_tensor(arrays[key], dtype=torch.float32, device="cuda") for key in ("past", "g"))
    b = g if metadata["control"] else torch.as_tensor(arrays["b"], dtype=torch.float32, device="cuda")
    batch = request["batch_size"]
    batches = [(x, g, b)] if batch == 8 else [(x[index:index+1], g[index:index+1], b[index:index+1]) for index in range(8)]
    parity_predictions = []
    with torch.inference_mode():
        for values in batches:
            _worker_deadline(request)
            output = head(*values, control=metadata["control"])
            torch.cuda.synchronize()
            parity_predictions.append(output.cpu().numpy())
            del output
    prediction = np.concatenate(parity_predictions)
    if not np.allclose(prediction, arrays["reference"], atol=1e-5, rtol=1e-4):
        raise ValueError("GPU head component differs from the fixed CPU Mcal reference")
    parity = dict(status="PASS", atol=1e-5, rtol=1e-4, n_origins=8,
                  reference_prediction_binding=metadata["reference_prediction_binding"],
                  max_abs=float(np.max(np.abs(prediction-arrays["reference"]))))
    trials = []
    with torch.inference_mode():
        for phase, count in (("warmup", 10), ("timed", 20)):
            for index in range(count):
                _worker_deadline(request)
                values = batches[0] if batch == 8 else batches[index % 8]
                torch.cuda.synchronize()
                if phase == "timed":
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = head(*values, control=metadata["control"])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                trials.append(dict(phase=phase, trial=index, whole_seconds=elapsed,
                    origins=arrays["origins"].tolist() if batch == 8 else [int(arrays["origins"][index % 8])], F_calls=0,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if phase == "timed" else None,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2 if phase == "timed" else None))
                del output
    _worker_deadline(request)
    row = dict(component_cost_id=request["component_cost_id"], whole_cost_id=request["whole_cost_id"],
        run_id=metadata["run_id"], dataset=metadata["dataset"], mode=metadata["mode"], seed=metadata["seed"], batch_size=batch,
        device="cuda", dtype="float32", forward_scope=SCOPE, F_calls=0, fixed_first8_origins=arrays["origins"].tolist(),
        process_pid=os.getpid(), process_create_time=psutil.Process(os.getpid()).create_time(),
        peak_allocated_MiB=max(trial["peak_allocated_MiB"] for trial in trials if trial["phase"] == "timed"),
        peak_reserved_MiB=max(trial["peak_reserved_MiB"] for trial in trials if trial["phase"] == "timed"),
        resident_parameters=sum(value.numel() for value in head.parameters()),
        process_GPU_sampled_peak_MiB=None, process_GPU_sampling_status="NOT_MEASURED",
        input_shape=[batch, *arrays["past"].shape[1:]], resident_input_shape=list(arrays["past"].shape),
        head_config=metadata["head_config"], state_sha256=metadata["state_sha256"],
        component_includes_parent_computation=False, component_includes_H2D=False, component_includes_CPU_return=False,
        component_includes_model_load=False, component_includes_feature_construction=True,
        **latency_statistics(trials, batch))
    result = dict(status="COMPLETE", row=row, trials=trials, parity=parity, request_binding=binding(request_path, ctx),
        input_binding=request["input_binding"], head_binding=request["head_binding"],
        prerequisite_bindings=proof["prerequisite_bindings"],
        process_lineage=lineage, request_collector_identity=request["collector_identity"],
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
            CUDA=torch.version.cuda, device=torch.cuda.get_device_name(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            float32_matmul_precision=torch.get_float32_matmul_precision(), autocast_enabled=False,
            precision_policy="Existing pinned-runtime backend defaults preserved; FP32 model/inputs; inference_mode; no autocast"))
    write_new(output_path, result)
    return result


def _append_owned(ctx, event):
    path = _journal_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _spawn_owned(ctx, proof, request, request_path, output_path):
    import psutil
    session = uuid.uuid4().hex
    started = time.time()
    _append_owned(ctx, dict(status="STARTED", session_id=session, time_unix=started,
                           parent_pid=os.getpid(), request_binding=binding(request_path, ctx)))
    process, code, launcher = None, None, None
    try:
        with output_path.with_suffix(".stdout.log").open("xb") as stdout, output_path.with_suffix(".stderr.log").open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-u", "-m", "research."+NAME+".run", "worker",
                "--request", str(request_path), "--output", str(output_path)], cwd=ctx["repo"], stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            launcher = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time())
            try:
                code = process.wait(timeout=max(.001, request["worker_deadline_unix"]-time.time()))
            except subprocess.TimeoutExpired:
                raise TimeoutError("Owned component worker exhausted original reservation")
        if code != 0:
            raise RuntimeError("Owned component worker failed; preserve stderr and prior completed facets")
    finally:
        stopped_after_wrapper = False
        if process is not None:
            if code != 0 or process.poll() is None:
                _stop_owned_tree(process, request, request_path, output_path, launcher, ctx)
            else:
                stopped_after_wrapper = _confirm_owned_exit(process, request, request_path, output_path, launcher, ctx) is False
                if stopped_after_wrapper:
                    code = None
        finished = time.time()
        _append_owned(ctx, dict(status="COMPLETE" if code == 0 else "FAILED", session_id=session,
            time_unix=finished, wall_seconds=finished-started, worker_pid=None if process is None else process.pid,
            launch_identity=launcher, collector_identity=request["collector_identity"],
            exit_code=code, request_binding=binding(request_path, ctx),
            accounting="Owned fresh worker full spawn-to-exit wall upper bound, including cold imports/load/preparation/warmup/parity; not kernel time"))
        if stopped_after_wrapper:
            raise TimeoutError("Actual owned worker outlived its successful wrapper and was stopped at the original deadline")
    return dict(collector=request["collector_identity"], launcher=launcher)


def _started_worker(request, request_path, output_path, launcher, ctx):
    startup_path = Path(output_path).with_suffix(".started.json")
    if not startup_path.exists():
        return None
    startup = read(startup_path)
    lineage = startup["process_lineage"]
    if (startup["request_binding"] != binding(request_path, ctx) or request["collector_identity"] not in lineage["ancestors"]
            or (lineage["worker"] != launcher and launcher not in lineage["ancestors"])):
        raise ValueError("Unverified started worker lineage; keep the owned reservation open")
    return lineage["worker"]


def _confirm_owned_exit(process, request, request_path, output_path, launcher, ctx):
    import psutil
    identity = _started_worker(request, request_path, output_path, launcher, ctx)
    if identity is None:
        raise RuntimeError("Successful wrapper has no worker-start proof; preserve open reservation")
    try:
        child = psutil.Process(identity["pid"])
        if child.create_time() != identity["create_time"]:
            return True
    except psutil.NoSuchProcess:
        return True
    _, alive = psutil.wait_procs([child], timeout=max(0., request["worker_deadline_unix"]-time.time()))
    if alive:
        _stop_owned_tree(process, request, request_path, output_path, launcher, ctx)
        return False
    return True


def _stop_owned_tree(process, request, request_path, output_path, launcher, ctx):
    """Stop only this Popen process and descendants proved by its live lineage."""
    import psutil
    owned = {}
    try:
        parent = psutil.Process(process.pid)
        if launcher is not None and parent.create_time() == launcher["create_time"]:
            owned[parent.pid] = parent
            owned.update({child.pid:child for child in parent.children(recursive=True)})
    except psutil.NoSuchProcess:
        pass
    identity = _started_worker(request, request_path, output_path, launcher, ctx)
    if identity is None and not owned:
        raise RuntimeError("No started-worker proof or live owned tree remains; keep the reservation open until actual exit is proved")
    if identity is not None:
        try:
            child = psutil.Process(identity["pid"])
            if child.create_time() == identity["create_time"]:
                owned[child.pid] = child
                owned.update({descendant.pid:descendant for descendant in child.children(recursive=True)})
        except psutil.NoSuchProcess:
            pass
    for child in owned.values():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    if alive:
        raise RuntimeError("Owned component descendants still live; preserve open reservation and inspect these handles")
    process.wait(timeout=3)


def collect(ctx, scope_path, preflight_path):
    started = time.time()
    proof = gates(ctx, scope_path, preflight_path)
    if owned_wall(_events(ctx), time.time())["open_sessions"]:
        raise ValueError("Unclosed owned-worker session must be resolved from its actual process before another collection")
    prepared = prepare_inputs(ctx, proof)
    import psutil
    collector_identity = dict(pid=os.getpid(), create_time=psutil.Process(os.getpid()).create_time())
    facets = read(resolve(proof["prerequisite_bindings"]["cost_index"], ctx))["facets"]
    receipts = []
    directory = ctx["cache"]/"head_component_costs_001"
    for item in prepared:
        for batch in (1, 8):
            metadata = item["metadata"]
            identifier = metadata["run_id"]+"_head_component_B"+str(batch)
            done_path = directory/"completed"/(identifier+".json")
            if done_path.exists():
                previous = read(done_path)
                receipt = read(resolve(previous["receipt_binding"], ctx))
                if receipt["prerequisite_bindings"] != proof["prerequisite_bindings"]:
                    raise ValueError("Prior component prerequisites changed; preserve completed measurement")
                receipts.append(receipt)
                continue
            matches = [facet for facet in facets if facet["run_id"] == metadata["run_id"] and facet["schedule"] == "native" and facet["batch"] == batch]
            if len(matches) != 1:
                raise ValueError("Exact same-head native whole-path cost_id required for T3 join")
            used = owned_wall(_events(ctx), time.time())["seconds"]
            remaining = min(proof["original_deadline_unix"]-time.time(), 14400-proof["GPU_base_seconds"]-used)
            if remaining <= 0:
                raise TimeoutError("Original GPU/wall budget exhausted before next component facet")
            attempt = 1
            while (directory/"attempts"/identifier/f"attempt{attempt:03}").exists():
                attempt += 1
            attempt_dir = directory/"attempts"/identifier/f"attempt{attempt:03}"
            attempt_dir.mkdir(parents=True)
            request_path, output_path = attempt_dir/"request.json", attempt_dir/"receipt.json"
            request = dict(asset_root=str(ctx["asset"]), repo_root=str(ctx["repo"]), scope_binding=binding(scope_path, ctx),
                preflight_binding=binding(preflight_path, ctx), prerequisite_bindings=proof["prerequisite_bindings"],
                metadata=metadata, input_binding=item["input_binding"], head_binding=item["head_binding"],
                collector_identity=collector_identity,
                component_cost_id=identifier, whole_cost_id=matches[0]["cost_id"], batch_size=batch,
                original_deadline_unix=proof["original_deadline_unix"],
                worker_deadline_unix=min(proof["original_deadline_unix"], time.time()+remaining))
            write_new(request_path, request)
            launch_proof = _spawn_owned(ctx, proof, request, request_path, output_path)
            receipt = read(output_path)
            if receipt["request_binding"] != binding(request_path, ctx):
                raise ValueError("Fresh owned-worker result bound to different request")
            receipt["launch_proof"] = launch_proof
            validate_lineage(receipt)
            verified_path = output_path.with_name("verified_receipt.json")
            write_new(verified_path, receipt)
            write_new(done_path, dict(receipt_binding=binding(verified_path, ctx), raw_worker_receipt_binding=binding(output_path, ctx)))
            receipts.append(receipt)
    audit_receipts(receipts)
    if binding(ctx["cache"]/"EVENTS.jsonl", ctx) != proof["prerequisite_bindings"]["primary_EVENTS"]:
        raise ValueError("Primary EVENTS changed during component measurement")
    owned = owned_wall(_events(ctx), time.time())
    if owned["open_sessions"] or proof["GPU_base_seconds"]+owned["seconds"] > 14400:
        raise ValueError("Owned component ledger must close within original GPU reservation")
    finished = time.time()
    if finished > proof["original_deadline_unix"]:
        raise TimeoutError("Original deadline exhausted before component publication")
    resources = dict(prerequisite_bindings=proof["prerequisite_bindings"], GPU_base_seconds=proof["GPU_base_seconds"],
        component_owned_worker_wall_seconds=owned["seconds"], GPU_total_process_wall_upper_bound_seconds=proof["GPU_base_seconds"]+owned["seconds"],
        GPU_limit_seconds=14400, original_deadline_unix=proof["original_deadline_unix"],
        owned_sessions=owned, owned_events_binding=binding(_journal_path(ctx), ctx), owned_events=_events(ctx),
        primary_EVENTS_before=proof["prerequisite_bindings"]["primary_EVENTS"], primary_EVENTS_after=binding(ctx["cache"]/"EVENTS.jsonl", ctx),
        primary_EVENTS_appended=False, started_unix=started, finished_unix=finished, collection_wall_seconds=finished-started,
        collection_wall_scope="Admission, CPU preparation, all owned subprocesses and receipt checks through the frozen snapshot; public serialization/copy/replay are separately included in the root completion receipt",
        V_prefix_numeric_decoded_by_bounded_preparation=True, numeric_computations_split="M_cal only",
        GPU_accounting="Original frozen base plus each new owned worker full spawn-to-exit wall exactly once, including failures; not kernel time")
    attempt = 1
    while (directory/f"public_output{attempt:03}").exists():
        attempt += 1
    output = directory/f"public_output{attempt:03}"
    result = render_public(receipts, resources, output)
    for name in (CSV_NAME, TRIALS_NAME, VALIDATION_NAME, INPUTS_NAME):
        target = ctx["report"]/name
        data = (output/name).read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError("Preserve existing public component output: "+name)
        else:
            with target.open("xb") as stream:
                stream.write(data)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "worker", "replay"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "collect":
        if args.asset_root is None or args.scope is None or args.preflight is None:
            parser.error("collect requires --asset-root, --scope and --preflight")
        result = collect(context(args.asset_root), args.scope, args.preflight)
    elif args.stage == "worker":
        if args.request is None or args.output is None:
            parser.error("worker requires --request and --output")
        result = worker(args.request, args.output)
    else:
        if args.source_dir is None or args.output_dir is None:
            parser.error("replay requires --source-dir and --output-dir")
        result = replay(args.source_dir, args.output_dir)
    print(json.dumps(dict(status=result["status"], stage=args.stage)))


if __name__ == "__main__":
    main()
