"""Explicit ROOT orchestration of the frozen selection and original final APIs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path, root):
    return dict(public_relative_path=path.relative_to(root).as_posix(), sha256=sha(path))


def prepare(public, cache, plan_path):
    if sha(plan_path) != "7b5aff975382f4b8a3ecad6e97209c782da80a51f8dde8ef18a2854bf8b96742":
        raise ValueError("Explicit selection46 plan changed")
    plan = read(plan_path)
    pins_path = public / "diagnostics/multik_postfinal_resource_1792096163404501000/PLAN.json"
    if sha(pins_path) != "45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601":
        raise ValueError("Postfinal source identity preparation changed")
    pins = read(pins_path)
    for name, digest in pins["source39"].items():
        if sha(public / name) != digest:
            raise ValueError("Frozen source39 changed: " + name)
    replay_path = public / "execution_batches/V_replay_1792096495154205000.json"
    if sha(replay_path) != "9af03b72efff479cf498c9341d06db65d5deb772337b69eaf8d82035d5756e57":
        raise ValueError("Actual completed replay50 batch changed")
    replay = read(replay_path)
    if (replay["status"] != "COMPLETE_SELECTED_ROUTES" or replay["succeeded"] != 50
            or replay["failed"] != 0 or replay["deferred_run_ids"] != []
            or replay["run_ids"] != plan["replay_run_ids"]):
        raise ValueError("Exact actual replay50 is not complete")
    if len(plan["selected_run_ids"]) != 46 or len(set(plan["selected_run_ids"])) != 46:
        raise ValueError("Exact46 selection is required")
    for row in replay["records"]:
        if row["status"] != "PASS" or sha(public / row["receipt_public_relative_path"]) != row["receipt_sha256"]:
            raise ValueError("Actual replay record changed")
    with (public / "EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["execution_status"] == "RUNNING" for row in rows) or (cache / "ACTIVE_GPU_LEASE.json").exists():
        raise ValueError("Final selection requires quiescent campaign")
    if sha(public / "EXPERIMENT_REGISTRY.csv") != "816209d21b6345b1b3e65b4ae9ff42d65ada8e6e342f42f179141d8dcee88fab":
        raise ValueError("Registry bytes changed before selection")
    for relative, digest in [
        ("diagnostics/publication_pack_execution_1792096101396562900/RESULT.json", "ddb8ac8f74aac0bcacd9433a5c772ea9040d09926f4ee0c39d2c89b008bc1cee"),
        ("diagnostics/publication_multik_lossless_1792094047117000000/bundle_1792096101396562900/BUILD_RESULT.json", "50313829567552c3f7be9ca57e4cde0ef0455fa488d1d298c559da6b9d1a0f40"),
        ("publication_derivatives/gzip_aux95_1792094747124151400/AUX_RESULT.json", "e3d27096f71dd63f01e76366b82bdb5809028aef1a5517ee1ad3f0f5eae121a0")]:
        if sha(public / relative) != digest:
            raise ValueError("Actual lossless packaging proof changed")
    return plan, pins, binding(replay_path, public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    public = Path(os.environ["INFORMS_PUBLIC_ROOT"]).resolve()
    cache = Path(os.environ["INFORMS_CACHE_ROOT"]).resolve()
    asset = Path(os.environ["INFORMS_ASSET_ROOT"]).resolve()
    plan, pins, replay_binding = prepare(public, cache, args.selection_plan.resolve())
    if not args.execute:
        print(json.dumps(dict(status="ROOT_FINAL_API_READY_METADATA_ONLY", selected=46, actual_replay_PASS=50,
                              lock_written=False, T_executed=False, GPU_started=False)))
        return
    sys.path.insert(0, str(public))
    import final_evaluation
    import resources
    import psutil
    public_bytes = sum(path.stat().st_size for path in public.rglob("*") if path.is_file() and not path.is_symlink())
    combined = resources.check_storage_budget(cache, max_bytes=32212254720-public_bytes, pending_bytes=1073741824)
    stamp = str(time.time_ns())
    folder = public / "diagnostics" / ("ROOT_final_T46_E0B55_actual_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    new(folder / "PRELOCK_GATES.json", dict(status="PASS_ACTUAL_REPLAY_AND_FROZEN_INPUTS",
        replay=replay_binding, source39=pins["source39"], selected_run_ids=plan["selected_run_ids"],
        selection_plan_sha256=sha(args.selection_plan), combined_cache_pending_bytes=combined,
        public_actual_bytes=public_bytes, combined_pending_allowance_bytes=1073741824,
        public_incremental_cap_bytes=262144000, byte_exact_base_and_pack_projection_bytes=184812902,
        T46_gzip_ratio_scenario_bytes=56138954, scenario_is_not_an_upper_bound=True,
        actual_final_candidate_cap_check_still_required=True, no_numeric_reduction_permitted=True,
        created_unix=time.time()))
    lock_started = time.time()
    lock = final_evaluation.lock_selection(plan["selected_run_ids"], plan["decision"], public, cache)
    new(folder / "LOCK_RESULT.json", dict(status=lock["status"], selection=binding(public/"SELECTION_LOCK.json",public),
        selected=len(lock["allowed_run_ids"]), started_unix=lock_started, finished_unix=time.time(),
        mandatory_coverage_complete=lock["mandatory_coverage_complete"], T_targets_read_during_selection=False))
    print(json.dumps(dict(phase="LOCKED", selected=46, status=lock["status"])), flush=True)
    launcher_path = public / "diagnostics/e0_b_execution_ready_20261016T012806/e0_b_launcher.py"
    if sha(launcher_path) != "f406d0566ec5803762854c0dccba7dc93485e6d7707594bbf29d9ecb8e3240f0":
        raise ValueError("Original E0 pinned launcher changed")
    spec = importlib.util.spec_from_file_location("root_pinned_e0_launcher", launcher_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    lease = resources.GPUReservation(cache, total_gpu_seconds=43200, reserve_seconds=0,
        wall_deadline=1792223150, purpose="audit", kind="ROOT_final_T46_and_exact_E0B55")
    phase_outputs = {}
    failures = []
    registry_before = sha(public / "EXPERIMENT_REGISTRY.csv")
    with lease:
        try:
            started = time.time()
            prior_batches = set((public/"execution_batches").glob("T_final_*.json"))
            try:
                result = final_evaluation.execute_T(cache, public, lease, lease.deadline)
                candidates = [p for p in (public/"execution_batches").glob("T_final_*.json")
                              if p not in prior_batches and read(p) == result]
                if len(candidates) != 1:
                    raise ValueError("Exact new T output binding is ambiguous")
                phase_outputs["T_final"] = binding(candidates[0], public)
                new(folder / "T_PHASE_RESULT.json", dict(output_binding=phase_outputs["T_final"], started_unix=started,
                    finished_unix=time.time(), status=result["status"], succeeded=result["succeeded"], failed=result["failed"],
                    deferred_run_ids=result["deferred_run_ids"]))
                print(json.dumps(dict(phase="T_final", status=result["status"], succeeded=result["succeeded"],
                    failed=result["failed"], remaining_seconds=lease.remaining)), flush=True)
            except Exception as error:
                failures.append(dict(phase="T_final", error_type=type(error).__name__))
                new(cache/"final_execution"/("ROOT_final_T_error_"+stamp+".json"),
                    dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            started = time.time()
            try:
                if not (all(sha(public/name) == digest for name,digest in pins["source39"].items())
                        and sha(public/"EXPERIMENT_REGISTRY.csv") == registry_before):
                    raise ValueError("Frozen inputs changed; independent E0 cannot run safely")
                if lease.remaining <= 30:
                    raise resources.BudgetExceeded("No bounded time remains for independent E0 inference")
                output = legacy.run_with_owner(lease, asset, cache, max_seconds=lease.remaining-15, gpu_devices="0")
                if not output.get("registry_relative_path"):
                    raise ValueError("E0 actual registry output is absent")
                phase_outputs["E0_B"] = binding(public/output["registry_relative_path"], public)
                new(folder / "E0_PHASE_RESULT.json", dict(output_binding=phase_outputs["E0_B"], started_unix=started,
                    finished_unix=time.time(), launcher_summary=output))
                print(json.dumps(dict(phase="E0_B", status=output["status"], rescored=output["checkpoints_rescored"],
                    matching=output["checkpoints_matching_recorded_curve"], remaining_seconds=lease.remaining)), flush=True)
            except Exception as error:
                failures.append(dict(phase="E0_B", error_type=type(error).__name__))
                new(cache/"final_execution"/("ROOT_final_E0_error_"+stamp+".json"),
                    dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        finally:
            for child in list(lease._authorized_children.values()):
                try:
                    process = psutil.Process(child["pid"])
                    if abs(process.create_time()-child["create_time"]) < .001:
                        process.wait(timeout=max(1., min(60., lease.remaining)))
                except psutil.NoSuchProcess:
                    pass
            safe_owner = dict(lease_id=lease.lease_id, pid=lease.pid, create_time=lease.create_time,
                started_unix=lease.started_unix, deadline=lease.deadline,
                authorized_children=list(lease._authorized_children.values()))
            new(folder/"OWNER_SAFE_FIELDS.json", safe_owner)
    closure = cache/"gpu_receipts"/(lease.lease_id+".json")
    close = read(closure)
    sources_unchanged = all(sha(public/name) == digest for name,digest in pins["source39"].items())
    registry_unchanged = registry_before == sha(public/"EXPERIMENT_REGISTRY.csv")
    lock_unchanged = final_evaluation._validated_lock(cache, public)["selection_sha256"] == lock["selection_sha256"]
    if not (sources_unchanged and registry_unchanged and lock_unchanged):
        raise ValueError("Frozen final inputs changed")
    new(folder/"RESULT.json", dict(status="TERMINAL_ACTUAL_PHASES_REVIEW_REQUIRED", phase_outputs=phase_outputs,
        failures=failures, owner_safe_fields=binding(folder/"OWNER_SAFE_FIELDS.json", public),
        central_closure=dict(cache_relative_path=closure.relative_to(cache).as_posix(), sha256=sha(closure)),
        closure_status=close["status"], source39_unchanged=True, registry_bytes_unchanged=True,
        selection_lock_unchanged=True, total_GPU_process_wall_charged=resources.gpu_usage_seconds(cache),
        created_unix=time.time(), campaign_completion_claimed=False))
    print(json.dumps(dict(status="ROOT_FINAL_PHASES_TERMINAL", output=folder.relative_to(public).as_posix(),
        failures=failures, closed_GPU_seconds=resources.gpu_usage_seconds(cache))), flush=True)


if __name__ == "__main__":
    main()
