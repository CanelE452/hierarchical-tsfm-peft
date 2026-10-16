"""Fresh NO_F native depth routes. Default CLI reads standard-library metadata only."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
DATASETS = ("jena", "electricity", "pems08")
FAMILIES = ("DIRECT_G", "DIRECT_B2", "DIRECT_B4")
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status",
           "publication_status", "reason", "resume_command"}
ANCHOR = "diagnostics/new12_saved_V_rescore_1792083486504811200/RESULT.json"
ANCHOR_SHA = "f38d7946b08b9d2ac29a8ccfe52ab322184a97e445b6032065855fd3b05f1289"
HARD_WALL = 1792050350 + 172800
STORAGE_LIMIT = 30 * 1024 ** 3
PENDING_BYTES = 512 * 1024 ** 2
RSS_LIMIT = 4 * 1024 ** 3


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _registry(public):
    blob = (public / "EXPERIMENT_REGISTRY.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
    rows = list(reader)
    if len({row["experiment_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate registered experiment ID")
    return blob, rows


def _sources(public, expected=None):
    if expected is None:
        path = public / ANCHOR
        if file_sha(path) != ANCHOR_SHA:
            raise ValueError("Immutable source33 audit anchor differs")
        expected = _json(path)["sources_after"]
    if any(not (public / name).is_file() or file_sha(public / name) != value for name, value in expected.items()):
        raise ValueError("Frozen scientific source changed")
    return dict(expected)


def _open_selection(public):
    path = public / "SELECTION_LOCK.json"
    if path.exists():
        value = _json(path)
        if not isinstance(value, dict) or not isinstance(value.get("locked"), bool) or value["locked"]:
            raise ValueError("Final selection is closed or malformed; CPU fresh work cannot execute")


def _science(row):
    return {key: str(value) for key, value in row.items() if key not in MUTABLE}


def _fresh_paths(public, cache, identity):
    return (public / "runs" / identity, cache / "runs" / identity,
            public / "predictions" / identity, cache / "predictions" / identity,
            cache / "native_cpu_depth_claims" / (identity + ".json"))


def _safe_reason(error, public, cache):
    reason = str(error)
    roots = ((public, "<public>"), (cache, "<cache>"),
             (Path(os.environ.get("INFORMS_ASSET_ROOT", str(cache.parents[1]))), "<assets>"))
    for root, label in roots:
        for value in (str(root), str(root).replace("\\", "/"), str(root).replace("/", "\\")):
            reason = re.sub(re.escape(value), lambda match: label, reason, flags=re.IGNORECASE)
    if re.search(r"[A-Za-z]:[\\/]|\\\\[^\\]|/(?:Users|home|mnt|media)/", reason):
        return "Route error details preserved in the private diagnostic"
    return reason


def build_plan(public_root, cache_root, *, expected_sources=None):
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    pins = _sources(public, expected_sources)
    blob, rows = _registry(public)
    registered = {row["experiment_id"]: row for row in rows}
    assets_path = public / "DATA_CONTRACT.json"
    assets = _json(assets_path)["assets"]
    routes, proposals = [], []
    for dataset in DATASETS:
        asset = assets[dataset]
        channels, horizon = len(asset["columns"]), int(asset["H"])
        for family in FAMILIES:
            initial = f"primary_{dataset}_E1_{family}_k0_s92601"
            identity = initial + "__cpu_fresh_e3"
            row = registered[initial]
            source_path = public / "runs" / initial / "SOURCE_CONTRACT.json"
            receipt_path = source_path.with_name("BASE_FIT_RECEIPT.json")
            source, receipt = _json(source_path), _json(receipt_path)
            spec = source["spec"]
            axes = dict(L=512, H=horizon, C=channels, columns=asset["columns"], fold="primary",
                        source_array_fingerprints=asset["array_sha256"])
            required = dict(run_id=initial, dataset=dataset, family=family, latent_k=channels, seed=92601,
                            fold="primary", lr=.001, microbatch=4, effective_batch=16, device="cuda")
            if (any(spec.get(key) != value for key, value in required.items())
                    or spec.get("role", "base_parent") != "base_parent"
                    or source["data_sha"] != asset["data_sha256"] or source["model_sha"] != "NO_FOUNDATION"
                    or source["config_sha"] != _fingerprint(dict(spec=spec, axes=axes))
                    or source["source_sha"] != _fingerprint(source["files"])
                    or any(pins.get(name) != sig for name, sig in source["files"].items())
                    or receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != initial
                    or row["family"] != family or row["fold"] != "primary" or row["latent_k"] != "0"):
                raise ValueError("Initial CPU fresh provenance/recipe identity differs: " + initial)
            fresh = copy.deepcopy(spec)
            fresh.update(run_id=identity, device="cpu", epochs=[.25, 1., 3.],
                         stage_label="registered_cpu_fresh_native_depth",
                         provenance=dict(initial_run_id=initial, source_contract_sha256=file_sha(source_path),
                                         source_contract_public_relative_path=source_path.relative_to(public).as_posix(),
                                         initial_receipt_sha256=file_sha(receipt_path),
                                         initial_artifact_sha256_as_declared=receipt["artifact_sha256"],
                                         original_artifact_or_optimizer_or_RNG_reused=False,
                                         foundation_identity="NO_FOUNDATION", fresh_CPU_condition=True))
            changed = {key for key in set(spec) | set(fresh) if spec.get(key) != fresh.get(key)}
            if changed - {"run_id", "device", "epochs", "stage_label", "provenance"}:
                raise ValueError("CPU fresh route changed an unapproved recipe field")
            proposal = dict(row, experiment_id=identity, requirement="REGISTERED_CPU_FRESH_NATIVE_DEPTH",
                            execution_status="PENDING_QUEUE", training_status="NOT_STARTED", validity_status="NOT_CHECKED",
                            scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED",
                            reason="Fresh CPU NO_FOUNDATION native search .25/1/3 and selected-epoch A60 refit; GPU state not reused",
                            planned_epoch_endpoints="0.25|1.0|3.0",
                            resume_command="python native_cpu_depth.py --execute --plan-file <ROOT_saved_plan>")
            registration = "ROOT_PREREGISTRATION_REQUIRED"
            if identity in registered:
                registration = "REGISTERED" if _science(registered[identity]) == _science(proposal) else "CONFLICTING_OWN_ROW_BLOCKED"
            routes.append(dict(run_id=identity, initial_run_id=initial, dataset=dataset, family=family,
                               spec=fresh, expected_data_sha256=asset["data_sha256"], foundation_identity="NO_FOUNDATION",
                               registration_status=registration,
                               existing_paths=[path.relative_to(public if path.is_relative_to(public) else cache).as_posix()
                                               for path in _fresh_paths(public, cache, identity) if path.exists()]))
            proposals.append(proposal)
    return dict(status="CPU_FRESH_NATIVE_METADATA_PLAN_ONLY", routes=routes, proposed_registry_rows=proposals,
                exact_run_ids=[route["run_id"] for route in routes], source_pins=pins,
                helper_sha256=file_sha(Path(__file__)), data_contract_sha256=file_sha(assets_path),
                registry_sha256_at_plan=hashlib.sha256(blob).hexdigest(), epochs=[.25, 1., 3.],
                NO_FOUNDATION=True, actual_F_execution_planned=0, GPU_lease_planned=False,
                CPU_threads=2, RSS_limit_bytes=RSS_LIMIT, max_initial_wall_seconds=1800,
                hard_wall_deadline_unix=HARD_WALL, combined_storage_limit_bytes=STORAGE_LIMIT,
                metadata_only=True, registry_written=False, model_or_data_loaded=False,
                CPU_native_runtime_estimate="UNKNOWN: only DIRECT_G CPU measured previously; DIRECT_B2/B4 CPU unmeasured",
                E2_parent_selection_modified=False, old_GPU_extension_completion_claimed=False)


def _storage(public, cache):
    used = 0
    for root in (public, cache):
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not Path(directory, name).is_symlink()]
            for name in files:
                path = Path(directory, name)
                try:
                    if not path.is_symlink(): used += path.stat().st_size
                except FileNotFoundError:
                    continue
    if used + PENDING_BYTES > STORAGE_LIMIT or any(shutil.disk_usage(root).free < PENDING_BYTES for root in (public, cache)):
        raise RuntimeError("Combined campaign storage30GiB or pending512MiB free-space limit")


class _Runtime:
    def __init__(self):
        os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        for key in ("INFORMS_GPU_TOKEN", "INFORMS_GPU_OWNER_PID", "INFORMS_GPU_OWNER_CREATE_TIME",
                    "INFORMS_GPU_LEASE_ID", "INFORMS_GPU_OWNER_PORT", "INFORMS_GPU_DEADLINE"):
            os.environ.pop(key, None)
        import torch
        import psutil
        import data_contract
        import fit_base
        import prediction_cache
        import registry_extension
        torch.set_num_threads(2)
        if torch.cuda.is_initialized(): raise RuntimeError("Fresh CPU process required; CUDA is initialized")
        self.process = psutil.Process()
        self.data, self.base, self.pred, self.registry = data_contract, fit_base, prediction_cache, registry_extension

    def load_task(self, dataset, cache):
        return self.data.load_task(dataset, cache_dir=cache)

    def fit(self, spec, task, cache, public, foundation, deadline):
        self.registry.require_pre_selection(public, "fresh CPU native fit")
        return self.base.fit_base(spec, task, cache, public, foundation=foundation, deadline=deadline)

    def predict(self, identity, task, split, cache, public, foundation, **kwargs):
        self.registry.require_pre_selection(public, "fresh CPU native V inference")
        return self.pred.predict_parent(identity, task, split, cache, public, foundation, **kwargs)


def execute_plan(plan, public_root, cache_root, *, max_seconds=1800, expected_sources=None,
                 _runtime=None, _skip_resource_monitor=False):
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    _open_selection(public)
    if not 0 < float(max_seconds) <= 1800: raise ValueError("Initial CPU wall must be positive and <=1800s")
    rebuilt = build_plan(public, cache, expected_sources=expected_sources)
    static = ("run_id", "initial_run_id", "dataset", "family", "spec", "expected_data_sha256", "foundation_identity")
    if (plan["helper_sha256"] != file_sha(Path(__file__)) or plan["source_pins"] != rebuilt["source_pins"]
            or plan["data_contract_sha256"] != rebuilt["data_contract_sha256"]
            or plan["exact_run_ids"] != rebuilt["exact_run_ids"]
            or [route["run_id"] for route in plan["routes"]] != rebuilt["exact_run_ids"]
            or plan["proposed_registry_rows"] != rebuilt["proposed_registry_rows"]
            or any(any(a[key] != b[key] for key in static) for a, b in zip(plan["routes"], rebuilt["routes"]))):
        raise ValueError("Saved CPU plan source/recipe/proposed scientific identities differ")
    started = time.time(); deadline = min(started + float(max_seconds), HARD_WALL)
    stamp = str(time.time_ns()); output = public / "diagnostics" / ("native_cpu_depth_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    private = cache / "native_cpu_depth" / stamp; private.mkdir(parents=True, exist_ok=False)
    before, before_rows = _registry(public)
    _new(output / "STARTED.json", dict(status="STARTED_CPU_FRESH_NO_F", exact_run_ids=plan["exact_run_ids"],
          source_pins=plan["source_pins"], helper_sha256=plan["helper_sha256"], deadline_unix=deadline,
          pid=os.getpid(), private_diagnostic_cache_relative_path=private.relative_to(cache).as_posix(),
          registry_write_by_worker=False, existing_GPU_state_reused=False, GPU_lease=False))
    records, tasks = [], {}
    runtime = _runtime
    stop = threading.Event(); monitor = {"peak_RSS_bytes": 0}
    def watch():
        while not stop.wait(.2):
            rss = runtime.process.memory_info().rss
            monitor["peak_RSS_bytes"] = max(monitor["peak_RSS_bytes"], rss)
            if rss > RSS_LIMIT:
                _new(output / "RESOURCE_STOP.json", dict(status="OWN_CPU_WORKER_RSS_LIMIT", RSS_bytes=rss,
                     pid=os.getpid(), current_run_id=identity, expected_process_exit_code=98))
                os._exit(98)
    thread = None
    proposals = {row["experiment_id"]: row for row in plan["proposed_registry_rows"]}
    try:
        for route in plan["routes"]:
            identity = route["run_id"]; record = dict(run_id=identity, device="cpu", foundation_identity="NO_FOUNDATION")
            try:
                if time.time() >= deadline:
                    record.update(status="DEFERRED_RESOURCE", reason="CPU work-boundary deadline reached"); records.append(record); continue
                _open_selection(public); _sources(public, expected_sources)
                _, current = _registry(public); own = [row for row in current if row["experiment_id"] == identity]
                if len(own) != 1 or _science(own[0]) != _science(proposals[identity]) or own[0]["execution_status"] != "PENDING_QUEUE":
                    raise ValueError("Exact ROOT preregistered PENDING CPU identity is required")
                if any(path.exists() for path in _fresh_paths(public, cache, identity)):
                    raise ValueError("Existing run/prediction/claim preserved; CPU fresh bridge never resumes")
                _storage(public, cache)
            except Exception as error:
                record.update(status="NOT_EXECUTED_PRECONDITION", error_type=type(error).__name__,
                              reason=_safe_reason(error, public, cache)); records.append(record); continue
            try:
                _new(cache / "native_cpu_depth_claims" / (identity + ".json"), dict(run_id=identity, pid=os.getpid(),
                     started_unix=time.time(), deadline_unix=deadline, helper_sha256=plan["helper_sha256"], fresh_CPU=True))
                if runtime is None:
                    runtime = _Runtime()
                    if not _skip_resource_monitor:
                        thread = threading.Thread(target=watch, daemon=True); thread.start()
                ds = route["dataset"]
                if ds not in tasks: tasks[ds] = runtime.load_task(ds, cache)
                task, spec = tasks[ds], route["spec"]
                if (task["data_sha256"] != route["expected_data_sha256"] or len(task["columns"]) != spec["latent_k"]
                        or int(task["L"]) != 512 or int(task["H"]) != int(proposals[identity]["horizon"])):
                    raise ValueError("Actual safe-prefix Task axes/data differ from CPU plan")
                _open_selection(public)
                receipt = runtime.fit(spec, task, cache, public, None, deadline)
                if receipt.get("status") != "COMPLETE_VALID":
                    record.update(status="DEFERRED_RESOURCE", fit_status=receipt.get("status"), training_status=receipt.get("training_status")); records.append(record); continue
                source = _json(public / "runs" / identity / "SOURCE_CONTRACT.json")
                if source.get("model_sha") != "NO_FOUNDATION" or source.get("spec") != spec or receipt.get("F_logical_streams") != 0:
                    raise ValueError("Actual source/fit is not this fresh NO_FOUNDATION F0 recipe")
                for stage in ("search", "refit"):
                    stage_receipt = _json(public / "runs" / identity / stage / "RECEIPT.json")
                    counters = {key:value for key,value in stage_receipt.items() if key.startswith("F_") and "streams" in key}
                    if not counters or any(value != 0 for value in counters.values()):
                        raise ValueError("Actual native training F counters are not zero")
                _open_selection(public)
                prediction = runtime.predict(identity, task, "V_select", cache, public, None,
                                             fold="primary", role="base_parent", deadline=deadline)
                cost, manifest = prediction["cost_record"], prediction["manifest"]
                if cost["actual_F_streams"] != 0 or cost["expected_F_streams"] != 0 or manifest["run_id"] != identity or manifest["split"] != "V_select":
                    raise ValueError("Actual CPU V identity/F counters differ")
                path = public / "predictions" / identity / "V_select/PREDICTION_MANIFEST.json"
                record.update(status="FIT_AND_V_COMPLETE_NO_F_CPU", training_status=receipt["training_status"],
                              requested_epochs=receipt["epochs"], selected_epoch=receipt["selected_epoch"],
                              actual_search_updates=receipt["search_updates"], actual_refit_updates=receipt["refit_updates"],
                              actual_F_streams=cost["actual_F_streams"], expected_F_streams=cost["expected_F_streams"],
                              manifest_public_relative_path=path.relative_to(public).as_posix(), manifest_sha256=file_sha(path),
                              score=manifest["score"], original_optimizer_RNG_or_search_reused=False)
            except Exception as error:
                _new(private / (identity + ".json"), dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
                record.update(status="FAILED_PRESERVED", error_type=type(error).__name__,
                              reason=_safe_reason(error, public, cache))
            records.append(record)
    finally:
        stop.set()
        if thread is not None: thread.join(timeout=1)
    _sources(public, expected_sources)
    after, after_rows = _registry(public)
    before_map, after_map = ({row["experiment_id"]: row for row in rows} for rows in (before_rows, after_rows))
    external = [identity for identity in before_map.keys() & after_map.keys() if before_map[identity] != after_map[identity]]
    if any(_science(before_map[identity]) != _science(after_map[identity]) for identity in before_map.keys() & after_map.keys()):
        raise ValueError("Registry scientific identity changed while CPU worker ran; preserve outputs for review")
    completed = sum(row["status"] == "FIT_AND_V_COMPLETE_NO_F_CPU" for row in records)
    result = dict(status="COMPLETE_CPU_FRESH_NO_F_ROUTES" if completed == 9 else "PARTIAL_CPU_FRESH_PRESERVED",
                  records=records, completed_routes=completed, requested_routes=9, started_unix=started, finished_unix=time.time(),
                  source_pins_unchanged=True, registry_before_sha256=hashlib.sha256(before).hexdigest(),
                  registry_after_sha256=hashlib.sha256(after).hexdigest(), external_mutable_changed_ids=external,
                  external_added_registry_ids=sorted(after_map.keys()-before_map.keys()), registry_worker_writes=0,
                  monitor=monitor, GPU_lease_acquired=False, foundation_instances=0, T_accessed=False,
                  old_GPU_extension_completion_claimed=False, convergence_claimed=False, E2_parent_selection_modified=False,
                  new_CPU_device_condition=True, public_directory=output.relative_to(public).as_posix())
    _new(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", default=str(HERE))
    parser.add_argument("--cache-root", default=os.environ.get("INFORMS_CACHE_ROOT"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-file")
    parser.add_argument("--output-plan")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not args.cache_root: parser.error("Explicit existing campaign cache root required")
    if args.execute:
        if not args.plan_file: parser.error("ROOT-saved plan and separate preregistration are required")
        result = execute_plan(_json(args.plan_file), args.public_root, args.cache_root, max_seconds=args.max_seconds)
    else:
        result = build_plan(args.public_root, args.cache_root)
        if args.output_plan: _new(args.output_plan, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
