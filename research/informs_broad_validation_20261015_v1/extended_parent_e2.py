"""Explicit extended-A-parent E2 bridge; default mode reads stdlib metadata only."""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
MAIN = ("beijing", "jena", "electricity", "pems08")
TOTAL_SECONDS = 43200
WALL_DEADLINE = 1792050350 + 172800
PENDING_BYTES = 1024**3
STORAGE_BYTES = 30 * 1024**3
SOURCE18 = {
    "audit_campaign.py": "f7b695a33d48515b17b7e25a4e2ffc5485c96c1cbb9fc83a53322cbb7c5a433f",
    "audit_predictions.py": "a7d93919362e61acef15f84501783630d8ba07bb46920d29c1b2fa4ac7215dd2",
    "combination.py": "266986206456312031c618251388793ffae18bf53682e65aa9980156b14cba93",
    "data_contract.py": "0e5fb9a03bd44fe13a9777b774601a54573f221abc0eb41a5bb41764b864b31b",
    "deployment.py": "9ccf6a6e142ce49ab3cb1c1a0dc8d1594890f05c3a7dfea80fa217111c57deba",
    "evaluation.py": "275cd8be02b277ceba916eb6b00e7296f46694d0d592dbc1e0fb231f349f2fbd",
    "final_evaluation.py": "b59a378dad9b0edaf93f5eb3c09a58eedc7ceabc4409e808453744fe3d5efdde",
    "fit_base.py": "c59cc77ba48e3f061ffa8d2385442286ea989b3ab86c17ea352caeb6d2c1bf20",
    "fit_control.py": "12876eda8c0dbdb2957ff618be810c2ed496467475ddf3eb996ccc23670489fb",
    "gpu_runtime.py": "c633f1ebf4eb24ea3fa2b025ffa6cea4ec3697c8e9285d971174179560596158",
    "networks.py": "787f9caf91f32a001f929f324c7bcb570b9270331f6f60f0f56872529f86dc54",
    "ordered_pilot.py": "1fa570104a8ca5167e5bc3559c322a326cb0ae8e440baf7807a9dc92736ea56e",
    "prediction_cache.py": "45f082290593e9d7af7bf899faba8221c1be2d9f2534b2b48855767cfdaa5482",
    "raw_calibration.py": "95d7085f257b0285d58ee0fdc00a54f1bab31d8afd715a318194f594feb8be09",
    "ridge_baseline.py": "a0aa9487d8d66491952a8112dc00ad450b6915802b3d650dbe4c73e9ba3ad533",
    "route_catalog.py": "b016e7defcc2016372d0bd6e87eb9c041605098c9cac90886f77548465cd57db",
    "route_predictions.py": "4f90d3d9a4129070938aec199910746cc8013d6880ddda092b3cb2935c36b405",
    "training.py": "ca19a2a93b68e4fcea8d8782494915de54c86eeddd85568aab66e4174ea84f07",
}
SOURCE_SIX = {n: SOURCE18[n] for n in ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")}
CORE_FILES = tuple(SOURCE18) + ("postfit.py", "campaign.py", "resources.py", "registry_extension.py")
OWNER_KEYS = ("INFORMS_GPU_TOKEN", "INFORMS_GPU_OWNER_PID", "INFORMS_GPU_OWNER_CREATE_TIME", "INFORMS_GPU_LEASE_ID", "INFORMS_GPU_OWNER_PORT", "INFORMS_GPU_DEADLINE")
IDENTITY = ("experiment_id", "dataset", "panel", "fold", "block", "family", "num_channels", "latent_k", "seed", "length", "horizon", "requirement", "planned_epoch_endpoints")


class BridgeGuardError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""): h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _inside(root, relative):
    relative = Path(relative); root = Path(root).resolve(); path = root / relative
    if relative.is_absolute() or path.resolve() == root or not path.resolve().is_relative_to(root):
        raise BridgeGuardError("OUTSIDE_DECLARED_ROOT")
    for part in (path, *path.parents):
        if part == root: break
        if part.is_symlink(): raise BridgeGuardError("SYMLINK_ARTIFACT")
    return path


def _registry(root):
    with (root / "EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fields, rows = reader.fieldnames, list(reader)
    if len({r["experiment_id"] for r in rows}) != len(rows): raise BridgeGuardError("DUPLICATE_REGISTRY_ID")
    return fields, rows


def verify_sources(root):
    actual = {name: file_sha(root / name) for name in CORE_FILES}
    if any(actual[name] != expected for name, expected in SOURCE18.items()):
        raise BridgeGuardError("FROZEN_SOURCE18_MISMATCH")
    return actual


def require_unlocked(root):
    path = root / "SELECTION_LOCK.json"
    if path.is_file():
        lock = _json(path)
        if not isinstance(lock, dict) or not isinstance(lock.get("locked"), bool) or lock["locked"]:
            raise BridgeGuardError("FINAL_SELECTION_CLOSED_OR_MALFORMED")


def _same_identity(expected, actual):
    if any(str(expected.get(k, "")) != str(actual.get(k, "")) for k in IDENTITY):
        raise BridgeGuardError("EXISTING_SCIENTIFIC_IDENTITY_CONFLICT")


def _parent(row, root, asset, family, k):
    run = row["experiment_id"]; directory = root / "runs" / run
    receipt_path = directory / "BASE_FIT_RECEIPT.json"
    if not receipt_path.is_file(): return None
    receipt = _json(receipt_path)
    if receipt.get("status") != "COMPLETE_VALID": return None
    source_path = directory / "SOURCE_CONTRACT.json"; source = _json(source_path); spec = source["spec"]
    a50 = asset["primary_split_ranges"]["A_fit"][1]; a60 = asset["primary_split_ranges"]["A_stop"][1]
    if (row["family"] != family or row["dataset"] != spec.get("dataset") or row["fold"] != "primary" or row["block"] != "E1"
            or int(row["latent_k"]) != k or int(row["seed"]) != 92601
            or (int(row["num_channels"]), int(row["length"]), int(row["horizon"])) != (asset["shape"][1], asset["L"], asset["H"])
            or spec.get("run_id") != run or spec.get("family") != family or spec.get("fold", "primary") != "primary"
            or spec.get("role", "base_parent") != "base_parent" or int(spec.get("seed", -1)) != 92601
            or int(spec.get("latent_k", -1)) != (k or asset["shape"][1]) or float(spec.get("lr", 0)) != .001
            or receipt.get("run_id") != run or receipt.get("family") != family
            or receipt.get("search_fit_end") != a50 or receipt.get("refit_fit_end") != a60
            or receipt.get("normalization_fit_end") != a60 or a60 != asset["primary_split_ranges"]["M_cal"][0]):
        raise BridgeGuardError("PARENT_NOT_EXACT_PRIMARY_A_ONLY_A60")
    if source.get("files") != SOURCE_SIX or source.get("source_sha") != fingerprint(SOURCE_SIX) or receipt.get("source_sha") != source["source_sha"]:
        raise BridgeGuardError("PARENT_SCIENTIFIC_SOURCE_MISMATCH")
    if source.get("data_sha") != asset["data_sha256"]:
        raise BridgeGuardError("PARENT_DATA_IDENTITY_MISMATCH")
    epochs = [float(e) for e in receipt.get("epochs", [])]
    if not epochs or epochs != [float(e) for e in spec.get("epochs", [])] or epochs != sorted(set(epochs)):
        raise BridgeGuardError("PARENT_REQUESTED_ENDPOINT_IDENTITY_MISMATCH")
    endpoint = max(epochs)
    if endpoint not in (.25, 1., 3.): return None
    relative = f"runs/{run}/selected_bundle.pt"
    if receipt.get("artifact_cache_relative_path") != relative or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("artifact_sha256"))) is None:
        raise BridgeGuardError("PARENT_SELECTED_ARTIFACT_IDENTITY_INVALID")
    return dict(run_id=run, family=family, endpoint=endpoint, selected_epoch=receipt.get("selected_epoch"),
        artifact_cache_relative_path=relative, artifact_sha256=receipt["artifact_sha256"],
        receipt_relative_path=receipt_path.relative_to(root).as_posix(), receipt_sha256=file_sha(receipt_path),
        source_relative_path=source_path.relative_to(root).as_posix(), source_contract_sha256=file_sha(source_path),
        data_sha=source["data_sha"], foundation_model_sha=source.get("model_sha"), a60_fit_end=a60,
        refit_scaler_sha=receipt.get("refit_scaler_sha"), registry_template=row)


def _routes(dataset, endpoint, parents):
    template = parents["B2"]["registry_template"]; q = template["latent_k"]
    binding = {f: {k: v[k] for k in ("run_id", "artifact_sha256", "source_contract_sha256")} for f, v in parents.items()}
    suffix = "__xp_e" + str(int(endpoint)) + "_" + fingerprint(binding)[:12]
    descriptions = (("CONDITIONAL", "conditional", "B2", q), ("CONDITIONAL", "conditional", "B0", q),
                    ("PAST_CONTROL", "past_control", "DIRECT_G", "0"), ("ALPHA_FIT", "alpha_family", "B2", q))
    routes = []
    for family, variant, parent_family, k in descriptions:
        label = "G" if parent_family == "DIRECT_G" else parent_family
        run = f"primary_{dataset}_E2_{family}_k{k}_s92601_{label}" + suffix
        spec = dict(run_id=run, variant=variant, seed=92601, device="cuda", learning_rate=.001,
            epoch_endpoints=[.25, 1., 3.], effective_batch=16, microbatch=4,
            parent_run_id=parents[parent_family]["run_id"], g_run_id=parents["DIRECT_G"]["run_id"], fold="primary")
        aliases = [(family, run)]
        if variant == "alpha_family":
            aliases = [(f, run.replace("_ALPHA_FIT_", "_" + f + "_")) for f in ("ALPHA0", "ALPHA05", "ALPHA1", "ALPHA_FIT")]
        rows = [dict(template, experiment_id=alias, block="E2", family=f, latent_k=str(k),
            requirement="REGISTERED_EXTENDED_PARENT_E2", execution_status="PENDING_QUEUE", training_status="NOT_STARTED",
            validity_status="NOT_CHECKED", scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED",
            planned_epoch_endpoints="0.25|1.0|3.0", reason="Frozen A60 parents b=" + spec["parent_run_id"] + "; g=" + spec["g_run_id"] + "; requested parent endpoint=" + str(endpoint) + "; no parent M training; exact SHA binding in bridge plan/spec",
            resume_command="python extended_parent_e2.py --execute --plan-json <SAVED_PLAN_JSON>") for f, alias in aliases]
        routes.append(dict(spec=spec, registry_rows=rows, parent_family=parent_family))
    return routes


def build_plan(public_root, cache_root=None, datasets=None):
    root = Path(public_root).resolve(); sources = verify_sources(root); _, rows = _registry(root)
    registry = {r["experiment_id"]: r for r in rows}; assets = _json(root / "DATA_CONTRACT.json")["assets"]
    datasets = list(datasets or MAIN)
    if len(set(datasets)) != len(datasets) or any(ds not in MAIN for ds in datasets): raise BridgeGuardError("MAIN4_DATASETS_REQUIRED")
    planned, proposed = [], []
    for ds in datasets:
        asset = assets.get(ds)
        if not asset or not asset.get("primary_split_ranges"):
            planned.append(dict(dataset=ds, disposition="DEFERRED_DATA_METADATA", routes=[])); continue
        q = (int(asset["shape"][1]) + 3) // 4; candidates = {}; omissions = []
        for family, k in (("DIRECT_G", 0), ("B0", q), ("B2", q)):
            base = f"primary_{ds}_E1_{family}_k{k}_s92601"
            candidates[family] = {}
            for row in rows:
                if re.fullmatch(re.escape(base) + r"(?:__e[13])*", row["experiment_id"]) is None: continue
                try:
                    parent = _parent(row, root, asset, family, k)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    source_path = root/"runs"/row["experiment_id"]/"SOURCE_CONTRACT.json"
                    try: rejected_source = _json(source_path)
                    except (OSError, ValueError): rejected_source = {}
                    omissions.append(dict(run_id=row["experiment_id"], reason=str(error) if isinstance(error,BridgeGuardError) else "INVALID_PARENT_METADATA",
                        error_type=type(error).__name__, source_contract_sha256=file_sha(source_path) if source_path.is_file() else None,
                        actual_source_sha=rejected_source.get("source_sha"), actual_source_files=rejected_source.get("files"),
                        required_source_sha=fingerprint(SOURCE_SIX), required_source_files=SOURCE_SIX,
                        equivalence_to_current_source_claimed=False))
                    continue
                if parent:
                    endpoint = parent["endpoint"]
                    if endpoint in candidates[family]: raise BridgeGuardError("AMBIGUOUS_COMPLETED_PARENT_AT_ENDPOINT")
                    candidates[family][endpoint] = parent
        common = set.intersection(*(set(v) for v in candidates.values()))
        if not common:
            planned.append(dict(dataset=ds, disposition="DEFERRED_SOURCE_PARENT" if omissions else "DEFERRED_NO_COMMON_COMPLETE_PARENT",
                parent_omissions=omissions, routes=[])); continue
        endpoint = next(e for e in (3., 1., .25) if e in common)
        parents = {f: values[endpoint] for f, values in candidates.items()}
        if parents["B0"]["foundation_model_sha"] != parents["B2"]["foundation_model_sha"]:
            raise BridgeGuardError("B0_B2_FOUNDATION_IDENTITY_MISMATCH")
        scalers = {v["refit_scaler_sha"] for v in parents.values() if v["refit_scaler_sha"] is not None}
        if len(scalers) > 1: raise BridgeGuardError("PARENT_A60_SCALERS_DIFFER")
        routes = [] if endpoint == .25 else _routes(ds, endpoint, parents)
        for route in routes:
            for row in route["registry_rows"]:
                if row["experiment_id"] in registry: _same_identity(row, registry[row["experiment_id"]])
                proposed.append(row)
        planned.append(dict(dataset=ds, endpoint=endpoint, parents=parents, routes=routes,
            parent_omissions=omissions,
            disposition="REUSE_INITIAL_E2_SKIP" if endpoint == .25 else "EXTENDED_PARENT_E2_READY",
            a60_fit_end=asset["primary_split_ranges"]["A_stop"][1], data_sha=asset["data_sha256"],
            initial_E2_reused_without_new_fit=endpoint == .25, initial_E2_completion_claimed=False,
            calibration_AplusM_parent_allowed=False))
    binding = dict(sources=sources, bridge_source_sha256=file_sha(__file__), datasets=[dict(dataset=v["dataset"],
        disposition=v["disposition"], endpoint=v.get("endpoint"), parents={f: {k:p[k] for k in ("run_id", "artifact_sha256", "receipt_sha256", "source_contract_sha256")} for f,p in v.get("parents",{}).items()},
        specs=[r["spec"] for r in v["routes"]]) for v in planned])
    return dict(status="PLAN_ONLY_NOT_EXECUTED", created_unix=time.time(), metadata_only=True, data_or_model_loaded=False,
        registry_written=False, T_requested=False, datasets=planned, proposed_registry_rows=proposed,
        new_fit_count=sum(len(v["routes"]) for v in planned), new_output_rows=len(proposed), source_pins=sources,
        plan_signature=fingerprint(binding), registry_sha256=file_sha(root / "EXPERIMENT_REGISTRY.csv"),
        data_contract_sha256=file_sha(root / "DATA_CONTRACT.json"), source_six=SOURCE_SIX,
        parent_artifact_bytes_read=False, framework_imports=[n for n in ("torch", "numpy", "pandas") if n in sys.modules])


def _storage(root, cache):
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root): raise BridgeGuardError("PUBLIC_PRIVATE_ROOTS_MUST_BE_DISJOINT")
    sizes = {}
    for label, path in (("public", root), ("private", cache)):
        total = 0
        for directory, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [n for n in dirs if not Path(directory, n).is_symlink()]
            for name in files:
                f = Path(directory, name)
                if not f.is_symlink(): total += f.stat().st_size
        sizes[label + "_bytes"] = total
        if shutil.disk_usage(path).free < PENDING_BYTES: raise BridgeGuardError("DEFERRED_FREE_SPACE")
    if sum(sizes.values()) + PENDING_BYTES > STORAGE_BYTES: raise BridgeGuardError("DEFERRED_COMBINED_STORAGE")
    return dict(sizes, pending_bytes=PENDING_BYTES, combined_limit_bytes=STORAGE_BYTES)


def _load_resources():
    return importlib.import_module("resources"), importlib.import_module("registry_extension")


def _require_gpu_handoff():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1": raise BridgeGuardError("GPU_HANDOFF_REQUIRED")


def _load_runtime(root, cache, assets):
    _require_gpu_handoff()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[key] = "2"
    os.environ["INFORMS_ASSET_ROOT"], os.environ["INFORMS_CACHE_ROOT"] = str(assets), str(cache)
    postfit = importlib.import_module("postfit")
    if Path(postfit.HERE).resolve() != root or Path(postfit.CACHE).resolve() != cache or Path(postfit.ASSET_ROOT).resolve() != assets:
        raise BridgeGuardError("IMPORTED_RUNTIME_ROOTS_DIFFER")
    postfit.torch.set_num_threads(2)
    return SimpleNamespace(postfit=postfit, data=importlib.import_module("data_contract"),
        combination=importlib.import_module("combination"), audit=importlib.import_module("audit_campaign"))


def _update_status(root, expected, **values):
    if set(values) - {"execution_status", "training_status", "validity_status", "reason"}: raise BridgeGuardError("MUTABLE_STATUS_FIELDS_ONLY")
    fields, rows = _registry(root); found = [r for r in rows if r["experiment_id"] == expected["experiment_id"]]
    if len(found) != 1: raise BridgeGuardError("REGISTERED_ALIAS_REQUIRED")
    _same_identity(expected, found[0]); found[0].update({k:str(v) for k,v in values.items()})
    with (root / "EXPERIMENT_REGISTRY.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _guard(root, cache, sources, lease, reg):
    lease.check()
    if time.time() >= lease.deadline - 5: raise TimeoutError("FINALIZATION_HEADROOM")
    require_unlocked(root); reg.require_pre_selection(root, "extended-parent E2 fitting")
    if verify_sources(root) != sources: raise BridgeGuardError("RUNTIME_SOURCE_CHANGED")
    _storage(root, cache)
    lease.check()
    if time.time() >= lease.deadline - 5: raise TimeoutError("FINALIZATION_HEADROOM")


def _cache_metadata(value, parent, split, dense, item):
    meta = value["metadata"]
    if (meta.get("run_id") != parent["run_id"] or meta.get("artifact_sha256") != parent["artifact_sha256"]
            or meta.get("role") != "base_parent" or meta.get("fold") != "primary" or meta.get("family") != parent["family"]
            or meta.get("split") != split or meta.get("dense") is not dense or meta.get("data_sha256") != item["data_sha"]
            or meta.get("parent_fit_end") != item["a60_fit_end"] or meta.get("canonical_scaler_fit_end") != item["a60_fit_end"]
            or not meta.get("model_sha") or meta.get("cohort") != item["dataset"]+":primary:"+split+":stride"+str(1 if dense else item["parents"]["DIRECT_G"]["registry_template"]["horizon"])):
        raise BridgeGuardError("PREDICTION_PARENT_NOT_PINNED_A_ONLY_A60")
    if any(meta.get("source_files", {}).get(name) != digest for name, digest in SOURCE_SIX.items()):
        raise BridgeGuardError("PREDICTION_PARENT_SOURCE_SIX_MISMATCH")
    if parent["refit_scaler_sha"] is not None and meta.get("scaler_sha") != parent["refit_scaler_sha"]:
        raise BridgeGuardError("PREDICTION_PARENT_SCALER_MISMATCH")


def _preflight_output(route, root, cache):
    run = route["spec"]["run_id"]; public = root / "combinations" / run; private = cache / "combinations" / run
    receipt = public / "receipt.json"
    if receipt.is_file():
        value = _json(receipt)
        if value.get("run_id") != run or value.get("execution_status") != "COMPLETE_VALID":
            raise BridgeGuardError("PARTIAL_COMBINATION_PRESERVED")
        if _json(private / "spec.json") != route["spec"]: raise BridgeGuardError("EXISTING_SPEC_IDENTITY_CONFLICT")
        return value
    if public.exists() or private.exists(): raise BridgeGuardError("PARTIAL_COMBINATION_PRESERVED")
    return None


def _resource_error(error, resources=None):
    return (isinstance(error, TimeoutError) or
            resources is not None and isinstance(error, (resources.BudgetExceeded, resources.ResourceBusy)) or
            isinstance(error, BridgeGuardError) and str(error).startswith("DEFERRED_"))


def execute(public_root, cache_root, asset_root, reserve_seconds=6000, *, plan=None, datasets=None, plan_path=None):
    root, cache, assets = (Path(v).resolve() for v in (public_root, cache_root, asset_root))
    stamp = str(time.time_ns()) + "_" + uuid.uuid4().hex[:8]
    path = root / "execution_batches" / ("extended_parent_e2_" + stamp + ".json")
    result = dict(status="EXECUTION_IN_PROGRESS", started_unix=time.time(), routes=[], parent_predictions=[],
        fit_calls_attempted_now=0, completed_new_fits=0, completed_new_aliases=0, complete_fits_reused=0,
        new_alpha_family_fits=0, fit_receipts_returned_now=0, reused_standard_aliases=0,
        resource_status="NOT_ACQUIRED", resource_events=[],
        runtime_import_attempted=False, T_requested=False, root_scores_written=False,
        batch_receipt_relative_path=path.relative_to(root).as_posix())
    lease = None; lease_id = None; sources = None; resources = None; registered = False
    try:
        require_unlocked(root)
        if any(os.environ.get(k) for k in OWNER_KEYS): raise BridgeGuardError("NO_NESTED_GPU_RESERVATION")
        if not 0 <= reserve_seconds < TOTAL_SECONDS: raise BridgeGuardError("INVALID_GPU_RESERVE")
        if plan_path is not None: plan = _json(plan_path)
        if datasets and plan and {v["dataset"] for v in plan["datasets"]} != set(datasets):
            raise BridgeGuardError("DATASET_PLAN_DIFFERS")
        current = build_plan(root, cache, datasets=[d["dataset"] for d in plan["datasets"]] if plan else datasets)
        if plan and current["plan_signature"] != plan["plan_signature"]: raise BridgeGuardError("PINNED_PLAN_IDENTITY_CHANGED")
        plan = current; sources = current["source_pins"]; result.update(plan=plan, source_files_before=sources)
        if not plan["new_fit_count"]:
            result["status"] = "NO_NEW_EXTENDED_PARENTS_REUSE_INITIAL_E2" if all(
                v["disposition"]=="REUSE_INITIAL_E2_SKIP" for v in plan["datasets"]) else "DEFERRED_SOURCE_OR_PARENT_METADATA"
            return result
        _require_gpu_handoff()
        result["storage_before"] = _storage(root, cache)
        resources, reg = _load_resources()
        reg.require_pre_selection(root, "extended-parent E2 registration")
        lease = resources.GPUReservation(cache, total_gpu_seconds=TOTAL_SECONDS, reserve_seconds=reserve_seconds,
            wall_deadline=WALL_DEADLINE, purpose="training", kind="extended_parent_e2")
        with lease:
            lease_id = lease.lease_id; result.update(lease_id=lease_id, lease_deadline_unix=lease.deadline)
            result["resource_status"] = "WITHIN_LIMITS"
            _guard(root, cache, sources, lease, reg)
            for item in plan["datasets"]:
                for parent in item.get("parents", {}).values():
                    if file_sha(_inside(cache, parent["artifact_cache_relative_path"])) != parent["artifact_sha256"]:
                        raise BridgeGuardError("ACTUAL_PARENT_ARTIFACT_SHA_CHANGED")
            additions = reg.register(root, plan["proposed_registry_rows"], "Extended A-only parents and explicit SHA-bound E2 routes; registered before all new fit; no T")
            registered = True; result["new_registry_rows"] = len(additions)
            result["runtime_import_attempted"] = True; runtime = _load_runtime(root, cache, assets)
            for item in plan["datasets"]:
                if not item["routes"]: continue
                pool = None; task = None; predictions = {}; group_error = None
                try:
                    _guard(root, cache, sources, lease, reg)
                    task = runtime.data.load_task(item["dataset"])
                    if task["data_sha256"] != item["data_sha"]: raise BridgeGuardError("LOADED_TASK_DATA_SHA_CHANGED")
                    template = item["parents"]["DIRECT_G"]["registry_template"]
                    if (task["L"], task["H"], len(task["columns"])) != tuple(int(template[k]) for k in ("length", "horizon", "num_channels")):
                        raise BridgeGuardError("LOADED_TASK_L_H_C_CHANGED")
                    task = dict(task, split_ranges=runtime.data.split_ranges(task, "primary"))
                    if task["split_ranges"]["A_stop"][1] != item["a60_fit_end"]: raise BridgeGuardError("LOADED_TASK_A60_CHANGED")
                    mean, std, _ = runtime.data.fit_statistics(task, item["a60_fit_end"])
                    pool = runtime.postfit.ParentPool(task, fold="primary", deadline=lease.deadline-5)
                except Exception as error: group_error = error
                try:
                    for route in item["routes"]:
                        spec = route["spec"]; entry = dict(run_id=spec["run_id"], dataset=item["dataset"], variant=spec["variant"],
                            parent_run_id=spec["parent_run_id"], g_run_id=spec["g_run_id"], fit_call_attempted_now=False,
                            training_executed_now=False, executed_now=False, reused_complete_fit=False, status="PENDING",
                            numeric_status="NO_FIT_RESULT", standard_outputs_complete=False, resource_status="WITHIN_LIMITS")
                        result["routes"].append(entry)
                        try:
                            if group_error is not None: raise group_error
                            _guard(root, cache, sources, lease, reg)
                            existing = _preflight_output(route, root, cache)
                            for family in ("DIRECT_G", route["parent_family"]):
                                parent = item["parents"][family]
                                for split, dense in (("M_cal", True), ("V_select", False)):
                                    key = (parent["run_id"], split)
                                    if key not in predictions:
                                        _guard(root, cache, sources, lease, reg)
                                        entry["phase"] = "PARENT_PREDICT"
                                        value = pool.predict(parent["run_id"], split, dense)
                                        result["parent_predictions"].append(dict(run_id=parent["run_id"], split=split,
                                            reused_prediction_cache=value["metadata"].get("reused_prediction_cache")))
                                        entry["phase"] = "AFTER_PARENT_PREDICT"
                                        _guard(root, cache, sources, lease, reg)
                                        _cache_metadata(value, parent, split, dense, item); predictions[key] = value
                            bid, gid = spec["parent_run_id"], spec["g_run_id"]
                            bm, bv, gm, gv = (predictions[(bid,"M_cal")], predictions[(bid,"V_select")], predictions[(gid,"M_cal")], predictions[(gid,"V_select")])
                            if bm["metadata"].get("model_sha") != bv["metadata"].get("model_sha") or gm["metadata"].get("model_sha") != gv["metadata"].get("model_sha"):
                                raise BridgeGuardError("M_V_PARENT_MODEL_IDENTITY_CHANGED")
                            entry["phase"] = "BEFORE_PREPARE"
                            _guard(root, cache, sources, lease, reg)
                            refs = runtime.audit.prepare_combination_reference(spec["run_id"], task, bv, gv, mean, std, cache, root, spec=spec)
                            entry["phase"] = "AFTER_PREPARE"
                            _guard(root, cache, sources, lease, reg)
                            entry["phase"] = "BEFORE_FIT_OR_REUSE"
                            _guard(root, cache, sources, lease, reg)
                            if existing is None:
                                entry.update(fit_call_attempted_now=True, training_executed_now=None, executed_now=True)
                                result["fit_calls_attempted_now"] += 1
                                entry["phase"] = "FIT"
                                receipt = runtime.combination.fit_combination(spec, task, bm, bv, gm, gv, mean, std, cache, root/"combinations", lease.deadline-5)
                                result["fit_receipts_returned_now"] += 1
                                entry["training_executed_now"] = bool(receipt.get("updates",0))
                            else:
                                runtime.combination.validate_completed_receipt(existing, spec, task, bm, bv, gm, gv, mean, std, cache)
                                receipt = existing; entry["reused_complete_fit"] = True
                            entry.update(status=receipt["execution_status"], numeric_status=receipt["execution_status"],
                                training_status=receipt["training_status"], updates=receipt.get("updates"))
                            receipt_path = root/"combinations"/spec["run_id"]/"receipt.json"
                            entry.update(fit_receipt_relative_path=receipt_path.relative_to(root).as_posix(), fit_receipt_sha256=file_sha(receipt_path))
                            if receipt["execution_status"] == "COMPLETE_VALID":
                                if existing is None:
                                    result["completed_new_fits"] += 1
                                    result["new_alpha_family_fits"] += int(spec["variant"] == "alpha_family")
                                else: result["complete_fits_reused"] += 1
                            entry["phase"] = "AFTER_FIT_OR_REUSE"
                            _guard(root, cache, sources, lease, reg)
                            if receipt["execution_status"] == "COMPLETE_VALID":
                                existing_aliases = {row["experiment_id"] for row in route["registry_rows"]
                                    if (root/"predictions"/row["experiment_id"]/"V_select/PREDICTION_MANIFEST.json").is_file()}
                                entry["phase"] = "BEFORE_WRAP"
                                _guard(root, cache, sources, lease, reg)
                                runtime.audit.wrap_combination_output(receipt, refs, cache, root)
                                entry["standard_manifests"] = []
                                for row in route["registry_rows"]:
                                    f = root/"predictions"/row["experiment_id"]/"V_select/PREDICTION_MANIFEST.json"
                                    entry["standard_manifests"].append(dict(file=f.relative_to(root).as_posix(),sha256=file_sha(f)))
                                result["completed_new_aliases"] += len(route["registry_rows"])-len(existing_aliases)
                                result["reused_standard_aliases"] += len(existing_aliases)
                                entry.update(standard_outputs_complete=True, phase="AFTER_WRAP")
                                _guard(root, cache, sources, lease, reg)
                            else: entry["status"] = "DEFERRED_RESOURCE" if receipt["execution_status"] == "BUDGET_LIMIT" else receipt["execution_status"]
                            for row in route["registry_rows"]:
                                _update_status(root, row, execution_status=receipt["execution_status"], training_status=receipt["training_status"],
                                    validity_status="PENDING_INDEPENDENT_VALIDATION" if receipt["execution_status"]=="COMPLETE_VALID" else "NOT_COMPLETE",
                                    reason="Extended A-only parent E2; exact frozen parents/spec in immutable receipt; no T")
                            if entry["status"] == "DEFERRED_RESOURCE": raise TimeoutError("COMBINATION_BUDGET_LIMIT")
                        except Exception as error:
                            deferred = _resource_error(error, resources)
                            entry.update(status="DEFERRED_RESOURCE" if deferred else "EXECUTION_FAILED", error_type=type(error).__name__)
                            if deferred:
                                entry["resource_status"] = "DEFERRED_RESOURCE"
                                result["resource_events"].append(dict(run_id=entry["run_id"], phase=entry.get("phase"), error_type=type(error).__name__))
                            _save_failure(cache, stamp, entry, error)
                            for row in route["registry_rows"]:
                                values = dict(execution_status=entry["status"], validity_status="PENDING_INDEPENDENT_VALIDATION" if entry["standard_outputs_complete"] else "NOT_COMPLETE",
                                    reason=type(error).__name__+"; preserved state; see relative bridge batch receipt")
                                if entry["fit_call_attempted_now"] and entry["training_executed_now"] is None:
                                    values["training_status"] = "EXECUTION_UNKNOWN_PARTIAL_PRESERVED"
                                elif entry.get("training_status") is not None: values["training_status"] = entry["training_status"]
                                _update_status(root, row, **values)
                            if deferred: raise
                finally:
                    if pool is not None: pool.close()
            _guard(root, cache, sources, lease, reg)
            if any(v["disposition"].startswith("DEFERRED_") for v in plan["datasets"]):
                result["status"] = "PARTIAL_WITH_DEFERRED_DATASETS"
            else:
                result["status"] = "COMPLETE_REQUESTED_E2_ROUTES" if all(r["status"]=="COMPLETE_VALID" for r in result["routes"]) else "PARTIAL_WITH_PRESERVED_FAILURES"
    except Exception as error:
        result.update(status="DEFERRED_RESOURCE" if _resource_error(error, resources) else "EXECUTION_ABORTED", fatal_error_type=type(error).__name__)
        if _resource_error(error, resources):
            result["resource_status"] = "DEFERRED_RESOURCE"
            if not result["resource_events"]: result["resource_events"].append(dict(phase="BATCH", error_type=type(error).__name__))
        _save_failure(cache, stamp, result, error)
        if result["status"] == "DEFERRED_RESOURCE" and plan:
            seen = {r["run_id"] for r in result["routes"]}
            for item in plan["datasets"]:
                for route in item["routes"]:
                    if route["spec"]["run_id"] in seen: continue
                    result["routes"].append(dict(run_id=route["spec"]["run_id"], dataset=item["dataset"],
                        variant=route["spec"]["variant"], status="DEFERRED_RESOURCE", fit_call_attempted_now=False,
                        training_executed_now=False, executed_now=False, reused_complete_fit=False,
                        numeric_status="NO_FIT_RESULT", standard_outputs_complete=False, resource_status="DEFERRED_RESOURCE"))
                    if registered:
                        for row in route["registry_rows"]:
                            _update_status(root,row,execution_status="DEFERRED_RESOURCE",reason="Earlier route/resource boundary; no fit attempted by this batch")
    finally:
        try:
            result["source_files_after"] = verify_sources(root)
            result["source_pins_unchanged"] = sources is not None and result["source_files_after"] == sources
            if sources is not None and not result["source_pins_unchanged"]: result["status"] = "EXECUTION_ABORTED_SOURCE_CHANGE"
        except Exception as error:
            result.update(source_pins_unchanged=False, final_source_error_type=type(error).__name__)
            if sources is not None: result["status"] = "EXECUTION_ABORTED_SOURCE_CHANGE"
        try:
            charge = cache/"gpu_receipts"/(str(lease_id)+".json")
            result["central_lease_receipt_exists"] = lease_id is not None and charge.is_file()
            if result["central_lease_receipt_exists"]:
                result.update(lease_receipt_cache_relative_path=charge.relative_to(cache).as_posix(), lease_receipt_sha256=file_sha(charge))
                closed = _json(charge)
                charged, declared_overrun, finished, deadline = (float(closed[k]) for k in
                    ("charged_process_seconds", "deadline_overrun_seconds", "finished_unix", "deadline"))
                if (closed.get("lease_id") != lease_id or closed.get("status") not in {"CLOSED", "FAILED"}
                        or deadline != result["lease_deadline_unix"] or not all(math.isfinite(v) for v in (charged, declared_overrun, finished, deadline))
                        or min(charged, declared_overrun) < 0):
                    raise BridgeGuardError("CLOSED_LEASE_RECEIPT_IDENTITY_INVALID")
                overrun = max(declared_overrun, finished-deadline, 0.)
                result.update(lease_receipt_verified=True, central_lease_status=closed["status"],
                    charged_process_seconds=charged, deadline_overrun_seconds=overrun, lease_finished_unix=finished)
                if overrun > 0:
                    result["resource_status"] = "LIMIT_VIOLATION"
                    result["resource_events"].append(dict(phase="CLOSED_LEASE", kind="DEADLINE_OVERRUN", seconds=overrun))
            elif lease_id is not None:
                result["resource_status"] = "UNVERIFIED_CLOSED_LEASE"
        except Exception as error:
            result.update(lease_receipt_read_error_type=type(error).__name__, lease_receipt_verified=False,
                resource_status="UNVERIFIED_CLOSED_LEASE")
        if result["resource_status"] in {"LIMIT_VIOLATION", "UNVERIFIED_CLOSED_LEASE"} and result["status"].startswith("COMPLETE_"):
            result["status"] = "RESOURCE_LIMIT_VIOLATION" if result["resource_status"]=="LIMIT_VIOLATION" else "RESOURCE_ACCOUNTING_UNVERIFIED"
        planned_fits = result.get("plan", {}).get("new_fit_count", 0)
        planned_aliases = result.get("plan", {}).get("new_output_rows", 0)
        result["numeric_completion"] = dict(
            all_planned_fits_complete=bool(planned_fits) and result["completed_new_fits"]+result["complete_fits_reused"]==planned_fits,
            all_planned_standard_outputs_complete=bool(planned_aliases) and result["completed_new_aliases"]+result["reused_standard_aliases"]==planned_aliases)
        result.update(finished_unix=time.time(), source_six=SOURCE_SIX,
            unresolved_training_attempts=sum(r["fit_call_attempted_now"] and r["training_executed_now"] is None for r in result["routes"]))
        _new_json(path, result)
    return result


def _save_failure(cache, stamp, entry, error):
    try:
        path = cache/"extended_parent_e2_failures"/(stamp+"_"+uuid.uuid4().hex[:8]+".json")
        _new_json(path, dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        entry.update(private_failure_cache_relative_path=path.relative_to(cache).as_posix(), private_failure_sha256=file_sha(path))
    except Exception as failure:
        entry["private_failure_write_error_type"] = type(failure).__name__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=HERE)
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("INFORMS_CACHE_ROOT", HERE.parents[1]/".cache"/HERE.name)))
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("INFORMS_ASSET_ROOT", HERE.parents[1])))
    parser.add_argument("--dataset", action="append", choices=MAIN)
    parser.add_argument("--reserve-seconds", type=float, default=6000)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        result = execute(args.public_root,args.cache_root,args.asset_root,args.reserve_seconds,
            datasets=args.dataset,plan_path=args.plan_json)
    else:
        result = build_plan(args.public_root,args.cache_root,args.dataset)
        if args.plan_json and _json(args.plan_json)["plan_signature"] != result["plan_signature"]:
            raise BridgeGuardError("PINNED_PLAN_IDENTITY_CHANGED")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__=="__main__": main()
