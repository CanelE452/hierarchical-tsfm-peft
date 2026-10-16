"""Predeclared pending-grid bridge; dry run uses only standard-library metadata."""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
TOTAL_GPU_SECONDS = 43200
WALL_DEADLINE = 1792050350 + 172800
DEFAULT_RESERVE_SECONDS = 6000
STORAGE_LIMIT = 30 * 1024**3
PENDING_BYTES = 512 * 1024**2
SMALL_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
SOURCE_PINS = {
    "fit_base.py": "c59cc77ba48e3f061ffa8d2385442286ea989b3ab86c17ea352caeb6d2c1bf20",
    "training.py": "ca19a2a93b68e4fcea8d8782494915de54c86eeddd85568aab66e4174ea84f07",
    "networks.py": "787f9caf91f32a001f929f324c7bcb570b9270331f6f60f0f56872529f86dc54",
    "data_contract.py": "0e5fb9a03bd44fe13a9777b774601a54573f221abc0eb41a5bb41764b864b31b",
    "evaluation.py": "275cd8be02b277ceba916eb6b00e7296f46694d0d592dbc1e0fb231f349f2fbd",
    "gpu_runtime.py": "c633f1ebf4eb24ea3fa2b025ffa6cea4ec3697c8e9285d971174179560596158",
}
OWNER_ENVIRONMENT_KEYS = (
    "INFORMS_GPU_TOKEN", "INFORMS_GPU_OWNER_PID", "INFORMS_GPU_OWNER_CREATE_TIME",
    "INFORMS_GPU_LEASE_ID", "INFORMS_GPU_OWNER_PORT", "INFORMS_GPU_DEADLINE",
)
SCIENTIFIC_FIELDS = (
    "experiment_id", "dataset", "panel", "fold", "block", "family", "num_channels",
    "latent_k", "seed", "length", "horizon", "requirement", "planned_epoch_endpoints",
)
MUTABLE_FIELDS = {"execution_status", "training_status", "validity_status", "reason"}
DATASET_AXES = {"beijing": (11, 48), "jena": (21, 48), "electricity": (321, 48), "pems08": (170, 12)}
FIT_FAMILIES = {"B0", "B2", "B4", "LORA"}
SCOPES = ("all", "primary", "secondary")


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_registry(public_root):
    path = Path(public_root) / "EXPERIMENT_REGISTRY.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    ids = [row["experiment_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate registry experiment_id")
    return fields, rows


def select_pending(rows, scope="all"):
    if scope not in SCOPES:
        raise ValueError("Unsupported scope; expected all, primary, or secondary")
    selected = []
    for row in rows:
        if row["execution_status"] != "PENDING_QUEUE":
            continue
        primary = (row["fold"] == "primary" and row["block"] == "E1"
                   and row["requirement"] == "FULL_REGISTERED_GRID" and row["family"] in {"B0", "B2"})
        secondary = row["fold"] == "secondary" and row["requirement"] == "TIME_TRANSFER_REQUIRED"
        if not (primary or secondary):
            continue
        dataset, family = row["dataset"], row["family"]
        if dataset not in DATASET_AXES or (secondary and dataset not in {"beijing", "jena"}):
            raise ValueError("Unsupported registered dataset for this bounded bridge")
        if row["block"] != "E1" or (secondary and family not in FIT_FAMILIES | {"RAW"}):
            raise ValueError("Unsupported registered family/block for this bounded bridge")
        width, horizon = DATASET_AXES[dataset]
        if (int(row["num_channels"]), int(row["length"]), int(row["horizon"]), int(row["seed"])) != (width, 512, horizon, 92601):
            raise ValueError("Registered axes or seed differ from the frozen bridge contract")
        k = int(row["latent_k"])
        if family in {"RAW", "LORA"}:
            if k != 0:
                raise ValueError("RAW/LORA registry K must remain zero")
        elif not 1 <= k <= width:
            raise ValueError("Registered latent K outside original channels")
        expected_id = f"{row['fold']}_{dataset}_E1_{family}_k{k}_s92601"
        if row["experiment_id"] != expected_id:
            raise ValueError("Registered ID differs from its immutable scientific fields")
        selected.append(dict(row))
    return selected if scope == "all" else [row for row in selected if row["fold"] == scope]


def route_spec(row):
    family = row["family"]
    if family == "RAW":
        return dict(run_id=row["experiment_id"], action="fixed_RAW_V_inference", baseline="RAW",
                    registered_family="RAW", fit_family=None, fold=row["fold"], epochs=[],
                    planned_training_updates=0, foundation="pinned_local_Bolt_small")
    if family not in FIT_FAMILIES:
        raise ValueError("Unsupported family; no implicit alias or monkeypatch is permitted")
    return dict(run_id=row["experiment_id"], dataset=row["dataset"], family=family,
                latent_k=int(row["latent_k"]) or int(row["num_channels"]),
                seed=int(row["seed"]), fold=row["fold"], epochs=[.25],
                lr=.0001 if family == "LORA" else .001, device="cuda", microbatch=4,
                effective_batch=16, F_row_microbatch=128, checkpoint_inputs=False,
                lora_rank=8, lora_alpha=16,
                stage_label="registered_time_transfer_secondary" if row["fold"] == "secondary" else "registered_initial_breadth")


def source_inventory(public_root):
    root = Path(public_root)
    return {name: dict(expected_sha256=expected,
                       actual_sha256=file_sha(root / name) if (root / name).is_file() else None)
            for name, expected in SOURCE_PINS.items()}


def verify_sources(public_root):
    inventory = source_inventory(public_root)
    if any(value["actual_sha256"] != value["expected_sha256"] for value in inventory.values()):
        raise ValueError("Frozen scientific source pins differ or are absent")
    return {name: value["actual_sha256"] for name, value in inventory.items()}


def require_unlocked(public_root):
    path = Path(public_root) / "SELECTION_LOCK.json"
    if path.is_file():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict) or not isinstance(lock.get("locked"), bool):
            raise ValueError("Malformed selection lock; execution is closed")
        if lock["locked"]:
            raise ValueError("Final selection is locked; same-campaign fitting is closed")


def build_plan(public_root, cache_root, reserve_seconds=DEFAULT_RESERVE_SECONDS, scope="all"):
    root, cache = Path(public_root), Path(cache_root)
    _, rows = read_registry(root)
    chosen = select_pending(rows, scope=scope)
    sources = source_inventory(root)
    routes = []
    for row in chosen:
        run_id = row["experiment_id"]
        routes.append(dict(run_id=run_id, registered_family=row["family"], fold=row["fold"],
                           dataset=row["dataset"], registered_latent_k=int(row["latent_k"]),
                           spec=route_spec(row), prediction_split="V_select",
                           dependency="One caller-owned local Bolt-small with matching H; no trained parent dependency",
                           source_pins_match=all(v["actual_sha256"] == v["expected_sha256"] for v in sources.values()),
                           fit_receipt_exists=(root / "runs" / run_id / "BASE_FIT_RECEIPT.json").is_file(),
                           V_manifest_exists=(root / "predictions" / run_id / "V_select" / "PREDICTION_MANIFEST.json").is_file(),
                           estimated_seconds=None, actual_execution_performed=False))
    return dict(status="PLAN_ONLY_NOT_EXECUTED", scope=scope, registered_pending_routes=len(routes),
                primary_full_grid_routes=sum(row["fold"] == "primary" for row in chosen),
                secondary_transfer_routes=sum(row["fold"] == "secondary" for row in chosen),
                exact_run_ids=[row["experiment_id"] for row in chosen], routes=routes,
                source_pins=sources, model_revision=SMALL_REVISION,
                total_gpu_process_seconds=TOTAL_GPU_SECONDS, reserve_seconds=reserve_seconds,
                hard_wall_deadline_unix=WALL_DEADLINE, selection_lock_present=(root / "SELECTION_LOCK.json").is_file(),
                active_gpu_record_present=(cache / "ACTIVE_GPU_LEASE.json").is_file(),
                owner_environment_present=any(os.environ.get(key) for key in OWNER_ENVIRONMENT_KEYS),
                metadata_only=True, data_or_model_loaded=False, registry_written=False,
                runtime_or_cost_measured=False,
                limitations="A .25-epoch screen is not convergence; the plan is not learning, inference, cost measurement, validation, or publication.")


def combined_storage(public_root, cache_root):
    sizes = {}
    for label, root in (("public", Path(public_root)), ("private", Path(cache_root))):
        total = 0
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs if not Path(directory, name).is_symlink()]
            for name in files:
                path = Path(directory, name)
                if not path.is_symlink():
                    total += path.stat().st_size
        sizes[label + "_bytes"] = total
    sizes.update(pending_bytes=PENDING_BYTES, limit_bytes=STORAGE_LIMIT)
    if sum(sizes[key] for key in ("public_bytes", "private_bytes")) + PENDING_BYTES > STORAGE_LIMIT:
        raise RuntimeError("Combined public/private campaign storage budget is exhausted")
    for root in (Path(public_root), Path(cache_root)):
        if shutil.disk_usage(root).free < PENDING_BYTES:
            raise RuntimeError("Filesystem free space is below the pending 512 MiB allowance")
    return sizes


def _within(root, relative):
    relative = Path(relative)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Existing receipt points outside the private campaign cache")
    return resolved


def preflight_route(row, public_root, cache_root):
    root, cache = Path(public_root), Path(cache_root)
    run_id = row["experiment_id"]
    receipt_path = root / "runs" / run_id / "BASE_FIT_RECEIPT.json"
    public_fit, private_fit = root / "runs" / run_id, cache / "runs" / run_id
    receipt = None
    if receipt_path.is_file():
        if row["family"] == "RAW":
            raise ValueError("Fixed RAW route has an unexpected fitted artifact")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source = json.loads((public_fit / "SOURCE_CONTRACT.json").read_text(encoding="utf-8"))
        if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id
                or source.get("spec") != route_spec(row) or source.get("files") != SOURCE_PINS):
            raise ValueError("Existing complete fit source/config/identity differs; preserve it")
        artifact = _within(cache, receipt["artifact_cache_relative_path"])
        if not artifact.is_file() or file_sha(artifact) != receipt["artifact_sha256"]:
            raise ValueError("Existing selected artifact is absent or its SHA differs")
    elif public_fit.exists() or private_fit.exists():
        raise ValueError("Existing partial fit is preserved; this bridge will not resume it")
    public_v = root / "predictions" / run_id / "V_select"
    private_v = cache / "predictions" / run_id / "V_select.npz"
    if public_v.exists() or private_v.exists():
        if row["family"] != "RAW" and receipt is None:
            raise ValueError("Existing learned-route V has no verified parent receipt; preserve it")
        required = ("REFERENCE_CONTRACT.json", "PREDICTION_MANIFEST.json", "PUBLIC_ERROR_STATS.csv.gz", "SCORES.csv")
        if not private_v.is_file() or not all((public_v / name).is_file() for name in required):
            raise ValueError("Existing partial V output is preserved; this bridge will not overwrite it")
    return receipt


def update_status(public_root, expected_row, **values):
    if set(values) - MUTABLE_FIELDS:
        raise ValueError("Only mutable execution/training/validity/reason fields may change")
    path = Path(public_root) / "EXPERIMENT_REGISTRY.csv"
    original = path.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode("utf-8"), newline=""))
    fields, rows = reader.fieldnames, list(reader)
    matched = [row for row in rows if row["experiment_id"] == expected_row["experiment_id"]]
    if len(matched) != 1 or any(matched[0].get(key) != expected_row.get(key) for key in SCIENTIFIC_FIELDS):
        raise ValueError("Current registry scientific identity differs; no update performed")
    matched[0].update({key: str(value) for key, value in values.items()})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if path.read_bytes() != original:
        raise RuntimeError("Registry changed concurrently; preserve the other writer")
    temporary = path.with_name(path.name + ".remaining_breadth_" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(output.getvalue())
    if path.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError("Registry changed concurrently; preserve the other writer")
    os.replace(temporary, path)


def _new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _receipt_summary(receipt):
    keys = ("status", "training_status", "selected_epoch", "selected_astop_mse", "search_updates",
            "refit_updates", "search_exposures", "refit_exposures", "F_logical_streams",
            "artifact_cache_relative_path", "artifact_sha256", "model_state_sha", "source_sha",
            "last_cp_best", "last_scores_still_falling", "reused_complete_artifact", "updates",
            "exposures", "phase", "completed_epochs", "actual_epochs", "F_total_logical_streams")
    return {key: receipt[key] for key in keys if key in receipt}


def deadline_timeout(error, deadline, now=None):
    return isinstance(error, TimeoutError) and (time.time() if now is None else now) >= deadline


def finish_execution_result(result, public_root, cache_root, lease_id, receipt_path):
    root, cache = Path(public_root), Path(cache_root)
    try:
        inventory = source_inventory(root)
        result["source_files_after"] = {name: value["actual_sha256"] for name, value in inventory.items()}
        result["source_six_unchanged"] = result["source_files_before"] == result["source_files_after"]
    except Exception as error:
        result["source_six_unchanged"] = False
        result["source_verification_error_type"] = type(error).__name__
    lease_receipt = cache / "gpu_receipts" / (lease_id + ".json")
    result["central_lease_receipt_exists"] = lease_receipt.is_file()
    if lease_receipt.is_file():
        result["lease_receipt_cache_relative_path"] = lease_receipt.relative_to(cache).as_posix()
        result["lease_receipt_sha256"] = file_sha(lease_receipt)
    result["finished_unix"] = time.time()
    result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
    result["fit_calls_with_unresolved_execution"] = sum(
        bool(entry.get("fit_call_attempted_now")) and entry.get("training_executed_now") is None
        for entry in result["routes"])
    if result.get("fatal_error_type"):
        result["status"] = "EXECUTION_ABORTED_WITH_PRESERVED_STATE"
    elif not result["source_six_unchanged"]:
        result["status"] = "EXECUTION_ABORTED_SOURCE_CHANGE"
    else:
        result["status"] = "COMPLETE_REQUESTED_PENDING_ROUTES" if result["routes"] and all(
            entry["status"] == "COMPLETE_VALID" for entry in result["routes"]
        ) else "PARTIAL_OR_DEFERRED_ROUTES"
    _new_json(receipt_path, result)


def execute(public_root, cache_root, asset_root, reserve_seconds=DEFAULT_RESERVE_SECONDS, scope="all"):
    if scope not in SCOPES:
        raise ValueError("Unsupported scope; expected all, primary, or secondary")
    root, cache, assets = Path(public_root).resolve(), Path(cache_root).resolve(), Path(asset_root).resolve()
    require_unlocked(root)
    if any(os.environ.get(key) for key in OWNER_ENVIRONMENT_KEYS):
        raise ValueError("Standalone GPU ownership required; do not nest a reservation in an owner or delegated child")
    if not 0 <= reserve_seconds < TOTAL_GPU_SECONDS:
        raise ValueError("Reserve must be nonnegative and below the total GPU process-wall limit")
    source_before = verify_sources(root)
    storage_before = combined_storage(root, cache)
    _, registry_rows = read_registry(root)
    candidates = select_pending(registry_rows, scope=scope)
    plan = build_plan(root, cache, reserve_seconds, scope=scope)
    if not candidates:
        return dict(status="NO_PENDING_ROUTES", actual_training_runs=0, actual_V_predictions=0, plan=plan)
    # Imports and any CUDA use are execution-only, after the stdlib gates.
    from resources import BudgetExceeded, GPUReservation
    import registry_extension
    registry_extension.require_pre_selection(root, "remaining registered breadth execution")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
        raise ValueError("Execution requires the handed-off GPU; the CPU-only environment is plan-only")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["INFORMS_ASSET_ROOT"], os.environ["INFORMS_CACHE_ROOT"] = str(assets), str(cache)
    stamp = str(time.time_ns())
    private_batch = cache / ("remaining_breadth_" + stamp)
    public_receipt = root / "execution_batches" / ("remaining_breadth_" + stamp + ".json")
    result = dict(status="EXECUTION_IN_PROGRESS", started_unix=time.time(), plan=plan,
                  source_files_before=source_before, storage_before=storage_before,
                  resource_scope="Single exclusive GPU process-wall lease, including loading/fitting/V inference",
                  actual_fit_calls=0, actual_training_runs=0, reused_training_artifacts=0, actual_V_predictions=0,
                  reused_V_predictions=0, routes=[], T_requested=False, root_scores_written=False)
    lease = GPUReservation(cache, total_gpu_seconds=TOTAL_GPU_SECONDS, reserve_seconds=reserve_seconds,
                           wall_deadline=WALL_DEADLINE, purpose="training", kind="remaining_registered_breadth")
    try:
        with lease:
            private_batch.mkdir()
            result["lease_id"] = lease.lease_id
            result["lease_deadline_unix"] = lease.deadline
            import torch
            import data_contract
            import fit_base
            import prediction_cache
            from gpu_runtime import FrozenBolt
            torch.set_num_threads(2)
            foundation, foundation_h = None, None
            try:
                for index, row in enumerate(candidates):
                    run_id = row["experiment_id"]
                    entry = dict(run_id=run_id, dataset=row["dataset"], fold=row["fold"],
                                 registered_family=row["family"], started_unix=time.time(),
                                 training_executed_now=False, prediction_executed_now=False)
                    result["routes"].append(entry)
                    fitted, before_f = None, None
                    try:
                        lease.check()
                        if time.time() >= lease.deadline - 5:
                            raise BudgetExceeded("Five-second finalization headroom reached")
                        registry_extension.require_pre_selection(root, "remaining breadth fit/V")
                        verify_sources(root)
                        combined_storage(root, cache)
                        existing = preflight_route(row, root, cache)
                        task = data_contract.load_task(row["dataset"])
                        if (len(task["columns"]), task["L"], task["H"]) != (int(row["num_channels"]), int(row["length"]), int(row["horizon"])):
                            raise ValueError("Loaded permitted task axes differ from the registered route")
                        horizon = int(task["H"])
                        if foundation is None or foundation_h != horizon:
                            if foundation is not None:
                                foundation.assert_frozen_unchanged()
                            foundation = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            model_path = assets / ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots" / SMALL_REVISION
                            foundation = FrozenBolt(model_path, horizon, 128, False)
                            foundation_h = horizon
                        before_f = {name: int(getattr(foundation, name)) for name in
                                    ("streams", "row_calls", "physical_streams", "physical_calls")}
                        if row["family"] == "RAW":
                            entry.update(training_status="NO_TRAINING", optimizer="NONE_FIXED_BASELINE",
                                         search_updates=0, refit_updates=0, training_executed_now=False,
                                         fixed_baseline_no_training=True)
                        else:
                            registry_extension.require_pre_selection(root, "remaining breadth fit " + run_id)
                            entry["fit_call_attempted_now"] = True
                            entry["training_executed_now"] = None
                            entry["training_status"] = "PARTIAL_TRAINING_UNRESOLVED"
                            result["actual_fit_calls"] += 1
                            fitted = fit_base.fit_base(route_spec(row), task, cache, root, foundation, lease.deadline - 5)
                            entry["fit"] = _receipt_summary(fitted)
                            reused_fit = existing is not None or fitted.get("reused_complete_artifact", False)
                            known_updates = int(fitted.get("search_updates", fitted.get("updates", 0))) + int(fitted.get("refit_updates", 0))
                            entry["training_executed_now"] = not reused_fit and (known_updates > 0 or fitted.get("phase") == "refit")
                            result["actual_training_runs"] += int(entry["training_executed_now"])
                            result["reused_training_artifacts"] += int(reused_fit)
                            if fitted["status"] != "COMPLETE_VALID":
                                entry["status"] = "DEFERRED_RESOURCE" if fitted["status"] == "BUDGET_LIMIT" else fitted["status"]
                                entry["training_status"] = fitted["training_status"]
                                update_status(root, row, execution_status=entry["status"], training_status=fitted["training_status"],
                                              validity_status="INVALID_OR_NOT_COMPLETED",
                                              reason="Preserved partial fit; no V inference; actual receipt status " + fitted["status"])
                                if fitted["status"] == "BUDGET_LIMIT":
                                    raise BudgetExceeded("Fit returned the actual budget-limit status")
                                continue
                            entry["training_status"] = fitted["training_status"]
                            path = root / "runs" / run_id / "BASE_FIT_RECEIPT.json"
                            entry.update(fit_receipt_relative_path=path.relative_to(root).as_posix(), fit_receipt_sha256=file_sha(path))
                        lease.check()
                        predicted = prediction_cache.predict_parent(
                            run_id, task, "V_select", cache, root, foundation,
                            baseline="RAW" if row["family"] == "RAW" else None,
                            fold=row["fold"], deadline=lease.deadline - 5)
                        reused = bool(predicted["metadata"]["reused_prediction_cache"])
                        entry["prediction_executed_now"] = not reused
                        result["actual_V_predictions"] += int(not reused)
                        result["reused_V_predictions"] += int(reused)
                        manifest_path = root / "predictions" / run_id / "V_select" / "PREDICTION_MANIFEST.json"
                        after_f = {name: int(getattr(foundation, name)) for name in before_f}
                        entry.update(status="COMPLETE_VALID", score=predicted["score"],
                                     V_cost_record=predicted["cost_record"],
                                     F_counter_delta={name: after_f[name] - before_f[name] for name in before_f},
                                     prediction_manifest_relative_path=manifest_path.relative_to(root).as_posix(),
                                     prediction_manifest_sha256=file_sha(manifest_path))
                        update_status(root, row, execution_status="COMPLETE_VALID",
                                      training_status=entry["training_status"],
                                      validity_status="PENDING_INDEPENDENT_VALIDATION",
                                      reason="Actual full registered V prediction; fixed baseline without training" if row["family"] == "RAW"
                                      else "Actual .25 screen/A_stop selection/new A60 refit and full V; convergence unresolved")
                        print(json.dumps(dict(event="ROUTE_DONE", **entry)), flush=True)
                        del predicted, task
                    except Exception as error:
                        if isinstance(error, BudgetExceeded) or deadline_timeout(error, lease.deadline - 5):
                            entry["status"] = "DEFERRED_RESOURCE"
                            update_status(root, row, execution_status="DEFERRED_RESOURCE",
                                          training_status=entry.get("training_status", "NOT_STARTED"),
                                          validity_status="INVALID_OR_NOT_COMPLETED",
                                          reason="GPU process-wall or campaign wall deadline; existing partial files preserved")
                            for remaining in candidates[index + 1:]:
                                result["routes"].append(dict(run_id=remaining["experiment_id"], status="DEFERRED_RESOURCE",
                                                             training_executed_now=False, prediction_executed_now=False))
                                update_status(root, remaining, execution_status="DEFERRED_RESOURCE",
                                              reason="Not started: centralized remaining-breadth resource deadline reached")
                            break
                        entry.update(status="EXECUTION_FAILED", error_type=type(error).__name__)
                        if foundation is not None and before_f is not None:
                            entry["F_counter_delta"] = {name: int(getattr(foundation, name)) - before_f[name] for name in before_f}
                        error_path = private_batch / (run_id + ".traceback.txt")
                        with error_path.open("x", encoding="utf-8") as stream:
                            stream.write(traceback.format_exc())
                        entry["private_traceback_cache_relative_path"] = error_path.relative_to(cache).as_posix()
                        update_status(root, row, execution_status="EXECUTION_FAILED",
                                      training_status=entry.get("training_status", "NOT_STARTED"),
                                      validity_status="INVALID_OR_NOT_COMPLETED",
                                      reason=type(error).__name__ + "; route-local failure; original outputs preserved; independent routes continue")
                        print(json.dumps(dict(event="ROUTE_FAILED", run_id=run_id, error_type=type(error).__name__)), flush=True)
                        # Adapter construction can fail before fit_base's cleanup try.
                        # Discard this entire caller-owned foundation on every error.
                        foundation, foundation_h = None, None
                        gc.collect()
                        torch.cuda.empty_cache()
                    finally:
                        if foundation is not None and before_f is not None:
                            entry["F_counter_delta"] = {name: int(getattr(foundation, name)) - before_f[name] for name in before_f}
                        entry["finished_unix"] = time.time()
                        entry["elapsed_seconds"] = entry["finished_unix"] - entry.get("started_unix", entry["finished_unix"])
                        verify_sources(root)
                        gc.collect()
                        torch.cuda.empty_cache()
            finally:
                if foundation is not None:
                    foundation.assert_frozen_unchanged()
                foundation = None
                gc.collect()
                torch.cuda.empty_cache()
    except BudgetExceeded:
        # No lease was available: leave actual execution counts at zero.
        if not result["routes"]:
            for row in candidates:
                result["routes"].append(dict(run_id=row["experiment_id"], status="DEFERRED_RESOURCE",
                                             training_executed_now=False, prediction_executed_now=False))
                update_status(root, row, execution_status="DEFERRED_RESOURCE",
                              reason="Not started: centralized GPU or campaign wall budget exhausted")
    except Exception as error:
        result["fatal_error_type"] = type(error).__name__
        private_batch.mkdir(exist_ok=True)
        error_path = private_batch / "worker.traceback.txt"
        with error_path.open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        result["private_traceback_cache_relative_path"] = error_path.relative_to(cache).as_posix()
        already_recorded = {entry["run_id"] for entry in result["routes"]}
        result["routes"].extend(dict(run_id=row["experiment_id"], status="NOT_EXECUTED_AFTER_FATAL_ERROR",
                                     training_executed_now=False, prediction_executed_now=False)
                                for row in candidates if row["experiment_id"] not in already_recorded)
    finally:
        finish_execution_result(result, root, cache, lease.lease_id, public_receipt)
    result["public_receipt_relative_path"] = public_receipt.relative_to(root).as_posix()
    result["public_receipt_sha256"] = file_sha(public_receipt)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--public-root", type=Path, default=HERE)
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("INFORMS_ASSET_ROOT", str(HERE.parents[1]))))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--reserve-seconds", type=float, default=DEFAULT_RESERVE_SECONDS)
    parser.add_argument("--scope", choices=SCOPES, default="all")
    args = parser.parse_args(argv)
    cache = args.cache_root or Path(os.environ.get("INFORMS_CACHE_ROOT", str(args.asset_root / ".cache" / HERE.name)))
    value = execute(args.public_root, cache, args.asset_root, args.reserve_seconds, scope=args.scope) if args.execute else build_plan(args.public_root, cache, args.reserve_seconds, scope=args.scope)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if value["status"] in {"PLAN_ONLY_NOT_EXECUTED", "NO_PENDING_ROUTES", "COMPLETE_REQUESTED_PENDING_ROUTES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

