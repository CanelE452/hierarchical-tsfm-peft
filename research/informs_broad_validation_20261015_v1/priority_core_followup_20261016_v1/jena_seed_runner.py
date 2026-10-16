"""Two fresh Jena B0/B2 seed92602 development runs; default performs no I/O.

The original fit_base._stage performs all optimization and selection. Its
explicit bounds are used because split_ranges on the shortened pre-T Task
would change the experiment. ROOT must seal a new plan and supervise CPU
RSS/deadline/quiet cost windows; this module never starts another process.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

NAME = "priority_core_followup_20261016_v1"
UPSTREAM = "priority_b2_validation_20261016_v1"
UPSTREAM_PLAN_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
PRIORITY_RUNNER_SHA = "f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07"
PREFIX_SHA = "6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541"
IDS = {family: "priority_seed2_jena_" + family + "_k6_s92602_v1" for family in ("B0", "B2")}
SEED, L, H, C, K = 92602, 512, 48, 21, 6
BOUNDS = {"A_fit": [0, 10482], "A_stop": [10482, 12579], "M_cal": [12579, 14675],
          "V_select": [14675, 16772], "T_final": [16772, 20965]}
HARD_DEADLINE = 1792222550


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inside(root, relative):
    root = Path(root).absolute()
    path = root / relative
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Nonsymlink paths required")
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes declared root")
    return path


def new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def spec_for(family):
    if family not in IDS:
        raise ValueError("Only B0/B2 are preregistered")
    return dict(run_id=IDS[family], dataset="jena", family=family, latent_k=K, seed=SEED, fold="primary",
                epochs=[.25, 1., 3.], lr=.001, device="cpu", microbatch=4, effective_batch=16,
                F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16,
                stage_label="POST_T_FRESH_SECOND_SEED_FIXED_ORIGINAL_BOUNDS")


def validate_proposal(plan):
    if (plan.get("schema") != "PRIORITY_JENA_SECOND_SEED_v1" or plan.get("namespace") != NAME
            or plan.get("stage") != "POST_T_V_ONLY_DEVELOPMENT" or plan.get("status") != "SEALED_BY_ROOT"):
        raise ValueError("ROOT-sealed second-seed plan required; a draft is not executable")
    if (plan.get("seed") != SEED or plan.get("role_run_ids") != IDS
            or plan.get("conditions") != [spec_for(family) for family in IDS]
            or plan.get("maximum_new_training_conditions") != 2):
        raise ValueError("Exactly two fresh B0/B2 seed92602 recipes are allowed")
    if (plan.get("original_primary_split_ranges") != BOUNDS or plan.get("materialized_prefix_end_exclusive") != 16772
            or plan.get("new_T_target_operations_allowed") is not False
            or plan.get("protected_tail_numeric_parse_allowed") is not False
            or plan.get("upstream_priority_plan_sha256") != UPSTREAM_PLAN_SHA):
        raise ValueError("Original pre-T information and split contracts changed")
    resource = plan["resources"]
    if (resource.get("CPU_threads") != 2 or resource.get("max_worker_RSS_bytes") != 3 * 1024**3
            or resource.get("maximum_model_workers") != 2 or resource.get("storage_limit_bytes") != 30 * 1024**3
            or resource.get("admission_reserve_bytes") != 256 * 1024**2
            or resource.get("costs_must_not_overlap_training") is not True
            or resource.get("ROOT_CPU_supervisor_required") is not True
            or not isinstance(plan.get("deadline_unix"), (int, float))
            or not math.isfinite(plan["deadline_unix"]) or plan["deadline_unix"] > HARD_DEADLINE):
        raise ValueError("Original CPU/resource limits must be retained")


def metadata(public, cache, asset, plan_sha, process_api):
    public, cache, asset = map(lambda p: inside(p, "."), (public, cache, asset))
    plan_path = inside(public, NAME + "/PREREG_PLAN.json")
    if not plan_sha or sha(plan_path) != plan_sha:
        raise ValueError("Explicit current second-seed plan SHA required")
    plan = read(plan_path)
    validate_proposal(plan)
    if sha(inside(public, NAME + "/jena_seed_runner.py")) != plan["runner_sha256"]:
        raise ValueError("Second-seed runner source changed")
    for relative, digest in ((UPSTREAM + "/priority_runner.py", PRIORITY_RUNNER_SHA),
                             (UPSTREAM + "/safe_jena_prefix.py", PREFIX_SHA)):
        if sha(inside(public, relative)) != digest:
            raise ValueError("Reviewed upstream wrapper/loader changed")
    priority = module("priority_seed_shared_guards", inside(public, UPSTREAM + "/priority_runner.py"))
    upstream = priority.metadata(public, cache, UPSTREAM_PLAN_SHA, process_api)
    if (plan["task_contract"] != upstream["task_contract"] or plan["source39"] != upstream["source39"]
            or plan["immutable_public"] != upstream["immutable_public"]):
        raise ValueError("Second-seed data and original source pins must inherit the sealed contract")
    for field in ("foundation_tensor_sha256", "model_asset_relative_path", "instance_norm_source_relative_path", "instance_norm_source_sha256"):
        if plan[field] != upstream[field]:
            raise ValueError("Original foundation identity changed")
    norm = inside(asset, plan["instance_norm_source_relative_path"])
    live_norm = inside(public.parent, "temporal_frontend_campaign_20261015_v1/runtime.py")
    if sha(norm) != plan["instance_norm_source_sha256"] or sha(live_norm) != plan["instance_norm_source_sha256"]:
        raise ValueError("Actual original CPU normalization source changed")
    priority.check_deadline(plan["deadline_unix"])
    return plan, priority


def require_new_fit(public_scope, cache_scope, run_id):
    for root in (Path(public_scope), Path(cache_scope)):
        if (root / "runs" / run_id).exists() or (root / "predictions" / run_id).exists():
            raise FileExistsError("Existing attempt is preserved; no resume or implicit cache reuse")


def resource_preflight(public, cache, plan, process_api):
    total = sum(path.stat().st_size for root in (Path(public), Path(cache)) for path in root.rglob("*") if path.is_file())
    limits = plan["resources"]
    if total + limits["admission_reserve_bytes"] > limits["storage_limit_bytes"]:
        raise RuntimeError("Original combined storage admission limit reached")
    import shutil
    if shutil.disk_usage(cache).free < limits["admission_reserve_bytes"]:
        raise RuntimeError("Private cache free-space reserve unavailable")
    if process_api.Process().memory_info().rss > limits["max_worker_RSS_bytes"]:
        raise MemoryError("Worker RSS exceeds the original limit")
    return total


def fit_stages(fit, spec, task, foundation, contract, public_run, private_run, deadline):
    """Call original stage API with fixed original bounds, never split a truncated Task."""
    search = fit._stage(spec, task, BOUNDS["A_fit"][1], BOUNDS["A_stop"], spec["epochs"],
                        private_run / "search", public_run / "search", foundation, deadline, contract)
    if search["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="BUDGET_LIMIT_PRESERVED", phase="search", summary=search["summary"])
    selected = search["summary"]["selected_epoch"]
    if selected not in (0., .25, 1., 3.):
        raise ValueError("Original A_stop selected an undeclared epoch")
    refit = fit._stage(spec, task, BOUNDS["A_stop"][1], None, [selected] if selected else [],
                       private_run / "refit", public_run / "refit", foundation, deadline, contract)
    if refit["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="BUDGET_LIMIT_PRESERVED", phase="refit", summary=refit["summary"])
    return dict(status="COMPLETE_STAGES", search=search, refit=refit, selected_epoch=selected)


def fit_and_save(spec, task, foundation, context, public_run, private_run, deadline):
    fit, priority = context["fit"], context["priority"]
    source_sha, source_files = fit._source_contract()
    axes = dict(L=L, H=H, C=C, K=K, columns=task["columns"], fold="primary",
                inherited_source_array_fingerprints=task["array_sha256"],
                materialized_prefix_sha256=task["materialized_prefix_sha256"])
    contract = dict(source_sha=source_sha, config_sha=fingerprint(dict(spec=spec, axes=axes)),
                    data_sha=task["data_sha256"], model_sha=fit._model_sha(foundation))
    new(public_run / "SOURCE_CONTRACT.json", dict(**contract, files=source_files, spec=spec,
        inherited_full_task_identity_status="INHERITED_NOT_RECOMPUTED", materialized_prefix_sha256=task["materialized_prefix_sha256"]))
    stages = fit_stages(fit, spec, task, foundation, contract, public_run, private_run, deadline)
    if stages["status"] != "COMPLETE_STAGES":
        return stages
    search, refit, selected = stages["search"], stages["refit"], stages["selected_epoch"]
    for label, stage in (("search", search), ("refit", refit)):
        priority.write_stage(public_run / label, stage)
    state = priority.state_sha(refit["model"], context["data"])
    bundle = dict(model=refit["model"], normalization=refit["normalization"], spec=spec,
                  contract=refit["summary"]["contract"], selected_search_epoch=selected,
                  search_summary=search["summary"], refit_summary=refit["summary"], model_state_sha=state,
                  plan_sha256=context["plan_sha"], materialized_prefix_sha256=task["materialized_prefix_sha256"])
    artifact = private_run / "selected_bundle.pt"
    priority.save_bundle(artifact, bundle, context["torch"])
    values = [row["astop_mse"] for row in search["curves"] if row["astop_mse"] is not None]
    receipt = dict(status="COMPLETE_VALID", run_id=spec["run_id"], family=spec["family"], seed=SEED,
        epochs=spec["epochs"], selected_epoch=selected, selected_astop_mse=search["summary"]["selected_astop_mse"],
        search_fit_end=10482, refit_fit_end=12579, search_updates=search["summary"]["updates"],
        refit_updates=refit["summary"]["updates"], search_exposures=search["summary"]["exposures"],
        refit_exposures=refit["summary"]["exposures"], source_sha=source_sha,
        artifact_sha256=sha(artifact), model_state_sha=state,
        refit_scaler_sha=fit.scaler_fingerprint(bundle["normalization"]["mean"], bundle["normalization"]["std"]),
        artifact_cache_relative_path=artifact.relative_to(context["cache_scope"]).as_posix(),
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if selected == 3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",
        last_cp_best=bool(values and values[-1] <= min(values)), source_stage_API="original_fit_base._stage",
        original_full_task_identity="INHERITED_NOT_RECOMPUTED", materialized_prefix_sha256=task["materialized_prefix_sha256"],
        post_T_development=True, protected_T_target_operations=0, reused_complete_artifact=False)
    new(public_run / "BASE_FIT_RECEIPT.json", receipt)
    model, regenerated = fit.create_model(spec, task, 12579, foundation)
    model.load_state_dict(bundle["model"], strict=True)
    if any(not context["np"].array_equal(regenerated[name], bundle["normalization"][name]) for name in ("mean", "std")):
        raise ValueError("Fresh deployment statistics differ from selected artifact")
    return dict(status="COMPLETE_VALID", receipt=receipt, bundle=bundle, model=model, artifact=artifact)


def predict_full(fitted, task, foundation, reference, context, public_folder, private_folder, deadline):
    np, torch, data, priority = (context[name] for name in ("np", "torch", "data", "priority"))
    norm, spec = fitted["bundle"]["normalization"], fitted["bundle"]["spec"]
    arrays = priority.cohort_arrays(task, BOUNDS, norm["mean"], norm["std"], data, np)
    priority.validate_cohort(arrays, reference, context["evaluation"])
    public_folder.mkdir(parents=True, exist_ok=False)
    private_folder.mkdir(parents=True, exist_ok=False)
    identity = dict(run_id=spec["run_id"], split="V_select", cohort="jena:primary:V_select:stride48",
        data_sha=task["data_sha256"], scaler_sha=fitted["receipt"]["refit_scaler_sha"],
        source_sha=fingerprint(context["new_sources"]), config_sha=fingerprint(spec),
        model_sha=fingerprint(dict(artifact_sha=fitted["receipt"]["artifact_sha256"], model_state_sha=fitted["receipt"]["model_state_sha"])))
    new(public_folder / "REFERENCE_CONTRACT.json", dict(**identity,
        arrays={name: context["evaluation"].array_fingerprint(value) for name, value in arrays.items()}, expected_F_streams=43 * K))
    windows = context["training"].FastWindows(task, norm["mean"], norm["std"], "cpu")
    before = dict(streams=foundation.streams, physical_streams=foundation.physical_streams, physical_calls=foundation.physical_calls)
    predictions = []
    fitted["model"].eval()
    with torch.no_grad():
        for offset in range(0, 43, 4):
            priority.check_deadline(deadline)
            x, _, _ = windows.get(arrays["origins"][offset:offset + 4])
            predictions.append(context["fit"].predict_model(fitted["model"], x, foundation, spec["family"], K).cpu().numpy().astype(np.float32))
    arrays["pred_normalized"] = np.concatenate(predictions)
    delta = {key: int(getattr(foundation, key)) - before[key] for key in before}
    if arrays["pred_normalized"].shape != (43, H, C) or delta != dict(streams=258, physical_streams=258, physical_calls=11):
        raise ValueError("Full V43 shape or actual row128 foundation accounting differs")
    archive = private_folder / "V_select.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    score, statistics = context["evaluation"].error_statistics(arrays["pred_normalized"], arrays["target_normalized"],
        arrays["observed"], arrays["origins"], spec["run_id"], "V_select", identity["cohort"], arrays["channel_ids"])
    with gzip.open(public_folder / "PUBLIC_ERROR_STATS.csv.gz", "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0])); writer.writeheader(); writer.writerows(statistics)
    priority.csv_new(public_folder / "SCORES.csv", [score])
    manifest = dict(**identity, archive_sha256=sha(archive), arrays={name: context["evaluation"].array_fingerprint(value) for name, value in arrays.items()},
        score=score, family=spec["family"], seed=SEED, artifact_sha256=fitted["receipt"]["artifact_sha256"],
        device="cpu", F_counters=delta, post_T_development=True, protected_T_target_operations=0,
        independent_generalization_claimed=False, full_V_checkpoint_replay_claimed=False)
    new(public_folder / "PREDICTION_MANIFEST.json", manifest)
    return manifest


def load_completed(public_run, private_run, spec, context):
    receipt = read(public_run / "BASE_FIT_RECEIPT.json")
    source = read(public_run / "SOURCE_CONTRACT.json")
    artifact = private_run / "selected_bundle.pt"
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != spec["run_id"]
            or receipt.get("family") != spec["family"] or receipt.get("seed") != SEED or source["spec"] != spec
            or sha(artifact) != receipt["artifact_sha256"]):
        raise ValueError("Exact completed seed2 artifact and scientific recipe required")
    return receipt, artifact


def actual_route(fitted, task, foundation_pool, context):
    import deployment
    class SeedRoute(deployment.Route):
        def __init__(self):
            self.spec = copy.deepcopy(fitted["bundle"]["spec"])
            self.task, self.device, self.run_id = task, "cpu", self.spec["run_id"]
            self.family, self.model = self.spec["family"], fitted["model"]
            norm = fitted["bundle"]["normalization"]
            self.mean, self.std = norm["mean"], norm["std"]
            self.pool, self.owns_pool = foundation_pool, False
            self.g_route = self.b_route = self.ridge_artifact = None
            self._closed, self._stage_mode, self._stages, self.last_receipt = False, False, {}, {}
            self.latent_k, self.artifact_sha256 = K, fitted["receipt"]["artifact_sha256"]
            self.mean_t = context["torch"].as_tensor(self.mean, device="cpu")
            self.std_t = context["torch"].as_tensor(self.std, device="cpu")
            self.model.eval()
    return SeedRoute()


def replay(fitted, arrays, task, pool, context, output, deadline):
    np, priority = context["np"], context["priority"]
    indices = priority.replay_indices(43, np)
    origins = arrays["origins"][indices]
    route = actual_route(fitted, task, pool, context)
    before, predicted = pool.counters(), []
    try:
        for start in range(0, 6, 4):
            priority.check_deadline(deadline)
            batch = origins[start:start + 4]
            physical = np.stack([task["physical"][int(o) - L:int(o)] for o in batch])
            observed = np.stack([task["observed"][int(o) - L:int(o)] for o in batch])
            raw = route.predict_physical(physical, observed)
            predicted.append(((raw - route.mean.astype(np.float64)) / route.std.astype(np.float64)).astype(np.float32))
        actual = np.concatenate(predicted)
        delta = {key: pool.counters()[key] - before[key] for key in ("streams", "physical_streams", "physical_calls")}
        if delta != dict(streams=36, physical_streams=36, physical_calls=2):
            raise ValueError("Fresh six-origin 4+2 replay F accounting differs")
        passed = bool(np.allclose(actual, arrays["pred_normalized"][indices], atol=1e-5, rtol=1e-4))
        result = dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY", executed_now=True,
            run_id=fitted["receipt"]["run_id"], artifact_sha256=fitted["receipt"]["artifact_sha256"], origins=origins.tolist(),
            replay_selection_seed=92601, microbatches=[4, 2], F_counters=delta, atol=1e-5, rtol=1e-4,
            max_abs=float(np.abs(actual.astype(np.float64) - arrays["pred_normalized"][indices].astype(np.float64)).max()),
            full_V_saved_reinference_claimed=False, new_T_target_operations=0)
    finally:
        route.close()
    new(output / "REPLAY_RECEIPT.json", result)
    if not passed:
        raise ValueError("Actual selected checkpoint replay parity failed")
    return result


def cost(fitted, task, pool, context, output, batch, deadline, quiet_proof, quiet_sha):
    if batch not in (1, 8) or not quiet_proof or not quiet_sha or sha(quiet_proof) != quiet_sha:
        raise ValueError("Cost requires B1/B8 and an explicit ROOT quiet-window proof SHA")
    quiet = read(quiet_proof)
    if quiet.get("status") != "QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers") != [] or quiet.get("active_cost_workers") != []:
        raise ValueError("Training and other costs must be closed before precise cost measurement")
    import measure_costs
    np, priority = context["np"], context["priority"]
    origins = context["data"].origins(BOUNDS["V_select"], L, H, H)[:batch]
    physical = np.stack([task["physical"][int(o) - L:int(o)] for o in origins])
    observed = np.stack([task["observed"][int(o) - L:int(o)] for o in origins])
    route = actual_route(fitted, task, pool, context)
    class Check:
        def check(self):
            priority.check_deadline(deadline)
    try:
        rows = measure_costs.measure_balanced([route], physical, observed, warmups=10, timed_trials=20, stage_trials=20, lease=Check())
        if any((row["F_logical_streams"], row["F_physical_streams"], row["F_calls"]) != (batch * K, batch * K, 1) for row in rows):
            raise ValueError("Actual CPU whole-path F counters differ")
        memory = route.memory_state()
    finally:
        route.close()
    trials = output / "COST_TRIALS.csv.gz"
    measure_costs._write_trials(trials, rows)
    result = dict(status="COMPLETE_PROTOCOL", run_id=fitted["receipt"]["run_id"], device="cpu", CPU_threads=2,
        batch_size=batch, warmups=10, timed_trials=20, stage_trials=20, actual_trial_rows=len(rows),
        summaries=measure_costs.summarize(rows), memory=memory, artifact_sha256=fitted["receipt"]["artifact_sha256"],
        physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(), observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        origins=origins.tolist(), trials_sha256=sha(trials), quiet_window_proof_sha256=quiet_sha,
        cached_Z_or_forecast_reads=0, source_protocol="original_measure_costs.measure_balanced",
        stage_probe_separate_intrusive_scope=True, training_peak_included=False, new_T_target_operations=0)
    new(output / "COST_RECEIPT.json", result)
    return result


def execute(public, cache, asset, plan_sha, run_id, mode, deadline, output, batch=None, quiet_proof=None, quiet_sha=None):
    import psutil
    public, cache, asset = map(lambda p: inside(p, "."), (public, cache, asset))
    plan, priority = metadata(public, cache, asset, plan_sha, psutil)
    family = next((key for key, value in IDS.items() if value == run_id), None)
    if family is None or mode not in ("fit", "rescore", "replay", "cost"):
        raise ValueError("One exact preregistered second-seed ID/mode required")
    if not isinstance(deadline, (int, float)) or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Live deadline within original limits required")
    if (mode == "cost" and batch not in (1, 8)) or (mode != "cost" and batch is not None):
        raise ValueError("Only cost accepts B1/B8")
    priority.require_environment()
    public_scope, cache_scope = inside(public, NAME), inside(cache, NAME)
    if not cache_scope.is_dir():
        raise ValueError("ROOT must prepare an empty namespace cache before execution")
    output = inside(public_scope, os.path.relpath(Path(output).absolute(), public_scope))
    if output == public_scope or output.exists() or not output.is_relative_to(public_scope / "diagnostics"):
        raise FileExistsError("Fresh diagnostic output path required")
    spec = spec_for(family)
    if mode == "fit":
        require_new_fit(public_scope, cache_scope, run_id)
    storage = resource_preflight(public, cache, plan, psutil)
    public_run, private_run = public_scope / "runs" / run_id, cache_scope / "runs" / run_id
    public_prediction, private_prediction = public_scope / "predictions" / run_id / "V_select", cache_scope / "predictions" / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    new(output / "STARTED.json", dict(run_id=run_id, mode=mode, plan_sha256=plan_sha, actual_pid=os.getpid(),
        actual_create_time=psutil.Process().create_time(), combined_storage_bytes=storage, protected_T_target_operations=0))
    context, pool, result, failure, post_error = None, None, None, None, None
    try:
        sys.path.insert(0, str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve() != public / "data_contract.py":
            raise ValueError("Original source module must come from the declared campaign")
        context = dict(np=np, data=data, evaluation=evaluation, priority=priority, plan_sha=plan_sha,
            cache_scope=cache_scope, new_sources={"jena_seed_runner.py": plan["runner_sha256"],
                "priority_runner.py": PRIORITY_RUNNER_SHA, "safe_jena_prefix.py": PREFIX_SHA})
        reference = read(inside(public, plan["V_reference_public_relative_path"]))
        if sha(inside(public, plan["V_reference_public_relative_path"])) != plan["V_reference_sha256"]:
            raise ValueError("Pinned original V43 reference changed")
        if mode == "rescore":
            receipt, artifact = load_completed(public_run, private_run, spec, context)
            arrays, manifest = priority.load_saved_prediction(public_prediction, private_prediction, reference, context)
            validate_saved(manifest, receipt, spec, plan_sha, context)
            with (public_prediction / "SCORES.csv").open(encoding="utf-8", newline="") as stream:
                scores = list(csv.DictReader(stream))
            if len(scores) != 1 or any(scores[0][key] != manifest[key] for key in ("run_id", "split", "cohort")):
                raise ValueError("Exactly one identity-matched score row required")
            with gzip.open(public_prediction / "PUBLIC_ERROR_STATS.csv.gz", "rt", encoding="utf-8", newline="") as stream:
                statistics = list(csv.DictReader(stream))
            result = priority.independent_rescore(arrays, manifest, statistics, scores[0], np)
            result.update(run_id=run_id, artifact_sha256=receipt["artifact_sha256"], manifest_sha256=sha(public_prediction / "PREDICTION_MANIFEST.json"))
            new(output / "RESCORE_RECEIPT.json", result)
        else:
            import torch
            import fit_base as fit
            import training
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized():
                raise RuntimeError("Unexpected CUDA initialization")
            context.update(torch=torch, fit=fit, training=training)
            task, bounds, access = priority.load_task_prefix(asset, plan, context, public)
            if {key: list(value) for key, value in bounds.items()} != BOUNDS:
                raise ValueError("Original split boundaries differ")
            new(output / "PREFIX_ACCESS_RECEIPT.json", access)
            pool = deployment.FoundationPool(inside(asset, plan["model_asset_relative_path"]), "cpu")
            foundation = pool.get(H)
            if (gpu_runtime.state_hash(foundation.model) != plan["foundation_tensor_sha256"]
                    or foundation.row_microbatch != 128 or foundation.checkpoint_inputs is not False
                    or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]):
                raise ValueError("Actual frozen CPU foundation identity/runtime differs")
            if mode == "fit":
                public_run.mkdir(parents=True, exist_ok=False); private_run.mkdir(parents=True, exist_ok=False)
                fitted = fit_and_save(spec, task, foundation, context, public_run, private_run, deadline)
                if "model" not in fitted:
                    result = fitted
                else:
                    manifest = predict_full(fitted, task, foundation, reference, context, public_prediction, private_prediction, deadline)
                    result = dict(status="COMPLETE_FIT_AND_FULL_V", fit_receipt=fitted["receipt"], V_score=manifest["score"],
                        full_V_completed=True, independent_rescore_pending=True, checkpoint_replay_pending=True)
            else:
                receipt, artifact = load_completed(public_run, private_run, spec, context)
                bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                if (bundle["plan_sha256"] != plan_sha or priority.state_sha(bundle["model"], data) != receipt["model_state_sha"]
                        or bundle["spec"] != spec or bundle["contract"]["model_sha"] != plan["foundation_tensor_sha256"]
                        or bundle["contract"]["source_sha"] != fit._source_contract()[0]):
                    raise ValueError("Selected artifact tensor/source/foundation identity differs")
                model, norm = fit.create_model(spec, task, 12579, foundation)
                model.load_state_dict(bundle["model"], strict=True)
                if any(not np.array_equal(norm[key], bundle["normalization"][key]) for key in ("mean", "std")):
                    raise ValueError("Selected canonical scaler differs")
                fitted = dict(receipt=receipt, artifact=artifact, bundle=bundle, model=model)
                manifest = read(public_prediction / "PREDICTION_MANIFEST.json")
                validate_saved(manifest, receipt, spec, plan_sha, context)
                if mode == "replay":
                    arrays, _ = priority.load_saved_prediction(public_prediction, private_prediction, reference, context)
                    result = replay(fitted, arrays, task, pool, context, output, deadline)
                else:
                    result = cost(fitted, task, pool, context, output, batch, deadline, quiet_proof, quiet_sha)
            foundation.assert_frozen_unchanged()
            result["F_final_state_sha256"] = gpu_runtime.state_hash(foundation.model)
        priority.check_deadline(deadline)
    except Exception as error:
        failure = type(error).__name__
        path = cache_scope / "failures" / (mode + "_" + run_id + "_" + str(time.time_ns()) + ".json")
        new(path, dict(error_class=failure, error_message=str(error), traceback=traceback.format_exc()))
        result = dict(status="FAILED_PRESERVED", error_class=failure, private_failure_cache_relative_path=path.relative_to(cache).as_posix())
    finally:
        try:
            if pool is not None:
                pool.close()
            metadata(public, cache, asset, plan_sha, psutil)
            resource_preflight(public, cache, plan, psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized():
                raise RuntimeError("CPU operation initialized CUDA")
        except Exception as error:
            post_error = type(error).__name__
            path = cache_scope / "failures" / ("postflight_" + mode + "_" + run_id + "_" + str(time.time_ns()) + ".json")
            new(path, dict(error_class=post_error, error_message=str(error), traceback=traceback.format_exc()))
        new(output / "RESULT.json", dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status", "INCOMPLETE"),
            run_id=run_id, mode=mode, plan_sha256=plan_sha, scientific_result=result, error_class=failure,
            postflight_error_class=post_error, elapsed_wall_seconds=time.time() - started,
            original_source39_immutable9_preserved=post_error is None, protected_T_target_operations=0,
            original_full_task_identity_inherited_not_recomputed=True, independent_generalization_claimed=False,
            interrupt_scope="Original deadline checks at optimizer boundaries preserve stage checkpoints; arbitrary signals are not checkpoint-safe here",
            ROOT_RSS_deadline_quiet_supervisor_required=True))
    if failure or post_error:
        raise RuntimeError("Second-seed operation failed; private traceback and public status preserved")
    return result


def validate_saved(manifest, receipt, spec, plan_sha, context):
    if (manifest.get("run_id") != spec["run_id"] or manifest.get("family") != spec["family"] or manifest.get("seed") != SEED
            or manifest.get("split") != "V_select" or manifest.get("artifact_sha256") != receipt["artifact_sha256"]
            or manifest.get("config_sha") != fingerprint(spec) or manifest.get("source_sha") != fingerprint(context["new_sources"])
            or manifest.get("scaler_sha") != receipt["refit_scaler_sha"]
            or manifest.get("model_sha") != fingerprint(dict(artifact_sha=receipt["artifact_sha256"], model_state_sha=receipt["model_state_sha"]))):
        raise ValueError("Saved accuracy identity differs from the exact trained second-seed route")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-triggered", action="store_true")
    for name in ("public-root", "cache-root", "asset-root", "plan-sha", "run-id", "output", "quiet-proof", "quiet-sha"):
        parser.add_argument("--" + name)
    parser.add_argument("--mode", choices=("fit", "rescore", "replay", "cost"))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--batch", type=int, choices=(1, 8))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED", input_reads=0, writes=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
            science_or_child_operations=0, proposed_run_ids=list(IDS.values()), actual_new_training_conditions=0)))
        return 0
    if not args.root_triggered or any(getattr(args, key) is None for key in ("public_root", "cache_root", "asset_root", "plan_sha", "run_id", "mode", "deadline", "output")):
        parser.error("Execution requires ROOT trigger, sealed plan SHA, exact ID, roots, deadline and fresh output")
    result = execute(args.public_root, args.cache_root, args.asset_root, args.plan_sha, args.run_id, args.mode,
                     args.deadline, args.output, args.batch, args.quiet_proof, args.quiet_sha)
    print(json.dumps(dict(status=result["status"], run_id=args.run_id, mode=args.mode)))
    return 0 if result["status"].startswith(("PASS", "COMPLETE")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
