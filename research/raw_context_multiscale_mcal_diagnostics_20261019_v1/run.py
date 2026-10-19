"""Evaluate final M_cal models from existing parent caches on CPU only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import time

import numpy as np
import torch

from research.raw_context_multiscale_20261018_v1 import calibration, common, report


NAME = "raw_context_multiscale_mcal_diagnostics_20261019_v1"
CSV_NAME = "MCAL_FINAL_FIT_DIAGNOSTICS.csv"
VALIDATION_NAME = "MCAL_FINAL_FIT_VALIDATION.json"
METRICS = ("M_fit_selected_model_mse", "M_fit_selected_model_mae", "M_select_selected_model_mse",
           "M_select_selected_model_mae", "M_cal_final_model_mse", "M_cal_final_model_mae", "V_report_mse", "V_report_mae")


def deadline_check(deadline):
    if deadline is not None and time() >= deadline:
        raise TimeoutError("Original execution deadline exhausted; preserve completed diagnostic arrays")


def _check_sources(ctx, sources):
    for relative, digest in sources.items():
        path = (Path(ctx["repo"])/relative).resolve()
        if not path.is_relative_to(Path(ctx["repo"]).resolve()) or common.sha(path) != digest:
            raise ValueError("Frozen diagnostic dependency changed: "+relative)


def gates(ctx, scope_path, preflight_path):
    """Only JSON/text/source fingerprints are inspected before this gate returns."""
    scope, preflight = common.read(scope_path), common.read(preflight_path)
    diagnostic = scope["diagnostic_scope"]
    if (scope.get("status") != "REGISTERED_BEFORE_NEW_V" or scope.get("scientific_contract_changed") is not False
            or scope.get("selection_changes") is not False or scope.get("new_training_updates") != 0
            or diagnostic.get("split") != "M_cal" or diagnostic.get("models") != 40
            or diagnostic.get("alpha_models") != 32 or diagnostic.get("head_models") != 8
            or diagnostic.get("head_batch_size") != 64 or diagnostic.get("device") != "cpu"
            or any(diagnostic.get(key) is not False for key in ("new_F_inference", "V_selection", "new_V_forecasts", "T_access"))
            or tuple(diagnostic.get("metrics", [])) != METRICS
            or diagnostic.get("tolerance") != dict(metric_atol=1e-12, sufficient_stats_atol=1e-12, sufficient_stats_rtol=1e-13)):
        raise ValueError("Exact registered, CPU-only final M_cal diagnostic scope required")
    if (preflight.get("status") != "PASS_SYNTHETIC_AND_FROZEN_CODE"
            or preflight.get("scope_binding") != common.binding(scope_path, ctx)
            or preflight.get("actual_evaluation_performed") is not False):
        raise ValueError("Frozen synthetic code preflight bound to this scope is required")
    expected = {"research/"+NAME+"/"+filename for filename in ("__init__.py", "run.py")}
    if set(preflight["source_files"]) != expected or set(preflight["test_source_files"]) != {"tests/test_raw_context_multiscale_mcal_diagnostics.py"}:
        raise ValueError("Both new diagnostic sources and the focused test must be frozen")
    for key in ("primary_source_files", "supplement_source_files", "independent_source_files"):
        _check_sources(ctx, scope[key])
    _check_sources(ctx, preflight["source_files"])
    _check_sources(ctx, preflight["test_source_files"])
    primary_plan_path = common.resolve_binding(scope["primary_plan_binding"], ctx)
    if common.read(primary_plan_path)["source_files"] != scope["primary_source_files"] or len(scope["primary_source_files"]) != 13:
        raise ValueError("Thirteen exact primary producer sources required")
    common.resolve_binding(scope["supplement_plan_binding"], ctx)
    common.resolve_binding(scope["independent_plan_binding"], ctx)
    start_path = common.resolve_binding(scope["original_execution_start_binding"], ctx)
    start = common.read(start_path)
    deadline = start["started_unix"]+start["limits"]["wall_seconds"]
    if deadline != scope["original_deadline_unix"]:
        raise ValueError("Original immutable wall deadline changed")
    deadline_check(deadline)
    terminal_path = Path(ctx["cache"])/"TERMINAL_CORE_002.json"
    terminal = common.read(terminal_path)
    if terminal.get("status") != "COMPLETE" or terminal.get("exit_code") != 0:
        raise ValueError("Main002 must be COMPLETE exit0 before private numerical access")
    validation_path = Path(ctx["report"])/"VALIDATION_REPORT.json"
    validation = common.read(validation_path)
    if validation.get("status") != "PASS" or validation.get("models") != 56:
        raise ValueError("Complete primary independent numerical validation must PASS")
    if validation["manifest_bindings"]["registered_plan"] != scope["primary_plan_binding"]:
        raise ValueError("Primary audit binds a different producer plan")
    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json"):
        if common.resolve_binding(validation["manifest_bindings"][name], ctx) != Path(ctx["report"])/name:
            raise ValueError("Exact current primary manifests must be independently audited")
    return dict(scope=scope, preflight=preflight, deadline=deadline,
        scope_binding=common.binding(scope_path, ctx), preflight_binding=common.binding(preflight_path, ctx),
        terminal_binding=common.binding(terminal_path, ctx), primary_validation_binding=common.binding(validation_path, ctx))


def independent_statistics(prediction, target, observed, *, deadline=None):
    """Scalar FP64 fsum per horizon/channel, independent of producer np.sum."""
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    if prediction.ndim != 3 or prediction.shape != target.shape or target.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Exact aligned original-channel arrays and original bool mask required")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Finite fixed-model predictions and observed targets required")
    horizon, channels = prediction.shape[1:]
    sse, sae = np.zeros((horizon, channels), np.float64), np.zeros((horizon, channels), np.float64)
    counts = np.zeros((horizon, channels), np.int64)
    for h in range(horizon):
        deadline_check(deadline)
        for c in range(channels):
            selected = mask[:, h, c]
            difference = prediction[selected, h, c].astype(np.float64)-target[selected, h, c].astype(np.float64)
            sse[h, c] = math.fsum(float(value)*float(value) for value in difference)
            sae[h, c] = math.fsum(abs(float(value)) for value in difference)
            counts[h, c] = int(selected.sum())
    n = counts.sum(0)
    active = np.flatnonzero(n)
    if not len(active):
        raise ValueError("At least one observed channel required")
    return dict(mse=math.fsum(math.fsum(sse[:, c])/int(n[c]) for c in active)/len(active),
        mae=math.fsum(math.fsum(sae[:, c])/int(n[c]) for c in active)/len(active),
        n_origins=len(prediction), n_observed=int(n.sum()),
        horizon_channel_sse=sse.tolist(), horizon_channel_sae=sae.tolist(), horizon_channel_count=counts.tolist())


def check_statistics(actual, reference, tolerance):
    for metric in ("mse", "mae"):
        if not math.isclose(actual[metric], reference[metric], rel_tol=0., abs_tol=tolerance["metric_atol"]):
            raise ValueError("Independent M_cal "+metric+" differs")
    for key in ("n_origins", "n_observed", "horizon_channel_count"):
        if actual[key] != reference[key]:
            raise ValueError("Independent observed counts differ")
    for key in ("horizon_channel_sse", "horizon_channel_sae"):
        if not np.allclose(actual[key], reference[key], atol=tolerance["sufficient_stats_atol"], rtol=tolerance["sufficient_stats_rtol"]):
            raise ValueError("Independent sufficient statistics differ: "+key)


def load_parent(ctx, task, binding, spec, cache, deadline):
    identity = binding["sha256"]
    if identity in cache:
        values = cache[identity]
        if values["metadata"]["route_spec"] != spec:
            raise ValueError("Shared parent bytes cannot represent a different route")
        return values
    deadline_check(deadline)
    path = common.resolve_binding(binding, ctx)
    with np.load(path, allow_pickle=False) as archive:
        values = {key:archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")}
        metadata = json.loads(str(archive["metadata_json"].item()))
    lo, hi = task["split_ranges"]["M_cal"]
    origins = np.arange(max(lo, task["L"]), hi-task["H"]+1, dtype=np.int64)
    mean, std, _ = ctx["modules"]["data"].fit_statistics(task, lo)
    rows = origins[:, None]+np.arange(task["H"])[None]
    observed = task["observed"][rows]
    target = np.where(observed, (task["physical"][rows]-mean)/std, 0).astype(np.float32)
    scaler = ctx["modules"]["fit"].scaler_fingerprint(mean, std)
    if (not np.array_equal(values["origins"], origins) or values["observed"].dtype != np.bool_
            or not np.array_equal(values["observed"], observed) or not np.array_equal(values["target_normalized"], target)
            or not np.array_equal(values["channel_ids"], np.asarray(task["columns"]))
            or values["pred_normalized"].shape != target.shape or values["pred_normalized"].dtype != np.float32
            or not np.isfinite(values["pred_normalized"]).all()):
        raise ValueError("Parent M_cal canonical origins/targets/mask/channels/FP32 forecasts differ")
    if (metadata["route_spec"] != spec or metadata["split"] != "M_cal" or metadata["parent_fit_end"] != lo
            or metadata["data_sha256"] != task["data_sha256"] or metadata["scaler_sha256"] != scaler):
        raise ValueError("Exact parent route/data/A60-normalization identity differs")
    values.update(metadata=metadata, mean=mean, std=std, scaler_sha256=scaler, binding=binding)
    cache[identity] = values
    return values


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError("Only CPU head state tensors admitted")
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def predict_head_bounded(state, past, g, b, *, control, deadline, batch_size=64):
    outputs = []
    for start in range(0, len(g), batch_size):
        deadline_check(deadline)
        selected = slice(start, start+batch_size)
        outputs.append(calibration.predict_head(state, past[selected], g[selected], None if control else b[selected],
                                                control=control, batch_size=batch_size))
    deadline_check(deadline)
    return np.concatenate(outputs)


def final_prediction(ctx, task, arm, parent_cache, past_cache, deadline):
    g = load_parent(ctx, task, arm["parent_g_prediction_binding"], arm["parent_g"]["route_spec"], parent_cache, deadline)
    b = None if arm["mode"] == "past_control" else load_parent(ctx, task, arm["parent_b_prediction_binding"], arm["parent_b"]["route_spec"], parent_cache, deadline)
    if b is not None and any(not np.array_equal(g[key], b[key]) for key in ("target_normalized", "observed", "origins", "mean", "std", "channel_ids")):
        raise ValueError("Two fixed parents must use the exact same M_cal cohort/scaler/targets/mask")
    if arm["mode"] == "alpha":
        alpha = float(arm["alpha"])
        if arm["route_spec"]["alpha"] != alpha or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Final fixed alpha changed")
        return (g["pred_normalized"]+alpha*(b["pred_normalized"]-g["pred_normalized"])).astype(np.float32), g, None
    control = arm["mode"] == "past_control"
    path = common.resolve_binding(arm["head_binding"], ctx)
    if arm["route_spec"]["head_state_sha256"] != arm["head_binding"]["sha256"]:
        raise ValueError("Final route head artifact changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    contract = saved["contract"]
    if (saved["receipt"] != arm["refit"] or saved["search"] != arm["search"]
            or saved["head_config"] != dict(length=task["L"], horizon=task["H"], hidden=32)
            or state_hash(saved["state"]) != arm["refit"]["final_state_sha256"]
            or contract["dataset"] != arm["dataset"] or contract["seed"] != arm["seed"]
            or contract["data_sha256"] != task["data_sha256"] or contract["control"] is not control
            or contract["g"] != arm["parent_g_prediction_binding"]
            or contract["b"] != (None if control else arm["parent_b_prediction_binding"])
            or contract["calibration_source_sha256"] != common.sha(calibration.__file__)
            or contract["origins"] != common.array_identity(g["origins"])
            or contract["scaler"] != dict(mean=common.array_identity(g["mean"]), std=common.array_identity(g["std"]))):
        raise ValueError("Fresh final head state/search/refit/scaler/parent contract differs")
    if task["dataset"] not in past_cache:
        past_cache[task["dataset"]] = report.canonical_past(task, g["origins"], g["mean"], g["std"])
    past = past_cache[task["dataset"]]
    if common.array_identity(past) != contract["past"]:
        raise ValueError("Canonical filled A60-normalized past differs from head contract")
    prediction = predict_head_bounded(saved["state"], past, g["pred_normalized"],
        None if control else b["pred_normalized"], control=control, deadline=deadline)
    return prediction.astype(np.float32), g, state_hash(saved["state"])


def row_metadata(arm, stats, parent, prediction_binding, head_state_hash):
    row = dict(run_id=arm["run_id"], dataset=arm["dataset"], mode=arm["mode"], seed=arm["seed"], split="M_cal",
        base_role=arm.get("base_role"), alpha_label=arm.get("alpha_label"),
        alpha_fit=None, alpha_final=arm.get("alpha"), alpha_used_for_selected_scores=None, selected_M_select_epoch=None,
        M_cal_final_model_mse=stats["mse"], M_cal_final_model_mae=stats["mae"],
        V_report_mse=arm["score"]["mse"], V_report_mae=arm["score"]["mae"], V_report_scope="Copied unchanged completed primary SCORES; development V, no new forecast",
        M_cal_mean_batch_loss_not_full_metric=arm.get("refit", {}).get("mean_batch_loss"),
        M_cal_final_metric_scope="In-sample error of already fitted final M_cal model; never used for fitting or reselection",
        n_origins=stats["n_origins"], n_observed=stats["n_observed"], A60_normalization_fit_end=parent["metadata"]["parent_fit_end"],
        scaler_sha256=parent["scaler_sha256"], origins_identity=common.array_identity(parent["origins"]),
        target_identity=common.array_identity(parent["target_normalized"]), observed_identity=common.array_identity(parent["observed"]),
        prediction_binding=prediction_binding, head_checkpoint_binding=arm.get("head_binding"), head_final_state_sha256=head_state_hash,
        parent_g_run_id=arm["parent_g"]["run_id"] if "run_id" in arm["parent_g"] else arm["parent_g"]["route_spec"]["run_id"],
        parent_g_model_sha256=arm["parent_g"]["route_spec"].get("artifact_sha256"),
        parent_g_prediction_sha256=arm["parent_g_prediction_binding"]["sha256"],
        parent_g_cache_binding=arm["parent_g_prediction_binding"],
        parent_b_run_id=arm["parent_b"]["run_id"] if arm.get("parent_b") and "run_id" in arm["parent_b"] else arm["parent_b"]["route_spec"]["run_id"] if arm.get("parent_b") else None,
        parent_b_model_sha256=arm["parent_b"]["route_spec"].get("artifact_sha256") if arm.get("parent_b") else None,
        parent_b_prediction_sha256=arm["parent_b_prediction_binding"]["sha256"] if arm.get("parent_b_prediction_binding") else None,
        parent_b_cache_binding=arm.get("parent_b_prediction_binding"), device="cpu", new_training_updates=0,
        new_F_inference=False, new_V_forecasts=False, V_selection=False, T_access=False)
    if arm["mode"] == "alpha":
        point = next(value for value in arm["diagnostic"]["rows"] if value["label"] == arm["alpha_label"])
        row.update(alpha_fit=arm["diagnostic"]["fit_alpha"]["alpha"], alpha_used_for_selected_scores=point["alpha"])
    else:
        point = min(arm["search"]["curves"], key=lambda value:(value["M_select_mse"], value["endpoint_epochs"]))
        row["selected_M_select_epoch"] = point["endpoint_epochs"]
    for name in ("M_fit", "M_select"):
        for metric in ("mse", "mae"):
            row[name+"_selected_model_"+metric] = point[name+"_"+metric]
    row["selection_metric_scope"] = "Preserved pre-refit M_fit/M_select diagnostic at the previously selected checkpoint/coefficient; not recomputed by this diagnostic"
    return row


def evaluate(ctx, scope_path, preflight_path):
    started = time()
    proof = gates(ctx, scope_path, preflight_path)
    deadline, scope = proof["deadline"], proof["scope"]
    events_path = Path(ctx["cache"])/"EVENTS.jsonl"
    events_binding = common.binding(events_path, ctx)
    csv_path, validation_path = Path(ctx["report"])/CSV_NAME, Path(ctx["report"])/VALIDATION_NAME
    if validation_path.exists():
        previous = common.read(validation_path)
        if (previous.get("status") != "PASS" or previous.get("scope_binding") != proof["scope_binding"]
                or previous.get("preflight_binding") != proof["preflight_binding"]):
            raise ValueError("Existing diagnostic validation identity differs; preserve prior attempt")
        common.resolve_binding(previous["CSV_binding"], ctx)
        for item in previous["evidence"]:
            common.resolve_binding(item["prediction_binding"], ctx)
        return previous
    scores = common.read(Path(ctx["report"])/"SCORES.json")
    calibrations = common.read(Path(ctx["cache"])/"CALIBRATION_MODELS.json")
    arms = [arm for arm in scores["arms"] if arm["mode"] in ("alpha", "conditional", "past_control")]
    expected = {(dataset, "alpha", seed, role, label) for dataset in common.EXPECTED for seed in common.SEEDS
                for role in ("Raw", "G") for label in ("alpha0", "alpha05", "alpha1", "fitted")}
    expected |= {(dataset, mode, seed, None, None) for dataset in common.EXPECTED for seed in common.SEEDS for mode in ("conditional", "past_control")}
    if (len(arms) != 40 or {(arm["dataset"], arm["mode"], arm["seed"], arm.get("base_role"), arm.get("alpha_label")) for arm in arms} != expected
            or len(calibrations) != 40 or {arm["run_id"] for arm in arms} != {arm["run_id"] for arm in calibrations}):
        raise ValueError("All 32 fixed alpha variants and eight final heads are required")
    indexed = {arm["run_id"]:arm for arm in arms}
    for original in calibrations:
        if any(indexed[original["run_id"]].get(key) != value for key, value in original.items()):
            raise ValueError("Scored calibration differs from its already sealed final state")
    tasks = {dataset:common.load_task(ctx, dataset) for dataset in common.EXPECTED}
    parent_cache, past_cache, rows, evidence = {}, {}, [], []
    tolerance = scope["diagnostic_scope"]["tolerance"]
    for arm in arms:
        deadline_check(deadline)
        task = tasks[arm["dataset"]]
        prediction, parent, head_state = final_prediction(ctx, task, arm, parent_cache, past_cache, deadline)
        stats = independent_statistics(prediction, parent["target_normalized"], parent["observed"], deadline=deadline)
        output = Path(ctx["cache"])/"mcal_final_fit_diagnostics_001"/"predictions"/(arm["run_id"]+".npz")
        saved = common.save_prediction(output, prediction=prediction, target=parent["target_normalized"], observed=parent["observed"],
            origins=parent["origins"], task=task, metadata=dict(run_id=arm["run_id"], dataset=arm["dataset"], split="M_cal",
                model_route=arm["route_spec"], scope_binding=proof["scope_binding"], preflight_binding=proof["preflight_binding"],
                normalization_fit_end=parent["metadata"]["parent_fit_end"], scaler_sha256=parent["scaler_sha256"],
                new_F_inference=False, new_training_updates=0, new_V_forecasts=False, V_selection=False, T_access=False))
        check_statistics(stats, saved["score"], tolerance)
        with np.load(output, allow_pickle=False) as archive:
            for key in ("target_normalized", "observed", "origins", "channel_ids"):
                if not np.array_equal(archive[key], parent[key]):
                    raise ValueError("Saved diagnostic cohort/target/mask/channel identities changed")
            reloaded = independent_statistics(archive["pred_normalized"], archive["target_normalized"], archive["observed"], deadline=deadline)
        check_statistics(reloaded, saved["score"], tolerance)
        binding = common.binding(output, ctx)
        rows.append(row_metadata(arm, reloaded, parent, binding, head_state))
        evidence.append(dict(run_id=arm["run_id"], prediction_binding=binding, arrays=saved["arrays"], statistics=reloaded,
                             independent_fsum_vs_separate_numpy_sum="PASS", independent_reloaded_fsum="PASS"))
    if len(parent_cache) != 8:
        raise ValueError("Exactly the eight already frozen parent M_cal caches must be reused")
    if common.binding(events_path, ctx) != events_binding:
        raise ValueError("Primary EVENTS changed during this separate CPU diagnostic")
    deadline_check(deadline)
    if csv_path.exists():
        if report._read_csv(csv_path) != rows:
            raise ValueError("Preserve differing prior diagnostic CSV")
    else:
        report._csv(csv_path, rows)
    finished = time()
    deadline_check(deadline)
    result = dict(status="PASS", scope_binding=proof["scope_binding"], preflight_binding=proof["preflight_binding"],
        primary_plan_binding=scope["primary_plan_binding"], primary_validation_binding=proof["primary_validation_binding"],
        terminal_binding=proof["terminal_binding"], original_execution_start_binding=scope["original_execution_start_binding"],
        original_deadline_unix=deadline, CSV_binding=common.binding(csv_path, ctx), evidence=evidence,
        models=40, alpha_models=32, head_models=8, reused_parent_cache_count=8,
        parent_cache_bindings=[value["binding"] for _, value in sorted(parent_cache.items())],
        actual_parent_normalization="Recomputed exact original A60 scaler at M_cal start; cache scaler and head scaler/past contracts matched",
        data_access=ctx.get("data_access", {}), V_prefix_numeric_decoded_by_existing_bounded_loader=True,
        computations_scope="M_cal targets/past/fixed parents only; V metrics copied unchanged from completed SCORES",
        new_V_forecasts=False, V_selection=False, T_access=False, new_training_updates=0, new_F_inference=False,
        GPU_seconds_added=0., GPU_computation_performed=False, device="cpu", head_batch_size=64,
        primary_EVENTS_binding_before=events_binding, primary_EVENTS_binding_after=common.binding(events_path, ctx),
        primary_EVENTS_appended=False, started_unix=started, finished_unix=finished, wall_seconds=finished-started,
        original_wall_budget_remaining_seconds=deadline-finished,
        accounting="Separate CPU wall under original immutable deadline; do not add overlapping elapsed wall intervals as serial work",
        selected_metrics_scope="Preserved prior M_fit/M_select choice metrics; final full-M_cal errors are in-sample; V remains development evaluation")
    common.sealed_json(validation_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate",))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    ctx = common.context(args.asset_root, device="cpu")
    result = evaluate(ctx, args.scope, args.preflight)
    print(json.dumps(dict(status=result["status"], models=result["models"], new_training_updates=0, GPU_seconds_added=0.)))


if __name__ == "__main__":
    main()
