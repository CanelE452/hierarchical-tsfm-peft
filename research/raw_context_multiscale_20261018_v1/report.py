"""Independent numerical audit and public-only deterministic report replay."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import common


TABLES = ("RUNS", "COMPARISONS", "COMPLEMENTARITY", "TRAINING_CURVES",
          "CHANNEL_HORIZON_STATS", "COST", "COST_TRIALS", "JOB_STATUS",
          "T1_ACCURACY_COST", "T2_SINGLE_HYPOTHESIS", "T3_OUTPUT_COMBINATION", "PAIRED_DELTAS", "FIGURE_POINTS")
COLORS = dict(S="#0072B2", A="#009E73", B="#D55E00", Raw="#777777", G="#CC79A7",
              LoRA="#56B4E9", Ridge="#E69F00", alpha="#888888", conditional="#332288", past_control="#AA4499")


def independent_statistics(prediction, target, mask):
    """FP64 per-cell accumulation, independent of the producer metric helper."""
    p, y, observed = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if p.ndim != 3 or p.shape != y.shape or observed.shape != y.shape or observed.dtype != np.bool_:
        raise ValueError("Aligned original-channel prediction/target and bool mask required")
    if not np.isfinite(p).all() or not np.isfinite(y[observed]).all():
        raise ValueError("Nonfinite prediction or observed target")
    h, c = p.shape[1:]
    sse, sae, count = np.zeros((h, c)), np.zeros((h, c)), np.zeros((h, c), np.int64)
    for horizon in range(h):
        for channel in range(c):
            selected = observed[:, horizon, channel]
            difference = p[selected, horizon, channel].astype(np.float64)-y[selected, horizon, channel].astype(np.float64)
            sse[horizon, channel] = math.fsum(float(v)*float(v) for v in difference)
            sae[horizon, channel] = math.fsum(abs(float(v)) for v in difference)
            count[horizon, channel] = int(selected.sum())
    channels = count.sum(0)
    active = np.flatnonzero(channels)
    if not len(active):
        raise ValueError("No observed target")
    mse = math.fsum(math.fsum(sse[:, k].tolist())/int(channels[k]) for k in active)/len(active)
    mae = math.fsum(math.fsum(sae[:, k].tolist())/int(channels[k]) for k in active)/len(active)
    return dict(mse=mse, mae=mae, n_observed=int(channels.sum()), n_origins=len(p),
                horizon=h, channels=c, channel_counts=channels.tolist(),
                horizon_channel_sse=sse.tolist(), horizon_channel_sae=sae.tolist(), horizon_channel_count=count.tolist())


def earliest_selection(curves, *, score="astop_mse", epoch="epoch"):
    values = [(float(row[epoch]), float(row[score])) for row in curves]
    if (not values or len({v[0] for v in values}) != len(values)
            or not any(value[0] == 0 for value in values)
            or any(not np.isfinite(value).all() for value in values)):
        raise ValueError("A finite unique checkpoint curve including CP0 is required")
    return min(values, key=lambda value: (value[1], value[0]))


def independent_alpha(g, b, target, mask):
    g, b, y = (np.asarray(value, np.float64) for value in (g, b, target))
    mask = np.asarray(mask)
    if g.shape != b.shape or g.shape != y.shape or mask.shape != y.shape or g.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("Aligned fixed parent arrays and original bool mask required")
    counts = mask.sum(axis=(0, 1))
    products, squares = [], []
    for channel in np.flatnonzero(counts):
        observed = mask[:, :, channel]
        d = b[:, :, channel][observed]-g[:, :, channel][observed]
        e = y[:, :, channel][observed]-g[:, :, channel][observed]
        products.append(math.fsum((d*e).tolist())/int(counts[channel]))
        squares.append(math.fsum((d*d).tolist())/int(counts[channel]))
    denominator = math.fsum(squares)
    return float(np.clip(math.fsum(products)/denominator, 0, 1)) if denominator else 0.


def _parent_mcal(ctx, task, binding, spec, cache):
    identity = binding["sha256"]
    if identity in cache:
        return cache[identity]
    path = common.resolve_binding(binding, ctx)
    with np.load(path, allow_pickle=False) as archive:
        values = {key:archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")}
        metadata = json.loads(str(archive["metadata_json"].item()))
    m0, m1 = task["split_ranges"]["M_cal"]
    origins = np.arange(max(m0, task["L"]), m1-task["H"]+1, dtype=np.int64)
    rows = origins[:, None]+np.arange(task["H"])[None]
    mean, std, _ = ctx["modules"]["data"].fit_statistics(task, m0)
    mask = task["observed"][rows]
    target = np.where(mask, (task["physical"][rows]-mean)/std, 0).astype(np.float32)
    if (not np.array_equal(values["origins"], origins) or values["observed"].dtype != np.bool_
            or not np.array_equal(values["observed"], mask) or not np.array_equal(values["target_normalized"], target)
            or not np.array_equal(values["channel_ids"], np.asarray(task["columns"]))):
        raise ValueError("Fixed M_cal parent cache canonical cohort/target/mask changed")
    if (metadata["route_spec"] != spec or metadata["split"] != "M_cal" or metadata["parent_fit_end"] != m0
            or metadata["data_sha256"] != task["data_sha256"]
            or metadata["scaler_sha256"] != ctx["modules"]["fit"].scaler_fingerprint(mean, std)):
        raise ValueError("Fixed parent M_cal prediction metadata changed")
    if values["pred_normalized"].shape != target.shape or not np.isfinite(values["pred_normalized"]).all():
        raise ValueError("Fixed parent forecast shape or finite values changed")
    cache[identity] = values
    return values


def _verify_alpha(ctx, task, arm, cache):
    if arm["mode"] != "alpha":
        return
    g = _parent_mcal(ctx, task, arm["parent_g_prediction_binding"], arm["parent_g"]["route_spec"], cache)
    b = _parent_mcal(ctx, task, arm["parent_b_prediction_binding"], arm["parent_b"]["route_spec"], cache)
    m0, m1 = task["split_ranges"]["M_cal"]
    cut = m0+(3*(m1-m0))//5
    fit, select = g["origins"]+task["H"] <= cut, g["origins"] >= cut
    diagnostic = arm["diagnostic"]
    audit_key = ("alpha_audit", arm["parent_g_prediction_binding"]["sha256"], arm["parent_b_prediction_binding"]["sha256"], common.fingerprint(diagnostic))
    if audit_key in cache:
        refitted = cache[audit_key]
        expected = refitted if arm["alpha_label"] == "fitted" else {"alpha0":0., "alpha05":.5, "alpha1":1.}[arm["alpha_label"]]
        _close(expected, arm["alpha"], "deployed alpha")
        return
    if not fit.any() or not select.any() or np.any(fit & select):
        raise ValueError("Nonoverlapping M_fit/M_select targets required")
    fitted = independent_alpha(g["pred_normalized"][fit], b["pred_normalized"][fit],
        g["target_normalized"][fit], g["observed"][fit])
    refitted = independent_alpha(g["pred_normalized"], b["pred_normalized"], g["target_normalized"], g["observed"])
    _close(fitted, diagnostic["fit_alpha"]["alpha"], "M_fit analytic alpha")
    _close(refitted, diagnostic["refit_alpha"]["alpha"], "M_cal analytic alpha")
    for point in diagnostic["rows"]:
        expected_alpha = fitted if point["label"] == "fitted" else {"alpha0":0., "alpha05":.5, "alpha1":1.}[point["label"]]
        _close(expected_alpha, point["alpha"], "fixed/fitted alpha")
        for name, selected in (("M_fit", fit), ("M_select", select)):
            forecast = g["pred_normalized"][selected].astype(np.float64)+expected_alpha*(
                b["pred_normalized"][selected].astype(np.float64)-g["pred_normalized"][selected].astype(np.float64))
            score = independent_statistics(forecast, g["target_normalized"][selected], g["observed"][selected])
            for metric in ("mse", "mae"):
                _close(score[metric], point[name+"_"+metric], name+" "+metric)
    expected_deployed = refitted if arm["alpha_label"] == "fitted" else {"alpha0":0., "alpha05":.5, "alpha1":1.}[arm["alpha_label"]]
    _close(expected_deployed, arm["alpha"], "deployed alpha")
    cache[audit_key] = refitted


def _head_state_hash(state):
    result = hashlib.sha256()
    for key, value in state.items():
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _verify_head_artifact(ctx, task, arm, cache):
    if arm["mode"] not in ("conditional", "past_control"):
        return
    path = common.resolve_binding(arm["head_binding"], ctx)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(length=task["L"], horizon=task["H"], hidden=32)
    if (saved["head_config"] != config or saved["search"] != arm["search"] or saved["receipt"] != arm["refit"]
            or _head_state_hash(saved["state"]) != arm["refit"]["final_state_sha256"]):
        raise ValueError("Fresh head state/config/search/refit receipt differs")
    contract = saved["contract"]
    if (contract["data_sha256"] != task["data_sha256"] or contract["control"] != (arm["mode"] == "past_control")
            or contract["calibration_source_sha256"] != common.sha(Path(__file__).with_name("calibration.py"))):
        raise ValueError("Head calibration data/source/control contract changed")
    g = _parent_mcal(ctx, task, contract["g"], arm["parent_g"]["route_spec"], cache)
    if contract["origins"] != common.array_identity(g["origins"]):
        raise ValueError("Head fitting origins differ from fixed M_cal parents")
    if arm["mode"] == "past_control":
        if contract["b"] is not None:
            raise ValueError("Past control cannot require a B2 forecast cache")
    else:
        _parent_mcal(ctx, task, contract["b"], arm["parent_b"]["route_spec"], cache)
    past_key = ("canonical_head_past", arm["dataset"])
    if past_key not in cache:
        mean, std, _ = ctx["modules"]["data"].fit_statistics(task, task["split_ranges"]["M_cal"][0])
        cache[past_key] = canonical_past(task, g["origins"], mean, std)
    past = cache[past_key]
    if common.array_identity(past) != contract["past"]:
        raise ValueError("Head canonical filled/scaled physical histories changed")
    b = None if contract["b"] is None else cache[contract["b"]["sha256"]]["pred_normalized"]
    verify_head_checkpoints(saved, past, g["pred_normalized"], b, g["target_normalized"], g["observed"], g["origins"], task)


def canonical_past(task, origins, mean, std):
    """Independent per-window forward fill, with no observation before its left edge."""
    origins = np.asarray(origins, np.int64)
    c = len(task["columns"])
    current = np.broadcast_to(np.asarray(mean, np.float32), (len(origins), c)).copy()
    filled = np.empty((len(origins), task["L"], c), np.float32)
    for offset in range(task["L"]):
        rows = origins-task["L"]+offset
        current = np.where(task["observed"][rows], task["physical"][rows], current)
        filled[:, offset] = current
    return ((filled-mean)/std).astype(np.float32)


def verify_head_checkpoints(saved, past, g, b, target, mask, origins, task):
    from .calibration import predict_head
    m0, m1 = task["split_ranges"]["M_cal"]
    cut = m0+(3*(m1-m0))//5
    subsets = dict(M_fit=origins+task["H"] <= cut, M_select=origins >= cut)
    checkpoints = saved["search_checkpoints"]
    curves = saved["search"]["curves"]
    if len(checkpoints) != len(curves) or [r["endpoint_epochs"] for r in curves] != [0, .25, 1, 3]:
        raise ValueError("All registered M_select checkpoints including CP0 must be retained")
    control = saved["contract"]["control"]
    for checkpoint, curve in zip(checkpoints, curves):
        if checkpoint["curve"] != curve or _head_state_hash(checkpoint["state"]) != curve["checkpoint_state_sha256"]:
            raise ValueError("Head search checkpoint state/curve identity changed")
        for name, selected in subsets.items():
            prediction = predict_head(checkpoint["state"], past[selected], g[selected],
                None if control else b[selected], control=control, batch_size=saved["search"]["microbatch"])
            if curve["endpoint_epochs"] == 0 and not np.array_equal(prediction, g[selected]):
                raise ValueError("Head CP0 must reproduce its fixed G parent exactly")
            independent = independent_statistics(prediction, target[selected], mask[selected])
            for metric in ("mse", "mae", "n_observed"):
                _close(independent[metric], curve[name+"_"+metric], "independent "+name+" checkpoint "+metric)


def _close(actual, expected, name, *, atol=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=atol):
        raise ValueError("Independent value differs: "+name)


def _required(ctx):
    paths = [Path(ctx["report"])/name for name in ("MODELS_SEALED_BEFORE_V.json", "SCORES.json", "COST_INDEX.json")]
    if not all(path.is_file() for path in paths):
        raise ValueError("Report gate requires all sealed models, complete SCORES and complete COST_INDEX before prediction access")
    seal = common.read(paths[0])
    costs = common.read(paths[2])
    if seal.get("status") != "SEALED" or seal.get("V_used") is not False or costs.get("status") != "COMPLETE":
        raise ValueError("Model-before-V seal or complete cost index missing")
    scores = common.read(paths[1])
    if scores.get("status") != "COMPLETE" or scores.get("V_independent_test") is not False:
        raise ValueError("Complete development-V scores required")
    return seal, scores["arms"], costs["facets"]


def _artifact(ctx, spec):
    root = Path(ctx["cache"]).resolve()
    path = (root/spec["artifact_cache_relative_path"]).resolve()
    if not path.is_relative_to(root) or common.sha(path) != spec["artifact_sha256"]:
        raise ValueError("Deployment artifact cache path or SHA256 changed")
    return path


def _verify_route(ctx, task, spec):
    family = spec["family"]
    boundary = task["split_ranges"]["M_cal"][0]
    if family == "COMBINED":
        operation = spec["operation"]
        alpha = float(spec.get("alpha", .5))
        parents = []
        if operation != "alpha" or alpha != 1:
            parents.append(spec["g_route"])
        if operation == "conditional" or (operation == "alpha" and alpha != 0):
            parents.append(spec["b_route"])
        for parent in parents:
            _verify_route(ctx, task, parent)
        if operation != "alpha":
            path = Path(ctx["cache"])/spec["head_state_path"]
            if common.sha(path) != spec["head_state_sha256"]:
                raise ValueError("Fresh conditional head state changed")
        return
    if family == "RAW":
        if spec.get("canonical_fit_end", boundary) != boundary:
            raise ValueError("Raw canonical scaler scope changed")
        return
    path = _artifact(ctx, spec)
    if family == "RAW_RIDGE":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["data_sha"].item()) != task["data_sha256"]:
                raise ValueError("Ridge data identity changed")
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)
    norm, contract = saved["normalization"], saved["contract"]
    if norm["fit_end"] != boundary or norm.get("train_end", boundary) > boundary or contract["data_sha"] != task["data_sha256"]:
        raise ValueError("Parent/base fitting scope or data identity changed")
    state = common.fingerprint({key: ctx["modules"]["data"].array_sha256(value.numpy()) for key, value in saved["model"].items()})
    if state != saved["model_state_sha"]:
        raise ValueError("Parent/current saved model tensor hash changed")
    mean, std, _ = ctx["modules"]["data"].fit_statistics(task, boundary)
    if not np.array_equal(norm["mean"], mean) or not np.array_equal(norm["std"], std):
        raise ValueError("Parent/current canonical A60 scaler changed")
    scaler = ctx["modules"]["fit"].scaler_fingerprint(mean, std)
    if contract["scaler_sha"] != scaler:
        raise ValueError("Bundle scaler identity changed")
    axes = dict(L=int(task["L"]), H=int(task["H"]), C=len(task["columns"]), columns=list(task["columns"]),
                fold=saved["spec"].get("fold", "primary"), source_array_fingerprints=task.get("array_sha256", {}))
    if contract["config_sha"] != common.fingerprint(dict(spec=saved["spec"], axes=axes)):
        raise ValueError("Bundle config/axis identity differs from its training contract")
    if not all(isinstance(contract.get(key), str) and len(contract[key]) == 64
               for key in ("source_sha", "config_sha", "data_sha", "model_sha")):
        raise ValueError("Complete source/model/config/data identities required")
    new_sources = norm.get("new_scientific_source_hashes", {})
    for name, expected in new_sources.items():
        if common.sha(Path(ctx["repo"])/name) != expected:
            raise ValueError("Bound model producer source changed: "+name)


def _verify_prediction(ctx, task, arm):
    path = common.resolve_binding(arm["prediction_binding"], ctx)
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")}
        metadata = json.loads(str(archive["metadata_json"].item()))
    lo, hi = task["split_ranges"]["V_select"]
    origins = np.arange(max(lo, task["L"]), hi-task["H"]+1, task["H"], dtype=np.int64)
    if not np.array_equal(saved["origins"], origins) or not np.array_equal(saved["channel_ids"], np.asarray(task["columns"])):
        raise ValueError("Canonical V origins or original channel order changed")
    if metadata["route_spec"] != arm["route_spec"] or metadata["data_sha256"] != task["data_sha256"] or metadata["split"] != "V_select":
        raise ValueError("Prediction source/model/config/data metadata changed")
    boundary = task["split_ranges"]["M_cal"][0]
    mean, std, _ = ctx["modules"]["data"].fit_statistics(task, boundary)
    if metadata["scaler_sha256"] != ctx["modules"]["fit"].scaler_fingerprint(mean, std):
        raise ValueError("Prediction scaler binding changed")
    if metadata["parent_fit_end"] != boundary or metadata.get("T_access") is not False:
        raise ValueError("Prediction parent scope or protected data policy changed")
    if metadata["forecast_source_sha256"] != common.sha(Path(__file__).with_name("pipeline.py")):
        raise ValueError("Forecast producer source changed")
    rows = origins[:, None]+np.arange(task["H"])[None]
    mask = task["observed"][rows]
    target = np.where(mask, (task["physical"][rows]-mean)/std, 0).astype(np.float32)
    if saved["observed"].dtype != np.bool_ or not np.array_equal(mask, saved["observed"]) or not np.array_equal(target, saved["target_normalized"]):
        raise ValueError("Canonical original target or bool mask changed")
    stats = independent_statistics(saved["pred_normalized"], target, mask)
    for key in ("mse", "mae", "n_observed", "n_origins", "horizon", "channels"):
        _close(stats[key], arm["score"][key], key)
    for key in ("horizon_channel_sse", "horizon_channel_sae", "horizon_channel_count", "channel_counts"):
        if not np.allclose(stats[key], arm["score"][key], rtol=1e-13, atol=1e-12):
            raise ValueError("Producer sufficient statistics changed: "+key)
    for key, value in saved.items():
        if common.array_identity(value) != arm["array_identities"][key]:
            raise ValueError("Private prediction array identity changed: "+key)
    _verify_route(ctx, task, arm["route_spec"])
    return stats


def verify_cost(receipt, arm, *, trial_csv=None, pid_identities=None):
    row, trials = receipt["row"], receipt["trials"]
    if receipt.get("status") != "COMPLETE" or row.get("cost_valid") is not True or row.get("fresh_process_cli") is not True:
        raise ValueError("Valid fresh-process cost receipt required")
    identity = (row["process_pid"], float(row["process_create_time"]))
    if pid_identities is not None:
        if identity in pid_identities:
            raise ValueError("Cost facets must have distinct PID/create-time identities")
        pid_identities.add(identity)
    if row["accuracy_run_id"] != arm["run_id"] or row["dataset"] != arm["dataset"]:
        raise ValueError("Cost/accuracy identity differs")
    if sum(v["phase"] == "warmup" for v in trials) != 10 or sum(v["phase"] == "timed" for v in trials) != 20:
        raise ValueError("Registered ten warmups and twenty timed trials required")
    if trial_csv is not None:
        if common.sha(trial_csv) != receipt["trials_sha256"]:
            raise ValueError("Raw cost CSV SHA256 changed")
        with Path(trial_csv).open(newline="", encoding="utf8") as stream:
            raw = list(csv.DictReader(stream))
        if len(raw) != len(trials):
            raise ValueError("Raw cost trial count changed")
        for actual, saved in zip(raw, trials):
            for key, value in saved.items():
                expected = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else "" if value is None else str(value)
                if actual[key] != expected:
                    raise ValueError("Raw CSV trial differs from receipt: "+key)
    timed = [v for v in trials if v["phase"] == "timed"]
    ms = np.array([v["whole_seconds"]*1000 for v in timed], np.float64)
    if not np.isfinite(ms).all() or np.any(ms <= 0):
        raise ValueError("Positive finite timed trial duration required")
    q25, median, q75, p95 = np.percentile(ms, [25, 50, 75, 95])
    for key, value in (("median_batch_ms", median), ("p50_batch_ms", median), ("IQR_batch_ms", q75-q25),
                       ("p95_batch_ms", p95), ("median_ms_per_origin", median/row["batch_size"]),
                       ("p95_ms_per_origin", p95/row["batch_size"])):
        _close(row[key], value, key)
    for key in ("MSE", "MAE"):
        _close(row[key], arm["score"][key.lower()], key, atol=common.ATOL)
    parity = receipt["parity"]
    if parity.get("status") != "PASS" or parity.get("saved_reference") != "PASS" or parity["n_origins"] != arm["score"]["n_origins"]:
        raise ValueError("Full saved-reference V deployment parity required")
    if parity["origins"] != arm["array_identities"]["origins"]:
        raise ValueError("Cost whole-V origin identity differs")
    if parity.get("atol") != common.ATOL or parity.get("rtol") != common.RTOL:
        raise ValueError("Cost parity tolerance changed")
    for trial in trials:
        calls, streams = 0, 0
        for invocation in trial["F_invocations"]:
            batch, length, width = invocation["input_shape"]
            if batch != row["batch_size"] or invocation["requested_context"] != length or invocation["actual_context"] != length:
                raise ValueError("Actual F input/time lengths differ")
            expected = math.ceil(batch*width/128)
            if (invocation["F_calls"] != expected or invocation["F_physical_streams"] != batch*width
                    or invocation["F_logical_streams"] != batch*width):
                raise ValueError("Actual F physical call/stream count differs")
            if invocation["wrapper_horizon"] != row["H"]:
                raise ValueError("Shared foundation must retain full task H")
            calls += expected
            streams += batch*width
        if trial["F_calls"] != calls or trial["F_physical_streams"] != streams or trial["F_logical_streams"] != streams:
            raise ValueError("Trial F work aggregate differs")
    mode = arm["mode"]
    invocations = timed[0]["F_invocations"]
    if mode in ("A", "B", "S"):
        lengths = [v["requested_context"] for v in invocations]
        expected_lengths = [512, 256] if mode == "B" else [512, 512] if row["schedule"] == "two_call" else [512]
        if lengths != expected_lengths:
            raise ValueError("Native/matched schedule branch lengths differ")
        if mode == "B" and [v["used_horizon"] for v in invocations] != [row["H"], row["H"]//2]:
            raise ValueError("Coarse F future slice must retain its fixed H/2 semantics")
    if mode in ("G", "Ridge", "past_control") and invocations:
        raise ValueError("Direct/control path must not invoke F")
    if mode == "Ridge" and (row["device"] != "cpu" or row["comparison_scope"] != "CPU_RIDGE_CONTEXT"):
        raise ValueError("Ridge must retain separate CPU context")
    for key in ("peak_allocated_MiB", "peak_reserved_MiB"):
        values = [v[key] for v in timed if v[key] is not None]
        if values:
            _close(row[key], max(values), key)
        elif row[key] is not None:
            raise ValueError("Unavailable allocator memory must remain null")
    if row["process_GPU_sampling_status"] != "AVAILABLE" and row["process_GPU_sampled_peak_MiB"] is not None:
        raise ValueError("Unavailable per-PID memory cannot be replaced by device totals")
    return row


def _verify_selection(arm):
    if arm["mode"] not in ("S", "A", "B"):
        return
    epoch, score = earliest_selection(arm["curves"])
    _close(epoch, arm["search_summary"]["selected_epoch"], "selected A_stop epoch")
    _close(score, arm["search_summary"]["selected_astop_mse"], "selected A_stop MSE")
    if "refit_summary" in arm:
        if arm["refit_summary"].get("stop_origins", 0) != 0:
            raise ValueError("Fresh A60 refit must not select from A_stop")


def _verify_head(arm, task):
    if arm["mode"] not in ("conditional", "past_control"):
        return
    search, refit = arm["search"], arm["refit"]
    epoch, score = earliest_selection(search["curves"], score="M_select_mse", epoch="endpoint_epochs")
    _close(epoch, search["selected_epoch"], "head M_select epoch")
    _close(score, search["selected_M_select_mse"], "head M_select MSE")
    if (search["checkpoint_split"] != "M_select" or search["fit_split"] != "M_fit"
            or refit["fit_split"] != "M_cal_dense_stride1" or refit["fresh_initialization"] is not True
            or refit["initialization_sha256"] != search["initialization_sha256"]
            or refit["selected_epoch"] != epoch or arm.get("V_used") is not False):
        raise ValueError("Head must select within M_cal then fresh fit all M_cal")
    m0, m1 = task["split_ranges"]["M_cal"]
    cut = m0+(3*(m1-m0))//5
    if search["mcut"] != cut or refit["n_fit"] != m1-task["H"]+1-max(m0, task["L"]):
        raise ValueError("Head M_cal target range or split changed")
    for name in ("parent_g", "parent_b"):
        parent = arm.get(name)
        if parent is not None and (parent["parent_fit_end"] > m0
                or parent.get("M_cal_targets_used_for_parent_fitting_or_selection", False)):
            raise ValueError("Head parent was not fixed before M_cal")


def _audit(ctx):
    seal, arms, facets = _required(ctx)
    from .run import load_registered_plan, registered_plan_path
    plan = load_registered_plan(ctx)
    for name, expected_hash in plan["source_files"].items():
        if common.sha(Path(ctx["repo"])/name) != expected_hash:
            raise ValueError("Registered experiment producer source changed: "+name)
    tasks = {dataset: common.load_task(ctx, dataset) for dataset in common.EXPECTED}
    ids = [arm["run_id"] for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scored model identities")
    calibration_path = Path(ctx["cache"])/"CALIBRATION_MODELS.json"
    calibration_models = common.read(calibration_path)
    if {a["run_id"] for a in arms if a["mode"] in ("alpha", "conditional", "past_control")} != {a["run_id"] for a in calibration_models}:
        raise ValueError("All sealed alpha/head models must remain in the report")
    expected_calibration = {(dataset, "alpha", seed, role, label) for dataset in common.EXPECTED
        for seed in common.SEEDS for role in ("Raw", "G") for label in ("alpha0", "alpha05", "alpha1", "fitted")}
    expected_calibration |= {(dataset, mode, seed, None, None) for dataset in common.EXPECTED
        for seed in common.SEEDS for mode in ("conditional", "past_control")}
    actual_calibration = {(a["dataset"], a["mode"], a["seed"], a.get("base_role"), a.get("alpha_label"))
                          for a in arms if a["mode"] in ("alpha", "conditional", "past_control")}
    if actual_calibration != expected_calibration:
        raise ValueError("Registered eight parent pairs and eight head configurations are required")
    baselines = {(a["dataset"], a["mode"]) for a in arms if a["mode"] not in ("S", "A", "B", "alpha", "conditional", "past_control")}
    if baselines != {("jena", "Raw"), ("jena", "G"), ("jena", "LoRA"), ("jena", "Ridge"), ("pems32", "Raw"), ("pems32", "G")}:
        raise ValueError("All registered practical and historical baselines are required")
    selection = common.read(common.resolve_binding(seal["selection_binding"], ctx))
    transfer = selection["transfer"]
    body = {(arm["dataset"], arm["mode"], arm["seed"]) for arm in arms if arm["mode"] in ("S", "A", "B")}
    expected = {("jena", mode, seed) for mode in ("S", "A", "B") for seed in common.SEEDS}
    expected |= {("pems32", mode, seed) for mode in ("S", transfer["selected_mode"]) for seed in common.SEEDS}
    if body != expected or selection.get("V_used") is not False or transfer.get("V_used") is not False:
        raise ValueError("Registered S/A/B two-seed model coverage or pre-V selection incomplete")
    means = {mode: math.fsum(float(a["search_summary"]["selected_astop_mse"]) for a in arms
                            if a["dataset"] == "jena" and a["mode"] == mode)/2 for mode in ("A", "B")}
    chosen = "A" if means["A"] <= means["B"] else "B"
    if chosen != transfer["selected_mode"] or means != transfer["mean_astop_mse"]:
        raise ValueError("Two-seed Jena A_stop-only transfer with exact tie A changed")
    models = {a["run_id"]: a for a in arms}
    if {a["run_id"] for a in seal["finals"]} != {a["run_id"] for a in arms if a["mode"] in ("S", "A", "B")}:
        raise ValueError("Scored bodies differ from the before-V model seal")
    for sealed in seal["finals"]:
        if sealed["route_spec"] != models[sealed["run_id"]]["route_spec"]:
            raise ValueError("Deployment model changed after the before-V seal")
    stats, calibration_cache = {}, {}
    for arm in arms:
        _verify_selection(arm)
        if arm["mode"] in ("S", "A", "B"):
            cap = selection["groups"][arm["dataset"]]["cap"]
            if max(float(v["epoch"]) for v in arm["curves"]) != cap:
                raise ValueError("Matched models received different search cap opportunities")
        _verify_head(arm, tasks[arm["dataset"]])
        _verify_head_artifact(ctx, tasks[arm["dataset"]], arm, calibration_cache)
        _verify_alpha(ctx, tasks[arm["dataset"]], arm, calibration_cache)
        stats[arm["run_id"]] = _verify_prediction(ctx, tasks[arm["dataset"]], arm)
    for dataset, group in selection["groups"].items():
        if group["extension_required"] and not group["extension_executed"] and not any(
                item["dataset"] == dataset and item["reason"] == "BUDGET_LIMITED_CONVERGENCE_UNRESOLVED"
                for item in selection.get("resume_queue", [])):
            raise ValueError("Budget-limited matched extension needs an explicit unresolved resume record")
    required_facets = {(a["run_id"], schedule, batch) for a in arms for batch in (1, 8)
                       for schedule in (("native", "two_call") if a["mode"] == "A" else ("native",))}
    actual_facets = {(f["run_id"], f["schedule"], f["batch"]) for f in facets}
    if actual_facets != required_facets or len(actual_facets) != len(facets):
        raise ValueError("All native B1/B8 and A matched two-call cost facets required")
    pid_identities, costs, trials = set(), [], []
    for facet in facets:
        path = common.resolve_binding(facet["output"], ctx)
        receipt = common.read(path)
        request_path = common.resolve_binding(receipt["request_binding"], ctx)
        request = common.read(request_path)
        contract = common.fingerprint({key:value for key, value in request.items() if not key.startswith("_")})
        arm = models[facet["run_id"]]
        expected_spec = dict(arm["route_spec"], schedule="two_call") if facet["schedule"] == "two_call" else arm["route_spec"]
        if (receipt["request_contract_sha256"] != contract or receipt["route_spec"] != request["route_spec"]
                or request["route_spec"] != expected_spec or request["accuracy_run_id"] != arm["run_id"]
                or receipt["prediction_reference_sha256"] != arm["prediction_binding"]["sha256"]
                or request["prediction_sha256"] != arm["prediction_binding"]["sha256"]):
            raise ValueError("Exact cost request/model/prediction provenance differs")
        for name, expected_hash in receipt["producer_sources"].items():
            if common.sha(Path(ctx["repo"])/name) != expected_hash:
                raise ValueError("Cost producer source changed: "+name)
        if not receipt.get("producer_environment"):
            raise ValueError("Cost producer runtime/environment record is required")
        row = verify_cost(receipt, models[facet["run_id"]], trial_csv=path.with_suffix(".trials.csv"), pid_identities=pid_identities)
        if (row["batch_size"], row["schedule"]) != (facet["batch"], facet["schedule"]):
            raise ValueError("Cost index facet differs from measured schedule/batch")
        costs.append(row)
        task = tasks[arm["dataset"]]
        lo, hi = task["split_ranges"]["V_select"]
        origins = np.arange(max(lo, task["L"]), hi-task["H"]+1, task["H"], dtype=np.int64)
        first = origins[np.arange(8) % len(origins)]
        for trial in receipt["trials"]:
            expected_origins = first.tolist() if row["batch_size"] == 8 else [int(first[trial["trial"] % 8])]
            if trial["origins"] != expected_origins:
                raise ValueError("Timed origin bundle differs from the fixed matched cohort")
        trials.extend(dict(cost_id=row["cost_id"], **v) for v in receipt["trials"])
    smoke_files = sorted(Path(ctx["report"]).glob("VALIDATION_*GPU*.json"))
    current_smokes = []
    for path in smoke_files:
        value = common.read(path)
        if value.get("status") != "PASS" or not value.get("device", "").startswith("cuda"):
            continue
        sources = value.get("scientific_source_hashes", {})
        if sources and all(common.sha(Path(ctx["repo"])/name) == digest for name, digest in sources.items()):
            current_smokes.append(common.binding(path, ctx))
    if not current_smokes:
        raise ValueError("Actual-F initialization/grid/gradient/deployment minimum gates must pass")
    return dict(arms=arms, stats=stats, costs=costs, trials=trials, selection=selection,
                channel_ids={name: list(task["columns"]) for name, task in tasks.items()},
                actual_F_gate_bindings=current_smokes,
                manifest_bindings={**{name: common.binding(Path(ctx["report"])/name, ctx)
                    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json")},
                    "registered_plan": common.binding(registered_plan_path(ctx), ctx)})


def verify(ctx):
    try:
        data = _audit(ctx)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        result = dict(status="FAIL", error_type=type(error).__name__, reason=str(error),
                      T_access=False, V_independent_test=False)
        common.atomic_json(Path(ctx["report"])/"VALIDATION_REPORT.json", result)
        raise
    result = dict(status="PASS", manifest_bindings=data["manifest_bindings"],
        actual_F_gate_bindings=data["actual_F_gate_bindings"],
        models=len(data["arms"]), cost_facets=len(data["costs"]),
        checks=dict(independent_FP64_rescore="PASS", canonical_target_mask_origin_channel_scaler="PASS",
            source_model_config_data_bundle_identity="PASS", earliest_A_stop_including_CP0="PASS",
            transfer_two_seed_A_stop_tie_A="PASS", M_cal_head_selection_fresh_fit="PASS",
            all_whole_V_saved_reference_parity="PASS", raw_cost_trials_statistics="PASS",
            fresh_PID_create_time_per_facet="PASS", actual_F_work_lengths="PASS",
            allocator_reserved_per_PID_memory_distinct="PASS", minimum_actual_F_gates="PASS"),
        required_gates=dict(G1_data_masks_origins="PASS: canonical arrays and source-bound actual-F future canary/batch independence",
            G2_shapes_original_time_grid="PASS: actual 512/256 invocations and original-grid horizon/lag contract; toy regressions are identified by the recorded unit suite",
            G3_identical_control_initialization="PASS: source-bound actual-F initialization and A/B(s1) output/gradient parity",
            G4_gradient_shared_F="PASS: source-bound actual-F gradient and frozen-weight gates",
            G5_selection_cache_scope="PASS: SHA-bound pre-M_cal parents; independent alpha/head selection replay; before-V model seal"),
        accumulation="independent FP64 channel/horizon SSE SAE count; observed equal-channel macro",
        independence="Scoring, alpha algebra and window fill are independent; saved head checkpoint inference reuses the pure existing head forward",
        T_access=False, V_independent_test=False, overlap_is_not_independent_replication=True,
        convergence_scope="Budget-limited cap results remain unresolved when the last checkpoint is best",
        publication_verification="No remote publication claim is made by this numerical audit")
    common.atomic_json(Path(ctx["report"])/"VALIDATION_REPORT.json", result)
    ctx["_report_verified_data"] = data
    return result


def _comparison(candidate, reference, contrast, *, schedule="accuracy"):
    result = dict(dataset=candidate["dataset"], run_id=candidate["run_id"], reference_run_id=reference["run_id"],
                  mode=candidate["mode"], reference_mode=reference["mode"], seed=candidate.get("seed"),
                  contrast=contrast, schedule=schedule, evaluation="V_development")
    for metric in ("mse", "mae"):
        actual, base = float(candidate["score"][metric]), float(reference["score"][metric])
        result[metric] = actual
        result["reference_"+metric] = base
        result["delta_"+metric] = actual-base
        result["delta_"+metric+"_percent"] = 100*(actual-base)/base if base else None
    return result


def _public_tables(data):
    arms, stats, costs = data["arms"], data["stats"], data["costs"]
    runs, curves, sufficient, complementarity, jobs = [], [], [], [], []
    for arm in arms:
        mode, identifier = arm["mode"], arm["run_id"]
        score = stats[identifier]
        selection = arm.get("search_summary", {})
        run = dict(run_id=identifier, dataset=arm["dataset"], mode=mode, seed=arm.get("seed"),
            row_kind="model", mse=score["mse"], mae=score["mae"], n_origins=score["n_origins"],
            n_observed=score["n_observed"], evaluation="V_development", V_independent_test=False,
            C=score["channels"], L=512, H=score["horizon"], K=common.EXPECTED[arm["dataset"]]["K"] if mode in ("S", "A", "B") else None,
            K_split="half+half" if mode in ("A", "B") else "all_context" if mode == "S" else None,
            scales="1+2" if mode == "B" else "1+1" if mode == "A" else "1",
            reused=bool(arm.get("reused", False)), selected_A_stop_epoch=selection.get("selected_epoch"),
            A_stop_mse=selection.get("selected_astop_mse"),
            selected_M_select_epoch=arm.get("search", {}).get("selected_epoch"),
            M_select_mse=arm.get("search", {}).get("selected_M_select_mse"),
            M_cal_fit_mse=arm.get("refit", {}).get("mean_batch_loss"),
            alpha=arm.get("alpha"), alpha_label=arm.get("alpha_label"), base_role=arm.get("base_role"),
            status="COMPLETE", convergence="BUDGET_LIMITED_CONVERGENCE_UNRESOLVED" if arm.get("curves") and
                earliest_selection(arm["curves"])[0] == max(float(r["epoch"]) for r in arm["curves"]) else "selected_within_registered_budget")
        runs.append(run)
        jobs.append(dict(run_id=identifier, dataset=arm["dataset"], phase="reuse" if arm.get("reused") else "final_model",
                         status="COMPLETE", reused=bool(arm.get("reused")), reason=run["convergence"]))
        for point in arm.get("curves", []):
            curves.append(dict(run_id=identifier, dataset=arm["dataset"], mode=mode, seed=arm.get("seed"),
                phase="A_fit_search", selected=float(point["epoch"]) == float(run["selected_A_stop_epoch"]), **point))
        for point in arm.get("search", {}).get("curves", []):
            curves.append(dict(run_id=identifier, dataset=arm["dataset"], mode=mode, seed=arm.get("seed"),
                phase="M_fit_head_search", selected=float(point["endpoint_epochs"]) == float(run["selected_M_select_epoch"]), **point))
        for key, phase in (("search_training_segments", "A_fit_training_segment"), ("refit_training_segments", "A60_refit_training_segment")):
            for point in arm.get(key, []):
                curves.append(dict(point, run_id=identifier, dataset=arm["dataset"], mode=mode, seed=arm.get("seed"),
                    phase=phase, epoch=point["endpoint_epoch"], exposures=point["end_exposures"], selected=False,
                    TRAIN_segment_batch_loss=point.get("mean_observed_training_batch_loss"),
                    current_checkpoint_full_TRAIN_MSE=None))
        counts = np.asarray(score["horizon_channel_count"])
        sse, sae = np.asarray(score["horizon_channel_sse"]), np.asarray(score["horizon_channel_sae"])
        for h in range(score["horizon"]):
            for c in range(score["channels"]):
                sufficient.append(dict(run_id=identifier, dataset=arm["dataset"], mode=mode, seed=arm.get("seed"),
                    horizon=h+1, channel=c, channel_id=data.get("channel_ids", {}).get(arm["dataset"], list(range(score["channels"])))[c],
                    sse=float(sse[h, c]), sae=float(sae[h, c]), count=int(counts[h, c])))
        if mode == "alpha":
            for point in arm["diagnostic"]["rows"]:
                if point["label"] == arm["alpha_label"]:
                    complementarity.append(dict(run_id=identifier, dataset=arm["dataset"], seed=arm["seed"],
                        mode=mode, base_role=arm["base_role"], V_mse=score["mse"], V_mae=score["mae"],
                        M_cal_alpha=arm["alpha"], **point))
        elif mode in ("conditional", "past_control"):
            chosen = min(arm["search"]["curves"], key=lambda r:(r["M_select_mse"], r["endpoint_epochs"]))
            complementarity.append(dict(run_id=identifier, dataset=arm["dataset"], seed=arm["seed"], mode=mode,
                V_mse=score["mse"], V_mae=score["mae"], M_cal_batch_fit_loss=arm["refit"]["mean_batch_loss"],
                M_cal_refit_is_full_evaluation=False, **chosen))
    groups = {}
    for row in runs:
        key = (row["dataset"], row["mode"], row["base_role"], row["alpha_label"])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items(), key=lambda p:str(p[0])):
        if len(rows) == 2 and {r["seed"] for r in rows} == set(common.SEEDS):
            summary = dict(rows[0], run_id="mean_"+"_".join(str(v) for v in key if v is not None),
                           seed="mean", row_kind="two_seed_mean", mse=math.fsum(r["mse"] for r in rows)/2,
                           mae=math.fsum(r["mae"] for r in rows)/2)
            runs.append(summary)
    comparisons = []
    for arm in arms:
        for reference in arms:
            if reference["dataset"] != arm["dataset"] or arm is reference:
                continue
            matched = reference.get("seed") == arm.get("seed")
            pair = (reference["mode"], arm["mode"])
            if matched and pair in (("S", "A"), ("S", "B"), ("A", "B"), ("past_control", "conditional")):
                comparisons.append(_comparison(arm, reference, reference["mode"]+"_to_"+arm["mode"]))
            elif arm["mode"] in ("S", "A", "B") and reference["mode"] in ("Raw", "G"):
                comparisons.append(_comparison(arm, reference, "versus_"+reference["mode"]))
    paired = []
    for comparison in comparisons:
        if comparison["contrast"] not in ("S_to_A", "A_to_B", "S_to_B"):
            continue
        candidate, reference = stats[comparison["run_id"]], stats[comparison["reference_run_id"]]
        counts = np.asarray(candidate["horizon_channel_count"])
        for metric, key in (("mse", "horizon_channel_sse"), ("mae", "horizon_channel_sae")):
            delta = np.asarray(candidate[key])-np.asarray(reference[key])
            for channel in range(candidate["channels"]):
                n = int(counts[:, channel].sum())
                paired.append(dict(dataset=comparison["dataset"], contrast=comparison["contrast"], seed=comparison["seed"],
                    metric=metric, axis="channel", index=channel, delta=float(delta[:, channel].sum()/n) if n else None,
                    count=n, run_id=comparison["run_id"], reference_run_id=comparison["reference_run_id"]))
            for label, indices in zip(("near", "middle", "far"), np.array_split(np.arange(candidate["horizon"]), 3)):
                n = counts[indices].sum(0)
                active = n > 0
                value = float(np.mean(delta[indices].sum(0)[active]/n[active])) if active.any() else None
                paired.append(dict(dataset=comparison["dataset"], contrast=comparison["contrast"], seed=comparison["seed"],
                    metric=metric, axis="horizon_third", index=label, delta=value, count=int(n.sum()),
                    run_id=comparison["run_id"], reference_run_id=comparison["reference_run_id"]))
    t1 = []
    for cost in costs:
        arm = next(a for a in arms if a["run_id"] == cost["accuracy_run_id"])
        if arm["mode"] not in ("S", "A", "B", "Raw", "G", "LoRA", "Ridge"):
            continue
        columns = ("cost_id", "accuracy_run_id", "dataset", "device", "memory_layer", "batch_size", "B", "C", "K", "L", "H",
            "schedule", "MSE", "MAE", "median_batch_ms", "IQR_batch_ms", "p95_batch_ms", "median_ms_per_origin", "p95_ms_per_origin",
            "peak_allocated_MiB", "peak_reserved_MiB", "process_GPU_sampled_peak_MiB", "process_GPU_sampling_status",
            "CPU_RSS_sampled_peak_MiB", "resident_parameters", "trainable_parameters", "resident_weights_gpu_MiB", "resident_weights_cpu_MiB", "F_calls")
        row = dict({key:cost.get(key) for key in columns}, mode=arm["mode"], seed=arm.get("seed"), row_kind="model")
        for reference_mode in ("S", "Raw"):
            reference = next((a for a in arms if a["dataset"] == arm["dataset"] and a["mode"] == reference_mode
                and (reference_mode == "Raw" or a.get("seed") == arm.get("seed"))), None)
            if reference is not None:
                row["reference_"+reference_mode+"_run_id"] = reference["run_id"]
                for metric in ("mse", "mae"):
                    denominator = reference["score"][metric]
                    row["delta_"+metric+"_vs_"+reference_mode+"_percent"] = 100*(arm["score"][metric]-denominator)/denominator if denominator else None
        g = next((a for a in arms if a["dataset"] == arm["dataset"] and a["mode"] == "G"), None)
        gcost = next((c for c in costs if g and c["accuracy_run_id"] == g["run_id"]
            and c["device"] == cost["device"] and c["batch_size"] == cost["batch_size"] and c["memory_layer"] == cost["memory_layer"]), None)
        row["G_dominates_MSE_latency_allocated"] = bool(gcost and cost["device"].startswith("cuda") and
            all(gcost[key] <= cost[key] for key in ("MSE", "median_batch_ms", "peak_allocated_MiB")) and
            any(gcost[key] < cost[key] for key in ("MSE", "median_batch_ms", "peak_allocated_MiB")))
        t1.append(row)
    for dataset in common.EXPECTED:
        for mode in ("S", "A", "B"):
            for batch in (1, 8):
                for schedule in ("native", "two_call"):
                    paired_rows = [row for row in t1 if row["dataset"] == dataset and row["mode"] == mode
                                   and row["batch_size"] == batch and row["schedule"] == schedule and row["row_kind"] == "model"]
                    if len(paired_rows) != 2 or {row["seed"] for row in paired_rows} != set(common.SEEDS):
                        continue
                    summary = dict(paired_rows[0], cost_id=f"mean_{dataset}_{mode}_{schedule}_B{batch}", accuracy_run_id=None,
                                   seed="mean", row_kind="two_seed_mean")
                    for key in tuple(summary):
                        values = [row.get(key) for row in paired_rows]
                        if key.startswith("reference_"):
                            summary[key] = None
                        elif key == "G_dominates_MSE_latency_allocated":
                            summary[key] = all(values)
                        elif key in ("MSE", "MAE", "median_batch_ms", "IQR_batch_ms", "p95_batch_ms", "median_ms_per_origin", "p95_ms_per_origin",
                                "peak_allocated_MiB", "peak_reserved_MiB", "process_GPU_sampled_peak_MiB", "CPU_RSS_sampled_peak_MiB") or key.startswith("delta_"):
                            summary[key] = math.fsum(values)/2 if all(v is not None for v in values) else None
                    t1.append(summary)
    t2 = [dict(row, changed_operation="raw/context projection split" if row["contrast"] == "S_to_A" else
               "context averaging + coarse F forecast + original-grid block repeat") for row in comparisons if row["contrast"] in ("S_to_A", "A_to_B")]
    for b in (c for c in costs if next(a for a in arms if a["run_id"] == c["accuracy_run_id"])["mode"] == "B"):
        arm_b = next(a for a in arms if a["run_id"] == b["accuracy_run_id"])
        arm_a = next((a for a in arms if a["dataset"] == arm_b["dataset"] and a["mode"] == "A" and a["seed"] == arm_b["seed"]), None)
        if arm_a is None:
            continue
        for a in (c for c in costs if c["accuracy_run_id"] == arm_a["run_id"] and c["batch_size"] == b["batch_size"] and c["device"] == b["device"]):
            t2.append(dict(dataset=arm_b["dataset"], run_id=arm_b["run_id"], reference_run_id=arm_a["run_id"], seed=arm_b["seed"],
                contrast="A_to_B_cost", schedule="matched_two_call" if a["schedule"] == "two_call" else "native_deployment",
                batch_size=b["batch_size"], device=b["device"], memory_layer=b["memory_layer"],
                latency_B_over_A=b["median_batch_ms"]/a["median_batch_ms"],
                allocated_B_minus_A=b["peak_allocated_MiB"]-a["peak_allocated_MiB"]))
    t3 = []
    for cost in costs:
        arm = next(a for a in arms if a["run_id"] == cost["accuracy_run_id"])
        if arm["mode"] not in ("alpha", "conditional", "past_control", "Raw", "G", "S"):
            continue
        row = dict(cost, mode=arm["mode"], seed=arm.get("seed"),
                   head_search_seconds=arm.get("head_search_seconds"), head_refit_seconds=arm.get("head_refit_seconds"),
                   head_only_inference_ms=None, head_only_inference_scope="No instrumented stage time is substituted for whole-path latency")
        if arm["mode"] in ("alpha", "conditional", "past_control"):
            for name, role in (("g", arm.get("base_role", "G")), ("b", "S")):
                if name == "b" and (arm["mode"] == "past_control" or (arm["mode"] == "alpha" and arm["alpha"] == 0)):
                    continue
                if name == "g" and arm["mode"] == "alpha" and arm["alpha"] == 1:
                    continue
                bound_spec = arm["route_spec"][name+"_route"]
                def same_model(left, right):
                    if left["family"] != right["family"]:
                        return False
                    if left["family"] == "RAW":
                        return left.get("canonical_fit_end") == right.get("canonical_fit_end")
                    return left.get("artifact_sha256") == right.get("artifact_sha256")
                parent = next((a for a in arms if a["dataset"] == arm["dataset"] and same_model(a["route_spec"], bound_spec)), None)
                if parent is None and name == "b":
                    parent = next((a for a in arms if a["dataset"] == arm["dataset"] and a["mode"] == "alpha"
                        and a["alpha"] == 1 and same_model(a["route_spec"]["b_route"], bound_spec)), None)
                if parent is None:
                    row["parent_"+name+"_standalone_status"] = "NOT_MEASURED_SEPARATELY"
                    continue
                parent_cost = next(c for c in costs if c["accuracy_run_id"] == parent["run_id"] and
                    c["batch_size"] == cost["batch_size"] and c["device"] == cost["device"] and c["schedule"] == "native")
                row["parent_"+name+"_cost_id"] = parent_cost["cost_id"]
                row["parent_"+name+"_standalone_batch_ms"] = parent_cost["median_batch_ms"]
                row["parent_"+name+"_standalone_status"] = "EXACT_MODEL_ENDPOINT" if parent["mode"] == "alpha" else "EXACT_MODEL"
        t3.append(row)
    result = dict(RUNS=runs, COMPARISONS=comparisons, COMPLEMENTARITY=complementarity, TRAINING_CURVES=curves,
                CHANNEL_HORIZON_STATS=sufficient, COST=costs, COST_TRIALS=data["trials"], JOB_STATUS=jobs,
                T1_ACCURACY_COST=t1, T2_SINGLE_HYPOTHESIS=t2, T3_OUTPUT_COMBINATION=t3, PAIRED_DELTAS=paired)
    result["FIGURE_POINTS"] = figure_points(result)
    return result


def figure_points(tables):
    """Declare each plotted cost facet, including coincident endpoint forecasts."""
    models = {row["run_id"]: row for row in tables["RUNS"] if row["row_kind"] == "model"}
    points = []
    for cost in tables["COST"]:
        if cost.get("status", "COMPLETE") != "COMPLETE" or cost.get("cost_valid", True) is not True:
            continue
        model = models[cost["accuracy_run_id"]]
        mode, seed = model["mode"], model.get("seed")
        seed_label = f" s{seed}" if seed is not None else ""
        if mode == "alpha":
            group = "alpha_"+model["base_role"]
            label = model["alpha_label"]+seed_label
            marker = {"alpha0":"o", "alpha05":"s", "alpha1":"^", "fitted":"P"}[model["alpha_label"]]
            color = "#777777" if model["base_role"] == "Raw" else "#CC79A7"
        else:
            group = "heads" if mode in ("conditional", "past_control") else "body_baselines"
            label = mode.replace("_", " ")+seed_label
            marker = {"A":"s", "B":"^", "conditional":"o", "past_control":"s"}.get(mode, "D" if mode in ("Raw", "G", "LoRA", "Ridge") else "o")
            color = COLORS.get(mode, "#333333")
        identity = dict(dataset=cost["dataset"], device=cost["device"], batch_size=cost["batch_size"],
            schedule=cost["schedule"], run_id=model["run_id"], cost_id=cost["cost_id"], mode=mode, seed=seed,
            group=group, label=label, marker=marker, color=color, filled=seed != common.SEEDS[1], y=cost["MSE"])
        if cost["schedule"] == "native":
            device_label = cost["device"].replace(":", "_")
            name = f"F3_NATIVE_ALL_{cost['dataset']}_{device_label}_B{cost['batch_size']}"
            metrics = [("latency", "median_batch_ms")]
            metrics.append(("allocated", "peak_allocated_MiB") if cost["device"].startswith("cuda") else ("RSS", "CPU_RSS_sampled_peak_MiB"))
            for metric, key in metrics:
                if cost.get(key) is not None:
                    points.append(dict(identity, view="all_native", figure=name, metric=metric, x=cost[key]))
        if mode in ("A", "B") and (mode == "A" or cost["schedule"] == "native") and cost["device"].startswith("cuda"):
            label = mode+" "+cost["schedule"]+seed_label
            points.append(dict(identity, view="matched_two_call", figure="F3_MATCHED_TWO_CALL", metric="latency",
                               label=label, marker="s" if cost["schedule"] == "two_call" else "o" if mode == "A" else "^",
                               x=cost["median_batch_ms"]))
    return points


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fields or ["status"])
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                         if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def _read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as stream:
        values = list(csv.DictReader(stream))
    def decode(value):
        if value == "": return None
        if value in ("True", "False"): return value == "True"
        try: return json.loads(value)
        except (json.JSONDecodeError, TypeError): return value
    return [{key:decode(value) for key, value in row.items()} for row in values]


def _resource_ledger(ctx, data):
    path = Path(ctx["cache"])/"EVENTS.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()] if path.exists() else []
    from .run import load_registered_plan, registered_plan_path
    plan = load_registered_plan(ctx)
    ends = [row["time_unix"] for row in events]
    from time import time
    finished = max(ends+[time()])
    start_path = Path(ctx["report"])/"EXECUTION_START.json"
    execution_start = common.read(start_path) if start_path.exists() else {}
    core_started = execution_start.get("started_unix")
    core_elapsed = max(0., finished-core_started) if core_started is not None else None
    gpu = [r for r in events if r["stage"] == "GPU_SESSION" and r["status"] in ("COMPLETE", "FAILED")]
    closed = {row.get("session_id") for row in gpu}
    unclosed = [row for row in events if row["stage"] == "GPU_SESSION" and row["status"] == "STARTED"
                and row.get("session_id") not in closed]
    pending_seconds = sum(max(0., min(finished, row["budget_deadline_unix"])-row["time_unix"]) for row in unclosed)
    arms = data["arms"]
    parent_bindings = {a[key]["reference"]: a[key] for a in arms
                       for key in ("parent_g_prediction_binding", "parent_b_prediction_binding") if a.get(key)}
    parent_storage = [dict(binding=value, bytes=common.resolve_binding(value, ctx).stat().st_size)
                      for _, value in sorted(parent_bindings.items())]
    head_bindings = {a["head_binding"]["reference"]: a["head_binding"] for a in arms if a.get("head_binding")}
    head_storage = [dict(binding=value, bytes=common.resolve_binding(value, ctx).stat().st_size)
                    for _, value in sorted(head_bindings.items())]
    parent_events = [row for row in events if row["stage"] == "parent_cache" and row["status"] == "COMPLETE"]
    return dict(campaign_wall_seconds=finished-plan["started_unix"],
        registered_plan_binding=common.binding(registered_plan_path(ctx), ctx),
        registered_plan_revision=registered_plan_path(ctx).name, plan_supersedes=plan.get("supersedes"),
        overall_wall_seconds_since_first_CPU_audit=finished-plan["started_unix"],
        core_execution_started_unix=core_started, core_execution_wall_seconds=core_elapsed,
        preparation_elapsed_before_core_seconds=core_started-plan["started_unix"] if core_started is not None else None,
        registered_core_wall_budget_seconds=plan["limits"]["wall_seconds"],
        core_wall_budget_remaining_seconds=plan["limits"]["wall_seconds"]-core_elapsed if core_elapsed is not None else None,
        core_budget_anchor="EXECUTION_START.json after preceding dataset main publication; immutable across resumes",
        started_unix=plan["started_unix"], accounted_through_unix=finished,
        GPU_reserved_process_wall_upper_bound_seconds=sum(r.get("wall_seconds", 0) for r in gpu)+pending_seconds,
        unmatched_started_GPU_sessions=len(unclosed), unmatched_started_GPU_accounted_seconds=pending_seconds,
        GPU_kernel_busy_seconds=None, GPU_accounting="GPU-owned wall upper bound; CPU and subprocess waits may be included",
        preparation_profile_discarded_seconds=sum(r.get("wall_seconds", 0) for r in events if r["stage"].startswith("gates_")),
        new_body_configurations=sum(a["mode"] in ("A", "B") for a in arms),
        body_searches=sum(a["mode"] in ("S", "A", "B") and not a.get("reused", False) for a in arms),
        body_fresh_refits=sum("refit_summary" in a for a in arms), reused_bodies=sum(a["mode"] == "S" and a.get("reused", False) for a in arms),
        head_searches=sum(a["mode"] in ("conditional", "past_control") for a in arms),
        head_fresh_refits=sum(a["mode"] in ("conditional", "past_control") for a in arms),
        analytic_alpha_M_fit_fits=len({a["run_id"].rsplit("_", 1)[0] for a in arms if a["mode"] == "alpha"}),
        analytic_alpha_M_cal_fits=len({a["run_id"].rsplit("_", 1)[0] for a in arms if a["mode"] == "alpha"}),
        DirectG_seed_count_per_dataset=1, DirectG_shared_across_S_seeds=True,
        fixed_parent_M_cal_cache_count=len(parent_storage),
        fixed_parent_M_cal_cache_bytes=sum(item["bytes"] for item in parent_storage),
        fixed_parent_M_cal_cache_artifacts=parent_storage,
        fixed_parent_M_cal_cache_generation_seconds=sum(row.get("wall_seconds", 0) for row in parent_events),
        fixed_parent_M_cal_cache_timing_scope="Recorded parent_cache event wall includes generation or resumed serialization/parity validation; not kernel time",
        fixed_parent_M_cal_cache_events=[dict(run_id=row["run_id"], wall_seconds=row.get("wall_seconds"),
                                             prediction_binding=row.get("prediction_binding")) for row in parent_events],
        head_artifact_count=len(head_storage), head_artifact_bytes=sum(item["bytes"] for item in head_storage),
        head_artifacts=head_storage,
        head_search_fit_seconds=sum(a.get("head_search_seconds", 0) for a in arms),
        head_fresh_refit_fit_seconds=sum(a.get("head_refit_seconds", 0) for a in arms),
        head_fit_timing_scope="CPU head fitting only; these are not head-only deployment inference times",
        cost_fresh_process_facets=len(data["costs"]), warmup_calls=10*len(data["costs"]), timed_calls=20*len(data["costs"]),
        stages=[dict(stage=r["stage"], status=r["status"], label=r.get("label"), wall_seconds=r.get("wall_seconds")) for r in events],
        includes_preparation_CPU_gates_profile_cache_head_cost_and_failed_sessions=True)


def _figure_files(fig, output, name):
    import matplotlib.pyplot as plt
    fig.savefig(output/(name+".png"), dpi=180, metadata={"Software": "raw-context public CSV replay"})
    fig.savefig(output/(name+".svg"), metadata={"Date": None, "Creator": "raw-context public CSV replay"})
    plt.close(fig)


def render_figures(tables, output):
    """Render only public sufficient statistics and public summaries."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
            "axes.spines.right": False, "svg.hashsalt": common.NAME, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for row, dataset in enumerate(("jena", "pems32")):
            for phase, column, metric, xlabel in (("A_fit_search", 0, "astop_mse", "Optimizer updates"),
                                                  ("M_fit_head_search", 1, "M_select_mse", "Optimizer updates")):
                ax = axes[row, column]
                values = [r for r in tables["TRAINING_CURVES"] if r["dataset"] == dataset and r["phase"] == phase]
                for identifier in sorted({v["run_id"] for v in values}):
                    points = sorted((v for v in values if v["run_id"] == identifier), key=lambda r: r["updates"])
                    mode, seed = points[0]["mode"], points[0]["seed"]
                    style = "-" if seed == common.SEEDS[0] else "--"
                    ax.plot([v["updates"] for v in points], [v[metric] for v in points], style,
                            marker="o", markersize=3, color=COLORS.get(mode, "#333333"), label=f"{mode} s{seed}")
                    selected = [v for v in points if v["selected"]]
                    ax.scatter([v["updates"] for v in selected], [v[metric] for v in selected],
                               marker="*", s=90, color=COLORS.get(mode, "#333333"), zorder=5)
                ax.set(title=f"{dataset}: {'A_stop body selection' if column == 0 else 'M_select head selection'}", xlabel=xlabel,
                       ylabel="Observed channel-macro MSE")
                ax.grid(alpha=.18)
                if values: ax.legend(fontsize=7, ncol=2)
                else: ax.text(.5, .5, "No completed curve", transform=ax.transAxes, ha="center")
        fig.suptitle("Registered checkpoint curves; stars = selected checkpoints\nTwo seeds are paired runs, not confidence intervals. Training segments are shown separately.", fontsize=11)
        _figure_files(fig, output, "F1_TRAINING_CURVES")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for ax, dataset in zip(axes, ("jena", "pems32")):
            values = [r for r in tables["TRAINING_CURVES"] if r["dataset"] == dataset and
                      r["phase"] in ("A_fit_training_segment", "A60_refit_training_segment") and r.get("TRAIN_segment_batch_loss") is not None]
            for identifier, phase in sorted({(v["run_id"], v["phase"]) for v in values}):
                points = sorted((r for r in values if r["run_id"] == identifier and r["phase"] == phase), key=lambda r:r["updates"])
                mode, seed = points[0]["mode"], points[0]["seed"]
                ax.plot([r["updates"] for r in points], [r["TRAIN_segment_batch_loss"] for r in points],
                    "--" if phase == "A60_refit_training_segment" else "-", marker="o", markersize=3,
                    color=COLORS.get(mode, "#333333"), alpha=.75,
                    label=f"{mode} s{seed} {'refit' if phase == 'A60_refit_training_segment' else 'search'}")
            ax.set(title=dataset, xlabel="Optimizer updates within the fitting stage", ylabel="Mean batch loss during segment")
            ax.grid(alpha=.18)
            if values: ax.legend(fontsize=6, ncol=2)
            else: ax.text(.5, .5, "Historical producer did not retain TRAIN segments", transform=ax.transAxes, ha="center", fontsize=8)
        fig.suptitle("TRAIN segments describe losses during optimization, not full-dataset scores at a checkpoint", fontsize=11)
        _figure_files(fig, output, "F1_TRAIN_SEGMENTS")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for row, dataset in enumerate(("jena", "pems32")):
            for col, metric in enumerate(("mse", "mae")):
                ax = axes[row, col]
                values = [r for r in tables["PAIRED_DELTAS"] if r["dataset"] == dataset and r["metric"] == metric]
                contrasts = sorted({v["contrast"] for v in values})
                for contrast_index, contrast in enumerate(contrasts):
                    for seed in common.SEEDS:
                        points = sorted((v for v in values if v["contrast"] == contrast and v["seed"] == seed and v["axis"] == "channel"), key=lambda r:int(r["index"]))
                        if not points: continue
                        color = ["#0072B2", "#D55E00", "#009E73"][contrast_index % 3]
                        ax.plot([v["index"] for v in points], [v["delta"] for v in points],
                            "-" if seed == common.SEEDS[0] else "--", color=color, alpha=.8, label=f"{contrast} s{seed}")
                thirds = [r for r in values if r["axis"] == "horizon_third"]
                text = []
                for contrast in contrasts:
                    cells = []
                    for part in ("near", "middle", "far"):
                        paired = [r["delta"] for r in thirds if r["contrast"] == contrast and r["index"] == part and r["delta"] is not None]
                        cells.append(f"{part} {np.mean(paired):+.4g}" if paired else f"{part} N/A")
                    text.append(contrast+": "+", ".join(cells))
                ax.set(title=f"{dataset}: paired channel Δ{metric.upper()}", xlabel="Original channel index", ylabel=f"Candidate - reference {metric.upper()}")
                ax.axhline(0, color="black", lw=.8)
                ax.grid(alpha=.18)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.margins(y=.2)
                if values: ax.legend(fontsize=7, ncol=2, loc="upper right")
                ax.text(.01, .02, "\n".join(text), transform=ax.transAxes, fontsize=7,
                        va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        fig.suptitle("All original channels and fixed horizon thirds; negative Δ favors candidate\nSolid/dashed = the two seeds. Horizon annotations are seed means, not uncertainty intervals.", fontsize=11)
        _figure_files(fig, output, "F2_PAIRED_DELTAS")

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        for row, dataset in enumerate(("jena", "pems32")):
            for col, (batch, xkey, xlabel) in enumerate(((1, "median_batch_ms", "B1 whole-path median (ms/batch)"),
                    (8, "median_batch_ms", "B8 whole-path median (ms/batch)"),
                    (8, "peak_allocated_MiB", "B8 peak allocated tensors (MiB)"))):
                ax = axes[row, col]
                values = [r for r in tables["T1_ACCURACY_COST"] if r["dataset"] == dataset and r["batch_size"] == batch
                          and r["device"].startswith("cuda") and r["schedule"] == "native" and r[xkey] is not None and r["row_kind"] == "model"]
                for mode in ("Raw", "G", "LoRA", "S", "A", "B"):
                    points = [v for v in values if v["mode"] == mode]
                    if points:
                        ax.scatter([v[xkey] for v in points], [v["MSE"] for v in points],
                            color=COLORS[mode], marker="D" if mode in ("Raw", "G", "LoRA") else "o", s=45, label=mode)
                ax.set(title=f"{dataset}: native deployment", xlabel=xlabel, ylabel="V channel-macro MSE")
                ax.grid(alpha=.18)
                if values: ax.legend(fontsize=8)
        fig.suptitle("Body and baseline focus: actual whole physical-path GPU costs\nEvery native route, including combinations and CPU Ridge, is shown in the separate F3_NATIVE_ALL figures.", fontsize=11)
        _figure_files(fig, output, "F3_ACCURACY_COST")

        points = tables["FIGURE_POINTS"]
        group_titles = dict(body_baselines="Bodies and baselines", alpha_Raw="Raw + fixed S: alpha", alpha_G="G + fixed S: alpha", heads="Conditional and past control")
        for name in sorted({point["figure"] for point in points if point["view"] == "all_native"}):
            selected = [point for point in points if point["figure"] == name]
            groups = [group for group in group_titles if any(point["group"] == group for point in selected)]
            fig, axes = plt.subplots(2, len(groups), figsize=(4.4*len(groups), 8), squeeze=False, constrained_layout=True)
            gpu = selected[0]["device"].startswith("cuda")
            for col, group in enumerate(groups):
                for row, metric in enumerate(("latency", "allocated" if gpu else "RSS")):
                    ax = axes[row, col]
                    values = sorted((point for point in selected if point["group"] == group and point["metric"] == metric), key=lambda p:p["run_id"])
                    for point in values:
                        ax.scatter(point["x"], point["y"], marker=point["marker"], s=54,
                            facecolors=point["color"] if point["filled"] else "none", edgecolors=point["color"],
                            linewidths=1.4, label=point["label"])
                    ax.set(title=group_titles[group], ylabel="V channel-macro MSE",
                        xlabel="Whole-path median (ms/batch; log axis)" if metric == "latency" else
                               "Peak allocated tensors (MiB)" if gpu else "Sampled CPU RSS peak (MiB)")
                    if metric == "latency":
                        ax.set_xscale("log")
                    ax.grid(alpha=.18)
                    group_has_memory = any(point["group"] == group and point["metric"] != "latency" for point in selected)
                    if values and (row == 1 or not group_has_memory):
                        ax.legend(fontsize=7, ncol=1 if group == "heads" else 2,
                            loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)
                    if not values:
                        ax.text(.5, .5, "Unavailable measurement; no zero substituted", transform=ax.transAxes, ha="center", fontsize=8)
            first = selected[0]
            fig.suptitle(f"{first['dataset']} | {first['device']} | B{first['batch_size']}: every completed native route\n"
                "For paired routes, filled/open markers = s92601/s92602. Legends identify each route; coincident endpoints are retained.", fontsize=10)
            _figure_files(fig, output, name)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        for row, dataset in enumerate(("jena", "pems32")):
            for col, batch in enumerate((1, 8)):
                ax = axes[row, col]
                values = sorted((point for point in points if point["view"] == "matched_two_call"
                    and point["dataset"] == dataset and point["batch_size"] == batch), key=lambda p:(p["mode"], p["schedule"], p["seed"]))
                for point in values:
                    ax.scatter(point["x"], point["y"], marker=point["marker"], s=60,
                        facecolors=point["color"] if point["filled"] else "none", edgecolors=point["color"],
                        linewidths=1.4, label=point["label"])
                absent = {"A", "B"}-{point["mode"] for point in values}
                scope = " ("+"/".join(sorted(absent))+" not registered)" if absent and values else ""
                ax.set(title=f"{dataset}: CUDA B{batch} schedule comparison"+scope, xlabel="Whole-path median (ms/batch)", ylabel="V channel-macro MSE")
                ax.grid(alpha=.18)
                if values: ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)
                else: ax.text(.5, .5, "No registered A/B matched comparison", transform=ax.transAxes, ha="center", fontsize=8)
        fig.suptitle("A native one-call remains visible; A matched two-call uses the same trained weights\nBoth B1 and B8 are separated; filled/open markers and labels identify the two seeds.", fontsize=11)
        _figure_files(fig, output, "F3_MATCHED_TWO_CALL")


def _text_report(tables, resources, queue):
    rows = [r for r in tables["RUNS"] if r["row_kind"] == "model"]
    lines = ["# 원본·문맥 표현 분리와 다중 해상도 잠재 예측", "",
        "[확인] 원채널 normalized observed channel-macro MSE가 주지표이고 MAE는 보조지표다. 모든 완료 seed와 불리한 결과를 포함했다.", "",
        "S는 ALL_CONTEXT B2, A는 앞 K/2 PCA 축의 원본 투영과 뒤 K/2 축의 문맥 투영, B는 A의 문맥 잠재 경로만 2배 평균한 모델이다. P는 모두 길이512에서 계산하며 anchor는 한 번만 더한다. B의 F 입력은 실제512/256이고 decoder lag는 원격자에 적용한다.", "",
        "V는 반복 사용된 개발 평가다. 새 모델·전이·head 선택은 V 전에 봉인했지만 V의 과거 노출, 상류 전처리와 사전학습 중복 가능성을 지우지 않는다. 보호 T에 새 수치 접근을 하지 않았다.", "", "```text",
        "dataset   mode          seed       MSE           MAE"]
    for row in rows:
        if row["mode"] in ("S", "A", "B", "Raw", "G", "LoRA", "Ridge"):
            lines.append(f"{row['dataset']:<9} {row['mode']:<13} {str(row['seed']):<10} {row['mse']:.10g}  {row['mae']:.10g}")
    lines += ["```", "", "두 seed 산술평균과 각 seed는 RUNS.csv에 모두 있다. 원점·채널·겹치는 창을 독립 표본으로 간주한 신뢰구간이나 유의성 검정은 만들지 않았다.", ""]
    lines += [f"현재 등록 계획: {resources.get('registered_plan_revision', 'EXPERIMENT_PLAN.json')}. 수리 리비전이 있으면 원래 계획의 바이트를 보존하고, 같은 과학적 계약을 승계하는 해당 리비전의 지문을 검산에 연결한다.", ""]
    lines += ["B8 native 비용의 같은 장치·메모리 계층 안에서 비교한다. 아래 ms는 배치 전체와 원점당 값을 구분한다.", "", "```text",
              "dataset   mode    seed    device  ms/batch   ms/origin  allocatedMiB  reservedMiB"]
    def number(value):
        return "N/A" if value is None else f"{float(value):.6g}"
    for cost in tables["T1_ACCURACY_COST"]:
        if cost["batch_size"] == 8 and cost["schedule"] == "native":
            lines.append(f"{cost['dataset']:<9} {cost['mode']:<7} {str(cost['seed']):<7} {cost['device']:<7} "
                f"{number(cost['median_batch_ms']):<10} {number(cost.get('median_ms_per_origin')):<10} "
                f"{number(cost['peak_allocated_MiB']):<13} {number(cost['peak_reserved_MiB'])}")
    lines += ["```", "", "mean은 두 seed facet 요약값의 산술평균이며 별도의 실행 측정치나 독립 반복을 뜻하지 않는다. per-PID GPU 측정 상태와 RSS는 COST.csv에 별도 열로 공개했다.", ""]
    for contrast in ("S_to_A", "A_to_B"):
        for dataset in ("jena", "pems32"):
            values = [r for r in tables["COMPARISONS"] if r["contrast"] == contrast and r["dataset"] == dataset]
            if not values:
                lines.append(f"{dataset} {contrast}: 등록된 전이 family에 포함되지 않아 대응 비교가 없다.")
                continue
            direction = "이 고정 조건에서 두 seed 모두 MSE 감소" if all(r["delta_mse"] < 0 for r in values) else "두 seed 모두 MSE 증가/동일" if all(r["delta_mse"] >= 0 for r in values) else "seed별 방향이 다른 혼합 결과"
            lines.append(f"{dataset} {contrast}: {direction}; 평균 ΔMSE={np.mean([r['delta_mse'] for r in values]):+.10g}, 평균 ΔMAE={np.mean([r['delta_mae'] for r in values]):+.10g}.")
    for dataset in common.EXPECTED:
        values = [r for r in tables["COMPARISONS"] if r["dataset"] == dataset and r["contrast"] == "past_control_to_conditional"]
        if values:
            claim = "두 seed 모두 conditional MSE가 낮았다" if all(r["delta_mse"] < 0 for r in values) else "추가 B2 조건 정보의 이득이 두 seed에서 일관되게 확인되지 않았다"
            lines.append(f"{dataset} 원본 예측 결합: {claim}; conditional−past_control 평균 ΔMSE={np.mean([r['delta_mse'] for r in values]):+.10g}, ΔMAE={np.mean([r['delta_mae'] for r in values]):+.10g}.")
    lines += ["", "정확도와 비용은 T1_ACCURACY_COST.csv, 원인 대조는 T2_SINGLE_HYPOTHESIS.csv, 결합은 T3_OUTPUT_COMBINATION.csv 및 COMPLEMENTARITY.csv에 있다. F3_NATIVE_ALL 그림은 COST의 모든 완료 native 경로를 데이터·장치·B1/B8별로 분리하며 alpha·conditional·past_control과 CPU Ridge도 포함한다. F3_MATCHED_TWO_CALL은 B1/B8을 나누어 각 seed의 A native/two_call과 B를 보여준다. FIGURE_POINTS.csv와 FIGURE_MANIFEST.json에 실제 그림의 경로별 식별자를 기록했다.", "",
        "A의 native 한 호출과 비용 전용 two_call을 함께 공개한다. B의 낮은 요청 미래 길이는 Bolt 고정 head의 연산이 비례해 줄었다는 증거가 아니다. T3의 head_search_seconds/head_refit_seconds는 적합 비용이며 standalone 부모 지연을 빼서 head-only 추론 시간으로 해석하지 않는다.", "",
        "G가 MSE·whole-path 지연·allocated 피크에서 모두 지배하는 행은 T1의 G_dominates_MSE_latency_allocated에 표시했다. 이런 조건에서는 F를 유지할 실용 선택 이유가 확보되지 않는다. Conditional이 past_control보다 나아야 추가 B2 조건 정보 가치의 근거가 생기며, 이것만으로 F의 인과적 필요성이나 일반적 우수성이 증명되지는 않는다.", "",
        "M_fit 적합, M_select 선택, A_stop 선택, V 보고 점수는 다른 열이다. M_cal_batch_fit_loss 및 TRAIN_segment_batch_loss는 학습 도중 배치 평균 손실로서 전체 구간의 현재 checkpoint MSE와 다르다. 새 본체 segment는 기록했고, 과거 재사용 S의 미기록 segment를 복원하거나 만들어 넣지 않았다.", "",
        "Ridge는 CPU 지연·RSS 맥락이다. CPU/GPU 속도비를 만들지 않으며 allocated, reserved, 측정 가능한 per-PID GPU memory를 다른 열로 유지한다. WDDM에서 per-PID 측정이 불가하면 N/A다. p95는20회 trial의 서술 분위수이며 서비스 꼬리 지연 보장이 아니다.", "",
        f"실제 캠페인 wall={resources['campaign_wall_seconds']:.3f}초; GPU 소유 wall 상한={resources['GPU_reserved_process_wall_upper_bound_seconds']:.3f}초. GPU kernel busy 시간은 측정하지 않았다. 새 본체 설정 {resources['new_body_configurations']}개, 본체 fresh refit {resources['body_fresh_refits']}개, 재사용 S {resources['reused_bodies']}개, head search/fresh refit 각각 {resources['head_searches']}/{resources['head_fresh_refits']}개다.", "",
        "cap에서 계속 좋아지는 경로는 BUDGET_LIMITED_CONVERGENCE_UNRESOLVED이며 수렴/구조적 실패로 판정하지 않는다. 실제 제외·연장 이유와 재개 명령은 RESUME_QUEUE.md를 따른다.", "",
        "발표에서 지지 가능한 문장: 이 고정 데이터·K·두 seed·예산에서 측정한 오차와 실제 비용의 교환관계; S→A와 A→B가 바꾼 연산; 음의 결과와 seed 불일치; V가 개발 평가라는 한정.", "",
        "아직 쓰면 안 되는 문장: 모든 데이터에서 우수하다; 같은 K이므로 FLOPs·메모리가 같다;256시점이 더 긴 과거다; coarse H/2이면 head 계산도 절반이다; F가 반드시 필요하다; 독립 TEST나 유의성이 확인됐다.", "",
        "공개 재생성: `python -m research.raw_context_multiscale_20261018_v1.report replay --source-dir reports/raw_context_multiscale_20261018_v1 --output-dir <새 출력 폴더>`.",
        "실제 수치 검산: `python -m research.raw_context_multiscale_20261018_v1.run verify --asset-root <local-asset-root>`.",
        "원격 게시 여부는 별도 실제 원격 검증 기록을 따른다. 이 보고 생성만으로 remote verified를 선언하지 않는다.", ""]
    return "\n".join(lines)


METHOD = """# 방법 차이와 문헌의 범위

S: Z=(P(X)-mu)E, U=F(Z), 기존 원격자 lag7 D와 원채널 마지막 값 연결.
A: E의 앞 절반은 (X-mu), 뒤 절반은 (P(X)-mu)에 적용한다. 총 P/E/D 용량과 PCA 열 순서를 유지한다.
B: A의 context 잠재축만 창 내부 인접2시점 평균 → 길이256의 실제 F 예측 → 과거·미래 block repeat → 원격자 lag7 decoder. P 자체는512시점에서 한 번 계산하고 anchor도 한 번만 더한다.

S→A는 원본/문맥 투영 분리, A→B는 평균화·낮은 격자 예측·반복 복원의 전체 차이다. A의 native 한 호출과 같은 weights의 비용 전용 두 호출을 나누어 제시한다. 같은 K·파라미터 수가 같은 유효 표현·학습 동역학·FLOPs를 뜻하지 않는다.

[TimeMixer, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)는 과거와 미래 예측의 여러 시간 해상도 활용 원리를 제공한다. 이 구현은 TimeMixer의 분해·mixing 전체 구조를 재현하지 않는다.
[NHITS, AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854)는 multi-rate sampling과 미래 보간이라는 인접 원리를 제공한다. 여기의 고정 block repeat는 NHITS의 학습된 계층적 보간 재현이 아니다.
[Bates와 Granger, The Combination of Forecasts,1969](https://doi.org/10.1057/jors.1969.103)는 고정 부모 오차를 이용한 예측 결합의 배경이다. 본 실험의 alpha는 M_fit에서 채널 동일 가중 MSE로 적합하고 M_select 진단 후 전체 M_cal에서 fresh 적합한다.

우리 Bolt용 원본/문맥 분리,512/256 두 호출, 원격자 lag 복원의 조합은 신규 변형이다. 위 문헌이 이 조합의 성공·효율성·신규성을 증명한다고 주장하지 않는다.

Conditional: g+H(past-current,g-current,b-g). Past control은 세 번째 항에 마지막 H개 실제 과거를 사용한다. hidden32 채널공유 head는 M_cal 내부 target-disjoint60/40에서 선택한 epoch로 전체 M_cal을 fresh fit한다. Past control은 B2/F를 적재하거나 호출하지 않는다.

원래 H×C 관측 target을 유지하며 normalized observed equal-channel macro MSE가 주지표다. MAE를 사후 주지표로 바꾸지 않는다. V는 개발 평가이며 새 보호 T 접근은 없다.
"""


def render_public(tables, resources, queue, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        _csv(output/(name+".csv"), tables[name])
    common.atomic_json(output/"RESOURCE_LEDGER.json", resources)
    common.atomic_json(output/"REPORT_INPUTS.json", dict(resume_queue=queue,
        tables={name:common.sha(output/(name+".csv")) for name in TABLES},
        code="research/raw_context_multiscale_20261018_v1/report.py"))
    (output/"README_KO.md").write_text(_text_report(tables, resources, queue), encoding="utf8", newline="\n")
    (output/"METHOD_DELTA_KO.md").write_text(METHOD, encoding="utf8", newline="\n")
    resume = ["# 재개 상태", "", "완료 수치는 보존한다. 점수에 따라 후보·방법군·선택규칙을 바꾸지 않는다.", ""]
    for item in queue:
        resume.append(f"- {item['dataset']}: {item['reason']}; scope={item.get('scope')}; requested cap={item.get('requested_cap')}.")
    if not queue:
        resume.append("등록된 단계는 완료됐다. cap에서 개선 중인 모델의 수렴은 미확정이며 추가 예산 실험은 별도 승인·등록 범위다.")
    resume += ["", "기존 상태를 보존하며 같은 실행기를 재개하는 명령:", "", "```text",
        "python -m research.raw_context_multiscale_20261018_v1.run execute --asset-root <local-asset-root> --plan reports/raw_context_multiscale_20261018_v1/EXPERIMENT_PLAN.json",
        "```", "", "예산 부족으로 미입장한 연장은 기존 immutable decision을 덮어써서 수행하지 않는다. 추가 예산·새 attempt를 등록한 후 진행한다.", ""]
    (output/"RESUME_QUEUE.md").write_text("\n".join(resume), encoding="utf8", newline="\n")
    render_figures(tables, output)
    figure_names = {"F1_TRAINING_CURVES", "F1_TRAIN_SEGMENTS", "F2_PAIRED_DELTAS", "F3_ACCURACY_COST", "F3_MATCHED_TWO_CALL"}
    figure_names.update(point["figure"] for point in tables["FIGURE_POINTS"])
    common.atomic_json(output/"FIGURE_MANIFEST.json", dict(
        point_table=dict(file="FIGURE_POINTS.csv", sha256=common.sha(output/"FIGURE_POINTS.csv")),
        figures=[dict(name=name, png_sha256=common.sha(output/(name+".png")), svg_sha256=common.sha(output/(name+".svg")),
                      route_point_rows=sum(point["figure"] == name for point in tables["FIGURE_POINTS"]))
                 for name in sorted(figure_names)],
        all_native_facets=[{key:point[key] for key in ("figure", "run_id", "cost_id", "dataset", "device", "batch_size", "schedule", "seed")}
                           for point in tables["FIGURE_POINTS"] if point["view"] == "all_native" and point["metric"] == "latency"]))
    return dict(status="COMPLETE", files=sorted(p.name for p in output.iterdir() if p.is_file()))


def replay(source_dir, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError("Public replay requires a separate output directory")
    inputs = common.read(source/"REPORT_INPUTS.json")
    for name, expected in inputs["tables"].items():
        if common.sha(source/(name+".csv")) != expected:
            raise ValueError("Public source CSV changed: "+name)
    tables = {name:_read_csv(source/(name+".csv")) for name in TABLES}
    resources = common.read(source/"RESOURCE_LEDGER.json")
    return render_public(tables, resources, inputs["resume_queue"], output)


def generate(ctx, output_dir=None):
    output = Path(output_dir).resolve() if output_dir is not None else Path(ctx["report"]).resolve()
    if output_dir is not None:
        if not output.is_relative_to(Path(ctx["cache"]).resolve()):
            raise ValueError("Alternative campaign output must stay in its private cache")
        if (Path(ctx["report"])/"REPORT_INPUTS.json").exists():
            return replay(ctx["report"], output)
    data = ctx.get("_report_verified_data")
    if data is None:
        verify(ctx)
        data = ctx["_report_verified_data"]
    for name, value in data["manifest_bindings"].items():
        common.resolve_binding(value, ctx)
    tables = _public_tables(data)
    ledger = _resource_ledger(ctx, data)
    queue = data["selection"].get("resume_queue", [])
    result = render_public(tables, ledger, queue, output)
    from time import time
    ledger["report_generation_seconds"] = time()-ledger["accounted_through_unix"]
    ledger["accounted_through_unix"] += ledger["report_generation_seconds"]
    ledger["campaign_wall_seconds"] = ledger["accounted_through_unix"]-ledger["started_unix"]
    ledger["overall_wall_seconds_since_first_CPU_audit"] = ledger["campaign_wall_seconds"]
    if ledger["core_execution_started_unix"] is not None:
        ledger["core_execution_wall_seconds"] = ledger["accounted_through_unix"]-ledger["core_execution_started_unix"]
        ledger["core_wall_budget_remaining_seconds"] = ledger["registered_core_wall_budget_seconds"]-ledger["core_execution_wall_seconds"]
    common.atomic_json(output/"RESOURCE_LEDGER.json", ledger)
    (output/"README_KO.md").write_text(_text_report(tables, ledger, queue), encoding="utf8", newline="\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "generate", "replay"))
    parser.add_argument("--asset-root")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "replay":
        if args.source_dir is None or args.output_dir is None:
            parser.error("replay requires --source-dir and --output-dir")
        result = replay(args.source_dir, args.output_dir)
    else:
        if args.asset_root is None:
            parser.error("verify/generate requires --asset-root")
        ctx = common.context(args.asset_root, device="cpu")
        result = verify(ctx) if args.stage == "verify" else generate(ctx, args.output_dir)
    print(json.dumps(dict(status=result["status"], stage=args.stage)))


if __name__ == "__main__":
    main()
