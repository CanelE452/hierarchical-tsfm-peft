"""Compact saved-array E/D audit; no Bolt, optimizer, backward or training replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import sys

import audit_projection_results_v1 as prior

HERE, ROOT = prior.HERE, prior.ROOT
require = prior.require
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
LAGS = {"L0": (0,), "L7": (0, 1, 2, 4, 8, 16, 32)}
SEEDS, CHECKPOINTS = (92601, 92602), (0, 100, 200, 400, 600)
FIT_N, TUNE_N = 256, 64
TRAIN_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
PROBE_SHA = "00440bf6e09ac31db4666ffc656ec348cc4b5f698b18600c65d0ff99cf0d1b8d"
CONTROLLER_SHA = "e268528d77d17b0c8f5a300ff8d667ce5ddd25b1f61754c3895df87929fcd1f3"
PURE_SHA = "17c8ac7d20b9f3dd759a3b3fb2e019e027469e9c8a057eb1176da98b9c319931"
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    prior.initialize_numpy(); np = sys.modules["numpy"]


def exact(a, b, label):
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), "BYTE_EXACT:" + label)


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": prior.file_sha(path)}


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = np.ascontiguousarray(value)
        for part in (name.encode(), str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()): digest.update(part)
    return digest.hexdigest()


def selected_geometry(initial, selected, latent):
    changes = {key: {"delta_L2": float(np.linalg.norm(selected[key].astype(np.float64) - initial[key].astype(np.float64))), "delta_max_abs": float(np.abs(selected[key].astype(np.float64) - initial[key].astype(np.float64)).max()), "same_bytes": selected[key].tobytes() == initial[key].tobytes()} for key in ("E", "D", "center")}
    singular = np.linalg.svd(selected["E"].astype(np.float64), compute_uv=False)
    return {"CP0_to_selected": changes, "E_same_bytes": changes["E"]["same_bytes"], "selected_E_singularvalues": singular.tolist(), "E_numeric_rank_relative_cutoff": 1e-8, "E_numeric_rank": int((singular > singular[0] * 1e-8).sum()), "rank_scope": "Auditor diagnostic of selected FP32 E interpreted as FP64; no rank regularizer, winner selection or universal information claim", "selected_TUNE_latent_variance": np.var(latent.astype(np.float64), axis=(0, 1)).tolist()}


def delta_features(past, future, lags):
    n, h, k = future.shape; out = np.empty((n, h, len(lags) * k), np.float32)
    for hi in range(h):
        for li, lag in enumerate(lags):
            source = future[:, hi - lag] if hi + 1 > lag else past[:, past.shape[1] + hi - lag]
            out[:, hi, li * k:(li + 1) * k] = source - past[:, -1 - lag]
    return out


def replay(saved, state, raw, lags):
    c, k = raw["history"].shape[-1], state["E"].shape[-1]
    require(set(state) == {"E", "D", "center"} and all(v.dtype == np.float32 and np.isfinite(v).all() for v in state.values()) and state["E"].shape == (c, k) and state["center"].shape == (c,) and state["D"].shape == (len(lags) * k, c), "FINITE_FP32_ADAPTER_STATE_SHAPES")
    for name in ("target", "observed", "raw_last", "origins", "mean", "std"): exact(saved[name], raw[name], "saved_original_" + name)
    z = (raw["history"] - state["center"]) @ state["E"]
    prior.close(saved["latent_history"], z, "independent_NumPy_encoding")
    future = saved["latent_future"]
    require(future.dtype == np.float32 and future.shape == (len(z), 48, k) and np.isfinite(future).all(), "FINITE_SAVED_ACTUAL_F_FUTURE_NOT_REINFERRED")
    features = delta_features(saved["latent_history"], future, lags)
    exact(saved["lag_features"], features, "independent_causal_lag_features")
    prediction = raw["raw_last"][:, None] + features @ state["D"]
    prior.close(saved["prediction"], prediction, "independent_NumPy_fixed_D_forecast")
    return prior.scores(saved["prediction"], raw["target"], raw["observed"], raw["std"])


def check_score(entry, saved, raw):
    count = prior.compare_scores(entry["scores_all_observed"], prior.scores(saved["prediction"], raw["target"], raw["observed"], raw["std"]), "saved_actual_all_observed")
    complete = np.broadcast_to(raw["observed"].all(axis=2, keepdims=True), raw["observed"].shape)
    if complete.any(): count += prior.compare_scores(entry["scores_complete_mask"], prior.scores(saved["prediction"], raw["target"], complete, raw["std"]), "saved_actual_complete_C")
    else: require(entry["scores_complete_mask"] == {"status": "NO_COMPLETE_C"}, "NO_COMPLETE_C_EXPLICIT")
    return count


def unit_score(recorded, calculated):
    for field in ("mse", "mae", "per_channel_mse", "per_channel_mae"):
        values = [0. if value is None and not calculated["observed_counts"][i] else value for i, value in enumerate(recorded[field])] if field.startswith("per_channel") else recorded[field]
        prior.close(values, calculated[field], "CP_TUNE_" + field, atol=1e-10, rtol=0.)
    require(recorded["observed_counts"] == calculated["observed_counts"] and recorded["active_channels"] == calculated["active_channels"], "CP_TUNE_COUNTS")


def check_logs(row, fit):
    path = Path(row["updates_log"]["path"]); require(prior.file_sha(path) == row["updates_log"]["sha256"], "ACTUAL_UPDATE_LOG_SHA")
    logs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(len(logs) == row["training_updates_completed"] == 600 and [r["update"] for r in logs] == list(range(1, 601)), "ALL_600_LOGGED_UPDATES")
    for item in logs:
        fields = ("loss", "E_gradient_norm_before_clip", "D_gradient_norm_before_clip", "global_gradient_norm_before_clip")
        require(item["stage"] == "TRAINING_UPDATE" and all(type(item[f]) in (int, float) and math.isfinite(item[f]) and item[f] >= 0 for f in fields) and item["clipped"] == (item["global_gradient_norm_before_clip"] > 1.), "FINITE_LOGGED_LOSS_NORMS_CLIP_DECISION")
        require(row["learn_E"] or item["E_gradient_norm_before_clip"] == 0., "FIXED_E_NO_RECORDED_GRADIENT")
    prior.close(row["global_clipping_fraction"], sum(r["clipped"] for r in logs) / 600, "logged_clipping_fraction", atol=1e-10, rtol=0.)
    counts = fit["observed"].sum(axis=(0, 1)).tolist(); require(row["FIT_channel_observed_counts"] == counts, "ENTIRE_FIT_CHANNEL_DENOMINATORS")
    order = np.random.default_rng(row["seed"]).integers(0, len(fit["history"]), size=(600, 4))
    require(hashlib.sha256(order.tobytes()).hexdigest() == row["order_sha256"], "EXACT_SEED_SAME_600_BATCH_ORDER")
    return len(logs)


def source_contract(controller, pure):
    cf = {n.name: n for n in ast.parse(controller).body if isinstance(n, ast.FunctionDef)}
    train = ast.unparse(cf["train_condition"]); execute = ast.unparse(cf["execute"])
    for expression in ("np.random.default_rng(seed).integers(0, len(fit['history']), size=(updates, 4))", "torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)", "torch.nn.utils.clip_grad_norm_(parameters, 1.0)", "pure.global_fit_macro_mse(prediction, tensors['target'][ids], tensors['observed'][ids], counts, len(fit['history']))"):
        require(expression in train, "FIXED_TRAINING_SOURCE_CONTRACT:" + expression)
    require("F_FROZEN_EXCLUDED_FROM_OPTIMIZER" in train and "NO_DEV_ARRAY_IO_BEFORE_16_LOCKS" in execute and execute.index("resolved = train_all_conditions(") < execute.index("report['CP0_LS_and_RAW_baselines']") < execute.index("infer_development(report, resolved, development)"), "F_EXCLUSION_AND_ALL_LOCKS_BEFORE_DEV_SOURCE")
    pf = next(n for n in ast.parse(pure).body if isinstance(n, ast.FunctionDef) and n.name == "global_fit_macro_mse")
    loss = next(n.value for n in pf.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "loss" for t in n.targets))
    expected = ast.parse("(fit_windows/len(prediction))*(per_channel_SSE[active]/counts[active].to(prediction.dtype)).mean()", mode="eval").body
    require("safe_target = torch.where(observed, target, prediction.detach())" in ast.unparse(pf) and ast.dump(loss) == ast.dump(expected), "SAFE_MASK_AND_FIXED_FIT_MACRO_DENOMINATORS_SOURCE")
    return {"fixed_FIT_denominators_and_N_over_B": True, "F_excluded_from_optimizer": True, "all_16_locks_before_DEV_source": True, "independent_training_or_backward_replay": False}


def metadata(path, check):
    require(isinstance(TRAIN_SHA, str) and len(TRAIN_SHA) == 64, "ACTUAL_TRAIN_RESULT_NOT_YET_REGISTERED")
    binding = file_record(path); require(binding["sha256"] == TRAIN_SHA, "EXACT_TRAIN_RESULT_SHA")
    result = prior.verified_json(binding, check)
    require(result["mode"] == "train" and result["status"] in ("COMPLETE", "PARTIAL_COMPLETED"), "COMPLETED_OR_PRESERVED_PARTIAL_TRAIN_ONLY")
    require(result["controller_source"]["sha256"] == CONTROLLER_SHA and result["pure_source"]["sha256"] == PURE_SHA and result["probe_result"]["sha256"] == PROBE_SHA, "EXECUTED_SOURCE_AND_PROBE_BINDINGS")
    return binding, result


def audit_result(path, meter):
    binding, result = metadata(path, meter.check); check = meter.check
    science = prior.verified_json(result["causal_lag_diagnostic"], check); accepted = prior.verified_json(result["causal_lag_audit"], check)
    require(science["status"] == "COMPLETE" and accepted["status"] == "PASS" and accepted["diagnostic"]["sha256"] == result["causal_lag_diagnostic"]["sha256"], "EXACT_PREVIOUS_SCIENCE_PASS")
    probe = prior.verified_json(result["probe_result"], check)
    require(probe["status"] == "COMPLETE" and probe["mode"] == "probe" and probe["probe_optimizer_updates_completed"] == 4 and not probe["failed_paths"], "FOUR_COMPLETED_TECHNICAL_PROBES")
    for key in ("controller_source", "pure_source", "plan", "causal_lag_diagnostic", "causal_lag_audit", "parent_report", "parent_report_manifest", "parent_report_audit"): require(probe[key] == result[key], "SAME_TECHNICAL_PROBE_PARENT:" + key)
    guarded = [binding, result["probe_result"], *result["source_records"], *result["protected_prior_records"], *result["input_records"], result["model_binary_before"], result["model_config_before"]]
    for item in guarded:
        check(); require(prior.file_sha(item["path"]) == item["sha256"], "ORIGINAL_SOURCE_ASSET_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_SHA")
    source = source_contract(Path(result["controller_source"]["path"]).read_text(encoding="utf-8-sig"), Path(result["pure_source"]["path"]).read_text(encoding="utf-8-sig"))
    require(result["F_before"] == result["F_after"] == probe["F_before"] == probe["F_after"] and result["protected_TEST_accessed"] is result["new_backbone_training"] is result["independent_holdout"] is result["strict_K_only"] is result["whole_deployment_cost_measured"] is False, "F_FROZEN_NO_TEST_NO_HOLDOUT_OR_COST_CLAIM")
    registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    require(registry["status"] == "ALL_16_RESOLVED_BEFORE_DEV" and len(registry["conditions"]) == result["resolved_selection_locks"] == 16 and registry["DEV_opened_before_seal"] is False and result["all_16_locks_before_any_DEV"] is True, "ALL_16_RESOLUTIONS_BEFORE_DEV")
    keys = {(d, l, learn, s) for d in DATASETS for l in LAGS for learn in (False, True) for s in SEEDS}; locks = {}
    for item in registry["conditions"]:
        key = item["dataset"], item["lag"], item["learn_E"], item["seed"]; require(key not in locks and key in keys, "UNIQUE_REGISTERED_SELECTION_KEY")
        lock = prior.verified_json(item["selection_lock"], check); require(lock["sealed_unix"] <= registry["sealed_unix"] and lock["DEV_opened_before_lock"] is False, "LOCK_TIMESTAMP_BEFORE_REGISTRY")
        locks[key] = (item["selection_lock"], lock); guarded.append(item["selection_lock"])
    totals = {k: 0 for k in ("adapter_fits_checked", "checkpoint_states_checked", "checkpoint_TUNE_predictions_checked", "selected_FIT_DEV_predictions_checked", "CP0_FIT_DEV_baselines_checked", "logged_updates_checked", "score_fields_checked", "probe_cases_checked")}; totals["selection_locks_checked"] = len(locks)
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in ("probe_result", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry")}, "source_contract": source, "totals": totals, "datasets": {}, "partial_conditions": [], "protected_TEST_accessed": False, "independent_F_forward_replayed": False, "independent_backward_or_optimizer_updates_replayed": False, "logged_training_losses_numerically_recomputed": False, "scope": "Saved actual-F future reused only to verify E/lag/D arithmetic; exact saved-array scores/checkpoint selection/log counts/source and lock provenance. Probe gradient values remain recorded producer evidence, not an independent VJP replay."}
    raw = {}; initial = {}; cp0 = {}
    for data, (c, k, dev, dev_n) in DATASETS.items():
        report["datasets"][data] = {}
        for split, n in (("fit", FIT_N), ("tune", TUNE_N), (dev, dev_n)):
            record = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]["input_source"]
            raw[data, split] = prior.verified_arrays(record, check); prior.check_raw_split(raw[data, split], channels=c, windows=n)
        for lag, lags in LAGS.items():
            origin = result["initial_sources"][data][lag]; owner = science["datasets"][data]["spaces"]["PCA"]
            require(origin["E"] == owner["E_source"] and origin["D"] == owner["conditions"][lag + "_FORECAST"]["state_artifact"], "EXACT_SAME_PCA_FORECAST_LS_INITIALIZATION")
            e, d = prior.verified_arrays(origin["E"], check), prior.verified_arrays(origin["D"], check)
            initial[data, lag] = {"E": e["encoder"].astype(np.float32), "D": d["decoder_FP32"], "center": e["center"].astype(np.float32)}
            p = probe["cases"][data][lag]; require(p["status"] == "PASS" and p["F_parameters_in_optimizer"] is False and p["F_parameter_gradients_none"] is True and p["gradient_diagnostics"]["F_path_available"] is True and p["gradient_diagnostics"]["E_F_path_gradient_norm"] > 0., "RECORDED_FINITE_NONZERO_F_PATH_TECHNICAL_PROBE")
            for group in ("CP0_NumPy_parity", "canonical32_vs_batch4"):
                for value in p[group].values(): require(value["pass"] is True and value["atol"] == 1e-5 and value["rtol"] == 2e-6, "REGISTERED_PROBE_PARITY")
            pa = prior.verified_arrays(p["artifact"], check)
            require(all(np.isfinite(pa[prefix + field]).all() for prefix in ("before_", "after_") for field in ("E", "D", "center")) and math.isfinite(p["gradient_diagnostics"]["E_F_path_gradient_norm"]), "FINITE_PROBE_COEFFICIENTS_AND_RECORDED_F_GRADIENT")
            for key, value in initial[data, lag].items(): exact(pa["before_" + key], value, "probe_initial_state")
            for key in ("E", "D"): require(not np.array_equal(pa["before_" + key], pa["after_" + key]) and p[key + "_bytes_changed"] is True, "ONE_PROBE_OPTIMIZER_STEP_CHANGED_ADAPTER")
            exact(pa["before_center"], pa["after_center"], "probe_fixed_center")
            parity = prior.verified_arrays(p["CP0_artifact"], check)
            for component in ("prediction", "latent_history", "latent_future", "lag_features"):
                for prefix, group in (("NumPy_", "CP0_NumPy_parity"), ("Torch4_", "canonical32_vs_batch4")):
                    a, b = parity["Torch32_" + component], parity[prefix + component]; prior.close(a, b, "independent_saved_probe_parity")
                    error = a.astype(np.float64) - b.astype(np.float64)
                    for field, value in (("max_abs", float(np.abs(error).max())), ("rmse", float(np.sqrt(np.mean(error * error))))): prior.close(p[group][component][field], value, "saved_probe_parity_" + field, atol=1e-12, rtol=0.)
            saved = {**raw[data, "tune"], **{key: parity["Torch32_" + key] for key in ("prediction", "latent_history", "latent_future", "lag_features")}}
            replay(saved, initial[data, lag], raw[data, "tune"], lags); totals["score_fields_checked"] += check_score(p["CP0_TUNE_scores"], saved, raw[data, "tune"])
            exact(pa["batch_indices"], np.random.default_rng(SEEDS[0]).integers(0, FIT_N, size=(600, 4))[0], "probe_same_first_training_batch")
            exact(pa["FIT_channel_observed_counts"], raw[data, "fit"]["observed"].sum(axis=(0, 1)), "probe_entire_FIT_denominators")
            totals["probe_cases_checked"] += 1
    for key in sorted(keys):
        check(); data, lag, learn, seed = key; name = data + "_" + lag + ("_E_LEARN_s" if learn else "_E_FIXED_s") + str(seed)
        row = result["datasets"][data]["conditions"][name]; lock_record, lock = locks[key]
        require(row["selection_lock"] == lock_record and row["status"] == lock["status"], "RESULT_SELECTION_LOCK_STATUS")
        if row["status"] != "COMPLETE": report["partial_conditions"].append(name); continue
        require((row["dataset"], row["lag"], row["learn_E"], row["seed"]) == key and [cp["update"] for cp in row["checkpoints"]] == list(CHECKPOINTS), "FIVE_DECLARED_CP_STATES")
        require(lock["checkpoints"] == row["checkpoints"] and lock["initial_state_sha256"] == row["initial_state_sha256"] and lock["order_sha256"] == row["order_sha256"], "LOCKED_CP_METADATA_INITIALIZATION_AND_ORDER_UNCHANGED")
        require(row["initial_state_sha256"] == state_hash(initial[data, lag]), "IDENTICAL_INITIAL_STATE_FOR_FIXED_LEARNED_AND_SEEDS")
        states = {}; scores = []; latents = {}
        for cp in row["checkpoints"]:
            state = prior.verified_arrays(cp["state"], check); saved = prior.verified_arrays(cp["artifact"], check)
            exact(state["center"], initial[data, lag]["center"], "center_always_fixed")
            if not learn or cp["update"] == 0: exact(state["E"], initial[data, lag]["E"], "fixed_or_CP0_E")
            if cp["update"] == 0:
                for field in state: exact(state[field], initial[data, lag][field], "CP0_same_initial_" + field)
                if (data, lag) in cp0: exact(saved["prediction"], cp0[data, lag], "same_backend_CP0_all_four_conditions")
                else: cp0[data, lag] = saved["prediction"]
            score = replay(saved, state, raw[data, "tune"], LAGS[lag]); unit_score(cp["tune"], score["normalized"])
            totals["score_fields_checked"] += 6 + check_score(cp, saved, raw[data, "tune"])
            prior.close(cp["TUNE_latent_variance"], np.var(saved["latent_history"].astype(np.float64), axis=(0, 1)), "TUNE_latent_variance", atol=1e-10, rtol=0.)
            states[cp["update"]] = state; latents[cp["update"]] = saved["latent_history"]; scores.append((score["normalized"]["mse"], cp["update"])); totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_predictions_checked"] += 1
        selected = min(scores)[1]; cp = next(p for p in row["checkpoints"] if p["update"] == selected)
        require(row["selected_update"] == lock["selected_update"] == selected and row["selected_state"] == lock["selected_state"] == cp["state"] and row["selected_state_sha256"] == state_hash(states[selected]) and row["CP0_TUNE"] == row["checkpoints"][0], "TUNE_ONLY_EARLIEST_EXACT_TIE_SELECTED_CP")
        unit_score(row["selected_tune"], dict(cp["tune"], per_channel_mse=[0. if v is None else v for v in cp["tune"]["per_channel_mse"]], per_channel_mae=[0. if v is None else v for v in cp["tune"]["per_channel_mae"]]))
        totals["logged_updates_checked"] += check_logs(row, raw[data, "fit"]); totals["adapter_fits_checked"] += 1
        result_scores = {}
        for label, split in (("FIT", "fit"), ("DEV", DATASETS[data][2])):
            entry = row[label]; saved = prior.verified_arrays(entry["artifact"], check); replay(saved, states[selected], raw[data, split], LAGS[lag])
            totals["score_fields_checked"] += check_score(entry, saved, raw[data, split]); totals["selected_FIT_DEV_predictions_checked"] += 1
            if label == "DEV": require(Path(entry["artifact"]["path"]).stat().st_mtime >= registry["sealed_unix"], "DEV_ARTIFACT_AFTER_ALL_LOCKS")
            result_scores[split] = entry["scores_all_observed"]["normalized"]
        totals["score_fields_checked"] += prior.compare_scores(row["RAW_TUNE_scores"], prior.scores(raw[data, "tune"]["raw_forecast"], raw[data, "tune"]["target"], raw[data, "tune"]["observed"], raw[data, "tune"]["std"]), "same_Raw_TUNE")
        report["datasets"][data][name] = {"selected_update": selected, "CP0_TUNE": row["checkpoints"][0]["tune"], "selected_TUNE": cp["tune"], "selected_split_scores": result_scores, "order_sha256": row["order_sha256"], "parameter_change": selected_geometry(initial[data, lag], states[selected], latents[selected])}
    for data, (_, _, dev, _) in DATASETS.items():
        for lag in LAGS:
            for split in ("fit", dev):
                base = result["CP0_LS_and_RAW_baselines"][data][lag][split]; require(base["all_16_locks_already_sealed"] is True, "BASELINE_AFTER_ALL_LOCKS")
                saved = prior.verified_arrays(base["CP0_LS"]["artifact"], check); replay(saved, initial[data, lag], raw[data, split], LAGS[lag]); totals["score_fields_checked"] += check_score(base["CP0_LS"], saved, raw[data, split]); totals["CP0_FIT_DEV_baselines_checked"] += 1
                totals["score_fields_checked"] += prior.compare_scores(base["RAW_scores"], prior.scores(raw[data, split]["raw_forecast"], raw[data, split]["target"], raw[data, split]["observed"], raw[data, split]["std"]), "same_Raw_baseline")
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "POST_AUDIT_OLD_BYTES_UNCHANGED")
    if result["status"] == "COMPLETE":
        require(totals["adapter_fits_checked"] == result["adaptation_training_fits"] == 16 and totals["logged_updates_checked"] == result["training_updates_completed"] == 9600 and totals["checkpoint_states_checked"] == result["checkpoint_states_saved"] == 80 and totals["selected_FIT_DEV_predictions_checked"] == 32 and totals["CP0_FIT_DEV_baselines_checked"] == 8 and not result["failed_paths"] and not report["partial_conditions"], "FULL_16_FITS_9600_LOGGED_STEPS_80_CP_COVERAGE")
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_INDEPENDENT_ADAPTATION_AUDIT_OUTPUT")
    meter = Meter("audit_linear_e_" + output.stem, gpu=False); failure = None; source = file_record(__file__)
    try:
        meter.protected_check(); metadata(path, meter.check); initialize_numpy()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2): report = audit_result(path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally:
        report["audit_source"] = source; report["resource"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
