"""Independent saved-array stop-gradient audit; no F, backward or optimizer replay."""
from pathlib import Path
import argparse
import ast
import json
import math
import sys

import audit_linear_e_adaptation_v1 as base

prior, require, HERE = base.prior, base.require, base.HERE
DATASETS, SEEDS = base.DATASETS, base.SEEDS
RESULT_SHA = "e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a"
CONTROLLER_SHA = "0b13a1a6d2d2ac73af74a1e9bcc1383a1404b12016fc42a3cea0dbfdaf1a1ed5"
PARENT_AUDIT_SHA = "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
REPORT_SHA = "eff21bf19811b2f5a09945494aeca5817d22a2dd4f6d7da8b406a46a8f467230"
MANIFEST_SHA = "67b73f080dedb654b26bb49c14e445ab7914af6d4900b504a2bf45aa0654a488"
REPORT_AUDIT_SHA = "755574afa0e1e1a85f4e29de6453da18080f15fbdc4589a4b248bb61ded6ff05"
GRADIENT_FIELDS = ("full_E_gradient", "full_D_gradient", "F_future_gradient", "F_input_gradient", "F_path_E_gradient", "direct_E_gradient", "stop_E_gradient", "stop_D_gradient")
COMPONENTS = ("prediction", "latent_history", "latent_future", "lag_features")
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy(); np = sys.modules["numpy"]


def analytic_probe(arrays, entry, fit, initial, seed):
    """Differentiate L0 in NumPy; the saved F-input VJP is conditional evidence."""
    ids = np.random.default_rng(seed).integers(0, len(fit["history"]), size=(600, 4))[0]
    base.exact(arrays["batch_indices"], ids, "probe_first_same_seed_minibatch")
    counts = fit["observed"].sum((0, 1)); base.exact(arrays["FIT_channel_observed_counts"], counts, "probe_full_FIT_counts")
    for field in initial: base.exact(arrays["initial_" + field], initial[field], "probe_initial_" + field)
    require(entry["initial_state_sha256"] == base.state_hash(initial), "PROBE_INITIAL_STATE_HASH")
    require(entry["status"] == "PASS" and entry["optimizer_updates"] == 0 and entry["forward_byte_exact"] is entry["finite_gradient_vectors"] is entry["F_parameters_frozen_and_grad_none"] is entry["parameter_grad_fields_none"] is True, "RECORDED_MATCHED_FORWARD_ZERO_UPDATE_PROBE")
    x = fit["history"][ids]; z = arrays["full_latent_history"]; future = arrays["full_latent_future"]; prediction = arrays["full_prediction"]
    for field in ("prediction", "latent_history", "latent_future"): base.exact(arrays["full_" + field], arrays["stop_" + field], "probe_full_stop_" + field)
    prior.close(z, (x - initial["center"]) @ initial["E"], "probe_independent_E_encoding")
    require(z.dtype == future.dtype == prediction.dtype == np.float32 and z.shape == (len(ids), x.shape[1], initial["E"].shape[1]) and future.shape == (len(ids), 48, initial["E"].shape[1]) and prediction.shape == (len(ids), 48, x.shape[2]), "PROBE_FINITE_FP32_SHAPES")
    delta = future - z[:, -1:]; prior.close(prediction, fit["raw_last"][ids, None] + delta @ initial["D"], "probe_current_C_anchor_and_L0_D")
    mask = fit["observed"][ids]; target = fit["target"][ids]
    require(np.isfinite(x).all() and np.isfinite(target[mask]).all(), "PROBE_TRUE_OBSERVED_TARGETS_ONLY")
    error = np.zeros(prediction.shape, np.float64); error[mask] = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
    active = counts > 0; require(active.any(), "AT_LEAST_ONE_FULL_FIT_ACTIVE_CHANNEL")
    weight = np.zeros(len(counts), np.float64); weight[active] = len(fit["history"]) / len(ids) / active.sum() / counts[active]
    prior.close(entry["loss"], float(np.sum(error * error * weight)), "independent_fixed_FIT_macro_probe_loss")
    output_gradient = 2 * error * weight; d = initial["D"].astype(np.float64)
    future_gradient = output_gradient @ d.T
    direct = (x[:, -1] - initial["center"]).astype(np.float64).T @ (-future_gradient.sum(1))
    via = (x - initial["center"]).astype(np.float64).reshape(-1, x.shape[2]).T @ arrays["F_input_gradient"].astype(np.float64).reshape(-1, initial["E"].shape[1])
    d_gradient = np.einsum("nhk,nhc->kc", delta.astype(np.float64), output_gradient)
    expected = {"F_future_gradient": future_gradient, "full_D_gradient": d_gradient, "stop_D_gradient": d_gradient, "direct_E_gradient": direct, "stop_E_gradient": direct, "F_path_E_gradient": via, "full_E_gradient": direct + via}
    shapes = {**{key: value.shape for key, value in expected.items()}, "F_input_gradient": z.shape}
    for key in GRADIENT_FIELDS:
        value = arrays[key]; require(value.dtype == np.float32 and value.shape == shapes[key] and np.isfinite(value).all(), "FINITE_GRADIENT_VECTOR:" + key)
        prior.close(entry["gradient_norms"][key], float(np.linalg.norm(value.astype(np.float64))), "saved_vector_norm:" + key, atol=1e-12, rtol=0.)
    base.exact(arrays["direct_E_gradient"], arrays["full_E_gradient"] - arrays["F_path_E_gradient"], "producer_direct_is_saved_total_minus_saved_F_path")
    errors = {key: prior.close(arrays[key], value, "independent_analytic:" + key) for key, value in expected.items()}
    for label, a, b in (("stop_E_equals_full_minus_F_path", arrays["stop_E_gradient"], arrays["direct_E_gradient"]), ("stop_D_equals_full_D", arrays["stop_D_gradient"], arrays["full_D_gradient"])):
        recorded = entry["gradient_checks"][label]; require(recorded["pass"] is True and recorded["atol"] == 1e-5 and recorded["rtol"] == 2e-6, "REGISTERED_GRADIENT_COMPARISON")
        prior.close(a, b, "saved_pair:" + label); error = a.astype(np.float64) - b.astype(np.float64)
        for key, value in (("max_abs", float(np.abs(error).max())), ("rmse", float(np.sqrt(np.mean(error * error))))): prior.close(recorded[key], value, "gradient_comparison_" + key, atol=1e-12, rtol=0.)
    return {"analytic_comparisons": len(expected), "gradient_vector_fields": len(GRADIENT_FIELDS), "analytic_max_abs_errors": errors, "saved_F_input_VJP_independently_replayed": False}


def cp0_parity(row, previous, initial, tune, check):
    require(row["reference_CP0"] == previous["CP0_TUNE"]["artifact"] and row["probe"]["reference_CP0"] == row["reference_CP0"] and row["probe"]["CP0_TUNE_parity_artifact"] == row["CP0_TUNE_parity_artifact"], "EXACT_REUSED_FULL_CP0_BINDING")
    saved = prior.verified_arrays(row["CP0_TUNE_parity_artifact"], check); old = prior.verified_arrays(row["reference_CP0"], check)
    base.exact(saved["origins"], tune["origins"], "CP0_same_origins")
    for field in COMPONENTS:
        base.exact(saved["full_" + field], saved["stop_" + field], "CP0_full_stop_" + field); base.exact(saved["full_" + field], old[field], "CP0_full_parent_" + field)
        require(row["CP0_TUNE_byte_checks"][field] == {"full_stop_byte_exact": True, "full_parent_byte_exact": True}, "RECORDED_CP0_COMPONENT_PARITY")
    base.replay({**tune, **{field: saved["stop_" + field] for field in COMPONENTS}}, initial, tune, (0,))
    return 2 * len(COMPONENTS)


def check_logs(row, fit):
    count = base.check_logs(row, fit)
    logs = [json.loads(line) for line in Path(row["updates_log"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    for item in logs:
        scale = min(1., 1. / (item["global_gradient_norm_before_clip"] + 1e-6))
        require(item["D_gradient_scaling_is_AdamW_step_norm"] is False, "CLIPPED_GRADIENT_IS_NOT_OPTIMIZER_STEP")
        prior.close(item["global_clip_scale_from_logged_norm"], scale, "logged_clip_scale_formula", atol=1e-12, rtol=0.)
        prior.close(item["D_gradient_norm_after_clip_formula"], scale * item["D_gradient_norm_before_clip"], "logged_D_clipped_norm_formula", atol=1e-12, rtol=0.)
    return count


def source_contract(text):
    tree = ast.parse(text); functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FDetach")
    call = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    require(ast.dump(call.body[0]) == ast.dump(ast.parse("return self.bolt(z).detach()").body[0]), "DETACH_ONLY_ACTUAL_F_OUTPUT_SOURCE")
    probe = functions["gradient_probe"]
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("step", "backward") for n in ast.walk(probe)), "NO_PROBE_OPTIMIZER_OR_BACKWARD_MUTATION_SOURCE")
    execute = ast.unparse(functions["execute"]); resolve = ast.unparse(functions["resolve_four"])
    require("train_condition(stopped, stop_bolt, fit, tune, seed, check, checkpoint)" in execute and "stop_bolt = FDetach(bolt)" in execute and "ALL_FOUR_LOCKS_BEFORE_DEV_ARRAY_IO" in execute and execute.index("resolved = resolve_four(") < execute.index("for key, row in resolved.items():") and "report['selection_registry'] = registry(rows)" in resolve, "ACTUAL_F_SAME_TRAINER_AND_REGISTRY_BEFORE_DEV_SOURCE")
    return {"detach_only_F_output": True, "same_original_optimizer_and_fixed_FIT_loss_source": True, "probe_optimizer_updates": 0, "registry_before_DEV_source": True}


def metadata(path, check):
    require(isinstance(RESULT_SHA, str) and len(RESULT_SHA) == 64, "ACTUAL_STOP_GRADIENT_RESULT_NOT_YET_REGISTERED")
    binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_STOP_GRADIENT_RESULT_SHA")
    result = prior.verified_json(binding, check)
    require(result["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and result["controller_source"]["sha256"] == CONTROLLER_SHA, "COMPLETED_OR_PRESERVED_PARTIAL_EXECUTED_SOURCE")
    for key, sha in (("full_gradient_parent", base.TRAIN_SHA), ("full_gradient_parent_audit", PARENT_AUDIT_SHA), ("parent_report", REPORT_SHA), ("parent_report_manifest", MANIFEST_SHA), ("parent_report_audit", REPORT_AUDIT_SHA)):
        require(result[key]["sha256"] == sha, "EXACT_PARENT_BINDING:" + key)
    return binding, result


def audit_result(path, meter):
    check = meter.check; binding, result = metadata(path, check)
    parent = prior.verified_json(result["full_gradient_parent"], check); accepted = prior.verified_json(result["full_gradient_parent_audit"], check)
    require(parent["status"] == "COMPLETE" and accepted["status"] == "PASS" and accepted["diagnostic"] == result["full_gradient_parent"], "EXACT_PREVIOUS_FULL_GRADIENT_PASS")
    report_audit = prior.verified_json(result["parent_report_audit"], check)
    require(report_audit["status"] == "PASS" and all(report_audit[key][field] == result[bound][field] for key, bound in (("report", "parent_report"), ("manifest", "parent_report_manifest")) for field in ("path", "sha256")), "EXACT_PREVIOUS_REPORT_PASS")
    science = prior.verified_json(result["causal_lag_diagnostic"], check); science_audit = prior.verified_json(result["causal_lag_audit"], check)
    require(science["status"] == "COMPLETE" and science_audit["status"] == "PASS" and science_audit["diagnostic"] == result["causal_lag_diagnostic"] and all(result[k] == parent[k] for k in ("causal_lag_diagnostic", "causal_lag_audit")), "SAME_ACCEPTED_CAUSAL_RAW_INPUT_LINEAGE")
    guarded = [binding, *[result[k] for k in ("full_gradient_parent", "full_gradient_parent_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "causal_lag_diagnostic", "causal_lag_audit", "plan")], *result["source_records"], *result["protected_prior_records"], *result["input_records"], result["model_binary_before"], result["model_config_before"]]
    for item in guarded:
        check(); require(prior.file_sha(item["path"]) == item["sha256"], "SOURCE_ASSET_OR_PARENT_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_SHA")
    source = source_contract(Path(result["controller_source"]["path"]).read_text(encoding="utf-8-sig"))
    base.source_contract(Path(parent["controller_source"]["path"]).read_text(encoding="utf-8-sig"), Path(parent["pure_source"]["path"]).read_text(encoding="utf-8-sig"))
    require(parent["controller_source"]["sha256"] == base.CONTROLLER_SHA and parent["pure_source"]["sha256"] == base.PURE_SHA and result["F_before"] == result["F_after"] == parent["F_before"] == parent["F_after"], "IMMUTABLE_REUSED_TRAINER_PURE_AND_F_SIGNATURES")
    require(all(result[k] is False for k in ("new_backbone_training", "protected_TEST_accessed", "independent_holdout", "strict_K_only", "whole_deployment_cost_measured")) and result["new_full_gradient_fits"] == result["probe_optimizer_updates"] == 0 and result["full_gradient_conditions_reused"] == 4, "NO_NEW_FULL_GRADIENT_OR_BACKBONE_FIT_TEST_OR_COST_CLAIM")
    registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    require(registry["status"] == "ALL_FOUR_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and result["all_4_locks_before_DEV"] is True and len(registry["conditions"]) == result["resolved_selection_locks"] == 4, "FOUR_RESOLUTIONS_REGISTRY_BEFORE_DEV")
    locks = {}
    for item in registry["conditions"]:
        key = item["dataset"], item["seed"]; require(key not in locks and key[0] in DATASETS and key[1] in SEEDS, "UNIQUE_REGISTERED_CONDITION")
        lock = prior.verified_json(item["selection_lock"], check); require(lock["sealed_unix"] <= registry["sealed_unix"] and lock["DEV_opened_before_lock"] is False, "SELECTION_LOCK_PRECEDES_REGISTRY")
        locks[key] = item, lock; guarded.append(item["selection_lock"])
    totals = {key: 0 for key in ("stop_gradient_fits_checked", "checkpoint_states_checked", "checkpoint_TUNE_predictions_checked", "selected_FIT_DEV_predictions_checked", "logged_updates_checked", "probe_cases_checked", "reused_full_gradient_conditions_checked", "CP0_full_stop_parent_component_byte_comparisons_checked", "saved_gradient_vector_fields_checked", "analytic_gradient_comparisons_checked", "score_fields_checked")}; totals["selection_locks_checked"] = len(locks)
    keys = ("full_gradient_parent", "full_gradient_parent_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "plan")
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in keys}, "source_contract": source, "totals": totals, "datasets": {}, "partial_conditions": [], "independent_F_forward_replayed": False, "independent_F_input_VJP_replayed": False, "independent_backward_or_optimizer_updates_replayed": False, "logged_training_losses_numerically_recomputed": False, "protected_TEST_accessed": False, "scope": "Saved actual F futures validate independent NumPy E/L0/D arithmetic, scores, CP selection and registered source/lock/log provenance. Probe output and D/direct-E derivatives are analytic; F-path E chain rule is conditional on the recorded F-input VJP, whose F Jacobian is not independently replayed. No independent optimizer, per-update training loss or F-gradient trajectory replay; no holdout or F-value necessity proof."}
    for data, (c, k, dev, ndev) in DATASETS.items():
        raw = {}
        for split, n in (("fit", base.FIT_N), ("tune", base.TUNE_N), (dev, ndev)):
            record = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]["input_source"]
            raw[split] = prior.verified_arrays(record, check); prior.check_raw_split(raw[split], channels=c, windows=n)
        report["datasets"][data] = {}
        for seed in SEEDS:
            check(); row = result["datasets"][data]["conditions"][str(seed)]; registered, lock = locks[data, seed]
            require(row["selection_lock"] == registered["selection_lock"] and row["status"] == lock["status"] == registered["status"], "RESULT_AND_LOCK_RESOLUTION_STATUS")
            if row["status"] != "COMPLETE": report["partial_conditions"].append({"dataset": data, "seed": seed}); continue
            name = data + "_L0_E_LEARN_s" + str(seed); previous = parent["datasets"][data]["conditions"][name]
            require((row["dataset"], row["seed"], row["lag"], row["learn_E"]) == (data, seed, "L0", True) and row["full_gradient_reference"] == {"condition": name, "selected_update": previous["selected_update"], "selected_state": previous["selected_state"]}, "EXACT_L0_REUSED_FULL_GRADIENT_PAIR")
            require(row["checkpoints"] == lock["checkpoints"] and row["order_sha256"] == lock["order_sha256"] == previous["order_sha256"] and row["initial_state_sha256"] == lock["initial_state_sha256"] == previous["initial_state_sha256"] and [cp["update"] for cp in row["checkpoints"]] == list(base.CHECKPOINTS), "LOCKED_SAME_INITIALIZATION_ORDER_AND_FIVE_CP")
            initial = prior.verified_arrays(previous["CP0_TUNE"]["state"], check); require(base.state_hash(initial) == row["initial_state_sha256"], "SAME_INITIAL_STATE_HASH")
            totals["CP0_full_stop_parent_component_byte_comparisons_checked"] += cp0_parity(row, previous, initial, raw["tune"], check)
            probe = analytic_probe(prior.verified_arrays(row["probe"]["artifact"], check), row["probe"], raw["fit"], initial, seed)
            totals["probe_cases_checked"] += 1; totals["saved_gradient_vector_fields_checked"] += probe["gradient_vector_fields"]; totals["analytic_gradient_comparisons_checked"] += probe["analytic_comparisons"]
            states, latents = {}, {}
            for cp in row["checkpoints"]:
                state = prior.verified_arrays(cp["state"], check); saved = prior.verified_arrays(cp["artifact"], check); base.exact(state["center"], initial["center"], "fixed_center_buffer")
                if cp["update"] == 0:
                    for key in initial: base.exact(state[key], initial[key], "same_CP0_" + key)
                    for key in COMPONENTS: base.exact(saved[key], prior.verified_arrays(row["reference_CP0"], check)[key], "CP0_saved_train_matches_parent_" + key)
                score = base.replay(saved, state, raw["tune"], (0,)); base.unit_score(cp["tune"], score["normalized"])
                totals["score_fields_checked"] += 6 + base.check_score(cp, saved, raw["tune"]); totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_predictions_checked"] += 1
                states[cp["update"]], latents[cp["update"]] = state, saved["latent_history"]
            selected_cp = min(row["checkpoints"], key=lambda cp: (cp["tune"]["mse"], cp["update"])); selected = selected_cp["update"]
            require(row["selected_update"] == lock["selected_update"] == selected and row["selected_state"] == lock["selected_state"] == selected_cp["state"] and row["selected_state_sha256"] == lock["selected_state_sha256"] == base.state_hash(states[selected]) and row["selected_tune"] == selected_cp["tune"], "TUNE_ONLY_EARLIEST_EXACT_TIE_SELECTED_CP")
            totals["logged_updates_checked"] += check_logs(row, raw["fit"]); totals["stop_gradient_fits_checked"] += 1; totals["reused_full_gradient_conditions_checked"] += 1
            splits = {}
            for label, split in (("FIT", "fit"), ("DEV", dev)):
                entry = row[label]; saved = prior.verified_arrays(entry["artifact"], check); base.replay(saved, states[selected], raw[split], (0,))
                totals["score_fields_checked"] += base.check_score(entry, saved, raw[split]); totals["selected_FIT_DEV_predictions_checked"] += 1; splits[split] = entry["scores_all_observed"]["normalized"]
                if label == "DEV": require(Path(entry["artifact"]["path"]).stat().st_mtime >= registry["sealed_unix"], "DEV_SAVED_AFTER_ALL_FOUR_LOCKS")
            report["datasets"][data][str(seed)] = {"selected_update": selected, "CP0_TUNE": row["checkpoints"][0]["tune"], "selected_TUNE": row["selected_tune"], "selected_split_scores": splits, "full_gradient_reference": row["full_gradient_reference"], "gradient_probe": probe, "order_sha256": row["order_sha256"], "parameter_change": base.selected_geometry(initial, states[selected], latents[selected])}
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "POST_AUDIT_PRIOR_FILES_UNCHANGED")
    totals["protected_prior_file_hashes_checked"] = len(guarded)
    if result["status"] == "COMPLETE": require(totals["stop_gradient_fits_checked"] == result["stop_gradient_training_fits"] == 4 and totals["checkpoint_states_checked"] == result["checkpoint_states_saved"] == 20 and totals["logged_updates_checked"] == result["training_updates_completed"] == 2400 and totals["selected_FIT_DEV_predictions_checked"] == 8 and totals["analytic_gradient_comparisons_checked"] == 28 and not result["failed_paths"] and not report["partial_conditions"], "FOUR_FITS_2400_UPDATES_20_CP_COMPLETE_SCOPE")
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_INDEPENDENT_STOP_GRADIENT_AUDIT_OUTPUT")
    source = base.file_record(__file__); meter = Meter("audit_F_input_gradient_" + output.stem, gpu=False); failure = None
    try:
        meter.protected_check(); metadata(path, meter.check); initialize_numpy()
        sys.path.insert(0, str(base.ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
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
