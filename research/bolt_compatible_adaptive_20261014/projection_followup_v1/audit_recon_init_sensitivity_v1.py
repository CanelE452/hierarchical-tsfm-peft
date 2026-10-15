"""Saved-training audit plus explicitly new selected-space calculations; no F replay."""
from pathlib import Path
import argparse
import ast
import json
import sys

import audit_linear_e_adaptation_v1 as base
import audit_causal_lag_objective_v2 as geometry

HERE, require, prior = base.HERE, base.require, base.prior
SEEDS, LAGS = (92601, 92602), (0, 1, 2, 4, 8, 16, 32)
RESULT_SHA = "f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8"
CONTROLLER_SHA = "a00300046d2236c46be7d6d532cead122cd178f066a62fd4bbbea76566d2534b"
SPACE_SOURCE_SHA = "08f5806c80b825c9510eaaad9eaf85c9fb0e9633668e3d33be0317162e50f795"
PARENT_SHA = {"full_gradient_parent": base.TRAIN_SHA, "full_gradient_parent_audit": "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d", "learned_output_space_diagnostic": "cf042cc764ff36cbb877211777721783efadbaab6ba8741268eaccba26cbb34b", "learned_output_space_audit": "1b0dbacb711becce70ac877825cd53b84069918ff62ac79dc9b9b5d8ac410738"}
COMPONENTS = ("prediction", "latent_history", "latent_future", "lag_features")
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy(); geometry.initialize_numpy(); np = sys.modules["numpy"]


def initial_state(e, d, forecast_initial):
    state = {"E": e["encoder"].astype(np.float32), "D": d["decoder_FP32"], "center": e["center"].astype(np.float32)}
    base.exact(d["lags"], np.asarray(LAGS, np.int64), "RECON_initial_L7_lags")
    require(state["D"].dtype == np.float32 and state["D"].shape == (len(LAGS)*state["E"].shape[1], len(state["center"])) and all(np.isfinite(v).all() for v in state.values()), "FINITE_EXISTING_RECON_D_INITIALIZATION")
    for key in ("E", "center"): base.exact(state[key], forecast_initial[key], "same_PCA_initial_"+key)
    return state


def check_probe(row, arrays, reference, initial, tune):
    probe = row["probe"]; require(probe["status"] == "PASS" and probe["optimizer_updates"] == 0 and probe["initial_state_sha256"] == base.state_hash(initial), "TWO_RECORDED_ZERO_UPDATE_CP0_PROBES")
    for key, original in (("target", "target"), ("observed", "observed"), ("currentC", "raw_last"), ("origins", "origins"), ("FIT_mean", "mean"), ("FIT_std", "std")): base.exact(reference[key], tune[original], "CP0_RECON_original_"+key)
    base.exact(arrays["origins"], tune["origins"], "CP0_parity_origins")
    expected = {"prediction": reference["ACTUAL_F_prediction_FP32"], "latent_history": reference["latent_history"], "latent_future": reference["actual_F_future"], "lag_features": geometry.delta_features(reference["latent_history"], reference["actual_F_future"], LAGS)}
    for key, value in expected.items():
        base.exact(arrays["previous_RECON_"+key], value, "exact_original_cached_RECON_"+key); actual = arrays["Torch_"+key]
        prior.close(actual, value, "registered_Torch_NumPy_CP0:"+key); error = actual.astype(np.float64) - value.astype(np.float64); recorded = probe["checks"][key]
        require(recorded["pass"] is True and recorded["atol"] == 1e-5 and recorded["rtol"] == 2e-6, "ORIGINAL_CP0_PARITY_TOLERANCES")
        for field, expected_error in (("max_abs", float(np.abs(error).max())), ("rmse", float(np.sqrt(np.mean(error*error))))): prior.close(recorded[field], expected_error, "CP0_error_summary:"+field, atol=1e-12, rtol=0.)
    base.replay({**tune, **{key: arrays["Torch_"+key] for key in COMPONENTS}}, initial, tune, LAGS)
    return len(expected)


def new_space_diagnostic(saved, state, raw, selected_update, destination):
    import run_learned_output_space_diagnostic_v1 as calculator
    import fixed_decoder_span_v1 as io
    require(prior.file_sha(calculator.__file__) == SPACE_SOURCE_SHA and not Path(destination).exists(), "FROZEN_NEW_SPACE_CALCULATOR_AND_FRESH_ARRAY_OUTPUT")
    calculator.initialize_numpy(); cell, arrays = calculator.diagnose_cell(saved, state, raw, LAGS)
    cell["selected_E_D_previously_supervised_by_FIT_future_targets"] = selected_update > 0
    cell.update(calculation_role="NEW_OUTPUT_SPACE_DIAGNOSTIC_COMPUTED_DURING_THIS_AUDIT", selected_update=selected_update, original_training_verification_replaced_by_this_calculation=False, independent_external_replay_of_new_diagnostic_claimed=False, numerical_paths="Existing frozen calculator: pure SVD and prior independent exact-zero-reduced LS/full SVD; not a third external replay", calculator_source=base.file_record(calculator.__file__), artifact=io.save_npz(destination, arrays))
    return cell


def metadata(path, check):
    require(isinstance(RESULT_SHA, str) and len(RESULT_SHA) == 64, "ACTUAL_RECON_INITIALIZATION_RESULT_NOT_YET_REGISTERED")
    binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_RECON_INITIALIZATION_RESULT")
    result = prior.verified_json(binding, check); require(result["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and result["controller_source"]["sha256"] == CONTROLLER_SHA, "PRESERVED_COMPLETE_OR_PARTIAL_EXECUTED_SOURCE")
    for key, sha in PARENT_SHA.items(): require(result[key]["sha256"] == sha, "EXACT_ACCEPTED_PARENT:"+key)
    return binding, result


def source_contract(result, parent):
    text = Path(result["controller_source"]["path"]).read_text(encoding="utf-8-sig"); functions = {n.name: ast.unparse(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    require("train_condition(model, bolt, fit, tune, seed, check, checkpoint)" in functions["execute"] and "TWO_LOCK_REGISTRY_BEFORE_ANY_DEV_ARRAY" in functions["execute"] and functions["execute"].index("resolved = resolve_two(") < functions["execute"].index("for seed, row in resolved.items():") and "report['selection_registry'] = registry(rows)" in functions["resolve_two"], "SAME_FULL_GRADIENT_TRAINER_AND_TWO_LOCKS_BEFORE_DEV_SOURCE")
    base.source_contract(Path(parent["controller_source"]["path"]).read_text(encoding="utf-8-sig"), Path(parent["pure_source"]["path"]).read_text(encoding="utf-8-sig"))


def audit_result(path, meter, diagnostic_dir):
    check = meter.check; binding, result = metadata(path, check); parents = {key: prior.verified_json(result[key], check) for key in PARENT_SHA}
    parent = parents["full_gradient_parent"]; accepted = parents["full_gradient_parent_audit"]
    require(parent["status"] == "COMPLETE" and accepted["status"] == "PASS" and accepted["diagnostic"] == result["full_gradient_parent"] and parents["learned_output_space_audit"]["status"] == "PASS" and parents["learned_output_space_audit"]["diagnostic"] == result["learned_output_space_diagnostic"], "PRIOR_TRAINING_AND_SPACE_PASSES_REUSED")
    require(result["F_before"] == result["F_after"] == parent["F_before"] == parent["F_after"] and result["probe_optimizer_updates"] == result["new_forecast_initialization_fits"] == 0 and result["forecast_initialization_conditions_reused"] == 2, "SAME_F_AND_NO_NEW_FORECAST_INITIALIZATION_CONTROL_FITS")
    for key in ("new_backbone_training", "new_D_initialization_fit", "protected_TEST_accessed", "independent_holdout", "strict_K_only", "whole_deployment_cost_measured"): require(result[key] is False, "TRAINING_SCOPE:"+key)
    require(result["causal_lag_diagnostic"] == parent["causal_lag_diagnostic"] and result["causal_lag_audit"] == parent["causal_lag_audit"], "SAME_ORIGINAL_RECON_AND_RAW_INPUT_LINEAGE")
    science = prior.verified_json(result["causal_lag_diagnostic"], check); owner = science["datasets"]["robin"]["spaces"]["PCA"]; recon = owner["conditions"]["L7_RECON"]
    guarded = [binding, *[result[k] for k in PARENT_SHA], result["causal_lag_diagnostic"], result["plan"], *result["source_records"]]
    for item in guarded:
        require(prior.file_sha(item["path"]) == item["sha256"], "PARENT_OR_SOURCE_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_SHA")
    source_contract(result, parent); registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    require(registry["status"] == "ALL_TWO_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == result["resolved_selection_locks"] == 2 and result["all_2_locks_before_DEV"] is True, "ALL_TWO_SELECTION_LOCKS_BEFORE_DEV")
    locks = {}
    for item in registry["conditions"]:
        require(item["dataset"] == "robin" and item["seed"] in SEEDS and item["seed"] not in locks, "UNIQUE_TWO_SEED_LOCKS")
        lock = prior.verified_json(item["selection_lock"], check); require(lock["sealed_unix"] <= registry["sealed_unix"] and lock["DEV_opened_before_lock"] is False, "LOCK_TIMESTAMP_BEFORE_DEV_REGISTRY")
        locks[item["seed"]] = item, lock; guarded.append(item["selection_lock"])
    cache = {}
    def load(record):
        key = record["path"], record["sha256"]
        if key not in cache: cache[key] = prior.verified_arrays(record, check); guarded.append(record)
        return cache[key]
    raw = {split: load(recon["splits"][split]["input_source"]) for split in ("fit", "tune", "val")}
    for split, n in (("fit", 256), ("tune", 64), ("val", 30)): prior.check_raw_split(raw[split], channels=17, windows=n)
    totals = {key: 0 for key in ("recon_initialization_fits_checked", "checkpoint_states_checked", "checkpoint_TUNE_predictions_checked", "selected_FIT_DEV_predictions_checked", "logged_updates_checked", "CP0_probe_cases_checked", "CP0_parity_components_checked", "reused_FORECAST_initialization_conditions_checked", "training_score_fields_checked", "new_space_diagnostic_cells_computed", "new_space_LS_SVD_crosschecks_computed")}; totals["selection_locks_checked"] = len(locks)
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "training_verification_status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL", "new_space_diagnostic_status": "RUNNING", "diagnostic": binding, **{key: result[key] for key in (*PARENT_SHA, "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "plan")}, "totals": totals, "datasets": {"robin": {"conditions": {}}}, "partial_paths": [], "new_output_space_diagnostics": {}, "new_output_space_diagnostic_failures": [], "protected_TEST_accessed": False, "independent_F_backward_optimizer_or_training_loss_replayed": False, "new_whole_deployment_cost_measured": False, "scope": "Saved E/lag/D arithmetic, actual score/CP/order/log/lock and existing RECON initialization audit; no training/F/VJP replay. Selected-space diagnostics are NEW computations in this audit using two frozen math paths, not an independent external replay of a preexisting diagnostic; normalized retained numeric-space scope only."}
    Path(diagnostic_dir).mkdir(parents=True, exist_ok=False)
    def audit_seed(seed):
        check(); row = result["datasets"]["robin"]["conditions"][str(seed)]; item, lock = locks[seed]
        require(row["selection_lock"] == item["selection_lock"] and row["status"] == lock["status"] == item["status"], "UNCHANGED_RESOLUTION_STATUS")
        if row["status"] != "COMPLETE": report["partial_paths"].append({"seed": seed, "status": row["status"]}); return
        previous = parent["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s"+str(seed)]
        require(row["dataset"] == "robin" and row["lag"] == "L7" and row["learn_E"] is True and row["seed"] == seed and row["order_sha256"] == lock["order_sha256"] == previous["order_sha256"] and row["checkpoints"] == lock["checkpoints"] and row["full_gradient_reference"] == {"condition": "robin_L7_E_LEARN_s"+str(seed), "selected_update": previous["selected_update"], "selected_state": previous["selected_state"], "CP0_TUNE": previous["CP0_TUNE"]}, "SAME_L7_RECIPE_ORDER_AND_REUSED_FORECAST_PAIR")
        require(row["initial_sources"] == {"PCA_E": owner["E_source"], "RECON_D": recon["state_artifact"], "FORECAST_LS_CP0": previous["CP0_TUNE"]["state"]}, "EXACT_EXISTING_INITIALIZATION_ASSETS")
        initial = initial_state(load(owner["E_source"]), load(recon["state_artifact"]), load(previous["CP0_TUNE"]["state"]))
        require(row["initial_state_sha256"] == lock["initial_state_sha256"] == base.state_hash(initial) and row["probe"]["reference"] == recon["splits"]["tune"]["artifact"], "RECON_INITIAL_TENSOR_HASH_AND_CP0_REFERENCE")
        probe_arrays = load(row["probe"]["artifact"])
        totals["CP0_parity_components_checked"] += check_probe(row, probe_arrays, load(row["probe"]["reference"]), initial, raw["tune"]); totals["CP0_probe_cases_checked"] += 1
        states, saves = {}, {}; require([cp["update"] for cp in row["checkpoints"]] == list(base.CHECKPOINTS), "FIVE_ORIGINAL_CHECKPOINT_UPDATES")
        for cp in row["checkpoints"]:
            state, saved = load(cp["state"]), load(cp["artifact"]); base.exact(state["center"], initial["center"], "always_fixed_center")
            if cp["update"] == 0:
                for key in initial: base.exact(state[key], initial[key], "RECON_CP0_"+key)
                for key in COMPONENTS: base.exact(saved[key], probe_arrays["Torch_"+key], "CP0_checkpoint_matches_probe_"+key)
            score = base.replay(saved, state, raw["tune"], LAGS); base.unit_score(cp["tune"], score["normalized"])
            prior.close(cp["TUNE_latent_variance"], np.var(saved["latent_history"].astype(np.float64), axis=(0, 1)), "saved_TUNE_variance", atol=1e-10, rtol=0.)
            totals["training_score_fields_checked"] += 6 + base.check_score(cp, saved, raw["tune"]); states[cp["update"]], saves[cp["update"]] = state, saved; totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_predictions_checked"] += 1
        cp = min(row["checkpoints"], key=lambda p: (p["tune"]["mse"], p["update"])); update = cp["update"]
        require(row["selected_update"] == lock["selected_update"] == update and row["selected_state"] == lock["selected_state"] == cp["state"] and row["selected_state_sha256"] == base.state_hash(states[update]) and row["selected_tune"] == cp["tune"] and row["CP0_TUNE"] == row["checkpoints"][0], "TUNE_ONLY_EARLIEST_TIE_CP_SELECTED")
        totals["logged_updates_checked"] += base.check_logs(row, raw["fit"]); totals["recon_initialization_fits_checked"] += 1; totals["reused_FORECAST_initialization_conditions_checked"] += 1
        selected = {"tune": saves[update]}
        for label, split in (("FIT", "fit"), ("DEV", "val")):
            selected[split] = load(row[label]["artifact"]); base.replay(selected[split], states[update], raw[split], LAGS); totals["training_score_fields_checked"] += base.check_score(row[label], selected[split], raw[split]); totals["selected_FIT_DEV_predictions_checked"] += 1
            if label == "DEV": require(Path(row[label]["artifact"]["path"]).stat().st_mtime >= registry["sealed_unix"], "DEV_AFTER_BOTH_LOCKS")
        report["datasets"]["robin"]["conditions"][str(seed)] = {"selected_update": update, "selected_state": row["selected_state"], "selected_TUNE": row["selected_tune"], "CP0_TUNE": row["CP0_TUNE"]["tune"], "order_sha256": row["order_sha256"], "parameter_change": base.selected_geometry(initial, states[update], saves[update]["latent_history"])}
        report["new_output_space_diagnostics"][str(seed)] = {}
        for split in ("tune", "val"):
            try:
                cell = new_space_diagnostic(selected[split], states[update], raw[split], update, Path(diagnostic_dir)/("s"+str(seed)+"_"+split+".npz")); cell.update(selected_state=row["selected_state"], selected_forecast_source=cp["artifact"] if split == "tune" else row["DEV"]["artifact"], input_source=recon["splits"][split]["input_source"])
                report["new_output_space_diagnostics"][str(seed)][split] = cell; totals["new_space_diagnostic_cells_computed"] += 1; totals["new_space_LS_SVD_crosschecks_computed"] += int(cell["independent_reduced_LS_vs_SVD_verified"])
            except Exception as error: report["new_output_space_diagnostic_failures"].append({"seed": seed, "split": split, "type": type(error).__name__, "message": str(error)})
    for seed in SEEDS:
        try: audit_seed(seed)
        except Exception as error: report["partial_paths"].append({"seed": seed, "stage": "INDEPENDENT_TRAINING_VERIFICATION", "type": type(error).__name__, "message": str(error)})
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "ORIGINAL_PARENT_SOURCE_AND_ARRAYS_UNCHANGED")
    totals["protected_source_and_input_file_hashes_checked"] = len(guarded)
    if report["partial_paths"]: report.update(status="PARTIAL_AUDITED_NO_FULL_PASS", training_verification_status="PARTIAL")
    if result["status"] == "COMPLETE" and not report["partial_paths"]: require(totals["recon_initialization_fits_checked"] == result["recon_initialization_training_fits"] == 2 and totals["logged_updates_checked"] == result["training_updates_completed"] == 1200 and totals["checkpoint_states_checked"] == result["checkpoint_states_saved"] == 10 and totals["selected_FIT_DEV_predictions_checked"] == 4 and not result["failed_paths"], "FULL_TWO_FITS_1200_UPDATES_TEN_CP")
    report["new_space_diagnostic_status"] = "COMPLETE" if totals["new_space_diagnostic_cells_computed"] == 4 and not report["new_output_space_diagnostic_failures"] else "PARTIAL_COMPLETED"
    if report["new_space_diagnostic_status"] != "COMPLETE" and report["status"] == "PASS": report["status"] = "PARTIAL_AUDITED_DIAGNOSTICS_INCOMPLETE"
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_RECON_INITIALIZATION_AUDIT_OUTPUT")
    source = base.file_record(__file__); meter = Meter("audit_RECON_init_"+output.stem, gpu=False); failure = None
    try:
        meter.protected_check(); metadata(path, meter.check); initialize_numpy(); sys.path.insert(0, str(base.ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report = audit_result(path, meter, output.with_name(output.stem+"_new_space_arrays"))
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error: failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally: report["audit_source"] = source; report["resource"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
