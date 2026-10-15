"""Saved E-only training audit and separately new geometry; no F/optimizer replay."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import audit_linear_e_adaptation_v1 as base
import audit_recon_init_sensitivity_v2 as previous_audit

HERE, prior, require = base.HERE, base.prior, base.require
SEEDS, LAGS, COMPONENTS = previous_audit.SEEDS, previous_audit.LAGS, previous_audit.COMPONENTS
RESULT_SHA = "4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b"
CONTROLLER_SHA = "1c5f5a1e8f878b15705fc8f808af35ab8bc0ea9cdbc21375b7ad789a5dc45ff1"
JOINT_SHA = "f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8"
JOINT_AUDIT_SHA = "b1d28415ade18db8ddc0c987a14e879268413efdd5d1c62ad93e27b24becd52e"
HELPER_SHA = "55066f6dbc6ce8c6a175fe4ce5c8952fe8eabdde15492fd097b3a8d98c1ecdfd"
FLAGS = {"trainable_parameter_names": ["E"], "D_requires_grad": False, "D_grad_is_None": True, "center_requires_grad": False}
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy(); previous_audit.initialize_numpy(); np = sys.modules["numpy"]


def same_binding(a, b):
    require(Path(a["path"]).resolve() == Path(b["path"]).resolve() and a["sha256"] == b["sha256"], "SAME_PATH_SHA_OPTIONAL_BYTES_ALLOWED")


def fixed_state(state, initial):
    for key in ("D", "center"): base.exact(state[key], initial[key], "fixed_initial_"+key)


def flags(recorded): require(recorded == FLAGS, "RECORDED_E_ONLY_D_FROZEN_GRAD_NONE_FLAGS")


def check_probe(row, arrays, original, initial, tune):
    probe = row["probe"]; require(probe["status"] == "PASS" and probe["optimizer_updates"] == 0 and probe["initial_state_sha256"] == base.state_hash(initial), "ZERO_UPDATE_SAME_INITIAL_CP0_PROBE")
    require(probe["component_byte_exact"] == {key: True for key in COMPONENTS}, "ALL_FOUR_CP0_COMPONENTS_EXACT"); flags(probe["fixed_decoder_status"])
    base.exact(arrays["origins"], tune["origins"], "CP0_original_origins")
    for key in COMPONENTS:
        base.exact(arrays["E_ONLY_"+key], arrays["JOINT_CP0_"+key], "E_only_joint_CP0_"+key); base.exact(arrays["JOINT_CP0_"+key], original[key], "original_joint_CP0_"+key)
    base.replay({**tune, **{key: arrays["E_ONLY_"+key] for key in COMPONENTS}}, initial, tune, LAGS)
    return 2*len(COMPONENTS)


def check_logs(row, fit):
    count = base.check_logs(row, fit)
    logs = [json.loads(line) for line in Path(row["updates_log"]["path"]).read_text(encoding="utf-8").splitlines()]
    for item in logs:
        flags({key: item[key] for key in FLAGS}); require(item["D_gradient_norm_before_clip"] == 0. and item["gradient_clipping_parameter_scope"] == "E_ONLY", "LOGGED_D_NORM_ZERO_E_ONLY_CLIPPING")
    flags(row["fixed_decoder_final_status"])
    require(row["D_frozen_after_optimizer_step_checks"] == count == 600 and row["D_and_center_checkpoint_byte_checks"] == 5 and row["D_bytes_checked_at_checkpoints_and_final_only"] is True and row["independent_per_update_D_bytes_replayed"] is False, "RECORDED_STEP_FLAGS_VS_CP_BYTES_SCOPE")
    return count


def new_space(saved, state, raw, update, destination):
    require(prior.file_sha(previous_audit.__file__) == HELPER_SHA, "IMMUTABLE_ACCEPTED_SPACE_CALCULATOR_WRAPPER")
    cell = previous_audit.new_space_diagnostic(saved, state, raw, update, destination)
    cell.pop("selected_E_D_previously_supervised_by_FIT_future_targets")
    cell.update(selected_E_previously_supervised_by_FIT_future_targets=update > 0, selected_D_previously_supervised_by_FIT_future_targets=False, selected_D_held_fixed_at_past_RECON_initialization=True, E_training_procedure_used_FIT_future_targets=True, combined_E_D_supervision_flag_removed_as_ambiguous=True, audit_helper_source=base.file_record(previous_audit.__file__))
    return cell


def metadata(path, check):
    require(all(isinstance(v, str) and len(v) == 64 for v in (RESULT_SHA, CONTROLLER_SHA)), "ACTUAL_FIXED_D_E_ONLY_RESULT_NOT_YET_REGISTERED")
    binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_FIXED_D_E_ONLY_RESULT")
    result = prior.verified_json(binding, check); require(result["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and result["controller_source"]["sha256"] == CONTROLLER_SHA and result["joint_recon_diagnostic"]["sha256"] == JOINT_SHA and result["joint_recon_audit"]["sha256"] == JOINT_AUDIT_SHA, "ACCEPTED_JOINT_CONTROLS_AND_NEW_EXECUTED_SOURCE")
    return binding, result


def audit_result(path, meter, directory):
    check = meter.check; binding, result = metadata(path, check); joint = prior.verified_json(result["joint_recon_diagnostic"], check); joint_audit = prior.verified_json(result["joint_recon_audit"], check)
    require(joint["status"] == "COMPLETE" and joint_audit["status"] == joint_audit["training_verification_status"] == "PASS" and joint_audit["new_space_diagnostic_status"] == "COMPLETE", "ACCEPTED_RECON_JOINT_TRAINING_AND_GEOMETRY"); same_binding(joint_audit["diagnostic"], result["joint_recon_diagnostic"])
    require(joint_audit["audit_source"]["sha256"] == HELPER_SHA and result["F_before"] == result["F_after"] == joint["F_before"] == joint["F_after"] and result["joint_RECON_conditions_reused"] == 2 and result["new_joint_RECON_fits"] == result["probe_optimizer_updates"] == 0, "SAME_F_AND_TWO_REUSED_JOINT_CONTROLS")
    for key in ("new_backbone_training", "new_D_initialization_fit", "D_updated_during_training", "D_initialization_supervised_by_FIT_future", "protected_TEST_accessed", "independent_holdout", "strict_K_only", "whole_deployment_cost_measured", "new_output_space_diagnostics_computed_by_training_producer"): require(result[key] is False, "E_ONLY_SCOPE:"+key)
    require(result["E_training_supervised_by_FIT_future"] is result["F_input_gradient_retained"] is True and result["gradient_clipping_parameter_scope"] == "E_ONLY", "E_SUPERVISION_AND_ACTUAL_F_GRADIENT_PROCEDURE")
    same_binding(result["causal_lag_diagnostic"], joint["causal_lag_diagnostic"]); science = prior.verified_json(result["causal_lag_diagnostic"], check); recon = science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
    guarded = [binding, result["joint_recon_diagnostic"], result["joint_recon_audit"], result["causal_lag_diagnostic"], result["plan"], *result["source_records"]]
    for item in guarded:
        require(prior.file_sha(item["path"]) == item["sha256"], "SOURCE_PARENT_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "SOURCE_SNAPSHOT_SHA")
    registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    require(registry["status"] == "ALL_TWO_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == result["resolved_selection_locks"] == 2 and result["all_2_locks_before_DEV"] is True, "TWO_RESOLUTIONS_BEFORE_DEV")
    locks = {}
    for item in registry["conditions"]:
        require(item["dataset"] == "robin" and item["seed"] in SEEDS and item["seed"] not in locks, "UNIQUE_ROBIN_TWO_SEED_RESOLUTIONS")
        lock = prior.verified_json(item["selection_lock"], check); require(lock["DEV_opened_before_lock"] is False and lock["sealed_unix"] <= registry["sealed_unix"], "LOCK_BEFORE_REGISTRY"); locks[item["seed"]] = item, lock; guarded.append(item["selection_lock"])
    cache = {}
    def load(item):
        key = item["path"], item["sha256"]
        if key not in cache: cache[key] = prior.verified_arrays(item, check); guarded.append(item)
        return cache[key]
    raw = {split: load(recon["splits"][split]["input_source"]) for split in ("fit", "tune", "val")}
    for split, n in (("fit", 256), ("tune", 64), ("val", 30)): prior.check_raw_split(raw[split], channels=17, windows=n)
    totals = {key: 0 for key in ("E_only_fits_checked", "checkpoint_states_checked", "checkpoint_TUNE_predictions_checked", "selected_FIT_DEV_predictions_checked", "logged_updates_checked", "recorded_post_optimizer_D_flag_checks", "CP0_probe_cases_checked", "CP0_component_byte_comparisons_checked", "fixed_D_center_CP_byte_comparisons_checked", "reused_joint_conditions_checked", "training_score_fields_checked", "new_space_diagnostic_cells_computed", "new_space_LS_SVD_crosschecks_computed")}; totals["selection_locks_checked"] = 2
    keys = ("joint_recon_diagnostic", "joint_recon_audit", "full_gradient_parent", "full_gradient_parent_audit", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "joint_selection_registry", "plan")
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "training_verification_status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL", "new_space_diagnostic_status": "RUNNING", "diagnostic": binding, **{key: result[key] for key in keys}, "totals": totals, "datasets": {"robin": {"conditions": {}}}, "partial_paths": [], "new_output_space_diagnostics": {}, "new_output_space_diagnostic_failures": [], "protected_TEST_accessed": False, "independent_F_backward_optimizer_or_per_update_D_bytes_replayed": False, "new_whole_deployment_cost_measured": False, "scope": "Saved E/lag/D replay, actual scores/order/CP/locks, fixed initial D/center at CP and selected states, recorded per-step flags/D-norm-zero. No independent backward/optimizer or per-update D byte replay. Geometry is newly computed here via two frozen numeric paths, not a third external replay; E/D supervision histories are separate."}
    Path(directory).mkdir(parents=True, exist_ok=False)
    def audit_seed(seed):
        row = result["datasets"]["robin"]["conditions"][str(seed)]; item, lock = locks[seed]; same_binding(row["selection_lock"], item["selection_lock"])
        require(row["status"] == lock["status"] == item["status"], "UNCHANGED_LOCK_STATUS")
        if row["status"] != "COMPLETE": report["partial_paths"].append({"seed": seed, "status": row["status"]}); return
        original = joint["datasets"]["robin"]["conditions"][str(seed)]
        require(row["seed"] == seed and row["dataset"] == "robin" and row["lag"] == "L7" and row["learn_E"] is True and row["D_trainable"] is False and row["order_sha256"] == lock["order_sha256"] == original["order_sha256"] and row["checkpoints"] == lock["checkpoints"] and row["initial_sources"] == original["initial_sources"], "MATCHED_INITIALIZATION_ORDER_AND_E_ONLY_RECIPE")
        require(row["joint_recon_reference"] == {"seed": seed, "selected_update": original["selected_update"], "selected_state": original["selected_state"], "CP0_TUNE": original["CP0_TUNE"]}, "ORIGINAL_REUSED_JOINT_PAIR")
        initial = load(original["CP0_TUNE"]["state"]); require(row["initial_state_sha256"] == lock["initial_state_sha256"] == original["initial_state_sha256"] == base.state_hash(initial) and row["initial_D_value_sha256"] == hashlib.sha256(initial["D"].tobytes()).hexdigest(), "EXACT_JOINT_CP0_INITIAL_AND_D_HASH")
        same_binding(row["probe"]["reference"], original["CP0_TUNE"]["artifact"]); probe = load(row["probe"]["artifact"])
        totals["CP0_component_byte_comparisons_checked"] += check_probe(row, probe, load(original["CP0_TUNE"]["artifact"]), initial, raw["tune"]); totals["CP0_probe_cases_checked"] += 1
        states, saved = {}, {}; require([cp["update"] for cp in row["checkpoints"]] == list(base.CHECKPOINTS), "FIVE_CP_UPDATES")
        for cp in row["checkpoints"]:
            state, values = load(cp["state"]), load(cp["artifact"]); fixed_state(state, initial); flags(cp["fixed_decoder_status"]); require(cp["D_and_center_exact_initial_bytes"] is True, "RECORDED_CP_FIXED_VALUES")
            if cp["update"] == 0:
                for key in initial: base.exact(state[key], initial[key], "CP0_initial_"+key)
                for key in COMPONENTS: base.exact(values[key], probe["E_ONLY_"+key], "CP0_probe_checkpoint_"+key)
            score = base.replay(values, state, raw["tune"], LAGS); base.unit_score(cp["tune"], score["normalized"]); prior.close(cp["TUNE_latent_variance"], np.var(values["latent_history"].astype(np.float64), axis=(0, 1)), "TUNE_variance", atol=1e-10, rtol=0.)
            totals["training_score_fields_checked"] += 6+base.check_score(cp, values, raw["tune"]); totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_predictions_checked"] += 1; totals["fixed_D_center_CP_byte_comparisons_checked"] += 2; states[cp["update"]], saved[cp["update"]] = state, values
        selected = min(row["checkpoints"], key=lambda cp: (cp["tune"]["mse"], cp["update"])); update = selected["update"]
        require(row["selected_update"] == lock["selected_update"] == update and row["selected_state"] == lock["selected_state"] == selected["state"] and row["selected_state_sha256"] == base.state_hash(states[update]) and row["selected_tune"] == selected["tune"] and row["CP0_TUNE"] == row["checkpoints"][0], "TUNE_ONLY_EARLIEST_CP_SELECTED")
        count = check_logs(row, raw["fit"]); totals["logged_updates_checked"] += count; totals["recorded_post_optimizer_D_flag_checks"] += row["D_frozen_after_optimizer_step_checks"]; totals["E_only_fits_checked"] += 1; totals["reused_joint_conditions_checked"] += 1
        values = {"tune": saved[update]}
        for label, split in (("FIT", "fit"), ("DEV", "val")):
            values[split] = load(row[label]["artifact"]); fixed_state(states[update], initial); base.replay(values[split], states[update], raw[split], LAGS); totals["training_score_fields_checked"] += base.check_score(row[label], values[split], raw[split]); totals["selected_FIT_DEV_predictions_checked"] += 1
            if label == "DEV": require(Path(row[label]["artifact"]["path"]).stat().st_mtime >= registry["sealed_unix"], "DEV_AFTER_BOTH_LOCKS")
        report["datasets"]["robin"]["conditions"][str(seed)] = {"selected_update": update, "selected_state": row["selected_state"], "selected_TUNE": row["selected_tune"], "order_sha256": row["order_sha256"], "parameter_change": base.selected_geometry(initial, states[update], saved[update]["latent_history"])}; report["new_output_space_diagnostics"][str(seed)] = {}
        for split in ("tune", "val"):
            try:
                cell = new_space(values[split], states[update], raw[split], update, Path(directory)/("s"+str(seed)+"_"+split+".npz")); cell.update(selected_state=row["selected_state"], selected_forecast_source=selected["artifact"] if split == "tune" else row["DEV"]["artifact"], input_source=recon["splits"][split]["input_source"]); report["new_output_space_diagnostics"][str(seed)][split] = cell; totals["new_space_diagnostic_cells_computed"] += 1; totals["new_space_LS_SVD_crosschecks_computed"] += int(cell["independent_reduced_LS_vs_SVD_verified"])
            except Exception as error: report["new_output_space_diagnostic_failures"].append({"seed": seed, "split": split, "type": type(error).__name__, "message": str(error)})
    for seed in SEEDS:
        try: audit_seed(seed)
        except Exception as error: report["partial_paths"].append({"seed": seed, "stage": "TRAINING_AUDIT", "type": type(error).__name__, "message": str(error)})
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "SOURCE_PARENT_AND_ORIGINAL_ARRAYS_UNCHANGED")
    totals["protected_source_and_input_file_hashes_checked"] = len(guarded)
    if report["partial_paths"]: report.update(status="PARTIAL_AUDITED_NO_FULL_PASS", training_verification_status="PARTIAL")
    if result["status"] == "COMPLETE" and not report["partial_paths"]: require(totals["E_only_fits_checked"] == result["fixed_D_E_only_training_fits"] == 2 and totals["logged_updates_checked"] == result["training_updates_completed"] == 1200 and totals["checkpoint_states_checked"] == result["checkpoint_states_saved"] == 10 and totals["selected_FIT_DEV_predictions_checked"] == 4 and not result["failed_paths"], "TWO_E_ONLY_FITS_TEN_CP_1200_UPDATES")
    report["new_space_diagnostic_status"] = "COMPLETE" if totals["new_space_diagnostic_cells_computed"] == totals["new_space_LS_SVD_crosschecks_computed"] == 4 and not report["new_output_space_diagnostic_failures"] else "PARTIAL_COMPLETED"
    if report["status"] == "PASS" and report["new_space_diagnostic_status"] != "COMPLETE": report["status"] = "PARTIAL_AUDITED_DIAGNOSTICS_INCOMPLETE"
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_FIXED_D_E_ONLY_AUDIT_OUTPUT"); meter = Meter("audit_fixed_RECON_D_"+output.stem, gpu=False); source = base.file_record(__file__); failure = None
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
