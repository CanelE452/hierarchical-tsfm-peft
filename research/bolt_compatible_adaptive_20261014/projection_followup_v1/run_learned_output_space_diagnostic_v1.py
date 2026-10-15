"""Saved learned E/D output-space diagnosis; no F, training, or CP reselection."""
from pathlib import Path
import argparse
import json
import shutil
import sys

import run_f_input_gradient_cost_v1 as gates

HERE, ROOT, require, record, bound = gates.HERE, gates.ROOT, gates.require, gates.record, gates.bound
STOP_AUDIT_SHA = "64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f"
FULL_RESULT_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
FULL_AUDIT_SHA = "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
LAGS = {"L0": (0,), "L7": (0, 1, 2, 4, 8, 16, 32)}
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    import audit_causal_lag_objective_v2 as independent
    independent.initialize_numpy(); np = sys.modules["numpy"]


def load_gates(full_path, full_audit_path, stop_path, stop_audit_path):
    require(all(isinstance(x, str) and len(x) == 64 for x in (STOP_AUDIT_SHA, FULL_RESULT_SHA, FULL_AUDIT_SHA)), "ACTUAL_PARENT_PASS_NOT_REGISTERED")
    require(record(full_path)["sha256"] == FULL_RESULT_SHA and record(full_audit_path)["sha256"] == FULL_AUDIT_SHA and record(stop_audit_path)["sha256"] == STOP_AUDIT_SHA, "EXACT_LEARNED_PARENT_RESULTS_AND_PASSES")
    stopped, audit, full, full_audit, science, subset, original, refs = gates.load_gates(stop_path, stop_audit_path)
    bound(refs["full_gradient_parent"], record(full_path)); bound(refs["full_gradient_parent_audit"], record(full_audit_path))
    return stopped, audit, full, full_audit, science, subset, original, refs


def scored(prediction, raw, mask):
    import audit_projection_results_v1 as prior
    if not mask.any(): return {"status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
    return prior.scores(prediction, raw["target"], mask, raw["std"])


def diagnose_cell(saved, state, raw, lags):
    import audit_linear_e_adaptation_v1 as learned
    import audit_causal_lag_objective_v2 as independent
    import causal_lag_decoder_v1 as producer
    prior = independent.prior; c = raw["history"].shape[-1]; k = state["E"].shape[-1]
    require(set(state) == {"E", "D", "center"} and all(v.dtype == np.float32 and np.isfinite(v).all() for v in state.values()) and state["E"].shape == (c, k) and state["center"].shape == (c,) and state["D"].shape == (len(lags)*k, c), "EXACT_SELECTED_LEARNED_STATE")
    for name in ("target", "observed", "raw_last", "origins", "mean", "std"): learned.exact(saved[name], raw[name], "selected_saved_source_" + name)
    past, future = saved["latent_history"], saved["latent_future"]
    prior.close(past, (raw["history"]-state["center"])@state["E"], "independent_NumPy_encoding")
    features = independent.delta_features(past, future, lags)
    learned.exact(features, saved["lag_features"], "independent_selected_lag_features")
    replay = raw["raw_last"][:, None] + features@state["D"]
    prior.close(saved["prediction"], replay, "independent_NumPy_fixed_D_forecast")
    complete = raw["observed"].all(axis=2); mask = prior.oracle_mask(raw["observed"])
    ey_input = np.where(complete[..., None], raw["target"], raw["raw_last"][:, None]).astype(np.float32)
    ey = (ey_input-state["center"])@state["E"]
    flat = np.ascontiguousarray(np.broadcast_to(past[:, -1:], future.shape))
    futures = {"ACTUAL_F": future, "EY": ey, "FLAT": flat}
    arrays = {**{name: raw[name] for name in ("target", "observed", "raw_last", "origins", "mean", "std")}, "latent_history": past, "EY_input_C_FP32": ey_input, "complete_horizons": complete}
    row = {"status": "COMPLETE" if complete.any() else "NO_COMPLETE_C_TARGET_POSITIONS", "complete_target_positions": int(complete.sum()), "target_positions": int(complete.size), "joint_unknown_future_coordinates": future.shape[1]*k, "joint_output_coordinates_before_mask": future.shape[1]*c, "lag_slope_parameters_per_C": len(lags)*k, "lags": list(lags), "methods": {}, "RAW_bound_applied": False, "new_F_forward": False, "minimum_only_normalized_MSE": True, "all_real_latent_or_all_FP32_lower_bound_claimed": False, "old_initial_D_minimum_reused": False, "diagnostic_future_labels_used_for_new_fit_or_selection": False, "selected_E_D_previously_supervised_by_FIT_future_targets": True, "EY_encoding_backend": "NumPy FP32 GEMM with selected E/center; no Torch or F call; may differ from Torch encoding rounding", "EY_incomplete_C_horizons_filled_with_current_C": int((~complete).sum()), "EY_fill_can_affect_later_L7_complete_outputs": len(lags)>1, "minimum_normalized_MSE": None, "independent_reduced_LS_vs_SVD_verified": False}
    row["RAW_BOLT_scores_all_observed"] = scored(raw["raw_forecast"], raw, raw["observed"])
    row["RAW_BOLT_scores_complete_mask"] = scored(raw["raw_forecast"], raw, mask)
    persistence = np.ascontiguousarray(np.broadcast_to(raw["raw_last"][:, None], raw["target"].shape))
    row["current_C_persistence_scores_complete_mask"] = scored(persistence, raw, mask)
    arrays["RAW_BOLT_prediction_FP32"], arrays["current_C_persistence_FP32"] = raw["raw_forecast"], persistence
    minimum = independent.joint_minimum(raw, past, state["D"], lags)
    if complete.any():
        svd = producer.masked_joint_minimum(past, raw["raw_last"], state["D"], lags, raw["target"], raw["observed"])
        prior.close(svd["minimum_normalized_MSE"], minimum["minimum_normalized_MSE_real_joint_D32"], "new_selected_D_SVD_vs_reduced_LS_minimum", atol=1e-10, rtol=1e-10)
        prior.close(svd["prediction_FP64"], minimum["minimum_solution_prediction"], "new_selected_D_SVD_vs_reduced_LS_minimum_prediction", atol=1e-10, rtol=1e-10)
        row.update(minimum_normalized_MSE=minimum["minimum_normalized_MSE_real_joint_D32"], minimum_solution_scores_complete_C=minimum["minimum_solution_scores_complete_C"], independent_reduced_LS_vs_SVD_verified=True, joint_metadata=svd["metadata"], independent_LS_certificates=minimum["certificates"])
        arrays.update(joint_A_FP64=minimum["joint_map"], joint_known_offset_FP64=minimum["known_offset"], joint_minimum_prediction_FP64=minimum["minimum_solution_prediction"], joint_best_future_FP64=minimum["best_absolute_future_latent"])
    for name, f in futures.items():
        prediction = saved["prediction"] if name == "ACTUAL_F" else producer.predict(past, f, raw["raw_last"], state["D"], lags)
        independent_prediction = independent.anchored_forecast(past, f, raw["raw_last"], state["D"], lags)
        prior.close(prediction, independent_prediction, "independent_selected_method_FP32:" + name)
        real = independent.anchored_forecast(past, f, raw["raw_last"], state["D"], lags, real=True)
        prior.close(real, producer.real_trace(past, f, raw["raw_last"], state["D"], lags), "independent_selected_real_trace:" + name, atol=1e-10, rtol=1e-10)
        out = {"scores_all_observed": scored(prediction, raw, raw["observed"]), "scores_complete_mask": scored(prediction, raw, mask), "role": "ACTUAL_DEPLOYMENT_SAVED_FUTURE" if name == "ACTUAL_F" else "DIAGNOSTIC_FUTURE_Y_NOT_DEPLOYABLE" if name == "EY" else "PAST_ONLY_LATENT_PERSISTENCE"}
        arrays.update({name+"_future_FP32": f, name+"_prediction_FP32": prediction, name+"_real_trace_FP64": real})
        if complete.any():
            trace = independent.diagnostic_trace(prediction, f, raw, past, state["D"], lags, minimum)
            arrays[name+"_retained_trace_FP64"] = trace.pop("_retained_prediction_FP64")
            arrays[name+"_discarded_output_FP64"] = trace.pop("_discarded_output_FP64")
            out.update(trace)
        row["methods"][name] = out
    return row, arrays


def conditions(full, stopped):
    for data in gates.DATASETS:
        for lag in LAGS:
            for seed in gates.SEEDS:
                yield data, "FULL_"+lag+"_s"+str(seed), full["datasets"][data]["conditions"][data+"_"+lag+"_E_LEARN_s"+str(seed)], LAGS[lag], "FULL_REUSED"
        for seed in gates.SEEDS:
            yield data, "STOP_L0_s"+str(seed), stopped["datasets"][data]["conditions"][str(seed)], LAGS["L0"], "STOP_REUSED"


def execute(args):
    stopped, audit, full, full_audit, science, subset, original, refs = load_gates(args.full_result, args.full_audit, args.stop_result, args.stop_audit)
    output = gates.old.fresh_output(args.tag); output.mkdir(parents=True, exist_ok=False)
    import continuation_runtime_v1 as rt
    meter = rt.Meter("learned_output_space_"+args.tag, gpu=False); failure = None
    report = {"status": "RUNNING", **refs, "plan": record(HERE/"PLAN_LEARNED_OUTPUT_SPACE_DIAGNOSTIC_KO_v1.md"), "datasets": {}, "failed_paths": [], "source_records": [], "new_F_calls": 0, "new_model_objects_loaded": 0, "new_training_updates": 0, "new_E_D_fit": False, "CP_or_branch_reselection": False, "oracle_used_for_fit_or_selection": False, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "new_F_state_hash_computed": False, "whole_deployment_cost_measured": False, "numerical_crosscheck_is_not_independent_generalization": True}
    checked, raw_cache = {}, {}
    try:
        initialize_numpy(); import fixed_decoder_span_v1 as io
        import audit_linear_e_adaptation_v1 as learned
        import audit_causal_lag_objective_v2 as independent
        import causal_lag_decoder_v1 as producer
        from threadpoolctl import threadpool_limits
        from run_fit_target_comparison_v1 import prior_file_records
        sys.path.insert(0, str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard(); meter.protected_check()
        protected = prior_file_records(stopped, audit, full, full_audit, science, subset, original, *refs.values())
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__), Path(gates.__file__), Path(learned.__file__), Path(independent.__file__), Path(producer.__file__), HERE/"PLAN_LEARNED_OUTPUT_SPACE_DIAGNOSTIC_KO_v1.md", Path(io.__file__), Path(rt.__file__)):
            destination=folder/path.name; shutil.copyfile(path,destination); report["source_records"].append({**record(path), "snapshot_path": str(destination)})
        def load(item): return io.checked_payload({**item, "path": str(bound(item))}, meter, checked)
        with threadpool_limits(limits=2):
            for data, name, chosen, lags, origin in conditions(full, stopped):
                owner=report["datasets"].setdefault(data, {"conditions": {}})["conditions"]
                row=owner[name]={"status": "RUNNING", "origin": origin, "seed": chosen["seed"], "selected_update": chosen["selected_update"], "selected_state": chosen["selected_state"], "selection_lock": chosen["selection_lock"], "splits": {}}
                try:
                    bound(chosen["selection_lock"]); state=load(chosen["selected_state"])
                    require(learned.state_hash(state)==chosen["selected_state_sha256"], "EXACT_SELECTED_LEARNED_STATE_HASH")
                    cp=gates.old.selected_TUNE_checkpoint(chosen)
                    for split, source in (("tune",cp),(gates.DATASETS[data][2],chosen["DEV"])):
                        try:
                            raw_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]["input_source"]
                            if (data,split) not in raw_cache:
                                raw_cache[data,split]=load(raw_source); independent.prior.check_raw_split(raw_cache[data,split], channels=gates.DATASETS[data][0], windows=64 if split=="tune" else gates.DATASETS[data][3])
                            raw=raw_cache[data,split]; saved=load(source["artifact"])
                            cell,arrays=diagnose_cell(saved,state,raw,lags)
                            for key,mask in (("scores_all_observed",raw["observed"]),("scores_complete_mask",independent.prior.oracle_mask(raw["observed"]))):
                                if mask.any(): independent.prior.compare_scores(source[key],cell["methods"]["ACTUAL_F"][key],"accepted_selected_original_score:"+key)
                            cell.update(input_source=raw_source,selected_forecast_source=source["artifact"],artifact=io.save_npz(output/(data+"_"+name+"_"+split+"_DIAGNOSTIC.npz"),arrays)); row["splits"][split]=cell
                        except Exception as error:
                            fail={"dataset":data,"condition":name,"split":split,"type":type(error).__name__,"message":str(error)}; row["splits"][split]={"status":"FAILED_PATH_PRESERVED","failure":fail}; report["failed_paths"].append(fail)
                    row["status"]="COMPLETE" if all(x["status"]=="COMPLETE" for x in row["splits"].values()) else "PARTIAL_COMPLETED"
                except Exception as error:
                    fail={"dataset":data,"condition":name,"type":type(error).__name__,"message":str(error)}; row.update(status="FAILED_PATH_PRESERVED",failure=fail); report["failed_paths"].append(fail)
        cells=[cell for data in report["datasets"].values() for row in data["conditions"].values() for cell in row["splits"].values()]
        done=sum(c["status"]=="COMPLETE" for c in cells)
        for item in protected+list(checked.values())+report["source_records"]:
            bound(item)
            if "snapshot_path" in item: require(record(item["snapshot_path"])["sha256"]==item["sha256"],"SOURCE_SNAPSHOT_UNCHANGED")
        report.update(completed_diagnostic_cells=done,expected_diagnostic_cells=24,selected_states_diagnosed=sum(row["status"]=="COMPLETE" for data in report["datasets"].values() for row in data["conditions"].values()),saved_ACTUAL_F_predictions_diagnosed=done,EY_diagnostics=done,latent_persistence_diagnostics=done,independent_LS_SVD_crosschecks=done,input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if done==24 and not report["failed_paths"] else "PARTIAL_COMPLETED")
        meter.protected_check(); json.dumps(report,allow_nan=False)
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)}; report.update(status="FAILED_PRESERVED",failure=failure)
    finally:
        report["resource_terminal"]=meter.close(report["status"],failure); rt.write_new(output/"RESULT.json",report)
    print(json.dumps({"status":report["status"],"result":str(output/"RESULT.json"),"completed":report.get("completed_diagnostic_cells")}),flush=True)
    return 0 if report["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","full-result","full-audit","stop-result","stop-audit"): parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
