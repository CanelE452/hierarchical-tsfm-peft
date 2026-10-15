"""Selected TUNE-locked shrinkage whole Bolt costs; no new rho selection."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT")
SHRINK_SHA = "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532"
SHRINK_AUDIT_SHA = "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9"
PARENT_KEYS = ("local_decoder_diagnostic", "local_decoder_audit", "producer", "fit_target_audit",
    "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit")
AUDIT_COUNTS = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6,
    "positive_rho_backend_states_checked": 88, "window_fits_checked": 13504, "channel_slots_checked": 187664,
    "observed_channel_fits_checked": 187664, "positive_lambda_solver_calls_checked": 187664,
    "zero_trace_prior_fallbacks_checked": 0, "no_observation_prior_fallbacks_checked": 0,
    "native_seed_D_reuses_checked": 6752, "FIT_TUNE_positive_rho_prediction_arrays_checked": 120,
    "selected_DEV_prediction_arrays_checked": 12, "reused_control_method_pairs_checked": 36,
    "reused_control_prediction_arrays_byte_preserved": 72, "selection_locks_checked": 4,
    "score_field_checks": 6960, "protected_prior_file_hashes_checked": 916}
CANDIDATE_ORDER = ("STATIC", "LOCAL", "RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
RHO_MAP = dict(zip(CANDIDATE_ORDER[2:], (.001, .01, .1, 1., 10.)))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(condition, label):
    if not condition: raise ValueError(label)


def resolve(item):
    path = Path(item["path"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bind(item, path, label):
    require(resolve(item) == Path(path).resolve() and item["sha256"] == sha(path), "EXACT_SOURCE_BINDING:" + label)


def selected_locks(shrink, audit):
    locks = {}
    require(set(shrink["selection_locks"]) == set(DATASETS), "FOUR_REGISTERED_SELECTION_LOCKS_REQUIRED")
    for dataset, (_, _, dev) in DATASETS.items():
        require(set(shrink["selection_locks"][dataset]) == set(PROJECTIONS), "TWO_LOCKED_PROJECTION_RULES_PER_DATASET")
        locks[dataset] = {}
        for route in PROJECTIONS:
            item = shrink["selection_locks"][dataset][route]
            bind(item, resolve(item), "selected TUNE lock")
            lock = read(resolve(item))
            proof = audit["selection_locks"][dataset][route]
            require(proof["record"] == item and proof["selected_candidate"] == lock["selected_candidate"]
                    and proof["selected_rho"] == lock["selected_rho"]
                    and proof["recomputed_winner_equal_registered_winner"] is True, "INDEPENDENTLY_VERIFIED_SAME_TUNE_LOCK")
            label, rho = lock["selected_candidate"], lock["selected_rho"]
            require(label in RHO_MAP and type(rho) in (int, float) and math.isfinite(rho) and rho == RHO_MAP[label], "REGISTERED_POSITIVE_SELECTED_RHO_ONLY")
            require(lock["status"] == "LOCKED" and lock["dataset"] == dataset and lock["route"] == route
                    and lock["selection_predictor"] == "BOLT_NUMPY" and lock["selection_split"] == "TUNE"
                    and lock["selection_metric"] == "normalized_macro_channel_MSE" and lock["selection_mask"] == "all_true_observed"
                    and lock["MAE_used_for_selection"] is False and lock["native_specific_selection"] is False
                    and lock["development_or_oracle_scores_used_for_selection"] is False
                    and lock["native_uses_same_Bolt_selected_candidate"] is True and tuple(lock["candidate_tie_order"]) == CANDIDATE_ORDER,
                    "LOCKED_NORMAL_TUNE_BOLT_ONLY_SELECTION_SCOPE")
            scores = lock["TUNE_Bolt_normalized_MSE_by_candidate"]
            require(set(scores) == set(CANDIDATE_ORDER) and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in scores.values()), "ALL_SEVEN_LOCKED_TUNE_SCORES_REQUIRED")
            winner = min(CANDIDATE_ORDER, key=lambda name: (scores[name], CANDIDATE_ORDER.index(name)))
            require(winner == label and lock["selected_TUNE_Bolt_normalized_MSE"] == scores[label], "VERIFY_EXISTING_TUNE_WINNER_WITHOUT_NEW_SELECTION")
            cell = shrink["datasets"][dataset]["splits"][dev]["routes"][route]
            require(cell["status"] == "COMPLETE" and cell["selection_lock"] == item and cell["selected_candidate"] == label
                    and cell["unselected_positive_rho_DEV_candidates_computed"] is False
                    and set(cell["positive_rho_candidates"]) == {label}, "ONLY_ORIGINAL_LOCK_SELECTED_DEV_CANDIDATE")
            branch = cell["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"]
            require(branch["status"] == "COMPLETE" and branch["state_metadata"]["rho"] == rho
                    and cell["selected_methods"]["BOLT"]["artifact"] == branch["artifact"]
                    and cell["selected_methods"]["BOLT"]["prediction_member"] == "BOLT_prediction", "SAME_SELECTED_NUMPY_STATE_AND_ACTUAL_FORECAST_ARTIFACT")
            locks[dataset][route] = {"record": item, "selected_candidate": label, "rho": float(rho)}
    return locks


def load_gates(shrink_path, audit_path):
    """Exact sealed JSON/source/selection locks, before scientific imports or F."""
    shrink, audit = read(shrink_path), read(audit_path)
    require(shrink.get("status") == "COMPLETE" and audit.get("status") == "PASS", "COMPLETE_SHRINKAGE_AND_EXACT_INDEPENDENT_PASS_BEFORE_SCIENCE")
    require(sha(shrink_path) == SHRINK_SHA and sha(audit_path) == SHRINK_AUDIT_SHA, "EXACT_ACCEPTED_SHRINKAGE_RESULT_AND_AUDIT_SHA")
    bind(audit["diagnostic"], shrink_path, "shrinkage independent audit")
    for key in PARENT_KEYS:
        require(audit[key] == shrink[key], "SAME_SHRINKAGE_PREDECESSOR_CHAIN:" + key)
        bind(shrink[key], resolve(shrink[key]), "accepted predecessor " + key)
    require(all(audit["totals"].get(key) == value for key, value in AUDIT_COUNTS.items()), "FULL_REGISTERED_SHRINKAGE_AUDIT_COUNTS")
    expected = {"completed_positive_rho_backend_batches": 88, "completed_window_matrix_fits": 13504,
        "completed_channel_slots": 187664, "completed_observed_channel_fits": 187664, "completed_ridge_solver_calls": 187664,
        "completed_zero_trace_prior_slope_fallbacks": 0, "completed_no_observation_fallbacks": 0,
        "native_seed_D_reuses": 6752, "completed_FIT_TUNE_positive_rho_method_predictions": 120,
        "completed_selected_DEV_method_predictions": 12, "completed_reused_control_method_pairs": 36,
        "completed_RAW_control_cells": 6, "completed_selection_locks": 4, "new_E_F_native_head_forward_runs": 0}
    require(all(shrink.get(key) == value for key, value in expected.items()) and not shrink["failed_paths"], "COMPLETE_SEALED_SHRINKAGE_COUNTS")
    for key in ("new_E_F_or_native_head_training", "new_backbone_training", "new_GPU_execution", "protected_TEST_accessed",
                "future_labels_used_for_decoder_fit", "native_specific_selection", "development_positive_rho_sweep_executed",
                "whole_selected_deployment_cost_measured", "same_decoder_side_information_as_static", "strict_K_only_compression",
                "old_static_or_global_rankK_floor_reused", "independent_holdout"):
        require(shrink[key] is False, "SEALED_SHRINKAGE_SCOPE:" + key)
    for key in ("TUNE_future_labels_used_for_normal_hyperparameter_selection", "native_uses_same_Bolt_selected_candidate",
                "all_selection_lock_files_saved_before_any_DEV_candidate_compute", "fit_uses_TUNE_DEV_known_past_at_origin",
                "original_source_cache_model_and_selection_SHA_unchanged"):
        require(shrink[key] is True, "EXPLICIT_EARLIER_TUNE_SELECTION_AND_ORIGINAL_PRESERVATION:" + key)
    require(audit["protected_TEST_accessed"] is False and audit["future_labels_used_for_decoder_fit"] is False, "NO_TEST_NO_FUTURE_DECODER_FIT_AUDIT")
    code = audit["audit_source"]
    require(Path(code["path"]).name == "audit_local_decoder_shrinkage_v1.py", "CANONICAL_SHRINKAGE_AUDIT_SOURCE")
    bind(code, resolve(code), "shrinkage audit code")
    for item in shrink["source_records"]:
        bind(item, resolve(item), "sealed executed shrinkage source")
    if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
    from run_local_decoder_shrinkage_v1 import load_gates as old_gates
    _, producer, bindings = old_gates(resolve(shrink["local_decoder_diagnostic"]), resolve(shrink["local_decoder_audit"]))
    require(all(shrink[key] == bindings[key] for key in PARENT_KEYS), "SAME_ACCEPTED_LOCAL_AND_FIT_SOURCE_CHAIN")
    require(set(shrink["datasets"]) == set(DATASETS), "REGISTERED_TWO_SHRINKAGE_DATASETS")
    for dataset, (c, k, dev) in DATASETS.items():
        row, source = shrink["datasets"][dataset], producer["datasets"][dataset]
        require(row["channels"] == c and row["K"] == source["K"] == k and row["development_split"] == dev, "FIXED_SHRINKAGE_C_K_DEVELOPMENT")
        for split in ("fit", "tune", dev):
            require(row["splits"][split]["status"] == "COMPLETE" and row["splits"][split]["input_source"] == source["inputs"][split], "SAME_SEALED_SHRINKAGE_INPUTS")
    locks = selected_locks(shrink, audit)
    return producer, shrink, audit, locks


def static_predictor(state, frozen_F):
    def predict(history, history_observed):
        z = state.encode(history)
        return state.forecast(history[:, -1], frozen_F(z), z[:, -1])
    return predict


def shrunk_predictor(state, frozen_F, rho):
    import numpy as np
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    prior32 = np.ascontiguousarray(state.decoder.astype(np.float32))
    require(rho in RHO_MAP.values(), "ONLY_PREVIOUSLY_LOCKED_POSITIVE_RHO")
    def predict(history, history_observed):
        z = state.encode(history)
        fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)
        return fitted.forecast(history[:, -1], frozen_F(z), z[:, -1])
    return predict


def physical_shrunk_precheck(physical, observed, mean, std, values, state, cached, arrays, metadata, rho):
    import numpy as np
    import whole_selected_shrinkage_cost_v1 as wc
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    history = wc.preprocess(physical[:8], observed[:8], mean, std)
    z = state.encode(history)
    prior32 = np.ascontiguousarray(state.decoder.astype(np.float32))
    fitted = fit_shrunk_local_decoder(history, observed[:8], z, prior32, rho)
    require(wc.array_record(prior32) == metadata["fit_input_bindings"]["prior_decoder"], "EXACT_STORED_STATIC_PRIOR_DEPLOYMENT_CAST")
    require(metadata["rho"] == rho and metadata["lambda_rule"] == fitted.diagnostics["lambda_rule"]
            and metadata["slope_singular_cutoff_used"] is False, "EXACT_LOCKED_RHO_TRACE_LAMBDA_RULE_NO_CUTOFF")
    checks = {"normalized_history": wc.parity(history, values["history"][:8]),
        "normalized_current_C": wc.parity(history[:, -1], values["raw_last"][:8]),
        "encoded_past": wc.parity(z, cached["latent_history"][:8]),
        "decoder_FP32": wc.parity(fitted.decoder32, arrays["decoder_deployed_FP32"][:8]),
        "decoder_bias_FP32": wc.parity(fitted.bias32, arrays["decoder_bias_FP32"][:8])}
    for key in ("actual_lambda", "local_covariance_trace", "ridge_matrix_condition", "real_stationarity_residual_norm"):
        checks[key] = wc.parity(np.asarray(fitted.diagnostics[key]), np.asarray(metadata[key])[:8])
    decisions = {}
    for key in ("true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept", "past_metric_valid", "ridge_matrix_condition_valid"):
        a, b = np.asarray(fitted.diagnostics[key]), np.asarray(metadata[key])[:8]
        decisions[key] = bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
    decisions["positive_lambda_solver_decision"] = bool(np.array_equal(np.asarray(fitted.diagnostics["actual_lambda"]) > 0, np.asarray(metadata["actual_lambda"])[:8] > 0))
    mask_equal = bool(observed[:8].shape == values["history_observed"][:8].shape and observed[:8].dtype == values["history_observed"][:8].dtype
        and observed[:8].tobytes() == values["history_observed"][:8].tobytes())
    result = {"status": "PASS" if all(v["pass"] for v in checks.values()) and all(decisions.values()) and mask_equal else "FAILED_PRECHECK_PRESERVED",
        "scope": "Actual physical first8 -> fill/FIT normalize -> fixed NumPy E -> selected rho past-only ridge and full implemented diagnostics; no F yet",
        "checks": checks, "exact_decisions": decisions, "history_observed_byte_exact": mask_equal,
        "rho": rho, "lambda_rule": fitted.diagnostics["lambda_rule"], "origins": values["origins"][:8].tolist(),
        "decision_acceptance_by_output_tolerance_only": False, "coefficient_tolerance_loosened": False,
        "fitted_shrunk_metadata": fitted.metadata(), "deployed_static_prior": wc.array_record(prior32)}
    payload = {**fitted.to_npz(), "preprocessed_history": history, "latent_history": z, "history_observed": observed[:8].copy(),
        "origins": values["origins"][:8].copy(), "prior_decoder_FP32": prior32}
    return result, payload


def execute(tag, shrink_path, audit_path):
    producer, shrink, audit, locks = load_gates(shrink_path, audit_path)
    require(Path(tag).name == tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_RUN_TAG_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_selected_shrinkage_cost_v1 as wc
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import matched_readout_contract_v1 as contract
    import cost_readout_v1 as old_cost
    from projection_math_v1 import ProjectionState
    from run_fit_target_comparison_v1 import prior_file_records
    require(tuple(wc.ROUTES) == ROUTES and len(wc.balanced_schedule()) == 6, "FIVE_F_ONLY_COST_ROUTES_REQUIRED")
    output = HERE / "runs" / tag
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)
    meter = rt.Meter(tag, gpu=True)
    inputs.install_protected_test_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", "producer": shrink["producer"], "fit_target_audit": shrink["fit_target_audit"],
        "shrinkage_diagnostic": record(shrink_path), "shrinkage_audit": record(audit_path),
        "selection_locks": shrink["selection_locks"], "selected_rule_metadata": locks, "cost": {}, "datasets": {},
        "source_snapshots": [], "source_records": [], "failed_paths": [], "routes": list(ROUTES),
        "protected_TEST_accessed": False, "independent_holdout": False, "new_training_updates": 0,
        "projection_fits": 0, "inference_time_local_decoder_fits": True, "future_labels_used_for_shrunk_fit": False,
        "new_backbone_training": False, "new_native_head_forward": False, "native_deployment_cost_measured": False,
        "new_rho_or_method_selection_in_cost": False, "uses_preexisting_TUNE_Bolt_selection": True,
        "TUNE_future_labels_used_for_earlier_rho_selection": True, "past_C_condition_metadata_added": True,
        "same_decoder_side_information_as_static": False, "strict_K_only_compression": False,
        "static_D_or_global_rankK_oracle_is_local_lower_bound": False,
        "decoder_input_interface": "K streams + current C + origin-specific KxC coefficients from known C past512 and observed mask",
        "local_ridge_and_implemented_diagnostics_inside_timer": True, "fit_input_binding_hashes_inside_timer": True,
        "source_file_state_metadata_hashes_outside_timed_callback": True, "local_coefficients_cached_inside_timed_pipeline": False,
        "native_heads_loaded_or_timed": False, "hard_elapsed_or_cumulative_resource_caps": None,
        "authorization": record(rt.AUTH), "continuation_manifest": record(rt.MANIFEST),
        "balanced_scope": "First five circular rotations perfectly balanced; sixth seeded extra is not full-six Latin balance",
        "timing_helper_copy": {"original": record(HERE / "whole_local_past_decoder_cost_v1.py"),
            "changes": "Five F-only routes including two selected-rho callbacks, original observed mask; frozen timer bounds/parity retained"}}
    guarded, failure, started = [], None, time.perf_counter()
    def checked_load(item):
        arrays, checked = probes.load_checked_npz(item)
        result["source_records"].append(checked)
        guarded.append(dict(item))
        return {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    def exact(a, b, label):
        require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), "EXACT_CONDITION:" + label)
    try:
        meter.protected_check()
        guarded = prior_file_records(producer, shrink, audit, record(shrink_path), record(audit_path))
        for parent in (producer,):
            for item in parent.get("source_snapshots", []):
                require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "ACCEPTED_SOURCE_SNAPSHOT_CHANGED")
                guarded.append({"path": item["snapshot_path"], "sha256": item["sha256"]})
        source_folder = output / "sources"
        source_folder.mkdir()
        for path in (Path(__file__), HERE / "whole_selected_shrinkage_cost_v1.py", HERE / "local_decoder_shrinkage_v1.py", HERE / "local_past_decoder_v1.py",
                     HERE / "projection_math_v1.py", HERE / "run_local_decoder_shrinkage_v1.py",
                     HERE / "PLAN_SELECTED_SHRINKAGE_COST_KO_v1.md", HERE / "continuation_runtime_v1.py",
                     BASE / "cost_readout_v1.py", OLD / "run_condition_probes.py", OLD / "prepare_inputs.py", OLD / "matched_readout_contract_v1.py"):
            target = source_folder / path.name
            with path.open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(producer[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt()
        result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == producer["F_before"] == producer["F_after"], "EXACT_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        bolt.model.eval().requires_grad_(False)
        def frozen_F(history):
            with torch.no_grad():
                return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)
        def raw_predict(history, history_observed):
            return frozen_F(history)
        with threadpool_limits(limits=2):
            for dataset, (channels, k, dev) in DATASETS.items():
                result["datasets"][dataset] = {"status": "RUNNING", "K": k, "channels": channels, "development_split": dev}
                try:
                    meter.check({"stage": "selected_shrinkage_whole_cost", "dataset": dataset})
                    source, diagnostic = producer["datasets"][dataset], shrink["datasets"][dataset]["splits"][dev]
                    values, fit = checked_load(source["inputs"][dev]), checked_load(source["inputs"]["fit"])
                    exact(values["mean"], fit["mean"], "FIT_mean")
                    exact(values["std"], fit["std"], "FIT_std")
                    data_contract = contract.validate_raw_split(dataset, dev, values, fit["mean"], fit["std"])
                    physical = checked_load(source["physical_bridge"])
                    exact(physical["origins"], values["origins"], "physical_origins")
                    exact(physical["history_observed"], values["history_observed"], "physical_observed_mask")
                    old_cost.validate_physical_inputs(physical["history_physical"], physical["history_observed"],
                        values["history"], fit["mean"], fit["std"], values["origins"])
                    predictors, predictions, metadata, prechecks = {"RAW_BOLT": raw_predict}, {"RAW_BOLT": values["raw_forecast"]}, {}, {}
                    metadata["RAW_BOLT"] = {"source_inputs": source["inputs"][dev], "reference_prediction_array": "raw_forecast",
                        "shared_resident_frozen_F": True, "local_fit": False, "native_head_loaded_or_timed": False}
                    for route in PROJECTIONS:
                        condition = source["projections"][route]
                        registered = locks[dataset][route]
                        label, rho = registered["selected_candidate"], registered["rho"]
                        selected_cell = diagnostic["routes"][route]
                        backend = selected_cell["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"]
                        control_backend = selected_cell["control_backends"]["BOLT_NUMPY"]
                        state = ProjectionState.from_npz(checked_load(condition["state_artifact"]))
                        cached = checked_load(condition["bolt"]["artifacts"][dev])
                        local_arrays = checked_load(backend["state_artifact"])
                        references = checked_load(backend["artifact"])
                        control_references = checked_load(control_backend["artifact"])
                        exact(cached["origins"], values["origins"], "cached_BOLT_origins")
                        exact(references["origins"], values["origins"], "LocalD_origins")
                        exact(references["raw_last"], values["raw_last"], "LocalD_currentC")
                        exact(references["latent_history"], cached["latent_history"], "LocalD_E_past")
                        exact(references["BOLT_cached_future"], cached["latent_future"], "LocalD_F_future")
                        exact(control_references["origins"], values["origins"], "static_control_origins")
                        exact(control_references["BOLT_static_prediction"], cached["prediction"], "static_original_forecast")
                        check, arrays = physical_shrunk_precheck(physical["history_physical"], physical["history_observed"],
                            fit["mean"], fit["std"], values, state, cached, local_arrays, backend["state_metadata"], rho)
                        check["artifact"] = wc.save_outputs(output / (dataset + "_" + route + "_PHYSICAL_SHRUNK_PRECHECK.npz"), arrays)
                        check.update(source_shrunk_state=backend["state_artifact"], source_shrunk_forecast=backend["artifact"],
                                     selection_lock=registered["record"], selected_candidate=label,
                                     source_state=condition["state_artifact"], source_BOLT_cache=condition["bolt"]["artifacts"][dev])
                        prechecks[route] = check
                        result["datasets"][dataset]["physical_shrunk_prechecks"] = prechecks
                        wc.write_json_once(output / (dataset + "_" + route + "_PHYSICAL_SHRUNK_PRECHECK.json"), check)
                        require(check["status"] == "PASS", "PHYSICAL_SHRUNK_COEFFICIENT_LAMBDA_AND_DECISIONS_PRECHECK_FAILED:" + route)
                        static_route = route + "_BOLT"
                        local_route = route.replace("_PINV", "_SHRUNK") + "_BOLT"
                        predictors[static_route], predictors[local_route] = static_predictor(state, frozen_F), shrunk_predictor(state, frozen_F, rho)
                        predictions[static_route], predictions[local_route] = control_references["BOLT_static_prediction"], references["BOLT_prediction"]
                        for name, is_local in ((static_route, False), (local_route, True)):
                            metadata[name] = {"source_state": condition["state_artifact"], "selected_CPU_reference": backend["artifact"] if is_local else control_backend["artifact"],
                                "reference_prediction_array": "BOLT_prediction" if is_local else "BOLT_static_prediction",
                                "local_fit": is_local, "local_fit_uses_current_known_past_and_true_observed_mask": is_local,
                                "local_ridge_and_implemented_diagnostics_inside_timer": is_local,
                                "fit_input_binding_hashes_inside_timer": is_local, "cached_local_D_used_for_timing": False,
                                "rho": rho if is_local else None, "selection_lock": registered["record"],
                                "selected_candidate": label, "new_rho_selection_in_cost": False, "shared_resident_frozen_F": True,
                                "new_E_F_or_head_training": False, "native_head_loaded_or_timed": False,
                                "past_C_condition_metadata_added": is_local, "same_decoder_side_information_as_static": not is_local,
                                "strict_K_only_compression": False}
                    result["datasets"][dataset].update(inputs={"fit": source["inputs"]["fit"], dev: source["inputs"][dev]},
                        physical_bridge=source["physical_bridge"], selected_CPU_cell_source=diagnostic["input_source"],
                        data_contract=data_contract, selected_metadata=metadata)
                    memory = lambda: {"resident_rss_bytes": psutil.Process().memory_info().rss,
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                    result["cost"][dataset] = wc.measure_cost(dataset, physical["history_physical"], physical["history_observed"],
                        fit["mean"], fit["std"], values["origins"], predictors, predictions, metadata,
                        output / ("cost_" + dataset), meter.check, torch.cuda.synchronize, memory)
                    result["datasets"][dataset]["status"] = "COMPLETE"
                    print(json.dumps({"event": "SELECTED_SHRINKAGE_WHOLE_COST_COMPLETE", "dataset": dataset, "cells": 10}), flush=True)
                except Exception as error:
                    failed = {"dataset": dataset, "status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
                    cost_receipt = output / ("cost_" + dataset) / "RESULT.json"
                    if cost_receipt.is_file():
                        result["cost"][dataset] = read(cost_receipt)
                        failed["preserved_partial_cost_receipt"] = record(cost_receipt)
                    else:
                        failed["whole_cost_status"] = "NOT_MEASURED: failed before five-route timing cohort"
                    result["datasets"][dataset].update(failed)
                    result["failed_paths"].append(failed)
                    print(json.dumps({"event": "SELECTED_SHRINKAGE_COST_PATH_FAILURE_PRESERVED", **failed}), flush=True)
        result["F_after"] = probes.model_hashes(bolt)
        require(result["F_before"] == result["F_after"], "FROZEN_BOLT_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in result["source_snapshots"]:
            require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_CHANGED")
        for item in guarded:
            probes.verified_file(item)
        load_gates(shrink_path, audit_path)
        meter.protected_check()
        rows = [row for value in result["cost"].values() for row in value["rows"]]
        samples = sum(value["recorded_timer_samples"] for value in result["cost"].values())
        complete = len(rows) == 20 and samples == 120 and not result["failed_paths"]
        result.update(status="COMPLETE" if complete else "PARTIAL_COMPLETED", expected_cost_cells=20,
            completed_cost_cells=len(rows), recorded_timer_samples=samples,
            completed_cell_timer_samples=sum(row["repeats"] for row in rows),
            sample_count_scope="All saved measured samples including partial failed cohorts; completed-cell samples recorded separately",
            source_cache_selection_SHA_unchanged=True, original_selected_forecasts_and_locks_unchanged=True,
            protected_prior_records=guarded)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        result.update(status="FAILED_PRESERVED", failure=failure)
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "output": str(output / "RESULT.json")}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--shrinkage-diagnostic", required=True, type=Path)
    parser.add_argument("--shrinkage-audit", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(args.tag, args.shrinkage_diagnostic, args.shrinkage_audit))
