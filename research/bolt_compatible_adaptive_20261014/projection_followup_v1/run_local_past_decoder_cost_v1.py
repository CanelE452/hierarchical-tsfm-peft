"""Whole frozen-Bolt deployment cost including actual origin-specific past LS."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_LOCAL_BOLT", "FIT_TARGET_LOCAL_BOLT")
LOCAL_AUDIT_COUNTS = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6,
    "backend_states_checked": 24, "completed_backend_batches_checked": 24, "method_pairs_checked": 36,
    "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304,
    "native_seed_D_reuses_checked": 1632, "static_prediction_arrays_byte_preserved": 36,
    "local_prediction_arrays_checked": 36, "static_einsum_control_arrays_checked": 36,
    "cached_future_arrays_byte_preserved": 36, "score_field_checks": 2592,
    "new_per_origin_output_diagnostic_backend_cells_checked": 24,
    "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0}
PARENT_KEYS = ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit",
               "past_constant_whole_cost", "past_constant_whole_cost_audit")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved(reference):
    path = Path(reference["path"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bind(reference, path, label):
    require(resolved(reference) == Path(path).resolve() and reference["sha256"] == sha(path), "EXACT_SOURCE_BINDING:" + label)


def load_gates(local_path, local_audit_path):
    """JSON and file hashes only; all accepted passes precede scientific imports."""
    local, audit = read(local_path), read(local_audit_path)
    require(local.get("status") == "COMPLETE" and audit.get("status") == "PASS", "COMPLETE_LOCAL_PRODUCER_AND_INDEPENDENT_PASS_REQUIRED")
    bind(audit["diagnostic"], local_path, "LocalD audit")
    for key in PARENT_KEYS:
        require(audit.get(key) == local.get(key), "EXACT_LOCAL_PREDECESSOR_RECORD:" + key)
        bind(local[key], resolved(local[key]), "LocalD predecessor " + key)
    code = audit["audit_source"]
    require(Path(code["path"]).name == "audit_local_past_decoder_probe_v1.py", "REGISTERED_INDEPENDENT_LOCAL_AUDITOR_REQUIRED")
    bind(code, resolved(code), "LocalD independent audit code")
    require(all(audit.get("totals", {}).get(key) == value for key, value in LOCAL_AUDIT_COUNTS.items()), "FULL_LOCAL_PAST_DECODER_INDEPENDENT_AUDIT_REQUIRED")
    total = audit["totals"]
    solvers = total.get("observed_channel_solver_calls_checked")
    fallbacks = total.get("no_observation_fallbacks_checked")
    require(isinstance(solvers, int) and isinstance(fallbacks, int) and solvers >= 0 and fallbacks >= 0
            and solvers + fallbacks == total["channel_slots_checked"], "OBSERVED_SOLVERS_AND_NOOBS_FALLBACKS_SEPARATE_FROM_CHANNEL_SLOTS")
    require(local.get("completed_method_pairs") == 36 and local.get("completed_backend_batches") == 24
            and local.get("completed_window_matrix_fits") == 3264 and local.get("native_seed_D_reuses") == 1632
            and local.get("completed_channel_solver_calls") == solvers
            and local.get("completed_no_observation_fallbacks") == fallbacks
            and not local.get("failed_paths"), "FULL_COMPLETE_LOCAL_PRODUCER_COUNTS_REQUIRED")
    require(local.get("protected_TEST_accessed") is False and audit.get("protected_TEST_accessed") is False
            and local.get("new_E_F_native_head_forward_runs") == 0
            and local.get("new_E_F_or_native_head_training") is False
            and local.get("new_backbone_training") is False
            and local.get("future_labels_used_for_local_fit") is False
            and local.get("method_checkpoint_threshold_selection_used") is False
            and local.get("original_source_cache_selection_SHA_unchanged") is True,
            "UNCHANGED_PAST_ONLY_NO_TEST_NO_SELECTION_LOCAL_SCOPE_REQUIRED")
    require(local.get("same_decoder_side_information_as_static") is False
            and local.get("past_C_condition_metadata_added") is True
            and local.get("strict_K_only_compression") is False
            and local.get("static_D_or_global_rankK_oracle_reused_as_local_lower_bound") is False,
            "EXPLICIT_NEW_PAST_C_DECODER_SIDE_INFORMATION_REQUIRED")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from run_local_past_decoder_probe_v1 import load_gates as predecessor_gates
    producer, _, _, bindings = predecessor_gates(*(resolved(local[key]) for key in PARENT_KEYS))
    require(all(bindings[key] == local[key] for key in PARENT_KEYS), "SAME_ACCEPTED_PREDECESSOR_CHAIN")
    require(set(local["datasets"]) == set(DATASETS), "REGISTERED_TWO_LOCAL_DATASETS")
    for dataset, (c, k, dev) in DATASETS.items():
        row, source = local["datasets"][dataset], producer["datasets"][dataset]
        require(row["channels"] == c and row["K"] == source["K"] == k, "REGISTERED_LOCAL_C_K")
        for split in ("fit", "tune", dev):
            cell = row["splits"][split]
            require(cell["status"] == "COMPLETE" and cell["input_source"] == source["inputs"][split], "SAME_LOCAL_KNOWN_PAST_ORIGINS:" + split)
            for route in PROJECTIONS:
                backend = cell["routes"][route]["backends"]["BOLT_NUMPY"]
                require(backend["status"] == "COMPLETE"
                        and backend["original_state_source"] == source["projections"][route]["state_artifact"]
                        and backend["cached_method_sources"]["BOLT"] == source["projections"][route]["bolt"]["artifacts"][split],
                        "SAME_FIXED_E_BOLT_NUMPY_CACHE_AND_LOCAL_DECODER")
    return producer, local, audit


def static_predictor(state, frozen_F):
    def predict(history, history_observed):
        z = state.encode(history)
        return state.forecast(history[:, -1], frozen_F(z), z[:, -1])
    return predict


def local_predictor(state, frozen_F):
    from local_past_decoder_v1 import fit_local_past_decoder
    def predict(history, history_observed):
        z = state.encode(history)
        local = fit_local_past_decoder(history, history_observed, z)
        return local.forecast(history[:, -1], frozen_F(z), z[:, -1])
    return predict


def physical_local_precheck(physical, observed, mean, std, values, state, cached, local_arrays, local_metadata):
    """No F call; require physical first8 parity including rank/cutoff decisions."""
    import numpy as np
    import whole_local_past_decoder_cost_v1 as wc
    from local_past_decoder_v1 import fit_local_past_decoder, SINGULAR_RELATIVE_CUT
    n = 8
    history = wc.preprocess(physical[:n], observed[:n], mean, std)
    z = state.encode(history)
    fitted = fit_local_past_decoder(history, observed[:n], z)
    require(local_metadata["singular_relative_cut"] == SINGULAR_RELATIVE_CUT == 1e-4, "EXACT_PREDECLARED_LOCAL_SINGULAR_CUTOFF")
    checks = {"normalized_history": wc.parity(history, values["history"][:n]),
        "normalized_current_C": wc.parity(history[:, -1], values["raw_last"][:n]),
        "encoded_past": wc.parity(z, cached["latent_history"][:n]),
        "decoder_FP32": wc.parity(fitted.decoder32, local_arrays["decoder_deployed_FP32"][:n]),
        "decoder_bias_FP32": wc.parity(fitted.bias32, local_arrays["decoder_bias_FP32"][:n]),
        "singularvalues_FP64": wc.parity(np.asarray(fitted.diagnostics["singularvalues_observed_centered_Z"]),
            np.asarray(local_metadata["singularvalues_observed_centered_Z"])[:n])}
    decisions = {}
    for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept",
                "no_observation_current_C_fallback", "rank_deficient_slope_channels", "observed_count_below_two"):
        a, b = np.asarray(fitted.diagnostics[key]), np.asarray(local_metadata[key])[:n]
        decisions[key] = bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
    mask_equal = bool(observed[:n].dtype == values["history_observed"][:n].dtype
                      and observed[:n].tobytes() == values["history_observed"][:n].tobytes())
    metadata = {"status": "PASS" if all(item["pass"] for item in checks.values()) and all(decisions.values()) and mask_equal else "FAILED_PRECHECK_PRESERVED",
        "scope": "Actual physical first8 -> causal fill/FIT normalize -> fixed NumPy E -> local observed-past LS and implemented diagnostics; no F yet",
        "checks": checks, "exact_decisions": decisions, "history_observed_byte_exact": mask_equal,
        "singular_relative_cut": SINGULAR_RELATIVE_CUT, "origins": values["origins"][:n].tolist(),
        "rank_acceptance_by_output_tolerance_only": False, "coefficient_tolerance_loosened": False,
        "physical_input_rounding_can_change_local_solver_or_output": True,
        "fitted_local_metadata": fitted.metadata()}
    arrays = {**fitted.to_npz(), "preprocessed_history": history, "latent_history": z,
              "history_observed": observed[:n].copy(), "origins": values["origins"][:n].copy()}
    return metadata, arrays


def execute(tag, local_path, local_audit_path):
    producer, local, audit = load_gates(local_path, local_audit_path)
    require(Path(tag).name == tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_RUN_TAG_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_local_past_decoder_cost_v1 as wc
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
    result = {"status": "RUNNING", "producer": local["producer"], "fit_target_audit": local["fit_target_audit"],
        "local_diagnostic": record(local_path), "local_audit": record(local_audit_path), "cost": {}, "datasets": {},
        "source_snapshots": [], "source_records": [], "failed_paths": [], "routes": list(ROUTES),
        "protected_TEST_accessed": False, "independent_holdout": False, "new_training_updates": 0,
        "projection_fits": 0, "inference_time_local_decoder_fits": True, "future_labels_used_for_local_fit": False,
        "new_backbone_training": False, "new_native_head_forward": False, "native_deployment_cost_measured": False,
        "method_checkpoint_threshold_selection_used": False, "past_C_condition_metadata_added": True,
        "same_decoder_side_information_as_static": False, "strict_K_only_compression": False,
        "static_D_or_global_rankK_oracle_is_local_lower_bound": False,
        "decoder_input_interface": "K streams + current C + origin-specific KxC coefficients from known C past512 and observed mask",
        "local_LS_and_implemented_diagnostics_inside_timer": True, "local_coefficients_cached_inside_timed_pipeline": False,
        "native_heads_loaded_or_timed": False, "hard_elapsed_or_cumulative_resource_caps": None,
        "authorization": record(rt.AUTH), "continuation_manifest": record(rt.MANIFEST),
        "balanced_scope": "First five circular rotations perfectly balanced; sixth seeded extra is not full-six Latin balance",
        "timing_helper_copy": {"original": record(HERE / "whole_fit_target_cost_v1.py"),
            "changes": "Five F-only routes including two localD callbacks, explicit observed mask; frozen timer bounds/parity retained"}}
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
        guarded = prior_file_records(producer, local, audit, record(local_path), record(local_audit_path))
        for parent in (producer,):
            for item in parent.get("source_snapshots", []):
                require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "ACCEPTED_SOURCE_SNAPSHOT_CHANGED")
                guarded.append({"path": item["snapshot_path"], "sha256": item["sha256"]})
        source_folder = output / "sources"
        source_folder.mkdir()
        for path in (Path(__file__), HERE / "whole_local_past_decoder_cost_v1.py", HERE / "local_past_decoder_v1.py",
                     HERE / "projection_math_v1.py", HERE / "run_local_past_decoder_probe_v1.py",
                     HERE / "PLAN_LOCAL_PAST_DECODER_COST_KO_v1.md", HERE / "continuation_runtime_v1.py",
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
                    meter.check({"stage": "local_past_whole_cost", "dataset": dataset})
                    source, diagnostic = producer["datasets"][dataset], local["datasets"][dataset]["splits"][dev]
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
                        backend = diagnostic["routes"][route]["backends"]["BOLT_NUMPY"]
                        state = ProjectionState.from_npz(checked_load(condition["state_artifact"]))
                        cached = checked_load(condition["bolt"]["artifacts"][dev])
                        local_arrays = checked_load(backend["local_state_artifact"])
                        references = checked_load(backend["artifact"])
                        exact(cached["origins"], values["origins"], "cached_BOLT_origins")
                        exact(references["origins"], values["origins"], "LocalD_origins")
                        exact(references["raw_last"], values["raw_last"], "LocalD_currentC")
                        exact(references["latent_history"], cached["latent_history"], "LocalD_E_past")
                        exact(references["BOLT_cached_future"], cached["latent_future"], "LocalD_F_future")
                        exact(references["BOLT_static_prediction"], cached["prediction"], "static_original_forecast")
                        check, arrays = physical_local_precheck(physical["history_physical"], physical["history_observed"],
                            fit["mean"], fit["std"], values, state, cached, local_arrays, backend["local_fit_metadata"])
                        check["artifact"] = wc.save_outputs(output / (dataset + "_" + route + "_PHYSICAL_LOCAL_PRECHECK.npz"), arrays)
                        check.update(source_local_state=backend["local_state_artifact"], source_local_forecast=backend["artifact"],
                                     source_state=condition["state_artifact"], source_BOLT_cache=condition["bolt"]["artifacts"][dev])
                        prechecks[route] = check
                        result["datasets"][dataset]["physical_local_prechecks"] = prechecks
                        wc.write_json_once(output / (dataset + "_" + route + "_PHYSICAL_LOCAL_PRECHECK.json"), check)
                        require(check["status"] == "PASS", "PHYSICAL_LOCAL_COEFFICIENT_RANK_AND_CUTOFF_PRECHECK_FAILED:" + route)
                        static_route = route + "_BOLT"
                        local_route = route.replace("_PINV", "_LOCAL") + "_BOLT"
                        predictors[static_route], predictors[local_route] = static_predictor(state, frozen_F), local_predictor(state, frozen_F)
                        predictions[static_route], predictions[local_route] = references["BOLT_static_prediction"], references["BOLT_local_prediction"]
                        for name, is_local in ((static_route, False), (local_route, True)):
                            metadata[name] = {"source_state": condition["state_artifact"], "CPU_local_reference": backend["artifact"],
                                "reference_prediction_array": "BOLT_local_prediction" if is_local else "BOLT_static_prediction",
                                "local_fit": is_local, "local_fit_uses_current_known_past_and_true_observed_mask": is_local,
                                "local_LS_and_implemented_diagnostics_inside_timer": is_local, "cached_local_D_used_for_timing": False,
                                "singular_relative_cut": 1e-4 if is_local else None, "shared_resident_frozen_F": True,
                                "new_E_F_or_head_training": False, "native_head_loaded_or_timed": False,
                                "past_C_condition_metadata_added": is_local, "same_decoder_side_information_as_static": not is_local,
                                "strict_K_only_compression": False}
                    result["datasets"][dataset].update(inputs={"fit": source["inputs"]["fit"], dev: source["inputs"][dev]},
                        physical_bridge=source["physical_bridge"], local_CPU_cell_source=diagnostic["input_source"],
                        data_contract=data_contract, selected_metadata=metadata)
                    memory = lambda: {"resident_rss_bytes": psutil.Process().memory_info().rss,
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                    result["cost"][dataset] = wc.measure_cost(dataset, physical["history_physical"], physical["history_observed"],
                        fit["mean"], fit["std"], values["origins"], predictors, predictions, metadata,
                        output / ("cost_" + dataset), meter.check, torch.cuda.synchronize, memory)
                    result["datasets"][dataset]["status"] = "COMPLETE"
                    print(json.dumps({"event": "LOCAL_PAST_WHOLE_COST_COMPLETE", "dataset": dataset, "cells": 10}), flush=True)
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
                    print(json.dumps({"event": "LOCAL_PAST_COST_PATH_FAILURE_PRESERVED", **failed}), flush=True)
        result["F_after"] = probes.model_hashes(bolt)
        require(result["F_before"] == result["F_after"], "FROZEN_BOLT_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in result["source_snapshots"]:
            require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_CHANGED")
        for item in guarded:
            probes.verified_file(item)
        load_gates(local_path, local_audit_path)
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
    parser.add_argument("--local-diagnostic", required=True, type=Path)
    parser.add_argument("--local-audit", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(args.tag, args.local_diagnostic, args.local_audit))
