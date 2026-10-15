"""Actual whole F-only costs for explicitly passed past-constant output gates."""
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
ROUTES = ("RAW_BOLT", "RAW_BOLT_GATED", "PCA_PINV_BOLT", "PCA_PINV_BOLT_GATED", "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_BOLT_GATED")


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(condition, message):
    if not condition: raise ValueError(message)


def bind(reference, path, label):
    name = Path(reference["path"])
    resolved = name.resolve() if name.is_absolute() else (ROOT / name).resolve()
    require(resolved == Path(path).resolve() and reference["sha256"] == sha(path), "EXACT_SOURCE_BINDING:" + label)


def load_gates(producer_path, fit_audit_path, cpu_path, cpu_audit_path):
    """JSON/hashes only before any scientific asset or GPU loading."""
    producer, fit_audit, cpu, audit = (read(path) for path in (producer_path, fit_audit_path, cpu_path, cpu_audit_path))
    require(producer.get("status") == cpu.get("status") == "COMPLETE", "BOTH_COMPLETE_PRODUCERS_REQUIRED")
    require(fit_audit.get("status") == audit.get("status") == "PASS", "BOTH_INDEPENDENT_PASS_REQUIRED")
    bind(fit_audit["producer"], producer_path, "FIT audit")
    bind(cpu["producer"], producer_path, "CPU source")
    bind(cpu["independent_FIT_target_audit"], fit_audit_path, "CPU accepted FIT audit")
    bind(audit["diagnostic"], cpu_path, "CPU independent audit")
    bind(audit["producer"], producer_path, "CPU audit source")
    bind(audit["fit_target_audit"], fit_audit_path, "CPU audit FIT source")
    code = fit_audit["audit_source"]
    require(Path(code["path"]).name == "audit_fit_target_comparison_v2.py", "ACCEPTED_INDEPENDENT_FIT_V2_AUDIT_REQUIRED")
    bind(code, code["path"], "FIT audit code")
    require(all(fit_audit["totals"].get(key) == value for key, value in {
        "projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2,
        "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20,
        "recorded_timer_samples_checked": 120, "saved_prediction_arrays_checked": 88,
        "output_subspace_diagnostic_cells_checked": 12}.items()), "FULL_FIT_TARGET_V2_AUDIT_REQUIRED")
    require(all(audit["totals"].get(key) == value for key, value in {"dataset_split_cells_checked": 6,
        "method_pairs_checked": 42, "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42,
        "score_field_checks": 2016}.items()), "FULL_CPU_GATE_AUDIT_REQUIRED")
    require(cpu.get("completed_dataset_split_cells") == 6 and cpu.get("completed_method_pairs") == 42
            and not cpu.get("failed_paths") and cpu.get("new_training") is False
            and cpu.get("new_encoder_forecaster_or_head_forward") is False and cpu.get("E_F_native_head_forward_runs") == 0
            and cpu.get("future_labels_used_for_gate") is False and cpu.get("method_selection_used") is False,
            "FULL_PAST_ONLY_CACHED_CPU_PROBE_REQUIRED")
    require(cpu.get("past_C_condition_metadata_added") is True and cpu.get("same_decoder_side_information_as_ungated") is False
            and cpu.get("strict_K_only_compression") is False and cpu.get("static_original_fixed_D_floor_reused_as_gated_lower_bound") is False,
            "EXPLICIT_CHANGED_DECODER_SIDE_INFORMATION_REQUIRED")
    for value in (producer, fit_audit, cpu, audit): require(value.get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST_REQUIRED")
    require(producer["F_before"] == producer["F_after"] and producer["new_backbone_training"] is False
            and producer["prior_source_results_caches_and_selection_locks_unchanged"] is True
            and cpu["source_cache_selection_SHA_unchanged"] is True, "FROZEN_F_AND_ORIGINAL_RESULTS")
    require(set(producer["datasets"]) == set(cpu["datasets"]) == set(DATASETS), "TWO_REGISTERED_DATASETS")
    for dataset, (channels, k, dev) in DATASETS.items():
        source, diagnostic = producer["datasets"][dataset], cpu["datasets"][dataset]
        require(source["status"] == "COMPLETE" and source["K"] == diagnostic["K"] == k
                and diagnostic["channels"] == channels and set(source["projections"]) == set(PROJECTIONS), "SAME_C_K_FROZEN_PROJECTIONS")
        case = diagnostic["splits"][dev]
        require(case["status"] == "COMPLETE" and case["input_source"] == source["inputs"][dev], "SAME_DEVELOPMENT_INPUTS")
        for route in ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT"):
            require(case["methods"][route]["unchanged_coordinates_byte_exact"] is True
                    and case["methods"][route]["gated_coordinates_current_anchor_byte_exact"] is True,
                    "BYTE_EXACT_CPU_GATE_COORDINATES_REQUIRED")
    return producer, fit_audit, cpu, audit


def projected_predictor(center, encoder, decoder, frozen_F):
    import numpy as np
    center, encoder, decoder = (np.ascontiguousarray(value, dtype=np.float32).copy() for value in (center, encoder, decoder))
    require(center.ndim == 1 and encoder.shape[0] == len(center) and decoder.shape == (encoder.shape[1], len(center))
            and all(np.isfinite(value).all() for value in (center, encoder, decoder)), "MATCHED_FROZEN_FP32_E_D")
    def predict(history, history_observed):
        latent = (history - center) @ encoder
        future = frozen_F(latent)
        return np.ascontiguousarray(history[:, -1:] + (future - latent[:, -1:]) @ decoder, dtype=np.float32)
    return predict


def gated_predictor(ungated):
    from past_constant_gate_v1 import past_constant_gate, apply_output_gate
    def predict(history, history_observed):
        original = ungated(history, history_observed)
        controls = past_constant_gate(history, history_observed)
        return apply_output_gate(original, history[:, -1], controls["keep_forecast"])
    return predict


def execute(tag, producer_path, fit_audit_path, cpu_path, cpu_audit_path):
    producer, fit_audit, cpu, audit = load_gates(producer_path, fit_audit_path, cpu_path, cpu_audit_path)
    require(Path(tag).name == tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_RUN_TAG_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_past_constant_gate_cost_v1 as wc
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import matched_readout_contract_v1 as contract
    import cost_readout_v1 as old_cost
    from past_constant_gate_v1 import past_constant_gate

    require(tuple(wc.ROUTES) == ROUTES and len(wc.balanced_schedule()) == 6, "SIX_AUTHENTIC_F_ONLY_LATIN_COST_ROUTES")
    output = HERE / "runs" / tag; output.parent.mkdir(exist_ok=True); output.mkdir(exist_ok=False)
    meter = rt.Meter(tag, gpu=True); inputs.install_protected_test_guard()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", "producer": record(producer_path), "fit_target_audit": record(fit_audit_path),
        "gate_diagnostic": record(cpu_path), "gate_audit": record(cpu_audit_path), "cost": {}, "datasets": {},
        "source_snapshots": [], "source_records": [], "routes": list(ROUTES), "protected_TEST_accessed": False,
        "independent_holdout": False, "new_training_updates": 0, "projection_fits": 0, "new_backbone_training": False,
        "new_native_head_forward": False, "native_deployment_cost_measured": False, "method_selection_used": False,
        "decoder_input_interface": "K streams + current C + past-constant C-bit flags (observed counts computed from original past mask)",
        "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
        "strict_K_only_compression": False, "static_original_fixed_D_floor_is_gated_lower_bound": False,
        "hard_elapsed_or_cumulative_resource_caps": None, "authorization": record(rt.AUTH), "continuation_manifest": record(rt.MANIFEST),
        "forecast_contract": "FP32 raw_last + (latent_future - encoded_history_last) @ decoder, followed by past-only coordinate gate",
        "gate_contract": "force current C only if every one of 512 normalized causal-filled values equals last FP32 and true observed past count >=2",
        "gated_RAW_baseline_included": True, "balanced_scope": "All six routes occupy each of six positions exactly once",
        "timing_helper_copy": {"original": record(HERE / "whole_cost_v1.py"), "changes": "Six routes/Latin6 and explicit predictor(history, history_observed)"}}
    guarded, failure, started = [], None, time.perf_counter()
    def checked_load(item):
        arrays, checked = probes.load_checked_npz(item); result["source_records"].append(checked); guarded.append(dict(item))
        return {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    def exact(actual, expected, label):
        require(actual.dtype == expected.dtype and actual.shape == expected.shape and probes.array_sha(actual) == probes.array_sha(expected), "EXACT_CONDITION:" + label)
    try:
        meter.protected_check()
        for item in [producer["selection_lock"], cpu["scope"], *cpu["source_records"], *producer["protected_prior_records"]]:
            probes.verified_file(item); guarded.append(dict(item))
        for item in producer["source_snapshots"]:
            require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "ACCEPTED_PRODUCER_SOURCE_CHANGED")
            guarded.extend((dict(item), {"path": item["snapshot_path"], "sha256": item["sha256"]}))
        source_folder = output / "sources"; source_folder.mkdir()
        for path in (Path(__file__), HERE / "whole_past_constant_gate_cost_v1.py", HERE / "past_constant_gate_v1.py",
                     HERE / "continuation_runtime_v1.py", BASE / "cost_readout_v1.py", OLD / "run_condition_probes.py",
                     OLD / "prepare_inputs.py", OLD / "matched_readout_contract_v1.py"):
            target = source_folder / path.name
            with path.open("rb") as source, target.open("xb") as destination: shutil.copyfileobj(source, destination)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(producer[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt(); result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == producer["F_before"] == producer["F_after"], "EXACT_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        bolt.model.eval().requires_grad_(False)
        def frozen_F(history):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)
        def raw_predict(history, history_observed): return frozen_F(history)
        with threadpool_limits(limits=2):
            for dataset, (channels, k, dev) in DATASETS.items():
                meter.check({"stage": "load_past_constant_cost", "dataset": dataset})
                source = producer["datasets"][dataset]; diagnostic = cpu["datasets"][dataset]["splits"][dev]
                values, fit = checked_load(source["inputs"][dev]), checked_load(source["inputs"]["fit"])
                exact(values["mean"], fit["mean"], "FIT_mean"); exact(values["std"], fit["std"], "FIT_std")
                data_contract = contract.validate_raw_split(dataset, dev, values, fit["mean"], fit["std"])
                physical = checked_load(source["physical_bridge"])
                old_cost.validate_physical_inputs(physical["history_physical"], physical["history_observed"], values["history"], fit["mean"], fit["std"], values["origins"])
                refs = checked_load(diagnostic["artifact"])
                for key, value in (("origins", values["origins"]), ("raw_last", values["raw_last"]), ("FIT_mean", fit["mean"]), ("FIT_std", fit["std"])):
                    exact(refs[key], value, key)
                controls = past_constant_gate(values["history"], values["history_observed"])
                for key, value in controls.items(): exact(refs[key], value, "CPU_gate_" + key)
                physical_prefix = wc.preprocess(physical["history_physical"][:8], physical["history_observed"][:8], fit["mean"], fit["std"])
                physical_controls = past_constant_gate(physical_prefix, physical["history_observed"][:8])
                for key, value in physical_controls.items():
                    exact(value, refs[key][:8], "physical_PRE_TIMER_gate_" + key)
                gate_precheck = {"status": "PASS", "scope": "Same first eight origins cover B1 and B8; actual physical causal-fill/FIT-normalize FP32 output before timer",
                    "physical_source": source["physical_bridge"], "CPU_gate_source": diagnostic["artifact"],
                    "origins": values["origins"][:8].tolist(), "history_observed": wc.array_record(physical["history_observed"][:8]),
                    "preprocessed_history": wc.array_record(physical_prefix),
                    "normalized_history_parity": wc.parity(physical_prefix, values["history"][:8]),
                    "normalized_current_C_parity": wc.parity(physical_prefix[:, -1], refs["raw_last"][:8]),
                    "actual_physical_gate_arrays": {key: wc.array_record(value) for key, value in physical_controls.items()},
                    "CPU_reference_gate_arrays": {key: wc.array_record(refs[key][:8]) for key in physical_controls},
                    "all_three_gate_arrays_byte_exact": True,
                    "tolerance_parity_used_to_accept_gate_decision": False}
                require(gate_precheck["normalized_history_parity"]["pass"] and gate_precheck["normalized_current_C_parity"]["pass"], "PHYSICAL_PRE_TIMER_NORMALIZED_INPUT_PARITY")
                predictors = {"RAW_BOLT": raw_predict, "RAW_BOLT_GATED": gated_predictor(raw_predict)}
                predictions, metadata = {}, {}
                for projection in PROJECTIONS:
                    state_record = source["projections"][projection]["state_artifact"]; state = checked_load(state_record)
                    base = projection + "_BOLT"
                    predictors[base] = projected_predictor(state["center"], state["encoder"], state["decoder"], frozen_F)
                    predictors[base + "_GATED"] = gated_predictor(predictors[base])
                    cached = checked_load(source["projections"][projection]["bolt"]["artifacts"][dev])
                    exact(refs[base + "_ungated_prediction"], cached["prediction"], "accepted_" + base)
                exact(refs["RAW_BOLT_ungated_prediction"], values["raw_forecast"], "accepted_RAW")
                for route in ROUTES:
                    gated = route.endswith("_GATED"); base = route[:-6] if gated else route
                    prediction_key = base + ("_gated_prediction" if gated else "_ungated_prediction")
                    predictions[route] = refs[prediction_key]
                    metadata[route] = {"base_method": base, "gated": gated, "shared_resident_frozen_F": True,
                        "native_head_loaded_or_timed": False, "saved_reference": diagnostic["artifact"], "reference_prediction_array": prediction_key,
                        "history_observed_explicitly_passed": True, "past_C_metadata_added": gated,
                        "same_decoder_side_information_as_ungated": not gated,
                        "gate_rule": result["gate_contract"] if gated else None, "new_fit": False}
                    if base != "RAW_BOLT": metadata[route]["source_state"] = source["projections"][base[:-5]]["state_artifact"]
                result["datasets"][dataset] = {"status": "RUNNING", "K": k, "development_split": dev,
                    "inputs": {"fit": source["inputs"]["fit"], dev: source["inputs"][dev]}, "data_contract": data_contract,
                    "physical_bridge": source["physical_bridge"], "CPU_gate_reference": diagnostic["artifact"],
                    "gate_keep_metadata": wc.array_record(controls["keep_forecast"]), "physical_gate_precheck": gate_precheck,
                    "selected_metadata": metadata}
                memory = lambda: {"resident_rss_bytes": psutil.Process().memory_info().rss,
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                result["cost"][dataset] = wc.measure_cost(dataset, physical["history_physical"], physical["history_observed"],
                    fit["mean"], fit["std"], values["origins"], predictors, predictions, metadata,
                    output / ("cost_" + dataset), meter.check, torch.cuda.synchronize, memory)
                result["datasets"][dataset]["status"] = "COMPLETE"
                print(json.dumps({"event": "PAST_CONSTANT_COST_COMPLETE", "dataset": dataset, "cells": 12}), flush=True)
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"] == result["F_after"], "FROZEN_BOLT_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in result["source_snapshots"]: require(sha(item["path"]) == sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_CHANGED")
        for item in guarded: probes.verified_file(item)
        load_gates(producer_path, fit_audit_path, cpu_path, cpu_audit_path); meter.protected_check()
        result.update(status="COMPLETE", expected_cost_cells=24, recorded_timer_samples=144,
                      source_cache_selection_SHA_unchanged=True, original_selected_forecasts_and_locks_unchanged=True)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure); raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started; result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output / "RESULT.json", result); print(json.dumps({"status": result["status"], "output": str(output / "RESULT.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("tag", "producer", "fit-audit", "gate-diagnostic", "gate-audit"):
        parser.add_argument("--" + name, required=True, type=str if name == "tag" else Path)
    args = parser.parse_args(); execute(args.tag, args.producer, args.fit_audit, args.gate_diagnostic, args.gate_audit)
