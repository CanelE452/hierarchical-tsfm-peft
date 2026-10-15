"""Fresh matched TUNE cost: Raw plus reused full-gradient and new stop-gradient L0."""
from pathlib import Path
import argparse
import json
import shutil
import sys

import run_linear_e_cost_v1 as old

HERE, ROOT, BASE, OLD = old.HERE, old.ROOT, old.BASE, old.OLD
DATASETS, SEEDS = old.DATASETS, (92601, 92602)
ROUTES = ("RAW_BOLT", "FULL_s92601", "FULL_s92602", "STOP_s92601", "STOP_s92602")
STOP_RESULT_SHA = "e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a"
STOP_AUDIT_SHA = "64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f"
STOP_AUDIT_COUNTS = {"stop_gradient_fits_checked": 4, "checkpoint_states_checked": 20, "checkpoint_TUNE_predictions_checked": 20, "selected_FIT_DEV_predictions_checked": 8, "logged_updates_checked": 2400, "probe_cases_checked": 4, "selection_locks_checked": 4, "reused_full_gradient_conditions_checked": 4, "CP0_full_stop_parent_component_byte_comparisons_checked": 32, "saved_gradient_vector_fields_checked": 32, "analytic_gradient_comparisons_checked": 28}
require, record, bound, read = old.require, old.record, old.bound, old.read
physical_precheck, adapter_predictor = old.physical_precheck, old.adapter_predictor


def selected_rows(full, stopped, dataset):
    return [("FULL_s" + str(seed), full["datasets"][dataset]["conditions"][dataset + "_L0_E_LEARN_s" + str(seed)], "FULL_REUSED") for seed in SEEDS] + [("STOP_s" + str(seed), stopped["datasets"][dataset]["conditions"][str(seed)], "STOP_NEW") for seed in SEEDS]


def load_gates(result_path, audit_path):
    require(all(isinstance(v, str) and len(v) == 64 for v in (STOP_RESULT_SHA, STOP_AUDIT_SHA)), "ACTUAL_STOP_GRADIENT_PASS_NOT_REGISTERED")
    stopped, audit = read(result_path), read(audit_path); sr, ar = record(result_path), record(audit_path)
    require(sr["sha256"] == STOP_RESULT_SHA and ar["sha256"] == STOP_AUDIT_SHA and stopped["status"] == "COMPLETE" and audit["status"] == "PASS", "EXACT_COMPLETED_STOP_GRADIENT_AND_INDEPENDENT_PASS")
    bound(audit["diagnostic"], sr); bound(audit["audit_source"])
    require(all(audit["totals"].get(key) == n for key, n in STOP_AUDIT_COUNTS.items()), "FULL_FOUR_CONDITION_INDEPENDENT_COVERAGE")
    for key in ("full_gradient_parent", "full_gradient_parent_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "plan"): bound(audit[key], stopped[key])
    require(stopped["stop_gradient_training_fits"] == 4 and stopped["training_updates_completed"] == 2400 and stopped["checkpoint_states_saved"] == 20 and stopped["resolved_selection_locks"] == 4 and stopped["all_4_locks_before_DEV"] is True and stopped["probe_optimizer_updates"] == stopped["new_full_gradient_fits"] == 0 and stopped["full_gradient_conditions_reused"] == 4 and not stopped["failed_paths"], "CURRENT_COST_REQUIRES_FOUR_COMPLETE_STOP_CONDITIONS")
    for key in ("protected_TEST_accessed", "new_backbone_training", "independent_holdout", "strict_K_only", "whole_deployment_cost_measured"): require(stopped[key] is False, "EXACT_SCIENCE_SCOPE:" + key)
    require(stopped["controller_source"]["sha256"] == "0b13a1a6d2d2ac73af74a1e9bcc1383a1404b12016fc42a3cea0dbfdaf1a1ed5", "IMMUTABLE_EXECUTED_STOP_CONTROLLER"); bound(stopped["controller_source"])
    full, full_audit, science, subset, original, _, _ = old.load_gates(stopped["full_gradient_parent"]["path"], stopped["full_gradient_parent_audit"]["path"])
    require(full["F_before"] == full["F_after"] == stopped["F_before"] == stopped["F_after"], "SAME_FROZEN_F_FOR_BOTH_TRAINING_CONDITIONS")
    registry = read(bound(stopped["selection_registry"])); require(registry["status"] == "ALL_FOUR_RESOLVED_BEFORE_DEV" and len(registry["conditions"]) == 4 and registry["DEV_opened_before_seal"] is False, "ALL_FOUR_STOP_LOCKS_BEFORE_DEV")
    seen = set()
    for item in registry["conditions"]:
        key = item["dataset"], item["seed"]; require(key not in seen and key[0] in DATASETS and key[1] in SEEDS, "UNIQUE_STOP_SELECTION_KEY"); seen.add(key)
        chosen = stopped["datasets"][key[0]]["conditions"][str(key[1])]; lock = read(bound(item["selection_lock"], chosen["selection_lock"]))
        require(lock["status"] == chosen["status"] == "COMPLETE" and lock["selected_state"] == chosen["selected_state"] and lock["selected_update"] == chosen["selected_update"], "UNCHANGED_STOP_SELECTED_CP")
    refs = {"stop_gradient_diagnostic": sr, "stop_gradient_audit": ar, **{key: stopped[key] for key in ("full_gradient_parent", "full_gradient_parent_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry")}, "full_gradient_selection_registry": full["selection_registry"]}
    return stopped, audit, full, full_audit, science, subset, original, refs


def execute(tag, result_path, audit_path):
    output = old.fresh_output(tag)
    stopped, audit, full, full_audit, science, subset, original, refs = load_gates(result_path, audit_path)
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_f_input_gradient_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from linear_e_adaptation_v1 import LinearLagAdapter
    from native_head_v1 import state_hash
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(tuple(wc.ROUTES) == ROUTES and (wc.REPEATS, wc.WARMUP, wc.ORDER_SEED) == (5, 2, 20261015), "REGISTERED_FIVE_ROUTE_COST_CONTRACT")
    output.mkdir(parents=True, exist_ok=False); meter = rt.Meter("F_input_gradient_cost_" + tag, gpu=True)
    cost = {"status": "RUNNING", **refs, "cost_plan": record(HERE / "PLAN_F_INPUT_GRADIENT_ABLATION_KO_v1.md"), "routes": list(ROUTES), "cost_split": "tune", "datasets": {}, "cost": {}, "failed_paths": [], "source_snapshots": [], "source_records": [], "selection_locks": {}, "new_training_updates": 0, "new_encoder_decoder_fit": False, "new_backbone_training": False, "new_CP_or_seed_or_lag_selection": False, "per_origin_decoder_fit": False, "cached_future_returned_instead_of_actual_F_inside_timer": False, "selected_E_D_center_loaded_outside_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "native_predictor_loaded_or_timed": False, "seed_scope": "Two fixed-initialization minibatch-order conditions per gradient mode; not independent model initialization", "training_detach_does_not_change_deployment_formula": True}
    checked, models, prepared, failure = {}, {}, {}, None
    def load(item): return io.checked_payload({**item, "path": str(bound(item))}, meter, checked)
    def fail(error, **where):
        row = {"status": "FAILED_PATH_PRESERVED", **where, "failure": {"type": type(error).__name__, "message": str(error)}}; cost["failed_paths"].append(row); return row
    try:
        install_protected_test_guard(); meter.protected_check(); protected = prior_file_records(stopped, audit, full, full_audit, science, subset, original, *refs.values())
        folder = output / "sources"; folder.mkdir()
        for path in (Path(__file__), HERE / "whole_f_input_gradient_cost_v1.py", HERE / "run_linear_e_cost_v1.py", HERE / "whole_linear_e_cost_v1.py", HERE / "run_f_input_gradient_ablation_v1.py", HERE / "run_linear_e_adaptation_v1.py", HERE / "linear_e_adaptation_v1.py", HERE / "native_head_v1.py", HERE / "fixed_decoder_span_v1.py", HERE / "continuation_runtime_v1.py", HERE / "PLAN_F_INPUT_GRADIENT_ABLATION_KO_v1.md", OLD / "prepare_inputs.py", OLD / "run_condition_probes.py", BASE / "cost_readout_v1.py"):
            destination = folder / path.name; shutil.copyfile(path, destination); cost["source_snapshots"].append({**record(path), "snapshot_path": str(destination)})
        engine = probes.import_old_engine(); bolt = engine.ex.previous.FrozenBolt(); cost["F_before"] = probes.model_hashes(bolt)
        require(cost["F_before"] == stopped["F_before"], "EXACT_ORIGINAL_FROZEN_BOLT")
        weight, config = (probes.verified_file(original[key]) for key in ("model_binary_before", "model_config_before")); cost.update(model_binary_before=weight, model_config_before=config)
        cost["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        def raw_callback(history, history_observed):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)
        with threadpool_limits(limits=2):
            for data, (c, k, _, _) in DATASETS.items():
                row = cost["datasets"][data] = {"status": "PREPARING", "physical_prechecks": {}}
                try:
                    raw = load(subset["datasets"][data]["raw_input_sources"]["tune"]); proof = old.canonical_tune_bridge(original, data); physical = load(proof["canonical_record"]); row["canonical_physical_bridge"] = proof
                    require(raw["history"].shape == (64, 512, c) and all(old.byte_equal(raw[name], physical[name]) for name in ("origins", "history_observed", "mean", "std")), "SAME_ORIGINAL_PHYSICAL_TUNE_ORIGINS_MASK_SCALER")
                    p = {"raw": raw, "physical": physical, "predictors": {"RAW_BOLT": raw_callback}, "references": {"RAW_BOLT": raw["raw_forecast"]}, "metadata": {"RAW_BOLT": {"source": subset["datasets"][data]["raw_input_sources"]["tune"], "scope": "same-cohort actual full-C frozen Bolt"}}, "unavailable": {}}; prepared[data] = p
                except Exception as error: row.update(fail(error, dataset=data, stage="PHYSICAL_INPUT")); continue
                for route, selected, origin in selected_rows(full, stopped, data):
                    try:
                        state = load(selected["selected_state"]); require(set(state) == {"E", "D", "center"} and all(v.dtype == np.float32 for v in state.values()), "SELECTED_L0_FP32_STATE_ONLY")
                        model = LinearLagAdapter(state["center"], state["E"], state["D"], (0,), learn_E=True).to("cuda").eval().requires_grad_(False)
                        require(state_hash(model.state_dict()) == selected["selected_state_sha256"], "EXACT_SELECTED_E_D_CENTER_TENSOR_HASH"); models[data, route] = model
                        cp = old.selected_TUNE_checkpoint(selected); reference = load(cp["artifact"])
                        require(all(old.byte_equal(reference[name], raw[name]) for name in ("origins", "mean", "std")), "SAME_SELECTED_TUNE_REFERENCE_SCALER_ORIGINS")
                        lock = selected["selection_lock"]; bound(lock); cost["selection_locks"][data + "_" + route] = lock
                        p["predictors"][route] = adapter_predictor(model, bolt, engine.tensor); p["references"][route] = reference["prediction"]
                        p["metadata"][route] = {"origin": origin, "seed": selected["seed"], "lag": "L0", "selected_update": selected["selected_update"], "selected_state": selected["selected_state"], "selection_lock": lock, "reference": cp["artifact"], "reference_split": "tune", "reference_prediction_array": "prediction", "actual_F_forward": True, "current_C_anchor": True, "strict_K_only": False}
                        precheck, arrays = physical_precheck(physical, raw, reference, model, bolt, engine.tensor); precheck["artifact"] = io.save_npz(output / (data + "_" + route + "_PHYSICAL_FIRST8.npz"), arrays); row["physical_prechecks"][route] = precheck
                        require(precheck["input_backend_pass"] and precheck["prediction_backend_pass"], "REGISTERED_SELECTED_PHYSICAL_PREDICTION_PARITY")
                    except Exception as error:
                        error_row = fail(error, dataset=data, route=route, stage="SELECTED_STATE_OR_PRECHECK"); p["predictors"].pop(route, None); p["references"].pop(route, None); p["unavailable"][route] = error_row; p["metadata"].setdefault(route, {"unavailable": error_row})
            cost["selected_states_before"] = old.adapter_hashes(models); process = psutil.Process()
            def memory(): return {"RSS_bytes": process.memory_info().rss, "allocated_VRAM_bytes": torch.cuda.memory_allocated(), "reserved_VRAM_bytes": torch.cuda.memory_reserved(), "max_allocated_VRAM_bytes": torch.cuda.max_memory_allocated(), "max_reserved_VRAM_bytes": torch.cuda.max_memory_reserved(), "scope": "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation"}
            for data, p in prepared.items():
                try:
                    cost["cost"][data] = wc.measure_cost(data, p["physical"]["history_physical"], p["physical"]["history_observed"], p["raw"]["mean"], p["raw"]["std"], p["raw"]["origins"], p["predictors"], p["references"], p["metadata"], output / ("cost_" + data), meter.check, torch.cuda.synchronize, memory, p["unavailable"])
                    row = cost["datasets"][data]; row["status"] = cost["cost"][data]["status"]; cost["failed_paths"] += [{"dataset": data, **f} for f in cost["cost"][data]["failed_paths"]]
                except Exception as error:
                    cost["datasets"][data].update(fail(error, dataset=data, stage="TIMING")); partial = output / ("cost_" + data) / "RESULT.json"
                    if partial.exists(): cost["cost"][data] = read(partial)
        cost["F_after"] = probes.model_hashes(bolt); cost["selected_states_after"] = old.adapter_hashes(models)
        require(cost["F_before"] == cost["F_after"] and cost["selected_states_before"] == cost["selected_states_after"], "F_AND_ALL_SELECTED_STATES_UNCHANGED")
        cost["model_binary_after"], cost["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in protected + list(checked.values()) + list(cost["selection_locks"].values()): bound(item)
        for item in cost["source_snapshots"]: require(record(item["path"])["sha256"] == record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        rows = [r for dataset in cost["cost"].values() for r in dataset["rows"]]; complete, samples = sum(r["status"] == "COMPLETE" for r in rows), sum(len(r["samples"]) for r in rows)
        cost.update(selected_adapters_loaded=len(models), completed_cost_cells=complete, recorded_timer_samples=samples, expected_cost_cells=20, expected_timer_samples=100, input_records=list(checked.values()), protected_prior_records=protected, status="COMPLETE" if complete == 20 and samples == 100 and not cost["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; cost.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        cost["resource_terminal"] = meter.close(cost["status"], failure); rt.write_new(output / "RESULT.json", cost)
    print(json.dumps({"status": cost["status"], "result": str(output / "RESULT.json")}), flush=True)
    return 0 if cost["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tag", "stop-result", "stop-audit"): parser.add_argument("--" + key, required=True)
    arguments = parser.parse_args(); raise SystemExit(execute(arguments.tag, arguments.stop_result, arguments.stop_audit))
