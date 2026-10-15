"""Actual selected linear-adapter whole cost; closed until exact training PASS."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

from run_linear_e_adaptation_v1 import (HERE, BASE, ROOT, OLD, DATASETS, LAGS, CONDITION_KEYS,
    require, resolve, record, bound, read, fresh_output, byte_equal, name, validate_probe_registry)


TRAIN_RESULT_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
TRAIN_AUDIT_SHA = "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
TRAIN_AUDIT_COUNTS = {"adapter_fits_checked": 16, "checkpoint_states_checked": 80,
    "checkpoint_TUNE_predictions_checked": 80, "selected_FIT_DEV_predictions_checked": 32,
    "CP0_FIT_DEV_baselines_checked": 8, "logged_updates_checked": 9600,
    "probe_cases_checked": 4, "selection_locks_checked": 16}
CONTROLLER_SHA = "e268528d77d17b0c8f5a300ff8d667ce5ddd25b1f61754c3895df87929fcd1f3"
PURE_SHA = "17c8ac7d20b9f3dd759a3b3fb2e019e027469e9c8a057eb1176da98b9c319931"
ROUTES = ("RAW_BOLT", "L0_E_FIXED_s92601", "L0_E_FIXED_s92602", "L0_E_LEARN_s92601", "L0_E_LEARN_s92602",
    "L7_E_FIXED_s92601", "L7_E_FIXED_s92602", "L7_E_LEARN_s92601", "L7_E_LEARN_s92602")


def route(key):
    return name(key)[len(key[0])+1:]


def dataset_keys(dataset):
    return tuple(key for key in CONDITION_KEYS if key[0] == dataset)


def canonical_tune_bridge(producer, dataset):
    parent_binding = producer["source_parent_results"][dataset]
    parent = read(bound(parent_binding))
    require(parent.get("status") == "COMPLETE" and parent.get("protected_TEST_accessed") is False, "COMPLETE_NO_TEST_FIRST_PARENT")
    bridge_binding = parent["bridge_provenance"]["input_bridge"]
    bridge = read(bound(bridge_binding))
    require(bridge.get("status") == "COMPLETE" and bridge.get("actual_status", {}).get("protected_test_opened") is False,
        "COMPLETE_NO_TEST_CANONICAL_INPUT_BRIDGE")
    matches = [row for row in bridge["artifacts"] if row.get("dataset") == dataset and row.get("split") == "tune"]
    require(len(matches) == 1, "EXACTLY_ONE_EXISTING_CANONICAL_TUNE_PHYSICAL_ARTIFACT")
    canonical = matches[0]; path = resolve(canonical["path"])
    require(path.is_relative_to(ROOT) and path.suffix.lower() == ".npz" and path.name.lower() not in ("test.npz", "jena_test.npz"), "LOCAL_NON_TEST_TUNE_PHYSICAL_BINDING")
    require(path.is_file(), "EXISTING_TUNE_PHYSICAL_FILE_REQUIRED")
    arrays, raw = canonical.get("arrays", {}), producer["datasets"][dataset]["inputs"]["tune"]
    require({"history_physical", "history_observed", "origins", "mean", "std"}.issubset(arrays), "EXISTING_FULL_PHYSICAL_ARRAY_MANIFEST_REQUIRED")
    for field in ("history", "history_observed", "target", "observed", "raw_last", "origins", "mean", "std", "raw_forecast"):
        require(arrays.get(field) == raw["arrays"][field], "CANONICAL_TUNE_AND_RAW_ARRAY_BINDING:"+field)
    c = DATASETS[dataset][0]
    require(arrays["history"]["shape"] == arrays["history_physical"]["shape"] == arrays["history_observed"]["shape"] == [64, 512, c]
        and arrays["origins"]["shape"] == [64] and arrays["history_physical"]["dtype"] == "float32"
        and arrays["history_observed"]["dtype"] == "bool", "REGISTERED_EXISTING_TUNE_PHYSICAL_DIMENSIONS")
    return {"status": "EXACT_EXISTING_TUNE_CANONICAL_METADATA", "dataset": dataset, "split": "tune",
        "canonical_record": {**canonical, "path": str(path)}, "original_first_parent": parent_binding,
        "canonical_input_bridge": bridge_binding, "normalized_raw_input": raw,
        "complete_array_manifest_preserved": True, "inverse_normalization_used": False, "DEV_physical_alias_used": False,
        "loader_file_SHA_member_shape_dtype_SHA_checks_relaxed": False, "numeric_tolerance_changed": False}


def selected_TUNE_checkpoint(condition):
    matches = [row for row in condition["checkpoints"] if row["update"] == condition["selected_update"]]
    require(len(matches) == 1 and matches[0]["state"] == condition["selected_state"]
        and matches[0]["tune"] == condition["selected_tune"], "EXACT_REGISTERED_SELECTED_TUNE_CHECKPOINT")
    return matches[0]


def load_gates(training_path, audit_path):
    require(isinstance(TRAIN_RESULT_SHA, str) and len(TRAIN_RESULT_SHA) == 64
        and isinstance(TRAIN_AUDIT_SHA, str) and len(TRAIN_AUDIT_SHA) == 64 and bool(TRAIN_AUDIT_COUNTS), "ACTUAL_TRAINING_PASS_NOT_REGISTERED")
    training, audit = read(training_path), read(audit_path)
    training_record, audit_record = record(training_path), record(audit_path)
    require(training_record["sha256"] == TRAIN_RESULT_SHA and audit_record["sha256"] == TRAIN_AUDIT_SHA
        and training.get("status") == "COMPLETE" and audit.get("status") == "PASS", "EXACT_COMPLETE_TRAINING_AND_PASS_REQUIRED_BEFORE_SCIENTIFIC_IMPORT")
    bound(audit["diagnostic"], training_record)
    require(all(audit.get("totals", {}).get(key) == value for key, value in TRAIN_AUDIT_COUNTS.items()), "FULL_ACCEPTED_ADAPTATION_AUDIT_COVERAGE")
    require(training.get("mode") == "train" and training.get("adaptation_training_fits") == 16
        and training.get("training_updates_completed") == 9600 and training.get("checkpoint_states_saved") == 80
        and training.get("resolved_selection_locks") == 16 and training.get("all_16_locks_before_any_DEV") is True
        and training.get("failed_paths") == [] and training.get("F_before") == training.get("F_after"), "ALL_SIXTEEN_LOCKED_FRESH_ADAPTATION_CONDITIONS_REQUIRED")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "whole_deployment_cost_measured"):
        require(training.get(key) is False, "ACCEPTED_TRAINING_SCOPE:"+key)
    for key in ("probe_result", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry"):
        bound(audit[key], training[key])
    require(training["controller_source"]["sha256"] == CONTROLLER_SHA and training["pure_source"]["sha256"] == PURE_SHA,
        "EXACT_EXECUTED_ADAPTATION_SOURCES")
    bound(training["controller_source"]); bound(training["pure_source"]); bound(audit["audit_source"])
    from run_linear_e_adaptation_v1 import load_gates as parent_gates
    science, subset, original, parent_refs = parent_gates(*(training[key]["path"] for key in
        ("causal_lag_diagnostic", "causal_lag_audit", "parent_report", "parent_report_manifest", "parent_report_audit")))
    require(training["F_before"] == original["F_before"] == original["F_after"], "SAME_ORIGINAL_FROZEN_BOLT")
    probe = read(bound(training["probe_result"])); validate_probe_registry(probe)
    require(all(probe["cases"][data][lag]["status"] == "PASS" for data in DATASETS for lag in LAGS), "ALL_FOUR_ACTUAL_PROBE_PATHS_PASS")
    registry = read(bound(training["selection_registry"]))
    require(registry.get("status") == "ALL_16_RESOLVED_BEFORE_DEV" and registry.get("DEV_opened_before_seal") is False
        and len(registry.get("conditions", [])) == 16, "ALL_SIXTEEN_SELECTION_LOCK_REGISTRY")
    locks = {}
    for item in registry["conditions"]:
        key = item["dataset"], item["lag"], item["learn_E"], item["seed"]
        require(key in CONDITION_KEYS and key not in locks and item["status"] == "COMPLETE", "EXACT_UNIQUE_TRAINING_CONDITION")
        selected = training["datasets"][key[0]]["conditions"][name(key)]
        lock = read(bound(item["selection_lock"], selected["selection_lock"]))
        require(lock["status"] == "COMPLETE" and lock["DEV_opened_before_lock"] is False
            and lock["selected_state"] == selected["selected_state"] and lock["selected_update"] == selected["selected_update"]
            and selected["learn_E"] == key[2] and selected["seed"] == key[3] and selected["lag"] == key[1], "EXACT_SELECTED_CP_LOCK_BEFORE_DEV")
        locks[key] = item["selection_lock"]
    require(set(locks) == set(CONDITION_KEYS), "ALL_SIXTEEN_EXACT_SELECTION_KEYS")
    return training, audit, science, subset, original, {"linear_e_diagnostic": training_record, "linear_e_audit": audit_record,
        **{key: training[key] for key in ("probe_result", "selection_registry", "causal_lag_diagnostic", "causal_lag_audit")},
        **{key: value for key, value in parent_refs.items() if key.startswith("parent_report")}}, locks


def adapter_predictor(model, bolt, tensor):
    def callback(history, history_observed):
        import numpy as np
        import torch
        with torch.no_grad():
            return np.ascontiguousarray(model(tensor(history), bolt).detach().cpu().numpy().copy(), dtype=np.float32)
    return callback


def adapter_hashes(models):
    import hashlib
    from native_head_v1 import state_hash
    return {str(key): {"state_sha256": state_hash(model.state_dict()),
        "arrays": {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for name, value in model.state_dict().items()}} for key, model in models.items()}


def physical_precheck(physical, raw, reference, model, bolt, tensor):
    import numpy as np
    import torch
    import whole_linear_e_cost_v1 as wc
    history = wc.preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
    with torch.no_grad():
        prediction, z, f = model(tensor(history), bolt, return_components=True)
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in
            (("prediction", prediction), ("latent_history", z), ("latent_future", f), ("lag_features", model.lag_features(z, f)))}
    exact = {"physical_raw_origins": byte_equal(physical["origins"][:8], raw["origins"][:8]),
        "selected_raw_origins": byte_equal(reference["origins"][:8], raw["origins"][:8]),
        "physical_raw_history_observed": byte_equal(physical["history_observed"][:8], raw["history_observed"][:8]),
        "normalized_history_byte_exact": byte_equal(history, raw["history"][:8])}
    input_checks = {"history": wc.parity(history, raw["history"][:8]), "currentC": wc.parity(history[:, -1], raw["raw_last"][:8])}
    predicted = arrays["prediction"]; expected = reference["prediction"][:8]
    prediction_checks = {"normalized": wc.parity(predicted, expected), "physical": wc.parity(
        (predicted*raw["std"]+raw["mean"]).astype(np.float32), (expected*raw["std"]+raw["mean"]).astype(np.float32), wc.PHYSICAL_ATOL, wc.PHYSICAL_RTOL)}
    intermediate = {key: wc.parity(arrays[key], reference[key][:8]) for key in ("latent_history", "latent_future", "lag_features")}
    arrays.update(normalized_history=history, history_observed=physical["history_observed"][:8].copy(), origins=raw["origins"][:8].copy())
    return {"exact_lineage": exact, "input_checks": input_checks, "intermediate_checks": intermediate,
        "input_backend_pass": all(exact.values()) and all(row["pass"] for row in input_checks.values()),
        "prediction_backend_pass": all(row["pass"] for row in prediction_checks.values()), "prediction_checks": prediction_checks,
        "intermediate_parity_required_for_acceptance": False,
        "scope": "Exact input/state lineage and registered normalized/physical selected prediction parity; batch8 intermediate differences recorded separately"}, arrays


def execute(tag, training_path, audit_path):
    output = fresh_output(tag)
    training, audit, science, subset, original, bindings, locks = load_gates(training_path, audit_path)
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as runtime
    import fixed_decoder_span_v1 as io
    import whole_linear_e_cost_v1 as wc
    import linear_e_adaptation_v1 as pure
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from native_head_v1 import state_hash
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(tuple(wc.ROUTES) == ROUTES and wc.REPEATS == 9 and wc.WARMUP == 2, "FIXED_NINE_ROUTE_COST_CONTRACT")
    cost_plan = record(HERE/"PLAN_LINEAR_E_WHOLE_COST_KO_v1.md")
    output.mkdir(parents=True, exist_ok=False)
    meter = runtime.Meter("linear_E_cost_"+tag, gpu=True)
    result = {"status": "RUNNING", **bindings, "cost_plan": cost_plan, "routes": list(ROUTES), "cost": {}, "datasets": {},
        "selection_locks": {name(key): value for key, value in locks.items()}, "source_snapshots": [], "source_records": [], "failed_paths": [],
        "new_training_updates": 0, "new_backbone_training": False, "new_encoder_decoder_fit": False,
        "new_CP_or_seed_or_lag_selection": False, "native_predictor_loaded_or_timed": False,
        "selected_E_D_center_loaded_outside_timer": True, "per_origin_decoder_fit": False,
        "cached_future_returned_instead_of_actual_F_inside_timer": False,
        "all_source_file_state_metadata_hashes_outside_timer": True, "protected_TEST_accessed": False,
        "independent_holdout": False, "strict_K_only": False,
        "seed_scope": "Two minibatch-order conditions per design; not independent encoder initialization or independent methods",
        "cost_split": "tune", "cost_origins": "Same first B1/B8 TUNE origins; no DEV timings or inverse reconstructed physical inputs"}
    checked, protected, models, prepared, failure = {}, [], {}, {}, None
    def load(binding):
        return io.checked_payload({**binding, "path": str(bound(binding))}, meter, checked)
    def fail(error, **where):
        row = {"status": "FAILED_PATH_PRESERVED", **where, "failure": {"type": type(error).__name__, "message": str(error)}}
        result["failed_paths"].append(row); return row
    def unavailable(p, names, error):
        for route_name in names:
            p["predictors"].pop(route_name, None); p["references"].pop(route_name, None)
            p["unavailable"][route_name] = error; p["metadata"].setdefault(route_name, {"unavailable": error})
    try:
        install_protected_test_guard(); meter.protected_check()
        protected = prior_file_records(training, audit, science, subset, original, *bindings.values())
        folder = output/"sources"; folder.mkdir()
        for path in (Path(__file__), HERE/"whole_linear_e_cost_v1.py", HERE/"linear_e_adaptation_v1.py", HERE/"run_linear_e_adaptation_v1.py",
                HERE/"run_causal_lag_cost_v1.py", HERE/"continuation_runtime_v1.py", HERE/"fixed_decoder_span_v1.py", HERE/"native_head_v1.py",
                HERE/"PLAN_LINEAR_E_WHOLE_COST_KO_v1.md", OLD/"prepare_inputs.py",
                OLD/"run_condition_probes.py", BASE/"cost_readout_v1.py"):
            target = folder/path.name; shutil.copyfile(path, target)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(original[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt(); result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == training["F_before"], "SAME_ACTUAL_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        def raw_callback(history, history_observed):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)
        with threadpool_limits(limits=2):
            for data, (c, k, _, _) in DATASETS.items():
                row = result["datasets"][data] = {"status": "PREPARING", "physical_prechecks": {}}
                try:
                    raw = load(subset["datasets"][data]["raw_input_sources"]["tune"])
                    proof = canonical_tune_bridge(original, data)
                    physical = load(proof["canonical_record"]); row["canonical_physical_bridge"] = proof
                    require(raw["history"].shape == (64, 512, c) and byte_equal(raw["origins"], physical["origins"])
                        and byte_equal(raw["history_observed"], physical["history_observed"])
                        and byte_equal(raw["mean"], physical["mean"]) and byte_equal(raw["std"], physical["std"]), "SAME_PHYSICAL_INPUT_MASK_FIT_SCALER_ORIGINS")
                    p = {"raw": raw, "physical": physical, "predictors": {"RAW_BOLT": raw_callback}, "references": {"RAW_BOLT": raw["raw_forecast"]},
                        "metadata": {"RAW_BOLT": {"source": subset["datasets"][data]["raw_input_sources"]["tune"], "scope": "actual full-C frozen Bolt on registered TUNE origins"}}, "unavailable": {}}
                    prepared[data] = p
                except Exception as error: row.update(fail(error, dataset=data, stage="CANONICAL_PHYSICAL_INPUT")); continue
                for key in dataset_keys(data):
                    route_name = route(key)
                    try:
                        selected = training["datasets"][data]["conditions"][name(key)]
                        state = load(selected["selected_state"])
                        require(set(state) == {"E", "D", "center"} and all(value.dtype == np.float32 for value in state.values()), "EXACT_SELECTED_ADAPTER_STATE_MEMBERS_FP32")
                        model = pure.LinearLagAdapter(state["center"], state["E"], state["D"], LAGS[key[1]], learn_E=key[2]).to("cuda").eval().requires_grad_(False)
                        require(state_hash(model.state_dict()) == selected["selected_state_sha256"], "EXACT_TUNE_SELECTED_TENSOR_STATE_HASH")
                        models[key] = model
                        selected_cp = selected_TUNE_checkpoint(selected)
                        reference = load(selected_cp["artifact"])
                        require(byte_equal(reference["origins"], raw["origins"]) and byte_equal(reference["mean"], raw["mean"])
                            and byte_equal(reference["std"], raw["std"]), "SAME_SELECTED_TUNE_ORIGINS_AND_FIT_SCALER")
                        p["predictors"][route_name] = adapter_predictor(model, bolt, engine.tensor)
                        p["references"][route_name] = reference["prediction"]
                        p["metadata"][route_name] = {"training_condition": name(key), "lag": key[1], "lags": list(LAGS[key[1]]), "learn_E": key[2], "seed": key[3],
                            "selected_update": selected["selected_update"], "selected_state": selected["selected_state"], "selection_lock": locks[key],
                            "reference": selected_cp["artifact"], "reference_split": "tune", "reference_prediction_array": "prediction", "actual_F_forward": True,
                            "current_C_anchor": True, "known_K_past_buffer": True, "strict_K_only": False, "per_origin_fit": False}
                        precheck, arrays = physical_precheck(physical, raw, reference, model, bolt, engine.tensor)
                        precheck["artifact"] = io.save_npz(output/(data+"_"+route_name+"_PHYSICAL_FIRST8.npz"), arrays)
                        row["physical_prechecks"][route_name] = precheck
                        require(precheck["input_backend_pass"] and precheck["prediction_backend_pass"], "REGISTERED_SELECTED_PHYSICAL_PRECHECK")
                    except Exception as error:
                        unavailable(p, [route_name], fail(error, dataset=data, route=route_name, stage="SELECTED_STATE_OR_PHYSICAL_PRECHECK"))
            result["selected_states_before"] = adapter_hashes(models)
            process = psutil.Process()
            def memory():
                return {"RSS_bytes": process.memory_info().rss, "allocated_VRAM_bytes": torch.cuda.memory_allocated(), "reserved_VRAM_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_VRAM_bytes": torch.cuda.max_memory_allocated(), "max_reserved_VRAM_bytes": torch.cuda.max_memory_reserved(),
                    "scope": "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation"}
            for data, p in prepared.items():
                try:
                    result["cost"][data] = wc.measure_cost(data, p["physical"]["history_physical"], p["physical"]["history_observed"], p["raw"]["mean"], p["raw"]["std"],
                        p["raw"]["origins"], p["predictors"], p["references"], p["metadata"], output/("cost_"+data), meter.check, torch.cuda.synchronize, memory, p["unavailable"])
                    result["datasets"][data]["status"] = result["cost"][data]["status"]
                    result["failed_paths"] += [{"dataset": data, **item} for item in result["cost"][data]["failed_paths"]]
                except Exception as error:
                    result["datasets"][data].update(fail(error, dataset=data, stage="WHOLE_COST"))
                    partial = output/("cost_"+data)/"RESULT.json"
                    if partial.exists(): result["cost"][data] = read(partial)
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"] == result["F_after"], "UNCHANGED_F_DUAL_SIGNATURE")
        result["selected_states_after"] = adapter_hashes(models)
        require(result["selected_states_before"] == result["selected_states_after"], "UNCHANGED_SELECTED_E_D_CENTER")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in protected+list(checked.values())+list(locks.values()): bound(item)
        for item in result["source_snapshots"]:
            require(record(item["path"])["sha256"] == record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        rows = [row for value in result["cost"].values() for row in value["rows"]]
        complete, samples = sum(row["status"] == "COMPLETE" for row in rows), sum(len(row["samples"]) for row in rows)
        result.update(selected_adapters_loaded=len(models), completed_cost_cells=complete, recorded_timer_samples=samples,
            expected_cost_cells=36, expected_timer_samples=324,
            status="COMPLETE" if complete == 36 and samples == 324 and not result["failed_paths"] else "PARTIAL_COMPLETED",
            input_records=list(checked.values()), protected_prior_records=protected)
        meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        result["resource_terminal"] = meter.close(result["status"], failure)
        runtime.write_new(output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output/"RESULT.json")}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("tag", "training-result", "training-audit"): parser.add_argument("--"+argument, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.tag, args.training_result, args.training_audit))
