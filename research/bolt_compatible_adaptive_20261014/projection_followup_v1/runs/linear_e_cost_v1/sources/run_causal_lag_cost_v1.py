"""Gated actual frozen-F whole cost for fixed causal-lag decoder conditions."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
BASE, ROOT = HERE.parent, HERE.parents[2]
OLD = ROOT/"research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
SPACES = ("PCA", "FIT_TARGET", "RAW_SUBSET")
CONDITIONS = {"L0_RECON": ([0], "RECON"), "L7_RECON": ([0, 1, 2, 4, 8, 16, 32], "RECON"),
    "L0_FORECAST": ([0], "FORECAST"), "L7_FORECAST": ([0, 1, 2, 4, 8, 16, 32], "FORECAST")}
SCIENCE_RESULT_SHA = "4168cd05962e479130a722919310f4c192c1bab03d9b78cb7983754f2e502c17"
SCIENCE_AUDIT_SHA = "d395a82e76a31e38520db1dd4d56642963ca4e362f358a6d86480049d402a31d"
PURE_SHA = "f9846499e2cb9c8a322b25cb1e2c328a5eb3ff059d674a20ba172258c207ee6b"
SCIENCE_AUDIT_COUNTS = {"decoder_states_checked": 24, "fit_lock_files_checked": 24, "FIT_source_contexts_checked": 6,
    "cached_Bolt_source_contexts_checked": 18, "cached_D_split_prediction_arrays_checked": 72,
    "independent_FP32_method_prediction_arrays_checked": 216, "real_trace_prediction_arrays_checked": 216,
    "joint_output_diagnostic_cells_checked": 72, "EY_prediction_arrays_checked": 72, "flat_prediction_arrays_checked": 72,
    "retained_trace_inequalities_checked": 216, "original_real_trace_inequalities_evaluated": 216,
    "FP32_with_allowance_inequalities_evaluated": 216, "RAW_bound_inequalities_checked": 0,
    "cached_future_label_perturbation_checks": 72, "independent_fit_response_columns_checked": 336, "score_field_checks": 14688}


def require(value, message):
    if not value: raise ValueError(message)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def record(path):
    path = resolve(path)
    with path.open("rb") as handle: return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def bound(item, expected=None):
    path = resolve(item["path"])
    require(path.is_relative_to(ROOT) and path.name.lower() not in ("test.npz", "jena_test.npz"), "EXACT_LOCAL_NON_TEST_ASSET_REQUIRED")
    require(record(path)["sha256"] == item["sha256"], "EXACT_FILE_SHA:"+path.name)
    if "bytes" in item: require(path.stat().st_size == item["bytes"], "EXACT_FILE_BYTES")
    if expected is not None: require(path == resolve(expected["path"]) and item["sha256"] == expected["sha256"], "SAME_EXACT_PARENT_BINDING")
    return path


def read(path):
    path = resolve(path); require(path.suffix.lower() == ".json", "JSON_ONLY_METADATA_GATE")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fresh_output(tag):
    require(isinstance(tag, str) and bool(tag) and Path(tag).name == tag and tag not in (".", ".."), "ONE_FRESH_RUN_TAG_REQUIRED")
    output = (HERE/"runs"/tag).resolve()
    require(output.parent == (HERE/"runs").resolve() and not output.exists(), "FRESH_COST_OUTPUT_REQUIRED")
    return output


def load_gates(science_path, audit_path):
    require(all(isinstance(value, str) and len(value) == 64 for value in (SCIENCE_RESULT_SHA, SCIENCE_AUDIT_SHA)),
        "ACTUAL_CAUSAL_LAG_COMPLETE_AND_PASS_NOT_YET_REGISTERED")
    science, audit = read(science_path), read(audit_path)
    science_record, audit_record = record(science_path), record(audit_path)
    require(science_record["sha256"] == SCIENCE_RESULT_SHA and audit_record["sha256"] == SCIENCE_AUDIT_SHA
        and science.get("status") == "COMPLETE" and audit.get("status") == "PASS", "EXACT_COMPLETE_PASS_BEFORE_SCIENCE_IMPORT")
    bound(audit["diagnostic"], science_record)
    require(all(audit.get("totals", {}).get(key) == count for key, count in SCIENCE_AUDIT_COUNTS.items()), "FULL_CAUSAL_LAG_SCIENCE_AUDIT_COVERAGE")
    require(science.get("decoder_fits_completed") == science.get("resolved_fit_lock_files") == 24
        and science.get("cached_D_predictions_completed") == science.get("joint_output_diagnostics_completed") == 72
        and science.get("EY_diagnostics_completed") == science.get("flat_diagnostics_completed") == 72
        and science.get("actual_F_cache_splits_reused") == 18 and science.get("failed_paths") == []
        and science.get("all_24_locks_before_TUNE_DEV_NPZ") is True, "FULL_24_LOCK_72_PREDICTION_SCIENCE_REQUIRED")
    for value in (science, audit): require(value.get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST")
    require(science.get("new_F_calls") == science.get("new_native_head_training") == 0
        and science.get("new_backbone_training") is False and science.get("independent_holdout") is False
        and science.get("TUNE_DEV_used_for_fit_or_selection") is False and science.get("oracle_used_for_fit_or_selection") is False,
        "CACHED_CPU_SCIENCE_AND_FIT_ONLY_D_SUPERVISION_SCOPE")
    for key in ("raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "plan", "fit_registry"):
        bound(audit[key], science[key])
    bound(audit["audit_source"])
    require(record(HERE/"causal_lag_decoder_v1.py")["sha256"] == PURE_SHA, "IMMUTABLE_ACCEPTED_PURE_DECODER")
    for item in science["source_records"]: bound(item)
    subset = read(bound(science["raw_subset_diagnostic"]))
    require(subset["status"] == "COMPLETE" and subset["F_before"] == subset["F_after"]
        and science["frozen_F_parent_signatures"] == {"before": subset["F_before"], "after": subset["F_after"]}, "ORIGINAL_F_SIGNATURE_LINEAGE")
    source = read(bound(subset["source_producer"]))
    require(source["status"] == "COMPLETE" and source["F_before"] == source["F_after"] == subset["F_before"], "ORIGINAL_F_MODEL_AND_CONFIG_SOURCE")
    registry = read(bound(science["fit_registry"]))
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and len(registry["conditions"]) == 24
        and registry["failed_fit_paths"] == 0 and registry["TUNE_DEV_NPZ_opened_before_seal"] is False, "ALL_24_SUCCESSFUL_FIT_LOCKS_REQUIRED")
    locks = {}
    for item in registry["conditions"]:
        key = item["dataset"], item["space"], item["condition"]
        require(key not in locks, "UNIQUE_FIT_CONDITION_REGISTRY")
        lock = read(bound(item["fit_lock"]))
        require(lock["status"] == "LOCKED_FIT_ONLY" and (lock["dataset"], lock["space"], lock["condition"]) == key
            and lock["TUNE_DEV_used_for_fit_or_selection"] is False and lock["oracle_used_for_fit_or_selection"] is False,
            "IMMUTABLE_FIT_ONLY_D_LOCK")
        lags, objective = CONDITIONS[key[2]]
        entry = science["datasets"][key[0]]["spaces"][key[1]]["conditions"][key[2]]
        require(lock["lags"] == entry["lags"] == lags and lock["objective"] == entry["objective"] == objective
            and lock["state_artifact"] == entry["state_artifact"] and item["fit_lock"] == entry["fit_lock"], "EXACT_DECLARED_LAGS_OBJECTIVE_AND_D_STATE")
        require(all(entry["splits"][split]["status"] == "COMPLETE" for split in ("fit", "tune", DATASETS[key[0]][2])), "COMPLETE_SCIENCE_REFERENCE_SPLITS")
        locks[key] = item["fit_lock"]
    require(set(locks) == {(data, space, condition) for data in DATASETS for space in SPACES for condition in CONDITIONS}, "EXACT_24_REGISTERED_D_KEYS")
    return science, subset, source, {"causal_lag_diagnostic": science_record, "causal_lag_audit": audit_record,
        **{key: science[key] for key in ("raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "plan", "fit_registry")}}, locks


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def bolt_predictor(state, indices, decoder, lags, frozen_F):
    from run_raw_subset_comparison_v2 import encode_numpy
    from causal_lag_decoder_v1 import predict
    def callback(history, history_observed):
        z = encode_numpy(state, indices, history)
        future = frozen_F(z)
        return predict(z, future, history[:, -1], decoder, lags)
    return callback


def physical_precheck(physical, observed, mean, std, raw, fixed, cached, state, indices, decoders, source_predictions, frozen_F):
    import numpy as np
    import whole_causal_lag_cost_v1 as wc
    from run_raw_subset_comparison_v2 import encode_numpy
    from causal_lag_decoder_v1 import predict
    history = wc.preprocess(physical[:8], observed[:8], mean, std)
    z = encode_numpy(state, indices, history); future = frozen_F(z)
    exact = {"normalized_history": byte_equal(history, raw["history"][:8]),
        "history_observed": byte_equal(observed[:8], raw["history_observed"][:8]),
        "origins": byte_equal(raw["origins"][:8], fixed["origins"][:8])}
    checks = {"history": wc.parity(history, raw["history"][:8]), "currentC": wc.parity(history[:, -1], raw["raw_last"][:8]),
        "latent_history": wc.parity(z, fixed["latent_history_BOLT_NUMPY"][:8]), "latent_future": wc.parity(future, cached["latent_future"][:8])}
    arrays = {"preprocessed_history": history, "history_observed": observed[:8].copy(), "origins": raw["origins"][:8].copy(),
        "latent_history": z, "actual_F_latent_future": future}
    predictions = {}
    for condition, (d, lags) in decoders.items():
        actual = predict(z, future, history[:, -1], d, lags)
        reference = source_predictions[condition][:8]
        predictions[condition] = {"normalized": wc.parity(actual, reference),
            "physical": wc.parity((actual*std+mean).astype(np.float32), (reference*std+mean).astype(np.float32), wc.PHYSICAL_ATOL, wc.PHYSICAL_RTOL)}
        arrays[condition+"_prediction_FP32"] = actual
        arrays[condition+"_decoder_FP32"], arrays[condition+"_lags"] = d.copy(), np.asarray(lags, np.int64)
    input_pass = all(exact.values()) and all(value["pass"] for value in checks.values())
    return {"status": "PASS" if input_pass and all(v["pass"] for pair in predictions.values() for v in pair.values()) else "FAILED_PRECHECK_PRESERVED",
        "input_backend_pass": input_pass, "checks": checks, "exact_decisions": exact, "prediction_checks": predictions,
        "origins": raw["origins"][:8].tolist(), "actual_F_invoked": True, "per_origin_decoder_fit": False,
        "scope": "first8 physical causal fill/FIT norm->actual E/F->each frozen lag D/currentC; no refit or cached-F return"}, arrays


def mark_unavailable(prepared, routes, failure):
    for route in routes:
        prepared["predictors"].pop(route, None); prepared["references"].pop(route, None)
        prepared["unavailable"][route] = failure; prepared["metadata"].setdefault(route, {"unavailable": failure})


def execute(tag, science_path, audit_path):
    output = fresh_output(tag)
    science, subset, source, bindings, locks = load_gates(science_path, audit_path)
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_causal_lag_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from projection_math_v1 import ProjectionState
    from run_fit_target_comparison_v1 import prior_file_records
    from run_matched_shrunk_native_cost_v2 import canonical_physical_bridge
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True, exist_ok=False)
    meter = rt.Meter("causal_lag_cost_"+tag, gpu=True)
    result = {"status": "RUNNING", **bindings, "routes": list(wc.ROUTES), "datasets": {}, "cost": {}, "failed_paths": [],
        "source_snapshots": [], "source_records": [], "new_training_updates": 0, "new_backbone_training": False,
        "new_native_head_training": 0, "native_heads_loaded_or_timed": False, "per_origin_decoder_fit": False,
        "fixed_FIT_decoder_loaded_before_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True,
        "cached_future_returned_instead_of_actual_F_inside_timer": False, "strict_K_only_compression": False,
        "input_scope": "K streams/history buffer and current C anchor; fixed D, no origin C regression coefficients",
        "protected_TEST_accessed": False, "independent_holdout": False, "new_D_or_lag_objective_selection": False,
        "fit_locks": {data: {space: {condition: locks[(data, space, condition)] for condition in CONDITIONS} for space in SPACES} for data in DATASETS},
        "cost_plan": record(HERE/"PLAN_CAUSAL_LAG_WHOLE_COST_KO_v1.md"), "hard_elapsed_or_cumulative_resource_caps": None}
    failure, checked, guarded, prepared, states, decoders = None, {}, [], {}, {}, {}
    started = time.perf_counter()
    def load(item):
        path = bound(item); values = io.checked_payload({**item, "path": str(path)}, meter, checked)
        result["source_records"].append({"path": str(path), "sha256": item["sha256"]}); return values
    def F(z):
        with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(z)).detach().cpu().numpy(), dtype=np.float32)
    def fail(error, **where):
        row = {"status": "FAILED_PATH_PRESERVED", **where, "failure": {"type": type(error).__name__, "message": str(error)}}
        result["failed_paths"].append(row); return row
    def state_hashes():
        return {"E": {str(key): {name: io.array_sha(np.asarray(getattr(value, name))) for name in ("center", "encoder", "decoder", "decoder_bias")} for key, value in states.items()},
            "D_states": {str(key): {name: io.array_sha(value) for name, value in arrays.items()} for key, arrays in decoders.items()}}
    try:
        install_protected_test_guard(); meter.protected_check()
        guarded = prior_file_records(science, read(audit_path), subset, source, *bindings.values())
        folder = output/"sources"; folder.mkdir()
        for path in (Path(__file__), HERE/"whole_causal_lag_cost_v1.py", HERE/"causal_lag_decoder_v1.py",
                HERE/"run_causal_lag_objective_v1.py", HERE/"run_raw_subset_comparison_v2.py", HERE/"raw_subset_projection_v2.py",
                HERE/"raw_subset_projection_v1.py", HERE/"projection_math_v1.py", HERE/"fixed_decoder_span_v1.py",
                HERE/"run_matched_shrunk_native_cost_v2.py", HERE/"PLAN_CAUSAL_LAG_WHOLE_COST_KO_v1.md", HERE/"continuation_runtime_v1.py",
                BASE/"cost_readout_v1.py", OLD/"run_condition_probes.py", OLD/"prepare_inputs.py"):
            target = folder/path.name
            with path.open("rb") as src, target.open("xb") as dst: shutil.copyfileobj(src, dst)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(source[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt(); bolt.model.eval().requires_grad_(False)
        result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == source["F_before"], "ORIGINAL_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        with threadpool_limits(limits=2):
            for dataset, (c, k, dev, count) in DATASETS.items():
                row = result["datasets"][dataset] = {"status": "PREPARING", "spaces": {}, "physical_prechecks": {}}
                try:
                    raw = load(subset["datasets"][dataset]["raw_input_sources"][dev])
                    proof = canonical_physical_bridge(source, dataset)
                    require(proof["canonical_record"] == subset["datasets"][dataset]["physical_bridge_canonical"], "SAME_CANONICAL_PHYSICAL_FULL_MANIFEST")
                    physical = load(proof["canonical_record"]); row["canonical_physical_bridge"] = proof
                    require(raw["history"].shape == (count, 512, c) and byte_equal(raw["origins"], physical["origins"])
                        and byte_equal(raw["history_observed"], physical["history_observed"])
                        and byte_equal(raw["mean"], physical["mean"]) and byte_equal(raw["std"], physical["std"]), "PHYSICAL_RAW_ORIGIN_MASK_FIT_SCALER_LINEAGE")
                    p = {"raw": raw, "physical": physical, "predictors": {"RAW_BOLT": lambda h, observed: F(h)},
                        "references": {"RAW_BOLT": raw["raw_forecast"]}, "metadata": {"RAW_BOLT": {"scope": "actual full-C frozen F", "source": subset["datasets"][dataset]["raw_input_sources"][dev]}}, "unavailable": {}}
                    prepared[dataset] = p
                except Exception as error: row.update(fail(error, dataset=dataset, stage="CANONICAL_PHYSICAL_INPUT")); continue
                for space in SPACES:
                    names = [space+"_"+condition+"_BOLT" for condition in CONDITIONS]
                    try:
                        parent = subset["datasets"][dataset]["spaces"][space]
                        branch = science["datasets"][dataset]["spaces"][space]
                        e = load(parent["E_state_artifact"])
                        indices = e["selected_indices"].tolist() if space == "RAW_SUBSET" else None
                        state = ProjectionState(e["center"], e["encoder"], np.zeros((k, c)), np.zeros(c), {})
                        states[(dataset, space)] = state
                        fixed = load(parent["inputs"][dev]); cached = load(parent["Bolt"][dev]["artifact"])
                        require(byte_equal(raw["origins"], fixed["origins"]) and byte_equal(raw["origins"], cached["origins"]), "SAME_E_AND_CACHED_F_ORIGINS")
                        ready, source_predictions = {}, {}
                        for condition, (lags, objective) in CONDITIONS.items():
                            route = space+"_"+condition+"_BOLT"
                            try:
                                entry = branch["conditions"][condition]; lock = read(bound(locks[(dataset, space, condition)]))
                                arrays = load(lock["state_artifact"])
                                require(set(arrays) == {"decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32", "lags"}
                                    and arrays["decoder_FP32"].dtype == np.float32 and arrays["decoder_FP32"].shape == (len(lags)*k, c)
                                    and arrays["lags"].dtype == np.int64 and arrays["lags"].tolist() == lags, "LOCKED_LAG_MAJOR_D_STATE")
                                reference = entry["splits"][dev]; prediction = load(reference["artifact"])
                                require(byte_equal(prediction["origins"], raw["origins"]), "SAME_SCIENCE_D_PREDICTION_ORIGINS")
                                d = arrays["decoder_FP32"]; ready[condition] = d, lags; decoders[(dataset, space, condition)] = arrays
                                source_predictions[condition] = prediction["ACTUAL_F_prediction_FP32"]
                                p["predictors"][route] = bolt_predictor(state, indices, d, lags, F)
                                p["references"][route] = source_predictions[condition]
                                p["metadata"][route] = {"space": space, "condition": condition, "lags": lags, "objective": objective,
                                    "E_state": parent["E_state_artifact"], "D_state": entry["state_artifact"], "fit_lock": locks[(dataset, space, condition)],
                                    "reference": reference["artifact"], "reference_prediction_array": "ACTUAL_F_prediction_FP32",
                                    "actual_F_forward": True, "cached_F_return_inside_timer": False, "per_origin_fit": False,
                                    "current_C_anchor": True, "known_K_past_buffer": True, "strict_K_only": False}
                            except Exception as error: mark_unavailable(p, [route], fail(error, dataset=dataset, space=space, condition=condition, stage="D_STATE_SETUP"))
                        check, arrays = physical_precheck(physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"],
                            raw, fixed, cached, state, indices, ready, source_predictions, F)
                        check["artifact"] = io.save_npz(output/(dataset+"_"+space+"_PHYSICAL_FIRST8.npz"), arrays)
                        row["physical_prechecks"][space] = check
                        require(check["input_backend_pass"], "ACTUAL_PHYSICAL_E_F_PRECHECK_FAILED")
                        for condition, checks in check["prediction_checks"].items():
                            if not all(value["pass"] for value in checks.values()):
                                mark_unavailable(p, [space+"_"+condition+"_BOLT"], fail(ValueError(condition), dataset=dataset, space=space, condition=condition, stage="D_PREDICTION_PRECHECK"))
                        row["spaces"][space] = {"E_state": parent["E_state_artifact"], "indices": indices,
                            "available_conditions": [name for name in CONDITIONS if space+"_"+name+"_BOLT" not in p["unavailable"]]}
                    except Exception as error: mark_unavailable(p, names, fail(error, dataset=dataset, space=space, stage="E_F_SETUP_OR_PRECHECK"))
            result["fixed_states_before"] = state_hashes()
            process = psutil.Process()
            def memory():
                return {"RSS_bytes": process.memory_info().rss, "allocated_VRAM_bytes": torch.cuda.memory_allocated(),
                    "reserved_VRAM_bytes": torch.cuda.memory_reserved(), "max_allocated_VRAM_bytes": torch.cuda.max_memory_allocated(),
                    "max_reserved_VRAM_bytes": torch.cuda.max_memory_reserved(), "scope": "shared resident frozen F, six E and all fixed D; no method-exclusive allocation"}
            for dataset, p in prepared.items():
                try:
                    value = wc.measure_cost(dataset, p["physical"]["history_physical"], p["physical"]["history_observed"],
                        p["raw"]["mean"], p["raw"]["std"], p["raw"]["origins"], p["predictors"], p["references"], p["metadata"],
                        output/("cost_"+dataset), meter.check, torch.cuda.synchronize, memory, p["unavailable"])
                    result["cost"][dataset] = value; result["datasets"][dataset]["status"] = value["status"]
                    result["failed_paths"] += [{"dataset": dataset, **item} for item in value["failed_paths"]]
                except Exception as error:
                    result["datasets"][dataset].update(fail(error, dataset=dataset, stage="WHOLE_COST"))
                    path = output/("cost_"+dataset)/"RESULT.json"
                    if path.exists(): result["cost"][dataset] = read(path)
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"] == result["F_after"], "FROZEN_F_CHANGED")
        result["fixed_states_after"] = state_hashes()
        require(result["fixed_states_before"] == result["fixed_states_after"], "FIXED_E_OR_D_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in guarded+list(checked.values())+list(locks.values()): bound(item)
        for item in result["source_snapshots"]:
            require(record(item["path"])["sha256"] == record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_CHANGED")
        rows = [row for value in result["cost"].values() for row in value["rows"]]
        complete, samples = sum(row["status"] == "COMPLETE" for row in rows), sum(len(row["samples"]) for row in rows)
        result.update(completed_cost_cells=complete, recorded_timer_samples=samples,
            completed_cell_timer_samples=sum(len(row["samples"]) for row in rows if row["status"] == "COMPLETE"),
            expected_cost_cells=52, expected_timer_samples=676,
            status="COMPLETE" if complete == 52 and samples == 676 and not result["failed_paths"] else "PARTIAL_COMPLETED",
            input_records=list(checked.values()), protected_prior_records=guarded)
        meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        result["wall_seconds"] = time.perf_counter()-started; result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output/"RESULT.json")}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tag", "science-result", "science-audit"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.tag, args.science_result, args.science_audit))
