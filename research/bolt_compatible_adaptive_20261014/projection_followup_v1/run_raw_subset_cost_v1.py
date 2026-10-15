"""Gated fixed-global-D raw-subset whole costs; science/model imports are deferred."""
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
SPACES, SEEDS = ("PCA", "FIT_TARGET", "RAW_SUBSET"), (92601, 92602)
SCIENCE_RUNNER = "run_raw_subset_comparison_v2"
SCIENCE_PROJECTION_MODULE = "raw_subset_projection_v2"
SCIENCE_AUDITOR = "audit_raw_subset_comparison_v3.py"
SCIENCE_RESULT_SHA = "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff"
SCIENCE_AUDIT_SHA = "e177652ac34414bccfbfc564da820fca0bde5a462922ceb0edaf04c47081e95d"
SCIENCE_AUDIT_COUNTS = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83,
    "global_decoder_states_checked": 12, "source_encoder_states_checked": 6, "source_split_contexts_checked": 18,
    "native_fits_checked": 12, "training_updates_logged_checked": 7200, "checkpoint_states_checked": 60,
    "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36,
    "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6,
    "actual_subset_F_model_batch_calls_registered": 103, "reused_mixed_F_cache_lineages_checked": 12,
    "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12,
    "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36,
    "new_fixed_D_real_method_inequalities_checked": 90, "RAW_bound_inequalities_checked": 0,
    "current_C_persistence_split_baselines_computed": 6, "score_field_checks": 1800,
    "protected_prior_file_hashes_checked": 778}
SCIENCE_SOURCE_SHA = {"run_raw_subset_comparison_v2.py": "2f7b162e196abb413dafee18aa6344a75a203ee7670fca6225ee093f5602bae4",
    "raw_subset_projection_v2.py": "39afcf042dfb8bd19497c28c1ff60b44720dd69fb18a9f632ed6cfe141e6f405",
    "raw_subset_native_head_v1.py": "4081130169e4e72de0d46e4130f79e5d1220ccf685a010802a1242c841bc9784"}
SCIENCE_EXPECTED = {"subset_selection_fits": 2, "global_decoder_fits": 12, "native_training_fits": 12,
    "training_updates": 7200, "checkpoint_states_saved": 60, "saved_native_split_predictions": 36,
    "saved_Bolt_split_predictions": 18, "actual_subset_F_split_inferences": 6,
    "reused_mixed_F_split_caches": 12, "resolved_native_lock_files": 12, "cached_future_label_perturbation_checks": 12}


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle: return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def read(path):
    value = resolve(path)
    if value.suffix != ".json": raise ValueError("JSON_ONLY_GATE_REQUIRED")
    return json.loads(value.read_text(encoding="utf-8-sig"))


def require(value, label):
    if not value: raise ValueError(label)


def bound(item, expected=None, label="binding"):
    path = resolve(item["path"])
    require(path.is_relative_to(ROOT) and record(path)["sha256"] == item["sha256"], "EXACT_LOCAL_SOURCE_SHA:"+label)
    if expected is not None: require(resolve(expected["path"])==path and expected["sha256"]==item["sha256"], "SAME_EXACT_BINDING:"+label)
    if "bytes" in item: require(item["bytes"]==path.stat().st_size, "EXACT_OPTIONAL_BYTES:"+label)
    return path


def load_gates(science_path, audit_path):
    require(all(isinstance(value, str) and len(value)==64 for value in (SCIENCE_RESULT_SHA, SCIENCE_AUDIT_SHA))
        and bool(SCIENCE_AUDIT_COUNTS), "ACTUAL_RAW_SUBSET_INDEPENDENT_PASS_NOT_YET_REGISTERED")
    science_path, audit_path = resolve(science_path), resolve(audit_path)
    science, audit = read(science_path), read(audit_path)
    science_record, audit_record = record(science_path), record(audit_path)
    require(science.get("status")=="COMPLETE" and audit.get("status")=="PASS"
        and science_record["sha256"]==SCIENCE_RESULT_SHA and audit_record["sha256"]==SCIENCE_AUDIT_SHA,
        "EXACT_COMPLETE_SCIENCE_AND_INDEPENDENT_PASS_BEFORE_ASSETS")
    require(all(record(HERE/name)["sha256"]==digest for name,digest in SCIENCE_SOURCE_SHA.items()), "FROZEN_EXECUTED_V2_SCIENCE_SOURCE_IDENTITY")
    bound(audit["diagnostic"], science_record, "raw subset accepted science")
    require(all(audit.get("totals", {}).get(key)==value for key, value in SCIENCE_AUDIT_COUNTS.items()), "FULL_RAW_SUBSET_INDEPENDENT_AUDIT_COVERAGE")
    require(all(science.get(key)==value for key, value in SCIENCE_EXPECTED.items()) and science.get("failed_paths")==[], "COMPLETE_PREDECLARED_TWELVE_FRESH_HEAD_SCIENCE_COHORT")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "existing_head_weights_reused",
        "future_labels_used_for_subset_or_decoder_fit", "per_origin_decoder_fit", "whole_deployment_cost_measured", "oracle_used_for_selection", "strict_K_only_compression"):
        require(science.get(key) is False, "REGISTERED_SCIENCE_SCOPE:"+key)
    require(audit.get("protected_TEST_accessed") is False and science["all_twelve_locks_before_any_DEV_input_or_prediction"] is True
        and science["F_before"]==science["F_after"], "UNCHANGED_F_NO_TEST_AND_TWELVE_LOCKS_BEFORE_DEV")
    code = audit["audit_source"]
    require(resolve(code["path"]).name==SCIENCE_AUDITOR, "CANONICAL_RAW_SUBSET_AUDITOR")
    bound(code, label="raw subset auditor source")
    for key in ("source_producer", "source_audit"): bound(audit[key], science[key], "same accepted parent "+key)
    for key in ("prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum"):
        bound(science[key], label="preserved v1 numeric repair "+key)
    prior = read(resolve(science["prior_partial"]["path"]))
    require(prior["status"]=="PARTIAL_COMPLETED" and prior["subset_selection_fits"]==prior["actual_subset_F_split_inferences"]==0
        and prior["native_training_fits"]==8 and prior["training_updates"]==4800 and prior["checkpoint_states_saved"]==40,
        "PRESERVED_V1_SUBSET_FAILURE_AND_COMPLETED_EIGHT_MIXED_FITS")
    identity = science["mixed_reexecution_identity"]
    expected_identity = {"mixed_native_conditions_compared":8, "checkpoint_states_and_TUNE_arrays_compared":40,
        "selected_native_split_arrays_compared":24, "mixed_Bolt_split_arrays_compared":12, "global_D_states_compared":8,
        "learning_curves_compared":8, "total_physical_native_fit_attempts_across_v1_v2":20, "distinct_registered_native_conditions":12}
    require(identity["status"]=="PASS" and not identity["mismatches"] and all(identity.get(key)==value for key,value in expected_identity.items())
        and identity["tolerance_used"] is identity["new_heads_warm_started_or_reused"] is identity["two_runs_independent_conditions_claimed"] is False,
        "EXACT_MIXED_REEXECUTION_IDENTITY_NOT_EIGHT_NEW_INDEPENDENT_CONDITIONS")
    source = read(bound(science["source_producer"], label="original FIT source"))
    parent_pass = read(bound(science["source_audit"], label="original FIT audit"))
    require(source.get("status")=="COMPLETE" and parent_pass.get("status")=="PASS"
        and source["protected_TEST_accessed"] is parent_pass["protected_TEST_accessed"] is False, "ORIGINAL_FIT_SOURCE_STILL_COMPLETE_PASS_NO_TEST")
    registry = read(bound(science["selection_registry"], label="all twelve native registry"))
    require(registry["status"]=="ALL_TWELVE_RESOLVED_BEFORE_DEV" and len(registry["conditions"])==12
        and registry["failed_paths_at_seal"]==[], "TWELVE_SUCCESSFUL_FIXED_CP_LOCKS")
    resolved, subset_locks = {}, {}
    for condition in registry["conditions"]:
        key = (condition["dataset"], condition["space"], condition["seed"])
        require(key not in resolved, "UNIQUE_REGISTERED_NATIVE_CONDITION")
        lock = read(bound(condition["lock"], label="native selected CP"))
        require(lock["status"]=="LOCKED" and (lock["dataset"], lock["space"], lock["seed"])==key
            and lock["DEV_used_for_CP_selection"] is False and lock["selected_update"] in (0, 100, 200, 400, 600), "TUNE_ONLY_SELECTED_NATIVE_CP")
        resolved[key] = condition["lock"]
    require(set(resolved)=={(dataset, space, str(seed)) for dataset in DATASETS for space in SPACES for seed in SEEDS}, "EXACT_TWELVE_REGISTERED_NATIVE_KEYS")
    for dataset in DATASETS:
        row = science["datasets"][dataset]
        require(set(row["spaces"])==set(SPACES), "THREE_REGISTERED_SPACES")
        subset_locks[dataset] = row["subset_selection_lock"]
        subset = read(bound(subset_locks[dataset], label="FIT-only subset lock"))
        require(subset["status"]=="LOCKED_FIT_PAST_ONLY" and len(subset["selected_indices"])==DATASETS[dataset][1], "FIXED_FIT_ONLY_SMALL_K_SUBSET")
        for space in SPACES:
            item = row["spaces"][space]
            require(item["status"]=="COMPLETE" and item["fixed_E_hashes_before"]==item["fixed_E_hashes_after"], "FIXED_COMPLETED_E_SPACE")
            for seed in SEEDS:
                head = item["native"][str(seed)]; lock = read(resolve(resolved[(dataset, space, str(seed))]["path"]))
                require(head["status"]=="COMPLETE" and head["metadata"]["updates"]==600
                    and head["metadata"]["selected_update"]==lock["selected_update"]
                    and head["metadata"]["canonical_prediction_batch_size"]==32
                    and head["metadata"]["fixed_global_FIT_decoder"] is True, "SAME_SELECTED_FRESH_NATIVE_AND_FIXED_GLOBAL_D")
                chosen = next(point for point in head["checkpoints"] if point["update"]==lock["selected_update"])
                bound(lock["selected_state_artifact"], chosen["state_artifact"], "same selected head state")
        bridge = row["canonical_physical_bridge_binding"]
        require(row["physical_bridge_canonical"]==bridge["canonical_record"] and "arrays" in row["physical_bridge_canonical"]
            and bridge["same_physical_file_path_and_SHA"] is bridge["complete_array_manifest_preserved"] is True
            and bridge["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is bridge["numeric_tolerance_changed"] is False,
            "CANONICAL_PHYSICAL_ARRAY_MANIFEST_NOT_ARRAYLESS_ALIAS")
    return science, source, {"raw_subset_diagnostic": science_record, "raw_subset_audit": audit_record,
        **{key:science[key] for key in ("source_producer", "source_audit", "prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum")}}, resolved, subset_locks


def byte_equal(a, b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def bolt_predictor(state, indices, decoder, frozen_F):
    from run_raw_subset_comparison_v2 import encode_numpy
    def predict(history, history_observed):
        z = encode_numpy(state, indices, history)
        return history[:, -1, None, :]+(frozen_F(z)-z[:, -1, None, :])@decoder
    return predict


def native_predictor(projection, indices, model):
    from run_raw_subset_comparison_v2 import encode_native
    from raw_subset_native_head_v1 import predict_components
    def predict(history, history_observed):
        z = encode_native(projection, indices, history)
        return predict_components(model, z, history[:, -1], batch_size=32)[0]
    return predict


def physical_backend_precheck(physical, observed, mean, std, raw, fixed, state, projection, indices,
                              decoder, backend, source_predictions, frozen_F=None, models=None, cached_future=None):
    import numpy as np
    import whole_raw_subset_cost_v1 as wc
    from run_raw_subset_comparison_v2 import encode_numpy, encode_native
    from raw_subset_native_head_v1 import predict_components
    h = wc.preprocess(physical[:8], observed[:8], mean, std)
    z = encode_numpy(state, indices, h) if backend=="BOLT_NUMPY" else encode_native(projection, indices, h)
    stored = fixed["latent_history_"+backend][:8]
    checks = {"history": wc.parity(h, raw["history"][:8]), "current_C": wc.parity(h[:, -1], raw["raw_last"][:8]),
        "latent_history": wc.parity(z, stored), "latent_last": wc.parity(z[:, -1], stored[:, -1])}
    exact = {"history_observed": byte_equal(observed[:8], raw["history_observed"][:8]),
        "origins": byte_equal(raw["origins"][:8], fixed["origins"][:8]), "normalized_history": byte_equal(h, raw["history"][:8])}
    arrays = {"preprocessed_history": h, "latent_history": z, "history_observed": observed[:8].copy(),
        "origins": raw["origins"][:8].copy(), "fixed_decoder_FP32": decoder.copy()}
    input_backend_pass = all(entry["pass"] for entry in checks.values()) and all(exact.values())
    predictions = {}
    if backend=="BOLT_NUMPY":
        future = frozen_F(z); p = h[:, -1, None, :]+(future-z[:, -1, None, :])@decoder
        checks["latent_future"] = wc.parity(future, cached_future[:8]); checks["prediction"] = wc.parity(p, source_predictions["BOLT"][:8])
        arrays.update(prediction=p, latent_future=future); predictions["BOLT"] = checks["prediction"]
    else:
        require(bool(models) and set(models)<=set(str(seed) for seed in SEEDS), "REGISTERED_AVAILABLE_SELECTED_NATIVE_HEADS_REQUIRED")
        for seed in models:
            try:
                p, f = predict_components(models[str(seed)], z, h[:, -1], batch_size=32)
                checks["native_s"+str(seed)] = wc.parity(p, source_predictions[str(seed)][:8])
                arrays["native_s"+str(seed)+"_prediction"], arrays["native_s"+str(seed)+"_latent_future"] = p, f
            except Exception as error:
                checks["native_s"+str(seed)] = {"pass": False, "failure": {"type": type(error).__name__, "message": str(error)}}
            predictions[str(seed)] = checks["native_s"+str(seed)]
    return {"status": "PASS" if all(entry["pass"] for entry in checks.values()) and all(exact.values()) else "FAILED_PRECHECK_PRESERVED",
        "backend": backend, "checks": checks, "exact_decisions": exact, "prediction_checks": predictions, "input_backend_pass": input_backend_pass,
        "per_origin_decoder_fit": False, "fixed_D_byte_source": wc.array_record(decoder), "origins": raw["origins"][:8].tolist(),
        "scope": "Physical first8->causal fill/FIT norm->declared E backend->actual F or selected canonical32 CPU2 head->global fixed D/currentC; no fit"}, arrays


def mark_unavailable(prepared, routes, failure):
    for route in routes:
        prepared["predictors"].pop(route, None); prepared["references"].pop(route, None)
        prepared["metadata"][route] = failure; prepared["unavailable"][route] = failure


def execute(tag, science_path, audit_path):
    science, source, bindings, locks, subset_locks = load_gates(science_path, audit_path)
    require(Path(tag).name==tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_COST_TAG_REQUIRED")
    output = HERE/"runs"/tag; require(not output.exists(), "FRESH_RAW_SUBSET_COST_OUTPUT_DIRECTORY")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_raw_subset_cost_v1 as wc
    import fixed_decoder_span_v1 as io
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from prepare_inputs import install_protected_test_guard
    from native_head_v1 import FrozenProjection, clone_state, state_hash
    from projection_math_v1 import ProjectionState
    from raw_subset_native_head_v1 import NativeFixedDecoder
    from run_fit_target_comparison_v1 import prior_file_records
    require(wc.sha(HERE/"whole_matched_shrunk_native_cost_v1.py")==wc.COPIED_TIMER_SHA
        and len(wc.ROUTES)==10 and wc.REPEATS==10 and len(wc.balanced_schedule())==10, "IMMUTABLE_COPIED_TIMER_AND_TEN_REGISTERED_ROUTES")
    output.mkdir(parents=True, exist_ok=False); meter = rt.Meter(tag, gpu=True)
    install_protected_test_guard(); meter.protected_check(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", **bindings, "routes": list(wc.ROUTES), "datasets": {}, "cost": {}, "failed_paths": [],
        "source_snapshots": [], "source_records": [], "new_training_updates": 0, "new_backbone_training": False,
        "new_encoder_subset_decoder_or_CP_selection": False, "per_origin_decoder_fit": False,
        "fixed_FIT_decoder_loaded_before_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True,
        "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only_compression": False,
        "native_head_device": "cpu", "native_threads": 2, "canonical_native_prediction_batch_size": 32,
        "native_deployment_cost_measured": False, "loaded_native_heads_count": 0,
        "selection_registry": science["selection_registry"], "native_selection_locks": {
            dataset: {space: {str(seed): locks[(dataset, space, str(seed))] for seed in SEEDS} for space in SPACES} for dataset in DATASETS},
        "subset_selection_locks": subset_locks, "cost_plan": record(HERE/"PLAN_RAW_SUBSET_WHOLE_COST_KO_v1.md"),
        "hard_elapsed_or_cumulative_resource_caps": None, "authorization": record(rt.AUTH), "continuation_manifest": record(rt.MANIFEST)}
    failure, checked, guarded, all_states, all_models, all_projections, all_decoders = None, {}, [], {}, {}, {}, {}
    immutable, prepared, started = {}, {}, time.perf_counter()
    def load(item):
        path = resolve(item["path"]); require(path.is_relative_to(ROOT), "LOCAL_NON_TEST_BOUND_ASSET")
        value = io.checked_payload({**item, "path": str(path)}, meter, checked)
        result["source_records"].append({"path": str(path), "sha256": item["sha256"]}); return value
    def F(z):
        with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(z)).detach().cpu().numpy(), dtype=np.float32)
    def fail(error, **where):
        row = {**where, "status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
        result["failed_paths"].append(row); return row
    try:
        guarded = prior_file_records(science, read(audit_path), source, *bindings.values())
        folder = output/"sources"; folder.mkdir()
        paths = (Path(__file__), HERE/"whole_raw_subset_cost_v1.py", HERE/(SCIENCE_PROJECTION_MODULE+".py"), HERE/"raw_subset_native_head_v1.py",
            HERE/(SCIENCE_RUNNER+".py"), HERE/"native_head_v1.py", HERE/"projection_math_v1.py", HERE/"fixed_decoder_span_v1.py",
            HERE/"continuation_runtime_v1.py", HERE/"PLAN_RAW_SUBSET_WHOLE_COST_KO_v1.md", BASE/"cost_readout_v1.py",
            OLD/"run_condition_probes.py", OLD/"prepare_inputs.py")
        for path in paths:
            target = folder/path.name
            with path.open("rb") as a, target.open("xb") as b: shutil.copyfileobj(a, b)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        weight, config = (probes.verified_file(science[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        engine = probes.import_old_engine(); bolt = engine.ex.previous.FrozenBolt(); bolt.model.eval().requires_grad_(False)
        result["F_before"] = probes.model_hashes(bolt); require(result["F_before"]==science["F_before"], "SAME_ACCEPTED_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size",
            "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used",
            "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
            require(result["runtime_behavior_contract"][key]==science["runtime_behavior_contract"][key], "SAME_SCIENTIFIC_F_RUNTIME:"+key)
        with threadpool_limits(limits=2):
            for dataset, (c, k, dev, count) in DATASETS.items():
                row = {"status": "PREPARING", "spaces": {}, "physical_backend_prechecks": {}}
                result["datasets"][dataset] = row
                try:
                    srow = science["datasets"][dataset]
                    raw = load(srow["raw_input_sources"][dev]); physical = load(srow["physical_bridge_canonical"])
                    require(raw["history"].shape==(count, 512, c) and raw["origins"].dtype==np.int64
                        and byte_equal(raw["origins"], physical["origins"]) and byte_equal(raw["history_observed"], physical["history_observed"])
                        and byte_equal(raw["mean"], physical["mean"]) and byte_equal(raw["std"], physical["std"]), "EXACT_PHYSICAL_PAST_ORIGINS_MASK_AND_FIT_STATS")
                    row.update(raw_input_source=srow["raw_input_sources"][dev], physical_bridge_canonical=srow["physical_bridge_canonical"],
                        canonical_physical_bridge_binding=srow["canonical_physical_bridge_binding"])
                    prepared[dataset] = {"raw": raw, "physical": physical, "predictors": {"RAW_BOLT": lambda h, m: F(h)},
                        "references": {"RAW_BOLT": raw["raw_forecast"]}, "metadata": {"RAW_BOLT": {"source_input": srow["raw_input_sources"][dev],
                            "reference_prediction_array": "raw_forecast", "full_C_F_called": True, "per_origin_decoder_fit": False}}, "unavailable": {}}
                except Exception as error: row.update(fail(error, dataset=dataset, stage="SHARED_PHYSICAL_INPUT")); continue
                p = prepared[dataset]
                for space in SPACES:
                    names = [space+"_BOLT", *(space+"_NATIVE_s"+str(seed) for seed in SEEDS)]
                    try:
                        condition = srow["spaces"][space]; E = load(condition["E_state_artifact"])
                        indices = E["selected_indices"].tolist() if space=="RAW_SUBSET" else None
                        state = ProjectionState(E["center"], E["encoder"], np.zeros((k,c)), np.zeros(c), {}) if indices is not None else ProjectionState.from_npz(load(condition["original_E_state"]))
                        require(byte_equal(np.asarray(state.center), E["center"]) and byte_equal(np.asarray(state.encoder), E["encoder"]), "SAME_SCIENTIFIC_FIXED_E_ARRAYS")
                        projection = FrozenProjection(state).cpu().eval().requires_grad_(False)
                        all_states[(dataset, space)], all_projections[(dataset, space)] = state, projection
                        require(state_hash(clone_state(projection))==condition["fixed_E_hashes_before"]["native_projection_state"], "SAME_SCIENTIFIC_NATIVE_E_STATE")
                        fixed = load(condition["inputs"][dev])
                        require(byte_equal(fixed["origins"], raw["origins"]) and byte_equal(fixed["history"], raw["history"]), "MATCHED_SCIENTIFIC_SPACE_INPUT_ORIGINS")
                    except Exception as error:
                        failure_row = fail(error, dataset=dataset, space=space, stage="SHARED_FIXED_E_AND_SPACE_INPUT")
                        row["spaces"][space] = failure_row; mark_unavailable(p, names, failure_row); continue
                    state_refs, prechecks, models, native_predictions = {}, {}, {}, {}
                    row["physical_backend_prechecks"][space] = prechecks
                    for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                        backend_routes = names[:1] if backend=="BOLT_NUMPY" else names[1:]
                        try:
                            item = condition["backends"][backend]["state_artifact"]; d = load(item)["decoder_FP32"]
                            require(d.dtype==np.float32 and d.shape==(k,c) and np.isfinite(d).all(), "GLOBAL_FIXED_FIT_D32_REQUIRED")
                            d = d.copy(); all_decoders[(dataset, space, backend)] = d; state_refs[backend] = item
                            if backend=="BOLT_NUMPY":
                                bolt_source = load(condition["Bolt"][dev]["artifact"])
                                require(byte_equal(bolt_source["origins"], raw["origins"]), "MATCHED_BOLT_REFERENCE_ORIGINS")
                                p["predictors"][space+"_BOLT"] = bolt_predictor(state, indices, d, F)
                                p["references"][space+"_BOLT"] = bolt_source["prediction"]
                                p["metadata"][space+"_BOLT"] = {"space": space, "backend": backend, "E_state": condition["E_state_artifact"],
                                    "global_D_state": item, "selected_CPU_reference": condition["Bolt"][dev]["artifact"], "reference_prediction_array": "prediction",
                                    "subset_selection_lock": subset_locks[dataset] if indices is not None else None, "selected_indices": indices,
                                    "actual_K_channel_F_inference": True, "full_C_slice_used_as_inference": False, "per_origin_decoder_fit": False}
                            else:
                                for seed in SEEDS:
                                    route = space+"_NATIVE_s"+str(seed)
                                    try:
                                        lock = read(resolve(locks[(dataset, space, str(seed))]["path"]))
                                        saved = load(lock["selected_state_artifact"]); model = NativeFixedDecoder(d).cpu()
                                        model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in saved.items()}, strict=True); model.eval().requires_grad_(False)
                                        require(byte_equal(model.decoder.detach().numpy(), d), "HEAD_HAS_SAME_FIXED_GLOBAL_NATIVE_D")
                                        ref = condition["native"][str(seed)]["artifacts"][dev]; prediction = load(ref)
                                        require(byte_equal(prediction["origins"], raw["origins"]), "SAME_SELECTED_NATIVE_ORIGINS")
                                        models[str(seed)] = model; native_predictions[str(seed)] = prediction["prediction"]
                                        all_models[(dataset, space, str(seed))] = model; result["loaded_native_heads_count"] += 1
                                        p["predictors"][route] = native_predictor(projection, indices, model); p["references"][route] = prediction["prediction"]
                                        p["metadata"][route] = {"space": space, "backend": backend, "E_state": condition["E_state_artifact"],
                                            "global_D_state": item, "selected_CPU_reference": ref, "reference_prediction_array": "prediction",
                                            "selected_head_lock": locks[(dataset, space, str(seed))], "selected_head_state": lock["selected_state_artifact"],
                                            "native_seed": seed, "selected_update": lock["selected_update"], "head_device": "cpu", "head_threads": 2,
                                            "canonical_prediction_batch_size": 32, "per_origin_decoder_fit": False, "F_forward_in_this_route": False}
                                    except Exception as error: mark_unavailable(p, [route], fail(error, dataset=dataset, route=route, stage="SELECTED_NATIVE_HEAD_SETUP"))
                            if backend=="NATIVE_TORCH32" and not models: continue
                            check, arrays = physical_backend_precheck(physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"], raw,
                                fixed, state, projection, indices, d, backend,
                                {"BOLT": bolt_source["prediction"]} if backend=="BOLT_NUMPY" else native_predictions,
                                F if backend=="BOLT_NUMPY" else None, models if backend=="NATIVE_TORCH32" else None,
                                bolt_source["latent_future"] if backend=="BOLT_NUMPY" else None)
                            check["artifact"] = io.save_npz(output/(dataset+"_"+space+"_"+backend+"_PHYSICAL_FIRST8.npz"), arrays); prechecks[backend] = check
                            require(check["input_backend_pass"], "PHYSICAL_FIRST8_BACKEND_INPUT_PARITY_FAILED")
                            for key, prediction_check in check["prediction_checks"].items():
                                if not prediction_check["pass"]:
                                    route = space+"_BOLT" if key=="BOLT" else space+"_NATIVE_s"+key
                                    mark_unavailable(p, [route], fail(ValueError(route), dataset=dataset, route=route, stage="PHYSICAL_FIRST8_PREDICTION_PARITY"))
                            if backend=="BOLT_NUMPY": require(check["checks"]["latent_future"]["pass"], "PHYSICAL_BOLT_FUTURE_PARITY_FAILED")
                        except Exception as error: mark_unavailable(p, backend_routes, fail(error, dataset=dataset, space=space, backend=backend, stage="FIXED_D_BACKEND_SETUP_OR_PRECHECK"))
                    row["spaces"][space] = {"status": "COMPLETE_PREPARED" if all(route not in p["unavailable"] for route in names) else "PARTIAL_PREPARED",
                        "E_state": condition["E_state_artifact"], "global_D_states": state_refs, "indices": indices,
                        "model_states_before": {seed: state_hash(clone_state(model)) for seed, model in models.items()}}
            immutable = {"E": {str(key): {name: io.array_sha(np.asarray(getattr(state,name))) for name in ("center","encoder","decoder","decoder_bias")} for key,state in all_states.items()},
                "projections": {str(key): state_hash(clone_state(value)) for key,value in all_projections.items()},
                "heads": {str(key): state_hash(clone_state(value)) for key,value in all_models.items()},
                "global_decoders": {str(key): io.array_sha(value) for key,value in all_decoders.items()}}
            result["fixed_states_before"] = immutable
            process = psutil.Process()
            def memory():
                return {"RSS_bytes": process.memory_info().rss, "allocated_VRAM_bytes": torch.cuda.memory_allocated(),
                    "reserved_VRAM_bytes": torch.cuda.memory_reserved(), "max_allocated_VRAM_bytes": torch.cuda.max_memory_allocated(),
                    "max_reserved_VRAM_bytes": torch.cuda.max_memory_reserved(), "scope": "shared resident F/E/D/all native heads"}
            for dataset, p in prepared.items():
                row = result["datasets"][dataset]; row["status"] = "MEASURING"
                try:
                    cost = wc.measure_cost(dataset, p["physical"]["history_physical"], p["physical"]["history_observed"], p["raw"]["mean"], p["raw"]["std"],
                        p["raw"]["origins"], p["predictors"], p["references"], p["metadata"], output/("cost_"+dataset),
                        meter.check, torch.cuda.synchronize, memory, p["unavailable"])
                    result["cost"][dataset] = cost; row["status"] = cost["status"]
                    result["failed_paths"] += [{"dataset": dataset, **item} for item in cost["failed_paths"]]
                except Exception as error:
                    row.update(fail(error, dataset=dataset, stage="WHOLE_COST"))
                    path = output/("cost_"+dataset)/"RESULT.json"
                    if path.exists(): result["cost"][dataset] = read(path)
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"]==result["F_after"], "FROZEN_BOLT_CHANGED")
        after = {"E": {str(key): {name: io.array_sha(np.asarray(getattr(state,name))) for name in ("center","encoder","decoder","decoder_bias")} for key,state in all_states.items()},
            "projections": {str(key): state_hash(clone_state(value)) for key,value in all_projections.items()},
            "heads": {str(key): state_hash(clone_state(value)) for key,value in all_models.items()},
            "global_decoders": {str(key): io.array_sha(value) for key,value in all_decoders.items()}}
        result["fixed_states_after"] = after; require(after==immutable, "FIXED_E_D_OR_SELECTED_HEAD_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in guarded+list(checked.values())+list(locks.values())+list(subset_locks.values())+[science["selection_registry"]]: bound(item, label="protected prior source state input lock")
        for item in result["source_snapshots"]:
            require(record(item["path"])["sha256"]==record(item["snapshot_path"])["sha256"]==item["sha256"], "EXECUTED_COST_SOURCE_CHANGED")
        rows = [row for value in result["cost"].values() for row in value["rows"]]
        complete = sum(row["status"]=="COMPLETE" for row in rows); samples = sum(len(row["samples"]) for row in rows)
        result.update(completed_cost_cells=complete, recorded_timer_samples=samples,
            completed_cell_timer_samples=sum(len(row["samples"]) for row in rows if row["status"]=="COMPLETE"),
            expected_cost_cells=40, expected_timer_samples=400, native_deployment_cost_measured=any("_NATIVE_s" in row["route"] and row["samples"] for row in rows),
            status="COMPLETE" if complete==40 and samples==400 and not result["failed_paths"] else "PARTIAL_COMPLETED",
            input_records=list(checked.values()), protected_prior_records=guarded)
        meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        result["wall_seconds"] = time.perf_counter()-started; result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output/"RESULT.json")}), flush=True)
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tag", "science-result", "science-audit"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.tag, args.science_result, args.science_audit))
