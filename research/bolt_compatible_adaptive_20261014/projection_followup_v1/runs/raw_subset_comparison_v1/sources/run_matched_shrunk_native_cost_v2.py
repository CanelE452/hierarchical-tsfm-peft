"""Seven actual whole-pipeline costs for locked Bolt and freshly matched native heads."""
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
PROJECTIONS, SEEDS = ("PCA_PINV", "FIT_TARGET_PINV"), (92601, 92602)
ROUTES = ("RAW_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT",
    "PCA_SHRUNK_NATIVE_s92601", "PCA_SHRUNK_NATIVE_s92602", "FIT_TARGET_SHRUNK_NATIVE_s92601", "FIT_TARGET_SHRUNK_NATIVE_s92602")
MATCHED_SHA = "e13587c86a9f28b3dccc5bf445cb85cd855a8a20a0ce1d9caad7dd57b93ba77a"
MATCHED_AUDIT_SHA = "0eb57a67b2ee3c25a7a5309efbdb62e55f756fb0c9552bd251b50f2ea6505077"
MATCHED_AUDIT_COUNTS = {"native_fits_checked": 8, "training_updates_logged_checked": 4800,
    "checkpoint_states_checked": 40, "checkpoint_TUNE_prediction_arrays_checked": 40,
    "selected_FIT_TUNE_DEV_prediction_arrays_checked": 24, "source_backend_contexts_checked": 12,
    "preserved_control_prediction_arrays_checked": 48, "persistence_baseline_contexts_computed": 12,
    "unique_persistence_split_cells_computed": 6, "selection_locks_checked": 8,
    "resolved_registry_conditions_checked": 8, "future_label_perturbation_scope_checks": 8,
    "score_field_checks": 2064, "protected_prior_file_hashes_checked": 1129}
FAILED_COST_SHA = "6940e169e2acaa3c27b08c1c80e1e8ae2ddcef944a100af331629165fff09ea9"
FAILURE_IDENTITY_SHA = "408933948ecfa4e171a493f868c136a96b906417e0096c661e95394655d63171"
V1_RUNNER_SHA = "13a91ce10db4fe9a71f1d67b238bc81aad8d139def9c4309b03a66f4cb0ea255"
TIMER_HELPER_SHA = "6779646de2a809c625a2b503b60ac03a64ae72826b15b8efad1d6d2213ae6cc2"


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition: raise ValueError(message)


def verify_failed_cost_origin():
    path = HERE / "runs/matched_shrunk_native_cost_v1/RESULT.json"
    binding = record(path)
    require(binding["sha256"] == FAILED_COST_SHA,
        "PRESERVED_ACTUAL_FAILED_COST_V1_REQUIRED")
    value = read(path)
    require(value["status"] == "PARTIAL_COMPLETED" and value["completed_cost_cells"] == 0
            and value["recorded_timer_samples"] == 0 and len(value["failed_paths"]) == 2
            and all(row["failure"] == {"type": "KeyError", "message": "'arrays'"} for row in value["failed_paths"])
            and value["F_before"] == value["F_after"], "ZERO_SAMPLE_ARRAYS_SCHEMA_FAILURE_SCOPE_REQUIRED")
    require(record(HERE / "run_matched_shrunk_native_cost_v1.py")["sha256"] == V1_RUNNER_SHA
            and record(HERE / "whole_matched_shrunk_native_cost_v1.py")["sha256"] == TIMER_HELPER_SHA,
            "ORIGINAL_EXECUTED_COST_RUNNER_AND_TIMER_PRESERVED")
    identity_path = HERE / "audits/MATCHED_NATIVE_COST_INPUT_SCHEMA_FAILURE_IDENTITY_v1.json"
    require(record(identity_path)["sha256"] == FAILURE_IDENTITY_SHA, "EXACT_INPUT_SCHEMA_FAILURE_IDENTITY_REQUIRED")
    identity = read(identity_path)
    require(identity["failed_cost"] == binding and identity["actual_cost_cells"] == 0 and identity["actual_timer_samples"] == 0
            and identity["method_or_training_failure_inferred"] is False and identity["original_files_changed"] is False
            and identity["protected_TEST_accessed"] is False, "PRESERVED_ZERO_SAMPLE_IMPLEMENTATION_FAILURE_IDENTITY_SCOPE")
    return binding


def canonical_physical_bridge(producer, dataset):
    """Rebind one existing alias to its exact complete bridge array manifest, JSON only."""
    require(dataset in DATASETS, "REGISTERED_PHYSICAL_BRIDGE_DATASET_REQUIRED")
    alias = producer["datasets"][dataset]["physical_bridge"]
    def resolve(item):
        path = Path(item["path"])
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    parent_binding = producer["source_parent_results"][dataset]
    parent_path = resolve(parent_binding)
    require(record(parent_path)["sha256"] == parent_binding["sha256"], "EXACT_ORIGINAL_FIRST_PARENT_JSON_BINDING")
    parent = read(parent_path)
    require(parent.get("status") == "COMPLETE" and parent.get("protected_TEST_accessed") is False,
        "COMPLETE_NO_TEST_ORIGINAL_FIRST_PARENT_REQUIRED")
    bridge_binding = parent["bridge_provenance"]["input_bridge"]
    bridge_path = resolve(bridge_binding)
    require(record(bridge_path)["sha256"] == bridge_binding["sha256"], "EXACT_CANONICAL_INPUT_BRIDGE_JSON_BINDING")
    bridge = read(bridge_path)
    require(bridge.get("status") == "COMPLETE" and bridge.get("actual_status", {}).get("protected_test_opened") is False,
        "COMPLETE_NO_TEST_CANONICAL_INPUT_BRIDGE_REQUIRED")
    dev = DATASETS[dataset][2]
    candidates = [row for row in bridge["artifacts"] if row.get("dataset") == dataset and row.get("split") == dev]
    require(len(candidates) == 1, "EXACTLY_ONE_CANONICAL_DATASET_DEV_ARTIFACT_REQUIRED")
    canonical = candidates[0]
    require(resolve(canonical) == resolve(alias) and canonical["sha256"] == alias["sha256"]
            and alias.get("verified_sha256", alias["sha256"]) == alias["sha256"], "ALIAS_AND_CANONICAL_SAME_PHYSICAL_FILE_AND_SHA")
    require(resolve(canonical).is_relative_to(ROOT) and resolve(canonical).suffix.lower() == ".npz"
            and resolve(canonical).name.lower() not in ("test.npz", "jena_test.npz"), "APPROVED_LOCAL_NON_TEST_PHYSICAL_ALIAS")
    arrays = canonical.get("arrays")
    require(isinstance(arrays, dict) and {"history_physical", "history_observed", "origins", "mean", "std"}.issubset(arrays),
        "FULL_CANONICAL_ARRAY_MANIFEST_REQUIRED_NOT_INFERRED")
    raw_arrays = producer["datasets"][dataset]["inputs"][dev]["arrays"]
    for name in ("history", "target", "observed", "raw_last", "origins", "mean", "std", "history_observed"):
        require(name in arrays and arrays[name] == raw_arrays[name], "CANONICAL_PHYSICAL_AND_ACCEPTED_RAW_ARRAY_BINDING:" + name)
    normalized = {**canonical, "path": str(resolve(canonical))}
    return {"status": "EXACT_CANONICAL_METADATA_REBOUND", "original_alias": alias, "original_first_parent": parent_binding,
        "canonical_input_bridge": bridge_binding, "canonical_record": normalized, "dataset": dataset, "split": dev,
        "same_physical_file_path_and_SHA": True, "complete_array_manifest_preserved": True,
        "loader_file_SHA_member_shape_dtype_SHA_checks_relaxed": False, "numeric_tolerance_changed": False,
        "metadata_only_derivation_no_NPZ_or_model_open": True}


def load_gates(matched_path, audit_path):
    """Only JSON and source hashes; no scientific import before accepted eight locks."""
    require(isinstance(MATCHED_AUDIT_SHA, str) and len(MATCHED_AUDIT_SHA) == 64 and bool(MATCHED_AUDIT_COUNTS),
        "ACTUAL_INDEPENDENT_MATCHED_PASS_BINDING_NOT_YET_REGISTERED")
    matched, audit = read(matched_path), read(audit_path)
    binding, audit_binding = record(matched_path), record(audit_path)
    require(matched.get("status") == "COMPLETE" and audit.get("status") == "PASS"
            and binding["sha256"] == MATCHED_SHA and audit_binding["sha256"] == MATCHED_AUDIT_SHA,
            "EXACT_COMPLETE_MATCHED_AND_ACTUAL_INDEPENDENT_PASS_BEFORE_SCIENCE")
    require(audit.get("diagnostic") == binding, "EXACT_MATCHED_INDEPENDENT_DIAGNOSTIC_BINDING")
    require(all(audit.get("totals", {}).get(key) == value for key, value in MATCHED_AUDIT_COUNTS.items()),
        "FULL_MATCHED_NATIVE_INDEPENDENT_AUDIT_COVERAGE_REQUIRED")
    expected = {"native_training_fits": 8, "completed_training_updates": 4800, "checkpoint_states_saved": 40,
        "completed_fit_tune_predictions": 16, "completed_DEV_predictions": 8, "completed_selection_locks": 8,
        "resolved_condition_lock_files": 8, "completed_source_backend_cells": 12, "future_label_perturbation_checks": 8}
    require(all(matched.get(key) == value for key, value in expected.items()) and matched.get("failed_paths") == [],
        "ALL_EIGHT_COMPLETE_FRESH_MATCHED_NATIVE_FITS_REQUIRED")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "new_E_F_forward_or_training",
                "new_encoder_decoder_fit", "new_GPU_execution", "rho_selection_repeated", "rho_selected_by_native",
                "same_Bolt_native_lambda_or_D_bytes_claimed", "native_original_static_head_weights_reused",
                "whole_matched_native_deployment_cost_measured", "old_static_or_global_rankK_floor_reused"):
        require(matched.get(key) is False, "MATCHED_NATIVE_SCOPE_REQUIRED:" + key)
    for key in ("native_backend_cached_D_required", "shared_Bolt_TUNE_selected_rho",
                "all_eight_condition_locks_saved_before_any_new_DEV_prediction",
                "all_conditions_resolved_registry_saved_before_any_DEV_input_load", "original_source_cache_model_and_selection_SHA_unchanged"):
        require(matched.get(key) is True, "MATCHED_NATIVE_SEAL_REQUIRED:" + key)
    require(audit.get("protected_TEST_accessed") is False, "NO_TEST_MATCHED_NATIVE_INDEPENDENT_AUDIT_REQUIRED")
    for key in ("shrinkage_diagnostic", "shrinkage_audit", "producer", "fit_target_audit"):
        item = matched[key]
        require(audit.get(key) == item and record(item["path"]) == item, "EXACT_MATCHED_PARENT_BINDING:" + key)
    code = audit.get("audit_source", {})
    require(Path(code.get("path", "")).name == "audit_matched_shrunk_native_v1.py" and record(code["path"]) == code,
        "EXACT_MATCHED_INDEPENDENT_AUDIT_SOURCE_REQUIRED")
    if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
    from run_matched_shrunk_native_v1 import load_gates as source_gates, resolved_registry_before_development, SELECTED
    shrink, producer, parents = source_gates(matched["shrinkage_diagnostic"]["path"], matched["shrinkage_audit"]["path"])
    require(all(matched[key] == parents[key] for key in parents), "SAME_ACCEPTED_SHRINKAGE_AND_FIT_SOURCE_CHAIN")
    locks = {(dataset, route, str(seed)): matched["condition_selection_locks"][dataset][route][str(seed)]
             for dataset in DATASETS for route in PROJECTIONS for seed in SEEDS}
    resolved = resolved_registry_before_development(locks)
    require(all(value["status"] == "LOCKED" for value in resolved.values()), "ALL_EIGHT_MATCHED_CP_LOCKS_COMPLETE")
    registry = matched["selection_lock"]
    require(record(registry["path"]) == registry and read(registry["path"])["complete_fit_tune_conditions"] == 8,
        "UNCHANGED_ALL_EIGHT_MATCHED_SELECTION_REGISTRY")
    for dataset, (c, k, dev) in DATASETS.items():
        row = matched["datasets"][dataset]
        require(row["channels"] == c and row["K"] == k and row["development_split"] == dev,
            "ORIGINAL_MATCHED_C_K_DEVELOPMENT_CONTRACT")
        for route in PROJECTIONS:
            condition = row["routes"][route]
            label, rho = SELECTED[(dataset, route)]
            require(condition["rho"] == rho and condition["selected_candidate"] == label
                    and condition["old_Bolt_rho_selection_lock"] == shrink["selection_locks"][dataset][route],
                    "UNCHANGED_SHARED_BOLT_SELECTED_RHO")
            for seed in SEEDS:
                entry, lock = condition["native"][str(seed)], resolved[(dataset, route, str(seed))]
                require(entry["status"] == "COMPLETE" and entry["metadata"]["updates"] == 600
                    and entry["metadata"]["selected_update"] == lock["selected_update"]
                    and entry["metadata"]["selected_tune"] == lock["selected_tune"]
                    and entry["metadata"]["canonical_checkpoint_prediction_batch_size"] == 32,
                    "SAME_FRESH_MATCHED_HEAD_CP_AND_CANONICAL_ARITHMETIC")
                require(lock["selected_state_artifact"] == next(item for item in entry["checkpoints"]
                    if item["update"] == lock["selected_update"])["state_artifact"], "SAME_SELECTED_HEAD_STATE_ARTIFACT")
    return matched, shrink, producer, {"matched_native_diagnostic": binding, "matched_native_audit": audit_binding,
        **{key: matched[key] for key in ("shrinkage_diagnostic", "shrinkage_audit", "producer", "fit_target_audit")}}, resolved


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def native_latents(projection, history):
    import numpy as np
    import torch
    pieces = []
    with torch.no_grad():
        for start in range(0, len(history), 32):
            value = torch.from_numpy(np.ascontiguousarray(history[start:start+32]))
            pieces.append(projection.encode(value).detach().cpu())
    return np.ascontiguousarray(torch.cat(pieces).numpy(), dtype=np.float32)


def bolt_predictor(state, frozen_F, rho):
    import numpy as np
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    prior = np.ascontiguousarray(state.decoder.astype(np.float32))
    def predict(history, history_observed):
        z = state.encode(history)
        fitted = fit_shrunk_local_decoder(history, history_observed, z, prior, rho)
        return fitted.forecast(history[:, -1], frozen_F(z), z[:, -1])
    return predict


def native_predictor(projection, prior, rho, model):
    import numpy as np
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    from native_shrunk_decoder_head_v1 import predict_cached_latents
    prior32 = np.ascontiguousarray(prior, dtype=np.float32)
    def predict(history, history_observed):
        z = native_latents(projection, history)
        fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)
        return predict_cached_latents(model, z, history[:, -1], fitted.decoder32, batch_size=32)
    return predict


def physical_backend_precheck(physical, observed, mean, std, values, state, projection, backend,
                              cached, state_arrays, metadata, rho, models=None, predictions=None):
    import numpy as np
    import whole_matched_shrunk_native_cost_v1 as wc
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    from native_shrunk_decoder_head_v1 import predict_cached_latents
    require(backend in ("BOLT_NUMPY", "NATIVE_TORCH32"), "ACTUAL_DECLARED_ENCODER_BACKEND_REQUIRED")
    history = wc.preprocess(physical[:8], observed[:8], mean, std)
    z = state.encode(history) if backend == "BOLT_NUMPY" else native_latents(projection, history)
    prior = np.ascontiguousarray(state.decoder.astype(np.float32))
    fitted = fit_shrunk_local_decoder(history, observed[:8], z, prior, rho)
    require(wc.array_record(prior) == metadata["fit_input_bindings"]["prior_decoder"]
            and metadata["rho"] == rho and metadata["lambda_rule"] == fitted.diagnostics["lambda_rule"]
            and metadata["slope_singular_cutoff_used"] is False, "EXACT_PRIOR_RHO_LAMBDA_RULE_NO_CUTOFF_REQUIRED")
    checks = {"normalized_history": wc.parity(history, values["history"][:8]),
        "normalized_current_C": wc.parity(history[:, -1], values["raw_last"][:8]),
        "encoded_past": wc.parity(z, cached["latent_history"][:8]),
        "decoder_FP32": wc.parity(fitted.decoder32, state_arrays["decoder_deployed_FP32"][:8]),
        "decoder_bias_FP32": wc.parity(fitted.bias32, state_arrays["decoder_bias_FP32"][:8])}
    for name in ("actual_lambda", "local_covariance_trace", "ridge_matrix_condition", "real_stationarity_residual_norm"):
        checks[name] = wc.parity(np.asarray(fitted.diagnostics[name]), np.asarray(metadata[name])[:8])
    decisions = {name: byte_equal(np.asarray(fitted.diagnostics[name]), np.asarray(metadata[name])[:8]) for name in (
        "true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept", "past_metric_valid", "ridge_matrix_condition_valid")}
    decisions["positive_lambda_solver_decision"] = byte_equal(np.asarray(fitted.diagnostics["actual_lambda"]) > 0,
        np.asarray(metadata["actual_lambda"])[:8] > 0)
    mask_equal = byte_equal(observed[:8], values["history_observed"][:8])
    head_checks = {}
    payload = {**fitted.to_npz(), "preprocessed_history": history, "latent_history": z,
        "history_observed": observed[:8].copy(), "origins": values["origins"][:8].copy(), "prior_decoder_FP32": prior}
    if backend == "NATIVE_TORCH32":
        require(set(models or {}) == {str(seed) for seed in SEEDS} and set(predictions or {}) == {str(seed) for seed in SEEDS},
            "BOTH_LOCKED_MATCHED_NATIVE_HEADS_REQUIRED_FOR_PHYSICAL_PRECHECK")
        for seed in SEEDS:
            value = predict_cached_latents(models[str(seed)], z, history[:, -1], fitted.decoder32, batch_size=32)
            head_checks[str(seed)] = wc.parity(value, predictions[str(seed)][:8])
            payload["MATCHED_NATIVE_s" + str(seed) + "_prediction"] = value
    result = {"status": "PASS" if all(value["pass"] for value in checks.values()) and all(decisions.values())
            and mask_equal and all(value["pass"] for value in head_checks.values()) else "FAILED_PRECHECK_PRESERVED",
        "backend": backend, "checks": checks, "exact_decisions": decisions, "history_observed_byte_exact": mask_equal,
        "matched_native_head_prediction_checks": head_checks, "rho": rho, "lambda_rule": fitted.diagnostics["lambda_rule"],
        "origins": values["origins"][:8].tolist(), "fitted_shrunk_metadata": fitted.metadata(),
        "decision_acceptance_by_output_tolerance_only": False, "coefficient_tolerance_loosened": False,
        "scope": "Actual physical first8 -> causal fill/FIT normalize -> declared fixed E backend -> new past shrinkage D; native additionally executes selected matched head; no new fit of E/head or F forward in this precheck"}
    return result, payload


def execute(tag, matched_path, audit_path):
    matched, shrink, producer, bindings, locks = load_gates(matched_path, audit_path)
    failed_cost_binding = verify_failed_cost_origin()
    physical_bindings = {dataset: canonical_physical_bridge(producer, dataset) for dataset in DATASETS}
    require(Path(tag).name == tag and tag not in ("", ".", ".."), "FRESH_SIMPLE_COST_RUN_TAG_REQUIRED")
    output = HERE / "runs" / tag
    require(not output.exists(), "FRESH_MATCHED_COST_OUTPUT_DIRECTORY_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import whole_matched_shrunk_native_cost_v1 as wc
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import matched_readout_contract_v1 as contract
    import cost_readout_v1 as old_cost
    import fixed_decoder_span_v1 as io
    from projection_math_v1 import ProjectionState
    from native_head_v1 import FrozenProjection, state_hash, clone_state
    from native_shrunk_decoder_head_v1 import NativeShrunkForecaster
    from run_fit_target_comparison_v1 import prior_file_records
    require(tuple(wc.ROUTES) == ROUTES and wc.REPEATS == 7 and len(wc.balanced_schedule()) == 7,
        "EXACT_SEVEN_ROUTES_AND_FULL_SEVEN_ROTATIONS")
    output.mkdir(parents=True, exist_ok=False)
    meter = rt.Meter(tag, gpu=True)
    inputs.install_protected_test_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = {"status": "RUNNING", **bindings, "cost": {}, "datasets": {}, "routes": list(ROUTES),
        "source_records": [], "source_snapshots": [], "failed_paths": [], "new_training_updates": 0,
        "prior_failed_cost_attempt": failed_cost_binding,
        "input_schema_failure_identity": record(HERE / "audits/MATCHED_NATIVE_COST_INPUT_SCHEMA_FAILURE_IDENTITY_v1.json"),
        "canonical_physical_bridge_bindings": physical_bindings,
        "repair_addendum": record(HERE / "MATCHED_NATIVE_COST_PHYSICAL_BINDING_REPAIR_KO_v2.md"),
        "protected_TEST_accessed": False, "independent_holdout": False, "new_backbone_training": False,
        "new_encoder_fit": False, "new_rho_head_seed_or_method_selection_in_cost": False,
        "native_deployment_cost_measured": False, "native_heads_loaded_or_timed": False,
        "native_deployment_cost_planned": True, "loaded_native_heads_count": 0,
        "local_ridge_and_implemented_diagnostics_inside_timer": True, "fit_input_binding_hashes_inside_timer": True,
        "local_coefficients_cached_inside_timed_pipeline": False, "native_seed_D_cache_shared_inside_timer": False,
        "source_file_state_metadata_hashes_outside_timed_callback": True,
        "same_Bolt_native_lambda_or_D_bytes_claimed": False, "strict_K_only_compression": False,
        "past_C_condition_metadata_added": True, "static_D_or_global_rankK_oracle_is_local_lower_bound": False,
        "selection_locks": matched["condition_selection_locks"], "all_native_selection_registry": matched["selection_lock"],
        "old_Bolt_rho_selection_locks": shrink["selection_locks"],
        "hard_elapsed_or_cumulative_resource_caps": None, "authorization": record(rt.AUTH), "continuation_manifest": record(rt.MANIFEST),
        "balanced_scope": "All seven circular rotations; each route exactly once at every one of seven positions",
        "timing_helper_copy": {"original": record(HERE / "whole_selected_shrinkage_cost_v1.py"),
            "changes": "New seven-route cohort and seven full rotations; timer/preprocess/parity boundaries retained; actual matched native callbacks added"}}
    failure, guarded, input_records, started = None, [], {}, time.perf_counter()
    projections, native_projections, native_models, immutable = {}, {}, {}, {}
    def checked_load(item):
        value = io.checked_payload(item, meter, input_records)
        result["source_records"].append({"path": item["path"], "sha256": item["sha256"]})
        return value
    def exact(a, b, name):
        require(byte_equal(a, b), "EXACT_MATCHED_COST_SOURCE_BYTES:" + name)
    try:
        meter.protected_check()
        guarded = prior_file_records(matched, shrink, read(audit_path), producer, *bindings.values(),
            failed_cost_binding, physical_bindings, result["repair_addendum"], result["input_schema_failure_identity"])
        source_folder = output / "sources"; source_folder.mkdir()
        for path in (Path(__file__), HERE / "whole_matched_shrunk_native_cost_v1.py", HERE / "native_shrunk_decoder_head_v1.py",
                     HERE / "native_head_v1.py", HERE / "local_decoder_shrinkage_v1.py", HERE / "local_past_decoder_v1.py",
                     HERE / "projection_math_v1.py", HERE / "fixed_decoder_span_v1.py", HERE / "run_matched_shrunk_native_v1.py",
                     HERE / "PLAN_MATCHED_SHRUNK_NATIVE_COST_KO_v1.md",
                     HERE / "MATCHED_NATIVE_COST_PHYSICAL_BINDING_REPAIR_KO_v2.md", HERE / "continuation_runtime_v1.py",
                     BASE / "cost_readout_v1.py", OLD / "run_condition_probes.py", OLD / "prepare_inputs.py", OLD / "matched_readout_contract_v1.py"):
            target = source_folder / path.name
            with path.open("rb") as source_file, target.open("xb") as destination: shutil.copyfileobj(source_file, destination)
            result["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        for dataset, (c, k, dev) in DATASETS.items():
            projections[dataset], native_projections[dataset], native_models[dataset] = {}, {}, {}
            for route in PROJECTIONS:
                state = ProjectionState.from_npz(checked_load(producer["datasets"][dataset]["projections"][route]["state_artifact"]))
                projections[dataset][route] = state
                projection = FrozenProjection(state).cpu().eval().requires_grad_(False)
                native_projections[dataset][route] = projection
                immutable[dataset + ":" + route + ":E"] = state_hash(clone_state(projection))
                native_models[dataset][route] = {}
                for seed in SEEDS:
                    model = NativeShrunkForecaster(k).cpu()
                    locked = locks[(dataset, route, str(seed))]
                    arrays = checked_load(locked["selected_state_artifact"])
                    model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in arrays.items()})
                    model.eval().requires_grad_(False)
                    native_models[dataset][route][str(seed)] = model
                    immutable[dataset + ":" + route + ":s" + str(seed)] = state_hash(clone_state(model))
                    result["loaded_native_heads_count"] += 1
        result["native_and_E_state_hashes_before"] = immutable.copy()
        result["native_heads_loaded_or_timed"] = True
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(producer[key]) for key in ("model_binary_before", "model_config_before"))
        result.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt()
        result["F_before"] = probes.model_hashes(bolt)
        require(result["F_before"] == producer["F_before"] == producer["F_after"], "EXACT_FROZEN_BOLT_DUAL_SIGNATURE")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        bolt.model.eval().requires_grad_(False)
        def frozen_F(history):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(), dtype=np.float32)
        def raw_predict(history, history_observed): return frozen_F(history)
        with threadpool_limits(limits=2):
            for dataset, (c, k, dev) in DATASETS.items():
                result["datasets"][dataset] = {"status": "RUNNING", "channels": c, "K": k, "development_split": dev}
                try:
                    meter.check({"stage": "matched_native_whole_cost", "dataset": dataset})
                    source = producer["datasets"][dataset]
                    values, fit, physical = (checked_load(source["inputs"][dev]),
                        checked_load(source["inputs"]["fit"]), checked_load(physical_bindings[dataset]["canonical_record"]))
                    exact(values["mean"], fit["mean"], "FIT_mean"); exact(values["std"], fit["std"], "FIT_std")
                    exact(physical["origins"], values["origins"], "physical_origins")
                    exact(physical["history_observed"], values["history_observed"], "physical_observed")
                    data_contract = contract.validate_raw_split(dataset, dev, values, fit["mean"], fit["std"])
                    old_cost.validate_physical_inputs(physical["history_physical"], physical["history_observed"],
                        values["history"], fit["mean"], fit["std"], values["origins"])
                    predictors, predictions = {"RAW_BOLT": raw_predict}, {"RAW_BOLT": values["raw_forecast"]}
                    metadata = {"RAW_BOLT": {"source_inputs": source["inputs"][dev], "reference_prediction_array": "raw_forecast",
                        "shared_resident_frozen_F": True, "native_head_loaded_or_timed": False, "local_fit": False}}
                    prechecks = {}
                    for route in PROJECTIONS:
                        condition = matched["datasets"][dataset]["routes"][route]
                        label, rho = condition["selected_candidate"], condition["rho"]
                        backends = shrink["datasets"][dataset]["splits"][dev]["routes"][route]["positive_rho_candidates"][label]["backends"]
                        fixed = checked_load(condition["inputs"][dev]["fixed_inputs_and_controls_artifact"])
                        state, projection = projections[dataset][route], native_projections[dataset][route]
                        base = route.replace("_PINV", "_SHRUNK")
                        prechecks[route] = {}
                        matched_predictions = {str(seed): checked_load(condition["native"][str(seed)]["artifacts"][dev]) for seed in SEEDS}
                        for seed in SEEDS:
                            exact(matched_predictions[str(seed)]["origins"], values["origins"], "matched_head_origins")
                        for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                            selected = backends[backend]
                            cached, arrays = checked_load(selected["artifact"]), checked_load(selected["state_artifact"])
                            for name in ("target", "observed", "raw_last", "origins"):
                                exact(cached[name], values[name], backend + ":" + name)
                            if backend == "NATIVE_TORCH32":
                                exact(cached["latent_history"], fixed["latent_history"], "matched_native_past")
                                exact(arrays["decoder_deployed_FP32"], fixed["decoder"], "matched_native_D")
                            check, check_arrays = physical_backend_precheck(physical["history_physical"], physical["history_observed"],
                                fit["mean"], fit["std"], values, state, projection, backend, cached, arrays,
                                selected["state_metadata"], rho, native_models[dataset][route] if backend == "NATIVE_TORCH32" else None,
                                {seed: entry["prediction"] for seed, entry in matched_predictions.items()} if backend == "NATIVE_TORCH32" else None)
                            prefix = dataset + "_" + route + "_" + backend + "_PHYSICAL_PRECHECK"
                            check["artifact"] = wc.save_outputs(output / (prefix + ".npz"), check_arrays)
                            check.update(source_selected_state=selected["state_artifact"], source_selected_forecasts=selected["artifact"],
                                source_fixed_E=source["projections"][route]["state_artifact"], old_Bolt_rho_selection_lock=condition["old_Bolt_rho_selection_lock"])
                            prechecks[route][backend] = check
                            result["datasets"][dataset]["physical_backend_prechecks"] = prechecks
                            wc.write_json_once(output / (prefix + ".json"), check)
                            require(check["status"] == "PASS", "MATCHED_BACKEND_COEFFICIENT_LAMBDA_DECISION_HEAD_PRECHECK_FAILED:" + route + ":" + backend)
                        bolt_cached = checked_load(backends["BOLT_NUMPY"]["artifact"])
                        bolt_name = base + "_BOLT"
                        predictors[bolt_name], predictions[bolt_name] = bolt_predictor(state, frozen_F, rho), bolt_cached["BOLT_prediction"]
                        metadata[bolt_name] = {"source_fixed_E": source["projections"][route]["state_artifact"],
                            "selected_CPU_reference": backends["BOLT_NUMPY"]["artifact"], "reference_prediction_array": "BOLT_prediction",
                            "backend": "BOLT_NUMPY", "rho": rho, "old_Bolt_rho_selection_lock": condition["old_Bolt_rho_selection_lock"],
                            "native_head_loaded_or_timed": False, "shared_resident_frozen_F": True, "local_fit": True,
                            "cached_local_D_used_for_timing": False, "fit_input_binding_hashes_inside_timer": True}
                        prior = np.ascontiguousarray(state.decoder.astype(np.float32))
                        for seed in SEEDS:
                            name = base + "_NATIVE_s" + str(seed)
                            predictors[name] = native_predictor(projection, prior, rho, native_models[dataset][route][str(seed)])
                            predictions[name] = matched_predictions[str(seed)]["prediction"]
                            metadata[name] = {"source_fixed_E": source["projections"][route]["state_artifact"],
                                "selected_CPU_reference": condition["native"][str(seed)]["artifacts"][dev], "reference_prediction_array": "prediction",
                                "selected_head_state": locks[(dataset, route, str(seed))]["selected_state_artifact"],
                                "selected_head_lock": matched["condition_selection_locks"][dataset][route][str(seed)],
                                "selected_update": locks[(dataset, route, str(seed))]["selected_update"], "native_seed": seed,
                                "backend": "NATIVE_TORCH32", "rho": rho, "head_device": "cpu", "head_threads": 2,
                                "canonical_prediction_batch_size": 32, "native_head_loaded_or_timed": True,
                                "F_forward_in_this_route": False, "local_fit": True, "cached_local_D_used_for_timing": False,
                                "native_seed_D_cache_shared_inside_timer": False, "fit_input_binding_hashes_inside_timer": True,
                                "old_Bolt_rho_selection_lock": condition["old_Bolt_rho_selection_lock"]}
                    result["datasets"][dataset].update(inputs={"fit": source["inputs"]["fit"], dev: source["inputs"][dev]},
                        physical_bridge=source["physical_bridge"], physical_bridge_canonical=physical_bindings[dataset]["canonical_record"],
                        physical_bridge_binding=physical_bindings[dataset], data_contract=data_contract, selected_metadata=metadata)
                    memory = lambda: {"resident_rss_bytes": psutil.Process().memory_info().rss,
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                    result["cost"][dataset] = wc.measure_cost(dataset, physical["history_physical"], physical["history_observed"],
                        fit["mean"], fit["std"], values["origins"], predictors, predictions, metadata,
                        output / ("cost_" + dataset), meter.check, torch.cuda.synchronize, memory)
                    result["datasets"][dataset]["status"] = "COMPLETE"
                    print(json.dumps({"event": "MATCHED_NATIVE_WHOLE_COST_COMPLETE", "dataset": dataset, "cells": 14}), flush=True)
                except Exception as error:
                    failed = {"dataset": dataset, "status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
                    receipt = output / ("cost_" + dataset) / "RESULT.json"
                    if receipt.is_file(): result["cost"][dataset] = read(receipt); failed["preserved_partial_cost_receipt"] = record(receipt)
                    else: failed["whole_cost_status"] = "NOT_MEASURED: failed before full seven-route cohort"
                    result["datasets"][dataset].update(failed); result["failed_paths"].append(failed)
        after = {}
        for dataset in DATASETS:
            for route in PROJECTIONS:
                after[dataset + ":" + route + ":E"] = state_hash(clone_state(native_projections[dataset][route]))
                for seed in SEEDS: after[dataset + ":" + route + ":s" + str(seed)] = state_hash(clone_state(native_models[dataset][route][str(seed)]))
        result["native_and_E_state_hashes_after"] = after
        require(after == immutable, "FROZEN_NATIVE_HEAD_OR_E_CHANGED")
        result["F_after"] = probes.model_hashes(bolt)
        require(result["F_before"] == result["F_after"], "FROZEN_BOLT_CHANGED")
        result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        for item in result["source_snapshots"]:
            require(record(item["path"])["sha256"] == record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_CHANGED")
        for item in guarded + list(input_records.values()): probes.verified_file(item)
        load_gates(matched_path, audit_path); meter.protected_check()
        rows = [row for value in result["cost"].values() for row in value["rows"]]
        samples = sum(value["recorded_timer_samples"] for value in result["cost"].values())
        complete = len(rows) == 28 and samples == 196 and not result["failed_paths"]
        result.update(status="COMPLETE" if complete else "PARTIAL_COMPLETED", expected_cost_cells=28,
            completed_cost_cells=len(rows), recorded_timer_samples=samples, completed_cell_timer_samples=sum(row["repeats"] for row in rows),
            native_deployment_cost_measured=any("_NATIVE_s" in row["route"] for row in rows),
            sample_count_scope="All saved samples include partial failed cohorts; completed-cell sample counts separate",
            source_cache_selection_SHA_unchanged=True, original_selected_forecasts_and_locks_unchanged=True, protected_prior_records=guarded)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["resource_terminal"] = meter.close(result["status"], failure)
        rt.write_new(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "output": str(output / "RESULT.json")}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tag", "matched-diagnostic", "matched-audit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.tag, args.matched_diagnostic, args.matched_audit))
