"""Fresh actual-Y native heads matched to already locked past-only shrunk D[n]."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
ROUTES, SEEDS = ("PCA_PINV", "FIT_TARGET_PINV"), (92601, 92602)
SELECTED = {("robin", "PCA_PINV"): ("RHO_0.1", .1), ("robin", "FIT_TARGET_PINV"): ("RHO_0.1", .1),
    ("beijing_aotizhongxin", "PCA_PINV"): ("RHO_0.1", .1), ("beijing_aotizhongxin", "FIT_TARGET_PINV"): ("RHO_1", 1.)}
SHRINK_SHA = "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532"
SHRINK_AUDIT_SHA = "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9"
PARENT_SHAS = {"producer": "5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98",
    "fit_target_audit": "14399051dc1614f785f4548daa139d92c8769e59fcbee9f38a98126914aedab6"}
SHRINK_COUNTS = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6,
    "reused_control_method_pairs_checked": 36, "reused_control_prediction_arrays_byte_preserved": 72,
    "positive_rho_backend_states_checked": 88, "FIT_TUNE_positive_rho_prediction_arrays_checked": 120,
    "selected_DEV_prediction_arrays_checked": 12, "positive_rho_forecast_arrays_checked": 132,
    "window_fits_checked": 13504, "channel_slots_checked": 187664, "observed_channel_fits_checked": 187664,
    "positive_lambda_solver_calls_checked": 187664, "zero_trace_prior_fallbacks_checked": 0,
    "no_observation_prior_fallbacks_checked": 0, "native_seed_D_reuses_checked": 6752,
    "selection_locks_checked": 4, "score_field_checks": 6960, "protected_prior_file_hashes_checked": 916}
HEAD_SHA = "00c8aa2628d4f9a01697e7a4e0706a856bc8738a10c10db90e1ad7e18fe38476"
PLAN_SHA = "df1d507e83ec2d68d5bfaf16c9511cc5d438ebfce07cb8a3fceec0a6510d9199"


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_gates(shrinkage_path, audit_path):
    """Gate on JSON/source hashes before importing scientific IO or Torch."""
    shrink, audit = read(shrinkage_path), read(audit_path)
    bindings = {"shrinkage_diagnostic": record(shrinkage_path), "shrinkage_audit": record(audit_path)}
    if (shrink.get("status") != "COMPLETE" or audit.get("status") != "PASS"
            or bindings["shrinkage_diagnostic"]["sha256"] != SHRINK_SHA
            or bindings["shrinkage_audit"]["sha256"] != SHRINK_AUDIT_SHA
            or audit.get("diagnostic") != bindings["shrinkage_diagnostic"]):
        raise ValueError("EXACT_COMPLETE_SHRINKAGE_AND_INDEPENDENT_PASS_REQUIRED_BEFORE_SCIENCE")
    if any(audit.get("totals", {}).get(key) != value for key, value in SHRINK_COUNTS.items()):
        raise ValueError("FULL_INDEPENDENT_SHRINKAGE_COVERAGE_REQUIRED")
    expected = {"protected_TEST_accessed": False, "new_E_F_or_native_head_training": False,
        "new_E_F_native_head_forward_runs": 0, "new_backbone_training": False, "new_GPU_execution": False,
        "future_labels_used_for_decoder_fit": False, "native_specific_selection": False,
        "native_uses_same_Bolt_selected_candidate": True, "all_selection_lock_files_saved_before_any_DEV_candidate_compute": True,
        "development_positive_rho_sweep_executed": False, "independent_holdout": False,
        "original_source_cache_model_and_selection_SHA_unchanged": True}
    if any(shrink.get(key) != value for key, value in expected.items()) or shrink.get("failed_paths") != []:
        raise ValueError("COMPLETE_UNCHANGED_PAST_ONLY_SHRINKAGE_SCOPE_REQUIRED")
    if (audit.get("protected_TEST_accessed") is not False or audit.get("new_F_E_head_forward_or_training") is not False
            or audit.get("future_labels_used_for_decoder_fit") is not False or audit.get("DEV_positive_rho_sweep_executed") is not False):
        raise ValueError("INDEPENDENT_PAST_ONLY_NO_TEST_SHRINKAGE_REQUIRED")
    code = audit.get("audit_source", {})
    if Path(code.get("path", "")).name != "audit_local_decoder_shrinkage_v1.py" or record(code["path"]) != code:
        raise ValueError("EXACT_INDEPENDENT_SHRINKAGE_AUDIT_SOURCE_REQUIRED")
    parents = {}
    for name, expected_sha in PARENT_SHAS.items():
        binding = shrink.get(name, {})
        if binding.get("sha256") != expected_sha or audit.get(name) != binding or record(binding["path"]) != binding:
            raise ValueError("EXACT_IMMUTABLE_FIT_TARGET_BINDING_REQUIRED:" + name)
        value = read(binding["path"])
        if value.get("status") != ("COMPLETE" if name == "producer" else "PASS") or value.get("protected_TEST_accessed") is not False:
            raise ValueError("COMPLETE_PASS_NO_TEST_FIT_TARGET_PARENT_REQUIRED")
        parents[name], bindings[name] = value, binding
    if parents["fit_target_audit"].get("producer") != bindings["producer"]:
        raise ValueError("EXACT_FIT_TARGET_INDEPENDENT_PARENT_BINDING_REQUIRED")
    for path, expected_sha in ((HERE / "native_shrunk_decoder_head_v1.py", HEAD_SHA),
                               (HERE / "PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md", PLAN_SHA)):
        if record(path)["sha256"] != expected_sha:
            raise ValueError("ROOT_REVIEWED_MATCHED_NATIVE_SOURCE_OR_PLAN_CHANGED")
    if set(shrink.get("datasets", {})) != set(DATASETS):
        raise ValueError("EXACT_TWO_REGISTERED_SHRINKAGE_DATASETS_REQUIRED")
    for dataset in DATASETS:
        for route in ROUTES:
            binding = shrink["selection_locks"][dataset][route]
            if record(binding["path"]) != binding:
                raise ValueError("OLD_BOLT_RHO_SELECTION_LOCK_CHANGED")
            lock = read(binding["path"])
            label, rho = SELECTED[(dataset, route)]
            if (lock.get("status") != "LOCKED" or lock.get("selected_candidate") != label
                    or lock.get("selected_rho") != rho or lock.get("native_specific_selection") is not False
                    or lock.get("development_or_oracle_scores_used_for_selection") is not False):
                raise ValueError("UNCHANGED_PREVIOUS_BOLT_SELECTED_POSITIVE_RHO_REQUIRED")
    return shrink, parents["producer"], bindings


def byte_equal(first, second):
    import numpy as np
    return first.dtype == second.dtype and first.shape == second.shape and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes()


def fresh_output_path(output_path):
    output = Path(output_path).resolve()
    if output.parent != (HERE / "runs").resolve() or output.exists():
        raise ValueError("FRESH_MATCHED_NATIVE_RUN_DIRECTORY_REQUIRED")
    return output


def resolved_registry_before_development(locks):
    expected = {(dataset, route, str(seed)) for dataset in DATASETS for route in ROUTES for seed in SEEDS}
    if set(locks) != expected:
        raise ValueError("ALL_EIGHT_NATIVE_CONDITION_LOCKS_REQUIRED_BEFORE_DEV")
    resolved = {}
    for key, binding in locks.items():
        if record(binding["path"]) != binding:
            raise ValueError("MATCHED_NATIVE_SELECTION_LOCK_CHANGED")
        value = read(binding["path"])
        if (value.get("dataset"), value.get("route"), value.get("seed")) != key or value.get("status") not in ("LOCKED", "FAILED_PATH_PRESERVED"):
            raise ValueError("EVERY_CONDITION_MUST_BE_SEALED_OR_EXPLICITLY_FAILED")
        if value["status"] == "LOCKED" and (value.get("selected_update") not in (0, 100, 200, 400, 600)
                or value.get("rho_selected_by_native") is not False or value.get("DEV_predictions_computed_before_lock") is not False):
            raise ValueError("VALID_TUNE_ONLY_NATIVE_CP_LOCK_REQUIRED")
        resolved[key] = value
    return resolved


def match_payload(values, candidate, state, first_cache, second_cache, n, c, k):
    import numpy as np
    for name in ("origins", "target", "observed", "raw_last"):
        if not byte_equal(values[name], candidate[name]):
            raise ValueError("SELECTED_SHRINKAGE_AND_RAW_INPUT_BYTES_REQUIRED:" + name)
    for cached in (first_cache, second_cache):
        if not byte_equal(cached["origins"], values["origins"]) or not byte_equal(cached["latent_history"], candidate["latent_history"]):
            raise ValueError("EXACT_SHARED_NATIVE_SEED_PAST_AND_ORIGINS_REQUIRED")
    decoder = state["decoder_deployed_FP32"]
    if (decoder.dtype != np.float32 or decoder.shape != (n, k, c) or not np.isfinite(decoder).all()
            or candidate["latent_history"].dtype != np.float32 or candidate["latent_history"].shape != (n, 512, k)
            or values["history"].shape != (n, 512, c) or values["target"].shape != (n, 48, c)
            or not byte_equal(values["raw_last"], values["history"][:, -1])):
        raise ValueError("REGISTERED_NATIVE_TORCH32_Z_D_CURRENT_AND_TARGET_CONTRACT_REQUIRED")
    return {"latent_history": candidate["latent_history"], "decoder": decoder, "raw_last": values["raw_last"],
        "target": values["target"], "observed": values["observed"], "origins": values["origins"]}


def score_prediction(prediction, values):
    import numpy as np
    from decoder_metric_probe_v1 import masked_scores
    observed = values["observed"]
    complete = np.broadcast_to(observed.all(axis=2, keepdims=True), observed.shape)
    def scores(mask):
        return masked_scores(prediction, values["target"], mask, values["std"]) if mask.any() else {
            "status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
    return {"scores_all_observed": scores(observed), "scores_complete_mask": scores(complete)}


def future_label_perturbation(model, payload, predictor):
    import numpy as np
    tiny = {name: value[:2].copy() for name, value in payload.items()}
    changed = {name: value.copy() for name, value in tiny.items()}
    changed["target"][:] = np.float32(1e6)
    changed["observed"][:] = False
    def forward(values):
        return predictor(model, values["latent_history"], values["raw_last"], values["decoder"], batch_size=32)
    first, second = forward(tiny), forward(changed)
    if not byte_equal(first, second):
        raise ValueError("FROZEN_MATCHED_NATIVE_FUTURE_LABEL_PERTURBATION_FAILED")
    return {"exact_prediction_bytes_equal": True, "windows": len(first), "target_and_future_mask_changed": True,
        "scope": "Fixed audited cached past Z, currentC and D[n]; does not replay upstream raw future or refit E/D",
        "new_encoder_decoder_fit_or_F_forward": False}


def execute(shrinkage_path, audit_path, output_path):
    shrink, producer, bindings = load_gates(shrinkage_path, audit_path)
    output = fresh_output_path(output_path)
    sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import native_shrunk_decoder_head_v1 as native
    import fixed_decoder_span_v1 as io
    from continuation_runtime_v1 import Meter, write_new
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("matched_shrunk_native_" + output.name, gpu=False)
    result = {"status": "RUNNING", **bindings, "datasets": {}, "failed_paths": [], "execution_events": [],
        "protected_TEST_accessed": False, "independent_holdout": False, "new_backbone_training": False,
        "new_E_F_forward_or_training": False, "new_encoder_decoder_fit": False, "new_GPU_execution": False,
        "rho_selection_repeated": False, "rho_selected_by_native": False, "shared_Bolt_TUNE_selected_rho": True,
        "same_Bolt_native_lambda_or_D_bytes_claimed": False, "native_backend_cached_D_required": True,
        "native_original_static_head_weights_reused": False, "native_head_frozen_intervention_not_reoptimized": False,
        "native_training_attempts": 0, "native_training_fits": 0, "completed_training_updates": 0,
        "checkpoint_states_saved": 0, "completed_fit_tune_predictions": 0, "completed_DEV_predictions": 0,
        "completed_selection_locks": 0, "resolved_condition_lock_files": 0, "completed_source_backend_cells": 0,
        "future_label_perturbation_checks": 0, "whole_matched_native_deployment_cost_measured": False,
        "cost_status": "NOT_MEASURED; cached native head CPU training/inference is not whole deployment cost",
        "strict_K_only_compression": False, "past_C_decoder_coefficients_added_to_information": True,
        "old_static_or_global_rankK_floor_reused": False, "full_research_resolution_claimed": False,
        "individual_exact_human_method_approval_claimed": False, "continuation_scope_is_documented_judgment": True,
        "hard_elapsed_or_cumulative_resource_caps": None, "source_snapshots": [], "source_records": [],
        "all_eight_condition_locks_saved_before_any_new_DEV_prediction": False,
        "all_conditions_resolved_registry_saved_before_any_DEV_input_load": False}
    failure, protected, checked, locks, models, contexts = None, [], {}, {}, {}, {}
    started = time.perf_counter()

    def event(kind, **details):
        value = {"sequence": len(result["execution_events"]), "kind": kind, **details}
        result["execution_events"].append(value)
        meter.check(value)

    def failed(error, **details):
        row = {"status": "FAILED_PATH_PRESERVED", **details, "failure": {"type": type(error).__name__, "message": str(error)}}
        result["failed_paths"].append(row)
        return row

    def save(name, arrays):
        meter.check({"stage": "save", "name": name})
        return io.save_npz(output / (name + ".npz"), {key: np.ascontiguousarray(value) for key, value in arrays.items()})

    def load_context(dataset, route, split, count):
        c, k, development, _ = DATASETS[dataset]
        values_record = producer["datasets"][dataset]["inputs"][split]
        values = io.checked_payload(values_record, meter, checked)
        label, rho = SELECTED[(dataset, route)]
        route_row = shrink["datasets"][dataset]["splits"][split]["routes"][route]
        if split == development and route_row.get("selected_candidate") != label:
            raise ValueError("UNCHANGED_PREVIOUS_SELECTED_DEV_RHO_REQUIRED")
        backend = route_row["positive_rho_candidates"][label]["backends"]["NATIVE_TORCH32"]
        if backend.get("status") != "COMPLETE" or backend["state_metadata"].get("rho") != rho or backend.get("native_past_byte_exact_before_D_reuse") is not True:
            raise ValueError("COMPLETE_SELECTED_NATIVE_BACKEND_D_REQUIRED")
        state = io.checked_payload(backend["state_artifact"], meter, checked)
        cached = io.checked_payload(backend["artifact"], meter, checked)
        originals = [io.checked_payload(backend["cached_method_sources"]["NATIVE_s" + str(seed)], meter, checked) for seed in SEEDS]
        payload = match_payload(values, cached, state, *originals, count, c, k)
        bolt = route_row["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"]
        bolt_saved = io.checked_payload(bolt["artifact"], meter, checked)
        for name in ("origins", "target", "observed", "raw_last"):
            if not byte_equal(values[name], bolt_saved[name]): raise ValueError("BOLT_NATIVE_COMPARISON_RAW_INPUTS_DIFFER:" + name)
        past_bindings = backend["state_metadata"]["fit_input_bindings"]
        if any(past_bindings[name]["sha256"] != io.array_sha(value) for name, value in (
                ("history", values["history"]), ("history_observed", values["history_observed"]), ("latent_history", payload["latent_history"]))):
            raise ValueError("D_FIT_BINDING_NOT_THIS_PAST_OR_NATIVE_BACKEND")
        prefix = dataset + "_" + route + "_" + split
        controls = {"RAW_BOLT": values["raw_forecast"], "SELECTED_BOLT": bolt_saved["BOLT_prediction"],
            **{"OLD_FROZEN_NATIVE_s" + str(seed): cached["NATIVE_s" + str(seed) + "_prediction"] for seed in SEEDS}}
        archived = {**payload, "mean": values["mean"], "std": values["std"],
            **{name + "_prediction": prediction for name, prediction in controls.items()}}
        artifact = save(prefix + "_FIXED_INPUTS_AND_CONTROLS", archived)
        row = {"status": "COMPLETE", "rho": rho, "selected_candidate": label, "backend": "NATIVE_TORCH32",
            "input_source": values_record, "selected_native_state_source": backend["state_artifact"],
            "selected_native_forecasts_source": backend["artifact"], "selected_Bolt_forecasts_source": bolt["artifact"],
            "original_native_cached_sources": backend["cached_method_sources"], "fixed_inputs_and_controls_artifact": artifact,
            "seed_cached_past_byte_identical_and_D_shared": True, "D_refit": False, "control_predictions_copied_byte_exact": True,
            "controls": {name: {**score_prediction(prediction, values), "prediction_member": name + "_prediction"} for name, prediction in controls.items()}}
        result["datasets"][dataset]["routes"][route]["inputs"][split] = row
        result["completed_source_backend_cells"] += 1
        return {"payload": payload, "values": values, "binding": row}

    def train_one(dataset, route, seed):
        key = (dataset, route, str(seed))
        entry = {"status": "TRAINING", "checkpoints": [], "scores": {}, "artifacts": {}, "learning_curve": [], "progress_events": []}
        result["datasets"][dataset]["routes"][route]["native"][str(seed)] = entry
        fit, tune = contexts[(dataset, route, "fit")], contexts[(dataset, route, "tune")]
        cp_predictions = {}
        prefix = dataset + "_" + route + "_MATCHED_NATIVE_s" + str(seed)
        def checkpoint(update, state, score, prediction):
            state_record = save(prefix + "_u" + str(update) + "_STATE", {name: value.numpy().copy() for name, value in state.items()})
            predicted = save(prefix + "_u" + str(update) + "_TUNE", {"prediction": prediction, "origins": tune["payload"]["origins"]})
            entry["checkpoints"].append({"update": update, "tune": score, "state_sha256": native.state_hash(state),
                "state_artifact": state_record, "tune_prediction_artifact": predicted, "canonical_prediction_batch_size": 32})
            cp_predictions[update] = prediction.copy()
            result["checkpoint_states_saved"] += 1
        def progress(value):
            entry["progress_events"].append(value)
            if value["stage"] == "training": entry["learning_curve"].append({name: value[name] for name in ("update", "loss", "gradient_norm_before_clip")})
            meter.check({"dataset": dataset, "route": route, "seed": seed, **value})
        result["native_training_attempts"] += 1
        event("NATIVE_TRAIN_BEGIN", dataset=dataset, route=route, seed=str(seed))
        trained = native.train_native_shrunk(fit["payload"], tune["payload"], seed, progress, checkpoint, updates=600, batch_size=4)
        result["native_training_fits"] += 1
        result["completed_training_updates"] += len(trained["learning_curve"])
        model = trained["model"]
        entry.update(metadata=trained["metadata"], learning_curve=trained["learning_curve"])
        for split in ("fit", "tune"):
            context = contexts[(dataset, route, split)]; payload = context["payload"]
            prediction, future = native.predict_cached_components(model, payload["latent_history"], payload["raw_last"], payload["decoder"], batch_size=32)
            if split == "tune" and not byte_equal(prediction, cp_predictions[entry["metadata"]["selected_update"]]):
                raise ValueError("SELECTED_TUNE_FORECAST_NOT_EXACT_CP_SCORED_BATCH32_ARRAY")
            entry["scores"][split] = score_prediction(prediction, context["values"])
            entry["artifacts"][split] = save(prefix + "_" + split, {"prediction": prediction, "latent_future": future, "origins": payload["origins"]})
            result["completed_fit_tune_predictions"] += 1
        selected = next(item for item in entry["checkpoints"] if item["update"] == entry["metadata"]["selected_update"])
        lock = {"status": "LOCKED", "dataset": dataset, "route": route, "seed": str(seed),
            "selected_update": entry["metadata"]["selected_update"], "selected_state_artifact": selected["state_artifact"],
            "selected_CP_TUNE_prediction_artifact": selected["tune_prediction_artifact"], "selected_tune": selected["tune"],
            "rho": SELECTED[(dataset, route)][1], "old_Bolt_rho_selection_lock": shrink["selection_locks"][dataset][route],
            "rho_selected_by_native": False, "DEV_predictions_computed_before_lock": False,
            "checkpoint_selection": "Full TUNE observed normalized C-macro MSE; exact tie earliest update",
            "sealed_unix": time.time()}
        entry["status"] = "COMPLETE_FIT_TUNE_SEALED"
        models[key] = model
        result["completed_selection_locks"] += 1
        event("NATIVE_TRAIN_COMPLETE", dataset=dataset, route=route, seed=str(seed), selected_update=lock["selected_update"])
        return lock

    try:
        result["protected_guard_paths"] = install_protected_test_guard()
        meter.protected_check()
        source_paths = [Path(__file__), HERE / "native_shrunk_decoder_head_v1.py", HERE / "native_head_v1.py",
            HERE / "PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md", HERE / "fixed_decoder_span_v1.py", HERE / "decoder_metric_probe_v1.py",
            HERE / "continuation_runtime_v1.py", HERE / "run_fit_target_comparison_v1.py", OLD / "prepare_inputs.py"]
        snapshots = output / "sources"; snapshots.mkdir()
        for path in source_paths:
            source_record = record(path); target = snapshots / path.name
            with path.open("rb") as source_file, target.open("xb") as destination: shutil.copyfileobj(source_file, destination)
            result["source_records"].append(source_record)
            result["source_snapshots"].append({**source_record, "snapshot_path": str(target)})
        protected = prior_file_records(shrink, read(audit_path), producer, *bindings.values(), *result["source_records"])
        scope = {"status": "CONTINUING_GOAL_MATCHED_PAST_DECODER_NATIVE_TRAINING_JUDGMENT", "bindings": bindings,
            "plan": record(HERE / "PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md"), "individual_exact_human_method_approval_claimed": False,
            "new_native_fits": 8, "updates_each": 600, "CP_each": [0, 100, 200, 400, 600],
            "seeds": list(SEEDS), "new_backbone_or_encoder_decoder_training": False,
            "decoder_and_rho_selected_already_by_Bolt_TUNE": True, "native_specific_rho_selection": False,
            "NativeTorch32_backend_specific_selected_D_and_cached_past": True,
            "head_architecture": "K independent affine512-to48; same original zero-init actual-C masked-macro objective",
            "all_eight_resolved_locks_before_DEV": True, "strict_K_only_interface": False,
            "whole_matched_native_cost_measured": False, "independent_holdout": False,
            "old_static_or_global_rankK_floor_reused": False, "F_necessity_proven": False}
        write_new(output / "SCOPE.json", scope); result["scope"] = record(output / "SCOPE.json")
        with threadpool_limits(limits=2):
            for dataset, (c, k, development, dev_count) in DATASETS.items():
                result["datasets"][dataset] = {"channels": c, "K": k, "development_split": development,
                    "routes": {route: {"rho": SELECTED[(dataset, route)][1], "selected_candidate": SELECTED[(dataset, route)][0],
                        "old_Bolt_rho_selection_lock": shrink["selection_locks"][dataset][route], "inputs": {}, "native": {}} for route in ROUTES}}
                for route in ROUTES:
                    preparation_error = None
                    try:
                        for split, count in (("fit", 256), ("tune", 64)):
                            event("FIT_TUNE_INPUT_LOAD", dataset=dataset, route=route, split=split)
                            contexts[(dataset, route, split)] = load_context(dataset, route, split, count)
                    except Exception as error:
                        preparation_error = error
                        failed(error, dataset=dataset, route=route, stage="FIT_TUNE_INPUT_BINDING")
                    for seed in SEEDS:
                        try:
                            if preparation_error is not None: raise preparation_error
                            lock = train_one(dataset, route, seed)
                        except Exception as error:
                            lock = failed(error, dataset=dataset, route=route, seed=str(seed), stage="NATIVE_FIT_TUNE")
                            result["datasets"][dataset]["routes"][route]["native"].setdefault(str(seed), {}).update(
                                status="FAILED_PATH_PRESERVED", failure=lock["failure"])
                        lock_path = output / (dataset + "_" + route + "_s" + str(seed) + "_SELECTION_LOCK.json")
                        write_new(lock_path, lock); locks[(dataset, route, str(seed))] = record(lock_path)
                        result["resolved_condition_lock_files"] += 1
                        event("CONDITION_SELECTION_LOCK_SAVED", dataset=dataset, route=route, seed=str(seed), status=lock["status"])
            resolved = resolved_registry_before_development(locks)
            registry = {"status": "ALL_EIGHT_CONDITIONS_RESOLVED_BEFORE_ANY_NEW_DEV_INPUT_OR_PREDICTION",
                "conditions": [{"dataset": key[0], "route": key[1], "seed": key[2], "status": resolved[key]["status"],
                    "lock": locks[key]} for key in sorted(locks)], "failed_paths_at_seal": list(result["failed_paths"]),
                "sealed_unix": time.time(), "complete_fit_tune_conditions": len(models), "expected_conditions": 8}
            write_new(output / "SELECTION_LOCK.json", registry); result["selection_lock"] = record(output / "SELECTION_LOCK.json")
            result["condition_selection_locks"] = {dataset: {route: {str(seed): locks[(dataset, route, str(seed))] for seed in SEEDS}
                for route in ROUTES} for dataset in DATASETS}
            result["all_eight_condition_locks_saved_before_any_new_DEV_prediction"] = True
            result["all_conditions_resolved_registry_saved_before_any_DEV_input_load"] = True
            event("ALL_EIGHT_CONDITION_LOCKS_AND_REGISTRY_SAVED", complete_conditions=len(models), failed_conditions=8-len(models))
            for dataset, (c, k, development, dev_count) in DATASETS.items():
                for route in ROUTES:
                    ready = [seed for seed in SEEDS if (dataset, route, str(seed)) in models]
                    if not ready: continue
                    resolved_registry_before_development(locks)
                    if record(result["selection_lock"]["path"]) != result["selection_lock"]: raise ValueError("ALL_CONDITION_REGISTRY_CHANGED_BEFORE_DEV")
                    try:
                        event("DEV_INPUT_LOAD_AFTER_ALL_LOCKS", dataset=dataset, route=route, split=development)
                        context = load_context(dataset, route, development, dev_count)
                    except Exception as error:
                        failed(error, dataset=dataset, route=route, stage="DEV_INPUT_BINDING")
                        continue
                    payload = context["payload"]
                    for seed in ready:
                        entry = result["datasets"][dataset]["routes"][route]["native"][str(seed)]
                        try:
                            event("DEV_NATIVE_PREDICTION_AFTER_ALL_LOCKS", dataset=dataset, route=route, seed=str(seed))
                            model = models[(dataset, route, str(seed))]
                            prediction, future = native.predict_cached_components(model, payload["latent_history"], payload["raw_last"], payload["decoder"], batch_size=32)
                            entry["scores"][development] = score_prediction(prediction, context["values"])
                            entry["artifacts"][development] = save(dataset + "_" + route + "_MATCHED_NATIVE_s" + str(seed) + "_" + development,
                                {"prediction": prediction, "latent_future": future, "origins": payload["origins"]})
                            result["completed_DEV_predictions"] += 1
                            entry["future_label_perturbation"] = future_label_perturbation(model, payload, native.predict_cached_latents)
                            result["future_label_perturbation_checks"] += 1
                            entry["status"] = "COMPLETE"
                        except Exception as error:
                            entry.update(status="PARTIAL_COMPLETED", development_failure=failed(error, dataset=dataset, route=route, seed=str(seed), stage="DEV_NATIVE_PREDICTION"))
        for binding in protected + list(checked.values()) + list(locks.values()) + [result["selection_lock"]]:
            if record(binding["path"])["sha256"] != binding["sha256"]: raise ValueError("PROTECTED_SOURCE_CACHE_OR_SELECTION_CHANGED")
        for binding in result["source_snapshots"]:
            if record(binding["snapshot_path"])["sha256"] != binding["sha256"]: raise ValueError("EXECUTED_SOURCE_SNAPSHOT_CHANGED")
        meter.protected_check()
        result.update(protected_prior_records=protected, input_records=list(checked.values()), original_source_cache_model_and_selection_SHA_unchanged=True)
        expected = {"native_training_fits": 8, "completed_training_updates": 4800, "checkpoint_states_saved": 40,
            "completed_fit_tune_predictions": 16, "completed_DEV_predictions": 8, "completed_selection_locks": 8,
            "resolved_condition_lock_files": 8, "completed_source_backend_cells": 12, "future_label_perturbation_checks": 8}
        result["completeness"] = {"expected": expected, "actual": {key: result[key] for key in expected}}
        complete = not result["failed_paths"] and result["completeness"]["actual"] == expected
        result.update(status="COMPLETE" if complete else "PARTIAL_COMPLETED", full_scope_complete_claimed=complete)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        result.update(status="FAILED_PRESERVED", failure=failure, full_scope_complete_claimed=False)
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["resource_terminal"] = meter.close(result["status"], failure)
        write_new(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "output": str(output / "RESULT.json"),
        "native_fits": result["native_training_fits"], "CP_states": result["checkpoint_states_saved"],
        "DEV_predictions": result["completed_DEV_predictions"], "failure": failure}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shrinkage-diagnostic", "shrinkage-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.shrinkage_diagnostic, args.shrinkage_audit, args.output))
