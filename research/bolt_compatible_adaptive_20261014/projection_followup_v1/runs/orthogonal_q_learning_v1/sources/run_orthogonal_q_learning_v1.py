"""Gated six-parameter orthogonal basis learning through the unchanged frozen median Bolt."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import time

import run_orthogonal_basis_controls_v1 as fixed

old, HERE, ROOT, BASE, OLD, require = fixed.old, fixed.HERE, fixed.ROOT, fixed.BASE, fixed.OLD, fixed.require
STATES, SEEDS, LAGS = fixed.STATES, (92601, 92602), fixed.LAGS
CONDITIONS = tuple((state, seed) for state in STATES for seed in SEEDS)
FIXED_SOURCE_SHA = "1c0676b75b2baaa1d937bade5e1893d06ba46c01f7c73d48953144e9d09d6daa"
FIXED_RESULT_SHA = "67661b1468384aed58d57c8f3c84e3a04d8f6e806cbfb4dbdb92d0f6c34cf42b"
FIXED_AUDIT_SHA = "156e7513b200215f574b511828428c3e11c5b8a79ab2f79c85448a0eb87b4977"
FIXED_AUDIT_SOURCE_SHA = "4d16612c65b9b1ee3e33a95639b87992b73ce6aa3991dea4c922f58ae17d9f43"
FIXED_AUDIT_COUNTS = {"fixed_control_conditions_checked": 15, "FIT_TUNE_prediction_evaluations_checked": 30, "DEV_prediction_evaluations_checked": 15, "control_locks_checked": 15, "produced_NPZ_artifacts_checked": 105, "actual_F_calls_registered": 165, "null_control_split_cells_checked": 18, "null_C_future_byte_equalities_observed": 36, "analytic_transports_checked": 45, "independent_forward_arithmetic_cells_checked": 45, "score_fields_checked": 4320, "commutator_summaries_checked": 180, "frozen_E_D_center_sources_checked": 3, "protected_prior_file_hashes_checked": 1927}
GRADIENT_CLOSURE_ATOL, GRADIENT_CLOSURE_RTOL = 1e-6, 1e-4
CP_SCORE_LINK_ATOL, CP_SCORE_LINK_RTOL = 1e-12, 1e-12


def load_gates(args):
    require(all(isinstance(v, str) and len(v) == 64 for v in (FIXED_RESULT_SHA, FIXED_AUDIT_SHA, FIXED_AUDIT_SOURCE_SHA)) and isinstance(FIXED_AUDIT_COUNTS, dict) and bool(FIXED_AUDIT_COUNTS), "FIXED_Q_INDEPENDENT_AUDIT_NOT_YET_REGISTERED")
    require(old.record(fixed.__file__)["sha256"] == FIXED_SOURCE_SHA, "EXECUTED_FIXED_Q_CONTROLLER_UNCHANGED")
    calibration, native, science, full, refs = fixed.load_gates(args)
    binding, audit_binding = old.record(args.fixed_controls_result), old.record(args.fixed_controls_audit)
    require(binding["sha256"] == FIXED_RESULT_SHA and audit_binding["sha256"] == FIXED_AUDIT_SHA, "EXACT_FIXED_Q_RESULT_AND_AUDIT")
    controls, audit = old.read(binding["path"]), old.read(audit_binding["path"])
    require(controls["status"] == "COMPLETE" and audit["status"] == "PASS" and controls["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False, "ACCEPTED_FIXED_Q_CONTROLS_NO_TEST")
    old.bound(audit["diagnostic"], binding); old.bound(audit["audit_source"])
    require(audit["audit_source"]["sha256"] == FIXED_AUDIT_SOURCE_SHA and all(type(audit["totals"].get(k)) is int and audit["totals"][k] == v for k, v in FIXED_AUDIT_COUNTS.items()), "EXACT_FIXED_Q_AUDITOR_SOURCE_COVERAGE")
    old.bound(controls["controller_source"], {"path": fixed.__file__, "sha256": FIXED_SOURCE_SHA})
    old.bound(controls["module_source"], {"path": str(HERE/"orthogonal_latent_basis_v1.py"), "sha256": fixed.MODULE_SHA})
    require(controls["controls_completed"] == controls["resolved_control_locks"] == 15 and controls["FIT_TUNE_evaluations_completed"] == 30 and controls["DEV_evaluations_completed"] == 15 and controls["all_15_locks_before_DEV"] is True and not controls["failed_paths"], "ALL_FIFTEEN_FIXED_CONTROLS_RESOLVED")
    for key, item in refs.items(): old.bound(controls[key], item)
    require(controls["F_before"] == controls["F_after"] == calibration["F_parent_signatures_reused"]["before"] == calibration["F_parent_signatures_reused"]["after"], "SAME_FIXED_CONTROL_AND_CALIBRATION_F")
    refs.update(fixed_controls_diagnostic=binding, fixed_controls_audit=audit_binding, fixed_controls_auditor_source=audit["audit_source"], fixed_control_registry=controls["control_registry"])
    return calibration, native, science, full, controls, audit, refs


def training_status(model, bolt):
    model.frozen_status()
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    require(names == ["theta"] and model.theta.numel() == 6 and model.learn_Q and not model.identity and not model.center.requires_grad, "ONLY_SIX_THETA_NO_ZERO_THETA_BYPASS")
    require(hasattr(bolt, "model") and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()), "F_FROZEN_AND_GRAD_NONE")
    return {"trainable_parameter_names": names, "theta_parameters": 6, "E_requires_grad": False, "D_requires_grad": False, "E_grad_is_None": True, "D_grad_is_None": True, "center_requires_grad": False, "F_requires_grad": False, "F_grad_is_None": True, "gradient_clipping_parameter_scope": "THETA_ONLY", "learned_zero_theta_bypass": False}


def checked_basis(model):
    value = model.basis_diagnostics()
    require(value["finite"] and value["orthogonality_max_abs"] <= 1e-5 and abs(value["determinant"]-1.) <= 1e-5, "VALID_FP32_SPECIAL_ORTHOGONAL_Q")
    return value


def check_score_link(score, entry):
    require(all(math.isclose(score[key], entry["scores_all_observed"]["normalized"][key], abs_tol=CP_SCORE_LINK_ATOL, rel_tol=CP_SCORE_LINK_RTOL) for key in ("mse", "mae")), "OWN_TUNE_SCORE_MATCHES_SAVED_CANONICAL_OUTPUT")


def first_batch_gradients(model, loss, rotated, F_future, future):
    import numpy as np
    import torch
    require(rotated.requires_grad and F_future.requires_grad, "F_INPUT_AUTOGRAD_PATH_REQUIRED")
    full = torch.autograd.grad(loss, model.theta, retain_graph=True)[0]
    input_gradient, future_gradient = torch.autograd.grad(loss, (rotated, future), retain_graph=True)
    through_input = torch.autograd.grad(rotated, model.theta, grad_outputs=input_gradient.detach(), retain_graph=True)[0]
    inverse_only = F_future.detach() @ model.basis().T
    through_inverse = torch.autograd.grad(inverse_only, model.theta, grad_outputs=future_gradient.detach(), retain_graph=True)[0]
    residual = full-through_input-through_inverse
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in (("theta_gradient_full", full), ("theta_gradient_via_F_input", through_input), ("theta_gradient_via_inverse_Q", through_inverse), ("theta_gradient_closure_residual", residual), ("rotated_F_input_gradient", input_gradient), ("unrotated_future_gradient", future_gradient), ("theta_FP32", model.theta), ("Q_FP32", model.basis()))}
    finite = all(np.isfinite(value).all() for value in arrays.values())
    closure = finite and bool(torch.allclose(full, through_input+through_inverse, atol=GRADIENT_CLOSURE_ATOL, rtol=GRADIENT_CLOSURE_RTOL))
    norm = lambda value: float(torch.linalg.vector_norm(value.detach().double()).cpu())
    proof = {"status": "PASS" if closure else "FAILED_GRADIENT_CLOSURE_PRESERVED", "finite": finite, "gradient_closure_pass": closure, "gradient_closure_atol": GRADIENT_CLOSURE_ATOL, "gradient_closure_rtol": GRADIENT_CLOSURE_RTOL, "full_theta_gradient_norm": norm(full), "theta_F_input_chain_norm": norm(through_input), "theta_inverse_Q_contribution_norm": norm(through_inverse), "rotated_F_input_gradient_norm": norm(input_gradient), "closure_residual_norm": norm(residual), "F_input_path_nonzero_observed": norm(through_input) > 0., "nonzero_total_theta_gradient_alone_proves_F_input_path": False, "zero_gradient_does_not_prove_missing_autograd_path": True, "optimizer_updates_in_probe": 0, "probe_reuses_first_training_forward": True}
    return proof, arrays


def train_Q(model, bolt, fit, tune, cached_tune, seed, check, checkpoint, gradient_callback, *, updates=600):
    import numpy as np
    import torch
    from linear_e_adaptation_v1 import global_fit_macro_mse
    from native_head_v1 import clone_state, score_predictions, select_checkpoint, state_hash
    require(type(updates) is int and 0 <= updates <= 600, "ZERO_TO_SIX_HUNDRED_UPDATES")
    flags = training_status(model, bolt); require(bool((model.theta.detach() == 0).all()), "FRESH_ZERO_THETA_REQUIRED")
    initial = old.state_arrays(model); tensors = {key: torch.as_tensor(fit[key], device=model.E.device) for key in ("history", "target", "observed")}
    counts = tensors["observed"].sum((0, 1)); require(bool((counts > 0).any()), "OBSERVED_FIT_CHANNEL_REQUIRED")
    order = np.random.default_rng(seed).integers(0, len(fit["history"]), size=(updates, 4))
    optimizer = torch.optim.AdamW([model.theta], lr=.001, weight_decay=0.)
    cp_updates = sorted({0, updates, *(n for n in (100, 200, 400, 600) if n <= updates)})
    checkpoints, curve, probe = [], [], None
    for update in range(updates+1):
        if update in cp_updates:
            outputs, _ = fixed.infer_control(model, bolt, tune["history"], cached_tune, check)
            score = score_predictions(outputs["prediction"], tune["target"], tune["observed"]); state = clone_state(model); digest = state_hash(state)
            require(all(old.byte_equal(state[k].numpy(), initial[k]) for k in ("E", "D", "center", "fixed_Q")), "FROZEN_E_D_CENTER_AT_EVERY_CP")
            metadata = checkpoint(update, state, score, outputs)
            require(state_hash(state) == digest and metadata["status"] == "COMPLETE", "IMMUTABLE_CP_STATE_AND_VALID_DEPLOYMENT")
            checkpoints.append({"update": update, "state": state, "state_sha256": digest, "tune": score, "artifact_metadata": metadata})
        if update == updates: break
        ids = torch.as_tensor(order[update], dtype=torch.long, device=model.E.device); optimizer.zero_grad(set_to_none=True); capture = {}
        def captured_F(rotated):
            capture["rotated"] = rotated; capture["F_future"] = bolt(rotated); return capture["F_future"]
        prediction, _, future = model(tensors["history"][ids], captured_F, return_components=True)
        loss = global_fit_macro_mse(prediction, tensors["target"][ids], tensors["observed"][ids], counts, len(fit["history"]))
        if update == 0:
            probe, arrays = first_batch_gradients(model, loss, capture["rotated"], capture["F_future"], future)
            arrays["FIT_window_indices"] = order[0].copy(); probe["artifact"] = gradient_callback(probe, arrays)
            require(probe["status"] == "PASS", "FIRST_BATCH_F_INPUT_AND_INVERSE_Q_GRADIENT_CLOSURE")
        loss.backward(); require(model.theta.grad is not None and bool(torch.isfinite(model.theta.grad).all()), "FINITE_THETA_GRADIENT")
        gradient = model.theta.grad.detach().cpu().tolist(); norm = torch.nn.utils.clip_grad_norm_([model.theta], 1.)
        require(bool(torch.isfinite(norm)), "FINITE_CLIP_GRADIENT_NORM"); optimizer.step(); check({"stage": "OPTIMIZER_STEP_EXECUTED", "update": update+1})
        require(bool(torch.isfinite(model.theta).all()), "FINITE_THETA_AFTER_OPTIMIZER")
        flags = training_status(model, bolt)
        event = {"stage": "TRAINING_UPDATE", "update": update+1, "loss": float(loss.detach()), "theta_gradient_before_clip": gradient, "theta_gradient_norm_before_clip": float(norm.detach()), "global_gradient_norm_before_clip": float(norm.detach()), "clipped": bool(norm > 1.), "theta_after_optimizer_step": model.theta.detach().cpu().tolist(), **flags}
        curve.append(event); check(event)
    selected = select_checkpoint(checkpoints); model.load_state_dict(selected["state"]); model.zero_grad(set_to_none=True); model.eval(); training_status(model, bolt)
    require(all(old.byte_equal(old.state_arrays(model)[k], initial[k]) for k in ("E", "D", "center", "fixed_Q")), "RESTORED_SELECTION_PRESERVES_FROZEN_BYTES")
    return {"model": model, "checkpoints": checkpoints, "learning_curve": curve, "training_updates_completed": updates, "selected_update": selected["update"], "selected_tune": selected["tune"], "selected_state_sha256": selected["state_sha256"], "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "FIT_channel_observed_counts": counts.cpu().tolist(), "global_clipping_fraction": sum(v["clipped"] for v in curve)/max(1, len(curve)), "loss": "N_fit/B/C_active * sum_c SSE_batch_c/N_FIT_c; fixed entire FIT observed counts", "first_batch_gradient_probe": probe, "frozen_state_checkpoint_byte_checks": len(checkpoints), "post_optimizer_frozen_flag_checks": updates, "frozen_bytes_checked_at_CP_and_final_only": True, "independent_per_step_backward_optimizer_or_frozen_bytes_replayed": False, "final_frozen_status": flags}


def evaluate_arrays(outputs, raw, cached, frozen, Q, *, cp0=False):
    import numpy as np
    from causal_lag_decoder_v1 import predict
    from decoder_metric_probe_v1 import masked_scores
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in outputs.values()), "FINITE_ACTUAL_FP32_Q_COMPONENTS")
    independent_future = np.ascontiguousarray(outputs["F_rotated_future"]@Q.T, dtype=np.float32)
    independent_prediction = predict(outputs["old_past"], independent_future, raw["raw_last"], frozen["D"], LAGS)
    checks = {"old_past_vs_cached": old.parity(outputs["old_past"], cached["latent_history"]), "NumPy_fresh_E": old.parity(outputs["old_past"], (raw["history"]-frozen["center"])@frozen["E"]), "NumPy_rotated_past": old.parity(outputs["rotated_past"], outputs["old_past"]@Q), "NumPy_inverse_Q_future": old.parity(outputs["unrotated_future"], independent_future), "NumPy_original_lag_D": old.parity(outputs["prediction"], independent_prediction), "analytic_transport_future": old.parity(outputs["transport_future"], cached["latent_future"]), "analytic_transport_C": old.parity(outputs["transport_prediction"], cached["prediction"])}
    if cp0:
        checks.update(CP0_future_vs_cached=old.parity(outputs["unrotated_future"], cached["latent_future"]), CP0_C_vs_cached=old.parity(outputs["prediction"], cached["prediction"]))
    complete = np.broadcast_to(raw["observed"].all(2, keepdims=True), raw["observed"].shape)
    def scores(value): return {"all_observed": masked_scores(value, raw["target"], raw["observed"], raw["std"]), "complete_C": masked_scores(value, raw["target"], complete, raw["std"]) if complete.any() else {"status": "NO_COMPLETE_C"}}
    actual = scores(outputs["prediction"])
    row = {"status": "COMPLETE" if all(p["pass"] for p in checks.values()) else "FAILED_PARITY_PRESERVED", "checks": checks, "scores_all_observed": actual["all_observed"], "scores_complete_mask": actual["complete_C"], "baseline_Bolt_scores": scores(cached["prediction"]), "RAW_BOLT_scores": scores(raw["raw_forecast"]), "latent_commutator": fixed.difference(outputs["unrotated_future"], cached["latent_future"]), "C_commutator": fixed.difference(outputs["prediction"], cached["prediction"]), "CP0_parity_required": cp0, "learned_zero_theta_bypass": False, "FIT_future_is_training_response_only": True, "future_truth_is_forward_input": False, "complete_target_positions": int(complete[:, :, 0].sum()), "canonical_batch_size": 32}
    return row, {"independent_NumPy_future": independent_future, "independent_NumPy_prediction": independent_prediction}


def resolve_all(result, run, seal, registry):
    resolved, locks = {}, []
    for state, seed in CONDITIONS:
        try: row = run(state, seed)
        except Exception as error: row = {"status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
        row["selection_lock"] = seal(state, seed, row); result["states"][state]["learning"][str(seed)] = row
        locks.append({"state": state, "seed": seed, "status": row["status"], "selection_lock": row["selection_lock"]})
        result["training_updates_completed"] += row.get("training_updates_completed", 0); result["checkpoint_states_saved"] += row.get("checkpoint_count", 0)
        if row["status"] == "COMPLETE": resolved[state, seed] = row; result["Q_training_fits"] += 1
        else: result["failed_paths"].append({"state": state, "seed": seed, "stage": "FIT_TUNE", "failure": row.get("failure")})
    result["selection_registry"] = registry(locks); result.update(resolved_selection_locks=len(locks), all_6_locks_before_DEV=True)
    return resolved


def infer_development(result, resolved, infer):
    require(result["all_6_locks_before_DEV"] and result["resolved_selection_locks"] == 6, "ALL_SIX_LOCKS_BEFORE_ANY_DEV")
    for (state, seed), row in resolved.items():
        try:
            row["DEV"] = infer(state, seed, row); require(row["DEV"]["status"] == "COMPLETE", "VALID_SELECTED_DEV_DEPLOYMENT"); result["DEV_evaluations_completed"] += 1
        except Exception as error: result["failed_paths"].append({"state": state, "seed": seed, "stage": "DEV", "failure": {"type": type(error).__name__, "message": str(error)}})


def execute(args):
    calibration, native, science, full, controls, controls_audit, refs = load_gates(args); output = old.fresh_output(args.tag); sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import orthogonal_latent_basis_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from native_head_v1 import state_hash
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(old.record(module.__file__)["sha256"] == fixed.MODULE_SHA, "EXECUTED_ORTHOGONAL_MODULE_UNCHANGED")
    output.mkdir(parents=True, exist_ok=False); meter = rt.Meter("orthogonal_Q_learning_"+args.tag, gpu=True); failure = None
    result = {"status": "RUNNING", **refs, "states": {state: {"learning": {}} for state in STATES}, "failed_paths": [], "source_records": [], "Q_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0, "resolved_selection_locks": 0, "all_6_locks_before_DEV": False, "DEV_evaluations_completed": 0, "new_F_forward_calls": 0, "new_F_forward_attempts": 0, "new_F_model_objects_loaded": 0, "new_E_D_training_updates": 0, "new_backbone_training": False, "whole_deployment_cost_measured": False, "new_output_space_computed": False, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "selection_rule": "own canonical32 normalized TUNE macro MSE, exact ties earliest CP", "training_recipe": {"optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0., "batch_size": 4, "clip_norm": 1., "updates_per_condition": 600, "checkpoint_updates": [0, 100, 200, 400, 600], "theta_parameters": 6}, "seed_scope": "Same zero theta; minibatch order only; three previously frozen E/D histories", "gradient_probe_scope": "First training batch only; direct theta gradient versus F-input chain plus independent inverse-Q VJP; no independent full training replay"}
    checked, contexts, caches, states, models, selected_states, protected = {}, {}, {}, {}, {}, {}, []
    try:
        install_protected_test_guard(); meter.protected_check(); protected = prior_file_records(calibration, native, science, full, controls, controls_audit, refs)
        snapshots = output/"sources"; snapshots.mkdir()
        for path in (Path(__file__), HERE/"PLAN_Q_LEARNING_KO_v1.md", HERE/"orthogonal_latent_basis_v1.py", HERE/"run_orthogonal_basis_controls_v1.py", HERE/"linear_e_adaptation_v1.py", HERE/"run_linear_e_adaptation_v1.py", HERE/"native_head_v1.py", HERE/"causal_lag_decoder_v1.py", HERE/"fixed_decoder_span_v1.py", HERE/"decoder_metric_probe_v1.py", HERE/"continuation_runtime_v1.py", OLD/"run_condition_probes.py", OLD/"prepare_inputs.py", BASE/"cost_readout_v1.py"):
            destination = snapshots/path.name; shutil.copyfile(path, destination); result["source_records"].append({**old.record(path), "snapshot_path": str(destination)})
        result.update(controller_source=old.record(__file__), module_source=old.record(module.__file__), plan=old.record(HERE/"PLAN_Q_LEARNING_KO_v1.md"))
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        subset = old.read(old.bound(science["raw_subset_diagnostic"])); original = old.read(old.bound(subset["source_producer"]))
        engine = probes.import_old_engine(); weight, config = (probes.verified_file(original[key]) for key in ("model_binary_before", "model_config_before")); bolt = engine.ex.previous.FrozenBolt(); result["new_F_model_objects_loaded"] = 1
        result.update(F_before=probes.model_hashes(bolt), model_binary_before=weight, model_config_before=config)
        require(result["F_before"] == controls["F_before"] == controls["F_after"], "EXACT_FIXED_CONTROL_F_DUAL_HASH")
        result["runtime_behavior_contract"] = runtime.bolt_runtime_contract(engine, bolt, meter.check); require(result["runtime_behavior_contract"] == full["runtime_behavior_contract"] == controls["runtime_behavior_contract"], "UNCHANGED_MEDIAN_F_RUNTIME_CONTRACT")
        def actual_F(z):
            result["new_F_forward_attempts"] += 1; future = bolt(z); result["new_F_forward_calls"] += 1; return future
        actual_F.model = bolt.model
        def load(binding): return io.checked_payload({**binding, "path": str(old.bound(binding))}, meter, checked)
        def context(split):
            require(split in ("fit", "tune") or result["all_6_locks_before_DEV"], "SIX_LOCK_REGISTRY_BEFORE_DEV_ARRAY_IO")
            if split not in contexts:
                source = science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"]; raw = load(source); n = {"fit": 256, "tune": 64, "val": 30}[split]
                require(raw["history"].dtype == raw["target"].dtype == np.float32 and raw["history"].shape == (n, 512, 17) and raw["target"].shape == raw["observed"].shape == (n, 48, 17) and raw["observed"].dtype == np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"], raw["history"][:, -1]), "EXACT_ORIGINAL_ROBIN_CONTEXT_MASK_CURRENT_C")
                contexts[split] = raw
            return contexts[split]
        def frozen(state):
            if state not in states:
                source = native["states"][state]["frozen_state_source"]; old.bound(source, controls["states"][state]["frozen_state_source"]); value = load(source)
                require(set(value) == {"E", "D", "center"} and value["E"].shape == (17, 4) and value["D"].shape == (28, 17) and value["center"].shape == (17,) and all(a.dtype == np.float32 and np.isfinite(a).all() for a in value.values()), "FROZEN_C17_K4_L7_STATE")
                states[state] = value; result["states"][state].update(frozen_state_source=source, frozen_state_value_sha256=state_hash({k: torch.as_tensor(v) for k, v in value.items()}), E_previously_supervised_by_FIT_future=state != "RECON_CP0", D_previously_supervised_by_FIT_future=False)
            return states[state]
        def cache(state, split):
            if (state, split) not in caches:
                raw = context(split); value = load(native["states"][state]["Bolt"][split]["artifact"])
                for key in ("target", "observed", "raw_last", "origins", "mean", "std"): require(old.byte_equal(value[key], raw[key]), "MATCHED_BASELINE_CONTEXT:"+key)
                require(value["latent_history"].shape == (len(raw["history"]), 512, 4) and value["latent_future"].shape == (len(raw["history"]), 48, 4) and value["prediction"].shape == raw["target"].shape and all(value[k].dtype == np.float32 and np.isfinite(value[k]).all() for k in ("latent_history", "latent_future", "prediction")), "MATCHED_FINITE_BASELINE_CACHE")
                caches[state, split] = value
            return caches[state, split]
        def save_evaluation(label, model, state, split, outputs, *, cp0=False):
            raw, cached = context(split), cache(state, split); Q = model.basis().detach().cpu().numpy().copy()
            artifact = io.save_npz(output/(label+".npz"), {**outputs, "Q_FP32": Q, "theta_FP32": model.theta.detach().cpu().numpy().copy(), **{k: raw[k] for k in ("target", "observed", "raw_last", "origins", "mean", "std")}})
            try:
                entry, independent = evaluate_arrays(outputs, raw, cached, frozen(state), Q, cp0=cp0)
                entry["independent_artifact"] = io.save_npz(output/(label+"_INDEPENDENT.npz"), independent)
            except Exception as error: entry = {"status": "FAILED_DIAGNOSTIC_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
            entry.update(artifact=artifact, input_source=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"], cached_Bolt_source=native["states"][state]["Bolt"][split]["artifact"])
            return entry
        def run(state, seed):
            label = state+"_Q_LEARN_s"+str(seed); row = {"status": "FAILED_PATH_PRESERVED", "state_label": state, "seed": seed, "training_updates_completed": 0, "checkpoint_count": 0}; checkpoints = []; log_path = output/(label+"_UPDATES.jsonl")
            try:
                v = frozen(state); model = module.OrthogonalLatentBasisAdapter(v["center"], v["E"], v["D"], LAGS, learn_Q=True).to("cuda").eval(); training_status(model, actual_F)
                initial_hash = state_hash(model.state_dict()); row.update(initial_state_sha256=initial_hash, initial_theta_is_zero=True, frozen_state_source=native["states"][state]["frozen_state_source"])
                def checkpoint(update, tensors, score, outputs):
                    saved = io.save_npz(output/(label+"_CP"+str(update)+"_STATE.npz"), {k: value.numpy().copy() for k, value in tensors.items()})
                    entry = save_evaluation(label+"_CP"+str(update)+"_TUNE", model, state, "tune", outputs, cp0=update == 0)
                    item = {"update": update, "state": saved, "tune": score, "basis_diagnostics": model.basis_diagnostics(), "frozen_status": training_status(model, actual_F), "E_D_center_exact_initial_bytes": all(old.byte_equal(tensors[k].numpy(), v[k]) for k in v), "canonical_score_link_atol": CP_SCORE_LINK_ATOL, "canonical_score_link_rtol": CP_SCORE_LINK_RTOL, **entry}
                    checkpoints.append(item); row["checkpoint_count"] += 1
                    item["basis_diagnostics"] = checked_basis(model); require(item["E_D_center_exact_initial_bytes"], "FROZEN_BYTES_IN_SAVED_CP")
                    if entry["status"] == "COMPLETE": check_score_link(score, entry)
                    return item
                def gradient_callback(proof, arrays):
                    artifact = io.save_npz(output/(label+"_FIRST_BATCH_GRADIENTS.npz"), arrays); row["first_batch_gradient_probe"] = {**proof, "artifact": artifact}; return artifact
                with log_path.open("x", encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"] = event["update"]
                        if event and event.get("stage") == "TRAINING_UPDATE": log.write(json.dumps(event, allow_nan=False)+"\n"); log.flush()
                    trained = train_Q(model, actual_F, context("fit"), context("tune"), cache(state, "tune"), seed, check, checkpoint, gradient_callback)
                require(trained["order_sha256"] == native["states"][state]["native"][str(seed)]["order_sha256"], "SAME_PARENT_600_BY_FOUR_ORDER")
                row.update({k: value for k, value in trained.items() if k not in ("model", "checkpoints", "learning_curve")}); selected = next(cp for cp in checkpoints if cp["update"] == trained["selected_update"])
                outputs, _ = fixed.infer_control(model, actual_F, context("fit")["history"], cache(state, "fit"), meter.check)
                fit_entry = save_evaluation(label+"_SELECTED_FIT", model, state, "fit", outputs, cp0=trained["selected_update"] == 0)
                row.update(status="COMPLETE" if fit_entry["status"] == "COMPLETE" else "FAILED_PATH_PRESERVED", checkpoints=checkpoints, updates_log=old.record(log_path), selected_state=selected["state"], selected_basis_diagnostics=checked_basis(model), FIT=fit_entry, TUNE={k: v for k, v in selected.items() if k not in ("state", "basis_diagnostics", "frozen_status", "E_D_center_exact_initial_bytes")})
                models[state, seed] = model; selected_states[state, seed] = old.state_arrays(model)
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED", failure={"type": type(error).__name__, "message": str(error)}, checkpoints=checkpoints)
                if log_path.exists(): row["updates_log"] = old.record(log_path)
            return row
        def seal(state, seed, row):
            path = output/(state+"_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path, {**row, "DEV_opened_before_lock": False, "sealed_unix": time.time()}); return old.record(path)
        def registry(rows):
            path = output/"SELECTION_REGISTRY.json"; rt.write_new(path, {"status": "ALL_SIX_RESOLVED_BEFORE_DEV", "conditions": rows, "DEV_opened_before_seal": False, "sealed_unix": time.time()}); return old.record(path)
        def development(state, seed, row):
            old.bound(result["selection_registry"]); old.bound(row["selection_lock"]); old.bound(row["selected_state"]); model = models[state, seed]; training_status(model, actual_F)
            outputs, _ = fixed.infer_control(model, actual_F, context("val")["history"], cache(state, "val"), meter.check)
            return save_evaluation(state+"_Q_LEARN_s"+str(seed)+"_SELECTED_DEV", model, state, "val", outputs, cp0=row["selected_update"] == 0)
        with threadpool_limits(limits=2): resolved = resolve_all(result, run, seal, registry); infer_development(result, resolved, development)
        for key, values in selected_states.items(): require(all(old.byte_equal(v, old.state_arrays(models[key])[k]) for k, v in values.items()), "SELECTED_STATE_IMMUTABLE_DURING_FINAL_INFERENCE")
        result["checkpoint_artifact_hashes_rechecked"] = 0
        for state in STATES:
            for row in result["states"][state]["learning"].values():
                old.bound(row["selection_lock"])
                for cp in row.get("checkpoints", []):
                    for key in ("state", "artifact", "independent_artifact"):
                        if key in cp: old.bound(cp[key]); result["checkpoint_artifact_hashes_rechecked"] += 1
                for key in ("FIT", "DEV", "first_batch_gradient_probe"):
                    for field in ("artifact", "independent_artifact"):
                        if field in row.get(key, {}): old.bound(row[key][field])
        old.bound(result["selection_registry"])
        result["F_after"] = probes.model_hashes(bolt); result["model_binary_after"], result["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        require(result["F_before"] == result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()), "FINAL_F_DUAL_HASH_FROZEN_NO_GRADIENT")
        for item in protected+list(checked.values())+result["source_records"]: old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_SNAPSHOTS_UNCHANGED")
        result.update(input_records=list(checked.values()), protected_prior_records=protected, new_optimizer_updates=result["training_updates_completed"], new_Q_learning_updates=result["training_updates_completed"], status="COMPLETE" if result["Q_training_fits"] == 6 and result["training_updates_completed"] == 3600 and result["checkpoint_states_saved"] == 30 and result["DEV_evaluations_completed"] == 6 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check(); json.dumps(result, allow_nan=False)
    except Exception as error: failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
    finally: result["resource_terminal"] = meter.close(result["status"], failure); rt.write_new(output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output/"RESULT.json")}), flush=True); return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tag", "fixed-controls-result", "fixed-controls-audit", "calibration-result", "calibration-audit", "calibration-cost", "calibration-cost-audit", "report", "report-manifest", "report-audit"): parser.add_argument("--"+key, required=True)
    raise SystemExit(execute(parser.parse_args()))
