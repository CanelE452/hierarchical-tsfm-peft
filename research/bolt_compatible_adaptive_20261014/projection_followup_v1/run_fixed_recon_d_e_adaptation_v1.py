"""Fresh Robin L7 E-only fits with the existing RECON D frozen; no new math loop."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old
from run_linear_e_adaptation_v1 import train_condition, predict_components

HERE, ROOT, BASE, OLD, require = old.HERE, old.ROOT, old.BASE, old.OLD, old.require
SEEDS, LAGS = (92601, 92602), (0, 1, 2, 4, 8, 16, 32)
FIXED_SHA = {"joint": "f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8", "audit": "b1d28415ade18db8ddc0c987a14e879268413efdd5d1c62ad93e27b24becd52e", "report": "cb82dbac77968b510dda060b9534ded20f2386b178f5e9fdc651b473e16e7082", "manifest": "f63836e8060d0b4619e5d9ad1f712aab2884389555b9cac39e8b2c266ba766b7", "report_audit": "66ac298d8847bf1e96659a066e092e64b56d5c81530263d8da830f2a40aa75ab"}
JOINT_AUDIT_SOURCE_SHA = "55066f6dbc6ce8c6a175fe4ce5c8952fe8eabdde15492fd097b3a8d98c1ecdfd"
JOINT_SOURCE_SHA = "a00300046d2236c46be7d6d532cead122cd178f066a62fd4bbbea76566d2534b"
AUDIT_COUNTS = {"recon_initialization_fits_checked": 2, "checkpoint_states_checked": 10, "checkpoint_TUNE_predictions_checked": 10, "selected_FIT_DEV_predictions_checked": 4, "logged_updates_checked": 1200, "CP0_probe_cases_checked": 2, "CP0_parity_components_checked": 8, "reused_FORECAST_initialization_conditions_checked": 2, "selection_locks_checked": 2, "new_space_diagnostic_cells_computed": 4, "new_space_LS_SVD_crosschecks_computed": 4}


def fixed_decoder_status(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    require(names == ["E"] and model.E.requires_grad and not model.D.requires_grad and model.D.grad is None and not model.center.requires_grad, "E_ONLY_OPTIMIZER_SET_AND_FIXED_D_GRAD_NONE")
    return {"trainable_parameter_names": names, "D_requires_grad": False, "D_grad_is_None": True, "center_requires_grad": False}


def freeze_fresh_decoder(model):
    require(all(parameter.grad is None for parameter in model.parameters()), "FRESH_ADAPTER_WITH_NO_STALE_GRADIENTS")
    model.D.requires_grad_(False)
    return fixed_decoder_status(model)


def train_E_only(model, bolt, fit, tune, seed, check, checkpoint, *, updates=600):
    freeze_fresh_decoder(model); initial = old.state_arrays(model); step_checks = 0; cp_checks = 0
    def event(value=None):
        nonlocal step_checks
        flags = fixed_decoder_status(model)
        if value and value.get("stage") == "OPTIMIZER_STEP_EXECUTED": step_checks += 1
        if value and value.get("stage") == "TRAINING_UPDATE":
            require(value["D_gradient_norm_before_clip"] == 0., "FROZEN_D_HAS_NO_RECORDED_GRADIENT")
            value = {**value, **flags, "gradient_clipping_parameter_scope": "E_ONLY"}
        check(value)
    def save(update, state, score, outputs):
        nonlocal cp_checks
        fixed_decoder_status(model)
        for key in ("D", "center"): require(old.byte_equal(state[key].detach().cpu().numpy(), initial[key]), "EVERY_CP_FIXED_INITIAL_"+key)
        cp_checks += 1
        return checkpoint(update, state, score, outputs)
    result = train_condition(model, bolt, fit, tune, seed, event, save, updates=updates)
    final = old.state_arrays(result["model"]); flags = fixed_decoder_status(result["model"])
    require(step_checks == updates and all(old.byte_equal(final[key], initial[key]) for key in ("D", "center")), "SELECTED_STATE_RESTORE_PRESERVES_FIXED_D_AND_FLAGS")
    result.update(D_frozen_after_optimizer_step_checks=step_checks, D_and_center_checkpoint_byte_checks=cp_checks, fixed_decoder_final_status=flags, initial_D_value_sha256=hashlib.sha256(initial["D"].tobytes()).hexdigest(), D_bytes_checked_at_checkpoints_and_final_only=True, independent_per_update_D_bytes_replayed=False, gradient_clipping_parameter_scope="E_ONLY")
    return result


def resolve_two(report, run, seal, registry):
    resolved, rows = {}, []
    for seed in SEEDS:
        try: row = run(seed)
        except Exception as error: row = {"status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
        lock = seal(seed, row); row["selection_lock"] = lock; report["datasets"]["robin"]["conditions"][str(seed)] = row
        rows.append({"dataset": "robin", "seed": seed, "status": row["status"], "selection_lock": lock}); report["training_updates_completed"] += row.get("training_updates_completed", 0); report["checkpoint_states_saved"] += row.get("checkpoint_count", 0)
        if row["status"] == "COMPLETE": resolved[seed] = row; report["fixed_D_E_only_training_fits"] += 1
        else: report["failed_paths"].append({"seed": seed, **row})
    report["selection_registry"] = registry(rows); report.update(resolved_selection_locks=len(rows), all_2_locks_before_DEV=True)
    return resolved


def load_gates(joint_path, audit_path, report_path, manifest_path, report_audit_path):
    require(all(isinstance(value, str) and len(value) == 64 for value in (*FIXED_SHA.values(), JOINT_AUDIT_SOURCE_SHA)), "CURRENT_RECON_JOINT_AUDIT_REPORT_NOT_YET_REGISTERED")
    paths = {"joint": joint_path, "audit": audit_path, "report": report_path, "manifest": manifest_path, "report_audit": report_audit_path}; records = {key: old.record(path) for key, path in paths.items()}
    require(all(records[key]["sha256"] == FIXED_SHA[key] for key in paths), "EXACT_CURRENT_JOINT_AND_REPORT_BUNDLE")
    joint, audit, manifest, report_audit = (old.read(paths[key]) for key in ("joint", "audit", "manifest", "report_audit"))
    require(joint["status"] == manifest["status"] == "COMPLETE" and audit["status"] == report_audit["status"] == "PASS" and audit["training_verification_status"] == "PASS" and audit["new_space_diagnostic_status"] == "COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"], "CURRENT_JOINT_TRAIN_SPACE_AND_REPORT_COMPLETE_PASS")
    for actual, expected in ((audit["diagnostic"], records["joint"]), (report_audit["report"], records["report"]), (report_audit["manifest"], records["manifest"]), (manifest["inputs"]["train"], records["joint"]), (manifest["inputs"]["audit"], records["audit"])): old.bound(actual, expected)
    require(audit["audit_source"]["sha256"] == JOINT_AUDIT_SOURCE_SHA and joint["controller_source"]["sha256"] == JOINT_SOURCE_SHA and all(audit["totals"].get(key) == value for key, value in AUDIT_COUNTS.items()), "EXACT_AUDITOR_SOURCE_AND_TWO_CONDITION_COVERAGE"); old.bound(audit["audit_source"]); old.bound(joint["controller_source"])
    for item in manifest["files"]:
        path = (old.resolve(manifest_path).parent/item["path"]).resolve(); require(path.is_relative_to(old.resolve(manifest_path).parent), "PARENT_REPORT_FILE_CONTAINMENT"); old.bound({**item, "path": str(path)})
    from run_linear_e_cost_v1 import load_gates as parent_gates
    full, full_audit, science, subset, original, _, _ = parent_gates(joint["full_gradient_parent"]["path"], joint["full_gradient_parent_audit"]["path"])
    require(joint["F_before"] == joint["F_after"] == full["F_before"] == full["F_after"] and joint["all_2_locks_before_DEV"] is True and joint["recon_initialization_training_fits"] == 2 and not joint["failed_paths"], "SAME_FROZEN_F_AND_ACCEPTED_TWO_JOINT_CONTROLS")
    refs = {"joint_recon_diagnostic": records["joint"], "joint_recon_audit": records["audit"], "parent_report": records["report"], "parent_report_manifest": records["manifest"], "parent_report_audit": records["report_audit"], "joint_selection_registry": joint["selection_registry"], **{key: joint[key] for key in ("full_gradient_parent", "full_gradient_parent_audit", "causal_lag_diagnostic", "causal_lag_audit")}}
    return joint, audit, full, full_audit, science, subset, original, refs


def execute(args):
    joint, joint_audit, full, full_audit, science, subset, original, refs = load_gates(args.joint_result, args.joint_audit, args.report, args.report_manifest, args.report_audit)
    output = old.fresh_output(args.tag); sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from linear_e_adaptation_v1 import LinearLagAdapter, parameter_diagnostics
    from native_head_v1 import state_hash
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True, exist_ok=False); meter = rt.Meter("fixed_RECON_D_"+args.tag, gpu=True)
    result = {"status": "RUNNING", **refs, "datasets": {"robin": {"conditions": {}}}, "failed_paths": [], "source_records": [], "fixed_D_E_only_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0, "probe_optimizer_updates": 0, "joint_RECON_conditions_reused": 2, "new_joint_RECON_fits": 0, "new_backbone_training": False, "new_D_initialization_fit": False, "D_updated_during_training": False, "D_initialization_supervised_by_FIT_future": False, "E_training_supervised_by_FIT_future": True, "F_input_gradient_retained": True, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "whole_deployment_cost_measured": False, "new_output_space_diagnostics_computed_by_training_producer": False, "all_2_locks_before_DEV": False, "gradient_clipping_parameter_scope": "E_ONLY", "scope": "Same RECON initialization/order/update recipe; fixed D plus E-only adaptation versus reused joint E/D. Procedure comparison with different clipping scope; no independent generalization or pure single-cause effect."}
    checked, contexts, models, failure = {}, {}, {}, None
    try:
        install_protected_test_guard(); meter.protected_check(); protected = prior_file_records(joint, joint_audit, full, full_audit, science, subset, original, *refs.values())
        folder = output/"sources"; folder.mkdir()
        for path in (Path(__file__), HERE/"PLAN_FIXED_RECON_D_E_ADAPTATION_KO_v1.md", HERE/"run_recon_init_sensitivity_v1.py", HERE/"run_linear_e_adaptation_v1.py", HERE/"linear_e_adaptation_v1.py", HERE/"run_linear_e_cost_v1.py", HERE/"native_head_v1.py", HERE/"fixed_decoder_span_v1.py", HERE/"continuation_runtime_v1.py", OLD/"prepare_inputs.py", OLD/"run_condition_probes.py", BASE/"cost_readout_v1.py"):
            destination = folder/path.name; shutil.copyfile(path, destination); result["source_records"].append({**old.record(path), "snapshot_path": str(destination)})
        result.update(controller_source=old.record(__file__), plan=old.record(HERE/"PLAN_FIXED_RECON_D_E_ADAPTATION_KO_v1.md"))
        engine = probes.import_old_engine(); bolt = engine.ex.previous.FrozenBolt()
        result.update(F_before=probes.model_hashes(bolt), model_binary_before=probes.verified_file(original["model_binary_before"]), model_config_before=probes.verified_file(original["model_config_before"]))
        require(result["F_before"] == joint["F_before"], "EXACT_ORIGINAL_FROZEN_F"); result["runtime_behavior_contract"] = runtime.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        recon = science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
        def load(item): return io.checked_payload({**item, "path": str(old.bound(item))}, meter, checked)
        def context(split):
            require(split in ("fit", "tune") or result["all_2_locks_before_DEV"], "TWO_LOCK_REGISTRY_BEFORE_ANY_DEV_ARRAY")
            if split not in contexts:
                values = load(recon["splits"][split]["input_source"]); n = {"fit": 256, "tune": 64, "val": 30}[split]
                require(values["history"].shape == (n, 512, 17) and values["target"].shape == values["observed"].shape == (n, 48, 17) and np.isfinite(values["history"]).all() and np.isfinite(values["target"][values["observed"]]).all() and old.byte_equal(values["raw_last"], values["history"][:, -1]), "SAME_FIXED_ROBIN_INPUT_MASK_AND_CURRENT_C"); contexts[split] = values
            return contexts[split]
        def save_forecast(path, values, raw):
            mask = np.broadcast_to(raw["observed"].all(2, keepdims=True), raw["observed"].shape)
            return {"artifact": io.save_npz(path, {**values, **{key: raw[key] for key in ("target", "observed", "raw_last", "origins", "mean", "std")}}), "scores_all_observed": masked_scores(values["prediction"], raw["target"], raw["observed"], raw["std"]), "scores_complete_mask": masked_scores(values["prediction"], raw["target"], mask, raw["std"]) if mask.any() else {"status": "NO_COMPLETE_C"}}
        def run(seed):
            label = "robin_L7_FIXED_RECON_D_s"+str(seed); row = {"status": "FAILED_PATH_PRESERVED", "dataset": "robin", "seed": seed, "lag": "L7", "learn_E": True, "D_trainable": False, "training_updates_completed": 0, "checkpoint_count": 0}; checkpoints = []
            try:
                fit, tune = context("fit"), context("tune"); previous = joint["datasets"]["robin"]["conditions"][str(seed)]; initial = load(previous["CP0_TUNE"]["state"])
                require(set(initial) == {"E", "D", "center"} and all(value.dtype == np.float32 and np.isfinite(value).all() for value in initial.values()) and initial["E"].shape == (17, 4) and initial["D"].shape == (28, 17), "EXACT_JOINT_RECON_CP0_INITIAL_STATE")
                model = LinearLagAdapter(initial["center"], initial["E"], initial["D"], LAGS, learn_E=True).to("cuda").eval(); freeze_fresh_decoder(model)
                require(state_hash(model.state_dict()) == previous["initial_state_sha256"], "SAME_RECON_INITIAL_STATE_HASH")
                before = state_hash(model.state_dict()); outputs = predict_components(model, bolt, tune["history"], meter.check); reference = load(previous["CP0_TUNE"]["artifact"])
                for key in ("target", "observed", "raw_last", "origins", "mean", "std"): require(old.byte_equal(reference[key], tune[key]), "SAME_JOINT_CP0_INPUT:"+key)
                byte_checks = {key: old.byte_equal(outputs[key], reference[key]) for key in outputs}; probe = {"status": "PASS" if all(byte_checks.values()) else "FAILED_PARITY_PRESERVED", "optimizer_updates": 0, "component_byte_exact": byte_checks, "initial_state_sha256": before, "reference": previous["CP0_TUNE"]["artifact"], "fixed_decoder_status": fixed_decoder_status(model), "artifact": io.save_npz(output/(label+"_CP0_PARITY.npz"), {**{"E_ONLY_"+key: value for key, value in outputs.items()}, **{"JOINT_CP0_"+key: reference[key] for key in outputs}, "origins": tune["origins"]})}
                row.update(probe=probe, initial_state_sha256=before, initial_sources=previous["initial_sources"], joint_recon_reference={"seed": seed, "selected_update": previous["selected_update"], "selected_state": previous["selected_state"], "CP0_TUNE": previous["CP0_TUNE"]})
                require(probe["status"] == "PASS" and state_hash(model.state_dict()) == before and all(p.grad is None for p in model.parameters()), "CP0_BYTE_PARITY_ZERO_UPDATE_ARRAYS_PRESERVED")
                def checkpoint(update, state, score, values):
                    flags = fixed_decoder_status(model); item = {"update": update, "tune": score, "state": io.save_npz(output/(label+"_CP"+str(update)+"_STATE.npz"), {key: value.numpy().copy() for key, value in state.items()}), "fixed_decoder_status": flags, "D_and_center_exact_initial_bytes": all(old.byte_equal(state[key].numpy(), initial[key]) for key in ("D", "center")), "parameter_diagnostics": parameter_diagnostics(model), "TUNE_latent_variance": np.var(values["latent_history"].astype(np.float64), axis=(0, 1)).tolist(), **save_forecast(output/(label+"_CP"+str(update)+"_TUNE.npz"), values, tune)}
                    require(item["D_and_center_exact_initial_bytes"], "EXACT_FIXED_D_AND_CENTER_AT_CP"); checkpoints.append(item); row["checkpoint_count"] += 1; return item
                log_path = output/(label+"_UPDATES.jsonl")
                with log_path.open("x", encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"] = event["update"]
                        if event and event.get("stage") == "TRAINING_UPDATE": log.write(json.dumps(event, allow_nan=False)+"\n"); log.flush(); row["training_updates_completed"] = event["update"]
                    trained = train_E_only(model, bolt, fit, tune, seed, check, checkpoint)
                row.update({key: value for key, value in trained.items() if key not in ("model", "checkpoints", "learning_curve")}); require(row["order_sha256"] == previous["order_sha256"], "SAME_600_BY_4_MINIBATCH_ORDER")
                row.update(status="COMPLETE", checkpoints=checkpoints, updates_log=old.record(log_path), selected_state=next(cp["state"] for cp in checkpoints if cp["update"] == trained["selected_update"]), CP0_TUNE=checkpoints[0], FIT=save_forecast(output/(label+"_SELECTED_FIT.npz"), predict_components(model, bolt, fit["history"], meter.check), fit)); fixed_decoder_status(model); models[seed] = model
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED", failure={"type": type(error).__name__, "message": str(error)}, checkpoints=checkpoints)
                if (output/(label+"_UPDATES.jsonl")).exists(): row["updates_log"] = old.record(output/(label+"_UPDATES.jsonl"))
            return row
        def seal(seed, row):
            path = output/("robin_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path, {**row, "DEV_opened_before_lock": False, "sealed_unix": time.time()}); return old.record(path)
        def registry(rows):
            path = output/"SELECTION_REGISTRY.json"; rt.write_new(path, {"status": "ALL_TWO_RESOLVED_BEFORE_DEV", "conditions": rows, "DEV_opened_before_seal": False, "sealed_unix": time.time()}); return old.record(path)
        with threadpool_limits(limits=2):
            resolved = resolve_two(result, run, seal, registry)
            for seed, row in resolved.items():
                try:
                    old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); raw = context("val"); fixed_decoder_status(models[seed]); row["DEV"] = save_forecast(output/("robin_L7_FIXED_RECON_D_s"+str(seed)+"_SELECTED_DEV.npz"), predict_components(models[seed], bolt, raw["history"], meter.check), raw)
                except Exception as error: row["DEV_failure"] = {"type": type(error).__name__, "message": str(error)}; result["failed_paths"].append({"seed": seed, "stage": "DEV", **row["DEV_failure"]})
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"] == result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()), "F_FROZEN_AND_GRAD_NONE_AFTER_E_ONLY_FITS")
        for item in protected+result["source_records"]+list(checked.values())+[result["model_binary_before"], result["model_config_before"]]: old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()), protected_prior_records=protected, status="COMPLETE" if result["fixed_D_E_only_training_fits"] == 2 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error: failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", failure=failure)
    finally: result["resource_terminal"] = meter.close(result["status"], failure); rt.write_new(output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output/"RESULT.json")}), flush=True); return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tag", "joint-result", "joint-audit", "report", "report-manifest", "report-audit"): parser.add_argument("--"+key, required=True)
    raise SystemExit(execute(parser.parse_args()))
