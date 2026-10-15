"""Fresh L0 stop-F-input-gradient fits; old full-gradient models are references only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old
from run_linear_e_adaptation_v1 import train_condition, predict_components

HERE, ROOT, BASE, OLD = old.HERE, old.ROOT, old.BASE, old.OLD
DATASETS, SEEDS = old.DATASETS, old.SEEDS
KEYS = tuple((data, seed) for data in DATASETS for seed in SEEDS)
TRAIN_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
AUDIT_SHA = "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
REPORT_SHA = "eff21bf19811b2f5a09945494aeca5817d22a2dd4f6d7da8b406a46a8f467230"
MANIFEST_SHA = "67b73f080dedb654b26bb49c14e445ab7914af6d4900b504a2bf45aa0654a488"
REPORT_AUDIT_SHA = "755574afa0e1e1a85f4e29de6453da18080f15fbdc4589a4b248bb61ded6ff05"
require = old.require


class FDetach:
    def __init__(self, bolt):
        self.bolt, self.model = bolt, bolt.model

    def __call__(self, z):
        return self.bolt(z).detach()


def gradient_probe(full, stopped, bolt, x, target, observed, counts, nfit):
    import numpy as np
    import torch
    from linear_e_adaptation_v1 import global_fit_macro_mse
    from native_head_v1 import state_hash
    before = state_hash(full.state_dict()); require(before == state_hash(stopped.state_dict()), "MATCHED_INITIAL_STATE_REQUIRED")
    require(all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()), "F_PARAMETERS_FROZEN_BEFORE_GRADIENT_PROBE")
    p, z, f = full(x, bolt, return_components=True)
    q, zs, fs = stopped(x, FDetach(bolt), return_components=True)
    forward_same = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in ((p.contiguous(), q.contiguous()), (z.contiguous(), zs.contiguous()), (f.contiguous(), fs.contiguous())))
    loss = global_fit_macro_mse(p, target, observed, counts, nfit); stop_loss = global_fit_macro_mse(q, target, observed, counts, nfit)
    ef, df = torch.autograd.grad(loss, (full.E, full.D), retain_graph=True)
    future_g = torch.autograd.grad(loss, f, retain_graph=True)[0]
    via_z = torch.autograd.grad(f, z, grad_outputs=future_g, retain_graph=True)[0]
    via_e = torch.autograd.grad(z, full.E, grad_outputs=via_z, retain_graph=True)[0]
    es, ds = torch.autograd.grad(stop_loss, (stopped.E, stopped.D))
    direct = ef - via_e
    tensors = {"full_E_gradient": ef, "full_D_gradient": df, "F_future_gradient": future_g, "F_input_gradient": via_z, "F_path_E_gradient": via_e, "direct_E_gradient": direct, "stop_E_gradient": es, "stop_D_gradient": ds, "full_prediction": p, "stop_prediction": q, "full_latent_history": z, "stop_latent_history": zs, "full_latent_future": f, "stop_latent_future": fs}
    finite = all(bool(torch.isfinite(v).all()) for v in tensors.values())
    def comparison(a, b):
        return old.parity(a.detach().cpu().numpy(), b.detach().cpu().numpy()) if torch.isfinite(a).all() and torch.isfinite(b).all() else {"pass": False, "failure": "NONFINITE_GRADIENT_VECTOR", "atol": 1e-5, "rtol": 2e-6}
    checks = {"stop_E_equals_full_minus_F_path": comparison(es, direct), "stop_D_equals_full_D": comparison(ds, df)}
    require(state_hash(full.state_dict()) == state_hash(stopped.state_dict()) == before and all(p.grad is None for model in (full, stopped, bolt.model) for p in model.parameters()), "PROBE_NO_OPTIMIZER_UPDATE_OR_GRAD_FIELD_MUTATION")
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in tensors.items()}
    arrays.update({"initial_" + key: value for key, value in old.state_arrays(full).items()})
    passed = finite and forward_same and bool(torch.equal(loss, stop_loss)) and all(c["pass"] for c in checks.values())
    return {"status": "PASS" if passed else "FAILED_PROBE_PRESERVED", "optimizer_updates": 0, "forward_byte_exact": forward_same, "finite_gradient_vectors": finite, "loss": float(loss.detach()), "gradient_checks": checks, "initial_state_sha256": before, "F_parameters_frozen_and_grad_none": True, "parameter_grad_fields_none": True, "gradient_norms": {key: float(np.linalg.norm(value.astype(np.float64))) if np.isfinite(value).all() else None for key, value in arrays.items() if key.endswith("gradient")}, "scope": "Same actual F forward; detach only its output; encoded-past reference path remains. Vectors are preclip; not F-value or TSFM necessity."}, arrays


def resolve_four(report, run, seal, registry):
    resolved, rows = {}, []
    for key in KEYS:
        try: row = run(key)
        except Exception as error: row = {"status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
        lock = seal(key, row); row["selection_lock"] = lock
        report["datasets"].setdefault(key[0], {"conditions": {}})["conditions"][str(key[1])] = row
        rows.append({"dataset": key[0], "seed": key[1], "status": row["status"], "selection_lock": lock})
        report["training_updates_completed"] += row.get("training_updates_completed", 0); report["checkpoint_states_saved"] += row.get("checkpoint_count", 0)
        if row["status"] == "COMPLETE": resolved[key] = row; report["stop_gradient_training_fits"] += 1
        else: report["failed_paths"].append({"dataset": key[0], "seed": key[1], **row})
    report["selection_registry"] = registry(rows); report.update(resolved_selection_locks=len(rows), all_4_locks_before_DEV=True)
    return resolved


def load_gates(training, audit, report, manifest, report_audit):
    require(all(isinstance(v, str) and len(v) == 64 for v in (REPORT_SHA, MANIFEST_SHA, REPORT_AUDIT_SHA)), "CURRENT_REPORT_GATE_NOT_REGISTERED")
    refs = {key: old.record(path) for key, path in (("parent_report", report), ("parent_report_manifest", manifest), ("parent_report_audit", report_audit))}
    require([refs[k]["sha256"] for k in refs] == [REPORT_SHA, MANIFEST_SHA, REPORT_AUDIT_SHA], "EXACT_CURRENT_REPORT_BUNDLE")
    verified = old.read(report_audit); require(verified["status"] == "PASS", "CURRENT_REPORT_INDEPENDENT_PASS_REQUIRED")
    old.bound(verified["report"], refs["parent_report"]); old.bound(verified["manifest"], refs["parent_report_manifest"])
    for item in old.read(manifest)["files"]:
        path = (old.resolve(manifest).parent / item["path"]).resolve(); require(path.is_relative_to(old.resolve(manifest).parent), "PARENT_REPORT_FILE_CONTAINMENT"); old.bound({**item, "path": str(path)})
    from run_linear_e_cost_v1 import load_gates as full_gates
    full, checked, science, subset, original, _, _ = full_gates(training, audit)
    require(old.record(training)["sha256"] == TRAIN_SHA and old.record(audit)["sha256"] == AUDIT_SHA, "EXACT_REUSED_FULL_GRADIENT_FOUR_CONDITIONS")
    require(old.record(HERE / "run_linear_e_adaptation_v1.py")["sha256"] == "e268528d77d17b0c8f5a300ff8d667ce5ddd25b1f61754c3895df87929fcd1f3" and old.record(HERE / "linear_e_adaptation_v1.py")["sha256"] == "17c8ac7d20b9f3dd759a3b3fb2e019e027469e9c8a057eb1176da98b9c319931", "IMMUTABLE_REUSED_TRAINER_AND_PURE")
    refs.update(full_gradient_parent=old.record(training), full_gradient_parent_audit=old.record(audit), causal_lag_diagnostic=full["causal_lag_diagnostic"], causal_lag_audit=full["causal_lag_audit"])
    return full, checked, science, subset, original, refs


def execute(args):
    full_parent, parent_audit, science, subset, original, refs = load_gates(args.training_result, args.training_audit, args.report, args.report_manifest, args.report_audit)
    output = old.fresh_output(args.tag)
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from linear_e_adaptation_v1 import LinearLagAdapter, parameter_diagnostics
    from native_head_v1 import state_hash
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True, exist_ok=False); meter = rt.Meter("F_input_gradient_" + args.tag, gpu=True)
    result = {"status": "RUNNING", **refs, "datasets": {}, "failed_paths": [], "source_records": [], "stop_gradient_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0, "probe_optimizer_updates": 0, "full_gradient_conditions_reused": 4, "new_full_gradient_fits": 0, "new_backbone_training": False, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "whole_deployment_cost_measured": False, "all_4_locks_before_DEV": False, "D_step_scope": "Logged preclip norms, reconstructed global gradient scaling and CP coefficient changes; per-step AdamW parameter-update norms are not measured for the reused full-gradient baseline"}
    checked, contexts, models, failure = {}, {}, {}, None
    try:
        install_protected_test_guard(); meter.protected_check(); protected = prior_file_records(full_parent, parent_audit, science, subset, original, *refs.values())
        folder = output / "sources"; folder.mkdir()
        for path in (Path(__file__), HERE / "PLAN_F_INPUT_GRADIENT_ABLATION_KO_v1.md", HERE / "run_linear_e_adaptation_v1.py", HERE / "linear_e_adaptation_v1.py", HERE / "run_linear_e_cost_v1.py", HERE / "native_head_v1.py", HERE / "fixed_decoder_span_v1.py", HERE / "continuation_runtime_v1.py", OLD / "prepare_inputs.py", OLD / "run_condition_probes.py", BASE / "cost_readout_v1.py"):
            destination = folder / path.name; shutil.copyfile(path, destination); result["source_records"].append({**old.record(path), "snapshot_path": str(destination)})
        result["controller_source"] = old.record(__file__); result["plan"] = old.record(HERE / "PLAN_F_INPUT_GRADIENT_ABLATION_KO_v1.md")
        engine = probes.import_old_engine(); bolt = engine.ex.previous.FrozenBolt(); stop_bolt = FDetach(bolt)
        result.update(F_before=probes.model_hashes(bolt), model_binary_before=probes.verified_file(original["model_binary_before"]), model_config_before=probes.verified_file(original["model_config_before"]))
        require(result["F_before"] == full_parent["F_before"], "EXACT_FROZEN_PARENT_F")
        result["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        def load(record): return io.checked_payload({**record, "path": str(old.bound(record))}, meter, checked)
        def context(data, split):
            require(split in ("fit", "tune") or result["all_4_locks_before_DEV"], "ALL_FOUR_LOCKS_BEFORE_DEV_ARRAY_IO")
            if (data, split) not in contexts:
                source = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]["input_source"]
                value = load(source); c, k, dev, ndev = DATASETS[data]; n = {"fit": 256, "tune": 64, dev: ndev}[split]
                require(value["history"].shape == (n, 512, c) and value["target"].shape == value["observed"].shape == (n, 48, c) and np.isfinite(value["history"]).all() and np.isfinite(value["target"][value["observed"]]).all() and old.byte_equal(value["raw_last"], value["history"][:, -1]), "SAME_PAST_ACTUAL_Y_MASK_ANCHOR")
                contexts[data, split] = value
            return contexts[data, split]
        def save_forecast(path, outputs, values):
            complete = np.broadcast_to(values["observed"].all(2, keepdims=True), values["observed"].shape)
            return {"artifact": io.save_npz(path, {**outputs, **{k: values[k] for k in ("target", "observed", "raw_last", "origins", "mean", "std")}}), "scores_all_observed": masked_scores(outputs["prediction"], values["target"], values["observed"], values["std"]), "scores_complete_mask": masked_scores(outputs["prediction"], values["target"], complete, values["std"]) if complete.any() else {"status": "NO_COMPLETE_C"}}
        def run(key):
            data, seed = key; label = data + "_L0_STOP_s" + str(seed); row = {"status": "FAILED_PATH_PRESERVED", "dataset": data, "seed": seed, "lag": "L0", "learn_E": True, "training_updates_completed": 0, "checkpoint_count": 0}; checkpoints = []
            try:
                fit, tune = context(data, "fit"), context(data, "tune")
                reference_name = data + "_L0_E_LEARN_s" + str(seed); previous = full_parent["datasets"][data]["conditions"][reference_name]
                initial = load(previous["CP0_TUNE"]["state"])
                def model(): return LinearLagAdapter(initial["center"], initial["E"], initial["D"], (0,), learn_E=True).to("cuda").eval()
                full, stopped = model(), model(); require(state_hash(full.state_dict()) == previous["initial_state_sha256"], "SAME_FULL_GRADIENT_INITIALIZATION")
                a = predict_components(full, bolt, tune["history"], meter.check); b = predict_components(stopped, stop_bolt, tune["history"], meter.check); old_cp0 = load(previous["CP0_TUNE"]["artifact"])
                cp0_checks = {field: {"full_stop_byte_exact": old.byte_equal(a[field], b[field]), "full_parent_byte_exact": old.byte_equal(a[field], old_cp0[field])} for field in a}
                cp0_binding = io.save_npz(output / (label + "_CP0_PARITY.npz"), {**{"full_" + k: v for k, v in a.items()}, **{"stop_" + k: v for k, v in b.items()}, "origins": tune["origins"]})
                row.update(CP0_TUNE_parity_artifact=cp0_binding, CP0_TUNE_byte_checks=cp0_checks, reference_CP0=previous["CP0_TUNE"]["artifact"])
                require(all(all(values.values()) for values in cp0_checks.values()), "FULL_STOP_AND_PARENT_CP0_TUNE_BYTE_PARITY; arrays preserved")
                ids = np.random.default_rng(seed).integers(0, len(fit["history"]), size=(600, 4))[0]
                x, y, mask = (torch.as_tensor(fit[k][ids], device="cuda") for k in ("history", "target", "observed")); counts = torch.as_tensor(fit["observed"].sum((0, 1)), device="cuda")
                torch.cuda.synchronize(); started = time.perf_counter(); probe, arrays = gradient_probe(full, stopped, bolt, x, y, mask, counts, len(fit["history"])); torch.cuda.synchronize()
                probe["wall_seconds"] = time.perf_counter() - started; arrays.update(batch_indices=ids, FIT_channel_observed_counts=counts.cpu().numpy())
                probe.update(artifact=io.save_npz(output / (label + "_GRADIENT_PROBE.npz"), arrays), CP0_TUNE_parity_artifact=cp0_binding, reference_CP0=previous["CP0_TUNE"]["artifact"])
                row.update(probe=probe, full_gradient_reference={"condition": reference_name, "selected_update": previous["selected_update"], "selected_state": previous["selected_state"]}, initial_state_sha256=previous["initial_state_sha256"])
                require(probe["status"] == "PASS", "MATCHED_GRADIENT_PROBE_FAILED; vectors preserved, independent paths continue")
                def checkpoint(update, state, score, outputs):
                    item = {"update": update, "tune": score, "state": io.save_npz(output / (label + "_CP" + str(update) + "_STATE.npz"), {k: v.numpy().copy() for k, v in state.items()}), "parameter_diagnostics": parameter_diagnostics(stopped), **save_forecast(output / (label + "_CP" + str(update) + "_TUNE.npz"), outputs, tune)}
                    checkpoints.append(item); row["checkpoint_count"] += 1; return item
                log_path = output / (label + "_UPDATES.jsonl")
                with log_path.open("x", encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"] = event["update"]
                        if event and event.get("stage") == "TRAINING_UPDATE":
                            scale = min(1., 1. / (event["global_gradient_norm_before_clip"] + 1e-6))
                            log.write(json.dumps({**event, "global_clip_scale_from_logged_norm": scale, "D_gradient_norm_after_clip_formula": scale * event["D_gradient_norm_before_clip"], "D_gradient_scaling_is_AdamW_step_norm": False}, allow_nan=False) + "\n"); log.flush(); row["training_updates_completed"] = event["update"]
                    trained = train_condition(stopped, stop_bolt, fit, tune, seed, check, checkpoint)
                row.update({k: v for k, v in trained.items() if k not in ("model", "checkpoints", "learning_curve")})
                require(row["order_sha256"] == previous["order_sha256"], "SAME_600_BATCH_ORDER_AS_REUSED_FULL_GRADIENT")
                row.update(status="COMPLETE", checkpoints=checkpoints, updates_log=old.record(log_path), selected_state=next(cp["state"] for cp in checkpoints if cp["update"] == trained["selected_update"]), FIT=save_forecast(output / (label + "_SELECTED_FIT.npz"), predict_components(stopped, stop_bolt, fit["history"], meter.check), fit))
                models[key] = stopped
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED", failure={"type": type(error).__name__, "message": str(error)}, checkpoints=checkpoints)
                log_path = output / (label + "_UPDATES.jsonl")
                if log_path.exists(): row["updates_log"] = old.record(log_path)
            return row
        def seal(key, row):
            path = output / (key[0] + "_s" + str(key[1]) + "_SELECTION_LOCK.json"); rt.write_new(path, {**row, "DEV_opened_before_lock": False, "sealed_unix": time.time()}); return old.record(path)
        def registry(rows):
            path = output / "SELECTION_REGISTRY.json"; rt.write_new(path, {"status": "ALL_FOUR_RESOLVED_BEFORE_DEV", "conditions": rows, "DEV_opened_before_seal": False, "sealed_unix": time.time()}); return old.record(path)
        with threadpool_limits(limits=2):
            resolved = resolve_four(result, run, seal, registry)
            for key, row in resolved.items():
                try:
                    old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); values = context(key[0], DATASETS[key[0]][2])
                    row["DEV"] = save_forecast(output / (key[0] + "_L0_STOP_s" + str(key[1]) + "_SELECTED_DEV.npz"), predict_components(models[key], stop_bolt, values["history"], meter.check), values)
                except Exception as error: row["DEV_failure"] = {"type": type(error).__name__, "message": str(error)}; result["failed_paths"].append({"dataset": key[0], "seed": key[1], "stage": "DEV", **row["DEV_failure"]})
        result["F_after"] = probes.model_hashes(bolt); require(result["F_before"] == result["F_after"] and all(p.grad is None and not p.requires_grad for p in bolt.model.parameters()), "F_FROZEN_AND_GRAD_NONE_AFTER_FOUR_FITS")
        for item in protected + result["source_records"] + list(checked.values()) + [result["model_binary_before"], result["model_config_before"]]: old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"] == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()), protected_prior_records=protected, status="COMPLETE" if not result["failed_paths"] and result["stop_gradient_training_fits"] == 4 else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; result.update(status="FAILED_PRESERVED", terminal_failure=failure)
    finally:
        result["resource_terminal"] = meter.close(result["status"], failure); rt.write_new(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": old.record(output / "RESULT.json")}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--tag", required=True)
    parser.add_argument("--training-result", type=Path, default=HERE / "runs/linear_e_train_v1/RESULT.json"); parser.add_argument("--training-audit", type=Path, default=HERE / "audits/LINEAR_E_ADAPTATION_AUDIT_v1.json")
    for name in ("report", "report-manifest", "report-audit"): parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))
