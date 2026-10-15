"""Fresh, gated PCA-E adaptation through frozen Bolt; actual execution is opt-in."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
BASE, ROOT = HERE.parent, HERE.parents[2]
OLD = ROOT / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
LAGS = {"L0": (0,), "L7": (0, 1, 2, 4, 8, 16, 32)}
SEEDS = (92601, 92602)
CONDITION_KEYS = tuple((data, lag, learn, seed) for data in DATASETS for lag in LAGS for learn in (False, True) for seed in SEEDS)
REPORT_SHA = "d8c7b1e09c2b1677bc40dea8bccc5ec081756ca9cbf088fccb9d8ec52fcf3928"
MANIFEST_SHA = "e4e04e2a86a58f02da2fed28cda4eb2e59c0f1973db98d1f57c470645e0965ea"
REPORT_AUDIT_SHA = "4f208f364222f6810a6366eb242e7e4abf90a8a317fdda831b7d11f0d5ff2ca2"


def require(value, message):
    if not value: raise ValueError(message)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def record(path):
    path = resolve(path)
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def bound(binding, expected=None):
    path = resolve(binding["path"])
    require(path.is_relative_to(ROOT) and path.name.lower() not in ("test.npz", "jena_test.npz"), "LOCAL_NON_TEST_ASSET_REQUIRED")
    require(record(path)["sha256"] == binding["sha256"], "EXACT_SOURCE_SHA:" + path.name)
    if "bytes" in binding: require(path.stat().st_size == binding["bytes"], "EXACT_SOURCE_BYTES")
    if expected: require(path == resolve(expected["path"]) and binding["sha256"] == expected["sha256"], "EXACT_PARENT_BINDING")
    return path


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def load_gates(science_path, science_audit_path, report_path, manifest_path, report_audit_path):
    require(all(value is not None for value in (REPORT_SHA, MANIFEST_SHA, REPORT_AUDIT_SHA)), "REPORT_GATE_NOT_REGISTERED")
    refs = {"parent_report": record(report_path), "parent_report_manifest": record(manifest_path), "parent_report_audit": record(report_audit_path)}
    require([refs[key]["sha256"] for key in refs] == [REPORT_SHA, MANIFEST_SHA, REPORT_AUDIT_SHA], "EXACT_ACCEPTED_REPORT_BUNDLE_REQUIRED_BEFORE_ARRAY_MODEL_IMPORT")
    audit, manifest = read(report_audit_path), read(manifest_path)
    require(audit.get("status") == "PASS" and manifest.get("status") == "COMPLETE"
        and audit.get("files_checked") == 31 and audit.get("total_CSV_primitive_fields_checked") == 967908
        and audit.get("local_links_checked") == 56 and audit.get("PNG_count") == audit.get("SVG_count") == 6,
        "FULL_ACCEPTED_REPORT_VERIFICATION_REQUIRED")
    bound(audit["report"], refs["parent_report"]); bound(audit["manifest"], refs["parent_report_manifest"])
    for entry in manifest["files"]:
        path = (resolve(manifest_path).parent / entry["path"]).resolve()
        require(path.is_relative_to(resolve(manifest_path).parent), "REPORT_FILE_CONTAINMENT")
        bound({**entry, "path": str(path)})
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    from run_causal_lag_cost_v1 import load_gates as prior_gates
    science, subset, original, bindings, locks = prior_gates(science_path, science_audit_path)
    bound(audit["inputs"]["science"], record(science_path)); bound(audit["inputs"]["audit"], record(science_audit_path))
    refs.update(causal_lag_diagnostic=record(science_path), causal_lag_audit=record(science_audit_path))
    return science, subset, original, refs


def fresh_output(tag):
    require(isinstance(tag, str) and bool(tag) and Path(tag).name == tag and tag not in (".", ".."), "SINGLE_FRESH_TAG_REQUIRED")
    output = (HERE / "runs" / tag).resolve()
    require(output.parent == (HERE / "runs").resolve() and not output.exists(), "FRESH_OUTPUT_REQUIRED")
    return output


def byte_equal(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def parity(actual, reference):
    import numpy as np
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32, "SAME_FP32_PARITY_SHAPE_REQUIRED")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "FINITE_PARITY_VALUES_REQUIRED")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {"max_abs": float(np.abs(delta).max(initial=0.)), "rmse": float(np.sqrt(np.mean(delta*delta))),
        "pass": bool(np.allclose(actual, reference, atol=1e-5, rtol=2e-6)), "atol": 1e-5, "rtol": 2e-6}


def name(key):
    return "_".join((key[0], key[1], "E_LEARN" if key[2] else "E_FIXED", "s" + str(key[3])))


def state_arrays(model):
    return {key: tensor.detach().cpu().numpy().copy() for key, tensor in model.state_dict().items()}


def predict_components(model, bolt, history, check, batch_size=32):
    import numpy as np
    import torch
    pieces = {key: [] for key in ("prediction", "latent_history", "latent_future", "lag_features")}
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            x = torch.as_tensor(history[start:start+batch_size], dtype=torch.float32, device=model.E.device)
            p, z, f = model(x, bolt, return_components=True)
            for key, value in zip(pieces, (p, z, f, model.lag_features(z, f))):
                pieces[key].append(value.detach().cpu().numpy().copy())
            check({"stage": "CANONICAL_PREDICTION", "batch_size": batch_size, "start": start})
    result = {key: np.ascontiguousarray(np.concatenate(values), dtype=np.float32) for key, values in pieces.items()}
    require(all(np.isfinite(value).all() for value in result.values()), "FINITE_PREDICTION_COMPONENTS")
    return result


def train_condition(model, bolt, fit, tune, seed, check, checkpoint_callback, *, updates=600):
    import numpy as np
    import torch
    import linear_e_adaptation_v1 as pure
    from native_head_v1 import clone_state, state_hash, score_predictions, select_checkpoint
    require(0 <= updates <= 600, "TRAINING_BUDGET_MAXIMUM_600")
    device = model.E.device
    tensors = {key: torch.as_tensor(fit[key], device=device) for key in ("history", "target", "observed")}
    counts = tensors["observed"].sum((0, 1))
    require(bool((counts > 0).any()), "FIT_HAS_NO_OBSERVED_CHANNELS")
    order = np.random.default_rng(seed).integers(0, len(fit["history"]), size=(updates, 4))
    parameters = [value for value in model.parameters() if value.requires_grad]
    require({id(value) for value in parameters} == {id(value) for value in (model.E, model.D) if value.requires_grad}, "ONLY_E_D_IN_OPTIMIZER")
    if hasattr(bolt, "model"):
        require(all(not p.requires_grad and p.grad is None for p in bolt.model.parameters())
            and not any(id(p) in {id(value) for value in parameters} for p in bolt.model.parameters()), "F_FROZEN_EXCLUDED_FROM_OPTIMIZER")
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0.)
    cp_updates = sorted({0, updates, *(step for step in (100, 200, 400, 600) if step <= updates)})
    checkpoints, curve = [], []
    for update in range(updates+1):
        if update in cp_updates:
            outputs = predict_components(model, bolt, tune["history"], check)
            score = score_predictions(outputs["prediction"], tune["target"], tune["observed"])
            state = clone_state(model)
            checkpoint = {"update": update, "tune": score, "state": state, "state_sha256": state_hash(state),
                "parameter_diagnostics": pure.parameter_diagnostics(model),
                "TUNE_latent_variance": np.var(outputs["latent_history"].astype(np.float64), axis=(0, 1)).tolist()}
            checkpoint["artifact_metadata"] = checkpoint_callback(update, state, score, outputs)
            checkpoints.append(checkpoint)
        if update == updates: break
        ids = torch.as_tensor(order[update], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensors["history"][ids], bolt)
        loss = pure.global_fit_macro_mse(prediction, tensors["target"][ids], tensors["observed"][ids], counts, len(fit["history"]))
        require(bool(torch.isfinite(loss)), "NONFINITE_TRAINING_LOSS")
        loss.backward()
        e_norm = float(model.E.grad.norm().detach()) if model.E.grad is not None else 0.
        d_norm = float(model.D.grad.norm().detach()) if model.D.grad is not None else 0.
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        require(bool(torch.isfinite(norm)), "NONFINITE_TRAINING_GRADIENT")
        optimizer.step()
        check({"stage": "OPTIMIZER_STEP_EXECUTED", "update": update+1})
        require(all(bool(torch.isfinite(value).all()) for value in parameters), "NONFINITE_TRAINING_PARAMETERS")
        row = {"stage": "TRAINING_UPDATE", "update": update+1, "loss": float(loss.detach()),
            "E_gradient_norm_before_clip": e_norm, "D_gradient_norm_before_clip": d_norm,
            "global_gradient_norm_before_clip": float(norm.detach()), "clipped": bool(norm > 1.)}
        curve.append(row); check(row)
    selected = select_checkpoint(checkpoints)
    model.load_state_dict(selected["state"]); model.eval()
    return {"model": model, "checkpoints": checkpoints, "learning_curve": curve, "training_updates_completed": updates,
        "selected_update": selected["update"], "selected_tune": selected["tune"], "selected_state_sha256": selected["state_sha256"],
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "FIT_channel_observed_counts": counts.cpu().tolist(),
        "loss": "N_fit/B/C_active * sum_c SSE_batch_c/N_fit_c; fixed entire-FIT counts", "global_clipping_fraction": sum(row["clipped"] for row in curve)/max(1, len(curve))}


def validate_probe_registry(probe):
    expected = {(data, lag) for data in DATASETS for lag in LAGS}
    found = {(data, lag) for data, row in probe.get("cases", {}).items() for lag in row}
    require(probe.get("mode") == "probe" and found == expected, "ALL_FOUR_PROBE_PATHS_MUST_BE_RESOLVED")
    require(all(probe["cases"][data][lag]["status"] in ("PASS", "FAILED_PATH_PRESERVED") for data, lag in found), "RESOLVED_PROBE_STATUS")


def probe_allows(probe, key):
    return probe["cases"][key[0]][key[1]]["status"] == "PASS"


def train_all_conditions(report, do_condition, seal, registry, check):
    resolved, locks = {}, []
    for key in CONDITION_KEYS:
        check({"stage": "TRAINING_CONDITION", "condition": name(key)})
        try:
            row = do_condition(key)
        except Exception as error:
            row = {"status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
        binding = seal(key, row)
        row["selection_lock"] = binding
        report["datasets"].setdefault(key[0], {"conditions": {}})["conditions"][name(key)] = row
        locks.append({"dataset": key[0], "lag": key[1], "learn_E": key[2], "seed": key[3], "selection_lock": binding, "status": row["status"]})
        report["training_updates_completed"] += row.get("training_updates_completed", 0)
        report["checkpoint_states_saved"] += row.get("checkpoint_count", 0)
        if row["status"] == "COMPLETE":
            resolved[key] = row; report["adaptation_training_fits"] += 1
        else: report["failed_paths"].append({"condition": name(key), **row})
    report["selection_registry"] = registry(locks)
    report["resolved_selection_locks"], report["all_16_locks_before_any_DEV"] = len(locks), True
    return resolved


def infer_development(report, resolved, predict):
    require(report.get("all_16_locks_before_any_DEV") is True and report.get("resolved_selection_locks") == 16, "ALL_16_SELECTION_LOCKS_REQUIRED_BEFORE_DEV")
    for key, row in resolved.items():
        try: predict(key, row)
        except Exception as error:
            failure = {"condition": name(key), "stage": "DEV_INFERENCE", "type": type(error).__name__, "message": str(error)}
            row["DEV_failure"] = failure; report["failed_paths"].append(failure)


def execute(args):
    science, subset, original, refs = load_gates(args.science, args.science_audit, args.report, args.report_manifest, args.report_audit)
    output = fresh_output(args.tag)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as runtime
    import fixed_decoder_span_v1 as io
    import linear_e_adaptation_v1 as pure
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    from decoder_metric_probe_v1 import masked_scores
    from native_head_v1 import score_predictions, state_hash
    output.mkdir(parents=True, exist_ok=False)
    meter = runtime.Meter("linear_E_"+args.mode+"_"+args.tag, gpu=True)
    report = {"status": "RUNNING", "mode": args.mode, **refs, "datasets": {}, "cases": {}, "failed_paths": [],
        "adaptation_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0,
        "probe_optimizer_updates_completed": 0, "protected_TEST_accessed": False, "new_backbone_training": False,
        "independent_holdout": False, "strict_K_only": False, "same_initialization_seeds_change_order_only": True,
        "whole_deployment_cost_measured": False, "all_16_locks_before_any_DEV": False,
        "training_contract": {"updates_per_fit": 600, "checkpoints": [0, 100, 200, 400, 600], "batch_size": 4,
            "canonical_prediction_batch_size": 32, "optimizer": "AdamW", "learning_rate": .001,
            "weight_decay": 0., "global_gradient_clip": 1., "full_FIT_channel_count_loss": True,
            "E_center_fixed": True, "F_parameters_optimized": False, "F_only_gradient_probe_is_necessity_test": False}}
    checked, contexts, initial, trained = {}, {}, {}, {}
    failure, bolt, protected = None, None, []
    try:
        install_protected_test_guard(); meter.protected_check()
        protected = prior_file_records(science, *refs.values())
        report["controller_source"], report["pure_source"], report["plan"] = (record(path) for path in
            (Path(__file__), HERE/"linear_e_adaptation_v1.py", HERE/"PLAN_LINEAR_E_ADAPTATION_KO_v1.md"))
        source_paths = [Path(__file__), HERE/"linear_e_adaptation_v1.py", HERE/"PLAN_LINEAR_E_ADAPTATION_KO_v1.md",
            HERE/"continuation_runtime_v1.py", HERE/"fixed_decoder_span_v1.py", HERE/"decoder_metric_probe_v1.py", HERE/"native_head_v1.py",
            HERE/"run_causal_lag_cost_v1.py", OLD/"prepare_inputs.py", OLD/"run_condition_probes.py", BASE/"cost_readout_v1.py",
            ROOT/"research/smallk_predictive_alignment_20261013/engine.py",
            ROOT/"research/forecast_generalization_study_20261013/experiment.py",
            ROOT/"research/forecast_generalization_study_20261013/diagnose_constant_gradient.py",
            ROOT/"research/forecast_component_study_20261013/component_experiment.py"]
        folder = output/"sources"; folder.mkdir()
        report["source_records"] = []
        for path in source_paths:
            destination = folder/path.name; shutil.copyfile(path, destination)
            binding = record(path); require(record(destination)["sha256"] == binding["sha256"], "EXACT_SOURCE_SNAPSHOT")
            report["source_records"].append({**binding, "snapshot_path": str(destination)})
        probe = None
        if args.mode == "train":
            probe = read(args.probe_result); validate_probe_registry(probe)
            for key in ("controller_source", "pure_source", "plan", "causal_lag_diagnostic", "causal_lag_audit", "parent_report", "parent_report_manifest", "parent_report_audit"):
                bound(probe[key], report[key])
            require(probe.get("F_before") == probe.get("F_after") == original["F_before"], "UNCHANGED_ACCEPTED_PROBE_F")
            report["probe_result"] = record(args.probe_result); protected.extend(prior_file_records(probe))
        engine = probes.import_old_engine()
        report["model_binary_before"], report["model_config_before"] = (probes.verified_file(original[key]) for key in ("model_binary_before", "model_config_before"))
        bolt = engine.ex.previous.FrozenBolt(); report["F_before"] = probes.model_hashes(bolt)
        require(report["F_before"] == original["F_before"], "EXACT_FROZEN_F_DUAL_SIGNATURE")
        report["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        def check(event=None):
            meter.check(event)
            if event and event.get("stage") == "TRAINING_UPDATE" and event["update"] % 20 == 0:
                print(json.dumps(event, ensure_ascii=False), flush=True)
        def load_context(data, split):
            require(split in ("fit", "tune") or report["all_16_locks_before_any_DEV"], "NO_DEV_ARRAY_IO_BEFORE_16_LOCKS")
            key = data, split
            if key not in contexts:
                c, k, dev, dev_n = DATASETS[data]
                require(split in ("fit", "tune", dev), "REGISTERED_SPLIT_ONLY")
                entry = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]
                values = io.checked_payload(entry["input_source"], meter, checked)
                n = {"fit": 256, "tune": 64}.get(split, dev_n)
                require(values["history"].dtype == values["target"].dtype == np.float32 and values["history"].shape == (n, 512, c)
                    and values["target"].shape == (n, 48, c) and values["observed"].dtype == np.bool_
                    and values["observed"].shape == values["target"].shape and np.isfinite(values["history"]).all()
                    and np.isfinite(values["target"][values["observed"]]).all() and byte_equal(values["raw_last"], values["history"][:, -1]), "EXACT_NORMALIZED_PAST_ACTUAL_Y_MASK_ANCHOR")
                contexts[key] = values
            return contexts[key]
        def initialization(data, lag):
            key = data, lag
            if key not in initial:
                parent = science["datasets"][data]["spaces"]["PCA"]
                e = io.checked_payload(parent["E_source"], meter, checked)
                entry = parent["conditions"][lag+"_FORECAST"]
                d = io.checked_payload(entry["state_artifact"], meter, checked)
                require(d["lags"].tolist() == list(LAGS[lag]), "EXACT_INITIAL_FORECAST_LAGS")
                initial[key] = (e["center"].astype(np.float32), e["encoder"].astype(np.float32), d["decoder_FP32"].copy())
                report.setdefault("initial_sources", {}).setdefault(data, {})[lag] = {"E": parent["E_source"], "D": entry["state_artifact"], "fit_lock": entry["fit_lock"]}
            return initial[key]
        def new_model(data, lag, learn):
            return pure.LinearLagAdapter(*initialization(data, lag), LAGS[lag], learn_E=learn).to("cuda")
        def scores(outputs, values):
            observed = values["observed"]; complete = np.broadcast_to(observed.all(2)[:, :, None], observed.shape)
            return {"scores_all_observed": masked_scores(outputs["prediction"], values["target"], observed, values["std"]),
                "scores_complete_mask": masked_scores(outputs["prediction"], values["target"], complete, values["std"]) if complete.any() else {"status": "NO_COMPLETE_C"}}
        def save_forecast(path, outputs, values):
            arrays = {**outputs, **{key: values[key] for key in ("target", "observed", "raw_last", "origins", "mean", "std")}}
            return {**scores(outputs, values), "artifact": io.save_npz(path, arrays)}
        with threadpool_limits(limits=2):
            if args.mode == "probe":
                for data in DATASETS:
                    report["cases"][data] = {}
                    for lag in LAGS:
                        case = {"status": "FAILED_PATH_PRESERVED", "probe_optimizer_updates_completed": 0}
                        report["cases"][data][lag] = case
                        try:
                            fit, tune = load_context(data, "fit"), load_context(data, "tune")
                            model, fixed = new_model(data, lag, True), new_model(data, lag, False)
                            actual = predict_components(model, bolt, tune["history"], check)
                            frozen = predict_components(fixed, bolt, tune["history"], check)
                            require(all(byte_equal(actual[key], frozen[key]) for key in actual), "FIXED_LEARNED_CP0_BYTE_IDENTITY")
                            batch4 = predict_components(model, bolt, tune["history"], check, 4)
                            entry = science["datasets"][data]["spaces"]["PCA"]["conditions"][lag+"_FORECAST"]["splits"]["tune"]
                            cached = io.checked_payload(entry["cached_Bolt_source"], meter, checked)
                            old = io.checked_payload(entry["artifact"], meter, checked)
                            from causal_lag_decoder_v1 import lagged_delta_features
                            reference = {"latent_history": cached["latent_history"], "latent_future": cached["latent_future"],
                                "lag_features": lagged_delta_features(cached["latent_history"], cached["latent_future"], LAGS[lag]),
                                "prediction": old["ACTUAL_F_prediction_FP32"]}
                            case.update(CP0_NumPy_parity={key: parity(actual[key], reference[key]) for key in actual},
                                canonical32_vs_batch4={key: parity(actual[key], batch4[key]) for key in actual}, fixed_learned_CP0_byte_exact=True,
                                CP0_TUNE_scores=scores(actual, tune), CP0_reference_TUNE_scores=scores(reference, tune),
                                RAW_TUNE_scores=masked_scores(tune["raw_forecast"], tune["target"], tune["observed"], tune["std"]))
                            case["CP0_artifact"] = io.save_npz(output/(data+"_"+lag+"_CP0_PARITY.npz"),
                                {**{"Torch32_"+key: value for key, value in actual.items()},
                                 **{"Torch4_"+key: value for key, value in batch4.items()}, **{"NumPy_"+key: value for key, value in reference.items()}})
                            require(all(row["pass"] for group in (case["CP0_NumPy_parity"], case["canonical32_vs_batch4"]) for row in group.values()), "REGISTERED_CP0_BACKEND_PARITY_FAILED")
                            ids = np.random.default_rng(SEEDS[0]).integers(0, len(fit["history"]), size=(600, 4))[0]
                            x, y = (torch.as_tensor(fit[key][ids], device="cuda") for key in ("history", "target"))
                            observed = torch.as_tensor(fit["observed"][ids], device="cuda")
                            counts = torch.as_tensor(fit["observed"].sum((0, 1)), device="cuda")
                            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
                            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
                            require(optimizer_ids == {id(model.E), id(model.D)}
                                and not any(id(p) in optimizer_ids for p in bolt.model.parameters()), "PROBE_ONLY_E_D_IN_OPTIMIZER")
                            before_state = state_arrays(model)
                            torch.cuda.synchronize(); started = time.perf_counter()
                            prediction, z, future = model(x, bolt, return_components=True)
                            loss = pure.global_fit_macro_mse(prediction, y, observed, counts, len(fit["history"]))
                            diagnostic = pure.gradient_diagnostics(model, loss, z, future)
                            require(diagnostic.get("F_path_available") is True and diagnostic.get("E_F_path_gradient_norm", 0.) > 0., "FINITE_NONZERO_F_INPUT_GRADIENT_REQUIRED")
                            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(norm)), "FINITE_PROBE_LOSS_GRADIENT")
                            optimizer.step(); torch.cuda.synchronize()
                            case["probe_optimizer_updates_completed"] = 1; report["probe_optimizer_updates_completed"] += 1
                            elapsed = time.perf_counter()-started
                            check({"stage": "PROBE_STEP_COMPLETE", "dataset": data, "lag": lag})
                            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()) and all(p.grad is None for p in bolt.model.parameters()), "FINITE_PROBE_PARAMETERS_F_GRAD_NONE")
                            arrays = {"batch_indices": ids, "FIT_channel_observed_counts": counts.cpu().numpy(),
                                **{"CP0_"+key: value for key, value in actual.items()}, **{"NumPy_"+key: value for key, value in reference.items()},
                                **{"before_"+key: value for key, value in before_state.items()}, **{"after_"+key: value for key, value in state_arrays(model).items()}}
                            case.update(status="PASS", gradient_diagnostics=diagnostic, loss=float(loss.detach()),
                                global_gradient_norm_before_clip=float(norm.detach()), forward_gradient_decomposition_backward_optimizer_wall_seconds=elapsed,
                                E_bytes_changed=not byte_equal(before_state["E"], state_arrays(model)["E"]),
                                D_bytes_changed=not byte_equal(before_state["D"], state_arrays(model)["D"]), F_parameters_in_optimizer=False, F_parameter_gradients_none=True,
                                parameters_after_step=pure.parameter_diagnostics(model), artifact=io.save_npz(output/(data+"_"+lag+"_PROBE.npz"), arrays))
                        except Exception as error:
                            case["failure"] = {"type": type(error).__name__, "message": str(error)}
                            report["failed_paths"].append({"dataset": data, "lag": lag, **case})
                validate_probe_registry(report)
            else:
                def do_condition(key):
                    data, lag, learn, seed = key
                    require(probe_allows(probe, key), "MATCHING_PROBE_PATH_FAILED; independent paths continue")
                    fit, tune = load_context(data, "fit"), load_context(data, "tune")
                    model = new_model(data, lag, learn); checkpoints = []
                    row = {"status": "FAILED_PATH_PRESERVED", "dataset": data, "lag": lag, "learn_E": learn, "seed": seed,
                        "training_updates_completed": 0, "checkpoint_count": 0, "initial_state_sha256": state_hash(model.state_dict())}
                    def checkpoint(update, state, score, outputs):
                        arrays = {key: value.numpy().copy() for key, value in state.items()}
                        entry = {"update": update, "tune": score, "state": io.save_npz(output/(name(key)+"_CP"+str(update)+"_STATE.npz"), arrays),
                            "parameter_diagnostics": pure.parameter_diagnostics(model),
                            "TUNE_latent_variance": np.var(outputs["latent_history"].astype(np.float64), axis=(0, 1)).tolist(),
                            **save_forecast(output/(name(key)+"_CP"+str(update)+"_TUNE.npz"), outputs, tune)}
                        checkpoints.append(entry); row["checkpoint_count"] += 1
                        return entry
                    log_path = output/(name(key)+"_UPDATES.jsonl")
                    with log_path.open("x", encoding="utf-8") as log:
                        def progress_check(event=None):
                            check(event)
                            if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"] = event["update"]
                            if event and event.get("stage") == "TRAINING_UPDATE":
                                log.write(json.dumps(event, allow_nan=False)+"\n"); log.flush()
                                row["training_updates_completed"] = event["update"]
                        try:
                            result = train_condition(model, bolt, fit, tune, seed, progress_check, checkpoint)
                            row.update({key: value for key, value in result.items() if key not in ("model", "checkpoints", "learning_curve")})
                            row.update(status="COMPLETE", checkpoint_count=len(checkpoints), checkpoints=checkpoints, updates_log=record(log_path),
                                selected_state=next(item["state"] for item in checkpoints if item["update"] == result["selected_update"]),
                                CP0_TUNE=checkpoints[0], RAW_TUNE_scores=masked_scores(tune["raw_forecast"], tune["target"], tune["observed"], tune["std"]),
                                FIT=save_forecast(output/(name(key)+"_SELECTED_FIT.npz"), predict_components(model, bolt, fit["history"], check), fit))
                            require(all(p.grad is None and not p.requires_grad for p in bolt.model.parameters()), "F_PARAMETERS_REMAIN_FROZEN_WITHOUT_GRADS")
                            trained[key] = model
                        except Exception as error:
                            row.update(status="FAILED_PATH_PRESERVED", failure={"type": type(error).__name__, "message": str(error)}, checkpoints=checkpoints, updates_log=record(log_path))
                    return row
                def seal(key, row):
                    path = output/(name(key)+"_SELECTION_LOCK.json")
                    runtime.write_new(path, {**row, "DEV_opened_before_lock": False, "sealed_unix": time.time()})
                    return record(path)
                def registry(locks):
                    path = output/"SELECTION_REGISTRY.json"
                    runtime.write_new(path, {"status": "ALL_16_RESOLVED_BEFORE_DEV", "conditions": locks,
                        "DEV_opened_before_seal": False, "sealed_unix": time.time()})
                    return record(path)
                resolved = train_all_conditions(report, do_condition, seal, registry, check)
                report["CP0_LS_and_RAW_baselines"] = {}
                for data in DATASETS:
                    report["CP0_LS_and_RAW_baselines"][data] = {}
                    for lag in LAGS:
                        if not any(key[:2] == (data, lag) for key in resolved): continue
                        baseline_model = new_model(data, lag, False)
                        baseline = report["CP0_LS_and_RAW_baselines"][data][lag] = {}
                        for split in ("fit", DATASETS[data][2]):
                            try:
                                values = load_context(data, split)
                                baseline[split] = {"CP0_LS": save_forecast(output/(data+"_"+lag+"_CP0_"+split+".npz"),
                                    predict_components(baseline_model, bolt, values["history"], check), values),
                                    "RAW_scores": masked_scores(values["raw_forecast"], values["target"], values["observed"], values["std"]),
                                    "all_16_locks_already_sealed": True}
                            except Exception as error:
                                failure_row = {"dataset": data, "lag": lag, "split": split, "stage": "CP0_BASELINE",
                                    "type": type(error).__name__, "message": str(error)}
                                baseline[split] = failure_row; report["failed_paths"].append(failure_row)
                def development(key, row):
                    bound(row["selection_lock"]); bound(report["selection_registry"])
                    values = load_context(key[0], DATASETS[key[0]][2])
                    row["DEV"] = save_forecast(output/(name(key)+"_SELECTED_DEV.npz"), predict_components(trained[key], bolt, values["history"], check), values)
                infer_development(report, resolved, development)
        report["F_after"] = probes.model_hashes(bolt)
        require(report["F_after"] == report["F_before"], "UNCHANGED_F_DUAL_SIGNATURE_AFTER_EXECUTION")
        for binding in protected + report["source_records"] + list(checked.values()): bound(binding)
        for key in ("model_binary_before", "model_config_before"): bound(report[key])
        meter.protected_check()
        report.update(input_records=list(checked.values()), protected_prior_records=protected,
            old_assets_sources_F_weights_unchanged=True, status="COMPLETE" if not report["failed_paths"] else "PARTIAL_COMPLETED")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report.update(status="FAILED_PRESERVED", terminal_failure=failure)
    finally:
        report["resource"] = meter.close(report["status"], failure)
        runtime.write_new(output/"RESULT.json", report)
    print(json.dumps({"status": report["status"], "result": record(output/"RESULT.json")}, ensure_ascii=False), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("probe", "train")); parser.add_argument("--tag", required=True)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--science", type=Path, default=HERE/"runs/actual_causal_lag_objective_v1/RESULT.json")
    parser.add_argument("--science-audit", type=Path, default=HERE/"audits/CAUSAL_LAG_OBJECTIVE_AUDIT_v2.json")
    parser.add_argument("--report", type=Path, default=HERE/"reports/causal_lag_objective_v1/REPORT_KO.md")
    parser.add_argument("--report-manifest", type=Path, default=HERE/"reports/causal_lag_objective_v1/MANIFEST.json")
    parser.add_argument("--report-audit", type=Path, default=HERE/"audits/CAUSAL_LAG_REPORT_AUDIT_v1.json")
    args = parser.parse_args(argv)
    if args.mode == "train" and args.probe_result is None: parser.error("--probe-result is required in train mode")
    return args


if __name__ == "__main__":
    result = execute(parse_args())
    raise SystemExit({"COMPLETE": 0, "PARTIAL_COMPLETED": 2, "FAILED_PRESERVED": 1}[result["status"]])
