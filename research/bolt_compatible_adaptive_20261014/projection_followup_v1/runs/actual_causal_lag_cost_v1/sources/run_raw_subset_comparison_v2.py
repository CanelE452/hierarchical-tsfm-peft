"""Fresh full subset comparison after direct-residual numeric repair; preserve v1."""
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
SPACES = {"PCA": "PCA_PINV", "FIT_TARGET": "FIT_TARGET_PINV", "RAW_SUBSET": None}
SEEDS = (92601, 92602)
SOURCE_SHA = "5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98"
SOURCE_AUDIT_SHA = "14399051dc1614f785f4548daa139d92c8769e59fcbee9f38a98126914aedab6"
PRIOR_PARTIAL_SHA = "4240cedea826e052b227ef2fba4e16ff52cdcd6db69281844485eca6c0b0bd6b"
PRIOR_MATH_SHA = "2583b221d1d271a38a3070455d4a2a72c561c101d66ac5b9707822f33ed8909b"
PRIOR_CONTROLLER_SHA = "33b21b100a4b50812573ebcd27daedf7fb094188202c762911e5ba227d7aa290"
PRIOR_HEAD_SHA = "4081130169e4e72de0d46e4130f79e5d1220ccf685a010802a1242c841bc9784"


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle: return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value: raise ValueError(message)


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_gates(source_path, audit_path):
    source, audit = read(source_path), read(audit_path)
    require(source.get("status") == "COMPLETE" and audit.get("status") == "PASS"
        and record(source_path)["sha256"] == SOURCE_SHA and record(audit_path)["sha256"] == SOURCE_AUDIT_SHA
        and audit.get("producer") == record(source_path), "EXACT_FIT_TARGET_COMPLETE_AND_INDEPENDENT_PASS_BEFORE_SCIENCE")
    expected = {"projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2,
        "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
        "saved_prediction_arrays_checked": 88, "output_subspace_diagnostic_cells_checked": 12,
        "fixed_D_real_method_inequalities_checked": 48, "bound_RAW_inequalities_checked": 0}
    require(all(audit.get("totals", {}).get(key) == value for key, value in expected.items()), "FULL_ACCEPTED_SOURCE_AUDIT_COVERAGE")
    require(source.get("protected_TEST_accessed") is False and audit.get("protected_TEST_accessed") is False
        and source.get("independent_holdout") is False and source["F_before"] == source["F_after"], "UNCHANGED_FROZEN_NO_TEST_EXPOSED_DEV_SOURCE")
    for key in ("audit_source",): require(record(audit[key]["path"]) == audit[key], "EXACT_INDEPENDENT_AUDIT_SOURCE")
    require((HERE / "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md").is_file(), "PREDECLARED_RAW_SUBSET_PLAN_REQUIRED")
    return source, {"source_producer": record(source_path), "source_audit": record(audit_path)}



def load_prior_partial(path):
    prior = read(path)
    require(record(path)["sha256"] == PRIOR_PARTIAL_SHA and prior.get("status") == "PARTIAL_COMPLETED",
        "EXACT_PRESERVED_V1_PARTIAL_REQUIRED_BEFORE_SCIENCE")
    require(prior.get("source_producer", {}).get("sha256") == SOURCE_SHA
        and prior.get("source_audit", {}).get("sha256") == SOURCE_AUDIT_SHA,
        "SAME_ACCEPTED_FIT_TARGET_LINEAGE_IN_PRIOR_PARTIAL")
    expected = {"subset_selection_fits": 0, "actual_subset_F_split_inferences": 0, "global_decoder_fits": 8,
        "native_training_fits": 8, "training_updates": 4800, "checkpoint_states_saved": 40,
        "saved_native_split_predictions": 24, "saved_Bolt_split_predictions": 12}
    require(all(prior.get(key) == value for key, value in expected.items()), "EXACT_PRIOR_PARTIAL_COUNTS_REQUIRED")
    require(prior.get("protected_TEST_accessed") is False and prior.get("independent_holdout") is False
        and prior["F_before"] == prior["F_after"], "FROZEN_NO_TEST_PRIOR_PARTIAL_REQUIRED")
    for dataset in DATASETS:
        row = prior["datasets"][dataset]
        require(row["subset_selection_failure"]["failure"]["message"] == "NEGATIVE_RECONSTRUCTION_MSE_BEYOND_FP64_ROUNDOFF",
            "EXACT_PRIOR_GRAM_CANCELLATION_FAILURE_REQUIRED")
        require(all(row["spaces"][space]["status"] == "COMPLETE" for space in ("PCA", "FIT_TARGET")),
            "PRESERVED_COMPLETED_EIGHT_MIXED_CONDITIONS_REQUIRED")
    for filename, sha in (("raw_subset_projection_v1.py", PRIOR_MATH_SHA),
                          ("run_raw_subset_comparison_v1.py", PRIOR_CONTROLLER_SHA),
                          ("raw_subset_native_head_v1.py", PRIOR_HEAD_SHA)):
        require(record(HERE / filename)["sha256"] == sha, "ORIGINAL_EXECUTED_SOURCE_MUST_BE_UNCHANGED")
    diagnostic_path = HERE / "audits/RAW_SUBSET_GRAM_FAILURE_DIAGNOSTIC_ROOT_v1.json"
    diagnostic = read(diagnostic_path)
    require(diagnostic.get("status") == "CONFIRMED_FIRST_STAGE_GRAM_CANCELLATION_VS_DIRECT_RESIDUAL"
        and diagnostic.get("partial_result") == record(path)
        and diagnostic.get("old_source", {}).get("sha256") == PRIOR_MATH_SHA
        and diagnostic.get("FIT_future_targets_read") is False and diagnostic.get("TUNE_DEV_read") is False
        and diagnostic.get("protected_TEST_accessed") is False, "EXACT_PAST_ONLY_GRAM_FAILURE_DIAGNOSTIC_REQUIRED")
    addendum_path = HERE / "RAW_SUBSET_GRAM_NUMERIC_REPAIR_KO_v2.md"
    require(addendum_path.is_file(), "NUMERIC_REPAIR_ADDENDUM_REQUIRED")
    return prior, {"prior_partial": record(path), "Gram_failure_diagnostic": record(diagnostic_path),
        "numeric_repair_addendum": record(addendum_path)}


def mixed_reexecution_identity(prior, current):
    """Compare metadata and SHA-bound array bytes; no scientific array IO or tolerance."""
    conditions, mismatches = [], []
    cp_count = native_arrays = Bolt_arrays = D_states = curve_count = 0
    def same(label, old, new):
        if old != new: mismatches.append(label)
    for dataset, (_, _, dev, _) in DATASETS.items():
        for space in ("PCA", "FIT_TARGET"):
            old = prior["datasets"][dataset]["spaces"][space]
            new = current["datasets"].get(dataset, {}).get("spaces", {}).get(space, {})
            prefix = dataset+"/"+space
            space_before = len(mismatches)
            same(prefix+"/status", old["status"], new.get("status"))
            if new.get("status") != "COMPLETE":
                conditions.extend({"dataset": dataset, "space": space, "seed": str(seed), "status": "MISMATCH_PRESERVED"} for seed in SEEDS)
                continue
            same(prefix+"/E_arrays", old["E_state_artifact"]["arrays"], new["E_state_artifact"]["arrays"])
            for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                a, b = old["backends"][backend], new["backends"][backend]
                same(prefix+"/"+backend+"/D_arrays", a["state_artifact"]["arrays"], b["state_artifact"]["arrays"])
                same(prefix+"/"+backend+"/D_metadata", a["metadata"], b["metadata"])
                D_states += 1
            for split in ("fit", "tune", dev):
                same(prefix+"/"+split+"/input_arrays", old["inputs"][split]["arrays"], new["inputs"][split]["arrays"])
                for score in ("scores_all_observed", "scores_complete_mask"):
                    same(prefix+"/"+split+"/Bolt/"+score, old["Bolt"][split][score], new["Bolt"][split][score])
                same(prefix+"/"+split+"/Bolt/arrays", old["Bolt"][split]["artifact"]["arrays"], new["Bolt"][split]["artifact"]["arrays"])
                Bolt_arrays += 1
            common_mismatch = len(mismatches) > space_before
            for seed in SEEDS:
                a, b = old["native"][str(seed)], new["native"][str(seed)]
                head_prefix = prefix+"/s"+str(seed); before = len(mismatches)
                same(head_prefix+"/metadata_without_wall", {k: v for k, v in a["metadata"].items() if k != "train_wall_seconds"},
                    {k: v for k, v in b["metadata"].items() if k != "train_wall_seconds"})
                same(head_prefix+"/learning_curve", a["learning_curve"], b["learning_curve"])
                curve_count += 1
                same(head_prefix+"/scores", a["scores"], b["scores"])
                same(head_prefix+"/CP_count", len(a["checkpoints"]), len(b["checkpoints"]))
                for old_cp, new_cp in zip(a["checkpoints"], b["checkpoints"]):
                    cp_count += 1
                    for key in ("update", "tune", "state_sha256"):
                        same(head_prefix+"/CP/"+str(old_cp["update"])+"/"+key, old_cp[key], new_cp[key])
                    for key in ("state_artifact", "tune_prediction_artifact"):
                        same(head_prefix+"/CP/"+str(old_cp["update"])+"/"+key+"/arrays", old_cp[key]["arrays"], new_cp[key]["arrays"])
                for split in ("fit", "tune", dev):
                    same(head_prefix+"/"+split+"/arrays", a["artifacts"][split]["arrays"], b["artifacts"][split]["arrays"])
                    native_arrays += 1
                conditions.append({"dataset": dataset, "space": space, "seed": str(seed),
                    "status": "PASS" if not common_mismatch and len(mismatches) == before else "MISMATCH_PRESERVED",
                    "selected_update_prior": a["metadata"]["selected_update"], "selected_update_new": b["metadata"]["selected_update"]})
    return {"status": "PASS" if not mismatches and len(conditions) == 8 else "MISMATCH_PRESERVED",
        "conditions": conditions, "mismatches": mismatches, "mixed_native_conditions_compared": len(conditions),
        "checkpoint_states_and_TUNE_arrays_compared": cp_count, "selected_native_split_arrays_compared": native_arrays,
        "mixed_Bolt_split_arrays_compared": Bolt_arrays, "global_D_states_compared": D_states,
        "learning_curves_compared": curve_count, "tolerance_used": False,
        "comparison_scope": "Exact JSON numeric equality and SHA/dtype/shape of logical arrays; archive container bytes need not equal",
        "prior_array_files_checked_by_preservation_guard": True,
        "new_heads_warm_started_or_reused": False, "two_runs_independent_conditions_claimed": False,
        "total_physical_native_fit_attempts_across_v1_v2": prior["native_training_fits"]+current["native_training_fits"],
        "distinct_registered_native_conditions": 12}

def resolved_locks(locks):
    expected = {(dataset, space, str(seed)) for dataset in DATASETS for space in SPACES for seed in SEEDS}
    require(set(locks) == expected, "ALL_TWELVE_NATIVE_CONDITIONS_RESOLVED_BEFORE_DEV")
    values = {}
    for key, binding in locks.items():
        require(record(binding["path"]) == binding, "NEW_NATIVE_SELECTION_LOCK_CHANGED")
        value = read(binding["path"])
        require((value["dataset"], value["space"], value["seed"]) == key
            and value["status"] in ("LOCKED", "FAILED_PATH_PRESERVED"), "COMPLETE_OR_EXPLICIT_FAILED_CONDITION_REQUIRED")
        if value["status"] == "LOCKED": require(value["selected_update"] in (0, 100, 200, 400, 600)
            and value["DEV_used_for_CP_selection"] is False, "FIXED_TUNE_ONLY_CP_LOCK")
        values[key] = value
    return values


def encode_numpy(state, indices, history):
    from raw_subset_projection_v2 import encode_subset
    return encode_subset(history, indices) if indices is not None else state.encode(history)


def encode_native(projection, indices, history):
    import numpy as np
    import torch
    pieces = []
    with torch.no_grad():
        for start in range(0, len(history), 32):
            x = torch.from_numpy(np.ascontiguousarray(history[start:start+32]))
            pieces.append(x.index_select(2, torch.tensor(indices, dtype=torch.int64)) if indices is not None else projection.encode(x))
    return np.ascontiguousarray(torch.cat(pieces).numpy(), dtype=np.float32)


def scores(prediction, values):
    import numpy as np
    from decoder_metric_probe_v1 import masked_scores
    mask = values["observed"]
    complete = np.broadcast_to(mask.all(axis=2, keepdims=True), mask.shape)
    def score(m):
        return masked_scores(prediction, values["target"], m, values["std"]) if m.any() else {"status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
    return {"scores_all_observed": score(mask), "scores_complete_mask": score(complete)}


def cached_future_perturbation(model, ctx):
    import numpy as np
    from raw_subset_native_head_v1 import predict_components
    raw = {key: value[:2].copy() for key, value in ctx["values"].items() if value.ndim and len(value) == len(ctx["z_native"])}
    changed = {key: value.copy() for key, value in raw.items()}
    changed["target"][:] = np.float32(1e6); changed["observed"][:] = False
    first, _ = predict_components(model, ctx["z_native"][:2], raw["raw_last"])
    second, _ = predict_components(model, ctx["z_native"][:2], changed["raw_last"])
    require(byte_equal(first, second), "FIXED_CACHED_NATIVE_FUTURE_LABEL_PERTURBATION")
    return {"target_and_future_mask_changed": True, "prediction_byte_exact": True, "windows": len(first),
        "scope": "Fixed cached past Z,currentC and FIT-global D only; no upstream raw-future replay or E/D refit"}


def execute(source_path, audit_path, output_path, prior_partial_path):
    source, bindings = load_gates(source_path, audit_path)
    prior, prior_bindings = load_prior_partial(prior_partial_path)
    bindings.update(prior_bindings)
    output = Path(output_path).resolve()
    require(output.parent == (HERE / "runs").resolve() and not output.exists(), "FRESH_RAW_SUBSET_RUN_DIRECTORY_REQUIRED")
    sys.path[:0] = [str(HERE), str(BASE), str(OLD), str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import raw_subset_projection_v2 as subset
    import raw_subset_native_head_v1 as native
    import run_condition_probes as probes
    import cost_readout_v1 as old_cost
    import matched_readout_contract_v1 as contract
    from projection_math_v1 import ProjectionState
    from native_head_v1 import FrozenProjection, state_hash, clone_state
    from run_fit_target_comparison_v1 import prior_file_records
    from run_matched_shrunk_native_cost_v2 import canonical_physical_bridge
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True, exist_ok=False)
    meter = rt.Meter("raw_subset_" + output.name, gpu=True)
    report = {"status": "RUNNING", **bindings, "datasets": {}, "failed_paths": [], "execution_events": [],
        "plan": record(HERE / "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md"),
        "subset_selection_fits": 0, "global_decoder_fits": 0, "native_training_fits": 0, "training_updates": 0,
        "checkpoint_states_saved": 0, "saved_native_split_predictions": 0, "saved_Bolt_split_predictions": 0,
        "actual_subset_F_split_inferences": 0, "reused_mixed_F_split_caches": 0, "resolved_native_lock_files": 0,
        "actual_subset_F_model_batch_calls": 0,
        "cached_future_label_perturbation_checks": 0,
        "protected_TEST_accessed": False, "independent_holdout": False, "new_backbone_training": False,
        "existing_head_weights_reused": False, "future_labels_used_for_subset_or_decoder_fit": False,
        "FIT_TARGET_E_already_FIT_future_supervised": True, "per_origin_decoder_fit": False,
        "all_twelve_locks_before_any_DEV_input_or_prediction": False, "whole_deployment_cost_measured": False,
        "cost_status": "NOT_MEASURED; whole cost is a separate stage with a newly frozen cohort",
        "new_output_space_diagnostic_status": "NOT_MEASURED; separate independent auditor must use each new fixed D32",
        "oracle_used_for_selection": False, "strict_K_only_compression": False,
        "source_snapshots": [], "individual_exact_human_method_approval_claimed": False,
        "hard_elapsed_or_cumulative_resource_caps": None}
    checked, guarded, contexts, states, models, locks, failure = {}, [], {}, {}, {}, {}, None
    started = time.perf_counter()
    def event(kind, **value):
        report["execution_events"].append({"sequence": len(report["execution_events"]), "kind": kind, **value}); meter.check({"stage": kind, **value})
    def failed(error, **value):
        row = {"status": "FAILED_PATH_PRESERVED", **value, "failure": {"type": type(error).__name__, "message": str(error)}}
        report["failed_paths"].append(row); return row
    def save(name, arrays):
        meter.check(); return io.save_npz(output / (name+".npz"), {key: np.ascontiguousarray(value) for key, value in arrays.items()})
    def load_input(dataset, split, count):
        c, _, _, _ = DATASETS[dataset]
        values = io.checked_payload(source["datasets"][dataset]["inputs"][split], meter, checked)
        require(values["history"].shape == (count, 512, c) and byte_equal(values["raw_last"], values["history"][:, -1]), "REGISTERED_RAW_PAST_AND_CURRENT_CONTRACT")
        fit = contexts[(dataset, "RAW", "fit")]
        contract.validate_raw_split(dataset, split, values, fit["mean"], fit["std"])
        return values
    def context(dataset, space, split, values):
        state, projection, indices = states[(dataset, space)]
        z_numpy, z_native = encode_numpy(state, indices, values["history"]), encode_native(projection, indices, values["history"])
        lineage = {}
        if indices is None:
            original = source["datasets"][dataset]["projections"][SPACES[space]]
            cached = io.checked_payload(original["bolt"]["artifacts"][split], meter, checked)
            require(byte_equal(z_numpy, cached["latent_history"]) and byte_equal(values["origins"], cached["origins"]), "EXACT_MIXED_NUMPY_E_AND_F_CACHE_LINEAGE")
            for seed in SEEDS:
                old = io.checked_payload(original["native"][str(seed)]["artifacts"][split], meter, checked)
                require(byte_equal(z_native, old["latent_history"]) and byte_equal(values["origins"], old["origins"]), "EXACT_MIXED_NATIVE_CACHED_PAST_LINEAGE")
            future = cached["latent_future"].copy()
            lineage = {"mixed_F_cache_reused": True, "original_F_cache": original["bolt"]["artifacts"][split],
                "original_previous_D_prediction_SHA": io.array_sha(cached["prediction"])}
            report["reused_mixed_F_split_caches"] += 1
        else:
            require(record(report["datasets"][dataset]["subset_selection_lock"]["path"]) == report["datasets"][dataset]["subset_selection_lock"], "FIT_SUBSET_LOCK_BEFORE_ACTUAL_K_F_INFERENCE")
            event("ACTUAL_NEW_SUBSET_F_INFERENCE", dataset=dataset, split=split)
            pieces = []
            with torch.no_grad():
                for start in range(0, len(z_numpy), 8):
                    meter.check(); pieces.append(bolt(engine.tensor(z_numpy[start:start+8])).detach().cpu().numpy().copy())
                    report["actual_subset_F_model_batch_calls"] += 1
            future = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)
            sliced = values["raw_forecast"][..., indices]
            lineage = {"mixed_F_cache_reused": False, "new_K_F_inference_executed": True,
                "full_C_forecast_slice_used_as_new_inference": False, "full_C_slice_diagnostic": {
                    "max_abs": float(np.max(np.abs(future-sliced))), "RMSE": float(np.sqrt(np.mean((future.astype(np.float64)-sliced)**2))),
                    "byte_equal": byte_equal(future, sliced), "diagnostic_only": True}}
            report["actual_subset_F_split_inferences"] += 1
        ctx = {"values": values, "z_numpy": z_numpy, "z_native": z_native, "future_Bolt": future, "lineage": lineage}
        contexts[(dataset, space, split)] = ctx
        return ctx
    def score_and_save(dataset, space, split, ctx):
        condition = report["datasets"][dataset]["spaces"][space]
        d = condition["backends"]["BOLT_NUMPY"]["decoder_FP32"]
        p = ctx["values"]["raw_last"][:, None]+(ctx["future_Bolt"]-ctx["z_numpy"][:, -1:])@d
        condition["Bolt"][split] = {**scores(p, ctx["values"]), **ctx["lineage"], "artifact": save(dataset+"_"+space+"_BOLT_"+split,
            {"prediction": p, "latent_history": ctx["z_numpy"], "latent_future": ctx["future_Bolt"], "origins": ctx["values"]["origins"]})}
        condition["inputs"][split] = save(dataset+"_"+space+"_FIXED_INPUTS_"+split, {"history": ctx["values"]["history"],
            "history_observed": ctx["values"]["history_observed"], "latent_history_BOLT_NUMPY": ctx["z_numpy"], "latent_history_NATIVE_TORCH32": ctx["z_native"],
            **{key: ctx["values"][key] for key in ("target", "observed", "raw_last", "origins", "mean", "std", "raw_forecast")}})
        report["saved_Bolt_split_predictions"] += 1
    def payload(ctx):
        return {"latent_history": ctx["z_native"], **{key: ctx["values"][key] for key in ("raw_last", "target", "observed", "origins")}}
    def train_one(dataset, space, seed):
        condition = report["datasets"][dataset]["spaces"][space]
        entry = {"status": "TRAINING", "checkpoints": [], "scores": {}, "artifacts": {}, "progress_events": []}
        condition["native"][str(seed)] = entry
        fit, tune = contexts[(dataset, space, "fit")], contexts[(dataset, space, "tune")]
        prefix = dataset+"_"+space+"_NATIVE_s"+str(seed); cp_predictions = {}
        def callback(update, state, metrics, prediction):
            row = {"update": update, "tune": metrics, "state_sha256": state_hash(state),
                "state_artifact": save(prefix+"_u"+str(update)+"_STATE", {name: value.numpy().copy() for name, value in state.items()}),
                "tune_prediction_artifact": save(prefix+"_u"+str(update)+"_TUNE", {"prediction": prediction, "origins": tune["values"]["origins"]})}
            entry["checkpoints"].append(row); cp_predictions[update] = prediction.copy(); report["checkpoint_states_saved"] += 1
        def progress(value): entry["progress_events"].append(value); meter.check({"dataset": dataset, "space": space, "seed": seed, **value})
        event("NATIVE_TRAIN_BEGIN", dataset=dataset, space=space, seed=str(seed))
        trained = native.train_native_fixed(condition["backends"]["NATIVE_TORCH32"]["decoder_FP32"], payload(fit), payload(tune), seed, progress, callback)
        report["native_training_fits"] += 1; report["training_updates"] += len(trained["learning_curve"])
        entry.update(metadata=trained["metadata"], learning_curve=trained["learning_curve"])
        model = trained["model"]
        for split, ctx in (("fit", fit), ("tune", tune)):
            p, f = native.predict_components(model, ctx["z_native"], ctx["values"]["raw_last"])
            if split == "tune": require(byte_equal(p, cp_predictions[entry["metadata"]["selected_update"]]), "SELECTED_CP_SCORED_AND_SAVED_TUNE_EXACT")
            entry["scores"][split] = scores(p, ctx["values"])
            entry["artifacts"][split] = save(prefix+"_"+split, {"prediction": p, "latent_future": f, "origins": ctx["values"]["origins"]})
            report["saved_native_split_predictions"] += 1
        chosen = next(row for row in entry["checkpoints"] if row["update"] == entry["metadata"]["selected_update"])
        entry["status"] = "COMPLETE_FIT_TUNE_SEALED"; models[(dataset, space, str(seed))] = model
        return {"status": "LOCKED", "dataset": dataset, "space": space, "seed": str(seed), "selected_update": chosen["update"],
            "selected_state_artifact": chosen["state_artifact"], "selected_TUNE_prediction_artifact": chosen["tune_prediction_artifact"],
            "selected_tune": chosen["tune"], "DEV_used_for_CP_selection": False, "sealed_unix": time.time()}
    try:
        report["protected_guard_paths"] = install_protected_test_guard(); meter.protected_check()
        guarded = prior_file_records(source, prior, read(audit_path), *bindings.values())
        folder = output / "sources"; folder.mkdir()
        for path in (Path(__file__), HERE / "raw_subset_projection_v2.py", HERE / "raw_subset_projection_v1.py",
                     HERE / "run_raw_subset_comparison_v1.py", HERE / "RAW_SUBSET_GRAM_NUMERIC_REPAIR_KO_v2.md",
                     HERE / "raw_subset_native_head_v1.py", HERE / "native_head_v1.py",
                     HERE / "projection_math_v1.py", HERE / "fixed_decoder_span_v1.py", HERE / "decoder_metric_probe_v1.py",
                     HERE / "run_matched_shrunk_native_cost_v2.py", HERE / "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md", HERE / "continuation_runtime_v1.py",
                     BASE / "cost_readout_v1.py", OLD / "run_condition_probes.py", OLD / "prepare_inputs.py", OLD / "matched_readout_contract_v1.py"):
            target = folder / path.name
            with path.open("rb") as a, target.open("xb") as b: shutil.copyfileobj(a, b)
            report["source_snapshots"].append({**record(path), "snapshot_path": str(target)})
        engine = probes.import_old_engine()
        weight, config = (probes.verified_file(source[key]) for key in ("model_binary_before", "model_config_before"))
        report.update(model_binary_before=weight, model_config_before=config)
        bolt = engine.ex.previous.FrozenBolt(); bolt.model.eval().requires_grad_(False)
        report["F_before"] = probes.model_hashes(bolt)
        require(report["F_before"] == source["F_before"], "ORIGINAL_FROZEN_BOLT_DUAL_SIGNATURE")
        report["runtime_behavior_contract"] = old_cost.bolt_runtime_contract(engine, bolt, meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        with threadpool_limits(limits=2):
            for dataset, (c, k, dev, dev_count) in DATASETS.items():
                row = {"channels": c, "K": k, "development_split": dev, "spaces": {}, "raw_scores": {},
                    "raw_input_sources": source["datasets"][dataset]["inputs"]}; report["datasets"][dataset] = row
                try:
                    fit = io.checked_payload(source["datasets"][dataset]["inputs"]["fit"], meter, checked)
                    contexts[(dataset, "RAW", "fit")] = fit
                    fit = load_input(dataset, "fit", 256); tune = load_input(dataset, "tune", 64)
                except Exception as error:
                    row["dataset_input_failure"] = failed(error, dataset=dataset, stage="FIT_TUNE_INPUT_BINDING")
                    for space in SPACES:
                        row["spaces"][space] = {"status": "FAILED_PATH_PRESERVED", "backends": {}, "native": {}}
                        for seed in SEEDS:
                            lock = failed(error, dataset=dataset, space=space, seed=str(seed), stage="FIT_TUNE_DEPENDENT_CONDITION")
                            row["spaces"][space]["native"][str(seed)] = lock
                            path = output / (dataset+"_"+space+"_s"+str(seed)+"_SELECTION_LOCK.json")
                            rt.write_new(path, lock); locks[(dataset, space, str(seed))] = record(path)
                            report["resolved_native_lock_files"] += 1
                    continue
                for split, values in (("fit", fit), ("tune", tune)): row["raw_scores"][split] = scores(values["raw_forecast"], values)
                try:
                    selection = subset.greedy_raw_subset(fit["history"], fit["history_observed"], k)
                    report["subset_selection_fits"] += 1
                    proof = canonical_physical_bridge(source, dataset)
                    row["canonical_physical_bridge_binding"] = proof
                    row["physical_bridge_canonical"] = proof["canonical_record"]
                    bridge = read(proof["canonical_input_bridge"]["path"])
                    fit_records = [item for item in bridge["artifacts"] if item["dataset"] == dataset and item["split"] == "fit"]
                    require(len(fit_records) == 1, "ONE_CANONICAL_FIT_BRIDGE_FOR_ORIGINAL_CHANNEL_NAMES")
                    physical_fit = io.checked_payload(fit_records[0], meter, checked)
                    require(byte_equal(physical_fit["history"], fit["history"]) and byte_equal(physical_fit["history_observed"], fit["history_observed"]), "CHANNEL_NAMES_FROM_SAME_FIT_INPUT")
                    names = physical_fit["columns"].tolist()
                    lock_path = output / (dataset+"_SUBSET_SELECTION_LOCK.json")
                    rt.write_new(lock_path, {"status": "LOCKED_FIT_PAST_ONLY", "selected_indices": selection["selected_indices"],
                        "selection_order": selection["selection_order"], "selected_channel_names": [names[index] for index in selection["selected_indices"]],
                        "selection_metadata": selection["metadata"], "FIT_source": source["datasets"][dataset]["inputs"]["fit"]})
                    row["subset_selection_lock"] = record(lock_path)
                except Exception as error:
                    selection = None; row["subset_selection_failure"] = failed(error, dataset=dataset, stage="FIT_PAST_SUBSET_SELECTION")
                for space, source_space in SPACES.items():
                    condition = {"status": "RUNNING", "backends": {}, "Bolt": {}, "native": {}, "inputs": {}}; row["spaces"][space] = condition
                    try:
                        if source_space is None:
                            require(selection is not None, "SUBSET_SELECTION_REQUIRED")
                            indices = selection["selected_indices"]
                            state = ProjectionState(selection["center_FP32"], selection["encoder_FP32"], np.zeros((k, c)), np.zeros(c), selection["metadata"])
                            condition["subset_selection_lock"] = row["subset_selection_lock"]
                        else:
                            indices = None; ref = source["datasets"][dataset]["projections"][source_space]["state_artifact"]
                            state = ProjectionState.from_npz(io.checked_payload(ref, meter, checked)); condition["original_E_state"] = ref
                        states[(dataset, space)] = (state, FrozenProjection(state).cpu().eval().requires_grad_(False), indices)
                        condition["fixed_E_hashes_before"] = {"numpy_center": io.array_sha(np.asarray(state.center)),
                            "numpy_encoder": io.array_sha(np.asarray(state.encoder)),
                            "native_projection_state": state_hash(clone_state(states[(dataset, space)][1]))}
                        condition["E_state_artifact"] = save(dataset+"_"+space+"_E", {"center": state.center, "encoder": state.encoder,
                            "selected_indices": np.array(indices if indices is not None else [], dtype=np.int64)})
                        for split, values in (("fit", fit), ("tune", tune)): context(dataset, space, split, values)
                        for backend, member in (("BOLT_NUMPY", "z_numpy"), ("NATIVE_TORCH32", "z_native")):
                            fitted = subset.fit_global_observed_decoder(fit["history"], fit["history_observed"], contexts[(dataset, space, "fit")][member])
                            report["global_decoder_fits"] += 1
                            condition["backends"][backend] = {"metadata": fitted["metadata"], "decoder_FP32": fitted["decoder_FP32"],
                                "state_artifact": save(dataset+"_"+space+"_"+backend+"_GLOBAL_D", {key: fitted[key] for key in ("decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32")})}
                        for split in ("fit", "tune"): score_and_save(dataset, space, split, contexts[(dataset, space, split)])
                    except Exception as error: condition.update(failed(error, dataset=dataset, space=space, stage="FIXED_E_AND_FIT_GLOBAL_D"))
                    for seed in SEEDS:
                        try:
                            require(condition["status"] == "RUNNING", "SUCCESSFUL_FIXED_D_CONDITION_REQUIRED")
                            lock = train_one(dataset, space, seed)
                        except Exception as error:
                            lock = failed(error, dataset=dataset, space=space, seed=str(seed), stage="NATIVE_FIT_TUNE")
                            condition["native"].setdefault(str(seed), {}).update(status="FAILED_PATH_PRESERVED", failure=lock["failure"])
                        path = output / (dataset+"_"+space+"_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path, lock)
                        locks[(dataset, space, str(seed))] = record(path); report["resolved_native_lock_files"] += 1
            resolved_locks(locks)
            rt.write_new(output / "SELECTION_REGISTRY.json", {"status": "ALL_TWELVE_RESOLVED_BEFORE_DEV", "conditions": [
                {"dataset": key[0], "space": key[1], "seed": key[2], "lock": value} for key, value in locks.items()], "failed_paths_at_seal": report["failed_paths"]})
            report["selection_registry"] = record(output / "SELECTION_REGISTRY.json"); report["all_twelve_locks_before_any_DEV_input_or_prediction"] = True
            event("ALL_TWELVE_LOCKS_AND_REGISTRY_SAVED", successful_heads=len(models))
            for dataset, (c, k, dev, count) in DATASETS.items():
                if report["datasets"][dataset].get("dataset_input_failure") is not None: continue
                try:
                    resolved_locks(locks); event("DEV_INPUT_LOAD_AFTER_ALL_TWELVE_LOCKS", dataset=dataset)
                    values = load_input(dataset, dev, count); report["datasets"][dataset]["raw_scores"][dev] = scores(values["raw_forecast"], values)
                except Exception as error: failed(error, dataset=dataset, stage="DEV_INPUT"); continue
                for space in SPACES:
                    condition = report["datasets"][dataset]["spaces"][space]
                    if condition["status"] != "RUNNING": continue
                    try:
                        ctx = context(dataset, space, dev, values); score_and_save(dataset, space, dev, ctx)
                        for seed in SEEDS:
                            key = (dataset, space, str(seed))
                            if key not in models: continue
                            entry = condition["native"][str(seed)]
                            try:
                                p, f = native.predict_components(models[key], ctx["z_native"], values["raw_last"])
                                entry["scores"][dev] = scores(p, values); entry["artifacts"][dev] = save(dataset+"_"+space+"_NATIVE_s"+str(seed)+"_"+dev,
                                    {"prediction": p, "latent_future": f, "origins": values["origins"]})
                                entry["future_label_perturbation"] = cached_future_perturbation(models[key], ctx)
                                entry["status"] = "COMPLETE"; report["saved_native_split_predictions"] += 1
                                report["cached_future_label_perturbation_checks"] += 1
                            except Exception as error:
                                entry.update(status="PARTIAL_COMPLETED", development_failure=failed(error, dataset=dataset,
                                    space=space, seed=str(seed), stage="DEV_NATIVE_PREDICTION"))
                        condition["status"] = "COMPLETE" if all(item["status"] == "COMPLETE" for item in condition["native"].values()) else "PARTIAL_COMPLETED"
                    except Exception as error: condition.update(failed(error, dataset=dataset, space=space, stage="DEV_SPACE_PREDICTION"))
        report["mixed_reexecution_identity"] = mixed_reexecution_identity(prior, report)
        report["F_after"] = probes.model_hashes(bolt); require(report["F_before"] == report["F_after"], "FROZEN_BOLT_CHANGED")
        for (dataset, space), (state, projection, indices) in states.items():
            condition = report["datasets"][dataset]["spaces"][space]
            condition["fixed_E_hashes_after"] = {"numpy_center": io.array_sha(np.asarray(state.center)),
                "numpy_encoder": io.array_sha(np.asarray(state.encoder)), "native_projection_state": state_hash(clone_state(projection))}
            require(condition["fixed_E_hashes_before"] == condition["fixed_E_hashes_after"], "FIXED_E_CHANGED")
        report["model_binary_after"], report["model_config_after"] = probes.verified_file(weight), probes.verified_file(config)
        subset_locks = [row["subset_selection_lock"] for row in report["datasets"].values() if "subset_selection_lock" in row]
        for binding in guarded+list(checked.values())+list(locks.values())+subset_locks+[report["selection_registry"]]:
            require(record(binding["path"])["sha256"] == binding["sha256"], "PROTECTED_SOURCE_CACHE_OR_SELECTION_CHANGED")
        for binding in report["source_snapshots"]:
            require(record(binding["path"])["sha256"] == record(binding["snapshot_path"])["sha256"] == binding["sha256"], "EXECUTED_SOURCE_CHANGED")
        meter.protected_check(); report["protected_prior_records"] = guarded; report["input_records"] = list(checked.values())
        expected = {"subset_selection_fits": 2, "global_decoder_fits": 12, "native_training_fits": 12, "training_updates": 7200,
            "checkpoint_states_saved": 60, "saved_native_split_predictions": 36, "saved_Bolt_split_predictions": 18,
            "actual_subset_F_split_inferences": 6, "reused_mixed_F_split_caches": 12, "resolved_native_lock_files": 12,
            "cached_future_label_perturbation_checks": 12}
        report["completeness"] = {"expected": expected, "actual": {key: report[key] for key in expected}}
        require(report["mixed_reexecution_identity"]["status"] == "PASS", "MIXED_REEXECUTION_IDENTITY_DIFFERENCE_PRESERVED")
        report["status"] = "COMPLETE" if not report["failed_paths"] and report["completeness"]["actual"] == expected else "PARTIAL_COMPLETED"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        for row in report["datasets"].values():
            for condition in row["spaces"].values():
                for backend in condition["backends"].values(): backend.pop("decoder_FP32", None)
        report["wall_seconds"] = time.perf_counter()-started; report["resource_terminal"] = meter.close(report["status"], failure)
        rt.write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "result": str(output / "RESULT.json"), "native_fits": report["native_training_fits"], "failure": failure}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-result", "source-audit", "output", "prior-partial"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.source_result, args.source_audit, args.output, args.prior_partial))
