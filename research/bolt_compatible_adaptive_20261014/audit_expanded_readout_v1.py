"""Independent verification of the single gated Robin FIT512 diagnostic."""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
np = sys.modules.get("numpy")


def deterministic_legal_origins(bounds, maximum=512):
    lower, upper = bounds
    count = upper - 48 - (lower + 512) + 1
    if count <= 0:
        raise ValueError("NO_LEGAL_WITHIN_FIT_WINDOWS")
    if count <= maximum:
        return np.arange(lower + 512, upper - 48 + 1, dtype=np.int64)
    # Integer arithmetic independently encodes equally spaced floor indices.
    positions = (np.arange(maximum, dtype=np.int64) * (count - 1)) // (maximum - 1)
    return (lower + 512 + positions).astype(np.int64)


def reconstructed_windows(physical, observed, origins, mean, std):
    from audit_cost_v1 import scalar_preprocess
    past = np.stack([physical[origin - 512:origin] for origin in origins]).astype(np.float32)
    past_mask = np.stack([observed[origin - 512:origin] for origin in origins]).astype(np.bool_)
    future = np.stack([physical[origin:origin + 48] for origin in origins]).astype(np.float32)
    future_mask = np.stack([observed[origin:origin + 48] for origin in origins]).astype(np.bool_)
    history = scalar_preprocess(past, past_mask, mean, std)
    safe_future = np.where(future_mask, future, mean).astype(np.float32)
    target = ((safe_future - mean) / std).astype(np.float32)
    return {"history": history, "history_observed": past_mask, "target": target, "observed": future_mask,
            "origins": origins, "raw_last": history[:, -1].copy(), "mean": mean, "std": std}


def audit_result(path, limit):
    import audit_readout_v2 as audit
    check = limit.shared.checkfast
    record = {"path": str(audit.resolve(path)), "sha256": audit.file_sha(audit.resolve(path))}
    result = audit.verified_json(record, check)
    audit.require(result["status"] == "COMPLETE" and result["phase"] == "MATCHED_FIT512_ONE_DIAGNOSTIC"
                  and result["dataset"] == "robin", "SINGLE_GATED_ROBIN_EXPANDED_DIAGNOSTIC_REQUIRED")
    audit.require(result["permission_gate"] == limit.authorization_record and result["run_manifest"] == limit.run_manifest_record,
                  "SAME_APPROVED_CAMPAIGN_REQUIRED")
    audit.require(result["protected_TEST_accessed"] is False and result["GPU_execution"] is False
                  and result["model_weights_loaded"] is False and result["old_campaign_restart"] is False
                  and result["independent_holdout"] is False and result["overlapping_training_windows_not_independent_examples"] is True,
                  "EXPANDED_SCOPE_DECLARATION_REQUIRED")
    first = audit.verified_json(result["first256_result"], check)
    audit.require(first["status"] == "COMPLETE" and first["dataset"] == "robin"
                  and first["branch_lock"]["next_branch"] == "MATCHED_FIT512_ONE_DIAGNOSTIC", "INITIAL_FIT_TUNE_OVERFIT_GATE_REQUIRED")
    phase0 = json.loads((HERE / "ENVIRONMENT_ASSET_MANIFEST_v1.json").read_text(encoding="utf-8-sig"))
    locked = next(value for value in phase0["dataset_metadata"] if value["dataset"] == "robin")
    contract = audit.verified_json({"path": locked["contract_path"], "sha256": locked["contract_SHA256"]}, check)
    bounds = contract["bounds"]["fit"]
    source_path = audit.resolve(contract["source"])
    audit.require(audit.file_sha(source_path) == contract["source_sha256"], "ORIGINAL_RAW_TRAINVAL_SOURCE_BYTES_MISMATCH")
    check()
    with np.load(source_path, allow_pickle=False) as source:
        physical = (source["x"][:bounds[1]] * source["std"] + source["mean"]).astype(np.float32)
        observed = source["finite"][:bounds[1]].copy()
    physical[~observed] = np.nan
    original_fit = audit.verified_arrays(first["inputs"]["fit"], check)
    mean, std = original_fit["mean"], original_fit["std"]
    reconstructed_first = reconstructed_windows(physical, observed, original_fit["origins"], mean, std)
    for name, values in reconstructed_first.items():
        audit.exact(values, original_fit[name], "independent_original256_reconstruction:" + name)
    expected_origins = deterministic_legal_origins(bounds)
    raw = {split: audit.verified_arrays(result["inputs"][split], check) for split in ("fit", "tune", "val")}
    audit.require(set(raw["fit"]) == set(reconstructed_first) and "raw_forecast" not in raw["fit"],
                  "NEW_FIT_RAW_BOLT_FORECAST_MUST_NOT_BE_INVENTED")
    reconstructed_dense = reconstructed_windows(physical, observed, expected_origins, mean, std)
    for name, values in reconstructed_dense.items():
        audit.exact(values, raw["fit"][name], "independent_dense512_reconstruction:" + name)
    audit.require(result["legal_FIT_bounds"] == bounds and result["expanded_FIT_origin_count"] == len(expected_origins) == 512
                  and result["expanded_FIT_origin_first_last"] == [int(expected_origins[0]), int(expected_origins[-1])]
                  and result["original256_reconstruction_exact_pass"] is True
                  and result["original_FIT_statistics_unchanged"] is True, "EXPANDED_FIT_CONTRACT_METADATA_MISMATCH")
    overlap = int(np.intersect1d(expected_origins, original_fit["origins"]).size)
    audit.require(result["expanded_and_original_origin_overlap_count"] == overlap, "ORIGINAL_DENSE_ORIGIN_OVERLAP_MISMATCH")
    for split in ("tune", "val"):
        prior = audit.verified_arrays(first["inputs"][split], check)
        for key in prior:
            audit.exact(raw[split][key], prior[key], "unchanged_original_tune_dev:" + split + ":" + key)
        audit.check_raw_split(raw[split], expected_channels=17, expected_origins=locked["origins"][split])
        audit.compare_scores(result["references"][split]["RAW_BOLT"],
                             audit.scores(raw[split]["raw_forecast"], raw[split]["target"], raw[split]["observed"], std), "RAW_BOLT:" + split)
    source_checks = 0
    for source in result["source_records"] + result["source_snapshots"]:
        check()
        audit.require(audit.file_sha(audit.resolve(source["path"])) == source["sha256"], "EXPANDED_DECLARED_SOURCE_BYTES_CHANGED")
        source_checks += 1
        if "snapshot_path" in source:
            audit.require(audit.file_sha(audit.resolve(source["snapshot_path"])) == source["sha256"], "EXPANDED_SOURCE_SNAPSHOT_CHANGED")
            source_checks += 1
    shared = audit.verified_arrays(result["shared_mean_artifact"], check)
    shared_mean, shared_count = audit.fit_mean_innovation(raw["fit"]["target"], raw["fit"]["raw_last"], raw["fit"]["observed"])
    audit.exact(shared["innovation"], shared_mean, "new_FIT512_shared_FP32_mean")
    audit.exact(shared["observed_counts"], shared_count, "new_FIT512_mean_counts")
    controls = {(kind, split): (np.repeat(values["raw_last"][:, None], 48, axis=1) if kind == "PERSIST"
                              else values["raw_last"][:, None] + shared_mean)
                for split, values in raw.items() for kind in ("PERSIST", "MEAN")}
    routes = ("SELF33", "JOINT33")
    audit.require(set(result["families"]) == set(routes), "SAME_TWO_FIT512_ROUTES_REQUIRED")
    selected, summaries = {}, {}
    pool_count, candidate_prediction_count, selected_prediction_count = 0, 0, 0
    for route in routes:
        limit.checkfull()
        state_meta = result["feature_states"][route]
        state = audit.verified_arrays(state_meta["artifact"], check)
        center, scale = audit.expected_feature_state(route, raw["fit"]["history"])
        audit.exact(state["center"], center, "dense_FIT_center:" + route)
        audit.exact(state["scale"], scale, "dense_FIT_scale:" + route)
        audit.require(state_meta["fit_rows"] == 512 and state_meta["input_channels"] == 17
                      and state_meta["fit_raw_features_sha256"] == audit.array_sha(audit.raw_features(route, raw["fit"]["history"])),
                      "DENSE_FIT_FEATURE_STATE_CONTENT_MISMATCH")
        digest = audit.feature_state_digest(route, state_meta, state)
        audit.require(digest == state_meta["sha256"], "DENSE_FEATURE_STATE_DIGEST_MISMATCH")
        design = audit.independent_design(route, raw["fit"]["history"], center, scale)
        family = result["families"][route]
        audit.require(family["status"] == "COMPLETE" and family["new_training"] is True
                      and tuple(entry["alpha"] for entry in family["alpha_grid"]) == audit.ALPHAS,
                      "DENSE_TWO_NEW_SEVEN_ALPHA_MODELS_REQUIRED")
        candidates, equations = {}, []
        for index, entry in enumerate(family["alpha_grid"]):
            check()
            coefficients = audit.verified_arrays(entry["coefficient_artifact"], check)
            audit.require(set(coefficients) == set(audit.SOLUTION_KEYS), "DENSE_COEFFICIENT_KEYS_REQUIRED")
            metadata = entry["model_metadata"]
            audit.require(metadata["alpha"] == entry["alpha"] and metadata["feature_state_sha256"] == digest
                          and audit.model_digest(route, digest, metadata, coefficients) == metadata["model_sha256"] == entry["model_sha256"],
                          "DENSE_MODEL_DIGEST_BINDING_MISMATCH")
            counts = raw["fit"]["observed"].sum(axis=0)
            channel_counts = counts.sum(axis=0)
            audit.exact(coefficients["observed_counts"], counts, "dense_output_counts")
            audit.exact(coefficients["channel_counts"], channel_counts, "dense_channel_counts")
            audit.exact(coefficients["normal_equation_denominators"], np.broadcast_to(channel_counts, counts.shape).astype(np.float64), "dense_alpha_Nc")
            equations.append(audit.normal_equation_checks(route, design, raw["fit"]["target"], raw["fit"]["raw_last"],
                              raw["fit"]["observed"], coefficients["coefficients"], coefficients["intercept"], entry["alpha"]))
            for split in ("fit", "tune"):
                saved = audit.verified_arrays(entry[split + "_artifact"], check)
                audit.exact(saved["origins"], raw[split]["origins"], "dense_candidate_origins")
                prediction = audit.prediction_from_saved(route, raw[split]["history"], raw[split]["raw_last"], center, scale,
                                                        coefficients["coefficients"], coefficients["intercept"])
                audit.close(saved["prediction"], prediction, "dense_saved_prediction:" + route)
                audit.compare_scores(entry[split + "_scores"], audit.scores(saved["prediction"], raw[split]["target"], raw[split]["observed"], std), "dense_candidate_scores")
                candidates[index, split] = saved["prediction"]
                candidate_prediction_count += 1
            audit.close(entry["tune_actual_Y_mse"], entry["tune_scores"]["normalized"]["mse"], "dense_primary_alias", rtol=0, atol=0)
        for index, entry in enumerate(family["pool"]):
            for split in ("fit", "tune"):
                prediction = controls[entry["kind"], split] if index < 2 else candidates[index - 2, split]
                audit.compare_scores(entry[split + "_scores"], audit.scores(prediction, raw[split]["target"], raw[split]["observed"], std), "dense_nine_pool_score")
            pool_count += 1
        index = audit.select_pool(family["pool"])
        entry = family["pool"][index]
        audit.require(family["selected_index"] == index and family["selected_kind"] == entry["kind"]
                      and family["selected_alpha"] == entry.get("alpha"), "DENSE_GLOBAL_SELECTION_MISMATCH")
        ordered = sorted(row["tune_scores"]["normalized"]["mse"] for row in family["pool"])
        audit.close(family["tune_selection_margin_mse"], ordered[1] - ordered[0], "dense_selection_margin", rtol=1e-12, atol=1e-12)
        for split in ("fit", "tune", "val"):
            saved = audit.verified_arrays(family["selected_artifacts"][split], check)
            for key in ("target", "observed", "origins", "raw_last", "mean", "std"):
                audit.exact(saved[key], raw[split][key], "dense_selected_input_contract:" + key)
            if index < 2:
                prediction = controls[entry["kind"], split]
            elif split in ("fit", "tune"):
                prediction = candidates[index - 2, split]
            else:
                coefficients = audit.verified_arrays(entry["coefficient_artifact"], check)
                prediction = audit.prediction_from_saved(route, raw[split]["history"], raw[split]["raw_last"], center, scale,
                                                        coefficients["coefficients"], coefficients["intercept"])
            audit.close(saved["prediction"], prediction, "dense_selected_prediction")
            audit.compare_scores(family["splits"][split], audit.scores(saved["prediction"], raw[split]["target"], raw[split]["observed"], std), "dense_selected_scores")
            selected[route, split] = saved["prediction"]
            selected_prediction_count += 1
        if entry["kind"] == "RIDGE":
            audit.require(family["selected_model_metadata"] == entry["model_metadata"] and family["selected_coefficient_artifact"] == entry["coefficient_artifact"],
                          "DENSE_SELECTED_MODEL_BINDING_MISMATCH")
        summaries[route] = {"selected_index": index, "selected_kind": entry["kind"], "selected_alpha": entry.get("alpha"), "normal_equations": equations}
    tune = raw["tune"]
    blocks = {route: audit.chronological_blocks(selected[route, "tune"], tune["target"], tune["observed"], tune["origins"], std) for route in routes}
    independent_gate = audit.common_information_gate(result["families"]["SELF33"]["splits"]["tune"], result["families"]["JOINT33"]["splits"]["tune"], blocks["SELF33"], blocks["JOINT33"])
    lock = audit.verified_json(result["branch_lock"], check)
    audit.require(lock["status"] == "LOCKED_BEFORE_NEW_DEV_SCORING" and lock["scientific_inputs_only"] == ["FIT", "TUNE"]
                  and lock["new_DEV_scored_before_lock"] is False and lock["new_features_and_targets_constructed_from_FIT_only"] is True,
                  "DENSE_FIT_TUNE_BRANCH_LOCK_REQUIRED")
    gate = lock["common_information_gate"]
    audit.require(gate["passed"] == independent_gate["passed"] and gate["ratio_eligible"] == (independent_gate["status"] != "INCONCLUSIVE"),
                  "DENSE_COMMON_GATE_MISMATCH")
    audit.close(gate["mse_gain_pct"], independent_gate["mse_relative_reduction"] * 100, "dense_MSE_gate_ratio", rtol=1e-11, atol=1e-11)
    audit.close(gate["mae_change_pct"], independent_gate["mae_relative_increase"] * 100, "dense_MAE_gate_ratio", rtol=1e-11, atol=1e-11)
    for index, rows in enumerate(np.array_split(np.arange(len(tune["origins"])), 4)):
        entry = gate["four_chronological_blocks"][index]
        audit.require(entry["indexes"] == rows.tolist(), "DENSE_CHRONOLOGICAL_BLOCK_INDEX_MISMATCH")
        own, joint = blocks["SELF33"][index]["scores"]["normalized"]["mse"], blocks["JOINT33"][index]["scores"]["normalized"]["mse"]
        audit.close(entry["self_mse"], own, "dense_SELF_block", rtol=1e-12, atol=1e-12)
        audit.close(entry["joint_mse"], joint, "dense_JOINT_block", rtol=1e-12, atol=1e-12)
        audit.require(entry["joint_improved"] == (joint < own), "DENSE_BLOCK_IMPROVEMENT_MISMATCH")
    expected_branch = "SMALLK_PRESERVATION_COMPARISON" if independent_gate["passed"] else "STOP_COMMON_REPRESENTATION_NO_GAIN_AFTER_MATCHED_FIT512"
    audit.require(lock["next_branch"] == expected_branch and result["pool_comparisons"] == pool_count == 18
                  and result["new_model_bundles"] == 14 and result["reused_model_bundles"] == 0, "DENSE_ADAPTIVE_BRANCH_AND_EXECUTION_COUNTS_MISMATCH")
    limit.checkfull()
    return {"status": "PASS", "producer_record": {"dataset": "robin", **record}, "first256_producer_record": result["first256_result"],
            "pool_candidates_checked": pool_count, "candidate_predictions_checked": candidate_prediction_count,
            "selected_predictions_checked": selected_prediction_count, "new_model_bundles": 14,
            "source_file_bindings_checked": source_checks, "routes": summaries, "independent_common_gate": independent_gate,
            "independent_chronological_blocks": blocks, "next_branch": expected_branch,
            "original256_and_new512_reconstructed_from_original_FIT_prefix": True, "original_dense_origin_overlap": overlap,
            "scope": {"original_raw_preprocessing_reconstructed": True, "actual_future_targets_FIT_only": True,
                      "E_F_D_forward_replayed": False, "alpha0_minimum_norm_reproved": False,
                      "protected_TEST_accessed": False, "independent_holdout": False,
                      "read_order_proved_by_file_bytes": False, "old_exposed_development_rows_in_raw_member": True}}


def main():
    global np
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    sys.path[:0] = [str(HERE), str(ROOT / "research/smallk_failure_followup_20261014"), str(ROOT)]
    from adaptive_runtime_v1 import ContinuationLimit
    limit = ContinuationLimit(args.tag, gpu=False)
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    report = {"status": "RUNNING", "scientific_execution": True}
    error = None
    try:
        import numpy as np
        import audit_readout_v2
        import audit_cost_v1
        audit_readout_v2.initialize_numpy()
        audit_cost_v1.initialize_numpy()
        audit_readout_v2.require(output.is_relative_to(HERE) and not output.exists(), "NEW_CAMPAIGN_AUDIT_OUTPUT_ONLY")
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report.update(audit_result(args.input_result, limit))
    except Exception as failure:
        import traceback
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        report.update(status="FAIL", failure=error)
    finally:
        terminal = limit.close("COMPLETE" if report["status"] == "PASS" else "FAILED_PRESERVED", error)
        report["resource_terminal"] = terminal
        if terminal["terminal_status"] != "COMPLETE":
            report["status"] = "FAIL"
        if output.is_relative_to(HERE) and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"audit": str(output), "status": report["status"], "pool_candidates_checked": report.get("pool_candidates_checked"), "failure": error}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
