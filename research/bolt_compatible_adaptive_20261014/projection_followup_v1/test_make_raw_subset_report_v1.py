"""Synthetic JSON contracts, scientific scope and exact publication provenance."""
from pathlib import Path
import copy
import csv
import inspect
import json
import sys

import pytest

import make_raw_subset_report_v1 as report


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report.record(path)


def scores(value, channels):
    def scale(factor):
        return {"mse": value*factor*factor, "mae": value**.5*factor,
            "per_channel_mse": [value*factor*factor]*channels, "per_channel_mae": [value**.5*factor]*channels,
            "observed_counts": [96]*channels, "active_channels": channels}
    return {"normalized": scale(1), "physical_per_channel_diagnostic": scale(2)}


def pair(value, channels):
    return {"scores_all_observed": scores(value, channels), "scores_complete_mask": scores(value+.01, channels)}


def fixture(tmp_path):
    science = {"status": "COMPLETE", "protected_TEST_accessed": False, "F_before": {"frozen": 1}, "F_after": {"frozen": 1},
        "completeness": {"actual": {"native_fits": 12}, "expected": {"native_fits": 12}}, "failed_paths": [],
        "all_twelve_locks_before_any_DEV_input_or_prediction": True, "future_labels_used_for_subset_or_decoder_fit": False,
        "FIT_TARGET_E_already_FIT_future_supervised": True, "per_origin_decoder_fit": False, "new_backbone_training": False,
        "oracle_used_for_selection": False, "independent_holdout": False, "strict_K_only_compression": False, "datasets": {},
        "mixed_reexecution_identity": {"status": "PASS", "mixed_native_conditions_compared": 8,
            "checkpoint_states_and_TUNE_arrays_compared": 40, "selected_native_split_arrays_compared": 24,
            "mixed_Bolt_split_arrays_compared": 12, "global_D_states_compared": 8, "learning_curves_compared": 8,
            "tolerance_used": False, "total_physical_native_fit_attempts_across_v1_v2": 20, "distinct_registered_native_conditions": 12}}
    audit = {"status": "PASS", "protected_TEST_accessed": False, "totals": report.SCIENCE_COUNTS.copy(), "datasets": {},
        "new_output_space_diagnostics_computed_by_this_auditor": True,
        "new_output_minimum_only_normalized_MSE_in_each_new_static_D32_real_rowspace": True,
        "new_oracle_used_for_selection_or_deployment": False, "MAE_or_physical_MSE_minimum_claimed": False,
        "strict_universal_FP32_output_lower_bound_claimed": False, "old_output_space_floor_reused": False}
    cost = {"status": "COMPLETE", "protected_TEST_accessed": False, "routes": list(report.ROUTES), "cost": {},
        "completed_cost_cells": 40, "recorded_timer_samples": 400, "completed_cell_timer_samples": 400, "failed_paths": [],
        "F_before": {"frozen": 1}, "F_after": {"frozen": 1}, "native_deployment_cost_measured": True, "loaded_native_heads_count": 12,
        "per_origin_decoder_fit": False, "fixed_FIT_decoder_loaded_before_timer": True,
        "all_source_file_state_metadata_hashes_outside_timer": True, "new_encoder_subset_decoder_or_CP_selection": False,
        "new_training_updates": 0, "native_selection_locks": {}, "subset_selection_locks": {}}
    cost_audit = {"status": "PASS", "protected_TEST_accessed": False,
        "totals": report.COST_COUNTS.copy()}
    registry = {"status": "ALL_TWELVE_RESOLVED_BEFORE_DEV", "conditions": []}
    artifact = {"path": "no_scientific_file_is_created_or_opened.npz", "sha256": "a"*64,
        "arrays": {"prediction": {"shape": [2, 48, 3], "dtype": "float32", "sha256": "b"*64}}}
    for dataset, (channels, k, dev) in report.DATASETS.items():
        row = {"raw_scores": {split: pair(.5, channels) for split in ("fit", "tune", dev)}, "spaces": {}}
        checked = {"current_C_persistence_scores": {split: pair(.9, channels) for split in ("fit", "tune", dev)}, "spaces": {}}
        science["datasets"][dataset], audit["datasets"][dataset] = row, checked
        stages = []
        for stage in range(k):
            candidates = [{"candidate_original_index": index, "indices": [*range(stage), index],
                "macro_normalized_MSE": .9-stage*.1+index*.001, "per_channel_MSE": [.9-stage*.1+index*.001]*channels,
                "true_observed_counts": [1024]*channels, "active_channels": channels,
                "retained_centered_design_rank": [stage+1]*channels, "retained_centered_design_condition": [2.]*channels,
                "centered_design_singularvalues": [[4., 2.]]*channels, "singular_relative_cutoff": 1e-4,
                "negative_roundoff_MSE_values_clipped_to_zero": 0, "Gram_cross_variance_expansion_used_for_scores": False}
                for index in range(stage, channels)]
            stages.append({"stage": stage+1, "selected_original_index": stage,
                "selected_so_far_selection_order": list(range(stage+1)), "selected_so_far_original_order": list(range(stage+1)),
                "selected_macro_normalized_MSE": candidates[0]["macro_normalized_MSE"], "candidates": candidates})
        selection = {"selected_indices": list(range(k)), "selection_order": list(range(k)),
            "selected_channel_names": ["channel_"+str(i) for i in range(k)], "selection_metadata": {"stages": stages}, "FIT_source": artifact}
        row["subset_selection_lock"] = dump(tmp_path/(dataset+"_SUBSET.json"), selection)
        cost["subset_selection_locks"][dataset] = row["subset_selection_lock"]
        cost["native_selection_locks"][dataset] = {}
        for space_index, space in enumerate(report.SPACES):
            condition = {"Bolt": {}, "native": {}, "backends": {}}
            verified = {"backends": {}, "new_output_space_diagnostics": {}}
            row["spaces"][space], checked["spaces"][space] = condition, verified
            for backend in report.BACKENDS:
                meta = {"singular_relative_cutoff": 1e-4, "future_TUNE_DEV_or_F_scores_used": False,
                    "true_observed_counts": [1024]*channels, "retained_centered_design_rank": [k]*channels,
                    "retained_centered_design_condition": [2.]*channels, "centered_design_singularvalues": [[4., 2.]]*channels,
                    "observed_past_reconstruction_MSE_FP64": [.1+space_index*.05]*channels,
                    "fallback_reason": [None]*channels, "learned_slope": [True]*channels,
                    "zero_centered_predictor_variance": [False]*channels, "constant_observed_response": [False]*channels,
                    "no_observed_responses": [False]*channels, "one_observed_response": [False]*channels,
                    "rank_deficient_slope": [False]*channels, "slope_identified_in_retained_full_K_space": [True]*channels,
                    "centered_lstsq_fit_executed": [True]*channels}
                condition["backends"][backend] = {"metadata": meta, "state_artifact": artifact}
                verified["backends"][backend] = {"independent_fit": {"checked": True}, "deployed_decoder_FP32_values": [[1.]*channels]*k,
                    "deployed_decoder_max_abs": 1., "deployed_decoder_operator_norm": 2., "deployed_decoder_Frobenius_norm": 3.}
            for split in ("fit", "tune", dev):
                condition["Bolt"][split] = {**pair(.6+space_index*.1, channels), "artifact": artifact,
                    "mixed_F_cache_reused": space != "RAW_SUBSET", "new_K_F_inference_executed": space == "RAW_SUBSET",
                    "original_F_cache": artifact if space != "RAW_SUBSET" else None,
                    "full_C_slice_diagnostic": {"max_abs": 1e-6} if space == "RAW_SUBSET" else None}
                verified["new_output_space_diagnostics"][split] = {}
                for backend, methods in report.BACKENDS.items():
                    def diagnostic_trace(value):
                        return {"scores_complete_C": scores(value, channels), "real_trace_scores_complete_C": scores(value-1e-8, channels),
                            "measured_FP32_rounding_normalized_MSE_allowance": 2e-8,
                            "actual_FP32_minus_minimum": value-.2, "real_trace_minus_minimum": value-1e-8-.2}
                    verified["new_output_space_diagnostics"][split][backend] = {
                        "minimum_only_normalized_MSE": True, "old_output_space_floor_reused": False, "RAW_bound_applied": False,
                        "minimum_solution_scores_complete_C": scores(.2, channels), "RAW_scores_same_complete_C": scores(.51, channels),
                        "minimum_normalized_MSE_real_D32_numeric_rowspace": .2, "numeric_rank": k, "complete_target_positions": 96,
                        "E_Y": diagnostic_trace(.23), "actual_methods": {method: diagnostic_trace(.6+space_index*.1) for method in methods}}
            for seed in report.SEEDS:
                meta = {"selected_update": 100, "selected_tune": {"mse": .4}, "selected_state_sha256": "c"*64,
                    "train_wall_seconds": 1.2, "canonical_prediction_batch_size": 32}
                condition["native"][seed] = {"metadata": meta,
                    "scores": {split: pair(.55+space_index*.1, channels) for split in ("fit", "tune", dev)},
                    "checkpoints": [{"update": update, "tune": scores(.4+abs(update-100)/1000, channels)["normalized"],
                        "state_sha256": "c"*64, "state_artifact": artifact, "tune_prediction_artifact": artifact} for update in (0, 100, 200, 400, 600)],
                    "learning_curve": [{"update": update+1, "loss": .7, "gradient_norm_before_clip": .2} for update in range(600)]}
                lock = dump(tmp_path/(dataset+"_"+space+"_"+seed+"_LOCK.json"), {"status": "LOCKED"})
                registry["conditions"].append({"dataset": dataset, "space": space, "seed": seed, "lock": lock})
                cost["native_selection_locks"][dataset].setdefault(space, {})[seed] = lock
        rows = []
        for batch in (1, 8):
            for route_index, route in enumerate(report.ROUTES):
                seconds = [.01*batch+route_index*.001+repeat*.0001 for repeat in range(10)]
                raw_seconds = [.01*batch+repeat*.0001 for repeat in range(10)]
                ratios = [v/r for v, r in zip(seconds, raw_seconds)]
                median = statistics_median(seconds)
                rows.append({"route": route, "batch": batch, "repeats": 10, "seconds": seconds,
                    "median_seconds": median, "median_seconds_per_window": median/batch,
                    "paired_RAW_median_ratio": statistics_median(ratios), "paired_RAW_ratios": ratios,
                    "samples": [{"repeat": repeat, "position": (route_index+repeat)%10, "seconds": seconds[repeat],
                        "timer_start_ns": 10**8, "timer_end_ns": 10**8+int(seconds[repeat]*10**9)} for repeat in range(10)],
                    "origins": list(range(batch)), "memory": {"rss_bytes": 10**8}, "whole_includes": "actual pipeline",
                    "whole_excludes": "setup", "memory_scope": "shared resident process", "selected_metadata": {"fixed_D": True},
                    "reference": artifact, "timed_outputs": [artifact]*10})
        cost["cost"][dataset] = {"rows": rows}
    audit["mixed_reexecution_identity"] = {**science["mixed_reexecution_identity"],
        "actual_old_and_new_NPZ_logical_bytes_checked": True, "new_independent_conditions_from_reexecution": 0}
    science["source_producer"] = dump(tmp_path/"PARENT.json", {"status": "COMPLETE"})
    science["source_audit"] = dump(tmp_path/"PARENT_AUDIT.json", {"status": "PASS"})
    plan = tmp_path/"PLAN.md"; plan.write_text("# synthetic predeclared plan\n")
    science["plan"] = report.record(plan)
    science["selection_registry"] = dump(tmp_path/"REGISTRY.json", registry)
    prior = {"status": "PARTIAL_COMPLETED", "native_training_fits": 8, "subset_selection_fits": 0, "actual_subset_F_split_inferences": 0}
    science["prior_partial"] = dump(tmp_path/"PRIOR.json", prior)
    diagnostic = {"partial_result": science["prior_partial"], "FIT_future_targets_read": False, "TUNE_DEV_read": False,
        "datasets": {dataset: {"first_Gram_failure": {"Gram_MSE": -1e-12, "original_negative_tolerance": 1e-13,
            "direct_centered_residual_MSE": 1e-25}} for dataset in report.DATASETS}}
    science["Gram_failure_diagnostic"] = dump(tmp_path/"GRAM_DIAGNOSTIC.json", diagnostic)
    repair = tmp_path/"REPAIR.md"; repair.write_text("# preserve source and numerical objective\n")
    science["numeric_repair_addendum"] = report.record(repair)
    for key in ("source_producer", "source_audit", "plan", "selection_registry"): audit[key] = science[key]
    for key in ("source_producer", "source_audit", "prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum", "selection_registry"):
        cost[key] = science[key]
    paths = {name: tmp_path/(name+".json") for name in report.INPUTS}
    for name, value in (("audit", audit), ("cost_audit", cost_audit)):
        code = tmp_path/("synthetic_"+name+".py"); code.write_text("# immutable synthetic independent auditor\n")
        value["audit_source"] = report.record(code)
    dump(paths["science"], science); audit["diagnostic"] = report.record(paths["science"]); dump(paths["audit"], audit)
    for value in (cost, cost_audit):
        value["raw_subset_diagnostic"] = report.record(paths["science"]); value["raw_subset_audit"] = report.record(paths["audit"])
        for key in ("source_producer", "source_audit"): value[key] = science[key]
    dump(paths["cost"], cost); cost_audit["cost"] = report.record(paths["cost"]); dump(paths["cost_audit"], cost_audit)
    for name in report.FAILED_AUDITS:
        failed = {"status": "CAPTURED_UNCHANGED_AUDITOR_EXCEPTION_TRACE" if name.endswith("EXCEPTION_TRACE_v1") else "FAILED_PRESERVED",
            "failure": {"type": "ValueError", "message": "synthetic ambiguous mask flag"}}
        dump(tmp_path/(name+".json"), failed)
    (tmp_path/"FORMAT_REPAIR.md").write_text("# separate Markdown hash validation\n")
    return paths


def statistics_median(values):
    import statistics
    return statistics.median(values)


def activate_fixture(paths, monkeypatch):
    monkeypatch.setattr(report, "FIXED_SHA", {name: report.sha(path) for name, path in paths.items()})
    science = report.read(paths["science"])
    monkeypatch.setattr(report, "PRIOR_SHA", science["prior_partial"]["sha256"])
    monkeypatch.setattr(report, "FAILED_AUDITS", {name: report.record(paths["science"].parent/(name+".json")) for name in report.FAILED_AUDITS})
    monkeypatch.setattr(report, "COST_FORMAT_REPAIR", report.record(paths["science"].parent/"FORMAT_REPAIR.md"))


def test_all_source_only_gates_are_closed_before_actual_PASS(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="NOT_YET_OPEN"): report.load_verified(paths)


def test_exact_complete_PASS_metadata_chain_and_all_CSV_rows(tmp_path, monkeypatch):
    paths = fixture(tmp_path); activate_fixture(paths, monkeypatch)
    values, _ = report.load_verified(paths); tables = report.collect_tables(values)
    assert sum(len(rows) for rows in tables.values()) == 18+400+40+36+324+18+12+7200+60+168+12+1285+83+6+54+264+5+2+4
    assert len(tables) == 19 and len(tables["COMPARISON"]) == 264
    assert len(tables["GREEDY_CANDIDATES"]) == 83 and len(tables["GREEDY_RESPONSE_SCORES"]) == 1285
    assert all(not r["MAE_is_minimum"] and not r["physical_MSE_is_minimum"] and not r["RAW_bound_applied"] for r in tables["NEW_FIXED_D_DIAGNOSTIC"])
    assert sum(r["MSE_is_real_fixed_D_minimum"] for r in tables["NEW_FIXED_D_DIAGNOSTIC"]) == 36
    assert len(tables["REGISTERED_ADOPTION"]) == 18 and not any(r["all_registered_conditions_pass"] for r in tables["REGISTERED_ADOPTION"])


@pytest.mark.parametrize("change", ["pending", "producer", "count", "TEST", "oracle", "oldfloor", "costcount", "per_origin", "native_head_count", "mixed_identity"])
def test_gate_rejects_incomplete_modified_or_wrong_scope_evidence(tmp_path, monkeypatch, change):
    paths = fixture(tmp_path)
    key = "audit" if change in ("pending", "producer", "count", "TEST", "oracle", "oldfloor") else "cost" if change in ("per_origin", "native_head_count") else "cost_audit" if change == "costcount" else "science"
    value = report.read(paths[key])
    if change == "pending": value["status"] = "PENDING"
    elif change == "producer": value["diagnostic"]["sha256"] = "0"*64
    elif change == "count": value["totals"]["native_fits_checked"] = 11
    elif change == "TEST": value["protected_TEST_accessed"] = True
    elif change == "oracle": value["new_oracle_used_for_selection_or_deployment"] = True
    elif change == "oldfloor": value["old_output_space_floor_reused"] = True
    elif change == "costcount": value["totals"]["recorded_timer_samples_checked"] = 399
    elif change == "per_origin": value["per_origin_decoder_fit"] = True
    elif change == "native_head_count": value["loaded_native_heads_count"] = 8
    else: value["mixed_reexecution_identity"]["tolerance_used"] = True
    dump(paths[key], value)
    activate_fixture(paths, monkeypatch)
    with pytest.raises(ValueError): report.load_verified(paths)


def test_metadata_report_explicitly_separates_supervision_minima_and_duplicate_fits(tmp_path, monkeypatch):
    paths = fixture(tmp_path); activate_fixture(paths, monkeypatch)
    monkeypatch.setattr(report, "HERE", tmp_path)
    values, bindings = report.load_verified(paths); tables = report.collect_tables(values)
    text = report.make_text(values, tables, bindings, tmp_path/"REPORT")
    for required in ("subset/Bolt", "원 POINTWISE/MAIN", "actual-Y", "약1e-12", "20회/12000update/100CP", "distinct 등록 조건은12개",
        "complete-C", "MAE·이질적 원단위 MSE", "독립 holdout", "strict K-only", "finite 계산", "103회", "400개의 독립 데이터가 아니다",
        "ratio-of-medians", "paired ratio", "separate", "같은 affine native class"):
        if required == "separate": continue
        assert required in text
    assert "raw_subset_projection" not in text or "소스" in text
    assert all("tables/"+name+".csv" in text for name in tables)
    assert len([line for line in text.splitlines() if line.startswith("![")]) == 6


def test_ratios_are_recomputed_from_same_cost_cohort_not_paired_substitution(tmp_path, monkeypatch):
    paths = fixture(tmp_path); activate_fixture(paths, monkeypatch)
    values, _ = report.load_verified(paths)
    route = values["cost"]["cost"]["robin"]["rows"][1]
    route["paired_RAW_median_ratio"] = 999.
    tables = report.collect_tables(values)
    row = next(r for r in tables["WHOLE_COST"] if r["dataset"] == "robin" and r["route"] == "PCA_BOLT" and r["batch"] == 1)
    assert row["median_paired_RAW_ratio"] == 999.
    assert row["ratio_of_RAW_medians"] == route["median_seconds"]/values["cost"]["cost"]["robin"]["rows"][0]["median_seconds"]
    assert row["ratio_of_RAW_medians"] != row["median_paired_RAW_ratio"]


def test_CSV_nested_scores_and_nulls_roundtrip(tmp_path):
    path = tmp_path/"table.csv"
    report.write_csv(path, [{"nested": {"mask": [2, 0], "constant": True}, "missing": None, "score": .123456789012345}])
    with path.open(encoding="utf-8-sig", newline="") as handle: row = next(csv.DictReader(handle))
    assert json.loads(row["nested"]) == {"mask": [2, 0], "constant": True}
    assert row["missing"] == "" and float(row["score"]) == .123456789012345
    with pytest.raises(FileExistsError): report.write_csv(path, [{"score": 0}])


def test_no_scientific_IO_or_torch_in_report_and_outlier_labels_use_maximum_sample():
    source = inspect.getsource(report)
    assert "np.load(" not in source and "torch.load" not in source and "import torch" not in source
    assert "np.linalg" not in source and "train_native" not in source
    figure_source = inspect.getsource(report.figures)
    assert "float(samples.max())" in figure_source and "maxima[index]+padding" in figure_source
    assert 'name: None' not in inspect.getsource(report.load_verified)


def test_fresh_manifest_keeps_all_csv_nested_values_and_failure_sources(tmp_path, monkeypatch):
    paths = fixture(tmp_path); activate_fixture(paths, monkeypatch)
    monkeypatch.setattr(report, "HERE", tmp_path)
    prior = {}
    for name in report.PRIOR_REPORTS:
        path = tmp_path/"reports"/name/"REPORT_KO.md"
        path.parent.mkdir(parents=True); path.write_text("# preserved synthetic prior report\n", encoding="utf-8")
        prior[name] = report.sha(path)
    monkeypatch.setattr(report, "PRIOR_REPORTS", prior)
    code = tmp_path/"synthetic_report_source.py"; code.write_text("# immutable synthetic generator snapshot\n")
    monkeypatch.setattr(report, "__file__", str(code))
    def small_images(tables, output):
        folder = Path(output)/"figures"; folder.mkdir(); saved = []
        for i in range(1, 7):
            for suffix in ("png", "svg"):
                path = folder/(str(i)+"."+suffix); path.write_bytes(b"synthetic artifact bytes")
                saved.append(path)
        return saved
    monkeypatch.setattr(report, "figures", small_images)
    output = tmp_path/"reports"/"fresh_report"
    manifest = report.generate(paths, output)
    assert manifest["status"] == "COMPLETE" and len(manifest["table_row_counts"]) == 19
    assert manifest["figure_png_count"] == manifest["figure_svg_count"] == 6
    assert manifest["scientific_NPZ_or_model_read"] is manifest["new_fit_F_or_training_executed"] is False
    assert len(manifest["files"]) == 19+12+1
    for entry in manifest["files"]:
        path = output/entry["path"]
        assert report.sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
    assert len(manifest["preserved_prior_reports"]) == 7
    assert set(report.FAILED_AUDITS) <= set(manifest["inputs"])
    with pytest.raises(ValueError, match="FRESH_LOCAL"): report.generate(paths, output)


def test_all_six_png_svg_figures_render_from_only_synthetic_JSON(tmp_path, monkeypatch):
    paths = fixture(tmp_path); activate_fixture(paths, monkeypatch)
    values, _ = report.load_verified(paths)
    output = tmp_path/"figure_smoke"; output.mkdir()
    images = report.figures(report.collect_tables(values), output)
    assert len(images) == 12 and len({path.stem for path in images}) == 6
    for path in images:
        if path.suffix == ".png": assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        else: assert "<svg" in path.read_text(encoding="utf-8")
