"""Generated JSON-only report fixtures; no model, NPZ or scientific inference."""
from pathlib import Path
import ast
import builtins
import copy
import csv
import inspect
import json
import statistics
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import make_causal_lag_report_v1 as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return report.record(path)


def score(mse, mae, channels, count=47):
    common = {"observed_counts": [count]*channels, "active_channels": channels}
    return {"normalized": {"mse": mse, "mae": mae, "per_channel_mse": [mse]*channels, "per_channel_mae": [mae]*channels, **common},
        "physical_per_channel_diagnostic": {"mse": 4*mse, "mae": 2*mae, "per_channel_mse": [4*mse]*channels, "per_channel_mae": [2*mae]*channels, **common}}


def opaque(folder, name):
    return {"path": str(folder/(name+".npz")), "sha256": "1"*64, "bytes": 123,
        "arrays": {"latent_history": {"shape": [1,512,1], "dtype": "float32", "sha256": "2"*64},
            "actual_F_future": {"shape": [1,48,1], "dtype": "float32", "sha256": "3"*64},
            "origins": {"shape": [1], "dtype": "int64", "sha256": "4"*64}}}


def fixture_values(folder):
    parent = write(folder/"PARENT.json", {"status": "COMPLETE"})
    oldaudit = write(folder/"PARENT_PASS.json", {"status": "PASS"})
    oldreport = folder/"PARENT_REPORT.md"; oldreport.write_text("Synthetic prior report.\n", encoding="utf-8")
    oldmanifest = write(folder/"PARENT_MANIFEST.json", {"status": "COMPLETE"})
    oldreportaudit = write(folder/"PARENT_REPORT_PASS.json", {"status": "PASS"})
    plan = folder/"PLAN.md"; plan.write_text("Synthetic frozen plan.\n", encoding="utf-8")
    science = {"status": "COMPLETE", "protected_TEST_accessed": False, "decoder_fits_completed": 24, "resolved_fit_lock_files": 24,
        "cached_D_predictions_completed": 72, "joint_output_diagnostics_completed": 72, "EY_diagnostics_completed": 72, "flat_diagnostics_completed": 72,
        "actual_F_cache_splits_reused": 18, "all_24_locks_before_TUNE_DEV_NPZ": True, "failed_paths": [], "new_F_calls": 0,
        "new_native_head_training": 0, "new_backbone_training": False, "TUNE_DEV_used_for_fit_or_selection": False,
        "oracle_used_for_fit_or_selection": False, "independent_holdout": False, "strict_K_only_compression": False,
        "new_representation_learning_claimed": False, "whole_deployment_cost_measured": False, "datasets": {},
        "plan": report.record(plan), "raw_subset_diagnostic": parent, "raw_subset_audit": oldaudit,
        "parent_report": report.record(oldreport), "parent_report_manifest": oldmanifest, "parent_report_audit": oldreportaudit,
        "resource_terminal": {"process_cpu_seconds": 1.25, "wall_seconds": 2.5}}
    registry, fit_locks = [], {}
    for di, (data, (channels, k, dev)) in enumerate(report.DATASETS.items()):
        science["datasets"][data] = {"spaces": {}}; fit_locks[data] = {}
        for si, space in enumerate(report.SPACES):
            owner = science["datasets"][data]["spaces"][space] = {"E_source": opaque(folder, data+space+"E"), "conditions": {}}
            fit_locks[data][space] = {}
            for ci, condition in enumerate(report.CONDITIONS):
                lags = [0,1,2,4,8,16,32] if "L7" in condition else [0]; p = len(lags)*k
                state = opaque(folder, data+space+condition+"STATE")
                meta = {"slope_parameters_per_C": p, "singular_relative_cutoff": 1e-4, "centered_design": "RECON" in condition,
                    "intercept_fitted": "RECON" in condition, "rank": [p]*channels,
                    "singularvalues": [[1/(i+1) for i in range(p)] for channel in range(channels)], "retained_condition": [p]*channels,
                    "direct_residual_MSE": [.1]*channels, "fallback_reason": [None]*channels, "fit_executed": [True]*channels,
                    "rank_deficient_slope": [False]*channels, "full_slope_identified": [True]*channels, "constant_observed_response": [False]*channels,
                    "all_predictor_entries_zero": [False]*channels, "zero_centered_predictor_variance": [False]*channels,
                    "true_observed_counts": [480 if "RECON" in condition else 48]*channels,
                    "no_observed_response": [False]*channels, "one_observed_response": [False]*channels}
                entry = {"status": "LOCKED_FIT_ONLY", "lags": lags, "objective": "RECON" if "RECON" in condition else "FORECAST",
                    "state_artifact": state, "fit_metadata": meta, "fit_wall_seconds": .01, "FIT_input_source": opaque(folder,data+space+"FITINPUT"),
                    "FIT_Bolt_cache": opaque(folder,data+space+"FITCACHE"), "FIT_future_targets_used": "FORECAST" in condition,
                    "TUNE_DEV_used_for_fit_or_selection": False, "oracle_used_for_fit_or_selection": False, "splits": {}}
                lock = write(folder/(data+space+condition+"LOCK.json"), {**entry, "dataset": data, "space": space, "condition": condition})
                entry["fit_lock"] = lock; fit_locks[data][space][condition] = lock
                registry.append({"dataset": data, "space": space, "condition": condition, "fit_lock": lock})
                owner["conditions"][condition] = entry
                for split in ("fit", "tune", dev):
                    raw_mse, raw_mae = (1., .6) if di else (.1, .2)
                    factor = (.7 if (si, ci) == (0, 0) else 1.1+.05*ci+.02*si) if split == "tune" else 2.+.1*ci+.02*si
                    actual = score(raw_mse*factor, raw_mae*factor, channels)
                    raw, persist = score(raw_mse,raw_mae,channels), score(raw_mse*3,raw_mae*2,channels)
                    case = {"status": "COMPLETE", "diagnostic_failures": [], "scores_all_observed": actual,
                        "scores_complete_mask": actual, "RAW_BOLT": {"scores_all_observed": raw,"scores_complete_mask": raw},
                        "current_C_persistence": {"scores_all_observed": persist,"scores_complete_mask": persist},
                        "input_source": opaque(folder,data+space+split+"INPUT"), "cached_Bolt_source": opaque(folder,data+space+split+"CACHE"),
                        "artifact": opaque(folder,data+space+condition+split+"PRED"), "complete_target_positions": 47,
                        "cached_future_label_perturbation": {"status": "PASS", "windows": 1, "prediction_byte_exact": True}, "diagnostics": {"methods": {}}}
                    certificate = {"origins_zero_based": [0], "complete_positions_per_origin": 47, "selected_output_rows": 47*channels,
                        "unknown_columns_kept": 48*k, "numeric_rank": 47*k, "singularvalues": [.5]*(47*k),
                        "normal_equation_max_abs": 0., "retained_left_space_residual_max_abs": 0.}
                    case["diagnostics"]["joint_floor"] = {"status": "COMPLETE", "minimum_normalized_MSE": raw_mse*.4,
                        "minimum_solution_scores_complete_mask": score(raw_mse*.4, raw_mae*.4, channels),
                        "all_HK_future_variables_retained": True, "complete_C_output_rows_only": True,
                        "metadata": {"certificates": [certificate], "numeric_rank_per_origin": [47*k],
                            "minimum_space_definition": "Synthetic registered retained-space metadata", "all_real_latent_forecasts_satisfy_registered_bound_claimed": False}}
                    for method, m in (("ACTUAL_F", actual), ("EY", score(raw_mse*.8,raw_mae*.8,channels)), ("FLAT", persist)):
                        case["diagnostics"]["methods"][method] = {"scores_all_observed": m,"scores_complete_mask": m,
                            "real_trace_scores_complete_mask": m,"retained_output_real_trace_scores_complete_mask": m,
                            "real_trace_minus_minimum_normalized_MSE": m["normalized"]["mse"]-raw_mse*.4,
                            "measured_FP32_rounding_normalized_MSE_allowance": 0., "numerical_MSE_tolerance": 1e-10,
                            "real_trace_bound_inequality_pass": True,"FP32_bound_with_measured_allowance_pass": True,
                            "retained_output_trace_bound_inequality_pass": True,"discarded_numeric_map_output_squared_norm": 0.,
                            "discarded_numeric_map_output_max_abs": 0.}
                    entry["splits"][split] = case
    science["fit_registry"] = write(folder/"REGISTRY.json", {"status": "ALL_24_RESOLVED_BEFORE_TUNE_DEV", "conditions": registry,
        "failed_fit_paths": 0, "TUNE_DEV_NPZ_opened_before_seal": False})
    science_path = folder/"SCIENCE.json"; science_binding = write(science_path, science)
    science_code = folder/"AUDIT_SCIENCE.py"; science_code.write_text("# synthetic independent source\n",encoding="utf-8")
    audit_path = folder/"AUDIT.json"; audit = {"status":"PASS","protected_TEST_accessed":False,"diagnostic":science_binding,
        "audit_source":report.record(science_code),"totals":report.SCIENCE_COUNTS}; audit_binding=write(audit_path,audit)
    cost = {"status":"COMPLETE","protected_TEST_accessed":False,"causal_lag_diagnostic":science_binding,"causal_lag_audit":audit_binding,
        **{key:science[key] for key in ("fit_registry","raw_subset_diagnostic","raw_subset_audit","parent_report","parent_report_manifest","parent_report_audit")},
        "routes":list(report.ROUTES),"completed_cost_cells":52,"recorded_timer_samples":676,"completed_cell_timer_samples":676,"failed_paths":[],
        "F_before":{"fixed":True},"F_after":{"fixed":True},"fixed_states_before":{"fixed":True},"fixed_states_after":{"fixed":True},
        "new_training_updates":0,"new_native_head_training":0,"new_backbone_training":False,"native_heads_loaded_or_timed":False,
        "per_origin_decoder_fit":False,"cached_future_returned_instead_of_actual_F_inside_timer":False,"new_D_or_lag_objective_selection":False,
        "fixed_FIT_decoder_loaded_before_timer":True,"all_source_file_state_metadata_hashes_outside_timer":True,
        "fit_locks":fit_locks,"cost_plan":report.record(plan),"cost":{},"resource_terminal":{"process_cpu_seconds":1.,"wall_seconds":2.}}
    raw = [1,4,2,8,3,9,5,10,6,12,7,13,11]
    for data in report.DATASETS:
        rows=[]
        for batch in (1,8):
            raw_seconds=[v*.001*batch for v in raw]
            for route_index,route in enumerate(report.ROUTES):
                paired=[1. if route_index==0 else .5+.025*repeat+.001*route_index for repeat in range(13)]
                seconds=[r*p for r,p in zip(raw_seconds,paired)]; samples=[]
                for repeat, elapsed in enumerate(seconds):
                    start=1000000000+repeat*100000000
                    samples.append({"repeat":repeat,"position":(route_index+repeat)%13,"seconds":elapsed,
                        "perf_start_ns":start,"perf_end_ns":start+round(elapsed*1e9),"unix_start_ns":start,"unix_end_ns":start+round(elapsed*1e9)})
                rows.append({"route":route,"batch":batch,"status":"COMPLETE","repeats":13,"seconds":seconds,"samples":samples,
                    "median_seconds":statistics.median(seconds),"median_seconds_per_window":statistics.median(seconds)/batch,
                    "paired_RAW_ratios":paired,"paired_RAW_median_ratio":statistics.median(paired),"origins":list(range(batch)),
                    "memory":[{"shared":True}]*13,"selected_metadata":{"generated_only":True},"reference":opaque(folder,route+"REF"),
                    "timed_outputs":opaque(folder,route+"TIMED"),"whole_includes":"synthetic protocol metadata","whole_excludes":"synthetic setup metadata"})
        cost["cost"][data]={"rows":rows}
    cost_path=folder/"COST.json"; cost_binding=write(cost_path,cost)
    cost_code=folder/"AUDIT_COST.py";cost_code.write_text("# synthetic independent cost source\n",encoding="utf-8")
    costaudit={"status":"PASS","protected_TEST_accessed":False,"audit_source":report.record(cost_code),"totals":report.COST_COUNTS,
        "cost":cost_binding,"causal_lag_diagnostic":science_binding,"causal_lag_audit":audit_binding,
        **{key:science[key] for key in ("fit_registry","raw_subset_diagnostic","raw_subset_audit","parent_report","parent_report_manifest","parent_report_audit")}}
    cost_audit_path=folder/"COST_PASS.json";write(cost_audit_path,costaudit)
    failed=write(folder/"FAILED_AUDIT.json",{"status":"FAILED","totals":{"partial_cells":57},"failures":[{"reason":"synthetic exact-zero-column rank issue"}]})
    rankdiag=write(folder/"RANK_DIAG.json",{"status":"DIAGNOSTIC_ONLY","exact_zero_columns":True})
    repair=folder/"REPAIR.md";repair.write_text("Synthetic algebraically equivalent audit repair.\n",encoding="utf-8")
    d2=folder/"research/smallk_failure_followup_20261014/validation_v1/D2_ARRAY_AUDIT_v1.json"
    write(d2,{"status":"PASS_WITHOUT_INDEPENDENT_F_FORWARD_REPLAY","checks":28365,"failed_count":0})
    d2source=d2.parent.parent/"frozen_decoder_refit.py";d2source.write_text("# synthetic old D2 source\n",encoding="utf-8")
    (d2.parent.parent/"run_joint_objectives_v2.py").write_text("# synthetic old joint-Y source\n",encoding="utf-8")
    write(d2.parent/"JOINT_ARRAY_AUDIT_v1.json",{"status":"PASS_WITHOUT_F_OR_PER_UPDATE_FORWARD_REPLAY","checks":11216})
    paths={"science":science_path,"audit":audit_path,"cost":cost_path,"cost_audit":cost_audit_path}
    config={"ROOT":folder,"HERE":folder,"FIXED_SHA":{key:report.record(path)["sha256"] for key,path in paths.items()},
        "AUDITOR_SOURCES":{"audit":report.record(science_code),"cost_audit":report.record(cost_code)},
        "PRESERVED_FAILURES":{"science_audit_v1":failed},"REPAIR_EVIDENCE":{"joint_zero_column_rank_diagnostic":rankdiag,"joint_auditor_repair_addendum":report.record(repair)}}
    return paths,config


class CausalLagReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_causal_report_",dir=report.HERE)
        self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name);self.paths,self.config=fixture_values(self.folder)

    def load(self):
        with patch.multiple(report,**self.config): return report.load_verified(self.paths)

    def test_pending_gate_stops_without_any_metadata_open(self):
        with patch.object(report,"FIXED_SHA",{key:None for key in report.INPUTS}),patch("builtins.open",side_effect=AssertionError("must remain closed")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            report.load_verified(self.paths)

    def test_complete_JSON_gate_never_opens_opaque_NPZ(self):
        original=builtins.open;seen=[]
        def safe_open(path,*args,**kwargs):
            seen.append(str(path))
            if str(path).lower().endswith(".npz"):raise AssertionError("report opened scientific array")
            return original(path,*args,**kwargs)
        with patch("builtins.open",side_effect=safe_open): values,bindings=self.load()
        self.assertEqual(values["science"]["status"],"COMPLETE");self.assertIn("science_audit_v1",bindings)
        self.assertEqual(list(self.folder.rglob("*.npz")),[])

    def test_missing_science_or_cost_audit_coverage_rejected(self):
        original_bytes={key:path.read_bytes() for key,path in self.paths.items()}
        for name in ("audit","cost_audit"):
            with self.subTest(name=name):
                try:
                    value=report.read(self.paths[name]);value["totals"]={};changed=write(self.paths[name],value)
                    if name=="audit":
                        cost=report.read(self.paths["cost"]);cost["causal_lag_audit"]=changed;cost_binding=write(self.paths["cost"],cost)
                        cost_audit=report.read(self.paths["cost_audit"]);cost_audit["causal_lag_audit"]=changed;cost_audit["cost"]=cost_binding;write(self.paths["cost_audit"],cost_audit)
                    config={**self.config,"FIXED_SHA":{key:report.record(path)["sha256"] for key,path in self.paths.items()}}
                    with patch.multiple(report,**config),self.assertRaisesRegex(ValueError,"AUDIT_COVERAGE"):report.load_verified(self.paths)
                finally:
                    for key,data in original_bytes.items():self.paths[key].write_bytes(data)

    def test_report_preserves_all_cases_and_original_metric_scales(self):
        values,_=self.load();tables=report.collect_tables(values)
        expected={"COMPARISON":336,"DELTA_COMPARISONS":72,"FACTORIAL_CONTRASTS":72,"DECODER_STATES":24,
            "DECODER_CHANNELS":336,"DECODER_SINGULARVALUES":4320,"FIT_LOCKS":24,"F_CACHE_LINEAGE":18,"JOINT_GEOMETRY":72,
            "JOINT_MASK_CERTIFICATES":72,"FUTURE_DIAGNOSTIC_SCORES":1872,"RETAINED_DISCARDED":216,"FORECAST_LABEL_PROBES":72,
            "WHOLE_COST":52,"WHOLE_COST_SAMPLES":676,"REGISTERED_ADOPTION":24,"EXECUTION_RESOURCES":4,"PRESERVED_FAILURES":1}
        self.assertEqual({key:len(rows) for key,rows in tables.items()},expected)
        original=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L0_RECON"]["splits"]["tune"]["scores_all_observed"]
        for scale in report.SCALES:
            row=report.choose(tables["COMPARISON"],dataset="robin",space="PCA",condition="L0_RECON",split="tune",method="ACTUAL_F",mask="all_observed",scale=scale)
            self.assertEqual(row["mse"],original[scale]["mse"]);self.assertEqual(row["per_channel_mae"],original[scale]["per_channel_mae"])

    def test_adoption_uses_TUNE_and_two_cost_ratios_remain_distinct(self):
        values,_=self.load();tables=report.collect_tables(values)
        point=report.choose(tables["REGISTERED_ADOPTION"],dataset="robin",space="PCA",condition="L0_RECON")
        self.assertTrue(point["all_registered_conditions_pass"])
        self.assertGreater(report.choose(tables["DELTA_COMPARISONS"],dataset="robin",space="PCA",condition="L0_RECON",split="val")["MSE_ratio_to_RAW"],1)
        self.assertFalse(point["DEV_used_for_selection"])
        timing=report.choose(tables["WHOLE_COST"],dataset="robin",route="PCA_L0_RECON_BOLT",batch=8)
        self.assertNotEqual(timing["ratio_of_RAW_medians"],timing["median_paired_RAW_ratio"])

    def test_false_original_bound_boolean_is_preserved(self):
        values,_=self.load();method=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["val"]["diagnostics"]["methods"]["ACTUAL_F"]
        method["real_trace_bound_inequality_pass"]=False;method["FP32_bound_with_measured_allowance_pass"]=False;method["discarded_numeric_map_output_squared_norm"]=2.
        row=report.choose(report.collect_tables(values)["RETAINED_DISCARDED"],dataset="robin",space="PCA",condition="L7_RECON",split="val",method="ACTUAL_F")
        self.assertFalse(row["real_trace_bound_inequality_pass"]);self.assertTrue(row["retained_output_trace_bound_inequality_pass"])

    def test_csv_preserves_typed_nested_values_and_exclusive_creation(self):
        path=self.folder/"values.csv";report.write_csv(path,[{"number":.12345678901234568,"nested":{"passed":False,"x":[None,1]}}])
        with path.open(encoding="utf-8-sig") as handle:row=next(csv.DictReader(handle))
        self.assertEqual(float(row["number"]),.12345678901234568);self.assertEqual(json.loads(row["nested"]),{"passed":False,"x":[None,1]})
        with self.assertRaises(FileExistsError):report.write_csv(path,[])

    def test_changed_state_lock_or_false_TEST_scope_rejected(self):
        science=report.read(self.paths["science"]);science["protected_TEST_accessed"]=True;write(self.paths["science"],science)
        config={**self.config,"FIXED_SHA":{**self.config["FIXED_SHA"],"science":report.record(self.paths["science"])["sha256"]}}
        with patch.multiple(report,**config),self.assertRaisesRegex(ValueError,"PROTECTED_TEST"):report.load_verified(self.paths)

    def test_missing_or_reclassified_prior_failure_is_not_suppressed(self):
        failed=self.config["PRESERVED_FAILURES"]["science_audit_v1"];write(Path(failed["path"]),{"status":"PASS"})
        config={**self.config,"PRESERVED_FAILURES":{"science_audit_v1":report.record(failed["path"])}}
        with patch.multiple(report,**config),self.assertRaisesRegex(ValueError,"REMAIN_FAILED"):report.load_verified(self.paths)

    def test_source_contains_no_scientific_array_or_model_loader(self):
        tree=ast.parse(inspect.getsource(report));calls={ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
        self.assertFalse(any(call in calls for call in ("np.load","torch.load","FrozenBolt","fit_forecast","fit_reconstruction","measure_cost")))

    def test_six_figures_render_from_synthetic_JSON_and_manifest_links_are_relative(self):
        output=self.folder/"reports/toy"
        with patch.multiple(report,**self.config):manifest=report.generate(self.paths,output)
        self.assertEqual(manifest["figure_png_count"],6);self.assertEqual(manifest["figure_svg_count"],6)
        self.assertEqual(len(manifest["files"]),31);self.assertEqual(len(manifest["table_row_counts"]),18)
        for path in (output/"figures").glob("*.png"):
            with path.open("rb") as handle:header=handle.read(24)
            self.assertEqual(header[:8],b"\x89PNG\r\n\x1a\n");width,height=struct.unpack(">II",header[16:24]);self.assertGreater(width,1000);self.assertGreater(height,500)
        for path in (output/"figures").glob("*.svg"):self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))
        text=(output/"REPORT_KO.md").read_text(encoding="utf-8")
        self.assertIn("figures/01_actual_accuracy.png",text);self.assertNotIn("file://",text)
        self.assertIn("중첩된 행을 독립 표본 수로 세지 않는다",text);self.assertIn("예측 목적",text)
        with patch.multiple(report,**self.config),self.assertRaisesRegex(ValueError,"FRESH_LOCAL_REPORT"):report.generate(self.paths,output)


if __name__=="__main__":unittest.main()
