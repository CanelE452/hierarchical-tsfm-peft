"""JSON-only generated report fixtures; no research arrays, model or scientific replay."""
from pathlib import Path
import copy
import csv
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import make_horizon_decoder_report_v1 as report


def scores(mse,c):
    value={"mse":mse,"mae":math.sqrt(mse)/2,"per_channel_mse":[mse]*c,"per_channel_mae":[math.sqrt(mse)/2]*c,"observed_counts":[10]*c,"active_channels":c}
    return {"normalized":copy.deepcopy(value),"physical_per_channel_diagnostic":copy.deepcopy(value)}


def entry(mse,c):return {"status":"COMPLETE","scores_all_observed":scores(mse,c),"scores_complete_mask":scores(mse,c)}


def temporal(mse,c):
    return {"per_horizon":[scores(mse,c)["normalized"] for _ in range(48)],"registered_intervals":{name:scores(mse,c)["normalized"] for name in ("1_16","17_32","33_48")},"horizon_averages_not_equal_full_observed_channel_macro":True}


def fixture_values():
    resource={"status":"COMPLETE","error":None,"process_cpu_seconds":1.2,"wall_seconds":1.4,"gpu_process":False,"path":"synthetic_resource.json","sha256":"0"*64};science={"datasets":{},"resource_terminal":resource};audit={"datasets":{},"resource_terminal":{**resource,"status":"PASS"}};cost={"cost":{},"resource_terminal":{**resource,"gpu_process":True}};ca={"resource_terminal":{**resource,"status":"PASS"}}
    opaque={"path":"generated_arrays_not_opened.npz","sha256":"1"*64}
    for data,(c,k,dev) in report.DATASETS.items():
        science["datasets"][data]={"spaces":{}};audit["datasets"][data]={"spaces":{}}
        for si,space in enumerate(report.spaces(data)):
            owner={"conditions":{},"contexts":{}};science["datasets"][data]["spaces"][space]=owner;audit["datasets"][data]["spaces"][space]={"conditions":{}};width=c if space=="RAW_C" else k if si==1 else 2*k
            base=(.3 if data=="robin" else .6)+.1*si
            for split,factor in (("fit",.8),("tune",.5),(dev,1.)):
                owner["contexts"][split]={"K":width,"input_source":opaque,"baseline_scores":{"F":entry(base*factor,c),"FLAT":entry((base+.7)*factor,c)},"current_C_persistence_scores":entry(1.4*factor,c),"baseline_temporal_diagnostics":{"F":temporal(base*factor,c),"FLAT":temporal((base+.7)*factor,c)},"latent_error_diagnostics":{"complete_C_horizons":10,"unit_scope":"synthetic fixed-space units","F":{"mse":base*factor,"mae":.1,"per_latent_mse":[base*factor]*width,"per_horizon_mse":[base*factor]*48,"per_horizon_complete_C_counts":[2]*48},"FLAT":{"mse":(base+.5)*factor,"mae":.2,"per_latent_mse":[(base+.5)*factor]*width,"per_horizon_mse":[(base+.5)*factor]*48,"per_horizon_complete_C_counts":[2]*48}}}
            for mode in report.MODES:
                for arm in report.ARMS:
                    name=mode+"_"+arm;zero=data=="robin" and space=="RAW_C" and mode=="F";delta=0. if zero else (.02 if data=="robin" else -.01) if arm=="STATIC_DELTA" else .03
                    selected_mse=(base+delta if mode=="F" else base+.4+delta);win=10 if zero else 4;row={"K":width,"selected_candidate":"ZERO_DELTA" if zero else "ALPHA_0.1","selected_kind":"ZERO_DELTA" if zero else "ALPHA","selected_alpha":None if zero else .1,"selected_TUNE_MSE":selected_mse*.5,"additional_slope_parameters":7*width*c,"selected_coefficients":opaque,"selection_lock":{"path":"generated_lock.json","sha256":"2"*64},"fit_wall_seconds":.01,"fit_metadata":{"groups":[{"observed_count":10,"retained_rank":2}],"feature_centering":False,"intercept_fitted":False},"fit_artifact":opaque,"candidates":[]}
                    for label,factor in (("FIT",.8),("TUNE",.5),("DEV",1.)):row[label]=entry(selected_mse*factor,c)
                    for i,identifier in enumerate(["ALPHA_"+format(a,"g") for a in report.ALPHAS]+["ZERO_DELTA"]):
                        value=row["selected_TUNE_MSE"]+abs(i-win)*.001;row["candidates"].append({"id":identifier,"kind":"ZERO_DELTA" if i==10 else "ALPHA","alpha":None if i==10 else report.ALPHAS[i],"status":"COMPLETE","TUNE_MSE":value,"FIT":row["FIT"] if i==win else entry(value*1.6,c),"TUNE":row["TUNE"] if i==win else entry(value,c),"coefficient_array_records":{},"coefficient_value_sha256":"3"*64,"prediction_artifact":opaque,"parity_checks":{"FIT":{"pass":True},"TUNE":{"pass":True}}})
                    row["selected_FIT_TUNE_prediction_artifact"]=opaque;row["selected_temporal_diagnostics"]={"FIT":temporal(selected_mse*.8,c),"TUNE":temporal(selected_mse*.5,c)};row["DEV"]["temporal_diagnostics"]=temporal(selected_mse,c);owner["conditions"][name]=row;audit["datasets"][data]["spaces"][space]["conditions"][name]={"independent_fit":{"groups":[{"rank":2,"observed_count":10}]}}
        routes=report.routes(data);schedule=[list(routes[i:]+routes[:i]) for i in range(5)];rows=[];times={}
        for batch in (1,8):
            for ri,route in enumerate(routes):times[route,batch]={i:(1000000+ri*17000+i*12345)*batch/1e9 for i in range(5)}
            for ri,route in enumerate(routes):
                samples=[]
                for i in range(5):
                    duration=int(round(times[route,batch][i]*1e9));start=1000000000+(batch*1000+i*20+ri)*10000000;samples.append({"dataset":data,"route":route,"batch":batch,"repeat":i,"block":i,"position":schedule[i].index(route),"perf_start_ns":start,"perf_end_ns":start+duration,"unix_start_ns":start,"unix_end_ns":start+duration,"seconds":duration/1e9,"parity":{"normalized":{"pass":True}}})
                seconds=[s["seconds"] for s in samples];row={"dataset":data,"route":route,"batch":batch,"status":"COMPLETE","repeats":5,"seconds":seconds,"samples":samples,"median_seconds":sorted(seconds)[2],"median_seconds_per_window":sorted(seconds)[2]/batch,"matched_baseline_route":report.matched_baseline(route),"declared_route_deleted":False}
                for label,denominator in (("RAW","RAW_C_F_BASELINE"),("matched_baseline",report.matched_baseline(route))):
                    denom=[times[denominator,batch][i] for i in range(5)];paired=[a/b for a,b in zip(seconds,denom)];row.update({"paired_"+label+"_seconds":denom,"paired_"+label+"_ratios":paired,"paired_"+label+"_median_ratio":sorted(paired)[2],"ratio_of_"+label+"_medians":row["median_seconds"]/sorted(denom)[2]})
                rows.append(row)
        cost["cost"][data]={"routes":list(routes),"rows":rows,"schedule":schedule}
    return {"science":science,"science_audit":audit,"cost":cost,"cost_audit":ca,"repeat_cost":copy.deepcopy(cost),"repeat_cost_audit":copy.deepcopy(ca)}


class HorizonReportTests(unittest.TestCase):
    def test_exact_cohorts_keep_all_masks_candidates_temporal_and_raw_samples(self):
        tables=report.collect_tables(fixture_values());self.assertEqual({k:len(v) for k,v in tables.items()},{"SELECTED":24,"CANDIDATES":264,"SCORES":72,"BASELINE_SCORES":54,"TEMPORAL":3672,"LATENT":36,"SOLVERS":24,"WHOLE_COST":72,"WHOLE_COST_SAMPLES":360,"WHOLE_COST_REPEAT":72,"WHOLE_COST_REPEAT_SAMPLES":360,"RESOURCES":6})
        self.assertTrue(all("all_observed" in r and "complete_C" in r for r in tables["SCORES"]));self.assertEqual(sum(r["period"]=="horizon" for r in tables["TEMPORAL"]),3456);self.assertEqual(sum(r["period"]=="interval" for r in tables["TEMPORAL"]),216)

    def test_changed_ns_or_either_ratio_is_rejected(self):
        for mutation in ("ns","RAW","matched_baseline"):
            with self.subTest(mutation=mutation):
                values=fixture_values();row=values["cost"]["cost"]["robin"]["rows"][2]
                if mutation=="ns":row["samples"][0]["perf_end_ns"]+=1
                else:row["paired_"+mutation+"_median_ratio"]+=.1
                with self.assertRaises(ValueError):report.checked_cost_rows(values["cost"])

    def test_nonminimal_TUNE_selection_and_partial_menu_are_rejected(self):
        values=fixture_values();row=values["science"]["datasets"]["robin"]["spaces"]["PCA_K4"]["conditions"]["F_STATIC_DELTA"];row["selected_candidate"]="ALPHA_10000"
        with self.assertRaisesRegex(ValueError,"OWN_FP32_TUNE_WINNER"):report.collect_tables(values)
        row["candidates"].pop()
        with self.assertRaisesRegex(ValueError,"ELEVEN_PREDECLARED"):report.selected(row)

    def test_pending_cost_audit_closes_before_metadata_reads(self):
        with patch.dict(report.FIXED_SHA,{"cost_audit":None}),patch.object(report,"read",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):report.load_verified({key:"missing" for key in report.FIXED_SHA})

    def test_main_text_has_derived_conditional_conclusions_and_no_claimed_new_floor(self):
        values=fixture_values();tables=report.collect_tables(values);text=report.make_text(values,tables,{},report.HERE/"reports"/"synthetic")
        self.assertIn("12/12",text);self.assertIn("4/4",text);self.assertIn("18/18",text);self.assertIn("독립 일반화",text);self.assertIn("계산하지 않았다",text);self.assertIn("같은 space/mode baseline",text);self.assertIn("strict K-only0",text);self.assertIn("5회전",text);self.assertIn("UnboundLocalError",text);self.assertIn("지역 식별자 두 곳",text);self.assertIn("원·반복 모두 PASS",text)

    def test_all_four_figures_and_portable_CSV_manifest_render_from_synthetic_metadata(self):
        values=fixture_values()
        with tempfile.TemporaryDirectory(dir=report.HERE,prefix="synthetic_horizon_report_") as directory:
            folder=Path(directory);(folder/"reports").mkdir();output=folder/"reports"/"synthetic";inputs=folder/"generated_input.json";inputs.write_text("{}",encoding="utf-8");bindings={"synthetic_metadata":report.record(inputs)}
            with patch.object(report,"HERE",folder),patch.object(report,"load_verified",return_value=(values,bindings)):
                manifest=report.generate({},output)
                with self.assertRaisesRegex(ValueError,"FRESH_LOCAL_REPORT"):report.generate({},output)
            self.assertEqual(manifest["PNG_count"],4);self.assertEqual(manifest["SVG_count"],4);self.assertEqual(len(manifest["files"]),21);self.assertEqual(len(list((output/"figures").glob("*.png"))),4)
            with (output/"tables"/"SELECTED.csv").open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle));self.assertEqual(len(rows),24);self.assertIn("generated_arrays_not_opened.npz",rows[0]["selected_coefficients"])
            self.assertFalse(manifest["scientific_NPZ_or_model_read"]);self.assertFalse(manifest["original_v1_76_cell_cost_or_ridge_extension_mixed"])


if __name__=="__main__":unittest.main()
