"""Small generated metadata only; full report rendering never opens scientific payloads."""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import statistics
import tempfile
import unittest
import xml.etree.ElementTree as ET

import make_new_direction_comparison_report_v1 as report


def reference(name): return {"path":str(report.HERE/(name+".npz")),"sha256":"a"*64}


def scores(value):
    common = {"observed_counts":[48,24],"active_channels":2}
    return {"normalized":{"mse":value,"mae":value**.5,"per_channel_mse":[value*.5,value*1.5],"per_channel_mae":[value**.5]*2,**common},"physical_per_channel_diagnostic":{"mse":value*3.25,"mae":value**.5*1.5,"per_channel_mse":[value*.5,value*6],"per_channel_mae":[value**.5,value**.5*2],**common}}


def entry(value): return {"status":"COMPLETE","scores_all_observed":scores(value),"scores_complete_mask":scores(value*1.01),"artifact":reference("prediction")}


def cost_fixture():
    cost = {"status":"COMPLETE","failed_paths":[],"cost":{}}
    for data in report.DATASETS:
        routes = report.routes(data); schedule=[list(routes[r:]+routes[:r]) for r in range(5)]; rows=[]
        for batch in (1,8):
            seconds={}
            for index,route in enumerate(routes):
                durations = [.1,.2,.3,.4,.5] if index == 0 else [.5,.1,.2,.3,.4]
                seconds[route] = [int(round(value*(1+index*.1)*batch*1e9))/1e9 for value in durations]
            for index,route in enumerate(routes):
                values=seconds[route]; peer=seconds[report.matched_bolt(route)]; raw=seconds["RAW_BOLT"]; samples=[]
                for repeat,value in enumerate(values):
                    start=10000000000+repeat*10000000000+index*1000000; delta=int(round(value*1e9)); value=delta/1e9
                    samples.append({"dataset":data,"route":route,"batch":batch,"repeat":repeat,"block":repeat,"position":schedule[repeat].index(route),"seconds":value,"perf_start_ns":start,"perf_end_ns":start+delta,"unix_start_ns":start+1000000000000,"unix_end_ns":start+delta+1000000000000,"parity":{key:{"pass":True,"max_abs":1e-7} for key in ("normalized","physical","scientific_normalized","scientific_physical")},"memory":{"RSS_bytes":1000,"allocated_VRAM_bytes":2000,"reserved_VRAM_bytes":3000,"scope":"shared"},"sample_artifact":reference(route+str(repeat))})
                actual=[p["seconds"] for p in samples]; rr=[value/other for value,other in zip(actual,raw)]; mr=[value/other for value,other in zip(actual,peer)]
                rows.append({"dataset":data,"route":route,"batch":batch,"status":"COMPLETE","declared_route_deleted":False,"samples":samples,"seconds":actual,"median_seconds":statistics.median(actual),"median_seconds_per_window":statistics.median(actual)/batch,"paired_RAW_ratios":rr,"paired_RAW_seconds":raw,"paired_RAW_median_ratio":statistics.median(rr),"ratio_of_RAW_medians":statistics.median(actual)/statistics.median(raw),"matched_Bolt_route":report.matched_bolt(route),"paired_matched_Bolt_ratios":mr,"paired_matched_Bolt_seconds":peer,"paired_matched_Bolt_median_ratio":statistics.median(mr),"ratio_of_matched_Bolt_medians":statistics.median(actual)/statistics.median(peer),"positions":[p["position"] for p in samples],"origins":list(range(512,512+batch)),"memory":[p["memory"] for p in samples],"selected_metadata":{"fixed_v1":True},"timed_outputs":reference(route+"timed"),"reference":reference(route+"reference"),"failure":None,"whole_includes":"physical causal fill/scaler and actual prediction/return/sync","whole_excludes":"setup hashes references warm save memory meter","memory_scope":"shared, not exclusive"})
        cost["cost"][data]={"routes":list(routes),"rows":rows,"schedule":schedule}
    return cost


def fixture_values():
    values={key:{"status":"PASS" if key.endswith("audit") else "COMPLETE","failed_paths":[],"protected_TEST_accessed":False,"resource_terminal":{"status":"PASS" if key.endswith("audit") else "COMPLETE","process_cpu_seconds":.5,"wall_seconds":1.,"gpu_process":key in ("cross","native","cost")}} for key in report.FIXED_SHA}
    values["cost"].update(cost_fixture()); cross,native,ridge=(values[key] for key in ("cross","native","ridge"))
    for key in ("cross","native","ridge"): values[key]["datasets"]={data:{"routes":{}} if key == "ridge" else {"conditions":{}} for data in report.DATASETS}
    values["ridge_audit"]["datasets"]={data:{} for data in report.DATASETS}
    for di,(data,(c,k,dev)) in enumerate(report.DATASETS.items()):
        for si,name in enumerate(report.state_names(data)):
            owner={"kind":name.rsplit("_K",1)[0],"K":int(name.rsplit("_K",1)[1]),"state_artifact":reference(name+"state"),"splits":{}}; cross["datasets"][data]["conditions"][name]=owner
            nowner={"native":{}}; native["datasets"][data]["conditions"][name]=nowner
            for split in ("fit","tune",dev):
                rawv=.05 if split != dev else .4+di*.2; fv=rawv*(1.3+si*.1); methods={}
                for method,value in (("ACTUAL_F",fv),("RAW_BOLT",rawv),("E_Y",rawv*.7)):
                    methods[method]={"scores_oracle_common_mask":scores(value*1.2)}
                    if method != "E_Y": methods[method].update(scores_all_observed=scores(value),scores_complete_mask=scores(value*1.01))
                owner["splits"][split]={"status":"COMPLETE","methods":methods,"artifact":reference(data+name+split),"input_source":reference(data+split+"input"),"oracle_valid_horizons":12,"complete_C_horizons":20}
            for seed in report.SEEDS:
                cps=[]
                for update,multiplier in zip((0,100,200,400,600),(1.2,1.,1.1,1.15,1.2)):
                    value=.08*(1+si*.1)*multiplier; cp={**entry(value),"update":update,"tune":{"mse":value,"mae":value**.5},"state":reference(data+name+str(seed)+str(update)),"head_state_sha256":"b"*64,"checks":{"cached_Torch_latent_history_byte_exact":True}}
                    cps.append(cp)
                selected=cps[1]
                nowner["native"][str(seed)]={"checkpoints":cps,"selected_update":100,"selected_tune":selected["tune"],"selected_state":selected["state"],"selection_lock":{"path":str(report.HERE/"synthetic_lock.json"),"sha256":"c"*64},"updates_log":{"path":str(report.HERE/"synthetic_updates.jsonl"),"sha256":"d"*64},"FIT":entry(.06+si*.01),"TUNE":selected,"DEV":entry(.5+di*.1+si*.02)}
        for route in report.ridge_routes(data):
            candidates=[]; raw_current=route == "RAW_CURRENT_C"
            for index,alpha in enumerate(report.ALPHAS):
                tune_values=(.4,.3,.2,.16,.12,.13,.14) if raw_current else (.4,.3,.2,.16,.14,.13,.12)
                value=tune_values[index]+di*.01
                candidates.append({"status":"COMPLETE","alpha":alpha,"FIT":entry(value*.7),"TUNE":entry(value),"TUNE_MSE":value,"prediction_artifact":reference(data+route+str(alpha)),"coefficient_value_sha256":"e"*64,"coefficient_array_records":{"coefficient_FP32":{"shape":[2,96],"dtype":"float32","sha256":"e"*64}}})
            chosen=candidates[4] if raw_current else candidates[-1]
            row={"candidates":candidates,"selected_alpha":chosen["alpha"],"selected_TUNE_MSE":chosen["TUNE_MSE"],"FIT":chosen["FIT"],"TUNE":chosen["TUNE"],"DEV":entry(.25+di*.1),"selected_FIT_TUNE_prediction_artifact":chosen["prediction_artifact"],"selected_coefficients":reference(data+route+"coeff"),"selection_lock":{"path":str(report.HERE/"synthetic_ridge_lock.json"),"sha256":"f"*64},"feature_dimension":2 if raw_current else 1024,"response_dimension":96,"normalization":"PER_RESPONSE_MASK" if raw_current else "GLOBAL_FIT","response":"level" if raw_current else "innovation","fit_metadata":{"group_count":1,"groups":[{"observed_count":4,"minimum_norm_retained_rank":2}],"minimum_norm_relative_cutoff":1e-8},"fit_artifact":reference("fitinfo"),"FIT_sources":{"input_source":reference(data+"FIT")},"coefficient_deployment":"FP32 fused features @ coefficient + intercept + anchor"}
            ridge["datasets"][data]["routes"][route]=row; values["ridge_audit"]["datasets"][data][route]={"fit_diagnostic":{"mask_groups":[{"rank":2}],"selected_alpha":chosen["alpha"]}}
    return values


class NewDirectionReportTests(unittest.TestCase):
    def test_twelve_tables_cover_all_routes_candidates_masks_resources_and_v1_sources(self):
        values=fixture_values(); tables=report.collect_tables(values)
        self.assertEqual([len(tables[key]) for key in report.TABLES],[456,96,80,84,28,72,12,76,380,76,8,0]); json.dumps(tables,allow_nan=False)
        self.assertEqual(sum(r.get("selected_upper_grid_boundary",False) for r in tables["SELECTIONS"]),10)
        self.assertTrue(all(r["Raw_point_summary"] == "MEDIAN_0.5" for r in tables["SCORES"]))
        self.assertTrue(all(isinstance(r["condition"],str) and isinstance(r["checkpoint_state"],dict) for r in tables["NATIVE_CHECKPOINTS"]))
        self.assertTrue(all(r["mask"] == "oracle_dependency_common_complete_C_only" for r in tables["ORACLE_CONTEXT"]))
        self.assertTrue(all(r["process_cpu_seconds"] == .5 for r in tables["RESOURCES"]))

    def test_integer_timers_recompute_two_denominators_and_ratio_of_medians_is_distinct(self):
        rows,samples=report.checked_cost_rows(cost_fixture()); self.assertEqual((len(rows),len(samples)),(76,380))
        row=next(r for r in rows if r["route"].endswith("_BOLT") and r["route"] != "RAW_BOLT")
        self.assertNotEqual(row["paired_RAW_median_ratio"],row["ratio_of_RAW_medians"])
        for kind in ("ns","seconds","ratio","denominator","origin","position","aggregate"):
            v=cost_fixture(); target=v["cost"]["robin"]["rows"][1]
            if kind == "ns": target["samples"][0]["perf_end_ns"] += 1
            elif kind == "seconds": target["seconds"][0] += .01
            elif kind == "ratio": target["paired_RAW_ratios"][0] += .01
            elif kind == "denominator": target["paired_RAW_seconds"][0] += .01
            elif kind == "origin": target["origins"][0] += 1
            elif kind == "position": target["samples"][0]["position"] += 1
            else: target["ratio_of_matched_Bolt_medians"] += .01
            with self.subTest(kind=kind),self.assertRaises(ValueError): report.checked_cost_rows(v)

    def test_dynamic_text_uses_metadata_CP_histogram_alpha_endpoint_and_oracle_limits(self):
        values=fixture_values(); tables=report.collect_tables(values); text=report.make_text(values,tables,{},report.HERE/"reports/synthetic")
        self.assertIn("CP100 16/16",text); self.assertIn("10/12",text); self.assertIn("19위치 중5회전은 완전 균형이 아니다",text)
        self.assertIn("Beijing의 oracle MSE를 all-observed MSE와 직접 비교하지 않는다",text); self.assertIn("query 또는 stream 수의 비를 latency 비로 대체하지 않는다",text)
        self.assertIn("보호 TEST 접근0",text); self.assertIn("quantile-summary 선정 Raw로 대체하지 않았고",text); self.assertIn("미선정72계수",text)
        self.assertEqual(text.count("!["),4)

    def test_own_TUNE_rules_exact_earlier_CP_and_registered_alpha_order(self):
        values=fixture_values(); row=values["native"]["datasets"]["robin"]["conditions"]["PCA_K4"]["native"]["92601"]
        for cp in row["checkpoints"]: cp["tune"]={"mse":.1,"mae":.2}
        row.update(selected_update=0,selected_tune=row["checkpoints"][0]["tune"],selected_state=row["checkpoints"][0]["state"]); self.assertEqual(report.native_selected(row)["update"],0)
        row["checkpoints"][-1]["tune"]["mse"] -= 1e-16
        with self.assertRaises(ValueError): report.native_selected(row)
        ridge=values["ridge"]["datasets"]["robin"]["routes"]["RAW_HISTORY512"]
        for candidate in ridge["candidates"]: candidate["TUNE_MSE"] = .1
        chosen=ridge["candidates"][0]; ridge.update(selected_alpha=0.,selected_TUNE_MSE=.1,FIT=chosen["FIT"],TUNE=chosen["TUNE"],selected_FIT_TUNE_prediction_artifact=chosen["prediction_artifact"])
        self.assertEqual(report.ridge_selected(ridge)["alpha"],0.)
        ridge["candidates"][-1]["TUNE_MSE"] -= 1e-16
        with self.assertRaises(ValueError): report.ridge_selected(ridge)

    def test_failed_cost_cell_stays_in_table_with_status_and_no_invented_sample(self):
        values=fixture_values(); cost=values["cost"]; row=cost["cost"]["robin"]["rows"][1]; row.update(status="PARTIAL_OR_FAILED_PATH_PRESERVED",failure={"stage":"synthetic"}); cost["status"]="PARTIAL_COMPLETED"
        tables=report.collect_tables(values); self.assertEqual(len(tables["WHOLE_COST"]),76); self.assertEqual(len(tables["FAILURES"]),1)
        self.assertEqual(choose_one(tables["WHOLE_COST"],row["route"],row["batch"])["status"],"PARTIAL_OR_FAILED_PATH_PRESERVED")

    def test_full_synthetic_render_manifest_CSV_relative_links_and_image_formats(self):
        values=fixture_values()
        with tempfile.TemporaryDirectory(dir=report.HERE) as temp:
            base=Path(temp); (base/"reports").mkdir(); output=base/"reports/synthetic"
            with patch.object(report,"HERE",base),patch.object(report,"load_verified",return_value=(values,{})):
                manifest=report.generate({},output)
            self.assertEqual(len(manifest["files"]),21); self.assertEqual(manifest["PNG_count"],4); self.assertEqual(manifest["SVG_count"],4)
            for item in manifest["files"]:
                path=output/item["path"]; self.assertEqual(report.sha(path),item["sha256"])
                if path.suffix == ".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n"); self.assertGreater(path.stat().st_size,10000)
                elif path.suffix == ".svg": self.assertEqual(ET.parse(path).getroot().tag.split("}")[-1],"svg")
            with (output/"tables/SCORES.csv").open(encoding="utf-8-sig",newline="") as handle:
                rows=list(csv.DictReader(handle)); self.assertEqual(len(rows),456); source=json.loads(rows[0]["source"]); self.assertFalse(Path(source["path"]).is_absolute())
            with (output/"tables/ORACLE_CONTEXT.csv").open(encoding="utf-8-sig",newline="") as handle: self.assertEqual(len(list(csv.DictReader(handle))),72)
            text=(output/"REPORT_KO.md").read_text(encoding="utf-8"); self.assertEqual(text.count("](figures/"),4); self.assertNotIn("file://",text)
            with self.assertRaises(ValueError): report.generate({},output)

    def test_closed_future_cost_gate_precedes_metadata_IO(self):
        with patch.object(report,"FIXED_SHA",{**report.FIXED_SHA,"cost_audit":None}),patch.object(report,"read",side_effect=AssertionError("actual metadata forbidden")),patch.object(report,"record",side_effect=AssertionError("IO forbidden")):
            with self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): report.load_verified({key:"missing" for key in report.FIXED_SHA})


def choose_one(rows,route,batch): return next(row for row in rows if row["dataset"] == "robin" and row["route"] == route and row["batch"] == batch)


if __name__ == "__main__": unittest.main()
