"""Small generated metadata and figures; no scientific payload or real report."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import make_purged_readout_stability_report_v1 as report


def fixture():
    def score(value):return {"status":"COMPLETE","scores_all_observed":{"normalized":{"mse":value,"observed_counts":[48]*17,"active_channels":17}}}
    def terms(baseline,interaction,energy):
        fields={"baseline_MSE":baseline,"selected_MSE":baseline+interaction+energy,"delta_MSE":interaction+energy,"interaction_term":interaction,"correction_energy":energy,"closure_residual":0.,"closure_pass":True}
        per=[{"channel":j,"status":"COMPLETE","observed_count":48,**fields} for j in range(17)]
        return {"status":"COMPLETE","observed_counts":[48]*17,"observed_values":816,"active_channels":17,"per_channel":per,**fields}
    partition={"cohorts":{},"purge":{"A_candidate_count":128,"B_candidate_count":64,"C_count":64,"A_removed_count":47,"B_removed_count":47,"first_B_origin":128,"first_C_origin":192,"predicate":"o+47<next first origin"}}
    for name,start,count in (("A",0,81),("B",128,17),("C",192,64)):
        partition["cohorts"][name]={"origin_count":count,"origins_min":start,"origins_max":start+count-1,"first_target_timestamp":start,"last_target_timestamp":start+count+46,"horizon_slots":count*48,"unique_target_timestamps":count+47,"max_target_multiplicity":min(count,48),"target_multiplicity_histogram":[{"multiplicity":1,"unique_timestamp_count":2}],"observed_counts_per_channel":[count*48]*17,"unique_observed_target_timestamps_per_channel":[count+47]*17}
    result={"status":"COMPLETE","partition":partition,"routes":{},"comparisons":{},"failed_paths":[],"totals":{"fits_attempted":2,"fits_completed":2,"alpha_candidates_completed":20,"ZERO_candidates_completed":2,"candidate_A_B_evaluations_completed":44,"selected_C_evaluations_completed":2,"resolved_selection_locks":2},"resource_terminal":{"process_cpu_seconds":.5,"wall_seconds":.75,"status":"COMPLETE"},"fixed_full_FIT_upstream":True,"conditional_readout_diagnostic":True,"independent_holdout":False,"end_to_end_temporal_validation":False}
    for route in ("PCA_K8","RAW_C"):
        baselines={split:score(value) for split,value in zip(("A","B","C"),(.1,.2,.3) if route=="PCA_K8" else (.05,.1,.15))};candidates=[]
        for index,alpha in enumerate((*report.ALPHAS,None)):
            identifier="ZERO_DELTA" if alpha is None else "ALPHA_"+format(alpha,"g");candidates.append({"id":identifier,"kind":"ZERO_DELTA" if alpha is None else "ALPHA","alpha":alpha,"status":"COMPLETE","A":score(.1+index*.001),"B":score(.2+index*.001),"B_MSE":.2+index*.001})
        reference={"path":str(report.HERE/"synthetic_only.npz"),"sha256":"a"*64};row={"status":"COMPLETE","candidates":candidates,"selected_candidate":"ALPHA_1","selected_kind":"ALPHA","selected_alpha":1.,"selected_B_MSE":.18,"selection_lock":{"path":str(report.HERE/"synthetic_lock.json"),"sha256":"a"*64},"baseline_scores":baselines,"K":8 if route=="PCA_K8" else 17,"additional_slope_parameters":7*(8 if route=="PCA_K8" else 17)*17,"source_context":{"input_source":reference},"fit_metadata":{"only_A_rows_used_for_RMS":True},"selected_coefficients":reference,"selected_coefficient_value_sha256":"b"*64,"selected_A_B_prediction_artifact":reference}
        for split in ("A","B","C"):
            baseline=baselines[split]["scores_all_observed"]["normalized"]["mse"];interaction=-.03 if split!="C" or route=="RAW_C" else .02;energy=.01;cell=terms(baseline,interaction,energy);row[split]={**score(cell["selected_MSE"]),"error_terms":{"full":cell,"registered_intervals":{key:terms(baseline,interaction,energy) for key in ("1_16","17_32","33_48")}}}
        row["selected_B_MSE"]=row["B"]["scores_all_observed"]["normalized"]["mse"]
        for candidate in candidates:
            if candidate["id"]==row["selected_candidate"]:candidate["A"]=score(row["A"]["scores_all_observed"]["normalized"]["mse"]);candidate["B"]=score(row["selected_B_MSE"]);candidate["B_MSE"]=row["selected_B_MSE"]
            elif candidate["kind"]=="ZERO_DELTA":candidate["A"]=baselines["A"];candidate["B"]=baselines["B"];candidate["B_MSE"]=baselines["B"]["scores_all_observed"]["normalized"]["mse"]
        result["routes"][route]=row
    for split in ("B","C"):
        p=result["routes"]["PCA_K8"][split]["error_terms"]["full"];r=result["routes"]["RAW_C"][split]["error_terms"]["full"];result["comparisons"][split]={"PCA_selected_minus_ZERO":p["delta_MSE"],"Raw_selected_minus_ZERO":r["delta_MSE"],"PCA_selected_minus_Raw_selected":p["selected_MSE"]-r["selected_MSE"]}
    return result,{"status":"PASS","resource_terminal":{"process_cpu_seconds":1.,"wall_seconds":1.5,"status":"PASS"}}


class PurgedReportTests(unittest.TestCase):
    def test_all_six_tables_preserve_selection_candidates_terms_and_chronology(self):
        result,audit=fixture();values=report.tables(result,audit)
        self.assertEqual({name:len(rows) for name,rows in values.items()},{"SELECTED":2,"CANDIDATES":22,"SPLITS":3,"ERROR_TERMS":6,"MARGINAL_TERMS":120,"RESOURCES":2});self.assertEqual(values["SPLITS"][0]["origin_count"],81);self.assertEqual(values["SPLITS"][0]["purge"]["A_removed_count"],47)
        self.assertEqual(values["ERROR_TERMS"][2]["interaction_term"],.02);self.assertEqual(values["ERROR_TERMS"][2]["delta_MSE"],.03);self.assertEqual(sum(row["selected"] for row in values["CANDIDATES"]),2);json.dumps(values,allow_nan=False)

    def test_failed_path_is_visible_and_missing_C_metrics_are_not_zero_filled(self):
        result,audit=fixture();row=result["routes"]["PCA_K8"];result["routes"]["PCA_K8"]={"status":"FAILED_PATH_PRESERVED","failure":{"message":"synthetic failure"},"candidates":row["candidates"][:2]};values=report.tables(result,audit)
        first=values["SELECTED"][0];self.assertEqual(first["status"],"FAILED_PATH_PRESERVED");self.assertIsNone(first["C_selected_MSE"]);self.assertEqual(len(values["ERROR_TERMS"]),3)
        result,audit=fixture();result["routes"]["PCA_K8"].pop("C");result["failed_paths"]=[{"space":"PCA_K8","stage":"SELECTED_C","failure":{"message":"synthetic C failure"}}];values=report.tables(result,audit);first=values["SELECTED"][0];self.assertEqual(first["C_status"],"UNAVAILABLE");self.assertIsNone(first["C_selected_MSE"]);self.assertEqual(len(first["recorded_failures"]),1);self.assertEqual(len(values["ERROR_TERMS"]),5)

    def test_full_three_figures_six_csv_manifest_relative_links(self):
        result,audit=fixture()
        with tempfile.TemporaryDirectory(dir=report.HERE,prefix="synthetic_purged_report_") as directory:
            base=Path(directory);(base/"reports").mkdir();output=base/"reports/purged"
            with patch.object(report,"HERE",base),patch.object(report,"load_verified",return_value=(result,audit,{})):manifest=report.generate("synthetic","synthetic",output)
            self.assertEqual(len(manifest["files"]),13);self.assertEqual((manifest["PNG_count"],manifest["SVG_count"]),(3,3));self.assertFalse(manifest["independent_holdout"]);self.assertFalse(manifest["new_coefficient_cost_measured"])
            for entry in manifest["files"]:
                path=output/entry["path"];self.assertEqual(report.sha(path),entry["sha256"])
                if path.suffix==".png":self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n");self.assertGreater(path.stat().st_size,10000)
                elif path.suffix==".svg":self.assertEqual(ET.parse(path).getroot().tag.split("}")[-1],"svg")
            for name,count in manifest["table_row_counts"].items():
                with (output/("tables/"+name+".csv")).open(encoding="utf-8-sig",newline="") as handle:self.assertEqual(len(list(csv.DictReader(handle))),count)
            text=(output/"REPORT_KO.md").read_text(encoding="utf-8");self.assertIn("full-FIT upstream",text);self.assertIn("독립 holdout",text);self.assertIn("이전 timer",text);self.assertIn("2경로 중 1경로",text);self.assertIn("C baseline·feature",text);self.assertEqual(text.count("!["),3)
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):self.assertTrue((output/target).resolve().is_file())

    def test_closed_actual_gate_precedes_IO_and_import_is_array_free(self):
        with patch.object(report,"FIXED_SHA",{"result":None,"audit":None}),patch.object(report,"read",side_effect=AssertionError("actual IO forbidden")),self.assertRaisesRegex(ValueError,"NOT_REGISTERED"):report.load_verified("missing","missing")
        script="import sys;import make_purged_readout_stability_report_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules";completed=subprocess.run([sys.executable,"-c",script],cwd=report.HERE,capture_output=True,text=True);self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=="__main__":unittest.main()
