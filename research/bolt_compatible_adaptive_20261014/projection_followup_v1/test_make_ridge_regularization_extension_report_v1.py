"""Generated metadata/figures only; no scientific arrays or actual report execution."""
from pathlib import Path
from unittest.mock import patch
import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import make_ridge_regularization_extension_report_v1 as report


def reference(name): return {"path":str(report.HERE/(name+".npz")),"sha256":"a"*64}


def score(value): return {"scores_all_observed":{"normalized":{"mse":value}}}


def fixture():
    result={"datasets":{}}; original={"datasets":{}}; alphas=(0.,1e-4,.001,.01,.1,1.,10.,100.,1000.,10000.,None,None)
    identifiers=tuple("ALPHA_"+format(a,"g") for a in alphas[:10])+("INTERCEPT_ONLY","CURRENT_C_ANCHOR")
    for data,k in (("robin",4),("beijing_aotizhongxin",2)):
        result["datasets"][data]={"routes":{}}; original["datasets"][data]={"routes":{}}
        for route in ("RAW_HISTORY512",*(kind+"_K"+str(width)+"_HISTORY512" for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k))):
            candidates=[]; tunes=[.4,.35,.3,.28,.26,.24,.2,.22 if data == "robin" else .18,.23,.25,.27,.29]
            for index,(alpha,identifier,value) in enumerate(zip(alphas,identifiers,tunes)):
                candidates.append({"candidate_id":identifier,"kind":"RIDGE" if index < 10 else identifier,"alpha":alpha,"new_in_this_extension":index >= 7,"FIT":score(value*.5),"TUNE_MSE":value,"prediction_artifact":reference(data+route+identifier),"coefficient_value_sha256":"b"*64})
            chosen=candidates[6 if data == "robin" else 7]; old_dev=.5; new_dev=old_dev if data == "robin" else .45
            result["datasets"][data]["routes"][route]={"candidates":candidates,"selected_candidate":chosen["candidate_id"],"selected_alpha":chosen["alpha"],"selected_TUNE_MSE":chosen["TUNE_MSE"],"DEV":{**score(new_dev),"new_DEV_inference":data != "robin","artifact":reference(data+route+"DEV")},"selected_original_candidate":data == "robin","selected_new_alpha_upper_boundary":False,"zero_slope_selected":False,"selected_coefficients":reference(data+route+"coefficient"),"selection_lock":{"path":str(report.HERE/"synthetic_lock.json"),"sha256":"c"*64}}
            original["datasets"][data]["routes"][route]={"selected_alpha":10.,"selected_TUNE_MSE":.2,"DEV":score(old_dev)}
    return result,original


class ExtensionReportTests(unittest.TestCase):
    def test_all_120_candidate_values_and_ten_selected_rows_preserve_metadata_primitives(self):
        result,original=fixture(); candidates,selected=report.tables(result,original)
        self.assertEqual((len(candidates),len(selected)),(120,10)); self.assertEqual(sum(r["selected"] for r in candidates),10)
        self.assertEqual(sum(r["new_in_extension"] for r in candidates),50); self.assertEqual(sum(r["original_candidate_reselected"] for r in selected),5)
        self.assertEqual(sum(r["new_DEV_inference"] for r in selected),5); self.assertTrue(all(r["new_alpha"] == (10. if r["dataset"] == "robin" else 100.) for r in selected))
        for row in candidates:
            source=next(c for c in result["datasets"][row["dataset"]]["routes"][row["route"]]["candidates"] if c["candidate_id"] == row["candidate"])
            self.assertEqual(row["FIT_MSE"],source["FIT"]["scores_all_observed"]["normalized"]["mse"]); self.assertEqual(row["TUNE_MSE"],source["TUNE_MSE"]); self.assertEqual(row["prediction_artifact"],source["prediction_artifact"])
        json.dumps((candidates,selected),allow_nan=False)

    def test_missing_candidate_or_condition_fails_complete_table_counts(self):
        for kind in ("candidate","condition"):
            result,original=fixture()
            if kind == "candidate": result["datasets"]["robin"]["routes"]["RAW_HISTORY512"]["candidates"].pop()
            else: result["datasets"]["robin"]["routes"].pop("RAW_HISTORY512")
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,"TABLE_COUNTS"): report.tables(result,original)

    def test_two_full_synthetic_figures_CSV_portability_manifest_and_relative_links(self):
        result,original=fixture()
        with tempfile.TemporaryDirectory(dir=report.HERE) as temp:
            base=Path(temp); (base/"reports/new_direction_comparison_v1").mkdir(parents=True); (base/"reports/new_direction_comparison_v1/REPORT_KO.md").write_text("# Synthetic comparison",encoding="utf-8"); output=base/"reports/extension"
            with patch.object(report,"HERE",base),patch.object(report,"load_verified",return_value=(result,original,{})): manifest=report.generate("synthetic","synthetic",output)
            self.assertEqual(len(manifest["files"]),7); self.assertEqual((manifest["PNG_count"],manifest["SVG_count"]),(2,2)); self.assertFalse(manifest["whole_extension_cost_measured"])
            for item in manifest["files"]:
                path=output/item["path"]; self.assertEqual(report.sha(path),item["sha256"])
                if path.suffix == ".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n"); self.assertGreater(path.stat().st_size,10000)
                elif path.suffix == ".svg": self.assertEqual(ET.parse(path).getroot().tag.split("}")[-1],"svg")
            with (output/"tables/CANDIDATES.csv").open(encoding="utf-8-sig",newline="") as handle:
                rows=list(csv.DictReader(handle)); self.assertEqual(len(rows),120); self.assertFalse(Path(json.loads(rows[0]["prediction_artifact"])["path"]).is_absolute())
            with (output/"tables/SELECTED.csv").open(encoding="utf-8-sig",newline="") as handle: self.assertEqual(len(list(csv.DictReader(handle))),10)
            text=(output/"REPORT_KO.md").read_text(encoding="utf-8"); self.assertEqual(text.count("!["),2); self.assertIn("5/10조건",text); self.assertIn("변경된 Beijing5계수에 그 비용을 대입하지 않는다",text); self.assertIn("선정되지 않은 새 계수",text)
            with self.assertRaises(ValueError): report.generate("synthetic","synthetic",output)

    def test_closed_gate_precedes_IO_and_source_has_no_scientific_payload_loading(self):
        with patch.object(report,"AUDIT_SHA",None),patch.object(report,"read",side_effect=AssertionError("actual IO forbidden")):
            with self.assertRaisesRegex(ValueError,"NOT_REGISTERED"): report.load_verified("missing","missing")
        tree=ast.parse(Path(report.__file__).read_text(encoding="utf-8")); calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(call in ("np.load","torch.load") or call.endswith(("fit_path","FrozenBolt","train_head","backward")) for call in calls))
        command="import sys;sys.path.insert(0,"+repr(str(report.HERE))+");import make_ridge_regularization_extension_report_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('PURE_IMPORT_PASS')"
        completed=subprocess.run([sys.executable,"-c",command],capture_output=True,text=True,check=True); self.assertIn("PURE_IMPORT_PASS",completed.stdout)


if __name__ == "__main__": unittest.main()
