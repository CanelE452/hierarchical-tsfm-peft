"""Synthetic JSON primitives and full figure rendering; no scientific payload."""
import ast
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

import make_horizon_error_terms_report_v1 as report


def fixture():
    def cell(c,interaction,energy,absent=False):
        values={"baseline_MSE":1.,"selected_MSE":1.+interaction+energy,"delta_MSE":interaction+energy,"interaction_term":interaction,"correction_energy":energy,"direct_squared_error_change":interaction+energy,"closure_residual":0.,"direct_closure_residual":0.,"delta_minus_direct_change":0.,"closure_tolerance":2e-12,"closure_pass":True}
        rows=[{"channel":j,"observed_count":0 if absent else 4,"status":"NO_OBSERVED_TARGETS" if absent else "COMPLETE",**({key:None for key in values} if absent else values)} for j in range(c)]
        return {"status":"NO_OBSERVED_TARGETS" if absent else "COMPLETE","observed_counts":[0 if absent else 4]*c,"observed_values":0 if absent else 4*c,"active_channels":0 if absent else c,"per_channel":rows,**({key:None for key in values} if absent else values),"source_score_links":{"baseline":{"MSE_discrepancy":None if absent else 0.},"selected":{"MSE_discrepancy":None if absent else 0.}}}
    result={"status":"COMPLETE","datasets":{},"totals":{"conditions_checked":24,"full_cells_completed":72,"horizon_cells_completed":3456,"interval_cells_completed":216,"per_channel_cells_completed":52416},"maximum_closure_abs":0.,"resource_terminal":{"status":"COMPLETE","process_cpu_seconds":.5,"wall_seconds":.75},"parent_resource_receipts":{"horizon_diagnostic":{"status":"COMPLETE","process_cpu_seconds":1.,"wall_seconds":1.25},"horizon_audit":{"status":"PASS","process_cpu_seconds":2.,"wall_seconds":2.25}}}
    for data,c,k in (("robin",17,4),("beijing_aotizhongxin",11,2)):
        result["datasets"][data]={"spaces":{}}
        for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)):
            owner=result["datasets"][data]["spaces"][space]={"conditions":{}}
            for mode in ("F","FLAT"):
                for arm in ("STATIC_DELTA","RAMP_DELTA"):
                    zero=data=="robin" and space=="RAW_C" and mode=="F";row={"selected_candidate":"ZERO_DELTA" if zero else "ALPHA_10","selected_kind":"ZERO_DELTA" if zero else "ALPHA","selected_alpha":None if zero else 10.,"selected_TUNE_MSE":1.,"selected_coefficients":{"path":str(report.HERE/"synthetic_only.npz"),"sha256":"a"*64},"selected_coefficient_value_sha256":"b"*64,"selection_lock":{"path":str(report.HERE/"synthetic_only.json"),"sha256":"c"*64},"K":c if space=="RAW_C" else int(space.split("K")[1])}
                    for split in ("FIT","TUNE","DEV"):
                        interaction,energy=(0.,0.) if zero else ((.03,.02) if split=="DEV" and mode=="F" and space!="RAW_C" else (-.03,.01))
                        row[split]={"full":cell(c,interaction,energy),"per_horizon":[cell(c,interaction,energy,absent=h==0) for h in range(48)],"registered_intervals":{key:cell(c,interaction,energy) for key in ("1_16","17_32","33_48")},"source_bindings":{"input_source":{"path":str(report.HERE/"synthetic_only.npz"),"sha256":"a"*64}},"prediction_baseline_byte_equal":zero}
                    row["TUNE_DEV_observation"]={"TUNE_delta_negative_DEV_delta_positive":mode=="F" and space!="RAW_C","causal_distribution_shift_or_overfit_proven":False};owner["conditions"][mode+"_"+arm]=row
    return result


class ErrorTermsReportTests(unittest.TestCase):
    def test_all_metadata_primitives_counts_and_absent_cells_preserved(self):
        result=fixture();tables=report.tables(result)
        self.assertEqual({name:len(rows) for name,rows in tables.items()},report.COUNTS)
        first=tables["FULL"][0];source=result["datasets"][first["dataset"]]["spaces"][first["space"]]["conditions"][first["condition"]][first["split"]]["full"]
        self.assertEqual(first["interaction_term"],source["interaction_term"]);self.assertEqual(first["observed_counts"],source["observed_counts"])
        missing=next(row for row in tables["HORIZONS"] if row["horizon"]==1);self.assertIsNone(missing["delta_MSE"]);self.assertEqual(missing["status"],"NO_OBSERVED_TARGETS")
        self.assertEqual(len(tables["CHANNEL_BALANCE"]),48);self.assertEqual(sum(row["TUNE_improved_DEV_worse"] for row in tables["CONDITIONS"]),8);json.dumps(tables,allow_nan=False)

    def test_missing_condition_horizon_or_interval_fails(self):
        for key in ("condition","horizon","interval"):
            with self.subTest(key=key):
                result=fixture();owner=result["datasets"]["robin"]["spaces"]["RAW_C"]
                if key=="condition":owner["conditions"].pop("F_STATIC_DELTA")
                elif key=="horizon":owner["conditions"]["F_STATIC_DELTA"]["FIT"]["per_horizon"].pop()
                else:owner["conditions"]["F_STATIC_DELTA"]["FIT"]["registered_intervals"].pop("1_16")
                with self.assertRaises(ValueError):report.tables(result)

    def test_full_four_figures_52416_per_channel_csv_manifest_and_links(self):
        result=fixture()
        with tempfile.TemporaryDirectory(dir=report.HERE,prefix="synthetic_error_report_") as directory:
            base=Path(directory);(base/"reports").mkdir();output=base/"reports/error_terms"
            with patch.object(report,"HERE",base),patch.object(report,"load_verified",return_value=(result,{})):manifest=report.generate("synthetic",output)
            self.assertEqual(len(manifest["files"]),16);self.assertEqual((manifest["PNG_count"],manifest["SVG_count"]),(4,4));self.assertFalse(manifest["scientific_NPZ_or_model_read"]);self.assertFalse(manifest["external_independent_audit_claimed"])
            for item in manifest["files"]:
                path=output/item["path"];self.assertEqual(report.sha(path),item["sha256"])
                if path.suffix==".png":self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n");self.assertGreater(path.stat().st_size,10000)
                elif path.suffix==".svg":self.assertEqual(ET.parse(path).getroot().tag.split("}")[-1],"svg")
            for name,count in report.COUNTS.items():
                with (output/("tables/"+name+".csv")).open(encoding="utf-8-sig",newline="") as handle:self.assertEqual(sum(1 for _ in csv.DictReader(handle)),count)
            text=(output/"REPORT_KO.md").read_text(encoding="utf-8");self.assertEqual(text.count("!["),4);self.assertIn("8/24",text);self.assertIn("유일한 인과",text);self.assertIn("반환된 FP32",text)
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):self.assertTrue((output/link).resolve().is_file(),link)

    def test_gate_precedes_IO_import_is_array_free_and_payload_loading_absent(self):
        with patch.object(report,"RESULT_SHA",None),patch.object(report,"read",side_effect=AssertionError("IO forbidden")),self.assertRaisesRegex(ValueError,"NOT_REGISTERED"):report.load_verified("missing")
        tree=ast.parse(Path(report.__file__).read_text(encoding="utf-8"));calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)];self.assertFalse(any(call in ("np.load","torch.load") or call.endswith(("fit_path","backward","train_head")) for call in calls))
        script="import sys;import make_horizon_error_terms_report_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules";completed=subprocess.run([sys.executable,"-c",script],cwd=report.HERE,capture_output=True,text=True);self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=="__main__":unittest.main()
