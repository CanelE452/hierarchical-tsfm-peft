"""Tiny report fixtures with invented values only; no actual evidence parsed."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("build_core_contrast.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("prepared_core_builder",SOURCE)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
else:builder=None


class CoreReportTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(builder,"Core report builder not implemented")

    def rows(self):
        rows=[]
        for role,k,mse,reuse in (("B0_K6",6,.12345678901234567,False),("B2_K6",6,.1,True),("B2_K11",11,.12,False)):
            row=dict(role=role,run_id=role,K=k,MSE=mse,MAE=.2,accuracy_reused=reuse,search_max_epoch=3,selected_refit_epoch=1,search_updates=1868,refit_updates=754,search_exposures=29880,refit_exposures=12063,search_seconds=10.,refit_seconds=3.,training_status="SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",CPU_threads=2,observed_labels=20539,V_origins=43,F_used=True,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
            for batch in (1,8):row.update({f"B{batch}_median_seconds":.1,f"B{batch}_p90_seconds":.2,f"B{batch}_CPU_weights_MiB":200.,f"B{batch}_observed_tree_peak_RSS_bytes":1000,f"B{batch}_F_instances":1,f"B{batch}_F_streams_per_trial":batch*k,f"B{batch}_F_calls_per_trial":1})
            rows.append(row)
        return rows

    def test_default_and_missing_trigger_do_no_IO(self):
        out=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("read")),contextlib.redirect_stdout(out):self.assertEqual(builder.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"],0)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):builder.main(["--execute"])

    def test_audit_must_be_complete_not_an_integrity_only_success(self):
        for value in ({"status":"NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"},{"status":"PASS","whole_MASTER_complete":True}):
            with self.assertRaises(ValueError):builder.validate_audit(value)

    def test_raw_precision_and_reuse_flag_survive_CSV_round_trip(self):
        texts=builder.compose(self.rows());rows=list(csv.DictReader(io.StringIO(texts["CORE_CONTRAST_RESULTS_v1.csv"])))
        self.assertEqual(float(rows[0]["MSE"]),.12345678901234567)
        self.assertEqual([r["accuracy_reused"] for r in rows],["False","True","False"])
        self.assertEqual(rows[0]["F_resident_bytes"],"")
        self.assertEqual(rows[0]["F_resident_bytes_scope"],"NOT_SEPARATELY_RECORDED")

    def test_delta_uses_named_denominator_and_zero_is_NA(self):
        self.assertAlmostEqual(builder.relative_delta(.1,.2),-.5)
        self.assertIsNone(builder.relative_delta(.1,0.))

    def test_report_keeps_Bolt_and_full_MASTER_limit(self):
        text=builder.compose(self.rows())["CORE_CONTRAST_KO_v1.md"]
        self.assertIn("frozen Bolt",text);self.assertIn("전체 MASTER 미완료",text)
        self.assertIn("독립 일반화",text);self.assertNotIn("noBolt 실험 완료",text)

    def test_existing_output_is_rejected_before_reading_actual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scope=root/builder.NAME;scope.mkdir();(scope/builder.OUTPUTS[0]).write_text("keep")
            with self.assertRaises(FileExistsError):builder.execute(root,root,"audit.json","x","x")
            self.assertEqual((scope/builder.OUTPUTS[0]).read_text(),"keep")


if __name__=="__main__":unittest.main()
