"""Generated metadata only; the fixture writer is test-only, never imported by the verifier."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_recon_init_report_v1 as audit
from test_make_recon_init_sensitivity_report_v1 import fixture_values,module as fixture_writer


class MetadataVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent); self.folder=Path(self.temp.name)
        self.values=fixture_values(self.folder/"metadata")
    def tearDown(self): self.temp.cleanup()
    def test_closed_report_gate_reads_no_inputs(self):
        with patch.object(audit,"REPORT_SHA",None),patch.object(audit,"metadata",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            audit.verify(self.folder/"absent")
    def test_all_ten_tables_independently_match_generated_synthetic_CSV(self):
        writer=fixture_writer(); actual=writer.collect_tables(self.values); expected=audit.expected_tables(self.values)
        self.assertEqual([len(expected[k]) for k in audit.TABLES],[84,24,20,2400,4,2,8,24,8,2])
        compared=0
        for name in audit.TABLES:
            path=self.folder/(name+".csv"); writer.write_csv(path,actual[name],self.folder)
            compared+=audit.check_table(path,audit.relative_values(expected[name],self.folder))
        self.assertGreater(compared,25000)
        self.assertEqual(sum(r["new_execution"] for r in expected["UPDATES"]),1200)
    def test_per_channel_score_and_false_bound_mutations_are_rejected(self):
        writer=fixture_writer(); actual=writer.collect_tables(self.values); expected=audit.expected_tables(self.values)
        for name,mutate in (("SCORES",lambda r:r[0]["per_channel_mse"].__setitem__(0,99.)),("SPACE_METHODS",lambda r:r[0].__setitem__("real_trace_bound_inequality_pass",True))):
            with self.subTest(table=name):
                import copy
                rows=copy.deepcopy(actual[name]); mutate(rows); path=self.folder/(name+".csv"); writer.write_csv(path,rows,self.folder)
                with self.assertRaisesRegex(ValueError,"CSV_CELL"): audit.check_table(path,audit.relative_values(expected[name],self.folder))
    def test_wrong_selected_checkpoint_is_rejected(self):
        self.values["train"]["datasets"]["robin"]["conditions"]["92601"]["selected_update"]=100
        with self.assertRaisesRegex(ValueError,"TUNE_ONLY"): audit.expected_tables(self.values)
    def test_unobserved_and_complete_masks_and_original_false_flags_stay_distinct(self):
        out=audit.expected_tables(self.values)
        base={"method":"RECON_INIT_TRAINED","seed":92601,"split":"val","scale":"normalized"}
        self.assertNotEqual(audit.find(out["SCORES"],**base,mask="all_observed")["mse"],audit.find(out["SCORES"],**base,mask="complete_C")["mse"])
        row=audit.find(out["SPACE_METHODS"],initialization="RECON",seed=92601,split="val",method="ACTUAL_F")
        self.assertFalse(row["real_trace_bound_inequality_pass"]); self.assertTrue(row["retained_output_trace_bound_inequality_pass"])
    def test_markdown_numbers_scope_and_missing_relative_link(self):
        writer=fixture_writer(); tables=audit.expected_tables(self.values); output=self.folder/"report"; output.mkdir(); (output/"tables").mkdir(); (output/"figures").mkdir()
        for name in audit.TABLES: (output/"tables"/(name+".csv")).write_text("generated placeholder",encoding="utf-8")
        for name in ("01_initialization_errors","02_TUNE_checkpoints","03_selected_output_spaces"): (output/"figures"/(name+".png")).write_bytes(b"generated placeholder")
        path=output/"REPORT_KO.md"; text=writer.make_text(self.values,tables,{},output); path.write_text(text,encoding="utf-8")
        self.assertEqual(audit.check_markdown(output,tables),13)
        path.write_text(text.replace("RECON CP0 TUNE 0.1030000000","RECON CP0 TUNE 9.1030000000"),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"MD_INITIAL_BASELINE"): audit.check_markdown(output,tables)
        path.write_text(text,encoding="utf-8"); (output/"tables/SCORES.csv").unlink()
        with self.assertRaisesRegex(ValueError,"EXISTING_MARKDOWN_LINK"): audit.check_markdown(output,tables)
    def test_verifier_source_does_not_import_generator_or_scientific_packages(self):
        import ast
        text=Path(audit.__file__).read_text(encoding="utf-8"); imports=[]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom): imports.append(node.module)
        self.assertFalse(any(x.startswith(("make_","torch","numpy","run_")) for x in imports))


if __name__=="__main__": unittest.main()
