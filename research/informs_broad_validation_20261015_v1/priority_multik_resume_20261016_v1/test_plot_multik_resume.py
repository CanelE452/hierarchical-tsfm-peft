"""Tiny renderer-contract fixtures; no campaign data or graphical imports."""
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("plot_multik_resume.py")


def load():
    spec=importlib.util.spec_from_file_location("multik_plot_fixture",SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(module):
    rows=[]
    for identifier,(scope,role,k) in module.expected_rows().items():
        reused=scope!="NEW_TRAINED_PARENT_PREFIX"
        row=dict(run_id=identifier,row_scope=scope,role=role,K=str(k),CPU_threads="2",V_origins="43",observed_labels="20539",
            accuracy_reused=str(reused),cost_reused=str(reused),MSE="0.12345678901234567",MAE="0.22345678901234567",
            B1_median_seconds="0.03456789012345678",B8_median_seconds="0.23456789012345678",F_resident_bytes="",F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
        if not reused:
            for batch in (1,8):
                for key in ("raw_receipt_total_F_counters","receipt_validation_proof_SHA_map","physical_output_dtype_from_receipt"):row[f"B{batch}_"+key]=""
                row[f"B{batch}_raw_receipt_field_scope"]="NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT"
        rows.append(row)
    report=dict(schema="MULTIK_RESUME_REPORT_BINDINGS_v1",status="FROM_ACTUAL_MULTIK_AUDIT_PASS_AND_PINNED_REUSED_REFERENCES",
        audit_relative_path=module.NAME+"/audit.json",audit_sha256="a"*64,auditor_source_sha256=module.AUDITOR_SHA,builder_source_sha256=module.BUILDER_SHA,
        input_SHA_before_after_equal=True,source48_preserved=True,resumed_parent_refits=2,new_registered_conditions=0,new_prefix_accuracy_outputs=6,
        new_cost_conditions=12,reused_fixed_nested_PCA_rows=3,reused_independent_K6_rows=1,whole_E4_complete=False,whole_MASTER_complete=False,
        independent_generalization_claimed=False,unlogged_native_cost_fields_preserved_null=True,
        outputs=[dict(relative_path=module.NAME+"/MULTIK_RESUME_RESULTS_v1.csv",sha256="c"*64),dict(relative_path=module.NAME+"/MULTIK_RESUME_KO_v1.md",sha256="d"*64)])
    return rows,report


class Line:
    def __init__(self,x,y):self.x,self.y=list(x),list(y)
    def get_xdata(self):return self.x
    def get_ydata(self):return self.y


class Axis:
    def __init__(self):self.lines=[]
    def plot(self,x,y,**kwargs):
        line=Line(x,y);self.lines.append(line);return [line]
    def __getattr__(self,key):return lambda *a,**kw:None


class Tests(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest("Renderer is not implemented yet")
        self.module=load()

    def test_default_has_no_IO_or_scientific_imports(self):
        stream=io.StringIO()
        with patch.object(Path,"open",side_effect=AssertionError("Unexpected input")),contextlib.redirect_stdout(stream):
            self.assertEqual(self.module.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["input_reads"],0);self.assertEqual(result["files_written"],0)
        self.assertFalse(result["numpy_imported"]);self.assertFalse(result["torch_imported"]);self.assertFalse(result["matplotlib_imported"])

    def test_exact_six_new_four_reference_rows_become_three_curves_and_one_point(self):
        rows,report=fixture(self.module);data=self.module.prepare_data(rows,report,"c"*64,"a"*64)
        self.assertEqual([len(v["rows"]) for v in data["series"]],[3,3,3,1])
        self.assertEqual([v["row_scope"] for v in data["series"]],["NEW_TRAINED_PARENT_PREFIX","NEW_TRAINED_PARENT_PREFIX","REUSED_FIXED_NESTED_PCA","REUSED_INDEPENDENT_K6"])
        self.assertEqual([r["K"] for r in data["series"][-1]["rows"]],["6"])

    def test_actual_artist_bindings_copy_all_forty_unrounded_values(self):
        rows,report=fixture(self.module);data=self.module.prepare_data(rows,report,"c"*64,"a"*64)
        axes=[Axis() for _ in range(4)];artists=self.module.draw_panels(axes,data)
        self.module.validate_artists(artists,data)
        self.assertEqual(len(artists),16);self.assertEqual(sum(len(r["values"]) for r in artists),40)
        self.assertEqual(artists[0]["values"][0],float(rows[0]["MSE"]))
        self.assertEqual(artists[0]["source_values"][0],rows[0]["MSE"])
        changed=copy.deepcopy(artists);changed[0]["values"][0]+=.01
        with self.assertRaises(ValueError):self.module.validate_artists(changed,data)

    def test_missing_duplicate_wrong_K_or_scope_is_rejected(self):
        for change in (lambda rows:rows.pop(),lambda rows:rows.__setitem__(-1,copy.deepcopy(rows[0])),lambda rows:rows[0].update(K="4"),lambda rows:rows[-1].update(row_scope="NEW_TRAINED_PARENT_PREFIX")):
            rows,report=fixture(self.module);change(rows)
            with self.assertRaises(ValueError):self.module.prepare_data(rows,report,"c"*64,"a"*64)

    def test_partial_wrong_sources_or_unbound_actual_CSV_is_rejected(self):
        for change in (lambda r:r.update(status="PARTIAL"),lambda r:r.update(builder_source_sha256="0"*64),lambda r:r.update(audit_sha256="0"*64),lambda r:r["outputs"][0].update(sha256="0"*64),lambda r:r.update(whole_MASTER_complete=True)):
            rows,report=fixture(self.module);change(report)
            with self.assertRaises(ValueError):self.module.prepare_data(rows,report,"c"*64,"a"*64)

    def test_inferred_native_fields_or_nonfinite_measurements_are_rejected(self):
        for change in (lambda rows:rows[0].update(B1_raw_receipt_total_F_counters="0"),lambda rows:rows[0].update(F_resident_bytes="10"),lambda rows:rows[0].update(MSE="nan"),lambda rows:rows[0].update(CPU_threads="4")):
            rows,report=fixture(self.module);change(rows)
            with self.assertRaises(ValueError):self.module.prepare_data(rows,report,"c"*64,"a"*64)

    def test_single_point_cannot_become_independent_K3_K11_line(self):
        rows,report=fixture(self.module);data=self.module.prepare_data(rows,report,"c"*64,"a"*64)
        artists=self.module.draw_panels([Axis() for _ in range(4)],data)
        point=[r for r in artists if r["series"]=="independent"]
        self.assertEqual(len(point),4);self.assertTrue(all(r["K"]==[6] for r in point))

    def test_current_builder_actual_compose_API_with_only_synthetic_inputs(self):
        builder_path=SOURCE.with_name("build_multik_resume_report.py")
        self.assertEqual(hashlib.sha256(builder_path.read_bytes()).hexdigest(),self.module.BUILDER_SHA)
        fixture_path=SOURCE.with_name("test_build_multik_resume_report.py")
        spec=importlib.util.spec_from_file_location("multik_builder_fixture_for_plot",fixture_path)
        tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny);builder=tiny.load()
        fake_rows,report=fixture(self.module)
        references=[]
        for row in fake_rows:
            if row["row_scope"]!="NEW_TRAINED_PARENT_PREFIX":
                row=dict(row,K=int(row["K"]),accuracy_reused=True,cost_reused=True)
                references.append(row)
        text=builder.compose(tiny.fixture(),references,"audit.json")["MULTIK_RESUME_RESULTS_v1.csv"]
        csv_sha=hashlib.sha256(text.encode()).hexdigest();report["outputs"][0]["sha256"]=csv_sha
        rows=list(csv.DictReader(io.StringIO(text)));data=self.module.prepare_data(rows,report,csv_sha,"a"*64)
        self.assertEqual([len(s["rows"]) for s in data["series"]],[3,3,3,1])

    def test_existing_output_refused_before_any_input(self):
        with patch.object(Path,"exists",return_value=True),patch.object(Path,"open",side_effect=AssertionError("Unexpected read")):
            with self.assertRaises(FileExistsError):self.module.execute("public","cache","p"*64,"b"*64,"c"*64,"a"*64)


class ImplementationExists(unittest.TestCase):
    def test_renderer_is_implemented(self):self.assertTrue(SOURCE.exists(),"RED: multi-K renderer is not implemented")


if __name__=="__main__":unittest.main()
