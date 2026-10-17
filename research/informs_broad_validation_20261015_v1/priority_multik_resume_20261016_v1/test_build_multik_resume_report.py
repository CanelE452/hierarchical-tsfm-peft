"""Tiny stdlib report-contract fixtures; no actual data/model/statistics."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name("build_multik_resume_report.py")


def load():
    if not SOURCE.exists():raise AssertionError("RED: multi-K report builder is not implemented")
    spec=importlib.util.spec_from_file_location("multik_report_fixture",SOURCE);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def fixture():
    parents=[];rows=[];costs=[]
    for kind,role in (("shared","ORDERED_MULTI_K"),("max_only","MAXK_PCA_TRAINED")):
        parent=f"postT_cpu_beijing_E4_{role}_k11_s92601_v1"
        parents.append(dict(run_id=parent,kind=kind,status="COMPLETE_VALID",selected_epoch=3,training_status="UNDERTRAINED_OR_BUDGET_LIMIT",search_updates=1868,search_exposures=29880,refit_updates=2262,refit_exposures=36189,search_new_optimizer_updates=0,search_new_F_calls=0,full_V_completed=False,search_seconds_this_resume_call=2.5,historical_saved_search_segment_seconds=90.0,refit_seconds_this_resume_call=120.0,old_unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED",stage_F_counters={"search":{},"refit":{}}))
        for k in (3,6,11):
            tag=role if k==11 else ("SHARED_PREFIX" if kind=="shared" else "POSTHOC_PREFIX")
            row=dict(run_id=f"postT_cpu_beijing_E4_{tag}_k{k}_s92601_v1",parent_run_id=parent,kind=kind,variant="shared" if kind=="shared" else ("max_only" if k==11 else "posthoc"),latent_k=k,mse=.12345678901234567,mae=.2,artifact_sha256="a"*64,manifest_sha256="b"*64)
            for batch in (1,8):
                cost=dict(summaries=[dict(phase="timed",batch_size=batch,trials=20,median_seconds=.01*k*batch,p90_seconds=.02*k*batch)],memory_scope={"resident_weights_cpu_MiB":180.,"foundation_instances_created":1},guard_sampled_peak_RSS_bytes=1000,raw_receipt_total_F_counters=None,receipt_validation_proof_SHA_map=None,physical_output_dtype_from_receipt=None,raw_receipt_field_scope="NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT",derived_trial_F_counters=dict(F_logical_streams=50*batch*k,F_physical_streams=50*batch*k,F_calls=50,scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS"))
                row[f"cost_B{batch}"]=cost;costs.append(dict(run_id=row["run_id"],batch=batch))
            rows.append(row)
    return dict(schema="INDEPENDENT_MULTIK_RESUME_AUDIT_v1",status="PASS",error=None,input_SHA_before_after_equal=True,source48_preserved=True,whole_MASTER_complete=False,independent_generalization_claimed=False,parents=parents,rows=rows,accuracy=[{}]*6,cost_conditions=costs)


class Tests(unittest.TestCase):
    def setUp(self):self.b=load()

    def test_six_aliases_are_only_two_trained_parents_and_keep_precise_values(self):
        audit=fixture();self.b.validate_audit(audit);texts=self.b.compose(audit,[],"audit.json")
        rows=list(csv.DictReader(io.StringIO(texts["MULTIK_RESUME_RESULTS_v1.csv"])))
        self.assertEqual(len(rows),6);self.assertEqual(len({r["parent_run_id"] for r in rows}),2)
        self.assertEqual(float(rows[0]["MSE"]),.12345678901234567)
        self.assertIn("UNKNOWN_NOT_LOGGED",texts["MULTIK_RESUME_KO_v1.md"])

    def test_refuses_partial_fabricated_full_v_or_duplicate_alias(self):
        for change in (lambda v:v.update(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"),lambda v:v["parents"][0].update(full_V_completed=True),lambda v:v["rows"][-1].update(run_id=v["rows"][0]["run_id"])):
            value=fixture();change(value)
            with self.assertRaises(ValueError):self.b.validate_audit(value)

    def test_missing_raw_fields_stay_unlogged_not_zero_or_derived_replacements(self):
        audit=fixture();texts=self.b.compose(audit,[],"audit.json")
        rows=list(csv.DictReader(io.StringIO(texts["MULTIK_RESUME_RESULTS_v1.csv"])))
        self.assertEqual(rows[0]["B1_raw_receipt_total_F_counters"],"")
        self.assertEqual(rows[0]["B1_derived_trial_F_logical_streams"],"150")
        audit["rows"][0]["cost_B1"]["raw_receipt_total_F_counters"]={"streams":150}
        with self.assertRaises(ValueError):self.b.validate_audit(audit)

    def test_restored_search_call_is_not_historical_search_training_time(self):
        texts=self.b.compose(fixture(),[],"audit.json")
        rows=list(csv.DictReader(io.StringIO(texts["MULTIK_RESUME_RESULTS_v1.csv"])))
        self.assertEqual(rows[0]["search_restore_call_seconds"],"2.5")
        self.assertEqual(rows[0]["historical_saved_search_segment_seconds"],"90.0")
        self.assertIn("과거 search 전체 wall",texts["MULTIK_RESUME_KO_v1.md"])

    def test_exposure_or_k_counter_mutation_is_rejected(self):
        for change in (lambda v:v["parents"][0].update(search_new_optimizer_updates=1),lambda v:v["parents"][1].update(refit_exposures=36188),lambda v:v["rows"][0]["cost_B8"]["derived_trial_F_counters"].update(F_calls=49)):
            value=fixture();change(value)
            with self.assertRaises(ValueError):self.b.validate_audit(value)

    def test_reused_reference_rows_are_labeled_and_not_counted_as_new_fits(self):
        ref=dict(row_scope="REUSED_INDEPENDENT_K6",role="INDEPENDENT_K6_REFERENCE",run_id="postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",parent_run_id="postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",K=6,MSE=.3,MAE=.4,accuracy_reused=True,cost_reused=True,decoder_scope="TRAINED_B0_LAG7_ANCHOR",reference_source="pinned.csv",B1_median_seconds=.1,B1_p90_seconds=.2,B8_median_seconds=.8,B8_p90_seconds=.9)
        texts=self.b.compose(fixture(),[ref],"audit.json")
        rows=list(csv.DictReader(io.StringIO(texts["MULTIK_RESUME_RESULTS_v1.csv"])))
        self.assertEqual(len(rows),7);self.assertEqual(rows[-1]["accuracy_reused"],"True")
        self.assertIn("독립 K3/K11",texts["MULTIK_RESUME_KO_v1.md"])

    def test_reference_loader_keeps_only_pinned_three_pca_and_one_independent_k6(self):
        nested=[dict(K=str(k),run_id=f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1",accuracy_device="cpu",CPU_threads="2",V_origins="43",observed_labels="20539",C="11",L="512",H="48",MSE=".3",MAE=".4",B1_median_seconds=".1",B1_p90_seconds=".2",B8_median_seconds=".8",B8_p90_seconds=".9") for k in (3,6,11)]
        core=[dict(role="B0_K6",run_id="postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",K="6",accuracy_device="cpu",training_device="cpu",CPU_threads="2",V_origins="43",observed_labels="20539",MSE=".3",MAE=".4",training_status="UNDERTRAINED_OR_BUDGET_LIMIT",B1_median_seconds=".1",B1_p90_seconds=".2",B8_median_seconds=".8",B8_p90_seconds=".9")]
        rows=self.b.reference_rows(nested,core)
        self.assertEqual(len(rows),4);self.assertTrue(all(r["accuracy_reused"] for r in rows))
        core[0]["observed_labels"]="20538"
        with self.assertRaises(ValueError):self.b.reference_rows(nested,core)

    def test_default_no_io_without_scientific_imports(self):
        stream=io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(stream):self.assertEqual(self.b.main([]),0)
        value=json.loads(stream.getvalue());self.assertEqual(value["input_reads"],0);self.assertEqual(value["files_written"],0)
        self.assertFalse(value["numpy_imported"]);self.assertFalse(value["torch_imported"])


if __name__=="__main__":unittest.main()
