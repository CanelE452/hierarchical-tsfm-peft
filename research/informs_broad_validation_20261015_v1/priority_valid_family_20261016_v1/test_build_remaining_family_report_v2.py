"""Small invented audit/report fixtures; no actual values or experiments."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("build_remaining_family_report_v2.py")


def load():
    spec=importlib.util.spec_from_file_location("remaining_report_fixture",SOURCE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def fixture(builder,epoch=.25):
    rows=[]
    for dataset,identifier,family,k,c,n,train,exposures,updates in builder.CONDITIONS:
        count=__import__("math").ceil(epoch*train)
        receipt=dict(status="COMPLETE_VALID",run_id=identifier,family=family,seed=92601,epochs=[.25,1.,3.],selected_epoch=epoch,
            training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",search_updates=updates,
            search_exposures=exposures,refit_updates=__import__("math").ceil(count/16),refit_exposures=count,artifact_sha256="a"*64,
            refit_scaler_sha="b"*64,protected_T_target_operations=0,probe_weights_optimizer_or_sampler_reused=False)
        search=dict(selected_epoch=epoch,completed_epochs=[0.,.25,1.,3.],seconds=100.,updates=updates,exposures=exposures)
        refit=dict(selected_epoch=epoch,completed_epochs=[0.]+([epoch] if epoch else []),seconds=20.,updates=receipt["refit_updates"],exposures=count)
        row=dict(receipt,dataset=dataset,latent_k=k,C=c,mse=.12345678901234567,mae=.2,manifest_sha256="c"*64,fit_receipt=receipt,search_stage=search,refit_stage=refit)
        for batch in (1,8):
            summaries=[dict(phase=p,batch_size=batch,trials=20,median_seconds=.01*batch,p90_seconds=.02*batch) for p in ("timed","stage_probe")]
            memory=dict(resident_weights_cpu_MiB=180.,foundation_instances_created=1)
            raw=dict(status="COMPLETE_PROTOCOL",device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,
                summaries=summaries,memory=memory,F_counters=dict(streams=50*batch*k,physical_streams=50*batch*k,physical_calls=50),
                physical_output_and_inverse_dtype="float64",cached_Z_or_forecast_reads=0,training_peak_included=False,
                LoRA_merge_and_model_loading_in_whole_trial_timer=False,base_restored_after_original_Route_close=True,model_load_seconds=.1,
                effective_F_state_scope="original frozen unmerged F",validation_result_sha256=dict(rescore="d"*64,replay="e"*64))
            row[f"cost_B{batch}"]=dict(summaries=summaries,memory_scope=memory,guard_sampled_peak_RSS_bytes=1000,receipt=raw)
        rows.append(row)
    return dict(schema="INDEPENDENT_REMAINING_FAMILY_AUDIT_v1",status="PASS",error=None,input_SHA_before_after_equal=True,source48_preserved=True,
        whole_MASTER_complete=False,independent_generalization_claimed=False,rows=rows,accuracy=[{},{}],cost_conditions=[{}]*4)


class Tests(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest("Builder not implemented yet")
        self.b=load()

    def test_default_no_IO_or_scientific_imports(self):
        output=io.StringIO()
        with patch.object(Path,"open",side_effect=AssertionError("Input")),contextlib.redirect_stdout(output):self.assertEqual(self.b.main([]),0)
        result=json.loads(output.getvalue());self.assertEqual(result["input_reads"],0);self.assertEqual(result["files_written"],0)
        self.assertFalse(result["numpy_imported"]);self.assertFalse(result["torch_imported"])

    def test_each_dataset_keeps_its_own_cohort_and_precise_values(self):
        audit=fixture(self.b);self.b.validate_audit(audit);texts=self.b.compose(audit,[],"audit.json")
        rows=list(csv.DictReader(io.StringIO(texts[self.b.OUTPUTS[0]])))
        self.assertEqual([(r["dataset"],r["C"],r["observed_labels"]) for r in rows],[("jena","21","43344"),("beijing","11","20539")])
        self.assertEqual(float(rows[0]["MSE"]),.12345678901234567)
        md=texts[self.b.OUTPUTS[1]];self.assertIn("Jena C21",md);self.assertIn("Beijing C11",md);self.assertIn("자료 간 MSE 순위",md)

    def test_selected_zero_quarter_one_three_are_actual_not_assumed_three(self):
        for epoch in (0.,.25,1.,3.):
            audit=fixture(self.b,epoch);self.b.validate_audit(audit);rows=self.b.new_rows(audit)
            self.assertEqual([r["selected_refit_epoch"] for r in rows],[epoch,epoch])
            self.assertEqual(rows[0]["refit_exposures"],__import__("math").ceil(epoch*12020))

    def test_partial_wrong_cohort_or_fake_selected_activity_refused(self):
        for change in (lambda a:a.update(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"),lambda a:a["rows"][0].update(C=11),lambda a:a["rows"][1]["fit_receipt"].update(refit_exposures=1),lambda a:a.update(whole_MASTER_complete=True)):
            audit=fixture(self.b);change(audit)
            with self.assertRaises(ValueError):self.b.validate_audit(audit)

    def test_native_receipt_counters_and_whole_batch_units_are_preserved(self):
        audit=fixture(self.b);rows=self.b.new_rows(audit)
        self.assertEqual(rows[0]["B8_median_seconds"],.08);self.assertEqual(rows[1]["B8_native_F_counters"]["streams"],4400)
        self.assertIsNone(rows[0]["F_resident_bytes"])
        audit["rows"][1]["cost_B8"]["receipt"]["F_counters"]["physical_calls"]=100
        with self.assertRaises(ValueError):self.b.validate_audit(audit)

    def test_reference_Jena_B2_uses_only_timed_rows_and_original_training_metadata(self):
        rows=[]
        for batch in (1,8):
            for phase in ("timed","stage_probe"):
                rows.append(dict(run_id=self.b.B2_RUN,dataset="jena",family="B2",seed="92601",C="21",K="6",L="512",H="48",n_origins="43",n_observed="43344",device="cpu",CPU_threads="2",worker_RSS_limit_bytes="3221225472",accuracy_reused="True",new_training="0",batch_size=str(batch),phase=phase,trials="20",normalized_MSE=".1",normalized_MAE=".2",median_seconds=str(batch if phase=="timed" else 999),p90_seconds=str(batch+1),F_resident_bytes="NA",F_resident_bytes_scope="NOT_SEPARATELY_RECORDED"))
        receipt=dict(run_id=self.b.B2_RUN,family="B2",selected_epoch=1.,search_updates=1863,search_exposures=29769,refit_updates=752,refit_exposures=12020,training_status="SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
        result=self.b.b2_reference(rows,receipt)
        self.assertEqual(result["B8_median_seconds"],8.);self.assertEqual(result["selected_refit_epoch"],1.)
        self.assertIsNone(result["F_resident_bytes"]);self.assertEqual(result["F_resident_bytes_scope"],"NOT_SEPARATELY_RECORDED")
        before=copy.deepcopy(rows)
        for sentinel in (None,"","NA"):
            for row in rows:row["F_resident_bytes"]=sentinel
            self.assertEqual(self.b.b2_reference(rows,receipt),result)
        for row in rows:row["F_resident_bytes"]="NA"
        self.assertEqual(rows,before)
        for invalid in ("na","N/A"," NA",0,"0","123"):
            rows[0]["F_resident_bytes"]=invalid
            with self.assertRaises(ValueError):self.b.b2_reference(rows,receipt)
        rows[0]["F_resident_bytes"]="NA"
        rows[0]["F_resident_bytes_scope"]="ESTIMATED_FROM_ROUTE_WEIGHTS"
        with self.assertRaises(ValueError):self.b.b2_reference(rows,receipt)
        rows[0]["F_resident_bytes_scope"]="NOT_SEPARATELY_RECORDED"
        rows[0]["n_observed"]="20539"
        with self.assertRaises(ValueError):self.b.b2_reference(rows,receipt)

    def test_existing_output_refused_before_inputs(self):
        with patch.object(Path,"exists",return_value=True),patch.object(Path,"open",side_effect=AssertionError("Input")):
            with self.assertRaises(FileExistsError):self.b.execute("public","cache","audit.json","a"*64,"b"*64,"c"*64)

    def test_v2_byte_diff_is_only_B2_NA_sentinel_and_self_filename(self):
        original=SOURCE.with_name("build_remaining_family_report.py").read_bytes()
        self.assertEqual(__import__("hashlib").sha256(original).hexdigest(),"67d19ed46ee94192de20303aeb8480cb69641fdaf9246c507a5fccf68d12e35d")
        expected=original.replace(b'r.get("F_resident_bytes") not in (None,"")',b'r.get("F_resident_bytes") not in (None,"","NA")').replace(b'"build_remaining_family_report.py"',b'"build_remaining_family_report_v2.py"').replace(b'/build_remaining_family_report.py"',b'/build_remaining_family_report_v2.py"')
        self.assertEqual(SOURCE.read_bytes(),expected)

    def test_reference_Beijing_K11_uses_exact_registered_required_ID(self):
        initial=[]
        for role,family,k,identifier in (("LORA_R8_C21","LORA",21,"postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1"),("B4_K6","B4",6,"postT_cpu_jena_E1_B4_k6_s92601_family_v1")):
            initial.append(dict(role=role,run_id=identifier,accuracy_device="cpu",training_device="cpu",CPU_threads="2",V_origins="43",observed_labels="43344",effective_F_width=str(k),F_resident_bytes="",F_resident_bytes_scope="NOT_SEPARATELY_RECORDED",lr=".001",MSE=".1",MAE=".2",selected_refit_epoch="1",search_updates="1863",refit_updates="752",search_exposures="29769",refit_exposures="12020",search_seconds="100",refit_seconds="20",training_status="SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",B1_median_seconds=".1",B1_p90_seconds=".2",B8_median_seconds=".8",B8_p90_seconds=".9"))
        core=[dict(initial[0],role="B2_K11",run_id="postT_cpu_beijing_E1_B2_k11_s92601_required_v1",K="11",observed_labels="20539",initialization="identity_at_K_equals_C")]
        rows=self.b.other_references(initial,core)
        self.assertEqual(len(rows),3);self.assertEqual(rows[-1]["run_id"],core[0]["run_id"])
        self.assertEqual(rows[-1]["dataset"],"beijing");self.assertTrue(all(r["accuracy_reused"] and r["cost_reused"] for r in rows))
        core[0]["observed_labels"]="43344"
        with self.assertRaises(ValueError):self.b.other_references(initial,core)


class Exists(unittest.TestCase):
    def test_builder_exists(self):self.assertTrue(SOURCE.exists(),"RED: remaining-family report implementation missing")


if __name__=="__main__":unittest.main()
