"""Stdlib fixtures; never read live N4 outcomes or scientific arrays."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("build_beijing_report.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("beijing_report_builder",SOURCE)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
else:builder=None


class ReportTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(builder,"N4 report builder not implemented")

    def manifest(self):
        return dict(run_id="toy",split="V_select",cohort="beijing:primary:V_select:stride48",data_sha=builder.DATA_SHA,
            scaler_sha=builder.SCALER_SHA,arrays=copy.deepcopy(builder.V_FINGERPRINTS),score=dict(mse=.12345678901234567,mae=.2,n_origins=43,n_observed=20539,num_channels=11,horizon=48,metric_scale="A_normalized_original_channels"))

    def outer(self,mode,status):
        return dict(run_id="toy",mode=mode,status=status,plan_sha256=builder.PLAN_SHA,error_class=None,postflight_error_class=None,
            original_source39_immutable9_preserved=True,new_T_target_operations=0,scientific_result={"status":status})

    def test_default_does_no_input_or_output(self):
        out=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("Unexpected input")),contextlib.redirect_stdout(out):self.assertEqual(builder.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"],0)

    def test_partial_real_queue_cannot_emit_completed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            public=Path(directory)/"p";cache=Path(directory)/"c";public.mkdir();cache.mkdir();(public/builder.NAME).mkdir()
            q=cache/builder.QUEUE_RESULT;q.parent.mkdir(parents=True);q.write_text(json.dumps({"status":"PARTIAL_FITS_PRESERVED","conditions":[]}))
            with self.assertRaises(ValueError):builder.execute(public,cache)
            self.assertFalse(any((public/builder.NAME/name).exists() for name in builder.OUTPUTS))

    def test_original_cohort_rejects_changed_mask_or_scaler(self):
        builder.validate_cohort(self.manifest())
        for field in ("observed","target_normalized","origins","channel_ids"):
            manifest=self.manifest();manifest["arrays"][field]["sha256"]="f"*64
            with self.assertRaises(ValueError):builder.validate_cohort(manifest)
        manifest=self.manifest();manifest["scaler_sha"]="f"*64
        with self.assertRaises(ValueError):builder.validate_cohort(manifest)

    def test_failed_postflight_is_not_a_completed_outer(self):
        outer=self.outer("fit","COMPLETE_FIT_AND_FULL_V");builder.validate_outer(outer,"toy","fit","COMPLETE_FIT_AND_FULL_V")
        outer["postflight_error_class"]="PinnedSourceChanged"
        with self.assertRaises(ValueError):builder.validate_outer(outer,"toy","fit","COMPLETE_FIT_AND_FULL_V")

    def test_actual_saved_prediction_rescore_is_required(self):
        science={"status":"PASS_INDEPENDENT_FULL_V_FP64","artifact_sha256":"a","manifest_sha256":"b","saved_predictions_rescored":True,"new_model_inference":False,"n_observed":20539,"full_V_origins":43}
        builder.validate_validation(science,"rescore","a","b")
        science["saved_predictions_rescored"]=False
        with self.assertRaises(ValueError):builder.validate_validation(science,"rescore","a","b")

    def test_replay_requires_fresh_forward_same_artifact_and_geometry(self):
        science={"status":"PASS_FRESH_CPU6_REPLAY","artifact_sha256":"a","manifest_sha256":"b","executed_now":True,"origins":[14726,14774,14966,15638,15734,16742],"microbatches":[4,2],"F_counters":{"streams":36,"physical_streams":36,"physical_calls":2},"atol":1e-5,"rtol":1e-4}
        builder.validate_validation(science,"replay","a","b")
        for change in ({"executed_now":False},{"artifact_sha256":"foreign"},{"F_counters":{"streams":0,"physical_streams":0,"physical_calls":0}}):
            with self.assertRaises(ValueError):builder.validate_validation(dict(science,**change),"replay","a","b")

    def receipt(self):
        return dict(status="COMPLETE_PROTOCOL",run_id="toy",device="cpu",CPU_threads=2,batch_size=8,warmups=10,timed_trials=20,stage_trials=20,
            actual_trial_rows=50,accuracy_manifest_sha256="m",artifact_sha256="a",trials_sha256="t",quiet_proof_sha256="q",
            validation_result_sha256={"rescore":"r","replay":"p"},F_logical_streams_per_trial=48,F_physical_streams_per_trial=48,F_calls_per_trial=1,
            cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,new_T_target_operations=0,
            physical_input_sha256="i",observation_mask_sha256="o",origins=[14726+i*48 for i in range(8)],
            summaries=[dict(phase="timed",trials=20,batch_size=8,median_seconds=.1,p90_seconds=.2),dict(phase="stage_probe",trials=20,batch_size=8,median_seconds=.11,p90_seconds=.21)],memory={"resident_weights_cpu_MiB":182.})

    def test_whole_cpu2_cost_rejects_cache_wrong_precision_device_or_streams(self):
        receipt=self.receipt();builder.validate_cost(receipt,"toy",8,"a","m","t",{"rescore":"r","replay":"p"})
        for change in ({"device":"cuda"},{"CPU_threads":4},{"cached_Z_or_forecast_reads":1},{"F_calls_per_trial":2},{"physical_output_and_inverse_dtype":"float32"},{"stage_trials":0}):
            with self.assertRaises(ValueError):builder.validate_cost(dict(receipt,**change),"toy",8,"a","m","t",{"rescore":"r","replay":"p"})

    def test_cost_cannot_use_another_checkpoint_manifest_or_validation_sha(self):
        for artifact,manifest,proofs in (("bad","m",{"rescore":"r","replay":"p"}),("a","bad",{"rescore":"r","replay":"p"}),("a","m",{"rescore":"bad","replay":"p"})):
            with self.assertRaises(ValueError):builder.validate_cost(self.receipt(),"toy",8,artifact,manifest,"t",proofs)

    def test_guard_requires_actual_tree_closure_and_zero_exit(self):
        guard=dict(status="COMPLETE",helper_unchanged=True,remaining_owned_processes=[],supervisor_failure_class=None,
            processes=[dict(id="target",returncode=0,stop_reason=None,remaining_owned_processes=[],observed_tree_join_confirmed=True,process_tree_errors=[],signals=[])])
        builder.validate_guard(guard,"target")
        for change in ({"returncode":1},{"remaining_owned_processes":[{"pid":1}]},{"observed_tree_join_confirmed":False}):
            changed=copy.deepcopy(guard);changed["processes"][0].update(change)
            with self.assertRaises(ValueError):builder.validate_guard(changed,"target")
        sibling=dict(id="sibling",returncode=1,stop_reason="WORKER_NONZERO_EXIT",remaining_owned_processes=[],observed_tree_join_confirmed=True,process_tree_errors=[],signals=[])
        shared=copy.deepcopy(guard);shared["status"]="STOPPED_RESOURCE_OR_WORKER_FAILURE";shared["processes"].append(sibling)
        try:builder.validate_guard(shared,"target")
        except ValueError:self.fail("A closed failed sibling must not change the successful exact child's status")

    def test_training_records_actual_selected_epoch_and_segmented_exposures(self):
        receipt={"status":"COMPLETE_VALID","run_id":"toy","family":"B2","seed":92601,"epochs":[.25,1.,3.],"selected_epoch":.25,"search_exposures":29880,"search_updates":1868,"refit_exposures":3016,"refit_updates":189,"refit_scaler_sha":builder.SCALER_SHA}
        search={"status":"SEGMENT_COMPLETE","completed_epochs":[0.,.25,1.,3.],"train_origins":9960,"exposures":29880,"updates":1868,"selected_epoch":.25}
        refit={"status":"SEGMENT_COMPLETE","completed_epochs":[0.,.25],"train_origins":12063,"exposures":3016,"updates":189}
        builder.validate_training(receipt,search,refit,"toy","B2")
        with self.assertRaises(ValueError):builder.validate_training(dict(receipt,refit_exposures=3015),search,refit,"toy","B2")

    def test_input_SHA_mutation_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/"input.json";p.write_text("{}");inputs=builder.Inputs(root,root);inputs.json("public","input.json");p.write_text('{"changed":true}')
            self.assertFalse(inputs.preserved())
            with self.assertRaises(ValueError):inputs.capture("public","../escape.json")

    def test_duplicate_matching_proofs_are_ambiguous(self):
        rows=[("a",self.outer("replay","PASS_FRESH_CPU6_REPLAY")),("b",self.outer("replay","PASS_FRESH_CPU6_REPLAY"))]
        with self.assertRaises(ValueError):builder.select_result(rows,"toy","replay")

    def test_baseline_devices_are_not_promoted_to_same_cpu_scope(self):
        manifest=self.manifest();manifest["device"]="cuda"
        self.assertEqual(builder.baseline_scope(manifest,"RAW"),"REFERENCE_DEVICE_NOT_CPU")
        manifest.pop("device")
        self.assertEqual(builder.baseline_scope(manifest,"DIRECT_G"),"REFERENCE_DEVICE_NOT_RECORDED")
        self.assertEqual(builder.baseline_scope(manifest,"RAW_RIDGE"),"CPU_NATIVE_NUMPY_SOURCE_TRACED")

    def test_csv_keeps_full_precision_and_explicit_row_order(self):
        rows=[{"role":"B2","MSE":.12345678901234567},{"role":"LAG_CAUSAL","MSE":.12}]
        text=builder.csv_text(rows,["role","MSE"]);parsed=list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([r["role"] for r in parsed],["B2","LAG_CAUSAL"])
        self.assertEqual(float(parsed[0]["MSE"]),.12345678901234567)

    def test_existing_output_is_rejected_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scope=root/builder.NAME;scope.mkdir();(scope/builder.OUTPUTS[0]).write_text("keep")
            with self.assertRaises(FileExistsError):builder.execute(root,root)
            self.assertEqual((scope/builder.OUTPUTS[0]).read_text(),"keep")


if __name__=="__main__":unittest.main()
