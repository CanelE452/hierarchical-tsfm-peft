"""Stdlib synthetic core-contrast receipts/trials; actual results are not read."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("audit_core_contrast.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("prepared_core_audit",SOURCE)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
else:audit=None


class CoreAuditTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(audit,"Core auditor not implemented")

    def helper(self):return audit.load_helpers(SOURCE.parents[1])[0]

    def trials(self,k=11,batch=8):
        rows=[]
        for phase,n in (("warmup",10),("timed",20),("stage_probe",20)):
            for i in range(n):
                row=dict(run_id="toy",phase=phase,trial=str(i),order_position="0",batch_size=str(batch),whole_seconds=str(float(i+1)),
                    F_logical_streams=str(batch*k),F_physical_streams=str(batch*k),F_calls="1",prediction_sha256="a"*64,row_microbatch="128",stage_instrumentation=str(phase=="stage_probe"))
                row.update({name:(".125" if phase=="stage_probe" else "") for name in audit.STAGE_FIELDS});rows.append(row)
        return rows

    def test_default_and_missing_ROOT_trigger_read_nothing(self):
        out=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("read")),contextlib.redirect_stdout(out):self.assertEqual(audit.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"],0)
        with patch("builtins.open",side_effect=AssertionError("read")),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):audit.main(["--execute"])

    def test_existing_output_is_preserved_before_input_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"keep.json";output.write_text("keep")
            with self.assertRaises(FileExistsError):audit.execute(folder,folder,"x","x","x",output)
            self.assertEqual(output.read_text(),"keep")

    def test_two_conditions_have_eight_exact_POST_steps(self):
        post=dict(status="COMPLETE_TWO_VALIDATIONS_AND_FOUR_COSTS",skipped=[],error=None,pins_before_after_unchanged=True,changed_or_unreadable_pinned_paths=[],steps=[dict(id=k,success=True,strong_observed_tree_closure_verified=True) for k in audit.expected_steps()])
        audit.validate_post(post)
        for change in ({"status":"PARTIAL_VALIDATIONS_COSTS_PRESERVED"},{"steps":post["steps"][:-1]},{"steps":post["steps"][:-1]+[post["steps"][0]]}):
            with self.assertRaises(ValueError):audit.validate_post(dict(post,**change))

    def test_K11_cost_counters_are_not_silently_K6(self):
        result=audit.aggregate_trials(self.trials(),"toy",8,11)
        self.assertEqual((result["trial_rows"],result["F_logical_streams"],result["F_calls"]),(50,4400,50))
        self.assertEqual(result["summaries"][0]["mean_seconds"],10.5)
        self.assertEqual(result["summaries"][0]["median_seconds"],10.5)
        with self.assertRaises(ValueError):audit.aggregate_trials(self.trials(),"toy",8,6)

    def test_missing_duplicate_trial_wrong_stage_or_F_call_is_rejected(self):
        for change in ({"trial":"1"},{"F_calls":"2"},{"stage_instrumentation":"True"}):
            rows=self.trials();rows[0].update(change)
            with self.assertRaises(ValueError):audit.aggregate_trials(rows,"toy",8,11)
        with self.assertRaises(ValueError):audit.aggregate_trials(self.trials()[:-1],"toy",8,11)

    def test_cost_summary_one_ULP_is_rejected_without_new_tolerance(self):
        h=self.helper();summaries=audit.aggregate_trials(self.trials(),"toy",8,11)["summaries"]
        saved=copy.deepcopy(summaries);self.assertEqual(h.compare_summaries(summaries,saved),0.)
        saved[0]["mean_seconds"]=math.nextafter(saved[0]["mean_seconds"],math.inf)
        with self.assertRaises(ValueError):h.compare_summaries(summaries,saved)

    def test_reused_macro_helper_keeps_channel_balancing(self):
        h=self.helper();rows=[]
        for channel,count,sse,sae in (("a",1,4.,2.),("b",2,2.,2.)):
            rows.append(dict(run_id="toy",split="V_select",cohort=h.COHORT,origin="100",channel=channel,horizon_start="0",horizon_end="2",count=str(count),sse=str(sse),sae=str(sae)))
        got=h.aggregate_statistics(rows,"toy",("a","b"),(100,),((0,2),),3)
        self.assertEqual((got["mse"],got["mae"]),(2.5,1.5));self.assertNotEqual(got["mse"],6/3)

    def test_validation_binds_actual_replay_width_and_same_artifact(self):
        science=dict(status="PASS_FRESH_CPU6_REPLAY",executed_now=True,artifact_sha256="a",manifest_sha256="m",origins=[14726,14774,14966,15638,15734,16742],microbatches=[4,2],F_counters=dict(streams=66,physical_streams=66,physical_calls=2),atol=1e-5,rtol=1e-4)
        audit.validate_validation(science,"replay","a","m",11)
        for change in ({"executed_now":False},{"artifact_sha256":"foreign"},{"F_counters":dict(streams=36,physical_streams=36,physical_calls=2)}):
            with self.assertRaises(ValueError):audit.validate_validation(dict(science,**change),"replay","a","m",11)

    def test_guard_requires_exact_spec_tree_and_STARTED_creation_identity(self):
        started=dict(actual_pid=123,actual_create_time=10.)
        command=["python","--run-id","toy","--mode","replay","--plan-sha","plan","--output","output"]
        process=dict(id="step",root=dict(pid=123,create_time=10.),returncode=0,stop_reason=None,end_unix=12.,owned_processes=[dict(pid=123,create_time=10.)],remaining_owned_processes=[],process_tree_errors=[],observed_tree_join_confirmed=True,signals=[],sampled_peak_RSS_bytes=100)
        effective=dict(schema="CPU_GROUP_SUPERVISOR_v1",asset_root="asset",cache_root="cache",output_root="output",max_each_RSS=3221225472,max_group_RSS=3221225472,deadline_unix=1792222550,poll_seconds=2,terminate_grace_seconds=10,kill_grace_seconds=5,processes=[dict(id="step",command=command,cwd="cwd",max_seconds=600)])
        guard=dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1",status="COMPLETE",helper_sha256=audit.GUARD_SHA,helper_unchanged=True,spec_file_sha256="spec",supervisor_failure_class=None,remaining_owned_processes=[],processes=[process],effective_spec=effective,start_unix=9.,end_unix=12.,sampled_group_peak_RSS_bytes=100)
        spec=copy.deepcopy(effective)
        audit.validate_guard(guard,spec,"spec","step",command,started)
        for change in ({"actual_pid":124},{"actual_create_time":11.}):
            with self.assertRaises(ValueError):audit.validate_guard(guard,spec,"spec","step",command,dict(started,**change))
        bad=copy.deepcopy(guard);bad["processes"][0]["process_tree_errors"]=["denied"]
        with self.assertRaises(ValueError):audit.validate_guard(bad,spec,"spec","step",command,started)
        changed=copy.deepcopy(spec);changed["processes"][0]["max_seconds"]=601
        with self.assertRaises(ValueError):audit.validate_guard(guard,changed,"spec","step",command,started)
        changed=copy.deepcopy(guard);changed["sampled_group_peak_RSS_bytes"]=3221225473
        with self.assertRaises(ValueError):audit.validate_guard(changed,spec,"spec","step",command,started)

    def test_new_and_reused_band_counts_must_match(self):
        audit.require_count_binding([dict(band_count_binding_sha256="same"),dict(band_count_binding_sha256="same")],dict(band_count_binding_sha256="same"))
        with self.assertRaises(ValueError):audit.require_count_binding([dict(band_count_binding_sha256="foreign")],dict(band_count_binding_sha256="same"))


if __name__=="__main__":unittest.main()
