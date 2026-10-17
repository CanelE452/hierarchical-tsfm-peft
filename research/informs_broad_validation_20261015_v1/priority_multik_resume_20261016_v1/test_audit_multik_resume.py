"""Small stdlib-only fixtures; no actual campaign inputs or subprocesses."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("audit_multik_resume.py")


def load():
    spec=importlib.util.spec_from_file_location("n7_audit_tiny",SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parent(a):
    c=a.parents()[0]
    v=dict(status="COMPLETE_VALID",run_id=c["run_id"],family="B0",seed=92601,epochs=[.25,1.,3.],selected_epoch=3.,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT",full_V_completed=False,actual_saved_point_resume=True,
        search_updates=1868,search_exposures=29880,refit_updates=2262,refit_exposures=36189,
        search_fit_end=10519,refit_fit_end=12622,search_train_origins=9960,refit_train_origins=12063,
        search_new_optimizer_updates=0,search_new_F_calls=0,refit_scaler_sha=a.SCALER,
        artifact_sha256="a"*64,model_state_sha="b"*64,artifact_cache_relative_path="runs/"+c["run_id"]+"/selected_bundle.pt",
        source_stage_API="original_fit_base._stage",old_unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED",
        search_seconds_this_resume_call=.1,historical_saved_search_segment_seconds=1.,refit_seconds_this_resume_call=2.,
        search_total_wall_seconds_known=False,full_interrupted_instant_resume=False,uninterrupted_bitwise_parity_claimed=False,
        independent_K_training_completed=False,new_T_target_operations=0)
    search=dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,.25,1.,3.],train_origins=9960,
        updates=1868,exposures=29880,F_session_logical_streams=0,F_session_physical_streams=0,
        contract=dict(scaler_sha=a.SEARCH_SCALER))
    refit=dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,3.],train_origins=12063,
        updates=2262,exposures=36189,contract=dict(scaler_sha=a.SCALER))
    return c,v,search,refit


def trials(a,k=3,batch=8):
    rows=[]
    for phase,count in (("warmup",10),("timed",20),("stage_probe",20)):
        for index in range(count):
            row=dict(run_id="synthetic",phase=phase,trial=str(index),order_position="0",batch_size=str(batch),
                whole_seconds=str(.125+index/128),F_logical_streams=str(batch*k),F_physical_streams=str(batch*k),
                F_calls="1",prediction_sha256="a"*64,row_microbatch="128",stage_instrumentation=str(phase=="stage_probe"))
            row.update({key:"0.001" if phase=="stage_probe" else "" for key in a.STAGES});rows.append(row)
    return rows


class Tests(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest("Auditor not implemented yet")
        self.a=load()

    def test_default_no_IO_or_scientific_imports(self):
        stream=io.StringIO()
        with patch.object(Path,"open",side_effect=AssertionError("Unexpected I/O")),contextlib.redirect_stdout(stream):
            self.assertEqual(self.a.main([]),0)
        value=json.loads(stream.getvalue())
        self.assertEqual(value["input_reads"],0);self.assertEqual(value["files_written"],0)
        self.assertFalse(value["numpy_imported"]);self.assertFalse(value["torch_imported"])

    def test_exact_two_parents_six_aliases_are_distinct(self):
        a=self.a;rows=a.outputs();self.assertEqual(len({r["run_id"] for r in rows}),6)
        self.assertEqual([(c["kind"],c["grid"]) for c in a.parents()],[("shared",[3,6,11]),("max_only",[11])])
        self.assertEqual([r["latent_k"] for r in rows],[3,6,11,3,6,11])
        self.assertEqual(rows[-1]["variant"],"max_only")

    def test_parent_retains_no_V_and_exact_saved_activity(self):
        c,v,s,r=parent(self.a);self.a.validate_parent(v,s,r,c)
        for field,value in (("full_V_completed",True),("search_new_F_calls",1),("refit_updates",2261),("training_status","CONVERGED")):
            changed=copy.deepcopy(v);changed[field]=value
            with self.assertRaises(ValueError):self.a.validate_parent(changed,s,r,c)

    def test_search_session_F_is_zero_but_cumulative_is_not_fabricated(self):
        c,v,s,r=parent(self.a);s["F_total_logical_streams"]=12345
        self.a.validate_parent(v,s,r,c)
        s["F_session_logical_streams"]=1
        with self.assertRaises(ValueError):self.a.validate_parent(v,s,r,c)

    def test_strict_partial_POST_never_PASS(self):
        rows=[dict(id=k,run_id=v["run_id"],mode=v["mode"],batch=v["batch"],success=True) for k,v in self.a.expected_steps().items()]
        post=dict(status="COMPLETE_SIX_PREFIXES_TWELVE_VALIDATIONS_TWELVE_COSTS",steps=rows,skipped=[],error=None)
        self.a.validate_post(post)
        post["status"]="PARTIAL_MULTIK_PREFIX_VALIDATIONS_COSTS_PRESERVED"
        with self.assertRaises(ValueError):self.a.validate_post(post)
        post["status"]="COMPLETE_SIX_PREFIXES_TWELVE_VALIDATIONS_TWELVE_COSTS";post["steps"][0]["success"]=False
        with self.assertRaises(ValueError):self.a.validate_post(post)

    def test_B0_six_stage_trial_schema_and_dynamic_K(self):
        for k in (3,6,11):
            v=self.a.aggregate_trials(trials(self.a,k),"synthetic",8,k)
            self.assertEqual(v["trial_rows"],50);self.assertEqual(v["F_logical_streams"],50*8*k)
            self.assertEqual(v["F_calls"],50);self.assertEqual(len(v["summaries"]),2)
        with self.assertRaises(ValueError):self.a.aggregate_trials(trials(self.a,21),"synthetic",8,21)

    def test_trial_missing_stage_duplicate_or_wrong_counter_rejected(self):
        for mutation in (lambda r:r[0].pop("stage_D_seconds"),lambda r:r.__setitem__(1,copy.deepcopy(r[0])),
                lambda r:r[0].__setitem__("F_calls","2"),lambda r:r[0].__setitem__("stage_F_seconds","0")):
            rows=trials(self.a);mutation(rows)
            with self.assertRaises(ValueError):self.a.aggregate_trials(rows,"synthetic",8,3)

    def test_native_cost_does_not_require_unlogged_totals_or_dtype(self):
        a=self.a;row=a.outputs()[0];v=dict(status="COMPLETE_PROTOCOL",run_id=row["run_id"],artifact_sha256="a"*64,
            manifest_sha256="b"*64,device="cpu",CPU_threads=2,batch_size=8,warmups=10,timed_trials=20,
            stage_trials=20,actual_trial_rows=50,physical_input_sha256="c"*64,observation_mask_sha256="d"*64,
            origins=[14726+48*i for i in range(8)],trials_sha256="e"*64,quiet_window_proof_sha256="f"*64,
            memory=dict(foundation_instances_created=1,resident_weights_cpu_MiB=182.),
            cached_Z_or_forecast_reads=0,training_peak_included=False,new_T_target_operations=0,
            source_protocol="original_measure_costs.measure_balanced + original deployment.Route constructor",
            stage_probe_separate_intrusive_scope=True)
        a.validate_cost(v,row,8,"a"*64,"b"*64,"f"*64)
        self.assertNotIn("actual_F_counters",v);self.assertNotIn("physical_output_dtype",v)
        v["origins"][0]+=1
        with self.assertRaises(ValueError):a.validate_cost(v,row,8,"a"*64,"b"*64,"f"*64)

    def test_fingerprint_spec_is_not_K_independent(self):
        a=self.a;c,v,_,_=parent(a);x=a.outputs()[0];y=a.outputs()[1]
        self.assertNotEqual(a.fingerprint(a.route_spec(x,v)),a.fingerprint(a.route_spec(y,v)))
        self.assertEqual(a.route_spec(x,v)["canonical_fit_end"],12622)
        self.assertEqual(a.route_spec(x,v)["prefix_variant"],"shared")

    def test_CPU6_must_be_actual_4plus2_and_include_row_calls(self):
        a=self.a;r=a.outputs()[0]
        v=dict(status="PASS_FRESH_CPU6_REPLAY",run_id=r["run_id"],artifact_sha256="a"*64,manifest_sha256="b"*64,
            executed_now=True,origins=[14726,14774,14822,15734,16742,16694],microbatches=[4,2],
            F_counters=dict(streams=18,physical_streams=18,row_calls=2,physical_calls=2),atol=1e-5,rtol=1e-4,new_T_target_operations=0)
        a.validate_validation(v,"replay",r,"a"*64,"b"*64)
        v["executed_now"]=False
        with self.assertRaises(ValueError):a.validate_validation(v,"replay",r,"a"*64,"b"*64)
        v["executed_now"]=True;v["F_counters"].pop("row_calls")
        with self.assertRaises(ValueError):a.validate_validation(v,"replay",r,"a"*64,"b"*64)

    def test_argv_exact_identity_and_duplicate_flags(self):
        a=self.a;cmd=["python","entry.py","--run-id","x","--mode","cost","--plan-sha","a"*64]
        a.require_argv(cmd,"--run-id","x")
        with self.assertRaises(ValueError):a.require_argv(cmd+["--run-id","x"],"--run-id","x")

    def test_existing_output_rejected_before_any_input_read(self):
        with patch.object(Path,"exists",return_value=True),patch.object(Path,"open",side_effect=AssertionError("Read")):
            with self.assertRaises(FileExistsError):self.a.execute("p","c",{},"a"*64,"already.json")

    def test_integer_argv_and_float_STARTED_deadline_have_same_value(self):
        self.assertTrue(hasattr(self.a,"validate_deadline"),"Need numeric comparison for original argparse float deadline")
        self.a.validate_deadline(["--deadline","1792221950"],1792221950.0,True)
        with self.assertRaises(ValueError):self.a.validate_deadline(["--deadline","1792221951"],1792221950.0,True)
        with self.assertRaises(ValueError):self.a.validate_deadline(["--deadline","1792221950"],1792221950.0,False)

    def test_frozen_arithmetic_macro_and_exact_cost_summary_policy(self):
        helpers=self.a.load_helpers(SOURCE.parent.parent);h=helpers["arithmetic"]
        rows=[]
        for channel,multiplier in (("a",1.),("b",4.)):
            for start,stop in ((0,12),(12,24),(24,48)):
                n=1 if channel=="a" else stop-start
                rows.append(dict(run_id="synthetic",split="V_select",cohort=h.COHORT,origin="14726",channel=channel,
                    horizon_start=str(start),horizon_end=str(stop),sse=str(n*multiplier),sae=str(n),count=str(n)))
        result=h.aggregate_statistics(rows,"synthetic",["a","b"],[14726],((0,12),(12,24),(24,48)),51)
        self.assertEqual(result["mse"],2.5);self.assertNotEqual(result["mse"],195/51)
        value=self.a.aggregate_trials(trials(self.a),"synthetic",8,3)["summaries"]
        h.compare_summaries(value,copy.deepcopy(value))
        changed=copy.deepcopy(value)
        import math
        changed[0]["mean_seconds"]=math.nextafter(changed[0]["mean_seconds"],math.inf)
        with self.assertRaises(ValueError):h.compare_summaries(value,changed)
        with self.assertRaises(ValueError):h.validate_intervals([dict(start=1.,end=3.)],[dict(start=2.,end=4.)])

    def test_frozen_guard_rejects_open_tree_or_unowned_STARTED_identity(self):
        g=self.a.load_helpers(SOURCE.parent.parent)["guard"]
        command=["python","synthetic"]
        child=dict(id="toy",root=dict(pid=123,create_time=1.),returncode=0,end_unix=10.,stop_reason=None,signals=[],
            owned_processes=[dict(pid=123,create_time=1.)],remaining_owned_processes=[],process_tree_errors=[],
            observed_tree_join_confirmed=True,sampled_peak_RSS_bytes=100)
        spec=dict(schema="CPU_GROUP_SUPERVISOR_v1",deadline_unix=self.a.DEADLINE,max_each_RSS=3221225472,max_group_RSS=3221225472,
            poll_seconds=2,terminate_grace_seconds=10,kill_grace_seconds=5,asset_root="synthetic",cache_root="synthetic/cache",
            output_root="synthetic/out",processes=[dict(id="toy",command=command,max_seconds=600,cwd="synthetic")])
        guard=dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1",status="COMPLETE",helper_sha256=self.a.GUARD_SHA,helper_unchanged=True,
            spec_file_sha256="a"*64,supervisor_failure_class=None,remaining_owned_processes=[],sampled_group_peak_RSS_bytes=100,
            effective_spec=copy.deepcopy(spec),processes=[child])
        started=dict(actual_pid=123,actual_create_time=1.)
        g.validate_guard(guard,spec,"a"*64,"toy",command,started)
        started["actual_pid"]=124
        with self.assertRaises(ValueError):g.validate_guard(guard,spec,"a"*64,"toy",command,started)
        started["actual_pid"]=123;child["remaining_owned_processes"]=[dict(pid=123,create_time=1.)]
        with self.assertRaises(ValueError):g.validate_guard(guard,spec,"a"*64,"toy",command,started)


class ImplementationExists(unittest.TestCase):
    def test_new_auditor_implementation_exists(self):self.assertTrue(SOURCE.exists(),"N7 auditor implementation is still missing")


if __name__=="__main__":unittest.main()
