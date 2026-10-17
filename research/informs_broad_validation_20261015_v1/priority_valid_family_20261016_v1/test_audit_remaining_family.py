"""Synthetic stdlib-only contracts; no actual campaign results or subprocesses."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("audit_remaining_family.py")


def load():
    s=importlib.util.spec_from_file_location("remaining_audit_tiny",SOURCE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def training(a,dataset="jena",epoch=0.):
    t=next(r for r in a.targets() if r["dataset"]==dataset);n=t["search_end"]-559;nr=t["refit_end"]-559
    endpoints=[0]+[math.ceil(e*n) for e in (.25,1.,3.)];updates=sum(math.ceil((b-c)/16) for c,b in zip(endpoints,endpoints[1:]))
    search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],selected_epoch=epoch,train_origins=n,updates=updates,exposures=3*n,seconds=1.)
    refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),selected_epoch=epoch,train_origins=nr,updates=math.ceil(math.ceil(epoch*nr)/16),exposures=math.ceil(epoch*nr),seconds=.25)
    v=dict(status="COMPLETE_VALID",run_id=t["run_id"],family=t["family"],seed=92601,epochs=[.25,1.,3.],selected_epoch=epoch,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",
        search_fit_end=t["search_end"],refit_fit_end=t["refit_end"],search_updates=updates,search_exposures=3*n,
        refit_updates=refit["updates"],refit_exposures=refit["exposures"],refit_scaler_sha="s",optimizer="AdamW",weight_decay=0.)
    return t,v,search,refit


class Exists(unittest.TestCase):
    def test_implementation_exists(self):self.assertTrue(SOURCE.exists(),"Remaining-family auditor is not implemented")


class Tests(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest("Not implemented")
        self.a=load()

    def test_default_zero_IO_and_no_scientific_imports(self):
        text=io.StringIO()
        with patch.object(Path,"open",side_effect=AssertionError("Unexpected input")),contextlib.redirect_stdout(text):self.assertEqual(self.a.main([]),0)
        v=json.loads(text.getvalue());self.assertEqual(v["input_reads"],0);self.assertEqual(v["files_written"],0)
        self.assertFalse(v["numpy_imported"]);self.assertFalse(v["torch_imported"])

    def test_exact_two_dataset_contracts(self):
        a,b=self.a.targets();self.assertEqual((a["C"],a["latent_k"],a["observed"],a["stat_rows"]),(21,6,43344,2709))
        self.assertEqual((b["C"],b["latent_k"],b["observed"],b["stat_rows"]),(11,11,20539,1419))
        self.assertNotEqual(a["origins"],b["origins"])

    def test_zero_epoch_is_valid_for_both_datasets(self):
        for dataset in ("jena","beijing"):
            t,v,s,r=training(self.a,dataset,0.);self.a.validate_training(v,s,r,t,"s")
            self.assertEqual(r["updates"],0);self.assertEqual(r["exposures"],0)

    def test_selected_endpoint_activity_is_not_assumed_three(self):
        for dataset in ("jena","beijing"):
            for epoch in (.25,1.,3.):
                t,v,s,r=training(self.a,dataset,epoch);self.a.validate_training(v,s,r,t,"s")
                changed=copy.deepcopy(r);changed["updates"]+=1
                with self.assertRaises(ValueError):self.a.validate_training(v,s,changed,t,"s")

    def test_foreign_dataset_or_false_convergence_rejected(self):
        t,v,s,r=training(self.a,"jena",0.)
        v["training_status"]="CONVERGED"
        with self.assertRaises(ValueError):self.a.validate_training(v,s,r,t,"s")
        t,v,s,r=training(self.a,"jena",1.);t["refit_end"]=12622
        with self.assertRaises(ValueError):self.a.validate_training(v,s,r,t,"s")

    def test_two_cohorts_aggregate_without_relabeling(self):
        rows=[]
        for cohort in ("jena:primary:V_select:stride48","beijing:primary:V_select:stride48"):
            for channel,multiplier in (("a",1.),("b",4.)):
                for start,stop in ((0,12),(12,24),(24,48)):
                    n=1 if channel=="a" else stop-start
                    rows.append(dict(run_id="synthetic",split="V_select",cohort=cohort,origin="100",channel=channel,horizon_start=str(start),horizon_end=str(stop),sse=str(n*multiplier),sae=str(n),count=str(n)))
        for start,cohort in ((0,"jena:primary:V_select:stride48"),(6,"beijing:primary:V_select:stride48")):
            r=self.a.aggregate_statistics(rows[start:start+6],"synthetic",["a","b"],[100],51,cohort)
            self.assertEqual(r["mse"],2.5);self.assertNotEqual(r["mse"],195/51)
        with self.assertRaises(ValueError):self.a.aggregate_statistics(rows[:6],"synthetic",["a","b"],[100],51,"beijing:primary:V_select:stride48")

    def test_missing_duplicate_or_mask_mismatch_rejected(self):
        row=dict(run_id="r",split="V_select",cohort="jena:primary:V_select:stride48",origin="100",channel="a",horizon_start="0",horizon_end="12",sse="1",sae="1",count="1")
        with self.assertRaises(ValueError):self.a.aggregate_statistics([row,row],"r",["a"],[100],1,row["cohort"])

    def test_partial_evidence_never_claimed_COMPLETE(self):
        steps=[dict(id=k,run_id=r["run_id"],mode=r["mode"],batch=r["batch"],success=True) for k,r in self.a.expected_steps().items()]
        self.a.validate_post(dict(status="COMPLETE_REMAINING_TWO_VALIDATIONS_AND_FOUR_COSTS",steps=steps,skipped=[],error=None))
        with self.assertRaises(ValueError):self.a.validate_post(dict(status="PARTIAL_REMAINING_FAMILY_VALIDATIONS_COSTS_PRESERVED",steps=steps,skipped=[],error=None))

    def test_deadline_argparse_float_not_string_identity(self):
        self.a.deadline_arg(["--deadline","1792221950"],1792221950.0,True)
        with self.assertRaises(ValueError):self.a.deadline_arg(["--deadline","1792221950"],1792221950.0,False)

    def test_frozen_helpers_remain_stdlib_and_summary_ULP_strict(self):
        h=self.a.load_helpers(SOURCE.parent.parent);self.assertFalse("numpy" in h["cost"].sys.modules)
        actual=[dict(run_id="x",phase=p,batch_size=1,trials=20,median_seconds=.2,mean_seconds=.2,min_seconds=.1,max_seconds=.3,p90_seconds=.29,stage_instrumentation=p=="stage_probe",stage_sum_equals_whole_claimed=False) for p in ("timed","stage_probe")]
        h["arithmetic"].compare_summaries(actual,copy.deepcopy(actual));changed=copy.deepcopy(actual);changed[0]["mean_seconds"]=math.nextafter(.2,math.inf)
        with self.assertRaises(ValueError):h["arithmetic"].compare_summaries(actual,changed)

    def test_family_ADAPTS_B4_six_stage_header_and_K11_counters(self):
        h=self.a.load_helpers(SOURCE.parent.parent)["cost"]
        for k in (6,11):
            rows=[]
            for phase,n in (("warmup",10),("timed",20),("stage_probe",20)):
                for i in range(n):
                    row=dict(run_id="r",phase=phase,trial=str(i),order_position="0",batch_size="8",whole_seconds=str(.25+i/128),
                        F_logical_streams=str(8*k),F_physical_streams=str(8*k),F_calls="1",prediction_sha256="a"*64,row_microbatch="128",stage_instrumentation=str(phase=="stage_probe"))
                    row.update({f:"0.001" if phase=="stage_probe" else "" for f in h.STAGES});rows.append(row)
            result=h.aggregate_trials(rows,"r",8,k)
            self.assertEqual(result["F_calls"],50);self.assertEqual(result["F_logical_streams"],400*k)
            rows[0].pop("stage_P_E_seconds")
            with self.assertRaises(ValueError):h.aggregate_trials(rows,"r",8,k)

    def test_existing_output_refused_before_inputs(self):
        with patch.object(Path,"exists",return_value=True),patch.object(Path,"open",side_effect=AssertionError("Input")):
            with self.assertRaises(FileExistsError):self.a.execute("p","c",{},"s","already.json")


if __name__=="__main__":unittest.main()
