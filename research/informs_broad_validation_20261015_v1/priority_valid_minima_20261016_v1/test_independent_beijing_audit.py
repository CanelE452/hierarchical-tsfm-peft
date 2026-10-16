"""Stdlib toy sufficient-statistic/trial fixtures; no actual campaign values."""
import contextlib
import copy
import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("independent_beijing_audit.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("prepared_independent_beijing_audit",SOURCE)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
else:audit=None


class IndependentArithmeticTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(audit,"Independent auditor not implemented")

    def stats(self):
        rows=[]
        for origin in (100,148):
            for channel,count,sse,sae in (("a",1,4.,2.),("b",2,2.,2.)):
                rows.append(dict(run_id="toy",split="V_select",cohort=audit.COHORT,origin=str(origin),channel=channel,
                    horizon_start="0",horizon_end="2",sse=str(sse),sae=str(sae),count=str(count)))
        return rows

    def aggregate(self,rows):
        return audit.aggregate_statistics(rows,"toy",("a","b"),(100,148),((0,2),),6)

    def trials(self,batch=1):
        rows=[]
        for phase,n in (("warmup",10),("timed",20),("stage_probe",20)):
            for trial in range(n):
                row=dict(run_id="toy",phase=phase,trial=str(trial),order_position="0",batch_size=str(batch),
                    whole_seconds=str(float(trial+1)),F_logical_streams=str(batch*6),F_physical_streams=str(batch*6),F_calls="1",
                    prediction_sha256="a"*64,row_microbatch="128",stage_instrumentation=str(phase=="stage_probe"))
                row.update({key:("0.125" if phase=="stage_probe" else "") for key in audit.STAGE_FIELDS});rows.append(row)
        return rows

    def test_default_and_missing_root_trigger_do_no_io(self):
        stream=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("input read")),contextlib.redirect_stdout(stream):self.assertEqual(audit.main([]),0)
        self.assertEqual(json.loads(stream.getvalue())["input_reads"],0)
        with patch("builtins.open",side_effect=AssertionError("input read")),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):audit.main(["--execute"])

    def test_macro_uses_channel_global_counts_not_global_or_origin_average(self):
        result=self.aggregate(self.stats())
        self.assertEqual((result["mse"],result["mae"],result["n_observed"]),(2.5,1.5,6))
        self.assertNotEqual(result["mse"],12./6)
        self.assertEqual(result["public_stats_rows"],4)

    def test_missingness_does_not_turn_metric_into_mean_of_origin_scores(self):
        rows=self.stats();rows[0].update(sse="9",sae="3");rows[2].update(count="2",sse="2",sae="2")
        rows[1].update(sse="0",sae="0");rows[3].update(sse="0",sae="0")
        result=audit.aggregate_statistics(rows,"toy",("a","b"),(100,148),((0,2),),7)
        self.assertEqual(result["mse"],11./6.)
        self.assertNotEqual(result["mse"],2.5)

    def test_zero_observation_rows_are_required_and_preserved(self):
        rows=self.stats();rows[0].update(count="0",sse="0",sae="0")
        result=audit.aggregate_statistics(rows,"toy",("a","b"),(100,148),((0,2),),5)
        self.assertEqual(result["public_stats_rows"],4)
        with self.assertRaises(ValueError):audit.aggregate_statistics(rows[1:],"toy",("a","b"),(100,148),((0,2),),5)

    def test_missing_duplicate_or_foreign_origin_is_rejected(self):
        for changed in (self.stats()[:-1],self.stats()+[self.stats()[0]],self.stats()):
            if len(changed)==4:changed[0]["origin"]="16772"
            with self.assertRaises(ValueError):self.aggregate(changed)

    def test_count_bounds_nonfinite_and_nonzero_empty_errors_rejected(self):
        for change in ({"count":"3"},{"sse":"nan"},{"sae":"-1"},{"count":"0"}):
            rows=self.stats();rows[0].update(change)
            with self.assertRaises(ValueError):self.aggregate(rows)

    def test_same_mask_counts_required_across_three_conditions(self):
        a=self.aggregate(self.stats());b=copy.deepcopy(a);audit.require_same_counts([a,b])
        b["count_by_key"][0]["count"]+=1
        with self.assertRaises(ValueError):audit.require_same_counts([a,b])

    def test_original_accuracy_tolerance_is_used_but_scores_exactly_bind_manifest(self):
        got=self.aggregate(self.stats());manifest=dict(score=dict(mse=2.5,mae=1.5,n_observed=6,active_channels=2))
        score=dict(mse="2.5",mae="1.5",n_observed="6",active_channels="2");receipt=dict(mse=2.5+1e-8,mae=1.5,n_observed=6,public_stats_rows=4)
        audit.compare_accuracy(got,manifest,score,receipt)
        with self.assertRaises(ValueError):audit.compare_accuracy(got,manifest,dict(score,mse="2.500000000000001"),receipt)
        with self.assertRaises(ValueError):audit.compare_accuracy(got,manifest,score,dict(receipt,mse=2.6))
        with self.assertRaises(ValueError):audit.compare_accuracy(got,dict(score=dict(manifest["score"],active_channels=1)),score,receipt)

    def test_cost_50_rows_and_dynamic_batch_streams_are_recomputed(self):
        got=audit.aggregate_trials(self.trials(8),"toy",8)
        self.assertEqual((got["trial_rows"],got["F_logical_streams"],got["F_physical_streams"],got["F_calls"]),(50,2400,2400,50))
        self.assertEqual(got["phase_rows"],dict(warmup=10,timed=20,stage_probe=20))
        self.assertEqual(len(got["summaries"]),2)
        self.assertEqual(got["summaries"][0]["mean_seconds"],10.5)
        self.assertEqual(got["summaries"][0]["median_seconds"],10.5)
        self.assertAlmostEqual(got["summaries"][0]["p90_seconds"],18.1)

    def test_cost_rejects_duplicate_trial_partial_stream_or_wrong_stage_flag(self):
        for field,value in (("trial","1"),("F_calls","2"),("F_logical_streams","0"),("stage_instrumentation","True")):
            rows=self.trials();rows[0][field]=value
            with self.assertRaises(ValueError):audit.aggregate_trials(rows,"toy",1)
        with self.assertRaises(ValueError):audit.aggregate_trials(self.trials()[:-1],"toy",1)

    def test_whole_and_stage_times_have_distinct_validated_scopes(self):
        rows=self.trials();audit.aggregate_trials(rows,"toy",1)
        rows[0][audit.STAGE_FIELDS[0]]="0.1"
        with self.assertRaises(ValueError):audit.aggregate_trials(rows,"toy",1)
        rows=self.trials();rows[-1][audit.STAGE_FIELDS[0]]="-0.1"
        with self.assertRaises(ValueError):audit.aggregate_trials(rows,"toy",1)

    def test_all_summary_numbers_match_without_new_cost_tolerance(self):
        got=audit.aggregate_trials(self.trials(),"toy",1);saved=copy.deepcopy(got["summaries"])
        self.assertEqual(audit.compare_summaries(got["summaries"],saved),0.)
        saved[0]["mean_seconds"]+=1e-12
        with self.assertRaises(ValueError):audit.compare_summaries(got["summaries"],saved)

    def test_cost_intervals_cannot_overlap_fits_or_other_costs(self):
        costs=[dict(id="a",start=10.,end=11.),dict(id="b",start=11.,end=12.)]
        audit.validate_intervals(costs,[dict(id="fit",start=1.,end=9.)])
        with self.assertRaises(ValueError):audit.validate_intervals(costs,[dict(id="fit",start=8.,end=10.1)])
        with self.assertRaises(ValueError):audit.validate_intervals([costs[0],dict(id="b",start=10.9,end=12.)],[])

    def test_partial_or_duplicate_post_steps_are_not_complete(self):
        expected=audit.expected_steps();post=dict(status="COMPLETE_THREE_VALIDATIONS_AND_SIX_COSTS",steps=[dict(id=k,success=True) for k in expected],skipped=[])
        audit.validate_post(post)
        with self.assertRaises(ValueError):audit.validate_post(dict(post,status="PARTIAL_VALIDATIONS_COSTS_PRESERVED"))
        with self.assertRaises(ValueError):audit.validate_post(dict(post,steps=post["steps"][:-1]+[post["steps"][0]]))

    def test_gzip_reader_requires_actual_sha_and_bounded_rows(self):
        class Inputs:
            def capture(self,root,name,expected=None):
                if expected is not None and audit.sha(path)!=expected:raise ValueError("changed")
                return path
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"toy.csv.gz"
            with gzip.open(path,"wt",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["a"]);writer.writeheader();writer.writerows([{"a":1},{"a":2}])
            self.assertEqual(len(audit.gzip_rows(Inputs(),"public","toy",2,audit.sha(path))),2)
            with self.assertRaises(ValueError):audit.gzip_rows(Inputs(),"public","toy",1)
            with self.assertRaises(ValueError):audit.gzip_rows(Inputs(),"public","toy",2,"f"*64)

    def test_existing_output_is_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/"keep.json";output.write_text("keep")
            with self.assertRaises(FileExistsError):audit.execute(root,root,"x","x","x",output)
            self.assertEqual(output.read_text(),"keep")

    def test_incomplete_or_disappeared_inputs_write_an_honest_rejection(self):
        for disappeared in (False,True):
            with self.subTest(disappeared=disappeared),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);scope=root/audit.NAME;scope.mkdir();source=scope/"independent_beijing_audit.py";source.write_text("synthetic source")
                class Inputs:
                    def __init__(self,*args):self.files={}
                    def capture(self,root,name,expected):self.files[root,name]=expected
                    def preserved(self):
                        if disappeared:raise FileNotFoundError("synthetic deleted input")
                        return True
                helper=types.SimpleNamespace(Inputs=Inputs);output=scope/"audit.json"
                with patch.object(audit,"__file__",str(source)),patch.object(audit,"load_builder",return_value=helper), \
                        patch.object(audit,"collect",side_effect=ValueError("synthetic incomplete POST")):
                    try:value=audit.execute(root,root,audit.PLAN_SHA,"a"*64,audit.sha(source),output)
                    except FileNotFoundError:value=None
                self.assertIsNotNone(value,"Missing input must remain a recorded incomplete audit")
                self.assertEqual(value["status"],"NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED")
                self.assertIsNotNone(value["error"])
                self.assertEqual(json.loads(output.read_text())["status"],value["status"])


if __name__=="__main__":unittest.main()
