"""Stdlib diagnostic fixtures only; ROOT executes after precision costs stop."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tiny_reference_raw_diagnostic",HERE/"diagnose_reference_raw_row_parity.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RawRowDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.h=load()

    def returned_parity(self,first_blocked=False):
        variants=[]
        for index,row in enumerate(("full",32,8,4)):
            if index==0 and first_blocked:variants.append(dict(row_microbatch=row,status="BLOCKED_CUDA_OOM"))
            else:variants.append(dict(row_microbatch=row,status="FAIL" if index==3 else "PASS",max_abs=.25 if index==3 else 0.,
                prediction_sha256=str(index)*64,F_logical_streams=21))
        return dict(status="CHECKED",reference_row_microbatch=32 if first_blocked else "full",variants=variants,
            all_passed=False,successful_variants_parity=False,atol=1e-5,rtol=1e-4,scope="Actual physical-output inference, not cached forecasts")

    def test_original_failed_gate_return_is_saved_before_any_downstream_assertion(self):
        result=self.returned_parity();calls=[]
        def original(*args,**kwargs):
            calls.append((args,kwargs));return result
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"RAW_PARITY_B1.json"
            value=self.h.capture_original(original,"route","physical","observed",target)
            self.assertEqual(value,result)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),result)
            self.assertEqual(calls,[(('route','physical','observed'),dict(lease=None))])
            with self.assertRaises(FileExistsError):self.h.capture_original(original,"r","p","o",target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),result)

    def test_original_four_variants_allow_false_gate_and_first_success_reference(self):
        for blocked in (False,True):
            value=self.returned_parity(blocked)
            self.h.validate_original(value)
            for mutation in ("order","tol","reference","count"):
                bad=copy.deepcopy(value)
                if mutation=="order":bad["variants"]=list(reversed(bad["variants"]))
                elif mutation=="tol":bad["atol"]=1e-3
                elif mutation=="reference":bad["reference_row_microbatch"]=4
                else:bad["variants"].pop()
                with self.subTest(blocked=blocked,mutation=mutation),self.assertRaises(ValueError):self.h.validate_original(bad)

    def test_exception_before_structured_return_is_not_a_false_parity_result(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"RAW_PARITY_B8.json"
            def fail(*args,**kwargs):raise RuntimeError("synthetic execution entered; returned no variants")
            with self.assertRaises(RuntimeError):self.h.capture_original(fail,"r","p","o",target)
            self.assertFalse(target.exists())
            result=self.returned_parity()
            self.assertEqual(self.h.capture_original(lambda *a,**k:result,"r","p","o",target),result)

    def test_F_counters_distinguish_B1_B8_and_deployed128_row_calls(self):
        for batch in (1,8):
            for row in ("full",32,8,4,128):
                expected=dict(F_logical_streams=batch*21,F_physical_streams=batch*21,
                    F_calls=1 if row=="full" else math.ceil(batch*21/row))
                self.assertEqual(self.h.counter_expectation(batch,row),expected)
        self.assertEqual(self.h.counter_expectation(8,128)["F_calls"],2)
        self.assertEqual(self.h.counter_expectation(8,"full")["F_calls"],1)

        h=self.h
        class Array:
            def __init__(self,value):self.value=value
            def copy(self):return Array(self.value)
            def tolist(self):return [[[self.value]*21]]
            def tobytes(self):return struct.pack("d",self.value)*21
        class Route:
            def __init__(self):self.spec=dict(row_microbatch=128);self.calls=[];self.deactivated=0
            def predict_physical(self,physical,observed):
                self.calls.append(self.spec["row_microbatch"])
                self.last_receipt=h.counter_expectation(1,self.spec["row_microbatch"])
                return Array(1.)
            def deactivate(self):self.deactivated+=1
        route=Route()
        rows=h.detail_pass(route,["history"],["observed"],[2.]*21,None,lambda:None,
            lambda *a:dict(error="unexpected"))
        self.assertEqual(route.calls,["full",32,8,4,128,128])
        self.assertEqual(route.spec["row_microbatch"],128)
        self.assertEqual(route.deactivated,6)
        self.assertEqual([row["scope"] for row in rows[-2:]],["extra_deployed128","extra_deployed128_repeat"])
        self.assertTrue(rows[-1]["comparison_to_deployed128_first"]["allclose"])
        self.assertTrue(all(row["F_counters_match"] for row in rows))

    def test_comparison_rejects_changed_shape_scale_and_preserves_nonfinite_failure(self):
        reference=[[[0.]*21]];prediction=[[[0.]*21]];std=[2.]*21
        for bad_reference,bad_prediction,bad_std in ((reference,[],std),(reference,[[[0.]*20]],std),
                (reference,prediction,[0.]+[2.]*20),
                (reference,prediction,[float("inf")]+[2.]*20)):
            with self.subTest(std=bad_std[0],shape=len(bad_prediction)),self.assertRaises(ValueError):
                self.h.comparison_summary(bad_reference,bad_prediction,bad_std)
        for nonfinite in (float("nan"),float("inf")):
            result=self.h.comparison_summary(reference,[[[nonfinite]+[0.]*20]],std)
            self.assertFalse(result["finite"])
            self.assertFalse(result["allclose"])
            self.assertIsNone(result["max_abs"])
            self.assertIsNone(result["max_A_normalized_abs"])
            self.assertEqual(result["failed_cells"],1)
            self.assertEqual(result["per_channel"][0]["nonfinite_cells"],1)

    def test_comparison_keeps_NumPy_second_operand_tolerance_and_channel_units(self):
        edge=1.00005e-5
        forward=self.h.comparison_summary([[[0.]*21]],[[[edge]+[0.]*20]],[2.]*21)
        reverse=self.h.comparison_summary([[[edge]+[0.]*20]],[[[0.]*21]],[2.]*21)
        self.assertTrue(forward["allclose"])
        self.assertFalse(reverse["allclose"])
        self.assertEqual((forward["failed_cells"],reverse["failed_cells"]),(0,1))
        reference=[[[0.,-2e-5]+[1.]*19]]
        predicted=[[[1e-3,2e-5]+[1.]*19]]
        result=self.h.comparison_summary(reference,predicted,[2.]*21)
        self.assertEqual(result["shape"],[1,1,21])
        self.assertEqual((result["total_cells"],result["failed_cells"]),(21,2))
        self.assertAlmostEqual(result["max_abs"],1e-3)
        self.assertAlmostEqual(result["max_A_normalized_abs"],5e-4)
        self.assertEqual(result["near_zero_failed_cells"],2)
        self.assertEqual(result["failed_sign_crossings"],1)
        self.assertEqual(result["exact_zero_reference_failed_cells"],1)
        self.assertEqual(result["per_channel"][0]["failed_cells"],1)
        self.assertEqual(result["per_channel"][1]["failed_sign_crossings"],1)

    def test_closed_failed_worker_is_diagnostic_input_but_live_or_mismatched_worker_rejected(self):
        h=self.h
        with tempfile.TemporaryDirectory() as td:
            public,cache,asset=(Path(td)/name for name in ("public","cache","asset"))
            for path in (public,cache,asset):path.mkdir()
            spec=cache/"original_spec.json";spec.write_text("{}",encoding="utf-8")
            command=[sys.executable,"-B",str(public/"measure_costs.py"),"--worker","--route-spec",str(spec),
                "--public-root",str(public),"--cache-root",str(cache),"--model",str(asset/"model"),"--device","cpu", "--batch","8",
                "--warmups","10","--timed-trials","20","--stage-trials","20","--output",str(public/h.NAME/"costs"/"original")]
            result=dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1",status="STOPPED_RESOURCE_OR_WORKER_FAILURE",helper_unchanged=True,
                remaining_owned_processes=[],supervisor_failure_class=None,GPU_owner_or_token_created=False,CUDA_VISIBLE_DEVICES="-1",
                effective_spec=dict(processes=[dict(command=command)],max_each_RSS=3*1024**3,max_group_RSS=3*1024**3),
                processes=[dict(returncode=1,observed_tree_join_confirmed=True,remaining_owned_processes=[],process_tree_errors=[],
                    owned_processes=[dict(pid=101,create_time=1.)])],unobserved_reparented_descendants_absence_proven=False)
            checked=h.validate_closed(result,8,spec,public,cache,asset)
            self.assertEqual(checked["returncode"],1)
            self.assertFalse(checked["unobserved_reparented_descendants_absence_proven"])
            for mutation in ("live","unknown","error","device","batch","timed","duplicate","spec"):
                bad=copy.deepcopy(result)
                argv=bad["effective_spec"]["processes"][0]["command"]
                if mutation=="live":bad["remaining_owned_processes"]=[dict(pid=101,create_time=1.)]
                elif mutation=="unknown":bad["processes"][0]["returncode"]=None
                elif mutation=="error":bad["processes"][0]["process_tree_errors"]=[dict(error="unknown")]
                elif mutation=="duplicate":argv.extend(["--batch","8"])
                else:
                    flag,value=dict(device=("--device","cuda"),batch=("--batch","1"),timed=("--timed-trials","19"),spec=("--route-spec",str(cache/"other.json")))[mutation]
                    argv[argv.index(flag)+1]=value
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.validate_closed(bad,8,spec,public,cache,asset)

    def test_bare_default_imports_no_scientific_or_process_modules_and_writes_none(self):
        before={name:name in sys.modules for name in ("numpy","torch","psutil")}
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["files_written"],0)
        self.assertEqual(result["numpy_imported"],before["numpy"])
        self.assertEqual(result["torch_imported"],before["torch"])
        self.assertEqual(result["psutil_imported"],before["psutil"])
        self.assertEqual(before,{name:name in sys.modules for name in before})


if __name__=="__main__":unittest.main(verbosity=2)
