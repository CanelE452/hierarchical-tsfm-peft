"""SOURCE-only synthetic CPU and metadata gates; no campaign arrays/F/native-head weights."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

import run_fixed_l7_latent_calibration_v1 as run
import fixed_l7_latent_calibration_v1 as pure
from linear_e_adaptation_v1 import LinearLagAdapter


def fixture():
    rng=np.random.default_rng(9); past=rng.normal(size=(2,40,1)).astype(np.float32); future=rng.normal(size=(2,48,1)).astype(np.float32); current=rng.normal(size=(2,3)).astype(np.float32)
    model=LinearLagAdapter(np.zeros(3,np.float32),np.array([[1.],[.2],[-.1]],np.float32),rng.normal(size=(7,3)).astype(np.float32),run.LAGS,learn_E=False).eval().requires_grad_(False)
    with torch.no_grad(): prediction=model.decode(torch.from_numpy(current),torch.from_numpy(past),torch.from_numpy(future)).numpy().copy()
    return model,{"raw_last":current},{"latent_history":past,"latent_future":future,"prediction":prediction}


class CalibrationControllerTests(unittest.TestCase):
    def test_closed_gate_before_any_file_or_scientific_import(self):
        with patch.object(run,"MODULE_SHA",None),patch.object(run.old,"record",side_effect=AssertionError("closed file")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            run.load_gates(SimpleNamespace())

    def test_metadata_gate_positive_and_changed_report_rejected_without_arrays(self):
        with tempfile.TemporaryDirectory(dir=run.HERE) as folder:
            root=Path(folder)
            def put(name,value):
                path=root/name; path.write_text(json.dumps(value),encoding="utf-8"); return run.old.record(path)
            report=root/"REPORT.md"; report.write_text("Synthetic accepted report",encoding="utf-8"); report_ref=run.old.record(report)
            reader=root/"reader.py"; reader.write_text("# Synthetic metadata reader\n",encoding="utf-8"); reader_ref=run.old.record(reader)
            bindings={key:put(key+".json",{"status":"PASS" if "audit" in key else "COMPLETE"}) for key in ("train","audit","cost","cost_audit")}
            native={key:put(key+".json",{}) for key in ("selection_registry","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit")}
            manifest=put("MANIFEST.json",{"status":"COMPLETE","protected_TEST_accessed":False,"source":reader_ref,"files":[]})
            audit=put("REPORT_AUDIT.json",{"status":"PASS","files_checked":22,"PNG_count":4,"SVG_count":4,"local_links_checked":41,"protected_TEST_accessed":False,"report":report_ref,"manifest":manifest,"inputs":bindings,"audit_source":reader_ref})
            args=SimpleNamespace(report=str(report),report_manifest=manifest["path"],report_audit=audit["path"],native_result=bindings["train"]["path"],native_audit=bindings["audit"]["path"],native_cost=bindings["cost"]["path"],native_cost_audit=bindings["cost_audit"]["path"])
            accepted=SimpleNamespace(load_verified=lambda paths:({"train":native,"science":{"status":"COMPLETE"}},bindings))
            with patch.multiple(run,REPORT_SHA=report_ref["sha256"],MANIFEST_SHA=manifest["sha256"],REPORT_AUDIT_SHA=audit["sha256"],READER_SHA=reader_ref["sha256"],MODULE_SHA=run.old.record(pure.__file__)["sha256"]),patch.dict(sys.modules,{"make_matched_l7_native_report_v2":accepted}):
                actual,science,refs=run.load_gates(args); self.assertEqual(actual,native); self.assertEqual(science["status"],"COMPLETE"); self.assertEqual(refs["matched_l7_native_diagnostic"],bindings["train"])
                report.write_text("changed",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"EXACT_ACCEPTED_NATIVE_REPORT"): run.load_gates(args)

    def test_six_condition_failure_isolation_and_registry_before_any_DEV(self):
        events=[]; result={"states":{s:{"families":{}} for s in run.STATES},"failed_paths":[],"all_6_locks_before_DEV":False,"resolved_selection_locks":0,"selected_DEV_evaluations_completed":0}
        def fit(state,family):
            events.append(("FIT",state,family))
            if (state,family)==run.CONDITIONS[0]: raise ValueError("one family failed")
            return {"status":"COMPLETE","selected_candidate":"IDENTITY"}
        def seal(state,family,row): events.append(("LOCK",state,family)); return {"path":state+family}
        def registry(rows): events.append(("REGISTRY",len(rows))); return {"path":"registry"}
        with self.assertRaisesRegex(ValueError,"SIX_LOCKS"): run.infer_development(result,{},lambda *args: (_ for _ in ()).throw(AssertionError("DEV before locks")))
        resolved=run.resolve_families(result,fit,seal,registry); self.assertEqual(events[-1],("REGISTRY",6)); self.assertEqual(len(resolved),5)
        def infer(state,family,row):
            self.assertTrue(result["all_6_locks_before_DEV"]); self.assertEqual(sum(e[0]=="LOCK" for e in events),6); events.append(("DEV",state,family))
            return {"status":"FAILED_DEPLOYMENT_PARITY_PRESERVED" if (state,family)==run.CONDITIONS[1] else "COMPLETE","artifact":{"path":"preserved"}}
        run.infer_development(result,resolved,infer); self.assertEqual(result["selected_DEV_evaluations_completed"],4); self.assertEqual(len(result["failed_paths"]),2)
        self.assertEqual(result["states"][run.CONDITIONS[1][0]]["families"][run.CONDITIONS[1][1]]["DEV"]["artifact"]["path"],"preserved")

    def test_identity_original_objects_and_bytes_without_redecoding_candidate(self):
        model,raw,cached=fixture(); candidate=pure.identity_candidate(1,"SCALE_ONLY"); before=run.value_hash(run.coefficient_arrays(candidate)); events=[]
        prediction,future,checks=run.predict_candidate(model,raw,cached,candidate,pure,events.append,batch_size=1)
        self.assertTrue(run.old.byte_equal(prediction,cached["prediction"])); self.assertTrue(run.old.byte_equal(future,cached["latent_future"])); self.assertEqual(checks["identity_object_checks"],2); self.assertEqual(before,run.value_hash(run.coefficient_arrays(candidate))); self.assertEqual(len(events),2)

    def test_actual_FP32_independent_q_and_lag_C_gate_is_separate_from_surrogate(self):
        model,raw,cached=fixture()
        for family in pure.FAMILIES:
            candidate=pure.identity_candidate(1,family); candidate.update(kind="RIDGE",id="RIDGE_1",**{"lambda":1.}); candidate["a_FP64"][:]=-.2; candidate["a_FP32"][:]=-.2
            if family=="SCALE_BIAS": candidate["b_FP64"][:]=.1; candidate["b_FP32"][:]=.1
            prediction,future,_=run.predict_candidate(model,raw,cached,candidate,pure,lambda _:None); checks,arrays=run.independent_deployment(prediction,future,cached,raw,model.D.numpy(),candidate)
            with self.subTest(family=family): self.assertTrue(all(v["pass"] for v in checks.values())); self.assertEqual(arrays["independent_NumPy_future_FP32"].dtype,np.float32)
            bad=prediction.copy(); bad.flat[0]+=.1; failed,_=run.independent_deployment(bad,future,cached,raw,model.D.numpy(),candidate); self.assertFalse(failed["C_prediction"]["pass"])

    def test_failed_nonfinite_coefficient_artifacts_keep_FP64_and_skip_missing_FP32(self):
        candidate=pure.identity_candidate(1,"SCALE_ONLY"); candidate.update(a_FP32=None,b_FP32=None,coefficients_FP32=None,status="FAILED_NONFINITE_COEFFICIENTS_PRESERVED"); candidate["a_FP64"][:]=1e300
        arrays=run.coefficient_arrays(candidate); self.assertEqual(set(arrays),{"a_FP64","b_FP64","coefficients_FP64"}); self.assertEqual(arrays["a_FP64"][0],1e300)
        before=run.value_hash(arrays); arrays["a_FP64"][0]=1e299; self.assertNotEqual(before,run.value_hash(arrays))


if __name__=="__main__": unittest.main()
