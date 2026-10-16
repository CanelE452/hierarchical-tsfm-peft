"""Tiny invented plot inputs; Matplotlib and actual results stay unread."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("plot_core_contrast.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("core_plot_prep",SOURCE)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
else:p=None


class Tests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(p,"Core plot source not implemented")

    def data(self):
        actual=[];csv=[]
        for i,(role,identifier,k) in enumerate(zip(p.ROLES,p.IDS,(6,6,11))):
            row=dict(role=role,run_id=identifier,K=k,accuracy_device="cpu",training_device="cpu",CPU_threads=2,observed_labels=20539,V_origins=43,F_used=True,accuracy_reused=i==1,search_max_epoch=3,selected_refit_epoch=1,search_updates=1868,refit_updates=754,search_exposures=29880,refit_exposures=12063,training_status="SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",initialization="identity_at_K_equals_C" if k==11 else "A_fitted_centered_PCA",MSE=.12345678901234567+i*.01,MAE=.2+i*.01,B1_median_seconds=.1+i*.01,B8_median_seconds=.8+i*.01)
            actual.append(row);csv.append({k:str(v) for k,v in row.items()})
        audit=dict(schema="INDEPENDENT_BEIJING_CORE_CONTRAST_v1",status="PASS",error=None,audit_source_sha256=p.AUDITOR_SHA,input_SHA_before_after_equal=True,whole_MASTER_complete=False,independent_generalization_claimed=False,rows=actual)
        report=dict(schema="BEIJING_CORE_CONTRAST_REPORT_BINDINGS_v1",status="FROM_ACTUAL_COMPLETE_CORE_AUDIT",input_SHA_before_after_equal=True,whole_MASTER_complete=False,builder_source_sha256=p.BUILDER_SHA,auditor_source_sha256=p.AUDITOR_SHA,new_accuracy_conditions=2,reused_accuracy_conditions=1,new_cost_conditions=4,reused_cost_conditions=2)
        return csv,audit,report

    def test_default_zero_IO_and_no_renderer_import(self):
        out=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("read")),contextlib.redirect_stdout(out):self.assertEqual(p.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"],0)
        self.assertNotIn("matplotlib",sys.modules)

    def test_exact_three_roles_precision_and_reuse(self):
        rows,audit,report=self.data();prepared=p.prepare_data(rows,audit,report)
        self.assertEqual(float(prepared["rows"][0]["MSE"]),audit["rows"][0]["MSE"])
        self.assertEqual(prepared["p90_confirmed"],{1:False,8:False})
        self.assertIsNone(prepared["pca_reference"])

    def test_partial_or_changed_value_or_wrong_cpu_scope_rejected(self):
        rows,audit,report=self.data()
        for changes in ({"MSE":".7"},{"CPU_threads":"4"},{"accuracy_reused":"True"},{"K":"11"}):
            changed=copy.deepcopy(rows);changed[0].update(changes)
            with self.assertRaises(ValueError):p.prepare_data(changed,audit,report)
        with self.assertRaises(ValueError):p.prepare_data(rows,dict(audit,status="NOT_COMPLETE"),report)
        with self.assertRaises(ValueError):p.prepare_data(rows[:-1],audit,report)

    def test_caption_preserves_comparison_limits(self):
        for token in ("PCA 초기화","identity 초기화","P/K","capacity","선택 refit","반복 개발 V","수렴","독립 일반화","forecast MSE"):
            self.assertIn(token,p.CAPTION)
        self.assertNotIn("reconstruction failure",p.CAPTION)


if __name__=="__main__":unittest.main()
