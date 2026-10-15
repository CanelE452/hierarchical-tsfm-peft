"""Synthetic CPU q pipelines only; no actual data, F, model or GPU."""
from pathlib import Path
import ast
import importlib.util
import itertools
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE=Path(__file__).with_name("whole_fixed_l7_latent_calibration_cost_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("fixed q cost helper missing")
    spec=importlib.util.spec_from_file_location("fixed_q_cost_fixture",SOURCE); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class WholeCostTests(unittest.TestCase):
    def setUp(self): self.w=module()
    def test_ten_routes_five_partial_rotations_and_matching_Bolt(self):
        self.assertEqual(len(self.w.ROUTES),10); orders=self.w.balanced_schedule(); self.assertEqual(len(orders),5)
        for route in self.w.ROUTES:
            self.assertEqual(sum(order.count(route) for order in orders),5)
            self.assertEqual(len({order.index(route) for order in orders}),5)
            if "_SCALE_" in route: self.assertEqual(self.w.MATCHED_BOLT[route],route.split("_SCALE_")[0]+"_BOLT")
    def test_immutable_whole_pipeline_and_preprocess_functions_reused(self):
        import whole_f_input_gradient_cost_v1 as old
        self.assertIs(self.w.preprocess,old.preprocess); self.assertIs(self.w.whole_predict,old.whole_predict); self.assertIs(self.w.parity,old.parity)
    def test_actual_timed_callback_and_final_sync_AST_match_frozen_parent(self):
        import whole_f_input_gradient_cost_v1 as old
        def timed_body(path):
            tree=ast.parse(Path(path).read_text(encoding="utf-8")); found=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Try) and any(isinstance(n,ast.Assign) and "perf_end_ns" in ast.unparse(n.targets) for n in node.body): found.append(node.body)
            self.assertEqual(len(found),1); return [ast.dump(n,include_attributes=False) for n in found[0]]
        self.assertEqual(timed_body(SOURCE),timed_body(old.__file__))
    def fixture(self):
        rng=np.random.default_rng(4); physical=rng.normal(size=(8,512,2)).astype(np.float32); observed=np.ones_like(physical,bool)
        observed[:,:3,0]=False; physical[:,:3,0]=np.nan; mean=np.array([2.,3.],np.float32); std=np.array([2.,4.],np.float32); origins=np.arange(8,dtype=np.int64)
        def callback(offset): return lambda x,mask:np.ascontiguousarray(np.broadcast_to(x[:,-1:,]+np.float32(offset),(len(x),48,2)))
        callbacks={name:callback(i*.001) for i,name in enumerate(self.w.ROUTES)}; normalized=self.w.preprocess(physical,observed,mean,std)
        references={k:f(normalized,observed) for k,f in callbacks.items()}; metadata={k:{"synthetic":True} for k in self.w.ROUTES}
        return physical,observed,mean,std,origins,callbacks,references,metadata
    def test_all20_cells100_samples_and_both_denominators(self):
        values=self.fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as temp:
            ticks=itertools.count(0,1000000)
            with patch.object(self.w.time,"perf_counter_ns",side_effect=lambda:next(ticks)):
                result=self.w.measure_cost("synthetic_robin",*values,Path(temp)/"cost",lambda:None,lambda:None,lambda:{"scope":"shared fixture"})
            self.assertEqual(result["status"],"COMPLETE"); self.assertEqual((result["completed_cost_cells"],result["recorded_timer_samples"]),(20,100))
            for row in result["rows"]:
                self.assertEqual(row["ratio_of_RAW_medians"],1.); self.assertEqual(row["paired_RAW_median_ratio"],1.)
                self.assertEqual(row["ratio_of_matched_Bolt_medians"],1.); self.assertEqual(row["paired_matched_Bolt_median_ratio"],1.)
                self.assertEqual(len(row["seconds"]),5); self.assertIn("not",row["balanced_scope"])
    def test_missing_route_is_preserved_as_partial_not_deleted(self):
        values=list(self.fixture()); missing=self.w.ROUTES[-1]; values[5].pop(missing); values[6].pop(missing)
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as temp:
            result=self.w.measure_cost("synthetic",*values,Path(temp)/"cost",lambda:None,lambda:None,lambda:{},unavailable_routes={missing:"fixture unavailable"})
            self.assertEqual(result["status"],"PARTIAL_COMPLETED"); self.assertEqual(len(result["rows"]),20)
            self.assertEqual((result["completed_cost_cells"],result["recorded_timer_samples"]),(18,90))
    def test_bad_physical_reference_is_not_counted_as_success(self):
        values=list(self.fixture()); route=self.w.ROUTES[1]; values[6][route]=values[6][route]+1.
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as temp:
            result=self.w.measure_cost("synthetic",*values,Path(temp)/"cost",lambda:None,lambda:None,lambda:{})
            self.assertEqual(result["status"],"PARTIAL_COMPLETED"); self.assertTrue(result["failed_paths"])


if __name__=="__main__": unittest.main()
