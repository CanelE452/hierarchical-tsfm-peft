"""Small CPU synthetic timer/causal-fill fixtures; never actual F or model assets."""
from pathlib import Path
import ast
import inspect
import json
import statistics
import tempfile
import unittest

import numpy as np

import whole_raw_subset_cost_v1 as cost


def inputs():
    rng = np.random.default_rng(92601)
    physical = rng.normal(size=(8,512,3)).astype(np.float32)
    mask = np.ones_like(physical, dtype=bool); mask[:, :2, 0] = False; mask[:, 80:84, 1] = False
    physical[~mask] = np.nan
    mean, std = np.array([1.,2.,3.],np.float32), np.array([2.,3.,4.],np.float32)
    normalized = cost.preprocess(physical, mask, mean, std)
    reference = np.repeat(normalized[:, -1:], 48, axis=1)
    callbacks = {route: lambda h,m: np.repeat(h[:, -1:],48,axis=1) for route in cost.ROUTES}
    return physical, mask, mean, std, np.arange(8,dtype=np.int64)+512, callbacks, {route: reference.copy() for route in cost.ROUTES}, {route:{"synthetic":True} for route in cost.ROUTES}


class WholeRawSubsetCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.folder=Path(self.temp.name)

    def run_cost(self, name="run", change=None):
        values = list(inputs())
        if change: change(values)
        events=[]
        result=cost.measure_cost("synthetic",*values,self.folder/name,lambda:events.append("guard"),
            lambda:events.append("sync"),lambda:{"shared_process":True,"sampled_RSS":123})
        return result,events

    def test_ten_routes_fully_balance_all_ten_positions(self):
        schedule=cost.balanced_schedule()
        self.assertEqual(len(schedule),10); self.assertEqual(schedule,cost.balanced_schedule(20261015))
        for order in schedule: self.assertEqual(set(order),set(cost.ROUTES))
        for route in cost.ROUTES: self.assertEqual(sorted(order.index(route) for order in schedule),list(range(10)))

    def test_two_synthetic_datasets_retain_forty_cells_and_four_hundred_outputs(self):
        first,_=self.run_cost("one"); second,_=self.run_cost("two")
        self.assertEqual(first["status"],"COMPLETE"); self.assertEqual(second["status"],"COMPLETE")
        self.assertEqual(first["completed_cost_cells"]+second["completed_cost_cells"],40)
        self.assertEqual(first["recorded_timer_samples"]+second["recorded_timer_samples"],400)
        for row in first["rows"]:
            self.assertEqual(row["repeats"],10)
            self.assertEqual(sorted(row["positions"]),list(range(10)))
            self.assertEqual(row["median_seconds"],statistics.median(row["seconds"]))
            self.assertEqual(row["paired_RAW_median_ratio"],statistics.median(row["paired_RAW_ratios"]))
            self.assertFalse(row["per_origin_decoder_fit"])
            for sample in row["samples"]:
                self.assertEqual(sample["seconds"],(sample["perf_end_ns"]-sample["perf_start_ns"])/1e9)
                self.assertGreater(sample["seconds"],0)
                self.assertTrue(all(value["pass"] for value in sample["parity"].values()))
                self.assertTrue(Path(sample["sample_artifact"]["path"]).exists())

    def test_reference_parity_failure_isolated_to_one_declared_route(self):
        bad=cost.ROUTES[1]
        def changed(values): values[6][bad] += np.float32(.1)
        result,_=self.run_cost(change=changed)
        self.assertEqual(result["status"],"PARTIAL_COMPLETED")
        self.assertEqual(result["routes"],list(cost.ROUTES)); self.assertEqual(len(result["rows"]),20)
        self.assertEqual(result["completed_cost_cells"],18); self.assertEqual(result["recorded_timer_samples"],180)
        self.assertEqual({row["route"] for row in result["failed_paths"]},{bad})
        self.assertTrue(all(not row["declared_route_deleted"] for row in result["rows"]))

    def test_timed_callback_failure_preserves_saved_samples_and_other_routes(self):
        bad=cost.ROUTES[-1]; calls={1:0,8:0}
        def change(values):
            def predict(h,m):
                calls[len(h)]+=1
                if calls[len(h)]==6: raise ValueError("synthetic branch failure after two timed outputs")
                return np.repeat(h[:, -1:],48,axis=1)
            values[5][bad]=predict
        result,_=self.run_cost(change=change)
        self.assertEqual(result["status"],"PARTIAL_COMPLETED")
        self.assertEqual(result["completed_cost_cells"],18)
        partial=[row for row in result["rows"] if row["route"]==bad]
        self.assertEqual([row["repeats"] for row in partial],[2,2])
        self.assertEqual(result["recorded_timer_samples"],184)
        self.assertEqual(len(list((self.folder/"run").rglob("*_FAILED.json"))),2)

    def test_causal_fill_uses_FIT_mean_only_before_first_observation(self):
        x=np.array([[[99.,2.],[1.,99.],[99.,4.]]],np.float32); m=np.array([[[False,True],[True,False],[False,True]]])
        result=cost.fill_vectorized(x,m,np.array([5.,6.],np.float32))
        np.testing.assert_array_equal(result,np.array([[[5.,2.],[1.,2.],[1.,4.]]],np.float32))
        changed=x.copy(); changed[:,2,1]=1000
        np.testing.assert_array_equal(cost.fill_vectorized(changed,m,np.array([5.,6.],np.float32))[:,:2],result[:,:2])

    def test_whole_output_is_physical_materialized_FP32_CPU_array(self):
        x,m,mean,std,*_=inputs()
        actual,physical=cost.whole_predict(lambda h,m:np.repeat(h[:,-1:],48,axis=1),x,m,mean,std)
        self.assertEqual(physical.dtype,np.float32); self.assertTrue(physical.flags.owndata)
        np.testing.assert_array_equal(physical,(actual*std+mean).astype(np.float32))
        self.assertFalse(np.shares_memory(physical,actual))

    def test_unchanged_legacy_fill_normalization_and_parity_AST(self):
        old=ast.parse((cost.HERE/"whole_matched_shrunk_native_cost_v1.py").read_text(encoding="utf-8"))
        new=ast.parse(Path(cost.__file__).read_text(encoding="utf-8"))
        for name in ("fill_vectorized","preprocess","parity"):
            left=next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
            right=next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name==name)
            self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False),name)

    def test_no_future_target_API_no_decoder_fitting_or_hash_inside_timer(self):
        self.assertFalse({"target","future","Y"}&set(inspect.signature(cost.measure_cost).parameters))
        source=inspect.getsource(cost.measure_cost)
        timed=source.split("unix_start_ns, perf_start_ns =",1)[1].split("seconds =",1)[0]
        self.assertIn("whole_predict",timed); self.assertIn("synchronize()",timed)
        self.assertNotIn("sha(",timed); self.assertNotIn("fit(",timed); self.assertNotIn("memory_snapshot",timed)

    def test_normalized_and_physical_parities_keep_registered_tolerances(self):
        value=np.ones((1,48,3),np.float32)
        self.assertTrue(cost.parity(value,value)["pass"])
        self.assertFalse(cost.parity(value+np.float32(1e-3),value)["pass"])
        self.assertFalse(cost.parity(value,np.ones((1,47,3),np.float32))["pass"])
        self.assertEqual((cost.NORM_ATOL,cost.NORM_RTOL,cost.PHYSICAL_ATOL,cost.PHYSICAL_RTOL),(1e-5,2e-6,2e-3,2e-5))

    def test_fresh_output_refuses_overwrite(self):
        self.run_cost()
        with self.assertRaises(FileExistsError): self.run_cost()


if __name__=="__main__": unittest.main()
