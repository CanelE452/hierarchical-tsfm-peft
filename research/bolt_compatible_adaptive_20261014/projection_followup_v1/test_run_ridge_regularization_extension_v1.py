"""Generated metadata and CPU arrays only; no research assets or actual extension fit."""
from pathlib import Path
import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import run_ridge_regularization_extension_v1 as runner


def prior_menu():
    candidates=[]
    for alpha in runner.parent.ALPHAS:
        candidates.append({"alpha":alpha,"status":"COMPLETE","TUNE_MSE":1./(1+alpha),"prediction_artifact":{"path":"generated_only_"+str(alpha),"sha256":"0"*64},"coefficient_value_sha256":"1"*64,"FIT":{"fixture":True},"TUNE":{"fixture":True}})
    return {"candidates":candidates}


def blank_result():
    return {"datasets":{data:{"routes":{}} for data in runner.parent.DATASETS},"failed_paths":[],"new_selected_DEV_prediction_evaluations":0,"reused_original_DEV_prediction_evaluations":0,"all_10_locks_before_DEV":False,"resolved_selection_locks":0}


class RidgeExtensionControllerTests(unittest.TestCase):
    def test_pending_audit_gate_is_closed_before_io_or_output_creation(self):
        with patch.object(runner,"ORIGINAL_AUDIT_SHA",None),patch.object(runner.old,"record",side_effect=AssertionError("no IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):runner.load_gates("missing","missing")

    def test_history_only_ten_paths_and_twelve_candidate_order(self):
        self.assertEqual(len(runner.CONDITIONS),10);self.assertTrue(all(route!="RAW_CURRENT_C" for _,route in runner.CONDITIONS));self.assertEqual(len(runner.MENU),12)
        self.assertEqual(runner.MENU[-5:],("ALPHA_100","ALPHA_1000","ALPHA_10000","INTERCEPT_ONLY","CURRENT_C_ANCHOR"));self.assertEqual(len(set(runner.MENU)),12)

    def test_original_seven_scores_and_forecasts_are_reused_without_mutation(self):
        old=prior_menu();before=copy.deepcopy(old);menu=runner.original_menu(old);self.assertEqual(old,before);self.assertEqual(len(menu),7)
        for index,candidate in enumerate(menu):
            self.assertEqual(candidate["TUNE_MSE"],old["candidates"][index]["TUNE_MSE"]);self.assertEqual(candidate["prediction_artifact"],old["candidates"][index]["prediction_artifact"]);self.assertFalse(candidate["new_FIT_TUNE_inference"]);self.assertFalse(candidate["new_in_this_extension"])
        self.assertEqual(runner.select_candidate(menu)["alpha"],10.)

    def test_exact_ties_use_full_declared_order_and_candidates_fail_independently(self):
        menu=[{"candidate_id":identifier,"status":"COMPLETE","TUNE_MSE":.5} for identifier in reversed(runner.MENU)]
        self.assertEqual(runner.select_candidate(menu)["candidate_id"],runner.MENU[0]);menu[-1]["status"]="FAILED_CANDIDATE_PRESERVED";self.assertEqual(runner.select_candidate(menu)["candidate_id"],runner.MENU[1]);menu[0]["TUNE_MSE"]=.1;self.assertEqual(runner.select_candidate(menu)["candidate_id"],"CURRENT_C_ANCHOR")
        with self.assertRaisesRegex(ValueError,"UNIQUE_REGISTERED"):runner.select_candidate(menu+[menu[0]])

    def test_intercept_and_anchor_controls_use_only_true_observed_fit_innovation(self):
        x=np.zeros((5,3),np.float32);y=np.array([[2.,10.],[3.,99.],[4.,12.],[5.,99.],[6.,14.]],np.float64);mask=np.ones_like(y,bool);mask[[1,3],1]=False;y[~mask]=np.nan
        mean,anchor=runner.zero_slope_states(x,y,mask);np.testing.assert_array_equal(mean["intercept_FP64"],[4.,12.]);np.testing.assert_array_equal(anchor["intercept_FP64"],0.)
        for state in (mean,anchor):np.testing.assert_array_equal(state["coefficient_FP64"],0.);self.assertEqual(state["coefficient_FP32"].dtype,np.float32)
        changed=x+1000
        for a,b in zip((mean,anchor),runner.zero_slope_states(changed,y,mask)):
            for key in runner.COEFFICIENT_KEYS:np.testing.assert_array_equal(a[key],b[key])
        mask[:,0]=False
        with self.assertRaisesRegex(ValueError,"TWO_TRUE"):runner.zero_slope_states(x,y,mask)

    def test_anchor_forward_exact_and_intercept_response_mean_is_not_lag_flat(self):
        x=np.ones((3,2),np.float32);y=np.tile(np.array([1.,-2.,3.,4.,-1.,2.],np.float64),(3,1));mask=np.ones_like(y,bool);mean,zero=runner.zero_slope_states(x,y,mask)
        # The canonical production helper has H48; generate the corresponding tiny C1 case.
        response=np.tile(np.arange(48,dtype=np.float64),(3,1));mean,zero=runner.zero_slope_states(x,response,np.ones_like(response,bool));current=np.array([[2.],[3.],[4.]],np.float32)
        p=runner.parent.predict(x,zero,current);np.testing.assert_array_equal(p,np.broadcast_to(current[:,None],(3,48,1)))
        expected=current[:,None]+response.astype(np.float32).reshape(3,48,1);np.testing.assert_array_equal(runner.parent.predict(x,mean,current),expected)

    def test_all_ten_success_or_failure_locks_precede_any_dev_and_other_paths_continue(self):
        result=blank_result();events=[];bad=runner.CONDITIONS[3]
        def run(data,route):
            events.append(("FIT_TUNE",data,route))
            if (data,route)==bad:raise ValueError("synthetic one route failure")
            row={"status":"COMPLETE","selected_original_candidate":route=="RAW_HISTORY512"};result["datasets"][data]["routes"][route]=row;return row
        def seal(data,route,row):events.append(("LOCK",data,route,row["status"]));return {"generated_lock":data+route}
        def registry(locks):events.append(("REGISTRY",len(locks)));return {"generated_registry":True}
        resolved=runner.resolve_paths(result,run,seal,registry);self.assertEqual(len(resolved),9);self.assertEqual(len(result["failed_paths"]),1);self.assertEqual(sum(e[0]=="LOCK" for e in events),10)
        def infer(data,route,row):
            self.assertTrue(result["all_10_locks_before_DEV"]);self.assertEqual(events[20],("REGISTRY",10));events.append(("DEV",data,route));return {"status":"COMPLETE"}
        runner.infer_development(result,resolved,infer);self.assertEqual(result["new_selected_DEV_prediction_evaluations"]+result["reused_original_DEV_prediction_evaluations"],9)
        self.assertEqual(result["reused_original_DEV_prediction_evaluations"],2);self.assertNotIn(bad,resolved)

    def test_dev_does_not_open_with_only_nine_locks(self):
        result=blank_result();result.update(all_10_locks_before_DEV=True,resolved_selection_locks=9)
        with self.assertRaisesRegex(ValueError,"ALL_TEN_EXTENSION_LOCKS"):runner.infer_development(result,{},lambda *args:self.fail("no DEV"))

    def test_source_never_patches_original_grid_or_imports_F_GPU_and_baseline_cost_is_not_reassigned(self):
        text=Path(runner.__file__).read_text(encoding="utf-8");tree=ast.parse(text);calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn("parent.ALPHAS=",text.replace(" ",""));self.assertFalse(any(c.endswith(("FrozenBolt","train_head","backward","optimizer.step")) for c in calls));self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=="torch" for a in n.names) for n in ast.walk(tree)))
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute");self.assertIn("load_gates",ast.unparse(execute.body[0]));self.assertIn('"original_v1_whole_cost_reassigned_to_extension":False',text)
        self.assertIn("original_DEV_prediction_reused",text);self.assertIn("RESELECTED_OLD_MENU_MUST_BE_ORIGINAL_WINNER",text)


if __name__=="__main__":unittest.main()
