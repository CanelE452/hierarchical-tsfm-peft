"""Tiny generated arrays/menus only; no actual extension inputs or fit/audit run."""
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import json
import subprocess
import sys
import unittest

import numpy as np

import audit_ridge_regularization_extension_v1 as audit
import masked_direct_ridge_extension_v1 as solver
import run_ridge_regularization_extension_v1 as producer
from test_audit_new_direction_direct_ridge_v1 import fixture

audit.initialize_numpy()


def selected_fixture(alpha=100.):
    x,y,mask=fixture(); states,metadata=solver.fit_path(x,y,mask,normalization="GLOBAL_FIT"); summary,info=producer.parent.fit_info(metadata); state={k:v for k,v in states[list(solver.ALPHAS).index(alpha)].items() if k != "alpha"}
    return x,y,mask,alpha,summary,info,state


def menu_fixture():
    old={"candidates":[],"selected_alpha":10.,"selected_coefficients":{"path":"old_selected","sha256":"a"*64},"selected_coefficient_value_sha256":"b"*64}
    for index,alpha in enumerate(audit.OLD_ALPHAS):
        value=1.-index*.05; old["candidates"].append({"alpha":alpha,"status":"COMPLETE","TUNE_MSE":value,"FIT":{"synthetic":index},"TUNE":{"scores_all_observed":{"normalized":{"mse":value}}},"coefficient_value_sha256":"b"*64,"coefficient_array_records":{},"prediction_artifact":{"path":"old_prediction"+str(index),"sha256":"c"*64}})
    candidates=[]
    for index,item in enumerate(old["candidates"]): candidates.append(copy.deepcopy({**item,"candidate_id":audit.MENU[index],"kind":"RIDGE","new_in_this_extension":False,"original_candidate_index":index,"original_prediction_artifact":item["prediction_artifact"],"new_FIT_TUNE_inference":False}))
    for index,identifier in enumerate(audit.MENU[7:]):
        value=.4+index*.05; candidates.append({"candidate_id":identifier,"kind":"RIDGE" if index < 3 else identifier,"alpha":audit.NEW_ALPHAS[index] if index < 3 else None,"status":"COMPLETE","TUNE_MSE":value,"FIT":{"synthetic":7+index},"TUNE":{"scores_all_observed":{"normalized":{"mse":value}}},"coefficient_value_sha256":"d"*64,"coefficient_array_records":{},"prediction_artifact":{"path":"new_prediction"+str(index),"sha256":"e"*64},"new_in_this_extension":True,"new_FIT_TUNE_inference":True})
    row={"route":"RAW_HISTORY512","candidates":candidates}; select_row(row,candidates[7]); return row,old


def select_row(row,chosen):
    row.update(selected_candidate=chosen["candidate_id"],selected_kind=chosen["kind"],selected_alpha=chosen["alpha"],selected_TUNE_MSE=chosen["TUNE_MSE"],selected_original_candidate=not chosen["new_in_this_extension"],selected_FIT_TUNE_prediction_artifact=copy.deepcopy(chosen["prediction_artifact"]),selected_coefficient_value_sha256=chosen["coefficient_value_sha256"],selected_new_alpha_upper_boundary=chosen["alpha"] == audit.NEW_ALPHAS[-1],zero_slope_selected=chosen["kind"] != "RIDGE",history_features_needed_to_predict=chosen["kind"] == "RIDGE",encoder_needed_to_predict=chosen["kind"] == "RIDGE" and row["route"] != "RAW_HISTORY512",F_needed_to_predict=False,FIT=copy.deepcopy(chosen["FIT"]),TUNE=copy.deepcopy(chosen["TUNE"]))


def registry_fixture():
    result={"datasets":{data:{"routes":{}} for data in audit.DATASETS},"all_10_locks_before_DEV":True,"resolved_selection_locks":10}; registry={"status":"ALL_TEN_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"sealed_unix":20.,"conditions":[]}; locks={}
    for data,route in audit.CONDITIONS:
        name=data+route; binding={"path":name,"sha256":"a"*64}; row={"status":"COMPLETE","selected_candidate":"ALPHA_100","selected_kind":"RIDGE","selected_alpha":100.,"selected_TUNE_MSE":.1,"selected_coefficients":{"path":name+"coef","sha256":"b"*64},"selected_coefficient_value_sha256":"c"*64,"selected_original_candidate":False,"selection_lock":binding}
        result["datasets"][data]["routes"][route]=row; registry["conditions"].append({"dataset":data,"route":route,"status":"COMPLETE","selection_lock":binding}); locks[name]={k:v for k,v in row.items() if k != "selection_lock"}|{"dataset":data,"route":route,"DEV_opened_before_lock":False,"sealed_unix":10.}
    return result,registry,locks


class ExtensionAuditTests(unittest.TestCase):
    def test_selected_new_positive_dual_matches_thin_SVD_all_three_alphas_without_replaying_SVD(self):
        for alpha in audit.NEW_ALPHAS:
            values=selected_fixture(alpha)
            with patch.object(audit.np.linalg,"svd",side_effect=AssertionError("audit must use independent dual")):
                result=audit.selected_positive_fit(*values)
            with self.subTest(alpha=alpha):
                self.assertEqual(len(result["mask_groups"]),3); self.assertTrue(result["selected_new_fit_only_independently_solved"]); self.assertFalse(result["unselected_coefficients_independently_refitted"])
                self.assertTrue(all(g["normal_equation_max_abs"] < 1e-10 for g in result["mask_groups"])); json.dumps(result,allow_nan=False)
        self.assertEqual(audit.original_audit.ALPHAS,(0.,1e-4,.001,.01,.1,1.,10.))

    def test_masked_placeholder_invariance_and_selected_coefficient_scale_cast_mask_mutations(self):
        values=selected_fixture(); changed=copy.deepcopy(values); changed[1][~changed[2]]=1e200; audit.selected_positive_fit(*changed)
        for kind in ("coefficient","cast","scale","mask","rank","grid"):
            v=copy.deepcopy(values)
            if kind == "coefficient": v[6]["coefficient_FP64"][0,0]+=.1; v[6]["coefficient_FP32"]=v[6]["coefficient_FP64"].astype(np.float32)
            elif kind == "cast": v[6]["intercept_FP32"][0]+=.1
            elif kind == "scale": v[5]["global_feature_scale_FP64"][0]*=2
            elif kind == "mask": v[5]["group_0_observed_mask"][0]=False
            elif kind == "rank": v[4]["groups"][0]["minimum_norm_retained_rank"]+=1
            else: v[4]["alpha_grid"]=list(audit.OLD_ALPHAS)
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.selected_positive_fit(*v)

    def test_zero_control_FIT_means_streamed_hashes_and_canonical_FP32_predictions(self):
        rng=np.random.default_rng(45); x=rng.normal(size=(6,3)).astype(np.float32); y=rng.normal(size=(6,96)); mask=np.ones_like(y,bool); mask[1:3,4:8]=False; y[~mask]=np.nan
        expected=producer.zero_slope_states(x,y,mask); controls=audit.zero_controls(x,y,mask); anchor=rng.normal(size=(6,2)).astype(np.float32); anchor[0,0]=np.float32(-0.)
        for kind,state in zip(("INTERCEPT_ONLY","CURRENT_C_ANCHOR"),expected):
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(controls[kind],state["intercept_FP64"]); self.assertEqual(audit.zero_records(3,controls[kind]),producer.parent.coefficient_records(state)); audit.base.exact(audit.zero_prediction(anchor,controls[kind]),producer.parent.predict(x,state,anchor),"same_actual_FP32_zero_control")
        changed=y.copy(); changed[~mask]=np.inf
        for kind in controls: np.testing.assert_array_equal(controls[kind],audit.zero_controls(x+100,changed,mask)[kind])
        mask[:,0]=False
        with self.assertRaises(ValueError): audit.zero_controls(x,y,mask)

    def test_old_seven_candidates_are_immutable_and_new_menu_selection_uses_exact_registered_order(self):
        row,old=menu_fixture(); self.assertEqual(audit.check_menu(row,old)["candidate_id"],"ALPHA_100")
        row["candidates"][8]["TUNE_MSE"] = .4; row["candidates"][8]["TUNE"]["scores_all_observed"]["normalized"]["mse"] = .4; self.assertEqual(audit.check_menu(row,old)["candidate_id"],"ALPHA_100")
        row["candidates"][8]["TUNE_MSE"] -= 1e-16; row["candidates"][8]["TUNE"]["scores_all_observed"]["normalized"]["mse"] = row["candidates"][8]["TUNE_MSE"]
        with self.assertRaises(ValueError): audit.check_menu(row,old)
        for kind in ("oldscore","oldartifact","oldinference","order","kind","fitreuse","featureclaim"):
            r,o=menu_fixture()
            if kind == "oldscore": r["candidates"][0]["TUNE_MSE"]+=.1
            elif kind == "oldartifact": r["candidates"][0]["prediction_artifact"]["sha256"]="0"*64
            elif kind == "oldinference": r["candidates"][0]["new_FIT_TUNE_inference"]=True
            elif kind == "order": r["candidates"][-1],r["candidates"][-2]=r["candidates"][-2],r["candidates"][-1]
            elif kind == "kind": r["candidates"][7]["alpha"]=10.
            elif kind == "fitreuse": r["FIT"]={"changed":True}
            else: r["history_features_needed_to_predict"]=False
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_menu(r,o)

    def test_zero_slope_selection_has_no_history_encoder_or_F_requirement(self):
        row,old=menu_fixture(); row["route"]="PCA_K4_HISTORY512"; chosen=row["candidates"][-1]; chosen["TUNE_MSE"]=.1; chosen["TUNE"]["scores_all_observed"]["normalized"]["mse"]=.1; select_row(row,chosen)
        self.assertEqual(audit.check_menu(row,old)["candidate_id"],"CURRENT_C_ANCHOR"); self.assertFalse(row["encoder_needed_to_predict"]); self.assertFalse(row["history_features_needed_to_predict"]); self.assertFalse(row["F_needed_to_predict"])

    def test_reselected_original_winner_and_DEV_binding_are_reused_without_new_inference(self):
        row,old=menu_fixture()
        for candidate in row["candidates"][7:]: candidate["TUNE_MSE"]=2.; candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"]=2.
        chosen=row["candidates"][6]; select_row(row,chosen); row["selected_coefficients"]=copy.deepcopy(old["selected_coefficients"])
        self.assertEqual(audit.check_menu(row,old)["alpha"],10.)
        original={"artifact":{"path":"original_DEV","sha256":"f"*64},"status":"COMPLETE","scores":{"synthetic":.3}}
        entry={**copy.deepcopy(original),"original_DEV_prediction_reused":True,"new_DEV_inference":False,"original_DEV_source":copy.deepcopy(original["artifact"])}; audit.check_reused_DEV(entry,original)
        entry["new_DEV_inference"]=True
        with self.assertRaises(ValueError): audit.check_reused_DEV(entry,original)
        old["selected_alpha"]=0.
        with self.assertRaisesRegex(ValueError,"ORIGINAL_WINNER"): audit.check_menu(row,old)

    def test_ten_complete_selection_locks_and_failure_tampering(self):
        r,g,l=registry_fixture(); self.assertEqual(len(audit.check_registry(r,g,lambda item:l[item["path"]])),10)
        for kind in ("duplicate","earlyDEV","late","alpha","failed","count"):
            r,g,l=copy.deepcopy(registry_fixture())
            if kind == "duplicate": g["conditions"][-1]=g["conditions"][0]
            elif kind == "earlyDEV": g["DEV_opened_before_seal"]=True
            elif kind == "late": next(iter(l.values()))["sealed_unix"]=21.
            elif kind == "alpha": next(iter(l.values()))["selected_alpha"]=10.
            elif kind == "failed": g["conditions"][0]["status"]="FAILED"
            else: r["resolved_selection_locks"]=9
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_registry(r,g,lambda item:l[item["path"]])

    def test_closed_actual_result_gate_and_pure_import_do_not_read_scientific_inputs(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.base,"file_record",side_effect=AssertionError("IO forbidden")):
            with self.assertRaisesRegex(ValueError,"NOT_REGISTERED"): audit.metadata("missing",lambda:None)
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8")); imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(name in ("run_ridge_regularization_extension_v1","masked_direct_ridge_extension_v1","torch") for name in imports))
        command="import sys;sys.path.insert(0,"+repr(str(audit.HERE))+");import audit_ridge_regularization_extension_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('PURE_IMPORT_PASS')"
        completed=subprocess.run([sys.executable,"-c",command],capture_output=True,text=True,check=True); self.assertIn("PURE_IMPORT_PASS",completed.stdout)


if __name__ == "__main__": unittest.main()
