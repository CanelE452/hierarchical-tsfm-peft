"""Tiny fixtures only: independent dual/lstsq, mask contracts and saved CPU readout."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import audit_new_direction_direct_ridge_v1 as audit
import masked_direct_ridge_v1 as fixture_solver
import run_new_direction_direct_ridge_v1 as fixture_producer

audit.initialize_numpy()


def fixture():
    rng=np.random.default_rng(361);x=rng.normal(size=(11,17)).astype(np.float32);x[:,2]=7.;x[:,5]=x[:,4]
    y=rng.normal(size=(11,6));mask=np.ones(y.shape,bool);mask[1:4,2:4]=False;mask[7:,4:]=False;y[~mask]=np.nan
    return x,y,mask


def raw_fixture():
    rng=np.random.default_rng(926);history=rng.normal(size=(7,512,2)).astype(np.float32);target=rng.normal(size=(7,48,2)).astype(np.float32);observed=np.ones(target.shape,bool);observed[2,1,0]=False;observed[3,4,1]=False;target[~observed]=np.nan
    return {"history":history,"raw_last":history[:,-1].copy(),"target":target,"observed":observed,"std":np.array([2.,.3],np.float32)}


class RidgeIndependentAuditTests(unittest.TestCase):
    def test_selected_positive_dual_and_zero_lstsq_both_normalizations_masked_rank_deficient(self):
        x,y,mask=fixture()
        for normalization in ("GLOBAL_FIT","PER_RESPONSE_MASK"):
            states,metadata=fixture_solver.fit_path(x,y,mask,normalization=normalization);summary,info=fixture_producer.fit_info(metadata)
            for index in (0,4):
                state={k:v for k,v in states[index].items() if k!="alpha"}
                with self.subTest(normalization=normalization,alpha=states[index]["alpha"]):
                    result=audit.selected_fit(x,y,mask,states[index]["alpha"],normalization,summary,info,state)
                    self.assertEqual(len(result["mask_groups"]),3);self.assertTrue(result["independent_selected_fit_only"]);self.assertFalse(result["unselected_coefficients_independently_refitted"])
                    self.assertTrue(all(g["spectrum_independently_recomputed"]==(index==0) for g in result["mask_groups"]))
                    self.assertTrue(all(g["normal_equation_max_abs"]<1e-10 for g in result["mask_groups"]))
        # A changed masked-out value does not enter either independent solution.
        changed=y.copy();changed[~mask]=1e250
        audit.selected_fit(x,changed,mask,.1,normalization,summary,info,{k:v for k,v in states[4].items() if k!="alpha"})

    def test_positive_alpha_uses_dual_n_by_n_and_no_svd_replay(self):
        x,y,mask=fixture();states,meta=fixture_solver.fit_path(x,y,mask,normalization="GLOBAL_FIT");summary,info=fixture_producer.fit_info(meta);state={k:v for k,v in states[5].items() if k!="alpha"};shapes=[];original=np.linalg.solve
        def solve(a,b):shapes.append(a.shape);return original(a,b)
        with patch.object(audit.np.linalg,"solve",side_effect=solve),patch.object(audit.np.linalg,"svd",side_effect=AssertionError("positive audit must not decompose primal")):
            audit.selected_fit(x,y,mask,1.,"GLOBAL_FIT",summary,info,state)
        self.assertEqual(shapes,[(11,11),(8,8),(7,7)]);self.assertNotIn((17,17),shapes)

    def test_coefficient_cast_scaling_mask_and_state_corruptions_are_detected(self):
        x,y,mask=fixture();states,meta=fixture_solver.fit_path(x,y,mask,normalization="GLOBAL_FIT");summary,info=fixture_producer.fit_info(meta);state={k:v for k,v in states[5].items() if k!="alpha"}
        cases=(
            (lambda s,i,m:s["coefficient_FP64"].__setitem__((0,0),s["coefficient_FP64"][0,0]+.1),"fused_cast"),
            (lambda s,i,m:i["group_0_feature_scale_FP64"].__setitem__(0,i["group_0_feature_scale_FP64"][0]*2),"feature_scale"),
            (lambda s,i,m:i["group_1_observed_mask"].__setitem__(0,False),"true_observed"),
            (lambda s,i,m:m["groups"][1].update(observed_count=99),"OBSERVED_RESPONSE_GROUP"),
        )
        for change,label in cases:
            s,i,m=copy.deepcopy(state),copy.deepcopy(info),copy.deepcopy(summary);change(s,i,m)
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,label):audit.selected_fit(x,y,mask,1.,"GLOBAL_FIT",m,i,s)
        corrupted=copy.deepcopy(state);corrupted["coefficient_FP64"][0,0]+=.1;corrupted["coefficient_FP32"]=corrupted["coefficient_FP64"].astype(np.float32)
        with self.assertRaisesRegex(ValueError,"independent_selected_coefficient64"):audit.selected_fit(x,y,mask,1.,"GLOBAL_FIT",summary,info,corrupted)

    def test_features_current_horizon_complete_vs_history_response_masks_and_past_only(self):
        raw=raw_fixture();latent=raw["history"][:,:,:1].copy()
        for route,z in (("RAW_CURRENT_C",None),("RAW_HISTORY512",None),("PCA_K1_HISTORY512",latent)):
            with self.subTest(route=route):
                x,a,y,m=audit.feature_response(raw,route,z);px,pa=fixture_producer.feature_payload(raw,route,z);py,pm=fixture_producer.response_payload(raw,route)
                for value,wanted in ((x,px),(a,pa),(y,py),(m,pm)):audit.base.exact(value,wanted,"fixture_contract")
                self.assertTrue(np.isfinite(y).all())
                if route=="RAW_CURRENT_C":self.assertFalse(m.reshape(7,48,2)[2,1].any());self.assertTrue((a==0).all())
                else:self.assertTrue(m.reshape(7,48,2)[2,1,1]);self.assertTrue((x.reshape(7,512,-1)[:,-1]==0).all())
                changed=copy.deepcopy(raw);changed["target"][changed["observed"]]+=200
                x2,a2,_,_=audit.feature_response(changed,route,z);audit.base.exact(x,x2,"future_does_not_enter_features");audit.base.exact(a,a2,"future_does_not_enter_anchor")

    def test_saved_cpu_canonical32_scores_per_channel_masks_and_hash_are_checked(self):
        raw=raw_fixture();x,a,_,_=audit.feature_response(raw,"RAW_CURRENT_C");rng=np.random.default_rng(19);coefficient=rng.normal(size=(2,96));intercept=rng.normal(size=96);state={"coefficient_FP64":coefficient,"intercept_FP64":intercept,"coefficient_FP32":coefficient.astype(np.float32),"intercept_FP32":intercept.astype(np.float32)}
        prediction=audit.deployed_prediction(x,a,state);expected=fixture_producer.predict(x,{"alpha":0.,**state},a);audit.base.exact(prediction,expected,"canonical_CPU32")
        complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape);entry={"status":"COMPLETE","scores_all_observed":audit.prior.scores(prediction,raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":audit.prior.scores(prediction,raw["target"],complete,raw["std"])}
        self.assertEqual(audit.score_check(entry,prediction,raw),24);self.assertEqual(audit.coefficient_records(state),fixture_producer.coefficient_records(state))
        for field in ("mse","per_channel_mse","observed_counts"):
            broken=copy.deepcopy(entry)
            if field=="mse":broken["scores_all_observed"]["normalized"][field]+=.1
            else:broken["scores_all_observed"]["normalized"][field][0]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):audit.score_check(broken,prediction,raw)

    def test_alpha_selection_exact_tie_order_own_mask_score_and_boundary(self):
        candidates=[{"alpha":a,"status":"COMPLETE","TUNE_MSE":2.,"TUNE":{"scores_all_observed":{"normalized":{"mse":2.}}}} for a in audit.ALPHAS]
        self.assertEqual(audit.winner(candidates)["alpha"],0.)
        candidates[-1]["TUNE_MSE"]=1.;candidates[-1]["TUNE"]["scores_all_observed"]["normalized"]["mse"]=1.;self.assertEqual(audit.winner(candidates)["alpha"],10.)
        candidates[-1]["TUNE_MSE"]-=.1
        with self.assertRaisesRegex(ValueError,"OWN_ACTUAL_TUNE"):audit.winner(candidates)
        with self.assertRaisesRegex(ValueError,"SEVEN"):audit.winner(candidates[:-1])

    def test_cpu_receipt_verifies_record_and_authorization_without_gpu_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            def save(name,value):
                path=folder/name;path.write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8");return audit.base.file_record(path)
            authorization=save("authorization.json",{"status":"APPROVED_HUMAN_UNCAPPED_CONTINUATION","latest_human_override":"상한내수행하지 말고 쭉해줘"});manifest=save("manifest.json",{"authorization_sha256":authorization["sha256"],"runtime_sha256":audit.prior.file_sha(audit.HERE/"continuation_runtime_v1.py")})
            payload={"process_cpu_seconds":.2,"wall_seconds":.3,"status":"COMPLETE","gpu_process":False,"original_budget_records_unchanged":True,"elapsed_or_cumulative_budget_caps":None,"authorization":authorization,"continuation_manifest":manifest}
            receipt={**payload,**save("receipt.json",payload)};self.assertEqual(len(audit.receipt_check(receipt,lambda:None)),3)
            receipt["gpu_process"]=True
            with self.assertRaisesRegex(ValueError,"CPU_ONLY"):audit.receipt_check(receipt,lambda:None)

    def test_closed_result_gate_before_io_and_fresh_import_stdlib_only(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.base,"file_record",side_effect=AssertionError("forbidden IO")):
            with self.assertRaisesRegex(ValueError,"NOT_REGISTERED"):audit.metadata("missing",lambda:None)
        code="import sys;sys.path.insert(0,"+repr(str(audit.HERE))+");import audit_new_direction_direct_ridge_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('STDLIB_IMPORT_PASS')"
        completed=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True);self.assertIn("STDLIB_IMPORT_PASS",completed.stdout)


if __name__=="__main__":unittest.main()
