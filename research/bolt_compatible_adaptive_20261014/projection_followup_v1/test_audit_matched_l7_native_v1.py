"""Generated saved native states/logs and geometry only; no campaign assets or F."""
from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np

import audit_matched_l7_native_v1 as audit
from test_run_learned_output_space_diagnostic_v1 import fixture

audit.initialize_numpy()


def native_fixture():
    state,raw,saved=fixture(audit.LAGS); rng=np.random.default_rng(82)
    state.update({"head.weight":rng.normal(scale=.001,size=(1,48,512)).astype(np.float32),"head.bias":rng.normal(scale=.001,size=(1,48)).astype(np.float32)})
    saved["latent_future"]=audit.head_future(state,saved["latent_history"]); saved["lag_features"]=audit.base.delta_features(saved["latent_history"],saved["latent_future"],audit.LAGS); saved["prediction"]=raw["raw_last"][:,None]+saved["lag_features"]@state["D"]
    return state,raw,saved


def logged_fixture(folder):
    state,raw,saved=native_fixture(); path=folder/"HEAD_UPDATES.jsonl"
    logs=[{"stage":"TRAINING_UPDATE","update":i+1,"loss":.4,"head_weight_gradient_norm_before_clip":.3,"head_bias_gradient_norm_before_clip":.4,"global_gradient_norm_before_clip":.5,"clipped":False,"gradient_clipping_parameter_scope":"HEAD_ONLY",**copy.deepcopy(audit.FLAGS)} for i in range(600)]
    path.write_text("\n".join(json.dumps(v) for v in logs),encoding="utf-8"); order=np.random.default_rng(92601).integers(0,len(raw["history"]),size=(600,4))
    row={"updates_log":audit.base.file_record(path),"training_updates_completed":600,"seed":92601,"order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"FIT_channel_observed_counts":raw["observed"].sum((0,1)).tolist(),"post_optimizer_frozen_flag_checks":600,"frozen_state_checkpoint_byte_checks":5,"final_frozen_status":copy.deepcopy(audit.FLAGS),"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_update_frozen_bytes_replayed":False,"gradient_clipping_parameter_scope":"HEAD_ONLY","global_clipping_fraction":0.}
    return row,raw,logs,path


class MatchedNativeAuditTests(unittest.TestCase):
    def test_affine_is_per_latent_and_head_last_anchor_with_independent_loops(self):
        rng=np.random.default_rng(1); past=rng.normal(size=(2,512,2)).astype(np.float32)
        state={"E":np.zeros((3,2),np.float32),"D":np.zeros((14,3),np.float32),"center":np.zeros(3,np.float32),"head.weight":rng.normal(scale=.001,size=(2,48,512)).astype(np.float32),"head.bias":rng.normal(scale=.001,size=(2,48)).astype(np.float32)}
        actual=audit.head_future(state,past); expected=np.empty_like(actual)
        for n in range(2):
            for k in range(2):
                for h in range(48): expected[n,h,k]=past[n,-1,k]+((past[n,:,k]-past[n,-1,k])@state["head.weight"][k,h]+state["head.bias"][k,h])
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=2e-6)
        changed=copy.deepcopy(state); changed["head.weight"][1]+=np.float32(.01)
        audit.base.exact(actual[:,:,0],audit.head_future(changed,past)[:,:,0],"other_latent_unchanged")
        state["head.weight"][:]=0.; state["head.bias"][:]=0.
        audit.base.exact(audit.head_future(state,past),np.ascontiguousarray(np.broadcast_to(past[:,-1:],(2,48,2))),"zero_head_future")

    def test_independent_native_future_cached_history_and_lag_scores_with_masked_NaN(self):
        state,raw,saved=native_fixture(); scores=audit.replay(saved,state,raw,{"latent_history":saved["latent_history"].copy()})
        self.assertTrue(np.isnan(raw["target"][~raw["observed"]]).all()); self.assertTrue(np.isfinite(scores["normalized"]["mse"]))
        for key,label in (("latent_future","native_affine"),("latent_history","BYTE_EXACT"),("prediction","fixed_D_forecast")):
            bad=copy.deepcopy(saved); bad[key].flat[0]+=.1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,label): audit.replay(bad,state,raw,{"latent_history":saved["latent_history"]})

    def test_frozen_E_D_center_byte_change_rejected_head_may_change(self):
        state,_,_=native_fixture(); initial={k:state[k].copy() for k in ("E","D","center")}; audit.frozen_members(state,initial)
        for key in initial:
            bad=copy.deepcopy(state); bad[key].flat[0]=np.nextafter(bad[key].flat[0],np.float32(np.inf))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"BYTE_EXACT"): audit.frozen_members(bad,initial)

    def test_recorded_head_only_flags_full_FIT_counts_seed_and_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            row,raw,logs,path=logged_fixture(Path(folder)); self.assertEqual(audit.check_logs(row,raw),600)
            for field,value in (("D_requires_grad",True),("head_weight_gradient_norm_before_clip",float("inf")),("gradient_clipping_parameter_scope","E_AND_D")):
                bad=copy.deepcopy(logs); bad[3][field]=value; path.write_text("\n".join(json.dumps(v) for v in bad),encoding="utf-8"); row["updates_log"]=audit.base.file_record(path)
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,"HEAD_ONLY"): audit.check_logs(row,raw)
            path.write_text("\n".join(json.dumps(v) for v in logs),encoding="utf-8"); row["updates_log"]=audit.base.file_record(path)
            for key,value,label in (("FIT_channel_observed_counts",[1,1,1],"FULL_FIT"),("order_sha256","wrong","SEED_ORDER"),("independent_per_update_frozen_bytes_replayed",True,"FROZEN_SCOPE")):
                bad=copy.deepcopy(row); bad[key]=value
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,label): audit.check_logs(bad,raw)

    def test_new_space_native_label_and_separate_frozen_E_D_and_head_supervision(self):
        state,raw,saved=native_fixture()
        with tempfile.TemporaryDirectory() as folder:
            for name,head_cp,e_cp in (("E_ONLY_s92601",0,400),("RECON_CP0",100,0)):
                path=Path(folder)/(name+".npz"); cell=audit.new_space(saved,state,raw,name,head_cp,e_cp,path)
                self.assertEqual(set(cell["methods"]),{"ACTUAL_NATIVE","EY","FLAT"}); self.assertEqual(cell["selected_E_previously_supervised_by_FIT_future_targets"],e_cp>0); self.assertFalse(cell["selected_D_previously_supervised_by_FIT_future_targets"]); self.assertEqual(cell["selected_native_head_previously_supervised_by_FIT_future_targets"],head_cp>0)
                self.assertTrue(cell["native_head_training_procedure_used_FIT_future_targets"]); self.assertTrue(cell["duplicate_EY_flat_minimum_across_head_seeds_are_not_independent_datasets"])
                self.assertFalse(cell["independent_external_replay_of_new_diagnostic_claimed"]); self.assertFalse(cell["all_real_latent_or_all_FP32_lower_bound_claimed"]); self.assertTrue(cell["independent_reduced_LS_vs_SVD_verified"])
                self.assertNotIn("selected_E_D_previously_supervised_by_FIT_future_targets",cell)
                with np.load(path,allow_pickle=False) as arrays:
                    self.assertFalse(any(k.startswith("ACTUAL_F_") for k in arrays.files)); audit.base.exact(arrays["ACTUAL_NATIVE_prediction_FP32"],saved["prediction"],"saved_native")
                    audit.base.exact(arrays["EY_input_C_FP32"][0,1],raw["raw_last"][0],"complete_C_fill")
                json.dumps(cell,allow_nan=False)

    def test_closed_actual_gate_and_optional_bytes_record_identity(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.base,"file_record",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): audit.metadata("missing",lambda:None)
        item={"path":str(audit.HERE/"fixture.json"),"sha256":"a"*64}; audit.same_binding(item,{**item,"bytes":3})
        with self.assertRaises(ValueError): audit.same_binding(item,{**item,"sha256":"b"*64})


if __name__=="__main__": unittest.main()
