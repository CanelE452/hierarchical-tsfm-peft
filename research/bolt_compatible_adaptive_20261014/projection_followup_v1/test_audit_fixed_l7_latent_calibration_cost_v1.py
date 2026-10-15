"""Generated CPU arrays/metadata only; no actual result, model, F or GPU access."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_fixed_l7_latent_calibration_cost_v1 as audit
import whole_fixed_l7_latent_calibration_cost_v1 as timer
from causal_lag_decoder_v1 import predict,lagged_delta_features

audit.initialize_numpy()


def memory():
    return {"RSS_bytes":4096,"allocated_VRAM_bytes":100,"reserved_VRAM_bytes":200,"max_allocated_VRAM_bytes":300,"max_reserved_VRAM_bytes":400,"scope":audit.MEMORY_SCOPE}


def cost_fixture(folder):
    rng=np.random.default_rng(66); x=rng.normal(size=(8,512,3)).astype(np.float32); mask=np.ones_like(x,bool); mask[:,:5,0]=False; x[~mask]=np.nan
    mean,std=np.array([1.,2.,3.],np.float32),np.array([2.,3.,4.],np.float32); normalized=audit.prior.causal_physical_preprocess(x,mask,mean,std); origins=np.arange(8,dtype=np.int64)+512
    raw={"history":normalized,"history_observed":mask,"mean":mean,"std":std,"origins":origins,"raw_last":normalized[:,-1].copy(),"target":np.zeros((8,48,3),np.float32),"observed":np.ones((8,48,3),bool)}
    callbacks={route:(lambda h,m,j=j:np.broadcast_to(h[:,-1:]+np.float32(j*.01),(len(h),48,h.shape[-1])).copy()) for j,route in enumerate(audit.ROUTES)}
    predictions={route:callback(normalized,mask) for route,callback in callbacks.items()}; metadata={route:{"fixture_route":route} for route in audit.ROUTES}
    result=timer.measure_cost("synthetic",x,mask,mean,std,origins,callbacks,predictions,metadata,folder,lambda:None,lambda:None,memory)
    return result,{"history_physical":x,"history_observed":mask,"origins":origins},raw,predictions,metadata


def reference_fixture(kind="RIDGE",family="SCALE_BIAS"):
    rng=np.random.default_rng(317); p=rng.normal(size=(8,512,2)).astype(np.float32); u=rng.normal(size=(8,48,2)).astype(np.float32); d=rng.normal(scale=.1,size=(14,3)).astype(np.float32); current=rng.normal(size=(8,3)).astype(np.float32)
    a=np.array([.2,-.3],np.float32) if kind=="RIDGE" else np.zeros(2,np.float32); b=np.array([.1,-.15],np.float32) if kind=="RIDGE" and family=="SCALE_BIAS" else np.zeros(2,np.float32)
    q=u if kind=="IDENTITY" else u+a*(u-p[:,-1:])+b
    baseline={"prediction":predict(p,u,current,d,audit.LAGS),"latent_history":p,"latent_future":u,"lag_features":lagged_delta_features(p,u,audit.LAGS),"raw_last":current,"origins":np.arange(8,dtype=np.int64)}
    selected={"prediction_FP32":predict(p,q,current,d,audit.LAGS),"latent_future_calibrated_FP32":q,"origins":baseline["origins"].copy()}
    compound={**baseline,"prediction":selected["prediction_FP32"],"latent_future":q,"lag_features":lagged_delta_features(p,q,audit.LAGS)}
    return compound,baseline,selected,{"a_FP32":a,"b_FP32":b},kind,d


class FixedQCostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(dir=audit.HERE); cls.addClassCleanup(cls.folder.cleanup); cls.values=cost_fixture(Path(cls.folder.name)/"cost")

    def test_all_twenty_cells_one_hundred_saved_timers_and_both_denominators(self):
        result=audit.check_dataset(*self.values,lambda:None,0.)
        self.assertEqual((result["complete_cells"],result["saved_samples"]),(20,100)); self.assertEqual(len(result["rows"]),20); self.assertFalse(result["F_or_timer_replayed"])
        for row in result["rows"]:
            if "_SCALE_" in row["route"]: self.assertEqual(row["matched_Bolt"]["route"],row["route"].split("_SCALE_")[0]+"_BOLT")
        json.dumps(result,allow_nan=False)

    def test_changed_ns_or_selected_reference_or_both_ratio_types_rejected(self):
        for key in ("ns","reference","raw_ratio","matching_ratio","matching_route"):
            values=copy.deepcopy(self.values); row=values[0]["rows"][2]
            if key=="ns": row["samples"][0]["perf_end_ns"]+=1
            elif key=="reference": values[3][row["route"]][0,0,0]+=.1
            elif key=="raw_ratio": row["paired_RAW_median_ratio"]+=.1
            elif key=="matching_ratio": row["ratio_of_matched_Bolt_medians"]+=.1
            else: row["matched_Bolt_route"]="RAW_BOLT"
            with self.subTest(key=key),self.assertRaises(ValueError): audit.check_dataset(*values,lambda:None,0.)

    def test_missing_row_repeat_before_lock_or_exclusive_memory_rejected(self):
        for key in ("route","repeat","memory","prelock"):
            values=copy.deepcopy(self.values)
            if key=="route": values[0]["rows"].pop()
            elif key=="repeat": values[0]["rows"][0]["samples"][1]["repeat"]=0
            elif key=="memory": values[0]["rows"][0]["memory"][0]["scope"]="exclusive"
            with self.subTest(key=key),self.assertRaises(ValueError): audit.check_dataset(*values,lambda:None,1e20 if key=="prelock" else 0.)

    def test_both_q_families_and_identity_composed_reference_have_independent_numpy_certificates(self):
        for family,kind in (("SCALE_ONLY","RIDGE"),("SCALE_BIAS","RIDGE"),("SCALE_ONLY","IDENTITY")):
            with self.subTest(family=family,kind=kind):
                result=audit.check_composed_reference(*reference_fixture(kind,family)); self.assertTrue(result["selected_C_prediction_bytes_preserved"]); self.assertTrue(result["independent_NumPy_q_and_lag_D_from_canonical_F_reference"]); self.assertFalse(result["actual_precheck_original_F_future_independently_replayed"]); json.dumps(result,allow_nan=False)

    def test_changed_composed_history_q_C_origins_or_coefficients_rejected(self):
        for key in ("history","q","C","origins","coefficients"):
            values=list(copy.deepcopy(reference_fixture()))
            if key=="history": values[0]["latent_history"]=values[0]["latent_history"].copy(); values[0]["latent_history"][0,0,0]+=.1
            elif key=="q":
                values[2]["latent_future_calibrated_FP32"][0,0,0]+=.1; values[0]["lag_features"]=lagged_delta_features(values[1]["latent_history"],values[0]["latent_future"],audit.LAGS)
            elif key=="C": values[2]["prediction_FP32"][0,0,0]+=.1
            elif key=="origins": values[2]["origins"][0]+=1
            else: values[3]["a_FP32"][0]+=.1
            with self.subTest(key=key),self.assertRaises(ValueError): audit.check_composed_reference(*values)

    def test_five_of_ten_positions_and_actual_F_identity_source_boundary(self):
        self.assertEqual(audit.schedule(),timer.balanced_schedule())
        for route in audit.ROUTES: self.assertEqual(len({order.index(route) for order in audit.schedule()}),5)
        texts=[(audit.HERE/name).read_text(encoding="utf-8") for name in ("whole_fixed_l7_latent_calibration_cost_v1.py","run_fixed_l7_latent_calibration_cost_v1.py","whole_f_input_gradient_cost_v1.py","calibrated_l7_forecaster_v1.py")]
        scope=audit.source_contract(*texts); self.assertTrue(scope["identity_cost_runs_actual_F_and_D"]); self.assertTrue(scope["internal_shape_finite_guards_inside_timer"]); self.assertFalse(scope["complete_ten_position_balance_claimed"])
        texts[-1]=texts[-1].replace("original_future=bolt(zpast)","original_future=zpast[:,-1:].expand(-1,48,-1)")
        with self.assertRaises(ValueError): audit.source_contract(*texts)

    def test_actual_precheck_E_lag_D_replay_does_not_claim_F_replay(self):
        from calibrated_l7_forecaster_v1 import CalibratedL7Forecaster
        from run_linear_e_cost_v1 import physical_precheck
        rng=np.random.default_rng(97); x=rng.normal(size=(8,512,3)).astype(np.float32); mask=np.ones_like(x,bool); origins=np.arange(8,dtype=np.int64)
        raw={"history":x,"history_observed":mask,"raw_last":x[:,-1].copy(),"target":np.zeros((8,48,3),np.float32),"observed":np.ones((8,48,3),bool),"mean":np.zeros(3,np.float32),"std":np.ones(3,np.float32),"origins":origins}; physical={"history_physical":x,"history_observed":mask,"origins":origins}
        state={"center":np.zeros(3,np.float32),"E":rng.normal(scale=.1,size=(3,2)).astype(np.float32),"D":rng.normal(scale=.1,size=(14,3)).astype(np.float32)}
        model=CalibratedL7Forecaster(state["center"],state["E"],state["D"],np.array([.1,-.2],np.float32),np.array([.3,-.1],np.float32),family="SCALE_BIAS")
        def toyF(z): return z[:,-1:].expand(-1,48,-1)+.2
        with torch.no_grad(): p,z,q=model(torch.as_tensor(x),toyF,return_components=True)
        reference={**raw,"prediction":p.numpy().copy(),"latent_history":z.numpy().copy(),"latent_future":q.numpy().copy(),"lag_features":model.lag_features(z,q).numpy().copy()}
        entry,arrays=physical_precheck(physical,raw,reference,model,toyF,lambda v:torch.as_tensor(v)); result=audit.legacy.legacy.precheck(entry,arrays,physical,raw,state,reference,audit.LAGS)
        self.assertEqual(result["windows_checked"],8); self.assertEqual(result["independent_F_calls"],0); self.assertTrue(result["intermediate_parity_is_separate_from_registered_output_acceptance"])

    def test_unregistered_cost_and_science_pass_gate_precedes_any_asset_read(self):
        with patch.object(audit,"COST_SHA",None),patch.object(audit.linear,"file_record",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): audit.audit_result("missing",None)


if __name__=="__main__": unittest.main()
