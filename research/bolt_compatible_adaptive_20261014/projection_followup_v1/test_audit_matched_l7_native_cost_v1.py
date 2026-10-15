"""Generated CPU arrays only; no real result, checkpoint, data or F access."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_matched_l7_native_cost_v1 as audit
import whole_matched_l7_native_cost_v1 as timer
from lagged_native_head_v1 import LaggedNativeForecaster
from run_linear_e_cost_v1 import physical_precheck

audit.initialize_numpy()


def memory():
    return {"RSS_bytes":4096,"allocated_VRAM_bytes":100,"reserved_VRAM_bytes":200,"max_allocated_VRAM_bytes":300,"max_reserved_VRAM_bytes":400,"scope":audit.MEMORY_SCOPE}


def fixture(folder):
    rng=np.random.default_rng(357); physical=rng.normal(size=(8,512,3)).astype(np.float32); mask=np.ones_like(physical,bool); mask[:,:7,1]=False; physical[~mask]=np.nan
    mean,std=np.array([1.,2.,3.],np.float32),np.array([2.,3.,4.],np.float32)
    history=audit.prior.causal_physical_preprocess(physical,mask,mean,std); origins=np.arange(8,dtype=np.int64)+512
    raw={"history":history,"history_observed":mask,"origins":origins,"mean":mean,"std":std,"raw_last":history[:,-1].copy(),"target":np.zeros((8,48,3),np.float32),"observed":np.ones((8,48,3),bool)}
    callbacks={route:(lambda h,m,j=j:np.broadcast_to(h[:,-1:]+np.float32(j*.001),(len(h),48,h.shape[-1])).copy()) for j,route in enumerate(audit.ROUTES)}
    predictions={r:fn(history,mask) for r,fn in callbacks.items()}; metadata={r:{"fixture_route":r} for r in audit.ROUTES}
    result=timer.measure_cost("synthetic",physical,mask,mean,std,origins,callbacks,predictions,metadata,folder,lambda:None,lambda:None,memory)
    return result,{"history_physical":physical,"history_observed":mask,"origins":origins},raw,predictions,metadata


class MatchedL7CostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(dir=audit.HERE); cls.addClassCleanup(cls.folder.cleanup)
        cls.values=fixture(Path(cls.folder.name)/"cost")

    def test_all_twenty_cells_one_hundred_saved_samples_and_both_denominators(self):
        result=audit.check_dataset(*self.values,lambda:None,0.)
        self.assertEqual((result["complete_cells"],result["saved_samples"]),(20,100)); self.assertEqual(len(result["rows"]),20)
        native=next(row for row in result["rows"] if "_NATIVE_" in row["route"])
        self.assertEqual(native["matched_Bolt"]["route"],native["route"].split("_NATIVE_")[0]+"_BOLT")
        self.assertFalse(result["F_or_timer_replayed"]); json.dumps(result,allow_nan=False)

    def test_changed_ns_reference_or_denominator_is_rejected(self):
        for key in ("ns","reference","matched_ratio","raw_ratio","matched_route"):
            values=copy.deepcopy(self.values); row=values[0]["rows"][2]
            if key=="ns": row["samples"][0]["perf_end_ns"]+=1
            elif key=="reference": values[3][row["route"]][0,0,0]+=.1
            elif key=="matched_ratio": row["ratio_of_matched_Bolt_medians"]+=.1
            elif key=="raw_ratio": row["paired_RAW_median_ratio"]+=.1
            else: row["matched_Bolt_route"]="RAW_BOLT"
            with self.subTest(key=key),self.assertRaises(ValueError): audit.check_dataset(*values,lambda:None,0.)

    def test_missing_route_duplicate_repeat_or_exclusive_memory_is_rejected(self):
        for key in ("route","repeat","memory","prelock"):
            values=copy.deepcopy(self.values)
            if key=="route": values[0]["rows"].pop()
            elif key=="repeat": values[0]["rows"][0]["samples"][1]["repeat"]=0
            elif key=="memory": values[0]["rows"][0]["memory"][0]["scope"]="exclusive route allocation"
            with self.subTest(key=key),self.assertRaises(ValueError): audit.check_dataset(*values,lambda:None,1e20 if key=="prelock" else 0.)

    def test_schedule_is_five_distinct_positions_not_complete_ten_balance(self):
        self.assertEqual(audit.schedule(),timer.balanced_schedule())
        for route in audit.ROUTES: self.assertEqual(len({order.index(route) for order in audit.schedule()}),5)
        texts=[(audit.HERE/name).read_text(encoding="utf-8") for name in ("whole_matched_l7_native_cost_v1.py","run_matched_l7_native_cost_v1.py","whole_f_input_gradient_cost_v1.py")]
        scope=audit.source_contract(*texts)
        self.assertTrue(scope["internal_shape_finite_guards_inside_timer"]); self.assertFalse(scope["complete_ten_position_balance_claimed"])
        wrong=texts.copy(); wrong[1]=wrong[1].replace('canonical_tune_bridge(original,"robin")','canonical_tune_bridge(original,"bad")')
        with self.assertRaises(ValueError): audit.source_contract(*wrong)

    def test_native_physical_precheck_and_independent_head_equation(self):
        _,physical,raw,_,_=self.values; rng=np.random.default_rng(112)
        state={"center":np.array([.1,.2,.3],np.float32),"E":rng.normal(size=(3,2)).astype(np.float32),"D":rng.normal(scale=.01,size=(14,3)).astype(np.float32)}
        model=LaggedNativeForecaster(state["center"],state["E"],state["D"],audit.LAGS).eval().requires_grad_(False)
        with torch.no_grad():
            model.head.weight.copy_(torch.as_tensor(rng.normal(scale=.0001,size=(2,48,512)).astype(np.float32)))
            model.head.bias.copy_(torch.as_tensor(rng.normal(scale=.001,size=(2,48)).astype(np.float32)))
            prediction,z,future=model(torch.as_tensor(raw["history"]),None,return_components=True)
            reference={**raw,"prediction":prediction.numpy().copy(),"latent_history":z.numpy().copy(),"latent_future":future.numpy().copy(),"lag_features":model.lag_features(z,future).numpy().copy()}
        entry,arrays=physical_precheck(physical,raw,reference,model,None,lambda x:torch.as_tensor(x))
        self.assertEqual(audit.legacy.precheck(entry,arrays,physical,raw,state,reference,audit.LAGS)["windows_checked"],8)
        selected={key:value.detach().numpy().copy() for key,value in model.state_dict().items()}; audit.check_native_head(arrays,selected)
        changed=copy.deepcopy(selected); changed["head.bias"][0,0]+=.1
        with self.assertRaises(ValueError): audit.check_native_head(arrays,changed)
        changed=copy.deepcopy(state); changed["E"][0,0]+=.1
        with self.assertRaises(ValueError): audit.legacy.precheck(entry,arrays,physical,raw,changed,reference,audit.LAGS)

    def test_all_full_state_and_head_only_signatures_are_distinct(self):
        state={"E":np.zeros((3,2),np.float32),"D":np.zeros((14,3),np.float32),"center":np.zeros(3,np.float32),"head.weight":np.zeros((2,48,512),np.float32),"head.bias":np.zeros((2,48),np.float32)}
        before=audit.state_signature(state); state["head.bias"][0,0]=1
        after=audit.state_signature(state); self.assertNotEqual(before["state_sha256"],after["state_sha256"]); self.assertEqual(before["arrays"]["E"],after["arrays"]["E"])

    def test_unregistered_cost_gate_does_not_read_assets(self):
        with patch.object(audit,"COST_SHA",None),patch.object(audit.linear,"file_record",side_effect=AssertionError("closed asset IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): audit.audit_result("missing",None)


if __name__=="__main__": unittest.main()
