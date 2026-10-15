"""Small synthetic CPU audit controls; no actual scientific payload or model."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_orthogonal_q_learning_v1 as audit
import run_orthogonal_q_learning_v1 as producer
from test_run_orthogonal_q_learning_v1 import fixture,graph

audit.initialize_numpy(); torch.set_num_threads(2)


def cell(*,cp0=False,complete=True):
    model,bolt,raw,cached,frozen = fixture()
    if complete:
        raw["observed"][:,:,5] = True; raw["target"][:,:,5] = np.float32(.2)
    if not cp0:
        with torch.no_grad(): model.theta.copy_(torch.tensor([.1,-.2,.03,.04,-.08,.12]))
    outputs,_ = producer.fixed.infer_control(model,bolt,raw["history"],cached,lambda x:None)
    q,theta = model.basis().detach().numpy().copy(),model.theta.detach().numpy().copy()
    row,independent = producer.evaluate_arrays(outputs,raw,cached,frozen,q,cp0=cp0)
    arrays = {**outputs,"Q_FP32":q,"theta_FP32":theta,**{key:raw[key].copy() for key in ("target","observed","raw_last","origins","mean","std")}}
    return row,arrays,independent,raw,cached,frozen,theta,q


def logs_fixture():
    fit = {"history":np.zeros((8,35,6),np.float32),"observed":np.ones((8,5,6),bool)}
    fit["observed"][:,:,5] = False; gradient = np.arange(1,7,dtype=np.float32)*np.float32(.1)
    norm = float(np.linalg.norm(gradient)); logs = []
    for update in range(1,601):
        logs.append({"stage":"TRAINING_UPDATE","update":update,"loss":.1,"theta_gradient_before_clip":gradient.tolist(),"theta_gradient_norm_before_clip":norm,"global_gradient_norm_before_clip":norm,"clipped":norm > 1.,"theta_after_optimizer_step":np.full(6,update*1e-5,np.float32).tolist(),**audit.FLAGS})
    order = np.random.default_rng(92601).integers(0,8,size=(600,4)); digest = hashlib.sha256(order.tobytes()).hexdigest()
    row = {"seed":92601,"training_updates_completed":600,"post_optimizer_frozen_flag_checks":600,"frozen_state_checkpoint_byte_checks":5,"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_step_backward_optimizer_or_frozen_bytes_replayed":False,"global_clipping_fraction":float(norm > 1.),"FIT_channel_observed_counts":fit["observed"].sum((0,1)).tolist(),"loss":"N_fit/B/C_active * sum_c SSE_batch_c/N_FIT_c; fixed entire FIT observed counts","order_sha256":digest,"final_frozen_status":audit.FLAGS.copy()}
    return row,logs,fit,digest


def probe_fixture(*,linear=False,zero_decoder=False):
    model,bolt,raw,_,_ = fixture(linear=linear,zero_decoder=zero_decoder)
    batch = {key:value[:4] if key in ("history","target","observed") else value for key,value in raw.items()}
    loss,capture,future = graph(model,bolt,batch)
    row,arrays = producer.first_batch_gradients(model,loss,capture["rotated"],capture["F_future"],future)
    arrays["FIT_window_indices"] = np.arange(4,dtype=np.int64); row["artifact"] = {"path":"synthetic_probe","sha256":"a"*64}
    return row,arrays,np.arange(4,dtype=np.int64)[None],{"theta_gradient_before_clip":arrays["theta_gradient_full"].tolist()}


def selection_fixture():
    checkpoints = []
    for update in audit.CPS:
        checkpoints.append({"update":update,"tune":{"mse":.1,"mae":.2},"state":{"path":"synthetic_CP"+str(update),"sha256":str(update).zfill(64)},"basis_diagnostics":{"synthetic":True},"frozen_status":audit.FLAGS.copy(),"E_D_center_exact_initial_bytes":True,"canonical_score_link_atol":1e-12,"canonical_score_link_rtol":1e-12})
    selected = checkpoints[0]
    row = {"checkpoint_count":5,"selected_update":0,"selected_tune":selected["tune"].copy(),"selected_state":selected["state"].copy(),"selected_basis_diagnostics":selected["basis_diagnostics"].copy(),"TUNE":{k:v for k,v in selected.items() if k not in ("state","basis_diagnostics","frozen_status","E_D_center_exact_initial_bytes")}}
    return row,checkpoints


def registry_fixture():
    result = {"all_6_locks_before_DEV":True,"resolved_selection_locks":6,"states":{s:{"learning":{}} for s in audit.STATES}}
    registry = {"status":"ALL_SIX_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"sealed_unix":20.,"conditions":[]}; locks = {}
    for state in audit.STATES:
        for seed in audit.SEEDS:
            path = state+str(seed); binding = {"path":path,"sha256":"a"*64}
            row = {"status":"COMPLETE","state_label":state,"seed":seed,"selected_update":600,"selection_lock":binding,"DEV":{"status":"COMPLETE"}}
            result["states"][state]["learning"][str(seed)] = row
            registry["conditions"].append({"state":state,"seed":seed,"status":"COMPLETE","selection_lock":binding})
            locks[path] = {k:v for k,v in row.items() if k not in ("DEV","selection_lock")}|{"DEV_opened_before_lock":False,"sealed_unix":10.}
    return result,registry,locks


class QLearningIndependentAuditTests(unittest.TestCase):
    def test_independent_skew_exponential_and_SO_diagnostics_at_zero_and_dense_theta(self):
        for cp0 in (False,True):
            values = cell(cp0=cp0); model,_,_,_,_ = fixture()
            with torch.no_grad(): model.theta.copy_(torch.from_numpy(values[6]))
            result = audit.check_basis(values[6],values[7],model.basis_diagnostics())
            self.assertFalse(result["basis_diagnostics"]["identity_bypass"]); json.dumps(result,allow_nan=False)
            expected = torch.matrix_exp(torch.tensor([[0.,.1,0.,0.],[-.1,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]],dtype=torch.float64)).numpy()
            np.testing.assert_allclose(audit.independent_Q(np.array([.1,0,0,0,0,0],np.float32)),expected,atol=2e-9,rtol=2e-9)

    def test_Q_values_dtype_determinant_and_theta_mapping_mutations_fail(self):
        for kind in ("Q","theta","dtype","det","bypass"):
            row,arrays,_,_,_,_,theta,q = cell(); model,_,_,_,_ = fixture()
            with torch.no_grad(): model.theta.copy_(torch.from_numpy(theta))
            recorded = model.basis_diagnostics()
            if kind == "Q": q[0,1] += .01
            elif kind == "theta": theta[0] += .1
            elif kind == "dtype": q = q.astype(np.float64)
            elif kind == "det": recorded["determinant"] += .1
            else: recorded["identity_bypass"] = True
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_basis(theta,q,recorded)

    def test_independent_inverse_Q_original_lag_D_masks_scores_and_CP0(self):
        for cp0 in (False,True):
            with self.subTest(cp0=cp0):
                result = audit.check_cell(*cell(cp0=cp0),cp0=cp0)
                self.assertEqual(result["score_fields_checked"],72); self.assertFalse(result["F_forward_independently_replayed"])
                self.assertEqual(result["CP0_parity_required"],cp0); json.dumps(result,allow_nan=False)

    def test_saved_forward_independent_context_score_and_tolerance_tampering_fails(self):
        for kind in ("rotated","inverse","C","transport","independent","theta","origins","score","count","tol","summary"):
            values = cell()
            if kind == "rotated": values[1]["rotated_past"][0,0,0] += .1
            elif kind == "inverse": values[1]["unrotated_future"][0,0,0] += .1
            elif kind == "C": values[1]["prediction"][0,0,0] += .1
            elif kind == "transport": values[1]["transport_prediction"][0,0,0] += .1
            elif kind == "independent": values[2]["independent_NumPy_future"][0,0,0] += .1
            elif kind == "theta": values[1]["theta_FP32"] = values[6]+np.float32(.1)
            elif kind == "origins": values[1]["origins"][0] += 1
            elif kind == "score": values[0]["scores_all_observed"]["normalized"]["mse"] += .01
            elif kind == "count": values[0]["complete_target_positions"] += 1
            elif kind == "tol": values[0]["checks"]["NumPy_inverse_Q_future"]["atol"] = .1
            else: values[0]["C_commutator"]["by_horizon_RMS"][0] += .1
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_cell(*values)

    def test_unobserved_labels_do_not_change_scores_and_no_complete_C_is_not_zero(self):
        values = cell(complete=False); result = audit.check_cell(*values)
        self.assertEqual(result["score_fields_checked"],36); self.assertEqual(result["scores_complete_mask"],{"status":"NO_COMPLETE_C"})
        values[3]["target"][~values[3]["observed"]] = np.inf; values[1]["target"] = values[3]["target"].copy()
        self.assertEqual(result,audit.check_cell(*values))
        values[3]["target"][values[3]["observed"]] = np.nan; values[1]["target"] = values[3]["target"].copy()
        with self.assertRaises(ValueError): audit.check_cell(*values)

    def test_original_selection_reduction_matches_all_and_partial_observed_native_score(self):
        from native_head_v1 import score_predictions
        rng = np.random.default_rng(391); prediction = rng.normal(size=(7,11,6)).astype(np.float32); target = rng.normal(size=prediction.shape).astype(np.float32)
        for partial in (False,True):
            observed = rng.uniform(size=target.shape) > .3 if partial else np.ones_like(target,bool)
            if partial: observed[:,:,5] = False; target[~observed] = np.nan
            self.assertEqual(audit.selection_score(prediction,target,observed),score_predictions(prediction,target,observed))

    def test_log_counts_global_FIT_denominators_norms_flags_and_parent_order(self):
        values = logs_fixture(); order = audit.check_logs(*values)
        self.assertEqual(order.shape,(600,4)); self.assertEqual(order.dtype,np.int64)
        for kind in ("duplicate","loss","norm","clip","Fgrad","counts","order","theta"):
            v = copy.deepcopy(values)
            if kind == "duplicate": v[1][2]["update"] = 2
            elif kind == "loss": v[1][0]["loss"] = float("nan")
            elif kind == "norm": v[1][0]["theta_gradient_norm_before_clip"] += .1
            elif kind == "clip": v[1][0]["clipped"] = not v[1][0]["clipped"]
            elif kind == "Fgrad": v[1][0]["F_grad_is_None"] = False
            elif kind == "counts": v[0]["FIT_channel_observed_counts"][0] -= 1
            elif kind == "order": v[0]["order_sha256"] = "0"*64
            else: v[1][0]["theta_after_optimizer_step"][0] = float("inf")
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_logs(*v)

    def test_saved_gradient_closure_norms_first_batch_and_no_nonzero_requirement(self):
        for linear in (False,True):
            with self.subTest(linear=linear):
                result = audit.check_probe(*probe_fixture(linear=linear),length=35,horizon=5)
                self.assertFalse(result["independent_F_VJP_or_full_backward_replayed"]); json.dumps(result,allow_nan=False)
        zero = audit.check_probe(*probe_fixture(zero_decoder=True),length=35,horizon=5)
        self.assertEqual(zero["full_theta_gradient_norm"],0.); self.assertFalse(zero["F_input_path_nonzero_observed"])
        for kind in ("residual","norm","full","first_log","batch","tol","claim"):
            v = copy.deepcopy(probe_fixture())
            if kind == "residual": v[1]["theta_gradient_closure_residual"][0] += .1
            elif kind == "norm": v[0]["theta_F_input_chain_norm"] += .1
            elif kind == "full": v[1]["theta_gradient_full"][0] += .1
            elif kind == "first_log": v[3]["theta_gradient_before_clip"][0] += .1
            elif kind == "batch": v[1]["FIT_window_indices"][0] += 1
            elif kind == "tol": v[0]["gradient_closure_atol"] = .1
            else: v[0]["nonzero_total_theta_gradient_alone_proves_F_input_path"] = True
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_probe(*v,length=35,horizon=5)

    def test_CP_selection_uses_exact_earliest_tie_and_reuses_selected_entry(self):
        row,checkpoints = selection_fixture(); self.assertEqual(audit.check_selection(row,checkpoints)["update"],0)
        checkpoints[-1]["tune"]["mse"] -= 1e-16
        with self.assertRaisesRegex(ValueError,"OWN_TUNE_EXACT"): audit.check_selection(row,checkpoints)
        for kind in ("missing","selected","state","TUNE"):
            r,c = copy.deepcopy(selection_fixture())
            if kind == "missing": c.pop()
            elif kind == "selected": r["selected_update"] = 600
            elif kind == "state": r["selected_state"]["sha256"] = "a"*64
            else: r["TUNE"]["update"] = 600
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_selection(r,c)

    def test_six_locks_bind_entire_pre_DEV_condition_and_precede_registry(self):
        r,g,l = registry_fixture(); self.assertEqual(len(audit.check_registry(r,g,lambda item:l[item["path"]])),6)
        for kind in ("duplicate","earlyDEV","changedCP","late","failed","lockscope"):
            r,g,l = copy.deepcopy(registry_fixture())
            if kind == "duplicate": g["conditions"][-1] = copy.deepcopy(g["conditions"][0])
            elif kind == "earlyDEV": g["DEV_opened_before_seal"] = True
            elif kind == "changedCP": next(iter(l.values()))["selected_update"] = 400
            elif kind == "late": next(iter(l.values()))["sealed_unix"] = 21.
            elif kind == "failed": g["conditions"][0]["status"] = "FAILED"
            else: next(iter(l.values()))["DEV_opened_before_lock"] = True
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_registry(r,g,lambda item:l[item["path"]])

    def test_actual_hash_closed_gate_precedes_IO_and_import_is_numpy_torch_free(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.base,"file_record",side_effect=AssertionError("IO forbidden")):
            with self.assertRaisesRegex(ValueError,"NOT_REGISTERED"): audit.audit_result("missing",None)
        with patch.object(audit.base,"file_record",return_value={"path":"synthetic","sha256":"0"*64}),patch.object(audit.prior,"verified_json",side_effect=AssertionError("scientific IO forbidden")):
            with self.assertRaisesRegex(ValueError,"EXACT_ACTUAL"): audit.audit_result("synthetic",type("Meter",(),{"check":lambda self:None})())
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(n in ("run_orthogonal_q_learning_v1","orthogonal_latent_basis_v1","torch") for n in imports))
        command = "import sys;sys.path.insert(0,"+repr(str(audit.HERE))+");import audit_orthogonal_q_learning_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('PURE_IMPORT_PASS')"
        completed = subprocess.run([sys.executable,"-c",command],capture_output=True,text=True,check=True)
        self.assertIn("PURE_IMPORT_PASS",completed.stdout)


if __name__ == "__main__": unittest.main()
