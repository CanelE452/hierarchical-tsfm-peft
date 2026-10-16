"""Stdlib-only fixtures; ROOT must run these after the precision-cost window."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("tiny_required_minima",HERE/"required_minima_runner.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class RequiredMinimaTests(unittest.TestCase):
    def setUp(self):
        self.h = load()

    def condition(self,family="B2",k=6):
        return next(row for row in self.h.conditions("campaign-sha") if row["family"]==family and row["latent_k"]==k)

    def test_exact_six_original_recipes_and_two_supported_K(self):
        rows = self.h.conditions("campaign-sha")
        self.assertEqual(len(rows),6)
        self.assertEqual({(r["family"],r["latent_k"]) for r in rows},{(f,k) for f in ("B2","LAG_CAUSAL","LAG_BIDIR") for k in (6,11)})
        for row in rows:
            spec = row["spec"]
            self.assertEqual(spec["epochs"],[.25,1.,3.]); self.assertEqual(spec["device"],"cpu")
            self.assertEqual((spec["microbatch"],spec["effective_batch"],spec["F_row_microbatch"]),(4,16,128))
            self.assertFalse(spec["checkpoint_inputs"])
            self.assertEqual(spec["provenance"]["upstream_registered_id"],row["upstream_registered_id"])
            old = self.h.original_spec(row)
            self.assertEqual(old["epochs"],[.25]); self.assertEqual(old["device"],"cuda")
            self.assertEqual(old["run_id"],row["upstream_registered_id"])
        with self.assertRaises(ValueError): self.h.choose(dict(conditions=rows),"unknown")

    def test_original_recipe_metadata_and_source_binding_mutations_reject(self):
        condition = self.condition()
        contract = dict(columns=["c"+str(i) for i in range(11)],array_sha256={"physical":"opaque"},data_sha256="data")
        source39 = {key:key+"-sha" for key in self.h.FIT_SOURCES}
        spec = self.h.original_spec(condition)
        axes = dict(L=512,H=48,C=11,columns=contract["columns"],fold="primary",source_array_fingerprints=contract["array_sha256"])
        value = dict(spec=spec,files=source39,source_sha=self.h.fingerprint(source39),
            config_sha=self.h.fingerprint(dict(spec=spec,axes=axes)),data_sha="data",model_sha="foundation")
        self.h.validate_original_recipe(value,condition,source39,contract,"foundation")
        for key,patch in (("spec",dict(spec,lr=.01)),("files",dict(source39,**{"training.py":"changed"})),("data_sha","other")):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.h.validate_original_recipe(dict(value,**{key:patch}),condition,source39,contract,"foundation")

    def test_opaque_source_pin_and_safe_paths_reject_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path=root/"source.py"; path.write_bytes(b"tiny")
            self.h.check_pins(root,{"source.py":self.h.sha(path)})
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): self.h.check_pins(root,{"source.py":"0"*64})
            with self.assertRaises(ValueError): self.h.relative(root,"../outside")

    def test_existing_runs_predictions_or_claim_cannot_be_resumed(self):
        condition = self.condition()
        for kind in ("runs","predictions","claims"):
            with tempfile.TemporaryDirectory() as td:
                public,cache = Path(td)/"public",Path(td)/"cache"
                folder = public/self.h.NAME/kind
                folder.mkdir(parents=True)
                if kind=="claims": (folder/("fit_"+condition["run_id"]+".json")).write_text("{}")
                else: (folder/condition["run_id"]).mkdir()
                with self.assertRaises(FileExistsError): self.h.require_fresh(public,cache,"fit",condition["run_id"])

    def test_original_probe_loss_population_and_actual_F_counts_for_K6_K11(self):
        for k in (6,11):
            for updates in (1,3):
                condition = self.condition("LAG_CAUSAL",k)
                foundation = types.SimpleNamespace(streams=0,row_calls=0,physical_streams=0,physical_calls=0,assert_frozen_unchanged=mock.Mock())
                captured={}
                def segment(model,windows,legal,sampler,optimizer,end,predict,**kwargs):
                    captured.update(legal=legal,end=end,kwargs=kwargs)
                    foundation.streams=foundation.physical_streams=16*updates*k
                    foundation.row_calls=foundation.physical_calls=4*updates
                    return dict(status="SEGMENT_COMPLETE",updates=updates,exposures=16*updates,optimizer_updates_this_segment=updates,
                        k_exposures={},gradient_norm_min=.1,gradient_norm_max=.2,segment_seconds=.01)
                result = self.h.probe_call(segment,None,None,range(9960),None,None,lambda x,k:None,foundation,condition,1.,updates)
                self.assertEqual(len(captured["legal"]),9960)
                self.assertEqual(captured["kwargs"],dict(effective_batch=16,microbatch=4,starting_updates=0,budget_deadline=1.))
                self.assertEqual(result["F_counter_delta"]["streams"],16*updates*k)
                self.assertTrue(result["full_A50_loss_normalization_preserved"])
                self.assertTrue(result["probe_state_discarded"])
        bad = dict(status="SEGMENT_COMPLETE",updates=1,exposures=16,optimizer_updates_this_segment=1,k_exposures={},gradient_norm_min=.1,gradient_norm_max=.2)
        with self.assertRaises(ValueError):
            self.h.probe_call(lambda *a,**kw:bad,None,None,range(9960),None,None,None,foundation,self.condition(),1.,1)

    def test_probe_input_gradient_hook_shape_and_no_future_fit_state(self):
        events = [dict(shape=[4,512,11],requires_grad=True,grad_enabled=True) for _ in range(12)]
        self.h.validate_input_gradient_events(events,11,3)
        wrong=copy.deepcopy(events);wrong[0]["requires_grad"]=False
        with self.assertRaises(ValueError): self.h.validate_input_gradient_events(wrong,11,3)

    def test_actual_backward_events_require_full_count_finite_and_some_nonzero(self):
        for k in (6,11):
            for updates in (1,3):
                events = [dict(shape=[4,512,k],finite=True,absmax=.001 if index==0 else 0.) for index in range(4*updates)]
                value = self.h.validate_backward_gradient_events(events,k,updates)
                self.assertTrue(value["actual_backward_input_gradients_observed"])
                self.assertEqual(value["actual_backward_input_gradient_count"],4*updates)
                self.assertEqual(value["actual_backward_input_gradient_nonzero_count"],1)
                for mutation in ("missing","nonfinite","nan","allzero","shape"):
                    wrong=copy.deepcopy(events)
                    if mutation=="missing":wrong.pop()
                    elif mutation=="nonfinite":wrong[0]["finite"]=False
                    elif mutation=="nan":wrong[0]["absmax"]=float("nan")
                    elif mutation=="shape":wrong[0]["shape"]=[4,512,3]
                    else:
                        for row in wrong:row["absmax"]=0.
                    with self.subTest(k=k,updates=updates,mutation=mutation),self.assertRaises(ValueError):
                        self.h.validate_backward_gradient_events(wrong,k,updates)

    def test_fresh_fit_probe_gate_requires_actual_backward_events_not_forward_flag(self):
        condition=self.condition(); updates=1
        plan=dict(task_contract=dict(primary_split_ranges=dict(A_fit=[0,10519])))
        events=[dict(shape=[4,512,6],finite=True,absmax=.001) for _ in range(4)]
        science=dict(probe_updates=updates,F_counter_delta=dict(streams=96,row_calls=4,physical_streams=96,physical_calls=4),
            full_A50_legal_origins=9960,full_A50_loss_normalization_preserved=True,probe_state_discarded=True,
            probe_state_reused_for_fit=False,saved_model_or_optimizer=False,completed_fit=False,V_scoring=False,
            actual_parameter_gradients_finite=True,no_foundation_parameter_gradients=True,foundation_input_requires_grad_observed=True,
            foundation_input_requires_grad_events=[dict(shape=[4,512,6],requires_grad=True,grad_enabled=True) for _ in range(4)],
            actual_backward_input_gradient_events=events,segment=dict(status="SEGMENT_COMPLETE",updates=1,exposures=16,optimizer_updates_this_segment=1,k_exposures={}),
            **self.h.validate_backward_gradient_events(events,6,1))
        value=dict(status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED",mode="probe",run_id=condition["run_id"],
            plan_sha256="plan",runner_sha256=self.h.sha(HERE/"required_minima_runner.py"),frozen_foundation_unchanged=True,
            original_source_registry_selection_T_pins_preserved=True,cuda_initialized=False,new_T_target_or_scoring=False,scientific_result=science)
        self.h.validate_probe_proof(value,condition,plan,"plan")
        for mutation in ("missing","allzero","summary"):
            wrong=copy.deepcopy(value);body=wrong["scientific_result"]
            if mutation=="missing":body.pop("actual_backward_input_gradient_events")
            elif mutation=="summary":body["actual_backward_input_gradient_count"]=1
            else:
                for row in body["actual_backward_input_gradient_events"]:row["absmax"]=0.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.h.validate_probe_proof(wrong,condition,plan,"plan")

    @unittest.skipUnless(os.environ.get("INFORMS_REQUIRED_AUTOGRAD_TOY")=="1","ROOT opt-in after precision-cost quiet window; imports real Torch")
    def test_F_module_backward_excludes_decoder_direct_z_gradient_when_F_detached(self):
        import torch
        class TinyF(torch.nn.Module):
            def __init__(self,detach_output):
                super().__init__(); self.detach_output=detach_output
            def forward(self,value):
                future=2*value
                return future.detach() if self.detach_output else future
        for detached in (True,False):
            foundation=TinyF(detached)
            forward_events,backward_events=[],[]
            forward_handle,backward_handle=self.h.attach_probe_hooks(foundation,torch,forward_events,backward_events)
            try:
                for _ in range(4):
                    z=torch.ones(4,512,6,requires_grad=True)
                    future=foundation(z)
                    (z+future).square().mean().backward()
                    self.assertIsNotNone(z.grad)
                    self.assertTrue(torch.isfinite(z.grad).all().item())
                    self.assertGreater(z.grad.abs().max().item(),0.)
            finally:
                forward_handle.remove(); backward_handle.remove()
            self.h.validate_input_gradient_events(forward_events,6,1)
            if detached:
                self.assertEqual(backward_events,[])
                with self.assertRaises(ValueError):self.h.validate_backward_gradient_events(backward_events,6,1)
            else:
                self.assertEqual(len(backward_events),4)
                facts=self.h.validate_backward_gradient_events(backward_events,6,1)
                self.assertEqual(facts["actual_backward_input_gradient_nonzero_count"],4)
                self.assertEqual(facts["actual_backward_input_gradient_scope"],"F_module_grad_input_only")

    def test_full_search_and_zero_refit_are_distinct_from_partial_screen(self):
        condition=self.condition(); n50,n60=9960,12063
        plan=dict(task_contract=dict(primary_split_ranges=dict(A_fit=[0,10519],A_stop=[10519,12622]),refit_scaler_sha256="scaler"))
        updates=sum((b-a+15)//16 for a,b in zip([0,2490,9960],[2490,9960,29880]))
        base=dict(status="COMPLETE_VALID",run_id=condition["run_id"],family="B2",epochs=[.25,1.,3.],seed=92601,
            reused_complete_artifact=False,search_exposures=3*n50,search_updates=updates,selected_epoch=0.,refit_exposures=0,refit_updates=0,
            search_fit_end=10519,refit_fit_end=12622,normalization_fit_end=12622,refit_scaler_sha="scaler")
        search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=3*n50,updates=updates,train_origins=n50)
        refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.],exposures=0,updates=0,train_origins=n60)
        self.h.validate_complete_fit(base,search,refit,condition,plan)
        with self.assertRaises(ValueError): self.h.validate_complete_fit(base,dict(search,completed_epochs=[0.,.25]),refit,condition,plan)

    def test_V43_times_K_and_T_rejection(self):
        for k in (6,11):
            condition=self.condition(k=k)
            prediction=dict(score=dict(n_origins=43),metadata=dict(reused_prediction_cache=False),cost_record=dict(actual_F_streams=43*k,expected_F_streams=43*k))
            api=mock.Mock(return_value=prediction)
            self.h.predict_call(api,condition,{},None,None,None,{},1.)
            self.assertEqual(api.call_args.args[2],"V_select")
            with self.assertRaises(ValueError): self.h.predict_call(api,condition,{},None,None,None,{},1.,split="T_final")

    def test_bare_default_imports_no_science_and_writes_nothing(self):
        before=set(sys.modules); capture=io.StringIO()
        with contextlib.redirect_stdout(capture): self.assertEqual(self.h.main([]),0)
        value=json.loads(capture.getvalue())
        self.assertEqual(value["files_written"],0); self.assertEqual(value["registered_conditions"],6)
        self.assertFalse(value["numpy_imported"]); self.assertFalse(value["torch_imported"]); self.assertFalse(value["psutil_imported"])
        self.assertFalse({"numpy","torch","psutil"}&(set(sys.modules)-before))


if __name__=="__main__": unittest.main()
