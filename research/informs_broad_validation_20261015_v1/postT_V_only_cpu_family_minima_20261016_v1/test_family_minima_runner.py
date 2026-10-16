"""Stdlib synthetic definitions only; ROOT runs after precision costs are quiet."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tiny_family_minima",HERE/"family_minima_runner.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FamilyMinimaTests(unittest.TestCase):
    def setUp(self):
        self.h=load()
        self.upstream={}
        for entry in self.h.TARGETS:
            spec=self.h.original_spec(*entry)
            self.upstream[spec["run_id"]]=dict(spec=spec)
        self.rows=self.h.conditions(self.upstream)

    def task(self,dataset):
        a50,a60,c=(10482,12579,21) if dataset=="jena" else (10519,12622,11)
        return dict(columns=["channel"+str(i) for i in range(c)],data_sha256=dataset+"-data",array_sha256={"physical":"opaque-array"},
            search_scaler_sha256=dataset+"-A50",refit_scaler_sha256=dataset+"-A60",
            primary_split_ranges=dict(A_fit=[0,a50],A_stop=[a50,a60]))

    def science(self,condition):
        k=condition["effective_F_width"]
        result=dict(probe_updates=3,F_counter_delta=dict(streams=48*k,row_calls=12,physical_streams=48*k,physical_calls=12),
            full_A50_legal_origins=self.task(condition["dataset"])["primary_split_ranges"]["A_fit"][1]-559,
            full_A50_loss_normalization_preserved=True,probe_state_discarded=True,probe_state_reused_for_fit=False,
            saved_model_or_optimizer=False,completed_fit=False,V_scoring=False,actual_parameter_gradients_finite=True,
            actual_parameter_gradient_nonzero=True,no_foundation_parameter_gradients=True,base_hash_restored_after_close=True,
            segment=dict(status="SEGMENT_COMPLETE",updates=3,exposures=48,optimizer_updates_this_segment=3,k_exposures={}))
        if condition["family"]=="LORA":
            result.update(gradient_scope="LORA_adapter_parameters_only",adapter_parameter_gradient_absmax=[0.,.001],adapter_hooks_removed_after_close=True)
        else:
            result.update(gradient_scope="F_module_grad_input_only",
                foundation_input_requires_grad_events=[dict(shape=[4,512,k],requires_grad=True,grad_enabled=True) for _ in range(12)],
                actual_backward_input_gradient_events=[dict(shape=[4,512,k],finite=True,absmax=.001 if i==0 else 0.) for i in range(12)])
        return result

    def test_four_exact_original_recipes_preserve_LoRA_C21_and_ADAPTS_alias(self):
        expected=["postT_cpu_jena_E1_B4_k6_s92601_family_v1","postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1",
            "postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1","postT_cpu_beijing_E1_B4_k11_s92601_family_v1"]
        self.assertEqual([row["run_id"] for row in self.rows],expected)
        self.assertEqual([row["effective_F_width"] for row in self.rows],[6,6,21,11])
        for row in self.rows:
            original=self.upstream[row["upstream_registered_id"]]["spec"]
            final=row["spec"]
            for key,value in original.items():
                if key not in {"run_id","device","epochs","stage_label"}:self.assertEqual(final[key],value)
            self.assertEqual(final["epochs"],[.25,1.,3.]);self.assertEqual(final["device"],"cpu")
            self.assertEqual((final["microbatch"],final["effective_batch"],final["F_row_microbatch"]),(4,16,128))
            self.assertFalse(final["checkpoint_inputs"])
            self.assertFalse(final["provenance"]["old_weights_optimizer_RNG_or_state_reused"])
        self.assertEqual((self.rows[1]["registered_family"],self.rows[1]["family"]),("ADAPTS_BOLT_PORT","ADAPTS"))
        lora=self.rows[2]
        self.assertEqual((lora["registered_latent_k"],lora["latent_k"],lora["spec"]["lr"]),(0,21,.0001))
        self.assertEqual((lora["spec"]["lora_rank"],lora["spec"]["lora_alpha"]),(8,16))
        with self.assertRaises(ValueError):self.h.choose(dict(conditions=self.rows),"unregistered")

    def test_original_recipe_source_data_and_axis_mutations_reject(self):
        h=self.h
        for condition in self.rows:
            task=self.task(condition["dataset"])
            spec=self.upstream[condition["upstream_registered_id"]]["spec"]
            sources={name:name+"-sha" for name in h.SOURCES}
            axes=dict(L=512,H=48,C=len(task["columns"]),columns=task["columns"],fold="primary",source_array_fingerprints=task["array_sha256"])
            record=dict(spec=spec,files=sources,source_sha=h.fingerprint(sources),config_sha=h.fingerprint(dict(spec=spec,axes=axes)),
                data_sha=task["data_sha256"],model_sha="foundation")
            h.validate_original(record,condition,sources,task,"foundation")
            for key,value in (("data_sha","other"),("model_sha","other"),("config_sha","other"),("files",{})):
                with self.subTest(run=condition["run_id"],key=key),self.assertRaises(ValueError):
                    h.validate_original(dict(record,**{key:value}),condition,sources,task,"foundation")
        wrong=copy.deepcopy(self.upstream)
        wrong[self.rows[2]["upstream_registered_id"]]["spec"]["latent_k"]=0
        with self.assertRaises(ValueError):h.conditions(wrong)
        wrong=copy.deepcopy(self.upstream)
        wrong[self.rows[0]["upstream_registered_id"]]["spec"]["lr"]=.01
        with self.assertRaises(ValueError):h.conditions(wrong)

    def test_B4_and_ADAPTS_require_F_module_backward_events_not_direct_z_flags(self):
        for condition in [row for row in self.rows if row["family"]!="LORA"]:
            science=self.science(condition);task=self.task(condition["dataset"])
            self.h.validate_probe_science(science,condition,task)
            for mutation in ("scope","missing","zero","nonfinite","shape","counter","basegrad"):
                wrong=copy.deepcopy(science)
                if mutation=="scope":wrong["gradient_scope"]="z_tensor_hook"
                elif mutation=="missing":wrong.pop("actual_backward_input_gradient_events")
                elif mutation=="zero":
                    for event in wrong["actual_backward_input_gradient_events"]:event["absmax"]=0.
                elif mutation=="nonfinite":wrong["actual_backward_input_gradient_events"][0]["absmax"]=float("nan")
                elif mutation=="shape":wrong["actual_backward_input_gradient_events"][0]["shape"]=[4,512,1]
                elif mutation=="counter":wrong["F_counter_delta"]["physical_calls"]=0
                else:wrong["no_foundation_parameter_gradients"]=False
                with self.subTest(run=condition["run_id"],mutation=mutation),self.assertRaises(ValueError):
                    self.h.validate_probe_science(wrong,condition,task)

    def test_LoRA_uses_adapter_gradient_scope_C21_and_close_restoration(self):
        condition=self.rows[2];task=self.task("jena");science=self.science(condition)
        self.assertEqual(science["F_counter_delta"]["streams"],1008)
        self.h.validate_probe_science(science,condition,task)
        mutations=[dict(adapter_parameter_gradient_absmax=v) for v in ([],[0.,0.],[float("nan")],[True])]
        mutations += [dict(base_hash_restored_after_close=False),dict(adapter_hooks_removed_after_close=False),
            dict(no_foundation_parameter_gradients=False),dict(gradient_scope="F_module_grad_input_only"),
            dict(foundation_input_requires_grad_events=[]),dict(actual_backward_input_gradient_events=[])]
        for patch in mutations:
            with self.subTest(patch=patch),self.assertRaises(ValueError):self.h.validate_probe_science(dict(science,**patch),condition,task)

    def test_actual_probe_proof_binds_one_condition_plan_runner_and_discarded_science(self):
        for condition in self.rows:
            task=self.task(condition["dataset"])
            proof=dict(status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED",mode="probe",run_id=condition["run_id"],
                plan_sha256="plan",runner_sha256="runner",frozen_foundation_unchanged=True,
                original_source_registry_selection_T_pins_preserved=True,cuda_initialized=False,new_T_target_or_scoring=False,
                scientific_result=self.science(condition))
            self.h.validate_probe_proof(proof,condition,task,"plan","runner")
            for key,value in (("run_id","other"),("runner_sha256","other"),("plan_sha256","other"),("scientific_result",None),
                    ("cuda_initialized",True),("frozen_foundation_unchanged",False),("new_T_target_or_scoring",True)):
                with self.subTest(run=condition["run_id"],key=key),self.assertRaises(ValueError):
                    self.h.validate_probe_proof(dict(proof,**{key:value}),condition,task,"plan","runner")

    def test_full_search_and_selected_zero_refit_use_pinned_original_validator(self):
        for condition in (self.rows[0],self.rows[2],self.rows[3]):
            task=self.task(condition["dataset"]);bounds=task["primary_split_ranges"]
            n50,n60=bounds["A_fit"][1]-559,bounds["A_stop"][1]-559
            endpoints=[0]+[math.ceil(epoch*n50) for epoch in (.25,1.,3.)]
            updates=sum(math.ceil((b-a)/16) for a,b in zip(endpoints,endpoints[1:]))
            for epoch in (0.,3.):
                exposures=math.ceil(epoch*n60)
                base=dict(status="COMPLETE_VALID",run_id=condition["run_id"],family=condition["family"],epochs=[.25,1.,3.],seed=92601,
                    selected_epoch=epoch,reused_complete_artifact=False,search_exposures=3*n50,search_updates=updates,
                    refit_exposures=exposures,refit_updates=math.ceil(exposures/16),search_fit_end=bounds["A_fit"][1],
                    refit_fit_end=bounds["A_stop"][1],normalization_fit_end=bounds["A_stop"][1],refit_scaler_sha=task["refit_scaler_sha256"])
                search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=3*n50,updates=updates,train_origins=n50)
                refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),exposures=exposures,updates=math.ceil(exposures/16),train_origins=n60)
                self.h.validate_complete_fit(base,search,refit,condition,task)
                with self.assertRaises(ValueError):self.h.validate_complete_fit(base,dict(search,completed_epochs=[0.,.25,1.]),refit,condition,task)
                with self.assertRaises(ValueError):self.h.validate_complete_fit(dict(base,reused_complete_artifact=True),search,refit,condition,task)

    def test_full_V43_original_predict_API_keeps_effective_width_and_rejects_cache_or_T(self):
        for condition in self.rows:
            width=condition["effective_F_width"]
            value=dict(score=dict(n_origins=43),metadata=dict(reused_prediction_cache=False),
                cost_record=dict(actual_F_streams=43*width,expected_F_streams=43*width))
            api=mock.Mock(return_value=value)
            self.h.predict_call(api,condition,"task","cache","public","F","receipt",100.)
            api.assert_called_once_with(condition["run_id"],"task","V_select","cache","public","F",receipt="receipt",fold="primary",deadline=100.)
            with self.assertRaises(ValueError):self.h.predict_call(api,condition,"t","c","p","F","r",100.,split="T_final")
            for bad in (dict(value,metadata=dict(reused_prediction_cache=True)),dict(value,cost_record=dict(actual_F_streams=0,expected_F_streams=0))):
                with self.assertRaises(ValueError):self.h.predict_call(mock.Mock(return_value=bad),condition,"t","c","p","F","r",100.)

    def test_existing_runs_claims_unsafe_paths_and_changed_source_fail_closed(self):
        run_id=self.rows[0]["run_id"]
        for kind in ("runs","predictions","claims"):
            with tempfile.TemporaryDirectory() as td:
                public,cache=Path(td)/"public",Path(td)/"cache"
                folder=public/self.h.NAME/kind;folder.mkdir(parents=True)
                if kind=="claims":(folder/("fit_"+run_id+".json")).write_text("{}")
                else:(folder/run_id).mkdir()
                with self.assertRaises(FileExistsError):self.h.require_fresh(public,cache,"fit",run_id)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/"tiny.py";path.write_bytes(b"tiny")
            pins={"tiny.py":self.h.sha(path)};self.h.check_pins(root,pins)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):self.h.check_pins(root,pins)
            with self.assertRaises(ValueError):self.h.relative(root,"../outside")

    def test_bare_metadata_default_does_not_import_science_or_write_outputs(self):
        before={name:name in sys.modules for name in ("numpy","torch","psutil")}
        with mock.patch.object(self.h,"proposal",side_effect=AssertionError("no roots")), \
             mock.patch.object(self.h,"metadata",side_effect=AssertionError("no sealed roots")), \
             mock.patch.object(self.h,"execute",side_effect=AssertionError("no execute")), \
             mock.patch.object(self.h,"seal_plan",side_effect=AssertionError("no seal")),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual((result["registered_conditions"],result["files_written"],result["actual_Task_model_array_training_or_child_operations"]),(4,0,0))
        self.assertEqual(before,{name:name in sys.modules for name in before})


if __name__=="__main__":unittest.main(verbosity=2)
