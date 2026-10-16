"""Actual CPU G fits/inference and temporary registries; no campaign mutation."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

import control_deepen as cd
import fit_base
import fit_control
import prediction_cache


SOURCE_ROOT = Path(__file__).resolve().parent


class Lease:
    deadline = 10**12
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def check(self): pass


class ControlDeepenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory(prefix="informs_control_deepen_")
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name); self.public=root/"public"; self.cache=root/"private"
        self.public.mkdir(); self.cache.mkdir()
        self.core_before=fit_base._source_contract()
        self.control_source=fit_control.source_contract()
        self.task=dict(name="beijing",physical=np.cumsum(np.random.default_rng(52).normal(size=(240,3)),axis=0).astype(np.float32),
            observed=np.ones((240,3),bool),columns=["a","b","c"],L=64,H=4,data_sha256="toy-allowed-prefix")
        self.parent="primary_beijing_E2_G_REFIT_APLUSM_k0_s92601_G"
        self.spec=dict(run_id=self.parent,family="DIRECT_G",role="AplusM_control",seed=92601,fold="primary",
            epochs=[.25],lr=.001,device="cpu",microbatch=4,effective_batch=16)
        self.parent_receipt=fit_control.fit_control(self.spec,self.task,self.cache,self.public,None,None)
        self.row=dict(experiment_id=self.parent,dataset="beijing",panel="",fold="primary",block="E2",family="G_REFIT_APLUSM",
            num_channels="3",latent_k="0",seed="92601",length="64",horizon="4",requirement="MANDATORY_MINIMUM",
            execution_status="COMPLETE_VALID",training_status="LIMITED_SCREEN",validity_status="PENDING_INDEPENDENT_VALIDATION",
            scientific_status="UNASSESSED",publication_status="NOT_PUBLISHED",reason="toy",planned_epoch_endpoints="0.25|1|3",resume_command="")
        self.write_registry([self.row])
        self.patches=[patch.multiple(cd,HERE=self.public,CACHE=self.cache),patch.object(cd,"_source_contract",return_value=self.control_source),
                      patch.object(cd.data,"load_task",return_value=self.task)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)

    def tearDown(self):
        self.assertEqual(fit_base._source_contract(),self.core_before)

    def write_registry(self,rows):
        with (self.public/"EXPERIMENT_REGISTRY.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.row)); writer.writeheader();writer.writerows(rows)

    def test_plan_is_read_only_and_preserves_all_hyperparameters_except_id_epochs(self):
        before=(self.public/"EXPERIMENT_REGISTRY.csv").read_bytes()
        plan=cd.build_plan(["beijing"])
        self.assertEqual(plan["unavailable"],[]);self.assertEqual(len(plan["jobs"]),1)
        job=plan["jobs"][0]
        self.assertEqual(job["run_id"],self.parent+"__e3");self.assertEqual(job["spec"]["epochs"],[.25,1.,3.])
        self.assertEqual({k:v for k,v in job["spec"].items() if k not in {"epochs","run_id"}},
                         {k:v for k,v in self.spec.items() if k not in {"epochs","run_id"}})
        self.assertEqual(job["provenance"]["scaler_sha"],self.parent_receipt["scaler_sha"])
        self.assertFalse((self.public/"runs"/job["run_id"]).exists())
        self.assertEqual((self.public/"EXPERIMENT_REGISTRY.csv").read_bytes(),before)
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION_*")))

    def test_module_import_and_preflight_dependencies_do_not_import_torch(self):
        code="import control_deepen,sys; assert 'torch' not in sys.modules; print('TORCH_FREE_IMPORT')"
        result=subprocess.run([sys.executable,"-c",code],cwd=SOURCE_ROOT,capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn("TORCH_FREE_IMPORT",result.stdout)

    def test_actual_cpu_fresh_fit_and_actual_V_inference_F0_preregister_first(self):
        original_parent=(self.public/"runs"/self.parent/"CONTROL_FIT_RECEIPT.json").read_bytes()
        events=[]; real_register=cd.registry_extension.register; real_fit=fit_control.fit_control
        real_predict=prediction_cache.predict_parent
        def register(*args): events.append("register"); return real_register(*args)
        def fit(*args):
            events.append("fit")
            self.assertEqual(args[4],None)
            self.assertIn(self.parent+"__e3",{r["experiment_id"] for r in cd._registry()})
            return real_fit(*args)
        def predict(*args,**kwargs):
            events.append("V_actual")
            self.assertEqual(args[5],None)
            return real_predict(*args,**kwargs)
        def reserve(*args,**kwargs):
            events.append("lease");self.assertEqual(kwargs["reserve_seconds"],9000);self.assertEqual(kwargs["total_gpu_seconds"],43200)
            self.assertEqual(kwargs["purpose"],"training");return Lease()
        with patch.object(cd,"GPUReservation",side_effect=reserve),patch.object(cd,"check_storage_budget"),\
             patch.object(cd.registry_extension,"register",side_effect=register),patch.object(fit_control,"fit_control",side_effect=fit),\
             patch.object(prediction_cache,"predict_parent",side_effect=predict):
            result=cd.run(["beijing"])
        self.assertEqual(result["status"],"FINISHED_REQUIRES_INDEPENDENT_AUDIT",result)
        self.assertEqual(events,["lease","register","fit","V_actual"])
        row=result["results"][0];self.assertTrue(row["V_prediction_executed_now"])
        self.assertEqual(row["actual_V_F_streams"],0);self.assertEqual(row["training_F_streams"],0)
        self.assertGreater(row["updates"],self.parent_receipt["updates"])
        self.assertFalse(row["convergence_claimed"]);self.assertFalse(row["T_accessed"])
        target=json.loads((self.public/"runs"/(self.parent+"__e3")/"CONTROL_FIT_RECEIPT.json").read_text())
        self.assertEqual(target["initial_model_state_sha"],self.parent_receipt["initial_model_state_sha"])
        self.assertEqual(target["epochs"],[.25,1.,3.]);self.assertEqual(target["control_fit_end"],168)
        self.assertEqual(target["normalization_fit_end"],144)
        self.assertEqual((self.public/"runs"/self.parent/"CONTROL_FIT_RECEIPT.json").read_bytes(),original_parent)
        self.assertEqual(cd._registry()[0],self.row)
        # A fully matching completed target may be reused, with honest inference/cache status.
        with patch.object(cd,"GPUReservation",return_value=Lease()),patch.object(cd,"check_storage_budget"):
            repeated=cd.run(["beijing"])
        self.assertEqual(repeated["results"][0]["status"],"COMPLETE_VALID")
        self.assertTrue(repeated["results"][0]["fit_artifact_reused"])
        self.assertFalse(repeated["results"][0]["V_prediction_executed_now"])

    def test_partial_new_target_requires_explicit_resume_and_same_contract(self):
        plan=cd.build_plan(["beijing"]);job=plan["jobs"][0]
        partial=fit_control.fit_control(job["spec"],self.task,self.cache,self.public,None,time.time()-1)
        self.assertEqual(partial["status"],"BUDGET_LIMIT")
        path=self.cache/"runs"/job["run_id"]/"CONTROL_RUN_CONTRACT.json";before=path.read_bytes()
        refused=cd.build_plan(["beijing"])
        self.assertEqual(refused["jobs"],[]);self.assertIn("Explicit --resume",refused["unavailable"][0]["reason"])
        self.assertEqual(path.read_bytes(),before)
        resumed=cd.build_plan(["beijing"],resume=True)
        self.assertEqual(len(resumed["jobs"]),1);self.assertTrue(resumed["jobs"][0]["target_partial"])
        source=self.public/"runs"/job["run_id"]/"CONTROL_SOURCE_CONTRACT.json"
        value=json.loads(source.read_text());value["spec"]["lr"]=.1;source.write_text(json.dumps(value))
        self.assertEqual(cd.build_plan(["beijing"],resume=True)["jobs"],[])

    def test_changed_parent_checkpoint_refuses_without_overwrite(self):
        artifact=self.cache/self.parent_receipt["artifact_cache_relative_path"]
        artifact.write_bytes(b"CHANGED")
        plan=cd.build_plan(["beijing"])
        self.assertEqual(plan["jobs"],[]);self.assertIn("artifact",plan["unavailable"][0]["reason"])
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION_*")))

    def test_locked_T_rejects_before_lease_or_registration(self):
        (self.public/"SELECTION_LOCK.json").write_text(json.dumps(dict(locked=True)))
        before=(self.public/"EXPERIMENT_REGISTRY.csv").read_bytes()
        with patch.object(cd,"GPUReservation") as gpu,patch.object(cd.registry_extension,"register") as register:
            with self.assertRaisesRegex(ValueError,"selection"):cd.run(["beijing"])
            gpu.assert_not_called();register.assert_not_called()
        self.assertEqual((self.public/"EXPERIMENT_REGISTRY.csv").read_bytes(),before)

    def test_missing_parent_does_not_drop_independent_ready_dataset(self):
        missing=dict(self.row,experiment_id="primary_jena_E2_G_REFIT_APLUSM_k0_s92601_G",dataset="jena")
        self.write_registry([self.row,missing])
        plan=cd.build_plan(["jena","beijing"])
        self.assertEqual(len(plan["jobs"]),1);self.assertEqual(plan["jobs"][0]["dataset"],"beijing")
        self.assertEqual(plan["unavailable"][0]["status"],"PARENT_NOT_COMPLETE")

    def test_individual_fit_failure_continues_other_dataset(self):
        second="primary_jena_E2_G_REFIT_APLUSM_k0_s92601_G"
        task2=dict(self.task,name="jena");spec2=dict(self.spec,run_id=second)
        fit_control.fit_control(spec2,task2,self.cache,self.public,None,None)
        self.write_registry([self.row,dict(self.row,experiment_id=second,dataset="jena")])
        original=fit_control.fit_control
        def fit(spec,*args):
            if spec["run_id"].startswith("primary_beijing"):raise ValueError("synthetic independent failure")
            return original(spec,*args)
        with patch.object(cd.data,"load_task",side_effect=lambda ds:self.task if ds=="beijing" else task2),\
             patch.object(cd,"GPUReservation",return_value=Lease()),patch.object(cd,"check_storage_budget"),\
             patch.object(fit_control,"fit_control",side_effect=fit):
            result=cd.run(["beijing","jena"])
        self.assertEqual([row["status"] for row in result["results"]],["EXECUTION_FAILED","COMPLETE_VALID"])
        self.assertEqual(result["status"],"PARTIAL_MISSING_OR_FAILED_ROUTES")

    def test_unknown_dataset_and_mixed_all4_rejected(self):
        for datasets in (["robin"],["all4","jena"],["beijing","beijing"]):
            with self.subTest(datasets=datasets),self.assertRaises(ValueError):cd.build_plan(datasets)


if __name__=="__main__":unittest.main()
