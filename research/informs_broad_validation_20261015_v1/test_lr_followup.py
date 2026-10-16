"""CPU-only LR runner tests: synthetic assets, mocked fit/F and temporary registry."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_contract as data
import fit_base
import lr_followup as lr
from resources import ResourceBusy


class Lease:
    deadline=10**12
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def check(self):pass


class LRFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="informs_lr_");self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.public=root/"public";self.cache=root/"private";self.assets=root/"assets"
        self.public.mkdir();self.cache.mkdir();self.assets.mkdir()
        self.task=dict(name="electricity",physical=np.random.default_rng(8).normal(size=(1400,321)).astype(np.float32),
            observed=np.ones((1400,321),bool),columns=[str(i) for i in range(321)],L=512,H=48,
            data_sha256="synthetic_data_only",array_sha256={"physical":"synthetic_array"})
        source_sha,files=fit_base._source_contract();self.before_core=(source_sha,files)
        self.rows=[]
        bounds=data.split_ranges(self.task)
        for parent,family in zip(lr.PARENTS,("B0","B2")):
            row=dict(experiment_id=parent,dataset="electricity",panel="",fold="primary",block="E1",family=family,
                num_channels="321",latent_k="81",seed="92601",length="512",horizon="48",requirement="MANDATORY_MINIMUM",
                execution_status="PARTIAL_SCREEN025",training_status="LIMITED_SCREEN",validity_status="PENDING_INDEPENDENT_VALIDATION",
                scientific_status="UNASSESSED",publication_status="NOT_PUBLISHED",reason="synthetic parent",planned_epoch_endpoints="0.25",resume_command="")
            self.rows.append(row)
            spec=dict(run_id=parent,dataset="electricity",family=family,latent_k=81,seed=92601,fold="primary",epochs=[.25],lr=.001,
                device="cuda",microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,
                lora_rank=8,lora_alpha=16,stage_label="registered_initial_breadth")
            folder=self.public/"runs"/parent;folder.mkdir(parents=True)
            artifact=self.cache/"runs"/parent/"selected_bundle.pt";artifact.parent.mkdir(parents=True);artifact.write_bytes(("SYNTHETIC_"+family).encode())
            source=dict(source_sha=source_sha,files=files,spec=spec,data_sha=self.task["data_sha256"],
                        model_sha="synthetic_frozen_state",config_sha=lr._config_sha(spec,self.task))
            (folder/"SOURCE_CONTRACT.json").write_text(json.dumps(source))
            self.scalers={}
            for stage,end in (("search",bounds["A_fit"][1]),("refit",bounds["A_stop"][1])):
                mean,std,_=data.fit_statistics(self.task,end)
                scaler=fit_base.scaler_fingerprint(mean,std);self.scalers[stage]=scaler
                path=folder/stage/"RECEIPT.json";path.parent.mkdir()
                path.write_text(json.dumps(dict(contract=dict(scaler_sha=scaler,initialization_sha=family+"_"+stage+"_same_init"))))
            receipt=dict(status="COMPLETE_VALID",run_id=parent,family=family,selected_epoch=0.,source_sha=source_sha,
                search_fit_end=bounds["A_fit"][1],refit_fit_end=bounds["A_stop"][1],normalization_fit_end=bounds["A_stop"][1],
                refit_scaler_sha=self.scalers["refit"],artifact_cache_relative_path=artifact.relative_to(self.cache).as_posix(),artifact_sha256=data.sha256_file(artifact))
            (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(receipt))
        with (self.public/"EXPERIMENT_REGISTRY.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
        foundation=self.assets/".cache/huggingface/models--amazon--chronos-bolt-small/snapshots"/lr.campaign.SMALL_REV
        foundation.mkdir(parents=True)
        (foundation/"config.json").write_text("{}")
        (foundation/"model.safetensors").write_bytes(b"SYNTHETIC_TEST_ASSET_NOT_MODEL")
        self.patches=[patch.multiple(lr,HERE=self.public,CACHE=self.cache,ASSET_ROOT=self.assets),
            patch.multiple(lr.campaign,HERE=self.public,CACHE=self.cache),patch.object(lr.data,"load_task",return_value=self.task)]
        for item in self.patches:item.start();self.addCleanup(item.stop)

    def test_plan_copies_all_spec_except_lr_epochs_id_and_reads_no_V(self):
        _,jobs,proposals=lr.build_plan()
        self.assertEqual(len(jobs),2);self.assertEqual(len(proposals),2)
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION*")))
        for job in jobs:
            proof=job["provenance"]
            old,new=proof["original_spec"],job["spec"]
            self.assertEqual(new["lr"],.0001);self.assertEqual(new["epochs"],[.25,1.])
            self.assertEqual(new["seed"],92601);self.assertEqual(new["latent_k"],81)
            self.assertEqual({k:v for k,v in old.items() if k not in lr.SPEC_DIFFERENCES},{k:v for k,v in new.items() if k not in lr.SPEC_DIFFERENCES})
            self.assertNotEqual(proof["parent_config_sha256"],proof["new_config_sha256"])
            self.assertTrue(proof["fresh_A50_search"]);self.assertTrue(proof["fresh_A60_refit"])
            self.assertFalse(proof["parent_weights_optimizer_RNG_reused"])
            self.assertFalse(proof["V_used_for_LR_choice"])
        self.assertEqual(fit_base._source_contract(),self.before_core)

    def test_unsupported_parent_and_wrong_epoch_schedule_refuse_before_registration(self):
        for ids,epochs in ((["primary_electricity_E1_B4_k81_s92601"],None),([lr.PARENTS[0]],[.25,1,3])):
            with self.subTest(ids=ids),patch.object(lr.registry_extension,"register") as register,patch.object(lr,"GPUReservation") as gpu:
                with self.assertRaises(ValueError):lr.run(ids,epochs)
                register.assert_not_called();gpu.assert_not_called()

    def test_locked_T_refuses_before_any_registry_or_GPU_mutation(self):
        (self.public/"SELECTION_LOCK.json").write_text(json.dumps(dict(locked=True)))
        before=(self.public/"EXPERIMENT_REGISTRY.csv").read_bytes()
        with patch.object(lr,"GPUReservation") as gpu:
            with self.assertRaisesRegex(ValueError,"selection"):lr.run()
            gpu.assert_not_called()
        self.assertEqual((self.public/"EXPERIMENT_REGISTRY.csv").read_bytes(),before)
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION*")))

    def test_changed_parent_checkpoint_or_scientific_source_refused(self):
        parent=lr.PARENTS[0]
        artifact=self.cache/"runs"/parent/"selected_bundle.pt";original=artifact.read_bytes();artifact.write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError,"checkpoint"):lr.build_plan([parent])
        artifact.write_bytes(original)
        path=self.public/"runs"/parent/"SOURCE_CONTRACT.json";source=json.loads(path.read_text())
        source["files"]["training.py"]="changed";path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError,"source"):lr.build_plan([parent])
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION*")))

    def fake_runtime(self,fail_first=False,cached_first=False):
        events=[];outer=self
        class Foundation:
            initial_hash="synthetic_frozen_state"
            streams=0
        foundation=Foundation()
        class Pool:
            def __init__(self,*args,**kwargs):pass
            def get_foundation(self,*args):events.append("foundation_load");return foundation
            def close(self):events.append("close")
            def predict(self,run_id):
                events.append("V:"+run_id)
                receipt=json.loads((outer.public/"runs"/run_id/"BASE_FIT_RECEIPT.json").read_text())
                if not cached_first:foundation.streams+=162
                return dict(metadata=dict(run_id=run_id,artifact_sha256=receipt["artifact_sha256"],data_sha256=outer.task["data_sha256"],
                    reused_prediction_cache=cached_first),cost_record=dict(actual_F_streams=162),score=dict(mse=.2,mae=.1))
        real_register=lr.registry_extension.register
        def register(*args):events.append("register");return real_register(*args)
        def gpu(*args,**kwargs):events.append("GPU_lease");outer.assertEqual(kwargs["reserve_seconds"],9000);return Lease()
        def fit(spec,task,cache,public,f,deadline):
            events.append("fit:"+spec["run_id"])
            if fail_first and spec["family"]=="B0":raise ValueError("synthetic per-route failure")
            folder=public/"runs"/spec["run_id"]
            (folder/"SOURCE_CONTRACT.json").write_text(json.dumps(dict(spec=spec,config_sha=lr._config_sha(spec,task))))
            for stage in ("search","refit"):
                path=folder/stage/"RECEIPT.json";path.parent.mkdir()
                path.write_text(json.dumps(dict(contract=dict(initialization_sha=spec["family"]+"_"+stage+"_same_init",scaler_sha=outer.scalers[stage]))))
            receipt=dict(status="COMPLETE_VALID",training_status="LIMITED_SCREEN",refit_scaler_sha=outer.scalers["refit"],
                artifact_sha256="synthetic_new_artifact",selected_epoch=.25,search_updates=20,refit_updates=10)
            (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(receipt));return receipt
        return events,[patch.object(lr.registry_extension,"register",side_effect=register),patch.object(lr,"GPUReservation",side_effect=gpu),
            patch.object(lr,"ParentPool",Pool),patch.object(lr.fit_base,"fit_base",side_effect=fit),patch.object(lr,"check_storage_budget")]

    def exercise(self,**kwargs):
        events,patches=self.fake_runtime(**kwargs)
        for item in patches:item.start();self.addCleanup(item.stop)
        before=[copy.deepcopy(row) for row in self.rows]
        result=lr.run()
        self.assertLess(events.index("GPU_lease"),events.index("register"))
        self.assertLess(events.index("register"),events.index("foundation_load"))
        rows={row["experiment_id"]:row for row in lr.campaign.registry_rows()}
        for row in before:self.assertEqual(rows[row["experiment_id"]],row)
        self.assertEqual(fit_base._source_contract(),self.before_core)
        return result,events,rows

    def test_registration_precedes_mocked_GPU_and_both_routes_get_actual_new_V_path(self):
        result,events,rows=self.exercise()
        self.assertEqual(result["status"],"PAIR_FINISHED_REQUIRES_INDEPENDENT_AUDIT")
        self.assertEqual(len(result["results"]),2)
        self.assertTrue(all(row["V_prediction_executed_now"] and row["actual_new_V_F_streams"]==162 for row in result["results"]))
        self.assertTrue(all(rows[parent+"__lr0001_e1"]["execution_status"]=="PARTIAL_SCREEN_LT3" for parent in lr.PARENTS))
        extensions=list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION_*.json"));self.assertEqual(len(extensions),1)
        self.assertTrue(json.loads(extensions[0].read_text())["before_execution"])
        self.assertFalse(result["M_E2_parents_replaced"]);self.assertFalse(result["T_accessed"])

    def test_one_failure_does_not_skip_second_matched_route(self):
        result,events,rows=self.exercise(fail_first=True)
        self.assertEqual([row["status"] for row in result["results"]],["EXECUTION_FAILED","PARTIAL_SCREEN_LT3"])
        self.assertIn("V:"+lr.PARENTS[1]+"__lr0001_e1",events)

    def test_new_route_cache_only_result_is_refused(self):
        result,events,rows=self.exercise(cached_first=True)
        self.assertTrue(all(row["status"]=="EXECUTION_FAILED" for row in result["results"]))
        self.assertTrue(all(rows[parent+"__lr0001_e1"]["execution_status"]=="EXECUTION_FAILED" for parent in lr.PARENTS))

    def test_active_GPU_owner_returns_deferred_without_waiting_or_loading_model(self):
        before=(self.public/"EXPERIMENT_REGISTRY.csv").read_bytes()
        with patch.object(lr,"GPUReservation",side_effect=ResourceBusy("synthetic active owner")) as gpu, \
             patch.object(lr,"ParentPool") as pool:
            result=lr.run()
        gpu.assert_called_once();pool.assert_not_called()
        self.assertEqual(result["status"],"DEFERRED_RESOURCE_OR_ACTIVE_OWNER")
        self.assertEqual((self.public/"EXPERIMENT_REGISTRY.csv").read_bytes(),before)
        self.assertFalse(list(self.public.glob("EXPERIMENT_REGISTRY_EXTENSION*")))

    def test_V_deadline_preserves_fit_and_never_leaves_running_status(self):
        events,patches=self.fake_runtime()
        for item in patches:item.start();self.addCleanup(item.stop)
        with patch.object(lr.ParentPool,"predict",side_effect=TimeoutError("synthetic V deadline")):
            result=lr.run()
        self.assertEqual(result["status"],"DEFERRED_RESOURCE")
        rows={row["experiment_id"]:row for row in lr.campaign.registry_rows()}
        self.assertEqual(rows[lr.PARENTS[0]+"__lr0001_e1"]["execution_status"],"DEFERRED_RESOURCE")
        self.assertEqual(rows[lr.PARENTS[1]+"__lr0001_e1"]["execution_status"],"PENDING_QUEUE")
        self.assertTrue((self.public/"runs"/(lr.PARENTS[0]+"__lr0001_e1")/"BASE_FIT_RECEIPT.json").is_file())


if __name__=="__main__":unittest.main()
