"""Tiny synthetic metadata/API fixtures; no real scientific imports or arrays."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
HELPER = HERE/"verify_multik_outputs.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class MultiKVerifierTests(unittest.TestCase):
    def setUp(self):
        self.h = load("synthetic_multik_verifier",HELPER)
        self.r = self.h.load_runner()
        source = {key:"tiny-source-"+key for key in self.h.ORDERED_SOURCES}
        source["campaign.py"] = "tiny-campaign"
        fits, outputs = self.r.recipe_manifest(dict(synthetic=True),source)
        contract = dict(shape=[21038,11],L=512,H=48,columns=["c"+str(i) for i in range(11)],array_sha256={"physical":"tiny","observed":"tiny-mask"},
            data_sha256="tiny-data",search_scaler_sha256="tiny-A50-scaler",refit_scaler_sha256="tiny-A60-scaler",
            legal_origin_counts_primary=dict(V_select=43),primary_split_ranges=dict(A_fit=[0,10519],A_stop=[10519,12622]))
        self.plan = dict(fits=fits,outputs=outputs,source39=source,task_contract=contract,foundation_tensor_sha256="tiny-foundation",
            deadline_unix=time.time()+50,max_RSS_bytes=3221225472,storage_limit_bytes=32212254720,stage="POST_T_V_ONLY_DEVELOPMENT",
            model_asset_relative_path="tiny_model",postT_registry_relative_path="REGISTRY.csv",postT_registry_sha256="pin")

    def parent(self, condition):
        r, plan = self.r,self.plan
        row = next(row for row in plan["fits"] if row["run_id"] == condition["parent_run_id"])
        c = plan["task_contract"]
        n, nr = c["primary_split_ranges"]["A_fit"][1]-559,c["primary_split_ranges"]["A_stop"][1]-559
        files = {key:plan["source39"][key] for key in self.h.FIT_SOURCES}
        axes = dict(L=512,H=48,C=11,columns=c["columns"],fold="primary",source_array_fingerprints=c["array_sha256"])
        shared = dict(source_sha=r.fingerprint(files),config_sha=r.fingerprint(dict(spec=row["spec"],axes=axes)),data_sha=c["data_sha256"],model_sha=plan["foundation_tensor_sha256"])
        source = dict(shared,files=files,spec=copy.deepcopy(row["spec"]))
        base = dict(status="COMPLETE_VALID",run_id=row["run_id"],family="B0",reused_complete_artifact=False,epochs=[.25,1.,3.],seed=92601,
            selected_epoch=0.,search_updates=2000,search_exposures=3*n,refit_updates=0,refit_exposures=0,search_train_origins=n,
            search_fit_end=c["primary_split_ranges"]["A_fit"][1],refit_fit_end=c["primary_split_ranges"]["A_stop"][1],normalization_fit_end=c["primary_split_ranges"]["A_stop"][1],
            refit_scaler_sha=c["refit_scaler_sha256"],source_sha=shared["source_sha"],artifact_cache_relative_path=f"runs/{row['run_id']}/selected_bundle.pt",artifact_sha256="tiny-artifact",model_state_sha="tiny-state")
        search = dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],actual_epochs=3.,exposures=3*n,updates=2000,train_origins=n,
                      contract=dict(shared,scaler_sha=c["search_scaler_sha256"]))
        refit = dict(status="SEGMENT_COMPLETE",selected_epoch=0.,actual_epochs=0.,exposures=0,updates=0,train_origins=nr,
                     contract=dict(shared,scaler_sha=c["refit_scaler_sha256"]))
        return base,source,search,refit

    def nested_contract(self):
        return dict(source_sha=self.r.fingerprint({name:self.plan["source39"][name] for name in self.h.ORDERED_SOURCES}),data_sha="tiny-data",fit_end=12622,artifact_sha256="tiny-artifact")

    def manifest(self, condition, parent):
        h,r,plan = self.h,self.r,self.plan
        c,k = plan["task_contract"],condition["latent_k"]
        run_id = condition["run_id"]
        meta = dict(data_sha256=c["data_sha256"],fold="primary",parent_fit_end=12622)
        if condition["kind"] == "independent":
            files = {name:plan["source39"][name] for name in h.PREDICTION_SOURCES}
            spec = next(row["spec"] for row in plan["fits"] if row["run_id"]==condition["parent_run_id"])
            source = r.fingerprint(dict(deployment_sources=files,training_source_sha=parent["source_sha"]))
            config = r.fingerprint(dict(parent_spec=spec,dense=False,fold="primary",role="base_parent"))
            model = r.fingerprint(dict(artifact_sha=parent["artifact_sha256"],model_state_sha=parent["model_state_sha"],normalization=parent["refit_scaler_sha"]))
            meta.update(family="B0",artifact_sha256=parent["artifact_sha256"],canonical_scaler_fit_end=12622,parent_normalization_fit_end=12622,
                        source_files=files,training_source_sha=parent["source_sha"],dense=False,role="base_parent")
            archive = f"predictions/{run_id}/V_select.npz"
            stats = f"predictions/{run_id}/V_select/PUBLIC_ERROR_STATS.csv.gz"
        else:
            nested = condition["kind"] == "nested"
            parent_id = "nested_pca" if nested else condition["parent_run_id"]
            source = r.fingerprint(dict(sources={name:plan["source39"][name] for name in h.ORDERED_SOURCES},training_source_sha=parent["source_sha"]))
            config = r.fingerprint(dict(parent_run_id=parent_id,variant=condition["variant"],latent_k=k,fold="primary"))
            model = "tiny-nested-model" if nested else r.fingerprint(dict(artifact_sha=parent["artifact_sha256"],state_sha=parent["model_state_sha"]))
            meta.update(parent_run_id=parent_id,parent_artifact_sha256=parent["artifact_sha256"],variant=condition["variant"],family="NESTED_PCA" if nested else "B0",
                inference_latent_k=k,stored_latent_k=11,foundation_sha=plan["foundation_tensor_sha256"],actual_F_streams=43*k,T_final_accessed=False,
                posthoc_prefix_was_trained=condition["variant"]=="shared" or condition["variant"] in {"posthoc","max_only"} and k==11)
            archive = f"ordered_predictions/{run_id}/V_select.npz"
            stats = f"ordered_predictions/{run_id}/PUBLIC_ERROR_STATS.csv.gz"
        score = dict(run_id=run_id,split="V_select",cohort="beijing:primary:V_select:stride48",mse=.3,mae=.2,n_origins=43,horizon=48,num_channels=11)
        shapes = dict(pred_normalized=[43,48,11],target_normalized=[43,48,11],observed=[43,48,11],origins=[43],channel_ids=[11])
        return dict(run_id=run_id,split="V_select",cohort=score["cohort"],data_sha="tiny-data",scaler_sha="tiny-A60-scaler",source_sha=source,config_sha=config,model_sha=model,
            metadata=meta,arrays={key:dict(shape=shape) for key,shape in shapes.items()},cost_record=dict(run_id=run_id,batch=43,latent_k=k,actual_F_streams=43*k,expected_F_streams=43*k,F_calls=1),
            score=score,cache_relative_path=archive,archive_sha256="tiny-archive",public_stats_relative_path=stats)

    def test_default_is_stdlib_metadata_only_and_no_scientific_operation(self):
        result = subprocess.run([sys.executable,"-B",str(HELPER)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual((value["registered_outputs"],value["fit_parents"]),(12,5))
        self.assertFalse(value["numpy_imported"] or value["torch_imported"] or value["training_or_replay_or_T_executed"])
        self.assertEqual(value["actual_model_or_array_access"],0)
        with self.assertRaises(ValueError):
            self.h.metadata(self.r,Path("tiny"),Path("tiny"),Path("tiny"),"unsealed")

    def test_exact_parent_spec_normalization_three_epoch_search_and_zero_refit(self):
        h = self.h
        for condition in self.plan["outputs"]:
            if condition["kind"] == "nested":
                continue
            base,source,search,refit = self.parent(condition)
            h.validate_parent(self.r,self.plan,condition,base,source,search,refit)
            for branch in ("short_search","wrong_scaler","wrong_model","wrong_grid","refit_nonzero"):
                args = copy.deepcopy([base,source,search,refit])
                if branch == "short_search": args[2]["actual_epochs"] = .25
                if branch == "wrong_scaler": args[3]["contract"]["scaler_sha"] = "wrong"
                if branch == "wrong_model": args[1]["model_sha"] = "wrong"
                if branch == "wrong_grid": args[1]["spec"]["k_grid"] = [99]
                if branch == "refit_nonzero": args[3]["updates"] = 1
                with self.subTest(role=condition["role"],branch=branch),self.assertRaises(ValueError):
                    h.validate_parent(self.r,self.plan,condition,*args)

    def test_all_twelve_original_layouts_shapes_parent_artifacts_and_streams(self):
        h = self.h
        kinds = []
        for condition in self.plan["outputs"]:
            kinds.append(condition["kind"])
            parent = self.nested_contract() if condition["kind"]=="nested" else self.parent(condition)[0]
            manifest = self.manifest(condition,parent)
            h.validate_saved_v(self.r,self.plan,condition,manifest,parent)
            self.assertEqual(h.folder_for(Path("tiny"),condition).parts[1],"predictions" if condition["kind"]=="independent" else "ordered_predictions")
            for branch in ("horizon","scaler","artifact","streams","source","split"):
                bad = copy.deepcopy(manifest)
                if branch == "horizon": bad["arrays"]["pred_normalized"]["shape"][1]=12
                if branch == "scaler": bad["scaler_sha"]="wrong"
                if branch == "artifact": bad["metadata"]["artifact_sha256" if condition["kind"]=="independent" else "parent_artifact_sha256"]="wrong"
                if branch == "streams": bad["cost_record"]["actual_F_streams"]+=1
                if branch == "source": bad["source_sha"]="wrong"
                if branch == "split": bad["split"]="T_final"
                with self.subTest(role=condition["role"],branch=branch),self.assertRaises(ValueError):
                    h.validate_saved_v(self.r,self.plan,condition,bad,parent)
        self.assertEqual((kinds.count("independent"),kinds.count("ordered"),kinds.count("nested")),(3,6,3))
        with self.assertRaises(ValueError): h.choose(self.plan,self.plan["outputs"][0]["run_id"],"T_final")

    def test_nested_contract_has_no_fabricated_BASE_optimizer_and_actual_basis_guard(self):
        h,r = self.h,self.r
        c = self.nested_contract()
        h.validate_nested_contract(r,self.plan,c)
        self.assertNotIn("optimizer",c)
        for key in ("fit_end","data_sha","source_sha"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                h.validate_nested_contract(r,self.plan,dict(c,**{key:"wrong"}))
        class Array:
            def __init__(self,shape,item=None): self.shape,self.value=shape,item
            def item(self): return self.value
            def __gt__(self,other): return self
            def all(self): return True
        values = dict(basis=Array((11,11)),center=Array((11,)),mean=Array((11,)),std=Array((11,)),
                      fit_end=Array((),12622),data_sha=Array((),"tiny-data"),source_sha=Array((),c["source_sha"]))
        class Archive(dict):
            @property
            def files(self): return list(self)
            def __enter__(self): return self
            def __exit__(self,*args): pass
        np = types.SimpleNamespace(load=lambda *a,**kw:Archive(values),isfinite=lambda x:x)
        evaluation = types.SimpleNamespace(array_fingerprint=lambda value:dict(shape=list(value.shape),sha256="tiny-basis"))
        fit = types.SimpleNamespace(scaler_fingerprint=lambda *a:"tiny-A60-scaler")
        manifest = dict(model_sha=r.fingerprint(dict(full_basis=evaluation.array_fingerprint(values["basis"]),center=evaluation.array_fingerprint(values["center"]),artifact_sha=c["artifact_sha256"],foundation_sha="tiny-foundation")))
        result = h.validate_nested_arrays(r,self.plan,manifest,c,"unopened",np,evaluation,fit)
        self.assertEqual(result["actual_optimizer_updates"],0)
        self.assertTrue(result["no_BASE_fit_receipt_required"])
        values["basis"] = Array((11,3))
        with self.assertRaises(ValueError): h.validate_nested_arrays(r,self.plan,manifest,c,"unopened",np,evaluation,fit)

    def test_projection_preserves_twelve_and_local_failure_does_not_block_valid_one(self):
        h = self.h
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            condition = self.plan["outputs"][0]
            base = self.parent(condition)[0]
            manifest = self.manifest(condition,base)
            paths = [(public/"runs"/base["run_id"]/"BASE_FIT_RECEIPT.json",base),
                     (h.folder_for(public,condition)/"PREDICTION_MANIFEST.json",manifest)]
            for path,value in paths:
                path.parent.mkdir(parents=True,exist_ok=True);h.new(path,value)
            bad = public/"runs"/self.plan["outputs"][4]["parent_run_id"]/"BASE_FIT_RECEIPT.json"
            bad.parent.mkdir(parents=True);h.new(bad,dict(status="BUDGET_LIMIT"))
            broken = h.folder_for(public,self.plan["outputs"][3])/"PREDICTION_MANIFEST.json"
            broken.parent.mkdir(parents=True);broken.write_text("{",encoding="utf-8")
            rows = h.registry_projection(self.plan,public)
            self.assertEqual(len(rows),12)
            self.assertEqual(rows[0]["execution_status"],"COMPLETE_VALID")
            self.assertEqual(rows[4]["execution_status"],"BUDGET_LIMIT")
            self.assertEqual(rows[3]["execution_status"],"INVALID_OR_UNREADABLE_V_METADATA")
            self.assertEqual(rows[1]["execution_status"],"NO_COMPLETE_V_MANIFEST")
            self.assertTrue(all(row["reason"] for row in rows))

    def test_prepare_inputs_all_kinds_uses_exact_roots_and_hashes_no_deserialization(self):
        h,r = self.h,self.r
        for kind in ("independent","ordered","nested"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);old_public,old_cache=root/"p",root/"c"
                public,private=old_public/r.NAME,old_cache/r.NAME
                condition=next(row for row in self.plan["outputs"] if row["kind"]==kind)
                parent=self.nested_contract() if kind=="nested" else self.parent(condition)[0]
                artifact=private/("ordered/primary_beijing/nested_pca.npz" if kind=="nested" else parent["artifact_cache_relative_path"])
                artifact.parent.mkdir(parents=True);artifact.write_bytes(b"tiny unopened artifact")
                parent["artifact_sha256"]=h.sha(artifact)
                manifest=self.manifest(condition,parent)
                archive=private/manifest["cache_relative_path"];archive.parent.mkdir(parents=True);archive.write_bytes(b"tiny unopened archive")
                manifest["archive_sha256"]=h.sha(archive)
                if kind=="nested": h.new(artifact.with_name("nested_pca_contract.json"),parent)
                else:
                    base,source,search,refit=self.parent(condition);base.update(artifact_sha256=parent["artifact_sha256"])
                    run=public/"runs"/base["run_id"]
                    for name,value in (("BASE_FIT_RECEIPT.json",base),("SOURCE_CONTRACT.json",source),("search/RECEIPT.json",search),("refit/RECEIPT.json",refit)):
                        path=run/name;path.parent.mkdir(parents=True,exist_ok=True);h.new(path,value)
                target=h.folder_for(public,condition);target.mkdir(parents=True)
                h.new(target/"PREDICTION_MANIFEST.json",manifest)
                reference={key:manifest[key] for key in ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha")}
                h.new(target/"REFERENCE_CONTRACT.json",reference)
                with (target/"SCORES.csv").open("x",newline="",encoding="utf-8") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(manifest["score"]));writer.writeheader();writer.writerow(manifest["score"])
                (target/"PUBLIC_ERROR_STATS.csv.gz").write_bytes(b"tiny unopened statistics")
                values,score,paths,bindings=h.prepare_inputs(r,self.plan,old_public,old_cache,condition["run_id"],condition)
                self.assertEqual(score["n_origins"],"43")
                self.assertTrue(any(row["root"]=="cache" and row["sha256"]==h.sha(artifact) for row in bindings))
                self.assertEqual(len(values["parent_receipts_at_capture"]),5)
                self.assertEqual(sum(row["receipt_present"] for row in values["parent_receipts_at_capture"]),0 if kind=="nested" else 1)
                artifact.write_bytes(b"tampered")
                with self.assertRaises(ValueError):h.prepare_inputs(r,self.plan,old_public,old_cache,condition["run_id"],condition)

    def test_registry_alias_exact_twelve_bytes_no_overwrite_and_original_api_arguments(self):
        h = self.h
        with tempfile.TemporaryDirectory() as folder:
            public=Path(folder)
            content=("experiment_id\n"+"\n".join(row["run_id"] for row in self.plan["outputs"])+"\n").encode()
            (public/"REGISTRY.csv").write_bytes(content)
            plan=dict(self.plan,postT_registry_sha256=h.sha(public/"REGISTRY.csv"))
            self.assertTrue(h.ensure_registry_alias(self.r,public,plan)["created_now"])
            self.assertFalse(h.ensure_registry_alias(self.r,public,plan)["created_now"])
            (public/"EXPERIMENT_REGISTRY.csv").write_bytes(b"wrong")
            with self.assertRaises(ValueError):h.ensure_registry_alias(self.r,public,plan)
            self.assertEqual((public/"REGISTRY.csv").read_bytes(),content)
        calls=[];ids=[row["run_id"] for row in self.plan["outputs"]]
        manifest={"cost_record":{"actual_F_streams":129}}
        h.rescore_call(lambda *a:calls.append(a),manifest,"ref","score",["stat"],["registry12"],ids,"newprivate")
        self.assertEqual(calls[0],(manifest,"ref","score",["stat"],["registry12"],ids,[manifest["cost_record"]],"newprivate"))

    def test_replay_fixed_tolerance_six_unique_origins_actual_6K_and_tri_state(self):
        h=self.h
        for k in (3,6,11):
            calls=[]
            def api(*args,**kwargs):
                calls.append((args,kwargs))
                return dict(status="PASS",run_id="tiny",origins=list(range(6)),actual_F_streams=6*k,expected_F_streams=6*k,F_physical_streams=6*k,
                    atol=1e-5,rtol=1e-4,executed_now=True,T_final_accessed=False)
            value=h.replay_call(api,"tiny",{},"private","public","model",100,k)
            self.assertEqual(calls[0][1],dict(device="cpu",deadline=100,lease=None,batch_size=4,atol=1e-5,rtol=1e-4))
            self.assertIs(h.forward_evidence(True,value)["actual_checkpoint_forward_in_this_command"],True)
            self.assertIs(h.forward_evidence(True,dict(value,executed_now=False))["actual_checkpoint_forward_in_this_command"],False)
            with self.assertRaises(ValueError):h.replay_call(lambda *a,**kw:dict(value,F_physical_streams=6*k+1),"tiny",{},"p","c","m",100,k)
        self.assertIs(h.forward_evidence(False,None)["actual_checkpoint_forward_in_this_command"],False)
        self.assertIsNone(h.forward_evidence(True,None)["actual_checkpoint_forward_in_this_command"])
        self.assertFalse(h.forward_evidence(True,None)["checkpoint_replay_receipt_returned"])

    def test_execute_not_attempted_exception_fresh_and_cache_hit_with_fake_apis(self):
        h=self.h
        for branch in ("before_call","forward_then_error","fresh","cached"):
            with self.subTest(branch=branch),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);public,cache,asset=root/"p",root/"c",root/"a"
                condition=self.plan["outputs"][0];run_id=condition["run_id"]
                post=public/self.r.NAME;post.mkdir(parents=True)
                folder_path=h.folder_for(post,condition)
                manifest_path=folder_path/"PREDICTION_MANIFEST.json"
                records,forwards={},[]
                utility=types.SimpleNamespace(rss_record=lambda *a:{},save_private_failure=lambda *a:dict(phase=a[-2],error_class=type(a[-1]).__name__))
                pca=types.SimpleNamespace(validate_task=lambda *a:{"tiny":True})
                def dependency(*args):return utility if args[1]==self.r.UTILITY_PATH else pca
                runner=types.SimpleNamespace(NAME=self.r.NAME,UTILITY_PATH=self.r.UTILITY_PATH,PCA_RUNNER_PATH=self.r.PCA_RUNNER_PATH,
                    load_dependency=dependency,environment=lambda *a:None,relative=self.r.relative)
                process=types.SimpleNamespace(pid=123,create_time=lambda:1.,memory_info=lambda:types.SimpleNamespace(rss=1))
                def load_task(*args,**kwargs):
                    if branch=="before_call":raise RuntimeError("synthetic failure before replay")
                    return {"tiny":True}
                artifact=dict(route=dict(cache_relative_path="tiny",sha256="artifact"))
                def replay(*args,**kwargs):
                    forwards.append("fake forward")
                    if branch=="forward_then_error":raise RuntimeError("synthetic error after fake forward")
                    return dict(status="PASS",run_id=run_id,origins=list(range(6)),actual_F_streams=18,expected_F_streams=18,F_physical_streams=18,
                        atol=1e-5,rtol=1e-4,executed_now=branch=="fresh",T_final_accessed=False,selected_artifact_bindings=artifact,
                        standard_manifest_sha256="pin",standard_manifest_public_relative_path=manifest_path.relative_to(post).as_posix())
                modules=dict(psutil=types.SimpleNamespace(Process=lambda *a:process),
                    torch=types.SimpleNamespace(set_num_threads=lambda *a:None,cuda=types.SimpleNamespace(is_initialized=lambda:False)),
                    data_contract=types.SimpleNamespace(load_task=load_task),fit_base=types.SimpleNamespace(),route_predictions=types.SimpleNamespace(replay_route=replay))
                values=dict(fit_receipt=dict(artifact_sha256="artifact"),manifest=dict(archive_sha256="archive"),parent_receipts_at_capture=[])
                reference=dict(standard_archive_sha256="archive",expected_F_streams=18,created_before_inference=True,artifact_bindings=artifact)
                with mock.patch.dict(sys.modules,modules),mock.patch.object(h,"load_runner",return_value=runner), \
                     mock.patch.object(h,"metadata",return_value=self.plan),mock.patch.object(h,"sha",return_value="pin"), \
                     mock.patch.object(h,"new",side_effect=lambda path,value:records.__setitem__(Path(path).name,value)), \
                     mock.patch.object(h,"read",return_value=reference),mock.patch.object(h,"prepare_inputs",return_value=(values,{},dict(manifest=manifest_path),[])), \
                     mock.patch.object(h,"registry_projection",return_value=[]),mock.patch.object(h,"ensure_registry_alias",return_value=dict(sha256="pin")):
                    code=h.execute(public,cache,asset,"pin","replay",run_id,time.time()+10,post/"result")
                result=records["RESULT.json"]
                self.assertIs(result["actual_checkpoint_forward_in_this_command"],dict(before_call=False,forward_then_error=None,fresh=True,cached=False)[branch])
                self.assertEqual(result["checkpoint_replay_call_attempted"],branch!="before_call")
                self.assertEqual(result["checkpoint_replay_receipt_returned"],branch in {"fresh","cached"})
                self.assertEqual(code,0 if branch in {"fresh","cached"} else 1)
                self.assertEqual(len(forwards),int(branch!="before_call"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
