"""Synthetic opaque shadows and fake CPU replay only; no real data/model API."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
HELPER=HERE/"prepare_reference_cpu_replay.py"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class Watch:
    def __init__(self,*args):self.process=types.SimpleNamespace(pid=123,create_time=lambda:1.)
    def close(self):pass


class ReplayShadowTests(unittest.TestCase):
    def setUp(self):
        self.h=load("synthetic_reference_replay",HELPER)
        fixture=load("reference_metadata_tiny_fixtures",HERE/"test_prepare_reference_cpu_costs.py")
        self.fixture_case=fixture.ReferenceCostTests();self.fixture_case.setUp()

    def fixture(self,folder,family="RAW"):
        public,cache,asset,runner,plan,run_id=self.fixture_case.fixture(Path(folder),family)
        ref=self.fixture_case.h
        for name in self.h.SOURCE_NAMES:plan["source39"].setdefault(name,"tiny-"+name)
        plan["immutable_public"]["EXPERIMENT_REGISTRY.csv"]=ref.sha(public/"EXPERIMENT_REGISTRY.csv")
        post,private=public/runner.NAME,cache/runner.NAME
        post.mkdir();private.mkdir()
        env=dict(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",INFORMS_ASSET_ROOT=str(asset),INFORMS_CACHE_ROOT=str(private))
        return public,cache,asset,runner,plan,run_id,ref,post,private,env

    def context(self,ref,runner,env):
        return (mock.patch.object(self.h,"load_reference",return_value=ref),mock.patch.object(ref,"load_runner",return_value=runner),
                mock.patch.object(self.h,"Watch",Watch),mock.patch.dict(os.environ,env,clear=True))

    def prepared(self,root,family="RAW"):
        data=self.fixture(root,family)
        public,cache,asset,runner,plan,run_id,ref,post,private,env=data
        a,b,c,d=self.context(ref,runner,env)
        with a,b,c,d:result=self.h.prepare(public,cache,asset,ref.PLAN_SHA,run_id,post/"prepare",time.time()+10)
        self.assertEqual(result["status"],"PREPARED_FRESH_PRIVATE_CPU_REPLAY_SHADOW",result)
        config_path=cache/result["shadow_config_cache_relative_path"]
        return data,result,config_path

    def test_default_is_metadata_no_copy_science_or_process(self):
        result=subprocess.run([sys.executable,"-B",str(HELPER)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout)
        self.assertFalse(value["numpy_imported"] or value["torch_imported"])
        self.assertEqual(value["files_written"],0)
        self.assertEqual(value["actual_copy_or_model_or_array_or_replay_or_child_operations"],0)

    def test_source_dependency_pin_is_checked_before_importing_any_changed_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"prepare_reference_cpu_costs.py").write_text("raise AssertionError('must never import')",encoding="utf-8")
            with mock.patch.object(self.h,"HERE",root),mock.patch.object(self.h.importlib.util,"spec_from_file_location") as imported:
                with self.assertRaises(ValueError):self.h.load_reference()
                imported.assert_not_called()

    def test_three_independent_shadows_copy_only_exact_metadata_weights_V_never_old_replay(self):
        for family in ("RAW","RAW_RIDGE","DIRECT_G"):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,runner,plan,run_id,ref,post,private,env=self.fixture(folder,family)
                old=cache/"route_replays"/run_id/"V_subset.npz";old.parent.mkdir(parents=True);old.write_bytes(b"old GPU evidence must stay")
                original={str(path):ref.sha(path) for root in (public,cache) for path in root.rglob("*") if path.is_file()}
                a,b,c,d=self.context(ref,runner,env)
                with a,b,c,d:result=self.h.prepare(public,cache,asset,ref.PLAN_SHA,run_id,post/"prepare",time.time()+10)
                self.assertEqual(result["status"],"PREPARED_FRESH_PRIVATE_CPU_REPLAY_SHADOW",result)
                shadow=cache/result["shadow_cache_relative_root"]
                self.assertEqual(result["copied_files"],4 if family=="RAW" else 5)
                self.assertEqual(result["source_code_or_foundation_model_copies"],0)
                self.assertEqual(result["selected_adapter_or_ridge_artifact_copies"],0 if family=="RAW" else 1)
                self.assertFalse((shadow/"cache/route_replays").exists())
                self.assertFalse((shadow/"public/route_replays").exists())
                self.assertFalse((shadow/"public/SELECTION_LOCK.json").exists())
                self.assertFalse(any(path.suffix==".py" for path in shadow.rglob("*")))
                self.assertTrue(all(ref.sha(Path(path))==digest for path,digest in original.items()))
                self.assertEqual(result["actual_model_construction_or_training_or_forward_or_array_deserialization"],0)
                encoded=json.dumps(result)
                self.assertNotIn(str(public),encoded);self.assertNotIn(str(cache),encoded);self.assertNotIn(str(asset),encoded)
                config=runner.read(cache/result["shadow_config_cache_relative_path"])
                self.assertEqual(config["original_roots"]["public"],str(public))
                self.assertEqual(len(config["truth_arrays"]),4)

    def test_opaque_copy_wrong_existing_tamper_or_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/"source";target=root/"new/target"
            source.write_bytes(b"tiny opaque")
            ref=self.fixture_case.h
            row=dict(bytes=source.stat().st_size,sha256=ref.sha(source))
            self.h.copy_one(ref,source,target,row)
            with self.assertRaises(FileExistsError):self.h.copy_one(ref,source,target,row)
            source.write_bytes(b"modified")
            with self.assertRaises(ValueError):self.h.copy_one(ref,source,root/"another",row)
            with self.assertRaises(ValueError):self.h.inside(root,"../outside")

    def test_config_pins_roots_helper_and_selected_scope_and_existing_replay_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            data,result,path=self.prepared(folder,"RAW_RIDGE")
            public,cache,asset,runner,plan,run_id,ref,post,private,env=data
            a,b,c,d=self.context(ref,runner,env)
            with a,b,c,d:
                _,state=self.h.metadata(public,cache,asset,run_id,ref.PLAN_SHA)
                config,shadow,dest=self.h.load_shadow(ref,state,public,cache,asset,run_id,path,result["shadow_config_sha256"])
                with self.assertRaises(ValueError):self.h.load_shadow(ref,state,public,cache,asset,run_id,path,"changed")
                tampered=copy.deepcopy(config);tampered["original_roots"]["cache"]="outside"
                path.write_text(json.dumps(tampered),encoding="utf-8")
                with self.assertRaises(ValueError):self.h.load_shadow(ref,state,public,cache,asset,run_id,path,ref.sha(path))
                occupied=dest["cache"]/"route_replays"/run_id;occupied.mkdir(parents=True)
                with self.assertRaises(FileExistsError):self.h.empty_replay_paths(dest["public"],dest["cache"],run_id)

    def test_fixed_CPU_replay_contract_rejects_cache_hit_counter_or_tolerance_changes(self):
        for width in (0,21):
            value=dict(run_id="tiny",executed_now=True,origins=list(range(6)),actual_F_streams=6*width,expected_F_streams=6*width,F_physical_streams=6*width,atol=1e-5,rtol=1e-4,T_final_accessed=False)
            self.h.validate_replay(value,"tiny",width)
            for changed in (dict(executed_now=False),dict(F_physical_streams=1+6*width),dict(atol=1e-3),dict(origins=[1]*6),dict(T_final_accessed=True)):
                with self.subTest(changed=changed),self.assertRaises(ValueError):self.h.validate_replay(dict(value,**changed),"tiny",width)
        self.assertIs(self.h.replay_evidence(False,None)["actual_checkpoint_forward_in_this_command"],False)
        self.assertIsNone(self.h.replay_evidence(True,None)["actual_checkpoint_forward_in_this_command"])
        self.assertFalse(self.h.replay_evidence(True,None)["checkpoint_replay_receipt_returned"])

    def test_replay_runtime_original_API_arguments_are_cpu_fixed_tol_and_original_loader_cache(self):
        calls=[]
        runtime=self.h.ReplayRuntime.__new__(self.h.ReplayRuntime)
        runtime.api=lambda *args,**kwargs:calls.append((args,kwargs))
        roots=dict(public="shadowpublic",cache="shadowcache")
        runtime.replay("oldID","task",roots,"oldModel",100)
        self.assertEqual(calls[0],(("oldID","task","shadowcache","shadowpublic","oldModel"),dict(device="cpu",deadline=100,lease=None,batch_size=4,atol=1e-5,rtol=1e-4)))

    def test_fake_replay_fresh_success_error_unknown_notattempted_and_cached_rejected(self):
        h=self.h
        for branch in ("fresh","forward_then_error","before_call","cached","numerical_mismatch"):
            with self.subTest(branch=branch),tempfile.TemporaryDirectory() as folder:
                data,preparation,path=self.prepared(folder,"RAW")
                public,cache,asset,runner,plan,run_id,ref,post,private,env=data
                config=runner.read(path)
                calls=[]
                def task(*args):
                    self.assertEqual(args[-1],cache)
                    if branch=="before_call":raise RuntimeError("synthetic task failure")
                    return {"tiny":True},{"tiny-binding":True}
                def replay_api(identity,task,roots,model,deadline):
                    calls.append("fake forward")
                    if branch=="forward_then_error":raise RuntimeError("synthetic error after a fake forward")
                    folder=roots["public"]/"route_replays"/identity;folder.mkdir(parents=True)
                    reference=dict(source_sha=config["source_sha"],standard_manifest_sha256=next(row["sha256"] for row in config["files"] if row["relative_path"].endswith("PREDICTION_MANIFEST.json")),
                        standard_archive_sha256=next(row["sha256"] for row in config["files"] if row["relative_path"].endswith("V_select.npz")),created_before_inference=True,created_unix=1)
                    runner.new(folder/"PRE_INFERENCE_REFERENCE.json",reference)
                    archive_relative="route_replays/"+identity+"/V_subset.npz"
                    archive=roots["cache"]/archive_relative;archive.parent.mkdir(parents=True);archive.write_bytes(b"synthetic new replay archive")
                    receipt=dict(status="FAIL_NUMERICAL_PARITY" if branch=="numerical_mismatch" else "PASS",run_id=identity,origins=list(range(6)),
                        actual_F_streams=126,expected_F_streams=126,F_physical_streams=126,atol=1e-5,rtol=1e-4,T_final_accessed=False,
                        reference_sha=ref.fingerprint(reference),cache_relative_path=archive_relative,archive_sha256=ref.sha(archive))
                    runner.new(folder/"REPLAY_RECEIPT.json",receipt)
                    return dict(receipt,executed_now=branch!="cached")
                runtime=types.SimpleNamespace(task=task,replay=replay_api)
                a,b,c,d=self.context(ref,runner,env)
                with a,b,c,d:
                    result=h.replay(public,cache,asset,ref.PLAN_SHA,run_id,post/"actual_fake_replay",time.time()+10,path,preparation["shadow_config_sha256"],_runtime=runtime)
                    with self.assertRaises(FileExistsError):h.replay(public,cache,asset,ref.PLAN_SHA,run_id,post/"forbidden_second",time.time()+10,path,preparation["shadow_config_sha256"],_runtime=runtime)
                self.assertEqual(result["status"],"PASS_FRESH_CPU_SIX_ORIGIN_PARITY" if branch=="fresh" else "FAILED_OR_PARITY_MISMATCH_PRESERVED")
                expected=dict(fresh=True,forward_then_error=None,before_call=False,cached=False,numerical_mismatch=True)[branch]
                self.assertIs(result["actual_checkpoint_forward_in_this_command"],expected)
                self.assertEqual(result["checkpoint_replay_call_attempted"],branch!="before_call")
                self.assertEqual(result["checkpoint_replay_receipt_returned"],branch not in {"before_call","forward_then_error"})
                self.assertEqual(len(calls),int(branch!="before_call"))
                self.assertFalse(result["new_T_target_or_scoring"] or result["new_training"] or result["cost_timing_performed"])
                self.assertTrue(result["original_source_registry_scores_cost_T_and_selected_inputs_preserved"])


if __name__=="__main__":unittest.main(verbosity=2)
