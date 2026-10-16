"""Tiny stdlib fixtures only; ROOT runs after the precision-cost quiet window."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("tiny_required_verifier", HERE / "verify_required_minima_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RequiredVerifierTests(unittest.TestCase):
    def setUp(self):
        self.h = load()
        self.runner = self.h.load_runner()

    def fixture(self, root, k=6, epoch=0.):
        h, runner = self.h, self.runner
        public, cache = root / "public", root / "cache"
        post, private = public / runner.NAME, cache / runner.NAME
        condition = next(row for row in runner.conditions("campaign") if row["family"] == "B2" and row["latent_k"] == k)
        run_id = condition["run_id"]
        contract = dict(shape=[21038,11], L=512, H=48, columns=[str(i) for i in range(11)], array_sha256={"physical":"array"},
            data_sha256="data", search_scaler_sha256="A50", refit_scaler_sha256="A60",
            legal_origin_counts_primary=dict(V_select=43), primary_split_ranges=dict(A_fit=[0,10519], A_stop=[10519,12622]))
        plan = dict(conditions=runner.conditions("campaign"), task_contract=contract, foundation_tensor_sha256="foundation",
            source39={name:name+"-sha" for name in set(runner.FIT_SOURCES)|set(h.PREDICTION_SOURCES)}, postT_registry_relative_path="REGISTRY.csv")
        folder = post / "runs" / run_id
        pred = post / "predictions" / run_id / "V_select"
        folder.mkdir(parents=True); pred.mkdir(parents=True); (private / "runs" / run_id).mkdir(parents=True)
        (private / "predictions" / run_id).mkdir(parents=True)
        artifact = private / "runs" / run_id / "selected_bundle.pt"
        artifact.write_bytes(b"opaque selected adapter")
        archive = private / "predictions" / run_id / "V_select.npz"
        archive.write_bytes(b"opaque saved array; never decoded by fixtures")
        spec = condition["spec"]
        axes = dict(L=512,H=48,C=11,columns=contract["columns"],fold="primary",source_array_fingerprints=contract["array_sha256"])
        fit_files={name:plan["source39"][name] for name in runner.FIT_SOURCES}
        source = dict(spec=spec, files=fit_files, source_sha=runner.fingerprint(fit_files),
            config_sha=runner.fingerprint(dict(spec=spec,axes=axes)),data_sha="data",model_sha="foundation")
        core = {key:source[key] for key in ("source_sha","config_sha","data_sha","model_sha")}
        exposures = math.ceil(epoch*12063)
        base = dict(status="COMPLETE_VALID",run_id=run_id,family="B2",epochs=[.25,1.,3.],seed=92601,selected_epoch=epoch,
            reused_complete_artifact=False,search_exposures=29880,search_updates=1868,refit_exposures=exposures,refit_updates=math.ceil(exposures/16),
            search_fit_end=10519,refit_fit_end=12622,normalization_fit_end=12622,refit_scaler_sha="A60",source_sha=source["source_sha"],
            artifact_cache_relative_path=artifact.relative_to(private).as_posix(),artifact_sha256=h.sha(artifact),model_state_sha="model-state")
        search = dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=29880,updates=1868,train_origins=9960,
            contract=dict(core,scaler_sha="A50",initialization_sha="a"*64))
        refit = dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),exposures=exposures,updates=math.ceil(exposures/16),
            train_origins=12063,contract=dict(core,scaler_sha="A60",initialization_sha="b"*64))
        arrays = {key:dict(shape=shape,dtype="tiny",sha256=key) for key,shape in dict(pred_normalized=[43,48,11],
            target_normalized=[43,48,11],observed=[43,48,11],origins=[43],channel_ids=[11]).items()}
        score = dict(run_id=run_id,split="V_select",cohort="beijing:primary:V_select:stride48",mse=.5,mae=.4,n_origins=43,num_channels=11,horizon=48)
        prediction_files={name:plan["source39"][name] for name in h.PREDICTION_SOURCES}
        identity = dict(run_id=run_id,split="V_select",cohort=score["cohort"],
            source_sha=runner.fingerprint(dict(deployment_sources=prediction_files,training_source_sha=source["source_sha"])),
            config_sha=runner.fingerprint(dict(parent_spec=spec,dense=False,fold="primary",role="base_parent")),data_sha="data",
            model_sha=runner.fingerprint(dict(artifact_sha=base["artifact_sha256"],model_state_sha="model-state",normalization="A60")),scaler_sha="A60")
        reference = dict(identity,arrays={key:value for key,value in arrays.items() if key!="pred_normalized"},expected_F_streams=43*k)
        manifest = dict(identity,arrays=arrays,score=score,cache_relative_path=archive.relative_to(private).as_posix(),archive_sha256=h.sha(archive),
            metadata=dict(artifact_sha256=base["artifact_sha256"],canonical_scaler_fit_end=12622,parent_normalization_fit_end=12622,
                parent_fit_end=12622,family="B2",fold="primary",role="base_parent",dense=False,reused_prediction_cache=False,
                training_source_sha=source["source_sha"],source_files=prediction_files),
            cost_record=dict(run_id=run_id,batch=43,latent_k=k,actual_F_streams=43*k,expected_F_streams=43*k,F_calls=1),
            public_stats_relative_path="predictions/"+run_id+"/V_select/PUBLIC_ERROR_STATS.csv.gz")
        for path,value in ((folder/"BASE_FIT_RECEIPT.json",base),(folder/"SOURCE_CONTRACT.json",source),
                (pred/"PREDICTION_MANIFEST.json",manifest),(pred/"REFERENCE_CONTRACT.json",reference)):
            h.new(path,value)
        for label,value in (("search",search),("refit",refit)):
            (folder/label).mkdir(); h.new(folder/label/"RECEIPT.json",value)
        import csv
        with (pred/"SCORES.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(score));writer.writeheader();writer.writerow(score)
        (pred/"PUBLIC_ERROR_STATS.csv.gz").write_bytes(b"opaque statistics")
        (post/"REGISTRY.csv").write_bytes(b"experiment_id\ntiny\n")
        plan["postT_registry_sha256"] = h.sha(post/"REGISTRY.csv")
        return public,cache,plan,condition

    def test_default_cli_has_no_science_imports_or_metadata_reads(self):
        before = {name:name in sys.modules for name in ("numpy","torch","psutil")}
        with mock.patch.object(self.h,"load_runner",side_effect=AssertionError("no root metadata requested")),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["files_written"],0)
        self.assertEqual(result["actual_Task_array_model_or_child_operations"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})

    def test_single_valid_condition_is_not_blocked_by_other_five_missing_or_corrupt(self):
        with tempfile.TemporaryDirectory() as td:
            public,cache,plan,condition=self.fixture(Path(td))
            other=public/self.runner.NAME/"runs"/plan["conditions"][1]["run_id"]
            other.mkdir(parents=True);(other/"BASE_FIT_RECEIPT.json").write_text("{")
            values,_,_,bindings=self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
            self.assertEqual(values["fit_receipt"]["refit_updates"],0)
            projection=self.h.registry_projection(plan,public/self.runner.NAME)
            self.assertEqual(len(projection),6)
            self.assertEqual([r["execution_status"] for r in projection[:3]],["COMPLETE_VALID","UNREADABLE_BASE_FIT_RECEIPT","NO_COMPLETE_BASE_FIT_RECEIPT"])
            self.h.check_bindings(public,cache,bindings)
            path=cache/self.runner.NAME/values["fit_receipt"]["artifact_cache_relative_path"]
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):self.h.check_bindings(public,cache,bindings)

    def test_both_K_and_selected_zero_or_three_keep_full_search_and_selected_refit(self):
        for k,epoch in ((6,0.),(11,3.)):
            with self.subTest(k=k,epoch=epoch),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td),k,epoch)
                self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
                self.assertEqual(self.h.choose(self.runner,plan,condition["run_id"],"rescore"),condition)
                for mode,run_id in (("T_final",condition["run_id"]),("rescore","other")):
                    with self.assertRaises(ValueError):self.h.choose(self.runner,plan,run_id,mode)

    def test_other_nonobject_or_malformed_status_receipts_do_not_block_selected_run(self):
        for label in ("list","null","number","string","bool","status_list","status_dict","identity_type"):
            with self.subTest(label=label),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td))
                other=plan["conditions"][1]
                record=dict(run_id=other["run_id"],family=other["family"],status="COMPLETE_VALID")
                if label in {"list","null","number","string","bool"}:
                    bad={"list":[],"null":None,"number":7,"string":"invalid","bool":True}[label]
                elif label=="identity_type":bad=dict(record,run_id=[])
                else:bad=dict(record,status=[] if label=="status_list" else {})
                folder=public/self.runner.NAME/"runs"/other["run_id"]
                folder.mkdir(parents=True);(folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(bad),encoding="utf-8")
                self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
                rows=self.h.registry_projection(plan,public/self.runner.NAME)
                self.assertEqual(rows[0]["execution_status"],"COMPLETE_VALID")
                self.assertEqual(rows[1]["execution_status"],"UNREADABLE_BASE_FIT_RECEIPT")
                self.assertEqual(len(rows),6)
                selected=public/self.runner.NAME/"runs"/condition["run_id"]/"BASE_FIT_RECEIPT.json"
                if isinstance(bad,dict):bad=dict(bad,run_id=condition["run_id"],family=condition["family"])
                if label=="identity_type":bad=dict(bad,run_id=[])
                selected.write_text(json.dumps(bad),encoding="utf-8")
                with self.assertRaises(ValueError):self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)

    def test_recipe_stage_scaler_model_shape_artifact_and_T_mutations_reject(self):
        mutations=(("source_contract","config_sha","changed"),("search","completed_epochs",[0.,.25,1.]),
            ("refit","contract",{}),("manifest","split","T_final"),("manifest","model_sha","changed"),
            ("manifest","scaler_sha","A50"),("manifest","arrays",{}),("fit_receipt","artifact_sha256","changed"))
        for name,key,value in mutations:
            with self.subTest(name=name,key=key),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td))
                folder=public/self.runner.NAME
                paths=dict(source_contract=folder/"runs"/condition["run_id"]/"SOURCE_CONTRACT.json",search=folder/"runs"/condition["run_id"]/"search/RECEIPT.json",
                    refit=folder/"runs"/condition["run_id"]/"refit/RECEIPT.json",manifest=folder/"predictions"/condition["run_id"]/"V_select/PREDICTION_MANIFEST.json",
                    fit_receipt=folder/"runs"/condition["run_id"]/"BASE_FIT_RECEIPT.json")
                path=paths[name];record=self.h.read(path);record[key]=value;path.write_text(json.dumps(record))
                with self.assertRaises((ValueError,KeyError)):self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)

    def test_registry_alias_is_exact_additive_and_wrong_existing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            public,_,plan,_=self.fixture(Path(td));post=public/self.runner.NAME
            original=(post/"REGISTRY.csv").read_bytes()
            self.assertTrue(self.h.ensure_registry_alias(self.runner,post,plan)["created_now"])
            self.assertFalse(self.h.ensure_registry_alias(self.runner,post,plan)["created_now"])
            (post/"EXPERIMENT_REGISTRY.csv").write_bytes(b"wrong")
            with self.assertRaises(ValueError):self.h.ensure_registry_alias(self.runner,post,plan)
            self.assertEqual((post/"REGISTRY.csv").read_bytes(),original)

    def test_auditor_original_API_receives_single_bundle_and_complete_registry_scope(self):
        api=mock.Mock(return_value=dict(status="PASS"));manifest=dict(cost_record=dict(actual_F_streams=258))
        self.h.rescore_call(api,manifest,"reference","score",["statistics"],["all six"],["ids"],"private")
        api.assert_called_once_with(manifest,"reference","score",["statistics"],["all six"],["ids"],[manifest["cost_record"]],"private")

    def test_original_CPU_replay_single_batch_six_and_strict_counters_tolerance(self):
        for k in (6,11):
            value=dict(status="PASS",run_id="tiny",origins=list(range(6)),actual_F_streams=6*k,expected_F_streams=6*k,F_physical_streams=6*k,
                F_calls=1,atol=1e-5,rtol=1e-4,executed_now=True,T_final_accessed=False,independent_training_rerun=False,model_size="small",
                scope="actual_physical_input_selected_checkpoint_V_subset_replay")
            api=mock.Mock(return_value=value)
            actual=self.h.replay_call(api,"tiny","task","cache","public","model",100.)
            api.assert_called_once_with("tiny","task","cache","public","model",device="cpu",deadline=100.,lease=None,batch_size=6,atol=1e-5,rtol=1e-4)
            self.h.validate_replay(actual,"tiny",k)
            for key,wrong in (("F_calls",2),("F_physical_streams",0),("actual_F_streams",0),("atol",1e-3),("T_final_accessed",True)):
                with self.subTest(k=k,key=key),self.assertRaises(ValueError):self.h.validate_replay(dict(value,**{key:wrong}),"tiny",k)

    def test_forward_tri_state_distinguishes_unattempted_error_fresh_and_cache(self):
        for attempted,value,forward,returned in ((False,None,False,False),(True,None,None,False),
                (True,dict(executed_now=True),True,True),(True,dict(executed_now=False),False,True),
                (True,dict(status="invalid"),None,True)):
            facts=self.h.forward_facts(attempted,value)
            self.assertIs(facts["actual_checkpoint_forward_in_this_command"],forward)
            self.assertEqual(facts["checkpoint_replay_receipt_returned"],returned)
            self.assertEqual(facts["checkpoint_replay_call_attempted"],attempted)

    def test_replay_public_receipt_reference_selected_artifact_and_archive_are_bound(self):
        h=self.h
        with tempfile.TemporaryDirectory() as td:
            old_public,old_cache,plan,condition=self.fixture(Path(td))
            values,_,paths,_=h.prepare_inputs(self.runner,plan,old_public,old_cache,condition["run_id"],condition)
            public,private=old_public/self.runner.NAME,old_cache/self.runner.NAME
            run_id=condition["run_id"]
            folder=public/"route_replays"/run_id;folder.mkdir(parents=True)
            cachefolder=private/"route_replays"/run_id;cachefolder.mkdir(parents=True)
            archive=cachefolder/"V_subset.npz";archive.write_bytes(b"opaque replay")
            artifact={"route":dict(cache_relative_path=values["fit_receipt"]["artifact_cache_relative_path"],sha256=values["fit_receipt"]["artifact_sha256"])}
            reference=dict(run_id=run_id,split="V_select",origins=list(range(6)),route_sha="route",created_before_inference=True,
                standard_manifest_sha256=h.sha(paths["manifest"]),standard_archive_sha256=values["manifest"]["archive_sha256"],
                artifact_bindings=artifact,expected_F_streams=36,selection_policy="first/middle/last plus 3 remaining origins from seed92601; sorted unique")
            receipt=dict(run_id=run_id,origins=list(range(6)),route_sha="route",reference_sha=self.runner.fingerprint(reference),
                selected_artifact_bindings=artifact,standard_manifest_public_relative_path=paths["manifest"].relative_to(public).as_posix(),
                standard_manifest_sha256=h.sha(paths["manifest"]),expected_F_streams=36,cache_relative_path=archive.relative_to(private).as_posix(),archive_sha256=h.sha(archive))
            h.new(folder/"PRE_INFERENCE_REFERENCE.json",reference);h.new(folder/"REPLAY_RECEIPT.json",receipt)
            value=dict(receipt,executed_now=True)
            self.assertEqual(h.validate_replay_bindings(self.runner,value,values,paths,private,public)[2],archive)
            wrong=copy.deepcopy(value);wrong["selected_artifact_bindings"]["route"]["sha256"]="other"
            with self.assertRaises(ValueError):h.validate_replay_bindings(self.runner,wrong,values,paths,private,public)
            archive.write_bytes(b"changed")
            with self.assertRaises(ValueError):h.validate_replay_bindings(self.runner,value,values,paths,private,public)

    def test_execute_preserves_error_and_receipt_facts_without_real_scientific_modules(self):
        h=self.h
        for branch in ("before_call","forward_then_error","fresh","cached","receipt_then_validation_error"):
            with self.subTest(branch=branch),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td))
                plan.update(deadline_unix=time.time()+30,max_RSS_bytes=3221225472,storage_limit_bytes=30*1024**3,
                    admission_reserve_bytes=256*1024**2,model_asset_relative_path="tiny-model")
                asset=Path(td)/"asset";asset.mkdir()
                records,forwards={},[]
                process=types.SimpleNamespace(pid=123,create_time=lambda:1.,memory_info=lambda:types.SimpleNamespace(rss=1))
                utility=types.SimpleNamespace(rss_record=lambda *a:dict(rss_bytes=1),validate_task=lambda *a:{"tiny":True},
                    save_private_failure=lambda *a:dict(error_class=type(a[-1]).__name__,phase=a[-2]))
                def load_task(*args,**kwargs):
                    if branch=="before_call":raise RuntimeError("tiny before replay")
                    return {"tiny":True}
                def replay(*args,**kwargs):
                    forwards.append(True)
                    if branch=="forward_then_error":raise RuntimeError("tiny after fake forward")
                    return dict(status="PASS",run_id=condition["run_id"],origins=list(range(6)),actual_F_streams=36,expected_F_streams=36,
                        F_physical_streams=36,F_calls=2 if branch=="receipt_then_validation_error" else 1,atol=1e-5,rtol=1e-4,
                        executed_now=branch!="cached",T_final_accessed=False,independent_training_rerun=False,model_size="small",
                        scope="actual_physical_input_selected_checkpoint_V_subset_replay")
                modules=dict(psutil=types.SimpleNamespace(Process=lambda:process),
                    torch=types.SimpleNamespace(set_num_threads=lambda *a:None,get_num_threads=lambda:2,cuda=types.SimpleNamespace(is_initialized=lambda:False)),
                    data_contract=types.SimpleNamespace(load_task=load_task),fit_base=types.SimpleNamespace(),route_predictions=types.SimpleNamespace(replay_route=replay))
                with mock.patch.dict(sys.modules,modules),mock.patch.object(h,"load_runner",return_value=self.runner), \
                     mock.patch.object(h,"metadata",return_value=plan),mock.patch.object(self.runner,"environment"), \
                     mock.patch.object(self.runner,"load_dependency",return_value=utility),mock.patch.object(h,"require_original_module"), \
                     mock.patch.object(h,"new",side_effect=lambda p,v:records.__setitem__(Path(p).name,v)), \
                     mock.patch.object(h,"validate_replay_bindings",return_value=()):
                    code=h.execute(public,cache,asset,"tiny-plan","replay",condition["run_id"],time.time()+10,
                        public/self.runner.NAME/"diagnostics"/"tiny-result")
                result=records["RESULT.json"]
                expected=dict(before_call=False,forward_then_error=None,fresh=True,cached=False,receipt_then_validation_error=True)[branch]
                self.assertIs(result["actual_checkpoint_forward_in_this_command"],expected)
                self.assertEqual(result["checkpoint_replay_receipt_returned"],branch in {"fresh","cached","receipt_then_validation_error"})
                self.assertEqual(result["actual_fresh_CPU_replay_verified"],branch=="fresh")
                self.assertEqual(code,0 if branch in {"fresh","cached"} else 1)
                self.assertEqual(len(forwards),int(branch!="before_call"))
                self.assertEqual(result["cuda_initialized"],False)

    def test_metadata_rejects_changed_plan_and_registry_pins_before_science(self):
        fake=types.SimpleNamespace(metadata=mock.Mock(return_value=dict(runner_sha256=self.h.RUNNER_SHA,postT_registry_sha256="wrong")))
        with self.assertRaises(ValueError):self.h.metadata(fake,None,None,None,"wrong-plan")
        fake.metadata.assert_not_called()
        with self.assertRaises(ValueError):self.h.metadata(fake,None,None,None,self.h.PLAN_SHA)

    def test_fresh_output_and_safe_path_do_not_resolve_away_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);public=root/"public";public.mkdir();output=public/"fresh"
            self.h.require_new_output(self.runner,output,public)
            output.mkdir()
            with self.assertRaises(FileExistsError):self.h.require_new_output(self.runner,output,public)
            with self.assertRaises(ValueError):self.h.require_new_output(self.runner,root/"escaped",public)
            with mock.patch.object(Path,"is_symlink",return_value=True),mock.patch.object(Path,"resolve",side_effect=AssertionError("must check links before resolve")):
                with self.assertRaises(ValueError):self.h.plain_path(public)


if __name__ == "__main__":unittest.main(verbosity=2)
