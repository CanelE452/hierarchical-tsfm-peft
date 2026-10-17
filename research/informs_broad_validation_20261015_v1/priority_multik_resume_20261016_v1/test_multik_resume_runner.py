"""Synthetic metadata/files only; no Task, checkpoint tensors or scientific import."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name("multik_resume_runner.py")


def load():
    if not SOURCE.is_file():
        return types.SimpleNamespace()
    spec = importlib.util.spec_from_file_location("tested_multik_resume", SOURCE)
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    return api


def stage_summary(phase):
    search = phase == "search"
    return dict(status="SEGMENT_COMPLETE", selected_epoch=3., selected_astop_mse=.6 if search else None,
        completed_epochs=[0., .25, 1., 3.] if search else [0., 3.],
        train_origins=9960 if search else 12063, updates=1868 if search else 2262,
        exposures=29880 if search else 36189, F_session_logical_streams=0 if search else 42,
        F_session_physical_streams=0 if search else 42, seconds=.01)


class ResumeTests(unittest.TestCase):
    def test_default_reads_and_writes_nothing_and_imports_no_science(self):
        api = load(); self.assertTrue(hasattr(api, "main"), "resume CLI missing")
        names = ("numpy", "torch", "psutil", "fit_base", "data_contract")
        before = {key: key in sys.modules for key in names}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(api.main([]), 0)
        receipt = json.loads(output.getvalue())
        self.assertEqual((receipt["input_reads"],receipt["writes"],receipt["scientific_operations"]),(0,0,0))
        self.assertEqual(before,{key:key in sys.modules for key in names})

    def test_plan_requires_root_seal_and_preserves_exact_two_original_specs(self):
        api = load(); self.assertTrue(hasattr(api,"draft_plan"), "sealed plan validator missing")
        draft = api.draft_plan()
        with self.assertRaises(ValueError): api.validate_plan(draft)
        plan = copy.deepcopy(draft); plan.update(status="SEALED_BY_ROOT",runner_sha256="a"*64)
        api.validate_plan(plan)
        for key,value in (("deadline_unix",1792222551),("materialized_prefix_rows",21038),("protected_T_numeric_allowed",True)):
            changed = copy.deepcopy(plan); changed[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError): api.validate_plan(changed)
        for index in (0,1):
            changed = copy.deepcopy(plan); changed["conditions"][index]["spec"]["run_id"] += "_renamed"
            with self.assertRaises(ValueError): api.validate_plan(changed)
        changed = copy.deepcopy(plan); changed["conditions"][1]["spec"].pop("k_grid")
        with self.assertRaises(ValueError): api.validate_plan(changed)

    def test_outputs_are_six_trained_prefixes_and_never_nested_or_independent_fits(self):
        api = load(); self.assertTrue(hasattr(api,"outputs"), "prefix manifest missing")
        rows = api.outputs()
        self.assertEqual(len(rows),6)
        self.assertEqual({(row["kind"],row["latent_k"]) for row in rows},{(kind,k) for kind in ("shared","max_only") for k in (3,6,11)})
        self.assertEqual(len({row["run_id"] for row in rows}),6)
        self.assertTrue(all(row["split"] == "V_select" for row in rows))

    def test_original_axes_and_full_spec_are_part_of_config_identity(self):
        api = load(); self.assertTrue(hasattr(api,"validate_lineage"), "lineage guard missing")
        condition = api.conditions()[0]
        task = dict(L=512,H=48,columns=[str(i) for i in range(11)],array_sha256={"physical":"old-full"},data_sha256=api.DATA_SHA)
        contract = dict(source_sha=api.SOURCE_SHA,config_sha=api.fingerprint(dict(spec=condition["spec"],axes=dict(L=512,H=48,C=11,columns=task["columns"],fold="primary",source_array_fingerprints=task["array_sha256"]))),data_sha=api.DATA_SHA,model_sha=api.FOUNDATION_SHA)
        source = dict(contract,spec=condition["spec"])
        self.assertEqual(api.validate_lineage(condition,source,contract,task),contract)
        changed = copy.deepcopy(task); changed["array_sha256"]["physical"] = "prefix-hash"
        with self.assertRaises(ValueError): api.validate_lineage(condition,source,contract,changed)
        changed = copy.deepcopy(source); changed["spec"]["stage_label"] = "new"
        with self.assertRaises(ValueError): api.validate_lineage(condition,changed,contract,task)

    def test_initialization_is_checked_beyond_original_five_field_guard(self):
        api = load(); self.assertTrue(hasattr(api,"validate_checkpoint"), "explicit initialization guard missing")
        condition = api.conditions()[0]
        for phase in ("search","refit"):
            expected = condition["checkpoint_contracts"][phase]
            search = phase == "search"
            checkpoint = dict(contract=copy.deepcopy(expected),updates=1868 if search else 0,
                sampler=dict(exposures=29880 if search else 0,rng={"state":1}),optimizer=dict(state={1:{}} if search else {}),
                torch_rng=b"opaque",cuda_rng=[],extra=dict(completed_epochs=[0.,.25,1.,3.] if search else [0.],best_epoch=3. if search else 0.))
            api.validate_checkpoint(checkpoint,condition,phase)
            changed = copy.deepcopy(checkpoint); changed["contract"]["initialization_sha"] = "f"*64
            with self.assertRaises(ValueError): api.validate_checkpoint(changed,condition,phase)
            changed = copy.deepcopy(checkpoint); changed.pop("torch_rng")
            with self.assertRaises(ValueError): api.validate_checkpoint(changed,condition,phase)

    def test_copy_is_byte_exact_and_preserves_all_original_and_backup_files(self):
        api = load(); self.assertTrue(hasattr(api,"copy_chain"), "exact checkpoint transport missing")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); cache=root/"cache"; public=root/"public"; cache.mkdir(); public.mkdir()
            condition=dict(run_id="same-original-id",source_contract_public_relative_path="old/SOURCE_CONTRACT.json",source_contract_sha256=None,
                run_contract_cache_relative_path="old/RUN_CONTRACT.json",run_contract_sha256=None,checkpoints=[])
            for target in (public/"old/SOURCE_CONTRACT.json",cache/"old/RUN_CONTRACT.json"):
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b'{"opaque":"same"}\n')
            condition["source_contract_sha256"]=api.sha(public/condition["source_contract_public_relative_path"])
            condition["run_contract_sha256"]=api.sha(cache/condition["run_contract_cache_relative_path"])
            for index in range(5):
                phase="search" if index<4 else "refit"; basename=f"checkpoint_{index if index<4 else 0:06d}.pt"; payload=("opaque "+str(index)).encode()
                relative="old/"+phase+"/"+basename; backup="backup/"+phase+"/"+basename
                for name in (relative,backup):
                    target=cache/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(payload)
                condition["checkpoints"].append(dict(phase=phase,basename=basename,cache_relative_path=relative,backup_cache_relative_path=backup,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
            before={path:api.sha(path) for path in root.rglob("*") if path.is_file()}
            proof=api.copy_chain(cache,public,condition,cache/"newrun",public/"newrun")
            self.assertEqual(proof["checkpoint_files_copied"],5)
            self.assertEqual(before,{path:api.sha(path) for path in before})
            for row in condition["checkpoints"]: self.assertEqual(api.sha(cache/"newrun"/row["phase"]/row["basename"]),row["sha256"])
            with self.assertRaises(FileExistsError): api.copy_chain(cache,public,condition,cache/"newrun",public/"newrun")
            (cache/condition["checkpoints"][0]["backup_cache_relative_path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError): api.copy_chain(cache,public,condition,cache/"another",public/"another")
            self.assertFalse((cache/"another").exists())

    def test_safe_relative_rejects_escape_and_absolute_paths(self):
        api=load(); self.assertTrue(hasattr(api,"relative"), "path boundary missing")
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("../escape","a/../../escape",str(Path(temporary).resolve())):
                with self.subTest(name=name),self.assertRaises(ValueError): api.relative(temporary,name)

    def test_completed_search_skips_all_updates_and_then_resumes_refit_cp_zero(self):
        api=load(); self.assertTrue(hasattr(api,"resume_stages"), "explicit stage resume missing")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); calls=[]; foundation=types.SimpleNamespace(streams=0,row_calls=0,physical_streams=0,physical_calls=0)
            private=root/"private"; public=root/"public"
            for label in ("search","refit"):
                (private/label).mkdir(parents=True); (public/label).mkdir(parents=True)
                (private/label/"checkpoint_000000.pt").write_bytes(b"opaque")
            def stage(spec,task,fit_end,stop,epochs,private_dir,public_dir,foundation,deadline,contract):
                calls.append((fit_end,stop,epochs,spec["k_grid"]))
                return dict(summary=stage_summary("search" if fit_end==10519 else "refit"),curves=[])
            result=api.resume_stages(types.SimpleNamespace(_stage=stage),api.conditions()[0],{},foundation,{},public,private,900.)
            self.assertEqual(result["status"],"COMPLETE_STAGES")
            self.assertEqual(calls,[(10519,[10519,12622],[.25,1.,3.],[3,6,11]),(12622,None,[3.],[3,6,11])])
            self.assertEqual(result["search_new_optimizer_updates"],0)
            self.assertEqual(result["search_new_F_calls"],0)

    def test_search_extra_forward_or_event_is_rejected_before_refit(self):
        api=load(); self.assertTrue(hasattr(api,"resume_stages"), "search no-op proof missing")
        for change in ("F","checkpoint","event"):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); private=root/"private"; public=root/"public"; calls=[]
                (private/"search").mkdir(parents=True); (public/"search").mkdir(parents=True)
                foundation=types.SimpleNamespace(streams=0,row_calls=0,physical_streams=0,physical_calls=0)
                def stage(*args):
                    calls.append(1)
                    if change=="F": foundation.row_calls+=1
                    elif change=="checkpoint": (private/"search/checkpoint_new.pt").write_bytes(b"new")
                    else: (public/"search/event_000000.json").write_text("{}")
                    return dict(summary=stage_summary("search"),curves=[])
                with self.assertRaises(ValueError): api.resume_stages(types.SimpleNamespace(_stage=stage),api.conditions()[0],{},foundation,{},public,private,900.)
                self.assertEqual(len(calls),1)

    def test_limited_refit_is_preserved_and_not_reported_complete(self):
        api=load(); self.assertTrue(hasattr(api,"resume_stages"), "partial-stage result missing")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/"private/search").mkdir(parents=True); (root/"public/search").mkdir(parents=True)
            def stage(*args):
                value=stage_summary("search" if args[2]==10519 else "refit")
                if args[2]==12622: value.update(status="BUDGET_LIMIT",updates=12,exposures=192)
                return dict(summary=value,curves=[])
            result=api.resume_stages(types.SimpleNamespace(_stage=stage),api.conditions()[0],{},types.SimpleNamespace(),{},root/"public",root/"private",900.)
            self.assertEqual(result["status"],"BUDGET_LIMIT_PRESERVED"); self.assertEqual(result["phase"],"refit")

    def test_refit_must_reach_original_exposure_and_update_counts(self):
        api=load(); self.assertTrue(hasattr(api,"validate_stages"), "completion proof missing")
        api.validate_stages(stage_summary("search"),stage_summary("refit"))
        for key,value in (("exposures",36188),("updates",2261),("completed_epochs",[0.,1.,3.])):
            changed=stage_summary("refit"); changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): api.validate_stages(stage_summary("search"),changed)

    def test_actual_route_retains_parent_weights_and_posthoc_registration(self):
        api=load(); self.assertTrue(hasattr(api,"route_spec"), "original Route specification missing")
        parent=api.conditions()[1]; output=next(row for row in api.outputs() if row["kind"]=="max_only" and row["latent_k"]==3)
        receipt=dict(artifact_sha256="b"*64,model_state_sha="c"*64)
        spec=api.route_spec(output,parent,receipt)
        self.assertEqual((spec["canonical_fit_end"],spec["latent_k"],spec["prefix_variant"]),(12622,3,"posthoc"))
        self.assertEqual(spec["source_parent_run_id"],parent["run_id"])
        self.assertEqual(spec["artifact_cache_relative_path"],"runs/"+parent["run_id"]+"/selected_bundle.pt")

    def test_cost_requires_fresh_quiet_proof(self):
        api=load(); self.assertTrue(hasattr(api,"require_quiet"), "quiet admission guard missing")
        proof=dict(status="QUIET_CPU_COST_WINDOW",active_model_workers=[],active_cost_workers=[],captured_unix=999.)
        api.require_quiet(proof,1000.)
        for field,value in (("active_model_workers",[1]),("active_cost_workers",[2]),("captured_unix",800.)):
            bad=dict(proof); bad[field]=value
            with self.assertRaises(ValueError): api.require_quiet(bad,1000.)

    def test_model_config_and_weight_blob_identity_are_both_required(self):
        api=load(); self.assertTrue(hasattr(api,"check_model_snapshot"),"model config/weight guard missing")
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); model=root/"snapshot"; model.mkdir(); records=[]
            for name,payload in (("config.json",b"config"),("model.safetensors",b"weight")):
                path=model/name; path.write_bytes(payload)
                records.append(dict(snapshot_relative_path="snapshot/"+name,resolved_blob_relative_path="snapshot/"+name,sha256=api.sha(path),bytes=len(payload)))
            api.check_model_snapshot(root,records)
            (model/"config.json").write_bytes(b"mutant")
            with self.assertRaises(ValueError): api.check_model_snapshot(root,records)

    def test_saved_prediction_identity_cannot_swap_prefix_or_model(self):
        api=load(); self.assertTrue(hasattr(api,"validate_saved_identity"),"saved prefix model guard missing")
        condition=api.conditions()[0]; output=api.outputs()[0]; receipt=dict(artifact_sha256="a"*64,model_state_sha="b"*64)
        sources={"adapter":"c"*64}; spec=api.route_spec(output,condition,receipt)
        manifest=dict(run_id=output["run_id"],split="V_select",cohort="beijing:primary:V_select:stride48",source_sha=api.fingerprint(sources),
            config_sha=api.fingerprint(spec),actual_route_spec=spec,artifact_sha256=receipt["artifact_sha256"],scaler_sha=api.SCALERS["refit"],data_sha=api.DATA_SHA,
            parent_run_id=condition["run_id"],latent_k=output["latent_k"],variant=output["variant"],model_sha=api.fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"],latent_k=output["latent_k"],variant=output["variant"])))
        api.validate_saved_identity(manifest,output,condition,receipt,sources)
        for key,value in (("model_sha","f"*64),("latent_k",11),("variant","posthoc"),("cohort","T_final")):
            changed=copy.deepcopy(manifest); changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): api.validate_saved_identity(changed,output,condition,receipt,sources)

    def test_cost_requires_both_exact_per_prefix_successful_result_proofs(self):
        api=load(); self.assertTrue(hasattr(api,"require_prior_proofs"),"validation prerequisite guard missing")
        output=api.outputs()[0]; receipt=dict(artifact_sha256="a"*64); manifest_sha="b"*64
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); proofs={}
            for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
                value=dict(run_id=output["run_id"],mode=mode,status=status,error_class=None,postflight_error_class=None,
                    scientific_result=dict(status=status,artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha,executed_now=True))
                path=root/(mode+".json"); api.new(path,value); proofs[mode]=(path,api.sha(path))
            api.require_prior_proofs(output,receipt,manifest_sha,proofs)
            with self.assertRaises(ValueError): api.require_prior_proofs(output,receipt,"c"*64,proofs)
            with self.assertRaises(ValueError): api.require_prior_proofs(output,receipt,manifest_sha,{"rescore":proofs["rescore"]})

    def test_execute_preserves_stage_budget_checkpoint_status_at_deadline(self):
        current=load()
        text=SOURCE.read_text(encoding="utf-8")
        branch='if result.get("status")!="BUDGET_LIMIT_PRESERVED": p1.check_deadline(deadline)'
        self.assertEqual(text.count(branch),1)
        old=types.ModuleType("before_deadline_fix"); old.__file__=str(SOURCE)
        exec(compile(text.replace(branch,'p1.check_deadline(deadline)'),str(SOURCE),"exec"),old.__dict__)

        def exercise(api):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); public=root/"public"; cache=root/"cache"; asset=root/"asset"
                for folder in (public,cache,asset,cache/api.NAME): folder.mkdir(parents=True,exist_ok=True)
                output=public/api.NAME/"diagnostics"/"budget_result"
                def expired_deadline(_): raise TimeoutError("Synthetic deadline reached after saved optimizer boundary")
                priority=types.SimpleNamespace(require_environment=lambda:None,check_deadline=expired_deadline)
                seed=types.SimpleNamespace(resource_preflight=lambda *args:0)
                model=types.SimpleNamespace(parameters=lambda:[])
                foundation=types.SimpleNamespace(model=model,row_microbatch=128,checkpoint_inputs=False,
                    instance_norm_source_sha256="normalization",assert_frozen_unchanged=lambda:None)
                pool=types.SimpleNamespace(get=lambda horizon:foundation,close=lambda:None)
                fake_modules={
                    "psutil":types.SimpleNamespace(Process=lambda:types.SimpleNamespace(create_time=lambda:123.)),
                    "numpy":types.SimpleNamespace(),
                    "data_contract":types.SimpleNamespace(__file__=str(public/"data_contract.py")),
                    "evaluation":types.SimpleNamespace(),
                    "torch":types.SimpleNamespace(set_num_threads=lambda count:None,cuda=types.SimpleNamespace(is_initialized=lambda:False)),
                    "fit_base":types.SimpleNamespace(),"training":types.SimpleNamespace(),
                    "deployment":types.SimpleNamespace(FoundationPool=lambda *args:pool),
                    "gpu_runtime":types.SimpleNamespace(state_hash=lambda module:api.FOUNDATION_SHA),
                }
                metadata_result=(dict(runner_sha256="a"*64),dict(model_asset_relative_path="snapshot",instance_norm_source_sha256="normalization"),dict(priority=priority,seed=seed))
                saved=dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=dict(updates=12,exposures=192))
                with mock.patch.dict(sys.modules,fake_modules), \
                        mock.patch.object(api.time,"time",return_value=api.DEADLINE-1), \
                        mock.patch.object(api,"metadata",return_value=metadata_result), \
                        mock.patch.object(api,"read",return_value={}), \
                        mock.patch.object(api,"safe_task",return_value=({},dict(status="SYNTHETIC_NO_DATA"))), \
                        mock.patch.object(api,"resume_and_save",return_value=saved):
                    try:
                        value=api.execute(public,cache,asset,"sealed",api.conditions()[0]["run_id"],"resume",api.DEADLINE,output)
                    except RuntimeError:
                        value=None
                receipt=json.loads((output/"RESULT.json").read_text(encoding="utf-8"))
                return value,receipt

        before,before_receipt=exercise(old)
        self.assertIsNone(before)
        self.assertEqual((before_receipt["status"],before_receipt["error_class"]),("FAILED_PRESERVED","TimeoutError"))
        after,after_receipt=exercise(current)
        self.assertEqual(after["status"],"BUDGET_LIMIT_PRESERVED")
        self.assertEqual((after_receipt["status"],after_receipt["error_class"]),("BUDGET_LIMIT_PRESERVED",None))
        self.assertEqual(after_receipt["scientific_result"]["summary"],dict(updates=12,exposures=192))


if __name__=="__main__": unittest.main()
