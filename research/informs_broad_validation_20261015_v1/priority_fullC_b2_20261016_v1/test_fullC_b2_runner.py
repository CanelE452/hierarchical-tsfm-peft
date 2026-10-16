"""Small stdlib fixtures only; no source data, model, checkpoint or experiment."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("fullC_b2_runner.py")
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("prepared_fullC_b2",SOURCE)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
else:runner=None


class FullCSafetyTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(runner,"Full-C prefix adapter not implemented")

    def sealed(self):
        plan=runner.draft_plan();plan.update(status="SEALED_BY_ROOT",runner_sha256="a"*64,ROOT_after_N4_report_and_main_admitted=True)
        return plan

    def test_default_and_untriggered_cli_read_nothing(self):
        stream=io.StringIO()
        with patch("builtins.open",side_effect=AssertionError("Input read")),contextlib.redirect_stdout(stream):self.assertEqual(runner.main([]),0)
        value=json.loads(stream.getvalue());self.assertEqual((value["input_reads"],value["writes"],value["actual_new_training_conditions"]),(0,0,0))
        self.assertFalse(value["numpy_imported"] or value["torch_imported"])
        with patch("builtins.open",side_effect=AssertionError("Input read")),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):runner.main(["--execute"])

    def test_seal_requires_positive_post_N4_ROOT_admission(self):
        with self.assertRaises(ValueError):runner.validate_plan(runner.draft_plan())
        runner.validate_plan(self.sealed())
        for key,value in (("runner_sha256",None),("ROOT_after_N4_report_and_main_admitted",False)):
            with self.assertRaises(ValueError):runner.validate_plan(dict(self.sealed(),**{key:value}))

    def test_only_original_two_fullC_recipes_are_admitted(self):
        rows=runner.conditions();self.assertEqual([row["latent_k"] for row in rows],[21,11])
        self.assertEqual([row["spec"]["stage_label"] for row in rows],["postT_cpu_fresh_V_only_remaining_core29","postT_cpu_fresh_V_only_required_minima"])
        self.assertEqual(rows[0]["spec"]["provenance"]["bitwise_training_equivalence_claimed"],False)
        for change in ({"latent_k":6},{"k_grid":[11,21]},{"seed":92602},{"device":"cuda"}):
            plan=self.sealed();plan["conditions"][0]["spec"].update(change)
            with self.assertRaises(ValueError):runner.validate_plan(plan)
        with self.assertRaises(ValueError):runner.choose(self.sealed(),"primary_jena_E1_B2_k21_s92601")

    def test_original_CPU_namespaces_and_new_partial_are_preserved(self):
        for namespace in (runner.CORE,runner.REQUIRED,runner.NAME):
            with tempfile.TemporaryDirectory() as folder:
                public=Path(folder)/"p";cache=Path(folder)/"c";public.mkdir();cache.mkdir()
                identifier=runner.conditions()[0]["run_id"];path=cache/namespace/"runs"/identifier;path.mkdir(parents=True)
                with self.assertRaises(FileExistsError):runner.require_unstarted(public,cache,identifier)

    def test_explicit_new_namespace_route_has_fullC_width_and_A60(self):
        for condition,end in zip(runner.conditions(),(12579,12622)):
            route=runner.canonical_route(condition,{"artifact_sha256":"a"*64})
            self.assertEqual((route["family"],route["latent_k"],route["canonical_fit_end"]),("B2",condition["latent_k"],end))
            self.assertEqual(route["artifact_cache_relative_path"],runner.NAME+"/runs/"+condition["run_id"]+"/selected_bundle.pt")

    def test_fullC_streams_and_row128_splits_are_not_fixed_K6(self):
        self.assertEqual(runner.expected_counters(21,[4]*10+[3]),dict(streams=903,physical_streams=903,physical_calls=11))
        self.assertEqual(runner.expected_counters(11,[4]*10+[3]),dict(streams=473,physical_streams=473,physical_calls=11))
        self.assertEqual(runner.expected_counters(21,[4,2]),dict(streams=126,physical_streams=126,physical_calls=2))
        self.assertEqual(runner.expected_counters(11,[4,2]),dict(streams=66,physical_streams=66,physical_calls=2))
        self.assertEqual(runner.expected_counters(21,[8]),dict(streams=168,physical_streams=168,physical_calls=2))
        self.assertEqual(runner.expected_counters(11,[8]),dict(streams=88,physical_streams=88,physical_calls=1))

    def test_identity_initialization_rejects_fullC_PCA_rotation(self):
        class NP:
            float32="float32"
            def eye(self,width,dtype=None):return [[int(i==j) for j in range(width)] for i in range(width)]
            def zeros(self,width,dtype=None):return [0]*width
            def array_equal(self,left,right):return left==right
        np=NP();normalization=dict(basis_initialization="identity_at_K_equals_C",basis=np.eye(2),pca_center=[0,0])
        runner.validate_initialization(normalization,2,np)
        for change in ({"basis_initialization":"A_fitted_centered_PCA"},{"basis":[[0,1],[1,0]]},{"pca_center":[1,0]}):
            with self.assertRaises(ValueError):runner.validate_initialization(dict(normalization,**change),2,np)

    def test_actual_route_rejects_foreign_plan_or_state_and_closes(self):
        condition=runner.conditions()[0];receipt=dict(artifact_sha256="a"*64,model_state_sha="b"*64,refit_scaler_sha="s")
        bundle=dict(spec=copy.deepcopy(condition["spec"]),model_state_sha="b"*64,plan_sha256="sealed",contract=dict(data_sha="d",model_sha="f"),normalization=dict(fit_end=12579))
        class Foundation:
            def assert_frozen_unchanged(self):pass
        class Route:
            family="B2";latent_k=21;mean="mean";std="std"
            def __init__(self):self.bundle=copy.deepcopy(bundle);self.closed=False
            def close(self):self.closed=True
        route=Route();calls=[]
        class Deployment:
            def Route(self,spec,*args,**kwargs):calls.append(spec);return route
        context=dict(deployment=Deployment(),cache=Path("cache"),plan_sha="sealed",original=dict(data_sha256="d"),foundation_sha="f",fit=types.SimpleNamespace(scaler_fingerprint=lambda *args:"s"))
        self.assertIs(runner.actual_route({"receipt":receipt},condition,{},types.SimpleNamespace(foundation=Foundation()),context),route)
        self.assertIn(runner.NAME,calls[0]["artifact_cache_relative_path"])
        route.bundle["plan_sha256"]="foreign"
        with self.assertRaises(ValueError):runner.actual_route({"receipt":receipt},condition,{},types.SimpleNamespace(foundation=Foundation()),context)
        self.assertTrue(route.closed)

    def test_saved_output_is_write_exclusive_and_SHA_changes_detectably(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"receipt.json";runner.new(path,{"old":1});before=runner.sha(path)
            with self.assertRaises(FileExistsError):runner.new(path,{"old":2})
            self.assertEqual(json.loads(path.read_text()),{"old":1});path.write_text("changed")
            self.assertNotEqual(runner.sha(path),before)

    def test_metadata_header_allows_no_extra_K_or_T_mode(self):
        plan=self.sealed();self.assertEqual(plan["maximum_new_training_conditions"],2)
        self.assertFalse(plan["protected_tail_numeric_parse_allowed"] or plan["new_T_target_operations_allowed"])
        self.assertEqual(plan["materialized_prefix_rows"],dict(jena=16772,beijing=16830))
        self.assertEqual(plan["original_primary_split_ranges"]["jena"]["T_final"],[16772,20965])
        self.assertFalse(plan["probe_mode_implemented"])

    def test_quiet_requires_two_empty_arrays_and_current_timestamp(self):
        proof=dict(status="QUIET_CPU_COST_WINDOW",active_model_workers=[],active_cost_workers=[],captured_unix=100.)
        runner.require_quiet(proof,101.)
        for change in ({"active_model_workers":[1]},{"active_cost_workers":[1]},{"captured_unix":-21.}):
            with self.assertRaises(ValueError):runner.require_quiet(dict(proof,**change),101.)

    def test_reused_stage_helper_keeps_original_bounds_and_segment_updates(self):
        common=runner.module(SOURCE.parents[1]/"priority_valid_family_20261016_v1/family_minimum_runner.py","tiny_fullC_stage_helper")
        class Fit:
            def __init__(self):self.calls=[]
            def _stage(self,spec,task,end,stop,epochs,*args):
                self.calls.append((end,stop,epochs));return dict(summary=dict(status="SEGMENT_COMPLETE",selected_epoch=.25))
        for condition,want,nsearch,search_updates,nrefit,refit_exposures in (
                (runner.conditions()[0],[(10482,[10482,12579],[.25,1.,3.]),(12579,None,[.25])],9923,1863,12020,3005),
                (runner.conditions()[1],[(10519,[10519,12622],[.25,1.,3.]),(12622,None,[.25])],9960,1868,12063,3016)):
            fit=Fit();common.fit_stages(fit,condition,{},{},{},Path("p"),Path("c"),1.)
            self.assertEqual(fit.calls,want)
            search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=nsearch,exposures=3*nsearch,updates=search_updates,selected_epoch=.25)
            refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25],train_origins=nrefit,exposures=refit_exposures,updates=188 if nrefit==12020 else 189)
            common.validate_stages(condition,search,refit)
            with self.assertRaises(ValueError):common.validate_stages(condition,dict(search,updates=1861),refit)

    def test_prefix_shape_helper_never_recomputes_original_T_bounds(self):
        common=runner.module(SOURCE.parents[1]/"priority_valid_family_20261016_v1/family_minimum_runner.py","tiny_fullC_boundary_helper")
        for condition,rows,width,permitted in ((runner.conditions()[0],16772,21,20965),(runner.conditions()[1],16830,11,21038)):
            bounds=copy.deepcopy(runner.BOUNDS[condition["dataset"]]);original=dict(shape=[permitted,width],primary_split_ranges=bounds,data_sha256="a")
            task=dict(physical=types.SimpleNamespace(shape=(rows,width)),split_ranges=copy.deepcopy(bounds),data_sha256="a")
            common.validate_prefix_shape(task,original,condition["dataset"])
            task["split_ranges"]["T_final"]=[rows-100,rows]
            with self.assertRaises(ValueError):common.validate_prefix_shape(task,original,condition["dataset"])

    def test_cost_manifest_can_bind_safe_fullV_without_reading_forecasts(self):
        self.assertTrue(callable(getattr(runner,"validate_saved_manifest",None)),"Metadata-only cost identity validator required")
        condition=runner.conditions()[0];receipt=dict(artifact_sha256="a"*64,model_state_sha="b"*64,refit_scaler_sha="s")
        four={key:dict(sha256="c"*64) for key in ("target_normalized","observed","origins","channel_ids")}
        context=dict(original=dict(data_sha256="d"),new_sources={},reference=dict(arrays=four))
        manifest=dict(run_id=condition["run_id"],family="B2",seed=92601,split="V_select",cohort="jena:primary:V_select:stride48",device="cpu",data_sha="d",artifact_sha256="a"*64,scaler_sha="s",source_sha=runner.fingerprint({}),config_sha=runner.fingerprint(condition["spec"]),model_sha=runner.fingerprint(dict(artifact_sha="a"*64,model_state_sha="b"*64)),
            arrays=dict(four,pred_normalized=dict(shape=[43,48,21],dtype="float32")),score=dict(n_origins=43,n_observed=43344,num_channels=21,horizon=48))
        with patch("builtins.open",side_effect=AssertionError("Forecast archive read")):runner.validate_saved_manifest(manifest,condition,receipt,context)
        for change in ({"artifact_sha256":"foreign"},{"device":"cuda"},{"cohort":"jena:primary:T_final:stride48"}):
            with self.assertRaises(ValueError):runner.validate_saved_manifest(dict(manifest,**change),condition,receipt,context)


    def test_execute_preserves_budget_partial_after_deadline(self):
        text=SOURCE.read_text(encoding="utf-8")
        branch='if result.get("status")!="BUDGET_LIMIT_PRESERVED":p1.check_deadline(deadline)'
        old=types.ModuleType("fullC_before_deadline_fix");old.__file__=str(SOURCE)
        exec(compile(text.replace(branch,"p1.check_deadline(deadline)"),str(SOURCE),"exec"),old.__dict__)

        def exercise(api):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);public=root/"public";cache=root/"cache";asset=root/"asset"
                for folder in (public,cache,asset,cache/api.NAME):folder.mkdir(parents=True,exist_ok=True)
                output=public/api.NAME/"diagnostics"/"budget_result"
                condition=api.conditions()[0]
                def expired_deadline(_):raise TimeoutError("Synthetic deadline after saved optimizer boundary")
                priority=types.SimpleNamespace(require_environment=lambda:None,check_deadline=expired_deadline)
                seed=types.SimpleNamespace(resource_preflight=lambda *args:0)
                model=types.SimpleNamespace(parameters=lambda:[])
                foundation=types.SimpleNamespace(model=model,row_microbatch=128,checkpoint_inputs=False,
                    instance_norm_source_sha256="normalization",assert_frozen_unchanged=lambda:None)
                pool=types.SimpleNamespace(get=lambda horizon:foundation,close=lambda:None,
                    counters=lambda:dict(streams=0,physical_streams=0,physical_calls=0),instances_created=1)
                process=types.SimpleNamespace(create_time=lambda:123.,cpu_times=lambda:(0.,0.),
                    memory_info=lambda:types.SimpleNamespace(rss=0,peak_wset=0))
                fake_modules={
                    "psutil":types.SimpleNamespace(Process=lambda:process),"numpy":types.SimpleNamespace(),
                    "data_contract":types.SimpleNamespace(__file__=str(public/"data_contract.py")),
                    "evaluation":types.SimpleNamespace(),"fit_base":types.SimpleNamespace(),
                    "torch":types.SimpleNamespace(set_num_threads=lambda count:None,cuda=types.SimpleNamespace(is_initialized=lambda:False)),
                    "deployment":types.SimpleNamespace(FoundationPool=lambda *args:pool),
                    "gpu_runtime":types.SimpleNamespace(state_hash=lambda module:"f"*64),
                }
                old_plan=dict(task_contracts={"jena":{}},foundation_tensor_sha256="f"*64,
                    model_asset_relative_path="snapshot",instance_norm_source_sha256="normalization")
                metadata_result=(dict(runner_sha256="a"*64),old_plan,condition,dict(priority=priority,seed=seed,common=types.SimpleNamespace()))
                saved=dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=dict(updates=12,exposures=192))
                with patch.dict(sys.modules,fake_modules),patch.object(api.sys,"path",list(sys.path)), \
                        patch.object(api.time,"time",return_value=api.DEADLINE-1), \
                        patch.object(api,"metadata",return_value=metadata_result),patch.object(api,"read",return_value={}), \
                        patch.object(api,"safe_task",return_value=({},dict(status="SYNTHETIC_NO_DATA"))), \
                        patch.object(api,"fit_and_save",return_value=saved):
                    try:value=api.execute(public,cache,asset,"sealed",condition["run_id"],"fit",api.DEADLINE,output)
                    except RuntimeError:value=None
                return value,json.loads((output/"RESULT.json").read_text(encoding="utf-8"))

        before,before_receipt=exercise(old)
        self.assertIsNone(before)
        self.assertEqual((before_receipt["status"],before_receipt["error_class"]),("FAILED_PRESERVED","TimeoutError"))
        after,after_receipt=exercise(runner)
        self.assertIsNotNone(after,"A preserved budget partial must survive the final deadline check")
        self.assertEqual((after_receipt["status"],after_receipt["error_class"]),("BUDGET_LIMIT_PRESERVED",None))
        self.assertEqual(after_receipt["scientific_result"]["summary"],dict(updates=12,exposures=192))

if __name__=="__main__":unittest.main()
