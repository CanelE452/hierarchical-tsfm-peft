"""Small stdlib fixtures; real source data/models are never loaded."""
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

SOURCE = Path(__file__).with_name("multik_independent_runner.py")
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location("prepared_independent", SOURCE)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
else:
    runner = None


class IndependentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "Independent prefix adapter is not implemented")

    def sealed(self):
        value = runner.draft_plan()
        value.update(status="SEALED_BY_ROOT", runner_sha256="a"*64,
                     ROOT_after_N4_report_and_main_admitted=True)
        return value

    def test_default_and_missing_trigger_do_no_IO_or_scientific_import(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=AssertionError("input read")), contextlib.redirect_stdout(output):
            self.assertEqual(runner.main([]), 0)
        receipt = json.loads(output.getvalue())
        self.assertEqual((receipt["input_reads"], receipt["writes"], receipt["actual_training"]), (0,0,0))
        self.assertFalse(receipt["numpy_imported"] or receipt["torch_imported"])
        with patch("builtins.open", side_effect=AssertionError("input read")), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): runner.main(["--execute"])

    def test_new_seal_and_positive_admission_are_required(self):
        with self.assertRaises(ValueError): runner.validate_plan(runner.draft_plan())
        runner.validate_plan(self.sealed())
        for change in ({"ROOT_after_N4_report_and_main_admitted":False}, {"runner_sha256":None}):
            with self.assertRaises(ValueError): runner.validate_plan(dict(self.sealed(), **change))

    def test_ROOT_sealed_PLAN_remains_valid_after_JSON_serialization(self):
        runner.validate_plan(json.loads(json.dumps(self.sealed())))

    def test_execute_preserves_saved_partial_when_final_deadline_has_expired(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);public,cache,asset=root/"p",root/"c",root/"a"
            for p in (public,cache,asset,cache/runner.NAME):p.mkdir(parents=True,exist_ok=True)
            output=public/runner.NAME/"diagnostics"/"partial";condition=runner.conditions()[0]
            def expired(_):raise TimeoutError("Synthetic deadline after saved boundary")
            priority=types.SimpleNamespace(require_environment=lambda:None,check_deadline=expired)
            foundation=types.SimpleNamespace(model=types.SimpleNamespace(parameters=lambda:[]),row_microbatch=128,checkpoint_inputs=False,instance_norm_source_sha256="norm",assert_frozen_unchanged=lambda:None)
            pool=types.SimpleNamespace(get=lambda horizon:foundation,close=lambda:None,counters=lambda:dict(streams=0,physical_streams=0,physical_calls=0),instances_created=1)
            process=types.SimpleNamespace(create_time=lambda:123.,cpu_times=lambda:(0.,0.),memory_info=lambda:types.SimpleNamespace(rss=0,peak_wset=0))
            modules={"psutil":types.SimpleNamespace(Process=lambda:process),"numpy":types.SimpleNamespace(),
                "data_contract":types.SimpleNamespace(__file__=str(public/"data_contract.py")),"evaluation":types.SimpleNamespace(),"fit_base":types.SimpleNamespace(),
                "torch":types.SimpleNamespace(set_num_threads=lambda count:None,cuda=types.SimpleNamespace(is_initialized=lambda:False)),
                "deployment":types.SimpleNamespace(FoundationPool=lambda *args:pool),"gpu_runtime":types.SimpleNamespace(state_hash=lambda model:"f")}
            original=dict(task_contract={},foundation_tensor_sha256="f",model_asset_relative_path="snapshot",instance_norm_source_sha256="norm")
            metadata_result=(dict(runner_sha256="a"*64),original,condition,dict(priority=priority,common=types.SimpleNamespace(),seed=types.SimpleNamespace(resource_preflight=lambda *args:0)))
            saved=dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=dict(updates=12,exposures=192))
            with patch.dict(sys.modules,modules),patch.object(runner.sys,"path",list(sys.path)),patch.object(runner.time,"time",return_value=runner.DEADLINE-1),patch.object(runner,"metadata",return_value=metadata_result),patch.object(runner,"read",return_value={}),patch.object(runner,"safe_task",return_value=({},dict(status="SYNTHETIC_NO_DATA"))),patch.object(runner,"fit_and_save",return_value=saved):
                value=runner.execute(public,cache,asset,"sealed",condition["run_id"],"fit",runner.DEADLINE,output)
            receipt=json.loads((output/"RESULT.json").read_text())
            self.assertEqual((value["status"],receipt["status"],receipt["error_class"]),("BUDGET_LIMIT_PRESERVED","BUDGET_LIMIT_PRESERVED",None))
            self.assertEqual(receipt["scientific_result"]["summary"],dict(updates=12,exposures=192))

    def test_registered_single_K_recipes_reject_grid_seed_or_family_changes(self):
        rows = runner.conditions()
        self.assertEqual([r["run_id"] for r in rows], [
            "postT_cpu_beijing_E4_INDEPENDENT_B0_k3_s92601_v1",
            "postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",
            "postT_cpu_beijing_E4_INDEPENDENT_B0_k11_s92601_v1"])
        for change in ({"k_grid":[3]}, {"family":"INDEPENDENT_B0"}, {"seed":92602}, {"latent_k":9}):
            plan = self.sealed(); plan["conditions"][0]["spec"].update(change)
            with self.assertRaises(ValueError): runner.validate_plan(plan)
        with self.assertRaises(ValueError): runner.choose(self.sealed(), "primary_beijing_E1_B0_k3_s92601")

    def test_old_or_new_partial_and_claims_cannot_be_overwritten(self):
        for namespace,kind in ((runner.OLD,"runs"),(runner.NAME,"runs"),(runner.OLD,"predictions"),(runner.NAME,"claims")):
            with tempfile.TemporaryDirectory() as directory:
                public,cache = Path(directory)/"p",Path(directory)/"c"; public.mkdir(); cache.mkdir()
                ident=runner.conditions()[0]["run_id"]
                path=cache/namespace/kind/("fit_"+ident+".json" if kind=="claims" else ident)
                path.parent.mkdir(parents=True); path.write_text("preserve")
                with self.assertRaises(FileExistsError): runner.require_unstarted(public,cache,ident)
                self.assertEqual(path.read_text(),"preserve")

    def test_canonical_route_uses_new_artifact_with_original_single_K_and_A60(self):
        for condition,k in zip(runner.conditions(),(3,6,11)):
            route=runner.canonical_route(condition,{"artifact_sha256":"a"*64})
            self.assertEqual((route["family"],route["latent_k"],route["canonical_fit_end"]),("B0",k,12622))
            self.assertEqual(route["artifact_cache_relative_path"],runner.NAME+"/runs/"+condition["run_id"]+"/selected_bundle.pt")
            self.assertNotIn("k_grid",route)

    def test_counters_are_dynamic_K_and_batch_instead_of_hardcoded_K6(self):
        for k,streams in ((3,129),(6,258),(11,473)):
            self.assertEqual(runner.expected_counters(k,[4]*10+[3]),dict(streams=streams,physical_streams=streams,physical_calls=11))
        self.assertEqual(runner.expected_counters(3,[4,2]),dict(streams=18,physical_streams=18,physical_calls=2))
        self.assertEqual(runner.expected_counters(11,[8]),dict(streams=88,physical_streams=88,physical_calls=1))
        with self.assertRaises(ValueError): runner.expected_counters(9,[8])

    def test_exact_V_origins_and_original_prefix_contract(self):
        runner.validate_origins([14726,16742])
        for origins in ([14725],[16783],[16830],[]):
            with self.assertRaises(ValueError): runner.validate_origins(origins)
        common=runner.module(SOURCE.parents[1]/"priority_valid_family_20261016_v1/family_minimum_runner.py","tiny_independent_bound")
        original=dict(shape=[21038,11],data_sha256="old")
        task=dict(physical=types.SimpleNamespace(shape=(16830,11)),data_sha256="old",split_ranges=copy.deepcopy(runner.BOUNDS))
        common.validate_prefix_shape(task,original,"beijing")
        task["split_ranges"]["T_final"]=[16830,16830]
        with self.assertRaises(ValueError): common.validate_prefix_shape(task,original,"beijing")

    def test_PCA_is_fit_at_each_stage_and_full_C_remains_identity(self):
        class NP:
            float32="f4"
            @staticmethod
            def array_equal(a,b): return a==b
            @staticmethod
            def eye(n,dtype=None): return [[int(i==j) for j in range(n)] for i in range(n)]
            @staticmethod
            def zeros(n,dtype=None): return [0]*n
        calls=[]
        def pca(task,end,k,mean,std):
            calls.append((end,k)); return dict(E=[[1,0,0]]*11,pca_center=[.5]*11)
        data=types.SimpleNamespace(initialize_pca=pca)
        norm=dict(fit_end=10519,mean="m",std="s",basis_initialization="A_fitted_centered_PCA",basis=[[1,0,0]]*11,pca_center=[.5]*11)
        runner.validate_initialization(norm,3,10519,{},data,NP)
        self.assertEqual(calls,[(10519,3)])
        with self.assertRaises(ValueError): runner.validate_initialization(dict(norm,pca_center=[0]*11),3,10519,{},data,NP)
        identity=dict(fit_end=12622,basis_initialization="identity_at_K_equals_C",basis=NP.eye(11),pca_center=[0]*11)
        runner.validate_initialization(identity,11,12622,{},data,NP)
        with self.assertRaises(ValueError): runner.validate_initialization(dict(identity,basis_initialization="A_fitted_centered_PCA"),11,12622,{},data,NP)

    def test_actual_route_rejects_foreign_selected_bundle_and_closes(self):
        condition=runner.conditions()[0]
        receipt=dict(artifact_sha256="a"*64,model_state_sha="s",refit_scaler_sha="scale")
        bundle=dict(spec=copy.deepcopy(condition["spec"]),model_state_sha="s",plan_sha256="plan",contract=dict(data_sha="d",model_sha="f"),normalization=dict(fit_end=12622))
        route=types.SimpleNamespace(family="B0",latent_k=3,bundle=bundle,mean="m",std="s",closed=False)
        route.close=lambda:setattr(route,"closed",True)
        pool=types.SimpleNamespace(foundation=types.SimpleNamespace(assert_frozen_unchanged=lambda:None))
        context=dict(cache=Path("cache"),plan_sha="plan",foundation_sha="f",original=dict(data_sha256="d"),
            deployment=types.SimpleNamespace(Route=lambda *args,**kwargs:route),fit=types.SimpleNamespace(scaler_fingerprint=lambda *args:"scale"))
        self.assertIs(runner.actual_route({"receipt":receipt},condition,{},pool,context),route)
        route.bundle["spec"]["k_grid"]=[3]
        with self.assertRaises(ValueError): runner.actual_route({"receipt":receipt},condition,{},pool,context)
        self.assertTrue(route.closed)

    def test_manifest_cost_binding_reads_no_forecast_archive(self):
        c=runner.conditions()[0];receipt=dict(artifact_sha256="a",refit_scaler_sha="s",model_state_sha="m")
        fingerprints={k:dict(sha256="fp") for k in ("target_normalized","observed","origins","channel_ids")}
        context=dict(original=dict(data_sha256="d"),new_sources={},reference=dict(arrays=fingerprints))
        manifest=dict(run_id=c["run_id"],family="B0",seed=92601,split="V_select",cohort="beijing:primary:V_select:stride48",device="cpu",data_sha="d",artifact_sha256="a",scaler_sha="s",source_sha=runner.fingerprint({}),config_sha=runner.fingerprint(c["spec"]),model_sha=runner.fingerprint(dict(artifact_sha="a",model_state_sha="m")),arrays=dict(fingerprints,pred_normalized=dict(shape=[43,48,11],dtype="float32")),score=dict(n_origins=43,n_observed=20539,num_channels=11,horizon=48))
        with patch("builtins.open",side_effect=AssertionError("forecast read")): runner.validate_saved_manifest(manifest,c,receipt,context)
        for change in ({"device":"cuda"},{"artifact_sha256":"foreign"},{"split":"T_final"}):
            with self.assertRaises(ValueError):runner.validate_saved_manifest(dict(manifest,**change),c,receipt,context)

    def test_quiet_proof_rejects_busy_or_stale_context(self):
        good=dict(status="QUIET_CPU_COST_WINDOW",active_model_workers=[],active_cost_workers=[],captured_unix=100.)
        runner.require_quiet(good,110.)
        for change in ({"active_model_workers":[1]},{"active_cost_workers":[1]},{"captured_unix":-20.1}):
            with self.assertRaises(ValueError):runner.require_quiet(dict(good,**change),110.)

    def test_cost_requires_successful_matching_actual_rescore_and_replay(self):
        common=runner.module(SOURCE.parents[1]/"priority_valid_family_20261016_v1/family_minimum_runner.py","tiny_independent_proofs")
        with tempfile.TemporaryDirectory() as directory:
            proofs={};c=runner.conditions()[0];receipt=dict(artifact_sha256="a")
            for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
                p=Path(directory)/(mode+".json")
                runner.new(p,dict(run_id=c["run_id"],mode=mode,status=status,error_class=None,postflight_error_class=None,scientific_result=dict(status=status,artifact_sha256="a",manifest_sha256="manifest",executed_now=True)))
                proofs[mode]=(p,runner.sha(p))
            common.verify_cost_prerequisites(c,receipt,"manifest",proofs)
            with self.assertRaises(ValueError):common.verify_cost_prerequisites(c,receipt,"foreign",proofs)
            proofs["replay"][0].write_text("changed")
            with self.assertRaises(ValueError):common.verify_cost_prerequisites(c,receipt,"manifest",proofs)

    def test_original_stage_boundaries_select_zero_without_refit_updates(self):
        common=runner.module(SOURCE.parents[1]/"priority_valid_family_20261016_v1/family_minimum_runner.py","tiny_independent_stages")
        calls=[]
        def stage(spec,task,end,stop,epochs,*args):
            calls.append((end,stop,epochs));return dict(summary=dict(status="SEGMENT_COMPLETE",selected_epoch=0.))
        result=common.fit_stages(types.SimpleNamespace(_stage=stage),runner.conditions()[0],{},{},{},Path("p"),Path("c"),1.)
        self.assertEqual(calls,[(10519,[10519,12622],[.25,1.,3.]),(12622,None,[])])
        self.assertEqual(result["selected_epoch"],0.)

    def test_write_exclusive_and_changed_pins_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"new.json";runner.new(p,{"old":1});digest=runner.sha(p)
            with self.assertRaises(FileExistsError):runner.new(p,{"bad":2})
            runner.check_pins(directory,{"new.json":digest});p.write_text("mutation")
            with self.assertRaises(ValueError):runner.check_pins(directory,{"new.json":digest})


if __name__=="__main__":unittest.main()
