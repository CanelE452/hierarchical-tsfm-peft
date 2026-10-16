"""Small metadata fixtures only; no model, array, real replay or cost worker."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
HELPER = HERE/"prepare_multik_transition_costs.py"


def load():
    spec = importlib.util.spec_from_file_location("tiny_transition_cost_preparation",HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransitionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(),"The explicit CPU transition cost preparer is missing")
        self.h = load()
        self.cost = self.h.load_cost_preparer()
        self.v = self.cost.load_verifier()
        self.r = self.v.load_runner()
        fits,outputs = self.r.recipe_manifest(dict(chosen_family="B0",k_grid=[3,6,11]))
        self.plan = dict(fits=fits,outputs=outputs,task_contract=dict(primary_split_ranges=dict(A_stop=[10519,12622])))

    def test_ten_fresh_worker_scopes_and_exact_cpu_protocol(self):
        mapping = self.h.strategy_map(self.plan)
        self.assertEqual(set(mapping),set(self.h.STRATEGIES))
        self.assertEqual(len({identity for group in mapping.values() for identity in group.values()}),12)
        argv = self.h.worker_arguments("transition_specs/tiny/PLAN.json","costs/tiny/shared_B1",1,"shared","model/small")
        for flag,value in (("--public-root","{POST_T_PUBLIC_ROOT}"),("--cache-root","{POST_T_CACHE_ROOT}"),
            ("--device","cpu"),("--strategy","shared"),("--batch","1"),("--warmups","10"),("--timed-trials","20")):
            self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertNotIn("--make-plan",argv); self.assertNotIn("--stage-trials",argv)
        self.assertEqual(5*2,10)
        self.assertEqual(self.h.PROTOCOL,dict(directed_transitions=6,warmup_per_transition=10,timed_per_transition=20,
            total_trials_per_worker=180,timed_trials_per_worker=120,worker_scopes=10))
        with self.assertRaises(ValueError): self.h.worker_arguments("x","y",1,"all","m")

    def test_strategy_gate_requires_only_its_three_registered_outputs(self):
        chosen = self.h.selected_conditions(self.plan,"shared")
        self.assertEqual([r["latent_k"] for r in chosen],[3,6,11])
        self.assertTrue(all(row["variant"]=="shared" for row in chosen))
        fake = types.SimpleNamespace(readiness=mock.Mock(side_effect=lambda v,r,p,pu,pr,c:
            dict(ready=c["latent_k"]!=6,missing=[] if c["latent_k"]!=6 else ["manifest"])))
        state = self.h.strategy_readiness(fake,self.v,self.r,self.plan,Path("public"),Path("cache"),"shared")
        self.assertFalse(state["ready"])
        self.assertEqual(len(state["outputs"]),3)
        self.assertEqual(fake.readiness.call_count,3)
        self.assertTrue(state["other_strategies_are_not_a_gate"])

    def test_shared_max_and_independent_saved_training_grids_preserved(self):
        for strategy in self.h.STRATEGIES:
            self.assertEqual(len(self.h.selected_conditions(self.plan,strategy)),3)
        for strategy,role,grid in (("shared","ORDERED_MULTI_K",[11]),("posthoc","MAXK_PCA_TRAINED",[3,6,11]),
                                   ("independent_hot","INDEPENDENT_B0",[3])):
            bad = copy.deepcopy(self.plan)
            next(row for row in bad["fits"] if row["role"]==role)["spec"]["k_grid"] = grid
            with self.subTest(strategy=strategy),self.assertRaises(ValueError): self.h.selected_conditions(bad,strategy)
        bad = copy.deepcopy(self.plan)
        next(row for row in bad["outputs"] if row["variant"]=="shared")["latent_k"] = 4
        with self.assertRaises(ValueError): self.h.strategy_map(bad)

    def test_single_artifact_strategies_and_three_independent_artifacts(self):
        same = [dict(artifact_relative="basis",parent=dict(artifact_sha256="a")) for _ in range(3)]
        for strategy in ("shared","posthoc","nested"):
            self.assertEqual(self.h.validate_artifact_sharing(strategy,same)["unique_artifact_paths"],1)
        different = [dict(artifact_relative="weights"+str(k),parent=dict(artifact_sha256=str(k))) for k in (3,6,11)]
        self.assertEqual(self.h.validate_artifact_sharing("independent_hot",different)["unique_artifact_paths"],3)
        with self.assertRaises(ValueError): self.h.validate_artifact_sharing("independent_cold",same)
        with self.assertRaises(ValueError): self.h.validate_artifact_sharing("nested",different)

    def cpu_result(self):
        condition = self.h.selected_conditions(self.plan,"shared")[0]
        receipt = dict(status="PASS",run_id=condition["run_id"],max_abs=.000001)
        value = dict(status="PASS",run_id=condition["run_id"],mode="replay",plan_sha256=self.h.PLAN_SHA,
            verifier_sha256=self.cost.DEPENDENCIES["verify_multik_outputs.py"],scientific_result=dict(receipt,executed_now=True),
            actual_checkpoint_forward_in_this_command=True,checkpoint_replay_call_attempted=True,checkpoint_replay_receipt_returned=True,
            original_source_registry_selection_T_and_captured_inputs_preserved=True,new_T_target_or_scoring=False,
            saved_full_V_rescoring=False,subset_checkpoint_replay=True,training_rerun=False,cuda_initialized=False,
            full_V_origin_count=43,replay_origin_count=6,dataset="beijing",horizon=48,channels=11,latent_k=3,
            registered_role=condition["role"],output_kind=condition["kind"])
        return condition,receipt,value

    def test_cpu_evidence_requires_fresh_actual_frozen_verifier_and_identical_receipt(self):
        condition,receipt,value = self.cpu_result()
        self.h.validate_cpu_replay_result(self.cost,condition,receipt,value)
        for key,new in (("actual_checkpoint_forward_in_this_command",False),("status","FAILED"),
                        ("verifier_sha256","changed"),("cuda_initialized",True),("latent_k",11)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.h.validate_cpu_replay_result(self.cost,condition,receipt,dict(value,**{key:new}))
        changed = dict(value,scientific_result=dict(receipt,executed_now=False))
        with self.assertRaises(ValueError): self.h.validate_cpu_replay_result(self.cost,condition,receipt,changed)
        changed = dict(value,scientific_result=dict(receipt,executed_now=True,max_abs=.1))
        with self.assertRaises(ValueError): self.h.validate_cpu_replay_result(self.cost,condition,receipt,changed)

    def test_explicit_proof_index_sha_duplicates_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as td:
            public,cache = Path(td)/"public",Path(td)/"cache"
            public.mkdir(); cache.mkdir()
            rows = []
            for condition in self.h.selected_conditions(self.plan,"shared"):
                path = public/self.r.NAME/"diagnostics"/condition["run_id"]/"RESULT.json"
                path.parent.mkdir(parents=True); path.write_text("{}",encoding="utf-8")
                rows.append(dict(run_id=condition["run_id"],public_relative_path=path.relative_to(public).as_posix(),sha256=self.h.sha(path)))
            index = cache/"proofs.json"
            def write(values):
                index.write_text(json.dumps(dict(schema="MULTIK_CPU_REPLAY_PROOFS_v1",outputs=values)),encoding="utf-8")
            write(rows)
            value,bindings = self.h.load_proof_index(self.r,self.plan,public,cache,"shared",index,self.h.sha(index))
            self.assertEqual(len(value),3); self.assertEqual(len(bindings),4)
            with self.assertRaises(ValueError): self.h.load_proof_index(self.r,self.plan,public,cache,"shared",index,"0"*64)
            write(rows+[rows[0]])
            with self.assertRaises(ValueError): self.h.load_proof_index(self.r,self.plan,public,cache,"shared",index,self.h.sha(index))
            escaped = copy.deepcopy(rows); escaped[0]["public_relative_path"]="../outside/RESULT.json"
            write(escaped)
            with self.assertRaises(ValueError): self.h.load_proof_index(self.r,self.plan,public,cache,"shared",index,self.h.sha(index))

    def test_dependency_mutation_rejected_before_import(self):
        with mock.patch.object(self.h,"sha",return_value="changed"),mock.patch.object(self.h.importlib.util,"spec_from_file_location") as call:
            with self.assertRaises(ValueError): self.h.load_cost_preparer()
            call.assert_not_called()

    def test_existing_output_prevents_task_claim_and_worker_plan(self):
        with tempfile.TemporaryDirectory() as td:
            public,cache,asset = (Path(td)/name for name in ("public","cache","asset"))
            output = public/self.r.NAME/"diagnostics"/"existing"
            output.mkdir(parents=True)
            (output/"old").write_bytes(b"preserve")
            fake_runner = types.SimpleNamespace(NAME=self.r.NAME,environment=mock.Mock(),relative=self.r.relative)
            runtime = types.SimpleNamespace(task=mock.Mock(),route=mock.Mock())
            state = (self.cost,self.v,fake_runner,self.plan,dict(ready=True))
            with mock.patch.object(self.h,"metadata",return_value=state):
                with self.assertRaises(FileExistsError):
                    self.h.execute(public,cache,asset,self.h.PLAN_SHA,"shared",output,deadline=2.,cpu_replay_proofs=cache/"proof.json",cpu_replay_proofs_sha="0"*64,_runtime=runtime)
            runtime.task.assert_not_called(); runtime.route.assert_not_called()
            self.assertEqual((output/"old").read_bytes(),b"preserve")
            self.assertFalse((cache/self.r.NAME/"transition_cost_preparation_claims").exists())

    def test_default_cli_has_no_scientific_import_write_or_worker(self):
        result = subprocess.run([sys.executable,"-B",str(HELPER)],capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["numpy_imported"]); self.assertFalse(value["torch_imported"]); self.assertFalse(value["psutil_imported"])
        self.assertEqual(value["files_written"],0); self.assertEqual(value["actual_scientific_or_child_operations"],0)
        self.assertEqual(value["whole_protocol_worker_scopes"],10)


if __name__ == "__main__": unittest.main()
