"""Tiny CPU-cost preparation fixtures; no models, arrays, workers or real fits."""
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
HELPER = HERE / "prepare_multik_cpu_costs.py"


def load():
    spec = importlib.util.spec_from_file_location("tiny_multik_cpu_cost_preparation", HELPER)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CostPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "The explicit multi-K CPU cost preparer is missing")
        self.h = load()
        self.v = self.h.load_verifier()
        self.r = self.v.load_runner()
        self.plan = dict(foundation_tensor_sha256="foundation", model_asset_relative_path="model/small",
            task_contract=dict(primary_split_ranges=dict(A_stop=[10519, 12622])),
            source39={name: name+"-sha" for name in self.h.REPLAY_SOURCES})

    def binding(self, kind, k=3):
        c = dict(run_id="tiny_"+kind, kind=kind, role=kind, latent_k=k, parent_run_id="parent", variant="shared")
        if kind == "nested": c.update(parent_run_id=None, variant="nested_pca")
        base = dict(artifact_sha256="weights", model_state_sha="state", refit_scaler_sha="scaler",
                    artifact_cache_relative_path="runs/parent/selected_bundle.pt")
        nested = dict(artifact_sha256="basis", source_sha="basis-source", data_sha="data", fit_end=12622)
        folder = "predictions/tiny_independent/V_select" if kind == "independent" else "ordered_predictions/tiny_"+kind
        manifest = dict(model_sha="saved-V-model", archive_sha256="saved-V-bytes", cache_relative_path="private-v.npz")
        parent = nested if kind == "nested" else base
        binding = dict(condition=c, manifest=manifest, parent=parent, manifest_relative=folder+"/PREDICTION_MANIFEST.json",
            manifest_sha256="manifest-bytes", receipt_relative="runs/parent/BASE_FIT_RECEIPT.json", receipt_sha256="receipt-bytes",
            artifact_relative="ordered/primary_beijing/nested_pca.npz" if kind=="nested" else base["artifact_cache_relative_path"])
        if kind == "independent": manifest["model_sha"] = self.r.fingerprint(dict(artifact_sha="weights", model_state_sha="state", normalization="scaler"))
        canonical = dict(run_id=c["run_id"], dataset="beijing", family="NESTED_PCA" if kind=="nested" else "B0", fold="primary",
            latent_k=k, canonical_fit_end=12622, foundation_sha="foundation", artifact_cache_relative_path=binding["artifact_relative"],
            artifact_sha256=parent["artifact_sha256"], model_size="small")
        if kind != "nested":
            canonical.update(expected_prediction_model_sha=self.r.fingerprint(dict(artifact_sha="weights",model_state_sha="state",normalization="scaler")),
                fit_receipt_public_relative_path=binding["receipt_relative"],fit_receipt_sha256=binding["receipt_sha256"])
        if kind == "independent":
            canonical["standard_V_binding"] = dict(model_sha=manifest["model_sha"], archive_sha256="saved-V-bytes",
                manifest_public_relative_path=binding["manifest_relative"], manifest_sha256="manifest-bytes")
        else:
            canonical.update(ordered_manifest_public_relative_path=binding["manifest_relative"], ordered_manifest_sha256="manifest-bytes")
            if kind != "nested": canonical.update(source_parent_run_id="parent", prefix_variant="shared")
        return binding, canonical

    def test_original_three_route_shapes_preserve_distinct_ordered_model_identity(self):
        for kind in ("independent", "ordered", "nested"):
            b, canonical = self.binding(kind)
            result = self.h.cost_spec(canonical, "small", b, self.plan, self.r.fingerprint)
            self.assertEqual(result, dict(canonical, accuracy_run_id=b["condition"]["run_id"], row_microbatch=128))
            self.assertEqual(canonical.get("accuracy_run_id"), None)
        b, canonical = self.binding("ordered")
        self.assertNotEqual(canonical["expected_prediction_model_sha"], b["manifest"]["model_sha"])
        bad = dict(canonical, expected_prediction_model_sha=b["manifest"]["model_sha"])
        with self.assertRaises(ValueError): self.h.cost_spec(bad, "small", b, self.plan, self.r.fingerprint)

    def test_unknown_field_k_and_fabricated_nested_receipt_are_rejected(self):
        for kind, patch in (("independent",dict(latent_k=6)), ("ordered",dict(g_route={})),
                            ("nested",dict(fit_receipt_public_relative_path="invented"))):
            b, canonical = self.binding(kind)
            with self.assertRaises(ValueError): self.h.cost_spec(dict(canonical, **patch), "small", b, self.plan, self.r.fingerprint)

    def test_one_route_readiness_does_not_require_other_eleven_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            p, q = Path(td)/"public", Path(td)/"cache"
            p.mkdir(); q.mkdir()
            b, _ = self.binding("independent")
            condition = b["condition"]
            (p/"REGISTRY.csv").write_text("tiny alias", encoding="utf-8")
            (p/"EXPERIMENT_REGISTRY.csv").write_bytes((p/"REGISTRY.csv").read_bytes())
            plan = dict(self.plan, outputs=[condition]+[dict(run_id="missing_"+str(i)) for i in range(11)],
                postT_registry_relative_path="REGISTRY.csv",postT_registry_sha256=self.h.sha(p/"REGISTRY.csv"))
            files = self.h.requested_paths(self.v,self.r,plan,p,q,condition)
            for key, path in files.items():
                if key == "alias": continue
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("{}",encoding="utf-8")
            self.assertEqual(self.h.readiness(self.v,self.r,plan,p,q,condition)["missing"], [])
            (p/"EXPERIMENT_REGISTRY.csv").write_text("changed",encoding="utf-8")
            with self.assertRaises(ValueError): self.h.readiness(self.v,self.r,plan,p,q,condition)

    def test_frozen_replay_source_artifact_and_route_binding(self):
        b, canonical = self.binding("ordered",6)
        reference = dict(run_id=b["condition"]["run_id"], split="V_select", origins=[1,2,3,4,5,6],
            route_sha=self.r.fingerprint(canonical), standard_manifest_sha256="manifest-bytes",standard_archive_sha256="saved-V-bytes",
            source_sha=self.r.fingerprint(self.plan["source39"]), artifact_bindings={"route":dict(cache_relative_path=b["artifact_relative"],sha256="weights")},
            expected_F_streams=36, created_before_inference=True, created_unix=1.)
        receipt = dict(status="PASS",run_id=b["condition"]["run_id"],origins=reference["origins"],route_sha=reference["route_sha"],
            model_size="small",reference_sha=self.r.fingerprint(reference),selected_artifact_bindings=reference["artifact_bindings"],
            standard_manifest_public_relative_path=b["manifest_relative"],standard_manifest_sha256="manifest-bytes",atol=1e-5,rtol=1e-4,
            actual_F_streams=36,expected_F_streams=36,F_physical_streams=36,T_final_accessed=False,independent_training_rerun=False)
        self.h.validate_replay(self.r,self.plan,b,canonical,reference,receipt)
        for field, value in (("status","FAIL_NUMERICAL_PARITY"),("F_physical_streams",66),("rtol",.1),("route_sha","wrong")):
            with self.assertRaises(ValueError): self.h.validate_replay(self.r,self.plan,b,canonical,reference,dict(receipt,**{field:value}))
        changed = dict(reference,source_sha="wrong")
        with self.assertRaises(ValueError): self.h.validate_replay(self.r,self.plan,b,canonical,changed,dict(receipt,reference_sha=self.r.fingerprint(changed)))

    def test_exact_worker_argv_protocol_and_postT_roots(self):
        argv = self.h.worker_arguments("cpu_cost_specs/token/B1.json","costs/token/B1.json",1,"model/small")
        self.assertEqual(argv[:6],["{PYTHON}","-X","utf8","-u","-B","{ORIGINAL_PUBLIC_ROOT}/measure_costs.py"])
        self.assertIn("--worker",argv)
        for flag,value in (("--public-root","{POST_T_PUBLIC_ROOT}"),("--cache-root","{POST_T_CACHE_ROOT}"),
            ("--device","cpu"),("--warmups","10"),("--timed-trials","20"),("--stage-trials","20")):
            self.assertEqual(argv[argv.index(flag)+1],value)
        with self.assertRaises(ValueError): self.h.worker_arguments("x","y",4,"model/small")

    def test_changed_dependency_is_rejected_before_import(self):
        with mock.patch.object(self.h,"sha",return_value="changed"), mock.patch.object(self.h.importlib.util,"spec_from_file_location") as importer:
            with self.assertRaises(ValueError): self.h.load_verifier()
            importer.assert_not_called()

    def test_existing_output_is_rejected_before_task_claim_or_science(self):
        with tempfile.TemporaryDirectory() as td:
            public,cache,asset = (Path(td)/name for name in ("public","cache","asset"))
            output = public/self.r.NAME/"diagnostics"/"already_exists"
            output.mkdir(parents=True)
            (output/"kept.txt").write_bytes(b"keep")
            fake_runner = types.SimpleNamespace(NAME=self.r.NAME,environment=mock.Mock(),relative=self.r.relative)
            runtime = types.SimpleNamespace(task=mock.Mock(),route=mock.Mock())
            state = (self.v,fake_runner,self.plan,dict(run_id="one"),dict(ready=True))
            with mock.patch.object(self.h,"metadata",return_value=state):
                with self.assertRaises(FileExistsError):
                    self.h.execute(public,cache,asset,self.h.PLAN_SHA,"one",output,deadline=2.,_runtime=runtime)
            runtime.task.assert_not_called(); runtime.route.assert_not_called()
            self.assertEqual((output/"kept.txt").read_bytes(),b"keep")
            self.assertFalse((cache/self.r.NAME/"cpu_cost_preparation_claims").exists())

    def test_default_cli_imports_no_science_and_starts_no_child_or_write(self):
        result = subprocess.run([sys.executable,"-B",str(HELPER)],text=True,capture_output=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["numpy_imported"]); self.assertFalse(value["torch_imported"])
        self.assertEqual(value["actual_model_array_worker_or_cost_execution"],0)
        self.assertEqual(value["files_written"],0)
        self.assertEqual(value["registered_outputs"],12)


if __name__ == "__main__": unittest.main()
