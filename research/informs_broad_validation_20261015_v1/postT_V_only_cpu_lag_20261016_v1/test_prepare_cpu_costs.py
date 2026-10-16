"""Synthetic metadata/route preparation tests; no arrays, models or cost timing."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
HELPER = HERE / "prepare_cpu_costs.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("prepare_cpu_costs_synthetic", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Preparation(unittest.TestCase):
    def setUp(self):
        self.helper = load_helper()
        self.runner = self.helper.load_runner()

    def fixture(self, folder):
        r = self.runner
        public, cache, asset = [Path(folder) / name for name in ("public", "cache", "asset")]
        for root in (public, cache, asset):
            root.mkdir()
        post, private = public / r.NAME, cache / r.NAME
        post.mkdir(); private.mkdir()
        names = ["campaign.py", *self.helper.FIT_SOURCES, "measure_costs.py", "route_catalog.py", "deployment.py"]
        names += ["source_%02d.py" % i for i in range(39-len(names))]
        sources = {}
        for name in names:
            (public / name).write_text("# frozen synthetic source\n", encoding="utf-8")
            sources[name] = r.sha(public / name)
        def csv_new(path, rows):
            with path.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        upstream, registered, conditions = [], [], []
        for family, run_id in r.RUN_IDS.items():
            old_id = r.UPSTREAM_IDS[family]
            upstream.append(dict(experiment_id=old_id, dataset="jena", family=family, fold="primary", latent_k="6", seed="92601", length="512", horizon="48"))
            registered.append(dict(experiment_id=run_id, dataset="jena", family=family, fold="primary", latent_k="6"))
            spec = dict(run_id=run_id, dataset="jena", family=family, latent_k=6, seed=92601, fold="primary", epochs=[.25, 1., 3.],
                        lr=.001, device="cpu", microbatch=4, effective_batch=16, F_row_microbatch=128, checkpoint_inputs=False,
                        lora_rank=8, lora_alpha=16, stage_label="postT_cpu_fresh_V_only", provenance={"upstream_registered_id": old_id})
            conditions.append(dict(run_id=run_id, family=family, upstream_registered_id=old_id, spec=spec,
                                   recipe_reference=dict(relative_path="campaign.py", sha256=sources["campaign.py"])))
        csv_new(public / "EXPERIMENT_REGISTRY.csv", upstream)
        csv_new(post / "REGISTRY.csv", registered)
        (post / "EXPERIMENT_REGISTRY.csv").write_bytes((post / "REGISTRY.csv").read_bytes())
        r.new(public / "SELECTION_LOCK.json", {"locked": True})
        r.new(public / "DATA_CONTRACT.json", {"frozen": "synthetic"})
        (asset / "norm.py").write_text("# frozen synthetic norm\n", encoding="utf-8")
        worktree_norm = public.parent / "temporal_frontend_campaign_20261015_v1/runtime.py"
        worktree_norm.parent.mkdir(); worktree_norm.write_bytes((asset / "norm.py").read_bytes())
        (asset / "model" / "revision").mkdir(parents=True)
        task = dict(dataset="jena", L=512, H=48, C=21, K=6, seed=92601, fold="primary", expected_V_origins=43,
                    data_sha256="data", columns=["c%d" % i for i in range(21)], array_sha256={"physical": "physical"},
                    primary_split_ranges={"A_fit": [0, 10482], "A_stop": [10482, 12579], "M_cal": [12579, 14675], "V_select": [14675, 16772], "T_final": [16772, 20965]},
                    A_fit_statistics_fingerprints={"mean": "mean", "std": "std", "count": "count"}, search_scaler_sha256="search", refit_scaler_sha256="refit")
        plan = dict(stage="POST_T_V_ONLY_DEVELOPMENT", runner_sha256=r.sha(HERE / "cpu_lag_runner.py"),
                    source39=sources, immutable_public={name: r.sha(public / name) for name in ("EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json")},
                    output_public_relative_root=r.NAME, output_cache_relative_root=r.NAME, postT_registry_relative_path="REGISTRY.csv", postT_registry_sha256=r.sha(post / "REGISTRY.csv"),
                    task_contract=task, conditions=conditions, CPU_threads=2, max_RSS_bytes=4294967296,
                    maximum_concurrent_workers=4, total_owned_RSS_limit_bytes=8589934592, storage_limit_bytes=32212254720, deadline_unix=1792222550,
                    model_asset_relative_path="model/revision", model_revision="revision", foundation_tensor_sha256="foundation",
                    instance_norm_source_relative_path="norm.py", instance_norm_source_sha256=r.sha(asset / "norm.py"))
        r.new(post / "PLAN.json", plan)
        for condition in conditions:
            run_id, family = condition["run_id"], condition["family"]
            run = post / "runs" / run_id; run.mkdir(parents=True)
            artifact = private / "runs" / run_id / "selected_bundle.pt"; artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"synthetic selected artifact, not a model")
            archive = private / "predictions" / run_id / "V_select.npz"; archive.parent.mkdir(parents=True)
            archive.write_bytes(b"synthetic saved archive, not numeric arrays")
            receipt = dict(status="COMPLETE_VALID", run_id=run_id, family=family, reused_complete_artifact=False, epochs=[.25, 1., 3.],
                           search_fit_end=10482, refit_fit_end=12579, refit_scaler_sha="refit", source_sha="training_source",
                           artifact_cache_relative_path=artifact.relative_to(private).as_posix(), artifact_sha256=r.sha(artifact))
            r.new(run / "BASE_FIT_RECEIPT.json", receipt)
            r.new(run / "SOURCE_CONTRACT.json", dict(spec=condition["spec"], data_sha="data", model_sha="foundation", source_sha="training_source", files={name: sources[name] for name in self.helper.FIT_SOURCES}))
            folder = post / "predictions" / run_id / "V_select"; folder.mkdir(parents=True)
            arrays = {name: dict(shape=shape, dtype="synthetic", sha256=name) for name, shape in self.helper.SHAPES.items()}
            manifest = dict(run_id=run_id, split="V_select", cohort="jena:primary:V_select:stride48", source_sha="prediction_source", config_sha="configuration",
                            data_sha="data", model_sha="prediction_"+family, scaler_sha="refit", arrays=arrays,
                            cache_relative_path=archive.relative_to(private).as_posix(), archive_sha256=r.sha(archive),
                            metadata=dict(family=family, fold="primary", artifact_sha256=receipt["artifact_sha256"], reused_prediction_cache=False, canonical_scaler_fit_end=12579),
                            score=dict(run_id=run_id, split="V_select", n_origins=43, num_channels=21, horizon=48),
                            cost_record=dict(run_id=run_id, batch=43, latent_k=6, actual_F_streams=258, expected_F_streams=258, F_calls=1))
            r.new(folder / "PREDICTION_MANIFEST.json", manifest)
            reference = {key: manifest[key] for key in self.helper.IDENTITY}
            reference.update(arrays={key: value for key, value in arrays.items() if key != "pred_normalized"}, expected_F_streams=258)
            r.new(folder / "REFERENCE_CONTRACT.json", reference)
        return public, cache, asset, plan

    def environment(self, cache, asset):
        result = {key: value for key, value in os.environ.items() if not key.startswith("INFORMS_GPU_")}
        result.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", INFORMS_ASSET_ROOT=str(asset), INFORMS_CACHE_ROOT=str(cache / self.runner.NAME))
        return result

    def canonical(self, binding, plan):
        receipt, manifest = binding["receipt"], binding["manifest"]
        run_id = binding["run_id"]
        receipt_path = "runs/"+run_id+"/BASE_FIT_RECEIPT.json"
        manifest_path = "predictions/"+run_id+"/V_select/PREDICTION_MANIFEST.json"
        return dict(run_id=run_id, dataset="jena", family=binding["family"], fold="primary", latent_k=6, canonical_fit_end=12579,
                    foundation_sha="foundation", model_size="small", artifact_cache_relative_path=receipt["artifact_cache_relative_path"], artifact_sha256=receipt["artifact_sha256"],
                    expected_prediction_model_sha=manifest["model_sha"], fit_receipt_public_relative_path=receipt_path, fit_receipt_sha256=binding["metadata_pins"][receipt_path],
                    standard_V_binding=dict(model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"], manifest_public_relative_path=manifest_path, manifest_sha256=binding["metadata_pins"][manifest_path]))

    def runtime(self, state):
        owner = self
        class Runtime:
            calls = []
            def task(self, runner, plan, cache):
                self.calls.append("task")
                return "synthetic task metadata"
            def route(self, run_id, task, private, public):
                self.calls.append(run_id)
                binding = next(row for row in state["bindings"] if row["run_id"] == run_id)
                return owner.canonical(binding, state["plan"]), "small"
        return Runtime()

    def test_default_cli_is_metadata_only_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _ = self.fixture(folder)
            before = {str(p.relative_to(folder)): self.runner.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            result = subprocess.run([sys.executable, "-B", str(HELPER), "--public-root", str(public), "--cache-root", str(cache), "--asset-root", str(asset)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["status"], "METADATA_READY_FOR_EXPLICIT_PREPARATION")
            self.assertFalse(value["numpy_imported"]); self.assertFalse(value["torch_imported"])
            self.assertEqual(value["actual_model_array_or_cost_execution"], 0)
            self.assertEqual(before, {str(p.relative_to(folder)): self.runner.sha(p) for p in Path(folder).rglob("*") if p.is_file()})

    def test_waits_for_four_complete_fits_and_exact_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, plan = self.fixture(folder)
            post = public / self.runner.NAME
            (post / "EXPERIMENT_REGISTRY.csv").unlink()
            (post / "runs" / plan["conditions"][0]["run_id"] / "BASE_FIT_RECEIPT.json").unlink()
            state = self.helper.metadata(public, cache, asset)
            self.assertFalse(state["ready"]); self.assertFalse(state["alias_ready"])
            self.assertEqual(len(state["bindings"]), 3)
            with self.assertRaises(ValueError):
                self.helper.execute(public, cache, asset, self.runner.sha(post / "PLAN.json"), post / "new_output", _runtime=object())
            self.assertFalse((post / "new_output").exists())

    def test_recipe_old_source_alias_and_V_stream_tamper_are_rejected(self):
        for change in ("recipe", "old_source", "alias", "streams", "scaler"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, plan = self.fixture(folder)
                post = public / self.runner.NAME
                if change == "recipe":
                    plan["conditions"][0]["spec"]["lr"] = .002
                    (post / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
                elif change == "old_source":
                    (public / "measure_costs.py").write_text("changed", encoding="utf-8")
                elif change == "alias":
                    (post / "EXPERIMENT_REGISTRY.csv").write_text("changed", encoding="utf-8")
                else:
                    path = post / "predictions" / plan["conditions"][0]["run_id"] / "V_select/PREDICTION_MANIFEST.json"
                    value = self.runner.read(path)
                    if change == "streams": value["cost_record"]["actual_F_streams"] = 257
                    else: value["scaler_sha"] = "changed"
                    path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.helper.metadata(public, cache, asset)

    def test_canonical_is_preserved_except_the_two_measurement_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _ = self.fixture(folder)
            state = self.helper.metadata(public, cache, asset)
            binding = state["bindings"][0]
            canonical = self.canonical(binding, state["plan"])
            value = self.helper.cost_spec(canonical, binding, state["plan"], self.runner)
            self.assertEqual({k: v for k, v in value.items() if k not in {"accuracy_run_id", "row_microbatch"}}, canonical)
            self.assertEqual(value["accuracy_run_id"], canonical["run_id"]); self.assertEqual(value["row_microbatch"], 128)
            canonical["foundation_sha"] = "different foundation"
            with self.assertRaises(ValueError):
                self.helper.cost_spec(canonical, binding, state["plan"], self.runner)

    def test_prepares_eight_complete_original_worker_arguments_without_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _ = self.fixture(folder)
            post = public / self.runner.NAME
            state = self.helper.metadata(public, cache, asset)
            runtime = self.runtime(state)
            with patch.dict(os.environ, self.environment(cache, asset), clear=True), patch("subprocess.Popen", side_effect=AssertionError("No worker may start")):
                result = self.helper.execute(public, cache, asset, self.runner.sha(post / "PLAN.json"), post / "prepared", _runtime=runtime)
            self.assertEqual(result["worker_count"], 8); self.assertEqual(result["actual_subprocesses_started"], 0)
            self.assertEqual(len(runtime.calls), 5)
            self.assertEqual({(j["family"], j["batch"]) for j in result["jobs"]}, {(f, b) for f in self.runner.RUN_IDS for b in (1, 8)})
            for job in result["jobs"]:
                args = job["arguments"]
                self.assertIn("--worker", args)
                for option, value in (("--device", "cpu"), ("--batch", str(job["batch"])), ("--warmups", "10"), ("--timed-trials", "20"), ("--stage-trials", "20"), ("--public-root", "{POST_T_PUBLIC_ROOT}"), ("--cache-root", "{POST_T_CACHE_ROOT}")):
                    self.assertEqual(args[args.index(option)+1], value)
                self.assertEqual(job["expected_F_streams_per_trial"], job["batch"]*6)
                path = cache / self.runner.NAME / job["spec_cache_relative_path"]
                self.assertEqual(self.runner.sha(path), job["spec_sha256"])
                spec = self.runner.read(path)
                self.assertEqual(spec["standard_V_binding"], job["standard_V_binding"])
                self.assertEqual(spec["artifact_sha256"], job["artifact_sha256"])
            text = (post / "prepared/COST_EXECUTION_PLAN.JSON").read_text(encoding="utf-8")
            for private_path in (str(public), str(cache), str(asset)):
                self.assertNotIn(private_path, text)

    def test_existing_output_or_wrong_cpu_environment_stops_before_task(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _ = self.fixture(folder)
            post = public / self.runner.NAME
            state = self.helper.metadata(public, cache, asset)
            runtime = self.runtime(state)
            existing = post / "existing"; existing.mkdir()
            with patch.dict(os.environ, self.environment(cache, asset), clear=True):
                with self.assertRaises(FileExistsError):
                    self.helper.execute(public, cache, asset, self.runner.sha(post / "PLAN.json"), existing, _runtime=runtime)
            env = self.environment(cache, asset); env["INFORMS_GPU_TOKEN"] = "forbidden inherited lease"
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    self.helper.execute(public, cache, asset, self.runner.sha(post / "PLAN.json"), post / "wrong_env", _runtime=runtime)
            self.assertEqual(runtime.calls, [])
            self.assertFalse((cache / self.runner.NAME / "cost_specs").exists())

    def test_postflight_source_change_preserves_freeze_failure_without_execution_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _ = self.fixture(folder)
            post = public / self.runner.NAME
            state = self.helper.metadata(public, cache, asset)
            runtime = self.runtime(state)
            original = runtime.route
            def changed(*args):
                (public / "measure_costs.py").write_text("changed during preparation", encoding="utf-8")
                return original(*args)
            runtime.route = changed
            with patch.dict(os.environ, self.environment(cache, asset), clear=True):
                with self.assertRaises(ValueError):
                    self.helper.execute(public, cache, asset, self.runner.sha(post / "PLAN.json"), post / "changed", _runtime=runtime)
            self.assertFalse((post / "changed").exists())
            self.assertFalse((cache / self.runner.NAME / "cost_specs").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
