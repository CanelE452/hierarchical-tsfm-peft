"""Synthetic post-T CPU runner guards; no real data, model, or fit execution."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "cpu_lag_runner.py"


class RunnerGuards(unittest.TestCase):
    def setUp(self):
        self.assertTrue(RUNNER.is_file(), "Post-T CPU runner is not implemented")
        spec = importlib.util.spec_from_file_location("postT_cpu_runner", RUNNER)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def fixture(self, folder):
        r = self.runner
        public, cache, asset = [Path(folder) / name for name in ("public", "cache", "asset")]
        for root in (public, cache, asset):
            root.mkdir()
        post = public / HERE.name
        post.mkdir()
        sources = {"campaign.py": None, **{"source_%02d.py" % i: None for i in range(38)}}
        for name in sources:
            r.new(public / name, {"source": name})
            sources[name] = r.sha(public / name)
        upstream = []
        conditions = []
        for family, run_id in r.RUN_IDS.items():
            old = r.UPSTREAM_IDS[family]
            upstream.append(dict(experiment_id=old, dataset="jena", family=family, latent_k="6", seed="92601", fold="primary", length="512", horizon="48"))
            spec = dict(run_id=run_id, dataset="jena", family=family, latent_k=6, seed=92601, fold="primary",
                        epochs=[.25, 1, 3], lr=.001, device="cpu", microbatch=4, effective_batch=16,
                        F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16,
                        stage_label="postT_cpu_fresh_V_only", provenance={"upstream_registered_id": old})
            conditions.append(dict(run_id=run_id, family=family, upstream_registered_id=old, spec=spec,
                                   recipe_reference=dict(relative_path="campaign.py", sha256=sources["campaign.py"])))
        def csv_new(path, rows):
            with path.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        csv_new(public / "EXPERIMENT_REGISTRY.csv", upstream)
        csv_new(post / "REGISTRY.csv", [dict(experiment_id=c["run_id"], family=c["family"]) for c in conditions])
        r.new(public / "SELECTION_LOCK.json", {"locked": True})
        r.new(public / "DATA_CONTRACT.json", {"frozen": True})
        r.new(asset / "norm.py", {"source": "frozen_normalization"})
        worktree_norm = public.parent / "temporal_frontend_campaign_20261015_v1/runtime.py"
        worktree_norm.parent.mkdir()
        worktree_norm.write_bytes((asset / "norm.py").read_bytes())
        (asset / "model").mkdir()
        plan = dict(source39=sources, immutable_public={name: r.sha(public / name) for name in ("EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json")},
                    runner_sha256=r.sha(RUNNER), stage="POST_T_V_ONLY_DEVELOPMENT",
                    output_public_relative_root=HERE.name, output_cache_relative_root=HERE.name,
                    postT_registry_relative_path="REGISTRY.csv", postT_registry_sha256=r.sha(post / "REGISTRY.csv"),
                    model_asset_relative_path="model", model_revision="model", foundation_tensor_sha256="frozen_tensor",
                    instance_norm_source_relative_path="norm.py", instance_norm_source_sha256=r.sha(asset / "norm.py"),
                    task_contract=dict(dataset="jena", L=512, H=48, C=21, K=6, seed=92601, fold="primary",
                                       expected_V_origins=43, data_sha256="data", columns=["c%d" % i for i in range(21)],
                                       primary_split_ranges={"A_fit": [0, 1000], "A_stop": [1000, 1200], "M_cal": [1200, 1400], "V_select": [1400, 3499], "T_final": [3499, 4000]},
                                       array_sha256={"physical": "hash"}, A_fit_statistics_fingerprints={"mean": {}, "std": {}, "count": {}},
                                       search_scaler_sha256="search_scaler", refit_scaler_sha256="refit_scaler"),
                    conditions=conditions, deadline_unix=1792222550, max_RSS_bytes=4294967296, CPU_threads=2,
                    maximum_concurrent_workers=4, total_owned_RSS_limit_bytes=8589934592, storage_limit_bytes=32212254720)
        r.new(post / "PLAN.json", plan)
        return public, cache, asset, post / "PLAN.json", plan

    def test_metadata_fresh_cli_imports_no_science_and_writes_nothing(self):
        r = self.runner
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, plan_path, plan = self.fixture(folder)
            before = {str(p.relative_to(folder)): r.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            self.assertEqual(r.metadata(public, cache, asset, r.sha(plan_path)), plan)
            completed = subprocess.run([sys.executable, "-B", str(RUNNER), "--public-root", str(public),
                "--cache-root", str(cache), "--asset-root", str(asset), "--plan-sha", r.sha(plan_path)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["registered_conditions"], 4)
            self.assertFalse(result["numpy_imported"])
            self.assertFalse(result["torch_imported"])
            self.assertFalse(result["science_modules_imported"])
            self.assertEqual(before, {str(p.relative_to(folder)): r.sha(p) for p in Path(folder).rglob("*") if p.is_file()})

    def test_source_global_lock_registry_and_postT_registry_tampering_rejected(self):
        r = self.runner
        for name in ("source_00.py", "SELECTION_LOCK.json", "EXPERIMENT_REGISTRY.csv", HERE.name + "/REGISTRY.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, plan_path, _ = self.fixture(folder)
                (public / name).write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    r.metadata(public, cache, asset, r.sha(plan_path))

    def test_wrong_family_recipe_and_device_cannot_be_hidden_by_new_plan_sha(self):
        r = self.runner
        for key, value in (("microbatch", 2), ("device", "cuda"), ("epochs", [.25]), ("family", "DIRECT_G")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, plan_path, plan = self.fixture(folder)
                plan["conditions"][0]["spec"][key] = value
                plan_path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaises(ValueError):
                    r.metadata(public, cache, asset, r.sha(plan_path))

    def test_existing_run_or_prediction_path_refused_before_core(self):
        r = self.runner
        for root_name, directory in (("cache", "runs"), ("public", "runs"), ("cache", "predictions"), ("public", "predictions")):
            with self.subTest(root=root_name, directory=directory), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, _, plan = self.fixture(folder)
                condition = plan["conditions"][0]
                root = cache if root_name == "cache" else public
                (root / HERE.name / directory / condition["run_id"]).mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    r.require_fresh_paths(plan, public, cache, condition["run_id"])

    def test_fake_core_fit_then_only_full_V_and_partial_fit_skips_prediction(self):
        r = self.runner
        calls = []
        def fit(spec, task, cache, public, foundation, deadline):
            calls.append(("fit", spec["run_id"], cache, public, foundation, deadline))
            return dict(status="COMPLETE_VALID", run_id=spec["run_id"], reused_complete_artifact=False)
        def predict(run_id, task, split, cache, public, foundation, **kwargs):
            calls.append(("predict", split, cache, public, foundation, kwargs))
            return dict(score={"n_origins": 43}, metadata={"reused_prediction_cache": False},
                        cost_record={"actual_F_streams": 258, "expected_F_streams": 258})
        spec = dict(run_id="new_condition")
        result = r.fit_and_predict(fit, predict, spec, "task", "private_new", "public_new", "foundation", 100)
        self.assertEqual([c[0] for c in calls], ["fit", "predict"])
        self.assertEqual(calls[1][1], "V_select")
        self.assertEqual(calls[1][2:5], ("private_new", "public_new", "foundation"))
        self.assertEqual(result["prediction"]["score"]["n_origins"], 43)
        def wrong_streams(*args, **kwargs):
            return dict(score={"n_origins": 43}, metadata={"reused_prediction_cache": False},
                        cost_record={"actual_F_streams": 257, "expected_F_streams": 258})
        with self.assertRaises(ValueError):
            r.fit_and_predict(fit, wrong_streams, spec, "task", "private_new", "public_new", "foundation", 100)
        calls.clear()
        result = r.fit_and_predict(lambda *a: {"status": "BUDGET_LIMIT"}, predict, spec, "task", "p", "q", "f", 100)
        self.assertIsNone(result["prediction"])
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            r.fit_and_predict(fit, predict, spec, "task", "p", "q", "f", 100, split="T_final")
        self.assertEqual(calls, [])

    def test_probe_calls_original_first_effective_update_and_does_not_touch_fit(self):
        r = self.runner
        calls = []
        def segment(*args, **kwargs):
            calls.append((args, kwargs))
            return dict(status="SEGMENT_COMPLETE", updates=1, exposures=16)
        value = r.probe_update(segment, "model", "windows", "all_A50_origins", "sampler", "optimizer", "predict", 100)
        self.assertEqual(value["updates"], 1)
        self.assertEqual(calls[0][0], ("model", "windows", "all_A50_origins", "sampler", "optimizer", 16, "predict"))
        self.assertEqual(calls[0][1], {"effective_batch": 16, "microbatch": 4, "starting_updates": 0, "budget_deadline": 100})

    def test_T_cli_and_unregistered_run_refused_without_import_or_output(self):
        r = self.runner
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, _, plan = self.fixture(folder)
            with self.assertRaises(ValueError):
                r.select_condition(plan, "unknown_run", "fit")
            with self.assertRaises(ValueError):
                r.select_condition(plan, plan["conditions"][1]["run_id"], "probe")
            result = subprocess.run([sys.executable, "-B", str(RUNNER), "--split", "T_final"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_full_error_and_traceback_remain_private_and_public_reference_is_relative(self):
        r = self.runner
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            post_cache = cache / HERE.name
            try:
                raise ValueError("synthetic original diagnostic detail")
            except ValueError as error:
                reference = r.save_private_failure(post_cache, cache, "probe", "synthetic_run", "execution", error)
            self.assertFalse(Path(reference["cache_relative_path"]).is_absolute())
            self.assertEqual(set(reference), {"cache_relative_path", "error_class", "phase"})
            stored = r.read(cache / reference["cache_relative_path"])
            self.assertEqual(stored["error"], "synthetic original diagnostic detail")
            self.assertIn("ValueError: synthetic original diagnostic detail", stored["traceback"])
            try:
                raise RuntimeError("synthetic separate postflight failure")
            except RuntimeError as error:
                postflight = r.save_private_failure(post_cache, cache, "probe", "synthetic_run", "postflight", error)
            self.assertNotEqual(reference["cache_relative_path"], postflight["cache_relative_path"])
            self.assertEqual(r.read(cache / reference["cache_relative_path"])["error_class"], "ValueError")
            self.assertEqual(r.read(cache / postflight["cache_relative_path"])["error_class"], "RuntimeError")
            self.assertEqual(len(list((post_cache / "failures").iterdir())), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
