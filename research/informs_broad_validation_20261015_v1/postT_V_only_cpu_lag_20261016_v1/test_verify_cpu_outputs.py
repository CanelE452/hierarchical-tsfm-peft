"""Synthetic verification/replay guards; no real research arrays or models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
HELPER = HERE / "verify_cpu_outputs.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifyGuards(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.exists(), "Output verifier is not implemented")
        self.helper = load("verify_postT_cpu", HELPER)
        self.fixtures = load("cpu_runner_fixtures", HERE / "test_cpu_lag_runner.py").RunnerGuards()
        self.fixtures.setUp()

    def test_default_metadata_imports_no_science_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, plan_path, _ = self.fixtures.fixture(folder)
            h = self.helper
            before = {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            result = subprocess.run([sys.executable, "-B", str(HELPER), "--public-root", str(public), "--cache-root", str(cache),
                "--asset-root", str(asset), "--plan-sha", h.sha(plan_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["numpy_imported"])
            self.assertFalse(report["torch_imported"])
            self.assertEqual(report["actual_model_or_array_access"], 0)
            self.assertEqual(before, {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()})

    def test_registry_projection_keeps_all_four_and_preserves_actual_receipt_status(self):
        h = self.helper
        with tempfile.TemporaryDirectory() as folder:
            public, _, _, _, plan = self.fixtures.fixture(folder)
            post = public / HERE.name
            first, second = plan["conditions"][:2]
            for condition, status in ((first, "COMPLETE_VALID"), (second, "BUDGET_LIMIT")):
                target = post / "runs" / condition["run_id"]
                target.mkdir(parents=True)
                h.new(target / "BASE_FIT_RECEIPT.json", dict(run_id=condition["run_id"], family=condition["family"], status=status))
            rows = h.registry_projection(plan, post)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["execution_status"], "COMPLETE_VALID")
            self.assertEqual(rows[1]["execution_status"], "BUDGET_LIMIT")
            self.assertEqual(rows[2]["execution_status"], "NO_COMPLETE_BASE_FIT_RECEIPT")
            self.assertTrue(all(row["reason"] for row in rows))

    def test_registry_alias_is_exact_x_copy_and_wrong_existing_is_rejected(self):
        h = self.helper
        with tempfile.TemporaryDirectory() as folder:
            public, _, _, _, plan = self.fixtures.fixture(folder)
            post = public / HERE.name
            original = (post / "REGISTRY.csv").read_bytes()
            first = h.ensure_registry_alias(post, plan)
            self.assertTrue(first["created_now"])
            self.assertEqual((post / "EXPERIMENT_REGISTRY.csv").read_bytes(), original)
            self.assertFalse(h.ensure_registry_alias(post, plan)["created_now"])
            (post / "EXPERIMENT_REGISTRY.csv").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                h.ensure_registry_alias(post, plan)
            self.assertEqual((post / "REGISTRY.csv").read_bytes(), original)

    def test_rescore_forwards_full_registry_and_saved_bundle_to_original_api(self):
        h = self.helper
        calls = []
        def original(*args):
            calls.append(args)
            return dict(status="PASS", prediction_rescoring=True, checkpoint_replay=False, n_origins=43)
        manifest = {"cost_record": {"actual_F_streams": 258}}
        rows = [{"run_id": "a"}, {"run_id": "b"}, {"run_id": "c"}, {"run_id": "d"}]
        report = h.rescore_call(original, manifest, "reference", "score", ["stats"], rows, ["a", "b", "c", "d"], "private_new")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (manifest, "reference", "score", ["stats"], rows, ["a", "b", "c", "d"], [manifest["cost_record"]], "private_new"))
        self.assertTrue(report["prediction_rescoring"])
        self.assertFalse(report["checkpoint_replay"])

    def test_replay_calls_original_cpu_six_origin_api_and_preserves_reuse_status(self):
        h = self.helper
        calls = []
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return dict(status="PASS", run_id="new", origins=list(range(6)), actual_F_streams=36, expected_F_streams=36,
                        atol=1e-5, rtol=1e-4, executed_now=True, T_final_accessed=False)
        result = h.replay_call(original, "new", "task", "cache_new", "public_new", "model", 100)
        self.assertTrue(result["executed_now"])
        self.assertEqual(calls[0][0], ("new", "task", "cache_new", "public_new", "model"))
        self.assertEqual(calls[0][1], dict(device="cpu", deadline=100, lease=None, batch_size=4, atol=1e-5, rtol=1e-4))
        def cached(*args, **kwargs):
            return dict(original(*args, **kwargs), executed_now=False)
        self.assertFalse(h.replay_call(cached, "new", "task", "c", "p", "m", 100)["executed_now"])
        def wrong(*args, **kwargs):
            return dict(original(*args, **kwargs), actual_F_streams=35)
        with self.assertRaises(ValueError):
            h.replay_call(wrong, "new", "task", "c", "p", "m", 100)

    def test_T_modes_unregistered_ID_and_existing_output_are_refused(self):
        h = self.helper
        with tempfile.TemporaryDirectory() as folder:
            public, _, _, _, plan = self.fixtures.fixture(folder)
            with self.assertRaises(ValueError):
                h.choose(plan, "unknown", "rescore")
            with self.assertRaises(ValueError):
                h.choose(plan, plan["conditions"][0]["run_id"], "T_final")
            output = public / HERE.name / "old_output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                h.require_new_output(output, public / HERE.name)
            result = subprocess.run([sys.executable, "-B", str(HELPER), "--mode", "T_final"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_private_traceback_preserves_original_error_and_public_reference_only(self):
        h = self.helper
        runner = h.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            try:
                raise ValueError("synthetic saved rescore failure")
            except ValueError as error:
                record = runner.save_private_failure(cache / HERE.name, cache, "rescore", "new", "verification", error)
            self.assertEqual(set(record), {"cache_relative_path", "error_class", "phase"})
            self.assertEqual(h.read(cache / record["cache_relative_path"])["error"], "synthetic saved rescore failure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
