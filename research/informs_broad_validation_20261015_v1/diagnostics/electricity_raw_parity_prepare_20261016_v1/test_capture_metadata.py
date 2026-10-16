"""Synthetic JSON/metadata guards only; no science imports or GPU execution."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
HELPER = HERE / "capture_original_parity.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("electricity_capture", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureMetadata(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "Electricity capture helper is not implemented")
        self.helper = load_helper()

    def fixture(self, folder):
        h = self.helper
        public, cache, asset = [Path(folder) / name for name in ("public", "cache", "asset")]
        for path in (public, cache, asset):
            path.mkdir()
        source = {}
        for index in range(39):
            name = "source_%02d.py" % index
            h.new(public / name, {"index": index})
            source[name] = h.sha(public / name)
        immutable = {}
        for name in ("EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json"):
            h.new(public / name, {"frozen": name})
            immutable[name] = h.sha(public / name)
        manifest = dict(run_id="primary_electricity_E1_RAW_BOLT_SMALL_k0_s92601", split="V_select",
                        data_sha="data", scaler_sha="scaler", model_sha="model", archive_sha256="archive")
        h.new(public / "V.json", manifest)
        h.new(asset / "norm.py", {"normalization": "fixed"})
        (asset / "model").mkdir()
        conditions = []
        for batch in (1, 8):
            spec = dict(run_id=manifest["run_id"] + "__rowsfull", accuracy_run_id=manifest["run_id"],
                        family="RAW", dataset="electricity", fold="primary", latent_k=321,
                        row_microbatch="full", foundation_sha="foundation", model_size="small",
                        expected_prediction_model_sha="model",
                        standard_V_binding=dict(manifest_public_relative_path="V.json", manifest_sha256=h.sha(public / "V.json"),
                                                model_sha="model", archive_sha256="archive"))
            spec_name = "spec%d.json" % batch
            h.new(cache / spec_name, spec)
            failures = []
            for index in range(4):
                ename, lname = "failure%d_%d.json" % (batch, index), "log%d_%d.txt" % (batch, index)
                h.new(public / ename, {"returncode": 1, "success_receipt_present": False})
                (cache / lname).write_text("AssertionError: original gate", encoding="utf-8")
                failures.append(dict(execution_public_relative_path=ename, execution_sha256=h.sha(public / ename),
                                     private_log_cache_relative_path=lname, private_log_sha256=h.sha(cache / lname)))
            conditions.append(dict(batch=batch, spec_cache_relative_path=spec_name, spec_sha256=h.sha(cache / spec_name),
                                   original_failures=failures))
        plan = dict(source39=source, immutable_public=immutable, helper_sha256=h.sha(HELPER),
                    dataset="electricity", L=512, H=48, C=321, foundation_tensor_sha256="foundation",
                    model_asset_relative_path="model", external_instance_norm_source_relative_path="norm.py",
                    external_instance_norm_source_sha256=h.sha(asset / "norm.py"),
                    canonical_accuracy_manifest=dict(relative_path="V.json", sha256=h.sha(public / "V.json"),
                                                     data_sha256="data", scaler_sha="scaler"),
                    conditions=conditions, cost_cumulative_ceiling_seconds=43200,
                    candidates=["full", 32, 8, 4], atol=1e-5, rtol=1e-4)
        plan_dir = public / "diagnostics" / HERE.name
        plan_dir.mkdir(parents=True)
        h.new(plan_dir / "PLAN.json", plan)
        return public, cache, asset, plan_dir / "PLAN.json", plan

    def test_metadata_and_fresh_cli_import_no_science_and_do_not_write(self):
        h = self.helper
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, plan_path, plan = self.fixture(folder)
            before = {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            self.assertEqual(h.metadata(public, cache, asset, h.sha(plan_path)), plan)
            result = subprocess.run([sys.executable, "-B", str(HELPER), "--public-root", str(public),
                                     "--cache-root", str(cache), "--asset-root", str(asset), "--plan-sha", h.sha(plan_path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertFalse(output["numpy_imported"])
            self.assertFalse(output["torch_imported"])
            self.assertEqual(output["actual_GPU_parity_calls"], 0)
            self.assertEqual(before, {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()})

    def test_wrong_plan_or_source_or_lock_is_rejected(self):
        h = self.helper
        for change in ("plan", "source", "lock"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, plan_path, _ = self.fixture(folder)
                digest = h.sha(plan_path)
                target = dict(plan=plan_path, source=public / "source_00.py", lock=public / "SELECTION_LOCK.json")[change]
                target.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    h.metadata(public, cache, asset, digest)

    def test_spec_and_original_failure_tamper_rejected(self):
        h = self.helper
        for name in ("spec1.json", "log8_3.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                public, cache, asset, plan_path, _ = self.fixture(folder)
                (cache / name).write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    h.metadata(public, cache, asset, h.sha(plan_path))

    def test_capture_saves_false_gate_without_changing_defaults(self):
        h = self.helper
        result = dict(successful_variants_parity=False, variants=[dict(status="BLOCKED_CUDA_OOM"), dict(status="FAIL", max_abs=.25)])
        calls = []
        def original(route, physical, observed, *, lease):
            calls.append((route, physical, observed, lease))
            return result
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "RAW_PARITY_B1.json"
            self.assertEqual(h.capture(original, "r", "p", "o", "l", target), result)
            self.assertEqual(h.read(target), result)
            self.assertEqual(calls, [("r", "p", "o", "l")])
            with self.assertRaises(FileExistsError):
                h.capture(original, "r", "p", "o", "l", target)
            self.assertEqual(h.read(target), result)

    def test_execution_requires_explicit_plan_sha_before_any_import_or_write(self):
        h = self.helper
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            with self.assertRaises(ValueError):
                h.execute_child(Path(folder), Path(folder), Path(folder), output, None)
            self.assertFalse(output.exists())

    def test_no_nested_owner_and_no_parity_tolerance_override(self):
        h = self.helper
        with self.assertRaises(ValueError):
            h.run_with_owner(None, "asset", "cache", "public", "pin")
        tree = ast.parse(HELPER.read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == "GPUReservation" for n in ast.walk(tree)))
        capture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
        call = next(n for n in ast.walk(capture) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "function")
        self.assertEqual(len(call.args), 3)
        self.assertEqual([k.arg for k in call.keywords], ["lease"])

    def test_owned_child_normal_timeout_and_kill_fallback_are_bounded(self):
        self.assertIn("process_api", self.helper.wait_owned_child.__code__.co_varnames, "Owned-tree watchdog is not implemented")
        class MissingProcess(Exception):
            pass
        class API:
            NoSuchProcess = MissingProcess
            def __init__(self, force_kill):
                self.events, self.force_kill = [], force_kill
                self.processes = {pid: Process(self, pid) for pid in (101, 202)}
            def Process(self, pid):
                process = self.processes[pid]
                if not process.alive:
                    raise MissingProcess()
                return process
            def wait_procs(self, processes, timeout):
                self.events.append(("wait_procs", timeout))
                return ([p for p in processes if not p.alive], [p for p in processes if p.alive])
        class Process:
            def __init__(self, api, pid):
                self.api, self.pid, self.alive = api, pid, True
            def create_time(self):
                return float(self.pid)
            def children(self, recursive):
                self.api.events.append(("descendants", self.pid, recursive))
                return [self.api.processes[202]]
            def terminate(self):
                self.api.events.append(("terminate", self.pid))
                self.alive = self.api.force_kill
            def kill(self):
                self.api.events.append(("kill", self.pid))
                self.alive = False
        class Child:
            pid = 101
            def __init__(self, api, timed_out):
                self.api, self.timed_out = api, timed_out
            def wait(self, timeout):
                self.api.events.append(("wait", timeout))
                if self.timed_out:
                    raise subprocess.TimeoutExpired("owned_child", timeout)
                return 0
            def poll(self):
                return None if self.api.processes[101].alive else 1
        for timed_out, force_kill in ((False, False), (True, False), (True, True)):
            with self.subTest(timed_out=timed_out, force_kill=force_kill):
                api = API(force_kill)
                child = Child(api, timed_out)
                result = self.helper.wait_owned_child(child, 60, process_api=api)
                self.assertEqual(result["child_timeout"], timed_out)
                self.assertEqual(result["child_terminate_requested"], timed_out)
                self.assertEqual(result["child_kill_requested"], force_kill)
                self.assertEqual(result["actual_exitcode"], 1 if timed_out else 0)
                self.assertTrue(result["child_join_confirmed"])
                self.assertTrue(result["child_tree_join_confirmed"])
                self.assertEqual(result["remaining_owned_processes"], [])
                self.assertEqual(api.events[0], ("wait", 60))
                if timed_out:
                    self.assertEqual([row["pid"] for row in result["child_tree_snapshot"]], [101, 202])
                    self.assertEqual([event for event in api.events if event[0] == "terminate"], [("terminate", 202), ("terminate", 101)])
                    self.assertIn(("wait_procs", 10), api.events)
                    self.assertEqual([event for event in api.events if event[0] == "kill"], [("kill", 202), ("kill", 101)] if force_kill else [])
                    if force_kill:
                        self.assertIn(("wait_procs", 5), api.events)

    @unittest.skipUnless(sys.platform == "win32", "Windows venv launcher/runtime lifecycle test")
    def test_cpu_only_owned_windows_launcher_and_runtime_are_actually_joined(self):
        import os
        import psutil
        env = {key: value for key, value in os.environ.items() if not key.startswith("INFORMS_GPU_")}
        env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(20)"],
                                 env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        expected = []
        try:
            root = psutil.Process(child.pid)
            deadline = time.monotonic() + 2
            descendants = root.children(recursive=True)
            while not descendants and time.monotonic() < deadline:
                time.sleep(.01)
                descendants = root.children(recursive=True)
            expected = [dict(pid=p.pid, create_time=p.create_time()) for p in [root] + descendants]
            self.assertGreaterEqual(len(expected), 2, "Expected the actual Windows venv launcher and runtime")
            result = self.helper.wait_owned_child(child, .2)
            self.assertTrue(result["child_timeout"])
            self.assertTrue(result["child_tree_join_confirmed"])
            self.assertIsNotNone(result["actual_exitcode"])
            self.assertEqual(result["remaining_owned_processes"], [])
            actual = {(row["pid"], row["create_time"]) for row in result["child_tree_snapshot"]}
            self.assertTrue({(row["pid"], row["create_time"]) for row in expected}.issubset(actual))
            for row in expected:
                try:
                    process = psutil.Process(row["pid"])
                except psutil.NoSuchProcess:
                    continue
                self.assertNotAlmostEqual(process.create_time(), row["create_time"], places=3)
            print(json.dumps(dict(status="PASS_ACTUAL_CPU_OWNED_WINDOWS_SLEEP_TREE", sleep_seconds=20,
                                  max_wait_seconds=.2, expected_processes=expected, lifecycle=result,
                                  new_GPU_or_model_execution=0), sort_keys=True), flush=True)
        finally:
            if child.poll() is None:
                self.helper.wait_owned_child(child, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
