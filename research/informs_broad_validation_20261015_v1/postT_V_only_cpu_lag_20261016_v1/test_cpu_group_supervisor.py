import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location("cpu_group_supervisor_tested", HERE / "cpu_group_supervisor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gone(Exception):
    pass


class FakeProcess:
    def __init__(self, pid, created=1., children=()):
        self.pid, self.created, self.child_rows = pid, created, list(children)
        self.actions = []
        self.dead = False

    def create_time(self):
        if self.dead:
            raise Gone()
        return self.created

    def children(self, recursive=False):
        return self.child_rows

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")
        self.dead = True

    def memory_info(self):
        return type("Memory", (), {"rss": 12})()

    def cpu_times(self):
        return type("CPU", (), {"user": .1, "system": .2})()

    def status(self):
        return "running"


class FakeAPI:
    NoSuchProcess = Gone
    AccessDenied = PermissionError
    STATUS_ZOMBIE = "zombie"

    def __init__(self, *processes):
        self.processes = {p.pid: p for p in processes}
        self.waits = []

    def Process(self, pid):
        p = self.processes.get(pid)
        if p is None or p.dead:
            raise Gone()
        return p

    def wait_procs(self, processes, timeout):
        self.waits.append(timeout)
        return [], [p for p in processes if not p.dead]


class SupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = load_module()

    def spec(self, root):
        return dict(schema="CPU_GROUP_SUPERVISOR_v1", asset_root=str(root), cache_root=str(root),
                    output_root=str(root / "new-output"), deadline_unix=min(time.time() + 30, self.s.HARD_DEADLINE),
                    processes=[dict(id="one", command=[sys.executable, "-c", "pass"], cwd=str(root))])

    def test_spec_rejects_limits_duplicates_existing_output_and_shell_strings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = self.spec(root)
            self.assertEqual(len(self.s.validate_spec(original)["processes"]), 1)
            mutations = [dict(max_each_RSS=4294967297), dict(max_group_RSS=8589934593),
                         dict(deadline_unix=self.s.HARD_DEADLINE + 1),
                         dict(processes=original["processes"] * 5),
                         dict(processes=original["processes"] * 2),
                         dict(processes=[dict(id="bad", command="python -c pass", cwd=str(root))])]
            for patch in mutations:
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    self.s.validate_spec(dict(original, **patch))
            (root / "new-output").mkdir()
            with self.assertRaises(ValueError):
                self.s.validate_spec(original)

    def test_environment_cpu_and_gpu_authorization_removed(self):
        env = self.s.child_environment(dict(asset_root="asset", cache_root="cache"),
            dict(PATH="keep", INFORMS_GPU_TOKEN="bad", INFORMS_GPU_OWNER_PID="11",
                 CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", NVIDIA_VISIBLE_DEVICES="all"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["OMP_NUM_THREADS"], "2")
        self.assertEqual(env["INFORMS_CACHE_ROOT"], "cache")
        self.assertFalse(any(k.startswith("INFORMS_GPU_") for k in env))
        self.assertNotIn("CUDA_DEVICE_ORDER", env)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", env)

    def test_disappearing_descendant_does_not_drop_other_processes(self):
        gone, live = FakeProcess(2), FakeProcess(3)
        gone.dead = True
        root = FakeProcess(1, children=[gone, live])
        api = FakeAPI(root, gone, live)
        tree = self.s.OwnedTree(api, dict(pid=1, create_time=1.))
        tree.observe()
        self.assertEqual({r["pid"] for r in tree.identities()}, {1, 3})
        self.assertEqual(tree.sample()["RSS_bytes"], 24)

    def test_pid_reuse_never_signalled_and_known_child_survives_launcher(self):
        root, runtime = FakeProcess(1, children=[]), FakeProcess(2)
        root.child_rows = [runtime]
        api = FakeAPI(root, runtime)
        tree = self.s.OwnedTree(api, dict(pid=1, create_time=1.))
        tree.observe()
        root.dead = True
        api.processes[1] = FakeProcess(1, created=2.)
        tree.observe()
        self.assertEqual(tree.sample()["live_pids"], [2])
        result = tree.stop("test")
        self.assertEqual(api.processes[1].actions, [])
        self.assertEqual(runtime.actions, ["terminate", "kill"])
        self.assertEqual(api.waits, [10, 5])
        self.assertEqual(result["remaining_owned_processes"], [])

    def test_enumeration_error_is_recorded_without_destroying_known_tree(self):
        root, runtime = FakeProcess(1), FakeProcess(2)
        root.child_rows = [runtime]
        api = FakeAPI(root, runtime)
        tree = self.s.OwnedTree(api, dict(pid=1, create_time=1.))
        tree.observe()
        def denied(recursive=False):
            raise PermissionError()
        root.children = denied
        tree.observe()
        self.assertTrue(tree.errors)
        self.assertEqual({r["pid"] for r in tree.identities()}, {1, 2})

    def test_default_metadata_has_no_spec_or_process_execution(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(self.s.main([]), 0)
        value = json.loads(stream.getvalue())
        self.assertEqual(value["status"], "METADATA_ONLY")
        self.assertEqual(value["child_processes_started"], 0)

    def test_actual_two_owned_launchers_exit_before_runtime_deadline_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = self.spec(root)
            # Both launchers live long enough for the fixed two-second observation.
            command = [sys.executable, "-u", "-c",
                       "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True); time.sleep(3.2)"]
            spec["deadline_unix"] = time.time() + 12
            spec["processes"] = [dict(id=f"worker{i}", command=command, cwd=str(root), max_seconds=5) for i in range(2)]
            result = self.s.supervise(spec)
            self.assertEqual(result["status"], "STOPPED_RESOURCE_OR_WORKER_FAILURE")
            self.assertEqual(len(result["processes"]), 2)
            for row in result["processes"]:
                self.assertEqual(row["returncode"], 0)
                self.assertEqual(row["stop_reason"], "WORKER_WALL_LIMIT")
                self.assertGreaterEqual(len(row["owned_processes"]), 2)
                self.assertTrue(row["observed_tree_join_confirmed"])
                self.assertEqual(row["remaining_owned_processes"], [])
            self.assertEqual(result["remaining_owned_processes"], [])
            saved = json.loads((root / "new-output" / "RESULT.json").read_text())
            self.assertEqual(saved, result)
            self.assertTrue((root / "new-output" / "PROGRESS.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
