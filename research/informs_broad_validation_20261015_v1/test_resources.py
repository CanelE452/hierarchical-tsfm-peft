"""CPU-only OS lease and budget tests; no model or dataset imports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


class ResourcesTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "resources.py"
        self.assertTrue(path.is_file(), "The central OS GPU reservation is not implemented")
        spec = importlib.util.spec_from_file_location("informs_resources", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.temp = tempfile.TemporaryDirectory(prefix="informs_resource_test_")
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)

    def child(self, code, env=None):
        setup = "import sys;sys.path.insert(0," + repr(str(HERE)) + ");from resources import *;"
        return subprocess.run([sys.executable, "-c", setup + code], env=env,
                              capture_output=True, text=True, timeout=15)

    def test_real_child_cannot_take_parent_os_lock_and_receipt_has_no_token(self):
        with self.mod.GPUReservation(self.cache, total_gpu_seconds=30, reserve_seconds=2) as lease:
            result = self.child("\ntry:\n with GPUReservation(" + repr(str(self.cache)) +
                                ",total_gpu_seconds=30,reserve_seconds=0): pass\nexcept ResourceBusy: print('BUSY')")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "BUSY")
            self.assertGreater(lease.remaining, 0)
            lease.check()
        receipts = list((self.cache / "gpu_receipts").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text())
        self.assertNotIn("token", receipt)
        self.assertGreater(receipt["charged_process_seconds"], 0)
        self.assertEqual(receipt["status"], "CLOSED")

    def test_authenticated_child_requires_live_owner_and_correct_token(self):
        with self.mod.GPUReservation(self.cache, total_gpu_seconds=30, reserve_seconds=0) as lease:
            env = dict(os.environ, **lease.child_environment())
            code = "\nwith validate_gpu_child(" + repr(str(self.cache)) + ") as child:\n child.check();print('AUTHORIZED')"
            result = self.child(code, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "AUTHORIZED")
            env["INFORMS_GPU_TOKEN"] = "wrong"
            denied = self.child(code, env)
            self.assertNotEqual(denied.returncode, 0)
        stale = self.child(code, dict(os.environ, **lease.child_environment()))
        self.assertNotEqual(stale.returncode, 0)

    def test_crashed_process_is_charged_once(self):
        code = "\nimport os,time\nx=GPUReservation(" + repr(str(self.cache)) + ",total_gpu_seconds=30,reserve_seconds=0);x.__enter__();time.sleep(.15);os._exit(7)"
        result = self.child(code)
        self.assertEqual(result.returncode, 7)
        prior = json.loads((self.cache / "ACTIVE_GPU_LEASE.json").read_text())
        with self.mod.GPUReservation(self.cache, total_gpu_seconds=30, reserve_seconds=0):
            pass
        records = [json.loads(x) for x in (self.cache / "GPU_USAGE.jsonl").read_text().splitlines()]
        recovered = [x for x in records if x.get("lease_id") == prior["lease_id"]]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["status"], "CRASH_RECOVERED_UNKNOWN_CONSERVATIVE")
        self.assertGreaterEqual(recovered[0]["charged_process_seconds"], .15)
        (self.cache / "ACTIVE_GPU_LEASE.json").write_text(json.dumps(prior))
        with self.mod.GPUReservation(self.cache, total_gpu_seconds=30, reserve_seconds=0):
            pass
        records = [json.loads(x) for x in (self.cache / "GPU_USAGE.jsonl").read_text().splitlines()]
        self.assertEqual(sum(x.get("lease_id") == prior["lease_id"] for x in records), 1)

    def test_pid_reuse_is_not_live_owner_and_reserve_is_enforced(self):
        import psutil
        now = time.time()
        prior = dict(lease_id="reused_pid", pid=os.getpid(), create_time=psutil.Process().create_time()-10,
                     started_unix=now-3, deadline=now+3, token="private")
        (self.cache / "ACTIVE_GPU_LEASE.json").write_text(json.dumps(prior))
        with self.mod.GPUReservation(self.cache, total_gpu_seconds=10, reserve_seconds=6) as lease:
            self.assertLess(lease.remaining, 1.1)
        with self.assertRaises(self.mod.BudgetExceeded):
            with self.mod.GPUReservation(self.cache, total_gpu_seconds=3, reserve_seconds=0):
                pass
        with self.assertRaises(self.mod.BudgetExceeded):
            with self.mod.GPUReservation(self.cache, total_gpu_seconds=30, reserve_seconds=0,
                                         wall_deadline=time.time()-1):
                pass

    def test_existing_usage_and_storage_download_caps(self):
        old = '{"kind":"old_probe","charged_process_seconds":8}\n'
        (self.cache / "GPU_USAGE.jsonl").write_text(old)
        with self.assertRaises(self.mod.BudgetExceeded):
            with self.mod.GPUReservation(self.cache, total_gpu_seconds=10, reserve_seconds=3):
                pass
        self.assertEqual((self.cache / "GPU_USAGE.jsonl").read_text(), old)
        (self.cache / "bytes.bin").write_bytes(b"x"*1024)
        with self.assertRaises(self.mod.BudgetExceeded):
            self.mod.check_storage_budget(self.cache, max_bytes=100)
        self.mod.record_download(self.cache, "toy", 10, max_bytes=20)
        with self.assertRaises(self.mod.BudgetExceeded):
            self.mod.record_download(self.cache, "toy2", 11, max_bytes=20)

    def test_dead_owner_with_still_live_recorded_child_is_not_reclaimed(self):
        import psutil
        now = time.time()
        prior = dict(lease_id="orphan_child", pid=os.getpid(), create_time=psutil.Process().create_time()-10,
                     started_unix=now-3, deadline=now+3, token="private",
                     authorized_children=[dict(pid=os.getpid(), create_time=psutil.Process().create_time())])
        (self.cache / "ACTIVE_GPU_LEASE.json").write_text(json.dumps(prior))
        with self.assertRaises(self.mod.ResourceBusy):
            with self.mod.GPUReservation(self.cache, total_gpu_seconds=10, reserve_seconds=0):
                pass
        self.assertFalse((self.cache / "GPU_USAGE.jsonl").exists())

    def test_prior_download_receipt_and_metadata_transfers_are_included(self):
        directory = self.cache / "sources"
        directory.mkdir()
        (directory / "electricity_download_receipt.json").write_text(json.dumps(dict(download_bytes_this_invocation=8)))
        (directory / "source_evidence_receipt.json").write_text(json.dumps(dict(a=dict(download_bytes_this_invocation=3))))
        with self.assertRaises(self.mod.BudgetExceeded):
            self.mod.record_download(self.cache, "new", 10, max_bytes=20)
        self.assertEqual(self.mod.record_download(self.cache, "new", 9, max_bytes=20), 20)


if __name__ == "__main__":
    unittest.main()
