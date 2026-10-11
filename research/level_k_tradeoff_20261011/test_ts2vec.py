"""CPU-only native compatibility fixtures; root owns the scientific GPU runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest

import numpy as np
import torch

import ts2vec_adapter as adapter_module


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "cpu_fixture_artifacts"
RECEIPT = HERE / "cpu_fixture_receipt.json"
STARTED = time.perf_counter()
FIXTURE_FIT = None


class NativeTs2VecFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global FIXTURE_FIT
        torch.set_num_threads(2)
        FIXTURE.mkdir(exist_ok=True)
        configuration = {
            "hidden_dims": 16, "depth": 3, "output_dims": "dataset K",
            "max_train_length": 512, "batch_size": 32,
            "learning_rate": 0.001, "temporal_unit": 0,
            "n_iters": 1, "n_epochs": None,
        }
        prior_models = sys.modules.get("models")
        prior_utils = sys.modules.get("utils")
        sentinel_models = types.ModuleType("models")
        sentinel_utils = types.ModuleType("utils")
        sys.modules["models"] = sentinel_models
        sys.modules["utils"] = sentinel_utils
        cls.adapter = adapter_module.NativeTs2VecAdapter(3, 2, 314159, "cpu", configuration)
        cls.alias_restored = sys.modules.get("models") is sentinel_models and sys.modules.get("utils") is sentinel_utils
        for name, prior in (("models", prior_models), ("utils", prior_utils)):
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
        windows = np.random.default_rng(2718).normal(size=(32, 16, 3)).astype(np.float32)
        FIXTURE_FIT = cls.adapter.fit(windows)

    def test_vendor_hashes_and_namespace_restoration(self):
        receipt = adapter_module.vendor_provenance()
        self.assertEqual(len(receipt["files"]), 7)
        self.assertTrue(self.alias_restored)

    def test_actual_single_native_update_and_crop(self):
        self.assertEqual(FIXTURE_FIT["actual_optimizer_steps"], 1)
        self.assertEqual(FIXTURE_FIT["native_iterations"], 1)
        self.assertEqual(FIXTURE_FIT["swa_n_averaged"], 2)
        self.assertEqual(len(FIXTURE_FIT["iteration_losses"]), 1)
        self.assertTrue(np.isfinite(FIXTURE_FIT["iteration_losses"][0]["loss"]))
        self.assertEqual(FIXTURE_FIT["optimizer_parameters"]["weight_decay"], 0.01)
        optimizer_steps = [int(item["step"].item()) for item in self.adapter.optimizer_state["state"].values()]
        self.assertTrue(optimizer_steps)
        self.assertEqual(set(optimizer_steps), {1})

    def test_small_k_shape_frozen_state_and_input_preservation(self):
        values = torch.from_numpy(np.random.default_rng(99).normal(size=(2, 40, 3)).astype(np.float32))
        original = values.clone()
        before = adapter_module.state_hash(self.adapter)
        encoded = self.adapter.encode(values)
        self.assertEqual(tuple(encoded.shape), (2, 40, 2))
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertTrue(torch.equal(values, original))
        self.assertEqual(before, adapter_module.state_hash(self.adapter))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in self.adapter.parameters()))

    def test_future_perturbation_causality(self):
        values = torch.from_numpy(np.random.default_rng(77).normal(size=(2, 40, 3)).astype(np.float32))
        changed = values.clone()
        changed[:, 19:] += 17
        first = self.adapter.encode(values)
        second = self.adapter.encode(changed)
        self.assertTrue(torch.equal(first[:, :19], second[:, :19]))
        self.assertFalse(torch.equal(first[:, 19:], second[:, 19:]))

    def test_200_padding_finite_and_past_boundary(self):
        values = torch.from_numpy(np.random.default_rng(19).normal(size=(1, 260, 3)).astype(np.float32))
        changed = values.clone()
        changed[:, :20] += 31
        first = self.adapter.encode(values)
        second = self.adapter.encode(changed)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first[:, 259], second[:, 259]))

    def test_wrapper_native_export_parity(self):
        values = np.random.default_rng(14).normal(size=(2, 24, 3)).astype(np.float32)
        native = self.adapter.native.encode(values.copy(), encoding_window=None, causal=True, sliding_length=1, sliding_padding=200, batch_size=256)
        wrapped = self.adapter.encode(torch.from_numpy(values)).numpy()
        np.testing.assert_array_equal(native, wrapped)

    def test_native_save_reload_and_wrapper_checkpoint(self):
        native_path = FIXTURE / "native_one_step.pkl"
        wrapper_path = FIXTURE / "wrapper_one_step.pt"
        self.adapter.native.save(native_path)
        other = adapter_module.NativeTs2VecAdapter(3, 2, 999, "cpu", self.adapter.configuration)
        other.native.load(native_path)
        other.freeze()
        values = torch.from_numpy(np.random.default_rng(42).normal(size=(1, 16, 3)).astype(np.float32))
        np.testing.assert_array_equal(self.adapter.encode(values).numpy(), other.encode(values).numpy())
        self.adapter.save_checkpoint(wrapper_path, {"role": "CPU fixture only", "scientific_fit": False})
        loaded, payload = adapter_module.NativeTs2VecAdapter.load_checkpoint(wrapper_path, "cpu")
        self.assertEqual(payload["native_iterations"], 1)
        self.assertEqual(adapter_module.state_hash(self.adapter), adapter_module.state_hash(loaded))
        np.testing.assert_array_equal(self.adapter.encode(values).numpy(), loaded.encode(values).numpy())


if __name__ == "__main__":
    if RECEIPT.exists():
        raise SystemExit("CPU fixture receipt already exists; do not silently repeat smoke training")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeTs2VecFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "schema": "native_ts2vec_cpu_fixture_v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "wall_seconds": time.perf_counter() - STARTED,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "device": "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "actual_smoke_optimizer_steps": FIXTURE_FIT["actual_optimizer_steps"] if FIXTURE_FIT else 0,
        "scientific_fit_starts": 0,
        "backbone_calls": 0,
        "fit_receipt": FIXTURE_FIT,
        "fixture_source_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_source_sha256": __import__("hashlib").sha256((HERE / "ts2vec_adapter.py").read_bytes()).hexdigest(),
        "vendor_commit": adapter_module.COMMIT,
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
