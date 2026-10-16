"""Synthetic CPU-only base-fit chronology, identity and resume checks."""
import importlib.util
import pathlib
import tempfile
import time
import unittest

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent


def toy_task():
    rng = np.random.default_rng(12)
    values = np.cumsum(rng.normal(size=(240, 3)), axis=0).astype(np.float32)
    return dict(name="toy", physical=values, observed=np.ones_like(values, bool),
                columns=["a", "b", "c"], L=64, H=4, data_sha256="toy-data")


class FitBaseTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "fit_base.py"
        if not path.is_file():
            self.fail("Required chronological base fit is not implemented")
        spec = importlib.util.spec_from_file_location("fit_base", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temporary = tempfile.TemporaryDirectory(prefix="informs_base_fit_")
        self.root = pathlib.Path(self.temporary.name)
        def cleanup():
            assert self.root.resolve().is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve())
            assert self.root.name.startswith("informs_base_fit_")
            self.temporary.cleanup()
        self.addCleanup(cleanup)
        torch.set_num_threads(2)

    def test_direct_cpu_search_refit_and_exact_contract_reuse(self):
        spec = dict(run_id="toy_direct", family="DIRECT_G", seed=3, epochs=[0.25],
                    lr=1e-3, device="cpu", microbatch=4, effective_batch=16)
        receipt = self.module.fit_base(spec, toy_task(), self.root / "cache", self.root / "public", None, None)
        self.assertEqual(receipt["status"], "COMPLETE_VALID")
        self.assertEqual(receipt["training_status"], "LIMITED_SCREEN")
        self.assertEqual(receipt["optimizer"], "AdamW")
        self.assertEqual(receipt["weight_decay"], 0)
        self.assertIsInstance(receipt["last_cp_best"], bool)
        self.assertIsInstance(receipt["last_scores_still_falling"], bool)
        self.assertEqual(receipt["search_fit_end"], 120)
        self.assertEqual(receipt["refit_fit_end"], 144)
        self.assertEqual(receipt["search_train_origins"], 53)
        self.assertEqual(receipt["search_stop_origins"], 6)
        self.assertEqual(receipt["search_exposures"], 14)
        self.assertEqual(receipt["F_logical_streams"], 0)
        bundle = torch.load(self.root / "cache" / receipt["artifact_cache_relative_path"], weights_only=False)
        self.assertEqual(bundle["normalization"]["fit_end"], 144)
        np.testing.assert_allclose(bundle["normalization"]["mean"], toy_task()["physical"][:144].mean(0), rtol=1e-6)
        self.assertIn(receipt["selected_epoch"], (0, .25))
        reused = self.module.fit_base(spec, toy_task(), self.root / "cache", self.root / "public", None, None)
        self.assertTrue(reused["reused_complete_artifact"])
        changed = dict(spec, lr=1e-4)
        with self.assertRaises(ValueError):
            self.module.fit_base(changed, toy_task(), self.root / "cache", self.root / "public", None, None)

    def test_full_channel_independent_adapter_initialization_is_identity(self):
        task = toy_task()
        for family in ("B0", "B2", "B4"):
            spec = dict(family=family, latent_k=3, seed=1, device="cpu")
            model, normalization = self.module.create_model(spec, task, 120, object())
            torch.testing.assert_close(model.E, torch.eye(3), atol=0, rtol=0)
            torch.testing.assert_close(model.center, torch.zeros(3), atol=0, rtol=0)
            self.assertEqual(normalization["basis_initialization"], "identity_at_K_equals_C")

    def test_resume_restores_next_batch_optimizer_rng_and_exact_update(self):
        training = self.module.training
        networks = self.module.networks
        task = toy_task()
        windows = training.FastWindows(task, np.zeros(3), np.ones(3), "cpu")
        origins = np.arange(64, 117)
        contract = {key: "same" for key in ("source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")}
        torch.manual_seed(42)
        a = networks.DirectG(64, 4)
        opt_a = torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=0)
        stream_a = training.PermutationStream(origins, 5)
        training.train_segment(a, windows, origins, stream_a, opt_a, 16, lambda x,k:a(x))
        bundle = training.checkpoint_bundle(a, opt_a, stream_a, contract, 1)
        import copy
        bundle = copy.deepcopy(bundle)
        expected_rng = torch.rand(4)
        training.train_segment(a, windows, origins, stream_a, opt_a, 32, lambda x,k:a(x), starting_updates=1)
        b = networks.DirectG(64, 4)
        opt_b = torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=0)
        stream_b = training.PermutationStream(origins, 5)
        training.restore_bundle(bundle, b, opt_b, stream_b, contract)
        torch.testing.assert_close(torch.rand(4), expected_rng, atol=0, rtol=0)
        training.train_segment(b, windows, origins, stream_b, opt_b, 32, lambda x,k:b(x), starting_updates=1)
        self.assertEqual(stream_a.state_dict(), stream_b.state_dict())
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=0, rtol=0)

    def test_budget_stop_then_resume_matches_uninterrupted_direct_fit(self):
        spec = dict(run_id="resumable", family="DIRECT_G", seed=3, epochs=[.25],
                    lr=1e-3, device="cpu", microbatch=4, effective_batch=16)
        interrupted = self.module.fit_base(spec, toy_task(), self.root / "resume", self.root / "resume_public", None, time.time() - 1)
        self.assertEqual(interrupted["status"], "BUDGET_LIMIT")
        resumed = self.module.fit_base(spec, toy_task(), self.root / "resume", self.root / "resume_public", None, None)
        full = self.module.fit_base(spec, toy_task(), self.root / "full", self.root / "full_public", None, None)
        self.assertEqual(resumed["model_state_sha"], full["model_state_sha"])
        self.assertEqual(resumed["selected_epoch"], full["selected_epoch"])

    def test_all_adapter_families_with_frozen_cpu_foundation_and_lora_cleanup(self):
        networks = self.module.networks
        class FakeFoundation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleDict({"q": torch.nn.Linear(64, 4), "v": torch.nn.Linear(64, 4)}).requires_grad_(False)
                self.initial_hash = networks.module_state_hash(self.model)
                self.streams = self.physical_streams = self.row_calls = self.physical_calls = 0
            def forward(self, x):
                rows = x.transpose(1, 2).reshape(-1, 64)
                self.streams += len(rows); self.physical_streams += len(rows)
                self.row_calls += 1; self.physical_calls += 1
                y = self.model["q"](rows) + self.model["v"](rows)
                return y.reshape(x.shape[0], x.shape[2], 4).transpose(1, 2)
            def assert_frozen_unchanged(self):
                if networks.module_state_hash(self.model) != self.initial_hash:
                    raise AssertionError("Synthetic foundation changed")
                if any(p.grad is not None for p in self.model.parameters()):
                    raise AssertionError("Synthetic foundation acquired parameter gradient")
        for family in ("B0", "B2", "B4", "LAG_CAUSAL", "LAG_BIDIR", "ADAPTS", "PCA_FIXED", "LORA", "DIRECT_B2", "DIRECT_B4"):
            with self.subTest(family=family):
                foundation = None if family.startswith("DIRECT") else FakeFoundation()
                spec = dict(run_id=family, family=family, latent_k=2, seed=3,
                            epochs=[.25], lr=1e-3, device="cpu", microbatch=4, effective_batch=16)
                receipt = self.module.fit_base(spec, toy_task(), self.root / "all_cache", self.root / "all_public", foundation, None)
                self.assertEqual(receipt["status"], "COMPLETE_VALID")
                if foundation is not None:
                    foundation.assert_frozen_unchanged()
                    self.assertTrue(all(not layer._forward_hooks for layer in foundation.model.values()))


if __name__ == "__main__":
    unittest.main()
