"""CPU-only continuation: full state, immutable parents, and strict identities."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

import fit_base
from test_fit_base import toy_task


class LearningExtensionTests(unittest.TestCase):
    def setUp(self):
        import learning_extension
        self.module = learning_extension
        self.temporary = tempfile.TemporaryDirectory(prefix="informs_extension_")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.cache, self.public = self.root / "cache", self.root / "public"
        self.task = toy_task()
        self.spec = dict(run_id="parent", family="DIRECT_G", seed=3, epochs=[.25],
                         lr=.001, device="cpu", microbatch=4, effective_batch=16)
        torch.set_num_threads(2)
        self.parent = fit_base.fit_base(self.spec, self.task, self.cache, self.public, None, None)

    def extend(self, new_id="extended", epochs=None, task=None):
        return self.module.extend_fit("parent", new_id, epochs or [.25, 1., 3.],
                                      task or self.task, self.cache, self.public, None, None)

    def latest(self, run_id, stage):
        return torch.load(sorted((self.cache / "runs" / run_id / stage).glob("checkpoint_*.pt"))[-1],
                          map_location="cpu", weights_only=False)

    def test_exact_full_state_continuation_and_fresh_selected_refit(self):
        before = self.module.parent_inventory("parent", self.cache, self.public)
        old = self.latest("parent", "search")
        result = self.extend()
        full = fit_base.fit_base(dict(self.spec, run_id="scratch", epochs=[.25, 1., 3.]),
                                 self.task, self.cache, self.public, None, None)
        for stage in ("search", "refit"):
            a, b = self.latest("extended", stage), self.latest("scratch", stage)
            for key in ("model", "optimizer", "sampler", "torch_rng", "cuda_rng", "updates"):
                self.assertEqual(self.module.state_fingerprint(a[key]), self.module.state_fingerprint(b[key]), key)
            self.assertEqual(self.module.state_fingerprint(a["extra"]["best_state"]),
                             self.module.state_fingerprint(b["extra"]["best_state"]))
        self.assertEqual(result["model_state_sha"], full["model_state_sha"])
        self.assertEqual(result["selected_epoch"], full["selected_epoch"])
        self.assertEqual(result["selected_astop_mse"], full["selected_astop_mse"])
        accounting = result["extension_accounting"]
        self.assertEqual(accounting["inherited_search_updates"], old["updates"])
        self.assertEqual(accounting["new_search_updates"], result["search_updates"] - old["updates"])
        self.assertEqual(accounting["new_refit_updates"], result["refit_updates"])
        self.assertEqual(before, self.module.parent_inventory("parent", self.cache, self.public))
        self.assertTrue(result["extension_provenance"]["payload_unchanged"])
        self.assertTrue(result["extension_provenance"]["fresh_A60_refit"])

    def test_complete_extension_reuse_is_strict_and_does_not_write(self):
        first = self.extend()
        before = self.module.parent_inventory("extended", self.cache, self.public)
        reused = self.extend()
        self.assertTrue(reused["reused_complete_artifact"])
        self.assertEqual(first["artifact_sha256"], reused["artifact_sha256"])
        self.assertEqual(before, self.module.parent_inventory("extended", self.cache, self.public))
        with self.assertRaisesRegex(ValueError, "migration|identity|contract"):
            self.extend(epochs=[.25, 1., 4.])

    def test_checkpoint_contract_and_optimizer_recipe_tampering_rejected(self):
        path = sorted((self.cache / "runs" / "parent" / "search").glob("checkpoint_*.pt"))[-1]
        original = torch.load(path, weights_only=False)
        for change in ("scaler_sha", "initialization_sha", "optimizer_lr", "sampler"):
            with self.subTest(change=change):
                cp = copy.deepcopy(original)
                if change == "optimizer_lr":
                    cp["optimizer"]["param_groups"][0]["lr"] = .1
                elif change == "sampler":
                    cp["sampler"]["position"] += 1
                else:
                    cp["contract"][change] = "changed"
                torch.save(cp, path)
                with self.assertRaises(ValueError):
                    self.extend(new_id="bad_" + change)
        torch.save(original, path)

    def test_source_data_foundation_and_lr_changes_rejected(self):
        source_path = self.public / "runs" / "parent" / "SOURCE_CONTRACT.json"
        source = json.loads(source_path.read_text())
        changed = copy.deepcopy(source)
        changed["spec"]["lr"] = .01
        source_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "config|recipe"):
            self.extend()
        source_path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "data"):
            self.extend(task=dict(self.task, data_sha256="different"))
        changed = copy.deepcopy(source)
        changed["files"]["fit_base.py"] = "different"
        source_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "source"):
            self.extend()
        source_path.write_text(json.dumps(source))
        class DifferentFoundation:
            initial_hash = "different"
        with self.assertRaisesRegex(ValueError, "model|foundation"):
            self.module.extend_fit("parent", "bad_f", [.25, 1.], self.task, self.cache,
                                   self.public, DifferentFoundation(), None)

    def test_original_segment_boundary_cannot_be_removed_or_shortened(self):
        for epochs in ([1., 3.], [.25], [.1, .25, 3.], [.25, .25, 1.]):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                self.extend(epochs=epochs)

    def test_parent_change_after_migration_and_foreign_new_id_rejected(self):
        self.extend()
        artifact = self.cache / self.parent["artifact_cache_relative_path"]
        artifact.write_bytes(artifact.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact|parent"):
            self.extend()
        (self.cache / "runs" / "foreign").mkdir(parents=True)
        (self.cache / "runs" / "foreign" / "unrelated.txt").write_text("existing")
        with self.assertRaises(ValueError):
            self.extend(new_id="foreign")

    def test_budget_stop_then_resume_is_exact_and_diagnostic_limitation_persists(self):
        limited = self.module.extend_fit("parent", "resume", [.25, 1., 3.], self.task,
                                         self.cache, self.public, None, time.time() - 1)
        self.assertEqual(limited["status"], "BUDGET_LIMIT")
        self.assertFalse(limited["extension_accounting"]["clipping_diagnostics_complete"])
        resumed = self.extend(new_id="resume")
        full = fit_base.fit_base(dict(self.spec, run_id="scratch", epochs=[.25, 1., 3.]),
                                 self.task, self.cache, self.public, None, None)
        self.assertEqual(resumed["model_state_sha"], full["model_state_sha"])
        self.assertFalse(resumed["extension_accounting"]["clipping_diagnostics_complete"])
        self.assertTrue(resumed["extension_accounting"]["budget_interruption_events"])

    def test_changed_actual_initialization_and_migrated_payload_rejected(self):
        changed = copy.deepcopy(self.task)
        changed["physical"] = changed["physical"] + np.float32(2.)
        with self.assertRaisesRegex(ValueError, "initialization|scaler"):
            self.extend(task=changed)
        self.extend()
        path = self.cache / "runs" / "extended" / "search" / "checkpoint_000000.pt"
        cp = torch.load(path, weights_only=False)
        cp["torch_rng"][0] = (int(cp["torch_rng"][0]) + 1) % 255
        torch.save(cp, path)
        with self.assertRaisesRegex(ValueError, "migrated checkpoint"):
            self.extend()

    def test_frozen_foundation_continuation_matches_scratch_and_cost_is_incremental(self):
        class FakeFoundation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(918)
                self.model = torch.nn.Linear(64, 4).requires_grad_(False)
                self.initial_hash = fit_base.networks.module_state_hash(self.model)
                self.streams = self.physical_streams = self.row_calls = self.physical_calls = 0
            def forward(self, x):
                rows = x.transpose(1, 2).reshape(-1, 64)
                self.streams += len(rows)
                self.physical_streams += len(rows)
                self.row_calls += 1
                self.physical_calls += 1
                return self.model(rows).reshape(x.shape[0], x.shape[2], 4).transpose(1, 2)
            def assert_frozen_unchanged(self):
                if fit_base.networks.module_state_hash(self.model) != self.initial_hash:
                    raise AssertionError("Synthetic frozen foundation changed")
        f = FakeFoundation()
        spec = dict(self.spec, run_id="fp", family="B0", latent_k=2)
        parent = fit_base.fit_base(spec, self.task, self.cache, self.public, f, None)
        result = self.module.extend_fit("fp", "fx", [.25, 1.], self.task, self.cache, self.public, f, None)
        full = fit_base.fit_base(dict(spec, run_id="ff", epochs=[.25, 1.]),
                                 self.task, self.cache, self.public, f, None)
        self.assertEqual(result["model_state_sha"], full["model_state_sha"])
        old = self.latest("fp", "search")
        cost = result["extension_accounting"]
        self.assertEqual(cost["inherited_search_F_logical_streams"], old["extra"]["total_streams"])
        self.assertGreater(cost["inherited_search_F_logical_streams"], 0)
        self.assertEqual(cost["new_total_F_logical_streams"],
                         result["F_logical_streams"] - cost["inherited_search_F_logical_streams"])
        self.assertEqual(parent["artifact_sha256"],
                         json.loads((self.public / "runs" / "fp" / "BASE_FIT_RECEIPT.json").read_text())["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
