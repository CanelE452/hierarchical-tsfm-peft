"""CPU synthetic tests; no real data, backbone, GPU or performance evidence."""
import copy
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

import combination as c
from data_contract import origins, split_ranges
from networks import ConditionalHead, parameter_count


class CombinationTests(unittest.TestCase):
    def test_macro_alpha_matches_dense_grid_with_uneven_mask(self):
        rng = np.random.default_rng(17)
        g, b, y = (rng.normal(size=(7, 4, 3)).astype(np.float32) for _ in range(3))
        mask = np.ones(y.shape, bool)
        mask[:6, :, 1] = False
        mask[:5, :, 2] = False
        actual = c.fit_alpha_family(g, b, y, mask)
        scores = [c.macro_metrics(g + a * (b - g), y, mask)["mse"] for a in np.linspace(0, 1, 10001)]
        self.assertLessEqual(actual["fitted"]["mse"], min(scores) + 1e-12)
        self.assertAlmostEqual(actual["alpha0"]["mse"], c.macro_metrics(g, y, mask)["mse"])
        self.assertEqual(c.fit_alpha_family(g, g, y, mask)["alpha_fit"], 0)

    def test_zero_head_preserves_g_and_control_has_equal_capacity(self):
        torch.manual_seed(8)
        x, g, b = torch.randn(3, 40, 5), torch.randn(3, 4, 5), torch.randn(3, 4, 5)
        conditional, past = ConditionalHead(40, 4), ConditionalHead(40, 4)
        self.assertEqual(parameter_count(conditional), parameter_count(past))
        torch.testing.assert_close(conditional(x, g, b), g, atol=0, rtol=0)
        torch.testing.assert_close(past(x, g, b, control=True), g, atol=0, rtol=0)
        b.requires_grad_(True)
        conditional(x, g, b).sum().backward()
        self.assertIsNone(b.grad)
        self.assertGreater(float(conditional.output.weight.grad.abs().sum()), 0)
        for control in (False, True):
            conditional.zero_grad(set_to_none=True)
            with torch.no_grad():
                conditional.output.weight.fill_(.1)
            conditional(x, g, b, control=control).square().sum().backward()
            grad = conditional.input.weight.grad
            for start, stop in ((0, 40), (40, 44), (44, 48)):
                self.assertGreater(float(grad[:, start:stop].abs().sum()), 0)

    def toy(self):
        rng = np.random.default_rng(12)
        y = rng.normal(size=(800, 3)).astype(np.float32)
        task = {"name": "synthetic", "physical": y, "observed": np.ones(y.shape, bool), "L": 40, "H": 4,
                "columns": ["a", "b", "c"], "data_sha256": "synthetic_data", "split_ranges": split_ranges(800)}
        bundles = []
        for split in ("M_cal", "V_select"):
            oo = origins(task["split_ranges"][split], task["L"], task["H"], 1 if split == "M_cal" else task["H"])
            target = np.stack([y[o:o + task["H"]] for o in oo])
            for label, delta in (("parent", .4), ("g", -.3)):
                bundles.append({"pred_normalized": target + delta, "target_normalized": target.copy(), "observed": np.ones(target.shape, bool),
                                "origins": oo.copy(), "channel_ids": np.array(task["columns"]),
                                "metadata": {"data_sha256": "synthetic_data", "scaler_sha": c.scaler_fingerprint(np.zeros(3), np.ones(3)), "model_sha": label,
                                             "cohort": f"synthetic_{split}", "split": split, "parent_fit_end": task["split_ranges"]["A_stop"][1]}})
        return task, bundles[0], bundles[2], bundles[1], bundles[3]

    def test_alignment_refuses_corrupt_channels_or_parent_cohort(self):
        task, pm, pv, gm, gv = self.toy()
        c.validate_parent_caches(task, pm, pv, gm, gv, np.zeros(3), np.ones(3))
        corrupt = copy.deepcopy(pm)
        corrupt["channel_ids"] = corrupt["channel_ids"][[1, 0, 2]]
        with self.assertRaises(ValueError):
            c.validate_parent_caches(task, corrupt, pv, gm, gv, np.zeros(3), np.ones(3))
        corrupt = copy.deepcopy(pm)
        corrupt["metadata"]["parent_fit_end"] = task["split_ranges"]["M_cal"][1]
        with self.assertRaises(ValueError):
            c.validate_parent_caches(task, corrupt, pv, gm, gv, np.zeros(3), np.ones(3))

    def test_real_toy_fit_receipt_and_replay_and_parent_unchanged(self):
        task, pm, pv, gm, gv = self.toy()
        before = c.cache_fingerprint(pm)
        with tempfile.TemporaryDirectory() as temp:
            spec = {"run_id": "toy_conditional", "variant": "conditional", "seed": 19, "device": "cpu", "epoch_endpoints": [.25, 1],
                    "effective_batch": 16, "microbatch": 4, "learning_rate": .001}
            receipt = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), Path(temp) / "cache", Path(temp) / "public", None)
            self.assertEqual(receipt["execution_status"], "COMPLETE_VALID")
            self.assertGreater(receipt["updates"], 0)
            self.assertEqual(receipt["foundation_calls_during_H_fit"], 0)
            self.assertTrue(receipt["parents_unchanged"])
            self.assertTrue((Path(temp) / "public/toy_conditional/curve.csv").is_file())
            self.assertEqual(c.cache_fingerprint(pm), before)
            head = c.load_conditional_checkpoint(Path(temp) / "cache/combinations/toy_conditional/selected.pt", "cpu")
            result = c.predict_head(head, task, pv, gv, np.zeros(3), np.ones(3), device="cpu")
            with np.load(Path(temp) / "cache/combinations/toy_conditional/V_select.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(result, saved["pred_normalized"], atol=1e-6, rtol=1e-6)

    def test_head_windows_never_materialize_T_final(self):
        task, pm, pv, gm, gv = self.toy()
        original = c.FastWindows
        lengths = []
        def bounded_windows(prefix, *args, **kwargs):
            lengths.append(len(prefix["physical"]))
            self.assertLessEqual(lengths[-1], task["split_ranges"]["T_final"][0])
            return original(prefix, *args, **kwargs)
        head = ConditionalHead(task["L"], task["H"])
        with patch.object(c, "FastWindows", side_effect=bounded_windows):
            result = c.predict_head(head, task, pv, gv, np.zeros(3), np.ones(3), device="cpu")
        np.testing.assert_array_equal(result, gv["pred_normalized"])
        self.assertTrue(lengths)

    def test_saved_past_control_replay_preserves_feature_contract(self):
        task, pm, pv, gm, gv = self.toy()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"run_id": "past", "variant": "past_control", "seed": 44, "device": "cpu", "epoch_endpoints": [.25, 1]}
            c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root / "cache", root / "public", None)
            head = c.load_conditional_checkpoint(root / "cache/combinations/past/selected.pt", "cpu")
            self.assertTrue(head.default_control)
            result = c.predict_head(head, task, pv, gv, np.zeros(3), np.ones(3), device="cpu")
            with np.load(root / "cache/combinations/past/V_select.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(result, saved["pred_normalized"], atol=1e-6, rtol=1e-6)

    def test_legacy_residual_avoids_current_double_count(self):
        g = np.full((2, 4, 3), 7, np.float32)
        b = np.full((2, 4, 3), 10, np.float32)
        current = np.full((2, 1, 3), 5, np.float32)
        np.testing.assert_array_equal(c.legacy_residual_prediction(b, g, current), np.full((2, 4, 3), 12))
        head = c.LegacyResidualHead(40, 4)
        x, gp, bp = torch.randn(2, 40, 3), torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        torch.testing.assert_close(head(x, gp, bp), bp, atol=0, rtol=0)

    def test_expired_budget_checkpoint_exact_resume_matches_uninterrupted(self):
        task, pm, pv, gm, gv = self.toy()
        with tempfile.TemporaryDirectory() as temp:
            spec = {"run_id": "paused", "variant": "legacy_residual", "seed": 9, "device": "cpu", "epoch_endpoints": [.25, 1],
                    "effective_batch": 16, "microbatch": 4, "learning_rate": .001}
            base = Path(temp)
            partial = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), base / "cache", base / "public", time.time() - 1)
            self.assertEqual(partial["execution_status"], "BUDGET_LIMIT")
            self.assertEqual(partial["updates"], 0)
            resumed_spec = dict(spec, run_id="resumed", resume_checkpoint=str(base / "cache" / partial["last_full_checkpoint_reference"]))
            resumed = c.fit_combination(resumed_spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), base / "cache", base / "public", None)
            whole = c.fit_combination(dict(spec, run_id="whole"), task, pm, pv, gm, gv, np.zeros(3), np.ones(3), base / "cache", base / "public", None)
            self.assertEqual(resumed["updates"], whole["updates"])
            self.assertEqual(resumed["last_state_sha"], whole["last_state_sha"])
            self.assertEqual(resumed["selected_state_sha"], whole["selected_state_sha"])

    def test_resume_restores_nonempty_optimizer_and_sampler(self):
        task, pm, pv, gm, gv = self.toy()
        counter, original_step = {"n": 0}, torch.optim.AdamW.step
        def counted_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            counter["n"] += 1
            return result
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"run_id": "partial", "variant": "conditional", "seed": 81, "device": "cpu", "epoch_endpoints": [.25, 1],
                    "effective_batch": 16, "microbatch": 4, "learning_rate": .001}
            with patch.object(torch.optim.AdamW, "step", new=counted_step), patch.object(c.time, "time", side_effect=lambda: 10. if counter["n"] >= 2 else 0.):
                partial = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root / "cache", root / "public", 5.)
            self.assertEqual(partial["execution_status"], "BUDGET_LIMIT")
            self.assertEqual(partial["updates"], 2)
            cp = root / "cache" / partial["last_full_checkpoint_reference"]
            saved = torch.load(cp, map_location="cpu", weights_only=True)
            self.assertTrue(saved["optimizer"]["state"])
            resumed = c.fit_combination(dict(spec, run_id="continued", resume_checkpoint=str(cp)), task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root / "cache", root / "public", None)
            whole = c.fit_combination(dict(spec, run_id="uninterrupted"), task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root / "cache", root / "public", None)
            self.assertEqual(resumed["last_state_sha"], whole["last_state_sha"])
            self.assertEqual(resumed["selected_state_sha"], whole["selected_state_sha"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
