"""CPU-only paired time blocks: masks, immutable metadata, and exact cohorts."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_contract as data
import prediction_cache


class TimeBlockTests(unittest.TestCase):
    def setUp(self):
        import time_block_analysis
        self.module = time_block_analysis

    def record(self, pred, target, mask, run_id="left", origins=None):
        n, horizon, channels = pred.shape
        origins = np.arange(n, dtype=np.int64) * horizon if origins is None else origins
        return self.module.record_from_arrays(dict(pred_normalized=pred, target_normalized=target,
            observed=mask, origins=origins, channel_ids=np.asarray([str(i) for i in range(channels)])),
            dict(run_id=run_id, dataset="toy", fold="primary", cohort="toy:primary:V_select", data_sha="d", scaler_sha="s",
                 family="CURRENT", seed=1, k=0, budget="fixed", parent_signature="", anchors={}))

    def test_masked_macro_and_overall_are_not_cell_or_block_means(self):
        p = np.zeros((3, 2, 2), np.float32); y = np.zeros_like(p); m = np.ones_like(p, bool)
        p[0, :, 0] = 1.; p[1:, :, 0] = 4.; p[:, :, 1] = 3.
        m[1:, :, 1] = False
        left, right = self.record(p, y, m), self.record(np.zeros_like(p), y, m, "right")
        overall = self.module.paired_summary(left, right, np.arange(3))
        self.assertAlmostEqual(overall["delta_mse"], ((2 + 64) / 6 + 9) / 2)
        self.assertEqual(overall["active_channels"], 2)
        self.assertEqual(overall["observed_target_cells"], 8)
        a = self.module.paired_summary(left, right, np.asarray([0]))
        b = self.module.paired_summary(left, right, np.asarray([1, 2]))
        self.assertNotAlmostEqual(overall["delta_mse"], (a["delta_mse"] + b["delta_mse"]) / 2)
        self.assertNotAlmostEqual(overall["delta_mse"], 84 / 8)

    def test_metadata_blocks_no_overlap_and_short_tail(self):
        origins = np.arange(100, 100 + 7 * 4, 4, dtype=np.int64)
        plan = self.module.block_plan(origins, horizon=4, period=10)
        self.assertEqual(plan["block_origins"], 3)
        self.assertEqual([b["indices"] for b in plan["blocks"]], [[0, 1, 2], [3, 4, 5], [6]])
        self.assertTrue(plan["blocks"][-1]["short_tail"])
        all_indices = [i for block in plan["blocks"] for i in block["indices"]]
        self.assertEqual(all_indices, list(range(7)))
        self.assertEqual(plan["blocks"][0]["target_end_exclusive"], plan["blocks"][1]["origin_start"])
        with self.assertRaisesRegex(ValueError, "strideH"):
            self.module.block_plan(np.asarray([10, 14, 19]), 4, 10)

    def test_same_epoch_seed_k_but_different_or_unknown_lr_is_not_matched(self):
        p = np.ones((2, 2, 2), np.float32); m = np.ones_like(p, bool)
        left, right = self.record(p, p, m), self.record(p, p, m, "right")
        left["identity"].update(family="B2", optimizer_trained=True, lr_known=True, learning_rate=.001, fit_config_sha="left")
        right["identity"].update(family="B0", optimizer_trained=True, lr_known=True, learning_rate=.0001, fit_config_sha="right")
        self.assertFalse(self.module._recipe_pair(left, right))
        right["identity"]["learning_rate"] = .001
        self.assertTrue(self.module._recipe_pair(left, right))
        right["identity"]["lr_known"] = False
        self.assertFalse(self.module._recipe_pair(left, right))

    def test_target_mask_channel_origin_and_cohort_mismatches_rejected(self):
        p = np.ones((3, 2, 2), np.float32); y = np.zeros_like(p); m = np.ones_like(p, bool)
        left = self.record(p, y, m)
        for field in ("target", "mask", "channel", "origin", "cohort", "scaler", "data"):
            right = self.record(p, y, m, "right")
            if field in {"target", "mask", "channel", "origin"}:
                key = {"target": "target_normalized", "mask": "observed", "channel": "channel_ids", "origin": "origins"}[field]
                right["array_contract"][key]["sha256"] = "swapped"
            elif field == "cohort": right["identity"]["cohort"] = "other"
            else: right["identity"][field + "_sha"] = "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.paired_summary(left, right, np.arange(3))

    def test_actual_toy_generation_future_suffix_invariant_and_originals_preserved(self):
        with tempfile.TemporaryDirectory(prefix="informs_time_blocks_") as tmp:
            public, cache = Path(tmp) / "public", Path(tmp) / "cache"
            public.mkdir(); cache.mkdir()
            x = np.cumsum(np.random.default_rng(2).normal(size=(240, 3)), 0).astype(np.float32)
            task = dict(name="toy", physical=x, observed=np.ones_like(x, bool), columns=["a", "b", "c"],
                        L=64, H=4, period=10, data_sha256="toy_data")
            rows = []
            for name, family, baseline in (("current", "CURRENT", "CURRENT"), ("seasonal", "SEASONAL_NAIVE", "SEASONAL")):
                prediction_cache.predict_parent(name, task, "V_select", cache, public, None, baseline=dict(kind=baseline, period=10))
                rows.append(dict(experiment_id=name, dataset="toy", family=family, seed=1, latent_k=0, fold="primary"))
            with (public / "EXPERIMENT_REGISTRY.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            immutable = {str(p): data.sha256_file(p) for p in list(public.rglob("*.json")) + list(cache.rglob("*.npz"))}
            with patch.object(self.module.data, "load_task", return_value=task):
                first = self.module.generate(public, cache)
            changed = copy.deepcopy(task); changed["physical"][192:] = np.nan; changed["observed"][192:] = False
            with patch.object(self.module.data, "load_task", return_value=changed):
                second = self.module.generate(public, cache)
            for name in ("OVERALL_PAIRED.csv", "TIME_BLOCK_PAIRED.csv"):
                self.assertEqual((Path(first) / name).read_bytes(), (Path(second) / name).read_bytes())
            self.assertEqual(immutable, {name: data.sha256_file(name) for name in immutable})
            manifest = json.loads((Path(first) / "TIME_BLOCK_MANIFEST.json").read_text())
            self.assertEqual(manifest["paired_comparisons"], 1)
            self.assertFalse(manifest["blocks_assumed_IID"])
            self.assertFalse(manifest["confidence_intervals_computed"])
            with self.assertRaisesRegex(ValueError, "SELECTION_LOCK"):
                self.module.generate(public, cache, split="T_final")


if __name__ == "__main__":
    unittest.main()
