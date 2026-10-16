"""Independent native-unit aggregation and strict pre-existing cache/scaler gates."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import data_contract
import prediction_cache


class PhysicalMetricTests(unittest.TestCase):
    def setUp(self):
        import physical_metrics
        self.module = physical_metrics

    def test_macro_and_pooled_with_missingness_and_channel_scales(self):
        p = np.ones((2, 2, 3)); y = np.zeros_like(p); mask = np.ones_like(p, bool)
        mask[0, 1:, 1] = False; mask[1, :, 1] = False; mask[:, :, 2] = False
        score, channels = self.module.physical_statistics(p, y, mask, np.array([2., 10., 4.]))
        self.assertEqual(score["physical_macro_mse"], 52.)
        self.assertEqual(score["physical_macro_mae"], 6.)
        self.assertAlmostEqual(score["physical_macro_rmse"], np.sqrt(52))
        self.assertEqual(score["physical_pooled_mse"], 116 / 5)
        self.assertEqual(score["physical_pooled_mae"], 18 / 5)
        self.assertEqual(score["active_channels"], 2)
        self.assertIsNone(channels[2]["physical_mse"])

    def test_invalid_scale_and_observed_nonfinite_are_rejected(self):
        p = np.ones((1, 2, 2)); m = np.ones_like(p, bool)
        for std in ([1., 0.], [1., np.nan], [1.]):
            with self.assertRaises(ValueError): self.module.physical_statistics(p, p, m, np.asarray(std))
        p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): self.module.physical_statistics(p, p, m, np.ones(2))

    def test_actual_toy_cache_without_task_or_target_access_and_bad_scaler(self):
        with tempfile.TemporaryDirectory(prefix="informs_physical_") as tmp:
            public, cache = Path(tmp) / "public", Path(tmp) / "cache"; public.mkdir(); cache.mkdir()
            task = dict(name="toy", physical=np.random.default_rng(4).normal(size=(120, 2)).astype(np.float32),
                        observed=np.ones((120, 2), bool), columns=["a", "b"], L=16, H=4, data_sha256="toy_data")
            prediction_cache.predict_parent("current", task, "V_select", cache, public, None, baseline=dict(kind="CURRENT"))
            (public / "EXPERIMENT_REGISTRY.csv").write_text("experiment_id,dataset,family,fold,seed,latent_k\ncurrent,toy,CURRENT,primary,1,0\n")
            ranges = data_contract.split_ranges(task)
            (public / "DATA_CONTRACT.json").write_text(json.dumps({"assets": {"toy": {"shape": [120, 2], "columns": task["columns"],
                 "data_sha256": "toy_data", "primary_split_ranges": ranges, "interval_seconds": 1}}}))
            mean, std, _ = data_contract.fit_statistics(task, ranges["A_stop"][1])
            supplied = {("toy", "primary"): dict(mean=mean, std=std, fit_end=ranges["A_stop"][1], data_sha="toy_data", input_files=[])}
            with patch.object(data_contract, "load_task", side_effect=AssertionError("No Task read")), patch.object(data_contract, "target_window", side_effect=AssertionError("No new target")):
                output = self.module.generate(public, cache, normalizations=supplied)
            manifest = json.loads((output / "PHYSICAL_METRIC_MANIFEST.json").read_text())
            self.assertEqual(manifest["prediction_count"], 1)
            self.assertEqual(manifest["new_target_windows"], 0)
            self.assertTrue((output / "PHYSICAL_SCORES.csv").is_file())
            supplied[("toy", "primary")]["std"] = std * 2
            bad = self.module.generate(public, cache, normalizations=supplied)
            bad_manifest = json.loads((bad / "PHYSICAL_METRIC_MANIFEST.json").read_text())
            self.assertEqual(bad_manifest["prediction_count"], 0)
            self.assertTrue(bad_manifest["excluded_prediction_inputs"])
            with self.assertRaisesRegex(ValueError, "SELECTION_LOCK"):
                self.module.generate(public, cache, split="T_final", normalizations=supplied)


if __name__ == "__main__": unittest.main()
