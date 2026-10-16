"""CPU-only checks for raw-history ridge semantics and masked chronology."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


class RidgeTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "ridge_baseline.py"
        self.assertTrue(path.is_file(), "The own-channel raw ridge baseline is missing")
        spec = importlib.util.spec_from_file_location("ridge_baseline", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.rng = np.random.default_rng(13)

    def task(self):
        values = self.rng.normal(size=(240, 3)).astype(np.float32).cumsum(0)
        mask = np.ones_like(values, bool)
        mask[10:100, 0] = False
        mask[::5, 1] = False
        values[:, 2] = 4
        values[~mask] = np.nan
        return dict(name="toy", physical=values, observed=mask, L=64, H=4,
                    columns=["a", "b", "constant"], data_sha256="toy")

    def test_prefix_features_equal_each_reset_history_window_and_use_no_future(self):
        import data_contract
        task = self.task()
        mean, std, _ = data_contract.fit_statistics(task, 120)
        origins = np.array([64, 78, 100, 120, 140])
        prepared = self.mod.PrefixFeatures(task, mean, std, 144)
        expected = np.concatenate([self.mod.temporal_features(data_contract.history_window(task, int(o), mean, std)[0][None])
                                   for o in origins])
        for channel in range(3):
            np.testing.assert_allclose(prepared.channel_features(channel, origins), expected[:, channel], atol=1e-12, rtol=1e-12)
        changed = dict(task, physical=task["physical"].copy())
        changed["physical"][120:] = 100000
        second = self.mod.PrefixFeatures(changed, mean, std, 144)
        np.testing.assert_array_equal(prepared.channel_features(0, np.array([120])), second.channel_features(0, np.array([120])))

    def test_features_and_prediction_match_explicit_block_means_anchor(self):
        x = self.rng.normal(size=(2, 64, 3)).astype(np.float32)
        features = self.mod.temporal_features(x)
        self.assertEqual(features.shape, (2, 3, 33))
        np.testing.assert_array_equal(features[:, :, -1], 1)
        np.testing.assert_allclose(features[0, :, 0], x[0, :2].astype(np.float64).mean(0)-x[0, -1], atol=1e-12)
        coefficients = np.zeros((3, 33, 4))
        coefficients[:, -1] = 2
        actual = self.mod.predict_normalized(dict(coefficients=coefficients), x)
        np.testing.assert_allclose(actual, x[:, -1:, :] + np.full((2, 4, 3), 2))

    def test_masked_target_values_do_not_affect_fit_and_future_is_never_scored(self):
        task = self.task()
        task2 = dict(task, physical=task["physical"].copy())
        task2["physical"][~task["observed"]] = 123456
        task2["physical"][144:] += 99999
        with tempfile.TemporaryDirectory(prefix="informs_ridge_test_") as directory:
            root = Path(directory)
            receipt, artifact = self.mod.fit_task(task, root/"a", root/"public_a")
            receipt2, artifact2 = self.mod.fit_task(task2, root/"b", root/"public_b")
            np.testing.assert_array_equal(artifact["coefficients"], artifact2["coefficients"])
            self.assertEqual(receipt["selected_alpha"], receipt2["selected_alpha"])
            self.assertEqual(receipt["train_origins_refit"], 77)
            self.assertEqual(receipt["A_stop_origins"], 6)
            self.assertEqual(receipt["fit_end"], 144)
            self.assertEqual(receipt["T_final_target_access"], False)
            self.assertEqual(receipt["M_cal_target_access"], False)
            self.assertEqual(receipt["V_select_target_access"], False)
            self.assertEqual(receipt["actual_parameter_count"], 3*33*4)
            restored = self.mod.load_artifact(root/"a"/receipt["artifact_cache_relative_path"])
            np.testing.assert_array_equal(restored["coefficients"], artifact["coefficients"])
            self.assertEqual(restored["coefficients"].dtype, np.float64)
            self.assertEqual(len(receipt["alpha_candidates"]), 5)


if __name__ == "__main__":
    unittest.main()
