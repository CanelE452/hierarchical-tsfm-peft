"""Synthetic tests of chronology, A-only fitting, masks and protected parsing."""
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_contract as d


class DataContractTests(unittest.TestCase):
    def sample(self):
        x = np.arange(2000, dtype=np.float32).reshape(1000, 2)
        return {"physical": x, "observed": np.ones_like(x, bool), "L": 32, "H": 12}

    def test_splits_and_target_purge(self):
        ranges = d.split_ranges(1000)
        self.assertEqual(ranges["A_fit"], (0, 500))
        self.assertEqual(ranges["T_final"], (800, 1000))
        for start, stop in ranges.values():
            points = d.origins((start, stop), 32, 12, 12)
            self.assertTrue(np.all(points >= max(start, 32)))
            self.assertTrue(np.all(points + 12 <= stop))
            self.assertTrue(np.all(np.diff(points) == 12))
        self.assertEqual(d.origins((0, 62), 32, 12, 12).tolist(), [32, 44])
        secondary = d.split_ranges(1000, fold="secondary")
        self.assertEqual(secondary["T_final"][1], 800)

    def test_only_A_statistics_and_PCA(self):
        data = self.sample()
        mean, std, count = d.fit_statistics(data, 500)
        pca = d.initialize_pca(data, 500, 1)
        data["physical"][500:] = 1e9
        mean2, std2, count2 = d.fit_statistics(data, 500)
        pca2 = d.initialize_pca(data, 500, 1)
        for before, after in ((mean, mean2), (std, std2), (count, count2), (pca["E"], pca2["E"]), (pca["pca_center"], pca2["pca_center"])):
            np.testing.assert_array_equal(before, after)

    def test_window_reset_and_zero_is_observed(self):
        x = np.array([[99, 99], [np.nan, 0], [np.nan, 4], [3, np.nan], [5, 6]], np.float32)
        data = {"physical": x, "observed": np.isfinite(x), "L": 3, "H": 1}
        history, mask = d.history_window(data, 4, np.array([1, 2]), np.ones(2), length=3)
        np.testing.assert_array_equal(history, [[0, -2], [0, 2], [2, 2]])
        self.assertTrue(mask[0, 1])
        future, observed = d.target_window(data, 4, np.zeros(2), np.ones(2), horizon=1)
        np.testing.assert_array_equal(future, [[5, 6]])
        original = history.copy()
        data["physical"][4] = 1e9
        np.testing.assert_array_equal(d.history_window(data, 4, np.array([1, 2]), np.ones(2), length=3)[0], original)

    def test_constant_scale_and_missing_channel(self):
        data = {"physical": np.array([[0, 1e-10], [0, 2e-10]], np.float32), "observed": np.ones((2, 2), bool)}
        mean, std, count = d.fit_statistics(data, 2)
        self.assertEqual(std[0], 1)
        self.assertGreater(std[1], 0)
        self.assertLess(std[1], 1e-9)
        data["observed"][:, 0] = False
        with self.assertRaises(ValueError):
            d.fit_statistics(data, 2)

    def test_electricity_tail_never_float_parsed(self):
        text = "@relation Electricity\n@attribute series_name string\n@attribute start_timestamp date\n@frequency hourly\n@missing false\n@equallength true\n@data\nT1:2012-01-01 00-00-01:0.0,1.0,PROTECTED_POISON,PROTECTED_POISON\n"
        values, times, metadata = d.parse_electricity_prefix(io.StringIO(text), cutoff="2012-01-01T02:00:00", expected_channels=1)
        np.testing.assert_array_equal(values[:, 0], [0, 1])
        self.assertEqual(metadata["numeric_values_parsed"], 2)
        self.assertEqual(metadata["tail_numeric_values_not_parsed"], 2)
        self.assertEqual(times[0], np.datetime64("2012-01-01T00:00:01", "ns").astype(np.int64))

    def test_invalid_bounds_rejected(self):
        data = self.sample()
        with self.assertRaises(ValueError):
            d.history_window(data, 20, np.zeros(2), np.ones(2), length=32)
        with self.assertRaises(ValueError):
            d.target_window(data, 995, np.zeros(2), np.ones(2), horizon=12)
        with self.assertRaises(ValueError):
            d.fit_statistics(data, 1001)


if __name__ == "__main__":
    unittest.main()
