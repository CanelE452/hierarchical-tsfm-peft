"""Legal FIT-prefix windows and independent missing-target reconstruction."""
import unittest

import numpy as np

import audit_expanded_readout_v1 as audit


class ExpandedReadoutFixtures(unittest.TestCase):
    def test_robin_dense_origins_are_512_unique_legal_sorted_windows(self):
        origins = audit.deterministic_legal_origins([528, 2121])
        self.assertEqual(len(origins), 512)
        self.assertEqual(int(origins[0]), 1040)
        self.assertEqual(int(origins[-1]), 2073)
        self.assertTrue(np.all(np.diff(origins) > 0))
        self.assertTrue(np.all(origins - 512 >= 528))
        self.assertTrue(np.all(origins + 48 <= 2121))

    def test_short_fit_has_no_invented_windows(self):
        with self.assertRaisesRegex(ValueError, "NO_LEGAL"):
            audit.deterministic_legal_origins([100, 600])

    def test_masked_future_is_zero_normalized_without_future_fill(self):
        physical = np.arange(600, dtype=np.float32)[:, None]
        mask = np.ones_like(physical, np.bool_)
        mask[512, 0] = False
        physical[512, 0] = np.nan
        values = audit.reconstructed_windows(physical, mask, np.array([512], np.int64),
                                              np.zeros(1, np.float32), np.ones(1, np.float32))
        self.assertEqual(values["target"][0, 0, 0], 0)
        self.assertFalse(values["observed"][0, 0, 0])
        self.assertEqual(values["target"][0, 1, 0], 513)
        self.assertEqual(values["raw_last"][0, 0], 511)


if __name__ == "__main__":
    unittest.main()
