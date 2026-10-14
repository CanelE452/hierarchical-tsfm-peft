import unittest

import numpy as np
from run_expanded_readout_v1 import legal_fit_origins


class ExpandedOriginTests(unittest.TestCase):
    def test_robin_same_fit_period_512_legal_origins(self):
        origins = legal_fit_origins((528, 2121))
        self.assertEqual(len(origins), 512)
        self.assertEqual(int(origins[0]), 1040)
        self.assertEqual(int(origins[-1]), 2073)
        self.assertTrue(np.all(np.diff(origins) > 0))
        self.assertTrue(np.all(origins-512 >= 528))
        self.assertTrue(np.all(origins+48 <= 2121))

    def test_no_invented_windows_when_legal_pool_small(self):
        origins = legal_fit_origins((0, 570))
        np.testing.assert_array_equal(origins, np.arange(512, 523, dtype=np.int64))

    def test_no_future_targets_or_history_outside_fit(self):
        with self.assertRaisesRegex(ValueError, 'NO_LEGAL'):
            legal_fit_origins((528, 1087))


if __name__ == '__main__':
    unittest.main()
