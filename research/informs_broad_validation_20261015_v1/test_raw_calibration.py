import unittest
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import raw_calibration as raw
from raw_calibration import fit_bias
import test_combination as fixtures


class BiasTest(unittest.TestCase):
    def test_mask_and_least_squares_stationarity(self):
        p = np.array([[[1., 4.], [2., 8.]], [[3., 9.], [4., 12.]]])
        y = p + np.array([2., -3.])
        mask = np.ones_like(p, bool)
        mask[0, 0, 1] = False
        y[0, 0, 1] = 99999
        bias = fit_bias(p, y, mask)
        np.testing.assert_allclose(bias, [2., -3.])
        residual = np.where(mask, p + bias - y, 0)
        np.testing.assert_allclose(residual.sum((0, 1)), 0, atol=1e-14)

    def test_absent_channel_rejected(self):
        a = np.zeros((1, 2, 2)); mask = np.ones_like(a, bool); mask[:, :, 0] = False
        with self.assertRaises(ValueError):
            fit_bias(a, a, mask)

    def test_completed_fit_reuses_only_verified_parent_and_output_bytes(self):
        task, pm, pv, _, _ = fixtures.CombinationTests().toy()
        for parent in (pm, pv):
            parent['metadata']['run_id'] = 'parent'
        mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
        with tempfile.TemporaryDirectory() as temp:
            cache, public = Path(temp) / 'cache', Path(temp) / 'public'
            first = raw.fit_raw_calibration('bias', task, pm, pv, mean, std, cache, public)
            with patch.object(raw, 'fit_bias', side_effect=AssertionError('must not refit')):
                again = raw.fit_raw_calibration('bias', task, pm, pv, mean, std, cache, public)
            self.assertEqual(first, again)
            corrupt = copy.deepcopy(pm)
            corrupt['pred_normalized'][0, 0, 0] += 1
            with self.assertRaises(ValueError):
                raw.fit_raw_calibration('bias', task, corrupt, pv, mean, std, cache, public)
            for field in ('source_sha', 'config_sha', 'data_sha', 'scaler_sha', 'parent_model_sha'):
                altered = copy.deepcopy(first)
                altered['contract'][field] = 'wrong identity'
                with self.assertRaises(ValueError):
                    raw.validate_completed_receipt(altered, 'bias', task, pm, pv, mean, std, cache)
            for name in ('bias.npz', 'M_cal.npz', 'V_select.npz'):
                path = cache / 'raw_calibration/bias' / name
                original = path.read_bytes(); path.write_bytes(original + b'changed')
                with self.assertRaises(ValueError):
                    raw.validate_completed_receipt(first, 'bias', task, pm, pv, mean, std, cache)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
