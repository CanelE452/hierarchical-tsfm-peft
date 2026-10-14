"""Independent physical parity and window-local preprocessing checks."""
import unittest
import numpy as np
import verify_deployment_outputs as vd


class DeploymentOutputTests(unittest.TestCase):
    def test_causal_fill_resets_each_window_and_ignores_unobserved_values(self):
        raw = np.array([[[999., 999.], [3., 4.], [999., 6.]], [[999., 999.], [7., 8.], [9., 999.]]], np.float32)
        mask = np.array([[[0, 0], [1, 1], [0, 1]], [[0, 0], [1, 1], [1, 0]]], bool)
        mean = np.array([1., 2.], np.float32); std = np.array([2., 2.], np.float32)
        actual = vd.independent_preprocess(raw, mask, mean, std)
        expected = np.array([[[0., 0.], [1., 1.], [1., 2.]], [[0., 0.], [3., 3.], [4., 3.]]], np.float32)
        self.assertTrue(np.array_equal(actual, expected))
        changed = raw.copy(); changed[~mask] = np.nan
        self.assertTrue(np.array_equal(vd.independent_preprocess(changed, mask, mean, std), expected))

    def test_saved_actual_physical_parity_reports_normalized_gap(self):
        science = np.ones((2, 48, 3), np.float32)
        mean = np.array([10., 100., 1000.], np.float32); std = np.array([2., 3., 4.], np.float32)
        reference = science * std + mean
        actual = reference.copy(); actual[0, 0, 0] += .001
        check = vd.independent_parity(actual, science, mean, std)
        self.assertTrue(check['physical_parity_pass'])
        self.assertGreater(check['normalized_max_abs'], 0.)
        actual[0, 0, 0] += .1
        check = vd.independent_parity(actual, science, mean, std)
        self.assertFalse(check['physical_parity_pass'])
        self.assertEqual(check['failed_physical_cells'], 1)

    def test_nonfinite_or_shape_mismatch_cannot_pass_audit(self):
        y = np.zeros((1, 48, 2), np.float32)
        m = np.zeros(2, np.float32); s = np.ones(2, np.float32)
        altered = y.copy(); altered[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            vd.independent_parity(altered, y, m, s)
        with self.assertRaisesRegex(ValueError, 'SHAPE'):
            vd.independent_parity(y[:, :1], y, m, s)


if __name__ == '__main__': unittest.main()
