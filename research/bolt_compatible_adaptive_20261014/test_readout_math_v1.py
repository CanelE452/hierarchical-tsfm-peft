"""Meaningful solver, provenance-free math and past-only readout fixtures."""
import inspect
import unittest

import numpy as np
import readout_math_v1 as m


class ReadoutMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(317)
        self.past = rng.normal(size=(18, 512, 3)).astype(np.float32)
        self.last = self.past[:, -1].copy()
        self.target = (self.last[:, None] + rng.normal(size=(18, 48, 3))).astype(np.float32)
        self.mask = np.ones(self.target.shape, bool)
        self.mask[::3, 2:12, 1] = False
        self.mask[:, 37, 2] = False

    def test_same_temporal_summary_joint_and_self(self):
        self_features = m.self_features('SELF33', self.past)
        np.testing.assert_array_equal(self_features.reshape(18, -1), m.joint.past_features('RAW33C', self.past))

    def test_fit_statistics_frozen_and_other_channels_cannot_change_self_channel(self):
        prepared = m.prepare_self('SELF33', self.past, self.target, self.last, self.mask)
        model = prepared.fit(.01)
        before = model.feature_state.sha256
        changed = self.past.copy()
        changed[:, :, 1:] += np.float32(400.)
        original = model.forecast(self.past, self.last)
        perturbed = model.forecast(changed, changed[:, -1])
        np.testing.assert_array_equal(original[:, :, 0], perturbed[:, :, 0])
        self.assertEqual(before, model.feature_state.sha256)
        self.assertEqual(tuple(inspect.signature(model.forecast).parameters), ('past', 'raw_last'))

    def test_masked_normal_equation_uses_whole_channel_count(self):
        prepared = m.prepare_self('SELF33', self.past, self.target, self.last, self.mask)
        model = prepared.fit(.1)
        x = prepared.feature_state.transform(self.past)
        y = (self.target-self.last[:, None]).astype(np.float64)
        for c in range(3):
            nc = self.mask[:, :, c].sum()
            np.testing.assert_array_equal(model.arrays['normal_equation_denominators'][:, c], np.full(48, nc))
            for h in (0, 5, 37):
                rows = self.mask[:, h, c]
                if not rows.any():
                    np.testing.assert_array_equal(model.arrays['coefficients'][:, h, c], 0.)
                    self.assertEqual(model.arrays['intercept'][h, c], 0.)
                    continue
                w = model.arrays['coefficients'][:, h, c]
                b = model.arrays['intercept'][h, c]
                error = x[rows, :, c] @ w+b-y[rows, h, c]
                np.testing.assert_allclose(x[rows, :, c].T @ error+.1*nc*w, 0., atol=2e-10)
                self.assertAlmostEqual(float(error.sum()), 0., places=10)

    def test_self_single_channel_parity_with_old_joint_seven_alpha(self):
        prepared = m.prepare_self('SELF33', self.past[:, :, :1], self.target[:, :, :1],
                                  self.last[:, :1], self.mask[:, :, :1])
        old = m.joint.prepare_route('RAW33C', self.past[:, :, :1], self.target[:, :, :1],
                                   self.last[:, :1], self.mask[:, :, :1])
        for alpha in m.ALPHAS:
            np.testing.assert_allclose(prepared.fit(alpha).forecast(self.past[:, :, :1], self.last[:, :1]),
                old.fit(alpha).forecast(self.past[:, :, :1], self.last[:, :1]), atol=1e-6, rtol=1e-6)

    def test_under_determined_self512_and_restore(self):
        prepared = m.prepare_self('SELF512', self.past, self.target, self.last, self.mask)
        model = prepared.fit(0.)
        restored = m.restore_self_model(prepared.feature_state.metadata(),
            {'center': prepared.feature_state.center, 'scale': prepared.feature_state.scale}, model.metadata(), model.arrays)
        np.testing.assert_array_equal(model.forecast(self.past, self.last), restored.forecast(self.past, self.last))
        self.assertTrue(np.isfinite(restored.forecast(self.past, self.last)).all())
        self.assertEqual(prepared.feature_state.center.shape, (512, 3))

    def test_mean_subtracts_fp32_before_averaging_and_missing_output_zero(self):
        y = np.full((3, 48, 1), np.float32(1.e8+8))
        last = np.array([[1.e8], [-1.e8], [1.e-5]], np.float32)
        mask = np.ones_like(y, bool)
        mask[:, 7] = False
        got = m.fit_shared_mean(y, last, mask)
        expected = (y-last[:, None]).astype(np.float64).mean(axis=0).astype(np.float32)
        expected[7] = 0.
        np.testing.assert_array_equal(got['innovation'], expected)
        np.testing.assert_array_equal(got['observed_counts'][7], 0)

    def test_one_global_macro_alpha_and_earliest_exact_tie(self):
        rows = [{'kind': 'PERSIST', 'tune_scores': {'normalized': {'mse': 3.}}},
                {'kind': 'MEAN', 'tune_scores': {'normalized': {'mse': 2.}}}]
        rows += [{'kind': 'RIDGE', 'alpha': a, 'tune_scores': {'normalized': {'mse': 2. if a == 0 else 4.}}}
                 for a in m.ALPHAS]
        self.assertEqual(m.select_pool(rows), 1)
        rows[2]['tune_scores']['normalized']['mse'] = 1.
        self.assertEqual(m.select_pool(rows), 2)
        target = np.zeros((2, 48, 2), np.float32)
        observed = np.ones_like(target, bool)
        observed[:, :24, 1] = False
        prediction = np.zeros_like(target)
        prediction[:, :, 0] = 1.
        prediction[:, :, 1] = 2.
        self.assertEqual(m.observed_score(prediction, target, observed)['mse'], 2.5)

    def test_common_gate_blocks_mae_and_zero_denominator(self):
        target = np.zeros((8, 48, 2), np.float32)
        observed = np.ones_like(target, bool)
        base = np.ones_like(target)
        better = base*.9
        result = m.common_information_gate(base, better, target, observed)
        self.assertTrue(result['passed'])
        self.assertEqual(sum(x['joint_improved'] for x in result['four_chronological_blocks']), 4)
        zero = m.common_information_gate(target, target, target, observed)
        self.assertFalse(zero['passed'])
        self.assertIsNone(zero['mse_gain_pct'])


if __name__ == '__main__':
    unittest.main()
