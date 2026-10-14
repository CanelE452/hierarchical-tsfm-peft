"""SOURCE-only prepared mathematical fixtures; no actual assets or execution."""
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_readout_math_v1 as matched
import stage_separation_math_v1 as stage


class MatchedMathTests(unittest.TestCase):
    def test_RAW33C_is_exact_old_summary_order_and_E512K_is_full_flatten(self):
        raw = np.arange(2*512*3, dtype=np.float32).reshape(2, 512, 3)
        np.testing.assert_array_equal(matched.past_features('RAW33C', raw), stage.fixed_past_features(raw))
        np.testing.assert_array_equal(matched.past_features('E512K', raw), raw.reshape(2, 1536))
        self.assertEqual(matched.past_features('RAW33C', raw).dtype, np.float32)
        with self.assertRaises(ValueError):
            matched.past_features('NEW_ROUTE', raw)

    def test_feature_stats_FIT_only_and_feature_artifact_is_locked(self):
        raw = np.arange(4*512, dtype=np.float32).reshape(4, 512, 1)
        state = matched.fit_feature_state('E512K', raw)
        np.testing.assert_array_equal(state.center, raw.reshape(4, 512).astype(np.float64).mean(axis=0))
        before = state.sha256
        state.transform(np.ones_like(raw)*1e4)
        self.assertEqual(state.sha256, before)
        self.assertFalse(state.center.flags.writeable)
        with self.assertRaises(ValueError):
            state.transform(raw + np.float32(1), require_fit_artifact=True)

    def test_thin_SVD_matches_old_primal_objective_and_dual_solution(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(8, 19))
        y = rng.normal(size=(8, 2, 3))
        mask = np.ones(y.shape, bool)
        prepared = matched.prepare_thin_masked_ridge(x, y, mask, kind='COMPARE', feature_state_sha256='fixture')
        reference = stage.prepare_masked_ridge(x, y, mask, kind='COMPARE', feature_state_sha256='fixture')
        xc, yc = x-x.mean(axis=0), y.reshape(8, -1)-y.reshape(8, -1).mean(axis=0)
        for alpha in matched.ALPHAS:
            actual = prepared.fit(alpha)
            old = reference.fit(alpha)
            np.testing.assert_allclose(actual.coefficients, old.coefficients, atol=1e-11, rtol=1e-10)
            np.testing.assert_allclose(actual.intercept, old.intercept, atol=1e-11, rtol=1e-10)
            if alpha > 0:
                penalty = alpha*16
                dual = xc.T @ np.linalg.solve(xc@xc.T + penalty*np.eye(8), yc)
                primal = np.linalg.solve(xc.T@xc + penalty*np.eye(19), xc.T@yc)
                np.testing.assert_allclose(actual.coefficients.reshape(19, -1), dual, atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(dual, primal, atol=1e-9, rtol=1e-9)

    def test_rank_deficient_alpha0_centered_minimum_norm_and_bias(self):
        x = np.array([[-1., -1.], [1., 1.], [-1., -1.], [1., 1.]])
        y = x[:, :1] + 7.
        prepared = matched.prepare_thin_masked_ridge(x, y, np.ones_like(y, bool), kind='RANK', feature_state_sha256='f')
        model = prepared.fit(0.)
        np.testing.assert_allclose(model.coefficients[:, 0], [.5, .5], atol=1e-12)
        np.testing.assert_allclose(model.intercept, [7.], atol=1e-12)
        np.testing.assert_array_equal(model.design_ranks, [1])
        self.assertTrue(np.isinf(model.regularized_gram_conditions[0]))

    def test_high_dimension_alpha0_singular_cutoff_matches_old_lstsq(self):
        # Disjoint signed row pairs give exactly centered, orthogonal columns.
        # Spectrum is safely either side of cutoff, without a cancellation tie.
        n, p = 6, 11
        cutoff = np.finfo(np.float64).eps * max(n, p)
        x = np.zeros((n, p), np.float64)
        for column, singular in enumerate((1., 128.*cutoff, cutoff/128.)):
            x[2*column, column] = singular / np.sqrt(2.)
            x[2*column+1, column] = -singular / np.sqrt(2.)
        y = (x[:, 0] + 2.*x[:, 1] + 5.*x[:, 2])[:, None]
        mask = np.ones(y.shape, bool)
        new = matched.prepare_thin_masked_ridge(x, y, mask, kind='CUTOFF', feature_state_sha256='f').fit(0.)
        old = stage.prepare_masked_ridge(x, y, mask, kind='CUTOFF', feature_state_sha256='f').fit(0.)
        np.testing.assert_array_equal(new.design_ranks, [2])
        np.testing.assert_array_equal(old.design_ranks, new.design_ranks)
        np.testing.assert_allclose(new.coefficients, old.coefficients, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(new.predict(x), old.predict(x), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(new.coefficients[:3, 0], [1., 2., 0.], atol=1e-12, rtol=1e-12)

    def test_unequal_horizon_observations_use_channel_total_not_horizon_count(self):
        x = np.array([[-1.], [1.], [-1.], [1.]])
        y = np.repeat(x[:, None], 2, axis=1)
        mask = np.ones(y.shape, bool); mask[2:, 1] = False
        p = matched.prepare_thin_masked_ridge(x, y, mask, kind='Nc', feature_state_sha256='f')
        model = p.fit(1.)
        np.testing.assert_allclose(model.coefficients[0, :, 0], [.4, .25], atol=1e-12)
        np.testing.assert_array_equal(model.normal_equation_denominators, [[6], [6]])
        self.assertFalse(np.allclose(model.coefficients[0, :, 0], [.5, .5]))

    def test_masked_NaN_zero_output_and_rankzero(self):
        x = np.zeros((4, 2))
        y = np.ones((4, 2, 2)); mask = np.ones(y.shape, bool)
        mask[:, 1, 0] = False; mask[:, :, 1] = False; y[~mask] = np.nan
        p = matched.prepare_thin_masked_ridge(x, y, mask, kind='ZERO', feature_state_sha256='f')
        for alpha in matched.ALPHAS:
            s = p.fit(alpha)
            self.assertTrue(np.isfinite(s.coefficients).all())
            np.testing.assert_array_equal(s.coefficients, 0.)
            np.testing.assert_array_equal(s.intercept, [[1., 0.], [0., 0.]])
            np.testing.assert_array_equal(s.design_ranks, 0)

    def test_shared_mask_factorization_shapes_are_thin_and_reused(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, 23)); y = rng.normal(size=(7, 3, 2)); mask = np.ones(y.shape, bool)
        checks = []
        p = matched.prepare_thin_masked_ridge(x, y, mask, kind='THIN', feature_state_sha256='f', check=lambda: checks.append(True))
        self.assertEqual(len(p.groups), 1)
        self.assertEqual(p.groups[0].right_vectors.shape, (7, 23))
        self.assertEqual(p.groups[0].singular_values.shape, (7,))
        group_identity = id(p.groups[0])
        self.assertEqual(len(p.fit_all()), 7)
        self.assertEqual(id(p.groups[0]), group_identity)
        self.assertGreaterEqual(len(checks), 3)

    def test_generic_new_thin_solver_matches_existing_E33K_readout(self):
        rng = np.random.default_rng(5)
        past = rng.normal(size=(9, 512, 1)).astype(np.float32)
        state = stage.fit_feature_state(past)
        y = rng.normal(size=(9, 48, 2)).astype(np.float32)
        last = rng.normal(size=(9, 2)).astype(np.float32)
        mask = np.ones(y.shape, bool); mask[2, 4:9, 0] = False
        old = stage.prepare_n2(state, past, y, last, mask)
        new = matched.prepare_thin_masked_ridge(state.transform(past), y-last[:, None], mask,
            kind='N2_DIRECT_C', feature_state_sha256=state.sha256)
        for alpha in matched.ALPHAS:
            np.testing.assert_allclose(old.fit(alpha).solution.predict(state.transform(past)), new.fit(alpha).predict(state.transform(past)),
                                       atol=1e-8, rtol=1e-8)

    def test_FP32_innovation_subtraction_is_before_FP64_solver(self):
        raw = np.zeros((2, 512, 1), np.float32)
        y = np.full((2, 48, 1), np.float32(1e10), np.float32)
        last = np.ones((2, 1), np.float32)
        p = matched.prepare_route('RAW33C', raw, y, last, np.ones(y.shape, bool))
        # FP32 subtraction rounds 1e10-1 back to 1e10; FP64 subtraction differs by1.
        expected = np.float64(np.float32(np.float32(1e10)-np.float32(1)))
        self.assertEqual(p.fit(0.).solution.intercept[0, 0], expected)
        self.assertNotEqual(expected, np.float64(np.float32(1e10))-1.)

    def test_model_public_forecast_has_only_past_and_current_anchor(self):
        self.assertEqual(tuple(inspect.signature(matched.ReadoutModel.forecast).parameters), ('self', 'past', 'raw_last'))
        raw = np.zeros((3, 512, 1), np.float32)
        y = np.ones((3, 48, 1), np.float32); last = np.zeros((3, 1), np.float32); mask = np.ones(y.shape, bool)
        p = matched.prepare_route('E512K', raw, y, last, mask)
        m = p.fit(.1)
        np.testing.assert_array_equal(m.forecast(raw, last), y)
        self.assertFalse(m.solution.coefficients.flags.writeable)
        json.dumps(m.metadata(), allow_nan=False)
        json.dumps(p.metadata(), allow_nan=False)

    def test_model_state_restore_uses_coefficients_without_refit(self):
        raw = np.zeros((3, 512, 1), np.float32)
        y = np.ones((3, 48, 1), np.float32); last = np.zeros((3, 1), np.float32); mask = np.ones(y.shape, bool)
        p = matched.prepare_route('RAW33C', raw, y, last, mask); model = p.fit(1.)
        arrays = {name: getattr(model.solution, name) for name in ('coefficients', 'intercept', 'observed_counts', 'channel_counts',
                    'normal_equation_denominators', 'design_ranks', 'regularized_gram_conditions')}
        restored = matched.restore_model(p.feature_state.metadata(), {'center': p.feature_state.center, 'scale': p.feature_state.scale}, model.metadata(), arrays)
        self.assertEqual(restored.sha256, model.sha256)
        np.testing.assert_array_equal(restored.forecast(raw, last), model.forecast(raw, last))
        with self.assertRaises(ValueError):
            matched.restore_model(p.feature_state.metadata(), {'center': p.feature_state.center+1, 'scale': p.feature_state.scale}, model.metadata(), arrays)

    def test_exact_seven_alpha_actual_Y_TUNE_selection_and_earliest_tie(self):
        rows = [{'alpha': a, 'tune_actual_Y_mse': 1. if a in (0., .001) else 2.} for a in matched.ALPHAS]
        self.assertEqual(matched.select_alpha(rows), 0)
        with self.assertRaises(ValueError):
            matched.select_alpha(rows[:6])


if __name__ == '__main__':
    unittest.main()
