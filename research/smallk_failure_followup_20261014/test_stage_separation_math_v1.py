"""SOURCE ONLY fixtures. Do not execute until a new resource scope is approved."""
from pathlib import Path
import inspect
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_separation_math_v1 as stage


class FeatureAndMaskTests(unittest.TestCase):
    def test_fixed_33K_features_and_FIT_only_statistics(self):
        past = np.arange(2 * 512 * 2, dtype=np.float32).reshape(2, 512, 2)
        raw = stage.fixed_past_features(past)
        self.assertEqual(raw.shape, (2, 66))
        np.testing.assert_array_equal(raw[:, -2:], past[:, -1])
        np.testing.assert_array_equal(raw[:, :2], past[:, :16].mean(axis=1))
        state = stage.fit_feature_state(past)
        before = state.sha256
        state.transform(past + 1000)
        self.assertEqual(state.sha256, before)
        with self.assertRaisesRegex(ValueError, 'SAME_FIT'):
            state.transform(past + 1, require_fit_artifact=True)

    def test_complete_prefix_missing_future_stays_excluded_and_broadcasts_K(self):
        mask = np.ones((2, 48, 3), dtype=bool)
        mask[0, 2, 1] = False
        mask[1, 0, 2] = False
        latent = stage.complete_prefix_latent_mask(mask, 4)
        self.assertEqual(latent.shape, (2, 48, 4))
        self.assertTrue(latent[0, :2].all())
        self.assertFalse(latent[0, 2:].any())
        self.assertFalse(latent[1].any())

    def test_N1_and_N2_share_exact_feature_state_but_separate_teacher_counts(self):
        past = np.zeros((4, 512, 1), np.float32)
        past[:, :, 0] = np.arange(4)[:, None]
        state = stage.fit_feature_state(past)
        oracle = np.repeat(past[:, -1:], 48, axis=1)
        observed = np.ones((4, 48, 2), bool)
        observed[0, 1, 0] = False
        n1 = stage.prepare_n1(state, past, oracle, observed)
        n2 = stage.prepare_n2(state, past, np.ones((4, 48, 2)), np.zeros((4, 2)), observed)
        self.assertIs(n1.feature_state, n2.feature_state)
        self.assertEqual(n1.ridge.feature_state_sha256, n2.ridge.feature_state_sha256)
        self.assertNotEqual(int(n1.ridge.counts.reshape(48, 1)[2, 0]), int(n2.ridge.counts.reshape(48, 2)[2, 0]))


class RidgeObjectiveTests(unittest.TestCase):
    def test_unequal_horizon_counts_use_channel_macro_denominator(self):
        design = np.array([[-1.], [1.], [-1.], [1.]])
        y = np.repeat(design[:, None], 2, axis=1)
        mask = np.ones_like(y, dtype=bool)
        mask[2:, 1] = False
        prepared = stage.prepare_masked_ridge(design, y, mask, kind='N2_DIRECT_C', feature_state_sha256='synthetic')
        result = prepared.fit(1.)
        np.testing.assert_allclose(result.coefficients[0, :, 0], [.4, .25], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(result.normal_equation_denominators[:, 0], [6., 6.])
        optimum = stage.masked_macro_mse(result.predict(design), y, mask) + float((result.coefficients ** 2).sum())
        wrong = np.ones_like(y) * design[:, None] * .5
        per_h_normalized_objective = stage.masked_macro_mse(wrong, y, mask) + .5
        self.assertLess(optimum, per_h_normalized_objective)
        np.testing.assert_allclose(2 * np.array([4., 2.]) * (np.array([.4, .25]) - 1.) / 6 + 2 * np.array([.4, .25]), 0., atol=1e-12)

    def test_positive_alpha_does_not_penalize_intercept(self):
        design = np.array([[-1.], [1.]])
        y = np.full((2, 1, 1), 7.)
        result = stage.prepare_masked_ridge(design, y, np.ones_like(y, bool), kind='N2_DIRECT_C', feature_state_sha256='x').fit(10.)
        np.testing.assert_allclose(result.coefficients, 0., atol=1e-12)
        np.testing.assert_allclose(result.intercept, 7., atol=1e-12)

    def test_alpha0_minimum_norm_rank_deficient_and_zero_rank(self):
        design = np.array([[-1., -1.], [1., 1.], [-2., -2.], [2., 2.]])
        y = (3 + design[:, :1])[:, None]
        prepared = stage.prepare_masked_ridge(design, y, np.ones_like(y, bool), kind='N2_DIRECT_C', feature_state_sha256='rank')
        result = prepared.fit(0.)
        np.testing.assert_allclose(result.coefficients[:, 0, 0], [.5, .5], atol=1e-12)
        self.assertEqual(int(result.design_ranks[0, 0]), 1)
        zero = stage.prepare_masked_ridge(np.zeros((4, 2)), y, np.ones_like(y, bool), kind='N2_DIRECT_C', feature_state_sha256='zero').fit(0.)
        np.testing.assert_array_equal(zero.coefficients, np.zeros_like(zero.coefficients))
        self.assertEqual(int(zero.design_ranks[0, 0]), 0)

    def test_masked_nan_zero_observed_horizon_and_zero_channel(self):
        design = np.array([[-1.], [1.], [-1.], [1.]])
        y = np.ones((4, 2, 2))
        mask = np.ones_like(y, bool)
        mask[:, 1, 0] = False
        mask[:, :, 1] = False
        y[~mask] = np.nan
        result = stage.prepare_masked_ridge(design, y, mask, kind='N2_DIRECT_C', feature_state_sha256='missing').fit(.1)
        self.assertTrue(np.isfinite(result.predict(design)).all())
        np.testing.assert_array_equal(result.coefficients[:, 1, 0], [0.])
        self.assertEqual(float(result.intercept[1, 0]), 0.)
        np.testing.assert_array_equal(result.observed_counts[:, 1], [0, 0])

    def test_identical_masks_factorization_is_shared_across_seven_alphas(self):
        design = np.arange(12, dtype=float).reshape(4, 3)
        y = np.ones((4, 2, 2))
        prepared = stage.prepare_masked_ridge(design, y, np.ones_like(y, bool), kind='N2_DIRECT_C', feature_state_sha256='groups')
        self.assertEqual(len(prepared.groups), 1)
        self.assertEqual(len(prepared.fit_all()), 7)
        self.assertEqual(tuple(model.alpha for model in prepared.fit_all()), stage.ALPHAS)

    def test_earliest_tie_actual_Y_TUNE_selection_only(self):
        rows = [{'alpha': alpha, 'tune_actual_Y_mse': 1., 'latent_tune_mse': 100 - i, 'eval_mse': -i}
                for i, alpha in enumerate(stage.ALPHAS)]
        self.assertEqual(stage.select_alpha(rows), 0)
        rows[3]['tune_actual_Y_mse'] = .9
        self.assertEqual(stage.select_alpha(rows), 3)


class DecoderAndAnchorTests(unittest.TestCase):
    def fixtures(self):
        future = np.ones((4, 48, 1), np.float32)
        last_z = np.zeros((4, 1, 1), np.float32)
        raw_last = np.arange(8, dtype=np.float32).reshape(4, 2)
        target = raw_last[:, None] + np.ones((4, 48, 2), np.float32)
        hidden_weight, hidden_bias = np.zeros((32, 1), np.float32), np.zeros(32, np.float32)
        outgoing = np.zeros((33, 2), np.float32)
        outgoing[0] = [2., 3.]
        return future, last_z, raw_last, target, hidden_weight, hidden_bias, outgoing

    def test_no_intercept_decoder_preserves_zero_innovation_anchor(self):
        future, last_z, raw_last, target, hidden_w, hidden_b, outgoing = self.fixtures()
        prepared = stage.prepare_o1(future, last_z, target, raw_last, np.ones_like(target, bool), hidden_w, hidden_b, outgoing)
        model = prepared.fit(.1)
        self.assertFalse(model.scaled_solution.with_intercept)
        prediction = model.forecast(np.repeat(last_z, 48, axis=1), last_z, raw_last)
        np.testing.assert_array_equal(prediction, stage.persistence_forecast(raw_last))
        self.assertTrue(model.diagnostic_only)

    def test_decoder_zero_observed_channel_preserves_original_outgoing(self):
        future, last_z, raw_last, target, hidden_w, hidden_b, outgoing = self.fixtures()
        mask = np.ones_like(target, bool)
        mask[:, :, 1] = False
        target[:, :, 1] = np.nan
        model = stage.prepare_o1(future, last_z, target, raw_last, mask, hidden_w, hidden_b, outgoing).fit(1.)
        np.testing.assert_array_equal(model.weight[:, 1], outgoing[:, 1])
        self.assertEqual(int(model.observed_counts[1]), 0)
        self.assertTrue(np.isfinite(model.weight).all())

    def test_decoder_RMS_and_positive_alpha_match_D2_mean_MSE_penalty(self):
        future, last_z, raw_last, target, hidden_w, hidden_b, outgoing = self.fixtures()
        future *= 2.
        model = stage.prepare_o1(future, last_z, target, raw_last, np.ones_like(target, bool), hidden_w, hidden_b, outgoing).fit(1.)
        self.assertEqual(float(model.feature_rms[0]), 2.)
        np.testing.assert_allclose(model.weight[0], [.25, .25], atol=1e-7)
        prediction = model.forecast(future, last_z, raw_last)
        np.testing.assert_allclose(prediction - raw_last[:, None], .5, atol=1e-7)

    def test_N3_keeps_selected_N1_coefficients_and_low_level_rejects_wrong_hash(self):
        past = np.zeros((4, 512, 1), np.float32)
        past[:, :, 0] = np.arange(4)[:, None]
        state = stage.fit_feature_state(past)
        oracle = np.repeat(past[:, -1:] + 1, 48, axis=1)
        ymask = np.ones((4, 48, 2), bool)
        native = stage.prepare_n1(state, past, oracle, ymask).fit(.1)
        before = native.sha256
        _, _, raw_last, target, hidden_w, hidden_b, outgoing = self.fixtures()
        n3 = stage.prepare_n3(native, past, target, raw_last, ymask, hidden_w, hidden_b, outgoing).fit(.1)
        self.assertEqual(native.sha256, before)
        self.assertEqual(n3.fixed_n1_model_sha256, before)
        self.assertFalse(n3.diagnostic_only)
        with self.assertRaisesRegex(ValueError, 'FIXED_SELECTED_N1'):
            n3.forecast(native.latent_forecast(past), past[:, -1:], raw_last, fixed_n1_model_sha256='changed')

    def test_N3_deployment_has_no_future_argument_and_recomputes_bound_N1(self):
        past = np.zeros((4, 512, 1), np.float32)
        past[:, :, 0] = np.arange(4)[:, None]
        state = stage.fit_feature_state(past)
        oracle = np.repeat(past[:, -1:] + 1, 48, axis=1)
        mask = np.ones((4, 48, 2), bool)
        native = stage.prepare_n1(state, past, oracle, mask).fit(.1)
        _, _, raw_last, target, hidden_w, hidden_b, outgoing = self.fixtures()
        n3 = stage.prepare_n3(native, past, target, raw_last, mask, hidden_w, hidden_b, outgoing).fit(.1)
        self.assertEqual(list(inspect.signature(n3.forecast_from_selected_n1).parameters), ['model', 'latent_past', 'raw_last'])
        predicted = n3.forecast_from_selected_n1(native, past, raw_last)
        arithmetic = n3.forecast(native.latent_forecast(past), past[:, -1:], raw_last, fixed_n1_model_sha256=native.sha256)
        np.testing.assert_array_equal(predicted, arithmetic)
        with self.assertRaisesRegex(ValueError, 'NATIVE_MODEL_OBJECT'):
            n3.forecast_from_selected_n1(oracle, past, raw_last)
        with self.assertRaises(TypeError):
            n3.forecast_from_selected_n1(native, past, raw_last, z_future=oracle)
        changed = stage.prepare_n1(state, past, oracle + 2, mask).fit(.1)
        with self.assertRaisesRegex(ValueError, 'FIXED_SELECTED_N1'):
            n3.forecast_from_selected_n1(changed, past, raw_last)

    def test_zero_features_N2_equals_FIT_intercept_baseline_and_not_new_information(self):
        past = np.ones((4, 512, 1), np.float32) * 3
        raw_last = np.arange(8, dtype=np.float32).reshape(4, 2)
        target = raw_last[:, None] + np.arange(48)[None, :, None] + np.arange(4)[:, None, None]
        mask = np.ones_like(target, bool)
        mask[0, :5, 1] = False
        target[~mask] = np.nan
        state = stage.fit_feature_state(past)
        baseline = stage.fit_intercept_only(target, raw_last, mask)
        for native in stage.prepare_n2(state, past, target, raw_last, mask).fit_all():
            np.testing.assert_array_equal(native.direct_forecast(past, raw_last), baseline.forecast(raw_last))
        self.assertEqual(baseline.observed_counts.shape, (48, 2))


if __name__ == '__main__':
    unittest.main()
