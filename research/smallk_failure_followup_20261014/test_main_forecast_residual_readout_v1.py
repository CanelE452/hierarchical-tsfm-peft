"""SOURCE fixtures only; actual test/solver execution requires a fresh ROOT gate."""
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_forecast_residual_readout_v1 as residual
import matched_readout_math_v1 as matched
import stage_separation_math_v1 as stage


class MainResidualTests(unittest.TestCase):
    def fixture(self, route='E33K'):
        past = np.zeros((4, 512, 1), np.float32)
        past[:, :16, 0] = np.array([-1., 1., -1., 1.], np.float32)[:, None]
        main = np.full((4, 48, 2), .25, np.float32)
        y = main + np.float32(.125)
        mask = np.ones(y.shape, bool)
        prepared = residual.prepare_route(route, past, main, y, mask)
        return prepared, past, main, y, mask

    def test_zero_correction_keeps_exact_original_MAIN_bits_and_is_not_alpha0(self):
        prepared, past, main, _, _ = self.fixture()
        main[0, 0, 0] = np.float32(-0.)
        prediction = prepared.zero_model().forecast(past, main)
        np.testing.assert_array_equal(prediction.view(np.uint32), main.view(np.uint32))
        self.assertIsNone(prepared.zero_model().alpha)
        self.assertEqual(prepared.candidates()[0].candidate_id, 'ZERO_CORRECTION')
        self.assertEqual(prepared.candidates()[1].candidate_id, 'MEAN_CORRECTION')
        self.assertEqual(prepared.candidates()[2].candidate_id, 'RIDGE_ALPHA_0')
        self.assertEqual(prepared.candidates()[2].alpha, 0.)

    def test_future_shaped_forecast_input_is_prediction_not_future_truth_signature(self):
        signature = inspect.signature(residual.CorrectionModel.forecast)
        self.assertEqual(tuple(signature.parameters), ('self', 'past', 'main_forecast'))
        self.assertNotIn('target', signature.parameters)
        self.assertNotIn('observed', signature.parameters)
        source = inspect.getsource(residual)
        self.assertIn('Array shape cannot establish that provenance', source)
        self.assertNotIn('torch.load', source)
        self.assertNotIn('np.load(', source)

    def test_E33K_and_E512K_keep_previous_feature_dimensions_and_precision(self):
        past = np.arange(3*512*2, dtype=np.float32).reshape(3, 512, 2)
        np.testing.assert_array_equal(residual.past_features('E33K', past), stage.fixed_past_features(past))
        np.testing.assert_array_equal(residual.past_features('E512K', past), matched.past_features('E512K', past))
        self.assertEqual(residual.past_features('E33K', past).shape, (3, 66))
        self.assertEqual(residual.past_features('E512K', past).shape, (3, 1024))
        self.assertEqual(residual.past_features('E33K', past).dtype, np.float32)

    def test_residual_label_FP32_subtraction_precedes_FP64_solver(self):
        y = np.full((2, 48, 1), np.float32(1e10), np.float32)
        main = np.ones(y.shape, np.float32)
        target = residual.fit_residual_target(main, y, np.ones(y.shape, bool))
        self.assertEqual(target.dtype, np.float32)
        self.assertEqual(float(target[0, 0, 0]), float(np.float32(1e10)))
        self.assertNotEqual(np.float64(target[0, 0, 0]), np.float64(y[0, 0, 0])-1.)

    def test_channel_total_Nc_penalty_unequal_horizon_counts(self):
        past = np.zeros((4, 512, 1), np.float32)
        x = np.array([-1., 1., -1., 1.], np.float32)
        past[:, :16, 0] = x[:, None]
        main = np.zeros((4, 48, 1), np.float32)
        y = np.repeat(x[:, None, None], 48, axis=1)
        mask = np.zeros(y.shape, bool); mask[:, 0] = True; mask[:2, 1] = True
        prepared = residual.prepare_route('E33K', past, main, y, mask)
        solution = prepared.fit(1.).solution
        np.testing.assert_allclose(solution.coefficients[0, :2, 0], [.4, .25], atol=1e-12)
        np.testing.assert_array_equal(solution.normal_equation_denominators[:, 0], np.full(48, 6.))
        np.testing.assert_array_equal(solution.observed_counts[:2, 0], [4, 2])
        self.assertFalse(np.allclose(solution.coefficients[0, :2, 0], [.5, .5]))

    def test_zero_observed_outputs_keep_original_MAIN_and_allow_masked_NaN(self):
        prepared, past, main, y, mask = self.fixture()
        mask[:, 7, 0] = False; mask[:, :, 1] = False; y[~mask] = np.nan
        prepared = residual.prepare_route('E33K', past, main, y, mask)
        for alpha in residual.ALPHAS:
            model = prepared.fit(alpha)
            prediction = model.forecast(past, main)
            np.testing.assert_array_equal(prediction[:, 7, 0], main[:, 7, 0])
            np.testing.assert_array_equal(prediction[:, :, 1], main[:, :, 1])
            self.assertTrue(np.isfinite(model.solution.coefficients).all())

    def test_FIT_state_is_fixed_at_inference_and_exact_FIT_artifact_required(self):
        prepared, past, main, y, mask = self.fixture()
        before = prepared.feature_state.sha256
        prepared.fit(.1).forecast(past + np.float32(100.), main)
        self.assertEqual(prepared.feature_state.sha256, before)
        self.assertFalse(prepared.feature_state.center.flags.writeable)
        with self.assertRaises(ValueError):
            residual.prepare_with_state(prepared.feature_state, past+np.float32(1.), main, y, mask)

    def test_candidate_TUNE_ties_select_original_MAIN_and_lower_is_better(self):
        rows = [{'candidate_id': candidate, 'alpha': alpha, 'tune_actual_Y_mse': 1.}
                for candidate, alpha in zip(residual.CANDIDATE_IDS, (None,None)+residual.ALPHAS)]
        self.assertEqual(residual.select_candidate(rows), 0)
        rows[0]['tune_actual_Y_mse'] = 2.
        self.assertEqual(residual.select_candidate(rows), 1)
        rows[3]['tune_actual_Y_mse'] = .9
        self.assertEqual(residual.select_candidate(rows), 3)
        with self.assertRaises(ValueError):
            residual.select_candidate(rows[1:])

    def test_failed_candidates_retained_but_baseline_remains_selectable(self):
        rows = [{'candidate_id': candidate, 'alpha': alpha, 'status':'COMPLETE', 'tune_actual_Y_mse': 1.}
                for candidate, alpha in zip(residual.CANDIDATE_IDS, (None,None)+residual.ALPHAS)]
        rows[1].update(status='FAILED_PRESERVED', tune_actual_Y_mse=None)
        self.assertEqual(residual.select_candidate(rows), 0)
        rows[2]['tune_actual_Y_mse'] = float('nan')
        with self.assertRaises(ValueError):
            residual.select_candidate(rows)

    def test_rankzero_ridge_equals_explicit_intercept_control_with_no_past_gain(self):
        past = np.zeros((3, 512, 1), np.float32)
        main = np.full((3, 48, 1), .25, np.float32)
        y = main + np.array([.125, .25, .375], np.float32)[:, None, None]
        mask = np.ones(y.shape, bool)
        prepared = residual.prepare_route('E512K', past, main, y, mask)
        control = residual.fit_intercept_only(main, y, mask)
        for alpha in residual.ALPHAS:
            model = prepared.fit(alpha)
            np.testing.assert_array_equal(model.solution.design_ranks, 0)
            np.testing.assert_array_equal(model.forecast(past, main), control.forecast(main))
            self.assertFalse(model.solution.coefficients.flags.writeable)
            json.dumps(model.metadata(), allow_nan=False)
        self.assertEqual(residual.CANDIDATE_IDS[1], 'MEAN_CORRECTION')

    def test_saved_coefficients_and_baseline_restore_without_refit(self):
        prepared, past, main, _, _ = self.fixture()
        state = prepared.feature_state
        features = {'center':state.center, 'scale':state.scale}
        for model in (prepared.zero_model(), prepared.mean_model(), prepared.fit(.1)):
            solution = model.solution
            arrays = None if solution is None else {name:getattr(solution,name) for name in
                ('coefficients','intercept','observed_counts','channel_counts','normal_equation_denominators',
                 'design_ranks','regularized_gram_conditions')}
            if model.mean_control is not None:
                arrays = {'correction':model.mean_control.correction,'observed_counts':model.mean_control.observed_counts}
            restored = residual.restore_model(state.metadata(), features, model.metadata(), arrays)
            self.assertEqual(restored.sha256, model.sha256)
            np.testing.assert_array_equal(restored.forecast(past, main), model.forecast(past, main))
        with self.assertRaises(ValueError):
            residual.restore_model(state.metadata(), {'center':state.center+1.,'scale':state.scale}, prepared.zero_model().metadata())

    def test_whole_MAIN_forecast_shape_dtype_and_observed_truth_must_be_valid(self):
        prepared, past, main, y, mask = self.fixture()
        for wrong in (main[:, 0], main.astype(np.float64), main[:, :47]):
            with self.assertRaises(ValueError): prepared.zero_model().forecast(past, wrong)
        y[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): residual.fit_residual_target(main, y, mask)

    def test_RAW33C_control_and_shared_mean_used_by_all_three_routes(self):
        prepared, latent, main, y, mask = self.fixture()
        raw = np.concatenate((latent, latent*np.float32(.5)), axis=2)
        shared = residual.fit_intercept_only(main, y, mask)
        for route, past in (('RAW33C',raw),('E33K',latent),('E512K',latent)):
            family = residual.prepare_route(route,past,main,y,mask,mean_control=shared)
            self.assertIs(family.mean_control,shared)
            np.testing.assert_array_equal(family.mean_model().forecast(past,main),shared.forecast(main))
            self.assertEqual(len(family.candidates()),9)
        self.assertEqual(residual.past_features('RAW33C',raw).shape,(4,66))
        wrong_mean = residual.fit_intercept_only(main,y+np.float32(.1),mask)
        with self.assertRaises(ValueError):
            residual.prepare_route('E33K',latent,main,y,mask,mean_control=wrong_mean)

    def test_zero_and_mean_models_available_without_ridge_factorization(self):
        past = np.zeros((3,512,1),np.float32)
        state = residual.fit_feature_state('E33K',past)
        main = np.ones((3,48,1),np.float32)
        zero = residual.zero_correction_model(state,1)
        np.testing.assert_array_equal(zero.forecast(past,main),main)
        shared = residual.fit_intercept_only(main,main+np.float32(.25),np.ones(main.shape,bool))
        mean = residual.mean_correction_model(state,1,shared)
        np.testing.assert_array_equal(mean.forecast(past,main),shared.forecast(main))
        self.assertNotIn('prepare_thin_masked_ridge',inspect.getsource(residual.zero_correction_model))
        self.assertNotIn('prepare_thin_masked_ridge',inspect.getsource(residual.mean_correction_model))


if __name__ == '__main__':
    unittest.main()
