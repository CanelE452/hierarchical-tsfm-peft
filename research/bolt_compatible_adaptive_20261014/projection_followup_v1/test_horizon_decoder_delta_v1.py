"""CPU synthetic checks of objective, masks, equal coefficient count and ramp endpoints."""
import unittest
import numpy as np
import horizon_decoder_delta_v1 as module


class DecoderDeltaTests(unittest.TestCase):
    def fixture(self):
        random = np.random.default_rng(711)
        features = random.normal(size=(8, 4, 3)).astype(np.float32)
        features[:, :, 2] = 0
        target = random.normal(size=(8, 4, 2)).astype(np.float64)
        mask = np.ones_like(target, bool)
        mask[::2, 1:, 1] = False
        scale, constant = module.feature_scale(features)
        return features, target, mask, scale, constant

    def test_shared_scale_is_uncentered_rms_and_zero_feature_is_registered(self):
        x, _, _, scale, constant = self.fixture()
        expected = np.sqrt(np.mean(x.astype(np.float64)**2, axis=(0, 1)))
        expected[2] = 1
        np.testing.assert_array_equal(scale, expected)
        np.testing.assert_array_equal(constant, [False, False, True])
        self.assertGreater(float(np.max(np.abs(scale[:2]-x[:, :, :2].std(axis=(0, 1))))), 1e-5)

    def test_ramp_index_and_same_free_coefficient_count(self):
        x, y, mask, scale, constant = self.fixture()
        ramp = module.gated_features(x, "RAMP_DELTA")
        np.testing.assert_array_equal(ramp[:, 0], 0)
        np.testing.assert_array_equal(ramp[:, -1], x[:, -1])
        shapes = []
        for arm in module.ARMS:
            states, meta = module.fit_path(module.gated_features(x, arm), y, mask, scale, constant)
            shapes.append(states[0]["coefficient_FP32"].shape)
            np.testing.assert_array_equal(meta["scale_FP64"], scale)
            self.assertFalse(meta["intercept_fitted"])
            self.assertFalse(meta["feature_centering"])
        self.assertEqual(shapes, [(3, 2), (3, 2)])

    def test_positive_ridge_matches_per_channel_observed_normal_equation(self):
        x, y, mask, scale, constant = self.fixture()
        for arm in module.ARMS:
            feature = module.gated_features(x, arm)
            states, _ = module.fit_path(feature, y, mask, scale, constant)
            design = feature.astype(np.float64).reshape(-1, 3)/scale
            for state in states[1:]:
                for channel in range(2):
                    valid = mask[:, :, channel].reshape(-1)
                    a, b = design[valid], y[:, :, channel].reshape(-1)[valid]
                    standardized = np.linalg.solve(a.T@a + valid.sum()*state["alpha"]*np.eye(3), a.T@b)
                    expected = standardized/scale
                    np.testing.assert_allclose(state["coefficient_FP64"][:, channel], expected, atol=1e-11, rtol=1e-9)

    def test_alpha_zero_uses_design_minimum_norm_without_centering(self):
        x, y, mask, scale, constant = self.fixture()
        states, _ = module.fit_path(x, y, mask, scale, constant)
        a = x.astype(np.float64).reshape(-1, 3)/scale
        for channel in range(2):
            valid = mask[:, :, channel].reshape(-1)
            expected = np.linalg.lstsq(a[valid], y[:, :, channel].reshape(-1)[valid], rcond=1e-8)[0]/scale
            np.testing.assert_allclose(states[0]["coefficient_FP64"][:, channel], expected, atol=1e-12)

    def test_missing_placeholders_do_not_affect_fit_and_unobserved_C_is_zero(self):
        x, y, mask, scale, constant = self.fixture()
        y[~mask] = np.nan
        a, _ = module.fit_path(x, y, mask, scale, constant)
        y[~mask] = 1e200
        b, _ = module.fit_path(x, y, mask, scale, constant)
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first["coefficient_FP64"], second["coefficient_FP64"])
        mask[:, :, 1] = False
        states, meta = module.fit_path(x, y, mask, scale, constant)
        self.assertEqual(meta["groups"][1]["fallback"], "NO_OBSERVED_RESPONSE_ZERO_CORRECTION")
        for state in states:
            np.testing.assert_array_equal(state["coefficient_FP32"][:, 1], 0)

    def test_actual_FP32_prediction_has_registered_operation_order_and_first_horizon(self):
        x, y, mask, scale, constant = self.fixture()
        states, _ = module.fit_path(module.gated_features(x, "RAMP_DELTA"), y, mask, scale, constant)
        baseline = y.astype(np.float32)
        state = states[4]
        result = module.predict(baseline, x, state, "RAMP_DELTA")
        expected = baseline + (module.gated_features(x, "RAMP_DELTA").reshape(-1, 3)@state["coefficient_FP32"]).reshape(8, 4, 2)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result[:, 0], baseline[:, 0])
        np.testing.assert_array_equal(state["coefficient_FP32"], state["coefficient_FP64"].astype(np.float32))

    def test_not_per_horizon_fit_and_reject_wrong_contract(self):
        x, y, mask, scale, constant = self.fixture()
        for bad in ("FULL_HORIZON", "ramp"):
            with self.assertRaises(ValueError):
                module.gated_features(x, bad)
        with self.assertRaises(ValueError):
            module.fit_path(x.astype(np.float64), y, mask, scale, constant)
        with self.assertRaises(ValueError):
            module.fit_path(x, y, mask, scale.astype(np.float32), constant)


if __name__ == "__main__":
    unittest.main()
