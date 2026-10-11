"""Small invariants; these tests never open forecast/target data or a backbone."""

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits


spec = importlib.util.spec_from_file_location("robin_readout_run_under_test", Path(__file__).with_name("run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class ReadoutInvariants(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.history = rng.normal(size=(7, 8, 3)).astype(np.float32)
        self.raw = rng.normal(size=(7, 5, 3)).astype(np.float32)
        self.mask = np.ones_like(self.raw, dtype=bool)

    def test_residual_zero_is_bitwise_identity(self):
        spec = run.SPECS["RES_FULL_D8"]
        x = run.features_for(self.history, self.raw, spec)
        coeffs, _, _ = run.fit_grid(x, self.raw, self.mask, self.raw, spec, [.01, 1000.])
        self.assertTrue(np.array_equal(coeffs, np.zeros_like(coeffs)))
        for coeff in coeffs:
            self.assertTrue(np.array_equal(run.predict(x, coeff, self.raw, True), self.raw))

    def test_ridge_limit_has_expected_prior(self):
        target = (2. * self.raw + .3).astype(np.float32)
        zero = run.SPECS["ZERO_FULL_D8"]
        residual = run.SPECS["RES_FULL_D8"]
        x = run.features_for(self.history, self.raw, zero)
        zero_coeff, _, _ = run.fit_grid(x, target, self.mask, self.raw, zero, [1e16])
        res_coeff, _, _ = run.fit_grid(x, target, self.mask, self.raw, residual, [1e16])
        np.testing.assert_allclose(run.predict(x, zero_coeff[0], self.raw, False), 0., atol=1e-12)
        self.assertTrue(np.array_equal(run.predict(x, res_coeff[0], self.raw, True), self.raw))

    def test_residual_matches_ridge_centered_at_scaled_identity(self):
        spec = run.SPECS["RES_FULL_D8"]
        x = run.features_for(self.history, self.raw, spec)
        target = (self.raw * .8 + self.history[:, -1:, :] * .2).astype(np.float32)
        lam = 3.
        delta, scale, _ = run.fit_grid(x, target, self.mask, self.raw, spec, [lam])
        scaled_x = x.reshape(-1, x.shape[2]) / scale
        identity = np.zeros((x.shape[2], 3), dtype=np.float64)
        identity[:3] = np.eye(3)
        scaled_identity = identity * scale[:, None]
        fitted = np.linalg.solve(scaled_x.T @ scaled_x + lam * np.eye(x.shape[2]),
                                 scaled_x.T @ target.reshape(-1, 3).astype(np.float64) + lam * scaled_identity)
        np.testing.assert_allclose(delta[0] + identity, fitted / scale[:, None], rtol=1e-11, atol=1e-11)

    def test_diagonal_cannot_mix_channels(self):
        spec = run.SPECS["RES_DIAG_D8"]
        x = run.features_for(self.history, self.raw, spec)
        target = np.roll(self.raw, 1, axis=2)
        coeffs, _, _ = run.fit_grid(x, target, self.mask, self.raw, spec, [1.])
        support = run.support_matrix(x.shape[2], 3, spec)
        self.assertTrue(np.array_equal(coeffs[0][~support], np.zeros((~support).sum())))
        changed = self.raw.copy()
        changed[:, :, 1:] += 100.
        changed_history = self.history.copy()
        changed_history[:, :, 1:] -= 20.
        changed_x = run.features_for(changed_history, changed, spec)
        expected = run.predict(x, coeffs[0], self.raw, True)
        actual = run.predict(changed_x, coeffs[0], changed, True)
        self.assertTrue(np.array_equal(expected[:, :, 0], actual[:, :, 0]))

    def test_dc_difference_preserves_constant_offsets(self):
        spec = run.SPECS["RES_DC_DIFF_D8"]
        history = np.arange(2 * 8 * 3).reshape(2, 8, 3).astype(np.float32)
        raw = np.arange(2 * 5 * 3).reshape(2, 5, 3).astype(np.float32)
        offset = np.asarray([64., -32., 16.], dtype=np.float32)
        x = run.features_for(history, raw, spec)
        shifted_x = run.features_for(history + offset, raw + offset, spec)
        self.assertTrue(np.array_equal(x, shifted_x))
        coeff = np.full((x.shape[2], 3), .125, dtype=np.float64)
        expected = run.predict(x, coeff, raw, True) + offset
        actual = run.predict(shifted_x, coeff, raw + offset, True)
        self.assertTrue(np.array_equal(expected, actual))

    def test_lag_boundary_uses_history_and_saved_forecasts_only(self):
        history = np.asarray([[[1.], [2.]]], dtype=np.float32)
        raw = np.asarray([[[10.], [20.], [30.]]], dtype=np.float32)
        x = run.future_features(history, raw, 2)
        np.testing.assert_array_equal(x, [[[10., 2., 1.], [20., 10., 2.], [30., 20., 10.]]])
        raw[:, 2, 0] = -500.
        changed = run.future_features(history, raw, 2)
        self.assertTrue(np.array_equal(x[:, :2], changed[:, :2]))

    def test_chronological_folds_purge_target_overlap(self):
        origins = np.arange(390, dtype=np.int64)
        folds = run.chronological_folds(origins, [0, 100, 200, 300, 400], horizon=10)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            fit = origins[fold["fit_indices"]]
            evaluate = origins[fold["evaluation_indices"]]
            self.assertLessEqual(int(fit.max()) + 10, int(evaluate.min()))
            self.assertGreaterEqual(int(evaluate.min()), fold["cut"])
            self.assertLessEqual(int(evaluate.max()) + 10, fold["evaluation_end"])
        self.assertEqual([fold["fit_origins"] for fold in folds], [91, 191, 291])

    def test_masks_define_scale_and_target_fit(self):
        spec = run.SPECS["RES_FULL_D0"]
        x = run.features_for(self.history, self.raw, spec)
        target = (.5 * self.raw).astype(np.float32)
        mask = self.mask.copy()
        mask[0] = False
        mask[1, :, 1] = False
        coeff, scale, _ = run.fit_grid(x, target, mask, self.raw, spec, [1.])
        changed_x = x.copy()
        changed_x[0] = 1e8
        changed_target = target.copy()
        changed_target[~mask] = -1e8
        changed_coeff, changed_scale, _ = run.fit_grid(changed_x, changed_target, mask, self.raw, spec, [1.])
        self.assertTrue(np.array_equal(scale, changed_scale))
        self.assertTrue(np.array_equal(coeff, changed_coeff))

    def test_macro_metric_keeps_observed_constant_channels(self):
        target = np.zeros((1, 2, 2), dtype=np.float32)
        pred = np.asarray([[[2., 0.], [2., 0.]]], dtype=np.float32)
        metrics = run.metrics(pred, target, np.ones_like(pred, dtype=bool))
        self.assertEqual(metrics["mse"], 2.)
        self.assertEqual(metrics["mae"], 1.)

    def test_protocol_drift_is_rejected(self):
        protocol = run.read_json(Path(__file__).with_name("protocol.json"))
        run.validate_protocol(protocol)
        for mutation in ("arms", "cuts", "budget"):
            changed = copy.deepcopy(protocol)
            if mutation == "arms":
                changed["arms"][2]["parameters"] = 17
            elif mutation == "cuts":
                changed["cv"]["timestamp_cuts"][1] += 1
            else:
                changed["limits"]["new_f_calls"] = 1
            with self.subTest(mutation=mutation):
                with self.assertRaises(RuntimeError):
                    run.validate_protocol(changed)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
