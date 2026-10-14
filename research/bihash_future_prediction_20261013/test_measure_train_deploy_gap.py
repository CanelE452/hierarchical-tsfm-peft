"""CPU mechanism tests; these do not stand in for actual FrozenBolt diagnosis."""
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn

import measure_train_deploy_gap as gap


class TrainDeployGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_macro_metric_weights_channels_equally_not_cells(self):
        y = np.zeros((1, 2, 2), np.float32)
        pred = np.array([[[2., 4.], [2., 999.]]], np.float32)
        mask = np.array([[[True, True], [True, False]]])
        score = gap.macro_scores(pred, y, mask)
        self.assertEqual(score['mse'], 10.)
        self.assertEqual(score['mae'], 3.)
        self.assertEqual(score['per_channel_observed_count'], [2, 1])

    def test_missing_channel_and_nonfinite_cannot_be_scored(self):
        values = np.zeros((2, 2, 2), np.float32)
        mask = np.ones_like(values, bool)
        mask[:, :, 1] = False
        with self.assertRaisesRegex(ValueError, 'WITHOUT_OBSERVATION'):
            gap.macro_scores(values, values, mask)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            gap.macro_scores(values, values, np.ones_like(values, bool))

    def test_analytic_flip_probability_matches_two_class_gumbel(self):
        values = torch.tensor([-1.5, -.3, 0., .3, 1.5]).expand(25000, -1)
        generator = torch.Generator().manual_seed(43)
        actual = gap.bm.binary_code(values, True, .25, 'gumbel', generator)
        empirical = (torch.sign(actual) != torch.where(values >= 0, 1., -1.)).float().mean(0).numpy()
        expected = gap.analytic_flip_probability(values[0].numpy())
        np.testing.assert_allclose(empirical, expected, atol=.013, rtol=0.)
        self.assertEqual(float(gap.analytic_flip_probability(np.array(0.))), .5)

    def test_temperature_does_not_remove_gumbel_argmax_flips(self):
        values = torch.tensor([[-.05, .05]]).expand(2000, -1)
        soft_hot = gap.bm.binary_code(values, True, 1., 'gumbel', torch.Generator().manual_seed(44))
        soft_cold = gap.bm.binary_code(values, True, .01, 'gumbel', torch.Generator().manual_seed(44))
        self.assertTrue(torch.equal(soft_hot >= 0, soft_cold >= 0))
        self.assertGreater(float(((soft_cold >= 0) != (values >= 0)).float().mean()), .3)

    def test_protected_artifact_rejected_before_hash(self):
        for name in ('x_TEST.npz', 'EVAL_future_target_payload.npz'):
            with mock.patch.object(gap.rt, 'sha', side_effect=AssertionError('must not hash')):
                with self.assertRaisesRegex(ValueError, 'REJECTED_BEFORE_HASH'):
                    gap.safe_fit_artifact({'path': name, 'sha256': 'not_used'})

    def test_mode_scan_detects_auxiliary_randomness_and_batchnorm(self):
        module = nn.Sequential(nn.Dropout(.2), nn.BatchNorm1d(2), nn.Dropout(0.))
        self.assertEqual(len(gap.mode_sensitive_modules(module)), 2)

    def _model_inputs(self, method):
        basis = np.array([[1.], [0.]], np.float32)
        model = gap.bm.FutureBiHash(np.zeros(2, np.float32), basis, 45, method,
            bank_delta=np.array([[[1., 2.]] * 48, [[2., 3.]] * 48], np.float32),
            bank_observed=np.ones((2, 48, 2), bool), bank_origins=np.array([20, 80], np.int64))
        model.eval().requires_grad_(False)
        model.freeze_memory()
        # Small synthetic CPU fixture only; actual experiment keeps L512/H48.
        history = np.random.default_rng(3).normal(size=(8, 8, 2)).astype(np.float32)
        inputs = {'history': history, 'target': np.zeros((8, 48, 2), np.float32),
            'observed': np.ones((8, 48, 2), bool), 'origins': np.arange(200, 208, dtype=np.int64),
            'indices': np.arange(248, 256, dtype=np.int64), 'mean': np.zeros(2, np.float32),
            'std': np.ones(2, np.float32), 'selected_update': 100}
        return model, inputs

    def test_actual_training_codes_counter_replay_and_buffer_restoration(self):
        for method in gap.METHODS:
            with self.subTest(method=method):
                model, inputs = self._model_inputs(method)
                before = gap.rt.state_hash(model)
                arrays, evidence = gap.measure_fixed_model(model, inputs,
                    lambda z: z[:, -1:].expand(-1, 48, -1), lambda: None)
                self.assertEqual(evidence['model_before_sha256'], before)
                self.assertEqual(gap.rt.state_hash(model), before)
                self.assertFalse(model.training)
                self.assertEqual(evidence['primary_update'], 100)
                self.assertEqual(evidence['temperature'], .875)
                self.assertEqual(arrays['train_predictions'].shape, (8, 8, 48, 2))
                self.assertTrue(arrays['eligible'].all())
                if method == 'B_TANH':
                    self.assertEqual(arrays['query_gumbel_noise'].size, 0)
                    self.assertTrue((arrays['noise_counters'] == 0).all())
                    self.assertTrue(np.array_equal(arrays['train_predictions'][0], arrays['train_predictions'][7]))
                else:
                    np.testing.assert_array_equal(arrays['noise_counters'],
                        np.stack((np.arange(0, 16, 2), np.arange(2, 18, 2)), -1))
                    self.assertEqual(arrays['noise_seeds'].shape, (8, 2))
                    self.assertTrue(np.any(gap.sign(arrays['train_query_codes']) != arrays['deploy_query_codes'][None]))
                summary = gap.summarize(arrays, model._mode())
                self.assertEqual(summary['eligible_row_count'], 64)
                self.assertEqual(summary['used_key_count'], 2)
                self.assertTrue(np.isfinite(summary['forecast_gap_rms']))

    def test_exception_still_restores_fixed_checkpoint(self):
        model, inputs = self._model_inputs('B_GUMBEL')
        before = gap.rt.state_hash(model)
        with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
            gap.measure_fixed_model(model, inputs, lambda z: z[:, -1:].expand(-1, 48, -1),
                lambda: (_ for _ in ()).throw(RuntimeError('synthetic stop')))
        self.assertEqual(gap.rt.state_hash(model), before)
        self.assertFalse(model.training)

    def test_no_eligible_memory_measured_without_selecting_easier_windows(self):
        model, inputs = self._model_inputs('B_GUMBEL')
        inputs['origins'][:] = 40
        arrays, _ = gap.measure_fixed_model(model, inputs,
            lambda z: z[:, -1:].expand(-1, 48, -1), lambda: None)
        summary = gap.summarize(arrays, model._mode())
        self.assertEqual(summary['eligible_row_count'], 0)
        self.assertIsNone(summary['query_code_sign_flip_fraction_eligible_rows'])
        self.assertEqual(summary['forecast_gap_max_abs'], 0.)


if __name__ == '__main__':
    unittest.main()
