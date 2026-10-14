"""Focused independent FIT-only train/deploy audit contracts."""
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import verify_train_deploy_gap as audit


class TrainDeployAuditTests(unittest.TestCase):
    def test_two_class_decision_is_per_bit_and_selected_temperature(self):
        logits = np.array([[-.3, 0., .3]], np.float32)
        noise = np.array([[[[0., 1.], [2., 0.], [0., 1.]]]], np.float32)
        soft = audit.reconstruct_codes(logits, noise, .875, 'gumbel')
        expected = np.tanh(np.array([[[-1.6, 2., -.4]]]) / (2 * .875))
        np.testing.assert_allclose(soft, expected, atol=2e-7, rtol=2e-7)
        hard = audit.reconstruct_codes(logits, noise, .875, 'st')
        np.testing.assert_array_equal(hard, [[[-1., 1., -1.]]])
        tanh = audit.reconstruct_codes(logits, np.empty(0), .875, 'tanh')
        np.testing.assert_allclose(tanh, np.tanh(logits / .875)[None], atol=2e-7, rtol=2e-7)
        self.assertFalse(np.allclose(tanh, np.tanh(logits / .25)[None]))
        with self.assertRaisesRegex(ValueError, 'TWO_CLASS_NOISE_SHAPE'):
            audit.reconstruct_codes(logits, noise[..., 0], .875, 'gumbel')

    def test_noise_clock_uses_checkpoint_counter_and_two_branches(self):
        counters, seeds = audit.expected_clock(92601, 199, 'st')
        np.testing.assert_array_equal(counters, np.stack((np.arange(199, 215, 2), np.arange(201, 217, 2)), -1))
        self.assertEqual(seeds[0, 0], 92601 * 1000003 + 199)
        self.assertEqual(seeds[-1, 1], 92601 * 1000003 + 214)
        counters, seeds = audit.expected_clock(1, 4, 'tanh')
        np.testing.assert_array_equal(counters, np.full((8, 2), 4))
        self.assertEqual(seeds.shape, (8, 0))

    def test_metrics_weight_observed_channels_and_no_memory_is_explicit(self):
        prediction = np.array([[[2., 4.], [2., 999.]]], np.float32)
        target = np.zeros_like(prediction)
        observed = np.array([[[True, True], [True, False]]])
        arrays = {'deploy_prediction': prediction, 'target': target, 'observed': observed,
            'train_predictions': np.repeat(prediction[None], 8, 0),
            'train_query_codes': np.ones((8, 1, 2, 2), np.float32), 'deploy_query_codes': np.ones((1, 2, 2), np.float32),
            'train_key_codes': np.ones((8, 3, 2), np.float32), 'deploy_key_codes': np.ones((3, 2), np.float32),
            'eligible': np.zeros((1, 2, 3), bool), 'query_logits': np.zeros((1, 2, 2), np.float32),
            'cached_key_logits': np.zeros((3, 2), np.float32), 'recomputed_query_logits': np.zeros((1, 2, 2), np.float32),
            'recomputed_key_logits': np.zeros((3, 2), np.float32), 'recomputed_sign_prediction': prediction.copy()}
        metrics = audit.recalculate_metrics(arrays, 'gumbel')
        self.assertEqual(metrics['deploy_fit']['mse'], 10.)
        self.assertEqual(metrics['deploy_fit']['mae'], 3.)
        self.assertEqual(metrics['deploy_fit']['per_channel_observed_count'], [2, 1])
        self.assertIsNone(metrics['query_code_sign_flip_fraction_eligible_rows'])
        self.assertIsNone(metrics['key_code_sign_flip_fraction_used_keys'])
        self.assertEqual(metrics['analytic_gumbel_flip_probability']['query_mean'], .5)
        self.assertIsNone(metrics['analytic_gumbel_flip_probability']['used_key_mean'])

    def test_fit_guard_rejects_target_or_test_before_hash(self):
        for name in ('EVAL_future_target_payload.npz', 'protected_TEST.npz'):
            with mock.patch.object(audit.vp, 'file_hash', side_effect=AssertionError('must not hash')):
                with self.assertRaises(PermissionError):
                    audit.fit_path({'path': name, 'sha256': 'unused'})

    def test_affine_roundoff_bound_detects_wrong_weights(self):
        x = np.array([[1., -2., 3.]], np.float32)
        weight = np.array([[2., 3., 4.], [-.2, .7, 1.]], np.float32)
        bias = np.array([.5, -.3], np.float32)
        logits = (x @ weight.T + bias).astype(np.float32)
        valid = audit.affine_budget(x, weight, bias, logits)
        self.assertTrue(valid['passed'])
        logits[0, 0] += .001
        self.assertFalse(audit.affine_budget(x, weight, bias, logits)['passed'])


if __name__ == '__main__':
    unittest.main()
