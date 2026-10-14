"""CPU audit fixtures: masks, selection, isolation and tamper rejection."""
import importlib
from pathlib import Path
import unittest
import numpy as np


class IndependentAuditTests(unittest.TestCase):
    def module(self):
        self.assertTrue(Path(__file__).with_name('validation_contracts.py').is_file(),
                        'Independent validator implementation is missing')
        return importlib.import_module('validation_contracts')

    def test_channel_macro_weights_channels_equally(self):
        v = self.module()
        prediction = np.array([[[2., 10.], [2., 999.]]], np.float32)
        target = np.zeros_like(prediction)
        observed = np.array([[[True, True], [True, False]]])
        actual = v.independent_score(prediction, target, observed)
        self.assertEqual(actual['mse'], 52.)
        self.assertEqual(actual['mae'], 6.)
        self.assertEqual(actual['observed_counts'], [2, 1])

    def test_nonfinite_unobserved_target_is_ignored_but_prediction_is_not(self):
        v = self.module()
        target = np.array([[[1., np.nan]]])
        mask = np.array([[[True, False]]])
        self.assertEqual(v.independent_score(np.zeros_like(target), target, mask)['mse'], 1.)
        with self.assertRaises(ValueError):
            v.independent_score(np.array([[[0., np.nan]]]), target, mask)

    def test_checkpoint_uses_tune_only_and_earliest_exact_tie(self):
        v = self.module()
        cp = [{'update': 0, 'forecast': {'tune': {'normalized': {'mse': 2.}}}},
              {'update': 100, 'forecast': {'tune': {'normalized': {'mse': 1.}}}},
              {'update': 600, 'forecast': {'tune': {'normalized': {'mse': 1.}}}}]
        self.assertEqual(v.independent_checkpoint_choice(cp), 100)

    def test_selection_requires_unique_dataset_seed_coverage(self):
        v = self.module()
        duplicated = [{'dataset': 'jena', 'seed': 92601} for _ in range(4)]
        self.assertFalse(v.complete_development_coverage(duplicated))
        valid = [{'dataset': d, 'seed': s} for d in ('jena', 'robin') for s in (92601, 92602)]
        self.assertTrue(v.complete_development_coverage(valid))

    def test_each_window_resets_missing_initial_values(self):
        v = self.module()
        physical = np.array([[[999., 4.], [8., 999.]]], np.float32)
        mask = np.array([[[False, True], [True, False]]])
        actual = v.independent_window_fill(physical, mask, np.array([2., 2.], np.float32), np.array([2., 2.], np.float32))
        self.assertTrue(np.array_equal(actual, np.array([[[0., 1.], [3., 1.]]], np.float32)))

    def test_fit_statistics_preserve_source_reduction_layout(self):
        v = self.module()
        rng = np.random.default_rng(27)
        physical = np.asfortranarray((1000. + rng.normal(size=(11841, 3))).astype(np.float32))
        mask = np.ones_like(physical, dtype=bool)
        mean, std = v.independent_fit_statistics(physical, mask, (0, len(physical)))
        self.assertTrue(np.array_equal(mean, np.nanmean(physical, axis=0)))
        self.assertTrue(np.array_equal(std, np.nanstd(physical, axis=0)))
        c_mean = np.nanmean(np.ascontiguousarray(physical), axis=0)
        self.assertGreater(float(np.max(np.abs(c_mean - mean))), 0.)

    def test_missing_projection_requires_observed_full_rank(self):
        v = self.module()
        basis = np.eye(3, dtype=np.float32)[:, :2]
        y = np.array([[[3., 5., 7.], [3., 5., 7.]]], np.float32)
        mask = np.array([[[True, True, False], [True, False, True]]])
        coord, valid = v.independent_coordinates(y, mask, np.zeros((1, 3), np.float32), basis)
        self.assertTrue(np.array_equal(valid, np.array([[True, False]])))
        self.assertTrue(np.array_equal(coord[0, 0], np.array([3., 5.])))

    def test_two_adjacent_points_are_true_ls_slope(self):
        v = self.module()
        coord = np.array([[[3.], [5.], [0.]]], np.float32)
        valid = np.array([[True, True, False]])
        summary, mask = v.independent_summary(coord, valid)
        self.assertEqual(float(summary[0, -1]), 2.)
        self.assertFalse(bool(mask[0, 1]))

    def test_eval_is_denied_before_explicit_audit_authorization(self):
        v = self.module()
        with self.assertRaises(PermissionError):
            v.require_eval_authorization(False)
        v.require_eval_authorization(True)

    def test_running_receipt_is_not_terminal_evidence(self):
        v = self.module()
        with self.assertRaises(ValueError):
            v.require_terminal({'status': 'RUNNING'})
        v.require_terminal({'status': 'COMPLETE'})

    def test_checkpoint_coverage_rejects_duplicate_and_missing_updates(self):
        v = self.module()
        records = [{'dataset': 'jena', 'label': 'm', 'update': u} for u in (0, 100, 100, 600)]
        actual = v.checkpoint_coverage(records, {('jena', 'm'): (0, 100, 200, 400, 600)})
        self.assertEqual(actual['duplicates'], [('jena', 'm', 100)])
        self.assertEqual(actual['missing'], [('jena', 'm', 200), ('jena', 'm', 400)])
        self.assertFalse(actual['complete'])

    def test_q_reapplication_preserves_matrix_orientation(self):
        v = self.module()
        before = np.array([[[1., 2.], [3., 4.]]], np.float32)
        weight = np.array([[2., 1.], [0., 3.]], np.float32)
        bias = np.array([5., 7.], np.float32)
        actual = v.independent_q_apply(before, weight, bias)
        self.assertTrue(np.array_equal(actual, np.array([[[9., 13.], [15., 19.]]], np.float32)))

    def test_latent_diag_ignores_invalid_horizon_and_keeps_summary_mask(self):
        v = self.module()
        before = np.array([[[1.], [3.], [999.]]], np.float32)
        target = np.array([[[0.], [0.], [0.]]], np.float32)
        valid = np.array([[True, True, False]])
        summary, _ = v.independent_summary(target, valid)
        actual = v.independent_alignment_diagnostics(before, target, valid, np.ones(1),
                    np.zeros(3), np.ones(3), summary, np.array([False]))
        self.assertEqual(actual['latent_future_normalized_mse'], 5.)
        self.assertIsNone(actual['complete_summary_normalized_mse'])


if __name__ == '__main__':
    unittest.main()
