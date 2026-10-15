"""Synthetic-only purged readout contracts; never opens research assets."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.find_spec('run_purged_readout_stability_v1')
if SPEC is not None:
    import run_purged_readout_stability_v1 as run
else:
    run = None


class RunnerAvailabilityTests(unittest.TestCase):
    def test_runner_exists_before_any_real_asset_can_be_run(self):
        self.assertIsNotNone(run, 'purged readout controller is not implemented')


@unittest.skipIf(run is None, 'controller not implemented')
class PurgedReadoutTests(unittest.TestCase):
    def test_sorted_partition_purges_response_overlap_without_deleting_history_overlap(self):
        origins = np.arange(512, 768, dtype=np.int64)[::-1].copy()
        parts, summary = run.partition_origins(origins)
        np.testing.assert_array_equal(origins[parts['A']], np.arange(512, 593))
        np.testing.assert_array_equal(origins[parts['B']], np.arange(640, 657))
        np.testing.assert_array_equal(origins[parts['C']], np.arange(704, 768))
        self.assertEqual(summary['purge']['A_removed_count'], 47)
        self.assertEqual(summary['purge']['B_removed_count'], 47)
        self.assertLess(int(origins[parts['A']].max()) + 47, 640)
        self.assertLess(int(origins[parts['B']].max()) + 47, 704)

    def test_duplicate_origins_are_rejected_without_repartition(self):
        origins = np.arange(512, 768, dtype=np.int64)
        origins[-1] = origins[-2]
        with self.assertRaisesRegex(ValueError, 'UNIQUE'):
            run.partition_origins(origins)

    def test_timestamp_counts_are_unique_time_facts_not_independent_samples(self):
        origins = np.array([512, 513], dtype=np.int64)
        mask = np.ones((2, 48, 2), dtype=bool)
        mask[1, :, 1] = False
        facts = run.timestamp_facts(origins, mask)
        self.assertEqual(facts['horizon_slots'], 96)
        self.assertEqual(facts['unique_target_timestamps'], 49)
        self.assertEqual(facts['max_target_multiplicity'], 2)
        self.assertEqual(facts['observed_counts_per_channel'], [96, 48])
        self.assertEqual(facts['unique_observed_target_timestamps_per_channel'], [49, 48])

    def test_scores_use_observed_channel_macro_and_ignore_nan_placeholders(self):
        target = np.zeros((1, 48, 2), np.float32)
        observed = np.zeros_like(target, dtype=bool)
        observed[:, :, 0] = True
        observed[:, 0, 1] = True
        target[~observed] = np.nan
        prediction = np.ones_like(target)
        prediction[:, 0, 1] = 3
        raw = {'target': target, 'observed': observed, 'std': np.ones(2, np.float32)}
        self.assertEqual(run.score_prediction(prediction, raw)['scores_all_observed']['normalized']['mse'], 5.)

    def test_B_selection_uses_registered_ties_and_does_not_read_C(self):
        rows = [{'id': 'ALPHA_0', 'status': 'COMPLETE', 'B_MSE': 1.},
                {'id': 'ZERO_DELTA', 'status': 'COMPLETE', 'B_MSE': 1.}]
        self.assertEqual(run.select_B(rows)['id'], 'ALPHA_0')
        rows[-1]['B_MSE'] = .9
        self.assertEqual(run.select_B(rows)['id'], 'ZERO_DELTA')

    def test_both_route_locks_and_registry_precede_any_C_inference_even_on_failure(self):
        result = {'routes': {}, 'failed_paths': [], 'totals': {'resolved_selection_locks': 0, 'selected_C_evaluations_completed': 0}}
        events = []
        def fit(space):
            if space == 'RAW_C':
                raise ValueError('synthetic route failure')
            return {'status': 'COMPLETE'}
        def seal(space, row):
            events.append(('lock', space))
            return {'path': space + '_lock'}
        def registry(rows):
            events.append(('registry', len(rows)))
            return {'path': 'registry'}
        resolved = run.resolve_paths(result, fit, seal, registry)
        self.assertEqual(len(result['failed_paths']), 1)
        def infer(space, row):
            self.assertEqual(events, [('lock', 'PCA_K8'), ('lock', 'RAW_C'), ('registry', 2)])
            return {'status': 'COMPLETE'}
        run.infer_C(result, resolved, infer)
        self.assertEqual(result['totals']['selected_C_evaluations_completed'], 1)
        with self.assertRaisesRegex(ValueError, 'BOTH'):
            run.infer_C({'both_locks_before_C': False}, resolved, infer)

    def test_A_only_scale_and_residual_fit_do_not_depend_on_B_or_C_labels(self):
        import horizon_decoder_delta_v1 as delta
        origins = np.arange(512, 768, dtype=np.int64) * 48
        parts, _ = run.partition_origins(origins)
        x = np.ones((256, 48, 2), np.float32)
        x[128:] = 1000
        scale, constant = delta.feature_scale(x[parts['A']])
        np.testing.assert_array_equal(scale, [1., 1.])
        target = np.full((256, 48, 1), 2., np.float32)
        observed = np.ones_like(target, dtype=bool)
        base = np.zeros_like(target)
        states, _ = run.fit_A(x, base, target, observed, parts['A'])
        target[128:] = -9999
        repeated, _ = run.fit_A(x, base, target, observed, parts['A'])
        np.testing.assert_array_equal(states[4]['coefficient_FP32'], repeated[4]['coefficient_FP32'])

    def test_import_is_stdlib_only_and_closed_gates_precede_io(self):
        code = 'import sys;sys.path.insert(0,' + repr(str(HERE)) + ');import run_purged_readout_stability_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_ONLY")'
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        self.assertIn('STDLIB_ONLY', done.stdout)
        from unittest.mock import patch
        with patch.object(run, 'PLAN_SHA', None), patch.object(run.old, 'bound', side_effect=AssertionError('forbidden IO')):
            with self.assertRaisesRegex(ValueError, 'REGISTERED'):
                run.load_gates(None)


if __name__ == '__main__':
    unittest.main()
