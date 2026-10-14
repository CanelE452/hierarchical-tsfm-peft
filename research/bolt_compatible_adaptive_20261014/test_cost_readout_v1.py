"""Meaningful source fixtures; execute only in the fresh approved runtime."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cost_readout_v1 as cost


class CostTests(unittest.TestCase):
    def test_balanced_routes_each_position_once(self):
        schedule = cost.balanced_schedule()
        self.assertEqual(len(schedule), 6)
        for order in schedule:
            self.assertEqual(set(order), set(cost.ROUTES))
        for position in range(6):
            self.assertEqual({order[position] for order in schedule}, set(cost.ROUTES))

    def test_causal_fill_resets_each_window(self):
        physical = np.zeros((2, 512, 2), np.float32)
        observed = np.zeros_like(physical, dtype=np.bool_)
        mean = np.array([2., -3.], np.float32)
        physical[0, 10] = [7., 9.]
        observed[0, 10] = True
        before = cost.fill_vectorized(physical, observed, mean)
        self.assertTrue(np.array_equal(before[0, :10], np.broadcast_to(mean, (10, 2))))
        self.assertTrue(np.array_equal(before[1], np.broadcast_to(mean, (512, 2))))
        physical[0, 500] = [100., 200.]
        observed[0, 500] = True
        after = cost.fill_vectorized(physical, observed, mean)
        self.assertTrue(np.array_equal(before[0, :500], after[0, :500]))

    def test_whole_cost_preserves_output_and_balances_both_batches(self):
        physical = np.arange(8 * 512 * 2, dtype=np.float32).reshape(8, 512, 2) / 100
        observed = np.ones_like(physical, dtype=np.bool_)
        mean, std = np.array([1., 2.], np.float32), np.array([2., 3.], np.float32)
        history = cost.preprocess(physical, observed, mean, std)
        origins = np.arange(8, dtype=np.int64) + 512
        def predictor(value):
            return np.repeat(value[:, -1:], 48, axis=1)
        expected = predictor(history)
        with tempfile.TemporaryDirectory() as directory:
            result = cost.measure_dataset('fixture', physical, observed, history, mean, std, origins,
                {route: predictor for route in cost.ROUTES},
                {route: {'selected_kind': 'PERSIST'} for route in cost.ROUTES},
                {route: expected for route in cost.ROUTES}, Path(directory) / 'cost',
                lambda: None, lambda: None, lambda: None, lambda: {'peak_allocated_vram_bytes': 0})
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(result['rows']), 12)
            for row in result['rows']:
                self.assertEqual(row['repeats'], 6)
                self.assertEqual(set(row['positions']), set(range(6)))
                self.assertTrue(all(all(value['pass'] for value in checks.values()) for checks in row['parity']))
                with np.load(row['timed_outputs']['path'], allow_pickle=False) as opened:
                    self.assertTrue(np.array_equal(opened['normalized'][0], expected[:row['batch']]))


if __name__ == '__main__':
    unittest.main()
