import tempfile
import unittest
from pathlib import Path

import numpy as np
import cost_expanded_readout_v1 as expanded
import cost_readout_v1 as cost


class ExpandedCostTests(unittest.TestCase):
    def test_two_routes_balanced_without_raw_comparator(self):
        order = expanded.schedule()
        self.assertEqual(len(order), 6)
        for route in expanded.ROUTES:
            self.assertEqual([row.index(route) for row in order].count(0), 3)
            self.assertEqual([row.index(route) for row in order].count(1), 3)

    def test_cpu_whole_outputs_preserved_and_no_raw_ratio_fabricated(self):
        physical = np.arange(8*512*2, dtype=np.float32).reshape(8, 512, 2) / 100
        observed = np.ones_like(physical, dtype=bool)
        mean, std = np.array([1., 2.], np.float32), np.array([2., 3.], np.float32)
        history = cost.preprocess(physical, observed, mean, std)
        def predictor(x):
            return np.repeat(x[:, -1:], 48, axis=1)
        pipelines = {route: cost.CPUOutputPipeline(predictor, mean, std) for route in expanded.ROUTES}
        with tempfile.TemporaryDirectory() as directory:
            rows = expanded.measure_pair(pipelines, physical, observed,
                {route: predictor(history) for route in expanded.ROUTES}, np.arange(8, dtype=np.int64)+512,
                Path(directory)/'outputs', lambda: None, cost)
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertIsNone(row['paired_RAW_ratio'])
                self.assertFalse(row['GPU_executed'])
                self.assertFalse(row['GPU_resident_model'])
                self.assertEqual(row['repeats'], 6)
                self.assertTrue(all(value['pass'] for value in row['normalized_parity']))
                with np.load(row['timed_outputs']['path'], allow_pickle=False) as opened:
                    self.assertTrue(np.array_equal(opened['normalized'][0], predictor(history)[:row['batch']]))


if __name__ == '__main__':
    unittest.main()
