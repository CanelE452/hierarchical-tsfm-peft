"""CPU information and graph-contract tests; no scientific neural fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Budget, access_guard, write, sha


class GraphTests(unittest.TestCase):
    def setUp(self):
        from e6 import (fit_graph, empty_graph, wrong_graph, graph_messages,
                        lagged_features, select_edges, pooled_pairs, readout_features)
        self.fit_graph, self.empty_graph = fit_graph, empty_graph
        self.wrong_graph, self.messages = wrong_graph, graph_messages
        self.features, self.edges = lagged_features, select_edges
        self.pairs, self.readout_features = pooled_pairs, readout_features

    def test_lag_direction_and_no_current_timestamp(self):
        x = np.arange(40*3).reshape(40, 3)
        rows = self.features(x, np.array([32, 39]))
        np.testing.assert_array_equal(rows[0, :3], x[31])
        np.testing.assert_array_equal(rows[0, -3:], x[0])
        np.testing.assert_array_equal(rows[1, :3], x[38])

    def test_edges_exclude_self_and_normalize_signed_weights(self):
        beta = np.zeros((32*4, 4))
        beta[0, 0] = 100
        beta[4+1, 0] = -4
        beta[8+2, 0] = 2
        g = self.edges(beta)
        self.assertEqual(g['source'][0].tolist(), [1, 2])
        self.assertEqual(g['lag'][0].tolist(), [2, 3])
        np.testing.assert_allclose(g['weight'][0], [-2/3, 1/3])
        self.assertLessEqual(float(abs(g['weight']).sum(1).max()), 1.)

    def test_wrong_edges_preserve_counts_lags_and_signed_weights(self):
        rng = np.random.default_rng(5)
        g = self.edges(rng.normal(size=(32*6, 6)))
        wrong = self.wrong_graph(g)
        np.testing.assert_array_equal(g['lag'], wrong['lag'])
        np.testing.assert_array_equal(g['weight'], wrong['weight'])
        self.assertTrue(np.all(wrong['source'] != np.arange(6)[:, None]))
        self.assertTrue(np.all(wrong['source'] != g['source']))

    def test_neighbor_message_is_signed_past_value(self):
        x = np.arange(12*3, dtype=np.float32).reshape(12, 3)
        g = self.empty_graph(3)
        g['source'][1, 0], g['lag'][1, 0], g['weight'][1, 0] = 0, 3, -.5
        msg, trace = self.messages(x, np.ones_like(x, bool), 'FIXED_LAG_GRAPH', g, 1.)
        np.testing.assert_array_equal(msg[:3, 1], 0)
        np.testing.assert_array_equal(msg[3:, 1], -.5*x[:-3, 0])
        self.assertEqual(trace['effective_cutoff_inclusive'][0], -1)

    def test_updated_prefix_has_no_suffix_or_external_history(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(160, 3)).astype(np.float32)
        altered = x.copy(); altered[100:] *= 30
        mask = np.ones_like(x, bool)
        a, ta = self.messages(x, mask, 'PAST_UPDATED_LAG_GRAPH', self.empty_graph(3), .01)
        b, tb = self.messages(altered, mask, 'PAST_UPDATED_LAG_GRAPH', self.empty_graph(3), .01)
        np.testing.assert_array_equal(a[:100], b[:100])
        np.testing.assert_array_equal(ta['weight'][:3], tb['weight'][:3])
        self.assertEqual(ta['update_indices'].tolist(), [31, 63, 95, 127, 159])
        self.assertEqual(ta['effective_cutoff_inclusive'].tolist(), [-1, 63, 95, 127, 159])
        self.assertTrue(np.all(ta['effective_cutoff_inclusive'] <= ta['update_indices']))
        self.assertTrue(np.all(a[:63] == 0))

    def test_graph_fit_observes_only_its_argument_and_observed_targets(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(128, 3)); mask = np.ones_like(x, bool)
        mask[32:, 2] = False
        g = self.fit_graph(x, mask, .01)
        self.assertEqual(g['fit_samples_by_destination'][2], 0)
        self.assertTrue(np.all(g['source'][2] == -1))
        self.assertTrue(np.all(g['weight'][2] == 0))
        self.assertEqual(g['fit_support_length'], 128)

    def test_common_causal_encoder_preserves_past(self):
        from model_neural import TinyEncoder
        torch.manual_seed(3)
        e = TinyEncoder(6, causal=True).eval()
        x = torch.randn(2, 100, 6); z = x.clone(); z[:, 66:] *= 20
        with torch.no_grad():
            a, b = e(x), e(z)
        torch.testing.assert_close(a[:, :66], b[:, :66], atol=0, rtol=0)
        self.assertEqual(a.shape, (2, 100, 16))

    def test_pooled_origin_selection_retains_record_ids(self):
        records = [{'origins': {'fit': np.arange(512, 1000)}},
                   {'origins': {'fit': np.arange(512, 1000)}}]
        p = self.pairs(records, 'fit', 256)
        self.assertEqual(p.shape, (256, 2))
        self.assertEqual(set(p[:, 0]), {0, 1})
        self.assertTrue(np.all(p[:, 1] >= 512))
        latent = np.arange(2*512*16).reshape(2, 512, 16)
        f = self.readout_features(latent)
        np.testing.assert_array_equal(f[0, :16], latent[0, -1])
        np.testing.assert_array_equal(f[0, -16:], latent[0, -33])

    def test_window_initial_fill_does_not_take_pre_history_value(self):
        from e6 import prepare_history
        physical = np.arange(600*3,dtype=np.float32).reshape(600,3)
        mask = np.ones_like(physical,bool); mask[48:51,1] = False
        physical[~mask] = np.nan
        data = {'physical':physical,'mask':mask,'mean':np.array([1,2,3],np.float32),
                'std':np.ones(3,np.float32)}
        x = prepare_history(data,560)
        np.testing.assert_array_equal(x[:3,1],0.)
        self.assertEqual(x[3,1],physical[51,1]-2)
        altered = physical.copy(); altered[:48] = -1e6
        np.testing.assert_array_equal(x,prepare_history({**data,'physical':altered},560))


if __name__ == '__main__':
    access_guard(); torch.set_num_threads(2)
    with threadpool_limits(limits=2), Budget(2, gpu=False, label='E6 CPU graph-contract tests') as budget:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GraphTests))
        budget.check()
        write(HERE/'receipts'/f'E6_CPU_TEST_{budget.path.stem}.json',
              {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'science_training_updates': 0, 'gpu_calls': 0, 'budget_ledger': str(budget.path),
               'source_pins':{str(p):sha(p) for p in [HERE/'e6.py',Path(__file__),HERE/'common.py',HERE/'model_neural.py']}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
