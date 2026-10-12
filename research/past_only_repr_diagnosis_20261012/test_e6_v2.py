"""Finite diagnostic and actual thread-contract fixtures, not scientific fits."""
from pathlib import Path
import json
import sys
import unittest
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import Budget,access_guard,write,sha
from test_e6 import GraphTests
import e6
import e6_v2


class V2Tests(GraphTests):
    def setUp(self):
        self.fit_graph,self.empty_graph = e6_v2.fit_graph,e6_v2.empty_graph
        self.wrong_graph,self.messages = e6_v2.wrong_graph,e6_v2.graph_messages
        self.features,self.edges = e6_v2.lagged_features,e6_v2.select_edges
        self.pairs,self.readout_features = e6_v2.pooled_pairs,e6_v2.readout_features

    def test_single_static_snapshot_has_no_observed_transition(self):
        source = np.zeros((2,1,3,2),np.int16)
        lag = np.ones_like(source); weight = np.zeros_like(source,dtype=np.float32)
        diag = e6_v2.graph_change_diagnostics(source,lag,weight,'FIXED_LAG_GRAPH')
        self.assertEqual(diag['observed_transition_count'],0)
        self.assertEqual(diag['graph_update_event_count'],0)
        self.assertEqual(diag['declared_static_graph_change'],0.)
        self.assertIsNone(diag['mean_foreign_edge_id_change_per_recorded_event'])
        self.assertEqual(diag['missing_change_statistic_reason'],'NO_OBSERVED_STATE_TRANSITION')
        json.dumps(diag,allow_nan=False)

    def test_repeated_static_snapshots_have_observed_zero_change(self):
        source = np.zeros((2,17,3,2),np.int16)
        lag = np.ones_like(source); weight = np.zeros_like(source,dtype=np.float32)
        diag = e6_v2.graph_change_diagnostics(source,lag,weight,'WRONG_EDGE_GRAPH')
        self.assertEqual(diag['observed_transition_count'],32)
        self.assertEqual(diag['observed_changed_event_count'],0)
        self.assertEqual(diag['mean_foreign_edge_id_change_per_recorded_event'],0.)
        self.assertEqual(diag['graph_update_event_count'],0)
        self.assertIsNone(diag['missing_change_statistic_reason'])
        json.dumps(diag,allow_nan=False)

    def test_actual_v4_nan_is_empty_tune_change_regime(self):
        records = [{'std':np.ones(6),'mean':np.zeros(6),
                    'metadata':{'change_timestamp_effective':1000}}]
        pairs = np.array([[0,600]],np.int64)
        pred = np.zeros((1,48,6)); mask = np.ones(pred.shape,bool)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            old = e6.physical_metrics(records,pairs,pred,pred,mask)
        self.assertTrue(np.isnan(old['per_record'][0]['after_change']['mse']))
        with self.assertRaises(ValueError):
            json.dumps(old,allow_nan=False)
        fixed = e6_v2.physical_metrics(records,pairs,pred,pred,mask)
        after = fixed['per_record'][0]['after_change']
        self.assertIsNone(after['mse']); self.assertIsNone(after['mae'])
        self.assertEqual(after['reason'],'NO_OBSERVED_TARGET_IN_THIS_REGIME')
        self.assertFalse(after['zero_error_claim'])
        self.assertEqual(fixed['per_record'][0]['before_change']['mse'],0.)
        json.dumps(fixed,allow_nan=False)

    def test_effective_numpy_and_torch_threads_are_two(self):
        with e6_v2.cpu_thread_contract() as evidence:
            self.assertEqual(evidence['torch_threads'],2)
            numerical = [p for p in evidence['loaded_libraries_effective'] if p['user_api'] in ('blas','openmp')]
            self.assertTrue(numerical)
            self.assertTrue(all(p['num_threads'] <= 2 for p in numerical))

    def test_preserved_v4_static_trace_has_seventeen_states(self):
        path = e6_v2.CACHE/'E6_synthetic_changed_synthetic_changed_cpu_v4'/'GRAPH_INPUT_fit.npz'
        self.assertTrue(path.is_file())
        with np.load(path,allow_pickle=False) as z:
            source = z['FIXED_LAG_GRAPH_trace_source']
            lag = z['FIXED_LAG_GRAPH_trace_lag']
            weight = z['FIXED_LAG_GRAPH_trace_weight']
            self.assertEqual(source.shape,(256,17,6,2))
            self.assertEqual(z['PAST_UPDATED_LAG_GRAPH_trace_source'].shape,(256,16,6,2))
        diag = e6_v2.graph_change_diagnostics(source,lag,weight,'FIXED_LAG_GRAPH')
        self.assertEqual(diag['observed_transition_count'],256*16)
        self.assertEqual(diag['observed_changed_event_count'],0)
        json.dumps(diag,allow_nan=False)

    def test_preserved_v4_all_tune_targets_precede_change(self):
        path = HERE/'plans'/'E6_synthetic_changed_synthetic_changed_cpu_v4.json'
        plan = e6_v2.read(path)
        for rid,o in plan['pairs']['tune']:
            change = plan['records'][rid]['synthetic']['change_timestamp_effective']
            self.assertLess(o+48-1,change)


if __name__ == '__main__':
    access_guard()
    default_libraries = e6_v2.threadpool_info()
    with Budget(2,gpu=False,label='E6_v2 finite/thread CPU fixtures') as budget:
        with e6_v2.cpu_thread_contract() as actual_threads:
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(V2Tests))
            budget.check()
            if budget.record['wall_seconds'] > 2:
                raise RuntimeError('Allocated2-second numerical fixture exceeded')
            write(HERE/'receipts'/f'E6_V2_CPU_TEST_{budget.path.stem}.json',
                  {'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                   'actual_science_updates':0,'GPU_calls':0,'budget_ledger':str(budget.path),
                   'default_thread_libraries':default_libraries,'effective_thread_contract':actual_threads,
                   'source_pins':{str(p):sha(p) for p in [HERE/'e6.py',HERE/'e6_v2.py',Path(__file__),HERE/'common.py']},
                   'preserved_failure_artifact_pins':{str(p):sha(p) for p in [
                       HERE/'plans'/'E6_synthetic_changed_synthetic_changed_cpu_v4.json',
                       e6_v2.CACHE/'E6_synthetic_changed_synthetic_changed_cpu_v4'/'GRAPH_INPUT_fit.npz']}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
