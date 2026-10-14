"""CPU checks for independent, batch-matched post-cost output evidence."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import verify_deployment_outputs_batch_v3 as audit


def example_pack():
    history = np.zeros((8, 512, 1), np.float32)
    return {'mean': np.zeros(1, np.float32), 'std': np.ones(1, np.float32),
        'history': {'eval': history}, 'raw_history': {'eval': history.copy()},
        'raw_history_observed': {'eval': np.ones_like(history, bool)},
        'origins': {'eval': np.arange(8, dtype=np.int64)}}


class OutputBatchAuditTests(unittest.TestCase):
    def test_comparison_preserves_distinct_matched_and_bulk_physical_results(self):
        bulk = np.full((1, 48, 1), -2., np.float32)
        matched = np.nextafter(bulk, np.float32(-np.inf))
        mean = np.array([65536.], np.float32); std = np.array([32768.], np.float32)
        checks = audit.comparison_bundle(matched * std + mean, matched, bulk, mean, std)
        self.assertTrue(checks['matched_science_to_original_bulk_normalized']['gate_pass'])
        self.assertTrue(checks['actual_to_matched_physical']['physical_parity_pass'])
        self.assertFalse(checks['actual_to_original_bulk_physical']['physical_parity_pass'])
        self.assertTrue(checks['critical_parity_gates_pass'])
        self.assertFalse(checks['original_bulk_physical_failure_relabelled'])

    def test_normalized_bulk_gap_cannot_be_hidden_by_large_physical_scale(self):
        bulk = np.zeros((1, 48, 1), np.float32); matched = bulk + .001
        checks = audit.comparison_bundle(matched, matched, bulk, np.zeros(1, np.float32), np.ones(1, np.float32))
        self.assertTrue(checks['actual_to_matched_physical']['physical_parity_pass'])
        self.assertFalse(checks['matched_science_to_original_bulk_normalized']['gate_pass'])
        self.assertFalse(checks['critical_parity_gates_pass'])

    def test_one_actual_call_preserves_failure_arrays_and_model_cache_change(self):
        class WithCache(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1)); self.register_buffer('cache', torch.zeros(1))
        model = WithCache(); pack = example_pack()
        route = {'dataset': 'sml', 'identity': 'row', 'route': 'CORE', 'split': 'eval'}
        reference = np.zeros((1, 48, 1), np.float32)
        calls = []
        def deployment(*args):
            calls.append(args); model.cache.add_(1)
            return np.full((1, 48, 1), .1, np.float32)
        with patch.object(audit.torch.cuda, 'synchronize'):
            record, arrays = audit.audit_single_batch(pack, route, 1, reference, reference, {}, None, model, None, deployment)
        self.assertEqual(len(calls), 1)
        self.assertEqual(arrays['actual_physical'].shape, (1, 48, 1))
        self.assertTrue(np.array_equal(arrays['actual_physical'], np.full((1, 48, 1), .1, np.float32)))
        self.assertFalse(record['critical_gates_pass'])
        self.assertTrue(record['deployment_registered_parameters_unchanged'])
        self.assertFalse(record['deployment_cache_buffers_unchanged'])
        self.assertFalse(record['not_added_to_latency_statistics'] is False)

    def test_source_prediction_and_input_never_mutated_or_used_as_actual(self):
        pack = example_pack(); before = copy.deepcopy(pack)
        route = {'dataset': 'sml', 'identity': 'row', 'route': 'CORE', 'split': 'eval'}
        matched = np.zeros((8, 48, 1), np.float32)
        with patch.object(audit.torch.cuda, 'synchronize'):
            record, arrays = audit.audit_single_batch(pack, route, 8, matched, matched, {}, None, None, None,
                lambda *args: np.ones((8, 48, 1), np.float32))
        self.assertFalse(record['critical_gates_pass'])
        self.assertTrue(np.array_equal(matched, np.zeros_like(matched)))
        self.assertTrue(np.array_equal(pack['history']['eval'], before['history']['eval']))
        self.assertFalse(np.array_equal(arrays['actual_physical'], arrays['matched_science_normalized']))

    def test_nonfinite_actual_is_preserved_as_failed_output(self):
        pack = example_pack(); route = {'dataset': 'sml', 'identity': 'row', 'route': 'CORE', 'split': 'eval'}
        reference = np.zeros((1, 48, 1), np.float32)
        actual = reference.copy(); actual[0, 0, 0] = np.nan
        with patch.object(audit.torch.cuda, 'synchronize'):
            record, arrays = audit.audit_single_batch(pack, route, 1, reference, reference, {}, None, None, None,
                lambda *args: actual)
        self.assertFalse(record['critical_gates_pass'])
        self.assertIn('NONFINITE', record['comparisons']['comparison_error'])
        self.assertTrue(np.isnan(arrays['actual_physical'][0, 0, 0]))

    def test_wrong_actual_dtype_cannot_pass_original_fp32_implementation_audit(self):
        pack = example_pack(); route = {'dataset': 'sml', 'identity': 'row', 'route': 'CORE', 'split': 'eval'}
        reference = np.zeros((1, 48, 1), np.float32)
        with patch.object(audit.torch.cuda, 'synchronize'):
            record, arrays = audit.audit_single_batch(pack, route, 1, reference, reference, {}, None, None, None,
                lambda *args: np.zeros((1, 48, 1), np.float64))
        self.assertFalse(record['critical_gates_pass'])
        self.assertFalse(record['FP32_dtype_contract_pass'])
        self.assertEqual(arrays['actual_physical'].dtype, np.float64)

    def test_independent_readout_model_is_separate_and_only_memory_buffers_may_change(self):
        class WithMemory(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
                self.register_buffer('memory_embedding', torch.zeros(1))
        science = WithMemory(); deployment = WithMemory()
        route = {'route': 'E_ENRICHED_RIDGE', 'split': 'eval', 'case': {}, 'readout': {'method': 'E_ENRICHED_RIDGE'}}
        def evaluate(thin, split, case, model, readout, runtime, limit):
            self.assertNotIn('target', thin); self.assertNotIn('observed', thin)
            model.memory_embedding.add_(1)
            return np.zeros((1, 48, 1), np.float32)
        with patch.object(audit.bs, 'load_selected_model', return_value=science), patch.object(audit.pr, 'evaluate_readout', side_effect=evaluate):
            prediction, metadata = audit.independent_science_reference(example_pack(), route, 1, None, None, device='cpu')
        self.assertEqual(prediction.shape, (1, 48, 1))
        self.assertEqual(float(deployment.memory_embedding), 0.)
        self.assertEqual(metadata['changed_buffer_names'], ['memory_embedding'])
        self.assertTrue(metadata['only_declared_science_memory_buffers_changed'])

    def test_independent_science_rejects_changed_parameters(self):
        model = torch.nn.Linear(1, 1)
        route = {'route': 'CORE', 'split': 'eval', 'case': {}}
        def predict(model, *args, **kwargs):
            with torch.no_grad(): model.weight.add_(1)
            return np.zeros((1, 48, 1), np.float32)
        with patch.object(audit.bs, 'load_selected_model', return_value=model), patch.object(audit.tr, 'predict', side_effect=predict):
            with self.assertRaisesRegex(ValueError, 'SCIENCE_REGISTERED_PARAMETERS_CHANGED'):
                audit.independent_science_reference(example_pack(), route, 1, None, None, device='cpu')

    def test_independent_science_rejects_undeclared_readout_buffer_change(self):
        class Undeclared(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1)); self.register_buffer('other', torch.zeros(1))
        model = Undeclared()
        route = {'route': 'E_RAW_RIDGE', 'split': 'eval', 'case': {}, 'readout': {'method': 'E_RAW_RIDGE'}}
        def evaluate(*args):
            model.other.add_(1)
            return np.zeros((1, 48, 1), np.float32)
        with patch.object(audit.bs, 'load_selected_model', return_value=model), patch.object(audit.pr, 'evaluate_readout', side_effect=evaluate):
            with self.assertRaisesRegex(ValueError, 'SCIENCE_UNDECLARED_BUFFER_CHANGED'):
                audit.independent_science_reference(example_pack(), route, 1, None, None, device='cpu')

    def test_coefficient_hash_detects_dtype_or_value_changes_without_refit(self):
        ridge = {'weight': np.zeros((3, 48), np.float32), 'bias': np.zeros(48, np.float32)}
        before = audit.coefficient_hash(ridge)
        ridge['weight'][0, 0] = 1.
        self.assertNotEqual(before, audit.coefficient_hash(ridge))
        ridge['weight'][0, 0] = 0.
        self.assertEqual(before, audit.coefficient_hash(ridge))
        ridge['weight'] = ridge['weight'].astype(np.float64)
        self.assertNotEqual(before, audit.coefficient_hash(ridge))

    def test_sealed_cost_requires_all_536_real_positive_seven_repeat_rows(self):
        routes = [{'dataset': 'sml', 'identity': str(index), 'route': 'CORE', 'split': 'eval'} for index in range(268)]
        records = [{**route, 'batch': batch, 'warmup': 2, 'repeats': 7, 'seconds': [.01] * 7,
            'median_seconds': .01, 'median_seconds_per_window': .01 / batch,
            'parity_pass': True, 'physical_parity_atol': .002, 'physical_parity_rtol': 2e-5}
            for route in routes for batch in (1, 8)]
        original_rows = copy.deepcopy(records)
        mapping = audit.sealed_cost_mapping({'status': 'COMPLETE', 'records': records}, routes)
        self.assertEqual(len(mapping), 536); self.assertEqual(records, original_rows)
        for change in ('missing', 'duplicate', 'zero_time', 'wrong_gate', 'wrong_median', 'failed_parity', 'failed_status'):
            bad = {'status': 'COMPLETE', 'records': copy.deepcopy(records)}
            if change == 'missing': bad['records'].pop()
            elif change == 'duplicate': bad['records'][1] = bad['records'][0]
            elif change == 'zero_time': bad['records'][0]['seconds'][0] = 0.
            elif change == 'wrong_gate': bad['records'][0]['physical_parity_atol'] = .02
            elif change == 'wrong_median': bad['records'][0]['median_seconds'] = .02
            elif change == 'failed_status': bad['status'] = 'FAILED_PRESERVED'
            else: bad['records'][0]['parity_pass'] = False
            with self.subTest(change=change), self.assertRaises(ValueError): audit.sealed_cost_mapping(bad, routes)

    def test_bulk_loader_reads_only_prediction_prefix_origins_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.npz'; pack = example_pack()
            predictions = np.zeros((20, 48, 1), np.float32)
            np.savez(path, eval_prediction=predictions, eval_origins=np.arange(20, dtype=np.int64),
                mean=pack['mean'], std=pack['std'])
            route = {'split': 'eval', 'science_prediction': {'path': str(path), 'sha256': audit.rt.sha(path)}}
            with patch.object(audit.bs, 'verify_artifact', return_value=path):
                result, metadata = audit.load_bulk_prefix(route, pack)
            self.assertEqual(result.shape, (8, 48, 1))
            self.assertEqual(metadata['full_source_window_count'], 20)
            self.assertFalse(metadata['EVAL_target_arrays_opened'])
            self.assertEqual(metadata['accessed_members'], ['eval_prediction', 'eval_origins', 'mean', 'std'])

    def test_implementation_uses_recorded_producer_and_rejects_mixed_ridge_backends(self):
        routes = [{'dataset': 'sml', 'identity': str(i), 'route': 'CORE' if i < 148 else 'RAW_JOINT_RIDGE', 'split': 'eval'} for i in range(268)]
        records = [{**route, 'batch': batch} for route in routes for batch in (1, 8)]
        producers = [{'key': list(audit.bs.row_key(row)), 'deployment_implementation':
            'original_cuda_fp32' if row['route'] == 'CORE' else 'cpu_numpy_fp32_ridge_v1'} for row in records]
        for row in records:
            if row['route'] != 'CORE': row['deployment_implementation'] = 'cpu_numpy_fp32_ridge_v1'
        cost = {'records': records, 'timing_producers': producers}
        mapping = audit.implementation_mapping(cost, routes)
        self.assertEqual(len(mapping), 536)
        bad = copy.deepcopy(cost); bad['timing_producers'][296]['deployment_implementation'] = 'original_cuda_fp32'
        with self.assertRaisesRegex(ValueError, 'IMPLEMENTATION'):
            audit.implementation_mapping(bad, routes)

    def test_process_wall_cap_holds_before_extra_work(self):
        current = SimpleNamespace(check=lambda: {'elapsed_seconds': 599.})
        self.assertEqual(audit.CappedLimit(current).check()['elapsed_seconds'], 599.)
        current.check = lambda: {'elapsed_seconds': 600.001}
        with self.assertRaisesRegex(audit.rt.BudgetHeld, '600SECONDS'): audit.CappedLimit(current).check()


if __name__ == '__main__': unittest.main()
