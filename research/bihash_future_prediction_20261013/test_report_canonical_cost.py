"""Synthetic CPU-only report checks; no study arrays or model execution."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import make_report as mr


class CanonicalCostReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.here = self.root / 'research/bihash_future_prediction_20261013'
        self.here.mkdir(parents=True)
        self.patch = mock.patch.multiple(mr, ROOT=self.root, HERE=self.here,
                                        CACHE=self.root / '.cache/bihash_future_prediction_20261013')
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def file(self, name, value):
        path = self.here / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            path.write_text(json.dumps(value), encoding='utf-8')
        else:
            np.savez(path, **value)
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': mr.sha(path)}

    def npz(self, name, arrays):
        path = self.here / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': mr.sha(path)}

    def arrays(self, batch=1, legacy=False):
        matched = np.zeros((batch, 48, 1), np.float32)
        bulk = matched.copy()
        bulk[..., 0] = 8e-6
        history = np.zeros((batch, 512, 1), np.float32)
        arrays = {'actual_physical': matched.copy(), 'raw_history': history,
                  'history_observed': np.ones_like(history, bool), 'origins': np.arange(batch, dtype=np.int64),
                  'mean': np.zeros(1, np.float32), 'std': np.full(1, 1000, np.float32),
                  'source_history_normalized': history, 'independent_history_normalized': history.copy()}
        if legacy:
            arrays['science_prediction'] = matched
        else:
            arrays.update(matched_science_normalized=matched, original_bulk_science_normalized=bulk,
                          all_7_physical_outputs=np.stack([matched] * 7))
        return arrays

    def record(self, arrays=None):
        return {'dataset': 'jena', 'identity': 'fixture', 'route': 'E_RAW_RIDGE', 'batch': 1,
                'actual_output': self.npz('fixture.npz', self.arrays() if arrays is None else arrays),
                'deployment_implementation': mr.CPU_RIDGE_IMPLEMENTATION}

    def test_new_batch_schema_keeps_bulk_physical_failure_and_critical_pass_separate(self):
        rows = mr.deployment_rows({'records': [self.record()]})
        row = rows[0]
        self.assertTrue(row['report_actual_to_matched_physical_parity_pass'])
        self.assertTrue(row['report_matched_to_original_bulk_normalized_parity_pass'])
        self.assertFalse(row['report_actual_to_original_bulk_physical_parity_pass'])
        self.assertTrue(row['report_recomputed_critical_parity_pass'])
        self.assertEqual(row['report_reference_kind'], 'independent_batch_matched_science')
        self.assertIn('one-call', row['report_comparison_scope'])

    def test_new_schema_requires_original_bulk_array(self):
        arrays = self.arrays()
        del arrays['original_bulk_science_normalized']
        with self.assertRaisesRegex(ValueError, 'MISSING.*BULK'):
            mr.deployment_rows({'records': [self.record(arrays)]})

    def test_legacy_science_prediction_stays_supported_with_explicit_scope(self):
        row = mr.deployment_rows({'records': [self.record(self.arrays(legacy=True))]})[0]
        self.assertTrue(row['report_actual_physical_parity_pass'])
        self.assertEqual(row['report_reference_kind'], 'legacy_saved_science_prediction')
        self.assertIsNone(row['report_actual_to_original_bulk_physical_parity_pass'])
        self.assertIsNone(row['report_matched_to_original_bulk_normalized_parity_pass'])

    def receipt(self, name, gpu=True, seconds=1.):
        return self.file('receipts/' + name + '.json', {'status': 'COMPLETE', 'tag': name, 'gpu': gpu,
                         'pid': name, 'started_unix': 1., 'finished_unix': 1. + seconds, 'wall_seconds': seconds})

    def canonical(self, all_seven=True):
        old_receipt = self.receipt('old', seconds=10.)
        recovery_receipt = self.receipt('recovery', seconds=20.)
        cpu_receipt = self.receipt('cpu_head', seconds=30.)
        merge_receipt = self.receipt('merge', gpu=False, seconds=2.)
        output_by_batch = {}
        for batch in (1, 8):
            arrays = self.arrays(batch)
            if not all_seven:
                del arrays['all_7_physical_outputs']
            output_by_batch[batch] = self.npz('timed_B' + str(batch) + '.npz', arrays)
        cuda, heads = [], []
        for implementation, count, target, route in (('original_cuda_fp32', 148, cuda, 'CORE'),
                (mr.CPU_RIDGE_IMPLEMENTATION, 120, heads, 'E_RAW_RIDGE')):
            for index in range(count):
                for batch in (1, 8):
                    row = {'dataset': 'jena', 'identity': implementation + str(index), 'route': route,
                           'split': 'val', 'batch': batch, 'warmup': 2, 'repeats': 7,
                           'seconds': [.01] * 7, 'median_seconds': .01, 'median_seconds_per_window': .01 / batch,
                           'physical_parity_atol': .002, 'physical_parity_rtol': 2e-5, 'parity_pass': True}
                    if implementation == mr.CPU_RIDGE_IMPLEMENTATION:
                        row.update(deployment_implementation=implementation, actual_output=output_by_batch[batch])
                    target.append(row)
        failed_rows = [{**copy.deepcopy(heads[i]), 'parity_pass': False} for i in range(40)]
        original = self.file('old/RESULT.json', {'status': 'FAILED_PRESERVED', 'records': cuda + heads[:108],
                                               'resource_receipt': old_receipt})
        recovery = self.file('recovery/RESULT.json', {'status': 'FAILED_PRESERVED', 'records': cuda + failed_rows + heads[40:],
                         'resource_receipt': recovery_receipt, 'failed_paths': [{'reason': 'RIDGE_PHYSICAL_PARITY', 'row': i} for i in range(40)]})
        cpu = self.file('cpu/RESULT.json', {'status': 'COMPLETE', 'records': heads, 'resource_receipt': cpu_receipt})
        records = cuda + heads
        lineage = []
        for index, row in enumerate(records):
            ridge = row['route'] in mr.RIDGE_ROUTES
            lineage.append({'key': [row[k] for k in ('dataset', 'identity', 'route', 'batch')],
                'deployment_implementation': mr.CPU_RIDGE_IMPLEMENTATION if ridge else 'original_cuda_fp32',
                'source_result': cpu if ridge else recovery, 'source_row_index': index - 296 if ridge else index,
                'source_row_object_unchanged': True, 'resource_receipt': cpu_receipt if ridge else recovery_receipt})
        return {'status': 'COMPLETE', 'stage': 'cost', 'records': records, 'timing_producers': lineage,
            'backend_contract': {'CPU_implementation_id': mr.CPU_RIDGE_IMPLEMENTATION, 'CPU_BLAS_threads': 2,
                                 'physical_atol': .002, 'physical_rtol': 2e-5},
            'producers': [{'source_result': recovery, 'original_status': 'FAILED_PRESERVED', 'selected_rows': 296},
                          {'source_result': cpu, 'original_status': 'COMPLETE', 'selected_rows': 240}],
            'original_cuda_536_failure_preserved': {'source_result': recovery, 'status': 'FAILED_PRESERVED',
                'failed_paths': [{'reason': 'RIDGE_PHYSICAL_PARITY', 'row': i} for i in range(40)], 'failed_row_count': 40,
                'original_404_failed_cost': {'source_result': original, 'status': 'FAILED_PRESERVED'},
                'status_or_objects_rewritten': False},
            'timing_resource_receipts_counted_once': [old_receipt, recovery_receipt, cpu_receipt],
            'resource_receipt': merge_receipt}

    def test_canonical_receipts_count_underlying_processes_once_and_include_proof(self):
        cost = self.canonical()
        proof_receipt = self.receipt('proof', seconds=40.)
        rows = mr.resources_from_inputs({'cost': cost, 'deployment_outputs': {'resource_receipt': proof_receipt}})
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(row['wall_seconds'] for row in rows if row['gpu']), 100.)
        self.assertEqual(sum(row['wall_seconds'] for row in rows if not row['gpu']), 2.)

    def test_canonical_provenance_preserves_source_objects_failed40_and_seven_output_scope(self):
        cost = self.canonical()
        before = copy.deepcopy(cost['records'])
        evidence = mr.canonical_cost_evidence(cost)
        self.assertEqual(cost['records'], before)
        self.assertEqual(evidence['coverage_status'], 'VERIFIED_CANONICAL_536_WITH_RECORDED_RECEIPTS')
        self.assertEqual(evidence['cpu_ridge_rows'], 240)
        self.assertEqual(evidence['cuda_rows'], 296)
        self.assertEqual(evidence['cpu_rows_with_all_seven_outputs'], 240)
        self.assertEqual(evidence['preserved_cuda_failed_rows'], 40)
        text = '\n'.join(mr.canonical_cost_text(evidence))
        for phrase in ('404', '536', '296', '240', 'FAILED_PRESERVED', 'BLAS', 'bulk', '7회'):
            self.assertIn(phrase, text)
        self.assertEqual(len(evidence['timing_rows']), 536)
        self.assertEqual(len(evidence['preserved_failures']), 40)

    def test_missing_receipt_or_incomplete_status_cannot_supply_complete_canonical_claim(self):
        cost = self.canonical()
        cost['status'] = 'FAILED_PRESERVED'
        evidence = mr.canonical_cost_evidence(cost)
        self.assertNotEqual(evidence['coverage_status'], 'VERIFIED_CANONICAL_536_WITH_RECORDED_RECEIPTS')
        cost['status'] = 'COMPLETE'
        del cost['timing_resource_receipts_counted_once']
        evidence = mr.canonical_cost_evidence(cost)
        self.assertNotEqual(evidence['coverage_status'], 'VERIFIED_CANONICAL_536_WITH_RECORDED_RECEIPTS')

    def test_modified_reused_row_is_refused(self):
        cost = self.canonical()
        cost['records'][0]['median_seconds'] = 9.
        with self.assertRaisesRegex(ValueError, 'SOURCE_ROW'):
            mr.canonical_cost_evidence(cost)

    def test_missing_seven_timed_arrays_is_unavailable_instead_of_inferred_from_one_call(self):
        evidence = mr.canonical_cost_evidence(self.canonical(all_seven=False))
        self.assertEqual(evidence['cpu_rows_with_all_seven_outputs'], 0)
        self.assertEqual(evidence['coverage_status'], 'PARTIAL_OR_INCONSISTENT_CANONICAL_EVIDENCE')
        self.assertTrue(all(row['timed_output_count'] == 0 for row in evidence['timed_output_rows']))

    def test_report_selected_B_is_separate_and_missing_actual_proof_keeps_status_incomplete(self):
        core = [{'dataset': dataset, 'seed': seed, 'split': 'eval', 'method': method,
                 'variant': 'regular', 'chosen_weight': True, 'windows': 19,
                 'norm_mse': 1., 'norm_mae': 1.}
                for dataset in mr.DATASETS for seed in mr.SEEDS
                for method in ('B_GUMBEL', 'B_FIXED_GUMBEL', 'NO_ALIGN')]
        bundle = {'core': core, 'readouts': [], 'controls': [],
                  'lock': {'selected_B_method': 'B_FIXED_GUMBEL', 'weights': {}},
                  'deployment_output_count': 0, 'deployment_output_receipt_terminal': False}
        text = mr.report_text({'cost': {'status': 'COMPLETE'}}, bundle, [], [],
                              {'status': 'PASS_WITH_DECLARED_SCOPE', 'failed_count': 0}, {},
                              [{'item': 'fixture', 'actual': 1, 'planned': 1}], {}, 'fixture-font')
        self.assertIn('실제 상태: 미완료', text)
        self.assertIn('TUNE에서 고정한 주 B 방법은 `B_FIXED_GUMBEL`', text)
        self.assertIn('사전 지정 B_GUMBEL', text)
        self.assertIn('weight .1', text)
        self.assertIn('seed당 `[19]`창', text)

    def test_old_cost_input_has_no_invented_canonical_backend(self):
        evidence = mr.canonical_cost_evidence({'status': 'FAILED_PRESERVED', 'records': []})
        self.assertEqual(evidence['coverage_status'], 'CANONICAL_BACKEND_LINEAGE_NOT_PROVIDED')
        self.assertEqual(evidence['timing_rows'], [])
        self.assertIn('미제공', '\n'.join(mr.canonical_cost_text(evidence)))

    def runtime_profile(self):
        source = self.here / 'verify_deployment_outputs_batch_v2.py'
        source.write_text('synthetic original source', encoding='utf-8')
        source_ref = {'path': source.relative_to(self.root).as_posix(), 'sha256': mr.sha(source)}
        newer = self.here / 'verify_deployment_outputs_batch_v3.py'
        newer.write_text('synthetic cap-only source', encoding='utf-8')
        receipt = self.receipt('partial402', seconds=300.956)
        receipt_path = self.root / receipt['path']
        receipt_value = mr.read(receipt_path)
        receipt_value['status'] = 'BUDGET_HELD_PRESERVED'
        receipt = self.file('receipts/partial402.json', receipt_value)
        partial = self.file('deployment_output_audit_batch_v2/outputs_batch_actual_v1/RESULT.json',
            {'status': 'BUDGET_HELD_PRESERVED', 'records': [{'critical_gates_pass': True}] * 402,
             'actual_saved_batch_outputs': 402, 'missing_batch_outputs': 134, 'failed_routes': [],
             'backbone_before_sha256': 'a' * 64, 'resource_receipt': receipt,
             'script_sha256': source_ref['sha256'], 'source_hashes': {'original16.py': 'b' * 64}})
        profile = {'status': 'RUNTIME_CAP_ADJUSTMENT_BEFORE_FRESH_RUN', 'original_cap_seconds': 300,
            'new_cap_seconds': 600, 'shared_GPU_process_wall_cap_seconds': 14400,
            'shared_campaign_wall_cap_seconds': 28800, 'math_data_model_selection_or_tolerances_changed': False,
            'preserved_partial': {**partial, 'status': 'BUDGET_HELD_PRESERVED', 'actual_outputs': 402,
                'all_402_critical_gates_pass': True, 'backbone_after_sha256_recorded': False, 'resource_receipt': receipt},
            'source_producer': source_ref, 'new_producer_path': newer.relative_to(self.root).as_posix(),
            'scientific_source16_unchanged': {'original16.py': 'b' * 64}}
        profile_ref = self.file('DEPLOYMENT_PROOF_WALL_CAP_ADJUSTMENT_v1.json', profile)
        full = {'status': 'COMPLETE', 'records': [{'critical_gates_pass': True}] * 536,
                'runtime_wall_cap_seconds': 600, 'runtime_cap_adjustment': profile_ref,
                'script_sha256': mr.sha(newer), 'source_hashes': {'original16.py': 'b' * 64}}
        return profile_ref, partial, full

    def test_runtime_partial_stays_budget_held_and_missing_F_closure_even_if_fresh_result_complete(self):
        profile, partial, full = self.runtime_profile()
        evidence = mr.deployment_runtime_evidence(SimpleNamespace(), full)
        row = evidence['rows'][0]
        self.assertEqual(row['partial_terminal_status'], 'BUDGET_HELD_PRESERVED')
        self.assertEqual(row['partial_actual_outputs'], 402)
        self.assertEqual(row['partial_missing_outputs'], 134)
        self.assertFalse(row['partial_backbone_after_sha256_recorded'])
        self.assertFalse(row['partial_backbone_hash_closure_established'])
        self.assertEqual(row['partial_GPU_process_wall_seconds'], 300.956)
        self.assertTrue(row['partial_all_saved_critical_gates_pass'])
        text = '\n'.join(mr.deployment_runtime_text(evidence))
        for phrase in ('402', 'BUDGET_HELD_PRESERVED', 'F-after', '300', '600', '합치지'):
            self.assertIn(phrase, text)
        resources = mr.resources_from_inputs({'partial': evidence['partial_result'],
                                             'same_partial_reference': evidence['partial_result'],
                                             'fresh': {'resource_receipt': self.receipt('full536', seconds=400.)}})
        self.assertEqual(len(resources), 2)
        self.assertAlmostEqual(sum(row['wall_seconds'] for row in resources), 700.956)
        self.assertEqual(sum(row['status'] == 'BUDGET_HELD_PRESERVED' for row in resources), 1)

    def test_runtime_profile_SHA_mismatch_is_refused(self):
        profile, partial, full = self.runtime_profile()
        full['runtime_cap_adjustment']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA'):
            mr.deployment_runtime_evidence(SimpleNamespace(), full)

    def test_runtime_profile_does_not_invent_missing_fresh_proof(self):
        self.runtime_profile()
        evidence = mr.deployment_runtime_evidence(SimpleNamespace(), None)
        self.assertIsNone(evidence['rows'][0]['fresh_result_terminal_status'])
        self.assertIn('미제공', '\n'.join(mr.deployment_runtime_text(evidence)))

    def test_absent_runtime_profile_is_explicitly_unavailable(self):
        evidence = mr.deployment_runtime_evidence(SimpleNamespace(), None)
        self.assertEqual(evidence['status'], 'RUNTIME_PROFILE_NOT_PROVIDED')
        self.assertEqual(evidence['rows'], [])

    def test_nested_actual_audit_gates_and_deployment_alias_are_read(self):
        audit = {'gates': {name: {'passed': True} for name in
                          ('joint_barrier', 'parameter_updates', 'causality', 'deployment_parity')}}
        for key in ('joint_barrier', 'parameter_updates', 'causality', 'deployment_output_parity'):
            self.assertTrue(mr.audit_gate(audit, key).startswith('PASS'))

    def test_explicit_gate_failure_wins_over_other_pass_and_missing_stays_unknown(self):
        self.assertEqual(mr.audit_gate({'gates': {'deployment_parity': {'passed': False}}},
                                       'deployment_output_parity'), 'FAIL')
        conflict = {'joint_barrier': {'passed': False}, 'gates': {'joint_barrier': {'passed': True}}}
        self.assertEqual(mr.audit_gate(conflict, 'joint_barrier'), 'FAIL')
        self.assertTrue(mr.audit_gate({'status': 'PASS', 'gates': {}}, 'joint_barrier').startswith('미확인'))
        self.assertTrue(mr.audit_gate({'gates': {'joint_barrier': {'passed': 'true'}}},
                                     'joint_barrier').startswith('미확인'))
        self.assertTrue(mr.audit_gate({'joint_barrier': {'passed': True}}, 'joint_barrier').startswith('PASS'))

    def test_cost_plot_joins_exact_routes_and_preserves_core3_vs_raw1_counts(self):
        core, costs = [], []
        for method in ('NO_ALIGN', 'B_GUMBEL', 'B_FIXED_GUMBEL', 'B_UNIFORM'):
            for seed in mr.SEEDS:
                label = method + str(seed)
                core.append({'dataset': 'jena', 'seed': seed, 'method': method, 'label': label,
                    'variant': 'regular', 'chosen_weight': True, 'split': 'val', 'norm_mse': 1.})
                costs.extend({'dataset': 'jena', 'identity': label, 'route': 'CORE', 'batch': batch,
                              'median_seconds_per_window': .03 if batch == 1 else 999.} for batch in (1, 8))
        controls = [{'dataset': 'jena', 'method': method, 'split': 'val', 'norm_mse': .5}
                    for method in ('RAW_BOLT', 'RAW_SELF33')]
        costs.extend({'dataset': 'jena', 'identity': method, 'route': method, 'batch': 1,
                      'median_seconds_per_window': .01} for method in ('RAW_BOLT', 'RAW_SELF33'))
        costs.append({'dataset': 'robin', 'identity': 'RAW_BOLT', 'route': 'RAW_BOLT', 'batch': 1,
                      'median_seconds_per_window': 888.})
        bundle = {'core': core, 'controls': controls, 'lock': {'selected_B_method': 'B_FIXED_GUMBEL'}}
        points = mr.accuracy_cost_points(bundle, {'records': costs})
        self.assertEqual({row['method'] for row in points},
                         {'NO_ALIGN', 'B_GUMBEL', 'B_FIXED_GUMBEL', 'RAW_BOLT', 'RAW_SELF33'})
        by_method = {row['method']: row for row in points}
        self.assertEqual(by_method['NO_ALIGN']['N'], 3)
        self.assertEqual(by_method['NO_ALIGN']['median_ms_mean'], 30.)
        self.assertEqual(by_method['RAW_BOLT']['N'], 1)
        self.assertEqual(by_method['RAW_BOLT']['median_ms_mean'], 10.)
        self.assertIsNone(by_method['RAW_BOLT']['norm_mse_seed_sd'])

    def test_full_audit_repair_keeps_original_fail_and_links_final_and_supplement(self):
        original = self.file('validation_v1/PRIMARY_FULL_AUDIT_v1.json',
                             {'status': 'FAIL', 'failed_count': 3, 'check_count': 100})
        final = self.file('validation_v1/PRIMARY_FULL_AUDIT_v2.json',
                          {'status': 'PASS_WITH_DECLARED_SCOPE', 'failed_count': 0, 'check_count': 101})
        self.file('validation_v1/PRIMARY_FULL_AUDIT_REPAIR_AND_CPU_ACCOUNTING_v1.json',
            {'status': 'PASS_WITH_LIMITATIONS', 'failed_count': 0, 'check_count': 14,
             'original_failed_audit': {**original, 'status': 'FAIL', 'failures': ['scope', 'status', 'gate']},
             'final_audit': {**final, 'status': 'PASS_WITH_DECLARED_SCOPE'},
             'science_data_models_metric_tolerances_changed': False, 'numeric_improvement_or_generalization_claim': False,
             'changes': ['metadata adapter only'], 'independent_CPU_declared_body_seconds': 10.,
             'ledger_CPU_process_seconds_already_separate': 2., 'cpu_wall_seconds': .1})
        evidence = mr.audit_repair_evidence(self.root / final['path'])
        self.assertEqual(evidence['rows'][0]['status'], 'FAIL')
        self.assertEqual(evidence['rows'][0]['failed_count'], 3)
        self.assertTrue(evidence['selected_audit_matches_final'])
        self.assertEqual(evidence['rows'][2]['check_count'], 14)
        text = '\n'.join(mr.audit_repair_text(evidence))
        for phrase in ('v1', 'v2', 'FAIL', '14', '감사 코드', '허용오차'):
            self.assertIn(phrase, text)
        self.assertEqual(len(evidence['files']), 3)


if __name__ == '__main__':
    unittest.main()
