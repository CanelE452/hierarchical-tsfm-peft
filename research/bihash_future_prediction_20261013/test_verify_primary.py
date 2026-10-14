"""Independent numerical and negative-control tests for the primary auditor."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import verify_primary as vp


class NumericContractTests(unittest.TestCase):
    def test_macro_counts_missing_channels_and_physical_scaling(self):
        p = np.array([[[1., 2., 99.], [3., 4., 99.]]], np.float32)
        y = np.zeros_like(p)
        mask = np.array([[[True, True, False], [True, False, False]]])
        y[~mask] = np.nan
        result = vp.forecast_score(p, y, mask, np.array([10., 20., 30.], np.float32),
                                   np.array([2., 3., 4.], np.float32))
        self.assertEqual(result['normalized']['observed_counts'], [2, 1, 0])
        self.assertEqual(result['normalized']['mse'], 4.5)
        self.assertEqual(result['normalized']['mae'], 2.)
        self.assertEqual(result['mse'], 28.)
        self.assertEqual(result['mae'], 5.)
        self.assertNotEqual(result['normalized']['mse'], (1 + 9 + 4) / 3)

    def test_nonfinite_prediction_or_no_observation_rejected(self):
        p = np.zeros((1, 2, 1), np.float32)
        with self.assertRaisesRegex(ValueError, 'NO_OBSERVED'):
            vp.macro_score(p, p, np.zeros_like(p, bool))
        p[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            vp.macro_score(p, np.zeros_like(p), np.ones_like(p, bool))

    def test_fit_float64_statistics_are_layout_independent_and_ignore_tune(self):
        raw = np.array([[1., 10.], [3., 14.], [100., 999.]], np.float32)
        mask = np.isfinite(raw)
        c = vp.fit_statistics(np.ascontiguousarray(raw), mask, (0, 2))
        f = vp.fit_statistics(np.asfortranarray(raw), mask, (0, 2))
        for actual, expected in zip(c, (np.array([2., 12.]), np.array([1., 2.]))):
            np.testing.assert_array_equal(actual, expected)
        for a, b in zip(c, f):
            np.testing.assert_array_equal(a, b)

    def test_constant_std_threshold_is_applied_before_float32_rounding(self):
        delta = 1e-6 + 1e-14
        raw = np.array([[-delta], [delta]], np.float64)
        _, std = vp.fit_statistics(raw, np.ones_like(raw, bool), (0, 2))
        self.assertLess(float(std[0]), 1e-5)

    def test_history_reset_and_future_mask_are_separate(self):
        physical = np.array([[np.nan, 6.], [4., np.nan], [100., 8.], [np.nan, 10.]], np.float32)
        observed = np.isfinite(physical)
        mean, std = np.array([2., 4.], np.float32), np.array([2., 2.], np.float32)
        history, target, hm, tm = vp.raw_windows(physical, observed, [2, 3], mean, std, length=2, horizon=1)
        np.testing.assert_array_equal(history[0], [[0., 1.], [1., 1.]])
        # A new window starts at the FIT mean; it cannot inherit the preceding 6.
        np.testing.assert_array_equal(history[1], [[1., 0.], [49., 2.]])
        np.testing.assert_array_equal(target[:, 0], [[49., 2.], [0., 3.]])
        self.assertFalse(tm[1, 0, 0])
        self.assertFalse(hm[1, 0, 1])

    def test_temporal_eligibility_exact_boundary(self):
        actual = vp.memory_eligibility([10], [2, 3, 7], length=3, horizon=4)
        np.testing.assert_array_equal(actual[0], [[True, True, False], [True, True, False], [True, True, False]])
        boundary = vp.memory_eligibility([10], [4, 5], length=3, horizon=4)
        np.testing.assert_array_equal(boundary[0], [[True, False], [True, True], [True, True]])

    def test_numpy_attention_masks_every_row_and_no_batch_mixing(self):
        q = np.array([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]])
        k = np.array([[1., 0.], [0., 1.]])
        eligible = np.array([[[True, False], [False, False]], [[True, True], [False, True]]])
        a = vp.numpy_attention(q, k, eligible)
        self.assertEqual(a[0, 0, 0], 1.)
        np.testing.assert_array_equal(a[0, 1], [0., 0.])
        np.testing.assert_allclose(a[1, 0].sum(), 1.)
        np.testing.assert_array_equal(a[:1], vp.numpy_attention(q[:1], k, eligible[:1]))

    def test_deployment_parity_is_elementwise_and_separate_from_timing(self):
        science = np.zeros((1, 48, 2), np.float32)
        mean, std = np.array([0., 1000.], np.float32), np.ones(2, np.float32)
        actual = science * std + mean
        actual[0, 0, 0] = .003
        result = vp.deployment_parity(actual, science, mean, std)
        self.assertFalse(result['physical_parity_pass'])
        self.assertEqual(result['failed_physical_cells'], 1)
        actual[0, 0, 0] = .001
        self.assertTrue(vp.deployment_parity(actual, science, mean, std)['physical_parity_pass'])

    def test_batch_reference_preserves_original_bulk_physical_failure(self):
        bulk = np.zeros((1, 48, 1), np.float32)
        matched = np.full_like(bulk, 1e-5)
        mean, std = np.zeros(1, np.float32), np.full(1, 10000., np.float32)
        actual = matched * std + mean
        result = vp.batch_reference_comparisons(actual, matched, bulk, mean, std)
        self.assertTrue(result['critical_parity_gates_pass'])
        self.assertTrue(result['actual_to_matched_physical']['physical_parity_pass'])
        self.assertFalse(result['actual_to_original_bulk_physical']['physical_parity_pass'])
        self.assertEqual(result['actual_to_original_bulk_physical']['failed_physical_cells'], 48)
        self.assertFalse(result['original_bulk_physical_failure_relabelled'])
        matched += .001
        self.assertFalse(vp.batch_reference_comparisons(actual, matched, bulk, mean, std)['critical_parity_gates_pass'])

    def test_timed_physical_parity_retains_fp32_arithmetic_and_failed_coordinates(self):
        normalized = np.full((1, 48, 2), .1234567, np.float32)
        mean, std = np.array([0., 1e8], np.float32), np.array([1., 3.], np.float32)
        actual = normalized * std + mean
        actual[0, 7, 0] += .003
        result = vp.timed_physical_parity(actual, normalized, mean, std)
        self.assertFalse(result['parity_pass'])
        self.assertEqual(result['failed_cell_count'], 1)
        cell = result['failed_cells'][0]
        self.assertEqual((cell['window_index'], cell['horizon_1based'], cell['channel_index']), (0, 8, 0))
        self.assertEqual(cell['absolute_gap'], float(np.abs(actual - (normalized * std + mean))[0, 7, 0]))
        self.assertEqual(cell['allowed_gap'], float((.002 + 2e-5 * np.abs(normalized * std + mean))[0, 7, 0]))
        self.assertEqual(result['normalized_max_abs_vs_science'], float(np.abs((actual - mean) / std - normalized).max()))
        with self.assertRaisesRegex(ValueError, 'INVALID_TIMED_OUTPUT_SHAPE'):
            vp.timed_physical_parity(actual[:, :1], normalized[:, :1], mean, std)
        with self.assertRaisesRegex(ValueError, 'INVALID_TIMED_OUTPUT_OR_SCALE'):
            vp.timed_physical_parity(actual, normalized, mean, np.zeros_like(std))

    def test_backend_contract_is_uniform_for_all_ridge_routes(self):
        expected = {('d', 'core', 'CORE', 1), ('d', 'ridge', 'RAW_JOINT_RIDGE', 1)}
        rows = [{'dataset': d, 'identity': i, 'route': r, 'batch': b,
                 'deployment_implementation': vp.CPU_RIDGE_IMPLEMENTATION if r in vp.RIDGE_ROUTES else vp.CUDA_IMPLEMENTATION}
                for d, i, r, b in sorted(expected)]
        producers = [{'key': [r[k] for k in ('dataset', 'identity', 'route', 'batch')],
                      'deployment_implementation': r['deployment_implementation']} for r in rows]
        self.assertTrue(vp.cost_backend_contract(rows, producers, expected, expected_counts=(1, 1))['passed'])
        producers[1]['deployment_implementation'] = vp.CUDA_IMPLEMENTATION
        self.assertFalse(vp.cost_backend_contract(rows, producers, expected, expected_counts=(1, 1))['passed'])
        producers.append(copy.deepcopy(producers[0]))
        self.assertFalse(vp.cost_backend_contract(rows, producers, expected, expected_counts=(1, 1))['passed'])

    def test_canonical_timing_lineage_keeps_original_objects_and_source_indexes(self):
        core = {'dataset': 'd', 'identity': 'core', 'route': 'CORE', 'batch': 1, 'median_seconds': .2}
        failed = {'dataset': 'd', 'identity': 'ridge', 'route': 'RAW_JOINT_RIDGE', 'batch': 1, 'parity_pass': False}
        ridge = {**failed, 'parity_pass': True, 'median_seconds': .1, 'deployment_implementation': vp.CPU_RIDGE_IMPLEMENTATION}
        cuda_ref, cpu_ref = {'path': 'cuda.json', 'sha256': 'c'*64}, {'path': 'cpu.json', 'sha256': 'a'*64}
        cost = {'records': [core, ridge], 'producers': [{'source_result': cuda_ref, 'original_status': 'FAILED_PRESERVED'},
                                                     {'source_result': cpu_ref, 'original_status': 'COMPLETE'}],
                'timing_producers': [{'key': ['d', 'core', 'CORE', 1], 'deployment_implementation': vp.CUDA_IMPLEMENTATION,
                    'canonical_selected_from_producer': cuda_ref, 'source_row_index': 0},
                    {'key': ['d', 'ridge', 'RAW_JOINT_RIDGE', 1], 'deployment_implementation': vp.CPU_RIDGE_IMPLEMENTATION,
                     'source_result': cpu_ref, 'source_row_index': 0}]}
        bodies = {'cuda.json': {'status':'FAILED_PRESERVED','records':[core, failed]}, 'cpu.json':{'status':'COMPLETE','records':[ridge]}}
        self.assertTrue(vp.canonical_row_lineage(cost, bodies)['passed'])
        wrong = copy.deepcopy(cost); wrong['records'][1]['median_seconds'] = .09
        self.assertFalse(vp.canonical_row_lineage(wrong, bodies)['passed'])
        wrong = copy.deepcopy(cost); wrong['timing_producers'][1]['source_result'] = cuda_ref
        self.assertFalse(vp.canonical_row_lineage(wrong, bodies)['passed'])

    def test_proof_runtime_adjustment_rejects_math_or_gate_change(self):
        original = "if accounting['elapsed_seconds'] > 300:\n    raise Exception('DEPLOYMENT_OUTPUT_GPU_PROCESS_WALL_300SECONDS')\nfolder = 'deployment_output_audit_batch_v2'\n    output = {\n        'physical_atol': .002}\n"
        changed = original.replace('> 300', '> 600').replace('_300SECONDS', '_600SECONDS').replace('batch_v2', 'batch_v3')
        changed = changed.replace("    output = {\n", "    output = {\n        'runtime_wall_cap_seconds': 600,\n        'runtime_cap_adjustment': rt._artifact(rt.HERE / 'DEPLOYMENT_PROOF_WALL_CAP_ADJUSTMENT_v1.json'),\n")
        self.assertTrue(vp.proof_runtime_only_source_change(original, changed))
        self.assertFalse(vp.proof_runtime_only_source_change(original, changed.replace("'physical_atol': .002", "'physical_atol': .003")))
        self.assertFalse(vp.proof_runtime_only_source_change(original, changed.replace('> 600', '> 900')))

    def test_MSE_difference_bound_uses_observed_channel_macro_weights(self):
        first = np.array([[[1., 2.], [3., 4.]]])
        second = np.zeros_like(first); target = np.zeros_like(first)
        mask = np.array([[[True, True], [True, False]]])
        self.assertEqual(vp.mse_difference_bound(first, second, target, mask), 4.5)
        second[:] = 1.
        delta = abs(vp.macro_score(first, target, mask)['mse'] - vp.macro_score(second, target, mask)['mse'])
        self.assertGreaterEqual(vp.mse_difference_bound(first, second, target, mask), delta)

    def test_checkpoint_tune_earliest_tie_and_duplicate_rejection(self):
        rows = [{'update': 200, 'tune': {'normalized': {'mse': .5}}},
                {'update': 0, 'tune': {'normalized': {'mse': 1.}}},
                {'update': 100, 'tune': {'normalized': {'mse': .5}}}]
        self.assertEqual(vp.checkpoint_choice(rows), 100)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            vp.checkpoint_choice(rows + [copy.deepcopy(rows[0])])

    def test_batch_disjointness_rejects_duplicates_and_horizon_overlap(self):
        origins = np.array([100, 148, 196, 197])
        self.assertEqual(vp.batch_contract(origins, [[0, 1, 2]], expected_updates=1, batch_size=3)['minimum_future_gap'], 48)
        with self.assertRaisesRegex(ValueError, 'OVERLAP'):
            vp.batch_contract(origins, [[0, 2, 3]], expected_updates=1, batch_size=3)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            vp.batch_contract(origins, [[0, 0, 2]], expected_updates=1, batch_size=3)


class ArtifactGateTests(unittest.TestCase):
    def fresh_lineage_fixture(self, root):
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {'path': str(path), 'sha256': vp.file_hash(path), 'bytes': path.stat().st_size}
        source_names = ['research/bihash_future_prediction_20261013/run.py'] + ['source' + str(i) + '.py' for i in range(15)]
        hashes = {}
        for name in source_names:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable fixture ' + name)
            hashes[name] = vp.file_hash(path)
        lock = {'source_hashes': hashes, 'weights': {m: 0. for m in vp.PRIMARY_METHODS}}
        lock_record = save(root / 'METHOD_LOCK.json', lock)
        failure = {'stage': 'data_prepare', 'dataset': vp.FRESH_DATASETS[0], 'seed': None,
                   'label': vp.FRESH_DATASETS[0], 'error': 'raw missing URI spelling'}
        fresh = {'status': 'COMPLETE', 'stage': 'fresh', 'source_hashes': hashes, 'method_lock': lock_record,
                 'cases': [], 'future_pretrains': [], 'contracts': {}, 'data': {}, 'failed_paths': [], 'producers': [],
                 'recovered_failed_paths': [], 'protected_TEST_opened': False, 'EVAL_future_target_opened': False,
                 'new_training_updates_in_merge': 0, 'merge_GPU_work': False, 'remaining_unresolved_failures': 0,
                 'original_failed_attempt_changed': False}
        fresh['producer_merge_script'] = save(root / 'MERGER.py', {'fixture': 'read-only merger'})
        for dataset, tag, status in ((vp.FRESH_DATASETS[1], 'original_v1', 'FAILED_PRESERVED'), (vp.FRESH_DATASETS[0], 'house_v1', 'COMPLETE')):
            snapshot = root / '.cache' / vp.HERE.name / 'source_snapshots' / tag
            for name in source_names:
                destination = snapshot / name; destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((root / name).read_bytes())
            snapshot_record = save(snapshot / 'MANIFEST.json', {'tag': tag, 'sha256': hashes})
            resource = save(root / 'resources' / (tag + '.json'), {'status': status, 'gpu': True, 'tag': tag, 'wall_seconds': 2.})
            producer = {'stage': 'fresh', 'tag': tag, 'status': status, 'source_hashes': hashes,
                'method_lock': {k: lock_record[k] for k in ('path', 'sha256')}, 'resource_receipt': resource,
                'contracts': {dataset: {'dataset': dataset}}, 'data': {dataset: {'contract': {'dataset': dataset}}},
                'failed_paths': [failure] if status == 'FAILED_PRESERVED' else [],
                'cases': [], 'future_pretrains': [], 'backbone_before_sha256': 'F' * 64, 'backbone_after_sha256': 'F' * 64}
            if status == 'COMPLETE':
                script = save(root / 'RECOVERY.py', {'fixture': 'subset wrapper'})
                copy_script = snapshot / 'RECOVERY_PRODUCER.py'; copy_script.write_bytes(Path(script['path']).read_bytes())
                helper = save(root / 'HELPER.py', {'fixture': 'sealed merge helper'})
                copy_helper = snapshot / 'RECOVERY_MERGE_HELPER.py'; copy_helper.write_bytes(Path(helper['path']).read_bytes())
                producer.update(execution_subset=[dataset], selected_datasets=[dataset], producer={
                    'script': script, 'snapshot_script': {'path': str(copy_script), 'sha256': vp.file_hash(copy_script)},
                    'helper_script': helper, 'snapshot_helper_script': {'path': str(copy_helper), 'sha256': vp.file_hash(copy_helper)}})
            else:
                script = {'path': str(snapshot / source_names[0]), 'sha256': hashes[source_names[0]]}
            for seed in vp.PRIMARY_SEEDS:
                producer['future_pretrains'].append({'dataset': dataset, 'seed': seed, 'updates': 400,
                                                     'curve': [{'update': u} for u in range(1, 401)]})
                for method in vp.PRIMARY_METHODS:
                    label = method + '_s' + str(seed)
                    cpbase = root / '.cache' / vp.HERE.name / ('run_' + tag) / dataset / label
                    case = {'dataset': dataset, 'seed': seed, 'method': method, 'weight': 0., 'label': label,
                            'training': {'updates': 600, 'curve': [{'update': u} for u in range(1, 601)],
                                         'checkpoints': [{'update': u, 'path': str(cpbase / ('u' + str(u) + '.pt'))} for u in (0, 100, 200, 400, 600)]}}
                    producer['cases'].append(case)
            result_path = root / ('run_' + tag) / 'RESULT.json'
            result_record = save(result_path, producer)
            meta = {'result': result_record, 'original_status': status, 'tag': tag,
                    'source_snapshot_manifest': snapshot_record, 'producer_script': script, 'resource_receipt': resource,
                    'valid_case_count': 36, 'valid_future_AE_count': 3, 'datasets': [dataset],
                    'original_failed_paths': copy.deepcopy(producer['failed_paths'])}
            fresh['producers'].append(meta)
            fresh['contracts'].update(producer['contracts']); fresh['data'].update(producer['data'])
            for case in producer['cases']:
                original = save(result_path.parent / dataset / (case['label'] + '.json'), case)
                copied = copy.deepcopy(case); copied['producer_lineage'] = {'producer_result': result_record,
                    'original_case_json': original, 'original_status': status, 'producer_tag': tag, 'validated_actual_checkpoints': 5}
                fresh['cases'].append(copied)
            for ae in producer['future_pretrains']:
                copied = copy.deepcopy(ae); copied['producer_lineage'] = {'producer_result': result_record, 'original_status': status, 'producer_tag': tag}
                fresh['future_pretrains'].append(copied)
            if status == 'FAILED_PRESERVED':
                fresh['recovered_failed_paths'].append({**failure, 'producer_result': result_record})
        fresh.update(backbone_before_sha256='F' * 64, backbone_after_sha256='F' * 64,
                     original_gpu_resource_receipts=[p['resource_receipt'] for p in fresh['producers']],
                     original_gpu_wall_seconds_counted_once=4.)
        return fresh, lock

    def test_fresh_recovery_lineage_preserves_failure_and_resource_without_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fresh, lock = self.fresh_lineage_fixture(root)
            audit = vp.Audit(); summary = vp.audit_fresh_lineage(audit, fresh, lock, root=root)
            self.assertEqual(audit.close()['failed_count'], 0)
            self.assertEqual(summary['producer_case_count'], 72)
            self.assertEqual(summary['original_GPU_receipts_counted_once'], 2)
            for mutation in ('drop_failure', 'forged_lineage', 'duplicate_case', 'double_gpu_time'):
                bad = copy.deepcopy(fresh)
                if mutation == 'drop_failure': bad['recovered_failed_paths'] = []
                elif mutation == 'forged_lineage': bad['cases'][0]['producer_lineage']['producer_tag'] = 'other'
                elif mutation == 'duplicate_case': bad['cases'][-1] = copy.deepcopy(bad['cases'][0])
                else: bad['original_gpu_wall_seconds_counted_once'] = 8.
                audit = vp.Audit(); vp.audit_fresh_lineage(audit, bad, lock, root=root)
                self.assertGreater(audit.close()['failed_count'], 0, mutation)

    def test_fresh_recovery_rejects_changed_producer_source_and_unexplained_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fresh, lock = self.fresh_lineage_fixture(root)
            producer_path = Path(fresh['producers'][0]['result']['path'])
            producer = json.loads(producer_path.read_text())
            producer['failed_paths'].append({'stage': 'train', 'dataset': vp.FRESH_DATASETS[1], 'seed': 92601, 'label': 'NO_ALIGN'})
            producer_path.write_text(json.dumps(producer))
            source = root / 'source0.py'; source.write_text('# modified science')
            (root / 'HELPER.py').write_text('# modified wrapper helper')
            audit = vp.Audit(); vp.audit_fresh_lineage(audit, fresh, lock, root=root)
            failed_names = {c['name'] for c in audit.close()['checks'] if not c['passed']}
            self.assertTrue(any('only_expected_Household_data_prepare_failure' in n for n in failed_names))
            self.assertTrue(any('current_SHA:source0.py' in n for n in failed_names))
            self.assertTrue(any('producer_result:sha256' in n for n in failed_names))
            self.assertTrue(any('subset_helper_same_current_snapshot_SHA' in n for n in failed_names))

    def test_independent_fresh_ingestion_boundaries_missing_masks_and_duplicate_rules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
            source = root / 'household_power_consumption.txt'
            lines = ['Date;Time;' + ';'.join(columns)]
            for offset in range(121):
                timestamp = np.datetime64('2020-01-01T00:20') + np.timedelta64(offset, 'm')
                day, clock = str(timestamp).split('T')
                yyyy, mm, dd = day.split('-')
                values = ['2'] * 7
                if offset == 45:
                    values[0] = '?'
                lines.append(f'{dd}/{mm}/{yyyy};{clock}:00;' + ';'.join(values))
            source.write_text('\n'.join(lines), encoding='utf-8')
            contract = {'dataset': 'uci235_household_hourly', 'columns': columns,
                'source_records': [{'path': str(source), 'sha256': vp.file_hash(source)}],
                'parser_metadata': {'hourly_rows': 1, 'raw_minute_rows': 121, 'structural_boundary_hours_excluded': 2}}
            audit = vp.Audit(); values, mask, times = vp.raw_fresh_source(audit, contract)
            np.testing.assert_array_equal(values, np.full((1, 7), 2., np.float32))
            self.assertTrue(mask.all()); self.assertEqual(times[0], '2020-01-01T02:00:00')
            self.assertEqual(audit.close()['failed_count'], 0)
            files = []
            for name, offsets in (('NEW-DATA-1.T15.txt', (0, 15)), ('NEW-DATA-2.T15.txt', (15, 45))):
                path = root / name
                lines = ['# synthetic 24-column source']
                for offset in offsets:
                    clock = f'00:{offset:02d}'
                    lines.append('2020-01-01 ' + clock + ' ' + ' '.join([str(offset)] * 21) + ' 0')
                path.write_text('\n'.join(lines), encoding='utf-8')
                files.append({'path': str(path), 'sha256': vp.file_hash(path)})
            contract = {'dataset': 'uci274_sml2010', 'columns': ['sensor' + str(c) for c in range(21)],
                'source_records': files, 'parser_metadata': {'identical_duplicates_collapsed': 1, 'full_grid_rows': 4}}
            audit = vp.Audit(); values, mask, times = vp.raw_fresh_source(audit, contract)
            self.assertEqual(values.shape, (4, 21)); self.assertFalse(mask[2].any())
            self.assertTrue(np.isnan(values[2]).all()); self.assertEqual(times[-1], '2020-01-01T00:45:00Z')
            self.assertEqual(audit.close()['failed_count'], 0)

    def test_stored_checkpoint_adapter_recomputes_selection_and_finds_false_choice(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mean, std = np.zeros(1, np.float32), np.ones(1, np.float32)
            target = np.zeros((1, 2, 1), np.float32)
            observed = np.ones_like(target, bool)
            origins = np.array([100], np.int64)
            checkpoints = []
            states = {}
            for update, value in ((0, 2.), (100, 1.), (200, .5), (400, .5), (600, .75)):
                state = {'student_hash.weight': torch.tensor([float(update)]),
                         'future_hash.weight': torch.tensor([float(update)])}
                states[update] = state
                cp = root / f'u{update}.pt'
                torch.save({'model': state, 'method': 'B_GUMBEL', 'seed': 1, 'weight': 0., 'update': update,
                            'optimizer': {'state': {0: {'step': torch.tensor(float(update))}}}}, cp)
                p = np.full_like(target, value)
                npz = root / f'u{update}.npz'
                np.savez(npz, tune_prediction=p, tune_target=target, tune_observed=observed,
                         tune_origins=origins, mean=mean, std=std)
                checkpoints.append({'update': update, 'path': str(cp), 'sha256': vp.file_hash(cp),
                    'tune_prediction': {'path': str(npz), 'sha256': vp.file_hash(npz)},
                    'tune': vp.forecast_score(p, target, observed, mean, std)})
            p = np.full_like(target, .5)
            selected = root / 'SELECTED.npz'
            arrays = {'mean': mean, 'std': std}
            for split in ('fit', 'tune'):
                arrays.update({split + '_prediction': p, split + '_target': target,
                               split + '_observed': observed, split + '_origins': origins})
            np.savez(selected, **arrays)
            case = {'dataset': 'fixture', 'label': 'B_GUMBEL_s1', 'method': 'B_GUMBEL', 'seed': 1,
                'weight': 0., 'selected_tune_mse': .25,
                'training': {'updates': 600, 'curve': [{'update': n} for n in range(1, 601)],
                    'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 0., 'codec_clip': 1., 'side_clip': 1.,
                    'checkpoints': checkpoints, 'selected_update': 200, 'selected_checkpoint': checkpoints[2],
                    'selected_hash': vp.tensor_state_hash(states[200])},
                'predictions': {'path': str(selected), 'sha256': vp.file_hash(selected)},
                'scores': {s: vp.forecast_score(p, target, observed, mean, std) for s in ('fit', 'tune')}}
            audit = vp.Audit(); summary = vp.audit_case(audit, case)
            self.assertEqual(audit.close()['failed_count'], 0)
            self.assertEqual(summary['rescore_selected_update'], 200)
            case['training']['selected_update'] = 400
            audit = vp.Audit(); vp.audit_case(audit, case)
            self.assertGreater(audit.close()['failed_count'], 0)
            case['training']['selected_update'] = 200
            case['producer_lineage'] = {'selected_TUNE_mse_rescored': .25, 'selected_CP_TUNE_selection_mse': .25,
                'selected_vs_CP_FP32_reinference': {'max_abs_array_gap': 0., 'observed_macro_mse_abs_gap': 0.,
                    'observed_macro_mse_delta_bound': 0., 'array_atol': 1e-5, 'array_rtol': 2e-6}}
            audit = vp.Audit(); vp.audit_case(audit, case)
            self.assertEqual(audit.close()['failed_count'], 0)
            case['producer_lineage']['selected_TUNE_mse_rescored'] = .26
            broken = torch.load(root / 'u600.pt', weights_only=True)
            broken['optimizer']['state'][0]['step'] = torch.tensor(700.)
            torch.save(broken, root / 'u600.pt')
            checkpoints[-1]['sha256'] = vp.file_hash(root / 'u600.pt')
            audit = vp.Audit(); vp.audit_case(audit, case)
            failed_names = {c['name'] for c in audit.close()['checks'] if not c['passed']}
            self.assertTrue(any('actual_optimizer_step_matches_update' in n for n in failed_names))
            self.assertTrue(any('lineage_selected_TUNE_mse_rescored' in n for n in failed_names))

    def test_prediction_rescore_detects_json_mismatch_and_target_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'TUNE.npz'
            p = np.array([[[1., 2.], [3., 4.]]], np.float32)
            y = np.zeros_like(p); mask = np.ones_like(p, bool)
            mean, std = np.zeros(2, np.float32), np.ones(2, np.float32)
            np.savez(path, tune_prediction=p, tune_target=y, tune_observed=mask,
                     tune_origins=np.array([100], np.int64), mean=mean, std=std)
            artifact = {'path': str(path), 'sha256': vp.file_hash(path)}
            claimed = vp.forecast_score(p, y, mask, mean, std)
            reference = {'mean': mean, 'std': std, 'origins': {'tune': np.array([100])},
                         'target': {'tune': y}, 'observed': {'tune': mask}}
            audit = vp.Audit()
            vp.audit_prediction(audit, 'fixture', artifact, {'tune': claimed}, reference)
            self.assertEqual(audit.close()['failed_count'], 0)
            wrong = copy.deepcopy(claimed); wrong['normalized']['mse'] += .1
            audit = vp.Audit()
            vp.audit_prediction(audit, 'wrong_score', artifact, {'tune': wrong}, reference)
            self.assertGreater(audit.close()['failed_count'], 0)
            reference['target']['tune'] = y + 2
            audit = vp.Audit()
            vp.audit_prediction(audit, 'wrong_target', artifact, {'tune': claimed}, reference)
            self.assertGreater(audit.close()['failed_count'], 0)

    def test_independent_method_weight_selection_uses_equal_groups_and_tune_only(self):
        rows = []
        for ds in ('a', 'b'):
            rows.append({'dataset': ds, 'seed': 1, 'method': 'NO_ALIGN', 'weight': 0., 'selected_tune_mse': 1.})
            for method, weight, value in (('B_GUMBEL', 0., .8), ('B_COSINE', 0., .9),
                                          ('A_CONTINUOUS', .03, .9), ('A_CONTINUOUS', .1, .8)):
                rows.append({'dataset': ds, 'seed': 1, 'method': method, 'weight': weight, 'selected_tune_mse': value,
                             'evaluation': {'scores': {'normalized': {'mse': 99. if weight == .1 else .01}}}})
        result = vp.method_selection(rows, expected_groups={('a', 1), ('b', 1)}, method_order=('B_COSINE', 'B_GUMBEL'))
        self.assertEqual(result['weights']['A_CONTINUOUS'], .1)
        self.assertEqual(result['selected_B_method'], 'B_GUMBEL')
        with self.assertRaisesRegex(ValueError, 'COVERAGE'):
            vp.method_selection(rows[:-1], expected_groups={('a', 1), ('b', 1)}, method_order=('B_COSINE', 'B_GUMBEL'))

    def test_eval_denied_before_any_payload_read_and_file_exclusive_serialization(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing = root / 'EVAL_DOES_NOT_EXIST.npz'
            with self.assertRaises(PermissionError):
                vp.load_prediction(missing, ('eval',), eval_authorized=False)
            out = root / 'receipt.json'
            vp.write_fresh(out, {'count': np.int64(2), 'array': np.array([1., 2.])})
            self.assertEqual(json.loads(out.read_text())['count'], 2)
            with self.assertRaises(FileExistsError):
                vp.write_fresh(out, {'different': True})
            invalid = root / 'invalid.json'
            with self.assertRaises(ValueError):
                vp.write_fresh(invalid, {'x': float('nan')})
            self.assertFalse(invalid.exists())

    def test_joint_barrier_requires_all_models_before_either_eval(self):
        expected = {('new_a', 1, 'BASE', 0.), ('new_b', 1, 'BASE', 0.)}
        rows = [{'dataset': d, 'seed': s, 'method': m, 'weight': w, 'path': d + '.pt', 'sha256': 'a' * 64}
                for d, s, m, w in sorted(expected)]
        lock = {'ALL_SELECTED': True, 'created_unix': 100., 'selected_models': rows,
                'newdata_contracts': {'new_a': {}, 'new_b': {}}}
        events = [{'dataset': 'new_a', 'opened_unix': 101., 'lock_sha256': 'b' * 64},
                  {'dataset': 'new_b', 'opened_unix': 102., 'lock_sha256': 'b' * 64}]
        self.assertTrue(vp.joint_barrier(lock, events, expected, 'b' * 64)['passed'])
        incomplete = copy.deepcopy(lock); incomplete['selected_models'].pop()
        self.assertFalse(vp.joint_barrier(incomplete, events, expected, 'b' * 64)['passed'])
        early = copy.deepcopy(events); early[0]['opened_unix'] = 99.
        self.assertFalse(vp.joint_barrier(lock, early, expected, 'b' * 64)['passed'])

    def test_joint_barrier_keeps_legacy_models_locked_without_new_legacy_first_open(self):
        expected = {(d, 1, 'BASE', 0.) for d in ('legacy_a', 'legacy_b', 'new_a', 'new_b')}
        models = [{'dataset': d, 'seed': s, 'method': m, 'weight': w} for d, s, m, w in sorted(expected)]
        lock = {'ALL_SELECTED': True, 'created_unix': 100., 'selected_models': models,
                'newdata_contracts': {'new_a': {}, 'new_b': {}}}
        events = [{'dataset': d, 'opened_unix': 101., 'lock_sha256': 'b' * 64} for d in ('new_a', 'new_b')]
        self.assertTrue(vp.joint_barrier(lock, events, expected, 'b' * 64, fresh_datasets=('new_a','new_b'))['passed'])
        incomplete = copy.deepcopy(lock); incomplete['selected_models'].pop(0)
        self.assertFalse(vp.joint_barrier(incomplete, events, expected, 'b' * 64, fresh_datasets=('new_a','new_b'))['passed'])
        self.assertFalse(vp.joint_barrier(lock, events[:1], expected, 'b' * 64, fresh_datasets=('new_a','new_b'))['passed'])
        self.assertFalse(vp.joint_barrier(lock, events + [events[0]], expected, 'b' * 64, fresh_datasets=('new_a','new_b'))['passed'])
        early = copy.deepcopy(events); early[1]['opened_unix'] = 99.
        self.assertFalse(vp.joint_barrier(lock, early, expected, 'b' * 64, fresh_datasets=('new_a','new_b'))['passed'])

    def test_exact_coverage_rejects_count_preserving_duplicate_and_missing(self):
        expected = {('a', 1, 'BASE', 0.), ('b', 1, 'BASE', 0.)}
        rows = [{'dataset': d, 'seed': s, 'method': m, 'weight': w} for d, s, m, w in sorted(expected)]
        self.assertTrue(vp.exact_coverage(rows, expected, ('dataset', 'seed', 'method', 'weight'))['passed'])
        self.assertFalse(vp.exact_coverage([rows[0], rows[0]], expected, ('dataset', 'seed', 'method', 'weight'))['passed'])
        self.assertFalse(vp.exact_coverage(rows[:1], expected, ('dataset', 'seed', 'method', 'weight'))['passed'])

    def test_final_cli_rejects_eval_audit_before_root_authorization(self):
        with self.assertRaises(PermissionError):
            vp.full_audit({'dev': 'missing.json'}, eval_authorized=False)
        with self.assertRaises(PermissionError):
            vp.audit_batch_deployment_outputs(vp.Audit(), Path('missing.json'), {}, set(), {}, eval_authorized=False)

    def test_dev_only_adapter_rejects_nonterminal_or_nondev_before_assets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'RESULT.json'
            for value in ({'stage': 'dev', 'status': 'RUNNING'}, {'stage': 'fresh', 'status': 'COMPLETE'}):
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'COMPLETED_DEV_ONLY'):
                    vp.audit_dev_result(path)

    def test_bias_diagnostic_rejects_nonterminal_or_GPU_scope_before_assets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'RESULT.json'
            for value in ({'status': 'RUNNING', 'GPU_executed': False}, {'status': 'COMPLETE', 'GPU_executed': True}):
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'COMPLETED_CPU_DIAGNOSTIC_ONLY'):
                    vp.audit_bias_diagnostic(path)

    def test_actual_model_cpu_independent_fixture(self):
        result = vp.cpu_model_fixture()
        failed = [c for c in result['checks'] if not c['passed']]
        self.assertEqual(failed, [])
        self.assertEqual(result['device'], 'cpu')
        self.assertFalse(result['generalization_verified'])
        self.assertGreater(len(result['checks']), 30)


if __name__ == '__main__':
    unittest.main()
