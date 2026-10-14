"""CPU checks for source guards and canonical past-only reconstruction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import deployment_past_inputs as dp


class PastInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(dp.rt, 'ROOT', self.root),
                        patch.object(dp.dt, 'ROOT', self.root),
                        patch.object(dp.dt, 'access_guard', return_value=[]),
                        patch.object(dp.dt, '_protected_legacy_test_paths', return_value=set())]
        for p in self.patches: p.start()
        self.addCleanup(self.temp.cleanup)
        for p in self.patches: self.addCleanup(p.stop)
        self.physical = (np.arange(1800 * 3, dtype=np.float32).reshape(1800, 3) % 71) + 100
        self.mask = np.ones_like(self.physical, dtype=bool)
        self.mask[788:800, 0] = False
        self.mask[820:823, 1] = False
        self.source_mean = np.array([100, 120, 140], np.float32)
        self.source_std = np.array([2, 0, 4], np.float32)
        guarded_std = np.where(self.source_std > 1e-6, self.source_std, 1).astype(np.float32)
        self.x = (self.physical - self.source_mean) / guarded_std
        self.origins = np.arange(1300, 1540, 24, dtype=np.int64)
        self.source = self.root / 'jena_trainval.npz'
        np.savez(self.source, x=self.x, finite=self.mask, mean=self.source_mean, std=self.source_std,
                 columns=np.array(['c0', 'c1', 'c2']), times=np.arange(1800), val_origins=self.origins)
        self.reconstructed = (self.x * guarded_std + self.source_mean).astype(np.float32)
        self.reconstructed[~self.mask] = np.nan
        self.mean, self.std = dp.dt._fit_statistics(self.reconstructed, self.mask, (0, 960))
        source_ref = self.ref(self.source, role='legacy_trainval_npz')
        contract = {'dataset': 'jena', 'trainval': source_ref, 'bounds': {'train': [0, 1200]}}
        self.contract_path = self.root / 'legacy_contract.json'
        self.contract_path.write_text(json.dumps(contract), encoding='utf-8')
        contract_ref = self.ref(self.contract_path, role='legacy_contract')
        sources = {}
        for i in range(16):
            p = self.root / ('source_' + str(i) + '.py')
            p.write_text(str(i), encoding='utf-8')
            sources[p.name] = dp.rt.sha(p)
        self.method_path = self.root / 'METHOD_LOCK.json'
        self.method_path.write_text(json.dumps({'locked': True, 'status': 'LOCKED', 'source_hashes': sources}), encoding='utf-8')
        method_ref = self.ref(self.method_path)
        self.dev = {'status': 'COMPLETE', 'source_hashes': sources, 'method_lock': method_ref,
                    'data': {'jena': {'C': 3, 'K': 1, 'source_records': [contract_ref, source_ref],
                    'contract': {'legacy_contract_sha256': contract_ref['sha256'],
                                 'legacy_contract_path': contract_ref['path'], 'bounds': {'fit': [0, 960]}},
                    'mean_sha256': dp.dt.array_hash(self.mean), 'std_sha256': dp.dt.array_hash(self.std),
                    'origin_sha256': {'val': dp.dt.array_hash(self.origins)}}}}
        self.fresh = {'status': 'COMPLETE', 'source_hashes': sources, 'method_lock': method_ref}
        global_path = self.root / 'GLOBAL_MODEL_LOCK.json'
        global_path.write_text(json.dumps({'status': 'ALL_SELECTED', 'ALL_SELECTED': True,
            'all_core_training_completed': True, 'all_readouts_and_controls_selected': True,
            'method_lock_sha256': method_ref['sha256'], 'source_hashes': sources}), encoding='utf-8')
        science = self.root / 'regular_VAL.npz'
        np.savez(science, mean=self.mean, std=self.std, val_origins=self.origins)
        self.evaluation = {'status': 'COMPLETE', 'source_hashes': sources,
            'global_model_lock': self.ref(global_path), 'core_evaluation': [{'dataset': 'jena',
            'split': 'val', 'entries': [{'variant': 'regular', 'prediction': self.ref(science)}]}]}

    def ref(self, path, **extra):
        return {'path': path.name, 'sha256': dp.rt.sha(path), **extra}

    def load(self, **kwargs):
        return dp.load_past_pack('jena', self.dev, self.fresh, self.evaluation, self.method_path, **kwargs)

    def test_exact_prefix_masks_fit_stats_and_missing_forward_fill(self):
        pack, evidence = self.load()
        expected = dp.dt._history_windows(self.reconstructed, self.mask, self.origins[:8], self.mean, self.std)
        np.testing.assert_array_equal(pack['history']['val'], expected[0])
        np.testing.assert_array_equal(pack['raw_history']['val'], expected[1])
        np.testing.assert_array_equal(pack['raw_history_observed']['val'], expected[2])
        np.testing.assert_array_equal(pack['origins']['val'], self.origins[:8])
        np.testing.assert_array_equal(pack['mean'], self.mean)
        self.assertEqual(pack['history']['val'][0, 0, 0], 0)
        self.assertEqual(evidence['source_window_count'], 10)
        self.assertTrue(evidence['source']['legacy_trainval_raw_bytes_opened'])
        self.assertFalse(evidence['target_arrays_opened'])
        self.assertEqual(set(pack['history']), {'val'})

    def test_no_target_pca_assemble_calls_and_only_allowed_npz_members(self):
        actual_load = np.load
        accessed = []
        class Tracked:
            def __init__(self, path, **kwargs): self.z = actual_load(path, **kwargs)
            def __enter__(self): return self
            def __exit__(self, *args): self.z.close()
            def __getitem__(self, key):
                accessed.append(key)
                if key not in dp.LEGACY_MEMBERS: raise AssertionError(key)
                return self.z[key]
        with patch.object(dp.np, 'load', Tracked), patch.object(dp.dt, '_target_windows', side_effect=AssertionError), \
                patch.object(dp.dt, '_assemble_tuple', side_effect=AssertionError), \
                patch.object(dp.dt, '_pca_from_history_pool', side_effect=AssertionError):
            self.load()
        self.assertEqual(accessed, list(dp.LEGACY_MEMBERS) + ['mean', 'std', 'val_origins'])

    def test_future_changes_cannot_affect_history_prefix(self):
        before, _ = self.load(windows=1)
        changed = self.reconstructed.copy()
        changed[self.origins[0]:] += 9999
        after = dp.dt._history_windows(changed, self.mask, self.origins[:1], self.mean, self.std)[0]
        np.testing.assert_array_equal(before['history']['val'], after)

    def test_source16_change_rejected_before_npz_access(self):
        self.fresh['source_hashes'] = dict(self.fresh['source_hashes'], extra='bad')
        with patch.object(dp.np, 'load', side_effect=AssertionError('payload read')), \
                self.assertRaisesRegex(ValueError, 'SOURCE16'):
            self.load()

    def test_global_lock_changed_rejected_before_npz_access(self):
        self.evaluation['global_model_lock']['sha256'] = 'bad'
        with patch.object(dp.np, 'load', side_effect=AssertionError('payload read')), \
                self.assertRaisesRegex(ValueError, 'ARTIFACT_SHA'):
            self.load()

    def test_science_stats_and_origins_guard(self):
        path = self.root / 'regular_VAL.npz'
        np.savez(path, mean=self.mean + 1, std=self.std, val_origins=self.origins)
        self.evaluation['core_evaluation'][0]['entries'][0]['prediction'] = self.ref(path)
        with self.assertRaisesRegex(ValueError, 'SCIENCE_STATISTICS_OR_ORIGINS'):
            self.load()

    def test_invalid_window_count_and_protected_file_refused(self):
        for windows in (0, 9, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'WINDOWS'):
                self.load(windows=windows)
        with self.assertRaisesRegex(PermissionError, 'TEST'):
            dp._checked_artifact({'path': 'test.npz', 'sha256': 'none'})
        with self.assertRaisesRegex(ValueError, 'TARGET_PAYLOAD'):
            dp._checked_artifact({'path': 'EVAL_future_target_payload.npz', 'sha256': 'none'})

    def test_fresh_uses_frozen_past_loader_and_trims_only_canonical_prefix(self):
        ds = 'uci274_sml2010'
        self.fresh['contracts'] = {ds: {'payloads': {'eval_past_input': {'path': 'EVAL_past_input_payload.npz'}}}}
        h, raw, mask = dp.dt._history_windows(self.reconstructed, self.mask, self.origins, self.mean, self.std)
        full = {'dataset': ds, 'C': 3, 'K': 1, 'columns': ['c0', 'c1', 'c2'], 'mean': self.mean, 'std': self.std,
                'history': {'eval': h}, 'raw_history': {'eval': raw}, 'raw_history_observed': {'eval': mask},
                'origins': {'eval': self.origins}}
        science = self.root / 'regular_EVAL.npz'
        np.savez(science, mean=self.mean, std=self.std, eval_origins=self.origins)
        self.evaluation['core_evaluation'] = [{'dataset': ds, 'split': 'eval',
             'entries': [{'variant': 'regular', 'prediction': self.ref(science)}]}]
        with patch.object(dp.bs, 'load_past_only', return_value=(full, {'accessed_members': list(dp.PAST_MEMBERS)})) as loader, \
                patch.object(dp.dt, 'open_new_evaluation', side_effect=AssertionError('future target open')):
            pack, evidence = dp.load_past_pack(ds, self.dev, self.fresh, self.evaluation, self.method_path)
        loader.assert_called_once()
        self.assertEqual(pack['history']['eval'].shape, (8, 512, 3))
        np.testing.assert_array_equal(pack['origins']['eval'], self.origins[:8])
        self.assertEqual(evidence['source_window_count'], 10)
        self.assertTrue(evidence['fresh_EVAL_past_allowed_by_existing_global_open_receipt'])


if __name__ == '__main__':
    unittest.main()
