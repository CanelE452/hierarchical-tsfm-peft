"""CPU-only recovery provenance/coverage fixtures; no experiment assets opened."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import merge_fresh_results as mf
import run_fresh_subset as subset


class FreshRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.here = cls.root / 'research/bihash_future_prediction_20261013'
        cls.cache = cls.root / '.cache/bihash_future_prediction_20261013'
        cls.here.mkdir(parents=True); cls.cache.mkdir(parents=True)
        cls.sources = {}
        for index in range(16):
            path = cls.here / ('run.py' if index == 0 else f'source{index}.py')
            path.write_text(f'# synthetic source {index}\n', encoding='utf-8')
            cls.sources[path.relative_to(cls.root).as_posix()] = mf.sha(path)
        cls.lock = cls.here / 'METHOD_LOCK.json'
        cls.lock.write_text(json.dumps({'locked': True, 'source_hashes': cls.sources, 'weights': {m: (.03 if m.startswith('A_') else 0.) for m in mf.METHODS}}), encoding='utf-8')
        lock_sha = mf.sha(cls.lock)
        cls.producers = []
        cls.payloads = {}
        for dataset, tag, status in ((mf.SML, 'fresh_v1', 'FAILED_PRESERVED'), (mf.HOUSE, 'household_recovery_v1', 'COMPLETE')):
            folder = cls.here / ('run_' + tag); folder.mkdir()
            snap = cls.cache / 'source_snapshots' / tag; snap.mkdir(parents=True)
            for relative in cls.sources:
                path = snap / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((cls.root / relative).read_bytes())
            (snap / 'MANIFEST.json').write_text(json.dumps({'tag': tag, 'sha256': cls.sources}), encoding='utf-8')
            raw = cls.cache / (dataset + '_raw.bin'); raw.write_bytes(b'synthetic raw archive')
            base = cls.cache / dataset / lock_sha; base.mkdir(parents=True)
            payload = base / 'FIT_TUNE_payload.npz'
            fit_origins = np.arange(512, 1024, 2, dtype=np.int64); tune_origins = np.arange(1080, 1208, 2, dtype=np.int64)
            np.savez(payload, physical=np.zeros((1300, 1), np.float32), mask=np.ones((1300, 1), bool),
                     mean=np.zeros(1, np.float32), std=np.ones(1, np.float32), origins_fit=fit_origins, origins_tune=tune_origins,
                     basis=np.ones((1, 1), np.float32), pca_mean=np.zeros(1, np.float32))
            cls.payloads[dataset] = payload
            eval_input = base / 'EVAL_past_input_payload.npz'; eval_input.write_bytes(b'must not be opened')
            eval_target = base / 'EVAL_future_target_payload.npz'; eval_target.write_bytes(b'must not be opened')
            contract = {'dataset': dataset, 'C': 1, 'K': 1, 'L': 512, 'H': 48,
                        'method_lock': mf.ref(cls.lock, cls.root), 'source_records': [mf.ref(raw, cls.root)],
                        'payloads': {'fit_tune': mf.ref(payload, cls.root), 'eval_past_input': mf.ref(eval_input, cls.root), 'eval_future_target': mf.ref(eval_target, cls.root)}}
            contract_path = base / 'contract.json'; contract_path.write_text(json.dumps(contract), encoding='utf-8')
            registry = cls.cache / 'contract' / (dataset + '_' + lock_sha + '.json'); registry.parent.mkdir(exist_ok=True)
            registry.write_text(json.dumps({'dataset': dataset, 'method_lock_sha256': lock_sha, 'contract_path': str(contract_path), 'contract_sha256': mf.sha(contract_path)}), encoding='utf-8')
            attempts = {'tag': tag, 'stage': 'fresh', 'status': status, 'source_hashes': cls.sources,
                        'backbone_before_sha256': 'a'*64, 'backbone_after_sha256': 'a'*64,
                        'method_lock': mf.ref(cls.lock, cls.root), 'data': {dataset: {'contract': contract}},
                        'contracts': {dataset: contract}, 'future_pretrains': [], 'cases': [], 'failed_paths': []}
            if status == 'FAILED_PRESERVED':
                attempts['failed_paths'] = [{'stage': 'data_prepare', 'dataset': mf.HOUSE, 'seed': None, 'label': mf.HOUSE,
                                           'status': 'FAILED_PATH_PRESERVED', 'error': 'synthetic URI metadata mismatch'}]
            else:
                wrapper = cls.here / 'run_fresh_subset.py'; wrapper.write_text('# synthetic subset producer\n', encoding='utf-8')
                copied = snap / 'RECOVERY_PRODUCER.py'; copied.write_bytes(wrapper.read_bytes())
                attempts['execution_subset'] = [dataset]; attempts['selected_datasets'] = [dataset]
                attempts['producer'] = {'script': mf.ref(wrapper, cls.root), 'snapshot_script': mf.ref(copied, cls.root)}
            order = np.tile(np.arange(0, 256, 32), (600, 1)).astype(np.int64)
            for seed in mf.SEEDS:
                ae_path = cls.cache / ('run_' + tag) / dataset / f'future_AE_s{seed}' / 'future_AE_u400.pt'
                ae_path.parent.mkdir(parents=True)
                ae_state = {'encoder.weight': torch.full((1, 1), float(seed % 3))}
                torch.save({'update': 400, 'seed': seed, 'future_autoencoder': ae_state, 'optimizer': {'state': {0: {'step': torch.tensor(400.)}}}}, ae_path)
                ae = {'dataset': dataset, 'seed': seed, 'updates': 400, 'batch_size': 64, 'curve': [{'update': u, 'loss': .1} for u in range(1, 401)], 'checkpoint': mf.ref(ae_path, cls.root)}
                attempts['future_pretrains'].append(ae)
                for method in mf.METHODS:
                    weight = .03 if method.startswith('A_') else 0.
                    label = f'{method}_w{weight:g}_s{seed}'
                    model_folder = cls.cache / ('run_' + tag) / dataset / label; model_folder.mkdir()
                    state = {'future_autoencoder.encoder.weight': ae_state['encoder.weight'].clone(), 'codec.weight': torch.ones((1, 1))}
                    cps = []
                    for update in mf.CHECKPOINTS:
                        cp = model_folder / f'u{update}.pt'
                        torch.save({'update': update, 'seed': seed, 'method': method, 'weight': weight, 'model': state, 'optimizer': {'state': {0: {'step': torch.tensor(float(update))}}}}, cp)
                        npz = model_folder / f'u{update}_TUNE.npz'
                        np.savez(npz, mean=np.zeros(1, np.float32), std=np.ones(1, np.float32), tune_origins=tune_origins,
                                 tune_prediction=np.zeros((64, 48, 1), np.float32), tune_target=np.zeros((64, 48, 1), np.float32), tune_observed=np.ones((64, 48, 1), bool))
                        cps.append({'update': update, **mf.ref(cp, cls.root), 'tune_prediction': mf.ref(npz, cls.root), 'tune': {'normalized': {'mse': 0.}}})
                    selected = model_folder / 'SELECTED_FIT_TUNE.npz'
                    np.savez(selected, mean=np.zeros(1, np.float32), std=np.ones(1, np.float32),
                             fit_origins=fit_origins, tune_origins=tune_origins,
                             fit_prediction=np.zeros((256, 48, 1), np.float32), fit_target=np.zeros((256, 48, 1), np.float32), fit_observed=np.ones((256, 48, 1), bool),
                             tune_prediction=np.zeros((64, 48, 1), np.float32), tune_target=np.zeros((64, 48, 1), np.float32), tune_observed=np.ones((64, 48, 1), bool))
                    case = {'dataset': dataset, 'seed': seed, 'method': method, 'weight': weight, 'label': label, 'K': 1, 'L': 512, 'H': 48,
                            'initial_hash': f'initial_{seed}', 'future_pretrain': ae['checkpoint'], 'selected_tune_mse': 0.,
                            'training': {'updates': 600, 'batch_size': 8, 'batch_indices': order.tolist(), 'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                                         'curve': [{'update': u, 'main_loss': .1} for u in range(1, 601)], 'checkpoints': cps,
                                         'selected_update': 0, 'selected_checkpoint': cps[0], 'selected_hash': mf.tensor_hash(state)},
                            'predictions': mf.ref(selected, cls.root), 'scores': {s: {'normalized': {'mse': 0.}} for s in ('fit', 'tune')}}
                    attempts['cases'].append(case)
                    case_path = folder / dataset / (label + '.json'); case_path.parent.mkdir(exist_ok=True)
                    case_path.write_text(json.dumps(case), encoding='utf-8')
            receipt = cls.here / 'resources' / (tag + '.json'); receipt.parent.mkdir(exist_ok=True)
            receipt.write_text(json.dumps({'tag': tag, 'gpu': True, 'status': status, 'wall_seconds': 3.}), encoding='utf-8')
            attempts['resource_receipt'] = mf.ref(receipt, cls.root)
            path = folder / 'RESULT.json'; path.write_text(json.dumps(attempts), encoding='utf-8')
            cls.producers.append(path)
        cls.baseline = [p.read_bytes() for p in cls.producers]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tearDown(self):
        for path, value in zip(self.producers, self.baseline): path.write_bytes(value)

    def change(self, callback):
        value = mf.read(self.producers[0]); callback(value)
        self.producers[0].write_text(json.dumps(value), encoding='utf-8')

    def build(self):
        return mf.build_merged(self.producers, self.lock, root=self.root)

    def test_valid_merge_preserves_original_failure_and_exact_scientific_coverage(self):
        result = self.build()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(len(result['cases']), 72)
        self.assertEqual(len(result['future_pretrains']), 6)
        self.assertEqual(result['validated_checkpoint_count'], 360)
        self.assertEqual(result['producers'][0]['original_status'], 'FAILED_PRESERVED')
        self.assertEqual(len(result['recovered_failed_paths']), 1)
        self.assertEqual(result['original_gpu_wall_seconds_counted_once'], 6.)
        self.assertTrue(all(c['producer_lineage']['original_case_json']['sha256'] for c in result['cases']))

    def test_duplicate_case_is_rejected(self):
        self.change(lambda v: v['cases'].append(copy.deepcopy(v['cases'][0])))
        with self.assertRaisesRegex(ValueError, 'COVERAGE'): self.build()

    def test_incomplete_curve_is_rejected(self):
        self.change(lambda v: v['cases'][0]['training']['curve'].pop())
        with self.assertRaisesRegex(ValueError, '600|CASE_JSON'): self.build()

    def test_changed_lock_reference_is_rejected(self):
        self.change(lambda v: v['method_lock'].__setitem__('sha256', '0'*64))
        with self.assertRaisesRegex(ValueError, 'LOCK'): self.build()

    def test_missing_checkpoint_is_rejected(self):
        self.change(lambda v: v['cases'][0]['training']['checkpoints'].pop())
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT|CASE_JSON'): self.build()

    def test_unexplained_failed_path_is_rejected(self):
        self.change(lambda v: v['failed_paths'].append({'stage': 'core', 'dataset': mf.SML, 'seed': 92601}))
        with self.assertRaisesRegex(ValueError, 'FAILURE'): self.build()

    def test_global_error_type_cannot_be_reclassified_as_missing_dataset(self):
        self.change(lambda v: v.__setitem__('error_type', 'BudgetHeld'))
        with self.assertRaisesRegex(ValueError, 'TOP_FAILURE'): self.build()

    def test_live_attempt_cannot_be_read_as_terminal(self):
        self.change(lambda v: v.__setitem__('status', 'RUNNING'))
        with self.assertRaisesRegex(ValueError, 'TERMINAL'): self.build()

    def test_case_provenance_cannot_silently_change(self):
        self.change(lambda v: v['cases'][0].__setitem__('initial_hash', 'changed'))
        with self.assertRaisesRegex(ValueError, 'CASE_JSON'): self.build()

    def test_source_hash_mismatch_is_rejected(self):
        self.change(lambda v: v['source_hashes'].__setitem__(next(iter(v['source_hashes'])), '1'*64))
        with self.assertRaisesRegex(ValueError, 'SOURCE'): self.build()

    def test_small_fp32_reinference_gap_with_bound_is_allowed(self):
        original = np.ones((64, 48, 1), np.float32)
        current = original + np.float32(3.5e-6)
        target = np.full_like(original, 2.)
        result = mf.check_reinference(current, original, target, np.ones_like(original, bool))
        self.assertGreater(result['max_abs_array_gap'], 0.)
        self.assertLessEqual(result['observed_macro_mse_abs_gap'], result['observed_macro_mse_delta_bound'] + 1e-12)

    def test_large_reinference_gap_is_rejected(self):
        original = np.ones((64, 48, 1), np.float32)
        current = original + np.float32(3e-5)
        with self.assertRaisesRegex(ValueError, 'REINFERENCE_ARRAY_GAP'):
            mf.check_reinference(current, original, np.zeros_like(original), np.ones_like(original, bool))

    def test_valid_merge_does_not_open_or_hash_EVAL_payload(self):
        original_sha = mf.sha
        def guarded_sha(path):
            if Path(path).name in ('EVAL_past_input_payload.npz', 'EVAL_future_target_payload.npz'):
                raise AssertionError('EVAL payload must stay unopened')
            return original_sha(path)
        with mock.patch.object(mf, 'sha', side_effect=guarded_sha):
            result = self.build()
        self.assertFalse(result['EVAL_future_target_opened'])

    def test_subset_invalid_dataset_tag_or_existing_directory_refused_before_GPU(self):
        with self.assertRaisesRegex(ValueError, 'DATASET'):
            subset.validate_request(['unknown'], 'safe', self.lock, root=self.root, expected_lock_sha=mf.sha(self.lock))
        with self.assertRaisesRegex(ValueError, 'TAG'):
            subset.validate_request([mf.HOUSE], '../unsafe', self.lock, root=self.root, expected_lock_sha=mf.sha(self.lock))
        with self.assertRaises(FileExistsError):
            subset.validate_request([mf.HOUSE], 'household_recovery_v1', self.lock, root=self.root, expected_lock_sha=mf.sha(self.lock))

    def test_complete_registered_contract_is_reused_without_recreating_payload(self):
        before = mf.sha(self.payloads[mf.HOUSE])
        contracts, metadata = subset.contracts_for_reuse([mf.HOUSE], self.lock, root=self.root)
        self.assertEqual(set(contracts), {mf.HOUSE})
        self.assertEqual(metadata[mf.HOUSE]['mode'], 'REUSE_REGISTERED_IMMUTABLE_CONTRACT')
        self.assertEqual(mf.sha(self.payloads[mf.HOUSE]), before)

    def test_partial_existing_canonical_data_is_preserved_and_refused(self):
        registry = self.cache / 'contract' / (mf.HOUSE + '_' + mf.sha(self.lock) + '.json')
        backup = registry.with_suffix('.held')
        registry.rename(backup)
        try:
            with self.assertRaisesRegex(ValueError, 'PARTIAL_CANONICAL'):
                subset.contracts_for_reuse([mf.HOUSE], self.lock, root=self.root)
            self.assertTrue(self.payloads[mf.HOUSE].is_file())
        finally:
            backup.rename(registry)

    def test_actual_checkpoint_file_missing_is_rejected(self):
        first = mf.read(self.producers[0])['cases'][0]['training']['checkpoints'][0]
        path = self.root / first['path']; backup = path.with_suffix('.held')
        path.rename(backup)
        try:
            with self.assertRaisesRegex(ValueError, 'ARTIFACT_MISSING'):
                self.build()
        finally:
            backup.rename(path)


if __name__ == '__main__': unittest.main()
