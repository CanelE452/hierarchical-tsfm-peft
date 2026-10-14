"""Synthetic CPU/source-only checks; never open old scientific assets."""
from pathlib import Path
from contextlib import nullcontext
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_condition_probes as runner


def arrays(n=4, length=6, channels=3, horizon=2):
    random = np.random.default_rng(816)
    history = random.normal(size=(n, length, channels)).astype(np.float32)
    future = np.repeat(history[:, -1:], horizon, axis=1)
    return {'history': history, 'history_observed': np.ones_like(history, bool),
            'raw_last': history[:, -1].copy(), 'raw_forecast': future,
            'target': future.copy(), 'observed': np.ones_like(future, bool),
            'origins': np.arange(n, dtype=np.int64) + length}


class ProvenanceTests(unittest.TestCase):
    def test_terminal_budget_guard_cannot_leave_complete_result(self):
        result = {'status': 'COMPLETE'}
        receipt = {'terminal_status': 'BUDGET_HELD_PRESERVED',
                   'terminal_error': {'type': 'BudgetHeld', 'message': 'shared storage cap'}}
        runner.apply_resource_terminal(result, receipt)
        self.assertEqual(result['status'], 'BUDGET_HELD_PRESERVED')
        self.assertEqual(result['failure'], receipt['terminal_error'])

    def test_terminal_complete_does_not_erase_original_failure(self):
        result = {'status': 'FAILED_PRESERVED', 'failure': {'type': 'ValueError', 'message': 'parity'}}
        runner.apply_resource_terminal(result, {'terminal_status': 'COMPLETE', 'terminal_error': None})
        self.assertEqual(result['status'], 'FAILED_PRESERVED')
        self.assertEqual(result['failure']['message'], 'parity')

    def test_terminal_guard_keeps_original_failure_evidence(self):
        result = {'status': 'FAILED_PRESERVED', 'failure': {'type': 'ValueError', 'message': 'parity'}}
        runner.apply_resource_terminal(result, {'terminal_status': 'BUDGET_HELD_PRESERVED',
                                                'terminal_error': {'type': 'BudgetHeld', 'message': 'wall'}})
        self.assertEqual(result['status'], 'BUDGET_HELD_PRESERVED')
        self.assertEqual(result['failure']['message'], 'parity')
        self.assertEqual(result['resource_terminal_error']['message'], 'wall')

    def test_legacy_terminal_receipt_preserves_existing_status(self):
        for status in ('COMPLETE', 'FAILED_PRESERVED'):
            result = {'status': status}
            runner.apply_resource_terminal(result, {'path': 'synthetic_legacy_receipt'})
            self.assertEqual(result['status'], status)

    def test_import_and_describe_have_no_actual_execution(self):
        info = runner.describe()
        self.assertFalse(info['actual_data_opened'])
        self.assertFalse(info['gpu_executed'])
        self.assertNotIn('followup_original_engine', sys.modules)

    def test_gate_precedes_assets_and_any_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RESULT.json'
            path.write_text(json.dumps({'status': 'FAILED_PRESERVED'}), encoding='utf-8')
            with patch.object(runner, 'load_checked_npz', side_effect=AssertionError('NPZ before gate')):
                with self.assertRaisesRegex(ValueError, 'ACTUAL_BRIDGE_COMPLETE'):
                    runner.load_bridge(path, 'robin', lambda *args: {})

    def test_artifact_hash_tampering_rejected_before_npz_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.npz'
            np.savez(path, value=np.ones(2))
            with self.assertRaisesRegex(ValueError, 'SOURCE_FILE_HASH'):
                runner.load_checked_npz({'path': str(path), 'sha256': '0' * 64})

    def test_array_sha_detects_manifest_content_lie(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.npz'
            artifact = runner.save_npz(path, {'value': np.ones(2, np.float32)})
            artifact['arrays']['value']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'ARRAY_CONTENT'):
                runner.load_checked_npz(artifact)

    def test_matching_unobserved_nan_does_not_erase_observed_guard(self):
        values = arrays()
        values['target'][0, 0, 0] = np.nan
        values['observed'][0, 0, 0] = False
        mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
        reference = {'mean': mean, 'std': std}
        for field in ('target', 'observed', 'origins'):
            reference['fit_' + field] = values[field].copy()
        reference['fit_prediction'] = values['raw_forecast'].copy()
        runner.compare_reference(reference, {'fit': values}, mean, std)
        reference['fit_origins'] = reference['fit_origins'][::-1]
        with self.assertRaisesRegex(ValueError, 'REFERENCE_CONDITION'):
            runner.compare_reference(reference, {'fit': values}, mean, std)

    def test_cli_requires_execute_scope_and_fresh_tag(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(['--execute'])
        with self.assertRaises(SystemExit):
            runner.parse_args(['--describe', '--tag', '../overwrite'])
        parsed = runner.parse_args(['--execute', '--stage', 'D2', '--dataset', 'robin',
                                    '--seed', '92601', '--input-bridge', 'result.json', '--tag', 'd2_robin_s92601_v1'])
        self.assertEqual(parsed.seed, 92601)


class AbsoluteLatentTests(unittest.TestCase):
    def synthetic_codec(self):
        import torch
        from torch import nn
        torch.manual_seed(418)

        class Codec(nn.Module):
            def __init__(self):
                super().__init__()
                self.project = nn.Linear(3, 2)
                self.decode_linear = nn.Linear(2, 3)
                self.residual = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))

            def encode(self, x):
                return self.project(x)

            def decode(self, z):
                return self.decode_linear(z) + self.residual(z)

        codec = Codec().eval().requires_grad_(False)
        state = {
            'codec.decode.weight': codec.decode_linear.weight,
            'codec.decode.bias': codec.decode_linear.bias,
            **{'residual.' + key: value for key, value in codec.residual.state_dict().items()},
        }
        return codec, state

    def extracted(self):
        import torch
        codec, state = self.synthetic_codec()
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)

        def forecaster(z):
            return z[:, -1:].repeat(1, 2, 1) + torch.tensor([.05, -.02])

        splits = {name: arrays(n=n) for name, n in (('fit', 4), ('tune', 3), ('val', 2))}
        with torch.no_grad():
            for values in splits.values():
                x = tensor(values['history'])
                z = codec.encode(x)
                values['original_prediction'] = (x[:, -1:] + codec.decode(forecaster(z)) - codec.decode(z[:, -1:])).numpy()
        checks = []
        extracted, timings, parity = runner.extract_absolute_latents(
            splits, codec, forecaster, tensor=tensor, no_grad=torch.no_grad,
            synchronize=lambda: None, check=lambda: checks.append(1), batch_size=2)
        return codec, state, splits, extracted, timings, parity, checks

    def test_extracts_absolute_future_and_last_without_substituting_delta(self):
        codec, _, splits, extracted, timings, parity, checks = self.extracted()
        import torch
        for split, values in splits.items():
            absolute = codec.encode(torch.from_numpy(values['history'])).numpy()
            np.testing.assert_array_equal(extracted[split]['z_last'], absolute[:, -1:])
            np.testing.assert_allclose(extracted[split]['z_future'], np.repeat(absolute[:, -1:] + [.05, -.02], 2, axis=1),
                                       atol=1e-7, rtol=1e-7)
            self.assertFalse(np.allclose(extracted[split]['z_future'], extracted[split]['z_future'] - extracted[split]['z_last']))
            self.assertTrue(parity[split]['pass'])
            self.assertEqual(timings[split]['batch_size'], 2)
        self.assertGreater(len(checks), 1)

    def test_source_architecture_anchored_D_roundtrip_into_refit(self):
        from frozen_decoder_refit import run_decoder_refit
        _, state, _, extracted, _, _, _ = self.extracted()
        with tempfile.TemporaryDirectory() as directory:
            result = run_decoder_refit(Path(directory) / 'synthetic_refit', splits=extracted,
                                      codec_state=state, mean=np.zeros(3, np.float32),
                                      std=np.ones(3, np.float32), source_record={'fixture_only': True},
                                      synthetic=True, parity_atol=runner.ATOL, parity_rtol=runner.RTOL)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertFalse(result['scientific_execution'])
            self.assertEqual(set(result['refits']), {'Y', 'RAW', 'MIXED'})
            self.assertTrue(all(row['source_D_parity_pass'] and row['factored_D_parity_pass']
                                for row in result['reference_comparisons'].values()))

    def test_actual_decoder_source_record_contract_and_missing_alias(self):
        from frozen_decoder_refit import DecoderState, _validate
        random = np.random.default_rng(719)
        state = DecoderState.from_codec_state({
            'codec.decode.weight': random.normal(size=(17, 4)).astype(np.float32),
            'codec.decode.bias': np.zeros(17, np.float32),
            'residual.0.weight': random.normal(size=(32, 4)).astype(np.float32),
            'residual.0.bias': np.zeros(32, np.float32),
            'residual.2.weight': random.normal(size=(17, 32)).astype(np.float32),
            'residual.2.bias': np.zeros(17, np.float32),
        })
        splits = {}
        for split, n in (('fit', 256), ('tune', 64), ('val', 30)):
            shape = (n, 48, 17)
            splits[split] = {'z_future': np.zeros((n, 48, 4), np.float32),
                             'z_last': np.zeros((n, 1, 4), np.float32),
                             'raw_last': np.zeros((n, 17), np.float32),
                             'original_prediction': np.zeros(shape, np.float32),
                             'raw_forecast': np.zeros(shape, np.float32), 'target': np.zeros(shape, np.float32),
                             'observed': np.ones(shape, bool), 'origins': np.arange(n, dtype=np.int64)}
        # Synthetic declarations exercise the helper contract only; no file/asset reads.
        source = {'path': 'synthetic_fixture', 'sha256': '1' * 64, 'verified_sha256': '1' * 64}
        record = runner.decoder_source_record('robin', 92601, state, {'fixture_only': True},
                                               {'checkpoint': source}, source, source, [source], [], {})
        mean, std = np.zeros(17, np.float32), np.ones(17, np.float32)
        _validate(splits, mean, std, state, record, synthetic=False)
        record['source_artifacts'] = [entry for entry in record['source_artifacts'] if entry['alias'] != 'MAIN_CP']
        with self.assertRaisesRegex(ValueError, 'SOURCE_ARTIFACT_ALIASES'):
            _validate(splits, mean, std, state, record, synthetic=False)

    def test_original_codec_loader_case_CP_mismatch_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'synthetic_cp.pt'
            checkpoint.write_bytes(b'synthetic fixture, not a scientific checkpoint')
            case = Path(directory) / 'synthetic_case.json'
            case.write_text(json.dumps({'dataset': 'robin', 'seed': 92601,
                'architecture': 'ANCHOR_NONLINEAR_D', 'training': {
                    'selected_checkpoint': {'path': str(checkpoint), 'sha256': '0' * 64}}}), encoding='utf-8')
            bridge = {'declared_asset_hashes': {'checkpoints_file_hash_only': {
                'robin_92601': {'path': str(checkpoint), 'sha256': runner.sha(checkpoint)}}}}
            with patch.dict(runner.CASE_PATHS, {'robin': str(case)}):
                with patch('torch.load', side_effect=AssertionError('CP load before lineage')):
                    with self.assertRaisesRegex(ValueError, 'MAIN_SELECTED_CP_LINEAGE'):
                        runner.load_main_codec(None, bridge, 'robin', 92601)

    def test_original_codec_loader_valid_selection_restores_exact_state(self):
        import torch
        state = {'fixed_basis': torch.zeros(17, 4), 'synthetic_parameter': torch.ones(4)}
        expected_hash = runner.old_state_hash(state)
        class SyntheticCodec:
            def __init__(self, mean, basis, seed, architecture):
                self.state = None
            def load_state_dict(self, value, strict):
                self.state = value
                if strict is not True:
                    raise AssertionError('strict restore missing')
            def state_dict(self):
                return self.state
            def eval(self):
                return self
            def requires_grad_(self, value):
                if value is not False:
                    raise AssertionError('codec not frozen')
                return self
            def to(self, device):
                return self  # Synthetic contract double: never initializes CUDA.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'synthetic_cp.pt'
            checkpoint.write_bytes(b'synthetic fixture')
            source_sha = runner.sha(checkpoint)
            case = Path(directory) / 'synthetic_case.json'
            case.write_text(json.dumps({'dataset': 'robin', 'seed': 92601, 'K': 4,
                'architecture': 'ANCHOR_NONLINEAR_D', 'training': {'batch_size': 4,
                'selected_update': 100, 'selected_codec_hash': expected_hash,
                'selected_checkpoint': {'path': str(checkpoint), 'sha256': source_sha}}}), encoding='utf-8')
            bridge = {'declared_asset_hashes': {'checkpoints_file_hash_only': {
                'robin_92601': {'path': str(checkpoint), 'sha256': source_sha}}}}
            payload = {'codec': state, 'seed': 92601, 'architecture': 'ANCHOR_NONLINEAR_D',
                       'update': 100, 'method': 'MAIN', 'weight': 0.}
            with patch.dict(runner.CASE_PATHS, {'robin': str(case)}), \
                    patch.dict(runner.EXPECTED_CODECS, {('robin', 92601): (100, expected_hash)}), \
                    patch('torch.load', return_value=payload):
                codec, loaded, record = runner.load_main_codec(SimpleNamespace(Codec=SyntheticCodec), bridge, 'robin', 92601)
                self.assertEqual(runner.old_state_hash(codec), expected_hash)
                self.assertIs(loaded, state)
                self.assertEqual(record['training_batch_size'], 4)
                self.assertEqual(record['original_prediction_batch_size'], 8)
                payload['architecture'] = 'DIRECT_PCA'
                with self.assertRaisesRegex(ValueError, 'MAIN_CP_METADATA'):
                    runner.load_main_codec(SimpleNamespace(Codec=SyntheticCodec), bridge, 'robin', 92601)

    def test_main_reports_nonzero_on_terminal_frozen_check_failure(self):
        with patch.object(runner, 'execute', return_value={'status': 'FAILED_PRESERVED'}):
            code = runner.main(['--execute', '--stage', 'D2', '--dataset', 'robin', '--seed', '92601',
                                '--input-bridge', 'synthetic_result.json', '--tag', 'synthetic_frozen_failure'])
            self.assertEqual(code, 1)

    def test_dual_state_hash_algorithms_differ_but_repeat(self):
        codec, _ = self.synthetic_codec()
        old, bi = runner.old_state_hash(codec), runner.bi_state_hash(codec)
        self.assertNotEqual(old, bi)
        self.assertEqual(old, runner.old_state_hash(codec))
        self.assertEqual(bi, runner.bi_state_hash(codec))

    def test_D1_all_ranks_remain_separate_from_actual_science(self):
        from projected_past_probe import run_probe
        splits = {name: arrays(n=n) for name, n in (('fit', 4), ('tune', 3), ('val', 2))}
        predictor = lambda history: np.repeat(history[:, -1:], 2, axis=1)
        with tempfile.TemporaryDirectory() as directory:
            result = run_probe(splits, predictor, np.zeros(3, np.float32), np.ones(3, np.float32),
                               Path(directory) / 'synthetic_D1', provenance={'fixture_only': True},
                               actual=False, identity_atol=runner.ATOL, identity_rtol=runner.RTOL)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertFalse(result['scientific_execution'])
            self.assertEqual(len(result['candidates']), 7)
            self.assertEqual(result['candidates'][0]['name'], 'identity_C')
            self.assertTrue(all(row['identity_parity_pass'] for row in result['candidates'][0]['splits'].values()))


if __name__ == '__main__':
    unittest.main()
