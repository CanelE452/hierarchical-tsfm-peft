import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import provided_audit_entry as entry


class CallerTests(unittest.TestCase):
    def fixture(self, root):
        source = Path(__file__).resolve().parents[1] / 'public_horizon_summary_20261016_v1/test_horizon_summary.py'
        spec = importlib.util.spec_from_file_location('horizon_fixture_for_provided_test', source)
        module = importlib.util.module_from_spec(spec)
        hs, _ = entry.dependencies()
        with mock.patch.dict('sys.modules', {'horizon_summary': hs}):
            spec.loader.exec_module(module)
        result = module.FileContractTests().fixture(root)
        (root / 'EXPERIMENT_REGISTRY.csv').write_text('experiment_id,execution_status\nrun,COMPLETE_VALID\n', encoding='utf-8')
        return result

    def test_provided_original_three_selftests_are_separate_synthetic(self):
        _, original = entry.dependencies()
        result = original.self_test()
        self.assertEqual(result['status'], 'SELF_TEST_PASS')
        self.assertEqual(len(result['tests']), 3)
        self.assertFalse(result['scientific_results_tested'])

    def test_metadata_never_opens_or_hashes_gzip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, _, _ = self.fixture(root)
            hs, original = entry.dependencies(); real_sha = hs.file_sha
            def guarded(path):
                if Path(path).suffix == '.gz':
                    self.fail('metadata read gzip bytes')
                return real_sha(path)
            with mock.patch.object(entry, 'dependencies', return_value=(hs, original)), \
                    mock.patch.object(hs, 'file_sha', side_effect=guarded), \
                    mock.patch.object(hs.gzip, 'open', side_effect=AssertionError('gzip opened')):
                plan = entry.metadata_only(scores, root, bindings)
            self.assertEqual(plan['requested_rows'], 1)
            self.assertEqual(plan['gzip_bytes_read'], 0)
            self.assertFalse((root / 'diagnostics').exists())

    def test_byte_exact_flat_copy_and_all_manifest_files_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, stats, _ = self.fixture(root)
            score_sha, stats_sha = entry.sha(scores), entry.sha(stats)
            output = entry.execute_audit(scores, root, bindings)
            result = json.loads((output / 'RESULT.json').read_text())
            ledger = json.loads((output / 'INPUT_BINDINGS.json').read_text())['flat_mapping']
            self.assertEqual(result['status'], 'PASS_PROVIDED_ORIGINAL_AUDIT')
            self.assertEqual(result['provided_audit']['checked_score_rows'], 1)
            self.assertEqual(result['provided_audit']['statistics_files'], 1)
            self.assertEqual(result['provided_audit']['verified_file_hashes'], 2)
            self.assertEqual(result['provided_audit']['manifest_status'], 'PASS')
            self.assertEqual(ledger[0]['original_sha256'], stats_sha)
            self.assertTrue(ledger[0]['copy_verified'])
            self.assertTrue(result['temporary_directory_removed'])
            self.assertEqual(entry.sha(scores), score_sha)
            self.assertEqual(entry.sha(stats), stats_sha)

    def test_stats_sha_tamper_prevents_original_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, stats, _ = self.fixture(root)
            data = bytearray(stats.read_bytes()); data[9] ^= 1; stats.write_bytes(data)
            output = entry.execute_audit(scores, root, bindings)
            result = json.loads((output / 'RESULT.json').read_text())
            self.assertEqual(result['status'], 'FAIL_PROVIDED_AUDIT_OR_COPY')
            self.assertFalse(result['provided_audit_called'])
            self.assertEqual(result['copied_statistics_files'], 0)

    def test_duplicate_scores_are_never_renamed_or_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, _, _ = self.fixture(root)
            lines = scores.read_text(encoding='utf-8').splitlines()
            scores.write_text('\n'.join(lines + [lines[1]]) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                entry.metadata_only(scores, root, bindings)

    def test_postflight_failure_preserves_original_numeric_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, _, _ = self.fixture(root)
            real = entry.source_snapshot(root)
            changed = dict(real, registry_sha256='different')
            with mock.patch.object(entry, 'source_snapshot', side_effect=[real, changed]):
                output = entry.execute_audit(scores, root, bindings)
            result = json.loads((output / 'RESULT.json').read_text())
            self.assertEqual(result['status'], 'INCOMPLETE_POSTFLIGHT')
            self.assertEqual(result['provided_audit']['status'], 'PASS')
            self.assertEqual(result['copied_statistics_files'], 1)
            self.assertFalse(result['postflight_unchanged'])

    def test_original_cauchy_bound_rejection_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, stats, _ = self.fixture(root)
            with gzip.open(stats, 'rt', encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
            rows[0]['sae'] = '1000'
            with gzip.open(stats, 'wt', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            declared = json.loads(bindings.read_text())
            record = next(x for x in declared['inputs'] if x['file'].endswith('.gz'))
            record.update(sha256=entry.sha(stats), bytes=stats.stat().st_size)
            bindings.write_text(json.dumps(declared), encoding='utf-8')
            output = entry.execute_audit(scores, root, bindings)
            result = json.loads((output / 'RESULT.json').read_text())
            self.assertTrue(result['provided_audit_called'])
            self.assertEqual(result['status'], 'FAIL_PROVIDED_AUDIT_OR_COPY')
            self.assertIn('Cauchy', result['failure']['reason'])


if __name__ == '__main__':
    unittest.main()
