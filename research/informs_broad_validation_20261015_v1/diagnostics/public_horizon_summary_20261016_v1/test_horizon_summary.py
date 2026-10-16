import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import horizon_summary as hs


def statistic_rows(horizon=12):
    result = []
    for origin in (20, 32):
        for channel in ('PM2.5', 'B'):
            for index, (label, start, end) in enumerate(hs.bands(horizon)):
                count = end - start if channel == 'PM2.5' else (1 if origin == 20 and index == 0 else 0)
                squared, absolute = (1.0, 1.0) if channel == 'PM2.5' else (9.0, 3.0)
                result.append(dict(run_id='run', split='V_select', cohort='toy:primary:V_select:stride12',
                                   origin=str(origin), channel=channel, horizon_start=str(start),
                                   horizon_end=str(end), sse=repr(squared * count), sae=repr(absolute * count), count=str(count)))
    return result


def expected_score():
    return dict(run_id='run', split='V_select', cohort='toy:primary:V_select:stride12',
                mse=5.0, mae=2.0, n_observed=25, n_origins=2, num_channels=2, active_channels=2, horizon=12)


class AggregationTests(unittest.TestCase):
    def test_h12_h28_h48_exact_bands(self):
        for horizon, edges in ((12, (0, 3, 6, 12)), (28, (0, 7, 14, 28)), (48, (0, 12, 24, 48))):
            with self.subTest(horizon=horizon):
                self.assertEqual([(a, b) for _, a, b in hs.bands(horizon)], list(zip(edges, edges[1:])))

    def test_uneven_mask_macro_is_not_micro_and_overall_is_reconciled(self):
        groups, overall = hs.aggregate_rows(iter(statistic_rows()), expected_score())
        self.assertEqual(overall['mse'], 5.0)
        self.assertEqual(overall['mae'], 2.0)
        self.assertNotEqual(overall['mse'], 33 / 25)
        self.assertEqual([r['mse'] for r in groups], [5.0, 1.0, 1.0])
        self.assertNotEqual(sum(r['mse'] for r in groups) / 3, overall['mse'])
        self.assertEqual([r['active_channels'] for r in groups], [2, 1, 1])

    def test_all_unobserved_band_is_kept_as_na(self):
        rows = statistic_rows()
        for row in rows:
            if row['horizon_start'] == '3':
                row.update(sse='0.0', sae='0.0', count='0')
        score = dict(expected_score(), n_observed=19)
        groups, _ = hs.aggregate_rows(iter(rows), score)
        self.assertIsNone(groups[1]['mse'])
        self.assertIsNone(groups[1]['mae'])
        self.assertEqual(groups[1]['n_observed'], 0)
        self.assertEqual(groups[1]['active_channels'], 0)
        self.assertEqual(groups[1]['n_origins'], 2)

    def test_duplicate_cell_rejected(self):
        rows = statistic_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            hs.aggregate_rows(iter(rows + [rows[0]]), expected_score())

    def test_wrong_cohort_rejected(self):
        rows = statistic_rows(); rows[0]['cohort'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identity'):
            hs.aggregate_rows(iter(rows), expected_score())

    def test_missing_zero_cell_rejected(self):
        rows = statistic_rows(); rows.pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            hs.aggregate_rows(iter(rows), expected_score())

    def test_invalid_numbers_ranges_and_zero_observation_error_rejected(self):
        for change in ({'origin': '20.0'}, {'horizon_end': '4'}, {'count': '4'},
                       {'sse': 'nan'}, {'sae': '-1'}, {'count': '0', 'sse': '1'}, {'channel': ''}):
            with self.subTest(change=change):
                rows = statistic_rows(); rows[0].update(change)
                with self.assertRaises(ValueError):
                    hs.aggregate_rows(iter(rows), expected_score())

    def test_wrong_overall_score_or_count_rejected(self):
        for change in ({'mse': 4.0}, {'n_observed': 24}, {'n_origins': 3}, {'num_channels': 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                hs.aggregate_rows(iter(statistic_rows()), dict(expected_score(), **change))


class FileContractTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'predictions/run/V_select'; folder.mkdir(parents=True)
        stats = folder / 'PUBLIC_ERROR_STATS.csv.gz'
        with gzip.open(stats, 'wt', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=hs.STAT_FIELDS); writer.writeheader(); writer.writerows(statistic_rows())
        ids = {key: key + '_sha' for key in hs.IDENTITY_HASHES}
        manifest = dict(expected_score(), score=expected_score(), arrays={'pred_normalized': {'shape': [2, 12, 2]}},
                        public_stats_relative_path=stats.relative_to(root).as_posix(), **ids)
        path = folder / 'PREDICTION_MANIFEST.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
        row = dict(expected_score(), manifest_relative_path=path.relative_to(root).as_posix(), dataset='toy',
                   fold='primary', family='B0', latent_k=1, seed=1, learning_rate='0.001', max_epoch=1, selected_epoch=1, **ids)
        scores = root / 'SCORES.csv'
        with scores.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        bindings = root / 'INPUT_MANIFEST.json'
        bindings.write_text(json.dumps({'inputs': [dict(scope='public', file=p.relative_to(root).as_posix(),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in (path, stats)]}), encoding='utf-8')
        return scores, bindings, stats, path

    def test_metadata_does_not_open_or_hash_gzip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, stats, _ = self.fixture(root)
            original = hs.file_sha
            def guarded(path):
                if Path(path).suffix == '.gz':
                    self.fail('metadata mode read gzip bytes')
                return original(path)
            with mock.patch.object(hs, 'file_sha', side_effect=guarded), mock.patch.object(hs.gzip, 'open', side_effect=AssertionError('gzip opened')):
                plan = hs.metadata_plan(scores, root, bindings)
            self.assertEqual(plan['status'], 'METADATA_ONLY_NO_GZIP_READ')
            self.assertEqual(len(plan['routes']), 1)

    def test_manifest_tamper_and_root_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, _, path = self.fixture(root)
            path.write_text(path.read_text() + ' ', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                hs.metadata_plan(scores, root, bindings)
            with self.assertRaisesRegex(ValueError, 'relative'):
                hs.inside(root, '../escape')

    def test_full_synthetic_report_preserves_recipe_and_rejects_stats_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scores, bindings, stats, _ = self.fixture(root)
            output = hs.generate(scores, root, bindings)
            result = json.loads((output / 'RESULT.json').read_text())
            self.assertEqual(result['status'], 'PASS_PUBLIC_STATISTICS_ONLY')
            with (output / 'HORIZON_SCORES.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['learning_rate'], '0.001')
            self.assertEqual(rows[0]['config_sha'], 'config_sha_sha')
            payload = bytearray(stats.read_bytes()); payload[9] ^= 1
            stats.write_bytes(payload)
            bad = hs.generate(scores, root, bindings)
            self.assertEqual(json.loads((bad / 'RESULT.json').read_text())['status'], 'INCOMPLETE_FAILED_INPUTS')


if __name__ == '__main__':
    unittest.main()
