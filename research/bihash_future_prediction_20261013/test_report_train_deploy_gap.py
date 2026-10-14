"""Synthetic JSON-only integration tests; no model, arrays or real report generation."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest import mock

import make_report as mr


class TrainDeployGapReportTests(unittest.TestCase):
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
        path.write_text(json.dumps(value), encoding='utf-8')
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': mr.sha(path)}

    def gap(self, count=48, declared_status='COMPLETE'):
        source = self.file('measure_train_deploy_gap.py', 'Synthetic source fixture')
        conditions = []
        for dataset in mr.DATASETS:
            for seed in mr.SEEDS:
                for method in ('B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL', 'B_TANH'):
                    train_mse = .98 if method == 'B_TANH' else 1.02
                    conditions.append({'dataset': dataset, 'seed': seed, 'method': method,
                        'selected_checkpoint': {'path': '.cache/fixed/selected.pt', 'sha256': 'a' * 64},
                        'evidence': {'primary_update': 100, 'temperature': .875},
                        'metrics': {'deploy_fit': {'mse': 1., 'mae': .5},
                            'train_draw_mean_mse': train_mse, 'train_draw_mean_mae': .51,
                            'train_draw_fit': [{'mse': train_mse, 'mae': .51}] * 8,
                            'forecast_gap_rms': .04, 'query_code_sign_flip_fraction_eligible_rows': .2,
                            'key_code_sign_flip_fraction_used_keys': .1, 'cache_recompute_forecast_max_abs': 0.}})
        gpu = self.file('resources/gap_gpu.json', {'tag': 'gap', 'status': 'COMPLETE', 'gpu': True,
            'pid': 1, 'started_unix': 1., 'finished_unix': 77., 'wall_seconds': 76.})
        value = {'status': declared_status, 'actual_completed_conditions': count,
                 'conditions': conditions[:count], 'failed_conditions': [],
                 'script_sha256': source['sha256'], 'resource_receipt': gpu}
        reference = self.file('train_deploy_gap_v1/actual_selected_v1/RESULT.json', value)
        for name in ('ANALYSIS_KO.md', 'CONDITION_METRICS_v1.csv', 'GROUP_SUMMARY_v1.csv'):
            self.file('train_deploy_gap_v1/actual_selected_v1/' + name, 'Synthetic document fixture')
        cpu = self.file('resources/gap_numeric_cpu.json', {'tag': 'gap_numeric', 'status': 'COMPLETE', 'gpu': False,
            'pid': 2, 'started_unix': 100., 'finished_unix': 108., 'wall_seconds': 8.})
        self.file('train_deploy_gap_v1/actual_selected_v1/NUMERIC_RECHECK_v1.json',
            {'status': 'PASS', 'checks': 10, 'source_result': reference,
             'max_primary_saved_FIT_forecast_gap': 0., 'body_wall_seconds': 6., 'resource_receipt': cpu})
        return value

    def test_full_actual_schema_has48x8_and_keeps_fit_scope(self):
        self.gap()
        notes = mr.decision_notes(SimpleNamespace())
        evidence = mr.train_deploy_gap_evidence(notes)
        text = '\n'.join(mr.train_deploy_gap_text(evidence))
        self.assertEqual(evidence['coverage_status'], 'COMPLETE_ACTUAL_48X8_FIT_DIAGNOSTIC')
        self.assertEqual(evidence['actual_draws'], 384)
        self.assertEqual(len(evidence['groups']), 16)
        self.assertTrue(all(row['split'] == 'FIT_DIAGNOSTIC' for row in evidence['rows']))
        self.assertIn('heldout/EVAL', text)
        self.assertIn('별도 agent', text)
        self.assertIn('temperature(selected_update)', text)
        self.assertIn('고정 .25 inference-only tanh', text)
        self.assertEqual(text.count('```text'), 1)
        self.assertTrue(evidence['numeric_recheck_matches_result'])
        self.assertEqual(evidence['original_selected_fit_forecast_max_abs'], 0.)

    def test_missing_optional_gap_does_not_claim_measurement(self):
        notes = mr.decision_notes(SimpleNamespace())
        evidence = mr.train_deploy_gap_evidence(notes)
        text = '\n'.join(mr.train_deploy_gap_text(evidence))
        self.assertEqual(evidence['coverage_status'], 'NOT_PROVIDED')
        self.assertEqual(evidence['rows'], [])
        self.assertIn('제공되지', text)
        self.assertIn('미확인', text)

    def test_declared_complete_but_missing_conditions_stays_partial(self):
        self.gap(count=3)
        evidence = mr.train_deploy_gap_evidence(mr.decision_notes(SimpleNamespace()))
        self.assertEqual(evidence['coverage_status'], 'PARTIAL_OR_INCONSISTENT_DECLARED_COVERAGE')
        self.assertEqual(evidence['actual_conditions'], 3)
        self.assertEqual(evidence['actual_draws'], 24)
        self.assertEqual(len(evidence['missing_conditions']), 45)
        self.assertIn('미완료 조건', '\n'.join(mr.train_deploy_gap_text(evidence)))

    def test_numeric_check_of_different_result_cannot_supply_zero_gap(self):
        self.gap()
        path = self.here / 'train_deploy_gap_v1/actual_selected_v1/NUMERIC_RECHECK_v1.json'
        numeric = mr.read(path)
        numeric['source_result']['sha256'] = '0' * 64
        path.write_text(json.dumps(numeric), encoding='utf-8')
        evidence = mr.train_deploy_gap_evidence(mr.decision_notes(SimpleNamespace()))
        self.assertFalse(evidence['numeric_recheck_matches_result'])
        self.assertIsNone(evidence['original_selected_fit_forecast_max_abs'])

    def test_duplicate_condition_cannot_be_reported_as_more_evidence(self):
        self.gap(count=3)
        notes = mr.decision_notes(SimpleNamespace())
        value = notes['train_deploy_gap']['value']
        value['conditions'].append(copy.deepcopy(value['conditions'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            mr.train_deploy_gap_evidence(notes)

    def test_relative_links_exist_and_cpu_gpu_receipts_are_counted_once(self):
        value = self.gap()
        notes = mr.decision_notes(SimpleNamespace())
        note = notes['train_deploy_gap']
        report_folder = self.here / 'report_fixture'
        for key, target in note.items():
            if key.endswith('_link'):
                self.assertFalse(Path(target).is_absolute())
                self.assertTrue((report_folder / target).resolve().is_file())
        resources = mr.resources_from_inputs({'gap': value})
        self.assertEqual(sum(row['wall_seconds'] for row in resources if row['gpu']), 76.)
        audit = self.file('audit_fixture.json', {'status': 'PASS'})
        cpu = mr.cpu_accounting(resources, notes, self.root / audit['path'])
        counted = sum(row['counted_wall_seconds'] for row in cpu if row['counted_wall_seconds'] is not None)
        self.assertEqual(counted, 8.)
        self.assertFalse(any(row['source_path'].endswith('gap_gpu.json') for row in cpu if row['source_path']))
        self.assertFalse(any(row['source_path'].endswith('actual_selected_v1/RESULT.json')
                             for row in cpu if row['source_path']))


if __name__ == '__main__':
    unittest.main()
