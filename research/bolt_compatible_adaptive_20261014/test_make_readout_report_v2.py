"""Report regression against the actual first/dense JSON cohorts only.

The ROOT-approved CPU test wrapper must run this source. These checks open
terminal JSON metadata, never prediction NPZs, model weights or TEST.
"""
import copy
import unittest
from pathlib import Path

import make_readout_report_v2 as report


HERE = Path(__file__).resolve().parent


def first_fixture():
    producers, records = [], []
    for dataset in report.DATASETS:
        pool = [{'kind': 'PERSIST', 'tune_scores': {'normalized': {'mse': 1.}}},
                {'kind': 'MEAN', 'tune_scores': {'normalized': {'mse': 1.}}}]
        pool += [{'kind': 'RIDGE', 'alpha': alpha, 'tune_scores': {'normalized': {'mse': 2.}}}
                 for alpha in (0., 1e-4, .001, .01, .1, 1., 10.)]
        producers.append({'dataset': dataset, 'status': 'COMPLETE', 'protected_TEST_accessed': False,
            'independent_holdout': False, 'pool_comparisons': 27,
            'permission_gate': {'sha256': 'auth'}, 'run_manifest': {'sha256': 'clock'},
            'families': {route: {'pool': copy.deepcopy(pool), 'selected_index': 0,
                                'selected_kind': 'PERSIST', 'selected_alpha': None} for route in report.ROUTES}})
        records.append({'dataset': dataset, 'sha256': dataset})
    audit = {'status': 'PASS', 'totals': {'pool_candidates_checked': 54}, 'producer_records': copy.deepcopy(records)}
    return producers, records, audit


class ReportGateRegression(unittest.TestCase):
    def test_exact_tie_global_selection(self):
        p, r, a = first_fixture()
        self.assertTrue(report.validate_scientific(p, r, a, 'auth', 'clock'))
        p[0]['families']['SELF33'].update(selected_index=1, selected_kind='MEAN')
        with self.assertRaisesRegex(ValueError, 'EARLIEST_GLOBAL_TUNE_SELECTION'):
            report.validate_scientific(p, r, a, 'auth', 'clock')

    def test_audit_must_bind_the_exact_producer(self):
        p, r, a = first_fixture()
        a['producer_records'][0]['sha256'] = 'another-producer'
        with self.assertRaisesRegex(ValueError, 'EXACT_SHA_BINDING'):
            report.validate_scientific(p, r, a, 'auth', 'clock')

    def test_old_authorization_and_partial_cost_rejected(self):
        p, r, a = first_fixture()
        p[0]['permission_gate']['sha256'] = 'expired-auth'
        with self.assertRaisesRegex(ValueError, 'APPROVED_COMPLETE'):
            report.validate_scientific(p, r, a, 'auth', 'clock')
        audit = {'status': 'PASS', 'cost_producer_record': {'sha256': 'cost'},
                 'rows_checked': 24, 'timed_repeats_checked': 143, 'source_before_after_SHA_pass': True}
        with self.assertRaisesRegex(ValueError, '144_REPEAT'):
            report.validate_independent_cost(audit, {'sha256': 'cost'})


class ActualCohortRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = [HERE/'runs/first_robin_v1/RESULT.json', HERE/'runs/first_beijing_v1/RESULT.json']
        cls.first = [report.read(path) for path in cls.paths]
        cls.dense_path = HERE/'runs/expanded_robin_v2/RESULT.json'
        cls.dense = report.read(cls.dense_path)
        cls.cost_path = HERE/'cost_runs/first_readout24_v2/RESULT.json'
        cls.cost = report.read(cls.cost_path)
        cls.dense_cost_path = HERE/'cost_runs/expanded_robin4_v1/RESULT.json'
        cls.dense_cost = report.read(cls.dense_cost_path)
        cls.first_records = [report.record(path, 'actual_first') for path in cls.paths]
        cls.dense_record = report.record(cls.dense_path, 'actual_dense')

    def test_actual_first_three_routes_and_dense_two_routes_have_distinct_tables(self):
        self.assertTrue(all(set(p['families']) == set(report.ROUTES) for p in self.first))
        self.assertEqual(set(self.dense['families']), {'SELF33', 'JOINT33'})
        self.assertNotIn('reference_shared_controls', self.dense)
        self.assertNotIn('fit', self.dense['references'])
        self.assertEqual(len(report.metric_rows(self.first)), 72)
        self.assertEqual(len(report.pool_rows(self.first)), 54)
        self.assertEqual(len(report.capacity_rows(self.first)), 6)
        self.assertEqual(len(report.expanded_metric_rows(self.dense)), 12)
        self.assertEqual(len(report.pool_rows([self.dense])), 18)
        capacities = report.capacity_rows([self.dense])
        self.assertEqual(len(capacities), 2)
        self.assertEqual({r['channels'] for r in capacities}, {17})
        self.assertEqual({r['features_per_output_channel'] for r in capacities}, {33, 561})

    def test_all_four_actual_audits_bind_the_selected_producers(self):
        auth = report.read(HERE/'AUTHORIZATION_v1.json')
        auth_sha, run_sha = report.sha(HERE/'AUTHORIZATION_v1.json'), report.sha(HERE/'CAMPAIGN_MANIFEST_v1.json')
        self.assertEqual(auth['status'], 'APPROVED')
        report.validate_scientific(self.first, self.first_records,
            report.read(HERE/'audits/FIRST_READOUT_AUDIT_v2.json'), auth_sha, run_sha)
        report.validate_expanded(self.dense, self.dense_record,
            report.read(HERE/'audits/EXPANDED_READOUT_AUDIT_v2.json'), self.first_records, auth_sha, run_sha)
        report.validate_independent_cost(report.read(HERE/'audits/COST_AUDIT_v2.json'), report.record(self.cost_path, 'cost'))
        report.validate_expanded_cost(self.dense_cost, report.record(self.dense_cost_path, 'dense_cost'),
            report.read(HERE/'audits/EXPANDED_COST_AUDIT_v1.json'), self.dense_record)

    def test_actual_reporting_text_includes_counterexample_and_failed_attempt_accounting(self):
        _, summary = report.execution_attempt_rows(self.first_records + [self.dense_record])
        text = report.make_text(self.first, report.read(HERE/'audits/FIRST_READOUT_AUDIT_v2.json'),
            {'status': 'PASS', 'whole_rows_checked': 24}, report.cost_rows(self.cost),
            self.dense, report.expanded_cost_rows(self.dense_cost), summary)
        for phrase in ('반례', '포함 관계가 아니다', '5.25e−8', '4.37498e−5',
                       '재구성 목적', 'EXECUTION_ATTEMPTS.csv'):
            self.assertIn(phrase, text)
        self.assertEqual(summary['accepted_new_model_bundles'], 42)
        self.assertEqual(summary['accepted_reused_model_bundles'], 14)
        self.assertEqual(summary['accepted_pool_comparisons'], 72)
        self.assertGreaterEqual(summary['all_recorded_new_model_bundles_including_unaccepted_attempts'], 56)
        presentation = report.make_presentation_text(self.first, report.cost_rows(self.cost), self.dense)
        self.assertEqual(presentation.count('\n## '), 6)


if __name__ == '__main__':
    unittest.main()
