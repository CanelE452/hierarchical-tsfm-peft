"""Report gates reject unbound producers and changed selection contracts."""
import copy
import unittest

import make_readout_report_v1 as report


def fixture():
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


class ReportGates(unittest.TestCase):
    def test_earliest_exact_tie_accepts_only_persist(self):
        p, r, a = fixture()
        self.assertTrue(report.validate_scientific(p, r, a, 'auth', 'clock'))
        p[0]['families']['SELF33'].update(selected_index=1, selected_kind='MEAN')
        with self.assertRaisesRegex(ValueError, 'EARLIEST_GLOBAL_TUNE_SELECTION'):
            report.validate_scientific(p, r, a, 'auth', 'clock')

    def test_another_producer_same_cohort_fails_exact_sha(self):
        p, r, a = fixture()
        a['producer_records'][0]['sha256'] = 'another-producer'
        with self.assertRaisesRegex(ValueError, 'EXACT_SHA_BINDING'):
            report.validate_scientific(p, r, a, 'auth', 'clock')

    def test_old_authorization_or_partial_candidate_audit_rejected(self):
        p, r, a = fixture()
        p[0]['permission_gate']['sha256'] = 'expired-auth'
        with self.assertRaisesRegex(ValueError, 'APPROVED_COMPLETE'):
            report.validate_scientific(p, r, a, 'auth', 'clock')
        p, r, a = fixture()
        a['totals']['pool_candidates_checked'] = 53
        with self.assertRaisesRegex(ValueError, '54_POOL'):
            report.validate_scientific(p, r, a, 'auth', 'clock')

    def test_cost_requires_exact_result_and_all_actual_repeats(self):
        audit = {'status': 'PASS', 'cost_producer_record': {'sha256': 'actual-cost'},
                 'rows_checked': 24, 'timed_repeats_checked': 144, 'source_before_after_SHA_pass': True}
        report.validate_independent_cost(audit, {'sha256': 'actual-cost'})
        with self.assertRaisesRegex(ValueError, 'EXACT_INDEPENDENT'):
            report.validate_independent_cost(audit, {'sha256': 'another-same-cohort-cost'})
        audit['timed_repeats_checked'] = 143
        with self.assertRaisesRegex(ValueError, '144_REPEAT'):
            report.validate_independent_cost(audit, {'sha256': 'actual-cost'})


if __name__ == '__main__':
    unittest.main()
