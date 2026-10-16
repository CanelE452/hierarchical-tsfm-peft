"""Functional future interventions must expose deliberately leaky callables."""
import unittest
from unittest.mock import patch

import numpy as np

import data_contract


class InformationAuditTests(unittest.TestCase):
    def setUp(self):
        import information_audit
        self.audit = information_audit
        rng = np.random.default_rng(23)
        self.task = {"physical": rng.normal(size=(50, 4)).astype(np.float32),
                     "observed": np.ones((50, 4), bool), "L": 8, "H": 3}
        self.task["observed"][23:25, 1] = False
        self.task["physical"][23:25, 1] = np.nan

    def test_actual_functions_are_invariant_and_source_arrays_unchanged(self):
        before = data_contract.array_sha256(self.task["physical"])
        stats = self.audit.audit_statistics(self.task, 20, 2)
        self.assertEqual(stats["status"], "PASS")
        self.assertTrue(stats["mean_exact"] and stats["std_exact"] and stats["pca_projector_exact"])
        mean, std, _ = data_contract.fit_statistics(self.task, 20)
        with patch.object(data_contract, "target_window", side_effect=AssertionError("Future target read")):
            windows = self.audit.audit_histories(self.task, [25, 32, 40], mean, std)
        self.assertEqual(windows["status"], "PASS")
        self.assertEqual(len(windows["windows"]), 3)
        self.assertEqual(before, data_contract.array_sha256(self.task["physical"]))

    def test_leaky_mean_function_actually_fails(self):
        def leaky(data, end):
            return data_contract.fit_statistics(data, len(data["physical"]))
        result = self.audit.audit_statistics(self.task, 20, 2, statistics_fn=leaky)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["mean_exact"])

    def test_leaky_history_function_actually_fails(self):
        def leaky(data, origin, mean, std, length=None):
            x, mask = data_contract.history_window(data, origin, mean, std, length)
            return x + data["physical"][origin, 0], mask
        mean, std, _ = data_contract.fit_statistics(self.task, 20)
        result = self.audit.audit_histories(self.task, [25], mean, std, history_fn=leaky)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["windows"][0]["history_exact"])

    def test_mask_and_origin_boundary_are_part_of_intervention(self):
        changed = self.audit.perturb_copy(self.task, 25)
        np.testing.assert_array_equal(changed["observed"][:25], self.task["observed"][:25])
        np.testing.assert_array_equal(changed["observed"][25:], ~self.task["observed"][25:])
        np.testing.assert_array_equal(changed["physical"][:25], self.task["physical"][:25])
        self.assertTrue(np.isfinite(changed["physical"][25:]).all())
        mean, std, _ = data_contract.fit_statistics(self.task, 20)
        a, mask_a = data_contract.history_window(self.task, 25, mean, std)
        b, mask_b = data_contract.history_window(changed, 25, mean, std, length=self.task["L"])
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(mask_a, mask_b)
        self.assertFalse(mask_a[-1, 1])

    def test_source_call_contract_and_intentional_target_argument_rejection(self):
        result = self.audit.audit_source_calls()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(row["target_argument_absent"] for row in result["functions"]))
        self.assertFalse(self.audit.arguments_have_no_targets(["self", "x", "future_target"]))


if __name__ == "__main__": unittest.main()
