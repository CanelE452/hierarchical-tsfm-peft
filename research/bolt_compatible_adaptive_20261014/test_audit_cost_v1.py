"""Independent cost audit fixtures for missing-data chronology and arithmetic."""
import unittest

import numpy as np

import audit_cost_v1 as audit


def cost_row(seconds, raw, batch=8):
    return {"seconds": seconds, "batch": batch, "median_seconds": 3.5,
            "median_seconds_per_window": 3.5 / batch,
            "paired_RAW_seconds": raw, "paired_RAW_ratios": [value / baseline for value, baseline in zip(seconds, raw)],
            "paired_RAW_median_ratio": float(np.median([value / baseline for value, baseline in zip(seconds, raw)])),
            "repeats": 6, "warmup_each_route": 2, "positions": [0, 1, 2, 3, 4, 5]}


class CostAuditFixtures(unittest.TestCase):
    def test_missing_history_resets_to_fit_mean_at_each_window(self):
        physical = np.full((2, 512, 1), np.nan, np.float32)
        mask = np.zeros_like(physical, np.bool_)
        physical[0, 1, 0] = 6
        mask[0, 1, 0] = True
        normalized = audit.scalar_preprocess(physical, mask, np.array([4], np.float32), np.array([2], np.float32))
        self.assertEqual(normalized[0, 0, 0], 0)
        self.assertEqual(normalized[0, 1, 0], 1)
        self.assertEqual(normalized[0, -1, 0], 1)
        self.assertTrue(np.all(normalized[1] == 0))

    def test_later_observation_does_not_fill_an_earlier_missing_value(self):
        physical = np.zeros((1, 512, 1), np.float32)
        mask = np.zeros_like(physical, np.bool_)
        mask[0, 100, 0] = True
        physical[0, 100, 0] = 7
        before = audit.scalar_preprocess(physical, mask, np.array([2], np.float32), np.ones(1, np.float32))
        physical[0, 100, 0] = 999
        after = audit.scalar_preprocess(physical, mask, np.array([2], np.float32), np.ones(1, np.float32))
        np.testing.assert_array_equal(before[:, :100], after[:, :100])
        self.assertNotEqual(before[0, 100, 0], after[0, 100, 0])

    def test_even_repeat_median_batch_scaling_and_paired_raw_ratios(self):
        seconds = [6., 1., 3., 4., 2., 5.]
        raw = [12.] * 6
        row = cost_row(seconds, raw)
        result = audit.audit_recorded_times(row, {"seconds": raw})
        self.assertEqual(result["median_seconds"], 3.5)
        self.assertEqual(result["median_seconds_per_window"], .4375)
        self.assertFalse(result["clock_readings_independently_remeasured"])

    def test_time_arithmetic_does_not_silently_repair_wrong_batch_division(self):
        raw = [12.] * 6
        row = cost_row([6., 1., 3., 4., 2., 5.], raw)
        row["median_seconds_per_window"] = 3.5
        with self.assertRaisesRegex(audit.AuditFailure, "ARITHMETIC"):
            audit.audit_recorded_times(row, {"seconds": raw})

    def test_duplicate_route_position_is_not_balanced(self):
        raw = [12.] * 6
        row = cost_row([6., 1., 3., 4., 2., 5.], raw)
        row["positions"] = [0, 1, 2, 3, 4, 4]
        with self.assertRaisesRegex(audit.AuditFailure, "BALANCE"):
            audit.audit_recorded_times(row, {"seconds": raw})

    def test_wrong_raw_pairing_is_rejected_even_if_median_is_unchanged(self):
        raw = [12.] * 6
        row = cost_row([6., 1., 3., 4., 2., 5.], raw)
        row["paired_RAW_seconds"][0] = 13.
        with self.assertRaisesRegex(audit.AuditFailure, "RATIOS"):
            audit.audit_recorded_times(row, {"seconds": [12.] * 6})

    def test_parity_checks_saved_arrays_instead_of_trusting_pass_flag(self):
        reference = np.ones((1, 48, 1), np.float32)
        changed = reference.copy()
        changed[0, 0, 0] += .01
        actual = audit.parity(changed, reference)
        self.assertFalse(actual["pass"])
        with self.assertRaisesRegex(audit.AuditFailure, "PARITY_FAILURE"):
            audit.compare_parity({**actual, "pass": True}, actual, "tampered_pass")


if __name__ == "__main__":
    unittest.main()
