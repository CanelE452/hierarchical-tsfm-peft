"""Adversarial array fixtures for the independent saved-readout verifier."""
import unittest

import numpy as np

import audit_readout_v1 as audit


def pool_with_scores(values):
    return [{"kind": "PERSIST" if index == 0 else "MEAN" if index == 1 else "RIDGE",
             **({"alpha": audit.ALPHAS[index - 2]} if index >= 2 else {}),
             "tune_scores": {"normalized": {"mse": value}}}
            for index, value in enumerate(values)]


def gate_blocks(values):
    return [{"block": index, "rows": 16, "first_origin": index * 16,
             "last_origin": index * 16 + 15,
             "scores": {"normalized": {"mse": value}}}
            for index, value in enumerate(values)]


class AuditFixtures(unittest.TestCase):
    def test_feature_order_and_self_joint_relationship(self):
        history = np.arange(512, dtype=np.float32)[None, :, None]
        history = np.repeat(history, 2, axis=2)
        history[:, :, 1] += 1000
        own = audit.raw_features("SELF33", history)
        self.assertEqual(own[0, 0, 0], 7.5)
        self.assertEqual(own[0, 0, 1], 1007.5)
        self.assertEqual(own[0, -1, 0], 511)
        np.testing.assert_array_equal(audit.raw_features("JOINT33", history), own.reshape(1, -1))
        np.testing.assert_array_equal(audit.raw_features("SELF512", history), history)

    def test_fit_scale_is_population_and_constant_scale_one(self):
        history = np.zeros((2, 512, 2), np.float32)
        history[1, :, 0] = 2
        center, scale = audit.expected_feature_state("SELF33", history)
        np.testing.assert_array_equal(center[:, 0], np.ones(33))
        np.testing.assert_array_equal(scale, np.ones((33, 2)))
        fresh = history + 100
        changed = audit.independent_design("SELF33", fresh, center, scale)
        self.assertEqual(changed[0, 0, 0], 99)
        np.testing.assert_array_equal(center[:, 0], np.ones(33))

    def test_self_prediction_cannot_use_another_channel(self):
        history = np.zeros((2, 512, 2), np.float32)
        center = np.zeros((33, 2), np.float64)
        scale = np.ones_like(center)
        weights = np.zeros((33, 48, 2), np.float64)
        weights[0] = 1
        bias = np.zeros((48, 2), np.float64)
        before = audit.prediction_from_saved("SELF33", history, history[:, -1], center, scale, weights, bias)
        history[:, :, 1] = 100
        after = audit.prediction_from_saved("SELF33", history, history[:, -1], center, scale, weights, bias)
        np.testing.assert_array_equal(after[:, :, 0], before[:, :, 0])
        self.assertTrue(np.all(after[:, :, 1] == 200))

    def test_mean_uses_fp32_subtraction_before_fp64_average(self):
        target = np.repeat(np.array([1e8, -1e8], np.float32)[:, None, None], 48, axis=1)
        last = np.full((2, 1), np.float32(3.14))
        mask = np.ones_like(target, dtype=np.bool_)
        mean, counts = audit.fit_mean_innovation(target, last, mask)
        np.testing.assert_array_equal(mean, np.zeros((48, 1), np.float32))
        np.testing.assert_array_equal(counts, np.full((48, 1), 2))
        old_order = (target.astype(np.float64) - last[:, None].astype(np.float64)).mean(axis=0)
        self.assertGreater(float(np.max(np.abs(old_order - mean))), 3)

    def test_normal_equation_penalty_is_whole_channel_nc(self):
        design = np.array([-1, 0, 1], np.float64)[:, None, None]
        target = np.repeat(design.astype(np.float32), 2, axis=1)
        last = np.zeros((3, 1), np.float32)
        observed = np.ones_like(target, dtype=np.bool_)
        weights = np.full((1, 2, 1), 2 / 2.6, np.float64)
        bias = np.zeros((2, 1), np.float64)
        checked = audit.normal_equation_checks("SELF33", design, target, last, observed, weights, bias, .1)
        self.assertEqual(checked["channel_counts"], [6])
        wrong = np.full((1, 2, 1), 2 / 2.3, np.float64)
        with self.assertRaisesRegex(audit.AuditFailure, "ALPHA_NC"):
            audit.normal_equation_checks("SELF33", design, target, last, observed, wrong, bias, .1)

    def test_alpha0_equations_do_not_claim_minimum_norm_or_refitting(self):
        design = np.array([-1, 0, 1], np.float64)[:, None, None]
        target = design.astype(np.float32)
        result = audit.normal_equation_checks("SELF33", design, target, np.zeros((3, 1), np.float32),
            np.ones_like(target, dtype=np.bool_), np.ones((1, 1, 1)), np.zeros((1, 1)), 0.)
        self.assertFalse(result["alpha0_minimum_norm_reproved"])
        self.assertFalse(result["independent_solver_replayed"])

    def test_no_observations_require_zero_untrained_output(self):
        x = np.zeros((3, 1, 1))
        y = np.zeros((3, 2, 1), np.float32)
        mask = np.zeros_like(y, np.bool_)
        with self.assertRaisesRegex(audit.AuditFailure, "UNLEARNED"):
            audit.normal_equation_checks("SELF33", x, y, np.zeros((3, 1), np.float32), mask,
                                         np.ones((1, 2, 1)), np.zeros((2, 1)), .1)

    def test_macro_score_is_not_observation_weighted_micro_score(self):
        target = np.zeros((3, 1, 2), np.float32)
        prediction = np.array([[[2, 4]], [[2, 100]], [[2, 100]]], np.float32)
        mask = np.ones_like(target, np.bool_)
        mask[1:, :, 1] = False
        result = audit.scores(prediction, target, mask, np.ones(2, np.float32))
        self.assertEqual(result["normalized"]["mse"], 10)
        self.assertEqual(result["normalized"]["mae"], 3)
        self.assertEqual(result["normalized"]["observed_counts"], [3, 1])
        self.assertNotEqual(result["normalized"]["mse"], (4 * 3 + 16) / 4)

    def test_physical_per_channel_diagnostic_uses_each_std(self):
        target = np.zeros((1, 1, 2), np.float32)
        pred = np.ones_like(target)
        result = audit.scores(pred, target, np.ones_like(target, np.bool_), np.array([2, 3], np.float32))
        self.assertEqual(result["physical_per_channel_diagnostic"]["per_channel_mse"], [4, 9])
        self.assertEqual(result["physical_per_channel_diagnostic"]["mae"], 2.5)

    def test_score_cannot_accept_an_all_unobserved_evaluation(self):
        value = np.zeros((1, 1, 1), np.float32)
        with self.assertRaisesRegex(audit.AuditFailure, "NO_OBSERVED"):
            audit.scores(value, value, np.zeros_like(value, np.bool_), np.ones(1, np.float32))

    def test_global_pool_selects_earliest_exact_tie(self):
        self.assertEqual(audit.select_pool(pool_with_scores([.1] * 9)), 0)
        self.assertEqual(audit.select_pool(pool_with_scores([1, .1, .1, 2, 2, 2, 2, 2, 2])), 1)

    def test_missing_or_reordered_alpha_is_rejected(self):
        pool = pool_with_scores([1] * 9)
        pool[2]["alpha"] = 10
        with self.assertRaisesRegex(audit.AuditFailure, "GRID"):
            audit.select_pool(pool)
        with self.assertRaisesRegex(audit.AuditFailure, "NINE"):
            audit.select_pool(pool[:-1])

    def test_chronological_blocks_keep_sorted_origins(self):
        target = np.zeros((8, 1, 1), np.float32)
        pred = np.arange(8, dtype=np.float32)[:, None, None]
        origins = np.arange(8, dtype=np.int64) * 24 + 512
        blocks = audit.chronological_blocks(pred, target, np.ones_like(target, np.bool_), origins, np.ones(1, np.float32))
        self.assertEqual([block["first_origin"] for block in blocks], [512, 560, 608, 656])
        with self.assertRaisesRegex(audit.AuditFailure, "SORTED"):
            audit.chronological_blocks(pred, target, np.ones_like(target, np.bool_), origins[::-1], np.ones(1, np.float32))

    def test_gate_separates_gain_mae_block_and_inconclusive_conditions(self):
        own = {"normalized": {"mse": 1., "mae": 1.}}
        joint = {"normalized": {"mse": .98, "mae": .99}}
        result = audit.common_information_gate(own, joint, gate_blocks([1] * 4), gate_blocks([.98] * 4))
        self.assertTrue(result["passed"])
        result = audit.common_information_gate(own, joint, gate_blocks([1] * 4), gate_blocks([.98, .98, 1, 1]))
        self.assertFalse(result["passed"])
        bad_mae = {"normalized": {"mse": .98, "mae": 1.02}}
        self.assertFalse(audit.common_information_gate(own, bad_mae, gate_blocks([1] * 4), gate_blocks([.98] * 4))["passed"])
        zero = {"normalized": {"mse": 0., "mae": 0.}}
        self.assertEqual(audit.common_information_gate(zero, joint, gate_blocks([1] * 4), gate_blocks([.98] * 4))["status"], "INCONCLUSIVE")

    def test_array_digest_covers_shape_and_dtype(self):
        values = np.ones(4, np.float32)
        self.assertNotEqual(audit.array_sha(values), audit.array_sha(values.reshape(2, 2)))
        self.assertNotEqual(audit.array_sha(values), audit.array_sha(values.astype(np.float64)))


if __name__ == "__main__":
    unittest.main()
