"""Independent causal constant-past, byte-preservation and future counterexamples."""
import inspect
import json
import copy
import unittest

import numpy as np

import audit_past_constant_gate_v1 as audit

audit.initialize_numpy()


def synthetic_case():
    history = np.zeros((2, 512, 2), np.float32)
    history[:, :, 1] = np.arange(512, dtype=np.float32)
    current = history[:, -1].copy()
    target = np.broadcast_to(current[:, None], (2, 48, 2)).copy()
    target[0, 0, 0] = 1.
    observed = np.ones_like(target, bool)
    observed[1, 0, 1] = False
    values = {"history": history, "history_observed": np.ones_like(history, bool), "target": target,
              "observed": observed, "raw_last": current, "origins": np.array([512, 1024], np.int64),
              "mean": np.zeros(2, np.float32), "std": np.ones(2, np.float32)}
    force = np.tile([True, False], (2, 1))
    complete = observed.all(axis=-1)
    arrays = {"keep_forecast": ~force, "exact_flat": force.copy(), "gate_eligible": force,
              "true_observed_counts": np.full((2, 2), 512, np.int64),
              "target": target, "observed": observed, "raw_last": current, "origins": values["origins"],
              "FIT_std": values["std"], "FIT_mean": values["mean"], "complete_target_positions_mask": complete}
    summary = {"status": "COMPLETE", "windows": 2, "channels": 2, "methods": {},
               "exact_flat_windows_per_channel": [2, 0], "gate_eligible_windows_per_channel": [2, 0],
               "complete_target_positions": 95, "total_target_positions": 96,
               "eligible_windows_with_any_observed_future_per_channel": [2, 0],
               "eligible_windows_with_observed_future_change_per_channel": [1, 0],
               "eligible_observed_future_positions_per_channel": [96, 0],
               "eligible_observed_future_changed_positions_per_channel": [1, 0]}
    predictions = {}
    for index, method in enumerate(audit.METHODS):
        original = np.broadcast_to(current[:, None], target.shape).copy() + np.float32(index + 1)
        gated = original.copy()
        gated[:, :, 0] = current[:, None, 0]
        predictions[method] = original
        arrays[method + "_ungated_prediction"], arrays[method + "_gated_prediction"] = original, gated
        mask = np.broadcast_to(complete[:, :, None], observed.shape)
        summary["methods"][method] = {
            "ungated_scores_all_observed": audit.prior.scores(original, target, observed, values["std"]),
            "gated_scores_all_observed": audit.prior.scores(gated, target, observed, values["std"]),
            "ungated_scores_complete_mask": audit.prior.scores(original, target, mask, values["std"]),
            "gated_scores_complete_mask": audit.prior.scores(gated, target, mask, values["std"]),
            "unchanged_coordinates_byte_exact": True, "gated_coordinates_current_anchor_byte_exact": True,
            "native_head_frozen_intervention": "NATIVE" in method, "optimized_gated_native_head_claimed": False,
            "changed_prediction_coordinates_per_channel": [96, 0]}
    return summary, arrays, values, predictions


class IndependentPastConstantFixtures(unittest.TestCase):
    def test_exact_constant_requires_at_least_two_observed_past_positions(self):
        history = np.full((3, 512, 2), 7., np.float32)
        observed = np.zeros_like(history, bool)
        observed[0, :2] = True
        observed[1, :1] = True
        mask, counts = audit.force_current_from_past(history, observed)
        np.testing.assert_array_equal(mask, [[True, True], [False, False], [False, False]])
        np.testing.assert_array_equal(counts, [[2, 2], [1, 1], [0, 0]])

    def test_single_FP32_ulp_change_prevents_gate_without_tolerance(self):
        history = np.ones((1, 512, 2), np.float32)
        history[0, 123, 0] = np.nextafter(np.float32(1.), np.float32(2.))
        force, _ = audit.force_current_from_past(history, np.ones_like(history, bool))
        np.testing.assert_array_equal(force, [[False, True]])

    def test_constant_after_FF_is_the_contract_not_only_observed_constancy(self):
        history = np.full((1, 512, 1), 8., np.float32)
        history[0, :5] = 0.
        observed = np.ones_like(history, bool)
        observed[0, :5] = False
        force, _ = audit.force_current_from_past(history, observed)
        self.assertFalse(force[0, 0])

    def test_fixed_current_and_unchanged_coordinates_are_byte_preserved(self):
        prediction = np.full((1, 48, 2), [4., -0.], np.float32)
        current = np.array([[-0., 9.]], np.float32)
        force = np.array([[True, False]])
        result = audit.independently_gate_prediction(prediction, current, force)
        audit.exact(result[0, :, 0], np.full(48, -0., np.float32), "current_negative_zero")
        audit.exact(result[0, :, 1], prediction[0, :, 1], "unchanged_negative_zero")
        audit.exact(prediction, np.full((1, 48, 2), [4., -0.], np.float32), "input_not_modified")

    def test_gate_function_has_no_future_or_predictor_specific_arguments(self):
        self.assertEqual(set(inspect.signature(audit.force_current_from_past).parameters), {"history", "history_observed"})
        self.assertEqual(set(inspect.signature(audit.independently_gate_prediction).parameters), {"prediction", "current", "force_current"})

    def test_constant_past_can_have_future_jump_and_gate_can_harm_perfect_forecast(self):
        history = np.zeros((1, 512, 1), np.float32)
        force, _ = audit.force_current_from_past(history, np.ones_like(history, bool))
        current = history[:, -1]
        target = np.ones((1, 48, 1), np.float32)
        values = {"target": target, "observed": np.ones_like(target, bool), "raw_last": current, "std": np.array([1.], np.float32)}
        result = audit.gated_score_pair(target, audit.independently_gate_prediction(target, current, force), values, force)
        self.assertEqual(result["all_observed"]["original"]["normalized"]["mse"], 0.)
        self.assertEqual(result["all_observed"]["gated"]["normalized"]["mse"], 1.)
        diagnostic = audit.future_diagnostic_counts(target, values["observed"], current, force)
        self.assertEqual(diagnostic["totals"]["gated_future_changed_positions"], 48)
        self.assertFalse(diagnostic["constant_past_guarantees_constant_future_claimed"])

    def test_future_missing_positions_are_neither_flat_nor_changed_evidence(self):
        target = np.full((1, 48, 2), np.nan, np.float32)
        observed = np.zeros_like(target, bool)
        target[0, 0], target[0, 1] = [0., 1.], [2., 0.]
        observed[0, :2] = True
        result = audit.future_diagnostic_counts(target, observed, np.zeros((1, 2), np.float32), np.ones((1, 2), bool))
        self.assertEqual(result["totals"]["gated_future_observed_positions"], 4)
        self.assertEqual(result["totals"]["gated_future_changed_positions"], 2)
        self.assertEqual(result["totals"]["gated_future_equal_current_positions"], 2)
        self.assertEqual(result["totals"]["gated_windows_with_all_48_future_observed"], 0)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_conditional_gate_changes_output_space_so_old_static_D_floor_is_invalid(self):
        original = np.full((1, 48, 2), [1., 1.], np.float32)
        target = np.full_like(original, [1., 0.])
        current = np.zeros((1, 2), np.float32)
        force = np.array([[False, True]])
        gated = audit.independently_gate_prediction(original, current, force)
        values = {"target": target, "observed": np.ones_like(target, bool), "raw_last": current, "std": np.ones(2, np.float32)}
        result = audit.gated_score_pair(original, gated, values, force)
        old_D = np.array([[1., 1.]], np.float32)
        old_minimum = audit.decoder_audit.span_audit.independent_span(target, values["observed"], current, old_D, values["std"])["scores"]["normalized"]["mse"]
        self.assertAlmostEqual(old_minimum, .25)
        self.assertEqual(result["all_observed"]["gated"]["normalized"]["mse"], 0.)
        self.assertFalse(result["fixed_static_D_error_floor_applied_to_gated_forecast"])

    def test_same_gate_applies_to_RAW_and_both_projection_forecasters(self):
        current = np.array([[1., 2.]], np.float32)
        force = np.array([[True, False]])
        forecasts = [np.full((1, 48, 2), value, np.float32) for value in (3., 4., 5.)]
        for prediction in forecasts:
            result = audit.independently_gate_prediction(prediction, current, force)
            np.testing.assert_array_equal(result[0, :, 0], np.ones(48, np.float32))
            audit.exact(result[0, :, 1], prediction[0, :, 1], "shared_control_rule")

    def test_full_observed_and_complete_C_scores_keep_distinct_denominators(self):
        target = np.zeros((1, 48, 2), np.float32)
        observed = np.zeros_like(target, bool)
        observed[0, 0] = True
        observed[0, 1, 0] = True
        original = np.zeros_like(target)
        original[0, 0], original[0, 1] = [2., 4.], [10., 8.]
        current = np.zeros((1, 2), np.float32)
        force = np.array([[False, True]])
        values = {"target": target, "observed": observed, "raw_last": current, "std": np.ones(2, np.float32)}
        result = audit.gated_score_pair(original, audit.independently_gate_prediction(original, current, force), values, force)
        self.assertEqual(result["all_observed"]["original"]["normalized"]["mse"], 34.)
        self.assertEqual(result["complete_C"]["original"]["normalized"]["mse"], 10.)

    def test_nonfinite_past_cannot_be_accepted_as_constant(self):
        history = np.zeros((1, 512, 1), np.float32)
        history[0, 20, 0] = np.nan
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FINITE_FP32_512"):
            audit.force_current_from_past(history, np.ones_like(history, bool))

    def test_seven_method_pair_schema_recalculates_336_fields_and_serializes(self):
        summary, arrays, values, predictions = synthetic_case()
        result = audit.audit_case(summary, arrays, values, predictions)
        self.assertEqual(result["score_field_checks"], 7 * 4 * 12)
        self.assertEqual(result["future_change_diagnostic"]["totals"]["gated_future_changed_positions"], 1)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_wrong_future_diagnostic_counts_cannot_hide_behind_matching_scores(self):
        summary, arrays, values, predictions = synthetic_case()
        summary["eligible_observed_future_changed_positions_per_channel"][0] = 0
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FUTURE_DIAGNOSTIC_COUNTS"):
            audit.audit_case(summary, arrays, values, predictions)

    def test_wrong_observed_past_counts_are_rejected_before_performance_claim(self):
        summary, arrays, values, predictions = synthetic_case()
        arrays["true_observed_counts"][0, 0] = 1
        with self.assertRaisesRegex(audit.prior.AuditFailure, "true_observed_counts"):
            audit.audit_case(summary, arrays, values, predictions)

    def test_score_change_or_wrong_gated_coordinate_is_rejected(self):
        summary, arrays, values, predictions = synthetic_case()
        summary["methods"]["RAW_BOLT"]["gated_scores_all_observed"]["normalized"]["mse"] += 1e-7
        with self.assertRaisesRegex(audit.prior.AuditFailure, "NUMERIC_MISMATCH"):
            audit.audit_case(summary, arrays, values, predictions)
        summary, arrays, values, predictions = synthetic_case()
        arrays["PCA_PINV_BOLT_gated_prediction"][0, 0, 1] += 1.
        with self.assertRaisesRegex(audit.prior.AuditFailure, "independent_gate_output"):
            audit.audit_case(summary, arrays, values, predictions)


if __name__ == "__main__":
    unittest.main()
