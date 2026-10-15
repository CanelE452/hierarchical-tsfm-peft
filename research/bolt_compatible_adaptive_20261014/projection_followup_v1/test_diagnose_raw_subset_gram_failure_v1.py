"""Synthetic arithmetic fixtures only; never opens FIT, targets or model payloads."""
import copy
import inspect
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_raw_subset_gram_failure_v1 as diagnosis

diagnosis.initialize_numpy()


def toy():
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
    x = np.stack((s, t, s + t), axis=1)[None]
    return x, np.ones(x.shape, bool)


def recorded_case():
    return {"candidate_index": 1, "response_channel": 1, "observed_rows": 131072, "Gram": [[1.155108443331784]], "cross": [1.1551084433323342], "variance": 1.1551084433317942, "beta_Gram": [1.0000000000004763], "cross_contribution": 1.1551084433328844, "quadratic_contribution": 1.1551084433328844, "Gram_MSE": -1.0902390101819037e-12, "original_negative_tolerance": 1.3132062614805414e-13, "negative_over_tolerance": 8.302115533265438, "direct_lstsq_beta": [.999999999999568], "direct_centered_residual_MSE": 2.1555447200758566e-25, "direct_rank": 1, "direct_singular_values": [389.1045796241033], "direct_residual_max_abs": 1.538325022920617e-12}


class PreservedGramFailureTests(unittest.TestCase):
    def test_source_contract_is_original_executed_subtractive_guard(self):
        self.assertEqual(diagnosis.source_contract()["sha256"], diagnosis.OLD_MATH_SHA)

    def test_self_channel_direct_residual_is_nonnegative_and_zero_with_small_exact_design(self):
        flat, mask, stats = diagnosis.channel_statistics(*toy())
        result = diagnosis.inspect_response(flat, mask, stats, 0, [0])
        self.assertFalse(result["negative_score_guard_failed"])
        self.assertLess(result["direct_centered_lstsq_residual_MSE"], 1e-26)
        self.assertEqual(result["direct_centered_lstsq_rank"], 1)

    def test_sufficient_stat_roundoff_can_trigger_guard_while_direct_squared_residual_remains_zero(self):
        flat, mask, stats = diagnosis.channel_statistics(*toy())
        stats[0]["cross"][0] += 1e-12
        result = diagnosis.inspect_response(flat, mask, stats, 0, [0])
        self.assertTrue(result["negative_score_guard_failed"])
        self.assertGreater(result["negative_error_divided_by_original_tolerance"], 1)
        self.assertGreaterEqual(result["same_Gram_coefficients_direct_residual_MSE"], 0)
        self.assertLess(result["direct_centered_lstsq_residual_MSE"], 1e-26)
        self.assertFalse(result["original_tolerance_relaxed"])

    def test_first_failure_trace_preserves_stage_candidate_response_without_adopting_subset(self):
        x, observed = toy(); flat, mask, stats = diagnosis.channel_statistics(x, observed)
        stats[0]["cross"][0] += 1e-12
        with patch.object(diagnosis, "channel_statistics", return_value=(flat, mask, stats)):
            result = diagnosis.first_failure(x, observed, 2)
        self.assertEqual((result["failure_stage"], result["candidate_original_index"], result["first_failure"]["response_original_index"]), (1, 0, 0))
        self.assertFalse(result["new_subset_selected"])

    def test_inactive_response_is_skipped_and_no_failure_has_explicit_status(self):
        x, observed = toy(); observed[:, :, 2] = False
        result = diagnosis.first_failure(x, observed, 1)
        self.assertEqual(result["status"], "ORIGINAL_NEGATIVE_GUARD_NOT_REPRODUCED")
        self.assertFalse(result["new_subset_selected"])

    def test_true_response_mask_ignores_missing_placeholders_when_not_a_selected_feature(self):
        x, observed = toy(); observed[:, :60, 1] = False
        a = diagnosis.inspect_response(*diagnosis.channel_statistics(x, observed), 1, [0])
        changed = x.copy(); changed[:, :60, 1] = 12345.
        b = diagnosis.inspect_response(*diagnosis.channel_statistics(changed, observed), 1, [0])
        self.assertEqual(a["direct_centered_lstsq_residual_MSE"], b["direct_centered_lstsq_residual_MSE"])

    def test_known_recorded_scalar_terms_are_recomputed_without_FIT_reopen(self):
        result = diagnosis.verify_recorded_Gram_case(recorded_case())
        self.assertLess(result["recomputed_subtractive_Gram_MSE"], -result["unchanged_original_guard_tolerance"])
        self.assertFalse(result["direct_LS_reexecuted_by_this_metadata_check"])
        self.assertFalse(result["model_prediction_quality_failure_inferred"])

    def test_wrong_tolerance_or_negative_direct_residual_is_rejected(self):
        for field, value in (("original_negative_tolerance", 1e-8), ("direct_centered_residual_MSE", -1e-25), ("Gram_MSE", 0.)):
            row = copy.deepcopy(recorded_case()); row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): diagnosis.verify_recorded_Gram_case(row)

    def test_no_future_or_forecast_inputs_in_reproducer_API(self):
        self.assertEqual(list(inspect.signature(diagnosis.first_failure).parameters), ["history", "observed", "k"])
        self.assertEqual(list(inspect.signature(diagnosis.channel_statistics).parameters), ["history", "observed"])


if __name__ == "__main__": unittest.main()
