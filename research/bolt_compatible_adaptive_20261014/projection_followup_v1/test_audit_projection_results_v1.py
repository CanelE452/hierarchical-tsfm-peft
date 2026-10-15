"""Synthetic counterexamples; no real dataset, result, checkpoint or Bolt reads."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_projection_results_v1 as audit


class IndependentProjectionAuditFixtures(unittest.TestCase):
    def test_macro_is_channel_balanced_not_pooled_and_uses_same_fit_units(self):
        prediction = np.array([[[2., 4., np.nan], [2., np.nan, np.nan]]], np.float32)
        target = np.zeros_like(prediction)
        observed = np.array([[[True, True, False], [True, False, False]]])
        value = audit.scores(prediction, target, observed, np.array([3., 2., 100.]))
        self.assertEqual(value["normalized"]["mse"], 10.)
        self.assertEqual(value["normalized"]["mae"], 3.)
        self.assertEqual(value["physical_per_channel_diagnostic"]["mse"], 50.)
        self.assertEqual(value["physical_per_channel_diagnostic"]["mae"], 7.)
        self.assertEqual(value["normalized"]["observed_counts"], [2, 1, 0])

    def test_nonfinite_observed_prediction_is_rejected(self):
        with self.assertRaisesRegex(audit.AuditFailure, "NONFINITE_OBSERVED"):
            audit.scores(np.array([[[np.nan]]]), np.zeros((1, 1, 1)), np.ones((1, 1, 1), bool), [1.])

    def test_same_count_endpoints_cannot_hide_changed_middle_origin(self):
        original = np.array([512, 600, 700], np.int64)
        lock = {"count": 3, "first": 512, "last": 700, "sha256": hashlib.sha256(original.tobytes()).hexdigest()}
        self.assertEqual(audit.verify_origins(original, lock), 3)
        with self.assertRaisesRegex(audit.AuditFailure, "ORIGIN_LOCK_DIGEST"):
            audit.verify_origins(np.array([512, 601, 700], np.int64), lock)

    def test_anchor_preserves_current_C_even_when_K_cannot_reconstruct_it(self):
        history = np.array([[[1., 99.], [4., 100.]]], np.float32)
        encoder = np.array([[1.], [0.]])
        decoder = np.array([[1., 0.]])
        mean = np.array([2., 70.])
        z_last = audit.encode(history[:, -1:], mean, encoder)
        predicted = audit.anchor_prediction(history, history[:, -1], z_last, mean, encoder, decoder)
        np.testing.assert_array_equal(predicted, history[:, -1:])
        changed = history[:, -1].copy()
        changed[0, 1] += 1.
        with self.assertRaisesRegex(audit.AuditFailure, "original_current_C_anchor"):
            audit.anchor_prediction(history, changed, z_last, mean, encoder, decoder)

    def test_future_mask_and_future_values_cannot_change_pointwise_past_encoding(self):
        past = np.array([[[1., 2.], [3., 5.]]], np.float32)
        future = np.array([[[4., 6.]]], np.float32)
        before = audit.encode(past, [1., 1.], np.array([[1.], [2.]]))
        future[:] = -10000.
        np.testing.assert_array_equal(before, audit.encode(past, [1., 1.], np.array([[1.], [2.]])))
        prefix = audit.encode(past[:, :1], [1., 1.], np.array([[1.], [2.]]))
        np.testing.assert_array_equal(prefix, before[:, :1])

    def test_foreca_whitened_components_pass_and_zero_component_fails(self):
        x = np.array([[[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]]], np.float32)
        e = np.diag([.5, 1.])
        d = np.diag([2., 1.])
        report = audit.check_projection_state("FORECA", x, np.zeros(2), e, d,
                                              fit_observed=np.ones_like(x, bool), decoder_bias=np.zeros(2))
        self.assertEqual(report["effective_rank"], 2)
        with self.assertRaisesRegex(audit.AuditFailure, "unit_variance"):
            audit.check_projection_state("FORECA", x, np.zeros(2), np.zeros_like(e), d,
                                         fit_observed=np.ones_like(x, bool), decoder_bias=np.zeros(2))

    def test_pca_top_variance_direction_is_checked_not_just_orthogonality(self):
        x = np.array([[[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]]], np.float32)
        report = audit.check_projection_state("PCA", x, np.zeros(2), np.array([[.5], [0.]]), np.array([[2., 0.]]),
                                              fit_observed=np.ones_like(x, bool), decoder_bias=np.zeros(2))
        self.assertEqual(report["components"], 1)
        with self.assertRaisesRegex(audit.AuditFailure, "top_K_covariance"):
            audit.check_projection_state("PCA", x, np.zeros(2), np.array([[0.], [1.]]), np.array([[0., 1.]]),
                                         fit_observed=np.ones_like(x, bool), decoder_bias=np.zeros(2))

    def test_rank_and_fit_decoder_are_separate_constraints(self):
        rank_one = np.array([[[-1., -1.], [1., 1.]]], np.float32)
        with self.assertRaisesRegex(audit.AuditFailure, "RANK_BELOW_K"):
            audit.check_projection_state("PCA", rank_one, np.zeros(2), np.eye(2), np.eye(2),
                                         fit_observed=np.ones_like(rank_one, bool), decoder_bias=np.zeros(2))
        x = np.array([[[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]]], np.float32)
        with self.assertRaisesRegex(audit.AuditFailure, "D_normal_equations"):
            audit.check_projection_state("PCA", x, np.zeros(2), np.array([[.5], [0.]]), np.zeros((1, 2)),
                                         fit_observed=np.ones_like(x, bool), decoder_bias=np.zeros(2))

    def test_tune_exact_tie_selects_earliest_checkpoint_not_latest(self):
        rows = [{"update": update, "tune_mse": loss} for update, loss in zip(audit.CHECKPOINTS, [2., 1., 1., 1.5, 1.])]
        self.assertEqual(audit.select_checkpoint(rows)["selected_update"], 100)
        rows[2]["tune_mse"] = 1. - 1e-13
        self.assertEqual(audit.select_checkpoint(rows)["selected_update"], 200)

    def test_masked_D_does_not_treat_forward_filled_targets_as_observed(self):
        x = np.array([[[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]]], np.float32)
        observed = np.ones_like(x, bool)
        observed[0, [1, 3], 1] = False
        e, d, bias = np.array([[.5], [0.]]), np.array([[2., 0.]]), np.array([0., -1.])
        audit.check_projection_state("PCA", x, np.zeros(2), e, d,
                                     fit_observed=observed, decoder_bias=bias)
        with self.assertRaisesRegex(audit.AuditFailure, "D_normal_equations"):
            audit.check_projection_state("PCA", x, np.zeros(2), e, d,
                                         fit_observed=np.ones_like(x, bool), decoder_bias=bias)

    def test_native_zero_head_is_persistence_and_each_K_uses_own_past(self):
        z = np.array([[[1., 30.], [3., 20.], [5., 10.]]], np.float32)
        weights, bias = np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.float32)
        np.testing.assert_array_equal(audit.native_latent_forecast(z, weights, bias), np.repeat(z[:, -1:], 2, axis=1))
        weights[0, 0, 0] = 1.
        future = audit.native_latent_forecast(z, weights, bias)
        self.assertEqual(float(future[0, 0, 0]), 1.)
        changed = z.copy()
        changed[0, 0, 1] = -900.
        np.testing.assert_array_equal(audit.native_latent_forecast(changed, weights, bias)[:, :, 0], future[:, :, 0])

    def test_frozen_F_hash_and_configuration_are_separate_certificates(self):
        before, configuration = {"weights": "123"}, {"H": 64, "median_index": 4}
        audit.check_frozen_model_records(before, before.copy(), configuration, configuration.copy())
        with self.assertRaisesRegex(audit.AuditFailure, "STATE_SIGNATURE"):
            audit.check_frozen_model_records(before, {"weights": "124"}, configuration, configuration)
        with self.assertRaisesRegex(audit.AuditFailure, "CONFIG_SIGNATURE"):
            audit.check_frozen_model_records(before, before, configuration, {"H": 64, "median_index": 3})

    def test_physical_fill_is_causal_and_resets_to_FIT_mean_every_window(self):
        physical = np.full((2, 512, 1), np.nan, np.float32)
        observed = np.zeros_like(physical, bool)
        physical[0, 3, 0] = 14.
        observed[0, 3, 0] = True
        value = audit.causal_physical_preprocess(physical, observed, np.array([10.], np.float32), np.array([2.], np.float32))
        np.testing.assert_array_equal(value[0, :3], np.zeros((3, 1), np.float32))
        np.testing.assert_array_equal(value[0, 3:], np.full((509, 1), 2., np.float32))
        np.testing.assert_array_equal(value[1], np.zeros((512, 1), np.float32))
        physical[0, 510, 0] = -200.
        observed[0, 510, 0] = True
        changed = audit.causal_physical_preprocess(physical, observed, np.array([10.], np.float32), np.array([2.], np.float32))
        np.testing.assert_array_equal(value[0, :510], changed[0, :510])

    def test_metric_recalculation_detects_reported_good_score_without_prediction_change(self):
        values = audit.scores(np.ones((2, 3, 2), np.float32), np.zeros((2, 3, 2), np.float32),
                              np.ones((2, 3, 2), bool), [2., 3.])
        self.assertEqual(audit.compare_scores(values, values, "same"), 12)
        mutated = {key: dict(row) for key, row in values.items()}
        mutated["normalized"]["mse"] = .9
        with self.assertRaisesRegex(audit.AuditFailure, "NUMERIC_MISMATCH"):
            audit.compare_scores(mutated, values, "changed")

    def test_projection_metadata_binds_actual_FIT_past_not_only_row_counts(self):
        pattern = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]], np.float32)
        x = np.tile(pattern, (128, 1))[None]
        observed = np.ones_like(x, bool)
        state = {"center": np.zeros(2), "encoder": np.array([[.5], [0.]]),
                 "decoder": np.array([[2., 0.]]), "decoder_bias": np.zeros(2)}
        metadata = {"route": "PCA", "channels": 2, "K": 1, "length": 512, "fit_windows": 1,
                    "fit_positions_with_overlap": 512, "whitening_rank": 2,
                    "eigenvalue_relative_cut": 1e-8, "latent_fit_covariance": [[1.]],
                    "history_sha256": audit.array_sha(x), "history_observed_sha256": audit.array_sha(observed),
                    "decoder_observed_counts": [512, 512], "strict_K_only_compression": False,
                    "actual_future_target_used_in_projection_or_decoder_fit": False,
                    "encoder_fit_dtype": "float64", "serialized_state_dtype": "float64",
                    "deployment_dtype": "float32", "foreca_search": []}
        report = audit.check_projection_metadata(metadata, state, x, observed)
        self.assertEqual(report["effective_rank"], 2)
        changed = x.copy()
        changed[0, 100, 1] += 1.
        with self.assertRaisesRegex(audit.AuditFailure, "FIT_PAST_ARRAY_SHA_BINDING"):
            audit.check_projection_metadata(metadata, state, changed, observed)

    def test_artifact_manifest_digest_is_raw_bytes_and_state_digest_has_type_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SYNTHETIC_ONLY.npz"
            array = np.array([1., 2.], np.float32)
            np.savez(path, value=array)
            record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "arrays": {"value": {"shape": [2], "dtype": "float32",
                                 "sha256": hashlib.sha256(array.tobytes()).hexdigest()}}}
            loaded = audit.verified_arrays(record, lambda: None)
            np.testing.assert_array_equal(loaded["value"], array)
            self.assertNotEqual(audit.array_sha(array), record["arrays"]["value"]["sha256"])
            record["arrays"]["value"]["sha256"] = audit.array_sha(array)
            with self.assertRaisesRegex(audit.AuditFailure, "NPZ_ARRAY_CONTENT"):
                audit.verified_arrays(record, lambda: None)

    def test_failed_producer_cannot_become_PASS_from_reported_completed_counts(self):
        result = {"status": "FAILED_PRESERVED", "projection_fits": 4, "native_training_fits": 8,
                  "failure": {"type": "ValueError", "message": "cost did not finish"}}
        report = audit.incomplete_producer_report(result)
        self.assertEqual(report["status"], "NOT_AUDITED_INCOMPLETE_PRODUCER")
        self.assertFalse(report["scientific_assets_opened"])
        self.assertFalse(report["full_scope_completion_or_PASS_claimed"])
        with self.assertRaisesRegex(audit.AuditFailure, "INCOMPLETE_STATUS"):
            audit.incomplete_producer_report({"status": "COMPLETE"})

    def test_native_checkpoint_cannot_change_frozen_projection_and_keep_declared_dimensions(self):
        state = {"center": np.zeros(2), "encoder": np.array([[.5], [0.]]),
                 "decoder": np.array([[2., 0.]]), "decoder_bias": np.zeros(2)}
        arrays = {"projection." + key: value.astype(np.float32) for key, value in state.items()}
        arrays.update({"head.weight": np.zeros((1, 48, 512), np.float32), "head.bias": np.zeros((1, 48), np.float32)})
        certificate = audit.native_state_certificate(arrays, state)
        self.assertEqual(certificate["K"], 1)
        arrays["projection.decoder"][0, 1] = .01
        with self.assertRaisesRegex(audit.AuditFailure, "frozen_native_projection"):
            audit.native_state_certificate(arrays, state)

    def test_checked_file_record_requires_exact_canonical_path_and_SHA_before_array_manifest_use(self):
        path = (audit.ROOT / "SYNTHETIC_ONLY.npz").resolve()
        checked = {"path": str(path), "sha256": "locked_file_digest"}
        canonical = {"path": "SYNTHETIC_ONLY.npz", "sha256": "locked_file_digest",
                     "arrays": {"value": {"dtype": "float32", "shape": [1], "sha256": "array_digest"}}}
        bound = audit.bind_canonical_npz(checked, canonical)
        self.assertEqual(bound["path"], str(path))
        self.assertEqual(bound["arrays"], canonical["arrays"])
        changed = {**canonical, "sha256": "different_file_digest"}
        with self.assertRaisesRegex(audit.AuditFailure, "CANONICAL_PATH_FILE_SHA"):
            audit.bind_canonical_npz(checked, changed)

    def test_frozen_ED_detects_one_coefficient_mutation(self):
        before = {"E": np.eye(2), "D": np.eye(2)}
        after = {key: value.copy() for key, value in before.items()}
        audit.check_frozen_arrays(before, after)
        after["D"][0, 0] += 1e-12
        with self.assertRaisesRegex(audit.AuditFailure, "EXACT_MISMATCH"):
            audit.check_frozen_arrays(before, after)

    def test_oracle_excludes_every_channel_at_partially_missing_time(self):
        observed = np.array([[[True, False], [True, True]]])
        expected = np.array([[[False, False], [True, True]]])
        contract = audit.check_oracle_contract(expected, observed, "diagnostic_only_not_deployment_or_selection")
        self.assertEqual(contract["complete_target_positions"], 1)
        self.assertFalse(contract["eligible_for_adoption"])
        with self.assertRaisesRegex(audit.AuditFailure, "oracle_same_complete"):
            audit.check_oracle_contract(observed, observed, "diagnostic_only_not_deployment_or_selection")
        with self.assertRaisesRegex(audit.AuditFailure, "DIAGNOSTIC_ROLE"):
            audit.check_oracle_contract(expected, observed, "deploy")

    def test_spectral_forecastability_is_scale_invariant_and_rejects_constant(self):
        time = np.arange(512)
        wave = np.sin(2 * np.pi * 8 * time / 512)[None, :, None]
        first = audit.spectral_diagnostics(wave)
        second = audit.spectral_diagnostics(wave * 100)
        np.testing.assert_allclose(first["probabilities"], second["probabilities"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(first["forecastability"], second["forecastability"], atol=1e-12)
        self.assertGreater(first["forecastability"][0], .5)
        with self.assertRaisesRegex(audit.AuditFailure, "COLLAPSED_SPECTRAL"):
            audit.spectral_diagnostics(np.ones((2, 512, 1)))

    def test_one_sided_internal_bins_are_doubled_but_endpoints_are_not(self):
        time = np.arange(512)
        wave = (np.sin(2 * np.pi * 7 * time / 512) + .2 * (-1.) ** time)[None, :, None]
        value = audit.spectral_diagnostics(wave)
        transformed = np.fft.rfft((wave - wave.mean()) * np.hanning(512)[None, :, None], axis=1)
        undoubled = np.abs(transformed[0, :, 0]) ** 2 / (512 * np.square(np.hanning(512)).sum())
        self.assertAlmostEqual(value["power"][0, 0], undoubled[0], places=12)
        self.assertAlmostEqual(value["power"][-1, 0], undoubled[-1], places=10)
        np.testing.assert_allclose(value["power"][1:-1, 0], 2 * undoubled[1:-1], atol=1e-10, rtol=1e-12)

    def test_cost_ratio_uses_paired_repeats_not_ratio_of_unpaired_medians(self):
        raw = {"seconds": [1., 2., 3., 4., 5., 6.], "batch": 8}
        seconds = [2., 8., 3., 8., 5., 12.]
        ratios = [a / b for a, b in zip(seconds, raw["seconds"])]
        row = {"seconds": seconds, "batch": 8, "median_seconds": 6.5, "median_seconds_per_window": .8125,
               "paired_RAW_median_ratio": 2., "paired_RAW_seconds": raw["seconds"],
               "paired_RAW_ratios": ratios, "repeats": 6, "warmup_each_route": 2}
        value = audit.audit_recorded_times(row, raw)
        self.assertEqual(value["samples_checked"], 6)
        row["paired_RAW_median_ratio"] = 6.5 / 3.5
        with self.assertRaisesRegex(audit.AuditFailure, "cost:paired_RAW"):
            audit.audit_recorded_times(row, raw)


if __name__ == "__main__":
    unittest.main()
