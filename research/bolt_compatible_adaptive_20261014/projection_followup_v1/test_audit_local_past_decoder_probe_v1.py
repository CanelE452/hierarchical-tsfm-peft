"""Independent local retained-space, fallback, arithmetic and output-class fixtures."""
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

import audit_local_past_decoder_probe_v1 as audit

audit.initialize_numpy()


def integrated_case(*, no_past_observation=False, backend="BOLT_NUMPY"):
    z = np.tile(np.array([-1., 1.], np.float32), 256).reshape(1, 512, 1)
    x = np.concatenate((2 * z + 3, -z + 4), axis=-1).astype(np.float32)
    mask = np.ones_like(x, bool) if not no_past_observation else np.zeros_like(x, bool)
    solved = audit.independently_fit_local(x, mask, z)
    d, b = solved["decoder_fit_FP64"], solved["decoder_bias_fit_FP64"]
    d32, b32 = d.astype(np.float32), b.astype(np.float32)
    counts, ranks = solved["true_observed_counts"], solved["centered_slope_ranks"]
    metadata = {key: solved[key].tolist() for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept", "retained_condition_number", "observed_reconstruction_MSE_real_fit", "observed_latent_means_per_C", "observed_raw_C_means", "singularvalues_observed_centered_Z")}
    metadata.update(windows=1, channels=2, K=1, length=512, horizon=48, singular_relative_cut=1e-4,
        equivalent_centered_covariance_eigen_relative_cut=1e-8, future_labels_used_in_local_decoder_fit=False,
        TUNE_or_DEV_threshold_selection=False, new_E_F_or_native_head_training=False, strict_K_only_compression=False,
        same_decoder_side_information_as_static=False, past_C_condition_metadata_added=True, inference_time_decoder_refitting=True,
        static_fixed_D_floor_reused_as_dynamic_lower_bound=False, representation_learning_improvement_claimed=False,
        fit_input_bindings={"history": audit.raw_binding(x), "history_observed": audit.raw_binding(mask), "cached_latent_history": audit.raw_binding(z)},
        state_arrays={name: audit.raw_binding(value) for name, value in (("decoder_fit64", d), ("decoder32", d32), ("bias_fit64", b), ("bias32", b32))},
        rank_deficient_slope_channels=(ranks < 1).tolist(), observed_count_below_two=(counts < 2).tolist(),
        no_observation_current_C_fallback=(counts == 0).tolist(), observed_reconstruction_MSE_valid=(counts > 0).tolist(),
        retained_condition_number_valid=(ranks > 0).tolist(), decoder_Frobenius_norm_per_window=np.linalg.norm(d32.astype(np.float64), axis=(1, 2)).tolist(),
        decoder_operator_norm_per_window=[float(np.linalg.norm(d32[0].astype(np.float64)))],
        max_abs_decoder_per_window=np.abs(d32).max(axis=(1, 2)).astype(np.float64).tolist(),
        max_retained_relative_condition_number=float(solved["retained_condition_number"].max()) if ranks.any() else None)
    state = {"decoder_fit_FP64": d, "decoder_deployed_FP32": d32, "decoder_bias_fit_FP64": b,
             "decoder_bias_FP32": b32, "metadata_json": np.array(json.dumps(metadata, allow_nan=False))}
    future = np.linspace(1., 2., 48, dtype=np.float32).reshape(1, 48, 1)
    original_d = np.array([[1., 1.]], np.float64)
    replay = audit.local_forecast(x[:, -1], future, z[:, -1], original_d.astype(np.float32)[None])
    local = audit.local_forecast(x[:, -1], future, z[:, -1], d32)
    target = local + np.float32(.25)
    target_mask = np.ones_like(target, bool)
    target_mask[0, 0, 0] = False
    values = {"history": x, "history_observed": mask, "target": target, "observed": target_mask,
              "raw_last": x[:, -1].copy(), "raw_forecast": np.broadcast_to(x[:, -1, None], target.shape).copy(),
              "std": np.array([2., 3.]), "origins": np.array([512], np.int64)}
    past = np.einsum("nlk,nkc->nlc", z, d32, optimize=False) + b32[:, None]
    arrays = {"latent_history": z.copy(), "past_prediction_FP32": past, "origins": values["origins"].copy(),
              "target": target.copy(), "observed": target_mask.copy(), "raw_last": values["raw_last"].copy()}
    entry = {"status": "COMPLETE", "local_fit_metadata": metadata,
             "native_cached_past_byte_exact_and_D_shared": backend == "NATIVE_TORCH32", "methods": {}}
    entry["past_reconstruction_deployment_scores_observed_only"] = (audit.prior.scores(past, x, mask, values["std"]) if mask.any()
        else {"status": "NO_OBSERVED_PAST_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None})
    cached = {}
    methods = ("BOLT",) if backend == "BOLT_NUMPY" else ("NATIVE_s92601", "NATIVE_s92602")
    for method in methods:
        cached[method] = {"latent_history": z.copy(), "latent_future": future.copy(), "prediction": replay.copy(), "origins": values["origins"].copy()}
        arrays.update({method + "_static_prediction": replay.copy(), method + "_local_prediction": local.copy(),
                       method + "_static_einsum_replay": replay.copy(), method + "_cached_future": future.copy()})
        scored = {}
        for key, prediction, observed in (("static_scores_all_observed", replay, target_mask), ("local_scores_all_observed", local, target_mask),
                                         ("static_scores_complete_mask", replay, audit.prior.oracle_mask(target_mask)), ("local_scores_complete_mask", local, audit.prior.oracle_mask(target_mask))):
            scored[key] = audit.prior.scores(prediction, target, observed, values["std"])
        scored["static_D_einsum_arithmetic_replay"] = {"max_abs_vs_original": 0., "MSE_vs_original": 0., "byte_exact": True,
            "scores_all_observed": scored["static_scores_all_observed"], "scope": "fixture"}
        scored.update(original_selected_prediction_byte_preserved=True, native_head_frozen_intervention_not_reoptimized=backend == "NATIVE_TORCH32")
        entry["methods"][method] = scored
    return entry, arrays, state, values, {"decoder": original_d}, cached, backend


class IndependentLocalPastFixtures(unittest.TestCase):
    def test_full_rank_affine_coefficients_and_unpenalized_intercept(self):
        z = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = z @ [2., -3.] + 7.
        value = audit.independent_centered_slope(z, y)
        np.testing.assert_allclose(value["slope"], [2., -3.], atol=1e-14)
        self.assertEqual(value["bias"], 7.)
        self.assertEqual(value["rank"], 2)

    def test_discarded_weak_predictable_direction_has_nonzero_full_gradient(self):
        strong, weak = np.array([-1., -1., 1., 1.]), np.array([-1., 1., -1., 1.])
        value = audit.independent_centered_slope(np.c_[strong, 9e-5 * weak], weak)
        self.assertEqual(value["rank"], 1)
        self.assertEqual(value["reconstruction_MSE"], 1.)
        np.testing.assert_array_equal(value["retained_normal_gradient"], [0.])
        np.testing.assert_allclose(value["full_normal_gradient"], [0., -9e-5], atol=1e-15)

    def test_clear_numeric_cutoff_below_and_above_boundary(self):
        strong, weak = np.array([-1., -1., 1., 1.]), np.array([-1., 1., -1., 1.])
        low = audit.independent_centered_slope(np.c_[strong, 0.9999e-4 * weak], weak)
        high = audit.independent_centered_slope(np.c_[strong, 1.0001e-4 * weak], weak)
        self.assertEqual(low["rank"], 1)
        self.assertEqual(high["rank"], 2)
        self.assertTrue(low["Gram_boundary_resolved"])
        self.assertTrue(high["Gram_boundary_resolved"])

    def test_exact_nominal_FP32_cutoff_is_resolved_by_direct_SVD_not_Gram_guess(self):
        strong, weak = np.array([-1., -1., 1., 1.]), np.array([-1., 1., -1., 1.])
        z = np.c_[strong, np.float32(1e-4) * weak].astype(np.float32)
        value = audit.independent_centered_slope(z, weak)
        self.assertEqual(value["rank"], 1)
        self.assertFalse(value["Gram_boundary_resolved"])

    def test_relative_condition_one_does_not_limit_absolute_forecast_gain(self):
        value = audit.independent_centered_slope([[-1e-6], [1e-6]], [-1., 1.])
        self.assertEqual(value["condition"], 1.)
        self.assertAlmostEqual(value["slope"][0], 1e6)
        self.assertAlmostEqual(.1 * value["slope"][0], 1e5)

    def test_rank_deficient_minimum_norm_depends_on_nonorthogonal_coordinate_units(self):
        z = np.array([[-1., -1.], [1., 1.], [-1., -1.], [1., 1.]])
        y, transform = np.array([-1., 1., -1., 1.]), np.diag([1., 2.])
        old, changed = audit.independent_centered_slope(z, y), audit.independent_centered_slope(z @ transform, y)
        self.assertLess(old["reconstruction_MSE"], 1e-28)
        self.assertLess(changed["reconstruction_MSE"], 1e-28)
        self.assertAlmostEqual(np.array([1., 0.]) @ old["slope"], .5)
        self.assertAlmostEqual(np.array([1., 0.]) @ transform @ changed["slope"], .2)

    def test_zero_and_single_observation_fallbacks_are_different_validity_scopes(self):
        history, z = np.ones((1, 512, 2), np.float32), np.ones((1, 512, 1), np.float32)
        history[0, :, 0], history[0, :, 1] = 7., 10.
        mask = np.zeros_like(history, bool)
        mask[0, 20, 1] = True
        result = audit.independently_fit_local(history, mask, z)
        np.testing.assert_array_equal(result["decoder_fit_FP64"], np.zeros((1, 1, 2)))
        np.testing.assert_array_equal(result["decoder_bias_fit_FP64"], [[7., 10.]])
        np.testing.assert_array_equal(result["affine_model_ranks_with_analytic_intercept"], [[0, 1]])

    def test_imputed_past_is_not_treated_as_observed_response(self):
        z = np.arange(512, dtype=np.float32).reshape(1, 512, 1)
        x = (2. * z + 3.).astype(np.float32)
        mask = np.zeros_like(x, bool)
        mask[0, [0, 511], 0] = True
        changed = x.copy()
        changed[~mask] = 1e6
        first = audit.independently_fit_local(x, mask, z)
        second = audit.independently_fit_local(changed, mask, z)
        np.testing.assert_array_equal(first["decoder_fit_FP64"], second["decoder_fit_FP64"])
        np.testing.assert_array_equal(first["decoder_bias_fit_FP64"], second["decoder_bias_fit_FP64"])

    def test_fit_API_has_no_future_target_or_selection_parameters(self):
        self.assertEqual(set(inspect.signature(audit.independently_fit_local).parameters), {"history", "observed", "latent_history"})
        self.assertNotIn("lstsq", inspect.getsource(audit.independent_centered_slope))

    def test_zero_latent_innovation_preserves_numeric_current_anchor(self):
        current, last = np.array([[4., 5.]], np.float32), np.ones((1, 1), np.float32)
        prediction = audit.local_forecast(current, np.ones((1, 48, 1), np.float32), last, np.array([[[2., 3.]]], np.float32))
        np.testing.assert_array_equal(prediction, np.broadcast_to(current[:, None], (1, 48, 2)))

    def test_past_flat_C_can_jump_after_local_slope_collapses_to_zero(self):
        history = np.zeros((1, 512, 1), np.float32)
        latent = np.arange(512, dtype=np.float32).reshape(1, 512, 1)
        result = audit.independently_fit_local(history, np.ones_like(history, bool), latent)
        forecast = audit.local_forecast(history[:, -1], np.full((1, 48, 1), 600., np.float32), latent[:, -1], result["decoder_fit_FP64"].astype(np.float32))
        self.assertEqual(float(forecast.max()), 0.)
        self.assertGreater(float(np.square(forecast - 1.).mean()), 0.)

    def test_origin_specific_rank_one_spaces_can_beat_old_global_rank_one_oracle(self):
        target = np.array([[[1., 0.]] * 48, [[0., 1.]] * 48], np.float32)
        values = {"target": target, "observed": np.ones_like(target, bool), "raw_last": np.zeros((2, 2), np.float32), "std": np.ones(2)}
        d = np.array([[[1., 0.]], [[0., 1.]]], np.float32)
        result = audit.per_origin_output_minimum(values, d)
        self.assertLess(result["scores"]["normalized"]["mse"], 1e-28)
        self.assertEqual(result["rank_D_per_origin"], [1, 1])
        self.assertEqual(np.linalg.matrix_rank(target.reshape(-1, 2)), 2)
        self.assertFalse(result["old_static_or_global_rank_K_minimum_reused"])

    def test_new_dynamic_output_diagnostic_excludes_incomplete_positions_and_zero_rank(self):
        target = np.ones((2, 48, 2), np.float32)
        mask = np.ones_like(target, bool)
        mask[0, 0, 0] = False
        values = {"target": target, "observed": mask, "raw_last": np.zeros((2, 2), np.float32), "std": np.ones(2)}
        result = audit.per_origin_output_minimum(values, np.zeros((2, 1, 2), np.float32))
        self.assertEqual(result["rank_D_per_origin"], [0, 0])
        self.assertEqual(result["scores"]["normalized"]["observed_counts"], [95, 95])
        self.assertEqual(result["scores"]["normalized"]["mse"], 1.)
        self.assertFalse(result["RAW_bound_inequality_applied"])

    def test_full_state_and_backend_scores_are_JSON_serializable(self):
        entry, arrays, state, values, original, cached, backend = integrated_case()
        result = audit.check_local_state(entry, state, values, arrays["latent_history"])
        checked = audit.audit_backend(entry, arrays, state, values, original, cached, backend)
        self.assertEqual(result["channel_slots_checked"], 2)
        self.assertEqual(checked["score_field_checks"], 72)
        self.assertEqual(checked["methods"]["BOLT"]["local_scores_complete_mask"]["normalized"]["mse"], .0625)
        json.loads(json.dumps({"state": result, "backend": checked}, allow_nan=False))

    def test_no_observed_past_has_explicit_score_status_and_forecast_still_checked(self):
        args = integrated_case(no_past_observation=True)
        checked_state = audit.check_local_state(args[0], args[2], args[3], args[1]["latent_history"])
        checked = audit.audit_backend(*args)
        self.assertEqual(checked_state["no_observation_fallbacks_checked"], 2)
        self.assertIsNone(checked["past_reconstruction_deployment_scores_observed_only"])
        self.assertEqual(checked["score_field_checks"], 60)
        self.assertEqual(checked["new_per_origin_output_space_diagnostic"]["rank_D_per_origin"], [0])

    def test_original_signed_zero_byte_corruption_is_rejected(self):
        args = integrated_case()
        args[5]["BOLT"]["prediction"][0, 0, 0] = np.float32(0.)
        args[1]["BOLT_static_prediction"][0, 0, 0] = np.float32(-0.)
        with self.assertRaisesRegex(ValueError, "BYTE"):
            audit.audit_backend(*args)

    def test_future_target_mutation_does_not_change_local_fit_and_scores_reject_stale_metadata(self):
        args = integrated_case()
        before = audit.check_local_state(args[0], args[2], args[3], args[1]["latent_history"])
        args[3]["target"] = args[3]["target"] + 1.
        args[1]["target"] = args[3]["target"].copy()
        after = audit.check_local_state(args[0], args[2], args[3], args[1]["latent_history"])
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, "NUMERIC_MISMATCH"):
            audit.audit_backend(*args)

    def test_native_seed_past_mismatch_is_rejected_and_same_D_reuse_audited(self):
        args = integrated_case(backend="NATIVE_TORCH32")
        checked = audit.audit_backend(*args)
        self.assertEqual(len(checked["methods"]), 2)
        args[5]["NATIVE_s92602"]["latent_history"][0, 0, 0] += .1
        with self.assertRaisesRegex(ValueError, "BYTE"):
            audit.audit_backend(*args)

    def test_coefficient_FP32_cast_binding_and_cutoff_corruption_rejected(self):
        entry, arrays, state, values, *_ = integrated_case()
        state["decoder_deployed_FP32"][0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, "BYTE"):
            audit.check_local_state(entry, state, values, arrays["latent_history"])
        entry, arrays, state, values, *_ = integrated_case()
        entry["local_fit_metadata"]["singular_relative_cut"] = 1e-8
        state["metadata_json"] = np.array(json.dumps(entry["local_fit_metadata"]))
        with self.assertRaisesRegex(ValueError, "TRUNCATION"):
            audit.check_local_state(entry, state, values, arrays["latent_history"])

    def test_native_one_failed_method_is_partial_and_other_prediction_audited(self):
        args = integrated_case(backend="NATIVE_TORCH32")
        args[0]["status"] = "PARTIAL_COMPLETED"
        args[0]["methods"]["NATIVE_s92602"] = {"status": "FAILED_PATH_PRESERVED", "failure": {"message": "synthetic nonfinite future"}}
        for suffix in ("_static_prediction", "_local_prediction", "_static_einsum_replay", "_cached_future"):
            del args[1]["NATIVE_s92602" + suffix]
        checked = audit.audit_backend(*args)
        self.assertEqual(set(checked["methods"]), {"NATIVE_s92601"})
        self.assertEqual(checked["score_field_checks"], 72)

    def test_source_AST_fitting_contract_reads_source_only(self):
        names = ("run_local_past_decoder_probe_v1.py", "local_past_decoder_v1.py", "PLAN_LOCAL_PAST_DECODER_KO_v1.md", "decoder_metric_probe_v1.py", "fixed_decoder_span_v1.py", "run_fit_target_comparison_v1.py", "run_past_constant_gate_v1.py", "run_past_constant_gate_cost_v1.py")
        records = [{"path": str(audit.HERE / name)} for name in names]
        self.assertTrue(audit.check_source_contract(records)["past_only_fit_call_AST_checked"])


if __name__ == "__main__":
    unittest.main()
