"""Independent LS/causal index/horizon coupling fixtures, without actual assets."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_causal_lag_objective_v1 as audit

audit.initialize_numpy()


def sample(n=2, k=2, c=3, h=48):
    rng = np.random.default_rng(118)
    history = rng.normal(size=(n, 512, c)).astype(np.float32)
    latent = np.ascontiguousarray(history[..., :k])
    future = rng.normal(size=(n, h, k)).astype(np.float32)
    target = rng.normal(size=(n, h, c)).astype(np.float32)
    return history, np.ones(history.shape, bool), latent, future, history[:, -1].copy(), target, np.ones(target.shape, bool)


class IndependentCausalLagMathTests(unittest.TestCase):
    def test_exact_delay_copy_and_known_prefix_h1_lag1_index(self):
        z = np.arange(512, dtype=np.float32)[None, :, None]
        f = np.array([[[600.], [700.], [800.]]], np.float32)
        actual = audit.delta_features(z, f, [0, 1, 2])
        expected = np.array([[[89., 1., 1.], [189., 90., 2.], [289., 190., 91.]]], np.float32)
        audit.exact(actual, expected, "causal_known_prefix_and_one_based_future")

    def test_flat_future_is_not_currentC_persistence_for_lagged_decoder(self):
        z = np.arange(512, dtype=np.float32)[None, :, None]
        f = np.full((1, 3, 1), z[0, -1, 0], np.float32)
        d = np.array([[0.], [1.]], np.float32); current = np.array([[5.]], np.float32)
        self.assertTrue(np.array_equal(audit.anchored_forecast(z, f, current, d, [0, 1]), np.full((1, 3, 1), 6., np.float32)))

    def test_joint_absolute_A_offset_equals_independent_direct_lag_sum(self):
        history, _, z, f, current, _, _ = sample(h=5)
        d = np.arange(18, dtype=np.float32).reshape(6, 3) / np.float32(30.)
        a, offset = audit.joint_absolute_map(z, current, d, [0, 1, 4], 5)
        expected = audit.anchored_forecast(z, f, current, d, [0, 1, 4], real=True)
        np.testing.assert_allclose(offset + (f.astype(np.float64).reshape(2, -1) @ a.T).reshape(2, 5, 3), expected, atol=1e-12)
        self.assertEqual(a.shape, (15, 10))

    def test_RECON_controls_share_exact_480_response_times(self):
        args = sample()
        short = audit.independent_fit(*args, [0], "RECON"); long = audit.independent_fit(*args, [0, 1, 2, 4, 8, 16, 32], "RECON")
        self.assertEqual(short["responses_FP64"].shape, (960, 3))
        self.assertTrue(np.array_equal(short["responses_FP64"], long["responses_FP64"]))
        self.assertTrue(np.array_equal(short["response_mask"], long["response_mask"]))
        self.assertEqual(long["features_FP64"].shape, (960, 14))

    def test_RECON_ignores_FIT_future_targets_but_FORECAST_uses_supervised_FIT(self):
        args = list(sample()); before = audit.independent_fit(*args, [0], "RECON")
        args[5] += np.float32(10.); args[6][:] = False
        after = audit.independent_fit(*args, [0], "RECON")
        audit.exact(before["decoder_FP64"], after["decoder_FP64"], "RECON_no_FIT_future_labels")
        forecast = audit.independent_fit(*args, [0], "FORECAST")
        self.assertEqual(forecast["true_observed_counts"], [0, 0, 0]); self.assertFalse(any(row["fit_executed"] for row in forecast["certificates"]))

    def test_masked_response_not_forward_fill_placeholder_is_fit_target(self):
        args = list(sample()); args[1][..., 2] = False; args[6][..., 2] = False
        for objective in ("RECON", "FORECAST"):
            out = audit.independent_fit(*args, [0], objective)
            self.assertEqual(out["true_observed_counts"][2], 0); self.assertTrue(np.all(out["decoder_FP64"][:, 2] == 0))

    def test_one_observation_noncentered_forecast_is_not_centered_RECON_fallback(self):
        args = list(sample(n=1, k=1, c=1)); args[1][:] = False; args[1][0, 100, 0] = True
        args[6][:] = False; args[6][0, 0, 0] = True; args[3][0, 0, 0] = args[2][0, -1, 0] + 1.
        recon = audit.independent_fit(*args, [0], "RECON"); forecast = audit.independent_fit(*args, [0], "FORECAST")
        self.assertEqual(recon["retained_rank"], [0]); self.assertEqual(forecast["retained_rank"], [1])
        self.assertTrue(forecast["certificates"][0]["one_observation_has_identifiable_noncentered_direction"])

    def test_FORECAST_target_subtracts_FP32_endpoints_after_cast_to_FP64(self):
        args = list(sample(n=1, k=1, c=1)); args[4][:] = np.float32(1.); args[5][:] = np.float32(1e8)
        out = audit.independent_fit(*args, [0], "FORECAST")
        self.assertEqual(out["responses_FP64"][0, 0], 99999999.)
        self.assertNotEqual(out["responses_FP64"][0, 0], float((args[5] - args[4][:, None])[0, 0, 0]))

    def test_missing_output_horizon_keeps_its_latent_unknown_for_later_lag(self):
        z = np.zeros((1, 512, 1), np.float32); current = np.zeros((1, 2), np.float32)
        d = np.array([[1., 0.], [0., 1.]], np.float32)
        target = np.array([[[0., 0.], [0., 1.], [0., 0.]]], np.float32); mask = np.zeros(target.shape, bool); mask[:, 1] = True
        target[~mask] = np.nan
        values = {"raw_last": current, "target": target, "observed": mask, "std": np.ones(2, np.float32)}
        out = audit.joint_minimum(values, z, d, [0, 1])
        self.assertAlmostEqual(out["minimum_normalized_MSE_real_joint_D32"], 0.)
        self.assertEqual(out["certificates"][0]["unknown_future_coordinates_kept"], 3)
        self.assertAlmostEqual(out["best_absolute_future_latent"][0, 0, 0], 1.)

    def test_weight_by_complete_position_count_instead_of_mean_of_window_MSE(self):
        z = np.zeros((2, 512, 1), np.float32); current = np.zeros((2, 2), np.float32); d = np.array([[1., 0.]], np.float32)
        target = np.zeros((2, 3, 2), np.float32); target[0, :, 1] = 1.; target[1, :, 1] = 3.
        mask = np.ones(target.shape, bool); mask[0, 1:] = False
        values = {"raw_last": current, "target": target, "observed": mask, "std": np.ones(2, np.float32)}
        out = audit.joint_minimum(values, z, d, [0])
        self.assertAlmostEqual(out["minimum_normalized_MSE_real_joint_D32"], 28. / 8)
        self.assertNotEqual(out["minimum_normalized_MSE_real_joint_D32"], (1. / 2 + 9. / 2) / 2)

    def test_joint_floor_trace_and_flat_semantics_with_measured_FP32_allowance(self):
        _, _, z, future, current, target, mask = sample(n=1, k=1, c=2, h=3)
        values = {"raw_last": current, "target": target, "observed": mask, "std": np.ones(2, np.float32)}
        d = np.array([[1., .5], [.1, .2]], np.float32)
        minimum = audit.joint_minimum(values, z, d, [0, 1])
        actual = audit.anchored_forecast(z, future, current, d, [0, 1])
        checked = audit.diagnostic_trace(actual, future, values, z, d, [0, 1], minimum)
        self.assertGreaterEqual(checked["real_trace_minus_minimum"], -1e-10)
        self.assertFalse(minimum["RAW_bound_applied"])

    def test_no_complete_positions_explicit_nonmeasurement(self):
        _, _, z, _, current, target, mask = sample(n=1, k=1, c=2, h=3); mask[:] = False
        values = {"raw_last": current, "target": target, "observed": mask, "std": np.ones(2, np.float32)}
        self.assertEqual(audit.joint_minimum(values, z, np.ones((1, 2), np.float32), [0])["status"], "NO_COMPLETE_C_TARGET_POSITIONS")

    def test_discarded_tiny_direction_can_beat_retained_floor_without_tolerance_change(self):
        z = np.zeros((1, 512, 2), np.float32); current = np.zeros((1, 2), np.float32)
        future = np.array([[[0., 1e16]]], np.float32); d = np.diag(np.array([1., 1e-16], np.float32))
        target = np.array([[[0., 1.]]], np.float32)
        values = {"raw_last": current, "target": target, "observed": np.ones(target.shape, bool), "std": np.ones(2, np.float32)}
        minimum = audit.joint_minimum(values, z, d, [0]); prediction = audit.anchored_forecast(z, future, current, d, [0])
        checked = audit.diagnostic_trace(prediction, future, values, z, d, [0], minimum)
        self.assertAlmostEqual(minimum["minimum_normalized_MSE_real_joint_D32"], .5)
        self.assertFalse(checked["real_trace_bound_inequality_pass"])
        self.assertFalse(checked["FP32_bound_with_measured_allowance_pass"])
        self.assertTrue(checked["retained_output_trace_bound_inequality_pass"])
        self.assertGreater(checked["discarded_numeric_map_output_max_abs"], .9)


class CompleteTinyProducerToAuditorTests(unittest.TestCase):
    def test_saved_toy_JSON_NPZ_entire_audit_then_json_serialization_and_final_counts(self):
        from threadpoolctl import threadpool_limits
        from test_run_causal_lag_objective_v1 import build_full_fixture
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=2):
            generated = build_full_fixture(Path(temporary) / "generated_only")
            result_path = generated["result"]; result = json.loads(result_path.read_text(encoding="utf-8"))
            source = json.loads(Path(result["raw_subset_diagnostic"]["path"]).read_text(encoding="utf-8"))
            result.update(frozen_F_parent_signatures={"before": source["F_before"], "after": source["F_after"]}, new_F_object_or_state_hash_computed=False, new_representation_learning_claimed=False, strict_K_only_compression=False, individual_exact_human_method_approval_claimed=False, source_cache_model_selection_SHA_unchanged=True)
            result["source_records"] = []
            snapshots = result_path.parent / "audit_fixture_sources"; snapshots.mkdir()
            result["source_snapshots"] = []
            for name in ("causal_lag_decoder_v1.py", "run_causal_lag_objective_v1.py", "PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md"):
                path = audit.HERE / name; record = {"path": str(path), "sha256": audit.prior.file_sha(path)}; result["source_records"].append(record)
                snapshot = snapshots / name; shutil.copyfile(path, snapshot)
                result["source_snapshots"].append({"path": str(snapshot), "sha256": audit.prior.file_sha(snapshot)})
                if name.endswith(".md"): result["plan"] = record
            events, seen = [], set()
            opened = result["fixture_open_order"]
            for index, (dataset, space, split) in enumerate(opened):
                if index == 6: events.append({"event": "ALL_24_FIT_LOCKS_SEALED", "registry": result["fit_registry"]})
                if (dataset, space, split) in seen: continue
                seen.add((dataset, space, split)); events.append({"event": "CONTEXT_OPENED", "dataset": dataset, "space": space, "split": split, "all_24_locks_already_sealed": index >= 6})
            result["execution_events"] = events
            receipt_path = result_path.parent / "SYNTHETIC_CPU_RECEIPT.json"; receipt_path.write_text(json.dumps({"status": result["status"], "elapsed_or_cumulative_budget_caps": None, "scope": "synthetic fixture, no scientific execution"}), encoding="utf-8")
            result["resource_terminal"] = {"path": str(receipt_path), "sha256": audit.prior.file_sha(receipt_path)}
            audit_input = result_path.parent / "SYNTHETIC_FULL_AUDIT_INPUT.json"; audit_input.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
            parents = {key: result[key]["sha256"] for key in audit.PARENT_SHA}
            meter = SimpleNamespace(check=lambda: None)
            with patch.object(audit, "DATASETS", generated["datasets"]), patch.object(audit, "PARENT_SHA", parents):
                report = audit.audit_result(audit_input, meter, expected_windows={"fit": 2, "tune": 1, "val": 1, "eval": 1})
            self.assertEqual(report["status"], "PASS", report["audit_failed_paths"])
            self.assertEqual(report["totals"]["decoder_states_checked"], 24)
            self.assertEqual(report["totals"]["cached_D_split_prediction_arrays_checked"], 72)
            self.assertEqual(report["totals"]["score_field_checks"], 14688)
            self.assertEqual(report["totals"]["joint_output_diagnostic_cells_checked"], 72)
            self.assertEqual(report["totals"]["retained_trace_inequalities_checked"], 216)
            json.dumps(report, allow_nan=False)

    def test_markdown_parent_is_SHA_checked_not_JSON_parsed_and_wrong_SHA_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "PLAN.md"; path.write_text("# Actual Markdown syntax\n", encoding="utf-8")
            record = {"path": str(path), "sha256": audit.prior.file_sha(path), "bytes": path.stat().st_size}
            self.assertIsNone(audit.verified_metadata(record, lambda: None))
            with self.assertRaises(ValueError): audit.verified_metadata({**record, "sha256": "wrong"}, lambda: None)


if __name__ == "__main__": unittest.main()
