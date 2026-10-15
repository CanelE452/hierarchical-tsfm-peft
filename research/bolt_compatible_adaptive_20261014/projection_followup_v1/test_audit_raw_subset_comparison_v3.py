"""Synthetic direct-score, SVD/global-D, cached CPU head and real-space fixtures."""
from pathlib import Path
import copy
import inspect
import ast
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_raw_subset_comparison_v3 as audit
import raw_subset_projection_v2 as fixture_producer

audit.initialize_numpy()


def toy():
    rng = np.random.default_rng(921)
    history = rng.normal(size=(2, 512, 4)).astype(np.float32)
    history[..., 3] = .3 * history[..., 0] + .7 * history[..., 1]
    observed = rng.random(history.shape) > .2
    return history, observed


def selection_fixture(history=None, observed=None, k=2):
    if history is None: history, observed = toy()
    result = fixture_producer.greedy_raw_subset(history, observed, k)
    lock = {"status": "LOCKED_FIT_PAST_ONLY", "FIT_source": {"arrays": {"history": audit.raw_binding(history), "history_observed": audit.raw_binding(observed)}}, "selection_metadata": result["metadata"], "selected_indices": result["selected_indices"], "selection_order": result["selection_order"], "selected_channel_names": ["c" + str(index) for index in result["selected_indices"]]}
    return lock, history, observed, k, ["c0", "c1", "c2", "c3"]


def global_fixture(history=None, observed=None, z=None):
    if history is None: history, observed = toy()
    if z is None: z = np.ascontiguousarray(history[..., :2])
    fitted = fixture_producer.fit_global_observed_decoder(history, observed, z)
    return fitted["metadata"], {key: fitted[key] for key in ("decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32")}, history, observed, z


def future_fixture(current=None, target=None, d=None):
    current = np.array([[1., 2., 3.]], np.float32) if current is None else current
    target = np.broadcast_to(current[:, None] + np.float32(1.), (1, 48, current.shape[-1])).copy() if target is None else target
    observed = np.ones(target.shape, bool)
    values = {"raw_last": current, "target": target, "observed": observed, "std": np.ones(target.shape[-1], np.float32), "raw_forecast": np.broadcast_to(current[:, None], target.shape).copy()}
    d = np.array([[1., 1., 0.]], np.float32) if d is None else d
    return values, d


class IndependentRawSubsetV2Tests(unittest.TestCase):
    def test_all_direct_greedy_candidates_winner_names_and_original_masks(self):
        result = audit.check_subset_trace(*selection_fixture())
        self.assertEqual(result["candidate_scores_checked"], 7)
        self.assertFalse(result["future_forecast_quality_or_global_subset_optimality_proven"])
        json.dumps(result, allow_nan=False)

    def test_tie_lowest_index_and_signed_zero_direct_slice(self):
        s = np.tile(np.array([-1., 1.], np.float32), 256)
        history = np.broadcast_to(s[None, :, None], (1, 512, 4)).copy()
        result = audit.check_subset_trace(*selection_fixture(history, np.ones(history.shape, bool), 3))
        self.assertEqual(result["selection_order"], [0, 1, 2])
        history[0, 0, 3] = np.float32(-0.)
        e = {"selected_indices": np.array([1, 3], np.int64), "center": np.zeros(4, np.float32), "encoder": np.eye(4, dtype=np.float32)[:, [1, 3]]}
        expected = np.ascontiguousarray(history[..., [1, 3]])
        audit.exact(audit.encode(history, e, "BOLT_NUMPY"), expected, "signed_zero_Bolt_subset")
        audit.exact(audit.encode(history, e, "NATIVE_TORCH32"), expected, "signed_zero_native_subset")

    def test_bad_score_count_rank_winner_or_FIT_binding_rejected(self):
        for what in ("score", "count", "rank", "winner", "FIT"):
            args = list(selection_fixture()); lock = args[0]
            row = lock["selection_metadata"]["stages"][0]["candidates"][0]
            if what == "score": row["macro_normalized_MSE"] += .1
            elif what == "count": row["true_observed_counts"][0] -= 1
            elif what == "rank": row["retained_centered_design_rank"][0] += 1
            elif what == "winner": lock["selection_metadata"]["stages"][0]["selected_original_index"] = 999
            else: lock["FIT_source"]["arrays"]["history"]["sha256"] = "wrong"
            with self.subTest(what=what), self.assertRaises(ValueError): audit.check_subset_trace(*args)

    def test_direct_score_no_Gram_clipping_and_inactive_one_observation_scope(self):
        x, mask = toy(); mask[..., 2] = False; mask[..., 3] = False; mask[0, 0, 3] = True
        score = audit.direct_subset_scores(x, mask, [0])
        self.assertIsNone(score["per_channel_MSE"][2]); self.assertEqual(score["per_channel_MSE"][3], 0.)
        self.assertEqual(score["active_channels"], 3)
        direct = fixture_producer.score_subset(fixture_producer.precompute_observed_stats(x, mask), [0])
        self.assertFalse(direct["Gram_cross_variance_expansion_used_for_scores"]); self.assertEqual(direct["negative_roundoff_MSE_values_clipped_to_zero"], 0)

    def test_global_D_manual_SVD_direct_LS_cast_intercept_and_nullspace(self):
        result = audit.check_global_state(*global_fixture())
        self.assertEqual(result["observed_response_channels_checked"], 4)
        self.assertTrue(result["independent_SVD_and_direct_LS_checked"])

    def test_constant_response_fullrank_is_identified_zero_slope_not_fallback(self):
        x, mask = toy(); x[..., 3] = np.float32(7.)
        args = global_fixture(x, mask)
        result = audit.check_global_state(*args)
        self.assertTrue(args[0]["learned_slope"][3]); self.assertTrue(args[0]["slope_identified_in_retained_full_K_space"][3])
        self.assertIsNone(args[0]["fallback_reason"][3])
        self.assertTrue(result["constant_response_identified_differentiated_from_unobserved_or_unidentified"])

    def test_noobs_oneobs_zero_design_and_rank_deficiency_differ(self):
        x, mask = toy(); mask[..., 2] = False; mask[..., 3] = False; mask[0, 0, 3] = True
        args = global_fixture(x, mask, np.ones((2, 512, 2), np.float32))
        audit.check_global_state(*args)
        self.assertTrue(args[0]["no_observed_responses"][2]); self.assertIsNone(args[0]["rank_deficient_slope"][2])
        self.assertTrue(args[0]["one_observed_response"][3]); self.assertTrue(args[0]["rank_deficient_slope"][3])
        self.assertTrue(args[0]["zero_centered_predictor_variance"][0])

    def test_rankdeficient_slopes_minimum_norm_and_not_full_K_identified(self):
        x, mask = toy(); z = np.repeat(x[..., :1], 2, axis=2)
        args = global_fixture(x, mask, z); audit.check_global_state(*args)
        self.assertEqual(args[0]["retained_centered_design_rank"], [1] * 4)
        self.assertFalse(any(args[0]["slope_identified_in_retained_full_K_space"]))
        np.testing.assert_allclose(args[1]["decoder_FP64"][0], args[1]["decoder_FP64"][1], atol=1e-10)

    def test_registered_singular_cutoff_both_sides_without_relaxation(self):
        s = np.tile(np.array([-1., 1.], np.float32), 256); t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
        for epsilon, rank in ((5e-5, 1), (2e-4, 2)):
            x = np.stack((s, np.float32(epsilon) * t, s + np.float32(epsilon) * t), axis=-1)[None]
            args = global_fixture(x, np.ones(x.shape, bool), np.ascontiguousarray(x[..., :2]))
            audit.check_global_state(*args); self.assertEqual(args[0]["retained_centered_design_rank"], [rank] * 3)

    def test_global_D_state_or_observation_hash_tampering_rejected(self):
        for field in ("coefficient", "cast", "maskhash", "flag"):
            args = list(global_fixture())
            if field == "coefficient": args[1]["decoder_FP64"][0, 0] += .1
            elif field == "cast": args[1]["decoder_FP32"][0, 0] += .1
            elif field == "maskhash": args[0]["fit_input_bindings"]["observed"]["sha256"] = "wrong"
            else: args[0]["constant_observed_response"][0] = not args[0]["constant_observed_response"][0]
            with self.subTest(field=field), self.assertRaises(ValueError): audit.check_global_state(*args)

    def test_independent_native_CP_math_matches_static_D_Torch32_components(self):
        import torch
        from raw_subset_native_head_v1 import NativeFixedDecoder, predict_components
        x, _ = toy(); z = np.ascontiguousarray(x[..., :2]); d = np.array([[1., 0., .5, .2], [0., 1., .5, .8]], np.float32)
        model = NativeFixedDecoder(d)
        with torch.no_grad(): model.head.bias.fill_(.1); model.head.weight.fill_(.0001)
        state = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
        actual = predict_components(model, z, x[:, -1])
        expected = audit.independent_native({"latent_history": z, "raw_last": x[:, -1]}, state)
        for a, b in zip(actual, expected): audit.exact(a, b, "independent_native_math")

    def test_CP0_persistence_future_labels_not_forward_input(self):
        x, _ = toy(); z = np.ascontiguousarray(x[..., :2]); d = np.zeros((2, 4), np.float32)
        state = {"decoder": d, "head.weight": np.zeros((2, 48, 512), np.float32), "head.bias": np.zeros((2, 48), np.float32)}
        payload = {"latent_history": z, "raw_last": x[:, -1], "target": np.zeros((2, 48, 4), np.float32)}
        a, _ = audit.independent_native(payload, state); payload["target"][:] = 1e6; b, _ = audit.independent_native(payload, state)
        audit.exact(a, b, "no_future_head_input")
        self.assertTrue(np.array_equal(a, np.broadcast_to(x[:, -1:], a.shape)))

    def test_new_fixed_D_space_minimum_same_complete_mask_and_denominator(self):
        values, d = future_fixture(); values["observed"][0, 0, 0] = False
        output = audit.new_fixed_space(values, d)
        self.assertEqual(output["complete_target_positions"], 47); self.assertEqual(output["numeric_rank"], 1)
        self.assertAlmostEqual(output["minimum_normalized_MSE_real_D32_numeric_rowspace"], 1. / 3)
        self.assertFalse(output["RAW_bound_applied"])

    def test_EY_oblique_coordinates_need_not_attain_minimum(self):
        values, d = future_fixture(current=np.zeros((1, 3), np.float32), target=np.broadcast_to(np.array([1., 2., 0.], np.float32), (1, 48, 3)).copy())
        floor = audit.new_fixed_space(values, d)
        ey_future = np.ones((1, 48, 1), np.float32); last = np.zeros((1, 1), np.float32)
        prediction = audit.fixed_forecast(values["raw_last"], ey_future, last, d, "BOLT_NUMPY")
        trace = audit.diagnostic_trace(prediction, values["raw_last"], ey_future, last, d, values, floor, "E_Y")
        self.assertGreater(trace["scores_complete_C"]["normalized"]["mse"], floor["minimum_normalized_MSE_real_D32_numeric_rowspace"])

    def test_FP32_rounding_may_beat_real_span_minimum_without_universal_bound_claim(self):
        current = np.array([[2**24, 0]], np.float32); target = np.broadcast_to(np.array([2**24, 1], np.float32), (1, 48, 2)).copy()
        values, d = future_fixture(current, target, np.array([[1., 1.]], np.float32))
        floor = audit.new_fixed_space(values, d); future = np.ones((1, 48, 1), np.float32); last = np.zeros((1, 1), np.float32)
        predicted = audit.fixed_forecast(current, future, last, d, "BOLT_NUMPY")
        trace = audit.diagnostic_trace(predicted, current, future, last, d, values, floor, "rounding_counterexample")
        self.assertEqual(trace["scores_complete_C"]["normalized"]["mse"], 0.)
        self.assertGreater(floor["minimum_normalized_MSE_real_D32_numeric_rowspace"], 0.)
        self.assertGreater(trace["measured_FP32_rounding_normalized_MSE_allowance"], 0.)

    def test_absent_complete_positions_have_explicit_non_measurement(self):
        values, d = future_fixture(); values["observed"][..., 0] = False
        self.assertEqual(audit.new_fixed_space(values, d)["status"], "NO_COMPLETE_C_TARGET_POSITIONS")

    def test_actual_source_contract_without_training_or_F_invocation(self):
        names = ("raw_subset_projection_v1.py", "raw_subset_projection_v2.py", "raw_subset_native_head_v1.py", "run_raw_subset_comparison_v2.py", "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md")
        result = audit.source_contract([{"path": str(audit.HERE / name)} for name in names])
        self.assertTrue(result["actual_new_subset_F_call_source_checked"]); self.assertFalse(result["F_calls_independently_replayed"])

    def test_registry_before_any_DEV_gate(self):
        locks, rows = {}, []
        for data in audit.DATASETS:
            for space in audit.SPACES:
                for seed in audit.SEEDS:
                    key = (data, space, str(seed)); record = {"path": "synthetic.json", "sha256": repr(key)}
                    locks[key] = (record, {"status": "LOCKED"}); rows.append({"dataset": data, "space": space, "seed": str(seed), "lock": record})
        registry = {"status": "ALL_TWELVE_RESOLVED_BEFORE_DEV", "conditions": rows}
        events = [{"sequence": 0, "kind": "ALL_TWELVE_LOCKS_AND_REGISTRY_SAVED"}, {"sequence": 1, "kind": "DEV_INPUT_LOAD_AFTER_ALL_TWELVE_LOCKS"}]
        self.assertTrue(audit.check_registry(registry, events, locks)["registry_before_any_new_DEV_checked"])
        events.reverse()
        for i, event in enumerate(events): event["sequence"] = i
        with self.assertRaises(ValueError): audit.check_registry(registry, events, locks)

    def test_pending_producer_never_reads_missing_scientific_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "PENDING.json"; path.write_text('{"status":"RUNNING"}', encoding="utf-8")
            _, _, source, passed = audit.load_gates(path, lambda: None)
        self.assertEqual((source, passed), ({}, {}))


class DiagnosticFinalizationLifecycleTests(unittest.TestCase):
    def lifecycle(self, module, *, complete=True, wrong_count=False):
        source = ast.parse(inspect.getsource(module.audit_result)).body[0]
        diagnostic = next(node for node in ast.walk(source) if isinstance(node, ast.For) and node.body and isinstance(node.body[0], ast.Assign) and isinstance(node.body[0].targets[0], ast.Name) and node.body[0].targets[0].id == "split_diagnostic")
        expected = next(node for node in source.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "expected_counts" for target in node.targets))
        final = source.body[source.body.index(expected):]
        complete_assignment = next(node for node in source.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Compare) and ast.unparse(node.value) == "result['status'] == 'COMPLETE'")
        namespace = dict(module.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[expected], type_ignores=[])), "synthetic_expected_counts", "exec"), namespace)
        totals = dict(namespace["expected_counts"])
        for key in ("new_fixed_D_output_space_cells_computed", "new_E_Y_diagnostic_predictions_computed", "new_fixed_D_real_method_inequalities_checked"): totals[key] = 0
        if wrong_count: totals["checkpoint_states_checked"] -= 1
        values, d = future_fixture(current=np.zeros((1, 2), np.float32), target=np.broadcast_to(np.array([1., 2.], np.float32), (1, 48, 2)).copy(), d=np.array([[1., .5]], np.float32))
        values["observed"][0, 0, 0] = False
        values["latent_history_BOLT_NUMPY"] = values["latent_history_NATIVE_TORCH32"] = np.zeros((1, 512, 1), np.float32)
        contexts = {split: copy.deepcopy(values) for split in ("fit", "tune", "dev")}
        future = np.ones((1, 48, 1), np.float32)
        saved = {"latent_future": future, "prediction": module.fixed_forecast(values["raw_last"], future, np.zeros((1, 1), np.float32), d, "BOLT_NUMPY")}
        namespace.update(result={"status": "COMPLETE" if complete else "PARTIAL_COMPLETED", "prior_partial": {}, "actual_subset_F_model_batch_calls": 103, "completeness": {"actual": {}, "expected": {}}, "failed_paths": []}, contexts=contexts, totals=totals, report={"status": "PASS" if complete else "PARTIAL_AUDITED_NO_FULL_PASS", "audit_failed_paths": []}, decoders={backend: d for backend in ("BOLT_NUMPY", "NATIVE_TORCH32")}, condition={"backends": {backend: {"state_artifact": {}} for backend in ("BOLT_NUMPY", "NATIVE_TORCH32")}}, output={"new_output_space_diagnostics": {}}, e={"center": np.zeros(2, np.float32), "encoder": np.array([[1.], [0.]], np.float32), "selected_indices": np.array([0], np.int64)}, bolt_arrays={split: saved for split in contexts}, native_arrays={seed: {split: saved for split in contexts} for seed in ("92601", "92602")}, check=lambda: None)
        outer = ast.For(target=ast.Name(id="condition_number", ctx=ast.Store()), iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(value=6)], keywords=[]), body=[diagnostic], orelse=[])
        fn = ast.FunctionDef(name="run_fixture", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[complete_assignment, outer, *final], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "source_exact_full_diagnostic_to_finalize", "exec"), namespace)
        with patch.object(module.prior, "verified_json", return_value={}), patch.dict(namespace, {"mixed_identity": lambda old, current, check: {"status": "SYNTHETIC_BYTE_IDENTITY_CHECK_REACHED"}}):
            result = namespace["run_fixture"]()
        return result, totals

    def test_actual_v2_shadow_failure_is_reproduced_after_all_diagnostics(self):
        import audit_raw_subset_comparison_v2 as old
        old.initialize_numpy()
        with self.assertRaisesRegex(ValueError, "truth value of an array"):
            self.lifecycle(old)

    def test_v3_all_diagnostics_then_count_and_mixed_identity_finalize(self):
        result, totals = self.lifecycle(audit)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["mixed_reexecution_identity"]["status"], "SYNTHETIC_BYTE_IDENTITY_CHECK_REACHED")
        self.assertEqual(totals["new_fixed_D_output_space_cells_computed"], 36)
        self.assertEqual(totals["new_E_Y_diagnostic_predictions_computed"], 36)
        self.assertEqual(totals["new_fixed_D_real_method_inequalities_checked"], 90)
        json.dumps(result, allow_nan=False)

    def test_incomplete_producer_is_never_promoted_by_complete_target_mask(self):
        result, _ = self.lifecycle(audit, complete=False)
        self.assertEqual(result["status"], "PARTIAL_AUDITED_NO_FULL_PASS")
        self.assertNotIn("mixed_reexecution_identity", result)

    def test_final_registered_count_guard_remains_active(self):
        with self.assertRaisesRegex(ValueError, "FULL_NEW_SUBSET"):
            self.lifecycle(audit, wrong_count=True)

    def test_executed_v2_to_v3_only_two_local_variable_names_changed(self):
        old = (audit.HERE / "audit_raw_subset_comparison_v2.py").read_text(encoding="utf-8")
        new = Path(audit.__file__).read_text(encoding="utf-8")
        class Normalize(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in ("producer_complete", "complete_target_positions"): node.id = "complete"
                return node
        self.assertEqual(ast.dump(ast.parse(old), include_attributes=False), ast.dump(Normalize().visit(ast.parse(new)), include_attributes=False))


if __name__ == "__main__": unittest.main()



